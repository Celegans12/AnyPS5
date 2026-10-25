#include "prx/libSceAgcDriver/Execution/include/CommandLog.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/CommitPrep.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ParallelProfile.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace Prep = AgcDriver::DriverDetail::CommitPrep;
using Plan = Prep::DrainPredictor::Plan;
using AgcDriver::DriverDetail::DrawPipeline;

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("commit lock split: " + what);
}

constexpr std::size_t BlockBytes = 65536;

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        check(block != nullptr, "cannot allocate the guest block");
        std::memset(block, 0, BlockBytes);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }
    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }
    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;
    std::uint64_t Address() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block)); }
    std::uint32_t Word(std::size_t offset) const {
        std::uint32_t value = 0;
        std::memcpy(&value, block + offset, sizeof(value));
        return value;
    }

private:
    std::uint8_t* block = nullptr;
};

void predictor() {
    Prep::DrainPredictor predict;
    constexpr std::uint64_t label = 0x1234567u;
    constexpr std::uint64_t other = 0x7654321u;
    check(predict.Predict(label) == Plan::None, "an unseen item was predicted to drain");
    predict.Learn(label, Plan::None, 0, 20);
    check(predict.Predict(label) == Plan::None, "an item that did not wait under the lock was predicted to drain");
    predict.Learn(label, Plan::None, 0, 5000);
    check(predict.Predict(label) == Plan::Submitted, "an item that waited under the lock is not pre-drained next time");
    check(predict.Predict(other) == Plan::None, "a different item shares the prediction");
    predict.Learn(label, Plan::Submitted, 4000, 0);
    check(predict.Predict(label) == Plan::Submitted, "a pre-drain that waited for the GPU was not kept");
    predict.Learn(label, Plan::Submitted, 0, 3000);
    check(predict.Predict(label) == Plan::Open, "an item still waiting after the submitted work drained does not drain its open batch next time");
    for (int round = 0; round < 8; ++round) predict.Learn(label, Plan::Open, 0, 0);
    check(predict.Predict(label) == Plan::None, "pre-drains that found nothing to wait for kept being made");
    const std::vector<Prep::Range> writes{{0x1000, 0x2000}};
    const std::vector<Prep::Range> reads{{0x3000, 0x3100}};
    check(Prep::ItemKey(false, 0x40, writes, reads, 0) == Prep::ItemKey(false, 0x40, writes, reads, 0), "an item key is not stable");
    check(Prep::ItemKey(false, 0x40, writes, reads, 0) != Prep::ItemKey(false, 0x48, writes, reads, 0), "two labels share an item key");
    check(Prep::ItemKey(true, 0, writes, reads, 77) != Prep::ItemKey(true, 0, writes, reads, 78), "two draw sites share an item key");
    check(Prep::ItemKey(true, 0, {}, {}, 0) != 0, "an item key is zero");
}

void slotClaim() {
    DrawPipeline::PrepareSlot claimed;
    check(claimed.Begin(), "the committer could not claim an unprepared slot");
    claimed.Finish(std::make_shared<int>(7));
    check(!claimed.Begin(), "the preparer prepared a slot the committer claimed");
    const auto result = std::static_pointer_cast<int>(claimed.Take());
    check(result != nullptr && *result == 7, "the committer's own preparation was not taken");
    DrawPipeline::PrepareSlot taken;
    check(taken.Take() == nullptr, "an unprepared slot returned a preparation");
    check(!taken.Begin(), "a slot already taken could still be prepared");
}

void preparedStamps(AgcDriver::VulkanDevice& device) {
    AgcDriver::Graphics::State state{};
    AgcDriver::Pm4::DrawParameters draw{};
    draw.indexCount = 3;
    draw.instanceCount = 1;
    const auto prepared = device.PrepareDraw(state, draw, {});
    if (prepared == nullptr) {
        std::puts("split draw builds are off: prepared stamps not checked");
        return;
    }
    const auto own = AgcDriver::Graphics::PreparedDrawDevice(*prepared);
    using AgcDriver::Graphics::PreparedDrawStamp;
    check(AgcDriver::Graphics::CheckPreparedDraw(*prepared, own, 0) == PreparedDrawStamp::Current, "an untouched preparation is stale");
    check(AgcDriver::Graphics::CheckPreparedDraw(*prepared, VK_NULL_HANDLE, 0) == PreparedDrawStamp::Device, "a preparation made on another device was accepted");
    check(AgcDriver::Graphics::CheckPreparedDraw(*prepared, own, 64) == PreparedDrawStamp::IndexBytes, "a preparation for other index bytes was accepted");
    {
        GuestBlock remapped;
        check(AgcDriver::Graphics::CheckPreparedDraw(*prepared, own, 0) == PreparedDrawStamp::Registry, "a guest mapping made between preparation and record did not make the preparation stale");
    }
}

void preDrainCoversLaterWork(AgcDriver::VulkanDevice& device) {
    GuestBlock block;
    const std::array<std::uint32_t, 4> first{0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
    const std::array<std::uint32_t, 4> second{0x55555555u, 0x66666666u, 0x77777777u, 0x88888888u};
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        if (!device.FillBuffer(block.Address(), 4096, first)) {
            std::puts("the device does not import guest memory: pre-drain not checked");
            return;
        }
    }
    if (!device.CanWaitUnlocked()) {
        std::puts("the recorder has no timeline: pre-drain not checked");
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        device.WaitIdle();
        return;
    }
    const auto drained = device.PreDrainUnlocked(true);
    check(drained.submitted, "the pre-drain did not submit the open batch");
    check(!AgcDriver::GuestMemory::GpuMutex().HeldByThisThread(), "the pre-drain returned holding the GPU mutex");
    check(block.Word(0) == first[0] && block.Word(12) == first[3], "the pre-drain returned before the recorded fill landed");
    const std::array<Prep::Range, 1> reads{{{block.Address() + 8192, block.Address() + 8192 + 4096}}};
    const auto before = Prep::Take(&device, reads);
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        check(device.FillBuffer(block.Address() + 8192, 4096, second), "the second fill was not recorded");
    }
    const auto moved = Prep::Moved(before, Prep::Retake(before, &device));
    check((moved & AgcDriver::Graphics::ParallelProfile::StampBit(AgcDriver::Graphics::ParallelProfile::Stamp::GpuWrite)) != 0, "a GPU writer recorded over the read range between preparation and record moved no stamp");
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        device.WaitIdle();
    }
    check(block.Word(8192) == second[0] && block.Word(8192 + 12) == second[3], "the inline drain after a pre-drain missed work recorded between them");
    const auto idle = device.PreDrainUnlocked(false);
    check(!idle.drained, "a pre-drain with nothing in flight waited");
    const auto readback = device.PreWaitStorageWriters({}, true);
    check(!readback.drained && !readback.submitted && !AgcDriver::GuestMemory::GpuMutex().HeldByThisThread(), "a label readback pre-wait without pending storage results waited or submitted");
}

void commandLog(AgcDriver::VulkanDevice& device) {
    if (!AgcDriver::CommandLog::Enabled()) {
        std::puts("APS5_COMMAND_LOG is unset: the command log is not checked");
        return;
    }
    GuestBlock block;
    const auto run = [&](std::uint32_t value) {
        AgcDriver::CommandLog::Reset();
        const std::array<std::uint32_t, 4> pattern{value, value + 1, value + 2, value + 3};
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        if (!device.FillBuffer(block.Address(), 4096, pattern)) return std::vector<AgcDriver::CommandLog::Stream>{};
        device.WaitIdle();
        return AgcDriver::CommandLog::Streams();
    };
    if (run(0x100).empty()) {
        std::puts("the device does not import guest memory: the command log is not checked");
        return;
    }
    const auto a = run(0x100);
    const auto b = run(0x100);
    const auto c = run(0x200);
    check(a.size() == b.size(), "identical work submitted to a different number of streams");
    std::uint64_t commands = 0;
    for (std::size_t index = 0; index < a.size(); ++index) {
        check(a[index].pool == b[index].pool && a[index].commands == b[index].commands && a[index].hash == b[index].hash, "identical work logged a different command stream");
        commands += a[index].commands;
    }
    check(commands != 0, "the command log saw no commands");
    bool differs = a.size() != c.size();
    for (std::size_t index = 0; !differs && index < a.size(); ++index) differs = a[index].hash != c[index].hash;
    check(differs, "a fill of other words logged the same command stream");
}

}

int main() {
    try {
        predictor();
        slotClaim();
        check(Prep::LockSplit() == !AgcDriver::Graphics::RuntimeSwitches::Set(AgcDriver::Graphics::RuntimeSwitch::NoCommitLockSplit), "the lock split ignores APS5_NO_COMMIT_LOCK_SPLIT");
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        preparedStamps(*device);
        preDrainCoversLaterWork(*device);
        commandLog(*device);
        std::puts("commit lock split tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
