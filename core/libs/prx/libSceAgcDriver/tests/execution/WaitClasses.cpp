#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/WaitMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::DriverDetail::WaitClass;
using AgcDriver::DriverDetail::WaitClassCount;
using AgcDriver::DriverDetail::WaitClassNames;
using AgcDriver::DriverDetail::WaitClassTotals;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

alignas(256) std::array<std::uint32_t, 15> SpinCode{0xf4080200u, 0xfa000000u, 0xbe840381u, 0x9304ff04u, 0x0019660du, 0x8004ff04u, 0x3c6ef35fu, 0x80828102u, 0xbf078002u, 0xbf85fff9u, 0xbf8cc07fu, 0x7e000204u, 0xe0700000u, 0x80020000u, 0xbf810000u};
alignas(256) std::array<std::uint32_t, 11> CopyCode{0xf4080200u, 0xfa000000u, 0xf4080300u, 0xfa000010u, 0xbf8cc07fu, 0xe0300000u, 0x80020000u, 0xbf8c3f70u, 0xe0700000u, 0x80030000u, 0xbf810000u};

constexpr std::uint32_t SpinIterations = 1u << 22u;
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t Sentinel = 0xdeadbeefu;

struct Slot {
    alignas(256) std::array<std::uint32_t, 4> spinOutput{};
    alignas(256) std::array<std::uint32_t, 4> copyOutput{};
    alignas(256) std::array<std::uint32_t, 4> spinDescriptor{};
    alignas(256) std::array<std::uint32_t, 8> copyDescriptors{};
    alignas(256) volatile std::uint32_t label = 0;
    alignas(256) volatile std::uint32_t done = 0;
    alignas(256) volatile std::uint32_t marker = 0;
};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "wait classes: cannot allocate the guest block");
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

    Slot& At(std::size_t index) { return *reinterpret_cast<Slot*>(block + index * 4096); }

private:
    std::uint8_t* block = nullptr;
};

std::uint32_t Low(const volatile void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint32_t High(const volatile void* pointer) {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)) >> 32u);
}

std::array<std::uint32_t, 4> Descriptor(const volatile void* buffer, std::uint32_t bytes) {
    return {Low(buffer), High(buffer) & 0xffffu, bytes, 0x31016facu};
}

void SetShaderRegister(std::vector<std::uint32_t>& words, std::uint32_t offset, std::uint32_t value) {
    words.insert(words.end(), {0xc0017600u, offset, value});
}

void Dispatch(std::vector<std::uint32_t>& words, const void* code, std::initializer_list<std::uint32_t> userData) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(code));
    SetShaderRegister(words, 0x20c, static_cast<std::uint32_t>(address >> 8u));
    SetShaderRegister(words, 0x20d, static_cast<std::uint32_t>(address >> 40u) & 0xffu);
    for (const std::uint32_t offset : {0x207u, 0x208u, 0x209u}) SetShaderRegister(words, offset, 1);
    SetShaderRegister(words, 0x212, 0);
    SetShaderRegister(words, 0x213, static_cast<std::uint32_t>(userData.size()) << 1u);
    std::uint32_t offset = 0x240;
    for (const auto value : userData) SetShaderRegister(words, offset++, value);
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x8041u});
}

void Spin(std::vector<std::uint32_t>& words, Slot& slot) {
    slot.spinDescriptor = Descriptor(slot.spinOutput.data(), sizeof(slot.spinOutput));
    Dispatch(words, SpinCode.data(), {Low(slot.spinDescriptor.data()), High(slot.spinDescriptor.data()), SpinIterations});
}

void Copy(std::vector<std::uint32_t>& words, Slot& source, Slot& slot) {
    const auto in = Descriptor(source.spinOutput.data(), sizeof(source.spinOutput));
    const auto out = Descriptor(slot.copyOutput.data(), sizeof(slot.copyOutput));
    std::memcpy(slot.copyDescriptors.data(), in.data(), sizeof(in));
    std::memcpy(slot.copyDescriptors.data() + 4, out.data(), sizeof(out));
    Dispatch(words, CopyCode.data(), {Low(slot.copyDescriptors.data()), High(slot.copyDescriptors.data())});
}

void ReleaseLabel(std::vector<std::uint32_t>& words, volatile std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0064900u, 0x514u, (1u << 29u) | (2u << 24u), Low(target), High(target), value, 0u, 0u});
}

void WriteData(std::vector<std::uint32_t>& words, volatile std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0033700u, 0x00100200u, Low(target), High(target), value});
}

void WaitEqual(std::vector<std::uint32_t>& words, volatile std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0053c00u, 0x13u, Low(target), High(target), value, 0xffffffffu, 0x19u});
}

void Submit(std::uint32_t queue, std::vector<std::uint32_t>& words) {
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    const int result = queue == 0 ? sceAgcDriverSubmitDcb(&packet) : sceAgcDriverSubmitAcb(queue, &packet);
    Require(result == 0, "submit failed on queue " + std::to_string(queue));
}

void AwaitValue(const volatile std::uint32_t& value, std::uint32_t expected, const char* what) {
    const auto start = std::chrono::steady_clock::now();
    while (value != expected) {
        Require(std::chrono::steady_clock::now() - start < std::chrono::seconds(20), std::string(what) + " did not arrive");
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

std::uint32_t SpinResult() {
    std::uint32_t value = 1;
    for (std::uint32_t i = 0; i < SpinIterations; ++i) value = value * 0x19660du + 0x3c6ef35fu;
    return value;
}

std::uint64_t Delta(const WaitClassTotals& before, const WaitClassTotals& after, WaitClass which) {
    const auto index = static_cast<std::size_t>(which);
    return after.count[index] - before.count[index];
}

void Report(const char* name, const WaitClassTotals& before, const WaitClassTotals& after) {
    std::cout << name << ":";
    for (std::size_t i = 0; i < WaitClassCount; ++i) {
        const auto count = after.count[i] - before.count[i];
        if (count != 0) std::cout << " " << WaitClassNames[i] << " " << count;
    }
    std::cout << "; drains skipped " << after.drainsSkipped - before.drainsSkipped << ", mutex tries skipped " << after.triesSkipped - before.triesSkipped << "\n";
}

void SameQueueGpuLabel(GuestBlock& block) {
    auto& slot = block.At(0);
    slot.spinOutput.fill(Sentinel);
    std::vector<std::uint32_t> words;
    Spin(words, slot);
    ReleaseLabel(words, &slot.label, 11);
    WaitEqual(words, &slot.label, 11);
    Copy(words, slot, slot);
    WriteData(words, &slot.done, 1);
    const auto before = AgcDriver::DriverDetail::WaitClassCounts();
    Submit(0x20, words);
    AwaitValue(slot.done, 1, "same queue: the end marker");
    AgcDriverWaitIdle_nid_postfix();
    const auto after = AgcDriver::DriverDetail::WaitClassCounts();
    Report("gpu label, same queue", before, after);
    Require(slot.copyOutput[0] == SpinResult(), "same queue: the copy after the wait read the spin result early");
    Require(Delta(before, after, WaitClass::GpuSameQueue) + Delta(before, after, WaitClass::HistorySameQueue) + Delta(before, after, WaitClass::Entry) == 1, "same queue: the wait was not taken from the queue's own label");
}

void CrossQueueGpuLabel(GuestBlock& block) {
    auto& writer = block.At(1);
    auto& reader = block.At(2);
    writer.spinOutput.fill(Sentinel);
    reader.copyOutput.fill(Sentinel);
    std::vector<std::uint32_t> first;
    Spin(first, writer);
    ReleaseLabel(first, &writer.label, 22);
    std::vector<std::uint32_t> second;
    WaitEqual(second, &writer.label, 22);
    Copy(second, writer, reader);
    WriteData(second, &reader.done, 1);
    const auto before = AgcDriver::DriverDetail::WaitClassCounts();
    Submit(0x20, first);
    const auto start = std::chrono::steady_clock::now();
    while (!AgcDriver::Graphics::Recorder::LookupLabelValue(reinterpret_cast<std::uint64_t>(&writer.label), 4, 0).has_value() && writer.label != 22) {
        Require(std::chrono::steady_clock::now() - start < std::chrono::seconds(20), "cross queue: the writer's label was never recorded");
        std::this_thread::yield();
    }
    Submit(0x24, second);
    AwaitValue(reader.done, 1, "cross queue: the end marker");
    AgcDriverWaitIdle_nid_postfix();
    const auto after = AgcDriver::DriverDetail::WaitClassCounts();
    Report("gpu label, cross queue", before, after);
    Require(reader.copyOutput[0] == SpinResult(), "cross queue: the copy after the wait read the spin result early");
    Require(Delta(before, after, WaitClass::GpuCrossQueue) + Delta(before, after, WaitClass::GpuLaterCrossQueue) + Delta(before, after, WaitClass::HistoryCrossQueue) + Delta(before, after, WaitClass::Entry) + Delta(before, after, WaitClass::MemoryGpu) == 1, "cross queue: the wait was not classed as a GPU-written label");
    Require(Delta(before, after, WaitClass::Landed) + Delta(before, after, WaitClass::LandedTimeline) <= 1, "cross queue: the wait landed more than once");
    if (!AgcDriver::DriverDetail::GpuWaits()) Require(Delta(before, after, WaitClass::LandedTimeline) == 0, "cross queue: a timeline landing with APS5_GPU_WAITS off");
}

void CpuLabel(GuestBlock& block) {
    auto& slot = block.At(3);
    slot.label = 0;
    std::vector<std::uint32_t> words;
    WaitEqual(words, &slot.label, 33);
    WriteData(words, &slot.done, 1);
    const auto before = AgcDriver::DriverDetail::WaitClassCounts();
    Submit(0x20, words);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Require(slot.done == 0, "cpu label: the wait passed before the CPU stored the label");
    slot.label = 33;
    AwaitValue(slot.done, 1, "cpu label: the end marker");
    AgcDriverWaitIdle_nid_postfix();
    const auto after = AgcDriver::DriverDetail::WaitClassCounts();
    Report("cpu label", before, after);
    Require(Delta(before, after, WaitClass::MemoryCpu) == 1, "cpu label: the wait was not classed as a CPU-written label");
}

void LaterGpuLabel(GuestBlock& block) {
    auto& writer = block.At(4);
    auto& reader = block.At(5);
    writer.label = 0;
    writer.spinOutput.fill(Sentinel);
    reader.copyOutput.fill(Sentinel);
    std::vector<std::uint32_t> waiting;
    WaitEqual(waiting, &writer.label, 44);
    Copy(waiting, writer, reader);
    WriteData(waiting, &reader.done, 1);
    std::vector<std::uint32_t> writing;
    Spin(writing, writer);
    ReleaseLabel(writing, &writer.label, 44);
    const auto before = AgcDriver::DriverDetail::WaitClassCounts();
    Submit(0x24, waiting);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Require(reader.done == 0 && reader.copyOutput[0] == Sentinel, "later label: the queue passed its wait before the label was recorded");
    Submit(0x20, writing);
    AwaitValue(reader.done, 1, "later label: the end marker");
    AgcDriverWaitIdle_nid_postfix();
    const auto after = AgcDriver::DriverDetail::WaitClassCounts();
    Report("gpu label recorded after the wait", before, after);
    Require(reader.copyOutput[0] == SpinResult(), "later label: the copy after the wait read the spin result early");
    Require(Delta(before, after, WaitClass::GpuLaterCrossQueue) + Delta(before, after, WaitClass::HistoryCrossQueue) + Delta(before, after, WaitClass::MemoryGpu) == 1, "later label: the wait was not classed as a label recorded later on the GPU");
}

void GraphicsQueueWaits(GuestBlock& block) {
    auto& slot = block.At(6);
    slot.label = 0;
    slot.marker = 0;
    slot.done = 0;
    std::vector<std::uint32_t> blocked;
    WriteData(blocked, &slot.marker, 5);
    WaitEqual(blocked, &slot.label, 55);
    WriteData(blocked, &slot.done, 1);
    const auto before = AgcDriver::DriverDetail::WaitClassCounts();
    Submit(0, blocked);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Require(slot.done == 0, "graphics queue: the wait passed before the CPU stored the label");
    slot.label = 55;
    AwaitValue(slot.done, 1, "graphics queue: the end marker after the CPU label");
    std::vector<std::uint32_t> ready;
    WriteData(ready, &slot.marker, 6);
    WaitEqual(ready, &slot.label, 55);
    WriteData(ready, &slot.done, 2);
    Submit(0, ready);
    AwaitValue(slot.done, 2, "graphics queue: the end marker after a satisfied wait");
    AgcDriverWaitIdle_nid_postfix();
    const auto after = AgcDriver::DriverDetail::WaitClassCounts();
    Report("graphics queue", before, after);
    Require(slot.marker == 6, "graphics queue: a store before the wait was lost");
    Require(Delta(before, after, WaitClass::MemoryCpu) == 1, "graphics queue: the blocked wait was not classed as a CPU-written label");
    if (!AgcDriver::DriverDetail::GpuWaits()) Require(after.drainsSkipped == before.drainsSkipped, "graphics queue: a pipeline drain was skipped with APS5_GPU_WAITS off");
}

}

int main() {
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        device.reset();
        GuestBlock block;
        std::cout << "APS5_GPU_WAITS " << (AgcDriver::DriverDetail::GpuWaits() ? "on" : "off") << "\n";
        SameQueueGpuLabel(block);
        CrossQueueGpuLabel(block);
        CpuLabel(block);
        LaterGpuLabel(block);
        GraphicsQueueWaits(block);
        const auto totals = AgcDriver::DriverDetail::WaitClassCounts();
        if (!AgcDriver::DriverDetail::GpuWaits()) {
            Require(totals.count[static_cast<std::size_t>(WaitClass::LandedTimeline)] == 0, "a wait landed on the timeline with APS5_GPU_WAITS off");
            Require(totals.triesSkipped == 0, "a mutex try was skipped with APS5_GPU_WAITS off");
        }
        AgcDriverShutdown_nid_postfix();
        std::cout << "wait classes: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "wait classes: " << error.what() << "\n";
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (const std::exception& shutdownError) {
            std::cerr << shutdownError.what() << "\n";
        }
        return 1;
    }
}
