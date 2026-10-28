#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyGuest.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using namespace AgcDriver::Graphics;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

constexpr std::uint32_t Pages = 16;
constexpr std::size_t Bytes = std::size_t{Pages} << ResidencyPageShift;
constexpr std::uint32_t WordsPerSubpage = ResidencySubpageBytes / 4;

std::uint32_t Pattern(std::uint32_t page, std::uint32_t word) {
    return (page << 24) | word;
}

class GuestPages {
public:
    GuestPages() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(Bytes, ResidencyPageBytes));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, Bytes, PAGE_READWRITE, Bytes);
#endif
        if (block == nullptr) throw std::runtime_error("cannot allocate guest pages");
        for (std::uint32_t page = 0; page < Pages; ++page) {
            for (std::uint32_t word = 0; word < ResidencyPageBytes / 4; ++word) Words(page)[word] = Pattern(page, word);
        }
        GuestAllocations::Mutation().Add(block, Bytes, true, true, true);
    }

    std::uint64_t Address(std::uint32_t page, std::uint32_t subpage = 0) const {
        return reinterpret_cast<std::uintptr_t>(block) + (std::uint64_t{page} << ResidencyPageShift) + (std::uint64_t{subpage} << ResidencySubpageShift);
    }
    std::uint32_t* Words(std::uint32_t page) const { return reinterpret_cast<std::uint32_t*>(block + (std::size_t{page} << ResidencyPageShift)); }

private:
    std::uint8_t* block = nullptr;
};

struct Harness {
    AgcDriver::VulkanDevice& device;
    const Context& context;
    Recorder& recorder;
    ResidencyDriver& driver;
    const GuestPages& guest;

    std::uint64_t Page(std::uint32_t index) const { return guest.Address(index) >> ResidencyPageShift; }

    void Barrier(VkCommandBuffer commands, VkAccessFlags from, VkPipelineStageFlags fromStage, VkAccessFlags to, VkPipelineStageFlags toStage) const {
        const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, from, to};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, fromStage, toStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    void Use(std::uint32_t index, int lane) const {
        const Recorder::LaneScope scope(lane);
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        driver.Manager().Use(guest.Address(index), ResidencyPageBytes, ResidencyAccess::Write);
    }

    void FillMirror(std::uint32_t index, std::uint32_t subpage, std::uint32_t value, int lane) const {
        const Recorder::LaneScope scope(lane);
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        const auto slot = driver.Manager().Slot(Page(index));
        if (!slot) throw std::runtime_error("the page to fill is not resident");
        const auto chunk = driver.Mirrors().Chunk(slot->chunk);
        const auto commands = recorder.Commands();
        Barrier(commands, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, chunk->buffer, (VkDeviceSize{slot->slot} << ResidencyPageShift) + (VkDeviceSize{subpage} << ResidencySubpageShift), ResidencySubpageBytes, value);
        Barrier(commands, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        recorder.MarkCovered(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        recorder.Keep(chunk);
        driver.Manager().NoteGpuWritten(guest.Address(index, subpage), ResidencySubpageBytes);
    }

    std::vector<std::uint32_t> ReadMirror(std::uint32_t index, int lane) const {
        auto readback = std::make_shared<Buffer>(context, ResidencyPageBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        {
            const Recorder::LaneScope scope(lane);
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            const auto slot = driver.Manager().Slot(Page(index));
            if (!slot) throw std::runtime_error("the page to read back is not resident");
            const auto chunk = driver.Mirrors().Chunk(slot->chunk);
            const auto commands = recorder.Commands();
            Barrier(commands, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            const VkBufferCopy region{VkDeviceSize{slot->slot} << ResidencyPageShift, 0, ResidencyPageBytes};
            context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(commands, chunk->buffer, readback->Handle(), 1, &region);
            Barrier(commands, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_HOST_BIT);
            recorder.MarkCovered(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
            recorder.Keep(chunk);
            recorder.Keep(readback);
            device.SubmitRecorded(false);
            device.WaitIdle();
        }
        std::vector<std::uint32_t> words(ResidencyPageBytes / 4);
        std::memcpy(words.data(), readback->Bytes().data(), ResidencyPageBytes);
        return words;
    }

    std::uint32_t Expected(std::uint32_t index, std::uint32_t word, std::uint32_t cpuSubpage, std::uint32_t cpuValue, std::uint32_t gpuSubpage, std::uint32_t gpuValue) const {
        const auto subpage = word / WordsPerSubpage;
        if (subpage == gpuSubpage) return gpuValue;
        if (subpage == cpuSubpage && word % WordsPerSubpage == 0) return cpuValue;
        return Pattern(index, word);
    }
};

void SubpageMerge(const Harness& harness, std::uint32_t index, int lane) {
    const auto tag = "lane " + std::to_string(lane) + ": ";
    const std::uint32_t gpuValue = 0x5a000000u + static_cast<std::uint32_t>(lane);
    const std::uint32_t cpuValue = 0x3c000000u + static_cast<std::uint32_t>(lane);
    const auto served = harness.driver.ServedFaults() + harness.driver.FastFaults();
    harness.Use(index, lane);
    Expect(harness.driver.Manager().Slot(harness.Page(index)).has_value(), tag + "the used page was not admitted");
    if (!harness.driver.Manager().Slot(harness.Page(index))) return;
    harness.FillMirror(index, 3, gpuValue, lane);
    Expect(GuestArena::GuestArenaAccessGuarded_nid_no_patch(harness.guest.Address(index, 3)), tag + "the GPU-written subpage 3 is not access-guarded");
    Expect(!GuestArena::GuestArenaAccessGuarded_nid_no_patch(harness.guest.Address(index, 2)), tag + "subpage 2 is guarded though the GPU did not write it");
    reinterpret_cast<volatile std::uint32_t*>(harness.guest.Words(index))[2 * WordsPerSubpage] = cpuValue;
    harness.Use(index, lane);
    const auto mirror = harness.ReadMirror(index, lane);
    Expect(mirror[2 * WordsPerSubpage] == cpuValue, tag + "the next use did not copy the CPU's store into subpage 2 of the mirror");
    Expect(mirror[3 * WordsPerSubpage + 7] == gpuValue, tag + "the next use overwrote the GPU-dirty subpage 3 of the mirror with stale guest bytes");
    const auto read = reinterpret_cast<volatile std::uint32_t*>(harness.guest.Words(index))[3 * WordsPerSubpage + 5];
    Expect(read == gpuValue, tag + "the CPU read of subpage 3 returned " + std::to_string(read) + ", expected the GPU's " + std::to_string(gpuValue));
    Expect(harness.driver.ServedFaults() + harness.driver.FastFaults() == served + 1, tag + "the read did not take exactly one residency fault");
    std::uint32_t wrong = 0;
    for (std::uint32_t word = 0; word < ResidencyPageBytes / 4; ++word) wrong += harness.guest.Words(index)[word] != harness.Expected(index, word, 2, cpuValue, 3, gpuValue);
    Expect(wrong == 0, tag + std::to_string(wrong) + " guest words differ after the merge and the write-back");
}

void CollectCannotHideCpuWrite(const Harness& harness, std::uint32_t index) {
    harness.Use(index, 0);
    if (!harness.driver.Manager().Slot(harness.Page(index))) {
        Expect(false, "collect: the page was not admitted");
        return;
    }
    const std::uint32_t value = 0x77000000u;
    reinterpret_cast<volatile std::uint32_t*>(harness.guest.Words(index))[4 * WordsPerSubpage] = value;
    static_cast<void>(AgcDriver::GuestMemory::CollectWritesUncached(harness.guest.Address(index), ResidencyPageBytes));
    std::array<std::uint64_t, 1> words{};
    static_cast<void>(GuestArena::GuestArenaArmOwned_nid_no_patch(static_cast<std::uintptr_t>(harness.guest.Address(index)), ResidencyPageBytes, words.data(), words.size(), 1));
    harness.Use(index, 0);
    const auto mirror = harness.ReadMirror(index, 0);
    Expect(mirror[4 * WordsPerSubpage] == value, "collect: a CPU store collected by the tracker and then taken by an owned arming was missing from the mirror at the next use");
    const auto other = static_cast<std::uint32_t>(index);
    reinterpret_cast<volatile std::uint32_t*>(harness.guest.Words(other))[5 * WordsPerSubpage] = value + 1;
    static_cast<void>(AgcDriver::GuestMemory::CollectWritesUncached(harness.guest.Address(other), ResidencyPageBytes));
    harness.driver.OrderingPoint();
    const auto later = harness.ReadMirror(other, 0);
    Expect(later[5 * WordsPerSubpage] == value + 1, "collect: an ordering point after the tracker's collect did not bring the CPU store into the mirror");
}

void OwnerIsNeverServed(const Harness& harness, std::uint32_t index) {
    harness.Use(index, 0);
    if (!harness.driver.Manager().Slot(harness.Page(index))) {
        Expect(false, "owner: the page was not admitted");
        return;
    }
    harness.FillMirror(index, 1, 0x0badf00du, 0);
    {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        const auto refused = harness.driver.RefusedFaults();
        harness.driver.AccessFault(harness.guest.Address(index, 1));
        Expect(harness.driver.RefusedFaults() == refused + 1, "owner: a fault on the GPU mutex's holder was served");
        Expect(GuestArena::GuestArenaAccessGuarded_nid_no_patch(harness.guest.Address(index, 1)), "owner: the refused fault dropped the guard");
    }
    Expect(reinterpret_cast<volatile std::uint32_t*>(harness.guest.Words(index))[WordsPerSubpage] == 0x0badf00du, "owner: a later read off the mutex was not served");
}

}

int main(int argc, char** argv) {
    const bool asyncLane = argc > 1 && std::string(argv[1]) == "--async-lane";
#ifdef _WIN32
    if (asyncLane) _putenv_s("APS5_ASYNC_LANE", "51");
#else
    std::puts("skipped: residency faults use the Windows guest arena guards");
    return VulkanTestSkipped;
#endif
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto context = device->graphicsContext();
        auto* recorder = Recorder::Active();
        if (recorder == nullptr) throw std::runtime_error("no active recorder");
        if (asyncLane && !recorder->AsyncLaneActive()) {
            std::puts("skipped: the device has no third queue for the async lane");
            return VulkanTestSkipped;
        }
        GuestPages guest;
        {
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            Expect(HostImportFor(context, guest.Address(0), Bytes) != nullptr, "the guest pages cannot be imported");
        }
        ArenaResidencyGuest arena(context);
        {
            ResidencyDriver driver(context, *recorder, arena, ResidencyPolicy{Pages, Bytes});
            driver.EnableFaults();
            const Harness harness{*device, context, *recorder, driver, guest};
            SubpageMerge(harness, 0, 0);
            if (asyncLane) SubpageMerge(harness, 1, 1);
            CollectCannotHideCpuWrite(harness, 2);
            OwnerIsNeverServed(harness, 3);
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            device->SubmitRecorded(false);
            device->WaitIdle();
        }
        Expect(GuestArena::GuestArenaAccessGuardedSubpages_nid_no_patch() == 0, "guards outlived the driver");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        return 1;
    }
    if (failures != 0) return 1;
    std::printf("residency fault tests passed%s\n", asyncLane ? " (both lanes)" : "");
    return 0;
}
