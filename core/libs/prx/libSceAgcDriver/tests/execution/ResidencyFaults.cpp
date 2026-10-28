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
#include <span>
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
        driver.Manager().NoteGpuWritten(guest.Address(index, subpage), ResidencySubpageBytes);
        VkBuffer target = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        std::shared_ptr<MirrorChunk> chunk;
        if (const auto slot = driver.Manager().Slot(Page(index))) {
            chunk = driver.Mirrors().Chunk(slot->chunk);
            target = chunk->buffer;
            offset = (VkDeviceSize{slot->slot} << ResidencyPageShift) + (VkDeviceSize{subpage} << ResidencySubpageShift);
        } else {
            const auto* import = HostImportFor(context, guest.Address(index, subpage), ResidencySubpageBytes);
            if (import == nullptr) throw std::runtime_error("the demoted page has no import to write");
            target = import->buffer;
            offset = HostImportOffset(*import, guest.Address(index, subpage));
        }
        const auto commands = recorder.Commands();
        Barrier(commands, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, target, offset, ResidencySubpageBytes, value);
        Barrier(commands, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT);
        recorder.MarkCovered(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        if (chunk != nullptr) recorder.Keep(chunk);
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

void RefusedGuardWritesBack(const Harness& harness, std::uint32_t index) {
    harness.Use(index, 0);
    if (!harness.driver.Manager().Slot(harness.Page(index))) {
        Expect(false, "refused guard: the page was not admitted");
        return;
    }
    const auto unguardable = harness.driver.Manager().Counts().accessUnguardable;
    GuestArena::GuestArenaPinWritable_nid_postfix(harness.guest.Words(index), ResidencyPageBytes);
    harness.FillMirror(index, 6, 0x6e6e6e6eu, 0);
    Expect(harness.driver.Manager().Counts().accessUnguardable == unguardable + 1, "refused guard: the pinned page's guard was not refused");
    Expect(!GuestArena::GuestArenaAccessGuarded_nid_no_patch(harness.guest.Address(index, 6)), "refused guard: a guard was armed over a pinned page");
    Expect(harness.driver.Manager().Dirty(harness.Page(index)) == 0, "refused guard: a subpage stayed GPU-dirty without a guard (the merge rule's invariant)");
    {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        harness.device.SubmitRecorded(false);
        harness.device.WaitIdle();
    }
    GuestArena::GuestArenaUnpinWritable_nid_postfix(harness.guest.Words(index), ResidencyPageBytes);
    Expect(!harness.driver.Manager().Slot(harness.Page(index)).has_value(), "refused guard: the page was not demoted, so the GPU write could not go to guest memory directly");
    Expect(harness.guest.Words(index)[6 * WordsPerSubpage + 9] == 0x6e6e6e6eu, "refused guard: the GPU write did not reach guest memory");
    Expect(harness.guest.Words(index)[5 * WordsPerSubpage + 9] == Pattern(index, 5 * WordsPerSubpage + 9), "refused guard: a subpage the GPU did not write changed");
}

void DriverReadOpensGuard(const Harness& harness, std::uint32_t index) {
    harness.Use(index, 0);
    if (!harness.driver.Manager().Slot(harness.Page(index))) {
        Expect(false, "driver read: the page was not admitted");
        return;
    }
    harness.FillMirror(index, 0, 0x44444444u, 0);
    std::array<std::uint32_t, 4> words{};
    AgcDriver::GuestMemory::ReadCommitted(harness.guest.Address(index), std::as_writable_bytes(std::span(words)));
    Expect(words[0] == 0x44444444u && words[3] == 0x44444444u, "driver read: a driver read off the GPU mutex did not get the GPU's bytes");
    Expect(!GuestArena::GuestArenaAccessGuarded_nid_no_patch(harness.guest.Address(index)), "driver read: the guard stayed after the read opened it");
}

bool Admitted(const Harness& harness, std::uint32_t index, std::uint32_t value, const char* site) {
    harness.Use(index, 0);
    if (!harness.driver.Manager().Slot(harness.Page(index))) {
        Expect(false, std::string(site) + ": the page was not admitted");
        return false;
    }
    harness.FillMirror(index, 0, value, 0);
    if (!AgcDriver::GuestMemory::GpuOwned(harness.guest.Address(index), 16)) {
        Expect(false, std::string(site) + ": the GPU-written page is not GPU-owned");
        return false;
    }
    return true;
}

void ExpectPublished(const Harness& harness, std::uint32_t index, const std::uint32_t* words, std::uint32_t value, std::uint64_t publishes, std::uint64_t refused, const char* site) {
    Expect(words[0] == value && words[3] == value, std::string(site) + ": the site did not read the GPU's bytes");
    Expect(harness.driver.DriverPublishes() > publishes, std::string(site) + ": the site published nothing");
    Expect(harness.driver.RefusedFaults() == refused, std::string(site) + ": the site took a refused driver fault");
    Expect(!AgcDriver::GuestMemory::GpuOwned(harness.guest.Address(index), 16), std::string(site) + ": the page stayed GPU-owned after the publish");
    Expect(!GuestArena::GuestArenaAccessGuarded_nid_no_patch(harness.guest.Address(index)), std::string(site) + ": the guard stayed after the publish");
}

void WriteBackStampsCaches(const Harness& harness, std::uint32_t index) {
    const auto address = harness.guest.Address(index);
    std::array<std::uint32_t, 4> words{};
    {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        AgcDriver::GuestMemory::ReadCommitted(address, std::as_writable_bytes(std::span(words)));
    }
    const auto clean = AgcDriver::GuestMemory::NewestStamp(address, ResidencyPageBytes);
    const auto stampsBefore = harness.driver.WriteBackStamps();
    {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        AgcDriver::GuestMemory::ReadCommitted(address, std::as_writable_bytes(std::span(words)));
    }
    Expect(AgcDriver::GuestMemory::NewestStamp(address, ResidencyPageBytes) == clean && harness.driver.WriteBackStamps() == stampsBefore, "stamps: a read with nothing written back changed the page's stamp (a cache hit lost)");
    if (!Admitted(harness, index, 0x55555555u, "stamps")) return;
    {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        AgcDriver::GuestMemory::ReadCommitted(address, std::as_writable_bytes(std::span(words)));
    }
    Expect(words[0] == 0x55555555u, "stamps: the publish did not read the GPU's bytes");
    Expect(harness.driver.WriteBackStamps() > stampsBefore, "stamps: the write-back stamped nothing");
    Expect(AgcDriver::GuestMemory::NewestStamp(address, ResidencyPageBytes) > clean, "stamps: a written-back page kept its stamp, so a snapshot cached over the old bytes would still hit");
}

void ProofIsNotProvenOverGpuOwned(const Harness& harness, std::uint32_t index) {
    const auto address = harness.guest.Address(index);
    std::vector<AgcDriver::ContentHash::Hash128> hashes(AgcDriver::GuestMemory::HashBlockCount(address, ResidencyPageBytes));
    {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        AgcDriver::GuestMemory::HashCommittedBlocks(address, ResidencyPageBytes, hashes);
        Expect(AgcDriver::GuestMemory::CommittedBlocksMatch(address, ResidencyPageBytes, hashes), "proof: unchanged blocks did not match their own hashes");
    }
    if (!Admitted(harness, index, 0x56565656u, "proof")) return;
    const auto publishes = harness.driver.DriverPublishes();
    const auto refused = harness.driver.RefusedFaults();
    Expect(!AgcDriver::GuestMemory::CommittedBlocksMatch(address, ResidencyPageBytes, hashes), "proof: a GPU-owned range was proven unchanged from the guest's stale bytes");
    Expect(harness.driver.DriverPublishes() == publishes && harness.driver.RefusedFaults() == refused, "proof: answering not proven read or published the GPU-owned range");
    Expect(AgcDriver::GuestMemory::GpuOwned(address, 16), "proof: answering not proven published the range");
    {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        AgcDriver::GuestMemory::PublishGpuOwned(address, ResidencyPageBytes);
    }
}

void DriverSites(const Harness& harness, std::uint32_t first) {
    {
        const char* site = "GuestMemory::ReadCommitted under the GPU mutex";
        if (Admitted(harness, first, 0x51515151u, site)) {
            const auto publishes = harness.driver.DriverPublishes();
            const auto refused = harness.driver.RefusedFaults();
            std::array<std::uint32_t, 4> words{};
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            AgcDriver::GuestMemory::ReadCommitted(harness.guest.Address(first), std::as_writable_bytes(std::span(words)));
            ExpectPublished(harness, first, words.data(), 0x51515151u, publishes, refused, site);
        }
    }
    {
        const char* site = "FlushGpuWrites before a raw read (CopyDrawInput, CapturedCurrent)";
        if (Admitted(harness, first + 1, 0x52525252u, site)) {
            const auto publishes = harness.driver.DriverPublishes();
            const auto refused = harness.driver.RefusedFaults();
            std::array<std::uint32_t, 4> words{};
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            AgcDriver::GuestMemory::FlushGpuWrites(harness.guest.Address(first + 1), sizeof(words));
            std::memcpy(words.data(), reinterpret_cast<const void*>(harness.guest.Address(first + 1)), sizeof(words));
            ExpectPublished(harness, first + 1, words.data(), 0x52525252u, publishes, refused, site);
        }
    }
    {
        const char* site = "GuardRead before a raw copy (PrepareDrawBindings)";
        if (Admitted(harness, first + 2, 0x53535353u, site)) {
            const auto publishes = harness.driver.DriverPublishes();
            const auto refused = harness.driver.RefusedFaults();
            std::array<std::uint32_t, 4> words{};
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            AgcDriver::GuestMemory::GuardRead(harness.guest.Address(first + 2), sizeof(words));
            std::memcpy(words.data(), reinterpret_cast<const void*>(harness.guest.Address(first + 2)), sizeof(words));
            ExpectPublished(harness, first + 2, words.data(), 0x53535353u, publishes, refused, site);
        }
    }
    {
        const char* site = "PublishGpuOwned before a GPU copy of the import (recordGpuCopies, DCC keys under the lock)";
        if (Admitted(harness, first + 3, 0x54545454u, site)) {
            Expect(AgcDriver::GuestMemory::GpuOwned(harness.guest.Address(first + 3), 16), "DCC keys off the GPU mutex: the owned range is not reported, so the presenter would read it");
            const auto publishes = harness.driver.DriverPublishes();
            const auto refused = harness.driver.RefusedFaults();
            {
                const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
                AgcDriver::GuestMemory::PublishGpuOwned(harness.guest.Address(first + 3), ResidencyPageBytes);
            }
            std::array<std::uint32_t, 4> words{};
            std::memcpy(words.data(), reinterpret_cast<const void*>(harness.guest.Address(first + 3)), sizeof(words));
            ExpectPublished(harness, first + 3, words.data(), 0x54545454u, publishes, refused, site);
        }
    }
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
    const bool sitesOnly = argc > 1 && std::string(argv[1]) == "--driver-sites";
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
            if (!sitesOnly) {
                SubpageMerge(harness, 0, 0);
                if (asyncLane) SubpageMerge(harness, 1, 1);
                CollectCannotHideCpuWrite(harness, 2);
                OwnerIsNeverServed(harness, 3);
                RefusedGuardWritesBack(harness, 4);
                DriverReadOpensGuard(harness, 5);
            }
            DriverSites(harness, 6);
            WriteBackStampsCaches(harness, 10);
            ProofIsNotProvenOverGpuOwned(harness, 11);
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
