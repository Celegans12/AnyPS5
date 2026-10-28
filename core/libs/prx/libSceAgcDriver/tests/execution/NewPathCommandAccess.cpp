#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandAccess.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DevicePageTable.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace AgcDriver::NewPath;
namespace BdaAbi = ShaderRecompiler::BdaAbi;

constexpr std::size_t BlockBytes = 4u << 16u;
constexpr std::uint64_t Page = std::uint64_t{1} << BdaAbi::PageShift;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(Page, BlockBytes));
#endif
        if (block == nullptr) throw std::runtime_error("cannot allocate a guest block");
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
    [[nodiscard]] std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(block); }

private:
    std::uint8_t* block = nullptr;
};

}

int main() {
    try {
        auto device = OpenVulkanTestDevice();
        if (device == nullptr) return VulkanTestSkipped;
        const GuestBlock guest;
        const auto base = guest.Address();
        const std::vector<BdaAbi::Range> ranges{
            {base, base + 2 * Page, 0x7f0000000000ull, BdaAbi::Read | BdaAbi::Write, 0u},
            {base + 2 * Page, base + 3 * Page, 0x7f0000020000ull, BdaAbi::Read, 0u},
        };
        DevicePageTable table([&](const AgcDriver::Graphics::Context& built) {
            AgcDriver::Graphics::HostImportSnapshot snapshot;
            snapshot.device = built.device;
            snapshot.ranges = ranges;
            return snapshot;
        }, [](VkDevice) { return std::uint64_t{1}; });
        const auto context = device->graphicsContext();
        const auto version = table.Refresh(context);
        const auto lists = table.Lists();
        Expect(lists.mapped->size() == 1 && lists.mapped->front() == AddressRange{base, base + 3 * Page}, "the mapped list is not the merged table ranges");
        Expect(lists.writable->size() == 1 && lists.writable->front() == AddressRange{base, base + 2 * Page}, "the writable list is not the read-write table ranges");

        auto* recorder = AgcDriver::Graphics::Recorder::Active();
        if (recorder == nullptr) {
            std::puts("skipped, the device has no active recorder");
            return VulkanTestSkipped;
        }
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            Expect(version->NeedsUploads(0) && version->NeedsUploads(1), "a new table version needs no upload");
            Expect(UploadTableForLane(context, *recorder, version, 0), "the first use on lane 0 recorded no upload");
            Expect(!UploadTableForLane(context, *recorder, version, 0), "a second use on lane 0 uploaded again");
            Expect(!version->NeedsUploads(0) && version->NeedsUploads(1), "the upload marks are not per lane");

            WriteSet bounded;
            bounded.ranges = {{base + 0x100, base + 0x200}};
            const auto boundedNotes = NoteCommandAccess(*recorder, bounded, lists, true);
            Expect(boundedNotes.boundedRanges == 1 && !boundedNotes.leasedWritable && boundedNotes.notedReads, "a bounded writer's notes are wrong");
            Expect(recorder->PendingWriteOverlaps(base + 0x180, 4), "a bounded store range is not a pending write");
            Expect(!recorder->PendingWriteOverlaps(base + Page + 0x180, 4), "a page outside the bounded stores became a pending write");
            Expect(recorder->PendingLeaseSerial(base + Page, 4) == 0, "a bounded writer leased the writable memory");
            Expect(recorder->PendingReadOverlaps(base + 2 * Page + 8, 4, false), "the mapped reads of a new-path command are not noted");

            const auto unbounded = ResolveWrites(table.Ranges(), Abi::StoreClass::Unbounded, {});
            const auto unboundedNotes = NoteCommandAccess(*recorder, unbounded, lists, true);
            Expect(unboundedNotes.leasedWritable && unboundedNotes.boundedRanges == 0, "an unbounded writer was not leased");
            Expect(recorder->PendingLeaseSerial(base + Page + 0x40, 4) != 0, "a writable page is not under the unbounded writer's lease");
            Expect(recorder->PendingLeaseSerial(base + 2 * Page + 0x40, 4) == 0, "a read-only page is under the unbounded writer's lease");
            Expect(recorder->PendingWriterSerial(base + Page + 0x40, 4) != 0, "an old-path writer check does not see the unbounded writer");
        }
        device->WaitIdle();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d new-path command access checks failed\n", failures);
        return 1;
    }
    std::puts("new-path command access tests passed");
    return 0;
}
