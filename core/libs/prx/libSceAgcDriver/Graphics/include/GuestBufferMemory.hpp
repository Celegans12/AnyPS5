#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GUESTBUFFERMEMORY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GUESTBUFFERMEMORY_HPP

#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "BdaAbi.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::GuestMemory {
struct Commitment;
}

namespace AgcDriver::Graphics {

void ShutdownGuestBufferWorkers();

struct GuestMemorySnapshot {
    std::uint64_t address;
    std::span<const std::byte> bytes;
};

class LiveSnapshotInvalidated final : public std::runtime_error {
public:
    LiveSnapshotInvalidated() : std::runtime_error("AGC graphics: guest snapshot differs from registered memory") {}
};

// A guest allocation imported with VK_EXT_external_memory_host, usable by the GPU in place.
struct HostImport {
    std::uint64_t base;
    std::uint64_t bytes;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceAddress address;
    void* alias = nullptr;
    std::shared_ptr<const GuestAllocations::Range> range {};
    std::vector<std::shared_ptr<const GuestAllocations::Range>> ranges {};
    // Identity for the life of this import (see HostImportSerial); 0 until first asked for.
    std::uint64_t serial = 0;
    bool unwatched = false;
    bool settleOnDemand = false;
    bool dmaBuf = false;
    bool joined = false;
    std::shared_ptr<void> chunk;
    mutable std::uint64_t lastUse = 0;
    mutable std::uint64_t spaceSerial = 0;
    std::uint64_t lead = 0;
    std::uint64_t memoryBase = 0;
    std::uint64_t pinnedBytes = 0;
    std::uint32_t memoryType = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t backingSerial = 0;
};

inline VkDeviceSize HostImportOffset(const HostImport& import, std::uint64_t address) {
    return address - import.base + import.lead;
}

enum class ImportWatch : std::uint8_t { Watch, Unwatch };

struct ImportProbe {
    const char* failure = nullptr;
    VkResult result = VK_SUCCESS;
    std::uint32_t pages = 0;
    std::uint32_t writtenAtImport = 0;
    std::uint32_t writtenAfterSubmit = 0;
    std::uint32_t writtenAfterSecondSubmit = 0;
    std::uint32_t writtenDuringSettle = 0;
    std::uint32_t writtenAfterSettle = 0;
    std::uint32_t writtenByCpu = 0;
    std::uint32_t writtenAfterFill = 0;
};

ImportProbe ProbeImportWriteProtection(const Context& context);
ImportProbe ProbeDmaBufImportWriteProtection(const Context& context);
ImportWatch PrepareImportWatch(const Context& context);
void SetImportWatch(const Context& context, ImportWatch watch);

enum class ImportRequester : std::uint8_t { Map, Build, Access, Join, Segment, ColorTarget };
inline constexpr std::size_t ImportRequesterCount = 6;

struct HostImportRequesterCounts {
    std::uint64_t made = 0;
    std::uint64_t madeBytes = 0;
    std::uint64_t refused = 0;
    std::uint64_t refusedBytes = 0;
    std::uint64_t failed = 0;
    std::uint64_t remembered = 0;
};

struct HostImportCounts {
    std::uint64_t live = 0;
    std::uint64_t liveBytes = 0;
    std::array<HostImportRequesterCounts, ImportRequesterCount> requesters{};
};

const char* ImportRequesterName(ImportRequester requester);
HostImportCounts HostImportCounters();
std::string FormatHostImportCounts(const HostImportCounts& counts);
void ReportHostImports();

// The host import of the registered allocation containing [address, address + bytes), made on demand
// (alignment and budget permitting), or null. Bytes at `address` are at `HostImportOffset(*import,
// address)` in the import's buffer.
const HostImport* HostImportFor(const Context& context, std::uint64_t address, std::size_t bytes, ImportRequester requester = ImportRequester::Access);
const HostImport* ExistingHostImportFor(const Context& context, std::uint64_t address, std::size_t bytes);
struct HostImportLookupCache {
    const HostImport* import = nullptr;
    std::uint64_t address = 0;
    std::uint64_t bytes = 0;
    std::uint64_t epoch = 0;
    std::uint64_t inserts = 0;
};
const HostImport* HostImportForCached(const Context& context, std::uint64_t address, std::size_t bytes, HostImportLookupCache& cache);
bool WatchedAfterSettle(const Context& context, std::uint64_t address, std::size_t bytes);
void SetHostImportWindow(std::uint64_t bytes);
// Tests: the window of one large registered range (64 MiB) and the import budget (0: the configured one).
void SetHostImportRangeWindow(std::uint64_t bytes);
void SetHostImportBudget(std::uint64_t bytes);
std::uint64_t HostImportBudgetFor(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, std::uint64_t configured, std::uint64_t liveBytes);
std::uint64_t HostImportBudgetWith(std::uint64_t configured, bool set, std::uint64_t reported, bool allocationFailed, std::uint64_t ceiling);
void SetAddressCopyLimit(std::uint64_t bytes);
void SetHostImportDeferredSettles(bool enabled);
// The larger of 6 GiB and half the physical memory, at most the largest host heap less the larger of 2 GiB and half
// that heap: the other half stays for the buffers, mirrors and textures that share it.
std::uint64_t HostImportBudget(const VkPhysicalDeviceMemoryProperties& memory, std::uint64_t physicalBytes, bool halfHeapReserve = true);
void SetHostImportMaxBytes(std::uint64_t bytes);
std::uint64_t HostImportLiveBytes();
std::uint64_t HostImportBudgetRemaining(const Context& context);
struct HostImportSettleCounts {
    std::uint64_t submitted = 0;
    std::uint64_t waited = 0;
    std::uint64_t rewatched = 0;
    std::uint64_t parked = 0;
    std::uint64_t unmapWaits = 0;
    std::uint64_t keptCompared = 0;
};
HostImportSettleCounts HostImportSettles();
// Tests: how many imports settleImport has used on the settle queue.
std::uint64_t HostImportsSettled();
bool ImportMappedRanges(const Context& context, const GuestAllocations::Mapped& ranges, std::uint64_t generation, bool adoptDevice);
// Whether an existing import covers [address, address + bytes), without reconciling the imports
// with the registry or making one (HostImportFor may take a registry lease): a hint for choices
// made outside the device lock (a sampled texture's path, a dispatch's pre-sync); the path taken
// re-checks with HostImportFor when it binds the import.
bool HostImportCovers(const Context& context, std::uint64_t address, std::size_t bytes);
bool ComparedHostImportCovers(const Context& context, std::uint64_t address, std::size_t bytes);
struct RegisteredPiece {
    std::uint64_t address;
    std::uint64_t bytes;
    bool writable;
    bool gpu;
};
// Whether a readable registered allocation contains [address, address + bytes) right now (one
// registry lease): a storage image whose memory was freed or re-registered has nothing to store to.
bool RegisteredReadableCovers(std::uint64_t address, std::size_t bytes, std::vector<RegisteredPiece>* pieces = nullptr);
std::uint64_t RegisteredReadableEnd(std::uint64_t address);

// A persistent device copy of one registered range of the main guest image (which cannot be host
// imported); see GuestBufferMemory.cpp.
struct ImageMirror;
class Recorder;

template<typename TRange>
void OverlappedRanges(std::span<const TRange> sorted, std::size_t first, std::span<const std::pair<std::uint64_t, std::uint64_t>> writes, std::vector<std::size_t>& indices) {
    indices.clear();
    for (const auto& [begin, end] : writes) {
        auto it = std::partition_point(sorted.begin() + static_cast<std::ptrdiff_t>(first), sorted.end(), [&](const TRange& range) { return range.end <= begin; });
        for (; it != sorted.end() && it->begin < end; ++it) indices.push_back(static_cast<std::size_t>(it - sorted.begin()));
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
}

// Identity of the import serving [address, address + bytes): a serial unique for the life of one
// import, 0 when no import serves the range. An equal serial later means the same VkBuffer still
// backs the range (a re-import gets a new serial), which is what a descriptor set written against it
// needs to stay valid (see the ShaderResources cache). With `reconcile`, imports whose registered
// range changed are dropped first, as an upload does; without it (right after an upload, before the
// work using the import is recorded) the imports are left as they are.
std::uint64_t HostImportSerial(const Context& context, std::uint64_t address, std::size_t bytes, bool reconcile);

// The import table's identity (the fast Revalidate): serials proved under one identity stand
// while the table still has it, i.e. the same device, the same epoch (bumped by every retire)
// and the registry generation it was reconciled with, which must still be the live one.
struct HostImportsProof {
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t epoch = 0;
    std::uint64_t refreshedGeneration = 0;
};
// Whether the table still matches `proof` (one lock); false for an empty or stale proof.
bool HostImportsUnchanged(const Context& context, const HostImportsProof& proof);
// The table's identity when it is reconciled with the live registry (one lock), else empty.
HostImportsProof HostImportsIdentity(const Context& context);

// Identity of the read-only image mirror serving [address, address + bytes) (the counterpart of
// HostImportSerial for main-image ranges, which no import serves): a serial unique for the life of
// one mirror with its top bit set, 0 when no read-only mirror whose registered range still exists
// covers the range. A read-only mirror's bytes and VkBuffer never change, so a descriptor set written
// against it stays valid while the serial is unchanged.
std::uint64_t ImageMirrorSerial(const Context& context, std::uint64_t address, std::size_t bytes);

// Deferred lease release (see GuestBufferMemory.cpp). An address-based build leases every readable
// registered allocation until its write-back, which runs when its batch completed; the guest's
// unmap/mprotect/free of a leased allocation waits for that through the registry's pin waiter, which
// submits the recorder and finishes its batches up to the newest one recorded with a lease. SyncLeaseWork
// says whether the build's work must instead be synced as soon as it is recorded
// (APS5_SYNC_LEASE_DISPATCH=1, the behaviour before deferral). The driver and the draw path count
// their choice with CountLeaseOutcome right after the lease-holding resources were kept by the open
// batch: `batchSerial` is that batch's serial (Recorder::Submissions() + 1 under the device lock), 0
// when synced, and the waiter targets the newest of them. Under APS5_PROFILE_DRAW CountLeaseOutcome
// prints the [address-sync] leases line every 10 s from LeaseCounters (cumulative).
bool SyncLeaseWork();
void CountLeaseOutcome(bool synced, std::uint64_t batchSerial);
void FinishLeaseWork();
struct LeaseStats {
    std::uint64_t deferred = 0;
    std::uint64_t synced = 0;
    // Pin-contention waits made by guest threads (the waiter calls), how many of them finished the
    // newest lease batch, how many had to drain the whole recorder instead (no lease batch noted),
    // how many only dropped the cached address space (no GPU wait, see AddressSpaceStats), and
    // their total time.
    std::uint64_t contentionWaits = 0;
    std::uint64_t contentionSyncs = 0;
    std::uint64_t contentionDrains = 0;
    std::uint64_t cacheDrops = 0;
    double contentionMs = 0;
};
LeaseStats LeaseCounters();

// The cached address space of address-based builds (see GuestBufferMemory.cpp): builds served by
// the cached space, rebuilds by reason, builds whose space could not be published (an import
// retired while it was built) and the pin waiter's drops (cumulative). `enabled` is false under
// APS5_NO_ADDRESS_SPACE_CACHE=1, when every build takes the per-build path.
struct AddressSpaceStats {
    bool enabled = true;
    std::uint64_t hits = 0;
    std::uint64_t rebuiltFirst = 0;
    std::uint64_t rebuiltGeneration = 0;
    std::uint64_t rebuiltEpoch = 0;
    std::uint64_t rebuiltDevice = 0;
    std::uint64_t rebuiltWaiterDrop = 0;
    std::uint64_t unpublished = 0;
    // Builds that left the space for the per-build path: a region partially overlapping one of
    // its regions, or an import retired under it before the upload.
    std::uint64_t dissolvedOverlap = 0;
    std::uint64_t dissolvedImports = 0;
    std::uint64_t waiterDrops = 0;
};
AddressSpaceStats AddressSpaceCounters();

enum class AddressReuse : std::uint8_t { Reused, NotReusable, InUse, Snapshots, Space, Proof, Imports, Threw, Moved, Count };
enum class SpaceResumeMiss : std::uint8_t { State, Replaced, Device, Generation, Epoch, Mirrors, Count };
struct AddressReuseStats {
    std::uint64_t proofs = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(AddressReuse::Count)> outcomes{};
    std::array<std::uint64_t, static_cast<std::size_t>(SpaceResumeMiss::Count)> resumeMisses{};
    std::uint64_t refusedProofNs = 0;
    std::uint64_t reusedHolds = 0;
    std::uint64_t budgetRefusals = 0;
    std::uint64_t budgetRefusalsHeld = 0;
    std::uint64_t heldImportRefusals = 0;
    std::uint64_t heldImportRefusalsHeld = 0;
    std::uint64_t sharedUses = 0;
    std::uint64_t perUseSets = 0;
};
void CountAddressReuseProof();
void CountSharedAddressUse();
void CountPerUseAddressSet();
void CountAddressReuse(AddressReuse outcome, std::uint64_t proofNs = 0);
void HoldReusedSpace(bool held);
AddressReuseStats AddressReuseCounters();

struct MirrorStats {
    std::uint64_t heapMirrors = 0;
    std::uint64_t heapBytes = 0;
    std::uint64_t rebuilds = 0;
    std::uint64_t blocksCopied = 0;
    std::uint64_t heapRefills = 0;
    std::uint64_t sweeps = 0;
    std::uint64_t heapChecks = 0;
    std::uint64_t sparseBinds = 0;
};
MirrorStats MirrorCounters();
std::uint64_t SnapshotMismatches();
struct HeapMirrorHold {
    std::uint64_t mirrors = 0;
    std::uint64_t bytes = 0;
};
HeapMirrorHold HeapMirrorsWithin(std::uint64_t begin, std::uint64_t end);
void ClearSnapshotCopies(VkDevice device);
std::vector<std::uint64_t> AddressTableKey(std::span<const std::pair<std::uint64_t, std::uint64_t>> writes, std::span<const ShaderRecompiler::BdaAbi::Range> own);
void ClearImageMirrors(VkDevice device);
void ClearHostImports(VkDevice device);

struct AddressCopy {
    std::uint64_t begin;
    std::uint64_t end;
    std::uint64_t committed;
    const char* reason;
    bool heap = false;
    bool overBudget = false;
};
std::string AddressCopyOverflow(std::vector<AddressCopy> copies, std::uint64_t limit);

class AddressCopyCounters {
public:
    struct Totals {
        std::uint64_t bytes = 0;
        std::uint64_t largest = 0;
    };

    static AddressCopyCounters& ThisThread();
    static Totals Fold();
    static std::uint64_t Limit();

    void Count(std::uint64_t copied);

private:
    alignas(64) std::atomic<std::uint64_t> bytes{0};
    std::atomic<std::uint64_t> largest{0};
};

class GuestBufferMemory {
public:
    explicit GuestBufferMemory(const Context& context);
    ~GuestBufferMemory();
    void AcquireRegistered();
    // APS5_PROFILE_DRAW: the parts of one address-based build ([address] line every 10 s): the
    // registry lease, the imports pass, the mirror preparation and the mirror compare (blocks and
    // copied blocks) are timed by AcquireRegistered on the calling thread; the caller adds the
    // snapshot compares it made after it (ShaderResources::prepareAddressBindings).
    static void CountAddressBuild(double snapshotsUs);
    // A guest range bound through a descriptor. Both read live guest memory at upload and bind the
    // same way (a storage buffer); AddWritable also notes the range in Writes(), so it gets the
    // write-back's reference copy, a write-back, the recorder's pending-write note and the
    // direct-write marks. AddReadable is for an element the shader is proved never to store to
    // (DescriptorBinding::bufferWritten): the CPU never has to wait for it. `atomic` marks an
    // element the shader updates atomically (DescriptorBinding::bufferAtomic), staged in device
    // memory whatever its size (see AllowDeviceStaging).
    void AddWritable(std::uint64_t address, std::size_t bytes, bool atomic = false, bool swept = false);
    void AddReadable(std::uint64_t address, std::size_t bytes);
    // Device-local staging of written and atomic elements inside host imports (see
    // GuestBufferMemory.cpp): allowed only for a build whose every use records its work and then
    // calls RecordCopyBacks (a dispatch), since a staged region's results reach guest memory by
    // that copy alone. Call before Upload.
    void AllowDeviceStaging() { stagingAllowed = true; }
    // An address-based build whose dispatch is too small for its BDA overlays' copies to pay off
    // (see resolveOverlays): its written elements stay in the imports.
    void DisallowAddressStaging() { addressStaging = false; }
    void AllowAdjustedRegions() { adjustedRegions = true; }
    // S1: UploadPrepare copies nothing from guest memory (a region it would copy waits for
    // UploadFinish), so a stage A made ahead of the draw reads no guest bytes.
    void DeferHostCopies() { deferHostCopies = true; }
    // Records the copy-in of every staged region anew for another use of this upload (a resource
    // cache hit, from ShaderResources::Revalidate, under GuestMemory::GpuMutex, once the imports
    // were confirmed unchanged): the previous use's copy-back left the shadow behind, and the next
    // RecordCopyBacks copies it back again. Nothing to do without staged regions.
    void RecordStagingCopies(Recorder& recorder);
    void AddSnapshot(const GuestMemorySnapshot& snapshot);
    // Upload is the two stages below back to back. UploadPrepare needs no device lock: it merges the
    // regions, binds the image mirrors and host imports that already serve them (an import pointer is
    // re-checked against the registry epoch later) and copies read-only regions. UploadFinish runs
    // under GuestMemory::GpuMutex: it reconciles and makes imports (retiring one hands its buffer to
    // the recorder), refreshes writable mirrors and flushes pending results (both may wait for
    // recorded work) and copies the regions a descriptor may write, so the window between that copy
    // and the recorder's pending-write note stays closed.
    void Upload(bool addressable);
    void UploadPrepare(bool addressable);
    void UploadFinish(bool addressable);
    VkDescriptorBufferInfo Descriptor(std::uint64_t address, std::size_t bytes, std::uint32_t& adjustment) const;
    static std::uint64_t ViewBytes(std::uint64_t bytes, std::uint32_t adjustment);
    std::vector<ShaderRecompiler::BdaAbi::Range> AddressRanges() const;
    // The BDA table of the cached address space when it serves this upload alone (an address-based
    // build with no region outside it): its ranges, immutable while the space lives, and the
    // space's serial, by which BdaResources reuses the table buffer built for it without a
    // compare. Nothing when a region of this build is not in the space (AddressRanges then merges).
    struct CachedTable {
        std::uint64_t serial;
        const std::vector<ShaderRecompiler::BdaAbi::Range>* ranges;
        // Shares a merged list (tableRanges), so BdaResources' table cache can keep it without a copy
        // (APS5_NO_LEAN_BDA_TABLES=1 copies, as before). Null for the space's own list: a reference to it would
        // keep the whole AddressSpace alive, and with it the lease pinning its guest allocations (best137: a
        // freed allocation waited 60 s for that pin and aborted); the space's list is cached about once per space
        // and is copied.
        std::shared_ptr<const std::vector<ShaderRecompiler::BdaAbi::Range>> owner;
    };
    std::optional<CachedTable> CachedAddressTable() const;
    // The registered memory an address-based build maps, shared by consecutive builds (see
    // GuestBufferMemory.cpp): its lease and the sorted, immutable regions of the ranges served in
    // place by imports and mirrors. Opaque outside GuestBufferMemory.cpp.
    struct AddressSpace;
    void WriteBack();
    bool WritesOverlap(std::uint64_t address, std::size_t bytes) const;
    bool CopiedWritesOverlap(std::uint64_t address, std::size_t bytes) const;
    // Guest ranges the shader may write through descriptors, as [begin, end).
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& Writes() const { return writes; }
    // Whether any written range lives in a copied buffer, so a write-back must run once the GPU is done.
    bool HasCopiedWrites() const;
    // Reports writes into host-imported memory (made by the GPU in place, or copied back into it by
    // RecordCopyBacks) to the write tracking now.
    void MarkDirectWrites() const;
    // Records, into the recorder's open batch, the copy of every written sub-range of a region the
    // GPU copied out of a host import (see Region::gpuCopy) back into the import: what the shader
    // wrote lands in guest memory by the GPU, ordered after the recorded work, so the region needs
    // no CPU write-back (HasCopiedWrites no longer counts it) and the caller notes the ranges as
    // pending writes and marks them as MarkDirectWrites does. Called right after the work using the
    // regions was recorded (ShaderResources::MarkGpuWrites), under GuestMemory::GpuMutex. A use that
    // never calls it (a synchronous draw) stores the staging bytes from the CPU in WriteBack.
    void RecordCopyBacks(Recorder& recorder);
    // GPU-sourced buffers (opt-in: APS5_GPU_SOURCED_DRAW_BUFFERS=1, APS5_IMPORT_SOURCED_COPY_IN=1
    // for the copy-in side; APS5_AB switches). A staged region whose copy-in found no pending writer
    // and whose copy-back marks found nothing stored over it since the copy-in (MarkOwnWrites) is
    // registered with its device-local shadow: until a later store (CPU or driver, any stamp after
    // the marks, a CPU stamp after the copy-in) or the shadow's next copy-in, the shadow holds exactly
    // the import's bytes. FindGpuSource answers a range inside one such region on the same lane once
    // the batch that wrote it completed; a draw binds the shadow instead of copying the bytes on the
    // CPU, a copy-in copies from it instead of a CPU snapshot of the import. Nullopt and `miss` say
    // why not.
    enum class GpuSourceMiss : std::uint8_t { None, NoSource, Partial, Released, Pending, Lane, Align, Registry, Stored, Cpu, Count };
    struct GpuSource {
        std::shared_ptr<Buffer> buffer;
        VkDeviceSize offset = 0;
        std::uint64_t since = 0;
    };
    static std::optional<GpuSource> FindGpuSource(std::uint64_t address, std::size_t bytes, VkDeviceSize alignment, GpuSourceMiss& miss);
    static const char* GpuSourceMissName(GpuSourceMiss miss);
    static bool GpuSourcedDrawBuffers();
    static bool ImportSourcedCopyIn();
    // APS5_PROFILE_DRAW: a draw buffer bound GPU-side or copied (bytes), a miss by reason, and the
    // [gpu-sourced] line every 10 s. APS5_VERIFY_GPU_SOURCED=<n>: the first n GPU-side binds per 10 s
    // also copy the import on the CPU and compare it with the bound shadow once the draw's batch ran.
    static void CountDrawBuffer(bool gpuSide, std::size_t bytes, GpuSourceMiss miss);
    static bool VerifyGpuSourceTurn();
    static void CountGpuSourceVerify(bool same);
    static void ReportGpuSources();
    static std::size_t GpuSourceEntries();
    static std::uint64_t GpuSourcedCopyIns();
    static std::array<std::uint64_t, 2> GpuSourceVerifyCounts();
    // Whether a region is copied out of an import by the GPU (RecordCopyBacks has work).
    bool HasGpuCopies() const {
        for (const auto& region : regions) if (region.gpuCopy) return true;
        return false;
    }
    // Whether registered allocations are pinned until write-back (address-based shaders): by this
    // build's own lease, or by the cached address space it holds.
    bool HoldsLease() const { return !lease.empty() || space != nullptr; }
    std::size_t CopiedBytes() const {
        std::size_t bytes = 0;
        for (const auto& region : regions) if (region.buffer != nullptr) bytes += static_cast<std::size_t>((region.bufferEnd != 0 ? region.bufferEnd : region.end) - region.begin);
        return bytes;
    }
    // Every uploaded region as [begin, end) when all of them are served by host imports, in place or
    // through a device-local staging copy of the import (nothing was copied through the CPU, so the
    // upload can serve a later identical build), else nothing.
    std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>> DirectRegions() const;
    struct CutRegion {
        std::uint64_t begin;
        std::uint64_t end;
        std::uint64_t committedEnd;
    };
    std::vector<CutRegion> CutRegions() const;
    // Every region the recorded work reads in place through a host import, as [begin, end): the
    // regions bound in place (`direct`), an address-based build's leased heaps included. A region
    // the GPU copies out of an import (gpuCopy) notes its read itself when the copy is recorded.
    // For the recorder's read tracking (ShaderResources::MarkGpuWrites); nothing once committed.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> InPlaceReads() const;
    void InPlaceReads(std::vector<std::pair<std::uint64_t, std::uint64_t>>& out) const;
    struct SplitReads {
        std::shared_ptr<const std::vector<std::pair<std::uint64_t, std::uint64_t>>> space;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> own;
    };
    std::optional<SplitReads> SplitInPlaceReads() const;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> DeviceReads() const;
    void DeviceReads(std::vector<std::pair<std::uint64_t, std::uint64_t>>& out) const;
    std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>> SpaceRegions() const;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> DirectWrites() const;
    // APS5_PROFILE_DRAW (Recorder::NoteRunAccess): {begin, end, flags} of the regions read in place
    // or copied out of their imports and of the written ranges.
    void RunAccesses(std::vector<std::array<std::uint64_t, 3>>& out) const;
    bool SameSnapshots(std::span<const GuestMemorySnapshot> snapshots) const;
    // Address-table templates (APS5_NO_BDA_TEMPLATES unset): whether a build of `snapshots` over the
    // current (resumed) address space would add no region of its own beyond this upload's: every
    // snapshot lies inside a live region (registered memory, an import, a mirror) and agrees with
    // it, or inside one of this upload's snapshot copies with the same bytes, and every snapshot copy
    // of this upload is captured again at the same address and size. A live region serves the GPU in
    // place, so a snapshot inside one may change between uses without a rebuild.
    bool SnapshotsServed(std::span<const GuestMemorySnapshot> snapshots) const;
    void SuspendSpace() noexcept;
    bool ResumeSpace();
    bool SpaceImportsCurrent();
    std::shared_ptr<const void> SpaceHold() const { return space; }

private:
    struct SnapshotUse {
        std::uint64_t address;
        std::size_t bytes;
        bool live;
    };
    struct Region {
        std::uint64_t begin;
        std::uint64_t end;
        bool writable;
        std::vector<std::byte> snapshot;
        // Shared so a recorded GPU copy into it (see gpuCopy) can keep it in the batch itself, before
        // the caller keeps the whole resources: a throw between the two would otherwise return it to
        // the pool under an unsubmitted copy command.
        std::shared_ptr<Buffer> buffer;
        // Guest bytes as uploaded; write-back only stores bytes the GPU changed.
        std::vector<std::byte> uploaded {};
        // Registered allocation that is imported: its bytes are read from live guest memory, not a snapshot.
        bool hostBacked = false;
        // Set when the region is served by an imported allocation; nothing is copied or written back.
        const HostImport* direct = nullptr;
        // Set when the region covers uncommitted pages: only the `backed` parts (possibly none) are guest
        // memory; the rest reads as zeros and is never stored.
        bool sparse = false;
        bool sparseHeap = false;
        bool runCut = false;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> backed {};
        std::uint64_t lastView = 0;
        std::uint64_t bufferEnd = 0;
        // Set when the region is served by an image mirror: nothing is copied; writable mirrors are
        // written back by comparing with the mirror's shadow. Kept alive here for recorded work.
        std::shared_ptr<ImageMirror> mirror {};
        // The mirror is a descriptor sub-range one found by UploadPrepare (not a lease mirror), which
        // UploadFinish confirms still has its Range before the region binds it.
        bool subrangeMirror = false;
        // Set by UploadPrepare when UploadFinish still has device-lock work for the region: a copy a
        // descriptor writes, a writable mirror's refresh, an import to reconcile or make, or the
        // pending-results flush of an import it took.
        bool pending = false;
        // The region lies inside a host import but is misaligned for binding in place: `buffer` is
        // filled by a vkCmdCopyBuffer from the import recorded into the batch (no CPU read, so no
        // flush-hook wait), from `copySource` (the import's VkBuffer, alive until the batch
        // completed, with the guest address of its first byte). `copiedBack` once RecordCopyBacks
        // recorded the written sub-ranges' copies back into the import.
        bool gpuCopy = false;
        VkBuffer copySource = VK_NULL_HANDLE;
        std::uint64_t copySourceBase = 0;
        // The import's device address (0 when it has none): APS5_STAGING_DIFF copy-backs store into it.
        VkDeviceAddress copySourceAddress = 0;
        bool copiedBack = false;
        // APS5_STAGING_DIFF=1: a device-local copy of a staged shadow's bytes right after its copy-in,
        // so the copy-back stores only the dwords the work changed (RecordCopyBacks).
        std::shared_ptr<Buffer> pristine {};
        // This use's copy-in recorded the pristine copy (APS5_STAGING_DIFF=alt may skip it).
        bool diffUse = false;
        // An element the shader updates atomically lies inside (AddWritable's `atomic`).
        bool atomic = false;
        bool swept = false;
        // The gpuCopy buffer is a device-local staging shadow (see stagingEligible): no host
        // mapping, so nothing is ever stored from it by the CPU, and the region is taken even when
        // the import could bind it in place.
        bool deviceLocal = false;
        // The tracker generation and registry generation at this use's copy-in when it may become a
        // GPU source (FindGpuSource), else 0; consumed by MarkDirectWrites.
        mutable std::uint64_t gpuSourceSince = 0;
        mutable std::uint64_t gpuSourceRegistry = 0;
        // Outside the write watch: the owned words armed at the copy-in (a CPU store moves them).
        mutable std::vector<std::uint64_t> gpuSourceWords {};
        // What the last registered source learned (bit 0 a draw or copy-in used it, bit 1 a CPU
        // store moved its words), and the back-off of uses that do not try (2^n - 1 skipped).
        mutable std::shared_ptr<std::atomic<std::uint8_t>> gpuSourceFeedback {};
        mutable std::uint8_t gpuSourceBackoff = 0;
        mutable std::uint16_t gpuSourceSkip = 0;
        // The device refused the shadow (memory or allocations exhausted): the region takes the
        // path it would take without staging, in both upload stages.
        bool unstaged = false;
        // A written descriptor range of an address-based build that lies inside one of the space's
        // import regions, given a region of its own so it can be staged device-local (see
        // resolveOverlays); the BDA table sends its guest addresses to this region's buffer instead
        // of the import (AddressRanges). APS5_NO_BDA_STAGING=1 serves it from the import as before.
        bool overlay = false;
    };

    // How [begin, end) lies against the space's base regions.
    enum class BaseOverlap { None, Inside, Partial };
    BaseOverlap baseOverlap(std::uint64_t begin, std::uint64_t end, const Region** owner) const;
    // Copies the space's base regions into this build's own regions (today's per-build form) and
    // drops the space: for a region that partially overlaps a base region, and for a build whose
    // imports changed under the space (`resolve`: the direct regions are re-resolved by UploadFinish).
    void dissolveSpace(bool resolve);
    void completeSpaceWrites() const;
    // The region containing `address`: this build's own regions (sorted after UploadPrepare) or
    // the space's base regions, or null.
    const Region* owner(std::uint64_t address, std::uint64_t end) const;
    // AddressRanges' last two passes (shared with incrementalAddressRanges): the overlays cut into the
    // ranges they lie in, then the runtime writes split out of read-only ranges as writable parts.
    std::vector<ShaderRecompiler::BdaAbi::Range> applyOverlaysAndWrites(std::vector<ShaderRecompiler::BdaAbi::Range> result, std::vector<ShaderRecompiler::BdaAbi::Range> overlays, std::span<const std::pair<std::uint64_t, std::uint64_t>> writtenRanges) const;
    // AddressRanges() for a build over the cached space, redoing only the neighbourhood of its own
    // regions, overlays and writes (see GuestBufferMemory.cpp).
    void incrementalAddressRanges(std::vector<ShaderRecompiler::BdaAbi::Range>& result) const;
    // Builds this build's regions from a lease as the per-build path always did: `base` gets the
    // ranges served in place (imports, mirrors), `copied` the ones copied per build.
    struct CopiedRange {
        std::uint64_t begin;
        std::uint64_t end;
        bool writable;
    };
    void addCopiedRange(const CopiedRange& range, const GuestMemory::Commitment* known = nullptr);
    // The BDA table entry of an uploaded region.
    static ShaderRecompiler::BdaAbi::Range addressRange(const Region& region);
    static std::uint64_t servedEnd(const Region& region);
    static std::uint64_t inPlaceEnd(const Region& region);
    void validate(std::uint64_t address, std::size_t bytes) const;
    // The region of a descriptor-bound range (AddWritable/AddReadable), committed pages only. A
    // range inside a base region of the space adds nothing: the region serves it, as today's merge
    // of the two did.
    void addDescriptorRegion(std::uint64_t address, std::size_t bytes, bool atomic, bool swept, bool readOnly);
    static bool servesView(const Region& region, std::uint64_t view);
    static bool sparseHeapCandidate(const Region& region);
    bool bindSparseMirror(Region& region, const GuestAllocations::Lease& lease);
    // Gives a region a buffer of its own with its bytes (guest memory for host-backed and writable
    // ranges, plus the write-back's reference copy for a range a descriptor writes; else its snapshot).
    void copyRegion(Region& region, bool addressable);
    // Whether a region inside a host import that cannot be bound in place is copied by the GPU
    // instead of the CPU (see Region::gpuCopy): live guest bytes, not sparse, at most the size
    // APS5_GPU_COPY_MAX_KIB allows, and APS5_CPU_COPIES unset.
    bool gpuCopyEligible(const Region& region) const;
    // Whether a region inside a host import is staged in device memory instead of bound in place
    // (see GuestBufferMemory.cpp): staging allowed, not address-based unless an overlay region, a
    // written element inside, and an atomic element or a size within the written-shadow window.
    // Independent of the import, so UploadPrepare and UploadFinish decide alike.
    bool stagingEligible(const Region& region, bool addressable) const;
    // Whether a written descriptor range inside the space's base region `inside` becomes an
    // overlay region (Region::overlay), and the UploadPrepare pass that keeps only the overlay
    // clusters that can be staged exactly (one import region, within the staging window, no other
    // descriptor range or own region crossing them).
    bool overlayCandidate(const Region& inside, bool readOnly) const;
    void resolveOverlays();
    bool bindableInPlace(std::uint64_t offset, bool addressable) const;
    // Records the import-to-buffer copies of the given gpuCopy regions into the open batch, with
    // the barriers that order them after earlier recorded writes and before the shaders reading them.
    void recordGpuCopies(std::span<Region* const> copies, bool addressable);
    void takeHeapReferences();
    Context context;
    bool stagingAllowed = false;
    bool addressStaging = true;
    bool bdaStaging = true;
    bool adjustedRegions = false;
    GuestAllocations::Lease lease;
    // The cached address space this build maps through (its lease pins the ranges); `regions` then
    // holds only the regions outside it (V#s, snapshots, ranges copied per build).
    std::shared_ptr<const AddressSpace> space;
    mutable std::uint64_t tableSerial = 0;
    mutable std::shared_ptr<const std::vector<ShaderRecompiler::BdaAbi::Range>> tableRanges;
    // Import registry epoch when `direct` pointers were taken at acquire time; they are reused while
    // no import was destroyed since.
    std::uint64_t importsEpoch = 0;
    std::uint64_t importFloor = 0;
    std::vector<Region> regions;
    // Whether `regions` is in ascending address order (true right after AcquireRegistered, whose
    // regions follow the registry's order), so AddSnapshot can search instead of scanning.
    bool regionsSorted = false;
    bool regionsOverlap = false;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> readsOverBase;
    // Descriptor ranges inside the space's base regions that its regions serve in place while
    // overlays are possible (see resolveOverlays): an overlay crossing one is not staged.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> servedInsideBase;
    std::vector<std::pair<std::uint64_t, std::vector<std::byte>>> heapReferences;
    std::vector<SnapshotUse> snapshotUses;
    std::uint64_t suspendedSpace = 0;
    std::vector<std::pair<std::uint64_t, std::shared_ptr<Buffer>>> heapReadbacks;
    // UploadPrepare ran (regions are frozen); `uploaded` once UploadFinish ran.
    bool prepared = false;
    bool uploaded = false;
    bool deferHostCopies = false;
    bool committed = false;
    mutable std::uint64_t directWritesGeneration = 0;
};

}

#endif
