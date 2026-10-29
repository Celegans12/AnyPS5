#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCY_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

constexpr std::uint32_t ResidencyPageShift = 16;
constexpr std::uint64_t ResidencyPageBytes = std::uint64_t{1} << ResidencyPageShift;
constexpr std::uint32_t ResidencySubpageShift = 12;
constexpr std::uint32_t ResidencySubpageBytes = 1u << ResidencySubpageShift;
constexpr std::uint32_t ResidencySubpages = 1u << (ResidencyPageShift - ResidencySubpageShift);
constexpr std::size_t ResidencyStreams = 16;
constexpr std::uint64_t ResidencyUnchecked = ~std::uint64_t{0};
constexpr std::uint64_t ResidencyMeasuredBudget = ~std::uint64_t{0};
constexpr std::uint32_t ResidencyHotQuietFrames = 4;
constexpr std::size_t ResidencyHotGapBuckets = 8;

enum class ResidencyRefusal : std::uint8_t { None, UnsettledImport, PrivateHostWrite, PrivatePinned, ResidentView, SharedPinned, AliasHostWrite, Unavailable };
constexpr std::size_t ResidencyRefusals = 8;
constexpr std::uint32_t ResidencyRefusalBit(ResidencyRefusal refusal) { return 1u << static_cast<std::uint32_t>(refusal); }

using ResidencyToken = std::array<std::uint64_t, ResidencyStreams>;
using ResidencyMask = std::uint16_t;
static_assert(sizeof(ResidencyMask) * 8 == ResidencySubpages);

constexpr ResidencyMask ResidencyWholePage = static_cast<ResidencyMask>(~ResidencyMask{0});

ResidencyToken MergeTokens(const ResidencyToken& first, const ResidencyToken& second);
bool TokenEmpty(const ResidencyToken& token);
ResidencyMask SubpageMask(std::uint64_t page, std::uint64_t address, std::uint64_t bytes);

enum class ResidencyAccess : std::uint8_t { Read, Write };
struct ResidencyImport {
    bool whole = false;
    bool cpuWritable = false;
    bool gpuWritable = false;
    bool aliased = false;
    std::uint64_t section = 0;
    std::uint64_t offset = 0;
    std::uint64_t serial = 0;
    bool resident = false;
};

struct ResidencyWrites {
    ResidencyMask cpu = ResidencyWholePage;
    bool other = true;
    std::uint64_t generation = 0;
};

struct ResidencyRange {
    std::uint64_t address = 0;
    std::uint64_t bytes = 0;
};

struct ResidencyReuseStamp {
    std::uint64_t writes = 0;
    std::uint64_t collect = 0;
    std::uint64_t imports = 0;
    std::uint64_t hostImports = 0;
    std::uint64_t batch = 0;
    bool operator==(const ResidencyReuseStamp&) const = default;
};

struct ResidencyPolicy {
    std::uint32_t chunkSlots = 1024;
    std::uint64_t budgetBytes = 0;
    std::uint32_t hotFaults = 2;
    std::uint32_t hotFrames = 2;
    std::uint32_t quietFrames = ResidencyHotQuietFrames;
    bool deferCopyIn = false;
};

struct MirrorSlot {
    std::uint32_t chunk = 0;
    std::uint32_t slot = 0;
    bool operator==(const MirrorSlot&) const = default;
};

struct ResidencyCopy {
    std::uint64_t page = 0;
    MirrorSlot mirror;
    std::uint32_t offset = 0;
    std::uint32_t bytes = 0;
};

struct ResidencyMirrorStore {
    MirrorSlot mirror;
    std::uint32_t offset = 0;
    std::uint32_t source = 0;
    std::uint32_t bytes = 0;
};

class IResidencyOwner {
public:
    virtual ~IResidencyOwner() = default;
    virtual void Retile(std::span<const ResidencyRange> ranges) = 0;
    virtual std::shared_ptr<IResidencyOwner> Hold() { return {std::shared_ptr<IResidencyOwner>{}, this}; }
};

class IResidencyDevice {
public:
    virtual ~IResidencyDevice() = default;
    virtual bool Reached(const ResidencyToken& token) = 0;
    virtual void Wait(const ResidencyToken& token) = 0;
    virtual bool WaitUnlocked(const ResidencyToken& token) {
        Wait(token);
        return true;
    }
    virtual bool Signaled(const ResidencyToken& token) {
        static_cast<void>(token);
        return false;
    }
    virtual ResidencyToken Newest() = 0;
    virtual ResidencyToken Current() = 0;
    virtual ResidencyImport Import(std::uint64_t page) = 0;
    virtual std::vector<ResidencyRange> Aliases(std::uint64_t address, std::uint64_t bytes) = 0;
    virtual std::uint64_t Generation() = 0;
    virtual void ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) = 0;
    virtual std::uint64_t AllocateChunk(std::uint32_t chunk, std::uint32_t slots) = 0;
    virtual void ReleaseChunk(std::uint32_t chunk) = 0;
    virtual ResidencyToken CopyIn(std::span<const ResidencyCopy> copies) = 0;
    virtual ResidencyToken WriteBack(std::span<const ResidencyCopy> copies) = 0;
    virtual ResidencyRefusal Guard(std::uint64_t page) = 0;
    virtual void Unguard(std::span<const std::uint64_t> pages) = 0;
    virtual ResidencyRefusal GuardAccess(std::uint64_t page, ResidencyMask mask) = 0;
    virtual void UnguardAccess(std::uint64_t page) = 0;
    virtual std::vector<ResidencyRange> WholeImports(std::uint64_t& generation) {
        static_cast<void>(generation);
        return {};
    }
    virtual void SettleWrites(std::uint64_t page) {
        static_cast<void>(page);
    }
    virtual bool ImportSettled(std::uint64_t page) {
        static_cast<void>(page);
        return true;
    }
    virtual std::uint64_t ImportStamp() { return 0; }
    virtual void SettleImport(std::uint64_t page, std::uint64_t stamp) {
        static_cast<void>(page);
        static_cast<void>(stamp);
    }
    virtual std::uint64_t Budget() { return ~std::uint64_t{0}; }
    virtual std::uint64_t OtherDeviceBytes() { return 0; }
    virtual std::uint64_t GuestBytes() { return 0; }
    virtual void CollectWrites(std::uint64_t firstPage, std::uint64_t pages) {
        static_cast<void>(firstPage);
        static_cast<void>(pages);
    }
    virtual ResidencyWrites PageWrites(std::uint64_t page, std::uint64_t generation) {
        static_cast<void>(page);
        static_cast<void>(generation);
        return {};
    }
    virtual std::optional<ResidencyReuseStamp> ReuseStamp() { return std::nullopt; }
};

struct ResidencyCounts {
    std::uint64_t admitted = 0;
    std::uint64_t cpuFillsBesideDirty = 0;
    std::uint64_t deferred = 0;
    std::uint64_t inadmissible = 0;
    std::uint64_t evicted = 0;
    std::uint64_t demoted = 0;
    std::uint64_t unusedDemoted = 0;
    std::uint64_t dropped = 0;
    std::uint64_t refreshed = 0;
    std::uint64_t faults = 0;
    std::uint64_t hot = 0;
    std::uint64_t cooled = 0;
    std::uint64_t unguardable = 0;
    std::uint64_t aliasSkipped = 0;
    std::uint64_t residentSkipped = 0;
    std::uint64_t conflicts = 0;
    std::uint64_t waits = 0;
    std::uint64_t copiedInBytes = 0;
    std::uint64_t cpuWriteBacks = 0;
    std::uint64_t accessUnguardable = 0;
    std::uint32_t accessGuarded = 0;
    std::uint32_t hotResident = 0;
    std::uint64_t writtenBackBytes = 0;
    std::uint64_t touchAdmitted = 0;
    std::uint64_t touchDeferred = 0;
    std::uint64_t importsSettled = 0;
    std::uint64_t unsettledGuards = 0;
    std::uint64_t aliasDemoted = 0;
    std::uint64_t recordRefreshed = 0;
    std::uint64_t hotRefreshed = 0;
    std::uint64_t hotSweeps = 0;
    std::uint64_t hotCollects = 0;
    std::uint64_t hotServed = 0;
    std::array<std::uint64_t, ResidencyHotGapBuckets> hotWriteGaps{};
    std::array<std::uint64_t, ResidencyRefusals> refusals{};
    std::uint64_t refusalRetries = 0;
    std::uint32_t refusedPages = 0;
    std::uint64_t retiles = 0;
    std::uint64_t retiledBytes = 0;
    std::uint64_t eagerRetiles = 0;
    std::uint32_t ownedPages = 0;
    std::uint32_t stalePages = 0;
    std::uint64_t chunksAllocated = 0;
    std::uint64_t chunksReleased = 0;
    std::uint32_t chunks = 0;
    std::uint32_t resident = 0;
    std::uint32_t dirty = 0;
    std::uint32_t retiring = 0;
    std::uint32_t free = 0;
    std::uint64_t mixedRanges = 0;
    std::uint64_t completedRanges = 0;
    std::uint64_t demotedRanges = 0;
    std::uint64_t rangeConflicts = 0;
    std::uint64_t rangeSetsUsed = 0;
    std::uint64_t rangeSetsReused = 0;
    std::uint64_t copyInDeferredBytes = 0;
    std::uint64_t behindFlips = 0;
    std::uint64_t caughtUp = 0;
    std::uint32_t behind = 0;
};

enum class ResidencyRangeState : std::uint8_t { Uniform, Completed, Demoted, Conflict };

class ResidencyMutex {
public:
    void lock();
    void unlock();
    bool HeldByThisThread() const;
    std::uint32_t ReleaseAll();
    void Reacquire(std::uint32_t depth);

private:
    std::mutex mutex;
    std::atomic<const void*> owner{nullptr};
    std::uint32_t depth = 0;
};

class ResidencyManager {
public:
    class CpuReadExemption {
    public:
        explicit CpuReadExemption(const IResidencyOwner* owner);
        ~CpuReadExemption();
        CpuReadExemption(const CpuReadExemption&) = delete;
        CpuReadExemption& operator=(const CpuReadExemption&) = delete;

    private:
        const IResidencyOwner* previous;
    };

    ResidencyManager(IResidencyDevice& device, const ResidencyPolicy& policy);
    ResidencyManager(const ResidencyManager&) = delete;
    ResidencyManager& operator=(const ResidencyManager&) = delete;
    ~ResidencyManager();

    void SetBudget(std::uint64_t bytes);
    void ApplyBudget();
    void AdmitTouched(std::span<const std::uint64_t> pages);
    bool Advance();
    void NextFrame();
    std::uint64_t Frame() const;
    void Apply();
    void AdmitEager();
    void Prepare();
    void Use(std::uint64_t address, std::uint64_t bytes, ResidencyAccess access);
    ResidencyRangeState UseRange(std::uint64_t address, std::uint64_t bytes, std::uint64_t largest);
    void UseRanges(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, std::uint64_t largest);
    void Pin(std::uint64_t address, std::uint64_t bytes, ResidencyAccess access, const ResidencyToken& token);
    void Demote(std::uint64_t address, std::uint64_t bytes);
    void DemoteWritable();
    void NoteUnboundedWriter(const ResidencyToken& token);
    void DropRange(std::uint64_t address, std::uint64_t bytes);
    void NoteImportWritten(std::uint64_t address, std::uint64_t bytes);
    std::vector<ResidencyMirrorStore> StoreThroughMirrors(std::uint64_t address, std::uint64_t bytes, const ResidencyToken& lands);
    void RefreshImportWritten(std::uint64_t address, std::uint64_t bytes);
    void QueueCpuWritten(std::span<const std::uint64_t> pages);
    void QueueRefusalCleared(std::uint64_t address, std::uint64_t bytes, std::uint32_t cleared);
    std::optional<ResidencyRefusal> Refused(std::uint64_t page) const;
    void NoteGpuWritten(std::uint64_t address, std::uint64_t bytes, bool written = false);
    void QueueGpuWritten(std::uint64_t address, std::uint64_t bytes);
    bool AnyDirty() const { return dirtyPages.load(std::memory_order_relaxed) != 0; }
    void PrepareImportWrite(std::uint64_t address, std::uint64_t bytes);
    void RefreshHot();
    void FlushForCpuRead(std::uint64_t address, std::uint64_t bytes);
    bool FlushForCpuReadUnlocked(std::uint64_t address, std::uint64_t bytes);
    bool FlushForCpuReadSettled(std::uint64_t address, std::uint64_t bytes);
    ResidencyMutex& Mutex() const { return structure; }
    void WriteBackAll();
    void Own(std::uint64_t address, std::uint64_t bytes, IResidencyOwner* owner);
    void MarkStale(std::uint64_t address, std::uint64_t bytes, IResidencyOwner* owner);
    bool RetileOwned(std::uint64_t address, std::uint64_t bytes, std::span<const IResidencyOwner* const> except = {});
    bool RetileAllOwned(std::span<const IResidencyOwner* const> except = {});
    bool RetileUnguarded();
    void Disown(IResidencyOwner* owner);
    void Forget(IResidencyOwner* owner);
    void ClearStale(std::uint64_t address, std::uint64_t bytes, const IResidencyOwner* owner);
    IResidencyOwner* OwnerOf(std::uint64_t page) const;
    ResidencyMask Stale(std::uint64_t page) const;
    std::vector<std::shared_ptr<IResidencyOwner>> HoldOwners(std::uint64_t address, std::uint64_t bytes) const;
    std::vector<std::shared_ptr<IResidencyOwner>> HoldStaleOwners() const;
    bool StaleOfOthers(std::uint64_t address, std::uint64_t bytes, const IResidencyOwner* owner) const;
    bool GuardedStaleIn(std::uint64_t address, std::uint64_t bytes, const IResidencyOwner* owner) const;
    std::uint64_t OwnershipSerial() const { return ownershipSerial.load(std::memory_order_acquire); }
    bool AnyStale() const { return staleCount.load(std::memory_order_acquire) != 0; }
    bool DirtyIn(std::uint64_t address, std::uint64_t bytes) const;
    bool StaleIn(std::uint64_t address, std::uint64_t bytes) const;
    bool CpuReadSettled(std::uint64_t address, std::uint64_t bytes) const;
    void Retire();

    std::optional<MirrorSlot> Slot(std::uint64_t page);
    std::optional<MirrorSlot> WriteSlot(std::uint64_t page);
    bool Used(std::uint64_t page);
    std::optional<std::uint64_t> Translate(std::uint64_t page);
    ResidencyMask Dirty(std::uint64_t page);
    bool AnyResident(std::uint64_t address, std::uint64_t bytes);
    bool AnyStaleOwned() const;
    std::vector<ResidencyRange> ResidentRanges();
    std::vector<std::uint64_t> TakeChanges();
    bool ChangesSince(std::uint64_t serial, std::uint64_t& now, std::vector<std::pair<std::uint64_t, std::uint64_t>>& changed);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> Mirrored(std::uint64_t& now);
    std::vector<std::uint32_t> Chunks() const;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ResidentGenerations() const;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> MirroredIn(std::uint64_t address, std::uint64_t bytes);
    std::vector<std::uint64_t> CurrentPages() const;
    void CatchUp(std::uint64_t address, std::uint64_t bytes);
    ResidencyMask Behind(std::uint64_t page) const;
    std::uint64_t ChunkAddress(std::uint32_t chunk) const;
    std::uint32_t ChunkSlots() const { return chunkSlots; }
    std::uint64_t Serial() const { return serial; }
    std::uint64_t Epoch() const { return epoch.load(std::memory_order_relaxed); }
    bool Hot(std::uint64_t page) const;
    bool HotResident(std::uint64_t page) const;
    bool ImportSettled(std::uint64_t page) const;
    std::uint64_t BudgetBytes() {
        const std::lock_guard lock(structure);
        return budgetBytes();
    }
    bool FullResidency() {
        const std::lock_guard lock(structure);
        return fullResidency();
    }
    ResidencyCounts Counts() const;

private:
    static constexpr std::uint32_t None = ~std::uint32_t{0};
    static constexpr std::uint32_t DirtyLeafShift = 16;
    static constexpr std::size_t DirtyLeaves = std::size_t{1} << (47 - ResidencyPageShift - DirtyLeafShift);

    struct DirtyLeaf {
        std::array<std::atomic<ResidencyMask>, std::size_t{1} << DirtyLeafShift> masks{};
        std::array<std::atomic<ResidencyMask>, std::size_t{1} << DirtyLeafShift> stale{};
        std::array<std::atomic<std::uint8_t>, std::size_t{1} << DirtyLeafShift> unsettled{};
    };
    static constexpr std::uint8_t UnsettledWriteBack = 1u;
    static constexpr std::uint8_t UnsettledWriter = 2u;
    static constexpr std::uint8_t UnsettledUnbounded = 4u;
    static constexpr std::uint32_t WholeEnd = static_cast<std::uint32_t>(ResidencyPageBytes);

    using Run = std::pair<std::uint32_t, std::uint32_t>;

    struct Page {
        std::uint64_t index = 0;
        std::uint64_t generation = 0;
        MirrorSlot slot;
        std::uint32_t older = None;
        std::uint32_t newer = None;
        std::uint64_t faultFrame = 0;
        std::uint64_t faultHistory = 0;
        ResidencyMask dirty = 0;
        ResidencyMask behind = 0;
        std::vector<Run> dirtyRuns;
        bool gpuWritable = false;
        bool cpuWritable = false;
        bool shared = false;
        bool importSettled = false;
        bool guarded = false;
        bool used = false;
        bool accessGuarded = false;
        bool settleIssued = false;
        ResidencyMask accessMask = 0;
        std::uint64_t importSerial = 0;
        std::uint64_t touchFrame = 0;
        std::uint64_t unboundedSerial = 0;
        ResidencyToken pin{};
        ResidencyToken settle{};
        std::uint64_t importStamp = 0;
    };

    struct Cooling {
        std::uint64_t generation = 0;
        std::uint64_t frame = 0;
    };

    struct Chunk {
        std::uint64_t address = 0;
        std::vector<std::uint64_t> freeBits;
        std::uint32_t freeSlots = 0;
        bool live = false;
        bool draining = false;
    };

    struct Retiring {
        MirrorSlot slot;
        ResidencyToken token{};
    };

    struct Fill {
        std::uint32_t id = 0;
        std::uint32_t begin = 0;
        std::uint32_t end = WholeEnd;
        bool cpu = false;
    };

    struct UnboundedWriter {
        std::uint64_t serial = 0;
        ResidencyToken token{};
    };
    struct WriteBacks {
        std::vector<ResidencyCopy> copies;
        std::vector<Retiring> slots;
        std::vector<std::uint64_t> pages;
        std::vector<std::uint64_t> unguard;
    };

    bool settled(const ResidencyToken& token);
    void applyDrops();
    void applyQueued();
    std::vector<std::uint32_t> residentIn(std::uint64_t address, std::uint64_t bytes) const;
    bool anyResident(std::uint64_t first, std::uint64_t last) const;
    std::vector<Fill> withAliases(std::uint64_t address, std::uint64_t bytes);
    std::optional<std::uint32_t> find(std::uint64_t page) const;
    std::vector<std::uint64_t> indicesOf(std::span<const std::uint32_t> ids) const;
    std::vector<std::uint32_t> stillResident(std::span<const std::uint32_t> ids, std::span<const std::uint64_t> indices) const;
    std::vector<std::uint64_t> indicesOf(std::span<const Fill> fills) const;
    std::vector<Fill> stillResident(std::span<const Fill> fills, std::span<const std::uint64_t> indices) const;
    void unlink(std::uint32_t id);
    void linkNewest(std::uint32_t id);
    std::uint32_t newPage(std::uint64_t page, MirrorSlot slot, const ResidencyImport& import, bool guarded);
    void releasePage(std::uint32_t id);
    std::optional<MirrorSlot> takeFree();
    void freeSlot(MirrorSlot slot);
    void retireReached();
    std::optional<std::uint32_t> allocateChunk();
    void releaseChunk(std::uint32_t chunk);
    std::uint64_t budgetBytes();
    std::uint32_t chunkLimit(bool growing);
    bool fullResidency();
    std::optional<std::uint32_t> victim(std::uint64_t protectFirst, std::uint64_t protectLast);
    void admit(std::uint64_t first, std::uint64_t last);
    ResidencyRangeState useRange(std::uint64_t address, std::uint64_t bytes, std::uint64_t largest, bool& stable);
    std::uint64_t admitPages(std::uint64_t first, std::uint64_t last, std::uint64_t* quota, bool evict = true, bool sharedOnly = false, std::vector<std::uint64_t>* reserved = nullptr);
    void writeBackDirty(std::span<const std::uint32_t> ids, bool keepPendingWrites = false);
    bool writePending(std::uint64_t page);
    void settleAccessGuards();
    void settleImports();
    void unsettle(std::uint32_t id);
    static bool dirtyable(const Page& page) { return page.shared || page.importSettled; }
    void admitEager();
    void admitReservedTouches();
    void rearm(std::uint32_t id);
    void coolHot(std::uint64_t first, std::span<const std::uint8_t> written);
    void markHot(std::uint64_t page, std::uint64_t at);
    void leaveForImport(std::uint64_t page, std::uint64_t at);
    void noteRefused(std::uint64_t page, ResidencyRefusal refusal);
    void applyRefusalClears(std::span<const std::tuple<std::uint64_t, std::uint64_t, std::uint32_t>> clears);
    void serveHotFromImports();
    void coolTouched(std::span<const std::uint64_t> touched);
    void unmarkHot(std::uint64_t page);
    std::optional<MirrorSlot> acquireSlot(std::uint64_t protectFirst, std::uint64_t protectLast, WriteBacks& writeBacks, bool* evicted = nullptr);
    void leave(std::uint32_t id, WriteBacks& writeBacks);
    void forget(std::uint32_t id);
    void recordWriteBacks(WriteBacks& writeBacks);
    void markDirty(std::uint32_t id, std::uint32_t begin, std::uint32_t end);
    void clearDirty(std::uint32_t id, std::uint32_t begin, std::uint32_t end);
    void clearAllDirty(std::uint32_t id);
    void publishDirty(std::uint64_t page, ResidencyMask mask);
    void publishStale(std::uint64_t page, ResidencyMask mask);
    void publishUnsettled(std::uint64_t page, std::uint8_t reason, bool set);
    void addUnsettled(std::unordered_map<std::uint64_t, ResidencyToken>& entries, std::uint64_t page, const ResidencyToken& token, std::uint8_t reason);
    void eraseUnsettled(std::unordered_map<std::uint64_t, ResidencyToken>& entries, std::uint64_t page, std::uint8_t reason);
    template<typename TPredicate>
    void eraseUnsettledIf(std::unordered_map<std::uint64_t, ResidencyToken>& entries, std::uint8_t reason, TPredicate predicate) {
        for (auto it = entries.begin(); it != entries.end();) {
            if (!predicate(*it)) {
                ++it;
                continue;
            }
            publishUnsettled(it->first, reason, false);
            it = entries.erase(it);
        }
    }
    void publishPage(std::uint64_t page);
    bool retileStale(std::span<const std::uint64_t> stalePages, std::span<const IResidencyOwner* const> except = {});
    std::vector<std::uint64_t> staleIn(std::uint64_t address, std::uint64_t bytes) const;
    bool isStale(std::uint64_t page) const;

    struct Owned {
        IResidencyOwner* owner = nullptr;
        ResidencyMask stale = 0;
        ResidencyMask guarded = 0;
    };
    void appendDirty(std::uint32_t id, std::vector<ResidencyCopy>& copies) const;
    void appendClean(std::uint32_t id, std::uint32_t begin, std::uint32_t end, std::vector<ResidencyCopy>& copies) const;
    void copyIn(std::span<const Fill> fills);
    std::vector<Fill> deferFills(std::span<const Fill> fills);
    void markBehind(std::uint32_t id, ResidencyMask mask);
    void clearBehind(std::uint32_t id);
    void catchUp(std::span<const std::uint32_t> ids, bool waitForWriters);
    void refresh(std::vector<Fill> fills);
    void refreshOrdered(std::span<const Fill> fills);
    bool waitWriters(std::span<const Fill> fills);
    void applyFaults(std::vector<std::uint64_t> written);
    void applyRecorded(std::uint64_t address, std::uint64_t bytes);
    bool unboundedPending(const Page& page) const { return page.gpuWritable && unboundedIssued > std::max(page.unboundedSerial, unboundedApplied); }
    ResidencyToken unboundedSince(std::uint64_t serial) const;
    void exposeToWriters(std::uint32_t id);
    bool settleUnbounded(std::span<const std::uint32_t> ids, bool guard);
    void liftUnboundedGuards();
    bool guardDirty(std::uint32_t id, std::uint32_t begin, std::uint32_t end);
    void evictDraining();
    void demote(std::span<const std::uint32_t> ids);
    std::vector<Fill> demoteUnused(std::span<const Fill> fills);
    void changed(std::uint64_t page);
    void waitFor(const ResidencyToken& token);
    bool flushForCpuRead(std::uint64_t address, std::uint64_t bytes, bool unlocked);
    void releaseCpuRead(std::uint64_t firstPage, std::uint64_t lastPage, bool writersSettled);

    mutable ResidencyMutex structure;
    IResidencyDevice& device;
    const std::uint32_t chunkSlots;
    const std::uint32_t hotFaults;
    const std::uint32_t hotFrames;
    const std::uint32_t quietFrames;
    const bool deferCopyIn;
    bool catchingUp = false;
    std::uint64_t budget = 0;
    std::uint32_t maxChunks = 0;
    std::uint64_t limitEpoch = 0;
    std::uint64_t measured = 0;
    std::uint64_t measuredEpoch = 0;
    std::uint64_t appliedOthers = 0;
    std::uint32_t liveChunks = 0;
    std::vector<Chunk> chunks;
    std::vector<Page> pages;
    std::vector<std::uint32_t> freePages;
    std::unordered_map<std::uint64_t, std::uint32_t> resident;
    std::uint32_t oldest = None;
    std::uint32_t newest = None;
    std::deque<Retiring> retiring;
    std::unordered_map<std::uint64_t, ResidencyToken> writers;
    std::unordered_map<std::uint64_t, ResidencyToken> writingBack;
    std::unordered_set<std::uint64_t> changes;
    std::unordered_map<std::uint64_t, Cooling> hot;
    std::unordered_map<std::uint64_t, std::uint64_t> hotResident;
    std::map<std::uint64_t, std::uint64_t> hotRuns;
    std::vector<std::uint64_t> leavingForImport;
    std::map<std::uint64_t, ResidencyRefusal> refused;
    std::uint64_t sweptEpoch = 0;
    std::unordered_set<std::uint64_t> accessGuards;
    std::unordered_set<std::uint64_t> unsettledImports;
    std::vector<std::uint64_t> settlesWithoutToken;
    bool withoutRecorder = false;
    std::vector<std::uint64_t> queuedCpuPages;
    mutable std::shared_mutex ownedMutex;
    std::unordered_map<std::uint64_t, Owned> owned;
    std::unordered_set<std::uint64_t> staleOwned;
    std::atomic<std::uint32_t> staleCount{0};
    std::atomic<std::uint64_t> ownershipSerial{0};
    std::uint64_t importsGeneration = 0;
    std::vector<ResidencyRange> pendingImports;
    std::vector<std::uint64_t> reservedTouches;
    std::deque<UnboundedWriter> unboundedWriters;
    std::vector<std::uint64_t> unboundedStored;
    std::vector<std::uint64_t> writerExposed;
    std::size_t writerExposedCount = 0;
    std::uint64_t unboundedIssued = 0;
    std::uint64_t unboundedApplied = 0;
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> canonical;
    std::unordered_map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>> aliasKeys;
    std::atomic<std::uint64_t> epoch{1};
    std::atomic<bool> queued{false};
    std::uint64_t serial = 0;
    std::uint64_t rangesSerial = ~std::uint64_t{0};
    static constexpr std::size_t ChangeLogPages = std::size_t{1} << 12;
    std::vector<std::uint32_t> changeLog;
    std::vector<ResidencyRange> ranges;
    std::atomic<std::uint32_t> dirtyPages{0};
    std::atomic<std::uint64_t> unsettledPages{0};
    std::unique_ptr<std::atomic<DirtyLeaf*>[]> dirtyLeaves;
    std::uint64_t touchFrame = 0;
    std::atomic<std::uint64_t> frame{1};
    std::uint64_t reuseVersion = 0;
    std::uint64_t reuseKeyVersion = ~std::uint64_t{0};
    std::uint64_t reuseFrame = 0;
    std::uint64_t reuseEpoch = 0;
    ResidencyReuseStamp reuseStamp;
    std::unordered_map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>> reused;
    ResidencyCounts counts;
    mutable std::mutex residentMutex;
    std::mutex queueMutex;
    std::vector<ResidencyRange> dropQueue;
    std::vector<std::uint64_t> cpuQueue;
    std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint32_t>> refusalQueue;
    std::vector<ResidencyRange> gpuQueue;
};

}

#endif
