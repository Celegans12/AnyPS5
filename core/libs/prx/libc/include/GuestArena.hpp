#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTARENA_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTARENA_HPP

#include <cstddef>
#include <cstdint>

// Guest virtual memory is placed inside one reserved arena below the PS5 application map limit
// absolute address and breaks on host addresses outside that range.
namespace GuestArena {

#ifdef ANYPS5_ENABLE_ASAN
inline constexpr std::uintptr_t GuestAddressOffset = 0x0000100000000000ull;
#else
inline constexpr std::uintptr_t GuestAddressOffset = 0;
#endif

// Ownership probe (WindowsMappings::ArmProbe, the driver's APS5_PROFILE_OWNERSHIP; Windows only):
// one-shot guards on 64 KiB blocks of private guest memory the GPU wrote, counting the CPU's first
// touch of each. A guarded interval ends with a read or a write fault, the block's next GPU write
// (rewritten untouched), or a disarm before a host write, a host read, a query covering it, a
// mapping or protection change, or owned-proof arming. After a read the block's writable pages stay
// read-only until the first store (written) or the next end (other).
enum class ProbeEnd : std::uint8_t { Read, Write, Rewritten, HostWrite, HostRead, Query, Mapping, Owned, Count };
enum class ProbeRefusal : std::uint8_t { Outside, View, Owned, Pinned, HostAccess, InFlight, NotPrivate, Protection, Empty, Failed, Cap, Count };
enum class ProbeTouch : std::uint8_t { Read, Write, WriteAfterRead };
inline constexpr std::size_t ProbeEndCount = static_cast<std::size_t>(ProbeEnd::Count);
inline constexpr std::size_t ProbeRefusalCount = static_cast<std::size_t>(ProbeRefusal::Count);

// frames: the touch's callers, newest first (return addresses; frames[0] is the faulting
// function's caller). The first `exactFrames` come from the x64 unwind tables (ucrtbase, the
// system DLLs and the functions of our modules that have them), so the first caller outside the
// CRT's memcpy/memcmp is exact; the rest are return addresses found by scanning the stack above
// the last exact frame (an address inside a loaded image just after a call instruction), which can
// include stale ones. context: what the faulting thread said it was doing, from the hook the
// driver sets (GuestArenaSetProbeContextHook; its PM4 packet and read site), 0 without one.
inline constexpr std::size_t ProbeFrames = 8;
struct GuestArenaProbeEvent {
    std::uint64_t block;
    std::uint64_t rip;
    std::uint64_t latencyUs;
    std::uint64_t frames[ProbeFrames];
    std::uint64_t context;
    std::uint32_t thread;
    std::uint8_t touch;
    std::uint8_t tag;
    std::uint8_t exactFrames;
    std::uint8_t frameCount;
};

// Since the last take: guarded intervals begun (arms), re-armed by a GPU write while untouched
// (also counted as ends[Rewritten]), touch events dropped (the event buffer was full), intervals
// whose guard the system dropped without a fault reaching the handler (kernel-mode access or a
// foreign protection change), guard faults after the block's first, after-read ends; live = blocks
// armed now.
struct GuestArenaProbeCounts {
    std::uint64_t arms;
    std::uint64_t live;
    std::uint64_t dropped;
    std::uint64_t lost;
    std::uint64_t extraFaults;
    std::uint64_t afterReadWritten;
    std::uint64_t afterReadOther;
    std::uint64_t ends[ProbeEndCount];
    std::uint64_t refused[ProbeRefusalCount];
    // Faults taken by a thread that already holds the mapping mutex (the handler cannot lock it
    // again): guard faults in the arena, let through (the system dropped the guard), and access
    // violations, left to the crash report. underLock is the first such fault of the run (block =
    // the faulting address, touch = Read/Write, tag 1 for a guard fault); underLockSeen says it is set.
    std::uint64_t faultsUnderLock;
    std::uint64_t violationsUnderLock;
    std::uint64_t underLockSeen;
    GuestArenaProbeEvent underLock;
};

#ifndef _WIN32
using SharedBackingResolver = bool (*)(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset);
using SharedBackingWriter = bool (*)(std::uintptr_t address, const void* source, std::size_t bytes);
#endif

extern "C" {

bool GuestArenaAvailable_nid_postfix();
bool GuestArenaContains_nid_postfix(const void* pointer, std::size_t bytes);
void* GuestArenaAllocate_nid_postfix(std::size_t bytes, std::size_t alignment);
void* GuestArenaAllocateAtOrAbove_nid_postfix(std::uintptr_t hint, std::size_t bytes, std::size_t alignment);
void* GuestArenaAllocateFromHint_nid_postfix(std::uintptr_t hint, std::size_t bytes, std::size_t alignment);
std::uintptr_t GuestArenaHighestFree_nid_no_patch(std::size_t bytes, std::size_t alignment);
bool GuestArenaNextUsed_nid_no_patch(std::uintptr_t address, std::uintptr_t* start, std::uintptr_t* end);
void GuestArenaMarkUsed_nid_postfix(const void* pointer, std::size_t bytes);
void GuestArenaRelease_nid_postfix(const void* pointer, std::size_t bytes);
// The reserved range, and whether it was reserved with page write watching (Windows MEM_WRITE_WATCH).
void GuestArenaRange_nid_postfix(std::uintptr_t* base, std::size_t* bytes);
bool GuestArenaWriteWatched_nid_postfix();
void GuestArenaSetProtection_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint32_t protection);
void GuestArenaCommit_nid_postfix(void* pointer, std::size_t bytes, std::uint32_t protection, std::size_t granule);
void GuestArenaReset_nid_postfix(void* pointer, std::size_t bytes);
void GuestArenaSetAccessObserver_nid_postfix(void (*callback)(std::uintptr_t address, std::size_t bytes));
#ifdef _WIN32
bool GuestArenaHandleWrite_nid_postfix(std::uintptr_t address);
bool GuestArenaHandleRead_nid_postfix(std::uintptr_t address);
void GuestArenaPinWritable_nid_postfix(const void* pointer, std::size_t bytes);
void GuestArenaUnpinWritable_nid_postfix(const void* pointer, std::size_t bytes);
bool GuestArenaProtection_nid_postfix(std::uintptr_t address, std::uint32_t* protection);
bool GuestArenaCollectWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear);
bool GuestArenaCollectArmedWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear);
bool GuestArenaCollectFreshWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear);
bool GuestArenaHostRegionOverlaps_nid_postfix(std::uintptr_t address, std::size_t bytes);
std::uint64_t GuestArenaCommitGeneration_nid_postfix();
void GuestArenaSetPrivateMappingObserver_nid_postfix(void (*callback)(std::uintptr_t address, std::size_t bytes, std::uint64_t generation));
void GuestArenaMap_nid_postfix(void* pointer, std::size_t bytes, void* section, std::uint64_t offset, std::uint32_t protection);
void GuestArenaMapDirect_nid_postfix(void* pointer, std::size_t bytes, void* section, std::uint64_t offset, std::uint32_t protection);
void GuestArenaMapFile_nid_postfix(void* pointer, std::size_t bytes, void* section, std::uint64_t offset, std::uint32_t protection, std::uint32_t viewProtection, std::size_t sectionBytes);
void* GuestArenaMapAlias_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestArenaUnmapAlias_nid_postfix(void* alias);
bool GuestArenaResidentSpan_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uintptr_t* start, std::uintptr_t* end);
void GuestArenaCompareWhenShared_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void GuestArenaCompareResidentWhenShared_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void GuestArenaComparedPages_nid_no_patch(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count);
void GuestArenaTakeUnreportedPages_nid_no_patch(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count);
std::size_t GuestArenaAliasedPages_nid_postfix(std::uintptr_t* pages, std::uintptr_t* identities, std::size_t capacity, std::size_t* pageBytes);
void GuestArenaForgetSection_nid_postfix(void* section);
void GuestArenaSavedMemory_nid_postfix(std::uint64_t* bytes, std::uint64_t* blocks);
bool GuestArenaBeginHostImport_nid_no_patch(void* pointer, std::size_t bytes);
void GuestArenaEndHostImport_nid_no_patch(void* pointer, std::size_t bytes);
bool GuestArenaForgetReusableSection_nid_no_patch(void* section);
#else
void GuestArenaSetSharedBacking_nid_postfix(SharedBackingResolver resolver);
void GuestArenaSetSharedBackingWriter_nid_no_patch(SharedBackingWriter writer);
bool GuestArenaWriteSharedBacking_nid_no_patch(std::uintptr_t address, const void* source, std::size_t bytes);
bool GuestArenaSharedBacking_nid_postfix(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset);
#endif
bool GuestArenaBeginHostWrite_nid_postfix(void* pointer, std::size_t bytes);
void GuestArenaEndHostWrite_nid_postfix(void* pointer, std::size_t bytes);
void GuestArenaSetHostWriteHook_nid_no_patch(void (*hook)(std::uintptr_t address, std::size_t bytes));
void GuestArenaSetHostReadHook_nid_no_patch(void (*hook)(std::uintptr_t address, std::size_t bytes));
void GuestArenaPrepareHostRead_nid_no_patch(const void* pointer, std::size_t bytes);
// Owned blocks (WindowsMappings::ArmOwned, the driver's owned proofs; write watch, no protection
// change and no fault): arms every 64 KiB block overlapping the range (`count` = their number) and
// stores their armed words, or arms none (false: a block that cannot be armed, outside the arena,
// or not Windows). Flag 1 lets private commits be armed. OwnedUnchanged: whether every block still
// holds the word armed and no page of the range was written since (GetWriteWatch without reset; no
// CPU store, no mapping or protection change). Counts, since the last call, up to `count` values:
// arms, blocks newly armed, refusals, guest write faults inside armed blocks (none raised or
// resolved by owned blocks), other disarms, blocks armed with private pages, blocks armed with
// nothing mapped, blocks armed partly mapped, disarms by a collect that took a written page, pages
// whose write bits an arming took, GetWriteWatch queries of OwnedUnchanged, OwnedUnchanged answers
// that found a written page, OwnedUnchanged answers asked again under the lock, answers with a mark
// held from the blocks' checks alone, answers with a mark that queried, then refusals by reason (WindowsMappings::OwnedRefusal: none, outside
// the arena or a host region, shared or aliased view, pinned or importing, host write open,
// protection change in flight, not writable or another mapping type, private with flag 1 clear,
// nothing mapped (no longer refused), not write-watched).
// OwnedMark: a new mark for OwnedUnchanged (0: none, every answer queries): an answer with a mark
// may rely on a block's check by a query that began after the mark (WindowsMappings::OwnedUnchanged).
// EndSetProtection ends the guest protection change that GuestArenaSetProtection began, after the
// caller's own protection call.
inline constexpr std::size_t GuestArenaOwnedCountSlots = 25;
bool GuestArenaArmOwned_nid_no_patch(std::uintptr_t address, std::size_t bytes, std::uint64_t* words, std::size_t count, unsigned flags);
bool GuestArenaOwnedUnchanged_nid_no_patch(std::uintptr_t address, std::size_t bytes, const std::uint64_t* words, std::size_t count, std::uint64_t mark);
std::uint64_t GuestArenaOwnedMark_nid_no_patch();
void GuestArenaTakeOwnedCounts_nid_no_patch(std::uint64_t* counts, std::size_t count);
void GuestArenaEndSetProtection_nid_no_patch(std::uintptr_t address, std::size_t bytes);
// Ownership probe: arms the 64 KiB block at `block` (re-arms it when armed and untouched; false
// when it cannot be armed, counted by reason). Take: the counts since the last take and up to
// `capacity` touch events (returns their number; the rest stay for the next take). Query is
// VirtualQuery that first disarms the probed blocks the answer would show, so no caller sees a
// probe's guard; with nothing armed it is VirtualQuery. Not Windows: nothing is armed.
bool GuestArenaProbeArm_nid_no_patch(std::uintptr_t block, std::uint8_t tag);
std::size_t GuestArenaProbeTake_nid_no_patch(GuestArenaProbeCounts* counts, GuestArenaProbeEvent* events, std::size_t capacity);
// The driver's description of the calling thread's current work, called from the fault handler on
// the faulting thread (must not lock, allocate or fault); null clears it.
void GuestArenaSetProbeContextHook_nid_no_patch(std::uint64_t (*hook)());
#ifdef _WIN32
// `context` is the fault's CONTEXT record (its Rip is the touch's site; the callers are unwound from it).
bool GuestArenaHandleProbe_nid_postfix(std::uintptr_t address, bool write, bool guard, const void* context);
// The thread id holding the mapping mutex (WindowsMappings), 0 when free. Test hook: runs `body`
// with the mapping mutex held by the calling thread.
std::uint32_t GuestArenaMappingLockOwner_nid_no_patch();
void GuestArenaRunUnderMappingLock_nid_no_patch(void (*body)(void*), void* argument);
std::size_t GuestArenaQuery_nid_no_patch(const void* address, void* information, std::size_t bytes);
#endif

}

class HostWrite {
public:
    HostWrite(void* pointer, std::size_t bytes) : pointer(pointer), bytes(bytes), open(GuestArenaBeginHostWrite_nid_postfix(pointer, bytes)) {}
    ~HostWrite() {
        if (open) GuestArenaEndHostWrite_nid_postfix(pointer, bytes);
    }
    HostWrite(const HostWrite&) = delete;
    HostWrite& operator=(const HostWrite&) = delete;
    bool Open() const { return open; }

private:
    void* pointer;
    std::size_t bytes;
    bool open;
};

#ifdef _WIN32
class HostImportAccess {
public:
    HostImportAccess(void* pointer, std::size_t bytes) : pointer(pointer), bytes(bytes), open(GuestArenaBeginHostImport_nid_no_patch(pointer, bytes)) {}
    ~HostImportAccess() {
        if (open) GuestArenaEndHostImport_nid_no_patch(pointer, bytes);
    }
    HostImportAccess(const HostImportAccess&) = delete;
    HostImportAccess& operator=(const HostImportAccess&) = delete;

private:
    void* pointer;
    std::size_t bytes;
    bool open;
};
#endif

}

#endif
