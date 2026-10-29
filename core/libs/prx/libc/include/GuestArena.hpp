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
bool GuestArenaOverlapsHostRegion_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void GuestArenaReleaseHostRegion_nid_no_patch(const void* pointer, std::size_t bytes);
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
std::uint32_t GuestArenaMappingLockOwner_nid_no_patch();
std::uint8_t GuestArenaGuardAccess_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void GuestArenaUnguardAccess_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void GuestArenaOpenAccessGuards_nid_no_patch(std::uintptr_t address, std::size_t bytes);
bool GuestArenaAccessGuarded_nid_no_patch(std::uintptr_t address);
std::size_t GuestArenaAccessGuardedSubpages_nid_no_patch();
void GuestArenaSetAccessGuardObserver_nid_no_patch(void (*callback)(std::uintptr_t address));
void GuestArenaSetMappingGuardObserver_nid_no_patch(void (*callback)(std::uintptr_t address, std::size_t bytes, bool releasing));
void GuestArenaRunUnderMappingLock_nid_no_patch(void (*body)(void*), void* argument);

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
