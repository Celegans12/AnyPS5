#ifndef CORE_SHADER_RECOMPILER_BDAABI_HPP
#define CORE_SHADER_RECOMPILER_BDAABI_HPP

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace ShaderRecompiler::BdaAbi {

inline constexpr std::uint32_t Version = 1;
inline constexpr std::uint32_t Read = 1;
inline constexpr std::uint32_t Write = 2;

struct Header {
    std::uint32_t version;
    std::uint32_t count;
    std::uint32_t entryBytes;
    std::uint32_t reserved;
};

struct Range {
    std::uint64_t begin;
    std::uint64_t end;
    std::uint64_t deviceAddress;
    std::uint32_t permissions;
    std::uint32_t reserved;
};

inline constexpr std::uint32_t PageShift = 16;
inline constexpr std::uint32_t WindowShift = 32;
inline constexpr std::uint32_t AddressBits = 47;
inline constexpr std::uint64_t PageTableEnd = std::uint64_t{1} << AddressBits;
inline constexpr std::uint32_t WindowCount = 1u << (AddressBits - WindowShift);
inline constexpr std::uint32_t WindowPages = 1u << (WindowShift - PageShift);
inline constexpr std::uint32_t SplitShift = 4;
inline constexpr std::uint32_t SplitChildren = 1u << SplitShift;
inline constexpr std::uint64_t LimitMax = (std::uint64_t{1} << 33) - 1u;
inline constexpr std::uint32_t FirstMask = 0x1ffffu;
inline constexpr std::uint32_t LimitHighShift = 17;
inline constexpr std::uint32_t PermissionShift = 18;
inline constexpr std::uint32_t KindShift = 30;
inline constexpr std::uint32_t MirroredShift = 20;
inline constexpr std::uint32_t RangeMirrored = 1;

enum class PageKind : std::uint32_t { None, Segment, Split };

struct PageEntry {
    std::uint64_t target;
    std::uint32_t bounds;
    std::uint32_t limit;
};

constexpr PageKind KindOf(const PageEntry& entry) {
    return static_cast<PageKind>(entry.bounds >> KindShift);
}

constexpr std::uint32_t FirstOf(const PageEntry& entry) {
    return entry.bounds & FirstMask;
}

constexpr std::uint64_t LimitOf(const PageEntry& entry) {
    return entry.limit | (static_cast<std::uint64_t>((entry.bounds >> LimitHighShift) & 1u) << 32u);
}

constexpr std::uint32_t PermissionsOf(const PageEntry& entry) {
    return (entry.bounds >> PermissionShift) & (Read | Write);
}

constexpr PageEntry SegmentEntry(std::uint64_t target, std::uint32_t first, std::uint64_t limit, std::uint32_t permissions) {
    return {target, first | static_cast<std::uint32_t>(limit >> 32u) << LimitHighShift | permissions << PermissionShift | static_cast<std::uint32_t>(PageKind::Segment) << KindShift, static_cast<std::uint32_t>(limit)};
}

constexpr PageEntry SplitEntry(std::uint64_t children) {
    return {children, static_cast<std::uint32_t>(PageKind::Split) << KindShift, 0u};
}

struct Grant {
    std::uint64_t begin;
    std::uint64_t end;
};

// Range index (APS5_NO_BDA_INDEX=1 leaves it out): a table whose Header::reserved is IndexedTable
// carries, after its ranges, an IndexHeader and `buckets + 1` words start[0..buckets], start[j] being
// the number of ranges whose begin lies below base + (j << shift). The lookup's binary search for the
// count of ranges beginning at or below an address (the upper bound) then starts from
// [start[j], start[j + 1]] for the address's bucket j = (address - base) >> shift instead of
// [0, count): every range in front of the bucket begins below the address, every range behind it
// above. Below `base` the bound is 0, at or past bucket `buckets` it is `count`. Exact: the same
// upper bound, so the same range or fault, as the search over the whole table.
inline constexpr std::uint32_t IndexedTable = 1;
struct IndexHeader {
    std::uint64_t base;
    std::uint32_t buckets;
    std::uint32_t shift;
};
inline constexpr std::uint32_t IndexMinRanges = 16;
inline constexpr std::uint64_t IndexMaxBuckets = std::uint64_t{1} << 16u;

// The index words (IndexHeader, then the starts) of `count` ranges ascending by begin; empty for a
// table too small to gain from one.
inline std::vector<std::uint32_t> BuildRangeIndex(const Range* ranges, std::size_t count) {
    if (count < IndexMinRanges || count > (std::size_t{1} << 24u)) return {};
    const auto base = ranges[0].begin;
    const auto span = ranges[count - 1].begin - base;
    const auto wanted = IndexMaxBuckets < 2u * count ? IndexMaxBuckets : std::uint64_t{2u * count};
    std::uint32_t shift = 0;
    while (shift < 63u && (span >> shift) >= wanted) ++shift;
    const auto buckets = static_cast<std::uint32_t>((span >> shift) + 1u);
    constexpr auto headerWords = sizeof(IndexHeader) / sizeof(std::uint32_t);
    std::vector<std::uint32_t> words(headerWords + buckets + 1u);
    const IndexHeader header{base, buckets, shift};
    std::memcpy(words.data(), &header, sizeof(header));
    std::size_t next = 0;
    for (std::uint32_t bucket = 0; bucket < buckets; ++bucket) {
        // bucket << shift <= span: no overflow.
        const auto limit = base + (std::uint64_t{bucket} << shift);
        while (next < count && ranges[next].begin < limit) ++next;
        words[headerWords + bucket] = static_cast<std::uint32_t>(next);
    }
    words.back() = static_cast<std::uint32_t>(count);
    return words;
}

inline bool RangesOrdered(const Range* ranges, std::size_t count) {
    for (std::size_t index = 1; index < count; ++index) {
        if (ranges[index].begin < ranges[index - 1].begin) return false;
    }
    return true;
}

enum class FaultState : std::uint32_t { Empty, Writing, Ready };
enum class FaultReason : std::uint32_t { Unmapped = 1, Permission, Overflow, InvalidTable, InvalidRectangle, LoopLimit, Unaligned, InvalidDescriptor, ImageTable, UnlinkedCall, WindowLaneSpill = 11, UnnormalizedWrap = 12, DirectImage = 13, BindlessMiss = 14, BindlessRefused = 15, Watched = 16 };

inline constexpr std::uint32_t DirectImageDimension = 1u;

inline constexpr std::uint32_t PackDirectImageFault(std::uint32_t resource) {
    return (resource << 8u) | DirectImageDimension;
}

inline constexpr std::uint32_t DirectImageFaultResource(std::uint32_t payload) {
    return payload >> 8u;
}

inline constexpr std::uint32_t DirectImageFaultReason(std::uint32_t payload) {
    return payload & 0xffu;
}

struct Fault {
    FaultState state;
    FaultReason reason;
    std::uint64_t address;
    std::uint32_t bytes;
    std::uint32_t stage;
    std::uint32_t instruction;
    std::uint32_t otherFaults;
};

static_assert(std::is_standard_layout_v<Header> && std::is_trivially_copyable_v<Header> && sizeof(Header) == 16);
static_assert(std::is_standard_layout_v<Range> && std::is_trivially_copyable_v<Range> && sizeof(Range) == 32);
static_assert(offsetof(Range, deviceAddress) == 16 && offsetof(Range, permissions) == 24);
static_assert(std::is_standard_layout_v<IndexHeader> && sizeof(IndexHeader) == 16 && offsetof(IndexHeader, buckets) == 8 && offsetof(IndexHeader, shift) == 12);
static_assert(std::is_standard_layout_v<Fault> && std::is_trivially_copyable_v<Fault> && sizeof(Fault) == 32);
static_assert(offsetof(Fault, address) == 8 && offsetof(Fault, instruction) == 24 && offsetof(Fault, otherFaults) == 28);

inline constexpr std::uint32_t WrittenPageShift = 12;
inline constexpr std::uint32_t WrittenChunkShift = 17;
inline constexpr std::uint32_t WrittenChunkPages = 1u << (WrittenChunkShift - WrittenPageShift);
inline constexpr std::uint32_t WrittenPageSlots = 2048;
inline constexpr std::uint32_t WrittenPageProbes = 8;
inline constexpr std::uint64_t WrittenAddressLimit = std::uint64_t{0xffffffffu} << WrittenChunkShift;
inline constexpr std::uint32_t WrittenOverflowWord = sizeof(Fault) / sizeof(std::uint32_t);
inline constexpr std::uint32_t WrittenCountWord = WrittenOverflowWord + 1;
inline constexpr std::uint32_t WrittenSlotsWord = WrittenCountWord + 1;
inline constexpr std::uint32_t WrittenListWord = WrittenSlotsWord + 2 * WrittenPageSlots;
inline constexpr std::uint32_t SrtGuardWord = WrittenListWord + WrittenPageSlots;
inline constexpr std::uint32_t SrtGuardStageShift = 24;
inline constexpr std::size_t FaultBufferBytes = (SrtGuardWord + 1) * sizeof(std::uint32_t);
static_assert((WrittenPageSlots & (WrittenPageSlots - 1)) == 0);
static_assert(WrittenChunkPages == 32);

inline constexpr std::uint32_t WrittenChunkKey(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> WrittenChunkShift) + 1u;
}

template<typename TMark>
bool ForEachWrittenRange(const std::uint32_t* words, TMark&& mark) {
    const auto count = words[WrittenCountWord];
    if (count > WrittenPageSlots) return false;
    for (std::uint32_t claim = 0; claim < count; ++claim) {
        const auto slot = words[WrittenListWord + claim];
        if (slot >= WrittenPageSlots || words[WrittenSlotsWord + 2 * slot] == 0) return false;
    }
    for (std::uint32_t claim = 0; claim < count; ++claim) {
        const auto slot = words[WrittenListWord + claim];
        const auto key = words[WrittenSlotsWord + 2 * slot];
        auto mask = words[WrittenSlotsWord + 2 * slot + 1];
        const auto base = static_cast<std::uint64_t>(key - 1u) << WrittenChunkShift;
        while (mask != 0) {
            const auto first = static_cast<std::uint32_t>(std::countr_zero(mask));
            const auto pages = static_cast<std::uint32_t>(std::countr_one(mask >> first));
            mark(base + (static_cast<std::uint64_t>(first) << WrittenPageShift), static_cast<std::size_t>(pages) << WrittenPageShift);
            mask = pages + first >= 32u ? 0u : mask & ~((1u << (first + pages)) - 1u);
        }
    }
    return true;
}

}

#endif
