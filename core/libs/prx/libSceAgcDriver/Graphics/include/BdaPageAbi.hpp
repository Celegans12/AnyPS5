#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BDAPAGEABI_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BDAPAGEABI_HPP

#include "BdaAbi.hpp"
#include <cstdint>

namespace ShaderRecompiler::BdaAbi {

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

}

#endif
