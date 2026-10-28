#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BDAPAGEREFERENCE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BDAPAGEREFERENCE_HPP

#include "BdaAbi.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace BdaPageReference {

namespace Abi = ShaderRecompiler::BdaAbi;

struct Translation {
    std::uint64_t device = 0;
    std::uint32_t reason = 0;
    bool operator==(const Translation&) const = default;
};

using Leaves = std::map<std::uint32_t, std::vector<Abi::PageEntry>>;

inline Translation Fault(Abi::FaultReason reason) {
    return {0u, static_cast<std::uint32_t>(reason)};
}

inline bool Permitted(std::uint32_t permissions, std::uint32_t permission, std::span<const Abi::Grant> grants, std::uint64_t address, std::uint64_t end) {
    if ((permissions & permission) == permission) return true;
    if (((permissions | Abi::Write) & permission) != permission) return false;
    const auto found = std::upper_bound(grants.begin(), grants.end(), address, [](std::uint64_t value, const Abi::Grant& grant) { return value < grant.begin; });
    return found != grants.begin() && end <= std::prev(found)->end;
}

inline Translation LookupList(std::span<const Abi::Range> ranges, std::uint64_t address, std::uint32_t bytes, std::uint32_t permission, std::span<const Abi::Grant> grants = {}) {
    const auto end = address + bytes;
    const auto found = std::upper_bound(ranges.begin(), ranges.end(), address, [](std::uint64_t value, const Abi::Range& range) { return value < range.begin; });
    if (found == ranges.begin()) return Fault(Abi::FaultReason::Unmapped);
    const auto& range = *std::prev(found);
    if (range.reserved != 0u || range.begin >= range.end) return Fault(Abi::FaultReason::InvalidTable);
    if (end > range.end) return Fault(Abi::FaultReason::Unmapped);
    if (!Permitted(range.permissions, permission, grants, address, end)) return Fault(Abi::FaultReason::Permission);
    if (range.deviceAddress == 0u) return Fault(Abi::FaultReason::InvalidTable);
    const auto device = range.deviceAddress + (address - range.begin);
    if (device < range.deviceAddress || device + bytes <= device) return Fault(Abi::FaultReason::Overflow);
    return {device, 0u};
}

inline std::optional<Translation> WalkPages(const Leaves& leaves, std::uint64_t address, std::uint32_t bytes, std::uint32_t permission, std::span<const Abi::Grant> grants = {}) {
    if (address >= Abi::PageTableEnd) return std::nullopt;
    const auto leaf = leaves.find(static_cast<std::uint32_t>(address >> Abi::WindowShift));
    if (leaf == leaves.end()) return std::nullopt;
    const auto& entries = leaf->second;
    auto entry = entries.at((address >> Abi::PageShift) & (Abi::WindowPages - 1u));
    std::uint32_t shift = Abi::PageShift;
    while (Abi::KindOf(entry) == Abi::PageKind::Split && shift != 0u) {
        shift -= Abi::SplitShift;
        entry = entries.at(entry.target / sizeof(Abi::PageEntry) + ((address >> shift) & (Abi::SplitChildren - 1u)));
    }
    const auto kind = Abi::KindOf(entry);
    if (kind != Abi::PageKind::None && kind != Abi::PageKind::Segment) return Fault(Abi::FaultReason::InvalidTable);
    const auto offset = address & ((std::uint64_t{1} << shift) - 1u);
    const auto limit = Abi::LimitOf(entry);
    const bool held = kind == Abi::PageKind::Segment && offset >= Abi::FirstOf(entry) && offset < limit;
    if (!held) return std::nullopt;
    if (offset + bytes > limit) return Fault(Abi::FaultReason::Unmapped);
    if (!Permitted(Abi::PermissionsOf(entry), permission, grants, address, address + bytes)) return Fault(Abi::FaultReason::Permission);
    return Translation{address + entry.target, 0u};
}

inline Translation Translate(const Leaves& leaves, std::span<const Abi::Range> ranges, std::uint64_t address, std::uint32_t bytes, std::uint32_t permission, std::span<const Abi::Grant> grants = {}) {
    if (bytes == 0u || address + bytes <= address) return Fault(Abi::FaultReason::Overflow);
    if (const auto walked = WalkPages(leaves, address, bytes, permission, grants)) return *walked;
    return LookupList(ranges, address, bytes, permission, grants);
}

inline std::vector<Abi::Range> Without(std::span<const Abi::Range> ranges, std::span<const Abi::Range> held) {
    std::vector<Abi::Range> rest;
    for (const auto& range : ranges) {
        if (std::none_of(held.begin(), held.end(), [&](const Abi::Range& other) { return other.begin == range.begin && other.end == range.end && other.deviceAddress == range.deviceAddress && other.permissions == range.permissions; })) rest.push_back(range);
    }
    return rest;
}

inline std::vector<Abi::Grant> RandomGrants(std::mt19937_64& random, std::span<const Abi::Range> ranges, std::size_t count) {
    std::vector<Abi::Grant> grants;
    if (ranges.empty()) return grants;
    for (std::size_t index = 0; index < count; ++index) {
        const auto& range = ranges[random() % ranges.size()];
        const auto size = range.end - range.begin;
        const auto begin = range.begin + (size > 1u ? random() % size : 0u) - (random() % 3u == 0u ? random() % 64u : 0u);
        const auto end = begin + 1u + random() % (random() % 2u == 0u ? 64u : 0x30000u);
        grants.push_back({begin, end});
    }
    std::sort(grants.begin(), grants.end(), [](const Abi::Grant& left, const Abi::Grant& right) { return left.begin < right.begin; });
    std::vector<Abi::Grant> merged;
    for (const auto& grant : grants) {
        if (!merged.empty() && grant.begin <= merged.back().end) merged.back().end = std::max(merged.back().end, grant.end);
        else merged.push_back(grant);
    }
    return merged;
}

template<typename TEncode>
Leaves EncodeLeaves(std::span<const Abi::Range> ranges, TEncode&& encode) {
    std::map<std::uint32_t, std::vector<Abi::Range>> windows;
    for (const auto& range : ranges) {
        for (auto window = static_cast<std::uint32_t>(range.begin >> Abi::WindowShift); window <= static_cast<std::uint32_t>((range.end - 1u) >> Abi::WindowShift); ++window) windows[window].push_back(range);
    }
    Leaves leaves;
    for (const auto& [window, held] : windows) leaves.emplace(window, encode(window, std::span<const Abi::Range>(held)));
    return leaves;
}

struct Table {
    std::vector<Abi::Range> list;
    std::vector<Abi::Range> pages;
};

inline Table RandomTable(std::mt19937_64& random, std::uint64_t base, std::size_t count) {
    static constexpr std::array<std::uint64_t, 6> Grains{1u, 4u, 16u, 256u, 4096u, 65536u};
    Table table;
    auto cursor = base;
    std::uint64_t device = 0x100000000ull;
    for (std::size_t index = 0; index < count; ++index) {
        const auto grain = Grains[random() % Grains.size()];
        const auto gap = (random() % 4u == 0u) ? 0u : (random() % 3u) * grain + (random() % 2u) * (random() % 70000u);
        const auto begin = cursor + gap;
        const auto size = std::max<std::uint64_t>(1u, (random() % 5u) * grain + (random() % 3u == 0u ? random() % 200000u : random() % 40u));
        const auto permissions = random() % 8u == 0u ? 0u : random() % 3u == 0u ? Abi::Read | Abi::Write : Abi::Read;
        device += 0x10000u + (random() % 64u) * 4u + (random() % 2u);
        table.list.push_back({begin, begin + size, device, permissions, 0u});
        if (random() % 4u != 0u) table.pages.push_back(table.list.back());
        device += size;
        cursor = begin + size;
    }
    return table;
}

inline std::vector<std::uint64_t> Probes(std::span<const Abi::Range> ranges) {
    std::vector<std::uint64_t> probes;
    for (const auto& range : ranges) {
        for (const auto edge : {range.begin, range.end}) {
            for (const std::int64_t delta : {-70000, -65537, -17, -16, -2, -1, 0, 1, 2, 15, 16, 4095, 65535, 65536}) probes.push_back(edge + static_cast<std::uint64_t>(delta));
        }
        probes.push_back(range.begin + (range.end - range.begin) / 2u);
    }
    return probes;
}

inline constexpr std::array<std::uint32_t, 9> ProbeBytes{1u, 2u, 3u, 4u, 8u, 16u, 64u, 65536u, 0xfffffff0u};

}

#endif
