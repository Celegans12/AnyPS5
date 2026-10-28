#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandWrites.hpp"
#include <algorithm>

namespace AgcDriver::NewPath {

std::uint64_t WriteSet::Bytes() const {
    std::uint64_t bytes = 0;
    for (const auto& [begin, end] : ranges) bytes += end - begin;
    return bytes;
}

std::vector<AddressRange> MergeRanges(std::vector<AddressRange> ranges) {
    ranges.erase(std::remove_if(ranges.begin(), ranges.end(), [](const AddressRange& range) { return range.second <= range.first; }), ranges.end());
    std::sort(ranges.begin(), ranges.end());
    std::vector<AddressRange> merged;
    for (const auto& range : ranges) {
        if (!merged.empty() && range.first <= merged.back().second) merged.back().second = std::max(merged.back().second, range.second);
        else merged.push_back(range);
    }
    return merged;
}

std::vector<AddressRange> WritableRanges(std::span<const ShaderRecompiler::BdaAbi::Range> table) {
    std::vector<AddressRange> writable;
    for (const auto& range : table) {
        if ((range.permissions & ShaderRecompiler::BdaAbi::Write) != 0) writable.emplace_back(range.begin, range.end);
    }
    return MergeRanges(std::move(writable));
}

WriteSet ResolveWrites(std::span<const ShaderRecompiler::BdaAbi::Range> table, Abi::StoreClass stores, std::span<const AddressRange> storeRanges) {
    WriteSet set;
    if (stores == Abi::StoreClass::None) return set;
    const auto writable = WritableRanges(table);
    if (stores == Abi::StoreClass::Unbounded) {
        set.unbounded = true;
        set.ranges = writable;
        return set;
    }
    set.ranges = MergeRanges({storeRanges.begin(), storeRanges.end()});
    for (const auto& [begin, end] : set.ranges) {
        auto cursor = begin;
        auto at = std::upper_bound(writable.begin(), writable.end(), cursor, [](std::uint64_t value, const AddressRange& range) { return value < range.second; });
        for (; at != writable.end() && at->first <= cursor && cursor < end; ++at) cursor = std::min(end, at->second);
        if (cursor < end) set.outsideTable = true;
    }
    return set;
}

}
