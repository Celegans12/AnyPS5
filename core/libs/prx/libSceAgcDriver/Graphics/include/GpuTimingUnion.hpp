#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GPUTIMINGUNION_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GPUTIMINGUNION_HPP

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// The [gputime-union] arithmetic (APS5_PROFILE_GPU): the union of timed GPU ranges across batches and
// lanes within one profile window, and the program families of APS5_PROFILE_GPU_FAMILY.
namespace AgcDriver::Graphics::GpuTimingUnion {

using Interval = std::pair<std::uint64_t, std::uint64_t>;

// Ticks covered by at least one interval ([begin, end), any order, overlaps merged).
inline std::uint64_t UnionTicks(std::vector<Interval> intervals) {
    std::sort(intervals.begin(), intervals.end());
    std::uint64_t covered = 0, reached = 0;
    bool any = false;
    for (const auto& [begin, end] : intervals) {
        if (end <= begin) continue;
        const auto from = any ? std::max(begin, reached) : begin;
        if (end > from) covered += end - from;
        reached = any ? std::max(reached, end) : end;
        any = true;
    }
    return covered;
}

// APS5_PROFILE_GPU_FAMILY=<name>:<address>,<address>...[;<name>:...] (addresses hexadecimal with or
// without 0x): named sets of program addresses whose ranges are also unioned per window.
struct Family {
    std::string name;
    std::vector<std::uint64_t> programs;
};

inline std::vector<Family> ParseFamilies(std::string_view text) {
    std::vector<Family> families;
    while (!text.empty()) {
        const auto end = text.find(';');
        const auto entry = text.substr(0, end);
        text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
        const auto colon = entry.find(':');
        if (colon == std::string_view::npos || colon == 0) continue;
        Family family{std::string(entry.substr(0, colon)), {}};
        auto list = entry.substr(colon + 1);
        while (!list.empty()) {
            const auto comma = list.find(',');
            const std::string item(list.substr(0, comma));
            list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
            char* parsed = nullptr;
            const auto value = std::strtoull(item.c_str(), &parsed, 16);
            if (!item.empty() && parsed == item.c_str() + item.size() && value != 0) family.programs.push_back(value);
        }
        if (!family.programs.empty()) families.push_back(std::move(family));
    }
    return families;
}

// Program address -> family index (the first family that lists it).
inline std::unordered_map<std::uint64_t, std::size_t> FamilyIndex(const std::vector<Family>& families) {
    std::unordered_map<std::uint64_t, std::size_t> index;
    for (std::size_t family = 0; family < families.size(); ++family) {
        for (const auto program : families[family].programs) index.try_emplace(program, family);
    }
    return index;
}

}

#endif
