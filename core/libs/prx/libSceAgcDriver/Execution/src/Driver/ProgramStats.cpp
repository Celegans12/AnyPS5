#include "prx/libSceAgcDriver/Execution/include/Driver/ProgramStats.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <stdexcept>

namespace AgcDriver::DriverDetail {

namespace {

std::uint64_t parseAddress(const std::string& item, const std::string& entry) {
    const auto digits = item.size() > 2 && item[0] == '0' && (item[1] == 'x' || item[1] == 'X') ? item.substr(2) : item;
    if (digits.empty() || digits.size() > 16 || digits.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) {
        throw std::runtime_error("a program list entry is not a hexadecimal address or address range: '" + entry + "'");
    }
    const auto address = std::strtoull(digits.c_str(), nullptr, 16);
    if (address == 0) throw std::runtime_error("a program list entry is address 0: '" + entry + "'");
    return address;
}

std::string trimmed(std::string item) {
    while (!item.empty() && item.front() == ' ') item.erase(item.begin());
    while (!item.empty() && item.back() == ' ') item.pop_back();
    return item;
}

}

ProgramList ParseProgramList(const char* text) {
    ProgramList list;
    if (text == nullptr) return list;
    const std::string value(text);
    std::size_t begin = 0;
    while (begin <= value.size()) {
        auto end = value.find(',', begin);
        if (end == std::string::npos) end = value.size();
        const auto item = trimmed(value.substr(begin, end - begin));
        if (!item.empty()) {
            const auto dash = item.find('-');
            const auto low = parseAddress(trimmed(item.substr(0, dash)), item);
            const auto high = dash == std::string::npos ? low : parseAddress(trimmed(item.substr(dash + 1)), item);
            if (high < low) throw std::runtime_error("a program list range ends before it starts: '" + item + "'");
            list.emplace_back(low, high);
        }
        begin = end + 1;
    }
    std::sort(list.begin(), list.end());
    ProgramList merged;
    for (const auto& range : list) {
        if (!merged.empty() && range.first <= merged.back().second + 1) merged.back().second = std::max(merged.back().second, range.second);
        else merged.push_back(range);
    }
    return merged;
}

std::uint32_t ParseStatsInterval(const char* text) {
    if (text == nullptr || *text == '\0') return 0;
    char* end = nullptr;
    const auto value = std::strtoul(text, &end, 10);
    if (end == nullptr || *end != '\0' || value > 1000000ul) throw std::runtime_error("APS5_DRAW_PROGRAM_STATS must be a present interval between 0 and 1000000");
    return static_cast<std::uint32_t>(value);
}

bool ListedProgram(const ProgramList& list, std::uint64_t address) {
    if (address == 0 || list.empty()) return false;
    const auto after = std::upper_bound(list.begin(), list.end(), address, [](std::uint64_t value, const std::pair<std::uint64_t, std::uint64_t>& range) { return value < range.first; });
    return after != list.begin() && address <= std::prev(after)->second;
}

namespace {

const ProgramList& listFromEnvironment(const char* name) {
    static std::mutex mutex;
    static std::map<std::string, ProgramList> lists;
    std::lock_guard lock(mutex);
    const auto found = lists.find(name);
    if (found != lists.end()) return found->second;
    auto list = ParseProgramList(std::getenv(name));
    if (!list.empty()) {
        std::string text;
        for (const auto& [low, high] : list) {
            char item[48];
            if (low == high) std::snprintf(item, sizeof(item), "%s0x%llx", text.empty() ? "" : ",", static_cast<unsigned long long>(low));
            else std::snprintf(item, sizeof(item), "%s0x%llx-0x%llx", text.empty() ? "" : ",", static_cast<unsigned long long>(low), static_cast<unsigned long long>(high));
            text += item;
        }
        std::fprintf(stderr, "[progstats] %s: %zu ranges (%s)\n", name, list.size(), text.c_str());
    }
    return lists.emplace(name, std::move(list)).first->second;
}

std::string targetText(const StatsTarget& target) {
    char text[112];
    if (target.samples > 1) std::snprintf(text, sizeof(text), "0x%llx:%ux%u:f%u:s%u", static_cast<unsigned long long>(target.address), target.width, target.height, target.format, target.samples);
    else std::snprintf(text, sizeof(text), "0x%llx:%ux%u:f%u", static_cast<unsigned long long>(target.address), target.width, target.height, target.format);
    std::string result = text;
    if (target.depthMode != 0) {
        std::snprintf(text, sizeof(text), ":test%u:write%u:cmp%u", (target.depthMode >> 8u) & 1u, (target.depthMode >> 4u) & 1u, target.depthMode & 0xfu);
        result += text;
    }
    return result;
}

}

const ProgramList& SkippedPixelPrograms() {
    static const auto& list = listFromEnvironment("APS5_SKIP_PIXEL_PROGRAMS");
    return list;
}

const ProgramList& TintedPixelPrograms() {
    static const auto& list = listFromEnvironment("APS5_TINT_PIXEL_PROGRAMS");
    return list;
}

const ProgramList& DumpedDrawPrograms() {
    static const auto& list = listFromEnvironment("APS5_DUMP_DRAW_PROGRAMS");
    return list;
}

const ProgramList& ProgramFilterTargets() {
    static const auto& list = listFromEnvironment("APS5_PROGRAM_TARGETS");
    return list;
}

const ProgramList& SkippedComputePrograms() {
    static const auto& list = listFromEnvironment("APS5_SKIP_COMPUTE_PROGRAMS");
    return list;
}

std::uint32_t ProgramBisectHash(std::uint64_t address) {
    auto value = address + 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return static_cast<std::uint32_t>(value ^ (value >> 31u));
}

std::optional<std::uint32_t> BisectBitForPresent(std::uint64_t present, std::uint32_t frame, std::uint32_t bits) {
    if (present < 4) return std::nullopt;
    const auto dump = (present - 4) / 10;
    if (dump <= frame || dump > static_cast<std::uint64_t>(frame) + bits) return std::nullopt;
    return static_cast<std::uint32_t>(dump - frame - 1);
}

namespace {

std::atomic<std::uint64_t> presentCount{0};

struct BisectSetting {
    bool active = false;
    std::uint32_t frame = 0;
    std::uint32_t bits = 0;
};

const BisectSetting& bisectSetting() {
    static const BisectSetting setting = [] {
        BisectSetting parsed;
        const char* text = std::getenv("APS5_SKIP_PIXEL_BITS");
        if (text == nullptr || *text == '\0') return parsed;
        char* end = nullptr;
        parsed.frame = static_cast<std::uint32_t>(std::strtoul(text, &end, 10));
        if (end == nullptr || *end != ':') throw std::runtime_error("APS5_SKIP_PIXEL_BITS must be <frame>:<bits>");
        parsed.bits = static_cast<std::uint32_t>(std::strtoul(end + 1, &end, 10));
        if (end == nullptr || *end != '\0' || parsed.bits == 0 || parsed.bits > 32) throw std::runtime_error("APS5_SKIP_PIXEL_BITS must be <frame>:<bits> with 1 to 32 bits");
        parsed.active = true;
        std::fprintf(stderr, "[progstats] APS5_SKIP_PIXEL_BITS: frame dumps %u..%u skip hash bits 0..%u\n", parsed.frame + 1, parsed.frame + parsed.bits, parsed.bits - 1);
        return parsed;
    }();
    return setting;
}

}

void CountPresent() {
    const auto present = presentCount.fetch_add(1, std::memory_order_relaxed) + 1;
    const auto& setting = bisectSetting();
    if (!setting.active) return;
    const auto before = BisectBitForPresent(present - 1, setting.frame, setting.bits);
    const auto now = BisectBitForPresent(present, setting.frame, setting.bits);
    if (before == now) return;
    if (now) std::fprintf(stderr, "[progstats] present %llu: skipping pixel programs with hash bit %u (frame dump %u)\n", static_cast<unsigned long long>(present), *now, setting.frame + 1 + *now);
    else std::fprintf(stderr, "[progstats] present %llu: hash-bit skipping off\n", static_cast<unsigned long long>(present));
}

bool BisectSkipsProgram(std::uint64_t address) {
    const auto& setting = bisectSetting();
    if (!setting.active || address == 0) return false;
    const auto bit = BisectBitForPresent(presentCount.load(std::memory_order_relaxed), setting.frame, setting.bits);
    return bit && ((ProgramBisectHash(address) >> *bit) & 1u) != 0;
}

std::uint32_t StatsDepthMode(bool test, bool write, std::uint32_t compare) {
    return 0x80000000u | (test ? 0x100u : 0u) | (write ? 0x10u : 0u) | (compare & 0xfu);
}

std::optional<char> DepthScheduleItem() {
    struct Schedule {
        bool active = false;
        std::uint32_t frame = 0;
        std::string items;
    };
    static const Schedule schedule = [] {
        Schedule parsed;
        const char* text = std::getenv("APS5_DEPTH_SCHEDULE");
        if (text == nullptr || *text == '\0') return parsed;
        char* end = nullptr;
        parsed.frame = static_cast<std::uint32_t>(std::strtoul(text, &end, 10));
        if (end == nullptr || *end != ':') throw std::runtime_error("APS5_DEPTH_SCHEDULE must be <frame>:<item>,...");
        for (const char* cursor = end + 1; *cursor != '\0'; ++cursor) {
            if (*cursor == ',') continue;
            if (*cursor != '-' && *cursor != 'p' && (*cursor < '0' || *cursor > '7')) throw std::runtime_error("APS5_DEPTH_SCHEDULE items are '-', 'p' or a VkCompareOp digit 0-7");
            parsed.items.push_back(*cursor);
        }
        if (parsed.items.empty()) throw std::runtime_error("APS5_DEPTH_SCHEDULE has no items");
        parsed.active = true;
        std::fprintf(stderr, "[progstats] APS5_DEPTH_SCHEDULE: frame dumps %u..%zu apply '%s'\n", parsed.frame + 1, parsed.frame + parsed.items.size(), parsed.items.c_str());
        return parsed;
    }();
    if (!schedule.active) return std::nullopt;
    const auto index = BisectBitForPresent(presentCount.load(std::memory_order_relaxed), schedule.frame, static_cast<std::uint32_t>(schedule.items.size()));
    if (!index || schedule.items[*index] == '-') return std::nullopt;
    return schedule.items[*index];
}

std::vector<ProgramList> ParseVertexSchedule(const char* text, std::uint32_t& frame) {
    std::vector<ProgramList> items;
    if (text == nullptr || *text == '\0') return items;
    char* end = nullptr;
    frame = static_cast<std::uint32_t>(std::strtoul(text, &end, 10));
    if (end == nullptr || *end != ':') throw std::runtime_error("APS5_SKIP_VERTEX_SCHEDULE must be <frame>:<list>;<list>;...");
    std::string rest(end + 1);
    std::size_t begin = 0;
    while (begin <= rest.size()) {
        auto stop = rest.find(';', begin);
        if (stop == std::string::npos) stop = rest.size();
        const auto item = rest.substr(begin, stop - begin);
        items.push_back(item == "-" || item.empty() ? ProgramList{} : ParseProgramList(item.c_str()));
        begin = stop + 1;
    }
    if (items.empty() || items.size() > 64) throw std::runtime_error("APS5_SKIP_VERTEX_SCHEDULE needs 1 to 64 items");
    return items;
}

bool VertexScheduleSkips(std::uint64_t vertex) {
    static std::uint32_t frame = 0;
    static const auto items = [] {
        auto parsed = ParseVertexSchedule(std::getenv("APS5_SKIP_VERTEX_SCHEDULE"), frame);
        if (!parsed.empty()) std::fprintf(stderr, "[progstats] APS5_SKIP_VERTEX_SCHEDULE: frame dumps %u..%zu skip listed pre-pass vertex programs\n", frame + 1, frame + parsed.size());
        return parsed;
    }();
    if (items.empty() || vertex == 0) return false;
    const auto index = BisectBitForPresent(presentCount.load(std::memory_order_relaxed), frame, static_cast<std::uint32_t>(items.size()));
    return index && ListedProgram(items[*index], vertex);
}

bool BlendableAttachmentFormat(VkFormat format) {
    switch (format) {
        case VK_FORMAT_UNDEFINED:
        case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R8G8B8_UINT: case VK_FORMAT_R8G8B8_SINT: case VK_FORMAT_B8G8R8_UINT: case VK_FORMAT_B8G8R8_SINT:
        case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_B8G8R8A8_SINT:
        case VK_FORMAT_A8B8G8R8_UINT_PACK32: case VK_FORMAT_A8B8G8R8_SINT_PACK32:
        case VK_FORMAT_A2R10G10B10_UINT_PACK32: case VK_FORMAT_A2R10G10B10_SINT_PACK32:
        case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_A2B10G10R10_SINT_PACK32:
        case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT:
        case VK_FORMAT_R16G16B16_UINT: case VK_FORMAT_R16G16B16_SINT: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32_SINT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT:
        case VK_FORMAT_R64_UINT: case VK_FORMAT_R64_SINT: case VK_FORMAT_R64G64_UINT: case VK_FORMAT_R64G64_SINT:
        case VK_FORMAT_R64G64B64_UINT: case VK_FORMAT_R64G64B64_SINT: case VK_FORMAT_R64G64B64A64_UINT: case VK_FORMAT_R64G64B64A64_SINT:
        case VK_FORMAT_R64_SFLOAT: case VK_FORMAT_R64G64_SFLOAT: case VK_FORMAT_R64G64B64_SFLOAT: case VK_FORMAT_R64G64B64A64_SFLOAT:
            return false;
        default:
            return true;
    }
}

std::size_t ApplyBlendTint(std::span<VkPipelineColorBlendAttachmentState> blends, std::array<float, 4>& constants, std::span<const VkFormat> formats) {
    std::size_t tinted = 0;
    for (std::size_t i = 0; i < blends.size(); ++i) {
        auto& blend = blends[i];
        if (blend.colorWriteMask == 0 || i >= formats.size() || !BlendableAttachmentFormat(formats[i])) continue;
        blend.blendEnable = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        blend.colorBlendOp = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blend.alphaBlendOp = VK_BLEND_OP_ADD;
        ++tinted;
    }
    if (tinted != 0) constants = {1.0f, 0.0f, 1.0f, 1.0f};
    return tinted;
}

void ProgramStats::NoteDraw(std::uint64_t pixel, std::uint64_t vertex, std::span<const StatsTarget> colors, std::optional<StatsTarget> depth, bool skipped, bool tinted) {
    if (!Enabled()) return;
    Group group{{colors.begin(), colors.end()}, depth};
    std::lock_guard lock(mutex);
    auto& count = groups[group][{pixel, vertex}];
    ++count.draws;
    if (skipped) ++count.skipped;
    if (tinted) ++count.tinted;
    ++draws;
}

void ProgramStats::NoteDrawSampled(std::uint64_t pixel, std::span<const std::uint64_t> addresses) {
    if (!Enabled() || addresses.empty()) return;
    std::lock_guard lock(mutex);
    for (const auto address : addresses) {
        auto& programs = sampled[address];
        if (programs.size() < 8) programs.insert(pixel);
    }
}

void ProgramStats::NoteMesh(std::uint64_t indexAddress, std::uint32_t indexCount, std::uint32_t firstVertex, std::uint32_t instances, std::uint64_t depth, std::uint64_t vertex, std::uint32_t depthMode, std::uint64_t color) {
    if (!Enabled() || indexAddress == 0) return;
    std::lock_guard lock(mutex);
    auto& users = meshes[{indexAddress, indexCount, firstVertex, instances, depth}];
    if (users.size() < 8) users.insert({vertex, depthMode, color});
}

void ProgramStats::NoteDispatch(std::uint64_t program, std::span<const StatsImage> written, bool skipped) {
    if (!Enabled()) return;
    std::lock_guard lock(mutex);
    auto& count = compute[program];
    ++count.dispatches;
    if (skipped) ++count.skipped;
    for (const auto& image : written) {
        if (count.written.size() < 16) count.written.insert(image);
    }
}

std::vector<std::string> ProgramStats::Present() {
    std::vector<std::string> lines;
    if (!Enabled()) return lines;
    std::lock_guard lock(mutex);
    ++presents;
    if (presents % every == 0) {
        std::uint64_t dispatches = 0;
        for (const auto& [program, count] : compute) dispatches += count.dispatches;
        char head[192];
        std::snprintf(head, sizeof(head), "[progstats] present %llu: %llu draws into %zu attachment groups, %llu dispatches of %zu compute programs",
            static_cast<unsigned long long>(presents), static_cast<unsigned long long>(draws), groups.size(), static_cast<unsigned long long>(dispatches), compute.size());
        lines.emplace_back(head);
        std::vector<std::pair<std::uint64_t, const decltype(groups)::value_type*>> ordered;
        for (const auto& entry : groups) {
            std::uint64_t total = 0;
            for (const auto& [programs, count] : entry.second) total += count.draws;
            ordered.emplace_back(total, &entry);
        }
        std::stable_sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        constexpr std::size_t MaxGroups = 96;
        constexpr std::size_t MaxPrograms = 256;
        for (std::size_t g = 0; g < ordered.size() && g < MaxGroups; ++g) {
            const auto& [group, programs] = *ordered[g].second;
            std::string line = "[progstats] present " + std::to_string(presents) + " color";
            if (group.colors.empty()) line += " none";
            for (const auto& color : group.colors) line += " " + targetText(color);
            if (group.depth) line += " depth " + targetText(*group.depth);
            line += " | " + std::to_string(ordered[g].first) + " draws:";
            std::vector<std::pair<std::pair<std::uint64_t, std::uint64_t>, PixelCount>> pixels(programs.begin(), programs.end());
            std::stable_sort(pixels.begin(), pixels.end(), [](const auto& a, const auto& b) { return a.second.draws > b.second.draws; });
            for (std::size_t p = 0; p < pixels.size() && p < MaxPrograms; ++p) {
                const auto& [key, count] = pixels[p];
                char item[160];
                std::snprintf(item, sizeof(item), " ps 0x%llx (vs 0x%llx) x%llu", static_cast<unsigned long long>(key.first), static_cast<unsigned long long>(key.second), static_cast<unsigned long long>(count.draws));
                line += item;
                if (count.skipped != 0) line += " skipped " + std::to_string(count.skipped);
                if (count.tinted != 0) line += " tinted " + std::to_string(count.tinted);
                line += p + 1 < pixels.size() && p + 1 < MaxPrograms ? "," : "";
            }
            if (pixels.size() > MaxPrograms) line += " (+" + std::to_string(pixels.size() - MaxPrograms) + " more programs)";
            lines.push_back(std::move(line));
        }
        if (ordered.size() > MaxGroups) lines.push_back("[progstats] present " + std::to_string(presents) + " (+" + std::to_string(ordered.size() - MaxGroups) + " more attachment groups)");
        std::vector<std::pair<std::uint64_t, const ComputeCount*>> programs;
        for (const auto& [program, count] : compute) programs.emplace_back(program, &count);
        std::stable_sort(programs.begin(), programs.end(), [](const auto& a, const auto& b) { return a.second->dispatches > b.second->dispatches; });
        std::string plain = "[progstats] present " + std::to_string(presents) + " compute without image writes seen:";
        bool anyPlain = false;
        for (const auto& [program, count] : programs) {
            char item[96];
            std::snprintf(item, sizeof(item), " 0x%llx x%llu", static_cast<unsigned long long>(program), static_cast<unsigned long long>(count->dispatches));
            std::string text = item;
            if (count->skipped != 0) text += " skipped " + std::to_string(count->skipped);
            if (count->written.empty()) {
                plain += text;
                anyPlain = true;
                continue;
            }
            std::string line = "[progstats] present " + std::to_string(presents) + " compute" + text + " writes";
            for (const auto& image : count->written) {
                line += " " + targetText({image.address, image.width, image.height, image.format, 1});
                if (const auto found = sampled.find(image.address); found != sampled.end()) {
                    line += " (sampled by ps";
                    for (const auto pixel : found->second) {
                        char ps[24];
                        std::snprintf(ps, sizeof(ps), " 0x%llx", static_cast<unsigned long long>(pixel));
                        line += ps;
                    }
                    line += ")";
                }
            }
            lines.push_back(std::move(line));
        }
        if (anyPlain) lines.push_back(std::move(plain));
        std::map<std::string, std::uint64_t> sets;
        for (const auto& [mesh, users] : meshes) {
            std::set<std::uint64_t> programs;
            for (const auto& user : users) programs.insert(std::get<0>(user));
            if (programs.size() < 2) continue;
            char depthText[40];
            std::snprintf(depthText, sizeof(depthText), "depth 0x%llx: ", static_cast<unsigned long long>(std::get<4>(mesh)));
            std::string text = depthText;
            for (const auto& [vertex, mode, color] : users) {
                char item[96];
                std::snprintf(item, sizeof(item), "%svs 0x%llx@0x%llx:test%u:write%u:cmp%u", text.back() == ' ' ? "" : " + ", static_cast<unsigned long long>(vertex), static_cast<unsigned long long>(color), (mode >> 8u) & 1u, (mode >> 4u) & 1u, mode & 0xfu);
                text += item;
            }
            ++sets[text];
        }
        std::vector<std::pair<std::string, std::uint64_t>> orderedSets(sets.begin(), sets.end());
        std::stable_sort(orderedSets.begin(), orderedSets.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        for (std::size_t i = 0; i < orderedSets.size() && i < 160; ++i) lines.push_back("[progstats] present " + std::to_string(presents) + " mesh x" + std::to_string(orderedSets[i].second) + ": " + orderedSets[i].first);
    }
    groups.clear();
    compute.clear();
    sampled.clear();
    meshes.clear();
    draws = 0;
    return lines;
}

ProgramStats& ProgramStats::Global() {
    static ProgramStats stats(ParseStatsInterval(std::getenv("APS5_DRAW_PROGRAM_STATS")));
    return stats;
}

}
