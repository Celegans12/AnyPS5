#include "prx/libSceAgcDriver/Graphics/include/ResidentCensus.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string_view>

namespace AgcDriver::Graphics::ResidentCensus {

namespace {

constexpr std::uint64_t firstPage(std::uint64_t begin) { return begin / PageBytes; }
constexpr std::uint64_t endPage(std::uint64_t end) { return (end + PageBytes - 1u) / PageBytes; }
constexpr std::size_t ClassCount = static_cast<std::size_t>(Class::Count);
// Ranges larger than this are not a buffer the GPU reads (a garbage V# or size): not counted.
constexpr std::uint64_t MaxRangeBytes = 16ull << 30u;

double mib(std::uint64_t pages) {
    return static_cast<double>(pages * PageBytes) / (1024.0 * 1024.0);
}

struct State {
    std::mutex mutex;
    Model model;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

State& Get() {
    static State state;
    return state;
}

void print(State& state, const char* window) {
    const auto line = Model::Format(window, state.model.Close());
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
}

void maybeReport(State& state) {
    if (ProfileWindows::ClockDue(state.lastReport)) print(state, "10 s");
}

}

bool Model::Lru::Touch(std::uint64_t page) {
    if (const auto found = where.find(page); found != where.end()) {
        order.splice(order.begin(), order, found->second);
        return false;
    }
    order.push_front(page);
    where.emplace(page, order.begin());
    if (order.size() <= capacity) return false;
    where.erase(order.back());
    order.pop_back();
    return true;
}

void Model::gpu(Class which, std::uint64_t begin, std::uint64_t end, bool written) {
    if (end <= begin || end - begin > MaxRangeBytes) return;
    auto& range = ranges[{firstPage(begin), endPage(end), static_cast<std::uint64_t>(which), written ? 1u : 0u}];
    ++seq;
    if (range.firstSeq == 0) range.firstSeq = seq;
    range.lastSeq = seq;
}

void Model::GpuRead(Class which, std::uint64_t begin, std::uint64_t end) {
    gpu(which, begin, end, false);
}

void Model::GpuWrite(Class which, std::uint64_t begin, std::uint64_t end) {
    gpu(which, begin, end, true);
}

void Model::CpuWrite(std::uint64_t begin, std::uint64_t end) {
    if (end <= begin) return;
    ++seq;
    cpuWrites.try_emplace({firstPage(begin), endPage(end)}, seq);
}

bool Model::GroupSeen(const void* key) {
    const auto found = groups.find(key);
    if (found == groups.end()) return false;
    found->second.seqs.lastSeq = ++seq;
    return true;
}

void Model::GpuReadGroup(const void* key, Class which, std::span<const std::pair<std::uint64_t, std::uint64_t>> read) {
    auto& group = groups[key];
    group.which = which;
    group.ranges.assign(read.begin(), read.end());
    group.seqs.firstSeq = group.seqs.lastSeq = ++seq;
}

Model::WindowResult Model::Close() {
    struct Page {
        std::uint32_t classes = 0;
        bool written = false;
        std::uint64_t lastSeq = 0;
        std::uint64_t lastRead = 0;
        std::uint64_t firstCpu = std::numeric_limits<std::uint64_t>::max();
    };
    std::unordered_map<std::uint64_t, Page> pages;
    const auto add = [&](std::uint64_t first, std::uint64_t end, Class which, bool written, const Range& seqs) {
        for (auto page = first; page < end; ++page) {
            auto& info = pages[page];
            info.classes |= 1u << static_cast<unsigned>(which);
            info.lastSeq = std::max(info.lastSeq, seqs.lastSeq);
            if (written) info.written = true;
            else info.lastRead = std::max(info.lastRead, seqs.lastSeq);
        }
    };
    for (const auto& [key, seqs] : ranges) add(key[0], key[1], static_cast<Class>(key[2]), key[3] != 0, seqs);
    for (const auto& [key, group] : groups) {
        for (const auto& [begin, end] : group.ranges) {
            if (end > begin && end - begin <= MaxRangeBytes) add(firstPage(begin), endPage(end), group.which, false, group.seqs);
        }
    }
    for (const auto& [key, cpuSeq] : cpuWrites) {
        for (auto page = key.first; page < key.second; ++page) {
            if (const auto found = pages.find(page); found != pages.end()) found->second.firstCpu = std::min(found->second.firstCpu, cpuSeq);
        }
    }
    WindowResult result;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> order;
    order.reserve(pages.size());
    for (const auto& [page, info] : pages) {
        const bool cpuThenGpu = !info.written && info.firstCpu < info.lastRead;
        const auto count = [&](Split& split) { ++(info.written ? split.written : cpuThenGpu ? split.cpuThenGpu : split.readOnly); };
        count(result.total);
        for (std::size_t which = 0; which < ClassCount; ++which) {
            if ((info.classes & (1u << which)) != 0) count(result.byClass[which]);
        }
        auto& seen = unionPages[page];
        seen |= (info.written ? 1u : cpuThenGpu ? 2u : 4u) | (info.classes << 8u);
        order.emplace_back(info.lastSeq, page);
    }
    std::sort(order.begin(), order.end());
    for (std::size_t index = 0; index < lrus.size(); ++index) {
        for (const auto& entry : order) result.evicted[index] += lrus[index].Touch(entry.second) ? 1u : 0u;
    }
    for (const auto& [page, bits] : unionPages) ++((bits & 1u) != 0 ? result.unionTotal.written : (bits & 2u) != 0 ? result.unionTotal.cpuThenGpu : result.unionTotal.readOnly);
    ranges.clear();
    cpuWrites.clear();
    groups.clear();
    return result;
}

std::string Model::Format(const char* window, const WindowResult& result) {
    const auto split = [](const Split& value) {
        char text[160];
        std::snprintf(text, sizeof(text), "%llu pages = %.1f MiB (gpu-written %.1f, cpu-then-gpu %.1f, read-only %.1f)", static_cast<unsigned long long>(value.Pages()), mib(value.Pages()), mib(value.written), mib(value.cpuThenGpu), mib(value.readOnly));
        return std::string(text);
    };
    std::string line = "[resident-census] window=" + std::string(window) + " " + split(result.total) + "; by class written/cpu-then-gpu/read-only MiB:";
    for (std::size_t which = 0; which < ClassCount; ++which) {
        char text[96];
        const auto& value = result.byClass[which];
        std::snprintf(text, sizeof(text), "%s %s %.1f/%.1f/%.1f", which == 0 ? "" : ",", ClassNames[which], mib(value.written), mib(value.cpuThenGpu), mib(value.readOnly));
        line += text;
    }
    line += "; lru evictions (pages)";
    for (std::size_t index = 0; index < LruMiB.size(); ++index) {
        line += std::string(index == 0 ? " " : ", ") + std::to_string(LruMiB[index]) + " MiB " + std::to_string(result.evicted[index]);
    }
    line += "; union " + split(result.unionTotal);
    return line;
}

bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_PROFILE_RESIDENT_CENSUS");
        return value != nullptr && value[0] != '\0' && std::string_view(value) != "0";
    }();
    return enabled;
}

void NoteGpuRead(Class which, std::uint64_t address, std::uint64_t bytes) {
    if (!Enabled() || bytes == 0) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    state.model.GpuRead(which, address, address + bytes);
    maybeReport(state);
}

void NoteGpuWrite(Class which, std::uint64_t address, std::uint64_t bytes) {
    if (!Enabled() || bytes == 0) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    state.model.GpuWrite(which, address, address + bytes);
    maybeReport(state);
}

bool NoteGroupSeen(const void* key) {
    if (!Enabled()) return true;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    return state.model.GroupSeen(key);
}

void NoteGpuReadGroup(const void* key, Class which, std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    if (!Enabled()) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    state.model.GpuReadGroup(key, which, ranges);
    maybeReport(state);
}

void NoteCpuWrite(std::uint64_t address, std::uint64_t bytes) {
    if (!Enabled() || bytes == 0) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    state.model.CpuWrite(address, address + bytes);
}

void NoteVertexBuffer(std::span<const std::uint32_t, 4> vsharp) {
    if (!Enabled()) return;
    // Buffer V#: base address bits 0-47 (dword 0, dword 1 bits 0-15), stride dword 1 bits 16-29,
    // records dword 2 (bytes when the stride is 0).
    const auto base = static_cast<std::uint64_t>(vsharp[0]) | (static_cast<std::uint64_t>(vsharp[1] & 0xffffu) << 32u);
    const auto stride = static_cast<std::uint64_t>((vsharp[1] >> 16u) & 0x3fffu);
    const auto records = static_cast<std::uint64_t>(vsharp[2]);
    const auto bytes = stride != 0 ? records * stride : records;
    if (base == 0 || bytes == 0) return;
    NoteGpuRead(Class::VertexIndex, base, bytes);
}

void NoteDrawInputs(const Pm4::DrawParameters& draw, std::span<const CompiledShader> stages) {
    if (!Enabled()) return;
    if (draw.indexed && draw.indexAddress != 0) NoteGpuRead(Class::VertexIndex, draw.indexAddress, static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize);
    if (draw.indirect) {
        NoteGpuRead(Class::Indirect, draw.indirect->arguments, std::max<std::uint64_t>(draw.indirect->RangeBytes(), draw.indirect->recordBytes));
        if (draw.indirect->countIndirect) NoteGpuRead(Class::Indirect, draw.indirect->countAddress, 4);
    }
    for (const auto& stage : stages) {
        if (stage.program == nullptr) continue;
        for (const auto& attribute : stage.program->vertexAttributes) NoteVertexBuffer(std::span<const std::uint32_t, 4>(attribute.resource.fields));
    }
}

void Close(const char* window) {
    if (!Enabled()) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    print(state, window);
}

}
