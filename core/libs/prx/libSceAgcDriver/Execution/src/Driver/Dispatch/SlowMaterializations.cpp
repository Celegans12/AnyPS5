// APS5_MATERIALIZE_STATS=1 (diagnostic): every 10 s, the dispatch captures + materializations that took
// at least APS5_MATERIALIZE_STATS_MS (default 10) ms, by program, and how each one's resource snapshot
// compares with the earlier captures of the same program: whether its image tables equal the previous
// capture's or one of the last few (the recompiler's table resolve memo keeps 8 per variant), how many
// table keys and distinct descriptor words differ, and whether the rest of the snapshot (buffers, images,
// samplers, user data, flattened SRT) is equal. Reads only; it changes nothing that is dispatched.
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/SlowMaterializations.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::size_t RecentCaptures = 8;

struct ProgramStats {
    std::uint64_t count = 0;
    std::uint64_t ahead = 0;
    std::uint64_t cacheHits = 0;
    double captureMs = 0;
    double materializeMs = 0;
    double maxMs = 0;
    std::size_t words = 0;
    std::size_t tables = 0;
    std::size_t keys = 0;
    std::size_t distinct = 0;
    std::uint64_t compared = 0;
    std::uint64_t tablesEqualPrevious = 0;
    std::uint64_t tablesEqualRecent = 0;
    std::uint64_t restEqualPrevious = 0;
    std::uint64_t layoutChanged = 0;
    std::uint64_t keysDiffering = 0;
    std::uint64_t wordsDiffering = 0;
    std::uint64_t wordsAdded = 0;
    std::set<std::uint32_t> queues;
};

struct State {
    std::mutex mutex;
    std::map<std::uint64_t, ProgramStats> programs;
    std::map<std::uint64_t, std::deque<std::shared_ptr<const ShaderRecompiler::ResourceCapture>>> recent;
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};

State& state() {
    static State instance;
    return instance;
}

double ThresholdMs() {
    static const double threshold = [] {
        const char* text = std::getenv("APS5_MATERIALIZE_STATS_MS");
        return text != nullptr && *text != '\0' ? std::strtod(text, nullptr) : 10.0;
    }();
    return threshold;
}

bool RestEqual(const ShaderRecompiler::ResourceSnapshot& left, const ShaderRecompiler::ResourceSnapshot& right) {
    return left.buffers == right.buffers && left.images == right.images && left.samplers == right.samplers && left.userData == right.userData && left.flattenedSrt == right.flattenedSrt && left.uniformFill == right.uniformFill && left.nullRootReads == right.nullRootReads;
}

void Compare(ProgramStats& stats, const ShaderRecompiler::ImageTableSnapshot& now, const ShaderRecompiler::ImageTableSnapshot& before) {
    if (now.tables.size() != before.tables.size()) {
        ++stats.layoutChanged;
        return;
    }
    bool layout = false;
    for (std::size_t table = 0; table < now.tables.size(); ++table) {
        const auto& a = now.tables[table];
        const auto& b = before.tables[table];
        if (a.base != b.base || a.size != b.size || a.records != b.records || a.keys != b.keys || a.codes.size() != b.codes.size() || a.addresses != b.addresses || a.readRanges != b.readRanges || a.fault != b.fault) layout = true;
        const auto shared = std::min(a.codes.size(), b.codes.size());
        for (std::size_t key = 0; key < shared; ++key) {
            if (a.codes[key] != b.codes[key]) ++stats.keysDiffering;
        }
    }
    if (layout || now.ranges != before.ranges) ++stats.layoutChanged;
    const auto shared = std::min(now.words.size(), before.words.size());
    for (std::size_t word = 0; word < shared; ++word) {
        if (!(now.words[word] == before.words[word])) ++stats.wordsDiffering;
    }
    if (now.words.size() > before.words.size()) stats.wordsAdded += now.words.size() - before.words.size();
}

void Report(State& current) {
    std::vector<std::pair<std::uint64_t, ProgramStats>> ranked(current.programs.begin(), current.programs.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) { return left.second.captureMs + left.second.materializeMs > right.second.captureMs + right.second.materializeMs; });
    std::uint64_t count = 0;
    double total = 0;
    for (const auto& [address, stats] : ranked) {
        count += stats.count;
        total += stats.captureMs + stats.materializeMs;
    }
    std::string text;
    char line[768];
    std::snprintf(line, sizeof(line), "[materialize] (10 s): %llu dispatch captures + materializations of at least %.0f ms, %.1f ms in total, %zu programs", static_cast<unsigned long long>(count), ThresholdMs(), total, ranked.size());
    text += line;
    for (std::size_t index = 0; index < ranked.size() && index < 8; ++index) {
        const auto& [address, stats] = ranked[index];
        std::string queues;
        for (const auto queue : stats.queues) {
            char name[16];
            std::snprintf(name, sizeof(name), "%s0x%x", queues.empty() ? "" : "/", queue);
            queues += name;
        }
        std::snprintf(line, sizeof(line), "; 0x%llx q%s x%llu (%llu ahead, %llu cache hits): capture %.1f + materialize %.1f ms (max %.1f), %zu words, %zu tables %zu keys %zu distinct; of %llu compared with the previous capture: tables equal %llu, equal to one of the last %zu %llu, rest equal %llu, layout changed %llu, keys differing %llu, distinct words differing %llu, added %llu",
            static_cast<unsigned long long>(address), queues.c_str(), static_cast<unsigned long long>(stats.count), static_cast<unsigned long long>(stats.ahead), static_cast<unsigned long long>(stats.cacheHits), stats.captureMs, stats.materializeMs, stats.maxMs, stats.words, stats.tables, stats.keys, stats.distinct,
            static_cast<unsigned long long>(stats.compared), static_cast<unsigned long long>(stats.tablesEqualPrevious), RecentCaptures, static_cast<unsigned long long>(stats.tablesEqualRecent), static_cast<unsigned long long>(stats.restEqualPrevious), static_cast<unsigned long long>(stats.layoutChanged), static_cast<unsigned long long>(stats.keysDiffering), static_cast<unsigned long long>(stats.wordsDiffering), static_cast<unsigned long long>(stats.wordsAdded));
        text += line;
    }
    text += "\n";
    AgcDriver::ProfilePrint_nid_no_patch("%s", text.c_str());
    current.programs.clear();
}

}

bool SlowMaterializationStats() {
    static const bool enabled = std::getenv("APS5_MATERIALIZE_STATS") != nullptr;
    return enabled;
}

void NoteSlowMaterialization(std::uint32_t queue, std::uint64_t address, const ShaderRecompiler::RecompileResult& result, const std::shared_ptr<const ShaderRecompiler::ResourceCapture>& capture, double captureMs, double materializeMs, bool ahead) {
    if (!SlowMaterializationStats() || capture == nullptr || captureMs + materializeMs < ThresholdMs()) return;
    auto& current = state();
    std::lock_guard lock(current.mutex);
    auto& stats = current.programs[address];
    ++stats.count;
    if (ahead) ++stats.ahead;
    if (result.cacheHit) ++stats.cacheHits;
    stats.captureMs += captureMs;
    stats.materializeMs += materializeMs;
    stats.maxMs = std::max(stats.maxMs, captureMs + materializeMs);
    stats.words = result.spirv.size();
    const auto& tables = capture->snapshot.tables;
    stats.tables = tables.tables.size();
    stats.keys = 0;
    for (const auto& table : tables.tables) stats.keys += table.codes.size();
    stats.distinct = tables.words.size();
    stats.queues.insert(queue);
    auto& history = current.recent[address];
    if (!history.empty()) {
        ++stats.compared;
        const auto& previous = history.back()->snapshot;
        if (tables == previous.tables) ++stats.tablesEqualPrevious;
        if (std::any_of(history.begin(), history.end(), [&](const auto& earlier) { return earlier->snapshot.tables == tables; })) ++stats.tablesEqualRecent;
        if (RestEqual(capture->snapshot, previous)) ++stats.restEqualPrevious;
        Compare(stats, tables, previous.tables);
    }
    history.push_back(capture);
    while (history.size() > RecentCaptures) history.pop_front();
    const auto now = std::chrono::steady_clock::now();
    if (now - current.last >= std::chrono::seconds(10)) {
        current.last = now;
        Report(current);
    }
}

}
