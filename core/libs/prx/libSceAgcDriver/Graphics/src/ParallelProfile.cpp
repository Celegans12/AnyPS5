#include "prx/libSceAgcDriver/Graphics/include/ParallelProfile.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/resource.h>
#include <time.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace AgcDriver::Graphics::ParallelProfile {

namespace {

constexpr std::size_t MaxPhases = 16;
constexpr std::size_t RunBuckets = 66;
constexpr std::array<const char*, static_cast<std::size_t>(Stamp::Count)> StampNames{"forget", "registry", "device", "epoch", "proof", "pending", "gpu-write", "cpu-write"};
constexpr std::array<const char*, static_cast<std::size_t>(Item::Count)> ItemNames{"draw", "label", "other"};
constexpr std::uint32_t SurvivalStamps = StampBit(Stamp::Forget) | StampBit(Stamp::Registry) | StampBit(Stamp::Device) | StampBit(Stamp::Pending) | StampBit(Stamp::GpuWrite) | StampBit(Stamp::CpuWrite);

struct ItemTotals {
    std::uint64_t count = 0;
    double wallUs = 0;
    double heldUs = 0;
    double lockedWaitUs = 0;
    std::uint64_t preDrained = 0;
    double preDrainUs = 0;
    std::uint64_t stageA = 0;
};

struct WindowTotals {
    std::uint64_t checked = 0;
    std::uint64_t survived = 0;
    std::uint64_t survivedWithProof = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(Stamp::Count)> moved{};
};

struct State {
    std::array<double, MaxPhases> phaseUs{};
    std::array<double, MaxPhases> movable{};
    std::array<const char*, MaxPhases> names{};
    std::size_t phases = 0;
    std::uint64_t draws = 0;
    std::array<ItemTotals, static_cast<std::size_t>(Item::Count)> items{};
    std::array<WindowTotals, static_cast<std::size_t>(Window::Count)> windows{};
    std::uint64_t passDraws = 0;
    std::uint64_t serialRun = 0;
    std::uint64_t itemRun = 0;
    std::array<std::uint64_t, RunBuckets> passLengths{};
    std::array<std::uint64_t, RunBuckets> serialRuns{};
    std::array<std::uint64_t, RunBuckets> itemRuns{};
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    double processCpuMs = 0;
    double committerCpuMs = 0;
    GuestMemory::TrackerLockStats tracker{};
    bool primed = false;
};

State& state() {
    static auto* created = new State();
    return *created;
}

thread_local bool committerThread = false;

void addRun(std::array<std::uint64_t, RunBuckets>& histogram, std::uint64_t& run) {
    if (run == 0) return;
    ++histogram[std::min<std::uint64_t>(run, RunBuckets - 1)];
    run = 0;
}

#ifdef _WIN32
double fileTimeMs(const FILETIME& time) {
    return static_cast<double>((static_cast<std::uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime) / 1e4;
}
#endif

double processCpuMs() {
#ifdef _WIN32
    FILETIME created{}, exited{}, kernel{}, user{};
    if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user) == 0) return 0;
    return fileTimeMs(kernel) + fileTimeMs(user);
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
    return (static_cast<double>(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1e3) + static_cast<double>(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1e3;
#endif
}

double threadCpuMs() {
#ifdef _WIN32
    FILETIME created{}, exited{}, kernel{}, user{};
    if (GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user) == 0) return 0;
    return fileTimeMs(kernel) + fileTimeMs(user);
#else
    timespec now{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now) != 0) return 0;
    return static_cast<double>(now.tv_sec) * 1e3 + static_cast<double>(now.tv_nsec) / 1e6;
#endif
}

}

bool On() {
    static const bool enabled = std::getenv("APS5_PROFILE_PARALLEL") != nullptr;
    return enabled;
}

void MarkCommitter() {
    committerThread = true;
}

bool OnCommitter() {
    return committerThread;
}

double Percentile(std::span<const std::uint64_t> histogram, double fraction) {
    std::uint64_t total = 0;
    for (const auto count : histogram) total += count;
    if (total == 0) return 0;
    const auto target = static_cast<double>(total) * fraction;
    std::uint64_t seen = 0;
    for (std::size_t index = 0; index < histogram.size(); ++index) {
        seen += histogram[index];
        if (static_cast<double>(seen) >= target) return static_cast<double>(index);
    }
    return static_cast<double>(histogram.size() - 1);
}

GateVerdict Gate(const GateInputs& inputs) {
    return {inputs.movableShare >= 0.40, inputs.survival >= 0.80, inputs.chunkP50 >= 8.0};
}

void NoteDraw(std::span<const double> phaseUs, std::span<const double> movable, std::span<const char* const> names) {
    if (!On() || !committerThread) return;
    auto& s = state();
    s.phases = std::min({phaseUs.size(), movable.size(), names.size(), MaxPhases});
    for (std::size_t index = 0; index < s.phases; ++index) {
        s.phaseUs[index] += phaseUs[index];
        s.movable[index] = movable[index];
        s.names[index] = names[index];
    }
    ++s.draws;
}

void NotePassDraw(bool continued) {
    if (!On() || !committerThread) return;
    auto& s = state();
    if (!continued) addRun(s.passLengths, s.passDraws);
    ++s.passDraws;
    ++s.serialRun;
    ++s.itemRun;
}

void NoteSerialWork() {
    if (!On() || !committerThread) return;
    addRun(state().serialRuns, state().serialRun);
}

void NoteStamps(Window window, std::uint32_t moved) {
    if (!On() || !committerThread) return;
    auto& totals = state().windows[static_cast<std::size_t>(window)];
    ++totals.checked;
    if ((moved & SurvivalStamps) == 0) ++totals.survived;
    if ((moved & (SurvivalStamps | StampBit(Stamp::Proof))) == 0) ++totals.survivedWithProof;
    for (std::size_t index = 0; index < totals.moved.size(); ++index) {
        if ((moved & (1u << index)) != 0) ++totals.moved[index];
    }
}

void NoteItem(const ItemSample& sample) {
    if (!On() || !committerThread) return;
    auto& s = state();
    auto& totals = s.items[static_cast<std::size_t>(sample.kind)];
    ++totals.count;
    totals.wallUs += sample.wallUs;
    totals.heldUs += sample.heldUs;
    totals.lockedWaitUs += sample.lockedWaitUs;
    totals.preDrained += sample.preDrained ? 1u : 0u;
    totals.preDrainUs += sample.preDrainUs;
    totals.stageA += sample.stageA ? 1u : 0u;
    if (sample.kind != Item::Draw) {
        addRun(s.serialRuns, s.serialRun);
        addRun(s.itemRuns, s.itemRun);
    }
    Report();
}

void Report() {
    if (!On() || !committerThread) return;
    auto& s = state();
    const auto now = std::chrono::steady_clock::now();
    if (!s.primed) {
        s.primed = true;
        s.last = now;
        s.processCpuMs = processCpuMs();
        s.committerCpuMs = threadCpuMs();
        s.tracker = GuestMemory::TrackerLockCounts();
        return;
    }
    const auto seconds = std::chrono::duration<double>(now - s.last).count();
    if (seconds < 10.0) return;
    const auto& draw = s.items[static_cast<std::size_t>(Item::Draw)];
    const double commits = static_cast<double>(std::max<std::uint64_t>(draw.count, 1));
    const double heldPer = draw.heldUs / commits;
    double phaseSum = 0;
    double movableSum = 0;
    double recordUs = 0;
    std::string phases;
    for (std::size_t index = 0; index < s.phases; ++index) {
        const double per = s.phaseUs[index] / commits;
        phaseSum += per;
        movableSum += per * s.movable[index];
        if (std::string_view(s.names[index]) == "record") recordUs = per;
        if (per < 0.05) continue;
        char text[64];
        std::snprintf(text, sizeof(text), " %s %.1f/%.1f", s.names[index], per, per * s.movable[index]);
        phases += text;
    }
    const double lockedWaitPer = draw.lockedWaitUs / commits;
    const double outside = std::max(0.0, heldPer - phaseSum);
    const double outsideMovable = std::max(0.0, outside - lockedWaitPer) * 0.10;
    movableSum += outsideMovable;
    const double share = heldPer > 0 ? movableSum / heldPer : 0;
    const double shareChunked = heldPer > 0 ? (movableSum + recordUs * 0.8) / heldPer : 0;
    std::string others;
    for (std::size_t kind = 1; kind < s.items.size(); ++kind) {
        const auto& item = s.items[kind];
        if (item.count == 0) continue;
        char text[192];
        std::snprintf(text, sizeof(text), " %s %llu held %.0f ms (locked GPU wait %.0f ms, pre-drained %llu waiting %.0f ms)", ItemNames[kind], static_cast<unsigned long long>(item.count), item.heldUs / 1e3, item.lockedWaitUs / 1e3, static_cast<unsigned long long>(item.preDrained), item.preDrainUs / 1e3);
        others += text;
    }
    std::string windows;
    double queueSurvival = 0;
    for (std::size_t window = 0; window < s.windows.size(); ++window) {
        const auto& totals = s.windows[window];
        const double checked = static_cast<double>(std::max<std::uint64_t>(totals.checked, 1));
        if (window == static_cast<std::size_t>(Window::Queue)) queueSurvival = totals.checked != 0 ? static_cast<double>(totals.survived) / checked : 0;
        char text[96];
        std::snprintf(text, sizeof(text), "; stamps moved %s (of %llu):", window == 0 ? "enqueue->commit" : "commit->lock", static_cast<unsigned long long>(totals.checked));
        windows += text;
        for (std::size_t stamp = 0; stamp < totals.moved.size(); ++stamp) {
            std::snprintf(text, sizeof(text), " %s %llu", StampNames[stamp], static_cast<unsigned long long>(totals.moved[stamp]));
            windows += text;
        }
        std::snprintf(text, sizeof(text), ", survival %.1f%% (%.1f%% with the global proof serial)", 100.0 * static_cast<double>(totals.survived) / checked, 100.0 * static_cast<double>(totals.survivedWithProof) / checked);
        windows += text;
    }
    addRun(s.passLengths, s.passDraws);
    const auto tracker = GuestMemory::TrackerLockCounts();
    const auto acquisitions = tracker.acquisitions - s.tracker.acquisitions;
    const auto waits = tracker.waits - s.tracker.waits;
    const auto waitNs = tracker.waitNs - s.tracker.waitNs;
    const auto process = processCpuMs();
    const auto committer = threadCpuMs();
    const auto cpus = std::max(1u, std::thread::hardware_concurrency());
    const double processMs = process - s.processCpuMs;
    const double spareMs = std::max(0.0, seconds * 1e3 * cpus - processMs);
    const double passP50 = Percentile(s.passLengths, 0.5), passP90 = Percentile(s.passLengths, 0.9);
    const double serialP50 = Percentile(s.serialRuns, 0.5), serialP90 = Percentile(s.serialRuns, 0.9);
    const double itemP50 = Percentile(s.itemRuns, 0.5), itemP90 = Percentile(s.itemRuns, 0.9);
    std::uint64_t passes = 0, serialRuns = 0, itemRuns = 0;
    for (std::size_t index = 0; index < RunBuckets; ++index) {
        passes += s.passLengths[index];
        serialRuns += s.serialRuns[index];
        itemRuns += s.itemRuns[index];
    }
    const auto verdict = Gate({share, queueSurvival, serialP50});
    AgcDriver::ProfilePrint_nid_no_patch(
        "[parallel] (%.0f s) draw commits %llu: held %.1f us (wall %.1f, locked GPU wait %.1f, outside the phases %.1f; P1: stage A by the committer %llu, pre-drained %llu waiting %.1f ms); phases (us/commit, movable estimate):%s; movable %.1f us = %.0f%% of the hold (%.0f%% with chunked recording); non-draw items:%s; passes %llu, draws per pass p50 %.0f p90 %.0f; draw runs between serial work %llu, p50 %.0f p90 %.0f; between non-draw items %llu, p50 %.0f p90 %.0f%s; tracker mutex %llu acquisitions, %llu waited %.0f us avg; CPU process %.0f ms of %u x %.0f s (spare %.0f ms), committer %.0f ms; gate: movable>=40%% %s, survival>=80%% %s, chunk p50>=8 %s -> %s\n",
        seconds, static_cast<unsigned long long>(draw.count), heldPer, draw.wallUs / commits, lockedWaitPer, outside, static_cast<unsigned long long>(draw.stageA), static_cast<unsigned long long>(draw.preDrained), draw.preDrainUs / 1e3, phases.c_str(), movableSum, share * 100.0, shareChunked * 100.0, others.c_str(), static_cast<unsigned long long>(passes), passP50, passP90, static_cast<unsigned long long>(serialRuns), serialP50, serialP90, static_cast<unsigned long long>(itemRuns), itemP50, itemP90, windows.c_str(),
        static_cast<unsigned long long>(acquisitions), static_cast<unsigned long long>(waits), waits != 0 ? static_cast<double>(waitNs) / static_cast<double>(waits) / 1e3 : 0.0, processMs, cpus, seconds, spareMs, committer - s.committerCpuMs,
        verdict.movable ? "yes" : "no", verdict.survival ? "yes" : "no", verdict.chunks ? "yes" : "no", verdict.Pass() ? "P1-P4" : "P1/P3 only");
    const auto names = s.names;
    const auto movable = s.movable;
    const auto phaseCount = s.phases;
    s = State{};
    s.names = names;
    s.movable = movable;
    s.phases = phaseCount;
    s.primed = true;
    s.last = now;
    s.processCpuMs = process;
    s.committerCpuMs = committer;
    s.tracker = tracker;
}

}
