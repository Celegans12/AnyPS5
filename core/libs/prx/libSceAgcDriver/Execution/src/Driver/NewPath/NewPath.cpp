#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/NewPath.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/C1Buffers.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyConfig.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencySession.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DevicePageTable.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ResolvedPush.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ResolvedTwins.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ShareCensus.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PipelineAheadConfig.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <chrono>
#include <cstdlib>
#include <string_view>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace AgcDriver::NewPath {

namespace {

bool EnvSet(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

constexpr std::size_t CostSteps = static_cast<std::size_t>(CostStep::Count);

struct DispatchCost {
    std::atomic<std::uint64_t> count{0}, deviceNs{0}, buildNs{0}, slow{0};
    std::array<std::atomic<std::uint64_t>, CostSteps> stepNs{};
};

struct StepClock {
    bool active = false;
    std::chrono::steady_clock::time_point last;
    std::array<std::uint64_t, CostSteps> ns{};
};

StepClock& ThreadSteps() {
    thread_local StepClock clock;
    return clock;
}

struct State {
    Router router{EmitDispatches()};
    std::array<DispatchCost, 6> costs;
    std::atomic<std::uint64_t> completions{0}, completionNs{0};
    std::mutex reporting;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

State& Get() {
    static State state;
    return state;
}

bool Profile() {
    static const bool profile = EnvSet("APS5_PROFILE_NEW_PATH");
    return profile;
}

struct Pending {
    bool active = false;
    CommandFacts facts;
};

Pending& ThreadPending() {
    thread_local Pending pending;
    return pending;
}

std::shared_ptr<EmitCandidate>& ThreadEmit() {
    thread_local std::shared_ptr<EmitCandidate> current;
    return current;
}

CommandFacts FactsAtDecode(CommandKind kind) {
    CommandFacts facts;
    facts.kind = kind;
    facts.kindSwitchOff = Graphics::RuntimeSwitches::Set(kind == CommandKind::Draw ? Graphics::RuntimeSwitch::NewPathNoGpuDraws : Graphics::RuntimeSwitch::NewPathNoGpuDispatches);
    facts.resolvedProgram = false;
    return facts;
}

void print(State& state, const char* window) {
    const auto line = Router::Format(state.router.Take(), window);
    const auto table = DeviceTable().Totals();
    const auto twins = Twins().Outcomes();
    AgcDriver::ProfilePrint_nid_no_patch("%s; page table builds %llu, patches %llu, ranges %llu, covered %.1f MiB, image mirrors %llu (%.1f MiB); emit %s: resolved twins built %llu, not address-based %llu, failed %llu, mismatched %llu, reused %llu; table compares covered %llu, missing %llu, other address %llu, narrower %llu\n", line.c_str(), static_cast<unsigned long long>(table.builds), static_cast<unsigned long long>(table.patches), static_cast<unsigned long long>(table.ranges), static_cast<double>(table.coveredBytes) / (1024.0 * 1024.0), static_cast<unsigned long long>(table.mirrorRanges), static_cast<double>(table.mirrorBytes) / (1024.0 * 1024.0), EmitDispatches() ? "on" : "off", static_cast<unsigned long long>(twins[0]), static_cast<unsigned long long>(twins[1]), static_cast<unsigned long long>(twins[2]), static_cast<unsigned long long>(twins[3]), static_cast<unsigned long long>(twins[4]), static_cast<unsigned long long>(table.coverHits[0]), static_cast<unsigned long long>(table.coverHits[1]), static_cast<unsigned long long>(table.coverHits[2]), static_cast<unsigned long long>(table.coverHits[3]));
    {
        std::string costs;
        static constexpr const char* names[] = {"new path (table deferred)", "new path (table built)", "fallback address-based", "fallback other", "new path C1", "fallback C1"};
        static_assert(std::size(names) == std::tuple_size_v<decltype(state.costs)>);
        for (std::size_t index = 0; index < state.costs.size(); ++index) {
            auto& cost = state.costs[index];
            const auto count = cost.count.load(std::memory_order_relaxed);
            char text[192];
            std::snprintf(text, sizeof(text), "%s%s %llu x %.1f us device call (resources %.1f us), %llu over 50 ms left out", costs.empty() ? "" : "; ", names[index], static_cast<unsigned long long>(count), count != 0 ? static_cast<double>(cost.deviceNs.load(std::memory_order_relaxed)) / 1000.0 / static_cast<double>(count) : 0.0, count != 0 ? static_cast<double>(cost.buildNs.load(std::memory_order_relaxed)) / 1000.0 / static_cast<double>(count) : 0.0, static_cast<unsigned long long>(cost.slow.load(std::memory_order_relaxed)));
            costs += text;
        }
        AgcDriver::ProfilePrint_nid_no_patch("[newpath-cost] window=%s claimed direct dispatches (cumulative): %s\n", window, costs.c_str());
        static constexpr const char* steps[] = {"pre-record", "record head", "ranges", "F4", "route", "twin pipeline", "ensure table", "record commands", "table upload", "record+push", "dispatch+marks", "post-record"};
        static_assert(std::size(steps) == CostSteps);
        std::string split;
        for (std::size_t index = 0; index < state.costs.size(); ++index) {
            auto& cost = state.costs[index];
            const auto count = cost.count.load(std::memory_order_relaxed);
            if (count == 0) continue;
            split += (split.empty() ? "" : "; ") + std::string(names[index]) + ":";
            for (std::size_t step = 0; step < CostSteps; ++step) {
                char text[64];
                std::snprintf(text, sizeof(text), " %s %.1f", steps[step], static_cast<double>(cost.stepNs[step].load(std::memory_order_relaxed)) / 1000.0 / static_cast<double>(count));
                split += text;
            }
        }
        const auto completions = state.completions.load(std::memory_order_relaxed);
        AgcDriver::ProfilePrint_nid_no_patch("[newpath-cost] window=%s us per dispatch after the build (cumulative): %s; command record completions %llu x %.1f us, records made %llu; F4 compares %llu, served by the space serial %llu\n", window, split.c_str(), static_cast<unsigned long long>(completions), completions != 0 ? static_cast<double>(state.completionNs.load(std::memory_order_relaxed)) / 1000.0 / static_cast<double>(completions) : 0.0, static_cast<unsigned long long>(table.records), static_cast<unsigned long long>(table.compares), static_cast<unsigned long long>(table.serialHits));
    }
    {
        const auto c1 = Twins().C1Outcomes();
        AgcDriver::ProfilePrint_nid_no_patch("[newpath] window=%s C1 twins (cumulative, newPathRuntimeBuffers requests): built %llu (binding no guest buffers %llu), failed %llu, mismatched %llu\n", window, static_cast<unsigned long long>(c1[0]), static_cast<unsigned long long>(Twins().C1RuntimeBuilt()), static_cast<unsigned long long>(c1[2]), static_cast<unsigned long long>(c1[3]));
    }
    if (Census::Enabled()) AgcDriver::ProfilePrint_nid_no_patch("%s\n", Census::Format(window).c_str());
    const auto ahead = Pipelines().Totals();
    AgcDriver::ProfilePrint_nid_no_patch("[newpath] window=%s pipelines ahead (gate %s): requested %llu, created %llu, failed %llu, inline %llu, blocked %llu, pending fallbacks %llu, prewarmed %llu\n", window, PipelineGateEnabled() ? "on" : "off", static_cast<unsigned long long>(ahead.requested), static_cast<unsigned long long>(ahead.created), static_cast<unsigned long long>(ahead.failed), static_cast<unsigned long long>(ahead.createdInline), static_cast<unsigned long long>(ahead.blockedWaits), static_cast<unsigned long long>(ahead.pendingFallbacks), static_cast<unsigned long long>(ahead.prewarmed));
    const auto twinPipes = TwinPipelineStore().Totals();
    AgcDriver::ProfilePrint_nid_no_patch("[newpath] window=%s twin pipelines (cumulative): created on workers %llu (%.0f ms), on the dispatch thread %llu (%.0f ms, max %.0f ms, %llu over 1 s); not ready at a dispatch %llu, layout mismatches %llu; module files written %llu (failed %llu), prewarm queued %llu, skipped %llu, unreadable %llu\n", window, static_cast<unsigned long long>(twinPipes.createdOnWorker), twinPipes.workerMs, static_cast<unsigned long long>(twinPipes.createdOnCaller), twinPipes.callerMs, twinPipes.maxCallerMs, static_cast<unsigned long long>(twinPipes.callerOverSecond), static_cast<unsigned long long>(twinPipes.notReady), static_cast<unsigned long long>(twinPipes.layoutMismatches), static_cast<unsigned long long>(twinPipes.modulesWritten), static_cast<unsigned long long>(twinPipes.moduleWriteFailures), static_cast<unsigned long long>(twinPipes.prewarmQueued), static_cast<unsigned long long>(twinPipes.prewarmSkipped), static_cast<unsigned long long>(twinPipes.moduleReadFailures));
    if (const auto residency = Graphics::ResidencyReportLine(); !residency.empty()) AgcDriver::ProfilePrint_nid_no_patch("%s\n", residency.c_str());
}

void report(State& state) {
    if (!Profile()) return;
    std::unique_lock lock(state.reporting, std::try_to_lock);
    if (lock.owns_lock() && ProfileWindows::ClockDue(state.lastReport)) print(state, "10 s");
}

void prepare(const VulkanDevice* device) {
    if (device == nullptr) return;
    if (DeviceTable().Stale(device->Device())) static_cast<void>(DeviceTable().Refresh(device->graphicsContext()));
    if (Graphics::CurrentResidencyConfig().residency && Graphics::Recorder::Active() != nullptr && !Graphics::ResidencyEnsured(device->Device())) Graphics::EnsureResidency(device->graphicsContext());
}

}

bool Enabled() {
    static const bool enabled = Graphics::CurrentResidencyConfig().newPath;
    return enabled;
}

bool EmitDispatches() {
    static const bool emit = Enabled() && EnvSet("APS5_NEW_PATH_EMIT_DISPATCHES");
    return emit;
}

Decision RouteCommand(CommandKind kind, const VulkanDevice* device) {
    auto& state = Get();
    prepare(device);
    const auto decision = state.router.Decide(FactsAtDecode(kind));
    state.router.Note(kind, decision);
    report(state);
    return decision;
}

void BeginCommand(CommandKind kind, const VulkanDevice* device) {
    prepare(device);
    auto& pending = ThreadPending();
    pending.active = true;
    pending.facts = FactsAtDecode(kind);
}

void EndCommand() {
    auto& state = Get();
    auto& pending = ThreadPending();
    if (pending.active) {
        pending.active = false;
        if (pending.facts.kind == CommandKind::Dispatch) Census::NoteUnclaimed();
        state.router.Note(pending.facts.kind, state.router.Decide(pending.facts));
    }
    report(state);
}

std::shared_ptr<EmitCandidate> ClaimDispatch(const ShaderRecompiler::RecompileResult& compiled) {
    auto candidate = std::make_shared<EmitCandidate>();
    auto& pending = ThreadPending();
    candidate->counted = pending.active && pending.facts.kind == CommandKind::Dispatch;
    candidate->facts = candidate->counted ? pending.facts : FactsAtDecode(CommandKind::Dispatch);
    if (candidate->counted) pending.active = false;
    else candidate->noted.store(true, std::memory_order_relaxed);
    candidate->twin = Twins().Of(compiled);
    candidate->facts.resolvedProgram = candidate->twin != nullptr;
    candidate->facts.c1 = !UsesAddressTables(compiled) && C1Shape(compiled);
    candidate->facts.c1Switch = candidate->facts.c1 && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NewPathNoC1);
    if (candidate->facts.c1Switch) {
        candidate->twin = nullptr;
        candidate->facts.resolvedProgram = false;
    }
    candidate->c1Twin = candidate->facts.c1 && candidate->twin != nullptr;
    candidate->runtimeBuffers = candidate->c1Twin && RuntimeBufferTwin(compiled, *candidate->twin);
    candidate->facts.pushFits = LegacyPushBytes <= Abi::PushBlockBytes - ResolvedDriverWords * 4u;
    candidate->facts.staticChains = true;
    Census::Fill(*candidate, compiled);
    return candidate;
}

EmitCandidate::~EmitCandidate() {
    if (census.active && !census.committed) Census::NoteCommit(*this, false, 0, 0);
    if (noted.exchange(true, std::memory_order_relaxed)) return;
    auto& state = Get();
    auto facts = this->facts;
    auto decision = state.router.Decide(facts);
    if (decision.route == Route::NewPath) decision = {Route::Fallback, FallbackReason::NotRecorded};
    state.router.Note(facts.kind, decision);
}

EmitScope::EmitScope(std::shared_ptr<EmitCandidate> candidate) : candidate(std::move(candidate)), previous(ThreadEmit()) {
    ThreadEmit() = this->candidate;
}

EmitScope::~EmitScope() {
    ThreadEmit() = std::move(previous);
}

EmitCandidate* CurrentEmit() {
    auto& current = ThreadEmit();
    if (current == nullptr || current->noted.load(std::memory_order_relaxed)) return nullptr;
    return current.get();
}

Decision NoteEmit(EmitCandidate& candidate, const CommandFacts& facts) {
    auto& state = Get();
    const auto decision = state.router.Decide(facts);
    if (!candidate.noted.exchange(true, std::memory_order_relaxed)) {
        state.router.Note(facts.kind, decision);
        candidate.routed = decision.route;
    }
    return decision;
}

bool InPlaceWrites(const EmitCandidate& candidate) {
    static const int mode = [] {
        if (!EmitDispatches() || EnvSet("APS5_NEW_PATH_NO_IN_PLACE_WRITES")) return 0;
        const char* scope = std::getenv("APS5_NEW_PATH_IN_PLACE_WRITES");
        return scope != nullptr && std::string_view(scope) == "all" ? 2 : 1;
    }();
    return mode == 2 || (mode == 1 && candidate.twin != nullptr);
}

bool MayEmitWithoutTable(const EmitCandidate& candidate, bool pipelineReady) {
    if (candidate.twin == nullptr || candidate.c1Twin || !WalkReplacesTables(*candidate.twin)) return false;
    auto facts = candidate.facts;
    facts.pipelineReady = pipelineReady;
    facts.clothGuard = false;
    facts.tableCovers = true;
    return Get().router.Decide(facts).route == Route::NewPath;
}

bool WouldEmit(const CommandFacts& facts) {
    return Get().router.Decide(facts).route == Route::NewPath;
}

void NoteDispatchCost(const EmitCandidate& candidate, std::uint64_t deviceNs, std::uint64_t buildNs) {
    if (!Profile()) return;
    CostLap(CostStep::PostRecord);
    auto& clock = ThreadSteps();
    clock.active = false;
    if (candidate.census.active && !candidate.census.committed) {
        candidate.census.committed = true;
        Census::NoteCommit(candidate, true, deviceNs, buildNs);
    }
    auto& cost = Get().costs[candidate.facts.c1 ? (candidate.routed == Route::NewPath ? 4 : 5) : candidate.routed != Route::NewPath ? (candidate.addressBased ? 2 : 3) : candidate.tableDeferred ? 0 : 1];
    if (deviceNs > 50'000'000ull) {
        cost.slow.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    cost.count.fetch_add(1, std::memory_order_relaxed);
    cost.deviceNs.fetch_add(deviceNs, std::memory_order_relaxed);
    cost.buildNs.fetch_add(buildNs, std::memory_order_relaxed);
    for (std::size_t step = 0; step < CostSteps; ++step) {
        if (clock.ns[step] != 0) cost.stepNs[step].fetch_add(clock.ns[step], std::memory_order_relaxed);
    }
}

void CostBegin() {
    if (!Profile()) return;
    auto& clock = ThreadSteps();
    clock.active = true;
    clock.ns = {};
    clock.last = std::chrono::steady_clock::now();
}

void CostLap(CostStep step) {
    auto& clock = ThreadSteps();
    if (!clock.active) return;
    const auto now = std::chrono::steady_clock::now();
    clock.ns[static_cast<std::size_t>(step)] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - clock.last).count());
    clock.last = now;
}

void NoteCompletionCost(std::uint64_t ns) {
    if (!Profile()) return;
    auto& state = Get();
    state.completions.fetch_add(1, std::memory_order_relaxed);
    state.completionNs.fetch_add(ns, std::memory_order_relaxed);
}

void NoteBufferCover(const EmitCandidate& candidate, const Graphics::ShaderResources& resources) {
    if (!candidate.census.active || candidate.twin != nullptr || candidate.addressBased || (candidate.census.shape & (Census::BuffersFromUser | Census::BuffersFromMemory)) == 0) return;
    auto cover = Census::BufferCover::Unavailable;
    try {
        if (const auto ranges = resources.GuestBufferRanges(); ranges != nullptr) {
            DevicePageTable::Range failed{};
            switch (DeviceTable().Probe(*ranges, &failed)) {
            case DevicePageTable::Cover::Covered: cover = Census::BufferCover::Covered; break;
            case DevicePageTable::Cover::Missing: cover = Census::BufferCover::Missing; break;
            case DevicePageTable::Cover::OtherAddress: {
                std::vector<std::array<std::uint64_t, 3>> accesses;
                resources.RunAccesses(accesses);
                const bool written = std::ranges::any_of(accesses, [&](const std::array<std::uint64_t, 3>& access) { return (access[2] & Graphics::Recorder::RunAccessImage) == 0 && (access[2] & Graphics::Recorder::RunAccessWritten) != 0 && access[0] < failed.end && failed.begin < access[1]; });
                cover = written ? Census::BufferCover::OtherAddressWritten : Census::BufferCover::OtherAddress;
                static std::atomic<int> shown{0};
                if (written && shown.fetch_add(1, std::memory_order_relaxed) < 40) {
                    std::size_t refused = 0;
                    for (const auto& range : *ranges) {
                        if (DevicePageTable::Compare(DeviceTable().Ranges(), std::span<const DevicePageTable::Range>(&range, 1)) != DevicePageTable::Cover::Covered) ++refused;
                    }
                    std::fprintf(stderr, "[newpath-share] F4 written range 0x%llx-0x%llx (%llu bytes, %s) at device 0x%llx refused; %zu of %zu build ranges refused\n", static_cast<unsigned long long>(failed.begin), static_cast<unsigned long long>(failed.end), static_cast<unsigned long long>(failed.end - failed.begin), (failed.begin >> 16) == ((failed.end - 1) >> 16) ? "one page" : "spanning", static_cast<unsigned long long>(failed.deviceAddress), refused, ranges->size());
                }
                break;
            }
            case DevicePageTable::Cover::Narrower: cover = Census::BufferCover::Narrower; break;
            }
        }
    } catch (const std::exception&) {
    }
    candidate.census.cover = static_cast<std::uint8_t>(cover);
}

namespace {

struct KeptRanges {
    struct Entry {
        std::weak_ptr<const Graphics::ShaderResources> resources;
        std::shared_ptr<const std::vector<DevicePageTable::Range>> ranges;
    };
    std::mutex mutex;
    std::unordered_map<const Graphics::ShaderResources*, Entry> entries;
    std::size_t prunedAt = 256;
};

KeptRanges& BuildRanges() {
    static auto* kept = new KeptRanges;
    return *kept;
}

}

void NoteBufferRanges(EmitCandidate& candidate, const std::shared_ptr<Graphics::ShaderResources>& resources, bool fromCache, bool moved) {
    candidate.bufferRanges = nullptr;
    if (!candidate.c1Twin || resources == nullptr || moved) return;
    auto& kept = BuildRanges();
    if (fromCache) {
        std::lock_guard lock(kept.mutex);
        if (const auto found = kept.entries.find(resources.get()); found != kept.entries.end() && found->second.resources.lock() == resources) {
            candidate.bufferRanges = found->second.ranges;
            return;
        }
    }
    try {
        candidate.bufferRanges = resources->GuestBufferRanges();
    } catch (const std::exception&) {
        return;
    }
    if (candidate.bufferRanges == nullptr || !resources->Reusable()) return;
    std::lock_guard lock(kept.mutex);
    if (kept.entries.size() >= kept.prunedAt) {
        std::erase_if(kept.entries, [](const auto& item) { return item.second.resources.expired(); });
        kept.prunedAt = std::max<std::size_t>(256, kept.entries.size() * 2);
    }
    kept.entries[resources.get()] = {resources, candidate.bufferRanges};
}

std::size_t KeptBufferRanges() {
    auto& kept = BuildRanges();
    std::lock_guard lock(kept.mutex);
    return kept.entries.size();
}

void CoverBuffers(const EmitCandidate& candidate, const Graphics::ShaderResources& resources, CommandFacts& facts, DevicePageTable::Snapshot* used, const ShaderRecompiler::RecompileResult* threadLimitOf) {
    facts.layoutFits = threadLimitOf == nullptr || !candidate.runtimeBuffers || (candidate.twin->imageMetadataDword == threadLimitOf->imageMetadataDword && std::ranges::any_of(candidate.twin->bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::ShaderData; }));
    if (candidate.bufferRanges == nullptr) {
        facts.tableCovers = false;
        facts.coverReason = FallbackReason::C1RangesUnavailable;
        return;
    }
    if (!SitesInside(candidate.twin->readRanges, *candidate.bufferRanges, false) || !SitesInside(candidate.twin->writeRanges, *candidate.bufferRanges, true)) {
        // A write site inside the build's ranges but not where the build writes in place: its written elements are
        // bound to a staged copy (the S2-share probe's dominant case).
        facts.tableCovers = false;
        facts.coverReason = SitesInside(candidate.twin->readRanges, *candidate.bufferRanges, false) && SitesInside(candidate.twin->writeRanges, *candidate.bufferRanges, false) ? FallbackReason::C1WrittenElsewhere : FallbackReason::C1NotCovered;
        static std::atomic<int> reported{0};
        if (Profile() && reported.fetch_add(1, std::memory_order_relaxed) < 12) {
            std::string text;
            const auto list = [&](const char* name, const auto& ranges, auto describe) {
                text += std::string(text.empty() ? "" : "; ") + name + ":";
                std::size_t shown = 0;
                for (const auto& range : ranges) {
                    if (++shown > 6) { text += " ..."; break; }
                    text += " " + describe(range);
                }
            };
            const auto hex = [](std::uint64_t value) { char buffer[24]; std::snprintf(buffer, sizeof(buffer), "0x%llx", static_cast<unsigned long long>(value)); return std::string(buffer); };
            const auto site = [&](const std::pair<std::uint64_t, std::uint64_t>& range) { return hex(range.first) + "-" + hex(range.second); };
            list("reads", candidate.twin->readRanges, site);
            list("writes", candidate.twin->writeRanges, site);
            list("build", *candidate.bufferRanges, [&](const DevicePageTable::Range& range) { return hex(range.begin) + "-" + hex(range.end) + ((range.permissions & ShaderRecompiler::BdaAbi::Write) != 0u ? "w" : "r"); });
            std::fprintf(stderr, "[newpath] C1 %s (unbounded reads %d writes %d): %s\n", FallbackReasonName(facts.coverReason), candidate.twin->unboundedReads ? 1 : 0, candidate.twin->unboundedWrites ? 1 : 0, text.c_str());
        }
        return;
    }
    auto& table = DeviceTable();
    const auto cover = table.Covers(candidate.bufferRanges, used);
    facts.tableCovers = cover == DevicePageTable::Cover::Covered;
    if (facts.tableCovers) return;
    facts.coverReason = FallbackReason::C1NotCovered;
    if (cover != DevicePageTable::Cover::OtherAddress) return;
    // Other address: on a range the build writes (its written elements bound to a staged copy), the dominant case
    // in the S2-share probe, named apart.
    DevicePageTable::Range failed{};
    if (table.Probe(*candidate.bufferRanges, &failed) != DevicePageTable::Cover::OtherAddress) return;
    thread_local std::vector<std::array<std::uint64_t, 3>> accesses;
    accesses.clear();
    resources.RunAccesses(accesses);
    if (std::ranges::any_of(accesses, [&](const std::array<std::uint64_t, 3>& access) { return (access[2] & Graphics::Recorder::RunAccessImage) == 0 && (access[2] & Graphics::Recorder::RunAccessWritten) != 0 && access[0] < failed.end && failed.begin < access[1]; })) facts.coverReason = FallbackReason::C1WrittenElsewhere;
}

void Close(const char* window) {
    if (!Enabled() || !Profile()) return;
    auto& state = Get();
    std::lock_guard lock(state.reporting);
    print(state, window);
}

}
