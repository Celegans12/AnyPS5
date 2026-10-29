#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/NewPath.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyConfig.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencySession.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DevicePageTable.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ResolvedPush.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ResolvedTwins.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PipelineAheadConfig.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PipelineAhead.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DrawTwins.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstdio>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>

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
    Router router{EmitDispatches(), EmitDraws()};
    std::array<DispatchCost, 4> costs;
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

std::shared_ptr<EmitCandidate>& ThreadDrawEmit() {
    thread_local std::shared_ptr<EmitCandidate> current;
    return current;
}

std::atomic<bool> drawTwinPipelinesUsed{false};

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
        static constexpr const char* names[] = {"new path (table deferred)", "new path (table built)", "fallback address-based", "fallback other"};
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
    const auto ahead = Pipelines().Totals();
    AgcDriver::ProfilePrint_nid_no_patch("[newpath] window=%s pipelines ahead (gate %s): requested %llu, created %llu, failed %llu, inline %llu, blocked %llu, pending fallbacks %llu, prewarmed %llu\n", window, PipelineGateEnabled() ? "on" : "off", static_cast<unsigned long long>(ahead.requested), static_cast<unsigned long long>(ahead.created), static_cast<unsigned long long>(ahead.failed), static_cast<unsigned long long>(ahead.createdInline), static_cast<unsigned long long>(ahead.blockedWaits), static_cast<unsigned long long>(ahead.pendingFallbacks), static_cast<unsigned long long>(ahead.prewarmed));
    const auto twinPipes = TwinPipelineStore().Totals();
    AgcDriver::ProfilePrint_nid_no_patch("[newpath] window=%s twin pipelines (cumulative): created on workers %llu (%.0f ms), on the dispatch thread %llu (%.0f ms, max %.0f ms, %llu over 1 s); not ready at a dispatch %llu, layout mismatches %llu; module files written %llu (failed %llu), prewarm queued %llu, skipped %llu, unreadable %llu\n", window, static_cast<unsigned long long>(twinPipes.createdOnWorker), twinPipes.workerMs, static_cast<unsigned long long>(twinPipes.createdOnCaller), twinPipes.callerMs, twinPipes.maxCallerMs, static_cast<unsigned long long>(twinPipes.callerOverSecond), static_cast<unsigned long long>(twinPipes.notReady), static_cast<unsigned long long>(twinPipes.layoutMismatches), static_cast<unsigned long long>(twinPipes.modulesWritten), static_cast<unsigned long long>(twinPipes.moduleWriteFailures), static_cast<unsigned long long>(twinPipes.prewarmQueued), static_cast<unsigned long long>(twinPipes.prewarmSkipped), static_cast<unsigned long long>(twinPipes.moduleReadFailures));
    if (EmitDraws()) {
        const auto drawPipes = DrawTwinPipelines().Totals();
        AgcDriver::ProfilePrint_nid_no_patch("%s; pipelines ahead requested %llu, created %llu, failed %llu, inline %llu\n", FormatDrawEmit(window, true).c_str(), static_cast<unsigned long long>(drawPipes.requested), static_cast<unsigned long long>(drawPipes.created), static_cast<unsigned long long>(drawPipes.failed), static_cast<unsigned long long>(drawPipes.createdInline));
    }
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

bool EmitDraws() {
    static const bool emit = Enabled() && EnvSet("APS5_NEW_PATH_EMIT_DRAWS");
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

void PrepareCommand(const VulkanDevice* device) {
    prepare(device);
    report(Get());
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
    candidate->facts.pushFits = LegacyPushBytes <= Abi::PushBlockBytes - ResolvedDriverWords * 4u;
    candidate->facts.staticChains = true;
    return candidate;
}

EmitCandidate::~EmitCandidate() {
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

bool MayEmitWithoutTable(const EmitCandidate& candidate, bool pipelineReady) {
    if (candidate.twin == nullptr || !WalkReplacesTables(*candidate.twin)) return false;
    auto facts = candidate.facts;
    facts.pipelineReady = pipelineReady;
    facts.clothGuard = false;
    facts.tableCovers = true;
    return Get().router.Decide(facts).route == Route::NewPath;
}

std::shared_ptr<EmitCandidate> ClaimDraw() {
    auto candidate = std::make_shared<EmitCandidate>();
    candidate->counted = true;
    candidate->facts = FactsAtDecode(CommandKind::Draw);
    candidate->facts.classA = false;
    candidate->facts.staticChains = true;
    return candidate;
}

DrawEmitScope::DrawEmitScope(std::shared_ptr<EmitCandidate> candidate) : candidate(std::move(candidate)), previous(ThreadDrawEmit()) {
    ThreadDrawEmit() = this->candidate;
}

DrawEmitScope::~DrawEmitScope() {
    ThreadDrawEmit() = std::move(previous);
}

EmitCandidate* CurrentDrawEmit() {
    auto& current = ThreadDrawEmit();
    if (current == nullptr || current->noted.load(std::memory_order_relaxed)) return nullptr;
    return current.get();
}

PipelineAhead& DrawTwinPipelines() {
    static auto* pipelines = new PipelineAhead(PipelineAhead::Settings{PipelineGateEnabled(), 2}, nullptr);
    return *pipelines;
}

void NoteDrawTwinPipelines() {
    drawTwinPipelinesUsed.store(true, std::memory_order_relaxed);
}

bool DrawTwinPipelinesUsed() {
    return drawTwinPipelinesUsed.load(std::memory_order_relaxed);
}

void StopDrawTwinPipelines() {
    if (DrawTwinPipelinesUsed()) DrawTwinPipelines().Reset();
}

bool WouldEmit(const CommandFacts& facts) {
    return Get().router.Decide(facts).route == Route::NewPath;
}

void NoteDispatchCost(const EmitCandidate& candidate, std::uint64_t deviceNs, std::uint64_t buildNs) {
    if (!Profile()) return;
    CostLap(CostStep::PostRecord);
    auto& clock = ThreadSteps();
    clock.active = false;
    auto& cost = Get().costs[candidate.routed != Route::NewPath ? (candidate.addressBased ? 2 : 3) : candidate.tableDeferred ? 0 : 1];
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

void Close(const char* window) {
    if (!Enabled() || !Profile()) return;
    auto& state = Get();
    std::lock_guard lock(state.reporting);
    print(state, window);
}

}
