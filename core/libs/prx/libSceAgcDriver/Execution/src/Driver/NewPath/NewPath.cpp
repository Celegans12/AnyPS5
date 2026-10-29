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
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "Recompiler.hpp"
#include <cstdio>
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

struct State {
    Router router{EmitDispatches()};
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
    AgcDriver::ProfilePrint_nid_no_patch("%s; page table builds %llu, patches %llu, ranges %llu, covered %.1f MiB; emit %s: resolved twins built %llu, not address-based %llu, failed %llu, mismatched %llu; table compares covered %llu, missing %llu, other address %llu, narrower %llu\n", line.c_str(), static_cast<unsigned long long>(table.builds), static_cast<unsigned long long>(table.patches), static_cast<unsigned long long>(table.ranges), static_cast<double>(table.coveredBytes) / (1024.0 * 1024.0), EmitDispatches() ? "on" : "off", static_cast<unsigned long long>(twins[0]), static_cast<unsigned long long>(twins[1]), static_cast<unsigned long long>(twins[2]), static_cast<unsigned long long>(twins[3]), static_cast<unsigned long long>(table.coverHits[0]), static_cast<unsigned long long>(table.coverHits[1]), static_cast<unsigned long long>(table.coverHits[2]), static_cast<unsigned long long>(table.coverHits[3]));
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
    candidate->facts.staticChains = !compiled.bindlessHeap;
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
    if (!candidate.noted.exchange(true, std::memory_order_relaxed)) state.router.Note(facts.kind, decision);
    return decision;
}

void Close(const char* window) {
    if (!Enabled() || !Profile()) return;
    auto& state = Get();
    std::lock_guard lock(state.reporting);
    print(state, window);
}

}
