#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/Router.hpp"
#include <cstdio>

namespace AgcDriver::NewPath {

const char* CommandKindName(CommandKind kind) {
    switch (kind) {
    case CommandKind::Draw: return "draw";
    case CommandKind::Dispatch: return "dispatch";
    case CommandKind::DispatchIndirect: return "dispatch-indirect";
    }
    return "unknown";
}

const char* FallbackReasonName(FallbackReason reason) {
    switch (reason) {
    case FallbackReason::KindSwitch: return "kind-switch";
    case FallbackReason::KindNotEmitted: return "kind-not-emitted";
    case FallbackReason::NoPipeline: return "no-pipeline";
    case FallbackReason::RangeNotCovered: return "range-not-covered";
    case FallbackReason::PushOverflow: return "push-overflow";
    case FallbackReason::ClothGuard: return "cloth-guard";
    case FallbackReason::NonStaticChain: return "non-static-chain";
    case FallbackReason::Suspended: return "suspended";
    case FallbackReason::NoResolvedModule: return "no-resolved-module";
    case FallbackReason::NotRecorded: return "not-recorded";
    case FallbackReason::C1NoTwin: return "c1-no-twin";
    case FallbackReason::C1Switch: return "c1-switch";
    case FallbackReason::C1RangesUnavailable: return "c1-ranges-unavailable";
    case FallbackReason::C1WrittenElsewhere: return "c1-written-elsewhere";
    case FallbackReason::C1NotCovered: return "c1-not-covered";
    case FallbackReason::C1Layout: return "c1-layout";
    case FallbackReason::NotClassA: return "not-class-a";
    case FallbackReason::DrawShape: return "draw-shape";
    case FallbackReason::NotLean: return "not-lean";
    case FallbackReason::TwinSkipped: return "twin-skipped";
    }
    return "unknown";
}

std::uint64_t RouteCounts::Commands() const {
    std::uint64_t total = NewPathCommands();
    for (const auto& reasons : fallback) {
        for (const auto count : reasons) total += count;
    }
    return total;
}

std::uint64_t RouteCounts::NewPathCommands() const {
    std::uint64_t total = 0;
    for (const auto count : newPath) total += count;
    return total;
}

Decision Router::Decide(const CommandFacts& facts) const {
    if (facts.suspended) return {Route::Fallback, FallbackReason::Suspended};
    if (facts.kindSwitchOff) return {Route::Fallback, FallbackReason::KindSwitch};
    if (facts.kind == CommandKind::Draw ? !emitDraws : (!emitDispatches || facts.kind != CommandKind::Dispatch)) return {Route::Fallback, FallbackReason::KindNotEmitted};
    if (!facts.classA) return {Route::Fallback, FallbackReason::NotClassA};
    if (!facts.shapeSupported) return {Route::Fallback, FallbackReason::DrawShape};
    if (!facts.lean) return {Route::Fallback, FallbackReason::NotLean};
    if (!facts.resolvedProgram) return {Route::Fallback, facts.twinSkipped ? FallbackReason::TwinSkipped : !facts.c1 ? FallbackReason::NoResolvedModule : facts.c1Switch ? FallbackReason::C1Switch : FallbackReason::C1NoTwin};
    if (!facts.pipelineReady) return {Route::Fallback, FallbackReason::NoPipeline};
    if (!facts.tableCovers) return {Route::Fallback, facts.coverReason};
    if (!facts.layoutFits) return {Route::Fallback, FallbackReason::C1Layout};
    if (!facts.pushFits && !facts.overflowPlaced) return {Route::Fallback, FallbackReason::PushOverflow};
    if (facts.clothGuard) return {Route::Fallback, FallbackReason::ClothGuard};
    if (!facts.staticChains) return {Route::Fallback, FallbackReason::NonStaticChain};
    return {Route::NewPath, FallbackReason::KindNotEmitted};
}

void Router::Note(CommandKind kind, const Decision& decision) {
    const auto index = static_cast<std::size_t>(kind);
    if (decision.route == Route::NewPath) newPath[index].fetch_add(1, std::memory_order_relaxed);
    else fallback[index][static_cast<std::size_t>(decision.reason)].fetch_add(1, std::memory_order_relaxed);
}

RouteCounts Router::Take() {
    RouteCounts counts;
    for (std::size_t kind = 0; kind < CommandKindCount; ++kind) {
        counts.newPath[kind] = newPath[kind].exchange(0, std::memory_order_relaxed);
        for (std::size_t reason = 0; reason < FallbackReasonCount; ++reason) counts.fallback[kind][reason] = fallback[kind][reason].exchange(0, std::memory_order_relaxed);
    }
    return counts;
}

std::string Router::Format(const RouteCounts& counts, const char* window) {
    const auto commands = counts.Commands();
    const auto onNewPath = counts.NewPathCommands();
    char head[160];
    std::snprintf(head, sizeof(head), "[newpath] window=%s commands %llu, new path %llu (%.1f %%)", window, static_cast<unsigned long long>(commands), static_cast<unsigned long long>(onNewPath), commands != 0 ? 100.0 * static_cast<double>(onNewPath) / static_cast<double>(commands) : 0.0);
    std::string line = head;
    for (std::size_t kind = 0; kind < CommandKindCount; ++kind) {
        std::string reasons;
        for (std::size_t reason = 0; reason < FallbackReasonCount; ++reason) {
            if (counts.fallback[kind][reason] == 0) continue;
            reasons += std::string(reasons.empty() ? "" : ", ") + FallbackReasonName(static_cast<FallbackReason>(reason)) + " " + std::to_string(counts.fallback[kind][reason]);
        }
        line += std::string("; ") + CommandKindName(static_cast<CommandKind>(kind)) + " new " + std::to_string(counts.newPath[kind]) + " fallback " + (reasons.empty() ? std::string("none") : reasons);
    }
    return line;
}

}
