#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/NewPathAbi.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/Router.hpp"
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

using AgcDriver::NewPath::CommandFacts;
using AgcDriver::NewPath::CommandKind;
using AgcDriver::NewPath::CommandKindCount;
using AgcDriver::NewPath::Decision;
using AgcDriver::NewPath::FallbackReason;
using AgcDriver::NewPath::FallbackReasonCount;
using AgcDriver::NewPath::Route;
using AgcDriver::NewPath::Router;

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void skeletonForwardsEveryKind() {
    Router router;
    for (std::size_t kind = 0; kind < CommandKindCount; ++kind) {
        CommandFacts facts;
        facts.kind = static_cast<CommandKind>(kind);
        const auto decision = router.Decide(facts);
        Require(decision.route == Route::Fallback, std::string("the skeleton put a ") + AgcDriver::NewPath::CommandKindName(facts.kind) + " on the new path");
        Require(decision.reason == FallbackReason::KindNotEmitted, "the skeleton's fallback reason is not kind-not-emitted");
    }
}

void kindSwitchWinsFirst() {
    Router router;
    CommandFacts facts;
    facts.kind = CommandKind::Dispatch;
    facts.kindSwitchOff = true;
    const auto decision = router.Decide(facts);
    Require(decision.route == Route::Fallback && decision.reason == FallbackReason::KindSwitch, "a set kind switch did not decide the fallback first");
}

void countsByKindAndReason() {
    Router router;
    router.Note(CommandKind::Draw, {Route::Fallback, FallbackReason::KindNotEmitted});
    router.Note(CommandKind::Draw, {Route::Fallback, FallbackReason::KindNotEmitted});
    router.Note(CommandKind::DispatchIndirect, {Route::Fallback, FallbackReason::KindSwitch});
    router.Note(CommandKind::Dispatch, {Route::NewPath, FallbackReason::KindNotEmitted});
    const auto counts = router.Take();
    Require(counts.Commands() == 4, "the counts do not total the noted commands");
    Require(counts.NewPathCommands() == 1, "the new-path count is wrong");
    Require(counts.fallback[static_cast<std::size_t>(CommandKind::Draw)][static_cast<std::size_t>(FallbackReason::KindNotEmitted)] == 2, "draw fallbacks were not counted by reason");
    Require(counts.fallback[static_cast<std::size_t>(CommandKind::DispatchIndirect)][static_cast<std::size_t>(FallbackReason::KindSwitch)] == 1, "indirect dispatch fallbacks were not counted by reason");
    Require(router.Take().Commands() == 0, "taking the counts did not start a new window");
}

void formatNamesShareAndReasons() {
    Router router;
    router.Note(CommandKind::Draw, {Route::Fallback, FallbackReason::KindNotEmitted});
    router.Note(CommandKind::Dispatch, {Route::NewPath, FallbackReason::KindNotEmitted});
    const auto line = Router::Format(router.Take(), "frame-2");
    Require(line.rfind("[newpath] window=frame-2 commands 2, new path 1 (50.0 %)", 0) == 0, "unexpected head: " + line);
    Require(line.find("draw new 0 fallback kind-not-emitted 1") != std::string::npos, "draw part missing: " + line);
    Require(line.find("dispatch new 1 fallback none") != std::string::npos, "dispatch part missing: " + line);
    Require(line.find("dispatch-indirect new 0 fallback none") != std::string::npos, "indirect part missing: " + line);
    for (std::size_t reason = 0; reason < FallbackReasonCount; ++reason) {
        const std::string name = AgcDriver::NewPath::FallbackReasonName(static_cast<FallbackReason>(reason));
        Require(name != "unknown" && name.find(' ') == std::string::npos, "a fallback reason has no one-word name");
    }
}

void pushBlockLimit() {
    AgcDriver::NewPath::Abi::PushLayout layout;
    layout.driverWords = 8;
    layout.userWordCount = {16, 0, 0, 16};
    Require(AgcDriver::NewPath::Abi::FitsPushBlock(layout), "32 user words and 8 driver words did not fit 256 bytes");
    layout.userWordCount = {16, 16, 16, 16};
    Require(!AgcDriver::NewPath::Abi::FitsPushBlock(layout), "64 user words and 8 driver words fit 256 bytes");
}

}

int main() {
    try {
        skeletonForwardsEveryKind();
        kindSwitchWinsFirst();
        countsByKindAndReason();
        formatNamesShareAndReasons();
        pushBlockLimit();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "agc_newpath_router: %s\n", error.what());
        return 1;
    }
    std::printf("agc_newpath_router: ok\n");
    return 0;
}
