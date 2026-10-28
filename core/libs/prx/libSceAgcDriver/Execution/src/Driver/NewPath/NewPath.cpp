#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/NewPath.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyConfig.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace AgcDriver::NewPath {

namespace {

struct State {
    Router router;
    std::mutex reporting;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

State& Get() {
    static State state;
    return state;
}

bool Profile() {
    static const bool profile = [] {
        const char* value = std::getenv("APS5_PROFILE_NEW_PATH");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return profile;
}

void print(State& state, const char* window) {
    const auto line = Router::Format(state.router.Take(), window);
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
}

}

bool Enabled() {
    static const bool enabled = Graphics::CurrentResidencyConfig().newPath;
    return enabled;
}

Decision RouteCommand(CommandKind kind) {
    auto& state = Get();
    CommandFacts facts;
    facts.kind = kind;
    const auto decision = state.router.Decide(facts);
    state.router.Note(kind, decision);
    if (Profile()) {
        std::unique_lock lock(state.reporting, std::try_to_lock);
        if (lock.owns_lock() && ProfileWindows::ClockDue(state.lastReport)) print(state, "10 s");
    }
    return decision;
}

void Close(const char* window) {
    if (!Enabled() || !Profile()) return;
    auto& state = Get();
    std::lock_guard lock(state.reporting);
    print(state, window);
}

}
