#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/NewPath.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyConfig.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencySession.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DevicePageTable.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include <cstdio>
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
    const auto table = DeviceTable().Totals();
    AgcDriver::ProfilePrint_nid_no_patch("%s; page table builds %llu, patches %llu, ranges %llu, covered %.1f MiB\n", line.c_str(), static_cast<unsigned long long>(table.builds), static_cast<unsigned long long>(table.patches), static_cast<unsigned long long>(table.ranges), static_cast<double>(table.coveredBytes) / (1024.0 * 1024.0));
    if (const auto residency = Graphics::ResidencyReportLine(); !residency.empty()) AgcDriver::ProfilePrint_nid_no_patch("%s\n", residency.c_str());
}

}

bool Enabled() {
    static const bool enabled = Graphics::CurrentResidencyConfig().newPath;
    return enabled;
}

Decision RouteCommand(CommandKind kind, const VulkanDevice* device) {
    auto& state = Get();
    if (device != nullptr && DeviceTable().Stale(device->Device())) static_cast<void>(DeviceTable().Refresh(device->graphicsContext()));
    if (device != nullptr && Graphics::CurrentResidencyConfig().residency && Graphics::Recorder::Active() != nullptr && !Graphics::ResidencyEnsured(device->Device())) Graphics::EnsureResidency(device->graphicsContext());
    CommandFacts facts;
    facts.kind = kind;
    facts.kindSwitchOff = Graphics::RuntimeSwitches::Set(kind == CommandKind::Draw ? Graphics::RuntimeSwitch::NewPathNoGpuDraws : Graphics::RuntimeSwitch::NewPathNoGpuDispatches);
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
