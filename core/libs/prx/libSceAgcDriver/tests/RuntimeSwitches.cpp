#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

using AgcDriver::Graphics::AbSchedule;
using AgcDriver::Graphics::RuntimeSwitch;
namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void SetEnvironment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

std::string Refusal(const char* ab, const char* start = "", const char* settle = "") {
    try {
        AbSchedule::Parse(ab, start, settle);
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}

void registry() {
    Require(RuntimeSwitches::Set(RuntimeSwitch::NoDispatchOverlap), "a switch set in the environment did not start set");
    Require(!RuntimeSwitches::Set(RuntimeSwitch::NoBdaStaging), "a switch unset in the environment started set");
    Require(RuntimeSwitches::Find("APS5_NO_BDA_STAGING") == RuntimeSwitch::NoBdaStaging, "APS5_NO_BDA_STAGING is not a runtime switch");
    Require(RuntimeSwitches::Find("APS5_NO_SPLIT_DRAW_BUILD") == RuntimeSwitch::NoSplitDrawBuild, "APS5_NO_SPLIT_DRAW_BUILD is not a runtime switch");
    Require(RuntimeSwitches::Find("APS5_NO_HTILE_GPU_EXPANSION") == RuntimeSwitch::NoHtileGpuExpansion, "APS5_NO_HTILE_GPU_EXPANSION is not a runtime switch");
    Require(RuntimeSwitches::Find("APS5_NO_GPU_INDIRECT_ARGS") == RuntimeSwitch::NoGpuIndirectArgs && RuntimeSwitches::Find("APS5_NO_BARRIER_BATCH") == RuntimeSwitch::NoBarrierBatch && RuntimeSwitches::Find("APS5_NO_EARLY_REVALIDATE") == RuntimeSwitch::NoEarlyRevalidate && RuntimeSwitches::Find("APS5_NO_BATCHED_STAGING") == RuntimeSwitch::NoBatchedStaging, "a combined-build switch is not a runtime switch");
    Require(!RuntimeSwitches::Find("APS5_ASYNC_LANE").has_value() && !RuntimeSwitches::Find("APS5_SPANNED_WAVE").has_value(), "a startup-only variable was found as a runtime switch");
    for (std::size_t index = 0; index < static_cast<std::size_t>(RuntimeSwitch::Count); ++index) {
        const auto id = static_cast<RuntimeSwitch>(index);
        Require(RuntimeSwitches::Find(RuntimeSwitches::Name(id)) == id, "a switch name does not find its switch");
    }
    RuntimeSwitches::Assign(RuntimeSwitch::NoBdaStaging, true);
    Require(RuntimeSwitches::Set(RuntimeSwitch::NoBdaStaging), "an assigned switch did not read back set");
    RuntimeSwitches::Assign(RuntimeSwitch::NoBdaStaging, false);
    Require(!RuntimeSwitches::Set(RuntimeSwitch::NoBdaStaging), "an assigned switch did not read back unset");
    Require(RuntimeSwitches::KeySalt() == 0, "the key salt is not 0 before the A/B starts");
}

void parse() {
    const auto config = AbSchedule::Parse(" 30 : APS5_NO_BDA_STAGING , APS5_NO_SPLIT_DRAW_BUILD", "815", "3");
    Require(config.phaseSeconds == 30 && config.settleSeconds == 3 && config.startPresent == 815 && !config.startSeconds.has_value(), "the phase settings were not parsed");
    Require(config.switches.size() == 2 && config.switches[0] == RuntimeSwitch::NoBdaStaging && config.switches[1] == RuntimeSwitch::NoSplitDrawBuild, "the switch list was not parsed in order");
    const auto bySeconds = AbSchedule::Parse("60:APS5_NO_DISPATCH_OVERLAP", "90s", "");
    Require(bySeconds.startSeconds == 90.0 && bySeconds.settleSeconds == 2 && bySeconds.startPresent == 0, "a start in seconds was not parsed");
    Require(Refusal("60:APS5_ASYNC_LANE").find("APS5_ASYNC_LANE cannot change while the game runs") != std::string::npos, "APS5_ASYNC_LANE was not refused with its reason");
    Require(Refusal("60:APS5_SPANNED_WAVE").find("read once at startup") != std::string::npos, "a startup-only variable was not refused");
    Require(Refusal("60:APS5_NO_BDA_STAGING,APS5_NO_BDA_STAGING").find("listed twice") != std::string::npos, "a duplicate switch was not refused");
    Require(!Refusal("APS5_NO_BDA_STAGING").empty(), "a missing phase length was not refused");
    Require(!Refusal("60:").empty(), "an empty switch list was not refused");
    Require(!Refusal("0.5:APS5_NO_BDA_STAGING").empty(), "a phase under 1 s was not refused");
    Require(!Refusal("x:APS5_NO_BDA_STAGING").empty(), "a bad phase length was not refused");
    Require(!Refusal("10:APS5_NO_BDA_STAGING", "", "10").empty(), "a settle as long as the phase was not refused");
    Require(!Refusal("10:APS5_NO_BDA_STAGING", "-3").empty() && !Refusal("10:APS5_NO_BDA_STAGING", "abc").empty(), "a bad start was not refused");
}

void schedule() {
    AbSchedule plan(AbSchedule::Parse("60:APS5_NO_BDA_STAGING", "10", "2"));
    using Event = AbSchedule::Event;
    const auto expect = [&](std::uint64_t present, double seconds, Event event, std::uint64_t phase, bool b) {
        const auto step = plan.AtPresent(present, seconds);
        Require(step.event == event && (event == Event::None || (step.phase == phase && step.b == b)), "unexpected step at present " + std::to_string(present));
    };
    expect(1, 0.0, Event::None, 0, false);
    expect(9, 100.0, Event::None, 0, false);
    expect(10, 102.0, Event::Begin, 0, false);
    expect(11, 103.0, Event::None, 0, false);
    expect(12, 104.5, Event::Settled, 0, false);
    expect(13, 150.0, Event::None, 0, false);
    expect(14, 161.9, Event::None, 0, false);
    expect(15, 162.0, Event::Begin, 1, true);
    expect(16, 163.0, Event::None, 1, true);
    expect(17, 300.0, Event::Begin, 2, false);
    expect(18, 301.0, Event::None, 2, false);
    expect(19, 302.5, Event::Settled, 2, false);
    expect(20, 362.5, Event::Begin, 3, true);

    AbSchedule timed(AbSchedule::Parse("30:APS5_NO_BDA_STAGING", "90s", ""));
    Require(timed.AtPresent(500, 1000.0).event == Event::None, "a start in seconds began at the first present");
    Require(timed.AtPresent(501, 1089.0).event == Event::None, "a start in seconds began early");
    const auto begin = timed.AtPresent(502, 1090.0);
    Require(begin.event == Event::Begin && begin.phase == 0 && !begin.b, "a start in seconds did not begin phase A");
}

void glue() {
    Require(AgcDriver::Graphics::AbConfigured(), "APS5_AB is not seen");
    Require(AgcDriver::Graphics::AbAtPresent(1, 0.0).empty(), "the A/B printed before its start");
    Require(RuntimeSwitches::KeySalt() == 0, "the key salt changed before the start");
    auto lines = AgcDriver::Graphics::AbAtPresent(3, 1.0);
    Require(lines.size() == 2 && lines[0].rfind("[ab] in-run A/B: phases of 5.0 s from present 3", 0) == 0 && lines[1] == "[ab] phase A 0 begins at present 3 (t 1.000 s): APS5_NO_BDA_STAGING unset, APS5_NO_DISPATCH_OVERLAP set\n", "the start lines differ: " + (lines.empty() ? std::string() : lines.back()));
    Require(!RuntimeSwitches::Set(RuntimeSwitch::NoBdaStaging) && RuntimeSwitches::Set(RuntimeSwitch::NoDispatchOverlap), "phase A does not have the environment's settings");
    const auto saltA = RuntimeSwitches::KeySalt();
    Require(saltA != 0, "phase A has no key salt");
    lines = AgcDriver::Graphics::AbAtPresent(4, 3.5);
    Require(lines.size() == 1 && lines[0] == "[ab] phase A 0 settled at present 4 (t 3.500 s)\n", "the settle line differs");
    lines = AgcDriver::Graphics::AbAtPresent(5, 6.0);
    Require(lines.size() == 1 && lines[0] == "[ab] phase B 1 begins at present 5 (t 6.000 s): APS5_NO_BDA_STAGING set, APS5_NO_DISPATCH_OVERLAP unset\n", "the phase B line differs");
    Require(RuntimeSwitches::Set(RuntimeSwitch::NoBdaStaging) && !RuntimeSwitches::Set(RuntimeSwitch::NoDispatchOverlap), "phase B did not toggle the listed switches");
    Require(!RuntimeSwitches::Set(RuntimeSwitch::NoSplitDrawBuild), "phase B toggled a switch it does not list");
    Require(RuntimeSwitches::KeySalt() != 0 && RuntimeSwitches::KeySalt() != saltA, "phase B has phase A's key salt");
    Require(AgcDriver::Graphics::AbAtPresent(6, 7.0).empty(), "a present inside the settle printed");
    lines = AgcDriver::Graphics::AbAtPresent(7, 11.5);
    Require(lines.size() == 1 && lines[0].rfind("[ab] phase A 2 begins at present 7", 0) == 0, "phase A did not come back");
    Require(!RuntimeSwitches::Set(RuntimeSwitch::NoBdaStaging) && RuntimeSwitches::Set(RuntimeSwitch::NoDispatchOverlap) && RuntimeSwitches::KeySalt() == saltA, "phase A did not restore the environment's settings");
}

}

int main() {
    try {
        SetEnvironment("APS5_NO_DISPATCH_OVERLAP", "1");
        SetEnvironment("APS5_NO_BDA_STAGING", nullptr);
        SetEnvironment("APS5_NO_SPLIT_DRAW_BUILD", nullptr);
        SetEnvironment("APS5_AB", "5:APS5_NO_BDA_STAGING,APS5_NO_DISPATCH_OVERLAP");
        SetEnvironment("APS5_AB_START", "3");
        SetEnvironment("APS5_AB_SETTLE", nullptr);
        registry();
        parse();
        schedule();
        glue();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
    std::printf("runtime switch tests passed\n");
    return 0;
}
