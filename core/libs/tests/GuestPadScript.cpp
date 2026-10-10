#include "prx/libScePad/include/Pad.hpp"
#include "prx/libScePad/include/PadScript.hpp"
#include "prx/libScePad/include/PadState.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

extern "C" {
int APS5_VABI scePadInit_nid_postfix(void);
int APS5_VABI scePadOpen_nid_postfix(int, int, int, const void*);
int APS5_VABI scePadRead_nid_postfix(int, PadData*, int);
int APS5_VABI scePadReadState(int, PadData*);
}

static void Require(bool value) { if (!value) std::abort(); }

constexpr std::uint32_t Options = 0x0008;
constexpr std::uint32_t Up = 0x0010;
constexpr std::uint32_t L2 = 0x0100;
constexpr std::uint32_t Circle = 0x2000;
constexpr std::uint32_t Cross = 0x4000;
constexpr std::uint32_t TouchPad = 0x100000;
constexpr int User = 0x10000000;

static bool RejectsButtons(const char* text) {
    try {
        Pad::ParseScriptButtons(text);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

static bool Rejects(const char* presses, const char* repeats) {
    try {
        Pad::PadScript::Parse(presses, repeats);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

static void CheckButtons() {
    Require(Pad::ParseScriptButtons("cross") == Cross);
    Require(Pad::ParseScriptButtons("Cross+UP") == (Cross | Up));
    Require(Pad::ParseScriptButtons("l2+touchpad") == (L2 | TouchPad));
    Require(Pad::ScriptButtonNames(Cross | Up) == "up+cross");
    Require(Pad::ScriptButtonNames(0) == "none");
    Require(RejectsButtons(""));
    Require(RejectsButtons("x"));
    Require(RejectsButtons("cross+"));
    Require(RejectsButtons("share"));
}

static void CheckParse() {
    Require(Pad::PadScript::Parse("", "").Empty());
    Require(Pad::PadScript::Parse(" ", " ").Empty());
    Require(!Pad::PadScript::Parse("150:cross:200,160:options", "").Empty());
    Require(!Pad::PadScript::Parse("", "200:5:cross,202.5:5:options:100").Empty());
    Require(Pad::PadScript::Parse("150:cross:200,160:options", "").Describe() ==
        "at 150.000s press cross for 200 ms; at 160.000s press options for 200 ms; each press stays down for at least 2 pad reads");
    Require(Pad::PadScript::Parse("", "202.5:5:options+cross:100").Describe() ==
        "from 202.500s every 5.000s press options+cross for 100 ms; each press stays down for at least 2 pad reads");
    Require(Rejects("abc:cross:200", ""));
    Require(Rejects("-1:cross:200", ""));
    Require(Rejects("150.:cross", ""));
    Require(Rejects(".5:cross", ""));
    Require(Rejects("150.1234567:cross", ""));
    Require(Rejects("150:cross:0", ""));
    Require(Rejects("150:cross:2.5", ""));
    Require(Rejects("150:cross:200:1", ""));
    Require(Rejects("150", ""));
    Require(Rejects("150:cross:200,", ""));
    Require(Rejects("150:bogus:200", ""));
    Require(Rejects("", "10:cross"));
    Require(Rejects("", "10:0:cross"));
    Require(Rejects("", "10:1:cross:1000"));
    Require(!Rejects("", "10:1:cross:999"));
}

static void CheckOneShot() {
    Pad::PadScript script = Pad::PadScript::Parse("1.5:cross:100", "");
    Require(script.Sample(1499999) == 0);
    Require(script.Sample(1500000) == Cross);
    Require(script.Sample(1550000) == Cross);
    Require(script.Sample(1600000) == 0);
    Require(script.Sample(9000000) == 0);
}

static void CheckLateRead() {
    Pad::PadScript script = Pad::PadScript::Parse("1:circle:100", "");
    Require(script.Sample(3000000) == Circle);
    Require(script.Sample(3000001) == Circle);
    Require(script.Sample(3000002) == 0);
    Require(script.Sample(4000000) == 0);
}

static void CheckRepeat() {
    Pad::PadScript script = Pad::PadScript::Parse("", "10:5:options:200");
    Require(script.Sample(9999999) == 0);
    Require(script.Sample(10000000) == Options);
    Require(script.Sample(10100000) == Options);
    Require(script.Sample(10200000) == 0);
    Require(script.Sample(12000000) == 0);
    Require(script.Sample(15000000) == Options);
    Require(script.Sample(15300000) == Options);
    Require(script.Sample(15400000) == 0);
    Require(script.Sample(16000000) == 0);
    Require(script.Sample(27000000) == Options);
    Require(script.Sample(27000001) == Options);
    Require(script.Sample(27000002) == 0);
    Require(script.Sample(29000000) == 0);
    Require(script.Sample(30000000) == Options);
}

static void CheckCombined() {
    Pad::PadScript script = Pad::PadScript::Parse("1:cross:100", "1:2:up:100");
    Require(script.Sample(1000000) == (Cross | Up));
    Require(script.Sample(1050000) == (Cross | Up));
    Require(script.Sample(1100000) == 0);
    Require(script.Sample(3000000) == Up);
}

static void CheckUnset() {
    Require(Pad::PadScript::FromEnvironment().Empty());
    Require(scePadInit_nid_postfix() == 0);
    const int handle = scePadOpen_nid_postfix(User, 0, 0, nullptr);
    PadData data{};
    Require(scePadReadState(handle, &data) == 0);
    Require(data.buttons == 0);
    Require(data.connected);
}

static void CheckEnvironment() {
    Require(scePadInit_nid_postfix() == 0);
    const int handle = scePadOpen_nid_postfix(User, 0, 0, nullptr);
    PadData first{};
    PadData second{};
    PadData third{};
    Require(scePadReadState(handle, &first) == 0);
    Require(first.buttons == (Cross | Up));
    Require(scePadRead_nid_postfix(handle, &second, 1) == 1);
    Require(second.buttons == (Cross | Up));
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    Require(scePadReadState(handle, &third) == 0);
    Require(third.buttons == 0);
    Require(third.timestamp > first.timestamp);
}

static void CheckInvalidEnvironment() {
    try {
        scePadInit_nid_postfix();
    } catch (const std::invalid_argument& error) {
        Require(std::string(error.what()).find("unknown pad button 'bogus'") != std::string::npos);
        return;
    }
    std::abort();
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "env") {
        CheckEnvironment();
        return 0;
    }
    if (mode == "invalid-env") {
        CheckInvalidEnvironment();
        return 0;
    }
    CheckButtons();
    CheckParse();
    CheckOneShot();
    CheckLateRead();
    CheckRepeat();
    CheckCombined();
    CheckUnset();
}
