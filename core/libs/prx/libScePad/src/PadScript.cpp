#include "prx/libScePad/include/PadScript.hpp"
#include "prx/libc/include/General.hpp"
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

struct ButtonName {
    std::string_view name;
    std::uint32_t bit;
};

constexpr ButtonName ButtonNames[] = {
    {"l3", 0x0002}, {"r3", 0x0004}, {"options", 0x0008}, {"up", 0x0010}, {"right", 0x0020}, {"down", 0x0040},
    {"left", 0x0080}, {"l2", 0x0100}, {"r2", 0x0200}, {"l1", 0x0400}, {"r1", 0x0800}, {"triangle", 0x1000},
    {"circle", 0x2000}, {"cross", 0x4000}, {"square", 0x8000}, {"touchpad", 0x100000}};

constexpr std::string_view PressVariable = "APS5_PAD_SCRIPT";
constexpr std::string_view RepeatVariable = "APS5_PAD_SCRIPT_REPEAT";
constexpr std::uint64_t MaxSecondsInteger = 1000000000ull;
constexpr std::uint64_t MaxHoldMs = 3600000ull;

std::string_view Trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) text.remove_suffix(1);
    return text;
}

std::vector<std::string_view> Split(std::string_view text, char separator) {
    std::vector<std::string_view> parts;
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = text.find(separator, begin);
        parts.push_back(Trim(text.substr(begin, end == std::string_view::npos ? std::string_view::npos : end - begin)));
        if (end == std::string_view::npos) return parts;
        begin = end + 1;
    }
}

[[noreturn]] void Invalid(std::string_view variable, std::string_view item, std::string_view reason) {
    throw std::invalid_argument(std::string(variable) + ": " + std::string(reason) + " in '" + std::string(item) + "'");
}

bool ParseDigits(std::string_view text, std::uint64_t limit, std::uint64_t* value) {
    if (text.empty()) return false;
    std::uint64_t result = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') return false;
        result = result * 10 + static_cast<std::uint64_t>(c - '0');
        if (result > limit) return false;
    }
    *value = result;
    return true;
}

std::uint64_t ParseSecondsUs(std::string_view text, std::string_view variable, std::string_view item, std::string_view what) {
    const std::size_t dot = text.find('.');
    const std::string_view whole = text.substr(0, dot);
    const std::string_view fraction = dot == std::string_view::npos ? std::string_view{} : text.substr(dot + 1);
    std::uint64_t seconds = 0;
    std::uint64_t micros = 0;
    const bool fractionValid = dot == std::string_view::npos || (!fraction.empty() && fraction.size() <= 6 && ParseDigits(fraction, 999999, &micros));
    if (!ParseDigits(whole, MaxSecondsInteger, &seconds) || !fractionValid) {
        Invalid(variable, item, std::string("expected ") + std::string(what) + " as decimal seconds (up to 6 fraction digits)");
    }
    for (std::size_t digits = fraction.size(); digits < 6; ++digits) micros *= 10;
    return seconds * 1000000ull + micros;
}

std::uint64_t ParseHoldUs(std::string_view text, std::string_view variable, std::string_view item) {
    std::uint64_t ms = 0;
    if (!ParseDigits(text, MaxHoldMs, &ms) || ms == 0) Invalid(variable, item, "expected the hold time as whole milliseconds from 1 to 3600000");
    return ms * 1000ull;
}

std::uint32_t ParseButtonsIn(std::string_view text, std::string_view variable, std::string_view item) {
    try {
        return Pad::ParseScriptButtons(text);
    } catch (const std::invalid_argument& error) {
        Invalid(variable, item, error.what());
    }
}

std::string FormatSeconds(std::uint64_t us) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.3fs", static_cast<double>(us) * 1e-6);
    return text;
}

constexpr std::size_t TracedCallCount = 4;
constexpr const char* TracedCallNames[TracedCallCount] = {
    "scePadRead", "scePadReadState", "scePadGetControllerInformation", "scePadGetExtControllerInformation"};
constexpr std::size_t TracedCallerCount = 4;
constexpr std::uint64_t TraceWindowUs = 1000000;

struct TraceState {
    std::mutex mutex;
    std::uint64_t windowStartUs = 0;
    std::uint64_t lastCallUs = 0;
    std::array<std::uint64_t, TracedCallCount> window{};
    std::array<std::uint64_t, TracedCallCount> total{};
    std::array<const void*, TracedCallerCount> callers{};
    std::array<std::uint64_t, TracedCallerCount> callerCounts{};
    std::uint64_t otherCalls = 0;
    std::uint32_t lastButtons = 0;
};

TraceState& Trace() {
    static TraceState state;
    return state;
}

std::string CallerText(const void* caller) {
#ifdef _WIN32
    const auto image = reinterpret_cast<std::uintptr_t>(GetModuleHandleW(nullptr));
#else
    const std::uintptr_t image = 0;
#endif
    char text[40];
    std::snprintf(text, sizeof(text), "exe+0x%llx", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(caller) - image));
    return text;
}

void ReportWindow(TraceState& trace, std::uint64_t endUs) {
    std::string line;
    char part[96];
    for (std::size_t index = 0; index < TracedCallCount; ++index) {
        if (trace.window[index] == 0) continue;
        std::snprintf(part, sizeof(part), "%s%s %llu (total %llu)", line.empty() ? "" : ", ", TracedCallNames[index],
            static_cast<unsigned long long>(trace.window[index]), static_cast<unsigned long long>(trace.total[index]));
        line += part;
    }
    if (line.empty()) return;
    line += "; callers";
    for (std::size_t slot = 0; slot < TracedCallerCount && trace.callers[slot] != nullptr; ++slot) {
        std::snprintf(part, sizeof(part), " %s x%llu", CallerText(trace.callers[slot]).c_str(), static_cast<unsigned long long>(trace.callerCounts[slot]));
        line += part;
    }
    if (trace.otherCalls != 0) {
        std::snprintf(part, sizeof(part), " other x%llu", static_cast<unsigned long long>(trace.otherCalls));
        line += part;
    }
    APS5_LOG_OUT("[pad-trace] t=%.3fs %.3fs window: %s; last buttons %s", static_cast<double>(endUs) * 1e-6,
        static_cast<double>(endUs - trace.windowStartUs) * 1e-6, line.c_str(), Pad::ScriptButtonNames(trace.lastButtons).c_str());
    trace.window = {};
    trace.callers = {};
    trace.callerCounts = {};
    trace.otherCalls = 0;
}

}

std::uint32_t Pad::ParseScriptButtons(std::string_view text) {
    std::uint32_t buttons = 0;
    for (const std::string_view part : Split(text, '+')) {
        std::string name(part);
        for (char& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        std::uint32_t bit = 0;
        for (const ButtonName& candidate : ButtonNames) {
            if (candidate.name == name) bit = candidate.bit;
        }
        if (bit == 0) {
            std::string known;
            for (const ButtonName& candidate : ButtonNames) known += (known.empty() ? "" : ", ") + std::string(candidate.name);
            throw std::invalid_argument("unknown pad button '" + std::string(part) + "' (known: " + known + ")");
        }
        buttons |= bit;
    }
    return buttons;
}

std::string Pad::ScriptButtonNames(std::uint32_t buttons) {
    std::string names;
    for (const ButtonName& candidate : ButtonNames) {
        if ((buttons & candidate.bit) == 0) continue;
        names += (names.empty() ? "" : "+") + std::string(candidate.name);
        buttons &= ~candidate.bit;
    }
    if (buttons != 0) {
        char rest[24];
        std::snprintf(rest, sizeof(rest), "0x%x", buttons);
        names += (names.empty() ? "" : "+") + std::string(rest);
    }
    return names.empty() ? "none" : names;
}

Pad::PadScript Pad::PadScript::Parse(std::string_view presses, std::string_view repeats) {
    PadScript script;
    if (!Trim(presses).empty()) {
        for (const std::string_view item : Split(presses, ',')) {
            const auto fields = Split(item, ':');
            if (fields.size() < 2 || fields.size() > 3) Invalid(PressVariable, item, "expected <seconds>:<buttons>[:<hold ms>]");
            Entry entry;
            entry.press.startUs = ParseSecondsUs(fields[0], PressVariable, item, "the press time");
            entry.press.buttons = ParseButtonsIn(fields[1], PressVariable, item);
            entry.press.holdUs = fields.size() == 3 ? ParseHoldUs(fields[2], PressVariable, item) : DefaultHoldMs * 1000ull;
            script.entries.push_back(entry);
        }
    }
    if (!Trim(repeats).empty()) {
        for (const std::string_view item : Split(repeats, ',')) {
            const auto fields = Split(item, ':');
            if (fields.size() < 3 || fields.size() > 4) Invalid(RepeatVariable, item, "expected <from s>:<every s>:<buttons>[:<hold ms>]");
            Entry entry;
            entry.press.startUs = ParseSecondsUs(fields[0], RepeatVariable, item, "the first press time");
            entry.press.everyUs = ParseSecondsUs(fields[1], RepeatVariable, item, "the period");
            entry.press.buttons = ParseButtonsIn(fields[2], RepeatVariable, item);
            entry.press.holdUs = fields.size() == 4 ? ParseHoldUs(fields[3], RepeatVariable, item) : DefaultHoldMs * 1000ull;
            if (entry.press.holdUs >= entry.press.everyUs) Invalid(RepeatVariable, item, "the hold time must be shorter than the period");
            script.entries.push_back(entry);
        }
    }
    return script;
}

Pad::PadScript Pad::PadScript::FromEnvironment() {
    const char* presses = std::getenv("APS5_PAD_SCRIPT");
    const char* repeats = std::getenv("APS5_PAD_SCRIPT_REPEAT");
    return Parse(presses != nullptr ? presses : "", repeats != nullptr ? repeats : "");
}

std::uint32_t Pad::PadScript::Sample(std::uint64_t nowUs) {
    std::uint32_t buttons = 0;
    for (Entry& entry : entries) {
        const ScriptPress& press = entry.press;
        if (entry.holding) {
            const std::uint64_t endUs = press.startUs + entry.cycle * press.everyUs + press.holdUs;
            if (nowUs >= endUs && entry.reads >= MinReads) {
                entry.holding = false;
                entry.nextCycle = entry.cycle + 1;
                continue;
            }
        } else {
            if (nowUs < press.startUs) continue;
            const std::uint64_t current = press.everyUs == 0 ? 0 : (nowUs - press.startUs) / press.everyUs;
            if (current < entry.nextCycle) continue;
            entry.holding = true;
            entry.cycle = current;
            entry.reads = 0;
        }
        ++entry.reads;
        buttons |= press.buttons;
    }
    return buttons;
}

std::string Pad::PadScript::Describe() const {
    std::string text;
    for (const Entry& entry : entries) {
        const ScriptPress& press = entry.press;
        text += text.empty() ? "" : "; ";
        if (press.everyUs == 0) text += "at " + FormatSeconds(press.startUs);
        else text += "from " + FormatSeconds(press.startUs) + " every " + FormatSeconds(press.everyUs);
        text += " press " + ScriptButtonNames(press.buttons) + " for " + std::to_string(press.holdUs / 1000) + " ms";
    }
    return text + "; each press stays down for at least " + std::to_string(MinReads) + " pad reads";
}

bool Pad::TraceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_TRACE_PAD");
        return value != nullptr && *value != '\0' && std::string_view(value) != "0";
    }();
    return enabled;
}

std::uint64_t Pad::MicrosSinceProcessStart() {
    using Clock = std::chrono::steady_clock;
    static const std::pair<Clock::time_point, std::uint64_t> origin = [] {
        std::uint64_t ageUs = 0;
#ifdef _WIN32
        FILETIME creation{};
        FILETIME exitTime{};
        FILETIME kernelTime{};
        FILETIME userTime{};
        if (!GetProcessTimes(GetCurrentProcess(), &creation, &exitTime, &kernelTime, &userTime)) {
            throw std::runtime_error("Pad: GetProcessTimes failed");
        }
        FILETIME now{};
        GetSystemTimeAsFileTime(&now);
        const auto ticks = [](const FILETIME& time) {
            return (static_cast<std::uint64_t>(time.dwHighDateTime) << 32) | time.dwLowDateTime;
        };
        if (ticks(now) > ticks(creation)) ageUs = (ticks(now) - ticks(creation)) / 10;
#endif
        return std::pair{Clock::now(), ageUs};
    }();
    return origin.second + static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - origin.first).count());
}

double Pad::ProcessSeconds() {
    return static_cast<double>(MicrosSinceProcessStart()) * 1e-6;
}

void Pad::TraceCall(TracedCall call, const void* caller, std::uint32_t buttons) {
    const std::uint64_t nowUs = MicrosSinceProcessStart();
    TraceState& trace = Trace();
    std::lock_guard lock(trace.mutex);
    const auto index = static_cast<std::size_t>(call);
    if (trace.total[index] == 0) {
        APS5_LOG_OUT("[pad-trace] t=%.3fs first %s call from %s", static_cast<double>(nowUs) * 1e-6, TracedCallNames[index], CallerText(caller).c_str());
    }
    if (trace.lastCallUs != 0 && nowUs - trace.lastCallUs >= TraceWindowUs) {
        ReportWindow(trace, trace.lastCallUs);
        APS5_LOG_OUT("[pad-trace] t=%.3fs %s after %.3fs without traced pad calls", static_cast<double>(nowUs) * 1e-6,
            TracedCallNames[index], static_cast<double>(nowUs - trace.lastCallUs) * 1e-6);
        trace.windowStartUs = nowUs;
    }
    if (trace.windowStartUs == 0) trace.windowStartUs = nowUs;
    trace.lastCallUs = nowUs;
    ++trace.total[index];
    ++trace.window[index];
    if (call == TracedCall::Read || call == TracedCall::ReadState) trace.lastButtons = buttons;
    std::size_t slot = 0;
    while (slot < TracedCallerCount && trace.callers[slot] != nullptr && trace.callers[slot] != caller) ++slot;
    if (slot == TracedCallerCount) {
        ++trace.otherCalls;
    } else {
        trace.callers[slot] = caller;
        ++trace.callerCounts[slot];
    }
    if (nowUs - trace.windowStartUs >= TraceWindowUs) {
        ReportWindow(trace, nowUs);
        trace.windowStartUs = nowUs;
    }
}
