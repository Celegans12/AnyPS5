#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace AgcDriver::Graphics {

namespace {

constexpr std::size_t SwitchCount = static_cast<std::size_t>(RuntimeSwitch::Count);

constexpr std::array<const char*, SwitchCount> SwitchNames{
    "APS5_NO_SPLIT_DRAW_BUILD",
    "APS5_NO_BDA_STAGING",
    "APS5_NO_DISPATCH_OVERLAP",
    "APS5_NO_HTILE_GPU_EXPANSION",
    "APS5_NO_GPU_INDIRECT_ARGS",
    "APS5_NO_BARRIER_BATCH",
    "APS5_NO_EARLY_REVALIDATE",
    "APS5_NO_BATCHED_STAGING",
    "APS5_QUIET_PUBLISH",
    "APS5_NO_STAGED_OVERLAP",
    "APS5_NO_OWED_RUN_CLOSE",
    "APS5_NO_RUN_ATOMIC_GUARD",
    "APS5_DRAW_INPUTS_IN_PLACE",
    "APS5_NO_RUN_STAGED_READ_GUARD",
    "APS5_NO_SMALL_DEVICE_TIER",
    "APS5_NO_EXACT_SIZE_REUSE",
    "APS5_NO_HELD_READ_SKIP",
    "APS5_NO_LANE_OPEN_SERIAL",
    "APS5_NO_LANE_TEMPLATE_EDGE",
    "APS5_LANE_SERIALIZE",
    "APS5_NO_LANE_OVERLAP",
    "APS5_NO_LANE_LABEL_SCOPE",
    "APS5_NO_LANE_FORWARD_EDGE",
    "APS5_NO_LANE_LEASE_WAIT",
    "APS5_NO_IMAGE_SLABS",
    "APS5_NO_PREPARED_TABLE_COUNT",
    "APS5_NO_OVERLAP_RANGE_CHECK",
    "APS5_NO_DRAW_AHEAD",
    "APS5_NO_PRESENT_FRAME_AGE",
    "APS5_NO_BDA_TEMPLATES",
    "APS5_NO_SLICE_COPY",
    "APS5_DEFER_REFRESH_FLUSH",
    "APS5_NO_DEPTH_INDEX",
    "APS5_NO_DEFERRED_DRAW_EVICTION",
    "APS5_NO_INCREMENTAL_BDA_RANGES",
    "APS5_NO_READ_ONLY_LEASE_SKIP",
    "APS5_NO_LEAN_BDA_TABLES",
    "APS5_NO_DEPTH_LOOKUP_FAST",
    "APS5_NO_REFUSAL_MEMO",
    "APS5_NO_DISTINCT_WRITE_SCANS",
    "APS5_NO_COLLECT_HEAP",
    "APS5_NO_LARGER_SMALL_POOLS",
    "APS5_DEFER_FLIP_STORE",
    "APS5_NO_DEFER_IMPORTED_STORES",
    "APS5_FLUSH_WAIT_WHOLE_RANGE",
    "APS5_NO_REFRESH_LOOKUP_CACHE",
    "APS5_PIPELINE_DISPATCH",
    "APS5_PIPELINE_DISPATCH_MISSES",
    "APS5_NO_OWNED_PROOFS",
    "APS5_NO_OWNED_PRIVATE",
    "APS5_NO_HIT_ITEM_HTILE_NOTE",
    "APS5_GPU_SOURCED_DRAW_BUFFERS",
    "APS5_IMPORT_SOURCED_COPY_IN",
    "APS5_NO_OWNED_DRAW_SNAPSHOTS",
    "APS5_NO_GPU_SOURCE_AFTER_WRITER",
    "APS5_NO_OWNED_VERTEX_INPUTS",
    "APS5_NO_FLUSH_LANE_WAIT",
    "APS5_NO_OWNED_EPOCH",
    "APS5_NEW_PATH_NO_GPU_DRAWS",
    "APS5_NEW_PATH_NO_GPU_DISPATCHES",
    "APS5_NO_INDEX_CLAMP_MEMO",
    "APS5_NO_GDS_GPU_FILL",
    "APS5_NO_DMA_COPY_BEHIND_WRITES",
    "APS5_NO_SLICE_EDGE_HASH",
    "APS5_NO_CLEAR_SNAPSHOT_REUSE",
    "APS5_NO_GPU_INDEX_CLAMP",
    "APS5_NO_GPU_WRITTEN_IN_PLACE",
    "APS5_NO_INCREMENTAL_SHADOW_COUNT",
};

struct Refusal {
    const char* name;
    const char* reason;
};

constexpr std::array<Refusal, 3> Refused{{
    {"APS5_ASYNC_LANE", "it selects the device's queues at creation, and a label stored in one lane and consumed after the change would lose its GPU edge"},
    {"APS5_NEW_PATH", "it selects the driver's memory and command model at device creation; the new path against the old one is read from separate long runs (A-B-A)"},
    {"APS5_NEW_PATH_NO_RESIDENCY", "the residency table owns resident and guarded pages from device creation on; turning it off mid-run would leave pages no reader publishes"},
}};

struct SwitchValues {
    std::array<std::atomic<bool>, SwitchCount> set{};
    SwitchValues() {
        for (std::size_t index = 0; index < SwitchCount; ++index) set[index].store(std::getenv(SwitchNames[index]) != nullptr, std::memory_order_relaxed);
    }
};

SwitchValues& Values() {
    static SwitchValues values;
    return values;
}

std::atomic<std::uint32_t> keySalt{0};

std::string_view Trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r' || text.back() == '\n')) text.remove_suffix(1);
    return text;
}

double ParseSeconds(std::string_view text, const char* what) {
    const std::string copy(Trim(text));
    char* end = nullptr;
    const double value = std::strtod(copy.c_str(), &end);
    if (copy.empty() || end != copy.c_str() + copy.size() || !std::isfinite(value) || value < 0) throw std::invalid_argument(std::string(what) + ": '" + copy + "' is not a number of seconds");
    return value;
}

std::string ArmText(const std::vector<RuntimeSwitch>& arm) {
    std::string text;
    for (const auto id : arm) text += std::string(text.empty() ? "" : "+") + SwitchNames[static_cast<std::size_t>(id)];
    return text;
}

std::string FlippableList() {
    std::string list;
    for (const auto* name : SwitchNames) list += std::string(list.empty() ? "" : ", ") + name;
    return list;
}

}

namespace RuntimeSwitches {

const char* Name(RuntimeSwitch id) {
    return SwitchNames.at(static_cast<std::size_t>(id));
}

bool Set(RuntimeSwitch id) {
    return Values().set[static_cast<std::size_t>(id)].load(std::memory_order_relaxed);
}

void Assign(RuntimeSwitch id, bool set) {
    Values().set.at(static_cast<std::size_t>(id)).store(set, std::memory_order_relaxed);
}

std::optional<RuntimeSwitch> Find(std::string_view name) {
    for (std::size_t index = 0; index < SwitchCount; ++index) {
        if (name == SwitchNames[index]) return static_cast<RuntimeSwitch>(index);
    }
    return std::nullopt;
}

std::uint32_t KeySalt() {
    return keySalt.load(std::memory_order_relaxed);
}

}

AbSchedule::Config AbSchedule::Parse(std::string_view ab, std::string_view start, std::string_view settle) {
    Config config;
    ab = Trim(ab);
    const auto colon = ab.find(':');
    if (colon == std::string_view::npos) throw std::invalid_argument("APS5_AB: expected <seconds>:<switch>[,<switch>...], got '" + std::string(ab) + "'");
    config.phaseSeconds = ParseSeconds(ab.substr(0, colon), "APS5_AB phase length");
    if (config.phaseSeconds < 1) throw std::invalid_argument("APS5_AB: a phase must last at least 1 s");
    auto list = ab.substr(colon + 1);
    while (!list.empty()) {
        const auto comma = list.find(',');
        auto group = list.substr(0, comma);
        list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
        std::vector<RuntimeSwitch> arm;
        while (!group.empty()) {
            const auto plus = group.find('+');
            const auto name = Trim(group.substr(0, plus));
            group = plus == std::string_view::npos ? std::string_view{} : group.substr(plus + 1);
            if (name.empty()) continue;
            const auto found = RuntimeSwitches::Find(name);
            if (!found.has_value()) {
                for (const auto& refusal : Refused) {
                    if (name == refusal.name) throw std::invalid_argument("APS5_AB: " + std::string(name) + " cannot change while the game runs: " + refusal.reason);
                }
                throw std::invalid_argument("APS5_AB: " + std::string(name) + " is read once at startup and cannot change while the game runs; switches the A/B can flip: " + FlippableList());
            }
            if (std::find(arm.begin(), arm.end(), *found) != arm.end()) throw std::invalid_argument("APS5_AB: " + std::string(name) + " is listed twice in one arm");
            arm.push_back(*found);
        }
        if (arm.empty()) continue;
        for (const auto& other : config.arms) {
            if (other.size() == arm.size() && std::is_permutation(other.begin(), other.end(), arm.begin())) throw std::invalid_argument("APS5_AB: the arm " + ArmText(arm) + " is listed twice");
        }
        for (const auto id : arm) {
            if (std::find(config.switches.begin(), config.switches.end(), id) == config.switches.end()) config.switches.push_back(id);
        }
        config.arms.push_back(std::move(arm));
    }
    if (config.switches.empty()) throw std::invalid_argument("APS5_AB: no switch listed after the phase length");
    start = Trim(start);
    if (!start.empty()) {
        if (start.back() == 's') {
            config.startSeconds = ParseSeconds(start.substr(0, start.size() - 1), "APS5_AB_START");
        } else {
            const std::string copy(start);
            char* end = nullptr;
            const auto value = std::strtoull(copy.c_str(), &end, 10);
            if (end != copy.c_str() + copy.size() || copy.front() == '-') throw std::invalid_argument("APS5_AB_START: '" + copy + "' is neither a present index nor <n>s");
            config.startPresent = value;
        }
    }
    if (!Trim(settle).empty()) config.settleSeconds = ParseSeconds(settle, "APS5_AB_SETTLE");
    if (config.settleSeconds >= config.phaseSeconds) throw std::invalid_argument("APS5_AB_SETTLE must be shorter than the phase");
    return config;
}

std::size_t AbSchedule::ArmOf(std::uint64_t phase, std::size_t arms) {
    if ((phase & 1u) == 0 || arms == 0) return 0;
    return static_cast<std::size_t>(((phase - 1) / 2) % arms) + 1;
}

AbSchedule::AbSchedule(Config config) : config(std::move(config)) {}

AbSchedule::Step AbSchedule::AtPresent(std::uint64_t present, double seconds) {
    if (!firstSeconds.has_value()) firstSeconds = seconds;
    if (!started) {
        const bool due = config.startSeconds.has_value() ? seconds - *firstSeconds >= *config.startSeconds : present >= config.startPresent;
        if (!due) return {};
        started = true;
        settled = false;
        phase = 0;
        phaseStart = seconds;
        return {Event::Begin, 0, false, 0};
    }
    if (seconds - phaseStart >= config.phaseSeconds) {
        ++phase;
        settled = false;
        phaseStart = seconds;
        return {Event::Begin, phase, (phase & 1u) != 0, ArmOf(phase, config.arms.size())};
    }
    if (!settled && seconds - phaseStart >= config.settleSeconds) {
        settled = true;
        return {Event::Settled, phase, (phase & 1u) != 0, ArmOf(phase, config.arms.size())};
    }
    return {};
}

namespace {

std::string Fixed1(double value) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.1f", value);
    return text;
}

const char* EnvText(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? value : "";
}

struct AbState {
    std::mutex mutex;
    std::optional<AbSchedule> schedule;
    std::array<bool, SwitchCount> environment{};
};

AbState& Ab() {
    static AbState* state = [] {
        auto* created = new AbState();
        if (std::getenv("APS5_AB") != nullptr) {
            created->schedule.emplace(AbSchedule::Parse(EnvText("APS5_AB"), EnvText("APS5_AB_START"), EnvText("APS5_AB_SETTLE")));
            for (std::size_t index = 0; index < SwitchCount; ++index) created->environment[index] = RuntimeSwitches::Set(static_cast<RuntimeSwitch>(index));
        }
        return created;
    }();
    return *state;
}

bool ArmSets(const AbState& state, RuntimeSwitch id, std::size_t arm) {
    const auto& arms = state.schedule->Settings().arms;
    const bool toggled = arm != 0 && std::find(arms[arm - 1].begin(), arms[arm - 1].end(), id) != arms[arm - 1].end();
    return state.environment[static_cast<std::size_t>(id)] != toggled;
}

std::string SwitchSettings(const AbState& state, std::size_t arm) {
    std::string text;
    for (const auto id : state.schedule->Settings().switches) {
        text += std::string(text.empty() ? "" : ", ") + RuntimeSwitches::Name(id) + (ArmSets(state, id, arm) ? " set" : " unset");
    }
    return text;
}

}

bool AbConfigured() {
    static const bool configured = std::getenv("APS5_AB") != nullptr;
    return configured;
}

std::vector<std::string> AbAtPresent(std::uint64_t present, double seconds) {
    if (!AbConfigured()) return {};
    auto& state = Ab();
    std::lock_guard lock(state.mutex);
    std::vector<std::string> lines;
    char text[512];
    const auto& config = state.schedule->Settings();
    const auto step = state.schedule->AtPresent(present, seconds);
    if (step.event == AbSchedule::Event::None) return lines;
    if (step.event == AbSchedule::Event::Begin) {
        if (step.phase == 0) {
            char from[96];
            if (config.startSeconds.has_value()) std::snprintf(from, sizeof(from), "%.1f s after the first present", *config.startSeconds);
            else std::snprintf(from, sizeof(from), "present %llu", static_cast<unsigned long long>(config.startPresent));
            std::string arms;
            if (config.arms.size() == 1) {
                arms = "A: " + SwitchSettings(state, 0) + "; B: " + SwitchSettings(state, 1);
            } else {
                arms = std::to_string(config.arms.size()) + " arms; A: " + SwitchSettings(state, 0);
                for (std::size_t arm = 1; arm <= config.arms.size(); ++arm) arms += "; B" + std::to_string(arm) + " (" + ArmText(config.arms[arm - 1]) + "): " + SwitchSettings(state, arm);
            }
            lines.push_back("[ab] in-run A/B: phases of " + Fixed1(config.phaseSeconds) + " s from " + from + ", settle " + Fixed1(config.settleSeconds) + " s; " + arms + "\n");
        }
        for (const auto id : config.switches) RuntimeSwitches::Assign(id, ArmSets(state, id, step.arm));
        keySalt.store(step.arm == 0 ? 0xab00000au : 0xab00000bu + (static_cast<std::uint32_t>(step.arm - 1) << 8), std::memory_order_relaxed);
        std::snprintf(text, sizeof(text), "[ab] phase %c %llu begins at present %llu (t %.3f s) arm %llu: ", step.b ? 'B' : 'A', static_cast<unsigned long long>(step.phase), static_cast<unsigned long long>(present), seconds, static_cast<unsigned long long>(step.arm));
        lines.push_back(text + SwitchSettings(state, step.arm) + "\n");
        return lines;
    }
    std::snprintf(text, sizeof(text), "[ab] phase %c %llu settled at present %llu (t %.3f s) arm %llu\n", step.b ? 'B' : 'A', static_cast<unsigned long long>(step.phase), static_cast<unsigned long long>(present), seconds, static_cast<unsigned long long>(step.arm));
    lines.emplace_back(text);
    return lines;
}

}
