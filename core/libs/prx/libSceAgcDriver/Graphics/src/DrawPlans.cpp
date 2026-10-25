#include "prx/libSceAgcDriver/Graphics/include/DrawPlans.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace AgcDriver::Graphics::DrawPlans {

namespace {

constexpr std::size_t MaxPlans = 16384;

struct Store {
    std::shared_mutex mutex;
    std::unordered_map<std::uint64_t, std::shared_ptr<const PipelinePlan>> plans;
    std::unordered_map<std::uint64_t, std::shared_ptr<const StorageSurfacePlan>> targets;
    std::uint64_t resets = 0;
};

Store& store() {
    static Store state;
    return state;
}

struct Atomics {
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Outcome::Count)> outcomes{};
    std::atomic<std::uint64_t> verified{0};
};

Atomics& atomics() {
    static Atomics state;
    return state;
}

std::uint32_t verifyBudget() {
    static const std::uint32_t budget = [] {
        const char* text = std::getenv("APS5_VERIFY_DRAW_PLANS");
        return text != nullptr ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 10)) : 0u;
    }();
    return budget;
}

constexpr std::array<const char*, static_cast<std::size_t>(Outcome::Count)> OutcomeNames{"hits", "no plan", "other template", "other layout", "pipeline gone", "store moved", "stored", "unstorable", "verify mismatches", "target hits", "target no plan", "target fallbacks", "target verify mismatches"};

std::uint64_t mix(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    hash *= 0x100000001b3ull;
    return hash;
}

}

bool Active() {
    return !RuntimeSwitches::Set(RuntimeSwitch::NoDrawPlans);
}

bool TakeVerify() {
    const auto budget = verifyBudget();
    if (budget == 0u) return false;
    static std::mutex mutex;
    static auto start = std::chrono::steady_clock::now();
    static std::uint32_t taken = 0;
    std::lock_guard lock(mutex);
    if (std::chrono::steady_clock::now() - start >= std::chrono::seconds(10)) {
        start = std::chrono::steady_clock::now();
        taken = 0;
    }
    if (taken >= budget) return false;
    ++taken;
    return true;
}

std::uint64_t& CurrentKey() {
    thread_local std::uint64_t key = 0;
    return key;
}

std::shared_ptr<const PipelinePlan> FindPipeline(std::uint64_t key) {
    auto& state = store();
    std::shared_lock lock(state.mutex);
    const auto found = state.plans.find(key);
    return found != state.plans.end() ? found->second : nullptr;
}

void StorePipeline(std::uint64_t key, std::shared_ptr<const PipelinePlan> plan) {
    auto& state = store();
    std::unique_lock lock(state.mutex);
    if (state.plans.size() >= MaxPlans && !state.plans.contains(key)) {
        state.plans.clear();
        ++state.resets;
    }
    state.plans[key] = std::move(plan);
}

void ForgetPipeline(std::uint64_t key) {
    auto& state = store();
    std::unique_lock lock(state.mutex);
    state.plans.erase(key);
}

bool PlannedTargets() {
    return Active() && !RuntimeSwitches::Set(RuntimeSwitch::NoPlannedTargets);
}

std::uint64_t TargetKey(std::uint64_t planKey, std::size_t index) {
    return mix(mix(0x7a29e7ull, planKey), index);
}

std::shared_ptr<const StorageSurfacePlan> FindTarget(std::uint64_t key) {
    auto& state = store();
    std::shared_lock lock(state.mutex);
    const auto found = state.targets.find(key);
    return found != state.targets.end() ? found->second : nullptr;
}

void StoreTarget(std::uint64_t key, std::shared_ptr<const StorageSurfacePlan> plan) {
    auto& state = store();
    std::unique_lock lock(state.mutex);
    if (state.targets.size() >= MaxPlans && !state.targets.contains(key)) state.targets.clear();
    state.targets[key] = std::move(plan);
}

void ForgetTarget(std::uint64_t key) {
    auto& state = store();
    std::unique_lock lock(state.mutex);
    state.targets.erase(key);
}

bool SameVertexInput(const VertexInputLayout& a, const VertexInputLayout& b) {
    if (a.bindings.size() != b.bindings.size() || a.attributes.size() != b.attributes.size()) return false;
    for (std::size_t i = 0; i < a.bindings.size(); ++i) {
        if (a.bindings[i].binding != b.bindings[i].binding || a.bindings[i].stride != b.bindings[i].stride || a.bindings[i].inputRate != b.bindings[i].inputRate) return false;
    }
    for (std::size_t i = 0; i < a.attributes.size(); ++i) {
        if (a.attributes[i].location != b.attributes[i].location || a.attributes[i].binding != b.attributes[i].binding || a.attributes[i].format != b.attributes[i].format || a.attributes[i].offset != b.attributes[i].offset) return false;
    }
    return true;
}

void Count(Outcome outcome) {
    atomics().outcomes[static_cast<std::size_t>(outcome)].fetch_add(1, std::memory_order_relaxed);
}

void NoteVerified() {
    atomics().verified.fetch_add(1, std::memory_order_relaxed);
}

Totals Counters() {
    Totals totals;
    for (std::size_t i = 0; i < totals.outcomes.size(); ++i) totals.outcomes[i] = atomics().outcomes[i].load(std::memory_order_relaxed);
    totals.verified = atomics().verified.load(std::memory_order_relaxed);
    return totals;
}

std::string ReportLine() {
    static Totals last;
    const auto now = Counters();
    std::string line = "[draw-plans] (10 s):";
    char text[96];
    for (std::size_t i = 0; i < now.outcomes.size(); ++i) {
        std::snprintf(text, sizeof(text), "%s %s %llu", i == 0 ? "" : ",", OutcomeNames[i], static_cast<unsigned long long>(now.outcomes[i] - last.outcomes[i]));
        line += text;
    }
    std::size_t plans = 0;
    std::uint64_t resets = 0;
    {
        auto& state = store();
        std::shared_lock lock(state.mutex);
        plans = state.plans.size();
        resets = state.resets;
    }
    std::snprintf(text, sizeof(text), "; verified %llu; plans %zu (resets %llu)", static_cast<unsigned long long>(now.verified - last.verified), plans, static_cast<unsigned long long>(resets));
    line += text;
    last = now;
    return line;
}

}
