#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/ShapeReplay.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawPlans.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "ShaderDiskCache.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace AgcDriver::DriverDetail::ShapeReplay {

namespace {

using DescriptorRole = ShaderRecompiler::DescriptorRole;

std::uint64_t mix(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    hash *= 0x100000001b3ull;
    return hash;
}

bool sameWords(const std::array<std::uint32_t, 8>& dwords, std::span<const std::uint32_t> group) {
    return std::equal(group.begin(), group.end(), dwords.begin());
}

struct Plan {
    std::weak_ptr<const ShaderRecompiler::IrResourcePlan> plan;
    std::weak_ptr<ShaderRecompiler::SourceEntry> source;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
    PatchMap map;
    std::atomic<bool> checked{false};
    std::atomic<bool> disabled{false};
};

struct Store {
    std::shared_mutex mutex;
    std::unordered_map<std::uint64_t, std::shared_ptr<Plan>> plans;
    std::unordered_map<std::uint64_t, std::shared_ptr<Plan>> old;
    std::uint64_t resets = 0;
};

Store& store() {
    static Store state;
    return state;
}

constexpr std::size_t MaxPlans = 8192;

struct AtomicCounters {
    std::atomic<std::uint64_t> captures{0}, eligible{0}, hits{0}, learned{0}, unlearnable{0}, refused{0}, probations{0}, probationMismatches{0}, verified{0}, verifyMismatches{0}, disabled{0}, replayNs{0}, materializeNs{0}, materializes{0}, absentBuffers{0}, keysCarried{0}, keyMismatches{0};
};

AtomicCounters& counters() {
    static AtomicCounters state;
    return state;
}

std::uint32_t verifyBudget() {
    static const std::uint32_t budget = [] {
        const char* text = std::getenv("APS5_VERIFY_SHAPE_REPLAY");
        return text != nullptr ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 10)) : 0u;
    }();
    return budget;
}

bool reporting() {
    static const bool enabled = Graphics::ShapeReplay::Enabled() || verifyBudget() != 0u || std::getenv("APS5_VERIFY_DRAW_PLANS") != nullptr || std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return enabled;
}

struct Window {
    std::mutex mutex;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::uint32_t verified = 0;
    Counters last{};
};

Window& window() {
    static Window state;
    return state;
}

bool takeVerify() {
    const auto budget = verifyBudget();
    if (budget == 0u) return false;
    auto& state = window();
    std::lock_guard lock(state.mutex);
    if (std::chrono::steady_clock::now() - state.start >= std::chrono::seconds(10)) {
        state.start = std::chrono::steady_clock::now();
        state.verified = 0;
    }
    if (state.verified >= budget) return false;
    ++state.verified;
    return true;
}

void report() {
    static std::atomic<std::int64_t> lastReport{0};
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = lastReport.load(std::memory_order_relaxed);
    if (last == 0) {
        lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10000 || !lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    const auto totals = Totals();
    Counters delta;
    std::size_t plans = 0;
    std::uint64_t resets = 0;
    {
        auto& state = window();
        std::lock_guard lock(state.mutex);
        const auto& before = state.last;
        delta = {totals.captures - before.captures, totals.eligible - before.eligible, totals.hits - before.hits, totals.learned - before.learned, totals.unlearnable - before.unlearnable, totals.refused - before.refused, totals.probations - before.probations, totals.probationMismatches - before.probationMismatches, totals.verified - before.verified, totals.verifyMismatches - before.verifyMismatches, totals.disabled - before.disabled, totals.replayNs - before.replayNs, totals.materializeNs - before.materializeNs, totals.materializes - before.materializes, totals.absentBuffers - before.absentBuffers, totals.keysCarried - before.keysCarried, totals.keyMismatches - before.keyMismatches};
        state.last = totals;
    }
    {
        auto& plansStore = store();
        std::shared_lock lock(plansStore.mutex);
        plans = plansStore.plans.size() + plansStore.old.size();
        resets = plansStore.resets;
    }
    const auto average = [](std::uint64_t ns, std::uint64_t count) { return count != 0 ? static_cast<double>(ns) / 1000.0 / static_cast<double>(count) : 0.0; };
    std::fprintf(stderr, "[shape-replay] D1 (10 s): %llu stage captures, %llu eligible, %llu replayed (%.1f %% of eligible; %llu carried their resource key words, %llu key mismatches), %llu learned, %llu unlearnable, %llu refused at patching, %llu with an absent V#; probations %llu (%llu mismatches), verified %llu (%llu mismatches, total %llu), %llu plans disabled; plans %zu (resets %llu); avg us: replay %.1f, materialize %.1f%s\n", static_cast<unsigned long long>(delta.captures), static_cast<unsigned long long>(delta.eligible), static_cast<unsigned long long>(delta.hits), delta.eligible != 0 ? 100.0 * static_cast<double>(delta.hits) / static_cast<double>(delta.eligible) : 0.0, static_cast<unsigned long long>(delta.keysCarried), static_cast<unsigned long long>(delta.keyMismatches), static_cast<unsigned long long>(delta.learned), static_cast<unsigned long long>(delta.unlearnable), static_cast<unsigned long long>(delta.refused), static_cast<unsigned long long>(delta.absentBuffers), static_cast<unsigned long long>(delta.probations), static_cast<unsigned long long>(delta.probationMismatches), static_cast<unsigned long long>(delta.verified), static_cast<unsigned long long>(delta.verifyMismatches), static_cast<unsigned long long>(totals.verifyMismatches), static_cast<unsigned long long>(delta.disabled), plans, static_cast<unsigned long long>(resets), average(delta.replayNs, delta.hits), average(delta.materializeNs, delta.materializes), Active() ? "" : " (replay off)");
    std::fprintf(stderr, "%s%s\n", Graphics::DrawPlans::ReportLine().c_str(), Graphics::DrawPlans::Active() ? "" : " (plans off)");
}

std::uint64_t elapsedNs(std::chrono::steady_clock::time_point since) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
}

}

std::optional<PatchMap> Learn(const ShaderRecompiler::RecompileResult& result, const ShaderRecompiler::ResourceSnapshot& snapshot, const ShaderRecompiler::ShaderVertexStageInfo* vertex) {
    PatchMap map;
    if (!result.imageTablePoison.empty() || !result.imageTableRanges.empty() || !result.imageTableReadRanges.empty()) return std::nullopt;
    const auto& words = result.shaderDataUserWords;
    const auto userWordsMatch = [&](std::span<const std::uint32_t> target) {
        if (target.size() < words.size()) return false;
        for (std::size_t i = 0; i < words.size(); ++i) {
            if (words[i] >= snapshot.userData.size() || target[i] != snapshot.userData[words[i]]) return false;
        }
        return true;
    };
    for (std::uint32_t b = 0; b < result.bindings.size(); ++b) {
        const auto& binding = result.bindings[b];
        switch (binding.role) {
        case DescriptorRole::FlattenedSrt:
            if (binding.guestDescriptor != snapshot.flattenedSrt) return std::nullopt;
            map.flatBindings.push_back(b);
            break;
        case DescriptorRole::GuestBuffers: {
            if (binding.guestDescriptor.size() % 4u != 0u) return std::nullopt;
            for (std::uint32_t element = 0; element * 4u < binding.guestDescriptor.size(); ++element) {
                const auto group = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(element * 4u, 4u);
                PatchMap::BufferElement entry{b, element, {}};
                for (std::uint32_t r = 0; r < snapshot.buffers.size(); ++r) {
                    if (snapshot.buffers[r].dwordCount == 4u && sameWords(snapshot.buffers[r].dwords, group)) entry.candidates.push_back(r);
                }
                if (entry.candidates.empty()) return std::nullopt;
                map.buffers.push_back(std::move(entry));
            }
            break;
        }
        case DescriptorRole::ShaderData:
            if (!userWordsMatch(binding.guestDescriptor)) return std::nullopt;
            map.shaderDataBinding = static_cast<std::int32_t>(b);
            break;
        case DescriptorRole::ImageTable:
        case DescriptorRole::SamplerTable:
        case DescriptorRole::ImageTableMap:
        case DescriptorRole::ImageStorageTable:
            return std::nullopt;
        default:
            break;
        }
    }
    if (map.shaderDataBinding < 0 && !words.empty()) {
        if (result.pushConstants.size() < words.size() * sizeof(std::uint32_t)) return std::nullopt;
        std::vector<std::uint32_t> pushed(words.size());
        std::memcpy(pushed.data(), result.pushConstants.data(), pushed.size() * sizeof(std::uint32_t));
        if (!userWordsMatch(pushed)) return std::nullopt;
        map.pushUserWords = true;
    }
    if (!result.vertexAttributes.empty()) {
        if (vertex == nullptr) return std::nullopt;
        for (const auto& attribute : result.vertexAttributes) {
            if (attribute.location >= vertex->resourcesNum || attribute.location >= vertex->resources.size() || attribute.resource.fields != vertex->resources[attribute.location].fields) return std::nullopt;
        }
        map.vertexAttributes = true;
    }
    return map;
}

std::shared_ptr<ShaderRecompiler::RecompileResult> Apply(const ShaderRecompiler::RecompileResult& plan, const PatchMap& map, const ShaderRecompiler::ResourceSnapshot& snapshot, const ShaderRecompiler::ShaderVertexStageInfo* vertex) {
    const auto& words = plan.shaderDataUserWords;
    for (const auto word : words) {
        if ((map.pushUserWords || map.shaderDataBinding >= 0) && word >= snapshot.userData.size()) return nullptr;
    }
    for (const auto& element : map.buffers) {
        const auto first = element.candidates.front();
        if (first >= snapshot.buffers.size() || snapshot.buffers[first].dwordCount != 4u) return nullptr;
        for (const auto other : element.candidates) {
            if (other >= snapshot.buffers.size() || snapshot.buffers[other].dwordCount != 4u || !std::equal(snapshot.buffers[other].dwords.begin(), snapshot.buffers[other].dwords.begin() + 4, snapshot.buffers[first].dwords.begin())) return nullptr;
        }
    }
    if (map.vertexAttributes) {
        if (vertex == nullptr) return nullptr;
        for (const auto& attribute : plan.vertexAttributes) {
            if (attribute.location >= vertex->resourcesNum || attribute.location >= vertex->resources.size()) return nullptr;
        }
    }
    auto result = std::make_shared<ShaderRecompiler::RecompileResult>(plan);
    for (const auto b : map.flatBindings) result->bindings[b].guestDescriptor = snapshot.flattenedSrt;
    for (const auto& element : map.buffers) {
        const auto& source = snapshot.buffers[element.candidates.front()].dwords;
        std::copy_n(source.begin(), 4u, result->bindings[element.binding].guestDescriptor.begin() + element.element * 4u);
    }
    if (map.shaderDataBinding >= 0) {
        auto& target = result->bindings[static_cast<std::size_t>(map.shaderDataBinding)].guestDescriptor;
        for (std::size_t i = 0; i < words.size(); ++i) target[i] = snapshot.userData[words[i]];
    }
    if (map.pushUserWords) {
        for (std::size_t i = 0; i < words.size(); ++i) std::memcpy(result->pushConstants.data() + i * sizeof(std::uint32_t), &snapshot.userData[words[i]], sizeof(std::uint32_t));
    }
    if (map.vertexAttributes) {
        for (auto& attribute : result->vertexAttributes) attribute.resource = vertex->resources[attribute.location];
    }
    return result;
}

std::string Difference(const ShaderRecompiler::RecompileResult& replayed, const ShaderRecompiler::RecompileResult& full) {
    std::vector<std::byte> left, right;
    ShaderRecompiler::ShaderDiskCache::EncodeResult(replayed, left);
    ShaderRecompiler::ShaderDiskCache::EncodeResult(full, right);
    if (left == right && replayed.cacheHit == full.cacheHit) return {};
    char text[192];
    if (replayed.variantId != full.variantId || replayed.specializationId != full.specializationId) {
        std::snprintf(text, sizeof(text), "variant 0x%llx/0x%llx vs 0x%llx/0x%llx", static_cast<unsigned long long>(replayed.variantId), static_cast<unsigned long long>(replayed.specializationId), static_cast<unsigned long long>(full.variantId), static_cast<unsigned long long>(full.specializationId));
        return text;
    }
    if (replayed.pushConstants != full.pushConstants) return "push constants";
    if (replayed.bindings.size() != full.bindings.size()) return "binding count";
    for (std::size_t b = 0; b < full.bindings.size(); ++b) {
        if (replayed.bindings[b].guestDescriptor == full.bindings[b].guestDescriptor) continue;
        std::size_t word = 0;
        while (word < full.bindings[b].guestDescriptor.size() && word < replayed.bindings[b].guestDescriptor.size() && replayed.bindings[b].guestDescriptor[word] == full.bindings[b].guestDescriptor[word]) ++word;
        std::snprintf(text, sizeof(text), "binding %zu (role %u) guest descriptor word %zu", b, static_cast<unsigned>(full.bindings[b].role), word);
        return text;
    }
    if (replayed.vertexAttributes.size() != full.vertexAttributes.size()) return "vertex attribute count";
    for (std::size_t a = 0; a < full.vertexAttributes.size(); ++a) {
        if (replayed.vertexAttributes[a].resource.fields != full.vertexAttributes[a].resource.fields || replayed.vertexAttributes[a].fetchIndex != full.vertexAttributes[a].fetchIndex || replayed.vertexAttributes[a].formatComponents != full.vertexAttributes[a].formatComponents) return "vertex attribute " + std::to_string(a);
    }
    if (replayed.cacheHit != full.cacheHit) return "cache hit flag";
    return "encoding";
}

bool KeyWordsCarry(const ShaderRecompiler::RecompileResult& plan, const ShaderRecompiler::RecompileResult& replayed, const PatchMap& map, bool movableBuffers) {
    if (!movableBuffers) return false;
    const auto empty = [](const std::uint32_t* words) { return words[2] == 0u || (words[0] == 0u && (words[1] & 0xffffu) == 0u); };
    for (const auto& element : map.buffers) {
        const auto& binding = replayed.bindings[element.binding];
        const auto* before = plan.bindings[element.binding].guestDescriptor.data() + element.element * 4u;
        const auto* after = binding.guestDescriptor.data() + element.element * 4u;
        const bool whole = binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u;
        const bool written = element.element >= binding.bufferWritten.size() || binding.bufferWritten[element.element];
        if (whole || written || empty(before) || empty(after)) {
            if (!std::equal(before, before + 4, after)) return false;
        } else if ((before[1] & 0xffff0000u) != (after[1] & 0xffff0000u) || before[3] != after[3]) {
            return false;
        }
    }
    return true;
}

std::uint64_t ReplayKey(const Graphics::ShapeReplay::StageRecord& record) {
    using Graphics::ShapeReplay::Diff;
    std::uint64_t key = 0xcbf29ce484222325ull;
    for (const auto diff : {Diff::Texture, Diff::Sampler, Diff::UniformFill, Diff::Vertex, Diff::Other}) key = mix(key, record.hashes[static_cast<std::size_t>(diff)]);
    return mix(key, record.bufferPlan);
}

bool Active() {
    static const bool memoOff = std::getenv("APS5_NO_RESULT_MEMO") != nullptr;
    return memoOff && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoShapeReplay);
}

std::uint64_t PlanKey(std::uint64_t shape, std::span<const Graphics::CompiledShader> stages) {
    if (shape == 0) return 0;
    auto key = mix(mix(0xd2a1a115ull, shape), stages.size());
    for (const auto& stage : stages) {
        if (stage.program == nullptr) return 0;
        const auto& program = *stage.program;
        key = mix(mix(mix(key, static_cast<std::uint64_t>(stage.stage)), stage.pushConstantOffset), program.PipelineVariantId());
        key = mix(mix(mix(key, program.pushConstants.size()), program.bindlessHeap ? 1u : 0u), program.bindings.size());
    }
    return key != 0 ? key : 1u;
}

std::uint64_t& DrawShape() {
    thread_local std::uint64_t shape = 0;
    return shape;
}

Counters Totals() {
    auto& state = counters();
    const auto load = [](const std::atomic<std::uint64_t>& value) { return value.load(std::memory_order_relaxed); };
    return {load(state.captures), load(state.eligible), load(state.hits), load(state.learned), load(state.unlearnable), load(state.refused), load(state.probations), load(state.probationMismatches), load(state.verified), load(state.verifyMismatches), load(state.disabled), load(state.replayNs), load(state.materializeNs), load(state.materializes), load(state.absentBuffers), load(state.keysCarried), load(state.keyMismatches)};
}

std::shared_ptr<const ShaderRecompiler::RecompileResult> Materialize(std::uint64_t shape, std::size_t stage, ShaderRecompiler::ShaderStage kind, std::uint32_t pushOffset, std::uint64_t program, const ShaderRecompiler::ResourceCapture& capture, const Graphics::ShapeReplay::StageRecord* record, const ShaderRecompiler::ShaderVertexStageInfo* vertex, const FullMaterialize& full) {
    auto& count = counters();
    const bool timed = reporting();
    count.captures.fetch_add(1, std::memory_order_relaxed);
    const auto materialize = [&] {
        const auto started = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        auto result = full();
        if (timed) {
            count.materializeNs.fetch_add(elapsedNs(started), std::memory_order_relaxed);
            count.materializes.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    };
    const auto finish = [&](std::shared_ptr<const ShaderRecompiler::RecompileResult> result) {
        if (timed) report();
        return result;
    };
    if (shape == 0 || record == nullptr || !record->eligible || capture.plan == nullptr || capture.source == nullptr || !Active()) return finish(materialize());
    count.eligible.fetch_add(1, std::memory_order_relaxed);
    if (!record->buffersPresent) {
        count.absentBuffers.fetch_add(1, std::memory_order_relaxed);
        return finish(materialize());
    }
    auto key = mix(mix(mix(mix(mix(0x5ba9e5ull, shape), stage), pushOffset), program), ReplayKey(*record));
    key = mix(key, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(capture.plan.get())));
    auto& plans = store();
    std::shared_ptr<Plan> plan;
    {
        std::shared_lock lock(plans.mutex);
        if (const auto found = plans.plans.find(key); found != plans.plans.end()) plan = found->second;
    }
    if (plan == nullptr) {
        std::unique_lock lock(plans.mutex);
        if (const auto found = plans.plans.find(key); found != plans.plans.end()) {
            plan = found->second;
        } else if (const auto aged = plans.old.find(key); aged != plans.old.end()) {
            plan = std::move(aged->second);
            plans.old.erase(aged);
            if (plans.plans.size() >= MaxPlans) {
                plans.old = std::move(plans.plans);
                plans.plans.clear();
                ++plans.resets;
            }
            plans.plans[key] = plan;
        }
    }
    if (plan != nullptr && (plan->plan.lock() != capture.plan || plan->source.lock() != capture.source)) plan = nullptr;
    if (plan != nullptr && !plan->disabled.load(std::memory_order_relaxed)) {
        const auto started = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        auto replayed = Apply(*plan->result, plan->map, capture.snapshot, vertex);
        if (replayed == nullptr) {
            count.refused.fetch_add(1, std::memory_order_relaxed);
            return finish(materialize());
        }
        static const bool movableBuffers = std::getenv("APS5_NO_MOVED_BUFFER_TEMPLATES") == nullptr;
        const bool carry = Graphics::DrawPlans::Active() && KeyWordsCarry(*plan->result, *replayed, plan->map, movableBuffers);
        auto carried = carry ? plan->result->drawKeyMemo.words.load(std::memory_order_acquire) : nullptr;
        if (carried != nullptr) {
            replayed->drawKeyMemo.words.store(carried, std::memory_order_release);
            count.keysCarried.fetch_add(1, std::memory_order_relaxed);
        }
        const bool probation = !plan->checked.exchange(true, std::memory_order_relaxed);
        const bool keyCheck = carried != nullptr && Graphics::DrawPlans::TakeVerify();
        if (probation || keyCheck || takeVerify()) {
            auto result = materialize();
            auto difference = Difference(*replayed, *result);
            if (difference.empty() && carried != nullptr) {
                const auto fresh = Graphics::ShaderResources::ContentKey(Graphics::CompiledShader{kind, result.get(), pushOffset}, true, movableBuffers);
                if (fresh != *carried) {
                    difference = "carried resource key words";
                    count.keyMismatches.fetch_add(1, std::memory_order_relaxed);
                }
            }
            (probation ? count.probations : count.verified).fetch_add(1, std::memory_order_relaxed);
            if (!difference.empty()) {
                (probation ? count.probationMismatches : count.verifyMismatches).fetch_add(1, std::memory_order_relaxed);
                if (!plan->disabled.exchange(true, std::memory_order_relaxed)) count.disabled.fetch_add(1, std::memory_order_relaxed);
                static std::atomic<std::uint32_t> printed{0};
                if (!probation || printed.fetch_add(1, std::memory_order_relaxed) < 20u) std::fprintf(stderr, "[shape-replay] %s: stage %zu (program 0x%llx, push offset %u) replayed unlike its full materialization in its %s; the plan is disabled\n", probation ? "probation" : "verify", stage, static_cast<unsigned long long>(program), pushOffset, difference.c_str());
            }
            return finish(result);
        }
        if (timed) count.replayNs.fetch_add(elapsedNs(started), std::memory_order_relaxed);
        count.hits.fetch_add(1, std::memory_order_relaxed);
        return finish(replayed);
    }
    auto result = materialize();
    if (plan == nullptr) {
        auto map = Learn(*result, capture.snapshot, vertex);
        auto fresh = std::make_shared<Plan>();
        fresh->plan = capture.plan;
        fresh->source = capture.source;
        fresh->result = result;
        if (map) {
            fresh->map = std::move(*map);
            count.learned.fetch_add(1, std::memory_order_relaxed);
        } else {
            fresh->disabled.store(true, std::memory_order_relaxed);
            fresh->result = nullptr;
            count.unlearnable.fetch_add(1, std::memory_order_relaxed);
        }
        std::unique_lock lock(plans.mutex);
        if (plans.plans.size() >= MaxPlans && !plans.plans.contains(key)) {
            plans.old = std::move(plans.plans);
            plans.plans.clear();
            ++plans.resets;
        }
        plans.plans[key] = std::move(fresh);
    }
    return finish(result);
}

}
