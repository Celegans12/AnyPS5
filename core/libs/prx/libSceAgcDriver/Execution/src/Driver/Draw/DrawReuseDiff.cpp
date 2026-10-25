#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawReuseDiff.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// APS5_DRAW_REUSE_DIFF=1 (profile-only, local; design.md 1.2). For each freshly captured stage the
// capture's runs are re-expressed relative to the user word each word was reached from (the SRT
// walk's read origins, SrtUserDataUse::reads; a run the walk did not read, such as the vertex fetch
// table the decoder read, is anchored to the nearest user-word pointer below it within 1 MiB), so
// two captures of the same (shape key, stage) can be compared position by position although their
// pointers moved. The stage is classified against a per-thread ring of the last RingPlans plans of
// the same (kind, shape key, stage), best class first (R0, R1, R2, R3, R4, R6, R5, plan differs),
// and inserted. Nothing here reads guest memory or changes a cache.

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::size_t RingPlans = 256;
constexpr std::size_t ThreadBudgetBytes = std::size_t{256} << 20u;
constexpr std::uint32_t ProximityBytes = 1u << 20u;

enum Class : std::size_t { R0, R1, R2, R3, R4, R5, R6, R7, Differs, ClassCount };
// Preference when several ring plans match: the most reusable class wins.
constexpr std::array<std::uint8_t, ClassCount> Rank{0, 1, 2, 3, 4, 6, 5, 8, 7};

struct Segment {
    std::uint64_t address = 0;
    std::uint32_t first = 0;
    std::uint32_t count = 0;
    std::int32_t anchor = -1;
    std::uint32_t rel = 0;
};

struct Plan {
    std::uint64_t frame = 0;
    std::vector<std::uint32_t> words;
    std::vector<Segment> segments;
    std::vector<std::uint32_t> data;
    std::vector<std::uint32_t> user;
    std::uint64_t valueMask = 0;
    std::uint64_t shapeHash = 0, addressHash = 0, wordsHash = 0;
    ShaderRecompiler::PipelineIdentity identity;
    std::shared_ptr<const ShaderRecompiler::ShaderVertexStageInfo> vertex;

    [[nodiscard]] std::size_t Bytes() const {
        return sizeof(Plan) + words.capacity() * sizeof(std::uint32_t) + segments.capacity() * sizeof(Segment) + data.capacity() * sizeof(std::uint32_t) + user.capacity() * sizeof(std::uint32_t) + (vertex != nullptr ? sizeof(ShaderRecompiler::ShaderVertexStageInfo) : 0u);
    }
    [[nodiscard]] bool IsData(std::uint32_t position) const { return std::binary_search(data.begin(), data.end(), position); }
};

struct Ring {
    std::vector<Plan> plans;
    std::size_t next = 0;
    std::size_t bytes = 0;
    std::list<std::uint64_t>::iterator order;
};

struct KindCounters {
    std::array<std::uint64_t, ClassCount> classes{};
    std::array<std::array<std::uint64_t, 3>, ClassCount> ages{};
    std::array<std::uint64_t, ClassCount> previousFrameMatch{};
    std::uint64_t r2Unmoved = 0, r3Unmoved = 0;
    std::array<std::uint64_t, 3> r5{};
    std::uint64_t moduleSame = 0, moduleCompared = 0, vertexSame = 0, vertexCompared = 0;
    std::uint64_t stages = 0, hits = 0, reused = 0, unkeyed = 0, compared = 0;
    double us = 0;
    std::unordered_map<std::uint32_t, std::uint64_t> moving;
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};

struct ThreadState {
    std::array<KindCounters, 2> counters;
    std::unordered_map<std::uint64_t, Ring> rings;
    std::list<std::uint64_t> lru;
    std::size_t bytes = 0;
    std::size_t plans = 0;
    std::uint32_t index = 0;
};

ThreadState& State() {
    static std::atomic<std::uint32_t> threads{0};
    thread_local ThreadState state = [] {
        ThreadState created;
        created.index = threads.fetch_add(1, std::memory_order_relaxed);
        return created;
    }();
    return state;
}

std::uint64_t Mix(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    return hash;
}

std::int32_t ProximityAnchor(std::uint64_t address, const std::vector<std::uint32_t>& user) {
    std::int32_t best = -1;
    std::uint32_t bestOffset = ProximityBytes;
    const auto low = static_cast<std::uint32_t>(address);
    const auto high = static_cast<std::uint32_t>(address >> 32u) & 0xffffu;
    for (std::size_t u = 0; u < user.size() && u < 64u; ++u) {
        const auto base = user[u] & ~3u;
        if (base < 0x10000u) continue;
        if (u + 1 < user.size() && (user[u + 1] & 0xffffu) != high) continue;
        const auto offset = low - base;
        if (offset < bestOffset) {
            bestOffset = offset;
            best = static_cast<std::int32_t>(u);
        }
    }
    return best;
}

Plan BuildPlan(const ReuseStageInput& input) {
    Plan plan;
    const auto& variant = *input.variant;
    plan.frame = GuestMemory::PresentIndex();
    plan.words = variant.words;
    plan.user.assign(input.userData.begin(), input.userData.end());
    plan.data = variant.dataPositions;
    std::sort(plan.data.begin(), plan.data.end());
    plan.data.erase(std::unique(plan.data.begin(), plan.data.end()), plan.data.end());
    plan.vertex = variant.vertexInfo;
    if (variant.compiled != nullptr) plan.identity = variant.compiled->Identity();
    const auto asBase = input.use != nullptr ? input.use->asBase : 0;
    const auto asValue = input.use != nullptr ? input.use->asValue : 0;
    const auto registers = variant.compiled != nullptr ? variant.compiled->userDataRegisterReads : 0;
    plan.valueMask = (asValue | registers) & ~(asBase & ~asValue);

    thread_local std::vector<std::pair<std::uint64_t, std::int32_t>> reads;
    reads.clear();
    if (input.use != nullptr) reads.assign(input.use->reads.begin(), input.use->reads.end());
    std::sort(reads.begin(), reads.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first < b.first : (a.second >= 0) != (b.second >= 0) ? a.second >= 0 : a.second < b.second; });
    reads.erase(std::unique(reads.begin(), reads.end(), [](const auto& a, const auto& b) { return a.first == b.first; }), reads.end());
    const auto origin = [&](std::uint64_t address) -> std::int32_t {
        const auto found = std::lower_bound(reads.begin(), reads.end(), address, [](const auto& read, std::uint64_t value) { return read.first < value; });
        if (found == reads.end() || found->first != address) return INT_MIN;
        return found->second >= 0 && static_cast<std::size_t>(found->second) < plan.user.size() ? found->second : -1;
    };
    thread_local std::vector<std::int32_t> anchors;
    std::uint32_t first = 0;
    for (const auto& [begin, end] : variant.runs) {
        const auto count = static_cast<std::uint32_t>((end - begin) / sizeof(std::uint32_t));
        anchors.assign(count, INT_MIN);
        std::int32_t current = INT_MIN;
        std::int32_t leading = INT_MIN;
        for (std::uint32_t k = 0; k < count; ++k) {
            const auto found = origin(begin + static_cast<std::uint64_t>(k) * sizeof(std::uint32_t));
            if (found != INT_MIN) {
                current = found;
                if (leading == INT_MIN) leading = found;
            }
            anchors[k] = current;
        }
        if (leading == INT_MIN) leading = ProximityAnchor(begin, plan.user);
        for (auto& anchor : anchors) {
            if (anchor != INT_MIN) break;
            anchor = leading;
        }
        for (std::uint32_t k = 0; k < count;) {
            auto next = k + 1;
            while (next < count && anchors[next] == anchors[k]) ++next;
            Segment segment;
            segment.address = begin + static_cast<std::uint64_t>(k) * sizeof(std::uint32_t);
            segment.first = first + k;
            segment.count = next - k;
            segment.anchor = anchors[k];
            if (segment.anchor >= 0) segment.rel = static_cast<std::uint32_t>(segment.address) - (plan.user[static_cast<std::size_t>(segment.anchor)] & ~3u);
            plan.segments.push_back(segment);
            k = next;
        }
        first += count;
    }
    std::uint64_t shape = Mix(0x51ed270b27f0a1c3ull, plan.words.size());
    std::uint64_t addresses = 0x2545f4914f6cdd1dull;
    for (const auto& segment : plan.segments) {
        shape = Mix(shape, static_cast<std::uint64_t>(static_cast<std::uint32_t>(segment.anchor)) | static_cast<std::uint64_t>(segment.count) << 32u);
        shape = Mix(shape, segment.anchor >= 0 ? segment.rel : 0u);
        addresses = Mix(addresses, segment.address);
    }
    std::uint64_t words = 0x9e3779b97f4a7c15ull;
    for (const auto word : plan.words) words = Mix(words, word);
    plan.shapeHash = shape;
    plan.addressHash = addresses;
    plan.wordsHash = words;
    return plan;
}

// Word types of the new plan: V# dword0 (1), V# dword1 (2), other V# dwords (3), T#/S# (4), found
// by matching the compiled bindings' guest descriptors and the vertex decoder's 16-byte reads
// against the captured words.
enum WordType : std::uint8_t { Untyped = 0, BufferBase0 = 1, BufferBase1 = 2, BufferOther = 3, ImageOrSampler = 4 };

void TypeWords(const Plan& plan, const ReuseStageInput& input, std::vector<std::uint8_t>& types) {
    types.assign(plan.words.size(), Untyped);
    const auto markBuffer = [&](std::size_t p) {
        if (p + 4 > types.size()) return;
        types[p] = BufferBase0;
        types[p + 1] = BufferBase1;
        types[p + 2] = types[p + 3] = BufferOther;
    };
    const auto find = [&](std::span<const std::uint32_t> descriptor, auto&& mark) {
        if (descriptor.empty() || descriptor.size() > plan.words.size()) return;
        for (std::size_t p = 0; p + descriptor.size() <= plan.words.size(); ++p) {
            if (plan.words[p] != descriptor[0] || !std::equal(descriptor.begin(), descriptor.end(), plan.words.begin() + static_cast<std::ptrdiff_t>(p))) continue;
            mark(p);
        }
    };
    if (input.variant->compiled != nullptr) {
        using Role = ShaderRecompiler::DescriptorRole;
        for (const auto& binding : input.variant->compiled->bindings) {
            const std::span<const std::uint32_t> words(binding.guestDescriptor);
            if (binding.role == Role::GuestBuffers) {
                for (std::size_t d = 0; d + 4 <= words.size(); d += 4) find(words.subspan(d, 4), markBuffer);
            } else if (binding.role == Role::GuestImages || binding.role == Role::GuestSamplers) {
                const std::size_t size = binding.role == Role::GuestImages ? 8u : 4u;
                for (std::size_t d = 0; d + size <= words.size(); d += size) {
                    find(words.subspan(d, size), [&](std::size_t p) {
                        for (std::size_t k = 0; k < size && p + k < types.size(); ++k) types[p + k] = ImageOrSampler;
                    });
                }
            }
        }
    }
    for (const auto& read : input.decodeReads) {
        if (read.size != 16u || (read.address & 3u) != 0) continue;
        for (const auto& segment : plan.segments) {
            if (read.address < segment.address || read.address >= segment.address + static_cast<std::uint64_t>(segment.count) * 4u) continue;
            markBuffer(segment.first + static_cast<std::size_t>((read.address - segment.address) / 4u));
            break;
        }
    }
}

struct PairResult {
    Class cls = Differs;
    std::uint8_t r5 = 0;
    bool unmoved = false;
};

PairResult ClassifyPair(const Plan& now, const Plan& old, const ReuseStageInput& input, std::vector<std::uint8_t>& types, bool& typed) {
    PairResult result;
    if (now.shapeHash != old.shapeHash || now.words.size() != old.words.size() || now.segments.size() != old.segments.size()) return result;
    bool moved = false, nestedMoved = false;
    for (std::size_t s = 0; s < now.segments.size(); ++s) {
        const auto& a = now.segments[s];
        const auto& b = old.segments[s];
        if (a.anchor != b.anchor || a.count != b.count || a.first != b.first || (a.anchor >= 0 && a.rel != b.rel)) return result;
        if (a.address != b.address) {
            moved = true;
            if (a.anchor < 0) nestedMoved = true;
        }
    }
    thread_local std::vector<std::uint32_t> nonData;
    nonData.clear();
    std::uint64_t dataDiff = 0;
    if (now.wordsHash != old.wordsHash) {
        for (std::uint32_t p = 0; p < now.words.size(); ++p) {
            if (now.words[p] == old.words[p]) continue;
            if (now.IsData(p) || old.IsData(p)) ++dataDiff;
            else nonData.push_back(p);
        }
    }
    bool valueDiff = now.user.size() != old.user.size();
    const auto values = now.valueMask | old.valueMask;
    for (std::size_t u = 0; !valueDiff && u < now.user.size() && u < 64u; ++u) valueDiff = (values >> u & 1u) != 0 && now.user[u] != old.user[u];
    Class memory;
    if (nestedMoved) {
        bool explained = true;
        for (std::size_t s = 0; s < now.segments.size() && explained; ++s) {
            const auto& a = now.segments[s];
            const auto& b = old.segments[s];
            if (a.anchor >= 0 || a.address == b.address) continue;
            const auto delta = static_cast<std::uint32_t>(a.address) - static_cast<std::uint32_t>(b.address);
            explained = std::any_of(nonData.begin(), nonData.end(), [&](std::uint32_t p) { return now.words[p] - old.words[p] == delta; });
        }
        if (!explained) return result;
        memory = R4;
    } else if (nonData.empty()) {
        memory = dataDiff == 0 ? (moved ? R1 : R0) : R2;
        result.unmoved = memory == R2 && !moved;
    } else {
        if (!typed) {
            TypeWords(now, input, types);
            typed = true;
        }
        bool bases = true, image = false, bufferOther = false;
        for (const auto p : nonData) {
            const auto type = types[p];
            if (type == BufferBase0) continue;
            if (type == BufferBase1 && ((now.words[p] ^ old.words[p]) & 0xffff0000u) == 0) continue;
            bases = false;
            if (type == ImageOrSampler) image = true;
            else if (type == BufferBase1 || type == BufferOther) bufferOther = true;
        }
        if (bases) {
            memory = R3;
            result.unmoved = !moved;
        } else {
            memory = R5;
            result.r5 = image ? 0 : bufferOther ? 1 : 2;
        }
    }
    result.cls = valueDiff && memory != R5 ? R6 : memory;
    return result;
}

const char* KindName(ReuseKind kind) { return kind == ReuseKind::Draw ? "draw-reuse" : "dispatch-reuse"; }

void MaybeReport(ReuseKind kind) {
    auto& state = State();
    auto& c = state.counters[static_cast<std::size_t>(kind)];
    const auto now = std::chrono::steady_clock::now();
    if (now - c.last < std::chrono::seconds(10)) return;
    const auto n = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    std::string ages;
    for (std::size_t k = 0; k < ClassCount; ++k) {
        if (k == R7 || c.classes[k] == 0) continue;
        static constexpr std::array<const char*, ClassCount> names{"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "differs"};
        char item[128];
        std::snprintf(item, sizeof(item), " %s %llu/%llu/%llu (%llu)", names[k], n(c.ages[k][0]), n(c.ages[k][1]), n(c.ages[k][2]), n(c.previousFrameMatch[k]));
        ages += item;
    }
    std::vector<std::pair<std::uint32_t, std::uint64_t>> moving(c.moving.begin(), c.moving.end());
    std::sort(moving.begin(), moving.end(), [](const auto& a, const auto& b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
    std::string registers;
    for (std::size_t r = 0; r < moving.size() && r < 8; ++r) {
        char item[64];
        std::snprintf(item, sizeof(item), " s%u:0x%x x%llu", moving[r].first >> 16u, moving[r].first & 0xffffu, n(moving[r].second));
        registers += item;
    }
    const auto stages = c.stages;
    AgcDriver::ProfilePrint_nid_no_patch("[%s] (10 s, thread %u): %llu stages classified (%.1f us, %.1f ring plans compared each), not captured: %llu draw-cache hit stages, %llu data-entry stages, %llu unkeyed; R0 same %llu, R1 moved-only %llu, R2 moved+data %llu (%llu unmoved), R3 +buffer base %llu (%llu unmoved), R4 nested %llu, R5 descriptor %llu (T#/S# %llu, other V# fields %llu, untyped words %llu), R6 value word %llu, R7 new %llu, plan differs %llu; best match same/previous/older frame (previous-frame match as good):%s; R1-R4 compiled module identical %llu of %llu, vertex info equal %llu of %llu; top moving user registers (stage:reg):%s; ring %zu (shape, stage) keys, %zu plans, %.1f MiB\n", KindName(kind), state.index, n(stages), stages != 0 ? c.us / static_cast<double>(stages) : 0.0, stages != 0 ? static_cast<double>(c.compared) / static_cast<double>(stages) : 0.0, n(c.hits), n(c.reused), n(c.unkeyed), n(c.classes[R0]), n(c.classes[R1]), n(c.classes[R2]), n(c.r2Unmoved), n(c.classes[R3]), n(c.r3Unmoved), n(c.classes[R4]), n(c.classes[R5]), n(c.r5[0]), n(c.r5[1]), n(c.r5[2]), n(c.classes[R6]), n(c.classes[R7]), n(c.classes[Differs]), ages.empty() ? " none" : ages.c_str(), n(c.moduleSame), n(c.moduleCompared), n(c.vertexSame), n(c.vertexCompared), registers.empty() ? " none" : registers.c_str(), state.rings.size(), state.plans, static_cast<double>(state.bytes) / 1048576.0);
    c = KindCounters{};
    c.last = now;
}

}

void NoteReuseUncaptured(ReuseKind kind, std::uint64_t hitStages, std::uint64_t reusedStages, std::uint64_t unkeyedStages) {
    auto& c = State().counters[static_cast<std::size_t>(kind)];
    c.hits += hitStages;
    c.reused += reusedStages;
    c.unkeyed += unkeyedStages;
    MaybeReport(kind);
}

void NoteReuseStage(const ReuseStageInput& input) {
    if (input.variant == nullptr) return;
    const auto started = std::chrono::steady_clock::now();
    auto& state = State();
    auto& c = state.counters[static_cast<std::size_t>(input.kind)];
    auto plan = BuildPlan(input);
    const auto key = Mix(Mix(input.shape, static_cast<std::uint64_t>(input.stage) | static_cast<std::uint64_t>(input.stageType) << 8u), static_cast<std::uint64_t>(input.kind));
    auto [found, created] = state.rings.try_emplace(key);
    auto& ring = found->second;
    if (created) {
        state.lru.push_front(key);
        ring.order = state.lru.begin();
    } else {
        state.lru.splice(state.lru.begin(), state.lru, ring.order);
    }
    ++c.stages;
    if (ring.plans.empty()) {
        ++c.classes[R7];
    } else {
        thread_local std::vector<std::uint8_t> types;
        bool typed = false;
        PairResult best;
        const Plan* bestPlan = nullptr;
        std::uint8_t previousBest = UINT8_MAX;
        const auto size = ring.plans.size();
        for (std::size_t k = 0; k < size; ++k) {
            const auto& old = ring.plans[(ring.next + size - 1 - k) % size];
            if (old.frame + 1 < plan.frame && bestPlan != nullptr && Rank[best.cls] <= Rank[R1]) break;
            ++c.compared;
            const auto result = ClassifyPair(plan, old, input, types, typed);
            if (bestPlan == nullptr || Rank[result.cls] < Rank[best.cls]) {
                best = result;
                bestPlan = &old;
            }
            if (old.frame + 1 == plan.frame) previousBest = std::min(previousBest, Rank[result.cls]);
        }
        ++c.classes[best.cls];
        if (bestPlan != nullptr && best.cls != Differs) {
            ++c.ages[best.cls][bestPlan->frame == plan.frame ? 0 : bestPlan->frame + 1 == plan.frame ? 1 : 2];
            if (previousBest <= Rank[best.cls]) ++c.previousFrameMatch[best.cls];
            if (best.cls == R2 && best.unmoved) ++c.r2Unmoved;
            if (best.cls == R3 && best.unmoved) ++c.r3Unmoved;
            if (best.cls == R5) ++c.r5[best.r5];
            if (best.cls <= R4) {
                ++c.moduleCompared;
                if (plan.identity == bestPlan->identity) ++c.moduleSame;
                if (plan.vertex != nullptr && bestPlan->vertex != nullptr && input.sameVertex != nullptr) {
                    ++c.vertexCompared;
                    if (input.sameVertex(*plan.vertex, *bestPlan->vertex)) ++c.vertexSame;
                }
            }
            if (best.cls >= R1 && best.cls <= R6 && best.cls != R5) {
                for (std::size_t u = input.hiddenUserWords; u < plan.user.size() && u < bestPlan->user.size() && u < 64u; ++u) {
                    if (plan.user[u] == bestPlan->user[u]) continue;
                    const auto reg = input.userDataBase + static_cast<std::uint32_t>(u) - input.hiddenUserWords;
                    ++c.moving[(input.stageType << 16u) | (reg & 0xffffu)];
                }
            }
        }
    }
    const auto bytes = plan.Bytes();
    if (ring.plans.size() < RingPlans) {
        ring.plans.push_back(std::move(plan));
        ring.next = ring.plans.size() % RingPlans;
        ++state.plans;
    } else {
        const auto replaced = ring.plans[ring.next].Bytes();
        ring.bytes -= replaced;
        state.bytes -= replaced;
        ring.plans[ring.next] = std::move(plan);
        ring.next = (ring.next + 1) % RingPlans;
    }
    ring.bytes += bytes;
    state.bytes += bytes;
    while (state.bytes > ThreadBudgetBytes && state.lru.size() > 1) {
        const auto victim = state.rings.find(state.lru.back());
        state.bytes -= victim->second.bytes;
        state.plans -= victim->second.plans.size();
        state.rings.erase(victim);
        state.lru.pop_back();
    }
    c.us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
    MaybeReport(input.kind);
}

}
