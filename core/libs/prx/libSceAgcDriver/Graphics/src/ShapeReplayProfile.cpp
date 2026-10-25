#include "prx/libSceAgcDriver/Graphics/include/ShapeReplayProfile.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>

namespace AgcDriver::Graphics::ShapeReplay {

namespace {

constexpr std::uint64_t Seed = 0xcbf29ce484222325ull;

std::uint64_t mix(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    hash *= 0x100000001b3ull;
    return hash;
}

std::uint64_t classSeed(std::size_t index) {
    return mix(Seed, 0x5eed0000u + index);
}

constexpr std::uint8_t StructuralReach = ReachImage | ReachSampler | ReachBufferShape | ReachCondition | ReachUniform | ReachOtherSource;

std::size_t classIndex(Diff diff) {
    return static_cast<std::size_t>(diff);
}

std::uint64_t stageKey(std::uint64_t shape, std::size_t stage) {
    return mix(mix(Seed, shape), 0x57a9e000u + stage);
}

std::uint64_t drawKey(std::uint64_t shape) {
    return mix(mix(Seed, shape), 0xd7a30000u);
}

void hashDescriptor(std::uint64_t& hash, const ShaderRecompiler::DescriptorValue& value) {
    hash = mix(hash, value.dwordCount);
    hash = mix(hash, value.availableDwordMask);
    for (std::uint32_t dword = 0; dword < value.dwordCount && dword < value.dwords.size(); ++dword) hash = mix(hash, value.dwords[dword]);
}

}

bool EnabledFromEnvironment() {
    const char* text = std::getenv("APS5_PROFILE_SHAPE_REPLAY");
    return text != nullptr && *text != '\0' && std::strcmp(text, "0") != 0;
}

SlotClasses ComputeSlotClasses(const ShaderRecompiler::IrResourcePlan& plan) {
    using IrOpcode = ShaderRecompiler::IrOpcode;
    using IrType = ShaderRecompiler::IrType;
    using IrValue = ShaderRecompiler::IrValue;
    SlotClasses result;
    const auto slots = plan.srtReads.size();
    result.slots.assign(slots, SlotClass::Structural);
    result.reach.assign(slots, 0u);
    result.eligible = plan.srtPlanComplete && !plan.requiresSpecializationMemory
        && std::none_of(plan.descriptorSources.begin(), plan.descriptorSources.end(), [](const ShaderRecompiler::DescriptorSource& source) { return source.tableColumn.has_value(); })
        && !(plan.uniformFill.fill.kind != ShaderRecompiler::UniformFillKind::None && plan.uniformFill.fill.words > plan.uniformFill.values.size());
    if (!result.eligible) {
        result.counts[static_cast<std::size_t>(SlotClass::Structural)] = static_cast<std::uint32_t>(slots);
        return result;
    }
    std::unordered_map<const IrValue*, std::uint32_t> readSlots;
    for (std::uint32_t slot = 0; slot < slots; ++slot) {
        if (plan.srtReads[slot].value != nullptr) readSlots.emplace(plan.srtReads[slot].value->Resolve(), slot);
    }
    std::uint64_t userReach[8] = {};
    std::unordered_set<const IrValue*> visited;
    std::vector<IrValue*> pending;
    const auto mark = [&](IrValue* root, std::uint8_t bit, bool stopAtSlots) {
        if (root == nullptr) return;
        visited.clear();
        pending.clear();
        pending.push_back(root);
        const auto bitIndex = static_cast<std::size_t>(std::countr_zero(static_cast<unsigned>(bit)));
        while (!pending.empty()) {
            IrValue* value = pending.back()->Resolve();
            pending.pop_back();
            if (!visited.insert(value).second) continue;
            if (value->Opcode() == IrOpcode::ReadConst && value->ArgumentCount() >= 2 && value->Argument(1) != nullptr) {
                const IrValue* slot = value->Argument(1)->Resolve();
                if (slot->HasImmediate() && slot->Type() == IrType::U32 && slot->ImmediateU32() < slots) {
                    const auto read = slot->ImmediateU32();
                    result.reach[read] |= bit;
                    if (!stopAtSlots && plan.srtReads[read].value != nullptr) pending.push_back(plan.srtReads[read].value);
                    continue;
                }
            }
            if (const auto found = readSlots.find(value); found != readSlots.end()) {
                result.reach[found->second] |= bit;
                if (stopAtSlots) continue;
            }
            if (value->Opcode() == IrOpcode::GetUserData && value->ArgumentCount() >= 1 && value->Argument(0) != nullptr) {
                const auto reg = value->Argument(0)->Resolve()->Register().index;
                if (reg >= plan.userDataBase && reg - plan.userDataBase < 64u) userReach[bitIndex] |= std::uint64_t{1} << (reg - plan.userDataBase);
            }
            for (std::size_t i = 0; i < value->ArgumentCount(); ++i) {
                if (value->Argument(i) != nullptr) pending.push_back(value->Argument(i));
            }
        }
    };
    std::vector<std::uint8_t> sourceKinds(plan.descriptorSources.size(), 0u);
    for (const auto& buffer : plan.info.buffers) {
        if (buffer.source < sourceKinds.size()) sourceKinds[buffer.source] |= ReachBufferBase;
    }
    for (const auto& image : plan.info.images) {
        if (image.source < sourceKinds.size()) sourceKinds[image.source] |= ReachImage;
    }
    for (const auto& sampler : plan.info.samplers) {
        if (sampler.source < sourceKinds.size()) sourceKinds[sampler.source] |= ReachSampler;
    }
    for (std::size_t source = 0; source < plan.descriptorSources.size(); ++source) {
        const auto& descriptor = plan.descriptorSources[source];
        const auto kinds = sourceKinds[source];
        for (std::uint32_t dword = 0; dword < descriptor.dwordCount && dword < descriptor.dwords.size(); ++dword) {
            if (kinds == 0u) {
                mark(descriptor.dwords[dword], ReachOtherSource, true);
                continue;
            }
            if ((kinds & ReachImage) != 0u) mark(descriptor.dwords[dword], ReachImage, true);
            if ((kinds & ReachSampler) != 0u) mark(descriptor.dwords[dword], ReachSampler, true);
            if ((kinds & ReachBufferBase) != 0u) mark(descriptor.dwords[dword], dword < 2u ? ReachBufferBase : ReachBufferShape, true);
        }
    }
    for (const auto& block : plan.controlFlow) mark(block.condition, ReachCondition, true);
    if (plan.uniformFill.fill.kind != ShaderRecompiler::UniformFillKind::None) {
        for (std::uint32_t i = 0; i < plan.uniformFill.fill.words; ++i) mark(plan.uniformFill.values[i], ReachUniform, true);
    }
    for (const auto& read : plan.srtReads) {
        if (read.value == nullptr) continue;
        IrValue* raw = read.value->Resolve();
        for (std::size_t i = 0; i < raw->ArgumentCount(); ++i) mark(raw->Argument(i), ReachCone, false);
    }
    const auto userIndex = [](std::uint8_t bit) { return static_cast<std::size_t>(std::countr_zero(static_cast<unsigned>(bit))); };
    result.valueUserWords = userReach[userIndex(ReachCondition)] | userReach[userIndex(ReachUniform)];
    result.coneUserWords = userReach[userIndex(ReachCone)];
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const auto reach = result.reach[slot];
        const auto slotClass = (reach & StructuralReach) != 0u ? SlotClass::Structural : (reach & ReachBufferBase) != 0u ? SlotClass::Base : (reach & ReachCone) != 0u ? SlotClass::Cone : SlotClass::Pure;
        result.slots[slot] = slotClass;
        ++result.counts[static_cast<std::size_t>(slotClass)];
    }
    return result;
}

const char* DiffName(Diff diff) {
    switch (diff) {
        case Diff::Texture: return "T#";
        case Diff::Sampler: return "S#";
        case Diff::BufferShape: return "V# size/stride/format";
        case Diff::BaseAlignment: return "base alignment";
        case Diff::Condition: return "condition";
        case Diff::UniformFill: return "uniform fill";
        case Diff::UserValue: return "user value word";
        case Diff::Vertex: return "vertex structural";
        case Diff::Other: return "other";
        case Diff::DrawParameters: return "draw parameters";
        case Diff::NoPlan: return "no plan";
        case Diff::Unrecorded: return "unrecorded";
        case Diff::NewShape: return "new shape";
        case Diff::Count: break;
    }
    return "?";
}

std::uint64_t StageRecord::Combined() const {
    std::uint64_t hash = Seed;
    for (const auto value : hashes) hash = mix(hash, value);
    return hash;
}

StageRecord DescribeStage(const SlotClasses& classes, const ShaderRecompiler::IrResourcePlan& plan, const ShaderRecompiler::ResourceSnapshot& snapshot, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderVertexStageInfo* vertex) {
    StageRecord record;
    record.eligible = classes.eligible;
    record.slotCounts = classes.counts;
    for (std::size_t c = 0; c < HashedClasses; ++c) record.hashes[c] = classSeed(c);
    if (!record.eligible) return record;
    auto& texture = record.hashes[classIndex(Diff::Texture)];
    texture = mix(texture, snapshot.images.size());
    for (const auto& image : snapshot.images) hashDescriptor(texture, image);
    auto& sampler = record.hashes[classIndex(Diff::Sampler)];
    sampler = mix(sampler, snapshot.samplers.size());
    for (const auto& value : snapshot.samplers) hashDescriptor(sampler, value);
    auto& shape = record.hashes[classIndex(Diff::BufferShape)];
    auto& alignment = record.hashes[classIndex(Diff::BaseAlignment)];
    shape = mix(shape, snapshot.buffers.size());
    for (const auto& buffer : snapshot.buffers) {
        shape = mix(shape, buffer.dwordCount);
        shape = mix(shape, buffer.availableDwordMask);
        for (std::uint32_t dword = 1; dword < buffer.dwordCount && dword < buffer.dwords.size(); ++dword) shape = mix(shape, dword == 1u ? buffer.dwords[dword] & 0xffff0000u : buffer.dwords[dword]);
        alignment = mix(alignment, buffer.dwords[0] & 3u);
    }
    const auto slotValue = [&](std::size_t slot) -> std::uint64_t { return slot < snapshot.flattenedSrt.size() ? snapshot.flattenedSrt[slot] : 0xffffffffffffffffull; };
    auto& condition = record.hashes[classIndex(Diff::Condition)];
    auto& uniform = record.hashes[classIndex(Diff::UniformFill)];
    auto& other = record.hashes[classIndex(Diff::Other)];
    for (std::size_t slot = 0; slot < classes.reach.size(); ++slot) {
        const auto reach = classes.reach[slot];
        if ((reach & ReachCondition) != 0u) condition = mix(mix(condition, slot), slotValue(slot));
        if ((reach & ReachUniform) != 0u) uniform = mix(mix(uniform, slot), slotValue(slot));
        if ((reach & ReachOtherSource) != 0u) other = mix(mix(other, slot), slotValue(slot));
    }
    const auto& fill = snapshot.uniformFill;
    uniform = mix(uniform, static_cast<std::uint64_t>(fill.kind));
    uniform = mix(uniform, fill.resource);
    for (const auto stride : fill.groupStride) uniform = mix(uniform, stride);
    uniform = mix(uniform, fill.words);
    uniform = mix(uniform, fill.value);
    auto& user = record.hashes[classIndex(Diff::UserValue)];
    for (std::uint32_t word = 0; word < 64u; ++word) {
        if ((classes.valueUserWords >> word & 1u) == 0u) continue;
        user = mix(mix(user, word), word < userData.size() ? userData[word] : 0xffffffffffffffffull);
    }
    auto& vertexHash = record.hashes[classIndex(Diff::Vertex)];
    if (vertex != nullptr) {
        vertexHash = mix(vertexHash, 1u);
        vertexHash = mix(vertexHash, vertex->resourcesNum);
        vertexHash = mix(vertexHash, vertex->fetchAttribReg);
        vertexHash = mix(vertexHash, vertex->fetchBufferReg);
        vertexHash = mix(vertexHash, vertex->fetchEmbedded ? 1u : 0u);
        vertexHash = mix(vertexHash, vertex->paClVsOutCntl);
        vertexHash = mix(vertexHash, static_cast<std::uint32_t>(vertex->drawBaseVertexSgpr));
        vertexHash = mix(vertexHash, static_cast<std::uint32_t>(vertex->drawStartInstanceSgpr));
        vertexHash = mix(vertexHash, vertex->depthRemapMode);
        vertexHash = mix(vertexHash, vertex->pointList ? 1u : 0u);
        vertexHash = mix(vertexHash, std::bit_cast<std::uint32_t>(vertex->pointSize));
        vertexHash = mix(vertexHash, std::bit_cast<std::uint32_t>(vertex->pointSizeMin));
        vertexHash = mix(vertexHash, std::bit_cast<std::uint32_t>(vertex->pointSizeMax));
        for (std::uint32_t i = 0; i < vertex->resourcesNum && i < vertex->resources.size(); ++i) {
            const auto& fields = vertex->resources[i].fields;
            vertexHash = mix(vertexHash, fields[0] & 3u);
            vertexHash = mix(vertexHash, fields[1] & 0xffff0000u);
            vertexHash = mix(vertexHash, fields[2]);
            vertexHash = mix(vertexHash, fields[3]);
            const auto& destination = vertex->resourcesDst[i];
            vertexHash = mix(vertexHash, static_cast<std::uint32_t>(destination.registerStart));
            vertexHash = mix(vertexHash, static_cast<std::uint32_t>(destination.registersNum));
            vertexHash = mix(vertexHash, static_cast<std::uint32_t>(destination.attrId));
            vertexHash = mix(vertexHash, destination.fetchIndex);
        }
    }
    other = mix(other, snapshot.srtPoison.size());
    other = mix(other, snapshot.nullRootReads);
    other = mix(other, snapshot.tables.tables.size());
    other = mix(other, plan.srtReads.size());
    return record;
}

std::shared_ptr<const StageRecord> DescribeCapture(const ShaderRecompiler::ResourceCapture* capture, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderVertexStageInfo* vertex) {
    if (capture == nullptr || capture->plan == nullptr) return std::make_shared<const StageRecord>();
    struct Known {
        std::weak_ptr<const ShaderRecompiler::IrResourcePlan> plan;
        std::shared_ptr<const SlotClasses> classes;
    };
    static std::mutex mutex;
    static std::unordered_map<const ShaderRecompiler::IrResourcePlan*, Known> known;
    const auto& plan = capture->plan;
    std::shared_ptr<const SlotClasses> classes;
    {
        std::lock_guard lock(mutex);
        if (const auto found = known.find(plan.get()); found != known.end() && found->second.plan.lock() == plan) classes = found->second.classes;
    }
    if (classes == nullptr) {
        classes = std::make_shared<const SlotClasses>(ComputeSlotClasses(*plan));
        std::lock_guard lock(mutex);
        if (known.size() >= 65536u) std::erase_if(known, [](const auto& entry) { return entry.second.plan.expired(); });
        if (known.size() >= 65536u) known.clear();
        known[plan.get()] = {plan, classes};
    }
    return std::make_shared<const StageRecord>(DescribeStage(*classes, *plan, capture->snapshot, userData, vertex));
}

std::uint64_t DrawParameterBits(bool indexed, std::uint32_t indexSize, std::uint32_t instanceCount, bool indirect) {
    return (indexed ? 1u : 0u) | (static_cast<std::uint64_t>(indexed ? indexSize : 0u) << 1u) | (instanceCount > 1u ? 1u << 8u : 0u) | (indirect ? 1u << 9u : 0u);
}

void Tally::Add(const Outcome& outcome) {
    ++lookups;
    if (outcome.hit) ++hits[static_cast<std::size_t>(outcome.age)];
    else if (outcome.diff < Diff::Count) ++misses[classIndex(outcome.diff)];
}

std::uint64_t Tally::Hits() const {
    std::uint64_t total = 0;
    for (const auto count : hits) total += count;
    return total;
}

Outcome Predictor::note(std::uint64_t key, const std::array<std::uint64_t, HashedClasses>& hashes, std::uint64_t combined, std::uint64_t frame) {
    if (rings.size() >= MaxRings && !rings.contains(key)) {
        rings.clear();
        ++resets;
    }
    auto& ring = rings[key];
    Outcome outcome;
    for (std::uint32_t entry = 0; entry < ring.size; ++entry) {
        if (ring.hashes[entry] != combined) continue;
        const auto seen = ring.frames[entry];
        outcome.hit = true;
        outcome.age = seen == frame ? Age::SameFrame : seen + 1u == frame ? Age::PreviousFrame : Age::Older;
        ring.frames[entry] = frame;
        ring.last = hashes;
        return outcome;
    }
    if (ring.size == 0) {
        outcome.diff = Diff::NewShape;
    } else {
        outcome.diff = Diff::Other;
        for (std::size_t c = 0; c < HashedClasses; ++c) {
            if (ring.last[c] == hashes[c]) continue;
            outcome.diff = static_cast<Diff>(c);
            break;
        }
    }
    ring.hashes[ring.next] = combined;
    ring.frames[ring.next] = frame;
    ring.next = static_cast<std::uint32_t>((ring.next + 1u) % RingEntries);
    ring.size = std::min<std::uint32_t>(ring.size + 1u, RingEntries);
    ring.last = hashes;
    return outcome;
}

Outcome Predictor::NoteStage(std::uint64_t shape, std::size_t stage, const StageRecord* record, std::uint64_t frame) {
    if (record == nullptr) return {false, Age::SameFrame, Diff::Unrecorded};
    if (!record->eligible) return {false, Age::SameFrame, Diff::NoPlan};
    return note(stageKey(shape, stage), record->hashes, record->Combined(), frame);
}

Outcome Predictor::NoteDraw(std::uint64_t shape, std::span<const StageRecord* const> stages, std::uint64_t drawParameters, std::uint64_t frame) {
    for (const auto* record : stages) {
        if (record == nullptr) return {false, Age::SameFrame, Diff::Unrecorded};
    }
    for (const auto* record : stages) {
        if (!record->eligible) return {false, Age::SameFrame, Diff::NoPlan};
    }
    std::array<std::uint64_t, HashedClasses> hashes{};
    for (std::size_t c = 0; c < HashedClasses; ++c) {
        auto hash = classSeed(c);
        for (std::size_t s = 0; s < stages.size(); ++s) hash = mix(mix(hash, s), stages[s]->hashes[c]);
        hashes[c] = hash;
    }
    hashes[classIndex(Diff::DrawParameters)] = mix(hashes[classIndex(Diff::DrawParameters)], drawParameters);
    std::uint64_t combined = Seed;
    for (const auto value : hashes) combined = mix(combined, value);
    return note(drawKey(shape), hashes, combined, frame);
}

void Predictor::Clear() {
    rings.clear();
}

namespace {

struct Totals {
    std::mutex mutex;
    Predictor predictor;
    Tally draws;
    std::uint64_t cumulativeDraws = 0;
    std::uint64_t cumulativeHits = 0;
    std::array<Tally, Predictor::MaxStages> stages{};
    std::uint64_t unkeyed = 0;
    std::uint64_t exact = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

Totals& totals() {
    static Totals state;
    return state;
}

std::atomic<std::uint64_t> currentFrame{0};

struct ResourceTotals {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::uint64_t> seen;
    std::uint64_t lookups = 0;
    std::uint64_t misses = 0;
    std::array<std::uint64_t, 4> split{};
};

ResourceTotals& resourceTotals() {
    static ResourceTotals state;
    return state;
}

struct RecordTotals {
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(RecordPart::Count)> nanoseconds{};
    std::atomic<std::uint64_t> draws{0};
};

RecordTotals& recordTotals() {
    static RecordTotals state;
    return state;
}

struct RecordLaps {
    bool active = false;
    std::chrono::steady_clock::time_point last{};
    std::array<std::uint64_t, static_cast<std::size_t>(RecordPart::Count)> nanoseconds{};
};

RecordLaps& recordLaps() {
    thread_local RecordLaps laps;
    return laps;
}

constexpr std::array<const char*, static_cast<std::size_t>(RecordPart::Count)> RecordPartNames{"prelude", "bindings snapshot", "bindings set", "bindings update", "pass access", "start pass", "emit", "leave pass"};

void appendf(std::string& text, const char* format, auto... values) {
    char buffer[512];
    const int written = std::snprintf(buffer, sizeof(buffer), format, values...);
    if (written > 0) text.append(buffer, std::min<std::size_t>(static_cast<std::size_t>(written), sizeof(buffer) - 1u));
}

double percent(std::uint64_t part, std::uint64_t whole) {
    return whole != 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
}

void appendTally(std::string& text, const Tally& tally) {
    const auto hits = tally.Hits();
    appendf(text, "predicted hits %llu = %.1f %% (same frame %llu, previous frame %llu, older %llu); misses by first differing class:", static_cast<unsigned long long>(hits), percent(hits, tally.lookups), static_cast<unsigned long long>(tally.hits[0]), static_cast<unsigned long long>(tally.hits[1]), static_cast<unsigned long long>(tally.hits[2]));
    for (std::size_t d = 0; d < static_cast<std::size_t>(Diff::Count); ++d) appendf(text, "%s %s %llu", d == 0 ? "" : ",", DiffName(static_cast<Diff>(d)), static_cast<unsigned long long>(tally.misses[d]));
}

}

void NoteFrame(std::uint64_t frame) {
    currentFrame.store(frame, std::memory_order_relaxed);
}

std::uint64_t Frame() {
    return currentFrame.load(std::memory_order_relaxed);
}

void NoteUnkeyedDraw() {
    auto& state = totals();
    std::lock_guard lock(state.mutex);
    ++state.unkeyed;
}

void NoteDraw(std::uint64_t shape, std::span<const StageRecord* const> stages, std::uint64_t drawParameters, std::uint64_t frame, bool exactHit) {
    NoteFrame(frame);
    auto& state = totals();
    std::string report;
    {
        std::lock_guard lock(state.mutex);
        const auto outcome = state.predictor.NoteDraw(shape, stages, drawParameters, frame);
        state.draws.Add(outcome);
        ++state.cumulativeDraws;
        if (outcome.hit) ++state.cumulativeHits;
        if (exactHit) ++state.exact;
        for (std::size_t s = 0; s < stages.size() && s < state.stages.size(); ++s) {
            auto& tally = state.stages[s];
            tally.Add(state.predictor.NoteStage(shape, s, stages[s], frame));
            if (stages[s] != nullptr) {
                for (std::size_t c = 0; c < 4; ++c) tally.slots[c] += stages[s]->slotCounts[c];
            }
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - state.lastReport >= std::chrono::seconds(10)) {
            state.lastReport = now;
            report = ReportLines(true);
        }
    }
    if (!report.empty()) std::fputs(report.c_str(), stderr);
}

void NoteResourceLookup(std::span<const std::uint32_t> key, bool hit) {
    std::uint64_t hash = Seed;
    hash = mix(hash, key.size());
    for (const auto word : key) hash = mix(hash, word);
    const auto frame = Frame();
    auto& state = resourceTotals();
    std::lock_guard lock(state.mutex);
    ++state.lookups;
    if (state.seen.size() >= (1u << 20u)) state.seen.clear();
    const auto [found, inserted] = state.seen.try_emplace(hash, frame);
    if (!hit) {
        ++state.misses;
        const auto previous = found->second;
        ++state.split[inserted ? 3u : previous == frame ? 1u : previous + 1u == frame ? 0u : 2u];
    }
    found->second = frame;
}

void RecordStart() {
    auto& laps = recordLaps();
    laps.active = true;
    laps.nanoseconds.fill(0);
    laps.last = std::chrono::steady_clock::now();
}

void RecordLap(RecordPart part) {
    auto& laps = recordLaps();
    if (!laps.active) return;
    const auto now = std::chrono::steady_clock::now();
    laps.nanoseconds[static_cast<std::size_t>(part)] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - laps.last).count());
    laps.last = now;
}

void RecordEnd() {
    auto& laps = recordLaps();
    if (!laps.active) return;
    laps.active = false;
    auto& state = recordTotals();
    for (std::size_t part = 0; part < laps.nanoseconds.size(); ++part) state.nanoseconds[part].fetch_add(laps.nanoseconds[part], std::memory_order_relaxed);
    state.draws.fetch_add(1, std::memory_order_relaxed);
}

std::string ReportLines(bool reset) {
    auto& state = totals();
    std::string text;
    appendf(text, "[shape-replay] 10 s: %llu draws (%llu unkeyed, %llu exact draw-cache hits today); per draw (all stages): ", static_cast<unsigned long long>(state.draws.lookups), static_cast<unsigned long long>(state.unkeyed), static_cast<unsigned long long>(state.exact));
    appendTally(text, state.draws);
    appendf(text, "; cumulative %.1f %% of %llu draws; rings %zu (resets %llu)\n", percent(state.cumulativeHits, state.cumulativeDraws), static_cast<unsigned long long>(state.cumulativeDraws), state.predictor.Rings(), static_cast<unsigned long long>(state.predictor.Resets()));
    for (std::size_t s = 0; s < state.stages.size(); ++s) {
        const auto& tally = state.stages[s];
        if (tally.lookups == 0) continue;
        appendf(text, "[shape-replay] stage %zu: %llu lookups, ", s, static_cast<unsigned long long>(tally.lookups));
        appendTally(text, tally);
        const auto average = [&](std::size_t c) { return static_cast<double>(tally.slots[c]) / static_cast<double>(tally.lookups); };
        appendf(text, "; slots per draw P %.1f C %.1f B %.1f S %.1f\n", average(0), average(1), average(2), average(3));
    }
    auto& record = recordTotals();
    const auto recorded = reset ? record.draws.exchange(0, std::memory_order_relaxed) : record.draws.load(std::memory_order_relaxed);
    appendf(text, "[shape-replay] record (10 s, %llu draws, avg us):", static_cast<unsigned long long>(recorded));
    for (std::size_t part = 0; part < record.nanoseconds.size(); ++part) {
        const auto nanoseconds = reset ? record.nanoseconds[part].exchange(0, std::memory_order_relaxed) : record.nanoseconds[part].load(std::memory_order_relaxed);
        appendf(text, "%s %s %.1f", part == 0 ? "" : ",", RecordPartNames[part], recorded != 0 ? static_cast<double>(nanoseconds) / 1000.0 / static_cast<double>(recorded) : 0.0);
    }
    text += "\n";
    {
        auto& resources = resourceTotals();
        std::lock_guard lock(resources.mutex);
        appendf(text, "[rescache] shape-replay split (10 s): %llu draw misses: %llu key seen in the previous frame, %llu earlier in this frame, %llu in an older frame, %llu never seen (%llu lookups, %zu keys known)\n", static_cast<unsigned long long>(resources.misses), static_cast<unsigned long long>(resources.split[0]), static_cast<unsigned long long>(resources.split[1]), static_cast<unsigned long long>(resources.split[2]), static_cast<unsigned long long>(resources.split[3]), static_cast<unsigned long long>(resources.lookups), resources.seen.size());
        if (reset) {
            resources.misses = resources.lookups = 0;
            resources.split.fill(0);
        }
    }
    if (reset) {
        state.draws = {};
        for (auto& tally : state.stages) tally = {};
        state.unkeyed = state.exact = 0;
    }
    return text;
}

}
