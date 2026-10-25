#include "prx/libSceAgcDriver/Graphics/include/ShapeReplayProfile.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/ShapeReplay.hpp"
#include <cstring>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ShapeReplay = AgcDriver::Graphics::ShapeReplay;
using ShapeReplay::Age;
using ShapeReplay::Diff;
using ShapeReplay::SlotClass;

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Builder {
    std::vector<std::unique_ptr<ShaderRecompiler::IrValue>> values;
    std::uint32_t ids = 0;
    ShaderRecompiler::IrValue* resource = nullptr;

    Builder() { resource = &make(ShaderRecompiler::IrOpcode::GetSrtResource, ShaderRecompiler::IrType::SrtResource); }

    ShaderRecompiler::IrValue& make(ShaderRecompiler::IrOpcode opcode, ShaderRecompiler::IrType type) {
        values.push_back(std::make_unique<ShaderRecompiler::IrValue>(opcode, type, ids++));
        return *values.back();
    }
    ShaderRecompiler::IrValue& constant(std::uint32_t value) {
        auto& immediate = make(ShaderRecompiler::IrOpcode::Void, ShaderRecompiler::IrType::U32);
        immediate.SetImmediateU32(value);
        return immediate;
    }
    ShaderRecompiler::IrValue& userData(std::uint32_t index) {
        auto& reg = make(ShaderRecompiler::IrOpcode::Void, ShaderRecompiler::IrType::ScalarReg);
        reg.SetRegister({ShaderRecompiler::RegisterBank::Scalar, index});
        auto& read = make(ShaderRecompiler::IrOpcode::GetUserData, ShaderRecompiler::IrType::U32);
        read.AddArgument(&reg);
        return read;
    }
    ShaderRecompiler::IrValue& pair(ShaderRecompiler::IrValue& low, ShaderRecompiler::IrValue& high) {
        auto& composed = make(ShaderRecompiler::IrOpcode::CompositeConstructU64, ShaderRecompiler::IrType::U64);
        composed.AddArgument(&low);
        composed.AddArgument(&high);
        return composed;
    }
    ShaderRecompiler::IrValue& rawRead(ShaderRecompiler::IrValue& address, std::uint32_t offset) {
        auto& read = make(ShaderRecompiler::IrOpcode::LoadAddressU32, ShaderRecompiler::IrType::U32);
        read.AddArgument(&address);
        read.AddArgument(&constant(offset));
        return read;
    }
    ShaderRecompiler::IrValue& readConst(std::uint32_t slot) {
        auto& read = make(ShaderRecompiler::IrOpcode::ReadConst, ShaderRecompiler::IrType::U32);
        read.AddArgument(resource);
        read.AddArgument(&constant(slot));
        return read;
    }
};

constexpr std::size_t DataSlot = 0, PointerSlot = 1, BaseSlot = 2, TextureSlot = 3, ConditionSlot = 4, SizeSlot = 5;
constexpr std::uint32_t ConditionUserWord = 9;
constexpr std::uint64_t Shape = 0x5a5e;

void buildPlan(Builder& ir, ShaderRecompiler::IrResourcePlan& plan) {
    auto& pointer = ir.rawRead(ir.pair(ir.userData(0), ir.userData(1)), 0);
    auto& data = ir.rawRead(ir.pair(ir.readConst(PointerSlot), ir.userData(4)), 8);
    auto& base = ir.rawRead(ir.pair(ir.userData(2), ir.userData(3)), 4);
    auto& texture = ir.rawRead(ir.pair(ir.userData(5), ir.userData(6)), 12);
    auto& condition = ir.rawRead(ir.pair(ir.userData(7), ir.userData(8)), 16);
    auto& size = ir.rawRead(ir.pair(ir.userData(10), ir.userData(11)), 20);
    plan.srtPlanComplete = true;
    plan.resourceTrackingComplete = true;
    plan.srtReads = {{&data, 0}, {&pointer, 1}, {&base, 2}, {&texture, 3}, {&condition, 4}, {&size, 5}};
    ShaderRecompiler::DescriptorSource buffer;
    buffer.dwordCount = 4;
    buffer.dwords[0] = &ir.readConst(BaseSlot);
    buffer.dwords[2] = &ir.readConst(SizeSlot);
    ShaderRecompiler::DescriptorSource image;
    image.dwordCount = 1;
    image.dwords[0] = &ir.readConst(TextureSlot);
    plan.descriptorSources = {buffer, image};
    plan.info.buffers.push_back({});
    plan.info.buffers.back().source = 0;
    plan.info.images.push_back({});
    plan.info.images.back().source = 1;
    auto& decided = ir.pair(ir.readConst(ConditionSlot), ir.userData(ConditionUserWord));
    plan.controlFlow.push_back({&decided, {}, {}, {}, {}});
}

struct Draw {
    ShaderRecompiler::ResourceSnapshot snapshot;
    std::vector<std::uint32_t> userData;
    ShaderRecompiler::ShaderVertexStageInfo vertex{};
    std::uint64_t parameters = ShapeReplay::DrawParameterBits(true, 2, 1, false);
};

Draw baseDraw() {
    Draw draw;
    draw.snapshot.flattenedSrt = {0x3f800000u, 0x11843460u, 0x20001000u, 0x00c0ffeeu, 1u, 256u};
    ShaderRecompiler::DescriptorValue buffer;
    buffer.dwords = {0x20001000u, 0x00100000u, 256u, 0x00027facu};
    buffer.dwordCount = 4;
    buffer.availableDwordMask = 0xfu;
    draw.snapshot.buffers = {buffer};
    ShaderRecompiler::DescriptorValue image;
    image.dwords = {0x00c0ffeeu, 0x00001234u, 0x3ff003ffu, 0x90500facu, 0u, 0u, 0u, 0u};
    image.dwordCount = 8;
    image.availableDwordMask = 0xffu;
    draw.snapshot.images = {image};
    draw.userData.assign(12, 0u);
    draw.userData[0] = 0x11843460u;
    draw.userData[ConditionUserWord] = 7u;
    draw.vertex.resourcesNum = 1;
    draw.vertex.resources[0].fields = {0x30000000u, 0x00200000u, 1024u, 0x00027facu};
    draw.vertex.resourcesDst[0] = {4, 4, 0, 0};
    draw.vertex.fetchAttribReg = 2;
    draw.vertex.fetchBufferReg = 4;
    return draw;
}

ShapeReplay::StageRecord describe(const ShapeReplay::SlotClasses& classes, const ShaderRecompiler::IrResourcePlan& plan, const Draw& draw) {
    return ShapeReplay::DescribeStage(classes, plan, draw.snapshot, draw.userData, &draw.vertex);
}

void classes() {
    Builder ir;
    ShaderRecompiler::IrResourcePlan plan;
    buildPlan(ir, plan);
    const auto computed = ShapeReplay::ComputeSlotClasses(plan);
    Require(computed.eligible, "slot classes: a complete plan was not eligible");
    const std::vector<SlotClass> expected{SlotClass::Pure, SlotClass::Cone, SlotClass::Base, SlotClass::Structural, SlotClass::Structural, SlotClass::Structural};
    Require(computed.slots == expected, "slot classes: pure / cone-only / V# base / structural were not told apart");
    Require((computed.reach[TextureSlot] & ShapeReplay::ReachImage) != 0 && (computed.reach[ConditionSlot] & ShapeReplay::ReachCondition) != 0 && (computed.reach[SizeSlot] & ShapeReplay::ReachBufferShape) != 0, "slot classes: a structural slot lacks its consumer");
    Require(computed.counts == std::array<std::uint32_t, 4>{1, 1, 1, 3}, "slot classes: counts per class");
    Require(computed.valueUserWords == (std::uint64_t{1} << ConditionUserWord), "slot classes: the user word read by a condition is not a value word");
    Require((computed.coneUserWords & 0x13u) == 0x13u, "slot classes: user words in an address cone were not cone-only");
    plan.requiresSpecializationMemory = true;
    const auto refused = ShapeReplay::ComputeSlotClasses(plan);
    Require(!refused.eligible && refused.counts[3] == plan.srtReads.size(), "slot classes: a plan with specialization memory was classified");
    plan.requiresSpecializationMemory = false;
    plan.srtPlanComplete = false;
    Require(!ShapeReplay::ComputeSlotClasses(plan).eligible, "slot classes: an incomplete plan was classified");
}

void predictedHits() {
    Builder ir;
    ShaderRecompiler::IrResourcePlan plan;
    buildPlan(ir, plan);
    const auto computed = ShapeReplay::ComputeSlotClasses(plan);
    ShapeReplay::Predictor predictor;

    const auto note = [&](const Draw& draw, std::uint64_t frame, std::uint64_t shape = Shape) {
        const auto record = describe(computed, plan, draw);
        const ShapeReplay::StageRecord* stages[] = {&record};
        return predictor.NoteDraw(shape, stages, draw.parameters, frame);
    };
    const auto first = note(baseDraw(), 1);
    Require(!first.hit && first.diff == Diff::NewShape, "predicted hits: the first draw of a shape was not a new-shape miss");

    auto moved = baseDraw();
    moved.snapshot.flattenedSrt[DataSlot] = 0x40000000u;
    moved.snapshot.flattenedSrt[PointerSlot] = 0x11908960u;
    moved.snapshot.flattenedSrt[BaseSlot] = 0x20002100u;
    moved.snapshot.buffers[0].dwords[0] = 0x20002100u;
    moved.userData[0] = 0x11908960u;
    moved.vertex.resources[0].fields[0] = 0x30004000u;
    const auto same = note(moved, 1);
    Require(same.hit && same.age == Age::SameFrame, "predicted hits: the same shape with moved data was not a same-frame hit");
    const auto next = note(moved, 2);
    Require(next.hit && next.age == Age::PreviousFrame, "predicted hits: a shape seen in the previous frame was not a previous-frame hit");
    const auto later = note(baseDraw(), 5);
    Require(later.hit && later.age == Age::Older, "predicted hits: a shape seen frames ago was not an older hit");

    const auto miss = [&](auto change, Diff expected, const char* what) {
        auto draw = baseDraw();
        change(draw);
        const auto outcome = note(draw, 6);
        Require(!outcome.hit && outcome.diff == expected, std::string("predicted hits: ") + what + " was not a miss of class " + ShapeReplay::DiffName(expected) + " (got " + (outcome.hit ? "a hit" : ShapeReplay::DiffName(outcome.diff)) + ")");
        Require(note(baseDraw(), 6).hit, std::string("predicted hits: the base draw after ") + what + " missed");
    };
    miss([](Draw& draw) { draw.snapshot.images[0].dwords[1] = 0x00005678u; }, Diff::Texture, "a changed T#");
    miss([](Draw& draw) { draw.snapshot.buffers[0].dwords[2] = 512u; draw.snapshot.flattenedSrt[SizeSlot] = 512u; }, Diff::BufferShape, "a changed V# size");
    miss([](Draw& draw) { draw.snapshot.buffers[0].dwords[1] = 0x00200000u; }, Diff::BufferShape, "a changed V# stride");
    miss([](Draw& draw) { draw.snapshot.buffers[0].dwords[0] += 2u; }, Diff::BaseAlignment, "a V# base off its dword alignment");
    miss([](Draw& draw) { draw.snapshot.flattenedSrt[ConditionSlot] = 0u; }, Diff::Condition, "a changed condition slot");
    miss([](Draw& draw) { draw.userData[ConditionUserWord] = 8u; }, Diff::UserValue, "a changed user value word");
    miss([](Draw& draw) { draw.vertex.resources[0].fields[1] = 0x00300000u; }, Diff::Vertex, "a changed vertex stride");
    miss([](Draw& draw) { draw.parameters = ShapeReplay::DrawParameterBits(true, 4, 1, false); }, Diff::DrawParameters, "a changed index size");
    const auto dataOnly = [&](auto change, Diff expected, bool data, const char* what) {
        auto draw = baseDraw();
        change(draw);
        const auto outcome = note(draw, 7);
        Require(!outcome.hit && outcome.diff == expected && outcome.descriptorData == data, std::string("descriptor data split: ") + what + (data ? " was not" : " was") + " counted as descriptor data only");
        Require(note(baseDraw(), 7).hit, std::string("descriptor data split: the base draw after ") + what + " missed");
    };
    dataOnly([](Draw& draw) { draw.snapshot.buffers[0].dwords[2] = 1024u; draw.snapshot.flattenedSrt[SizeSlot] = 1024u; }, Diff::BufferShape, true, "a V# size change");
    dataOnly([](Draw& draw) { draw.snapshot.buffers[0].dwords[1] = 0x00400000u; }, Diff::BufferShape, false, "a V# stride change");
    dataOnly([](Draw& draw) { draw.snapshot.images[0].dwords[0] = 0x00c0ff00u; }, Diff::Texture, true, "a T# base change");
    dataOnly([](Draw& draw) { draw.snapshot.images[0].dwords[2] = 0x1ff003ffu; }, Diff::Texture, false, "a T# size change");
    dataOnly([](Draw& draw) { draw.snapshot.images[0].dwords[0] = 0x00c0ff00u; draw.userData[ConditionUserWord] = 8u; }, Diff::Texture, false, "a T# base change with a user value word change");
    const auto other = note(baseDraw(), 6, Shape + 1);
    Require(!other.hit && other.diff == Diff::NewShape, "predicted hits: another shape shared a ring");

    const auto stage = predictor.NoteStage(Shape, 0, nullptr, 6);
    Require(!stage.hit && stage.diff == Diff::Unrecorded, "predicted hits: a stage without a record was not unrecorded");
    plan.srtPlanComplete = false;
    const auto ineligible = ShapeReplay::DescribeStage(ShapeReplay::ComputeSlotClasses(plan), plan, baseDraw().snapshot, baseDraw().userData, nullptr);
    const ShapeReplay::StageRecord* stages[] = {&ineligible};
    const auto noPlan = predictor.NoteDraw(Shape, stages, baseDraw().parameters, 6);
    Require(!noPlan.hit && noPlan.diff == Diff::NoPlan, "predicted hits: a draw with an ineligible stage was not a no-plan miss");
}

void patching() {
    namespace Replay = AgcDriver::DriverDetail::ShapeReplay;
    using ShaderRecompiler::DescriptorRole;
    ShaderRecompiler::ResourceSnapshot snapshot;
    snapshot.flattenedSrt = {1u, 2u, 3u};
    ShaderRecompiler::DescriptorValue first, second;
    first.dwords = {0x10000000u, 0x00100000u, 256u, 0x00027facu};
    first.dwordCount = 4;
    second.dwords = {0x20000000u, 0x00100000u, 128u, 0x00027facu};
    second.dwordCount = 4;
    snapshot.buffers = {first, second};
    snapshot.userData = {10u, 11u, 12u, 13u};
    ShaderRecompiler::ShaderVertexStageInfo vertex{};
    vertex.resourcesNum = 2;
    vertex.resources[1].fields = {0x30000000u, 0x00200000u, 64u, 0x00027facu};
    ShaderRecompiler::RecompileResult plan;
    plan.variantId = 7;
    plan.shaderDataUserWords = {2u, 0u};
    const auto binding = [](DescriptorRole role, std::vector<std::uint32_t> words) {
        ShaderRecompiler::DescriptorBinding made{};
        made.role = role;
        made.count = 1;
        made.guestDescriptor = std::move(words);
        return made;
    };
    plan.bindings.push_back(binding(DescriptorRole::FlattenedSrt, snapshot.flattenedSrt));
    plan.bindings.push_back(binding(DescriptorRole::GuestBuffers, {0x20000000u, 0x00100000u, 128u, 0x00027facu, 0x10000000u, 0x00100000u, 256u, 0x00027facu}));
    plan.bindings.push_back(binding(DescriptorRole::ShaderData, {12u, 10u, 0xabcu}));
    plan.vertexAttributes.push_back({1u, 4u, vertex.resources[1], 0u, 4u});
    const auto map = Replay::Learn(plan, snapshot, &vertex);
    Require(map.has_value() && map->flatBindings.size() == 1 && map->buffers.size() == 2 && map->shaderDataBinding == 2 && map->vertexAttributes && !map->pushUserWords, "patching: the data positions of a result were not learned");
    auto moved = snapshot;
    moved.flattenedSrt = {4u, 5u, 6u};
    moved.buffers[0].dwords[0] = 0x10000100u;
    moved.buffers[1].dwords[2] = 512u;
    moved.userData[2] = 99u;
    auto movedVertex = vertex;
    movedVertex.resources[1].fields[0] = 0x30000040u;
    const auto replayed = Replay::Apply(plan, *map, moved, &movedVertex);
    Require(replayed != nullptr, "patching: moved data was refused");
    auto expected = plan;
    expected.bindings[0].guestDescriptor = {4u, 5u, 6u};
    expected.bindings[1].guestDescriptor = {0x20000000u, 0x00100000u, 512u, 0x00027facu, 0x10000100u, 0x00100000u, 256u, 0x00027facu};
    expected.bindings[2].guestDescriptor = {99u, 10u, 0xabcu};
    expected.vertexAttributes[0].resource = movedVertex.resources[1];
    Require(Replay::Difference(*replayed, expected).empty(), "patching: the replayed result differs from the expected one in its " + Replay::Difference(*replayed, expected));
    auto pushed = expected;
    pushed.pushConstants.resize(4);
    Require(Replay::Difference(*replayed, pushed) == "push constants", "patching: a push constant difference was not reported");
    auto described = expected;
    described.bindings[1].guestDescriptor[6] = 0u;
    Require(Replay::Difference(*replayed, described).find("binding 1") != std::string::npos, "patching: a guest descriptor difference was not reported");

    auto pushPlan = plan;
    pushPlan.bindings.pop_back();
    pushPlan.pushConstants.resize(12);
    const std::uint32_t pushWords[3] = {12u, 10u, 0x55u};
    std::memcpy(pushPlan.pushConstants.data(), pushWords, sizeof(pushWords));
    const auto pushMap = Replay::Learn(pushPlan, snapshot, &vertex);
    Require(pushMap.has_value() && pushMap->pushUserWords, "patching: forwarded user words in push constants were not learned");
    const auto pushReplayed = Replay::Apply(pushPlan, *pushMap, moved, &movedVertex);
    std::uint32_t replayedPush[3] = {};
    std::memcpy(replayedPush, pushReplayed->pushConstants.data(), sizeof(replayedPush));
    Require(replayedPush[0] == 99u && replayedPush[1] == 10u && replayedPush[2] == 0x55u, "patching: forwarded user words were not refreshed in the push constants");

    auto twins = snapshot;
    twins.buffers[1] = twins.buffers[0];
    auto twinPlan = plan;
    twinPlan.bindings[1].guestDescriptor = {0x10000000u, 0x00100000u, 256u, 0x00027facu, 0x10000000u, 0x00100000u, 256u, 0x00027facu};
    const auto twinMap = Replay::Learn(twinPlan, twins, &vertex);
    Require(twinMap.has_value() && twinMap->buffers[0].candidates.size() == 2, "patching: equal V#s did not both stay candidates");
    auto split = twins;
    split.buffers[1].dwords[0] = 0x40000000u;
    Require(Replay::Apply(twinPlan, *twinMap, split, &vertex) == nullptr, "patching: an element whose candidate V#s diverged was patched");
    auto stale = plan;
    stale.bindings[0].guestDescriptor = {9u};
    Require(!Replay::Learn(stale, snapshot, &vertex).has_value(), "patching: a flattened SRT binding unlike the snapshot was learned");
    auto foreign = plan;
    foreign.bindings[1].guestDescriptor[0] = 0x77777777u;
    Require(!Replay::Learn(foreign, snapshot, &vertex).has_value(), "patching: a guest buffer element no snapshot V# explains was learned");
    Require(!Replay::Learn(plan, snapshot, nullptr).has_value(), "patching: vertex attributes without vertex stage info were learned");
}

void report() {
    Builder ir;
    ShaderRecompiler::IrResourcePlan plan;
    buildPlan(ir, plan);
    const auto computed = ShapeReplay::ComputeSlotClasses(plan);
    const auto record = describe(computed, plan, baseDraw());
    const ShapeReplay::StageRecord* stages[] = {&record};
    ShapeReplay::NoteDraw(7, stages, 0, 1, false);
    ShapeReplay::NoteDraw(7, stages, 0, 2, true);
    const std::vector<std::uint32_t> key{1, 2, 3};
    ShapeReplay::NoteFrame(2);
    ShapeReplay::NoteResourceLookup(key, false);
    ShapeReplay::NoteFrame(3);
    ShapeReplay::NoteResourceLookup(key, false);
    ShapeReplay::RecordStart();
    ShapeReplay::RecordLap(ShapeReplay::RecordPart::Prelude);
    ShapeReplay::RecordEnd();
    const auto text = ShapeReplay::ReportLines(true);
    std::fputs(text.c_str(), stdout);
    const auto has = [&](const char* part) { return text.find(part) != std::string::npos; };
    Require(has("[shape-replay] 10 s: 2 draws (0 unkeyed, 1 exact draw-cache hits today); per draw (all stages): predicted hits 1 = 50.0 % (same frame 0, previous frame 1, older 0)"), "report: the draw line");
    Require(has("[shape-replay] stage 0: 2 lookups, predicted hits 1 = 50.0 %") && has("slots per draw P 1.0 C 1.0 B 1.0 S 3.0"), "report: the stage line");
    Require(has("[shape-replay] record (10 s, 1 draws, avg us): prelude"), "report: the record line");
    Require(has("[rescache] shape-replay split (10 s): 2 draw misses: 1 key seen in the previous frame, 0 earlier in this frame, 0 in an older frame, 1 never seen"), "report: the resource cache split");
    const auto cleared = ShapeReplay::ReportLines(true);
    Require(cleared.find("[shape-replay] 10 s: 0 draws") != std::string::npos && cleared.find("cumulative 50.0 % of 2 draws") != std::string::npos, "report: a reset kept the window or lost the cumulative rate");
}

}

int main() {
    try {
        classes();
        predictedHits();
        patching();
        report();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::puts("shape replay profile: ok");
    return 0;
}
