#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DrawTwins.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ResolvedPush.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/Router.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace AgcDriver::NewPath;
using Stage = ShaderRecompiler::ShaderStage;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

CommandFacts ReadyDraw() {
    CommandFacts facts;
    facts.kind = CommandKind::Draw;
    facts.resolvedProgram = true;
    return facts;
}

void routerDrawFacts() {
    Expect(Router(true).Decide(ReadyDraw()).reason == FallbackReason::KindNotEmitted, "draws were emitted with only the dispatch switch");
    Expect(Router(false, true).Decide(ReadyDraw()).route == Route::NewPath, "a ready class A draw was not emitted with the draw switch");
    CommandFacts dispatch;
    dispatch.kind = CommandKind::Dispatch;
    dispatch.resolvedProgram = true;
    Expect(Router(false, true).Decide(dispatch).reason == FallbackReason::KindNotEmitted, "the draw switch emitted a dispatch");
    Expect(Router(true, false).Decide(dispatch).route == Route::NewPath, "the draw facts' defaults changed a dispatch's route");
    const Router router(false, true);
    struct Case {
        const char* name;
        void (*set)(CommandFacts&);
        FallbackReason reason;
    };
    const Case cases[] = {
        {"kind switch", [](CommandFacts& facts) { facts.kindSwitchOff = true; facts.classA = false; }, FallbackReason::KindSwitch},
        {"not class A", [](CommandFacts& facts) { facts.classA = false; facts.shapeSupported = false; facts.resolvedProgram = false; }, FallbackReason::NotClassA},
        {"shape", [](CommandFacts& facts) { facts.shapeSupported = false; facts.lean = false; }, FallbackReason::DrawShape},
        {"not lean", [](CommandFacts& facts) { facts.lean = false; facts.resolvedProgram = false; }, FallbackReason::NotLean},
        {"no twin", [](CommandFacts& facts) { facts.resolvedProgram = false; facts.pipelineReady = false; }, FallbackReason::NoResolvedModule},
        {"no pipeline", [](CommandFacts& facts) { facts.pipelineReady = false; facts.tableCovers = false; }, FallbackReason::NoPipeline},
        {"range", [](CommandFacts& facts) { facts.tableCovers = false; facts.pushFits = false; }, FallbackReason::RangeNotCovered},
        {"push", [](CommandFacts& facts) { facts.pushFits = false; facts.clothGuard = true; }, FallbackReason::PushOverflow},
        {"cloth", [](CommandFacts& facts) { facts.clothGuard = true; }, FallbackReason::ClothGuard},
    };
    for (const auto& item : cases) {
        auto facts = ReadyDraw();
        item.set(facts);
        const auto decision = router.Decide(facts);
        Expect(decision.route == Route::Fallback && decision.reason == item.reason, std::string("draw case ") + item.name + " decided " + FallbackReasonName(decision.reason));
    }
    for (const auto reason : {FallbackReason::NotClassA, FallbackReason::DrawShape, FallbackReason::NotLean}) Expect(std::string(FallbackReasonName(reason)).find(' ') == std::string::npos && std::string(FallbackReasonName(reason)) != "unknown", "a draw reason has no one-word name");
    Router counting(false, true);
    counting.Note(CommandKind::Draw, {Route::Fallback, FallbackReason::NotClassA});
    counting.Note(CommandKind::Draw, {Route::NewPath, FallbackReason::KindNotEmitted});
    const auto line = Router::Format(counting.Take(), "w");
    Expect(line.find("draw new 1 fallback not-class-a 1") != std::string::npos, "the draw part of the line: " + line);
}

std::shared_ptr<ShaderRecompiler::RecompileResult> Program(bool resolved, std::size_t pushBytes) {
    auto program = std::make_shared<ShaderRecompiler::RecompileResult>();
    program->newPathResolved = resolved;
    ShaderRecompiler::DescriptorBinding table{};
    table.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    table.role = ShaderRecompiler::DescriptorRole::BdaPagetable;
    table.binding = 3;
    table.count = 1;
    program->bindings.push_back(table);
    for (std::size_t index = 0; index < pushBytes; ++index) program->pushConstants.push_back(static_cast<std::byte>(index + 1));
    return program;
}

void shapeRefusals() {
    const auto vertex = Program(false, 16);
    const auto pixel = Program(false, 8);
    const std::vector<DrawStage> plain{{Stage::Vertex, vertex.get(), 0}, {Stage::Fragment, pixel.get(), 128}};
    Expect(DrawShapeRefusal(plain, {}) == ShapeRefusal::None, "a plain vs+ps draw was refused");
    const std::vector<DrawStage> geometry{{Stage::Vertex, vertex.get(), 0}, {Stage::Geometry, vertex.get(), 16}, {Stage::Fragment, pixel.get(), 128}};
    Expect(DrawShapeRefusal(geometry, {}) == ShapeRefusal::Stages, "a geometry stage was accepted");
    Expect(DrawShapeRefusal({}, {}) == ShapeRefusal::Stages, "a draw without stages was accepted");
    DrawShape shape;
    shape.tessellation = true;
    Expect(DrawShapeRefusal(plain, shape) == ShapeRefusal::Stages, "tessellation was accepted");
    shape = {};
    shape.depthRemap = true;
    Expect(DrawShapeRefusal(plain, shape) == ShapeRefusal::FixedPushWords, "a depth remap (fixed push words) was accepted");
    auto remapping = Program(false, 16);
    remapping->readsDepthRemapPush = true;
    const std::vector<DrawStage> remapped{{Stage::Vertex, remapping.get(), 0}, {Stage::Fragment, pixel.get(), 128}};
    Expect(DrawShapeRefusal(remapped, {}) == ShapeRefusal::FixedPushWords, "a stage reading the depth remap words was accepted");
    auto stencil = Program(false, 8);
    stencil->stencilReferences = {1};
    const std::vector<DrawStage> replayed{{Stage::Vertex, vertex.get(), 0}, {Stage::Fragment, stencil.get(), 128}};
    Expect(DrawShapeRefusal(replayed, {}) == ShapeRefusal::StencilReplay, "stencil replay was accepted");
    shape = {};
    shape.blendedTargets = true;
    Expect(DrawShapeRefusal(plain, shape) == ShapeRefusal::BlendedTargets, "blended targets were accepted");
    auto lds = Program(false, 8);
    lds->functionLdsBuffer = true;
    const std::vector<DrawStage> withLds{{Stage::Vertex, vertex.get(), 0}, {Stage::Fragment, lds.get(), 128}};
    Expect(DrawShapeRefusal(withLds, {}) == ShapeRefusal::PixelLds, "a pixel LDS buffer was accepted");
    auto bary = Program(false, 8);
    bary->barycentricEmulation.active = true;
    const std::vector<DrawStage> withBary{{Stage::Vertex, vertex.get(), 0}, {Stage::Fragment, bary.get(), 128}};
    Expect(DrawShapeRefusal(withBary, {}) == ShapeRefusal::BarycentricEmulation, "barycentric emulation was accepted");
    shape = {};
    shape.splitPipeline = true;
    Expect(DrawShapeRefusal(plain, shape) == ShapeRefusal::SplitPipeline, "a split pipeline was accepted");
}

void pushComposition() {
    const auto vertex = Program(false, 16);
    const auto pixel = Program(false, 72);
    const std::vector<DrawStage> slots{{Stage::Vertex, vertex.get(), 0}, {Stage::Fragment, pixel.get(), 128}};
    Expect(DrawPushEnd(slots) == 200 && DrawPushFits(slots), "fixed slots: the push end is not 200 or does not fit");
    const auto wide = Program(false, 112);
    const std::vector<DrawStage> overflow{{Stage::Vertex, vertex.get(), 0}, {Stage::Fragment, wide.get(), 128}};
    Expect(DrawPushEnd(overflow) == 240 && !DrawPushFits(overflow), "a block ending at 240 bytes fit the 24-byte prefix");
    const auto edge = Program(false, 104);
    const std::vector<DrawStage> exact{{Stage::Vertex, vertex.get(), 0}, {Stage::Fragment, edge.get(), 128}};
    Expect(DrawPushEnd(exact) == 232 && DrawPushFits(exact), "a block ending at 232 bytes did not fit");
    const auto none = Program(false, 0);
    const std::vector<DrawStage> empty{{Stage::Vertex, none.get(), 0}, {Stage::Fragment, none.get(), 128}};
    Expect(DrawPushEnd(empty) == 0 && DrawPushFits(empty), "a draw without push words did not fit");
    const std::vector<DrawStage> packed{{Stage::Vertex, vertex.get(), 0}, {Stage::Fragment, pixel.get(), 16}};
    Expect(DrawPushEnd(packed) == 88, "packed stages: the push end is not 88");

    std::array<std::byte, Abi::PushBlockBytes> assembled{};
    for (std::size_t index = 0; index < 16; ++index) assembled[index] = static_cast<std::byte>(0x10 + index);
    for (std::size_t index = 0; index < 72; ++index) assembled[128 + index] = static_cast<std::byte>(0x80 + index);
    const std::uint64_t root = 0x1122334455667788ull;
    const std::uint64_t record = 0x99aabbccddeeff00ull;
    const auto composed = ComposeDrawPush(assembled, slots, root, record);
    Expect(composed.has_value(), "a fitting draw push was not composed");
    if (composed) {
        std::uint64_t words[2]{};
        std::memcpy(&words[0], composed->data() + ResolvedPageTableWord * 4u, 8);
        std::memcpy(&words[1], composed->data() + ResolvedFaultRecordWord * 4u, 8);
        Expect(words[0] == root && words[1] == record, "the prefix does not hold the table root and the command record");
        Expect(std::memcmp(composed->data() + 24, assembled.data(), 16) == 0, "the vertex stage's words are not at 24");
        Expect(std::memcmp(composed->data() + 152, assembled.data() + 128, 72) == 0, "the pixel stage's words are not at 128 + 24");
        Expect(std::all_of(composed->begin() + 224, composed->end(), [](std::byte value) { return value == std::byte{0}; }), "bytes behind the shifted block are not zero");
        std::uint32_t overflowWords[2]{};
        std::memcpy(overflowWords, composed->data() + 16, 8);
        Expect(overflowWords[0] == 0 && overflowWords[1] == 0, "the overflow words 4-5 are not zero");
    }
    Expect(!ComposeDrawPush(assembled, overflow, root, record).has_value(), "an overflowing draw push was composed");
    Expect(!ComposeDrawPush(assembled, slots, 0, record).has_value(), "a draw push without a table root was composed");
    Expect(!ComposeDrawPush(assembled, slots, root, 0).has_value(), "a draw push without a command record was composed");
}

void stageTwins() {
    const auto compiled = Program(false, 16);
    const auto twin = Program(true, 16);
    Expect(StageTwinCompatible(*compiled, *twin), "a matching stage twin was refused: " + StageTwinDifference(*compiled, *twin));
    Expect(!StageTwinCompatible(*compiled, *compiled), "an unresolved stage was its own twin");
    auto exports = Program(true, 16);
    exports->parameterExports = {1};
    Expect(StageTwinDifference(*compiled, *exports) == "parameters", "other parameter exports: " + StageTwinDifference(*compiled, *exports));
    auto inputs = Program(true, 16);
    inputs->vertexInputs.push_back({0, 4, 0, 0});
    Expect(StageTwinDifference(*compiled, *inputs) == "vertex inputs", "other vertex inputs: " + StageTwinDifference(*compiled, *inputs));
    auto attributes = Program(true, 16);
    attributes->vertexAttributes.push_back({0, 4, {{1, 2, 3, 4}}, 0, 4});
    Expect(StageTwinDifference(*compiled, *attributes) == "vertex inputs", "other vertex attributes");
    auto patch = Program(true, 16);
    patch->memoryOffsetDword = 3;
    Expect(StageTwinDifference(*compiled, *patch) == "push patch positions", "another memory offset dword");
    auto stencil = Program(true, 16);
    stencil->stencilReferenceDword = 2;
    Expect(StageTwinDifference(*compiled, *stencil) == "stencil references", "another stencil reference dword");
    auto push = Program(true, 20);
    Expect(!StageTwinCompatible(*compiled, *push), "a stage twin with other push words was accepted");
    auto walkOnly = Program(true, 16);
    walkOnly->bindings.clear();
    Expect(StageTwinCompatible(*compiled, *walkOnly), "a stage twin that drops only the BDA table was refused");

    ResolvedTwins twins;
    const auto pixel = Program(false, 8);
    const auto pixelTwin = Program(true, 8);
    const std::vector<DrawStage> stages{{Stage::Vertex, compiled.get(), 0}, {Stage::Fragment, pixel.get(), 128}};
    twins.Remember(compiled, twin, TwinOutcome::Built);
    Expect(DrawTwinsOf(stages, twins).empty(), "a draw with a stage lacking its twin got twins");
    twins.Remember(pixel, nullptr, TwinOutcome::Failed);
    Expect(DrawTwinsOf(stages, twins).empty(), "a draw with a failed stage twin got twins");
    twins.Remember(pixel, pixelTwin, TwinOutcome::Built);
    const auto found = DrawTwinsOf(stages, twins);
    Expect(found.size() == 2 && found[0] == twin && found[1] == pixelTwin, "a draw's twins are not its stages' in order");
}

void emitLine() {
    DrawCounts().shapes[static_cast<std::size_t>(ShapeRefusal::Stages)].fetch_add(2);
    DrawCounts().covers[0].fetch_add(5);
    const auto line = FormatDrawEmit("10 s", true);
    Expect(line.rfind("[newpath-draw-emit] window=10 s emit draws on", 0) == 0, "line head: " + line);
    Expect(line.find("shape refusals stages 2") != std::string::npos && line.find("table compares covered 5") != std::string::npos, "line counts: " + line);
}

}

int main() {
    routerDrawFacts();
    shapeRefusals();
    pushComposition();
    stageTwins();
    emitLine();
    if (failures != 0) {
        std::fprintf(stderr, "agc_newpath_draw_twins: %d failures\n", failures);
        return 1;
    }
    std::printf("agc_newpath_draw_twins: ok\n");
    return 0;
}
