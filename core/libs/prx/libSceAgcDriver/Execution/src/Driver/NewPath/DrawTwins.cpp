#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DrawTwins.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ResolvedPush.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

namespace AgcDriver::NewPath {

const char* ShapeRefusalName(ShapeRefusal refusal) {
    switch (refusal) {
    case ShapeRefusal::None: return "none";
    case ShapeRefusal::Stages: return "stages";
    case ShapeRefusal::FixedPushWords: return "fixed-push-words";
    case ShapeRefusal::StencilReplay: return "stencil-replay";
    case ShapeRefusal::BlendedTargets: return "blended-targets";
    case ShapeRefusal::PixelLds: return "pixel-lds";
    case ShapeRefusal::BarycentricEmulation: return "barycentric-emulation";
    case ShapeRefusal::SplitPipeline: return "split-pipeline";
    }
    return "unknown";
}

ShapeRefusal DrawShapeRefusal(std::span<const DrawStage> stages, const DrawShape& shape) {
    using Stage = ShaderRecompiler::ShaderStage;
    if (stages.empty() || shape.tessellation || shape.mesh || shape.rectList) return ShapeRefusal::Stages;
    for (const auto& stage : stages) {
        if (stage.program == nullptr || (stage.stage != Stage::Vertex && stage.stage != Stage::Fragment)) return ShapeRefusal::Stages;
    }
    if (shape.depthRemap || std::ranges::any_of(stages, [](const DrawStage& stage) { return stage.program->readsDepthRemapPush; })) return ShapeRefusal::FixedPushWords;
    if (shape.stencilReplay || std::ranges::any_of(stages, [](const DrawStage& stage) { return !stage.program->stencilReferences.empty(); })) return ShapeRefusal::StencilReplay;
    if (shape.blendedTargets) return ShapeRefusal::BlendedTargets;
    if (std::ranges::any_of(stages, [](const DrawStage& stage) { return stage.program->functionLdsBuffer; })) return ShapeRefusal::PixelLds;
    if (std::ranges::any_of(stages, [](const DrawStage& stage) { return stage.program->barycentricEmulation.active; })) return ShapeRefusal::BarycentricEmulation;
    if (shape.splitPipeline) return ShapeRefusal::SplitPipeline;
    return ShapeRefusal::None;
}

std::uint32_t DrawPushEnd(std::span<const DrawStage> stages) {
    std::uint32_t end = 0;
    for (const auto& stage : stages) {
        if (stage.program == nullptr || stage.program->pushConstants.empty()) continue;
        end = std::max(end, stage.pushOffset + static_cast<std::uint32_t>(stage.program->pushConstants.size()));
    }
    return end;
}

bool DrawPushFits(std::span<const DrawStage> stages) {
    return DrawPushEnd(stages) + DrawPrefixBytes <= Abi::PushBlockBytes;
}

std::optional<PushBytes> ComposeDrawPush(const std::array<std::byte, Abi::PushBlockBytes>& assembled, std::span<const DrawStage> stages, std::uint64_t tableRoot, std::uint64_t commandRecord) {
    static_assert(DrawPrefixBytes == ResolvedDriverWords * 4u);
    const auto end = DrawPushEnd(stages);
    if (end + DrawPrefixBytes > Abi::PushBlockBytes) return std::nullopt;
    return ComposeResolvedPush(assembled, (end + 3u) & ~3u, tableRoot, commandRecord);
}

namespace {

bool sameAttributes(const ShaderRecompiler::RecompileResult& left, const ShaderRecompiler::RecompileResult& right) {
    if (left.vertexAttributes.size() != right.vertexAttributes.size()) return false;
    for (std::size_t index = 0; index < left.vertexAttributes.size(); ++index) {
        const auto& a = left.vertexAttributes[index];
        const auto& b = right.vertexAttributes[index];
        if (a.location != b.location || a.components != b.components || a.fetchIndex != b.fetchIndex || a.formatComponents != b.formatComponents || a.resource.fields != b.resource.fields) return false;
    }
    return true;
}

}

std::string StageTwinDifference(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin) {
    if (auto difference = TwinDifference(compiled, twin); !difference.empty()) return difference;
    if (compiled.memoryOffsetDword != twin.memoryOffsetDword || compiled.runtimeImageCount != twin.runtimeImageCount || compiled.runtimeImageResources != twin.runtimeImageResources) return "push patch positions";
    if (compiled.vertexInputs != twin.vertexInputs || compiled.vertexInputPatches != twin.vertexInputPatches || !sameAttributes(compiled, twin)) return "vertex inputs";
    if (compiled.vertexOffsetSgpr != twin.vertexOffsetSgpr || compiled.instanceOffsetSgpr != twin.instanceOffsetSgpr || compiled.vertexOffsetShared != twin.vertexOffsetShared || compiled.instanceOffsetShared != twin.instanceOffsetShared) return "vertex offsets";
    if (compiled.parameterExports != twin.parameterExports || compiled.fragmentParameters != twin.fragmentParameters) return "parameters";
    if (compiled.barycentricEmulation != twin.barycentricEmulation || compiled.meshBarycentrics != twin.meshBarycentrics) return "barycentrics";
    if (compiled.stencilReferences != twin.stencilReferences || compiled.stencilReferenceDword != twin.stencilReferenceDword || compiled.stencilReplaySafe != twin.stencilReplaySafe) return "stencil references";
    if (compiled.readsDepthRemapPush != twin.readsDepthRemapPush || compiled.functionLdsBuffer != twin.functionLdsBuffer || compiled.fullSubgroups != twin.fullSubgroups) return "stage options";
    return {};
}

bool StageTwinCompatible(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin) {
    return StageTwinDifference(compiled, twin).empty();
}

std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> DrawTwinsOf(std::span<const DrawStage> stages, const ResolvedTwins& twins) {
    std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> found;
    found.reserve(stages.size());
    for (const auto& stage : stages) {
        if (stage.program == nullptr) return {};
        auto twin = twins.Of(*stage.program);
        if (twin == nullptr) return {};
        found.push_back(std::move(twin));
    }
    return found;
}

ResolvedTwins& DrawStageTwins() {
    static auto* twins = new ResolvedTwins;
    return *twins;
}

std::optional<CommandFacts> DrawFactsWithoutTable(const CommandFacts& decoded, std::span<const DrawStage> stages, const DrawShape& shape, const ResolvedTwins& twins) {
    auto facts = decoded;
    facts.classA = true;
    facts.lean = true;
    facts.shapeSupported = DrawShapeRefusal(stages, shape) == ShapeRefusal::None;
    if (!facts.shapeSupported) return facts;
    const auto found = DrawTwinsOf(stages, twins);
    if (found.empty()) return std::nullopt;
    for (const auto& twin : found) {
        if (!WalkReplacesTables(*twin)) return std::nullopt;
    }
    facts.resolvedProgram = true;
    facts.pushFits = DrawPushFits(stages);
    facts.clothGuard = false;
    facts.tableCovers = true;
    facts.pipelineReady = true;
    return facts;
}

namespace {

struct DrawCost {
    std::atomic<std::uint64_t> count{0}, drawNs{0}, buildNs{0}, slow{0};
};
std::array<DrawCost, DrawCostBuckets>& DrawCosts() {
    static auto* costs = new std::array<DrawCost, DrawCostBuckets>;
    return *costs;
}

}

bool DrawCostProfiled() {
    static const bool profiled = [] { const char* value = std::getenv("APS5_PROFILE_NEW_PATH"); return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0; }();
    return profiled;
}

void NoteDrawCost(DrawCostBucket bucket, std::uint64_t drawNs, std::uint64_t buildNs) {
    auto& cost = DrawCosts()[static_cast<std::size_t>(bucket)];
    if (drawNs > 50'000'000ull) {
        cost.slow.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    cost.count.fetch_add(1, std::memory_order_relaxed);
    cost.drawNs.fetch_add(drawNs, std::memory_order_relaxed);
    cost.buildNs.fetch_add(buildNs, std::memory_order_relaxed);
}

std::string FormatDrawCost(const char* window) {
    static constexpr const char* names[] = {"new path (table deferred)", "new path (table built)", "new path (template hit)", "fallback class A", "other draws"};
    static_assert(std::size(names) == DrawCostBuckets);
    std::string line = std::string("[newpath-cost] window=") + window + " committer draws (cumulative, Graphics::Draw):";
    for (std::size_t index = 0; index < DrawCostBuckets; ++index) {
        const auto& cost = DrawCosts()[index];
        const auto count = cost.count.load(std::memory_order_relaxed);
        char text[192];
        std::snprintf(text, sizeof(text), "%s %s %llu x %.1f us (build %.1f us), %llu over 50 ms left out", index == 0 ? "" : ";", names[index], static_cast<unsigned long long>(count), count != 0 ? static_cast<double>(cost.drawNs.load(std::memory_order_relaxed)) / 1000.0 / static_cast<double>(count) : 0.0, count != 0 ? static_cast<double>(cost.buildNs.load(std::memory_order_relaxed)) / 1000.0 / static_cast<double>(count) : 0.0, static_cast<unsigned long long>(cost.slow.load(std::memory_order_relaxed)));
        line += text;
    }
    return line;
}

DrawEmitCounts& DrawCounts() {
    static auto* counts = new DrawEmitCounts;
    return *counts;
}

std::string FormatDrawEmit(const char* window, bool emitting) {
    auto& counts = DrawCounts();
    const auto twins = DrawStageTwins().Outcomes();
    const auto load = [](const std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.load(std::memory_order_relaxed)); };
    std::string shapes;
    for (std::size_t index = 1; index < ShapeRefusalCount; ++index) {
        if (counts.shapes[index].load(std::memory_order_relaxed) == 0) continue;
        shapes += std::string(shapes.empty() ? "" : ", ") + ShapeRefusalName(static_cast<ShapeRefusal>(index)) + " " + std::to_string(load(counts.shapes[index]));
    }
    char facts[448];
    std::snprintf(facts, sizeof(facts), "; class A by fact: draws %llu, lean %llu, shape ok %llu, twins %llu, push fits %llu, no cloth %llu, F4 covered %llu, all but the twin %llu; deferred-table templates: hits %llu, fell back to an own build %llu", load(counts.classA), load(counts.lean), load(counts.shapeOk), load(counts.twinned), load(counts.pushFits), load(counts.noCloth), load(counts.covered), load(counts.eligible), load(counts.templateHits), load(counts.templateFallbacks));
    char line[640];
    std::snprintf(line, sizeof(line), "[newpath-draw-emit] window=%s emit draws %s (cumulative): stage twins built %llu, failed %llu, mismatched %llu, reused %llu; shape refusals %s; twin pipelines requested %llu, not ready at a draw %llu, failed %llu, rebuilt after eviction %llu, uncached %llu; table compares covered %llu, missing %llu, other address %llu, narrower %llu; command records %llu, table uploads %llu", window, emitting ? "on" : "off", static_cast<unsigned long long>(twins[static_cast<std::size_t>(TwinOutcome::Built)]), static_cast<unsigned long long>(twins[static_cast<std::size_t>(TwinOutcome::Failed)]), static_cast<unsigned long long>(twins[static_cast<std::size_t>(TwinOutcome::Mismatch)]), static_cast<unsigned long long>(twins[static_cast<std::size_t>(TwinOutcome::Reused)]), shapes.empty() ? "none" : shapes.c_str(), load(counts.pipelineRequests), load(counts.pipelineNotReady), load(counts.pipelineFailed), load(counts.pipelineRebuilt), load(counts.uncachedPipelines), load(counts.covers[0]), load(counts.covers[1]), load(counts.covers[2]), load(counts.covers[3]), load(counts.records), load(counts.tableUploads));
    return std::string(line) + facts;
}

}
