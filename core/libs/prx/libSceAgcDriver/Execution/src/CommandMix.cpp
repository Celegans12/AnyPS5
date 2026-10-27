#include "prx/libSceAgcDriver/Execution/include/CommandMix.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace AgcDriver::CommandMix {

namespace {

constexpr std::array<const char*, 7> DescriptorNames{"ub", "sb", "utb", "stb", "sampled", "storage-image", "sampler"};

std::size_t sgprBucket(std::uint32_t count) {
    if (count == 0) return 0;
    if (count <= 2) return 1;
    if (count <= 4) return 2;
    if (count <= 8) return 3;
    if (count <= 16) return 4;
    return 5;
}

// The flat SRT slot a value reads (ReadConst of GetSrtResource at an immediate slot), if it is one.
std::optional<std::uint32_t> slotRead(const ShaderRecompiler::IrValue* value) {
    value = value->Resolve();
    if (value == nullptr || value->Opcode() != ShaderRecompiler::IrOpcode::ReadConst || value->ArgumentCount() != 2u) return std::nullopt;
    const auto* resource = value->Argument(0) != nullptr ? value->Argument(0)->Resolve() : nullptr;
    const auto* slot = value->Argument(1) != nullptr ? value->Argument(1)->Resolve() : nullptr;
    if (resource == nullptr || slot == nullptr || resource->Opcode() != ShaderRecompiler::IrOpcode::GetSrtResource || !slot->HasImmediate()) return std::nullopt;
    return slot->ImmediateU32();
}

struct State {
    std::mutex mutex;
    Window window;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
    std::array<std::uint64_t, static_cast<std::size_t>(Graphics::ShaderResources::TemplateOutcome::Count)> templatesSeen{};
    bool templatesStarted = false;
    // SRT depth per materialized program (by address: the results live in the driver's caches) and
    // per plan.
    std::unordered_map<const void*, std::int32_t> depthByProgram;
    std::unordered_map<const void*, std::int32_t> depthByPlan;
};

State& Get() {
    static State state;
    return state;
}

std::string templateDeltas(State& state) {
    const auto outcomes = Graphics::ShaderResources::TemplateOutcomes();
    std::string text;
    if (!state.templatesStarted) {
        // The first window counts from the start of the process.
        state.templatesSeen.fill(0);
        state.templatesStarted = true;
    }
    for (std::size_t index = 0; index < outcomes.size(); ++index) {
        const auto count = outcomes[index] - state.templatesSeen[index];
        if (count == 0) continue;
        text += std::string(text.empty() ? " " : ", ") + Graphics::ShaderResources::TemplateOutcomeName(static_cast<Graphics::ShaderResources::TemplateOutcome>(index)) + " " + std::to_string(count);
    }
    state.templatesSeen = outcomes;
    return text.empty() ? std::string(" none") : text;
}

void printWindow(State& state, const char* window) {
    const auto line = state.window.Format(window, templateDeltas(state));
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
    state.window = Window{};
}

void maybeReport(State& state) {
    if (ProfileWindows::ClockDue(state.lastReport)) printWindow(state, "10 s");
}

}

bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_PROFILE_COMMAND_MIX");
        return value != nullptr && value[0] != '\0' && std::string_view(value) != "0";
    }();
    return enabled;
}

void Window::AddDraw(const DrawUse& draw) {
    ++draws;
    if (draw.indirect) {
        ++indirect;
        if (draw.multi) ++multi;
        if (draw.countIndirect) ++countIndirect;
    } else {
        ++(draw.indexed ? indexed : nonIndexed);
        if (draw.instances > 1) ++instanced;
    }
    if (draw.recipe) ++recipe;
}

void Window::AddDispatch(bool isIndirect, bool cacheHit) {
    ++dispatches;
    ++(isIndirect ? indirectDispatches : directDispatches);
    if (cacheHit) ++cacheHits;
}

void Window::AddStage(bool draw, const StageUse& stage) {
    auto& stages = draw ? drawStages : dispatchStages;
    ++stages.uses;
    ++stages.sgprBuckets[sgprBucket(stage.userSgprs)];
    stages.sgprSum += stage.userSgprs;
    stages.sgprMax = std::max(stages.sgprMax, stage.userSgprs);
    ++stages.depthBuckets[stage.srtDepth < 0 ? 5 : static_cast<std::size_t>(std::min(stage.srtDepth, 4))];
    for (std::size_t kind = 0; kind < stages.descriptors.size(); ++kind) stages.descriptors[kind] += stage.descriptors[kind];
    stages.bindlessTables += stage.bindlessTables;
    if (stage.bda) ++stages.bda;
}

void Window::formatStages(std::string& line, const Stages& stages) {
    char text[512];
    std::snprintf(text, sizeof(text), " %llu: user-sgprs 0:%llu 1-2:%llu 3-4:%llu 5-8:%llu 9-16:%llu 17+:%llu mean %.1f max %u; srt-depth 0:%llu 1:%llu 2:%llu 3:%llu 4+:%llu unknown:%llu; descriptors", static_cast<unsigned long long>(stages.uses),
                  static_cast<unsigned long long>(stages.sgprBuckets[0]), static_cast<unsigned long long>(stages.sgprBuckets[1]), static_cast<unsigned long long>(stages.sgprBuckets[2]), static_cast<unsigned long long>(stages.sgprBuckets[3]), static_cast<unsigned long long>(stages.sgprBuckets[4]), static_cast<unsigned long long>(stages.sgprBuckets[5]),
                  stages.uses != 0 ? static_cast<double>(stages.sgprSum) / static_cast<double>(stages.uses) : 0.0, stages.sgprMax,
                  static_cast<unsigned long long>(stages.depthBuckets[0]), static_cast<unsigned long long>(stages.depthBuckets[1]), static_cast<unsigned long long>(stages.depthBuckets[2]), static_cast<unsigned long long>(stages.depthBuckets[3]), static_cast<unsigned long long>(stages.depthBuckets[4]), static_cast<unsigned long long>(stages.depthBuckets[5]));
    line += text;
    for (std::size_t kind = 0; kind < stages.descriptors.size(); ++kind) line += " " + std::string(DescriptorNames[kind]) + " " + std::to_string(stages.descriptors[kind]);
    line += "; bindless-tables " + std::to_string(stages.bindlessTables) + "; bda " + std::to_string(stages.bda);
}

std::string Window::Format(const char* window, const std::string& templates) const {
    char text[512];
    std::snprintf(text, sizeof(text), "[command-mix] window=%s draws %llu (indexed %llu, non-indexed %llu, indirect %llu (multi %llu, count %llu), instanced %llu, recipe %llu); dispatches %llu (direct %llu, indirect %llu, cache hits %llu); draw stages", window,
                  static_cast<unsigned long long>(draws), static_cast<unsigned long long>(indexed), static_cast<unsigned long long>(nonIndexed), static_cast<unsigned long long>(indirect), static_cast<unsigned long long>(multi), static_cast<unsigned long long>(countIndirect), static_cast<unsigned long long>(instanced), static_cast<unsigned long long>(recipe),
                  static_cast<unsigned long long>(dispatches), static_cast<unsigned long long>(directDispatches), static_cast<unsigned long long>(indirectDispatches), static_cast<unsigned long long>(cacheHits));
    std::string line = text;
    formatStages(line, drawStages);
    line += "; dispatch programs";
    formatStages(line, dispatchStages);
    line += "; bda-template builds:" + templates;
    return line;
}

std::int32_t SrtDepth(const ShaderRecompiler::IrResourcePlan& plan) {
    const auto slots = plan.srtReads.size();
    // -1 not yet known, -2 being computed (a cycle counts as no deeper read).
    std::vector<std::int32_t> depth(slots, -1);
    const auto below = [&](const ShaderRecompiler::IrValue* root, auto& self) -> std::int32_t {
        // The deepest slot reached from root's operands, stopping at slot reads.
        std::int32_t deepest = 0;
        std::vector<const ShaderRecompiler::IrValue*> stack;
        std::unordered_set<const ShaderRecompiler::IrValue*> seen;
        for (std::size_t index = 0; index < root->ArgumentCount(); ++index) {
            if (root->Argument(index) != nullptr) stack.push_back(root->Argument(index));
        }
        while (!stack.empty()) {
            const auto* value = stack.back();
            stack.pop_back();
            if (value == nullptr || !seen.insert(value).second) continue;
            if (const auto slot = slotRead(value)) {
                if (*slot < slots) deepest = std::max(deepest, self(*slot, self));
                continue;
            }
            const auto* resolved = value->Resolve();
            if (resolved != nullptr && resolved != value) {
                stack.push_back(resolved);
                continue;
            }
            for (std::size_t index = 0; index < value->ArgumentCount(); ++index) stack.push_back(value->Argument(index));
        }
        return deepest;
    };
    const auto slotDepth = [&](std::uint32_t slot, auto& self) -> std::int32_t {
        if (depth[slot] >= 0) return depth[slot];
        if (depth[slot] == -2) return 0;
        depth[slot] = -2;
        const auto* value = plan.srtReads[slot].value;
        const auto result = 1 + (value != nullptr ? below(value, self) : 0);
        depth[slot] = result;
        return result;
    };
    std::int32_t result = 0;
    for (std::uint32_t slot = 0; slot < slots; ++slot) result = std::max(result, slotDepth(slot, slotDepth));
    return result;
}

StageUse Classify(const ShaderRecompiler::RecompileResult& program) {
    StageUse use;
    use.userSgprs = static_cast<std::uint32_t>(std::popcount(program.userDataRegisterReads));
    for (const auto& binding : program.bindings) {
        const auto kind = static_cast<std::size_t>(binding.kind);
        if (kind < use.descriptors.size()) use.descriptors[kind] += std::max<std::uint32_t>(1u, binding.count);
    }
    use.bindlessTables = static_cast<std::uint32_t>(program.bindlessTables.size());
    use.bda = program.bdaAbiVersion != 0;
    return use;
}

void NoteMaterialized(const ShaderRecompiler::RecompileResult& program, const ShaderRecompiler::ResourceCapture& capture) {
    if (!Enabled() || capture.plan == nullptr) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    auto found = state.depthByPlan.find(capture.plan.get());
    if (found == state.depthByPlan.end()) {
        std::int32_t depth = -1;
        try {
            depth = SrtDepth(*capture.plan);
        } catch (const std::exception&) {
        }
        found = state.depthByPlan.emplace(capture.plan.get(), depth).first;
    }
    state.depthByProgram[&program] = found->second;
}

void NoteDraw(const Pm4::DrawParameters& draw, std::span<const Graphics::CompiledShader> stages, bool recipe) {
    if (!Enabled()) return;
    DrawUse use;
    use.indexed = draw.indexed;
    use.indirect = draw.indirect.has_value();
    if (use.indirect) {
        use.countIndirect = draw.indirect->countIndirect;
        use.multi = draw.indirect->count > 1 || use.countIndirect;
    }
    use.instances = draw.instanceCount;
    use.recipe = recipe;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    state.window.AddDraw(use);
    for (const auto& stage : stages) {
        if (stage.program == nullptr) continue;
        auto shape = Classify(*stage.program);
        if (const auto found = state.depthByProgram.find(stage.program); found != state.depthByProgram.end()) shape.srtDepth = found->second;
        state.window.AddStage(true, shape);
    }
    maybeReport(state);
}

void NoteDispatch(const ShaderRecompiler::RecompileResult& program, bool indirect, bool cacheHit) {
    if (!Enabled()) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    state.window.AddDispatch(indirect, cacheHit);
    auto shape = Classify(program);
    if (const auto found = state.depthByProgram.find(&program); found != state.depthByProgram.end()) shape.srtDepth = found->second;
    state.window.AddStage(false, shape);
    maybeReport(state);
}

void Close(const char* window) {
    if (!Enabled()) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    printWindow(state, window);
}

}
