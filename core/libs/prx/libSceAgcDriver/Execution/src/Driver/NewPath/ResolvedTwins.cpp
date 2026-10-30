#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ResolvedTwins.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PushBlock.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ResolvedPush.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace AgcDriver::NewPath {

struct ResolvedTwins::State {
    struct Entry {
        std::weak_ptr<const ShaderRecompiler::RecompileResult> compiled;
        std::shared_ptr<const ShaderRecompiler::RecompileResult> twin;
    };
    mutable std::mutex mutex;
    std::unordered_map<const ShaderRecompiler::RecompileResult*, Entry> entries;
    std::array<std::uint64_t, TwinOutcomeCount> outcomes{};
    std::array<std::uint64_t, TwinOutcomeCount> c1Outcomes{};
    std::uint64_t c1Runtime = 0;
    std::size_t prunedAt = 64;
};

ResolvedTwins::ResolvedTwins() : state(std::make_shared<State>()) {}

bool UsesAddressTables(const ShaderRecompiler::RecompileResult& compiled) {
    return std::ranges::any_of(compiled.bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; });
}

bool C1Shape(const ShaderRecompiler::RecompileResult& compiled) {
    using Role = ShaderRecompiler::DescriptorRole;
    if (compiled.newPathResolved || compiled.transferKernel.has_value() || compiled.bindlessHeap || !compiled.bindlessTables.empty()) return false;
    bool buffers = false;
    for (const auto& binding : compiled.bindings) {
        switch (binding.role) {
        case Role::GuestBuffers:
            if (binding.kind == ShaderRecompiler::DescriptorKind::UniformTexelBuffer || binding.kind == ShaderRecompiler::DescriptorKind::StorageTexelBuffer) return false;
            buffers = true;
            break;
        case Role::FlattenedSrt:
        case Role::ShaderData:
        case Role::FaultBuffer: break;
        default: return false;
        }
    }
    return buffers;
}

namespace {

bool Binds(const ShaderRecompiler::RecompileResult& result, ShaderRecompiler::DescriptorRole role) {
    return std::ranges::any_of(result.bindings, [role](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == role; });
}

}

bool RuntimeBufferTwin(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin) {
    return twin.newPathResolved && C1Shape(compiled) && !Binds(twin, ShaderRecompiler::DescriptorRole::GuestBuffers);
}

bool WalkReplacesTables(const ShaderRecompiler::RecompileResult& twin) {
    return twin.newPathResolved && std::ranges::none_of(twin.bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable || binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer; });
}

std::string TwinDifference(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin) {
    if (!twin.newPathResolved || compiled.newPathResolved) return "resolved marker";
    const bool runtimeBuffers = RuntimeBufferTwin(compiled, twin);
    std::size_t at = 0;
    for (std::size_t index = 0; index < compiled.bindings.size(); ++index) {
        const auto& left = compiled.bindings[index];
        const bool walkReplaces = left.role == ShaderRecompiler::DescriptorRole::BdaPagetable || left.role == ShaderRecompiler::DescriptorRole::FaultBuffer || (runtimeBuffers && left.role == ShaderRecompiler::DescriptorRole::GuestBuffers);
        if (at == twin.bindings.size() || twin.bindings[at].role != left.role || twin.bindings[at].binding != left.binding) {
            if (walkReplaces) continue;
            return "binding " + std::to_string(index) + " missing from the resolved module";
        }
        const auto& right = twin.bindings[at++];
        if (left.kind != right.kind || left.descriptorSet != right.descriptorSet || left.count != right.count || left.readOnly != right.readOnly) return "binding " + std::to_string(index) + " shape";
        if (left.guestDescriptor != right.guestDescriptor) return "binding " + std::to_string(index) + " guest descriptor words";
    }
    if (at != twin.bindings.size()) return "the resolved module binds more than today's (or renumbers a binding)";
    if (runtimeBuffers) {
        // The twin's own push words go to the device (no memory-offset dwords); its ShaderData, if it binds one, is
        // read from today's buffer, so its layout must be today's.
        if (twin.pushConstants.size() > LegacyPushBytes || twin.pushConstants.size() % 4u != 0u) return "push words of the runtime-buffer twin (" + std::to_string(twin.pushConstants.size()) + " bytes)";
        if (Binds(twin, ShaderRecompiler::DescriptorRole::ShaderData) && (compiled.memoryOffsetDword != twin.memoryOffsetDword || compiled.shaderDataDwords != twin.shaderDataDwords || compiled.imageMetadataDword != twin.imageMetadataDword || compiled.shaderDataUserWords != twin.shaderDataUserWords)) return "shader data layout of the runtime-buffer twin";
    } else if (compiled.pushConstants != twin.pushConstants) return "push words (" + std::to_string(compiled.pushConstants.size()) + " vs " + std::to_string(twin.pushConstants.size()) + " bytes)";
    if (compiled.specialization != twin.specialization) return "specialization";
    if (compiled.workgroupMemoryDwords != twin.workgroupMemoryDwords) return "workgroup memory";
    if (compiled.bindlessHeap != twin.bindlessHeap) return "bindless heap";
    if (compiled.hostSubgroupSize != twin.hostSubgroupSize) return "subgroup size";
    if (!runtimeBuffers && (compiled.imageMetadataDword != twin.imageMetadataDword || compiled.shaderDataDwords != twin.shaderDataDwords)) return "shader data";
    return {};
}

bool TwinCompatible(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin) {
    return TwinDifference(compiled, twin).empty();
}

void ResolvedTwins::Remember(const std::shared_ptr<const ShaderRecompiler::RecompileResult>& compiled, std::shared_ptr<const ShaderRecompiler::RecompileResult> twin, TwinOutcome outcome, bool c1) {
    std::lock_guard lock(state->mutex);
    ++state->outcomes[static_cast<std::size_t>(outcome)];
    if (c1) {
        ++state->c1Outcomes[static_cast<std::size_t>(outcome)];
        if (outcome == TwinOutcome::Built && compiled != nullptr && twin != nullptr && RuntimeBufferTwin(*compiled, *twin)) ++state->c1Runtime;
    }
    if (compiled == nullptr) return;
    if (state->entries.size() >= state->prunedAt) {
        std::erase_if(state->entries, [](const auto& item) { return item.second.compiled.expired(); });
        state->prunedAt = std::max<std::size_t>(64, state->entries.size() * 2);
    }
    state->entries[compiled.get()] = {compiled, outcome == TwinOutcome::Built ? std::move(twin) : nullptr};
}

std::shared_ptr<const ShaderRecompiler::RecompileResult> ResolvedTwins::Of(const ShaderRecompiler::RecompileResult& compiled) const {
    std::lock_guard lock(state->mutex);
    const auto found = state->entries.find(&compiled);
    if (found == state->entries.end()) return nullptr;
    const auto alive = found->second.compiled.lock();
    if (alive.get() != &compiled) return nullptr;
    return found->second.twin;
}

bool ResolvedTwins::Known(const ShaderRecompiler::RecompileResult& compiled) const {
    std::lock_guard lock(state->mutex);
    const auto found = state->entries.find(&compiled);
    if (found == state->entries.end()) return false;
    const auto alive = found->second.compiled.lock();
    if (alive.get() != &compiled) return false;
    ++state->outcomes[static_cast<std::size_t>(TwinOutcome::Reused)];
    return true;
}

std::array<std::uint64_t, TwinOutcomeCount> ResolvedTwins::Outcomes() const {
    std::lock_guard lock(state->mutex);
    return state->outcomes;
}

std::array<std::uint64_t, TwinOutcomeCount> ResolvedTwins::C1Outcomes() const {
    std::lock_guard lock(state->mutex);
    return state->c1Outcomes;
}

std::uint64_t ResolvedTwins::C1RuntimeBuilt() const {
    std::lock_guard lock(state->mutex);
    return state->c1Runtime;
}

std::size_t ResolvedTwins::Size() const {
    std::lock_guard lock(state->mutex);
    return state->entries.size();
}

ResolvedTwins& Twins() {
    static auto* twins = new ResolvedTwins;
    return *twins;
}

}
