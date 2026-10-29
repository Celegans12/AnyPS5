#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ResolvedTwins.hpp"
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
    std::size_t prunedAt = 64;
};

ResolvedTwins::ResolvedTwins() : state(std::make_shared<State>()) {}

bool UsesAddressTables(const ShaderRecompiler::RecompileResult& compiled) {
    return std::ranges::any_of(compiled.bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; });
}

bool TwinCompatible(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin) {
    if (!twin.newPathResolved || compiled.newPathResolved) return false;
    if (compiled.bindings.size() != twin.bindings.size()) return false;
    for (std::size_t index = 0; index < compiled.bindings.size(); ++index) {
        const auto& left = compiled.bindings[index];
        const auto& right = twin.bindings[index];
        if (left.kind != right.kind || left.role != right.role || left.descriptorSet != right.descriptorSet || left.binding != right.binding || left.count != right.count || left.readOnly != right.readOnly || left.guestDescriptor != right.guestDescriptor) return false;
    }
    return compiled.pushConstants == twin.pushConstants && compiled.specialization == twin.specialization && compiled.workgroupMemoryDwords == twin.workgroupMemoryDwords && compiled.bindlessHeap == twin.bindlessHeap && compiled.hostSubgroupSize == twin.hostSubgroupSize && compiled.imageMetadataDword == twin.imageMetadataDword && compiled.shaderDataDwords == twin.shaderDataDwords && compiled.bdaAbiVersion == twin.bdaAbiVersion;
}

void ResolvedTwins::Remember(const std::shared_ptr<const ShaderRecompiler::RecompileResult>& compiled, std::shared_ptr<const ShaderRecompiler::RecompileResult> twin, TwinOutcome outcome) {
    std::lock_guard lock(state->mutex);
    ++state->outcomes[static_cast<std::size_t>(outcome)];
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

std::array<std::uint64_t, TwinOutcomeCount> ResolvedTwins::Outcomes() const {
    std::lock_guard lock(state->mutex);
    return state->outcomes;
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
