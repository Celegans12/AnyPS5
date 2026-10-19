#ifndef PRX_LIBSCEAGCDRIVER_EXECUTION_DRIVER_SHADERS_PIPELINEFINGERPRINT_HPP
#define PRX_LIBSCEAGCDRIVER_EXECUTION_DRIVER_SHADERS_PIPELINEFINGERPRINT_HPP

#include "Recompiler.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver {

// What decides a compute pipeline's create info apart from the device, hashed (FNV-1a) as a whole and
// per component: the SPIR-V words, the specialization constants, the host subgroup size, the
// workgroup memory set, the push range and the descriptor set layout the dispatch path plans for the
// bindings (VulkanDevice::PlannedComputeLayoutKey: image tables at their device capacity, not at a
// dispatch's element count). A pipeline request the driver
// saves (APS5_PIPELINE_REQUESTS) carries the game's values in its .fp sidecar; agc_shader_replay
// --pipeline-only recomputes them from its own recompile and refuses a difference, naming the
// components that differ, so a precompiled pipeline is the one the game will look up in its
// VkPipelineCache.
struct ComputePipelineComponents {
    std::uint64_t whole = 0;
    std::uint64_t spirv = 0;
    std::uint64_t specialization = 0;
    std::uint64_t subgroup = 0;
    std::uint64_t workgroupMemory = 0;
    std::uint64_t push = 0;
    std::uint64_t bindings = 0;
};

namespace PipelineFingerprintDetail {
struct Fnv {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    void Mix(std::uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (value >> (byte * 8)) & 0xffu;
            hash *= 0x100000001b3ull;
        }
    }
};
}

inline ComputePipelineComponents ComputePipelineFingerprintComponents(const ShaderRecompiler::RecompileResult& shader, std::span<const std::uint32_t> plannedLayout) {
    using PipelineFingerprintDetail::Fnv;
    Fnv whole;
    Fnv part;
    ComputePipelineComponents components;
    const auto both = [&](std::uint64_t value) {
        whole.Mix(value);
        part.Mix(value);
    };
    const auto close = [&](std::uint64_t& out) {
        out = part.hash;
        part = Fnv{};
    };
    const auto words = shader.spirv.Words();
    both(words.size());
    for (const auto word : words) both(word);
    close(components.spirv);
    both(shader.specialization.size());
    for (const auto& constant : shader.specialization) both((static_cast<std::uint64_t>(constant.id) << 32u) | constant.value);
    close(components.specialization);
    both(shader.hostSubgroupSize);
    close(components.subgroup);
    both(shader.workgroupMemoryDwords);
    close(components.workgroupMemory);
    both(shader.pushConstants.empty() ? 0u : 1u);
    close(components.push);
    both(plannedLayout.size());
    for (const auto word : plannedLayout) both(word);
    close(components.bindings);
    components.whole = whole.hash;
    return components;
}

inline std::uint64_t ComputePipelineFingerprint(const ShaderRecompiler::RecompileResult& shader, std::span<const std::uint32_t> plannedLayout) {
    return ComputePipelineFingerprintComponents(shader, plannedLayout).whole;
}

// Everything a compute result's fingerprint is a function of on one device, kept by the pipeline-request
// capture (APS5_PIPELINE_REQUESTS) so a later materialization with the same inputs is known to be saved
// already without planning its layout and hashing its SPIR-V again (each costs a pass over up to 1.3M words).
// - The SPIR-V is compared by storage. The kept copy holds that storage, so it cannot be freed and reused at
//   the same address while the entry lives, and a shared module is copied before it is written. Equal storage
//   means equal words; another storage with equal words only misses here, and the fingerprint decides.
// - The planned layout (VulkanDevice::planComputeLayout) is a function of the device and of each binding's
//   set, number, kind, role, count and read-only flag, so those are compared instead of the layout.
// - The specialization, the host subgroup size, the workgroup memory and the push flag are compared as the
//   fingerprint mixes them.
struct ComputePipelineInputs {
    struct Binding {
        std::uint32_t set = 0;
        std::uint32_t binding = 0;
        std::uint32_t kind = 0;
        std::uint32_t role = 0;
        std::uint32_t count = 0;
        bool readOnly = false;
        bool operator==(const Binding&) const = default;
    };
    ShaderRecompiler::SharedSpirv spirv;
    std::vector<ShaderRecompiler::PipelineSpecializationConstant> specialization;
    std::uint32_t hostSubgroupSize = 0;
    std::uint32_t workgroupMemoryDwords = 0;
    bool push = false;
    std::vector<Binding> bindings;

    static Binding BindingOf(const ShaderRecompiler::DescriptorBinding& binding) {
        return {binding.descriptorSet, binding.binding, static_cast<std::uint32_t>(binding.kind), static_cast<std::uint32_t>(binding.role), binding.count, binding.readOnly};
    }
    static ComputePipelineInputs Of(const ShaderRecompiler::RecompileResult& shader) {
        ComputePipelineInputs inputs;
        inputs.spirv = shader.spirv;
        inputs.specialization = shader.specialization;
        inputs.hostSubgroupSize = shader.hostSubgroupSize;
        inputs.workgroupMemoryDwords = shader.workgroupMemoryDwords;
        inputs.push = !shader.pushConstants.empty();
        inputs.bindings.reserve(shader.bindings.size());
        for (const auto& binding : shader.bindings) inputs.bindings.push_back(BindingOf(binding));
        return inputs;
    }
    [[nodiscard]] bool Matches(const ShaderRecompiler::RecompileResult& shader) const {
        if (spirv.data() != shader.spirv.data() || spirv.size() != shader.spirv.size() || specialization != shader.specialization) return false;
        if (hostSubgroupSize != shader.hostSubgroupSize || workgroupMemoryDwords != shader.workgroupMemoryDwords || push != !shader.pushConstants.empty()) return false;
        if (bindings.size() != shader.bindings.size()) return false;
        for (std::size_t index = 0; index < bindings.size(); ++index) {
            if (!(bindings[index] == BindingOf(shader.bindings[index]))) return false;
        }
        return true;
    }
};

// The .fp sidecar body after the fingerprint line: one "<component> <hash>" line per component, then
// the values themselves ("spec <id> <value>", "binding <set> <binding> <kind> <role> <count> <readOnly>",
// "layout <binding> <type> <count> <stages> [<flags>]...", "subgroup", "workgroup_memory", "push",
// "spirv_words", "specialization_id") for a diff.
inline std::string ComputePipelineDescription(const ShaderRecompiler::RecompileResult& shader, const ComputePipelineComponents& components, std::span<const std::uint32_t> plannedLayout) {
    std::string text;
    char line[160];
    const auto add = [&](const char* name, std::uint64_t value) {
        std::snprintf(line, sizeof(line), "%s %016llx\n", name, static_cast<unsigned long long>(value));
        text += line;
    };
    add("hash_spirv", components.spirv);
    add("hash_specialization", components.specialization);
    add("hash_subgroup", components.subgroup);
    add("hash_workgroup_memory", components.workgroupMemory);
    add("hash_push", components.push);
    add("hash_bindings", components.bindings);
    std::snprintf(line, sizeof(line), "spirv_words %zu\nsubgroup %u\nworkgroup_memory %u\npush %u\nspecialization_id %016llx\nvariant_id %016llx\n", shader.spirv.size(), shader.hostSubgroupSize, shader.workgroupMemoryDwords, shader.pushConstants.empty() ? 0u : 1u,
                  static_cast<unsigned long long>(shader.specializationId), static_cast<unsigned long long>(shader.variantId));
    text += line;
    for (const auto& constant : shader.specialization) {
        std::snprintf(line, sizeof(line), "spec %u %08x\n", constant.id, constant.value);
        text += line;
    }
    for (const auto& binding : shader.bindings) {
        std::snprintf(line, sizeof(line), "binding %u %u %u %u %u %u\n", binding.descriptorSet, binding.binding, static_cast<unsigned>(binding.kind), static_cast<unsigned>(binding.role), binding.count, binding.readOnly ? 1u : 0u);
        text += line;
    }
    text += "layout";
    for (const auto word : plannedLayout) text += " " + std::to_string(word);
    text += "\n";
    return text;
}

}

#endif
