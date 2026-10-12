#ifndef PRX_LIBSCEAGCDRIVER_EXECUTION_DRIVER_SHADERS_PIPELINEFINGERPRINT_HPP
#define PRX_LIBSCEAGCDRIVER_EXECUTION_DRIVER_SHADERS_PIPELINEFINGERPRINT_HPP

#include "Recompiler.hpp"

#include <cstddef>
#include <cstdint>

namespace AgcDriver {

// FNV-1a over what decides a compute pipeline's create info apart from the device: the SPIR-V words,
// the specialization constants, the host subgroup size, the workgroup memory set, the push range and
// the descriptor bindings. A pipeline request the driver saves (APS5_PIPELINE_REQUESTS) carries the
// game's value; agc_shader_replay --pipeline-only recomputes it from its own recompile and refuses a
// difference, so a precompiled pipeline is the one the game will look up in its VkPipelineCache.
inline std::uint64_t ComputePipelineFingerprint(const ShaderRecompiler::RecompileResult& shader) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (value >> (byte * 8)) & 0xffu;
            hash *= 0x100000001b3ull;
        }
    };
    const auto words = shader.spirv.ReadTransient();
    mix(words ? words->size() : 0u);
    if (words) {
        for (const auto word : *words) mix(word);
    }
    mix(shader.specialization.size());
    for (const auto& constant : shader.specialization) mix((static_cast<std::uint64_t>(constant.id) << 32u) | constant.value);
    mix(shader.hostSubgroupSize);
    mix(shader.workgroupMemoryDwords);
    mix(shader.pushConstants.empty() ? 0u : 1u);
    mix(shader.bindings.size());
    for (const auto& binding : shader.bindings) {
        mix((static_cast<std::uint64_t>(binding.kind) << 32u) | static_cast<std::uint64_t>(binding.role));
        mix((static_cast<std::uint64_t>(binding.descriptorSet) << 32u) | binding.binding);
        mix((static_cast<std::uint64_t>(binding.count) << 1u) | (binding.readOnly ? 1u : 0u));
    }
    return hash;
}

}

#endif
