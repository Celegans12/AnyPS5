#ifndef CORE_SHADER_RECOMPILER_UNNORMALIZEDWRAP_HPP
#define CORE_SHADER_RECOMPILER_UNNORMALIZEDWRAP_HPP

#include "IntermediateRepresentation/IrMetadata/Resources.hpp"
#include <cstdlib>
#include <span>
#include <string_view>

namespace ShaderRecompiler {

inline bool UnnormalizedWrapEnabled() {
    const auto* value = std::getenv("APS5_UNNORM_WRAP_GUARD");
    return value != nullptr && std::string_view(value) == "1";
}

inline bool UnnormalizedWrapCandidate(const SamplerResource& sampler, IrShaderStage stage) {
    return stage == IrShaderStage::Compute && sampler.table == NoTable && !sampler.depthCompare
        && sampler.uses != 0u && (sampler.uses & ~static_cast<std::uint32_t>(SamplerUseExplicitLod | SamplerUseOffset)) == 0u;
}

inline std::uint32_t UnnormalizedWrapAxes(std::span<const std::uint32_t> words) {
    if (words.size() != 4u || (words[0] & (1u << 15u)) == 0u || ((words[2] >> 20u) & 15u) != 0u) return 0u;
    return ((words[0] & 7u) == 0u ? 1u : 0u) | (((words[0] >> 3u) & 7u) == 0u ? 2u : 0u);
}

}

#endif
