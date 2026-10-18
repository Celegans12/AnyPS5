#ifndef CORE_SHADER_RECOMPILER_ENVIRONMENTGUARDS_HPP
#define CORE_SHADER_RECOMPILER_ENVIRONMENTGUARDS_HPP

#include <cstdint>

namespace ShaderRecompiler {

enum class EnvironmentGuard : std::uint8_t { LoopGuard, UnnormalizedWrapGuard, Count };

[[nodiscard]] bool EnvironmentGuardEnabled(EnvironmentGuard guard);
void RefreshEnvironmentGuards();

}

#endif
