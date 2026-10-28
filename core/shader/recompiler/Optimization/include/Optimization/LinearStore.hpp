#ifndef CORE_SHADER_RECOMPILER_OPTIMIZATION_INCLUDE_OPTIMIZATION_LINEARSTORE_HPP
#define CORE_SHADER_RECOMPILER_OPTIMIZATION_INCLUDE_OPTIMIZATION_LINEARSTORE_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include <array>
#include <cstdint>
#include <optional>

namespace ShaderRecompiler {

[[nodiscard]] std::optional<LinearStoreSite> FindLinearStore(const IrProgram& program, const std::array<std::uint32_t, 3>& threads);

}

#endif
