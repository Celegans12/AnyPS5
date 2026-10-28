#ifndef CORE_SHADER_RECOMPILER_OPTIMIZATION_INCLUDE_OPTIMIZATION_TRANSFERKERNEL_HPP
#define CORE_SHADER_RECOMPILER_OPTIMIZATION_INCLUDE_OPTIMIZATION_TRANSFERKERNEL_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include <optional>

namespace ShaderRecompiler {

[[nodiscard]] std::optional<TransferKernelSite> FindTransferKernel(const IrProgram& program);

}

#endif
