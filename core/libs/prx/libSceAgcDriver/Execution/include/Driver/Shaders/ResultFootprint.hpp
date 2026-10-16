#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SHADERS_RESULTFOOTPRINT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SHADERS_RESULTFOOTPRINT_HPP

#include "Recompiler.hpp"
#include <cstdint>
#include <type_traits>
#include <vector>

namespace AgcDriver::DriverDetail {

template<typename TValue, typename TAllocator>
std::uint64_t VectorHeapBytes(const std::vector<TValue, TAllocator>& values) {
    if constexpr (std::is_same_v<TValue, bool>) return (values.capacity() + 7u) / 8u;
    else return values.capacity() * sizeof(TValue);
}

inline std::uint64_t ResultHeapBytes(const ShaderRecompiler::RecompileResult& result) {
    std::uint64_t bytes = sizeof(result);
    bytes += VectorHeapBytes(result.vertexInputPatches) + VectorHeapBytes(result.runtimeImageResources) + VectorHeapBytes(result.vertexInputs);
    bytes += VectorHeapBytes(result.parameterExports) + VectorHeapBytes(result.fragmentParameters) + VectorHeapBytes(result.stencilReferences);
    bytes += VectorHeapBytes(result.specialization) + VectorHeapBytes(result.bindings) + VectorHeapBytes(result.pushConstants) + VectorHeapBytes(result.vertexAttributes);
    bytes += VectorHeapBytes(result.imageTablePoison) + VectorHeapBytes(result.imageTableRanges) + VectorHeapBytes(result.imageTableReadRanges) + VectorHeapBytes(result.bindlessTables);
    for (const auto& binding : result.bindings) {
        bytes += VectorHeapBytes(binding.guestDescriptor) + VectorHeapBytes(binding.samplerDepthCompare) + VectorHeapBytes(binding.imageWritten);
        bytes += VectorHeapBytes(binding.imageDepthCompare) + VectorHeapBytes(binding.imageAtomic) + VectorHeapBytes(binding.imageAtomic64) + VectorHeapBytes(binding.imageShapes);
        bytes += VectorHeapBytes(binding.bufferAtomic) + VectorHeapBytes(binding.bufferWritten) + VectorHeapBytes(binding.samplerUnnormalized) + VectorHeapBytes(binding.samplerIntegerBorder);
        bytes += VectorHeapBytes(binding.imageUnnormalized) + VectorHeapBytes(binding.imageUnnormalizedBaseLevel) + VectorHeapBytes(binding.imageSamplers) + VectorHeapBytes(binding.bufferRead);
    }
    for (const auto& ranges : result.imageTableReadRanges) bytes += VectorHeapBytes(ranges);
    for (const auto& table : result.bindlessTables) bytes += VectorHeapBytes(table.uses);
    return bytes;
}

}

#endif
