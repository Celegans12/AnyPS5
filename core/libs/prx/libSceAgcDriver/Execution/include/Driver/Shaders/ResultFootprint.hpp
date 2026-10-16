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

}

#endif
