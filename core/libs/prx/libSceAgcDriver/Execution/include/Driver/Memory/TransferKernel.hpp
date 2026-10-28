#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_MEMORY_TRANSFERKERNEL_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_MEMORY_TRANSFERKERNEL_HPP

#include "Recompiler.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace AgcDriver::DriverDetail {

struct TransferRange {
    bool copy = false;
    std::uint64_t destination = 0;
    std::uint64_t source = 0;
    std::size_t bytes = 0;
    std::array<std::uint32_t, 4> pattern{};
    std::uint32_t components = 0;
    std::array<std::uint64_t, 4> reads{};
    std::uint32_t readMask = 0;
};

bool FinishPattern(TransferRange& range);
std::optional<TransferRange> ResolveTransfer(const ShaderRecompiler::TransferKernel& transfer, const ShaderRecompiler::LinearStore& destination, std::span<const std::uint32_t> userData, std::uint64_t threads);

}

#endif
