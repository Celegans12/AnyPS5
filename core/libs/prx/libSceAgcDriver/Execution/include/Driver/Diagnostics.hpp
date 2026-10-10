#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DIAGNOSTICS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DIAGNOSTICS_HPP

#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <cstddef>
#include <cstdint>
#include <span>

namespace ShaderRecompiler { struct RecompileRequest; struct MemoryRegion; }

namespace AgcDriver::DriverDetail {

double TraceMs();

void require(bool condition, const char* reason);

std::uint32_t readRegister(const Registers& registers, std::uint32_t offset);

std::uint32_t readUserData(const Registers& shader, std::uint32_t offset);

bool CaptureTerminalComputeFailure(const ShaderRecompiler::RecompileRequest& request, std::span<const ShaderRecompiler::MemoryRegion> captured, const char* error, std::uint32_t queue, std::uint64_t serial, std::size_t packetOffset, const char* directory = ".") noexcept;

}

#endif
