#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DIAGNOSTICS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DIAGNOSTICS_HPP

#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ShaderRecompiler { struct RecompileRequest; struct MemoryRegion; }

namespace AgcDriver::DriverDetail {

double TraceMs();

void require(bool condition, const char* reason);

std::uint32_t readRegister(const Registers& registers, std::uint32_t offset);

std::uint32_t readUserData(const Registers& shader, std::uint32_t offset);

// A program the recompiler refused when its draw or dispatch used it. With APS5_SKIP_REFUSED_SHADERS
// the queue worker skips that packet and reports the refusal once; otherwise it fails the queue.
class ShaderRefusal final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

bool SkipRefusedShaders();

bool CaptureTerminalComputeFailure(const ShaderRecompiler::RecompileRequest& request, std::span<const ShaderRecompiler::MemoryRegion> captured, const char* error, std::uint32_t queue, std::uint64_t serial, std::size_t packetOffset, const char* directory = ".") noexcept;

}

#endif
