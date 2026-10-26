#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DIAGNOSTICS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DIAGNOSTICS_HPP

#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ShaderRecompiler { struct RecompileRequest; struct MemoryRegion; }

namespace AgcDriver::DriverDetail {

double TraceMs();

bool SkippedShader(std::uint64_t address);

class RefusedDraw final : public std::runtime_error {
public:
    RefusedDraw(std::uint64_t address, const std::string& what) : std::runtime_error(what), address(address) {}
    std::uint64_t address;
};

bool SkipRefusedDraws();

void ReportRefusedDraw(const RefusedDraw& refused);

void require(bool condition, const char* reason);

std::uint32_t readRegister(const Registers& registers, std::uint32_t offset);

std::uint32_t readUserData(const Registers& shader, std::uint32_t offset);

// A program the recompiler refused when its draw or dispatch used it. With APS5_SKIP_REFUSED_SHADERS
// the queue worker skips that packet and reports the refusal once; otherwise it fails the queue.
// A refusal while preparing a program (subject names it, e.g. "domain shader 0x...") is one too:
// ABI resolution on the title's thread skips that preparation and the program's draws are skipped.
class ShaderRefusal final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    ShaderRefusal(const std::string& what, std::string subject) : std::runtime_error(what), subject(std::move(subject)) {}
    std::string subject;
};

bool SkipRefusedShaders();
bool SkipListedShader(std::uint64_t address);
bool SkipRefusedGraphicsRecompile(std::uint32_t stage, const char* what);

// "<stage> shader 0x<address>" for refusal reports; stage is a ShaderRecompiler::ShaderStage.
std::string RefusedShaderSubject(std::uint32_t stage, std::uint64_t address);

bool CaptureTerminalComputeFailure(const ShaderRecompiler::RecompileRequest& request, std::span<const ShaderRecompiler::MemoryRegion> captured, const char* error, std::uint32_t queue, std::uint64_t serial, std::size_t packetOffset, const char* directory = ".") noexcept;

}

#endif
