#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_RESOLVEDTWINS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_RESOLVEDTWINS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ShaderRecompiler {
struct RecompileResult;
}

namespace AgcDriver::NewPath {

enum class TwinOutcome : std::uint8_t { Built, NotAddressBased, Failed, Mismatch, Reused };
inline constexpr std::size_t TwinOutcomeCount = 5;

[[nodiscard]] bool UsesAddressTables(const ShaderRecompiler::RecompileResult& compiled);
// S2-share C1: a program that binds guest buffers and no other guest resource (no BDA table, images, samplers,
// bindless tables, GDS or texel buffers). Its twin is requested with SpirvTarget::newPathRuntimeBuffers.
[[nodiscard]] bool C1Shape(const ShaderRecompiler::RecompileResult& compiled);
// A resolved twin that reads the buffer V#s at run time: today's build binds guest buffers, the twin binds none.
// It pushes its own words (no per-buffer memory offsets), so TwinDifference compares its layout instead.
[[nodiscard]] bool RuntimeBufferTwin(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin);
[[nodiscard]] bool WalkReplacesTables(const ShaderRecompiler::RecompileResult& twin);
[[nodiscard]] std::string TwinDifference(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin);
[[nodiscard]] bool TwinCompatible(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin);

class ResolvedTwins {
public:
    ResolvedTwins();
    void Remember(const std::shared_ptr<const ShaderRecompiler::RecompileResult>& compiled, std::shared_ptr<const ShaderRecompiler::RecompileResult> twin, TwinOutcome outcome, bool c1 = false);
    [[nodiscard]] std::shared_ptr<const ShaderRecompiler::RecompileResult> Of(const ShaderRecompiler::RecompileResult& compiled) const;
    [[nodiscard]] bool Known(const ShaderRecompiler::RecompileResult& compiled) const;
    [[nodiscard]] std::array<std::uint64_t, TwinOutcomeCount> Outcomes() const;
    // The outcomes of C1 twin requests (Built, -, Failed, Mismatch, -; also counted in Outcomes) and how many of the
    // built ones bind no guest buffers (RuntimeBufferTwin).
    [[nodiscard]] std::array<std::uint64_t, TwinOutcomeCount> C1Outcomes() const;
    [[nodiscard]] std::uint64_t C1RuntimeBuilt() const;
    [[nodiscard]] std::size_t Size() const;

private:
    struct State;
    std::shared_ptr<State> state;
};

ResolvedTwins& Twins();

}

#endif
