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

enum class TwinOutcome : std::uint8_t { Built, NotAddressBased, Failed, Mismatch };
inline constexpr std::size_t TwinOutcomeCount = 4;

[[nodiscard]] bool UsesAddressTables(const ShaderRecompiler::RecompileResult& compiled);
[[nodiscard]] bool WalkReplacesTables(const ShaderRecompiler::RecompileResult& twin);
[[nodiscard]] std::string TwinDifference(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin);
[[nodiscard]] bool TwinCompatible(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin);

class ResolvedTwins {
public:
    ResolvedTwins();
    void Remember(const std::shared_ptr<const ShaderRecompiler::RecompileResult>& compiled, std::shared_ptr<const ShaderRecompiler::RecompileResult> twin, TwinOutcome outcome);
    [[nodiscard]] std::shared_ptr<const ShaderRecompiler::RecompileResult> Of(const ShaderRecompiler::RecompileResult& compiled) const;
    [[nodiscard]] std::array<std::uint64_t, TwinOutcomeCount> Outcomes() const;
    [[nodiscard]] std::size_t Size() const;

private:
    struct State;
    std::shared_ptr<State> state;
};

ResolvedTwins& Twins();

}

#endif
