#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_RESOLVEDPUSH_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_RESOLVEDPUSH_HPP

#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PushBlock.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ShaderRecompiler {
struct RecompileResult;
}

namespace AgcDriver::NewPath {

inline constexpr std::uint32_t ResolvedDriverWords = 6;

[[nodiscard]] bool ResolvedModule(const ShaderRecompiler::RecompileResult& program);
// A twin's own push words at offset 0 (the compute stage's), for ComposeResolvedPush.
[[nodiscard]] std::array<std::byte, Abi::PushBlockBytes> TwinPushWords(const ShaderRecompiler::RecompileResult& twin);
[[nodiscard]] std::optional<PushBytes> ComposeResolvedPush(const std::array<std::byte, Abi::PushBlockBytes>& legacy, std::uint32_t legacyBytes, std::uint64_t tableRoot, std::uint64_t faultRecord);

}

#endif
