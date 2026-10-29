#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ResolvedPush.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <cstring>

namespace AgcDriver::NewPath {

bool ResolvedModule(const ShaderRecompiler::RecompileResult& program) {
    return program.newPathResolved;
}

std::array<std::byte, Abi::PushBlockBytes> TwinPushWords(const ShaderRecompiler::RecompileResult& twin) {
    std::array<std::byte, Abi::PushBlockBytes> words{};
    std::copy_n(twin.pushConstants.begin(), std::min<std::size_t>(twin.pushConstants.size(), words.size()), words.begin());
    return words;
}

std::optional<PushBytes> ComposeResolvedPush(const std::array<std::byte, Abi::PushBlockBytes>& legacy, std::uint32_t legacyBytes, std::uint64_t tableRoot, std::uint64_t faultRecord) {
    if (tableRoot == 0 || faultRecord == 0 || legacyBytes % 4u != 0u || legacyBytes > Abi::PushBlockBytes - ResolvedDriverWords * 4u) return std::nullopt;
    PushBytes bytes{};
    std::memcpy(bytes.data() + ResolvedPageTableWord * 4u, &tableRoot, 8);
    std::memcpy(bytes.data() + ResolvedFaultRecordWord * 4u, &faultRecord, 8);
    std::copy_n(legacy.begin(), legacyBytes, bytes.begin() + ResolvedDriverWords * 4u);
    return bytes;
}

}
