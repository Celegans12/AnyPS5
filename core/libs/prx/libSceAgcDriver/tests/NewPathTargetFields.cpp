// agc_new_path_target_fields_tests: SpirvTarget::newPathResolved and bdaSegments (S4 epoch 1, piece 1) are part of the
// recompile cache key and survive request serialization (v49). bdaSegments is refused until its emission exists, so no
// module built today's way is ever cached under its key; newPathResolved recompiles (the page-table walk, compute).
#include "BdaAbi.hpp"
#include "CacheKey.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "Recompiler.hpp"
#include <spirv/unified1/spirv.hpp>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ShaderRecompiler;

void Expect(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error(what);
}

const std::vector<std::uint32_t> Code{0xbf810000u};
constexpr std::array<std::uint32_t, 2> Capabilities{spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses};
constexpr std::array<std::string_view, 1> Extensions{"SPV_KHR_physical_storage_buffer"};

RecompileRequest Request(bool resolved, bool segments) {
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x22000u, Code, 0, {}};
    request.context.waveSize = 64u;
    request.context.compute = ShaderComputeStageInfo{{64u, 1u, 1u}, 0u, {true, false, false}, false, 1u};
    request.target.vulkanVersion = 0x00401000u;
    request.target.spirvVersion = 0x00010300u;
    request.target.subgroupSize = 32u;
    request.target.bdaAbiVersion = BdaAbi::Version;
    request.target.supportedCapabilities = Capabilities;
    request.target.supportedExtensions = Extensions;
    request.target.newPathResolved = resolved;
    request.target.bdaSegments = segments;
    request.layout.pushConstantSizeBytes = 128u;
    request.useCache = false;
    return request;
}

std::vector<std::uint64_t> Key(bool resolved, bool segments) {
    std::vector<std::uint64_t> key;
    RecompileCacheKey::Build(Request(resolved, segments), key);
    return key;
}

bool Refused(bool resolved, bool segments) {
    try {
        static_cast<void>(Recompile(Request(resolved, segments)));
    } catch (const std::exception& error) {
        return std::string_view(error.what()).find("not implemented yet") != std::string_view::npos;
    }
    return false;
}

}

int main() try {
    const auto off = Key(false, false);
    Expect(off != Key(true, false), "newPathResolved is not part of the recompile cache key");
    Expect(off != Key(false, true) && Key(true, false) != Key(true, true), "bdaSegments is not part of the recompile cache key");
    for (const auto [resolved, segments] : {std::pair{true, false}, std::pair{false, true}, std::pair{true, true}}) {
        const auto read = RequestSerializer{}.Deserialize(RequestSerializer{}.Serialize(Request(resolved, segments)));
        Expect(read.request.target.newPathResolved == resolved && read.request.target.bdaSegments == segments, "the new-path target fields did not survive request serialization");
    }
    Expect(!Refused(false, false), "a request without the new-path fields was refused");
    Expect(!Refused(true, false), "a compute request with newPathResolved was refused");
    Expect(Refused(false, true) && Refused(true, true), "a request with bdaSegments recompiled before its emission exists");
    std::puts("NewPathTargetFields: newPathResolved and bdaSegments are keyed and serialized; bdaSegments is refused until implemented");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "NewPathTargetFields: %s\n", error.what());
    return 1;
}
