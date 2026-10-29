// agc_new_path_target_fields_tests: SpirvTarget::newPathResolved and bdaSegments (S4 epoch 1, piece 1) are part of the
// recompile cache key and survive request serialization (v49). bdaSegments is refused until its emission exists, so no
// module built today's way is ever cached under its key; newPathResolved recompiles (the page-table walk, compute).
// orderedWorkgroupWaves (v50) is keyed and serialized too, and leaves the module unchanged where it does not apply: a
// workgroup of one wave, and a program whose invocations a BDA fault can stop (the device test covers where it does).
// newPathRuntimeBuffers (v51) is keyed and serialized, and refused without newPathResolved.
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

RecompileRequest Request(bool resolved, bool segments, bool ordered = false, std::uint32_t wave = 64u, std::uint32_t threads = 64u) {
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x22000u, Code, 0, {}};
    request.context.waveSize = wave;
    request.context.compute = ShaderComputeStageInfo{{threads, 1u, 1u}, 0u, {true, false, false}, false, 1u};
    request.target.vulkanVersion = 0x00401000u;
    request.target.spirvVersion = 0x00010300u;
    request.target.subgroupSize = 32u;
    request.target.bdaAbiVersion = BdaAbi::Version;
    request.target.supportedCapabilities = Capabilities;
    request.target.supportedExtensions = Extensions;
    request.target.newPathResolved = resolved;
    request.target.bdaSegments = segments;
    request.target.orderedWorkgroupWaves = ordered;
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
    std::vector<std::uint64_t> ordered;
    RecompileCacheKey::Build(Request(false, false, true), ordered);
    Expect(off != ordered, "orderedWorkgroupWaves is not part of the recompile cache key");
    Expect(RequestSerializer{}.Deserialize(RequestSerializer{}.Serialize(Request(false, false, true))).request.target.orderedWorkgroupWaves, "orderedWorkgroupWaves did not survive request serialization");
    Expect(!RequestSerializer{}.Deserialize(RequestSerializer{}.Serialize(Request(false, false))).request.target.orderedWorkgroupWaves, "orderedWorkgroupWaves appeared in request serialization");
    for (const auto threads : {32u, 64u}) {
        const auto plain = Recompile(Request(false, false, false, 32u, threads));
        const auto waves = Recompile(Request(false, false, true, 32u, threads));
        Expect(!plain.spirv.empty() && plain.spirv == waves.spirv, "orderedWorkgroupWaves changed a wave32 module of " + std::to_string(threads) + " threads where it does not apply");
    }
    auto runtime = Request(true, false);
    runtime.target.newPathRuntimeBuffers = true;
    std::vector<std::uint64_t> runtimeKey;
    RecompileCacheKey::Build(runtime, runtimeKey);
    Expect(runtimeKey != Key(true, false), "newPathRuntimeBuffers is not part of the recompile cache key");
    Expect(RequestSerializer{}.Deserialize(RequestSerializer{}.Serialize(runtime)).request.target.newPathRuntimeBuffers, "newPathRuntimeBuffers did not survive request serialization");
    Expect(!RequestSerializer{}.Deserialize(RequestSerializer{}.Serialize(Request(true, false))).request.target.newPathRuntimeBuffers, "newPathRuntimeBuffers appeared in request serialization");
    Expect(!Recompile(runtime).spirv.empty(), "a resolved request with newPathRuntimeBuffers did not recompile");
    auto alone = Request(false, false);
    alone.target.newPathRuntimeBuffers = true;
    bool refused = false;
    try {
        static_cast<void>(Recompile(alone));
    } catch (const std::exception& error) {
        refused = std::string_view(error.what()).find("needs newPathResolved") != std::string_view::npos;
    }
    Expect(refused, "newPathRuntimeBuffers without newPathResolved was not refused");
    std::puts("NewPathTargetFields: newPathResolved, bdaSegments, orderedWorkgroupWaves and newPathRuntimeBuffers are keyed and serialized; bdaSegments is refused until implemented, newPathRuntimeBuffers without newPathResolved");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "NewPathTargetFields: %s\n", error.what());
    return 1;
}
