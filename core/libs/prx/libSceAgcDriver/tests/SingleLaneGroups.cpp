// agc_single_lane_groups_tests: SpirvTarget::singleLaneGroups (S4 prototype, APS5_NEW_PATH_ONE_LANE) lays a wave64
// compute program whose workgroup holds more than one wave out one guest lane per invocation, leaves a single-wave
// workgroup, wave32 and programs with returning atomics alone, is part of the recompile cache key and survives request
// serialization.
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
// buffer_atomic_add v1, v0, s[0:3], 0 offen glc (returning); s_waitcnt 0; s_endpgm.
const std::vector<std::uint32_t> ReturningAtomic{0xe0c85000u, 0x80000100u, 0xbf8c0000u, 0xbf810000u};
constexpr std::array<std::uint32_t, 4> UserData{0x10000000u, 0x00000000u, 0x100u, 0x31016facu};
constexpr std::array<std::uint32_t, 2> Capabilities{spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses};
constexpr std::array<std::string_view, 1> Extensions{"SPV_KHR_physical_storage_buffer"};

RecompileRequest Request(std::uint32_t threads, std::uint32_t wave, bool singleLane, const std::vector<std::uint32_t>& code = Code) {
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x22000u, code, 0, {}};
    request.context.waveSize = wave;
    request.context.userData = UserData;
    request.context.compute = ShaderComputeStageInfo{{threads, 1u, 1u}, 0u, {true, false, false}, false, 1u};
    request.target.vulkanVersion = 0x00401000u;
    request.target.spirvVersion = 0x00010300u;
    request.target.subgroupSize = 32u;
    request.target.bdaAbiVersion = BdaAbi::Version;
    request.target.supportedCapabilities = Capabilities;
    request.target.supportedExtensions = Extensions;
    request.target.maxWorkgroupSize = {1024u, 1024u, 64u};
    request.target.maxWorkgroupInvocations = 1024u;
    request.target.maxWorkgroupSharedMemoryBytes = 49152u;
    request.target.singleLaneGroups = singleLane;
    request.layout.pushConstantSizeBytes = 128u;
    request.useCache = false;
    return request;
}

std::uint32_t HostLanes(std::uint32_t threads, std::uint32_t wave, bool singleLane, const std::vector<std::uint32_t>& code = Code) {
    const auto result = Recompile(Request(threads, wave, singleLane, code));
    Expect(!result.spirv.empty(), "the program did not recompile");
    return result.hostSubgroupSize;
}

}

int main() try {
    const auto twoLane = HostLanes(1024u, 64u, false);
    const auto oneLane = HostLanes(1024u, 64u, true);
    Expect(oneLane == 64u, "a 1024-thread wave64 group with singleLaneGroups is laid out with host subgroup " + std::to_string(oneLane) + ", expected 64 (one guest lane per invocation)");
    Expect(twoLane != 64u, "a 1024-thread wave64 group without singleLaneGroups is already one lane per invocation; the test no longer tells the layouts apart");
    Expect(HostLanes(512u, 64u, true) == 64u, "a 512-thread wave64 group with singleLaneGroups is not one lane per invocation");
    Expect(HostLanes(64u, 64u, true) == HostLanes(64u, 64u, false), "singleLaneGroups changed a single-wave workgroup");
    Expect(HostLanes(256u, 32u, true) == HostLanes(256u, 32u, false), "singleLaneGroups changed a wave32 program");
    Expect(HostLanes(1024u, 64u, true, ReturningAtomic) == HostLanes(1024u, 64u, false, ReturningAtomic), "a program with a returning buffer atomic was laid out one lane per invocation; its two host subgroups would race (H2-5)");

    std::vector<std::uint64_t> keyOff;
    std::vector<std::uint64_t> keyOn;
    RecompileCacheKey::Build(Request(1024u, 64u, false), keyOff);
    RecompileCacheKey::Build(Request(1024u, 64u, true), keyOn);
    Expect(keyOff != keyOn, "singleLaneGroups is not part of the recompile cache key");

    const auto text = RequestSerializer{}.Serialize(Request(1024u, 64u, true));
    const auto read = RequestSerializer{}.Deserialize(text);
    Expect(read.request.target.singleLaneGroups, "singleLaneGroups did not survive request serialization");
    std::puts("SingleLaneGroups: multi-wave wave64 groups run one lane per invocation under singleLaneGroups, keyed and serialized");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "SingleLaneGroups: %s\n", error.what());
    return 1;
}
