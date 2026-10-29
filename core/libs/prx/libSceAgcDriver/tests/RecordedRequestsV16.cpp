// agc_recorded_requests_v16_tests: the offline tools read the version-16 compute requests of the 2026-10-09 corpus
// tree, which has neither the v13 float-mode bool nor the v16 float64DenormPreserve bool (tools/RecordedRequests.hpp).
// The test builds both v16 layouts from a current compute request without a float mode: the current writer's tail
// after narrowSubgroupClock is 32 bytes for a compute request without linked-call fallbacks (float64DenormPreserve 1,
// registrationModules 1, registrationWordLimit 4, vertexSubgroups 1, float16DenormPreserve 1, linkedCallFallbacks 8,
// spannedWaves 1, depthRangeRemap 1, float32FlushToZero 1, float32DenormPreserve 1, fusedExactArithmetic 1,
// largePixelFunctionLdsBuffer 1, pixelFunctionStorageDwords 4, singleWorkgroup 1, singleLaneGroups 1, newPathResolved 1, bdaSegments 1, orderedWorkgroupWaves 1, newPathRuntimeBuffers 1). Upstream v16 is
// the v51 stream without its last 31 bytes; the corpus tree's v16 is that without the last byte too and without the float-mode bool
// that follows the memory regions.
#include "BdaAbi.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/tools/RecordedRequests.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void Expect(bool condition, const char* what) {
    if (!condition) throw std::runtime_error(what);
}

std::string WithVersion(std::string bytes, std::uint32_t version, std::size_t drop, bool dropFloatMode = false) {
    Expect(bytes.size() > drop + 8u, "serialized request shorter than its tail");
    bytes.resize(bytes.size() - drop);
    if (dropFloatMode) {
        const auto at = AgcDriver::RecordedRequests::Detail::ComputeRegionsEnd(bytes);
        Expect(at.has_value() && *at < bytes.size() && bytes[*at] == '\0', "the float-mode bool is not where the test expects it");
        bytes.erase(*at, 1u);
    }
    for (std::uint32_t i = 0; i < 4u; ++i) bytes[4u + i] = static_cast<char>((version >> (8u * i)) & 0xffu);
    return AgcDriver::RecordedRequests::Detail::Encode(bytes);
}

}

int main() try {
    using namespace ShaderRecompiler;
    const std::vector<std::uint32_t> code{0xbf810000u};
    const std::array<std::uint32_t, 2> userData{0x200000u, 7u};
    const std::array<std::uint32_t, 4> srt{1u, 2u, 3u, 4u};
    const std::array<MemoryRegion, 1> regions{{{0x200000u, std::as_bytes(std::span(srt))}}};
    const std::array<std::uint32_t, 2> capabilities{spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses};
    const std::array<std::string_view, 1> extensions{"SPV_KHR_physical_storage_buffer"};
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x22000u, code, 0, {}};
    request.context.waveSize = 64u;
    request.context.userData = userData;
    request.context.memory = regions;
    request.context.compute = ShaderComputeStageInfo{{512u, 1u, 1u}, 0u, {true, false, false}, false, 1u};
    request.target.vulkanVersion = 0x00401000u;
    request.target.spirvVersion = 0x00010300u;
    request.target.subgroupSize = 32u;
    request.target.bdaAbiVersion = BdaAbi::Version;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.target.float64DenormPreserve = true;
    request.layout.pushConstantSizeBytes = 128u;
    const auto current = RequestSerializer{}.Serialize(request);
    const auto bytes = AgcDriver::RecordedRequests::Detail::Decode(current);
    Expect(AgcDriver::RecordedRequests::Detail::Encode(bytes) == current, "the tools' base64 does not round-trip the serializer's");

    bool repaired = true;
    const auto plain = AgcDriver::RecordedRequests::Deserialize(current, &repaired);
    Expect(!repaired, "a current request was repaired");

    const auto withBool = WithVersion(bytes, 16u, 31u);
    const auto read16 = RequestSerializer{}.Deserialize(withBool);
    Expect(read16.request.target.float64DenormPreserve, "v16 with float64DenormPreserve: the test's tail arithmetic no longer matches the writer");

    const auto withoutBool = WithVersion(bytes, 16u, 32u, true);
    bool failed = false;
    try {
        static_cast<void>(RequestSerializer{}.Deserialize(withoutBool));
    } catch (const std::runtime_error& error) {
        failed = std::string_view(error.what()) == "RequestSerializer: truncated data";
    }
    Expect(failed, "the reader accepts a corpus-tree v16 request; the repair is no longer needed");
    const auto old = AgcDriver::RecordedRequests::Deserialize(withoutBool, &repaired);
    Expect(repaired, "the corpus-tree v16 request was not repaired");
    Expect(!old.request.target.float64DenormPreserve, "the repaired v16 request did not default float64DenormPreserve to false");
    Expect(std::ranges::equal(old.request.shader.code, code), "the repaired request lost its code");
    Expect(std::ranges::equal(old.request.context.userData, userData), "the repaired request lost its user data");
    Expect(old.request.context.memory.size() == 1u && std::ranges::equal(old.request.context.memory[0].bytes, regions[0].bytes), "the repaired request lost its memory region");
    Expect(old.request.context.compute.has_value() && old.request.context.compute->numThreads[0] == 512u, "the repaired request lost its compute stage");

    failed = false;
    try {
        static_cast<void>(AgcDriver::RecordedRequests::Deserialize(WithVersion(bytes, 15u, 33u, true)));
    } catch (const std::runtime_error&) {
        failed = true;
    }
    Expect(failed, "a truncated request of another version was repaired");
    std::puts("RecordedRequestsV16: corpus-tree v16 compute requests load in the tools, current requests read unchanged");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "RecordedRequestsV16: %s\n", error.what());
    return 1;
}
