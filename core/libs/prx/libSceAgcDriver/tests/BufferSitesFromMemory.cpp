// agc_buffer_sites_from_memory_tests: a buffer store through a V# that the shader loads with s_buffer_load (a
// ReadConstBuffer chain) is a site read from guest memory, so the result says its ranges may be stale on a
// dispatch-cache hit (RecompileResult::rangesFromMemory). A store through a V# in user SGPRs is not, and its write range
// is the V#'s records. The recompiler half of the TransferKernel chain (fable-void 6105127910's fromMemory, extended to
// ReadConstBuffer); LinearStore and TransferKernel refuse such programs because the driver re-reads the user-data V#.
#include "BdaAbi.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/tools/RecordedRequests.hpp"
#include <fstream>
#include <sstream>
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

constexpr std::uint32_t Raw = 0xfacu | (22u << 12u) | (3u << 28u);
constexpr std::array<std::uint32_t, 2> Capabilities{spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses};
constexpr std::array<std::string_view, 1> Extensions{"SPV_KHR_physical_storage_buffer"};

alignas(256) std::array<std::uint32_t, 16> TableCode{
    0xf4280200u, 0xfa000000u,  // s_buffer_load_dwordx4 s[8:11], s[0:3], 0: the V# comes from the table in memory
    0xbf8cc07fu,               // s_waitcnt lgkmcnt(0)
    (0x1au << 25u) | (1u << 17u) | (0u << 9u) | 130u,  // v_lshlrev_b32 v1, 2, v0
    0xe0701000u, (0x80u << 24u) | (2u << 16u) | (0u << 8u) | 1u,  // buffer_store_dword v0, v1, s[8:11], 0 offen
    0xbf810000u};
alignas(256) std::array<std::uint32_t, 16> DirectCode{
    (0x1au << 25u) | (1u << 17u) | (0u << 9u) | 130u,  // v_lshlrev_b32 v1, 2, v0
    0xe0701000u, (0x80u << 24u) | (1u << 16u) | (0u << 8u) | 1u,  // buffer_store_dword v0, v1, s[4:7], 0 offen
    0xbf810000u};
alignas(256) std::array<std::uint32_t, 4> Table{};
alignas(256) std::array<std::uint32_t, 64> Output{};

std::array<std::uint32_t, 4> Descriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, Raw};
}

RecompileResult Compile(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, std::span<const MemoryRegion> regions) {
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}};
    request.context.waveSize = 64u;
    request.context.userData = userData;
    request.context.compute = ShaderComputeStageInfo{{64u, 1u, 1u}, 0u, {true, false, false}, false, 1u};
    request.context.memory = MemoryRegionView(regions);
    request.target.vulkanVersion = 0x00401000u;
    request.target.spirvVersion = 0x00010300u;
    request.target.subgroupSize = 32u;
    request.target.bdaAbiVersion = BdaAbi::Version;
    request.target.supportedCapabilities = Capabilities;
    request.target.supportedExtensions = Extensions;
    request.layout.pushConstantSizeBytes = 128u;
    request.useCache = false;
    return Recompile(request);
}

}

int Corpus(int argc, char** argv) {
    std::uint32_t requests = 0, linear = 0, fills = 0, copies = 0, fromMemory = 0, failed = 0;
    for (int index = 1; index < argc; ++index) {
        std::ifstream file(argv[index], std::ios::binary);
        std::ostringstream text;
        text << file.rdbuf();
        auto body = text.str();
        while (!body.empty() && (body.back() == '\n' || body.back() == '\r' || body.back() == ' ')) body.pop_back();
        try {
            auto deserialized = AgcDriver::RecordedRequests::Deserialize(body);
            auto request = deserialized.request;
            if (request.shader.stage != ShaderStage::Compute) continue;
            request.useCache = false;
            const auto result = Recompile(request);
            ++requests;
            linear += result.linearStore.has_value();
            if (result.transferKernel) (result.transferKernel->kind == TransferKernel::Kind::Copy ? copies : fills) += 1u;
            fromMemory += result.rangesFromMemory;
            if (result.linearStore || result.transferKernel) std::printf("%s: linear store%s%s\n", argv[index], result.transferKernel ? (result.transferKernel->kind == TransferKernel::Kind::Copy ? ", copy kernel" : ", fill kernel") : "", result.rangesFromMemory ? ", ranges from memory" : "");
        } catch (const std::exception& error) {
            ++failed;
            std::fprintf(stderr, "%s: %s\n", argv[index], error.what());
        }
    }
    std::printf("BufferSitesFromMemory corpus: %u compute requests, %u linear stores, %u fill kernels, %u copy kernels, %u with ranges from memory, %u failed\n", requests, linear, fills, copies, fromMemory, failed);
    return failed == 0u ? 0 : 1;
}

int main(int argc, char** argv) try {
    if (argc > 1) return Corpus(argc, argv);
    const auto out = Descriptor(Output.data(), sizeof(Output));
    Table = out;
    const auto table = Descriptor(Table.data(), sizeof(Table));
    const std::array<std::uint32_t, 8> userData{table[0], table[1], table[2], table[3], out[0], out[1], out[2], out[3]};
    const auto address = [](const auto& value) { return reinterpret_cast<std::uintptr_t>(value.data()); };
    const std::array<MemoryRegion, 3> regions{{
        {address(TableCode), std::as_bytes(std::span(TableCode))},
        {address(DirectCode), std::as_bytes(std::span(DirectCode))},
        {address(Table), std::as_bytes(std::span(Table))}}};
    const auto viaTable = Compile(std::span<const std::uint32_t>(TableCode).first(7), userData, std::span<const MemoryRegion>(regions));
    Expect(viaTable.rangesFromMemory, "a store through a V# loaded with s_buffer_load is not marked as read from guest memory");
    const auto direct = Compile(std::span<const std::uint32_t>(DirectCode).first(4), userData, std::span<const MemoryRegion>(regions));
    Expect(!direct.rangesFromMemory, "a store through a user-SGPR V# is marked as read from guest memory");
    const auto begin = static_cast<std::uint64_t>(address(Output));
    Expect(!direct.unboundedWrites && direct.writeRanges.size() == 1u && direct.writeRanges.front() == std::pair<std::uint64_t, std::uint64_t>{begin, begin + sizeof(Output)}, "the user-SGPR store's write range is not its V#'s records");
    Expect(!viaTable.linearStore && !viaTable.transferKernel, "a store through a V# from guest memory was recognized as a linear store or transfer kernel");
    std::printf("buffer sites from memory: s_buffer_load V# marked (%zu write ranges), user-SGPR V# not (range %zu bytes)\n", viaTable.writeRanges.size(), sizeof(Output));
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "BufferSitesFromMemory: %s\n", error.what());
    return 1;
}
