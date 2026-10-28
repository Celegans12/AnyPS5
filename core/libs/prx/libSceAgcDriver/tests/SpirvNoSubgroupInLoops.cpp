// agc_spirv_no_subgroup_in_loops [request.req...]: the new path's page-table walk (SpirvTarget::newPathResolved) keeps
// every call and every subgroup operation out of its lookup loops. On NVIDIA a subgroup operation reached inside a loop
// of a helper left the loop's warp mask stale (fable-void 2cb852b537, and Honza11's device losses at the reconvergence
// after note_bda_touch inside bda_translate's loop on the head that carried that fix), so the rule is structural: no
// OpGroupNonUniform* and no OpFunctionCall between a loop header and its merge, in the BDA lookup functions and in the
// entry function (where an inlined lookup would land). Default: a synthetic program with an address load and store;
// with request files: each compute request recompiled with newPathResolved (the corpus check). The entry function's
// own guest loops may use subgroup operations (waterfalls, ballots); they are counted and printed, not failed.
#include "BdaAbi.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/tools/RecordedRequests.hpp"
#include <spirv/unified1/spirv.hpp>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ShaderRecompiler;

struct LoopCounts {
    std::uint32_t lookupLoops = 0;
    std::uint32_t lookupViolations = 0;
    std::uint32_t entryLoops = 0;
    std::uint32_t entrySubgroupInLoops = 0;
    std::uint32_t entryCallsInLoops = 0;
    std::uint32_t touchNotes = 0;
};

bool Lookup(std::string_view name) {
    return name.starts_with("get_bda_") || name.starts_with("probe_bda_") || name == "bda_anchor_range";
}

LoopCounts Count(std::span<const std::uint32_t> words) {
    std::map<std::uint32_t, std::string> names;
    std::uint32_t entry = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpName) names[words[cursor + 1]] = reinterpret_cast<const char*>(&words[cursor + 2]);
        if (op == spv::OpEntryPoint) entry = words[cursor + 2];
    }
    LoopCounts counts;
    std::uint32_t function = 0;
    std::vector<std::uint32_t> openMerges;
    std::uint32_t pendingMerge = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpFunction) {
            function = words[cursor + 2];
            openMerges.clear();
            const auto& name = names[function];
            if (name.find("touch") != std::string::npos) ++counts.touchNotes;
        }
        if (op == spv::OpLabel) {
            const auto label = words[cursor + 1];
            while (!openMerges.empty() && openMerges.back() == label) openMerges.pop_back();
            if (pendingMerge != 0u) {
                openMerges.push_back(pendingMerge);
                pendingMerge = 0;
            }
        }
        if (op == spv::OpLoopMerge) {
            pendingMerge = words[cursor + 1];
            if (function == entry) ++counts.entryLoops;
            else if (Lookup(names[function])) ++counts.lookupLoops;
        }
        if (openMerges.empty()) continue;
        const bool subgroup = op >= spv::OpGroupNonUniformElect && op <= spv::OpGroupNonUniformQuadSwap;
        const bool call = op == spv::OpFunctionCall;
        if (!subgroup && !call) continue;
        if (function == entry) {
            counts.entrySubgroupInLoops += subgroup;
            counts.entryCallsInLoops += call;
        } else if (Lookup(names[function])) {
            ++counts.lookupViolations;
        }
    }
    return counts;
}

// SPIRV_NO_SUBGROUP_DUMP=<dir>: write each checked module there (for spirv-val; this build may lack SPIRV-Tools).
void Dump(std::span<const std::uint32_t> words, const std::string& name) {
    const char* directory = std::getenv("SPIRV_NO_SUBGROUP_DUMP");
    if (directory == nullptr || *directory == 0) return;
    std::ofstream file(std::string(directory) + "/" + name + ".spv", std::ios::binary);
    file.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size_bytes()));
}

void Expect(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error(what);
}

std::string ReadText(const char* path) {
    std::ifstream file(path, std::ios::binary);
    std::ostringstream stream;
    stream << file.rdbuf();
    auto text = stream.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
    return text;
}

}

int main(int argc, char** argv) try {
    static constexpr std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    static constexpr std::array<std::uint32_t, 8> capabilities{
        spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityGroupNonUniform,
        spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformVote, spv::CapabilityGroupNonUniformArithmetic};
    if (argc < 2) {
        // global_load_dword v2, v[0:1], off; s_waitcnt 0; global_store_dword v[0:1], v2, off; s_endpgm.
        const std::vector<std::uint32_t> code{0xdc308000u, 0x027d0000u, 0xbf8c0000u, 0xdc708000u, 0x007d0200u, 0xbf810000u};
        RecompileRequest request{};
        request.shader = {ShaderStage::Compute, 0x20000u, code, 0, {}};
        request.context.waveSize = 64u;
        request.context.compute = ShaderComputeStageInfo{{64u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
        request.target.vulkanVersion = 0x00401000u;
        request.target.spirvVersion = 0x00010300u;
        request.target.subgroupSize = 32u;
        request.target.bdaAbiVersion = BdaAbi::Version;
        request.target.supportedCapabilities = capabilities;
        request.target.supportedExtensions = extensions;
        request.target.newPathResolved = true;
        request.layout.pushConstantSizeBytes = 128;
        request.useCache = false;
        const auto result = Recompile(request);
        Expect(!result.spirv.empty(), "the address program did not recompile under newPathResolved");
        const auto words = result.spirv.Copy();
        Dump(words, "synthetic");
        const auto counts = Count(words);
        Expect(counts.lookupLoops != 0u, "no BDA lookup function has a loop: the page-table walk was not emitted");
        Expect(counts.lookupViolations == 0u, "a BDA lookup loop holds " + std::to_string(counts.lookupViolations) + " calls or subgroup operations");
        Expect(counts.entrySubgroupInLoops == 0u && counts.entryCallsInLoops == 0u, "the entry function of a loop-free program has calls or subgroup operations inside loops");
        Expect(counts.touchNotes == 0u, "a module of the walk defines a touch-note function");
        std::printf("SpirvNoSubgroupInLoops: synthetic address program: %u lookup loops, 0 calls or subgroup operations inside them\n", counts.lookupLoops);
        return 0;
    }
    std::uint32_t requests = 0;
    std::uint32_t failures = 0;
    for (int index = 1; index < argc; ++index) {
        auto deserialized = AgcDriver::RecordedRequests::Deserialize(ReadText(argv[index]));
        auto request = deserialized.request;
        if (request.shader.stage != ShaderStage::Compute) continue;
        request.target.newPathResolved = true;
        request.useCache = false;
        const auto result = Recompile(request);
        const auto words = result.spirv.Copy();
        Dump(words, std::filesystem::path(argv[index]).stem().string());
        const auto counts = Count(words);
        ++requests;
        failures += counts.lookupViolations != 0u || counts.touchNotes != 0u;
        std::printf("%s: words %zu, lookup loops %u, inside them %u; entry loops %u with %u subgroup operations and %u calls inside (guest loops, informational)\n", argv[index], result.spirv.size(), counts.lookupLoops, counts.lookupViolations, counts.entryLoops, counts.entrySubgroupInLoops, counts.entryCallsInLoops);
    }
    std::printf("SpirvNoSubgroupInLoops: %u compute requests, %u with a call or subgroup operation inside a lookup loop\n", requests, failures);
    return failures == 0u ? 0 : 1;
} catch (const std::exception& error) {
    std::fprintf(stderr, "SpirvNoSubgroupInLoops: %s\n", error.what());
    return 1;
}
