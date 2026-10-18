#include "BdaShader.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "SpirvBackend/SpirvBda.hpp"
#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace {

template<typename TAction>
void reject(TAction action, const char* expected) {
    try { action(); }
    catch (const std::runtime_error& error) {
        AgcDriver::Graphics::Require(std::string(error.what()).find(expected) != std::string::npos, std::string("unexpected contract error: ") + error.what());
        return;
    }
    throw std::runtime_error(std::string("missing contract rejection: ") + expected);
}

std::uint32_t CallSites(const std::vector<std::uint32_t>& spirv, const char* name) {
    std::uint32_t function = 0;
    std::uint32_t calls = 0;
    for (std::size_t at = 5; at < spirv.size() && (spirv[at] >> 16u) != 0u; at += spirv[at] >> 16u) {
        const auto opcode = spirv[at] & 0xffffu;
        if (opcode == spv::OpName && std::strcmp(reinterpret_cast<const char*>(&spirv[at + 2u]), name) == 0) function = spirv[at + 1u];
        if (opcode == spv::OpFunctionCall && function != 0u && spirv[at + 3u] == function) ++calls;
    }
    return calls;
}

}

void RunBdaContractTests() {
    using namespace ShaderRecompiler;
    const std::array<std::uint32_t, 3> capabilities{spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess};
    const std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    SpirvTargetOptions target{0x00401000u, 0x00010300u, 32, BdaAbi::Version, capabilities, extensions};
    IrProgram program;
    program.Resources().stage = IrShaderStage::Compute;
    program.Info().usesDma = true;
    ValidateBdaTarget(program, target);
    target.bdaAbiVersion = 0;
    reject([&] { ValidateBdaTarget(program, target); }, "ABI version");
    target.bdaAbiVersion = BdaAbi::Version;
    target.supportedCapabilities = {};
    reject([&] { ValidateBdaTarget(program, target); }, "capability");
    target.supportedCapabilities = capabilities;
    target.supportedExtensions = {};
    reject([&] { ValidateBdaTarget(program, target); }, "extension");
    target.supportedExtensions = extensions;
    auto& block = program.CreateBlock();
    block.AppendInstruction(&program.CreateValue(IrOpcode::Barrier, IrType::Void));
    program.BlockOrder().push_back(&block);
    ValidateBdaTarget(program, target);
    AgcDriver::Graphics::Require(!BdaInvocationsMayStop(program), "a BDA program with barriers must keep its invocations running");
    program.Resources().stage = IrShaderStage::TessellationControl;
    ValidateBdaTarget(program, target);
    program.Resources().stage = IrShaderStage::Compute;
    RecompileRequest request{};
    request.target.bdaAbiVersion = BdaAbi::Version;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.target.fragmentShaderBarycentricEnabled = false;
    const RequestSerializer serializer;
    const auto encoded = serializer.Serialize(request);
    const auto decoded = serializer.Deserialize(encoded);
    AgcDriver::Graphics::Require(decoded.request.target.bdaAbiVersion == BdaAbi::Version && decoded.request.target.supportedCapabilities.size() == capabilities.size() && decoded.request.target.supportedExtensions[1] == extensions[1], "BDA request serialization changed target contract");
    auto invalid = encoded;
    invalid[0] = invalid[0] == 'A' ? 'B' : 'A';
    reject([&] { static_cast<void>(serializer.Deserialize(invalid)); }, "request signature");
    reject([&] { static_cast<void>(serializer.Deserialize("NVNQQWMAAAA=")); }, "serialization version");
    for (const std::uint32_t bits : {8u, 16u, 32u}) {
        const auto shader = MakeBdaWriteTestShader(0x7fff1235fffdULL, 0xa1b2c3d4u, bits, false);
        const auto lookups = CallSites(shader, "get_bda_write_pointer");
        const auto loops = CallSites(shader, "write_bda_bytes");
        AgcDriver::Graphics::Require(loops == 1u && lookups == (bits == 32u ? 2u : 1u), "a runtime V# write of " + std::to_string(bits) + " bits has " + std::to_string(lookups) + " write lookups and " + std::to_string(loops) + " byte-loop calls, not its byte path through write_bda_bytes");
    }
    // The range index: for every probed address, its bucket's bounds hold the upper bound (the count
    // of ranges beginning at or below it) that the search over the whole table finds.
    AgcDriver::Graphics::Require(BdaAbi::BuildRangeIndex(nullptr, 0).empty(), "an empty BDA table got a range index");
    std::uint64_t seed = 0x9e3779b97f4a7c15ull;
    const auto next = [&] {
        seed ^= seed << 13u;
        seed ^= seed >> 7u;
        seed ^= seed << 17u;
        return seed;
    };
    for (std::uint32_t table = 0; table < 24u; ++table) {
        std::vector<BdaAbi::Range> ranges;
        std::uint64_t at = (table % 2u != 0u ? 0x7f0000000000ull : 0x100000ull) + (next() % 4096u) * 4096u;
        const auto count = BdaAbi::IndexMinRanges + static_cast<std::uint32_t>(next() % 1500u);
        for (std::uint32_t index = 0; index < count; ++index) {
            at += next() % 8u == 0u ? next() % (std::uint64_t{1} << 36u) : next() % 3u == 0u ? 0u : next() % 65536u;
            const auto length = 1u + next() % 262144u;
            // Some ranges overlap the next one (ascending by begin only).
            ranges.push_back({at, at + length + (index % 7u == 0u ? 4096u : 0u), 0x1000, BdaAbi::Read, 0});
            at += length;
        }
        const auto words = BdaAbi::BuildRangeIndex(ranges.data(), ranges.size());
        AgcDriver::Graphics::Require(!words.empty(), "a BDA table of " + std::to_string(count) + " ranges got no range index");
        BdaAbi::IndexHeader header{};
        std::memcpy(&header, words.data(), sizeof(header));
        constexpr std::size_t headerWords = sizeof(header) / sizeof(std::uint32_t);
        AgcDriver::Graphics::Require(header.buckets != 0u && header.shift <= 63u && words.size() == headerWords + header.buckets + 1u && header.buckets <= BdaAbi::IndexMaxBuckets, "a BDA range index has an invalid header");
        const auto check = [&](std::uint64_t address) {
            std::size_t upper = 0;
            while (upper < ranges.size() && ranges[upper].begin <= address) ++upper;
            std::uint32_t low = 0, high = 0;
            if (address < header.base) {
                low = high = 0;
            } else if (const auto bucket = (address - header.base) >> header.shift; bucket >= header.buckets) {
                low = high = count;
            } else {
                low = words[headerWords + bucket];
                high = words[headerWords + bucket + 1u];
            }
            AgcDriver::Graphics::Require(low <= upper && upper <= high && high <= count, "the BDA range index bounds miss the upper bound of address 0x" + std::to_string(address));
        };
        for (std::uint32_t probe = 0; probe < 600u; ++probe) {
            const auto& range = ranges[next() % ranges.size()];
            check(range.begin);
            check(range.begin - 1u);
            check(range.end - 1u);
            check(range.end);
            check(ranges.front().begin + next() % (ranges.back().end - ranges.front().begin + 1u));
        }
        check(0);
        check(~std::uint64_t{0});
    }
}
