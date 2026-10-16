#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

// s_buffer_load_dword/x2/x4/x8/x16 through a GPU V# (words from memory through v_readfirstlane), with immediate
// and SGPR offsets, against a per-dword model of the scalar buffer bounds: whole runs, runs partly past the
// range end (read 0 there), runs with dwords the program never uses, unaligned offsets and strided ranges.
namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 64;
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t TableBase = 0x0;
constexpr std::uint32_t DataBase = 0x1000;
constexpr std::uint32_t DataBytes = 0x4000;
constexpr std::uint32_t OutputBase = 0x8000;
constexpr std::uint32_t OutputStride = 128;
constexpr std::uint32_t Untouched = 0xdeadbeefu;
constexpr std::uint32_t Format = 0x00016facu;
constexpr std::uint32_t NullOffset = 0x7du;

struct Load {
    std::uint32_t dwords;
    std::uint32_t destination;
    bool sgprOffset;
    std::uint32_t immediate;
};

// Program 0 reads every dword of five loads into SGPRs 16..46 and stores them as output dwords 0..30.
constexpr std::array<Load, 5> Loads{{{16u, 16u, false, 0u}, {8u, 32u, true, 8u}, {4u, 40u, false, 20u}, {2u, 44u, false, 28u}, {1u, 46u, false, 32u}}};
// Program 1 reads an x8 at the SGPR offset and stores only its dwords 0, 2, 5 and 7.
constexpr std::array<std::uint32_t, 4> SparseDwords{0u, 2u, 5u, 7u};

std::uint32_t SmemOp(std::uint32_t dwords) {
    switch (dwords) {
    case 1u: return 8u;
    case 2u: return 9u;
    case 4u: return 10u;
    case 8u: return 11u;
    default: return 12u;
    }
}

void SBufferLoad(std::vector<std::uint32_t>& code, const Load& load) {
    code.push_back(0xf4000000u | (SmemOp(load.dwords) << 18u) | (load.destination << 6u) | (8u >> 1u));
    code.push_back(((load.sgprOffset ? 2u : NullOffset) << 25u) | load.immediate);
}

void Mubuf(std::uint32_t op, bool offen, std::uint32_t immediate, std::uint32_t vaddr, std::uint32_t vdata, std::uint32_t resource, std::uint32_t soffset, std::vector<std::uint32_t>& code) {
    code.push_back(0xe0000000u | (op << 18u) | (offen ? 0x1000u : 0u) | immediate);
    code.push_back((soffset << 24u) | (resource / 4u << 16u) | (vdata << 8u) | vaddr);
}

std::uint32_t MoveFromScalar(std::uint32_t vgpr, std::uint32_t source) {
    return 0x7e000000u | (vgpr << 17u) | (1u << 9u) | source;
}

alignas(256) std::array<std::array<std::uint32_t, 128>, 2> Programs{};

std::span<const std::uint32_t> Code(std::uint32_t variant) {
    // v[4:7] = table words 1..4 (V# word 1, records, word 3, offset); s9..s11 and s2 take them, so s[8:11] is a GPU V#.
    std::vector<std::uint32_t> code;
    Mubuf(14u, false, 4u, 0u, 4u, 4u, 0x80u, code);
    code.push_back(0xbf8c0000u);
    for (const auto word : {0x7e120504u, 0x7e140505u, 0x7e160506u, 0x7e040507u}) code.push_back(word);
    std::uint32_t outputs = 0;
    if (variant == 0u) {
        for (const auto& load : Loads) SBufferLoad(code, load);
        code.push_back(0xbf8cc07fu);
        for (std::uint32_t dword = 0; dword < 31u; ++dword) code.push_back(MoveFromScalar(8u + dword, 16u + dword));
        code.push_back(0x7e4e0280u);
        outputs = 32u;
    } else {
        SBufferLoad(code, {8u, 16u, true, 0u});
        code.push_back(0xbf8cc07fu);
        for (std::uint32_t index = 0; index < SparseDwords.size(); ++index) code.push_back(MoveFromScalar(8u + index, 16u + SparseDwords[index]));
        outputs = 4u;
    }
    code.push_back(0x34500087u);
    for (std::uint32_t store = 0; store < outputs / 4u; ++store) Mubuf(30u, true, store * 16u, 40u, 8u + store * 4u, 12u, 0x80u, code);
    code.push_back(0xbf8c3f70u);
    code.push_back(0xbf810000u);
    auto& program = Programs.at(variant);
    Require(code.size() <= program.size(), "gpu scalar buffer spans: program too long");
    std::copy(code.begin(), code.end(), program.begin());
    return std::span<const std::uint32_t>(program.data(), code.size());
}

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "gpu scalar buffer spans: cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint8_t* Data() { return block; }
    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(block); }

private:
    std::uint8_t* block = nullptr;
};

struct Case {
    const char* name;
    std::uint32_t stride;
    std::uint32_t records;
    std::uint32_t offset;
};

constexpr std::array<Case, 9> Cases{{
    {"records 4096", 0u, 4096u, 0u},
    {"records 37", 0u, 37u, 0u},
    {"records 37 offset 4", 0u, 37u, 4u},
    {"records 37 offset 2", 0u, 37u, 2u},
    {"records 61", 0u, 61u, 12u},
    {"records 3", 0u, 3u, 0u},
    {"records 0", 0u, 0u, 0u},
    {"stride 16 records 3", 16u, 3u, 0u},
    {"stride 8 records 5 offset 6", 8u, 5u, 6u},
}};

std::uint32_t InitialByte(std::uint32_t address) {
    return (address * 7u + 3u + (address >> 8u)) & 0xffu;
}

std::uint32_t DataWord(std::uint32_t byte) {
    Require(byte + 4u <= DataBytes, "gpu scalar buffer spans: a case reads outside the data region");
    std::uint32_t value = 0;
    for (std::uint32_t at = 0; at < 4u; ++at) value |= InitialByte(byte + at) << (at * 8u);
    return value;
}

std::uint32_t Expected(const Case& test, std::uint32_t operand, std::uint32_t immediate) {
    const std::uint64_t size = test.stride == 0u ? test.records : static_cast<std::uint64_t>(test.stride) * test.records;
    const auto address = operand + immediate;
    const auto byte = address & ~3u;
    return address >= operand && byte + 4ull <= size ? DataWord(byte) : 0u;
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

void Prepare(GuestBlock& guest, const Case& test) {
    auto* bytes = guest.Data();
    std::memset(bytes, 0, BlockBytes);
    const auto data = guest.Address() + DataBase;
    const std::array<std::uint32_t, 5> words{static_cast<std::uint32_t>(data), static_cast<std::uint32_t>((data >> 32u) & 0xffffu) | (test.stride << 16u), test.records, Format | (3u << 28u), test.offset};
    std::memcpy(bytes + TableBase, words.data(), sizeof(words));
    for (std::uint32_t at = 0; at < DataBytes; ++at) bytes[DataBase + at] = static_cast<std::uint8_t>(InitialByte(at));
    std::fill(reinterpret_cast<std::uint32_t*>(bytes + OutputBase), reinterpret_cast<std::uint32_t*>(bytes + OutputBase + Threads * OutputStride), Untouched);
}

ShaderRecompiler::RecompileResult Recompile(std::span<const std::uint32_t> code, std::uint32_t waveSize, GuestBlock& guest, const ShaderRecompiler::SpirvTarget& target) {
    const auto table = guest.Address() + TableBase;
    const auto data = guest.Address() + DataBase;
    const auto output = guest.Address() + OutputBase;
    const std::vector<std::uint32_t> userData{
        0u, 0u, 0u, 0u,
        static_cast<std::uint32_t>(table), static_cast<std::uint32_t>((table >> 32u) & 0xffffu), 1024u, Format | (3u << 28u),
        static_cast<std::uint32_t>(data), static_cast<std::uint32_t>((data >> 32u) & 0xffffu), 0u, 0u,
        static_cast<std::uint32_t>(output), static_cast<std::uint32_t>((output >> 32u) & 0xffffu), Threads * OutputStride, Format | (3u << 28u),
    };
    const std::array<ShaderRecompiler::MemoryRegion, 1> regions{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {waveSize, 0, userData, compute, std::nullopt, std::nullopt, regions},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

void Check(GuestBlock& guest, const Case& test, std::uint32_t variant, const std::string& name) {
    std::vector<std::uint32_t> expected;
    if (variant == 0u) {
        for (const auto& load : Loads) {
            for (std::uint32_t dword = 0; dword < load.dwords; ++dword) expected.push_back(Expected(test, load.sgprOffset ? test.offset : 0u, load.immediate + dword * 4u));
        }
        expected.push_back(0u);
    } else {
        for (const auto dword : SparseDwords) expected.push_back(Expected(test, test.offset, dword * 4u));
    }
    const auto* output = reinterpret_cast<const std::uint32_t*>(guest.Data() + OutputBase);
    for (std::uint32_t lane = 0; lane < Threads; ++lane) {
        for (std::uint32_t dword = 0; dword < OutputStride / 4u; ++dword) {
            const auto want = dword < expected.size() ? expected[dword] : Untouched;
            const auto actual = output[lane * OutputStride / 4u + dword];
            Require(actual == want, "gpu scalar buffer spans: " + name + " program " + std::to_string(variant) + " " + test.name + " lane " + std::to_string(lane) + " output dword " + std::to_string(dword) + " is " + Hex(actual) + ", expected " + Hex(want));
        }
    }
}

void Run(AgcDriver::VulkanDevice& device, GuestBlock& guest, std::uint32_t waveSize, const ShaderRecompiler::SpirvTarget& target, const std::string& name) {
    for (const std::uint32_t variant : {0u, 1u}) {
        const auto code = Code(variant);
        const auto shader = Recompile(code, waveSize, guest, target);
        for (const auto& test : Cases) {
            Prepare(guest, test);
            device.Dispatch(shader, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
            device.WaitIdle();
            Check(guest, test, variant, name);
        }
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock guest;
        const auto native = device->Target();
        const auto split = device->ComputeTarget(32);
        Run(*device, guest, 32, native, "wave32");
        Run(*device, guest, 64, native, "wave64");
        if (split.subgroupSize != native.subgroupSize) Run(*device, guest, 64, split, "wave64 split");
        std::puts("gpu scalar buffer span tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
