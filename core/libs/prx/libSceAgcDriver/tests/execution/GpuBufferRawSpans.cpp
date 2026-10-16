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
#include <initializer_list>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 64;
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t TableBase = 0x0;
constexpr std::uint32_t IndexTable = 32;
constexpr std::uint32_t OffsetTable = 32 + 4 * Threads;
constexpr std::uint32_t DataBase = 0x1000;
constexpr std::uint32_t DataBytes = 0x4000;
constexpr std::uint32_t OutputBase = 0x8000;
constexpr std::uint32_t OutputStride = 64;
constexpr std::uint32_t Untouched = 0xdeadbeefu;
constexpr std::uint32_t Format = 0x00016facu;

struct Load {
    std::uint32_t dwords;
    std::uint32_t immediate;
    std::uint32_t output;
};
constexpr std::array<Load, 4> Loads{{{4u, 0u, 0u}, {3u, 4u, 16u}, {2u, 8u, 28u}, {1u, 12u, 36u}}};

void Mubuf(std::uint32_t op, bool idxen, bool offen, std::uint32_t immediate, std::uint32_t vaddr, std::uint32_t vdata, std::uint32_t resource, std::uint32_t soffset, std::vector<std::uint32_t>& code) {
    code.push_back(0xe0000000u | (op << 18u) | (idxen ? 0x2000u : 0u) | (offen ? 0x1000u : 0u) | immediate);
    code.push_back((soffset << 24u) | (resource / 4u << 16u) | (vdata << 8u) | vaddr);
}

void Append(std::vector<std::uint32_t>& code, std::initializer_list<std::uint32_t> words) {
    for (const auto word : words) code.push_back(word);
}

std::uint32_t StoreOp(std::uint32_t dwords) {
    return std::array<std::uint32_t, 4>{28u, 29u, 31u, 30u}[dwords - 1u];
}

std::uint32_t LoadOp(std::uint32_t dwords) {
    return std::array<std::uint32_t, 4>{12u, 13u, 15u, 14u}[dwords - 1u];
}

alignas(256) std::array<std::array<std::uint32_t, 96>, 32> Programs{};
std::uint32_t ProgramCount = 0;

std::span<const std::uint32_t> Code(bool barrier, std::uint32_t storeDwords) {
    std::vector<std::uint32_t> code{0x34060082u};
    Mubuf(14u, false, false, 4u, 0u, 4u, 4u, 0x80u, code);
    Mubuf(12u, false, true, IndexTable, 3u, 1u, 4u, 0x80u, code);
    Mubuf(12u, false, true, OffsetTable, 3u, 2u, 4u, 0x80u, code);
    code.push_back(0xbf8c0000u);
    Append(code, {0x7e120504u, 0x7e140505u, 0x7e160506u, 0x7e040507u});
    if (barrier) code.push_back(0xbf8a0000u);
    if (storeDwords == 0u) {
        for (const auto& load : Loads) Mubuf(LoadOp(load.dwords), true, true, load.immediate, 1u, 8u + load.output / 4u, 8u, 2u, code);
        Append(code, {0xbf8c3f70u, 0x34280086u});
        for (const auto& load : Loads) Mubuf(StoreOp(load.dwords), false, true, load.output, 20u, 8u + load.output / 4u, 12u, 0x80u, code);
    } else {
        Append(code, {0x34100088u, 0x381010ffu, 0xa0000000u, 0x38121081u, 0x38141082u, 0x38161083u});
        Mubuf(StoreOp(storeDwords), true, true, 4u, 1u, 8u, 8u, 2u, code);
    }
    code.push_back(0xbf8c3f70u);
    code.push_back(0xbf810000u);
    auto& program = Programs.at(ProgramCount++);
    Require(code.size() <= program.size(), "gpu buffer raw spans: program too long");
    for (std::size_t index = 0; index < code.size(); ++index) program.at(index) = code[index];
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
        Require(block != nullptr, "gpu buffer raw spans: cannot allocate the guest block");
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
    bool swizzle;
    std::uint32_t records;
    std::uint32_t word3;
    std::uint32_t soffset;
    bool storeUnaligned;
};

constexpr std::uint32_t Raw = 3u << 28u;
constexpr std::uint32_t Structured = 0u;
constexpr std::uint32_t IndexOnly = 1u << 28u;
constexpr std::uint32_t RecordsOnly = 2u << 28u;

constexpr std::array<Case, 18> Cases{{
    {"raw records 37", 0u, false, 37u, Format | Raw, 0u, true},
    {"raw records 37 soffset 6", 0u, false, 37u, Format | Raw, 6u, true},
    {"raw records 3", 0u, false, 3u, Format | Raw, 0u, true},
    {"raw soffset past records", 0u, false, 37u, Format | Raw, 50u, true},
    {"raw records 4096", 0u, false, 4096u, Format | Raw, 0u, true},
    {"raw records 4096 soffset 1", 0u, false, 4096u, Format | Raw, 1u, true},
    {"raw stride 64 records 22", 64u, false, 22u, Format | Raw, 0u, true},
    {"raw stride 64 records 22 soffset 3", 64u, false, 22u, Format | Raw, 3u, true},
    {"raw swizzled stride 16 records 3", 16u, true, 3u, Format | Raw, 0u, false},
    {"raw swizzled stride 24 records 40 index stride 64", 24u, true, 40u, Format | Raw | (3u << 21u), 4u, false},
    {"structured stride 20 records 3", 20u, false, 3u, Format | Structured, 0u, false},
    {"structured stride 24 records 50", 24u, false, 50u, Format | Structured, 8u, false},
    {"structured stride 0", 0u, false, 50u, Format | Structured, 0u, false},
    {"index only stride 64 records 2", 64u, false, 2u, Format | IndexOnly, 0u, true},
    {"index only stride 64 records 40", 64u, false, 40u, Format | IndexOnly, 2u, true},
    {"records only stride 64 records 1", 64u, false, 1u, Format | RecordsOnly, 0u, true},
    {"records only records 0", 64u, false, 0u, Format | RecordsOnly, 0u, true},
    {"no data format", 0u, false, 4096u, (Format & ~0x7f000u) | Raw, 0u, true},
}};

std::uint32_t InitialByte(std::uint32_t address) {
    return (address * 7u + 3u + (address >> 8u)) & 0xffu;
}

struct Bounds {
    bool inBounds;
    std::uint32_t byte;
};

Bounds Model(const Case& test, std::uint32_t index, std::uint32_t offset) {
    const auto fits = [](std::uint32_t at, std::uint32_t size) { return size >= 4u && at <= size - 4u; };
    const bool swizzle = test.stride != 0u && test.swizzle;
    const std::uint32_t mode = (test.word3 >> 28u) & 3u;
    const std::uint32_t shift = ((test.word3 >> 21u) & 3u) + 3u;
    const std::uint32_t indices = 1u << shift;
    const std::uint32_t swizzled = ((index >> shift) * test.stride + (offset & ~3u)) * indices + ((index & (indices - 1u)) << 2u) + (offset & 3u);
    const std::uint32_t byte = (swizzle ? swizzled : index * test.stride + offset) + test.soffset;
    const bool indexInBounds = index < test.records;
    bool result = false;
    if (mode == 0u) result = indexInBounds && offset < test.stride;
    if (mode == 1u) result = indexInBounds;
    if (mode == 2u) result = test.records != 0u;
    if (mode == 3u) result = test.soffset <= test.records && (swizzle ? index < test.records - test.soffset && fits(offset, test.stride) : fits(offset, test.records - test.soffset));
    const bool valid = ((test.word3 >> 30u) & 3u) == 0u && ((test.word3 >> 12u) & 0x7fu) != 0u;
    return {valid && result, byte};
}

std::uint32_t LaneIndex(bool store, std::uint32_t lane) {
    return store ? lane : lane >> 4u;
}

std::uint32_t LaneOffset(const Case& test, bool store, std::uint32_t lane) {
    if (!store) return (lane & 15u) * 2u;
    if (test.stride == 0u) return lane * 20u + (test.storeUnaligned && (lane & 1u) != 0u ? 2u : 0u);
    return (lane & 7u) * 4u + (test.storeUnaligned && (lane & 8u) != 0u ? 2u : 0u);
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

void Prepare(GuestBlock& guest, const Case& test, bool store) {
    auto* bytes = guest.Data();
    std::memset(bytes, 0, BlockBytes);
    const auto data = guest.Address() + DataBase;
    const std::array<std::uint32_t, 5> words{static_cast<std::uint32_t>(data), static_cast<std::uint32_t>((data >> 32u) & 0xffffu) | (test.stride << 16u) | (test.swizzle ? 0x80000000u : 0u), test.records, test.word3, test.soffset};
    std::memcpy(bytes + TableBase, words.data(), sizeof(words));
    for (std::uint32_t lane = 0; lane < Threads; ++lane) {
        const auto index = LaneIndex(store, lane);
        const auto offset = LaneOffset(test, store, lane);
        std::memcpy(bytes + IndexTable + lane * 4u, &index, 4u);
        std::memcpy(bytes + OffsetTable + lane * 4u, &offset, 4u);
    }
    for (std::uint32_t at = 0; at < DataBytes; ++at) bytes[DataBase + at] = static_cast<std::uint8_t>(InitialByte(at));
    std::fill(reinterpret_cast<std::uint32_t*>(bytes + OutputBase), reinterpret_cast<std::uint32_t*>(bytes + OutputBase + Threads * OutputStride), Untouched);
}

std::uint32_t DataWord(std::uint32_t byte) {
    Require(byte + 4u <= DataBytes, "gpu buffer raw spans: a case reads outside the data region");
    std::uint32_t value = 0;
    for (std::uint32_t at = 0; at < 4u; ++at) value |= InitialByte(byte + at) << (at * 8u);
    return value;
}

ShaderRecompiler::RecompileResult Recompile(std::span<const std::uint32_t> code, std::uint32_t waveSize, GuestBlock& guest, const ShaderRecompiler::SpirvTarget& target) {
    const auto table = guest.Address() + TableBase;
    const auto data = guest.Address() + DataBase;
    const auto output = guest.Address() + OutputBase;
    const std::vector<std::uint32_t> userData{
        0u, 0u, 0u, 0u,
        static_cast<std::uint32_t>(table), static_cast<std::uint32_t>((table >> 32u) & 0xffffu), 1024u, Format | Raw,
        static_cast<std::uint32_t>(data), static_cast<std::uint32_t>((data >> 32u) & 0xffffu), 0u, 0u,
        static_cast<std::uint32_t>(output), static_cast<std::uint32_t>((output >> 32u) & 0xffffu), Threads * OutputStride, Format | Raw,
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

void CheckLoads(GuestBlock& guest, const Case& test, const std::string& name) {
    const auto* output = reinterpret_cast<const std::uint32_t*>(guest.Data() + OutputBase);
    for (std::uint32_t lane = 0; lane < Threads; ++lane) {
        for (std::uint32_t dword = 0; dword < OutputStride / 4u; ++dword) {
            std::uint32_t expected = Untouched;
            for (const auto& load : Loads) {
                if (dword < load.output / 4u || dword >= load.output / 4u + load.dwords) continue;
                const auto bounds = Model(test, LaneIndex(false, lane), LaneOffset(test, false, lane) + load.immediate + (dword - load.output / 4u) * 4u);
                expected = bounds.inBounds ? DataWord(bounds.byte) : 0u;
            }
            const auto actual = output[lane * OutputStride / 4u + dword];
            Require(actual == expected, "gpu buffer raw spans: " + name + " " + test.name + " lane " + std::to_string(lane) + " output dword " + std::to_string(dword) + " is " + Hex(actual) + ", expected " + Hex(expected));
        }
    }
}

void CheckStores(GuestBlock& guest, const Case& test, std::uint32_t dwords, const std::string& name) {
    std::vector<std::uint8_t> expected(DataBytes);
    std::vector<std::uint8_t> written(DataBytes, 0u);
    for (std::uint32_t at = 0; at < DataBytes; ++at) expected[at] = static_cast<std::uint8_t>(InitialByte(at));
    for (std::uint32_t lane = 0; lane < Threads; ++lane) {
        for (std::uint32_t dword = 0; dword < dwords; ++dword) {
            const auto bounds = Model(test, LaneIndex(true, lane), LaneOffset(test, true, lane) + 4u + dword * 4u);
            if (!bounds.inBounds) continue;
            const std::uint32_t value = 0xa0000000u | (lane << 8u) | dword;
            for (std::uint32_t at = 0; at < 4u; ++at) {
                Require(bounds.byte + at < DataBytes && written[bounds.byte + at] == 0u, "gpu buffer raw spans: store case " + std::string(test.name) + " has overlapping lanes");
                written[bounds.byte + at] = 1u;
                expected[bounds.byte + at] = static_cast<std::uint8_t>(value >> (at * 8u));
            }
        }
    }
    const auto* data = guest.Data() + DataBase;
    for (std::uint32_t at = 0; at < DataBytes; ++at) {
        Require(data[at] == expected[at], "gpu buffer raw spans: " + name + " " + test.name + " x" + std::to_string(dwords) + " store byte " + std::to_string(at) + " is " + Hex(data[at]) + ", expected " + Hex(expected[at]));
    }
}

void Run(AgcDriver::VulkanDevice& device, GuestBlock& guest, std::uint32_t waveSize, const ShaderRecompiler::SpirvTarget& target, const std::string& name) {
    for (const bool barrier : {false, true}) {
        for (const std::uint32_t storeDwords : {0u, 1u, 2u, 3u, 4u}) {
            const auto code = Code(barrier, storeDwords);
            const auto shader = Recompile(code, waveSize, guest, target);
            const auto label = name + (barrier ? " barrier" : "");
            for (const auto& test : Cases) {
                Prepare(guest, test, storeDwords != 0u);
                device.Dispatch(shader, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
                device.WaitIdle();
                if (storeDwords == 0u) CheckLoads(guest, test, label);
                else CheckStores(guest, test, storeDwords, label);
            }
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
        std::puts("gpu buffer raw span tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
