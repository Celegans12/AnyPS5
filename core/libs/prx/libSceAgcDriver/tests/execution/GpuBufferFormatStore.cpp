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
#include <iostream>
#include <span>
#include <string>
#include <string_view>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Lanes = 2;
constexpr std::uint32_t LaneWords = 4;
constexpr std::uint32_t Untouched = 0xdeadbeefu;
constexpr std::size_t BlockBytes = 65536;

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "gpu buffer format store: cannot allocate the guest block");
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

    std::span<std::uint32_t, 4> Descriptor() {
        return std::span<std::uint32_t, 4>(reinterpret_cast<std::uint32_t*>(block), 4);
    }

    std::span<std::uint32_t, LaneWords * Lanes> Data() {
        return std::span<std::uint32_t, LaneWords * Lanes>(reinterpret_cast<std::uint32_t*>(block + 256), LaneWords * Lanes);
    }

    std::span<std::uint32_t, LaneWords * Lanes> Output() {
        return std::span<std::uint32_t, LaneWords * Lanes>(reinterpret_cast<std::uint32_t*>(block + 512), LaneWords * Lanes);
    }

private:
    std::uint8_t* block = nullptr;
};

alignas(256) constexpr std::array<std::uint32_t, 11> Code{
    0xe0340000u, 0x80010200u, 0x34020084u, 0xe0381000u, 0x80030401u, 0xbf8c3f70u, 0x7e140502u, 0x7e160503u,
    0xe01c1000u, 0x80020401u, 0xbf810000u,
};

struct Case {
    std::uint32_t records;
    std::uint32_t format;
    std::array<std::uint32_t, LaneWords> input;
    std::array<std::uint32_t, LaneWords * Lanes> expected;
};

constexpr std::uint32_t U = Untouched;
constexpr std::array<Case, 9> Cases{{
    {32u, 77u, {0x3f800000u, 0xc0000000u, 0x3f000000u, 0x00000000u}, {0x3f800000u, 0xc0000000u, 0x3f000000u, 0x00000000u, 0x3f800000u, 0xc0000000u, 0x3f000000u, 0x00000000u}},
    {32u, 75u, {0x00000001u, 0x80000000u, 0xfffffffeu, 0x12345678u}, {0x00000001u, 0x80000000u, 0xfffffffeu, 0x12345678u, 0x00000001u, 0x80000000u, 0xfffffffeu, 0x12345678u}},
    {32u, 74u, {0x3f800000u, 0xc0000000u, 0x3f000000u, 0x40400000u}, {0x3f800000u, 0xc0000000u, 0x3f000000u, U, 0x3f800000u, 0xc0000000u, 0x3f000000u, U}},
    {32u, 71u, {0x3f800000u, 0xc0000000u, 0x3f000000u, 0x00000000u}, {0xc0003c00u, 0x00003800u, U, U, 0xc0003c00u, 0x00003800u, U, U}},
    {32u, 64u, {0x3f800000u, 0xc0000000u, 0x3f000000u, 0x00000000u}, {0x3f800000u, 0xc0000000u, U, U, 0x3f800000u, 0xc0000000u, U, U}},
    {32u, 56u, {0x3f800000u, 0x00000000u, 0x3f800000u, 0x00000000u}, {0x00ff00ffu, U, U, U, 0x00ff00ffu, U, U, U}},
    {32u, 60u, {0x00000001u, 0x00000002u, 0x00000003u, 0x000000ffu}, {0xff030201u, U, U, U, 0xff030201u, U, U, U}},
    {32u, 22u, {0x40490fdbu, 0x3f800000u, 0x3f800000u, 0x3f800000u}, {0x40490fdbu, U, U, U, 0x40490fdbu, U, U, U}},
    {16u, 77u, {0x3f800000u, 0xc0000000u, 0x3f000000u, 0x00000000u}, {0x3f800000u, 0xc0000000u, 0x3f000000u, 0x00000000u, U, U, U, U}},
}};

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

ShaderRecompiler::RecompileResult Compile(AgcDriver::VulkanDevice& device, GuestBlock& guest, std::uint32_t waveSize) {
    const auto descriptor = reinterpret_cast<std::uintptr_t>(guest.Descriptor().data());
    const auto data = reinterpret_cast<std::uintptr_t>(guest.Data().data());
    const auto output = reinterpret_cast<std::uintptr_t>(guest.Output().data());
    const std::array<std::uint32_t, 16> userData{
        0u, 0u, 0u, 0u,
        static_cast<std::uint32_t>(descriptor), static_cast<std::uint32_t>((descriptor >> 32u) & 0xffffu), static_cast<std::uint32_t>(guest.Descriptor().size_bytes()), 0x31016facu,
        static_cast<std::uint32_t>(output), static_cast<std::uint32_t>((output >> 32u) & 0xffffu), 0u, 0u,
        static_cast<std::uint32_t>(data), static_cast<std::uint32_t>((data >> 32u) & 0xffffu), static_cast<std::uint32_t>(guest.Data().size_bytes()), 0x31016facu,
    };
    const std::span<const std::uint32_t> code(Code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {waveSize, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

void Run(AgcDriver::VulkanDevice& device, const ShaderRecompiler::RecompileResult& shader, GuestBlock& guest, const Case& test) {
    auto descriptor = guest.Descriptor();
    descriptor[0] = test.records;
    descriptor[1] = 0x31000facu | (test.format << 12u);
    descriptor[2] = 0u;
    descriptor[3] = 0u;
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) std::copy(test.input.begin(), test.input.end(), guest.Data().begin() + lane * LaneWords);
    std::fill(guest.Output().begin(), guest.Output().end(), Untouched);
    device.Dispatch(shader, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(Code.data()));
    device.WaitIdle();
}

}

int main(int argc, char** argv) {
    try {
        const std::uint32_t waveSize = argc == 2 && std::string_view(argv[1]) == "--wave64" ? 64u : 32u;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock guest;
        const auto shader = Compile(*device, guest, waveSize);
        for (const auto& test : Cases) {
            Run(*device, shader, guest, test);
            for (std::uint32_t index = 0; index < LaneWords * Lanes; ++index) {
                const auto actual = guest.Output()[index];
                const auto expected = test.expected[index];
                Require(actual == expected, "gpu buffer format store: wave" + std::to_string(waveSize) + " format " + std::to_string(test.format) + " records " + std::to_string(test.records) + " lane " + std::to_string(index / LaneWords) + " dword " + std::to_string(index % LaneWords) + " is " + Hex(actual) + ", expected " + Hex(expected));
            }
        }
        std::printf("gpu buffer format store tests passed (wave%u)\n", waveSize);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
