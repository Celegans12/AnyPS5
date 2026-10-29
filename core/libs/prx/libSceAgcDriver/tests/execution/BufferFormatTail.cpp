// Format loads of the last bytes of a buffer whose num_records is not a multiple of 4 (8/16-bit formats, raw
// buffers): the hardware range-checks the element, so element bytes below num_records read memory and only an
// element past it reads 0. Through a V# in user data (the bound-buffer path) and through one whose records and
// format word come from memory at run time (the device-side V# path).
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
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Lanes = 32;
constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t DataOffset = 256;
constexpr std::size_t DataBytes = 64;
constexpr std::size_t OutputOffset = 512;
constexpr std::uint32_t Untouched = 0xdeadbeefu;

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "buffer format tail: cannot allocate the guest block");
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

    std::uint32_t* Descriptor() { return reinterpret_cast<std::uint32_t*>(block); }
    std::uint8_t* Data() { return block + DataOffset; }
    std::uint32_t* Output() { return reinterpret_cast<std::uint32_t*>(block + OutputOffset); }

private:
    std::uint8_t* block = nullptr;
};

// v1 = lane * 4, v2 = lane * 2; buffer_load_format_x v4, (v0 | v2), s[8:11], 0 offen; buffer_store_dword v4, v1,
// s[12:15], 0 offen. The runtime form first loads records and the format word (s10, s11) from s[4:7].
std::vector<std::uint32_t> Program(bool runtime, std::uint32_t elementBytes) {
    std::vector<std::uint32_t> code;
    if (runtime) code.insert(code.end(), {0xe0340000u, 0x80010500u});
    code.insert(code.end(), {0x34020082u, 0x34040081u});
    if (runtime) code.insert(code.end(), {0xbf8c3f70u, 0x7e140505u, 0x7e160506u});
    code.insert(code.end(), {0xe0001000u, 0x80020400u | (elementBytes == 2u ? 2u : 0u), 0xbf8c3f70u, 0xe0701000u, 0x80030401u, 0xbf810000u});
    return code;
}

struct Format {
    const char* name;
    std::uint32_t format;
    std::uint32_t bytes;
    bool sign;
};

constexpr std::array<Format, 4> Formats{{
    {"8_UINT", 5u, 1u, false},
    {"8_SINT", 6u, 1u, true},
    {"16_UINT", 11u, 2u, false},
    {"16_SINT", 12u, 2u, true},
}};

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

void Run(AgcDriver::VulkanDevice& device, GuestBlock& guest, bool runtime) {
    const std::string path = runtime ? "runtime V#" : "user-data V#";
    const auto address = [](const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); };
    for (const auto& format : Formats) {
        static std::array<std::vector<std::uint32_t>, 4> programs;
        auto& program = programs[(runtime ? 2u : 0u) + (format.bytes == 2u ? 1u : 0u)];
        program = Program(runtime, format.bytes);
        const std::span<const std::uint32_t> code(program);
        for (const std::uint32_t base : {0u, 1u, 2u, 3u}) {
            if (base % format.bytes != 0u) continue;
            for (const std::uint32_t records : {16u, 17u, 18u, 19u, 20u}) {
                const auto word3 = 0x31000facu | (format.format << 12u);
                guest.Descriptor()[0] = records;
                guest.Descriptor()[1] = word3;
                for (std::size_t index = 0; index < DataBytes; ++index) guest.Data()[index] = static_cast<std::uint8_t>(0x91u + index * 37u);
                std::fill(guest.Output(), guest.Output() + Lanes, Untouched);
                const auto data = address(guest.Data() + base);
                const auto output = address(guest.Output());
                const auto table = address(guest.Descriptor());
                const std::array<std::uint32_t, 16> userData{
                    0u, 0u, 0u, 0u,
                    static_cast<std::uint32_t>(table), static_cast<std::uint32_t>((table >> 32u) & 0xffffu), 16u, 0x31016facu,
                    static_cast<std::uint32_t>(data), static_cast<std::uint32_t>((data >> 32u) & 0xffffu), runtime ? 0u : records, runtime ? 0u : word3,
                    static_cast<std::uint32_t>(output), static_cast<std::uint32_t>((output >> 32u) & 0xffffu), Lanes * 4u, 0x31016facu,
                };
                const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{address(code.data()), std::as_bytes(code)}}};
                const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {false, false, false}, false, 1};
                ShaderRecompiler::RecompileRequest request{
                    {ShaderStage::Compute, address(code.data()), code, 0, {}},
                    {Lanes, 0, userData, compute, std::nullopt, std::nullopt, memory},
                    device.Target(),
                    {0, 0, 0, 128}
                };
                request.useCache = false;
                const auto result = ShaderRecompiler::Recompile(request);
                device.Dispatch(result, 1, 1, 1, {}, address(code.data()));
                device.WaitIdle();
                for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
                    const auto offset = lane * format.bytes;
                    std::uint32_t expected = 0;
                    if (offset + format.bytes <= records) {
                        expected = guest.Data()[base + offset];
                        if (format.bytes == 2u) expected |= static_cast<std::uint32_t>(guest.Data()[base + offset + 1u]) << 8u;
                        if (format.sign) expected = format.bytes == 1u ? static_cast<std::uint32_t>(static_cast<std::int32_t>(static_cast<std::int8_t>(expected))) : static_cast<std::uint32_t>(static_cast<std::int32_t>(static_cast<std::int16_t>(expected)));
                    }
                    const auto actual = guest.Output()[lane];
                    Require(actual == expected, "buffer format tail, " + path + ", " + format.name + " base +" + std::to_string(base) + " records " + std::to_string(records) + ": lane " + std::to_string(lane) + " (byte " + std::to_string(offset) + ") is " + Hex(actual) + ", expected " + Hex(expected));
                }
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
        int failures = 0;
        for (const bool runtime : {false, true}) {
            try {
                Run(*device, guest, runtime);
            } catch (const std::exception& error) {
                std::cerr << error.what() << '\n';
                ++failures;
            }
        }
        if (failures != 0) return 1;
        std::puts("buffer format tail tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
