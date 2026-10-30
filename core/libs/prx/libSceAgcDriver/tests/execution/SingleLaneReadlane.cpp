// agc_driver_single_lane_readlane_tests: a wave64 workgroup of two waves (128 threads) reads lanes 63 and 31 of its own
// wave with v_readlane_b32 and stores them per thread (the cloth programs' reduction tail: DPP row shifts, then
// v_readlane 31 and 63). Every thread must store 63 and 31. Run in both layouts: two guest lanes per invocation (the
// default) and one lane per invocation (SpirvTarget::singleLaneGroups), whose host subgroups are 32 lanes wide on
// NVIDIA: a lane-63 read inside one host subgroup has no lane 63 to read, so on such a host one lane per invocation spans
// each wave over two host subgroups (with APS5_SPANNED_WAVE=1) or keeps two lanes per invocation. Each thread also reads
// the LDS word its mirrored lane (lane ^ 63, the other half) wrote without a barrier, and the lane count of a 64-bit
// compare mask (40). Also run with APS5_SINGLE_LANE=all. Exit 1 when a layout stores a wrong value.
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

namespace {

constexpr std::uint32_t Threads = 128;
constexpr std::uint32_t Raw = 3u << 28u;
constexpr std::uint32_t Selects = 0xfacu;
constexpr std::uint32_t Float32 = 22u;

alignas(256) std::array<std::uint32_t, 64> Program{};

std::span<const std::uint32_t> Code() {
    std::vector<std::uint32_t> code;
    code.push_back((0x1bu << 25u) | (1u << 17u) | (0u << 9u) | 191u);   // v_and_b32 v1, 63, v0 (lane in wave)
    code.push_back(0xd7600009u); code.push_back((191u << 9u) | 257u);   // v_readlane_b32 s9, v1, 63
    code.push_back(0xd760000au); code.push_back((159u << 9u) | 257u);   // v_readlane_b32 s10, v1, 31
    code.push_back((0x3fu << 25u) | (2u << 17u) | (1u << 9u) | 9u);     // v_mov_b32 v2, s9
    code.push_back((0x3fu << 25u) | (4u << 17u) | (1u << 9u) | 10u);    // v_mov_b32 v4, s10
    code.push_back((0x1au << 25u) | (3u << 17u) | (0u << 9u) | 130u);   // v_lshlrev_b32 v3, 2, v0
    code.push_back(0xe0000000u | (0x1cu << 18u) | 0x1000u);             // buffer_store_dword v2, v3, s[4:7], 0 offen
    code.push_back((0x80u << 24u) | (1u << 16u) | (2u << 8u) | 3u);
    code.push_back(0xe0000000u | (0x1cu << 18u) | 0x1000u | 512u);      // buffer_store_dword v4, v3, s[4:7], 0 offen offset:512
    code.push_back((0x80u << 24u) | (1u << 16u) | (4u << 8u) | 3u);
    code.push_back((0x1du << 25u) | (6u << 17u) | (0u << 9u) | 191u);   // v_xor_b32 v6, 63, v0 (mirrored lane of the wave)
    code.push_back((0x1au << 25u) | (7u << 17u) | (6u << 9u) | 130u);   // v_lshlrev_b32 v7, 2, v6
    code.push_back(0xd8340000u); code.push_back(0x00000003u);           // ds_write_b32 v3, v0
    code.push_back(0xbf8c0000u);                                        // s_waitcnt 0
    code.push_back(0xd8d80000u); code.push_back(0x08000007u);           // ds_read_b32 v8, v7 (the other half's write, no barrier)
    code.push_back(0xbf8c0000u);                                        // s_waitcnt 0
    code.push_back(0xe0000000u | (0x1cu << 18u) | 0x1000u | 1024u);     // buffer_store_dword v8, v3, s[4:7], 0 offen offset:1024
    code.push_back((0x80u << 24u) | (1u << 16u) | (8u << 8u) | 3u);
    code.push_back(0x7d8802a8u);                                        // v_cmp_gt_u32 vcc, 40, v1 (lanes 0-39)
    code.push_back(0xbe8b106au);                                        // s_bcnt1_i32_b64 s11, vcc
    code.push_back((0x3fu << 25u) | (9u << 17u) | (1u << 9u) | 11u);    // v_mov_b32 v9, s11
    code.push_back(0xe0000000u | (0x1cu << 18u) | 0x1000u | 1536u);     // buffer_store_dword v9, v3, s[4:7], 0 offen offset:1536
    code.push_back((0x80u << 24u) | (1u << 16u) | (9u << 8u) | 3u);
    code.push_back(0xbf8c0000u);                                        // s_waitcnt 0
    code.push_back(0xbf810000u);                                        // s_endpgm
    std::copy(code.begin(), code.end(), Program.begin());
    return std::span<const std::uint32_t>(Program.data(), code.size());
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        constexpr std::size_t bytes = 4096;
        auto* memory = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, bytes * 2u, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        auto* input = memory;
        auto* output = memory + bytes / 4u;
        GuestAllocations::Mutation().Add(memory, bytes * 2u, true, true, true);
        const auto code = Code();
        const auto in = reinterpret_cast<std::uint64_t>(input);
        const auto out = reinterpret_cast<std::uint64_t>(output);
        const std::vector<std::uint32_t> userData{
            static_cast<std::uint32_t>(in), static_cast<std::uint32_t>((in >> 32u) & 0xffffu), static_cast<std::uint32_t>(bytes), Selects | (Float32 << 12u) | Raw,
            static_cast<std::uint32_t>(out), static_cast<std::uint32_t>((out >> 32u) & 0xffffu), static_cast<std::uint32_t>(bytes), Selects | (Float32 << 12u) | Raw,
        };
        const std::array<ShaderRecompiler::MemoryRegion, 1> regions{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
        const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, Threads, {true, false, false}, false, 1};
        int failures = 0;
        for (const bool oneLane : {false, true}) {
            std::memset(output, 0xff, bytes);
            ShaderRecompiler::RecompileRequest request{
                {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
                {64u, 0, userData, compute, std::nullopt, std::nullopt, regions},
                device->ComputeTarget(64),
                {0, 0, 0, 128}
            };
            request.target.singleLaneGroups = oneLane;
            request.useCache = false;
            const auto shader = ShaderRecompiler::Recompile(request);
            device->Dispatch(shader, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
            device->WaitIdle();
            std::uint32_t wrong = 0;
            std::uint32_t first = ~0u;
            for (std::uint32_t thread = 0; thread < Threads; ++thread) {
                const bool bad = output[thread] != 63u || output[128u + thread] != 31u || output[256u + thread] != (thread ^ 63u) || output[384u + thread] != 40u;
                if (bad && first == ~0u) first = thread;
                wrong += bad;
            }
            const bool spanned = (oneLane || std::getenv("APS5_SINGLE_LANE") != nullptr) && std::getenv("APS5_SPANNED_GROUPS") != nullptr && request.target.spannedWaves && request.target.subgroupSize == 32u;
            std::printf("%s (host subgroup %u, %s): %u of %u threads wrong", oneLane ? "one lane per invocation" : "two lanes per invocation", shader.hostSubgroupSize, shader.fullSubgroups ? "spanned waves" : "two lanes per invocation", wrong, Threads);
            if (wrong != 0u) std::printf("; first thread %u stored lane63=%u lane31=%u mirrored=%u count=%u", first, output[first], output[128u + first], output[256u + first], output[384u + first]);
            if (spanned != shader.fullSubgroups) std::printf("; the layout should %sbe spanned", spanned ? "" : "not ");
            std::printf("\n");
            failures += wrong != 0u || spanned != shader.fullSubgroups;
        }
        std::fflush(stdout);
        std::fflush(nullptr);
        TerminateProcess(GetCurrentProcess(), failures == 0 ? 0 : 1);
        std::_Exit(failures == 0 ? 0 : 1);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
