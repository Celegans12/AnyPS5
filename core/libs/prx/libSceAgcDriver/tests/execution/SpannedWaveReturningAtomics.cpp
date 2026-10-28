// agc_driver_spanned_wave_returning_atomics_tests [runs]: one wave64 workgroup (64 threads), every lane does a returning
// buffer_atomic_add of 1 on one counter and stores what it got back (S4 epoch 1, H2-5; newpath/s0/append-order.md §4
// test (b)). Run in both layouts: two guest lanes per invocation (no span: half 0 adds before half 1 in each
// invocation) and a spanned wave (SpirvTarget::spannedWaves: the wave over two 32-lane host subgroups). On the console
// one wave's lanes never race each other, so the lower 32 lanes must get 0..31 and the upper 32 lanes 32..63 in every
// run, and every run must give the same per-lane results. Exit 1 when a layout breaks either rule.
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
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

namespace {

constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t Raw = 3u << 28u;
constexpr std::uint32_t Selects = 0xfacu;
constexpr std::uint32_t Float32 = 22u;

alignas(256) std::array<std::uint32_t, 128> Program{};

std::span<const std::uint32_t> Code() {
    std::vector<std::uint32_t> code;
    code.push_back((0x1au << 25u) | (3u << 17u) | (0u << 9u) | 130u);   // v_lshlrev_b32 v3, 2, v0
    // Delay the lower half: its lanes chase 8 dependent loads through 32 distinct pages (the upper half's all hit one
    // address), so without ordering the upper subgroup tends to reach the atomic first. The loads read zeros, so the
    // added value stays 1.
    code.push_back((0x1au << 25u) | (4u << 17u) | (0u << 9u) | 140u);   // v_lshlrev_b32 v4, 12, v0
    code.push_back(0x7c000000u | (0xc4u << 17u) | (0u << 9u) | 160u);   // v_cmp_gt_u32 vcc, 32, v0
    code.push_back((0x01u << 25u) | (4u << 17u) | (4u << 9u) | 128u);   // v_cndmask_b32 v4, 0, v4, vcc
    for (int load = 0; load < 8; ++load) {
        code.push_back(0xe0000000u | (0x0cu << 18u) | 0x1000u);         // buffer_load_dword v5, v4, s[8:11], 0 offen
        code.push_back((0x80u << 24u) | (2u << 16u) | (5u << 8u) | 4u);
        code.push_back(0xbf8c0000u);                                    // s_waitcnt 0
        code.push_back((0x25u << 25u) | (4u << 17u) | (5u << 9u) | 260u); // v_add_nc_u32 v4, v4, v5
    }
    code.push_back((0x25u << 25u) | (2u << 17u) | (5u << 9u) | 129u);   // v_add_nc_u32 v2, 1, v5
    code.push_back(0xe0000000u | (0x32u << 18u) | (1u << 14u));         // buffer_atomic_add v2, off, s[0:3], 0 glc
    code.push_back((0x80u << 24u) | (0u << 16u) | (2u << 8u) | 0u);
    code.push_back(0xbf8c0000u);                                        // s_waitcnt 0
    code.push_back(0xe0000000u | (0x1cu << 18u) | 0x1000u);             // buffer_store_dword v2, v3, s[4:7], 0 offen
    code.push_back((0x80u << 24u) | (1u << 16u) | (2u << 8u) | 3u);
    code.push_back(0xbf8c0000u);                                        // s_waitcnt 0
    code.push_back(0xbf810000u);                                        // s_endpgm
    std::copy(code.begin(), code.end(), Program.begin());
    return std::span<const std::uint32_t>(Program.data(), code.size());
}

}

int main(int argc, char** argv) {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const std::uint32_t runs = argc > 1 ? static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 100u;
        constexpr std::size_t bytes = 4096;
        constexpr std::size_t delayBytes = 32u * 4096u;
        auto* memory = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, bytes * 2u + delayBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        auto* counter = memory;
        auto* output = memory + bytes / 4u;
        auto* delay = memory + bytes / 2u;
        GuestAllocations::Mutation().Add(memory, bytes * 2u + delayBytes, true, true, true);
        const auto pages = reinterpret_cast<std::uint64_t>(delay);
        const auto code = Code();
        const auto in = reinterpret_cast<std::uint64_t>(counter);
        const auto out = reinterpret_cast<std::uint64_t>(output);
        const std::vector<std::uint32_t> userData{
            static_cast<std::uint32_t>(in), static_cast<std::uint32_t>((in >> 32u) & 0xffffu), 4u, Selects | (Float32 << 12u) | Raw,
            static_cast<std::uint32_t>(out), static_cast<std::uint32_t>((out >> 32u) & 0xffffu), static_cast<std::uint32_t>(bytes), Selects | (Float32 << 12u) | Raw,
            static_cast<std::uint32_t>(pages), static_cast<std::uint32_t>((pages >> 32u) & 0xffffu), static_cast<std::uint32_t>(delayBytes), Selects | (Float32 << 12u) | Raw,
        };
        const std::array<ShaderRecompiler::MemoryRegion, 1> regions{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
        const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {true, false, false}, false, 1};
        int failures = 0;
        for (const bool spanned : {false, true}) {
            ShaderRecompiler::RecompileRequest request{
                {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
                {64u, 0, userData, compute, std::nullopt, std::nullopt, regions},
                device->ComputeTarget(64),
                {0, 0, 0, 128}
            };
            request.target.spannedWaves = spanned;
            request.useCache = false;
            const auto shader = ShaderRecompiler::Recompile(request);
            if (spanned && !shader.fullSubgroups) {
                std::printf("spanned wave: this host does not span (host subgroup %u, full subgroups off); skipped\n", shader.hostSubgroupSize);
                continue;
            }
            std::array<std::uint32_t, Threads> first{};
            std::uint32_t splitWrong = 0, differing = 0, laneOrder = 0, firstBad = ~0u;
            for (std::uint32_t run = 0; run < runs; ++run) {
                counter[0] = 0;
                std::memset(output, 0xff, Threads * 4u);
                device->Dispatch(shader, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
                device->WaitIdle();
                bool split = counter[0] == Threads;
                bool ordered = true;
                for (std::uint32_t lane = 0; lane < Threads; ++lane) {
                    split = split && (output[lane] < 32u) == (lane < 32u);
                    ordered = ordered && output[lane] == lane;
                }
                if (!split) {
                    ++splitWrong;
                    if (firstBad == ~0u) firstBad = run;
                }
                laneOrder += ordered;
                if (run == 0) std::copy(output, output + Threads, first.begin());
                else differing += !std::equal(first.begin(), first.end(), output);
            }
            std::printf("%s (host subgroup %u): %u of %u runs break the half split, %u differ from run 0, %u in exact lane order",
                spanned ? "spanned wave" : "two lanes per invocation", shader.hostSubgroupSize, splitWrong, runs, differing, laneOrder);
            if (firstBad != ~0u) std::printf("; first bad run %u", firstBad);
            std::printf("\n");
            failures += splitWrong != 0u || differing != 0u;
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
