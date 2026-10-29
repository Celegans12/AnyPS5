// agc_driver_ordered_workgroup_waves_tests [runs]: one wave32 workgroup of 64 threads (two console waves), every lane
// does a returning buffer_atomic_add of 1 on one counter and stores what it got back (S4 epoch 1, H3-1 region A: the
// two waves of program 8000227600 reach their allocating atomic in either order). Wave 0's lanes are delayed by 8
// dependent loads through 32 distinct pages (indexed by v_mbcnt_lo of EXEC, a wave operation like the title's, so a
// BDA fault does not stop invocations and the ordering applies), so without ordering wave 1 reaches the atomic first. On the
// console the order between the waves of a workgroup is not fixed either, but an exact replay needs one order in
// every run: with SpirvTarget::orderedWorkgroupWaves wave 0's lanes must get 0..31 and wave 1's 32..63 in every run.
// The layout without the field is run and reported too (its out-of-order count is the evidence the test needs the
// field). Exit 1 when the ordered layout breaks the rule; 77 when the host has no 32-lane subgroups.
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
constexpr std::uint32_t WaveSize = 32;
constexpr std::uint32_t Raw = 3u << 28u;
constexpr std::uint32_t Selects = 0xfacu;
constexpr std::uint32_t Float32 = 22u;

alignas(256) std::array<std::uint32_t, 128> Program{};

std::span<const std::uint32_t> Code() {
    std::vector<std::uint32_t> code;
    code.push_back((0x1au << 25u) | (3u << 17u) | (0u << 9u) | 130u);   // v_lshlrev_b32 v3, 2, v0
    code.push_back(0xd7650006u);                                        // v_mbcnt_lo_u32_b32 v6, exec_lo, 0
    code.push_back(0x0001007eu);
    code.push_back((0x1au << 25u) | (4u << 17u) | (6u << 9u) | 140u);   // v_lshlrev_b32 v4, 12, v6
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
        const std::uint32_t runs = argc > 1 ? static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 200u;
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
        for (const bool ordered : {false, true}) {
            ShaderRecompiler::RecompileRequest request{
                {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
                {WaveSize, 0, userData, compute, std::nullopt, std::nullopt, regions},
                device->ComputeTarget(WaveSize),
                {0, 0, 0, 128}
            };
            request.target.orderedWorkgroupWaves = ordered;
            request.useCache = false;
            const auto shader = ShaderRecompiler::Recompile(request);
            if (shader.hostSubgroupSize != WaveSize) {
                std::printf("host subgroup %u: one host subgroup is not one wave32; skipped\n", shader.hostSubgroupSize);
                std::fflush(nullptr);
                TerminateProcess(GetCurrentProcess(), VulkanTestSkipped);
                std::_Exit(VulkanTestSkipped);
            }
            std::array<std::uint32_t, Threads> first{};
            std::uint32_t outOfOrder = 0, differing = 0, firstBad = ~0u;
            for (std::uint32_t run = 0; run < runs; ++run) {
                counter[0] = 0;
                std::memset(output, 0xff, Threads * 4u);
                device->Dispatch(shader, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
                device->WaitIdle();
                bool inOrder = counter[0] == Threads;
                for (std::uint32_t lane = 0; lane < Threads; ++lane) inOrder = inOrder && output[lane] / WaveSize == lane / WaveSize;
                if (!inOrder) {
                    ++outOfOrder;
                    if (firstBad == ~0u) firstBad = run;
                }
                if (run == 0) {
                    std::copy(output, output + Threads, first.begin());
                    std::printf("%s run 0: counter %u, lanes 0/31/32/63 got %u %u %u %u\n", ordered ? "ordered" : "unordered", counter[0], output[0], output[31], output[32], output[63]);
                }
                else differing += !std::equal(first.begin(), first.end(), output);
            }
            std::printf("%s: %u of %u runs with the waves out of order, %u differ from run 0", ordered ? "orderedWorkgroupWaves" : "without the field", outOfOrder, runs, differing);
            if (firstBad != ~0u) std::printf("; first bad run %u", firstBad);
            std::printf("\n");
            if (ordered) failures += outOfOrder != 0u;
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
