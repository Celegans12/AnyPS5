#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t BufferBytes = Lanes * 4u;
constexpr std::uint32_t ComputeQueue = 0x20;
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t Builds = 8;

alignas(256) std::uint32_t CodeStore[Builds][64];

std::uint32_t* NewBlock() {
#ifdef _WIN32
    auto* block = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    auto* block = static_cast<std::uint32_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
    std::memset(block, 0, BlockBytes);
    GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    return block;
}

void Register(const std::uint32_t* code, std::size_t bytes) {
    Shader shader{};
    shader.file_header = 0x34333231;
    shader.version = 0x18;
    shader.header_size = sizeof(Shader);
    shader.shader_size = static_cast<std::uint32_t>(bytes);
    shader.code = code;
    AgcDriverRegisterShader_nid_postfix(&shader);
}

}

int main(int argc, char** argv) {
    try {
        {
            const auto probe = OpenVulkanTestDevice();
            if (!probe) return VulkanTestSkipped;
        }
        const std::uint32_t iterations = argc > 1 ? static_cast<std::uint32_t>(std::atoi(argv[1])) : 0x7fffu;
        const int rounds = argc > 2 ? std::atoi(argv[2]) : 40;
        const bool ordered = std::getenv("BENCH_ORDERED") != nullptr;
        std::array<std::uint32_t*, Builds> targets{};
        for (std::uint32_t build = 0; build < Builds; ++build) {
            const std::uint32_t code[] = {0x34020082u, 0x7e040280u, 0xb0040000u | (iterations & 0xffffu), 0x10040502u, 0x060404f2u, 0x80848104u, 0xbf078004u, 0xbf85fffbu, 0x4a040480u + build, 0xe0701000u, 0x80000201u, 0xbf810000u};
            std::memcpy(CodeStore[build], code, sizeof(code));
            Register(CodeStore[build], sizeof(code));
            targets[build] = NewBlock();
        }
        std::vector<double> times;
        for (int round = 0; round < rounds + 2; ++round) {
            std::vector<std::uint32_t> words{0xc0037600u, 0x207u, Lanes, 1u, 1u, 0xc0017600u, 0x213u, 4u << 1u};
            for (std::uint32_t build = 0; build < Builds; ++build) {
                if (ordered && build != 0) words.insert(words.end(), {0xc0004600u, 0x407u});
                const auto address = reinterpret_cast<std::uintptr_t>(targets[build]);
                words.insert(words.end(), {0xc0047600u, 0x240u, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), BufferBytes, 0x31016facu});
                const auto codeAddress = reinterpret_cast<std::uintptr_t>(CodeStore[build]);
                words.insert(words.end(), {0xc0027600u, 0x20cu, static_cast<std::uint32_t>(codeAddress >> 8u), static_cast<std::uint32_t>(codeAddress >> 40u), 0xc0031500u, 1u, 1u, 1u, 0x8041u});
            }
            Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
            const auto start = std::chrono::steady_clock::now();
            if (sceAgcDriverSubmitAcb(ComputeQueue, &packet) != 0) throw std::runtime_error("submission failed");
            AgcDriverWaitIdle_nid_postfix();
            for (auto* target : targets) AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(target), BufferBytes);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (round >= 2) times.push_back(ms);
        }
        AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(targets[0]), BufferBytes);
        std::sort(times.begin(), times.end());
        std::printf("overlap bench: %u iterations, %s, %zu rounds: submit-to-results median %.3f ms, p10 %.3f, p90 %.3f; lane 0 of build 0 = %u\n", iterations, ordered ? "ordered (CS partial flush between)" : "unordered", times.size(), times[times.size() / 2], times[times.size() / 10], times[times.size() * 9 / 10], targets[0][0]);
        LibcRunShutdown_nid_postfix();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
