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
constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t HalfBytes = BlockBytes / 2;
constexpr std::uint32_t Programs = 8;
constexpr std::uint32_t ProgramDwords = 64;
constexpr std::uint32_t OutputBase = 0x20;
constexpr std::uint32_t OutputStride = 0x400;
constexpr std::uint32_t CodeDwords = 13;

alignas(256) std::array<std::uint32_t, Programs * ProgramDwords> Code{};
std::uint32_t* Block = nullptr;

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("bda anchor overlap: " + what);
}

void MakePrograms() {
    for (std::uint32_t program = 0; program < Programs; ++program) {
        const std::array<std::uint32_t, CodeDwords> code{
            0x3404008au, 0x34060082u, 0x4a0606ffu, OutputBase + program * OutputStride,
            0xdc308010u, 0x04000002u, 0xdc308014u, 0x05000002u,
            0xbf8c3f70u, 0x4a080b04u, 0xe0701000u, 0x80010403u, 0xbf810000u};
        std::copy(code.begin(), code.end(), Code.begin() + program * ProgramDwords);
        Shader shader{};
        shader.file_header = 0x34333231;
        shader.version = 0x18;
        shader.header_size = sizeof(Shader);
        shader.shader_size = CodeDwords * 4u;
        shader.code = Code.data() + program * ProgramDwords;
        AgcDriverRegisterShader_nid_postfix(&shader);
    }
}

std::vector<std::uint32_t> Submission(std::uint32_t repeats) {
    const auto base = reinterpret_cast<std::uintptr_t>(Block);
    std::vector<std::uint32_t> words{
        0xc0037600u, 0x207u, Lanes, 1u, 1u,
        0xc0017600u, 0x213u, 8u << 1u,
        0xc0087600u, 0x240u, static_cast<std::uint32_t>(base), static_cast<std::uint32_t>(base >> 32u), 0u, 0u,
        static_cast<std::uint32_t>(base), static_cast<std::uint32_t>((base >> 32u) & 0xffffu), static_cast<std::uint32_t>(HalfBytes), 0x31016facu,
    };
    for (std::uint32_t repeat = 0; repeat < repeats; ++repeat) {
        for (std::uint32_t program = 0; program < Programs; ++program) {
            const auto codeAddress = reinterpret_cast<std::uintptr_t>(Code.data() + program * ProgramDwords);
            words.insert(words.end(), {0xc0027600u, 0x20cu, static_cast<std::uint32_t>(codeAddress >> 8u), static_cast<std::uint32_t>(codeAddress >> 40u), 0xc0031500u, 1u, 1u, 1u, 0x8041u});
        }
    }
    return words;
}

void Round(std::uint32_t queue, std::uint32_t round) {
    for (std::uint32_t dword = 0; dword < BlockBytes / 4u; ++dword) Block[dword] = 0x1000u + dword * 3u + round;
    for (std::uint32_t program = 0; program < Programs; ++program) {
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) Block[(OutputBase + program * OutputStride) / 4u + lane] = 0;
    }
    auto words = Submission(4);
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitAcb(queue, &packet) == 0, "submission failed");
    AgcDriverWaitIdle_nid_postfix();
    const auto initial = [&](std::uint32_t byte) { return 0x1000u + byte / 4u * 3u + round; };
    for (std::uint32_t program = 0; program < Programs; ++program) {
        const auto out = (OutputBase + program * OutputStride) / 4u;
        AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(Block + out), Lanes * 4u);
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const auto read = lane * 1024u;
            const auto expected = initial(read + 0x10u) + initial(read + 0x14u);
            const auto value = std::atomic_ref<std::uint32_t>(Block[out + lane]).load(std::memory_order_acquire);
            check(value == expected, "queue 0x" + std::to_string(queue) + " round " + std::to_string(round) + " program " + std::to_string(program) + " lane " + std::to_string(lane) + " is " + std::to_string(value) + ", expected " + std::to_string(expected));
        }
    }
}

}

int main(int argc, char** argv) {
    try {
        {
            const auto probe = OpenVulkanTestDevice();
            if (!probe) return VulkanTestSkipped;
        }
        const std::uint32_t queue = argc > 1 ? static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 16)) : 0x51u;
        const std::uint32_t rounds = argc > 2 ? static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 20u;
#ifdef _WIN32
        Block = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        Block = static_cast<std::uint32_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        check(Block != nullptr, "cannot allocate the guest block");
        GuestAllocations::Mutation().Add(Block, HalfBytes, true, true, true);
        GuestAllocations::Mutation().Add(reinterpret_cast<std::uint8_t*>(Block) + HalfBytes, HalfBytes, true, true, true);
        MakePrograms();
        for (std::uint32_t round = 0; round < rounds; ++round) Round(queue, round);
        std::printf("bda anchor overlap: %u rounds of %u unordered address-based dispatches on queue 0x%x read across two allocations\n", rounds, Programs * 4u, queue);
        LibcRunShutdown_nid_postfix();
        std::puts("bda anchor overlap tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
