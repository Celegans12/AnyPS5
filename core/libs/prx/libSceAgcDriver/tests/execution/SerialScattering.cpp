#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
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

// Workgroups whose store address comes from memory (ProgramScatters) all write their workgroup id to the slot an
// index buffer names. With APS5_EXACT_SERIAL_ALLOCATING such a dispatch is recorded one workgroup at a time, so the
// last workgroup's id is what remains; without it the dispatch is not split and the winner follows the host's
// schedule. The second program passes the index through LDS, so it also covers a program with workgroup memory.

namespace {

constexpr std::uint32_t Groups = 256;
constexpr std::uint32_t ComputeQueue = 0x20;
constexpr std::size_t BlockBytes = 65536;
constexpr int Rounds = 8;

// buffer_load_dword v1, off, s[0:3], 0; s_waitcnt vmcnt(0); v_lshlrev_b32 v1, 2, v1; v_mov_b32 v2, s8;
// buffer_store_dword v2, v1, s[4:7], 0 offen; s_endpgm
alignas(256) constexpr std::array<std::uint32_t, 8> ScatterCode{0xe0300000u, 0x80000100u, 0xbf8c3f70u, 0x34020282u, 0x7e040208u, 0xe0701000u, 0x80010201u, 0xbf810000u};
// as above, with the index written to LDS and read back (v_mov_b32 v0, 0; ds_write_b32 v0, v1; s_waitcnt lgkmcnt(0);
// ds_read_b32 v1, v0; s_waitcnt lgkmcnt(0)) before the shift
alignas(256) constexpr std::array<std::uint32_t, 15> ScatterLdsCode{0xe0300000u, 0x80000100u, 0xbf8c3f70u, 0x7e000280u, 0xd8340000u, 0x00000100u, 0xbf8cc07fu, 0xd8d80000u, 0x01000000u, 0xbf8cc07fu, 0x34020282u, 0x7e040208u, 0xe0701000u, 0x80010201u, 0xbf810000u};

std::uint32_t* NewBlock() {
#ifdef _WIN32
    auto* block = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    auto* block = static_cast<std::uint32_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
    if (block == nullptr) throw std::runtime_error("serial scattering: cannot allocate a guest block");
    std::memset(block, 0, BlockBytes);
    GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    return block;
}

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("serial scattering: " + what);
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

void AddBuffer(std::vector<std::uint32_t>& words, std::uint32_t reg, const std::uint32_t* memory, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(memory);
    words.insert(words.end(), {0xc0047600u, reg, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu});
}

std::uint32_t Load(const std::uint32_t* memory, std::uint32_t index) {
    return std::atomic_ref<std::uint32_t>(const_cast<std::uint32_t&>(memory[index])).load(std::memory_order_acquire);
}

// One dispatch of Groups one-thread workgroups; returns whether the last workgroup's id remained.
bool RunRound(const std::uint32_t* code, bool lds, std::uint32_t* index, std::uint32_t* list, bool serial, int round) {
    index[0] = 3;
    std::memset(list, 0xff, 16 * sizeof(std::uint32_t));
    const auto codeAddress = reinterpret_cast<std::uintptr_t>(code);
    // COMPUTE_PGM_RSRC2: 8 user SGPRs, workgroup id x; the LDS program asks for one LDS granule.
    const std::uint32_t rsrc2 = (8u << 1u) | 0x80u | (lds ? (1u << 15u) : 0u);
    std::vector<std::uint32_t> words{0xc0037600u, 0x207u, 1u, 1u, 1u, 0xc0017600u, 0x213u, rsrc2};
    AddBuffer(words, 0x240u, index, sizeof(std::uint32_t));
    AddBuffer(words, 0x244u, list, 16 * sizeof(std::uint32_t));
    words.insert(words.end(), {0xc0027600u, 0x20cu, static_cast<std::uint32_t>(codeAddress >> 8u), static_cast<std::uint32_t>(codeAddress >> 40u), 0xc0031500u, Groups, 1u, 1u, 0x8041u});
    const auto before = AgcDriver::VulkanDevice::SerialAllocatingCounts();
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitAcb(ComputeQueue, &packet) == 0, "submission failed");
    AgcDriverWaitIdle_nid_postfix();
    const auto after = AgcDriver::VulkanDevice::SerialAllocatingCounts();
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(list), 16 * sizeof(std::uint32_t));
    const std::string where = std::string(lds ? "LDS program" : "program") + ", round " + std::to_string(round + 1) + ": ";
    const auto winner = Load(list, 3);
    check(winner < Groups, where + "slot 3 holds " + std::to_string(winner) + ", not a workgroup id");
    for (std::uint32_t slot = 0; slot < 16; ++slot) check(slot == 3 || Load(list, slot) == 0xffffffffu, where + "slot " + std::to_string(slot) + " was written");
    if (serial) {
        check(winner == Groups - 1u, where + "slot 3 holds workgroup " + std::to_string(winner) + " with APS5_EXACT_SERIAL_ALLOCATING set, expected the last one");
        check(after.dispatches - before.dispatches == 1u, where + std::to_string(after.dispatches - before.dispatches) + " dispatches were recorded one workgroup at a time, expected 1");
        check(after.workgroups - before.workgroups == Groups, where + std::to_string(after.workgroups - before.workgroups) + " workgroups were recorded one at a time, expected " + std::to_string(Groups));
    } else {
        check(after.dispatches == before.dispatches && after.workgroups == before.workgroups, where + "a dispatch was split without APS5_EXACT_SERIAL_ALLOCATING");
    }
    return winner == Groups - 1u;
}

}

int main(int argc, char** argv) {
    try {
        {
            const auto probe = OpenVulkanTestDevice();
            if (!probe) return VulkanTestSkipped;
        }
        const bool serial = argc > 1 && std::string(argv[1]) == "--serial";
        check(serial == AgcDriver::VulkanDevice::ExactSerialAllocating(), serial ? "--serial needs APS5_EXACT_SERIAL_ALLOCATING" : "APS5_EXACT_SERIAL_ALLOCATING is set without --serial");
        Register(ScatterCode.data(), sizeof(ScatterCode));
        Register(ScatterLdsCode.data(), sizeof(ScatterLdsCode));
        auto* index = NewBlock();
        auto* list = NewBlock();
        for (const bool lds : {false, true}) {
            int last = 0;
            for (int round = 0; round < Rounds; ++round) last += RunRound(lds ? ScatterLdsCode.data() : ScatterCode.data(), lds, index, list, serial, round) ? 1 : 0;
            std::printf("serial scattering: %d of %d rounds of %u workgroups left the last workgroup's store (%s%s)\n", last, Rounds, Groups, lds ? "LDS program, " : "", serial ? "APS5_EXACT_SERIAL_ALLOCATING" : "default");
        }
        LibcRunShutdown_nid_postfix();
        std::puts("serial scattering tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
