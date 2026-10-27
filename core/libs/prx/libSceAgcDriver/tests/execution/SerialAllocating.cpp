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

constexpr std::uint32_t Groups = 256;
constexpr std::uint32_t ComputeQueue = 0x20;
constexpr std::size_t BlockBytes = 65536;
constexpr int Rounds = 8;

alignas(256) constexpr std::array<std::uint32_t, 9> AllocateCode{0x7e020281u, 0xe0c84000u, 0x80000100u, 0xbf8c3f70u, 0x34020282u, 0x7e040208u, 0xe0701000u, 0x80010201u, 0xbf810000u};

std::uint32_t* NewBlock() {
#ifdef _WIN32
    auto* block = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    auto* block = static_cast<std::uint32_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
    if (block == nullptr) throw std::runtime_error("serial allocating: cannot allocate a guest block");
    std::memset(block, 0, BlockBytes);
    GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    return block;
}

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("serial allocating: " + what);
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

bool RunRound(std::uint32_t* counter, std::uint32_t* list, bool serial, int round) {
    counter[0] = 0;
    std::memset(list, 0xff, Groups * sizeof(std::uint32_t));
    const auto codeAddress = reinterpret_cast<std::uintptr_t>(AllocateCode.data());
    std::vector<std::uint32_t> words{0xc0037600u, 0x207u, 1u, 1u, 1u, 0xc0017600u, 0x213u, (8u << 1u) | 0x80u};
    AddBuffer(words, 0x240u, counter, sizeof(std::uint32_t));
    AddBuffer(words, 0x244u, list, Groups * sizeof(std::uint32_t));
    words.insert(words.end(), {0xc0027600u, 0x20cu, static_cast<std::uint32_t>(codeAddress >> 8u), static_cast<std::uint32_t>(codeAddress >> 40u), 0xc0031500u, Groups, 1u, 1u, 0x8041u});
    const auto before = AgcDriver::VulkanDevice::SerialAllocatingCounts();
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitAcb(ComputeQueue, &packet) == 0, "submission failed");
    AgcDriverWaitIdle_nid_postfix();
    const auto after = AgcDriver::VulkanDevice::SerialAllocatingCounts();
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(counter), sizeof(std::uint32_t));
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(list), Groups * sizeof(std::uint32_t));
    const std::string where = "round " + std::to_string(round + 1) + ": ";
    check(Load(counter, 0) == Groups, where + "the counter is " + std::to_string(Load(counter, 0)) + ", expected " + std::to_string(Groups));
    std::vector<std::uint32_t> seen(Groups);
    for (std::uint32_t slot = 0; slot < Groups; ++slot) seen[slot] = Load(list, slot);
    auto sorted = seen;
    std::sort(sorted.begin(), sorted.end());
    for (std::uint32_t group = 0; group < Groups; ++group) check(sorted[group] == group, where + "the list is not a permutation of the workgroup ids (sorted entry " + std::to_string(group) + " is " + std::to_string(sorted[group]) + ")");
    bool inOrder = true;
    for (std::uint32_t slot = 0; slot < Groups; ++slot) inOrder = inOrder && seen[slot] == slot;
    if (serial) {
        for (std::uint32_t slot = 0; slot < Groups; ++slot) check(seen[slot] == slot, where + "slot " + std::to_string(slot) + " holds workgroup " + std::to_string(seen[slot]) + " with APS5_EXACT_SERIAL_ALLOCATING set");
        check(after.dispatches - before.dispatches == 1u, where + std::to_string(after.dispatches - before.dispatches) + " dispatches were recorded one workgroup at a time, expected 1");
        check(after.workgroups - before.workgroups == Groups, where + std::to_string(after.workgroups - before.workgroups) + " workgroups were recorded one at a time, expected " + std::to_string(Groups));
    } else {
        check(after.dispatches == before.dispatches && after.workgroups == before.workgroups, where + "a dispatch was split without APS5_EXACT_SERIAL_ALLOCATING");
    }
    return inOrder;
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
        Register(AllocateCode.data(), sizeof(AllocateCode));
        auto* counter = NewBlock();
        auto* list = NewBlock();
        int ordered = 0;
        for (int round = 0; round < Rounds; ++round) ordered += RunRound(counter, list, serial, round) ? 1 : 0;
        std::printf("serial allocating: %d of %d rounds of %u workgroups allocated in workgroup order (%s)\n", ordered, Rounds, Groups, serial ? "APS5_EXACT_SERIAL_ALLOCATING" : "default");
        LibcRunShutdown_nid_postfix();
        std::puts("serial allocating tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
