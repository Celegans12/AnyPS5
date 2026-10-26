#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <array>
#include <cstddef>
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

constexpr std::uint32_t Dispatches = 48;
constexpr std::size_t SrtBytes = 64;
constexpr std::size_t OutputBytes = 256;
constexpr std::size_t SrtOffset = 4096;
constexpr std::size_t OutputOffset = SrtOffset + static_cast<std::size_t>(Dispatches) * SrtBytes;
constexpr std::size_t BlockBytes = (OutputOffset + static_cast<std::size_t>(Dispatches) * OutputBytes + 65535) & ~static_cast<std::size_t>(65535);

constexpr std::array<std::uint32_t, 7> StoreCode{0xf4080100u, 0xfa000000u, 0xbf8c0000u, 0x7e020281u, 0xe0700000u, 0x80010100u, 0xbf810000u};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, PAGE_READWRITE, BlockBytes);
#else
        constexpr std::uintptr_t alignment = 65536;
        void* mapped = mmap(nullptr, BlockBytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + BlockBytes != begin + BlockBytes + alignment) munmap(reinterpret_cast<void*>(aligned + BlockBytes), begin + alignment - aligned);
            block = reinterpret_cast<std::uint8_t*>(aligned);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, BlockBytes);
        }
#endif
        Require(block != nullptr, "moved written templates: cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint64_t At(std::size_t offset) const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block + offset)); }
    std::uint8_t* Bytes(std::size_t offset) const { return block + offset; }

private:
    std::uint8_t* block = nullptr;
};

std::vector<std::uint32_t> Pm4(std::uint32_t opcode, std::initializer_list<std::uint32_t> body) {
    std::vector<std::uint32_t> words{0xc0000000u | (static_cast<std::uint32_t>(body.size() - 1) << 16u) | (opcode << 8u)};
    words.insert(words.end(), body);
    return words;
}

void Prepare(const GuestBlock& block) {
    std::memcpy(block.Bytes(0), StoreCode.data(), sizeof(StoreCode));
    for (std::uint32_t dispatch = 0; dispatch < Dispatches; ++dispatch) {
        const auto output = block.At(OutputOffset + static_cast<std::size_t>(dispatch) * OutputBytes);
        const std::array<std::uint32_t, 4> descriptor{static_cast<std::uint32_t>(output), static_cast<std::uint32_t>((output >> 32u) & 0xffffu), static_cast<std::uint32_t>(OutputBytes), 0x31016facu};
        std::memcpy(block.Bytes(SrtOffset + static_cast<std::size_t>(dispatch) * SrtBytes), descriptor.data(), sizeof(descriptor));
    }
    std::memset(block.Bytes(OutputOffset), 0, static_cast<std::size_t>(Dispatches) * OutputBytes);
}

void RunPass(const GuestBlock& block, std::uint32_t count = Dispatches, std::uint32_t repeats = 1) {
    const auto kernel = block.At(0);
    std::vector<std::uint32_t> words;
    for (std::uint32_t use = 0; use < count * repeats; ++use) {
        const auto dispatch = use % count;
        const auto srt = block.At(SrtOffset + static_cast<std::size_t>(dispatch) * SrtBytes);
        for (const auto& packet : {Pm4(0x76, {0x20c, static_cast<std::uint32_t>(kernel >> 8u)}), Pm4(0x76, {0x20d, static_cast<std::uint32_t>(kernel >> 40u) & 0xffu}), Pm4(0x76, {0x207, 64}), Pm4(0x76, {0x208, 1}), Pm4(0x76, {0x209, 1}), Pm4(0x76, {0x213, 2u << 1u}), Pm4(0x76, {0x240, static_cast<std::uint32_t>(srt)}), Pm4(0x76, {0x241, static_cast<std::uint32_t>(srt >> 32u)}), Pm4(0x15, {1, 1, 1, 0x41})}) {
            words.insert(words.end(), packet.begin(), packet.end());
        }
    }
    Packet dcb{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&dcb) == 0, "moved written templates: the dispatch submission failed");
    AgcDriverWaitIdle_nid_postfix();
}

std::uint32_t Stored(const GuestBlock& block, std::uint32_t dispatch) {
    std::uint32_t stored = 0;
    AgcDriver::GuestMemory::Read(block.At(OutputOffset + static_cast<std::size_t>(dispatch) * OutputBytes), std::as_writable_bytes(std::span(&stored, 1)), 4);
    return stored;
}

void ExpectStores(const GuestBlock& block, const std::string& what) {
    for (std::uint32_t dispatch = 0; dispatch < Dispatches; ++dispatch) {
        Require(Stored(block, dispatch) == 1u, "moved written templates: " + what + ": dispatch " + std::to_string(dispatch) + " stored " + std::to_string(Stored(block, dispatch)) + " at its own output, expected 1");
    }
}

void MovedOutputsShareOneTemplate(const GuestBlock& block) {
    const char* verifyText = std::getenv("APS5_VERIFY_MOVED_WRITTEN");
    const bool verify = verifyText != nullptr && std::strtoull(verifyText, nullptr, 10) >= Dispatches;
    auto& cache = AgcDriver::Graphics::SharedResourceCache();
    const auto fatesBefore = cache.ComputeFateCounts();
    const auto movedBefore = AgcDriver::VulkanDevice::MovedWrittenTotals();
    RunPass(block);
    ExpectStores(block, "written V#s moved");
    const auto fates = cache.ComputeFateCounts();
    const auto movedAfter = AgcDriver::VulkanDevice::MovedWrittenTotals();
    const auto hits = fates.hits - fatesBefore.hits;
    const auto rebound = movedAfter[0] - movedBefore[0];
    const auto verified = movedAfter[1] - movedBefore[1];
    const auto mismatches = movedAfter[2] - movedBefore[2];
    std::printf("moved written templates: %u dispatches with their own output V#: %llu template hits, %llu rebound a written V#, %llu verified, %llu mismatches\n", Dispatches, static_cast<unsigned long long>(hits), static_cast<unsigned long long>(rebound), static_cast<unsigned long long>(verified), static_cast<unsigned long long>(mismatches));
    Require(mismatches == 0, "moved written templates: a use bound its written V#s unlike a full build");
    Require(rebound + 1u >= Dispatches, "moved written templates: the dispatches with moved output V#s did not reuse one template");
    if (verify) Require(verified + 1u >= Dispatches, "moved written templates: verify mode did not compare every rebinding");
    AgcDriver::Graphics::RuntimeSwitches::Assign(AgcDriver::Graphics::RuntimeSwitch::NoMovedWrittenTemplates, true);
    cache.Clear();
    Prepare(block);
    const auto offBefore = cache.ComputeFateCounts();
    const auto offMoved = AgcDriver::VulkanDevice::MovedWrittenTotals();
    RunPass(block);
    ExpectStores(block, "APS5_NO_MOVED_WRITTEN_TEMPLATES");
    const auto offAfter = cache.ComputeFateCounts();
    const auto offHits = offAfter.hits - offBefore.hits;
    std::printf("moved written templates: switched off: %llu template hits\n", static_cast<unsigned long long>(offHits));
    Require(offHits == 0 && AgcDriver::VulkanDevice::MovedWrittenTotals()[0] == offMoved[0], "moved written templates: with APS5_NO_MOVED_WRITTEN_TEMPLATES a template served another output base");
    AgcDriver::Graphics::RuntimeSwitches::Assign(AgcDriver::Graphics::RuntimeSwitch::NoMovedWrittenTemplates, false);
}

void UseTheResourceCacheOnEveryDispatch() {
#ifdef _WIN32
    _putenv_s("APS5_NO_DISPATCH_RECIPE", "1");
    _putenv_s("APS5_PROFILE_DRAW", "1");
#else
    setenv("APS5_NO_DISPATCH_RECIPE", "1", 1);
    setenv("APS5_PROFILE_DRAW", "1", 1);
#endif
}

}

int main() {
    UseTheResourceCacheOnEveryDispatch();
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        auto& cache = AgcDriver::Graphics::SharedResourceCache();
        cache.Clear();
        GuestBlock block;
        Prepare(block);
        MovedOutputsShareOneTemplate(block);
        AgcDriverShutdown_nid_postfix();
        std::puts("moved written templates: one template served every output base, each store landed at its own base");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
