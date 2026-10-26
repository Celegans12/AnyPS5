#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
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
        Require(block != nullptr, "compute template budget: cannot allocate the guest block");
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
    Require(sceAgcDriverSubmitDcb(&dcb) == 0, "compute template budget: the dispatch submission failed");
    AgcDriverWaitIdle_nid_postfix();
}

AgcDriver::Graphics::ResourceCache::ComputeFates Counts() {
    return AgcDriver::Graphics::SharedResourceCache().ComputeFateCounts();
}

void Passes3(const GuestBlock& block, std::uint64_t& hits, std::uint64_t& evicted, std::uint64_t& lookups) {
    const auto before = Counts();
    for (std::uint32_t pass = 0; pass < 3; ++pass) RunPass(block);
    const auto after = Counts();
    hits = after.hits - before.hits;
    evicted = after.evicted - before.evicted;
    lookups = after.hits + after.misses - before.hits - before.misses;
}

void OwnBudgetKeepsTheWorkingSet(const GuestBlock& block) {
    auto& cache = AgcDriver::Graphics::SharedResourceCache();
    std::uint64_t hits = 0, evicted = 0, lookups = 0;
    Passes3(block, hits, evicted, lookups);
    const auto counts = Counts();
    std::printf("compute template budget: own budget: %llu of %llu lookups hit, %llu misses of evicted keys, %zu compute entries (bound %zu), %zu entries in all\n", static_cast<unsigned long long>(hits), static_cast<unsigned long long>(lookups), static_cast<unsigned long long>(evicted), counts.entries, counts.capacity, cache.Size());
    Require(counts.split && counts.capacity == 64, "compute template budget: the compute bound is not APS5_COMPUTE_TEMPLATE_ENTRIES");
    Require(hits >= 2u * Dispatches && evicted == 0, "compute template budget: " + std::to_string(Dispatches) + " compute sets over a shared bound of 8 were not all kept in their own bound of 64");
    Require(counts.entries >= Dispatches && counts.entries <= 64, "compute template budget: the compute entries are not counted in their own bound");
}

void SharedBoundEvicts(const GuestBlock& block) {
    auto& cache = AgcDriver::Graphics::SharedResourceCache();
    AgcDriver::Graphics::RuntimeSwitches::Assign(AgcDriver::Graphics::RuntimeSwitch::NoComputeTemplateBudget, true);
    cache.Clear();
    std::uint64_t hits = 0, evicted = 0, lookups = 0;
    Passes3(block, hits, evicted, lookups);
    std::printf("compute template budget: shared bound: %llu of %llu lookups hit, %llu misses of evicted keys, %zu entries\n", static_cast<unsigned long long>(hits), static_cast<unsigned long long>(lookups), static_cast<unsigned long long>(evicted), cache.Size());
    Require(cache.Size() <= 8, "compute template budget: with APS5_NO_COMPUTE_TEMPLATE_BUDGET the compute sets were not held to the shared bound");
    Require(evicted >= Dispatches && hits < Dispatches, "compute template budget: with APS5_NO_COMPUTE_TEMPLATE_BUDGET the misses were not charged to eviction");
    AgcDriver::Graphics::RuntimeSwitches::Assign(AgcDriver::Graphics::RuntimeSwitch::NoComputeTemplateBudget, false);
}

void UseTheResourceCacheOnEveryDispatch() {
#ifdef _WIN32
    _putenv_s("APS5_NO_DISPATCH_RECIPE", "1");
    _putenv_s("APS5_RESOURCE_CACHE_ENTRIES", "8");
    _putenv_s("APS5_COMPUTE_TEMPLATE_ENTRIES", "64");
    _putenv_s("APS5_PROFILE_DRAW", "1");
#else
    setenv("APS5_NO_DISPATCH_RECIPE", "1", 1);
    setenv("APS5_RESOURCE_CACHE_ENTRIES", "8", 1);
    setenv("APS5_COMPUTE_TEMPLATE_ENTRIES", "64", 1);
    setenv("APS5_PROFILE_DRAW", "1", 1);
#endif
}

}

int main() {
    UseTheResourceCacheOnEveryDispatch();
    AgcDriver::Graphics::RuntimeSwitches::Assign(AgcDriver::Graphics::RuntimeSwitch::NoMovedWrittenTemplates, true);
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        auto& cache = AgcDriver::Graphics::SharedResourceCache();
        cache.Clear();
        GuestBlock block;
        Prepare(block);
        OwnBudgetKeepsTheWorkingSet(block);
        SharedBoundEvicts(block);
        AgcDriverShutdown_nid_postfix();
        std::puts("compute template budget: compute sets keep their own bound, the switch restores the shared one");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
