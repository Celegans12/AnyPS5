#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VramBudget.hpp"
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
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::Graphics::SampledCacheCounters;
using AgcDriver::Graphics::SampledTextureMemory;
using AgcDriver::Graphics::SharedResourceCache;
using AgcDriver::Graphics::TextureCacheUsage;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Side = 256;
constexpr std::uint32_t TargetSide = 128;
constexpr std::uint32_t Format32Float = 22;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t TileR64KBX = 0x1b;
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t SurfaceBytes = std::size_t{Side} * Side * 4u;
constexpr std::size_t Surfaces = 24;
constexpr std::size_t TargetBytes = std::size_t{TargetSide} * TargetSide * 4u;
constexpr std::uint64_t Budget = 1ull << 20u;
constexpr std::uint64_t Ceiling = Budget + Budget / 4u;
constexpr std::array<std::uint32_t, 4> PointSampler{0x00000092u, 0x00fff000u, 0u, 0u};

// image_sample_lz v[10:13], v[20:21] (0, 0), s[4:11], s[12:15]; image_store v[10:13], v[30:31]
// (thread x, 0), s[16:23].
alignas(256) constexpr std::array<std::uint32_t, 10> SampleStoreCode{
    0x7e280280u, 0x7e2a0280u, 0xf09c0f08u, 0x00610a14u, 0x7e3c0300u, 0x7e3e0280u, 0xbf8c3f70u, 0xf0201f08u,
    0x00040a1eu, 0xbf810000u,
};

class Block {
public:
    explicit Block(std::size_t bytes) : bytes(bytes) {
#ifdef _WIN32
        memory = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        memory = std::aligned_alloc(65536, bytes);
#endif
        Require(memory != nullptr, "cannot allocate the test block");
        std::memset(memory, 0x3c, bytes);
        GuestAllocations::Mutation().Add(memory, bytes, true, true, true);
    }

    ~Block() {
        GuestAllocations::Mutation().Remove(memory);
#ifdef _WIN32
        VirtualFree(memory, 0, MEM_RELEASE);
#else
        std::free(memory);
#endif
    }

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;

    std::uint64_t Address(std::size_t offset = 0) const { return reinterpret_cast<std::uint64_t>(memory) + offset; }

private:
    std::size_t bytes;
    void* memory = nullptr;
};

std::array<std::uint32_t, 8> Descriptor(std::uint64_t address, std::uint32_t format, std::uint32_t side) {
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((side - 1u) & 3u) << 30u),
        ((side - 1u) >> 2u) | ((side - 1u) << 14u),
        0xfacu | (TileR64KBX << 20u) | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

// One dispatch sampling surface `surface` of `sampled` into the storage image in `target`: its
// resource set binds the surface's sampled texture and goes into the resource cache.
void Sample(AgcDriver::VulkanDevice& device, const Block& sampled, const Block& target, std::size_t surface) {
    std::vector<std::uint32_t> userData(24, 0u);
    const auto texture = Descriptor(sampled.Address(surface * SurfaceBytes), Format32Float, Side);
    const auto storage = Descriptor(target.Address(), Format32UInt, TargetSide);
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    std::copy(PointSampler.begin(), PointSampler.end(), userData.begin() + 12);
    std::copy(storage.begin(), storage.end(), userData.begin() + 16);
    const std::span<const std::uint32_t> code(SampleStoreCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{32, 1, 1}, 0u, {true, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
}

std::string MiB(std::uint64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f MiB", static_cast<double>(bytes) / (1 << 20));
    return text;
}

}

int main() {
    try {
#ifdef _WIN32
        Require(_putenv_s("APS5_TEXTURE_CACHE_MIB", "1") == 0, "cannot set the test cache budget");
#else
        Require(setenv("APS5_TEXTURE_CACHE_MIB", "1", 1) == 0, "cannot set the test cache budget");
#endif
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Block sampled(Surfaces * SurfaceBytes);
        Block target(65536);
        const auto baseline = SampledTextureMemory();
        const auto release = [&] {
            device->WaitIdle();
            AgcDriver::Graphics::ClearCachedTextures(device->Device());
            SharedResourceCache().Clear();
        };
        try {
            // Sets whose work completed bind textures nobody else uses: evicting such a texture
            // drops the sets, so every sampled image alive afterwards is one the cache accounts.
            const auto setsBefore = SharedResourceCache().Size();
            Sample(*device, sampled, target, 0);
            device->WaitIdle();
            const auto first = TextureCacheUsage();
            if (first.sampledBytes == 0) {
                std::puts("the sampled texture is a view of a storage image (the block is host-imported): pinned snapshots not tested");
                release();
                return 0;
            }
            Require(SharedResourceCache().Size() > setsBefore, "the sampling dispatch did not enter the resource cache");
            const auto perTexture = first.sampledBytes;
            const auto capacity = static_cast<std::size_t>(Budget / perTexture);
            Require(capacity >= 2u && capacity + 4u < Surfaces / 2u, "a " + MiB(perTexture) + " snapshot does not fit the test's 1 MiB budget as planned");
            const auto retired = capacity + 4u;
            for (std::size_t surface = 1; surface < retired; ++surface) {
                Sample(*device, sampled, target, surface);
                device->WaitIdle();
            }
            const auto afterRetired = SampledCacheCounters();
            const auto live = SampledTextureMemory() - baseline;
            std::printf("%zu completed dispatches over %s textures: %s of sampled images alive, %s cached; %llu evicted, %llu resource sets dropped with them\n", retired, MiB(perTexture).c_str(), MiB(live).c_str(), MiB(TextureCacheUsage().sampledBytes).c_str(), static_cast<unsigned long long>(afterRetired.evicted), static_cast<unsigned long long>(afterRetired.droppedSets));
            Require(live <= Budget, "after " + std::to_string(retired) + " completed dispatches " + MiB(live) + " of sampled images are alive, more than the " + MiB(Budget) + " budget: resource-cache sets keep evicted textures");
            Require(afterRetired.evicted >= retired - capacity && afterRetired.droppedSets >= retired - capacity, "the textures evicted past the budget did not take their resource sets along");

            // Dispatches whose work is still recorded use their textures: those are not evicted,
            // new ones past the ceiling are refused, and a refused texture's set stays out of the
            // resource cache, so it goes when its work completes.
            const auto burstValue = [](std::size_t surface) { return 0x3f800000u + static_cast<std::uint32_t>(surface); };
            for (std::size_t surface = retired; surface < retired + capacity + 4u; ++surface) {
                auto* texels = reinterpret_cast<std::uint32_t*>(sampled.Address(surface * SurfaceBytes));
                std::fill(texels, texels + SurfaceBytes / 4u, burstValue(surface));
                Sample(*device, sampled, target, surface);
            }
            const auto recorded = SampledCacheCounters();
            const auto recordedLive = SampledTextureMemory() - baseline;
            std::printf("the burst recorded, not yet waited for: %s of device-local sampled images, %llu snapshots placed in system memory\n", MiB(recordedLive).c_str(), static_cast<unsigned long long>(recorded.spilled - afterRetired.spilled));
            if (recorded.spilled != afterRetired.spilled) Require(recordedLive <= Ceiling, "with snapshots going to system memory, the recorded burst still took " + MiB(recordedLive) + " of device-local sampled images, past the " + MiB(Ceiling) + " ceiling");
            device->WaitIdle();
            AgcDriver::GuestMemory::FlushGpuWrites(target.Address(), TargetBytes);
            const auto stored = *reinterpret_cast<const std::uint32_t*>(target.Address());
            Require(stored == burstValue(retired + capacity + 3u), "the burst's last dispatch stored " + std::to_string(stored) + ", not the texel of its surface " + std::to_string(burstValue(retired + capacity + 3u)) + (recorded.spilled != afterRetired.spilled ? " (snapshots in system memory)" : ""));
            const auto afterBurst = SampledCacheCounters();
            const auto burstLive = SampledTextureMemory() - baseline;
            std::printf("a burst of %zu dispatches before one wait: %s alive afterwards; %llu refused, %llu evicted, %llu sets not cached for binding a refused texture\n", capacity + 4u, MiB(burstLive).c_str(), static_cast<unsigned long long>(afterBurst.refused - afterRetired.refused), static_cast<unsigned long long>(afterBurst.evicted - afterRetired.evicted), static_cast<unsigned long long>(afterBurst.refusedSets - afterRetired.refusedSets));
            Require(burstLive <= Ceiling, "after a burst of dispatches completed " + MiB(burstLive) + " of sampled images are alive, more than the " + MiB(Ceiling) + " ceiling");
            Require(afterBurst.refused + afterBurst.evicted > afterRetired.refused + afterRetired.evicted, "a burst past the budget neither refused nor evicted a texture");
            Require(afterBurst.refused == afterRetired.refused || afterBurst.refusedSets > afterRetired.refusedSets, "a refused texture's resource set was cached");

            // Out of device memory: the reclaim evicts every texture no recorded work uses, with the
            // sets binding them, and leaves the one the open batch uses.
            Sample(*device, sampled, target, Surfaces - 1u);
            const auto reclaimed = AgcDriver::Graphics::ReclaimTextureMemory(device->Device(), 1);
            const auto kept = TextureCacheUsage();
            std::printf("out-of-memory reclaim with one dispatch recorded: %s released, %zu entries (%s) left\n", MiB(reclaimed).c_str(), kept.sampledEntries, MiB(kept.sampledBytes).c_str());
            Require(reclaimed != 0, "the out-of-memory reclaim released nothing from a cache of completed textures");
            Require(kept.sampledEntries <= 1u && kept.sampledBytes <= perTexture, "the out-of-memory reclaim left textures no recorded work uses");
            Require(SampledTextureMemory() - baseline <= 2u * perTexture, "the out-of-memory reclaim evicted textures whose resource sets kept them alive");
            device->WaitIdle();
            AgcDriver::Graphics::ReclaimTextureMemory(device->Device(), 1);
            Require(TextureCacheUsage().sampledEntries == 0 && SampledTextureMemory() == baseline, "once its work completed, the last texture was not reclaimed");
            const auto counts = SampledCacheCounters();
            Require(counts.reclaims >= 1u && counts.reclaimedBytes >= reclaimed, "the reclaim was not counted");
        } catch (...) {
            release();
            throw;
        }
        release();
        std::puts("sampled cache pin tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
