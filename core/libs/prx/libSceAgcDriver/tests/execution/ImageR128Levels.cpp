#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Words = 32;
constexpr std::uint32_t Side = 256;
constexpr std::uint32_t BaseLevel = 2;
constexpr std::uint32_t LastLevel = 8;
constexpr std::uint32_t Resident = LastLevel - BaseLevel + 1u;
constexpr std::uint32_t ResidentSide = Side >> BaseLevel;
constexpr std::uint32_t Result = 8;
constexpr std::uint32_t Format8888UNorm = 56;
constexpr std::uint32_t TileStandard4KB = 0x05;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t R128 = 0x8000u;
constexpr std::uint32_t Garbage = 0xffffffffu;
alignas(256) std::array<std::uint32_t, Threads * Words> Buffer{};

constexpr std::uint32_t Prologue[] = {0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xbf8c3f70};
constexpr std::uint32_t Epilogue[] = {0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xbf810000};

constexpr std::array<std::uint32_t, 22> Program(std::uint32_t first, std::uint32_t second) {
    std::array<std::uint32_t, 22> code{};
    std::size_t index = 0;
    for (const auto word : Prologue) code[index++] = word;
    code[index++] = first;
    code[index++] = second;
    for (const auto word : Epilogue) code[index++] = word;
    return code;
}

alignas(256) constexpr auto Load2D = Program(0xf0001f08, 0x00010c02);
alignas(256) constexpr auto Load2DR128 = Program(0xf0001f08 | R128, 0x00010c02);
alignas(256) constexpr auto SampleLod2D = Program(0xf0900f08, 0x00610c02);
alignas(256) constexpr auto SampleLod2DR128 = Program(0xf0900f08 | R128, 0x00610c02);

std::array<std::uint32_t, 8> Descriptor(const void* data, std::uint32_t side, std::uint32_t base, std::uint32_t last, std::uint32_t maxMip, std::uint32_t tail) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format8888UNorm << 20u) | (((side - 1u) & 3u) << 30u),
        ((side - 1u) >> 2u) | ((side - 1u) << 14u),
        0xfacu | (base << 12u) | (last << 16u) | (TileStandard4KB << 20u) | (Type2D << 28u),
        tail,
        (maxMip << 4u) | tail,
        tail,
        tail,
    };
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::array<std::uint32_t, 4> SamplerDescriptor() {
    return {0u, 0xfffu << 12u, 1u << 26u, 0u};
}

void Dispatch(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData) {
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

using Input = std::function<std::array<std::uint32_t, 4>(std::uint32_t)>;

std::vector<std::uint32_t> Run(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::array<std::uint32_t, 8>& texture, const Input& input) {
    Buffer.fill(0xdeadbeefu);
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto words = input(tid);
        std::copy(words.begin(), words.end(), &Buffer[tid * Words]);
    }
    std::vector<std::uint32_t> userData(16, 0u);
    const auto buffer = BufferDescriptor(Buffer.data(), static_cast<std::uint32_t>(Buffer.size() * 4u));
    const auto sampler = SamplerDescriptor();
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    std::copy(sampler.begin(), sampler.end(), userData.begin() + 12);
    Dispatch(device, code, userData);
    std::vector<std::uint32_t> results;
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t component = 0; component < 4u; ++component) results.push_back(Buffer[tid * Words + Result + component]);
    }
    return results;
}

struct Streamed {
    std::vector<std::uint32_t> storage;
    std::uint32_t* texels = nullptr;
    std::size_t residentBytes = 0;

    Streamed() {
        const auto probe = Descriptor(reinterpret_cast<const void*>(std::uintptr_t{0x10000}), Side, BaseLevel, LastLevel, 0u, 0u);
        const auto resident = AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(AgcDriver::Graphics::R128TextureWords(probe)));
        auto chain = probe;
        chain[3] &= ~(0xfu << 12u);
        const auto full = AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(AgcDriver::Graphics::R128TextureWords(chain)));
        residentBytes = static_cast<std::size_t>(resident.guestBytes);
        Require(resident.mips.size() == Resident && full.mips.size() == LastLevel + 1u && residentBytes <= full.mips.at(1).tiledOffset && full.guestBytes > ((residentBytes + 0xffffu) & ~std::size_t{0xffffu}) + 0x10000u, "128-bit levels: the resident levels must be the start of the 9-level chain and much smaller than it");
        storage.assign((static_cast<std::size_t>(full.guestBytes) + 0x10000u) / 4u, 0xdeadbeefu);
        texels = reinterpret_cast<std::uint32_t*>((reinterpret_cast<std::uintptr_t>(storage.data()) + 0xffffu) & ~std::uintptr_t{0xffffu});
        for (std::size_t index = 0; index < residentBytes / 4u; ++index) texels[index] = 0x01000000u + static_cast<std::uint32_t>(index) * 2654435761u;
        GuestAllocations::Mutation mutation;
        mutation.Add(texels, (residentBytes + 0xffffu) & ~std::size_t{0xffffu}, true, true, true);
    }
    ~Streamed() {
        GuestAllocations::Mutation mutation;
        mutation.Remove(texels);
    }
    Streamed(const Streamed&) = delete;
    Streamed& operator=(const Streamed&) = delete;
};

void Compare(const std::vector<std::uint32_t>& actual, const std::vector<std::uint32_t>& expected, const std::string& name) {
    Require(actual.size() == expected.size(), name + ": result count differs");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        Require(actual[index] == expected[index], name + ": thread " + std::to_string(index / 4u) + " component " + std::to_string(index % 4u) + " is " + std::to_string(actual[index]) + ", the resident chain's T# gives " + std::to_string(expected[index]));
    }
}

void CheckLoads(AgcDriver::VulkanDevice& device, const Streamed& streamed) {
    const Input coordinates = [](std::uint32_t tid) { return std::array<std::uint32_t, 4>{(tid * 7u + 3u) % ResidentSide, (tid * 13u + 5u) % ResidentSide, 0u, 0u}; };
    const auto expected = Run(device, Load2D, Descriptor(streamed.texels, ResidentSide, 0u, Resident - 1u, Resident - 1u, 0u), coordinates);
    std::set<std::uint32_t> distinct;
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        Require(expected[tid * 4u + 3u] != 0u, "128-bit levels: the resident chain's T# read a zero texel");
        distinct.insert(expected[tid * 4u] ^ (expected[tid * 4u + 1u] << 1u) ^ (expected[tid * 4u + 2u] << 2u));
    }
    Require(distinct.size() > Threads / 2u, "128-bit levels: the resident chain's T# read too few distinct texels");
    Compare(Run(device, Load2DR128, Descriptor(streamed.texels, Side, BaseLevel, LastLevel, 0u, Garbage), coordinates), expected, "image_load through a 128-bit T# of levels 2..8 of a 256x256 chain whose levels 0 and 1 are not resident");
    Compare(Run(device, Load2D, Descriptor(streamed.texels, Side, BaseLevel, LastLevel, 0u, 0u), coordinates), expected, "image_load through a T# of levels 2..8 with zero words 4-7");
}

void CheckSamples(AgcDriver::VulkanDevice& device, const Streamed& streamed) {
    const Input coordinates = [](std::uint32_t tid) {
        const auto x = (static_cast<float>((tid * 7u + 3u) % ResidentSide) + 0.5f) / static_cast<float>(ResidentSide);
        const auto y = (static_cast<float>((tid * 13u + 5u) % ResidentSide) + 0.5f) / static_cast<float>(ResidentSide);
        return std::array<std::uint32_t, 4>{std::bit_cast<std::uint32_t>(x), std::bit_cast<std::uint32_t>(y), std::bit_cast<std::uint32_t>(static_cast<float>(tid % Resident)), 0u};
    };
    const auto expected = Run(device, SampleLod2D, Descriptor(streamed.texels, ResidentSide, 0u, Resident - 1u, Resident - 1u, 0u), coordinates);
    Compare(Run(device, SampleLod2DR128, Descriptor(streamed.texels, Side, BaseLevel, LastLevel, 0u, Garbage), coordinates), expected, "image_sample_l at LODs 0-6 through a 128-bit T# of levels 2..8 of a 256x256 chain whose levels 0 and 1 are not resident");
}

}

int main() {
    std::unique_ptr<AgcDriver::VulkanDevice> device;
    std::unique_ptr<Streamed> streamed;
    try {
        device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        streamed = std::make_unique<Streamed>();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    auto& gpu = *device;
    const std::array<std::function<void()>, 2> checks{
        [&] { CheckLoads(gpu, *streamed); },
        [&] { CheckSamples(gpu, *streamed); },
    };
    int failures = 0;
    for (const auto& check : checks) {
        try {
            check();
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            ++failures;
        }
    }
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        AgcDriver::Graphics::FlushCachedTextures(gpu.Device());
    }
    gpu.WaitIdle();
    streamed.reset();
    if (failures != 0) return 1;
    std::puts("128-bit T# levels tests passed");
    return 0;
}
