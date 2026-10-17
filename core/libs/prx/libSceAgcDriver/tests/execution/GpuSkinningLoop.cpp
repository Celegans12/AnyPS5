#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t Bones = 256;
constexpr std::uint32_t BoneBytes = 48;
constexpr std::uint32_t Raw = 3u << 28u;
constexpr std::uint32_t Structured = 0u;
constexpr std::uint32_t Selects = 0xfacu;
constexpr std::uint32_t Float32 = 22u;
constexpr std::uint32_t Uint8x4 = 60u;
constexpr std::uint32_t Unorm8x4 = 56u;

struct Layout {
    std::uint32_t groups;
    std::uint32_t boneRecords;
    std::uint32_t indexRecords;
    std::uint32_t weightRecords;
    std::uint32_t iterations;
    std::uint64_t vertices;
    std::uint64_t elements;
    std::uint64_t table;
    std::uint64_t bones;
    std::uint64_t indices;
    std::uint64_t weights;
    std::uint64_t output;
    std::uint64_t bytes;
};

Layout MakeLayout(std::uint32_t groups, std::uint32_t iterations, bool partial) {
    Layout layout{};
    layout.groups = groups;
    layout.iterations = iterations;
    layout.vertices = std::uint64_t{groups} * Threads;
    layout.elements = layout.vertices + iterations;
    layout.boneRecords = Bones * BoneBytes - (partial ? 8u : 0u);
    layout.indexRecords = static_cast<std::uint32_t>(layout.elements) - (partial ? 2u : 0u);
    layout.weightRecords = static_cast<std::uint32_t>(layout.elements) - (partial ? 1u : 0u);
    const auto align = [](std::uint64_t value) { return (value + 0xffffu) & ~std::uint64_t{0xffffu}; };
    layout.table = 0;
    layout.bones = 0x10000u;
    layout.indices = align(layout.bones + Bones * BoneBytes);
    layout.weights = align(layout.indices + layout.elements * 4u);
    layout.output = align(layout.weights + layout.elements * 4u);
    layout.bytes = align(layout.output + layout.vertices * 16u);
    return layout;
}

class GuestBlock {
public:
    explicit GuestBlock(std::size_t size, std::size_t reserved = 0) : bytes(size) {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, std::max(bytes, reserved), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(0x10000u, std::max(bytes, reserved)));
#endif
        Require(block != nullptr, "gpu skinning loop: cannot allocate a guest block");
        GuestAllocations::Mutation().Add(block, bytes, true, true, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint8_t* Data() { return block; }
    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(block); }

private:
    std::uint8_t* block = nullptr;
    std::size_t bytes = 0;
};

void Mubuf(std::uint32_t op, bool idxen, bool offen, std::uint32_t immediate, std::uint32_t vaddr, std::uint32_t vdata, std::uint32_t resource, std::uint32_t soffset, std::vector<std::uint32_t>& code) {
    code.push_back(0xe0000000u | (op << 18u) | (idxen ? 0x2000u : 0u) | (offen ? 0x1000u : 0u) | immediate);
    code.push_back((soffset << 24u) | (resource / 4u << 16u) | (vdata << 8u) | vaddr);
}

std::uint32_t Vop2(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t vsrc1) {
    return (op << 25u) | (vdst << 17u) | (vsrc1 << 9u) | src0;
}

constexpr std::uint32_t Vgpr(std::uint32_t index) {
    return 256u + index;
}

constexpr std::uint32_t VAddNcU32 = 0x25u;
constexpr std::uint32_t VMulU32U24 = 0x0bu;
constexpr std::uint32_t VLshlrevB32 = 0x1au;
constexpr std::uint32_t VAndB32 = 0x1bu;

alignas(256) std::array<std::uint32_t, 512> Program{};

std::span<const std::uint32_t> Code(std::uint32_t iterations, bool formats) {
    std::vector<std::uint32_t> code;
    for (std::uint32_t resource = 0; resource < 4u; ++resource) Mubuf(14u, false, false, resource * 16u, 0u, 4u + resource * 4u, 4u, 0x80u, code);
    code.push_back(0xbf8c0000u);
    for (std::uint32_t word = 0; word < 16u; ++word) code.push_back(0x7e000400u | ((12u + word) << 17u) | Vgpr(4u + word));
    code.push_back(0x9309c008u);
    code.push_back(Vop2(VAddNcU32, 1u, 9u, 0u));
    for (std::uint32_t sum = 0; sum < 4u; ++sum) code.push_back(0x7e000280u | ((20u + sum) << 17u));
    code.push_back(0xbe8a0380u);
    code.push_back(0xbe8b03ffu);
    code.push_back(iterations);
    const auto loop = code.size();
    code.push_back(Vop2(VAddNcU32, 2u, 10u, 1u));
    if (formats) {
        Mubuf(3u, true, false, 0u, 2u, 24u, 16u, 0x80u, code);
        Mubuf(3u, true, false, 0u, 2u, 28u, 20u, 0x80u, code);
    } else {
        for (std::uint32_t lane = 0; lane < 8u; ++lane) {
            code.push_back(Vop2(VAndB32, 24u + lane, 0xffu, 2u));
            code.push_back(0x3fu + lane * 0x10u);
        }
    }
    code.push_back(0xbf8c0000u);
    for (std::uint32_t bone = 0; bone < 4u; ++bone) code.push_back(Vop2(VMulU32U24, 32u + bone, 0xb0u, 24u + bone));
    for (std::uint32_t bone = 0; bone < 4u; ++bone) {
        for (std::uint32_t row = 0; row < 3u; ++row) Mubuf(14u, false, true, row * 16u, 32u + bone, 36u + (bone * 3u + row) * 4u, 12u, 0x80u, code);
    }
    code.push_back(0xbf8c0000u);
    for (std::uint32_t load = 0; load < 12u; ++load) {
        for (std::uint32_t component = 0; component < 4u; ++component) code.push_back(Vop2(VAddNcU32, 20u + component, Vgpr(36u + load * 4u + component), 20u + component));
    }
    for (std::uint32_t bone = 0; bone < 4u; ++bone) code.push_back(Vop2(VAddNcU32, 20u + bone, Vgpr(28u + bone), 20u + bone));
    code.push_back(0x810a810au);
    code.push_back(0xbf0a0b0au);
    const auto branch = static_cast<std::int32_t>(loop) - static_cast<std::int32_t>(code.size() + 1u);
    code.push_back(0xbf850000u | (static_cast<std::uint32_t>(branch) & 0xffffu));
    code.push_back(Vop2(VLshlrevB32, 3u, 0x84u, 1u));
    Mubuf(30u, false, true, 0u, 3u, 20u, 24u, 0x80u, code);
    code.push_back(0xbf8c0000u);
    code.push_back(0xbf810000u);
    Require(code.size() <= Program.size(), "gpu skinning loop: program too long");
    std::copy(code.begin(), code.end(), Program.begin());
    return std::span<const std::uint32_t>(Program.data(), code.size());
}

std::array<std::uint32_t, 4> Descriptor(std::uint64_t base, std::uint32_t stride, std::uint32_t records, std::uint32_t word3) {
    return {static_cast<std::uint32_t>(base), static_cast<std::uint32_t>((base >> 32u) & 0xffffu) | (stride << 16u), records, word3};
}

void Prepare(GuestBlock& guest, const Layout& layout, bool unorm) {
    auto* bytes = guest.Data();
    std::memset(bytes, 0, layout.bytes);
    const auto base = guest.Address();
    const std::array<std::array<std::uint32_t, 4>, 4> descriptors{{
        Descriptor(base + layout.bones, 0u, layout.boneRecords, Selects | (Float32 << 12u) | Raw),
        Descriptor(base + layout.indices, 4u, layout.indexRecords, Selects | (Uint8x4 << 12u) | Structured),
        Descriptor(base + layout.weights, 4u, layout.weightRecords, Selects | ((unorm ? Unorm8x4 : Uint8x4) << 12u) | Structured),
        Descriptor(base + layout.output, 0u, static_cast<std::uint32_t>(layout.vertices * 16u), Selects | (Float32 << 12u) | Raw),
    }};
    std::memcpy(bytes + layout.table, descriptors.data(), sizeof(descriptors));
    auto* bones = reinterpret_cast<std::uint32_t*>(bytes + layout.bones);
    for (std::uint32_t word = 0; word < Bones * BoneBytes / 4u; ++word) bones[word] = word * 2654435761u + 0x9e37u;
    for (std::uint64_t element = 0; element < layout.elements; ++element) {
        for (std::uint32_t byte = 0; byte < 4u; ++byte) {
            bytes[layout.indices + element * 4u + byte] = static_cast<std::uint8_t>((element * 37u + byte * 91u + (element >> 5u)) & 0xffu);
            bytes[layout.weights + element * 4u + byte] = static_cast<std::uint8_t>((element * 13u + byte * 7u + 5u) & 0xffu);
        }
    }
}

std::array<std::uint32_t, 4> Expected(GuestBlock& guest, const Layout& layout, std::uint64_t vertex) {
    const auto* bytes = guest.Data();
    const auto* bones = reinterpret_cast<const std::uint32_t*>(bytes + layout.bones);
    std::array<std::uint32_t, 4> sums{};
    for (std::uint32_t iteration = 0; iteration < layout.iterations; ++iteration) {
        const auto element = vertex + iteration;
        for (std::uint32_t bone = 0; bone < 4u; ++bone) {
            const std::uint32_t index = element < layout.indexRecords ? bytes[layout.indices + element * 4u + bone] : 0u;
            for (std::uint32_t row = 0; row < 3u; ++row) {
                for (std::uint32_t component = 0; component < 4u; ++component) {
                    const auto byte = index * BoneBytes + row * 16u + component * 4u;
                    sums[component] += byte + 4u <= layout.boneRecords ? bones[byte / 4u] : 0u;
                }
            }
        }
        for (std::uint32_t bone = 0; bone < 4u; ++bone) sums[bone] += element < layout.weightRecords ? bytes[layout.weights + element * 4u + bone] : 0u;
    }
    return sums;
}

ShaderRecompiler::RecompileResult Recompile(std::span<const std::uint32_t> code, GuestBlock& guest, const ShaderRecompiler::SpirvTarget& target) {
    const auto table = guest.Address();
    const std::vector<std::uint32_t> userData{
        0u, 0u, 0u, 0u,
        static_cast<std::uint32_t>(table), static_cast<std::uint32_t>((table >> 32u) & 0xffffu), 64u, Selects | (Float32 << 12u) | Raw,
    };
    const std::array<ShaderRecompiler::MemoryRegion, 1> regions{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {true, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64u, 0, userData, compute, std::nullopt, std::nullopt, regions},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

std::uint64_t OutputHash(GuestBlock& guest, const Layout& layout) {
    const auto* words = reinterpret_cast<const std::uint32_t*>(guest.Data() + layout.output);
    std::uint64_t hash = 1469598103934665603ull;
    for (std::uint64_t word = 0; word < layout.vertices * 4u; ++word) hash = (hash ^ words[word]) * 1099511628211ull;
    return hash;
}

std::uint32_t Parameter(const char* name, std::uint32_t fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? static_cast<std::uint32_t>(std::strtoul(value, nullptr, 0)) : fallback;
}

}

int main(int argc, char** argv) {
    try {
        const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
        if (bench && std::getenv("APS5_PROFILE_GPU") != nullptr) {
            std::cerr << "gpu skinning loop: --bench does not run with APS5_PROFILE_GPU (the profiler's wait hangs this harness)\n";
            return 2;
        }
        if (bench) {
            const auto limit = Parameter("GPUCOST_TIMEOUT", 300u);
            std::thread([limit] {
                std::this_thread::sleep_for(std::chrono::seconds(limit));
                std::fprintf(stderr, "gpu skinning loop: --bench exceeded %u s, exiting without waiting for the device\n", limit);
                std::fflush(stderr);
                std::_Exit(3);
            }).detach();
        }
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto groups = bench ? Parameter("GPUCOST_GROUPS", 2048u) : 4u;
        const auto iterations = bench ? Parameter("GPUCOST_ITERATIONS", 64u) : 5u;
        const auto extraRanges = bench ? Parameter("GPUCOST_RANGES", 0u) : 64u;
        const bool unorm = bench && Parameter("GPUCOST_UNORM", 1u) != 0u;
        const auto layout = MakeLayout(groups, iterations, !bench);
        std::vector<std::unique_ptr<GuestBlock>> extras;
        for (std::uint32_t range = 0; range < extraRanges; ++range) extras.push_back(std::make_unique<GuestBlock>(0x10000u, 0x20000u));
        GuestBlock guest(layout.bytes);
        const bool formats = !bench || Parameter("GPUCOST_FORMAT", 1u) != 0u;
        const auto code = Code(iterations, formats);
        const auto shader = Recompile(code, guest, device->ComputeTarget(64));
        std::printf("gpu skinning loop: %zu SPIR-V words, %u groups, %u iterations, %u extra ranges%s\n", shader.spirv.size(), groups, iterations, extraRanges, unorm ? ", unorm weights" : "");
        Prepare(guest, layout, unorm);
        const auto repeats = bench ? Parameter("GPUCOST_REPEATS", 10u) : 1u;
        double best = 0.0;
        double total = 0.0;
        for (std::uint32_t repeat = 0; repeat < repeats; ++repeat) {
            const auto started = std::chrono::steady_clock::now();
            device->Dispatch(shader, groups, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
            device->WaitIdle();
            const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            total += elapsed;
            if (repeat == 0 || elapsed < best) best = elapsed;
        }
        const auto hash = OutputHash(guest, layout);
        if (bench) {
            std::printf("BENCH words=%zu groups=%u iterations=%u ranges=%u best_ms=%.3f mean_ms=%.3f hash=%016llx\n", shader.spirv.size(), groups, iterations, extraRanges, best, total / repeats, static_cast<unsigned long long>(hash));
            return 0;
        }
        const auto* output = reinterpret_cast<const std::uint32_t*>(guest.Data() + layout.output);
        for (std::uint64_t vertex = 0; vertex < layout.vertices; ++vertex) {
            const auto expected = Expected(guest, layout, vertex);
            for (std::uint32_t component = 0; component < 4u; ++component) {
                Require(output[vertex * 4u + component] == expected[component], "gpu skinning loop: vertex " + std::to_string(vertex) + " component " + std::to_string(component) + " is " + std::to_string(output[vertex * 4u + component]) + ", expected " + std::to_string(expected[component]));
            }
        }
        std::puts("gpu skinning loop tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
