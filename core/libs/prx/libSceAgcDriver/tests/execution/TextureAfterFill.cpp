#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DrawPipeline;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Words = 32;
constexpr std::uint32_t Width = 32;
constexpr std::uint32_t Levels = 2;
constexpr std::uint32_t Format32x2Uint = 62;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t IdentitySwizzle = 0xfacu;
constexpr std::uint32_t TexelBytes = 8;
constexpr std::uint32_t FilledTexels = 16;
constexpr std::size_t BlockBytes = 65536;

constexpr std::array<std::uint32_t, 55> Program{
    0x34060087, 0x7e3c0300, 0x7e3e0280, 0x36420087, 0x7e440280, 0xd5480023, 0x02050700, 0xf0081f08,
    0x00010a1e, 0xf00c1f08, 0x00010e1e, 0xf0081a08, 0x0001121e, 0xf00c1408, 0x0001141e, 0xf0101f08,
    0x00011521, 0xf0141308, 0x00011921, 0xbf8c3f70, 0xe0701000, 0x80000a03, 0xe0701004, 0x80000b03,
    0xe0701008, 0x80000c03, 0xe070100c, 0x80000d03, 0xe0701010, 0x80000e03, 0xe0701014, 0x80000f03,
    0xe0701018, 0x80001003, 0xe070101c, 0x80001103, 0xe0701020, 0x80001203, 0xe0701024, 0x80001303,
    0xe0701028, 0x80001403, 0xe070102c, 0x80001503, 0xe0701030, 0x80001603, 0xe0701034, 0x80001703,
    0xe0701038, 0x80001803, 0xe070103c, 0x80001903, 0xe0701040, 0x80001a03, 0xbf810000,
};

std::uint32_t* Code = nullptr;
std::uint32_t* Texels = nullptr;
std::uint32_t* Results = nullptr;
std::uint32_t* Other = nullptr;
std::uint32_t* StoreCode = nullptr;
std::uint32_t* Source = nullptr;
constexpr std::size_t LargeFillBytes = 2 * BlockBytes;
constexpr std::array<std::uint32_t, 8> StoreProgram{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u, 0x4a060481u, 0xe0701000u, 0x80010301u, 0xbf810000u};

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("texture after fill: " + what);
}

std::uint32_t* NewBlock(std::size_t blockBytes = BlockBytes) {
    static std::uintptr_t next = 0x5b0000000ull;
    const auto wanted = next;
    next += blockBytes;
#ifdef _WIN32
    auto* block = static_cast<std::uint32_t*>(VirtualAlloc(reinterpret_cast<void*>(wanted), blockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (block == nullptr) block = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, blockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    static_cast<void>(wanted);
    auto* block = static_cast<std::uint32_t*>(std::aligned_alloc(BlockBytes, blockBytes));
#endif
    check(block != nullptr, "cannot allocate a guest block");
    std::memset(block, 0, blockBytes);
    GuestAllocations::Mutation().Add(block, blockBytes, true, true, false);
    return block;
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::array<std::uint32_t, 8> TextureDescriptor(const void* data) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32x2Uint << 20u) | (((Width - 1u) & 3u) << 30u),
        (Width - 1u) >> 2u,
        IdentitySwizzle | ((Levels - 1u) << 16u) | (Type2D << 28u),
        0u,
        (Levels - 1u) << 4u,
        0u,
        0u,
    };
}

void AppendFill(std::vector<std::uint32_t>& words, std::uint32_t value, const void* destination, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(destination);
    const std::array<std::uint32_t, 7> fill{0xc0055000u, 2u << 29u, value, 0u, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), bytes};
    words.insert(words.end(), fill.begin(), fill.end());
}

void AppendDispatch(std::vector<std::uint32_t>& words) {
    const auto codeAddress = reinterpret_cast<std::uintptr_t>(Code);
    const auto buffer = BufferDescriptor(Results, Threads * Words * 4u);
    const auto texture = TextureDescriptor(Texels);
    std::vector<std::uint32_t> user(buffer.begin(), buffer.end());
    user.insert(user.end(), texture.begin(), texture.end());
    user.resize(16, 0u);
    const std::vector<std::uint32_t> head{
        0xc0027600u, 0x20cu, static_cast<std::uint32_t>(codeAddress >> 8u), static_cast<std::uint32_t>(codeAddress >> 40u),
        0xc0037600u, 0x207u, Threads, 1u, 1u,
        0xc0017600u, 0x213u, 16u << 1u,
        0xc0107600u, 0x240u,
    };
    words.insert(words.end(), head.begin(), head.end());
    words.insert(words.end(), user.begin(), user.end());
    words.insert(words.end(), {0xc0031500u, 1u, 1u, 1u, 0x8041u});
}

void AppendStore(std::vector<std::uint32_t>& words) {
    const auto codeAddress = reinterpret_cast<std::uintptr_t>(StoreCode);
    const auto input = BufferDescriptor(Source, FilledTexels * TexelBytes);
    const auto output = BufferDescriptor(Texels, FilledTexels * TexelBytes);
    const std::vector<std::uint32_t> commands{
        0xc0027600u, 0x20cu, static_cast<std::uint32_t>(codeAddress >> 8u), static_cast<std::uint32_t>(codeAddress >> 40u),
        0xc0037600u, 0x207u, FilledTexels * TexelBytes / 4u, 1u, 1u,
        0xc0017600u, 0x213u, 8u << 1u,
        0xc0087600u, 0x240u, input[0], input[1], input[2], input[3], output[0], output[1], output[2], output[3],
        0xc0031500u, 1u, 1u, 1u, 0x8041u,
    };
    words.insert(words.end(), commands.begin(), commands.end());
}

void Submit(std::vector<std::uint32_t>& words) {
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&packet) == 0, "submission failed");
}

void Settle() {
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(Results), Threads * Words * 4u);
}

void ExpectTexels(std::uint32_t expected, const std::string& what) {
    for (std::uint32_t tid = 0; tid < FilledTexels; ++tid) {
        const auto actual = Results[tid * Words];
        check(actual == expected, what + ": thread " + std::to_string(tid) + " read " + std::to_string(actual) + ", expected " + std::to_string(expected));
    }
}

}

int main() {
#ifdef _WIN32
    _putenv_s("APS5_PIPELINED_DRAWS", "1");
#else
    setenv("APS5_PIPELINED_DRAWS", "1", 1);
#endif
    try {
        {
            const auto probe = OpenVulkanTestDevice();
            if (!probe) return VulkanTestSkipped;
        }
        Code = NewBlock();
        std::copy(Program.begin(), Program.end(), Code);
        Texels = NewBlock(LargeFillBytes);
        StoreCode = NewBlock();
        std::copy(StoreProgram.begin(), StoreProgram.end(), StoreCode);
        Source = NewBlock();
        Results = NewBlock();
        Other = NewBlock();
        Shader shader{};
        shader.file_header = 0x34333231;
        shader.version = 0x18;
        shader.header_size = sizeof(Shader);
        shader.shader_size = static_cast<std::uint32_t>(Program.size() * 4u);
        shader.code = Code;
        AgcDriverRegisterShader_nid_postfix(&shader);
        Shader store{};
        store.file_header = 0x34333231;
        store.version = 0x18;
        store.header_size = sizeof(Shader);
        store.shader_size = static_cast<std::uint32_t>(StoreProgram.size() * 4u);
        store.code = StoreCode;
        AgcDriverRegisterShader_nid_postfix(&store);
        constexpr std::uint32_t First = 0x01010101u;
        for (std::uint32_t word = 0; word < Width * TexelBytes / 4u * 2u; ++word) Texels[word] = First;
        {
            std::vector<std::uint32_t> words;
            AppendDispatch(words);
            Submit(words);
            Settle();
            ExpectTexels(First, "first read of the texture");
        }
        const char* only = std::getenv("TEXTURE_AFTER_FILL_ONLY");
        struct Way {
            const char* name;
            int kind;
            bool separate;
        };
        const std::array<Way, 5> ways{{
            {"small DMA_DATA fill, same submission", 0, false},
            {"small DMA_DATA fill, finished first", 0, true},
            {"large DMA_DATA fill, same submission", 1, false},
            {"large DMA_DATA fill, finished first", 1, true},
            {"compute buffer store, same submission", 2, false},
        }};
        std::uint32_t value = 0x02020202u;
        std::string failures;
        for (const auto& way : ways) {
            if (only != nullptr && std::string(only) != std::to_string(&way - ways.data())) continue;
            value += 0x01010101u;
            for (std::uint32_t word = 0; word < Threads * Words; ++word) Results[word] = 0xdeadbeefu;
            std::vector<std::uint32_t> words;
            if (way.kind == 0) AppendFill(words, value, Texels, FilledTexels * TexelBytes);
            else if (way.kind == 1) AppendFill(words, value, Texels, static_cast<std::uint32_t>(LargeFillBytes));
            else {
                for (std::uint32_t word = 0; word < FilledTexels * TexelBytes / 4u; ++word) Source[word] = value - 1u;
                AppendStore(words);
            }
            if (way.separate) {
                Submit(words);
                Settle();
                words.clear();
            }
            AppendDispatch(words);
            Submit(words);
            Settle();
            AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(Texels), FilledTexels * TexelBytes);
            check(Texels[0] == value, std::string(way.name) + ": the write did not reach the texture's memory (" + std::to_string(Texels[0]) + ")");
            bool stale = false;
            for (std::uint32_t tid = 0; tid < FilledTexels; ++tid) stale = stale || Results[tid * Words] != value;
            std::printf("texture after fill: %s: memory holds %u, the dispatch sampled %u%s\n", way.name, Texels[0], Results[0], stale ? " (STALE)" : "");
            if (stale) failures += std::string("; ") + way.name;
        }
        check(failures.empty(), "the dispatch sampled the texture as it was before the write of its memory:" + failures.substr(1));
        LibcRunShutdown_nid_postfix();
        std::puts("texture after fill tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
