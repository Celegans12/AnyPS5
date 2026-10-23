#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "Recompiler.hpp"
#include "Optimization/ResourceProgram.hpp"
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
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::ColorTargetLayout;
using AgcDriver::Graphics::ColorTileMode;
using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Width = 256;
constexpr std::uint32_t Height = 128;
constexpr std::uint32_t ScissorWidth = 128;
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t Color32FloatInfo = 0x00008710u;
constexpr std::uint32_t Color32UintInfo = 0x00010410u;
constexpr std::uint32_t LinearAttrib3 = 0x09000000u;
constexpr std::uint32_t Float32Export = 9;
constexpr std::uint32_t DrawPayloadObjPrimId = 0x9u;
constexpr std::uint32_t ObjPrimAddPipedPrimitiveId = 0x2u;
constexpr std::uint32_t ObjectIdBase = 3u << 17u;
constexpr std::uint32_t Cleared = 0xdeadbeefu;

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

constexpr std::array<std::array<float, 4>, 6> Triangles{{
    {-1.0f, -1.0f, 0.5f, 1.0f}, {1.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 1.0f, 0.5f, 1.0f},
    {1.0f, -1.0f, 0.5f, 1.0f}, {1.0f, 1.0f, 0.5f, 1.0f}, {-1.0f, 1.0f, 0.5f, 1.0f},
}};

alignas(256) std::array<std::uint32_t, 16> PixelCode{};

std::span<const std::uint32_t> Program() {
    std::size_t at = 0;
    const std::array<float, 4> color{0.25f, 0.5f, 0.75f, 1.0f};
    for (std::uint32_t channel = 0; channel < 4; ++channel) {
        PixelCode[at++] = 0x7e0002ffu | ((4u + channel) << 17u);
        PixelCode[at++] = std::bit_cast<std::uint32_t>(color[channel]);
    }
    PixelCode[at++] = 0xf800180fu;
    PixelCode[at++] = 0x07060504u;
    PixelCode[at++] = 0xbf810000u;
    return std::span<const std::uint32_t>(PixelCode.data(), at);
}

struct Block {
    explicit Block(std::size_t bytes) : bytes(bytes), watched(AgcDriver::GuestMemory::WriteWatched()) {
#ifdef _WIN32
        if (watched) {
            data = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, BlockBytes));
            GuestArena::GuestArenaCommit_nid_postfix(data, bytes, PAGE_READWRITE, bytes);
        } else {
            data = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        }
#else
        if (watched) {
            void* raw = mmap(nullptr, bytes + BlockBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            Require(raw != MAP_FAILED, "cannot map the color target");
            const auto begin = reinterpret_cast<std::uintptr_t>(raw);
            const auto aligned = (begin + BlockBytes - 1) & ~(static_cast<std::uintptr_t>(BlockBytes) - 1);
            if (aligned != begin) munmap(raw, aligned - begin);
            if (aligned + bytes != begin + bytes + BlockBytes) munmap(reinterpret_cast<void*>(aligned + bytes), begin + BlockBytes - aligned);
            data = reinterpret_cast<std::uint8_t*>(aligned);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(data, bytes);
        } else {
            data = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, bytes));
        }
#endif
        Require(data != nullptr, "cannot allocate the color target");
        GuestAllocations::Mutation mutation;
        mutation.Add(data, bytes, true, true, true);
    }
    ~Block() {
        AgcDriver::Graphics::StorageTexture::FlushPending(Address(), bytes, nullptr, "test release");
        {
            GuestAllocations::Mutation mutation;
            mutation.Remove(data);
        }
#ifdef _WIN32
        if (watched) {
            GuestArena::GuestArenaReset_nid_postfix(data, bytes);
            GuestArena::GuestArenaRelease_nid_postfix(data, bytes);
        } else {
            VirtualFree(data, 0, MEM_RELEASE);
        }
#else
        if (watched) {
            munmap(data, bytes);
            GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(data, bytes);
        } else {
            std::free(data);
        }
#endif
    }
    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(data); }
    std::size_t bytes;
    bool watched;
    std::uint8_t* data = nullptr;
};

void SetTarget(AgcDriver::Registers& cx, std::uint32_t slot, const Block& block, std::uint32_t info) {
    cx[0x318 + slot * 0xfu] = static_cast<std::uint32_t>(block.Address() >> 8u);
    cx[0x31b + slot * 0xfu] = 0;
    cx[0x31c + slot * 0xfu] = info;
    cx[0x31d + slot * 0xfu] = 0;
    cx[0x3b0 + slot] = ((Width - 1u) << 14u) | (Height - 1u);
    cx[0x3b8 + slot] = LinearAttrib3;
    cx[0x390 + slot] = static_cast<std::uint32_t>(block.Address() >> 40u);
}

std::vector<std::uint32_t> ReadBack(AgcDriver::VulkanDevice& device, const Block& block) {
    AgcDriver::Graphics::StorageTexture::FlushPending(block.Address(), block.bytes, nullptr, "test read-back");
    device.WaitIdle();
    std::vector<std::uint32_t> stored(block.bytes / 4u);
    AgcDriver::GuestMemory::Read(block.Address(), std::as_writable_bytes(std::span(stored)), 4);
    return stored;
}

void Fill(Block& block, std::uint32_t value) {
    std::vector<std::uint32_t> words(block.bytes / 4u, value);
    std::memcpy(block.data, words.data(), block.bytes);
}

void ObjPrimIdTests(AgcDriver::VulkanDevice& device) {
    Block colorBlock(ColorTargetLayout(Width, Height, ColorTileMode::Linear, 4).Bytes());
    Block idBlock(ColorTargetLayout(Width, Height, ColorTileMode::Linear, 4).Bytes());
    Fill(colorBlock, 0u);
    Fill(idBlock, Cleared);
    AgcDriver::Registers cx{{0x1b6, 0x8000}, {0x1b3, 0}, {0x1b4, 0}, {0x203, 0}, {0x1c5, Float32Export}, {0x202, 0x00cc0010u}, {0x8e, 0xffu}, {0x8f, 0xfu}, {0x2a6, DrawPayloadObjPrimId}, {0x20d, ObjPrimAddPipedPrimitiveId}};
    SetTarget(cx, 0, colorBlock, Color32FloatInfo);
    SetTarget(cx, 1, idBlock, Color32UintInfo);
    const auto output = AgcDriver::Graphics::DecodeObjPrimIdOutput(cx);
    Require(output.has_value() && output->slot == 1u && output->location == 1u, "the object/primitive ID payload did not select CB1 at location 1");
    auto color = AgcDriver::Graphics::DecodeColorBuffer(cx, 0);
    Require(color.format == VK_FORMAT_R32_SFLOAT, "CB0 did not decode as a 32-bit float target");
    auto id = AgcDriver::Graphics::DecodeColorBuffer(cx, 1);
    Require(id.format == VK_FORMAT_R32_UINT, "CB1 did not decode as a 32-bit unsigned integer target");
    id.exportIndex = output->location;
    id.objPrimId = true;

    const auto target = device.Target();
    constexpr std::uint32_t waveSize = 64;
    std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto address = reinterpret_cast<std::uintptr_t>(Triangles.data());
    const std::array<std::uint32_t, 4> vertexBuffer{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(Triangles.size()), 0x01016facu};
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {waveSize, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    const auto vertexResult = ShaderRecompiler::Recompile(vertex);
    const auto vertexPush = static_cast<std::uint32_t>(vertexResult.pushConstants.size());

    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, waveSize, waveSize, std::nullopt, std::nullopt};
    state.colors = {color, id};
    state.color = color;
    state.hasColorTarget = true;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {ScissorWidth, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = 0xf;
    VkPipelineColorBlendAttachmentState idBlend{};
    idBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    state.blends = {blend, idBlend};
    state.blend = blend;
    state.blendConstants = {};

    const auto pixelCode = Program();
    const auto codeAddress = reinterpret_cast<std::uintptr_t>(pixelCode.data());
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{codeAddress, std::as_bytes(pixelCode)}}};
    const auto pixelInfo = AgcDriver::Graphics::DecodePixelStageInfo(cx, AgcDriver::Graphics::ExportMappings(state));
    Require(pixelInfo.objPrimIdTarget == 2u, "the pixel stage did not request the object/primitive ID at location 1");
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, codeAddress, pixelCode, 0, {}},
        {waveSize, 0, {}, std::nullopt, pixelInfo, std::nullopt, pixelMemory},
        target,
        PixelPushLayout(vertexPush, target)
    };
    const auto handle = ShaderRecompiler::PrepareShader(fragment);
    AgcDriver::DriverDetail::ShaderSnapshot snapshot{codeAddress, 0, 1, {pixelCode.begin(), pixelCode.end()}, {}};
    snapshot.prepared->entries.push_back({0, handle});
    fragment.shader.code = snapshot.code;
    const ShaderRecompiler::SrtRuntime runtime{};
    const auto invocation = AgcDriver::DriverDetail::InvocationFor(snapshot, 0, fragment);
    const auto pixelResult = invocation.Materialize(*invocation.Capture(runtime));
    Require(pixelResult->readsObjPrimIdPush, "the pixel stage does not read the object/primitive ID base");
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &vertexResult, 0},
        {ShaderStage::Fragment, pixelResult.get(), PixelPushOffset(vertexPush, target)}
    }};
    AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(Triangles.size()), 0, 1, 0, false};
    draw.objPrimIdBase = ObjectIdBase;
    device.Draw(state, draw, shaders);

    const ColorTargetLayout layout(Width, Height, ColorTileMode::Linear, 4);
    const auto stored = ReadBack(device, idBlock);
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            const auto index = layout.Offset(x, y) / 4u;
            const double ndcX = 2.0 * (x + 0.5) / Width - 1.0;
            const double ndcY = 1.0 - 2.0 * (y + 0.5) / Height;
            const auto primitive = ndcX + ndcY < 0.0 ? 0u : 1u;
            const auto expected = x < ScissorWidth ? ObjectIdBase + primitive : Cleared;
            if (stored[index] == expected) continue;
            char detail[192];
            std::snprintf(detail, sizeof(detail), "object/primitive ID target: pixel (%u, %u) holds 0x%08x, expected 0x%08x", x, y, stored[index], expected);
            throw std::runtime_error(detail);
        }
    }
    const auto colors = ReadBack(device, colorBlock);
    Require(std::bit_cast<float>(colors[layout.Offset(4, 4) / 4u]) == 0.25f && colors[layout.Offset(ScissorWidth + 4, 4) / 4u] == 0u, "the exported color target beside the object/primitive ID target is wrong");
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        ObjPrimIdTests(*device);
        std::puts("object/primitive ID target tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
