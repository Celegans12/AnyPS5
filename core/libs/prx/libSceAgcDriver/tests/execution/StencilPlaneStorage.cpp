#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 32;
constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Format8UInt = 5;
constexpr std::uint32_t Format8UNorm = 1;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t SwizzleZ64KBX = 24;
constexpr std::uint32_t SwizzleR64KBX = 27;
constexpr std::uint8_t ClearStencil = 0x01;
constexpr std::uint8_t Written = 0x10;
constexpr std::uint8_t Mark = 0x04;
constexpr float HalfDepth = 0.5f;
constexpr std::uint32_t HalfDepthBits = 0x3f000000u;
constexpr std::size_t PlaneBytes = 65536;
constexpr std::size_t BlockBytes = 4 * PlaneBytes;

alignas(256) std::array<std::byte, Width * Height * 4> Pixels{};
alignas(256) std::array<std::uint32_t, Width * Height> Output{};
alignas(256) std::array<std::uint32_t, Width * Height * 4> TableOutput{};

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 4> PixelCode{
    0x7e0002f2, 0xf800180f, 0x00000000, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 16> MarkCode{
    0x7e080218, 0x343c0885, 0x4a3c3d00, 0x7e3e0219, 0xf0001108, 0x0001011e, 0xbf8c3f70, 0x340a3e86,
    0x4a0a0b1e, 0x340c0a82, 0xe0701000, 0x80000106, 0x38020284, 0xf0201108, 0x0001011e, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 32> MixedMarkCode{
    0xf4080100, 0xfa000000, 0xf4080200, 0xfa000010, 0xf4080700, 0xfa000020, 0x8f108202, 0xf4200444,
    0x20000000, 0x9312b011, 0xf42c0802, 0x24000000, 0x360200bf, 0x7e040202, 0xf0008f08, 0x00090401,
    0x7e120202, 0x34101286, 0x4a101100, 0x34101084, 0xe0781000, 0x80070408, 0xf40c0b00, 0xfa000040,
    0xbf8cc07f, 0xf0001108, 0x000b1401, 0xbf8c3f70, 0x38282884, 0xf0201108, 0x000b1401, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 19> ConflictingMarkCode{
    0x7e080218, 0x343c0885, 0x4a3c3d00, 0x7e3e0219, 0xf0001108, 0x0001011e, 0xbf8c3f70, 0x340a3e86,
    0x4a0a0b1e, 0x340c0a82, 0xe0701000, 0x80000106, 0x38020284, 0xf0201108, 0x0001011e, 0x7e0202f2,
    0xf0201108, 0x0003011e, 0xbf810000,
};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(PlaneBytes, BlockBytes));
#endif
        Require(block != nullptr, "stencil plane storage: cannot allocate the guest block");
        std::fill(block, block + BlockBytes, std::uint8_t{0});
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
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

    std::uint64_t Address(std::size_t offset) const { return reinterpret_cast<std::uintptr_t>(block) + offset; }

private:
    std::uint8_t* block = nullptr;
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t stride, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), count, stride == 0u ? 0x31016facu : 0x01016facu};
}

std::array<std::uint32_t, 8> PlaneDescriptor(std::uint64_t address, std::uint32_t swizzle, std::uint32_t format) {
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((Width - 1u) & 3u) << 30u),
        ((Width - 1u) >> 2u) | ((Height - 1u) << 14u),
        0x924u | (swizzle << 20u) | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

const std::vector<std::array<float, 4>>& Triangle(bool full) {
    static const std::vector<std::array<float, 4>> half{
        {-1.0f, -1.0f, 0.5f, 1.0f}, {1.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 1.0f, 0.5f, 1.0f},
    };
    static const std::vector<std::array<float, 4>> screen{
        {-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f},
    };
    return full ? screen : half;
}

struct Shaders {
    ShaderRecompiler::RecompileResult vertex;
    ShaderRecompiler::RecompileResult pixel;
    std::uint32_t vertexPush = 0;
};

Shaders CompileDraw(AgcDriver::VulkanDevice& device, bool full) {
    const auto target = device.Target();
    Shaders shaders;
    std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexBuffer = BufferDescriptor(Triangle(full).data(), 16u, static_cast<std::uint32_t>(Triangle(full).size()));
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {32, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    shaders.vertex = ShaderRecompiler::Recompile(vertex);
    shaders.vertexPush = static_cast<std::uint32_t>(shaders.vertex.pushConstants.size());

    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.wave32 = true;
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX) | ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionY);
    pixel.posX = true;
    pixel.posY = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    std::vector<std::uint32_t> pixelUserData(4, 0u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(PixelCode.data()), std::as_bytes(std::span(PixelCode))}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(PixelCode.data()), PixelCode, 0, {}},
        {32, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory},
        target,
        PixelPushLayout(shaders.vertexPush, target)
    };
    fragment.useCache = false;
    shaders.pixel = ShaderRecompiler::Recompile(fragment);
    return shaders;
}

template<class T = AgcDriver::Graphics::DepthTarget>
T Target(const GuestBlock& block, bool htileStencil, std::uint32_t swizzle = SwizzleZ64KBX, float clearDepth = 1.0f) {
    const auto depth = block.Address(0);
    const auto stencil = block.Address(PlaneBytes);
    T target{depth, stencil, {Width, Height}, VK_FORMAT_D32_SFLOAT_S8_UINT, clearDepth, ClearStencil, block.Address(2 * PlaneBytes)};
    if constexpr (requires { target.htileStencil; target.stencilSwizzle; }) {
        target.htileStencil = htileStencil;
        target.stencilSwizzle = swizzle;
    } else {
        Require(!htileStencil && swizzle == SwizzleZ64KBX, "stencil fixture requires explicit HTILE/swizzle target metadata");
    }
    return target;
}

std::vector<bool> Draw(AgcDriver::VulkanDevice& device, const Shaders& compiled, const AgcDriver::Graphics::DepthTarget& target, VkCompareOp compare, VkStencilOp pass, std::uint8_t reference, unsigned depthMode = 0) {
    Pixels.fill(std::byte{0});
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &compiled.vertex, 0},
        {ShaderStage::Fragment, &compiled.pixel, PixelPushOffset(compiled.vertexPush, device.Target())}
    }};
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 32u, 32u, std::nullopt, std::nullopt};
    state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, Pixels.size(), 0xe4u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.depth = target;
    state.depthTest = depthMode != 0;
    state.depthWrite = depthMode == 1;
    state.depthCompare = depthMode == 2 ? VK_COMPARE_OP_EQUAL : VK_COMPARE_OP_ALWAYS;
    state.stencilTest = true;
    state.stencilFront = {VK_STENCIL_OP_KEEP, pass, VK_STENCIL_OP_KEEP, compare, 0xffu, 0xffu, reference};
    state.stencilBack = state.stencilFront;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    const AgcDriver::Pm4::DrawParameters draw{0, 3u, 0, 1, 0, false};
    device.Draw(state, draw, shaders);
    device.WaitIdle();
    std::vector<bool> covered(Width * Height);
    for (std::size_t pixel = 0; pixel < covered.size(); ++pixel) covered[pixel] = std::to_integer<std::uint8_t>(Pixels[pixel * 4u]) == 255u;
    return covered;
}

void MarkPlane(AgcDriver::VulkanDevice& device, std::uint64_t address, std::uint32_t swizzle, std::uint32_t format = Format8UInt, bool conflictingAlias = false) {
    Output.fill(0xdeadbeefu);
    std::vector<std::uint32_t> userData(24, 0u);
    const auto output = BufferDescriptor(Output.data(), 0u, static_cast<std::uint32_t>(Output.size() * 4u));
    const auto plane = PlaneDescriptor(address, swizzle, format);
    std::copy(output.begin(), output.end(), userData.begin());
    std::copy(plane.begin(), plane.end(), userData.begin() + 4);
    if (conflictingAlias) {
        const auto alias = PlaneDescriptor(address, swizzle, Format8UNorm);
        std::copy(alias.begin(), alias.end(), userData.begin() + 12);
    }
    const auto code = conflictingAlias ? std::span<const std::uint32_t>(ConflictingMarkCode) : std::span<const std::uint32_t>(MarkCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {true, true, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    if (conflictingAlias) {
        std::uint32_t writtenFormats = 0;
        for (const auto& binding : result.bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.kind != ShaderRecompiler::DescriptorKind::StorageImage) continue;
            for (std::size_t element = 0; element < binding.count && element < binding.imageWritten.size() && (element + 1u) * 8u <= binding.guestDescriptor.size(); ++element) {
                if (!binding.imageWritten[element]) continue;
                const auto first = binding.guestDescriptor.begin() + element * 8u;
                if (std::equal(plane.begin(), plane.end(), first)) writtenFormats |= 1u;
                const auto alias = PlaneDescriptor(address, swizzle, Format8UNorm);
                if (std::equal(alias.begin(), alias.end(), first)) writtenFormats |= 2u;
            }
        }
        Require(writtenFormats == 3u, "conflicting stencil: both UINT and UNORM aliases must be materialized as writable storage");
    }
    device.Dispatch(result, Width / Threads, Height, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

void RequireRead(const std::vector<bool>& inside, std::uint32_t in, std::uint32_t out, const std::string& what) {
    for (std::uint32_t pixel = 0; pixel < Width * Height; ++pixel) {
        const std::uint32_t expected = inside[pixel] ? in : out;
        Require(Output[pixel] == expected, what + ": texel (" + std::to_string(pixel % Width) + ", " + std::to_string(pixel / Width) + ") read " + std::to_string(Output[pixel]) + ", expected " + std::to_string(expected));
    }
}

void MixedMarkPlane(AgcDriver::VulkanDevice& device, std::uint64_t address, const std::vector<bool>& inside, bool conflictingTable = false) {
    TableOutput.fill(0xdeadbeefu);
    alignas(256) std::array<std::uint32_t, 12> table{};
    alignas(256) std::array<std::uint32_t, Height> keys{};
    alignas(256) std::array<std::uint32_t, 24> root{};
    const auto plane = PlaneDescriptor(address, SwizzleZ64KBX, Format8UInt);
    const auto tablePlane = PlaneDescriptor(address, SwizzleZ64KBX, conflictingTable ? Format8UNorm : Format8UInt);
    std::copy(tablePlane.begin(), tablePlane.end(), table.begin() + 4);
    std::copy(plane.begin(), plane.end(), root.begin() + 16);
    auto code = MixedMarkCode;
    if (conflictingTable) code[14] = 0xf0208108u;
    const auto putBuffer = [&](const void* data, std::uint32_t bytes, std::size_t offset) {
        auto words = BufferDescriptor(data, 0u, bytes);
        words[3] = 0x31016facu;
        std::copy(words.begin(), words.end(), root.begin() + offset);
    };
    putBuffer(table.data(), sizeof(table), 0);
    putBuffer(keys.data(), sizeof(keys), 4);
    putBuffer(TableOutput.data(), sizeof(TableOutput), 8);
    const auto rootAddress = reinterpret_cast<std::uintptr_t>(root.data());
    const std::array<std::uint32_t, 2> userData{static_cast<std::uint32_t>(rootAddress), static_cast<std::uint32_t>(rootAddress >> 32u)};
    const std::array<ShaderRecompiler::MemoryRegion, 4> memory{{
        {reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(std::span(code))},
        {rootAddress, std::as_bytes(std::span(root))},
        {reinterpret_cast<std::uintptr_t>(table.data()), std::as_bytes(std::span(table))},
        {reinterpret_cast<std::uintptr_t>(keys.data()), std::as_bytes(std::span(keys))},
    }};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Width, 1, 1}, 0u, {true, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    bool directWritten = false;
    bool sampledAlias = false;
    bool tableWritten = false;
    for (const auto& binding : result.bindings) {
        for (std::size_t element = 0; element < binding.count && (element + 1u) * 8u <= binding.guestDescriptor.size(); ++element) {
            if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable && binding.kind == ShaderRecompiler::DescriptorKind::StorageImage && element < binding.imageWritten.size() && binding.imageWritten[element] && std::equal(tablePlane.begin(), tablePlane.end(), binding.guestDescriptor.begin() + element * 8u)) tableWritten = true;
            if (!std::equal(plane.begin(), plane.end(), binding.guestDescriptor.begin() + element * 8u)) continue;
            if (binding.role == ShaderRecompiler::DescriptorRole::GuestImages && binding.kind == ShaderRecompiler::DescriptorKind::StorageImage && element < binding.imageWritten.size() && binding.imageWritten[element]) directWritten = true;
            if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable && binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) sampledAlias = true;
        }
    }
    Require(directWritten && (conflictingTable ? tableWritten : sampledAlias), "mixed stencil: the direct writer and its table alias were not materialized");
    Require(result.imageTableFaults == 0u && result.poisonedSrtReads == 0u, "mixed stencil: valid descriptor sources must not fault");
    std::puts(conflictingTable ? "mixed stencil: conflicting direct and table writers materialized" : "mixed stencil: direct writable storage and sampled table alias materialized");
    device.Dispatch(result, Height, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    if (conflictingTable) return;
    for (std::uint32_t pixel = 0; pixel < Width * Height; ++pixel) {
        const auto expected = inside[pixel] ? Written : ClearStencil;
        for (std::uint32_t component = 0; component < 4; ++component) {
            Require(TableOutput[pixel * 4u + component] == expected, "mixed stencil: sampled table alias did not read the old stencil seed at pixel " + std::to_string(pixel) + ", component " + std::to_string(component));
        }
    }
    std::puts("mixed stencil: sampled table alias read the old stencil seed");
}

void RequireCoverage(const std::vector<bool>& covered, const std::vector<bool>& expected, const std::string& what) {
    for (std::uint32_t pixel = 0; pixel < Width * Height; ++pixel) {
        Require(covered[pixel] == expected[pixel], what + ": pixel (" + std::to_string(pixel % Width) + ", " + std::to_string(pixel / Width) + ") " + (covered[pixel] ? "passed" : "failed") + " the stencil test");
    }
}

void RequireRefused(AgcDriver::VulkanDevice& device, std::uint64_t address, std::uint32_t swizzle, const std::string& reason, const std::string& what) {
    std::string refusal;
    try {
        MarkPlane(device, address, swizzle);
    } catch (const std::exception& error) {
        refusal = error.what();
    }
    Require(refusal.find(reason) != std::string::npos, what + " was not refused: " + refusal);
}

}

int main() {
    try {
        GuestBlock block;
        GuestBlock tiled;
        GuestBlock mixed;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto shaders = CompileDraw(*device, false);
        const auto screen = CompileDraw(*device, true);
        const auto target = Target(block, false);
        const auto inside = Draw(*device, shaders, target, VK_COMPARE_OP_ALWAYS, VK_STENCIL_OP_REPLACE, Written, 1);
        const auto count = static_cast<std::size_t>(std::count(inside.begin(), inside.end(), true));
        Require(count != 0 && count != inside.size(), "the stencil draw covered " + std::to_string(count) + " of " + std::to_string(inside.size()) + " pixels");
        std::vector<bool> outside(inside.size());
        std::transform(inside.begin(), inside.end(), outside.begin(), [](bool value) { return !value; });
        RequireCoverage(Draw(*device, screen, target, VK_COMPARE_OP_ALWAYS, VK_STENCIL_OP_KEEP, 0, 2), inside, "depth EQUAL 0.5 before stencil storage");
        std::puts("stencil seed complete: distinct stencil and depth inside/outside the triangle");
        std::fflush(stdout);

        MarkPlane(*device, target.stencilAddress, SwizzleZ64KBX);
        RequireRead(inside, Written, ClearStencil, "image_load of the stencil plane after a stencil draw");
        RequireCoverage(Draw(*device, screen, target, VK_COMPARE_OP_ALWAYS, VK_STENCIL_OP_KEEP, 0, 2), inside, "depth EQUAL 0.5 preserved after stencil image_store OR 0x04");
        AgcDriver::Graphics::NoteHtileDepthClear(target.htileAddress);
        RequireCoverage(Draw(*device, screen, target, VK_COMPARE_OP_EQUAL, VK_STENCIL_OP_KEEP, Written | Mark), inside, "stencil EQUAL 0x14 after image_store OR 0x04 and a depth fast clear");
        RequireCoverage(Draw(*device, screen, target, VK_COMPARE_OP_EQUAL, VK_STENCIL_OP_KEEP, ClearStencil | Mark), outside, "stencil EQUAL 0x05 after image_store OR 0x04");
        RequireCoverage(Draw(*device, screen, target, VK_COMPARE_OP_EQUAL, VK_STENCIL_OP_KEEP, Written), std::vector<bool>(inside.size(), false), "stencil EQUAL 0x10 after image_store OR 0x04");

        Draw(*device, shaders, target, VK_COMPARE_OP_ALWAYS, VK_STENCIL_OP_INCREMENT_AND_CLAMP, 0);
        MarkPlane(*device, target.stencilAddress, SwizzleZ64KBX);
        RequireRead(inside, (Written | Mark) + 1u, ClearStencil | Mark, "image_load of the stencil plane after a stencil increment inside the triangle");

        const auto mixedTarget = Target(mixed, false);
        RequireCoverage(Draw(*device, shaders, mixedTarget, VK_COMPARE_OP_ALWAYS, VK_STENCIL_OP_REPLACE, Written, 1), inside, "mixed stencil seed coverage");
        MixedMarkPlane(*device, mixedTarget.stencilAddress, inside);
        RequireCoverage(Draw(*device, screen, mixedTarget, VK_COMPARE_OP_ALWAYS, VK_STENCIL_OP_KEEP, 0, 2), inside, "mixed stencil storage preserves depth EQUAL 0.5");
        RequireCoverage(Draw(*device, screen, mixedTarget, VK_COMPARE_OP_EQUAL, VK_STENCIL_OP_KEEP, Written | Mark), inside, "mixed stencil storage OR 0x04 copies back inside the triangle");
        RequireCoverage(Draw(*device, screen, mixedTarget, VK_COMPARE_OP_EQUAL, VK_STENCIL_OP_KEEP, ClearStencil | Mark), outside, "mixed stencil storage OR 0x04 copies back outside the triangle");
        std::puts("mixed stencil: image_store copyback and unchanged depth verified");

        std::string conflict;
        try {
            MarkPlane(*device, mixedTarget.stencilAddress, SwizzleZ64KBX, Format8UInt, true);
        } catch (const std::exception& error) {
            conflict = error.what();
        }
        Require(conflict == "AGC graphics: conflicting writable storage images alias the same depth/stencil plane", "conflicting stencil aliases did not fail at the pre-dispatch ownership guard: " + conflict);
        Require(std::all_of(Output.begin(), Output.end(), [](auto value) { return value == 0xdeadbeefu; }), "conflicting stencil aliases executed the shader's buffer store before refusal");
        RequireCoverage(Draw(*device, screen, mixedTarget, VK_COMPARE_OP_ALWAYS, VK_STENCIL_OP_KEEP, 0, 2), inside, "conflicting stencil refusal preserves depth");
        RequireCoverage(Draw(*device, screen, mixedTarget, VK_COMPARE_OP_EQUAL, VK_STENCIL_OP_KEEP, Written | Mark), inside, "conflicting stencil refusal preserves inside stencil");
        RequireCoverage(Draw(*device, screen, mixedTarget, VK_COMPARE_OP_EQUAL, VK_STENCIL_OP_KEEP, ClearStencil | Mark), outside, "conflicting stencil refusal preserves outside stencil");
        std::puts("conflicting stencil aliases refused before shader writes; stencil and depth unchanged");

        conflict.clear();
        try {
            MixedMarkPlane(*device, mixedTarget.stencilAddress, inside, true);
        } catch (const std::exception& error) {
            conflict = error.what();
        }
        Require(conflict == "AGC graphics: conflicting writable storage images alias the same depth/stencil plane", "conflicting stencil table did not fail at the pre-dispatch ownership guard: " + conflict);
        Require(std::all_of(TableOutput.begin(), TableOutput.end(), [](auto value) { return value == 0xdeadbeefu; }), "conflicting stencil table executed the shader's buffer store before refusal");
        RequireCoverage(Draw(*device, screen, mixedTarget, VK_COMPARE_OP_ALWAYS, VK_STENCIL_OP_KEEP, 0, 2), inside, "conflicting stencil table refusal preserves depth");
        RequireCoverage(Draw(*device, screen, mixedTarget, VK_COMPARE_OP_EQUAL, VK_STENCIL_OP_KEEP, Written | Mark), inside, "conflicting stencil table refusal preserves inside stencil");
        RequireCoverage(Draw(*device, screen, mixedTarget, VK_COMPARE_OP_EQUAL, VK_STENCIL_OP_KEEP, ClearStencil | Mark), outside, "conflicting stencil table refusal preserves outside stencil");
        std::puts("conflicting stencil table refused before shader writes; stencil and depth unchanged");

        RequireRefused(*device, target.stencilAddress, SwizzleR64KBX, "storage image access to the stencil plane", "an 8_UINT view in SW_64KB_R_X of a SW_64KB_Z_X stencil plane");
        const auto htile = Target(tiled, true, SwizzleZ64KBX, HalfDepth);
        Draw(*device, shaders, htile, VK_COMPARE_OP_ALWAYS, VK_STENCIL_OP_REPLACE, Written);
        RequireRefused(*device, htile.stencilAddress, SwizzleZ64KBX, "kept in HTILE", "storage access to a stencil plane kept in HTILE");
        MarkPlane(*device, htile.address, SwizzleZ64KBX, Format32UInt);
        RequireRead(inside, HalfDepthBits, HalfDepthBits, "image_load of the depth plane");

        const auto rebound = Target(block, false, SwizzleR64KBX);
        Draw(*device, shaders, rebound, VK_COMPARE_OP_ALWAYS, VK_STENCIL_OP_REPLACE, Written);
        MarkPlane(*device, rebound.stencilAddress, SwizzleR64KBX);
        RequireRead(inside, Written, ClearStencil, "image_load in SW_64KB_R_X of the stencil plane rebound with that SW_MODE");
        RequireRefused(*device, rebound.stencilAddress, SwizzleZ64KBX, "storage image access to the stencil plane", "an 8_UINT view in SW_64KB_Z_X of a stencil plane rebound in SW_64KB_R_X");
        std::puts("stencil plane storage tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
