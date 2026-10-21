#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
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

constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t InputBytes = Lanes * 4u;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Format16UInt = 11;
alignas(256) std::array<std::uint32_t, Lanes> Input{};
alignas(256) std::array<std::uint32_t, Lanes * 4u + 4u> Output{};
alignas(256) std::array<std::uint32_t, Lanes> Second{};

constexpr std::array<std::uint32_t, 12> LoadCode(std::uint32_t marker) {
    return {0x34020082u, 0xe0001000u, 0x80000201u, 0xe0301000u, 0x80000301u, 0xbf8c0000u,
            0xe0701000u, 0x80010201u, 0xe0701080u + marker * 4u, 0x80010301u, 0xbf800000u | marker, 0xbf810000u};
}

alignas(256) constexpr auto BoundFirstCode = LoadCode(0u);
alignas(256) constexpr auto NullFirstCode = LoadCode(1u);
alignas(256) constexpr auto FormatsCode = LoadCode(2u);

std::array<std::uint32_t, 4> Descriptor(const void* data, std::uint32_t records, std::uint32_t format, std::uint32_t stride = 0u) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    const auto oobSelect = stride != 0u ? 0u : 3u;
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), records, (0x01016facu & ~(0x7fu << 12u)) | (format << 12u) | (oobSelect << 28u)};
}

constexpr std::array<std::uint32_t, 4> GameNull{0x00000000u, 0x00080050u, 0x00000000u, 0x00005204u};
constexpr std::array<std::uint32_t, 4> ZeroNull{0u, 0u, 0u, 0u};

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

struct Outcome {
    std::uint64_t pipeline = 0;
    std::vector<std::uint32_t> spirv;
};

Outcome Run(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, std::uint32_t marker, const std::array<std::uint32_t, 4>& optional, const std::string& name, std::uint32_t mask) {
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) Input[lane] = 0x9e3779b9u * (lane + 1u) ^ 0x01234567u;
    Output.fill(Fill);
    std::vector<std::uint32_t> userData(optional.begin(), optional.end());
    const auto output = Descriptor(Output.data(), sizeof(Output), Format32UInt);
    userData.insert(userData.end(), output.begin(), output.end());
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {Lanes, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = true;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    const bool present = optional[2] != 0u;
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        const auto formatted = present ? Input[lane] & mask : 0u;
        const auto raw = present ? Input[lane] : 0u;
        Require(Output[lane] == formatted, name + ": lane " + std::to_string(lane) + " buffer_load_format_x is " + Hex(Output[lane]) + ", expected " + Hex(formatted));
        Require(Output[Lanes + marker + lane] == raw, name + ": lane " + std::to_string(lane) + " buffer_load_dword is " + Hex(Output[Lanes + marker + lane]) + ", expected " + Hex(raw));
    }
    Outcome outcome;
    outcome.pipeline = result.PipelineVariantId();
    outcome.spirv = result.spirv.Copy();
    Require(outcome.pipeline != 0u && !outcome.spirv.empty(), name + ": no specialized module");
    return outcome;
}

void BoundThenNull(AgcDriver::VulkanDevice& device) {
    const auto bound = Run(device, BoundFirstCode, 0u, Descriptor(Input.data(), InputBytes, Format32UInt), "bound", 0xffffffffu);
    const auto absent = Run(device, BoundFirstCode, 0u, GameNull, "null V#", 0xffffffffu);
    Require(absent.pipeline == bound.pipeline && absent.spirv == bound.spirv, "a null V# in an optional slot made another module than the bound slot");
    const auto zero = Run(device, BoundFirstCode, 0u, ZeroNull, "zero V#", 0xffffffffu);
    Require(zero.pipeline == bound.pipeline, "a zero V# made another module than the bound slot");
    const auto again = Run(device, BoundFirstCode, 0u, Descriptor(Input.data(), InputBytes, Format32UInt), "bound again", 0xffffffffu);
    Require(again.pipeline == bound.pipeline, "the bound slot changed modules after a null V#");
    const auto reformatted = Run(device, BoundFirstCode, 0u, Descriptor(Input.data(), 1u, Format16UInt, InputBytes), "bound 16_UINT structured stride 128", 0xffffu);
    Require(reformatted.pipeline != bound.pipeline && reformatted.spirv != bound.spirv, "a bound slot with another format and stride shared the substitute's module");
    const auto absentAfter = Run(device, BoundFirstCode, 0u, GameNull, "null V# after another format", 0xffffffffu);
    Require(absentAfter.pipeline == bound.pipeline, "a null V# did not keep the first bound slot's substitute");
}

bool RuntimeFormatEnabled() {
    return std::getenv("APS5_NO_ABSENT_RUNTIME_FORMAT") == nullptr;
}

void NullThenBound(AgcDriver::VulkanDevice& device) {
    const auto first = Run(device, NullFirstCode, 1u, GameNull, "null V# before any bound slot", 0xffffffffu);
    const auto bound = Run(device, NullFirstCode, 1u, Descriptor(Input.data(), InputBytes, Format32UInt), "bound after null", 0xffffffffu);
    if (RuntimeFormatEnabled()) Require(first.pipeline == bound.pipeline && first.spirv == bound.spirv, "a slot never seen bound did not give its module to the bound slot");
    else Require(first.pipeline != bound.pipeline, "a slot never seen bound did not specialize as absent");
    const auto learned = Run(device, NullFirstCode, 1u, ZeroNull, "zero V# after the slot was bound", 0xffffffffu);
    Require(learned.pipeline == bound.pipeline && learned.spirv == bound.spirv, "a null V# did not take the substitute learned from the bound slot");
}

void AbsentThenFormats(AgcDriver::VulkanDevice& device) {
    const auto absent = Run(device, FormatsCode, 2u, GameNull, "null V# before any format", 0xffffffffu);
    const auto wide = Run(device, FormatsCode, 2u, Descriptor(Input.data(), InputBytes, Format32UInt), "32_UINT after null", 0xffffffffu);
    const auto narrow = Run(device, FormatsCode, 2u, Descriptor(Input.data(), 1u, Format16UInt, InputBytes), "16_UINT structured stride 128 after 32_UINT", 0xffffu);
    if (RuntimeFormatEnabled()) {
        Require(absent.pipeline == wide.pipeline && absent.spirv == wide.spirv, "the first format built a module beside the absent slot's");
        Require(absent.pipeline == narrow.pipeline && absent.spirv == narrow.spirv, "the second format built a module beside the absent slot's");
    } else {
        Require(absent.pipeline != wide.pipeline && wide.pipeline != narrow.pipeline && absent.pipeline != narrow.pipeline, "APS5_NO_ABSENT_RUNTIME_FORMAT did not build one module per format word");
    }
}

constexpr std::array<std::uint32_t, 8> SingleCode(std::uint32_t marker) {
    return {0x34020082u, 0xe0001000u, 0x80000201u, 0xbf8c0000u, 0xe0701000u, 0x80010201u, 0xbf800000u | marker, 0xbf810000u};
}

constexpr std::array<std::uint32_t, 12> TwoCode(std::uint32_t marker) {
    return {0x34020082u, 0xe0001000u, 0x80000201u, 0xe0001000u, 0x80010301u, 0xbf8c0000u,
            0xe0701000u, 0x80020201u, 0xe0701080u, 0x80020301u, 0xbf800000u | marker, 0xbf810000u};
}

constexpr std::array<std::uint32_t, 14> XyzwCode(std::uint32_t marker) {
    return {0x34020082u, 0xe00c1000u, 0x80000201u, 0xbf8c0000u, 0xe0701000u, 0x80010201u, 0xe0701080u,
            0x80010301u, 0xe0701100u, 0x80010401u, 0xe0701180u, 0x80010501u, 0xbf800000u | marker, 0xbf810000u};
}

alignas(256) constexpr auto SingleAbsentCode = SingleCode(10u);
alignas(256) constexpr auto TwoBuffersCode = TwoCode(12u);
alignas(256) constexpr auto XyzwRuntimeCode = XyzwCode(13u);
alignas(256) constexpr auto XyzwIdentityReferenceCode = XyzwCode(14u);
alignas(256) constexpr auto XyzwSelectReferenceCode = XyzwCode(15u);

struct Wide {
    Outcome outcome;
    std::vector<std::uint32_t> words;
};

Wide Dispatch(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::vector<std::array<std::uint32_t, 4>>& slots) {
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        Input[lane] = 0x9e3779b9u * (lane + 1u) ^ 0x01234567u;
        Second[lane] = Input[lane] ^ 0x5a5a5a5au;
    }
    Output.fill(Fill);
    std::vector<std::uint32_t> userData;
    for (const auto& slot : slots) userData.insert(userData.end(), slot.begin(), slot.end());
    const auto output = Descriptor(Output.data(), sizeof(Output), Format32UInt);
    userData.insert(userData.end(), output.begin(), output.end());
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {Lanes, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = true;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    Wide wide;
    wide.outcome.pipeline = result.PipelineVariantId();
    wide.outcome.spirv = result.spirv.Copy();
    wide.words.assign(Output.begin(), Output.begin() + Lanes * 4u);
    Require(wide.outcome.pipeline != 0u && !wide.outcome.spirv.empty(), "no specialized module");
    return wide;
}

void RequireSameModule(const Wide& left, const Wide& right, const std::string& name) {
    if (RuntimeFormatEnabled()) Require(left.outcome.pipeline == right.outcome.pipeline && left.outcome.spirv == right.outcome.spirv, name);
}

void TwoFormattedBuffers(AgcDriver::VulkanDevice& device) {
    const auto wide = Descriptor(Input.data(), InputBytes, Format32UInt);
    const auto narrow = Descriptor(Second.data(), 1u, Format16UInt, InputBytes);
    const auto check = [&](const Wide& run, bool first, bool second, const std::string& name) {
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const auto a = first ? Input[lane] : 0u;
            const auto b = second ? Second[lane] & 0xffffu : 0u;
            Require(run.words[lane] == a && run.words[Lanes + lane] == b, name + ": lane " + std::to_string(lane) + " loaded " + Hex(run.words[lane]) + " and " + Hex(run.words[Lanes + lane]) + ", expected " + Hex(a) + " and " + Hex(b));
        }
    };
    const auto none = Dispatch(device, TwoBuffersCode, {GameNull, GameNull});
    check(none, false, false, "both V# absent");
    const auto one = Dispatch(device, TwoBuffersCode, {wide, GameNull});
    check(one, true, false, "first V# present");
    const auto both = Dispatch(device, TwoBuffersCode, {wide, narrow});
    check(both, true, true, "both V# present");
    const auto learned = Dispatch(device, TwoBuffersCode, {ZeroNull, ZeroNull});
    check(learned, false, false, "both V# learned");
    RequireSameModule(none, one, "a first V# made another module than the absent pair");
    RequireSameModule(none, both, "both V# made another module than the absent pair");
    RequireSameModule(none, learned, "the learned pair made another module than the absent pair");
    if (!RuntimeFormatEnabled()) Require(none.outcome.pipeline != both.outcome.pipeline, "APS5_NO_ABSENT_RUNTIME_FORMAT shared a module between the absent and the present pair");
}

std::array<std::uint32_t, 4> WithSelectors(std::array<std::uint32_t, 4> descriptor, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t w) {
    descriptor[3] = (descriptor[3] & ~0xfffu) | x | (y << 3u) | (z << 6u) | (w << 9u);
    return descriptor;
}

void AbsentLoadsZero(AgcDriver::VulkanDevice& device) {
    const auto identity = Descriptor(Input.data(), InputBytes, Format32UInt);
    const auto selected = WithSelectors(identity, 0u, 1u, 4u, 0u);
    const auto absent = Dispatch(device, XyzwRuntimeCode, {GameNull});
    for (std::uint32_t index = 0; index < absent.words.size(); ++index) Require(absent.words[index] == 0u, "load_format_xyzw through an absent V# returned " + Hex(absent.words[index]) + " at word " + std::to_string(index));
    const auto referenceIdentity = Dispatch(device, XyzwIdentityReferenceCode, {identity});
    const auto referenceSelected = Dispatch(device, XyzwSelectReferenceCode, {selected});
    const auto present = Dispatch(device, XyzwRuntimeCode, {identity});
    Require(present.words == referenceIdentity.words, "a present V# through the run-time-format module loaded other values than its own module");
    const auto remapped = Dispatch(device, XyzwRuntimeCode, {selected});
    Require(remapped.words == referenceSelected.words, "a ONE/ZERO destination select through the run-time-format module loaded other values than its own module");
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        Require(remapped.words[lane] == 0u && remapped.words[Lanes + lane] != 0u && remapped.words[Lanes * 2u + lane] == Input[lane] && remapped.words[Lanes * 3u + lane] == 0u, "a ONE/ZERO destination select loaded wrong values at lane " + std::to_string(lane));
    }
    RequireSameModule(absent, present, "a present V# made another module than the absent one");
    RequireSameModule(absent, remapped, "a remapped V# made another module than the absent one");
}

void MisalignedPresent(AgcDriver::VulkanDevice& device) {
    const auto absent = Dispatch(device, SingleAbsentCode, {GameNull});
    Require(std::ranges::all_of(absent.words, [](std::uint32_t word) { return word == 0u || word == Fill; }), "an absent V# loaded a value");
    const auto misaligned = Descriptor(reinterpret_cast<const std::uint8_t*>(Input.data()) + 2, InputBytes, Format16UInt);
    const auto shifted = Dispatch(device, SingleAbsentCode, {misaligned});
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) Require(shifted.words[lane] == ((Input[lane] >> 16u) & 0xffffu), "a misaligned V# loaded " + Hex(shifted.words[lane]) + " at lane " + std::to_string(lane));
    if (RuntimeFormatEnabled()) Require(shifted.outcome.pipeline != absent.outcome.pipeline, "a misaligned V# reused a module built for an aligned base");
    const auto aligned = Dispatch(device, SingleAbsentCode, {Descriptor(Input.data(), InputBytes, Format16UInt)});
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) Require(aligned.words[lane] == (Input[lane] & 0xffffu), "an aligned V# loaded " + Hex(aligned.words[lane]) + " at lane " + std::to_string(lane));
    RequireSameModule(absent, aligned, "an aligned V# made another module than the absent one");
}

constexpr std::uint32_t ManyBuffers = 8;
alignas(256) std::array<std::uint32_t, Lanes * ManyBuffers + 4u> ManyOutput{};

constexpr std::array<std::uint32_t, 3 + ManyBuffers * 4> ManyCode() {
    std::array<std::uint32_t, 3 + ManyBuffers * 4> code{};
    std::size_t cursor = 0;
    code[cursor++] = 0x34020082u;
    for (std::uint32_t index = 0; index < ManyBuffers; ++index) {
        code[cursor++] = 0xe0001000u;
        code[cursor++] = 0x80000001u | (index << 16u) | ((2u + index) << 8u);
    }
    code[cursor++] = 0xbf8c0000u;
    for (std::uint32_t index = 0; index < ManyBuffers; ++index) {
        code[cursor++] = 0xe0701000u + index * 0x80u;
        code[cursor++] = 0x80000001u | (ManyBuffers << 16u) | ((2u + index) << 8u);
    }
    code[cursor++] = 0xbf810000u;
    return code;
}

alignas(256) constexpr auto ManyBuffersCode = ManyCode();

Outcome DispatchMany(AgcDriver::VulkanDevice& device, std::uint32_t presentMask, const std::string& name) {
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) Input[lane] = 0x9e3779b9u * (lane + 1u) ^ 0x01234567u;
    ManyOutput.fill(Fill);
    std::vector<std::uint32_t> userData;
    for (std::uint32_t index = 0; index < ManyBuffers; ++index) {
        const auto slot = (presentMask >> index) & 1u ? Descriptor(Input.data(), InputBytes, Format32UInt) : GameNull;
        userData.insert(userData.end(), slot.begin(), slot.end());
    }
    const auto output = Descriptor(ManyOutput.data(), sizeof(ManyOutput), Format32UInt);
    userData.insert(userData.end(), output.begin(), output.end());
    const std::span<const std::uint32_t> code = ManyBuffersCode;
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {Lanes, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = true;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    for (std::uint32_t index = 0; index < ManyBuffers; ++index) {
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const auto expected = (presentMask >> index) & 1u ? Input[lane] : 0u;
            Require(ManyOutput[index * Lanes + lane] == expected, name + ": buffer " + std::to_string(index) + " lane " + std::to_string(lane) + " loaded " + Hex(ManyOutput[index * Lanes + lane]) + ", expected " + Hex(expected));
        }
    }
    Outcome outcome;
    outcome.pipeline = result.PipelineVariantId();
    outcome.spirv = result.spirv.Copy();
    Require(outcome.pipeline != 0u && !outcome.spirv.empty(), name + ": no specialized module");
    return outcome;
}

void ManyFormattedBuffers(AgcDriver::VulkanDevice& device) {
    const auto built = DispatchMany(device, 0x0fu, "four of eight V# present");
    const auto seven = DispatchMany(device, 0x7fu, "seven V# present");
    const auto all = DispatchMany(device, 0xffu, "eight V# present");
    const auto partial = DispatchMany(device, 0x15u, "three V# present after all were seen");
    if (RuntimeFormatEnabled()) {
        Require(built.pipeline == seven.pipeline && built.spirv == seven.spirv, "seven present V# built a module beside the four-present module");
        Require(built.pipeline == all.pipeline && built.spirv == all.spirv, "eight present V# built a module beside the four-present module");
        Require(built.pipeline == partial.pipeline && built.spirv == partial.spirv, "three present V# after all were seen built a module beside the four-present module");
    } else {
        Require(built.pipeline != seven.pipeline && seven.pipeline != all.pipeline && built.pipeline != all.pipeline, "APS5_NO_ABSENT_RUNTIME_FORMAT did not build one module per format word");
        Require(partial.pipeline == all.pipeline && partial.spirv == all.spirv, "absent V# did not take the substitutes learned from the eight present ones");
    }
}
}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        BoundThenNull(*device);
        NullThenBound(*device);
        AbsentThenFormats(*device);
        TwoFormattedBuffers(*device);
        AbsentLoadsZero(*device);
        MisalignedPresent(*device);
        ManyFormattedBuffers(*device);
        std::puts("buffer presence runtime tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
