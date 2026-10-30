// The device's f32 denormal flush against the emulated flush, bit for bit: the startup probe (FloatFlushProbe)
// lets a device that flushes f32 add, subtract, multiply and fused fma in hardware drop the software flush
// (SpirvTarget::float32FlushToZero, part of the cache key). The probe checks 96 rows per program and accepts any two
// NaNs as equal. This test checks the same three guest programs over every triple of 48 edge values, random words and
// random products near the f32 tininess boundary (and a fourth program with the cloth family's other f32 forms), in the probe's three float modes plus IEEE mode, and requires
// identical output words (NaN payloads included) from the module with the software flush and the module without it.
// A device whose probe keeps the emulated flush runs the comparison and reports it without failing.
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 1024;
constexpr std::uint32_t Inputs = 4;
constexpr std::uint32_t Stride = 19;
alignas(256) std::array<std::uint32_t, Threads * Inputs> Input{};
alignas(256) std::array<std::uint32_t, Threads * Stride> Output{};

// The probe's programs (FloatFlushProbe.cpp): f32 add, sub, mul, mul_legacy, fma, fmac, min, max, med3, frexp,
// floor-class ops, compares and cndmask (Mixed, 19 results at 76 B per lane); products and fma at the tininess
// boundary (Tiny, 12 results at 64 B per lane); v_fma_mix with f16 operands (MixedTiny, 8 results at 64 B per lane).
alignas(256) constexpr std::array<std::uint32_t, 46> MixedCode{
    0x34020084u, 0x160600ffu, 0x0000004cu, 0xe0381000u, 0x80000401u, 0xbf8c3f70u, 0x06140b04u, 0x08160905u,
    0x10180b04u, 0x0e1a0b04u, 0xd54b000eu, 0x041a0b04u, 0x7e1e0306u, 0x561e0b04u, 0x1e200b04u, 0xd5540011u,
    0x041a0b04u, 0xd5570012u, 0x041a0b04u, 0x7e264904u, 0x7e284505u, 0x7e2a8104u, 0x7c020b04u, 0xd5010016u,
    0x01a90280u, 0x7e2e1b04u, 0x7e307f05u, 0xd5080119u, 0x42020b04u, 0xd55f001au, 0x041a0b04u, 0xd550001bu,
    0x041a0b04u, 0xd550001cu, 0x03ca0b04u, 0xe0781000u, 0x80010a03u, 0xe0781010u, 0x80010e03u, 0xe0781020u,
    0x80011203u, 0xe0781030u, 0x80011603u, 0xe07c1040u, 0x80011a03u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 30> TinyCode{
    0x34020084u, 0x34040086u, 0xe0381000u, 0x80000401u, 0xbf8c3f70u, 0x10140b04u, 0x0e160b04u, 0xd54b000cu,
    0x041a0b04u, 0x7e1a0306u, 0x561a0b04u, 0x581c0d04u, 0x3f7ffffeu, 0x5a1e0b04u, 0x80000000u, 0xcc200010u,
    0x041a0b04u, 0xd5080011u, 0x22020b04u, 0xd5400012u, 0x041a0b04u, 0x7e260306u, 0x0c260b04u, 0xe0781000u,
    0x80010a02u, 0xe0781010u, 0x80010e02u, 0xe0741020u, 0x80011202u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 28> MixedTinyCode{
    0x34020084u, 0x34040086u, 0xe0381000u, 0x80000401u, 0xbf8c3f70u, 0xcc20000au, 0x141a0b04u, 0xcc20000bu,
    0x0c1a0905u, 0xcc20100cu, 0x141a0b04u, 0xcc20080du, 0x0c1a0905u, 0xcc20400eu, 0x141e0b04u, 0xcc20400fu,
    0x0c1e0905u, 0x7e1002ffu, 0x3f7ffffeu, 0xcc204010u, 0x041e1104u, 0xcc206011u, 0x041e1104u, 0xe0781000u,
    0x80010a02u, 0xe0781010u, 0x80010e02u, 0xbf810000u,
};

// The cloth family's other f32 forms (0x80001c0d00 and kin): v_mac_f32 (VOP2 and VOP3), v_mad_f32, v_madak/madmk
// (separately rounded, no software flush), rcp, rsq, sqrt, fract, exp, log, min, max, mul_legacy, cvt_f32_i32, and add,
// sub, mul again (19 results at 76 B per lane). Under the hardware flush the module declares no f32 denormal mode, so
// every one of these runs under the device's default: this program shows whether that changes any of them.
alignas(256) constexpr std::array<std::uint32_t, 46> ClothCode{
    0x34020084u, 0x160600ffu, 0x0000004cu, 0xe0381000u, 0x80000401u, 0xbf8c3f70u, 0x7e140306u, 0x3e140b04u,
    0xd541000bu, 0x041a0b04u, 0xd541000cu, 0x04160904u, 0x421a0b04u, 0x3f7ffffeu, 0x401c0b04u, 0x00800001u,
    0x7e1e5504u, 0x7e205d05u, 0x7e226704u, 0x10240d04u, 0x06260d05u, 0x08280d04u, 0x7e2a4104u, 0x1e2c0b04u,
    0x202e0d04u, 0x0e300d05u, 0x7e320304u, 0xd51f0019u, 0x00020d05u, 0x7e344b04u, 0x7e364f05u, 0x7e380b07u,
    0xe0781000u, 0x80010a03u, 0xe0781010u, 0x80010e03u, 0xe0781020u, 0x80011203u, 0xe0781030u, 0x80011603u,
    0xe07c1040u, 0x80011a03u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u,
};

struct Program {
    const char* name;
    std::span<const std::uint32_t> code;
    std::uint32_t laneDwords;
    std::uint32_t results;
};
constexpr std::array<Program, 4> Programs{{{"mixed", MixedCode, 19u, 19u}, {"tiny", TinyCode, 16u, 12u}, {"mixed-tiny", MixedTinyCode, 16u, 8u}, {"cloth", ClothCode, 19u, 19u}}};

constexpr std::array<std::uint32_t, 48> Edges{
    0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x00000003u, 0x80000002u, 0x00400000u, 0x007fffffu,
    0x807fffffu, 0x00800000u, 0x80800000u, 0x00800001u, 0x80800001u, 0x00ffffffu, 0x01000000u, 0x0f800000u,
    0x10000000u, 0x1e800000u, 0x1f000000u, 0x1f7fffffu, 0x1f800000u, 0x1fffffffu, 0x20000000u, 0xa0000000u,
    0x2f800000u, 0x33800000u, 0x34000000u, 0x3dcccccdu, 0x3effffffu, 0x3f000000u, 0x3f7ffffeu, 0x3f7fffffu,
    0x3f800000u, 0xbf800000u, 0x3f800001u, 0x40000000u, 0x41200000u, 0xc0490fdbu, 0x5f7fffffu, 0x5f800000u,
    0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xff9abcdeu,
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

std::string ModeName(const std::optional<ShaderRecompiler::ShaderFloatMode>& mode) {
    if (!mode.has_value()) return "no FLOAT_MODE";
    return "FLOAT_MODE " + Hex(mode->floatMode) + (mode->ieeeMode ? " IEEE" : "");
}

ShaderRecompiler::RecompileResult Compile(const ShaderRecompiler::SpirvTarget& base, bool hardwareFlush, const std::optional<ShaderRecompiler::ShaderFloatMode>& floatMode, const Program& program) {
    std::vector<std::uint32_t> userData(8, 0u);
    const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(Input.size() * 4u));
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(program.code.data()), std::as_bytes(program.code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    auto target = base;
    target.float32FlushToZero = hardwareFlush;
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(program.code.data()), program.code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    request.context.floatMode = floatMode;
    return ShaderRecompiler::Recompile(request);
}

std::vector<std::uint32_t> Run(AgcDriver::VulkanDevice& device, const ShaderRecompiler::RecompileResult& result, const Program& program) {
    Output.fill(0xdeadbeefu);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(program.code.data()));
    device.WaitIdle();
    return {Output.begin(), Output.end()};
}

// Rows: every (a, b, c) of the edge values, then random words, then random products whose exponent sum lies within
// three of the tininess boundary (with a random small addend for the fma forms).
std::vector<std::array<std::uint32_t, 4>> Rows() {
    std::vector<std::array<std::uint32_t, 4>> rows;
    for (const auto a : Edges) {
        for (const auto b : Edges) {
            for (const auto c : Edges) rows.push_back({a, b, c, a ^ b});
        }
    }
    std::mt19937 random(0x5eedc10du);
    for (std::uint32_t index = 0; index < 64u * Threads; ++index) rows.push_back({random(), random(), random(), random()});
    for (std::uint32_t index = 0; index < 64u * Threads; ++index) {
        const std::uint32_t first = random() % 255u;
        const std::int32_t second = std::clamp<std::int32_t>(128 - static_cast<std::int32_t>(first) + static_cast<std::int32_t>(random() % 7u) - 3, 0, 254);
        const auto a = (random() & 0x807fffffu) | (first << 23u);
        const auto b = (random() & 0x807fffffu) | (static_cast<std::uint32_t>(second) << 23u);
        const auto c = (random() & 3u) == 0u ? (random() & 0x80000000u) : (random() & 0x80ffffffu);
        rows.push_back({a, b, c, random()});
    }
    return rows;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto target = device->Target();
        const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 4> modes{std::nullopt, ShaderRecompiler::ShaderFloatMode{0xc0u, true, false, false}, ShaderRecompiler::ShaderFloatMode{0xc0u, true, true, false}, ShaderRecompiler::ShaderFloatMode{0x00u, false, true, false}};
        const auto rows = Rows();
        std::uint64_t compared = 0u;
        std::uint64_t differing = 0u;
        std::string first;
        std::map<std::string, std::uint64_t> bySlot;
        for (const auto& mode : modes) {
            for (const auto& program : Programs) {
                const auto software = Compile(target, false, mode, program);
                const auto hardware = Compile(target, true, mode, program);
                for (std::size_t begin = 0; begin < rows.size(); begin += Threads) {
                    Input.fill(0u);
                    for (std::uint32_t lane = 0; lane < Threads && begin + lane < rows.size(); ++lane) std::copy(rows[begin + lane].begin(), rows[begin + lane].end(), &Input[lane * Inputs]);
                    const auto expected = Run(*device, software, program);
                    const auto actual = Run(*device, hardware, program);
                    for (std::uint32_t lane = 0; lane < Threads && begin + lane < rows.size(); ++lane) {
                        for (std::uint32_t slot = 0; slot < program.results; ++slot) {
                            const auto index = lane * program.laneDwords + slot;
                            ++compared;
                            if (expected[index] == actual[index]) continue;
                            ++bySlot[std::string(program.name) + " result " + std::to_string(slot) + " in " + ModeName(mode)];
                            if (++differing == 1u) {
                                const auto& row = rows[begin + lane];
                                first = std::string(program.name) + " result " + std::to_string(slot) + " in " + ModeName(mode) + " of " + Hex(row[0]) + ", " + Hex(row[1]) + ", " + Hex(row[2]) + ", " + Hex(row[3]) + ": emulated " + Hex(expected[index]) + ", hardware " + Hex(actual[index]);
                            }
                        }
                    }
                }
            }
        }
        std::printf("f32 hardware flush vs emulated flush: %llu of %llu results differ (startup probe chose the %s flush)%s%s\n", static_cast<unsigned long long>(differing), static_cast<unsigned long long>(compared), target.float32FlushToZero ? "hardware" : "emulated", differing != 0u ? "; first: " : "", first.c_str());
        for (const auto& [slot, count] : bySlot) std::printf("  differs: %s: %llu\n", slot.c_str(), static_cast<unsigned long long>(count));
        Require(!target.float32FlushToZero || differing == 0u, "the startup probe chose the hardware f32 flush, but it differs from the emulated flush: " + first);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
