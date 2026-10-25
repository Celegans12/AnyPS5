#include "prx/libSceAgcDriver/Execution/include/CommandLog.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/ShapeReplay.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawPlans.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <span>
#include <initializer_list>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
namespace Replay = AgcDriver::DriverDetail::ShapeReplay;

alignas(256) constexpr std::array<std::uint32_t, 16> TriangleVertexCode{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 8> ConstantPixelCode{0xf4000080u, 0xfa000000u, 0xbf8cc07fu, 0x7e000202u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u};
std::span<std::byte> ColorTarget() {
    static std::byte* const base = [] {
#ifdef _WIN32
        return static_cast<std::byte*>(VirtualAlloc(nullptr, 65536u, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        return static_cast<std::byte*>(std::aligned_alloc(65536u, 65536u));
#endif
    }();
    Require(base != nullptr, "the color target pages were not allocated");
    return {base, 65536u};
}
alignas(256) std::array<float, 64> Constants{};

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header VertexHeader{};
alignas(8) Header PixelHeader{};

void Register(Header& header, const std::uint32_t* code, std::size_t bytes, std::uint8_t type) {
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(bytes);
    header.shader.code = code;
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

void Append(std::vector<std::uint32_t>& commands, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload);
}

std::vector<std::uint32_t> Draw(const float* constant, bool tiled) {
    const auto color = reinterpret_cast<std::uintptr_t>(ColorTarget().data());
    const auto vertex = reinterpret_cast<std::uintptr_t>(TriangleVertexCode.data());
    const auto pixel = reinterpret_cast<std::uintptr_t>(ConstantPixelCode.data());
    const auto pointer = reinterpret_cast<std::uintptr_t>(constant);
    constexpr float scale = 32.0f;
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> context{
        {0x2d5, 0x2000},
        {0x1b6, 0}, {0x207, 0}, {0x200, 0}, {0x203, 0x800},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, (63u << 14u) | 3u},
        {0x3b8, 0x9000000u | (tiled ? 0x1bu << 14u : 0u)}, {0x1e0, 0},
        {0xc, 0}, {0xd, 0x40040},
        {0x81, 0x80000000}, {0x82, 0x40040},
        {0x90, 0x80000000}, {0x91, 0x40040},
        {0x94, 0x80000000}, {0x95, 0x40040},
        {0x318, static_cast<std::uint32_t>(color >> 8u)}, {0x390, static_cast<std::uint32_t>(color >> 40u)},
        {0x10f, std::bit_cast<std::uint32_t>(scale)}, {0x110, std::bit_cast<std::uint32_t>(scale)},
        {0x111, std::bit_cast<std::uint32_t>(-2.0f)}, {0x112, std::bit_cast<std::uint32_t>(2.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0},
        {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
        {0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0},
    };
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> shader{
        {0xc8, static_cast<std::uint32_t>(vertex >> 8u)}, {0xc9, static_cast<std::uint32_t>(vertex >> 40u)}, {0x8b, 0}, {0x8c, 0},
        {0x008, static_cast<std::uint32_t>(pixel >> 8u)}, {0x009, static_cast<std::uint32_t>(pixel >> 40u)}, {0x00b, 2u << 1u},
    };
    std::vector<std::uint32_t> commands;
    Append(commands, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) Append(commands, 0x76, {offset, value});
    Append(commands, 0x76, {0x00c, static_cast<std::uint32_t>(pointer)});
    Append(commands, 0x76, {0x00d, static_cast<std::uint32_t>(pointer >> 32u)});
    Append(commands, 0x2d, {3, 2});
    return commands;
}

unsigned Drawn(const float* constant, bool tiled) {
    const auto target = ColorTarget();
    std::fill(target.begin(), target.end(), std::byte{0});
    auto commands = Draw(constant, tiled);
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "the draw submission failed");
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(target.data()), target.size(), nullptr, "test read-back");
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(target.data()), target.size());
    unsigned brightest = 0;
    for (const auto value : target) brightest = std::max(brightest, static_cast<unsigned>(value));
    return brightest;
}

void SetEnvironment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

}

int main() {
    try {
        SetEnvironment("APS5_NO_RESULT_MEMO", "1");
        AgcDriver::DriverDetail::DrawAhead::SetCompileAhead(false);
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        Register(VertexHeader, TriangleVertexCode.data(), sizeof(TriangleVertexCode), 2);
        Register(PixelHeader, ConstantPixelCode.data(), sizeof(ConstantPixelCode), 1);
        const bool on = !AgcDriver::Graphics::RuntimeSwitches::Set(AgcDriver::Graphics::RuntimeSwitch::NoShapeReplay);
        const std::array<std::pair<std::size_t, float>, 8> draws{{{0, 0.25f}, {4, 1.0f}, {8, 0.5f}, {12, 0.75f}, {16, 0.125f}, {20, 1.0f}, {24, 0.25f}, {28, 0.5f}}};
        for (const auto& [slot, value] : draws) Constants[slot] = value;
        for (const auto& [slot, value] : draws) {
            const auto expected = static_cast<unsigned>(value * 255.0f + 0.5f);
            const auto drawn = Drawn(&Constants[slot], false);
            Require(drawn == expected || drawn + 1u == expected || drawn == expected + 1u, "a draw of constant " + std::to_string(value) + " through a moved pointer wrote " + std::to_string(drawn) + ", expected " + std::to_string(expected));
        }
        for (const auto& [slot, value] : draws) static_cast<void>(Drawn(&Constants[slot], true));
        const auto totals = Replay::Totals();
        std::cout << "shape replay " << (on ? "on" : "off") << ": " << totals.captures << " stage captures, " << totals.eligible << " eligible, " << totals.hits << " replayed, " << totals.learned << " learned, " << totals.unlearnable << " unlearnable, " << totals.probations << " probations (" << totals.probationMismatches << " mismatches), " << totals.verified << " verified (" << totals.verifyMismatches << " mismatches)\n";
        Require(totals.probationMismatches == 0 && totals.verifyMismatches == 0, "a replayed stage differed from its full materialization");
        if (on) {
            Require(totals.learned >= 2 && totals.hits + totals.verified >= 6, "the same shape with moved data was replayed " + std::to_string(totals.hits) + " times and verified " + std::to_string(totals.verified) + " times, expected at least 6 together (two stages, eight draws, one learn and one probation each)");
            if (std::getenv("APS5_VERIFY_SHAPE_REPLAY") != nullptr) Require(totals.verified != 0, "the verify mode compared no replayed stage");
        } else {
            Require(totals.hits == 0 && totals.eligible == 0, "a draw was replayed with APS5_NO_SHAPE_REPLAY set");
        }
        namespace Plans = AgcDriver::Graphics::DrawPlans;
        const bool plans = Plans::Active();
        const auto planned = Plans::Counters();
        const auto outcome = [&](Plans::Outcome which) { return planned.outcomes[static_cast<std::size_t>(which)]; };
        std::cout << "draw plans " << (plans ? "on" : "off") << ": " << outcome(Plans::Outcome::Hit) << " pipeline hits, " << outcome(Plans::Outcome::Stored) << " stored, " << outcome(Plans::Outcome::NoPlan) << " without a plan, " << outcome(Plans::Outcome::Resources) << " other template, " << outcome(Plans::Outcome::Layout) << " other layout, " << planned.verified << " verified (" << outcome(Plans::Outcome::VerifyMismatch) << " mismatches), " << totals.keysCarried << " key words carried (" << totals.keyMismatches << " mismatches)\n";
        Require(outcome(Plans::Outcome::VerifyMismatch) == 0 && totals.keyMismatches == 0, "a draw plan differed from today's derivation");
        if (plans) {
            Require(outcome(Plans::Outcome::Hit) >= 6, "eight recorded draws of one shape took " + std::to_string(outcome(Plans::Outcome::Hit)) + " pipelines from their plan, expected at least 6");
            if (on && std::getenv("APS5_VERIFY_SHAPE_REPLAY") == nullptr) Require(totals.keysCarried >= 1, "no replayed stage carried its resource key words");
            if (std::getenv("APS5_VERIFY_DRAW_PLANS") != nullptr) Require(planned.verified != 0, "the draw plan verify mode compared nothing");
        } else {
            Require(outcome(Plans::Outcome::Hit) == 0 && totals.keysCarried == 0, "a draw used a plan with APS5_NO_DRAW_PLANS set");
        }
        if (AgcDriver::CommandLog::Enabled()) {
            for (const auto& stream : AgcDriver::CommandLog::Streams()) std::cout << "command-log: pool " << stream.pool << " commands " << stream.commands << " hash " << std::hex << stream.hash << std::dec << "\n";
        }
        AgcDriverShutdown_nid_postfix();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
