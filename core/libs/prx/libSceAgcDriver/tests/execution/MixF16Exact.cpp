// agc_driver_mix_f16_exact_tests <out.bin>: v_fma_mixlo_f16 over 1M input triples (random bit patterns, specials,
// f16 denormals and constructed f16 rounding midpoints), written to <out.bin>. Design-agent probe (local/design-gpu):
// compare its file byte for byte with one from a build that has only the f64 helper (2026-10-28: #10980's path vs
// 90e7077014 with APS5_NO_MIX_F32=1, identical over 4,194,304 results).
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace {

constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t Groups = 16384;
constexpr std::uint32_t Lanes = Threads * Groups;
constexpr std::uint32_t Raw = 3u << 28u;
constexpr std::uint32_t Selects = 0xfacu;
constexpr std::uint32_t Float32 = 22u;

void Mubuf(std::uint32_t op, bool offen, std::uint32_t immediate, std::uint32_t vaddr, std::uint32_t vdata, std::uint32_t resource, std::uint32_t soffset, std::vector<std::uint32_t>& code) {
    code.push_back(0xe0000000u | (op << 18u) | (offen ? 0x1000u : 0u) | immediate);
    code.push_back((soffset << 24u) | (resource / 4u << 16u) | (vdata << 8u) | vaddr);
}

// VOP3P: v_fma_mixlo_f16 (op 0x21) vdst, src0, src1, src2 with op_sel_hi bits (1 = f16 source).
void MixLo(std::uint32_t vdst, std::uint32_t src0, std::uint32_t src1, std::uint32_t src2, std::uint32_t opSelHi, std::vector<std::uint32_t>& code) {
    code.push_back(0xcc000000u | (0x21u << 16u) | (((opSelHi >> 2u) & 1u) << 14u) | vdst);
    code.push_back(((opSelHi & 3u) << 27u) | (src2 << 18u) | (src1 << 9u) | src0);
}

alignas(256) std::array<std::uint32_t, 256> Program{};

// Per lane: v1 = lane * 16 (byte offset), loads a,b,c (dwords 0..2) from buffer 0, four mixes with op_sel_hi
// 7 (all f16), 3 (a,b f16, c f32), 6 (a f32? no: bit0=src0) ... stores 4 results to buffer 1 at lane*16.
std::span<const std::uint32_t> Code() {
    std::vector<std::uint32_t> code;
    code.push_back(0x9309c008u);                                    // s_mul_i32 s9, s8, 64 (s8 = workgroup id x)
    code.push_back((0x25u << 25u) | (1u << 17u) | (0u << 9u) | 9u);  // v_add_nc_u32 v1, s9, v0
    code.push_back((0x1au << 25u) | (1u << 17u) | (1u << 9u) | 0x84u);  // v_lshlrev_b32 v1, 4, v1
    Mubuf(14u, true, 0u, 1u, 4u, 0u, 0x80u, code);                  // buffer_load_dwordx4 v[4:7], v1, s[0:3] offen
    code.push_back(0xbf8c0000u);
    MixLo(8u, 256u + 7u, 256u + 5u, 256u + 6u, 2u, code);           // f32 cf x f16 b + f32 (c word)  (game pattern 2)
    MixLo(9u, 256u + 7u, 256u + 5u, 256u + 6u, 6u, code);           // f32 cf x f16 b + f16 c        (game pattern 6)
    MixLo(10u, 256u + 4u, 256u + 7u, 256u + 6u, 1u, code);          // f16 a x f32 cf + f32 (c word) (game pattern 1)
    MixLo(11u, 256u + 4u, 256u + 5u, 256u + 6u, 7u, code);          // f16 a x f16 b + f16 c
    Mubuf(30u, true, 0u, 1u, 8u, 4u, 0x80u, code);                  // buffer_store_dwordx4 v[8:11], v1, s[4:7] offen
    code.push_back(0xbf8c0000u);
    code.push_back(0xbf810000u);
    std::copy(code.begin(), code.end(), Program.begin());
    return std::span<const std::uint32_t>(Program.data(), code.size());
}

std::uint16_t ToHalfBits(double value) {
    // Round-to-nearest-even conversion of a double that is exactly representable or a midpoint helper.
    const float f = static_cast<float>(value);
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    const std::uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x47800000u) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (magnitude < 0x33000000u) return static_cast<std::uint16_t>(sign);
    const int exponent = static_cast<int>(magnitude >> 23u) - 127;
    if (exponent < -14) {
        const std::uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
        const auto shift = static_cast<std::uint32_t>(-exponent - 14 + 13);
        return static_cast<std::uint16_t>(sign | (mantissa >> shift));
    }
    return static_cast<std::uint16_t>(sign | ((static_cast<std::uint32_t>(exponent + 15) << 10u) | ((magnitude >> 13u) & 0x3ffu)));
}

double HalfValue(std::uint16_t bits) {
    const int exponent = (bits >> 10) & 0x1f;
    const int mantissa = bits & 0x3ff;
    const double sign = (bits & 0x8000) ? -1.0 : 1.0;
    if (exponent == 0) return sign * std::ldexp(mantissa, -24);
    return sign * std::ldexp(1024 + mantissa, exponent - 25);
}

}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::fprintf(stderr, "usage: agc_driver_mix_f16_exact_tests <out.bin>\n");
            return VulkanTestSkipped;
        }
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const std::size_t bytes = std::size_t{Lanes} * 16u;
        auto* input = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, bytes * 2u + 0x10000u, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        auto* output = input + bytes / 4u;
        GuestAllocations::Mutation().Add(input, bytes * 2u, true, true, true);
        std::mt19937_64 random(12345);
        const std::array<std::uint16_t, 12> specials{0x0000u, 0x8000u, 0x7c00u, 0xfc00u, 0x7e00u, 0x7d01u, 0xfe55u, 0x0001u, 0x83ffu, 0x03ffu, 0x7bffu, 0x0400u};
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            auto* words = input + lane * 4u;
            const auto kind = lane % 8u;
            const auto r = random();
            std::uint16_t a = static_cast<std::uint16_t>(r);
            std::uint16_t b = static_cast<std::uint16_t>(r >> 16u);
            std::uint16_t c = static_cast<std::uint16_t>(r >> 32u);
            std::uint32_t cf = static_cast<std::uint32_t>(random());
            if (kind == 1u) a = specials[r % specials.size()];
            if (kind == 2u) c = specials[(r >> 8u) % specials.size()];
            if (kind >= 3u) {
                // Moderate magnitudes and constructed midpoints: c (f16 and f32) puts a*b + c at or next to an f16 midpoint.
                a = static_cast<std::uint16_t>((r & 0x83ffu) | (static_cast<std::uint32_t>(10u + (r >> 10u) % 10u) << 10u));
                b = static_cast<std::uint16_t>(((r >> 16u) & 0x83ffu) | (static_cast<std::uint32_t>(10u + (r >> 26u) % 10u) << 10u));
                const double product = HalfValue(a) * HalfValue(b);
                const std::uint16_t target = ToHalfBits(product * (1.0 + static_cast<double>((r >> 40u) & 0xff) / 64.0) + 1.0);
                const double midpoint = HalfValue(target) + std::ldexp(1.0, ((target >> 10) & 0x1f) - 26) * (kind == 7u ? -1.0 : 1.0);
                const double wanted = midpoint - product;
                const double nudge = kind == 4u ? 1.0 + std::ldexp(1.0, -23) : kind == 5u ? 1.0 - std::ldexp(1.0, -23) : 1.0;
                const float cfloat = static_cast<float>(wanted * nudge);
                std::memcpy(&cf, &cfloat, 4);
                c = ToHalfBits(wanted);
            }
            if (kind == 6u) cf = static_cast<std::uint32_t>(random()) & 0x807fffffu;  // f32 denormal c
            words[0] = a | (static_cast<std::uint32_t>(random()) << 16u);
            words[1] = b | (static_cast<std::uint32_t>(random()) << 16u);
            words[2] = c | (static_cast<std::uint32_t>(random()) << 16u);
            words[3] = cf;
        }
        std::memset(output, 0, bytes);
        const auto code = Code();
        const auto in = reinterpret_cast<std::uint64_t>(input);
        const auto out = reinterpret_cast<std::uint64_t>(output);
        const std::vector<std::uint32_t> userData{
            static_cast<std::uint32_t>(in), static_cast<std::uint32_t>((in >> 32u) & 0xffffu), static_cast<std::uint32_t>(bytes), Selects | (Float32 << 12u) | Raw,
            static_cast<std::uint32_t>(out), static_cast<std::uint32_t>((out >> 32u) & 0xffffu), static_cast<std::uint32_t>(bytes), Selects | (Float32 << 12u) | Raw,
        };
        const std::array<ShaderRecompiler::MemoryRegion, 1> regions{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
        const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {true, false, false}, false, 1};
        ShaderRecompiler::RecompileRequest request{
            {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
            {64u, 0, userData, compute, std::nullopt, std::nullopt, regions},
            device->ComputeTarget(64),
            {0, 0, 0, 128}
        };
        if (std::getenv("APS5_SPANNED_WAVE") != nullptr) request.target.spannedWaves = true;
        request.useCache = false;
        const auto shader = ShaderRecompiler::Recompile(request);
        device->Dispatch(shader, Groups, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
        device->WaitIdle();
        std::ofstream file(argv[1], std::ios::binary);
        file.write(reinterpret_cast<const char*>(output), static_cast<std::streamsize>(bytes));
        std::ofstream inputs(std::string(argv[1]) + ".in", std::ios::binary);
        inputs.write(reinterpret_cast<const char*>(input), static_cast<std::streamsize>(bytes));
        std::uint64_t hash = 1469598103934665603ull;
        std::uint32_t nonzero = 0;
        for (std::uint32_t word = 0; word < Lanes * 4u; ++word) {
            hash = (hash ^ output[word]) * 1099511628211ull;
            nonzero += output[word] != 0u;
        }
        std::printf("MIX words=%zu lanes=%u nonzero=%u hash=%016llx\n", shader.spirv.size(), Lanes, nonzero, static_cast<unsigned long long>(hash));
        std::fflush(stdout);
        std::fflush(nullptr);
        TerminateProcess(GetCurrentProcess(), 0);  // no DLL detach: ExitProcess after APS5_PROFILE_GPU hung in teardown
        std::_Exit(0);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
