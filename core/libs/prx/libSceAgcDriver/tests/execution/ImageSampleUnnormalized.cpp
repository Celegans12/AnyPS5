#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 128;
constexpr std::uint32_t Words = 16;
constexpr std::uint32_t ResultWord = 4;
constexpr std::uint32_t Results = 12;
constexpr std::int32_t Width = 13;
constexpr std::int32_t Height = 7;
constexpr std::uint32_t StorageLevels = 4;
constexpr std::uint32_t Format8888UNorm = 56;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Format32SInt = 21;
constexpr std::uint32_t Type1D = 8;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t Type3D = 10;
constexpr std::uint32_t OneZeroOneZero = 0x041u;
constexpr std::uint8_t OtherLevelTexel = 0x40u;
alignas(256) std::array<std::uint32_t, Threads * Words> Buffer{};
alignas(256) std::array<std::uint8_t, 16384> SingleLevel{};
alignas(256) std::array<std::uint8_t, 16384> MultiLevel{};

alignas(256) constexpr std::array<std::uint32_t, 40> Code{
    0x34020086, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xbf8c3f70, 0x7e0802ff, 0x402ccccd,
    0xf09c0f08, 0x00610802, 0xf0900f08, 0x00610c02, 0xf0800f08, 0x00611002, 0xbf8c3f70, 0xe0701010,
    0x80000801, 0xe0701014, 0x80000901, 0xe0701018, 0x80000a01, 0xe070101c, 0x80000b01, 0xe0701020,
    0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xe0701030,
    0x80001001, 0xe0701034, 0x80001101, 0xe0701038, 0x80001201, 0xe070103c, 0x80001301, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 45> OffsetCode{
    0x34020086, 0xe0301008, 0x80000201, 0xe0301000, 0x80000301, 0xe0301004, 0x80000401, 0xbf8c3f70,
    0x7e0a02ff, 0x402ccccd, 0x7e2802ff, 0x0000023d, 0xf0dc0f08, 0x00610802, 0xf0d00f08, 0x00610c02,
    0xf0dc0f0a, 0x00611014, 0x00000403, 0xbf8c3f70, 0xe0701010, 0x80000801, 0xe0701014, 0x80000901,
    0xe0701018, 0x80000a01, 0xe070101c, 0x80000b01, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01,
    0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xe0701030, 0x80001001, 0xe0701034, 0x80001101,
    0xe0701038, 0x80001201, 0xe070103c, 0x80001301, 0xbf810000,
};

alignas(256) constexpr auto GuardedCode = [] {
    auto code = Code;
    code[10] = code[12] = 0xf09c0f08u;
    return code;
}();

alignas(256) constexpr auto GuardedOffsetCode = [] {
    auto code = OffsetCode;
    code[14] = 0xf0dc0f08u;
    return code;
}();

alignas(256) constexpr auto GuardedInactiveCode = [] {
    std::array<std::uint32_t, GuardedCode.size() + 2> code{};
    std::copy_n(GuardedCode.begin(), 8, code.begin());
    code[8] = 0xbefe03ffu;
    code[9] = 0x55555555u;
    std::copy(GuardedCode.begin() + 8, GuardedCode.end(), code.begin() + 10);
    return code;
}();

enum class Offset { None, Lane, Constant };
constexpr std::int32_t ConstantOffsetU = -3;
constexpr std::int32_t ConstantOffsetV = 2;

struct Program {
    std::span<const std::uint32_t> code;
    std::array<const char*, 3> instructions;
    std::array<Offset, 3> offsets;
};

const Program Plain{Code, {"image_sample_lz", "image_sample_l 2.7", "image_sample"}, {Offset::None, Offset::None, Offset::None}};
const Program Offsets{OffsetCode, {"image_sample_lz_o", "image_sample_l_o 2.7", "image_sample_lz_o (-3, 2) NSA"}, {Offset::Lane, Offset::Lane, Offset::Constant}};
const Program Guarded{GuardedCode, {"image_sample_lz first", "image_sample_lz second", "image_sample_lz third"}, Plain.offsets};
const Program GuardedOffsets{GuardedOffsetCode, {"image_sample_lz_o first", "image_sample_lz_o second", "image_sample_lz_o (-3, 2) NSA"}, Offsets.offsets};
const Program GuardedInactive{GuardedInactiveCode, Guarded.instructions, Plain.offsets};

alignas(256) constexpr std::array<std::uint32_t, 34> ConstantCode{
    0x34020086, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xbf8c3f70, 0xf09c0608, 0x00610802,
    0xf11c0208, 0x00610a02, 0xf11c0408, 0x00610e02, 0xbf8c3f70, 0xe0701010, 0x80000801, 0xe0701014,
    0x80000901, 0xe0701018, 0x80000a01, 0xe070101c, 0x80000b01, 0xe0701020, 0x80000c01, 0xe0701024,
    0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xe0701030, 0x80001001, 0xe0701034,
    0x80001101, 0xbf810000,
};

struct SamplerCase {
    const char* name;
    std::array<std::uint32_t, 4> words;
    bool linear;
    bool border;
};

constexpr std::array<SamplerCase, 5> Samplers{{
    {"bilinear clamp-to-last-texel, point mips, MAX_LOD 0xfff", {0x00008092u, 0x00fff000u, 0x05500000u, 0u}, true, false},
    {"point clamp-to-last-texel, point mips", {0x00008092u, 0x00fff000u, 0x04000000u, 0u}, false, false},
    {"bilinear clamp-to-last-texel, linear mips, MIN_LOD 1, LOD bias 1.5", {0x00008092u, 0x00fff100u, 0x09500180u, 0u}, true, false},
    {"bilinear clamp-to-border, transparent black", {0x000080b6u, 0x00fff000u, 0x05500000u, 0u}, true, true},
    {"point clamp-to-border, no mips, transparent black", {0x000080b6u, 0u, 0u, 0u}, false, true},
}};

struct Coordinate {
    float u;
    float v;
    std::int32_t du = 0;
    std::int32_t dv = 0;
};

std::vector<Coordinate> Coordinates() {
    std::vector<float> us;
    for (std::int32_t k = 0; k < Width; ++k) {
        for (const float fraction : {0.5f, 0.0f, 0.25f, 0.75f}) us.push_back(static_cast<float>(k) + fraction);
    }
    for (const float edge : {-1.0f, -0.5f, 13.0f, 14.0f, 12.75f}) us.push_back(edge);
    constexpr std::array<float, 3> fractions{0.5f, 0.0f, 0.75f};
    std::vector<Coordinate> result;
    for (std::size_t index = 0; index < us.size(); ++index) {
        for (std::size_t fraction = 0; fraction < fractions.size(); ++fraction) {
            result.push_back({us[index], static_cast<float>((index * fractions.size() + fraction) % Height) + fractions[fraction]});
        }
    }
    for (std::int32_t x = 0; x < Width; ++x) {
        const float v = static_cast<float>(x % Height) + 0.5f;
        result.push_back({static_cast<float>(x) - 1.0f, v});
        result.push_back({static_cast<float>(x) + 2.0f, v});
    }
    return result;
}

std::vector<Coordinate> OffsetCoordinates() {
    constexpr std::array<std::int32_t, 7> us{-32, -5, -1, 0, 1, 3, 31};
    constexpr std::array<std::int32_t, 6> vs{-2, -1, 0, 2, -32, 31};
    auto result = Coordinates();
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index].du = us[index % us.size()];
        result[index].dv = vs[index % vs.size()];
    }
    return result;
}

std::array<double, 4> TexelOf(std::int32_t x, std::int32_t y) {
    return {240.0 * ((x + y) & 1), 16.0 * x + 16.0, 32.0 * y + 32.0, 240.0};
}

void FillTexture(std::span<std::uint8_t> texels, std::uint32_t levels) {
    const auto mips = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, Format8888UNorm, Width, Height, levels);
    Require(AgcDriver::Graphics::ComputeSurfaceSize(mips, 1) <= texels.size(), "image sample unnormalized: the mip chain does not fit the texel storage");
    std::fill(texels.begin(), texels.end(), std::uint8_t{0xeeu});
    for (std::uint32_t level = 0; level < levels; ++level) {
        const auto& mip = mips[level];
        for (std::uint32_t y = 0; y < mip.height; ++y) {
            for (std::uint32_t x = 0; x < mip.width; ++x) {
                auto* texel = &texels[mip.tiledOffset + static_cast<std::uint64_t>(y) * mip.pitchBytes + x * 4u];
                const auto value = TexelOf(static_cast<std::int32_t>(x), static_cast<std::int32_t>(y));
                for (std::uint32_t component = 0; component < 4u; ++component) {
                    texel[component] = level == 0u ? static_cast<std::uint8_t>(value[component]) : OtherLevelTexel;
                }
            }
        }
    }
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::array<std::uint32_t, 8> TextureDescriptor(const void* data, std::uint32_t lastLevel, std::uint32_t maxMip) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format8888UNorm << 20u) | ((static_cast<std::uint32_t>(Width - 1) & 3u) << 30u),
        (static_cast<std::uint32_t>(Width - 1) >> 2u) | (static_cast<std::uint32_t>(Height - 1) << 14u),
        0xfacu | (lastLevel << 16u) | (Type2D << 28u),
        0u,
        maxMip << 4u,
        0u,
        0u,
    };
}

using Samples = std::vector<std::array<std::uint32_t, Results>>;

Samples Run(AgcDriver::VulkanDevice& device, const Program& program, const std::array<std::uint32_t, 8>& texture, const std::array<std::uint32_t, 4>& sampler, const std::vector<Coordinate>& coordinates, bool guardedWrap = false, std::uint32_t waveSize = 32) {
    Samples samples;
    for (std::size_t first = 0; first < coordinates.size(); first += Threads) {
        Buffer.fill(0xdeadbeefu);
        for (std::uint32_t lane = 0; lane < Threads; ++lane) {
            const auto coordinate = first + lane < coordinates.size() ? coordinates[first + lane] : Coordinate{0.5f, 0.5f};
            Buffer[lane * Words] = std::bit_cast<std::uint32_t>(coordinate.u);
            Buffer[lane * Words + 1u] = std::bit_cast<std::uint32_t>(coordinate.v);
            Buffer[lane * Words + 2u] = 0xffffc0c0u | (static_cast<std::uint32_t>(coordinate.du) & 0x3fu) | ((static_cast<std::uint32_t>(coordinate.dv) & 0x3fu) << 8u);
        }
        std::vector<std::uint32_t> userData(16, 0u);
        const auto buffer = BufferDescriptor(Buffer.data(), static_cast<std::uint32_t>(Buffer.size() * 4u));
        std::copy(buffer.begin(), buffer.end(), userData.begin());
        std::copy(texture.begin(), texture.end(), userData.begin() + 4);
        std::copy(sampler.begin(), sampler.end(), userData.begin() + 12);
        const auto code = program.code;
        const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
        const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
        ShaderRecompiler::RecompileRequest request{
            {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
            {waveSize, 0, userData, compute, std::nullopt, std::nullopt, memory},
            device.Target(),
            {0, 0, 0, 128}
        };
        request.useCache = false;
        const auto result = ShaderRecompiler::Recompile(request);
        if (guardedWrap) Require(result.hasGuardedUnnormalizedWrap, "unnormalized wrap: emitted guard lost its invocation proof");
        device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
        device.WaitIdle();
        for (std::uint32_t lane = 0; lane < Threads && first + lane < coordinates.size(); ++lane) {
            std::array<std::uint32_t, Results> values{};
            std::copy_n(&Buffer[lane * Words + ResultWord], Results, values.begin());
            samples.push_back(values);
        }
    }
    return samples;
}

std::array<double, 4> Fetch(const SamplerCase& sampler, std::int32_t x, std::int32_t y) {
    if (sampler.border && (x < 0 || x >= Width || y < 0 || y >= Height)) return {0.0, 0.0, 0.0, 0.0};
    return TexelOf(std::clamp(x, 0, Width - 1), std::clamp(y, 0, Height - 1));
}

std::array<double, 4> Reference(const SamplerCase& sampler, const Coordinate& coordinate) {
    if (!sampler.linear) return Fetch(sampler, static_cast<std::int32_t>(std::floor(coordinate.u)), static_cast<std::int32_t>(std::floor(coordinate.v)));
    const double u = static_cast<double>(coordinate.u) - 0.5;
    const double v = static_cast<double>(coordinate.v) - 0.5;
    const auto i = static_cast<std::int32_t>(std::floor(u));
    const auto j = static_cast<std::int32_t>(std::floor(v));
    const double alpha = u - i;
    const double beta = v - j;
    std::array<double, 4> result{};
    for (std::uint32_t component = 0; component < 4u; ++component) {
        result[component] = (1.0 - alpha) * (1.0 - beta) * Fetch(sampler, i, j)[component] + alpha * (1.0 - beta) * Fetch(sampler, i + 1, j)[component]
            + (1.0 - alpha) * beta * Fetch(sampler, i, j + 1)[component] + alpha * beta * Fetch(sampler, i + 1, j + 1)[component];
    }
    return result;
}

void Check(const Program& program, const SamplerCase& sampler, const char* image, const std::vector<Coordinate>& coordinates, const Samples& samples) {
    Require(samples.size() == coordinates.size(), "image sample unnormalized: a dispatch lost samples");
    const double tolerance = sampler.linear ? 0.5 : 0.25;
    for (std::size_t index = 0; index < coordinates.size(); ++index) {
        const auto& coordinate = coordinates[index];
        for (std::uint32_t instruction = 0; instruction < program.instructions.size(); ++instruction) {
            const auto offset = program.offsets[instruction];
            const std::int32_t du = offset == Offset::Lane ? coordinate.du : offset == Offset::Constant ? ConstantOffsetU : 0;
            const std::int32_t dv = offset == Offset::Lane ? coordinate.dv : offset == Offset::Constant ? ConstantOffsetV : 0;
            const auto expected = Reference(sampler, {coordinate.u + static_cast<float>(du), coordinate.v + static_cast<float>(dv)});
            for (std::uint32_t component = 0; component < 4u; ++component) {
                const double value = static_cast<double>(std::bit_cast<float>(samples[index][instruction * 4u + component])) * 255.0;
                Require(std::fabs(value - expected[component]) <= tolerance, std::string(program.instructions[instruction]) + ", " + sampler.name + ", " + image + ": (" + std::to_string(coordinate.u) + ", " + std::to_string(coordinate.v) + ") offset (" + std::to_string(du) + ", " + std::to_string(dv) + ") component " + std::to_string(component) + " is " + std::to_string(value) + ", expected " + std::to_string(expected[component]));
            }
        }
    }
}

void ExpectFailure(AgcDriver::VulkanDevice& device, const std::array<std::uint32_t, 8>& texture, const std::array<std::uint32_t, 4>& sampler, std::string_view reason, const char* what) {
    const std::vector<Coordinate> one{{0.5f, 0.5f}};
    try {
        static_cast<void>(Run(device, Plain, texture, sampler, one));
    } catch (const std::exception& error) {
        Require(std::string_view(error.what()).find(reason) != std::string_view::npos, std::string(what) + ": unexpected error: " + error.what());
        return;
    }
    throw std::runtime_error(std::string(what) + " was accepted");
}

class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        file = std::tmpfile();
        Require(file != nullptr, "unnormalized wrap: cannot capture completion errors");
#ifdef _WIN32
        saved = _dup(_fileno(stderr));
        Require(saved >= 0 && _dup2(_fileno(file), _fileno(stderr)) == 0, "unnormalized wrap: cannot redirect stderr");
#else
        saved = dup(fileno(stderr));
        Require(saved >= 0 && dup2(fileno(file), fileno(stderr)) >= 0, "unnormalized wrap: cannot redirect stderr");
#endif
    }
    ~StderrCapture() { if (file != nullptr) static_cast<void>(Finish()); }
    std::string Finish() {
        std::fflush(stderr);
#ifdef _WIN32
        _dup2(saved, _fileno(stderr));
        _close(saved);
#else
        dup2(saved, fileno(stderr));
        close(saved);
#endif
        std::rewind(file);
        std::string text;
        std::array<char, 4096> buffer{};
        for (std::size_t read = 0; (read = std::fread(buffer.data(), 1, buffer.size(), file)) != 0;) text.append(buffer.data(), read);
        std::fclose(file);
        file = nullptr;
        std::fwrite(text.data(), 1, text.size(), stderr);
        return text;
    }
private:
    std::FILE* file = nullptr;
    int saved = -1;
};

constexpr std::uint32_t CapturedExtent = 1408;
constexpr std::uint32_t Format16UNorm = 7;
constexpr std::array<std::uint32_t, 4> CapturedSampler{0x00008000u, 0x00fff000u, 0x05000000u, 0u};
alignas(256) std::array<std::uint8_t, 4 * 1024 * 1024> CapturedTexels{};

std::array<std::uint32_t, 8> CapturedTexture() {
    const auto mips = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, Format16UNorm, CapturedExtent, CapturedExtent, 1);
    Require(AgcDriver::Graphics::ComputeSurfaceSize(mips, 1) <= CapturedTexels.size(), "unnormalized wrap: captured-size texture storage is too small");
    CapturedTexels.fill(0xeeu);
    for (std::uint32_t y = 0; y < CapturedExtent; ++y) {
        for (std::uint32_t x = 0; x < CapturedExtent; ++x) {
            const auto offset = mips[0].tiledOffset + y * mips[0].pitchBytes + x * 2u;
            const auto byte = static_cast<std::uint8_t>((x + y) % 2u != 0u ? 0xffu : 0u);
            CapturedTexels[offset] = CapturedTexels[offset + 1u] = byte;
        }
    }
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(CapturedTexels.data()));
    return {static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format16UNorm << 20u) | (((CapturedExtent - 1u) & 3u) << 30u),
        ((CapturedExtent - 1u) >> 2u) | ((CapturedExtent - 1u) << 14u), 0x24cu | (Type2D << 28u), 0u, 0u, 0u, 0u};
}

void CheckCaptured(const Program& program, const std::vector<Coordinate>& coordinates, const Samples& samples, bool masked = false) {
    Require(samples.size() == coordinates.size(), "unnormalized wrap: captured tile lost samples");
    for (std::size_t index = 0; index < coordinates.size(); ++index) {
        for (std::uint32_t instruction = 0; instruction < 3u; ++instruction) {
            const auto offset = program.offsets[instruction];
            const auto& coordinate = coordinates[index];
            const auto du = offset == Offset::Lane ? coordinate.du : offset == Offset::Constant ? ConstantOffsetU : 0;
            const auto dv = offset == Offset::Lane ? coordinate.dv : offset == Offset::Constant ? ConstantOffsetV : 0;
            const bool inactive = masked && index % 2u != 0u;
            std::uint32_t red = 0;
            if (!inactive) {
                const auto x = static_cast<std::int32_t>(coordinate.u + static_cast<float>(du));
                const auto y = static_cast<std::int32_t>(coordinate.v + static_cast<float>(dv));
                red = (x + y) % 2 != 0 ? 0x3f800000u : 0u;
            }
            for (std::uint32_t component = 0; component < 4u; ++component) {
                const auto expected = inactive ? 0xdeadbeefu : component == 0u ? red : 0x3f800000u;
                Require(samples[index][instruction * 4u + component] == expected,
                    std::string(program.instructions[instruction]) + ": captured R16 texel mismatch at sample " + std::to_string(index) + " component " + std::to_string(component));
            }
        }
    }
}

void CapturedSamplerCases(AgcDriver::VulkanDevice& device) {
    const auto texture = CapturedTexture();
    std::vector<Coordinate> tile;
    for (std::uint32_t y = 704; y < 720; ++y) {
        for (std::uint32_t x = 704; x < 720; ++x) tile.push_back({static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f});
    }
    auto clamp = CapturedSampler;
    clamp[0] |= 0x92u;
    CheckCaptured(Guarded, tile, Run(device, Guarded, texture, clamp, tile));
    const auto* option = std::getenv("APS5_UNNORM_WRAP_GUARD");
    ExpectFailure(device, texture, CapturedSampler,
        option != nullptr && std::string_view(option) == "1" ? "unnormalized wrap requires emitted per-sampler guard proof" : "clamp mode 0 on X",
        "a wrap sampler shared by LZ and non-LZ operations");
    if (option == nullptr || std::string_view(option) != "1") {
        try {
            static_cast<void>(Run(device, Guarded, texture, CapturedSampler, tile));
        } catch (const std::exception& error) {
            Require(std::string_view(error.what()).find("clamp mode 0 on X") != std::string_view::npos, std::string("unnormalized wrap OFF: unexpected refusal: ") + error.what());
            for (std::uint32_t lane = 0; lane < Threads; ++lane) {
                for (std::uint32_t result = 0; result < Results; ++result) Require(Buffer[lane * Words + ResultWord + result] == 0xdeadbeefu, "unnormalized wrap OFF: refused dispatch wrote output");
            }
            return;
        }
        throw std::runtime_error("unnormalized wrap OFF: captured sampler was accepted");
    }
    const auto execute = [&](const Program& program, const std::vector<Coordinate>& coordinates, bool fault, std::uint32_t waveSize = 32) {
        StderrCapture capture;
        const auto samples = Run(device, program, texture, CapturedSampler, coordinates, true, waveSize);
        const auto log = capture.Finish();
        const std::string prefix = "[gpu] deferred write-back failed: ";
        const auto first = log.find(prefix);
        if (fault) {
            Require(first != std::string::npos && log.find(prefix, first + prefix.size()) == std::string::npos
                && log.find(prefix + "unnormalized sampler: shader ") == first && log.find("outside [0.5, size - 0.5]", first) != std::string::npos,
                "unnormalized wrap: active invalid coordinate did not report exactly one typed completion error: " + log);
        } else {
            Require(first == std::string::npos, "unnormalized wrap: valid or inactive coordinates caused a completion error: " + log);
        }
        return samples;
    };
    CheckCaptured(Guarded, tile, execute(Guarded, tile, false));
    CheckCaptured(Guarded, tile, execute(Guarded, tile, false, 64));
    for (auto& coordinate : tile) { coordinate.du = -3; coordinate.dv = 2; }
    CheckCaptured(GuardedOffsets, tile, execute(GuardedOffsets, tile, false));
    std::vector<Coordinate> boundary(Threads, {704.5f, 704.5f});
    boundary[0] = {0.5f, 0.5f};
    boundary[1] = {1407.5f, 0.5f};
    boundary[2] = {0.5f, 1407.5f};
    boundary[3] = {1407.5f, 1407.5f};
    CheckCaptured(Guarded, boundary, execute(Guarded, boundary, false));
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<Coordinate, 12> invalid{{
        {0.0f, 704.5f}, {1408.0f, 704.5f}, {704.5f, 0.0f}, {704.5f, 1408.0f},
        {0.25f, 704.5f}, {1407.75f, 704.5f}, {704.5f, 0.25f}, {704.5f, 1407.75f},
        {nan, 704.5f}, {704.5f, nan}, {infinity, 704.5f}, {704.5f, -infinity},
    }};
    auto inactive = std::vector<Coordinate>(Threads, {704.5f, 705.5f});
    for (std::uint32_t lane = 1; lane < Threads; lane += 2) inactive[lane] = invalid[(lane / 2u) % invalid.size()];
    CheckCaptured(GuardedInactive, inactive, execute(GuardedInactive, inactive, false), true);
    for (const auto& coordinate : invalid) {
        auto active = std::vector<Coordinate>(Threads, {704.5f, 704.5f});
        active[7] = coordinate;
        static_cast<void>(execute(Guarded, active, true));
    }
    auto activeWave64 = std::vector<Coordinate>(Threads, {704.5f, 704.5f});
    activeWave64[63] = {nan, 704.5f};
    static_cast<void>(execute(Guarded, activeWave64, true, 64));
    for (const auto& coordinate : std::array<Coordinate, 3>{{{3.5f, 704.5f, -4, 0}, {1407.5f, 704.5f, 1, 0}, {2.5f, 704.5f, 0, 0}}}) {
        auto active = std::vector<Coordinate>(Threads, {704.5f, 704.5f});
        active[7] = coordinate;
        static_cast<void>(execute(GuardedOffsets, active, true));
    }
    CheckCaptured(Guarded, boundary, execute(Guarded, boundary, false));
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        FillTexture(SingleLevel, 1);
        FillTexture(MultiLevel, StorageLevels);
        const auto coordinates = Coordinates();
        const auto single = TextureDescriptor(SingleLevel.data(), 0u, 0u);
        const auto multi = TextureDescriptor(MultiLevel.data(), 0u, StorageLevels - 1u);
        const auto offsetCoordinates = OffsetCoordinates();
        for (const auto& sampler : Samplers) {
            const auto singleSamples = Run(*device, Plain, single, sampler.words, coordinates);
            Check(Plain, sampler, "1-level image", coordinates, singleSamples);
            const auto multiSamples = Run(*device, Plain, multi, sampler.words, coordinates);
            Check(Plain, sampler, "first level of a 4-level image", coordinates, multiSamples);
            Require(multiSamples == singleSamples, std::string(sampler.name) + ": the first level of a 4-level image does not sample like a 1-level image");
            Check(Offsets, sampler, "1-level image", offsetCoordinates, Run(*device, Offsets, single, sampler.words, offsetCoordinates));
        }
        ExpectFailure(*device, TextureDescriptor(MultiLevel.data(), 2u, 2u), Samplers[0].words, "single-level", "a 3-level view");
        ExpectFailure(*device, single, {0x00008092u, 0x00fff000u, 0x05100000u, 0u}, "different minification", "unequal minification and magnification filters");
        ExpectFailure(*device, single, {0x00008090u, 0x00fff000u, 0x05500000u, 0u}, "clamp mode 0", "wrap on X");
        const auto constantView = [&](std::uint32_t type, std::uint32_t format, std::uint32_t swizzle) {
            auto descriptor = TextureDescriptor(SingleLevel.data(), 0u, 0u);
            descriptor[1] = (descriptor[1] & ~(0x1ffu << 20u)) | (format << 20u);
            descriptor[3] = swizzle | (type << 28u);
            return descriptor;
        };
        for (const auto& sampler : Samplers) {
            for (const auto& sample : Run(*device, Plain, constantView(Type3D, Format8888UNorm, OneZeroOneZero), sampler.words, coordinates)) {
                for (std::uint32_t index = 0; index < Results; ++index) {
                    const std::uint32_t expected = index % 2u == 0u ? 0x3f800000u : 0u;
                    Require(sample[index] == expected, std::string(sampler.name) + ": a 3D view whose channels select (1, 0, 1, 0) returned " + Hex(sample[index]) + " for result " + std::to_string(index));
                }
            }
        }
        struct ConstantCase {
            const char* name;
            std::array<std::uint32_t, 8> texture;
            std::uint32_t one;
        };
        const std::array<ConstantCase, 4> constants{{
            {"a 3D UNORM view selecting (1, 0, 1, 0)", constantView(Type3D, Format8888UNorm, OneZeroOneZero), 0x3f800000u},
            {"a 32_UINT view selecting (1, 0, 1, 0)", constantView(Type2D, Format32UInt, OneZeroOneZero), 1u},
            {"a 32_SINT view selecting (1, 0, 1, 0)", constantView(Type2D, Format32SInt, OneZeroOneZero), 1u},
            {"a 1D view selecting (0, 0, 0, 0)", constantView(Type1D, Format8888UNorm, 0u), 0u},
        }};
        const std::array<const char*, 3> constantInstructions{"image_sample_lz dmask 0x6", "image_gather4_lz dmask 0x2", "image_gather4_lz dmask 0x4"};
        const Program constantProgram{ConstantCode, constantInstructions, Plain.offsets};
        for (const auto& constant : constants) {
            auto normalized = Samplers[0].words;
            normalized[0] &= ~0x8000u;
            for (const auto& sample : Run(*device, constantProgram, constant.texture, normalized, coordinates)) {
                const std::array<std::uint32_t, 10> expected{0u, constant.one, 0u, 0u, 0u, 0u, constant.one, constant.one, constant.one, constant.one};
                for (std::uint32_t index = 0; index < expected.size(); ++index) {
                    const auto instruction = constantInstructions[index < 2u ? 0u : index < 6u ? 1u : 2u];
                    Require(sample[index] == expected[index], std::string(instruction) + ", " + constant.name + ": result " + std::to_string(index) + " is " + Hex(sample[index]) + ", expected " + Hex(expected[index]));
                }
            }
        }
        CapturedSamplerCases(*device);
        std::puts("image sample unnormalized tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
