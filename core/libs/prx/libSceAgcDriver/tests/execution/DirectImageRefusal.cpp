#include "VulkanTestDevice.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "PipelineSpecialization.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

using AgcDriver::Graphics::Require;
using namespace ShaderRecompiler;

constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Fill = 0x5a5a5a5a;
constexpr std::uint32_t Sentinel = 0x13572468;
constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t TextureBytes = 4096;

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        data = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        data = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(data != nullptr, "direct image refusal: guest allocation failed");
        GuestAllocations::Mutation().Add(data, BlockBytes, true, true, true);
    }
    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(data);
#ifdef _WIN32
        VirtualFree(data, 0, MEM_RELEASE);
#else
        std::free(data);
#endif
    }
    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;
    std::uint8_t* Refused() const { return data; }
    std::uint8_t* Valid() const { return data + 0x4000; }
    std::uint32_t* Output() const { return reinterpret_cast<std::uint32_t*>(data + 0x8000); }
    std::uint32_t* Sampled() const { return reinterpret_cast<std::uint32_t*>(data + 0xc000); }
    void Reset() const { std::memset(data, 0x5a, BlockBytes); }
private:
    std::uint8_t* data = nullptr;
};

class LoopGuard {
public:
    explicit LoopGuard(const char* value) {
        if (const auto* old = std::getenv("APS5_LOOP_GUARD")) { present = true; previous = old; }
        Set(value);
    }
    ~LoopGuard() { Set(present ? previous.c_str() : nullptr); }
private:
    static void Set(const char* value) {
#ifdef _WIN32
        _putenv_s("APS5_LOOP_GUARD", value == nullptr ? "" : value);
#else
        if (value == nullptr) unsetenv("APS5_LOOP_GUARD"); else setenv("APS5_LOOP_GUARD", value, 1);
#endif
    }
    bool present = false;
    std::string previous;
};

class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        file = std::tmpfile();
        Require(file != nullptr, "direct image refusal: cannot capture stderr");
#ifdef _WIN32
        saved = _dup(_fileno(stderr));
        _dup2(_fileno(file), _fileno(stderr));
#else
        saved = dup(fileno(stderr));
        dup2(fileno(file), fileno(stderr));
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

struct Program {
    alignas(256) std::array<std::uint32_t, 128> words{};
    std::size_t size = 0;
    std::uint32_t storePc = 0;
    bool table = false;
    explicit Program(bool loop = false, bool withTable = false) : table(withTable) {
        std::vector<std::uint32_t> code{
            0xf4080100u, 0xfa000000u,
            0xf40c0200u, 0xfa000010u,
            0xf40c0400u, 0xfa000030u,
            0xf4000600u, 0xfa000050u,
            0xf4000640u, 0xfa000054u,
            0xbf8cc07fu,
            0x7e3c0300u, 0x7e3e0280u, 0x7e400280u,
            0x7e2802ffu, 0x3f800000u, 0x7e2a0280u, 0x7e2c0280u, 0x7e2e02ffu, 0x3f800000u,
            0xbf068018u, 0xbf850000u,
        };
        const auto skip = code.size() - 1;
        if (loop) code.push_back(0xbe9a0382u);
        const auto loopStart = code.size();
        code.push_back(0x7da80019u);
        storePc = static_cast<std::uint32_t>(code.size() * 4);
        code.insert(code.end(), {0xf0202f28u, 0x0002141eu, 0xbefe04c1u});
        if (loop) {
            code.insert(code.end(), {0x809a811au, 0xbf06801au, 0xbf840000u});
            code.back() |= static_cast<std::uint16_t>(static_cast<std::int32_t>(loopStart) - static_cast<std::int32_t>(code.size()));
        }
        code[skip] |= static_cast<std::uint32_t>(code.size() - skip - 1);
        code.push_back(0xbf8a0000u);
        if (table) {
            code.insert(code.end(), {
                0xf4080100u, 0xfa000060u, 0xf4080200u, 0xfa000070u,
                0x8f108202u, 0xf4200444u, 0x20000000u, 0x9312b011u, 0xf42c0802u, 0x24000000u,
                0x360200bfu, 0x7e020d01u, 0x060202f0u, 0x100202ffu, 0x3c800000u,
                0x7e0402ffu, 0x3e000000u, 0xf09c8f08u, 0x01090401u,
                0xf40c0400u, 0xfa000030u, 0xbf8cc07fu,
                0x4a3c00a0u, 0x7e3e0280u, 0x7e400280u, 0xf0202f28u, 0x0004041eu, 0xbf810000u,
            });
        } else {
            code.insert(code.end(), {
                0x4a3c00a0u, 0x7e280280u, 0x7e2a02ffu, 0x3f800000u,
                0xf0202f28u, 0x0004141eu,
                0x34020082u, 0x7e0402ffu, Sentinel, 0xe0701000u, 0x80010201u, 0xbf810000u,
            });
        }
        Require(code.size() <= words.size(), "direct image refusal: program too large");
        std::copy(code.begin(), code.end(), words.begin());
        size = code.size();
    }
    std::span<const std::uint32_t> Code() const { return {words.data(), size}; }
};

std::array<std::uint32_t, 8> Texture(const void* data, bool valid) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>(address >> 40u) | (56u << 20u) | (3u << 30u),
        (Width - 1u) >> 2u, ((valid ? 13u : 8u) << 28u) | 0xfacu, 0u, 0u, 0u, 0u};
}

struct Fixture {
    alignas(256) std::array<std::uint32_t, 40> root{};
    alignas(256) std::array<std::uint32_t, 12> table{};
    alignas(256) std::array<std::uint32_t, Lanes> keys{};
    std::array<std::uint32_t, 2> users{};
    std::array<MemoryRegion, 4> regions{};
    RecompileRequest request{};
    SrtRuntime runtime{};
    Fixture(AgcDriver::VulkanDevice& device, GuestBlock& guest, const Program& program) {
        const auto output = reinterpret_cast<std::uintptr_t>(guest.Output());
        root[0] = static_cast<std::uint32_t>(output);
        root[1] = static_cast<std::uint32_t>(output >> 32u);
        root[2] = Lanes * 4u;
        root[3] = 0x31016facu;
        const auto valid = Texture(guest.Valid(), true);
        std::copy(valid.begin(), valid.end(), root.begin() + 12);
        Descriptor(Texture(guest.Refused(), false));
        root[21] = Lanes;
        const auto buffer = [&](std::uint32_t slot, const void* data, std::uint32_t bytes) {
            const auto at = reinterpret_cast<std::uintptr_t>(data);
            root[slot] = static_cast<std::uint32_t>(at);
            root[slot + 1] = static_cast<std::uint32_t>(at >> 32u);
            root[slot + 2] = bytes;
            root[slot + 3] = 0x31016facu;
        };
        buffer(24, table.data(), sizeof(table));
        buffer(28, keys.data(), sizeof(keys));
        table[0] = 0x92u;
        table[1] = 0x00400000u;
        const auto sampled = reinterpret_cast<std::uintptr_t>(guest.Sampled());
        table[4] = static_cast<std::uint32_t>(sampled >> 8u);
        table[5] = static_cast<std::uint32_t>(sampled >> 40u) | (22u << 20u) | (3u << 30u);
        table[6] = (Width - 1u) >> 2u;
        table[7] = 0x90000facu;
        const auto address = reinterpret_cast<std::uintptr_t>(root.data());
        users = {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u)};
        regions = {{{reinterpret_cast<std::uintptr_t>(program.words.data()), std::as_bytes(program.Code())},
            {address, std::as_bytes(std::span(root))},
            {reinterpret_cast<std::uintptr_t>(table.data()), std::as_bytes(std::span(table))},
            {reinterpret_cast<std::uintptr_t>(keys.data()), std::as_bytes(std::span(keys))}}};
        request.shader = {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(program.words.data()), program.Code(), 0, {}};
        request.context = {32u, 0u, users, ShaderComputeStageInfo{{Lanes, 1, 1}, 0, {program.table, false, false}, false, 1}, {}, {}, regions};
        request.target = device.ComputeTarget(32);
        request.layout = {0, 0, 0, 128};
        runtime.userData = users;
        runtime.readMemory = Read;
        runtime.userContext = this;
    }
    void Descriptor(const std::array<std::uint32_t, 8>& words) { std::copy(words.begin(), words.end(), root.begin() + 4); }
    std::shared_ptr<const RecompileResult> Materialize(const SourceHandle& handle) {
        const auto capture = CaptureResources(request, runtime, handle);
        return MaterializeShader(request, *capture, handle);
    }
    static bool Read(void* context, std::uint64_t address, std::uint32_t* value) {
        const auto& fixture = *static_cast<Fixture*>(context);
        for (const auto& region : fixture.regions) {
            if (region.bytes.size() < 4 || address < region.guestAddress || address - region.guestAddress > region.bytes.size() - 4 || (address & 3u) != 0u) continue;
            std::memcpy(value, region.bytes.data() + (address - region.guestAddress), 4);
            return true;
        }
        return false;
    }
};

std::uint32_t ImageWrites(const RecompileResult& result) {
    std::uint32_t count = 0;
    for (std::size_t offset = 5; offset < result.spirv.size();) {
        const auto word = result.spirv[offset];
        Require((word >> 16u) != 0u && offset + (word >> 16u) <= result.spirv.size(), "direct image refusal: malformed SPIR-V");
        count += (word & 0xffffu) == spv::OpImageWrite;
        offset += word >> 16u;
    }
    return count;
}

std::uint32_t ImageDescriptors(const RecompileResult& result) {
    std::uint32_t count = 0;
    for (const auto& binding : result.bindings) if (binding.role == DescriptorRole::GuestImages) count += binding.count;
    return count;
}

std::string Hex(std::uint64_t value) {
    char text[32];
    std::snprintf(text, sizeof(text), "%llx", static_cast<unsigned long long>(value));
    return text;
}

void Execute(AgcDriver::VulkanDevice& device, GuestBlock& guest, const Program& program, const RecompileResult& result,
        bool fault, bool validDescriptor, std::uint32_t flag, std::uint32_t lanes) {
    guest.Reset();
    if (program.table) std::fill_n(guest.Sampled(), Width, 0x3f800000u);
    StderrCapture capture;
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(program.words.data()));
    device.WaitIdle();
    const auto log = capture.Finish();
    const std::string prefix = "[gpu] deferred write-back failed: ";
    if (fault) {
        const auto expected = prefix + "direct image: shader 0x" + Hex(reinterpret_cast<std::uintptr_t>(program.words.data()))
            + " stage 4 pc 0x" + Hex(program.storePc) + " resource 0: dimension";
        const auto at = log.find(expected);
        Require(at != std::string::npos && log.find(prefix) == at && log.find(prefix, at + expected.size()) == std::string::npos,
            std::string(program.table ? "direct image refusal: active mixed-table access did not report exactly one typed callback failure: "
                : "direct image refusal: active access did not report exactly one typed callback failure: ") + log);
    } else {
        Require(log.find(prefix) == std::string::npos, "direct image refusal: inactive or valid access failed: " + log);
        if (!program.table) for (std::uint32_t lane = 0; lane < Lanes; ++lane) Require(guest.Output()[lane] == Sentinel,
            "direct image refusal: independent work after barrier was lost");
    }
    device.WaitIdle();
    for (auto* texture : {guest.Refused(), guest.Valid()}) {
        AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(texture), TextureBytes, nullptr, "direct image refusal test");
    }
    device.WaitIdle();
    for (std::uint32_t x = 0; x < TextureBytes / 4; ++x) {
        std::uint32_t first = 0, second = 0;
        std::memcpy(&first, guest.Refused() + x * 4, 4);
        std::memcpy(&second, guest.Valid() + x * 4, 4);
        const auto firstExpected = validDescriptor && flag != 0 && x < lanes ? 0xff0000ffu : Fill;
        const auto secondExpected = x >= Lanes && x < Lanes * 2u ? (program.table ? 0xff0000ffu : 0xff00ff00u) : Fill;
        Require(first == firstExpected, "direct image refusal: refused or valid target changed unexpectedly at " + std::to_string(x));
        Require(second == secondExpected, "direct image refusal: refused access aliased the valid heap element, or useful store after barrier was lost at " + std::to_string(x));
    }
}

void RuntimeCases(AgcDriver::VulkanDevice& device, GuestBlock& guest, bool loop, bool table = false) {
    Program program(loop, table);
    Fixture fixture(device, guest, program);
    fixture.Descriptor(Texture(guest.Refused(), true));
    const auto handle = PrepareShader(fixture.request);
    const auto& artifact = GetPreparedArtifact(*handle);
    fixture.Descriptor(Texture(guest.Refused(), false));
    std::uint64_t refusedPipeline = 0;
    std::vector<PipelineSpecializationConstant> refusedConstants;
    for (const auto [flag, lanes] : std::array<std::array<std::uint32_t, 2>, 5>{{{0, Lanes}, {1, Lanes}, {0, Lanes}, {1, 0}, {1, 5}}}) {
        fixture.root[20] = flag;
        fixture.root[21] = lanes;
        const auto result = fixture.Materialize(*handle);
        Require(result->variantId == artifact.variantId, "direct image refusal: runtime branch replaced the prepared artifact");
        Require(result->hasDeferredDirectImage && result->specialization.empty(),
            "direct image refusal: refused invocation lost its completion metadata or retained finalized specialization");
        Require(ImageWrites(*result) == 1 && ImageDescriptors(*result) == 1, "direct image refusal: refused resource was not pruned independently from the shared heap");
        Require(std::any_of(result->bindings.begin(), result->bindings.end(), [](const auto& binding) { return binding.role == DescriptorRole::FaultBuffer; }), "direct image refusal: missing fault binding");
        if (table) {
            Require(std::getenv("APS5_ALL_BUFFERS_WRITTEN") == nullptr, "direct image refusal: forced buffer writes invalidate the table completion control");
            Require(result->imageTableFaults == 0 && result->poisonedSrtReads == 0,
                "direct image refusal: table control already requires completion for poisoned inputs");
            Require(std::any_of(result->bindings.begin(), result->bindings.end(), [](const auto& binding) { return binding.role == DescriptorRole::ImageTableMap; }),
                "direct image refusal: mixed fixture has no table map");
            Require(std::none_of(result->bindings.begin(), result->bindings.end(), [](const auto& binding) {
                return binding.role == DescriptorRole::BdaPagetable || std::any_of(binding.samplerUnnormalized.begin(), binding.samplerUnnormalized.end(), [](bool value) { return value; });
            }), "direct image refusal: BDA or unnormalized sampling independently forced table completion");
            Require(std::none_of(result->bindings.begin(), result->bindings.end(), [](const auto& binding) {
                return binding.role == DescriptorRole::GuestBuffers && (binding.bufferWritten.size() < binding.count
                    || std::any_of(binding.bufferWritten.begin(), binding.bufferWritten.end(), [](bool value) { return value; }));
            }), "direct image refusal: writable guest buffers independently forced table completion");
        }
        if (refusedPipeline == 0) { refusedPipeline = result->PipelineVariantId(); refusedConstants = result->specialization; }
        else Require(result->PipelineVariantId() == refusedPipeline && result->specialization == refusedConstants, "direct image refusal: runtime flag or EXEC mask entered specialization");
        Execute(device, guest, program, *result, flag != 0 && lanes != 0, false, flag, lanes);
    }
    std::uint64_t validPipeline = 0;
    for (const bool valid : {true, false, true}) {
        fixture.Descriptor(Texture(guest.Refused(), valid));
        fixture.root[20] = valid ? 1u : 0u;
        fixture.root[21] = Lanes;
        const auto result = fixture.Materialize(*handle);
        Require(result->variantId == artifact.variantId, "direct image refusal: descriptor transition replaced the prepared artifact");
        Require(result->hasDeferredDirectImage == !valid && result->specialization.empty(),
            "direct image refusal: descriptor transition retained stale completion metadata or finalized specialization");
        Require(ImageWrites(*result) == (valid ? 2u : 1u) && ImageDescriptors(*result) == (valid ? 2u : 1u), "direct image refusal: descriptor transition retained stale bindings or image operations");
        if (valid) {
            if (validPipeline == 0) validPipeline = result->PipelineVariantId();
            Require(result->PipelineVariantId() == validPipeline && validPipeline != refusedPipeline, "direct image refusal: valid and refused pipeline identities collided");
        } else Require(result->PipelineVariantId() == refusedPipeline, "direct image refusal: returning descriptor missed the prior specialization");
        Execute(device, guest, program, *result, false, valid, fixture.root[20], Lanes);
    }
}

void RelocatedCodeCase(AgcDriver::VulkanDevice& device, GuestBlock& guest) {
    Program original;
    Program relocated = original;
    Require(original.words.data() != relocated.words.data(), "direct image refusal: relocated shader address did not change");
    Fixture first(device, guest, original);
    Fixture second(device, guest, relocated);
    first.Descriptor(Texture(guest.Refused(), true));
    second.Descriptor(Texture(guest.Refused(), true));
    const auto originalHandle = PrepareShader(first.request);
    const auto relocatedHandle = PrepareShader(second.request);
    Require(GetPreparedArtifact(*originalHandle).variantId == GetPreparedArtifact(*relocatedHandle).variantId,
        "direct image refusal: identical relocated code did not share its compiled artifact");
    first.Descriptor(Texture(guest.Refused(), false));
    second.Descriptor(Texture(guest.Refused(), false));
    first.root[20] = second.root[20] = 1u;
    const auto originalResult = first.Materialize(*originalHandle);
    const auto relocatedResult = second.Materialize(*relocatedHandle);
    Require(originalResult->hasDeferredDirectImage && relocatedResult->hasDeferredDirectImage
            && originalResult->specialization.empty() && relocatedResult->specialization.empty(),
        "direct image refusal: relocated invocation lost its completion metadata or retained finalized specialization");
    Require(originalResult->PipelineVariantId() == relocatedResult->PipelineVariantId(),
        "direct image refusal: shader address entered the pipeline key");
    Execute(device, guest, original, *originalResult, true, false, 1u, Lanes);
    Execute(device, guest, relocated, *relocatedResult, true, false, 1u, Lanes);
}

template<typename F>
void Refuses(F&& action, const std::string& name, const char* expected = nullptr) {
    bool refused = false;
    try { action(); } catch (const std::exception& error) {
        refused = true;
        if (expected != nullptr) Require(std::string(error.what()) == expected,
            "direct image refusal: " + name + " returned an unexpected CPU refusal: " + error.what());
    }
    Require(refused, "direct image refusal: " + name + " escaped strict CPU refusal");
}

void StrictCases(AgcDriver::VulkanDevice& device, GuestBlock& guest) {
    Program program;
    Fixture fixture(device, guest, program);
    const auto handle = PrepareShader(fixture.request);
    fixture.root[20] = 0;
    const auto original = Texture(guest.Refused(), false);
    static_cast<void>(fixture.Materialize(*handle));
    struct Mutation { const char* name; std::uint32_t word, mask, value; };
    for (const auto& mutation : std::array<Mutation, 16>{{
        {"reserved selector 2", 3, 7, 2}, {"reserved selector 3", 3, 7, 3},
        {"nonzero 1D height", 2, 0, 1u << 14u}, {"nonzero 1D depth", 4, 0, 1},
        {"base array", 4, 0, 1u << 16u}, {"array pitch", 5, 0, 1},
        {"base mip", 3, 0, 1u << 12u}, {"last mip", 3, 0, 1u << 16u},
        {"unsupported format", 1, 0x1ffu << 20u, 22u << 20u},
        {"nonlinear swizzle", 3, 0, 1u << 20u}, {"BC swizzle", 3, 0, 1u << 25u},
        {"DCC", 6, 0, 1u << 21u},
        {"maximum mip", 5, 0, 1u << 4u}, {"corner sampling", 5, 0, 1u << 23u},
        {"partially resident color", 5, 0, 1u << 26u}, {"MSAA depth", 6, 0, 1u << 10u},
    }}) {
        auto changed = original;
        changed[mutation.word] = (changed[mutation.word] & ~mutation.mask) | mutation.value;
        fixture.Descriptor(changed);
        Refuses([&] { static_cast<void>(fixture.Materialize(*handle)); }, mutation.name);
    }
    fixture.Descriptor(original);
    const auto plan = GetResourcePlan(fixture.request);
    Require(plan->info.images.size() == 2u, "direct image refusal: fixture lost its two independent resources");
    const DescriptorValue descriptor{original, 8u};
    Require(ResourceMaterializer::DirectImageMode(plan->info, plan->stage, 0, descriptor, true) == PipelineSpecialization::DeferredDirectImageMode,
        "direct image refusal: positive classifier control did not produce the reserved selector");
    for (const auto count : {0u, 4u, 7u}) {
        auto truncated = descriptor;
        truncated.dwordCount = count;
        Refuses([&] { static_cast<void>(ResourceMaterializer::DirectImageMode(plan->info, plan->stage, 0, truncated, true)); }, "truncated image descriptor");
    }
    auto missingModes = plan->info;
    missingModes.runtimeImageModes.clear();
    Refuses([&] { static_cast<void>(ResourceMaterializer::DirectImageMode(missingModes, plan->stage, 0, descriptor, true)); },
        "missing mode list", "ShaderRecompiler: direct image runtime modes are missing");
    Refuses([&] { static_cast<void>(ResourceMaterializer::DirectImageMode(plan->info, plan->stage, static_cast<std::uint32_t>(plan->info.images.size()), descriptor, true)); },
        "missing image resource", "ShaderRecompiler: direct image resource is missing");
    Refuses([&] { static_cast<void>(ResourceMaterializer::DirectImageMode(plan->info, IrShaderStage::Pixel, 0, descriptor, true)); }, "noncompute stage");
    auto paired = plan->info;
    paired.sampledPairs.push_back({0, 0, 0});
    Refuses([&] { static_cast<void>(ResourceMaterializer::DirectImageMode(paired, plan->stage, 0, descriptor, true)); }, "coalesced sampled use");
    for (const auto opcode : {0xf0002f28u, 0xf03c2128u, 0xf0242f28u, 0xf0282f28u, 0xf09c0f28u}) {
        Program rejected;
        rejected.words[rejected.storePc / 4] = opcode;
        Fixture negative(device, guest, rejected);
        Refuses([&] { const auto prepared = PrepareShader(negative.request); static_cast<void>(negative.Materialize(*prepared)); }, "load, atomic, explicit mip, packed or sampled access");
    }
    for (const bool d16 : {false, true}) {
        Program rejected;
        rejected.words[rejected.storePc / 4 + (d16 ? 1 : 0)] |= d16 ? 0x80000000u : 0x8000u;
        Fixture negative(device, guest, rejected);
        Refuses([&] { const auto prepared = PrepareShader(negative.request); static_cast<void>(negative.Materialize(*prepared)); }, d16 ? "16-bit store" : "R128 store");
    }
    auto noAbiRequest = fixture.request;
    noAbiRequest.target.bdaAbiVersion = 0;
    Refuses([&] { const auto noAbi = PrepareShader(noAbiRequest); const auto capture = CaptureResources(noAbiRequest, fixture.runtime, *noAbi); static_cast<void>(MaterializeShader(noAbiRequest, *capture, *noAbi)); }, "missing fault ABI");
    Program guardedProgram(true);
    auto& guardedLoopCount = guardedProgram.words[guardedProgram.storePc / 4u - 2u];
    Require(guardedLoopCount == 0xbe9a0382u, "direct image refusal: guarded loop counter instruction changed");
    guardedLoopCount = 0xbe9a0383u;
    Fixture guarded(device, guest, guardedProgram);
    std::shared_ptr<const SourceHandle> guardedHandle;
    {
        LoopGuard guard("8");
        Refuses([&] { static_cast<void>(fixture.Materialize(*handle)); }, "warm artifact with loop guard enabled");
        guardedHandle = PrepareShader(guarded.request);
        Refuses([&] { static_cast<void>(guarded.Materialize(*guardedHandle)); }, "cold artifact with loop guard enabled");
    }
    Refuses([&] { static_cast<void>(guarded.Materialize(*guardedHandle)); }, "loop-instrumented artifact reused after disabling its guard");
    {
        LoopGuard disabled(nullptr);
        Program ordinaryLoopProgram(true);
        Fixture ordinaryLoop(device, guest, ordinaryLoopProgram);
        const auto ordinaryLoopHandle = PrepareShader(ordinaryLoop.request);
        Require(GetPreparedArtifact(*ordinaryLoopHandle).variantId != GetPreparedArtifact(*guardedHandle).variantId,
            "direct image refusal: guarded negative reused the ordinary loop artifact");
    }
    static_cast<void>(fixture.Materialize(*handle));
}

}

int main() {
    try {
        LoopGuard guard(nullptr);
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock guest;
        const auto phase = [](const char* name) { std::printf("direct image refusal phase: %s\n", name); std::fflush(stdout); };
        phase("strict cases");
        try {
            StrictCases(*device, guest);
        } catch (const std::exception& error) {
            std::fprintf(stdout, "direct image refusal strict exception before teardown: %s\n", error.what());
            std::fflush(stdout);
            throw;
        } catch (...) {
            std::puts("direct image refusal strict unknown exception before teardown");
            std::fflush(stdout);
            throw;
        }
        phase("runtime cases");
        RuntimeCases(*device, guest, false);
        phase("bounded loop cases");
        RuntimeCases(*device, guest, true);
        phase("mixed table cases");
        RuntimeCases(*device, guest, false, true);
        phase("relocated code case");
        RelocatedCodeCase(*device, guest);
        std::puts("direct image refusal tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stdout, "direct image refusal exception: %s\n", error.what());
        std::fflush(stdout);
        return 1;
    }
}
