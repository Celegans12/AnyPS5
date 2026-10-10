#include "VulkanTestDevice.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "ImageTableAbi.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

using AgcDriver::Graphics::Require;
namespace Abi = ShaderRecompiler::ImageTableAbi;
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t Stride = 48;
constexpr std::uint32_t Records = 3;
constexpr std::uint32_t Sentinel = 0xdeadbeefu;
constexpr std::array<std::uint32_t, 4> PointClamp{0x92u, (4u * 256u) << 12u, 0u, 0u};
constexpr std::array<std::uint32_t, 8> NotImage{30u, 0u, 2u, 0u, 0u, 0u, 687u, 262144u};
alignas(256) constexpr std::array<std::uint32_t, 30> Code{
    0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080700u, 0xfa000020u, 0x8f108202u, 0xf4200444u,
    0x20000000u, 0x9312b011u, 0xf4280802u, 0x24000000u, 0xf42c0902u, 0x24000010u,
    0x360200bfu, 0x7e020d01u, 0x060202f0u, 0x100202ffu,
    0x3c800000u, 0x7e0402ffu, 0x3e000000u, 0xf09c0f08u, 0x01090401u, 0x7e120202u, 0x34101286u, 0x4a101100u,
    0x34101084u, 0xdc788000u, 0x001c0408u, 0xbf810000u,
};

std::uint64_t Address(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer);
}

struct GuestBlock {
    GuestBlock() {
#ifdef _WIN32
        data = static_cast<std::byte*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        data = static_cast<std::byte*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(data != nullptr, "capture bind publication: guest allocation failed");
        std::memset(data, 0, BlockBytes);
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

    std::uint32_t* Words(std::size_t offset) { return reinterpret_cast<std::uint32_t*>(data + offset); }
    std::byte* data = nullptr;
};

std::array<std::uint32_t, 4> Buffer(const void* data, std::uint32_t bytes) {
    const auto address = Address(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) & 0xffffu, bytes, 0x31016facu};
}

struct Fixture {
    GuestBlock guest;
    std::uint32_t* srt = guest.Words(0);
    std::uint32_t* table = guest.Words(0x1000);
    std::uint32_t* keys = guest.Words(0x2000);
    std::uint32_t* output = guest.Words(0x3000);
    std::uint32_t* texels = guest.Words(0x4000);
    std::array<std::uint32_t, 2> userData{};
    ShaderRecompiler::RecompileRequest request{};

    Fixture() {
        const auto tableV = Buffer(table, Records * Stride);
        const auto keysV = Buffer(keys, Records * 4);
        for (std::uint32_t record = 0; record < Records; ++record) keys[record] = record;
        const auto outputV = Buffer(output, 64 * 4 * 4);
        std::copy(tableV.begin(), tableV.end(), srt);
        std::copy(keysV.begin(), keysV.end(), srt + 4);
        std::copy(outputV.begin(), outputV.end(), srt + 8);
        const auto textureAddress = Address(texels);
        const std::array<std::uint32_t, 4> image{static_cast<std::uint32_t>(textureAddress >> 8u),
            static_cast<std::uint32_t>(textureAddress >> 40u) | (22u << 20u) | (3u << 30u),
            15u | (3u << 14u) | (1u << 31u), 0x924u | (9u << 28u)};
        for (std::uint32_t record = 0; record < Records; ++record) std::copy(PointClamp.begin(), PointClamp.end(), table + record * Stride / 4);
        std::copy(image.begin(), image.end(), table + 4);
        std::copy(NotImage.begin(), NotImage.end(), table + 2 * Stride / 4 + 4);
        std::fill_n(texels, 64 * 4, std::bit_cast<std::uint32_t>(3.0f));
        std::fill_n(output, 64 * 4, Sentinel);
        const auto root = Address(srt);
        userData = {static_cast<std::uint32_t>(root), static_cast<std::uint32_t>(root >> 32u)};
        request.shader = {ShaderRecompiler::ShaderStage::Compute, Address(Code.data()), Code, 0, {}};
        request.context.waveSize = 64;
        request.context.userData = userData;
        request.context.compute = ShaderRecompiler::ShaderComputeStageInfo{{64, 1, 1}, 0, {true, false, false}, false, 1};
        request.layout = {0, 0, 0, 128};
    }

    std::uint32_t* Changed(std::uint32_t record) { return table + record * Stride / 4 + 8; }
};

struct Captured {
    std::unique_ptr<AgcDriver::ShaderMemory> memory;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
    std::vector<AgcDriver::Graphics::GuestMemorySnapshot> snapshots;
};

Captured Capture(Fixture& fixture) {
    const auto source = ShaderRecompiler::PrepareShader(fixture.request);
    const auto invocation = ShaderRecompiler::PreparedShaderInvocation::TryCreate(fixture.request, source);
    Require(invocation.has_value(), "capture bind publication: no prepared invocation");
    Captured result;
    result.memory = std::make_unique<AgcDriver::ShaderMemory>(std::span<const ShaderRecompiler::MemoryRegion>{});
    const auto capture = result.memory->Capture(*invocation);
    result.compiled = invocation->Materialize(*capture);
    for (const auto& region : result.memory->Regions()) result.snapshots.push_back({region.guestAddress, region.bytes});
    Require(std::any_of(result.compiled->bindings.begin(), result.compiled->bindings.end(), [](const auto& binding) {
        return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable;
    }), "capture bind publication: shader has no address binding");
    return result;
}

void CheckTable(const Captured& capture) {
    const auto& compiled = *capture.compiled;
    const auto binding = std::find_if(compiled.bindings.begin(), compiled.bindings.end(), [](const auto& value) {
        return value.role == ShaderRecompiler::DescriptorRole::ImageTableMap;
    });
    Require(binding != compiled.bindings.end(), "capture bind publication: no captured image table");
    const auto& map = binding->guestDescriptor;
    bool found = false;
    for (std::uint32_t table = 0; table < map.at(Abi::HeaderTableCount); ++table) {
        const auto header = Abi::TableHeader(table);
        Require(map.at(header + Abi::TableKeyCount) == Records, "capture bind publication: unexpected table bound");
        const auto first = map.at(header + Abi::TableMapStart);
        const auto code = map.at(first + 2);
        if (code == Abi::NullCode || (code & Abi::PoisonFlag) != 0u) continue;
        const auto poison = map.at(code + Abi::ImageRecordFlags);
        if ((poison & Abi::PoisonFlag) == 0 || compiled.imageTablePoison.at(poison & ~Abi::PoisonFlag).reason != static_cast<std::uint32_t>(Abi::PoisonReason::NotImage)) continue;
        Require(!found && map.at(first + 1) == Abi::NullCode && map.at(header + Abi::TableFault) == 0,
            "capture bind publication: image table lost its null row or acquired a table fault");
        found = true;
    }
    Require(found, "capture bind publication: non-null row was not captured as NotImage");
}

std::uint32_t CapturedWord(const Captured& capture, std::uint64_t address) {
    for (const auto& snapshot : capture.snapshots) {
        if (address < snapshot.address || address - snapshot.address > snapshot.bytes.size() || snapshot.bytes.size() - (address - snapshot.address) < 4) continue;
        std::uint32_t value;
        std::memcpy(&value, snapshot.bytes.data() + (address - snapshot.address), sizeof(value));
        return value;
    }
    throw std::runtime_error("capture bind publication: changed word was not captured");
}

void QueuePublication(AgcDriver::VulkanDevice& device, Fixture& fixture, std::uint32_t record) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    const std::array<std::uint32_t, 4> marker{7u, 7u, 7u, 7u};
    Require(device.FillBuffer(Address(fixture.guest.data + 0x5000), sizeof(marker), marker), "capture bind publication: host import required");
    const std::uint32_t value = 1;
    const auto destination = Address(fixture.Changed(record));
    Require(device.WriteLabelOnGpu(destination, std::as_bytes(std::span(&value, 1)), 1, 1, false) == 0,
        "capture bind publication: producer label was not GPU-recorded");
    Require(device.OpenWriteOverlaps(destination, sizeof(value)) && *fixture.Changed(record) == 0,
        "capture bind publication: producer was not pending with old host bytes");
}

void CheckOutput(const Fixture& fixture) {
    const auto expected = std::bit_cast<std::uint32_t>(3.0f);
    for (std::size_t index = 0; index < 64 * 4; ++index) Require(fixture.output[index] == expected,
        "capture bind publication: sampled output differs at dword " + std::to_string(index) + ": " + std::to_string(fixture.output[index]));
}

void BindAndDispatch(AgcDriver::VulkanDevice& device, const Captured& captured) {
    const auto prepared = device.PrepareDispatch(*captured.compiled, captured.snapshots);
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    device.Dispatch(*captured.compiled, 1, 1, 1, captured.snapshots, Address(Code.data()), prepared);
    device.WaitIdle();
}

void Run(AgcDriver::VulkanDevice& device, Fixture& fixture, bool beforeCapture, std::uint32_t record) {
    fixture.request.target = device.Target();
    if (beforeCapture) QueuePublication(device, fixture, record);
    auto captured = Capture(fixture);
    CheckTable(captured);
    Require(CapturedWord(captured, Address(fixture.Changed(record))) == (beforeCapture ? 1u : 0u),
        "capture bind publication: capture has the wrong producer value");
    if (beforeCapture) {
        Require(*fixture.Changed(record) == 1 && fixture.guest.Words(0x5000)[0] == 7,
            "capture bind publication: capture did not flush the prior producer");
    } else {
        Require(*fixture.Changed(record) == 0, "capture bind publication: unexpected initial word");
        QueuePublication(device, fixture, record);
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            device.WaitIdle();
        }
        Require(*fixture.Changed(record) == 1 && fixture.guest.Words(0x5000)[0] == 7,
            "capture bind publication: completed producer did not publish");
        Require(CapturedWord(captured, Address(fixture.Changed(record))) == 0, "capture bind publication: captured bytes alias live memory");
        std::string failure;
        try {
            BindAndDispatch(device, captured);
        } catch (const std::runtime_error& error) {
            failure = error.what();
        }
        Require(failure == "AGC graphics: guest snapshot differs from registered memory", "capture bind publication: expected strict bind rejection, got: " + failure);
        Require(std::all_of(fixture.output, fixture.output + 64 * 4, [](auto value) { return value == Sentinel; }),
            "capture bind publication: rejected consumer wrote output");
        captured = Capture(fixture);
        CheckTable(captured);
        Require(CapturedWord(captured, Address(fixture.Changed(record))) == 1, "capture bind publication: recapture retained stale bytes");
    }
    BindAndDispatch(device, captured);
    CheckOutput(fixture);
    std::cout << "capture bind publication: " << (beforeCapture ? "prior producer flushed" : "strict rejection then fresh recapture") << ", row " << record << '\n';
}

}

int main() {
    try {
        Require(std::getenv("APS5_NO_SNAPSHOT_CHECK") == nullptr, "capture bind publication: snapshot checks must stay enabled");
        for (const auto [before, record] : std::array<std::pair<bool, std::uint32_t>, 3>{{{true, 2}, {false, 1}, {false, 2}}}) {
            Fixture fixture;
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
            Run(*device, fixture, before, record);
        }
        std::puts("capture bind publication tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
