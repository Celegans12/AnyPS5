// agc_driver_newpath_runtime_buffers_tests: S2-share class C1 on the new path (SpirvTarget::newPathRuntimeBuffers). One
// wave64 reads two buffer V#s from an SRT (s_load_dwordx4 through the user-SGPR pointer, which the SRT walker flattens),
// then copies one dword per lane with buffer_load_dword / buffer_store_dword. Today's module binds both V#s as guest
// buffers (buf-mem+buf-write+flat-srt); the resolved module with newPathRuntimeBuffers binds no guest buffer, reads the
// V# words at run time and reaches both buffers through the device page table, keeping the bounded write range. The
// source's num_records (200 bytes) cuts the copy at lane 50 on both paths. Mapped: both modules store the same bytes,
// and the command record holds no fault and the output page. Unmapped (the source V# points at 64 KiB, which no
// allocation covers): the resolved module stores zeros, records an Unmapped fault inside the source range and the
// output page; today's module is run too and its outcome printed (its bound buffer has no guest memory there).
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandRecord.hpp"
#include "BdaAbi.hpp"
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
#include <cstring>
#include <iostream>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t Threads = 64;
constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t SourceOffset = 0x1000;
constexpr std::size_t OutputOffset = 0x2000;
constexpr std::uint32_t SourceRecords = 200;
constexpr std::uint32_t OutputRecords = Threads * 4u;
constexpr std::uint64_t UnmappedSource = 0x10000;
constexpr std::uint32_t Untouched = 0xdeadbeefu;

alignas(256) constexpr std::array<std::uint32_t, 13> Program{
    0xf4080200u, 0xfa000000u,   // s_load_dwordx4 s[8:11], s[0:1], 0x0
    0xf4080300u, 0xfa000010u,   // s_load_dwordx4 s[12:15], s[0:1], 0x10
    0x34020082u,                // v_lshlrev_b32 v1, 2, v0
    0xbf8cc07fu,                // s_waitcnt lgkmcnt(0)
    0xe0301000u, 0x80020401u,   // buffer_load_dword v4, v1, s[8:11], 0 offen
    0xbf8c3f70u,                // s_waitcnt vmcnt(0)
    0xe0701000u, 0x80030401u,   // buffer_store_dword v4, v1, s[12:15], 0 offen
    0xbf810000u,                // s_endpgm
    0xbf810000u,
};

std::array<std::uint32_t, 4> Descriptor(std::uint64_t address, std::uint32_t records) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), records, 0x31016facu};
}

bool BindsGuestBuffers(const ShaderRecompiler::RecompileResult& result) {
    return std::ranges::any_of(result.bindings, [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers; });
}

bool BindsFlattenedSrt(const ShaderRecompiler::RecompileResult& result) {
    return std::ranges::any_of(result.bindings, [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt; });
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        auto* block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (block == nullptr) throw std::runtime_error("cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
        auto* srt = reinterpret_cast<std::uint32_t*>(block);
        auto* source = reinterpret_cast<std::uint32_t*>(block + SourceOffset);
        auto* output = reinterpret_cast<std::uint32_t*>(block + OutputOffset);
        for (std::uint32_t lane = 0; lane < Threads; ++lane) source[lane] = 0x5a000000u | (lane * 0x10101u);
        const auto address = [](const void* pointer) { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)); };
        const std::span<const std::uint32_t> code(Program.data(), Program.size());
        const auto table = address(srt);
        const std::vector<std::uint32_t> userData{static_cast<std::uint32_t>(table), static_cast<std::uint32_t>(table >> 32u)};
        const std::array<ShaderRecompiler::MemoryRegion, 2> regions{{
            {reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)},
            {reinterpret_cast<std::uintptr_t>(srt), std::as_bytes(std::span<const std::uint32_t>(srt, 8u))},
        }};
        const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {true, false, false}, false, 1};
        namespace Bda = ShaderRecompiler::BdaAbi;
        std::mutex recordsMutex;
        std::vector<std::vector<std::uint32_t>> records;
        AgcDriver::NewPath::SetCommandRecordObserver([&](std::span<const std::uint32_t, AgcDriver::NewPath::CommandRecordWords> words) {
            std::lock_guard lock(recordsMutex);
            records.emplace_back(words.begin(), words.end());
        });
        int failures = 0;
        for (const bool unmapped : {false, true}) {
            const std::uint64_t in = unmapped ? UnmappedSource : address(source);
            const auto sourceDescriptor = Descriptor(in, SourceRecords);
            const auto outputDescriptor = Descriptor(address(output), OutputRecords);
            std::memcpy(srt, sourceDescriptor.data(), sizeof(sourceDescriptor));
            std::memcpy(srt + 4, outputDescriptor.data(), sizeof(outputDescriptor));
            std::array<std::vector<std::uint32_t>, 2> stored;
            std::array<bool, 2> ran{};
            for (const bool resolved : {false, true}) {
                std::fill(output, output + Threads, Untouched);
                {
                    std::lock_guard lock(recordsMutex);
                    records.clear();
                }
                ShaderRecompiler::RecompileRequest request{
                    {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
                    {64u, 0, userData, compute, std::nullopt, std::nullopt, regions},
                    device->ComputeTarget(64),
                    {0, 0, 0, 128}
                };
                request.target.newPathResolved = resolved;
                request.target.newPathRuntimeBuffers = resolved;
                request.useCache = false;
                const auto shader = ShaderRecompiler::Recompile(request);
                const char* name = resolved ? "resolved + runtime buffers" : "today's module";
                const bool shapeRight = resolved ? shader.newPathResolved && !BindsGuestBuffers(shader) && BindsFlattenedSrt(shader) && shader.bdaWrites
                                                 : !shader.newPathResolved && BindsGuestBuffers(shader) && BindsFlattenedSrt(shader) && shader.rangesFromMemory;
                const bool rangeKept = std::ranges::any_of(shader.writeRanges, [&](const auto& range) { return range.first <= address(output) && address(output) + OutputRecords <= range.second; });
                std::printf("%s, %s source: marker %d, guest buffers %d, flattened SRT %d, ranges from memory %d, %zu write ranges%s\n", name, unmapped ? "unmapped" : "mapped", shader.newPathResolved ? 1 : 0, BindsGuestBuffers(shader) ? 1 : 0, BindsFlattenedSrt(shader) ? 1 : 0, shader.rangesFromMemory ? 1 : 0, shader.writeRanges.size(), rangeKept ? " (the output's among them)" : "");
                failures += !shapeRight || !rangeKept;
                try {
                    device->Dispatch(shader, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
                    device->WaitIdle();
                    ran[resolved ? 1 : 0] = true;
                } catch (const std::exception& error) {
                    std::printf("  %s dispatch threw: %s\n", name, error.what());
                    device->WaitIdle();
                }
                stored[resolved ? 1 : 0].assign(output, output + Threads);
                std::uint32_t wrong = 0, first = ~0u;
                for (std::uint32_t lane = 0; lane < Threads; ++lane) {
                    const auto expected = unmapped || lane * 4u + 4u > SourceRecords ? 0u : source[lane];
                    if (output[lane] != expected) {
                        if (first == ~0u) first = lane;
                        ++wrong;
                    }
                }
                std::printf("  %u of %u lanes differ from the expected copy", wrong, Threads);
                if (first != ~0u) std::printf("; first lane %u stored %08x", first, output[first]);
                std::printf("\n");
                if (resolved || !unmapped) failures += wrong != 0u || !ran[resolved ? 1 : 0];
                std::lock_guard lock(recordsMutex);
                if (!resolved) {
                    if (!records.empty()) {
                        std::printf("  today's module: %zu command records taken, expected none\n", records.size());
                        ++failures;
                    }
                    continue;
                }
                if (records.size() != 1u) {
                    std::printf("  resolved: %zu command records read back, expected 1\n", records.size());
                    ++failures;
                    continue;
                }
                const auto& words = records.front();
                Bda::Fault fault{};
                std::memcpy(&fault, words.data(), sizeof(fault));
                bool outputPage = false;
                std::uint32_t ranges = 0;
                const bool valid = Bda::ForEachWrittenRange(words.data(), [&](std::uint64_t begin, std::size_t size) {
                    ++ranges;
                    if (begin <= address(output) && address(output) + OutputRecords <= begin + size) outputPage = true;
                });
                std::printf("  record: fault state %u reason %u address 0x%llx bytes %u; overflow %u; %u written ranges%s\n", static_cast<std::uint32_t>(fault.state), static_cast<std::uint32_t>(fault.reason), static_cast<unsigned long long>(fault.address), fault.bytes, words[Bda::WrittenOverflowWord], ranges, outputPage ? ", the output page among them" : "");
                const bool faultRight = unmapped ? fault.state == Bda::FaultState::Ready && fault.reason == Bda::FaultReason::Unmapped && fault.address >= in && fault.address < in + SourceRecords && fault.bytes != 0u
                                                 : fault.state == Bda::FaultState::Empty;
                failures += !faultRight || !valid || words[Bda::WrittenOverflowWord] != 0u || !outputPage;
            }
            if (ran[0] && ran[1]) {
                const bool same = stored[0] == stored[1];
                std::printf("  %s source: the two modules stored %s bytes\n", unmapped ? "unmapped" : "mapped", same ? "the same" : "different");
                failures += !same;
            }
        }
        AgcDriver::NewPath::SetCommandRecordObserver({});
        std::printf("%s\n", failures == 0 ? "newpath runtime buffers: passed" : "newpath runtime buffers: FAILED");
        std::fflush(stdout);
        std::fflush(nullptr);
        TerminateProcess(GetCurrentProcess(), failures == 0 ? 0 : 1);
        std::_Exit(failures == 0 ? 0 : 1);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
