// agc_driver_newpath_walk_device_tests: the new path's page-table walk on the GPU (S4 epoch 1, item 1). One wave64
// copies 64 dwords from one guest buffer to another through global_load/global_store with scalar base addresses, so
// every access goes through the BDA lookup. The program is recompiled with SpirvTarget::newPathResolved off (the
// range table) and on (the device page table, root pushed at words 0-1 by VulkanDevice through
// NewPath::ResolvedModule), and each copy must equal the source. A second pass points the source at an address no
// allocation covers (64 KiB): both layouts must store zeros there (Unmapped), and the device must survive.
// The resolved module records its fault and written pages in the dispatch's command record (push words 2-3, D's
// CommandRecord); the test reads each record back at completion (NewPath::SetCommandRecordObserver): a mapped copy
// leaves no fault and reports the output's 4 KiB page written; an unmapped source reports an Unmapped fault inside the
// source's 256 bytes, and still the output page (the zeros stored there).
// Cases are a subset of fable-void's PageTranslationWaves (wip/new-gpu-path b6fab39bf): split pages and segments need a
// hand-built table, which the driver's table does not take from a test; NewPathPageTable covers the table side.
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
#include <vector>

namespace {

constexpr std::uint32_t Threads = 64;

alignas(256) std::array<std::uint32_t, 32> Program{};

std::span<const std::uint32_t> Code() {
    std::vector<std::uint32_t> code;
    code.push_back((0x1au << 25u) | (1u << 17u) | (0u << 9u) | 130u);   // v_lshlrev_b32 v1, 2, v0
    code.push_back(0xdc308000u); code.push_back((2u << 24u) | (0u << 16u) | (0u << 8u) | 1u);   // global_load_dword v2, v1, s[0:1]
    code.push_back(0xbf8c0000u);                                        // s_waitcnt 0
    code.push_back(0xdc708000u); code.push_back((0u << 24u) | (2u << 16u) | (2u << 8u) | 1u);   // global_store_dword v1, v2, s[2:3]
    code.push_back(0xbf8c0000u);                                        // s_waitcnt 0
    code.push_back(0xbf810000u);                                        // s_endpgm
    std::copy(code.begin(), code.end(), Program.begin());
    return std::span<const std::uint32_t>(Program.data(), code.size());
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        constexpr std::size_t bytes = 4096;
        auto* memory = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, bytes * 2u, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        auto* source = memory;
        auto* output = memory + bytes / 4u;
        GuestAllocations::Mutation().Add(memory, bytes * 2u, true, true, true);
        for (std::uint32_t lane = 0; lane < Threads; ++lane) source[lane] = 0x5a000000u | (lane * 0x10101u);
        const auto code = Code();
        const std::array<ShaderRecompiler::MemoryRegion, 1> regions{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
        const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {true, false, false}, false, 1};
        const auto out = reinterpret_cast<std::uint64_t>(output);
        int failures = 0;
        namespace Bda = ShaderRecompiler::BdaAbi;
        std::mutex recordsMutex;
        std::vector<std::vector<std::uint32_t>> records;
        AgcDriver::NewPath::SetCommandRecordObserver([&](std::span<const std::uint32_t, AgcDriver::NewPath::CommandRecordWords> words) {
            std::lock_guard lock(recordsMutex);
            records.emplace_back(words.begin(), words.end());
        });
        for (const bool unmapped : {false, true}) {
            const std::uint64_t in = unmapped ? 0x10000u : reinterpret_cast<std::uint64_t>(source);
            const std::vector<std::uint32_t> userData{static_cast<std::uint32_t>(in), static_cast<std::uint32_t>(in >> 32u), static_cast<std::uint32_t>(out), static_cast<std::uint32_t>(out >> 32u)};
            for (const bool resolved : {false, true}) {
                std::memset(output, 0xff, Threads * 4u);
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
                request.useCache = false;
                const auto shader = ShaderRecompiler::Recompile(request);
                device->Dispatch(shader, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
                device->WaitIdle();
                std::uint32_t wrong = 0, first = ~0u;
                for (std::uint32_t lane = 0; lane < Threads; ++lane) {
                    const auto expected = unmapped ? 0u : source[lane];
                    if (output[lane] != expected) {
                        if (first == ~0u) first = lane;
                        ++wrong;
                    }
                }
                std::printf("%s, %s source (marker %d): %u of %u lanes wrong", resolved ? "page-table walk" : "range table", unmapped ? "unmapped" : "mapped", shader.newPathResolved ? 1 : 0, wrong, Threads);
                if (first != ~0u) std::printf("; first lane %u stored %08x", first, output[first]);
                std::printf("\n");
                failures += wrong != 0u || shader.newPathResolved != resolved;
                std::lock_guard lock(recordsMutex);
                if (!resolved) {
                    if (!records.empty()) {
                        std::printf("  range table: %zu command records taken, expected none\n", records.size());
                        ++failures;
                    }
                    continue;
                }
                if (records.size() != 1u) {
                    std::printf("  page-table walk: %zu command records read back, expected 1\n", records.size());
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
                    if (begin <= out && out + Threads * 4u <= begin + size) outputPage = true;
                });
                std::printf("  record: fault state %u reason %u address 0x%llx bytes %u; overflow %u; %u written ranges%s\n", static_cast<std::uint32_t>(fault.state), static_cast<std::uint32_t>(fault.reason), static_cast<unsigned long long>(fault.address), fault.bytes, words[Bda::WrittenOverflowWord], ranges, outputPage ? ", the output page among them" : "");
                const bool faultRight = unmapped ? fault.state == Bda::FaultState::Ready && fault.reason == Bda::FaultReason::Unmapped && fault.address >= in && fault.address < in + Threads * 4u && fault.bytes != 0u
                                                 : fault.state == Bda::FaultState::Empty;
                failures += !faultRight || !valid || words[Bda::WrittenOverflowWord] != 0u || !outputPage;
            }
        }
        AgcDriver::NewPath::SetCommandRecordObserver({});
        std::fflush(stdout);
        std::fflush(nullptr);
        TerminateProcess(GetCurrentProcess(), failures == 0 ? 0 : 1);
        std::_Exit(failures == 0 ? 0 : 1);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
