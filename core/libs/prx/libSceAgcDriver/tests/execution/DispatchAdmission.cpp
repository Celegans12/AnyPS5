#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::DriverDetail::Driver;

constexpr std::array<std::uint32_t, 7> Code{0xf4080100u, 0xfa000000u, 0xbf8cc07fu, 0x7e020280u, 0xe0702000u, 0x80010100u, 0xbf810000u};
constexpr std::size_t SrtOffset = 4096;
constexpr std::size_t OutputOffset = 8192;
constexpr std::size_t BlockBytes = 65536;

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, PAGE_READWRITE, BlockBytes);
#else
        void* mapped = mmap(nullptr, BlockBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            block = static_cast<std::uint8_t*>(mapped);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, BlockBytes);
        }
#endif
        Require(block != nullptr, "dispatch admission: cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint64_t At(std::size_t offset) const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block + offset)); }
    std::uint8_t* Bytes(std::size_t offset) const { return block + offset; }

private:
    std::uint8_t* block = nullptr;
};

std::vector<std::uint32_t> Pm4(std::uint32_t opcode, std::initializer_list<std::uint32_t> body) {
    std::vector<std::uint32_t> words{0xc0000000u | (static_cast<std::uint32_t>(body.size() - 1) << 16u) | (opcode << 8u)};
    words.insert(words.end(), body);
    return words;
}

void WriteSrt(const GuestBlock& block, std::size_t srt, std::size_t output) {
    const auto address = block.At(output);
    const std::array<std::uint32_t, 4> words{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u), 64u, 0x00014204u};
    std::memcpy(block.Bytes(srt), words.data(), sizeof(words));
}

void Dispatch(const GuestBlock& block, std::size_t srtOffset, std::size_t output) {
    std::uint32_t poison = 0xa5a5a5a5u;
    std::memcpy(block.Bytes(output), &poison, sizeof(poison));
    AgcDriver::GuestMemory::MarkWritten(block.At(output), sizeof(poison));
    const auto kernel = block.At(0);
    const auto srt = block.At(srtOffset);
    std::vector<std::uint32_t> words;
    for (const auto& packet : {Pm4(0x76, {0x20c, static_cast<std::uint32_t>(kernel >> 8u)}), Pm4(0x76, {0x20d, static_cast<std::uint32_t>(kernel >> 40u) & 0xffu}), Pm4(0x76, {0x207, 64}), Pm4(0x76, {0x208, 1}), Pm4(0x76, {0x209, 1}), Pm4(0x76, {0x213, 2u << 1u}), Pm4(0x76, {0x240, static_cast<std::uint32_t>(srt)}), Pm4(0x76, {0x241, static_cast<std::uint32_t>(srt >> 32u)}), Pm4(0x15, {1, 1, 1, 0x41})}) {
        words.insert(words.end(), packet.begin(), packet.end());
    }
    Packet dcb{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&dcb) == 0, "dispatch admission: the dispatch submission failed");
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(block.At(output), sizeof(poison));
    std::uint32_t stored = 0;
    std::memcpy(&stored, block.Bytes(output), sizeof(stored));
    char text[16];
    std::snprintf(text, sizeof(text), "%08x", stored);
    Require(stored == 0u, std::string("dispatch admission: the dispatch did not store its output (0x") + text + ")");
}

struct Counts {
    std::uint64_t inserts = 0, hits = 0, skipped = 0, admitted = 0, probes = 0, probeHits = 0;
};

Counts Read() {
    const auto entries = Driver::Get().DispatchCacheCounters();
    const auto admission = Driver::Get().DispatchAdmissionCounts();
    return {entries.inserts, entries.equal, admission[0], admission[1], admission[2], admission[3]};
}

Counts Delta(const Counts& before) {
    const auto now = Read();
    return {now.inserts - before.inserts, now.hits - before.hits, now.skipped - before.skipped, now.admitted - before.admitted, now.probes - before.probes, now.probeHits - before.probeHits};
}

void Expect(const Counts& delta, const Counts& expected, const std::string& what) {
    std::printf("%s: inserts %llu hits %llu skipped %llu admitted %llu probes %llu probe hits %llu\n", what.c_str(), static_cast<unsigned long long>(delta.inserts), static_cast<unsigned long long>(delta.hits), static_cast<unsigned long long>(delta.skipped), static_cast<unsigned long long>(delta.admitted), static_cast<unsigned long long>(delta.probes), static_cast<unsigned long long>(delta.probeHits));
    Require(delta.inserts == expected.inserts && delta.hits == expected.hits && delta.skipped == expected.skipped && delta.admitted == expected.admitted && delta.probes == expected.probes && delta.probeHits == expected.probeHits, "dispatch admission: " + what + " did not count as expected");
}

void RepeatedKey(const GuestBlock& block, bool verify) {
    const std::size_t srt = SrtOffset;
    WriteSrt(block, srt, OutputOffset);
    auto before = Read();
    Dispatch(block, srt, OutputOffset);
    Expect(Delta(before), verify ? Counts{1, 0, 1, 0, 1, 0} : Counts{0, 0, 1, 0, 0, 0}, "the first sighting of a key");
    before = Read();
    Dispatch(block, srt, OutputOffset);
    Expect(Delta(before), verify ? Counts{0, 1, 0, 0, 0, 1} : Counts{1, 0, 0, 1, 0, 0}, "the second sighting of a key");
    before = Read();
    Dispatch(block, srt, OutputOffset);
    Expect(Delta(before), {0, 1, 0, 0, 0, 0}, "the third sighting of a key");
}

void OneOffKeys(const GuestBlock& block, bool verify) {
    const auto before = Read();
    for (std::size_t index = 0; index < 4; ++index) {
        const auto srt = SrtOffset + 256u * (index + 1u);
        WriteSrt(block, srt, OutputOffset + 64u * (index + 1u));
        Dispatch(block, srt, OutputOffset + 64u * (index + 1u));
    }
    Expect(Delta(before), verify ? Counts{4, 0, 4, 0, 4, 0} : Counts{0, 0, 4, 0, 0, 0}, "four keys seen once");
}

void AdmissionOff(const GuestBlock& block) {
    AgcDriver::Graphics::RuntimeSwitches::Assign(AgcDriver::Graphics::RuntimeSwitch::NoDispatchAdmission, true);
    const std::size_t srt = SrtOffset + 2048u;
    WriteSrt(block, srt, OutputOffset + 1024u);
    auto before = Read();
    Dispatch(block, srt, OutputOffset + 1024u);
    Expect(Delta(before), {1, 0, 0, 0, 0, 0}, "a first dispatch with APS5_NO_DISPATCH_ADMISSION");
    before = Read();
    Dispatch(block, srt, OutputOffset + 1024u);
    Expect(Delta(before), {0, 1, 0, 0, 0, 0}, "a second dispatch with APS5_NO_DISPATCH_ADMISSION");
    AgcDriver::Graphics::RuntimeSwitches::Assign(AgcDriver::Graphics::RuntimeSwitch::NoDispatchAdmission, false);
}

}

int main() {
    try {
        const char* verifyText = std::getenv("APS5_VERIFY_DISPATCH_ADMISSION");
        const bool verify = verifyText != nullptr && std::strtoull(verifyText, nullptr, 10) >= 8u;
        if (std::getenv("APS5_NO_DISPATCH_ADMISSION") != nullptr) {
            std::puts("dispatch admission: APS5_NO_DISPATCH_ADMISSION is set, nothing to test");
            return 0;
        }
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        GuestBlock block;
        std::memcpy(block.Bytes(0), Code.data(), sizeof(Code));
        RepeatedKey(block, verify);
        OneOffKeys(block, verify);
        AdmissionOff(block);
        AgcDriverShutdown_nid_postfix();
        std::puts(verify ? "dispatch admission: probes inserted on a first sighting are counted when they hit" : "dispatch admission: a key enters the dispatch cache on its second sighting, keys seen once are not inserted");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { AgcDriverShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
