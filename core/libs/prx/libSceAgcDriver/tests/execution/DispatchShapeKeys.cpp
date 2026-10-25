#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/ShapeReplay.hpp"
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
        Require(block != nullptr, "dispatch shape keys: cannot allocate the guest block");
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
    Require(sceAgcDriverSubmitDcb(&dcb) == 0, "dispatch shape keys: the dispatch submission failed");
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(block.At(output), sizeof(poison));
    std::uint32_t stored = 0;
    std::memcpy(&stored, block.Bytes(output), sizeof(stored));
    char text[16];
    std::snprintf(text, sizeof(text), "%08x", stored);
    Require(stored == 0u, std::string("dispatch shape keys: the dispatch did not store its output (0x") + text + ")");
}

std::uint32_t Read(const GuestBlock& block, std::size_t offset) {
    AgcDriver::GuestMemory::FlushGpuWrites(block.At(offset), 4);
    std::uint32_t value = 0;
    std::memcpy(&value, block.Bytes(offset), sizeof(value));
    return value;
}

}

int main() {
    namespace Replay = AgcDriver::DriverDetail::ShapeReplay;
    try {
        const bool off = std::getenv("APS5_NO_DISPATCH_SHAPE_KEYS") != nullptr;
        const char* verifyText = std::getenv("APS5_VERIFY_DISPATCH_SHAPE_KEYS");
        const bool verify = verifyText != nullptr && std::strtoull(verifyText, nullptr, 10) >= 16u;
        if (std::getenv("APS5_NO_RESULT_MEMO") == nullptr) {
            std::puts("dispatch shape keys: the replay needs APS5_NO_RESULT_MEMO, nothing to test");
            return 0;
        }
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        GuestBlock block;
        std::memcpy(block.Bytes(0), Code.data(), sizeof(Code));
        const auto before = Replay::Totals(Replay::Kind::Dispatch);
        constexpr std::size_t Rounds = 8;
        for (std::size_t round = 0; round < Rounds; ++round) {
            const auto srt = SrtOffset + 256u * round;
            const auto output = OutputOffset + 256u * round;
            WriteSrt(block, srt, output);
            Dispatch(block, srt, output);
            for (std::size_t earlier = 0; earlier < round; ++earlier) {
                const auto old = OutputOffset + 256u * earlier;
                Require(Read(block, old) == 0xa5a5a5a5u, "dispatch shape keys: round " + std::to_string(round) + " stored into the output of round " + std::to_string(earlier) + " (a replayed V# kept the old base)");
            }
            std::uint32_t poison = 0xa5a5a5a5u;
            std::memcpy(block.Bytes(output), &poison, sizeof(poison));
            AgcDriver::GuestMemory::MarkWritten(block.At(output), sizeof(poison));
        }
        const auto after = Replay::Totals(Replay::Kind::Dispatch);
        const auto hits = after.hits - before.hits;
        const auto learned = after.learned - before.learned;
        const auto probations = after.probations - before.probations;
        const auto verified = after.verified - before.verified;
        const auto mismatches = after.verifyMismatches - before.verifyMismatches + after.probationMismatches - before.probationMismatches;
        std::printf("dispatch shape keys: %zu dispatches with moved SRTs and outputs: %llu learned, %llu probations, %llu replayed, %llu verified, %llu mismatches\n", Rounds, static_cast<unsigned long long>(learned), static_cast<unsigned long long>(probations), static_cast<unsigned long long>(hits), static_cast<unsigned long long>(verified), static_cast<unsigned long long>(mismatches));
        Require(mismatches == 0, "dispatch shape keys: a replayed materialization differed from the full one");
        if (off) {
            Require(hits == 0 && learned == 0, "dispatch shape keys: APS5_NO_DISPATCH_SHAPE_KEYS still replayed");
        } else if (verify) {
            Require(learned == 1 && probations == 1 && verified == Rounds - 2u && hits == 0, "dispatch shape keys: verify mode did not compare every replay after the probation");
        } else {
            Require(learned == 1 && probations == 1 && hits == Rounds - 2u, "dispatch shape keys: the moved dispatches were not replayed from the shape's plan");
        }
        AgcDriverShutdown_nid_postfix();
        std::puts(off ? "dispatch shape keys: off, every dispatch materialized in full" : "dispatch shape keys: one plan per shape, moved SRT pointers and V# bases patched exactly");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { AgcDriverShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
