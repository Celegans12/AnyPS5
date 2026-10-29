#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyTokens.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cstdlib>
#endif
#include <cstdio>
#include <exception>
#include <mutex>
#include <string>

namespace {

using AgcDriver::Graphics::Recorder;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

}

int main() {
    try {
        auto device = OpenVulkanTestDevice();
        if (device == nullptr) return VulkanTestSkipped;
        auto* recorder = Recorder::Active();
        if (recorder == nullptr) {
            std::puts("skipped, the device has no active recorder");
            return VulkanTestSkipped;
        }
        constexpr std::size_t BlockBytes = 65536;
#ifdef _WIN32
        auto* block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        auto* block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        const auto address = reinterpret_cast<std::uintptr_t>(block);
        constexpr std::size_t scratchBytes = 64;
        recorder->NotePendingWrite(address, scratchBytes);
        const auto caller = recorder->CallerToken();
        Expect(caller[0] == recorder->OpenSerial() && caller[1] == 0, "the caller's token does not name its lane's open batch only");
        const auto all = recorder->AllToken();
        Expect(all[0] == recorder->OpenSerial() && all[1] == recorder->OpenSerial(), "the token of all lanes does not name both lanes' open serial");
        Expect(Recorder::TokenThrough(all) == recorder->OpenSerial(), "a token's through serial is not its newest lane value");
        Expect(!recorder->TokenReached(all), "a token of the open batch counts as reached before it ran");
        recorder->FinishToken(all);
        Expect(recorder->TokenReached(all) && recorder->TokenReached(caller), "finishing a token did not reach it");
        Expect(!recorder->PendingWriteOverlaps(address, scratchBytes), "a finished token's batch still holds its pending write");
        Recorder::Token past{};
        past[2] = 1;
        bool refused = false;
        try {
            static_cast<void>(Recorder::TokenThrough(past));
        } catch (const std::exception&) {
            refused = true;
        }
        Expect(refused, "a token naming a stream past the two lanes was accepted");
        recorder->FinishToken(caller);

        recorder->NotePendingWrite(address + 128, scratchBytes);
        const auto held = recorder->AllToken();
        {
            const Recorder::CommandInHand inHand;
            bool refusedInHand = false;
            try {
                recorder->FinishToken(held);
            } catch (const std::exception&) {
                refusedInHand = true;
            }
            Expect(refusedInHand && !recorder->TokenReached(held), "a token was finished while a command buffer was in hand");
        }
        auto lanes = AgcDriver::Graphics::RecorderLanes(*recorder);
        AgcDriver::Graphics::ResidencyTokens tokens(*lanes);
        Expect(tokens.Caller() == recorder->CallerToken() && tokens.All() == recorder->AllToken(), "the residency tokens are not the recorder's tokens");
        {
            const Recorder::CommandInHand inHand;
            bool refusedInHand = false;
            try {
                tokens.Finish(held);
            } catch (const std::exception&) {
                refusedInHand = true;
            }
            Expect(refusedInHand, "the residency adapter finished a token while a command buffer was in hand (R-16)");
        }
        tokens.Finish(held);
        Expect(tokens.Reached(held) && recorder->TokenReached(held), "the residency adapter did not finish through the recorder's tokens");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d recorder token checks failed\n", failures);
        return 1;
    }
    std::puts("recorder token tests passed");
    return 0;
}
