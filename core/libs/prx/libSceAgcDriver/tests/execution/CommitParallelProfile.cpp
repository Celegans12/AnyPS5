#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/CommitPrep.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ParallelProfile.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace Profile = AgcDriver::Graphics::ParallelProfile;
namespace Prep = AgcDriver::DriverDetail::CommitPrep;
using Profile::Stamp;
using Profile::StampBit;

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("commit parallel profile: " + what);
}

void percentiles() {
    std::array<std::uint64_t, 66> histogram{};
    check(Profile::Percentile(histogram, 0.5) == 0, "an empty histogram has a percentile");
    histogram[1] = 5;
    histogram[4] = 4;
    histogram[20] = 1;
    check(Profile::Percentile(histogram, 0.5) == 1, "p50 of 5x1 4x4 1x20 is not 1");
    check(Profile::Percentile(histogram, 0.9) == 4, "p90 of 5x1 4x4 1x20 is not 4");
    check(Profile::Percentile(histogram, 1.0) == 20, "p100 of 5x1 4x4 1x20 is not 20");
}

void gate() {
    check(Profile::Gate({0.42, 0.85, 8}).Pass(), "a gate above every bound failed");
    const auto movable = Profile::Gate({0.39, 0.85, 8});
    check(!movable.Pass() && !movable.movable && movable.survival && movable.chunks, "a movable share below 40% passed");
    const auto survival = Profile::Gate({0.5, 0.79, 12});
    check(!survival.Pass() && !survival.survival, "a survival below 80% passed");
    const auto chunks = Profile::Gate({0.5, 0.9, 7});
    check(!chunks.Pass() && !chunks.chunks, "a chunk p50 below 8 draws passed");
}

void stamps() {
    int deviceA = 0, deviceB = 0;
    const std::vector<Prep::Range> reads{{0x10000, 0x10100}, {0x20000, 0x20040}};
    const auto then = Prep::Take(&deviceA, reads);
    check(then.taken && then.reads == 2, "the stamps did not keep the read ranges");
    check(Prep::Moved(then, Prep::Retake(then, &deviceA)) == 0, "unchanged state moved a stamp");
    check(Prep::Moved(then, Prep::Retake(then, &deviceB)) == StampBit(Stamp::Device), "a replaced device did not move the device stamp alone");
    AgcDriver::DriverDetail::DrawPipeline::EpochToken().fetch_add(1);
    check((Prep::Moved(then, Prep::Retake(then, &deviceA)) & StampBit(Stamp::Epoch)) != 0, "an epoch bump did not move the epoch stamp");
    AgcDriver::GuestMemory::BreakProofs();
    check((Prep::Moved(then, Prep::Retake(then, &deviceA)) & StampBit(Stamp::Proof)) != 0, "a proof break did not move the proof stamp");
    AgcDriver::Graphics::StorageTexture::BumpPendingSerial();
    check((Prep::Moved(then, Prep::Retake(then, &deviceA)) & StampBit(Stamp::Pending)) != 0, "a pending-results change did not move the pending stamp");
    static std::array<std::uint8_t, 4096> block{};
    GuestAllocations::Mutation().Add(block.data(), block.size(), true, true, true);
    const auto moved = Prep::Moved(then, Prep::Retake(then, &deviceA));
    GuestAllocations::Mutation().Remove(block.data());
    check((moved & StampBit(Stamp::Registry)) != 0, "a registry change did not move the registry stamp");
    const auto many = std::vector<Prep::Range>(12, Prep::Range{0x1000, 0x2000});
    check(Prep::Take(&deviceA, many).reads == Prep::Stamps::MaxReads, "the stamps kept more read ranges than they hold");
}

}

int main() {
    try {
        percentiles();
        gate();
        stamps();
        std::puts("commit parallel profile tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
