#include "prx/libSceAgcDriver/Graphics/include/ResidencyTokens.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;

int failures = 0;

void Check(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

struct Batch {
    int lane = 0;
    std::uint64_t openedAt = 0;
    std::uint64_t serial = 0;
    bool gpuDone = false;
    bool reaped = false;
};

class FakeLanes final : public IResidencyLanes {
public:
    int caller = 0;
    std::uint64_t submissions = 0;
    std::vector<Batch> batches;
    int submitAll = 0;
    int waits = 0;
    int completes = 0;

    void Open(int lane) {
        if (openOf(lane) == nullptr) batches.push_back({lane, submissions + 1, 0, false, false});
    }
    void Submit(int lane) {
        auto* batch = openOf(lane);
        if (batch != nullptr) batch->serial = ++submissions;
    }
    void GpuFinishes(std::uint64_t serial) {
        for (auto& batch : batches) {
            if (batch.serial != 0 && batch.serial <= serial) batch.gpuDone = true;
        }
    }

    int CallerLane() const override { return caller; }
    std::uint64_t OpenSerial() const override { return submissions + 1; }
    std::uint64_t SubmittedSerial() const override { return submissions; }
    std::uint64_t CompletedThrough() const override {
        std::uint64_t through = submissions;
        for (const auto& batch : batches) {
            if (!batch.reaped) through = std::min(through, batch.openedAt - 1);
        }
        return through;
    }
    void SubmitAll() override {
        ++submitAll;
        Submit(0);
        Submit(1);
    }
    void WaitThrough(std::uint64_t serial) override {
        ++waits;
        GpuFinishes(serial);
    }
    void Complete(std::uint64_t serial) override {
        ++completes;
        for (auto& batch : batches) {
            if (batch.serial != 0 && batch.serial <= serial && batch.gpuDone) batch.reaped = true;
        }
    }

private:
    Batch* openOf(int lane) {
        for (auto& batch : batches) {
            if (batch.lane == lane && batch.serial == 0) return &batch;
        }
        return nullptr;
    }
};

}

int main() {
    FakeLanes lanes;
    ResidencyTokens tokens(lanes);

    lanes.Open(0);
    lanes.Open(1);
    lanes.caller = 1;
    const auto lane1 = tokens.Caller();
    Check(lane1[1] == 1 && lane1[0] == 0, "a lane-1 caller's token names slot 1 with the open serial");
    lanes.caller = 0;
    const auto lane0 = tokens.Caller();
    Check(lane0[0] == 1 && lane0[1] == 0, "a lane-0 caller's token names slot 0");
    const auto all = tokens.All();
    Check(all[0] == 1 && all[1] == 1, "All names both lanes");

    lanes.Submit(0);
    lanes.GpuFinishes(1);
    lanes.Complete(1);
    Check(!tokens.Reached(lane1), "lane 1's open batch was opened at serial 1: its token is not reached when lane 0's batch 1 completed");
    Check(!tokens.WaitUnlocked(lane1), "WaitUnlocked never waits for an unreached token");
    Check(lanes.waits == 0, "WaitUnlocked waited");

    tokens.Finish(lane1);
    Check(tokens.Reached(lane1) && tokens.Reached(lane0) && tokens.Reached(all), "Finish drains: every token recorded before it is reached");
    Check(lanes.submitAll == 1 && lanes.waits == 1 && lanes.completes >= 1, "Finish submits both lanes, waits once and reaps");
    Check(lanes.batches[1].serial == 2, "lane 1's batch got the later serial");

    const auto before = lanes.waits;
    tokens.Finish(lane1);
    Check(lanes.waits == before, "Finish of a reached token does not wait");
    Check(tokens.WaitUnlocked(lane1), "WaitUnlocked of a reached token is true");

    ResidencyToken empty{};
    Check(tokens.Reached(empty) && ResidencyTokens::Through(empty) == 0, "the empty token is reached");

    ResidencyToken foreign{};
    foreign[2] = 1;
    bool threw = false;
    try {
        static_cast<void>(ResidencyTokens::Through(foreign));
    } catch (const std::logic_error&) {
        threw = true;
    }
    Check(threw, "a token naming a stream past the two lanes throws");

    lanes.caller = 2;
    threw = false;
    try {
        static_cast<void>(tokens.Caller());
    } catch (const std::logic_error&) {
        threw = true;
    }
    Check(threw, "a caller lane without a slot throws");

    if (failures != 0) return 1;
    std::printf("agc_residency_tokens: all checks passed\n");
    return 0;
}
