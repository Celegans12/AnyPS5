#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/CommitPrep.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ParallelProfile.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <cstdlib>
#include <utility>

namespace AgcDriver::DriverDetail::CommitPrep {

namespace {

thread_local const Stamps* commitStamps = nullptr;

void takeGlobal(Stamps& stamps, const void* device) {
    stamps.taken = true;
    stamps.device = device;
    stamps.forgetSerial = GuestMemory::ForgetSerial();
    stamps.registryGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    stamps.epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
    stamps.proofSerial = GuestMemory::ProofSerial();
    stamps.pendingSerial = Graphics::StorageTexture::PendingSerial();
}

void takeReads(Stamps& stamps) {
    stamps.gpuPending = 0;
    for (std::size_t index = 0; index < stamps.reads; ++index) {
        const auto [begin, end] = stamps.ranges[index];
        const auto bytes = static_cast<std::size_t>(end - begin);
        stamps.cpu[index] = GuestMemory::NewestStamp(begin, bytes);
        if (Graphics::Recorder::SnapshotWriteOverlaps(begin, bytes)) stamps.gpuPending |= 1u << index;
    }
}

}

Stamps Take(const void* device, std::span<const Range> reads) {
    Stamps stamps;
    takeGlobal(stamps, device);
    for (const auto& range : reads) {
        if (stamps.reads == Stamps::MaxReads) break;
        if (range.first >= range.second) continue;
        stamps.ranges[stamps.reads++] = range;
    }
    takeReads(stamps);
    return stamps;
}

Stamps Retake(const Stamps& then, const void* device) {
    Stamps now;
    takeGlobal(now, device);
    now.reads = then.reads;
    now.ranges = then.ranges;
    takeReads(now);
    return now;
}

std::uint32_t Moved(const Stamps& then, const Stamps& now) {
    using Graphics::ParallelProfile::Stamp;
    using Graphics::ParallelProfile::StampBit;
    std::uint32_t moved = 0;
    if (then.forgetSerial != now.forgetSerial || (now.forgetSerial & 1u) != 0) moved |= StampBit(Stamp::Forget);
    if (then.registryGeneration != now.registryGeneration) moved |= StampBit(Stamp::Registry);
    if (then.device != now.device) moved |= StampBit(Stamp::Device);
    if (then.epoch != now.epoch) moved |= StampBit(Stamp::Epoch);
    if (then.proofSerial != now.proofSerial) moved |= StampBit(Stamp::Proof);
    if (then.pendingSerial != now.pendingSerial) moved |= StampBit(Stamp::Pending);
    if ((now.gpuPending & ~then.gpuPending) != 0) moved |= StampBit(Stamp::GpuWrite);
    for (std::size_t index = 0; index < std::min(then.reads, now.reads); ++index) {
        if (then.cpu[index] != now.cpu[index]) {
            moved |= StampBit(Stamp::CpuWrite);
            break;
        }
    }
    return moved;
}

namespace {

thread_local bool stageAByCommitter = false;

constexpr std::uint8_t PredictScore = 2;
constexpr std::uint8_t MaxScore = 4;

std::uint64_t mixKey(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    return hash;
}

}

bool LockSplit() {
    return !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoCommitLockSplit);
}

bool VerifyPrep() {
    static const bool enabled = std::getenv("APS5_VERIFY_PREP") != nullptr;
    return enabled;
}

void NoteStageA() {
    stageAByCommitter = true;
}

bool TakeStageA() {
    return std::exchange(stageAByCommitter, false);
}

std::size_t DrainPredictor::index(std::uint64_t key) {
    return static_cast<std::size_t>((key * 0x9e3779b97f4a7c15ull) >> 52u) & (Entries - 1u);
}

DrainPredictor::Plan DrainPredictor::Predict(std::uint64_t key) const {
    if (key == 0) return Plan::None;
    const auto& entry = entries[index(key)];
    if (entry.key != key || entry.score < PredictScore) return Plan::None;
    return entry.open ? Plan::Open : Plan::Submitted;
}

void DrainPredictor::Learn(std::uint64_t key, Plan used, double preDrainUs, double lockedWaitUs) {
    if (key == 0) return;
    auto& entry = entries[index(key)];
    const bool waited = lockedWaitUs >= WaitThresholdUs;
    if (entry.key != key) {
        if (waited) entry = {key, PredictScore, false};
        return;
    }
    if (waited) {
        entry.score = std::min<std::uint8_t>(MaxScore, static_cast<std::uint8_t>(entry.score + 1u));
        if (used == Plan::Submitted) entry.open = true;
        return;
    }
    if (used != Plan::None && preDrainUs >= WaitThresholdUs) return;
    if (entry.score != 0) --entry.score;
    if (entry.score == 0) entry.open = false;
}

std::uint64_t ItemKey(bool draw, std::uint64_t labelAddress, std::span<const Range> writes, std::span<const Range> reads, std::uint64_t site) {
    std::uint64_t key = mixKey(0x51ed27u, draw ? 1u : 2u);
    if (site != 0) key = mixKey(key, site);
    if (site != 0) return key != 0 ? key : 1u;
    key = mixKey(key, labelAddress);
    if (!writes.empty()) key = mixKey(mixKey(key, writes.front().first), writes.front().second);
    if (!reads.empty()) key = mixKey(mixKey(key, reads.front().first), reads.front().second);
    return key != 0 ? key : 1u;
}

void SetCommitStamps(const Stamps* stamps) {
    commitStamps = stamps;
}

const Stamps* CommitStamps() {
    return commitStamps;
}

}
