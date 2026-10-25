#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/CommitPrep.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ParallelProfile.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>

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

void SetCommitStamps(const Stamps* stamps) {
    commitStamps = stamps;
}

const Stamps* CommitStamps() {
    return commitStamps;
}

}
