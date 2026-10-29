#include "prx/libSceAgcDriver/Graphics/include/ResidencyTokens.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>

namespace AgcDriver::Graphics {

static_assert(std::is_same_v<ResidencyToken, Recorder::Token>);
static_assert(ResidencyLaneSlots == Recorder::LaneStreams);

namespace {

class RecorderBackedLanes final : public IResidencyLanes {
public:
    explicit RecorderBackedLanes(Recorder& recorder) : recorder(recorder) {}

    int CallerLane() const override { return Recorder::ThreadLane(); }
    std::uint64_t OpenSerial() const override { return recorder.OpenSerial(); }
    std::uint64_t SubmittedSerial() const override { return recorder.UseSerial() - 1; }
    std::uint64_t CompletedThrough() const override {
        std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
        if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
        return recorder.CompletedThrough();
    }
    void SubmitAll() override {
        std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
        if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
        recorder.SubmitAllLanes();
    }
    void WaitThrough(std::uint64_t serial) override { recorder.WaitSerial(serial); }
    void Complete(std::uint64_t serial) override {
        std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
        if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
        recorder.FinishUpTo(serial);
    }
    bool NativeTokens() const override { return true; }
    ResidencyToken CallerToken() const override { return recorder.CallerToken(); }
    ResidencyToken AllToken() const override { return recorder.AllToken(); }
    bool TokenReached(const ResidencyToken& token) const override {
        std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
        if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
        return recorder.TokenReached(token);
    }
    void FinishToken(const ResidencyToken& token) override {
        std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
        const bool held = GuestMemory::GpuMutex().HeldByThisThread();
        if (!held) lock.lock();
        if (recorder.TokenReached(token)) return;
        if (Recorder::CommandBufferInHand()) throw std::logic_error("residency: a token was finished while this thread holds a command buffer for a command it has not recorded yet");
        recorder.SubmitAllLanes();
        const auto submitted = recorder.UseSerial() - 1;
        if (!held) lock.unlock();
        recorder.WaitSerial(submitted);
        if (!held) lock.lock();
        recorder.FinishUpTo(submitted);
        if (!recorder.TokenReached(token)) throw std::logic_error("residency: a token not reached after every submitted batch completed");
    }

private:
    Recorder& recorder;
};

}

std::unique_ptr<IResidencyLanes> RecorderLanes(Recorder& recorder) {
    return std::make_unique<RecorderBackedLanes>(recorder);
}

}
