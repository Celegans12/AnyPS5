#include "prx/libSceAgcDriver/Graphics/include/ResidencyTokens.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <memory>
#include <mutex>

namespace AgcDriver::Graphics {

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

private:
    Recorder& recorder;
};

}

std::unique_ptr<IResidencyLanes> RecorderLanes(Recorder& recorder) {
    return std::make_unique<RecorderBackedLanes>(recorder);
}

}
