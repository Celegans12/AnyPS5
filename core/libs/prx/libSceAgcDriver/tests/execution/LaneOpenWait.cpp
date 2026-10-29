// A CPU wait for one lane's batch while the other lane still holds an open (unsubmitted) batch.
// On the hardware the two rings run side by side, so a lane's work may wait for (spin on) memory
// the other ring writes later in program order; here that other work sits in the other lane's open
// batch, and only a submit by the waiting thread (which holds the GPU mutex every other driver
// thread queues on) can reach the GPU. The awaited batch is held by a host gate that a watcher
// opens once the other lane's open batch was submitted (the stand-in for the GPU dependency); a
// wait that does not submit the other lane first is the #675 lane-1 freeze (Dextroid17, 10-28:
// FinishWritersByLane -> finishLaneThrough -> finish blocking on a lane-1 batch, lane 0 open).
#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <SDL_loadso.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Recorder;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("lane open wait: " + message);
}

class Gate {
public:
    explicit Gate(AgcDriver::VulkanDevice& device) : device(device) {
#ifdef _WIN32
        constexpr const char* libraryName = "vulkan-1.dll";
#else
        constexpr const char* libraryName = "libvulkan.so.1";
#endif
        library.reset(SDL_LoadObject(libraryName));
        Require(library != nullptr, "cannot load Vulkan for the gate");
        resolve = reinterpret_cast<PFN_vkGetDeviceProcAddr>(SDL_LoadFunction(library.get(), "vkGetDeviceProcAddr"));
        Require(resolve != nullptr, "cannot resolve vkGetDeviceProcAddr");
        destroyEvent = function<PFN_vkDestroyEvent>("vkDestroyEvent");
        setEvent = function<PFN_vkSetEvent>("vkSetEvent");
        waitEvents = function<PFN_vkCmdWaitEvents>("vkCmdWaitEvents");
        const VkEventCreateInfo info{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
        Require(function<PFN_vkCreateEvent>("vkCreateEvent")(device.Device(), &info, nullptr, &event) == VK_SUCCESS, "cannot create the gate event");
    }

    ~Gate() {
        setEvent(device.Device(), event);
        try {
            device.WaitIdle();
        } catch (...) {
        }
        destroyEvent(device.Device(), event, nullptr);
    }

    Gate(const Gate&) = delete;
    Gate& operator=(const Gate&) = delete;

    void Record(VkCommandBuffer commands) {
        waitEvents(commands, 1, &event, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, nullptr, 0, nullptr, 0, nullptr);
    }

    void Release() {
        Require(setEvent(device.Device(), event) == VK_SUCCESS, "cannot release the gate");
    }

private:
    template <typename TFunction>
    TFunction function(const char* name) const {
        const auto found = reinterpret_cast<TFunction>(resolve(device.Device(), name));
        Require(found != nullptr, std::string("cannot resolve ") + name);
        return found;
    }

    AgcDriver::VulkanDevice& device;
    std::unique_ptr<void, decltype(&SDL_UnloadObject)> library{nullptr, &SDL_UnloadObject};
    PFN_vkGetDeviceProcAddr resolve = nullptr;
    PFN_vkDestroyEvent destroyEvent = nullptr;
    PFN_vkSetEvent setEvent = nullptr;
    PFN_vkCmdWaitEvents waitEvents = nullptr;
    VkEvent event = VK_NULL_HANDLE;
};

enum class Waiter { WritersByLane, FinishUpTo };

const char* WaiterName(Waiter waiter) {
    return waiter == Waiter::WritersByLane ? "FinishWritersByLane" : "FinishUpTo";
}

// The awaited batch goes to `awaitedLane` (gated, noting a write of the range); the other lane then
// opens a batch that does not touch the range and stays open. The waiter runs on another thread
// under the GPU mutex, as the driver's submit thread does. Returns whether the other lane's open
// batch reached the GPU before the watcher's deadline (false: the wait would have deadlocked; the
// watcher opened the gate itself so the test ends).
bool RunCase(AgcDriver::VulkanDevice& device, int awaitedLane, Waiter waiter) {
    static std::vector<std::uint8_t> written(4096);
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(written.data()));
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> ranges{{{address + 256, address + 272}}};
    const int otherLane = 1 - awaitedLane;
    std::unique_lock lock(AgcDriver::GuestMemory::GpuMutex());
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "no active recorder");
    recorder->Sync();
    Gate gate(device);
    std::uint64_t awaitedSerial = 0;
    {
        const Recorder::LaneScope scope(awaitedLane);
        gate.Record(recorder->Commands());
        recorder->NotePendingWrite(ranges[0].first, static_cast<std::size_t>(ranges[0].second - ranges[0].first));
        recorder->Submit();
        awaitedSerial = recorder->UseSerial() - 1;
    }
    {
        const Recorder::LaneScope scope(otherLane);
        static_cast<void>(recorder->Commands());
    }
    // Nothing of the other lane's open batch has reached the GPU yet.
    const auto submittedBefore = recorder->UseSerial();
    std::atomic<bool> releasedBySubmit{false};
    std::atomic<bool> stop{false};
    std::exception_ptr failure;
    lock.unlock();
    std::thread watcher([&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
        while (!stop.load() && std::chrono::steady_clock::now() < deadline) {
            if (recorder->UseSerial() > submittedBefore) {
                releasedBySubmit.store(true);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        gate.Release();
    });
    std::thread waiting([&] {
        try {
            std::lock_guard waiterLock(AgcDriver::GuestMemory::GpuMutex());
            if (waiter == Waiter::WritersByLane) {
                const auto wait = recorder->FinishWritersByLane(ranges, 0, 0);
                Require(wait.targets[static_cast<std::size_t>(awaitedLane)] == awaitedSerial, "the writer wait did not target the awaited lane's batch");
            } else {
                recorder->FinishUpTo(awaitedSerial);
            }
        } catch (...) {
            failure = std::current_exception();
        }
    });
    waiting.join();
    stop.store(true);
    watcher.join();
    if (failure) std::rethrow_exception(failure);
    lock.lock();
    recorder->Sync();
    return releasedBySubmit.load();
}

}

int main() {
#ifdef _WIN32
    _putenv_s("APS5_ASYNC_LANE", "51");
#else
    setenv("APS5_ASYNC_LANE", "51", 1);
#endif
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        {
            std::unique_lock lock(AgcDriver::GuestMemory::GpuMutex());
            auto* recorder = Recorder::Active();
            Require(recorder != nullptr, "no active recorder");
            if (!recorder->AsyncLaneActive()) {
                std::puts("skipped: the device has no third queue for the async lane");
                lock.unlock();
                device.reset();
                AgcDriverShutdown_nid_postfix();
                return VulkanTestSkipped;
            }
        }
        int failures = 0;
        for (const auto waiter : {Waiter::WritersByLane, Waiter::FinishUpTo}) {
            for (const int lane : {1, 0}) {
                const bool submitted = RunCase(*device, lane, waiter);
                if (submitted) {
                    std::printf("lane open wait: %s on a lane-%d batch submitted lane %d's open batch first\n", WaiterName(waiter), lane, 1 - lane);
                } else {
                    ++failures;
                    std::fprintf(stderr, "lane open wait: %s blocked on a lane-%d batch while lane %d's open batch, which the awaited work can depend on, was never submitted (deadlock; the watcher released the gate after 2 s)\n", WaiterName(waiter), lane, 1 - lane);
                }
            }
        }
        Require(failures == 0, std::to_string(failures) + " of 4 waits blocked in front of the other lane's unsubmitted batch");
        device.reset();
        std::puts("lane open wait tests passed");
        AgcDriverShutdown_nid_postfix();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (const std::exception& shutdownError) {
            std::cerr << shutdownError.what() << '\n';
        }
        return 1;
    }
}
