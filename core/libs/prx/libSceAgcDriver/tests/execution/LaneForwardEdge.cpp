#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <SDL_loadso.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <array>
#include <chrono>
#include <cstddef>
#include <thread>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("lane forward edge: " + message);
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

}

int main() {
    using AgcDriver::Graphics::Recorder;
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
            recorder->Sync();
            Gate gate(*device);
            {
                const Recorder::LaneScope async(1);
                gate.Record(recorder->Commands());
                recorder->Submit();
            }
            static_cast<void>(recorder->Commands());
            {
                const Recorder::LaneScope async(1);
                const Recorder::LaneSwitch toMain(recorder, 0);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            const auto unsignaled = recorder->UnsignaledBatches();
            Require(unsignaled == 1, "the main lane's batch, recorded before the async lane's work switched to the main lane, waited on the GPU for the async lane's later batch (" + std::to_string(unsignaled) + " batches unsignaled while the async lane was held, expected 1): a cross-lane edge held back work recorded before it");
            gate.Release();
            recorder->Sync();
            std::puts("lane forward edge: the main lane's earlier work ran while the async lane was held");
        }
        device.reset();
        std::puts("lane forward edge tests passed");
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
