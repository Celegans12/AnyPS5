#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <SDL_loadso.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("lane template edge: " + message);
}

alignas(256) constexpr std::array<std::uint32_t, 14> IndexStoreCode{0xf4040100u, 0xfa000000u, 0xf4080201u, 0xfa000000u, 0xbf8cc07fu, 0xf4000182u, 0xfa000000u, 0xbf8cc07fu, 0x7e000206u, 0x34020082u, 0xe0701000u, 0x80020001u, 0xbf810000u, 0xbf800000u};

constexpr std::size_t GuestBytes = 1u << 16u;

struct Slot {
    alignas(256) std::array<std::uint32_t, 4> value{};
    alignas(256) std::array<std::uint32_t, 2> table{};
};

struct Guest {
    std::array<Slot, 2> slots;
    alignas(256) std::array<std::uint32_t, 4> output{};
    alignas(256) std::array<std::uint32_t, 4> outputDescriptor{};
};

std::uint32_t Low(const void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint32_t High(const void* pointer) {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)) >> 32u);
}

ShaderRecompiler::RecompileResult Recompile(AgcDriver::VulkanDevice& device, Guest& guest, Slot& slot, std::uint32_t value) {
    slot.value = {value, 0, 0, 0};
    slot.table = {Low(slot.value.data()), High(slot.value.data())};
    const std::array<std::uint32_t, 4> userData{Low(slot.table.data()), High(slot.table.data()), Low(guest.outputDescriptor.data()), High(guest.outputDescriptor.data())};
    const std::array<ShaderRecompiler::MemoryRegion, 4> memory{{
        {reinterpret_cast<std::uintptr_t>(IndexStoreCode.data()), std::as_bytes(std::span(IndexStoreCode))},
        {reinterpret_cast<std::uintptr_t>(slot.table.data()), std::as_bytes(std::span(slot.table))},
        {reinterpret_cast<std::uintptr_t>(slot.value.data()), std::as_bytes(std::span(slot.value))},
        {reinterpret_cast<std::uintptr_t>(guest.outputDescriptor.data()), std::as_bytes(std::span(guest.outputDescriptor))},
    }};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{1, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(IndexStoreCode.data()), std::span(IndexStoreCode), 0, {}},
        {32, 0, std::span(userData), compute, std::nullopt, std::nullopt, memory},
        device.ComputeTarget(32),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
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

std::uint32_t Load(const std::uint32_t& value) {
    return reinterpret_cast<const volatile std::uint32_t&>(value);
}

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
        }
#ifdef _WIN32
        auto* block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, GuestBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        auto* block = static_cast<std::uint8_t*>(std::aligned_alloc(GuestBytes, GuestBytes));
#endif
        Require(block != nullptr, "cannot allocate guest memory");
        std::memset(block, 0, GuestBytes);
        GuestAllocations::Mutation().Add(block, GuestBytes, true, true, true);
        auto* guest = new (block) Guest{};
        guest->outputDescriptor = {Low(guest->output.data()), High(guest->output.data()) & 0xffffu, static_cast<std::uint32_t>(sizeof(guest->output)), 0x31016facu};
        const auto program = reinterpret_cast<std::uintptr_t>(IndexStoreCode.data());
        const auto first = Recompile(*device, *guest, guest->slots[0], 1);
        auto second = first;
        for (auto& binding : second.bindings) {
            if (binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt && binding.guestDescriptor.size() > 6 && binding.guestDescriptor[6] == 1u) binding.guestDescriptor[6] = 2u;
        }
        guest->slots[0].value[0] = 2;
        {
            std::unique_lock lock(AgcDriver::GuestMemory::GpuMutex());
            auto* recorder = Recorder::Active();
            recorder->Sync();
        }
        {
            for (int warm = 0; warm < 3; ++warm) {
                const auto& shader = warm == 1 ? second : first;
                const auto prepared = device->PrepareDispatch(shader, {}, 1, program);
                std::unique_lock lock(AgcDriver::GuestMemory::GpuMutex());
                device->Dispatch(shader, 1, 1, 1, {}, program, prepared);
                Recorder::Active()->Sync();
            }
            AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(guest->output.data()), sizeof(guest->output));
            Require(Load(guest->output[1]) == 1u && Load(guest->output[2]) == 2u, "the warm-up dispatches did not store their values");
            for (auto& word : guest->output) reinterpret_cast<volatile std::uint32_t&>(word) = 0;
            Gate gate(*device);
            const auto firstPrepared = device->PrepareDispatch(first, {}, 1, program);
            {
                std::unique_lock lock(AgcDriver::GuestMemory::GpuMutex());
                auto* recorder = Recorder::Active();
                const Recorder::LaneScope async(1);
                gate.Record(recorder->Commands());
                device->Dispatch(first, 1, 1, 1, {}, program, firstPrepared);
            }
            const auto secondPrepared = device->PrepareDispatch(second, {}, 1, program);
            {
                std::unique_lock lock(AgcDriver::GuestMemory::GpuMutex());
                auto* recorder = Recorder::Active();
                device->Dispatch(second, 1, 1, 1, {}, program, secondPrepared);
                recorder->Submit();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            gate.Release();
            {
                std::unique_lock lock(AgcDriver::GuestMemory::GpuMutex());
                Recorder::Active()->Sync();
            }
        }
        AgcDriverWaitIdle_nid_postfix();
        AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(guest->output.data()), sizeof(guest->output));
        const auto one = Load(guest->output[1]);
        const auto two = Load(guest->output[2]);
        std::printf("lane template edge: output %u %u %u %u\n", Load(guest->output[0]), one, two, Load(guest->output[3]));
        Require(two == 2u, "the main lane's dispatch stored " + std::to_string(two) + " at index 2, expected 2");
        Require(one == 1u, "the async lane's dispatch, recorded first, read the cached template's words after the main lane refreshed them: index 1 holds " + std::to_string(one) + ", expected 1");
        device.reset();
        std::puts("lane template edge tests passed");
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
