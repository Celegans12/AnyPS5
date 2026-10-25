#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include <SDL_loadso.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using AgcDriver::Graphics::DescribeSurface;
using AgcDriver::Graphics::GuestTextureResource;
using AgcDriver::Graphics::Recorder;
using AgcDriver::Graphics::StorageTexture;
using AgcDriver::Graphics::TextureDetiler;
using AgcDriver::Graphics::TextureDimension;
using AgcDriver::Graphics::TextureTileMode;

constexpr std::uint32_t Width = 128;
constexpr std::uint32_t Height = 128;
constexpr std::uint32_t Format32Float = 22;
constexpr std::size_t UnitBytes = 65536;
constexpr std::uint32_t Cleared = 0x3f800000u;
constexpr std::uint8_t Sentinel = 0xa5;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("storage flush lane wait: " + message);
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

class GuestBlock {
public:
    explicit GuestBlock(std::size_t bytes) : bytes(bytes) {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, UnitBytes));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
        void* mapped = mmap(nullptr, bytes + UnitBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + UnitBytes - 1) & ~(static_cast<std::uintptr_t>(UnitBytes) - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + bytes != begin + bytes + UnitBytes) munmap(reinterpret_cast<void*>(aligned + bytes), begin + UnitBytes - aligned);
            block = reinterpret_cast<std::uint8_t*>(aligned);
            if (AgcDriver::GuestMemory::WriteWatched()) GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
        }
#endif
        Require(block != nullptr, "cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, bytes, true, true, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(block, bytes);
        GuestArena::GuestArenaRelease_nid_postfix(block, bytes);
#else
        munmap(block, bytes);
        if (AgcDriver::GuestMemory::WriteWatched()) GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, bytes);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint8_t* Data() { return block; }
    std::uint64_t Address() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block)); }

private:
    std::uint8_t* block = nullptr;
    std::size_t bytes = 0;
};

struct Outcome {
    bool returnedWhileGated = false;
    bool stored = false;
};

Outcome RunCase(AgcDriver::VulkanDevice& device, bool asyncWriter) {
    GuestTextureResource resource{};
    resource.width = Width;
    resource.height = Height;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = Format32Float;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    const auto guestBytes = static_cast<std::size_t>(DescribeSurface(resource).guestBytes);
    GuestBlock block((guestBytes + UnitBytes - 1) / UnitBytes * UnitBytes);
    resource.baseAddress = block.Address();
    std::memset(block.Data(), Sentinel, guestBytes);
    AgcDriver::GuestMemory::CollectWrites(resource.baseAddress, guestBytes);
    AgcDriver::GuestMemory::BumpCollectEpoch();

    std::unique_lock lock(AgcDriver::GuestMemory::GpuMutex());
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "no active recorder");
    recorder->Sync();
    auto context = device.graphicsContext();
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    Outcome outcome;
    {
        auto texture = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const auto commands = recorder->Commands();
        recorder->Keep(texture);
        VkClearColorValue value{};
        value.uint32[0] = value.uint32[1] = value.uint32[2] = value.uint32[3] = Cleared;
        AgcDriver::Graphics::RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, texture->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
        AgcDriver::Graphics::RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        texture->MarkDirty();
        Require(StorageTexture::FindPending(resource.baseAddress, guestBytes) != nullptr, "the marked image is not pending");
        recorder->Submit();
        Gate gate(device);
        {
            const Recorder::LaneScope async(1);
            gate.Record(recorder->Commands());
            if (asyncWriter) recorder->NotePendingWrite(resource.baseAddress + 256, 16);
            recorder->Submit();
        }
        if (!asyncWriter) {
            static_cast<void>(recorder->Commands());
            recorder->NotePendingWrite(resource.baseAddress + 256, 16);
            Require(recorder->OpenWriteOverlaps(resource.baseAddress + 256, 16), "the main lane's writer is not a write of its open batch");
        }
        std::atomic<bool> finished{false};
        std::exception_ptr failure;
        lock.unlock();
        std::thread flusher([&] {
            try {
                std::lock_guard flusherLock(AgcDriver::GuestMemory::GpuMutex());
                Require(StorageTexture::FlushPending(resource.baseAddress, guestBytes, nullptr, "flush lane wait test"), "the pending image was not flushed");
            } catch (...) {
                failure = std::current_exception();
            }
            finished.store(true);
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(asyncWriter ? 400 : 3000);
        while (!finished.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        outcome.returnedWhileGated = finished.load();
        gate.Release();
        flusher.join();
        if (failure) std::rethrow_exception(failure);
        lock.lock();
        recorder->Sync();
        outcome.stored = StorageTexture::FindPending(resource.baseAddress, guestBytes) == nullptr;
        if (outcome.stored) {
            AgcDriver::GuestMemory::CollectWritesUncached(resource.baseAddress, guestBytes);
            for (std::size_t offset = 0; offset < guestBytes; offset += 4) {
                std::uint32_t word = 0;
                std::memcpy(&word, block.Data() + offset, 4);
                if (word != Cleared) {
                    outcome.stored = false;
                    break;
                }
            }
        }
    }
    recorder->Sync();
    return outcome;
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
        const auto main = RunCase(*device, false);
        Require(main.stored, "the flush over a main-lane writer did not store the image's cleared texels");
        Require(main.returnedWhileGated, "a flush whose only writer sat in the main lane's open batch waited for the async lane's unrelated batch (held by a gate) instead of finishing the main lane through its writer");
        std::puts("storage flush lane wait: a main-lane writer was waited for without the async lane");
        const auto async = RunCase(*device, true);
        Require(async.stored, "the flush over an async-lane writer did not store the image's cleared texels");
        Require(!async.returnedWhileGated, "a flush returned while the async lane's batch that writes the stored range was still held by a gate");
        std::puts("storage flush lane wait: an async-lane writer over the stored range was still waited for");
        device.reset();
        std::puts("storage flush lane wait tests passed");
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
