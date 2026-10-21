#include "prx/libSceAgcDriver/Execution/include/MemoryStats.hpp"
#include "prx/libSceAgcDriver/Execution/include/MemoryStatsDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <atomic>
#include <span>
#include "prx/libc/include/GuestArena.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace AgcDriver::MemoryStats {
namespace {

constexpr const char* CounterNames[] = {
    "dispatch_shader_memory_pages",
    "storage_texture_host_copies",
    "storage_texture_hashes",
};
static_assert(std::size(CounterNames) == static_cast<std::size_t>(Counter::Count));

enum class VkClass : std::uint8_t { Device, Bar, Host, Imported, Other, Count };
constexpr const char* VkClassNames[] = {"vk_device", "vk_bar", "vk_host", "vk_imported", "vk_other"};

struct DeviceEntry {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
    PFN_vkGetDeviceProcAddr real = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties2 budget = nullptr;
    PFN_vkAllocateMemory allocateMemory = nullptr;
    PFN_vkFreeMemory freeMemory = nullptr;
    PFN_vkCreateShaderModule createShaderModule = nullptr;
    PFN_vkDestroyShaderModule destroyShaderModule = nullptr;
    PFN_vkCreateGraphicsPipelines createGraphicsPipelines = nullptr;
    PFN_vkCreateComputePipelines createComputePipelines = nullptr;
    PFN_vkDestroyPipeline destroyPipeline = nullptr;
    PFN_vkGetPipelineCacheData getPipelineCacheData = nullptr;
    PFN_vkCreateImage createImage = nullptr;
    PFN_vkDestroyImage destroyImage = nullptr;
    PFN_vkCreateBuffer createBuffer = nullptr;
    PFN_vkDestroyBuffer destroyBuffer = nullptr;
    PFN_vkBindBufferMemory bindBufferMemory = nullptr;
    PFN_vkGetBufferDeviceAddress getBufferDeviceAddress = nullptr;
    PFN_vkCmdCopyBuffer cmdCopyBuffer = nullptr;
    PFN_vkCmdFillBuffer cmdFillBuffer = nullptr;
    PFN_vkCmdUpdateBuffer cmdUpdateBuffer = nullptr;
    PFN_vkCmdCopyImageToBuffer cmdCopyImageToBuffer = nullptr;
    PFN_vkCmdCopyQueryPoolResults cmdCopyQueryPoolResults = nullptr;
    PFN_vkUpdateDescriptorSets updateDescriptorSets = nullptr;
};

bool FaultAddressesTracked() {
    static const bool tracked = std::getenv("APS5_FAULT_ADDRESSES") != nullptr;
    return tracked;
}

struct AddressRegistry {
    struct Live {
        std::uint64_t buffer;
        VkDeviceSize bytes;
        VkBufferUsageFlags usage;
        Owner owner;
        double madeMs;
    };
    struct Created {
        VkDeviceSize bytes;
        VkBufferUsageFlags usage;
        Owner owner;
        std::uint64_t address;
    };
    struct Destroyed {
        std::uint64_t address;
        VkDeviceSize bytes;
        VkBufferUsageFlags usage;
        Owner owner;
        double madeMs;
        double destroyedMs;
    };
    std::mutex mutex;
    std::unordered_map<std::uint64_t, Created> created;
    std::unordered_map<std::uint64_t, std::pair<VkCommandBuffer, std::uint64_t>> writers;
    std::map<std::uint64_t, Live> live;
    std::vector<Destroyed> destroyed = std::vector<Destroyed>(65536);
    std::size_t destroyedNext = 0;
    std::uint64_t destroyedTotal = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double Now() const { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); }
};

AddressRegistry& Addresses() {
    static auto* registry = new AddressRegistry;
    return *registry;
}

void NoteBufferAddress(std::uint64_t buffer, std::uint64_t address) {
    if (address == 0) return;
    auto& registry = Addresses();
    std::lock_guard lock(registry.mutex);
    const auto found = registry.created.find(buffer);
    if (found == registry.created.end() || found->second.address == address) return;
    found->second.address = address;
    registry.live[address] = {buffer, found->second.bytes, found->second.usage, found->second.owner, registry.Now()};
}


struct State {
    std::array<Slot, static_cast<std::size_t>(VkClass::Count)> vk;
    Slot shaderModules, pipelines, images, buffers;
    std::atomic<std::uint64_t> pipelineStageSpirvBytes{0}, pipelineCreates{0}, shaderModuleCreatedBytes{0};
    std::atomic<std::uint64_t> pipelineCacheDataBytes{0};
    std::mutex devicesMutex;
    std::vector<DeviceEntry> devices;
    std::mutex memoryMutex;
    struct Allocation {
        VkDeviceSize bytes;
        VkClass kind;
        Owner owner;
    };
    std::unordered_map<std::uint64_t, Allocation> allocations;
    std::mutex objectMutex;
    std::unordered_map<std::uint64_t, Owner> imageOwners;
    std::unordered_map<std::uint64_t, Owner> bufferOwners;
    std::mutex shaderMutex;
    std::unordered_map<std::uint64_t, std::size_t> shaderModuleBytes;
    std::unordered_map<const void*, std::size_t> spirvSources;
    std::uint64_t spirvSourceBytes = 0;
    std::mutex reportMutex;
    std::condition_variable wake;
    bool stopping = false;
    bool started = false;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};

State& Get() {
    static auto* state = new State;
    return *state;
}

std::uint64_t HandleKey(auto handle) {
    std::uint64_t key = 0;
    static_assert(sizeof(handle) <= sizeof(key));
    std::memcpy(&key, &handle, sizeof(handle));
    return key;
}

const DeviceEntry* Find(VkDevice device) {
    auto& state = Get();
    std::lock_guard lock(state.devicesMutex);
    for (const auto& entry : state.devices) {
        if (entry.device == device) return &entry;
    }
    return state.devices.empty() ? nullptr : &state.devices.front();
}

void Bump(Slot& slot, std::int64_t bytes, std::int64_t count) {
    slot.bytes.fetch_add(bytes, std::memory_order_relaxed);
    slot.count.fetch_add(count, std::memory_order_relaxed);
}

VkClass Classify(const DeviceEntry& entry, const VkMemoryAllocateInfo& info) {
    for (auto* next = static_cast<const VkBaseInStructure*>(info.pNext); next != nullptr; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT) return VkClass::Imported;
    }
    if (info.memoryTypeIndex >= entry.memory.memoryTypeCount) return VkClass::Other;
    const auto flags = entry.memory.memoryTypes[info.memoryTypeIndex].propertyFlags;
    const bool device = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    const bool host = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    if (device && host) return VkClass::Bar;
    if (device) return VkClass::Device;
    if (host) return VkClass::Host;
    return VkClass::Other;
}

Slot& OwnerMemorySlot(Owner owner, VkClass kind) {
    auto& slots = OwnerAt(owner);
    if (kind == VkClass::Device || kind == VkClass::Bar) return slots.device;
    if (kind == VkClass::Imported) return slots.imported;
    return slots.host;
}

VKAPI_ATTR VkResult VKAPI_CALL TrackedAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* allocator, VkDeviceMemory* memory) {
    const auto* entry = Find(device);
    const auto result = entry->allocateMemory(device, info, allocator, memory);
    if (result == VK_SUCCESS && info != nullptr && memory != nullptr) {
        auto& state = Get();
        const auto kind = Classify(*entry, *info);
        Bump(state.vk[static_cast<std::size_t>(kind)], static_cast<std::int64_t>(info->allocationSize), 1);
        const auto owner = CurrentPoolTier() != Owner::None ? CurrentPoolTier() : CurrentOwner();
        Bump(OwnerMemorySlot(owner, kind), static_cast<std::int64_t>(info->allocationSize), 1);
        std::lock_guard lock(state.memoryMutex);
        state.allocations[HandleKey(*memory)] = {info->allocationSize, kind, owner};
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL TrackedFreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* allocator) {
    const auto* entry = Find(device);
    if (memory != VK_NULL_HANDLE) {
        auto& state = Get();
        std::lock_guard lock(state.memoryMutex);
        if (const auto found = state.allocations.find(HandleKey(memory)); found != state.allocations.end()) {
            Bump(state.vk[static_cast<std::size_t>(found->second.kind)], -static_cast<std::int64_t>(found->second.bytes), -1);
            Bump(OwnerMemorySlot(found->second.owner, found->second.kind), -static_cast<std::int64_t>(found->second.bytes), -1);
            state.allocations.erase(found);
        }
    }
    entry->freeMemory(device, memory, allocator);
}

VKAPI_ATTR VkResult VKAPI_CALL TrackedCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* info, const VkAllocationCallbacks* allocator, VkShaderModule* module) {
    const auto* entry = Find(device);
    const auto result = entry->createShaderModule(device, info, allocator, module);
    if (result == VK_SUCCESS && info != nullptr && module != nullptr) {
        auto& state = Get();
        Bump(state.shaderModules, static_cast<std::int64_t>(info->codeSize), 1);
        state.shaderModuleCreatedBytes.fetch_add(info->codeSize, std::memory_order_relaxed);
        std::lock_guard lock(state.shaderMutex);
        state.shaderModuleBytes[HandleKey(*module)] = info->codeSize;
        auto& known = state.spirvSources[info->pCode];
        state.spirvSourceBytes += info->codeSize - known;
        known = info->codeSize;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL TrackedDestroyShaderModule(VkDevice device, VkShaderModule module, const VkAllocationCallbacks* allocator) {
    const auto* entry = Find(device);
    if (module != VK_NULL_HANDLE) {
        auto& state = Get();
        std::lock_guard lock(state.shaderMutex);
        if (const auto found = state.shaderModuleBytes.find(HandleKey(module)); found != state.shaderModuleBytes.end()) {
            Bump(state.shaderModules, -static_cast<std::int64_t>(found->second), -1);
            state.shaderModuleBytes.erase(found);
        }
    }
    entry->destroyShaderModule(device, module, allocator);
}

void CountStageSpirv(const VkPipelineShaderStageCreateInfo& stage) {
    for (auto* next = static_cast<const VkBaseInStructure*>(stage.pNext); next != nullptr; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO) {
            Get().pipelineStageSpirvBytes.fetch_add(reinterpret_cast<const VkShaderModuleCreateInfo*>(next)->codeSize, std::memory_order_relaxed);
        }
    }
}

void CountPipelines(const VkPipeline* pipelines, std::uint32_t count) {
    std::int64_t created = 0;
    for (std::uint32_t i = 0; pipelines != nullptr && i < count; ++i) created += pipelines[i] != VK_NULL_HANDLE;
    Bump(Get().pipelines, 0, created);
    Get().pipelineCreates.fetch_add(static_cast<std::uint64_t>(created), std::memory_order_relaxed);
}

VKAPI_ATTR VkResult VKAPI_CALL TrackedCreateGraphicsPipelines(VkDevice device, VkPipelineCache cache, std::uint32_t count, const VkGraphicsPipelineCreateInfo* infos, const VkAllocationCallbacks* allocator, VkPipeline* pipelines) {
    const auto* entry = Find(device);
    for (std::uint32_t i = 0; infos != nullptr && i < count; ++i) {
        for (std::uint32_t s = 0; infos[i].pStages != nullptr && s < infos[i].stageCount; ++s) CountStageSpirv(infos[i].pStages[s]);
    }
    const auto result = entry->createGraphicsPipelines(device, cache, count, infos, allocator, pipelines);
    if (result == VK_SUCCESS || result == VK_PIPELINE_COMPILE_REQUIRED) CountPipelines(pipelines, count);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL TrackedCreateComputePipelines(VkDevice device, VkPipelineCache cache, std::uint32_t count, const VkComputePipelineCreateInfo* infos, const VkAllocationCallbacks* allocator, VkPipeline* pipelines) {
    const auto* entry = Find(device);
    for (std::uint32_t i = 0; infos != nullptr && i < count; ++i) CountStageSpirv(infos[i].stage);
    const auto result = entry->createComputePipelines(device, cache, count, infos, allocator, pipelines);
    if (result == VK_SUCCESS || result == VK_PIPELINE_COMPILE_REQUIRED) CountPipelines(pipelines, count);
    return result;
}

VKAPI_ATTR void VKAPI_CALL TrackedDestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* allocator) {
    const auto* entry = Find(device);
    if (pipeline != VK_NULL_HANDLE) Bump(Get().pipelines, 0, -1);
    entry->destroyPipeline(device, pipeline, allocator);
}

VKAPI_ATTR VkResult VKAPI_CALL TrackedGetPipelineCacheData(VkDevice device, VkPipelineCache cache, std::size_t* size, void* data) {
    const auto* entry = Find(device);
    const auto result = entry->getPipelineCacheData(device, cache, size, data);
    if ((result == VK_SUCCESS || result == VK_INCOMPLETE) && size != nullptr) Get().pipelineCacheDataBytes.store(*size, std::memory_order_relaxed);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL TrackedCreateImage(VkDevice device, const VkImageCreateInfo* info, const VkAllocationCallbacks* allocator, VkImage* image) {
    const auto* entry = Find(device);
    const auto result = entry->createImage(device, info, allocator, image);
    if (result == VK_SUCCESS && image != nullptr) {
        Bump(Get().images, 0, 1);
        const auto owner = CurrentOwner();
        Bump(OwnerAt(owner).images, 0, 1);
        std::lock_guard lock(Get().objectMutex);
        Get().imageOwners[HandleKey(*image)] = owner;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL TrackedDestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* allocator) {
    const auto* entry = Find(device);
    if (image != VK_NULL_HANDLE) {
        Bump(Get().images, 0, -1);
        std::lock_guard lock(Get().objectMutex);
        if (const auto found = Get().imageOwners.find(HandleKey(image)); found != Get().imageOwners.end()) {
            Bump(OwnerAt(found->second).images, 0, -1);
            Get().imageOwners.erase(found);
        }
    }
    entry->destroyImage(device, image, allocator);
}

std::atomic<bool> buffersShared{false};
std::array<std::uint32_t, 2> sharedFamilies{};

VKAPI_ATTR VkResult VKAPI_CALL TrackedCreateBuffer(VkDevice device, const VkBufferCreateInfo* info, const VkAllocationCallbacks* allocator, VkBuffer* buffer) {
    const auto* entry = Find(device);
    VkBufferCreateInfo shared{};
    if (info != nullptr && info->sharingMode == VK_SHARING_MODE_EXCLUSIVE && buffersShared.load(std::memory_order_acquire)) {
        shared = *info;
        shared.sharingMode = VK_SHARING_MODE_CONCURRENT;
        shared.queueFamilyIndexCount = 2;
        shared.pQueueFamilyIndices = sharedFamilies.data();
        info = &shared;
    }
    const auto result = entry->createBuffer(device, info, allocator, buffer);
    if (result == VK_SUCCESS && buffer != nullptr) {
        Bump(Get().buffers, 0, 1);
        const auto owner = CurrentOwner();
        Bump(OwnerAt(owner).buffers, 0, 1);
        {
            std::lock_guard lock(Get().objectMutex);
            Get().bufferOwners[HandleKey(*buffer)] = owner;
        }
        if (FaultAddressesTracked() && info != nullptr) {
            auto& registry = Addresses();
            std::lock_guard lock(registry.mutex);
            registry.created[HandleKey(*buffer)] = {info->size, info->usage, owner, 0};
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL TrackedBindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) {
    const auto* entry = Find(device);
    const auto result = entry->bindBufferMemory(device, buffer, memory, offset);
    if (result == VK_SUCCESS && FaultAddressesTracked() && entry->getBufferDeviceAddress != nullptr) {
        bool addressable = false;
        {
            auto& registry = Addresses();
            std::lock_guard lock(registry.mutex);
            const auto found = registry.created.find(HandleKey(buffer));
            addressable = found != registry.created.end() && (found->second.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
        }
        if (addressable) {
            VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            info.buffer = buffer;
            NoteBufferAddress(HandleKey(buffer), entry->getBufferDeviceAddress(device, &info));
        }
    }
    return result;
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL TrackedGetBufferDeviceAddress(VkDevice device, const VkBufferDeviceAddressInfo* info) {
    const auto* entry = Find(device);
    const auto address = entry->getBufferDeviceAddress(device, info);
    if (FaultAddressesTracked() && info != nullptr) NoteBufferAddress(HandleKey(info->buffer), address);
    return address;
}

void CheckDestroyedWriter(VkBuffer buffer);

VKAPI_ATTR void VKAPI_CALL TrackedDestroyBuffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* allocator) {
    const auto* entry = Find(device);
    if (buffer != VK_NULL_HANDLE && FaultAddressesTracked()) CheckDestroyedWriter(buffer);
    if (buffer != VK_NULL_HANDLE) {
        Bump(Get().buffers, 0, -1);
        std::lock_guard lock(Get().objectMutex);
        if (const auto found = Get().bufferOwners.find(HandleKey(buffer)); found != Get().bufferOwners.end()) {
            Bump(OwnerAt(found->second).buffers, 0, -1);
            Get().bufferOwners.erase(found);
        }
    }
    if (buffer != VK_NULL_HANDLE && FaultAddressesTracked()) {
        auto& registry = Addresses();
        std::lock_guard lock(registry.mutex);
        if (const auto found = registry.created.find(HandleKey(buffer)); found != registry.created.end()) {
            if (found->second.address != 0) {
                if (const auto at = registry.live.find(found->second.address); at != registry.live.end() && at->second.buffer == HandleKey(buffer)) {
                    registry.destroyed[registry.destroyedNext] = {found->second.address, at->second.bytes, at->second.usage, at->second.owner, at->second.madeMs, registry.Now()};
                    registry.destroyedNext = (registry.destroyedNext + 1u) % registry.destroyed.size();
                    ++registry.destroyedTotal;
                    registry.live.erase(at);
                }
            }
            registry.created.erase(found);
        }
    }
    entry->destroyBuffer(device, buffer, allocator);
}

std::atomic<std::uint64_t> writeChecks{0}, writeCheckFailures{0}, nullDescriptors{0};

void ReportWriteCheck(const char* call, const char* problem, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize bytes, VkDeviceSize known) {
    const auto count = writeCheckFailures.fetch_add(1, std::memory_order_relaxed);
    if (count >= 64u) return;
    unsigned long long frames[8] = {};
    const auto captured = GuestMemory::CaptureCallerOffsets(std::span<unsigned long long>(frames, 8), 1);
    std::string stack;
    char part[32];
    for (std::size_t i = 0; i < captured; ++i) {
        std::snprintf(part, sizeof(part), " +0x%llx", frames[i]);
        stack += part;
    }
    const auto packet = GuestMemory::CurrentPacket();
    std::fprintf(stderr, "[write-check] %s: %s (buffer 0x%llx, offset 0x%llx, bytes 0x%llx, buffer size 0x%llx); program 0x%llx, packet 0x%x queue 0x%x; libSceAgcDriver.prx%s\n", call, problem,
                 static_cast<unsigned long long>(HandleKey(buffer)), static_cast<unsigned long long>(offset), static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(known),
                 static_cast<unsigned long long>(Graphics::Recorder::CurrentWriteOrigin()), packet.opcode, packet.queue, stack.c_str());
}

std::atomic<std::uint64_t> destroyedWhilePending{0};

void CheckDestroyedWriter(VkBuffer buffer) {
    std::pair<VkCommandBuffer, std::uint64_t> writer{VK_NULL_HANDLE, 0};
    {
        auto& registry = Addresses();
        std::lock_guard lock(registry.mutex);
        const auto found = registry.writers.find(HandleKey(buffer));
        if (found == registry.writers.end()) return;
        writer = found->second;
        registry.writers.erase(found);
    }
    int lane = 0;
    if (!Graphics::Recorder::CommandBufferPending(writer.first, writer.second, &lane)) return;
    const auto count = destroyedWhilePending.fetch_add(1, std::memory_order_relaxed);
    if (count >= 64u) return;
    unsigned long long frames[12] = {};
    const auto captured = GuestMemory::CaptureCallerOffsets(std::span<unsigned long long>(frames, 12), 1);
    std::string stack;
    char part[32];
    for (std::size_t i = 0; i < captured; ++i) {
        std::snprintf(part, sizeof(part), " +0x%llx", frames[i]);
        stack += part;
    }
    const auto packet = GuestMemory::CurrentPacket();
    std::fprintf(stderr, "[destroy-check] vkDestroyBuffer 0x%llx while a recorded batch of lane %d that writes it is open or running (command buffer 0x%llx); packet 0x%x queue 0x%x, thread lane %d; libSceAgcDriver.prx%s\n", static_cast<unsigned long long>(HandleKey(buffer)), lane, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(writer.first)), packet.opcode, packet.queue, Graphics::Recorder::ThreadLane(), stack.c_str());
}

void CheckBufferWrite(const char* call, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize bytes, VkCommandBuffer commands = VK_NULL_HANDLE) {
    writeChecks.fetch_add(1, std::memory_order_relaxed);
    if (commands != VK_NULL_HANDLE && buffer != VK_NULL_HANDLE && Graphics::Recorder::PendingCommandsTracked()) {
        if (const auto generation = Graphics::Recorder::CommandBufferGeneration(commands); generation != 0) {
            auto& registry = Addresses();
            std::lock_guard lock(registry.mutex);
            registry.writers[HandleKey(buffer)] = {commands, generation};
        }
    }
    if (buffer == VK_NULL_HANDLE) {
        ReportWriteCheck(call, "VK_NULL_HANDLE destination", buffer, offset, bytes, 0);
        return;
    }
    VkDeviceSize known = 0;
    bool found = false;
    {
        auto& registry = Addresses();
        std::lock_guard lock(registry.mutex);
        const auto at = registry.created.find(HandleKey(buffer));
        if (at != registry.created.end()) {
            found = true;
            known = at->second.bytes;
        }
    }
    if (!found) ReportWriteCheck(call, "destination is not a live buffer", buffer, offset, bytes, 0);
    else if (bytes != VK_WHOLE_SIZE && (offset > known || bytes > known - offset)) ReportWriteCheck(call, "write past the buffer's end", buffer, offset, bytes, known);
}

VKAPI_ATTR void VKAPI_CALL TrackedCmdCopyBuffer(VkCommandBuffer commands, VkBuffer source, VkBuffer destination, std::uint32_t count, const VkBufferCopy* regions) {
    for (std::uint32_t i = 0; i < count; ++i) CheckBufferWrite("vkCmdCopyBuffer", destination, regions[i].dstOffset, regions[i].size, commands);
    Get().devices.back().cmdCopyBuffer(commands, source, destination, count, regions);
}

VKAPI_ATTR void VKAPI_CALL TrackedCmdFillBuffer(VkCommandBuffer commands, VkBuffer destination, VkDeviceSize offset, VkDeviceSize bytes, std::uint32_t data) {
    CheckBufferWrite("vkCmdFillBuffer", destination, offset, bytes, commands);
    Get().devices.back().cmdFillBuffer(commands, destination, offset, bytes, data);
}

VKAPI_ATTR void VKAPI_CALL TrackedCmdUpdateBuffer(VkCommandBuffer commands, VkBuffer destination, VkDeviceSize offset, VkDeviceSize bytes, const void* data) {
    CheckBufferWrite("vkCmdUpdateBuffer", destination, offset, bytes, commands);
    Get().devices.back().cmdUpdateBuffer(commands, destination, offset, bytes, data);
}

VKAPI_ATTR void VKAPI_CALL TrackedCmdCopyImageToBuffer(VkCommandBuffer commands, VkImage image, VkImageLayout layout, VkBuffer destination, std::uint32_t count, const VkBufferImageCopy* regions) {
    for (std::uint32_t i = 0; i < count; ++i) CheckBufferWrite("vkCmdCopyImageToBuffer", destination, regions[i].bufferOffset, 1, commands);
    Get().devices.back().cmdCopyImageToBuffer(commands, image, layout, destination, count, regions);
}

VKAPI_ATTR void VKAPI_CALL TrackedCmdCopyQueryPoolResults(VkCommandBuffer commands, VkQueryPool pool, std::uint32_t first, std::uint32_t count, VkBuffer destination, VkDeviceSize offset, VkDeviceSize stride, VkQueryResultFlags flags) {
    CheckBufferWrite("vkCmdCopyQueryPoolResults", destination, offset, count == 0 ? 0 : stride * (count - 1u) + ((flags & VK_QUERY_RESULT_64_BIT) != 0 ? 8u : 4u), commands);
    Get().devices.back().cmdCopyQueryPoolResults(commands, pool, first, count, destination, offset, stride, flags);
}

VKAPI_ATTR void VKAPI_CALL TrackedUpdateDescriptorSets(VkDevice device, std::uint32_t writeCount, const VkWriteDescriptorSet* writes, std::uint32_t copyCount, const VkCopyDescriptorSet* copies) {
    for (std::uint32_t i = 0; i < writeCount; ++i) {
        const auto& write = writes[i];
        const bool storage = write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        const bool uniform = write.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || write.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        if ((!storage && !uniform) || write.pBufferInfo == nullptr) continue;
        for (std::uint32_t d = 0; d < write.descriptorCount; ++d) {
            const auto& info = write.pBufferInfo[d];
            if (info.buffer == VK_NULL_HANDLE) {
                nullDescriptors.fetch_add(1, std::memory_order_relaxed);
                if (info.range != VK_WHOLE_SIZE || info.offset != 0) ReportWriteCheck(storage ? "storage descriptor" : "uniform descriptor", "VK_NULL_HANDLE with a range", info.buffer, info.offset, info.range, 0);
                continue;
            }
            CheckBufferWrite(storage ? "storage descriptor" : "uniform descriptor", info.buffer, info.offset, info.range);
        }
    }
    Get().devices.back().updateDescriptorSets(device, writeCount, writes, copyCount, copies);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL TrackedDeviceProc(VkDevice device, const char* name) {
    const auto* entry = Find(device);
    if (entry == nullptr || name == nullptr) return nullptr;
    const auto real = entry->real(device, name);
    if (real == nullptr) return real;
    const std::string_view view(name);
    if (view == "vkAllocateMemory") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedAllocateMemory);
    if (view == "vkFreeMemory") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedFreeMemory);
    if (view == "vkCreateShaderModule") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedCreateShaderModule);
    if (view == "vkDestroyShaderModule") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedDestroyShaderModule);
    if (view == "vkCreateGraphicsPipelines") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedCreateGraphicsPipelines);
    if (view == "vkCreateComputePipelines") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedCreateComputePipelines);
    if (view == "vkDestroyPipeline") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedDestroyPipeline);
    if (view == "vkGetPipelineCacheData") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedGetPipelineCacheData);
    if (view == "vkCreateImage") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedCreateImage);
    if (view == "vkDestroyImage") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedDestroyImage);
    if (view == "vkCreateBuffer") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedCreateBuffer);
    if (view == "vkDestroyBuffer") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedDestroyBuffer);
    if (FaultAddressesTracked() && entry->bindBufferMemory != nullptr && view == "vkBindBufferMemory") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedBindBufferMemory);
    if (FaultAddressesTracked() && entry->cmdCopyBuffer != nullptr && view == "vkCmdCopyBuffer") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedCmdCopyBuffer);
    if (FaultAddressesTracked() && entry->cmdFillBuffer != nullptr && view == "vkCmdFillBuffer") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedCmdFillBuffer);
    if (FaultAddressesTracked() && entry->cmdUpdateBuffer != nullptr && view == "vkCmdUpdateBuffer") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedCmdUpdateBuffer);
    if (FaultAddressesTracked() && entry->cmdCopyImageToBuffer != nullptr && view == "vkCmdCopyImageToBuffer") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedCmdCopyImageToBuffer);
    if (FaultAddressesTracked() && entry->cmdCopyQueryPoolResults != nullptr && view == "vkCmdCopyQueryPoolResults") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedCmdCopyQueryPoolResults);
    if (FaultAddressesTracked() && entry->updateDescriptorSets != nullptr && view == "vkUpdateDescriptorSets") return reinterpret_cast<PFN_vkVoidFunction>(&TrackedUpdateDescriptorSets);
    if (FaultAddressesTracked() && entry->getBufferDeviceAddress != nullptr && (view == "vkGetBufferDeviceAddress" || view == "vkGetBufferDeviceAddressKHR")) return reinterpret_cast<PFN_vkVoidFunction>(&TrackedGetBufferDeviceAddress);
    return real;
}

constexpr double MiB = 1024.0 * 1024.0;

void Append(std::string& line, const char* format, auto... args) {
    char buffer[256];
    const int written = std::snprintf(buffer, sizeof(buffer), format, args...);
    if (written > 0) line.append(buffer, static_cast<std::size_t>(std::min<int>(written, static_cast<int>(sizeof(buffer) - 1))));
}

#ifdef _WIN32
unsigned CrtCensusEvery() {
    static const unsigned every = [] {
        const char* value = std::getenv("APS5_MEM_STATS_CRT_EVERY");
        if (value == nullptr) return 30u;
        char* end = nullptr;
        const auto parsed = std::strtoul(value, &end, 10);
        if (end == value || *end != '\0' || parsed > 1000ul) throw std::runtime_error(std::string("APS5_MEM_STATS_CRT_EVERY=") + value + ": expected a report count from 0 (off) to 1000");
        return static_cast<unsigned>(parsed);
    }();
    return every;
}

void AppendProcess(std::string& line) {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters))) {
        Append(line, " process_private=%.0fMB process_private_peak=%.0fMB ws=%.0fMB", counters.PrivateUsage / MiB, counters.PeakPagefileUsage / MiB, counters.WorkingSetSize / MiB);
    }
    std::vector<HANDLE> heaps(64);
    DWORD count = GetProcessHeaps(static_cast<DWORD>(heaps.size()), heaps.data());
    if (count > heaps.size()) {
        heaps.resize(count + 16);
        count = GetProcessHeaps(static_cast<DWORD>(heaps.size()), heaps.data());
    }
    heaps.resize(std::min<std::size_t>(count, heaps.size()));
    const HANDLE crt = GetProcessHeap();
    std::uint64_t otherAllocated = 0, otherCommitted = 0, largestOther = 0;
    for (const auto heap : heaps) {
        if (heap == crt) continue;
        HEAP_SUMMARY summary{};
        summary.cb = sizeof(summary);
        if (!HeapSummary(heap, 0, &summary)) continue;
        otherAllocated += summary.cbAllocated;
        otherCommitted += summary.cbCommitted;
        largestOther = std::max<std::uint64_t>(largestOther, summary.cbCommitted);
    }
    Append(line, " other_heaps=%.0fMB(%zu) other_heaps_committed=%.0fMB largest_other_heap=%.0fMB", otherAllocated / MiB, heaps.size() > 0 ? heaps.size() - 1 : 0, otherCommitted / MiB, largestOther / MiB);
    static unsigned census = 0;
    if (CrtCensusEvery() != 0u && census++ % CrtCensusEvery() == 0u && HeapLock(crt)) {
        const auto heapStarted = std::chrono::steady_clock::now();
        std::unordered_map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>> sizes;
        std::unordered_map<std::uint64_t, std::uint64_t> smallSizes;
        std::uint64_t largeBytes = 0, largeBlocks = 0, smallBytes = 0, smallBlocks = 0;
        PROCESS_HEAP_ENTRY entry{};
        while (HeapWalk(crt, &entry)) {
            if ((entry.wFlags & PROCESS_HEAP_ENTRY_BUSY) == 0) continue;
            if (entry.cbData < (1ull << 20u)) {
                smallBytes += entry.cbData;
                ++smallBlocks;
                ++smallSizes[entry.cbData];
                continue;
            }
            largeBytes += entry.cbData;
            ++largeBlocks;
            auto& bucket = sizes[(entry.cbData + 0xffffu) >> 16u];
            bucket.first += entry.cbData;
            ++bucket.second;
        }
        HeapUnlock(crt);
        std::vector<std::pair<std::uint64_t, std::uint64_t>> top;
        for (const auto& [granules, bucket] : sizes) top.emplace_back(bucket.first, granules);
        const auto shown = std::min<std::size_t>(top.size(), 8);
        std::partial_sort(top.begin(), top.begin() + static_cast<std::ptrdiff_t>(shown), top.end(), std::greater<>());
        Append(line, " crt_small=%.0fMB(%llu) crt_large=%.0fMB(%llu) crt_large_top=", smallBytes / MiB, static_cast<unsigned long long>(smallBlocks), largeBytes / MiB, static_cast<unsigned long long>(largeBlocks));
        for (std::size_t i = 0; i < shown; ++i) Append(line, "%s%.2fMBx%llu", i == 0 ? "" : ",", static_cast<double>(top[i].second << 16u) / MiB, static_cast<unsigned long long>(sizes[top[i].second].second));
        std::vector<std::pair<std::uint64_t, std::uint64_t>> smallTop;
        for (const auto& [bytes, count] : smallSizes) smallTop.emplace_back(bytes * count, bytes);
        const auto smallShown = std::min<std::size_t>(smallTop.size(), 8);
        std::partial_sort(smallTop.begin(), smallTop.begin() + static_cast<std::ptrdiff_t>(smallShown), smallTop.end(), std::greater<>());
        line += " crt_small_top=";
        for (std::size_t i = 0; i < smallShown; ++i) Append(line, "%s%lluBx%llu", i == 0 ? "" : ",", static_cast<unsigned long long>(smallTop[i].second), static_cast<unsigned long long>(smallSizes[smallTop[i].second]));
        Append(line, " crt_ms=%lld", static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - heapStarted).count()));
    }
    if (GuestArena::GuestArenaAvailable_nid_postfix()) {
        std::uint64_t savedBytes = 0, savedBlocks = 0;
        GuestArena::GuestArenaSavedMemory_nid_postfix(&savedBytes, &savedBlocks);
        Append(line, " guest_saved=%.0fMB(%llu)", savedBytes / MiB, static_cast<unsigned long long>(savedBlocks));
    }

    const auto started = std::chrono::steady_clock::now();
    const auto deadline = started + std::chrono::milliseconds(1500);
    std::uintptr_t arenaBase = 0;
    std::size_t arenaBytes = 0;
    if (GuestArena::GuestArenaAvailable_nid_postfix()) GuestArena::GuestArenaRange_nid_postfix(&arenaBase, &arenaBytes);
    const auto inArena = [&](const void* address) {
        const auto value = reinterpret_cast<std::uintptr_t>(address);
        return value >= arenaBase && value - arenaBase < arenaBytes;
    };
    std::uint64_t privateCommitted = 0, mappedCommitted = 0, imageCommitted = 0, regions = 0, privateGuest = 0, mappedGuest = 0, hostWriteCombined = 0;
    std::unordered_map<std::uintptr_t, std::uint64_t> privateByAllocation, hostByAllocation;
    bool partial = false;
    MEMORY_BASIC_INFORMATION info;
    for (auto* address = static_cast<const char*>(nullptr); VirtualQuery(address, &info, sizeof(info)) == sizeof(info);) {
        if (info.State == MEM_COMMIT) {
            if (info.Type == MEM_PRIVATE) {
                privateCommitted += info.RegionSize;
                if (inArena(info.BaseAddress)) privateGuest += info.RegionSize;
                else {
                    hostByAllocation[reinterpret_cast<std::uintptr_t>(info.AllocationBase)] += info.RegionSize;
                    if ((info.Protect & PAGE_WRITECOMBINE) != 0) hostWriteCombined += info.RegionSize;
                }
                privateByAllocation[reinterpret_cast<std::uintptr_t>(info.AllocationBase)] += info.RegionSize;
            } else if (info.Type == MEM_MAPPED) {
                mappedCommitted += info.RegionSize;
                if (inArena(info.BaseAddress)) mappedGuest += info.RegionSize;
            } else if (info.Type == MEM_IMAGE) {
                imageCommitted += info.RegionSize;
            }
        }
        const auto* next = static_cast<const char*>(info.BaseAddress) + info.RegionSize;
        if (next <= address) break;
        address = next;
        if ((++regions & 1023u) == 0 && std::chrono::steady_clock::now() > deadline) {
            partial = true;
            break;
        }
    }
    std::vector<std::pair<std::uint64_t, std::uintptr_t>> largest;
    std::uint64_t over64 = 0, under1 = 0;
    std::size_t over64Count = 0;
    largest.reserve(privateByAllocation.size());
    for (const auto& [base, bytes] : privateByAllocation) {
        largest.emplace_back(bytes, base);
        if (bytes >= (64ull << 20u)) { over64 += bytes; ++over64Count; }
        if (bytes < (1ull << 20u)) under1 += bytes;
    }
    const auto top = std::min<std::size_t>(largest.size(), 4);
    std::partial_sort(largest.begin(), largest.begin() + static_cast<std::ptrdiff_t>(top), largest.end(), std::greater<>());
    Append(line, " vq_private=%.0fMB vq_mapped=%.0fMB vq_image=%.0fMB vq_private_allocs=%zu vq_private_ge64MB=%.0fMB(%zu) vq_private_lt1MB=%.0fMB",
           privateCommitted / MiB, mappedCommitted / MiB, imageCommitted / MiB, privateByAllocation.size(), over64 / MiB, over64Count, under1 / MiB);
    Append(line, " vq_private_guest_arena=%.0fMB vq_private_host=%.0fMB vq_mapped_guest_arena=%.0fMB", privateGuest / MiB, (privateCommitted - privateGuest) / MiB, mappedGuest / MiB);
    std::uint64_t hostSmall = 0, hostMid = 0, hostLarge = 0;
    std::size_t hostSmallCount = 0, hostMidCount = 0, hostLargeCount = 0;
    for (const auto& [base, bytes] : hostByAllocation) {
        if (bytes < (1ull << 20u)) {
            hostSmall += bytes;
            ++hostSmallCount;
        } else if (bytes < (64ull << 20u)) {
            hostMid += bytes;
            ++hostMidCount;
        } else {
            hostLarge += bytes;
            ++hostLargeCount;
        }
    }
    Append(line, " vq_host_lt1MB=%.0fMB(%zu) vq_host_1to64MB=%.0fMB(%zu) vq_host_ge64MB=%.0fMB(%zu) vq_host_write_combined=%.0fMB", hostSmall / MiB, hostSmallCount, hostMid / MiB, hostMidCount, hostLarge / MiB, hostLargeCount, hostWriteCombined / MiB);
    line += " vq_top=";
    for (std::size_t i = 0; i < top; ++i) Append(line, "%s0x%llx:%.0fMB", i == 0 ? "" : ",", static_cast<unsigned long long>(largest[i].second), largest[i].first / MiB);
    Append(line, " vq_ms=%lld%s", static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count()), partial ? " vq_partial=1" : "");
}
#else
void AppendProcess(std::string&) {}
#endif

void AppendVulkan(std::string& line) {
    auto& state = Get();
    for (std::size_t i = 0; i < state.vk.size(); ++i) {
        AppendMiB(line, VkClassNames[i], static_cast<std::uint64_t>(std::max<std::int64_t>(0, state.vk[i].bytes.load())), static_cast<std::uint64_t>(std::max<std::int64_t>(0, state.vk[i].count.load())));
    }
    AppendMiB(line, "vk_shader_modules", static_cast<std::uint64_t>(std::max<std::int64_t>(0, state.shaderModules.bytes.load())), static_cast<std::uint64_t>(std::max<std::int64_t>(0, state.shaderModules.count.load())));
    {
        std::lock_guard lock(state.shaderMutex);
        AppendMiB(line, "spirv_given_to_vk_unique", state.spirvSourceBytes, state.spirvSources.size());
    }
    Append(line, " vk_pipelines=%lld vk_pipelines_created=%llu vk_pipeline_spirv_in=%.0fMB vk_shader_module_spirv_in=%.0fMB vk_pipeline_cache_data=%.0fMB vk_images=%lld vk_buffers=%lld",
           static_cast<long long>(state.pipelines.count.load()), static_cast<unsigned long long>(state.pipelineCreates.load()), state.pipelineStageSpirvBytes.load() / MiB,
           state.shaderModuleCreatedBytes.load() / MiB, state.pipelineCacheDataBytes.load() / MiB, static_cast<long long>(state.images.count.load()), static_cast<long long>(state.buffers.count.load()));
    DeviceEntry entry;
    {
        std::lock_guard lock(state.devicesMutex);
        if (state.devices.empty()) return;
        entry = state.devices.front();
    }
    if (entry.budget == nullptr) return;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    properties.pNext = &reported;
    entry.budget(entry.physical, &properties);
    for (std::uint32_t heap = 0; heap < properties.memoryProperties.memoryHeapCount; ++heap) {
        const bool local = (properties.memoryProperties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        Append(line, " vk_heap%u_%s_usage=%.0fMB/%.0fMB", heap, local ? "local" : "sys", reported.heapUsage[heap] / MiB, reported.heapBudget[heap] / MiB);
    }
}

std::uint64_t Live(const std::atomic<std::int64_t>& value) {
    return static_cast<std::uint64_t>(std::max<std::int64_t>(0, value.load(std::memory_order_relaxed)));
}

// Per owner, only the non-zero parts: devmem_<owner> (DEVICE_LOCAL allocations charged to it, MB and
// allocation count), hostmem_/importmem_<owner>, img_/buf_<owner> (live VkImages/VkBuffers made under
// it) and pooled_<owner> / pooledhost_<owner> (live pooled buffer objects it asked for, capacity MB and
// count). vk_device_unattributed is the device memory made outside every owner scope.
void AppendOwners(std::string& line) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(Owner::Count); ++i) {
        const auto& slots = OwnerAt(static_cast<Owner>(i));
        const auto name = std::string(OwnerNames[i]);
        if (i == 0) {
            AppendMiB(line, "vk_device_unattributed", Live(slots.device.bytes), Live(slots.device.count));
            AppendCount(line, "vk_images_unattributed", Live(slots.images.count));
            AppendCount(line, "vk_buffers_unattributed", Live(slots.buffers.count));
            if (Live(slots.host.count) != 0) AppendMiB(line, "vk_host_unattributed", Live(slots.host.bytes), Live(slots.host.count));
            if (Live(slots.imported.count) != 0) AppendMiB(line, "vk_imported_unattributed", Live(slots.imported.bytes), Live(slots.imported.count));
            if (Live(slots.liveDevice.count) != 0) AppendMiB(line, "pooled_unattributed", Live(slots.liveDevice.bytes), Live(slots.liveDevice.count));
            if (Live(slots.liveHost.count) != 0) AppendMiB(line, "pooledhost_unattributed", Live(slots.liveHost.bytes), Live(slots.liveHost.count));
            continue;
        }
        if (Live(slots.device.count) != 0) AppendMiB(line, ("devmem_" + name).c_str(), Live(slots.device.bytes), Live(slots.device.count));
        if (Live(slots.host.count) != 0) AppendMiB(line, ("hostmem_" + name).c_str(), Live(slots.host.bytes), Live(slots.host.count));
        if (Live(slots.imported.count) != 0) AppendMiB(line, ("importmem_" + name).c_str(), Live(slots.imported.bytes), Live(slots.imported.count));
        if (Live(slots.images.count) != 0) AppendCount(line, ("img_" + name).c_str(), Live(slots.images.count));
        if (Live(slots.buffers.count) != 0) AppendCount(line, ("buf_" + name).c_str(), Live(slots.buffers.count));
        if (Live(slots.liveDevice.count) != 0) AppendMiB(line, ("pooled_" + name).c_str(), Live(slots.liveDevice.bytes), Live(slots.liveDevice.count));
        if (Live(slots.liveHost.count) != 0) AppendMiB(line, ("pooledhost_" + name).c_str(), Live(slots.liveHost.bytes), Live(slots.liveHost.count));
    }
}

void Report() {
    auto& state = Get();
    std::string line;
    line.reserve(2048);
    Append(line, "[mem] t=%.0f", std::chrono::duration<double>(std::chrono::steady_clock::now() - state.start).count());
    AppendProcess(line);
    AppendVulkan(line);
    AppendOwners(line);
    for (std::size_t i = 0; i < static_cast<std::size_t>(Counter::Count); ++i) {
        const auto& slot = At(static_cast<Counter>(i));
        AppendMiB(line, CounterNames[i], static_cast<std::uint64_t>(std::max<std::int64_t>(0, slot.bytes.load())), static_cast<std::uint64_t>(std::max<std::int64_t>(0, slot.count.load())));
    }
    const auto section = [&](void (*append)(std::string&), const char* name) {
        try {
            append(line);
        } catch (...) {
            Append(line, " %s_failed=1", name);
        }
    };
    section(&DriverDetail::Driver::AppendMemoryStats, "driver");
    section(&Graphics::AppendTextureMemoryStats, "textures");
    section(&Graphics::AppendGuestBufferMemoryStats, "guest_buffers");
    line += '\n';
    std::fwrite(line.data(), 1, line.size(), stderr);
    std::fflush(stderr);
}

}

std::vector<std::string> DescribeDeviceAddress(std::uint64_t address, std::uint64_t precision) {
    std::vector<std::string> lines;
    if (!Enabled() || !FaultAddressesTracked()) return lines;
    auto& registry = Addresses();
    std::unique_lock lock(registry.mutex, std::defer_lock);
    for (int attempt = 0; attempt < 400 && !lock.try_lock(); ++attempt) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (!lock.owns_lock()) {
        lines.emplace_back("address registry busy");
        return lines;
    }
    const auto now = registry.Now();
    const auto span = precision == 0 ? std::uint64_t{1} : precision;
    const auto low = address;
    const auto high = address + span;
    char text[320];
    std::size_t shown = 0;
    auto first = registry.live.upper_bound(low);
    if (first != registry.live.begin()) --first;
    for (auto at = first; at != registry.live.end() && at->first < high && shown < 8; ++at) {
        if (at->first + at->second.bytes <= low) continue;
        std::snprintf(text, sizeof(text), "live buffer 0x%llx+0x%llx usage 0x%x owner %s, made %.0f ms ago", static_cast<unsigned long long>(at->first), static_cast<unsigned long long>(at->second.bytes), static_cast<unsigned>(at->second.usage), OwnerNames[static_cast<std::size_t>(at->second.owner)], now - at->second.madeMs);
        lines.emplace_back(text);
        ++shown;
    }
    if (shown == 0) {
        auto below = registry.live.lower_bound(low);
        if (below != registry.live.begin()) {
            --below;
            std::snprintf(text, sizeof(text), "no live buffer; nearest below 0x%llx+0x%llx owner %s", static_cast<unsigned long long>(below->first), static_cast<unsigned long long>(below->second.bytes), OwnerNames[static_cast<std::size_t>(below->second.owner)]);
            lines.emplace_back(text);
        }
        const auto above = registry.live.lower_bound(low);
        if (above != registry.live.end()) {
            std::snprintf(text, sizeof(text), "nearest live above 0x%llx+0x%llx owner %s", static_cast<unsigned long long>(above->first), static_cast<unsigned long long>(above->second.bytes), OwnerNames[static_cast<std::size_t>(above->second.owner)]);
            lines.emplace_back(text);
        }
    }
    const auto count = std::min<std::uint64_t>(registry.destroyedTotal, registry.destroyed.size());
    std::size_t recent = 0;
    for (std::uint64_t back = 1; back <= count && recent < 8; ++back) {
        const auto& gone = registry.destroyed[(registry.destroyedNext + registry.destroyed.size() - back) % registry.destroyed.size()];
        if (gone.address >= high || gone.address + gone.bytes <= low) continue;
        std::snprintf(text, sizeof(text), "destroyed buffer 0x%llx+0x%llx usage 0x%x owner %s, destroyed %.0f ms ago (lived %.0f ms)", static_cast<unsigned long long>(gone.address), static_cast<unsigned long long>(gone.bytes), static_cast<unsigned>(gone.usage), OwnerNames[static_cast<std::size_t>(gone.owner)], now - gone.destroyedMs, gone.destroyedMs - gone.madeMs);
        lines.emplace_back(text);
        ++recent;
    }
    std::uint64_t lowest = 0, highest = 0;
    if (!registry.live.empty()) {
        lowest = registry.live.begin()->first;
        highest = std::prev(registry.live.end())->first + std::prev(registry.live.end())->second.bytes;
    }
    std::snprintf(text, sizeof(text), "registry: %zu live addressed buffers in 0x%llx..0x%llx, %llu destroyed (last %llu kept); write checks %llu, failed %llu, null descriptors %llu", registry.live.size(), static_cast<unsigned long long>(lowest), static_cast<unsigned long long>(highest), static_cast<unsigned long long>(registry.destroyedTotal), static_cast<unsigned long long>(count), static_cast<unsigned long long>(writeChecks.load()), static_cast<unsigned long long>(writeCheckFailures.load()), static_cast<unsigned long long>(nullDescriptors.load()));
    lines.emplace_back(text);
    return lines;
}

void ShareBuffersAcross(std::uint32_t first, std::uint32_t second) {
    sharedFamilies = {first, second};
    buffersShared.store(true, std::memory_order_release);
}

namespace {

PFN_vkGetDeviceProcAddr sharedRealProc = nullptr;
PFN_vkCreateBuffer sharedRealCreateBuffer = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL SharedCreateBuffer(VkDevice device, const VkBufferCreateInfo* info, const VkAllocationCallbacks* allocator, VkBuffer* buffer) {
    VkBufferCreateInfo shared{};
    if (info != nullptr && info->sharingMode == VK_SHARING_MODE_EXCLUSIVE && buffersShared.load(std::memory_order_acquire)) {
        shared = *info;
        shared.sharingMode = VK_SHARING_MODE_CONCURRENT;
        shared.queueFamilyIndexCount = 2;
        shared.pQueueFamilyIndices = sharedFamilies.data();
        info = &shared;
    }
    return sharedRealCreateBuffer(device, info, allocator, buffer);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL SharedDeviceProc(VkDevice device, const char* name) {
    if (name != nullptr && std::string_view(name) == "vkCreateBuffer") return reinterpret_cast<PFN_vkVoidFunction>(&SharedCreateBuffer);
    return sharedRealProc(device, name);
}

}

PFN_vkGetDeviceProcAddr ShareBuffersProc(VkDevice device, PFN_vkGetDeviceProcAddr real, std::uint32_t first, std::uint32_t second) {
    sharedRealProc = real;
    sharedRealCreateBuffer = reinterpret_cast<PFN_vkCreateBuffer>(real(device, "vkCreateBuffer"));
    ShareBuffersAcross(first, second);
    return sharedRealCreateBuffer != nullptr ? &SharedDeviceProc : real;
}

PFN_vkGetDeviceProcAddr TrackDevice(VkDevice device, VkPhysicalDevice physical, const VkPhysicalDeviceMemoryProperties& memory,
                                    PFN_vkGetDeviceProcAddr real, PFN_vkGetPhysicalDeviceMemoryProperties2 budget) {
    if (!Enabled() || real == nullptr || device == VK_NULL_HANDLE) return real;
    DeviceEntry entry;
    entry.device = device;
    entry.physical = physical;
    entry.memory = memory;
    entry.real = real;
    entry.budget = budget;
    entry.allocateMemory = reinterpret_cast<PFN_vkAllocateMemory>(real(device, "vkAllocateMemory"));
    entry.freeMemory = reinterpret_cast<PFN_vkFreeMemory>(real(device, "vkFreeMemory"));
    entry.createShaderModule = reinterpret_cast<PFN_vkCreateShaderModule>(real(device, "vkCreateShaderModule"));
    entry.destroyShaderModule = reinterpret_cast<PFN_vkDestroyShaderModule>(real(device, "vkDestroyShaderModule"));
    entry.createGraphicsPipelines = reinterpret_cast<PFN_vkCreateGraphicsPipelines>(real(device, "vkCreateGraphicsPipelines"));
    entry.createComputePipelines = reinterpret_cast<PFN_vkCreateComputePipelines>(real(device, "vkCreateComputePipelines"));
    entry.destroyPipeline = reinterpret_cast<PFN_vkDestroyPipeline>(real(device, "vkDestroyPipeline"));
    entry.getPipelineCacheData = reinterpret_cast<PFN_vkGetPipelineCacheData>(real(device, "vkGetPipelineCacheData"));
    entry.createImage = reinterpret_cast<PFN_vkCreateImage>(real(device, "vkCreateImage"));
    entry.destroyImage = reinterpret_cast<PFN_vkDestroyImage>(real(device, "vkDestroyImage"));
    entry.createBuffer = reinterpret_cast<PFN_vkCreateBuffer>(real(device, "vkCreateBuffer"));
    entry.destroyBuffer = reinterpret_cast<PFN_vkDestroyBuffer>(real(device, "vkDestroyBuffer"));
    entry.bindBufferMemory = reinterpret_cast<PFN_vkBindBufferMemory>(real(device, "vkBindBufferMemory"));
    entry.getBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(real(device, "vkGetBufferDeviceAddress"));
    if (entry.getBufferDeviceAddress == nullptr) entry.getBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(real(device, "vkGetBufferDeviceAddressKHR"));
    entry.cmdCopyBuffer = reinterpret_cast<PFN_vkCmdCopyBuffer>(real(device, "vkCmdCopyBuffer"));
    entry.cmdFillBuffer = reinterpret_cast<PFN_vkCmdFillBuffer>(real(device, "vkCmdFillBuffer"));
    entry.cmdUpdateBuffer = reinterpret_cast<PFN_vkCmdUpdateBuffer>(real(device, "vkCmdUpdateBuffer"));
    entry.cmdCopyImageToBuffer = reinterpret_cast<PFN_vkCmdCopyImageToBuffer>(real(device, "vkCmdCopyImageToBuffer"));
    entry.cmdCopyQueryPoolResults = reinterpret_cast<PFN_vkCmdCopyQueryPoolResults>(real(device, "vkCmdCopyQueryPoolResults"));
    entry.updateDescriptorSets = reinterpret_cast<PFN_vkUpdateDescriptorSets>(real(device, "vkUpdateDescriptorSets"));
    if (entry.allocateMemory == nullptr || entry.freeMemory == nullptr || entry.createShaderModule == nullptr || entry.destroyShaderModule == nullptr ||
        entry.createGraphicsPipelines == nullptr || entry.createComputePipelines == nullptr || entry.destroyPipeline == nullptr ||
        entry.getPipelineCacheData == nullptr || entry.createImage == nullptr || entry.destroyImage == nullptr ||
        entry.createBuffer == nullptr || entry.destroyBuffer == nullptr) return real;
    auto& state = Get();
    {
        std::lock_guard lock(state.devicesMutex);
        if (state.devices.capacity() == 0) state.devices.reserve(16);
        if (state.devices.size() == state.devices.capacity()) return real;
        state.devices.push_back(entry);
    }
    Start();
    return &TrackedDeviceProc;
}

void Start() {
    if (!Enabled()) return;
    auto& state = Get();
    {
        std::lock_guard lock(state.reportMutex);
        if (state.started || state.stopping) return;
        state.started = true;
    }
    std::fprintf(stderr, "[mem] reporter every %u s (APS5_MEM_STATS); fields: process_* from GetProcessMemoryInfo, other_heaps from HeapSummary (the CRT heap is not summarized: HeapSummary holds its lock for seconds once it reaches GBs), crt_small/crt_large/crt_large_top/crt_small_top from a HeapWalk of the process heap, which the Vulkan driver's allocations share, every APS5_MEM_STATS_CRT_EVERY reports (default 30, 0 = off; it holds the heap lock for crt_ms), guest_saved = private copies of unmapped resident direct memory, vq_* from a VirtualQuery walk (committed bytes; vq_private_lt1MB/ge64MB count guest arena mappings too, vq_host_* only allocations outside the arena, vq_host_write_combined = write-combined private memory, which the Vulkan driver commits for device-local allocations), vk_* from the device's vkAllocateMemory/vkFreeMemory, devmem_/hostmem_/importmem_<owner> charge each allocation to the owner scope it was made in (pool_* = the buffer pool's own allocations), img_/buf_<owner> count live VkImages/VkBuffers by owner, pooled_<owner> = live pooled buffer objects by the owner that asked, vk_device_unattributed = device memory made outside every owner scope, the other fields are live MB(objects)\n", PeriodSeconds());
    std::thread([] {
        auto& state = Get();
        std::unique_lock lock(state.reportMutex);
        while (!state.stopping) {
            state.wake.wait_for(lock, std::chrono::seconds(PeriodSeconds()), [&] { return state.stopping; });
            if (state.stopping) break;
            try {
                Report();
            } catch (...) {
                std::fprintf(stderr, "[mem] report failed\n");
            }
        }
    }).detach();
}

void Stop() noexcept {
    auto& state = Get();
    std::lock_guard lock(state.reportMutex);
    state.stopping = true;
    state.wake.notify_all();
}


}
