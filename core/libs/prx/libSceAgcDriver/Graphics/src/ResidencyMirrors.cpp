#include "prx/libSceAgcDriver/Graphics/include/ResidencyMirrors.hpp"
#include "prx/libSceAgcDriver/Execution/include/MemoryStats.hpp"
#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace AgcDriver::Graphics {

MirrorChunk::MirrorChunk(const Context& context, VkDeviceSize bytes) : context(context), bytes(bytes) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Check(context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &buffer), "vkCreateBuffer");
    try {
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
        const VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        const MemoryStats::OwnerScope owner(MemoryStats::Owner::GuestMirror);
        const auto result = context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory);
        if (result != VK_SUCCESS)
            throw std::runtime_error("residency: a mirror chunk of " + std::to_string(requirements.size >> 20) + " MiB was refused in device-local memory type " + std::to_string(allocation.memoryTypeIndex) + " (VkResult " + std::to_string(static_cast<int>(result)) + ")");
        Check(context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, buffer, memory, 0), "vkBindBufferMemory");
        VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        addressInfo.buffer = buffer;
        address = context.Function<PFN_vkGetBufferDeviceAddressKHR>("vkGetBufferDeviceAddressKHR")(context.device, &addressInfo);
        Require(address != 0, "vkGetBufferDeviceAddressKHR returned a null address for a mirror chunk");
    } catch (...) {
        if (memory != VK_NULL_HANDLE) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
        context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
        throw;
    }
}

MirrorChunk::~MirrorChunk() {
    context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
    context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
}

SparseResidency::SparseResidency(const Context& context) : context(context) {
    constexpr std::uint64_t MostBytes = std::uint64_t{512} << 30;
    constexpr std::uint64_t SpaceShares = 4;
    bytes = std::min<std::uint64_t>(MostBytes, context.limits.sparseAddressSpaceSize / SpaceShares) & ~(ResidencyPageBytes - 1u);
    Require(bytes != 0, "residency: the device has no sparse address space");
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkSemaphoreTypeCreateInfoKHR type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;
    const VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
    Check(context.Function<PFN_vkCreateSemaphore>("vkCreateSemaphore")(context.device, &semaphoreInfo, nullptr, &timeline), "vkCreateSemaphore sparse residency");
    Check(context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &buffer), "vkCreateBuffer sparse residency");
    VkMemoryRequirements requirements{};
    context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
    if (requirements.alignment > ResidencyPageBytes || ResidencyPageBytes % requirements.alignment != 0) {
        context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
        context.Function<PFN_vkDestroySemaphore>("vkDestroySemaphore")(context.device, timeline, nullptr);
        throw std::runtime_error("residency: the sparse residency buffer's block of " + std::to_string(requirements.alignment) + " bytes does not divide a residency page");
    }
    VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addressInfo.buffer = buffer;
    address = context.Function<PFN_vkGetBufferDeviceAddressKHR>("vkGetBufferDeviceAddressKHR")(context.device, &addressInfo);
    if (address == 0) {
        context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
        context.Function<PFN_vkDestroySemaphore>("vkDestroySemaphore")(context.device, timeline, nullptr);
        throw std::runtime_error("vkGetBufferDeviceAddressKHR returned a null address for the sparse residency buffer");
    }
}

std::shared_ptr<SparseResidency> SparseResidency::Of(const Context& context) {
    static std::mutex mutex;
    static std::map<VkDevice, std::weak_ptr<SparseResidency>> devices;
    const std::lock_guard lock(mutex);
    auto& held = devices[context.device];
    if (auto shared = held.lock()) return shared;
    auto created = std::make_shared<SparseResidency>(context);
    held = created;
    return created;
}

SparseResidency::~SparseResidency() {
    context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
    context.Function<PFN_vkDestroySemaphore>("vkDestroySemaphore")(context.device, timeline, nullptr);
}

std::uint64_t SparseResidency::Bind(std::span<const VkSparseMemoryBind> binds) {
    if (binds.empty()) return Signalled();
    const std::uint64_t value = signalled.load(std::memory_order_relaxed) + 1u;
    const VkSparseBufferMemoryBindInfo bufferBind{buffer, static_cast<std::uint32_t>(binds.size()), binds.data()};
    VkTimelineSemaphoreSubmitInfoKHR timelineInfo{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR};
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &value;
    VkBindSparseInfo info{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, &timelineInfo};
    info.bufferBindCount = 1;
    info.pBufferBinds = &bufferBind;
    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &timeline;
    if (context.sparseQueue != VK_NULL_HANDLE && context.sparseQueueMutex != nullptr) {
        const std::lock_guard lock(*context.sparseQueueMutex);
        Check(context.Function<PFN_vkQueueBindSparse>("vkQueueBindSparse")(context.sparseQueue, 1, &info, VK_NULL_HANDLE), "vkQueueBindSparse residency");
    } else {
        Check(context.Function<PFN_vkQueueBindSparse>("vkQueueBindSparse")(context.queue, 1, &info, VK_NULL_HANDLE), "vkQueueBindSparse residency");
    }
    signalled.store(value, std::memory_order_release);
    return value;
}

std::uint64_t SparseResidency::Completed() const {
    std::uint64_t value = 0;
    Check(context.Function<PFN_vkGetSemaphoreCounterValueKHR>("vkGetSemaphoreCounterValueKHR")(context.device, timeline, &value), "vkGetSemaphoreCounterValueKHR sparse residency");
    return value;
}

void SparseResidency::Finish(std::uint64_t value) const {
    VkSemaphoreWaitInfoKHR wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR};
    wait.semaphoreCount = 1;
    wait.pSemaphores = &timeline;
    wait.pValues = &value;
    Check(context.Function<PFN_vkWaitSemaphoresKHR>("vkWaitSemaphoresKHR")(context.device, &wait, ~std::uint64_t{0}), "vkWaitSemaphoresKHR sparse residency");
}

ResidencyMirrors::ResidencyMirrors(const Context& context, std::uint32_t chunkSlots)
    : context(context), chunkSlots(chunkSlots),
      copyBuffer(context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")),
      pipelineBarrier(context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")) {
    Require(context.bufferDeviceAddress, "residency mirrors need buffer device addresses");
}

std::uint64_t ResidencyMirrors::Allocate(std::uint32_t chunk) {
    if (chunk >= chunks.size()) chunks.resize(chunk + 1);
    Require(chunks[chunk] == nullptr, "residency: mirror chunk " + std::to_string(chunk) + " allocated twice");
    chunks[chunk] = std::make_shared<MirrorChunk>(context, VkDeviceSize{chunkSlots} << ResidencyPageShift);
    return chunks[chunk]->address;
}

void ResidencyMirrors::Release(std::uint32_t chunk) {
    Require(chunk < chunks.size() && chunks[chunk] != nullptr, "residency: mirror chunk " + std::to_string(chunk) + " released while not allocated");
    chunks[chunk].reset();
}

std::shared_ptr<MirrorChunk> ResidencyMirrors::Chunk(std::uint32_t chunk) const {
    Require(chunk < chunks.size() && chunks[chunk] != nullptr, "residency: mirror chunk " + std::to_string(chunk) + " is not allocated");
    return chunks[chunk];
}

std::pair<VkDeviceMemory, VkDeviceSize> ResidencyMirrors::Locate(VkDeviceAddress mirror) const {
    for (const auto& chunk : chunks) {
        if (chunk != nullptr && mirror >= chunk->address && mirror - chunk->address < chunk->bytes) return {chunk->memory, mirror - chunk->address};
    }
    throw std::runtime_error("residency: no mirror chunk holds the mirror address");
}

std::uint64_t ResidencyMirrors::DeviceBytes() const {
    std::uint64_t bytes = 0;
    for (const auto& chunk : chunks) {
        if (chunk != nullptr) bytes += chunk->bytes;
    }
    return bytes;
}

void ResidencyMirrors::RecordCopyIn(VkCommandBuffer commands, std::span<const ResidencyCopy> copies, const ResidencyImportLocator& import) const {
    record(commands, copies, import, true);
}

void ResidencyMirrors::RecordWriteBack(VkCommandBuffer commands, std::span<const ResidencyCopy> copies, const ResidencyImportLocator& import) const {
    record(commands, copies, import, false);
}

void ResidencyMirrors::record(VkCommandBuffer commands, std::span<const ResidencyCopy> copies, const ResidencyImportLocator& import, bool intoMirror) const {
    if (copies.empty()) return;
    std::map<std::pair<VkBuffer, VkBuffer>, std::vector<VkBufferCopy>> regions;
    for (const auto& copy : copies) {
        Require(copy.offset + copy.bytes <= ResidencyPageBytes && copy.mirror.slot < chunkSlots, "residency: a copy region past its page or chunk");
        const auto location = import(copy.page);
        Require(location.buffer != VK_NULL_HANDLE, "residency: page " + std::to_string(copy.page) + " has no import to copy from or to");
        const auto mirror = Chunk(copy.mirror.chunk);
        const VkDeviceSize mirrorOffset = (VkDeviceSize{copy.mirror.slot} << ResidencyPageShift) + copy.offset;
        const VkDeviceSize importOffset = location.offset + copy.offset;
        if (intoMirror) regions[{location.buffer, mirror->buffer}].push_back({importOffset, mirrorOffset, copy.bytes});
        else regions[{mirror->buffer, location.buffer}].push_back({mirrorOffset, importOffset, copy.bytes});
    }
    const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    pipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    for (const auto& [buffers, list] : regions) copyBuffer(commands, buffers.first, buffers.second, static_cast<std::uint32_t>(list.size()), list.data());
    const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT};
    pipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
}

}
