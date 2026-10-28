#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYMIRRORS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYMIRRORS_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Residency.hpp"
#include <vulkan/vulkan.h>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace AgcDriver::Graphics {

struct MirrorChunk {
    MirrorChunk(const Context& context, VkDeviceSize bytes);
    MirrorChunk(const MirrorChunk&) = delete;
    MirrorChunk& operator=(const MirrorChunk&) = delete;
    ~MirrorChunk();

    Context context;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    VkDeviceSize bytes = 0;
};

struct ResidencyImportLocation {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
};

using ResidencyImportLocator = std::function<ResidencyImportLocation(std::uint64_t page)>;

class SparseResidency {
public:
    explicit SparseResidency(const Context& context);
    SparseResidency(const SparseResidency&) = delete;
    SparseResidency& operator=(const SparseResidency&) = delete;
    ~SparseResidency();

    static std::shared_ptr<SparseResidency> Of(const Context& context);

    VkDeviceAddress Address() const { return address; }
    VkBuffer Handle() const { return buffer; }
    std::uint64_t Bytes() const { return bytes; }
    std::uint64_t Bind(std::span<const VkSparseMemoryBind> binds);
    VkSemaphore Timeline() const { return timeline; }
    std::uint64_t Signalled() const { return signalled.load(std::memory_order_acquire); }
    std::uint64_t Completed() const;
    void Finish(std::uint64_t value) const;

private:
    Context context;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
    std::atomic<std::uint64_t> signalled{0};
    VkDeviceAddress address = 0;
    std::uint64_t bytes = 0;
};

class ResidencyMirrors {
public:
    ResidencyMirrors(const Context& context, std::uint32_t chunkSlots);

    std::uint64_t Allocate(std::uint32_t chunk);
    void Release(std::uint32_t chunk);
    std::shared_ptr<MirrorChunk> Chunk(std::uint32_t chunk) const;
    std::pair<VkDeviceMemory, VkDeviceSize> Locate(VkDeviceAddress mirror) const;
    void RecordCopyIn(VkCommandBuffer commands, std::span<const ResidencyCopy> copies, const ResidencyImportLocator& import) const;
    void RecordWriteBack(VkCommandBuffer commands, std::span<const ResidencyCopy> copies, const ResidencyImportLocator& import) const;
    std::uint64_t DeviceBytes() const;
    std::uint32_t Slots() const { return chunkSlots; }

private:
    void record(VkCommandBuffer commands, std::span<const ResidencyCopy> copies, const ResidencyImportLocator& import, bool intoMirror) const;

    Context context;
    std::uint32_t chunkSlots;
    std::vector<std::shared_ptr<MirrorChunk>> chunks;
    PFN_vkCmdCopyBuffer copyBuffer = nullptr;
    PFN_vkCmdPipelineBarrier pipelineBarrier = nullptr;
};

}

#endif
