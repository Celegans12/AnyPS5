#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "VulkanTestDevice.hpp"
#include <SDL_loadso.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr VkDeviceSize MiB = VkDeviceSize{1} << 20u;

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void Call(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(static_cast<int>(result)));
}

struct Usage {
    std::vector<std::int64_t> heaps;
};

class Device {
public:
    Device() {
#ifdef _WIN32
        library = SDL_LoadObject("vulkan-1.dll");
#else
        library = SDL_LoadObject("libvulkan.so.1");
#endif
        Expect(library != nullptr, "cannot load Vulkan");
        try {
            instanceProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_LoadFunction(library, "vkGetInstanceProcAddr"));
            Expect(instanceProc != nullptr, "missing Vulkan instance resolver");
            VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            application.apiVersion = VK_API_VERSION_1_1;
            VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            info.pApplicationInfo = &application;
            Call(function<PFN_vkCreateInstance>("vkCreateInstance")(&info, nullptr, &instance), "vkCreateInstance");
            std::uint32_t count = 0;
            const auto enumerate = function<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
            Call(enumerate(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
            Expect(count != 0, "no Vulkan device");
            std::vector<VkPhysicalDevice> devices(count);
            Call(enumerate(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
            const auto describe = function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
            for (const auto candidate : devices) {
                VkPhysicalDeviceProperties properties{};
                describe(candidate, &properties);
                if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                    physical = candidate;
                    this->properties = properties;
                    break;
                }
            }
            if (physical == VK_NULL_HANDLE) return;
            const auto extensions = function<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
            Call(extensions(physical, nullptr, &count, nullptr), "vkEnumerateDeviceExtensionProperties");
            std::vector<VkExtensionProperties> available(count);
            Call(extensions(physical, nullptr, &count, available.data()), "vkEnumerateDeviceExtensionProperties");
            for (const auto& extension : available) budget |= std::strcmp(extension.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
            if (!budget) return;
            const auto queues = function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
            queues(physical, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            queues(physical, &count, families.data());
            while (family < count && (families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) ++family;
            Expect(family < count, "no Vulkan graphics queue");
            const float priority = 1;
            VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue.queueFamilyIndex = family;
            queue.queueCount = 1;
            queue.pQueuePriorities = &priority;
            const char* enabled[] = {VK_EXT_MEMORY_BUDGET_EXTENSION_NAME};
            VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            create.queueCreateInfoCount = 1;
            create.pQueueCreateInfos = &queue;
            create.enabledExtensionCount = 1;
            create.ppEnabledExtensionNames = enabled;
            Call(function<PFN_vkCreateDevice>("vkCreateDevice")(physical, &create, nullptr, &device), "vkCreateDevice");
            deviceProc = function<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
            function<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(physical, &memory);
            memoryProperties2 = function<PFN_vkGetPhysicalDeviceMemoryProperties2>("vkGetPhysicalDeviceMemoryProperties2");
            formatProperties = function<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties");
            imageFormatProperties = function<PFN_vkGetPhysicalDeviceImageFormatProperties>("vkGetPhysicalDeviceImageFormatProperties");
            at<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(device, family, 0, &queueHandle);
            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.queueFamilyIndex = family;
            Call(at<PFN_vkCreateCommandPool>("vkCreateCommandPool")(device, &pool, nullptr, &commandPool), "vkCreateCommandPool");
        } catch (...) {
            release();
            throw;
        }
    }
    ~Device() { release(); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    bool Usable() const { return device != VK_NULL_HANDLE; }

    template<typename TFunction>
    TFunction at(const char* name) const {
        const auto result = reinterpret_cast<TFunction>(deviceProc(device, name));
        Expect(result != nullptr, name);
        return result;
    }

    Usage Read() const {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &reported};
        memoryProperties2(physical, &properties);
        Usage usage;
        for (std::uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap) usage.heaps.push_back(static_cast<std::int64_t>(reported.heapUsage[heap]));
        return usage;
    }

    // The reported usage once two reads 50 ms apart agree (at most a second).
    Usage Settled() const {
        auto last = Read();
        for (int attempt = 0; attempt < 20; ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            auto next = Read();
            if (next.heaps == last.heaps) return next;
            last = std::move(next);
        }
        return last;
    }

    void Clear(const std::vector<VkImage>& images) const {
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = commandPool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        VkCommandBuffer commands = VK_NULL_HANDLE;
        Call(at<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(device, &allocate, &commands), "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        Call(at<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(commands, &begin), "vkBeginCommandBuffer");
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        const VkClearColorValue color{{0.25f, 0.5f, 0.75f, 1.0f}};
        for (const auto image : images) {
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange = range;
            at<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            at<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
        }
        Call(at<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(commands), "vkEndCommandBuffer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commands;
        Call(at<PFN_vkQueueSubmit>("vkQueueSubmit")(queueHandle, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
        Call(at<PFN_vkQueueWaitIdle>("vkQueueWaitIdle")(queueHandle), "vkQueueWaitIdle");
        at<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(device, commandPool, 1, &commands);
    }

    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
    PFN_vkGetPhysicalDeviceMemoryProperties2 memoryProperties2 = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties formatProperties = nullptr;
    PFN_vkGetPhysicalDeviceImageFormatProperties imageFormatProperties = nullptr;
    bool budget = false;

private:
    template<typename TFunction>
    TFunction function(const char* name) const {
        const auto result = reinterpret_cast<TFunction>(instanceProc(instance, name));
        Expect(result != nullptr, name);
        return result;
    }

    void release() noexcept {
        if (commandPool != VK_NULL_HANDLE) at<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(device, commandPool, nullptr);
        if (device != VK_NULL_HANDLE) function<PFN_vkDestroyDevice>("vkDestroyDevice")(device, nullptr);
        if (instance != VK_NULL_HANDLE) function<PFN_vkDestroyInstance>("vkDestroyInstance")(instance, nullptr);
        if (library != nullptr) SDL_UnloadObject(library);
        commandPool = VK_NULL_HANDLE;
        device = VK_NULL_HANDLE;
        instance = VK_NULL_HANDLE;
        library = nullptr;
    }

    void* library = nullptr;
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    PFN_vkGetDeviceProcAddr deviceProc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkQueue queueHandle = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
};

VkImageCreateInfo ImageInfo(VkFormat format, std::uint32_t side, std::uint32_t mips, VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL) {
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {side, side, 1};
    info.mipLevels = mips;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = tiling;
    info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = tiling == VK_IMAGE_TILING_LINEAR ? VK_IMAGE_LAYOUT_PREINITIALIZED : VK_IMAGE_LAYOUT_UNDEFINED;
    return info;
}

std::string Flags(VkMemoryPropertyFlags flags) {
    std::string text;
    if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) text += "DEVICE_LOCAL ";
    if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) text += "HOST_VISIBLE ";
    if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) text += "HOST_COHERENT ";
    if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) text += "HOST_CACHED ";
    if (flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) text += "LAZY ";
    if (text.empty()) return "none";
    text.pop_back();
    return text;
}

std::string Delta(const Usage& before, const Usage& after) {
    std::string text;
    for (std::size_t heap = 0; heap < before.heaps.size(); ++heap) {
        char part[64];
        std::snprintf(part, sizeof(part), "%sheap%zu %+.1f MiB", text.empty() ? "" : ", ", heap, static_cast<double>(after.heaps[heap] - before.heaps[heap]) / static_cast<double>(MiB));
        text += part;
    }
    return text;
}

std::optional<std::uint32_t> DeviceLocalHeap(const VkPhysicalDeviceMemoryProperties& memory) {
    std::optional<std::uint32_t> heap;
    for (std::uint32_t index = 0; index < memory.memoryHeapCount; ++index) {
        if ((memory.memoryHeaps[index].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 && (!heap || memory.memoryHeaps[index].size > memory.memoryHeaps[*heap].size)) heap = index;
    }
    return heap;
}

struct Images {
    const Device& device;
    std::vector<VkImage> images;
    std::vector<VkDeviceMemory> memories;
    VkDeviceSize bytes = 0;
    explicit Images(const Device& device) : device(device) {}
    Images(const Images&) = delete;
    Images& operator=(const Images&) = delete;
    ~Images() {
        for (const auto image : images) device.at<PFN_vkDestroyImage>("vkDestroyImage")(device.device, image, nullptr);
        for (const auto memory : memories) device.at<PFN_vkFreeMemory>("vkFreeMemory")(device.device, memory, nullptr);
    }
};

void ReportTypes(const Device& device) {
    const auto& memory = device.memory;
    std::printf("%s (vendor 0x%04x, driver 0x%08x)\n", device.properties.deviceName, device.properties.vendorID, device.properties.driverVersion);
    const auto usage = device.Read();
    for (std::uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap) std::printf("heap %u: %llu MiB%s, usage %llu MiB\n", heap, static_cast<unsigned long long>(memory.memoryHeaps[heap].size / MiB), (memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 ? " DEVICE_LOCAL" : "", static_cast<unsigned long long>(usage.heaps[heap] / static_cast<std::int64_t>(MiB)));
    for (std::uint32_t type = 0; type < memory.memoryTypeCount; ++type) std::printf("type %u: heap %u, %s\n", type, memory.memoryTypes[type].heapIndex, Flags(memory.memoryTypes[type].propertyFlags).c_str());
    struct Probe {
        const char* name;
        VkFormat format;
    };
    const Probe probes[] = {{"R8G8B8A8_UNORM", VK_FORMAT_R8G8B8A8_UNORM}, {"R16G16B16A16_SFLOAT", VK_FORMAT_R16G16B16A16_SFLOAT}, {"R32_SFLOAT", VK_FORMAT_R32_SFLOAT}, {"BC1_RGBA_UNORM", VK_FORMAT_BC1_RGBA_UNORM_BLOCK}, {"BC7_UNORM", VK_FORMAT_BC7_UNORM_BLOCK}};
    for (const auto& probe : probes) {
        for (const auto tiling : {VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_TILING_LINEAR}) {
            const auto mips = tiling == VK_IMAGE_TILING_OPTIMAL ? 8u : 1u;
            VkImageFormatProperties limits{};
            const auto info = ImageInfo(probe.format, 256, mips, tiling);
            if (device.imageFormatProperties(device.physical, info.format, info.imageType, info.tiling, info.usage, 0, &limits) != VK_SUCCESS) {
                std::printf("%s %s: not supported for sampling\n", probe.name, tiling == VK_IMAGE_TILING_OPTIMAL ? "optimal" : "linear");
                continue;
            }
            VkImage image = VK_NULL_HANDLE;
            Call(device.at<PFN_vkCreateImage>("vkCreateImage")(device.device, &info, nullptr, &image), "vkCreateImage");
            VkMemoryRequirements requirements{};
            device.at<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(device.device, image, &requirements);
            device.at<PFN_vkDestroyImage>("vkDestroyImage")(device.device, image, nullptr);
            std::printf("%s %s (256x256, %u mips, up to %u mips and %u layers): memoryTypeBits 0x%x, %llu bytes, alignment %llu\n", probe.name, tiling == VK_IMAGE_TILING_OPTIMAL ? "optimal" : "linear", mips, limits.maxMipLevels, limits.maxArrayLayers, requirements.memoryTypeBits, static_cast<unsigned long long>(requirements.size), static_cast<unsigned long long>(requirements.alignment));
        }
    }
}

// Allocates `count` R8G8B8A8 images of `side` with `mips` levels, each with an allocation of its own
// in `type` (or one shared allocation when `shared`), clears them on the GPU and returns them.
VkResult Allocate(const Device& device, Images& images, std::uint32_t type, std::uint32_t count, std::uint32_t side, std::uint32_t mips, bool shared = false, bool bind = true, VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL) {
    const auto info = ImageInfo(VK_FORMAT_R8G8B8A8_UNORM, side, mips, tiling);
    VkMemoryRequirements requirements{};
    for (std::uint32_t index = 0; index < count; ++index) {
        VkImage image = VK_NULL_HANDLE;
        Call(device.at<PFN_vkCreateImage>("vkCreateImage")(device.device, &info, nullptr, &image), "vkCreateImage");
        images.images.push_back(image);
        device.at<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(device.device, image, &requirements);
        Expect((requirements.memoryTypeBits & (1u << type)) != 0, "memory type " + std::to_string(type) + " is not allowed for the image");
    }
    if (!bind) return VK_SUCCESS;
    const auto stride = (requirements.size + requirements.alignment - 1) / requirements.alignment * requirements.alignment;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.memoryTypeIndex = type;
    if (shared) {
        allocation.allocationSize = stride * count;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        Call(device.at<PFN_vkAllocateMemory>("vkAllocateMemory")(device.device, &allocation, nullptr, &memory), "vkAllocateMemory");
        images.memories.push_back(memory);
        images.bytes += allocation.allocationSize;
        for (std::uint32_t index = 0; index < count; ++index) Call(device.at<PFN_vkBindImageMemory>("vkBindImageMemory")(device.device, images.images[images.images.size() - count + index], memory, stride * index), "vkBindImageMemory");
    } else {
        allocation.allocationSize = requirements.size;
        for (std::uint32_t index = 0; index < count; ++index) {
            VkDeviceMemory memory = VK_NULL_HANDLE;
            if (const auto result = device.at<PFN_vkAllocateMemory>("vkAllocateMemory")(device.device, &allocation, nullptr, &memory); result != VK_SUCCESS) {
                device.Clear(std::vector<VkImage>(images.images.end() - count, images.images.end() - count + index));
                return result;
            }
            images.memories.push_back(memory);
            images.bytes += allocation.allocationSize;
            Call(device.at<PFN_vkBindImageMemory>("vkBindImageMemory")(device.device, images.images[images.images.size() - count + index], memory, 0), "vkBindImageMemory");
        }
    }
    device.Clear(std::vector<VkImage>(images.images.end() - count, images.images.end()));
    return VK_SUCCESS;
}

struct Placement {
    std::uint32_t type = 0;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
    std::int64_t bytes = 0;
    std::vector<std::int64_t> moved;
};

// Where images of each memory type land: the reported usage of every heap before and after 24
// cleared 2048x2048 R8G8B8A8 images (16 MiB each) with an allocation each, optimal-tiled in every
// type their requirements allow, and linear-tiled in the host-visible types outside the
// device-local heaps.
std::vector<Placement> Placements(const Device& device) {
    std::vector<Placement> placements;
    constexpr std::uint32_t Count = 24;
    for (const auto tiling : {VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_TILING_LINEAR}) {
        const auto info = ImageInfo(VK_FORMAT_R8G8B8A8_UNORM, 2048, 1, tiling);
        VkImage probe = VK_NULL_HANDLE;
        Call(device.at<PFN_vkCreateImage>("vkCreateImage")(device.device, &info, nullptr, &probe), "vkCreateImage");
        VkMemoryRequirements requirements{};
        device.at<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(device.device, probe, &requirements);
        device.at<PFN_vkDestroyImage>("vkDestroyImage")(device.device, probe, nullptr);
        for (std::uint32_t type = 0; type < device.memory.memoryTypeCount; ++type) {
            if ((requirements.memoryTypeBits & (1u << type)) == 0) continue;
            const auto& memoryType = device.memory.memoryTypes[type];
            if (tiling == VK_IMAGE_TILING_LINEAR && ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0 || (device.memory.memoryHeaps[memoryType.heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)) continue;
            const auto before = device.Settled();
            Images images(device);
            const auto result = Allocate(device, images, type, Count, 2048, 1, false, true, tiling);
            const auto after = device.Settled();
            std::printf("%s images in type %u (heap %u, %s): %zu images, %llu MiB allocated%s: %s\n", tiling == VK_IMAGE_TILING_OPTIMAL ? "optimal" : "linear", type, memoryType.heapIndex, Flags(memoryType.propertyFlags).c_str(), images.memories.size(), static_cast<unsigned long long>(images.bytes / MiB), result == VK_SUCCESS ? "" : " (allocation failed)", Delta(before, after).c_str());
            Placement placement{type, tiling, static_cast<std::int64_t>(images.bytes), {}};
            for (std::size_t heap = 0; heap < before.heaps.size(); ++heap) placement.moved.push_back(after.heaps[heap] - before.heaps[heap]);
            placements.push_back(std::move(placement));
        }
    }
    return placements;
}

// The same with the device-local heap filled to its budget first (optimal images in a device-local
// type until the reported usage reaches the budget less 256 MiB, at most 64 GiB): whether the
// driver puts each candidate type's images in system memory once the VRAM is taken, and whether a
// device-local allocation then fails or goes past the budget.
void FullHeapPlacements(const Device& device, std::uint32_t local) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &reported};
    device.memoryProperties2(device.physical, &properties);
    const auto budget = static_cast<std::int64_t>(reported.heapBudget[local]);
    std::uint32_t deviceType = 0;
    while (deviceType < device.memory.memoryTypeCount && (device.memory.memoryTypes[deviceType].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) ++deviceType;
    Images filler(device);
    for (int round = 0; round < 1024 && device.Read().heaps[local] < budget - static_cast<std::int64_t>(256 * MiB); ++round) {
        if (Allocate(device, filler, deviceType, 4, 2048, 1) != VK_SUCCESS) break;
    }
    const auto full = device.Settled();
    std::printf("filled the device-local heap with %llu MiB of type %u images: %s of a %lld MiB budget\n", static_cast<unsigned long long>(filler.bytes / MiB), deviceType, Delta(Usage{std::vector<std::int64_t>(full.heaps.size(), 0)}, full).c_str(), static_cast<long long>(budget / static_cast<std::int64_t>(MiB)));
    for (const auto& placement : Placements(device)) (void)placement;
    {
        const auto before = device.Settled();
        Images past(device);
        const auto result = Allocate(device, past, deviceType, 64, 2048, 1);
        std::printf("1 GiB more of device-local type %u images past the budget: %zu allocated (%s), %s\n", deviceType, past.memories.size(), result == VK_SUCCESS ? "all" : "failed", Delta(before, device.Settled()).c_str());
        for (std::uint32_t type = 0; type < device.memory.memoryTypeCount; ++type) {
            if ((device.memory.memoryHeaps[device.memory.memoryTypes[type].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) continue;
            const auto info = ImageInfo(VK_FORMAT_R8G8B8A8_UNORM, 2048, 1);
            VkImage probe = VK_NULL_HANDLE;
            Call(device.at<PFN_vkCreateImage>("vkCreateImage")(device.device, &info, nullptr, &probe), "vkCreateImage");
            VkMemoryRequirements requirements{};
            device.at<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(device.device, probe, &requirements);
            device.at<PFN_vkDestroyImage>("vkDestroyImage")(device.device, probe, nullptr);
            if ((requirements.memoryTypeBits & (1u << type)) == 0) continue;
            const auto overflowBefore = device.Settled();
            Images overflow(device);
            auto overflowResult = VK_SUCCESS;
            for (int gib = 0; gib < 12 && overflowResult == VK_SUCCESS; ++gib) {
                overflowResult = Allocate(device, overflow, type, 64, 2048, 1);
                const auto now = device.Settled();
                std::printf("then optimal images in type %u with the device-local heap out of memory: %zu MiB allocated%s, heap0 usage %lld MiB, heap1 usage %lld MiB\n", type, overflow.memories.size() * 16u, overflowResult == VK_SUCCESS ? "" : " (failed)", static_cast<long long>(now.heaps[local] >> 20u), static_cast<long long>(now.heaps[1u - local] >> 20u));
            }
            std::printf("overflow type %u took %zu MiB past the full device-local heap: %s\n", type, overflow.memories.size() * 16u, Delta(overflowBefore, device.Settled()).c_str());
        }
    }
}

// What small images cost in the device-local heap beyond their memory requirements: images without
// memory, images that share one allocation, and images with an allocation each (the texture caches'
// snapshots), plus one view each.
void OverheadTests(const Device& device, std::uint32_t local) {
    for (const std::uint32_t side : {64u, 128u, 256u}) {
        const auto info = ImageInfo(VK_FORMAT_R8G8B8A8_UNORM, side, 1);
        VkImage probe = VK_NULL_HANDLE;
        Call(device.at<PFN_vkCreateImage>("vkCreateImage")(device.device, &info, nullptr, &probe), "vkCreateImage");
        VkMemoryRequirements requirements{};
        device.at<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(device.device, probe, &requirements);
        device.at<PFN_vkDestroyImage>("vkDestroyImage")(device.device, probe, nullptr);
        std::uint32_t type = 0;
        while (type < device.memory.memoryTypeCount && ((requirements.memoryTypeBits & (1u << type)) == 0 || (device.memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0)) ++type;
        Expect(type < device.memory.memoryTypeCount, "no device-local memory type for the image");
        constexpr std::uint32_t Count = 4096;
        const auto perImage = [&](const Usage& before, const Usage& after) { return static_cast<double>(after.heaps[local] - before.heaps[local]) / Count / 1024.0; };
        double unbound = 0, shared = 0, dedicated = 0, views = 0;
        {
            const auto before = device.Settled();
            Images images(device);
            Allocate(device, images, type, Count, side, 1, false, false);
            unbound = perImage(before, device.Settled());
        }
        {
            const auto before = device.Settled();
            Images images(device);
            Allocate(device, images, type, Count, side, 1, true);
            const auto bound = device.Settled();
            shared = perImage(before, bound) - static_cast<double>(images.bytes) / Count / 1024.0;
            std::vector<VkImageView> created;
            for (const auto image : images.images) {
                VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                view.image = image;
                view.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view.format = VK_FORMAT_R8G8B8A8_UNORM;
                view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkImageView handle = VK_NULL_HANDLE;
                Call(device.at<PFN_vkCreateImageView>("vkCreateImageView")(device.device, &view, nullptr, &handle), "vkCreateImageView");
                created.push_back(handle);
            }
            views = perImage(bound, device.Settled());
            for (const auto view : created) device.at<PFN_vkDestroyImageView>("vkDestroyImageView")(device.device, view, nullptr);
        }
        {
            const auto before = device.Settled();
            Images images(device);
            Allocate(device, images, type, Count, side, 1);
            dedicated = perImage(before, device.Settled()) - static_cast<double>(images.bytes) / Count / 1024.0;
        }
        const auto counted = static_cast<double>(AgcDriver::Graphics::ImageAllocationOverhead(requirements.size)) / 1024.0;
        std::printf("%u %ux%u images (%llu bytes, alignment %llu, memory type %u) beyond their memory requirements, per image in the device-local heap: %.1f KiB without memory, %.1f KiB sharing one allocation, %.1f KiB with an allocation each, %.1f KiB per view; the texture caches count %.1f KiB\n", Count, side, side, static_cast<unsigned long long>(requirements.size), static_cast<unsigned long long>(requirements.alignment), type, unbound, shared, dedicated, views, counted);
        Expect(counted >= dedicated * 0.75 && counted <= dedicated + 4.0, "the texture caches count " + std::to_string(counted) + " KiB beyond the memory requirements of a " + std::to_string(side) + "x" + std::to_string(side) + " image with an allocation of its own, which takes " + std::to_string(dedicated) + " KiB more in the device-local heap");
    }
}

}

int main(int argc, char** argv) {
    try {
        std::unique_ptr<Device> device;
        try {
            device = std::make_unique<Device>();
        } catch (const std::exception& error) {
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) throw;
            std::printf("skipped, no usable Vulkan device: %s\n", error.what());
            return VulkanTestSkipped;
        }
        if (!device->Usable()) {
            std::puts("skipped: no discrete Vulkan device with VK_EXT_memory_budget");
            return VulkanTestSkipped;
        }
        const auto local = DeviceLocalHeap(device->memory);
        if (!local) {
            std::puts("skipped: no device-local heap");
            return VulkanTestSkipped;
        }
        ReportTypes(*device);
        if (argc > 1 && std::string(argv[1]) == "full") {
            FullHeapPlacements(*device, *local);
            return 0;
        }
        for (const auto& placement : Placements(*device)) {
            const auto& type = device->memory.memoryTypes[placement.type];
            const auto system = AgcDriver::Graphics::SystemMemoryType(device->memory, 1u << placement.type);
            const auto inVram = placement.moved[*local] >= placement.bytes - placement.bytes / 4;
            const auto inSystem = placement.moved[*local] <= placement.bytes / 4 && type.heapIndex != *local && placement.moved[type.heapIndex] >= placement.bytes - placement.bytes / 4;
            const auto name = std::string(placement.tiling == VK_IMAGE_TILING_OPTIMAL ? "optimal" : "linear") + " images in memory type " + std::to_string(placement.type);
            Expect(inVram || inSystem, name + " moved the reported usage of no heap by their " + std::to_string(placement.bytes >> 20u) + " MiB");
            Expect(system.has_value() == inSystem, name + (inSystem ? " went to system memory, which SystemMemoryType does not take them for" : " are charged to the device-local heap, which SystemMemoryType takes for system memory"));
        }
        OverheadTests(*device, *local);
        std::puts("texture memory placement tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
