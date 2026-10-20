#include "prx/libSceAgcDriver/Execution/include/BdaFeatures.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PlaceholderImages.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
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
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;

constexpr std::uint32_t Side = 256;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t TileR64KBX = 0x1b;
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t SurfaceBytes = std::size_t{Side} * Side * 4u;
constexpr std::size_t Surfaces = 12;
constexpr std::size_t BlockBytes = SurfaceBytes * Surfaces;
constexpr std::size_t RegistrationGapBytes = 65536;
constexpr std::size_t AllocationBytes = BlockBytes + RegistrationGapBytes;
constexpr std::uint64_t Budget = 1ull << 20u;
constexpr std::uint64_t Ceiling = Budget + Budget / 4u;

class Device {
public:
    explicit Device(bool discrete = false, bool enableStorageImageMultisample = true) {
#ifdef _WIN32
        library = SDL_LoadObject("vulkan-1.dll");
#else
        library = SDL_LoadObject("libvulkan.so.1");
#endif
        Require(library != nullptr, "cannot load Vulkan");
        try {
            instanceProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_LoadFunction(library, "vkGetInstanceProcAddr"));
            Require(instanceProc != nullptr, "missing Vulkan instance resolver");
            VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            application.apiVersion = VK_API_VERSION_1_1;
            VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            info.pApplicationInfo = &application;
            Check(function<PFN_vkCreateInstance>("vkCreateInstance")(&info, nullptr, &instance), "vkCreateInstance");
            std::uint32_t count = 0;
            const auto enumerate = function<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
            Check(enumerate(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
            Require(count != 0, "no Vulkan device");
            std::vector<VkPhysicalDevice> devices(count);
            Check(enumerate(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
            context.physical = devices.front();
            for (const auto candidate : devices) {
                VkPhysicalDeviceProperties properties{};
                function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(candidate, &properties);
                if (discrete && properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                    context.physical = candidate;
                    break;
                }
            }
            const auto extensions = function<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
            Check(extensions(context.physical, nullptr, &count, nullptr), "vkEnumerateDeviceExtensionProperties");
            std::vector<VkExtensionProperties> available(count);
            Check(extensions(context.physical, nullptr, &count, available.data()), "vkEnumerateDeviceExtensionProperties");
            const auto hasExtension = [&](const char* name) {
                for (const auto& extension : available) {
                    if (std::strcmp(extension.extensionName, name) == 0) return true;
                }
                return false;
            };
            auto bytes = AgcDriver::QueryBdaByteFeatures(context.physical, function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2"), available);
            auto address = AgcDriver::QueryBdaFeatures(context.physical, function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2"), available);
            const auto queues = function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
            queues(context.physical, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            queues(context.physical, &count, families.data());
            std::uint32_t family = 0;
            while (family < count && (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) ++family;
            Require(family < count, "no Vulkan compute queue");
            const float priority = 1;
            VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue.queueFamilyIndex = family;
            queue.queueCount = 1;
            queue.pQueuePriorities = &priority;
            VkPhysicalDeviceFeatures supported{};
            function<PFN_vkGetPhysicalDeviceFeatures>("vkGetPhysicalDeviceFeatures")(context.physical, &supported);
            VkPhysicalDeviceFeatures enabled{};
            enabled.shaderInt64 = VK_TRUE;
            enabled.shaderStorageImageMultisample = enableStorageImageMultisample && supported.shaderStorageImageMultisample;
            context.storageImageMultisample = enabled.shaderStorageImageMultisample == VK_TRUE;
            address.pNext = &bytes;
            std::vector<const char*> extensionsEnabled{VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, VK_KHR_8BIT_STORAGE_EXTENSION_NAME};
            if (hasExtension(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
                extensionsEnabled.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
                VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
                VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &hostProperties};
                function<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(context.physical, &properties);
                context.hostImportAlignment = hostProperties.minImportedHostPointerAlignment;
            }
            VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &address};
            device.queueCreateInfoCount = 1;
            device.pQueueCreateInfos = &queue;
            device.enabledExtensionCount = static_cast<std::uint32_t>(extensionsEnabled.size());
            device.ppEnabledExtensionNames = extensionsEnabled.data();
            device.pEnabledFeatures = &enabled;
            Check(function<PFN_vkCreateDevice>("vkCreateDevice")(context.physical, &device, nullptr, &context.device), "vkCreateDevice");
            context.deviceProc = function<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
            function<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(context.physical, &context.memory);
            VkPhysicalDeviceProperties properties{};
            function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(context.physical, &properties);
            context.limits = properties.limits;
            context.bufferDeviceAddress = true;
            context.formatProperties = function<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties");
            context.imageFormatProperties = function<PFN_vkGetPhysicalDeviceImageFormatProperties>("vkGetPhysicalDeviceImageFormatProperties");
            if (hasExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) context.memoryProperties2 = function<PFN_vkGetPhysicalDeviceMemoryProperties2>("vkGetPhysicalDeviceMemoryProperties2");
            context.Function<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(context.device, family, 0, &context.queue);
            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool.queueFamilyIndex = family;
            Check(context.Function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(context.device, &pool, nullptr, &context.pool), "vkCreateCommandPool");
        } catch (...) {
            release();
            throw;
        }
    }

    ~Device() { release(); }
    const Context& GetContext() const { return context; }

private:
    template<typename TFunction>
    TFunction function(const char* name) const {
        const auto result = reinterpret_cast<TFunction>(instanceProc(instance, name));
        Require(result != nullptr, name);
        return result;
    }

    void release() noexcept {
        if (context.pool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(context.device, context.pool, nullptr);
        context.bufferPool.reset();
        if (context.device != VK_NULL_HANDLE) function<PFN_vkDestroyDevice>("vkDestroyDevice")(context.device, nullptr);
        if (instance != VK_NULL_HANDLE) function<PFN_vkDestroyInstance>("vkDestroyInstance")(instance, nullptr);
        if (library != nullptr) SDL_UnloadObject(library);
    }

    void* library = nullptr;
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    Context context{};
};

class Block {
public:
    explicit Block(const Context& context) : context(context) {
#ifdef _WIN32
        memory = VirtualAlloc(nullptr, AllocationBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        memory = std::aligned_alloc(65536, AllocationBytes);
#endif
        Require(memory != nullptr, "cannot allocate the texture block");
        std::memset(memory, 0x5a, BlockBytes);
        GuestAllocations::Mutation().Add(memory, BlockBytes, true, true, true);
    }

    ~Block() {
        ClearCachedTextures(context.device);
        GuestAllocations::Mutation().Remove(memory);
        HostImportFor(context, Surface(0), BlockBytes);
#ifdef _WIN32
        VirtualFree(memory, 0, MEM_RELEASE);
#else
        std::free(memory);
#endif
    }

    std::uint64_t Surface(std::size_t surface) const { return reinterpret_cast<std::uint64_t>(memory) + surface * SurfaceBytes; }

private:
    const Context& context;
    void* memory = nullptr;
};

std::array<std::uint32_t, 8> Descriptor(std::uint64_t address, std::uint32_t side = Side) {
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32UInt << 20u) | (((side - 1u) & 3u) << 30u),
        ((side - 1u) >> 2u) | ((side - 1u) << 14u),
        0xfacu | (TileR64KBX << 20u) | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

std::shared_ptr<Texture> Sampled(const Context& context, std::uint64_t address, std::uint32_t side = Side) {
    const auto words = Descriptor(address, side);
    return CachedSampledTexture(context, words);
}

GuestTextureResource Resource(std::uint64_t address, std::uint32_t side = Side) {
    const auto words = Descriptor(address, side);
    return DecodeTextureResource(words);
}

std::shared_ptr<ShaderResources> ImageResources(const Context& context, const Block& block, std::size_t first, std::uint32_t count, bool sampled) {
    ShaderRecompiler::DescriptorBinding binding{};
    binding.kind = sampled ? ShaderRecompiler::DescriptorKind::SampledImage : ShaderRecompiler::DescriptorKind::StorageImage;
    binding.role = ShaderRecompiler::DescriptorRole::GuestImages;
    binding.descriptorSet = 0;
    binding.binding = (sampled ? ShaderRecompiler::RuntimeAbi::FirstImageBinding : ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding) + PlaceholderDimensionsPerClass + 2u;
    binding.count = count;
    binding.imageShape = ShaderRecompiler::DescriptorImageShape::Image2D;
    binding.imageWritten.assign(count, false);
    binding.imageSamplers.assign(count, 0u);
    for (std::uint32_t element = 0; element < count; ++element) {
        const auto words = Descriptor(block.Surface(first + element));
        binding.guestDescriptor.insert(binding.guestDescriptor.end(), words.begin(), words.end());
    }
    ShaderRecompiler::RecompileResult program;
    program.bindings.push_back(std::move(binding));
    const CompiledShader shader{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    return std::make_shared<ShaderResources>(context, shader);
}

void recorderEvictionTests(const Context& context, const Block& block) {
    auto& cache = SharedResourceCache();
    cache.Clear();
    ClearCachedTextures(context.device);
    auto sampled = ImageResources(context, block, 0, 1, true);
    const auto images = sampled->StorageImages();
    Require(images.size() == 1, "the eviction fixture has no sampled storage image");
    const std::weak_ptr<ShaderResources> resources = sampled;
    const std::weak_ptr<Texture> view = sampled->SampledTextures().front();
    const std::weak_ptr<StorageTexture> image = sampled->SampledTextures().front()->SharedStorageSource();
    cache.Insert({1}, sampled);
    Require(cache.Find({1}) == sampled, "the recorder fixture's descriptor set was not cached");
    sampled.reset();
    Buffer readback(context, sizeof(std::uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    {
        Recorder recorder(context);
        recorder.Activate();
        recorder.Commands();
        for (std::size_t surface = 1; surface < Surfaces; ++surface) CachedStorageSurface(context, Resource(block.Surface(surface)));
        Require(cache.Find({1}) == nullptr, "storage eviction left the recorded descriptor set cached");
        Require(!resources.expired() && !view.expired() && !image.expired(), "eviction destroyed the recorder's descriptor set, view or image");
        Require(!image.lock()->Cached(), "the recorder fixture did not evict the source image");
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {1, 1, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, images.front().first, VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Sync();
        std::uint32_t pixel = 0;
        std::memcpy(&pixel, readback.Bytes().data(), sizeof(pixel));
        Require(pixel == 0x5a5a5a5a, "eviction changed the recorder's storage image pixel");
    }
    Require(resources.expired() && view.expired() && image.expired(), "an evicted image outlived its completed recorder");
    std::puts("storage eviction recorder lifetime and pixel readback passed");
}

bool resourceCacheTests(Context context, const std::string& mode) {
    PlaceholderImages placeholders(context);
    context.placeholderImages = &placeholders;
    TextureCache textures(context);
    context.textureCache = &textures;
    Block block(context);
    if (context.hostImportAlignment == 0 || HostImportFor(context, block.Surface(0), BlockBytes) == nullptr) {
        Require(std::getenv("ANYPS5_REQUIRE_VULKAN") == nullptr, "resource cache tests require host imports");
        std::puts("skipped, resource cache tests require host imports");
        return false;
    }
    auto& cache = SharedResourceCache();
    cache.Clear();
    try {
        if (mode == "resource-clear") {
            auto storage = ImageResources(context, block, 0, 1, false);
            auto sampled = ImageResources(context, block, 0, 1, true);
            Require(sampled->SampledTextures().front()->ViewsStorageImage(), "the sampled resource is not a storage-image view");
            const std::weak_ptr<StorageTexture> image = StorageTexture::FindLive(block.Surface(0), SurfaceBytes);
            const std::weak_ptr<ShaderResources> storageSet = storage;
            const std::weak_ptr<ShaderResources> sampledSet = sampled;
            cache.Insert({1}, storage);
            cache.Insert({2}, sampled);
            Require(cache.Size() == 2, "the resource cache did not accept the live descriptor sets");
            sampled.reset();
            ClearCachedTextures(context.device);
            Require(cache.Find({1}) == nullptr && cache.Find({2}) == nullptr && cache.Size() == 0, "clearing storage images left their resources cached");
            Require(sampledSet.expired(), "cache clear kept the sampled descriptor set alive");
            Require(!image.expired() && !storageSet.expired(), "cache clear destroyed an active caller's resources");
            storage.reset();
            Require(image.expired() && storageSet.expired(), "cleared storage image outlived its last active resource");
        } else if (mode == "resource-oversized") {
            for (const bool sampledImages : {false, true}) {
                const std::weak_ptr<StorageTexture> early = CachedStorageSurface(context, Resource(block.Surface(0)));
                auto oversized = ImageResources(context, block, 0, static_cast<std::uint32_t>(Surfaces), sampledImages);
                const std::weak_ptr<ShaderResources> rejected = oversized;
                auto image = early.lock();
                Require(oversized->Reusable(), "the oversized descriptor set is not eligible for resource caching");
                Require(image != nullptr && !image->Cached(), "the oversized descriptor set did not evict an early image while building");
                if (sampledImages) Require(oversized->SampledTextures().size() == Surfaces && oversized->SampledTextures().front()->StorageSource() == image.get(), "the oversized sampled descriptor set does not retain the early storage image");
                std::vector<std::shared_ptr<ShaderResources>> displaced;
                cache.Insert({1}, oversized, &displaced);
                Require(cache.Size() == 0 && displaced.empty(), sampledImages ? "an oversized sampled descriptor set entered the resource cache" : "an oversized storage descriptor set entered the resource cache");
                image.reset();
                oversized.reset();
                Require(rejected.expired() && early.expired(), "the rejected oversized descriptor set retained an evicted storage image");
                std::puts(sampledImages ? "resource cache oversized sampled lifetime passed" : "resource cache oversized storage lifetime passed");
                ClearCachedTextures(context.device);
            }
        } else {
            auto storage = ImageResources(context, block, 0, 1, false);
            auto sampled = ImageResources(context, block, 0, 1, true);
            Require(storage->Reusable() && sampled->Reusable(), "the stale descriptor sets are not eligible for resource caching");
            Require(sampled->SampledTextures().front()->ViewsStorageImage(), "the sampled resource is not a storage-image view");
            auto image = StorageTexture::FindLive(block.Surface(0), SurfaceBytes);
            const std::weak_ptr<StorageTexture> staleImage = image;
            const std::weak_ptr<ShaderResources> storageSet = storage;
            const std::weak_ptr<ShaderResources> sampledSet = sampled;
            Require(image != nullptr && image->Cached(), "the resource has no cached storage image");
            for (std::size_t surface = 1; surface < Surfaces && image->Cached(); ++surface) CachedStorageSurface(context, Resource(block.Surface(surface)));
            Require(!image->Cached(), "the test never evicted the oldest storage image");
            auto live = ImageResources(context, block, Surfaces - 1, 1, false);
            cache.Insert({1}, live);
            Require(cache.Find({1}) == live, "the resource cache rejected a live descriptor set");
            std::vector<std::shared_ptr<ShaderResources>> displaced;
            for (const auto& stale : {storage, sampled}) {
                cache.Insert({2}, stale, &displaced);
                Require(cache.Find({2}) == nullptr && cache.Size() == 1 && displaced.empty(), "an uncached image was admitted to the resource cache");
                cache.Insert({1}, stale, &displaced);
                Require(cache.Find({1}) == live && cache.Size() == 1 && displaced.empty(), "an uncached image replaced a live resource-cache entry");
            }
            image.reset();
            storage.reset();
            Require(storageSet.expired() && !sampledSet.expired() && !staleImage.expired(), "stale storage release did not preserve the sampled caller's image");
            std::puts("resource cache stale storage caller release passed");
            sampled.reset();
            Require(sampledSet.expired() && staleImage.expired(), "the rejected descriptor sets retained an evicted storage image");
            std::puts("resource cache stale sampled caller release passed");
        }
        if (mode == "resource-admission") recorderEvictionTests(context, block);
    } catch (...) {
        cache.Clear();
        throw;
    }
    cache.Clear();
    return true;
}

void viewTests(const Context& context) {
    if (context.hostImportAlignment == 0) {
        std::puts("host imports unavailable: views of storage images not tested");
        return;
    }
    Block block(context);
    if (HostImportFor(context, block.Surface(0), BlockBytes) == nullptr) {
        std::puts("host import of the texture block refused: views of storage images not tested");
        return;
    }
    const auto before = TextureCacheUsage();
    auto view = Sampled(context, block.Surface(0));
    Require(view->ViewsStorageImage(), "a sampled texture in host-imported memory is not a view of its storage image");
    const std::weak_ptr<Texture> viewed = view;
    const std::weak_ptr<StorageTexture> storage = view->SharedStorageSource();
    const auto after = TextureCacheUsage();
    Require(after.sampledEntries == before.sampledEntries + 1u, "the view did not enter the sampled texture cache");
    Require(after.sampledBytes == before.sampledBytes, "a view of a cached storage image counted " + std::to_string(after.sampledBytes - before.sampledBytes) + " bytes against the sampled texture cache");
    Require(after.storageBytes > before.storageBytes, "the storage image of the view is not counted by the storage image cache");
    Require(Sampled(context, block.Surface(0)) == view, "the second lookup of the surface did not hit its view");
    view.reset();
    for (std::size_t surface = 1; surface < Surfaces; ++surface) CachedStorageSurface(context, Resource(block.Surface(surface)));
    Require(viewed.expired() && storage.expired(), "the sampled texture cache kept the view of an evicted storage image");
    const auto evicted = TextureCacheUsage();
    Require(evicted.sampledEntries == after.sampledEntries - 1u && evicted.sampledBytes == before.sampledBytes, "dropping the view of the evicted storage image changed other sampled entries");
    auto again = Sampled(context, block.Surface(0));
    Require(again->ViewsStorageImage() && again->StorageSource() != nullptr && StorageImageCached(context, again->StorageSource()), "the surface's next lookup does not view its new cached storage image");
    const std::weak_ptr<Texture> heldView = again;
    const std::weak_ptr<StorageTexture> heldStorage = again->SharedStorageSource();
    for (std::size_t surface = 1; surface < Surfaces; ++surface) CachedStorageSurface(context, Resource(block.Surface(surface)));
    Require(!StorageImageCached(context, again->StorageSource()), "the held view's source was not evicted");
    Require(!heldView.expired() && !heldStorage.expired(), "eviction destroyed an active caller's view or image");
    again.reset();
    Require(heldView.expired() && heldStorage.expired(), "the evicted image outlived its last caller without a sampled lookup");
}

void reportedBudgetTests(const Context& context) {
    if (context.memoryProperties2 == nullptr) {
        std::puts("VK_EXT_memory_budget unavailable: the sampled texture cache keeps a quarter of the device-local heap");
        return;
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    properties.pNext = &reported;
    context.memoryProperties2(context.physical, &properties);
    const auto budgets = TextureCacheBudgetsFor(context.memory, &reported, {});
    const auto budget = budgets.sampled;
    Require(budget != 0 && budgets.ceiling > budget && budgets.storage <= TextureCacheBudget(context.memory) && budgets.target >= budgets.storage, "the texture cache budgets of the device are inconsistent");
    for (std::uint32_t heap = 0; heap < context.memory.memoryHeapCount; ++heap) {
        if ((context.memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
        std::printf("device-local heap %u: %llu MiB, VK_EXT_memory_budget budget %llu MiB, usage %llu MiB\n", heap, static_cast<unsigned long long>(context.memory.memoryHeaps[heap].size >> 20u), static_cast<unsigned long long>(reported.heapBudget[heap] >> 20u), static_cast<unsigned long long>(reported.heapUsage[heap] >> 20u));
    }
    std::printf("sampled texture cache budget %llu MiB (ceiling %llu MiB), storage image cache budget %llu MiB, VRAM share of the texture caches %llu MiB\n", static_cast<unsigned long long>(budget >> 20u), static_cast<unsigned long long>(budgets.ceiling >> 20u), static_cast<unsigned long long>(budgets.storage >> 20u), static_cast<unsigned long long>(budgets.target >> 20u));
}

void snapshotTests(const Context& context) {
    Block block(context);
    auto first = Sampled(context, block.Surface(0));
    Require(!first->ViewsStorageImage(), "a sampled texture outside host-imported memory is not a snapshot");
    const auto held = TextureCacheUsage().sampledBytes;
    Require(held == first->AllocationBytes(), "a snapshot counted " + std::to_string(held) + " bytes, not its image allocation of " + std::to_string(first->AllocationBytes()));
    const std::weak_ptr<Texture> oldest = first;
    first.reset();
    const auto capacity = static_cast<std::size_t>(Budget / held);
    Require(capacity >= 2u && capacity + 1u < Surfaces, "a snapshot of the 256 KiB surface counts " + std::to_string(held) + " bytes, which the test's 1 MiB budget does not fit as planned");
    for (std::size_t surface = 1; surface < capacity; ++surface) Sampled(context, block.Surface(surface));
    Require(!oldest.expired() && TextureCacheUsage().sampledEntries == capacity, "the sampled texture cache did not keep " + std::to_string(capacity) + " snapshots within its budget");
    Sampled(context, block.Surface(capacity));
    Require(oldest.expired(), "the least recently used snapshot was not evicted past the budget");
    Require(TextureCacheUsage().sampledBytes <= Budget, "the sampled texture cache holds more than its budget");
    ClearCachedTextures(context.device);
    constexpr std::uint32_t TinySide = 16;
    const auto tiny = Sampled(context, block.Surface(0), TinySide);
    const auto guestBytes = DescribeSurface(Resource(block.Surface(0), TinySide)).guestBytes;
    if (tiny->AllocationBytes() >= guestBytes) {
        std::printf("the 16x16 snapshot's image takes %llu bytes, not less than its %llu guest bytes: counting the allocation alone not tested\n", static_cast<unsigned long long>(tiny->AllocationBytes()), static_cast<unsigned long long>(guestBytes));
        return;
    }
    Require(TextureCacheUsage().sampledBytes == tiny->AllocationBytes(), "a snapshot whose " + std::to_string(tiny->AllocationBytes()) + "-byte image is smaller than its " + std::to_string(guestBytes) + " guest bytes counted " + std::to_string(TextureCacheUsage().sampledBytes) + " bytes, not its allocation");
}

// A snapshot is validated by block hashes of its guest bytes: an unchanged surface hits its entry,
// and one changed byte anywhere (either end, either side of a 64 KiB block boundary) makes it again.
// A loading screen (lean-x13): the present counter stands still while one build binds thousands of
// textures and holds them all until it is recorded. The cache's textures from seconds ago must make
// room instead of every new texture becoming a one-off; a texture the build binds twice is made
// once; and the device-local sampled images stay within the cache's ceiling however many the build
// holds (past it, new textures go to system memory when the device has a memory type for that).
void burstTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    const auto deviceBefore = SampledTextureMemory();
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    Require(capacity >= 2u && 2u * capacity + 2u <= Surfaces, "a snapshot of the 256 KiB surface holds " + std::to_string(bytes) + " bytes, which the test's 1 MiB budget does not fit as planned");
    std::vector<std::weak_ptr<Texture>> stale{Sampled(context, block.Surface(0))};
    for (std::size_t surface = 1; surface < capacity; ++surface) stale.push_back(Sampled(context, block.Surface(surface)));
    Require(TextureCacheUsage().sampledEntries == capacity, "the stale frame did not fill the cache");
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));
    std::vector<std::shared_ptr<Texture>> build;
    std::size_t madeTwice = 0;
    for (std::size_t surface = capacity; surface < Surfaces; ++surface) build.push_back(Sampled(context, block.Surface(surface)));
    std::size_t staleLeft = 0;
    for (const auto& texture : stale) staleLeft += !texture.expired();
    for (std::size_t surface = capacity; surface < Surfaces; ++surface) {
        auto again = Sampled(context, block.Surface(surface));
        madeTwice += again != build[surface - capacity];
        build.push_back(std::move(again));
    }
    const auto device = SampledTextureMemory() - deviceBefore;
    std::printf("a build holding %zu textures after 2 s without a present: %zu made twice, %zu of the %zu stale textures left, %llu KiB of device-local sampled images (ceiling %llu KiB)\n", Surfaces - capacity, madeTwice, staleLeft, capacity, static_cast<unsigned long long>(device >> 10u), static_cast<unsigned long long>(Ceiling >> 10u));
    Require(madeTwice == 0, std::to_string(madeTwice) + " textures the build binds twice were made twice");
    Require(staleLeft == 0, std::to_string(staleLeft) + " textures last used seconds before kept the cache from the build's textures");
    bool hostType = false;
    for (std::uint32_t type = 0; type < context.memory.memoryTypeCount; ++type) hostType |= (context.memory.memoryHeaps[context.memory.memoryTypes[type].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0;
    if (!hostType) {
        std::puts("no memory type outside the device-local heaps: the spill to system memory is not tested");
        return;
    }
    Require(device <= Ceiling, "a build holding " + std::to_string(Surfaces - capacity) + " textures took " + std::to_string(device >> 10u) + " KiB of device-local sampled images, past the " + std::to_string(Ceiling >> 10u) + " KiB ceiling");
    build.clear();
    ClearCachedTextures(context.device);
    Require(SampledTextureMemory() == deviceBefore, "the burst's textures outlived the build and the cache");
}

void snapshotChangeTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    auto texture = Sampled(context, block.Surface(1));
    Require(!texture->ViewsStorageImage(), "a sampled texture outside host-imported memory is not a snapshot");
    Require(Sampled(context, block.Surface(1)) == texture, "an unchanged snapshot was made again");
    auto* bytes = reinterpret_cast<std::uint8_t*>(block.Surface(1));
    for (const std::size_t offset : {std::size_t{0}, std::size_t{65535}, std::size_t{65536}, SurfaceBytes / 2 + 3, SurfaceBytes - 1}) {
        bytes[offset] ^= 0x10;
        auto again = Sampled(context, block.Surface(1));
        Require(again != texture, "a changed byte at +" + std::to_string(offset) + " of a snapshot's surface was not seen");
        texture = again;
        Require(Sampled(context, block.Surface(1)) == texture, "the snapshot made after a change at +" + std::to_string(offset) + " was made again");
    }
    texture.reset();
    ClearCachedTextures(context.device);
}

void heldTests(const Context& context) {
    Block block(context);
    auto held = Sampled(context, block.Surface(0));
    const auto bytes = TextureCacheUsage().sampledBytes;
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    Require((capacity + 1u) * bytes <= Ceiling && capacity + 2u < Surfaces, "a snapshot of the 256 KiB surface counts " + std::to_string(bytes) + " bytes, which the test's ceiling does not fit as planned");
    std::vector<std::weak_ptr<Texture>> released;
    for (std::size_t surface = 1; surface < capacity; ++surface) released.push_back(Sampled(context, block.Surface(surface)));
    Sampled(context, block.Surface(capacity));
    Require(Sampled(context, block.Surface(0)) == held, "the sampled texture cache evicted a texture its caller still holds");
    Require(released.front().expired(), "the least recently used texture nobody holds was not evicted in place of the held one");
    held.reset();
    ClearCachedTextures(context.device);
    std::vector<std::shared_ptr<Texture>> all;
    for (std::size_t surface = 0; surface <= capacity; ++surface) all.push_back(Sampled(context, block.Surface(surface)));
    Require(TextureCacheUsage().sampledEntries == capacity + 1u && TextureCacheUsage().sampledBytes > Budget, "textures their callers hold left a cache over its budget");
    for (std::size_t surface = 0; surface < all.size(); ++surface) Require(Sampled(context, block.Surface(surface)) == all[surface], "surface " + std::to_string(surface) + " was made again while its texture was held");
    const std::weak_ptr<Texture> refused = Sampled(context, block.Surface(capacity + 1u));
    Require(refused.expired(), "past the ceiling, a texture nobody holds any more was cached or kept");
    Require(TextureCacheUsage().sampledEntries == capacity + 1u && TextureCacheUsage().sampledBytes <= Ceiling, "held textures took the sampled texture cache past a quarter over its budget");
    for (std::size_t surface = 0; surface < all.size(); ++surface) Require(Sampled(context, block.Surface(surface)) == all[surface], "past the ceiling, held surface " + std::to_string(surface) + " was evicted, which frees nothing, instead of the new texture being refused");
    all.clear();
    Sampled(context, block.Surface(capacity + 2u));
    Require(TextureCacheUsage().sampledEntries == capacity && TextureCacheUsage().sampledBytes <= Budget, "released textures were not evicted down to the budget");
}

void releasedTests(Context context) {
    PlaceholderImages placeholders(context);
    context.placeholderImages = &placeholders;
    TextureCache textures(context);
    context.textureCache = &textures;
    ClearCachedTextures(context.device);
    auto& cache = SharedResourceCache();
    cache.Clear();
    try {
        Block block(context);
        auto pinned = ImageResources(context, block, 0, 1, true);
        Require(pinned->SampledTextures().size() == 1 && !pinned->SampledTextures().front()->ViewsStorageImage(), "the pinned descriptor set does not hold a sampled snapshot");
        const std::weak_ptr<Texture> texture = pinned->SampledTextures().front();
        const std::weak_ptr<ShaderResources> pinnedSet = pinned;
        cache.Insert({1}, pinned);
        Require(cache.Find({1}) == pinned, "the resource cache rejected a descriptor set whose texture is cached");
        pinned.reset();
        ClearCachedTextures(context.device);
        Require(cache.Find({1}) == nullptr && cache.Size() == 0, "the resource cache kept an entry holding a released texture");
        Require(pinnedSet.expired() && texture.expired(), "the resource cache kept a released texture alive");
        auto live = ImageResources(context, block, 1, 1, true);
        std::vector<std::shared_ptr<ShaderResources>> displaced;
        cache.Insert({2}, live, &displaced);
        Require(cache.Find({2}) == live && cache.Size() == 1 && displaced.empty(), "the resource cache rejected a descriptor set whose texture is cached");
        auto stale = ImageResources(context, block, 2, 1, true);
        ClearCachedTextures(context.device);
        Require(stale->SampledTextures().front()->Released(), "a held texture the sampled cache dropped is not marked released");
        Require(cache.Find({2}) == nullptr && cache.Size() == 0, "the live entry whose texture was released stayed in the resource cache");
        cache.Insert({3}, stale, &displaced);
        Require(cache.Find({3}) == nullptr && cache.Size() == 0, "a descriptor set holding a released texture entered the resource cache");
        displaced.clear();
    } catch (...) {
        cache.Clear();
        throw;
    }
    cache.Clear();
    ClearCachedTextures(context.device);
    std::puts("resource cache released texture tests passed");
}

void rescanTests(const Context& context) {
    constexpr std::uint32_t SmallSide = 128;
    constexpr std::size_t SmallBytes = std::size_t{SmallSide} * SmallSide * 4u;
    Block block(context);
    const auto small = [&](std::size_t surface) { return Sampled(context, block.Surface(0) + surface * SmallBytes, SmallSide); };
    std::vector<std::shared_ptr<Texture>> all{small(0)};
    const auto bytes = TextureCacheUsage().sampledBytes;
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    Require((capacity + 3u) * bytes <= Ceiling && (capacity + 3u) * SmallBytes <= BlockBytes, "a snapshot of the 64 KiB surface counts " + std::to_string(bytes) + " bytes, which the test's ceiling does not fit as planned");
    for (std::size_t surface = 1; surface <= capacity; ++surface) all.push_back(small(surface));
    const std::weak_ptr<Texture> newest = all.back();
    all.pop_back();
    all.push_back(small(capacity + 1u));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    all.push_back(small(capacity + 2u));
    Require(newest.expired(), "a released texture behind held ones was still cached a second later");
    Require(TextureCacheUsage().sampledEntries == capacity + 2u, "the rescan left " + std::to_string(TextureCacheUsage().sampledEntries) + " entries, not the " + std::to_string(capacity + 2u) + " textures still held");
}

void viewEvictionTests(const Context& context) {
    if (context.hostImportAlignment == 0) return;
    Block snapshots(context);
    Block imported(context);
    if (HostImportFor(context, imported.Surface(0), BlockBytes) == nullptr) return;
    const std::weak_ptr<Texture> view = Sampled(context, imported.Surface(0));
    Require(!view.expired() && TextureCacheUsage().sampledBytes == 0, "the view of a cached storage image did not enter the sampled texture cache at 0 bytes");
    Sampled(context, snapshots.Surface(0));
    const auto snapshotBytes = TextureCacheUsage().sampledBytes;
    Require(snapshotBytes != 0, "the snapshot fixture unexpectedly became a view of a cached storage image");
    const auto capacity = static_cast<std::size_t>(Budget / snapshotBytes);
    for (std::size_t surface = 1; surface <= capacity; ++surface) Sampled(context, snapshots.Surface(surface));
    Require(!view.expired(), "the sampled texture cache evicted a view of a cached storage image, which frees no memory");
}

void frameTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    const auto frame = capacity + 2u;
    Require(capacity >= 2u && 2u * frame <= Surfaces && TextureCacheUsage().sampledEntries == 1u, "a snapshot of the 256 KiB surface holds " + std::to_string(bytes) + " bytes, which the test's 1 MiB budget does not fit as planned");
    std::vector<std::weak_ptr<Texture>> made(frame);
    for (std::size_t surface = 0; surface < frame; ++surface) made[surface] = Sampled(context, block.Surface(surface));
    Require(TextureCacheUsage().sampledEntries == capacity, "a frame of " + std::to_string(frame) + " textures left " + std::to_string(TextureCacheUsage().sampledEntries) + " cached, not the " + std::to_string(capacity) + " the budget holds");
    for (std::size_t surface = capacity; surface < frame; ++surface) Require(made[surface].expired(), "surface " + std::to_string(surface) + ", used once in a frame whose textures fill the cache, was cached in place of an earlier one");
    Recorder::CountPresent();
    std::size_t hits = 0;
    for (std::size_t surface = 0; surface < frame; ++surface) {
        const auto again = Sampled(context, block.Surface(surface));
        hits += again == made[surface].lock();
    }
    Require(hits == capacity, "the next frame over the same " + std::to_string(frame) + " textures found " + std::to_string(hits) + " of them cached, not the " + std::to_string(capacity) + " the cache kept");
    if (std::getenv("APS5_SAMPLED_LATER_USE_REFUSAL") != nullptr) {
        auto refused = Sampled(context, block.Surface(frame - 1u));
        Sampled(context, block.Surface(frame - 1u));
        for (std::size_t surface = 0; surface < capacity; ++surface) Require(!made[surface].expired(), "a texture used a second time in the frame was cached in place of one of the frame's textures");
        Require(TextureCacheUsage().sampledEntries == capacity && TextureCacheUsage().sampledBytes <= Budget, "the texture used a second time left the cache over its budget");
    } else {
        auto admitted = Sampled(context, block.Surface(frame - 1u));
        Require(Sampled(context, block.Surface(frame - 1u)) == admitted, "a texture used a second time in the frame was not cached");
        Require(TextureCacheUsage().sampledEntries == capacity && TextureCacheUsage().sampledBytes <= Budget, "the texture admitted on its second use left the cache over its budget");
    }
    Recorder::CountPresent();
    std::vector<std::weak_ptr<Texture>> moved(frame);
    for (std::size_t surface = 0; surface < frame; ++surface) moved[surface] = Sampled(context, block.Surface(frame + surface));
    for (std::size_t surface = 0; surface < frame; ++surface) Require(moved[surface].expired(), "surface " + std::to_string(frame + surface) + ", first used in the frame after the cached textures were, was cached in place of one of them");
    Require(TextureCacheUsage().sampledEntries == capacity, "a frame over other textures left " + std::to_string(TextureCacheUsage().sampledEntries) + " of the previous frame's textures cached");
    Recorder::CountPresent();
    for (std::size_t surface = 0; surface < frame; ++surface) moved[surface] = Sampled(context, block.Surface(frame + surface));
    for (std::size_t surface = 0; surface < capacity; ++surface) Require(!moved[surface].expired(), "surface " + std::to_string(frame + surface) + " of a second frame over other textures did not replace the textures last used two frames before");
    Require(TextureCacheUsage().sampledEntries == capacity, "a second frame over other textures left " + std::to_string(TextureCacheUsage().sampledEntries) + " entries cached");
}

void scanTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    const auto working = 2u * capacity;
    Require(capacity >= 2u && working <= Surfaces && TextureCacheUsage().sampledEntries == 1u, "a snapshot of the 256 KiB surface holds " + std::to_string(bytes) + " bytes, which the test's 1 MiB budget does not fit as planned");
    constexpr std::size_t Presents = 6;
    constexpr std::size_t Uses = 2;
    std::vector<std::weak_ptr<Texture>> last(working);
    std::size_t hits = 0;
    std::size_t lookups = 0;
    for (std::size_t present = 0; present < Presents; ++present) {
        Recorder::CountPresent();
        for (std::size_t surface = 0; surface < working; ++surface) {
            for (std::size_t use = 0; use < Uses; ++use) {
                const auto texture = Sampled(context, block.Surface(surface));
                if (present != 0u) {
                    hits += texture == last[surface].lock();
                    ++lookups;
                }
                last[surface] = texture;
            }
        }
    }
    Require(TextureCacheUsage().sampledBytes <= Budget, "the sampled texture cache holds more than its budget after a scan");
    const auto wanted = (Presents - 1u) * Uses * capacity / 2u;
    Require(hits >= wanted, "a cyclic scan using each of " + std::to_string(working) + " textures twice in a row, twice the " + std::to_string(capacity) + " the budget holds, found " + std::to_string(hits) + " of " + std::to_string(lookups) + " lookups cached, fewer than " + std::to_string(wanted));
}

void shrinkTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    OverrideCacheBudgetForTest(2u * Budget);
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>((2u * Budget) / bytes);
    Require(capacity >= 4u && capacity + 1u <= Surfaces, "a snapshot of the 256 KiB surface holds " + std::to_string(bytes) + " bytes, which the test's 2 MiB budget does not fit as planned");
    for (std::size_t surface = 1; surface < capacity; ++surface) Sampled(context, block.Surface(surface));
    Require(TextureCacheUsage().sampledEntries == capacity, "the 2 MiB budget did not keep " + std::to_string(capacity) + " snapshots");
    Recorder::CountPresent();
    OverrideCacheBudgetForTest(Budget);
    for (std::size_t surface = 0; surface < capacity; ++surface) Sampled(context, block.Surface(surface));
    Sampled(context, block.Surface(capacity));
    Require(TextureCacheUsage().sampledBytes <= Ceiling, "after the budget fell to 1 MiB, one insert left " + std::to_string(TextureCacheUsage().sampledBytes) + " bytes cached with every entry in use this frame, past a quarter over the budget");
    OverrideCacheBudgetForTest(0);
}

void secondUseTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    Require(capacity >= 2u && capacity < Surfaces && TextureCacheUsage().sampledEntries == 1u, "a snapshot of the 256 KiB surface holds " + std::to_string(bytes) + " bytes, which the test's 1 MiB budget does not fit as planned");
    for (std::size_t surface = 1; surface < capacity; ++surface) Sampled(context, block.Surface(surface));
    Recorder::CountPresent();
    auto first = Sampled(context, block.Surface(capacity));
    Require(TextureCacheUsage().sampledEntries == capacity && first.use_count() == 1, "the first use in a frame of a texture the cache does not hold was cached in place of one the last frame used");
    const auto second = Sampled(context, block.Surface(capacity));
    Require(Sampled(context, block.Surface(capacity)) == second, "a texture used a second time in a frame whose least recently used entry is from the last frame was not cached");
    Require(TextureCacheUsage().sampledEntries == capacity && TextureCacheUsage().sampledBytes <= Budget, "the texture admitted on its second use left the cache over its budget");
}

void earlyMissTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    Require(capacity >= 2u && capacity < Surfaces && TextureCacheUsage().sampledEntries == 1u, "a snapshot of the 256 KiB surface holds " + std::to_string(bytes) + " bytes, which the test's 1 MiB budget does not fit as planned");
    std::vector<std::weak_ptr<Texture>> cached(capacity);
    for (std::size_t surface = 0; surface < capacity; ++surface) cached[surface] = Sampled(context, block.Surface(surface));
    Recorder::CountPresent();
    for (std::size_t surface = 0; surface < capacity; ++surface) {
        const auto again = Sampled(context, block.Surface(surface));
        Require(again == cached[surface].lock(), "surface " + std::to_string(surface) + " missed in a frame over the textures the cache holds");
    }
    Recorder::CountPresent();
    Sampled(context, block.Surface(capacity));
    std::size_t hits = 0;
    for (std::size_t surface = 0; surface < capacity; ++surface) {
        const auto again = Sampled(context, block.Surface(surface));
        hits += again == cached[surface].lock();
    }
    Require(hits == capacity, "a frame that first uses a new texture before its " + std::to_string(capacity) + " cached ones found " + std::to_string(hits) + " of them cached");
}

constexpr std::uint64_t KeyHashBasis = 14695981039346656037ull;
constexpr std::uint64_t KeyHashPrime = 1099511628211ull;

std::array<std::array<std::uint32_t, 8>, 2> CollidingDescriptors(VkDevice device, std::uint64_t address) {
    const auto words = Descriptor(address);
    auto prefix = KeyHashBasis ^ reinterpret_cast<std::uintptr_t>(device);
    for (std::size_t word = 0; word < 6; ++word) prefix = (prefix ^ words[word]) * KeyHashPrime;
    std::unordered_map<std::uint32_t, std::uint32_t> word6ByHighHalf;
    for (std::uint32_t candidate = 0; candidate < 0x200u; ++candidate) {
        const auto word6 = ((candidate >> 1u) << 24u) | (candidate & 1u);
        const auto hash = (prefix ^ word6) * KeyHashPrime;
        const auto [match, added] = word6ByHighHalf.emplace(static_cast<std::uint32_t>(hash >> 32u), word6);
        if (added) continue;
        auto first = words;
        auto second = words;
        first[6] = match->second;
        second[6] = word6;
        second[7] = static_cast<std::uint32_t>(hash ^ ((prefix ^ match->second) * KeyHashPrime));
        return {first, second};
    }
    throw std::runtime_error("no two descriptors of the surface that differ only in word 6's statistics counter and the unused meta address collide in TextureKeyHash");
}

void collidingKeyTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    Require(capacity >= 2u && capacity < Surfaces && TextureCacheUsage().sampledEntries == 1u, "a snapshot of the 256 KiB surface holds " + std::to_string(bytes) + " bytes, which the test's 1 MiB budget does not fit as planned");
    std::vector<std::weak_ptr<Texture>> cached(capacity);
    for (std::size_t surface = 0; surface < capacity; ++surface) cached[surface] = Sampled(context, block.Surface(surface));
    Recorder::CountPresent();
    const auto [first, second] = CollidingDescriptors(context.device, block.Surface(capacity));
    CachedSampledTexture(context, first);
    CachedSampledTexture(context, second);
    std::size_t kept = 0;
    for (const auto& texture : cached) kept += !texture.expired();
    Require(kept == capacity && TextureCacheUsage().sampledEntries == capacity, "the first use of a descriptor whose cache key hash equals that of another descriptor first used in the frame left " + std::to_string(kept) + " of the " + std::to_string(capacity) + " textures the last frame used cached");
}

constexpr std::uint64_t ReportedBudget = 64ull << 30u;
constexpr std::uint64_t ReportedUsage = 12ull << 30u;

void VKAPI_PTR ReportBudget(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2* properties) {
    auto* reported = static_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(properties->pNext);
    for (std::uint32_t heap = 0; heap < VK_MAX_MEMORY_HEAPS; ++heap) {
        reported->heapBudget[heap] = ReportedBudget;
        reported->heapUsage[heap] = ReportedUsage;
    }
}

std::uint64_t ExpectedBudget(const Context& context) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &reported};
    ReportBudget(context.physical, &properties);
    const auto use = TextureCacheUsage();
    return TextureCacheBudgetsFor(context.memory, &reported, {SampledTextureMemory(), use.sampledBytes, use.storageBytes}, CurrentTextureCacheBudgets(context).target).sampled;
}

std::uint64_t ReadBudget(const Context& context) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    return SampledTextureCacheBudget(context);
}

constexpr std::uint64_t PressedUsage = ReportedBudget - 11ull * (1ull << 30u);

void VKAPI_PTR ReportPressure(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2* properties) {
    auto* reported = static_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(properties->pNext);
    for (std::uint32_t heap = 0; heap < VK_MAX_MEMORY_HEAPS; ++heap) {
        reported->heapBudget[heap] = ReportedBudget;
        reported->heapUsage[heap] = PressedUsage;
    }
}

std::uint64_t ReadStorageBudget(const Context& context) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    return StorageTextureCacheBudget(context);
}

void memoryTests(Context context) {
    context.memoryProperties2 = &ReportBudget;
    const auto before = SampledTextureMemory();
    Block block(context);
    auto held = Sampled(context, block.Surface(0));
    Require(!held->ViewsStorageImage() && held->AllocationBytes() != 0, "the test texture is not a snapshot with an image of its own");
    Require(SampledTextureMemory() == before + held->AllocationBytes(), "the sampled texture memory grew by " + std::to_string(SampledTextureMemory() - before) + " bytes, not by the snapshot's allocation of " + std::to_string(held->AllocationBytes()));
    Require(ReadBudget(context) == ExpectedBudget(context), "the sampled texture cache budget does not count its cached texture as the cache's own memory");
    ClearCachedTextures(context.device);
    Require(TextureCacheUsage().sampledEntries == 0 && SampledTextureMemory() == before + held->AllocationBytes(), "a texture its caller still holds stopped counting as sampled texture memory when it left the cache");
    const auto budget = ReadBudget(context);
    Require(budget == ExpectedBudget(context), "the sampled texture cache budget does not count a texture its caller still holds outside the cache against the cache");
    std::printf("reported heap budget %llu MiB with %llu MiB in use: sampled texture cache budget %llu MiB with a %llu KiB texture held outside the cache\n", static_cast<unsigned long long>(ReportedBudget >> 20u), static_cast<unsigned long long>(ReportedUsage >> 20u), static_cast<unsigned long long>(budget >> 20u), static_cast<unsigned long long>(held->AllocationBytes() >> 10u));
    held.reset();
    Require(SampledTextureMemory() == before, "a released texture still counts as sampled texture memory");
}

// A device-local heap that is full past `limit` bytes of device-local allocations made through the
// wrapped device functions (the test's stand-in for VRAM running out under a burst).
struct HeapLimit {
    PFN_vkGetDeviceProcAddr real = nullptr;
    PFN_vkAllocateMemory allocate = nullptr;
    PFN_vkFreeMemory free = nullptr;
    VkPhysicalDeviceMemoryProperties memory{};
    std::mutex mutex;
    std::unordered_map<VkDeviceMemory, VkDeviceSize> local;
    VkDeviceSize used = 0;
    VkDeviceSize limit = ~VkDeviceSize{0};
    std::uint64_t refused = 0;
};

HeapLimit& Limit() {
    static HeapLimit limit;
    return limit;
}

VkResult VKAPI_PTR LimitedAllocate(VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory) {
    auto& limit = Limit();
    const bool local = (limit.memory.memoryTypes[info->memoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    {
        std::lock_guard lock(limit.mutex);
        if (local && limit.used + info->allocationSize > limit.limit) {
            ++limit.refused;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }
    const auto result = limit.allocate(device, info, callbacks, memory);
    if (result == VK_SUCCESS && local) {
        std::lock_guard lock(limit.mutex);
        limit.local[*memory] = info->allocationSize;
        limit.used += info->allocationSize;
    }
    return result;
}

void VKAPI_PTR LimitedFree(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks) {
    auto& limit = Limit();
    {
        std::lock_guard lock(limit.mutex);
        if (const auto found = limit.local.find(memory); found != limit.local.end()) {
            limit.used -= found->second;
            limit.local.erase(found);
        }
    }
    limit.free(device, memory, callbacks);
}

PFN_vkVoidFunction VKAPI_PTR LimitedProc(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkAllocateMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&LimitedAllocate);
    if (std::strcmp(name, "vkFreeMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&LimitedFree);
    return Limit().real(device, name);
}

// A loading-screen burst (lean-x21): one build holds mipmapped snapshots (no linear image in system
// memory for them) far past the sampled share while the device-local heap fills, then the build
// allocates a device buffer. The snapshots past the share go to the overflow type, a snapshot the
// full heap refuses does too, and so does the buffer, instead of the build failing.
int burstOverflowTests(Context context) {
    if (!OverflowMemoryType(context.memory, ~0u)) {
        std::puts("skipped: the device has no memory type outside its device-local heaps");
        return VulkanTestSkipped;
    }
    auto& limit = Limit();
    limit.real = context.deviceProc;
    limit.allocate = reinterpret_cast<PFN_vkAllocateMemory>(context.deviceProc(context.device, "vkAllocateMemory"));
    limit.free = reinterpret_cast<PFN_vkFreeMemory>(context.deviceProc(context.device, "vkFreeMemory"));
    limit.memory = context.memory;
    context.deviceProc = &LimitedProc;
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    constexpr std::size_t Stride = std::size_t{1} << 20u;
    constexpr std::size_t Count = 24;
    constexpr std::uint32_t LastMip = 8;
#ifdef _WIN32
    void* memory = VirtualAlloc(nullptr, Stride * Count, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* memory = std::aligned_alloc(65536, Stride * Count);
#endif
    Require(memory != nullptr, "cannot allocate the burst's surfaces");
    std::memset(memory, 0x5a, Stride * Count);
    GuestAllocations::Mutation().Add(memory, Stride * Count, true, true, true);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto before = SampledTextureMemory();
    std::vector<std::shared_ptr<Texture>> held;
    std::size_t overflowed = 0, system = 0;
    {
        std::lock_guard lock(limit.mutex);
        limit.limit = limit.used + 2 * Stride;
    }
    for (std::size_t surface = 0; surface < Count; ++surface) {
        auto words = Descriptor(base + surface * Stride);
        words[3] |= LastMip << 16u;
        words[5] |= LastMip << 4u;
        held.push_back(CachedSampledTexture(context, words));
        overflowed += held.back()->Spilled() && !held.back()->InHostMemory();
        system += held.back()->InHostMemory();
    }
    std::uint64_t counted = 0;
    for (const auto& texture : held) counted += texture->InHostMemory() ? 0u : texture->AllocationBytes();
    const auto sampled = SampledTextureMemory() - before;
    {
        std::lock_guard lock(limit.mutex);
        limit.limit = limit.used;
    }
    const auto refusedBefore = limit.refused;
    DeviceBuffer buffer(context, 2621440, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::printf("a build holding %zu mipmapped snapshots past a 1 MiB budget with a full device-local heap: %zu in the overflow type, %zu in system memory, %llu KiB counted as device-local sampled memory, %llu device-local allocations refused; a 2.5 MiB device buffer allocated after %llu refusals\n", Count, overflowed, system, static_cast<unsigned long long>(sampled >> 10u), static_cast<unsigned long long>(refusedBefore), static_cast<unsigned long long>(limit.refused - refusedBefore));
    Require(system == 0, "a mipmapped snapshot went to system memory");
    Require(overflowed + 4u >= Count, "only " + std::to_string(overflowed) + " of the " + std::to_string(Count) + " snapshots past the share went to the overflow type");
    Require(sampled == counted, "the snapshots in the overflow type are not counted as device-local sampled memory");
    Require(limit.refused > refusedBefore, "the device buffer did not meet the full heap");
    {
        std::lock_guard lock(limit.mutex);
        limit.limit = ~VkDeviceSize{0};
    }
    held.clear();
    std::this_thread::sleep_for(std::chrono::milliseconds(60));

    // Builds of a burst bind the same image table again and again: the spilled snapshots are kept
    // for reuse, so each surface is made once however many builds bind it, and the kept ones nothing
    // uses go down to the overflow budget (APS5_TEXTURE_OVERFLOW_MIB=4 for this test; the trim that
    // found everything in use above is repeated after 50 ms without a recorder retiring batches).
    constexpr std::size_t Table = 8;
    constexpr std::size_t Builds = 5;
    const auto first = SampledCacheCounters();
    std::uint64_t afterFirstBuild = 0;
    for (std::size_t build = 0; build < Builds; ++build) {
        std::vector<std::shared_ptr<Texture>> bound;
        for (std::size_t surface = 0; surface < Table; ++surface) {
            auto words = Descriptor(base + surface * Stride);
            words[3] |= LastMip << 16u;
            words[5] |= LastMip << 4u;
            bound.push_back(CachedSampledTexture(context, words));
        }
        if (build == 0) afterFirstBuild = SampledTextureOverflowMemory();
    }
    const auto last = SampledCacheCounters();
    const auto budget = CurrentTextureCacheBudgets(context).overflow;
    std::printf("%zu builds binding the same %zu mipmapped snapshots past the share: %llu made, %llu spilled ones reused, %llu kept ones dropped; %llu KiB in the overflow type after the first build (budget %llu KiB)\n", Builds, Table, static_cast<unsigned long long>(last.made - first.made), static_cast<unsigned long long>(last.spilledReuses - first.spilledReuses), static_cast<unsigned long long>(last.overflowDropped - first.overflowDropped), static_cast<unsigned long long>(afterFirstBuild >> 10u), static_cast<unsigned long long>(budget >> 10u));
    Require(last.made - first.made <= Table, std::to_string(last.made - first.made) + " snapshots were made for " + std::to_string(Builds) + " builds binding the same " + std::to_string(Table) + " surfaces");
    Require(last.spilledReuses - first.spilledReuses >= (Builds - 1u) * (Table - 4u), "the builds after the first reused only " + std::to_string(last.spilledReuses - first.spilledReuses) + " spilled snapshots");
    Require(last.overflowDropped > first.overflowDropped, "the burst's kept snapshots nothing used were not dropped down to the overflow budget");
    Require(budget == 4ull << 20u && afterFirstBuild <= budget + Stride, "the overflow type held " + std::to_string(afterFirstBuild >> 10u) + " KiB after the first build, past its " + std::to_string(budget >> 10u) + " KiB budget");
    ClearCachedTextures(context.device);
    GuestAllocations::Mutation().Remove(memory);
#ifdef _WIN32
    VirtualFree(memory, 0, MEM_RELEASE);
#else
    std::free(memory);
#endif
    std::puts("sampled texture burst overflow tests passed");
    return 0;
}

// Use-after-free watch on the device functions (no validation layer on this machine): which
// images and views each command buffer names (copies, clears, barriers, bound descriptor sets) from
// its begin until it is reset, freed or its submission's fence signals, and every image or view
// destroyed while a command buffer naming it is still pending.
struct HandleWatch {
    PFN_vkGetDeviceProcAddr real = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    std::mutex mutex;
    struct Commands {
        std::unordered_set<std::uint64_t> handles;
        bool submitted = false;
        VkFence fence = VK_NULL_HANDLE;
        std::uint64_t epoch = 0;
        bool idle = false;
        VkCommandPool pool = VK_NULL_HANDLE;
    };
    std::unordered_map<VkFence, std::uint64_t> fenceEpochs;
    std::unordered_map<VkCommandBuffer, Commands> commands;
    std::unordered_map<VkDescriptorSet, std::vector<std::uint64_t>> sets;
    std::uint64_t destroyed = 0;
    std::uint64_t violations = 0;
    std::string first;
    template<typename F>
    F at(const char* name) const { return reinterpret_cast<F>(real(device, name)); }
    bool pending(const Commands& state) const {
        if (!state.submitted) return true;
        if (state.idle) return false;
        if (state.fence == VK_NULL_HANDLE) return true;
        // A fence submitted again was reset first, which the earlier submission had to finish for.
        if (const auto found = fenceEpochs.find(state.fence); found != fenceEpochs.end() && found->second != state.epoch) return false;
        return at<PFN_vkGetFenceStatus>("vkGetFenceStatus")(device, state.fence) != VK_SUCCESS;
    }
    void name(VkCommandBuffer buffer, std::uint64_t handle) {
        std::lock_guard lock(mutex);
        commands[buffer].handles.insert(handle);
    }
    void destroying(std::uint64_t handle, const char* what) {
        std::lock_guard lock(mutex);
        ++destroyed;
        for (const auto& [buffer, state] : commands) {
            if (!state.handles.contains(handle) || !pending(state)) continue;
            if (violations++ == 0) {
                const auto status = state.fence != VK_NULL_HANDLE ? static_cast<int>(at<PFN_vkGetFenceStatus>("vkGetFenceStatus")(device, state.fence)) : 99;
                first = std::string(what) + " destroyed while a " + (state.submitted ? "submitted, unfinished" : "recorded, unsubmitted") + " command buffer names it (handle " + std::to_string(handle) + ", fence status " + std::to_string(status) + ", epoch " + std::to_string(state.epoch) + ", " + std::to_string(state.handles.size()) + " handles named)";
                if (std::getenv("APS5_WATCH_ABORT") != nullptr) {
                    std::fputs(first.c_str(), stderr);
                    std::abort();
                }
            }
            return;
        }
    }
    void reset(VkCommandBuffer buffer) {
        std::lock_guard lock(mutex);
        commands.erase(buffer);
    }
};

HandleWatch& Watch() {
    static HandleWatch watch;
    return watch;
}

template<typename T>
std::uint64_t Handle(T handle) {
    return reinterpret_cast<std::uint64_t>(handle);
}

VkResult VKAPI_PTR WatchedAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* buffers) {
    const auto result = Watch().at<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(device, info, buffers);
    if (result == VK_SUCCESS) {
        std::lock_guard lock(Watch().mutex);
        for (std::uint32_t i = 0; i < info->commandBufferCount; ++i) Watch().commands[buffers[i]] = {{}, false, VK_NULL_HANDLE, 0, false, info->commandPool};
    }
    return result;
}

VkResult VKAPI_PTR WatchedBegin(VkCommandBuffer buffer, const VkCommandBufferBeginInfo* info) {
    {
        std::lock_guard lock(Watch().mutex);
        auto& state = Watch().commands[buffer];
        const auto pool = state.pool;
        state = {{}, false, VK_NULL_HANDLE, 0, false, pool};
    }
    return Watch().at<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(buffer, info);
}

VkResult VKAPI_PTR WatchedResetCommandBuffer(VkCommandBuffer buffer, VkCommandBufferResetFlags flags) {
    {
        std::lock_guard lock(Watch().mutex);
        auto& state = Watch().commands[buffer];
        state.handles.clear();
        state.submitted = true;
        state.idle = true;
    }
    return Watch().at<PFN_vkResetCommandBuffer>("vkResetCommandBuffer")(buffer, flags);
}

void VKAPI_PTR WatchedFreeCommandBuffers(VkDevice device, VkCommandPool pool, std::uint32_t count, const VkCommandBuffer* buffers) {
    for (std::uint32_t i = 0; i < count; ++i) Watch().reset(buffers[i]);
    Watch().at<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(device, pool, count, buffers);
}

VkResult VKAPI_PTR WatchedResetCommandPool(VkDevice device, VkCommandPool pool, VkCommandPoolResetFlags flags) {
    {
        std::lock_guard lock(Watch().mutex);
        for (auto& [buffer, state] : Watch().commands) {
            if (state.pool != pool) continue;
            state.handles.clear();
            state.submitted = true;
            state.idle = true;
        }
    }
    return Watch().at<PFN_vkResetCommandPool>("vkResetCommandPool")(device, pool, flags);
}

VkResult VKAPI_PTR WatchedSubmit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
    {
        std::lock_guard lock(Watch().mutex);
        const auto epoch = fence != VK_NULL_HANDLE ? ++Watch().fenceEpochs[fence] : 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            for (std::uint32_t j = 0; j < submits[i].commandBufferCount; ++j) {
                auto& state = Watch().commands[submits[i].pCommandBuffers[j]];
                state.submitted = true;
                state.fence = fence;
                state.epoch = epoch;
                state.idle = false;
            }
        }
    }
    return Watch().at<PFN_vkQueueSubmit>("vkQueueSubmit")(queue, count, submits, fence);
}

// A fence the device waited for or reset (which needs its submission finished) completes the
// command buffers submitted with it.
void FenceDone(std::uint32_t count, const VkFence* fences) {
    std::lock_guard lock(Watch().mutex);
    for (auto& [buffer, state] : Watch().commands) {
        if (!state.submitted || state.fence == VK_NULL_HANDLE) continue;
        for (std::uint32_t i = 0; i < count; ++i) {
            if (fences[i] == state.fence) state.idle = true;
        }
    }
}

VkResult VKAPI_PTR WatchedWaitForFences(VkDevice device, std::uint32_t count, const VkFence* fences, VkBool32 all, std::uint64_t timeout) {
    const auto result = Watch().at<PFN_vkWaitForFences>("vkWaitForFences")(device, count, fences, all, timeout);
    if (result == VK_SUCCESS && (all || count == 1)) FenceDone(count, fences);
    return result;
}

VkResult VKAPI_PTR WatchedResetFences(VkDevice device, std::uint32_t count, const VkFence* fences) {
    FenceDone(count, fences);
    return Watch().at<PFN_vkResetFences>("vkResetFences")(device, count, fences);
}

VkResult VKAPI_PTR WatchedQueueWaitIdle(VkQueue queue) {
    const auto result = Watch().at<PFN_vkQueueWaitIdle>("vkQueueWaitIdle")(queue);
    std::lock_guard lock(Watch().mutex);
    for (auto& [buffer, state] : Watch().commands) {
        if (state.submitted) state.idle = true;
    }
    return result;
}

void VKAPI_PTR WatchedCopyBufferToImage(VkCommandBuffer buffer, VkBuffer source, VkImage image, VkImageLayout layout, std::uint32_t count, const VkBufferImageCopy* regions) {
    Watch().name(buffer, Handle(image));
    Watch().at<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(buffer, source, image, layout, count, regions);
}

void VKAPI_PTR WatchedClearColorImage(VkCommandBuffer buffer, VkImage image, VkImageLayout layout, const VkClearColorValue* color, std::uint32_t count, const VkImageSubresourceRange* ranges) {
    Watch().name(buffer, Handle(image));
    Watch().at<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(buffer, image, layout, color, count, ranges);
}

void VKAPI_PTR WatchedBarrier(VkCommandBuffer buffer, VkPipelineStageFlags from, VkPipelineStageFlags to, VkDependencyFlags flags, std::uint32_t memoryCount, const VkMemoryBarrier* memory, std::uint32_t bufferCount, const VkBufferMemoryBarrier* buffers, std::uint32_t imageCount, const VkImageMemoryBarrier* images) {
    for (std::uint32_t i = 0; i < imageCount; ++i) Watch().name(buffer, Handle(images[i].image));
    Watch().at<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(buffer, from, to, flags, memoryCount, memory, bufferCount, buffers, imageCount, images);
}

void VKAPI_PTR WatchedUpdateDescriptorSets(VkDevice device, std::uint32_t writeCount, const VkWriteDescriptorSet* writes, std::uint32_t copyCount, const VkCopyDescriptorSet* copies) {
    {
        std::lock_guard lock(Watch().mutex);
        for (std::uint32_t i = 0; i < writeCount; ++i) {
            if (writes[i].pImageInfo == nullptr) continue;
            auto& views = Watch().sets[writes[i].dstSet];
            for (std::uint32_t j = 0; j < writes[i].descriptorCount; ++j) views.push_back(Handle(writes[i].pImageInfo[j].imageView));
        }
    }
    Watch().at<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(device, writeCount, writes, copyCount, copies);
}

void VKAPI_PTR WatchedBindDescriptorSets(VkCommandBuffer buffer, VkPipelineBindPoint point, VkPipelineLayout layout, std::uint32_t first, std::uint32_t count, const VkDescriptorSet* sets, std::uint32_t dynamicCount, const std::uint32_t* dynamic) {
    {
        std::lock_guard lock(Watch().mutex);
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto found = Watch().sets.find(sets[i]);
            if (found == Watch().sets.end()) continue;
            for (const auto view : found->second) Watch().commands[buffer].handles.insert(view);
        }
    }
    Watch().at<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(buffer, point, layout, first, count, sets, dynamicCount, dynamic);
}

void VKAPI_PTR WatchedDestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* callbacks) {
    Watch().destroying(Handle(image), "an image");
    Watch().at<PFN_vkDestroyImage>("vkDestroyImage")(device, image, callbacks);
}

void VKAPI_PTR WatchedDestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks* callbacks) {
    Watch().destroying(Handle(view), "an image view");
    Watch().at<PFN_vkDestroyImageView>("vkDestroyImageView")(device, view, callbacks);
}

PFN_vkVoidFunction VKAPI_PTR WatchedProc(VkDevice device, const char* name) {
    static const std::unordered_map<std::string, PFN_vkVoidFunction> table{
        {"vkAllocateCommandBuffers", reinterpret_cast<PFN_vkVoidFunction>(&WatchedAllocateCommandBuffers)},
        {"vkBeginCommandBuffer", reinterpret_cast<PFN_vkVoidFunction>(&WatchedBegin)},
        {"vkResetCommandBuffer", reinterpret_cast<PFN_vkVoidFunction>(&WatchedResetCommandBuffer)},
        {"vkFreeCommandBuffers", reinterpret_cast<PFN_vkVoidFunction>(&WatchedFreeCommandBuffers)},
        {"vkResetCommandPool", reinterpret_cast<PFN_vkVoidFunction>(&WatchedResetCommandPool)},
        {"vkQueueSubmit", reinterpret_cast<PFN_vkVoidFunction>(&WatchedSubmit)},
        {"vkQueueWaitIdle", reinterpret_cast<PFN_vkVoidFunction>(&WatchedQueueWaitIdle)},
        {"vkWaitForFences", reinterpret_cast<PFN_vkVoidFunction>(&WatchedWaitForFences)},
        {"vkResetFences", reinterpret_cast<PFN_vkVoidFunction>(&WatchedResetFences)},
        {"vkCmdCopyBufferToImage", reinterpret_cast<PFN_vkVoidFunction>(&WatchedCopyBufferToImage)},
        {"vkCmdClearColorImage", reinterpret_cast<PFN_vkVoidFunction>(&WatchedClearColorImage)},
        {"vkCmdPipelineBarrier", reinterpret_cast<PFN_vkVoidFunction>(&WatchedBarrier)},
        {"vkUpdateDescriptorSets", reinterpret_cast<PFN_vkVoidFunction>(&WatchedUpdateDescriptorSets)},
        {"vkCmdBindDescriptorSets", reinterpret_cast<PFN_vkVoidFunction>(&WatchedBindDescriptorSets)},
        {"vkDestroyImage", reinterpret_cast<PFN_vkVoidFunction>(&WatchedDestroyImage)},
        {"vkDestroyImageView", reinterpret_cast<PFN_vkVoidFunction>(&WatchedDestroyImageView)},
    };
    if (const auto found = table.find(name); found != table.end()) return found->second;
    return Watch().real(device, name);
}

// lean-x25 lost the device in a burst with kept spilled snapshots being dropped. Kept snapshots a
// build let go of while their uploads (or uses) are still in the recorder's open batch must
// survive the overflow trim and the out-of-memory reclaim; once that batch retired they go, and no
// image or view is ever destroyed while a recorded or submitted command buffer names it.
int overflowLifetimeTests(Context context) {
    if (!OverflowMemoryType(context.memory, ~0u)) {
        std::puts("skipped: the device has no memory type outside its device-local heaps");
        return VulkanTestSkipped;
    }
    auto& watch = Watch();
    watch.real = context.deviceProc;
    watch.device = context.device;
    context.deviceProc = &WatchedProc;
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    constexpr std::size_t Stride = std::size_t{1} << 20u;
    constexpr std::size_t Count = 32;
    constexpr std::size_t Build = 12;
    constexpr std::uint32_t LastMip = 8;
#ifdef _WIN32
    void* memory = VirtualAlloc(nullptr, Stride * Count, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* memory = std::aligned_alloc(65536, Stride * Count);
#endif
    Require(memory != nullptr, "cannot allocate the surfaces");
    std::memset(memory, 0x5a, Stride * Count);
    GuestAllocations::Mutation().Add(memory, Stride * Count, true, true, true);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto lookup = [&](std::size_t surface) {
        auto words = Descriptor(base + surface * Stride);
        words[3] |= LastMip << 16u;
        words[5] |= LastMip << 4u;
        return CachedSampledTexture(context, words);
    };
    {
        Recorder recorder(context);
        recorder.Activate();
        const auto retire = [&] {
            recorder.Submit();
            Check(context.Function<PFN_vkQueueWaitIdle>("vkQueueWaitIdle")(context.queue), "vkQueueWaitIdle");
            while (recorder.RetireOldest()) {
            }
        };
        std::vector<std::weak_ptr<Texture>> first;
        {
            std::vector<std::shared_ptr<Texture>> build;
            for (std::size_t surface = 0; surface < Build; ++surface) build.push_back(lookup(surface));
            for (const auto& texture : build) first.push_back(texture);
        }
        Require(recorder.Recording(), "the first build's uploads are not in an open batch");
        const auto before = SampledCacheCounters();
        AgcDriver::Graphics::ReclaimTextureMemory(context.device, VkDeviceSize{1} << 30u);
        std::size_t reclaimedAlive = 0;
        for (const auto& texture : first) reclaimedAlive += !texture.expired();
        Require(reclaimedAlive == Build && SampledCacheCounters().overflowDropped == before.overflowDropped, "the out-of-memory reclaim dropped kept snapshots whose uploads are in the open batch");
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        {
            std::vector<std::shared_ptr<Texture>> build;
            for (std::size_t surface = Build; surface < 2 * Build; ++surface) build.push_back(lookup(surface));
        }
        AgcDriver::Graphics::ReclaimTextureMemory(context.device, VkDeviceSize{1} << 30u);
        std::size_t alive = 0;
        for (const auto& texture : first) alive += !texture.expired();
        const auto held = SampledCacheCounters();
        const auto waits = held.overflowWaits - before.overflowWaits;
        std::printf("a build let go of %zu kept spilled snapshots before its batch was submitted; a second build passed the 4 MiB overflow budget and the out-of-memory reclaim ran: %zu of them alive, %llu kept ones dropped, %llu batches waited for, %llu KiB in the overflow type\n", Build, alive, static_cast<unsigned long long>(held.overflowDropped - before.overflowDropped), static_cast<unsigned long long>(waits), static_cast<unsigned long long>(SampledTextureOverflowMemory() >> 10u));
        Require(watch.violations == 0, watch.first);
        Require(alive == Build || waits != 0, "kept snapshots whose uploads were in the open batch were dropped without waiting for it");
        retire();
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        lookup(2 * Build);
        retire();
        lookup(2 * Build + 1);
        retire();
        const auto after = SampledCacheCounters();
        std::size_t left = 0;
        for (const auto& texture : first) left += !texture.expired();
        std::printf("after the batches retired: %llu kept ones dropped, %zu of the first build's left, %llu KiB in the overflow type; %llu images and views destroyed, %llu while a command buffer named them\n", static_cast<unsigned long long>(after.overflowDropped - before.overflowDropped), left, static_cast<unsigned long long>(SampledTextureOverflowMemory() >> 10u), static_cast<unsigned long long>(watch.destroyed), static_cast<unsigned long long>(watch.violations));
        Require(watch.violations == 0, watch.first);
        Require(after.overflowDropped > before.overflowDropped && left == 0, "the kept snapshots were not dropped once the batches using them retired");
        Require(SampledTextureOverflowMemory() <= CurrentTextureCacheBudgets(context).overflow + Stride, "the overflow type stayed past its budget once nothing used the kept snapshots");
        ClearCachedTextures(context.device);
        retire();
    }
    Require(watch.violations == 0, watch.first);
    GuestAllocations::Mutation().Remove(memory);
#ifdef _WIN32
    VirtualFree(memory, 0, MEM_RELEASE);
#else
    std::free(memory);
#endif
    std::puts("sampled texture overflow lifetime tests passed");
    return 0;
}

void storageMemoryTests(Context context) {
    Block block(context);
    auto held = Sampled(context, block.Surface(0));
    for (const auto report : {&ReportBudget, &ReportPressure}) {
        context.memoryProperties2 = report;
        const auto budget = ReadStorageBudget(context);
        VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &reported};
        report(context.physical, &properties);
        const auto use = TextureCacheUsage();
        const auto expected = TextureCacheBudgetsFor(context.memory, &reported, {SampledTextureMemory(), use.sampledBytes, use.storageBytes}, CurrentTextureCacheBudgets(context).target).storage;
        Require(budget == expected, "with " + std::to_string(reported.heapUsage[0] >> 20u) + " MiB in use of a 64 GiB budget the storage image cache budget is " + std::to_string(budget >> 20u) + " MiB, not the VRAM budget's storage share of " + std::to_string(expected >> 20u) + " MiB");
        std::printf("reported heap budget %llu MiB with %llu MiB in use: storage image cache budget %llu MiB\n", static_cast<unsigned long long>(ReportedBudget >> 20u), static_cast<unsigned long long>(reported.heapUsage[0] >> 20u), static_cast<unsigned long long>(budget >> 20u));
    }
}

}

int main(int argc, char** argv) {
    try {
        const bool memory = argc > 1 && std::string(argv[1]) == "memory";
        const std::string mode = argc > 1 ? argv[1] : "";
        const bool resources = mode == "resource-admission" || mode == "resource-oversized" || mode == "resource-clear";
        const bool burst = mode == "burst-overflow";
        const bool lifetime = mode == "overflow-lifetime";
        const bool disableStorageImageMultisample = mode == "without-storage-image-multisample";
#ifdef _WIN32
        if (!memory) _putenv_s("APS5_TEXTURE_CACHE_MIB", "1");
#else
        if (!memory) setenv("APS5_TEXTURE_CACHE_MIB", "1", 1);
#endif
        std::unique_ptr<Device> device;
        try {
            device = std::make_unique<Device>(burst || lifetime, !disableStorageImageMultisample);
        } catch (const std::exception& error) {
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) throw;
            std::printf("skipped, no usable Vulkan device: %s\n", error.what());
            return VulkanTestSkipped;
        }
        if (burst) {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            return burstOverflowTests(device->GetContext());
        }
        if (lifetime) {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            return overflowLifetimeTests(device->GetContext());
        }
        auto context = device->GetContext();
        if (disableStorageImageMultisample) {
            PlaceholderImages placeholders(context);
            for (std::uint32_t heap = ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding;
                 heap < ShaderRecompiler::RuntimeAbi::FirstImageBinding + ShaderRecompiler::RuntimeAbi::ImageBindingCount; ++heap) {
                const auto shape = PlaceholderImageShapeOf(heap);
                if (shape.samples == VK_SAMPLE_COUNT_1_BIT) continue;
                bool refused = false;
                try {
                    placeholders.View(heap);
                } catch (const std::runtime_error&) {
                    refused = true;
                }
                Require(refused, "a device without storage-image MSAA returned a placeholder for typed multisample storage heap " + std::to_string(heap));
            }
        }
        TextureDetiler detiler(context);
        context.detiler = &detiler;
        if (resources) {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            if (!resourceCacheTests(context, mode)) return VulkanTestSkipped;
            std::puts("resource cache tests passed");
            return 0;
        }
        if (memory) {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            memoryTests(context);
            storageMemoryTests(context);
            ClearCachedTextures(context.device);
            std::puts("sampled texture memory budget tests passed");
            return 0;
        }
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            reportedBudgetTests(context);
            snapshotTests(context);
            heldTests(context);
            releasedTests(context);
            rescanTests(context);
            viewEvictionTests(context);
            viewTests(context);
            if (std::getenv("APS5_SAMPLED_LATER_USE_REFUSAL") != nullptr) scanTests(context);
            frameTests(context);
            shrinkTests(context);
            secondUseTests(context);
            earlyMissTests(context);
            collidingKeyTests(context);
            snapshotChangeTests(context);
            burstTests(context);
            ClearCachedTextures(context.device);
        }
        std::puts("sampled texture cache budget tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
