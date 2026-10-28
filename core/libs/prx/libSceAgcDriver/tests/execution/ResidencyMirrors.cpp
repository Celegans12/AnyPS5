#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Residency.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyMirrors.hpp"
#include "ResidencyRead_comp_spv.h"
#include <SDL_loadso.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

using namespace AgcDriver::Graphics;

constexpr int Skipped = 77;
constexpr std::uint32_t GuestPages = 8;
constexpr std::uint32_t ChunkSlots = 4;
constexpr std::uint64_t GuestBase = 0x100000000ull;
constexpr std::uint64_t BasePage = GuestBase >> ResidencyPageShift;
constexpr std::uint32_t WordsPerSubpage = ResidencySubpageBytes / 4;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

struct SkipTest : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Device {
public:
    Device() {
#ifdef _WIN32
        library = SDL_LoadObject("vulkan-1.dll");
#else
        library = SDL_LoadObject("libvulkan.so.1");
#endif
        if (library == nullptr) throw SkipTest("cannot load Vulkan");
        try {
            create();
        } catch (...) {
            release();
            throw;
        }
    }

    ~Device() { release(); }

    const Context& Get() const { return context; }
    std::uint32_t Family() const { return family; }

private:
    template<typename TFunction>
    TFunction function(const char* name) const {
        const auto result = reinterpret_cast<TFunction>(instanceProc(instance, name));
        if (result == nullptr) throw SkipTest(std::string("missing Vulkan function ") + name);
        return result;
    }

    void create() {
        instanceProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_LoadFunction(library, "vkGetInstanceProcAddr"));
        if (instanceProc == nullptr) throw SkipTest("missing vkGetInstanceProcAddr");
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instanceInfo.pApplicationInfo = &application;
        if (function<PFN_vkCreateInstance>("vkCreateInstance")(&instanceInfo, nullptr, &instance) != VK_SUCCESS) throw SkipTest("vkCreateInstance failed");
        std::uint32_t count = 0;
        const auto enumerate = function<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
        Check(enumerate(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
        if (count == 0) throw SkipTest("no Vulkan device");
        std::vector<VkPhysicalDevice> devices(count);
        Check(enumerate(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
        const auto properties = function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
        int rank = -1;
        for (const auto physical : devices) {
            VkPhysicalDeviceProperties candidate{};
            properties(physical, &candidate);
            if (candidate.apiVersion < VK_API_VERSION_1_1) continue;
            const int candidateRank = candidate.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 : candidate.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : candidate.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? 1 : 0;
            if (candidateRank <= rank) continue;
            rank = candidateRank;
            context.physical = physical;
        }
        if (context.physical == VK_NULL_HANDLE) throw SkipTest("no Vulkan 1.1 device");
        const auto extensions = function<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
        Check(extensions(context.physical, nullptr, &count, nullptr), "vkEnumerateDeviceExtensionProperties");
        std::vector<VkExtensionProperties> available(count);
        Check(extensions(context.physical, nullptr, &count, available.data()), "vkEnumerateDeviceExtensionProperties");
        const std::array<const char*, 2> wanted{VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME};
        for (const auto* name : wanted) {
            const bool found = std::any_of(available.begin(), available.end(), [&](const VkExtensionProperties& extension) { return std::strcmp(extension.extensionName, name) == 0; });
            if (!found) throw SkipTest(std::string("the device lacks ") + name);
        }
        VkPhysicalDeviceBufferDeviceAddressFeaturesKHR address{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &address};
        function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(context.physical, &features);
        if (address.bufferDeviceAddress != VK_TRUE || features.features.shaderInt64 != VK_TRUE) throw SkipTest("no buffer device address or 64-bit integers in shaders");
        const auto queues = function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
        queues(context.physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        queues(context.physical, &count, families.data());
        while (family < count && (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) ++family;
        if (family == count) throw SkipTest("no compute queue");
        const float priority = 1;
        VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue.queueFamilyIndex = family;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        VkPhysicalDeviceBufferDeviceAddressFeaturesKHR enabledAddress{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR};
        enabledAddress.bufferDeviceAddress = VK_TRUE;
        VkPhysicalDeviceFeatures enabled{};
        enabled.shaderInt64 = VK_TRUE;
        VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enabledAddress};
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queue;
        deviceInfo.enabledExtensionCount = static_cast<std::uint32_t>(wanted.size());
        deviceInfo.ppEnabledExtensionNames = wanted.data();
        deviceInfo.pEnabledFeatures = &enabled;
        Check(function<PFN_vkCreateDevice>("vkCreateDevice")(context.physical, &deviceInfo, nullptr, &context.device), "vkCreateDevice");
        context.deviceProc = function<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
        function<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(context.physical, &context.memory);
        VkPhysicalDeviceProperties deviceProperties{};
        properties(context.physical, &deviceProperties);
        context.limits = deviceProperties.limits;
        context.bufferDeviceAddress = true;
        context.Function<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(context.device, family, 0, &context.queue);
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex = family;
        Check(context.Function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(context.device, &pool, nullptr, &context.pool), "vkCreateCommandPool");
    }

    void release() noexcept {
        if (context.device != VK_NULL_HANDLE) context.Function<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(context.device);
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
    std::uint32_t family = 0;
};

struct FaultLog {
    std::uintptr_t base = 0;
    std::uintptr_t bytes = 0;
    std::array<std::atomic<bool>, GuestPages> guarded{};
    std::array<std::atomic<std::uint64_t>, 4096> pages{};
    std::atomic<std::uint32_t> count{0};
};

FaultLog faultLog;

bool Protect(std::uint32_t page, bool readOnly) {
    auto* address = reinterpret_cast<void*>(faultLog.base + (std::uintptr_t{page} << ResidencyPageShift));
#ifdef _WIN32
    DWORD previous = 0;
    return VirtualProtect(address, ResidencyPageBytes, readOnly ? PAGE_READONLY : PAGE_READWRITE, &previous) != 0;
#else
    return mprotect(address, ResidencyPageBytes, readOnly ? PROT_READ : PROT_READ | PROT_WRITE) == 0;
#endif
}

void ProtectOrThrow(std::uint32_t page, bool readOnly) {
    if (!Protect(page, readOnly)) throw std::runtime_error("changing the protection of guest page " + std::to_string(page) + " failed");
}

bool TakeFault(std::uintptr_t address) {
    if (address < faultLog.base || address >= faultLog.base + faultLog.bytes) return false;
    const auto page = static_cast<std::uint32_t>((address - faultLog.base) >> ResidencyPageShift);
    if (!faultLog.guarded[page].exchange(false)) return false;
    if (!Protect(page, false)) return false;
    const auto slot = faultLog.count.fetch_add(1);
    if (slot < faultLog.pages.size()) faultLog.pages[slot] = BasePage + page;
    return true;
}

#ifdef _WIN32
LONG CALLBACK FaultHandler(EXCEPTION_POINTERS* exception) {
    const auto* record = exception->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 || record->ExceptionInformation[0] != 1) return EXCEPTION_CONTINUE_SEARCH;
    return TakeFault(record->ExceptionInformation[1]) ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}
#else
struct sigaction previousAction {};

void FaultHandler(int signal, siginfo_t* info, void* context) {
    if (TakeFault(reinterpret_cast<std::uintptr_t>(info->si_addr))) return;
    sigaction(signal, &previousAction, nullptr);
    raise(signal);
    static_cast<void>(context);
}
#endif

class GuestRange {
public:
    explicit GuestRange(const Context& context) : context(context) {
        bytes = std::size_t{GuestPages} << ResidencyPageShift;
#ifdef _WIN32
        host = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (host == nullptr) throw std::runtime_error("VirtualAlloc failed");
        handler = AddVectoredExceptionHandler(1, FaultHandler);
#else
        reserved = bytes + ResidencyPageBytes;
        auto* raw = mmap(nullptr, reserved, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) throw std::runtime_error("mmap failed");
        mapping = raw;
        host = reinterpret_cast<void*>((reinterpret_cast<std::uintptr_t>(raw) + ResidencyPageBytes - 1) & ~(std::uintptr_t{ResidencyPageBytes} - 1));
        struct sigaction action {};
        action.sa_sigaction = FaultHandler;
        action.sa_flags = SA_SIGINFO;
        sigemptyset(&action.sa_mask);
        sigaction(SIGSEGV, &action, &previousAction);
#endif
        faultLog.base = reinterpret_cast<std::uintptr_t>(host);
        faultLog.bytes = bytes;
        importHost();
    }

    ~GuestRange() {
        for (std::uint32_t page = 0; page < GuestPages; ++page) {
            if (faultLog.guarded[page].exchange(false)) ProtectOrThrow(page, false);
        }
        context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
        context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
#ifdef _WIN32
        RemoveVectoredExceptionHandler(handler);
        VirtualFree(host, 0, MEM_RELEASE);
#else
        sigaction(SIGSEGV, &previousAction, nullptr);
        munmap(mapping, reserved);
#endif
    }

    std::uint32_t* Words(std::uint64_t page) const {
        return reinterpret_cast<std::uint32_t*>(static_cast<std::byte*>(host) + ((page - BasePage) << ResidencyPageShift));
    }

    VkBuffer Buffer() const { return buffer; }
    VkDeviceAddress Address(std::uint64_t page) const { return address + ((page - BasePage) << ResidencyPageShift); }

private:
    void importHost() {
        VkMemoryHostPointerPropertiesEXT pointer{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        Check(context.Function<PFN_vkGetMemoryHostPointerPropertiesEXT>("vkGetMemoryHostPointerPropertiesEXT")(context.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &pointer), "vkGetMemoryHostPointerPropertiesEXT");
        VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &buffer), "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
        VkImportMemoryHostPointerInfoEXT import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
        import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        import.pHostPointer = host;
        const VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, &import, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
        allocation.allocationSize = bytes;
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits & pointer.memoryTypeBits, 0);
        Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory), "vkAllocateMemory (host pointer import)");
        Check(context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, buffer, memory, 0), "vkBindBufferMemory");
        VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        addressInfo.buffer = buffer;
        address = context.Function<PFN_vkGetBufferDeviceAddressKHR>("vkGetBufferDeviceAddressKHR")(context.device, &addressInfo);
    }

    const Context& context;
    std::size_t bytes = 0;
    void* host = nullptr;
#ifdef _WIN32
    void* handler = nullptr;
#else
    void* mapping = nullptr;
    std::size_t reserved = 0;
#endif
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
};

class Stream {
public:
    explicit Stream(const Context& context) : context(context) {}

    ~Stream() {
        Submit();
        WaitAll();
    }

    VkCommandBuffer Commands() {
        if (open != VK_NULL_HANDLE) return open;
        VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        info.commandPool = context.pool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        Check(context.Function<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(context.device, &info, &open), "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        Check(context.Function<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(open, &begin), "vkBeginCommandBuffer");
        openValue = ++last;
        return open;
    }

    ResidencyToken Current() {
        Commands();
        ResidencyToken token{};
        token[0] = openValue;
        return token;
    }

    ResidencyToken Newest() const {
        ResidencyToken token{};
        token[0] = last;
        return token;
    }

    void Submit() {
        if (open == VK_NULL_HANDLE) return;
        Check(context.Function<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(open), "vkEndCommandBuffer");
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkCreateFence>("vkCreateFence")(context.device, &fenceInfo, nullptr, &fence), "vkCreateFence");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &open;
        Check(context.Function<PFN_vkQueueSubmit>("vkQueueSubmit")(context.queue, 1, &submit, fence), "vkQueueSubmit");
        inflight.push_back({openValue, open, fence});
        open = VK_NULL_HANDLE;
    }

    bool Reached(const ResidencyToken& token) {
        for (std::size_t stream = 1; stream < ResidencyStreams; ++stream) {
            if (token[stream] != 0) return false;
        }
        poll(false);
        return token[0] <= completed;
    }

    void Wait(const ResidencyToken& token) {
        if (open != VK_NULL_HANDLE && openValue <= token[0]) Submit();
        while (completed < token[0] && !inflight.empty()) {
            const auto fence = inflight.front().fence;
            Check(context.Function<PFN_vkWaitForFences>("vkWaitForFences")(context.device, 1, &fence, VK_TRUE, 10'000'000'000ull), "vkWaitForFences");
            poll(true);
        }
    }

    void WaitAll() { Wait(Newest()); }

private:
    struct Inflight {
        std::uint64_t value = 0;
        VkCommandBuffer commands = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
    };

    void poll(bool frontSignaled) {
        while (!inflight.empty()) {
            auto& front = inflight.front();
            if (!frontSignaled && context.Function<PFN_vkGetFenceStatus>("vkGetFenceStatus")(context.device, front.fence) != VK_SUCCESS) return;
            frontSignaled = false;
            completed = front.value;
            context.Function<PFN_vkDestroyFence>("vkDestroyFence")(context.device, front.fence, nullptr);
            context.Function<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(context.device, context.pool, 1, &front.commands);
            inflight.pop_front();
        }
    }

    const Context& context;
    VkCommandBuffer open = VK_NULL_HANDLE;
    std::uint64_t openValue = 0;
    std::uint64_t last = 0;
    std::uint64_t completed = 0;
    std::deque<Inflight> inflight;
};

class FixtureDevice final : public IResidencyDevice {
public:
    FixtureDevice(const Context& context, Stream& stream, GuestRange& guest) : stream(stream), guest(guest), mirrors(context, ChunkSlots) {}

    bool Reached(const ResidencyToken& token) override { return stream.Reached(token); }
    void Wait(const ResidencyToken& token) override { stream.Wait(token); }
    ResidencyToken Newest() override { return stream.Newest(); }
    ResidencyToken Current() override { return stream.Current(); }

    ResidencyImport Import(std::uint64_t page) override {
        if (page < BasePage || page >= BasePage + GuestPages || unmapped.contains(page)) return {};
        ResidencyImport import{true, true, true, false, 1, (page - BasePage) << ResidencyPageShift};
        import.resident = residentViews.contains(page);
        return import;
    }

    std::vector<ResidencyRange> Aliases(std::uint64_t, std::uint64_t) override { return {}; }
    std::uint64_t Generation() override { return generation; }

    void ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) override {
        for (std::size_t k = 0; k < generations.size(); ++k) {
            const auto stamp = stamps.find(firstPage + k);
            changed[k] = generations[k] != ResidencyUnchecked && (generations[k] == 0 || (stamp != stamps.end() && stamp->second > generations[k])) ? 1 : 0;
        }
    }

    std::uint64_t AllocateChunk(std::uint32_t chunk, std::uint32_t) override { return mirrors.Allocate(chunk); }
    void ReleaseChunk(std::uint32_t chunk) override { mirrors.Release(chunk); }

    ResidencyToken CopyIn(std::span<const ResidencyCopy> copies) override {
        mirrors.RecordCopyIn(stream.Commands(), copies, locator());
        return stream.Current();
    }

    ResidencyToken WriteBack(std::span<const ResidencyCopy> copies) override {
        mirrors.RecordWriteBack(stream.Commands(), copies, locator());
        return stream.Current();
    }

    ResidencyRefusal Guard(std::uint64_t page) override {
        ++guardCalls[page];
        const auto index = static_cast<std::uint32_t>(page - BasePage);
        ProtectOrThrow(index, true);
        faultLog.guarded[index] = true;
        return ResidencyRefusal::None;
    }

    void Unguard(std::span<const std::uint64_t> pages) override {
        for (const auto page : pages) {
            const auto index = static_cast<std::uint32_t>(page - BasePage);
            if (faultLog.guarded[index].exchange(false)) ProtectOrThrow(index, false);
        }
    }

    ResidencyRefusal GuardAccess(std::uint64_t page, ResidencyMask mask) override {
        accessGuards[page] = static_cast<ResidencyMask>(accessGuards[page] | mask);
        return ResidencyRefusal::None;
    }

    void UnguardAccess(std::uint64_t page) override { accessGuards.erase(page); }

    void Stamp(std::uint64_t page) { stamps[page] = ++generation; }

    ResidencyMask AccessGuard(std::uint64_t page) const {
        const auto found = accessGuards.find(page);
        return found == accessGuards.end() ? ResidencyMask{0} : found->second;
    }

    Stream& stream;
    GuestRange& guest;
    ResidencyMirrors mirrors;
    std::set<std::uint64_t> unmapped;
    std::set<std::uint64_t> residentViews;
    std::map<std::uint64_t, std::uint32_t> guardCalls;

private:
    ResidencyImportLocator locator() {
        return [this](std::uint64_t page) { return ResidencyImportLocation{guest.Buffer(), VkDeviceSize{page - BasePage} << ResidencyPageShift}; };
    }

    std::uint64_t generation = 1;
    std::map<std::uint64_t, std::uint64_t> stamps;
    std::map<std::uint64_t, ResidencyMask> accessGuards;
};

struct ReadPush {
    std::uint64_t table;
    std::uint64_t results;
    std::uint64_t guestBase;
    std::uint64_t first;
    std::uint32_t count;
    std::uint32_t stride;
};

class Fixture {
public:
    Fixture(const Context& context, std::uint64_t budgetBytes)
        : context(context), guest(context), stream(context), device(context, stream, guest),
          manager(device, ResidencyPolicy{ChunkSlots, budgetBytes, 2, 2}), results(context, 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        createPipeline();
        for (std::uint32_t page = 0; page < GuestPages; ++page) {
            auto* words = guest.Words(BasePage + page);
            for (std::uint32_t word = 0; word < ResidencyPageBytes / 4; ++word) words[word] = Pattern(BasePage + page, word);
        }
    }

    ~Fixture() {
        stream.Submit();
        stream.WaitAll();
        context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
        context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
        context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
    }

    static std::uint32_t Pattern(std::uint64_t page, std::uint32_t word) {
        return static_cast<std::uint32_t>(((page - BasePage) << 24) | word);
    }

    void Ordering() {
        const auto count = std::min<std::uint32_t>(faultLog.count.exchange(0), static_cast<std::uint32_t>(faultLog.pages.size()));
        std::vector<std::uint64_t> pages;
        for (std::uint32_t k = 0; k < count; ++k) pages.push_back(faultLog.pages[k]);
        manager.QueueCpuWritten(pages);
        manager.Prepare();
    }

    void Store(std::uint64_t page, std::uint32_t word, std::uint32_t value) {
        const auto start = std::chrono::steady_clock::now();
        const auto faultsBefore = faultLog.count.load();
        reinterpret_cast<volatile std::uint32_t*>(guest.Words(page))[word] = value;
        const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        if (faultLog.count.load() != faultsBefore) {
            faultMicroseconds += elapsed;
            ++faultsTimed;
        }
        device.Stamp(page);
        expected[{page, word}] = value;
    }

    std::uint32_t Expected(std::uint64_t page, std::uint32_t word) const {
        const auto found = expected.find({page, word});
        return found == expected.end() ? Pattern(page, word) : found->second;
    }

    void GpuFill(VkBuffer buffer, VkDeviceSize offset, std::uint32_t value) {
        auto commands = stream.Commands();
        const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
        const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT};
        const auto barrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, buffer, offset, ResidencySubpageBytes, value);
        barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    }

    void FillMirror(std::uint64_t page, std::uint32_t subpage, std::uint32_t value) {
        const auto slot = manager.Slot(page);
        Expect(slot.has_value(), "the page to fill is resident");
        if (!slot) return;
        GpuFill(device.mirrors.Chunk(slot->chunk)->buffer, (VkDeviceSize{slot->slot} << ResidencyPageShift) + (VkDeviceSize{subpage} << ResidencySubpageShift), value);
        stream.Submit();
        stream.WaitAll();
        manager.NoteGpuWritten(Address(page, subpage), ResidencySubpageBytes);
        for (std::uint32_t word = 0; word < WordsPerSubpage; ++word) expected[{page, subpage * WordsPerSubpage + word}] = value;
    }

    void FillImport(std::uint64_t page, std::uint32_t subpage, std::uint32_t value, bool note) {
        GpuFill(guest.Buffer(), (VkDeviceSize{page - BasePage} << ResidencyPageShift) + (VkDeviceSize{subpage} << ResidencySubpageShift), value);
        if (note) manager.NoteImportWritten(Address(page, subpage), ResidencySubpageBytes);
        stream.Submit();
        stream.WaitAll();
        for (std::uint32_t word = 0; word < WordsPerSubpage; ++word) expected[{page, subpage * WordsPerSubpage + word}] = value;
    }

    static std::uint64_t Address(std::uint64_t page, std::uint32_t subpage = 0) {
        return (page << ResidencyPageShift) + (std::uint64_t{subpage} << ResidencySubpageShift);
    }

    std::vector<std::uint64_t> Publish() {
        manager.TakeChanges();
        auto table = std::make_unique<Buffer>(context, GuestPages * sizeof(std::uint64_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        std::vector<std::uint64_t> entries(GuestPages);
        for (std::uint32_t page = 0; page < GuestPages; ++page) {
            const auto mirror = manager.Translate(BasePage + page);
            entries[page] = mirror ? *mirror : guest.Address(BasePage + page);
        }
        std::memcpy(table->Bytes().data(), entries.data(), entries.size() * sizeof(std::uint64_t));
        currentTable = table->DeviceAddress();
        tables.push_back(std::move(table));
        return entries;
    }

    std::vector<std::uint32_t> Read(std::uint64_t firstPage, std::uint32_t pages, std::uint32_t stride, std::uint32_t perPage, bool use = true) {
        return Collect(RecordRead(firstPage, pages, stride, perPage, use));
    }

    std::uint32_t RecordRead(std::uint64_t firstPage, std::uint32_t pages, std::uint32_t stride, std::uint32_t perPage, bool use) {
        if (use) manager.Use(Address(firstPage), std::uint64_t{pages} * ResidencyPageBytes, ResidencyAccess::Read);
        Publish();
        const auto count = pages * perPage;
        const ReadPush push{currentTable, results.DeviceAddress(), GuestBase, Address(firstPage), count, stride};
        auto commands = stream.Commands();
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, (count + 63) / 64, 1, 1);
        const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        return count;
    }

    std::vector<std::uint32_t> Collect(std::uint32_t count) {
        stream.Submit();
        stream.WaitAll();
        std::vector<std::uint32_t> values(count);
        std::memcpy(values.data(), results.Bytes().data(), count * sizeof(std::uint32_t));
        return values;
    }

    void CheckWords(std::uint64_t page, std::uint32_t words, const std::string& what) {
        const auto values = Read(page, 1, 4, words);
        for (std::uint32_t k = 0; k < words; ++k) {
            if (values[k] != Expected(page, k)) {
                Expect(false, what + ": page " + std::to_string(page - BasePage) + " word " + std::to_string(k) + " read " + std::to_string(values[k]) + ", expected " + std::to_string(Expected(page, k)));
                return;
            }
        }
    }

    void CheckRead(std::uint64_t firstPage, std::uint32_t pages, const std::string& what, bool use = true) {
        CheckValues(firstPage, Read(firstPage, pages, ResidencySubpageBytes, ResidencySubpages, use), what);
    }

    void CheckValues(std::uint64_t firstPage, const std::vector<std::uint32_t>& values, const std::string& what) {
        for (std::uint32_t k = 0; k < values.size(); ++k) {
            const auto page = firstPage + k / ResidencySubpages;
            const auto word = (k % ResidencySubpages) * WordsPerSubpage;
            if (values[k] != Expected(page, word)) {
                Expect(false, what + ": page " + std::to_string(page - BasePage) + " subpage " + std::to_string(k % ResidencySubpages) + " read " + std::to_string(values[k]) + ", expected " + std::to_string(Expected(page, word)));
                return;
            }
        }
    }

    const Context& context;
    GuestRange guest;
    Stream stream;
    FixtureDevice device;
    ResidencyManager manager;
    Buffer results;
    std::vector<std::unique_ptr<Buffer>> tables;
    std::uint64_t currentTable = 0;
    std::map<std::pair<std::uint64_t, std::uint32_t>, std::uint32_t> expected;
    double faultMicroseconds = 0;
    std::uint64_t faultsTimed = 0;

private:
    void createPipeline() {
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = sizeof(RESIDENCY_READ_COMP_SPV);
        moduleInfo.pCode = RESIDENCY_READ_COMP_SPV;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReadPush)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout");
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        pipelineInfo.layout = layout;
        Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines");
    }

    VkShaderModule module = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

void CopyInAndPublishedReads(Fixture& fixture) {
    fixture.Ordering();
    fixture.CheckRead(BasePage, 4, "a dispatch through the published table after the copy-in");
    const auto entries = fixture.Publish();
    const auto chunk = fixture.device.mirrors.Chunk(0);
    for (std::uint32_t page = 0; page < 4; ++page) {
        Expect(entries[page] >= chunk->address && entries[page] < chunk->address + chunk->bytes, "page " + std::to_string(page) + " is published at its mirror slot");
        Expect(entries[page] != fixture.guest.Address(BasePage + page), "page " + std::to_string(page) + " is not published at its import");
    }
    Expect(entries[5] == fixture.guest.Address(BasePage + 5), "a page that was never used is published at its import");
    Expect(fixture.device.mirrors.DeviceBytes() == std::uint64_t{ChunkSlots} << ResidencyPageShift, "one chunk of device memory");
    fixture.FillImport(BasePage + 1, 2, 0xa5a5a5a5u, false);
    const auto stale = fixture.Read(BasePage + 1, 1, ResidencySubpageBytes, ResidencySubpages, false);
    Expect(stale[2] == Fixture::Pattern(BasePage + 1, 2 * WordsPerSubpage), "the dispatch reads the mirror, not the import (a write the manager was not told about is not visible)");
    fixture.manager.NoteImportWritten(Fixture::Address(BasePage + 1, 2), ResidencySubpageBytes);
    fixture.CheckRead(BasePage + 1, 1, "after NoteImportWritten the mirror holds the import's new bytes");
}

void GuardedStores(Fixture& fixture) {
    const auto faults = fixture.manager.Counts().faults;
    fixture.Store(BasePage + 2, 5 * WordsPerSubpage, 0x12345678u);
    fixture.Store(BasePage + 2, 6 * WordsPerSubpage, 0x9abcdef0u);
    Expect(faultLog.count.load() == 1, "the first store into a guarded mirror faults, the second in the same epoch does not");
    fixture.Ordering();
    Expect(fixture.manager.Counts().faults == faults + 1, "the manager applied one fault");
    Expect(faultLog.guarded[2].load(), "the ordering point re-armed the guard");
    fixture.CheckRead(BasePage + 2, 1, "a CPU store reaches the mirror before the next batch, for a dispatch that names no pages", false);
    fixture.Ordering();
    fixture.Store(BasePage + 2, 7 * WordsPerSubpage, 0x0badf00du);
    Expect(faultLog.count.load() == 1, "after the re-arm the next store faults again");
    fixture.Ordering();
    fixture.CheckRead(BasePage + 2, 1, "a later epoch's store reaches the mirror", false);
    fixture.CheckRead(BasePage + 2, 1, "and a dispatch that names the page sees it too");
    Expect(fixture.manager.Slot(BasePage + 2).has_value() && fixture.manager.Counts().hot == 0, "faults two epochs apart keep the page resident");
}

void ImportWritesRefreshMirrors(Fixture& fixture) {
    Expect(fixture.manager.Slot(BasePage + 3).has_value(), "page 3 is resident before the store");
    const auto copied = fixture.manager.Counts().copiedInBytes;
    const auto writtenBack = fixture.manager.Counts().writtenBackBytes;
    fixture.FillImport(BasePage + 3, 4, 0x3c3c3c3cu, false);
    fixture.manager.FlushForCpuRead(Fixture::Address(BasePage + 3, 4), 4);
    const auto* words = fixture.guest.Words(BasePage + 3);
    Expect(words[4 * WordsPerSubpage] == 0x3c3c3c3cu && words[5 * WordsPerSubpage - 1] == 0x3c3c3c3cu, "a driver store into the import reaches guest memory with no copy out of the mirror");
    Expect(words[3 * WordsPerSubpage] == Fixture::Pattern(BasePage + 3, 3 * WordsPerSubpage) && words[5 * WordsPerSubpage] == Fixture::Pattern(BasePage + 3, 5 * WordsPerSubpage), "subpages the store did not write are unchanged");
    Expect(fixture.manager.Counts().copiedInBytes == copied && fixture.manager.Counts().writtenBackBytes == writtenBack, "the store and the flush recorded no copy");
    const auto stale = fixture.Read(BasePage + 3, 1, ResidencySubpageBytes, ResidencySubpages, false);
    Expect(stale[4] == Fixture::Pattern(BasePage + 3, 4 * WordsPerSubpage), "the mirror keeps its bytes until it is refreshed");
    fixture.device.Stamp(BasePage + 3);
    fixture.CheckRead(BasePage + 3, 1, "the next use of the changed page refreshes its mirror");
}

void WriteBackOut(Fixture& fixture) {
    const auto writtenBack = fixture.manager.Counts().writtenBackBytes;
    fixture.FillMirror(BasePage + 3, 1, 0x5a5a5a5au);
    Expect(fixture.manager.Dirty(BasePage + 3) == 0x0002, "the GPU write into the mirror marks subpage 1 dirty");
    Expect(fixture.device.AccessGuard(BasePage + 3) == 0x0002, "the dirty subpage, and only it, is access-guarded from the moment it is dirty");
    Expect(fixture.guest.Words(BasePage + 3)[WordsPerSubpage] == Fixture::Pattern(BasePage + 3, WordsPerSubpage), "guest memory does not have the GPU's bytes yet");
    fixture.CheckRead(BasePage + 3, 1, "a later dispatch reads the GPU's bytes from the mirror");
    Expect(fixture.manager.Counts().writtenBackBytes == writtenBack && fixture.manager.Dirty(BasePage + 3) == 0x0002, "the later dispatch's read wrote nothing back and left the page dirty");
    fixture.manager.FlushForCpuRead(Fixture::Address(BasePage + 3, 1), 4);
    const auto* words = fixture.guest.Words(BasePage + 3);
    Expect(words[WordsPerSubpage] == 0x5a5a5a5au && words[2 * WordsPerSubpage - 1] == 0x5a5a5a5au, "the write-back brought the GPU's bytes to guest memory");
    Expect(words[0] == Fixture::Pattern(BasePage + 3, 0) && words[2 * WordsPerSubpage] == Fixture::Pattern(BasePage + 3, 2 * WordsPerSubpage), "subpages the GPU did not write are unchanged");
    Expect(fixture.manager.Counts().writtenBackBytes == writtenBack + ResidencySubpageBytes, "the CPU read wrote back the one dirty subpage");
    Expect(fixture.manager.Dirty(BasePage + 3) == 0, "the page is clean after the flush");
    Expect(fixture.device.AccessGuard(BasePage + 3) == 0, "the flush drops the access guard");
}

void CpuAndGpuSubpages(Fixture& fixture) {
    const auto writtenBack = fixture.manager.Counts().writtenBackBytes;
    fixture.FillMirror(BasePage + 1, 9, 0x99999999u);
    const auto faults = faultLog.count.load();
    fixture.Store(BasePage + 1, 13 * WordsPerSubpage, 0x13131313u);
    Expect(faultLog.count.load() == faults + 1, "the CPU store into a clean subpage of the dirty page faults on the write guard");
    fixture.Ordering();
    Expect(fixture.manager.Dirty(BasePage + 1) == 0x0200, "the refresh after the CPU store leaves the GPU's subpage dirty");
    fixture.CheckRead(BasePage + 1, 1, "a dispatch reads the GPU's subpage and the CPU's store from one mirror page");
    fixture.manager.FlushForCpuRead(Fixture::Address(BasePage + 1), ResidencyPageBytes);
    const auto* words = fixture.guest.Words(BasePage + 1);
    Expect(words[9 * WordsPerSubpage] == 0x99999999u && words[10 * WordsPerSubpage - 1] == 0x99999999u, "the GPU's subpage reaches guest memory");
    Expect(words[13 * WordsPerSubpage] == 0x13131313u, "the CPU's store in another subpage of the page survives the write-back");
    Expect(fixture.manager.Counts().writtenBackBytes == writtenBack + ResidencySubpageBytes, "only the dirty subpage was written back");
    Expect(fixture.manager.Dirty(BasePage + 1) == 0 && fixture.device.AccessGuard(BasePage + 1) == 0, "the page is clean and unguarded after the flush");
}

void UnmapDropsDirty(Fixture& fixture) {
    const auto page = BasePage + 7;
    fixture.manager.Use(Fixture::Address(page), ResidencyPageBytes, ResidencyAccess::Read);
    Expect(fixture.manager.Slot(page).has_value(), "page 7 is resident before the unmap");
    fixture.FillMirror(page, 2, 0x22222222u);
    const auto before = fixture.manager.Counts();
    fixture.device.unmapped.insert(page);
    const std::array<std::uint64_t, 1> unmappedPage{page};
    fixture.device.Unguard(unmappedPage);
    fixture.manager.DropRange(Fixture::Address(page), ResidencyPageBytes);
    fixture.manager.Prepare();
    fixture.stream.Submit();
    fixture.stream.WaitAll();
    const auto after = fixture.manager.Counts();
    Expect(!fixture.manager.Slot(page) && after.dropped == before.dropped + 1, "the unmapped page leaves residency");
    Expect(after.writtenBackBytes == before.writtenBackBytes && fixture.guest.Words(page)[2 * WordsPerSubpage] == Fixture::Pattern(page, 2 * WordsPerSubpage), "an unmap drops the dirty mirror without a write-back");
    Expect(fixture.device.AccessGuard(page) == 0, "an unmap drops the access guard");
    for (std::uint32_t word = 0; word < WordsPerSubpage; ++word) fixture.expected.erase({page, 2 * WordsPerSubpage + word});
    fixture.device.unmapped.erase(page);
}

void PinsHoldInFlightWork(Fixture& fixture) {
    const auto before = fixture.manager.Counts();
    const auto count = fixture.RecordRead(BasePage, 4, ResidencySubpageBytes, ResidencySubpages, true);
    fixture.manager.Use(Fixture::Address(BasePage + 6), 2 * ResidencyPageBytes, ResidencyAccess::Read);
    const auto after = fixture.manager.Counts();
    Expect(after.evicted == before.evicted && after.deferred == before.deferred + 2, "pages a recorded dispatch reads are pinned: the two new pages are deferred, nothing is evicted");
    for (std::uint32_t page = 0; page < 4; ++page) Expect(fixture.manager.Slot(BasePage + page).has_value(), "pinned page " + std::to_string(page) + " stays resident");
    fixture.CheckValues(BasePage, fixture.Collect(count), "the dispatch recorded before the admissions");
}

void EvictionUnderSmallBudget(Fixture& fixture) {
    fixture.FillMirror(BasePage + 0, 7, 0x77777777u);
    fixture.manager.Use(Fixture::Address(BasePage + 2), 2 * ResidencyPageBytes, ResidencyAccess::Read);
    const auto slot0 = fixture.manager.Slot(BasePage + 0);
    const auto slot1 = fixture.manager.Slot(BasePage + 1);
    fixture.Ordering();
    fixture.manager.Use(Fixture::Address(BasePage + 4), 2 * ResidencyPageBytes, ResidencyAccess::Read);
    Expect(!fixture.manager.Slot(BasePage + 0) && !fixture.manager.Slot(BasePage + 1), "the two least recent pages are evicted");
    Expect(!faultLog.guarded[0].load() && !faultLog.guarded[1].load(), "evicted pages are unguarded");
    fixture.stream.Submit();
    fixture.stream.WaitAll();
    Expect(fixture.guest.Words(BasePage + 0)[7 * WordsPerSubpage] == 0x77777777u, "the dirty victim was written back");
    fixture.Ordering();
    fixture.manager.Use(Fixture::Address(BasePage + 4), 2 * ResidencyPageBytes, ResidencyAccess::Read);
    const auto slot4 = fixture.manager.Slot(BasePage + 4);
    const auto slot5 = fixture.manager.Slot(BasePage + 5);
    Expect(slot4 && slot5 && ((slot4 == slot0 && slot5 == slot1) || (slot4 == slot1 && slot5 == slot0)), "the new pages take the retired slots");
    const auto counts = fixture.manager.Counts();
    Expect(counts.chunks == 1 && counts.chunksAllocated == 1 && fixture.device.mirrors.DeviceBytes() == std::uint64_t{ChunkSlots} << ResidencyPageShift, "the budget of one chunk holds");
    fixture.CheckRead(BasePage + 0, 2, "evicted pages read from their imports", false);
    fixture.CheckRead(BasePage + 4, 2, "new pages read from their mirrors");
    const auto faults = faultLog.count.load();
    fixture.Store(BasePage + 0, 0, 0x01010101u);
    Expect(faultLog.count.load() == faults, "a store into an evicted page does not fault");
    fixture.Ordering();
}

void Frames(Fixture& fixture) {
    constexpr std::uint32_t FrameCount = 10;
    constexpr std::uint32_t OrderingPoints = 4;
    const auto hotBefore = fixture.manager.Counts().hot;
    std::uint64_t faultTotal = 0;
    bool wasResident = true;
    std::uint32_t departures = 0;
    std::uint32_t returns = 0;
    std::uint32_t mostInOneFrame = 0;
    std::printf("frames: faults per frame (constant buffer page 4 written once per frame, vertex ring page 5 written at every ordering point, page 6 written once per frame at its end or its start, page 2 static):");
    for (std::uint32_t frame = 0; frame < FrameCount; ++frame) {
        fixture.manager.NextFrame();
        const auto faults = fixture.manager.Counts().faults;
        std::uint32_t transitions = 0;
        for (std::uint32_t point = 0; point < OrderingPoints; ++point) {
            if (point == 0) fixture.Store(BasePage + 4, frame, 0x40000000u + frame);
            if (point == (frame % 2 == 0 ? OrderingPoints - 1 : 0)) fixture.Store(BasePage + 6, frame, 0x60000000u + frame);
            fixture.Store(BasePage + 5, frame * OrderingPoints + point, 0x50000000u + frame * OrderingPoints + point);
            fixture.Ordering();
            fixture.CheckRead(BasePage + 2, 1, "a static page");
            fixture.CheckWords(BasePage + 4, 64, "frame " + std::to_string(frame) + ", the constant buffer");
            fixture.CheckWords(BasePage + 5, 64, "frame " + std::to_string(frame) + ", the vertex ring");
            fixture.CheckWords(BasePage + 6, 64, "frame " + std::to_string(frame) + ", the page written once per frame at its end or its start");
            const bool resident = fixture.manager.Slot(BasePage + 6).has_value();
            if (resident != wasResident) {
                ++transitions;
                ++(resident ? returns : departures);
            }
            wasResident = resident;
        }
        mostInOneFrame = std::max(mostInOneFrame, transitions);
        const auto frameFaults = fixture.manager.Counts().faults - faults;
        faultTotal += frameFaults;
        std::printf(" %llu", static_cast<unsigned long long>(frameFaults));
    }
    std::printf("\n");
    const auto counts = fixture.manager.Counts();
    std::printf("frames: %llu faults in %u frames, %llu pages turned hot, page 4 %s, page 5 %s, page 6 left its mirror %u times and returned %u times, %.2f us per faulting store (%llu timed)\n", static_cast<unsigned long long>(faultTotal), FrameCount,
        static_cast<unsigned long long>(counts.hot - hotBefore), fixture.manager.Slot(BasePage + 4) ? "resident" : "not resident", fixture.manager.Slot(BasePage + 5) ? "resident" : "not resident", departures, returns,
        fixture.faultsTimed == 0 ? 0.0 : fixture.faultMicroseconds / static_cast<double>(fixture.faultsTimed), static_cast<unsigned long long>(fixture.faultsTimed));
    Expect(!fixture.manager.Slot(BasePage + 5).has_value() && fixture.manager.Hot(BasePage + 5) && !faultLog.guarded[5].load(), "the vertex ring faults in consecutive frames, turns hot and is served from its import without a write guard");
    Expect(!fixture.manager.Slot(BasePage + 4).has_value() && fixture.manager.Hot(BasePage + 4), "the constant buffer, written in consecutive frames, turns hot and is served from its import");
    Expect(departures == 1 && returns == 0 && mostInOneFrame <= 1, "a page written once per frame stays on one side: it left its mirror " + std::to_string(departures) + " times and returned " + std::to_string(returns) + " times, " + std::to_string(mostInOneFrame) + " moves in one frame");
    Expect(counts.hot - hotBefore == 3 && counts.hotResident == 0, "three pages turned hot and none of them is polled in a mirror");
    Expect(faultTotal == 9, "only the first two frames fault: the vertex ring at each of its four ordering points and then once, the constant buffer and page 6 once in each; " + std::to_string(faultTotal) + " faults");
}

void Release(Fixture& fixture) {
    fixture.stream.Submit();
    fixture.stream.WaitAll();
    fixture.manager.SetBudget(0);
    fixture.manager.Retire();
    Expect(fixture.manager.Counts().chunks == 0 && fixture.device.mirrors.DeviceBytes() == 0, "a zero budget releases every chunk once the GPU is done");
    for (std::uint32_t page = 0; page < GuestPages; ++page) Expect(!faultLog.guarded[page].load(), "no guard left after every page left residency");
}

void ResidentViewsStayOnImports(const Context& context) {
    Fixture fixture(context, std::uint64_t{ChunkSlots} << ResidencyPageShift);
    const auto view = BasePage + 2;
    fixture.device.residentViews.insert(view);
    fixture.Ordering();
    const auto before = fixture.manager.Counts();
    fixture.CheckRead(BasePage, 4, "resident view: the first dispatch reading pages 0 to 3");
    fixture.Store(view, 3 * WordsPerSubpage, 0x600dcafeu);
    fixture.Ordering();
    fixture.CheckRead(BasePage, 4, "resident view: a dispatch after the CPU stored into the resident view");
    fixture.CheckWords(view, ResidencyPageBytes / 4, "resident view: every word of the resident view after the store");
    const auto after = fixture.manager.Counts();
    Expect(!fixture.manager.Slot(view).has_value(), "resident view: the page the guest keeps resident was admitted");
    Expect(fixture.manager.Slot(BasePage + 1).has_value() && fixture.manager.Slot(BasePage + 3).has_value(), "resident view: its neighbours were not admitted");
    Expect(fixture.device.guardCalls[view] == 0 && !faultLog.guarded[2].load(), "resident view: " + std::to_string(fixture.device.guardCalls[view]) + " guard calls on the resident view");
    Expect(after.copiedInBytes == before.copiedInBytes + 3 * ResidencyPageBytes, "resident view: " + std::to_string((after.copiedInBytes - before.copiedInBytes) >> ResidencyPageShift) + " pages copied in, expected the three neighbours only");
    Expect(after.residentSkipped > before.residentSkipped, "resident view: the skipped admission is not counted");
}

void BackingCommitMany(const Context& context) {
#ifdef _WIN32
    constexpr VkDeviceSize Bytes = VkDeviceSize{64} << 20u;
    constexpr int Count = 16;
    const auto privateBytes = [] {
        PROCESS_MEMORY_COUNTERS_EX counters{};
        GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters));
        return static_cast<long long>(counters.PrivateUsage);
    };
    const auto workingSet = [] {
        PROCESS_MEMORY_COUNTERS_EX counters{};
        GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters));
        MEMORYSTATUSEX status{sizeof(status)};
        GlobalMemoryStatusEx(&status);
        return std::pair<long long, long long>(static_cast<long long>(counters.WorkingSetSize), static_cast<long long>(status.ullAvailPhys));
    };
    const auto allocate = context.Function<PFN_vkAllocateMemory>("vkAllocateMemory");
    const auto release = context.Function<PFN_vkFreeMemory>("vkFreeMemory");
    const auto type = context.MemoryType(~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    for (int pass = 0; pass < 4; ++pass) {
        const int variant = pass % 2;
        VkMemoryAllocateFlagsInfo addressFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, variant == 1 ? &addressFlags : nullptr};
        allocation.allocationSize = Bytes;
        allocation.memoryTypeIndex = type;
        std::vector<VkDeviceMemory> memories;
        const auto before = privateBytes();
        const auto setBefore = workingSet();
        for (int i = 0; i < Count; ++i) {
            VkDeviceMemory memory = VK_NULL_HANDLE;
            if (allocate(context.device, &allocation, nullptr, &memory) == VK_SUCCESS) memories.push_back(memory);
        }
        const auto held = privateBytes();
        const auto setHeld = workingSet();
        std::printf("backing commit many: working set %+.1f MiB, available RAM %+.1f MiB while held\n", static_cast<double>(setHeld.first - setBefore.first) / (1024.0 * 1024.0), static_cast<double>(setHeld.second - setBefore.second) / (1024.0 * 1024.0));
        for (const auto memory : memories) release(context.device, memory, nullptr);
        const auto freed = privateBytes();
        std::printf("backing commit many: %zu x 64 MiB device-local (type %u, %s): private %+.1f MiB while held, %+.1f MiB after free\n", memories.size(), type, variant == 1 ? "device address" : "plain", static_cast<double>(held - before) / (1024.0 * 1024.0), static_cast<double>(freed - before) / (1024.0 * 1024.0));
    }
#else
    static_cast<void>(context);
#endif
}

void BackingCommitByType(const Context& context) {
#ifdef _WIN32
    constexpr VkDeviceSize Bytes = VkDeviceSize{64} << 20u;
    const auto privateBytes = [] {
        PROCESS_MEMORY_COUNTERS_EX counters{};
        GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters));
        return static_cast<long long>(counters.PrivateUsage);
    };
    const auto allocate = context.Function<PFN_vkAllocateMemory>("vkAllocateMemory");
    const auto release = context.Function<PFN_vkFreeMemory>("vkFreeMemory");
    const auto createBuffer = context.Function<PFN_vkCreateBuffer>("vkCreateBuffer");
    const auto destroyBuffer = context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer");
    const auto requirementsOf = context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements");
    const auto bind = context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory");
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = Bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    for (std::uint32_t type = 0; type < context.memory.memoryTypeCount; ++type) {
        const auto flags = context.memory.memoryTypes[type].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) continue;
        for (int variant = 0; variant < 3; ++variant) {
            VkBuffer buffer = VK_NULL_HANDLE;
            Check(createBuffer(context.device, &info, nullptr, &buffer), "vkCreateBuffer backing probe");
            VkMemoryRequirements requirements{};
            requirementsOf(context.device, buffer, &requirements);
            if ((requirements.memoryTypeBits & (1u << type)) == 0) {
                destroyBuffer(context.device, buffer, nullptr);
                break;
            }
            VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
            dedicated.buffer = buffer;
            VkMemoryAllocateFlagsInfo addressFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, variant == 2 ? &dedicated : nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.pNext = variant == 0 ? nullptr : &addressFlags;
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = type;
            const auto before = privateBytes();
            VkDeviceMemory memory = VK_NULL_HANDLE;
            const auto result = allocate(context.device, &allocation, nullptr, &memory);
            const auto allocated = privateBytes();
            if (result == VK_SUCCESS) Check(bind(context.device, buffer, memory, 0), "vkBindBufferMemory backing probe");
            const auto bound = privateBytes();
            if (memory != VK_NULL_HANDLE) release(context.device, memory, nullptr);
            destroyBuffer(context.device, buffer, nullptr);
            const auto freed = privateBytes();
            const char* names[] = {"plain", "device address", "device address + dedicated"};
            std::printf("backing commit by type: memory type %u (flags 0x%x, heap %u) %s: VkResult %d, private +%.1f MiB at allocate, +%.1f MiB after bind, %+.1f MiB after free\n", type, static_cast<unsigned>(flags), context.memory.memoryTypes[type].heapIndex, names[variant], static_cast<int>(result), static_cast<double>(allocated - before) / (1024.0 * 1024.0), static_cast<double>(bound - before) / (1024.0 * 1024.0), static_cast<double>(freed - before) / (1024.0 * 1024.0));
        }
    }
#else
    static_cast<void>(context);
#endif
}

void BackingCommit(const Context& context) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX before{};
    GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&before), sizeof(before));
    ResidencyMirrors mirrors(context, 1024);
    mirrors.Allocate(0);
    PROCESS_MEMORY_COUNTERS_EX after{};
    GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&after), sizeof(after));
    const auto delta = static_cast<double>(static_cast<long long>(after.PrivateUsage) - static_cast<long long>(before.PrivateUsage)) / (1024.0 * 1024.0);
    std::printf("backing commit: one 64 MiB device-local mirror chunk adds %.1f MiB of private bytes (the RAM-side gate subtracts the device-local allocation sum)\n", delta);
    mirrors.Release(0);
#else
    static_cast<void>(context);
#endif
}

}

int main() {
    try {
        std::unique_ptr<Device> device;
        try {
            device = std::make_unique<Device>();
        } catch (const SkipTest& skip) {
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) throw;
            std::printf("skipped: %s\n", skip.what());
            return Skipped;
        }
        {
            Fixture fixture(device->Get(), std::uint64_t{ChunkSlots} << ResidencyPageShift);
            CopyInAndPublishedReads(fixture);
            GuardedStores(fixture);
            ImportWritesRefreshMirrors(fixture);
            WriteBackOut(fixture);
            CpuAndGpuSubpages(fixture);
            PinsHoldInFlightWork(fixture);
            EvictionUnderSmallBudget(fixture);
            Frames(fixture);
            UnmapDropsDirty(fixture);
            Release(fixture);
        }
        BackingCommit(device->Get());
        BackingCommitByType(device->Get());
        BackingCommitMany(device->Get());
        ResidentViewsStayOnImports(device->Get());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d residency fixture checks failed\n", failures);
        return 1;
    }
    std::printf("residency mirror fixture passed\n");
    return 0;
}
