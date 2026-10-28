#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYDRIVER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYDRIVER_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Residency.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyMirrors.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyTokens.hpp"
#include <vulkan/vulkan.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <optional>
#include <span>
#include <vector>

namespace AgcDriver::Graphics {

class Recorder;

struct ResidencyGuestImport {
    ResidencyImport attributes;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceAddress address = 0;
};

class IResidencyGuestMemory {
public:
    virtual ~IResidencyGuestMemory() = default;
    virtual std::optional<ResidencyGuestImport> ImportOf(std::uint64_t page) = 0;
    virtual std::vector<ResidencyRange> Aliases(std::uint64_t address, std::uint64_t bytes) = 0;
    virtual std::uint64_t Generation() = 0;
    virtual void ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) = 0;
    virtual ResidencyRefusal Guard(std::uint64_t page) = 0;
    virtual void Unguard(std::span<const std::uint64_t> pages) = 0;
    virtual ResidencyRefusal GuardAccess(std::uint64_t page, ResidencyMask mask) = 0;
    virtual void UnguardAccess(std::uint64_t page) = 0;
    virtual std::uint64_t GuestBytes() = 0;
    virtual ResidencyWrites PageWrites(std::uint64_t page, std::uint64_t generation) {
        static_cast<void>(page);
        static_cast<void>(generation);
        return {};
    }
    virtual std::vector<std::uint64_t> WrittenSince(std::span<const std::pair<std::uint64_t, std::uint64_t>> resident) {
        static_cast<void>(resident);
        return {};
    }
};

class RefusingResidencyGuest final : public IResidencyGuestMemory {
public:
    std::optional<ResidencyGuestImport> ImportOf(std::uint64_t page) override;
    std::vector<ResidencyRange> Aliases(std::uint64_t address, std::uint64_t bytes) override;
    std::uint64_t Generation() override;
    void ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) override;
    ResidencyRefusal Guard(std::uint64_t page) override;
    void Unguard(std::span<const std::uint64_t> pages) override;
    ResidencyRefusal GuardAccess(std::uint64_t page, ResidencyMask mask) override;
    void UnguardAccess(std::uint64_t page) override;
    std::uint64_t GuestBytes() override;
};

class ResidencyDriver final : public IResidencyDevice {
public:
    ResidencyDriver(const Context& context, Recorder& recorder, IResidencyGuestMemory& guest, const ResidencyPolicy& policy, std::unique_ptr<IResidencyLanes> lanes = nullptr);
    ResidencyDriver(const ResidencyDriver&) = delete;
    ResidencyDriver& operator=(const ResidencyDriver&) = delete;
    ~ResidencyDriver() override;

    static ResidencyDriver* Active();
    static bool Recording();
    static std::unique_ptr<ResidencyDriver> CreateIfEnabled(const Context& context, Recorder& recorder, IResidencyGuestMemory& guest);

    ResidencyManager& Manager() { return manager; }
    const ResidencyMirrors& Mirrors() const { return mirrors; }
    ResidencyTokens& Tokens() { return tokens; }
    void OrderingPoint();
    void EnableFaults();
    void DisableFaults();
    void AccessFault(std::uint64_t address);
    std::uint64_t FastFaults() const { return fastFaults.load(std::memory_order_relaxed); }
    std::uint64_t ServedFaults() const { return servedFaults.load(std::memory_order_relaxed); }
    std::uint64_t RefusedFaults() const { return refusedFaults.load(std::memory_order_relaxed); }
    std::uint64_t DriverPublishes() const { return driverPublishes.load(std::memory_order_relaxed); }
    std::uint64_t WriteBackStamps() const { return writeBackStamps.load(std::memory_order_relaxed); }

    bool Reached(const ResidencyToken& token) override;
    void Wait(const ResidencyToken& token) override;
    bool WaitUnlocked(const ResidencyToken& token) override;
    bool Signaled(const ResidencyToken& token) override;
    ResidencyToken Newest() override;
    ResidencyToken Current() override;
    ResidencyImport Import(std::uint64_t page) override;
    std::vector<ResidencyRange> Aliases(std::uint64_t address, std::uint64_t bytes) override;
    std::uint64_t Generation() override;
    void ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) override;
    std::uint64_t AllocateChunk(std::uint32_t chunk, std::uint32_t slots) override;
    void ReleaseChunk(std::uint32_t chunk) override;
    ResidencyToken CopyIn(std::span<const ResidencyCopy> copies) override;
    ResidencyToken WriteBack(std::span<const ResidencyCopy> copies) override;
    ResidencyRefusal Guard(std::uint64_t page) override;
    void Unguard(std::span<const std::uint64_t> pages) override;
    ResidencyRefusal GuardAccess(std::uint64_t page, ResidencyMask mask) override;
    void UnguardAccess(std::uint64_t page) override;
    std::uint64_t Budget() override;
    std::uint64_t GuestBytes() override;
    ResidencyWrites PageWrites(std::uint64_t page, std::uint64_t generation) override;

private:
    ResidencyToken record(std::span<const ResidencyCopy> copies, bool intoMirror);
    bool stampPending(std::uint64_t address, std::size_t bytes);
    void stampWrittenBack(std::uint64_t address, std::size_t bytes, bool all);
    struct PendingStamp {
        ResidencyToken token;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    };
    struct FaultServer;

    Context context;
    Recorder& recorder;
    IResidencyGuestMemory& guest;
    std::unique_ptr<IResidencyLanes> lanes;
    ResidencyTokens tokens;
    std::uint64_t budgetBytes = 0;
    ResidencyMirrors mirrors;
    ResidencyManager manager;
    std::unique_ptr<FaultServer> faultServer;
    std::atomic<std::uint64_t> fastFaults{0};
    std::atomic<std::uint64_t> servedFaults{0};
    std::atomic<std::uint64_t> refusedFaults{0};
    std::atomic<std::uint64_t> driverPublishes{0};
    std::atomic<std::uint64_t> writeBackStamps{0};
    std::mutex stampMutex;
    std::vector<PendingStamp> pendingStamps;
    std::atomic<bool> stampsPending{false};
};

}

#endif
