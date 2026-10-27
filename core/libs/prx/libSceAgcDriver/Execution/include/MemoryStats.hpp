#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_MEMORYSTATS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_MEMORYSTATS_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <string>

namespace AgcDriver::MemoryStats {

// DEBUG (local bring-up): APS5_MEM_STATS=<seconds> prints one "[mem]" line per period with process
// memory, Vulkan allocations and the live bytes/objects of the driver's large host-memory owners.
enum class Counter : std::uint8_t {
    ShaderMemoryPages,
    StorageTextureOriginal,
    StorageTextureHashes,
    Count,
};

struct Slot {
    std::atomic<std::int64_t> bytes{0};
    std::atomic<std::int64_t> count{0};
};

[[nodiscard]] inline unsigned PeriodSeconds() noexcept {
    static const unsigned seconds = [] {
        const char* value = std::getenv("APS5_MEM_STATS");
        if (value == nullptr || *value == '\0') return 0u;
        return static_cast<unsigned>(std::min<unsigned long>(std::strtoul(value, nullptr, 10), 3600ul));
    }();
    return seconds;
}

[[nodiscard]] inline bool Enabled() noexcept {
    return PeriodSeconds() != 0;
}

[[nodiscard]] inline Slot& At(Counter counter) noexcept {
    static std::array<Slot, static_cast<std::size_t>(Counter::Count)> slots;
    return slots[static_cast<std::size_t>(counter)];
}

inline void Add(Counter counter, std::int64_t bytes, std::int64_t count = 1) noexcept {
    if (!Enabled()) return;
    auto& slot = At(counter);
    slot.bytes.fetch_add(bytes, std::memory_order_relaxed);
    slot.count.fetch_add(count, std::memory_order_relaxed);
}

// Device memory by owner. A creation site names its owner with an OwnerScope on its thread; the
// device wrapper then charges vkAllocateMemory (bytes, by memory class) and vkCreateImage /
// vkCreateBuffer (objects) to the innermost owner, and frees/destroys to the owner charged. The
// buffer pool's own allocations are charged to its tier (PoolTierScope), whoever asked: retained
// pool memory outlives the asker. Pooled Buffer/DeviceBuffer objects are counted live by the owner
// that made them (LiveToken), so a pool tier splits into its live users and its retained rest.
enum class Owner : std::uint8_t {
    None,
    SampledTexture,
    StorageTexture,
    DepthPlaneStorage,
    AttachmentProxy,
    DepthCopy,
    DepthSurface,
    RenderTarget,
    ColorTransfer,
    TextureUpload,
    GuestStaging,
    GuestMirror,
    HostImport,
    UnitShadow,
    DrawSnapshot,
    DrawScratch,
    ShaderResources,
    Bda,
    WorkgroupMemory,
    Present,
    Misc,
    PoolDevice,
    PoolHost,
    PoolSlab,
    Count,
};

inline constexpr const char* OwnerNames[] = {
    "unattributed", "tex_sampled", "tex_storage", "depth_plane_storage", "attachment_proxy", "depth_copy", "depth_surface", "render_target",
    "color_transfer", "texture_upload", "guest_staging", "guest_mirror", "host_import", "unit_shadow", "draw_snapshot", "draw_scratch",
    "shader_resources", "bda", "workgroup_memory", "present", "misc", "pool_device", "pool_host", "pool_slab",
};
static_assert(std::size(OwnerNames) == static_cast<std::size_t>(Owner::Count));

struct OwnerSlots {
    Slot device;      // DEVICE_LOCAL allocations (with or without HOST_VISIBLE)
    Slot host;        // host-visible-only and other allocations (imports not included)
    Slot imported;    // VK_EXT_external_memory_host imports
    Slot images;      // live VkImages (count only)
    Slot buffers;     // live VkBuffers (count only)
    Slot liveDevice;  // live pooled device-local Buffer/DeviceBuffer objects (capacity bytes)
    Slot liveHost;    // live pooled host-visible Buffer objects (capacity bytes)
};

[[nodiscard]] inline OwnerSlots& OwnerAt(Owner owner) noexcept {
    static std::array<OwnerSlots, static_cast<std::size_t>(Owner::Count)> slots;
    return slots[static_cast<std::size_t>(owner)];
}

[[nodiscard]] inline Owner& CurrentOwner() noexcept {
    thread_local Owner owner = Owner::None;
    return owner;
}

[[nodiscard]] inline Owner& CurrentPoolTier() noexcept {
    thread_local Owner tier = Owner::None;
    return tier;
}

// `weak`: only when no enclosing scope named an owner (a storage image a depth surface makes for
// its plane stays the depth surface's).
class OwnerScope {
public:
    explicit OwnerScope(Owner owner, bool weak = false) noexcept : active(Enabled() && (!weak || CurrentOwner() == Owner::None)) {
        if (!active) return;
        previous = CurrentOwner();
        CurrentOwner() = owner;
    }
    ~OwnerScope() {
        if (active) CurrentOwner() = previous;
    }
    OwnerScope(const OwnerScope&) = delete;
    OwnerScope& operator=(const OwnerScope&) = delete;

private:
    bool active;
    Owner previous = Owner::None;
};

// Around the buffer pool's own vkAllocateMemory calls only.
class PoolTierScope {
public:
    explicit PoolTierScope(Owner tier) noexcept : active(Enabled()) {
        if (!active) return;
        previous = CurrentPoolTier();
        CurrentPoolTier() = tier;
    }
    ~PoolTierScope() {
        if (active) CurrentPoolTier() = previous;
    }
    PoolTierScope(const PoolTierScope&) = delete;
    PoolTierScope& operator=(const PoolTierScope&) = delete;

private:
    bool active;
    Owner previous = Owner::None;
};

// A pooled buffer object's share of its owner's live bytes: Set once the object holds memory, undone
// by the destructor (also after a constructor that threw past Set).
class LiveToken {
public:
    LiveToken() noexcept : owner(Enabled() ? CurrentOwner() : Owner::None) {}
    ~LiveToken() {
        if (bytes == 0) return;
        auto& slot = device ? OwnerAt(owner).liveDevice : OwnerAt(owner).liveHost;
        slot.bytes.fetch_sub(static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
        slot.count.fetch_sub(1, std::memory_order_relaxed);
    }
    LiveToken(const LiveToken&) = delete;
    LiveToken& operator=(const LiveToken&) = delete;
    void Set(bool deviceLocal, std::uint64_t held) noexcept {
        if (!Enabled() || bytes != 0 || held == 0) return;
        device = deviceLocal;
        bytes = held;
        auto& slot = device ? OwnerAt(owner).liveDevice : OwnerAt(owner).liveHost;
        slot.bytes.fetch_add(static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
        slot.count.fetch_add(1, std::memory_order_relaxed);
    }

private:
    Owner owner;
    bool device = false;
    std::uint64_t bytes = 0;
};

// Starts the reporter thread once (no-op when off); Stop() waits for an in-flight report.
void Start();
void Stop() noexcept;
// One [mem] line now, ending in ` window=<window>` (ProfileWindows::Close; nothing when off).
void ReportNow(const char* window);

inline void AppendMiB(std::string& line, const char* name, std::uint64_t bytes, std::uint64_t count) {
    char buffer[160];
    const int written = std::snprintf(buffer, sizeof(buffer), " %s=%.1fMB(%llu)", name, static_cast<double>(bytes) / 1048576.0, static_cast<unsigned long long>(count));
    if (written > 0) line.append(buffer, std::min<std::size_t>(static_cast<std::size_t>(written), sizeof(buffer) - 1));
}

inline void AppendMiB(std::string& line, const char* name, std::uint64_t bytes) {
    char buffer[160];
    const int written = std::snprintf(buffer, sizeof(buffer), " %s=%.1fMB", name, static_cast<double>(bytes) / 1048576.0);
    if (written > 0) line.append(buffer, std::min<std::size_t>(static_cast<std::size_t>(written), sizeof(buffer) - 1));
}

inline void AppendCount(std::string& line, const char* name, std::uint64_t count) {
    char buffer[160];
    const int written = std::snprintf(buffer, sizeof(buffer), " %s=%llu", name, static_cast<unsigned long long>(count));
    if (written > 0) line.append(buffer, std::min<std::size_t>(static_cast<std::size_t>(written), sizeof(buffer) - 1));
}

}

// Subsystem sections of the line, each defined next to the state it reads (run on the reporter thread).
namespace AgcDriver::Graphics {
void AppendTextureMemoryStats(std::string& line);
void AppendGuestBufferMemoryStats(std::string& line);
}

#endif
