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

// Starts the reporter thread once (no-op when off); Stop() waits for an in-flight report.
void Start();
void Stop() noexcept;

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
