#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENTCENSUS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENTCENSUS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// APS5_PROFILE_RESIDENT_CENSUS=1: the guest memory the GPU work of each profile window (a frame in
// agc_frame_replay, 10 s in the game; ProfileWindows) reaches, in distinct 64 KiB pages, as the
// residency design (PLAN S1) needs it:
//
//   [resident-census] window=<w> pages <n> = <MiB> MiB (gpu-written <MiB>, cpu-then-gpu <MiB>, read-only <MiB>);
//     by class written/cpu-then-gpu/read-only MiB: staged <a>/<b>/<c>, in-place ..., bda-leased ...,
//     storage-image ..., vertex-index ..., indirect ...; lru evictions (pages) 512 MiB <n>, 1024 MiB <n>,
//     2048 MiB <n>, 4096 MiB <n>; union <n> pages = <MiB> MiB (gpu-written <MiB>, cpu-then-gpu <MiB>, read-only <MiB>)
//
// (one physical line). Classes: staged = buffer ranges a build copies per use (GuestBufferMemory
// regions without an import or mirror), in-place = buffer ranges served in place by a host import
// or an image mirror, bda-leased = what an address-based build can reach (its lease and its
// address space's base regions; reachable, not proven read), storage-image = storage images a use
// binds, vertex-index = index buffers and fixed-function vertex buffers of committed draws,
// indirect = indirect draw and dispatch argument (and count) records. Uses are those that pass
// ShaderResources::MarkGpuWrites (draws, dispatches, recipe replays). A page counts once per class
// it is in, and once in the total. Split: gpu-written = a use wrote it in the window (its written
// descriptor ranges, written storage images); cpu-then-gpu = not GPU-written, and read by the GPU
// after a CPU write in the same window (CPU writes: the memory deltas agc_frame_replay restores,
// i.e. the title's recorded stores; none in the game); read-only = the rest. lru evictions: a
// least-recently-used set of N MiB of pages, touched in the order of each page's last use in the
// window, kept across windows: the pages it evicted in this window. union: every page seen so far,
// split as above by the strongest use any window made of it.
namespace AgcDriver::Pm4 {
struct DrawParameters;
}

namespace AgcDriver::Graphics {
struct CompiledShader;
}

namespace AgcDriver::Graphics::ResidentCensus {

enum class Class : std::uint8_t { Staged, InPlace, BdaLeased, StorageImage, VertexIndex, Indirect, Count };
inline constexpr std::array<const char*, static_cast<std::size_t>(Class::Count)> ClassNames{"staged", "in-place", "bda-leased", "storage-image", "vertex-index", "indirect"};
inline constexpr std::uint64_t PageBytes = 64u * 1024u;
inline constexpr std::array<std::uint64_t, 4> LruMiB{512, 1024, 2048, 4096};

// The census arithmetic, without the driver: notes in call order, Close per window.
class Model {
public:
    void GpuRead(Class which, std::uint64_t begin, std::uint64_t end);
    void GpuWrite(Class which, std::uint64_t begin, std::uint64_t end);
    void CpuWrite(std::uint64_t begin, std::uint64_t end);
    // A group of read ranges named by `key` (an address space or a lease: the same for many uses):
    // stored once per window, its later uses only move its last use.
    bool GroupSeen(const void* key);
    void GpuReadGroup(const void* key, Class which, std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges);

    struct Split {
        std::uint64_t written = 0;
        std::uint64_t cpuThenGpu = 0;
        std::uint64_t readOnly = 0;
        [[nodiscard]] std::uint64_t Pages() const { return written + cpuThenGpu + readOnly; }
    };
    struct WindowResult {
        Split total;
        std::array<Split, static_cast<std::size_t>(Class::Count)> byClass{};
        std::array<std::uint64_t, LruMiB.size()> evicted{};
        Split unionTotal;
    };
    // Ends the window: its pages, the LRU evictions, the union so far.
    WindowResult Close();
    [[nodiscard]] static std::string Format(const char* window, const WindowResult& result);

private:
    struct Range {
        std::uint64_t firstSeq = 0;
        std::uint64_t lastSeq = 0;
    };
    // (first page, end page, class, written)
    using RangeKey = std::array<std::uint64_t, 4>;
    struct Group {
        Class which = Class::BdaLeased;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        Range seqs;
    };
    void gpu(Class which, std::uint64_t begin, std::uint64_t end, bool written);
    class Lru {
    public:
        explicit Lru(std::uint64_t pages) : capacity(pages) {}
        // Returns whether a page was evicted.
        bool Touch(std::uint64_t page);

    private:
        std::uint64_t capacity;
        std::list<std::uint64_t> order;
        std::unordered_map<std::uint64_t, std::list<std::uint64_t>::iterator> where;
    };
    std::uint64_t seq = 0;
    std::map<RangeKey, Range> ranges;
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> cpuWrites;
    std::unordered_map<const void*, Group> groups;
    // Union: per page, bit 0 written, bit 1 cpu-then-gpu, bit 2 read; class bits from bit 8.
    std::unordered_map<std::uint64_t, std::uint32_t> unionPages;
    std::vector<Lru> lrus = [] {
        // Reserved: an Lru's map holds iterators into its own list, so it must never be copied.
        std::vector<Lru> result;
        result.reserve(LruMiB.size());
        for (const auto mib : LruMiB) result.emplace_back(mib * 1024u * 1024u / PageBytes);
        return result;
    }();
};

[[nodiscard]] bool Enabled();

// Driver hooks (no-ops when off).
void NoteGpuRead(Class which, std::uint64_t address, std::uint64_t bytes);
void NoteGpuWrite(Class which, std::uint64_t address, std::uint64_t bytes);
void NoteGpuReadGroup(const void* key, Class which, std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges);
// Whether the group `key` was already noted in this window (then the caller skips building it).
[[nodiscard]] bool NoteGroupSeen(const void* key);
// A CPU store into guest memory (agc_frame_replay: a restored memory delta).
void NoteCpuWrite(std::uint64_t address, std::uint64_t bytes);
// A fixed-function vertex buffer V# (dword 0-3): its range, as vertex-index.
void NoteVertexBuffer(std::span<const std::uint32_t, 4> vsharp);
// A committed draw's index buffer, fixed-function vertex buffers and indirect records.
void NoteDrawInputs(const Pm4::DrawParameters& draw, std::span<const CompiledShader> stages);

// Prints the window's line and starts the next (ProfileWindows::Close).
void Close(const char* window);

}

#endif
