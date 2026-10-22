#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

class DrawPipeline {
public:
    using Commit = std::function<void()>;
    // S1 (APS5_NO_SPLIT_DRAW_BUILD=1 off): an item's preparation, run by the preparer thread ahead
    // of the committer for items not yet committed (at most PrepareAhead() from the front). It
    // shares a PrepareSlot with the item's commit: the first of the two to claim the slot decides,
    // so a commit either takes the finished preparation, waits for one in progress, or runs alone.
    using Prepare = std::function<void()>;
    class PrepareSlot {
    public:
        // The preparer: claims the slot (false when the commit took it first).
        bool Begin();
        void Finish(std::shared_ptr<void> result);
        // The commit: the prepared result (waiting while the preparation runs), null when none ran.
        std::shared_ptr<void> Take();
    private:
        std::mutex mutex;
        std::condition_variable done;
        int state = 0;
        std::shared_ptr<void> result;
    };
    static std::size_t PrepareAhead();
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    enum class DrainReason : std::uint8_t { Packet, Flush, Labels, Capture, Indirect, Submission, Count };
    enum class Event : std::uint8_t { IndirectCommit, IndirectArgsDrain, SatisfiedWaitKept, Count };

    static DrawPipeline& Queue0();
    static std::size_t Depth();
    static std::size_t QueuedItems();
    static bool PipelineIndirect();
    static bool DrainBeforeRead(std::uint64_t address, std::size_t bytes);
    static bool& Active();
    static std::atomic<std::uint64_t>& EpochToken();
    static void FollowEpoch(std::uint64_t token);
    static unsigned Committers();

    void Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress = 0, std::vector<std::byte> labelBytes = {}, std::vector<Range> reads = {}, Prepare prepare = {});
    void Drain(DrainReason reason, std::uint32_t opcode = 0x100);
    void Stop();
    static std::uint64_t DrainRequestsByOpcode(std::uint32_t opcode);
    // DMA_DATA as pipeline items, and SET_PREDICATION, NOP sub-ops and non-overlapping DMA_DATA kept
    // without a drain (upstream #6208). LOCAL A/B: APS5_NO_PIPELINE_DMA=1 (any value but 0) restores
    // the behaviour before it: DMA_DATA runs on the worker after a drain.
    static bool DmaItems();
    bool Busy() const { return outstanding.load(std::memory_order_acquire) != 0; }
    std::size_t Outstanding() const { return outstanding.load(std::memory_order_acquire); }
    bool Overlaps(std::uint64_t address, std::size_t bytes);
    bool DrainIfOverlaps(std::span<const Range> ranges, DrainReason reason, std::uint32_t opcode = 0x100) {
        return DrainIfOverlaps(ranges, [](const Range& range) { return range; }, reason, opcode);
    }
    template<typename TRanges, typename TBounds>
    bool DrainIfOverlaps(const TRanges& ranges, TBounds&& bounds, DrainReason reason, std::uint32_t opcode = 0x100) {
        if (!Busy()) return false;
        if (!pendingWrites(ranges, bounds)) return false;
        Drain(reason, opcode);
        return true;
    }
    void Note(Event event) { events[static_cast<std::size_t>(event)].fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t Events(Event event) const { return events[static_cast<std::size_t>(event)].load(std::memory_order_relaxed); }
    bool OverlapsAny(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges);
    std::optional<std::uint64_t> PendingLabel(std::uint64_t address, std::size_t bytes);
    static bool HeldReads(std::uint64_t address, std::size_t bytes);
    void ReleaseRunningReads();

private:
    struct Item {
        Commit commit;
        std::vector<Range> writes;
        std::uint64_t labelAddress = 0;
        std::vector<std::byte> labelBytes;
        std::vector<Range> reads;
        std::int64_t pushedNs = 0;
        Prepare prepare;
        bool prepareTaken = false;
        // The span of `reads` (HeldReads skips an item it misses; APS5_NO_HELD_READ_BOUNDS=1 scans every range).
        Range readBounds{~std::uint64_t{0}, 0};
    };
    DrawPipeline() = default;
    template<typename TRanges, typename TBounds>
    bool pendingWrites(const TRanges& ranges, TBounds& bounds) {
        std::uint64_t lowest = ~std::uint64_t{0}, highest = 0;
        for (const auto& range : ranges) {
            const auto [first, end] = bounds(range);
            if (first >= end) continue;
            lowest = std::min(lowest, first);
            highest = std::max(highest, end);
        }
        if (lowest >= highest) return false;
        std::lock_guard lock(mutex);
        for (const auto& item : items) {
            for (const auto& [begin, limit] : item.writes) {
                if (begin >= highest || limit <= lowest) continue;
                for (const auto& range : ranges) {
                    const auto [first, end] = bounds(range);
                    if (first < limit && begin < end) return true;
                }
            }
        }
        return false;
    }
    void run();
    void prepareLoop();
    Item* nextPrepare();
    void retire(Item&& item);
    void reap();
    void rethrowFailure();
    void report(std::chrono::steady_clock::time_point now);

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::deque<Item> items;
    std::exception_ptr failure;
    std::atomic<std::size_t> outstanding{0};
    std::uint32_t idleWaiters = 0;
    bool committerWaiting = false;
    bool stopped = false;
    bool closing = false;
    std::thread thread;
    std::thread preparer;
    std::condition_variable prepareWake;
    bool preparerWaiting = false;
    std::mutex retireMutex;
    std::condition_variable reapWake;
    std::deque<Item> retired;
    std::thread reaper;
    bool reaperWaiting = false;
    std::uint64_t commits = 0;
    std::uint64_t commitNs = 0;
    std::uint64_t commitErrors = 0;
    std::uint64_t enqueued = 0;
    std::uint64_t fullWaitNs = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrainReason::Count)> drains{};
    std::array<std::uint64_t, static_cast<std::size_t>(DrainReason::Count)> drainWaitNs{};
    std::array<std::uint64_t, 257> drainOpcodes{};
    std::array<std::atomic<std::uint64_t>, 257> drainRequests{};
    std::uint64_t depthSum = 0;
    std::int64_t lastPopNs = 0;
    std::int64_t lastFinishNs = 0;
    std::uint64_t roomWaits = 0;
    std::uint64_t roomWakeNs = 0;
    std::uint64_t drainWaits = 0;
    std::uint64_t drainWakeNs = 0;
    std::uint64_t committerIdles = 0;
    std::uint64_t committerIdleNs = 0;
    std::uint64_t committerWakeNs = 0;
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Event::Count)> events{};
    std::array<std::uint64_t, static_cast<std::size_t>(Event::Count)> reportedEvents{};
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

}

#endif
