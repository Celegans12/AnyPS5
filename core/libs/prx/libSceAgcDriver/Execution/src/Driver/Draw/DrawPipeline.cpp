#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::array<const char*, static_cast<std::size_t>(DrawPipeline::DrainReason::Count)> DrainNames{"packet", "flush", "labels", "capture", "indirect", "submission"};

std::atomic<unsigned> runningCommitters{0};

std::uint64_t elapsedNs(std::chrono::steady_clock::time_point since) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
}

std::int64_t nowNs() {
    return static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

void traceWait(const char* scope, const char* stage, std::chrono::steady_clock::time_point start) {
    if (!PerformanceTrace::Recording()) return;
    if (auto* trace = PerformanceTrace::Thread()) trace->Add(scope, stage, start, std::chrono::steady_clock::now());
}

// S1 preparer counters (the [draw] pipelined line): preparations run and their time, slots the
// commit claimed first, commits that waited for a running preparation and how long.
std::atomic<std::uint64_t> preparedItems{0};
std::atomic<std::uint64_t> prepareNs{0};
std::atomic<std::uint64_t> prepareWaits{0};
std::atomic<std::uint64_t> prepareWaitNs{0};
std::atomic<std::uint64_t> prepareLost{0};

std::atomic<std::uint64_t> heldReadRanges{0};
std::atomic<std::uint64_t> heldReadQueries{0};
std::atomic<std::uint64_t> heldReadHits{0};

}

DrawPipeline& DrawPipeline::Queue0() {
    static DrawPipeline* pipeline = new DrawPipeline();
    return *pipeline;
}

std::size_t DrawPipeline::Depth() {
    static const std::size_t depth = [] {
        const char* text = std::getenv("APS5_PIPELINED_DRAWS");
        if (std::getenv("APS5_LOCKED_DRAW_PREPARE") != nullptr || std::getenv("APS5_DRAW_DRAIN") != nullptr || std::getenv("APS5_DRAIN_ALL") != nullptr || std::getenv("APS5_NO_WORDWISE_CAPTURE") != nullptr) return std::size_t{0};
        const std::size_t standard = PipelineIndirect() ? 64 : 8;
        if (text == nullptr) return standard;
        const auto value = std::strtoull(text, nullptr, 10);
        return value == 0 ? std::size_t{0} : value == 1 ? standard : static_cast<std::size_t>(std::min<unsigned long long>(value, 256));
    }();
    return depth;
}

std::size_t DrawPipeline::PrepareAhead() {
    static const std::size_t ahead = [] {
        const char* text = std::getenv("APS5_SPLIT_DRAW_AHEAD");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 16ull;
        return static_cast<std::size_t>(std::clamp<unsigned long long>(value, 1ull, 256ull));
    }();
    return ahead;
}

bool DrawPipeline::PrepareSlot::Begin() {
    std::lock_guard lock(mutex);
    if (state != 0) {
        prepareLost.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    state = 1;
    return true;
}

void DrawPipeline::PrepareSlot::Finish(std::shared_ptr<void> prepared) {
    {
        std::lock_guard lock(mutex);
        result = std::move(prepared);
        state = 2;
    }
    done.notify_all();
}

std::shared_ptr<void> DrawPipeline::PrepareSlot::Take() {
    std::unique_lock lock(mutex);
    if (state == 0) {
        state = 3;
        return nullptr;
    }
    if (state == 1) {
        const auto start = std::chrono::steady_clock::now();
        done.wait(lock, [&] { return state == 2; });
        prepareWaits.fetch_add(1, std::memory_order_relaxed);
        prepareWaitNs.fetch_add(elapsedNs(start), std::memory_order_relaxed);
        traceWait("DrawPipeline", "prepare_wait", start);
    }
    state = 3;
    return std::move(result);
}

DrawPipeline::Item* DrawPipeline::nextPrepare() {
    std::size_t index = 0;
    for (auto& item : items) {
        if (index++ >= PrepareAhead()) break;
        if (item.prepare && !item.prepareTaken) return &item;
    }
    return nullptr;
}

void DrawPipeline::prepareLoop() {
    PerformanceTrace::BindThread("draw preparer");
    for (;;) {
        Prepare job;
        {
            std::unique_lock lock(mutex);
            preparerWaiting = true;
            prepareWake.wait(lock, [&] { return nextPrepare() != nullptr; });
            preparerWaiting = false;
            auto* item = nextPrepare();
            item->prepareTaken = true;
            job = item->prepare;
        }
        const auto start = std::chrono::steady_clock::now();
        try {
            job();
        } catch (...) {
        }
        preparedItems.fetch_add(1, std::memory_order_relaxed);
        prepareNs.fetch_add(elapsedNs(start), std::memory_order_relaxed);
    }
}

bool DrawPipeline::PipelineIndirect() {
    static const bool enabled = [] { const char* text = std::getenv("APS5_PIPELINE_INDIRECT"); return text != nullptr && std::strcmp(text, "0") != 0; }();
    return enabled;
}

bool DrawPipeline::DrainBeforeRead(std::uint64_t address, std::size_t bytes) {
    if (!Active()) return false;
    auto& pipeline = Queue0();
    if (!pipeline.Busy() || !pipeline.Overlaps(address, bytes)) return false;
    pipeline.Drain(DrainReason::Capture);
    return true;
}

bool& DrawPipeline::Active() {
    static thread_local bool active = false;
    return active;
}

std::atomic<std::uint64_t>& DrawPipeline::EpochToken() {
    static std::atomic<std::uint64_t> token{1};
    return token;
}

void DrawPipeline::FollowEpoch(std::uint64_t token) {
    static thread_local std::uint64_t followed = 0;
    if (followed == token) return;
    followed = token;
    GuestMemory::BumpCollectEpoch();
}

unsigned DrawPipeline::Committers() {
    return runningCommitters.load(std::memory_order_acquire);
}

void DrawPipeline::rethrowFailure() {
    if (failure == nullptr) return;
    auto error = std::exchange(failure, nullptr);
    std::rethrow_exception(error);
}

void DrawPipeline::Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes, std::vector<Range> reads, Prepare prepare) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    std::unique_lock lock(mutex);
    rethrowFailure();
    if (closing) {
        lock.unlock();
        commit();
        return;
    }
    if (items.size() >= Depth()) {
        if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline full while the enqueuing thread holds the GPU lock");
        const auto start = std::chrono::steady_clock::now();
        ++idleWaiters;
        idle.wait(lock, [&] { return items.size() < Depth(); });
        --idleWaiters;
        fullWaitNs += elapsedNs(start);
        ++roomWaits;
        roomWakeNs += static_cast<std::uint64_t>(std::max<std::int64_t>(0, nowNs() - lastPopNs));
        traceWait("DrawPipeline", "room_wait", start);
        rethrowFailure();
    }
    depthSum += items.size();
    ++enqueued;
    heldReadRanges.fetch_add(reads.size(), std::memory_order_relaxed);
    const bool prepared = static_cast<bool>(prepare);
    items.push_back({std::move(commit), std::move(writes), labelAddress, std::move(labelBytes), std::move(reads), nowNs(), std::move(prepare)});
    for (const auto& [begin, limit] : items.back().reads) {
        items.back().readBounds.first = std::min(items.back().readBounds.first, begin);
        items.back().readBounds.second = std::max(items.back().readBounds.second, limit);
    }
    outstanding.fetch_add(1, std::memory_order_release);
    if (!thread.joinable()) {
        thread = std::thread([this] { run(); });
        runningCommitters.fetch_add(1, std::memory_order_acq_rel);
    }
    if (committerWaiting) wake.notify_one();
    if (prepared) {
        if (!preparer.joinable()) preparer = std::thread([this] { prepareLoop(); });
        if (preparerWaiting) prepareWake.notify_one();
    }
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport >= std::chrono::seconds(10)) report(now);
}

void DrawPipeline::report(std::chrono::steady_clock::time_point now) {
    const auto windowNs = std::max<double>(1.0, static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - lastReport).count()));
    lastReport = now;
    std::string drainText;
    for (std::size_t i = 0; i < drains.size(); ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu (%.1f ms)", DrainNames[i], static_cast<unsigned long long>(drains[i]), static_cast<double>(drainWaitNs[i]) / 1e6);
        drainText += item;
    }
    drainText += "; packet drains by opcode:";
    for (int shown = 0; shown < 5; ++shown) {
        const auto top = std::max_element(drainOpcodes.begin(), drainOpcodes.begin() + 0x100);
        if (*top == 0) break;
        char item[32];
        std::snprintf(item, sizeof(item), " 0x%02x x%llu", static_cast<unsigned>(top - drainOpcodes.begin()), static_cast<unsigned long long>(*top));
        drainText += item;
        *top = 0;
    }
    drainOpcodes = {};
    const auto perCommit = commits == 0 ? 0.0 : static_cast<double>(commitNs) / 1e3 / static_cast<double>(commits);
    const auto perDraw = enqueued == 0 ? 0.0 : static_cast<double>(fullWaitNs) / 1e3 / static_cast<double>(enqueued);
    const auto queued = enqueued == 0 ? 0.0 : static_cast<double>(depthSum) / static_cast<double>(enqueued);
    char reads[64];
    std::snprintf(reads, sizeof(reads), " %llu read ranges, %llu stores held", static_cast<unsigned long long>(heldReadRanges.load(std::memory_order_relaxed)), static_cast<unsigned long long>(heldReadHits.exchange(0, std::memory_order_relaxed)));
    drainText += reads;
    std::array<std::uint64_t, static_cast<std::size_t>(Event::Count)> counted{};
    for (std::size_t i = 0; i < counted.size(); ++i) {
        const auto total = events[i].load(std::memory_order_relaxed);
        counted[i] = total - reportedEvents[i];
        reportedEvents[i] = total;
    }
    const auto busy = static_cast<double>(commitNs) * 100.0 / windowNs;
    const auto average = [](std::uint64_t ns, std::uint64_t count) { return count == 0 ? 0.0 : static_cast<double>(ns) / 1e3 / static_cast<double>(count); };
    char handoffs[256];
    std::snprintf(handoffs, sizeof(handoffs), "; hand-offs: %llu room waits woken %.1f us after the pop, %llu drains woken %.1f us after the last pop, committer idle %.1f ms in %llu gaps woken %.1f us after the push", static_cast<unsigned long long>(roomWaits), average(roomWakeNs, roomWaits), static_cast<unsigned long long>(drainWaits), average(drainWakeNs, drainWaits), static_cast<double>(committerIdleNs) / 1e6, static_cast<unsigned long long>(committerIdles), average(committerWakeNs, committerIdles));
    drainText += handoffs;
    char kept[96];
    std::snprintf(kept, sizeof(kept), "; satisfied waits kept the pipeline %llu", static_cast<unsigned long long>(counted[static_cast<std::size_t>(Event::SatisfiedWaitKept)]));
    drainText += kept;
    char prepares[224];
    std::snprintf(prepares, sizeof(prepares), "; preparer: %llu prepared in %.1f ms, %llu claimed by the commit first, commits waited for %llu (%.1f ms)", static_cast<unsigned long long>(preparedItems.exchange(0, std::memory_order_relaxed)), static_cast<double>(prepareNs.exchange(0, std::memory_order_relaxed)) / 1e6, static_cast<unsigned long long>(prepareLost.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(prepareWaits.exchange(0, std::memory_order_relaxed)), static_cast<double>(prepareWaitNs.exchange(0, std::memory_order_relaxed)) / 1e6);
    drainText += prepares;
    std::fprintf(stderr, "[draw] pipelined (10 s, depth %zu): %llu commits at %.1f us each on the committer (%.1f %% busy), %llu failed; %llu enqueued behind %.2f on average, waiting %.1f us per draw for room;%s; indirect commits %llu, indirect drains (args overlap) %llu; drains:%s\n", Depth(), static_cast<unsigned long long>(commits), perCommit, busy, static_cast<unsigned long long>(commitErrors), static_cast<unsigned long long>(enqueued), queued, perDraw, reads, static_cast<unsigned long long>(counted[0]), static_cast<unsigned long long>(counted[1]), drainText.c_str());
    commits = commitNs = commitErrors = enqueued = fullWaitNs = depthSum = 0;
    roomWaits = roomWakeNs = drainWaits = drainWakeNs = committerIdles = committerIdleNs = committerWakeNs = 0;
    drains = {};
    drainWaitNs = {};
}

void DrawPipeline::run() {
    PerformanceTrace::BindThread("draw committer");
    StartCommitterSampler();
    for (;;) {
        Commit* commit = nullptr;
        SpinUntil(HandoffSpinWindow(), [&] { return outstanding.load(std::memory_order_acquire) != 0; });
        {
            std::unique_lock lock(mutex);
            committerWaiting = true;
            wake.wait(lock, [&] { return !items.empty() || closing; });
            committerWaiting = false;
            if (items.empty()) break;
            commit = stopped ? nullptr : &items.front().commit;
            items.front().prepareTaken = true;
            const auto begun = nowNs();
            if (lastFinishNs != 0 && items.front().pushedNs > lastFinishNs) {
                ++committerIdles;
                committerIdleNs += static_cast<std::uint64_t>(begun - lastFinishNs);
                committerWakeNs += static_cast<std::uint64_t>(std::max<std::int64_t>(0, begun - items.front().pushedNs));
            }
        }
        const auto start = std::chrono::steady_clock::now();
        std::exception_ptr fatal;
        bool failed = false;
        bool shutdown = false;
        try {
            if (commit != nullptr) (*commit)();
        } catch (const std::exception&) {
            failed = true;
            fatal = std::current_exception();
        } catch (const ProcessShutdown&) {
            shutdown = true;
            fatal = std::current_exception();
        } catch (...) {
            fatal = std::current_exception();
        }
        const auto spent = elapsedNs(start);
        traceWait("DrawPipeline", "commit", start);
        if (fatal != nullptr && !stopped) {
            stopped = true;
            if (!shutdown) Driver::Get().ReportFailure(fatal);
        }
        Item done;
        {
            std::lock_guard lock(mutex);
            done = std::move(items.front());
            items.pop_front();
            ++commits;
            commitNs += spent;
            if (failed) ++commitErrors;
            if (fatal != nullptr && failure == nullptr) failure = fatal;
            lastPopNs = lastFinishNs = nowNs();
            outstanding.fetch_sub(1, std::memory_order_release);
            if (idleWaiters != 0) idle.notify_all();
            if (preparerWaiting) prepareWake.notify_one();
        }
        retire(std::move(done));
    }
    runningCommitters.fetch_sub(1, std::memory_order_acq_rel);
}

void DrawPipeline::retire(Item&& item) {
    static const bool reaperEnabled = std::getenv("APS5_NO_COMMIT_REAPER") == nullptr;
    if (!reaperEnabled) {
        const Item dropped = std::move(item);
        return;
    }
    constexpr std::size_t ReapBatch = 64;
    std::lock_guard lock(retireMutex);
    retired.push_back(std::move(item));
    if (!reaper.joinable()) reaper = std::thread([this] { reap(); });
    if (retired.size() >= ReapBatch && reaperWaiting) reapWake.notify_one();
}

void DrawPipeline::reap() {
    for (;;) {
        std::deque<Item> batch;
        {
            std::unique_lock lock(retireMutex);
            reaperWaiting = true;
            reapWake.wait_for(lock, std::chrono::milliseconds(2), [&] { return retired.size() >= 64; });
            reaperWaiting = false;
            if (retired.empty()) continue;
            batch.swap(retired);
        }
        batch.clear();
    }
}

void DrawPipeline::Stop() {
    std::thread committer;
    {
        std::lock_guard lock(mutex);
        closing = true;
        committer = std::move(thread);
    }
    wake.notify_all();
    if (!committer.joinable()) return;
    if (committer.get_id() == std::this_thread::get_id()) committer.detach();
    else committer.join();
}

void DrawPipeline::Drain(DrainReason reason, std::uint32_t opcode) {
    drainRequests[std::min<std::uint32_t>(opcode, 0x100)].fetch_add(1, std::memory_order_relaxed);
    if (!Busy()) return;
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline drain under the GPU lock (reason " + std::to_string(static_cast<unsigned>(reason)) + ", opcode " + std::to_string(opcode) + ")");
    const auto start = std::chrono::steady_clock::now();
    SpinUntil(HandoffSpinWindow() * 2, [&] { return !Busy(); });
    std::unique_lock lock(mutex);
    ++idleWaiters;
    idle.wait(lock, [&] { return items.empty(); });
    --idleWaiters;
    const auto index = static_cast<std::size_t>(reason);
    ++drains[index];
    ++drainOpcodes[std::min<std::uint32_t>(opcode, 0x100)];
    drainWaitNs[index] += elapsedNs(start);
    ++drainWaits;
    drainWakeNs += static_cast<std::uint64_t>(std::max<std::int64_t>(0, nowNs() - lastPopNs));
    traceWait("DrawPipeline.drain", DrainNames[index], start);
    rethrowFailure();
}

bool DrawPipeline::DmaItems() {
    static const bool items = [] {
        const char* value = std::getenv("APS5_NO_PIPELINE_DMA");
        return value == nullptr || *value == '\0' || std::string_view(value) == "0";
    }();
    return items;
}

std::uint64_t DrawPipeline::DrainRequestsByOpcode(std::uint32_t opcode) {
    return Queue0().drainRequests[std::min<std::uint32_t>(opcode, 0x100)].load(std::memory_order_relaxed);
}

std::size_t DrawPipeline::QueuedItems() {
    auto& pipeline = Queue0();
    std::lock_guard lock(pipeline.mutex);
    return pipeline.items.size();
}

bool DrawPipeline::HeldReads(std::uint64_t address, std::size_t bytes) {
    static const bool unordered = std::getenv("APS5_NO_QUEUED_DRAW_READ_ORDER") != nullptr;
    if (bytes == 0 || unordered) return false;
    const auto end = address + bytes;
    auto& pipeline = Queue0();
    std::lock_guard lock(pipeline.mutex);
    heldReadQueries.fetch_add(1, std::memory_order_relaxed);
    static const bool bounds = std::getenv("APS5_NO_HELD_READ_BOUNDS") == nullptr;
    for (const auto& item : pipeline.items) {
        if (bounds && (address >= item.readBounds.second || item.readBounds.first >= end)) continue;
        for (const auto& [begin, limit] : item.reads) {
            if (address >= limit || begin >= end) continue;
            heldReadHits.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

void DrawPipeline::ReleaseRunningReads() {
    std::lock_guard lock(mutex);
    if (items.empty()) return;
    items.front().reads.clear();
    items.front().readBounds = {~std::uint64_t{0}, 0};
}

std::optional<std::uint64_t> DrawPipeline::PendingLabel(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || bytes > 8) return std::nullopt;
    const auto end = address + bytes;
    std::lock_guard lock(mutex);
    for (auto item = items.rbegin(); item != items.rend(); ++item) {
        const bool overlaps = std::any_of(item->writes.begin(), item->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
        if (!overlaps) continue;
        if (item->labelBytes.empty() || address < item->labelAddress || end > item->labelAddress + item->labelBytes.size()) return std::nullopt;
        std::uint64_t value = 0;
        std::memcpy(&value, item->labelBytes.data() + (address - item->labelAddress), bytes);
        return value;
    }
    return std::nullopt;
}

bool DrawPipeline::Overlaps(std::uint64_t address, std::size_t bytes) {
    const auto end = address + bytes;
    std::lock_guard lock(mutex);
    for (const auto& item : items) {
        for (const auto& [begin, limit] : item.writes) {
            if (address < limit && begin < end) return true;
        }
    }
    return false;
}

}
