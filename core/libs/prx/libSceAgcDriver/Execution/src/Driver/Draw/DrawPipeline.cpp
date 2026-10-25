#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ParallelProfile.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <string>
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

namespace {
thread_local bool finisherThread = false;
thread_local bool finisherProducer = false;
}

DrawFinisher& DrawFinisher::Get() {
    static DrawFinisher* finisher = new DrawFinisher();
    return *finisher;
}

bool DrawFinisher::Enabled() {
    return Graphics::DrawPipelineEnabled();
}

bool DrawPipeline::MetadataPassesPipelined() {
    return Graphics::DrawPipelineEnabled();
}

namespace {
std::atomic<std::uint64_t> pipelinedMetadataPasses{0};
}

std::uint64_t DrawPipeline::PipelinedMetadataPasses() {
    return pipelinedMetadataPasses.load(std::memory_order_relaxed);
}

void DrawPipeline::CountPipelinedMetadataPass() {
    pipelinedMetadataPasses.fetch_add(1, std::memory_order_relaxed);
}

bool DrawFinisher::Overlap() {
    return Graphics::DrawPipelineEnabled();
}

bool DrawFinisher::PendingOverlaps(std::uint64_t begin, std::uint64_t end) {
    if (!Busy() || !finisherProducer) return false;
    std::lock_guard lock(mutex);
    for (const auto& held : pending) {
        if (held.unknown) return true;
        for (const auto& [first, last] : held.writes) {
            if (begin < last && first < end) return true;
        }
    }
    return false;
}

DrawFinisher::LabelLookup DrawFinisher::PendingLabel(std::uint64_t address, std::size_t bytes, std::uint64_t& value) {
    if (!Busy() || !finisherProducer) return LabelLookup::NotHeld;
    const auto end = address + bytes;
    std::lock_guard lock(mutex);
    for (auto held = pending.rbegin(); held != pending.rend(); ++held) {
        if (held->unknown) return LabelLookup::Unknown;
        const bool overlaps = std::any_of(held->writes.begin(), held->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
        if (!overlaps) continue;
        if (held->labelBytes.empty() || address < held->labelAddress || end > held->labelAddress + held->labelBytes.size()) return LabelLookup::Unknown;
        value = 0;
        std::memcpy(&value, held->labelBytes.data() + (address - held->labelAddress), bytes);
        return LabelLookup::Value;
    }
    return LabelLookup::NotHeld;
}

bool DrawFinisher::PendingOverlapsAny(std::span<const Range> ranges) {
    if (!Busy() || !finisherProducer) return false;
    std::lock_guard lock(mutex);
    for (const auto& held : pending) {
        if (held.unknown) return true;
        for (const auto& [first, last] : held.writes) {
            for (const auto& [begin, end] : ranges) {
                if (begin < last && first < end) return true;
            }
        }
    }
    return false;
}

bool DrawFinisher::PendingKey(std::uint64_t key, std::uint64_t shape) {
    if (!Busy() || !finisherProducer) return false;
    std::lock_guard lock(mutex);
    return std::any_of(pending.begin(), pending.end(), [&](const Pending& held) { return held.key == key || (shape != 0 && held.shape == shape); });
}

namespace {
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawFinisher::DrainSite::Count)> finisherDrains{}, finisherSkips{};
}

void DrawFinisher::NoteDrain(DrainSite site, bool drained) {
    (drained ? finisherDrains : finisherSkips)[static_cast<std::size_t>(site)].fetch_add(1, std::memory_order_relaxed);
}

namespace {
std::array<std::atomic<std::uint64_t>, 257> drainedOpcodes{}, drainedBeside{};
}

void DrawFinisher::NoteDrainedOpcode(std::uint32_t opcode, bool besideCommits) {
    drainedOpcodes[std::min<std::uint32_t>(opcode, 256u)].fetch_add(1, std::memory_order_relaxed);
    if (besideCommits) drainedBeside[std::min<std::uint32_t>(opcode, 256u)].fetch_add(1, std::memory_order_relaxed);
}

std::string DrawFinisher::DrainText() {
    constexpr std::array<const char*, static_cast<std::size_t>(DrainSite::Count)> names{"lookup", "reads", "packet", "kept draw"};
    std::string text = "; worker drains of the finisher (drained/skipped):";
    for (std::size_t i = 0; i < names.size(); ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu/%llu", names[i], static_cast<unsigned long long>(finisherDrains[i].exchange(0)), static_cast<unsigned long long>(finisherSkips[i].exchange(0)));
        text += item;
    }
    text += "; drained by opcode (of them beside commits):";
    for (int shown = 0; shown < 6; ++shown) {
        std::size_t top = 0;
        for (std::size_t i = 1; i < drainedOpcodes.size(); ++i) {
            if (drainedOpcodes[i].load() > drainedOpcodes[top].load()) top = i;
        }
        const auto count = drainedOpcodes[top].exchange(0);
        if (count == 0) break;
        char item[48];
        std::snprintf(item, sizeof(item), " 0x%02zx x%llu (%llu)", top, static_cast<unsigned long long>(count), static_cast<unsigned long long>(drainedBeside[top].exchange(0)));
        text += item;
    }
    return text;
}

bool DrawFinisher::OnFinisherThread() {
    return finisherThread;
}

void DrawFinisher::Enqueue(Job job) {
    Enqueue(std::move(job), Pending{});
}

void DrawFinisher::Enqueue(Job job, Pending held) {
    finisherProducer = true;
    constexpr std::size_t depth = 16;
    struct ReclaimStorage {};
    auto& reclaim = HostThreadLocal<std::vector<Job>, ReclaimStorage>();
    reclaim.clear();
    std::unique_lock lock(mutex);
    if (!ran.empty()) std::swap(ran, reclaim);
    if (failure != nullptr) std::rethrow_exception(std::exchange(failure, nullptr));
    if (jobs.size() >= depth && GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher full while the enqueuing thread holds the GPU lock");
    idle.wait(lock, [&] { return jobs.size() < depth; });
    jobs.push_back(std::move(job));
    pending.push_back(std::move(held));
    outstanding.fetch_add(1, std::memory_order_release);
    if (!running) {
        running = true;
        thread = std::thread([this] {
            finisherThread = true;
            run();
        });
    }
    if (sleeping) wake.notify_one();
}

void DrawFinisher::run() {
    PerformanceTrace::BindThread("draw finisher");
    for (;;) {
        Job* job = nullptr;
        bool failed = false;
        SpinUntil(HandoffSpinWindow(), [&] { return outstanding.load(std::memory_order_acquire) != 0; });
        {
            std::unique_lock lock(mutex);
            sleeping = true;
            wake.wait(lock, [&] { return !jobs.empty(); });
            sleeping = false;
            job = &jobs.front();
            failed = failure != nullptr;
        }
        std::exception_ptr error;
        try {
            if (!failed) (*job)();
        } catch (...) {
            error = std::current_exception();
        }
        std::lock_guard lock(mutex);
        if (error != nullptr && failure == nullptr) failure = error;
        ran.push_back(std::move(jobs.front()));
        jobs.pop_front();
        pending.pop_front();
        outstanding.fetch_sub(1, std::memory_order_release);
        idle.notify_all();
    }
}

void DrawFinisher::Settle() {
    if (!Busy() || OnFinisherThread()) return;
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher settle under the GPU lock");
    std::unique_lock lock(mutex);
    idle.wait(lock, [&] { return jobs.empty(); });
}

void DrawFinisher::Drain() {
    if (OnFinisherThread()) return;
    if (Busy() && GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher drain under the GPU lock");
    SpinUntil(HandoffSpinWindow() * 2, [&] { return outstanding.load(std::memory_order_acquire) == 0; });
    std::unique_lock lock(mutex);
    if (!jobs.empty()) idle.wait(lock, [&] { return jobs.empty(); });
    if (failure != nullptr) std::rethrow_exception(std::exchange(failure, nullptr));
}

void DrawFinisher::ReleaseRan() {
    std::vector<Job> done;
    std::lock_guard lock(mutex);
    std::swap(done, ran);
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

void DrawPipeline::Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes, std::vector<Range> reads, Prepare prepare, bool draw) {
    if (DrawFinisher::Busy() && !DrawFinisher::OnFinisherThread()) DrawFinisher::Get().Drain();
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto lockStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::unique_lock lock(mutex);
    if (profile) lockWaitNs += elapsedNs(lockStart);
    rethrowFailure();
    if (closing) {
        lock.unlock();
        commit();
        return;
    }
    if (items.size() >= Depth()) {
        if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline full while the enqueuing thread holds the GPU lock");
        const auto start = std::chrono::steady_clock::now();
        resumeAt = Depth() - 1;
        ++idleWaiters;
        idle.wait(lock, [&] { return items.size() <= resumeAt; });
        --idleWaiters;
        resumeAt = 0;
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
    items.back().draw = draw;
    addWrites(items.back().writes);
    for (const auto& [begin, limit] : items.back().reads) {
        items.back().readBounds.first = std::min(items.back().readBounds.first, begin);
        items.back().readBounds.second = std::max(items.back().readBounds.second, limit);
    }
    outstanding.fetch_add(1, std::memory_order_release);
    if (!thread.joinable()) {
        thread = std::thread([this] {
            GuestMemory::MarkProofThread();
            run();
        });
        runningCommitters.fetch_add(1, std::memory_order_acq_rel);
    }
    if (committerWaiting) {
        ++wakes;
        wake.notify_one();
    }
    if (prepared) {
        if (!preparer.joinable()) preparer = std::thread([this] { prepareLoop(); });
        if (preparerWaiting) prepareWake.notify_one();
    }
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!profile && !stats) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport >= std::chrono::seconds(10)) report(now);
}

// APS5_PROFILE_DRAW: the pending-write scans (Overlaps, OverlapsAny, DrainIfOverlaps) walk every write range of
// every pending item; every 256th scan samples how many items and ranges that is and how many ranges are distinct.
// Under mutex.
void DrawPipeline::noteWriteScan() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile || (writeScans++ & 255u) != 0) return;
    static std::vector<Range> ranges;
    ranges.clear();
    for (const auto& item : items) ranges.insert(ranges.end(), item.writes.begin(), item.writes.end());
    ++writeScanSamples;
    writeScanItems += items.size();
    writeScanRanges += ranges.size();
    std::sort(ranges.begin(), ranges.end());
    writeScanDistinct += static_cast<std::uint64_t>(std::unique(ranges.begin(), ranges.end()) - ranges.begin());
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
    const auto decodeDrainCount = decodeDrains.load(std::memory_order_relaxed);
    char reads[96];
    std::snprintf(reads, sizeof(reads), " %llu read ranges, %llu stores held, %llu decode drains", static_cast<unsigned long long>(heldReadRanges.load(std::memory_order_relaxed)), static_cast<unsigned long long>(heldReadHits.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(decodeDrainCount - reportedDecodeDrains));
    reportedDecodeDrains = decodeDrainCount;
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
    const auto perSample = [&](std::uint64_t value) { return writeScanSamples == 0 ? 0.0 : static_cast<double>(value) / static_cast<double>(writeScanSamples); };
    char scans[160];
    std::snprintf(scans, sizeof(scans), "; write scans %llu over %.1f items, %.1f ranges, %.1f distinct", static_cast<unsigned long long>(writeScans), perSample(writeScanItems), perSample(writeScanRanges), perSample(writeScanDistinct));
    drainText += scans;
    writeScans = writeScanSamples = writeScanItems = writeScanRanges = writeScanDistinct = 0;
    AgcDriver::ProfilePrint_nid_no_patch("[draw] pipelined (10 s, depth %zu): %llu commits at %.1f us each on the committer (%.1f %% busy), %llu failed; %llu enqueued behind %.2f on average, waiting %.1f us per draw for room, %.2f us for the lock; %llu committer wakes;%s; indirect commits %llu, indirect drains (args overlap) %llu; drains:%s\n", Depth(), static_cast<unsigned long long>(commits), perCommit, busy, static_cast<unsigned long long>(commitErrors), static_cast<unsigned long long>(enqueued), queued, perDraw, enqueued == 0 ? 0.0 : static_cast<double>(lockWaitNs) / 1e3 / static_cast<double>(enqueued), static_cast<unsigned long long>(wakes), reads, static_cast<unsigned long long>(counted[0]), static_cast<unsigned long long>(counted[1]), drainText.c_str());
    commits = commitNs = commitErrors = enqueued = fullWaitNs = depthSum = wakes = lockWaitNs = 0;
    roomWaits = roomWakeNs = drainWaits = drainWakeNs = committerIdles = committerIdleNs = committerWakeNs = 0;
    drains = {};
    drainWaitNs = {};
}


void DrawPipeline::run() {
    PerformanceTrace::BindThread("draw committer");
    StartCommitterSampler();
    Graphics::ParallelProfile::MarkCommitter();
    for (;;) {
        Commit* commit = nullptr;
        bool isDraw = false;
        bool isLabel = false;
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
            isDraw = items.front().draw;
            isLabel = items.front().labelAddress != 0;
        }
        const auto start = std::chrono::steady_clock::now();
        const bool profiled = Graphics::ParallelProfile::On();
        const auto heldBefore = profiled ? GuestMemory::ThreadHeldNs() : 0;
        const auto lockedBefore = GuestMemory::ThreadLockedGpuWaitMs();
        std::exception_ptr fatal;
        bool failed = false;
        bool shutdown = false;
        try {
            if (commit != nullptr) (*commit)();
            if (!isDraw) GuestMemory::BreakProofs();
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
        if (profiled) {
            Graphics::ParallelProfile::ItemSample sample;
            sample.kind = isDraw ? Graphics::ParallelProfile::Item::Draw : isLabel ? Graphics::ParallelProfile::Item::Label : Graphics::ParallelProfile::Item::Other;
            sample.wallUs = static_cast<double>(spent) / 1e3;
            sample.heldUs = static_cast<double>(GuestMemory::ThreadHeldNs() - heldBefore) / 1e3;
            sample.lockedWaitUs = (GuestMemory::ThreadLockedGpuWaitMs() - lockedBefore) * 1e3;
            Graphics::ParallelProfile::NoteItem(sample);
        }
        if (fatal != nullptr && !stopped) {
            stopped = true;
            if (!shutdown) Driver::Get().ReportFailure(fatal);
        }
        Item done;
        {
            std::lock_guard lock(mutex);
            done = std::move(items.front());
            items.pop_front();
            removeWrites(done.writes);
            retiredItems.fetch_add(1, std::memory_order_release);
            ++commits;
            commitNs += spent;
            if (failed) ++commitErrors;
            if (fatal != nullptr && failure == nullptr) failure = fatal;
            lastPopNs = lastFinishNs = nowNs();
            outstanding.fetch_sub(1, std::memory_order_release);
            if (idleWaiters != 0 && (items.empty() || items.size() <= resumeAt)) idle.notify_all();
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

void DrawPipeline::ReleaseRetired() {
    std::deque<Item> done;
    std::lock_guard lock(retireMutex);
    std::swap(done, retired);
}

void DrawPipeline::Drain(DrainReason reason, std::uint32_t opcode) {
    drainRequests[std::min<std::uint32_t>(opcode, 0x100)].fetch_add(1, std::memory_order_relaxed);
    const bool finisherBusy = DrawFinisher::Busy() && !DrawFinisher::OnFinisherThread();
    if ((finisherBusy || Busy()) && GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline drain under the GPU lock (reason " + std::to_string(static_cast<unsigned>(reason)) + ", opcode " + std::to_string(opcode) + ")");
    if (finisherBusy) DrawFinisher::Get().Drain();
    if (!Busy()) return;
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

std::uint64_t DrawPipeline::HeldReadQueries() {
    return heldReadQueries.load(std::memory_order_relaxed);
}

void DrawPipeline::ReleaseRunningReads() {
    std::lock_guard lock(mutex);
    if (items.empty()) return;
    items.front().reads.clear();
    items.front().readBounds = {~std::uint64_t{0}, 0};
}

std::optional<std::uint64_t> DrawPipeline::PendingLabel(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || bytes > 8) return std::nullopt;
    if (DrawFinisher::Overlap()) {
        std::uint64_t held = 0;
        const auto lookup = DrawFinisher::Get().PendingLabel(address, bytes, held);
        if (lookup == DrawFinisher::LabelLookup::Value) return held;
        if (lookup == DrawFinisher::LabelLookup::Unknown) return std::nullopt;
    }
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

// APS5_NO_DISTINCT_WRITE_SCANS=1 (runtime switch): the pending-write scans walk every pending item's writes.
bool DrawPipeline::DistinctWriteScans() {
    return !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoDistinctWriteScans);
}

// Under mutex, for every item entering or leaving `items`, whatever the switch, so the list stays exact when it flips.
void DrawPipeline::addWrites(const std::vector<Range>& writes) {
    for (const auto& write : writes) {
        const auto found = std::ranges::find(distinctWrites, write, &std::pair<Range, std::uint32_t>::first);
        if (found != distinctWrites.end()) ++found->second;
        else distinctWrites.push_back({write, 1u});
    }
}

void DrawPipeline::removeWrites(const std::vector<Range>& writes) {
    for (const auto& write : writes) {
        const auto found = std::ranges::find(distinctWrites, write, &std::pair<Range, std::uint32_t>::first);
        if (found == distinctWrites.end()) continue;
        if (--found->second == 0u) {
            *found = distinctWrites.back();
            distinctWrites.pop_back();
        }
    }
}

bool DrawPipeline::OverlapsAny(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    if (ranges.empty()) return false;
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlapsAny(ranges)) return true;
    auto low = ranges.front().first;
    auto high = ranges.front().second;
    for (const auto& [begin, end] : ranges) {
        low = std::min(low, begin);
        high = std::max(high, end);
    }
    std::lock_guard lock(mutex);
    noteWriteScan();
    const auto overlaps = [&](std::uint64_t begin, std::uint64_t limit) {
        if (begin >= high || limit <= low) return false;
        return std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return range.first < limit && begin < range.second; });
    };
    if (DistinctWriteScans()) return std::ranges::any_of(distinctWrites, [&](const auto& write) { return overlaps(write.first.first, write.first.second); });
    for (const auto& item : items) {
        for (const auto& [begin, limit] : item.writes) {
            if (overlaps(begin, limit)) return true;
        }
    }
    return false;
}

DrawPipeline::Watch DrawPipeline::WatchWrites() {
    if (!Busy()) return {Retired(), 0, {}};
    std::lock_guard lock(mutex);
    Watch watch{retiredItems.load(std::memory_order_relaxed), items.size(), {}};
    for (const auto& item : items) watch.writes.insert(watch.writes.end(), item.writes.begin(), item.writes.end());
    return watch;
}

bool DrawPipeline::WrittenSince(const Watch& watch, const std::vector<Range>& reads) {
    const auto overlaps = [&reads](const std::vector<Range>& writes) {
        return std::any_of(reads.begin(), reads.end(), [&writes](const Range& read) {
            return std::any_of(writes.begin(), writes.end(), [&read](const Range& write) { return read.first < write.second && write.first < read.second; });
        });
    };
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlapsAny(reads)) return true;
    std::lock_guard lock(mutex);
    if (std::any_of(items.begin(), items.end(), [&](const Item& item) { return overlaps(item.writes); })) return true;
    const auto retiredNow = retiredItems.load(std::memory_order_relaxed);
    if (retiredNow == watch.retired) return false;
    return retiredNow - watch.retired > watch.items || overlaps(watch.writes);
}

bool DrawPipeline::Busy() const {
    return outstanding.load(std::memory_order_acquire) != 0 || (finisherProducer && DrawFinisher::Busy() && DrawFinisher::Overlap());
}

bool DrawPipeline::Overlaps(std::uint64_t address, std::size_t bytes) {
    const auto end = address + bytes;
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlaps(address, end)) return true;
    std::lock_guard lock(mutex);
    noteWriteScan();
    if (DistinctWriteScans()) return std::ranges::any_of(distinctWrites, [&](const auto& write) { return address < write.first.second && write.first.first < end; });
    for (const auto& item : items) {
        for (const auto& [begin, limit] : item.writes) {
            if (address < limit && begin < end) return true;
        }
    }
    return false;
}

}
