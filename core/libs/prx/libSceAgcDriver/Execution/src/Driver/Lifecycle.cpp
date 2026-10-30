#include "prx/libSceAgcDriver/Execution/include/MemoryStats.hpp"
#include "prx/libSceAgcDriver/Execution/include/AsyncLog.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineJournal.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ParallelCompare.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineLibrary.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "ShaderDiskCache.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include <algorithm>
#include <atomic>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cerrno>
#include <exception>
#include <stdexcept>
#include <system_error>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <ucontext.h>
#endif

namespace AgcDriver::DriverDetail {

namespace {

#if defined(__linux__)
// A SIGSEGV or SIGBUS that no handler takes ends the process, often without a kept core: report the faulting
// address, the instruction and a backtrace first. Earlier handlers (write watch, the guest's exception handlers)
// run first and keep their behaviour; this only speaks when the fault would be fatal anyway.
struct sigaction previousSegv{}, previousBus{};

void describeAddress(std::uintptr_t address) {
    Dl_info where{};
    if (address == 0 || dladdr(reinterpret_cast<void*>(address), &where) == 0 || where.dli_fname == nullptr) {
        std::fprintf(stderr, "0x%llx", static_cast<unsigned long long>(address));
        return;
    }
    const auto base = reinterpret_cast<std::uintptr_t>(where.dli_fbase);
    std::fprintf(stderr, "0x%llx %s+0x%llx %s", static_cast<unsigned long long>(address), where.dli_fname, static_cast<unsigned long long>(address - base), where.dli_sname != nullptr ? where.dli_sname : "");
}

void fatalFault(int signal, siginfo_t* info, void* context) {
    const auto& previous = signal == SIGBUS ? previousBus : previousSegv;
    if ((previous.sa_flags & SA_SIGINFO) != 0 && previous.sa_sigaction != nullptr) {
        previous.sa_sigaction(signal, info, context);
        return;
    }
    if (previous.sa_handler != SIG_DFL && previous.sa_handler != SIG_IGN && previous.sa_handler != nullptr) {
        previous.sa_handler(signal);
        return;
    }
    const auto* frame = static_cast<const ucontext_t*>(context);
    std::fprintf(stderr, "[fault] signal %d code %d at address 0x%llx, thread %ld, instruction ", signal, info->si_code, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info->si_addr)), static_cast<long>(gettid()));
    describeAddress(frame != nullptr ? static_cast<std::uintptr_t>(frame->uc_mcontext.gregs[REG_RIP]) : 0);
    std::fprintf(stderr, "\n");
    void* frames[48];
    const int count = backtrace(frames, 48);
    for (int index = 0; index < count; ++index) {
        std::fprintf(stderr, "[fault]   #%d ", index);
        describeAddress(reinterpret_cast<std::uintptr_t>(frames[index]));
        std::fprintf(stderr, "\n");
    }
    std::fflush(stderr);
    ::signal(signal, SIG_DFL);
}

void installFatalFaultReport() {
    struct sigaction action{};
    action.sa_sigaction = &fatalFault;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask);
    sigaction(SIGSEGV, &action, &previousSegv);
    sigaction(SIGBUS, &action, &previousBus);
}
#endif

constexpr std::uintptr_t InlineImportStackBytes = 256 * 1024;

std::atomic<std::uint64_t> gpuMapImportThreads{0};

void PrintFailure(std::exception_ptr error) {
    try {
        std::rethrow_exception(error);
    } catch (const std::exception& cause) {
        std::fprintf(stderr, "[gpu] driver failure: %s\n", cause.what());
    } catch (...) {
        std::fprintf(stderr, "[gpu] driver failure with a non-standard exception\n");
    }
    FlushAsyncLog_nid_no_patch();
    std::fflush(stderr);
}

constexpr std::uint64_t DriverMemoryAddress = 0xfe0000000ull + GuestArena::GuestAddressOffset;
constexpr std::size_t DriverMemoryBytes = 2u << 20u;
constexpr std::uint64_t FsTableAddress = DriverMemoryAddress + 0x40000ull;
constexpr std::uint64_t FsTableSlotBytes = 16u;
constexpr std::uint64_t FsTableSlots = 6u;
constexpr std::uint64_t FsRoutineRecords = 8u;
constexpr std::uint64_t FsRoutineRecordBytes = 16u;
constexpr std::uint64_t FsRoutineTableBytes = FsRoutineRecords * FsRoutineRecordBytes;
constexpr std::uint64_t FsRoutineTableAddress = FsTableAddress + FsTableSlots * FsTableSlotBytes;
constexpr std::uint64_t FsRoutineAddress = FsRoutineTableAddress + FsRoutineTableBytes;
constexpr std::uint32_t ScalarSetpcB64LinkS14 = 0xbe80200eu;
constexpr std::uint32_t ScalarEndProgram = 0xbf810000u;
constexpr std::uint64_t FsTableDword3 = 0x5204u;

void MapDriverMemory() {
    void* mapped;
#ifdef _WIN32
    mapped = reinterpret_cast<void*>(DriverMemoryAddress);
    GuestArena::GuestArenaCommit_nid_postfix(mapped, DriverMemoryBytes, PAGE_READWRITE, DriverMemoryBytes);
#else
    mapped = mmap(reinterpret_cast<void*>(DriverMemoryAddress), DriverMemoryBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (mapped == MAP_FAILED) throw std::system_error(errno, std::generic_category(), "map the AGC driver memory");
#endif
    if (reinterpret_cast<std::uint64_t>(mapped) != DriverMemoryAddress) throw std::runtime_error("the AGC driver memory was not mapped at its fixed address");
    auto* routineTable = reinterpret_cast<std::uint64_t*>(FsRoutineTableAddress);
    for (std::uint64_t record = 0; record < FsRoutineRecords; ++record) {
        routineTable[record * 2u] = FsRoutineAddress;
        routineTable[record * 2u + 1u] = 0u;
    }
    auto* routine = reinterpret_cast<std::uint32_t*>(FsRoutineAddress);
    routine[0] = ScalarSetpcB64LinkS14;
    routine[1] = ScalarEndProgram;
    auto* slot = reinterpret_cast<std::uint64_t*>(FsTableAddress);
    slot[0] = (FsRoutineTableAddress & 0xffffffffffffull) | 16ull << 48u;
    slot[1] = FsTableDword3 << 32u | FsRoutineRecords;
#ifndef _WIN32
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(mapped, DriverMemoryBytes);
#endif
    GuestAllocations::Mutation mutation;
    mutation.Add(mapped, DriverMemoryBytes, true, true, true);
}

bool StackHoldsImport() {
#ifdef _WIN32
    ULONG_PTR low = 0;
    ULONG_PTR high = 0;
    GetCurrentThreadStackLimits(&low, &high);
#else
    struct Bounds {
        std::uintptr_t low = 0;
        std::uintptr_t high = 0;
        Bounds() {
            pthread_attr_t attributes;
            if (pthread_getattr_np(pthread_self(), &attributes) != 0) return;
            void* base = nullptr;
            std::size_t size = 0;
            if (pthread_attr_getstack(&attributes, &base, &size) == 0 && base != nullptr) {
                low = reinterpret_cast<std::uintptr_t>(base);
                high = low + size;
            }
            pthread_attr_destroy(&attributes);
        }
    };
    thread_local const Bounds bounds;
    const auto low = bounds.low;
    const auto high = bounds.high;
#endif
    const auto frame = reinterpret_cast<std::uintptr_t>(__builtin_frame_address(0));
    return frame >= low && frame < high && frame - low >= InlineImportStackBytes;
}

}

std::uint64_t Driver::GpuMapImportThreads() {
    return gpuMapImportThreads.load(std::memory_order_relaxed);
}

Driver& Driver::Get() {
    static Driver driver;
    return driver;
}

Driver::~Driver() {
    GuestArena::GuestArenaSetHostWriteHook_nid_no_patch(nullptr);
    GuestAllocations::GuestAllocationsSetGpuMapObserver_nid_postfix(nullptr);
    stop();
}

void Driver::Shutdown() {
    stop();
    std::lock_guard lock(mutex);
    rethrowFailure();
}

void Driver::stop() {
    require(!onWorkerThread(), "worker cannot stop itself");
    std::lock_guard shutdownLock(shutdownMutex);
    if (stopped) return;
    MemoryStats::Stop();
    LibcRequestShutdown_nid_postfix();
    {
        std::lock_guard lock(mutex);
        stopping = true;
        eopFlips.clear();
        for (auto& [queue, worker] : workers) {
            worker.pending.clear();
            worker.unfinishedWrites.clear();
            worker.queued.store(0, std::memory_order_release);
        }
    }
    changed.notify_all();
    for (auto& [queue, worker] : workers) {
        if (worker.thread.joinable()) worker.thread.join();
    }
    drawAhead.reset();
    try {
        DrawFinisher::Get().Drain();
    } catch (...) {
    }
    DrawPipeline::Queue0().Stop();
    DrawPipeline::Queue0().ReleaseRetired();
    DrawFinisher::Get().ReleaseRan();
    StopWorkerSampler();
    CloseRegistrationPreparation();
    Graphics::StopPipelineLibraryWorker();
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    Graphics::ReportHostImports();
    device.Reset();
    replacedDevices.clear();
    Graphics::FlushPipelineJournal();
    Graphics::ShutdownGuestBufferWorkers();
    Graphics::StopCompareHelpers();
    ShaderRecompiler::ShaderDiskCache::Shutdown();
    PerformanceTrace::StopOutput();
    FlushAsyncLogAtEnd_nid_no_patch();
    StopProfileOutput_nid_no_patch();
    stopped = true;
}

Driver::Driver() {
    InstallAsyncLog_nid_no_patch();
#if defined(__linux__)
    static const bool faultReport = (installFatalFaultReport(), true);
    static_cast<void>(faultReport);
#endif
    static const bool driverMemory = (MapDriverMemory(), true);
    static_cast<void>(driverMemory);
    LibcSetExitDrain_nid_no_patch([] {
        const auto current = Driver::Get().device.Load();
        if (current) current->SavePipelineCache();
    });
    static_cast<void>(GuestMemory::TrackerGeneration());
    ShaderMemory::SetWaitedMsProvider(&Graphics::Recorder::ThreadWaitedMs);
    try {
        LibcRegisterShutdown_nid_postfix([] { Driver::Get().Shutdown(); });
    } catch (...) {
        stop();
        throw;
    }
    GuestArena::GuestArenaSetHostWriteHook_nid_no_patch(&GuestMemory::PrepareHostWrite);
    GuestAllocations::GuestAllocationsSetGpuMapObserver_nid_postfix([](const GuestAllocations::Mapped& ranges, std::uint64_t generation) {
        const auto import = [&] {
            try {
                const auto current = Driver::Get().device.Load();
                if (current == nullptr || current->ImportGuestMemory(ranges, generation, false)) return;
                std::lock_guard lock(GuestMemory::GpuMutex());
                if (Driver::Get().device.Load() == current) current->ImportGuestMemory(ranges, generation, true);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[gpu] import of GPU memory at its mapping failed: %s\n", error.what());
            }
        };
        if (GuestMemory::GpuMutex().HeldByThisThread() || StackHoldsImport()) {
            import();
            return;
        }
        std::thread host;
        try {
            host = std::thread(import);
        } catch (const std::system_error& error) {
            std::fprintf(stderr, "[gpu] import of GPU memory at its mapping failed: no host thread to run it on: %s\n", error.what());
            return;
        }
        gpuMapImportThreads.fetch_add(1, std::memory_order_relaxed);
        host.join();
    });
}

void Driver::WaitIdle() {
    require(!onWorkerThread(), "worker cannot wait for itself");
    {
        std::unique_lock lock(mutex);
        const auto target = accepted;
        ++idleWaiters;
        changed.wait(lock, [&] { return failure != nullptr || stopping || completed >= target; });
        --idleWaiters;
        rethrowFailure();
        checkStopping();
    }
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    if (const auto localDevice = device.Load()) localDevice->WaitIdle();
}

bool Driver::WaitIdleFor(std::chrono::milliseconds timeout) {
    require(!onWorkerThread(), "worker cannot wait for itself");
    std::unique_lock lock(mutex);
    const auto target = accepted;
    ++idleWaiters;
    const bool idle = changed.wait_for(lock, timeout, [&] { return failure != nullptr || stopping || completed >= target; });
    --idleWaiters;
    rethrowFailure();
    checkStopping();
    return idle;
}

bool Driver::blockedOnWaits() {
    if (!Quiescent()) return false;
    return std::any_of(awaitedSlots.begin(), awaitedSlots.end(), [](const AwaitedSlot& slot) { return slot.address.load(std::memory_order_acquire) != 0; });
}

Driver::CaptureSettle Driver::WaitCompletedOrBlockedFor(std::chrono::milliseconds timeout) {
    require(!onWorkerThread(), "worker cannot wait for itself");
    constexpr auto Quiet = std::chrono::milliseconds(5);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::uint64_t target = 0;
    {
        std::lock_guard lock(mutex);
        target = accepted;
    }
    std::optional<std::chrono::steady_clock::time_point> blockedSince;
    std::uint64_t lastCompleted = 0;
    std::uint64_t releases = WaitReleases();
    while (true) {
        std::uint64_t seen = 0;
        {
            std::unique_lock lock(mutex);
            ++idleWaiters;
            const bool done = changed.wait_for(lock, std::chrono::milliseconds(1), [&] { return failure != nullptr || stopping || completed >= target; });
            --idleWaiters;
            rethrowFailure();
            checkStopping();
            if (done) return CaptureSettle::Completed;
            seen = completed;
        }
        const auto now = std::chrono::steady_clock::now();
        const auto released = WaitReleases();
        if (seen != lastCompleted || released != releases || !blockedOnWaits()) {
            lastCompleted = seen;
            releases = released;
            blockedSince.reset();
        } else if (!blockedSince) {
            blockedSince = now;
        } else if (now - *blockedSince >= Quiet) {
            {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                if (auto* recorder = Graphics::Recorder::Active()) recorder->Sync();
                if (const auto localDevice = device.Load()) localDevice->WaitIdle();
            }
            const bool still = WaitReleases() == releases && blockedOnWaits();
            {
                std::lock_guard lock(mutex);
                rethrowFailure();
                if (completed >= target) return CaptureSettle::Completed;
                if (completed == seen && still) return CaptureSettle::Blocked;
            }
            blockedSince.reset();
        }
        if (now >= deadline) return CaptureSettle::TimedOut;
    }
}

std::vector<AwaitedWait> Driver::AwaitedWaits() {
    std::vector<AwaitedWait> result;
    for (const auto& slot : awaitedSlots) {
        const auto address = slot.address.load(std::memory_order_acquire);
        if (address == 0) continue;
        AwaitedWait wait;
        wait.address = address;
        wait.reference = slot.reference.load(std::memory_order_relaxed);
        wait.mask = slot.mask.load(std::memory_order_relaxed);
        const auto shape = slot.shape.load(std::memory_order_relaxed);
        wait.function = static_cast<std::uint32_t>(shape);
        wait.bytes = static_cast<std::uint32_t>(shape >> 32u);
        if (slot.address.load(std::memory_order_acquire) == address) result.push_back(wait);
    }
    return result;
}

namespace {

void putRegisters(std::vector<std::uint32_t>& words, const Registers& registers) {
    const auto at = words.size();
    words.push_back(0);
    registers.ForEach([&](std::uint32_t offset, std::uint32_t value) {
        words.push_back(offset);
        words.push_back(value);
        ++words[at];
    });
}

void put64(std::vector<std::uint32_t>& words, std::uint64_t value) {
    words.push_back(static_cast<std::uint32_t>(value));
    words.push_back(static_cast<std::uint32_t>(value >> 32u));
}

class QueueStateReader {
public:
    explicit QueueStateReader(std::span<const std::uint32_t> words) : words(words) {}

    std::uint32_t Take() {
        require(at < words.size(), "AGC driver: queue state record ends early");
        return words[at++];
    }

    std::uint64_t Take64() {
        const std::uint64_t low = Take();
        return low | (static_cast<std::uint64_t>(Take()) << 32u);
    }

    Registers TakeRegisters() {
        Registers registers;
        for (auto count = Take(); count != 0; --count) {
            const auto offset = Take();
            registers.insert_or_assign(offset, Take());
        }
        return registers;
    }

    bool Done() const { return at == words.size(); }

private:
    std::span<const std::uint32_t> words;
    std::size_t at = 0;
};

}

std::vector<std::uint32_t> Driver::SaveQueueStates() {
    std::lock_guard lock(mutex);
    std::vector<std::uint32_t> words{resetGraphics ? 1u : 0u, static_cast<std::uint32_t>(queues.size())};
    for (const auto& [id, state] : queues) {
        words.push_back(id);
        putRegisters(words, state.shader);
        putRegisters(words, state.context);
        putRegisters(words, state.userConfig);
        words.push_back(state.savedContext.has_value() ? 1u : 0u);
        if (state.savedContext) putRegisters(words, *state.savedContext);
        words.push_back(static_cast<std::uint32_t>(state.constantRam.size()));
        words.insert(words.end(), state.constantRam.begin(), state.constantRam.end());
        put64(words, state.indexBase);
        put64(words, state.drawIndirectBase);
        put64(words, state.dispatchIndirectBase);
        // Captures store one word, 0 for a size never written (a written zero fails at the draw either way).
        words.push_back(state.indexBufferSize.value_or(0));
        words.push_back(state.indexType);
        words.push_back(state.instanceCount);
        put64(words, state.predication.address);
        words.push_back(state.predication.operation);
        words.push_back(state.predication.executeWhenSet ? 1u : 0u);
    }
    return words;
}

void Driver::LoadQueueStates(std::span<const std::uint32_t> words) {
    QueueStateReader reader(words);
    const bool reset = reader.Take() != 0;
    std::map<std::uint32_t, QueueState> loaded;
    for (auto count = reader.Take(); count != 0; --count) {
        auto& state = loaded[reader.Take()];
        state.shader = reader.TakeRegisters();
        state.context = reader.TakeRegisters();
        state.userConfig = reader.TakeRegisters();
        if (reader.Take() != 0) state.savedContext = reader.TakeRegisters();
        state.constantRam.resize(reader.Take());
        for (auto& word : state.constantRam) word = reader.Take();
        state.indexBase = reader.Take64();
        state.drawIndirectBase = reader.Take64();
        state.dispatchIndirectBase = reader.Take64();
        if (const auto size = reader.Take(); size != 0) state.indexBufferSize = size;
        else state.indexBufferSize.reset();
        state.indexType = reader.Take();
        state.instanceCount = reader.Take();
        state.predication.address = reader.Take64();
        state.predication.operation = reader.Take();
        state.predication.executeWhenSet = reader.Take() != 0;
    }
    require(reader.Done(), "AGC driver: queue state record has trailing words");
    std::lock_guard lock(mutex);
    require(accepted == completed, "AGC driver: queue states can be loaded only while no submission is pending");
    queues = std::move(loaded);
    resetGraphics = reset;
}

bool Driver::SettleForCapture(std::chrono::milliseconds timeout) {
    if (!WaitIdleFor(timeout)) return false;
    const auto settle = [&] {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (auto* recorder = Graphics::Recorder::Active()) recorder->Sync();
        if (const auto localDevice = device.Load()) localDevice->WaitIdle();
    };
    settle();
    static_cast<void>(Graphics::FlushDepthPlanes(0, std::numeric_limits<std::size_t>::max(), Graphics::DepthPlaneScope::Whole));
    Graphics::StorageTexture::FlushAllPending("frame capture");
    {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (const auto localDevice = device.Load()) localDevice->PublishShadows();
    }
    settle();
    return !Graphics::StorageTexture::AnyPending();
}

bool Driver::Quiescent() {
    return runningWorkers.load(std::memory_order_acquire) == 0 && !DrawPipeline::Queue0().Busy() && !completionsPending();
}

bool Driver::SubmitOpenForReplay(bool report) {
    const auto workers = runningWorkers.load(std::memory_order_acquire);
    const bool queue0Busy = DrawPipeline::Queue0().Busy();
    const auto labels = Graphics::Recorder::PendingCompletionLabels();
    const auto writeBacks = Graphics::Recorder::PendingWriteBackCompletions();
    if (workers != 0 || queue0Busy || (labels == 0 && writeBacks == 0)) return false;
    {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (const auto localDevice = device.Load()) localDevice->SubmitRecorded(false);
    }
    {
        std::lock_guard lock(mutex);
    }
    changed.notify_all();
    if (report) std::fprintf(stderr, "[frame-replay] driver not quiescent while parked (running workers %u, queue 0 busy %d, completion labels %llu, write-back completions %llu): open batch submitted\n", workers, queue0Busy ? 1 : 0,
                             static_cast<unsigned long long>(labels), static_cast<unsigned long long>(writeBacks));
    return true;
}

std::pair<std::uint64_t, std::uint64_t> Driver::SerialProgress() {
    std::lock_guard lock(mutex);
    return {accepted, completed};
}

void Driver::CheckFailure() {
    if (failed.load(std::memory_order_acquire)) {
        std::lock_guard lock(mutex);
        rethrowFailure();
    }
    checkStopping();
}

void Driver::ReportFailure(std::exception_ptr error) {
    require(error != nullptr, "null asynchronous failure");
    bool first = false;
    {
        std::lock_guard lock(mutex);
        first = !failure;
        if (first) failure = error;
        failed.store(true, std::memory_order_release);
        for (const auto& [handle, output] : outputs) output->Fail(failure);
        for (auto& [queue, worker] : workers) {
            for (const auto& item : worker.pending) {
                for (const auto& [offset, flip] : item.flips) flip->Fail(failure);
            }
            worker.pending.clear();
            worker.unfinishedWrites.clear();
            worker.queued.store(0, std::memory_order_release);
        }
        if (first) PrintFailure(error);
    }
    changed.notify_all();
}

void Driver::rethrowFailure() const {
    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
}

void Driver::checkStopping() const {
    if (stopping.load(std::memory_order_acquire) || shutdownToken.stop_requested()) throw ProcessShutdown{};
}

bool& Driver::onWorkerThread() {
    static thread_local bool worker = false;
    return worker;
}

}

namespace AgcDriver {

void FlushAsyncLog_nid_no_patch() {
    AsyncLog::Get().Flush();
}

std::unique_lock<std::mutex> HoldAsyncLog() {
    return AsyncLog::Get().Hold();
}

}

extern "C" void* AgcDriverStaticTable_nid_postfix() {
    static_cast<void>(AgcDriver::DriverDetail::Driver::Get());
    return reinterpret_cast<void*>(AgcDriver::DriverDetail::FsTableAddress);
}
