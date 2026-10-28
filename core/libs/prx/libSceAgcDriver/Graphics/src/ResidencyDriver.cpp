#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyConfig.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics {

namespace {

constexpr std::uint64_t DefaultResidentBytes = std::uint64_t{1024} << 20u;

std::atomic<ResidencyDriver*> activeDriver{nullptr};

}

std::optional<ResidencyGuestImport> RefusingResidencyGuest::ImportOf(std::uint64_t) {
    return std::nullopt;
}

std::vector<ResidencyRange> RefusingResidencyGuest::Aliases(std::uint64_t, std::uint64_t) {
    return {};
}

std::uint64_t RefusingResidencyGuest::Generation() {
    return 0;
}

void RefusingResidencyGuest::ChangedPages(std::uint64_t, std::span<const std::uint64_t>, std::span<std::uint8_t> changed) {
    std::fill(changed.begin(), changed.end(), std::uint8_t{1});
}

ResidencyRefusal RefusingResidencyGuest::Guard(std::uint64_t) {
    return ResidencyRefusal::Unavailable;
}

void RefusingResidencyGuest::Unguard(std::span<const std::uint64_t>) {}

ResidencyRefusal RefusingResidencyGuest::GuardAccess(std::uint64_t, ResidencyMask) {
    return ResidencyRefusal::Unavailable;
}

void RefusingResidencyGuest::UnguardAccess(std::uint64_t) {}

std::uint64_t RefusingResidencyGuest::GuestBytes() {
    return 0;
}

ResidencyDriver::ResidencyDriver(const Context& context, Recorder& recorder, IResidencyGuestMemory& guest, const ResidencyPolicy& policy, std::unique_ptr<IResidencyLanes> lanes)
    : context(context), recorder(recorder), guest(guest), lanes(lanes != nullptr ? std::move(lanes) : RecorderLanes(recorder)), tokens(*this->lanes),
      budgetBytes(policy.budgetBytes), mirrors(context, policy.chunkSlots), manager(*this, policy) {
    ResidencyDriver* expected = nullptr;
    if (!activeDriver.compare_exchange_strong(expected, this)) throw std::logic_error("residency: a second residency driver while one is active");
}

struct ResidencyDriver::FaultServer {
    struct Request {
        std::uint64_t address = 0;
        bool done = false;
        std::exception_ptr failure;
    };

    explicit FaultServer(ResidencyDriver& driver) : driver(driver), thread([this] { run(); }) {}

    ~FaultServer() {
        {
            const std::lock_guard lock(mutex);
            stopping = true;
        }
        wake.notify_all();
        thread.join();
    }

    void Serve(std::uint64_t address) {
        Request request{address};
        std::unique_lock lock(mutex);
        queue.push_back(&request);
        wake.notify_all();
        finished.wait(lock, [&] { return request.done; });
        if (request.failure) std::rethrow_exception(request.failure);
    }

    void run() {
        std::unique_lock lock(mutex);
        for (;;) {
            wake.wait(lock, [&] { return stopping || !queue.empty(); });
            if (queue.empty()) return;
            auto* request = queue.front();
            queue.pop_front();
            lock.unlock();
            try {
                driver.manager.FlushForCpuRead(request->address & ~(ResidencyPageBytes - 1), ResidencyPageBytes);
            } catch (...) {
                request->failure = std::current_exception();
            }
            lock.lock();
            request->done = true;
            finished.notify_all();
        }
    }

    ResidencyDriver& driver;
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable finished;
    std::deque<Request*> queue;
    bool stopping = false;
    std::thread thread;
};

ResidencyDriver::~ResidencyDriver() {
    DisableFaults();
    ResidencyDriver* expected = this;
    activeDriver.compare_exchange_strong(expected, nullptr);
}

void ResidencyDriver::EnableFaults() {
    if (faultServer != nullptr) return;
    faultServer = std::make_unique<FaultServer>(*this);
    GuestArena::GuestArenaSetAccessGuardObserver_nid_no_patch([](std::uintptr_t address) {
        if (auto* driver = ResidencyDriver::Active()) driver->AccessFault(address);
    });
}

void ResidencyDriver::DisableFaults() {
    if (faultServer == nullptr) return;
    GuestArena::GuestArenaSetAccessGuardObserver_nid_no_patch(nullptr);
    faultServer.reset();
}

void ResidencyDriver::AccessFault(std::uint64_t address) {
    if (GuestMemory::GpuMutex().HeldByThisThread() || faultServer == nullptr) {
        refusedFaults.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const auto page = address & ~(ResidencyPageBytes - 1);
    try {
        if (manager.FlushForCpuReadSettled(page, ResidencyPageBytes)) {
            fastFaults.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        faultServer->Serve(page);
        servedFaults.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception& error) {
        refusedFaults.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr, "[residency] fault at 0x%llx not served: %s\n", static_cast<unsigned long long>(address), error.what());
    }
}

ResidencyDriver* ResidencyDriver::Active() {
    return activeDriver.load();
}

std::unique_ptr<ResidencyDriver> ResidencyDriver::CreateIfEnabled(const Context& context, Recorder& recorder, IResidencyGuestMemory& guest) {
    const auto& config = CurrentResidencyConfig();
    if (!config.residency) return nullptr;
    ResidencyPolicy policy;
    policy.budgetBytes = config.budgetForced ? config.residentBytes : DefaultResidentBytes;
    return std::make_unique<ResidencyDriver>(context, recorder, guest, policy);
}

void ResidencyDriver::OrderingPoint() {
    if (!manager.Advance()) return;
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    manager.Apply();
}

bool ResidencyDriver::Reached(const ResidencyToken& token) {
    return tokens.Reached(token);
}

void ResidencyDriver::Wait(const ResidencyToken& token) {
    tokens.Finish(token);
}

bool ResidencyDriver::WaitUnlocked(const ResidencyToken& token) {
    return tokens.WaitUnlocked(token);
}

bool ResidencyDriver::Signaled(const ResidencyToken& token) {
    return tokens.Reached(token);
}

ResidencyToken ResidencyDriver::Newest() {
    return tokens.All();
}

ResidencyToken ResidencyDriver::Current() {
    return tokens.Caller();
}

ResidencyImport ResidencyDriver::Import(std::uint64_t page) {
    const auto import = guest.ImportOf(page);
    return import ? import->attributes : ResidencyImport{};
}

std::vector<ResidencyRange> ResidencyDriver::Aliases(std::uint64_t address, std::uint64_t bytes) {
    return guest.Aliases(address, bytes);
}

std::uint64_t ResidencyDriver::Generation() {
    return guest.Generation();
}

void ResidencyDriver::ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) {
    guest.ChangedPages(firstPage, generations, changed);
}

std::uint64_t ResidencyDriver::AllocateChunk(std::uint32_t chunk, std::uint32_t slots) {
    Require(slots == 0 || slots <= mirrors.Slots(), "residency: a chunk of " + std::to_string(slots) + " slots past the mirrors' chunk size");
    return mirrors.Allocate(chunk);
}

void ResidencyDriver::ReleaseChunk(std::uint32_t chunk) {
    mirrors.Release(chunk);
}

ResidencyToken ResidencyDriver::CopyIn(std::span<const ResidencyCopy> copies) {
    return record(copies, true);
}

ResidencyToken ResidencyDriver::WriteBack(std::span<const ResidencyCopy> copies) {
    return record(copies, false);
}

ResidencyToken ResidencyDriver::record(std::span<const ResidencyCopy> copies, bool intoMirror) {
    if (copies.empty()) return tokens.Caller();
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    const ResidencyImportLocator locate = [&](std::uint64_t page) {
        const auto import = guest.ImportOf(page);
        return import ? ResidencyImportLocation{import->buffer, import->offset} : ResidencyImportLocation{};
    };
    const auto commands = recorder.Commands();
    if (intoMirror) mirrors.RecordCopyIn(commands, copies, locate);
    else mirrors.RecordWriteBack(commands, copies, locate);
    recorder.MarkCovered(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    std::vector<std::uint32_t> chunks;
    for (const auto& copy : copies) {
        if (std::find(chunks.begin(), chunks.end(), copy.mirror.chunk) == chunks.end()) chunks.push_back(copy.mirror.chunk);
    }
    for (const auto chunk : chunks) recorder.Keep(mirrors.Chunk(chunk));
    return tokens.Caller();
}

ResidencyRefusal ResidencyDriver::Guard(std::uint64_t page) {
    return guest.Guard(page);
}

void ResidencyDriver::Unguard(std::span<const std::uint64_t> pages) {
    guest.Unguard(pages);
}

ResidencyRefusal ResidencyDriver::GuardAccess(std::uint64_t page, ResidencyMask mask) {
    return guest.GuardAccess(page, mask);
}

void ResidencyDriver::UnguardAccess(std::uint64_t page) {
    guest.UnguardAccess(page);
}

std::uint64_t ResidencyDriver::Budget() {
    return budgetBytes;
}

std::uint64_t ResidencyDriver::GuestBytes() {
    return guest.GuestBytes();
}

}
