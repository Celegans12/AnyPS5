#include "prx/libSceAgcDriver/Graphics/include/ResidencySession.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyConfig.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyGuest.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DevicePageTable.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <memory>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>
#include <mutex>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace AgcDriver::Graphics {

namespace {

struct Session {
    std::mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    std::optional<Context> context;
    std::vector<std::uint64_t> patched;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> pendingRefresh;
    std::uint64_t changeMark = 0;
    std::uint64_t stagedReads = 0;
    std::uint64_t onePage = 0;
    std::uint64_t onePageBytes = 0;
    std::uint64_t multiPage = 0;
    std::uint64_t multiPageBytes = 0;
    std::uint64_t overlayMulti = 0;
    std::uint64_t slotsBound = 0;
    std::uint64_t slotBoundBytes = 0;
    std::uint64_t slotsRefused = 0;
    std::uint64_t writtenSlots = 0;
    std::uint64_t writtenSlotBytes = 0;
    std::uint64_t writtenRefused = 0;
    std::atomic<std::uint64_t> reentered{0};
    std::uint64_t overlayMultiBytes = 0;
    std::uint64_t stagedWrites = 0;
    std::uint64_t patches = 0;
    std::uint64_t patchedPages = 0;
    std::uint64_t unpatched = 0;
    std::unique_ptr<ArenaResidencyGuest> guest;
    std::unique_ptr<ResidencyDriver> driver;
};

std::atomic<VkDevice> ensured{VK_NULL_HANDLE};
std::atomic<bool> active{false};

Session& ResidencyState() {
    static auto* session = new Session;
    return *session;
}

double PrivateMiB() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters))) return static_cast<double>(counters.PrivateUsage) / (1024.0 * 1024.0);
#endif
    return 0.0;
}

std::uint64_t ChangeMark(const ResidencyCounts& counts) {
    return counts.admitted + counts.evicted + counts.demoted + counts.unusedDemoted + counts.dropped + counts.behindFlips;
}

// The device table's residency mapping follows the manager's current pages. The table itself
// keeps the patched set, so any table use after a change (DevicePageTable::Stale) patches it too,
// whichever thread made the change (a nested copy-back note included).
void PatchTable(Session& session) {
    const auto changed = NewPath::DeviceTable().PatchResident(*session.context);
    if (!changed) {
        ++session.unpatched;
        return;
    }
    if (*changed == 0) return;
    ++session.patches;
    session.patchedPages += *changed;
}

std::shared_ptr<HostImport> SlotImport(Session& session, std::uint64_t page, const MirrorSlot& slot, const std::shared_ptr<MirrorChunk>& chunk) {
    auto import = std::make_shared<HostImport>();
    import->base = page << ResidencyPageShift;
    import->bytes = ResidencyPageBytes;
    import->buffer = chunk->buffer;
    import->memory = chunk->memory;
    import->lead = std::uint64_t{slot.slot} << ResidencyPageShift;
    import->address = chunk->address + import->lead;
    import->chunk = chunk;
    import->serial = (std::uint64_t{1} << 62) | ChangeMark(session.driver->Manager().Counts());
    return import;
}

constexpr const char* RefusalNames[ResidencyRefusals] = {"none", "unsettled_import", "private_host_write", "private_pinned", "resident_view", "shared_pinned", "alias_host_write", "unavailable"};

}

bool ResidencyEnsured(VkDevice device) {
    return ensured.load(std::memory_order_acquire) == device;
}

void EnsureResidency(const Context& context) {
    if (!CurrentResidencyConfig().residency) return;
    auto& session = ResidencyState();
    const std::lock_guard lock(session.mutex);
    if (session.driver != nullptr && session.device == context.device) return;
    auto* recorder = Recorder::Active();
    if (recorder == nullptr) return;
    session.driver.reset();
    session.guest = std::make_unique<ArenaResidencyGuest>(context);
    session.driver = ResidencyDriver::CreateIfEnabled(context, *recorder, *session.guest);
    session.device = context.device;
    session.context = context;
    session.patched.clear();
    session.changeMark = 0;
    ensured.store(context.device, std::memory_order_release);
    if (session.driver == nullptr) return;
    session.driver->EnableFaults();
    NewPath::DeviceTable().SetResidentPieces([](std::uint64_t begin, std::uint64_t end) {
        std::vector<ShaderRecompiler::BdaAbi::Range> pieces;
        auto* driver = ResidencyDriver::Active();
        if (driver == nullptr || end <= begin) return pieces;
        for (const auto& [page, mirror] : driver->Manager().MirroredIn(begin, end - begin)) {
            const auto first = std::max(begin, page << ResidencyPageShift);
            const auto last = std::min(end, (page + 1) << ResidencyPageShift);
            pieces.push_back({first, last, mirror + (first - (page << ResidencyPageShift)), ShaderRecompiler::BdaAbi::Read | ShaderRecompiler::BdaAbi::Write, ShaderRecompiler::BdaAbi::RangeMirrored});
        }
        return pieces;
    }, NewPath::DevicePageTable::ResidentState{[]() -> std::uint64_t {
        auto* driver = ResidencyDriver::Active();
        if (driver == nullptr) return 0;
        static_cast<void>(driver->Manager().SettleCurrent());
        return ChangeMark(driver->Manager().Counts());
    }, []() {
        auto* driver = ResidencyDriver::Active();
        return driver != nullptr ? driver->Manager().CurrentPages() : std::vector<std::uint64_t>{};
    }});
    active.store(true, std::memory_order_release);
}

void ReleaseResidency(VkDevice device) {
    auto& session = ResidencyState();
    const std::lock_guard lock(session.mutex);
    if (session.device != device) return;
    active.store(false, std::memory_order_release);
    NewPath::DeviceTable().SetResidentPieces(nullptr, {});
    session.driver.reset();
    session.guest.reset();
    session.device = VK_NULL_HANDLE;
    session.context.reset();
    session.patched.clear();
    session.pendingRefresh.clear();
    ensured.store(VK_NULL_HANDLE, std::memory_order_release);
}

void NoteStagedOverlays(std::span<const std::pair<std::uint64_t, std::uint64_t>> overlays) {
    if (!active.load(std::memory_order_acquire) || overlays.empty() || ResidencyDriver::Recording()) return;
    auto& session = ResidencyState();
    const std::lock_guard lock(session.mutex);
    for (const auto& [address, bytes] : overlays) {
        if ((address >> ResidencyPageShift) == ((address + bytes - 1) >> ResidencyPageShift)) continue;
        ++session.overlayMulti;
        session.overlayMultiBytes += bytes;
    }
}

std::shared_ptr<HostImport> BindStagedSlot(std::uint64_t address, std::uint64_t bytes, bool written) {
    if (!active.load(std::memory_order_acquire) || !CurrentResidencyConfig().slotBinding || ResidencyDriver::Recording() || bytes == 0 || (address >> ResidencyPageShift) != ((address + bytes - 1) >> ResidencyPageShift)) return nullptr;
    auto& session = ResidencyState();
    const std::lock_guard lock(session.mutex);
    if (session.driver == nullptr) return nullptr;
    auto& manager = session.driver->Manager();
    manager.Use(address, bytes, written ? ResidencyAccess::Write : ResidencyAccess::Read);
    if (written) manager.NoteGpuWritten(address, bytes);
    const auto page = address >> ResidencyPageShift;
    const auto slot = manager.Slot(page);
    if (!slot) {
        ++session.slotsRefused;
        return nullptr;
    }
    auto chunk = session.driver->Mirrors().Chunk(slot->chunk);
    if (chunk == nullptr) {
        ++session.slotsRefused;
        return nullptr;
    }
    manager.Pin(address, bytes, written ? ResidencyAccess::Write : ResidencyAccess::Read, session.driver->Current());
    auto import = SlotImport(session, page, *slot, chunk);
    ++session.slotsBound;
    session.slotBoundBytes += bytes;
    if (session.context) PatchTable(session);
    return import;
}

std::shared_ptr<HostImport> BindWrittenSlot(std::uint64_t address, std::uint64_t bytes) {
    if (!active.load(std::memory_order_acquire) || !CurrentResidencyConfig().writtenSlots || ResidencyDriver::Recording() || bytes == 0 || (address >> ResidencyPageShift) != ((address + bytes - 1) >> ResidencyPageShift)) return nullptr;
    auto& session = ResidencyState();
    const std::lock_guard lock(session.mutex);
    if (session.driver == nullptr || !session.context) return nullptr;
    auto& manager = session.driver->Manager();
    manager.Use(address, bytes, ResidencyAccess::Read);
    const auto page = address >> ResidencyPageShift;
    const auto slot = manager.Slot(page);
    auto chunk = slot ? session.driver->Mirrors().Chunk(slot->chunk) : nullptr;
    if (!slot || chunk == nullptr || manager.Dirty(page) != 0) {
        ++session.writtenRefused;
        return nullptr;
    }
    if (!manager.HoldWritten(address, bytes)) {
        ++session.writtenRefused;
        return nullptr;
    }
    manager.Pin(address, bytes, ResidencyAccess::Write, session.driver->Current());
    auto import = SlotImport(session, page, *slot, chunk);
    ++session.writtenSlots;
    session.writtenSlotBytes += bytes;
    PatchTable(session);
    return import;
}

void ReleaseWrittenSlot(std::uint64_t address, std::uint64_t bytes) {
    if (auto* driver = ResidencyDriver::Active(); driver != nullptr) driver->Manager().ReleaseWritten(address, bytes);
}

std::uint64_t ResidencyBindingSerial() {
    auto* driver = ResidencyDriver::Active();
    if (driver == nullptr) return 0;
    return (std::uint64_t{1} << 62) | ChangeMark(driver->Manager().Counts());
}

bool ResidencyActive() {
    return active.load(std::memory_order_acquire);
}

void NoteStagedRegions(std::span<const std::pair<std::uint64_t, std::uint64_t>> givenReads, std::span<const std::pair<std::uint64_t, std::uint64_t>> givenWrites) {
    if (!active.load(std::memory_order_acquire) || (givenReads.empty() && givenWrites.empty())) return;
    thread_local std::vector<std::pair<std::uint64_t, std::uint64_t>> deferredReads;
    thread_local std::vector<std::pair<std::uint64_t, std::uint64_t>> deferredWrites;
    if (ResidencyDriver::Recording()) {
        if (auto* driver = ResidencyDriver::Active(); driver != nullptr) {
            for (const auto& [address, bytes] : givenWrites) driver->Manager().NoteOldPathWrite(address, bytes, false);
        }
        deferredReads.insert(deferredReads.end(), givenReads.begin(), givenReads.end());
        deferredWrites.insert(deferredWrites.end(), givenWrites.begin(), givenWrites.end());
        ++ResidencyState().reentered;
        return;
    }
    auto reads = std::exchange(deferredReads, {});
    auto writes = std::exchange(deferredWrites, {});
    reads.insert(reads.end(), givenReads.begin(), givenReads.end());
    writes.insert(writes.end(), givenWrites.begin(), givenWrites.end());
    auto& session = ResidencyState();
    const std::lock_guard lock(session.mutex);
    if (session.driver == nullptr || !session.context) return;
    auto& manager = session.driver->Manager();
    for (const auto& [address, bytes] : writes) manager.NoteOldPathWrite(address, bytes, true);
    for (const auto& [address, bytes] : session.pendingRefresh) manager.RefreshImportWritten(address, bytes);
    if (CurrentResidencyConfig().eagerCopyIn) session.pendingRefresh.assign(writes.begin(), writes.end());
    else session.pendingRefresh.clear();
    for (const auto& [address, bytes] : reads) {
        if ((address >> ResidencyPageShift) == ((address + bytes - 1) >> ResidencyPageShift)) manager.Use(address, bytes, ResidencyAccess::Read);
    }
    session.stagedReads += reads.size();
    for (const auto& [address, bytes] : reads) {
        const bool one = (address >> ResidencyPageShift) == ((address + bytes - 1) >> ResidencyPageShift);
        (one ? session.onePage : session.multiPage) += 1;
        (one ? session.onePageBytes : session.multiPageBytes) += bytes;
    }
    session.stagedWrites += writes.size();
    PatchTable(session);
}

std::string ResidencyReportLine() {
    auto& session = ResidencyState();
    const std::lock_guard lock(session.mutex);
    if (session.driver == nullptr) return {};
    const auto counts = session.driver->Manager().Counts();
    char line[2048];
    int at = std::snprintf(line, sizeof(line), "[residency] slots bound %llu (%.1f MiB), refused %llu; written slots %llu (%.1f MiB), refused %llu; old-path write flips %llu (dirty written back %llu, kept %llu), settle checks %llu (flips %llu), held slot ranges refused %llu (catch-up skipped %.1f KiB), table reads of current mirrors %llu; re-entered copy-back notes %llu; staged uses %llu (read; one page %llu, %.1f MiB; spanning %llu, %.1f MiB, of them address-based overlays %llu, %.1f MiB), written ranges refreshed %llu; table patches %llu (%llu pages, %llu without a table); resident %.1f MiB (%u pages, %u chunks), retired 0.0 MiB, private %.0f MiB; admitted %llu, evicted %llu, demoted %llu, dropped %llu, inadmissible %llu (unwatched imports %llu), deferred %llu; faults fast %llu served %llu refused %llu; driver publishes %llu, write-back stamps %llu; written back %.1f MiB, copied in %.1f MiB, copy-in deferred %.1f MiB (%u pages behind the import, %llu caught up); lease flushes 0; refusals:",
        static_cast<unsigned long long>(session.slotsBound), static_cast<double>(session.slotBoundBytes) / (1024.0 * 1024.0), static_cast<unsigned long long>(session.slotsRefused), static_cast<unsigned long long>(session.writtenSlots), static_cast<double>(session.writtenSlotBytes) / (1024.0 * 1024.0), static_cast<unsigned long long>(session.writtenRefused), static_cast<unsigned long long>(counts.oldPathFlips), static_cast<unsigned long long>(counts.oldPathDirtyWrittenBack), static_cast<unsigned long long>(counts.oldPathDirtyKept), static_cast<unsigned long long>(counts.settleChecks), static_cast<unsigned long long>(counts.settleFlips), static_cast<unsigned long long>(counts.heldRefusals), static_cast<double>(counts.heldSkippedBytes) / 1024.0, static_cast<unsigned long long>(NewPath::DeviceTable().Totals().mirroredReads), static_cast<unsigned long long>(session.reentered.load()),
        static_cast<unsigned long long>(session.stagedReads), static_cast<unsigned long long>(session.onePage), static_cast<double>(session.onePageBytes) / (1024.0 * 1024.0), static_cast<unsigned long long>(session.multiPage), static_cast<double>(session.multiPageBytes) / (1024.0 * 1024.0), static_cast<unsigned long long>(session.overlayMulti), static_cast<double>(session.overlayMultiBytes) / (1024.0 * 1024.0), static_cast<unsigned long long>(session.stagedWrites),
        static_cast<unsigned long long>(session.patches), static_cast<unsigned long long>(session.patchedPages), static_cast<unsigned long long>(session.unpatched),
        static_cast<double>(std::uint64_t{counts.resident} << ResidencyPageShift) / (1024.0 * 1024.0), counts.resident, counts.chunks, PrivateMiB(),
        static_cast<unsigned long long>(counts.admitted), static_cast<unsigned long long>(counts.evicted), static_cast<unsigned long long>(counts.demoted + counts.unusedDemoted), static_cast<unsigned long long>(counts.dropped), static_cast<unsigned long long>(counts.inadmissible), static_cast<unsigned long long>(session.guest->UnwatchedImports()), static_cast<unsigned long long>(counts.deferred),
        static_cast<unsigned long long>(session.driver->FastFaults()), static_cast<unsigned long long>(session.driver->ServedFaults()), static_cast<unsigned long long>(session.driver->RefusedFaults()), static_cast<unsigned long long>(session.driver->DriverPublishes()), static_cast<unsigned long long>(session.driver->WriteBackStamps()),
        static_cast<double>(counts.writtenBackBytes) / (1024.0 * 1024.0), static_cast<double>(counts.copiedInBytes) / (1024.0 * 1024.0), static_cast<double>(counts.copyInDeferredBytes) / (1024.0 * 1024.0), counts.behind, static_cast<unsigned long long>(counts.caughtUp));
    for (std::size_t reason = 1; reason < ResidencyRefusals && at > 0 && static_cast<std::size_t>(at) < sizeof(line); ++reason) {
        at += std::snprintf(line + at, sizeof(line) - static_cast<std::size_t>(at), " %s %llu", RefusalNames[reason], static_cast<unsigned long long>(counts.refusals[reason]));
    }
    return line;
}

}
