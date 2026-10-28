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
    return counts.admitted + counts.evicted + counts.demoted + counts.unusedDemoted + counts.dropped;
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
    });
    active.store(true, std::memory_order_release);
}

void ReleaseResidency(VkDevice device) {
    auto& session = ResidencyState();
    const std::lock_guard lock(session.mutex);
    if (session.device != device) return;
    active.store(false, std::memory_order_release);
    NewPath::DeviceTable().SetResidentPieces(nullptr);
    session.driver.reset();
    session.guest.reset();
    session.device = VK_NULL_HANDLE;
    session.context.reset();
    session.patched.clear();
    session.pendingRefresh.clear();
    ensured.store(VK_NULL_HANDLE, std::memory_order_release);
}

bool ResidencyActive() {
    return active.load(std::memory_order_acquire);
}

void NoteStagedRegions(std::span<const std::pair<std::uint64_t, std::uint64_t>> reads, std::span<const std::pair<std::uint64_t, std::uint64_t>> writes) {
    if (!active.load(std::memory_order_acquire) || (reads.empty() && writes.empty())) return;
    auto& session = ResidencyState();
    const std::lock_guard lock(session.mutex);
    if (session.driver == nullptr || !session.context) return;
    auto& manager = session.driver->Manager();
    for (const auto& [address, bytes] : session.pendingRefresh) manager.RefreshImportWritten(address, bytes);
    session.pendingRefresh.assign(writes.begin(), writes.end());
    for (const auto& [address, bytes] : reads) manager.Use(address, bytes, ResidencyAccess::Read);
    session.stagedReads += reads.size();
    session.stagedWrites += writes.size();
    const auto mark = ChangeMark(manager.Counts());
    if (mark == session.changeMark) return;
    session.changeMark = mark;
    std::vector<std::uint64_t> now;
    for (const auto& [page, generation] : manager.ResidentGenerations()) {
        static_cast<void>(generation);
        now.push_back(page);
    }
    std::sort(now.begin(), now.end());
    std::vector<std::uint64_t> changed;
    std::set_symmetric_difference(now.begin(), now.end(), session.patched.begin(), session.patched.end(), std::back_inserter(changed));
    if (changed.empty()) return;
    if (NewPath::DeviceTable().PatchPages(*session.context, changed) == nullptr) {
        ++session.unpatched;
        return;
    }
    session.patched = std::move(now);
    ++session.patches;
    session.patchedPages += changed.size();
}

std::string ResidencyReportLine() {
    auto& session = ResidencyState();
    const std::lock_guard lock(session.mutex);
    if (session.driver == nullptr) return {};
    const auto counts = session.driver->Manager().Counts();
    char line[1536];
    int at = std::snprintf(line, sizeof(line), "[residency] staged uses %llu (read), written ranges refreshed %llu; table patches %llu (%llu pages, %llu without a table); resident %.1f MiB (%u pages, %u chunks), retired 0.0 MiB, private %.0f MiB; admitted %llu, evicted %llu, demoted %llu, dropped %llu, inadmissible %llu (unwatched imports %llu), deferred %llu; faults fast %llu served %llu refused %llu; written back %.1f MiB, copied in %.1f MiB; lease flushes 0; refusals:",
        static_cast<unsigned long long>(session.stagedReads), static_cast<unsigned long long>(session.stagedWrites),
        static_cast<unsigned long long>(session.patches), static_cast<unsigned long long>(session.patchedPages), static_cast<unsigned long long>(session.unpatched),
        static_cast<double>(std::uint64_t{counts.resident} << ResidencyPageShift) / (1024.0 * 1024.0), counts.resident, counts.chunks, PrivateMiB(),
        static_cast<unsigned long long>(counts.admitted), static_cast<unsigned long long>(counts.evicted), static_cast<unsigned long long>(counts.demoted + counts.unusedDemoted), static_cast<unsigned long long>(counts.dropped), static_cast<unsigned long long>(counts.inadmissible), static_cast<unsigned long long>(session.guest->UnwatchedImports()), static_cast<unsigned long long>(counts.deferred),
        static_cast<unsigned long long>(session.driver->FastFaults()), static_cast<unsigned long long>(session.driver->ServedFaults()), static_cast<unsigned long long>(session.driver->RefusedFaults()),
        static_cast<double>(counts.writtenBackBytes) / (1024.0 * 1024.0), static_cast<double>(counts.copiedInBytes) / (1024.0 * 1024.0));
    for (std::size_t reason = 1; reason < ResidencyRefusals && at > 0 && static_cast<std::size_t>(at) < sizeof(line); ++reason) {
        at += std::snprintf(line + at, sizeof(line) - static_cast<std::size_t>(at), " %s %llu", RefusalNames[reason], static_cast<unsigned long long>(counts.refusals[reason]));
    }
    return line;
}

}
