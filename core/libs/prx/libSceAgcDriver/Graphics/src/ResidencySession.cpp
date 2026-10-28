#include "prx/libSceAgcDriver/Graphics/include/ResidencySession.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyConfig.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyGuest.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DevicePageTable.hpp"
#include <atomic>
#include <cstdio>
#include <memory>
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
    std::unique_ptr<ArenaResidencyGuest> guest;
    std::unique_ptr<ResidencyDriver> driver;
};

std::atomic<VkDevice> ensured{VK_NULL_HANDLE};

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
    ensured.store(context.device, std::memory_order_release);
    if (session.driver == nullptr) return;
    session.driver->EnableFaults();
    NewPath::DeviceTable().SetResidentPieces([](std::uint64_t begin, std::uint64_t end) {
        std::vector<ShaderRecompiler::BdaAbi::Range> pieces;
        auto* driver = ResidencyDriver::Active();
        if (driver == nullptr || end <= begin) return pieces;
        for (auto page = begin >> ResidencyPageShift; page <= (end - 1) >> ResidencyPageShift; ++page) {
            const auto mirror = driver->Manager().Translate(page);
            if (!mirror) continue;
            const auto first = std::max(begin, page << ResidencyPageShift);
            const auto last = std::min(end, (page + 1) << ResidencyPageShift);
            pieces.push_back({first, last, *mirror + (first - (page << ResidencyPageShift)), ShaderRecompiler::BdaAbi::Read | ShaderRecompiler::BdaAbi::Write, 0});
        }
        return pieces;
    });
}

void ReleaseResidency(VkDevice device) {
    auto& session = ResidencyState();
    const std::lock_guard lock(session.mutex);
    if (session.device != device) return;
    NewPath::DeviceTable().SetResidentPieces(nullptr);
    session.driver.reset();
    session.guest.reset();
    session.device = VK_NULL_HANDLE;
    ensured.store(VK_NULL_HANDLE, std::memory_order_release);
}

std::string ResidencyReportLine() {
    auto& session = ResidencyState();
    const std::lock_guard lock(session.mutex);
    if (session.driver == nullptr) return {};
    const auto counts = session.driver->Manager().Counts();
    char line[1024];
    int at = std::snprintf(line, sizeof(line), "[residency] resident %.1f MiB (%u pages, %u chunks), retired 0.0 MiB, private %.0f MiB; admitted %llu, inadmissible %llu (unwatched imports %llu), deferred %llu; faults fast %llu served %llu refused %llu; written back %.1f MiB, copied in %.1f MiB; lease flushes 0; refusals:",
        static_cast<double>(std::uint64_t{counts.resident} << ResidencyPageShift) / (1024.0 * 1024.0), counts.resident, counts.chunks, PrivateMiB(),
        static_cast<unsigned long long>(counts.admitted), static_cast<unsigned long long>(counts.inadmissible), static_cast<unsigned long long>(session.guest->UnwatchedImports()), static_cast<unsigned long long>(counts.deferred),
        static_cast<unsigned long long>(session.driver->FastFaults()), static_cast<unsigned long long>(session.driver->ServedFaults()), static_cast<unsigned long long>(session.driver->RefusedFaults()),
        static_cast<double>(counts.writtenBackBytes) / (1024.0 * 1024.0), static_cast<double>(counts.copiedInBytes) / (1024.0 * 1024.0));
    for (std::size_t reason = 1; reason < ResidencyRefusals && at > 0 && static_cast<std::size_t>(at) < sizeof(line); ++reason) {
        at += std::snprintf(line + at, sizeof(line) - static_cast<std::size_t>(at), " %s %llu", RefusalNames[reason], static_cast<unsigned long long>(counts.refusals[reason]));
    }
    return line;
}

}
