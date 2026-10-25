#include "prx/libSceAgcDriver/Execution/include/OwnershipProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libc/include/GuestArena.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace AgcDriver::OwnershipProbe {

namespace {

constexpr std::uint64_t Unit = 0x10000;
constexpr std::uint8_t TagAddressHeap = 1;

std::uint64_t Mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

bool Inside(const std::map<std::uint64_t, std::uint64_t>& ranges, std::uint64_t begin, std::uint64_t end) {
    auto it = ranges.upper_bound(begin);
    if (it != ranges.begin() && std::prev(it)->second > begin) return true;
    return it != ranges.end() && it->first < end;
}

void Insert(std::map<std::uint64_t, std::uint64_t>& ranges, std::uint64_t begin, std::uint64_t end) {
    auto it = ranges.upper_bound(begin);
    if (it != ranges.begin() && std::prev(it)->second >= begin) {
        --it;
        begin = it->first;
        end = std::max(end, it->second);
        it = ranges.erase(it);
    }
    while (it != ranges.end() && it->first <= end) {
        end = std::max(end, it->second);
        it = ranges.erase(it);
    }
    ranges.emplace(begin, end);
}

std::string Hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

std::string SiteName(std::uint64_t rip) {
#ifdef _WIN32
    HMODULE module = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(rip), &module) && module != nullptr) {
        char path[MAX_PATH] = "";
        const auto length = GetModuleFileNameA(module, path, MAX_PATH);
        std::string name(path, length);
        if (const auto slash = name.find_last_of("\\/"); slash != std::string::npos) name.erase(0, slash + 1);
        return name + "+" + Hex(rip - reinterpret_cast<std::uint64_t>(module));
    }
#endif
    return Hex(rip);
}

std::string ThreadName(std::uint32_t thread) {
#ifdef _WIN32
    using Describe = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    static const auto describe = reinterpret_cast<Describe>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription")));
    if (describe == nullptr) return {};
    const HANDLE handle = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, thread);
    if (handle == nullptr) return {};
    std::string name;
    PWSTR description = nullptr;
    if (SUCCEEDED(describe(handle, &description)) && description != nullptr) {
        char text[128] = "";
        if (WideCharToMultiByte(CP_UTF8, 0, description, -1, text, sizeof(text), nullptr, nullptr) > 0) name = text;
        LocalFree(description);
    }
    CloseHandle(handle);
    return name;
#else
    static_cast<void>(thread);
    return {};
#endif
}

// The faulting thread's work as the driver tagged it (GuestMemory::SetCurrentPacket, ReadSiteScope):
// bit 63 set, PM4 opcode in bits 0-31 (NoPacket for a thread that runs no packets, 0xffff the flip),
// queue in 32-47, read site in 48-55. TEB slots only, so it is safe in the fault handler.
std::uint64_t ProbeContext() {
    const auto packet = GuestMemory::CurrentPacket();
    return (std::uint64_t{1} << 63u) | packet.opcode | (std::uint64_t{packet.queue & 0xffffu} << 32u) | (std::uint64_t{static_cast<std::uint8_t>(GuestMemory::CurrentReadSite())} << 48u);
}

std::string ContextName(std::uint64_t context) {
    if ((context >> 63u) == 0) return "no context";
    const auto opcode = static_cast<std::uint32_t>(context);
    const auto queue = static_cast<std::uint32_t>((context >> 32u) & 0xffffu);
    const auto site = static_cast<GuestMemory::ReadSite>((context >> 48u) & 0xffu);
    const auto* siteName = static_cast<std::size_t>(site) < static_cast<std::size_t>(GuestMemory::ReadSite::Count) ? GuestMemory::ReadSiteName(site) : "?";
    char text[96];
    if (opcode == GuestMemory::NoPacket) std::snprintf(text, sizeof(text), "no packet, site %s", siteName);
    else if (opcode == 0xffffu) std::snprintf(text, sizeof(text), "flip q0x%x, site %s", queue, siteName);
    else std::snprintf(text, sizeof(text), "pm4 0x%x q0x%x, site %s", opcode, queue, siteName);
    return text;
}

std::string ModuleOf(std::uint64_t address) {
    auto name = SiteName(address);
    if (const auto plus = name.find('+'); plus != std::string::npos) name.erase(plus);
    return name;
}

// The CRT and system DLLs a touch is charged through: the caller that matters is the first frame
// outside them (memcpy/memcmp's caller).
bool SystemModule(std::uint64_t address) {
    auto name = ModuleOf(address);
    for (auto& c : name) c = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    for (const char* system : {"ucrtbase.dll", "ntdll.dll", "kernelbase.dll", "kernel32.dll", "vcruntime140.dll", "vcruntime140_1.dll", "msvcrt.dll", "libstdc++-6.dll", "libgcc_s_seh-1.dll", "libwinpthread-1.dll"}) {
        if (name == system) return true;
    }
    return false;
}

std::uint64_t Median(std::vector<std::uint64_t>& values) {
    if (values.empty()) return 0;
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

}

std::uint32_t RateFromEnvironment() {
    const char* value = std::getenv("APS5_PROFILE_OWNERSHIP");
    if (value == nullptr || *value == '\0') return 0;
    char* end = nullptr;
    const auto rate = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || rate > 1u << 20u) {
        std::fprintf(stderr, "[ownership] APS5_PROFILE_OWNERSHIP=%s is not a unit sample rate (1 = every unit, n = one in n): off\n", value);
        return 0;
    }
    return static_cast<std::uint32_t>(rate);
}

Sampler::Sampler(std::uint32_t rate) : rate(rate) {
    if (rate == 0) return;
    // Called once here so the TEB slots' indexes exist before the fault handler reads them.
    static_cast<void>(ProbeContext());
    GuestArena::GuestArenaSetProbeContextHook_nid_no_patch(&ProbeContext);
}

bool Sampler::Sampled(std::uint64_t unit) const {
    return rate != 0 && Mix(unit / Unit) % rate == 0;
}

bool Sampler::inAddressHeap(std::uint64_t unit) const {
    return Inside(heaps, unit, unit + Unit) || Inside(previousHeaps, unit, unit + Unit);
}

void Sampler::NoteGpuWrite(std::uint64_t begin, std::uint64_t end) {
    if (rate == 0 || end <= begin) return;
    std::lock_guard lock(mutex);
    for (auto unit = begin & ~(Unit - 1); unit < end; unit += Unit) {
        gpuUnits.insert(unit);
        ++unitWrites;
        if (!Sampled(unit)) continue;
        if (GuestArena::GuestArenaProbeArm_nid_no_patch(static_cast<std::uintptr_t>(unit), inAddressHeap(unit) ? TagAddressHeap : 0)) ++armed;
        else ++refused;
    }
}

void Sampler::NoteAddressHeap(std::uint64_t begin, std::uint64_t end) {
    if (rate == 0 || end <= begin) return;
    std::lock_guard lock(mutex);
    Insert(heaps, begin, end);
}

std::string Sampler::Take(double seconds) {
    std::lock_guard lock(mutex);
    GuestArena::GuestArenaProbeCounts counts{};
    std::vector<GuestArena::GuestArenaProbeEvent> events;
    std::array<GuestArena::GuestArenaProbeEvent, 4096> chunk{};
    for (bool first = true;; first = false) {
        const auto taken = GuestArena::GuestArenaProbeTake_nid_no_patch(first ? &counts : nullptr, chunk.data(), chunk.size());
        events.insert(events.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(taken));
        if (taken < chunk.size()) break;
    }
    std::uint64_t heapUnits = 0;
    for (const auto unit : gpuUnits) heapUnits += inAddressHeap(unit) ? 1 : 0;
    struct Site {
        std::uint64_t reads = 0;
        std::uint64_t writes = 0;
        std::uint32_t thread = 0;
    };
    std::unordered_map<std::uint64_t, Site> sites;
    // By caller: the first frame outside the CRT/system DLLs (the faulting RIP itself when it is
    // outside them), with the thread's packet/site context; one example chain per caller.
    struct Caller {
        std::uint64_t reads = 0;
        std::uint64_t writes = 0;
        std::uint64_t underMs = 0;
        std::uint32_t thread = 0;
        std::vector<std::uint64_t> latency;
        GuestArena::GuestArenaProbeEvent example{};
    };
    std::map<std::tuple<std::uint64_t, std::uint64_t>, Caller> callers;
    std::map<std::string, std::uint64_t> callerModules;
    std::unordered_map<std::uint64_t, bool> systemCache;
    const auto system = [&](std::uint64_t address) {
        const auto [it, inserted] = systemCache.try_emplace(address, false);
        if (inserted) it->second = SystemModule(address);
        return it->second;
    };
    std::unordered_map<std::uint32_t, std::uint64_t> threads;
    std::unordered_set<std::uint64_t> readUnits;
    std::unordered_set<std::uint64_t> writtenUnits;
    std::uint64_t readHeap = 0;
    std::uint64_t writtenHeap = 0;
    std::uint64_t mainModule = 0;
    constexpr std::array<std::uint64_t, 6> bounds{1000, 4000, 16000, 64000, 256000, 1000000};
    std::array<std::uint64_t, bounds.size() + 1> histogram{};
    std::vector<std::uint64_t> readLatency;
    std::vector<std::uint64_t> writeLatency;
#ifdef _WIN32
    const auto self = reinterpret_cast<std::uint64_t>(GetModuleHandleW(nullptr));
    HMODULE owner = nullptr;
#endif
    for (const auto& event : events) {
        const bool write = event.touch != static_cast<std::uint8_t>(GuestArena::ProbeTouch::Read);
        const bool heap = (event.tag & TagAddressHeap) != 0;
        auto& site = sites[event.rip];
        (write ? site.writes : site.reads) += 1;
        site.thread = event.thread;
        ++threads[event.thread];
        std::uint64_t caller = event.rip;
        if (system(caller)) {
            caller = 0;
            for (std::size_t i = 0; i < event.frameCount && i < GuestArena::ProbeFrames; ++i) {
                if (!system(event.frames[i])) {
                    caller = event.frames[i];
                    break;
                }
            }
        }
        auto& by = callers[{caller, event.context}];
        (write ? by.writes : by.reads) += 1;
        if (event.latencyUs < 1000) ++by.underMs;
        by.latency.push_back(event.latencyUs);
        by.thread = event.thread;
        by.example = event;
        ++callerModules[caller != 0 ? ModuleOf(caller) : std::string("unresolved")];
#ifdef _WIN32
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(event.rip), &owner) && reinterpret_cast<std::uint64_t>(owner) == self) ++mainModule;
#endif
        if (write) {
            if (writtenUnits.insert(event.block).second && heap) ++writtenHeap;
            if (event.touch == static_cast<std::uint8_t>(GuestArena::ProbeTouch::Write)) writeLatency.push_back(event.latencyUs);
        } else {
            if (readUnits.insert(event.block).second && heap) ++readHeap;
            readLatency.push_back(event.latencyUs);
            std::size_t bucket = 0;
            while (bucket < bounds.size() && event.latencyUs >= bounds[bucket]) ++bucket;
            ++histogram[bucket];
        }
    }
    const auto end = [&](GuestArena::ProbeEnd reason) { return static_cast<unsigned long long>(counts.ends[static_cast<std::size_t>(reason)]); };
    const auto by = [&](GuestArena::ProbeRefusal reason) { return static_cast<unsigned long long>(counts.refused[static_cast<std::size_t>(reason)]); };
    std::uint64_t closed = 0;
    for (const auto value : counts.ends) closed += value;
    const auto touched = counts.ends[static_cast<std::size_t>(GuestArena::ProbeEnd::Read)] + counts.ends[static_cast<std::size_t>(GuestArena::ProbeEnd::Write)];
    std::uint64_t refusals = 0;
    for (const auto value : counts.refused) refusals += value;
    const auto reads = readLatency.size();
    const auto firstWrites = writeLatency.size();
    const auto afterRead = counts.afterReadWritten;
    const auto readMedian = Median(readLatency);
    const auto writeMedian = Median(writeLatency);
    char line[2048];
    std::snprintf(line, sizeof(line),
        "[ownership] probe 1/%u (APS5_PROFILE_OWNERSHIP), last %.0f s: GPU-written %zu units (%llu unit writes; %llu units in address-based heaps read in place); "
        "probe arms %llu (driver %llu armed, %llu refused; refused: outside %llu, view %llu, owned %llu, pinned %llu, host access %llu, in flight %llu, not private %llu, protection %llu, empty %llu, failed %llu, cap %llu), %llu live; "
        "ends: read %llu, write %llu, rewritten untouched %llu, host write %llu, host read %llu, query %llu, mapping %llu, owned %llu; after read: written %llu, not %llu; lost guards %llu, extra faults %llu, events dropped %llu; "
        "touched %llu of %llu closed (%.1f %%); distinct units read %zu (address heaps %llu), written %zu (address heaps %llu); touches from the main module %llu of %zu; "
        "est. stage-2 faults %llu (x%u: read %zu, write %zu, write after read %llu); first read after GPU write: <1 ms %llu, <4 ms %llu, <16 ms %llu, <64 ms %llu, <256 ms %llu, <1 s %llu, >=1 s %llu, median %.1f ms; first write median %.1f ms\n",
        rate, seconds, gpuUnits.size(), static_cast<unsigned long long>(unitWrites), static_cast<unsigned long long>(heapUnits),
        static_cast<unsigned long long>(counts.arms), static_cast<unsigned long long>(armed), static_cast<unsigned long long>(refused), by(GuestArena::ProbeRefusal::Outside), by(GuestArena::ProbeRefusal::View), by(GuestArena::ProbeRefusal::Owned), by(GuestArena::ProbeRefusal::Pinned), by(GuestArena::ProbeRefusal::HostAccess), by(GuestArena::ProbeRefusal::InFlight), by(GuestArena::ProbeRefusal::NotPrivate), by(GuestArena::ProbeRefusal::Protection), by(GuestArena::ProbeRefusal::Empty), by(GuestArena::ProbeRefusal::Failed), by(GuestArena::ProbeRefusal::Cap), static_cast<unsigned long long>(counts.live),
        end(GuestArena::ProbeEnd::Read), end(GuestArena::ProbeEnd::Write), end(GuestArena::ProbeEnd::Rewritten), end(GuestArena::ProbeEnd::HostWrite), end(GuestArena::ProbeEnd::HostRead), end(GuestArena::ProbeEnd::Query), end(GuestArena::ProbeEnd::Mapping), end(GuestArena::ProbeEnd::Owned),
        static_cast<unsigned long long>(counts.afterReadWritten), static_cast<unsigned long long>(counts.afterReadOther), static_cast<unsigned long long>(counts.lost), static_cast<unsigned long long>(counts.extraFaults), static_cast<unsigned long long>(counts.dropped),
        static_cast<unsigned long long>(touched), static_cast<unsigned long long>(closed), closed != 0 ? 100.0 * static_cast<double>(touched) / static_cast<double>(closed) : 0.0, readUnits.size(), static_cast<unsigned long long>(readHeap), writtenUnits.size(), static_cast<unsigned long long>(writtenHeap), static_cast<unsigned long long>(mainModule), events.size(),
        static_cast<unsigned long long>((reads + firstWrites + afterRead) * rate), rate, reads, firstWrites, static_cast<unsigned long long>(afterRead),
        static_cast<unsigned long long>(histogram[0]), static_cast<unsigned long long>(histogram[1]), static_cast<unsigned long long>(histogram[2]), static_cast<unsigned long long>(histogram[3]), static_cast<unsigned long long>(histogram[4]), static_cast<unsigned long long>(histogram[5]), static_cast<unsigned long long>(histogram[6]),
        static_cast<double>(readMedian) / 1000.0, static_cast<double>(writeMedian) / 1000.0);
    std::string report = line;
    std::vector<std::pair<std::uint64_t, Site>> ranked(sites.begin(), sites.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second.reads + a.second.writes > b.second.reads + b.second.writes; });
    report += "[ownership] sites (top 8 of " + std::to_string(ranked.size()) + ", " + std::to_string(threads.size()) + " threads):";
    for (std::size_t i = 0; i < ranked.size() && i < 8; ++i) {
        const auto& [rip, site] = ranked[i];
        const auto name = ThreadName(site.thread);
        report += " " + SiteName(rip) + " r" + std::to_string(site.reads) + " w" + std::to_string(site.writes) + " (thread " + std::to_string(site.thread) + (name.empty() ? "" : " '" + name + "'") + ")";
        if (i + 1 < std::min<std::size_t>(ranked.size(), 8)) report += ";";
    }
    report += "\n";
    std::vector<std::pair<std::tuple<std::uint64_t, std::uint64_t>, Caller*>> rankedCallers;
    for (auto& [key, caller] : callers) rankedCallers.emplace_back(key, &caller);
    std::sort(rankedCallers.begin(), rankedCallers.end(), [](const auto& a, const auto& b) { return a.second->reads + a.second->writes > b.second->reads + b.second->writes; });
    report += "[ownership] callers (first frame outside the CRT, by caller and packet/site; top 12 of " + std::to_string(rankedCallers.size()) + "; touches by caller module:";
    for (const auto& [module, count] : callerModules) report += " " + module + " " + std::to_string(count);
    report += "):";
    for (std::size_t i = 0; i < rankedCallers.size() && i < 12; ++i) {
        const auto& [key, caller] = rankedCallers[i];
        const auto& example = caller->example;
        const auto name = ThreadName(caller->thread);
        char head[160];
        std::snprintf(head, sizeof(head), " r%llu w%llu <1ms %llu median %.1f ms", static_cast<unsigned long long>(caller->reads), static_cast<unsigned long long>(caller->writes), static_cast<unsigned long long>(caller->underMs), static_cast<double>(Median(caller->latency)) / 1000.0);
        report += " " + (std::get<0>(key) != 0 ? SiteName(std::get<0>(key)) : std::string("unresolved")) + head + " (" + ContextName(std::get<1>(key)) + "; thread " + std::to_string(caller->thread) + (name.empty() ? "" : " '" + name + "'") + ") chain " + SiteName(example.rip);
        for (std::size_t f = 0; f < example.frameCount && f < GuestArena::ProbeFrames; ++f) report += std::string(f < example.exactFrames ? " < " : " <~ ") + SiteName(example.frames[f]);
        if (i + 1 < std::min<std::size_t>(rankedCallers.size(), 12)) report += ";";
    }
    report += "\n";
    gpuUnits.clear();
    unitWrites = armed = refused = 0;
    previousHeaps.swap(heaps);
    heaps.clear();
    return report;
}

Sampler* Global() {
#ifdef _WIN32
    static Sampler* sampler = [] {
        const auto rate = RateFromEnvironment();
        if (rate == 0) return static_cast<Sampler*>(nullptr);
        std::fprintf(stderr, "[ownership] probing one 64 KiB unit in %u of the staged GPU writes (one-shot guards, APS5_PROFILE_OWNERSHIP)\n", rate);
        return new Sampler(rate);
    }();
    return sampler;
#else
    return nullptr;
#endif
}

void Report() {
    auto* sampler = Global();
    if (sampler == nullptr) return;
    static std::atomic<std::int64_t> last{-1};
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto previous = last.load(std::memory_order_relaxed);
    if (previous < 0) {
        last.compare_exchange_strong(previous, now);
        return;
    }
    if (now - previous < 10000 || !last.compare_exchange_strong(previous, now)) return;
    const auto text = sampler->Take(static_cast<double>(now - previous) / 1000.0);
    ProfilePrint_nid_no_patch("%s", text.c_str());
}

}
