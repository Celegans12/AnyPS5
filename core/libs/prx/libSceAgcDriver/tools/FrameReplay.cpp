#include "prx/libSceAgcDriver/State/include/Configuration.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"
#include "prx/libSceAgcDriver/Execution/include/FrameDumpIo.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidentCensus.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "SceTypes.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace {

using AgcDriver::FrameCall;
using AgcDriver::FrameCallKind;
using AgcDriver::FrameCaptureReader;
using AgcDriver::FrameMemoryChunk;
using AgcDriver::FrameMemoryKind;
using AgcDriver::FrameRangeAccess;
using AgcDriver::FrameRangeKind;
using Clock = std::chrono::steady_clock;

struct Dump {
    std::uint64_t address = 0;
    std::uint64_t bytes = 0;
    std::filesystem::path path;
};

struct Options {
    std::filesystem::path directory;
    std::vector<Dump> dumps;
    std::filesystem::path frames;
    bool pace = true;
    bool strict = false;
    bool exact = false;
    std::size_t frameLimit = 0;
    std::filesystem::path checkpointReport;
    std::chrono::milliseconds paceTimeout{10000};
    std::chrono::milliseconds memoryPaceTimeout{10000};
    std::chrono::milliseconds releaseAfter{50};
    std::chrono::milliseconds drainTimeout{10000};
    std::chrono::seconds timeout{900};
    bool release = true;
    bool synthesize = true;
    // One profile window per replayed frame (ProfileWindows), closed at the frame's flip; off
    // (--wall-windows): the lines close on their wall clock as in the game.
    bool frameWindows = true;
    // The bindless heap primer: the capture's BindlessTables record (off: --no-bindless-prime), or
    // --bindless-in <file> (a first replay's --bindless-out, for captures without the record).
    bool bindlessPrime = true;
    std::filesystem::path bindlessIn;
    std::filesystem::path bindlessOut;
    // --dedupe-prologue: skip a prologue ABI resolution that repeats an identical earlier one with no
    // shader registration or memory call in between (its inputs did not change).
    bool dedupePrologue = false;
    // --host-regions: print this process's host regions inside the guest arena and exit.
    bool hostRegions = false;
};

// The prologue's provably redundant ABI resolutions (--dedupe-prologue): a resolve call is redundant
// when an identical one (kind, queue, flags, address, size, values and registers) was made since the
// last register-shader or memory call; any other call kind starts a new epoch.
class PrologueDedupe {
public:
    bool Redundant(const FrameCall& call) {
        const bool resolve = call.kind == FrameCallKind::ResolveShaderAbi || call.kind == FrameCallKind::ResolveGraphicsAbi || call.kind == FrameCallKind::ResolveGraphicsStagesAbi;
        if (!resolve) {
            ++epoch;
            return false;
        }
        std::string key;
        const auto put = [&](const void* data, std::size_t bytes) { key.append(static_cast<const char*>(data), bytes); };
        const auto kind = static_cast<std::uint32_t>(call.kind);
        put(&kind, sizeof(kind));
        put(&call.queue, sizeof(call.queue));
        put(&call.flags, sizeof(call.flags));
        put(&call.address, sizeof(call.address));
        put(&call.size, sizeof(call.size));
        const auto count = [&](std::size_t value) { put(&value, sizeof(value)); };
        count(call.values.size());
        if (!call.values.empty()) put(call.values.data(), call.values.size() * sizeof(call.values[0]));
        count(call.context.size());
        for (const auto& reg : call.context) {
            put(&reg.offset, sizeof(reg.offset));
            put(&reg.value, sizeof(reg.value));
        }
        count(call.primitive.size());
        for (const auto& reg : call.primitive) {
            put(&reg.offset, sizeof(reg.offset));
            put(&reg.value, sizeof(reg.value));
        }
        auto [entry, inserted] = seen.try_emplace(std::move(key), epoch);
        if (inserted || entry->second != epoch) {
            entry->second = epoch;
            return false;
        }
        ++skipped;
        return true;
    }
    std::uint64_t Skipped() const { return skipped; }

private:
    std::unordered_map<std::string, std::uint64_t> seen;
    std::uint64_t epoch = 0;
    std::uint64_t skipped = 0;
};

class Frames {
public:
    void MarkReady() {
        std::lock_guard lock(mutex);
        ready.push_back(Clock::now());
    }

    void Failed(std::exception_ptr error) {
        std::lock_guard lock(mutex);
        if (!failure) failure = error;
    }

    std::vector<Clock::time_point> Times() const {
        std::lock_guard lock(mutex);
        return ready;
    }

    std::exception_ptr Failure() const {
        std::lock_guard lock(mutex);
        return failure;
    }

private:
    mutable std::mutex mutex;
    std::vector<Clock::time_point> ready;
    std::exception_ptr failure;
};

class ReplayFlip final : public AgcDriver::IFlipRequest {
public:
    explicit ReplayFlip(Frames& frames) : frames(frames) {}
    void GpuReady(const std::shared_ptr<AgcDriver::FrameTiming>&) override { frames.MarkReady(); }
    void Fail(std::exception_ptr error) noexcept override { frames.Failed(error); }

private:
    Frames& frames;
};

class ReplayRenderingWait final : public AgcDriver::IRenderingWait {
public:
    void Wait() override {}
};

class ReplayOutput final : public AgcDriver::IVideoOutput {
public:
    explicit ReplayOutput(Frames& frames) : frames(frames) {}
    std::shared_ptr<AgcDriver::IFlipRequest> Reserve(const AgcDriver::FlipInfo&) override { return std::make_shared<ReplayFlip>(frames); }
    std::shared_ptr<AgcDriver::IRenderingWait> CaptureRenderingWait(std::uint32_t) override { return std::make_shared<ReplayRenderingWait>(); }
    void Fail(std::exception_ptr error) noexcept override { frames.Failed(error); }

private:
    Frames& frames;
};

[[noreturn]] void hardExit(int code) {
    std::fflush(stdout);
    std::fflush(stderr);
#ifdef _WIN32
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(code));
#endif
    std::_Exit(code);
}

void startWatchdog(std::chrono::seconds timeout) {
    std::thread([timeout] {
        const auto deadline = Clock::now() + timeout;
        while (timeout.count() <= 0 || Clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (!AgcDriverFailed_nid_postfix()) continue;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::fprintf(stderr, "agc_frame_replay: the driver failed; if it lost the GPU device (Vulkan result -4), the NVIDIA shader store may have been reset: check the game dictionary before the next game run; exiting without a driver shutdown\n");
            hardExit(6);
        }
        std::fprintf(stderr, "agc_frame_replay: no result after %lld s (--timeout-s); exiting without a driver shutdown\n", static_cast<long long>(timeout.count()));
        hardExit(4);
    }).detach();
}

std::uint64_t headerNumber(const std::string& header, std::string_view key) {
    const auto quoted = "\"" + std::string(key) + "\":";
    const auto at = header.find(quoted);
    if (at == std::string::npos) throw std::runtime_error("frame replay: the capture header has no " + std::string(key));
    return std::strtoull(header.c_str() + at + quoted.size(), nullptr, 10);
}

// Host regions: what this process had allocated inside the guest arena's range before libc reserved it at static
// init (GuestArena.cpp: "_holes"/"_hostRegions": the process heap, early thread stacks, early CRT heaps). Their
// addresses are randomized per process start (high-entropy bottom-up ASLR), so in some starts one lands on a range
// the capture's process used, and the replay cannot map it ("fixed guest mapping ... overlaps a host region").
// HostCollisions lists the captured ranges that overlap one; the replay then starts itself again (main).
using Interval = std::pair<std::uint64_t, std::uint64_t>;

std::vector<Interval> HostCollisions(const FrameCaptureReader& reader, const std::function<bool(std::uint64_t, std::uint64_t)>& overlaps) {
    std::vector<Interval> found;
    for (const auto& chunk : reader.Chunks()) {
        if (chunk.kind != FrameMemoryKind::Range || chunk.rangeKind == FrameRangeKind::Host) continue;
        if (overlaps(chunk.address, chunk.bytes)) found.emplace_back(chunk.address, chunk.address + chunk.bytes);
    }
    return found;
}

#ifdef _WIN32
// What owns the host allocation at `address`: its region, type and protection, and whether it is a heap of this
// process, a thread's stack or an image.
std::string DescribeHostRegion(std::uint64_t address) {
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(address)), &info, sizeof(info)) == 0) return "not queryable";
    const auto base = reinterpret_cast<std::uintptr_t>(info.AllocationBase);
    std::string owner;
    if (info.Type == MEM_IMAGE) {
        char name[MAX_PATH]{};
        owner = GetModuleFileNameA(reinterpret_cast<HMODULE>(base), name, MAX_PATH) != 0 ? std::string("image ") + name : "image";
    }
    std::vector<HANDLE> heaps(256);
    const auto count = GetProcessHeaps(static_cast<DWORD>(heaps.size()), heaps.data());
    for (DWORD index = 0; index < count && index < heaps.size(); ++index) {
        if (reinterpret_cast<std::uintptr_t>(heaps[index]) == base) owner = std::string("heap ") + (heaps[index] == GetProcessHeap() ? "(the process heap)" : std::to_string(index));
    }
    using Query = LONG(WINAPI*)(HANDLE, int, void*, ULONG, ULONG*);
    const auto query = reinterpret_cast<Query>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQueryInformationThread")));
    if (const auto snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0); snapshot != INVALID_HANDLE_VALUE && query != nullptr) {
        THREADENTRY32 entry{};
        entry.dwSize = sizeof(entry);
        for (bool more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
            if (entry.th32OwnerProcessID != GetCurrentProcessId()) continue;
            const auto thread = OpenThread(THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID);
            if (thread == nullptr) continue;
            struct { LONG status; void* teb; void* process; void* thread; std::uintptr_t affinity; LONG priority; LONG basePriority; } basic{};
            if (query(thread, 0, &basic, sizeof(basic), nullptr) == 0 && basic.teb != nullptr) {
                const auto* tib = static_cast<const NT_TIB*>(basic.teb);
                MEMORY_BASIC_INFORMATION stack{};
                if (VirtualQuery(tib->StackLimit, &stack, sizeof(stack)) != 0 && stack.AllocationBase == info.AllocationBase) owner = "stack of thread " + std::to_string(entry.th32ThreadID);
                if (reinterpret_cast<std::uintptr_t>(basic.teb) >= base && reinterpret_cast<std::uintptr_t>(basic.teb) < reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize) owner = "TEB of thread " + std::to_string(entry.th32ThreadID);
            }
            CloseHandle(thread);
        }
        CloseHandle(snapshot);
    }
    char text[320];
    std::snprintf(text, sizeof(text), "region 0x%llx+0x%llx in allocation 0x%llx, %s %s, protect 0x%lx (allocated 0x%lx)%s%s",
                  static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info.BaseAddress)), static_cast<unsigned long long>(info.RegionSize), static_cast<unsigned long long>(base),
                  info.State == MEM_COMMIT ? "committed" : info.State == MEM_RESERVE ? "reserved" : "free",
                  info.Type == MEM_IMAGE ? "image" : info.Type == MEM_MAPPED ? "mapped" : "private", info.Protect, info.AllocationProtect,
                  owner.empty() ? "" : ": ", owner.c_str());
    return text;
}

// --host-regions: every host region inside the guest arena of this process, described.
int PrintHostRegions() {
    std::uintptr_t base = 0;
    std::size_t size = 0;
    GuestArena::GuestArenaRange_nid_postfix(&base, &size);
    std::size_t regions = 0;
    for (std::uintptr_t at = base; at < base + size;) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) == 0) break;
        const auto next = reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize;
        if (info.State != MEM_FREE && GuestArena::GuestArenaHostRegionOverlaps_nid_postfix(at, std::min<std::uintptr_t>(next, base + size) - at)) {
            std::printf("host region %s\n", DescribeHostRegion(at).c_str());
            ++regions;
        }
        at = next;
    }
    std::printf("%zu host regions in the guest arena 0x%llx+0x%llx\n", regions, static_cast<unsigned long long>(base), static_cast<unsigned long long>(size));
    return 0;
}
#endif

void restore(const FrameCaptureReader& reader, const FrameMemoryChunk& chunk) {
    reader.Read(chunk, std::span(reinterpret_cast<std::byte*>(static_cast<std::uintptr_t>(chunk.address)), static_cast<std::size_t>(chunk.bytes)));
}

bool committedWritable(std::uint64_t address, std::uint64_t bytes) {
#ifdef _WIN32
    for (auto at = address; at < address + bytes;) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), &info, sizeof(info)) == 0 || info.State != MEM_COMMIT) return false;
        const auto access = info.Protect & 0xffu;
        if ((info.Protect & PAGE_GUARD) != 0 || (access != PAGE_READWRITE && access != PAGE_EXECUTE_READWRITE)) return false;
        at = reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize;
    }
    return true;
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
    return false;
#endif
}

std::pair<std::uint64_t, std::uint64_t> progress();

struct Skipped {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::uint64_t bytes = 0;

    void Add(const FrameMemoryChunk& chunk) {
        ranges.emplace_back(chunk.address, chunk.address + chunk.bytes);
        bytes += chunk.bytes;
    }

    bool Overlaps(const FrameMemoryChunk& chunk) const {
        return std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return chunk.address < range.second && range.first < chunk.address + chunk.bytes; });
    }
};

void mapHostRanges(const FrameCaptureReader& reader, Skipped& skipped) {
    std::vector<const FrameMemoryChunk*> host;
    for (const auto& chunk : reader.Chunks()) {
        if (chunk.kind == FrameMemoryKind::Range && chunk.rangeKind == FrameRangeKind::Host) host.push_back(&chunk);
    }
#ifdef _WIN32
    constexpr std::uint64_t Granularity = 64u * 1024u;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> envelopes;
    for (const auto* chunk : host) {
        const auto begin = chunk->address & ~(Granularity - 1);
        const auto end = (chunk->address + chunk->bytes + Granularity - 1) & ~(Granularity - 1);
        if (!envelopes.empty() && begin <= envelopes.back().second) envelopes.back().second = std::max(envelopes.back().second, end);
        else envelopes.emplace_back(begin, end);
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> refused;
    for (const auto& [begin, end] : envelopes) {
        if (VirtualAlloc(reinterpret_cast<void*>(static_cast<std::uintptr_t>(begin)), static_cast<SIZE_T>(end - begin), MEM_RESERVE, PAGE_NOACCESS) == nullptr) refused.emplace_back(begin, end);
    }
    for (const auto* chunk : host) {
        const bool blocked = std::any_of(refused.begin(), refused.end(), [&](const auto& range) { return chunk->address < range.second && range.first < chunk->address + chunk->bytes; });
        if (!blocked && VirtualAlloc(reinterpret_cast<void*>(static_cast<std::uintptr_t>(chunk->address)), static_cast<SIZE_T>(chunk->bytes), MEM_COMMIT, PAGE_READWRITE) != nullptr) continue;
        if ((chunk->access & FrameRangeAccess::Gpu) != 0) throw std::runtime_error("frame replay: cannot map the GPU-visible host range at " + std::to_string(chunk->address));
        skipped.Add(*chunk);
    }
#else
    for (const auto* chunk : host) skipped.Add(*chunk);
#endif
}

struct AliasedPages {
    std::size_t pageBytes = 0;
    std::map<std::uint64_t, std::uint64_t> groupOf;
};

AliasedPages aliasedPages(const FrameCaptureReader& reader) {
    AliasedPages result;
    for (const auto& call : reader.Calls()) {
        if (call.kind != FrameCallKind::AliasedPages) continue;
        result.pageBytes = static_cast<std::size_t>(call.size);
        for (std::size_t index = 0; index + 1 < call.values.size(); index += 2) result.groupOf[call.values[index]] = call.values[index + 1];
    }
    return result;
}

void commitWithAliases(void* pointer, std::size_t bytes, const AliasedPages& aliases, std::map<std::uint64_t, void*>& sections) {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    const auto end = begin + bytes;
    auto alias = aliases.groupOf.lower_bound(begin);
    if (alias == aliases.groupOf.end() || alias->first >= end) {
        GuestArena::GuestArenaCommit_nid_postfix(pointer, bytes, 0x04, bytes);
        return;
    }
#ifdef _WIN32
    for (auto at = begin; at < end;) {
        if (alias != aliases.groupOf.end() && alias->first == at) {
            auto& section = sections[alias->second];
            if (section == nullptr) {
                section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(aliases.pageBytes), nullptr);
                if (section == nullptr) throw std::runtime_error("frame replay: cannot create the shared backing of an aliased page");
            }
            GuestArena::GuestArenaMapDirect_nid_postfix(reinterpret_cast<void*>(at), aliases.pageBytes, section, 0, PAGE_READWRITE);
            at += aliases.pageBytes;
            ++alias;
            continue;
        }
        const auto next = alias != aliases.groupOf.end() && alias->first < end ? static_cast<std::uintptr_t>(alias->first) : end;
        GuestArena::GuestArenaCommit_nid_postfix(reinterpret_cast<void*>(at), next - at, 0x04, next - at);
        at = next;
    }
#else
    throw std::runtime_error("frame replay: aliased guest pages are replayed on Windows only");
#endif
}

Skipped restoreMemory(const FrameCaptureReader& reader) {
    std::uintptr_t base = 0;
    std::size_t size = 0;
    GuestArena::GuestArenaRange_nid_postfix(&base, &size);
    if (!reader.Header().empty() && headerNumber(reader.Header(), "arena_base") != base) throw std::runtime_error("frame replay: the guest arena is at a different address than in the capture");
    progress();
    Skipped skipped;
    mapHostRanges(reader, skipped);
    const auto aliases = aliasedPages(reader);
    std::map<std::uint64_t, void*> sections;
    {
        GuestAllocations::Mutation mutation;
        for (const auto& chunk : reader.Chunks()) {
            if (chunk.kind != FrameMemoryKind::Range || chunk.rangeKind == FrameRangeKind::Host) continue;
            const bool gpu = (chunk.access & FrameRangeAccess::Gpu) != 0;
            if (committedWritable(chunk.address, chunk.bytes)) {
                if (!gpu) skipped.Add(chunk);
                else std::memset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(chunk.address)), 0, static_cast<std::size_t>(chunk.bytes));
                continue;
            }
            auto* const pointer = reinterpret_cast<void*>(static_cast<std::uintptr_t>(chunk.address));
            const auto bytes = static_cast<std::size_t>(chunk.bytes);
            try {
                GuestArena::GuestArenaMarkUsed_nid_postfix(pointer, bytes);
            } catch (const std::exception&) {
                if (gpu) throw;
                skipped.Add(chunk);
                continue;
            }
            commitWithAliases(pointer, bytes, aliases, sections);
            mutation.Add(pointer, bytes, (chunk.access & FrameRangeAccess::Readable) != 0, (chunk.access & FrameRangeAccess::Writable) != 0, gpu);
        }
    }
#ifdef _WIN32
    for (const auto& [group, section] : sections) CloseHandle(section);
#endif
    if (!aliases.groupOf.empty()) std::fprintf(stderr, "[frame-replay] %zu aliased pages share %zu backings\n", aliases.groupOf.size(), sections.size());
    std::vector<const FrameMemoryChunk*> snapshot;
    for (const auto& chunk : reader.Chunks()) {
        if (chunk.kind == FrameMemoryKind::Snapshot && !skipped.Overlaps(chunk)) snapshot.push_back(&chunk);
    }
    const auto workers = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    for (std::size_t first = 0; first < snapshot.size(); first += workers) {
        std::vector<std::future<void>> pending;
        for (auto index = first; index < std::min(snapshot.size(), first + workers); ++index) {
            pending.push_back(std::async(std::launch::async, [&reader, chunk = snapshot[index]] { restore(reader, *chunk); }));
        }
        for (auto& future : pending) future.get();
    }
    return skipped;
}

void restoreRewritten(const FrameCaptureReader& reader, std::span<const FrameMemoryChunk* const> rewritten, const Skipped& skipped) {
    std::vector<const FrameMemoryChunk*> snapshot;
    for (const auto& chunk : reader.Chunks()) {
        if (chunk.kind == FrameMemoryKind::Snapshot && !skipped.Overlaps(chunk)) snapshot.push_back(&chunk);
    }
    std::sort(snapshot.begin(), snapshot.end(), [](const auto* left, const auto* right) { return left->address < right->address; });
    std::map<const FrameMemoryChunk*, std::vector<std::pair<std::uint64_t, std::uint64_t>>> pieces;
    for (const auto* chunk : rewritten) {
        if (skipped.Overlaps(*chunk)) continue;
        const auto begin = chunk->address;
        const auto end = chunk->address + chunk->bytes;
        std::memset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(begin)), 0, static_cast<std::size_t>(chunk->bytes));
        auto at = std::upper_bound(snapshot.begin(), snapshot.end(), begin, [](std::uint64_t address, const auto* candidate) { return address < candidate->address; });
        if (at != snapshot.begin()) --at;
        for (; at != snapshot.end() && (*at)->address < end; ++at) {
            const auto low = std::max(begin, (*at)->address);
            const auto high = std::min(end, (*at)->address + (*at)->bytes);
            if (low < high) pieces[*at].emplace_back(low, high);
        }
    }
    for (const auto& [chunk, ranges] : pieces) {
        const auto bytes = reader.Read(*chunk);
        for (const auto& [low, high] : ranges) std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(low)), bytes.data() + (low - chunk->address), static_cast<std::size_t>(high - low));
    }
}

std::pair<std::uint64_t, std::uint64_t> progress() {
    std::uint64_t accepted = 0;
    std::uint64_t completed = 0;
    AgcDriverSerialProgress_nid_postfix(&accepted, &completed);
    return {accepted, completed};
}

struct Replay {
    const Options& options;
    const FrameCaptureReader& reader;
    Frames frames;
    std::map<std::uint32_t, std::shared_ptr<ReplayOutput>> outputs;
    std::mutex memoryMutex;
    std::map<std::uint64_t, std::vector<const FrameMemoryChunk*>> groups;
    Skipped skipped;
    std::atomic<std::uint64_t> paceTimeouts{0};
    std::atomic<std::uint64_t> memoryPaceTimeouts{0};
    std::atomic<std::uint64_t> staleRegistrations{0};
    std::atomic<std::uint64_t> failedRegistrations{0};
    std::atomic<std::uint64_t> submits{0};
    std::atomic<std::uint64_t> deltaBytes{0};
    std::atomic<std::int64_t> paceWaitedNs{0};
    std::atomic<bool> stalled{false};
    std::atomic<std::uint64_t> stalledAt{0};
    std::atomic<std::uint64_t> earlyReleases{0};
    std::atomic<std::uint64_t> lateReleases{0};
    std::atomic<std::uint64_t> syntheticReleases{0};
    std::atomic<std::size_t> sequenced{0};
    std::atomic<int> busy{0};
    std::atomic<int> pacing{0};
    struct LabelWrite {
        std::uint64_t address = 0;
        std::uint64_t value = 0;
        bool known = false;
    };
    std::vector<LabelWrite> labelWrites;
    std::map<std::uint64_t, std::uint64_t> waitOrdinals;
    std::atomic<std::uint64_t> parkedReleases{0};
    std::atomic<std::uint64_t> fallbackReleases{0};
    std::atomic<bool> parkedWithoutRelease{false};
    std::uint64_t unfinished = 0;
    std::size_t threads = 0;
    bool recording = false;
    // Frame windows: the call that ends a frame (a CPU flip, or the suspend point when the title
    // flips from the GPU), the windows closed and those closed before the GPU work drained.
    bool closeFrames = false;
    FrameCallKind boundary = FrameCallKind::Flip;
    std::atomic<std::size_t> framesClosed{0};
    std::atomic<std::size_t> undrainedWindows{0};
    // A frame's windows are closing: the next call waits for it, so the release timer (runWindow's
    // breaker) must not count that time as no progress and release awaited dwords early.
    std::atomic<bool> closingWindows{false};
    std::size_t bindlessPrimed = 0;
    std::vector<std::uint64_t> deferredBindless;
    bool bindlessWritten = false;

    void primeBindless(std::span<const std::uint64_t> words) {
        if (!options.bindlessPrime || words.empty()) return;
        AgcDriverPrimeBindlessTables_nid_postfix(words);
        bindlessPrimed += static_cast<std::size_t>(words.front());
    }

    // --bindless-out: the tables registered by the end of the first replayed frame (or of the replay).
    void writeBindless() {
        if (options.bindlessOut.empty() || bindlessWritten) return;
        bindlessWritten = true;
        const auto words = AgcDriverBindlessRegistrations_nid_postfix();
        std::ofstream file(options.bindlessOut, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * sizeof(std::uint64_t)));
        if (!file.good()) throw std::runtime_error("frame replay: cannot write " + options.bindlessOut.string());
        std::fprintf(stderr, "[frame-replay] %llu bindless tables written to %s\n", static_cast<unsigned long long>(words.empty() ? 0 : words.front()), options.bindlessOut.string().c_str());
    }
    Clock::time_point startedAt{};
    Clock::time_point endedAt{};

    double paceWaitedSeconds() const {
        return static_cast<double>(paceWaitedNs.load()) / 1e9;
    }

    void report() const {
        const auto [accepted, completed] = progress();
        std::fprintf(stderr, "[frame-replay] %llu submits, %zu frames ready, %.1f s since the first frame, pacing waited %.1f s (%llu timeouts), serials %llu/%llu\n",
                     static_cast<unsigned long long>(submits.load()), frames.Times().size(), std::chrono::duration<double>(Clock::now() - startedAt).count(), paceWaitedSeconds(),
                     static_cast<unsigned long long>(paceTimeouts.load()), static_cast<unsigned long long>(completed), static_cast<unsigned long long>(accepted));
    }

    static bool paced(FrameCallKind kind) {
        return kind == FrameCallKind::Submit || kind == FrameCallKind::Flip || kind == FrameCallKind::ReserveEopFlip || kind == FrameCallKind::Suspend || kind == FrameCallKind::Memory;
    }

    void pace(const FrameCall& call) {
        if (!options.pace || !recording || !paced(call.kind) || call.accepted < call.completed) return;
        const bool memory = call.kind == FrameCallKind::Memory;
        if (!memory && stalled.load()) {
            if (progress().second == stalledAt.load()) return;
            stalled.store(false);
        }
        const auto inFlight = options.exact || memory ? 0 : call.accepted - call.completed;
        const auto began = Clock::now();
        const auto timeout = memory ? options.memoryPaceTimeout : options.paceTimeout;
        auto lastCompleted = progress().second;
        auto deadline = began + timeout;
        auto quietSince = began;
        const auto waited = [&] { paceWaitedNs += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - began).count(); };
        struct Parked {
            std::atomic<int>& busy;
            std::atomic<int>& pacing;
            Parked(std::atomic<int>& counter, std::atomic<int>& waiting) : busy(counter), pacing(waiting) {
                --busy;
                ++pacing;
            }
            ~Parked() {
                --pacing;
                ++busy;
            }
        } parked{busy, pacing};
        while (true) {
            const auto [accepted, completed] = progress();
            if (accepted - completed <= inFlight) {
                if (options.exact && accepted == completed) AgcDriverWaitIdle_nid_postfix();
                return waited();
            }
            const auto now = Clock::now();
            if (completed != lastCompleted) {
                lastCompleted = completed;
                deadline = now + timeout;
                quietSince = now;
            } else if (!AgcDriverQuiescent_nid_postfix()) {
                deadline = now + timeout;
            }
            if (memory && deltaReleasesWait(call)) return waited();
            if (now - quietSince >= (options.exact ? ExactQuiet : BlockedQuiet) && blockedOnWaits()) return waited();
            if (now >= deadline) {
                waited();
                if (memory) {
                    ++memoryPaceTimeouts;
                    std::fprintf(stderr, "[frame-replay] memory pacing gave up at call %llu: no completion for %lld ms, serials %llu/%llu, the capture had %llu in flight\n", static_cast<unsigned long long>(call.ordinal),
                                 static_cast<long long>(timeout.count()), static_cast<unsigned long long>(completed), static_cast<unsigned long long>(accepted), static_cast<unsigned long long>(inFlight));
                    if (options.strict) throw std::runtime_error("frame replay: memory call " + std::to_string(call.ordinal) + " waited longer than the pacing timeout");
                    return;
                }
                stalled.store(true);
                stalledAt.store(completed);
                std::fprintf(stderr, "[frame-replay] pacing gave up at call %llu (%s): serials %llu/%llu, the capture had %llu in flight\n", static_cast<unsigned long long>(call.ordinal),
                             std::string(AgcDriver::FrameCallName(call.kind)).c_str(), static_cast<unsigned long long>(completed), static_cast<unsigned long long>(accepted), static_cast<unsigned long long>(inFlight));
                ++paceTimeouts;
                if (options.strict) throw std::runtime_error("frame replay: call " + std::to_string(call.ordinal) + " waited longer than the pacing timeout");
                return;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    static constexpr auto ExactQuiet = std::chrono::milliseconds(50);
    static constexpr auto ParkedQuiet = std::chrono::milliseconds(5);
    static constexpr auto BlockedQuiet = std::chrono::milliseconds(5);

    bool deltaReleasesWait(const FrameCall& call) {
        const auto found = groups.find(call.values.at(0));
        if (found == groups.end()) return false;
        std::array<AgcDriver::AwaitedWait, 64> awaited{};
        const auto count = AgcDriverAwaitedWaits_nid_postfix(awaited.data(), awaited.size());
        for (std::size_t slot = 0; slot < count; ++slot) {
            const auto& wait = awaited[slot];
            if (wait.address == 0) continue;
            for (const auto* chunk : found->second) {
                if (chunk->address <= wait.address && wait.address < chunk->address + chunk->bytes) return true;
            }
        }
        return false;
    }

    bool blockedOnWaits() {
        if (!AgcDriverQuiescent_nid_postfix()) return false;
        std::lock_guard lock(memoryMutex);
        std::array<AgcDriver::AwaitedWait, 64> awaited{};
        const auto count = AgcDriverAwaitedWaits_nid_postfix(awaited.data(), awaited.size());
        for (std::size_t slot = 0; slot < count; ++slot) {
            const auto& wait = awaited[slot];
            if (wait.address != 0 && (wait.function == 0 || !satisfies(wait, load(wait.address, wait.bytes == 8 ? 8 : 4)))) return true;
        }
        return false;
    }

    void execute(const FrameCall& call) {
        pace(call);
        switch (call.kind) {
        case FrameCallKind::RegisterShader:
            registration([&] { AgcDriverRegisterShader_nid_postfix(shader(call.address)); });
            return;
        case FrameCallKind::ResolveShaderAbi:
            registration([&] { AgcDriverResolveShaderAbi_nid_postfix(shader(call.address), call.context, call.primitive); });
            return;
        case FrameCallKind::ResolveGraphicsAbi:
            registration([&] { AgcDriverResolveGraphicsAbi_nid_postfix(shader(call.values.at(0)), shader(call.values.at(1)), static_cast<std::uint32_t>(call.values.at(2))); });
            return;
        case FrameCallKind::ResolveGraphicsStagesAbi: {
            std::vector<const Shader*> stages;
            for (const auto value : call.values) stages.push_back(shader(value));
            registration([&] { AgcDriverResolveGraphicsStagesAbi_nid_postfix(stages, call.context, call.primitive); });
            return;
        }
        case FrameCallKind::RegisterVideoOutput: {
            const auto handle = static_cast<std::uint32_t>(call.values.at(0));
            auto output = std::make_shared<ReplayOutput>(frames);
            AgcDriverRegisterVideoOutput_nid_postfix(handle, output);
            outputs.emplace(handle, std::move(output));
            return;
        }
        case FrameCallKind::UnregisterVideoOutput: {
            const auto handle = static_cast<std::uint32_t>(call.values.at(0));
            AgcDriverUnregisterVideoOutput_nid_postfix(handle, outputs.at(handle));
            outputs.erase(handle);
            return;
        }
        case FrameCallKind::SetTessellationFactorRing:
            sceAgcDriverSetTFRing(reinterpret_cast<const volatile void*>(static_cast<std::uintptr_t>(call.address)), static_cast<std::uint32_t>(call.size));
            return;
        case FrameCallKind::Submit: {
            const Packet packet{reinterpret_cast<std::uint32_t*>(static_cast<std::uintptr_t>(call.address)), static_cast<std::uint32_t>(call.size), static_cast<std::uint8_t>(call.flags), {}};
            if (recording && options.exact) noteLabelWrites(call.address, call.size, 0, call.ordinal);
            if (call.queue == 0) sceAgcDriverSubmitDcb(&packet);
            else sceAgcDriverSubmitAcb(call.queue, &packet);
            if (++submits % 25 == 0) report();
            return;
        }
        case FrameCallKind::Flip:
            AgcDriverSubmitFlip_nid_postfix(flipInfo(call));
            if (closeFrames && boundary == FrameCallKind::Flip) closeFrame();
            return;
        case FrameCallKind::ReserveEopFlip: {
            const auto token = static_cast<std::uint32_t>(call.values.at(4));
            if (AgcDriverReserveEopFlipAt_nid_postfix(flipInfo(call), token) != token) throw std::runtime_error("frame replay: EOP flip token " + std::to_string(token) + " was not reserved");
            return;
        }
        case FrameCallKind::Suspend:
            AgcDriverSuspendPoint_nid_postfix();
            if (closeFrames && boundary == FrameCallKind::Suspend) closeFrame();
            return;
        case FrameCallKind::AliasedPages:
        case FrameCallKind::Checkpoint:
            return;
        case FrameCallKind::BindlessTables:
            // In the prologue: primed once the prologue's memory is restored (run).
            if (!recording) deferredBindless = call.values;
            else primeBindless(call.values);
            return;
        case FrameCallKind::QueueStates: {
            std::vector<std::uint32_t> words;
            words.reserve(call.values.size());
            for (const auto value : call.values) words.push_back(static_cast<std::uint32_t>(value));
            AgcDriverLoadQueueStates_nid_postfix(words);
            return;
        }
        case FrameCallKind::Memory: {
            std::lock_guard lock(memoryMutex);
            const auto found = groups.find(call.values.at(0));
            if (found == groups.end()) return;
            for (const auto* chunk : found->second) {
                if (skipped.Overlaps(*chunk)) continue;
                restore(reader, *chunk);
                if (recording) {
                    deltaBytes += chunk->bytes;
                    AgcDriver::Graphics::ResidentCensus::NoteCpuWrite(chunk->address, chunk->bytes);
                }
            }
            return;
        }
        case FrameCallKind::FrameStart:
        case FrameCallKind::End:
        case FrameCallKind::DisplayBuffer:
            return;
        }
        throw std::runtime_error("frame replay: unknown call kind");
    }

    // Closes the profile windows of the frame that just flipped: once the GPU work submitted so far
    // completed (as --exact pacing waits; a GPU blocked on a wait that a later call releases ends
    // the wait after 10 s, and the window then holds what completed), as "frame-<n>".
    void closeFrame() {
        closingWindows.store(true);
        struct Closed {
            std::atomic<bool>& flag;
            ~Closed() { flag.store(false); }
        } closed{closingWindows};
        const auto deadline = Clock::now() + std::chrono::seconds(10);
        auto quietSince = Clock::now();
        auto lastCompleted = progress().second;
        bool drained = false;
        while (true) {
            const auto [accepted, completed] = progress();
            if (accepted == completed) {
                drained = true;
                break;
            }
            const auto now = Clock::now();
            if (completed != lastCompleted) {
                lastCompleted = completed;
                quietSince = now;
            }
            if (now >= deadline || (now - quietSince >= ExactQuiet && blockedOnWaits())) break;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        if (drained) AgcDriverWaitIdle_nid_postfix();
        else ++undrainedWindows;
        const auto label = "frame-" + std::to_string(++framesClosed);
        AgcDriver::ProfileWindows::Close(label.c_str());
        if (framesClosed.load() == 1) writeBindless();
    }

    static std::uint64_t load(std::uint64_t address, std::size_t bytes) {
        std::uint64_t value = 0;
        std::memcpy(&value, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(address)), bytes);
        return value;
    }

    static bool satisfies(const AgcDriver::AwaitedWait& wait, std::uint64_t value) {
        const auto masked = value & wait.mask;
        const auto reference = wait.reference & wait.mask;
        switch (wait.function) {
        case 1: return masked < reference;
        case 2: return masked <= reference;
        case 3: return masked == reference;
        case 4: return masked != reference;
        case 5: return masked >= reference;
        case 6: return masked > reference;
        default: return false;
        }
    }

    static std::optional<std::uint64_t> satisfying(const AgcDriver::AwaitedWait& wait, std::uint64_t current) {
        const auto keep = current & ~wait.mask;
        const auto reference = wait.reference & wait.mask;
        switch (wait.function) {
        case 2:
        case 3:
        case 5: return keep | reference;
        case 1: return reference == 0 ? std::nullopt : std::optional<std::uint64_t>(keep);
        case 4: return keep | (~reference & wait.mask);
        case 6: return reference == wait.mask ? std::nullopt : std::optional<std::uint64_t>(keep | ((reference + 1) & wait.mask));
        default: return std::nullopt;
        }
    }

    void noteLabelWrites(std::uint64_t address, std::uint64_t dwords, int depth, std::uint64_t ordinal) {
        if (address < 0x1000 || address >> 48 != 0 || dwords == 0 || dwords > 0xfffff || depth > 8) return;
        const auto* const words = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(address));
        const auto at48 = [](std::uint32_t low, std::uint32_t high) { return low | static_cast<std::uint64_t>(high & 0xffff) << 32; };
        std::vector<LabelWrite> found;
        std::vector<std::uint64_t> waits;
        for (std::uint64_t at = 0; at < dwords;) {
            const auto header = words[at];
            if (header >> 30 != 3) {
                ++at;
                continue;
            }
            const std::uint64_t length = ((header >> 16) & 0x3fff) + 1;
            if (at + 1 + length > dwords) break;
            const auto* const body = words + at + 1;
            switch ((header >> 8) & 0xff) {
            case 0x49:
                if (length >= 6) {
                    const auto select = (body[1] >> 29) & 7;
                    if (select == 1) found.push_back({at48(body[2], body[3]), body[4], true});
                    else if (select == 2) found.push_back({at48(body[2], body[3]), body[4] | static_cast<std::uint64_t>(body[5]) << 32, true});
                    else if (select != 0) found.push_back({at48(body[2], body[3]), 0, false});
                }
                break;
            case 0x47:
                if (length >= 4) {
                    const auto select = (body[2] >> 29) & 7;
                    if (select == 1 || select == 2) found.push_back({at48(body[1], body[2]), body[3], true});
                    else if (select != 0) found.push_back({at48(body[1], body[2]), 0, false});
                }
                break;
            case 0x37:
                if (length >= 4) found.push_back({at48(body[1], body[2]), body[3], true});
                break;
            case 0x40:
            case 0x50:
                if (length >= 5) found.push_back({at48(body[3], body[4]), 0, false});
                break;
            case 0x1e:
                if (length >= 3) found.push_back({at48(body[1], body[2]), 0, false});
                break;
            case 0x33:
            case 0x3f:
                if (length >= 3) noteLabelWrites(at48(body[0] & ~3u, body[1]), body[2] & 0xfffff, depth + 1, ordinal);
                break;
            case 0x3c:
                if (length >= 5 && (body[0] >> 4) & 1) waits.push_back(at48(body[1] & ~3u, body[2]));
                break;
            default:
                break;
            }
            at += 1 + length;
        }
        if (found.empty() && waits.empty()) return;
        std::lock_guard lock(memoryMutex);
        labelWrites.insert(labelWrites.end(), found.begin(), found.end());
        for (const auto wait : waits) waitOrdinals.emplace(wait, ordinal);
    }

    bool writerInWindow(const AgcDriver::AwaitedWait& wait) const {
        return std::any_of(labelWrites.begin(), labelWrites.end(), [&](const LabelWrite& write) { return write.address == wait.address && (!write.known || wait.function == 0 || satisfies(wait, write.value)); });
    }

    bool writeRecordedValue(const FrameCall& call, const AgcDriver::AwaitedWait& wait) {
        if (call.kind != FrameCallKind::Memory) return false;
        const auto found = groups.find(call.values.at(0));
        if (found == groups.end()) return false;
        const auto address = wait.address;
        for (const auto* chunk : found->second) {
            if (chunk->address > address || address + 4 > chunk->address + chunk->bytes || skipped.Overlaps(*chunk)) continue;
            const auto bytes = reader.Read(*chunk);
            const auto length = static_cast<std::size_t>(std::min<std::uint64_t>(wait.bytes == 8 ? 8 : 4, chunk->address + chunk->bytes - address));
            const auto* const recorded = bytes.data() + (address - chunk->address);
            auto* const target = reinterpret_cast<std::byte*>(static_cast<std::uintptr_t>(address));
            if (std::memcmp(target, recorded, length) == 0) return false;
            std::uint64_t value = 0;
            std::memcpy(&value, recorded, length);
            if (wait.function != 0 && !satisfies(wait, value)) return false;
            std::memcpy(target, recorded, length);
            return true;
        }
        return false;
    }

    std::size_t recordedIndex(std::span<const FrameCall> calls, std::size_t from, const AgcDriver::AwaitedWait& wait) const {
        for (auto index = from; index < calls.size(); ++index) {
            if (calls[index].kind != FrameCallKind::Memory) continue;
            const auto found = groups.find(calls[index].values.at(0));
            if (found == groups.end()) continue;
            for (const auto* chunk : found->second) {
                if (chunk->address <= wait.address && wait.address + 4 <= chunk->address + chunk->bytes) return index;
            }
        }
        return calls.size();
    }

    void releaseAwaited(std::span<const FrameCall> calls, std::size_t from) {
        std::lock_guard lock(memoryMutex);
        std::array<AgcDriver::AwaitedWait, 64> awaited{};
        const auto count = AgcDriverAwaitedWaits_nid_postfix(awaited.data(), awaited.size());
        if (options.exact) {
            std::array<std::tuple<std::size_t, std::uint64_t, std::uint64_t>, 64> keys{};
            std::array<std::size_t, 64> order{};
            for (std::size_t slot = 0; slot < count; ++slot) {
                const auto waiting = waitOrdinals.find(awaited[slot].address);
                keys[slot] = {awaited[slot].address == 0 ? calls.size() + 1 : recordedIndex(calls, from, awaited[slot]), waiting == waitOrdinals.end() ? ~std::uint64_t{0} : waiting->second, awaited[slot].address};
                order[slot] = slot;
            }
            std::sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(count), [&](std::size_t left, std::size_t right) { return keys[left] < keys[right]; });
            std::array<AgcDriver::AwaitedWait, 64> sorted{};
            for (std::size_t slot = 0; slot < count; ++slot) sorted[slot] = awaited[order[slot]];
            awaited = sorted;
        }
        for (int pass = options.exact ? 0 : 1; pass < 2; ++pass)
        for (std::size_t slot = 0; slot < count; ++slot) {
            const auto& wait = awaited[slot];
            if (wait.address == 0) continue;
            const std::size_t bytes = wait.bytes == 8 ? 8 : 4;
            if (wait.function != 0 && satisfies(wait, load(wait.address, bytes))) continue;
            const bool deferred = pass == 0 && writerInWindow(wait);
            bool released = false;
            for (auto index = from; index < calls.size() && !released; ++index) released = writeRecordedValue(calls[index], wait);
            for (auto index = std::min(from, calls.size()); index > 0 && !released; --index) {
                released = writeRecordedValue(calls[index - 1], wait);
                if (released) ++lateReleases;
            }
            if (!released && options.synthesize && !deferred) {
                if (const auto value = satisfying(wait, load(wait.address, bytes))) {
                    const auto before = load(wait.address, bytes);
                    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(wait.address)), &*value, bytes);
                    if (syntheticReleases.load() < 64) {
                        for (std::size_t other = 0; other < count; ++other) {
                            const auto& waiting = awaited[other];
                            if (waiting.address == 0) continue;
                            std::fprintf(stderr, "[frame-replay]   awaited 0x%llx == 0x%llx (function %u) holds 0x%llx\n", static_cast<unsigned long long>(waiting.address), static_cast<unsigned long long>(waiting.reference), waiting.function, static_cast<unsigned long long>(load(waiting.address, waiting.bytes == 8 ? 8 : 4)));
                        }
                        const auto [accepted, completed] = progress();
                        std::fprintf(stderr, "[frame-replay] synthesized 0x%llx: 0x%llx -> 0x%llx (function %u, reference 0x%llx, mask 0x%llx) at call %zu, serials %llu/%llu\n", static_cast<unsigned long long>(wait.address),
                                     static_cast<unsigned long long>(before), static_cast<unsigned long long>(*value), wait.function, static_cast<unsigned long long>(wait.reference), static_cast<unsigned long long>(wait.mask), from,
                                     static_cast<unsigned long long>(completed), static_cast<unsigned long long>(accepted));
                    }
                    ++syntheticReleases;
                    released = true;
                }
            }
            if (!released) continue;
            ++earlyReleases;
            if (options.exact) return;
        }
    }

    void runWindow(std::span<const FrameCall> calls) {
        std::map<std::uint64_t, std::vector<std::size_t>> byThread;
        for (std::size_t index = 0; index < calls.size(); ++index) byThread[calls[index].thread].push_back(index);
        threads = byThread.size();
        std::mutex mutex;
        std::condition_variable turn;
        std::size_t started = 0;
        std::exception_ptr failure;
        std::atomic<bool> finished{false};
        std::thread breaker([&] {
            if (!options.release) return;
            auto lastCompleted = progress().second;
            auto lastSequenced = sequenced.load();
            auto lastProgress = Clock::now();
            std::optional<Clock::time_point> parkedSince;
            const auto quiet = options.exact ? std::max<std::chrono::milliseconds>(options.releaseAfter, ExactQuiet * 4) : options.releaseAfter;
            while (!finished.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                const auto completed = progress().second;
                const auto issued = sequenced.load();
                if (completed != lastCompleted || issued != lastSequenced || closingWindows.load() || !AgcDriverQuiescent_nid_postfix()) {
                    lastCompleted = completed;
                    lastSequenced = issued;
                    lastProgress = Clock::now();
                    parkedSince.reset();
                    parkedWithoutRelease.store(false);
                    continue;
                }
                if (options.exact) {
                    const auto now = Clock::now();
                    if (busy.load() == 0 && pacing.load() == 0 && blockedOnWaits()) {
                        if (!parkedSince) parkedSince = now;
                        if (now - *parkedSince < ParkedQuiet) continue;
                        const auto before = earlyReleases.load();
                        releaseAwaited(calls, sequenced.load());
                        parkedWithoutRelease.store(earlyReleases.load() == before);
                        ++parkedReleases;
                        parkedSince.reset();
                        lastProgress = Clock::now();
                        continue;
                    }
                    parkedSince.reset();
                    if (now - lastProgress < quiet) continue;
                    ++fallbackReleases;
                    const auto [accepted, completed] = progress();
                    std::fprintf(stderr, "[frame-replay] --exact: no parked state for %lld ms (%d replay threads running, serials %llu/%llu): wall-clock fallback release at call %zu\n", static_cast<long long>(quiet.count()), busy.load(),
                                 static_cast<unsigned long long>(completed), static_cast<unsigned long long>(accepted), sequenced.load());
                    releaseAwaited(calls, sequenced.load());
                    lastProgress = Clock::now();
                    continue;
                }
                if (Clock::now() - lastProgress < quiet) continue;
                releaseAwaited(calls, sequenced.load());
                lastProgress = Clock::now();
            }
        });
        std::vector<std::thread> workers;
        for (const auto& entry : byThread) {
            workers.emplace_back([&, &indices = entry.second] {
                for (const auto index : indices) {
                    std::unique_lock lock(mutex);
                    turn.wait(lock, [&] { return started == index || failure != nullptr; });
                    if (failure != nullptr) return;
                    const auto advance = [&] {
                        ++started;
                        sequenced.store(started);
                        lock.unlock();
                        turn.notify_all();
                    };
                    const auto kind = calls[index].kind;
                    // A frame's closing call too when frame windows are on: the next frame's calls
                    // start after its windows closed.
                    const bool ordered = kind == FrameCallKind::Memory || kind == FrameCallKind::RegisterVideoOutput || kind == FrameCallKind::UnregisterVideoOutput || (closeFrames && kind == boundary);
                    if (!ordered) advance();
                    struct Running {
                        std::atomic<int>& busy;
                        explicit Running(std::atomic<int>& counter) : busy(counter) { ++busy; }
                        ~Running() { --busy; }
                    };
                    try {
                        const Running running{busy};
                        execute(calls[index]);
                        if (ordered) advance();
                    } catch (...) {
                        if (!lock.owns_lock()) lock.lock();
                        if (failure == nullptr) failure = std::current_exception();
                        turn.notify_all();
                        return;
                    }
                }
            });
        }
        for (auto& worker : workers) worker.join();
        if (failure == nullptr) drain();
        finished.store(true);
        breaker.join();
        if (options.exact && (parkedReleases.load() != 0 || fallbackReleases.load() != 0)) std::fprintf(stderr, "[frame-replay] --exact releases: %llu while parked, %llu by the wall-clock fallback\n", static_cast<unsigned long long>(parkedReleases.load()), static_cast<unsigned long long>(fallbackReleases.load()));
        if (failure != nullptr) std::rethrow_exception(failure);
    }

    void drain() {
        auto lastCompleted = progress().second;
        auto lastReleases = earlyReleases.load();
        auto lastProgress = Clock::now();
        const auto timeout = options.exact ? std::max<std::chrono::milliseconds>(options.drainTimeout, std::chrono::seconds(60)) : options.drainTimeout;
        while (true) {
            const auto [accepted, completed] = progress();
            unfinished = accepted - completed;
            if (unfinished == 0) return;
            if (completed != lastCompleted || earlyReleases.load() != lastReleases) {
                lastCompleted = completed;
                lastReleases = earlyReleases.load();
                lastProgress = Clock::now();
            } else if (options.exact && parkedWithoutRelease.load()) {
                std::fprintf(stderr, "[frame-replay] %llu submissions stay parked on waits nothing in the window releases (serials %llu/%llu)\n", static_cast<unsigned long long>(unfinished),
                             static_cast<unsigned long long>(completed), static_cast<unsigned long long>(accepted));
                return;
            } else if (Clock::now() - lastProgress >= timeout) {
                std::fprintf(stderr, "[frame-replay] %llu submissions did not complete after the last captured call (serials %llu/%llu)\n", static_cast<unsigned long long>(unfinished),
                             static_cast<unsigned long long>(completed), static_cast<unsigned long long>(accepted));
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    template <typename TFunction>
    void registration(TFunction&& function) {
        try {
            function();
        } catch (const std::exception& error) {
            if (recording && options.strict) throw;
            ++(recording ? failedRegistrations : staleRegistrations);
            std::fprintf(stderr, "[frame-replay] %s registration failed: %s\n", recording ? "recorded" : "prologue", error.what());
        }
    }

    static const Shader* shader(std::uint64_t address) {
        return reinterpret_cast<const Shader*>(static_cast<std::uintptr_t>(address));
    }

    static AgcDriver::FlipInfo flipInfo(const FrameCall& call) {
        return {static_cast<std::uint32_t>(call.values.at(0)), static_cast<std::int32_t>(call.values.at(1)), static_cast<std::uint32_t>(call.values.at(2)), static_cast<std::int64_t>(call.values.at(3))};
    }
};

int usage() {
    std::fprintf(stderr, "usage: agc_frame_replay <capture directory> [--no-pace] [--no-release] [--release-after-ms <n>] [--strict] [--exact] [--frame-limit <n>] [--timeout-s <n>] [--wall-windows]\n"
                         "                        [--no-bindless-prime] [--bindless-in <file>] [--bindless-out <file>] [--dedupe-prologue]\n"
                         "                        [--dump <address>:<bytes>:<file>]...\n"
                         "  replays the driver calls and guest memory that APS5_FRAME_CAPTURE=<directory> recorded, through this\n"
                         "  build's AGC driver with no title running, and prints one JSON line with the frame times\n"
                         "  --no-pace           issues calls without waiting for the in-flight depth the title had\n"
                         "  --no-release        never copies recorded CPU stores early to release a GPU wait\n"
                         "  --release-after-ms  milliseconds without a completion or a newly issued call before awaited dwords are released (50; at least 200 with --exact)\n"
                         "  --no-synthesize     releases a wait only with a value the capture recorded, never with its reference value\n"
                         "  --strict            fails on a pacing timeout or a failed registration, and exits with 5 unless the JSON gaps list is empty\n"
                         "  --exact             waits for every earlier submission and its GPU work to complete before each submission, flip and\n"
                         "                      memory delta,\n"
                         "                      so the GPU work runs in one order (a reproducible baseline, slower than the title), and releases\n"
                         "                      awaited dwords when every replay thread is parked and the driver is blocked on waits, in capture order,\n"
                         "                      one dword per parked state once no replay thread is still pacing, so queues released past the\n"
                         "                      window's end do not run beside each other; a dword that a submitted packet of the window writes\n"
                         "                      (RELEASE_MEM, EVENT_WRITE_EOP, WRITE_DATA, COPY_DATA, ATOMIC_MEM, DMA_DATA) is synthesized last;\n"
                         "                      synthesized dwords go in the capture order of the submission that waits on them\n"
                         "  --frame-limit <n>   replays the calls through the n-th CPU flip, or the n-th suspend point when the title flips from the GPU,\n"
                         "                      and compares the display buffer with the title's image at that frame\n"
                         "  --checkpoint <csv>  replays up to the checkpoint of a capture made with APS5_FRAME_CAPTURE_CHECKPOINT=<frame> and lists\n"
                         "                      the 64 KiB blocks whose replayed bytes differ from the title's at that point; their replayed bytes go to <csv>.replay.bin\n"
                         "  --timeout-s <n>     exits with code 4 when the replay has no result after n seconds (900; 0 waits forever)\n"
                         "  --dump              writes guest memory to a file after the replay (hexadecimal or decimal)\n"
                         "  --frames <dir>      writes each recorded and replayed display buffer as reference-<n>.bmp and replay-<n>.bmp\n"
                         "  --wall-windows      the profile lines ([gputime], [draws], [mem], [command-mix], [resident-census]) close their windows on\n"
                         "                      the wall clock as in the game; by default each replayed frame is one window, closed at its flip\n"
                         "                      (frame-<n>; the replay waits there for the frame's GPU work), and every line prints once more at\n"
                         "                      the end (replay-end)\n"
                         "  --no-bindless-prime the bindless heap starts empty (default: primed with the tables the capture's BindlessTables record lists)\n"
                         "  --bindless-in <f>   primes the bindless heap with the tables in <f> (a first replay's --bindless-out) before the frames\n"
                         "  --bindless-out <f>  writes the bindless tables registered by the end of the first replayed frame to <f>\n"
                         "  --dedupe-prologue   skips prologue ABI resolutions identical to an earlier one with no registration or memory call between\n"
                         "  the shader cache's Vulkan pipeline cache file is loaded as the title loads it (APS5_PIPELINE_CACHE_LOAD=0 starts empty)\n");
    return 2;
}

Dump parseDump(std::string_view text) {
    const auto first = text.find(':');
    const auto second = first == std::string_view::npos ? std::string_view::npos : text.find(':', first + 1);
    if (second == std::string_view::npos) throw std::invalid_argument("frame replay: --dump takes <address>:<bytes>:<file>");
    const auto number = [](std::string_view part) {
        const std::string value(part);
        char* end = nullptr;
        const auto parsed = std::strtoull(value.c_str(), &end, 0);
        if (value.empty() || end != value.c_str() + value.size()) throw std::invalid_argument("frame replay: '" + value + "' is not a number");
        return static_cast<std::uint64_t>(parsed);
    };
    return {number(text.substr(0, first)), number(text.substr(first + 1, second - first - 1)), std::filesystem::path(std::string(text.substr(second + 1)))};
}

// One write call per `chunk` bytes at most: a single ofstream::write of more than 2 GiB ends in the
// CRT's _write, whose count is 32-bit, and never returns.
constexpr std::uint64_t DumpChunkBytes = 256ull << 20u;

bool writeChunked(std::ostream& file, const std::byte* bytes, std::uint64_t size, std::uint64_t chunk = DumpChunkBytes) {
    for (std::uint64_t at = 0; at < size && file.good(); at += chunk) {
        const auto part = std::min(chunk, size - at);
        file.write(reinterpret_cast<const char*>(bytes + at), static_cast<std::streamsize>(part));
    }
    return file.good();
}

void writeDump(const Dump& dump) {
    auto* const pointer = reinterpret_cast<const std::byte*>(static_cast<std::uintptr_t>(dump.address));
    GuestArena::GuestArenaPrepareHostRead_nid_no_patch(pointer, static_cast<std::size_t>(dump.bytes));
    std::ofstream file(dump.path, std::ios::binary | std::ios::trunc);
    if (!writeChunked(file, pointer, dump.bytes) || !file.flush().good()) throw std::runtime_error("frame replay: cannot write " + dump.path.string());
}

struct FrameCheck {
    std::size_t compared = 0;
    std::size_t missing = 0;
    double identical = 1.0;
    int maxDifference = 0;
};

void writeFrame(const std::filesystem::path& path, const AgcDriver::DisplayBuffer& buffer, std::span<const std::byte> pixels) {
    std::FILE* file = std::fopen(path.string().c_str(), "wb");
    const auto failure = file != nullptr ? AgcDriver::WriteBmpAndClose_nid_no_patch(file, buffer.width, buffer.height, pixels) : AgcDriver::FrameDumpIoFailure{"open", errno};
    if (failure) std::fprintf(stderr, "[frame-replay] cannot write %s (%s failed)\n", path.string().c_str(), failure.operation);
}

FrameCheck compareFrames(const FrameCaptureReader& reader, const Options& options) {
    FrameCheck check;
    if (!options.frames.empty()) std::filesystem::create_directories(options.frames);
    for (const auto& call : reader.Calls()) {
        if (call.kind != FrameCallKind::DisplayBuffer || call.values.size() < 9) continue;
        AgcDriver::DisplayBuffer buffer{};
        buffer.address = call.address;
        buffer.pixelFormat = call.values[1];
        buffer.width = static_cast<std::uint32_t>(call.values[2]);
        buffer.height = static_cast<std::uint32_t>(call.values[3]);
        buffer.tilingMode = static_cast<std::uint32_t>(call.values[4]);
        buffer.pitchInPixel = static_cast<std::uint32_t>(call.values[5]);
        buffer.dccAddress = call.values[6];
        buffer.dccClearColor = call.values[7];
        buffer.alignment = static_cast<std::uint32_t>(call.values[8]);
        const auto chunk = std::find_if(reader.Chunks().begin(), reader.Chunks().end(), [&](const FrameMemoryChunk& candidate) { return candidate.kind == FrameMemoryKind::Reference && candidate.group == call.values[0]; });
        if (chunk == reader.Chunks().end()) {
            ++check.missing;
            continue;
        }
        const auto reference = reader.Read(*chunk);
        const bool perFrame = call.values.size() >= 11;
        if (perFrame && !options.frames.empty()) writeFrame(options.frames / ("game-frame-" + std::to_string(call.values[9]) + ".bmp"), buffer, reference);
        if (perFrame != (options.frameLimit != 0) || (perFrame && call.values[9] != options.frameLimit)) continue;
        const auto replayed = AgcDriver::ReadDisplayBuffer(buffer);
        ++check.compared;
        std::size_t same = 0;
        const auto pixels = std::min(reference.size(), replayed.size()) / 4;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            bool equal = true;
            for (std::size_t channel = 0; channel < 4; ++channel) {
                const auto difference = std::abs(static_cast<int>(reference[pixel * 4 + channel]) - static_cast<int>(replayed[pixel * 4 + channel]));
                check.maxDifference = std::max(check.maxDifference, difference);
                equal = equal && difference == 0;
            }
            same += equal ? 1 : 0;
        }
        const auto total = std::max(reference.size(), replayed.size()) / 4;
        check.identical = std::min(check.identical, total == 0 ? 1.0 : static_cast<double>(same) / static_cast<double>(total));
        if (options.frames.empty()) continue;
        const auto index = std::to_string(call.values[0]);
        writeFrame(options.frames / ("reference-" + index + ".bmp"), buffer, reference);
        writeFrame(options.frames / ("replay-" + index + ".bmp"), buffer, replayed);
    }
    return check;
}

void compareCheckpoint(const FrameCaptureReader& reader, const std::filesystem::path& path, const Skipped& skipped, bool drained = true) {
    if (drained) {
        AgcDriverWaitIdle_nid_postfix();
        if (!AgcDriverSettle_nid_postfix()) std::fprintf(stderr, "[frame-replay] storage images were still pending at the checkpoint\n");
    } else {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        auto quietSince = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() < deadline) {
            if (!AgcDriverQuiescent_nid_postfix()) quietSince = std::chrono::steady_clock::now();
            else if (std::chrono::steady_clock::now() - quietSince >= std::chrono::milliseconds(50)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::fprintf(stderr, "[frame-replay] checkpoint compared with submissions still parked on waits whose writers lie beyond the window (driver %s)\n", AgcDriverQuiescent_nid_postfix() ? "quiescent" : "still running after 60 s");
    }
    std::ofstream report(path, std::ios::trunc);
    auto blocksPath = path;
    blocksPath += ".replay.bin";
    std::ofstream replayed(blocksPath, std::ios::binary | std::ios::trunc);
    report << "address,bytes,differing_bytes,first_difference\n";
    std::uint64_t blocks = 0, differing = 0, differingBytes = 0;
    for (const auto& chunk : reader.Chunks()) {
        if (chunk.kind != FrameMemoryKind::Checkpoint || skipped.Overlaps(chunk)) continue;
        ++blocks;
        const auto expected = reader.Read(chunk);
        auto* const pointer = reinterpret_cast<const std::byte*>(static_cast<std::uintptr_t>(chunk.address));
        GuestArena::GuestArenaPrepareHostRead_nid_no_patch(pointer, static_cast<std::size_t>(chunk.bytes));
        std::uint64_t count = 0;
        std::int64_t first = -1;
        for (std::size_t at = 0; at < expected.size(); ++at) {
            if (pointer[at] == expected[at]) continue;
            if (first < 0) first = static_cast<std::int64_t>(at);
            ++count;
        }
        if (count == 0) continue;
        ++differing;
        differingBytes += count;
        const std::array<std::uint64_t, 2> head{chunk.address, chunk.bytes};
        replayed.write(reinterpret_cast<const char*>(head.data()), sizeof(head));
        replayed.write(reinterpret_cast<const char*>(pointer), static_cast<std::streamsize>(chunk.bytes));
        char line[96];
        std::snprintf(line, sizeof(line), "0x%llx,%llu,%llu,%lld\n", static_cast<unsigned long long>(chunk.address), static_cast<unsigned long long>(chunk.bytes), static_cast<unsigned long long>(count), static_cast<long long>(first));
        report << line;
    }
    std::fprintf(stderr, "[frame-replay] checkpoint: %llu of %llu changed blocks differ from the title (%.1f MiB of bytes), listed in %s\n", static_cast<unsigned long long>(differing), static_cast<unsigned long long>(blocks), static_cast<double>(differingBytes) / (1024.0 * 1024.0), path.string().c_str());
}

int run(const Options& options, const FrameCaptureReader& reader, Clock::time_point loadStart) {
    if (reader.Truncated()) std::fprintf(stderr, "[frame-replay] the capture is truncated; replaying the complete records\n");
    auto skipped = restoreMemory(reader);
    const auto loaded = Clock::now();
    Replay replay{options, reader};
    replay.skipped = std::move(skipped);
    if (const auto at = reader.Header().find("\"gds_address\":"); at != std::string::npos) AgcDriverUseGdsAt_nid_postfix(std::strtoull(reader.Header().c_str() + at + 14, nullptr, 10));
    for (const auto& chunk : reader.Chunks()) {
        if (chunk.kind == FrameMemoryKind::Delta) replay.groups[chunk.group].push_back(&chunk);
    }
    const auto& calls = reader.Calls();
    const auto frameStart = std::find_if(calls.begin(), calls.end(), [](const FrameCall& call) { return call.kind == FrameCallKind::FrameStart; });
    std::vector<const FrameMemoryChunk*> rewritten;
    const auto prologueStart = Clock::now();
    PrologueDedupe dedupe;
    for (auto call = calls.begin(); call != frameStart; ++call) {
        if (call->kind == FrameCallKind::Memory) {
            if (const auto found = replay.groups.find(call->values.at(0)); found != replay.groups.end()) rewritten.insert(rewritten.end(), found->second.begin(), found->second.end());
        }
        if (options.dedupePrologue && dedupe.Redundant(*call)) continue;
        replay.execute(*call);
    }
    if (!rewritten.empty()) restoreRewritten(reader, rewritten, replay.skipped);
    replay.primeBindless(replay.deferredBindless);
    if (!options.bindlessIn.empty()) {
        std::ifstream file(options.bindlessIn, std::ios::binary);
        std::vector<std::uint64_t> words;
        for (std::uint64_t word = 0; file.read(reinterpret_cast<char*>(&word), sizeof(word));) words.push_back(word);
        if (words.empty()) throw std::runtime_error("frame replay: no bindless tables in " + options.bindlessIn.string());
        replay.primeBindless(words);
    }
    if (replay.bindlessPrimed != 0) std::fprintf(stderr, "[frame-replay] bindless heap primed with %zu tables\n", replay.bindlessPrimed);
    const auto prologueSeconds = std::chrono::duration<double>(Clock::now() - prologueStart).count();
    const auto prologueSkipped = dedupe.Skipped();
    std::fprintf(stderr, "[frame-replay] prologue: %zu calls (%llu redundant resolutions skipped) in %.1f s, %zu shader images rewritten and restored, %llu stale registrations\n", static_cast<std::size_t>(frameStart - calls.begin()), static_cast<unsigned long long>(dedupe.Skipped()),
                 std::chrono::duration<double>(Clock::now() - prologueStart).count(), rewritten.size(), static_cast<unsigned long long>(replay.staleRegistrations.load()));
    if (options.frameWindows) AgcDriver::ProfileWindows::Close("prologue");
    replay.recording = true;
    replay.startedAt = Clock::now();
    if (frameStart != calls.end()) {
        auto window = std::span(calls).subspan(static_cast<std::size_t>(frameStart - calls.begin()) + 1);
        replay.boundary = std::any_of(window.begin(), window.end(), [](const FrameCall& call) { return call.kind == FrameCallKind::Flip; }) ? FrameCallKind::Flip : FrameCallKind::Suspend;
        replay.closeFrames = options.frameWindows;
        if (!options.checkpointReport.empty()) {
            const auto cut = std::find_if(window.begin(), window.end(), [](const FrameCall& call) { return call.kind == FrameCallKind::Checkpoint; });
            if (cut == window.end()) throw std::runtime_error("frame replay: --checkpoint needs a capture made with APS5_FRAME_CAPTURE_CHECKPOINT");
            window = window.first(static_cast<std::size_t>(cut - window.begin()));
        } else if (options.frameLimit != 0) {
            const bool cpuFlips = std::any_of(window.begin(), window.end(), [](const FrameCall& call) { return call.kind == FrameCallKind::Flip; });
            const auto boundary = cpuFlips ? FrameCallKind::Flip : FrameCallKind::Suspend;
            std::size_t boundaries = 0;
            const auto cut = std::find_if(window.begin(), window.end(), [&](const FrameCall& call) { return call.kind == boundary && ++boundaries == options.frameLimit; });
            if (cut != window.end()) window = window.first(static_cast<std::size_t>(cut - window.begin()) + 1);
        }
        replay.runWindow(window);
    }
    if (frameStart == calls.end()) AgcDriverWaitIdle_nid_postfix();
    if (!options.checkpointReport.empty()) compareCheckpoint(reader, options.checkpointReport, replay.skipped, replay.unfinished == 0);
    replay.endedAt = Clock::now();
    if (const auto failure = replay.frames.Failure()) std::rethrow_exception(failure);
    FrameCheck check;
    if (replay.unfinished == 0) {
        if (!AgcDriverSettle_nid_postfix()) std::fprintf(stderr, "[frame-replay] storage images were still pending before the comparison\n");
        for (const auto& dump : options.dumps) writeDump(dump);
        check = compareFrames(reader, options);
    } else if (!options.dumps.empty()) {
        std::fprintf(stderr, "[frame-replay] no dumps written: the GPU work did not finish\n");
    }
    if (check.compared == 0) check.identical = 0.0;
    // The final flush: what the last windows hold (a frame cut by --checkpoint, the settle).
    AgcDriver::ProfileWindows::Close("replay-end");
    replay.writeBindless();
    std::string gaps;
    const auto gap = [&](const char* name, bool present) {
        if (present) gaps += (gaps.empty() ? "" : ",") + std::string(name);
    };
    gap("truncated", reader.Truncated());
    gap("capture-warnings", reader.Header().find("\"warnings\":[]") == std::string::npos);
    gap("undrained", replay.unfinished != 0);
    gap("skipped-memory", replay.skipped.bytes != 0);
    gap("stale-registrations", replay.staleRegistrations.load() != 0);
    gap("failed-registrations", replay.failedRegistrations.load() != 0);
    gap("pacing-timeouts", replay.paceTimeouts.load() != 0 || replay.memoryPaceTimeouts.load() != 0);
    gap("synthesized-waits", replay.syntheticReleases.load() != 0);
    gap("partial-window", options.frameLimit != 0 && check.compared == 0);
    gap("no-reference-frames", check.compared == 0 && options.dumps.empty());
    gap("missing-reference-frames", check.missing != 0);
    gap("pixels-differ", check.compared != 0 && check.identical < 1.0);
    const auto ready = replay.frames.Times();
    std::vector<double> frameMs;
    for (std::size_t index = 1; index < ready.size(); ++index) frameMs.push_back(std::chrono::duration<double, std::milli>(ready[index] - ready[index - 1]).count());
    std::string frameList;
    for (const auto value : frameMs) frameList += (frameList.empty() ? "" : ",") + std::to_string(static_cast<long long>(value));
    std::sort(frameMs.begin(), frameMs.end());
    const auto percentile = [&](double fraction) { return frameMs.empty() ? 0.0 : frameMs[std::min(frameMs.size() - 1, static_cast<std::size_t>(fraction * static_cast<double>(frameMs.size())))]; };
    const auto seconds = std::chrono::duration<double>(replay.endedAt - replay.startedAt).count();
    std::printf("{\"frames\":%zu,\"replay_s\":%.3f,\"fps\":%.3f,\"frame_ms\":{\"p50\":%.3f,\"p90\":%.3f,\"max\":%.3f,\"each\":[%s]},\"submits\":%llu,\"delta_mib\":%.1f,\"pace_timeouts\":%llu,\"memory_pace_timeouts\":%llu,\"pace_wait_s\":%.3f,\"stale_registrations\":%llu,\"failed_registrations\":%llu,\"skipped_mib\":%.1f,\"threads\":%zu,\"released_waits\":%llu,\"of_them_from_earlier_writes\":%llu,\"of_them_synthesized\":%llu,\"unfinished\":%llu,\"frames_compared\":%zu,\"pixels_identical\":%.6f,\"max_channel_diff\":%d,\"load_s\":%.3f,\"prologue_s\":%.3f,\"prologue_skipped\":%llu,\"frame_windows\":%zu,\"undrained_windows\":%zu,\"bindless_primed\":%zu,\"gaps\":\"%s\"}\n",
                ready.size(), seconds, seconds > 0 ? static_cast<double>(ready.size()) / seconds : 0.0, percentile(0.5), percentile(0.9), frameMs.empty() ? 0.0 : frameMs.back(), frameList.c_str(),
                static_cast<unsigned long long>(replay.submits.load()), static_cast<double>(replay.deltaBytes.load()) / (1024.0 * 1024.0), static_cast<unsigned long long>(replay.paceTimeouts.load()), static_cast<unsigned long long>(replay.memoryPaceTimeouts.load()),
                replay.paceWaitedSeconds(), static_cast<unsigned long long>(replay.staleRegistrations.load()), static_cast<unsigned long long>(replay.failedRegistrations.load()),
                static_cast<double>(replay.skipped.bytes) / (1024.0 * 1024.0), replay.threads, static_cast<unsigned long long>(replay.earlyReleases.load()), static_cast<unsigned long long>(replay.lateReleases.load()),
                static_cast<unsigned long long>(replay.syntheticReleases.load()),
                static_cast<unsigned long long>(replay.unfinished), check.compared, check.identical, check.maxDifference, std::chrono::duration<double>(loaded - loadStart).count(), prologueSeconds, static_cast<unsigned long long>(prologueSkipped), replay.framesClosed.load(), replay.undrainedWindows.load(), replay.bindlessPrimed, gaps.c_str());
    std::fflush(stdout);
    if (options.strict && !gaps.empty()) {
        std::fprintf(stderr, "agc_frame_replay: --strict rejects this replay: %s\n", gaps.c_str());
        return 5;
    }
    return replay.unfinished == 0 ? 0 : 3;
}

}

int main(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--no-pace") options.pace = false;
        else if (argument == "--no-release") options.release = false;
        else if (argument == "--no-synthesize") options.synthesize = false;
        else if (argument == "--release-after-ms" && index + 1 < argc) options.releaseAfter = std::chrono::milliseconds(std::strtoll(argv[++index], nullptr, 10));
        else if (argument == "--strict") options.strict = true;
        else if (argument == "--exact") options.exact = true;
        else if (argument == "--checkpoint" && index + 1 < argc) options.checkpointReport = argv[++index];
        else if (argument == "--frame-limit" && index + 1 < argc) options.frameLimit = static_cast<std::size_t>(std::strtoull(argv[++index], nullptr, 10));
        else if (argument == "--timeout-s" && index + 1 < argc) options.timeout = std::chrono::seconds(std::strtoll(argv[++index], nullptr, 10));
        else if (argument == "--dump" && index + 1 < argc) options.dumps.push_back(parseDump(argv[++index]));
        else if (argument == "--frames" && index + 1 < argc) options.frames = argv[++index];
        else if (argument == "--wall-windows") options.frameWindows = false;
        else if (argument == "--host-regions") options.hostRegions = true;
        else if (argument == "--dedupe-prologue") options.dedupePrologue = true;
        else if (argument == "--no-bindless-prime") options.bindlessPrime = false;
        else if (argument == "--bindless-in" && index + 1 < argc) options.bindlessIn = argv[++index];
        else if (argument == "--bindless-out" && index + 1 < argc) options.bindlessOut = argv[++index];
        else if (!argument.starts_with("-") && options.directory.empty()) options.directory = argument;
        else return usage();
    }
#ifdef _WIN32
    if (options.hostRegions) return PrintHostRegions();
#endif
    if (options.directory.empty()) return usage();
#ifdef _WIN32
    _putenv_s("APS5_FRAME_CAPTURE", "");
    if (std::getenv("APS5_PIPELINE_CACHE_LOAD") == nullptr) _putenv_s("APS5_PIPELINE_CACHE_LOAD", "1");
#else
    unsetenv("APS5_FRAME_CAPTURE");
    setenv("APS5_PIPELINE_CACHE_LOAD", "1", 0);
#endif
    int code = 2;
    try {
        const auto loadStart = Clock::now();
        const FrameCaptureReader reader(options.directory);
        // A captured range on one of this process's host regions cannot be mapped: start again (a new process gets a
        // new random layout), at most 4 times. Before the watchdog, so a waiting parent is never killed by it.
        auto collisions = HostCollisions(reader, [](std::uint64_t address, std::uint64_t bytes) { return GuestArena::GuestArenaHostRegionOverlaps_nid_postfix(static_cast<std::uintptr_t>(address), static_cast<std::size_t>(bytes)); });
        const char* attemptText = std::getenv("APS5_REPLAY_ATTEMPT");
        const int attempt = attemptText != nullptr ? std::atoi(attemptText) : 1;
        // APS5_REPLAY_TEST_COLLISIONS=<n> (tests): starts 1..n report the first captured range as a collision.
        if (const char* test = std::getenv("APS5_REPLAY_TEST_COLLISIONS"); test != nullptr && collisions.empty() && attempt <= std::atoi(test)) {
            for (const auto& chunk : reader.Chunks()) {
                if (chunk.kind == FrameMemoryKind::Range && chunk.rangeKind != FrameRangeKind::Host) {
                    collisions.emplace_back(chunk.address, chunk.address + chunk.bytes);
                    break;
                }
            }
        }
        if (!collisions.empty()) {
            for (const auto& [begin, end] : collisions) {
                std::fprintf(stderr, "[frame-replay] start %d: the captured range 0x%llx+0x%llx lies on a host region of this process", attempt, static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end - begin));
#ifdef _WIN32
                for (auto at = begin; at < end;) {
                    MEMORY_BASIC_INFORMATION info{};
                    if (VirtualQuery(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), &info, sizeof(info)) == 0) break;
                    const auto next = reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize;
                    if (info.State != MEM_FREE && GuestArena::GuestArenaHostRegionOverlaps_nid_postfix(static_cast<std::uintptr_t>(at), static_cast<std::size_t>(std::min<std::uint64_t>(next, end) - at))) std::fprintf(stderr, "; %s", DescribeHostRegion(at).c_str());
                    at = next;
                }
#endif
                std::fprintf(stderr, "\n");
            }
#ifdef _WIN32
            if (attempt < 4) {
                const auto next = std::to_string(attempt + 1);
                _putenv_s("APS5_REPLAY_ATTEMPT", next.c_str());
                std::fprintf(stderr, "[frame-replay] starting again (start %s of at most 4)\n", next.c_str());
                std::fflush(stderr);
                STARTUPINFOW startup{};
                startup.cb = sizeof(startup);
                PROCESS_INFORMATION process{};
                std::wstring line = GetCommandLineW();
                if (!CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &process)) throw std::runtime_error("frame replay: cannot start again after a host-region collision");
                WaitForSingleObject(process.hThread, 0);
                WaitForSingleObject(process.hProcess, INFINITE);
                DWORD exitCode = 2;
                GetExitCodeProcess(process.hProcess, &exitCode);
                CloseHandle(process.hThread);
                CloseHandle(process.hProcess);
                hardExit(static_cast<int>(exitCode));
            }
#endif
            throw std::runtime_error("frame replay: captured ranges lie on host regions of this process in every start");
        }
        startWatchdog(options.timeout);
        AgcDriver::ProfileWindows::UseFrameWindows(options.frameWindows);
        code = run(options, reader, loadStart);
        if (code == 0) {
            AgcDriverShutdown_nid_postfix();
            return 0;
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "agc_frame_replay: %s\n", error.what());
    }
    hardExit(code);
}
