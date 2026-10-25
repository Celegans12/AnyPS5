#include "prx/libSceAgcDriver/Graphics/include/ResidentShadows.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

namespace AgcDriver::Graphics::ResidentShadows {

namespace {

using Key = std::tuple<std::uint64_t, std::uint64_t, bool>;

struct Registry {
    AgcDriver::Mutex mutex;
    std::map<Key, EntryPtr> entries;
    // Live entries by begin (several may start at one address with other ends or usages), for the
    // overlap scan of NoteWrite; `longest` bounds how far before a note an overlapping entry starts.
    std::multimap<std::uint64_t, Entry*> byBegin;
    std::uint64_t longest = 0;
    std::uint64_t bytes = 0;
    std::uint64_t tick = 0;
    std::atomic<std::size_t> live{0};
};

Registry& registry() {
    static auto* state = new Registry;
    return *state;
}

std::uint64_t budgetBytes() {
    static const std::uint64_t bytes = [] {
        const char* value = std::getenv("APS5_RESIDENT_MIB");
        return (value != nullptr ? std::strtoull(value, nullptr, 10) : 1024ull) << 20u;
    }();
    return bytes;
}

thread_local const void* ownNoter = nullptr;

void unlink(Registry& state, Entry& entry) {
    if (!entry.live) return;
    entry.live = false;
    for (auto [it, last] = state.byBegin.equal_range(entry.begin); it != last; ++it) {
        if (it->second == &entry) {
            state.byBegin.erase(it);
            break;
        }
    }
    state.bytes -= entry.end - entry.begin;
    state.live.fetch_sub(1, std::memory_order_relaxed);
}

void eraseLocked(Registry& state, const EntryPtr& entry) {
    if (entry == nullptr || !entry->live) return;
    unlink(state, *entry);
    const auto found = state.entries.find(Key{entry->begin, entry->end, entry->addressable});
    if (found != state.entries.end() && found->second == entry) state.entries.erase(found);
}

// Least recently used idle entries go until `needed` more bytes fit (an entry with an open use
// stays: its build owes the use's end).
bool makeRoom(Registry& state, std::uint64_t needed) {
    const auto budget = budgetBytes();
    if (needed > budget) return false;
    while (state.bytes + needed > budget) {
        EntryPtr oldest;
        for (const auto& [key, entry] : state.entries) {
            if (entry->user != nullptr) continue;
            if (oldest == nullptr || entry->lastUse < oldest->lastUse) oldest = entry;
        }
        if (oldest == nullptr) return false;
        eraseLocked(state, oldest);
        Counters().evictions.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

}

bool Enabled() {
    return budgetBytes() != 0;
}

std::unique_lock<AgcDriver::Mutex> Lock() {
    return std::unique_lock<AgcDriver::Mutex>(registry().mutex);
}

void Touch(Entry& entry) {
    entry.lastUse = ++registry().tick;
}

EntryPtr Find(std::uint64_t begin, std::uint64_t end, bool addressable) {
    auto& state = registry();
    if (state.live.load(std::memory_order_relaxed) == 0) return nullptr;
    std::lock_guard lock(state.mutex);
    const auto found = state.entries.find(Key{begin, end, addressable});
    return found != state.entries.end() && found->second->live ? found->second : nullptr;
}

EntryPtr Register(std::uint64_t begin, std::uint64_t end, bool addressable, std::shared_ptr<Buffer> shadow, std::uint64_t importSerial) {
    if (!Enabled() || shadow == nullptr || end <= begin) return nullptr;
    auto& state = registry();
    std::lock_guard lock(state.mutex);
    if (const auto found = state.entries.find(Key{begin, end, addressable}); found != state.entries.end()) eraseLocked(state, found->second);
    if (!makeRoom(state, end - begin)) {
        Counters().privateBudget.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    auto entry = std::make_shared<Entry>();
    entry->begin = begin;
    entry->end = end;
    entry->addressable = addressable;
    entry->shadow = std::move(shadow);
    entry->importSerial = importSerial;
    entry->lastUse = ++state.tick;
    state.entries.emplace(Key{begin, end, addressable}, entry);
    state.byBegin.emplace(begin, entry.get());
    state.longest = std::max(state.longest, end - begin);
    state.bytes += end - begin;
    state.live.fetch_add(1, std::memory_order_relaxed);
    return entry;
}

void Drop(const EntryPtr& entry) {
    auto& state = registry();
    std::lock_guard lock(state.mutex);
    eraseLocked(state, entry);
}

void Clear() {
    auto& state = registry();
    std::map<Key, EntryPtr> dropped;
    {
        std::lock_guard lock(state.mutex);
        for (auto& [key, entry] : state.entries) unlink(state, *entry);
        dropped.swap(state.entries);
        state.byBegin.clear();
        state.longest = 0;
    }
    // The registry's references go outside its lock (a last reference destroys a buffer).
    for (auto& [key, entry] : dropped) entry->shadow.reset();
}

void NoteWrite(std::uint64_t begin, std::uint64_t end) {
    auto& state = registry();
    if (end <= begin || state.live.load(std::memory_order_relaxed) == 0) return;
    std::lock_guard lock(state.mutex);
    const auto from = begin > state.longest ? begin - state.longest : 0;
    for (auto it = state.byBegin.lower_bound(from); it != state.byBegin.end() && it->first < end; ++it) {
        auto& entry = *it->second;
        if (entry.end <= begin || (entry.user != nullptr && entry.user == ownNoter)) continue;
        if (!entry.foreignNote) Counters().foreignNotes.fetch_add(1, std::memory_order_relaxed);
        entry.foreignNote = true;
    }
}

OwnNotes::OwnNotes(const void* owner) : previous(ownNoter) {
    ownNoter = owner;
}

OwnNotes::~OwnNotes() {
    ownNoter = previous;
}

Stats& Counters() {
    static Stats stats;
    return stats;
}

std::pair<std::uint64_t, std::uint64_t> Held() {
    auto& state = registry();
    std::lock_guard lock(state.mutex);
    return {state.live.load(std::memory_order_relaxed), state.bytes};
}

namespace {

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}

bool VerifyThisUse() {
    static const std::uint64_t limit = [] {
        const char* value = std::getenv("APS5_VERIFY_RESIDENT");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
    }();
    if (limit == 0) return false;
    static std::atomic<std::int64_t> windowStart{0};
    static std::atomic<std::uint64_t> taken{0};
    const auto now = nowMs();
    auto start = windowStart.load(std::memory_order_relaxed);
    if (now - start >= 10000 && windowStart.compare_exchange_strong(start, now)) taken.store(0, std::memory_order_relaxed);
    return taken.fetch_add(1, std::memory_order_relaxed) < limit;
}

void Report() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    static std::atomic<std::int64_t> last{0};
    const auto now = nowMs();
    auto previous = last.load();
    if (now - previous < 10000 || !last.compare_exchange_strong(previous, now)) return;
    auto& stats = Counters();
    // One reporter per period (the exchange above), so the previous totals need no lock.
    static std::uint64_t seen[24]{};
    const auto delta = [](std::atomic<std::uint64_t>& counter, std::uint64_t& before) {
        const auto value = counter.load(std::memory_order_relaxed);
        const auto change = value - before;
        before = value;
        return static_cast<unsigned long long>(change);
    };
    const auto uses = delta(stats.uses, seen[0]);
    const auto skipped = delta(stats.skipped, seen[1]);
    const auto partial = delta(stats.partial, seen[2]);
    const auto full = delta(stats.full, seen[3]);
    const auto saved = delta(stats.savedBytes, seen[4]);
    const auto copied = delta(stats.copiedBytes, seen[5]);
    unsigned long long by[static_cast<std::size_t>(Full::Count)];
    for (std::size_t reason = 0; reason < static_cast<std::size_t>(Full::Count); ++reason) by[reason] = delta(stats.fullBy[reason], seen[6 + reason]);
    static std::uint64_t more[14]{};
    const auto busy = delta(stats.privateBusy, more[0]);
    const auto budget = delta(stats.privateBudget, more[1]);
    const auto lane = delta(stats.privateLane, more[2]);
    const auto covered = delta(stats.privateCovered, more[3]);
    const auto refused = delta(stats.reuseRefused, more[4]);
    const auto takeovers = delta(stats.takeovers, more[5]);
    const auto closed = delta(stats.runsClosed, more[6]);
    const auto clean = delta(stats.endsClean, more[7]);
    const auto foreign = delta(stats.endsForeign, more[8]);
    const auto notes = delta(stats.foreignNotes, more[9]);
    const auto evicted = delta(stats.evictions, more[10]);
    const auto verified = delta(stats.verified, more[11]);
    const auto differs = delta(stats.verifyDiffers, more[12]);
    const auto [entries, bytes] = Held();
    AgcDriver::ProfilePrint_nid_no_patch("[resident] copy-in (APS5_NO_RESIDENT_COPY_IN %s), last 10 s: %llu staged uses on resident shadows: %llu skipped, %llu partial, %llu full (new %llu, switched off %llu, unclean %llu, import %llu, mapping %llu, foreign note %llu, lane %llu, lease %llu, unwatched %llu); %.0f KiB copied in, %.0f KiB saved; private shadows: busy %llu, budget %llu, lane %llu, not covered by writes %llu; reuses refused %llu, takeovers %llu, runs closed %llu; uses ended clean %llu, foreign %llu; foreign notes %llu; %llu entries (%.1f MiB of %llu MiB), %llu evicted; verify %llu checked, %llu differ\n",
        RuntimeSwitches::Set(RuntimeSwitch::NoResidentCopyIn) ? "set" : "unset", uses, skipped, partial, full, by[0], by[1], by[2], by[3], by[4], by[5], by[6], by[7], by[8], copied / 1024.0, saved / 1024.0,
        busy, budget, lane, covered, refused, takeovers, closed, clean, foreign, notes, static_cast<unsigned long long>(entries), bytes / 1048576.0, static_cast<unsigned long long>(budgetBytes() >> 20u), evicted, verified, differs);
}

}
