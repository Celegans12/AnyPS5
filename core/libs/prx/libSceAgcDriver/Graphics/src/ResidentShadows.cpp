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
    std::atomic<std::int64_t> lastSweep{0};
    bool copyInOff = false;
    bool pristineOff = false;
};

std::int64_t steadyMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::int64_t idleMs() {
    static const std::int64_t ms = [] {
        const char* value = std::getenv("APS5_RESIDENT_IDLE_MS");
        return value != nullptr ? static_cast<std::int64_t>(std::strtoll(value, nullptr, 10)) : std::int64_t{10000};
    }();
    return ms;
}

Registry& registry() {
    static auto* state = new Registry;
    return *state;
}

std::uint64_t budgetBytes() {
    static const std::uint64_t bytes = [] {
        const char* value = std::getenv("APS5_RESIDENT_MIB");
        return (value != nullptr ? std::strtoull(value, nullptr, 10) : 256ull) << 20u;
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
    state.bytes -= entry.end - entry.begin + entry.pristineBytes;
    entry.pristineBytes = 0;
    state.live.fetch_sub(1, std::memory_order_relaxed);
}

void eraseLocked(Registry& state, const EntryPtr& entry) {
    if (entry == nullptr || !entry->live) return;
    unlink(state, *entry);
    const auto found = state.entries.find(Key{entry->begin, entry->end, entry->addressable});
    if (found != state.entries.end() && found->second == entry) state.entries.erase(found);
}

// Least recently used idle entries go until `needed` more bytes fit, down to 7/8 of the budget in
// one pass so a full registry does not scan once per new entry (an entry with an open use stays:
// its build owes the use's end). The registry's references go to `dropped`, released by the caller
// outside the lock.
bool makeRoom(Registry& state, std::uint64_t needed, std::vector<EntryPtr>& dropped) {
    const auto budget = budgetBytes();
    if (needed > budget) return false;
    if (state.bytes + needed <= budget) return true;
    std::vector<EntryPtr> idle;
    for (const auto& [key, entry] : state.entries) {
        if (entry->user == nullptr) idle.push_back(entry);
    }
    // Entries no build holds any more go first (their shadows are the registry's VRAM alone), each
    // group least recently used first.
    const auto orphan = [](const EntryPtr& entry) { return entry->shadow == nullptr || entry->shadow.use_count() == 1; };
    std::sort(idle.begin(), idle.end(), [&](const EntryPtr& left, const EntryPtr& right) {
        const bool leftOrphan = orphan(left);
        const bool rightOrphan = orphan(right);
        if (leftOrphan != rightOrphan) return leftOrphan;
        return left->lastUse < right->lastUse;
    });
    const auto target = std::min(budget - needed, budget - budget / 8);
    for (const auto& entry : idle) {
        if (state.bytes <= target) break;
        eraseLocked(state, entry);
        dropped.push_back(entry);
        Counters().evictions.fetch_add(1, std::memory_order_relaxed);
    }
    return state.bytes + needed <= budget;
}

// Outside the registry lock: an evicted entry's buffers go with the last build holding them (every
// recorded use kept them in its batch).
void release(std::vector<EntryPtr>& dropped) {
    for (auto& entry : dropped) {
        if (entry->user != nullptr) continue;
        entry->shadow.reset();
        entry->pristine.reset();
    }
    dropped.clear();
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
    entry.lastUseMs = steadyMs();
}

bool ChargePristine(Entry& entry, std::uint64_t bytes) {
    auto& state = registry();
    std::vector<EntryPtr> dropped;
    bool charged = false;
    {
        std::lock_guard lock(state.mutex);
        if (entry.live && entry.pristineBytes == 0 && makeRoom(state, bytes, dropped)) {
            entry.pristineBytes = bytes;
            state.bytes += bytes;
            charged = true;
        }
    }
    release(dropped);
    if (!charged) Counters().pristineRefused.fetch_add(1, std::memory_order_relaxed);
    return charged;
}

void Sweep() {
    auto& state = registry();
    if (state.live.load(std::memory_order_relaxed) == 0) return;
    const auto now = steadyMs();
    auto last = state.lastSweep.load(std::memory_order_relaxed);
    if (now - last < 1000 || !state.lastSweep.compare_exchange_strong(last, now)) return;
    std::vector<EntryPtr> dropped;
    {
        std::lock_guard lock(state.mutex);
        for (auto it = state.entries.begin(); it != state.entries.end();) {
            auto& entry = it->second;
            if (entry->user != nullptr || now - entry->lastUseMs <= idleMs()) {
                ++it;
                continue;
            }
            dropped.push_back(entry);
            unlink(state, *entry);
            it = state.entries.erase(it);
            Counters().idleEvictions.fetch_add(1, std::memory_order_relaxed);
        }
    }
    release(dropped);
}

void Switches(bool copyInOff, bool pristineOff) {
    auto& state = registry();
    std::vector<EntryPtr> dropped;
    std::vector<std::shared_ptr<Buffer>> pristines;
    {
        std::lock_guard lock(state.mutex);
        if (copyInOff && !state.copyInOff) {
            for (auto& [key, entry] : state.entries) {
                unlink(state, *entry);
                dropped.push_back(entry);
            }
            state.entries.clear();
            state.byBegin.clear();
            state.longest = 0;
        }
        if (pristineOff && !state.pristineOff) {
            for (auto& [key, entry] : state.entries) {
                if (entry->user != nullptr || entry->pristine == nullptr) continue;
                pristines.push_back(std::move(entry->pristine));
                state.bytes -= entry->pristineBytes;
                entry->pristineBytes = 0;
                entry->pristineSame = false;
            }
        }
        state.copyInOff = copyInOff;
        state.pristineOff = pristineOff;
    }
    release(dropped);
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
    std::vector<EntryPtr> dropped;
    struct Released {
        std::vector<EntryPtr>& dropped;
        ~Released() { release(dropped); }
    } released{dropped};
    std::lock_guard lock(state.mutex);
    if (const auto found = state.entries.find(Key{begin, end, addressable}); found != state.entries.end()) {
        dropped.push_back(found->second);
        eraseLocked(state, found->second);
    }
    if (!makeRoom(state, end - begin, dropped)) {
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
    entry->lastUseMs = steadyMs();
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
    for (auto& [key, entry] : dropped) {
        entry->shadow.reset();
        entry->pristine.reset();
    }
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
    static std::uint64_t more[20]{};
    const auto idle = delta(stats.idleEvictions, more[17]);
    const auto pristineRefused = delta(stats.pristineRefused, more[18]);
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
    const auto pristineUses = delta(stats.pristineUses, more[13]);
    static std::uint64_t pristine[4]{};
    const auto pristineCopied = delta(stats.pristineCopiedBytes, pristine[0]);
    const auto pristineSaved = delta(stats.pristineSavedBytes, pristine[1]);
    const auto pristineKept = delta(stats.pristineKept, pristine[2]);
    const auto pristineLost = delta(stats.pristineLost, pristine[3]);
    const auto [entries, bytes] = Held();
    AgcDriver::ProfilePrint_nid_no_patch("[resident] copy-in (APS5_NO_RESIDENT_COPY_IN %s), last 10 s: %llu staged uses on resident shadows: %llu skipped, %llu partial, %llu full (new %llu, switched off %llu, unclean %llu, import %llu, mapping %llu, foreign note %llu, lane %llu, lease %llu, unwatched %llu); %.0f KiB copied in, %.0f KiB saved; private shadows: busy %llu, budget %llu, lane %llu, not covered by writes %llu; reuses refused %llu, takeovers %llu, runs closed %llu; uses ended clean %llu, foreign %llu; foreign notes %llu; %llu entries (%.1f MiB of %llu MiB with pristine copies), %llu evicted for room, %llu idle; verify %llu checked, %llu differ; pristine copies (APS5_NO_RESIDENT_PRISTINE %s): %llu uses, %.0f KiB copied, %.0f KiB saved, %llu kept by their copy-backs, %llu not, %llu over the budget\n",
        RuntimeSwitches::Set(RuntimeSwitch::NoResidentCopyIn) ? "set" : "unset", uses, skipped, partial, full, by[0], by[1], by[2], by[3], by[4], by[5], by[6], by[7], by[8], copied / 1024.0, saved / 1024.0,
        busy, budget, lane, covered, refused, takeovers, closed, clean, foreign, notes, static_cast<unsigned long long>(entries), bytes / 1048576.0, static_cast<unsigned long long>(budgetBytes() >> 20u), evicted, idle, verified, differs,
        RuntimeSwitches::Set(RuntimeSwitch::NoResidentPristine) ? "set" : "unset", pristineUses, pristineCopied / 1024.0, pristineSaved / 1024.0, pristineKept, pristineLost, pristineRefused);
}

}
