#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSMAPPINGS_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSMAPPINGS_HPP

#ifdef _WIN32
#include <windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace GuestArena {

class MappingMutex {
public:
    void lock() {
        inner.lock();
        owner.store(GetCurrentThreadId(), std::memory_order_relaxed);
    }
    bool try_lock() {
        if (!inner.try_lock()) return false;
        owner.store(GetCurrentThreadId(), std::memory_order_relaxed);
        return true;
    }
    void unlock() {
        owner.store(0, std::memory_order_relaxed);
        inner.unlock();
    }
    bool HeldByCurrentThread() const {
        return owner.load(std::memory_order_relaxed) == GetCurrentThreadId();
    }
    DWORD Owner() const {
        return owner.load(std::memory_order_relaxed);
    }

private:
    std::mutex inner;
    std::atomic<DWORD> owner{0};
};

class WindowsMappings {
public:
    enum class Collection { All, Armed, Fresh };

    static WindowsMappings& Get() {
        static auto* mappings = new WindowsMappings;
        return *mappings;
    }

#ifdef APS5_COUNT_MAPPING_QUERIES
    static std::uint64_t MemoryQueries() {
        return queries.load(std::memory_order_relaxed);
    }
#endif

    void* Reserve(void* address, std::size_t bytes) {
        return allocate(GetCurrentProcess(), address, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
    }

    std::vector<std::pair<std::uintptr_t, std::size_t>> Commit(void* address, std::size_t bytes, DWORD protection, std::size_t granule, bool watched) {
        if (granule == 0) throw std::invalid_argument("guest memory commit granule is zero");
        std::vector<std::pair<std::uintptr_t, std::size_t>> created;
        std::lock_guard lock(mutex);
        collectableRanges.clear();
        disownRange(reinterpret_cast<std::uintptr_t>(address), reinterpret_cast<std::uintptr_t>(address) + bytes);
        const DWORD flags = MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | (watched ? MEM_WRITE_WATCH : 0);
        const auto cut = [](std::uintptr_t at, std::size_t size, const char* operation) {
            if (!VirtualFree(reinterpret_cast<void*>(at), size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail(operation);
        };
        const auto replace = [&](std::uintptr_t at, std::size_t size) {
            if (!allocate(GetCurrentProcess(), reinterpret_cast<void*>(at), size, flags, protection, nullptr, 0)) fail("replace guest placeholder with private memory");
            privateAllocations.emplace(at, at + size);
            if (!created.empty() && created.back().first + created.back().second == at) created.back().second += size;
            else created.emplace_back(at, size);
        };
        const auto end = reinterpret_cast<std::uintptr_t>(address) + bytes;
        for (auto cursor = reinterpret_cast<std::uintptr_t>(address); cursor < end;) {
            const auto memory = query(cursor);
            const auto regionEnd = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
            const auto stop = std::min(end, regionEnd);
            if (memory.State == MEM_RESERVE && std::min(end, cursor + granule) <= regionEnd) {
                privateAllocations.erase(privateAllocations.lower_bound(cursor), privateAllocations.lower_bound(stop));
                forgetClean(cursor, stop);
                const auto holder = reinterpret_cast<std::uintptr_t>(memory.AllocationBase);
                if (holder != cursor) cut(holder, cursor - holder, "split guest placeholder prefix");
                if (stop != regionEnd) cut(cursor, stop - cursor, "split guest placeholder suffix");
                while (cursor < stop && std::min(end, cursor + granule) <= stop) {
                    const auto size = std::min(end, cursor + granule) - cursor;
                    if (cursor + size != stop) cut(cursor, size, "split guest placeholder suffix");
                    replace(cursor, size);
                    cursor += size;
                }
                continue;
            }
            if (memory.State == MEM_RESERVE) {
                const auto limit = std::min(end, cursor + granule);
                auto placeholderEnd = stop;
                while (placeholderEnd < limit) {
                    const auto next = query(placeholderEnd);
                    if (next.State != MEM_RESERVE) break;
                    placeholderEnd = reinterpret_cast<std::uintptr_t>(next.BaseAddress) + next.RegionSize;
                }
                const auto size = std::min(limit, placeholderEnd) - cursor;
                reset(cursor, size);
                replace(cursor, size);
                cursor += size;
            } else {
                if (memory.State != MEM_COMMIT) throw std::runtime_error("guest memory is not committed");
                for (auto mapped = views.lower_bound(cursor & ~(pageBytes - 1)); mapped != views.end() && mapped->first < stop; ++mapped) {
                    mapped->second.protection = protection;
                    mapped->second.armed = 0;
                    invalidate(mapped->first, mapped->second);
                }
                DWORD previous;
                if (!VirtualProtect(reinterpret_cast<void*>(cursor), stop - cursor, protection, &previous)) fail("protect guest memory");
                cursor = stop;
            }
        }
        return created;
    }

    void Reset(void* address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        reset(reinterpret_cast<std::uintptr_t>(address), bytes);
    }

    void Map(void* address, std::size_t bytes, HANDLE section, std::uint64_t offset, DWORD protection, bool watched = false, bool resident = false, DWORD viewProtection = PAGE_EXECUTE_READWRITE, std::size_t sectionBytes = SIZE_MAX, std::vector<std::pair<std::uintptr_t, std::size_t>>* createdPrivate = nullptr) {
        if (sectionBytes < bytes && (resident || sectionBytes == 0 || bytes - sectionBytes >= pageBytes)) throw std::runtime_error("shared guest mapping extends a whole page past its section");
        std::lock_guard lock(mutex);
        collectableRanges.clear();
        auto cursor = reinterpret_cast<std::uintptr_t>(address);
        if (createdPrivate != nullptr) createdPrivate->reserve(createdPrivate->size() + bytes / pageBytes);
        reset(cursor, bytes);
        if (resident) watchResident = watched;
        HANDLE duplicate = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), section, GetCurrentProcess(), &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS)) fail("keep shared guest section");
        const auto owned = std::make_shared<Section>(duplicate);
        const auto key = reinterpret_cast<std::uintptr_t>(section);
        const auto slotOf = [&](std::size_t done) -> const std::shared_ptr<Physical>& {
            auto& slot = physical[std::make_pair(key, offset + done)];
            if (!slot) slot = std::make_shared<Physical>();
            return slot;
        };
        const auto pageOf = [](Physical& slot) {
            auto shared = slot.page.lock();
            if (!shared) {
                shared = std::make_shared<SharedPage>();
                shared->comparedWhenShared = slot.comparedWhenShared;
                slot.page = shared;
            }
            return shared;
        };
        const auto single = [&](std::size_t done) {
            if (!resident || residentRun(key, offset + done) != residentRuns.end()) return false;
            const auto found = physical.find(std::make_pair(key, offset + done));
            if (found == physical.end()) return true;
            const auto shared = found->second->page.lock();
            return !found->second->stored && (!shared || shared->aliases.empty());
        };
        const bool perPage = !resident && sectionBytes == SIZE_MAX;
        if (perPage) commitSection(section, offset, bytes);
        for (std::size_t done = 0; done < bytes;) {
            const auto base = cursor + done;
            if (single(done)) {
                auto runBytes = pageBytes;
                while (done + runBytes < bytes && (base + runBytes) % chunkBytes != 0 && single(done + runBytes)) runBytes += pageBytes;
                split(base, runBytes);
                commitResident(base, runBytes);
                if (createdPrivate != nullptr) {
                    if (!createdPrivate->empty() && createdPrivate->back().first + createdPrivate->back().second == base) createdPrivate->back().second += runBytes;
                    else createdPrivate->emplace_back(base, runBytes);
                }
                const auto run = residentRuns.emplace(base, Run{base + runBytes, {}, static_cast<std::uint32_t>(runBytes >> 12), key, offset + done}).first;
                residentIndex.emplace(std::make_pair(key, offset + done), run);
                for (std::size_t at = 0; at < runBytes; at += pageBytes) {
                    bool compared = false;
                    if (const auto found = physical.find(std::make_pair(key, offset + done + at)); found != physical.end()) {
                        auto& slot = *found->second;
                        if (slot.saved) {
                            std::memcpy(reinterpret_cast<void*>(base + at), slot.saved->bytes + slot.savedOffset, pageBytes);
                            restored(slot);
                        }
                        compared = slot.comparedWhenShared;
                        physical.erase(found);
                    }
                    views.emplace(base + at, View{nullptr, protection, {}, 0, compared, 0, owned, offset + done + at, 0, 0, true});
                }
                forgetClean(base, base + runBytes);
                remember(freshRanges, base, base + runBytes);
                DWORD previous;
                if (protection != PAGE_READWRITE && !VirtualProtect(reinterpret_cast<void*>(base), runBytes, protection, &previous)) fail("protect resident guest pages");
                done += runBytes;
                continue;
            }
            if (sectionBytes != SIZE_MAX) {
                if (const auto run = residentRun(key, offset + done); run != residentRuns.end()) demote(run->first);
                const auto slot = slotOf(done);
                const auto shared = pageOf(*slot);
                for (const auto alias : shared->aliases) views.at(alias).aliasDemoted = false;
                const auto viewBytes = std::min(pageBytes, sectionBytes - done);
                split(base, pageBytes);
                mapShared(base, section, offset + done, protection, viewProtection, viewBytes);
                slot->stored = true;
                shared->aliases.push_back(base);
                const auto created = views.emplace(base, View{shared, protection, {}, 0, false, 0, owned, offset + done, 0, 0, false, slot, hostPages(viewBytes)}).first;
                if (shared->comparedWhenShared) created->second.unreported = allSubpages;
                remember(freshRanges, base, base + pageBytes);
                invalidate(*shared);
                done += pageBytes;
                continue;
            }
            std::vector<std::pair<std::shared_ptr<Physical>, std::shared_ptr<SharedPage>>> pages;
            pages.reserve((bytes - done) / pageBytes);
            auto runBytes = std::size_t{0};
            do {
                if (const auto run = residentRun(key, offset + done + runBytes); run != residentRuns.end()) demote(run->first);
                const auto slot = slotOf(done + runBytes);
                const auto shared = pageOf(*slot);
                for (const auto alias : shared->aliases) views.at(alias).aliasDemoted = false;
                pages.emplace_back(slot, shared);
                runBytes += pageBytes;
            } while (done + runBytes < bytes && !single(done + runBytes));
            if (!perPage) commitSection(section, offset + done, runBytes);
            split(base, runBytes);
            mapRun(base, runBytes, section, offset + done);
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(base), runBytes, protection, &previous)) fail("protect shared guest pages");
            collectableRanges.clear();
            for (std::size_t at = 0; at < runBytes; at += pageBytes) {
                auto& [slot, shared] = pages[at / pageBytes];
                slot->stored = true;
                shared->aliases.push_back(base + at);
                invalidate(*shared);
                const auto created = views.emplace(base + at, View{std::move(shared), protection, {}, 0, false, 0, owned, offset + done + at, 0, 0, false, std::move(slot)}).first;
                if (created->second.page->comparedWhenShared) created->second.unreported = allSubpages;
            }
            remember(freshRanges, base, base + runBytes);
            done += runBytes;
        }
    }

    std::size_t AliasedPages(std::uintptr_t* pages, std::uintptr_t* identities, std::size_t capacity, std::size_t* bytes) {
        std::lock_guard lock(mutex);
        if (bytes != nullptr) *bytes = pageBytes;
        std::size_t count = 0;
        for (const auto& [base, view] : views) {
            if (view.page->aliases.size() < 2) continue;
            if (count < capacity) {
                pages[count] = base;
                identities[count] = reinterpret_cast<std::uintptr_t>(view.page.get());
            }
            ++count;
        }
        return count;
    }

    // Begins a guest protection change of the range (the caller's VirtualProtect follows): owned blocks
    // are disarmed and none is armed until EndSetProtection.
    void SetProtection(std::uintptr_t address, std::size_t bytes, DWORD protection) {
        std::lock_guard lock(mutex);
        collectableRanges.clear();
        ++protectionsInFlight;
        disownRange(address, address + bytes);
        for (auto it = views.lower_bound(address); it != views.end() && it->first < address + bytes; ++it) {
            it->second.protection = protection;
            it->second.armed = 0;
            invalidate(it->first, it->second);
        }
    }

    void Pin(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        disownRange(address, address + bytes);
        pinnedRanges.emplace(address, address + bytes);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            if (it->second.resident) {
                ++it->second.pins;
                invalidate(it->first, it->second);
                continue;
            }
            auto& page = *it->second.page;
            ++page.pins;
            for (const auto alias : page.aliases) {
                auto& view = views.at(alias);
                if (view.armed == 0) continue;
                protectSubpages(alias, view.armed, view.protection, "pin shared guest page writable", view.viewBytes);
                view.armed = 0;
            }
            invalidate(page);
        }
    }

    void Unpin(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        eraseRange(pinnedRanges, address, address + bytes);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            auto& pins = it->second.resident ? it->second.pins : it->second.page->pins;
            if (pins != 0) --pins;
            invalidate(it->first, it->second);
        }
    }

    void CompareWhenShared(std::uintptr_t address, std::size_t bytes, bool residentOnly = false) {
        std::lock_guard lock(mutex);
        collectableRanges.clear();
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            if (residentOnly && !it->second.resident && !it->second.aliasDemoted) continue;
            if (it->second.resident) {
                it->second.comparedWhenShared = true;
                continue;
            }
            auto& page = *it->second.page;
            const bool marked = page.comparedWhenShared;
            page.comparedWhenShared = true;
            it->second.slot->comparedWhenShared = true;
            for (const auto alias : page.aliases) {
                auto& view = views.at(alias);
                if (view.resident) continue;
                if (!marked) view.unreported = allSubpages;
                if (view.armed == 0) continue;
                protectSubpages(alias, view.armed, view.protection, "stop arming a compared shared page", view.viewBytes);
                view.armed = 0;
            }
        }
    }

    void ComparedPages(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool unreported) {
        std::lock_guard lock(mutex);
        const auto capacity = *count;
        *count = 0;
        const auto end = address + bytes;
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
            auto& view = it->second;
            if (view.resident || !view.page->comparedWhenShared) continue;
            const auto listed = static_cast<std::uint8_t>(subpagesOf(it->first, address, end) & (unreported ? view.unreported : allSubpages));
            for (std::size_t sub = 0; sub < subpages; ++sub) {
                const auto bit = static_cast<std::uint8_t>(1u << sub);
                if ((listed & bit) == 0) continue;
                if (*count == capacity) return;
                pages[(*count)++] = reinterpret_cast<void*>(it->first + sub * subpageBytes);
                if (unreported) view.unreported &= static_cast<std::uint8_t>(~bit);
            }
        }
    }

    enum class AccessGuardRefusal : std::uint8_t { None, Misaligned, AliasedView, Pinned, HostAccess, Resident, NotAccessible, Failed, InFlight, Executable };

    AccessGuardRefusal GuardAccess(std::uintptr_t address, std::size_t bytes) {
        if (address % subpageBytes != 0 || bytes % subpageBytes != 0 || bytes == 0) return AccessGuardRefusal::Misaligned;
        guardsUsed.store(true, std::memory_order_seq_cst);
        std::lock_guard lock(mutex);
        const auto end = address + bytes;
        if (protectionsInFlight != 0 || changesActive.load(std::memory_order_seq_cst) != 0) return AccessGuardRefusal::InFlight;
        if (views.lower_bound(address & ~(pageBytes - 1)) != views.end() && views.lower_bound(address & ~(pageBytes - 1))->first < end) return AccessGuardRefusal::AliasedView;
        if (overlapsRange(pinnedRanges, address, end)) return AccessGuardRefusal::Pinned;
        if (overlapsRange(hostAccess, address, end)) return AccessGuardRefusal::HostAccess;
        auto run = residentRuns.upper_bound(address);
        if (run != residentRuns.begin() && std::prev(run)->second.end > address) --run;
        if (run != residentRuns.end() && run->first < end) return AccessGuardRefusal::Resident;
        std::vector<std::pair<std::uintptr_t, DWORD>> arming;
        for (auto subpage = address; subpage < end; subpage += subpageBytes) {
            if (accessGuards.contains(subpage)) continue;
            const auto memory = query(subpage);
            if (memory.State != MEM_COMMIT || (memory.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) return AccessGuardRefusal::NotAccessible;
            if ((memory.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) != 0) return AccessGuardRefusal::Executable;
            arming.emplace_back(subpage, memory.Protect);
        }
        for (std::size_t k = 0; k < arming.size(); ++k) {
            accessGuards.emplace(arming[k].first, AccessGuard{arming[k].second, ++guardArms});
            guardedSubpages.fetch_add(1, std::memory_order_release);
            DWORD previous = 0;
            if (VirtualProtect(reinterpret_cast<void*>(arming[k].first), subpageBytes, PAGE_NOACCESS, &previous)) continue;
            for (std::size_t undo = 0; undo <= k; ++undo) {
                if (undo < k) VirtualProtect(reinterpret_cast<void*>(arming[undo].first), subpageBytes, arming[undo].second, &previous);
                accessGuards.erase(arming[undo].first);
                guardedSubpages.fetch_sub(1, std::memory_order_release);
            }
            return AccessGuardRefusal::Failed;
        }
        return AccessGuardRefusal::None;
    }

    void UnguardAccess(std::uintptr_t address, std::size_t bytes) {
        if (guardedSubpages.load(std::memory_order_acquire) == 0) return;
        std::lock_guard lock(mutex);
        dropAccessGuards(address, address + bytes);
    }

    void DropAccessGuards(std::uintptr_t address, std::size_t bytes) {
        UnguardAccess(address, bytes);
    }

    bool AccessGuarded(std::uintptr_t address) {
        if (guardedSubpages.load(std::memory_order_acquire) == 0 || mutex.HeldByCurrentThread()) return false;
        std::lock_guard lock(mutex);
        return accessGuards.contains(address & ~std::uintptr_t{subpageBytes - 1});
    }

    std::uintptr_t FirstAccessGuarded(std::uintptr_t address, std::size_t bytes) {
        if (guardedSubpages.load(std::memory_order_acquire) == 0 || bytes == 0) return 0;
        std::lock_guard lock(mutex);
        const auto found = accessGuards.lower_bound(address & ~std::uintptr_t{subpageBytes - 1});
        return found != accessGuards.end() && found->first < address + bytes ? std::max(found->first, address) : 0;
    }

    std::size_t AccessGuardedSubpages() const {
        return guardedSubpages.load(std::memory_order_acquire);
    }

    void SetAccessGuardObserver(void (*callback)(std::uintptr_t address)) {
        accessGuardObserver.store(callback, std::memory_order_release);
    }

    bool AccessGuardsUsed() const {
        return guardsUsed.load(std::memory_order_seq_cst);
    }

    void ChangeBegin() {
        changesActive.fetch_add(1, std::memory_order_seq_cst);
    }

    void ChangeEnd() {
        changesActive.fetch_sub(1, std::memory_order_seq_cst);
    }

    void BeginMappingChange() {
        std::lock_guard lock(mutex);
        ++protectionsInFlight;
    }

    void EndMappingChange() {
        std::lock_guard lock(mutex);
        if (protectionsInFlight != 0) --protectionsInFlight;
    }

    bool OpenAccessGuards(std::uintptr_t address, std::size_t bytes) {
        while (const auto guarded = FirstAccessGuarded(address, bytes)) {
            const auto observer = accessGuardObserver.load(std::memory_order_acquire);
            if (observer == nullptr) return false;
            observer(guarded);
            if (FirstAccessGuarded(guarded, 1) != 0) return false;
        }
        return true;
    }

    DWORD MappingLockOwner() const {
        return mutex.Owner();
    }

    void RunUnderLock(void (*body)(void*), void* argument) {
        std::lock_guard lock(mutex);
        body(argument);
    }

    bool HandleWrite(std::uintptr_t address) {
        if (mutex.HeldByCurrentThread()) return false;
        if (guardsUsed.load(std::memory_order_acquire)) {
            if (const auto served = handleAccessGuard(address, true)) return *served;
        }
        std::lock_guard lock(mutex);
        if (ownedLive != 0 && address < ownedAddressLimit && (ownedWord(address & ~std::uintptr_t{ownedBlockBytes - 1}) & 1u) != 0) ownedFaults.fetch_add(1, std::memory_order_relaxed);
        const auto base = address & ~(pageBytes - 1);
        const auto found = views.find(base);
        if (found == views.end() || !writable(found->second.protection)) return false;
        auto& view = found->second;
        if (view.resident) {
            const auto memory = query(address);
            return memory.State == MEM_COMMIT && writable(memory.Protect & 0xffu);
        }
        const auto written = subpagesOf(base, address, address + 1);
        invalidate(*view.page, written);
        protectSubpages(base, written, view.protection, "resume shared memory write", view.viewBytes);
        view.armed &= static_cast<std::uint8_t>(~written);
        return true;
    }

    bool HandleRead(std::uintptr_t address) {
        if (mutex.HeldByCurrentThread()) return false;
        if (guardsUsed.load(std::memory_order_acquire)) {
            if (const auto served = handleAccessGuard(address, false)) return *served;
        }
        std::lock_guard lock(mutex);
        if (!views.contains(address & ~(pageBytes - 1))) return false;
        const auto memory = query(address);
        return memory.State == MEM_COMMIT && (memory.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
    }

    bool ResidentSpan(std::uintptr_t address, std::size_t bytes, std::uintptr_t* start, std::uintptr_t* end) {
        std::lock_guard lock(mutex);
        auto run = residentRuns.upper_bound(address);
        if (run != residentRuns.begin() && std::prev(run)->second.end > address) --run;
        if (run == residentRuns.end() || run->first >= address + bytes) return false;
        *start = run->first;
        for (; run != residentRuns.end() && run->first < address + bytes; ++run) *end = run->second.end;
        return true;
    }

    bool ForgetSection(HANDLE section) {
        std::lock_guard lock(mutex);
        const auto key = reinterpret_cast<std::uintptr_t>(section);
        const auto first = physical.lower_bound(std::make_pair(key, std::uint64_t{0}));
        const auto last = physical.lower_bound(std::make_pair(key + 1, std::uint64_t{0}));
        bool blank = std::none_of(first, last, [](const auto& entry) { return entry.second->stored || !entry.second->page.expired(); });
        physical.erase(first, last);
        for (auto run = residentIndex.lower_bound(std::make_pair(key, std::uint64_t{0})); run != residentIndex.end() && run->first.first == key; run = residentIndex.erase(run)) blank = false;
        return blank;
    }

    bool BeginHostWrite(std::uintptr_t address, std::size_t bytes, bool store = true) {
        std::lock_guard lock(mutex);
        const auto first = views.lower_bound(address & ~(pageBytes - 1));
        const auto end = address + bytes;
        for (auto it = first; it != views.end() && it->first < end; ++it) {
            if (!writable(it->second.protection)) return false;
        }
        disownRange(address, end);
        hostAccess.emplace(address, end);
        for (auto it = first; it != views.end() && it->first < end; ++it) {
            auto& view = it->second;
            if (store) ++view.hostWrites;
            else ++view.hostImports;
            const auto written = subpagesOf(it->first, address, end);
            if (store) invalidate(it->first, view, written);
            else if (view.resident) forgetClean(it->first, it->first + pageBytes);
            const auto opened = static_cast<std::uint8_t>((store || view.resident) ? view.armed & written : 0);
            if (opened == 0) continue;
            protectSubpages(it->first, opened, view.protection, "open shared memory to a host write", view.viewBytes);
            view.armed &= static_cast<std::uint8_t>(~opened);
        }
        return true;
    }

    void EndHostWrite(std::uintptr_t address, std::size_t bytes, bool store = true) {
        std::lock_guard lock(mutex);
        const auto end = address + bytes;
        eraseRange(hostAccess, address, end);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
            if (store) --it->second.hostWrites;
            else --it->second.hostImports;
            if (store) invalidate(it->first, it->second, subpagesOf(it->first, address, end));
            else if (it->second.resident) forgetClean(it->first, it->first + pageBytes);
        }
    }

    std::pair<std::uint64_t, std::uint64_t> SavedMemory() const {
        return {savedBytes.load(std::memory_order_relaxed), savedBlocks.load(std::memory_order_relaxed)};
    }

    void* MapAlias(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto refuse = [&](const char* reason) {
            char text[192];
            std::snprintf(text, sizeof(text), "read-write alias of shared guest memory 0x%llx+0x%llx: %s", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), reason);
            return std::runtime_error(text);
        };
        if (address % pageBytes != 0 || bytes % pageBytes != 0 || bytes == 0) throw refuse("the range is not made of whole shared pages");
        for (std::size_t done = 0; done < bytes; done += pageBytes) {
            const auto found = views.find(address + done);
            if (found == views.end() || !found->second.resident) continue;
            const auto run = std::prev(residentRuns.upper_bound(address + done));
            if (run->first < address || run->second.end > address + bytes) throw refuse("a resident run of direct memory extends outside the range");
            for (auto page = views.find(run->first); page != views.end() && page->first < run->second.end; ++page) {
                if (page->second.hostWrites != 0) return nullptr;
            }
        }
        for (std::size_t done = 0; done < bytes; done += pageBytes) {
            const auto found = views.find(address + done);
            if (found != views.end() && found->second.resident) demote(address + done, true);
        }
        auto view = views.find(address);
        if (view == views.end()) throw refuse("the range does not start at a shared view");
        const auto section = view->second.section;
        const auto offset = view->second.offset;
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        bool contiguous = true;
        for (std::size_t done = 0; done < bytes; done += pageBytes) {
            const auto page = views.find(address + done);
            if (page == views.end()) throw refuse("the range is not a run of shared views");
            if (page->second.viewBytes != pageBytes) throw refuse("a page extends past the end of its file");
            if (page->second.offset != offset + done || (page->second.section != section && !sameSection(page->second.section->handle, section->handle))) contiguous = false;
        }
        if (!contiguous) return mapAliasRuns(address, bytes);
        const auto lead = offset % system.dwAllocationGranularity;
        void* alias = map(section->handle, GetCurrentProcess(), nullptr, offset - lead, lead + bytes, 0, PAGE_READWRITE, nullptr, 0);
        if (alias == nullptr) {
            char text[160];
            std::snprintf(text, sizeof(text), "MapViewOfFile3 of a read-write alias of shared guest memory 0x%llx+0x%llx", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes));
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), text);
        }
        return static_cast<char*>(alias) + lead;
    }

    void UnmapAlias(void* alias) {
        if (alias == nullptr) return;
        {
            std::lock_guard lock(mutex);
            if (const auto found = splitAliases.find(reinterpret_cast<std::uintptr_t>(alias)); found != splitAliases.end()) {
                const auto error = releaseAlias(found->first, found->second);
                splitAliases.erase(found);
                if (error != ERROR_SUCCESS) throw std::system_error(static_cast<int>(error), std::system_category(), "unmap a run of a shared guest alias");
                return;
            }
        }
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        const auto base = reinterpret_cast<std::uintptr_t>(alias) & ~(static_cast<std::uintptr_t>(system.dwAllocationGranularity) - 1);
        if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(base), 0)) fail("unmap shared guest alias");
    }

    bool Protection(std::uintptr_t address, std::uint32_t* protection) {
        std::lock_guard lock(mutex);
        if (guardedSubpages.load(std::memory_order_acquire) != 0) {
            if (const auto guarded = accessGuards.find(address & ~std::uintptr_t{subpageBytes - 1}); guarded != accessGuards.end() && !views.contains(address & ~(pageBytes - 1))) {
                *protection = guarded->second.protection;
                return true;
            }
        }
        const auto found = views.find(address & ~(pageBytes - 1));
        if (found == views.end()) return false;
        *protection = found->second.protection;
        return true;
    }

    // Owned blocks (the driver's owned proofs, APS5_NO_OWNED_PROOFS): 64 KiB blocks of guest memory
    // whose every CPU change is seen without a fault. Each block has a word: bit 0 is set while the
    // block is armed, the rest counts the times it was disarmed. Nothing is write-protected: a CPU
    // store sets the page's write-watch bit (the pages are MEM_WRITE_WATCH memory, checked when
    // armed), and OwnedUnchanged asks GetWriteWatch, without resetting, whether a page of the range
    // was written (the first read of a never-touched page sets its bit too, a demand-zero fault: a
    // conservative answer once per page). A bit is only cleared by a reset, and every reset of guest memory is Collect's or
    // ArmOwned's, under the mutex: a reset that takes a written page of an armed block moves its word
    // (watchTaken, and a Fresh collection's reset of settled pages). Arming
    // resets the block's bits itself so that writes before it do not count against the proof, but keeps the pages it took (ownedPending), and Collect reports them
    // as written until a clearing collect takes them, so the write tracker sees every store it would
    // have seen without the arming. Every operation here that changes the block's mapping or
    // protection or opens it to a host call (Commit, Reset, Map, SetProtection, Pin, BeginHostWrite
    // and import, a resident run's demotion or eviction) moves the word first:
    // a recommitted page reads zeros without a write bit. So while OwnedUnchanged holds for the words
    // ArmOwned returned, no CPU store reached the range after ArmOwned. GPU writes through a host
    // import bypass the page tables and are not seen here.
    // A block's pages may be resident direct memory with a single guest view, private commits (the
    // guest's flexible and memory-pool memory, with OwnedPrivate), read-write or read-only, and
    // reserved placeholders. Refused, arming nothing (OwnedRefusal, counted): shared or aliased views
    // (tracked by protection, not write watch), pinned pages and imports being made, open host
    // writes, inaccessible, guarded or write-copy pages or other mapping types, any block while a
    // guest protection change is in flight, and pages that are not write-watched (Unwatched).
    static constexpr std::size_t ownedBlockBytes = 0x10000;
    static constexpr unsigned OwnedPrivate = 1;

    enum class OwnedRefusal : std::uint8_t { None, Outside, Shared, Pinned, HostWrite, InFlight, NotWritable, Private, Empty, Unwatched, Count };
    static constexpr std::size_t OwnedRefusalCount = static_cast<std::size_t>(OwnedRefusal::Count);

    // faults: guest write faults inside armed owned blocks. Owned blocks never change a protection,
    // so each one is another mechanism's (shared views, a genuine access violation) and none is resolved
    // for them. collected: disarms by a written page (a reset that took it, or a re-arm that found
    // it); taken: pages whose write bits an arming took (reported later as written); queries:
    // GetWriteWatch calls of OwnedUnchanged; dirty: OwnedUnchanged answers that found a written page;
    // locked: OwnedUnchanged answers asked again under the lock (a reset ran across the first);
    // epochHeld: answers with a mark taken from the blocks' checks alone (no query); epochQueried:
    // answers with a mark that had to query (a block not checked since the mark).
    struct OwnedCounts {
        std::uint64_t arms = 0;
        std::uint64_t armedBlocks = 0;
        std::uint64_t refused = 0;
        std::uint64_t faults = 0;
        std::uint64_t disowned = 0;
        std::uint64_t privateBlocks = 0;
        std::uint64_t unmappedBlocks = 0;
        std::uint64_t partialBlocks = 0;
        std::uint64_t collected = 0;
        std::uint64_t taken = 0;
        std::uint64_t queries = 0;
        std::uint64_t dirty = 0;
        std::uint64_t locked = 0;
        std::uint64_t epochHeld = 0;
        std::uint64_t epochQueried = 0;
        std::array<std::uint64_t, OwnedRefusalCount> reasons{};
    };

    // Arms every 64 KiB block overlapping [address, address + bytes) (count must be their number) and
    // stores each block's armed word, or arms none and returns false when one cannot be armed. Blocks
    // already armed keep their word unless a page of theirs was written since (then the word moves
    // and the block is armed afresh).
    bool ArmOwned(std::uintptr_t address, std::size_t bytes, std::uint64_t* words, std::size_t count, unsigned flags = 0) {
        if (bytes == 0 || bytes > ownedAddressLimit || address > ownedAddressLimit - bytes) return false;
        const auto first = address & ~std::uintptr_t{ownedBlockBytes - 1};
        const auto stop = (address + bytes + ownedBlockBytes - 1) & ~std::uintptr_t{ownedBlockBytes - 1};
        if ((stop - first) / ownedBlockBytes != count) return false;
        std::lock_guard lock(mutex);
        if (protectionsInFlight != 0) return NoteOwnedRefusal(OwnedRefusal::InFlight);
        std::vector<std::pair<std::uintptr_t, OwnedPlan>> plans;
        std::vector<OwnedRun> runs;
        for (auto block = first; block < stop; block += ownedBlockBytes) {
            if ((ownedWord(block) & 1u) != 0) {
                if (!blockWritten(block)) continue;
                disownBlock(block, ownedCollected);
            }
            OwnedPlan plan;
            if (const auto refusal = ownable(block, (flags & OwnedPrivate) != 0, plan); refusal != OwnedRefusal::None) return NoteOwnedRefusal(refusal);
            appendRuns(runs, block, plan.layout, block, block + ownedBlockBytes);
            plans.emplace_back(block, plan);
        }
        for (const auto& run : runs) {
            if (!queryWatch(run.begin, run.end).has_value()) return NoteOwnedRefusal(OwnedRefusal::Unwatched);
        }
        for (const auto& run : runs) takeWrites(run.begin, run.end);
        for (const auto& [block, plan] : plans) {
            auto& slot = ownedSlot(block);
            slot.store(slot.load(std::memory_order_relaxed) + 1, std::memory_order_seq_cst);
            {
                std::lock_guard layouts(ownedMutex);
                ownedLayouts[block] = plan.layout;
            }
            ++ownedLive;
            ownedArmedBlocks.fetch_add(1, std::memory_order_relaxed);
            if (plan.privatePages) ownedPrivateBlocks.fetch_add(1, std::memory_order_relaxed);
            if (!plan.mapped) ownedUnmappedBlocks.fetch_add(1, std::memory_order_relaxed);
            else if (plan.unmapped) ownedPartialBlocks.fetch_add(1, std::memory_order_relaxed);
        }
        std::size_t index = 0;
        for (auto block = first; block < stop; block += ownedBlockBytes, ++index) words[index] = ownedWord(block);
        ownedArms.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool NoteOwnedRefusal(OwnedRefusal refusal) {
        ownedRefused.fetch_add(1, std::memory_order_relaxed);
        ownedReasons[static_cast<std::size_t>(refusal)].fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // Whether every block of the range is still armed with the word ArmOwned stored for it and no
    // mapped page of [address, address + bytes) was written since: one GetWriteWatch without reset
    // per run of pages in one allocation (a call cannot span two). Without the mapping lock: the
    // words are read before and after the queries, and ownedResets (odd while a reset and the word
    // moves it implies are under way) must be even and unchanged across them, so no reset that took
    // a bit of the range falls between a query and the words; otherwise it is asked again under the
    // lock.
    //
    // `mark` (OwnedMark, the caller's last ordering point; 0: none) lets the answer come from the
    // blocks' checks: a block is checked at c when a query that began after a read of the clock
    // (ownedClock) returning c found none of the block's armed pages written and its word still the
    // one it was asked with. With every block checked at c >= mark and its word unchanged, the answer
    // holds without a query: the clock only moves by read-modify-writes, so a read returning
    // c >= mark reads from the mark's increment or a later one (its release sequence) and every CPU
    // store that happened before the mark had set its page's bit before that query, which then saw
    // it (no check) or found the word moved by the reset that took it (a reset moves the words inside
    // the reset window). A store after the mark may stay unseen until the caller's next mark, the
    // collect epoch's rule: the driver marks at its ordering points (a submission, a guest-memory
    // wait the CPU may have satisfied, a drain, a reap). Without a mark every answer queries, as
    // before. With a mark and a block not checked since it, the blocks are queried whole and checked;
    // a written page found in them is asked again over the range alone.
    bool OwnedUnchanged(std::uintptr_t address, std::size_t bytes, const std::uint64_t* words, std::size_t count, std::uint64_t mark = 0) {
        if (bytes == 0 || bytes > ownedAddressLimit || address > ownedAddressLimit - bytes) return false;
        const auto first = address & ~std::uintptr_t{ownedBlockBytes - 1};
        const auto stop = (address + bytes + ownedBlockBytes - 1) & ~std::uintptr_t{ownedBlockBytes - 1};
        if ((stop - first) / ownedBlockBytes != count) return false;
        if (mark != 0) {
            if (checkedSince(first, stop, words, mark)) {
                ownedEpochHeld.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            ownedEpochQueried.fetch_add(1, std::memory_order_relaxed);
            const auto clock = ownedClock.load(std::memory_order_seq_cst);
            if (!wordsHold(first, stop, words)) return false;
            if (unchangedNow(first, stop - first, first, stop, words)) {
                markChecked(first, stop, clock);
                return true;
            }
        }
        return unchangedNow(address, bytes, first, stop, words);
    }

    // A new mark: the clock after this call's increment (see OwnedUnchanged).
    std::uint64_t OwnedMark() {
        return ownedClock.fetch_add(1, std::memory_order_seq_cst) + 1;
    }

    OwnedCounts TakeOwnedCounts() {
        const auto take = [](std::atomic<std::uint64_t>& value) { return value.exchange(0, std::memory_order_relaxed); };
        OwnedCounts counts{take(ownedArms), take(ownedArmedBlocks), take(ownedRefused), take(ownedFaults), take(ownedDisowned), take(ownedPrivateBlocks), take(ownedUnmappedBlocks), take(ownedPartialBlocks), take(ownedCollected), take(ownedTaken), take(ownedQueries), take(ownedDirty), take(ownedLocked), take(ownedEpochHeld), take(ownedEpochQueried), {}};
        for (std::size_t i = 0; i < OwnedRefusalCount; ++i) counts.reasons[i] = take(ownedReasons[i]);
        return counts;
    }

    // Ends the guest protection change SetProtection began (the caller's own VirtualProtect landed).
    void EndSetProtection() {
        std::lock_guard lock(mutex);
        if (protectionsInFlight != 0) --protectionsInFlight;
    }

#ifdef APS5_COUNT_MAPPING_QUERIES
    void CollectWork(std::uint64_t* systemCalls, std::uint64_t* pieces) {
        std::lock_guard lock(mutex);
        *systemCalls = collectSystemCalls;
        *pieces = collectPieces;
    }
#endif

    bool Collect(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear, Collection collection = Collection::All) {
        std::lock_guard lock(mutex);
        const auto capacity = *count;
        *count = 0;
        const auto end = address + bytes;
        if (collection == Collection::Fresh) {
            const auto fresh = freshRanges.upper_bound(address);
            if ((fresh == freshRanges.begin() || std::prev(fresh)->second <= address) && (fresh == freshRanges.end() || fresh->first >= end)) return true;
            for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
                if (it->second.resident) {
                    if (!watchResident) return false;
                    continue;
                }
                if (it->second.page->comparedWhenShared) return false;
                if (it->second.protection != PAGE_NOACCESS) continue;
                for (auto at = std::max(address, it->first); at < std::min(end, it->first + pageBytes); at += 4096) {
                    if ((it->second.collected & (1u << ((at - it->first) / 4096))) == 0) return false;
                }
            }
        } else if (clear && !collectable(address, end)) return false;
        auto position = views.lower_bound(address & ~(pageBytes - 1));
        std::uintptr_t cleanStart = 0;
        std::uintptr_t cleanEnd = 0;
        const auto flushClean = [&] {
            if (cleanEnd > cleanStart) rememberClean(cleanStart, cleanEnd);
            cleanStart = cleanEnd = 0;
        };
        for (auto cursor = address; cursor < end;) {
            const auto nextClean = cleanRanges.upper_bound(cursor);
            if (nextClean != cleanRanges.begin()) {
                const auto clean = std::prev(nextClean);
                if (cursor < clean->second) {
                    cursor = std::min(end, clean->second);
                    continue;
                }
            }
            const auto base = cursor & ~(pageBytes - 1);
            const auto found = viewAt(base, position);
            if (found != views.end() && found->second.resident) {
                flushClean();
                if (!watchResident) return false;
                const auto available = capacity - *count;
                if (available == 0) return true;
                const auto run = std::prev(residentRuns.upper_bound(cursor));
                const auto stop = std::min({end, run->second.end, (cursor & ~std::uintptr_t{4095}) + available * 4096, nextClean == cleanRanges.end() ? end : nextClean->first});
                collectResident(run->first, run->second, cursor, stop, pages, count, clear, collection);
                if (*count == capacity) return true;
                cursor = stop;
            } else if (found != views.end()) {
                auto& view = found->second;
                auto stop = std::min(std::min(end, base + pageBytes), nextClean == cleanRanges.end() ? end : nextClean->first);
                if (collection != Collection::All) {
                    const auto selected = [&](std::uintptr_t at) {
                        const bool collected = (view.collected & (1u << ((at - base) / 4096))) != 0;
                        return collection == Collection::Armed ? collected : !collected;
                    };
                    while (cursor < stop && !selected(cursor)) cursor = std::min(stop, cursor + 4096);
                    if (cursor == stop) continue;
                    for (auto at = cursor + 4096; at < stop; at += 4096) {
                        if (!selected(at)) {
                            stop = at;
                            break;
                        }
                    }
                }
                if (view.protection == PAGE_NOACCESS) {
                    flushClean();
                    return false;
                }
                const bool pinned = view.page->pins != 0;
                std::uint8_t settled = 0;
                auto settledEnd = cursor;
                bool full = false;
                for (std::size_t sub = 0; sub < subpages; ++sub) {
                    const auto from = std::max(cursor, base + sub * subpageBytes);
                    const auto to = std::min(stop, base + (sub + 1) * subpageBytes);
                    if (from >= to) continue;
                    if (collection == Collection::Fresh || pinned || view.seen[sub] != view.page->generation[sub]) {
                        if (*count == capacity) {
                            full = true;
                            break;
                        }
                        pages[(*count)++] = reinterpret_cast<void*>(from);
                    }
                    settled |= static_cast<std::uint8_t>(1u << sub);
                    settledEnd = to;
                }
                if (clear) {
                    for (std::size_t sub = 0; sub < subpages; ++sub) {
                        const auto bit = static_cast<std::uint8_t>(1u << sub);
                        if ((settled & bit) == 0 || (view.collected & bit) != 0) continue;
                        view.collected |= bit;
                        forget(freshRanges, base + sub * subpageBytes, base + (sub + 1) * subpageBytes);
                    }
                }
                if (clear && !pinned && settled != 0) {
                    std::uint8_t untracked = 0;
                    for (const auto alias : view.page->aliases) {
                        auto& other = views.at(alias);
                        if (other.resident) continue;
                        const auto arming = static_cast<std::uint8_t>(settled & ~other.armed);
                        if (!writable(other.protection) || arming == 0) continue;
                        if (other.hostWrites != 0) {
                            untracked |= arming;
                            continue;
                        }
                        const DWORD protection = other.protection == PAGE_EXECUTE_READWRITE ? PAGE_EXECUTE_READ : PAGE_READONLY;
                        protectSubpages(alias, arming, protection, "arm shared memory write tracking", other.viewBytes);
                        other.armed |= arming;
                    }
                    const auto tracked = static_cast<std::uint8_t>(settled & ~untracked);
                    for (std::size_t sub = 0; sub < subpages; ++sub) {
                        if ((tracked & (1u << sub)) == 0) continue;
                        const auto from = std::max(cursor, base + sub * subpageBytes);
                        const auto to = std::min(settledEnd, base + (sub + 1) * subpageBytes);
                        if (from == base + sub * subpageBytes && to == base + (sub + 1) * subpageBytes) view.seen[sub] = view.page->generation[sub];
                        if (from != cleanEnd) {
                            flushClean();
                            cleanStart = from;
                        }
                        cleanEnd = to;
                    }
                }
                if (full) {
                    flushClean();
                    return true;
                }
                cursor = stop;
            } else {
                flushClean();
                if (collection == Collection::Fresh) {
                    const auto next = views.lower_bound(cursor);
                    cursor = next == views.end() ? end : std::min(end, next->first);
                    continue;
                }
                const auto allocationEnd = privateAllocationEnd(cursor);
                if (allocationEnd == 0) return false;
                const auto stop = std::min(end, allocationEnd);
                if (capacity == *count) return true;
                const auto touched = (((stop + 4095) & ~static_cast<std::uintptr_t>(4095)) - (cursor & ~static_cast<std::uintptr_t>(4095))) / 4096;
                const auto room = std::min<ULONG_PTR>(capacity - *count, touched);
                ULONG_PTR available = room;
                DWORD granularity = 0;
#ifdef APS5_COUNT_MAPPING_QUERIES
                ++collectSystemCalls;
                collectPieces += (stop + pageBytes - 1) / pageBytes - cursor / pageBytes;
#endif
                const bool committed = !verifyDirect() || committedPrivate(cursor, stop);
                const ResetWindow window(ownedResets, clear);
                if (GetWriteWatch(clear ? WRITE_WATCH_FLAG_RESET : 0, reinterpret_cast<void*>(cursor), stop - cursor, pages + *count, &available, &granularity) != 0) {
                    const auto memory = query(cursor);
                    if (memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE) return false;
                    fail("collect private guest writes");
                }
                if (!committed) staleProof(cursor, stop);
                *count += watchTaken(pages + *count, available, room, cursor, stop, clear);
                if (*count == capacity) return true;
                cursor = stop;
            }
        }
        flushClean();
        return true;
    }

private:
    std::optional<bool> handleAccessGuard(std::uintptr_t address, bool write) {
        const auto subpage = address & ~std::uintptr_t{subpageBytes - 1};
        std::uint64_t arm = 0;
        {
            std::lock_guard lock(mutex);
            if (const auto guarded = accessGuards.find(subpage); guarded != accessGuards.end()) {
                arm = guarded->second.arm;
            } else {
                const auto memory = query(address);
                const auto protection = memory.Protect & 0xffu;
                const bool allowed = memory.State == MEM_COMMIT && (memory.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0 && (!write || writable(protection) || protection == PAGE_WRITECOPY || protection == PAGE_EXECUTE_WRITECOPY);
                if (allowed) return true;
                return std::nullopt;
            }
        }
        const auto observer = accessGuardObserver.load(std::memory_order_acquire);
        if (observer == nullptr) return false;
        observer(address);
        std::lock_guard lock(mutex);
        const auto guarded = accessGuards.find(subpage);
        return guarded == accessGuards.end() || guarded->second.arm != arm;
    }

    void dropAccessGuards(std::uintptr_t address, std::uintptr_t end) {
        auto it = accessGuards.lower_bound(address & ~std::uintptr_t{subpageBytes - 1});
        while (it != accessGuards.end() && it->first < end) {
            DWORD previous = 0;
            const bool restored = VirtualProtect(reinterpret_cast<void*>(it->first), subpageBytes, it->second.protection, &previous) != 0;
            it = accessGuards.erase(it);
            guardedSubpages.fetch_sub(1, std::memory_order_release);
            if (!restored) fail("drop a guest access guard");
        }
    }

    bool collectable(std::uintptr_t address, std::uintptr_t end) {
        const auto proven = collectableRanges.upper_bound(address);
        if (proven != collectableRanges.begin() && std::prev(proven)->second >= end) return true;
        auto position = views.lower_bound(address & ~(pageBytes - 1));
        for (auto cursor = address; cursor < end;) {
            const auto base = cursor & ~(pageBytes - 1);
            if (const auto found = viewAt(base, position); found != views.end()) {
                if (found->second.protection == PAGE_NOACCESS || (found->second.resident ? !watchResident : found->second.page->comparedWhenShared)) return false;
                cursor = std::min(end, base + pageBytes);
                continue;
            }
            const auto allocationEnd = privateAllocationEnd(cursor);
            if (allocationEnd == 0) return false;
            cursor = std::min(end, allocationEnd);
        }
        remember(collectableRanges, address, end);
        return true;
    }

    static bool verifyDirect() {
        static const bool verify = std::getenv("APS5_VERIFY_DIRECT_COLLECT") != nullptr;
        return verify;
    }

    static bool committedPrivate(std::uintptr_t address, std::uintptr_t end) {
        for (auto cursor = address; cursor < end;) {
            const auto memory = query(cursor);
            if (memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE) return false;
            cursor = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
        }
        return true;
    }

    [[noreturn]] static void staleProof(std::uintptr_t address, std::uintptr_t end) {
        char text[192];
        std::snprintf(text, sizeof(text), "a private collect of guest writes succeeded over 0x%llx+0x%llx, which is no longer committed private memory (APS5_VERIFY_DIRECT_COLLECT)", static_cast<unsigned long long>(address), static_cast<unsigned long long>(end - address));
        throw std::runtime_error(text);
    }

    std::uintptr_t privateAllocationEnd(std::uintptr_t address) const {
        const auto next = privateAllocations.upper_bound(address);
        if (next == privateAllocations.begin()) return 0;
        const auto allocation = std::prev(next);
        return address < allocation->second ? allocation->second : 0;
    }

    static constexpr std::size_t pageBytes = 0x4000;
    static constexpr std::size_t subpageBytes = 0x1000;
    static_assert(subpageBytes == 4096, "Collect reports one page per subpage");
    static constexpr std::size_t subpages = pageBytes / subpageBytes;
    static constexpr std::uint8_t allSubpages = (1u << subpages) - 1;
    static constexpr std::size_t chunkBytes = 0x200000;
    struct SharedPage {
        std::array<std::uint64_t, subpages> generation = [] {
            std::array<std::uint64_t, subpages> initial{};
            initial.fill(1);
            return initial;
        }();
        std::vector<std::uintptr_t> aliases;
        std::uint32_t pins = 0;
        bool comparedWhenShared = false;
    };
    struct Section {
        HANDLE handle;
        explicit Section(HANDLE handle) : handle(handle) {}
        Section(const Section&) = delete;
        Section& operator=(const Section&) = delete;
        ~Section() { CloseHandle(handle); }
    };
    static inline std::atomic<std::uint64_t> savedBytes{0};
    static inline std::atomic<std::uint64_t> savedBlocks{0};
    struct Saved {
        unsigned char* bytes = nullptr;
        std::size_t length;
        std::array<std::uint64_t, chunkBytes / pageBytes / 64> committed{};
        explicit Saved(std::size_t length) : length(length) {
            bytes = static_cast<unsigned char*>(VirtualAlloc(nullptr, length, MEM_RESERVE, PAGE_READWRITE));
            if (bytes == nullptr) fail("reserve unmapped direct memory");
            savedBlocks.fetch_add(1, std::memory_order_relaxed);
        }
        Saved(const Saved&) = delete;
        Saved& operator=(const Saved&) = delete;
        ~Saved() {
            VirtualFree(bytes, 0, MEM_RELEASE);
            std::uint64_t pages = 0;
            for (const auto word : committed) pages += static_cast<std::uint64_t>(std::popcount(word));
            savedBytes.fetch_sub(pages * pageBytes, std::memory_order_relaxed);
            savedBlocks.fetch_sub(1, std::memory_order_relaxed);
        }

        void Commit(std::size_t offset) {
            const auto page = offset / pageBytes;
            const auto bit = std::uint64_t{1} << (page & 63);
            if ((committed[page >> 6] & bit) != 0) return;
            if (!VirtualAlloc(bytes + offset, pageBytes, MEM_COMMIT, PAGE_READWRITE)) fail("keep unmapped direct memory");
            committed[page >> 6] |= bit;
            savedBytes.fetch_add(pageBytes, std::memory_order_relaxed);
        }

        void Decommit(std::size_t offset) {
            const auto page = offset / pageBytes;
            const auto bit = std::uint64_t{1} << (page & 63);
            if ((committed[page >> 6] & bit) == 0) return;
            if (!VirtualFree(bytes + offset, pageBytes, MEM_DECOMMIT)) fail("drop the saved copy of remapped direct memory");
            committed[page >> 6] &= ~bit;
            savedBytes.fetch_sub(pageBytes, std::memory_order_relaxed);
        }
    };
    struct Physical {
        std::weak_ptr<SharedPage> page;
        bool stored = false;
        bool comparedWhenShared = false;
        std::shared_ptr<Saved> saved;
        std::size_t savedOffset = 0;
    };
    struct View {
        std::shared_ptr<SharedPage> page;
        DWORD protection;
        std::array<std::uint64_t, subpages> seen;
        std::uint8_t armed;
        bool comparedWhenShared = false;
        std::uint32_t pins = 0;
        std::shared_ptr<Section> section;
        std::uint64_t offset;
        std::uint32_t hostWrites;
        std::uint8_t collected = 0;
        bool resident = false;
        std::shared_ptr<Physical> slot;
        std::size_t viewBytes = pageBytes;
        std::uint32_t hostImports = 0;
        bool aliasDemoted = false;
        std::uint8_t unreported = 0;
    };
    struct Run {
        std::uintptr_t end;
        std::array<std::uint64_t, 8> collected;
        std::uint32_t fresh;
        std::uintptr_t key;
        std::uint64_t offset;
    };
    struct Window {
        void* view = nullptr;
        unsigned char* bytes = nullptr;
    };
    using Ranges = std::map<std::uintptr_t, std::uintptr_t>;

    static bool runCollected(const Run& run, std::size_t bit) {
        return (run.collected[bit >> 6] >> (bit & 63) & 1) != 0;
    }

    void collectResident(std::uintptr_t start, Run& run, std::uintptr_t from, std::uintptr_t stop, void** pages, std::size_t* count, bool clear, Collection collection) {
        const auto first = from & ~std::uintptr_t{4095};
        if (run.fresh == 0 && collection != Collection::Fresh) {
            const ULONG_PTR room = (stop - first + 4095) / 4096;
            ULONG_PTR reported = room;
            DWORD granularity = 0;
            const ResetWindow window(ownedResets, clear);
            if (GetWriteWatch(clear ? WRITE_WATCH_FLAG_RESET : 0, reinterpret_cast<void*>(first), stop - first, pages + *count, &reported, &granularity) != 0) fail("collect resident guest writes");
            reported = watchTaken(pages + *count, reported, room, first, stop, clear);
            for (ULONG_PTR index = 0; index < reported; ++index) pages[*count + index] = reinterpret_cast<void*>(std::max(from, reinterpret_cast<std::uintptr_t>(pages[*count + index])));
            *count += reported;
            return;
        }
        std::vector<std::uintptr_t> written;
        if (collection != Collection::Fresh) {
            std::vector<void*> found((stop - first + 4095) / 4096);
            ULONG_PTR reported = found.size();
            DWORD granularity = 0;
            const ResetWindow window(ownedResets, clear);
            if (GetWriteWatch(clear ? WRITE_WATCH_FLAG_RESET : 0, reinterpret_cast<void*>(first), stop - first, found.data(), &reported, &granularity) != 0) fail("collect resident guest writes");
            reported = watchTaken(found.data(), reported, found.size(), first, stop, clear);
            for (ULONG_PTR index = 0; index < reported; ++index) written.push_back(reinterpret_cast<std::uintptr_t>(found[index]));
        }
        auto next = written.begin();
        std::uintptr_t resetFrom = 0;
        const auto flushReset = [&](std::uintptr_t to) {
            if (resetFrom == 0) return;
            if (ownedLive == 0) {
                if (ResetWriteWatch(reinterpret_cast<void*>(resetFrom), to - resetFrom) != 0) fail("reset resident guest writes");
                resetFrom = 0;
                return;
            }
            const ResetWindow window(ownedResets, true);
            thread_local std::vector<void*> taken;
            taken.resize((to - resetFrom + 4095) / 4096);
            ULONG_PTR reported = taken.size();
            DWORD granularity = 0;
            if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, reinterpret_cast<void*>(resetFrom), to - resetFrom, taken.data(), &reported, &granularity) != 0) fail("reset resident guest writes");
            for (ULONG_PTR i = 0; i < reported; ++i) disownWritten(reinterpret_cast<std::uintptr_t>(taken[i]));
            resetFrom = 0;
        };
        for (auto at = first; at < stop; at += 4096) {
            const auto bit = (at - start) >> 12;
            const bool fresh = !runCollected(run, bit);
            bool wrote = false;
            while (next != written.end() && *next < at) ++next;
            if (next != written.end() && *next == at) wrote = true;
            const bool report = collection == Collection::Fresh ? fresh : collection == Collection::Armed ? !fresh && wrote : fresh || wrote;
            if (report) pages[(*count)++] = reinterpret_cast<void*>(std::max(at, from));
            const bool settle = clear && fresh && collection != Collection::Armed;
            if (settle) {
                run.collected[bit >> 6] |= 1ull << (bit & 63);
                --run.fresh;
                forget(freshRanges, at, at + 4096);
                if (collection == Collection::Fresh && resetFrom == 0) resetFrom = at;
            } else if (collection == Collection::Fresh) {
                flushReset(at);
            }
        }
        if (collection == Collection::Fresh) flushReset(stop);
    }

    static constexpr std::uintptr_t ownedAddressLimit = std::uintptr_t{1} << 47;
    static constexpr std::size_t ownedLeafBlocks = std::size_t{1} << 16;
    static constexpr unsigned ownedLeafShift = 32;
    static constexpr std::size_t ownedLeafCount = static_cast<std::size_t>(ownedAddressLimit >> ownedLeafShift);
    static_assert(ownedLeafBlocks * ownedBlockBytes == (std::uint64_t{1} << ownedLeafShift));

    std::uint64_t ownedWord(std::uintptr_t block) const {
        const auto* leaf = ownedLeaves[block >> ownedLeafShift].load(std::memory_order_acquire);
        return leaf != nullptr ? leaf[(block / ownedBlockBytes) & (ownedLeafBlocks - 1)].load(std::memory_order_relaxed) : 0;
    }

    // The block's word, its leaf made on first use (under the mutex; leaves live as long as the
    // mappings, so a lock-free reader never sees one go).
    std::atomic<std::uint64_t>& ownedSlot(std::uintptr_t block) {
        auto& entry = ownedLeaves[block >> ownedLeafShift];
        auto* leaf = entry.load(std::memory_order_relaxed);
        if (leaf == nullptr) {
            ownedChecked[block >> ownedLeafShift].store(new std::atomic<std::uint64_t>[ownedLeafBlocks](), std::memory_order_release);
            leaf = new std::atomic<std::uint64_t>[ownedLeafBlocks]();
            entry.store(leaf, std::memory_order_release);
        }
        return leaf[(block / ownedBlockBytes) & (ownedLeafBlocks - 1)];
    }

    static constexpr std::size_t ownedPagesPerBlock = ownedBlockBytes / subpageBytes;

    // The allocation each 4 KiB page of an armed block belongs to (0: not mapped, proved by the word
    // alone: every path that makes it present moves the word first).
    using OwnedLayout = std::array<std::uintptr_t, ownedPagesPerBlock>;
    struct OwnedPlan {
        OwnedLayout layout{};
        bool privatePages = false;
        bool mapped = false;
        bool unmapped = false;
    };
    struct OwnedRun {
        std::uintptr_t begin;
        std::uintptr_t end;
        std::uintptr_t allocation;
    };

    static bool overlapsRange(const std::multimap<std::uintptr_t, std::uintptr_t>& ranges, std::uintptr_t begin, std::uintptr_t end) {
        for (auto it = ranges.begin(); it != ranges.end() && it->first < end; ++it) {
            if (it->second > begin) return true;
        }
        return false;
    }

    static void eraseRange(std::multimap<std::uintptr_t, std::uintptr_t>& ranges, std::uintptr_t begin, std::uintptr_t end) {
        for (auto it = ranges.lower_bound(begin); it != ranges.end() && it->first == begin; ++it) {
            if (it->second != end) continue;
            ranges.erase(it);
            return;
        }
    }

    // Appends the mapped pages of `block` inside [from, to) to `runs`, the last run extended while the
    // pages are contiguous and in one allocation (GetWriteWatch cannot span two).
    static void appendRuns(std::vector<OwnedRun>& runs, std::uintptr_t block, const OwnedLayout& layout, std::uintptr_t from, std::uintptr_t to) {
        for (std::size_t page = 0; page < ownedPagesPerBlock; ++page) {
            const auto at = block + page * subpageBytes;
            if (at < from || at >= to || layout[page] == 0) continue;
            if (!runs.empty() && runs.back().end == at && runs.back().allocation == layout[page]) runs.back().end = at + subpageBytes;
            else runs.push_back({at, at + subpageBytes, layout[page]});
        }
    }

    // Whether a page of [begin, end), inside one allocation, was written since its bit was last reset;
    // nothing when the range is not write-watched memory.
    static std::optional<bool> queryWatch(std::uintptr_t begin, std::uintptr_t end) {
        ULONG_PTR reported = 1;
        void* page = nullptr;
        DWORD granularity = 0;
        if (GetWriteWatch(0, reinterpret_cast<void*>(begin), end - begin, &page, &reported, &granularity) != 0) return std::nullopt;
        return reported != 0;
    }

    // Arming's reset of [begin, end) (pages of blocks not armed yet): the written pages it takes are
    // kept in ownedPending for the next collects.
    void takeWrites(std::uintptr_t begin, std::uintptr_t end) {
        std::array<void*, 256> found;
        for (auto cursor = begin; cursor < end;) {
            ULONG_PTR reported = found.size();
            DWORD granularity = 0;
            if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, reinterpret_cast<void*>(cursor), end - cursor, found.data(), &reported, &granularity) != 0) fail("take owned guest writes");
            for (ULONG_PTR i = 0; i < reported; ++i) {
                const auto page = reinterpret_cast<std::uintptr_t>(found[i]);
                ownedPending.insert(page);
                forgetClean(page, page + subpageBytes);
            }
            ownedTaken.fetch_add(reported, std::memory_order_relaxed);
            if (reported < found.size()) break;
            cursor = reinterpret_cast<std::uintptr_t>(found[reported - 1]) + subpageBytes;
        }
    }

    // After a collect's GetWriteWatch over [first, stop): `pages` holds the `reported` written pages
    // in address order and has room for `room`. A clearing collect moves the word of every armed
    // block whose written page it took. The pages ArmOwned took in the range join the report (and
    // leave ownedPending when clearing); written pages past `room` wait in ownedPending instead.
    // Returns the number of pages reported.
    std::size_t watchTaken(void** pages, std::size_t reported, std::size_t room, std::uintptr_t first, std::uintptr_t stop, bool clear) {
        if (clear && ownedLive != 0) {
            for (std::size_t i = 0; i < reported; ++i) disownWritten(reinterpret_cast<std::uintptr_t>(pages[i]));
        }
        if (ownedPending.empty()) return reported;
        const auto low = ownedPending.lower_bound(first & ~std::uintptr_t{subpageBytes - 1});
        const auto high = ownedPending.lower_bound(stop);
        if (low == high) return reported;
        thread_local std::vector<std::uintptr_t> merged;
        merged.clear();
        std::size_t index = 0;
        for (auto it = low; index < reported || it != high;) {
            const auto next = index < reported ? reinterpret_cast<std::uintptr_t>(pages[index]) : UINTPTR_MAX;
            const auto held = it != high ? *it : UINTPTR_MAX;
            merged.push_back(std::min(next, held));
            if (next <= held) ++index;
            if (held <= next) ++it;
        }
        const auto kept = std::min(room, merged.size());
        if (clear) {
            for (std::size_t k = 0; k < merged.size(); ++k) {
                if (k < kept) ownedPending.erase(merged[k]);
                else ownedPending.insert(merged[k]);
            }
        }
        for (std::size_t k = 0; k < kept; ++k) pages[k] = reinterpret_cast<void*>(merged[k]);
        return kept;
    }

    // Whether a mapped page of an armed block was written since it was armed (or the query failed).
    bool blockWritten(std::uintptr_t block) {
        const auto found = ownedLayouts.find(block);
        if (found == ownedLayouts.end()) return true;
        thread_local std::vector<OwnedRun> runs;
        runs.clear();
        appendRuns(runs, block, found->second, block, block + ownedBlockBytes);
        for (const auto& run : runs) {
            const auto written = queryWatch(run.begin, run.end);
            if (!written.has_value() || *written) return true;
        }
        return false;
    }

    void forgetPending(std::uintptr_t start, std::uintptr_t end) {
        if (ownedPending.empty()) return;
        ownedPending.erase(ownedPending.lower_bound(start), ownedPending.lower_bound(end));
    }

    enum class OwnedAnswer : std::uint8_t { Holds, Written, Moved, Unsure };

    // OwnedUnchanged's queries and the words after them. Written: a page of the range has its write
    // bit (a store after the arming, which reset the bits); Moved: a word moved or the range is not
    // armed; Unsure: a query failed while the words held.
    OwnedAnswer ownedQuery(std::uintptr_t address, std::size_t bytes, std::uintptr_t first, std::uintptr_t stop, const std::uint64_t* words) {
        thread_local std::vector<OwnedRun> runs;
        runs.clear();
        const auto from = address & ~std::uintptr_t{subpageBytes - 1};
        const auto to = (address + bytes + subpageBytes - 1) & ~std::uintptr_t{subpageBytes - 1};
        {
            std::lock_guard layouts(ownedMutex);
            for (auto block = first; block < stop; block += ownedBlockBytes) {
                const auto found = ownedLayouts.find(block);
                if (found == ownedLayouts.end()) return OwnedAnswer::Moved;
                appendRuns(runs, block, found->second, from, to);
            }
        }
        for (const auto& run : runs) {
            ownedQueries.fetch_add(1, std::memory_order_relaxed);
            const auto written = queryWatch(run.begin, run.end);
            if (!written.has_value()) return wordsHold(first, stop, words) ? OwnedAnswer::Unsure : OwnedAnswer::Moved;
            if (*written) {
                ownedDirty.fetch_add(1, std::memory_order_relaxed);
                return OwnedAnswer::Written;
            }
        }
        return wordsHold(first, stop, words) ? OwnedAnswer::Holds : OwnedAnswer::Moved;
    }

    // Brackets a reset of write bits and the word moves it implies (see OwnedUnchanged).
    struct ResetWindow {
        std::atomic<std::uint64_t>& resets;
        bool open;
        ResetWindow(std::atomic<std::uint64_t>& resets, bool clear) : resets(resets), open(clear) {
            if (open) resets.fetch_add(1, std::memory_order_seq_cst);
        }
        ~ResetWindow() {
            if (open) resets.fetch_add(1, std::memory_order_seq_cst);
        }
        ResetWindow(const ResetWindow&) = delete;
        ResetWindow& operator=(const ResetWindow&) = delete;
    };

    bool wordsHold(std::uintptr_t first, std::uintptr_t stop, const std::uint64_t* words) const {
        std::size_t index = 0;
        for (auto block = first; block < stop; block += ownedBlockBytes, ++index) {
            const auto* leaf = ownedLeaves[block >> ownedLeafShift].load(std::memory_order_acquire);
            if (leaf == nullptr) return false;
            const auto word = leaf[(block / ownedBlockBytes) & (ownedLeafBlocks - 1)].load(std::memory_order_seq_cst);
            if ((word & 1u) == 0 || word != words[index]) return false;
        }
        return true;
    }

    // OwnedUnchanged's answer from queries (see there).
    bool unchangedNow(std::uintptr_t address, std::size_t bytes, std::uintptr_t first, std::uintptr_t stop, const std::uint64_t* words) {
        const auto before = ownedResets.load(std::memory_order_seq_cst);
        if (!wordsHold(first, stop, words)) return false;
        if ((before & 1u) == 0) {
            const auto answer = ownedQuery(address, bytes, first, stop, words);
            if (answer == OwnedAnswer::Written || answer == OwnedAnswer::Moved) return false;
            if (answer == OwnedAnswer::Holds && ownedResets.load(std::memory_order_seq_cst) == before) return true;
        }
        ownedLocked.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard lock(mutex);
        return ownedQuery(address, bytes, first, stop, words) == OwnedAnswer::Holds;
    }

    // Every block checked at a clock value >= mark (each check read first, acquire) and still armed
    // with its word (read after).
    bool checkedSince(std::uintptr_t first, std::uintptr_t stop, const std::uint64_t* words, std::uint64_t mark) const {
        for (auto block = first; block < stop; block += ownedBlockBytes) {
            const auto* checked = ownedChecked[block >> ownedLeafShift].load(std::memory_order_acquire);
            if (checked == nullptr || checked[(block / ownedBlockBytes) & (ownedLeafBlocks - 1)].load(std::memory_order_acquire) < mark) return false;
        }
        return wordsHold(first, stop, words);
    }

    // Stores `clock` as the blocks' check (release, after the queries and the words they were
    // answered with). A smaller value stored over a larger one only asks for a query sooner.
    void markChecked(std::uintptr_t first, std::uintptr_t stop, std::uint64_t clock) {
        for (auto block = first; block < stop; block += ownedBlockBytes) {
            auto* checked = ownedChecked[block >> ownedLeafShift].load(std::memory_order_acquire);
            if (checked != nullptr) checked[(block / ownedBlockBytes) & (ownedLeafBlocks - 1)].store(clock, std::memory_order_release);
        }
    }

    // Every 16 KiB page a single resident view (write-watched when watchResident), else 4 KiB pages
    // that are private commits (with `allowPrivate`; write watching is checked by ArmOwned) or reserved
    // placeholders. Views and pages may be read-write or read-only (with or without execute). Pinned
    // ranges (guest stacks, whose pages libkernel commits itself at thread exit) are refused whatever
    // their kind.
    OwnedRefusal ownable(std::uintptr_t block, bool allowPrivate, OwnedPlan& plan) {
        const auto end = block + ownedBlockBytes;
        std::uintptr_t regionBegin = 0;
        std::uintptr_t regionEnd = 0;
        std::uintptr_t regionAllocation = 0;
        for (auto at = block; at < end; at += pageBytes) {
            const auto found = views.find(at);
            if (found != views.end()) {
                const auto& view = found->second;
                if (!view.resident || view.page->aliases.size() != 1 || view.viewBytes != pageBytes) return OwnedRefusal::Shared;
                if (!watchResident) return OwnedRefusal::Unwatched;
                if (!ownedReadable(view.protection)) return OwnedRefusal::NotWritable;
                if (view.hostWrites != 0) return OwnedRefusal::HostWrite;
                if (view.hostImports != 0 || view.page->pins != 0) return OwnedRefusal::Pinned;
                if (at < regionBegin || at + pageBytes > regionEnd) {
                    const auto memory = query(at);
                    if (memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE) return OwnedRefusal::NotWritable;
                    regionBegin = reinterpret_cast<std::uintptr_t>(memory.BaseAddress);
                    regionEnd = regionBegin + memory.RegionSize;
                    regionAllocation = reinterpret_cast<std::uintptr_t>(memory.AllocationBase);
                }
                for (auto page = at; page < at + pageBytes; page += subpageBytes) plan.layout[(page - block) / subpageBytes] = regionAllocation;
                plan.mapped = true;
                continue;
            }
            for (auto cursor = at; cursor < at + pageBytes;) {
                const auto memory = query(cursor);
                const auto stop = std::min(at + pageBytes, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
                if (memory.State == MEM_COMMIT) {
                    if (memory.Type != MEM_PRIVATE || (memory.Protect & ~0xffu) != 0 || !ownedReadable(memory.Protect)) return OwnedRefusal::NotWritable;
                    if (!allowPrivate) return OwnedRefusal::Private;
                    for (auto page = cursor; page < stop; page += subpageBytes) plan.layout[(page - block) / subpageBytes] = reinterpret_cast<std::uintptr_t>(memory.AllocationBase);
                    plan.privatePages = true;
                    plan.mapped = true;
                } else if (memory.State == MEM_RESERVE) {
                    plan.unmapped = true;
                } else {
                    return OwnedRefusal::NotWritable;
                }
                cursor = stop;
            }
        }
        if (overlapsRange(pinnedRanges, block, end)) return OwnedRefusal::Pinned;
        if (plan.privatePages && overlapsRange(hostAccess, block, end)) return OwnedRefusal::HostWrite;
        return OwnedRefusal::None;
    }

    static bool ownedReadable(DWORD protection) {
        return protection == PAGE_READWRITE || protection == PAGE_EXECUTE_READWRITE || protection == PAGE_READONLY || protection == PAGE_EXECUTE_READ;
    }

    // Disarms an armed block: its word moves, counted in `reason`.
    void disownBlock(std::uintptr_t block, std::atomic<std::uint64_t>& reason) {
        auto* leaf = ownedLeaves[block >> ownedLeafShift].load(std::memory_order_relaxed);
        if (leaf == nullptr) return;
        auto& slot = leaf[(block / ownedBlockBytes) & (ownedLeafBlocks - 1)];
        const auto word = slot.load(std::memory_order_relaxed);
        if ((word & 1u) == 0) return;
        slot.store(word + 1, std::memory_order_seq_cst);
        --ownedLive;
        reason.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard layouts(ownedMutex);
        ownedLayouts.erase(block);
    }

    void disownWritten(std::uintptr_t page) {
        if (page < ownedAddressLimit) disownBlock(page & ~std::uintptr_t{ownedBlockBytes - 1}, ownedCollected);
    }

    void disownRange(std::uintptr_t start, std::uintptr_t end, std::atomic<std::uint64_t>* reason = nullptr) {
        if (guardedSubpages.load(std::memory_order_acquire) != 0 && end > start) {
            const auto guarded = accessGuards.lower_bound(start & ~std::uintptr_t{subpageBytes - 1});
            if (guarded != accessGuards.end() && guarded->first < end) fail("a mapping, protection or host-access change over an access-guarded page");
        }
        if (ownedLive == 0 || end <= start) return;
        end = std::min(end, ownedAddressLimit);
        for (auto block = start & ~std::uintptr_t{ownedBlockBytes - 1}; block < end && ownedLive != 0;) {
            if (ownedLeaves[block >> ownedLeafShift].load(std::memory_order_relaxed) == nullptr) {
                block = ((block >> ownedLeafShift) + 1) << ownedLeafShift;
                continue;
            }
            disownBlock(block, reason != nullptr ? *reason : ownedDisowned);
            block += ownedBlockBytes;
        }
    }

    Window open(HANDLE section, std::uint64_t offset, std::size_t bytes) {
        const auto lead = offset % granularity;
        Window window;
        window.view = map(section, GetCurrentProcess(), nullptr, offset - lead, lead + bytes, 0, PAGE_READWRITE, nullptr, 0);
        if (window.view == nullptr) fail("open a window on direct memory");
        window.bytes = static_cast<unsigned char*>(window.view) + lead;
        return window;
    }

    void close(Window& window) {
        if (window.view != nullptr && !unmap(GetCurrentProcess(), window.view, 0)) fail("close a window on direct memory");
        window = {};
    }

    void commitResident(std::uintptr_t base, std::size_t bytes) {
        const DWORD flags = MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | (watchResident ? MEM_WRITE_WATCH : 0);
        if (!allocate(GetCurrentProcess(), reinterpret_cast<void*>(base), bytes, flags, PAGE_READWRITE, nullptr, 0)) fail("replace guest placeholder with resident direct memory");
    }

    static std::size_t hostPages(std::size_t bytes) {
        return (bytes + subpageBytes - 1) & ~(subpageBytes - 1);
    }

    void commitSection(HANDLE section, std::uint64_t offset, std::size_t bytes) {
        auto window = open(section, offset, bytes);
        const auto memory = query(reinterpret_cast<std::uintptr_t>(window.bytes));
        if ((memory.State != MEM_COMMIT || memory.RegionSize < bytes) && !VirtualAlloc(window.bytes, bytes, MEM_COMMIT, PAGE_READWRITE)) {
            const auto error = GetLastError();
            close(window);
            throw std::system_error(static_cast<int>(error), std::system_category(), "commit direct memory pages");
        }
        close(window);
    }

    void mapShared(std::uintptr_t base, HANDLE section, std::uint64_t offset, DWORD protection, DWORD viewProtection = PAGE_EXECUTE_READWRITE, std::size_t sectionBytes = pageBytes, bool committed = false) {
        void* page = reinterpret_cast<void*>(base);
        const auto viewBytes = hostPages(sectionBytes);
        if (viewBytes != pageBytes) split(base, viewBytes);
        if (!map(section, GetCurrentProcess(), page, offset, sectionBytes, MEM_REPLACE_PLACEHOLDER, viewProtection, nullptr, 0)) fail("map shared guest page");
        if (!committed && query(base).State == MEM_RESERVE && !VirtualAlloc(page, viewBytes, MEM_COMMIT, PAGE_EXECUTE_READWRITE)) fail("commit shared guest page");
        DWORD previous;
        if (!VirtualProtect(page, viewBytes, protection, &previous)) fail("protect shared guest page");
        if (viewBytes == pageBytes) return;
        void* zeros = reinterpret_cast<void*>(base + viewBytes);
        if (!allocate(GetCurrentProcess(), zeros, pageBytes - viewBytes, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0)) fail("replace guest placeholder past the end of a file with zero pages");
        if (!VirtualProtect(zeros, pageBytes - viewBytes, protection, &previous)) fail("protect guest pages past the end of a file");
    }

    static void restored(Physical& slot) {
        if (slot.saved) slot.saved->Decommit(slot.savedOffset);
        slot.saved.reset();
    }

    void mapRun(std::uintptr_t base, std::size_t bytes, HANDLE section, std::uint64_t offset) {
        if (!map(section, GetCurrentProcess(), reinterpret_cast<void*>(base), offset, bytes, MEM_REPLACE_PLACEHOLDER, PAGE_EXECUTE_READWRITE, nullptr, 0)) fail("map a run of shared guest pages");
    }

    void breakView(std::uintptr_t at) {
        const auto memory = query(at);
        if (memory.State == MEM_FREE || memory.Type != MEM_MAPPED) return;
        const auto start = reinterpret_cast<std::uintptr_t>(memory.AllocationBase);
        if (start == at) return;
        const auto first = views.find(start);
        if (first == views.end()) return;
        struct Part {
            std::uintptr_t base;
            std::uintptr_t end;
            DWORD protection;
            bool committed;
        };
        std::vector<Part> parts;
        auto stop = start;
        for (;;) {
            const auto part = query(stop);
            if (part.AllocationBase != memory.AllocationBase) break;
            const auto partEnd = reinterpret_cast<std::uintptr_t>(part.BaseAddress) + part.RegionSize;
            parts.push_back({stop, partEnd, part.Protect, part.State == MEM_COMMIT});
            stop = partEnd;
        }
        const auto section = first->second.section;
        const auto offset = first->second.offset;
        if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(start), MEM_PRESERVE_PLACEHOLDER)) fail("unmap a run of shared guest pages to split it");
        split(start, at - start);
        mapRun(start, at - start, section->handle, offset);
        mapRun(at, stop - at, section->handle, offset + (at - start));
        for (const auto& part : parts) {
            if (!part.committed) continue;
            for (const auto [from, to] : {std::pair{part.base, std::min(part.end, at)}, std::pair{std::max(part.base, at), part.end}}) {
                DWORD previous;
                if (from < to && !VirtualProtect(reinterpret_cast<void*>(from), to - from, part.protection, &previous)) fail("protect a split run of shared guest pages");
            }
        }
    }

    static bool blank(std::uintptr_t page) {
        const auto* words = reinterpret_cast<const std::uint64_t*>(page);
        for (std::size_t i = 0; i < pageBytes / sizeof(std::uint64_t); ++i) {
            if (words[i] != 0) return false;
        }
        return true;
    }

    std::map<std::uintptr_t, Run>::iterator residentRun(std::uintptr_t key, std::uint64_t offset) {
        const auto next = residentIndex.upper_bound(std::make_pair(key, offset));
        if (next == residentIndex.begin()) return residentRuns.end();
        const auto& [first, run] = *std::prev(next);
        return first.first == key && offset - first.second < run->second.end - run->first ? run : residentRuns.end();
    }

    std::map<std::pair<std::uintptr_t, std::uint64_t>, std::map<std::uintptr_t, Run>::iterator>::iterator indexEntry(std::map<std::uintptr_t, Run>::iterator run) {
        const auto found = residentIndex.find(std::make_pair(run->second.key, run->second.offset));
        return found != residentIndex.end() && found->second == run ? found : residentIndex.end();
    }

    void demote(std::uintptr_t address, bool alias = false) {
        const auto run = std::prev(residentRuns.upper_bound(address));
        const auto start = run->first;
        const auto stop = run->second.end;
        disownRange(start, stop);
        forgetPending(start, stop);
        const auto entry = indexEntry(run);
        const bool indexed = entry != residentIndex.end();
        std::vector<std::pair<std::shared_ptr<SharedPage>, std::shared_ptr<Physical>>> created;
        created.reserve((stop - start) / pageBytes);
        std::map<std::pair<std::uintptr_t, std::uint64_t>, std::shared_ptr<Physical>> staged;
        for (auto page = start; page < stop; page += pageBytes) {
            const auto& view = views.at(page);
            auto shared = std::make_shared<SharedPage>();
            shared->aliases.push_back(page);
            shared->pins = view.pins;
            shared->comparedWhenShared = view.comparedWhenShared;
            auto slot = std::make_shared<Physical>();
            slot->page = shared;
            slot->comparedWhenShared = view.comparedWhenShared;
            if (indexed) staged.emplace(std::make_pair(run->second.key, view.offset), slot);
            created.emplace_back(std::move(shared), std::move(slot));
        }
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(start), stop - start, PAGE_READONLY, &previous)) fail("fence resident guest pages");
        const auto section = views.at(start).section;
        auto window = open(section->handle, views.at(start).offset, stop - start);
        if (query(reinterpret_cast<std::uintptr_t>(window.bytes)).State == MEM_RESERVE && !VirtualAlloc(window.bytes, stop - start, MEM_COMMIT, PAGE_READWRITE)) fail("commit direct memory pages");
        std::memcpy(window.bytes, reinterpret_cast<const void*>(start), stop - start);
        close(window);
        if (!VirtualFree(reinterpret_cast<void*>(start), stop - start, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release resident guest pages");
        physical.merge(staged);
        for (auto& [key, slot] : staged) physical.find(key)->second = std::move(slot);
        for (auto page = start; page < stop; page += pageBytes) {
            auto& view = views.at(page);
            split(page, pageBytes);
            mapShared(page, view.section->handle, view.offset, view.protection);
            auto& [shared, slot] = created[(page - start) / pageBytes];
            view.page = std::move(shared);
            view.slot = std::move(slot);
            view.slot->stored = true;
            view.resident = false;
            view.aliasDemoted = alias;
            view.unreported = view.page->comparedWhenShared ? allSubpages : std::uint8_t{0};
            view.seen.fill(0);
            view.armed = 0;
            view.collected = 0;
            for (std::size_t sub = 0; sub < subpages; ++sub) {
                if (runCollected(run->second, ((page - start) >> 12) + sub)) view.collected |= static_cast<std::uint8_t>(1u << sub);
            }
            invalidate(*view.page);
        }
        if (indexed) residentIndex.erase(entry);
        residentRuns.erase(run);
    }

    void breakRun(std::uintptr_t at) {
        const auto next = residentRuns.upper_bound(at);
        if (next == residentRuns.begin()) return;
        const auto run = std::prev(next);
        if (at > run->first && at < run->second.end) demote(at);
    }

    void evict(std::map<std::uintptr_t, Run>::iterator run) {
        const auto start = run->first;
        const auto stop = run->second.end;
        disownRange(start, stop);
        forgetPending(start, stop);
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(start), stop - start, PAGE_READONLY, &previous)) fail("open resident guest pages");
        const auto entry = indexEntry(run);
        const bool indexed = entry != residentIndex.end();
        std::shared_ptr<Saved> saved;
        for (auto page = start; page < stop; page += pageBytes) {
            const auto found = views.find(page);
            const auto& view = found->second;
            const bool written = indexed && !blank(page);
            if (written || (indexed && view.comparedWhenShared)) {
                auto& slot = physical[std::make_pair(run->second.key, view.offset)];
                if (!slot) slot = std::make_shared<Physical>();
                slot->comparedWhenShared = view.comparedWhenShared;
                if (written) {
                    if (!saved) saved = savedMemory(stop - start);
                    saved->Commit(page - start);
                    std::memcpy(saved->bytes + (page - start), reinterpret_cast<const void*>(page), pageBytes);
                    slot->saved = saved;
                    slot->savedOffset = page - start;
                }
            }
            views.erase(found);
        }
        if (indexed) residentIndex.erase(entry);
        forget(freshRanges, start, stop);
        residentRuns.erase(run);
    }

    std::shared_ptr<Saved> savedMemory(std::size_t bytes) {
        for (const auto& saved : savedPool) {
            if (saved.use_count() == 1 && saved->length == bytes) return saved;
        }
        constexpr std::size_t limit = 32u * 1024u * 1024u;
        const auto reserved = (bytes + granularity - 1) / granularity * granularity;
        for (auto it = savedPool.begin(); reserved <= limit && savedPoolBytes > limit - reserved && it != savedPool.end();) {
            if (it->use_count() != 1) {
                ++it;
                continue;
            }
            savedPoolBytes -= ((*it)->length + granularity - 1) / granularity * granularity;
            it = savedPool.erase(it);
        }
        auto saved = std::make_shared<Saved>(bytes);
        if (reserved <= limit && savedPoolBytes <= limit - reserved) {
            savedPool.push_back(saved);
            savedPoolBytes += reserved;
        }
        return saved;
    }

    static void forget(Ranges& ranges, std::uintptr_t start, std::uintptr_t end) {
        auto it = ranges.lower_bound(start);
        if (it != ranges.begin() && std::prev(it)->second > start) --it;
        while (it != ranges.end() && it->first < end) {
            const auto first = it->first;
            const auto last = it->second;
            it = ranges.erase(it);
            if (first < start) ranges.emplace(first, start);
            if (last > end) it = ranges.emplace(end, last).first;
        }
    }

    static void remember(Ranges& ranges, std::uintptr_t start, std::uintptr_t end) {
        auto it = ranges.lower_bound(start);
        if (it != ranges.begin() && std::prev(it)->second >= start) --it;
        while (it != ranges.end() && it->first <= end) {
            start = std::min(start, it->first);
            end = std::max(end, it->second);
            it = ranges.erase(it);
        }
        ranges.emplace(start, end);
    }

    void forgetClean(std::uintptr_t start, std::uintptr_t end) {
        forget(cleanRanges, start, end);
    }

    void rememberClean(std::uintptr_t start, std::uintptr_t end) {
        remember(cleanRanges, start, end);
    }

    void invalidate(SharedPage& page, std::uint8_t mask = allSubpages) {
        for (std::size_t sub = 0; sub < subpages; ++sub) {
            if ((mask & (1u << sub)) == 0) continue;
            ++page.generation[sub];
            for (const auto alias : page.aliases) forgetClean(alias + sub * subpageBytes, alias + (sub + 1) * subpageBytes);
        }
    }

    void invalidate(std::uintptr_t base, const View& view, std::uint8_t mask = allSubpages) {
        if (view.page) {
            invalidate(*view.page, mask);
            return;
        }
        for (std::size_t sub = 0; sub < subpages; ++sub) {
            if ((mask & (1u << sub)) != 0) forgetClean(base + sub * subpageBytes, base + (sub + 1) * subpageBytes);
        }
    }

    static std::uint8_t subpagesOf(std::uintptr_t base, std::uintptr_t address, std::uintptr_t end) {
        const auto from = std::max(address, base);
        const auto to = std::min(end, base + pageBytes);
        if (to <= from) return 0;
        const auto first = (from - base) / subpageBytes;
        const auto last = (to - 1 - base) / subpageBytes;
        return static_cast<std::uint8_t>(((1u << (last + 1)) - 1) & ~((1u << first) - 1));
    }

    static void protectSubpages(std::uintptr_t base, std::uint8_t mask, DWORD protection, const char* operation, std::size_t viewBytes = pageBytes) {
        for (std::size_t sub = 0; sub < subpages;) {
            if ((mask & (1u << sub)) == 0) {
                ++sub;
                continue;
            }
            auto stop = sub + 1;
            while (stop < subpages && (mask & (1u << stop)) != 0 && stop * subpageBytes != viewBytes) ++stop;
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(base + sub * subpageBytes), (stop - sub) * subpageBytes, protection, &previous)) fail(operation);
            sub = stop;
        }
    }

    bool sameSection(HANDLE first, HANDLE second) const {
        return compare != nullptr && compare(first, second);
    }

    static bool writable(DWORD protection) {
        return protection == PAGE_READWRITE || protection == PAGE_EXECUTE_READWRITE;
    }
    using AllocateFunction = PVOID (WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using MapFunction = PVOID (WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using UnmapFunction = BOOL (WINAPI*)(HANDLE, PVOID, ULONG);
    using CompareFunction = BOOL (WINAPI*)(HANDLE, HANDLE);

    WindowsMappings() {
        const auto module = GetModuleHandleW(L"KernelBase.dll");
        if (!module) fail("load Windows memory API");
        allocate = reinterpret_cast<AllocateFunction>(GetProcAddress(module, "VirtualAlloc2"));
        map = reinterpret_cast<MapFunction>(GetProcAddress(module, "MapViewOfFile3"));
        unmap = reinterpret_cast<UnmapFunction>(GetProcAddress(module, "UnmapViewOfFile2"));
        if (!allocate || !map || !unmap) throw std::runtime_error("Windows placeholder memory APIs are required");
        compare = reinterpret_cast<CompareFunction>(GetProcAddress(module, "CompareObjectHandles"));
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        granularity = system.dwAllocationGranularity;
    }

    [[noreturn]] static void fail(const char* operation) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation);
    }

    static MEMORY_BASIC_INFORMATION query(std::uintptr_t address) {
        MEMORY_BASIC_INFORMATION memory{};
#ifdef APS5_COUNT_MAPPING_QUERIES
        queries.fetch_add(1, std::memory_order_relaxed);
#endif
        if (VirtualQuery(reinterpret_cast<void*>(address), &memory, sizeof(memory)) != sizeof(memory)) fail("query guest memory");
        return memory;
    }

    static std::uintptr_t split(std::uintptr_t address, std::size_t bytes) {
        auto memory = query(address);
        memory = query(reinterpret_cast<std::uintptr_t>(memory.AllocationBase));
        if (memory.State != MEM_RESERVE) throw std::runtime_error("guest mapping requires a placeholder");
        const auto base = reinterpret_cast<std::uintptr_t>(memory.BaseAddress);
        if (address != base) {
            if (!VirtualFree(reinterpret_cast<void*>(base), address - base, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split guest placeholder prefix");
            memory = query(address);
        }
        if (memory.RegionSize < bytes) throw std::runtime_error("guest placeholder is too small");
        if (memory.RegionSize != bytes && !VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split guest placeholder suffix");
        return address + memory.RegionSize;
    }

    void reset(std::uintptr_t address, std::size_t bytes) {
        collectableRanges.clear();
        const auto end = address + bytes;
        disownRange(address, end);
        forgetPending(address, end);
        privateAllocations.erase(privateAllocations.lower_bound(address), privateAllocations.lower_bound(end));
        forgetClean(address, end);
        breakRun(address);
        breakRun(end);
        breakView(address);
        breakView(end);
        if (const auto run = residentRuns.find(address); run != residentRuns.end() && run->second.end == end) {
            const auto memory = query(address);
            if (reinterpret_cast<std::uintptr_t>(memory.AllocationBase) != address) throw std::runtime_error("cannot release part of a host allocation");
            if (memory.Type != MEM_PRIVATE) throw std::runtime_error("resident guest mapping is not private memory");
            auto allocationEnd = address;
            do {
                const auto part = query(allocationEnd);
                if (part.AllocationBase != memory.AllocationBase) break;
                allocationEnd = reinterpret_cast<std::uintptr_t>(part.BaseAddress) + part.RegionSize;
            } while (allocationEnd < end);
            if (allocationEnd != end || query(end).AllocationBase == memory.AllocationBase) throw std::runtime_error("guest release truncates a host allocation");
            evict(run);
            if (!VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release private guest memory");
            return;
        }
        for (auto cursor = address; cursor < end;) {
            const auto memory = query(cursor);
            if (memory.State == MEM_RESERVE) {
                cursor = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
                continue;
            }
            if (reinterpret_cast<std::uintptr_t>(memory.AllocationBase) != cursor) throw std::runtime_error("cannot release part of a host allocation");
            auto allocationEnd = cursor;
            do {
                const auto part = query(allocationEnd);
                if (part.AllocationBase != memory.AllocationBase) break;
                allocationEnd = reinterpret_cast<std::uintptr_t>(part.BaseAddress) + part.RegionSize;
            } while (allocationEnd < end);
            if (allocationEnd > end || query(allocationEnd).AllocationBase == memory.AllocationBase) throw std::runtime_error("guest release truncates a host allocation");
            if (memory.Type == MEM_MAPPED) {
                if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(cursor), MEM_PRESERVE_PLACEHOLDER)) fail("unmap shared guest page");
                for (auto found = views.lower_bound(cursor); found != views.end() && found->first < allocationEnd;) {
                    if (found->second.page != nullptr) std::erase(found->second.page->aliases, found->first);
                    found = views.erase(found);
                }
                forget(freshRanges, cursor, allocationEnd);
            } else if (memory.Type == MEM_PRIVATE) {
                if (const auto run = residentRuns.find(cursor); run != residentRuns.end()) evict(run);
                if (!VirtualFree(reinterpret_cast<void*>(cursor), allocationEnd - cursor, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release private guest memory");
            } else {
                throw std::runtime_error("unsupported guest mapping type");
            }
            cursor = allocationEnd;
        }
        const auto last = query(reinterpret_cast<std::uintptr_t>(query(end - 1).AllocationBase));
        const auto lastBase = reinterpret_cast<std::uintptr_t>(last.BaseAddress);
        if (lastBase + last.RegionSize > end) split(lastBase, end - lastBase);
        const auto first = query(address);
        split(address, std::min(bytes, reinterpret_cast<std::uintptr_t>(first.BaseAddress) + first.RegionSize - address));
        if (query(address).RegionSize != bytes && !VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) fail("coalesce guest placeholders");
    }

    void* mapAliasRuns(std::uintptr_t address, std::size_t bytes) {
        auto* reserved = allocate(GetCurrentProcess(), nullptr, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
        if (reserved == nullptr) fail("reserve a read-write alias of shared guest memory");
        const auto base = reinterpret_cast<std::uintptr_t>(reserved);
        try {
            for (std::size_t done = 0; done < bytes;) {
                const auto& first = views.at(address + done);
                auto run = pageBytes;
                while (done + run < bytes) {
                    const auto& next = views.at(address + done + run);
                    if (next.offset != first.offset + run || (next.section != first.section && !sameSection(next.section->handle, first.section->handle))) break;
                    run += pageBytes;
                }
                split(base + done, run);
                if (!map(first.section->handle, GetCurrentProcess(), reinterpret_cast<void*>(base + done), first.offset, run, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0)) fail("map a run of a read-write alias of shared guest memory");
                done += run;
            }
        } catch (...) {
            releaseAlias(base, bytes);
            throw;
        }
        splitAliases.emplace(base, bytes);
        return reserved;
    }

    DWORD releaseAlias(std::uintptr_t base, std::size_t bytes) {
        DWORD error = ERROR_SUCCESS;
        for (auto cursor = base; cursor < base + bytes;) {
            const auto memory = query(cursor);
            const auto released = memory.State == MEM_RESERVE ? VirtualFree(memory.BaseAddress, 0, MEM_RELEASE) : unmap(GetCurrentProcess(), memory.BaseAddress, 0);
            if (!released && error == ERROR_SUCCESS) error = GetLastError();
            cursor = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
        }
        return error;
    }

    Ranges cleanRanges;
    Ranges collectableRanges;
    Ranges privateAllocations;
#ifdef APS5_COUNT_MAPPING_QUERIES
    std::uint64_t collectSystemCalls = 0;
    std::uint64_t collectPieces = 0;
#endif
    Ranges freshRanges;
    std::map<std::uintptr_t, View> views;

    using Views = std::map<std::uintptr_t, View>;

    Views::iterator viewAt(std::uintptr_t base, Views::iterator& position) {
        if (position != views.end() && position->first < base) {
            ++position;
            if (position != views.end() && position->first < base) position = views.lower_bound(base);
        }
        return position != views.end() && position->first == base ? position : views.end();
    }
    std::map<std::pair<std::uintptr_t, std::uint64_t>, std::shared_ptr<Physical>> physical;
    std::map<std::uintptr_t, Run> residentRuns;
    std::map<std::pair<std::uintptr_t, std::uint64_t>, std::map<std::uintptr_t, Run>::iterator> residentIndex;
    std::vector<std::shared_ptr<Saved>> savedPool;
    std::size_t savedPoolBytes = 0;
    std::map<std::uintptr_t, std::size_t> splitAliases;
    std::size_t granularity = 0x10000;
    bool watchResident = false;
    std::array<std::atomic<std::atomic<std::uint64_t>*>, ownedLeafCount> ownedLeaves{};
    std::array<std::atomic<std::atomic<std::uint64_t>*>, ownedLeafCount> ownedChecked{};
    std::atomic<std::uint64_t> ownedClock{1};
    std::atomic<std::uint64_t> ownedEpochHeld{0};
    std::atomic<std::uint64_t> ownedEpochQueried{0};
    std::size_t ownedLive = 0;
    std::uint32_t protectionsInFlight = 0;
    std::atomic<std::uint64_t> ownedArms{0};
    std::atomic<std::uint64_t> ownedArmedBlocks{0};
    std::atomic<std::uint64_t> ownedRefused{0};
    std::atomic<std::uint64_t> ownedFaults{0};
    std::atomic<std::uint64_t> ownedDisowned{0};
    std::atomic<std::uint64_t> ownedPrivateBlocks{0};
    std::atomic<std::uint64_t> ownedUnmappedBlocks{0};
    std::atomic<std::uint64_t> ownedPartialBlocks{0};
    std::array<std::atomic<std::uint64_t>, OwnedRefusalCount> ownedReasons{};
    std::atomic<std::uint64_t> ownedCollected{0};
    std::atomic<std::uint64_t> ownedTaken{0};
    std::atomic<std::uint64_t> ownedQueries{0};
    std::atomic<std::uint64_t> ownedDirty{0};
    std::unordered_map<std::uintptr_t, OwnedLayout> ownedLayouts;
    std::mutex ownedMutex;
    std::atomic<std::uint64_t> ownedResets{0};
    std::atomic<std::uint64_t> ownedLocked{0};
    std::set<std::uintptr_t> ownedPending;
    std::multimap<std::uintptr_t, std::uintptr_t> pinnedRanges;
    std::multimap<std::uintptr_t, std::uintptr_t> hostAccess;
#ifdef APS5_COUNT_MAPPING_QUERIES
    static inline std::atomic<std::uint64_t> queries{0};
#endif
    MappingMutex mutex;
    struct AccessGuard {
        DWORD protection = 0;
        std::uint64_t arm = 0;
    };
    std::map<std::uintptr_t, AccessGuard> accessGuards;
    std::uint64_t guardArms = 0;
    std::atomic<std::size_t> guardedSubpages{0};
    std::atomic<bool> guardsUsed{false};
    std::atomic<std::uint32_t> changesActive{0};
    std::atomic<void (*)(std::uintptr_t)> accessGuardObserver{nullptr};
    AllocateFunction allocate = nullptr;
    MapFunction map = nullptr;
    UnmapFunction unmap = nullptr;
    CompareFunction compare = nullptr;
};

}
#endif

#endif
