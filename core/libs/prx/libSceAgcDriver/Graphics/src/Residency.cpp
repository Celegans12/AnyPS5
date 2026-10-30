#include "prx/libSceAgcDriver/Graphics/include/Residency.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace AgcDriver::Graphics {

namespace {

thread_local const IResidencyOwner* cpuReadExempt = nullptr;

const void* ResidencyThreadToken() {
    thread_local const char token = 0;
    return &token;
}

}

void ResidencyMutex::lock() {
    if (owner.load(std::memory_order_relaxed) == ResidencyThreadToken()) {
        ++depth;
        return;
    }
    mutex.lock();
    owner.store(ResidencyThreadToken(), std::memory_order_relaxed);
    depth = 1;
}

void ResidencyMutex::unlock() {
    if (--depth != 0) return;
    owner.store(nullptr, std::memory_order_relaxed);
    mutex.unlock();
}

bool ResidencyMutex::HeldByThisThread() const {
    return owner.load(std::memory_order_relaxed) == ResidencyThreadToken();
}

std::uint32_t ResidencyMutex::ReleaseAll() {
    if (!HeldByThisThread()) return 0;
    const auto held = depth;
    depth = 0;
    owner.store(nullptr, std::memory_order_relaxed);
    mutex.unlock();
    return held;
}

void ResidencyMutex::Reacquire(std::uint32_t held) {
    if (held == 0) return;
    mutex.lock();
    owner.store(ResidencyThreadToken(), std::memory_order_relaxed);
    depth = held;
}

namespace {

struct PageSpan {
    std::uint64_t first = 0;
    std::uint64_t last = 0;
    bool empty = true;
};

PageSpan PagesOf(std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return {};
    return {address >> ResidencyPageShift, (address + (bytes - 1)) >> ResidencyPageShift, false};
}

constexpr std::uint32_t WordBits = 64;
constexpr std::uint64_t TouchPagesPerFrame = 4096;

ResidencyMask RunMask(std::uint32_t begin, std::uint32_t end) {
    if (end <= begin) return 0;
    const auto first = begin >> ResidencySubpageShift;
    const auto last = (end - 1) >> ResidencySubpageShift;
    const auto count = last - first + 1;
    const std::uint32_t bits = count >= ResidencySubpages ? 0xffffu : (1u << count) - 1u;
    return static_cast<ResidencyMask>(bits << first);
}

}

ResidencyToken MergeTokens(const ResidencyToken& first, const ResidencyToken& second) {
    ResidencyToken merged{};
    for (std::size_t stream = 0; stream < ResidencyStreams; ++stream) merged[stream] = std::max(first[stream], second[stream]);
    return merged;
}

bool TokenEmpty(const ResidencyToken& token) {
    return std::all_of(token.begin(), token.end(), [](std::uint64_t value) { return value == 0; });
}

ResidencyMask SubpageMask(std::uint64_t page, std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return 0;
    const auto base = page << ResidencyPageShift;
    const auto first = std::max(address, base);
    const auto last = std::min(address + (bytes - 1), base + (ResidencyPageBytes - 1));
    if (first > last) return 0;
    const auto firstSubpage = static_cast<std::uint32_t>((first - base) >> ResidencySubpageShift);
    const auto lastSubpage = static_cast<std::uint32_t>((last - base) >> ResidencySubpageShift);
    const auto count = lastSubpage - firstSubpage + 1;
    const std::uint32_t bits = count >= ResidencySubpages ? 0xffffu : (1u << count) - 1u;
    return static_cast<ResidencyMask>(bits << firstSubpage);
}

ResidencyManager::ResidencyManager(IResidencyDevice& device, const ResidencyPolicy& policy)
    : device(device), chunkSlots(policy.chunkSlots), hotFaults(policy.hotFaults), hotFrames(policy.hotFrames), quietFrames(policy.quietFrames), deferCopyIn(policy.deferCopyIn), dirtyLeaves(new std::atomic<DirtyLeaf*>[DirtyLeaves]()) {
    if (chunkSlots == 0 || chunkSlots > (std::numeric_limits<std::uint32_t>::max() >> ResidencyPageShift))
        throw std::invalid_argument("residency: a chunk holds 1 to 65535 pages, not " + std::to_string(chunkSlots));
    if (hotFrames == 0 || hotFrames > 64 || hotFaults == 0 || hotFaults > hotFrames)
        throw std::invalid_argument("residency: a hot page faults in 1 to " + std::to_string(hotFrames) + " of the last 1 to 64 frames, not " + std::to_string(hotFaults) + " of " + std::to_string(hotFrames));
    if (quietFrames == 0) throw std::invalid_argument("residency: a hot page cools after at least one quiet frame");
    SetBudget(policy.budgetBytes);
}

ResidencyManager::~ResidencyManager() {
    std::vector<std::uint64_t> guarded;
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (pages[id].guarded) guarded.push_back(pages[id].index);
        if (pages[id].accessGuarded) accessGuards.insert(pages[id].index);
    }
    if (!guarded.empty()) device.Unguard(guarded);
    for (const auto page : accessGuards) device.UnguardAccess(page);
    for (std::uint32_t chunk = 0; chunk < chunks.size(); ++chunk) {
        if (chunks[chunk].live) device.ReleaseChunk(chunk);
    }
    for (std::size_t leaf = 0; leaf < DirtyLeaves; ++leaf) delete dirtyLeaves[leaf].load(std::memory_order_relaxed);
}

void ResidencyManager::publishDirty(std::uint64_t page, ResidencyMask mask) {
    const auto leafIndex = page >> DirtyLeafShift;
    if (leafIndex >= DirtyLeaves) throw std::out_of_range("residency: page " + std::to_string(page) + " lies past the 47-bit guest address space");
    auto* leaf = dirtyLeaves[leafIndex].load(std::memory_order_acquire);
    if (leaf == nullptr) {
        if (mask == 0) return;
        leaf = new DirtyLeaf();
        dirtyLeaves[leafIndex].store(leaf, std::memory_order_release);
    }
    leaf->masks[page & ((std::uint64_t{1} << DirtyLeafShift) - 1)].store(mask, std::memory_order_release);
}

void ResidencyManager::publishStale(std::uint64_t page, ResidencyMask mask) {
    const auto leafIndex = page >> DirtyLeafShift;
    if (leafIndex >= DirtyLeaves) throw std::out_of_range("residency: page " + std::to_string(page) + " lies past the 47-bit guest address space");
    auto* leaf = dirtyLeaves[leafIndex].load(std::memory_order_acquire);
    if (leaf == nullptr) {
        if (mask == 0) return;
        leaf = new DirtyLeaf();
        dirtyLeaves[leafIndex].store(leaf, std::memory_order_release);
    }
    leaf->stale[page & ((std::uint64_t{1} << DirtyLeafShift) - 1)].store(mask, std::memory_order_release);
}

void ResidencyManager::publishUnsettled(std::uint64_t page, std::uint8_t reason, bool set) {
    const auto leafIndex = page >> DirtyLeafShift;
    if (leafIndex >= DirtyLeaves) throw std::out_of_range("residency: page " + std::to_string(page) + " lies past the 47-bit guest address space");
    auto* leaf = dirtyLeaves[leafIndex].load(std::memory_order_acquire);
    if (leaf == nullptr) {
        if (!set) return;
        leaf = new DirtyLeaf();
        dirtyLeaves[leafIndex].store(leaf, std::memory_order_release);
    }
    auto& bits = leaf->unsettled[page & ((std::uint64_t{1} << DirtyLeafShift) - 1)];
    const auto before = bits.load(std::memory_order_relaxed);
    const auto after = static_cast<std::uint8_t>(set ? before | reason : before & ~reason);
    if (after == before) return;
    if (before == 0) unsettledPages.fetch_add(1, std::memory_order_acq_rel);
    bits.store(after, std::memory_order_release);
    if (after == 0) unsettledPages.fetch_sub(1, std::memory_order_acq_rel);
}

void ResidencyManager::addUnsettled(std::unordered_map<std::uint64_t, ResidencyToken>& entries, std::uint64_t page, const ResidencyToken& token, std::uint8_t reason) {
    auto& entry = entries[page];
    entry = MergeTokens(entry, token);
    publishUnsettled(page, reason, true);
}

void ResidencyManager::eraseUnsettled(std::unordered_map<std::uint64_t, ResidencyToken>& entries, std::uint64_t page, std::uint8_t reason) {
    if (entries.erase(page) != 0) publishUnsettled(page, reason, false);
}

bool ResidencyManager::CpuReadSettled(std::uint64_t address, std::uint64_t bytes) const {
    if (DirtyIn(address, bytes)) return false;
    if (unsettledPages.load(std::memory_order_acquire) == 0) return true;
    const auto span = PagesOf(address, bytes);
    if (span.empty) return true;
    for (auto page = span.first; page <= span.last; ++page) {
        const auto leafIndex = page >> DirtyLeafShift;
        if (leafIndex >= DirtyLeaves) return true;
        const auto* leaf = dirtyLeaves[leafIndex].load(std::memory_order_acquire);
        if (leaf == nullptr) {
            page |= (std::uint64_t{1} << DirtyLeafShift) - 1;
            continue;
        }
        if (leaf->unsettled[page & ((std::uint64_t{1} << DirtyLeafShift) - 1)].load(std::memory_order_acquire) != 0) return false;
    }
    return true;
}

bool ResidencyManager::DirtyIn(std::uint64_t address, std::uint64_t bytes) const {
    const auto span = PagesOf(address, bytes);
    if (span.empty || (dirtyPages.load(std::memory_order_acquire) == 0 && staleCount.load(std::memory_order_acquire) == 0)) return false;
    for (auto page = span.first; page <= span.last; ++page) {
        const auto leafIndex = page >> DirtyLeafShift;
        if (leafIndex >= DirtyLeaves) return false;
        const auto* leaf = dirtyLeaves[leafIndex].load(std::memory_order_acquire);
        if (leaf == nullptr) {
            page |= (std::uint64_t{1} << DirtyLeafShift) - 1;
            continue;
        }
        const auto index = page & ((std::uint64_t{1} << DirtyLeafShift) - 1);
        const auto mask = static_cast<ResidencyMask>(leaf->masks[index].load(std::memory_order_acquire) | leaf->stale[index].load(std::memory_order_acquire));
        if ((mask & SubpageMask(page, address, bytes)) != 0) return true;
    }
    return false;
}

bool ResidencyManager::StaleIn(std::uint64_t address, std::uint64_t bytes) const {
    if (staleCount.load(std::memory_order_acquire) == 0 || bytes == 0) return false;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - address) bytes = std::numeric_limits<std::uint64_t>::max() - address;
    const auto span = PagesOf(address, bytes);
    if (span.empty) return false;
    for (auto page = span.first; page <= span.last; ++page) {
        const auto leafIndex = page >> DirtyLeafShift;
        if (leafIndex >= DirtyLeaves) return false;
        const auto* leaf = dirtyLeaves[leafIndex].load(std::memory_order_acquire);
        if (leaf == nullptr) {
            page |= (std::uint64_t{1} << DirtyLeafShift) - 1;
            continue;
        }
        const auto mask = leaf->stale[page & ((std::uint64_t{1} << DirtyLeafShift) - 1)].load(std::memory_order_acquire);
        if ((mask & SubpageMask(page, address, bytes)) != 0) return true;
    }
    return false;
}

void ResidencyManager::publishPage(std::uint64_t page) {
    ResidencyMask mask = 0;
    if (const auto id = find(page)) mask = pages[*id].dirty;
    publishDirty(page, mask);
    const std::shared_lock lock(ownedMutex);
    ResidencyMask stale = 0;
    if (const auto entry = owned.find(page); entry != owned.end()) stale = entry->second.stale;
    publishStale(page, stale);
}

bool ResidencyManager::isStale(std::uint64_t page) const {
    const std::shared_lock lock(ownedMutex);
    return staleOwned.contains(page);
}

std::vector<std::uint64_t> ResidencyManager::staleIn(std::uint64_t address, std::uint64_t bytes) const {
    std::vector<std::uint64_t> stale;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - address) bytes = std::numeric_limits<std::uint64_t>::max() - address;
    const auto span = PagesOf(address, bytes);
    const std::shared_lock lock(ownedMutex);
    if (span.empty || staleOwned.empty()) return stale;
    if (span.last - span.first >= staleOwned.size()) {
        for (const auto page : staleOwned) {
            if (page >= span.first && page <= span.last) stale.push_back(page);
        }
        std::sort(stale.begin(), stale.end());
        return stale;
    }
    for (auto page = span.first; page <= span.last; ++page) {
        if (staleOwned.contains(page)) stale.push_back(page);
    }
    return stale;
}

ResidencyManager::CpuReadExemption::CpuReadExemption(const IResidencyOwner* owner) : previous(std::exchange(cpuReadExempt, owner)) {}

ResidencyManager::CpuReadExemption::~CpuReadExemption() {
    cpuReadExempt = previous;
}

bool ResidencyManager::retileStale(std::span<const std::uint64_t> stalePages, std::span<const IResidencyOwner* const> except) {
    struct Work {
        std::shared_ptr<IResidencyOwner> owner;
        std::vector<ResidencyRange> ranges;
    };
    std::vector<Work> work;
    {
        const std::unique_lock lock(ownedMutex);
        for (const auto page : stalePages) {
            const auto entry = owned.find(page);
            if (entry == owned.end() || entry->second.stale == 0) continue;
            auto* owner = entry->second.owner;
            if (std::find(except.begin(), except.end(), owner) != except.end()) continue;
            std::uint32_t bits = entry->second.stale;
            entry->second.stale = 0;
            entry->second.guarded = 0;
            if (staleOwned.erase(page) != 0) --staleCount;
            publishStale(page, 0);
            auto at = std::find_if(work.begin(), work.end(), [&](const Work& item) { return item.owner.get() == owner; });
            if (at == work.end()) {
                auto held = owner->Hold();
                if (held == nullptr) continue;
                at = work.insert(work.end(), {std::move(held), {}});
            }
            auto& ranges = at->ranges;
            while (bits != 0) {
                const auto first = static_cast<std::uint32_t>(std::countr_zero(bits));
                const auto run = static_cast<std::uint32_t>(std::countr_one(bits >> first));
                const auto address = (page << ResidencyPageShift) + (std::uint64_t{first} << ResidencySubpageShift);
                const auto length = std::uint64_t{run} << ResidencySubpageShift;
                if (!ranges.empty() && ranges.back().address + ranges.back().bytes == address) ranges.back().bytes += length;
                else ranges.push_back({address, length});
                counts.retiledBytes += length;
                bits &= run >= 32 ? 0u : ~(((1u << run) - 1u) << first);
            }
        }
    }
    if (work.empty()) return false;
    counts.retiles += work.size();
    const auto held = structure.ReleaseAll();
    for (const auto& item : work) item.owner->Retile(item.ranges);
    structure.Reacquire(held);
    return true;
}

void ResidencyManager::Own(std::uint64_t address, std::uint64_t bytes, IResidencyOwner* owner) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    const auto span = PagesOf(address, bytes);
    if (span.empty) return;
    std::vector<std::uint64_t> previous;
    bool changed = false;
    {
        const std::shared_lock lock(ownedMutex);
        for (auto page = span.first; page <= span.last; ++page) {
            const auto entry = owned.find(page);
            if (entry == owned.end() || entry->second.owner != owner) changed = true;
            if (entry != owned.end() && entry->second.owner != owner && entry->second.stale != 0) previous.push_back(page);
        }
    }
    retileStale(previous);
    if (changed) {
        const std::unique_lock lock(ownedMutex);
        for (auto page = span.first; page <= span.last; ++page) owned[page].owner = owner;
        ownershipSerial.fetch_add(1, std::memory_order_acq_rel);
    }
    admit(span.first, span.last);
    for (const auto id : residentIn(address, bytes)) pages[id].used = true;
}

void ResidencyManager::MarkStale(std::uint64_t address, std::uint64_t bytes, IResidencyOwner* owner) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    const auto span = PagesOf(address, bytes);
    if (span.empty) return;
    std::vector<ResidencyRange> eager;
    const auto addEager = [&](std::uint64_t begin, std::uint64_t end) {
        if (!eager.empty() && eager.back().address + eager.back().bytes == begin) eager.back().bytes += end - begin;
        else eager.push_back({begin, end - begin});
    };
    std::vector<bool> mine(span.last - span.first + 1, false);
    {
        const std::shared_lock lock(ownedMutex);
        for (auto page = span.first; page <= span.last; ++page) {
            const auto entry = owned.find(page);
            mine[page - span.first] = entry != owned.end() && entry->second.owner == owner;
        }
    }
    std::vector<ResidencyMask> guards(mine.size(), 0);
    for (auto page = span.first; page <= span.last; ++page) {
        const auto base = page << ResidencyPageShift;
        const auto begin = std::max(address, base);
        const auto end = std::min(address + bytes, base + ResidencyPageBytes);
        const auto mask = SubpageMask(page, address, bytes);
        if (!mine[page - span.first]) {
            addEager(begin, end);
            continue;
        }
        const auto id = find(page);
        ResidencyMask guarded = 0;
        if (id && dirtyable(pages[*id]) && !aliasKeys.contains(page)) {
            auto& record = pages[*id];
            const auto adding = static_cast<ResidencyMask>(mask & ~record.accessMask);
            if (adding != 0) {
                if (const auto refusal = device.GuardAccess(page, adding); refusal == ResidencyRefusal::None) {
                    record.accessMask = static_cast<ResidencyMask>(record.accessMask | adding);
                    record.accessGuarded = true;
                    accessGuards.insert(page);
                } else {
                    ++counts.accessUnguardable;
                    noteRefused(page, refusal);
                }
            }
            guarded = static_cast<ResidencyMask>(mask & record.accessMask);
        }
        guards[page - span.first] = guarded;
    }
    {
        const std::unique_lock lock(ownedMutex);
        for (auto page = span.first; page <= span.last; ++page) {
            if (!mine[page - span.first]) continue;
            const auto found = owned.find(page);
            if (found == owned.end() || found->second.owner != owner) continue;
            const auto mask = SubpageMask(page, address, bytes);
            auto& entry = found->second;
            if (entry.stale == 0 && staleOwned.insert(page).second) ++staleCount;
            entry.stale = static_cast<ResidencyMask>(entry.stale | mask);
            entry.guarded = static_cast<ResidencyMask>((entry.guarded & ~mask) | guards[page - span.first]);
            publishStale(page, entry.stale);
        }
    }
    if (eager.empty()) return;
    ++counts.eagerRetiles;
    for (const auto& range : eager) counts.retiledBytes += range.bytes;
    const auto held = structure.ReleaseAll();
    owner->Retile(eager);
    structure.Reacquire(held);
}

bool ResidencyManager::RetileOwned(std::uint64_t address, std::uint64_t bytes, std::span<const IResidencyOwner* const> except) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    if (staleCount.load(std::memory_order_acquire) == 0) return false;
    return retileStale(staleIn(address, bytes), except);
}

bool ResidencyManager::RetileUnguarded() {
    const std::lock_guard structureLock(structure);
    applyQueued();
    std::vector<std::uint64_t> unguarded;
    {
        const std::shared_lock lock(ownedMutex);
        for (const auto page : staleOwned) {
            const auto entry = owned.find(page);
            if (entry != owned.end() && (entry->second.stale & ~entry->second.guarded) != 0) unguarded.push_back(page);
        }
    }
    if (unguarded.empty()) return false;
    std::sort(unguarded.begin(), unguarded.end());
    return retileStale(unguarded);
}

bool ResidencyManager::RetileAllOwned(std::span<const IResidencyOwner* const> except) {
    const std::lock_guard structureLock(structure);
    std::vector<std::uint64_t> stale;
    {
        const std::shared_lock lock(ownedMutex);
        if (staleOwned.empty()) return false;
        stale.assign(staleOwned.begin(), staleOwned.end());
    }
    std::sort(stale.begin(), stale.end());
    return retileStale(stale, except);
}

void ResidencyManager::Disown(IResidencyOwner* owner) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    std::vector<std::uint64_t> stale;
    {
        const std::shared_lock lock(ownedMutex);
        for (const auto& [page, record] : owned) {
            if (record.owner == owner && record.stale != 0) stale.push_back(page);
        }
    }
    std::sort(stale.begin(), stale.end());
    retileStale(stale);
    Forget(owner);
}

void ResidencyManager::Forget(IResidencyOwner* owner) {
    const std::unique_lock lock(ownedMutex);
    for (auto entry = owned.begin(); entry != owned.end();) {
        if (entry->second.owner != owner) {
            ++entry;
            continue;
        }
        if (entry->second.stale != 0) {
            if (staleOwned.erase(entry->first) != 0) --staleCount;
            publishStale(entry->first, 0);
        }
        entry = owned.erase(entry);
    }
    ownershipSerial.fetch_add(1, std::memory_order_acq_rel);
}

void ResidencyManager::ClearStale(std::uint64_t address, std::uint64_t bytes, const IResidencyOwner* owner) {
    const auto span = PagesOf(address, bytes);
    if (span.empty) return;
    const std::unique_lock lock(ownedMutex);
    for (auto page = span.first; page <= span.last; ++page) {
        const auto entry = owned.find(page);
        if (entry == owned.end() || entry->second.owner != owner || entry->second.stale == 0) continue;
        entry->second.stale = static_cast<ResidencyMask>(entry->second.stale & ~SubpageMask(page, address, bytes));
        publishStale(page, entry->second.stale);
        if (entry->second.stale == 0 && staleOwned.erase(page) != 0) --staleCount;
    }
}

IResidencyOwner* ResidencyManager::OwnerOf(std::uint64_t page) const {
    const std::shared_lock lock(ownedMutex);
    const auto entry = owned.find(page);
    return entry == owned.end() ? nullptr : entry->second.owner;
}

ResidencyMask ResidencyManager::Stale(std::uint64_t page) const {
    const std::shared_lock lock(ownedMutex);
    const auto entry = owned.find(page);
    return entry == owned.end() ? ResidencyMask{0} : entry->second.stale;
}

std::vector<std::shared_ptr<IResidencyOwner>> ResidencyManager::HoldOwners(std::uint64_t address, std::uint64_t bytes) const {
    std::vector<std::shared_ptr<IResidencyOwner>> held;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - address) bytes = std::numeric_limits<std::uint64_t>::max() - address;
    const auto span = PagesOf(address, bytes);
    if (span.empty) return held;
    std::vector<IResidencyOwner*> seen;
    const auto take = [&](IResidencyOwner* owner) {
        if (std::find(seen.begin(), seen.end(), owner) != seen.end()) return;
        seen.push_back(owner);
        if (auto alive = owner->Hold()) held.push_back(std::move(alive));
    };
    const std::shared_lock lock(ownedMutex);
    if (owned.empty()) return held;
    if (span.last - span.first >= owned.size()) {
        std::vector<std::pair<std::uint64_t, IResidencyOwner*>> found;
        for (const auto& [page, record] : owned) {
            if (page >= span.first && page <= span.last) found.emplace_back(page, record.owner);
        }
        std::sort(found.begin(), found.end());
        for (const auto& [page, owner] : found) take(owner);
        return held;
    }
    for (auto page = span.first; page <= span.last; ++page) {
        if (const auto entry = owned.find(page); entry != owned.end()) take(entry->second.owner);
    }
    return held;
}

std::vector<std::shared_ptr<IResidencyOwner>> ResidencyManager::HoldStaleOwners() const {
    std::vector<std::shared_ptr<IResidencyOwner>> held;
    std::vector<IResidencyOwner*> seen;
    const std::shared_lock lock(ownedMutex);
    for (const auto page : staleOwned) {
        const auto entry = owned.find(page);
        if (entry == owned.end() || std::find(seen.begin(), seen.end(), entry->second.owner) != seen.end()) continue;
        seen.push_back(entry->second.owner);
        if (auto alive = entry->second.owner->Hold()) held.push_back(std::move(alive));
    }
    return held;
}

bool ResidencyManager::GuardedStaleIn(std::uint64_t address, std::uint64_t bytes, const IResidencyOwner* owner) const {
    const std::lock_guard structureLock(structure);
    if (!StaleIn(address, bytes)) return false;
    const auto span = PagesOf(address, bytes);
    const std::shared_lock lock(ownedMutex);
    for (auto page = span.first; page <= span.last; ++page) {
        const auto entry = owned.find(page);
        if (entry == owned.end() || entry->second.owner != owner || entry->second.stale == 0) continue;
        const auto id = find(page);
        if (id && (pages[*id].accessMask & entry->second.stale) != 0) return true;
    }
    return false;
}

bool ResidencyManager::StaleOfOthers(std::uint64_t address, std::uint64_t bytes, const IResidencyOwner* owner) const {
    if (!StaleIn(address, bytes)) return false;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - address) bytes = std::numeric_limits<std::uint64_t>::max() - address;
    const auto span = PagesOf(address, bytes);
    const std::shared_lock lock(ownedMutex);
    if (span.last - span.first >= staleOwned.size()) {
        for (const auto page : staleOwned) {
            if (page < span.first || page > span.last) continue;
            const auto entry = owned.find(page);
            if (entry != owned.end() && entry->second.owner != owner && (entry->second.stale & SubpageMask(page, address, bytes)) != 0) return true;
        }
        return false;
    }
    for (auto page = span.first; page <= span.last; ++page) {
        const auto entry = owned.find(page);
        if (entry != owned.end() && entry->second.owner != owner && (entry->second.stale & SubpageMask(page, address, bytes)) != 0) return true;
    }
    return false;
}

void ResidencyManager::WriteBackAll() {
    const std::lock_guard structureLock(structure);
    applyQueued();
    RetileAllOwned();
    std::vector<std::uint32_t> unbounded;
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (unboundedPending(pages[id])) unbounded.push_back(id);
    }
    settleUnbounded(unbounded, true);
    std::vector<std::uint32_t> ids;
    ResidencyToken pending{};
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (pages[id].dirtyRuns.empty()) continue;
        ids.push_back(id);
        const auto writer = writers.find(pages[id].index);
        if (writer != writers.end() && !settled(writer->second)) pending = MergeTokens(pending, writer->second);
    }
    if (!TokenEmpty(pending)) {
        const auto indices = indicesOf(ids);
        waitFor(pending);
        ids = stillResident(ids, indices);
    }
    writeBackDirty(ids);
    ResidencyToken wait{};
    for (const auto& [page, token] : writingBack) wait = MergeTokens(wait, token);
    if (!settled(wait)) waitFor(wait);
    eraseUnsettledIf(writingBack, UnsettledWriteBack, [](const auto&) { return true; });
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (!pages[id].accessGuarded) continue;
        accessGuards.insert(pages[id].index);
        pages[id].accessGuarded = false;
        pages[id].accessMask = 0;
        exposeToWriters(id);
    }
    for (const auto page : accessGuards) device.UnguardAccess(page);
    accessGuards.clear();
}

void ResidencyManager::settleImports() {
    for (const auto page : settlesWithoutToken) {
        if (const auto id = find(page); id && !pages[*id].importSettled && !pages[*id].settleIssued) pages[*id].settle = device.Newest();
    }
    settlesWithoutToken.clear();
    std::vector<std::uint64_t> settledPages;
    for (auto it = unsettledImports.begin(); it != unsettledImports.end();) {
        const auto id = find(*it);
        if (id && !pages[*id].settleIssued) {
            const auto newest = device.Newest();
            if (newest != pages[*id].settle) {
                pages[*id].settle = newest;
                pages[*id].settleIssued = true;
            }
        }
        if (id && (!pages[*id].settleIssued || !settled(pages[*id].settle))) {
            ++it;
            continue;
        }
        if (id) settledPages.push_back(*it);
        it = unsettledImports.erase(it);
    }
    if (settledPages.empty()) return;
    for (const auto page : settledPages) device.CollectWrites(page, 1);
    applyQueued();
    for (const auto page : settledPages) {
        const auto id = find(page);
        if (!id) continue;
        device.SettleImport(page, pages[*id].importStamp);
        if (!device.ImportSettled(page)) {
            unsettle(*id);
            continue;
        }
        pages[*id].importSettled = true;
        ++counts.importsSettled;
        unmarkHot(page);
        rearm(*id);
    }
}

void ResidencyManager::unsettle(std::uint32_t id) {
    auto& page = pages[id];
    page.importSettled = false;
    page.importStamp = device.ImportStamp();
    if (withoutRecorder) settlesWithoutToken.push_back(page.index);
    else page.settle = device.Newest();
    page.settleIssued = false;
    unsettledImports.insert(page.index);
    ++counts.unsettledGuards;
    markHot(page.index, Epoch());
    exposeToWriters(id);
}

bool ResidencyManager::Hot(std::uint64_t page) const {
    const std::lock_guard structureLock(structure);
    return hot.contains(page);
}

bool ResidencyManager::HotResident(std::uint64_t page) const {
    const std::lock_guard structureLock(structure);
    return hotResident.contains(page);
}

bool ResidencyManager::ImportSettled(std::uint64_t page) const {
    const std::lock_guard structureLock(structure);
    const auto id = find(page);
    return id && pages[*id].importSettled;
}

bool ResidencyManager::settled(const ResidencyToken& token) {
    return TokenEmpty(token) || device.Reached(token);
}

void ResidencyManager::waitFor(const ResidencyToken& token) {
    ++counts.waits;
    const auto held = structure.ReleaseAll();
    device.Wait(token);
    structure.Reacquire(held);
}

std::optional<std::uint32_t> ResidencyManager::find(std::uint64_t page) const {
    const auto found = resident.find(page);
    if (found == resident.end()) return std::nullopt;
    return found->second;
}

std::vector<std::uint64_t> ResidencyManager::indicesOf(std::span<const std::uint32_t> ids) const {
    std::vector<std::uint64_t> indices;
    indices.reserve(ids.size());
    for (const auto id : ids) indices.push_back(pages[id].index);
    return indices;
}

std::vector<std::uint32_t> ResidencyManager::stillResident(std::span<const std::uint32_t> ids, std::span<const std::uint64_t> indices) const {
    std::vector<std::uint32_t> kept;
    kept.reserve(ids.size());
    for (std::size_t at = 0; at < ids.size(); ++at) {
        const auto id = find(indices[at]);
        if (id && *id == ids[at]) kept.push_back(ids[at]);
    }
    return kept;
}

std::vector<std::uint64_t> ResidencyManager::indicesOf(std::span<const Fill> fills) const {
    std::vector<std::uint64_t> indices;
    indices.reserve(fills.size());
    for (const auto& fill : fills) indices.push_back(pages[fill.id].index);
    return indices;
}

std::vector<ResidencyManager::Fill> ResidencyManager::stillResident(std::span<const Fill> fills, std::span<const std::uint64_t> indices) const {
    std::vector<Fill> kept;
    kept.reserve(fills.size());
    for (std::size_t at = 0; at < fills.size(); ++at) {
        const auto id = find(indices[at]);
        if (id && *id == fills[at].id) kept.push_back(fills[at]);
    }
    return kept;
}

std::vector<std::uint32_t> ResidencyManager::residentIn(std::uint64_t address, std::uint64_t bytes) const {
    std::vector<std::uint32_t> ids;
    const auto span = PagesOf(address, bytes);
    if (span.empty) return ids;
    if (span.last - span.first >= resident.size()) {
        for (const auto& [page, id] : resident) {
            if (page >= span.first && page <= span.last) ids.push_back(id);
        }
        std::sort(ids.begin(), ids.end(), [this](std::uint32_t left, std::uint32_t right) { return pages[left].index < pages[right].index; });
        return ids;
    }
    for (auto page = span.first; page <= span.last; ++page) {
        if (const auto id = find(page)) ids.push_back(*id);
    }
    return ids;
}

bool ResidencyManager::anyResident(std::uint64_t first, std::uint64_t last) const {
    if (last - first >= resident.size()) return std::ranges::any_of(resident, [&](const auto& entry) { return entry.first >= first && entry.first <= last; });
    for (auto page = first; page <= last; ++page) {
        if (resident.contains(page)) return true;
    }
    return false;
}

std::vector<ResidencyManager::Fill> ResidencyManager::withAliases(std::uint64_t address, std::uint64_t bytes) {
    std::vector<Fill> fills;
    if (bytes == 0) return fills;
    auto spans = device.Aliases(address, bytes);
    spans.push_back({address, bytes});
    std::map<std::uint32_t, Run> runs;
    for (const auto& range : spans) {
        for (const auto id : residentIn(range.address, range.bytes)) {
            const auto base = pages[id].index << ResidencyPageShift;
            const auto begin = static_cast<std::uint32_t>(std::max(range.address, base) - base);
            const auto end = static_cast<std::uint32_t>(std::min(range.address + range.bytes, base + ResidencyPageBytes) - base);
            const auto [entry, inserted] = runs.emplace(id, Run{begin, end});
            if (inserted) continue;
            entry->second.first = std::min(entry->second.first, begin);
            entry->second.second = std::max(entry->second.second, end);
        }
    }
    for (const auto& [id, run] : runs) fills.push_back({id, run.first, run.second});
    std::sort(fills.begin(), fills.end(), [this](const Fill& left, const Fill& right) { return pages[left.id].index < pages[right.id].index; });
    return fills;
}

void ResidencyManager::unlink(std::uint32_t id) {
    auto& page = pages[id];
    if (page.older != None) pages[page.older].newer = page.newer;
    else oldest = page.newer;
    if (page.newer != None) pages[page.newer].older = page.older;
    else newest = page.older;
    page.older = None;
    page.newer = None;
}

void ResidencyManager::linkNewest(std::uint32_t id) {
    auto& page = pages[id];
    page.older = newest;
    page.newer = None;
    if (newest != None) pages[newest].newer = id;
    else oldest = id;
    newest = id;
}

void ResidencyManager::changed(std::uint64_t page) {
    ++reuseVersion;
    changes.insert(page);
    if (changeLog.empty()) changeLog.resize(ChangeLogPages);
    changeLog[serial % ChangeLogPages] = static_cast<std::uint32_t>(page);
    ++serial;
}

std::uint32_t ResidencyManager::newPage(std::uint64_t page, MirrorSlot slot, const ResidencyImport& import, bool guarded) {
    std::uint32_t id;
    if (!freePages.empty()) {
        id = freePages.back();
        freePages.pop_back();
    } else {
        id = static_cast<std::uint32_t>(pages.size());
        pages.emplace_back();
    }
    auto& record = pages[id];
    record = Page{};
    record.index = page;
    record.slot = slot;
    record.gpuWritable = import.gpuWritable;
    record.cpuWritable = import.cpuWritable;
    record.shared = import.section != 0;
    record.importSettled = record.shared;
    if (!record.shared) {
        record.importStamp = device.ImportStamp();
        record.settle = device.Newest();
        unsettledImports.insert(page);
    }
    record.importSerial = import.serial;
    record.guarded = guarded;
    record.unboundedSerial = unboundedIssued;
    exposeToWriters(id);
    {
        const std::lock_guard lock(residentMutex);
        resident.emplace(page, id);
    }
    linkNewest(id);
    changed(page);
    return id;
}

void ResidencyManager::releasePage(std::uint32_t id) {
    if (id / 64 < writerExposed.size() && (writerExposed[id / 64] & (std::uint64_t{1} << (id % 64))) != 0) {
        writerExposed[id / 64] &= ~(std::uint64_t{1} << (id % 64));
        --writerExposedCount;
    }
    clearBehind(id);
    pages[id] = Page{};
    freePages.push_back(id);
}

void ResidencyManager::forget(std::uint32_t id) {
    const auto page = pages[id].index;
    unlink(id);
    {
        const std::lock_guard lock(residentMutex);
        resident.erase(page);
    }
    if (!pages[id].dirtyRuns.empty()) --dirtyPages;
    pages[id].dirtyRuns.clear();
    pages[id].dirty = 0;
    publishDirty(page, 0);
    unmarkHot(page);
    unsettledImports.erase(page);
    eraseUnsettled(writers, page, UnsettledWriter);
    if (const auto key = aliasKeys.find(page); key != aliasKeys.end()) {
        canonical.erase(key->second);
        aliasKeys.erase(key);
    }
    changed(page);
    releasePage(id);
}

void ResidencyManager::markDirty(std::uint32_t id, std::uint32_t begin, std::uint32_t end) {
    if (end <= begin) return;
    if (pages[id].behind != 0) {
        const std::uint32_t ids[] = {id};
        catchUp(ids, false);
    }
    auto& page = pages[id];
    if (page.dirtyRuns.empty()) ++dirtyPages;
    auto& runs = page.dirtyRuns;
    auto at = std::lower_bound(runs.begin(), runs.end(), Run{begin, begin}, [](const Run& left, const Run& right) { return left.second < right.first; });
    auto last = at;
    while (last != runs.end() && last->first <= end) {
        begin = std::min(begin, last->first);
        end = std::max(end, last->second);
        ++last;
    }
    at = runs.erase(at, last);
    runs.insert(at, Run{begin, end});
    page.dirty = static_cast<ResidencyMask>(page.dirty | RunMask(begin, end));
    publishPage(page.index);
}

void ResidencyManager::clearDirty(std::uint32_t id, std::uint32_t begin, std::uint32_t end) {
    auto& page = pages[id];
    if (page.dirtyRuns.empty() || end <= begin) return;
    std::vector<Run> kept;
    kept.reserve(page.dirtyRuns.size() + 1);
    for (const auto& run : page.dirtyRuns) {
        if (run.second <= begin || run.first >= end) {
            kept.push_back(run);
            continue;
        }
        if (run.first < begin) kept.push_back({run.first, begin});
        if (run.second > end) kept.push_back({end, run.second});
    }
    page.dirtyRuns = std::move(kept);
    page.dirty = 0;
    for (const auto& run : page.dirtyRuns) page.dirty = static_cast<ResidencyMask>(page.dirty | RunMask(run.first, run.second));
    publishPage(page.index);
    if (page.dirtyRuns.empty()) --dirtyPages;
}

void ResidencyManager::clearAllDirty(std::uint32_t id) {
    auto& page = pages[id];
    if (page.dirtyRuns.empty()) return;
    page.dirtyRuns.clear();
    page.dirty = 0;
    publishPage(page.index);
    --dirtyPages;
}

void ResidencyManager::appendDirty(std::uint32_t id, std::vector<ResidencyCopy>& copies) const {
    const auto& page = pages[id];
    for (const auto& [begin, end] : page.dirtyRuns) copies.push_back({page.index, page.slot, begin, end - begin});
}

void ResidencyManager::appendClean(std::uint32_t id, std::uint32_t begin, std::uint32_t end, std::vector<ResidencyCopy>& copies) const {
    const auto& page = pages[id];
    auto cursor = begin;
    for (const auto& [dirtyBegin, dirtyEnd] : page.dirtyRuns) {
        if (dirtyEnd <= cursor) continue;
        if (dirtyBegin >= end) break;
        if (dirtyBegin > cursor) copies.push_back({page.index, page.slot, cursor, dirtyBegin - cursor});
        cursor = std::max(cursor, dirtyEnd);
        if (cursor >= end) return;
    }
    if (cursor < end) copies.push_back({page.index, page.slot, cursor, end - cursor});
}

std::optional<MirrorSlot> ResidencyManager::takeFree() {
    for (std::uint32_t chunk = 0; chunk < chunks.size(); ++chunk) {
        auto& record = chunks[chunk];
        if (!record.live || record.draining || record.freeSlots == 0) continue;
        for (std::uint32_t word = 0; word < record.freeBits.size(); ++word) {
            if (record.freeBits[word] == 0) continue;
            const auto bit = static_cast<std::uint32_t>(std::countr_zero(record.freeBits[word]));
            record.freeBits[word] &= ~(std::uint64_t{1} << bit);
            --record.freeSlots;
            return MirrorSlot{chunk, word * WordBits + bit};
        }
    }
    return std::nullopt;
}

void ResidencyManager::freeSlot(MirrorSlot slot) {
    auto& record = chunks[slot.chunk];
    record.freeBits[slot.slot / WordBits] |= std::uint64_t{1} << (slot.slot % WordBits);
    ++record.freeSlots;
    if (record.draining && record.freeSlots == chunkSlots) releaseChunk(slot.chunk);
}

void ResidencyManager::retireReached() {
    while (!retiring.empty() && settled(retiring.front().token)) {
        const auto slot = retiring.front().slot;
        retiring.pop_front();
        freeSlot(slot);
    }
}

std::uint64_t ResidencyManager::budgetBytes() {
    if (budget != ResidencyMeasuredBudget) return budget;
    if (measuredEpoch != Epoch() || measuredEpoch == 0) {
        measured = device.Budget();
        measuredEpoch = Epoch();
    }
    return measured;
}

bool ResidencyManager::fullResidency() {
    return budgetBytes() >= device.GuestBytes();
}

std::uint32_t ResidencyManager::chunkLimit(bool growing) {
    const auto chunkBytes = std::uint64_t{chunkSlots} << ResidencyPageShift;
    auto bytes = budgetBytes();
    if (budget == ResidencyMeasuredBudget && growing) bytes = bytes > chunkBytes ? bytes - chunkBytes : 0;
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(bytes / chunkBytes, std::numeric_limits<std::uint32_t>::max()));
}

std::optional<std::uint32_t> ResidencyManager::allocateChunk() {
    if (liveChunks >= maxChunks && limitEpoch != Epoch()) {
        limitEpoch = Epoch();
        maxChunks = std::max(maxChunks, chunkLimit(true));
    }
    if (liveChunks >= maxChunks) return std::nullopt;
    std::uint32_t chunk = 0;
    while (chunk < chunks.size() && chunks[chunk].live) ++chunk;
    const auto address = device.AllocateChunk(chunk, chunkSlots);
    if (chunk == chunks.size()) chunks.emplace_back();
    auto& record = chunks[chunk];
    record.address = address;
    record.freeBits.assign((chunkSlots + WordBits - 1) / WordBits, ~std::uint64_t{0});
    if (chunkSlots % WordBits != 0) record.freeBits.back() = (std::uint64_t{1} << (chunkSlots % WordBits)) - 1;
    record.freeSlots = chunkSlots;
    record.live = true;
    record.draining = false;
    ++liveChunks;
    ++counts.chunksAllocated;
    return chunk;
}

void ResidencyManager::releaseChunk(std::uint32_t chunk) {
    device.ReleaseChunk(chunk);
    auto& record = chunks[chunk];
    record = Chunk{};
    --liveChunks;
    ++counts.chunksReleased;
}

std::optional<std::uint32_t> ResidencyManager::victim(std::uint64_t protectFirst, std::uint64_t protectLast) {
    for (auto id = oldest; id != None; id = pages[id].newer) {
        const auto& page = pages[id];
        if (page.index >= protectFirst && page.index <= protectLast) continue;
        if (touchFrame != 0 && page.touchFrame == touchFrame) continue;
        if (isStale(page.index)) continue;
        if (!settled(page.pin) || unboundedPending(page)) continue;
        return id;
    }
    return std::nullopt;
}

std::optional<MirrorSlot> ResidencyManager::acquireSlot(std::uint64_t protectFirst, std::uint64_t protectLast, WriteBacks& writeBacks, bool* evicted) {
    retireReached();
    if (const auto slot = takeFree()) return slot;
    if (allocateChunk()) return takeFree();
    const auto id = victim(protectFirst, protectLast);
    if (!id) return std::nullopt;
    if (evicted != nullptr) *evicted = true;
    ++counts.evicted;
    leave(*id, writeBacks);
    retireReached();
    return takeFree();
}

void ResidencyManager::leave(std::uint32_t id, WriteBacks& writeBacks) {
    const auto& page = pages[id];
    if (page.guarded) writeBacks.unguard.push_back(page.index);
    if (!page.dirtyRuns.empty()) {
        appendDirty(id, writeBacks.copies);
        writeBacks.slots.push_back({page.slot, page.pin});
        writeBacks.pages.push_back(page.index);
    } else {
        retiring.push_back({page.slot, MergeTokens(page.pin, device.Newest())});
    }
    forget(id);
}

void ResidencyManager::recordWriteBacks(WriteBacks& writeBacks) {
    if (!writeBacks.unguard.empty()) device.Unguard(writeBacks.unguard);
    if (writeBacks.copies.empty()) {
        writeBacks = WriteBacks{};
        return;
    }
    const auto token = device.WriteBack(writeBacks.copies);
    for (const auto& copy : writeBacks.copies) counts.writtenBackBytes += copy.bytes;
    const auto newestToken = device.Newest();
    for (const auto& entry : writeBacks.slots) retiring.push_back({entry.slot, MergeTokens(MergeTokens(entry.token, token), newestToken)});
    for (const auto page : writeBacks.pages) addUnsettled(writingBack, page, token, UnsettledWriteBack);
    writeBacks = WriteBacks{};
}

std::vector<ResidencyManager::Fill> ResidencyManager::deferFills(std::span<const Fill> fills) {
    std::vector<Fill> now;
    if (!deferCopyIn || catchingUp) {
        now.assign(fills.begin(), fills.end());
        return now;
    }
    std::optional<std::uint64_t> generation;
    for (const auto& fill : fills) {
        auto& page = pages[fill.id];
        if (!page.dirtyRuns.empty()) {
            now.push_back(fill);
            continue;
        }
        if (fill.end <= fill.begin) continue;
        if (!generation) generation = device.Generation();
        markBehind(fill.id, RunMask(fill.begin, fill.end));
        counts.copyInDeferredBytes += fill.end - fill.begin;
        if (fill.begin == 0 && fill.end == WholeEnd) page.generation = *generation;
    }
    return now;
}

void ResidencyManager::markBehind(std::uint32_t id, ResidencyMask mask) {
    auto& page = pages[id];
    if (mask == 0) return;
    if (page.behind == 0) {
        ++counts.behind;
        ++counts.behindFlips;
    }
    page.behind = static_cast<ResidencyMask>(page.behind | mask);
}

void ResidencyManager::clearBehind(std::uint32_t id) {
    auto& page = pages[id];
    if (page.behind == 0) return;
    page.behind = 0;
    --counts.behind;
    ++counts.behindFlips;
}

void ResidencyManager::catchUp(std::span<const std::uint32_t> ids, bool waitForWriters) {
    std::vector<Fill> fills;
    for (const auto id : ids) {
        std::uint32_t bits = pages[id].behind;
        if (bits == 0) continue;
        clearBehind(id);
        ++counts.caughtUp;
        while (bits != 0) {
            const auto first = static_cast<std::uint32_t>(std::countr_zero(bits));
            const auto run = static_cast<std::uint32_t>(std::countr_one(bits >> first));
            fills.push_back({id, first << ResidencySubpageShift, (first + run) << ResidencySubpageShift});
            bits &= run >= 32 ? 0u : ~(((1u << run) - 1u) << first);
        }
    }
    if (fills.empty()) return;
    const bool was = catchingUp;
    catchingUp = true;
    if (waitForWriters) refresh(std::move(fills));
    else copyIn(fills);
    catchingUp = was;
}

void ResidencyManager::copyIn(std::span<const Fill> given) {
    const auto fills = deferFills(given);
    std::vector<ResidencyCopy> copies;
    for (const auto& fill : fills) appendClean(fill.id, fill.begin, fill.end, copies);
    if (copies.empty()) return;
    const auto generation = device.Generation();
    const auto copied = device.CopyIn(copies);
    for (const auto& copy : copies) counts.copiedInBytes += copy.bytes;
    for (const auto& fill : fills) {
        auto& page = pages[fill.id];
        page.pin = MergeTokens(page.pin, copied);
        if (fill.begin == 0 && fill.end == WholeEnd) page.generation = generation;
    }
}

bool ResidencyManager::waitWriters(std::span<const Fill> fills) {
    ResidencyToken pending{};
    for (const auto& fill : fills) {
        const auto writer = writers.find(pages[fill.id].index);
        if (writer != writers.end() && !settled(writer->second)) pending = MergeTokens(pending, writer->second);
    }
    if (TokenEmpty(pending)) return false;
    const auto own = device.Current();
    for (std::size_t stream = 0; stream < pending.size(); ++stream) {
        if (own[stream] != 0 && pending[stream] <= own[stream]) pending[stream] = 0;
    }
    if (TokenEmpty(pending) || settled(pending)) return false;
    waitFor(pending);
    return true;
}

void ResidencyManager::refresh(std::vector<Fill> fills) {
    if (deferCopyIn && !catchingUp) {
        fills = deferFills(fills);
        if (fills.empty()) return;
    }
    std::vector<std::uint32_t> ids;
    for (const auto& fill : fills) {
        if (fill.begin == 0 && fill.end == WholeEnd) ids.push_back(fill.id);
    }
    auto indices = indicesOf(fills);
    if (settleUnbounded(ids, true)) {
        fills = stillResident(fills, indices);
        indices = indicesOf(fills);
    }
    if (waitWriters(fills)) fills = stillResident(fills, indices);
    refreshOrdered(fills);
}

void ResidencyManager::refreshOrdered(std::span<const Fill> fills) {
    std::vector<Fill> needed;
    for (const auto& fill : fills) {
        const auto& page = pages[fill.id];
        if (fill.cpu && (RunMask(fill.begin, fill.end) & page.dirty) != 0) {
            ++counts.cpuFillsBesideDirty;
            const auto clean = static_cast<std::uint32_t>(RunMask(fill.begin, fill.end) & ~page.dirty);
            for (std::uint32_t bits = clean; bits != 0;) {
                const auto first = static_cast<std::uint32_t>(std::countr_zero(bits));
                const auto run = static_cast<std::uint32_t>(std::countr_one(bits >> first));
                needed.push_back({fill.id, first << ResidencySubpageShift, (first + run) << ResidencySubpageShift, true});
                ++counts.refreshed;
                bits &= run >= 32 ? 0u : ~(((1u << run) - 1u) << first);
            }
            continue;
        }
        if ((RunMask(fill.begin, fill.end) & page.dirty) != 0) {
            const bool overlaps = std::any_of(page.dirtyRuns.begin(), page.dirtyRuns.end(), [&](const Run& run) { return run.first < fill.end && run.second > fill.begin; });
            if (overlaps) ++counts.conflicts;
        }
        needed.push_back(fill);
        ++counts.refreshed;
    }
    copyIn(needed);
}

void ResidencyManager::applyDrops() {
    std::vector<ResidencyRange> queue;
    {
        const std::lock_guard lock(queueMutex);
        queue.swap(dropQueue);
        queued.store(!cpuQueue.empty() || !gpuQueue.empty() || !refusalQueue.empty(), std::memory_order_release);
    }
    for (const auto& range : queue) {
        for (const auto id : residentIn(range.address, range.bytes)) {
            const auto page = pages[id].index;
            const auto import = device.Import(page);
            if (import.whole && import.serial == pages[id].importSerial) continue;
            ++counts.dropped;
            if (pages[id].accessGuarded) {
                device.UnguardAccess(page);
                accessGuards.erase(page);
            }
            retiring.push_back({pages[id].slot, MergeTokens(pages[id].pin, device.Newest())});
            eraseUnsettled(writingBack, page, UnsettledWriteBack);
            forget(id);
        }
        const auto span = PagesOf(range.address, range.bytes);
        if (span.empty) continue;
        std::vector<std::uint64_t> candidates;
        {
            const std::shared_lock lock(ownedMutex);
            for (const auto& [index, record] : owned) {
                if (index >= span.first && index <= span.last) candidates.push_back(index);
            }
        }
        std::vector<std::uint64_t> dropped;
        for (const auto index : candidates) {
            const auto import = device.Import(index);
            const auto id = find(index);
            if (import.whole && id && import.serial == pages[*id].importSerial) continue;
            dropped.push_back(index);
        }
        if (!dropped.empty()) {
            const std::unique_lock lock(ownedMutex);
            for (const auto index : dropped) {
                if (staleOwned.erase(index) != 0) --staleCount;
                owned.erase(index);
                publishStale(index, 0);
            }
            ownershipSerial.fetch_add(1, std::memory_order_acq_rel);
        }
        eraseUnsettledIf(writingBack, UnsettledWriteBack, [&](const auto& entry) { return entry.first >= span.first && entry.first <= span.last; });
        std::erase_if(hot, [&](const auto& entry) { return entry.first >= span.first && entry.first <= span.last; });
        refused.erase(refused.lower_bound(span.first), refused.upper_bound(span.last));
        ++reuseVersion;
        for (auto it = accessGuards.begin(); it != accessGuards.end();) {
            if (*it < span.first || *it > span.last) {
                ++it;
                continue;
            }
            device.UnguardAccess(*it);
            if (const auto id = find(*it)) {
                pages[*id].accessGuarded = false;
                pages[*id].accessMask = 0;
                exposeToWriters(*id);
            }
            it = accessGuards.erase(it);
        }
    }
}

void ResidencyManager::QueueGpuWritten(std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return;
    const std::lock_guard lock(queueMutex);
    gpuQueue.push_back({address, bytes});
    queued.store(true, std::memory_order_release);
}

void ResidencyManager::applyQueued() {
    applyDrops();
    auto applied = unboundedApplied;
    while (!unboundedWriters.empty() && settled(unboundedWriters.front().token)) {
        applied = unboundedWriters.front().serial;
        unboundedWriters.pop_front();
    }
    std::vector<ResidencyRange> gpuWritten;
    {
        const std::lock_guard lock(queueMutex);
        gpuWritten.swap(gpuQueue);
    }
    for (const auto& range : gpuWritten) applyRecorded(range.address, range.bytes);
    if (!gpuWritten.empty() || applied != unboundedApplied) ++reuseVersion;
    unboundedApplied = std::max(unboundedApplied, applied);
    liftUnboundedGuards();
    std::vector<std::uint64_t> written;
    std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint32_t>> clears;
    {
        const std::lock_guard lock(queueMutex);
        written.swap(cpuQueue);
        clears.swap(refusalQueue);
        queued.store(!dropQueue.empty() || !gpuQueue.empty(), std::memory_order_release);
    }
    if (!clears.empty() || !written.empty()) ++reuseVersion;
    if (!clears.empty()) applyRefusalClears(clears);
    if (!written.empty()) applyFaults(std::move(written));
}

void ResidencyManager::applyRecorded(std::uint64_t address, std::uint64_t bytes) {
    const auto span = PagesOf(address, bytes);
    std::vector<Fill> imported;
    std::vector<std::uint32_t> unguardable;
    for (const auto& fill : withAliases(address, bytes)) {
        auto& page = pages[fill.id];
        const bool own = !span.empty && page.index >= span.first && page.index <= span.last;
        if (!own || !unboundedPending(page)) {
            imported.push_back(fill);
            continue;
        }
        page.used = true;
        unboundedStored.push_back(page.index);
        const bool guardable = dirtyable(page) && !aliasKeys.contains(page.index);
        if (!guardable || !guardDirty(fill.id, fill.begin, fill.end)) unguardable.push_back(fill.id);
        markDirty(fill.id, fill.begin, fill.end);
        const auto writes = device.PageWrites(page.index, page.generation);
        if (writes.cpu == 0) page.generation = std::max(page.generation, writes.generation);
    }
    if (!unguardable.empty()) writeBackDirty(unguardable);
    const auto fills = demoteUnused(imported);
    if (fills.empty()) return;
    counts.recordRefreshed += fills.size();
    refreshOrdered(fills);
}

bool ResidencyManager::guardDirty(std::uint32_t id, std::uint32_t begin, std::uint32_t end) {
    auto& page = pages[id];
    const auto adding = static_cast<ResidencyMask>(RunMask(begin, end) & ~page.accessMask);
    if (adding == 0) return true;
    if (const auto refusal = device.GuardAccess(page.index, adding); refusal != ResidencyRefusal::None) {
        ++counts.accessUnguardable;
        noteRefused(page.index, refusal);
        return false;
    }
    page.accessMask = static_cast<ResidencyMask>(page.accessMask | adding);
    page.accessGuarded = true;
    accessGuards.insert(page.index);
    return true;
}

void ResidencyManager::liftUnboundedGuards() {
    std::vector<std::uint32_t> unguardable;
    std::erase_if(unboundedStored, [&](std::uint64_t index) {
        const auto id = find(index);
        if (id && unboundedPending(pages[*id])) return false;
        if (writingBack.contains(index) || isStale(index)) return true;
        if (!id) {
            if (accessGuards.erase(index) != 0) device.UnguardAccess(index);
            return true;
        }
        auto& page = pages[*id];
        if (!page.accessGuarded || page.accessMask == page.dirty) return true;
        device.UnguardAccess(index);
        page.accessMask = 0;
        page.accessGuarded = false;
        exposeToWriters(*id);
        const auto refusal = page.dirty != 0 ? device.GuardAccess(index, page.dirty) : ResidencyRefusal::None;
        if (page.dirty != 0 && refusal == ResidencyRefusal::None) {
            page.accessMask = page.dirty;
            page.accessGuarded = true;
            return true;
        }
        accessGuards.erase(index);
        if (page.dirty != 0) {
            ++counts.accessUnguardable;
            noteRefused(index, refusal);
            unguardable.push_back(*id);
        }
        rearm(*id);
        return true;
    });
    if (!unguardable.empty()) writeBackDirty(unguardable);
}

bool ResidencyManager::settleUnbounded(std::span<const std::uint32_t> taken, bool guard) {
    auto cover = unboundedIssued;
    for (const auto id : taken) {
        if (unboundedPending(pages[id])) cover = std::min(cover, pages[id].unboundedSerial);
    }
    if (cover == unboundedIssued) return false;
    const auto pending = unboundedSince(cover);
    const auto indices = indicesOf(taken);
    const bool released = !settled(pending);
    if (released) waitFor(pending);
    const auto ids = released ? stillResident(taken, indices) : std::vector<std::uint32_t>(taken.begin(), taken.end());
    std::vector<ResidencyRange> recorded;
    {
        const std::lock_guard lock(queueMutex);
        recorded = gpuQueue;
    }
    std::vector<std::uint32_t> unguarded;
    for (const auto id : ids) {
        auto& page = pages[id];
        if (!unboundedPending(page)) continue;
        page.unboundedSerial = unboundedIssued;
        const auto base = page.index << ResidencyPageShift;
        for (const auto& range : recorded) {
            const auto first = std::max(range.address, base);
            const auto last = std::min(range.address + range.bytes, base + ResidencyPageBytes);
            if (first >= last) continue;
            const auto begin = static_cast<std::uint32_t>(first - base);
            const auto end = static_cast<std::uint32_t>(last - base);
            page.used = true;
            const bool guardable = dirtyable(page) && !aliasKeys.contains(page.index);
            if (guard && (!guardable || !guardDirty(id, begin, end)) && std::find(unguarded.begin(), unguarded.end(), id) == unguarded.end()) unguarded.push_back(id);
            markDirty(id, begin, end);
        }
    }
    if (!unguarded.empty()) writeBackDirty(unguarded);
    return released;
}

void ResidencyManager::applyFaults(std::vector<std::uint64_t> written) {
    std::sort(written.begin(), written.end());
    written.erase(std::unique(written.begin(), written.end()), written.end());
    const auto window = hotFrames >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << hotFrames) - 1;
    const auto now = Frame();
    std::vector<Fill> fills;
    std::vector<std::pair<std::uint32_t, std::uint64_t>> partial;
    std::vector<Fill> unused;
    for (const auto index : written) {
        const auto id = find(index);
        if (!id) continue;
        auto& page = pages[*id];
        if (!page.used) {
            unused.push_back({*id});
            continue;
        }
        ++counts.faults;
        if (hotResident.contains(index)) ++counts.hotRefreshed;
        const auto distance = now - page.faultFrame;
        page.faultHistory = (distance >= 64 ? 0 : page.faultHistory << distance) | 1u;
        page.faultFrame = now;
        if (static_cast<std::uint32_t>(std::popcount(page.faultHistory & window)) >= hotFaults && !hotResident.contains(index)) {
            ++counts.hot;
            leaveForImport(index, now);
            page.guarded = false;
        } else if (hotResident.contains(index)) {
            markHot(index, now);
            page.guarded = false;
        } else if (page.guarded) {
            if (const auto refusal = device.Guard(index); refusal != ResidencyRefusal::None) {
                ++counts.unguardable;
                page.guarded = false;
                noteRefused(index, refusal);
                leaveForImport(index, now);
            }
        }
        const auto writes = device.PageWrites(index, page.generation);
        if (writes.other) {
            fills.push_back({*id});
            continue;
        }
        std::uint32_t bits = writes.cpu;
        while (bits != 0) {
            const auto first = static_cast<std::uint32_t>(std::countr_zero(bits));
            const auto run = static_cast<std::uint32_t>(std::countr_one(bits >> first));
            fills.push_back({*id, first << ResidencySubpageShift, (first + run) << ResidencySubpageShift, true});
            bits &= run >= 32 ? 0u : ~(((1u << run) - 1u) << first);
        }
        partial.emplace_back(*id, writes.generation);
    }
    std::vector<std::uint64_t> partialPages;
    partialPages.reserve(partial.size());
    for (const auto& entry : partial) partialPages.push_back(pages[entry.first].index);
    const auto unusedPages = indicesOf(unused);
    refresh(std::move(fills));
    for (std::size_t at = 0; at < partial.size(); ++at) {
        const auto id = find(partialPages[at]);
        if (id && *id == partial[at].first) pages[*id].generation = std::max(pages[*id].generation, partial[at].second);
    }
    demoteUnused(stillResident(unused, unusedPages));
}

void ResidencyManager::coolHot(std::uint64_t first, std::span<const std::uint8_t> written) {
    const auto generation = device.Generation();
    const auto now = Frame();
    for (std::size_t k = 0; k < written.size(); ++k) {
        const auto entry = hot.find(first + k);
        if (entry == hot.end()) continue;
        if (written[k] != 0) {
            const auto gap = now - entry->second.frame;
            ++counts.hotWriteGaps[gap < 5 ? gap : gap <= 8 ? 5 : gap <= 16 ? 6 : 7];
            entry->second = {generation, now};
            continue;
        }
        if (now - entry->second.frame < quietFrames) continue;
        hot.erase(entry);
        ++reuseVersion;
        ++counts.cooled;
    }
}

bool ResidencyManager::Advance() {
    epoch.fetch_add(1, std::memory_order_relaxed);
    return queued.load(std::memory_order_acquire);
}

void ResidencyManager::NextFrame() {
    frame.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t ResidencyManager::Frame() const {
    return frame.load(std::memory_order_relaxed);
}

void ResidencyManager::Apply() {
    const std::lock_guard structureLock(structure);
    settleImports();
    applyQueued();
    retireReached();
    settleAccessGuards();
    if (budget == ResidencyMeasuredBudget && device.OtherDeviceBytes() != appliedOthers) ApplyBudget();
    admitReservedTouches();
    serveHotFromImports();
}

void ResidencyManager::AdmitEager() {
    const std::lock_guard structureLock(structure);
    applyQueued();
    admitEager();
}

void ResidencyManager::admitReservedTouches() {
    std::vector<std::uint64_t> waiting;
    for (std::size_t first = 0; first < reservedTouches.size();) {
        auto last = first;
        while (last + 1 < reservedTouches.size() && reservedTouches[last + 1] == reservedTouches[last] + 1) ++last;
        std::uint64_t quota = last - first + 1;
        const auto before = counts.admitted;
        const auto next = admitPages(reservedTouches[first], reservedTouches[last], &quota, false);
        counts.touchAdmitted += counts.admitted - before;
        for (auto index = first; index <= last; ++index) {
            if (reservedTouches[index] >= next) waiting.push_back(reservedTouches[index]);
            else if (const auto id = find(reservedTouches[index])) pages[*id].touchFrame = touchFrame;
        }
        first = last + 1;
    }
    reservedTouches = std::move(waiting);
}

void ResidencyManager::admitEager() {
    constexpr std::uint64_t PagesPerCall = 512;
    std::uint64_t generation = importsGeneration;
    auto imports = device.WholeImports(generation);
    if (generation != importsGeneration) {
        importsGeneration = generation;
        pendingImports = std::move(imports);
    }
    std::uint64_t quota = PagesPerCall;
    while (!pendingImports.empty() && quota != 0) {
        const auto range = pendingImports.back();
        const auto first = range.address >> ResidencyPageShift;
        const auto last = (range.address + range.bytes - 1) >> ResidencyPageShift;
        auto& entry = pendingImports.back();
        const auto next = admitPages(first, last, &quota, false, true);
        if (next <= last && quota != 0) {
            pendingImports.clear();
            break;
        }
        if (next <= last) {
            entry.address = next << ResidencyPageShift;
            entry.bytes = (last + 1 - next) << ResidencyPageShift;
            break;
        }
        pendingImports.pop_back();
    }
}

void ResidencyManager::Prepare() {
    const std::lock_guard structureLock(structure);
    Advance();
    Apply();
    admitEager();
}

void ResidencyManager::SetBudget(std::uint64_t bytes) {
    const std::lock_guard structureLock(structure);
    budget = bytes;
    measuredEpoch = 0;
    ApplyBudget();
}

void ResidencyManager::ApplyBudget() {
    const std::lock_guard structureLock(structure);
    applyQueued();
    measuredEpoch = 0;
    appliedOthers = device.OtherDeviceBytes();
    const auto before = maxChunks;
    const auto shrunk = chunkLimit(false);
    maxChunks = shrunk < liveChunks ? shrunk : std::max(liveChunks, chunkLimit(true));
    limitEpoch = Epoch();
    if (maxChunks > before) importsGeneration = 0;
    std::uint32_t ordinal = 0;
    for (auto& record : chunks) {
        if (!record.live) continue;
        record.draining = ordinal >= maxChunks;
        ++ordinal;
    }
    for (std::uint32_t chunk = 0; chunk < chunks.size(); ++chunk) {
        if (chunks[chunk].live && chunks[chunk].draining && chunks[chunk].freeSlots == chunkSlots) releaseChunk(chunk);
    }
    evictDraining();
}

void ResidencyManager::evictDraining() {
    if (liveChunks <= maxChunks) return;
    std::vector<std::uint64_t> stale;
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (chunks[pages[id].slot.chunk].draining && isStale(pages[id].index)) stale.push_back(pages[id].index);
    }
    retileStale(stale);
    std::vector<std::uint32_t> leaving;
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (chunks[pages[id].slot.chunk].draining && settled(pages[id].pin) && !unboundedPending(pages[id])) leaving.push_back(id);
    }
    WriteBacks writeBacks;
    for (const auto id : leaving) {
        ++counts.evicted;
        leave(id, writeBacks);
    }
    recordWriteBacks(writeBacks);
    retireReached();
}

void ResidencyManager::Pin(std::uint64_t address, std::uint64_t bytes, ResidencyAccess access, const ResidencyToken& token) {
    const std::lock_guard structureLock(structure);
    for (const auto id : residentIn(address, bytes)) {
        auto& page = pages[id];
        page.used = true;
        page.pin = MergeTokens(page.pin, token);
        if (access == ResidencyAccess::Write) {
            ++reuseVersion;
            addUnsettled(writers, page.index, token, UnsettledWriter);
        }
    }
}

void ResidencyManager::Use(std::uint64_t address, std::uint64_t bytes, ResidencyAccess access) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    const auto span = PagesOf(address, bytes);
    if (span.empty) return;
    const auto count = span.last - span.first + 1;
    const auto present = residentIn(address, bytes);
    std::vector<std::uint64_t> generations(count, ResidencyUnchecked);
    bool check = !present.empty();
    for (const auto id : present) generations[pages[id].index - span.first] = pages[id].generation;
    if (!hot.empty()) {
        for (std::uint64_t k = 0; k < count; ++k) {
            const auto entry = hot.find(span.first + k);
            if (entry == hot.end()) continue;
            generations[k] = entry->second.generation;
            check = true;
        }
    }
    if (check) {
        std::vector<std::uint8_t> stale(count, 0);
        device.ChangedPages(span.first, generations, stale);
        std::vector<Fill> fills;
        for (const auto id : present) {
            unlink(id);
            linkNewest(id);
            if (stale[pages[id].index - span.first] != 0) fills.push_back({id, 0, WholeEnd, true});
            stale[pages[id].index - span.first] = 0;
        }
        refresh(fills);
        if (!hot.empty()) coolHot(span.first, stale);
    }
    if (present.size() < count) {
        std::vector<std::uint32_t> aliases;
        for (const auto& fill : withAliases(address, bytes)) {
            if ((pages[fill.id].index < span.first || pages[fill.id].index > span.last) && !pages[fill.id].dirtyRuns.empty()) aliases.push_back(fill.id);
        }
        writeBackDirty(aliases);
        admit(span.first, span.last);
    }
    Pin(address, bytes, access, device.Current());
}

ResidencyRangeState ResidencyManager::UseRange(std::uint64_t address, std::uint64_t bytes, std::uint64_t largest) {
    bool stable = false;
    return useRange(address, bytes, largest, stable);
}

ResidencyRangeState ResidencyManager::useRange(std::uint64_t address, std::uint64_t bytes, std::uint64_t largest, bool& stable) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    const auto span = PagesOf(address, bytes);
    stable = true;
    if (span.empty) return ResidencyRangeState::Uniform;
    if (bytes > largest && !anyResident(span.first, span.last)) return ResidencyRangeState::Uniform;
    std::vector<std::uint64_t> served;
    bool admissible = true;
    bool mixed = false;
    for (auto page = span.first; page <= span.last; ++page) {
        if (find(page)) {
            mixed = true;
            continue;
        }
        const auto import = device.Import(page);
        if (!import.whole && !import.resident && import.serial == 0) continue;
        served.push_back(page);
        if (!import.whole || import.resident || hot.contains(page) || refused.contains(page) || (import.aliased && canonical.contains({import.section, import.offset}))) admissible = false;
    }
    if (served.empty()) {
        if (mixed) Use(address, bytes, ResidencyAccess::Read);
        return ResidencyRangeState::Uniform;
    }
    if (mixed) ++counts.mixedRanges;
    if (admissible && bytes <= budgetBytes() && (mixed || bytes <= largest)) {
        Use(address, bytes, ResidencyAccess::Read);
        if (std::all_of(served.begin(), served.end(), [&](std::uint64_t page) { return find(page).has_value(); })) {
            if (!mixed) return ResidencyRangeState::Uniform;
            ++counts.completedRanges;
            return ResidencyRangeState::Completed;
        }
        stable = false;
        if (!mixed) ++counts.mixedRanges;
    } else if (!mixed) {
        return ResidencyRangeState::Uniform;
    }
    std::vector<std::uint32_t> clean;
    bool kept = false;
    for (auto page = span.first; page <= span.last; ++page) {
        const auto id = find(page);
        if (!id) continue;
        if (pages[*id].dirtyRuns.empty()) clean.push_back(*id);
        else kept = true;
    }
    demote(clean);
    stable = false;
    if (kept) {
        Pin(address, bytes, ResidencyAccess::Read, device.Current());
        ++counts.rangeConflicts;
        return ResidencyRangeState::Conflict;
    }
    ++counts.demotedRanges;
    return ResidencyRangeState::Demoted;
}

void ResidencyManager::UseRanges(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, std::uint64_t largest) {
    const std::lock_guard structureLock(structure);
    const auto stamp = device.ReuseStamp();
    std::uint64_t key = 0xcbf29ce484222325ull;
    for (const auto& [begin, end] : ranges) key = ((key ^ begin) * 0x100000001b3ull ^ end) * 0x100000001b3ull;
    const auto keyed = [&] { return stamp && !queued.load(std::memory_order_acquire); };
    if (keyed()) {
        if (reuseKeyVersion != reuseVersion || reuseFrame != Frame() || reuseEpoch != Epoch() || !(reuseStamp == *stamp)) {
            reused.clear();
            reuseKeyVersion = reuseVersion;
            reuseFrame = Frame();
            reuseEpoch = Epoch();
            reuseStamp = *stamp;
        } else if (const auto found = reused.find(key); found != reused.end() && std::ranges::equal(found->second, ranges)) {
            ++counts.rangeSetsReused;
            return;
        }
    }
    const auto before = reuseVersion;
    bool stable = true;
    for (const auto& [begin, end] : ranges) {
        if (end <= begin) continue;
        bool rangeStable = false;
        useRange(begin, end - begin, largest, rangeStable);
        stable = stable && rangeStable;
    }
    ++counts.rangeSetsUsed;
    if (!stable || !keyed() || reuseVersion != before || reuseKeyVersion != before || reuseFrame != Frame() || reuseEpoch != Epoch()) return;
    const auto after = device.ReuseStamp();
    if (!after || !(*after == *stamp)) return;
    if (reused.size() >= 256) reused.clear();
    reused.insert_or_assign(key, std::vector<std::pair<std::uint64_t, std::uint64_t>>(ranges.begin(), ranges.end()));
}

void ResidencyManager::AdmitTouched(std::span<const std::uint64_t> touched) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    ++touchFrame;
    reservedTouches.clear();
    std::vector<std::uint64_t> sorted(touched.begin(), touched.end());
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    std::vector<std::uint64_t> missing;
    for (const auto page : sorted) {
        const auto id = find(page);
        if (!id) {
            missing.push_back(page);
            continue;
        }
        pages[*id].touchFrame = touchFrame;
        pages[*id].used = true;
        unlink(*id);
        linkNewest(*id);
    }
    coolTouched(missing);
    std::uint64_t quota = TouchPagesPerFrame;
    for (std::size_t first = 0; first < missing.size();) {
        auto last = first;
        while (last + 1 < missing.size() && missing[last + 1] == missing[last] + 1) ++last;
        if (quota == 0) {
            counts.touchDeferred += last - first + 1;
            first = last + 1;
            continue;
        }
        const auto before = counts.admitted;
        const auto next = admitPages(missing[first], missing[last], &quota, true, false, &reservedTouches);
        const auto admitted = counts.admitted - before;
        counts.touchAdmitted += admitted;
        if (next <= missing[last]) counts.touchDeferred += missing[last] - next + 1;
        for (auto index = first; index <= last; ++index) {
            if (const auto id = find(missing[index])) pages[*id].touchFrame = touchFrame;
        }
        first = last + 1;
    }
}

void ResidencyManager::admit(std::uint64_t first, std::uint64_t last) {
    admitPages(first, last, nullptr);
}

std::uint64_t ResidencyManager::admitPages(std::uint64_t first, std::uint64_t last, std::uint64_t* quota, bool evict, bool sharedOnly, std::vector<std::uint64_t>* reserved) {
    WriteBacks writeBacks;
    std::vector<Fill> fills;
    bool exhausted = false;
    auto page = first;
    for (; page <= last; ++page) {
        if (quota != nullptr && *quota == 0) break;
        if (find(page) || hot.contains(page) || refused.contains(page)) continue;
        const auto import = device.Import(page);
        if (!import.whole) {
            ++counts.inadmissible;
            continue;
        }
        if (import.resident) {
            ++counts.residentSkipped;
            continue;
        }
        if (sharedOnly && import.section == 0) continue;
        const auto key = std::pair{import.section, import.offset};
        if (import.aliased && canonical.contains(key)) {
            ++counts.aliasSkipped;
            continue;
        }
        std::optional<MirrorSlot> slot;
        bool evicted = false;
        if (!exhausted) {
            if (evict) {
                slot = acquireSlot(first, last, writeBacks, &evicted);
            } else {
                retireReached();
                slot = takeFree();
                if (!slot && allocateChunk()) slot = takeFree();
            }
        }
        if (!slot && evicted && reserved != nullptr) {
            reserved->push_back(page);
            if (quota != nullptr) --*quota;
            continue;
        }
        if (!slot) {
            ++counts.deferred;
            exhausted = exhausted || !evict || !victim(first, last);
            if (quota != nullptr) break;
            continue;
        }
        recordWriteBacks(writeBacks);
        bool guarded = false;
        if (import.cpuWritable && import.section != 0) {
            if (const auto refusal = device.Guard(page); refusal != ResidencyRefusal::None) {
                ++counts.unguardable;
                noteRefused(page, refusal);
                freeSlot(*slot);
                continue;
            }
            guarded = true;
        } else if (import.cpuWritable) {
            device.SettleWrites(page);
        }
        const auto id = newPage(page, *slot, import, guarded);
        pages[id].used = !sharedOnly;
        if (import.cpuWritable && !guarded) {
            ++counts.unsettledGuards;
            markHot(page, Epoch());
        }
        fills.push_back({id});
        if (import.aliased) {
            canonical.emplace(key, page);
            aliasKeys.emplace(page, key);
        }
        ++counts.admitted;
        if (quota != nullptr) --*quota;
    }
    recordWriteBacks(writeBacks);
    copyIn(fills);
    return page;
}

void ResidencyManager::demote(std::span<const std::uint32_t> taken) {
    const auto indices = indicesOf(taken);
    auto ids = settleUnbounded(taken, false) ? stillResident(taken, indices) : std::vector<std::uint32_t>(taken.begin(), taken.end());
    std::vector<std::uint64_t> stale;
    for (const auto id : ids) {
        if (isStale(pages[id].index)) stale.push_back(pages[id].index);
    }
    if (retileStale(stale)) ids = stillResident(taken, indices);
    ResidencyToken pending{};
    for (const auto id : ids) {
        const auto writer = writers.find(pages[id].index);
        if (writer != writers.end() && !settled(writer->second)) pending = MergeTokens(pending, writer->second);
    }
    if (!TokenEmpty(pending)) {
        waitFor(pending);
        ids = stillResident(taken, indices);
    }
    WriteBacks writeBacks;
    for (const auto id : ids) {
        ++counts.demoted;
        leave(id, writeBacks);
    }
    recordWriteBacks(writeBacks);
}

std::vector<ResidencyManager::Fill> ResidencyManager::demoteUnused(std::span<const Fill> fills) {
    std::vector<Fill> kept;
    std::vector<std::uint32_t> unused;
    for (const auto& fill : fills) {
        if (pages[fill.id].used) kept.push_back(fill);
        else unused.push_back(fill.id);
    }
    if (unused.empty()) return kept;
    std::sort(unused.begin(), unused.end());
    unused.erase(std::unique(unused.begin(), unused.end()), unused.end());
    counts.unusedDemoted += unused.size();
    const auto keptPages = indicesOf(kept);
    demote(unused);
    return stillResident(kept, keptPages);
}

void ResidencyManager::Demote(std::uint64_t address, std::uint64_t bytes) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    std::vector<std::uint32_t> ids;
    for (const auto& fill : withAliases(address, bytes)) ids.push_back(fill.id);
    demote(ids);
}

void ResidencyManager::DemoteWritable() {
    const std::lock_guard structureLock(structure);
    applyQueued();
    std::vector<std::uint32_t> ids;
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (pages[id].gpuWritable) ids.push_back(id);
    }
    demote(ids);
}

void ResidencyManager::NoteUnboundedWriter(const ResidencyToken& token) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    std::vector<std::uint32_t> unguardable;
    for (std::size_t word = 0; writerExposedCount != 0 && word < writerExposed.size(); ++word) {
        for (auto bits = std::exchange(writerExposed[word], 0); bits != 0; bits &= bits - 1) {
            --writerExposedCount;
            const auto id = static_cast<std::uint32_t>(word * 64 + static_cast<std::size_t>(std::countr_zero(bits)));
            auto& page = pages[id];
            if (!dirtyable(page) || aliasKeys.contains(page.index) || !guardDirty(id, 0, WholeEnd)) {
                unguardable.push_back(id);
                continue;
            }
            publishUnsettled(page.index, UnsettledUnbounded, true);
        }
    }
    demote(unguardable);
    unboundedWriters.push_back({++unboundedIssued, token});
}

ResidencyToken ResidencyManager::unboundedSince(std::uint64_t serial) const {
    ResidencyToken pending{};
    for (const auto& writer : unboundedWriters) {
        if (writer.serial > serial) pending = MergeTokens(pending, writer.token);
    }
    return pending;
}

void ResidencyManager::exposeToWriters(std::uint32_t id) {
    if (!pages[id].gpuWritable) return;
    if (id / 64 >= writerExposed.size()) writerExposed.resize(id / 64 + 1);
    const auto bit = std::uint64_t{1} << (id % 64);
    if ((writerExposed[id / 64] & bit) != 0) return;
    writerExposed[id / 64] |= bit;
    ++writerExposedCount;
}

void ResidencyManager::DropRange(std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return;
    const std::lock_guard lock(queueMutex);
    dropQueue.push_back({address, bytes});
    queued.store(true, std::memory_order_release);
}

void ResidencyManager::NoteImportWritten(std::uint64_t address, std::uint64_t bytes) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    auto fills = demoteUnused(withAliases(address, bytes));
    const auto indices = indicesOf(fills);
    if (waitWriters(fills)) fills = stillResident(fills, indices);
    for (const auto& fill : fills) clearDirty(fill.id, fill.begin, fill.end);
    refresh(std::move(fills));
}

std::vector<ResidencyMirrorStore> ResidencyManager::StoreThroughMirrors(std::uint64_t address, std::uint64_t bytes, const ResidencyToken& lands) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    std::vector<ResidencyMirrorStore> stores;
    if (bytes == 0) return stores;
    if (bytes > ResidencyPageBytes) throw std::invalid_argument("residency: a store through the mirrors spans more than one page: " + std::to_string(bytes) + " bytes");
    auto spans = device.Aliases(address, bytes);
    spans.push_back({address, bytes});
    std::vector<Fill> fills;
    std::vector<Fill> unused;
    for (const auto& range : spans) {
        for (const auto id : residentIn(range.address, range.bytes)) {
            if (!pages[id].used) {
                unused.push_back({id});
                continue;
            }
            const auto base = pages[id].index << ResidencyPageShift;
            const auto first = std::max(range.address, base);
            const auto last = std::min(range.address + range.bytes, base + ResidencyPageBytes);
            const auto begin = static_cast<std::uint32_t>(first - base);
            if (std::any_of(fills.begin(), fills.end(), [&](const Fill& fill) { return fill.id == id && fill.begin == begin; })) continue;
            fills.push_back({id, begin, static_cast<std::uint32_t>(last - base)});
            stores.push_back({pages[id].slot, begin, static_cast<std::uint32_t>((first - address) & (ResidencyPageBytes - 1)), static_cast<std::uint32_t>(last - first)});
        }
    }
    const auto keepResident = [&](std::span<const std::uint64_t> indices) {
        std::vector<Fill> liveFills;
        std::vector<ResidencyMirrorStore> liveStores;
        for (std::size_t at = 0; at < fills.size(); ++at) {
            const auto id = find(indices[at]);
            if (!id || *id != fills[at].id) continue;
            liveFills.push_back(fills[at]);
            liveStores.push_back(stores[at]);
        }
        fills = std::move(liveFills);
        stores = std::move(liveStores);
    };
    const auto waited = indicesOf(fills);
    const auto unusedPages = indicesOf(unused);
    if (waitWriters(fills)) {
        keepResident(waited);
        unused = stillResident(unused, unusedPages);
    }
    for (const auto& fill : fills) {
        clearDirty(fill.id, fill.begin, fill.end);
        pages[fill.id].pin = MergeTokens(pages[fill.id].pin, lands);
    }
    demoteUnused(unused);
    return stores;
}

void ResidencyManager::RefreshImportWritten(std::uint64_t address, std::uint64_t bytes) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    const auto fills = demoteUnused(withAliases(address, bytes));
    for (const auto& fill : fills) clearDirty(fill.id, fill.begin, fill.end);
    refreshOrdered(fills);
}

void ResidencyManager::QueueCpuWritten(std::span<const std::uint64_t> written) {
    if (written.empty()) return;
    const std::lock_guard lock(queueMutex);
    cpuQueue.insert(cpuQueue.end(), written.begin(), written.end());
    queued.store(true, std::memory_order_release);
}

void ResidencyManager::NoteGpuWritten(std::uint64_t address, std::uint64_t bytes, bool written) {
    const std::lock_guard structureLock(structure);
    ++reuseVersion;
    const auto span = PagesOf(address, bytes);
    if (span.empty) return;
    std::vector<std::uint32_t> unguardable;
    std::vector<std::uint32_t> aliases;
    std::vector<Fill> stored;
    for (const auto& fill : withAliases(address, bytes)) {
        auto& page = pages[fill.id];
        if (page.index < span.first || page.index > span.last) {
            aliases.push_back(fill.id);
            continue;
        }
        if (!dirtyable(page) || aliasKeys.contains(page.index) || !guardDirty(fill.id, fill.begin, fill.end)) {
            unguardable.push_back(fill.id);
            stored.push_back(fill);
            continue;
        }
        markDirty(fill.id, fill.begin, fill.end);
    }
    if (!aliases.empty()) {
        counts.aliasDemoted += aliases.size();
        const auto indices = indicesOf(unguardable);
        demote(aliases);
        std::vector<std::uint32_t> liveIds;
        std::vector<Fill> liveStored;
        for (std::size_t at = 0; at < unguardable.size(); ++at) {
            const auto id = find(indices[at]);
            if (!id || *id != unguardable[at]) continue;
            liveIds.push_back(unguardable[at]);
            liveStored.push_back(stored[at]);
        }
        unguardable = std::move(liveIds);
        stored = std::move(liveStored);
    }
    if (unguardable.empty()) return;
    if (!written) {
        demote(unguardable);
        return;
    }
    for (const auto& fill : stored) markDirty(fill.id, fill.begin, fill.end);
    writeBackDirty(unguardable);
    demote(unguardable);
}

void ResidencyManager::FlushForCpuRead(std::uint64_t address, std::uint64_t bytes) {
    const std::lock_guard structureLock(structure);
    flushForCpuRead(address, bytes, false);
}

bool ResidencyManager::FlushForCpuReadUnlocked(std::uint64_t address, std::uint64_t bytes) {
    const std::lock_guard structureLock(structure);
    return flushForCpuRead(address, bytes, true);
}

bool ResidencyManager::flushForCpuRead(std::uint64_t address, std::uint64_t bytes, bool unlocked) {
    const auto span = PagesOf(address, bytes);
    if (span.empty) {
        applyQueued();
        return true;
    }
    const auto inSpan = [&](std::uint64_t page) { return page >= span.first && page <= span.last; };
    const auto wait = [&](const ResidencyToken& token) {
        if (!unlocked) {
            waitFor(token);
            return true;
        }
        ++counts.waits;
        const auto held = structure.ReleaseAll();
        const bool alive = device.WaitUnlocked(token);
        structure.Reacquire(held);
        return alive;
    };
    for (;;) {
        applyQueued();
        const std::array<const IResidencyOwner*, 1> exempt{cpuReadExempt};
        const bool retiled = retileStale(staleIn(address, bytes), exempt);
        ResidencyToken pending = retiled ? device.Current() : ResidencyToken{};
        for (const auto& [page, token] : writers) {
            if (inSpan(page) && !settled(token)) pending = MergeTokens(pending, token);
        }
        auto cover = unboundedIssued;
        for (const auto id : residentIn(address, bytes)) {
            if (unboundedPending(pages[id])) cover = std::min(cover, pages[id].unboundedSerial);
        }
        if (cover != unboundedIssued) pending = MergeTokens(pending, unboundedSince(cover));
        if (!TokenEmpty(pending) && !settled(pending)) {
            if (!wait(pending)) return false;
            continue;
        }
        std::vector<ResidencyCopy> copies;
        std::vector<std::uint32_t> written;
        for (const auto id : residentIn(address, bytes)) {
            if (pages[id].dirtyRuns.empty()) continue;
            appendDirty(id, copies);
            written.push_back(id);
        }
        ResidencyToken copied{};
        if (!copies.empty()) {
            for (const auto id : written) clearAllDirty(id);
            copied = device.WriteBack(copies);
            ++counts.cpuWriteBacks;
            for (const auto& copy : copies) counts.writtenBackBytes += copy.bytes;
            for (const auto id : written) {
                pages[id].pin = MergeTokens(pages[id].pin, copied);
                addUnsettled(writingBack, pages[id].index, copied, UnsettledWriteBack);
            }
        }
        for (const auto& [page, token] : writingBack) {
            if (inSpan(page)) copied = MergeTokens(copied, token);
        }
        if (settled(copied)) break;
        if (!wait(copied)) return false;
        if (!unlocked) break;
    }
    releaseCpuRead(span.first, span.last, false);
    return true;
}

bool ResidencyManager::FlushForCpuReadSettled(std::uint64_t address, std::uint64_t bytes) {
    const std::lock_guard structureLock(structure);
    const auto span = PagesOf(address, bytes);
    if (span.empty) return true;
    const auto inSpan = [&](std::uint64_t page) { return page >= span.first && page <= span.last; };
    queuedCpuPages.clear();
    if (queued.load(std::memory_order_acquire)) {
        const std::lock_guard queueLock(queueMutex);
        for (const auto& range : dropQueue) {
            const auto dropped = PagesOf(range.address, range.bytes);
            if (!dropped.empty && dropped.first <= span.last && dropped.last >= span.first) return false;
        }
        for (const auto page : cpuQueue) {
            if (inSpan(page)) queuedCpuPages.push_back(page);
        }
    }
    if (!staleIn(address, bytes).empty()) return false;
    for (const auto* entries : {&writers, &writingBack}) {
        for (const auto& [page, token] : *entries) {
            if (inSpan(page) && !TokenEmpty(token) && !device.Signaled(token)) return false;
        }
    }
    for (const auto id : residentIn(address, bytes)) {
        if (!pages[id].dirtyRuns.empty() || unboundedPending(pages[id])) return false;
    }
    withoutRecorder = true;
    releaseCpuRead(span.first, span.last, true);
    withoutRecorder = false;
    queuedCpuPages.clear();
    return true;
}

void ResidencyManager::releaseCpuRead(std::uint64_t firstPage, std::uint64_t lastPage, bool writersSettled) {
    const auto inSpan = [&](std::uint64_t page) { return page >= firstPage && page <= lastPage; };
    eraseUnsettledIf(writingBack, UnsettledWriteBack, [&](const auto& entry) { return inSpan(entry.first); });
    eraseUnsettledIf(writers, UnsettledWriter, [&](const auto& entry) { return inSpan(entry.first) && (writersSettled || settled(entry.second)); });
    for (auto page = firstPage; page <= lastPage; ++page) {
        const auto id = find(page);
        if (!id || !unboundedPending(pages[*id])) publishUnsettled(page, UnsettledUnbounded, false);
        if (id) exposeToWriters(*id);
        if (id && (!pages[*id].dirtyRuns.empty() || unboundedPending(pages[*id]))) continue;
        if (isStale(page)) continue;
        if (!accessGuards.contains(page) && !(id && pages[*id].accessGuarded) && lastPage - firstPage >= 16) continue;
        device.UnguardAccess(page);
        accessGuards.erase(page);
        if (!id) continue;
        pages[*id].accessGuarded = false;
        pages[*id].accessMask = 0;
        exposeToWriters(*id);
        rearm(*id);
    }
}

void ResidencyManager::rearm(std::uint32_t id) {
    auto& page = pages[id];
    if (!page.cpuWritable || page.guarded) return;
    if (hotResident.contains(page.index)) return;
    if (withoutRecorder && std::find(queuedCpuPages.begin(), queuedCpuPages.end(), page.index) != queuedCpuPages.end()) return;
    if (!page.importSettled) {
        ++counts.unsettledGuards;
        markHot(page.index, Epoch());
        return;
    }
    const auto refusal = device.Guard(page.index);
    if (refusal == ResidencyRefusal::None) {
        page.guarded = true;
        return;
    }
    if (!page.shared && !device.ImportSettled(page.index)) {
        unsettle(id);
        return;
    }
    ++counts.unguardable;
    noteRefused(page.index, refusal);
    leaveForImport(page.index, Epoch());
}

bool ResidencyManager::writePending(std::uint64_t page) {
    const auto writer = writers.find(page);
    return writer != writers.end() && !settled(writer->second);
}

void ResidencyManager::writeBackDirty(std::span<const std::uint32_t> ids, bool keepPendingWrites) {
    std::vector<ResidencyCopy> copies;
    std::vector<std::uint32_t> written;
    for (const auto id : ids) {
        if (pages[id].dirtyRuns.empty()) continue;
        appendDirty(id, copies);
        written.push_back(id);
    }
    if (copies.empty()) return;
    for (const auto id : written) {
        if (!keepPendingWrites || !writePending(pages[id].index)) clearAllDirty(id);
    }
    const auto token = device.WriteBack(copies);
    for (const auto& copy : copies) counts.writtenBackBytes += copy.bytes;
    for (const auto id : written) {
        auto& page = pages[id];
        page.pin = MergeTokens(page.pin, token);
        addUnsettled(writingBack, page.index, token, UnsettledWriteBack);
    }
}

void ResidencyManager::PrepareImportWrite(std::uint64_t address, std::uint64_t bytes) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    std::vector<std::uint32_t> ids;
    for (const auto& fill : withAliases(address, bytes)) ids.push_back(fill.id);
    if (ids.empty()) return;
    if (settleUnbounded(ids, true)) {
        ids.clear();
        for (const auto& fill : withAliases(address, bytes)) ids.push_back(fill.id);
    }
    writeBackDirty(ids, true);
}

void ResidencyManager::settleAccessGuards() {
    std::vector<std::uint64_t> settledPages;
    for (const auto& [page, token] : writingBack) {
        if (settled(token)) settledPages.push_back(page);
    }
    for (const auto page : settledPages) {
        eraseUnsettled(writingBack, page, UnsettledWriteBack);
        if (!accessGuards.contains(page)) continue;
        const auto id = find(page);
        if (id && (!pages[*id].dirtyRuns.empty() || unboundedPending(pages[*id]))) continue;
        device.UnguardAccess(page);
        accessGuards.erase(page);
        if (!id) continue;
        pages[*id].accessGuarded = false;
        pages[*id].accessMask = 0;
        exposeToWriters(*id);
        rearm(*id);
    }
}

void ResidencyManager::markHot(std::uint64_t page, std::uint64_t at) {
    const auto [entry, inserted] = hotResident.insert_or_assign(page, at);
    static_cast<void>(entry);
    if (!inserted) return;
    ++reuseVersion;
    auto next = hotRuns.upper_bound(page);
    auto first = page;
    auto last = page;
    if (next != hotRuns.begin()) {
        const auto previous = std::prev(next);
        if (previous->second + 1 == page) {
            first = previous->first;
            hotRuns.erase(previous);
        }
    }
    if (next != hotRuns.end() && next->first == page + 1) {
        last = next->second;
        hotRuns.erase(next);
    }
    hotRuns.emplace(first, last);
}

void ResidencyManager::unmarkHot(std::uint64_t page) {
    if (hotResident.erase(page) == 0) return;
    ++reuseVersion;
    auto run = std::prev(hotRuns.upper_bound(page));
    const auto first = run->first;
    const auto last = run->second;
    hotRuns.erase(run);
    if (first < page) hotRuns.emplace(first, page - 1);
    if (page < last) hotRuns.emplace(page + 1, last);
}

void ResidencyManager::RefreshHot() {
    const std::lock_guard structureLock(structure);
    applyQueued();
    serveHotFromImports();
    if (hotResident.empty()) return;
    const auto now = Epoch();
    if (sweptEpoch != now) {
        sweptEpoch = now;
        ++counts.hotSweeps;
        for (const auto& [first, last] : hotRuns) {
            device.CollectWrites(first, last - first + 1);
            ++counts.hotCollects;
        }
        applyQueued();
    }
    serveHotFromImports();
}

void ResidencyManager::leaveForImport(std::uint64_t page, std::uint64_t at) {
    markHot(page, at);
    leavingForImport.push_back(page);
}

void ResidencyManager::noteRefused(std::uint64_t page, ResidencyRefusal refusal) {
    ++counts.refusals[static_cast<std::size_t>(refusal)];
    refused.insert_or_assign(page, refusal);
    ++reuseVersion;
}

void ResidencyManager::QueueRefusalCleared(std::uint64_t address, std::uint64_t bytes, std::uint32_t cleared) {
    if (bytes == 0) return;
    const std::lock_guard lock(queueMutex);
    refusalQueue.emplace_back(address, bytes, cleared);
    queued.store(true, std::memory_order_release);
}

std::optional<ResidencyRefusal> ResidencyManager::Refused(std::uint64_t page) const {
    const std::lock_guard structureLock(structure);
    const auto found = refused.find(page);
    if (found == refused.end()) return std::nullopt;
    return found->second;
}

void ResidencyManager::applyRefusalClears(std::span<const std::tuple<std::uint64_t, std::uint64_t, std::uint32_t>> clears) {
    std::vector<std::uint32_t> retry;
    for (const auto& [address, bytes, cleared] : clears) {
        const auto span = PagesOf(address, bytes);
        if (span.empty) continue;
        for (auto it = refused.lower_bound(span.first); it != refused.end() && it->first <= span.last;) {
            if ((cleared & ResidencyRefusalBit(it->second)) == 0) {
                ++it;
                continue;
            }
            ++counts.refusalRetries;
            if (const auto id = find(it->first); id && !pages[*id].guarded) retry.push_back(*id);
            it = refused.erase(it);
            ++reuseVersion;
        }
    }
    for (const auto id : retry) {
        if (!hotResident.contains(pages[id].index)) rearm(id);
    }
}

void ResidencyManager::serveHotFromImports() {
    if (leavingForImport.empty()) return;
    std::sort(leavingForImport.begin(), leavingForImport.end());
    leavingForImport.erase(std::unique(leavingForImport.begin(), leavingForImport.end()), leavingForImport.end());
    std::vector<std::uint32_t> ids;
    std::vector<std::uint64_t> served;
    std::vector<std::uint64_t> pinned;
    for (const auto index : leavingForImport) {
        const auto id = find(index);
        if (!id || !hotResident.contains(index) || pages[*id].guarded) continue;
        if (!pages[*id].shared && !pages[*id].importSettled) continue;
        if (!settled(pages[*id].pin)) {
            pinned.push_back(index);
            continue;
        }
        ids.push_back(*id);
        served.push_back(index);
    }
    leavingForImport = std::move(pinned);
    if (ids.empty()) return;
    counts.hotServed += ids.size();
    demote(ids);
    const auto generation = device.Generation();
    const auto now = Frame();
    for (const auto index : served) {
        if (!refused.contains(index)) hot.insert_or_assign(index, Cooling{generation, now});
        ++reuseVersion;
    }
}

void ResidencyManager::coolTouched(std::span<const std::uint64_t> touched) {
    if (hot.empty()) return;
    for (std::size_t first = 0; first < touched.size();) {
        if (!hot.contains(touched[first])) {
            ++first;
            continue;
        }
        auto last = first;
        while (last + 1 < touched.size() && touched[last + 1] == touched[last] + 1 && hot.contains(touched[last + 1])) ++last;
        std::vector<std::uint64_t> generations;
        for (auto index = first; index <= last; ++index) generations.push_back(hot.at(touched[index]).generation);
        std::vector<std::uint8_t> stale(generations.size(), 0);
        device.ChangedPages(touched[first], generations, stale);
        coolHot(touched[first], stale);
        first = last + 1;
    }
}

void ResidencyManager::Retire() {
    const std::lock_guard structureLock(structure);
    settleImports();
    applyQueued();
    retireReached();
    settleAccessGuards();
    admitEager();
    evictDraining();
    eraseUnsettledIf(writers, UnsettledWriter, [&](const auto& entry) { return settled(entry.second); });
    eraseUnsettledIf(writingBack, UnsettledWriteBack, [&](const auto& entry) { return settled(entry.second); });
}

std::optional<MirrorSlot> ResidencyManager::Slot(std::uint64_t page) {
    const std::lock_guard structureLock(structure);
    applyDrops();
    auto id = find(page);
    if (!id) return std::nullopt;
    if (pages[*id].behind != 0) {
        const std::uint32_t ids[] = {*id};
        catchUp(ids, true);
        id = find(page);
        if (!id) return std::nullopt;
    }
    return pages[*id].slot;
}

std::optional<MirrorSlot> ResidencyManager::WriteSlot(std::uint64_t page) {
    const std::lock_guard structureLock(structure);
    applyDrops();
    auto id = find(page);
    if (!id || !dirtyable(pages[*id])) return std::nullopt;
    if (pages[*id].behind != 0) {
        const std::uint32_t ids[] = {*id};
        catchUp(ids, true);
        id = find(page);
        if (!id || !dirtyable(pages[*id])) return std::nullopt;
    }
    return pages[*id].slot;
}

bool ResidencyManager::Used(std::uint64_t page) {
    applyDrops();
    const auto id = find(page);
    return id && pages[*id].used;
}

std::optional<std::uint64_t> ResidencyManager::Translate(std::uint64_t page) {
    const std::lock_guard structureLock(structure);
    const auto slot = Slot(page);
    if (!slot) return std::nullopt;
    return chunks[slot->chunk].address + (std::uint64_t{slot->slot} << ResidencyPageShift);
}

ResidencyMask ResidencyManager::Dirty(std::uint64_t page) {
    const std::lock_guard structureLock(structure);
    applyDrops();
    const auto id = find(page);
    return id ? pages[*id].dirty : ResidencyMask{0};
}

bool ResidencyManager::AnyStaleOwned() const {
    const std::shared_lock lock(ownedMutex);
    return !staleOwned.empty();
}

bool ResidencyManager::AnyResident(std::uint64_t address, std::uint64_t bytes) {
    const auto span = PagesOf(address, bytes);
    if (span.empty) return false;
    const std::lock_guard lock(residentMutex);
    if (span.last - span.first >= resident.size()) {
        return std::any_of(resident.begin(), resident.end(), [&](const auto& entry) { return entry.first >= span.first && entry.first <= span.last; });
    }
    for (auto page = span.first; page <= span.last; ++page) {
        if (resident.contains(page)) return true;
    }
    return false;
}

std::vector<ResidencyRange> ResidencyManager::ResidentRanges() {
    const std::lock_guard structureLock(structure);
    applyDrops();
    if (rangesSerial == serial) return ranges;
    std::vector<std::uint64_t> indices;
    indices.reserve(resident.size());
    for (const auto& entry : resident) indices.push_back(entry.first);
    std::sort(indices.begin(), indices.end());
    ranges.clear();
    for (const auto page : indices) {
        const auto address = page << ResidencyPageShift;
        if (!ranges.empty() && ranges.back().address + ranges.back().bytes == address) ranges.back().bytes += ResidencyPageBytes;
        else ranges.push_back({address, ResidencyPageBytes});
    }
    rangesSerial = serial;
    return ranges;
}

bool ResidencyManager::ChangesSince(std::uint64_t since, std::uint64_t& now, std::vector<std::pair<std::uint64_t, std::uint64_t>>& changed) {
    const std::lock_guard structureLock(structure);
    applyDrops();
    now = serial;
    if (since > serial || serial - since > ChangeLogPages) return false;
    std::vector<std::uint64_t> taken;
    taken.reserve(static_cast<std::size_t>(serial - since));
    for (auto at = since; at < serial; ++at) taken.push_back(changeLog[at % ChangeLogPages]);
    std::sort(taken.begin(), taken.end());
    taken.erase(std::unique(taken.begin(), taken.end()), taken.end());
    changed.clear();
    changed.reserve(taken.size());
    for (const auto page : taken) {
        const auto id = find(page);
        changed.emplace_back(page, id ? chunks[pages[*id].slot.chunk].address + (std::uint64_t{pages[*id].slot.slot} << ResidencyPageShift) : 0u);
    }
    return true;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> ResidencyManager::Mirrored(std::uint64_t& now) {
    const std::lock_guard structureLock(structure);
    applyDrops();
    now = serial;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> mirrored;
    mirrored.reserve(resident.size());
    for (const auto& [page, id] : resident) mirrored.emplace_back(page, chunks[pages[id].slot.chunk].address + (std::uint64_t{pages[id].slot.slot} << ResidencyPageShift));
    std::sort(mirrored.begin(), mirrored.end());
    return mirrored;
}

std::vector<std::uint64_t> ResidencyManager::TakeChanges() {
    const std::lock_guard structureLock(structure);
    applyDrops();
    std::vector<std::uint64_t> taken(changes.begin(), changes.end());
    changes.clear();
    std::sort(taken.begin(), taken.end());
    return taken;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> ResidencyManager::ResidentGenerations() const {
    const std::lock_guard structureLock(structure);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> resident;
    for (auto id = oldest; id != None; id = pages[id].newer) resident.emplace_back(pages[id].index, pages[id].generation);
    return resident;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> ResidencyManager::MirroredIn(std::uint64_t address, std::uint64_t bytes) {
    const std::lock_guard structureLock(structure);
    applyDrops();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> mirrored;
    for (const auto id : residentIn(address, bytes)) {
        if (pages[id].behind == 0) mirrored.emplace_back(pages[id].index, chunks[pages[id].slot.chunk].address + (std::uint64_t{pages[id].slot.slot} << ResidencyPageShift));
    }
    return mirrored;
}

std::vector<std::uint64_t> ResidencyManager::CurrentPages() const {
    const std::lock_guard structureLock(structure);
    std::vector<std::uint64_t> current;
    for (auto id = oldest; id != None; id = pages[id].newer) {
        if (pages[id].behind == 0) current.push_back(pages[id].index);
    }
    std::sort(current.begin(), current.end());
    return current;
}

void ResidencyManager::CatchUp(std::uint64_t address, std::uint64_t bytes) {
    const std::lock_guard structureLock(structure);
    applyQueued();
    std::vector<std::uint32_t> ids;
    for (const auto& fill : withAliases(address, bytes)) {
        if (pages[fill.id].behind != 0) ids.push_back(fill.id);
    }
    catchUp(ids, true);
}

ResidencyMask ResidencyManager::Behind(std::uint64_t page) const {
    const std::lock_guard structureLock(structure);
    const auto id = find(page);
    return id ? pages[*id].behind : ResidencyMask{0};
}

std::vector<std::uint32_t> ResidencyManager::Chunks() const {
    const std::lock_guard structureLock(structure);
    std::vector<std::uint32_t> live;
    for (std::uint32_t chunk = 0; chunk < chunks.size(); ++chunk) {
        if (chunks[chunk].live) live.push_back(chunk);
    }
    return live;
}

std::uint64_t ResidencyManager::ChunkAddress(std::uint32_t chunk) const {
    const std::lock_guard structureLock(structure);
    if (chunk >= chunks.size() || !chunks[chunk].live) throw std::out_of_range("residency: chunk " + std::to_string(chunk) + " is not live");
    return chunks[chunk].address;
}

ResidencyCounts ResidencyManager::Counts() const {
    const std::lock_guard structureLock(structure);
    auto result = counts;
    result.chunks = liveChunks;
    result.resident = static_cast<std::uint32_t>(resident.size());
    result.dirty = dirtyPages.load(std::memory_order_relaxed);
    {
        const std::shared_lock lock(ownedMutex);
        result.ownedPages = static_cast<std::uint32_t>(owned.size());
        result.stalePages = static_cast<std::uint32_t>(staleOwned.size());
    }
    result.accessGuarded = static_cast<std::uint32_t>(accessGuards.size());
    result.hotResident = static_cast<std::uint32_t>(hotResident.size());
    result.refusedPages = static_cast<std::uint32_t>(refused.size());
    result.retiring = static_cast<std::uint32_t>(retiring.size());
    result.free = 0;
    for (const auto& record : chunks) {
        if (record.live) result.free += record.freeSlots;
    }
    return result;
}

}
