#include "prx/libSceAgcDriver/Execution/include/MemoryStats.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "ThreadOwned.hpp"
#include "UnnormalizedWrap.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PlaceholderImages.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VramBudget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <stdexcept>
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libc/include/General.hpp"
#include "Optimization/include/Optimization/ShaderStageInputInfo.hpp"
#include "RdnaDecoder/include/RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "IntermediateRepresentation/IrMetadata/ShaderStage.hpp"
#include "ImageTableAbi.hpp"
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureService.hpp"

namespace AgcDriver::Graphics {
namespace {

bool overlap(std::uint64_t first, std::size_t firstSize, std::uint64_t second, std::size_t secondSize) {
    return first < second + secondSize && second < first + firstSize;
}

VkComponentSwizzle ComponentSwizzleFor(std::uint8_t dstSel) {
    switch (dstSel) {
        case 0: return VK_COMPONENT_SWIZZLE_ZERO;
        case 1: return VK_COMPONENT_SWIZZLE_ONE;
        case 4: return VK_COMPONENT_SWIZZLE_R;
        case 5: return VK_COMPONENT_SWIZZLE_G;
        case 6: return VK_COMPONENT_SWIZZLE_B;
        case 7: return VK_COMPONENT_SWIZZLE_A;
        default: throw std::runtime_error("AGC graphics: guest texture descriptor has an invalid destination channel selector " + std::to_string(dstSel));
    }
}

VkComponentMapping ViewComponents(const GuestTextureResource& resource) {
    if (IsConvertedTextureFormat(resource.format) && !IsRepackedTextureFormat(resource.format)) return {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    const auto channel = [&](std::uint8_t dstSel) { return TextureComponentChannel(resource.format, ComponentSwizzleFor(dstSel)); };
    return {channel(resource.dstSelX), channel(resource.dstSelY), channel(resource.dstSelZ), channel(resource.dstSelW)};
}

// Sampled textures are reused across draws and dispatches while their guest bytes are unchanged; block
// hashes of the guest surface (GuestMemory::HashBlocksOf of the uploaded bytes) validate each reuse,
// so CPU or GPU writes to it force a re-upload.
// The key is everything a lookup matches: the device, the eight descriptor words and the swizzle.
struct TextureKey {
    VkDevice device;
    std::array<std::uint32_t, 8> words;
    std::array<std::uint32_t, 4> components;
    bool depthCompare = false;
    bool operator==(const TextureKey&) const = default;
};

std::uint64_t hashWords(std::uint64_t hash, const std::uint32_t* words, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) hash = (hash ^ words[i]) * 1099511628211ull;
    return hash;
}

struct TextureKeyHash {
    std::size_t operator()(const TextureKey& key) const noexcept {
        auto hash = 14695981039346656037ull ^ reinterpret_cast<std::uintptr_t>(key.device);
        hash = hashWords(hash, key.words.data(), key.words.size()) ^ static_cast<std::uint64_t>(key.depthCompare);
        return static_cast<std::size_t>(hashWords(hash, key.components.data(), key.components.size()));
    }
};

TextureKey MakeTextureKey(VkDevice device, std::span<const std::uint32_t> words, VkComponentMapping components, bool depthCompare = false) {
    TextureKey key{device, {}, {static_cast<std::uint32_t>(components.r), static_cast<std::uint32_t>(components.g), static_cast<std::uint32_t>(components.b), static_cast<std::uint32_t>(components.a)}};
    key.depthCompare = depthCompare;
    std::copy(words.begin(), words.end(), key.words.begin());
    key.words[5] &= ~((0xfffu << 8u) | (1u << 25u));
    key.words[6] &= ~0xffu;
    return key;
}

// The owned proof of an untracked snapshot (APS5_NO_OWNED_PROOFS): the armed words of the surface's
// 64 KiB blocks (GuestMemory::ArmOwnedBlocks), taken before the walk and compare that set its
// blockSince and blockGuard; empty when that walk ran unarmed. While the words, the tracker stamps
// and the guard hold, the snapshot is current without the walk. A proof whose words moved (a CPU
// store into the surface's blocks) backs off arming: 2^n - 1 walks unarmed after the n-th in a row.
struct OwnedProof {
    std::vector<std::uint64_t> words;
    std::uint8_t backoff = 0;
    std::uint16_t skip = 0;
};

struct CachedTexture {
    TextureKey key;
    std::uint64_t address;
    // Size of the snapshot (0 for a view) and, for one of guest bytes (uncompressed keys), its block
    // hashes; a snapshot of clear texels needs none.
    std::uint64_t bytes = 0;
    std::vector<ContentHash::Hash128> hashes;
    std::shared_ptr<Texture> texture;
    // A fast-cleared surface is cached as its clear texels; it stays valid while the keys are unchanged.
    DccKeys keys = DccKeys::Uncompressed;
    // Write generation the snapshot is known current at (see GuestMemory::CollectWrites).
    std::uint64_t generation = 0;
    // A texture viewing a storage image on the GPU has no snapshot; it stays valid while that image
    // is still its surface's cached image and nothing wrote the guest memory since it matched.
    std::shared_ptr<StorageTexture> source;
    std::uint64_t sourceVersion = 0;
    std::uint64_t accounted = 0;
    std::uint64_t present = 0;
    // Recorder::Presents() at the last use (the eviction counters' age, in real presents).
    std::uint64_t presentSeen = 0;
    std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>> committed;
    // The untrackedKey the snapshot was last read or compared equal at, for a surface outside the
    // write watch; zero when none was taken.
    std::array<std::uint64_t, 4> untracked{};
    std::uint64_t blockSince = 0;
    std::array<std::uint64_t, 2> blockGuard{};
    GuestMemory::Commitment blockLayout;
    OwnedProof owned;
};

// Entries in use order (front = most recent) with a hash index by key: a lookup is O(1) and the
// eviction takes the back. APS5_NO_TEXTURE_HASH=1 finds entries by scanning the list (the index is
// still kept), the linear lookup of before.
struct TextureCache {
    AgcDriver::Mutex mutex;
    std::list<CachedTexture> entries;
    std::unordered_map<TextureKey, std::list<CachedTexture>::iterator, TextureKeyHash> index;
    std::unordered_multimap<const StorageTexture*, std::list<CachedTexture>::iterator> views;
    // Entries by their texture object (a Revalidate's touch of the textures its set binds).
    std::unordered_map<const Texture*, std::list<CachedTexture>::iterator> byTexture;
    // Textures of erased entries that resource-cache entries still pinned: the entries go with them
    // once the cache's mutex is released (dropReleasedDependents).
    std::vector<std::shared_ptr<Texture>> released;
    std::uint64_t bytes = 0;
    // The part of `bytes` that snapshot entries account (their own images: SampledTextureMemory's).
    std::uint64_t snapshotBytes = 0;
    std::size_t passed = 0;
    std::chrono::steady_clock::time_point passedSince{};
    std::uint64_t firstUsesPresent = 0;
    std::unordered_set<TextureKey, TextureKeyHash> firstUses;
    // The eviction order's frame (currentFrame): presents, plus one per second without one once
    // presents have stood still for StallBeforeTicks.
    std::uint64_t frame = 0;
    std::uint64_t framePresents = 0;
    std::chrono::steady_clock::time_point frameStart = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point presentAt = std::chrono::steady_clock::now();
    // Snapshots made for one use (refused), by key while some recorded work or build still holds
    // them: a lookup of the key meanwhile takes the same image (admitting it when it may) instead
    // of reading and uploading the surface again. Expired entries are swept as the map grows.
    // Spilled ones (past the share, Texture::Spilled) are also kept alive by the map (`kept`), in
    // least recently used order (`overflowLru`, front = most recent): the game binds whole image
    // tables, so every build of a loading burst binds the same thousands of textures past the share,
    // and lean-x24 made 82k spilled snapshots, a new set for each build, until the overflow type
    // ran out. The kept ones go least recently used first, unused ones only, while the overflow type
    // holds more than its budget (boundOverflow) or the device runs out of memory (the reclaim).
    struct OneOff {
        std::weak_ptr<Texture> texture;
        std::uint64_t bytes = 0;
        std::vector<ContentHash::Hash128> hashes;
        DccKeys keys = DccKeys::Uncompressed;
        std::uint64_t generation = 0;
        std::shared_ptr<Texture> kept;
        std::list<TextureKey>::iterator lru{};
    };
    std::unordered_map<TextureKey, OneOff, TextureKeyHash> oneOffs;
    std::size_t oneOffsSwept = 0;
    std::list<TextureKey> overflowLru;
    std::uint64_t overflowKept = 0;
    // boundOverflow's last trim that dropped nothing, repeated only once the recorder retired a
    // batch or 50 ms passed (as makeRoom's).
    bool overflowTrimFailed = false;
    std::uint64_t overflowTrimRetired = 0;
    std::chrono::steady_clock::time_point overflowTrimAt{};
    // Snapshots made and the distinct surfaces (address and size) they were made of, per report.
    std::unordered_set<std::uint64_t> madeSurfaces;
    // makeRoom's last scan that found nothing to evict, repeated only once the recorder retired a
    // batch or 50 ms passed (a build binding thousands of textures would scan per texture).
    bool roomFailed = false;
    std::uint64_t roomFailedRetired = 0;
    std::chrono::steady_clock::time_point roomFailedAt{};
    struct Counts {
        std::uint64_t inserted = 0;
        std::uint64_t uncached = 0;
        std::uint64_t evicted = 0;
        std::uint64_t passed = 0;
        std::uint64_t stale = 0;
        std::uint64_t evictedBytes = 0;
        std::uint64_t refused = 0;
        std::uint64_t refusedBytes = 0;
        std::uint64_t spilled = 0;
        std::uint64_t spilledBytes = 0;
        std::uint64_t oneOffReuses = 0;
        std::uint64_t made = 0;
        std::uint64_t spilledReuses = 0;
        std::uint64_t overflowDropped = 0;
        std::uint64_t overflowDroppedBytes = 0;
        std::uint64_t overflowWaits = 0;
        // Evictions by path (admitTexture, makeRoom, the out-of-memory reclaim) and how many of
        // them took an entry used at this present or the last.
        std::array<std::uint64_t, 3> evictedBy{};
        std::array<std::uint64_t, 3> evictedRecent{};
    } counts, totals;
    std::chrono::steady_clock::time_point countsReported = std::chrono::steady_clock::now();
    std::uint64_t erasures = 0;
};

// The snapshot bytes of the sampled cache, readable without its mutex (the budgets of both caches
// are computed under either cache's mutex).
std::atomic<std::uint64_t>& SampledCachedBytes() {
    static std::atomic<std::uint64_t> bytes{0};
    return bytes;
}

// Resource-cache sets dropped because a sampled texture they bind left the cache, inserts refused
// because the set binds such a texture, and out-of-memory reclaims (count, bytes).
struct SampledCacheAtomics {
    std::atomic<std::uint64_t> droppedSets{0};
    std::atomic<std::uint64_t> refusedSets{0};
    std::atomic<std::uint64_t> reclaims{0};
    std::atomic<std::uint64_t> reclaimedBytes{0};
};

SampledCacheAtomics& SampledAtomics() {
    static SampledCacheAtomics counters;
    return counters;
}

// The recorder serial a use made now is recorded under, and the newest one retired (see
// Texture::LastUse); 0 without a recorder (tests that build without one).
std::uint64_t useSerial() {
    const auto* recorder = Recorder::Active();
    return recorder != nullptr ? recorder->UseSerial() : 0;
}

std::uint64_t retiredSerial() {
    const auto* recorder = Recorder::Active();
    return recorder != nullptr ? recorder->RetiredSerial() : 0;
}

TextureCache& Textures() {
    static TextureCache cache;
    return cache;
}

thread_local unsigned sampledCacheHeld = 0;
thread_local unsigned storageCacheHeld = 0;
thread_local unsigned sampledCacheReclaimable = 0;
thread_local unsigned storageCacheReclaimable = 0;

struct CacheHold {
    unsigned& depth;
    explicit CacheHold(unsigned& depth) : depth(depth) { ++depth; }
    ~CacheHold() { --depth; }
    CacheHold(const CacheHold&) = delete;
    CacheHold& operator=(const CacheHold&) = delete;
};

// The sampled cache's mutex, taken at most once per thread: an allocation made under it (a new
// snapshot's image in cachedTexture) that runs out of device memory reclaims from the cache on the
// same thread (ReclaimTextureMemory), which then owns it already.
struct SampledCacheLock {
    explicit SampledCacheLock(TextureCache& cache) : lock(cache.mutex, std::defer_lock) {
        if (sampledCacheHeld != 0) return;
        lock.lock();
        ++sampledCacheHeld;
        owner = true;
    }
    ~SampledCacheLock() {
        if (owner) --sampledCacheHeld;
    }
    SampledCacheLock(const SampledCacheLock&) = delete;
    SampledCacheLock& operator=(const SampledCacheLock&) = delete;
    std::unique_lock<AgcDriver::Mutex> lock;
    bool owner = false;
};

struct UncachedImages {
    AgcDriver::Mutex mutex;
    std::vector<std::weak_ptr<StorageTexture>> images;
    std::atomic<bool> any{false};
};

UncachedImages& Uncached() {
    static UncachedImages uncached;
    return uncached;
}

void noteUncached(const std::shared_ptr<StorageTexture>& image) {
    auto& uncached = Uncached();
    std::lock_guard lock(uncached.mutex);
    uncached.images.push_back(image);
    uncached.any.store(true, std::memory_order_release);
}

bool SamplerCacheDisabled() {
    static const bool disabled = std::getenv("APS5_NO_SAMPLER_CACHE") != nullptr;
    return disabled;
}

bool TextureHashEnabled() {
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_HASH") != nullptr;
    return !disabled;
}

std::list<CachedTexture>::iterator findTexture(TextureCache& cache, const TextureKey& key) {
    if (TextureHashEnabled()) {
        const auto found = cache.index.find(key);
        return found == cache.index.end() ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key == key) return it;
    }
    return cache.entries.end();
}

void eraseTexture(TextureCache& cache, std::list<CachedTexture>::iterator it) {
    cache.bytes -= it->accounted;
    if (it->source == nullptr) {
        cache.snapshotBytes -= it->accounted;
        SampledCachedBytes().store(cache.snapshotBytes, std::memory_order_relaxed);
    }
    it->texture->SetSampledCacheState(Texture::CacheState::Released);
    if (it->texture->ResourcePins() != 0) cache.released.push_back(it->texture);
    cache.byTexture.erase(it->texture.get());
    if (it->source != nullptr) {
        for (auto [view, last] = cache.views.equal_range(it->source.get()); view != last; ++view) {
            if (view->second == it) {
                cache.views.erase(view);
                break;
            }
        }
    }
    cache.index.erase(it->key);
    cache.entries.erase(it);
    ++cache.erasures;
}

void dropUncachedViews(TextureCache& cache) {
    auto& uncached = Uncached();
    if (!uncached.any.load(std::memory_order_acquire)) return;
    std::vector<std::weak_ptr<StorageTexture>> images;
    {
        std::lock_guard lock(uncached.mutex);
        images.swap(uncached.images);
        uncached.any.store(false, std::memory_order_release);
    }
    for (const auto& weak : images) {
        const auto image = weak.lock();
        if (image == nullptr) continue;
        for (auto view = cache.views.find(image.get()); view != cache.views.end(); view = cache.views.find(image.get())) eraseTexture(cache, view->second);
    }
}

// The frame the eviction order counts in: one per present and, while the present counter stands
// still (a loading screen draws for minutes without one), one more per second, so a texture last
// used seconds ago is old and makes room rather than turning every new texture into a one-off.
// The seconds count only once presents have stood still for StallBeforeTicks: a frame that takes
// seconds to draw (0.4 fps in the world) would otherwise age the textures it used earlier, and the
// "used this frame or the last" protection would evict and re-snapshot them within the frame.
// APS5_NO_PRESENT_FRAME_AGE=1 counts the seconds from the last present (the old order).
constexpr auto StallBeforeTicks = std::chrono::seconds(10);

std::uint64_t currentFrame(TextureCache& cache) {
    const auto presents = Recorder::Presents();
    const auto now = std::chrono::steady_clock::now();
    if (presents != cache.framePresents) {
        cache.framePresents = presents;
        ++cache.frame;
        cache.frameStart = now;
        cache.presentAt = now;
    } else if (!RuntimeSwitches::Set(RuntimeSwitch::NoPresentFrameAge) && now - cache.presentAt < StallBeforeTicks) {
        return cache.frame;
    } else if (const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - cache.frameStart); elapsed.count() > 0) {
        cache.frame += static_cast<std::uint64_t>(elapsed.count());
        cache.frameStart += elapsed;
    }
    return cache.frame;
}

// Moves an entry to the front (most recently used); the recorded work about to bind it uses it.
void touchTexture(TextureCache& cache, std::list<CachedTexture>::iterator it) {
    cache.entries.splice(cache.entries.begin(), cache.entries, it);
    it->present = currentFrame(cache);
    it->presentSeen = Recorder::Presents();
    it->texture->NoteUse(useSerial());
}

// Whether evicting the entry would free nothing now: a holder other than the cache and the
// resource-cache entries binding it (a build, a caller, a set kept by recorded work), or a recorded
// batch that uses it and has not retired. Pins alone do not hold it: the eviction drops those sets.
bool textureInUse(const CachedTexture& entry, std::uint64_t retired) {
    const auto holders = static_cast<long>(entry.texture.use_count()) - 1 - static_cast<long>(entry.texture->ResourcePins());
    return holders > 0 || entry.texture->LastUse() > retired;
}

enum class EvictionPath : std::size_t { Admission, Share, Reclaim };

void countEvicted(TextureCache& cache, const CachedTexture& entry, EvictionPath path) {
    ++cache.counts.evicted;
    ++cache.totals.evicted;
    cache.counts.evictedBytes += entry.accounted;
    cache.totals.evictedBytes += entry.accounted;
    const auto index = static_cast<std::size_t>(path);
    ++cache.counts.evictedBy[index];
    ++cache.totals.evictedBy[index];
    if (entry.presentSeen + 1u >= Recorder::Presents()) {
        ++cache.counts.evictedRecent[index];
        ++cache.totals.evictedRecent[index];
    }
}

// Evicts unused entries (least recently used first) down to the budget, passing over the ones in
// use (textureInUse) and views that free nothing, then admits the incoming texture unless the cache
// would pass the ceiling: a refused texture serves its one use and goes with it.
bool admitTexture(TextureCache& cache, const TextureKey& key, std::uint64_t incoming, const TextureCacheBudgets& budgets) {
    const auto budget = budgets.sampled;
    if (cache.bytes + incoming <= budget) return true;
    if (const auto now = std::chrono::steady_clock::now(); now - cache.passedSince >= std::chrono::seconds(1)) {
        cache.passed = 0;
        cache.passedSince = now;
    }
    const auto present = currentFrame(cache);
    if (cache.firstUsesPresent != present) {
        cache.firstUses.clear();
        cache.firstUsesPresent = present;
    }
    static const bool laterUses = std::getenv("APS5_SAMPLED_LATER_USE_REFUSAL") != nullptr;
    const bool scan = incoming != 0 && Recorder::Presents() != 0;
    const bool firstUse = scan && !cache.firstUses.contains(key);
    const auto retired = retiredSerial();
    const auto ceiling = budget + budget / 4u;
    while (!cache.entries.empty() && cache.bytes + incoming > budget) {
        const auto back = std::prev(cache.entries.end());
        const bool over = cache.bytes + incoming > ceiling;
        if (laterUses ? scan && !over && (firstUse ? back->present + 1u >= present : back->present == present) : firstUse && back->present + 1u >= present) {
            cache.firstUses.insert(key);
            ++cache.counts.uncached;
            ++cache.totals.uncached;
            return false;
        }
        const bool keep = back->accounted == 0 || textureInUse(*back, retired);
        if (keep && cache.passed < cache.entries.size()) {
            cache.entries.splice(cache.entries.begin(), cache.entries, back);
            ++cache.passed;
            ++cache.counts.passed;
        } else if (keep) {
            break;
        } else {
            countEvicted(cache, *back, EvictionPath::Admission);
            eraseTexture(cache, back);
        }
    }
    if (incoming != 0 && cache.bytes + incoming > budgets.ceiling) {
        ++cache.counts.refused;
        ++cache.totals.refused;
        cache.counts.refusedBytes += incoming;
        cache.totals.refusedBytes += incoming;
        return false;
    }
    return true;
}

// Before a new snapshot is made: while every device-local sampled image (in the cache or not) and
// the new one would pass the share, evicts the entries no holder or recorded batch uses, least
// recently used first. False when that is not enough (what is left is in use): the snapshot then
// goes to system memory rather than past the share where it can (APS5_NO_TEXTURE_SPILL=1 keeps it device-local).
bool makeRoom(TextureCache& cache, std::uint64_t incoming, const TextureCacheBudgets& budgets) {
    const auto all = SampledTextureMemory();
    const auto live = all - std::min(all, SampledTextureOverflowMemory());
    if (budgets.share == 0 || live + incoming <= budgets.share) return true;
    const auto retired = retiredSerial();
    const auto now = std::chrono::steady_clock::now();
    if (cache.roomFailed && cache.roomFailedRetired == retired && now - cache.roomFailedAt < std::chrono::milliseconds(50)) return false;
    std::uint64_t freed = 0;
    const auto fits = [&] { return live - std::min(live, freed) + incoming <= budgets.share; };
    for (auto it = cache.entries.end(); it != cache.entries.begin() && !fits();) {
        const auto current = std::prev(it);
        if (current->source != nullptr || current->accounted == 0 || textureInUse(*current, retired)) {
            it = current;
            continue;
        }
        freed += current->accounted;
        countEvicted(cache, *current, EvictionPath::Share);
        eraseTexture(cache, current);
    }
    cache.roomFailed = !fits();
    cache.roomFailedRetired = retired;
    cache.roomFailedAt = now;
    return !cache.roomFailed;
}

bool SpillEnabled() {
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_SPILL") != nullptr;
    return !disabled;
}

void eraseOneOff(TextureCache& cache, std::unordered_map<TextureKey, TextureCache::OneOff, TextureKeyHash>::iterator it) {
    if (it->second.kept != nullptr) {
        cache.overflowKept -= it->second.kept->AllocationBytes();
        cache.overflowLru.erase(it->second.lru);
    }
    cache.oneOffs.erase(it);
}

void rememberOneOff(TextureCache& cache, const TextureKey& key, const std::shared_ptr<Texture>& texture, std::uint64_t bytes, std::vector<ContentHash::Hash128> hashes, DccKeys keys, std::uint64_t generation) {
    if (cache.oneOffs.size() >= 2 * cache.oneOffsSwept + 1024) {
        std::erase_if(cache.oneOffs, [](const auto& entry) { return entry.second.kept == nullptr && entry.second.texture.expired(); });
        cache.oneOffsSwept = cache.oneOffs.size();
    }
    if (const auto found = cache.oneOffs.find(key); found != cache.oneOffs.end()) eraseOneOff(cache, found);
    auto& oneOff = cache.oneOffs[key];
    oneOff = {texture, bytes, std::move(hashes), keys, generation, nullptr, {}};
    if (texture->Spilled()) {
        oneOff.kept = texture;
        oneOff.lru = cache.overflowLru.insert(cache.overflowLru.begin(), key);
        cache.overflowKept += texture->AllocationBytes();
    }
}

// Drops kept spilled snapshots no build or recorded work uses (no other holder, last used by a
// batch the recorder retired: the open batch's serial is past every retired one), least recently
// used first, until `excess` bytes are dropped; returns how many. The textures go to `doomed`, for
// the caller to hand to the open batch once the cache's mutex is released (keepUntilRetired).
std::size_t trimOverflow(TextureCache& cache, std::uint64_t excess, std::vector<std::shared_ptr<Texture>>& doomed) {
    const auto retired = retiredSerial();
    std::size_t dropped = 0;
    std::uint64_t droppedBytes = 0;
    for (auto it = cache.overflowLru.end(); it != cache.overflowLru.begin() && droppedBytes < excess;) {
        const auto current = std::prev(it);
        const auto found = cache.oneOffs.find(*current);
        if (found == cache.oneOffs.end() || found->second.kept == nullptr) {
            cache.overflowLru.erase(current);
            continue;
        }
        const auto& kept = found->second.kept;
        if (kept.use_count() > 1 || kept->LastUse() > retired) {
            it = current;
            continue;
        }
        ++dropped;
        droppedBytes += kept->AllocationBytes();
        ++cache.counts.overflowDropped;
        ++cache.totals.overflowDropped;
        cache.counts.overflowDroppedBytes += kept->AllocationBytes();
        cache.totals.overflowDroppedBytes += kept->AllocationBytes();
        doomed.push_back(kept);
        eraseOneOff(cache, found);
    }
    return dropped;
}

// Textures (and resource sets) the caches let go of are destroyed with the open batch's retirement
// when one is recording on this thread, not on the spot: nothing GPU-side may still name their
// images or views by then, whatever path bound them.
template<typename T>
void keepUntilRetired(std::vector<std::shared_ptr<T>>& objects) {
    if (objects.empty()) return;
    if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->Recording() && GuestMemory::GpuMutex().HeldByThisThread()) {
        for (auto& object : objects) recorder->Keep(std::move(object));
    }
    objects.clear();
}

// Drops the resource-cache sets that bind textures the sampled cache let go (cache.released), so
// they do not keep the images alive past the cache's budget; a set the next use needs is built
// again. Called with the cache's mutex released. The sets go to the open batch when there is one
// (their destruction then runs off the device lock once it completed), else here.
void dropReleasedDependents() {
    std::vector<std::shared_ptr<Texture>> released;
    {
        auto& cache = Textures();
        SampledCacheLock lock(cache);
        if (cache.released.empty()) return;
        released.swap(cache.released);
    }
    std::unordered_set<const Texture*> set;
    set.reserve(released.size());
    for (const auto& texture : released) set.insert(texture.get());
    std::vector<std::shared_ptr<ShaderResources>> resources;
    SharedResourceCache().RemoveTextures(set, &resources);
    if (!resources.empty()) {
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->Recording() && GuestMemory::GpuMutex().HeldByThisThread()) {
            for (auto& resource : resources) recorder->Keep(std::move(resource));
        }
    }
}

// The RAII form, declared before the cache's lock so it runs after the unlock.
struct DropReleasedDependents {
    DropReleasedDependents() = default;
    DropReleasedDependents(const DropReleasedDependents&) = delete;
    DropReleasedDependents& operator=(const DropReleasedDependents&) = delete;
    ~DropReleasedDependents() {
        try {
            dropReleasedDependents();
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[texture-cache] dropping the resource sets of released textures failed: %s\n", error.what());
        }
    }
};

void reportCacheCounts(TextureCache& cache, const TextureCacheBudgets& budgets) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - cache.countsReported < std::chrono::seconds(10)) return;
    cache.countsReported = now;
    const auto counts = std::exchange(cache.counts, {});
    static std::uint64_t droppedBefore = 0, refusedSetsBefore = 0, reclaimsBefore = 0, reclaimedBefore = 0;
    auto& atomics = SampledAtomics();
    const auto dropped = atomics.droppedSets.load(std::memory_order_relaxed);
    const auto refusedSets = atomics.refusedSets.load(std::memory_order_relaxed);
    const auto reclaims = atomics.reclaims.load(std::memory_order_relaxed);
    const auto reclaimed = atomics.reclaimedBytes.load(std::memory_order_relaxed);
    const auto sampled = SampledTextureMemory();
    const auto outside = sampled - std::min(sampled, cache.snapshotBytes);
    AgcDriver::ProfilePrint_nid_no_patch("[texture-cache] (10 s): %llu textures cached, %llu used uncached (first use this frame, the least recently used entry used this frame or the last), %llu refused over the ceiling (%llu MiB), %llu evicted (%llu MiB), %llu in-use ones passed over, %llu found changed; %llu resource sets dropped with released textures, %llu not cached for binding one; %llu out-of-memory reclaims (%llu MiB); %llu snapshots placed in system memory past the share (%llu MiB), %llu one-off snapshots reused; %llu MiB in %zu entries, budget %llu MiB, ceiling %llu MiB; sampled images outside the cache %llu MiB (in recorded work and builds); storage budget %llu MiB, VRAM share of the texture caches %llu MiB; frame %llu\n", static_cast<unsigned long long>(counts.inserted), static_cast<unsigned long long>(counts.uncached), static_cast<unsigned long long>(counts.refused), static_cast<unsigned long long>(counts.refusedBytes >> 20u), static_cast<unsigned long long>(counts.evicted), static_cast<unsigned long long>(counts.evictedBytes >> 20u), static_cast<unsigned long long>(counts.passed), static_cast<unsigned long long>(counts.stale), static_cast<unsigned long long>(dropped - droppedBefore), static_cast<unsigned long long>(refusedSets - refusedSetsBefore), static_cast<unsigned long long>(reclaims - reclaimsBefore), static_cast<unsigned long long>((reclaimed - reclaimedBefore) >> 20u), static_cast<unsigned long long>(counts.spilled), static_cast<unsigned long long>(counts.spilledBytes >> 20u), static_cast<unsigned long long>(counts.oneOffReuses), static_cast<unsigned long long>(cache.bytes >> 20u), cache.entries.size(), static_cast<unsigned long long>(budgets.sampled >> 20u), static_cast<unsigned long long>(budgets.ceiling >> 20u), static_cast<unsigned long long>(outside >> 20u), static_cast<unsigned long long>(budgets.storage >> 20u), static_cast<unsigned long long>(budgets.target >> 20u), static_cast<unsigned long long>(Recorder::Presents()));
    AgcDriver::ProfilePrint_nid_no_patch("[texture-cache] (10 s): %llu snapshots made of %zu distinct surfaces; %llu spilled snapshots reused, %llu kept ones dropped (%llu MiB), %llu batches waited for; overflow type %llu MiB of a %llu MiB budget, %llu MiB of it kept for reuse in %zu snapshots\n", static_cast<unsigned long long>(counts.made), cache.madeSurfaces.size(), static_cast<unsigned long long>(counts.spilledReuses), static_cast<unsigned long long>(counts.overflowDropped), static_cast<unsigned long long>(counts.overflowDroppedBytes >> 20u), static_cast<unsigned long long>(counts.overflowWaits), static_cast<unsigned long long>(SampledTextureOverflowMemory() >> 20u), static_cast<unsigned long long>(budgets.overflow >> 20u), static_cast<unsigned long long>(cache.overflowKept >> 20u), cache.overflowLru.size());
    AgcDriver::ProfilePrint_nid_no_patch("[texture-cache] (10 s) evictions by path: admission %llu (%llu used this present or the last), share %llu (%llu), reclaim %llu (%llu); eviction frame %llu at present %llu\n", static_cast<unsigned long long>(counts.evictedBy[0]), static_cast<unsigned long long>(counts.evictedRecent[0]), static_cast<unsigned long long>(counts.evictedBy[1]), static_cast<unsigned long long>(counts.evictedRecent[1]), static_cast<unsigned long long>(counts.evictedBy[2]), static_cast<unsigned long long>(counts.evictedRecent[2]), static_cast<unsigned long long>(cache.frame), static_cast<unsigned long long>(Recorder::Presents()));
    cache.madeSurfaces.clear();
    droppedBefore = dropped;
    refusedSetsBefore = refusedSets;
    reclaimsBefore = reclaims;
    reclaimedBefore = reclaimed;
}

struct ImageMemoEntry {
    VkDevice device = VK_NULL_HANDLE;
    std::array<std::uint32_t, 8> words{};
    GuestTextureResource resource{};
    std::uint64_t guestBytes = 0;
    VkComponentMapping components{};
    bool hasEntry = false;
    std::list<CachedTexture>::iterator entry{};
    std::uint64_t erasures = 0;
};

constexpr std::size_t ImageMemoSlots = 1024;
struct ImageMemoTag {};

ImageMemoEntry& imageMemoSlot(std::span<const std::uint32_t> words) {
    auto& imageMemo = HostThreadLocal<std::unique_ptr<std::array<ImageMemoEntry, ImageMemoSlots>>, ImageMemoTag>();
    if (imageMemo == nullptr) imageMemo = std::make_unique<std::array<ImageMemoEntry, ImageMemoSlots>>();
    const auto hash = hashWords(14695981039346656037ull, words.data(), words.size());
    return (*imageMemo)[(hash ^ (hash >> 29u)) % ImageMemoSlots];
}

const ImageMemoEntry& decodedImage(VkDevice device, std::span<const std::uint32_t> words) {
    auto& slot = imageMemoSlot(words);
    if (slot.device == device && std::equal(words.begin(), words.end(), slot.words.begin())) return slot;
    slot.device = VK_NULL_HANDLE;
    slot.resource = DecodeTextureResource(words);
    slot.guestBytes = DescribeSurface(slot.resource).guestBytes;
    slot.components = ViewComponents(slot.resource);
    std::copy(words.begin(), words.end(), slot.words.begin());
    slot.hasEntry = false;
    slot.device = device;
    return slot;
}

std::list<CachedTexture>::iterator findSampledEntry(TextureCache& cache, VkDevice device, const std::array<std::uint32_t, 8>& words, VkComponentMapping components) {
    auto& slot = imageMemoSlot(words);
    const bool same = slot.device == device && slot.words == words && slot.components.r == components.r && slot.components.g == components.g && slot.components.b == components.b && slot.components.a == components.a;
    if (same && slot.hasEntry && slot.erasures == cache.erasures) return slot.entry;
    const auto it = findTexture(cache, MakeTextureKey(device, words, components));
    if (same) {
        slot.hasEntry = it != cache.entries.end();
        slot.entry = it;
        slot.erasures = cache.erasures;
    }
    return it;
}

// APS5_PROFILE_DRAW: what the sampled-texture and storage-image lookups did, printed as [textures]
// every 10 s (cumulative). Storage-sourced sampled textures need no CPU read; snapshots do, and a
// pending read is a snapshot compare or read made while recorded work still wrote the surface (the
// flush hook may find that work already signaled, so this bounds the hook syncs from above; the
// [hooksync] line attributes the actual waits).
struct TextureCounters {
    std::atomic<std::uint64_t> fromStorage{0};
    std::atomic<std::uint64_t> snapshots{0};
    std::atomic<std::uint64_t> pendingReads{0};
    std::atomic<std::uint64_t> storageFallbacks{0};
    std::atomic<std::uint64_t> records{0};
    std::atomic<std::uint64_t> fastHits{0};
    std::atomic<std::uint64_t> fastMisses{0};
    std::atomic<std::uint64_t> storageHits{0};
    std::atomic<std::uint64_t> storageCreated{0};
    // Storage images a Revalidate refreshed directly instead of through the lookups (T1, see
    // ShaderResources::refreshOwnObjects).
    std::atomic<std::uint64_t> ownRefreshes{0};
    std::atomic<std::uint64_t> tablesReused{0};
    std::atomic<std::uint64_t> tablesReplayed{0};
    std::atomic<std::uint64_t> tablesResolved{0};
    std::atomic<std::uint64_t> tableWritesKept{0};
    std::atomic<std::uint64_t> tableWritesMade{0};
    // Snapshot hits over memory outside the write watch: kept by an unchanged UntrackedHint key, or
    // by a byte compare.
    std::atomic<std::uint64_t> untrackedKept{0};
    std::atomic<std::uint64_t> untrackedCompared{0};
    std::atomic<std::int64_t> lastReport{0};
};

TextureCounters& TextureCounts() {
    static TextureCounters counters;
    return counters;
}

enum class SnapshotCompare : std::size_t { Stamped, Layout, CommittedStamped, Pending, Untracked, HintMoved, OneOff, Count };
constexpr std::array<const char*, static_cast<std::size_t>(SnapshotCompare::Count)> SnapshotCompareNames{"stamped", "committed runs moved", "committed stamped", "pending write", "untracked without hint", "hint moved", "one-off"};

struct SnapshotCompareCounters {
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(SnapshotCompare::Count)> compares{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(SnapshotCompare::Count)> differed{};
    std::atomic<std::uint64_t> narrowed{0};
    std::atomic<std::uint64_t> hashedBlocks{0};
    std::atomic<std::uint64_t> skippedBlocks{0};
    std::atomic<std::uint64_t> hashedBytes{0};
    std::atomic<std::uint64_t> skippedBytes{0};
    std::array<std::atomic<std::uint64_t>, 8> untrackedKinds{};
    std::atomic<std::int64_t> lastReport{0};
};

SnapshotCompareCounters& SnapshotCompares() {
    static SnapshotCompareCounters counters;
    return counters;
}

bool CommittedProofEnabled() {
    static const bool disabled = std::getenv("APS5_NO_COMMITTED_PROOF") != nullptr;
    return !disabled;
}

bool BlockRevalidateEnabled() {
    static const bool disabled = std::getenv("APS5_NO_BLOCK_REVALIDATE") != nullptr;
    return !disabled;
}

bool UntrackedBlocksEnabled() {
    static const bool disabled = std::getenv("APS5_NO_UNTRACKED_BLOCKS") != nullptr;
    return !disabled && BlockRevalidateEnabled();
}

bool SnapshotComparesProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

void noteSnapshotCompare(SnapshotCompare reason, bool equal, bool narrowed, const GuestMemory::BlockMatch& match, const GuestMemory::UntrackedBlocks* kinds) {
    auto& counters = SnapshotCompares();
    const auto slot = static_cast<std::size_t>(reason);
    counters.compares[slot].fetch_add(1, std::memory_order_relaxed);
    if (!equal) counters.differed[slot].fetch_add(1, std::memory_order_relaxed);
    if (narrowed) counters.narrowed.fetch_add(1, std::memory_order_relaxed);
    counters.hashedBlocks.fetch_add(match.hashed, std::memory_order_relaxed);
    counters.skippedBlocks.fetch_add(match.skipped, std::memory_order_relaxed);
    counters.hashedBytes.fetch_add(match.hashedBytes, std::memory_order_relaxed);
    counters.skippedBytes.fetch_add(match.skippedBytes, std::memory_order_relaxed);
    if (kinds != nullptr) {
        const std::array<std::size_t, 8> values{kinds->covered, kinds->hinted, kinds->outside, kinds->hostRegion, kinds->unhinted, kinds->refused, kinds->partial, kinds->layoutChanged};
        for (std::size_t i = 0; i < values.size(); ++i) counters.untrackedKinds[i].fetch_add(values[i], std::memory_order_relaxed);
    }
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load();
    if (nowMs - last < 10000 || !counters.lastReport.compare_exchange_strong(last, nowMs)) return;
    std::string reasons;
    for (std::size_t i = 0; i < SnapshotCompareNames.size(); ++i) {
        const auto count = counters.compares[i].exchange(0, std::memory_order_relaxed);
        const auto changed = counters.differed[i].exchange(0, std::memory_order_relaxed);
        if (count == 0) continue;
        reasons += " " + std::string(SnapshotCompareNames[i]) + " " + std::to_string(count) + " (" + std::to_string(changed) + " changed)";
    }
    const auto take = [&](std::size_t i) { return static_cast<unsigned long long>(counters.untrackedKinds[i].exchange(0, std::memory_order_relaxed)); };
    const auto covered = take(0);
    const auto hinted = take(1);
    const auto outside = take(2);
    const auto hostRegion = take(3);
    const auto unhinted = take(4);
    const auto refused = take(5);
    const auto partial = take(6);
    const auto layoutChanged = take(7);
    AgcDriver::ProfilePrint_nid_no_patch("[texhash] snapshot compares (10 s) by reason:%s; %llu narrowed to written blocks; blocks hashed %llu (%.1f MiB), skipped %llu (%.1f MiB); untracked ranges' blocks: write-watched %llu, hinted %llu, outside the arena %llu, host region %llu, unwatched without hint %llu, walk refused %llu; partly readable %llu, layout changed %llu\n", reasons.c_str(), static_cast<unsigned long long>(counters.narrowed.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.hashedBlocks.exchange(0, std::memory_order_relaxed)), static_cast<double>(counters.hashedBytes.exchange(0, std::memory_order_relaxed)) / 1048576.0, static_cast<unsigned long long>(counters.skippedBlocks.exchange(0, std::memory_order_relaxed)), static_cast<double>(counters.skippedBytes.exchange(0, std::memory_order_relaxed)) / 1048576.0, covered, hinted, outside, hostRegion, unhinted, refused, partial, layoutChanged);
}

const std::vector<std::uint8_t>& pendingBlocks(std::uint64_t address, std::size_t bytes, std::size_t count) {
    thread_local std::vector<std::uint8_t>* pendingSlot = nullptr;
    auto& pending = ShaderRecompiler::ThreadOwned(pendingSlot);
    pending.clear();
    const auto writes = Recorder::PendingWriteSnapshot();
    const bool overlapping = Recorder::SnapshotOverlaps(writes.get(), address, bytes);
    if (overlapping) TextureCounts().pendingReads.fetch_add(1, std::memory_order_relaxed);
    if (!overlapping || count == 0) return pending;
    pending.assign(count, 0);
    std::size_t index = 0;
    const auto end = address + bytes;
    for (auto at = address; at < end && index < count; ++index) {
        const auto stop = std::min<std::uint64_t>(end, (at / GuestMemory::HashBlockBytes + 1) * GuestMemory::HashBlockBytes);
        pending[index] = Recorder::SnapshotOverlaps(writes.get(), at, static_cast<std::size_t>(stop - at)) ? 1 : 0;
        at = stop;
    }
    return pending;
}

bool snapshotStillMatches(std::uint64_t address, std::size_t bytes, std::span<const ContentHash::Hash128> hashes, std::uint64_t since, const std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>>& before, SnapshotCompare reason) {
    const bool narrow = since != 0 && BlockRevalidateEnabled() && hashes.size() == GuestMemory::HashBlockCount(address, bytes);
    const auto& pending = pendingBlocks(address, bytes, narrow ? hashes.size() : 0);
    const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::TextureCompare);
    GuestMemory::FlushGpuWrites(address, bytes);
    GuestMemory::BlockMatch match;
    const bool equal = narrow ? GuestMemory::CommittedBlocksMatchSince(address, bytes, hashes, since, before.has_value() ? &*before : nullptr, pending, &match) : GuestMemory::CommittedBlocksMatch(address, bytes, hashes);
    if (!SnapshotComparesProfiled()) return equal;
    if (!narrow) match = {hashes.size(), 0, bytes, 0};
    noteSnapshotCompare(reason, equal, narrow, match, nullptr);
    return equal;
}

std::array<std::uint64_t, 2> untrackedGuard(std::uint64_t address, std::size_t bytes) {
    const auto* recorder = Recorder::Active();
    return {GuestMemory::ForgetSerial(), recorder != nullptr ? recorder->NewestWriteNote(address, bytes) : 0};
}

std::uint64_t untrackedBaseline(std::uint64_t address, std::size_t bytes, GuestMemory::Commitment& layout) {
    layout = {};
    if (!UntrackedBlocksEnabled()) return 0;
    thread_local std::vector<std::uint8_t>* staleSlot = nullptr;
    auto& stale = ShaderRecompiler::ThreadOwned(staleSlot);
    stale.assign(GuestMemory::HashBlockCount(address, bytes), 1);
    return GuestMemory::WalkUntrackedBlocks(address, bytes, 0, stale, nullptr, nullptr, &layout);
}

// The walk proof of an untracked snapshot (as before owned proofs): walk the blocks' CPU writes
// since `since`, compare the blocks the walk could not prove unchanged.
bool untrackedWalkMatches(std::uint64_t address, std::size_t bytes, std::span<const ContentHash::Hash128> hashes, std::uint64_t& since, std::array<std::uint64_t, 2>& guard, GuestMemory::Commitment& layout, SnapshotCompare reason) {
    const auto count = hashes.size();
    if (!UntrackedBlocksEnabled() || count != GuestMemory::HashBlockCount(address, bytes)) {
        since = 0;
        return snapshotStillMatches(address, bytes, hashes, 0, std::nullopt, reason);
    }
    const auto now = untrackedGuard(address, bytes);
    const bool trusted = since != 0 && (now[0] & 1u) == 0 && now == guard;
    thread_local std::vector<std::uint8_t>* staleSlot = nullptr;
    auto& stale = ShaderRecompiler::ThreadOwned(staleSlot);
    stale.assign(count, 1);
    GuestMemory::UntrackedBlocks kinds;
    thread_local GuestMemory::Commitment* layoutSlot = nullptr;
    auto& current = ShaderRecompiler::ThreadOwned(layoutSlot);
    const bool layoutWalk = GuestMemory::UntrackedLayoutWalks();
    const auto walked = GuestMemory::WalkUntrackedBlocks(address, bytes, trusted ? since : 0, stale, &kinds, layoutWalk ? &layout : nullptr, layoutWalk ? &current : nullptr);
    const auto& pending = pendingBlocks(address, bytes, count);
    const bool pendingAny = std::any_of(pending.begin(), pending.end(), [](std::uint8_t flag) { return flag != 0; });
    for (std::size_t i = 0; i < pending.size() && i < stale.size(); ++i) stale[i] = static_cast<std::uint8_t>(stale[i] | pending[i]);
    const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::TextureCompare);
    GuestMemory::FlushGpuWrites(address, bytes);
    GuestMemory::BlockMatch match;
    const bool equal = GuestMemory::CommittedBlocksMatchMarked(address, bytes, hashes, stale, &match, layoutWalk ? &current : nullptr);
    static const bool verifyBlocks = std::getenv("APS5_VERIFY_UNTRACKED_BLOCKS") != nullptr;
    if (verifyBlocks && equal) {
        static std::atomic<std::uint64_t> checked{0}, differed{0}, compares{0};
        thread_local std::vector<std::uint8_t>* cleanSlot = nullptr;
        auto& clean = ShaderRecompiler::ThreadOwned(cleanSlot);
        clean.assign(count, 0);
        std::size_t cleanBlocks = 0;
        for (std::size_t i = 0; i < count && i < stale.size(); ++i) {
            if (stale[i] != 0) continue;
            clean[i] = 1;
            ++cleanBlocks;
        }
        if (cleanBlocks != 0) {
            GuestMemory::BlockMatch verify;
            const bool still = GuestMemory::CommittedBlocksMatchMarked(address, bytes, hashes, clean, &verify);
            checked.fetch_add(cleanBlocks, std::memory_order_relaxed);
            if (!still) differed.fetch_add(1, std::memory_order_relaxed);
            if (compares.fetch_add(1, std::memory_order_relaxed) % 20000 == 19999 || !still) std::fprintf(stderr, "[texhash] verify: %llu compares, %llu blocks proved clean re-hashed, %llu compares with a clean block that differs%c", static_cast<unsigned long long>(compares.load()), static_cast<unsigned long long>(checked.load()), static_cast<unsigned long long>(differed.load()), 10);
        }
    }
    if (equal && walked != 0 && (now[0] & 1u) == 0 && !pendingAny) {
        since = walked;
        guard = now;
        if (layoutWalk) std::swap(layout, current);
    } else {
        since = 0;
    }
    if (SnapshotComparesProfiled()) noteSnapshotCompare(reason, equal, trusted, match, &kinds);
    return equal;
}

// APS5_PROFILE_DRAW: the owned proofs of untracked snapshots ([owned-proofs], 10 s deltas).
struct OwnedProofCounters {
    std::atomic<std::uint64_t> held{0};
    std::atomic<std::uint64_t> wordsMoved{0};
    std::atomic<std::uint64_t> stampsMoved{0};
    std::atomic<std::uint64_t> walks{0};
    std::atomic<std::uint64_t> armedWalks{0};
    std::atomic<std::uint64_t> notArmable{0};
    std::atomic<std::uint64_t> backingOff{0};
    std::atomic<std::uint64_t> switchedOff{0};
    std::atomic<std::uint64_t> verified{0};
    std::atomic<std::uint64_t> disagreed{0};
    std::atomic<std::uint64_t> windowVerified{0};
    std::atomic<std::int64_t> windowStart{0};
    std::atomic<std::int64_t> lastReport{0};
};

OwnedProofCounters& OwnedProofCounts() {
    static OwnedProofCounters counters;
    return counters;
}

std::int64_t steadyMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void reportOwnedProofs() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& counters = OwnedProofCounts();
    const auto nowMs = steadyMs();
    auto last = counters.lastReport.load();
    if (nowMs - last < 10000 || !counters.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    const auto blocks = GuestMemory::TakeOwnedBlockCounts();
    const auto held = take(counters.held);
    const auto wordsMoved = take(counters.wordsMoved);
    const auto stampsMoved = take(counters.stampsMoved);
    const auto walks = take(counters.walks);
    const auto armedWalks = take(counters.armedWalks);
    const auto notArmable = take(counters.notArmable);
    const auto backingOff = take(counters.backingOff);
    const auto switchedOff = take(counters.switchedOff);
    const auto verified = take(counters.verified);
    const auto disagreed = take(counters.disagreed);
    const auto by = [&](std::size_t reason) { return static_cast<unsigned long long>(blocks.refusedBy[reason]); };
    AgcDriver::ProfilePrint_nid_no_patch("[owned-proofs] (10 s): armed proofs %llu held, %llu found the words moved, %llu found stamps or guard moved; fallback proofs (walks) %llu: %llu armed first, %llu not armable, %llu backing off, %llu switched off; arms %llu (blocks newly armed %llu: %llu with private pages, %llu with nothing mapped, %llu partly mapped), refused %llu (outside the arena or host region %llu, shared or aliased view %llu, pinned or importing %llu, host write open %llu, protection change in flight %llu, not writable or other mapping %llu, private not allowed %llu, nothing mapped %llu, not write-watched %llu); write-watch queries %llu (%llu found a written page, %llu proofs asked again under the lock), disarms by a collect %llu, pages taken at arming %llu, other disarms %llu; guest write faults in owned blocks %llu (owned blocks change no protection: none is theirs); verified %llu, disagreed %llu\n", held, wordsMoved, stampsMoved, walks, armedWalks, notArmable, backingOff, switchedOff, static_cast<unsigned long long>(blocks.arms), static_cast<unsigned long long>(blocks.armedBlocks), static_cast<unsigned long long>(blocks.privateBlocks), static_cast<unsigned long long>(blocks.unmappedBlocks), static_cast<unsigned long long>(blocks.partialBlocks), static_cast<unsigned long long>(blocks.refused), by(0), by(1), by(2), by(3), by(4), by(5), by(6), by(7), by(8), static_cast<unsigned long long>(blocks.queries), static_cast<unsigned long long>(blocks.dirty), static_cast<unsigned long long>(blocks.locked), static_cast<unsigned long long>(blocks.collected), static_cast<unsigned long long>(blocks.taken), static_cast<unsigned long long>(blocks.disowned), static_cast<unsigned long long>(blocks.faults), verified, disagreed);
}

// APS5_VERIFY_OWNED_PROOFS=<n>: the first n held armed proofs of every 10 s also run a full walk and
// compare (from generation 0, so every block is hashed) and print any disagreement.
bool takeOwnedVerify() {
    static const std::uint64_t budget = [] {
        const char* value = std::getenv("APS5_VERIFY_OWNED_PROOFS");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
    }();
    if (budget == 0) return false;
    auto& counters = OwnedProofCounts();
    const auto nowMs = steadyMs();
    auto start = counters.windowStart.load();
    if (nowMs - start >= 10000 && counters.windowStart.compare_exchange_strong(start, nowMs)) counters.windowVerified.store(0, std::memory_order_relaxed);
    return counters.windowVerified.fetch_add(1, std::memory_order_relaxed) < budget;
}

// An untracked snapshot's proof. With owned proofs (the default; APS5_NO_OWNED_PROOFS selects the
// walk alone) a surface whose blocks were armed before the walk that proved it current is proved
// again by the armed words, the tracker stamps and the guard: no walk, no compare. Anything else
// (not armable, armed words that moved, stamps or guard that moved) walks as before; the walk is
// armed first when it can be, so the next proof can take the armed path.
//
// Exactness: arming resets the blocks' write-watch bits before the walk starts, keeping the pages
// it takes reported to the walk's collect, so a CPU store before it is seen by the walk (dirty
// bits) or the compare (the bytes), exactly as without arming. A CPU store after it leaves a write
// bit that OwnedWordsHold finds (GetWriteWatch, no reset, no fault), or that a later collect takes,
// which moves the block's word; a mapping, protection or host write change moves the word before it
// happens. So unchanged words and no written page mean no such change since the walk started. The driver's and the GPU's writes reach the range through the tracker (MarkWritten
// stamps, untracked-host stores), recorded but unstamped ones through the guard's write note and
// the pending write snapshot, the same evidence the walk itself uses for the blocks it skips. The
// flush hook runs first, as before the walk's compare.
bool untrackedStillMatches(std::uint64_t address, std::size_t bytes, std::span<const ContentHash::Hash128> hashes, std::uint64_t& since, std::array<std::uint64_t, 2>& guard, GuestMemory::Commitment& layout, SnapshotCompare reason, OwnedProof& owned) {
    auto& counters = OwnedProofCounts();
    const auto count = hashes.size();
    if (RuntimeSwitches::Set(RuntimeSwitch::NoOwnedProofs) || !UntrackedBlocksEnabled() || count != GuestMemory::HashBlockCount(address, bytes)) {
        owned.words.clear();
        counters.walks.fetch_add(1, std::memory_order_relaxed);
        counters.switchedOff.fetch_add(1, std::memory_order_relaxed);
        reportOwnedProofs();
        return untrackedWalkMatches(address, bytes, hashes, since, guard, layout, reason);
    }
    if (!owned.words.empty() && since != 0) {
        const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::TextureCompare);
        GuestMemory::FlushGpuWrites(address, bytes);
        const auto now = untrackedGuard(address, bytes);
        const bool wordsHold = GuestMemory::OwnedWordsHold(address, bytes, owned.words);
        const bool holds = wordsHold && (now[0] & 1u) == 0 && now == guard && !Recorder::SnapshotWriteOverlaps(address, bytes) && GuestMemory::OwnedProofHolds(address, bytes, owned.words, since, true);
        if (holds) {
            owned.backoff = 0;
            bool equal = true;
            if (takeOwnedVerify()) {
                std::uint64_t fullSince = 0;
                auto fullGuard = guard;
                auto fullLayout = layout;
                equal = untrackedWalkMatches(address, bytes, hashes, fullSince, fullGuard, fullLayout, reason);
                counters.verified.fetch_add(1, std::memory_order_relaxed);
                if (!equal) {
                    counters.disagreed.fetch_add(1, std::memory_order_relaxed);
                    std::fprintf(stderr, "[owned-proofs] verify: 0x%llx+0x%zx (%zu blocks, since %llu): the armed proof held, the full walk and compare found the surface changed%c", static_cast<unsigned long long>(address), bytes, count, static_cast<unsigned long long>(since), 10);
                    owned.words.clear();
                    since = 0;
                }
            }
            if (equal) {
                counters.held.fetch_add(1, std::memory_order_relaxed);
                if (SnapshotComparesProfiled()) noteSnapshotCompare(reason, true, true, GuestMemory::BlockMatch{0, count, 0, bytes}, nullptr);
            }
            reportOwnedProofs();
            return equal;
        }
        if (!wordsHold) {
            counters.wordsMoved.fetch_add(1, std::memory_order_relaxed);
            owned.backoff = static_cast<std::uint8_t>(std::min<unsigned>(owned.backoff + 1u, 6u));
            owned.skip = static_cast<std::uint16_t>((1u << owned.backoff) - 1u);
        } else {
            counters.stampsMoved.fetch_add(1, std::memory_order_relaxed);
        }
        owned.words.clear();
    }
    counters.walks.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::uint64_t> words;
    bool armed = false;
    if (owned.skip != 0) {
        --owned.skip;
        counters.backingOff.fetch_add(1, std::memory_order_relaxed);
    } else {
        words.resize(count);
        armed = GuestMemory::ArmOwnedBlocks(address, bytes, words, !RuntimeSwitches::Set(RuntimeSwitch::NoOwnedPrivate));
        (armed ? counters.armedWalks : counters.notArmable).fetch_add(1, std::memory_order_relaxed);
    }
    const bool equal = untrackedWalkMatches(address, bytes, hashes, since, guard, layout, reason);
    if (armed && equal && since != 0) owned.words = std::move(words);
    else owned.words.clear();
    reportOwnedProofs();
    return equal;
}

thread_local std::uint64_t sampledLookupKeyScans = 0;

void reportTextureCounters() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& counters = TextureCounts();
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load();
    if (nowMs - last < 10000 || !counters.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto count = [](const std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.load(std::memory_order_relaxed)); };
    AgcDriver::ProfilePrint_nid_no_patch("[textures] sampled created: %llu from storage images, %llu snapshots (%llu snapshot reads over recorded writes inside cachedTexture, %llu storage-path fallbacks); stage-A records %llu: %llu fast hits, %llu full lookups; storage images %llu hits, %llu created, %llu own-object refreshes; image tables %llu reused, %llu resolved; untracked snapshots %llu kept by the write-watch hint, %llu compared\n", count(counters.fromStorage), count(counters.snapshots), count(counters.pendingReads), count(counters.storageFallbacks), count(counters.records), count(counters.fastHits), count(counters.fastMisses), count(counters.storageHits), count(counters.storageCreated), count(counters.ownRefreshes), count(counters.tablesReused), count(counters.tablesResolved), count(counters.untrackedKept), count(counters.untrackedCompared));
}

// APS5_PROFILE_DRAW: cachedTexture's slice copies (Texture::CopyPendingSlices), 10 s deltas: lookups
// that took the pending slices by device copies, the slices copied, and lookups whose texture could
// not take them (stored and re-read instead).
void reportSliceCopies(bool taken) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static std::atomic<std::uint64_t> refused{0};
    static std::atomic<long long> lastReport{0};
    if (!taken) refused.fetch_add(1, std::memory_order_relaxed);
    if (!profile) return;
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = lastReport.load();
    if (nowMs - last < 10000 || !lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto counts = Texture::TakeSliceCopyCounts();
    std::fprintf(stderr, "[slice-copy] (10 s): %llu lookups took pending slices by copies, %llu slices copied, %llu lookups stored them instead\n", static_cast<unsigned long long>(counts.lookups), static_cast<unsigned long long>(counts.copies), static_cast<unsigned long long>(refused.exchange(0, std::memory_order_relaxed)));
}

// What the sampled-texture lookups on this thread proved their returned objects current against,
// taken by object into the ShaderResources being built or revalidated (captureValidation), which
// then repeats the proof from write stamps on its next Revalidate. The lookups run inside one build
// on one thread; records nobody takes (a build that threw, a resident target looked up outside a
// build, a thread that never captures) are dropped by the next capture or by the bound below.
struct LookupRecord {
    const void* object;
    GuestTextureResource resource;
    std::uint64_t bytes;
    DccKeys keys;
    std::uint64_t generation;
    const StorageTexture* source;
    bool depth = false;
    std::uint64_t committedGeneration = 0;
    std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>> committed;
};

thread_local std::vector<LookupRecord>* lookupLogSlot = nullptr;
constexpr std::size_t LookupLogBound = 1024;
constexpr std::size_t NoBuildLookups = std::numeric_limits<std::size_t>::max();
thread_local std::size_t buildLookupsStart = NoBuildLookups;
thread_local bool buildLookupsPastBound = false;
std::atomic<std::uint64_t> lookupBuildsPastBound{0};

std::vector<LookupRecord>& lookupLog() {
    return ShaderRecompiler::ThreadOwned(lookupLogSlot);
}

void logLookup(const LookupRecord& record) {
    auto& log = lookupLog();
    if (log.size() >= LookupLogBound) {
        const auto dropped = std::min(LookupLogBound / 2, buildLookupsStart);
        if (dropped < LookupLogBound / 2) buildLookupsPastBound = true;
        log.erase(log.begin(), log.begin() + static_cast<std::ptrdiff_t>(dropped));
        if (buildLookupsStart != NoBuildLookups) buildLookupsStart -= dropped;
    }
    log.push_back(record);
}

class BuildLookups {
public:
    BuildLookups() : outermost(buildLookupsStart == NoBuildLookups) { buildLookupsStart = std::min(buildLookupsStart, lookupLog().size()); }
    ~BuildLookups() {
        if (!outermost) return;
        buildLookupsStart = NoBuildLookups;
        if (buildLookupsPastBound) lookupBuildsPastBound.fetch_add(1, std::memory_order_relaxed);
        buildLookupsPastBound = false;
        auto& log = lookupLog();
        if (log.size() > LookupLogBound) log.erase(log.begin(), log.end() - static_cast<std::ptrdiff_t>(LookupLogBound));
        if (log.capacity() > LookupLogBound + LookupLogBound / 2) log.shrink_to_fit();
    }
    BuildLookups(const BuildLookups&) = delete;
    BuildLookups& operator=(const BuildLookups&) = delete;

private:
    bool outermost;
};

void refreshForView(StorageTexture& source, const GuestTextureResource& resource) {
    if (resource.dimension == TextureDimension::k2D && source.ImageLayers() > 1) source.RefreshLayers(resource.baseArray, 1);
    else source.Refresh();
}

std::shared_ptr<StorageTexture> cachedStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint32_t mip, std::uint64_t guestBytes = 0, bool written = false, std::optional<std::pair<std::uint32_t, std::uint32_t>> layers = std::nullopt);

std::uint64_t heldBytes(const StorageTexture& image) {
    return image.BudgetBytes();
}

bool MetadataMoved(const StorageTexture& image, const GuestTextureResource& resource) {
    return resource.dccAddress != 0 && image.Descriptor().dccAddress != resource.dccAddress && !image.ServesKeysAt(resource.dccAddress);
}

// Whether a sampled texture over `resource` can be a view of the surface's cached storage image
// instead of a CPU snapshot (see cachedTexture): the format has a storage form and is not block
// compressed, and the surface lives in host-imported memory, where the image uploads and refreshes
// GPU-direct (a snapshot of such a surface reads and compares its bytes on the CPU, waiting for
// the recorded work that wrote them). APS5_NO_SAMPLED_FROM_STORAGE=1 keeps every sampled texture a
// snapshot as before.
bool SampledFromStorageEligible(const Context& context, std::uint32_t format, std::uint64_t address, std::uint64_t guestBytes) {
    static const bool disabled = std::getenv("APS5_NO_SAMPLED_FROM_STORAGE") != nullptr;
    if (disabled || IsBlockCompressed(format) || !StorageFormatAvailable(context, format)) return false;
    return HostImportCovers(context, address, static_cast<std::size_t>(guestBytes));
}

bool SampledFromStorageEligible(const Context& context, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    return SampledFromStorageEligible(context, resource.format, resource.baseAddress, guestBytes);
}

// Whether a sampled texture over a fast-cleared, storage-eligible surface views the cleared storage
// image (see cachedTexture). APS5_NO_CLEARED_VIEW=1 keeps such surfaces snapshots, as before.
bool ClearedViewEnabled() {
    static const bool disabled = std::getenv("APS5_NO_CLEARED_VIEW") != nullptr;
    return !disabled;
}

std::uint64_t DepthCopyMinBytes() {
    static const std::uint64_t bytes = [] {
        const char* text = std::getenv("APS5_DEPTH_COPY_MIN_KIB");
        return (text != nullptr ? std::strtoull(text, nullptr, 10) : 1024ull) << 10u;
    }();
    return bytes;
}

// Surfaces whose storage image could not be made (no writable committed pages, say): remembered
// so the sampled lookup does not throw and fall back on every use. A failure is keyed by the surface
// (address and size: the heap reuses addresses) and holds only while the guest mappings are what
// they were when it failed (the allocation generation): pages committed later, or another surface
// at the address, get a fresh attempt.
struct StorageFailures {
    AgcDriver::Mutex mutex;
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> generations;
};

StorageFailures& StorageFailed() {
    static StorageFailures failures;
    return failures;
}

// The cached storage image for a sampled descriptor, or null when it cannot be made (then the
// snapshot path serves the descriptor, as before). Every null return counts as a fallback.
std::shared_ptr<StorageTexture> sampledStorageSource(const Context& context, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    auto& counters = TextureCounts();
    auto& failures = StorageFailed();
    const auto surface = std::make_pair(resource.baseAddress, guestBytes);
    const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    {
        std::lock_guard lock(failures.mutex);
        if (const auto it = failures.generations.find(surface); it != failures.generations.end()) {
            if (it->second == generation) {
                counters.storageFallbacks.fetch_add(1, std::memory_order_relaxed);
                return nullptr;
            }
            failures.generations.erase(it);
        }
    }
    try {
        return cachedStorageTexture(context, {}, resource, 0, guestBytes);
    } catch (const std::exception& error) {
        counters.storageFallbacks.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard lock(failures.mutex);
        if (failures.generations.size() < 4096 && failures.generations.emplace(surface, generation).second) std::fprintf(stderr, "[textures] sampled texture 0x%llx (%ux%u format %u) keeps the snapshot path: %s\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, error.what());
        return nullptr;
    }
}

std::uint64_t storageCacheBytes();

std::optional<std::uint32_t> largestDeviceLocalHeap(const VkPhysicalDeviceMemoryProperties& memory) {
    std::optional<std::uint32_t> heap;
    for (std::uint32_t i = 0; i < std::min<std::uint32_t>(memory.memoryHeapCount, VK_MAX_MEMORY_HEAPS); ++i) {
        if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 && (!heap || memory.memoryHeaps[i].size > memory.memoryHeaps[*heap].size)) heap = i;
    }
    return heap;
}

std::atomic<std::uint64_t> budgetOverride{0};

std::uint64_t forcedCacheBudget() {
    if (const auto overridden = budgetOverride.load(std::memory_order_relaxed); overridden != 0) return overridden;
    static const std::uint64_t forced = [] {
        const char* text = std::getenv("APS5_TEXTURE_CACHE_MIB");
        return text != nullptr ? std::strtoull(text, nullptr, 10) << 20u : 0u;
    }();
    return forced;
}

// Both caches' budgets (TextureCacheBudgetsFor), read from VK_EXT_memory_budget at most once a
// second and shared by the two caches; under its own mutex, which nothing else is taken under.
struct BudgetState {
    std::mutex mutex;
    VkPhysicalDevice device = VK_NULL_HANDLE;
    std::chrono::steady_clock::time_point readAt{};
    TextureCacheBudgets budgets;
    std::uint64_t reportedBudget = 0;
    std::chrono::steady_clock::time_point reportedAt{};
};

BudgetState& Budgets() {
    static BudgetState state;
    return state;
}

std::uint64_t forcedOverflowBudget() {
    static const std::uint64_t forced = [] {
        const char* text = std::getenv("APS5_TEXTURE_OVERFLOW_MIB");
        return text != nullptr ? std::strtoull(text, nullptr, 10) << 20u : 0u;
    }();
    return forced;
}

TextureCacheBudgets readBudgets(const Context& context, bool reread = false);

TextureCacheBudgets currentBudgets(const Context& context) {
    auto budgets = readBudgets(context);
    if (const auto forced = forcedOverflowBudget(); forced != 0) budgets.overflow = forced;
    return budgets;
}

TextureCacheBudgets readBudgets(const Context& context, bool reread) {
    if (const auto forced = forcedCacheBudget(); forced != 0) return {forced, forced + forced / 4u, forced, 0, forced + forced / 4u, 0};
    if (context.memoryProperties2 == nullptr || context.physical == VK_NULL_HANDLE) return TextureCacheBudgetsFor(context.memory, nullptr, {});
    auto& state = Budgets();
    std::lock_guard lock(state.mutex);
    const auto now = std::chrono::steady_clock::now();
    if (!reread && state.device == context.physical && now - state.readAt < std::chrono::seconds(1)) return state.budgets;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    properties.pNext = &reported;
    context.memoryProperties2(context.physical, &properties);
    const TextureMemoryUse use{SampledTextureMemory(), SampledCachedBytes().load(std::memory_order_relaxed), storageCacheBytes(), SampledTextureOverflowMemory()};
    const bool fresh = state.device != context.physical;
    const auto budgets = TextureCacheBudgetsFor(context.memory, &reported, use, fresh ? 0u : state.budgets.target);
    if (const auto heap = largestDeviceLocalHeap(context.memory); heap && (fresh || SampledBudgetReportDue(state.reportedBudget, budgets.sampled, now - state.reportedAt))) {
        char was[48] = "";
        if (!fresh) std::snprintf(was, sizeof(was), " (was %llu MiB)", static_cast<unsigned long long>(state.reportedBudget >> 20u));
        const auto outside = use.sampled - std::min(use.sampled, use.sampledCached);
        std::fprintf(stderr, "[gpu] sampled texture cache budget %llu MiB%s, ceiling %llu MiB; storage image cache budget %llu MiB; the texture caches' share of the VRAM %llu MiB: device-local heap %u has a VK_EXT_memory_budget budget of %llu MiB and uses %llu MiB, %llu MiB of it sampled textures (%llu MiB in the cache, %llu MiB outside it) and %llu MiB cached storage images\n", static_cast<unsigned long long>(budgets.sampled >> 20u), was, static_cast<unsigned long long>(budgets.ceiling >> 20u), static_cast<unsigned long long>(budgets.storage >> 20u), static_cast<unsigned long long>(budgets.target >> 20u), *heap, static_cast<unsigned long long>(reported.heapBudget[*heap] >> 20u), static_cast<unsigned long long>(reported.heapUsage[*heap] >> 20u), static_cast<unsigned long long>(use.sampled >> 20u), static_cast<unsigned long long>(use.sampledCached >> 20u), static_cast<unsigned long long>(outside >> 20u), static_cast<unsigned long long>(use.storageCached >> 20u));
        state.reportedBudget = budgets.sampled;
        state.reportedAt = now;
    }
    state.device = context.physical;
    state.readAt = now;
    state.budgets = budgets;
    return budgets;
}

std::uint64_t storageBudget(const Context& context, std::uint64_t required = 0) {
    auto budgets = currentBudgets(context);
    if (forcedCacheBudget() != 0 || context.bufferPool == nullptr || context.memoryProperties2 == nullptr || context.physical == VK_NULL_HANDLE) return budgets.storage;
    const auto needed = std::min(required, TextureCacheBudget(context.memory));
    if (needed > budgets.storage && context.bufferPool->ReclaimIdleDevice(needed - budgets.storage, std::chrono::steady_clock::now()) != 0) budgets = readBudgets(context, true);
    return budgets.storage;
}

// A snapshot over memory outside the write watch is compared with guest memory again only when this
// key changed: a write-watch mark in the range, a driver store, a mapping change or a recorded GPU
// write (as StorageTexture::compareUntracked). Taken before the snapshot is read or compared, so a
// CPU store racing with that leaves a mark for the next key. Zero when the range has no hint.
std::array<std::uint64_t, 4> untrackedKey(std::uint64_t address, std::size_t bytes) {
    if (GuestMemory::CollectEpoch() == 0) return {};
    const auto hint = GuestMemory::UntrackedHint(address, bytes);
    if (hint == 0) return {};
    const auto* recorder = Recorder::Active();
    const std::array<std::uint64_t, 4> key{hint, GuestMemory::NewestStamp(address, bytes), GuestMemory::ForgetSerial(), recorder != nullptr ? recorder->NewestWriteNote(address, bytes) : 0};
    return (key[2] & 1u) == 0 ? key : std::array<std::uint64_t, 4>{};
}

std::span<const std::uint32_t> firstLayerWords(std::span<const std::uint32_t> words, std::optional<ShaderRecompiler::DescriptorImageShape> shape, std::array<std::uint32_t, 8>& narrowed, bool report) {
    if (shape != ShaderRecompiler::DescriptorImageShape::Image2D || words.size() != narrowed.size() || IsNullTextureResource(words)) return words;
    try {
        if (!FirstLayerDescriptor(words, narrowed)) return words;
    } catch (const std::runtime_error&) {
        return words;
    }
    if (report) {
        static std::atomic<std::uint64_t> narrowings{0};
        const auto count = narrowings.fetch_add(1) + 1;
        if (count <= 5 || (count & (count - 1)) == 0) {
            const auto resource = DecodeTextureResource(words);
            std::fprintf(stderr, "[gpu] 2D read of the 2D array 0x%llx (%ux%u format %u, slices %u..%u) built from slice %u alone; %llu such reads\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, resource.baseArray, resource.depthOrLastArray, resource.baseArray, static_cast<unsigned long long>(count));
        }
    }
    return narrowed;
}

void reportGuestAllocation(const GuestTextureResource& resource, std::uint64_t guestBytes) {
    const auto range = GuestAllocations::GuestAllocationsLookup_nid_postfix(resource.baseAddress, false);
    const auto geometry = DescribeSurface(resource);
    if (range == nullptr) {
        std::fprintf(stderr, "[gpu] view 0x%llx of %llu MiB lies in no registered guest allocation\n", static_cast<unsigned long long>(resource.baseAddress), static_cast<unsigned long long>(guestBytes >> 20u));
        return;
    }
    const auto end = range->address + range->bytes;
    const auto fitting = geometry.thick || geometry.layerBytes == 0 ? 0ull : std::min<std::uint64_t>(geometry.layers, (end - resource.baseAddress) / geometry.layerBytes);
    std::fprintf(stderr, "[gpu] view 0x%llx of %llu MiB lies in the guest allocation 0x%llx..0x%llx (%llu MiB, range 0x%llx..0x%llx, %s%s%s): %llu of its %u slices fit in the range\n", static_cast<unsigned long long>(resource.baseAddress), static_cast<unsigned long long>(guestBytes >> 20u), static_cast<unsigned long long>(range->allocationAddress), static_cast<unsigned long long>(range->allocationAddress + range->allocationBytes), static_cast<unsigned long long>(range->allocationBytes >> 20u), static_cast<unsigned long long>(range->address), static_cast<unsigned long long>(end), range->readable ? "r" : "-", range->writable ? "w" : "-", range->gpu ? "g" : "-", static_cast<unsigned long long>(fitting), geometry.layers);
}

// `guestBytes` is the surface size when the caller described the surface already (0: described here).
// Before a lookup (outside the cache's mutex): kept spilled snapshots nothing uses go while the
// overflow type (or system memory, for the linear ones) holds more than the overflow budget; while
// the budget is still passed, up to four of the oldest
// recorded batches are waited for (their one-offs go with them) and the kept ones trimmed again.
void boundOverflow(const Context& context) {
    if (SampledTextureOverflowMemory() == 0 && SampledTextureHostMemory() == 0) return;
    const auto budgets = currentBudgets(context);
    const auto excess = [&]() -> std::uint64_t {
        if (budgets.overflow == 0) return 0;
        const auto live = std::max(SampledTextureOverflowMemory(), SampledTextureHostMemory());
        return live > budgets.overflow ? live - budgets.overflow : 0;
    };
    auto& cache = Textures();
    std::vector<std::shared_ptr<Texture>> doomed;
    {
        const SampledCacheLock lock(cache);
        const auto over = excess();
        if (cache.overflowLru.empty() || over == 0) return;
        const auto retired = retiredSerial();
        const auto now = std::chrono::steady_clock::now();
        if (cache.overflowTrimFailed && cache.overflowTrimRetired == retired && now - cache.overflowTrimAt < std::chrono::milliseconds(50)) return;
        cache.overflowTrimFailed = trimOverflow(cache, over, doomed) == 0;
        cache.overflowTrimRetired = retired;
        cache.overflowTrimAt = now;
    }
    // Dropped while the open batch records: they go when it retires, so a wait comes first.
    const bool deferred = !doomed.empty();
    keepUntilRetired(doomed);
    if (!deferred && !cache.overflowTrimFailed) return;
    auto* recorder = Recorder::Active();
    for (int wait = 0; wait < 4 && recorder != nullptr && excess() != 0 && recorder->RetireOldest(); ++wait) {
        {
            const SampledCacheLock lock(cache);
            ++cache.counts.overflowWaits;
            ++cache.totals.overflowWaits;
            trimOverflow(cache, excess(), doomed);
        }
        keepUntilRetired(doomed);
    }
}

std::shared_ptr<Texture> cachedTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& requested, VkComponentMapping components, std::uint64_t guestBytes = 0, bool depthCompare = false, VkImageViewType multisampleViewType = VK_IMAGE_VIEW_TYPE_2D) {
    const auto resource = ArrayLayerWindow(requested, context.limits.maxImageArrayLayers);
    if (resource.baseAddress != requested.baseAddress) guestBytes = 0;
    if (resource.dimension == TextureDimension::k2DMsaa) {
        Require(!depthCompare, "native multisampled depth comparison sampling is not implemented");
        if (auto depth = DepthSurfaceTexture(context, words, resource, components, multisampleViewType)) {
            logLookup({depth.get(), resource, 0, DccKeys::Uncompressed, 0, nullptr, true});
            return depth;
        }
        auto target = MultisampleTargetTexture(context, resource, components, multisampleViewType);
        logLookup({target.get(), resource, 0, DccKeys::Uncompressed, 0, nullptr, true});
        return target;
    }
    if (resource.fmaskCmaskAddress != 0u) {
        Require(!depthCompare, "FMASK depth comparison sampling is not implemented");
        Require(resource.baseArray == 0u && resource.depthOrLastArray == 0u, "native FMASK array sampling is not implemented");
        CompletePendingCpuWrites(context, resource.fmaskCmaskAddress, CmaskBytes(resource.width, resource.height));
        auto native = MultisampleTargetFmaskTexture_nid_no_patch(context, resource, components);
        Require(native != nullptr, "compressed FMASK sampling without a matching resident native target is not implemented");
        return native;
    }
    CaptureTrace::Log("sampled-lookup address=%llx width=%u height=%u dcc=%llx", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, static_cast<unsigned long long>(resource.dccAddress));
    RequireNoMultisampleTarget(resource.baseAddress, guestBytes != 0 ? static_cast<std::size_t>(guestBytes) : 1u, "a texture read");
    if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->BoundKeptBytes();
    boundOverflow(context);
    if (auto depth = DepthSurfaceTexture(context, words, resource, components)) {
        logLookup({depth.get(), resource, 0, DccKeys::Uncompressed, 0, nullptr, true});
        return depth;
    }
    const auto depthBitsWidth = words.size() >= 4 ? ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) : 0u;
    constexpr auto float32 = static_cast<std::uint32_t>(ShaderRecompiler::IrBufferFormat::Format32Float);
    if (depthBitsWidth == 32u && resource.format != float32) {
        auto bits = resource;
        bits.format = float32;
        return cachedTexture(context, words, bits, components, guestBytes, depthCompare);
    }
    constexpr auto unorm16 = static_cast<std::uint32_t>(ShaderRecompiler::IrBufferFormat::Format16UNorm);
    if (depthBitsWidth == 16u && resource.format != unorm16) {
        auto normalized = resource;
        normalized.format = unorm16;
        return cachedTexture(context, words, normalized, components, guestBytes, depthCompare);
    }
    if (const auto window = ViewedArrayWindow(resource); window.baseAddress != resource.baseAddress) {
        static std::atomic<std::uint64_t> windows{0};
        const auto count = windows.fetch_add(1) + 1;
        if (count <= 5 || (count & (count - 1)) == 0) std::fprintf(stderr, "[gpu] sampled array view 0x%llx (%ux%u format %u, dimension %d, slices %u..%u) served from its viewed slices at 0x%llx; %llu such views\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<int>(resource.dimension), resource.baseArray, resource.depthOrLastArray, static_cast<unsigned long long>(window.baseAddress), static_cast<unsigned long long>(count));
        return cachedTexture(context, words, window, components, 0, depthCompare);
    }
    if (guestBytes == 0) guestBytes = DescribeSurface(resource).guestBytes;
    if (guestBytes >= (1ull << 30u)) {
        static std::atomic<std::uint64_t> large{0};
        const auto count = large.fetch_add(1) + 1;
        if (count <= 5 || (count & (count - 1)) == 0) {
            std::fprintf(stderr, "[gpu] sampled view 0x%llx (%ux%u format %u, dimension %d, tile %d, slices %u..%u) covers %llu MiB; %llu such views\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<int>(resource.dimension), static_cast<int>(resource.tileMode), resource.baseArray, resource.depthOrLastArray, static_cast<unsigned long long>(guestBytes >> 20u), static_cast<unsigned long long>(count));
            reportGuestAllocation(resource, guestBytes);
        }
    }
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    const bool profile = LookupOutcomes::Profiled();
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::optional<DccKeys> scanned;
    const auto scanKeys = [&] {
        if (scanned.has_value()) return *scanned;
        const auto scanStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        scanned = TextureClearKeys(resource, guestBytes);
        if (resource.dccAddress != 0) ++sampledLookupKeyScans;
        if (profile && resource.dccAddress != 0) LookupOutcomes::Add(LookupOutcomes::DccScan, scanStart);
        return *scanned;
    };
    auto& counters = TextureCounts();
    const auto address = resource.baseAddress;
    const auto bytes = static_cast<std::size_t>(guestBytes);
    static_cast<void>(FlushDepthPlanes(address, bytes));
    // A storage image whose results for this surface (or for a mip chain containing it) are still on
    // the GPU supplies the texture by a view of it; anything else needs those results in guest
    // memory first.
    const bool depthCopy = depthCompare && guestBytes >= DepthCopyMinBytes() && Texture::DepthCopyFormat(ResolveTextureFormat(resource.format)) != VK_FORMAT_UNDEFINED && resource.dimension != TextureDimension::k3D;
    const bool srgbExpand = !depthCompare && SrgbExpandsToRgba8(context, resource.format);
    auto source = StorageTexture::FindPending(address, guestBytes);
    // APS5_TRACE_SAMPLED_FLUSH: why a surface with pending storage results is not served from them
    // (the flush below then waits for the GPU under the mutex).
    const auto pendingSource = source;
    const char* dropReason = source == nullptr ? "no pending image covers it" : nullptr;
    const bool blockCopy = source != nullptr && !depthCompare && !srgbExpand && Texture::CanBlockCopyFrom(*source, resource);
    const bool repackCopy = source != nullptr && !depthCompare && Texture::CanRepackCopyFrom(*source, resource);
    if (source != nullptr && !blockCopy && !repackCopy && (depthCompare ? !depthCopy || !Texture::CanDepthCopyFrom(*source, resource) : srgbExpand ? !Texture::CanSrgbExpandFrom(context, *source, resource) : !Texture::CanCopyFrom(*source, resource))) {
        source.reset();
        dropReason = depthCompare ? "no depth copy" : srgbExpand ? "no sRGB expand" : "no copy path (format/layout)";
    }
    if (source != nullptr && MetadataMoved(*source, resource)) {
        source.reset();
        dropReason = "metadata moved";
    }
    std::optional<DccKeys> keys;
    bool clearThroughKeys = false;
    if (source != nullptr && resource.dccAddress != 0 && IsDccClear(source->FilledKeys())) {
        keys = scanKeys();
        if (*keys == source->FilledKeys()) {
            std::array<std::byte, 16> probe{};
            if (!FillDccClear(ResolveTextureFormat(resource.format), *keys, resource.dccAlphaOnMsb, probe)) {
                char text[256];
                std::snprintf(text, sizeof(text), "AGC graphics: sampled texture 0x%llx (%ux%u format %u, dcc 0x%llx) reads %s DCC keys filled over its pending image, a clear value the format has no encoding for", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<unsigned long long>(resource.dccAddress), DccKeysName(*keys));
                throw std::runtime_error(text);
            }
            static const bool traceKeys = std::getenv("APS5_TRACE_DCC_KEYS") != nullptr;
            if (traceKeys) std::fprintf(stderr, "[dcc-keys] sampled 0x%llx through keys 0x%llx reads the %s fill, not the pending image\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(resource.dccAddress), DccKeysName(*keys));
            source.reset();
            clearThroughKeys = true;
        }
    }
    if (source != nullptr && resource.dccAddress != 0 && (keys.has_value() ? *keys : scanKeys()) == DccKeys::ClearSingle) {
        source.reset();
        dropReason = "single-colour DCC clear";
    }

    // Otherwise a surface in host-imported memory is viewed through its cached storage image (made
    // here when there is none): its refresh after a CPU or GPU write is a GPU-direct detile from the
    // import, recorded behind the producer, so no bytes are read or compared on the CPU and nothing
    // waits for the producer. A fast-cleared surface (keys) is viewed only through an image whose own
    // descriptor carries the DCC address (below); it stays a snapshot otherwise, its texels not read.
    if ((!depthCompare || depthCopy) && source == nullptr && !clearThroughKeys && SampledFromStorageEligible(context, resource, guestBytes)) {

        keys = scanKeys();
        if (*keys == DccKeys::Uncompressed) {
            source = sampledStorageSource(context, resource, guestBytes);
            if (source != nullptr && depthCompare && !Texture::CanDepthCopyFrom(*source, resource)) source.reset();
            if (source != nullptr && srgbExpand && !Texture::CanSrgbExpandFrom(context, *source, resource)) source.reset();
        } else if (!depthCompare && ClearedViewEnabled() && StorageClearAvailable(context, resource.format, *keys)) {
            // A fast-cleared surface is viewed as well, through its image cleared on the GPU (the
            // lookup's Refresh, see StorageTexture::upload), when the image's own descriptor names
            // the same DCC metadata so the refresh sees the keys. A snapshot of the clear texels
            // (filled and uploaded on the CPU, then replaced by a view once results are pending: two
            // 4K uploads per clear at the movie stage) serves the surface otherwise, as before.
            auto candidate = sampledStorageSource(context, resource, guestBytes);
            if (candidate != nullptr && candidate->Descriptor().dccAddress == resource.dccAddress) source = std::move(candidate);
        }
        scanned.reset();
        // The lookup's Refresh may have flushed another image's results over the memory (which
        // marks the keys uncompressed): a surface that stays a snapshot re-reads them behind that
        // flush, as it does behind its own. APS5_NO_KEYS_RESCAN=1 keeps the keys scanned above.
        if (source != nullptr && IsRepackedTextureFormat(resource.format) && !Texture::CanRepackCopyFrom(*source, resource)) source.reset();
        static const bool rescan = std::getenv("APS5_NO_KEYS_RESCAN") == nullptr;
        if (rescan && source == nullptr) keys.reset();
    }
    // Pages of the surface written since they were last collected are stamped now, before any
    // UnchangedSince (here and on a cache hit) looks at them: the checks only compare stamped blocks,
    // so a write the game made since the last walk is invisible until a collect stamps its page. The
    // walk is memoized per packet, so a texture viewed from a storage image repeats it for free.
    auto generation = GuestMemory::CollectWrites(address, bytes);
    if (source != nullptr && !GuestMemory::UnchangedSince(address, bytes, source->GenerationFor(address, bytes))) {
        // The CPU wrote the memory while results were pending: Refresh merges them the usual way.
        refreshForView(*source, resource);
        scanned.reset();
    }
    // Results stored to guest memory just now are stamped newer than `generation`; a snapshot read
    // after them is current at the generation of a second (memoized) collect. The store marks the
    // surface's DCC keys uncompressed, so the keys are read after it.
    // A block-compressed surface the title compresses on the GPU one layer's level at a time, each
    // into a storage image of the block's texel format (Yotei's reflection probes, BC6H through
    // R32G32B32A32_UINT): a cached snapshot takes those results by device copies below
    // (Texture::CopyPendingSlices) instead of a store and a re-read, which waited for the open
    // batch that wrote them (~100 ms a frame on the GPU mutex). A new snapshot still reads guest
    // memory, storing them first. APS5_NO_SLICE_COPY=1 stores them for every lookup, as before.
    std::vector<PendingSlice> slices;
    bool sliceCopy = !disabled && generation != 0 && source == nullptr && !clearThroughKeys && !depthCompare && !srgbExpand && IsBlockCompressed(resource.format) && resource.dccAddress == 0 && Recorder::Active() != nullptr && Recorder::ThreadLane() == 0 && !Recorder::InCompletion() && !RuntimeSwitches::Set(RuntimeSwitch::NoSliceCopy) && PendingStorageOverlaps(address, bytes, nullptr) && StorageTexture::PendingSlicesOf(resource, DescribeSurface(resource), slices);
    // Marked when recognized, not when first copied: the flip's store of small images (which skips
    // marked ones) would otherwise stamp the surface every frame, so no lookup ever proves its
    // snapshot by stamps and the first copy never comes (best111).
    if (sliceCopy) {
        for (const auto& slice : slices) slice.image->MarkSliceServed();
    }
    const auto flushStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool stagedClear = source != nullptr && resource.dccAddress != 0 && [&] {
        const auto observedKeys = keys.has_value() ? *keys : scanKeys();
        return IsDccClear(observedKeys) || observedKeys == DccKeys::ClearSingle;
    }();
    bool stagedFlushed = false;
    static const bool traceSampledFlush = std::getenv("APS5_TRACE_SAMPLED_FLUSH") != nullptr;
    const auto traceStart = traceSampledFlush ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool flushAsked = (source == nullptr || stagedClear) && !clearThroughKeys && !sliceCopy;
    if (flushAsked && StorageTexture::FlushPending(address, bytes, nullptr, "sampled texture")) {
        if (profile) LookupOutcomes::Add(LookupOutcomes::PendingFlush, flushStart);
        if (traceSampledFlush) {
            const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - traceStart).count();
            char image[160] = "none";
            if (pendingSource != nullptr) {
                const auto& d = pendingSource->Descriptor();
                std::snprintf(image, sizeof(image), "0x%llx %ux%u fmt %u tile %d mips %u", static_cast<unsigned long long>(d.baseAddress), d.width, d.height, d.format, static_cast<int>(d.tileMode), d.mipCount);
            }
            static std::uint64_t describedAddress = 0;
            if (describedAddress != resource.baseAddress) {
                describedAddress = resource.baseAddress;
                const auto geometry = DescribeSurface(resource);
                std::fprintf(stderr, "[sampled-flush] geometry 0x%llx: layers %u imageLayers %u layerBytes 0x%llx guestBytes 0x%llx thick %d dim %d array %u\n", static_cast<unsigned long long>(resource.baseAddress), geometry.layers, geometry.imageLayers, static_cast<unsigned long long>(geometry.layerBytes), static_cast<unsigned long long>(geometry.guestBytes), geometry.thick ? 1 : 0, static_cast<int>(resource.dimension), resource.depthOrLastArray);
                for (std::size_t level = 0; level < geometry.mips.size(); ++level) {
                    const auto& m = geometry.mips[level];
                    std::fprintf(stderr, "[sampled-flush]   mip %zu: %ux%u tiled 0x%llx+0x%llx tail %d (%u,%u) blocksPerRow %u\n", level, m.width, m.height, static_cast<unsigned long long>(m.tiledOffset), static_cast<unsigned long long>(m.tiledSize), m.tail ? 1 : 0, m.tailX, m.tailY, m.blocksPerRow);
                }
            }
            std::fprintf(stderr, "[sampled-flush] %.1f ms: 0x%llx %ux%u fmt %u tile %d mips %u%s; reason: %s; pending image: %s\n", ms, static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.mipCount, depthCompare ? " depth-compare" : "", stagedClear ? "staged DCC clear" : dropReason != nullptr ? dropReason : "a storage image lookup found none (not eligible / keys not uncompressed)", image);
        }
        generation = GuestMemory::CollectWrites(address, bytes);
        if (stagedClear) {
            stagedFlushed = true;
            keys = DccKeys::Uncompressed;
        } else {
            keys.reset();
        }
        if (stagedFlushed) source.reset();
    }
    if (flushAsked) scanned.reset();
    if (!keys.has_value()) keys = scanKeys();
    if (disabled) {
        if (source != nullptr && depthCompare) {
            auto texture = std::make_shared<Texture>(context, source, resource, components, Texture::DepthCopyTag{});
            texture->RefreshDepthCopy();
            return texture;
        }
        if (source != nullptr && blockCopy) {
            auto texture = std::make_shared<Texture>(context, source, resource, components, Texture::BlockCopyTag{});
            texture->RefreshDepthCopy();
            return texture;
        }
        if (source != nullptr && IsRepackedTextureFormat(resource.format)) {
            auto texture = std::make_shared<Texture>(context, source, resource, components, Texture::RepackCopyTag{});
            texture->RefreshDepthCopy();
            return texture;
        }
        if (source != nullptr && srgbExpand) {
            auto texture = std::make_shared<Texture>(context, source, resource, components, Texture::SrgbExpandTag{});
            texture->RefreshDepthCopy();
            return texture;
        }
        if (source != nullptr) return std::make_shared<Texture>(context, source, resource, components);
        std::vector<std::byte> snapshot(bytes);
        ReadTextureSurface(resource, *keys, snapshot);
        return std::make_shared<Texture>(context, *context.detiler, resource, components, snapshot, depthCompare);
    }
    auto snapshotGeneration = generation;
    // Slice copies serve only a cached snapshot proved current by write stamps alone: any read of
    // the surface (a compare, a new snapshot) goes through the flush hook, which stores the pending
    // slices after `generation` was collected and waits for their writer; the snapshot made then
    // looks stale at the next lookup, whose compare stores the next slices, and so on every frame
    // (best110). So everything else stores them first and collects again, as without slice copies.
    // APS5_TRACE_SAMPLED_FLUSH: why a lookup with slices stored them instead of copying ([slice-store]).
    const auto storeSlices = [&](const char* why) {
        if (traceSampledFlush) std::fprintf(stderr, "[slice-store] 0x%llx: %s (%zu slices)\n", static_cast<unsigned long long>(address), why, slices.size());
        StorageTexture::FlushPending(address, bytes, nullptr, "sampled texture");
        generation = GuestMemory::CollectWrites(address, bytes);
        snapshotGeneration = generation;
        sliceCopy = false;
        reportSliceCopies(false);
    };
    const auto takeSlices = [&](Texture& texture) {
        if (texture.CopyPendingSlices(slices)) {
            reportSliceCopies(true);
            return true;
        }
        storeSlices("the texture cannot take copies");
        return false;
    };
    std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>> committed;
    if (source == nullptr && generation == 0) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        if (const auto partial = GuestMemory::CollectCommittedWrites(address, bytes, ranges); partial != 0) {
            snapshotGeneration = partial;
            committed = std::move(ranges);
        }
    }
    const auto untracked = source == nullptr && generation == 0 && !committed.has_value() && *keys == DccKeys::Uncompressed ? untrackedKey(address, bytes) : std::array<std::uint64_t, 4>{};
    auto& cache = Textures();
    const auto key = MakeTextureKey(context.device, words, components, depthCompare);
    const DropReleasedDependents dropAfterUnlock;
    const SampledCacheLock lock(cache);
    dropUncachedViews(cache);
    if (auto it = findTexture(cache, key); it != cache.entries.end()) {
        // Held while the hit is checked: a refresh below may allocate, and an out-of-memory reclaim
        // on this thread (ReclaimTextureMemory) then passes over the entry as in use.
        const auto checked = it->texture;
        if (it->source != nullptr) {
            // The view follows the storage image, whatever the GPU wrote to it since; guest memory
            // written meanwhile is taken in by refreshing the image. A fast clear the image cannot
            // see (keys, and no pending results to prefer) ends the view: a snapshot holds the clear.
            if (!stagedFlushed && (source == nullptr || source == it->source) && (source != nullptr || *keys == DccKeys::Uncompressed) && StorageImageServesKeys(*it->source, resource.dccAddress)) {
                if (!GuestMemory::UnchangedSince(address, bytes, it->source->GenerationFor(address, bytes))) refreshForView(*it->source, resource);
                if (it->texture->IsDepthCopy()) it->texture->RefreshDepthCopy();
                it->keys = *keys;
                touchTexture(cache, it);
                logLookup({it->texture.get(), resource, guestBytes, *keys, 0, it->source.get()});
                reportTextureCounters();
                if (profile) LookupOutcomes::Add(*keys != DccKeys::Uncompressed ? LookupOutcomes::SampledHitClearedView : LookupOutcomes::SampledHitView, start);
                return it->texture;
            }
        } else if (source == nullptr && it->bytes == guestBytes && it->keys == *keys) {
            // Unwritten pages need no comparison; partially resident textures compare only committed
            // pages. The compare goes through the flush hook: it waits for recorded work over the
            // surface (counted, and named for the [hooksync] line).
            const auto equalsCommitted = [&] {
                const bool collected = generation != 0 || committed.has_value();
                const auto reason = [&] {
                    if (generation != 0) return it->committed.has_value() ? SnapshotCompare::Layout : SnapshotCompare::Stamped;
                    if (committed.has_value()) {
                        if (it->committed != committed) return SnapshotCompare::Layout;
                        return Recorder::SnapshotWriteOverlaps(address, bytes) ? SnapshotCompare::Pending : SnapshotCompare::CommittedStamped;
                    }
                    return untracked[0] == 0 ? SnapshotCompare::Untracked : SnapshotCompare::HintMoved;
                }();
                if (!collected && *keys == DccKeys::Uncompressed) return untrackedStillMatches(address, bytes, it->hashes, it->blockSince, it->blockGuard, it->blockLayout, reason, it->owned);
                return snapshotStillMatches(address, bytes, it->hashes, collected ? it->generation : 0, it->committed, reason);
            };
            const auto unchanged = [&] {
                if (it->committed != committed) return false;
                return committed.has_value() ? !Recorder::SnapshotWriteOverlaps(address, bytes) && GuestMemory::UnchangedCommittedSince(*committed, it->generation) : GuestMemory::UnchangedSince(address, bytes, it->generation);
            };
            const auto kept = [&] {
                if (untracked[0] == 0) return false;
                const bool same = it->untracked == untracked && !Recorder::SnapshotWriteOverlaps(address, bytes);
                (same ? counters.untrackedKept : counters.untrackedCompared).fetch_add(1, std::memory_order_relaxed);
                return same;
            };
            // The slices' own GPU writes stamp the 64 KiB blocks they cover (the driver's MarkWritten,
            // no CPU store), so stamps alone never prove the snapshot while slices are pending
            // (best113: 7 of 141 blocks a frame, the current cubemap's). The snapshot holds when every
            // block stamped since it was taken was stamped by the driver, not a CPU store, and lies
            // under a pending slice the copy below replaces.
            const auto onlySliceStamps = [&] {
                const auto blocks = ((address + bytes + 0xffffu) >> 16u) - (address >> 16u);
                std::vector<std::uint64_t> generations(blocks, it->generation);
                std::vector<std::uint8_t> changed(blocks), cpu(blocks);
                GuestMemory::ChangedBlocks(address, bytes, generations, changed, cpu);
                for (std::size_t k = 0; k < blocks; ++k) {
                    if (changed[k] == GuestMemory::BlockUnchanged) continue;
                    if (cpu[k] != 0) return false;
                    const auto begin = ((address >> 16u) + k) << 16u;
                    const auto end = begin + 0x10000u;
                    const bool underSlice = std::any_of(slices.begin(), slices.end(), [&](const PendingSlice& slice) {
                        const auto sliceBegin = slice.image->Descriptor().baseAddress;
                        return sliceBegin < end && begin < sliceBegin + slice.image->GuestBytes();
                    });
                    if (!underSlice) return false;
                }
                return true;
            };
            const bool sliceHit = sliceCopy && *keys == DccKeys::Uncompressed && it->generation != 0 && (unchanged() || onlySliceStamps());
            if (sliceCopy && !sliceHit) {
                if (traceSampledFlush && *keys == DccKeys::Uncompressed && it->generation != 0) {
                    // Which 64 KiB tracker blocks moved since the snapshot, and whether a CPU store
                    // (a collect) or the driver's own MarkWritten stamped them.
                    const auto blocks = ((address + bytes + 0xffffu) >> 16u) - (address >> 16u);
                    std::vector<std::uint64_t> generations(blocks, it->generation);
                    std::vector<std::uint8_t> changed(blocks), cpu(blocks);
                    GuestMemory::ChangedBlocks(address, bytes, generations, changed, cpu);
                    std::size_t moved = 0, byCpu = 0;
                    std::string first;
                    const auto layerBytes = DescribeSurface(resource).layerBytes;
                    for (std::size_t k = 0; k < blocks; ++k) {
                        if (changed[k] == GuestMemory::BlockUnchanged) continue;
                        ++moved;
                        byCpu += cpu[k] != 0;
                        if (moved <= 6) {
                            const auto blockStart = ((address >> 16u) + k) << 16u;
                            const auto offset = blockStart > address ? blockStart - address : 0u;
                            char item[48];
                            std::snprintf(item, sizeof(item), " +0x%llx(layer %llu%s)", static_cast<unsigned long long>(offset), static_cast<unsigned long long>(layerBytes != 0 ? offset / layerBytes : 0), cpu[k] != 0 ? " cpu" : "");
                            first += item;
                        }
                    }
                    std::fprintf(stderr, "[slice-store] 0x%llx: cached snapshot stamped since generation %llu: %zu of %llu blocks moved, %zu by CPU stores:%s\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(it->generation), moved, static_cast<unsigned long long>(blocks), byCpu, first.c_str());
                }
                storeSlices(*keys != DccKeys::Uncompressed ? "keys not uncompressed" : "cached snapshot stamped");
            }
            // A slice copy the texture cannot take stores the slices instead: the snapshot is stale then.
            if ((sliceHit || (*keys != DccKeys::Uncompressed && *keys != DccKeys::ClearSingle) || unchanged() || kept() || equalsCommitted()) && (!sliceCopy || takeSlices(*it->texture))) {
                it->generation = snapshotGeneration;
                it->committed = std::move(committed);
                it->untracked = untracked;
                touchTexture(cache, it);
                logLookup({it->texture.get(), resource, guestBytes, *keys, generation, nullptr, false, it->committed.has_value() ? snapshotGeneration : 0, it->committed});
                reportTextureCounters();
                if (profile) LookupOutcomes::Add(LookupOutcomes::SampledHitSnapshot, start);
                return it->texture;
            }
        }
        eraseTexture(cache, it);
        ++cache.counts.stale;
    }
    if (sliceCopy) storeSlices("no current cached snapshot");
    const auto budgets = currentBudgets(context);
    const auto insert = [&](CachedTexture&& entry, const std::shared_ptr<Texture>& texture) {
        cache.bytes += entry.accounted;
        if (source == nullptr) {
            cache.snapshotBytes += entry.accounted;
            SampledCachedBytes().store(cache.snapshotBytes, std::memory_order_relaxed);
        }
        entry.present = currentFrame(cache);
        entry.presentSeen = Recorder::Presents();
        ++cache.counts.inserted;
        ++cache.totals.inserted;
        texture->SetSampledCacheState(Texture::CacheState::Cached);
        cache.entries.push_front(std::move(entry));
        cache.index[key] = cache.entries.begin();
        cache.byTexture[texture.get()] = cache.entries.begin();
        if (source != nullptr) cache.views.emplace(source.get(), cache.entries.begin());
    };
    if (source == nullptr) {
        if (const auto found = cache.oneOffs.find(key); found != cache.oneOffs.end()) {
            auto& oneOff = found->second;
            auto texture = oneOff.texture.lock();
            const auto current = [&] {
                if (*keys != DccKeys::Uncompressed || GuestMemory::UnchangedSince(address, bytes, oneOff.generation)) return true;
                return snapshotStillMatches(address, bytes, oneOff.hashes, generation != 0 ? oneOff.generation : 0, std::nullopt, SnapshotCompare::OneOff);
            };
            if (texture != nullptr && oneOff.bytes == guestBytes && oneOff.keys == *keys && current()) {
                ++cache.counts.oneOffReuses;
                ++cache.totals.oneOffReuses;
                if (oneOff.kept != nullptr) {
                    ++cache.counts.spilledReuses;
                    ++cache.totals.spilledReuses;
                    cache.overflowLru.splice(cache.overflowLru.begin(), cache.overflowLru, oneOff.lru);
                }
                texture->NoteUse(useSerial());
                CachedTexture entry{key, address, guestBytes, std::move(oneOff.hashes), texture, *keys, generation};
                entry.accounted = texture->AllocationBytes();
                if (!texture->Spilled() && admitTexture(cache, key, entry.accounted, budgets)) {
                    eraseOneOff(cache, found);
                    insert(std::move(entry), texture);
                } else {
                    oneOff.hashes = std::move(entry.hashes);
                    oneOff.generation = generation;
                }
                logLookup({texture.get(), resource, guestBytes, *keys, generation, nullptr});
                reportCacheCounts(cache, budgets);
                reportTextureCounters();
                if (profile) LookupOutcomes::Add(LookupOutcomes::SampledHitSnapshot, start);
                return texture;
            }
            eraseOneOff(cache, found);
        }
    }
    CachedTexture entry{key, address, source != nullptr ? 0u : guestBytes, {}, nullptr, *keys, snapshotGeneration};
    entry.committed = std::move(committed);
    const auto createTexture = [&](auto&&... args) {
        const CacheHold reclaimable(sampledCacheReclaimable);
        return std::make_shared<Texture>(context, std::forward<decltype(args)>(args)...);
    };
    if (source != nullptr) {
        entry.source = source;
        entry.sourceVersion = source->Version();
        if (depthCompare) {
            entry.texture = createTexture(source, resource, components, Texture::DepthCopyTag{});
            entry.texture->RefreshDepthCopy();
        } else if (blockCopy) {
            entry.texture = createTexture(source, resource, components, Texture::BlockCopyTag{});
            entry.texture->RefreshDepthCopy();
        } else if (IsRepackedTextureFormat(resource.format)) {
            entry.texture = createTexture(source, resource, components, Texture::RepackCopyTag{});
            entry.texture->RefreshDepthCopy();
        } else if (srgbExpand) {
            entry.texture = createTexture(source, resource, components, Texture::SrgbExpandTag{});
            entry.texture->RefreshDepthCopy();
        } else {
            entry.texture = createTexture(source, resource, components);
        }
        entry.accounted = (source->Cached() ? 0u : heldBytes(*source)) + (entry.texture->IsDepthCopy() ? entry.texture->AllocationBytes() : 0u);
        counters.fromStorage.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Snapshot before the upload so a write racing with it is caught by the next comparison.
        const bool pending = *keys == DccKeys::Uncompressed && Recorder::SnapshotWriteOverlaps(address, bytes);
        if (pending) counters.pendingReads.fetch_add(1, std::memory_order_relaxed);
        if (!pending && generation == 0 && !entry.committed.has_value() && *keys == DccKeys::Uncompressed) {
            entry.blockGuard = untrackedGuard(address, bytes);
            entry.blockSince = (entry.blockGuard[0] & 1u) == 0 ? untrackedBaseline(address, bytes, entry.blockLayout) : 0;
        }
        std::vector<std::byte> snapshot(bytes);
        ReadTextureSurface(resource, *keys, snapshot);
        if (!pending) entry.untracked = untracked;
        static const bool traceTextures = std::getenv("APS5_TRACE_TEXTURES") != nullptr;
        if (traceTextures) {
            std::size_t nonzero = 0;
            for (std::size_t i = 0; i < snapshot.size(); i += 64) nonzero += snapshot[i] != std::byte{0};
            std::fprintf(stderr, "[texture] 0x%llx %ux%u format %u tile %d: %zu of %zu sampled bytes nonzero\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), nonzero, snapshot.size() / 64);
        }
        const bool spill = SpillEnabled() && !makeRoom(cache, bytes, budgets);
        entry.texture = createTexture(*context.detiler, resource, components, snapshot, depthCompare, spill);
        ++cache.counts.made;
        ++cache.totals.made;
        cache.madeSurfaces.insert(address * 0x9e3779b97f4a7c15ull ^ guestBytes);
        if (entry.texture->Spilled()) {
            ++cache.counts.spilled;
            ++cache.totals.spilled;
            cache.counts.spilledBytes += entry.texture->AllocationBytes();
            cache.totals.spilledBytes += entry.texture->AllocationBytes();
        }
        if (*keys == DccKeys::Uncompressed) {
            entry.hashes.resize(GuestMemory::HashBlockCount(address, bytes));
            GuestMemory::HashBlocksOf(address, snapshot, entry.hashes);
        }
        entry.accounted = entry.texture->AllocationBytes();
        counters.snapshots.fetch_add(1, std::memory_order_relaxed);
    }
    auto texture = entry.texture;
    texture->NoteUse(useSerial());
    if (texture->Spilled() || !admitTexture(cache, key, entry.accounted, budgets)) {
        texture->SetSampledCacheState(Texture::CacheState::Released);
        if (source == nullptr) rememberOneOff(cache, key, texture, guestBytes, std::move(entry.hashes), *keys, generation);
        logLookup({texture.get(), resource, guestBytes, *keys, generation, source.get()});
        reportCacheCounts(cache, budgets);
        reportTextureCounters();
        if (profile) LookupOutcomes::Add(source != nullptr ? LookupOutcomes::SampledMadeView : LookupOutcomes::SampledMadeSnapshot, start);
        return texture;
    }
    logLookup({texture.get(), resource, guestBytes, *keys, generation, source.get(), false, source == nullptr && entry.committed.has_value() ? snapshotGeneration : 0, source == nullptr ? entry.committed : std::nullopt});
    insert(std::move(entry), texture);
    reportCacheCounts(cache, budgets);
    reportTextureCounters();
    if (profile) LookupOutcomes::Add(source != nullptr ? LookupOutcomes::SampledMadeView : LookupOutcomes::SampledMadeSnapshot, start);
    return texture;
}

// Storage images stay on the GPU between dispatches: while guest memory still holds what an image was
// last uploaded from or written back as, its next use skips the upload and detile.
struct StorageKey {
    VkDevice device;
    std::array<std::uint32_t, 8> words;
    bool operator==(const StorageKey&) const = default;
};

struct StorageKeyHash {
    std::size_t operator()(const StorageKey& key) const noexcept {
        return static_cast<std::size_t>(hashWords(14695981039346656037ull ^ reinterpret_cast<std::uintptr_t>(key.device), key.words.data(), key.words.size()));
    }
};

struct CachedStorageTexture {
    StorageKey key;
    std::uint32_t mip;
    std::shared_ptr<StorageTexture> texture;
    std::uint64_t accounted = 0;
};

// As TextureCache: use order with a hash index by key, plus one by image for StorageImageCached.
struct StorageTextureCache {
    AgcDriver::Mutex mutex;
    std::list<CachedStorageTexture> entries;
    std::unordered_map<StorageKey, std::list<CachedStorageTexture>::iterator, StorageKeyHash> index;
    std::unordered_map<const StorageTexture*, std::list<CachedStorageTexture>::iterator> byImage;
    std::atomic<std::uint64_t> bytes{0};
};

StorageTextureCache& StorageTextures() {
    static StorageTextureCache cache;
    return cache;
}

std::uint64_t storageCacheBytes() {
    return StorageTextures().bytes.load(std::memory_order_relaxed);
}

std::list<CachedStorageTexture>::iterator findStorage(StorageTextureCache& cache, const StorageKey& key) {
    if (TextureHashEnabled()) {
        const auto found = cache.index.find(key);
        return found == cache.index.end() ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key == key) return it;
    }
    return cache.entries.end();
}

std::list<CachedStorageTexture>::iterator findStorageByImage(StorageTextureCache& cache, VkDevice device, const StorageTexture* image) {
    if (TextureHashEnabled()) {
        const auto found = cache.byImage.find(image);
        return found == cache.byImage.end() || found->second->key.device != device ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key.device == device && it->texture.get() == image) return it;
    }
    return cache.entries.end();
}

// Evicts an entry: its pending results go to guest memory first (the image may die with the entry).
std::shared_ptr<StorageTexture> evictStorage(StorageTextureCache& cache, std::list<CachedStorageTexture>::iterator it) {
    auto texture = it->texture;
    it->texture->SetCached(false);
    noteUncached(it->texture);
    it->texture->Flush();
    cache.bytes -= it->accounted;
    cache.index.erase(it->key);
    cache.byImage.erase(it->texture.get());
    cache.entries.erase(it);
    return texture;
}

// Storage images are shared by every descriptor of one surface (address, extent, layers, format, tile
// mode): the image holds the whole mip chain, and render targets in the same memory attach to it.
std::array<std::uint32_t, 8> SurfaceKey(const Context& context, const GuestTextureResource& resource) {
    // Guest formats that store in the same Vulkan format share the image (views carry the difference).
    return {static_cast<std::uint32_t>(resource.baseAddress | resource.pipeBankXor), static_cast<std::uint32_t>(resource.baseAddress >> 32u), resource.width, resource.height, (resource.depthOrLastArray << 16u) | (resource.mipCount & 0xffffu), (static_cast<std::uint32_t>(resource.tileMode) << 12u) | (static_cast<std::uint32_t>(resource.dimension) << 20u), resource.baseArray, static_cast<std::uint32_t>(StorageFormatForGuest(context, resource.format))};
}

struct ExtendedSurfaces {
    AgcDriver::Mutex mutex;
    std::map<std::array<std::uint32_t, 8>, std::uint32_t> levels;
    std::atomic<bool> any{false};
};

ExtendedSurfaces& ExtendedChains() {
    static ExtendedSurfaces surfaces;
    return surfaces;
}

std::uint32_t AllocatedLevels(const GuestTextureResource& resource) {
    return resource.allocatedMipCount != 0 ? resource.allocatedMipCount : resource.mipCount;
}

GuestTextureResource StorageSurface(const Context& context, const GuestTextureResource& viewed) {
    auto& surfaces = ExtendedChains();
    const bool extended = viewed.mipCount > AllocatedLevels(viewed);
    if (!extended && !surfaces.any.load(std::memory_order_acquire)) return viewed;
    auto surface = viewed;
    surface.mipCount = AllocatedLevels(viewed);
    const auto identity = SurfaceKey(context, surface);
    std::lock_guard lock(surfaces.mutex);
    if (extended) {
        auto& levels = surfaces.levels[identity];
        levels = std::max(levels, viewed.mipCount);
        surfaces.any.store(true, std::memory_order_release);
        surface.mipCount = levels;
    } else if (const auto it = surfaces.levels.find(identity); it != surfaces.levels.end()) {
        surface.mipCount = it->second;
    } else {
        surface.mipCount = viewed.mipCount;
    }
    return surface;
}

struct WrittenStorage {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> bytes;
    std::vector<WrittenStorageExtent> extents;
};

static WrittenStorage WrittenStorageBytes(const GuestTextureResource& viewed, std::uint32_t mip) {
    WrittenStorage written;
    const auto geometry = DescribeSurface(viewed);
    if (geometry.thick || mip >= geometry.mips.size()) return written;
    const auto& level = geometry.mips[mip];
    const auto bytesPerElement = BytesPerElement(viewed.format);
    const auto covered = CoveredMipBytes(viewed.tileMode, bytesPerElement, level);
    const bool blocks = viewed.tileMode != TextureTileMode::kLinear && !level.tail;
    const auto first = geometry.layers > 1u && geometry.imageDepth <= 1u ? std::min(viewed.baseArray, geometry.layers) : 0u;
    for (auto layer = first; layer < geometry.layers; ++layer) {
        if (!geometry.HasLayer(mip, layer)) continue;
        const auto begin = viewed.baseAddress + geometry.GuestLayerOffset(layer) + level.tiledOffset;
        for (const auto& [rangeBegin, rangeEnd] : covered) {
            if (!written.bytes.empty() && written.bytes.back().second == begin + rangeBegin) written.bytes.back().second = begin + rangeEnd;
            else written.bytes.emplace_back(begin + rangeBegin, begin + rangeEnd);
        }
        if (blocks) written.extents.push_back({begin, level.width, level.height, bytesPerElement, ThinBlockLayout(viewed.tileMode, bytesPerElement), level.blocksPerRow});
    }
    return written;
}

// `guestBytes` is the surface size when the caller described the surface already (0: described here).
std::shared_ptr<StorageTexture> lookupStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& viewed, std::uint32_t mip, std::uint64_t guestBytes, std::optional<std::pair<std::uint32_t, std::uint32_t>> layers);

std::shared_ptr<StorageTexture> cachedStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& requested, std::uint32_t mip, std::uint64_t guestBytes, bool written, std::optional<std::pair<std::uint32_t, std::uint32_t>> layers) {
    const auto viewed = ArrayLayerWindow(requested, context.limits.maxImageArrayLayers);
    if (viewed.baseAddress != requested.baseAddress) guestBytes = 0;
    Require(viewed.dimension != TextureDimension::k2DMsaa, "storage image access to a multisampled texture is not implemented");
    RequireNoMultisampleTarget(viewed.baseAddress, guestBytes != 0 ? static_cast<std::size_t>(guestBytes) : 1u, "a storage image access");
    // A single-level 2D view of a depth or stencil plane: the surface's own storage image. Other
    // views over a depth surface's memory (array layers, see SeedStorageFromDepth) use the cache.
    if (auto depth = DepthSurfaceStorage(context, words, viewed, mip)) return depth;
    const auto viewedBytes = guestBytes != 0 ? guestBytes : DescribeSurface(viewed).guestBytes;
    if (viewedBytes >= (1ull << 30u)) {
        static std::atomic<std::uint64_t> large{0};
        const auto count = large.fetch_add(1) + 1;
        if (count <= 5 || (count & (count - 1)) == 0) {
            std::fprintf(stderr, "[gpu] storage view 0x%llx (%ux%u format %u, dimension %d, tile %d, slices %u..%u, mip %u) covers %llu MiB; %llu such views\n", static_cast<unsigned long long>(viewed.baseAddress), viewed.width, viewed.height, viewed.format, static_cast<int>(viewed.dimension), static_cast<int>(viewed.tileMode), viewed.baseArray, viewed.depthOrLastArray, mip, static_cast<unsigned long long>(viewedBytes >> 20u), static_cast<unsigned long long>(count));
            reportGuestAllocation(viewed, viewedBytes);
        }
    }
    if (DepthPlanesStale() || written) static_cast<void>(FlushDepthPlanes(viewed.baseAddress, static_cast<std::size_t>(viewedBytes), written ? DepthPlaneScope::Whole : DepthPlaneScope::Overlapped));
    auto texture = lookupStorageTexture(context, words, viewed, mip, guestBytes, layers);
    if (DepthSurfaceAt(viewed.baseAddress) && DepthSurfaceServesStorage(context, *texture)) SeedStorageFromDepth(context, texture);
    else if (written && DepthSurfaceOverlaps(context.device, viewed.baseAddress, viewedBytes)) {
        const auto storage = WrittenStorageBytes(viewed, mip);
        RetireCoveredDepthSurfaces(context.device, storage.bytes, storage.extents);
    }
    return texture;
}

std::shared_ptr<StorageTexture> lookupStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& viewed, std::uint32_t mip, std::uint64_t guestBytes, std::optional<std::pair<std::uint32_t, std::uint32_t>> layers) {
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    if (disabled) return StorageTexture::Create(context, *context.detiler, viewed, mip);
    static_cast<void>(words);
    if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->BoundKeptBytes();
    const bool profile = LookupOutcomes::Profiled();
    auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto& counters = TextureCounts();
    const auto resource = StorageSurface(context, viewed);
    const StorageKey key{context.device, SurfaceKey(context, resource)};
    auto& cache = StorageTextures();
    std::vector<std::shared_ptr<StorageTexture>> evicted;
    std::shared_ptr<StorageTexture> texture;
    {
        std::lock_guard lock(cache.mutex);
        const CacheHold held(storageCacheHeld);
        if (resource.mipCount > AllocatedLevels(resource)) {
            auto allocated = resource;
            allocated.mipCount = AllocatedLevels(resource);
            if (const auto found = findStorage(cache, {context.device, SurfaceKey(context, allocated)}); found != cache.entries.end()) evicted.push_back(evictStorage(cache, found));
        }
        auto it = findStorage(cache, key);
        if (it != cache.entries.end() && MetadataMoved(*it->texture, resource)) {
            evicted.push_back(evictStorage(cache, it));
            it = cache.entries.end();
        }
        if (it != cache.entries.end()) {
            if (layers) it->texture->RefreshLayers(layers->first, layers->second);
            else it->texture->Refresh();
            cache.entries.splice(cache.entries.begin(), cache.entries, it);
            texture = it->texture;
            counters.storageHits.fetch_add(1, std::memory_order_relaxed);
            if (profile) LookupOutcomes::Add(LookupOutcomes::StorageHit, start);
        } else {
            if (guestBytes == 0) guestBytes = DescribeSurface(resource).guestBytes;
            if (StorageTexture::FlushPending(resource.baseAddress, static_cast<std::size_t>(guestBytes), nullptr, "storage image creation", PublishScope::None) && profile) start = LookupOutcomes::Add(LookupOutcomes::PendingFlush, start);
            std::shared_ptr<StorageTexture> created;
            {
                const CacheHold reclaimable(storageCacheReclaimable);
                created = StorageTexture::Create(context, *context.detiler, resource, mip);
            }
            CachedStorageTexture entry{key, mip, std::move(created)};
            static const bool keepNew = std::getenv("APS5_NO_KEEP_NEW_STORAGE") == nullptr;
            if (auto* recorder = Recorder::Active(); keepNew && recorder != nullptr && GuestMemory::GpuMutex().HeldByThisThread() && recorder->Recording()) recorder->Keep(entry.texture);
            entry.accounted = heldBytes(*entry.texture);
            const auto budget = storageBudget(context, cache.bytes + entry.accounted);
            while (!cache.entries.empty() && cache.bytes + entry.accounted > budget) evicted.push_back(evictStorage(cache, std::prev(cache.entries.end())));
            cache.bytes += entry.accounted;
            texture = entry.texture;
            cache.entries.push_front(std::move(entry));
            cache.index[key] = cache.entries.begin();
            cache.byImage[texture.get()] = cache.entries.begin();
            texture->SetCached(true);
            counters.storageCreated.fetch_add(1, std::memory_order_relaxed);
            if (profile) LookupOutcomes::Add(LookupOutcomes::StorageMade, start);
        }
    }
    for (const auto& image : evicted) {
        std::vector<std::shared_ptr<ShaderResources>> resources;
        SharedResourceCache().RemoveImage(image.get(), &resources);
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->Recording()) {
            if (image.use_count() > 1) recorder->KeepBytes(image.get(), static_cast<std::size_t>(heldBytes(*image)));
            for (auto& resource : resources) {
                resource->KeepStorageImageBytes(*recorder);
                recorder->Keep(std::move(resource));
            }
        }
    }
    if (!evicted.empty()) {
        auto& sampled = Textures();
        std::lock_guard lock(sampled.mutex);
        dropUncachedViews(sampled);
    }
    return texture;
}

}

std::uint64_t TextureCacheBudget(const VkPhysicalDeviceMemoryProperties& memory) {
    const auto heap = largestDeviceLocalHeap(memory);
    return std::max<std::uint64_t>(2048ull << 20u, heap ? memory.memoryHeaps[*heap].size / 4u : 0u);
}

std::uint64_t StorageTextureBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, std::uint64_t sampledBytes, std::uint64_t storageBytes) {
    const auto heap = largestDeviceLocalHeap(memory);
    if (reported == nullptr || !heap || reported->heapBudget[*heap] == 0) return TextureCacheBudget(memory);
    const auto budget = reported->heapBudget[*heap];
    const auto usage = reported->heapUsage[*heap];
    const auto reserved = usage - std::min<std::uint64_t>(usage, sampledBytes + storageBytes) + sampledBytes + budget / 8u;
    const auto cap = std::min<std::uint64_t>(memory.memoryHeaps[*heap].size / 2u, budget / 2u);
    return std::max<std::uint64_t>(512ull << 20u, std::min<std::uint64_t>(cap, budget > reserved ? budget - reserved : 0u));
}

TextureCacheBudgets TextureCacheBudgetsFor(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, const TextureMemoryUse& use, std::uint64_t previousTarget) {
    constexpr std::uint64_t MiB = 1ull << 20u;
    const auto fixed = TextureCacheBudget(memory);
    const auto heap = largestDeviceLocalHeap(memory);
    if (reported == nullptr || !heap || reported->heapBudget[*heap] == 0) return {fixed, fixed + fixed / 4u, fixed, 0, fixed + fixed / 4u};
    auto limit = std::min<std::uint64_t>(reported->heapBudget[*heap], memory.memoryHeaps[*heap].size);
    if (const auto forced = VramBudget::ForcedLimit(); forced != 0) limit = std::min<std::uint64_t>(limit, forced);
    const auto headroom = VramBudget::ForcedHeadroom() != 0 ? VramBudget::ForcedHeadroom() : VramBudget::HeadroomFor(limit);
    const auto usage = reported->heapUsage[*heap];
    const auto other = usage - std::min<std::uint64_t>(usage, use.sampled + use.storageCached);
    const auto reserved = headroom + other;
    auto target = std::max<std::uint64_t>(limit > reserved ? limit - reserved : 0u, std::min<std::uint64_t>(limit / 16u, 512 * MiB));
    if (previousTarget != 0 && (target > previousTarget ? target - previousTarget : previousTarget - target) <= std::max<std::uint64_t>(previousTarget / 64u, 64 * MiB)) target = previousTarget;
    const auto storage = std::min<std::uint64_t>(fixed, std::max<std::uint64_t>(target / 2u, 256 * MiB));
    const auto outside = use.sampled - std::min(use.sampled, use.sampledCached + use.overflow);
    const auto taken = std::max<std::uint64_t>(storage, use.storageCached) + outside;
    const auto ceiling = std::max<std::uint64_t>(target > taken ? target - taken : 0u, std::min<std::uint64_t>(target / 8u, 128 * MiB));
    std::uint64_t overflow = 0;
    std::optional<std::uint32_t> system;
    for (std::uint32_t i = 0; i < std::min<std::uint32_t>(memory.memoryHeapCount, VK_MAX_MEMORY_HEAPS); ++i) {
        if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0 && (!system || memory.memoryHeaps[i].size > memory.memoryHeaps[*system].size)) system = i;
    }
    if (system && reported->heapBudget[*system] != 0) {
        const auto systemBudget = reported->heapBudget[*system];
        const auto reserve = reported->heapUsage[*system] + systemBudget / 16u;
        overflow = std::max<std::uint64_t>(systemBudget > reserve ? systemBudget - reserve : 0u, 1u);
    }
    return {ceiling - ceiling / 5u, ceiling, storage, target, ceiling + outside, overflow};
}

bool SampledBudgetReportDue(std::uint64_t reported, std::uint64_t budget, std::chrono::steady_clock::duration sinceReport) {
    const auto change = budget > reported ? budget - reported : reported - budget;
    return change > reported / 10u && sinceReport >= std::chrono::seconds(10);
}

void AppendTextureMemoryStats(std::string& line) {
    std::uint64_t snapshotBytes = 0, coveredBytes = 0, deviceBytes = 0, cachedSnapshots = 0;
    std::size_t entries = 0, snapshots = 0, views = 0;
    std::size_t pinned = 0;
    decltype(Textures().totals) totals;
    {
        auto& sampled = Textures();
        std::lock_guard lock(sampled.mutex);
        entries = sampled.entries.size();
        deviceBytes = sampled.bytes;
        cachedSnapshots = sampled.snapshotBytes;
        totals = sampled.totals;
        for (const auto& entry : sampled.entries) {
            snapshotBytes += entry.hashes.capacity() * sizeof(ContentHash::Hash128);
            coveredBytes += entry.bytes;
            snapshots += entry.bytes != 0;
            views += entry.source != nullptr;
            pinned += entry.texture->ResourcePins() != 0;
        }
    }
    // Host bytes held to validate snapshots (block hashes), and the guest bytes they stand for.
    MemoryStats::AppendMiB(line, "tex_sampled_host_snapshots", snapshotBytes, snapshots);
    MemoryStats::AppendMiB(line, "tex_sampled_snapshot_guest", coveredBytes);
    MemoryStats::AppendMiB(line, "tex_sampled_device", deviceBytes, entries);
    MemoryStats::AppendCount(line, "tex_sampled_views", views);
    // Sampled images alive outside the cache's accounting (refused or evicted ones still in
    // recorded work or builds), the cached ones resource-cache sets pin, the budgets of both
    // caches and their share of the VRAM, and what the eviction and the out-of-memory reclaim did.
    const auto sampledAll = SampledTextureMemory();
    MemoryStats::AppendMiB(line, "tex_sampled_outside", sampledAll - std::min(sampledAll, cachedSnapshots));
    MemoryStats::AppendCount(line, "tex_sampled_pinned", pinned);
    TextureCacheBudgets current;
    {
        auto& state = Budgets();
        std::lock_guard lock(state.mutex);
        current = state.budgets;
    }
    MemoryStats::AppendMiB(line, "tex_sampled_budget", current.sampled);
    MemoryStats::AppendMiB(line, "tex_sampled_ceiling", current.ceiling);
    MemoryStats::AppendMiB(line, "tex_storage_budget", current.storage);
    MemoryStats::AppendMiB(line, "tex_vram_share", current.target);
    MemoryStats::AppendMiB(line, "tex_sampled_evicted", totals.evictedBytes, totals.evicted);
    MemoryStats::AppendMiB(line, "tex_sampled_refused", totals.refusedBytes, totals.refused);
    auto& atomics = SampledAtomics();
    MemoryStats::AppendCount(line, "rescache_sets_dropped", atomics.droppedSets.load(std::memory_order_relaxed));
    MemoryStats::AppendCount(line, "rescache_sets_refused", atomics.refusedSets.load(std::memory_order_relaxed));
    MemoryStats::AppendMiB(line, "tex_sampled_host", SampledTextureHostMemory());
    MemoryStats::AppendMiB(line, "tex_sampled_spilled", totals.spilledBytes, totals.spilled);
    MemoryStats::AppendCount(line, "tex_oneoff_reuses", totals.oneOffReuses);
    MemoryStats::AppendMiB(line, "tex_sampled_share", current.share);
    MemoryStats::AppendMiB(line, "tex_overflow_live", SampledTextureOverflowMemory());
    MemoryStats::AppendMiB(line, "tex_overflow_budget", current.overflow);
    {
        auto& sampled = Textures();
        const SampledCacheLock lock(sampled);
        MemoryStats::AppendMiB(line, "tex_overflow_kept", sampled.overflowKept, sampled.overflowLru.size());
        MemoryStats::AppendCount(line, "tex_snapshots_made", sampled.totals.made);
        MemoryStats::AppendCount(line, "tex_spilled_reuses", sampled.totals.spilledReuses);
        MemoryStats::AppendMiB(line, "tex_overflow_dropped", sampled.totals.overflowDroppedBytes, sampled.totals.overflowDropped);
        MemoryStats::AppendCount(line, "tex_overflow_waits", sampled.totals.overflowWaits);
    }
    MemoryStats::AppendMiB(line, "tex_oom_reclaimed", atomics.reclaimedBytes.load(std::memory_order_relaxed), atomics.reclaims.load(std::memory_order_relaxed));
    std::size_t storageEntries = 0;
    {
        auto& storage = StorageTextures();
        std::lock_guard lock(storage.mutex);
        storageEntries = storage.entries.size();
    }
    MemoryStats::AppendMiB(line, "tex_storage_cache", storageCacheBytes(), storageEntries);
}

std::optional<std::uint32_t> LargestDeviceLocalHeap(const VkPhysicalDeviceMemoryProperties& memory) {
    return largestDeviceLocalHeap(memory);
}

std::uint64_t StorageTextureCacheBudget(const Context& context) {
    return storageBudget(context);
}

void OverrideCacheBudgetForTest(std::uint64_t bytes) {
    budgetOverride.store(bytes, std::memory_order_relaxed);
}

TextureCacheUse TextureCacheUsage() {
    TextureCacheUse use;
    {
        const DropReleasedDependents dropAfterUnlock;
        auto& sampled = Textures();
        const SampledCacheLock lock(sampled);
        dropUncachedViews(sampled);
        use.sampledEntries = sampled.entries.size();
        use.sampledBytes = sampled.bytes;
    }
    use.storageBytes = storageCacheBytes();
    return use;
}

std::uint64_t SampledTextureCacheBudget(const Context& context) {
    return currentBudgets(context).sampled;
}

TextureCacheBudgets CurrentTextureCacheBudgets(const Context& context) {
    return currentBudgets(context);
}

SampledCacheCounts SampledCacheCounters() {
    SampledCacheCounts counts;
    {
        auto& cache = Textures();
        const SampledCacheLock lock(cache);
        counts.evicted = cache.totals.evicted;
        counts.evictedBytes = cache.totals.evictedBytes;
        counts.refused = cache.totals.refused;
        counts.refusedBytes = cache.totals.refusedBytes;
        counts.spilled = cache.totals.spilled;
        counts.spilledBytes = cache.totals.spilledBytes;
        counts.oneOffReuses = cache.totals.oneOffReuses;
        counts.made = cache.totals.made;
        counts.spilledReuses = cache.totals.spilledReuses;
        counts.overflowDropped = cache.totals.overflowDropped;
        counts.overflowWaits = cache.totals.overflowWaits;
    }
    auto& atomics = SampledAtomics();
    counts.droppedSets = atomics.droppedSets.load(std::memory_order_relaxed);
    counts.refusedSets = atomics.refusedSets.load(std::memory_order_relaxed);
    counts.reclaims = atomics.reclaims.load(std::memory_order_relaxed);
    counts.reclaimedBytes = atomics.reclaimedBytes.load(std::memory_order_relaxed);
    return counts;
}

namespace {

VkDeviceSize reclaimSampledTextures(VkDevice device, VkDeviceSize wanted) {
    constexpr VkDeviceSize MiB = VkDeviceSize{1} << 20u;
    const auto goal = std::max<VkDeviceSize>(wanted * 4u, 256 * MiB);
    VkDeviceSize freed = 0;
    std::size_t evicted = 0;
    std::vector<std::shared_ptr<Texture>> doomed;
    std::vector<std::shared_ptr<Texture>> released;
    {
        auto& cache = Textures();
        const SampledCacheLock lock(cache);
        const auto retired = retiredSerial();
        for (auto it = cache.entries.end(); it != cache.entries.begin() && freed < goal;) {
            const auto current = std::prev(it);
            if (current->key.device != device || current->accounted == 0 || textureInUse(*current, retired)) {
                it = current;
                continue;
            }
            freed += current->accounted;
            ++evicted;
            countEvicted(cache, *current, EvictionPath::Reclaim);
            doomed.push_back(current->texture);
            eraseTexture(cache, current);
        }
        const auto keptBefore = cache.overflowKept;
        evicted += trimOverflow(cache, freed < goal ? goal - freed : 0u, doomed);
        freed += keptBefore - cache.overflowKept;
        released.swap(cache.released);
    }
    std::size_t sets = 0;
    if (!released.empty()) {
        std::unordered_set<const Texture*> set;
        for (const auto& texture : released) set.insert(texture.get());
        std::vector<std::shared_ptr<ShaderResources>> resources;
        SharedResourceCache().RemoveTextures(set, &resources);
        sets = resources.size();
        keepUntilRetired(resources);
    }
    keepUntilRetired(released);
    // The evicted textures themselves go now (their memory is what the allocation waits for): no
    // holder is left (a resource set or batch binding one holds it) and the newest batch that used
    // one retired (every lookup, hit, revalidation and table reuse stamps the use's serial; a set
    // binding a texture outside the cache is never reused without a lookup).
    doomed.clear();
    if (freed != 0) {
        auto& atomics = SampledAtomics();
        atomics.reclaims.fetch_add(1, std::memory_order_relaxed);
        atomics.reclaimedBytes.fetch_add(freed, std::memory_order_relaxed);
    }
    static std::atomic<std::uint64_t> reports{0};
    if (freed != 0 || reports.fetch_add(1, std::memory_order_relaxed) < 8) std::fprintf(stderr, "[texture-cache] out of device memory (%llu bytes wanted): evicted %zu sampled textures no recorded work uses (%.1f MiB) and %zu resource sets binding them\n", static_cast<unsigned long long>(wanted), evicted, static_cast<double>(freed) / (1u << 20u), sets);
    return freed;
}

[[maybe_unused]] const bool reclaimerRegistered = (SetTextureMemoryReclaimer(&reclaimSampledTextures), true);

}

std::shared_ptr<Texture> CachedSampledTexture(const Context& context, std::span<const std::uint32_t> words, VkImageViewType multisampleViewType) {
    const auto resource = DecodeTextureResource(words);
    return cachedTexture(context, words, resource, ViewComponents(resource), 0, false, multisampleViewType);
}

std::shared_ptr<Texture> CachedTableTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint64_t guestBytes, bool depthCompare) {
    CompletePendingCpuWrites(context, resource.baseAddress, static_cast<std::size_t>(guestBytes));
    if (resource.dccAddress != 0) CompletePendingCpuWrites(context, resource.dccAddress, DccKeyBytes(guestBytes));
    return cachedTexture(context, words, resource, ViewComponents(resource), guestBytes, depthCompare);
}

bool SampledTexturesShareEntry(std::span<const std::uint32_t> first, std::span<const std::uint32_t> second) {
    Require(first.size() == 8 && second.size() == 8, "sampled texture keys take eight descriptor words");
    const VkComponentMapping components{VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    const auto firstKey = MakeTextureKey(VK_NULL_HANDLE, first, components);
    const auto secondKey = MakeTextureKey(VK_NULL_HANDLE, second, components);
    return firstKey == secondKey && TextureKeyHash{}(firstKey) == TextureKeyHash{}(secondKey);
}

void FlushCachedTextures(VkDevice device) {
    Require(device != VK_NULL_HANDLE, "cannot flush textures without a Vulkan device");
    GuestMemory::AssertGpuLockHeld("FlushCachedTextures");
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    const CacheHold held(storageCacheHeld);
    for (const auto& entry : cache.entries) {
        if (entry.key.device == device) entry.texture->Flush();
    }
}

void ClearCachedTextures(VkDevice device) {
    Require(device != VK_NULL_HANDLE, "cannot clear textures without a Vulkan device");
    auto& sampled = Textures();
    {
        const SampledCacheLock lock(sampled);
        for (auto it = sampled.entries.begin(); it != sampled.entries.end();) {
            if (it->key.device == device) eraseTexture(sampled, it++);
            else ++it;
        }
        for (auto it = sampled.oneOffs.begin(); it != sampled.oneOffs.end();) {
            if (it->first.device == device) eraseOneOff(sampled, it++);
            else ++it;
        }
    }
    dropReleasedDependents();
    auto& storage = StorageTextures();
    std::vector<std::shared_ptr<StorageTexture>> cleared;
    {
        std::lock_guard lock(storage.mutex);
        const CacheHold held(storageCacheHeld);
        for (auto it = storage.entries.begin(); it != storage.entries.end();) {
            if (it->key.device != device) {
                ++it;
                continue;
            }
            cleared.push_back(it->texture);
            it->texture->SetCached(false);
            noteUncached(it->texture);
            storage.bytes -= it->accounted;
            storage.index.erase(it->key);
            storage.byImage.erase(it->texture.get());
            it = storage.entries.erase(it);
        }
    }
    for (const auto& image : cleared) {
        std::vector<std::shared_ptr<ShaderResources>> resources;
        SharedResourceCache().RemoveImage(image.get(), &resources);
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->Recording()) {
            for (auto& resource : resources) {
                resource->KeepStorageImageBytes(*recorder);
                recorder->Keep(std::move(resource));
            }
        }
    }
}

std::uint64_t ReleaseTextureCaches(std::uint64_t bytes, bool newImage) {
    thread_local bool releasing = false;
    if (releasing) return 0;
    releasing = true;
    struct Done {
        bool& flag;
        ~Done() { flag = false; }
    } done{releasing};
    std::uint64_t freed = 0;
    std::size_t sampledEntries = 0;
    std::size_t storageEntries = 0;
    std::size_t storageKept = 0;
    bool sampledSkipped = false;
    bool storageSkipped = false;
    {
        auto& cache = Textures();
        std::unique_lock lock(cache.mutex, std::defer_lock);
        const bool inherited = sampledCacheHeld != 0;
        if ((inherited && (!newImage || sampledCacheReclaimable == 0)) || (!inherited && !lock.try_lock())) {
            sampledSkipped = true;
        } else {
            for (auto it = cache.entries.end(); it != cache.entries.begin() && freed < bytes;) {
                --it;
                if (it->texture.use_count() != 1 || (it->source != nullptr && it->source->Dirty())) continue;
                const auto allocated = it->texture->AllocationBytes();
                const auto sourceBytes = it->source != nullptr ? it->source->AllocationBytes() : 0u;
                const std::weak_ptr<StorageTexture> source = it->source;
                const auto next = std::next(it);
                eraseTexture(cache, it);
                ++cache.counts.evicted;
                ++sampledEntries;
                freed += allocated;
                if (source.expired()) freed += sourceBytes;
                it = next;
            }
        }
    }
    if (freed < bytes) {
        auto& cache = StorageTextures();
        std::unique_lock<GuestMemory::GpuMutexType> gpu(GuestMemory::GpuMutex(), std::defer_lock);
        const bool inherited = storageCacheHeld != 0;
        if ((inherited && (!newImage || storageCacheReclaimable == 0)) || !gpu.try_lock()) {
            storageSkipped = true;
        } else {
            try {
                while (freed < bytes) {
                    std::shared_ptr<StorageTexture> image;
                    {
                        std::unique_lock lock(cache.mutex, std::defer_lock);
                        if (!inherited && !lock.try_lock()) {
                            storageSkipped = true;
                            break;
                        }
                        storageKept = 0;
                        for (auto it = cache.entries.end(); it != cache.entries.begin();) {
                            --it;
                            if (it->texture->Dirty()) {
                                ++storageKept;
                                continue;
                            }
                            image = evictStorage(cache, it);
                            break;
                        }
                        if (image == nullptr) break;
                    }
                    std::vector<std::shared_ptr<ShaderResources>> resources;
                    const bool removed = SharedResourceCache().TryRemoveImage(image.get(), &resources);
                    if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->Recording()) {
                        for (auto& resource : resources) recorder->Keep(std::move(resource));
                    }
                    resources.clear();
                    ++storageEntries;
                    if (image.use_count() == 1) freed += image->AllocationBytes();
                    image.reset();
                    if (!removed) {
                        storageSkipped = true;
                        break;
                    }
                }
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[gpu] texture cache release stopped: %s\n", error.what());
            }
        }
    }
    std::fprintf(stderr, "[gpu] texture caches released %llu MiB of the %llu MiB asked for: %zu sampled entries%s, %zu storage images%s, %zu with pending results kept\n", static_cast<unsigned long long>(freed >> 20u), static_cast<unsigned long long>(bytes >> 20u), sampledEntries, sampledSkipped ? " (cache busy, skipped)" : "", storageEntries, storageSkipped ? " (cache or device busy, skipped)" : "", storageKept);
    return freed;
}

bool StorageImageCached(const Context& context, const StorageTexture* image) {
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    const auto it = findStorageByImage(cache, context.device, image);
    if (it == cache.entries.end()) return false;
    cache.entries.splice(cache.entries.begin(), cache.entries, it);
    return true;
}

namespace {

// StorageImageCached for several images under one acquisition of the cache mutex.
bool StorageImagesCached(const Context& context, std::span<const StorageTexture* const> images) {
    if (images.empty()) return true;
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    for (const auto* image : images) {
        const auto it = findStorageByImage(cache, context.device, image);
        if (it == cache.entries.end()) return false;
        cache.entries.splice(cache.entries.begin(), cache.entries, it);
    }
    return true;
}

}

std::shared_ptr<StorageTexture> CachedStorageSurface(const Context& context, const GuestTextureResource& resource) {
    return cachedStorageTexture(context, {}, resource, 0);
}

std::shared_ptr<StorageTexture> CachedStorageLayers(const Context& context, const GuestTextureResource& resource, std::uint32_t firstLayer, std::uint32_t layerCount) {
    return cachedStorageTexture(context, {}, resource, 0, 0, false, std::pair{firstLayer, layerCount});
}

bool StorageImageServesKeys(const StorageTexture& image, std::uint64_t dccAddress) {
    GuestTextureResource resource{};
    resource.dccAddress = dccAddress;
    return !MetadataMoved(image, resource);
}

namespace {

const char* roleName(ShaderRecompiler::DescriptorRole role) {
    switch (role) {
        case ShaderRecompiler::DescriptorRole::GuestBuffers: return "GuestBuffers";
        case ShaderRecompiler::DescriptorRole::GuestImages: return "GuestImages";
        case ShaderRecompiler::DescriptorRole::GuestSamplers: return "GuestSamplers";
        case ShaderRecompiler::DescriptorRole::Gds: return "Gds";
        case ShaderRecompiler::DescriptorRole::BdaPagetable: return "BdaPagetable";
        case ShaderRecompiler::DescriptorRole::FaultBuffer: return "FaultBuffer";
        case ShaderRecompiler::DescriptorRole::FlattenedSrt: return "FlattenedSrt";
        case ShaderRecompiler::DescriptorRole::ShaderData: return "ShaderData";
        case ShaderRecompiler::DescriptorRole::ImageTable: return "ImageTable";
        case ShaderRecompiler::DescriptorRole::SamplerTable: return "SamplerTable";
        case ShaderRecompiler::DescriptorRole::ImageTableMap: return "ImageTableMap";
    }
    throw std::runtime_error("AGC graphics: unknown descriptor role");
}

const char* kindName(ShaderRecompiler::DescriptorKind kind) {
    switch (kind) {
        case ShaderRecompiler::DescriptorKind::UniformBuffer: return "UniformBuffer";
        case ShaderRecompiler::DescriptorKind::StorageBuffer: return "StorageBuffer";
        case ShaderRecompiler::DescriptorKind::UniformTexelBuffer: return "UniformTexelBuffer";
        case ShaderRecompiler::DescriptorKind::StorageTexelBuffer: return "StorageTexelBuffer";
        case ShaderRecompiler::DescriptorKind::SampledImage: return "SampledImage";
        case ShaderRecompiler::DescriptorKind::StorageImage: return "StorageImage";
        case ShaderRecompiler::DescriptorKind::Sampler: return "Sampler";
    }
    throw std::runtime_error("AGC graphics: unknown descriptor kind");
}

// Consecutive identical storage descriptors address successive mips of one texture (dynamic-mip
// storage writes), so the second and later ones share the first one's image lookup.
// APS5_NO_STORAGE_DEDUPE=1 looks each one up (and refreshes the image) separately as before.
bool StorageDedupeEnabled() {
    static const bool disabled = std::getenv("APS5_NO_STORAGE_DEDUPE") != nullptr;
    return !disabled;
}

// Whether storage element `element` of `binding` repeats the previous element's eight words.
bool SameAsPreviousStorageElement(const ShaderRecompiler::DescriptorBinding& binding, std::uint32_t element) {
    if (element == 0) return false;
    const auto words = binding.guestDescriptor.begin() + static_cast<std::size_t>(element) * 8u;
    return std::equal(words, words + 8, words - 8);
}

}

ShaderResources::ShaderResources(const Context& context, const ShaderRecompiler::RecompileResult& vertex, const ShaderRecompiler::RecompileResult& fragment, const ColorTarget& target, std::uint64_t indexAddress, std::size_t indexBytes) : ShaderResources(context, std::array<CompiledShader, 2>{{{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, {ShaderRecompiler::ShaderStage::Fragment, &fragment, static_cast<std::uint32_t>(vertex.pushConstants.size())}}}, target, 1, indexAddress, indexBytes) {}

void ShaderResources::KeepStorageImageBytes(Recorder& recorder) const {
    const auto keep = [&](const std::shared_ptr<StorageTexture>& image) {
        if (image != nullptr && !image->Cached()) recorder.KeepBytes(image.get(), static_cast<std::size_t>(image->BudgetBytes()));
    };
    for (const auto& image : storageTextures) keep(image);
    for (const auto& texture : textures) {
        if (texture != nullptr) keep(texture->SharedStorageSource());
    }
}

ShaderResources::ShaderResources(const Context& context, std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes, std::span<const GuestMemorySnapshot> snapshots) : context(context), guestMemory(context) {
    drawBuild = true;
    prepareAddressBindings(shaders, snapshots);
    build(shaders, &target, colorAttachments, indexAddress, indexBytes);
}

ShaderResources::ShaderResources(DeferredDrawBuild, const Context& context, std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes) : context(context), guestMemory(context) {
    drawBuild = true;
    deferredDraw = true;
    unlockedPrepare = true;
    Require(!StagesUseAddressTables(shaders), "a deferred draw build cannot map address tables");
    guestMemory.DeferHostCopies();
    prepareAddressBindings(shaders, {});
    buildPrepare(shaders, &target, colorAttachments, indexAddress, indexBytes);
}

void ShaderResources::CompleteDraw() {
    Require(deferredDraw && !completed, "shader resources were not prepared for a deferred draw or were already completed");
    buildComplete();
}

bool ShaderResources::StagesUseAddressTables(std::span<const CompiledShader> shaders) {
    return std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program != nullptr && std::any_of(shader.program->bindings.begin(), shader.program->bindings.end(), [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; }); });
}

ShaderResources::ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots) : ShaderResources(context, compute, snapshots, false) {}

namespace {

// Whether a compute stage maps registered guest memory through BDA tables (see prepareAddressBindings).
bool UsesAddressTables(const CompiledShader& compute) {
    Require(compute.program != nullptr, "missing compiled shader");
    return std::any_of(compute.program->bindings.begin(), compute.program->bindings.end(), [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; });
}

}

ShaderResources::ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots, bool deferred, std::uint64_t dispatchThreads) : context(context), guestMemory(context), dispatchThreads(dispatchThreads), deferredCompute(compute), deferredSnapshots(snapshots) {
    Require(compute.stage == ShaderRecompiler::ShaderStage::Compute, "compute resources require a compute shader");
    // Every use of a compute build is a recorded dispatch that calls MarkGpuWrites, which staged
    // buffers need (a synchronous draw's use would not).
    guestMemory.AllowDeviceStaging();
    // BDA overlays (GuestBufferMemory::resolveOverlays) copy each written element in and back per
    // dispatch: measured on the request bench, the 1024-thread cloth solver gains 64 % GPU time per
    // dispatch, the 512-thread one 4 %, while the 64-thread one loses 9 % (its copies cost more than its
    // PCIe accesses). APS5_BDA_STAGING_MIN_THREADS (default 512) is the smallest dispatch (threads in
    // all groups) that takes them; an indirect dispatch (unknown size) does.
    static const std::uint64_t overlayMinThreads = [] {
        const char* value = std::getenv("APS5_BDA_STAGING_MIN_THREADS");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 512ull;
    }();
    if (dispatchThreads != 0 && dispatchThreads < overlayMinThreads) guestMemory.DisallowAddressStaging();
    const std::span<const CompiledShader> shaders(&deferredCompute, 1);
    if (!deferred) {
        prepareAddressBindings(shaders, snapshots);
        build(shaders, nullptr, 0, 0, 0);
        forgetDeferredInputs();
        return;
    }
    // Address-based shaders pin and mirror registered memory in prepareAddressBindings (reconciling
    // imports retires buffers to the recorder, refreshing mirrors waits for recorded work): that is
    // device-lock work, so their whole build waits for Complete(). Without tables the call only
    // checks the bindings.
    lockedBuild = UsesAddressTables(compute);
    if (lockedBuild) return;
    unlockedPrepare = true;
    prepareAddressBindings(shaders, snapshots);
    buildPrepare(shaders, nullptr, 0, 0, 0);
}

void ShaderResources::Complete() {
    Require(!completed, "shader resources were already completed");
    const std::span<const CompiledShader> shaders(&deferredCompute, 1);
    if (lockedBuild) {
        prepareAddressBindings(shaders, deferredSnapshots);
        buildPrepare(shaders, nullptr, 0, 0, 0);
    }
    buildComplete();
    forgetDeferredInputs();
}

void ShaderResources::forgetDeferredInputs() {
    // The compiled shader and the captured regions belong to the dispatch, which returns while this
    // object lives on in the resource cache and the recorder: nothing may reach for them after the
    // build, so they are dropped rather than left dangling.
    deferredCompute = {};
    deferredSnapshots = {};
}

void ShaderResources::build(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes) {
    buildPrepare(shaders, target, colorAttachments, indexAddress, indexBytes);
    buildComplete();
}

namespace {

VkDescriptorSetLayout CreateSetLayout(const Context& context, std::span<const VkDescriptorSetLayoutBinding> bindings, bool updateAfterBind, std::span<const VkDescriptorBindingFlags> flags) {
    Require(flags.size() == bindings.size(), "descriptor binding flag count differs from layout");
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bindingFlags.bindingCount = static_cast<std::uint32_t>(flags.size());
    bindingFlags.pBindingFlags = flags.data();
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.pNext = std::ranges::any_of(flags, [](auto value) { return value != 0; }) ? &bindingFlags : nullptr;
    info.flags = updateAfterBind ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : 0;
    info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    info.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &info, nullptr, &layout), "vkCreateDescriptorSetLayout");
    return layout;
}

std::vector<VkDescriptorPoolSize> PoolSizes(std::span<const VkDescriptorSetLayoutBinding> bindings) {
    std::map<VkDescriptorType, std::uint32_t> counts;
    for (const auto& binding : bindings) counts[binding.descriptorType] += binding.descriptorCount;
    std::vector<VkDescriptorPoolSize> sizes;
    for (const auto& [type, count] : counts) sizes.push_back({type, count});
    return sizes;
}

VkDescriptorPool CreateSetPool(const Context& context, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind) {
    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.flags = updateAfterBind ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0;
    info.maxSets = 1;
    info.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    info.pPoolSizes = sizes.data();
    VkDescriptorPool pool = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &info, nullptr, &pool), "vkCreateDescriptorPool");
    return pool;
}

// APS5_PROFILE_DRAW: the [resources] phase totals. Each thread accumulates its builds' phases in
// arrays of its own and merges them into the shared totals every 1000 of its builds (and when it
// ends), so no build takes the shared mutex per phase; the totals lag by up to 999 builds per
// worker.
constexpr std::size_t BuildPhaseCount = static_cast<std::size_t>(ShaderResources::BuildPhase::Count);
constexpr std::array<const char*, BuildPhaseCount> BuildPhaseNames{"bindings", "precollect", "guest memory upload", "descriptors", "stage A", "images", "bda", "stage B"};

struct BuildProfile {
    AgcDriver::Mutex mutex;
    std::array<double, BuildPhaseCount> ms{};
    std::uint64_t builds = 0;
};

BuildProfile& Builds() {
    static BuildProfile profile;
    return profile;
}

bool BuildProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

struct ThreadBuildProfile {
    std::array<double, BuildPhaseCount> ms{};
    std::uint64_t builds = 0;
    ~ThreadBuildProfile() { merge(); }
    void merge() {
        auto& profile = Builds();
        std::lock_guard lock(profile.mutex);
        for (std::size_t i = 0; i < BuildPhaseCount; ++i) profile.ms[i] += ms[i];
        profile.builds += builds;
        ms = {};
        if (builds == 0) return;
        builds = 0;
        std::string report;
        for (std::size_t i = 0; i < BuildPhaseCount; ++i) report += " " + std::string(BuildPhaseNames[i]) + "=" + std::to_string(static_cast<long long>(profile.ms[i])) + "ms";
        AgcDriver::ProfilePrint_nid_no_patch("[resources] %llu builds, phase totals:%s\n", static_cast<unsigned long long>(profile.builds), report.c_str());
    }
};

ThreadBuildProfile& ThreadBuilds() {
    thread_local ThreadBuildProfile* profileSlot = nullptr;
    auto& profile = ShaderRecompiler::ThreadOwned(profileSlot);
    return profile;
}

void addBuildPhase(ShaderResources::BuildPhase which, double ms) {
    ThreadBuilds().ms[static_cast<std::size_t>(which)] += ms;
}

struct CompleteImageProfile {
    AgcDriver::Mutex mutex;
    std::array<std::array<std::uint64_t, LookupOutcomes::Count>, 2> counts{};
    std::array<std::array<double, LookupOutcomes::Count>, 2> ms{};
    std::array<std::uint64_t, 2> builds{}, deferred{};
    std::array<double, 2> deferredMs{}, storageTablesMs{}, imageTablesMs{}, depthMs{};
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

CompleteImageProfile& CompleteImages() {
    static CompleteImageProfile profile;
    return profile;
}

void noteCompleteImages(const LookupOutcomes& before, std::size_t deferred, double deferredMs, double storageTablesMs, double imageTablesMs, double depthMs) {
    const auto& after = ThreadLookupOutcomes();
    const std::size_t side = GuestMemory::GpuLockThreadTag() == 0xffffffffu ? 0 : 1;
    auto& profile = CompleteImages();
    std::lock_guard lock(profile.mutex);
    for (std::size_t kind = 0; kind < LookupOutcomes::Count; ++kind) {
        if (after.counts[kind] >= before.counts[kind]) profile.counts[side][kind] += after.counts[kind] - before.counts[kind];
        if (after.ms[kind] >= before.ms[kind]) profile.ms[side][kind] += after.ms[kind] - before.ms[kind];
    }
    ++profile.builds[side];
    profile.deferred[side] += deferred;
    profile.deferredMs[side] += deferredMs;
    profile.storageTablesMs[side] += storageTablesMs;
    profile.imageTablesMs[side] += imageTablesMs;
    profile.depthMs[side] += depthMs;
    const auto now = std::chrono::steady_clock::now();
    if (now - profile.lastReport < std::chrono::seconds(10)) return;
    profile.lastReport = now;
    static constexpr const char* sides[2] = {"untagged threads (draw committer, presenter)", "queue workers"};
    for (std::size_t which = 0; which < 2; ++which) {
        std::string report;
        char text[160];
        for (std::size_t kind = 0; kind < LookupOutcomes::Count; ++kind) {
            if (profile.counts[which][kind] == 0) continue;
            std::snprintf(text, sizeof(text), " %s %llux%.1fms", LookupOutcomes::Name(static_cast<LookupOutcomes::Kind>(kind)), static_cast<unsigned long long>(profile.counts[which][kind]), profile.ms[which][kind]);
            report += text;
        }
        AgcDriver::ProfilePrint_nid_no_patch("[complete-images] (10 s) %s: %llu stage-B builds; deferred image bindings %llu in %.1f ms, storage image tables %.1f ms, image tables %.1f ms, depth-write check %.1f ms; lookups by outcome (count x ms, the refresh/upload/scan/flush rows lie inside the sampled and storage rows):%s\n", sides[which], static_cast<unsigned long long>(profile.builds[which]), static_cast<unsigned long long>(profile.deferred[which]), profile.deferredMs[which], profile.storageTablesMs[which], profile.imageTablesMs[which], profile.depthMs[which], report.c_str());
        profile.counts[which] = {};
        profile.ms[which] = {};
        profile.builds[which] = profile.deferred[which] = 0;
        profile.deferredMs[which] = profile.storageTablesMs[which] = profile.imageTablesMs[which] = profile.depthMs[which] = 0;
    }
}

// APS5_PROFILE_DRAW: guest buffer elements bound by descriptor (addGuestBuffer) and how many the
// recompiler proved read-only, plus the pending-write notes their uses skipped (MarkGpuWrites),
// printed as [buffers] every 10 s from MarkGpuWrites. Cumulative.
struct BufferWriteCounters {
    std::atomic<std::uint64_t> elements{0};
    std::atomic<std::uint64_t> readOnly{0};
    std::atomic<std::uint64_t> notesSkipped{0};
    std::atomic<std::int64_t> lastReport{0};
};

BufferWriteCounters& BufferWrites() {
    static BufferWriteCounters counters;
    return counters;
}

}

double ShaderResources::phase(BuildPhase which) {
    if (!BuildProfiled()) return 0.0;
    const auto now = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration<double, std::milli>(now - phaseStart).count();
    addBuildPhase(which, ms);
    const auto waited = Recorder::ThreadWaitedMs();
    if (ms > (drawBuild ? 5.0 : 50.0)) {
        const auto tag = GuestMemory::CurrentPacket();
        const auto storage = !storageTextures.empty() && storageTextures.front() != nullptr ? storageTextures.front()->Descriptor().baseAddress : 0ull;
        AgcDriver::ProfilePrint_nid_no_patch("[resources] %s took %.0f ms (%zu textures, %zu storage images, %zu buffers, bda %d; %s, queue 0x%x packet 0x%x, GPU waits %.0f ms; storage image 0x%llx)\n", BuildPhaseNames[static_cast<std::size_t>(which)], ms, textures.size(), storageTextures.size(), allocations.size(), usesBda ? 1 : 0, drawBuild ? "draw" : "dispatch", tag.queue, tag.opcode & 0xffffu, waited - phaseWaited, static_cast<unsigned long long>(storage));
    }
    phaseWaited = waited;
    phaseStart = now;
    return ms;
}

namespace {

std::uint32_t IrStage(ShaderRecompiler::ShaderStage stage) {
    using Ir = ShaderRecompiler::IrShaderStage;
    switch (stage) {
        case ShaderRecompiler::ShaderStage::Compute: return static_cast<std::uint32_t>(Ir::Compute);
        case ShaderRecompiler::ShaderStage::Vertex: return static_cast<std::uint32_t>(Ir::Vertex);
        case ShaderRecompiler::ShaderStage::Fragment: return static_cast<std::uint32_t>(Ir::Pixel);
        case ShaderRecompiler::ShaderStage::Mesh: return static_cast<std::uint32_t>(Ir::Mesh);
        case ShaderRecompiler::ShaderStage::Local: return static_cast<std::uint32_t>(Ir::Local);
        case ShaderRecompiler::ShaderStage::TessellationControl: return static_cast<std::uint32_t>(Ir::TessellationControl);
        case ShaderRecompiler::ShaderStage::TessellationEvaluation: return static_cast<std::uint32_t>(Ir::TessellationEvaluation);
        case ShaderRecompiler::ShaderStage::Geometry: break;
    }
    return static_cast<std::uint32_t>(Ir::Unknown);
}

bool TableReuseEnabled() {
    static const bool disabled = std::getenv("APS5_NO_TABLE_REUSE") != nullptr;
    return !disabled;
}

bool TableReplay() {
    static const bool disabled = std::getenv("APS5_NO_TABLE_REPLAY") != nullptr;
    return !disabled;
}

struct TableState {
    std::uint64_t epoch = 0;
    std::uint64_t unwatched = 0;
    std::uint64_t allocations = 0;
    std::uint64_t depthSurfaces = 0;
    bool operator==(const TableState&) const = default;
};

TableState CurrentTableState() {
    if (!TableReuseEnabled()) return {};
    return {GuestMemory::ThreadCollectEpoch(), GuestMemory::UnwatchSerial(), GuestAllocations::GuestAllocationsGeneration_nid_postfix(), DepthSurfacesSerial()};
}

struct ReusedTableElement {
    TextureKey key{};
    std::weak_ptr<Texture> texture;
    GuestTextureResource resource{};
    std::uint64_t guestBytes = 0;
    bool firstLayer = false;
    bool rejected = false;
    std::string message;
};

struct ReusedTable {
    VkDevice device = VK_NULL_HANDLE;
    TableState state;
    std::uint64_t pendingSerial = 0;
    std::vector<std::uint32_t> words;
    std::vector<ShaderRecompiler::DescriptorImageShape> shapes;
    ShaderRecompiler::ElementFlags depthCompare;
    std::vector<ReusedTableElement> elements;
    std::vector<GuestMemory::UnchangedQuery> queries;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::vector<std::uint32_t> imageSamplers;
    std::vector<std::uint32_t> samplers;
    std::vector<std::string> bound;
    bool replayable = false;
};

constexpr std::size_t ReusedTableCount = 4;

std::vector<ReusedTable>& ReusedTables() {
    thread_local std::vector<ReusedTable>* slot = nullptr;
    return ShaderRecompiler::ThreadOwned(slot);
}

bool SameTable(const ReusedTable& table, VkDevice device, const ShaderRecompiler::DescriptorBinding& binding) {
    return table.device == device && table.words == binding.guestDescriptor && table.shapes == binding.imageShapes && table.depthCompare == binding.imageDepthCompare;
}

ReusedTable* FindReusedTable(VkDevice device, const ShaderRecompiler::DescriptorBinding& binding, const TableState& state) {
    if (state.epoch == 0) return nullptr;
    auto& tables = ReusedTables();
    std::erase_if(tables, [&](const ReusedTable& table) { return table.state != state; });
    for (auto& table : tables) {
        if (SameTable(table, device, binding)) return &table;
    }
    return nullptr;
}

bool TableCurrent(ReusedTable& table) {
    const auto serial = StorageTexture::PendingSerial();
    if (serial != table.pendingSerial) {
        if (StorageTexture::AnyPendingOverlaps(table.ranges) || StorageTexture::PendingSerial() != serial) return false;
        table.pendingSerial = serial;
    }
    return GuestMemory::UnchangedSinceAll(table.queries);
}

bool CachedTableTextures(const ReusedTable& table, std::vector<std::shared_ptr<Texture>>& cached) {
    auto& cache = Textures();
    std::lock_guard lock(cache.mutex);
    std::vector<std::list<CachedTexture>::iterator> entries;
    entries.reserve(table.queries.size());
    for (const auto& element : table.elements) {
        if (element.rejected) continue;
        const auto it = findTexture(cache, element.key);
        if (it == cache.entries.end() || it->texture != element.texture.lock()) return false;
        entries.push_back(it);
    }
    cached.reserve(entries.size());
    for (const auto it : entries) {
        cached.push_back(it->texture);
        touchTexture(cache, it);
    }
    return true;
}

void RememberTable(ReusedTable&& table, const ShaderRecompiler::DescriptorBinding& binding) {
    auto& tables = ReusedTables();
    std::erase_if(tables, [&](const ReusedTable& entry) { return SameTable(entry, table.device, binding); });
    table.words = binding.guestDescriptor;
    table.shapes = binding.imageShapes;
    table.depthCompare = binding.imageDepthCompare;
    tables.insert(tables.begin(), std::move(table));
    if (tables.size() > ReusedTableCount) tables.resize(ReusedTableCount);
}

std::string HexText(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "%llx", static_cast<unsigned long long>(value));
    return text;
}

std::string DescriptorLimitError(std::string_view description, std::uint64_t required, std::uint64_t limit) {
    return std::string(description) + ": required " + std::to_string(required) + ", device limit " + std::to_string(limit);
}

void ValidateDescriptorLayout(const Context& context, std::span<const VkDescriptorSetLayoutBinding> bindings, bool updateAfterBind) {
    const auto& indexed = context.descriptorIndexingLimits;
    const bool sampledUpdateAfterBind = updateAfterBind && context.sampledImageUpdateAfterBind;
    const auto maxStageResources = updateAfterBind ? indexed.maxPerStageUpdateAfterBindResources : context.limits.maxPerStageResources;
    const auto maxStageSampledImages = sampledUpdateAfterBind ? indexed.maxPerStageDescriptorUpdateAfterBindSampledImages : context.limits.maxPerStageDescriptorSampledImages;
    const auto maxStageSamplers = sampledUpdateAfterBind ? indexed.maxPerStageDescriptorUpdateAfterBindSamplers : context.limits.maxPerStageDescriptorSamplers;
    const auto maxSetSampledImages = sampledUpdateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindSampledImages : context.limits.maxDescriptorSetSampledImages;
    const auto maxSetSamplers = sampledUpdateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindSamplers : context.limits.maxDescriptorSetSamplers;
    std::map<VkDescriptorType, std::uint64_t> heapTotals;
    std::map<std::pair<VkShaderStageFlags, VkDescriptorType>, std::uint64_t> stageTotals;
    std::map<VkShaderStageFlags, std::uint64_t> stageResources;
    for (const auto& layout : bindings) {
        heapTotals[layout.descriptorType] += layout.descriptorCount;
        for (std::uint32_t bit = 0u; bit < 32u; ++bit) {
            const auto stage = VkShaderStageFlags{1u} << bit;
            if ((layout.stageFlags & stage) == 0u) continue;
            const auto count = stageTotals[{stage, layout.descriptorType}] += layout.descriptorCount;
            const auto all = stageResources[stage] += layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ? 0u : layout.descriptorCount;
            Require(all <= maxStageResources, "typed heaps exceed per-stage resource capacity");
            if (layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) Require(count <= maxStageSampledImages, "typed heaps exceed per-stage sampled image capacity");
            if (layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) Require(count <= context.limits.maxPerStageDescriptorStorageImages, "typed heaps exceed per-stage storage image capacity");
            if (layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER && count > maxStageSamplers) throw std::runtime_error(DescriptorLimitError("typed heaps exceed per-stage sampler capacity", count, maxStageSamplers));
        }
    }
    Require(heapTotals[VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] <= maxSetSampledImages && heapTotals[VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] <= context.limits.maxDescriptorSetStorageImages, "typed heaps exceed descriptor set capacity");
    if (heapTotals[VK_DESCRIPTOR_TYPE_SAMPLER] > maxSetSamplers) throw std::runtime_error(DescriptorLimitError("typed heaps exceed descriptor set sampler capacity", heapTotals[VK_DESCRIPTOR_TYPE_SAMPLER], maxSetSamplers));
}

void ValidateRuntimeResources(const CompiledShader& shader, std::span<const std::uint32_t> words) {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    const auto& descriptors = program.bindings;
    constexpr auto bindingCount = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
    std::array<const ShaderRecompiler::DescriptorBinding*, bindingCount> heaps{};
    if (program.runtimeImageCount != 0u) {
        for (const auto& binding : descriptors) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) continue;
            auto& heap = heaps[binding.binding % bindingCount];
            Require(heap == nullptr, "duplicate runtime image binding");
            heap = &binding;
        }
    }
    Require(words.size() == program.shaderDataDwords, "invalid compact shader data size");
    constexpr auto metadataWords = sizeof(ShaderRecompiler::RuntimeAbi::ResourceMetadata) / sizeof(std::uint32_t);
    Require(program.imageMetadataDword <= words.size() && program.runtimeImageCount <= (words.size() - program.imageMetadataDword) / metadataWords, "runtime image metadata exceeds shader data");
    for (const auto index : program.runtimeImageResources) {
        Require(index < program.runtimeImageCount, "runtime image resource exceeds its compact layout");
        const auto offset = program.imageMetadataDword + index * metadataWords;
        const auto kind = words[offset];
        const auto first = words[offset + 1u];
        const auto count = words[offset + 2u];
        Require(count != 0u, "runtime image metadata has no descriptor elements");
        Require(kind < heaps.size() && heaps[kind] != nullptr, "runtime metadata references an unbound image heap");
        Require(first < heaps[kind]->count && count <= heaps[kind]->count - first, "runtime metadata exceeds its bound heap");
    }
}

}

void ShaderResources::buildPrepare(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes) {
    PerformanceTimer frameTiming("Resources.Prepare");
    const auto stageStart = std::chrono::steady_clock::now();
    phaseStart = stageStart;
    if (BuildProfiled()) phaseWaited = Recorder::ThreadWaitedMs();
    if (BuildProfiled()) {
        auto& profile = ThreadBuilds();
        if (++profile.builds % 1000 == 0) profile.merge();
    }
    try {
        Require(!shaders.empty() && context.limits.maxBoundDescriptorSets >= 1, "shader descriptor set exceeds device limits");
        if (shaders.size() == 1u) refreshResourceKey = ContentKey(shaders.front(), false, true);
        std::set<std::uint32_t> occupied;
        for (const auto& shader : shaders) {
            Require(shader.program != nullptr, "missing compiled shader");
            ShaderRecompiler::RuntimeAbi::RequireVersion(shader.program->runtimeAbiVersion);
            if (shader.program->poisonedSrtReads != 0) {
                const auto& programBindings = shader.program->bindings;
                const auto flat = std::find_if(programBindings.begin(), programBindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt; });
                if (flat != programBindings.end()) srtGuardFaults.push_back({IrStage(shader.stage), flat->guestDescriptor});
            }
            const VkShaderStageFlags flags = VulkanStage(shader.stage);
            for (const auto& [base, bytes] : BindlessReadableRanges(*shader.program)) guestMemory.AddReadable(base, static_cast<std::size_t>(bytes));
            std::uint64_t stageStorageBuffers = 0;
            std::uint64_t stageResources = shader.stage == ShaderRecompiler::ShaderStage::Fragment ? colorAttachments : 0;
            Require(stageResources <= context.limits.maxPerStageResources, "shader descriptors exceed per-stage limits");
            std::vector<std::size_t> offsetsInData;
            std::int64_t shaderData = -1;
            const auto firstSampler = samplers.size();
            const auto firstDeferred = deferredImages.size();
            const auto requireStageLimits = [&] {
                if (stageStorageBuffers <= context.limits.maxPerStageDescriptorStorageBuffers && stageResources <= context.limits.maxPerStageResources) return;
                const auto& limits = context.descriptorIndexingLimits;
                Require(stageStorageBuffers <= limits.maxPerStageDescriptorUpdateAfterBindStorageBuffers && stageResources <= limits.maxPerStageUpdateAfterBindResources, "shader descriptors exceed per-stage limits");
                updateAfterBind = true;
            };
            for (const auto& binding : shader.program->bindings) {
                Require(binding.descriptorSet == ShaderRecompiler::RuntimeAbi::DescriptorSet || (shader.stage == ShaderRecompiler::ShaderStage::Fragment && binding.descriptorSet == ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet), "unexpected descriptor set: a shader resource must use its stage's descriptor set");
                if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable) {
                    const bool storage = binding.kind == ShaderRecompiler::DescriptorKind::StorageImage;
                    Require((storage || binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) && binding.binding == (storage ? ShaderRecompiler::RuntimeAbi::StorageImageTableBinding : ShaderRecompiler::RuntimeAbi::ImageTableBinding) && binding.guestDescriptor.size() == static_cast<std::size_t>(binding.count) * 8u && binding.imageShapes.size() == binding.count && binding.imageDepthCompare.size() == binding.count && binding.imageSamplers.size() == binding.count && (!storage || (!binding.readOnly && binding.imageWritten.size() == binding.count)), "image table binding is malformed");
                    auto& table = tableStage(shader);
                    auto& images = storage ? table.storageImages : table.images;
                    Require(images == nullptr, "a stage binds two image tables of the same class");
                    images = &binding;
                    table.firstSampler = firstSampler;
                    continue;
                }
                Require(occupied.insert(binding.binding).second, "duplicate shader binding");
                if (binding.role == ShaderRecompiler::DescriptorRole::SamplerTable) {
                    addSamplerTable(binding, flags, tableStage(shader));
                    continue;
                }
                if (binding.role == ShaderRecompiler::DescriptorRole::ImageTableMap) {
                    Require(binding.kind == ShaderRecompiler::DescriptorKind::StorageBuffer && binding.count == 1 && binding.guestDescriptor.size() >= ShaderRecompiler::ImageTableAbi::HeaderWords && !binding.readOnly && binding.guestDescriptor[ShaderRecompiler::ImageTableAbi::HeaderVersion] == ShaderRecompiler::ImageTableAbi::Version, "image table map binding is malformed");
                    const auto size = binding.guestDescriptor.size() * sizeof(std::uint32_t);
                    Require(size <= context.limits.maxStorageBufferRange, "image table map exceeds the storage buffer range limit");
                    stageStorageBuffers += 1;
                    stageResources += 1;
                    storageBuffers += 1;
                    requireStageLimits();
                    auto& table = tableStage(shader);
                    Require(table.mapAllocation == std::numeric_limits<std::size_t>::max(), "a stage binds two image table maps");
                    table.map = binding.guestDescriptor;
                    const MemoryStats::OwnerScope memStatsOwner(MemoryStats::Owner::ShaderResources);
                    allocations.push_back({0, size, false, std::make_unique<Buffer>(context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT), ShaderRecompiler::DescriptorRole::ImageTableMap});
                    table.mapAllocation = allocations.size() - 1u;
                    bindings.push_back({{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flags, nullptr}, {allocations.size() - 1u}});
                    continue;
                }
                const bool addressRole = binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable || binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer;
                const bool bufferRole = addressRole || binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers || binding.role == ShaderRecompiler::DescriptorRole::ShaderData || binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt || binding.role == ShaderRecompiler::DescriptorRole::Gds;
                const bool imageRole = binding.role == ShaderRecompiler::DescriptorRole::GuestImages || binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers;
                if (imageRole) {
                    if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage || binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) stageResources += binding.count;
                    requireStageLimits();
                    addImageBinding(binding, flags, shader.program->hasGuardedUnnormalizedWrap);
                    continue;
                }
                if (!bufferRole) Require(false, std::string("unsupported descriptor role ") + roleName(binding.role));
                if (binding.kind != ShaderRecompiler::DescriptorKind::StorageBuffer) Require(false, std::string("unsupported descriptor kind ") + kindName(binding.kind) + " for role " + roleName(binding.role) + ": only StorageBuffer is supported");
                Require(!binding.readOnly, "read-only descriptors are unsupported because the recompiler emits no NonWritable decoration");
                Require(binding.count != 0, "empty descriptor binding");
                stageStorageBuffers += binding.count;
                stageResources += binding.count;
                storageBuffers += binding.count;
                requireStageLimits();
                Binding item{{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, binding.count, flags, nullptr}, {}};
                if (binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers) {
                    Require(binding.guestDescriptor.size() == static_cast<std::uint64_t>(binding.count) * 4, "guest buffer descriptor must contain four DWORDs per array element");
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        // An element the recompiler did not classify (a producer without the
                        // vector) counts as written, like imageWritten below.
                        const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                        const bool atomic = element < binding.bufferAtomic.size() && binding.bufferAtomic[element];
                        const bool read = element < binding.bufferRead.size() && binding.bufferRead[element];
                        const std::uint32_t extent = element < binding.bufferExtent.size() ? binding.bufferExtent[element] : 0u;
                        const auto index = addGuestBuffer(std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 4, 4), target, indexAddress, indexBytes, written, atomic, read, extent);
                        const bool classified = element < binding.bufferWritten.size() && element < binding.bufferRead.size() && element < binding.bufferAtomic.size();
                        allocations[index].writeOnly = classified && allocations[index].guest && allocations[index].written && binding.bufferWritten[element] && !read && !atomic;
                        const auto& push = shader.program->pushConstants;
                        if (!push.empty()) {
                            const auto position = shader.program->memoryOffsetDword * 4u + element;
                            Require(position < push.size(), "guest buffer offset lies outside the shader's push constants");
                            allocations[index].pushByte = static_cast<std::int32_t>(shader.pushConstantOffset + position);
                        } else {
                            allocations[index].dataByte = shader.program->memoryOffsetDword * 4u + element;
                            offsetsInData.push_back(index);
                        }
                        item.allocations.push_back(index);
                    }
                } else if (addressRole) {
                    item.allocations.push_back(allocations.size());
                    allocations.push_back({0, 0, false, nullptr, binding.role});
                } else if (binding.role == ShaderRecompiler::DescriptorRole::Gds) {
                    Require(binding.count == 1 && binding.guestDescriptor.empty(), "invalid GDS descriptor contract");
                    const auto gds = Pm4::GdsAddress();
                    guestMemory.AddWritable(gds, Pm4::GdsBytes, true);
                    item.allocations.push_back(allocations.size());
                    allocations.push_back({gds, Pm4::GdsBytes, true, nullptr, ShaderRecompiler::DescriptorRole::ShaderData, true});
                } else {
                    Require(binding.count == 1, "shader data and flattened SRT descriptors must not be arrays");
                    Require(!binding.guestDescriptor.empty(), "empty shader data descriptor");
                    if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) {
                        ValidateRuntimeResources(shader, binding.guestDescriptor);
                    }
                    item.allocations.push_back(addDataBuffer(binding.guestDescriptor));
                    if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) shaderData = static_cast<std::int64_t>(item.allocations.back());
                }
                bindings.push_back(std::move(item));
            }
            for (const auto index : offsetsInData) allocations[index].dataAllocation = shaderData;
            for (auto index = firstDeferred; index < deferredImages.size(); ++index) {
                deferredImages[index].firstSampler = firstSampler;
                deferredImages[index].samplerCount = samplers.size() - firstSampler;
            }
            for (auto& table : tableStages) {
                if (table.program == shader.program && table.images != nullptr) table.samplerCount = samplers.size() - table.firstSampler;
            }
        }
        splitSets = SplitsSets(context, shaders);
        std::array<VkShaderStageFlags, 2> tableFlags{};
        std::array<VkShaderStageFlags, 2> storageTableFlags{};
        for (auto& table : tableStages) {
            Require(table.mapAllocation != std::numeric_limits<std::size_t>::max(), "an image table stage has no map");
            const auto& map = table.map;
            Require(map.size() >= ShaderRecompiler::ImageTableAbi::HeaderWords, "image table map is truncated");
            const auto elements = table.images != nullptr ? table.images->count : table.storageImages != nullptr ? table.storageImages->count : 0u;
            Require(table.storageImages == nullptr || table.storageImages->count == elements, "sampled and storage image tables disagree on their element domain");
            Require(map[ShaderRecompiler::ImageTableAbi::HeaderElementCount] == elements, "image table map disagrees with its image binding");
            const auto faults = map[ShaderRecompiler::ImageTableAbi::HeaderElementFaults];
            const auto samplerFaults = map[ShaderRecompiler::ImageTableAbi::HeaderSamplerFaults];
            Require(static_cast<std::uint64_t>(faults) + elements <= map.size() && static_cast<std::uint64_t>(samplerFaults) + map[ShaderRecompiler::ImageTableAbi::HeaderSamplerCount] / 4u <= map.size(), "image table map is truncated");
            if (table.images == nullptr && table.storageImages == nullptr) continue;
            const auto* tableBinding = table.images != nullptr ? table.images : table.storageImages;
            table.set = tableBinding->descriptorSet == ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet ? 1u : 0u;
            table.elementBase = tableElements[table.set];
            table.map[ShaderRecompiler::ImageTableAbi::HeaderElementBase] = table.elementBase;
            tableElements[table.set] += elements;
            if (table.images != nullptr) tableFlags[table.set] |= VulkanStage(table.stage);
            if (table.storageImages != nullptr) storageTableFlags[table.set] |= VulkanStage(table.stage);
        }
        for (std::uint32_t set = 0; set < tableFlags.size(); ++set) {
            std::uint32_t storageTableDescriptors = 0;
            if (storageTableFlags[set] != 0) {
                storageTableDescriptors = StorageImageTableDescriptorCount(context, tableElements[set], tableFlags[set] != 0);
                Require(storageTableDescriptors != 0, "storage image tables exceed device descriptor limits or require unavailable descriptor indexing features");
                bindings.push_back({{ShaderRecompiler::RuntimeAbi::StorageImageTableBinding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageTableDescriptors, storageTableFlags[set], nullptr}, {}, {}, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, static_cast<std::int32_t>(set)});
                storageTableBindings[set] = bindings.size() - 1u;
            }
            if (tableFlags[set] == 0) continue;
            Require(context.imageTableCapacity != 0, "image tables need runtimeDescriptorArray, descriptorBindingPartiallyBound and descriptorBindingVariableDescriptorCount, which the device lacks");
            sampledTableCapacity[set] = storageTableDescriptors == 0 ? context.imageTableCapacity : std::min(context.imageTableCapacity, ImageTableCapacity(context, storageTableDescriptors));
            Require(sampledTableCapacity[set] != 0, "image tables exceed device descriptor limits or require unavailable descriptor indexing features");
            const VkDescriptorBindingFlags imageTableFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | (context.imageTableUpdateAfterBind ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0);
            bindings.push_back({{ShaderRecompiler::RuntimeAbi::ImageTableBinding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sampledTableCapacity[set], tableFlags[set], nullptr}, {}, {}, imageTableFlags, static_cast<std::int32_t>(set)});
            tableBindings[set] = bindings.size() - 1u;
        }
        if (storageBuffers > context.limits.maxDescriptorSetStorageBuffers) {
            Require(storageBuffers <= context.descriptorIndexingLimits.maxDescriptorSetUpdateAfterBindStorageBuffers, "pipeline descriptors exceed device limits");
            updateAfterBind = true;
        }
        timing.bindingsMs = phase(BuildPhase::Bindings);
        // For every build, locked ones included: their stage B then takes the fast path too, and the
        // collects cost the same wherever they run.
        if (precollectImages()) phase(BuildPhase::Precollect);
        if (drawBuild && !usesBda && std::all_of(allocations.begin(), allocations.end(), [&](const Allocation& allocation) { return !allocation.guest || allocation.pushByte >= 0 || (allocation.dataAllocation >= 0 && allocation.dataByte < allocations[static_cast<std::size_t>(allocation.dataAllocation)].size); })) guestMemory.AllowAdjustedRegions();
        guestMemory.UploadPrepare(usesBda);
        timing.uploadMs = phase(BuildPhase::Upload);
        splitSets = SplitsSets(context, shaders);
        std::array<std::vector<VkDescriptorSetLayoutBinding>, 2> descriptions;
        bool tablesUpdateAfterBind = false;
        for (const auto& binding : bindings) {
            const auto set = setOf(binding);
            descriptions[set].push_back(binding.layout);
            (set == 0 ? layoutBindingFlags : fragmentBindingFlags).push_back(binding.flags | (updateAfterBind && binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0));
            tablesUpdateAfterBind |= (binding.flags & VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT) != 0;
            auto& key = set == 0 ? layoutKey : fragmentLayoutKey;
            key.insert(key.end(), {binding.layout.binding, static_cast<std::uint32_t>(binding.layout.descriptorType), binding.layout.descriptorCount, binding.layout.stageFlags});
        }
        updateAfterBind = updateAfterBind || tablesUpdateAfterBind;
        if (updateAfterBind) {
            layoutKey.push_back(VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);
            if (splitSets) fragmentLayoutKey.push_back(VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);
        }
        static const bool noLayoutCache = std::getenv("APS5_NO_LAYOUT_CACHE") != nullptr;
        static const bool noPoolCache = std::getenv("APS5_NO_POOL_CACHE") != nullptr;
        const auto tableDescriptorCount = [&](std::size_t set) { return std::max(1u, std::min(tableElements[set], sampledTableCapacity[set])); };
        const auto tableVariableCount = [&](std::size_t set) { return tableBindings[set] != std::numeric_limits<std::size_t>::max() ? std::optional<std::uint32_t>(tableDescriptorCount(set)) : std::nullopt; };
        const auto splitPoolSizes = [&](std::size_t set) {
            std::map<VkDescriptorType, std::uint32_t> counts;
            for (std::size_t index = 0; index < bindings.size(); ++index) {
                if (setOf(bindings[index]) != set) continue;
                counts[bindings[index].layout.descriptorType] += index == tableBindings[set] ? tableDescriptorCount(set) : bindings[index].layout.descriptorCount;
            }
            std::vector<VkDescriptorPoolSize> sizes;
            for (const auto& [type, count] : counts) sizes.push_back({type, count});
            return sizes;
        };
        const auto allocateVariableSet = [&](VkDescriptorPool from, VkDescriptorSetLayout layout, std::optional<std::uint32_t> variableCount) {
            const std::uint32_t variable = variableCount.value_or(0u);
            VkDescriptorSetVariableDescriptorCountAllocateInfoEXT variableInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT};
            variableInfo.descriptorSetCount = 1;
            variableInfo.pDescriptorCounts = &variable;
            VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocation.pNext = variableCount.has_value() ? &variableInfo : nullptr;
            allocation.descriptorPool = from;
            allocation.descriptorSetCount = 1;
            allocation.pSetLayouts = &layout;
            VkDescriptorSet set = VK_NULL_HANDLE;
            Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &set), "vkAllocateDescriptorSets");
            return set;
        };
        const auto createLayout = [&](const std::vector<std::uint32_t>& key, const std::vector<VkDescriptorSetLayoutBinding>& description, const std::vector<VkDescriptorBindingFlags>& flags, bool& owns) {
            if (context.descriptorCache != nullptr && !noLayoutCache) return context.descriptorCache->Layout(key, description, updateAfterBind, flags);
            ValidateDescriptorLayout(context, description, updateAfterBind);
            owns = true;
            return CreateSetLayout(context, description, updateAfterBind, flags);
        };
        _layout = createLayout(layoutKey, descriptions[0], layoutBindingFlags, ownsLayout);
        if (splitSets) {
            fragment.layout = createLayout(fragmentLayoutKey, descriptions[1], fragmentBindingFlags, fragment.ownsLayout);
            {
                const auto sizes = splitPoolSizes(1);
                const auto variableCount = tableVariableCount(1);
                if (context.descriptorCache != nullptr && !noPoolCache) {
                    fragment.cacheAllocation = context.descriptorCache->Allocate(fragment.layout, sizes, updateAfterBind, variableCount);
                    fragment.set = fragment.cacheAllocation.set;
                }
                if (fragment.set == VK_NULL_HANDLE) {
                    fragment.pool = CreateSetPool(context, sizes, updateAfterBind);
                    fragment.set = allocateVariableSet(fragment.pool, fragment.layout, variableCount);
                }
            }
        }
        if (splitSets) {
            const auto sizes = splitPoolSizes(0);
            const auto variableCount = tableVariableCount(0);
            if (context.descriptorCache != nullptr && !noPoolCache) {
                cacheAllocation = context.descriptorCache->Allocate(_layout, sizes, updateAfterBind, variableCount);
                _set = cacheAllocation.set;
            }
            if (_set == VK_NULL_HANDLE) {
                pool = CreateSetPool(context, sizes, updateAfterBind);
                _set = allocateVariableSet(pool, _layout, variableCount);
            }
        }
        if (!splitSets && !bindings.empty()) {
            // The set is sized from the plan: every image element becomes exactly one descriptor
            // when stage B looks it up.
            std::vector<VkDescriptorPoolSize> sizes;
            if (storageBuffers != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<std::uint32_t>(storageBuffers)});
            const auto tableDescriptors = tableBindings[0] != std::numeric_limits<std::size_t>::max() ? tableDescriptorCount(0) : 0u;
            if (plannedSampledImages + tableDescriptors != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, plannedSampledImages + tableDescriptors});
            const auto storageTableDescriptors = storageTableBindings[0] != std::numeric_limits<std::size_t>::max() ? bindings[storageTableBindings[0]].layout.descriptorCount : 0u;
            if (plannedStorageImages + storageTableDescriptors != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, plannedStorageImages + storageTableDescriptors});
            std::uint32_t samplerDescriptors = 0;
            for (const auto& binding : bindings) {
                if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER) samplerDescriptors += binding.layout.descriptorCount;
            }
            if (samplerDescriptors != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLER, samplerDescriptors});
            const auto variableCount = tableVariableCount(0);
            if (context.descriptorCache != nullptr && !noPoolCache) {
                const auto allocated = context.descriptorCache->Allocate(_layout, sizes, updateAfterBind, variableCount);
                _set = allocated.set;
                cacheAllocation = allocated;
            }
            if (_set == VK_NULL_HANDLE) {
                pool = CreateSetPool(context, sizes, updateAfterBind);
                const std::uint32_t variable = variableCount.value_or(0u);
                VkDescriptorSetVariableDescriptorCountAllocateInfoEXT variableInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT};
                variableInfo.descriptorSetCount = 1;
                variableInfo.pDescriptorCounts = &variable;
                VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                allocation.pNext = variableCount.has_value() ? &variableInfo : nullptr;
                allocation.descriptorPool = pool;
                allocation.descriptorSetCount = 1;
                allocation.pSetLayouts = &_layout;
                Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &_set), "vkAllocateDescriptorSets");
            }
        }
        writeOnlyRanges.clear();
        for (const auto& item : allocations) {
            if (!item.writeOnly || item.size == 0) continue;
            const auto begin = item.address;
            const auto end = item.address + item.size;
            const bool shared = std::any_of(allocations.begin(), allocations.end(), [&](const Allocation& other) { return !other.writeOnly && other.size != 0 && other.address < end && begin < other.address + other.size; });
            if (!shared) writeOnlyRanges.emplace_back(begin, end);
        }
        timing.descriptorsMs = phase(BuildPhase::Descriptors);
        if (BuildProfiled()) {
            timing.prepareMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            addBuildPhase(BuildPhase::StageA, timing.prepareMs);
        }
    } catch (...) {
        release();
        throw;
    }
}

bool SplitsSets(const Context& context, std::span<const CompiledShader> shaders) {
    const auto fragmentSet = [](const CompiledShader& shader) { return shader.program != nullptr && std::any_of(shader.program->bindings.begin(), shader.program->bindings.end(), [](const auto& binding) { return binding.descriptorSet == ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet; }); };
    const auto fragmentStage = [](const CompiledShader& shader) { return shader.stage == ShaderRecompiler::ShaderStage::Fragment; };
    return std::any_of(shaders.begin(), shaders.end(), fragmentSet) || (context.graphicsPipelineLibrary && std::any_of(shaders.begin(), shaders.end(), fragmentStage));
}

DescribedSetLayouts DescribeSetLayouts(const Context& context, std::span<const CompiledShader> shaders) {
    Require(context.descriptorCache != nullptr, "described set layouts need the descriptor cache");
    DescribedSetLayouts result;
    std::array<std::vector<VkDescriptorSetLayoutBinding>, 2> descriptions;
    const bool split = SplitsSets(context, shaders);
    std::array<std::vector<VkDescriptorBindingFlags>, 2> bindingFlags;
    std::array<VkShaderStageFlags, 2> tableFlags{};
    std::array<VkShaderStageFlags, 2> storageTableFlags{};
    std::array<std::uint32_t, 2> tableElements{};
    bool updateAfterBind = false;
    bool tablesUpdateAfterBind = false;
    std::uint64_t storageBuffers = 0;
    for (const auto& shader : shaders) {
        Require(shader.program != nullptr, "missing compiled shader");
        const VkShaderStageFlags flags = VulkanStage(shader.stage);
        std::uint64_t stageStorageBuffers = 0;
        std::uint64_t stageResources = 0;
        std::uint32_t stageTableElements = 0;
        std::size_t tableSet = 0;
        for (const auto& binding : shader.program->bindings) {
            const auto set = binding.descriptorSet == ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet ? 1 : 0;
            if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable) {
                tableSet = static_cast<std::size_t>(set);
                (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage ? storageTableFlags : tableFlags)[tableSet] |= flags;
                stageTableElements = std::max(stageTableElements, binding.count);
                continue;
            }
            if (binding.role == ShaderRecompiler::DescriptorRole::SamplerTable) {
                descriptions[set].push_back({binding.binding, VK_DESCRIPTOR_TYPE_SAMPLER, context.samplerTableCapacity, flags, nullptr});
                bindingFlags[set].push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | (context.samplerTableUpdateAfterBind ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0));
                tablesUpdateAfterBind |= context.samplerTableUpdateAfterBind;
                continue;
            }
            const bool image = binding.role == ShaderRecompiler::DescriptorRole::GuestImages || binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers;
            const auto type = !image ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : binding.kind == ShaderRecompiler::DescriptorKind::StorageImage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : binding.kind == ShaderRecompiler::DescriptorKind::SampledImage ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER;
            if (type != VK_DESCRIPTOR_TYPE_SAMPLER) stageResources += binding.count;
            if (type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                stageStorageBuffers += binding.count;
                storageBuffers += binding.count;
            }
            if (stageStorageBuffers > context.limits.maxPerStageDescriptorStorageBuffers || stageResources > context.limits.maxPerStageResources) updateAfterBind = true;
            descriptions[set].push_back({binding.binding, type, binding.count, flags, nullptr});
            bindingFlags[set].push_back(0);
        }
        tableElements[tableSet] += stageTableElements;
    }
    for (std::size_t set = 0; set < tableFlags.size(); ++set) {
        std::uint32_t storageTableDescriptors = 0;
        if (storageTableFlags[set] != 0) {
            storageTableDescriptors = StorageImageTableDescriptorCount(context, tableElements[set], tableFlags[set] != 0);
            descriptions[set].push_back({ShaderRecompiler::RuntimeAbi::StorageImageTableBinding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageTableDescriptors, storageTableFlags[set], nullptr});
            bindingFlags[set].push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
        }
        if (tableFlags[set] == 0) continue;
        const auto capacity = storageTableDescriptors == 0 ? context.imageTableCapacity : std::min(context.imageTableCapacity, ImageTableCapacity(context, storageTableDescriptors));
        descriptions[set].push_back({ShaderRecompiler::RuntimeAbi::ImageTableBinding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, capacity, tableFlags[set], nullptr});
        bindingFlags[set].push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | (context.imageTableUpdateAfterBind ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0));
        tablesUpdateAfterBind |= context.imageTableUpdateAfterBind;
    }
    if (storageBuffers > context.limits.maxDescriptorSetStorageBuffers) updateAfterBind = true;
    for (std::size_t set = 0; set < descriptions.size(); ++set) {
        for (std::size_t index = 0; index < descriptions[set].size(); ++index) {
            if (updateAfterBind && descriptions[set][index].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) bindingFlags[set][index] |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        }
    }
    updateAfterBind = updateAfterBind || tablesUpdateAfterBind;
    for (std::size_t set = 0; set < descriptions.size(); ++set) {
        auto& key = set == 0 ? result.mainKey : result.fragmentKey;
        for (const auto& binding : descriptions[set]) key.insert(key.end(), {binding.binding, static_cast<std::uint32_t>(binding.descriptorType), binding.descriptorCount, binding.stageFlags});
        if (updateAfterBind && (set == 0 || split)) key.push_back(VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);
    }
    result.layouts.split = split;
    result.layouts.main = context.descriptorCache->Layout(result.mainKey, descriptions[0], updateAfterBind, bindingFlags[0]);
    if (split) result.layouts.fragment = context.descriptorCache->Layout(result.fragmentKey, descriptions[1], updateAfterBind, bindingFlags[1]);
    return result;
}

void ShaderResources::buildComplete() {
    PerformanceTimer frameTiming("Resources.Complete");
    BuildLookups lookups;
    Require(!completed, "cannot update a published descriptor heap");
    const auto stageStart = std::chrono::steady_clock::now();
    phaseStart = stageStart;
    if (BuildProfiled()) phaseWaited = Recorder::ThreadWaitedMs();
    try {
        // The lookups run in plan order: Revalidate walks the bindings the same way, and consecutive
        // storage elements of one mip chain share the previous element's image.
        const bool imagesProfiled = BuildProfiled();
        const auto outcomesBefore = imagesProfiled ? ThreadLookupOutcomes() : LookupOutcomes{};
        const auto deferredCount = deferredImages.size();
        auto imagesMark = imagesProfiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto lapImages = [&] {
            if (!imagesProfiled) return 0.0;
            const auto now = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(now - imagesMark).count();
            imagesMark = now;
            return ms;
        };
        previousSampled = {};
        for (auto& source : samplerSources) source.singleLevelImage = source.mipmappedImage = false;
        for (const auto& deferred : deferredImages) resolveImageBinding(*deferred.binding, bindings[deferred.index], std::span<const std::shared_ptr<Sampler>>(samplers).subspan(deferred.firstSampler, deferred.samplerCount));
        previousSampled = {};
        deferredImages.clear();
        applyAnisoOverride();
        // The records served their purpose: each holds the cache entry's objects as of stage A,
        // which would otherwise keep a replaced texture or an evicted storage image (and its device
        // memory) alive, outside the caches' budgets, for as long as this object is cached.
        imageRecords.clear();
        imageRecords.shrink_to_fit();
        nextImageRecord = 0;
        Require(textures.size() == plannedSampledImages && storageTextures.size() == plannedStorageImages, "image lookups disagree with the descriptor plan");
        const double deferredMs = lapImages();
        resolveStorageImageTables();
        const double storageTablesMs = lapImages();
        const auto tablesStart = std::chrono::steady_clock::now();
        resolveImageTables();
        if (!tableStages.empty()) AddBindlessOldPathNanoseconds(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tablesStart).count()));
        const double imageTablesMs = lapImages();
        std::vector<std::shared_ptr<StorageTexture>> depthWrites;
        for (std::size_t index = 0; index < storageTextures.size(); ++index) {
            if (storageWritten[index] && storageTextures[index] != nullptr && DepthSurfaceAt(storageTextures[index]->Descriptor().baseAddress)) depthWrites.push_back(storageTextures[index]);
        }
        if (depthWrites.size() > 1) ValidateStorageDepthWrites(context, depthWrites);
        if (imagesProfiled) noteCompleteImages(outcomesBefore, deferredCount, deferredMs, storageTablesMs, imageTablesMs, lapImages());
        timing.bindingsMs += phase(BuildPhase::Images);
        guestMemory.UploadFinish(usesBda);
        timing.uploadMs += phase(BuildPhase::Upload);
        if (usesBda) bda = std::make_unique<BdaResources>(context, guestMemory);
        else if (usesFaultBuffer) bda = std::make_unique<BdaResources>(context);
        if (bda) bda->SetSrtGuards(std::move(srtGuardFaults));
        finishImageTables();
        phase(BuildPhase::Bda);
        if (_set != VK_NULL_HANDLE || fragment.set != VK_NULL_HANDLE) {
            // One update call for the whole set: the info arrays are sized up front so every write's
            // pointer into them stays valid until the call.
            std::size_t bufferCount = 0;
            std::size_t imageCount = 0;
            for (const auto& binding : bindings) {
                bufferCount += binding.allocations.size();
                if (binding.layout.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) imageCount += binding.flags != 0 ? binding.imageAllocations.size() : binding.layout.descriptorCount;
            }
            auto& buffers = descriptorBuffers;
            auto& images = descriptorImages;
            buffers.reserve(bufferCount);
            images.reserve(imageCount);
            auto& writes = descriptorWrites;
            writes.reserve(bindings.size());
            std::vector<std::pair<std::size_t, std::size_t>> keptWrites;
            const auto absentImage = [this](std::uint32_t binding) {
                constexpr auto bindingCount = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
                const auto view = context.nullDescriptors ? VK_NULL_HANDLE : context.placeholderImages->View(binding % bindingCount);
                return VkDescriptorImageInfo{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
            };
            for (const auto& binding : bindings) {
                if (binding.flags != 0) {
                    const auto set = setOf(binding);
                    const bool sampled = binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    const auto& allocation = set == 0 ? cacheAllocation : fragment.cacheAllocation;
                    const bool recorded = sampled && set < tableWrites.size() && allocation.layout != VK_NULL_HANDLE;
                    std::vector<std::pair<std::uint32_t, std::uint32_t>> runs;
                    const auto firstWrite = writes.size();
                    const auto firstImage = images.size();
                    for (std::size_t first = 0; first < binding.imageAllocations.size();) {
                        if (binding.imageAllocations[first] == std::numeric_limits<std::size_t>::max()) {
                            ++first;
                            continue;
                        }
                        auto last = first;
                        while (last < binding.imageAllocations.size() && binding.imageAllocations[last] != std::numeric_limits<std::size_t>::max()) ++last;
                        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                        write.dstSet = set == 0 ? _set : fragment.set;
                        write.dstBinding = binding.layout.binding;
                        write.dstArrayElement = static_cast<std::uint32_t>(first);
                        write.descriptorCount = static_cast<std::uint32_t>(last - first);
                        write.descriptorType = binding.layout.descriptorType;
                        write.pImageInfo = images.data() + images.size();
                        for (auto element = first; element < last; ++element) {
                            const auto index = binding.imageAllocations[element];
                            if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER) images.push_back({samplers[index]->Handle(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
                            else if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) images.push_back({VK_NULL_HANDLE, storageAtomic[index] ? storageTextures[index]->AtomicView(storageMips[index], storageFirstLayer[index]) : storageTextures[index]->StorageView(storageMips[index], storageFirstLayer[index]), VK_IMAGE_LAYOUT_GENERAL});
                            else images.push_back({VK_NULL_HANDLE, textureFirstLayer[index] ? textures[index]->FirstLayerView() : textures[index]->View(), textures[index]->Layout()});
                        }
                        if (recorded) runs.emplace_back(write.dstArrayElement, write.descriptorCount);
                        writes.push_back(write);
                        first = last;
                    }
                    if (!recorded) continue;
                    const auto& held = allocation.table;
                    auto& counters = TextureCounts();
                    const bool heldAlive = held != nullptr && held->textures != nullptr && std::none_of(held->textures->begin(), held->textures->end(), [](const std::weak_ptr<Texture>& texture) { return texture.expired(); });
                    if (heldAlive && held->binding == binding.layout.binding && held->runs == runs && std::equal(held->images.begin(), held->images.end(), images.begin() + static_cast<std::ptrdiff_t>(firstImage), images.end(), [](const VkDescriptorImageInfo& a, const VkDescriptorImageInfo& b) { return a.sampler == b.sampler && a.imageView == b.imageView && a.imageLayout == b.imageLayout; })) {
                        keptWrites.emplace_back(firstWrite, writes.size());
                        counters.tableWritesKept.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        counters.tableWritesMade.fetch_add(1, std::memory_order_relaxed);
                    }
                    tableWrites[set] = {true, false, binding.layout.binding, firstImage, images.size() - firstImage, std::move(runs)};
                    continue;
                }
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = setOf(binding) == 0 ? _set : fragment.set;
                write.dstBinding = binding.layout.binding;
                write.descriptorCount = binding.layout.descriptorCount;
                write.descriptorType = binding.layout.descriptorType;
                switch (binding.layout.descriptorType) {
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                        write.pBufferInfo = buffers.data() + buffers.size();
                        for (const auto index : binding.allocations) {
                            buffers.push_back(descriptor(allocations[index]));
                        }
                        break;
                    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                        write.pImageInfo = images.data() + images.size();
                        for (const auto index : binding.imageAllocations) {
                            if (index == std::numeric_limits<std::size_t>::max()) images.push_back(absentImage(binding.layout.binding));
                            else images.push_back({VK_NULL_HANDLE, textureSrgb[index] ? textures[index]->SrgbView(textureFirstLayer[index], textureBaseLevel[index]) : textureBaseLevel[index] ? textures[index]->BaseLevelView() : textureFirstLayer[index] ? textures[index]->FirstLayerView() : textures[index]->View(), textures[index]->Layout()});
                        }
                        Require(binding.imageAllocations.size() == binding.layout.descriptorCount, "descriptor allocations disagree with compact binding");
                        break;
                    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                        write.pImageInfo = images.data() + images.size();
                        for (const auto index : binding.imageAllocations) {
                            if (index == std::numeric_limits<std::size_t>::max()) images.push_back(absentImage(binding.layout.binding));
                            else if (storageTextures[index] == nullptr) images.push_back({VK_NULL_HANDLE, multisampleStorageViews[index], VK_IMAGE_LAYOUT_GENERAL});
                            else images.push_back({VK_NULL_HANDLE, storageAtomic64[index] ? storageTextures[index]->Atomic64View(storageMips[index], storageFirstLayer[index]) : storageAtomic[index] ? storageTextures[index]->AtomicView(storageMips[index], storageFirstLayer[index]) : storageTextures[index]->StorageView(storageMips[index], storageFirstLayer[index]), VK_IMAGE_LAYOUT_GENERAL});
                        }
                        Require(binding.imageAllocations.size() == binding.layout.descriptorCount, "descriptor allocations disagree with compact binding");
                        break;
                    case VK_DESCRIPTOR_TYPE_SAMPLER:
                        write.pImageInfo = images.data() + images.size();
                        for (const auto index : binding.imageAllocations) images.push_back({samplers[index]->Handle(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
                        Require(binding.imageAllocations.size() == binding.layout.descriptorCount, "descriptor allocations disagree with compact binding");
                        break;
                    default: throw std::runtime_error("AGC graphics: ShaderResources encountered an unknown descriptor type while writing the descriptor set");
                }
                writes.push_back(write);
            }
            if (keptWrites.empty()) {
                context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
            } else {
                std::vector<VkWriteDescriptorSet> made;
                made.reserve(writes.size());
                std::size_t next = 0;
                for (std::size_t index = 0; index < writes.size(); ++index) {
                    if (next < keptWrites.size() && index == keptWrites[next].first) {
                        index = keptWrites[next++].second - 1u;
                        continue;
                    }
                    made.push_back(writes[index]);
                }
                if (!made.empty()) context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(made.size()), made.data(), 0, nullptr);
            }
            for (auto& table : tableWrites) table.written = table.recorded;
        }
        for (const auto& allocation : allocations) {
            if (allocation.adjustment == 0) continue;
            if (allocation.pushByte >= 0) {
                pushPatches.emplace_back(static_cast<std::uint32_t>(allocation.pushByte), allocation.adjustment);
                continue;
            }
            Require(allocation.dataAllocation >= 0, "a guest buffer off the storage buffer offset alignment in a shader without shader data is not implemented");
            auto& data = allocations[static_cast<std::size_t>(allocation.dataAllocation)];
            Require(data.buffer != nullptr && allocation.dataByte < data.size, "guest buffer offset lies outside the shader's data buffer");
            data.buffer->Bytes()[allocation.dataByte] = static_cast<std::byte>(allocation.adjustment);
            dataPatches.push_back({static_cast<std::size_t>(allocation.dataAllocation), allocation.dataByte, allocation.adjustment});
        }
        timing.descriptorsMs += phase(BuildPhase::Descriptors);
        noteReusable();
        completed = true;
        if (BuildProfiled()) {
            timing.completeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            addBuildPhase(BuildPhase::StageB, timing.completeMs);
            reportDescriptorCaches();
        }
    } catch (...) {
        release();
        throw;
    }
}

// Whether this build can serve later dispatches of the same content (see ContentKey): nothing to do
// once the GPU completed (no BDA, no copied written buffers, no lease) and every guest buffer bound
// in place through a host import (or staged in device memory from one, see
// GuestBufferMemory::DirectRegions) whose identity is recorded for Revalidate.
namespace {

// APS5_NO_TEMPLATE_DATA_REFRESH=1: compute keys keep the ShaderData/FlattenedSrt words and no
// template refreshes its data buffers (see ShaderResources::ContentKey).
bool TemplateDataRefresh() {
    static const bool enabled = std::getenv("APS5_NO_TEMPLATE_DATA_REFRESH") == nullptr;
    return enabled;
}

constexpr std::size_t ContentKeyHeaderWords = 7;
constexpr std::size_t ContentKeyDataWordsAt = 2;
constexpr std::size_t ContentKeyVariantAt = 4;
constexpr std::size_t ContentKeyBindingsAt = 6;
constexpr std::size_t ContentKeyFlagSets = 10;

bool DataRole(ShaderRecompiler::DescriptorRole role) {
    return role == ShaderRecompiler::DescriptorRole::ShaderData || role == ShaderRecompiler::DescriptorRole::FlattenedSrt;
}

// The largest data buffer vkCmdUpdateBuffer refreshes; a template with a bigger one is not reused.
constexpr std::size_t MaxRefreshBytes = 65536;

// FNV-1a over one data buffer's words, its count first (DataWordsHash).
constexpr std::uint64_t FnvOffset = 14695981039346656037ull;
constexpr std::uint64_t FnvPrime = 1099511628211ull;
void mixDataWords(std::uint64_t& hash, std::span<const std::uint32_t> words) {
    hash = (hash ^ static_cast<std::uint64_t>(words.size())) * FnvPrime;
    for (const auto word : words) hash = (hash ^ word) * FnvPrime;
}

// ResourceCache::Find and Touch calls (the [rescache] line).
std::atomic<std::uint64_t> resourceCacheFinds{0};
std::atomic<std::uint64_t> resourceCacheTouches{0};
std::atomic<std::uint64_t> depthGuardRefusals{0};

}

namespace {

bool LoopGuardEnabled() {
    static const bool enabled = std::getenv("APS5_LOOP_GUARD") != nullptr;
    return enabled;
}

bool AddressReuseEnabled() {
    static const bool enabled = std::getenv("APS5_NO_ADDRESS_DRAW_REUSE") == nullptr;
    return enabled;
}

// APS5_NO_BDA_TEMPLATES=1 (an APS5_AB switch): read-only address-table draw builds are cached only
// by #6636's whole-build reuse (APS5_NO_ADDRESS_DRAW_REUSE unset), as before; else they are
// templates (SharesUses, SnapshotsServed, per-use descriptor sets for changed data words).
bool BdaTemplatesOn() {
    return !RuntimeSwitches::Set(RuntimeSwitch::NoBdaTemplates);
}

void CompleteSnapshotWrites(std::span<const GuestMemorySnapshot> snapshots) {
    auto* recorder = Recorder::Active();
    if (recorder == nullptr || Recorder::InCompletion()) return;
    for (const auto& snapshot : snapshots) {
        if (snapshot.bytes.empty() || !recorder->PendingWriteOverlaps(snapshot.address, snapshot.bytes.size())) continue;
        Recorder::CountSync(4);
        recorder->SyncThrough(snapshot.address, snapshot.bytes.size());
    }
}

}

void ShaderResources::noteReusable() {
    // Taken whether or not the object turns out reusable: the lookups' records are this build's.
    captureValidation();
    reusable = false;
    addressReuse = false;
    directRegions.clear();
    addressWrites.clear();
    static const bool profileTemplates = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const bool candidate = profileTemplates && usesBda && drawBuild && bdaReadOnly && BdaTemplatesOn();
    const auto refuse = [&](TemplateOutcome outcome) {
        reusableRefusal = outcome;
        if (candidate) CountTemplateOutcome(outcome);
    };
    bdaTemplate = false;
    sharedUses = false;
    if (storageTableBindings[0] != std::numeric_limits<std::size_t>::max() || storageTableBindings[1] != std::numeric_limits<std::size_t>::max()) return refuse(TemplateOutcome::StorageTable);
    const bool templated = usesBda && drawBuild && bdaReadOnly && BdaTemplatesOn();
    const bool addressBased = usesBda && drawBuild && (AddressReuseEnabled() || templated);
    if (guestMemory.HasCopiedWrites()) return refuse(TemplateOutcome::CopiedWrites);
    if ((!addressBased && (HoldsLease() || usesBda)) || (bda != nullptr && LoopGuardEnabled())) return refuse(TemplateOutcome::LoopGuard);
    if (!multisampleSources.empty()) return refuse(TemplateOutcome::Multisample);
    if (std::any_of(storageTextures.begin(), storageTextures.end(), [](const auto& texture) { return texture != nullptr && DepthSurfaceAt(texture->Descriptor().baseAddress); })) return refuse(TemplateOutcome::DepthStorage);
    if (addressBased && std::any_of(tableStages.begin(), tableStages.end(), [](const TableStage& table) { return std::find(table.elementTextures.begin(), table.elementTextures.end(), std::numeric_limits<std::size_t>::max()) != table.elementTextures.end(); })) return refuse(TemplateOutcome::TableElement);
    if (std::any_of(textures.begin(), textures.end(), [](const std::shared_ptr<Texture>& texture) { return texture != nullptr && texture->RefreshedPerUse(); })) return refuse(TemplateOutcome::RefreshedPerUse);
    if (TemplateDataRefresh() && std::any_of(allocations.begin(), allocations.end(), [](const Allocation& allocation) { return allocation.buffer != nullptr && !allocation.guest && allocation.role != ShaderRecompiler::DescriptorRole::ImageTableMap && allocation.size > MaxRefreshBytes; })) return refuse(TemplateOutcome::DataSize);
    const auto regions = addressBased ? guestMemory.SpaceRegions() : guestMemory.DirectRegions();
    if (!regions.has_value()) return refuse(TemplateOutcome::SpaceRegions);
    for (const auto& [begin, end] : *regions) {
        // No reconcile here: the upload just took these imports, and the set is about to be recorded
        // against them.
        auto serial = HostImportSerial(context, begin, static_cast<std::size_t>(end - begin), false);
        // Read-only image mirrors (exe ranges) are as stable as imports; their serials have the top
        // bit set, so the two spaces never collide.
        if (serial == 0) serial = ImageMirrorSerial(context, begin, static_cast<std::size_t>(end - begin));
        if (serial == 0) {
            directRegions.clear();
            return refuse(TemplateOutcome::Serial);
        }
        directRegions.push_back({begin, end, serial});
    }
    for (const auto& cut : guestMemory.CutRegions()) {
        for (auto& region : directRegions) {
            if (region.begin != cut.begin) continue;
            region.regionEnd = cut.end;
            region.committedEnd = cut.committedEnd;
        }
    }
    if (addressBased) addressWrites = guestMemory.DirectWrites();
    reusable = true;
    addressReuse = addressBased;
    bdaTemplate = templated;
    sharedUses = templated && addressWrites.empty() && guestMemory.Writes().empty() && !guestMemory.HasGpuCopies() && !guestMemory.HasCopiedWrites() && std::none_of(storageWritten.begin(), storageWritten.end(), [](bool written) { return written; });
    refuse(sharedUses ? TemplateOutcome::Shared : TemplateOutcome::NotShared);
}

namespace {
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ShaderResources::TemplateOutcome::Count)> templateOutcomes{};
}

void ShaderResources::CountTemplateOutcome(TemplateOutcome outcome) {
    templateOutcomes[static_cast<std::size_t>(outcome)].fetch_add(1, std::memory_order_relaxed);
}

std::array<std::uint64_t, static_cast<std::size_t>(ShaderResources::TemplateOutcome::Count)> ShaderResources::TemplateOutcomes() {
    std::array<std::uint64_t, static_cast<std::size_t>(TemplateOutcome::Count)> result{};
    for (std::size_t index = 0; index < result.size(); ++index) result[index] = templateOutcomes[index].load(std::memory_order_relaxed);
    return result;
}

const char* ShaderResources::TemplateOutcomeName(TemplateOutcome outcome) {
    switch (outcome) {
        case TemplateOutcome::Storing: return "draws storing through BDA";
        case TemplateOutcome::StoringExtOnly: return "of them only through extended instructions";
        case TemplateOutcome::StorageTable: return "storage image table";
        case TemplateOutcome::CopiedWrites: return "copied writes";
        case TemplateOutcome::Multisample: return "multisample source";
        case TemplateOutcome::LoopGuard: return "loop guard";
        case TemplateOutcome::DepthStorage: return "depth storage image";
        case TemplateOutcome::TableElement: return "rejected table element";
        case TemplateOutcome::RefreshedPerUse: return "texture refreshed per use";
        case TemplateOutcome::DataSize: return "data buffer too big";
        case TemplateOutcome::SpaceRegions: return "regions copied per build";
        case TemplateOutcome::Serial: return "region without import serial";
        case TemplateOutcome::Shared: return "templates";
        case TemplateOutcome::NotShared: return "templates one use at a time";
        case TemplateOutcome::Inserted: return "inserted";
        case TemplateOutcome::Count: break;
    }
    return "?";
}

bool ShaderResources::SharesUses() const {
    return sharedUses && BdaTemplatesOn();
}

std::uint64_t ShaderResources::DepthGuardRefusals() {
    return depthGuardRefusals.load(std::memory_order_relaxed);
}

bool ShaderResources::NeverReusable(std::span<const CompiledShader> shaders) {
    const bool loopGuard = LoopGuardEnabled();
    const bool addressReuse = AddressReuseEnabled();
    const bool never = std::any_of(shaders.begin(), shaders.end(), [loopGuard, addressReuse](const CompiledShader& shader) {
        return shader.program != nullptr && std::any_of(shader.program->bindings.begin(), shader.program->bindings.end(), [loopGuard, addressReuse](const ShaderRecompiler::DescriptorBinding& binding) { return (binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable && !addressReuse) || (loopGuard && binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer); });
    });
    if (!never || addressReuse || loopGuard || !BdaTemplatesOn()) return never;
    return StagesStoreThroughAddresses(shaders);
}

// APS5_PROFILE_DRAW: the per-device descriptor caches' counters, every 10 s.
void ShaderResources::reportDescriptorCaches() const {
    static AgcDriver::Mutex reportMutex;
    static auto lastReport = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(reportMutex);
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    const auto descriptors = context.descriptorCache != nullptr ? context.descriptorCache->Counters() : DescriptorCache::Stats{};
    const auto samplerHits = context.samplerCache != nullptr ? context.samplerCache->Hits() : 0;
    const auto samplerMisses = context.samplerCache != nullptr ? context.samplerCache->Misses() : 0;
    AgcDriver::ProfilePrint_nid_no_patch("[descriptors] layouts %llu hits / %llu created, sets %llu from %llu pools and %llu dedicated pools, samplers %llu hits / %llu created\n", static_cast<unsigned long long>(descriptors.layoutHits), static_cast<unsigned long long>(descriptors.layoutMisses), static_cast<unsigned long long>(descriptors.sets), static_cast<unsigned long long>(descriptors.pools), static_cast<unsigned long long>(descriptors.dedicatedPools), static_cast<unsigned long long>(samplerHits), static_cast<unsigned long long>(samplerMisses));
}

template<typename TVisitor>
void VisitContentKey(const CompiledShader& shader, bool dataWords, bool movableBuffers, const TVisitor& visit) {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    visit(program.hasGuardedUnnormalizedWrap ? 1u : 0u);
    visit(ShaderRecompiler::UnnormalizedWrapEnabled() ? 1u : 0u);
    visit(dataWords ? 1u : 0u);
    visit(static_cast<std::uint32_t>(shader.stage));
    visit(static_cast<std::uint32_t>(program.PipelineVariantId()));
    visit(static_cast<std::uint32_t>(program.PipelineVariantId() >> 32u));
    visit(static_cast<std::uint32_t>(program.bindings.size()));
    const auto packBits = [&](const ShaderRecompiler::ElementFlags& bits) {
        visit(static_cast<std::uint32_t>(bits.size()));
        std::uint32_t word = 0;
        for (std::size_t i = 0; i < bits.size(); ++i) {
            if (bits[i]) word |= 1u << (i % 32u);
            if (i % 32u == 31u || i + 1 == bits.size()) {
                visit(word);
                word = 0;
            }
        }
    };
    for (const auto& binding : program.bindings) {
        for (const auto word : {static_cast<std::uint32_t>(binding.kind), static_cast<std::uint32_t>(binding.role), binding.descriptorSet, binding.binding, binding.count, binding.readOnly ? 1u : 0u, binding.imageShape.has_value() ? static_cast<std::uint32_t>(*binding.imageShape) + 1u : 0u, static_cast<std::uint32_t>(binding.guestDescriptor.size())}) visit(word);
        if (movableBuffers && binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers && binding.guestDescriptor.size() == static_cast<std::size_t>(binding.count) * 4u) {
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u;
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                const bool empty = words[2] == 0 || (words[0] == 0 && (words[1] & 0xffffu) == 0);
                if (written || empty) { for (std::uint32_t index = 0; index < 4u; ++index) visit(words[index]); }
                else { for (const auto word : {0u, words[1] & 0xffff0000u, 0u, words[3]}) visit(word); }
            }
        } else if ((dataWords && !movableBuffers) || !DataRole(binding.role)) {
            for (const auto word : binding.guestDescriptor) visit(word);
        }
        packBits(binding.imageWritten);
        packBits(binding.samplerDepthCompare);
        packBits(binding.imageDepthCompare);
        packBits(binding.imageAtomic);
        packBits(binding.samplerUnnormalized);
        packBits(binding.samplerIntegerBorder);
        packBits(binding.samplerCoordinatesIgnored);
        packBits(binding.imageUnnormalized);
        packBits(binding.imageUnnormalizedBaseLevel);
        // Read-only elements are bound without a write set: an object built for one written set
        // must not serve a build with another (the variant implies it, this makes it explicit).
        packBits(binding.bufferWritten);
        visit(static_cast<std::uint32_t>(binding.imageSamplers.size()));
        for (const auto sampler : binding.imageSamplers) visit(sampler);
        visit(static_cast<std::uint32_t>(binding.imageShapes.size()));
        for (const auto shape : binding.imageShapes) visit(static_cast<std::uint32_t>(shape));
    }
}

template<typename TVisitor>
void VisitContentKeyTagged(const CompiledShader& shader, bool dataWords, bool movableBuffers, const TVisitor& tagged) {
    std::uint8_t role = 0;
    const auto visit = [&](std::uint32_t word) { tagged(word, role); };
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    visit(program.hasGuardedUnnormalizedWrap ? 1u : 0u);
    visit(ShaderRecompiler::UnnormalizedWrapEnabled() ? 1u : 0u);
    visit(dataWords ? 1u : 0u);
    visit(static_cast<std::uint32_t>(shader.stage));
    role = 1;
    visit(static_cast<std::uint32_t>(program.PipelineVariantId()));
    visit(static_cast<std::uint32_t>(program.PipelineVariantId() >> 32u));
    role = 2;
    visit(static_cast<std::uint32_t>(program.bindings.size()));
    const auto packBits = [&](const ShaderRecompiler::ElementFlags& bits) {
        role = 7;
        visit(static_cast<std::uint32_t>(bits.size()));
        std::uint32_t word = 0;
        for (std::size_t i = 0; i < bits.size(); ++i) {
            if (bits[i]) word |= 1u << (i % 32u);
            if (i % 32u == 31u || i + 1 == bits.size()) {
                visit(word);
                word = 0;
            }
        }
    };
    for (const auto& binding : program.bindings) {
        role = 2;
        for (const auto word : {static_cast<std::uint32_t>(binding.kind), static_cast<std::uint32_t>(binding.role), binding.descriptorSet, binding.binding, binding.count, binding.readOnly ? 1u : 0u, binding.imageShape.has_value() ? static_cast<std::uint32_t>(*binding.imageShape) + 1u : 0u, static_cast<std::uint32_t>(binding.guestDescriptor.size())}) visit(word);
        if (movableBuffers && binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers && binding.guestDescriptor.size() == static_cast<std::size_t>(binding.count) * 4u) {
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u;
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                const bool empty = words[2] == 0 || (words[0] == 0 && (words[1] & 0xffffu) == 0);
                role = 5;
                if (written || empty) { for (std::uint32_t index = 0; index < 4u; ++index) visit(words[index]); }
                else { for (const auto word : {0u, words[1] & 0xffff0000u, 0u, words[3]}) visit(word); }
            }
        } else if ((dataWords && !movableBuffers) || !DataRole(binding.role)) {
            for (std::size_t index = 0; index < binding.guestDescriptor.size(); ++index) {
                role = DataRole(binding.role) ? 5 : (index % 8u < 2u ? 3 : 4);
                visit(binding.guestDescriptor[index]);
            }
        }
        packBits(binding.imageWritten);
        packBits(binding.samplerDepthCompare);
        packBits(binding.imageDepthCompare);
        packBits(binding.imageAtomic);
        packBits(binding.samplerUnnormalized);
        packBits(binding.samplerIntegerBorder);
        packBits(binding.samplerCoordinatesIgnored);
        packBits(binding.imageUnnormalized);
        packBits(binding.imageUnnormalizedBaseLevel);
        // Read-only elements are bound without a write set: an object built for one written set
        // must not serve a build with another (the variant implies it, this makes it explicit).
        packBits(binding.bufferWritten);
        role = 6;
        visit(static_cast<std::uint32_t>(binding.imageSamplers.size()));
        for (const auto sampler : binding.imageSamplers) visit(sampler);
        role = 2;
        visit(static_cast<std::uint32_t>(binding.imageShapes.size()));
        for (const auto shape : binding.imageShapes) visit(static_cast<std::uint32_t>(shape));
    }
}

std::vector<std::uint32_t> ShaderResources::ContentKey(const CompiledShader& shader, bool dataWords, bool movableBuffers) {
    std::vector<std::uint32_t> key;
    AppendContentKey(shader, dataWords, movableBuffers, key);
    return key;
}

void ShaderResources::AppendContentKey(const CompiledShader& shader, bool dataWords, bool movableBuffers, std::vector<std::uint32_t>& key) {
    Require(shader.program != nullptr, "missing compiled shader");
    std::size_t words = key.size() + 7;
    const auto addWords = [&](std::size_t count) {
        if (count > key.max_size() - words) throw std::length_error("resource content key exceeds vector limits");
        words += count;
    };
    for (const auto& binding : shader.program->bindings) {
        addWords(20);
        addWords(binding.imageSamplers.size());
        addWords(binding.imageShapes.size());
        if ((dataWords && !movableBuffers) || !DataRole(binding.role)) addWords(binding.guestDescriptor.size());
        for (const auto* bits : {&binding.imageWritten, &binding.samplerDepthCompare, &binding.imageDepthCompare, &binding.imageAtomic, &binding.samplerUnnormalized, &binding.samplerIntegerBorder, &binding.samplerCoordinatesIgnored, &binding.imageUnnormalized, &binding.imageUnnormalizedBaseLevel, &binding.bufferWritten}) addWords(bits->size() / 32u + (bits->size() % 32u != 0 ? 1u : 0u));
    }
    key.reserve(words);
    VisitContentKey(shader, dataWords, movableBuffers, [&](std::uint32_t word) { key.push_back(word); });
}

// Debug (local, APS5_RESKEY_DIFF): the role of every word AppendContentKey appends, in the same order: 0 header,
// 1 pipeline variant id, 2 binding layout, 3 image/sampler descriptor address words, 4 other image/sampler descriptor
// words, 5 buffer descriptor words, 6 sampler indices, 7 flag bits.
void ShaderResources::AppendContentKeyRoles(const CompiledShader& shader, bool dataWords, bool movableBuffers, std::vector<std::uint8_t>& roles) {
    VisitContentKeyTagged(shader, dataWords, movableBuffers, [&](std::uint32_t, std::uint8_t role) { roles.push_back(role); });
}

namespace {

using Component = ContentKeyComponent;

bool LabelShaderKey(std::span<const std::uint32_t> key, std::size_t at, std::size_t end, bool movable, std::vector<std::uint8_t>& labels) {
    const auto label = [&](std::size_t first, std::size_t count, Component component) {
        if (first + count > end) return false;
        std::fill(labels.begin() + static_cast<std::ptrdiff_t>(first), labels.begin() + static_cast<std::ptrdiff_t>(first + count), static_cast<std::uint8_t>(component));
        return true;
    };
    if (!label(at, ContentKeyHeaderWords, Component::Layout)) return false;
    const bool dataWords = key[at + ContentKeyDataWordsAt] != 0;
    const auto bindings = key[at + ContentKeyBindingsAt];
    at += ContentKeyHeaderWords;
    for (std::uint32_t binding = 0; binding < bindings; ++binding) {
        if (!label(at, 8, Component::Layout)) return false;
        const auto role = static_cast<ShaderRecompiler::DescriptorRole>(key[at + 1]);
        const auto size = key[at + 7];
        at += 8;
        const auto words = DataRole(role) && !(dataWords && !movable) ? 0u : size;
        if (at + words > end) return false;
        for (std::uint32_t i = 0; i < words; ++i) {
            auto component = Component::Other;
            if (role == ShaderRecompiler::DescriptorRole::GuestBuffers) component = static_cast<Component>(static_cast<std::size_t>(Component::BufferWord0) + i % 4u);
            else if (role == ShaderRecompiler::DescriptorRole::GuestImages || role == ShaderRecompiler::DescriptorRole::ImageTable) component = Component::Image;
            else if (role == ShaderRecompiler::DescriptorRole::GuestSamplers || role == ShaderRecompiler::DescriptorRole::SamplerTable) component = Component::Sampler;
            else if (DataRole(role)) component = Component::ShaderData;
            labels[at + i] = static_cast<std::uint8_t>(component);
        }
        at += words;
        for (std::size_t flags = 0; flags < ContentKeyFlagSets; ++flags) {
            if (at >= end) return false;
            const auto count = 1 + (static_cast<std::size_t>(key[at]) + 31) / 32;
            if (!label(at, count, Component::Other)) return false;
            at += count;
        }
        if (at >= end) return false;
        const auto samplers = static_cast<std::size_t>(key[at]);
        if (!label(at, 1 + samplers, Component::Sampler)) return false;
        at += 1 + samplers;
        if (at >= end) return false;
        const auto shapes = static_cast<std::size_t>(key[at]);
        if (!label(at, 1 + shapes, Component::Image)) return false;
        at += 1 + shapes;
    }
    return at == end;
}

bool LabelContentKey(std::span<const std::uint32_t> key, bool draw, bool movable, std::vector<std::uint8_t>& labels) {
    labels.assign(key.size(), static_cast<std::uint8_t>(Component::Layout));
    if (!draw) {
        if (LabelShaderKey(key, 0, key.size(), movable, labels)) return true;
        if (key.size() < 2) return false;
        labels.assign(key.size(), static_cast<std::uint8_t>(Component::Layout));
        if (!LabelShaderKey(key, 0, key.size() - 2, movable, labels)) return false;
        labels[key.size() - 2] = static_cast<std::uint8_t>(Component::Other);
        labels[key.size() - 1] = static_cast<std::uint8_t>(Component::Other);
        return true;
    }
    if (key.size() < 4) return false;
    std::fill(labels.begin() + 1, labels.begin() + 3, static_cast<std::uint8_t>(Component::Other));
    std::size_t at = 4;
    for (std::uint32_t stage = 0; stage < key[3]; ++stage) {
        if (at >= key.size()) return false;
        const auto size = static_cast<std::size_t>(key[at]);
        ++at;
        if (at + size > key.size() || !LabelShaderKey(key, at, at + size, movable, labels)) return false;
        at += size;
    }
    if (at >= key.size()) return false;
    for (auto tail = at + 1; tail < key.size(); ++tail) labels[tail] = static_cast<std::uint8_t>(Component::DrawTarget);
    return true;
}

}

bool ShaderResources::DiffContentKeys(std::span<const std::uint32_t> previous, std::span<const std::uint32_t> key, bool draw, ContentKeyDiff& diff) {
    diff = {};
    std::vector<std::uint8_t> labels;
    bool labelled = false;
    for (const bool movable : {draw, false}) {
        labelled = LabelContentKey(key, draw, movable, labels);
        if (labelled) break;
    }
    if (!labelled || previous.size() != key.size()) {
        diff[static_cast<std::size_t>(Component::Layout)] = true;
        return false;
    }
    for (std::size_t i = 0; i < key.size(); ++i) {
        if (previous[i] != key[i]) diff[labels[i]] = true;
    }
    return true;
}

bool MatchesContentKey(const CompiledShader& shader, std::span<const std::uint32_t> key) {
    std::size_t cursor = 0;
    bool equal = true;
    VisitContentKey(shader, false, true, [&](std::uint32_t word) {
        equal &= cursor < key.size() && key[cursor] == word;
        ++cursor;
    });
    return equal && cursor == key.size();
}

namespace {

// APS5_PROFILE_DRAW: Revalidate outcomes and time, printed as [rescache] every 10 s next to the
// device's hit/miss line: failed = the object could not be reused (whichever image path it took);
// of the reused ones, fast = every image proved current from stamps, full = the lookups were repeated.
// Why the fast path left an object to the full walk (the "fast-fail by reason" counts).
using FastFail = ShaderResources::FastFail;
constexpr std::array<const char*, static_cast<std::size_t>(FastFail::Count)> FastFailNames{"no record", "collect", "pending image", "evicted image", "memory changed", "keys", "cleared view", "storage keys", "depth surface", "committed parts changed", "replaced texture", "audit"};
using OwnRefreshFallback = ShaderResources::OwnRefreshFallback;
constexpr std::array<const char*, static_cast<std::size_t>(OwnRefreshFallback::Count)> OwnRefreshFallbackNames{"disabled", "snapshot texture", "cleared view", "foreign view", "surface key", "not imported", "uncached", "re-run failed"};

struct RevalidateProfile {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> nanoseconds{0};
    std::atomic<std::uint64_t> fast{0};
    std::atomic<std::uint64_t> full{0};
    std::atomic<std::uint64_t> failed{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FastFail::Count)> fastFails{};
    // Epoch gate: registry scans skipped because the pending serial was unchanged, and import
    // serial loops skipped because the table's identity was unchanged.
    std::atomic<std::uint64_t> serialSkips{0};
    std::atomic<std::uint64_t> importSkips{0};
    // Elements with DCC keys the fast path accepted through the images' key proofs.
    std::atomic<std::uint64_t> keysProven{0};
    std::atomic<std::uint64_t> storageKeysProven{0};
    // The 'pending image' fast-fails by what overlapped: a surface with an own object (the query
    // excepts it, so the overlapping image is foreign to the surface) or a snapshot texture (no own
    // object); and the fails whose overlapping surfaces all have an own object (the T1 trim of
    // design_cpu_final applies: flush the foreign image, refresh the own object, no full walk).
    std::atomic<std::uint64_t> pendingForeign{0};
    std::atomic<std::uint64_t> pendingSnapshot{0};
    std::atomic<std::uint64_t> pendingT1Eligible{0};
    // The full walks by the reason of the fast proof that preceded them (fastFails also counts the
    // Pending failures the own-object refresh resolved), the Pending failures it resolved (with the
    // objects it refreshed: storage images, view sources) and those it left to the walk by reason;
    // view surfaces a foreign image overlapped that the proof accepted because FindPending names
    // their own source; surfaces a foreign image was still pending over after their own object's
    // refresh (its alias, a straddling image) that the re-run accepted; proofs the verify switch
    // checked against the full walk.
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FastFail::Count)> fullByReason{};
    std::atomic<std::uint64_t> ownRefreshed{0};
    std::atomic<std::uint64_t> ownStorageRefreshes{0};
    std::atomic<std::uint64_t> ownViewRefreshes{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(OwnRefreshFallback::Count)> ownFallbacks{};
    std::atomic<std::uint64_t> ownSourceViews{0};
    std::atomic<std::uint64_t> refreshedOverlaps{0};
    std::atomic<std::uint64_t> proofsVerified{0};
    std::atomic<std::int64_t> lastReport{0};
};

RevalidateProfile& Revalidations() {
    static RevalidateProfile profile;
    return profile;
}

void countFastFail(FastFail reason) {
    if (BuildProfiled()) Revalidations().fastFails[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void countPendingFail(std::span<const StorageTexture::PendingQuery> pending) {
    if (!BuildProfiled()) return;
    auto& profile = Revalidations();
    bool eligible = true;
    for (const auto& query : pending) {
        if (!query.overlaps) continue;
        if (query.except != nullptr) {
            profile.pendingForeign.fetch_add(1, std::memory_order_relaxed);
        } else {
            profile.pendingSnapshot.fetch_add(1, std::memory_order_relaxed);
            eligible = false;
        }
    }
    if (eligible) profile.pendingT1Eligible.fetch_add(1, std::memory_order_relaxed);
}

void countKeysProven(bool storage) {
    if (BuildProfiled()) (storage ? Revalidations().storageKeysProven : Revalidations().keysProven).fetch_add(1, std::memory_order_relaxed);
}

void countFullWalk(FastFail reason) {
    if (BuildProfiled() && reason != FastFail::Count) Revalidations().fullByReason[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void countOwnRefreshFallback(OwnRefreshFallback reason) {
    if (BuildProfiled()) Revalidations().ownFallbacks[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void countOwnRefresh(bool storage) {
    TextureCounts().ownRefreshes.fetch_add(1, std::memory_order_relaxed);
    if (BuildProfiled()) (storage ? Revalidations().ownStorageRefreshes : Revalidations().ownViewRefreshes).fetch_add(1, std::memory_order_relaxed);
}

// APS5_NO_OWN_IMAGE_REFRESH=1: a Pending failure of the fast proof takes the full walk, as before
// the own-object refresh (T1).
bool OwnImageRefreshEnabled() {
    static const bool disabled = std::getenv("APS5_NO_OWN_IMAGE_REFRESH") != nullptr;
    return !disabled;
}

// APS5_VERIFY_PROOFS=1: after every own-object refresh the full walk runs beside the proof and the
// process aborts when it would return another object or upload again (a different decision).
bool VerifyProofs() {
    static const bool enabled = std::getenv("APS5_VERIFY_PROOFS") != nullptr;
    return enabled;
}

// Whether a sampled surface's keys are the very keys of the image the view follows (the same
// metadata, extent, format and alpha placement: the same scan), so the image's proof serves it.
bool SameKeySurface(const StorageTexture* source, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    if (source == nullptr) return false;
    const auto& own = source->Descriptor();
    return own.dccAddress == resource.dccAddress && source->GuestBytes() == guestBytes && own.format == resource.format && own.dccAlphaOnMsb == resource.dccAlphaOnMsb;
}

void countRevalidate(bool fast, bool ok, std::chrono::steady_clock::time_point start) {
    auto& profile = Revalidations();
    const auto now = std::chrono::steady_clock::now();
    profile.calls.fetch_add(1, std::memory_order_relaxed);
    profile.nanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count()), std::memory_order_relaxed);
    (!ok ? profile.failed : fast ? profile.fast : profile.full).fetch_add(1, std::memory_order_relaxed);
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    auto last = profile.lastReport.load();
    if (nowMs - last < 10000 || !profile.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto byReason = [](auto& counts, const auto& names) {
        std::string reasons;
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto count = counts[i].exchange(0, std::memory_order_relaxed);
            if (count == 0) continue;
            reasons += " " + std::string(names[i]) + " " + std::to_string(count);
        }
        return reasons;
    };
    const auto reasons = byReason(profile.fastFails, FastFailNames);
    const auto fullReasons = byReason(profile.fullByReason, FastFailNames);
    const auto fallbacks = byReason(profile.ownFallbacks, OwnRefreshFallbackNames);
    AgcDriver::ProfilePrint_nid_no_patch("[rescache] revalidate (10 s): %llu calls %.1f ms: fast %llu, full %llu, failed %llu; fast-fail by reason:%s; full walks: T1 refreshed %llu (%llu storage images, %llu view sources, %llu views served by their own pending source, %llu overlaps left by the refresh accepted), full by reason:%s, T1 fallback by reason:%s, proofs verified %llu; fast-fail pending image: own 0 (the query excepts the own object), foreign %llu, snapshot %llu, T1 eligible %llu; keys proven: %llu sampled, %llu storage; epoch gate: %llu registry scans skipped, %llu import loops skipped; lookup log: %llu builds and full walks kept their records past the bound\n", static_cast<unsigned long long>(profile.calls.exchange(0, std::memory_order_relaxed)), profile.nanoseconds.exchange(0, std::memory_order_relaxed) / 1e6, static_cast<unsigned long long>(profile.fast.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.full.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.failed.exchange(0, std::memory_order_relaxed)), reasons.c_str(), static_cast<unsigned long long>(profile.ownRefreshed.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.ownStorageRefreshes.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.ownViewRefreshes.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.ownSourceViews.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.refreshedOverlaps.exchange(0, std::memory_order_relaxed)), fullReasons.c_str(), fallbacks.c_str(), static_cast<unsigned long long>(profile.proofsVerified.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.pendingForeign.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.pendingSnapshot.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.pendingT1Eligible.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.keysProven.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.storageKeysProven.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.serialSkips.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.importSkips.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(lookupBuildsPastBound.exchange(0, std::memory_order_relaxed)));
}

// APS5_NO_EPOCH_REVALIDATE=1: the per-element registry, cache and stamp checks and the per-region
// flush and import lookups of every Revalidate, as before the epoch gate.
bool EpochRevalidate() {
    static const bool enabled = std::getenv("APS5_NO_EPOCH_REVALIDATE") == nullptr;
    return enabled;
}

std::chrono::steady_clock::duration TemplateAuditInterval() {
    static const auto interval = [] {
        const char* text = std::getenv("APS5_TEMPLATE_AUDIT_MS");
        return std::chrono::milliseconds(text != nullptr ? std::strtoull(text, nullptr, 10) : 1000ull);
    }();
    return interval;
}

bool TemplateAuditDue(std::chrono::steady_clock::time_point last) {
    const auto interval = TemplateAuditInterval();
    return interval.count() != 0 && std::chrono::steady_clock::now() - last >= interval;
}

}

namespace {

bool touchSampledTextures(const std::vector<std::shared_ptr<Texture>>& textures) {
    const auto managed = [](const std::shared_ptr<Texture>& texture) { return texture != nullptr && texture->SampledCacheState() != Texture::CacheState::Unmanaged; };
    if (std::none_of(textures.begin(), textures.end(), managed)) return true;
    auto& cache = Textures();
    const SampledCacheLock lock(cache);
    for (const auto& texture : textures) {
        if (!managed(texture)) continue;
        const auto found = cache.byTexture.find(texture.get());
        if (found == cache.byTexture.end()) return false;
        touchTexture(cache, found->second);
    }
    return true;
}

}

void LastRecordIndices(std::span<const void* const> logged, std::span<const void* const> objects, std::vector<std::size_t>& indices) {
    indices.assign(objects.size(), logged.size());
    thread_local std::vector<std::pair<const void*, std::size_t>>* wantedSlot = nullptr;
    auto& wanted = ShaderRecompiler::ThreadOwned(wantedSlot);
    wanted.clear();
    for (std::size_t i = 0; i < objects.size(); ++i) wanted.emplace_back(objects[i], i);
    const auto byObject = [](const std::pair<const void*, std::size_t>& left, const std::pair<const void*, std::size_t>& right) { return std::less<const void*>{}(left.first, right.first); };
    std::sort(wanted.begin(), wanted.end(), byObject);
    std::size_t open = wanted.size();
    for (auto at = logged.size(); at-- > 0 && open != 0;) {
        const auto [first, last] = std::equal_range(wanted.begin(), wanted.end(), std::pair<const void*, std::size_t>{logged[at], 0}, byObject);
        if (first == last || indices[first->second] != logged.size()) continue;
        for (auto it = first; it != last; ++it) indices[it->second] = at;
        open -= static_cast<std::size_t>(last - first);
    }
    if (wanted.capacity() > LookupLogBound + LookupLogBound / 2) {
        wanted.clear();
        wanted.shrink_to_fit();
    }
}

// Turns the lookups' records into this object's per-texture validation records (see
// ValidatedSurface); the last record of an object is the freshest.
namespace {

std::array<std::uint64_t, 4> untrackedRangeStamp(std::uint64_t address, std::size_t bytes) {
    static const bool disabled = std::getenv("APS5_NO_UNTRACKED_FAST_PROOF") != nullptr;
    if (disabled) return {};
    const auto epoch = GuestMemory::CollectEpoch();
    if (epoch == 0 || bytes == 0 || (GuestMemory::ForgetSerial() & 1u) != 0) return {};
    const auto* recorder = Recorder::Active();
    return {epoch, GuestMemory::NewestStamp(address, bytes), GuestMemory::ForgetStamp(address, bytes), recorder != nullptr ? recorder->NewestWriteNote(address, bytes) : 0};
}

}

void ShaderResources::captureValidation() {
    const auto& log = lookupLog();
    thread_local std::vector<const void*>* loggedSlot = nullptr;
    auto& logged = ShaderRecompiler::ThreadOwned(loggedSlot);
    thread_local std::vector<const void*>* objectsSlot = nullptr;
    auto& objects = ShaderRecompiler::ThreadOwned(objectsSlot);
    thread_local std::vector<std::size_t>* indicesSlot = nullptr;
    auto& indices = ShaderRecompiler::ThreadOwned(indicesSlot);
    logged.clear();
    for (const auto& entry : log) logged.push_back(entry.object);
    objects.clear();
    for (const auto& texture : textures) objects.push_back(texture.get());
    LastRecordIndices(logged, objects, indices);
    validatedTextures.assign(textures.size(), {});
    for (std::size_t i = 0; i < textures.size(); ++i) {
        if (indices[i] != log.size()) {
            const auto& found = log[indices[i]];
            validatedTextures[i] = {found.resource, found.bytes, found.keys, found.committed.has_value() ? found.committedGeneration : found.generation, 0, found.source, found.depth, true, found.committed};
        }
        auto& surface = validatedTextures[i];
        if (surface.valid && !surface.depth && GuestMemory::CollectWrites(surface.resource.baseAddress, static_cast<std::size_t>(surface.bytes)) == 0) surface.untracked = untrackedRangeStamp(surface.resource.baseAddress, static_cast<std::size_t>(surface.bytes));
    }
    storageUntracked.assign(storageTextures.size(), {});
    for (std::size_t i = 0; i < storageTextures.size(); ++i) {
        const auto* image = storageTextures[i].get();
        if (image == nullptr) continue;
        const auto address = image->Descriptor().baseAddress;
        const auto bytes = static_cast<std::size_t>(image->GuestBytes());
        if (GuestMemory::CollectWrites(address, bytes) == 0) storageUntracked[i] = untrackedRangeStamp(address, bytes);
    }
    lookupLog().clear();
    const auto trim = [](auto& scratch) {
        if (scratch.capacity() <= LookupLogBound + LookupLogBound / 2) return;
        scratch.clear();
        scratch.shrink_to_fit();
    };
    trim(logged);
    trim(objects);
    trim(indices);
}

std::size_t ShaderResources::LookupLogCapacity() {
    return lookupLog().capacity();
}

// Proves every texture and storage image still current without repeating its lookup: exactly the
// "unchanged" branches of cachedTexture and StorageTexture::Refresh, evaluated from write stamps,
// the pending-results registry and the DCC keys. Per element: collect first (stamps come only from
// collects, and the generation a record moves to must predate the checks, so a write landing between
// them is stamped newer: the rule of Driver's dispatch cache), then no other image may have results
// pending over the memory (a lookup would flush them into it), a storage image must still be the
// cache's image of its surface, the memory must be unchanged since the object's content matched it,
// and the DCC keys must be what the content was made under whenever the surface has any (a fast
// clear touches only the keys). Anything else, including a failed collect (memory outside the arena,
// uncommitted pages), is left to the full walk.
// The epoch gate: while the pending registry's serial is what this object's last proof saw
// (pendingSerialSeen), the registry is what it was then (every change bumps the serial under the
// registry mutex), so the pending-image checks are skipped; otherwise one scan answers them for
// every surface at once. The stamp checks go through one tracker lock, the cache checks through
// the images' cached flags and one touch of the cache.
bool ShaderResources::fastRevalidate(std::uint64_t serialBefore, std::span<const PendingOverlap> refreshed, FastFail& reason, std::vector<PendingOverlap>& overlapping, bool& accepted) {
    reason = FastFail::Count;
    overlapping.clear();
    accepted = false;
    if (!EpochRevalidate()) return fastRevalidateEach();
    const auto fail = [&reason](FastFail why) {
        reason = why;
        countFastFail(why);
        return false;
    };
    if (validatedTextures.size() != textures.size()) return fail(FastFail::NoRecord);
    const bool unchanged = pendingSerialSeen != 0 && pendingSerialSeen == serialBefore;
    const bool keyProofs = KeyFastPath();
    thread_local std::vector<GuestMemory::UnchangedQuery>* queriesSlot = nullptr;
    auto& queries = ShaderRecompiler::ThreadOwned(queriesSlot);
    thread_local std::vector<StorageTexture::PendingQuery>* pendingSlot = nullptr;
    auto& pending = ShaderRecompiler::ThreadOwned(pendingSlot);
    // The element each pending query stands for, in query order.
    thread_local std::vector<PendingOverlap>* ownersSlot = nullptr;
    auto& owners = ShaderRecompiler::ThreadOwned(ownersSlot);
    thread_local std::vector<const StorageTexture*>* imagesSlot = nullptr;
    auto& images = ShaderRecompiler::ThreadOwned(imagesSlot);
    thread_local std::vector<DccKeys>* scannedKeysSlot = nullptr;
    auto& scannedKeys = ShaderRecompiler::ThreadOwned(scannedKeysSlot);
    queries.clear();
    pending.clear();
    owners.clear();
    images.clear();
    scannedKeys.assign(textures.size(), DccKeys::Uncompressed);
    const auto query = [&](std::uint64_t begin, std::uint64_t end, const StorageTexture* except, const StorageTexture* identity, PendingOverlap owner) {
        pending.push_back({begin, end, except, identity, false});
        owners.push_back(owner);
    };
    for (std::size_t i = 0; i < textures.size(); ++i) {
        auto& surface = validatedTextures[i];
        if (!surface.valid) return fail(FastFail::NoRecord);
        if (surface.depth) {
            if (!DepthSurfaceHolds(context, surface.resource, textures[i].get())) return fail(FastFail::Depth);
            continue;
        }
        if (textures[i]->Released()) return fail(FastFail::Replaced);
        const auto address = surface.resource.baseAddress;
        const auto bytes = static_cast<std::size_t>(surface.bytes);
        if (!FlushDepthPlanes(address, bytes).empty()) return fail(FastFail::Changed);
        surface.collected = GuestMemory::CollectWrites(address, bytes);
        bool committedProof = false;
        bool untracked = false;
        if (surface.collected == 0) {
            if (surface.source == nullptr && surface.committed.has_value() && CommittedProofEnabled()) {
                thread_local std::vector<std::pair<std::uint64_t, std::uint64_t>>* runsSlot = nullptr;
                auto& runs = ShaderRecompiler::ThreadOwned(runsSlot);
                surface.collected = GuestMemory::CollectCommittedWrites(address, bytes, runs);
                if (surface.collected == 0 || runs != *surface.committed || Recorder::SnapshotWriteOverlaps(address, bytes) || !GuestMemory::UnchangedCommittedSince(runs, surface.generation)) return fail(FastFail::Committed);
                committedProof = true;
            } else {
                if (surface.untracked[0] == 0 || untrackedRangeStamp(address, bytes) != surface.untracked) return fail(FastFail::Collect);
                untracked = true;
            }
        }
        const auto* source = surface.source;
        if (!keyProofs) {
            // Without proofs the keys are compared with the record's (a cleared view's identity
            // below is the record's too): the scan repeats on every call, as before.
            if (surface.resource.dccAddress != 0 && TextureClearKeys(surface.resource, surface.bytes) != surface.keys) return fail(FastFail::Keys);
            scannedKeys[i] = surface.keys;
            if (!unchanged) query(address, address + bytes, source, source != nullptr && surface.keys != DccKeys::Uncompressed ? source : nullptr, {i, false, source != nullptr && surface.keys == DccKeys::Uncompressed});
        } else if (source != nullptr) {
            // The view follows its image, so it stays valid while the image would Refresh as
            // unchanged: the image's own keys must be what its content was uploaded under (its
            // proof answers without a scan while they are unstamped), and its memory unchanged
            // since its generation (the query below). The surface's own keys come from the same
            // proof when they are the image's, else from the texture's.
            const auto& own = source->Descriptor();
            const auto sourceKeys = source->ProvedKeys();
            if (sourceKeys != source->UploadedKeys()) return fail(FastFail::Keys);
            const auto keys = surface.resource.dccAddress == 0 ? DccKeys::Uncompressed : SameKeySurface(source, surface.resource, surface.bytes) ? sourceKeys : ProvedClearKeys(surface.resource, surface.bytes, textures[i]->KeyProof());
            if (surface.resource.dccAddress != 0 && surface.resource.dccAddress != own.dccAddress && (keys != DccKeys::Uncompressed || !StorageImageServesKeys(*source, surface.resource.dccAddress))) return fail(FastFail::Keys);
            scannedKeys[i] = keys;
            // A view of a fast-cleared surface stays one only while its image still has results
            // pending over it (cachedTexture's hit rule), unless the image's own descriptor names
            // the surface's metadata (the lookup views the cleared image then). The identity is
            // asked whatever the registry's serial did: the keys may have moved since the last
            // proof while the registry did not.
            if (keys != DccKeys::Uncompressed && !(own.dccAddress == surface.resource.dccAddress && ClearedViewEnabled() && StorageClearAvailable(context, surface.resource.format, keys))) query(address, address + bytes, source, source, {i, false, false});
            else if (!unchanged) query(address, address + bytes, source, nullptr, {i, false, keys == DccKeys::Uncompressed});
            if (surface.resource.dccAddress != 0 || own.dccAddress != 0) countKeysProven(false);
        } else {
            // A snapshot holds its clear texels or the guest bytes: the keys must be what it was made under.
            const auto keys = surface.resource.dccAddress != 0 ? ProvedClearKeys(surface.resource, surface.bytes, textures[i]->KeyProof()) : DccKeys::Uncompressed;
            if (keys != surface.keys) return fail(FastFail::Keys);
            scannedKeys[i] = keys;
            if (!unchanged) query(address, address + bytes, nullptr, nullptr, {i, false, false});
            if (surface.resource.dccAddress != 0) countKeysProven(false);
        }
        if (source != nullptr) {
            if (!source->Cached()) return fail(FastFail::Evicted);
            if (!untracked) queries.push_back({address, bytes, source->GenerationFor(address, bytes)});
            images.push_back(source);
        } else if (!committedProof && !untracked) {
            queries.push_back({address, bytes, surface.generation});
        }
    }
    for (std::size_t i = 0; i < storageTextures.size(); ++i) {
        if (storageTextures[i] != nullptr && (i >= storageKeys.size() || !StorageImageServesKeys(*storageTextures[i], storageKeys[i]))) return fail(FastFail::StorageKeys);
        if (i != 0 && storageTextures[i] == storageTextures[i - 1]) continue;
        const auto* image = storageTextures[i].get();
        if (image == nullptr) return fail(FastFail::NoRecord);
        const auto& own = image->Descriptor();
        const auto address = own.baseAddress;
        const auto bytes = static_cast<std::size_t>(image->GuestBytes());
        if (DepthSurfaceOverlaps(context.device, address, bytes)) {
            depthGuardRefusals.fetch_add(1, std::memory_order_relaxed);
            return fail(FastFail::Depth);
        }
        bool untracked = false;
        if (GuestMemory::CollectWrites(address, bytes) == 0) {
            if (i >= storageUntracked.size() || storageUntracked[i][0] == 0 || untrackedRangeStamp(address, bytes) != storageUntracked[i]) return fail(FastFail::Collect);
            untracked = true;
        }
        if (own.dccAddress != 0) {
            // Refresh's unchanged branch (its key compare against the keys the content was
            // uploaded under, through the image's proof; the memory query below), minus its
            // byte-compare fallback; the image's generation is left where it is.
            if (!keyProofs || image->ProvedKeys() != image->UploadedKeys()) return fail(FastFail::StorageKeys);
            countKeysProven(true);
        }
        if (!unchanged) query(address, address + bytes, image, nullptr, {i, true, false});
        if (!image->Cached()) return fail(FastFail::Evicted);
        if (!untracked) queries.push_back({address, bytes, image->GenerationFor(address, bytes)});
        images.push_back(image);
    }
    if (!pending.empty()) {
        if (!StorageTexture::ScanPending(pending)) return fail(FastFail::ClearedView);
        // A foreign image over a view under uncompressed keys is no failure while FindPending
        // names the view's own source: the lookup views the source (cachedTexture's hit rule)
        // and touches nothing else. A surface whose own object T1 refreshed in this Revalidate
        // (`refreshed`) is no failure either: a storage element's lookup is that Refresh and
        // returns the image whatever stays registered over it; a view's lookup hits on the source
        // when FindPending names it, and views the cached image of its key (the source, proved by
        // refreshOwnObjects) when FindPending names nothing. Every other overlap is the full
        // walk's, or T1's (Revalidate).
        const auto refreshedOwner = [&](const PendingOverlap& owner) {
            return std::find_if(refreshed.begin(), refreshed.end(), [&](const PendingOverlap& entry) { return entry.element == owner.element && entry.storage == owner.storage; });
        };
        for (std::size_t k = 0; k < pending.size(); ++k) {
            if (!pending[k].overlaps) continue;
            if (OwnImageRefreshEnabled() && owners[k].viewUncompressed && pending[k].found == pending[k].except) {
                if (BuildProfiled()) Revalidations().ownSourceViews.fetch_add(1, std::memory_order_relaxed);
                accepted = true;
                continue;
            }
            if (const auto entry = refreshedOwner(owners[k]); entry != refreshed.end() && (entry->storage || pending[k].found == pending[k].except || (pending[k].found == nullptr && entry->sourceEligible))) {
                if (BuildProfiled()) Revalidations().refreshedOverlaps.fetch_add(1, std::memory_order_relaxed);
                accepted = true;
                continue;
            }
            overlapping.push_back(owners[k]);
        }
        if (!overlapping.empty()) {
            countPendingFail(pending);
            return fail(FastFail::Pending);
        }
    } else if (unchanged && BuildProfiled()) {
        Revalidations().serialSkips.fetch_add(1, std::memory_order_relaxed);
    }
    if (!GuestMemory::UnchangedSinceAll(queries)) return fail(FastFail::Changed);
    if (!StorageImagesCached(context, images)) return fail(FastFail::Evicted);
    for (const auto* image : images) image->NoteProved();
    for (std::size_t i = 0; i < validatedTextures.size(); ++i) {
        auto& surface = validatedTextures[i];
        if (surface.source == nullptr) surface.generation = surface.collected;
        surface.keys = scannedKeys[i];
    }
    return true;
}

bool ShaderResources::fastRevalidateEach() {
    if (validatedTextures.size() != textures.size()) return false;
    for (std::size_t i = 0; i < textures.size(); ++i) {
        auto& surface = validatedTextures[i];
        if (!surface.valid) return false;
        if (surface.depth) {
            if (!DepthSurfaceHolds(context, surface.resource, textures[i].get())) return false;
            continue;
        }
        if (textures[i]->Released()) return false;
        const auto address = surface.resource.baseAddress;
        const auto bytes = static_cast<std::size_t>(surface.bytes);
        if (!FlushDepthPlanes(address, bytes).empty()) return false;
        surface.collected = GuestMemory::CollectWrites(address, bytes);
        if (surface.collected == 0) return false;
        if (PendingStorageOverlaps(address, bytes, surface.source)) return false;
        if (surface.source != nullptr && surface.resource.dccAddress != 0 && surface.resource.dccAddress != surface.source->Descriptor().dccAddress) return false;
        if (surface.source != nullptr) {
            // The view follows the image: it needs the image current with guest memory, as the
            // lookup's Refresh would make it.
            if (!StorageImageCached(context, surface.source) || !GuestMemory::UnchangedSince(address, bytes, surface.source->GenerationFor(address, bytes))) return false;
        } else if (!GuestMemory::UnchangedSince(address, bytes, surface.generation)) {
            return false;
        }
        if (surface.resource.dccAddress != 0 && TextureClearKeys(surface.resource, surface.bytes) != surface.keys) return false;
        // A view made under fast-clear keys stays one only while its image still has results pending
        // over the surface (the lookup prefers them to the clear); once they are flushed the clear
        // the image cannot see wins and the lookup makes a snapshot (cachedTexture's hit rule).
        if (surface.source != nullptr && surface.keys != DccKeys::Uncompressed && StorageTexture::FindPending(address, surface.bytes).get() != surface.source) return false;
    }
    for (std::size_t i = 0; i < storageTextures.size(); ++i) {
        if (storageTextures[i] != nullptr && (i >= storageKeys.size() || (storageKeys[i] != 0 && storageKeys[i] != storageTextures[i]->Descriptor().dccAddress))) return false;
        // Consecutive elements of one mip chain share the image (see addStorageImageBinding).
        if (i != 0 && storageTextures[i] == storageTextures[i - 1]) continue;
        const auto* image = storageTextures[i].get();
        if (image == nullptr) return false;
        // StorageTexture::Refresh checks the image's own surface, not the element's descriptor (one
        // image serves every descriptor of its memory, with or without a DCC address), and compares
        // its keys with the keys it was uploaded under, which only the image knows (they move to
        // Uncompressed on every write-back; a fast clear after it puts the key bytes back to what a
        // record saw while the image holds last frame's results), so a surface with keys is left to
        // the full walk. Without a DCC address both sides of that compare are Uncompressed whatever
        // happened, and the memory checks below are Refresh's.
        const auto& own = image->Descriptor();
        if (own.dccAddress != 0) return false;
        const auto address = own.baseAddress;
        const auto bytes = static_cast<std::size_t>(image->GuestBytes());
        if (DepthSurfaceOverlaps(context.device, address, bytes)) {
            depthGuardRefusals.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (GuestMemory::CollectWrites(address, bytes) == 0) return false;
        if (PendingStorageOverlaps(address, bytes, image)) return false;
        if (!StorageImageCached(context, image)) return false;
        if (!GuestMemory::UnchangedSince(address, bytes, image->GenerationFor(address, bytes))) return false;
    }
    // Every check passed: snapshot textures are current at the collects issued before them. Storage
    // images keep their own generation: nothing was stamped above it, so the blocks a later write
    // stamps are the same set either way.
    for (auto& surface : validatedTextures) {
        if (surface.source == nullptr) surface.generation = surface.collected;
    }
    return true;
}

ShaderResources::OwnRefreshFallback ShaderResources::refreshOwnObjects(std::span<const CompiledShader> shaders, std::span<PendingOverlap> overlapping) {
    const auto listed = [&](std::size_t element, bool storage) {
        return std::find_if(overlapping.begin(), overlapping.end(), [&](const PendingOverlap& overlap) { return overlap.element == element && overlap.storage == storage; });
    };
    // A sampled view's lookup (cachedTexture): the pending image FindPending serves the surface
    // by, if it fits the view, else the cached storage image of the surface's key, refreshed; the
    // view stays the lookup's answer only when that image is its own source. `sourceEligible` is
    // recorded whichever branch answers: a later refresh of this call may flush the source, and
    // the re-run then judges the surface by the other branch.
    const auto refreshView = [&](std::size_t i, PendingOverlap& overlap) {
        const auto& surface = validatedTextures[i];
        const auto& source = textures[i]->SharedStorageSource();
        if (source == nullptr || source.get() != surface.source) return OwnRefreshFallback::Snapshot;
        if (!source->Cached()) return OwnRefreshFallback::Uncached;
        if (!StorageImageServesKeys(*source, surface.resource.dccAddress)) return OwnRefreshFallback::Keys;
        const auto address = surface.resource.baseAddress;
        const auto bytes = static_cast<std::size_t>(surface.bytes);
        const bool imported = SampledFromStorageEligible(context, surface.resource, surface.bytes);
        overlap.sourceEligible = imported && SurfaceKey(context, source->Descriptor()) == SurfaceKey(context, surface.resource);
        auto found = StorageTexture::FindPending(address, surface.bytes);
        if (found != nullptr && !Texture::CanCopyFrom(*found, surface.resource)) found.reset();
        if (found != nullptr) {
            if (found != source) return OwnRefreshFallback::ForeignView;
            if (!GuestMemory::UnchangedSince(address, bytes, source->GenerationFor(address, bytes))) {
                countOwnRefresh(false);
                refreshForView(*source, surface.resource);
            }
            return OwnRefreshFallback::Count;
        }
        if (!imported) return OwnRefreshFallback::NotImported;
        if (!overlap.sourceEligible) return OwnRefreshFallback::SurfaceKey;
        countOwnRefresh(false);
        // The cache's image of the key is this source while it is cached (an entry is replaced
        // only after its image left the cache), so the lookup would refresh and return it.
        source->Refresh();
        return OwnRefreshFallback::Count;
    };
    // A storage element's lookup (cachedStorageTexture): the cached image of its surface key,
    // refreshed; the key is the element's words', which mapped to this image at the build.
    const auto refreshStorage = [&](std::size_t i) {
        auto& image = storageTextures[i];
        if (image == nullptr || !image->Cached()) return OwnRefreshFallback::Uncached;
        if (i >= storageKeys.size() || !StorageImageServesKeys(*image, storageKeys[i])) return OwnRefreshFallback::Keys;
        countOwnRefresh(true);
        image->Refresh();
        return OwnRefreshFallback::Count;
    };
    std::size_t textureIndex = 0;
    std::size_t storageIndex = 0;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) return OwnRefreshFallback::Uncached;
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) continue;
            if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) {
                for (std::uint32_t element = 0; element < binding.count; ++element) {
                    const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                    if (IsNullTextureResource(words)) continue;
                    if (textureIndex >= textures.size()) return OwnRefreshFallback::Uncached;
                    const auto index = textureIndex++;
                    const auto overlap = listed(index, false);
                    if (overlap == overlapping.end()) continue;
                    if (const auto fallback = refreshView(index, *overlap); fallback != OwnRefreshFallback::Count) return fallback;
                }
            } else if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
                for (std::uint32_t element = 0; element < binding.count; ++element) {
                    const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                    if (IsNullTextureResource(words)) continue;
                    if (storageIndex >= storageTextures.size()) return OwnRefreshFallback::Uncached;
                    const auto index = storageIndex++;
                    if (listed(index, true) == overlapping.end()) continue;
                    if (const auto fallback = refreshStorage(index); fallback != OwnRefreshFallback::Count) return fallback;
                }
            }
        }
    }
    for (const auto& table : tableStages) {
        for (const auto index : table.elementTextures) {
            if (index == std::numeric_limits<std::size_t>::max()) continue;
            if (index != textureIndex++ || index >= textures.size()) return OwnRefreshFallback::Uncached;
            const auto overlap = listed(index, false);
            if (overlap == overlapping.end()) continue;
            if (const auto fallback = refreshView(index, *overlap); fallback != OwnRefreshFallback::Count) return fallback;
        }
    }
    return OwnRefreshFallback::Count;
}

const ShaderRecompiler::DescriptorBinding* ShaderResources::tableImages(std::span<const CompiledShader> shaders, ShaderRecompiler::ShaderStage stage) {
    for (const auto& shader : shaders) {
        if (shader.stage != stage || shader.program == nullptr) continue;
        for (const auto& binding : shader.program->bindings) {
            if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable && binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) return &binding;
        }
    }
    return nullptr;
}

bool AnyPendingCpuWrites(const Context& context) {
    if (Recorder::InCompletion() || Recorder::Active() == nullptr) return false;
    return (context.copiedWriters != nullptr && !context.copiedWriters->empty()) || !DrawCopiedWriters()->empty() || Recorder::PendingCompletionLabels() != 0;
}

bool CompletePendingCpuWrites(const Context& context, std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || Recorder::InCompletion()) return false;
    auto* recorder = Recorder::Active();
    if (recorder == nullptr) return false;
    const auto drawWriters = DrawCopiedWriters();
    const bool labels = Recorder::PendingCompletionLabels() != 0;
    if ((context.copiedWriters == nullptr || context.copiedWriters->empty()) && drawWriters->empty() && !labels) return false;
    if (!recorder->PendingWriteOverlaps(address, bytes)) return false;
    const auto overlaps = [&](const auto& writers) {
        return std::any_of(writers.begin(), writers.end(), [&](const auto& writer) { return writer->CopiedWritesOverlap(address, bytes); });
    };
    if ((context.copiedWriters != nullptr && overlaps(*context.copiedWriters)) || overlaps(*drawWriters) || (labels && recorder->PendingLabelIn(address, bytes))) {
        Recorder::CountSync(4);
        recorder->SyncThrough(address, bytes);
        return true;
    }
    return false;
}

bool ShaderResources::Revalidate(std::span<const CompiledShader> shaders, ProofReport* report, std::span<const GuestMemorySnapshot> snapshots) {
    if (!addressReuse) return proveBuild(shaders, report);
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto refuse = [&](AddressReuse reason) {
        CountAddressReuse(reason, profile ? static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()) : 0);
        return false;
    };
    CountAddressReuseProof();
    ProofReport proof;
    auto& outcome = report != nullptr ? *report : proof;
    outcome = {ProofPath::Full, ProofFailure::Other};
    if (!reusable) return refuse(AddressReuse::NotReusable);
    const bool shared = SharesUses();
    if (bdaTemplate && !BdaTemplatesOn()) return refuse(AddressReuse::NotReusable);
    if (!shared && InUse()) return refuse(AddressReuse::InUse);
    if (!bdaTemplate) {
        CompleteSnapshotWrites(snapshots);
        if (!guestMemory.SameSnapshots(snapshots)) return refuse(AddressReuse::Snapshots);
    }
    struct Suspend {
        GuestBufferMemory& memory;
        bool armed = true;
        ~Suspend() {
            if (armed) memory.SuspendSpace();
        }
    } suspend{guestMemory};
    try {
        if (!guestMemory.ResumeSpace()) return refuse(AddressReuse::Space);
        if (bdaTemplate) {
            CompleteSnapshotWrites(snapshots);
            if (!guestMemory.SnapshotsServed(snapshots)) return refuse(AddressReuse::Snapshots);
        }
        if (!proveBuild(shaders, &outcome)) return refuse(outcome.failure == ProofFailure::Imports ? AddressReuse::Imports : AddressReuse::Proof);
        if (!guestMemory.SpaceImportsCurrent()) {
            outcome.failure = ProofFailure::Imports;
            return refuse(AddressReuse::Imports);
        }
    } catch (const std::exception&) {
        outcome.failure = ProofFailure::Other;
        return refuse(AddressReuse::Threw);
    }
    suspend.armed = false;
    reuseProved = true;
    if (shared && InUse()) CountSharedAddressUse();
    return true;
}

bool ShaderResources::EpochProofs() {
    return DrawPipelineEnabled();
}

namespace {
void noteEpochProof(bool reused, bool verified, bool failed) {
    static std::atomic<std::uint64_t> reuses{0}, proofs{0}, checks{0}, failures{0};
    if (verified) checks.fetch_add(1, std::memory_order_relaxed);
    if (failed) failures.fetch_add(1, std::memory_order_relaxed);
    const auto count = (reused ? reuses : proofs).fetch_add(1, std::memory_order_relaxed) + 1u;
    static const bool report = std::getenv("APS5_PROFILE_DRAW") != nullptr || std::getenv("APS5_VERIFY_DRAW_PIPELINE") != nullptr;
    if (!report) return;
    if (count % 262144u == 0u || (count & (count - 1u)) == 0u) std::fprintf(stderr, "[epoch-proofs] resource proofs: %llu reused the epoch's, %llu proved; verify: %llu checked, %llu failed; storage refreshes reused %llu\n", static_cast<unsigned long long>(reuses.load()), static_cast<unsigned long long>(proofs.load()), static_cast<unsigned long long>(checks.load()), static_cast<unsigned long long>(failures.load()), static_cast<unsigned long long>(StorageTexture::epochRefreshes.load()));
}
}

bool ShaderResources::proveBuild(std::span<const CompiledShader> shaders, ProofReport* report) {
    if (EpochProofs()) {
        const auto epoch = GuestMemory::CollectEpoch();
        const auto proof = GuestMemory::ProofSerial();
        const auto pending = StorageTexture::PendingSerial();
        if (epoch != 0 && provedEpoch == epoch && provedProof == proof && provedPending == pending) {
            static const bool verify = std::getenv("APS5_VERIFY_DRAW_PIPELINE") != nullptr;
            thread_local std::uint64_t reuses = 0;
            if (verify && ++reuses % 64u == 0u) {
                const bool ok = revalidateFull(shaders, report);
                noteEpochProof(true, true, !ok);
                return ok;
            }
            if (report != nullptr) *report = {ProofPath::Full, ProofFailure::None};
            noteEpochProof(true, false, false);
            return true;
        }
        const bool ok = revalidateFull(shaders, report);
        noteEpochProof(false, false, false);
        if (ok) {
            provedEpoch = epoch;
            provedProof = proof;
            provedPending = pending;
        } else {
            provedEpoch = 0;
        }
        return ok;
    }
    return revalidateFull(shaders, report);
}

bool ShaderResources::revalidateFull(std::span<const CompiledShader> shaders, ProofReport* report) {
    if (report != nullptr) *report = {ProofPath::Full, ProofFailure::Other};
    if (!reusable || shaders.empty()) return false;
    if (std::any_of(storageTextures.begin(), storageTextures.end(), [](const auto& texture) { return texture != nullptr && DepthSurfaceAt(texture->Descriptor().baseAddress); })) return false;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    for (const auto& region : directRegions) {
        static_cast<void>(FlushDepthPlanes(region.begin, static_cast<std::size_t>(region.end - region.begin)));
        CompletePendingCpuWrites(context, region.begin, static_cast<std::size_t>(region.end - region.begin));
    }
    for (const auto& range : describedRanges) {
        if (!range.imageTable) RequireNoMultisampleColorTarget(range.address, static_cast<std::size_t>(range.bytes), "a cached image access");
        CompletePendingCpuWrites(context, range.address, static_cast<std::size_t>(range.bytes));
        if (range.dccAddress != 0) CompletePendingCpuWrites(context, range.dccAddress, DccKeyBytes(range.bytes));
    }
    for (const auto& image : storageTextures) {
        CompletePendingCpuWrites(context, image->Descriptor().baseAddress, static_cast<std::size_t>(image->GuestBytes()));
        if (image->Descriptor().dccAddress != 0) CompletePendingCpuWrites(context, image->Descriptor().dccAddress, DccKeyBytes(image->GuestBytes()));
    }
    // APS5_NO_FAST_REVALIDATE=1 always repeats the lookups.
    static const bool noFast = std::getenv("APS5_NO_FAST_REVALIDATE") != nullptr;
    const auto finish = [&](bool fast, bool ok, ProofFailure failure = ProofFailure::Other) {
        if (profile) countRevalidate(fast, ok, start);
        if (report != nullptr) report->failure = ok ? ProofFailure::None : failure;
        return ok;
    };
    // Loaded before any check (the memo rule of fastRevalidate): kept as the memo only when the
    // registry did not move through the whole proof, flushes of this call included.
    const auto serialBefore = StorageTexture::PendingSerial();
    // (1) Textures and storage images, from stamps when every element allows it (fastRevalidate),
    // else by the same lookups as the build (which refresh or replace them as guest memory changed):
    // they must hand back the very objects the set's views belong to. The build appended them stage
    // by stage, binding by binding, so the walk repeats that order.
    const auto fullWalk = [&] {
        lastFullWalk = std::chrono::steady_clock::now();
        lookupLog().clear();
        BuildLookups lookups;
        std::size_t textureIndex = 0;
        std::size_t storageIndex = 0;
        for (const auto& shader : shaders) {
            if (shader.program == nullptr) return false;
            for (const auto& binding : shader.program->bindings) {
                if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) continue;
                if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) {
                    const auto elementWords = binding.count != 0 ? binding.guestDescriptor.size() / binding.count : 0;
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        const auto raw = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
                        std::array<std::uint32_t, 8> narrowed{};
                        const auto words = firstLayerWords(raw, binding.imageShape, narrowed, false);
                        if (IsNullTextureResource(words)) continue;
                        const auto resource = DecodeTextureResource(words);
                        const VkComponentMapping components = ViewComponents(resource);
                        const VkImageViewType multisampleViewType = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2DMultisampleArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
                        if (textureIndex >= textures.size() || cachedTexture(context, words, resource, components, 0, !binding.imageDepthCompare.empty() && binding.imageDepthCompare.at(element), multisampleViewType) != textures[textureIndex]) return false;
                        ++textureIndex;
                    }
                } else if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                        if (IsNullTextureResource(words)) continue;
                        if (storageIndex >= storageTextures.size()) return false;
                        std::shared_ptr<StorageTexture> expected;
                        if (SameAsPreviousStorageElement(binding, element) && StorageDedupeEnabled()) expected = storageTextures[storageIndex - 1];
                        else expected = cachedStorageTexture(context, words, DecodeTextureResource(words), storageMips[storageIndex], 0, storageWritten[storageIndex]);
                        if (expected != storageTextures[storageIndex]) return false;
                        ++storageIndex;
                    }
                }
            }
        }
        for (const auto& table : tableStages) {
            if (table.elementTextures.empty()) continue;
            const auto* images = tableImages(shaders, table.stage);
            if (images == nullptr || images->count != table.elementTextures.size()) return false;
            std::vector<std::shared_ptr<Texture>> cached;
            if (auto* reused = FindReusedTable(context.device, *images, CurrentTableState()); reused != nullptr && TableCurrent(*reused) && CachedTableTextures(*reused, cached)) {
                auto next = cached.begin();
                bool same = true;
                for (std::uint32_t element = 0; same && element < images->count; ++element) {
                    const auto& entry = reused->elements[element];
                    const auto index = table.elementTextures[element];
                    if (entry.rejected) {
                        same = index == std::numeric_limits<std::size_t>::max();
                        continue;
                    }
                    const auto& texture = *next++;
                    if (index == std::numeric_limits<std::size_t>::max()) continue;
                    same = index == textureIndex && textureIndex < textures.size() && textures[textureIndex] == texture;
                    ++textureIndex;
                }
                if (!same) return false;
                TextureCounts().tablesReused.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            for (std::uint32_t element = 0; element < images->count; ++element) {
                if (table.elementTextures[element] == std::numeric_limits<std::size_t>::max()) continue;
                const auto raw = std::span<const std::uint32_t>(images->guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                std::array<std::uint32_t, 8> narrowed{};
                const auto words = firstLayerWords(raw, images->imageShapes.at(element), narrowed, false);
                const auto resource = DecodeTextureResource(words);
                const VkImageViewType multisampleViewType = images->imageShapes.at(element) == ShaderRecompiler::DescriptorImageShape::Image2DMultisampleArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
                if (textureIndex >= textures.size() || table.elementTextures[element] != textureIndex || cachedTexture(context, words, resource, ViewComponents(resource), 0, images->imageDepthCompare.at(element), multisampleViewType) != textures[textureIndex]) return false;
                ++textureIndex;
            }
        }
        if (textureIndex != textures.size() || storageIndex != storageTextures.size()) return false;
        // The walk proved every object current again: the records move to what it proved, or one
        // spurious stamp (a label sharing a 64 KiB block with a surface's edge) would keep this
        // object on the full walk for good.
        captureValidation();
        return true;
    };
    thread_local std::vector<PendingOverlap>* overlappingSlot = nullptr;
    auto& overlapping = ShaderRecompiler::ThreadOwned(overlappingSlot);
    thread_local std::vector<PendingOverlap>* refreshedSlot = nullptr;
    auto& refreshed = ShaderRecompiler::ThreadOwned(refreshedSlot);
    refreshed.clear();
    FastFail reason = FastFail::Count;
    bool accepted = false;
    bool fast = !noFast && fastRevalidate(serialBefore, refreshed, reason, overlapping, accepted);
    {
        static const bool verifyUntracked = std::getenv("APS5_VERIFY_UNTRACKED_PROOFS") != nullptr;
        const bool anyUntracked = std::any_of(validatedTextures.begin(), validatedTextures.end(), [](const ValidatedSurface& surface) { return surface.untracked[0] != 0; }) || std::any_of(storageUntracked.begin(), storageUntracked.end(), [](const auto& stamp) { return stamp[0] != 0; });
        if (verifyUntracked && fast && anyUntracked) {
            static std::atomic<std::uint64_t> checked{0}, disagreed{0};
            const auto saved = validatedTextures;
            const auto savedStorage = storageUntracked;
            if (!fullWalk()) disagreed.fetch_add(1, std::memory_order_relaxed);
            validatedTextures = saved;
            storageUntracked = savedStorage;
            if (checked.fetch_add(1, std::memory_order_relaxed) % 100000 == 99999) std::fprintf(stderr, "[verify-untracked] %llu fast proofs over untracked memory checked by the full walk, %llu disagreed%c", static_cast<unsigned long long>(checked.load()), static_cast<unsigned long long>(disagreed.load()), 10);
        }
    }
    // T1 (design_cpu_final M3, rule RT1): a Pending failure whose overlapping images are foreign
    // to the surfaces is resolved by the own objects' refresh (what the walk's lookups would do to
    // them) and the fast proof run again, which is then authoritative (it accepts what stays
    // pending over the refreshed surfaces, see fastRevalidate); a fallback, or a second failure,
    // takes the full walk as before.
    bool ownRefreshed = false;
    if (!fast && !overlapping.empty()) {
        auto fallback = OwnRefreshFallback::Count;
        if (!OwnImageRefreshEnabled()) fallback = OwnRefreshFallback::Disabled;
        else {
            refreshed = overlapping;
            fallback = refreshOwnObjects(shaders, refreshed);
        }
        if (fallback == OwnRefreshFallback::Count) {
            fast = fastRevalidate(serialBefore, refreshed, reason, overlapping, accepted);
            if (fast) ownRefreshed = true;
            else fallback = OwnRefreshFallback::Rerun;
        }
        if (fallback != OwnRefreshFallback::Count) countOwnRefreshFallback(fallback);
    }
    if (fast && TemplateAuditDue(lastFullWalk)) {
        fast = false;
        ownRefreshed = false;
        reason = FastFail::Audit;
    }
    if (report != nullptr) report->path = !fast ? ProofPath::Full : ownRefreshed ? ProofPath::OwnRefreshed : ProofPath::Fast;
    if (!fast) {
        countFullWalk(reason);
        if (!fullWalk()) {
            const auto failure = [&] {
                switch (reason) {
                    case FastFail::Pending: return ProofFailure::Pending;
                    case FastFail::Evicted: case FastFail::Replaced: return ProofFailure::Evicted;
                    case FastFail::Changed: return ProofFailure::Changed;
                    case FastFail::Keys: case FastFail::ClearedView: case FastFail::StorageKeys: return ProofFailure::Keys;
                    default: return ProofFailure::Other;
                }
            }();
            return finish(false, false, failure);
        }
    } else if (ownRefreshed || accepted) {
        if (profile && ownRefreshed) Revalidations().ownRefreshed.fetch_add(1, std::memory_order_relaxed);
        if (VerifyProofs()) {
            // The walk after the refresh, or beside a proof that accepted a foreign overlap, must
            // find every object in place and current: another object, or an upload (a moved
            // content version), is a decision the proof got wrong.
            thread_local std::vector<std::pair<const StorageTexture*, std::uint64_t>>* versionsSlot = nullptr;
            auto& versions = ShaderRecompiler::ThreadOwned(versionsSlot);
            versions.clear();
            for (const auto& surface : validatedTextures) {
                if (surface.source != nullptr) versions.emplace_back(surface.source, surface.source->Version());
            }
            for (const auto& image : storageTextures) versions.emplace_back(image.get(), image->Version());
            const bool same = fullWalk();
            const auto moved = std::find_if(versions.begin(), versions.end(), [](const auto& entry) { return entry.first->Version() != entry.second; });
            if (!same || moved != versions.end()) {
                std::fprintf(stderr, "[rescache] APS5_VERIFY_PROOFS: the T1 proof (%s) disagrees with the full walk (%s)\n", ownRefreshed ? "own-object refresh" : "accepted overlap", !same ? "another object" : "an upload");
                std::fflush(stderr);
                std::abort();
            }
            if (profile) Revalidations().proofsVerified.fetch_add(1, std::memory_order_relaxed);
        }
    }
    // (2) The imported buffers the set reads in place: results of storage images pending in them go
    // to guest memory first (as an upload does), then each import must still be the one the set was
    // written against. This comes last because the lookups and flushes above can reconcile imports
    // themselves; nothing else touches them between here and the dispatch being recorded. Under the
    // epoch gate the flush runs only for regions one registry scan finds pending images over (none
    // while the serial is the memo's), and the serial loop only while the import table's identity
    // moved since the last proof (a retire bumps its epoch, a registry change its generation).
    const auto serialLoop = [&] {
        for (const auto& region : directRegions) {
            auto serial = HostImportSerial(context, region.begin, static_cast<std::size_t>(region.end - region.begin), true);
            if (serial == 0) serial = ImageMirrorSerial(context, region.begin, static_cast<std::size_t>(region.end - region.begin));
            if (serial != region.serial) return false;
            if (region.regionEnd != 0) {
                const auto committed = GuestMemory::DescribeCommitted(region.begin, static_cast<std::size_t>(region.regionEnd - region.begin));
                if (committed.ranges.size() != 1 || committed.ranges.front().first != region.begin || committed.ranges.front().second != region.committedEnd) return false;
            }
        }
        return true;
    };
    if (EpochRevalidate()) {
        if ((!directRegions.empty() || !guestMemory.Writes().empty()) && !(pendingSerialSeen != 0 && pendingSerialSeen == StorageTexture::PendingSerial())) {
            thread_local std::vector<StorageTexture::PendingQuery>* regionsSlot = nullptr;
            auto& regions = ShaderRecompiler::ThreadOwned(regionsSlot);
            regions.clear();
            for (const auto& region : directRegions) regions.push_back({region.begin, region.end, nullptr, nullptr, false});
            for (const auto& [begin, end] : guestMemory.Writes()) regions.push_back({begin, end, nullptr, nullptr, false});
            StorageTexture::ScanPending(regions);
            // A region over a unit shadow's fresh results (a retile bumped the serial) is
            // published by the flush, as one over a pending image is stored.
            for (const auto& region : regions) {
                if (region.overlaps || AnyShadowedOverlaps(region.begin, static_cast<std::size_t>(region.end - region.begin))) StorageTexture::FlushPending(region.begin, static_cast<std::size_t>(region.end - region.begin), nullptr, "imported buffer region");
            }
        }
        if (HostImportsUnchanged(context, importsProof)) {
            if (profile) Revalidations().importSkips.fetch_add(1, std::memory_order_relaxed);
        } else {
            if (!serialLoop()) return finish(fast, false, ProofFailure::Imports);
            importsProof = HostImportsIdentity(context);
        }
    } else {
        for (const auto& region : directRegions) StorageTexture::FlushPending(region.begin, static_cast<std::size_t>(region.end - region.begin), nullptr, "imported buffer region");
        for (const auto& [begin, end] : guestMemory.Writes()) StorageTexture::FlushPending(begin, static_cast<std::size_t>(end - begin), nullptr, "written buffer range");
        if (!serialLoop()) return finish(fast, false, ProofFailure::Imports);
    }
    // (3) Buffers staged in device memory (GuestBufferMemory::AllowDeviceStaging) are copied in
    // from their imports anew for this use, after the flushes above and before the work is
    // recorded; a failure to record leaves the object unusable for this dispatch, not the batch.
    if (auto* recorder = Recorder::Active(); recorder != nullptr) {
        try {
            guestMemory.RecordStagingCopies(*recorder);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[resources] staging copies of a reused build failed: %s\n", error.what());
            return finish(fast, false);
        }
    }
    // (4) The sampled textures are used again: their cache entries move to the front of the
    // eviction order and take this batch's serial, like a lookup's hit. A texture that left the
    // cache meanwhile (evicted while another holder kept this object) fails the proof: the
    // rebuild takes a cached one, so the set never pins what the cache does not account.
    if (!touchSampledTextures(textures)) return finish(fast, false, ProofFailure::Evicted);
    pendingSerialSeen = EpochRevalidate() && StorageTexture::PendingSerial() == serialBefore ? serialBefore : 0;
    for (const auto& texture : textures) {
        if (texture != nullptr && texture->IsDepthCopy()) texture->RefreshDepthCopy();
    }
    return finish(fast, true);
}

namespace {

// APS5_PROFILE_DRAW: what makes content keys miss. On a miss the key is compared with the last key
// seen for the same shader variant(s) and the first differing word is charged to its binding's role
// (and to its index within the V# or T# element), printed as [rescache] every 10 s: it says whether
// V# bases of ring allocations, SRT words or image descriptors churn.
struct KeyMissWindow {
    std::uint64_t compared = 0;
    std::uint64_t noCandidate = 0;
    std::uint64_t unlabelled = 0;
    std::uint64_t identical = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(ContentKeyComponent::Count)> differing{};
};

struct ChurnCounts {
    std::array<std::uint64_t, 8> roles{};
    std::array<std::uint64_t, 4> bufferWords{};
    std::array<std::uint64_t, 8> imageWords{};
    std::uint64_t firstSeen = 0;
    std::uint64_t sameKey = 0;
    std::uint64_t layout = 0;
    std::uint64_t drawWords = 0;
    KeyMissWindow window;
};

// Dispatch keys and draw keys are counted apart: what churns a draw's key decides the draw steps.
struct ChurnProfile {
    AgcDriver::Mutex mutex;
    std::unordered_map<std::uint64_t, ResourceCache::Key> lastByVariant;
    ChurnCounts dispatch;
    ChurnCounts draws;
    std::unordered_map<std::uint64_t, std::uint32_t> tracedByVariant;
    std::size_t tracedLines = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

constexpr std::size_t MaxTracedLines = 256;
constexpr std::uint32_t MaxTracedLinesPerVariant = 4;

const char* componentName(std::size_t component) {
    constexpr std::array<const char*, static_cast<std::size_t>(ContentKeyComponent::Count)> names{"layout", "V# word 0", "V# word 1", "V# word 2", "V# word 3", "image", "sampler", "shader-data", "other", "draw target/index"};
    return names[component];
}

void traceMiss(ChurnProfile& churn, bool draw, std::uint64_t variants, const ContentKeyDiff* parts);

ChurnProfile& Churn() {
    static ChurnProfile profile;
    return profile;
}

// Charges word `diff` of `key` to the binding of the ContentKey that starts at `at` (the layout of
// ShaderResources::ContentKey), moving `at` past that key; false when `diff` lies beyond it.
bool chargeShaderKey(ChurnCounts& profile, const ResourceCache::Key& key, std::size_t& at, std::size_t diff) {
    if (at + ContentKeyHeaderWords > key.size() || diff < at + ContentKeyHeaderWords) {
        ++profile.layout;
        return true;
    }
    const bool dataWords = key[at + ContentKeyDataWordsAt] != 0;
    const auto bindings = key[at + ContentKeyBindingsAt];
    at += ContentKeyHeaderWords;
    for (std::uint32_t binding = 0; binding < bindings; ++binding) {
        if (at + 8 > key.size() || diff < at + 8) {
            ++profile.layout;
            return true;
        }
        const auto kind = key[at];
        const auto role = key[at + 1];
        const auto descriptorWords = dataWords || !DataRole(static_cast<ShaderRecompiler::DescriptorRole>(role)) ? key[at + 7] : 0u;
        at += 8;
        if (diff < at + descriptorWords) {
            if (role < profile.roles.size()) ++profile.roles[role];
            const auto index = diff - at;
            if (role == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorRole::GuestBuffers)) ++profile.bufferWords[index % 4];
            else if (role == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorRole::GuestImages) && (kind == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorKind::SampledImage) || kind == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorKind::StorageImage))) ++profile.imageWords[index % 8];
            return true;
        }
        at += descriptorWords;
        for (std::size_t flags = 0; flags < ContentKeyFlagSets + 2u; ++flags) {
            if (at >= key.size()) {
                ++profile.layout;
                return true;
            }
            const auto words = flags < ContentKeyFlagSets ? 1 + (static_cast<std::size_t>(key[at]) + 31) / 32 : 1 + static_cast<std::size_t>(key[at]);
            if (diff < at + words) {
                ++profile.layout;
                return true;
            }
            at += words;
        }
    }
    return false;
}

}

std::shared_ptr<ShaderResources> ResourceCache::Peek(const Key& key) {
    std::lock_guard lock(mutex);
    const auto found = index.find(key);
    return found == index.end() ? nullptr : found->second->second;
}

std::shared_ptr<ShaderResources> ResourceCache::Find(const Key& key) {
    resourceCacheFinds.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(mutex);
    const auto found = index.find(key);
    if (found == index.end()) {
        noteReturn(key);
        noteMiss(key);
        noteComputeLookup(key, false);
        return nullptr;
    }
    entries.splice(entries.begin(), entries, found->second);
    noteUse(key);
    noteComputeLookup(key, true);
    return found->second->second;
}

bool ResourceCache::computeEntry(const std::pair<Key, std::shared_ptr<ShaderResources>>& entry) const {
    return ComputeKey(entry.first) && !entry.second->ReusesAddressSpace();
}

namespace {

std::uint64_t PinnedTextureBytes(const ShaderResources& resources) {
    std::uint64_t bytes = 0;
    const auto& textures = resources.SampledTextures();
    for (std::size_t i = 0; i < textures.size(); ++i) {
        if (textures[i] == nullptr) continue;
        if (std::any_of(textures.begin(), textures.begin() + static_cast<std::ptrdiff_t>(i), [&](const auto& earlier) { return earlier == textures[i]; })) continue;
        bytes += textures[i]->AllocationBytes();
    }
    return bytes;
}

}

bool ResourceCache::ComputeBudgetRequested() {
    static const bool requested = std::getenv("APS5_COMPUTE_TEMPLATE_ENTRIES") != nullptr;
    return requested;
}

bool ResourceCache::computeSplit() const {
    return ComputeBudgetRequested() && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoComputeTemplateBudget);
}

void ResourceCache::noteFate(const Key& key, std::uint8_t fate) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile || !ComputeKey(key)) return;
    if (computeFates.size() >= (1u << 20u)) computeFates.clear();
    computeFates[KeyHash{}(key)] = fate;
}

void ResourceCache::NoteNotInserted(const Key& key, std::size_t refusal) {
    std::lock_guard lock(mutex);
    noteFate(key, static_cast<std::uint8_t>(FateNotReusable + std::min<std::size_t>(refusal, fateCounts.notReusable.size() - 1u)));
}

void ResourceCache::noteComputeLookup(const Key& key, bool hit) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile || !ComputeKey(key)) return;
    auto& counts = fateCounts;
    if (hit) {
        ++counts.hits;
    } else {
        ++counts.misses;
        const auto found = computeFates.find(KeyHash{}(key));
        const auto fate = found == computeFates.end() ? std::uint8_t{FateUnknown} : found->second;
        switch (fate) {
            case FateUnknown: ++counts.neverSeen; break;
            case FatePresent: ++counts.present; break;
            case FateEvicted: ++counts.evicted; break;
            case FateEvictedByDraw: ++counts.evicted; ++counts.evictedByDraws; break;
            case FateDropped: ++counts.dropped; break;
            case FateRemoved: ++counts.removed; break;
            case FateRefusedStorage: ++counts.refusedStorage; break;
            case FateRefusedTexture: ++counts.refusedTexture; break;
            default: ++counts.notReusable[std::min<std::size_t>(fate - FateNotReusable, counts.notReusable.size() - 1u)]; break;
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - fateReport < std::chrono::seconds(10)) return;
    fateReport = now;
    std::string refused;
    for (std::size_t reason = 0; reason < counts.notReusable.size(); ++reason) {
        if (counts.notReusable[reason] == 0) continue;
        char item[96];
        std::snprintf(item, sizeof(item), " %s %llu", reason < static_cast<std::size_t>(ShaderResources::TemplateOutcome::Count) ? ShaderResources::TemplateOutcomeName(static_cast<ShaderResources::TemplateOutcome>(reason)) : "other", static_cast<unsigned long long>(counts.notReusable[reason]));
        refused += item;
    }
    const auto total = counts.hits + counts.misses;
    AgcDriver::ProfilePrint_nid_no_patch("[rescache] compute templates (10 s): %llu lookups, %llu hits (%.1f %%); misses by the key's last fate: never cached %llu, evicted over the bound %llu (by a draw's insert %llu), dropped with an image or texture %llu, invalidated %llu, refused at insert: uncached storage image %llu, released texture %llu, not reusable:%s; still present %llu; compute entries %zu, bound %zu, sampled textures they pin %llu MiB of %llu MiB (%s)\n", static_cast<unsigned long long>(total), static_cast<unsigned long long>(counts.hits), total != 0 ? 100.0 * static_cast<double>(counts.hits) / static_cast<double>(total) : 0.0, static_cast<unsigned long long>(counts.neverSeen), static_cast<unsigned long long>(counts.evicted), static_cast<unsigned long long>(counts.evictedByDraws), static_cast<unsigned long long>(counts.dropped), static_cast<unsigned long long>(counts.removed), static_cast<unsigned long long>(counts.refusedStorage), static_cast<unsigned long long>(counts.refusedTexture), refused.empty() ? " none" : refused.c_str(), static_cast<unsigned long long>(counts.present), computeEntries, computeSplit() ? computeCapacity : capacity, static_cast<unsigned long long>(computePinnedBytes >> 20u), static_cast<unsigned long long>(computeByteBudget >> 20u), computeSplit() ? "own bound, APS5_COMPUTE_TEMPLATE_ENTRIES / _MIB" : "shared with draws");
    counts = {};
}

ResourceCache::ComputeFates ResourceCache::ComputeFateCounts() const {
    std::lock_guard lock(mutex);
    auto counts = fateCounts;
    counts.entries = computeEntries;
    counts.split = computeSplit();
    counts.capacity = counts.split ? computeCapacity : capacity;
    counts.pinnedBytes = computePinnedBytes;
    counts.byteBudget = computeByteBudget;
    return counts;
}

void ResourceCache::noteEvicted(const Key& key) {
    if (fixedCapacity) return;
    const auto hash = KeyHash{}(key);
    if (const auto found = evictedKeys.find(hash); found != evictedKeys.end()) evictedOrder.erase(found->second);
    evictedOrder.push_front(hash);
    evictedKeys[hash] = evictedOrder.begin();
    while (evictedOrder.size() > capacity) {
        evictedKeys.erase(evictedOrder.back());
        evictedOrder.pop_back();
    }
    if (++evictionsSinceCheck < capacity / 4u) return;
    const bool churning = returnsSinceCheck * 2u > evictionsSinceCheck;
    evictionsSinceCheck = 0;
    returnsSinceCheck = 0;
    if (!churning || capacity >= MaximumEntries) return;
    const auto previous = capacity;
    capacity = std::min(capacity * 2u, MaximumEntries);
    if (shrankLast) shrinkWindow = std::min(shrinkWindow * 2u, MaximumShrinkWindow);
    shrankLast = false;
    usedKeys.clear();
    usesSinceCheck = 0;
    std::fprintf(stderr, "[rescache] capacity %zu -> %zu entries: keys it evicted keep coming back\n", previous, capacity);
}

void ResourceCache::noteUse(const Key& key) {
    if (fixedCapacity || capacity <= baseEntries) return;
    usedKeys.insert(KeyHash{}(key));
    if (++usesSinceCheck < capacity * shrinkWindow) return;
    const auto used = usedKeys.size();
    const auto uses = usesSinceCheck;
    usedKeys.clear();
    usesSinceCheck = 0;
    if (used * 4u > capacity) return;
    const auto previous = capacity;
    capacity = std::max(capacity / 2u, baseEntries);
    shrankLast = true;
    evictionsSinceCheck = 0;
    returnsSinceCheck = 0;
    std::fprintf(stderr, "[rescache] capacity %zu -> %zu entries: %zu keys served the last %zu uses\n", previous, capacity, used, uses);
}

void ResourceCache::noteReturn(const Key& key) {
    if (fixedCapacity) return;
    const auto found = evictedKeys.find(KeyHash{}(key));
    if (found == evictedKeys.end()) return;
    evictedOrder.erase(found->second);
    evictedKeys.erase(found);
    ++returnsSinceCheck;
}

namespace {

void traceMiss(ChurnProfile& churn, bool draw, std::uint64_t variants, const ContentKeyDiff* parts) {
    if (churn.tracedLines >= MaxTracedLines || ++churn.tracedByVariant[variants] > MaxTracedLinesPerVariant) return;
    ++churn.tracedLines;
    std::string changed = parts == nullptr ? "first-seen" : "";
    for (std::size_t i = 0; parts != nullptr && i < parts->size(); ++i) {
        if (!(*parts)[i]) continue;
        if (!changed.empty()) changed += ", ";
        changed += componentName(i);
    }
    if (changed.empty()) changed = "no-component-change";
    const auto hashed = (variants ^ 0x9e3779b97f4a7c15ull) * 0xff51afd7ed558ccdull;
    AgcDriver::ProfilePrint_nid_no_patch("[keymiss] resource-content %s shader set %016llx: %s\n", draw ? "draw" : "dispatch", static_cast<unsigned long long>(hashed ^ (hashed >> 33u)), changed.c_str());
}

}

void ResourceCache::noteMiss(const Key& key) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static const bool trace = std::getenv("APS5_TRACE_RESOURCE_KEY_MISSES") != nullptr;
    if ((!profile && !trace) || key.size() < ContentKeyHeaderWords) return;
    // The variant(s) the key belongs to: a compute key starts with the data-words flag, the stage
    // and its variant, a draw key with a marker, the device and its stages' keys (Draw.cpp
    // DrawResourceKey).
    const bool draw = key[0] == 0xffffffffu;
    std::uint64_t variants = 0;
    if (draw) {
        std::size_t at = 4;
        for (std::uint32_t stage = 0; stage < key[3] && at + 2 + ContentKeyVariantAt < key.size(); ++stage) {
            variants = variants * 1000003ull ^ (key[at + 1 + ContentKeyVariantAt] | (static_cast<std::uint64_t>(key[at + 2 + ContentKeyVariantAt]) << 32u));
            at += 1 + key[at];
        }
    } else {
        variants = key[ContentKeyVariantAt] | (static_cast<std::uint64_t>(key[ContentKeyVariantAt + 1]) << 32u);
    }
    auto& churn = Churn();
    std::lock_guard lock(churn.mutex);
    auto& counts = draw ? churn.draws : churn.dispatch;
    const auto found = churn.lastByVariant.find(variants);
    if (found == churn.lastByVariant.end()) {
        ++counts.firstSeen;
        ++counts.window.noCandidate;
        if (trace) traceMiss(churn, draw, variants, nullptr);
        churn.lastByVariant.emplace(variants, key);
    } else {
        const auto& previous = found->second;
        const auto common = std::min(key.size(), previous.size());
        std::size_t diff = 0;
        while (diff < common && key[diff] == previous[diff]) ++diff;
        if (diff == common && key.size() == previous.size()) {
            ++counts.sameKey;
        } else if (draw) {
            std::size_t at = 4;
            bool charged = false;
            for (std::uint32_t stage = 0; stage < key[3] && at < key.size() && !charged; ++stage) {
                ++at;
                charged = chargeShaderKey(counts, key, at, diff);
            }
            if (!charged) ++counts.drawWords;
        } else {
            std::size_t at = 0;
            if (!chargeShaderKey(counts, key, at, diff)) ++counts.layout;
        }
        if (trace) {
            ContentKeyDiff parts;
            ++counts.window.compared;
            if (!ShaderResources::DiffContentKeys(previous, key, draw, parts)) ++counts.window.unlabelled;
            bool anyPart = false;
            for (std::size_t i = 0; i < parts.size(); ++i) {
                if (!parts[i]) continue;
                ++counts.window.differing[i];
                anyPart = true;
            }
            if (!anyPart) ++counts.window.identical;
            traceMiss(churn, draw, variants, &parts);
        }
        found->second = key;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - churn.lastReport < std::chrono::seconds(10)) return;
    churn.lastReport = now;
    const auto report = [](const char* what, const ChurnCounts& churn) {
        std::string line = std::string("[rescache] miss churn (") + what + "), first differing word by role:";
        char item[256];
        for (std::size_t role = 0; role < churn.roles.size(); ++role) {
            if (churn.roles[role] == 0) continue;
            std::snprintf(item, sizeof(item), " %s %llu", roleName(static_cast<ShaderRecompiler::DescriptorRole>(role)), static_cast<unsigned long long>(churn.roles[role]));
            line += item;
            if (role == static_cast<std::size_t>(ShaderRecompiler::DescriptorRole::GuestBuffers)) {
                std::snprintf(item, sizeof(item), " (V# word %llu/%llu/%llu/%llu)", static_cast<unsigned long long>(churn.bufferWords[0]), static_cast<unsigned long long>(churn.bufferWords[1]), static_cast<unsigned long long>(churn.bufferWords[2]), static_cast<unsigned long long>(churn.bufferWords[3]));
                line += item;
            } else if (role == static_cast<std::size_t>(ShaderRecompiler::DescriptorRole::GuestImages)) {
                std::snprintf(item, sizeof(item), " (T# word %llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu)", static_cast<unsigned long long>(churn.imageWords[0]), static_cast<unsigned long long>(churn.imageWords[1]), static_cast<unsigned long long>(churn.imageWords[2]), static_cast<unsigned long long>(churn.imageWords[3]), static_cast<unsigned long long>(churn.imageWords[4]), static_cast<unsigned long long>(churn.imageWords[5]), static_cast<unsigned long long>(churn.imageWords[6]), static_cast<unsigned long long>(churn.imageWords[7]));
                line += item;
            }
        }
        std::snprintf(item, sizeof(item), "; layout %llu, draw target/index %llu, same key %llu (not inserted or evicted), first seen %llu", static_cast<unsigned long long>(churn.layout), static_cast<unsigned long long>(churn.drawWords), static_cast<unsigned long long>(churn.sameKey), static_cast<unsigned long long>(churn.firstSeen));
        line += item;
        AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
    };
    if (profile) {
        report("dispatch", churn.dispatch);
        report("draws", churn.draws);
    }
    const auto keyMiss = [](const char* what, KeyMissWindow& window) {
        const auto count = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
        const auto part = [&](ContentKeyComponent component) { return count(window.differing[static_cast<std::size_t>(component)]); };
        AgcDriver::ProfilePrint_nid_no_patch("[keymiss] resource-content %s (10 s): misses %llu; differing: layout %llu, V# word0 %llu, V# word1 %llu, V# word2 %llu, V# word3 %llu, image %llu, sampler %llu, shader-data %llu, other %llu, draw target/index %llu; no-candidate %llu; identical %llu; not comparable %llu\n", what, count(window.compared + window.noCandidate), part(ContentKeyComponent::Layout), part(ContentKeyComponent::BufferWord0), part(ContentKeyComponent::BufferWord1), part(ContentKeyComponent::BufferWord2), part(ContentKeyComponent::BufferWord3), part(ContentKeyComponent::Image), part(ContentKeyComponent::Sampler), part(ContentKeyComponent::ShaderData), part(ContentKeyComponent::Other), part(ContentKeyComponent::DrawTarget), count(window.noCandidate), count(window.identical), count(window.unlabelled));
        window = KeyMissWindow{};
    };
    if (trace) {
        keyMiss("dispatch", churn.dispatch.window);
        keyMiss("draw", churn.draws.window);
    }
}

void ResourceCache::Insert(const Key& key, std::shared_ptr<ShaderResources> resources, std::vector<std::shared_ptr<ShaderResources>>* evicted) {
    std::lock_guard lock(mutex);
    if (!resources->HasCachedStorageImages()) {
        noteFate(key, FateRefusedStorage);
        return;
    }
    if (const auto found = index.find(key); found != index.end()) drop(found->second, evicted);
    // A set binding a sampled texture the sampled cache let go would keep its image alive outside
    // the cache's budget: it serves its own use only. Checked under this mutex, after which an
    // eviction's RemoveTextures finds the entry (the eviction marks the texture first).
    if (resources->BindsReleasedTexture()) {
        SampledAtomics().refusedSets.fetch_add(1, std::memory_order_relaxed);
        noteFate(key, FateRefusedTexture);
        return;
    }
    resources->PinTextures(true);
    if (resources->ReusesAddressSpace()) ++addressEntries;
    else if (ComputeKey(key)) {
        ++computeEntries;
        computePinnedBytes += PinnedTextureBytes(*resources);
    }
    noteFate(key, FatePresent);
    entries.emplace_front(key, std::move(resources));
    index.emplace(key, entries.begin());
    indexTextures(entries.begin(), true);
    if (capacity == 0) {
        const char* value = std::getenv("APS5_RESOURCE_CACHE_ENTRIES");
        const auto parsed = value ? std::strtoull(value, nullptr, 10) : 0ull;
        fixedCapacity = parsed != 0;
        capacity = fixedCapacity ? static_cast<std::size_t>(parsed) : baseEntries;
    }
    if (addressCapacity == 0) {
        const char* value = std::getenv("APS5_ADDRESS_TEMPLATE_ENTRIES");
        const auto parsed = value ? std::strtoull(value, nullptr, 10) : 0ull;
        addressCapacity = parsed != 0 ? static_cast<std::size_t>(parsed) : 2048u;
    }
    if (computeCapacity == 0) {
        const char* value = std::getenv("APS5_COMPUTE_TEMPLATE_ENTRIES");
        const auto parsed = value ? std::strtoull(value, nullptr, 10) : 0ull;
        computeCapacity = parsed != 0 ? static_cast<std::size_t>(parsed) : 4096u;
        const char* mib = std::getenv("APS5_COMPUTE_TEMPLATE_MIB");
        const auto parsedMib = mib ? std::strtoull(mib, nullptr, 10) : 0ull;
        computeByteBudget = static_cast<std::uint64_t>(parsedMib != 0 ? parsedMib : 256ull) << 20u;
    }
    noteUse(key);
    const bool split = computeSplit();
    const bool drawInsert = !ComputeKey(key);
    enum class Pool { Shared, Address, Compute };
    const auto poolOf = [&](const std::pair<Key, std::shared_ptr<ShaderResources>>& entry) { return entry.second->ReusesAddressSpace() ? Pool::Address : split && ComputeKey(entry.first) ? Pool::Compute : Pool::Shared; };
    for (;;) {
        const auto shared = entries.size() - addressEntries - (split ? computeEntries : 0u);
        const bool computeOver = split && (computeEntries > computeCapacity || computePinnedBytes > computeByteBudget);
        const auto pool = addressEntries > addressCapacity ? Pool::Address : computeOver ? Pool::Compute : Pool::Shared;
        if (!(addressEntries > addressCapacity || computeOver || shared > capacity)) break;
        auto victim = std::prev(entries.end());
        while (victim != entries.begin() && poolOf(*victim) != pool) --victim;
        if (poolOf(*victim) != pool) break;
        noteEvicted(victim->first);
        noteFate(victim->first, drawInsert ? FateEvictedByDraw : FateEvicted);
        drop(victim, evicted);
    }
    if (entries.front().second->TableElements() == 0) return;
    std::uint64_t tables = 0;
    for (const auto& entry : entries) tables += entry.second->TableElements();
    for (auto it = std::prev(entries.end()); tables > TableElementBudget() && it != entries.begin();) {
        const auto elements = it->second->TableElements();
        if (elements == 0) {
            --it;
            continue;
        }
        tables -= elements;
        const auto victim = it--;
        noteFate(victim->first, FateEvicted);
        drop(victim, evicted);
    }
}

void ResourceCache::Remove(const Key& key, const ShaderResources* object) {
    std::lock_guard lock(mutex);
    if (object != nullptr) {
        const auto found = index.find(key);
        if (found == index.end() || found->second->second.get() != object) return;
    }
    erase(key);
}

void ResourceCache::RemoveImage(const StorageTexture* image, std::vector<std::shared_ptr<ShaderResources>>* evicted) {
    std::lock_guard lock(mutex);
    removeImage(image, evicted);
}

bool ResourceCache::TryRemoveImage(const StorageTexture* image, std::vector<std::shared_ptr<ShaderResources>>* evicted) {
    std::unique_lock lock(mutex, std::try_to_lock);
    if (!lock.owns_lock()) return false;
    removeImage(image, evicted);
    return true;
}

void ResourceCache::removeImage(const StorageTexture* image, std::vector<std::shared_ptr<ShaderResources>>* evicted) {
    for (auto it = entries.begin(); it != entries.end();) {
        if (!it->second->ReadsImage(image)) {
            ++it;
            continue;
        }
        noteFate(it->first, FateDropped);
        drop(it++, evicted);
    }
}

void ResourceCache::RemoveTextures(const std::unordered_set<const Texture*>& textures, std::vector<std::shared_ptr<ShaderResources>>* evicted) {
    if (textures.empty()) return;
    std::lock_guard lock(mutex);
    std::vector<decltype(entries)::iterator> binding;
    for (const auto* texture : textures) {
        for (auto [found, last] = textureIndex.equal_range(texture); found != last; ++found) binding.push_back(found->second);
    }
    std::sort(binding.begin(), binding.end(), [](const auto& first, const auto& second) { return std::less<>{}(&*first, &*second); });
    binding.erase(std::unique(binding.begin(), binding.end()), binding.end());
    for (const auto entry : binding) {
        noteFate(entry->first, FateDropped);
        drop(entry, evicted);
    }
    if (!binding.empty()) SampledAtomics().droppedSets.fetch_add(binding.size(), std::memory_order_relaxed);
}

void ResourceCache::indexTextures(decltype(entries)::iterator entry, bool add) {
    thread_local std::vector<const Texture*>* uniqueSlot = nullptr;
    auto& unique = ShaderRecompiler::ThreadOwned(uniqueSlot);
    unique.clear();
    for (const auto& texture : entry->second->SampledTextures()) {
        if (texture != nullptr) unique.push_back(texture.get());
    }
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    for (const auto* texture : unique) {
        if (add) {
            textureIndex.emplace(texture, entry);
            continue;
        }
        for (auto [found, last] = textureIndex.equal_range(texture); found != last; ++found) {
            if (found->second == entry) {
                textureIndex.erase(found);
                break;
            }
        }
    }
}

void ResourceCache::drop(std::list<std::pair<Key, std::shared_ptr<ShaderResources>>>::iterator entry, std::vector<std::shared_ptr<ShaderResources>>* evicted) {
    entry->second->PinTextures(false);
    if (entry->second->ReusesAddressSpace()) --addressEntries;
    else if (ComputeKey(entry->first)) {
        --computeEntries;
        const auto bytes = PinnedTextureBytes(*entry->second);
        computePinnedBytes -= std::min(computePinnedBytes, bytes);
    }
    indexTextures(entry, false);
    index.erase(entry->first);
    if (evicted != nullptr) evicted->push_back(std::move(entry->second));
    entries.erase(entry);
}

bool ResourceCache::Touch(const Key& key) {
    resourceCacheTouches.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(mutex);
    const auto found = index.find(key);
    if (found == index.end()) return false;
    entries.splice(entries.begin(), entries, found->second);
    noteUse(key);
    return true;
}

std::uint64_t ResourceCache::Finds() {
    return resourceCacheFinds.load(std::memory_order_relaxed);
}

std::uint64_t ResourceCache::Touches() {
    return resourceCacheTouches.load(std::memory_order_relaxed);
}

void ResourceCache::Clear() {
    std::lock_guard lock(mutex);
    for (const auto& entry : entries) entry.second->PinTextures(false);
    textureIndex.clear();
    index.clear();
    entries.clear();
    addressEntries = 0;
    computeEntries = 0;
    computePinnedBytes = 0;
    evictedKeys.clear();
    evictedOrder.clear();
    evictionsSinceCheck = 0;
    returnsSinceCheck = 0;
    usedKeys.clear();
    usesSinceCheck = 0;
    shrinkWindow = baseShrinkWindow;
    shrankLast = false;
    if (!fixedCapacity) capacity = 0;
}

void ResourceCache::SetBase(std::size_t entries, std::size_t window) {
    std::lock_guard lock(mutex);
    baseEntries = entries;
    baseShrinkWindow = window;
}

std::size_t ResourceCache::Size() const {
    std::lock_guard lock(mutex);
    return entries.size();
}

std::size_t ResourceCache::Capacity() const {
    std::lock_guard lock(mutex);
    return capacity == 0 ? baseEntries : capacity;
}

std::uint64_t ResourceCache::TableElementBudget() {
    static const std::uint64_t elements = [] {
        const char* value = std::getenv("APS5_RESOURCE_CACHE_TABLE_ELEMENTS");
        const auto parsed = value ? std::strtoull(value, nullptr, 10) : 65536ull;
        return static_cast<std::uint64_t>(parsed != 0 ? parsed : 65536ull);
    }();
    return elements;
}

std::uint64_t ResourceCache::TableElements() const {
    std::lock_guard lock(mutex);
    std::uint64_t tables = 0;
    for (const auto& entry : entries) tables += entry.second->TableElements();
    return tables;
}

void ResourceCache::erase(const Key& key) {
    const auto found = index.find(key);
    if (found == index.end()) return;
    noteFate(key, FateRemoved);
    drop(found->second, nullptr);
}

ResourceCache& SharedResourceCache() {
    static auto* cache = new ResourceCache();
    return *cache;
}

namespace {
std::size_t KeptSets() {
    static const std::size_t count = [] {
        const char* text = std::getenv("APS5_KEPT_DESCRIPTOR_SETS");
        return text != nullptr ? static_cast<std::size_t>(std::strtoull(text, nullptr, 10)) : std::size_t{8};
    }();
    return count;
}

}

DescriptorCache::DescriptorCache(const Context& context) : context(context), destroyLayout(context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")), destroyPool(context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")), resetPool(context.Function<PFN_vkResetDescriptorPool>("vkResetDescriptorPool")), freeDescriptorSets(context.individuallyFreeDescriptorSets ? context.Function<PFN_vkFreeDescriptorSets>("vkFreeDescriptorSets") : nullptr) {
    kept.reserve(KeptSets() + 1u);
}

DescriptorCache::~DescriptorCache() {
    for (const auto& pool : pools) destroyPool(context.device, pool.handle, nullptr);
    for (const auto& pool : updateAfterBindPools) destroyPool(context.device, pool.handle, nullptr);
    for (const auto pool : dedicated) destroyPool(context.device, pool, nullptr);
    for (const auto& set : kept) destroyPool(context.device, set.pool, nullptr);
    for (const auto& [key, layout] : layouts) destroyLayout(context.device, layout, nullptr);
}

VkDescriptorSetLayout DescriptorCache::Layout(std::span<const std::uint32_t> key, std::span<const VkDescriptorSetLayoutBinding> bindings, bool updateAfterBind, std::span<const VkDescriptorBindingFlags> flags) {
    std::vector<VkDescriptorBindingFlags> defaults;
    if (flags.empty()) {
        for (const auto& binding : bindings) defaults.push_back(updateAfterBind && binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0);
        flags = defaults;
    }
    Require(flags.size() == bindings.size(), "descriptor binding flag count differs from layout");
    std::lock_guard lock(mutex);
    std::vector<std::uint32_t> keyCopy(key.begin(), key.end());
    keyCopy.push_back(static_cast<std::uint32_t>(flags.size()));
    keyCopy.insert(keyCopy.end(), flags.begin(), flags.end());
    if (const auto found = layouts.find(keyCopy); found != layouts.end()) {
        ++stats.layoutHits;
        return found->second;
    }
    ValidateDescriptorLayout(context, bindings, updateAfterBind);
    ++stats.layoutMisses;
    const auto layout = CreateSetLayout(context, bindings, updateAfterBind, flags);
    try {
        layouts.emplace(std::move(keyCopy), layout);
    } catch (...) {
        destroyLayout(context.device, layout, nullptr);
        throw;
    }
    return layout;
}

namespace {
std::uint32_t SetDescriptorLimit(const Context& context, VkDescriptorType type, bool updateAfterBind) {
    const auto& limits = context.limits;
    const auto& indexed = context.descriptorIndexingLimits;
    const bool sampledUpdateAfterBind = updateAfterBind && context.sampledImageUpdateAfterBind;
    switch (type) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return updateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindStorageBuffers : limits.maxDescriptorSetStorageBuffers;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return sampledUpdateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindSampledImages : limits.maxDescriptorSetSampledImages;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return limits.maxDescriptorSetStorageImages;
        case VK_DESCRIPTOR_TYPE_SAMPLER: return sampledUpdateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindSamplers : limits.maxDescriptorSetSamplers;
        default: throw std::runtime_error("AGC graphics: descriptor set uses an unsupported descriptor type " + std::to_string(type));
    }
}

// What one chain pool holds; a set needing more of any type gets a dedicated pool.
constexpr std::uint32_t ChainPoolSets = 1024;
constexpr std::array<VkDescriptorPoolSize, 4> ChainPoolSizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 512}}};
constexpr std::array<VkDescriptorPoolSize, 4> ChainPoolCreateSizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096 + 2}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 512}}};
}

DescriptorCache::SetAllocation DescriptorCache::Allocate(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind, std::optional<std::uint32_t> variableCount) {
    bool fits = true;
    for (const auto& size : sizes) {
        const auto limit = SetDescriptorLimit(context, size.type, updateAfterBind);
        Require(size.descriptorCount <= limit, "descriptor set exceeds the device's per-set descriptor limit");
        const auto capacity = std::find_if(ChainPoolSizes.begin(), ChainPoolSizes.end(), [&](const auto& item) { return item.type == size.type; });
        if (capacity == ChainPoolSizes.end() || size.descriptorCount > capacity->descriptorCount) fits = false;
    }
    std::lock_guard lock(mutex);
    const auto allocate = context.Resolved(&DeviceFunctions::allocateDescriptorSets, "vkAllocateDescriptorSets");
    const std::uint32_t variable = variableCount.value_or(0u);
    VkDescriptorSetVariableDescriptorCountAllocateInfoEXT variableInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT};
    variableInfo.descriptorSetCount = 1;
    variableInfo.pDescriptorCounts = &variable;
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.pNext = variableCount.has_value() ? &variableInfo : nullptr;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &layout;
    if (!fits) {
        dedicated.reserve(dedicated.size() + 1);
        const auto same = std::find_if(kept.begin(), kept.end(), [&](const SetAllocation& set) {
            return set.layout == layout && set.updateAfterBind == updateAfterBind && set.variableCount == variableCount && std::equal(set.sizes.begin(), set.sizes.end(), sizes.begin(), sizes.end(), [](const VkDescriptorPoolSize& a, const VkDescriptorPoolSize& b) { return a.type == b.type && a.descriptorCount == b.descriptorCount; });
        });
        if (same != kept.end()) {
            auto reused = std::move(*same);
            kept.erase(same);
            dedicated.push_back(reused.pool);
            ++stats.reusedSets;
            ++stats.sets;
            return reused;
        }
        const auto pool = CreateSetPool(context, sizes, updateAfterBind);
        allocation.descriptorPool = pool;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const auto result = allocate(context.device, &allocation, &set);
        if (result != VK_SUCCESS) {
            destroyPool(context.device, pool, nullptr);
            Check(result, "vkAllocateDescriptorSets");
        }
        dedicated.push_back(pool);
        ++stats.dedicatedPools;
        ++stats.sets;
        return {set, pool, 0, updateAfterBind, layout, std::vector<VkDescriptorPoolSize>(sizes.begin(), sizes.end()), variableCount, {}};
    }
    auto& chain = updateAfterBind ? updateAfterBindPools : pools;
    for (std::size_t index = chain.size(); index != 0; --index) {
        auto& pool = chain[index - 1];
        if (pool.exhausted) continue;
        allocation.descriptorPool = pool.handle;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const auto result = allocate(context.device, &allocation, &set);
        if (result == VK_SUCCESS) {
            ++stats.sets;
            ++pool.liveSets;
            return {set, pool.handle, index - 1, updateAfterBind, VK_NULL_HANDLE, {}, variableCount, {}};
        }
        if (result != VK_ERROR_OUT_OF_POOL_MEMORY && result != VK_ERROR_FRAGMENTED_POOL) Check(result, "vkAllocateDescriptorSets");
        pool.exhausted = true;
    }
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = updateAfterBind ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0;
    if (context.individuallyFreeDescriptorSets) poolInfo.flags |= VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = ChainPoolSets;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(ChainPoolCreateSizes.size());
    poolInfo.pPoolSizes = ChainPoolCreateSizes.data();
    chain.reserve(chain.size() + 1);
    VkDescriptorPool pool = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
    chain.push_back({pool});
    ++stats.pools;
    allocation.descriptorPool = pool;
    VkDescriptorSet set = VK_NULL_HANDLE;
    Check(allocate(context.device, &allocation, &set), "vkAllocateDescriptorSets");
    chain.back().liveSets = 1;
    ++stats.sets;
    return {set, pool, chain.size() - 1, updateAfterBind, VK_NULL_HANDLE, {}, variableCount, {}};
}

DescriptorCache::SetAllocation DescriptorCache::AllocateRewritten(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind, std::optional<std::uint32_t> variableCount) {
    {
        std::lock_guard lock(mutex);
        if (const auto found = recycled.find({layout, variableCount.value_or(0u)}); found != recycled.end() && !found->second.empty()) {
            const auto allocation = found->second.back();
            found->second.pop_back();
            return allocation;
        }
    }
    return Allocate(layout, sizes, updateAfterBind, variableCount);
}

void DescriptorCache::Recycle(VkDescriptorSetLayout layout, const SetAllocation& allocation) noexcept {
    if (allocation.set == VK_NULL_HANDLE || allocation.pool == VK_NULL_HANDLE) return;
    {
        std::lock_guard lock(mutex);
        try {
            auto& list = recycled[{layout, allocation.variableCount.value_or(0u)}];
            if (list.size() < 4096) {
                list.push_back(allocation);
                return;
            }
        } catch (...) {
        }
    }
    Free(allocation);
}

void DescriptorCache::Free(const SetAllocation& allocation, std::shared_ptr<const TableDescriptors> table) noexcept {
    if (allocation.set == VK_NULL_HANDLE || allocation.pool == VK_NULL_HANDLE) return;
    std::lock_guard lock(mutex);
    if (const auto found = std::find(dedicated.begin(), dedicated.end(), allocation.pool); found != dedicated.end()) {
        dedicated.erase(found);
        if (KeptSets() == 0u || allocation.layout == VK_NULL_HANDLE) {
            destroyPool(context.device, allocation.pool, nullptr);
            return;
        }
        if (kept.size() >= KeptSets()) {
            destroyPool(context.device, kept.back().pool, nullptr);
            kept.pop_back();
        }
        kept.insert(kept.begin(), allocation);
        kept.front().table = std::move(table);
        ++stats.keptSets;
        return;
    }
    auto& chain = allocation.updateAfterBind ? updateAfterBindPools : pools;
    if (allocation.poolIndex >= chain.size()) std::terminate();
    auto& pool = chain[allocation.poolIndex];
    if (pool.handle != allocation.pool || pool.liveSets == 0) std::terminate();
    if (freeDescriptorSets != nullptr && freeDescriptorSets(context.device, allocation.pool, 1, &allocation.set) != VK_SUCCESS) std::terminate();
    if (--pool.liveSets != 0) return;
    if (resetPool(context.device, pool.handle, 0) != VK_SUCCESS) std::terminate();
    pool.exhausted = false;
}

DescriptorCache::Stats DescriptorCache::Counters() const {
    std::lock_guard lock(mutex);
    return stats;
}

std::uint64_t ShaderResources::GuestBufferBytes(const Context& context, std::span<const std::uint32_t> words) {
    const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1] & ~0x40000000u, words[2], words[3]}};
    const auto address = descriptor.Base48();
    auto byteSize = descriptor.GetSize();
    if (byteSize > context.limits.maxStorageBufferRange) {
        const auto mappedEnd = RegisteredReadableEnd(address);
        if (mappedEnd <= address) byteSize = 0;
        else if (mappedEnd - address < context.limits.maxStorageBufferRange) byteSize = mappedEnd - address;
    }
    if (byteSize == 0 || address == 0) return 0;
    if (const std::uint64_t stride = descriptor.Stride(); stride != 0 && byteSize <= context.limits.maxStorageBufferRange) {
        constexpr std::uint64_t GuestPageBytes = 0x4000;
        const auto end = address + byteSize;
        const auto overhang = end % GuestPageBytes;
        if (overhang != 0 && overhang < stride) {
            const auto allocationEnd = RegisteredReadableEnd(address);
            if (allocationEnd > address && allocationEnd < end && end - allocationEnd < stride) byteSize = allocationEnd - address;
        }
    }
    return byteSize;
}

std::uint64_t ShaderResources::guestBufferBytes(std::span<const std::uint32_t> words) const {
    return GuestBufferBytes(context, words);
}

std::size_t ShaderResources::addGuestBuffer(std::span<const std::uint32_t> words, const ColorTarget* target, std::uint64_t indexAddress, std::size_t indexBytes, bool written, bool atomic, bool read, std::uint32_t extent) {
    Require(words.size() == 4, "buffer descriptor must contain four DWORDs");
    const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1] & ~0x40000000u, words[2], words[3]}};
    Require(descriptor.Type() == 0u, "buffer descriptor uses an unsupported type");
    const auto address = descriptor.Base48();
    const auto byteSize = guestBufferBytes(words);
    if (byteSize == 0 || address == 0) {
        allocations.push_back({0, EmptyBufferBytes, false, nullptr, ShaderRecompiler::DescriptorRole::GuestBuffers, false});
        return allocations.size() - 1;
    }
    Require(byteSize <= context.limits.maxStorageBufferRange, "shader buffer exceeds descriptor range limit");
    Require(byteSize <= std::numeric_limits<std::size_t>::max(), "shader buffer size exceeds host address space");
    auto size = static_cast<std::size_t>(byteSize);
    if (const std::uint64_t stride = descriptor.Stride(); !written && target != nullptr && stride != 0 && address < target->address && address + size > target->address && address + size - target->address < stride) size = static_cast<std::size_t>(target->address - address);
    Require(target == nullptr || !overlap(address, size, target->address, target->bytes), "shader buffer aliases the render target");
    // APS5_ALL_BUFFERS_WRITTEN=1: every element is noted as written, as before bufferWritten existed.
    static const bool allWritten = std::getenv("APS5_ALL_BUFFERS_WRITTEN") != nullptr;
    Require(!written || !overlap(address, size, indexAddress, indexBytes), "writable shader buffer aliases the index buffer");
    written = written || allWritten;
    const std::uint64_t recordBytes = descriptor.Stride() != 0 ? descriptor.Stride() : extent;
    const bool swept = read && dispatchThreads != 0 && recordBytes != 0 && dispatchThreads * recordBytes * 2u >= byteSize;
    if (written) guestMemory.AddWritable(address, size, atomic, swept);
    else {
        guestMemory.AddReadable(address, size);
        ++readOnlyBuffers;
    }
    if (BuildProfiled()) {
        auto& counters = BufferWrites();
        counters.elements.fetch_add(1, std::memory_order_relaxed);
        if (!written) counters.readOnly.fetch_add(1, std::memory_order_relaxed);
    }
    allocations.push_back({address, size, true, nullptr, ShaderRecompiler::DescriptorRole::ShaderData, written});
    return allocations.size() - 1;
}

std::string ShaderResources::Describe() const {
    const auto sample = [](std::uint64_t address, std::uint64_t bytes) {
        // Heaps bound whole can include uncommitted pages; those ranges are not sampled.
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), static_cast<std::size_t>(bytes))) return -1.0;
        std::size_t nonzero = 0;
        std::size_t samples = 0;
        const auto* data = reinterpret_cast<const std::uint32_t*>(address);
        const auto words = bytes / 4;
        const auto step = std::max<std::uint64_t>(1, words / 4096);
        for (std::uint64_t i = 0; i < words; i += step, ++samples) nonzero += data[i] != 0;
        return samples == 0 ? 0.0 : static_cast<double>(nonzero) / samples;
    };
    std::string text;
    char line[160];
    for (const auto& range : describedRanges) {
        std::snprintf(line, sizeof(line), " %s 0x%llx+0x%llx(%ux%u f%u t%d) nz=%.2f", range.kind, static_cast<unsigned long long>(range.address), static_cast<unsigned long long>(range.bytes), range.width, range.height, range.format, range.tileMode, sample(range.address, range.bytes));
        text += line;
        if (range.dccAddress != 0) {
            const auto keys = ReadDccKeys(range.dccAddress, range.bytes);
            std::snprintf(line, sizeof(line), " dcc=%s@0x%llx", DccKeysName(keys), static_cast<unsigned long long>(range.dccAddress));
            text += line;
            if (keys == DccKeys::Mixed) {
                // Where the keys change: first key, how many leading keys match it, and the next key.
                const auto* bytes = reinterpret_cast<const std::uint8_t*>(range.dccAddress);
                const auto count = static_cast<std::size_t>(range.bytes / 256u);
                std::size_t run = 1;
                while (run < count && bytes[run] == bytes[0]) ++run;
                std::snprintf(line, sizeof(line), "(%02x x%zu then %02x of %zu)", bytes[0], run, run < count ? bytes[run] : 0u, count);
                text += line;
            }
        }
    }
    // APS5_TRACE_DISPATCH_IO=2 also lists the words of small buffers (constants).
    static const bool words = [] { const char* value = std::getenv("APS5_TRACE_DISPATCH_IO"); return value != nullptr && value[0] == '2'; }();
    const auto appendWords = [&](const std::uint32_t* data, std::size_t bytes) {
        if (!words || bytes > 0x200) return;
        text += " [";
        for (std::size_t i = 0; i < bytes / 4; ++i) {
            char word[12];
            std::snprintf(word, sizeof(word), "%s%08x", i == 0 ? "" : " ", data[i]);
            text += word;
        }
        text += "]";
    };
    for (const auto& allocation : allocations) {
        if (allocation.guest) {
            std::snprintf(line, sizeof(line), " buffer%s 0x%llx+0x%zx nz=%.2f", allocation.written ? "" : "(ro)", static_cast<unsigned long long>(allocation.address), allocation.size, sample(allocation.address, allocation.size));
            text += line;
            if (GuestMemory::Accessible(reinterpret_cast<const void*>(allocation.address), allocation.size)) appendWords(reinterpret_cast<const std::uint32_t*>(allocation.address), allocation.size);
        } else if (allocation.buffer) {
            const auto bytes = allocation.buffer->Bytes();
            std::snprintf(line, sizeof(line), " data+0x%zx nz=%.2f", allocation.size, sample(reinterpret_cast<std::uint64_t>(bytes.data()), allocation.size));
            text += line;
            appendWords(reinterpret_cast<const std::uint32_t*>(bytes.data()), allocation.size);
        }
    }
    return text;
}

std::size_t ShaderResources::addDataBuffer(std::span<const std::uint32_t> words) {
    const auto size = words.size() * sizeof(std::uint32_t);
    Require(size <= context.limits.maxStorageBufferRange, "shader data buffer exceeds descriptor range limit");
    const bool refreshable = TemplateDataRefresh() && size <= MaxRefreshBytes;
    const MemoryStats::OwnerScope memStatsOwner(MemoryStats::Owner::ShaderResources);
    auto buffer = std::make_unique<Buffer>(context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memcpy(buffer->Bytes().data(), words.data(), size);
    Allocation allocation{0, size, false, std::move(buffer)};
    if (refreshable) allocation.dataWords.assign(words.begin(), words.end());
    allocations.push_back(std::move(allocation));
    mixDataWords(dataWordsHash, allocations.back().dataWords);
    return allocations.size() - 1;
}

void ShaderResources::rehashDataWords() {
    dataWordsHash = FnvOffset;
    // Data buffers are the non-guest allocations with a buffer (an address-role allocation has
    // none), appended in binding order by buildPrepare: the order DataWordsHash(shader) hashes.
    for (const auto& allocation : allocations) {
        if (!allocation.guest && allocation.buffer != nullptr && allocation.role != ShaderRecompiler::DescriptorRole::ImageTableMap) mixDataWords(dataWordsHash, allocation.dataWords);
    }
}

std::uint64_t ShaderResources::DataWordsHash(const CompiledShader& shader) {
    Require(shader.program != nullptr, "missing compiled shader");
    std::uint64_t hash = FnvOffset;
    for (const auto& binding : shader.program->bindings) {
        if (DataRole(binding.role)) mixDataWords(hash, binding.guestDescriptor);
    }
    return hash;
}

bool ShaderResources::DataWordsDiffer(const CompiledShader& shader) const {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    if (program.bindings.size() != bindings.size()) return true;
    for (std::size_t index = 0; index < program.bindings.size(); ++index) {
        const auto& binding = program.bindings[index];
        if (!DataRole(binding.role)) continue;
        if (bindings[index].allocations.size() != 1) return true;
        const auto& allocation = allocations[bindings[index].allocations.front()];
        if (allocation.dataWords.size() != binding.guestDescriptor.size() || !std::equal(allocation.dataWords.begin(), allocation.dataWords.end(), binding.guestDescriptor.begin())) return true;
    }
    return false;
}

bool ShaderResources::RefreshData(VkCommandBuffer commands, const CompiledShader& shader, Recorder* recorder) {
    Require(shader.program != nullptr, "missing compiled shader");
    Require(!refreshResourceKey.empty() && MatchesContentKey(shader, refreshResourceKey), "runtime data refresh cannot replace bound resources");
    return refreshData(commands, shader, recorder);
}

bool ShaderResources::refreshData(VkCommandBuffer commands, const CompiledShader& shader, Recorder* recorder) {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    Require(program.bindings.size() == bindings.size(), "template bindings disagree with the shader");
    bool recorded = false;
    // One [gputime] range of class TemplateDataRefresh around the updates, begun at the first one
    // (a refresh that finds every word equal records nothing and times nothing).
    auto timing = Recorder::NoTiming;
    std::uint64_t refreshedBytes = 0;
    for (std::size_t index = 0; index < program.bindings.size(); ++index) {
        const auto& binding = program.bindings[index];
        if (!DataRole(binding.role)) continue;
        Require(bindings[index].allocations.size() == 1, "data binding without its buffer");
        auto& allocation = allocations[bindings[index].allocations.front()];
        const auto size = binding.guestDescriptor.size() * sizeof(std::uint32_t);
        Require(allocation.buffer != nullptr && !allocation.guest && allocation.size == size && size <= MaxRefreshBytes, "template data buffer cannot take the dispatch's words");
        if (allocation.dataWords.size() == binding.guestDescriptor.size() && std::equal(allocation.dataWords.begin(), allocation.dataWords.end(), binding.guestDescriptor.begin())) continue;
        if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) ValidateRuntimeResources(shader, binding.guestDescriptor);
        if (recorder != nullptr && timing == Recorder::NoTiming) timing = recorder->BeginGpuTiming(Recorder::CommandClass::TemplateDataRefresh);
        writeDataWords(commands, bindings[index].allocations.front(), binding.guestDescriptor);
        allocation.dataWords.assign(binding.guestDescriptor.begin(), binding.guestDescriptor.end());
        refreshedBytes += size;
        recorded = true;
    }
    if (timing != Recorder::NoTiming) recorder->EndGpuTiming(timing, refreshedBytes);
    if (recorded) rehashDataWords();
    return recorded;
}

std::optional<std::size_t> ShaderResources::shaderDataAllocation(const CompiledShader& shader) const {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    if (program.bindings.size() != bindings.size()) return std::nullopt;
    for (std::size_t index = 0; index < program.bindings.size(); ++index) {
        if (program.bindings[index].role != ShaderRecompiler::DescriptorRole::ShaderData || bindings[index].allocations.size() != 1) continue;
        const auto allocation = bindings[index].allocations.front();
        if (allocations[allocation].guest || allocations[allocation].buffer == nullptr) return std::nullopt;
        return allocation;
    }
    return std::nullopt;
}

VkBuffer ShaderResources::shaderDataBuffer(const CompiledShader& shader) const {
    const auto allocation = shaderDataAllocation(shader);
    return allocation.has_value() ? allocations[*allocation].buffer->Handle() : VK_NULL_HANDLE;
}

void ShaderResources::forgetShaderDataWords(const CompiledShader& shader) {
    if (const auto allocation = shaderDataAllocation(shader)) {
        allocations[*allocation].dataWords.clear();
        rehashDataWords();
    }
}

void ShaderResources::writeDataWords(VkCommandBuffer commands, std::size_t allocation, std::span<const std::uint32_t> words) const {
    const auto& buffer = *allocations[allocation].buffer;
    const auto size = words.size() * sizeof(std::uint32_t);
    if (std::none_of(dataPatches.begin(), dataPatches.end(), [&](const DataPatch& patch) { return patch.allocation == allocation; })) {
        context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer")(commands, buffer.Handle(), 0, size, words.data());
        return;
    }
    std::vector<std::uint32_t> patched(words.begin(), words.end());
    auto* bytes = reinterpret_cast<std::byte*>(patched.data());
    for (const auto& patch : dataPatches) {
        if (patch.allocation == allocation && patch.byte < size) bytes[patch.byte] = static_cast<std::byte>(patch.adjustment);
    }
    context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer")(commands, buffer.Handle(), 0, size, patched.data());
}

void ShaderResources::PrecollectSurfaces() const {
    for (const auto& range : describedRanges) GuestMemory::CollectWrites(range.address, static_cast<std::size_t>(range.bytes));
}

void ShaderResources::addImageBinding(const ShaderRecompiler::DescriptorBinding& binding, VkShaderStageFlags flags, bool guardedUnnormalizedWrap) {
    Require(binding.count != 0, "empty descriptor binding");
    const auto heap = static_cast<ShaderRecompiler::RuntimeAbi::Binding>(binding.binding % static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count));
    const auto capacity = binding.count;
    Require(binding.count <= ShaderRecompiler::RuntimeAbi::HeapCapacity(heap), "typed descriptor heap capacity exceeded");
    const auto localBinding = static_cast<std::uint32_t>(heap);
    const bool sampledHeap = localBinding >= ShaderRecompiler::RuntimeAbi::FirstImageBinding && localBinding < ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding;
    const bool storageHeap = localBinding >= ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding && localBinding < static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Samplers);
    Require((binding.kind == ShaderRecompiler::DescriptorKind::SampledImage && sampledHeap) || (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage && storageHeap) || (binding.kind == ShaderRecompiler::DescriptorKind::Sampler && heap == ShaderRecompiler::RuntimeAbi::Binding::Samplers), "resource class disagrees with typed descriptor heap");
    Require(binding.kind == ShaderRecompiler::DescriptorKind::Sampler || context.nullDescriptors || context.placeholderImages != nullptr, "typed image heaps require nullDescriptor or placeholder images");
    if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
        // Storage images are guest textures the shader writes (looked up in stage B).
        Require(binding.role == ShaderRecompiler::DescriptorRole::GuestImages, "storage image binding has a non-image role");
        Require(binding.guestDescriptor.size() == static_cast<std::size_t>(binding.count) * 8u, "guest storage image descriptors must contain 8 dwords each");
        Require(context.detiler != nullptr, "device texture detiler is unavailable");
        Require(capacity <= context.limits.maxPerStageDescriptorStorageImages, "shader storage-image descriptors exceed per-stage limits");
        bindings.push_back({{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, capacity, flags, nullptr}, {}, {}});
        plannedStorageImages += binding.count;
        deferredImages.push_back({&binding, bindings.size() - 1});
        return;
    }
    const bool sampledImage = binding.kind == ShaderRecompiler::DescriptorKind::SampledImage;
    const bool samplerKind = binding.kind == ShaderRecompiler::DescriptorKind::Sampler;
    if (!(sampledImage || samplerKind)) Require(false, std::string("unsupported descriptor kind ") + kindName(binding.kind) + " for role " + roleName(binding.role));
    Require((sampledImage && binding.role == ShaderRecompiler::DescriptorRole::GuestImages) || (samplerKind && binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers), "guest image descriptor role disagrees with its kind");
    Require(binding.guestDescriptor.size() % binding.count == 0, "guest image descriptor size is not a multiple of the binding count");
    const auto elementWords = binding.guestDescriptor.size() / binding.count;

    Binding item{{binding.binding, sampledImage ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER, capacity, flags, nullptr}, {}, {}};

    if (sampledImage) {
        Require(elementWords == 8, "guest texture descriptor must contain 8 dwords");
        Require(binding.imageShape.has_value(), "guest image binding is missing an image shape");
        Require(context.detiler != nullptr, "device texture detiler is unavailable");
        Require(context.textureCache != nullptr, "device texture cache is unavailable");
        Require(capacity <= context.limits.maxPerStageDescriptorSampledImages, "shader sampled-image descriptors exceed per-stage limits");
        plannedSampledImages += binding.count;
        Require(plannedSampledImages <= context.limits.maxDescriptorSetSampledImages, "pipeline sampled-image descriptors exceed device limits");
    } else {
        Require(elementWords == 4, "guest sampler descriptor must contain 4 dwords");
        if (capacity > context.limits.maxPerStageDescriptorSamplers) throw std::runtime_error(DescriptorLimitError("shader sampler descriptors exceed per-stage limits", capacity, context.limits.maxPerStageDescriptorSamplers));
        Require(binding.samplerDepthCompare.size() == binding.count, "guest sampler binding is missing depth comparison metadata");
        Require(binding.samplerUnnormalized.empty() || binding.samplerUnnormalized.size() == binding.count, "guest sampler binding has unnormalized coordinate metadata of another size");
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            auto guestWords = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
            std::array<std::uint32_t, 4> normalizedWords{};
            if (element < binding.samplerCoordinatesIgnored.size() && binding.samplerCoordinatesIgnored[element]) {
                std::copy(guestWords.begin(), guestWords.end(), normalizedWords.begin());
                normalizedWords[0] &= ~(1u << 15u);
                guestWords = normalizedWords;
            }
            const auto words = guestWords;
            const bool compareEnable = binding.samplerDepthCompare.at(element);
            const bool unnormalized = element < binding.samplerUnnormalized.size() && binding.samplerUnnormalized[element];
            const bool integerBorder = element < binding.samplerIntegerBorder.size() && binding.samplerIntegerBorder[element];
            if (context.samplerCache != nullptr && !SamplerCacheDisabled()) {
                samplers.push_back(context.samplerCache->Get(context, words, compareEnable, unnormalized, true, integerBorder, guardedUnnormalizedWrap));
            } else {
                auto resource = DecodeSamplerResource(words, unnormalized, true, integerBorder, guardedUnnormalizedWrap);
                resource.compareEnable = compareEnable;
                samplers.push_back(std::make_shared<Sampler>(context, resource));
            }
            samplerSources.push_back({{words[0], words[1], words[2], words[3]}, compareEnable, unnormalized, integerBorder});
            item.imageAllocations.push_back(samplers.size() - 1);
        }
        Require(samplers.size() <= context.limits.maxDescriptorSetSamplers, "pipeline sampler descriptors exceed device limits");
    }

    bindings.push_back(std::move(item));
    if (sampledImage) deferredImages.push_back({&binding, bindings.size() - 1});
}

namespace {

// Calls `visit(binding, element, words)` for every sampled and storage image element of a compiled
// shader's bindings, in plan order (the order addImageBinding and resolveImageBinding walk).
template <typename Visit>
void forEachImageElement(const ShaderRecompiler::RecompileResult& program, Visit&& visit) {
    for (const auto& binding : program.bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.count == 0) continue;
        if (binding.kind != ShaderRecompiler::DescriptorKind::SampledImage && binding.kind != ShaderRecompiler::DescriptorKind::StorageImage) continue;
        const auto elementWords = binding.guestDescriptor.size() / binding.count;
        for (std::uint32_t element = 0; element < binding.count; ++element) visit(binding, element, std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords));
    }
}

}

bool ShaderResources::precollectImages() {
    // The image lookups of stage B (cachedTexture, StorageTexture::Refresh) each start with a
    // GuestMemory::CollectWrites of their surface, the GetWriteWatch walk that is most of the
    // lookups' time under the lock. The walk is memoized per thread epoch, and the worker thread
    // passes no ordering point between the stages (see GuestMemory::BumpCollectEpoch), so walking
    // every surface here makes stage B's collects memo hits: the walk leaves the lock, the lookups'
    // checks stay where they were. A memo miss (the ring overflowed, an uncommitted page) just walks
    // under the lock as before; a descriptor stage B rejects is left to it. APS5_NO_PRECOLLECT=1
    // disables the pass.
    // The pass also leaves one ImageRecord per element: the decode and the surface description are
    // made once for the build, and a sampled element's cache entry is taken here (a hash lookup
    // under the cache's own mutex) so that stage B, under the device lock, runs only the
    // fastRevalidate predicate on it (see fastTexture). The collect comes before the entry is read,
    // as the predicate's rule demands (the generation the entry moves to must predate the checks).
    static const bool disabled = std::getenv("APS5_NO_PRECOLLECT") != nullptr;
    static const bool noRecords = std::getenv("APS5_NO_STAGE_A_IMAGES") != nullptr;
    if (disabled || deferredImages.empty()) return false;
    imageRecords.clear();
    nextImageRecord = 0;
    auto& counters = TextureCounts();
    for (const auto& deferred : deferredImages) {
        const auto& binding = *deferred.binding;
        const auto elementWords = binding.guestDescriptor.size() / binding.count;
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const bool sampled = binding.kind == ShaderRecompiler::DescriptorKind::SampledImage;
            const auto raw = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
            std::array<std::uint32_t, 8> narrowed{};
            const auto words = firstLayerWords(raw, sampled ? binding.imageShape : std::nullopt, narrowed, false);
            if (sampled && !imageRecords.empty()) {
                const auto& previous = imageRecords.back();
                if (previous.sampled && previous.decoded && words.size() == 8 && std::equal(words.begin(), words.end(), previous.words.begin()) && (binding.imageDepthCompare.empty() || !binding.imageDepthCompare.at(element))) {
                    ImageRecord repeated;
                    repeated.sampled = true;
                    repeated.decoded = previous.decoded;
                    repeated.words = previous.words;
                    repeated.resource = previous.resource;
                    repeated.guestBytes = previous.guestBytes;
                    repeated.components = previous.components;
                    imageRecords.push_back(repeated);
                    continue;
                }
            }
            ImageRecord record;
            record.sampled = sampled;
            if (!IsNullTextureResource(words)) {
                const auto* memo = words.size() == 8 ? &decodedImage(context.device, words) : nullptr;
                record.resource = memo != nullptr ? memo->resource : DecodeTextureResource(words);
                record.guestBytes = memo != nullptr ? memo->guestBytes : DescribeSurface(record.resource).guestBytes;
                record.generation = GuestMemory::CollectWrites(record.resource.baseAddress, static_cast<std::size_t>(record.guestBytes));
                record.decoded = true;
                if (record.sampled && !noRecords && words.size() == 8 && (binding.imageDepthCompare.empty() || !binding.imageDepthCompare.at(element))) {
                    std::copy(words.begin(), words.end(), record.words.begin());
                    record.components = memo->components;
                    record.keys = TextureClearKeys(record.resource, record.guestBytes);
                    auto& cache = Textures();
                    std::lock_guard lock(cache.mutex);
                    if (const auto it = findSampledEntry(cache, context.device, record.words, record.components); it != cache.entries.end()) {
                        record.texture = it->texture;
                        record.source = it->source;
                        record.entryKeys = it->keys;
                        record.entryGeneration = it->generation;
                        if (record.generation == 0 && it->source == nullptr && it->committed.has_value() && CommittedProofEnabled()) record.entryCommitted = it->committed;
                        counters.records.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
            imageRecords.push_back(std::move(record));
        }
    }
    return true;
}

std::shared_ptr<Texture> ShaderResources::fastTexture(const ImageRecord& record) {
    if (record.texture == nullptr || record.texture->RefreshedPerUse() || record.texture->IsDepthCopy()) return nullptr;
    struct Outcome {
        bool profile;
        std::chrono::steady_clock::time_point start;
        bool hit = false;
        ~Outcome() {
            if (profile) LookupOutcomes::Add(hit ? LookupOutcomes::SampledFast : LookupOutcomes::SampledFastMiss, start);
        }
    } outcome{LookupOutcomes::Profiled(), LookupOutcomes::Profiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}};
    // Exactly the "unchanged" branches of cachedTexture, from write stamps, the pending-results
    // registry and the DCC keys, the way fastRevalidate proves a built object current: the collect
    // first (a memo hit: stage A walked the range), then no other image may have results pending
    // over the memory (a lookup would flush them into it, or view them instead), a storage-sourced
    // view needs its image to be the cache's image of the surface and current with guest memory,
    // a snapshot needs the memory unchanged since its content matched, and the keys must be what
    // the content was made under whenever the surface has any (re-read under the lock: a flush
    // between the stages marks them uncompressed).
    const auto address = record.resource.baseAddress;
    const auto bytes = static_cast<std::size_t>(record.guestBytes);
    RequireNoMultisampleColorTarget(address, bytes != 0 ? bytes : 1u, "a texture read");
    if (DepthSurfaceAt(address) || !FlushDepthPlanes(address, bytes).empty()) return nullptr;
    thread_local std::vector<std::pair<std::uint64_t, std::uint64_t>>* runsSlot = nullptr;
    auto& runs = ShaderRecompiler::ThreadOwned(runsSlot);
    std::uint64_t committedGeneration = 0;
    if (GuestMemory::CollectWrites(address, bytes) == 0) {
        if (record.source != nullptr || !record.entryCommitted.has_value()) return nullptr;
        committedGeneration = GuestMemory::CollectCommittedWrites(address, bytes, runs);
        if (committedGeneration == 0 || runs != *record.entryCommitted) return nullptr;
    }
    if (PendingStorageOverlaps(address, bytes, record.source.get())) return nullptr;
    auto keys = record.keys;
    if (record.resource.dccAddress != 0) {
        const auto scanStart = outcome.profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        // Through the proof of the image whose keys they are (the view's source when the surface
        // is the image's, else the texture's): a scan only when the key range was stamped.
        keys = ProvedClearKeys(record.resource, record.guestBytes, SameKeySurface(record.source.get(), record.resource, record.guestBytes) ? record.source->KeyProof() : record.texture->KeyProof());
        if (outcome.profile) LookupOutcomes::Add(LookupOutcomes::DccScan, scanStart);
    }
    // The keys the entry was made under (a view of a pending render target was made under its clear
    // keys and stays valid while they are unchanged); a change is the full lookup's to judge.
    if (keys != record.entryKeys) return nullptr;
    if (record.source != nullptr) {
        if (!StorageImageCached(context, record.source.get()) || !GuestMemory::UnchangedSince(address, bytes, record.source->GenerationFor(address, bytes))) return nullptr;
        if (record.resource.dccAddress != 0 && IsDccClear(record.source->FilledKeys())) return nullptr;
        // Under fast-clear keys the view holds only while its image's results are still pending over
        // the surface; flushed, the clear the image cannot see makes the lookup take a snapshot.
        if (keys != DccKeys::Uncompressed && StorageTexture::FindPending(address, record.guestBytes) != record.source) return nullptr;
    } else if (keys == DccKeys::Uncompressed || keys == DccKeys::ClearSingle) {
        if (committedGeneration != 0 ? Recorder::SnapshotWriteOverlaps(address, bytes) || !GuestMemory::UnchangedCommittedSince(runs, record.entryGeneration) : !GuestMemory::UnchangedSince(address, bytes, record.entryGeneration)) return nullptr;
    }
    // The entry must still be the cache's, holding this object (an evicted object is not reused: the
    // next lookup would make another), and it takes the stage-A generation like a hit would.
    auto& cache = Textures();
    std::lock_guard lock(cache.mutex);
    const auto it = findSampledEntry(cache, context.device, record.words, record.components);
    if (it == cache.entries.end() || it->texture != record.texture || (committedGeneration != 0 ? it->committed != runs : it->committed.has_value())) return nullptr;
    if (it->source == nullptr) it->generation = committedGeneration != 0 ? committedGeneration : record.generation;
    touchTexture(cache, it);
    if (committedGeneration != 0) logLookup({record.texture.get(), record.resource, record.guestBytes, keys, 0, nullptr, false, committedGeneration, runs});
    else logLookup({record.texture.get(), record.resource, record.guestBytes, keys, record.source != nullptr ? 0 : record.generation, record.source.get()});
    outcome.hit = true;
    return record.texture;
}

void ShaderResources::applyAnisoOverride() {
    for (std::size_t index = 0; index < samplerSources.size(); ++index) {
        const auto& source = samplerSources[index];
        const auto variant = SingleLevelSamplerWords(source.words, source.singleLevelImage, source.mipmappedImage);
        if (!variant.has_value()) continue;
        const auto& words = *variant;
        if (context.samplerCache != nullptr && !SamplerCacheDisabled()) {
            samplers[index] = context.samplerCache->Get(context, words, source.compareEnable, source.unnormalized, true, source.integerBorder);
        } else {
            auto resource = DecodeSamplerResource(words, source.unnormalized, true, source.integerBorder);
            resource.compareEnable = source.compareEnable;
            samplers[index] = std::make_shared<Sampler>(context, resource);
        }
    }
}

void ShaderResources::resolveImageBinding(const ShaderRecompiler::DescriptorBinding& binding, Binding& item, std::span<const std::shared_ptr<Sampler>> shaderSamplers) {
    auto& counters = TextureCounts();
    // The element's stage-A record, when the pass ran (records follow the plan order exactly).
    const auto nextRecord = [&]() -> const ImageRecord* { return nextImageRecord < imageRecords.size() ? &imageRecords[nextImageRecord++] : nullptr; };
    if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) {
        Require(binding.imageSamplers.size() == binding.count, "guest sampled image binding is missing its image-sampler pairs");
        const auto elementWords = binding.guestDescriptor.size() / binding.count;
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto raw = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
            std::array<std::uint32_t, 8> narrowed{};
            const auto words = firstLayerWords(raw, binding.imageShape, narrowed, true);
            const auto* record = nextRecord();
            if (IsNullTextureResource(words)) {
                item.imageAllocations.push_back(std::numeric_limits<std::size_t>::max());
                --plannedSampledImages;
                continue;
            }
            const auto resource = record != nullptr && record->decoded ? record->resource : DecodeTextureResource(words);
            const bool forcedSrgb = ForcesDegammaOn(resource.format, binding.imageSamplers[element], shaderSamplers);
            RequireNonSeamlessCube(resource.dimension == TextureDimension::kCube, binding.imageSamplers[element], shaderSamplers);
            const bool depthCompare = !binding.imageDepthCompare.empty() && binding.imageDepthCompare.at(element);
            const bool firstLayer = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && (resource.dimension == TextureDimension::k2DArray || (depthCompare && resource.dimension == TextureDimension::k1DArray));
            const bool depth1D = depthCompare && ((binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k1D) || (binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2DArray && resource.dimension == TextureDimension::k1DArray));
            const bool firstSample = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DMsaa;
            if (!firstLayer && !depth1D && !firstSample && !MatchesGuestDimension(*binding.imageShape, resource.dimension)) throw std::runtime_error("AGC graphics: guest texture dimension disagrees with the shader's declared image shape (shape " + std::to_string(static_cast<int>(*binding.imageShape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")");
            const VkComponentMapping components = ViewComponents(resource);
            const auto guestBytes = record != nullptr && record->decoded ? record->guestBytes : DescribeSurface(resource).guestBytes;
            CompletePendingCpuWrites(context, resource.baseAddress, static_cast<std::size_t>(guestBytes));
            if (resource.dccAddress != 0) CompletePendingCpuWrites(context, resource.dccAddress, DccKeyBytes(guestBytes));
            std::shared_ptr<Texture> texture;
            const VkImageViewType multisampleViewType = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2DMultisampleArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
            if (previousSampled.texture != nullptr && previousSampled.depthCompare == depthCompare && previousSampled.multisampleViewType == multisampleViewType && previousSampled.firstSample == firstSample && words.size() == 8 && std::equal(words.begin(), words.end(), previousSampled.words.begin())) texture = previousSampled.texture;
            if (texture == nullptr && !firstSample && record != nullptr && record->texture != nullptr) {
                texture = fastTexture(*record);
                (texture != nullptr ? counters.fastHits : counters.fastMisses).fetch_add(1, std::memory_order_relaxed);
            }
            if (texture == nullptr && firstSample) {
                Require(!depthCompare && !DepthSurfaceAt(resource.baseAddress), "reading sample 0 of a multisampled depth surface through a 2D image is not implemented");
                texture = MultisampleTargetFirstSample(context, resource, components);
            }
            if (texture == nullptr) texture = cachedTexture(context, words, resource, components, guestBytes, depthCompare, multisampleViewType);
            if (resource.dimension == TextureDimension::k2DMsaa) multisampleSources.push_back(resource.baseAddress);
            bool baseLevel = false;
            if (element < binding.imageUnnormalized.size() && binding.imageUnnormalized[element]) {
                const auto range = texture->SampledViewRange(firstLayer);
                const bool exactAtBase = element < binding.imageUnnormalizedBaseLevel.size() && binding.imageUnnormalizedBaseLevel[element];
                const bool singleLevel = range.levels == 1u && resource.baseLevel == 0u;
                const bool supported = range.layers == 1u && EffectiveMinLod(resource) == 0.0f && (singleLevel || exactAtBase);
                if (!supported || (range.type != VK_IMAGE_VIEW_TYPE_1D && range.type != VK_IMAGE_VIEW_TYPE_2D)) throw std::runtime_error("AGC graphics: guest texture sampled with unnormalized coordinates is not a single-layer 1D or 2D view without a minimum LOD clamp, and is neither a single-level view at mip 0 nor read only by image_sample_lz through an S# without minimum LOD or bias, which is not implemented (base level " + std::to_string(resource.baseLevel) + ", levels " + std::to_string(range.levels) + ", layers " + std::to_string(range.layers) + ", view type " + std::to_string(static_cast<int>(range.type)) + ")");
                baseLevel = range.levels > 1u;
            }
            RequireFilterMinmax(context, texture->ViewFormat(), binding.imageSamplers[element], shaderSamplers);
            RequireBorderSwizzle(resource.bcSwizzle, binding.imageSamplers[element], shaderSamplers);
            if (forcedSrgb && !texture->HasSrgbView()) Require(false, "a texture of guest format " + std::to_string(resource.format) + ", which is not sRGB and has no sRGB view, is sampled through a sampler that forces sRGB decoding, which is not implemented");
            if (!shaderSamplers.empty()) {
                const bool singleLevel = DescriptorSingleLevel(words);
                const auto firstSampler = static_cast<std::size_t>(shaderSamplers.data() - samplers.data());
                for (std::uint32_t sampler = 0; sampler < shaderSamplers.size() && sampler < 32u; ++sampler) {
                    if (((binding.imageSamplers[element] >> sampler) & 1u) == 0) continue;
                    auto& source = samplerSources.at(firstSampler + sampler);
                    (singleLevel ? source.singleLevelImage : source.mipmappedImage) = true;
                }
            }
            if (words.size() == 8) {
                std::copy(words.begin(), words.end(), previousSampled.words.begin());
                previousSampled.depthCompare = depthCompare;
                previousSampled.multisampleViewType = multisampleViewType;
                previousSampled.firstSample = firstSample;
                previousSampled.texture = texture;
            }
            textures.push_back(std::move(texture));
            textureFirstLayer.push_back(firstLayer);
            textureBaseLevel.push_back(baseLevel);
            textureSrgb.push_back(forcedSrgb);
            describedRanges.push_back({"texture", resource.baseAddress, guestBytes, resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.dccAddress});
            item.imageAllocations.push_back(textures.size() - 1);
        }
        // The line is due even when every element took the fast path (cachedTexture reports too).
        reportTextureCounters();
        return;
    }
    previousSampled = {};
    // Consecutive identical storage descriptors address successive mips of one texture (dynamic-mip
    // storage writes).
    std::uint32_t mipOffset = 0;
    for (std::uint32_t element = 0; element < binding.count; ++element) {
        const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
        const bool sameAsPrevious = SameAsPreviousStorageElement(binding, element);
        if (sameAsPrevious) ++mipOffset;
        else mipOffset = 0;
        const auto* record = nextRecord();
        if (IsNullTextureResource(words)) {
            item.imageAllocations.push_back(std::numeric_limits<std::size_t>::max());
            --plannedStorageImages;
            continue;
        }
        const auto resource = record != nullptr && record->decoded ? record->resource : DecodeTextureResource(words);
        const bool firstLayer = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DArray;
        if (binding.imageShape.has_value() && !firstLayer && !MatchesGuestDimension(*binding.imageShape, resource.dimension)) throw std::runtime_error("AGC graphics: guest storage texture dimension disagrees with the shader's declared image shape (shape " + std::to_string(static_cast<int>(*binding.imageShape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")");
        if (resource.dimension == TextureDimension::k2DMsaa) {
            Require(!(element < binding.imageAtomic.size() && binding.imageAtomic[element]) && !(element < binding.imageAtomic64.size() && binding.imageAtomic64[element]), "storage image atomics on a multisampled texture are not implemented");
            storageTextures.push_back(nullptr);
            multisampleStorageViews.resize(storageTextures.size(), VK_NULL_HANDLE);
            multisampleStorageViews.back() = MultisampleTargetStorageView(context, resource);
            multisampleSources.push_back(resource.baseAddress);
            storageMips.push_back(0);
            storageKeys.push_back(0);
            storageFirstLayer.push_back(false);
            storageWritten.push_back(false);
            storageAtomic.push_back(false);
            storageAtomic64.push_back(false);
            item.imageAllocations.push_back(storageTextures.size() - 1);
            continue;
        }
        const auto mip = std::min(resource.baseLevel + mipOffset, resource.mipCount - 1u);
        Require(resource.minLod <= mip * 256u, "guest storage texture descriptor clamps its minimum LOD above the level it addresses, which is not implemented");
        const auto guestBytes = record != nullptr && record->decoded ? record->guestBytes : DescribeSurface(resource).guestBytes;
        CompletePendingCpuWrites(context, resource.baseAddress, static_cast<std::size_t>(guestBytes));
        if (resource.dccAddress != 0) CompletePendingCpuWrites(context, resource.dccAddress, DccKeyBytes(guestBytes));
        // Images the shader only reads have nothing to store back.
        const bool written = element >= binding.imageWritten.size() || binding.imageWritten[element];
        // The same surface as the previous element: its image was just looked up and refreshed.
        if (sameAsPrevious && StorageDedupeEnabled()) storageTextures.push_back(storageTextures.back());
        else storageTextures.push_back(cachedStorageTexture(context, words, resource, mip, guestBytes, written));
        storageMips.push_back(mip);
        storageKeys.push_back(resource.dccAddress);
        storageFirstLayer.push_back(firstLayer);
        storageWritten.push_back(written);
        storageAtomic.push_back(element < binding.imageAtomic.size() && binding.imageAtomic[element]);
        storageAtomic64.push_back(element < binding.imageAtomic64.size() && binding.imageAtomic64[element]);
        describedRanges.push_back({"storage", resource.baseAddress, guestBytes, resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.dccAddress});
        item.imageAllocations.push_back(storageTextures.size() - 1);
    }
}

namespace {

std::uint64_t Room(std::uint64_t limit, std::uint64_t reserve) {
    return limit > reserve ? limit - reserve : 0u;
}

std::uint32_t ImageTableCapacityFor(const Context& context, bool updateAfterBind, std::uint32_t storageTableDescriptors = 0u) {
    if (!context.runtimeDescriptorArray || !context.partiallyBound || !context.variableDescriptorCount || !context.descriptorIndexing) return 0u;
    if (updateAfterBind && !context.sampledImageUpdateAfterBind) return 0u;
    constexpr std::uint64_t sampledHeaps = static_cast<std::uint64_t>(ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding - ShaderRecompiler::RuntimeAbi::FirstImageBinding) * ShaderRecompiler::RuntimeAbi::SampledHeapCapacity;
    constexpr std::uint64_t storageHeaps = static_cast<std::uint64_t>(static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Samplers) - ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding) * ShaderRecompiler::RuntimeAbi::StorageHeapCapacity;
    constexpr std::uint64_t buffers = ShaderRecompiler::RuntimeAbi::BufferCapacity + 8u;
    const auto& indexed = context.descriptorIndexingLimits;
    const std::uint64_t stageImages = updateAfterBind ? indexed.maxPerStageDescriptorUpdateAfterBindSampledImages : context.limits.maxPerStageDescriptorSampledImages;
    const std::uint64_t stageResources = updateAfterBind ? indexed.maxPerStageUpdateAfterBindResources : context.limits.maxPerStageResources;
    const std::uint64_t setImages = updateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindSampledImages : context.limits.maxDescriptorSetSampledImages;
    std::uint64_t capacity = static_cast<std::uint64_t>(ShaderRecompiler::RuntimeAbi::StageCount) * ShaderRecompiler::ImageTableAbi::ImageElementBudget;
    capacity = std::min(capacity, Room(stageImages, sampledHeaps));
    capacity = std::min(capacity, Room(stageResources, sampledHeaps + storageHeaps + buffers + storageTableDescriptors));
    capacity = std::min(capacity, Room(setImages, sampledHeaps * ShaderRecompiler::RuntimeAbi::StageCount));
    return static_cast<std::uint32_t>(capacity);
}

std::uint32_t SamplerTableCapacityFor(const Context& context, bool updateAfterBind) {
    if (!context.runtimeDescriptorArray || !context.partiallyBound) return 0u;
    if (updateAfterBind && !context.sampledImageUpdateAfterBind) return 0u;
    const auto& indexed = context.descriptorIndexingLimits;
    const std::uint64_t stageSamplers = updateAfterBind ? indexed.maxPerStageDescriptorUpdateAfterBindSamplers : context.limits.maxPerStageDescriptorSamplers;
    const std::uint64_t setSamplers = updateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindSamplers : context.limits.maxDescriptorSetSamplers;
    std::uint64_t capacity = ShaderRecompiler::ImageTableAbi::SamplerElementBudget;
    capacity = std::min(capacity, Room(stageSamplers, ShaderRecompiler::RuntimeAbi::SamplerHeapCapacity));
    capacity = std::min(capacity, Room(setSamplers / ShaderRecompiler::RuntimeAbi::StageCount, ShaderRecompiler::RuntimeAbi::SamplerHeapCapacity));
    return static_cast<std::uint32_t>(capacity - capacity % ShaderRecompiler::ImageTableAbi::SamplerEntryElements);
}

}

std::uint32_t ImageTableCapacity(const Context& context, std::uint32_t storageTableDescriptors) {
    if (storageTableDescriptors != 0u) return ImageTableCapacityFor(context, context.imageTableUpdateAfterBind, storageTableDescriptors);
    const auto capacity = ImageTableCapacityFor(context, false);
    return capacity != 0u ? capacity : ImageTableCapacityFor(context, true);
}

std::uint32_t StorageImageTableCapacity(const Context& context) {
    if (!context.runtimeDescriptorArray || !context.partiallyBound || !context.descriptorIndexing) return 0u;
    constexpr std::uint64_t sampledHeaps = static_cast<std::uint64_t>(ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding - ShaderRecompiler::RuntimeAbi::FirstImageBinding) * ShaderRecompiler::RuntimeAbi::SampledHeapCapacity;
    constexpr std::uint64_t storageHeaps = static_cast<std::uint64_t>(static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Samplers) - ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding) * ShaderRecompiler::RuntimeAbi::StorageHeapCapacity;
    constexpr std::uint64_t buffers = ShaderRecompiler::RuntimeAbi::BufferCapacity + 8u;
    const auto room = [](std::uint64_t limit, std::uint64_t reserve) { return limit > reserve ? limit - reserve : 0u; };
    std::uint64_t capacity = static_cast<std::uint64_t>(ShaderRecompiler::RuntimeAbi::StageCount) * ShaderRecompiler::ImageTableAbi::ImageElementBudget;
    capacity = std::min(capacity, room(context.limits.maxPerStageDescriptorStorageImages, storageHeaps));
    capacity = std::min(capacity, room(context.limits.maxPerStageResources, sampledHeaps + storageHeaps + buffers));
    capacity = std::min(capacity, room(context.limits.maxDescriptorSetStorageImages, storageHeaps * ShaderRecompiler::RuntimeAbi::StageCount));
    return static_cast<std::uint32_t>(capacity);
}

std::uint32_t StorageImageTableDescriptorCount(const Context& context, std::uint32_t elements, bool hasSampledTable) {
    const auto capacity = StorageImageTableCapacity(context);
    if (capacity == 0u) return 0u;
    if (elements == 0u) return 1u;
    const auto raw = std::min(elements, capacity);
    const auto rounded = (static_cast<std::uint64_t>(elements) + 511u) / 512u * 512u;
    auto descriptors = static_cast<std::uint32_t>(std::min<std::uint64_t>(rounded, capacity));
    if (hasSampledTable) {
        const auto oldSampled = std::min(context.imageTableCapacity, ImageTableCapacity(context, raw));
        const auto paddedSampled = std::min(context.imageTableCapacity, ImageTableCapacity(context, descriptors));
        descriptors -= oldSampled - paddedSampled;
    }
    return descriptors;
}

std::uint32_t SamplerTableCapacity(const Context& context) {
    const auto capacity = SamplerTableCapacityFor(context, false);
    return capacity != 0u ? capacity : SamplerTableCapacityFor(context, true);
}

bool ImageTableUpdateAfterBind(const Context& context) {
    return ImageTableCapacityFor(context, false) == 0u && ImageTableCapacityFor(context, true) != 0u;
}

bool SamplerTableUpdateAfterBind(const Context& context) {
    return SamplerTableCapacityFor(context, false) == 0u && SamplerTableCapacityFor(context, true) != 0u;
}

ShaderResources::TableStage& ShaderResources::tableStage(const CompiledShader& shader) {
    for (auto& table : tableStages) {
        if (table.program == shader.program) return table;
    }
    auto& table = tableStages.emplace_back();
    table.stage = shader.stage;
    table.program = shader.program;
    table.poison = shader.program->imageTablePoison;
    table.messages.assign(table.poison.size(), std::string{});
    return table;
}

std::uint32_t ShaderResources::rejectTableElement(TableStage& table, std::span<const std::uint32_t> words, const std::string& message) {
    ShaderRecompiler::ImageTableEntryPoison poison{};
    std::copy(words.begin(), words.end(), poison.words.begin());
    poison.dwordCount = static_cast<std::uint32_t>(words.size());
    poison.resource = ShaderRecompiler::NoTable;
    poison.reason = static_cast<std::uint32_t>(ShaderRecompiler::ImageTableAbi::PoisonReason::DriverRejected);
    table.poison.push_back(poison);
    table.messages.push_back(message);
    faultChecks = true;
    return ShaderRecompiler::ImageTableAbi::PoisonCode(static_cast<std::uint32_t>(table.poison.size() - 1u));
}

void ShaderResources::addSamplerTable(const ShaderRecompiler::DescriptorBinding& binding, VkShaderStageFlags flags, TableStage& table) {
    namespace Abi = ShaderRecompiler::ImageTableAbi;
    Require(binding.kind == ShaderRecompiler::DescriptorKind::Sampler && binding.guestDescriptor.size() == static_cast<std::size_t>(binding.count) * 4u && binding.samplerDepthCompare.size() == binding.count && binding.count % Abi::SamplerEntryElements == 0u, "sampler table binding is malformed");
    Require(context.samplerTableCapacity != 0, "sampler tables need runtimeDescriptorArray and descriptorBindingPartiallyBound, which the device lacks");
    const VkDescriptorBindingFlags samplerTableFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | (context.samplerTableUpdateAfterBind ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0);
    Binding item{{binding.binding, VK_DESCRIPTOR_TYPE_SAMPLER, context.samplerTableCapacity, flags, nullptr}, {}, {}, samplerTableFlags};
    static const bool noSamplerCache = std::getenv("APS5_NO_SAMPLER_CACHE") != nullptr;
    for (std::uint32_t entry = 0; entry < binding.count / Abi::SamplerEntryElements; ++entry) {
        const auto first = entry * Abi::SamplerEntryElements;
        const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(first) * 4u, 4u);
        std::string rejected;
        if (first + Abi::SamplerEntryElements > context.samplerTableCapacity) rejected = "the sampler table exceeds the device's capacity of " + std::to_string(context.samplerTableCapacity) + " samplers";
        for (std::uint32_t element = first; rejected.empty() && element < first + Abi::SamplerEntryElements; ++element) {
            try {
                const auto elementWords = std::span<const std::uint32_t, 4>(binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u, 4u);
                auto resource = DecodeSamplerResource(elementWords);
                resource.compareEnable = binding.samplerDepthCompare.at(element);
                Sampler::Validate(context, resource);
                Require(!Sampler::ReadsOpaqueBlackBorder(resource), "a sampler table entry with an opaque black border needs dynamic texture BC-swizzle pairing, which is not implemented");
                Require(resource.addressModeU != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER && resource.addressModeV != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER && resource.addressModeW != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, "a sampler table entry with a border needs dynamic texture numeric-class pairing, which is not implemented");
                Require(!resource.nonSeamlessCube || context.nonSeamlessCubeMap, "a sampler table entry that disables seamless cube filtering needs VK_EXT_non_seamless_cube_map because dynamic texture/sampler pairing is not implemented");
                Require(!SingleLevelSamplerWords(elementWords, true, false).has_value(), "a sampler table entry with anisotropic ANISO_OVERRIDE needs dynamic texture mip-level pairing, which is not implemented");
            } catch (const std::runtime_error& error) {
                rejected = error.what();
            }
        }
        if (!rejected.empty()) {
            table.samplerRejections.emplace_back(entry, rejectTableElement(table, words, rejected));
            item.imageAllocations.insert(item.imageAllocations.end(), Abi::SamplerEntryElements, std::numeric_limits<std::size_t>::max());
            continue;
        }
        for (std::uint32_t element = first; element < first + Abi::SamplerEntryElements; ++element) {
            const auto elementWords = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 4u, 4u);
            const bool compareEnable = binding.samplerDepthCompare.at(element);
            if (context.samplerCache != nullptr && !noSamplerCache) {
                samplers.push_back(context.samplerCache->Get(context, elementWords, compareEnable, false));
            } else {
                auto resource = DecodeSamplerResource(elementWords, false);
                resource.compareEnable = compareEnable;
                samplers.push_back(std::make_shared<Sampler>(context, resource));
            }
            samplerSources.push_back({{elementWords[0], elementWords[1], elementWords[2], elementWords[3]}, compareEnable, false});
            item.imageAllocations.push_back(samplers.size() - 1u);
        }
    }
    const auto setSamplers = context.samplerTableUpdateAfterBind ? context.descriptorIndexingLimits.maxDescriptorSetUpdateAfterBindSamplers : context.limits.maxDescriptorSetSamplers;
    Require(samplers.size() <= setSamplers, "pipeline sampler descriptors exceed device limits");
    bindings.push_back(std::move(item));
}

void ShaderResources::resolveStorageImageTables() {
    for (const auto index : storageTableBindings) {
        if (index != std::numeric_limits<std::size_t>::max()) bindings[index].imageAllocations.assign(bindings[index].layout.descriptorCount, std::numeric_limits<std::size_t>::max());
    }
    std::vector<std::shared_ptr<StorageTexture>> depthWrites;
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (storageWritten[index] && storageTextures[index] != nullptr && DepthSurfaceAt(storageTextures[index]->Descriptor().baseAddress)) depthWrites.push_back(storageTextures[index]);
    }
    for (auto& table : tableStages) {
        if (table.storageImages == nullptr) continue;
        auto& item = bindings[storageTableBindings[table.set]];
        const auto capacity = item.layout.descriptorCount;
        const auto& binding = *table.storageImages;
        const auto faults = table.map[ShaderRecompiler::ImageTableAbi::HeaderElementFaults];
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
            if (words[0] == 0u && (words[1] & 0xffu) == 0u) continue;
            const auto absolute = static_cast<std::uint64_t>(table.elementBase) + element;
            try {
                Require(absolute < capacity, "the storage image table exceeds the device's capacity of " + std::to_string(capacity) + " images");
                const auto resource = DecodeTextureResource(words);
                Require(binding.imageShapes.at(element) == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2D, "storage image tables require a 2D image view");
                Require(binding.imageWritten.at(element), "storage image tables require a writable image");
                const auto mip = std::min(resource.baseLevel, resource.mipCount - 1u);
                Require(resource.minLod <= mip * 256u, "guest storage texture descriptor clamps its minimum LOD above the level it addresses, which is not implemented");
                const auto guestBytes = DescribeSurface(resource).guestBytes;
                auto texture = cachedStorageTexture(context, words, resource, mip, guestBytes);
                if (texture != nullptr && DepthSurfaceAt(texture->Descriptor().baseAddress)) {
                    auto claimed = depthWrites;
                    claimed.push_back(texture);
                    ValidateStorageDepthWrites(context, claimed);
                    depthWrites = std::move(claimed);
                }
                storageTextures.push_back(std::move(texture));
                storageMips.push_back(mip);
                storageKeys.push_back(resource.dccAddress);
                storageFirstLayer.push_back(false);
                storageWritten.push_back(true);
                storageAtomic.push_back(element < binding.imageAtomic.size() && binding.imageAtomic[element]);
                storageAtomic64.push_back(false);
                describedRanges.push_back({"storage", resource.baseAddress, guestBytes, resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.dccAddress, true});
                item.imageAllocations[absolute] = storageTextures.size() - 1u;
            } catch (const std::runtime_error& error) {
                table.map[faults + element] = rejectTableElement(table, words, error.what());
            }
        }
    }
}

void ShaderResources::resolveImageTables() {
    if (tableBindings[0] == std::numeric_limits<std::size_t>::max() && tableBindings[1] == std::numeric_limits<std::size_t>::max()) return;
    auto& counters = TextureCounts();
    for (std::size_t set = 0; set < tableBindings.size(); ++set) {
        if (tableBindings[set] != std::numeric_limits<std::size_t>::max()) bindings[tableBindings[set]].imageAllocations.assign(std::min(tableElements[set], sampledTableCapacity[set]), std::numeric_limits<std::size_t>::max());
    }
    for (auto& table : tableStages) {
        if (table.images == nullptr) continue;
        auto& item = bindings[tableBindings[table.set]];
        const auto tableCapacity = sampledTableCapacity[table.set];
        const auto& binding = *table.images;
        const auto faults = table.map[ShaderRecompiler::ImageTableAbi::HeaderElementFaults];
        const auto stageSamplers = std::span<const std::shared_ptr<Sampler>>(samplers).subspan(table.firstSampler, table.samplerCount);
        table.elementTextures.assign(binding.count, std::numeric_limits<std::size_t>::max());
        const auto capacityMessage = [&] { return "the image table exceeds the device's capacity of " + std::to_string(tableCapacity) + " images"; };
        std::vector<std::uint32_t> samplerKey;
        for (std::uint32_t sampler = 0; sampler < table.samplerCount; ++sampler) {
            const auto& source = samplerSources.at(table.firstSampler + sampler);
            samplerKey.insert(samplerKey.end(), source.words.begin(), source.words.end());
            samplerKey.push_back((source.compareEnable ? 1u : 0u) | (source.unnormalized ? 2u : 0u) | (source.integerBorder ? 4u : 0u));
        }
        const auto place = [&](std::uint32_t element, std::shared_ptr<Texture> texture, const GuestTextureResource& resource, std::uint64_t guestBytes, bool firstLayer) {
            textures.push_back(std::move(texture));
            textureFirstLayer.push_back(firstLayer);
            describedRanges.push_back({"texture", resource.baseAddress, guestBytes, resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.dccAddress, true});
            table.elementTextures[element] = textures.size() - 1u;
            item.imageAllocations[static_cast<std::size_t>(table.elementBase) + element] = textures.size() - 1u;
        };
        const auto bind = [&](std::uint32_t element, std::span<const std::uint32_t> words, std::shared_ptr<Texture> texture, const GuestTextureResource& resource, std::uint64_t guestBytes, bool firstLayer) -> std::string {
            try {
                RequireFilterMinmax(context, texture->ViewFormat(), binding.imageSamplers.at(element), stageSamplers);
                RequireBorderSwizzle(resource.bcSwizzle, binding.imageSamplers.at(element), stageSamplers);
                RequireNonSeamlessCube(resource.dimension == TextureDimension::kCube, binding.imageSamplers.at(element), stageSamplers);
                const auto numeric = ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) != 0u ? ShaderRecompiler::IrTextureNumericClass::Float : ShaderRecompiler::SampledTextureNumericClass(static_cast<ShaderRecompiler::IrBufferFormat>(resource.format));
                const bool integerImage = numeric == ShaderRecompiler::IrTextureNumericClass::Uint || numeric == ShaderRecompiler::IrTextureNumericClass::Sint;
                for (std::uint32_t sampler = 0; sampler < stageSamplers.size() && sampler < 32u; ++sampler) {
                    if (((binding.imageSamplers.at(element) >> sampler) & 1u) == 0) continue;
                    const auto& source = samplerSources.at(table.firstSampler + sampler);
                    const auto descriptor = DecodeSamplerResource(source.words, source.unnormalized, true, source.integerBorder);
                    const bool border = descriptor.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || descriptor.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || descriptor.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                    Require(!border || source.integerBorder == integerImage, "an image table entry's sampler border color numeric class disagrees with its sampled image class, which needs dynamic pairing and is not implemented");
                    Require(!SingleLevelSamplerWords(source.words, true, false).has_value(), "an image table entry paired with anisotropic ANISO_OVERRIDE needs dynamic texture mip-level pairing, which is not implemented");
                }
            } catch (const std::runtime_error& error) {
                table.map[faults + element] = rejectTableElement(table, words, error.what());
                return error.what();
            }
            place(element, std::move(texture), resource, guestBytes, firstLayer);
            return {};
        };
        const auto state = CurrentTableState();
        std::vector<std::shared_ptr<Texture>> cached;
        if (auto* reused = table.storageImages == nullptr ? FindReusedTable(context.device, binding, state) : nullptr; reused != nullptr && TableCurrent(*reused) && CachedTableTextures(*reused, cached)) {
            const bool replay = TableReplay() && reused->replayable && reused->imageSamplers == binding.imageSamplers && reused->samplers == samplerKey;
            if (!replay) reused->bound.assign(binding.count, {});
            auto next = cached.begin();
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                const auto& entry = reused->elements[element];
                if (static_cast<std::uint64_t>(table.elementBase) + element >= tableCapacity) {
                    if (!entry.rejected) ++next;
                    table.map[faults + element] = rejectTableElement(table, words, capacityMessage());
                } else if (entry.rejected) {
                    table.map[faults + element] = rejectTableElement(table, words, entry.message);
                } else {
                    CompletePendingCpuWrites(context, entry.resource.baseAddress, static_cast<std::size_t>(entry.guestBytes));
                    if (!replay) {
                        reused->bound[element] = bind(element, words, std::move(*next++), entry.resource, entry.guestBytes, entry.firstLayer);
                    } else if (!reused->bound[element].empty()) {
                        ++next;
                        table.map[faults + element] = rejectTableElement(table, words, reused->bound[element]);
                    } else {
                        place(element, std::move(*next++), entry.resource, entry.guestBytes, entry.firstLayer);
                    }
                }
            }
            if (!replay) {
                reused->imageSamplers = binding.imageSamplers;
                reused->samplers = samplerKey;
                reused->replayable = static_cast<std::uint64_t>(table.elementBase) + binding.count <= tableCapacity;
            }
            (replay ? counters.tablesReplayed : counters.tablesReused).fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        ReusedTable fresh;
        fresh.bound.assign(binding.count, {});
        bool reusable = state.epoch != 0 && table.storageImages == nullptr;
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
            if (words[0] == 0u && (words[1] & 0xffu) == 0u) {
                reusable = false;
                continue;
            }
            const auto absolute = static_cast<std::uint64_t>(table.elementBase) + element;
            if (absolute >= tableCapacity) {
                table.map[faults + element] = rejectTableElement(table, words, capacityMessage());
                reusable = false;
                continue;
            }
            GuestTextureResource resource{};
            std::uint64_t guestBytes = 0;
            bool firstLayer = false;
            auto shape = ShaderRecompiler::DescriptorImageShape::Image2D;
            try {
                resource = DecodeTextureResource(words);
                shape = binding.imageShapes.at(element);
                const bool depthCompare = element < binding.imageDepthCompare.size() && binding.imageDepthCompare[element];
                firstLayer = shape == ShaderRecompiler::DescriptorImageShape::Image2D && (resource.dimension == TextureDimension::k2DArray || (depthCompare && resource.dimension == TextureDimension::k1DArray));
                const bool depth1D = depthCompare && ((shape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k1D) || (shape == ShaderRecompiler::DescriptorImageShape::Image2DArray && resource.dimension == TextureDimension::k1DArray));
                if (!firstLayer && !depth1D && !MatchesGuestDimension(shape, resource.dimension)) throw std::runtime_error("AGC graphics: guest texture dimension disagrees with the table entry's image shape (shape " + std::to_string(static_cast<int>(shape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")");
                guestBytes = DescribeSurface(resource).guestBytes;
                if (!GuestMemory::Accessible(reinterpret_cast<const void*>(resource.baseAddress), static_cast<std::size_t>(guestBytes)) && GuestMemory::CommittedRanges(resource.baseAddress, static_cast<std::size_t>(guestBytes), false).empty()) throw std::runtime_error("AGC graphics: the table entry's surface at 0x" + HexText(resource.baseAddress) + " (0x" + HexText(guestBytes) + " bytes) is not mapped");
                static_cast<void>(ResolveTextureFormat(resource.format));
            } catch (const std::runtime_error& error) {
                table.map[faults + element] = rejectTableElement(table, words, error.what());
                if (reusable) fresh.elements.push_back({{}, {}, {}, 0, false, true, error.what()});
                continue;
            }
            std::array<std::uint32_t, 8> narrowed{};
            const auto sampledWords = firstLayerWords(words, binding.imageShapes.at(element), narrowed, true);
            if (sampledWords.data() != words.data()) {
                resource = DecodeTextureResource(sampledWords);
                guestBytes = DescribeSurface(resource).guestBytes;
            }
            CompletePendingCpuWrites(context, resource.baseAddress, static_cast<std::size_t>(guestBytes));
            if (resource.dccAddress != 0) CompletePendingCpuWrites(context, resource.dccAddress, DccKeyBytes(guestBytes));
            std::shared_ptr<Texture> texture;
            const auto components = ViewComponents(resource);
            const bool depthCompare = binding.imageDepthCompare.at(element);
            const VkImageViewType multisampleViewType = shape == ShaderRecompiler::DescriptorImageShape::Image2DMultisampleArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
            try {
                texture = cachedTexture(context, sampledWords, resource, components, guestBytes, depthCompare, multisampleViewType);
            } catch (const std::runtime_error& error) {
                table.map[faults + element] = rejectTableElement(table, words, error.what());
                if (reusable) fresh.elements.push_back({{}, {}, {}, 0, false, true, error.what()});
                continue;
            }
            if (reusable) {
                const auto& log = lookupLog();
                const auto* looked = log.empty() ? nullptr : &log.back();
                reusable = looked != nullptr && looked->object == texture.get() && looked->source == nullptr && looked->generation != 0 && looked->keys == DccKeys::Uncompressed && resource.dccAddress == 0;
                if (reusable) {
                    fresh.elements.push_back({MakeTextureKey(context.device, sampledWords, components, depthCompare), texture, resource, guestBytes, firstLayer, false, {}});
                    fresh.queries.push_back({resource.baseAddress, static_cast<std::size_t>(guestBytes), looked->generation});
                    fresh.ranges.emplace_back(resource.baseAddress, resource.baseAddress + guestBytes);
                }
            }
            fresh.bound[element] = bind(element, sampledWords, std::move(texture), resource, guestBytes, firstLayer);
        }
        counters.tablesResolved.fetch_add(1, std::memory_order_relaxed);
        if (reusable && CurrentTableState() == state) {
            const auto serial = StorageTexture::PendingSerial();
            if (!StorageTexture::AnyPendingOverlaps(fresh.ranges) && StorageTexture::PendingSerial() == serial) {
                fresh.device = context.device;
                fresh.state = state;
                fresh.pendingSerial = serial;
                fresh.imageSamplers = binding.imageSamplers;
                fresh.samplers = std::move(samplerKey);
                fresh.replayable = static_cast<std::uint64_t>(table.elementBase) + binding.count <= tableCapacity;
                RememberTable(std::move(fresh), binding);
            }
        }
    }
    reportTextureCounters();
}

std::uint64_t SampledLookupKeyScans() {
    return sampledLookupKeyScans;
}

ImageTableReuseCounts ImageTableReuses() {
    const auto& counters = TextureCounts();
    return {counters.tablesReused.load(std::memory_order_relaxed) + counters.tablesReplayed.load(std::memory_order_relaxed), counters.tablesResolved.load(std::memory_order_relaxed), counters.tablesReplayed.load(std::memory_order_relaxed), counters.tableWritesKept.load(std::memory_order_relaxed), counters.tableWritesMade.load(std::memory_order_relaxed)};
}

void ShaderResources::finishImageTables() {
    namespace Abi = ShaderRecompiler::ImageTableAbi;
    if (tableStages.empty()) return;
    const auto overlaps = [](std::uint64_t base, std::uint64_t size, std::uint64_t address, std::uint64_t bytes) { return size != 0 && bytes != 0 && (base <= address ? address - base < size : base - address < bytes); };
    std::vector<BdaResources::ImageTableFaults> faults;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (auto& table : tableStages) {
        auto& map = table.map;
        for (const auto& [entry, code] : table.samplerRejections) map[map[Abi::HeaderSamplerFaults] + entry] = code;
        for (std::uint32_t index = 0; index < map[Abi::HeaderTableCount]; ++index) {
            const auto header = Abi::TableHeader(index);
            if (map[header + Abi::TableFault] != 0u) continue;
            const auto base = static_cast<std::uint64_t>(map[header + Abi::TableBaseLow]) | (static_cast<std::uint64_t>(map[header + Abi::TableBaseHigh]) << 32u);
            const auto size = static_cast<std::uint64_t>(map[header + Abi::TableSizeLow]) | (static_cast<std::uint64_t>(map[header + Abi::TableSizeHigh]) << 32u);
            const auto tableOverlaps = [&](std::uint64_t address, std::uint64_t bytes) {
                if (overlaps(base, size, address, bytes)) return true;
                if (index >= table.program->imageTableReadRanges.size()) return false;
                const auto& reads = table.program->imageTableReadRanges[index];
                return std::any_of(reads.begin(), reads.end(), [&](const auto& range) { return overlaps(range.first, range.second, address, bytes); });
            };
            bool written = false;
            for (const auto& allocation : allocations) written = written || (allocation.guest && allocation.written && tableOverlaps(allocation.address, allocation.size));
            for (std::size_t image = 0; image < storageTextures.size(); ++image) written = written || (storageWritten[image] && storageTextures[image] != nullptr && tableOverlaps(storageTextures[image]->Descriptor().baseAddress, storageTextures[image]->GuestBytes()));
            if (!written) continue;
            ShaderRecompiler::ImageTableEntryPoison poison{};
            poison.resource = index;
            poison.reason = static_cast<std::uint32_t>(Abi::PoisonReason::HeapWritten);
            table.poison.push_back(poison);
            table.messages.push_back("the table at 0x" + HexText(base) + " is written by its own dispatch");
            map[header + Abi::TableFault] = Abi::PoisonCode(static_cast<std::uint32_t>(table.poison.size() - 1u));
            faultChecks = true;
        }
        map[Abi::HeaderPoisonCount] = static_cast<std::uint32_t>(table.poison.size());
        auto& allocation = allocations.at(table.mapAllocation);
        Require(allocation.buffer != nullptr && allocation.size == map.size() * sizeof(std::uint32_t), "image table map buffer disagrees with the map");
        std::memcpy(allocation.buffer->Bytes().data(), map.data(), allocation.size);
        faults.push_back({IrStage(table.stage), table.program->imageTableShader, table.poison, table.messages});
        ranges.insert(ranges.end(), table.program->imageTableRanges.begin(), table.program->imageTableRanges.end());
    }
    if (bda == nullptr) return;
    bda->SetImageTables(std::move(faults), std::move(ranges));
}

std::vector<PresyncSurface> ShaderResources::PresyncSurfaces() const {
    std::vector<PresyncSurface> surfaces;
    // A surface's lookup reads guest memory on the CPU unless it is served GPU-direct from a host
    // import: a storage image of imported memory uploads and refreshes from the import, and a
    // sampled texture of one views that image (SampledFromStorageEligible), so neither waits for
    // the producer.
    const auto consider = [&](std::uint32_t format, std::uint64_t address, std::uint64_t guestBytes, bool sampled) {
        const bool gpuDirect = sampled ? SampledFromStorageEligible(context, format, address, guestBytes) : HostImportCovers(context, address, static_cast<std::size_t>(guestBytes));
        if (!gpuDirect) surfaces.push_back({address, guestBytes, false});
    };
    static const bool keyRanges = std::getenv("APS5_NO_DCC_PRESYNC") == nullptr;
    const auto considerKeys = [&](std::uint64_t dccAddress, std::uint64_t guestBytes) {
        const auto keyBytes = DccKeyBytes(guestBytes);
        if (keyRanges && dccAddress != 0 && keyBytes != 0) surfaces.push_back({dccAddress, keyBytes, true});
    };
    if (completed) {
        // A cached object (the resource cache): Revalidate repeats the lookups of the objects the
        // build made, which never change afterwards (a lookup returning another object fails the
        // revalidation instead), so the surfaces come from the build's own record of them. A
        // sampled texture viewing a storage image refreshes like that image; a snapshot texture
        // over an eligible surface is replaced by a view without reading. Storage images are
        // checked on their own surface (a mip chain can exceed the element's descriptor). Read
        // without the device lock: these members are fixed once the build completed.
        std::size_t textureIndex = 0;
        for (const auto& range : describedRanges) {
            if (std::strcmp(range.kind, "texture") == 0 && textureIndex < textures.size()) {
                const auto& texture = textures[textureIndex++];
                if (texture->ViewsStorageImage()) consider(range.format, range.address, range.bytes, false);
                else consider(range.format, range.address, range.bytes, true);
                considerKeys(range.dccAddress, range.bytes);
            }
        }
        for (std::size_t index = 0; index < storageTextures.size(); ++index) {
            if (storageTextures[index] == nullptr || (index != 0 && storageTextures[index] == storageTextures[index - 1])) continue;
            const auto& image = *storageTextures[index];
            consider(image.Descriptor().format, image.Descriptor().baseAddress, image.GuestBytes(), false);
            considerKeys(image.Descriptor().dccAddress, image.GuestBytes());
        }
        return surfaces;
    }
    if (!imageRecords.empty()) {
        for (const auto& record : imageRecords) {
            if (!record.decoded) continue;
            consider(record.resource.format, record.resource.baseAddress, record.guestBytes, record.sampled);
            considerKeys(record.resource.dccAddress, record.guestBytes);
        }
        return surfaces;
    }
    // Stage A has not run (an address-based build, or the pass is off): decoded from the shader.
    if (deferredCompute.program == nullptr) return surfaces;
    forEachImageElement(*deferredCompute.program, [&](const ShaderRecompiler::DescriptorBinding& binding, std::uint32_t, std::span<const std::uint32_t> words) {
        if (IsNullTextureResource(words)) return;
        const auto resource = DecodeTextureResource(words);
        const auto guestBytes = DescribeSurface(resource).guestBytes;
        consider(resource.format, resource.baseAddress, guestBytes, binding.kind == ShaderRecompiler::DescriptorKind::SampledImage);
        considerKeys(resource.dccAddress, guestBytes);
    });
    for (const auto& binding : deferredCompute.program->bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::ImageTable) continue;
        for (std::size_t element = 0; element < binding.guestDescriptor.size() / 8u; ++element) {
            try {
                const auto resource = DecodeTextureResource(std::span<const std::uint32_t>(binding.guestDescriptor).subspan(element * 8u, 8u));
                const auto guestBytes = DescribeSurface(resource).guestBytes;
                consider(resource.format, resource.baseAddress, guestBytes, binding.kind == ShaderRecompiler::DescriptorKind::SampledImage);
                considerKeys(resource.dccAddress, guestBytes);
            } catch (const std::runtime_error&) {
            }
        }
    }
    return surfaces;
}

ShaderResources::~ShaderResources() {
    release();
}

std::shared_ptr<const TableDescriptors> ShaderResources::keptTable(std::size_t set, std::shared_ptr<const std::vector<std::weak_ptr<Texture>>>& held) noexcept {
    const auto& table = tableWrites[set];
    if (!table.written || KeptSets() == 0u || table.firstImage + table.images > descriptorImages.size()) return nullptr;
    try {
        if (held == nullptr) {
            std::vector<std::weak_ptr<Texture>> weak;
            weak.reserve(textures.size());
            for (const auto& texture : textures) {
                if (texture != nullptr) weak.push_back(texture);
            }
            held = std::make_shared<const std::vector<std::weak_ptr<Texture>>>(std::move(weak));
        }
        auto kept = std::make_shared<TableDescriptors>();
        kept->binding = table.binding;
        kept->runs = table.runs;
        kept->images.assign(descriptorImages.begin() + static_cast<std::ptrdiff_t>(table.firstImage), descriptorImages.begin() + static_cast<std::ptrdiff_t>(table.firstImage + table.images));
        kept->textures = held;
        return kept;
    } catch (...) {
        return nullptr;
    }
}

void ShaderResources::release() noexcept {
    // A set from the cache's pool chain goes back to it; a dedicated pool dies with its set.
    std::shared_ptr<const std::vector<std::weak_ptr<Texture>>> held;
    if (cacheAllocation.pool && context.descriptorCache != nullptr) context.descriptorCache->Free(cacheAllocation, keptTable(0, held));
    if (pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
    if (_layout && ownsLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, _layout, nullptr);
    if (fragment.cacheAllocation.pool && context.descriptorCache != nullptr) context.descriptorCache->Free(fragment.cacheAllocation, keptTable(1, held));
    if (fragment.pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, fragment.pool, nullptr);
    if (fragment.layout && fragment.ownsLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, fragment.layout, nullptr);
    cacheAllocation = {};
    pool = VK_NULL_HANDLE;
    _set = VK_NULL_HANDLE;
    _layout = VK_NULL_HANDLE;
    fragment = {};
    tableWrites = {};
}

std::uint32_t ShaderResources::setOf(const Binding& binding) const {
    constexpr auto count = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
    if (binding.set >= 0) return static_cast<std::uint32_t>(binding.set);
    return splitSets && binding.layout.binding / count == static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Stage::Fragment) ? 1u : 0u;
}

VkDescriptorSetLayout ShaderResources::Layout() const {
    return _layout;
}

std::vector<VkDescriptorSetLayoutBinding> ShaderResources::LayoutBindings() const {
    std::vector<VkDescriptorSetLayoutBinding> result;
    for (const auto& binding : bindings) result.push_back(binding.layout);
    return result;
}

bool ShaderResources::LayoutUpdateAfterBind() const {
    return updateAfterBind;
}

ShaderResources::DrawBindings::~DrawBindings() {
    if (cache == nullptr) return;
    const auto release = [&](VkDescriptorSetLayout setLayout, const DescriptorCache::SetAllocation& set) {
        if (set.set == VK_NULL_HANDLE) return;
        if (setLayout != VK_NULL_HANDLE) cache->Recycle(setLayout, set);
        else cache->Free(set);
    };
    release(layout, allocation);
    release(fragmentLayout, fragmentAllocation);
}

std::optional<std::vector<ShaderResources::MovedBuffer>> ShaderResources::MovedReadOnlyBuffers(std::span<const CompiledShader> shaders, Recorder& recorder) const {
    std::vector<MovedBuffer> moved;
    if (_set == VK_NULL_HANDLE && fragment.set == VK_NULL_HANDLE) return moved;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) return std::nullopt;
        for (const auto& binding : shader.program->bindings) {
            if (DataRole(binding.role)) {
                const auto kept = std::find_if(bindings.begin(), bindings.end(), [&](const Binding& item) { return item.layout.binding == binding.binding; });
                if (kept == bindings.end() || kept->allocations.size() != 1) return std::nullopt;
                const auto index = kept->allocations.front();
                const auto& item = allocations[index];
                if (item.guest || item.buffer == nullptr || item.size != binding.guestDescriptor.size() * sizeof(std::uint32_t)) return std::nullopt;
                const bool patched = std::any_of(dataPatches.begin(), dataPatches.end(), [&](const DataPatch& patch) { return patch.allocation == index; });
                const bool same = !item.dataWords.empty() ? item.dataWords == binding.guestDescriptor : !patched && std::memcmp(item.buffer->Bytes().data(), binding.guestDescriptor.data(), item.size) == 0;
                if (same) continue;
                if (item.dataWords.empty() && patched) return std::nullopt;
                if (moved.empty()) moved.reserve(allocations.size());
                moved.push_back({index, 0, item.size, binding.guestDescriptor});
                continue;
            }
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
            const auto kept = std::find_if(bindings.begin(), bindings.end(), [&](const Binding& item) { return item.layout.binding == binding.binding; });
            if (kept == bindings.end() || kept->allocations.size() != binding.count || binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u) return std::nullopt;
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u;
                const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
                const auto address = descriptor.Base48();
                const auto size = GuestBufferBytes(context, std::span<const std::uint32_t>(words, 4));
                const auto index = kept->allocations[element];
                const auto& item = allocations[index];
                const bool empty = size == 0 || address == 0;
                if (!item.guest) {
                    if (empty) continue;
                    return std::nullopt;
                }
                if (item.address == address && item.size == size) continue;
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                if (written || empty || item.written || size > context.limits.maxStorageBufferRange) return std::nullopt;
                const auto begin = address - item.adjustment;
                const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(size, item.adjustment));
                if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) return std::nullopt;
                if (guestMemory.WritesOverlap(begin, bytes) || recorder.PendingWriteOverlaps(begin, bytes) || PendingStorageOverlaps(begin, bytes, nullptr) || AnyShadowedOverlaps(begin, bytes)) return std::nullopt;
                if (moved.empty()) moved.reserve(allocations.size());
                moved.push_back({index, address, static_cast<std::size_t>(size)});
            }
        }
    }
    if (usesBda && !moved.empty() && !(bdaTemplate && BdaTemplatesOn())) return std::nullopt;
    return moved;
}

const char* ShaderResources::MoveRefusalName(MoveRefusal refusal) {
    switch (refusal) {
        case MoveRefusal::None: return "none";
        case MoveRefusal::Shape: return "bindings differ";
        case MoveRefusal::NotGuest: return "not read in place";
        case MoveRefusal::Written: return "written or emptied";
        case MoveRefusal::Limit: return "over the limit";
        case MoveRefusal::Inaccessible: return "unmapped";
        case MoveRefusal::Overlap: return "pending write";
        case MoveRefusal::Bindings: return "no bindings made";
        case MoveRefusal::Count: break;
    }
    return "?";
}

std::optional<std::vector<ShaderResources::MovedBuffer>> ShaderResources::MovedDispatchBuffers(const CompiledShader& shader, Recorder* recorder, std::size_t maxBytes, MoveProof& proof) const {
    proof = {};
    const auto refuse = [&](MoveRefusal why) {
        proof.refusal = why;
        return std::optional<std::vector<MovedBuffer>>{};
    };
    std::vector<MovedBuffer> moved;
    if (shader.program == nullptr || usesBda) return refuse(MoveRefusal::Shape);
    if (_set == VK_NULL_HANDLE) return moved;
    for (const auto& binding : shader.program->bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
        const auto kept = std::find_if(bindings.begin(), bindings.end(), [&](const Binding& item) { return item.layout.binding == binding.binding; });
        if (kept == bindings.end() || kept->allocations.size() != binding.count || binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u) return refuse(MoveRefusal::Shape);
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u;
            const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
            const auto address = descriptor.Base48();
            const auto size = guestBufferBytes(std::span<const std::uint32_t>(words, 4));
            const auto& item = allocations[kept->allocations[element]];
            const bool empty = size == 0 || address == 0;
            if (!item.guest) {
                if (empty) continue;
                return refuse(MoveRefusal::NotGuest);
            }
            if (item.address == address && item.size == size) continue;
            const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
            if (written || empty || item.written) return refuse(MoveRefusal::Written);
            if (size > context.limits.maxStorageBufferRange) return refuse(MoveRefusal::Limit);
            const auto begin = address - item.adjustment;
            const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(size, item.adjustment));
            if (maxBytes != 0 && proof.bytes + bytes > maxBytes) return refuse(MoveRefusal::Limit);
            if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) return refuse(MoveRefusal::Inaccessible);
            const bool pending = recorder != nullptr ? recorder->PendingWriteOverlaps(begin, bytes) : Recorder::SnapshotWriteOverlaps(begin, bytes);
            const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> range{{{begin, begin + bytes}}};
            if (guestMemory.WritesOverlap(begin, bytes) || pending || StorageTexture::AnyPendingOverlaps(range) || AnyShadowedOverlaps(begin, bytes)) return refuse(MoveRefusal::Overlap);
            moved.push_back({kept->allocations[element], address, static_cast<std::size_t>(size)});
            ++proof.buffers;
            proof.bytes += bytes;
        }
    }
    return moved;
}

namespace {

enum class DrawSnapshotPath : std::size_t { Upload, TrackedHit, OwnedHit, TrackedCopy, UntrackedCopy, Count };
constexpr std::size_t DrawSnapshotPaths = static_cast<std::size_t>(DrawSnapshotPath::Count);
constexpr std::size_t OwnedSnapshotMisses = static_cast<std::size_t>(Recorder::OwnedSnapshotMiss::Count);

struct DrawSnapshotCounters {
    std::array<std::atomic<std::uint64_t>, DrawSnapshotPaths> count{};
    std::array<std::atomic<std::uint64_t>, DrawSnapshotPaths> bytes{};
    std::array<std::atomic<std::uint64_t>, OwnedSnapshotMisses> misses{};
    std::atomic<std::uint64_t> armed{0}, refused{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ShaderResources::DrawInputPath::Count)> inputCount{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ShaderResources::DrawInputPath::Count)> inputBytes{};
    std::array<std::atomic<std::uint64_t>, OwnedSnapshotMisses> inputMisses{};
    std::atomic<std::uint64_t> inputArmed{0}, inputRefused{0};
    std::atomic<std::int64_t> lastReport{0};
};

DrawSnapshotCounters& SnapshotPathCounts() {
    static DrawSnapshotCounters counters;
    return counters;
}

void reportDrawSnapshots(const Recorder& recorder) {
    auto& counters = SnapshotPathCounts();
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load();
    if (nowMs - last < 10000 || !counters.lastReport.compare_exchange_strong(last, nowMs)) return;
    static std::array<std::uint64_t, DrawSnapshotPaths> previousCount{}, previousBytes{};
    static std::array<std::uint64_t, OwnedSnapshotMisses> previousMisses{};
    static std::uint64_t previousArmed = 0, previousRefused = 0;
    std::array<unsigned long long, DrawSnapshotPaths> count{};
    std::array<double, DrawSnapshotPaths> mib{};
    for (std::size_t index = 0; index < DrawSnapshotPaths; ++index) {
        const auto c = counters.count[index].load();
        const auto b = counters.bytes[index].load();
        count[index] = static_cast<unsigned long long>(c - previousCount[index]);
        mib[index] = static_cast<double>(b - previousBytes[index]) / (1024.0 * 1024.0);
        previousCount[index] = c;
        previousBytes[index] = b;
    }
    std::array<unsigned long long, OwnedSnapshotMisses> misses{};
    for (std::size_t index = 0; index < OwnedSnapshotMisses; ++index) {
        const auto m = counters.misses[index].load();
        misses[index] = static_cast<unsigned long long>(m - previousMisses[index]);
        previousMisses[index] = m;
    }
    const auto armed = counters.armed.load();
    const auto refused = counters.refused.load();
    using Path = DrawSnapshotPath;
    using Miss = Recorder::OwnedSnapshotMiss;
    const auto at = [](auto value) { return static_cast<std::size_t>(value); };
    AgcDriver::ProfilePrint_nid_no_patch("[draw-snapshots] last 10 s (APS5_NO_OWNED_DRAW_SNAPSHOTS %s): uploads %llu (%.1f MiB copied), tracked reuses %llu (%.1f MiB), owned reuses %llu (%.1f MiB not copied), copies of tracked memory %llu (%.1f MiB), copies of untracked memory %llu (%.1f MiB); owned misses: absent %llu, backing off %llu, words moved %llu, stamps moved %llu, mapping %llu; armed %llu, arms refused %llu; %.0f MiB owned copies held\n", RuntimeSwitches::Set(RuntimeSwitch::NoOwnedDrawSnapshots) ? "set" : "unset", count[at(Path::Upload)], mib[at(Path::Upload)], count[at(Path::TrackedHit)], mib[at(Path::TrackedHit)], count[at(Path::OwnedHit)], mib[at(Path::OwnedHit)], count[at(Path::TrackedCopy)], mib[at(Path::TrackedCopy)], count[at(Path::UntrackedCopy)], mib[at(Path::UntrackedCopy)], misses[at(Miss::Absent)], misses[at(Miss::BackingOff)], misses[at(Miss::WordsMoved)], misses[at(Miss::StampsMoved)], misses[at(Miss::Mapping)], static_cast<unsigned long long>(armed - previousArmed), static_cast<unsigned long long>(refused - previousRefused), static_cast<double>(recorder.OwnedSnapshotBytes()) / (1024.0 * 1024.0));
    previousArmed = armed;
    previousRefused = refused;
    constexpr std::size_t InputPaths = static_cast<std::size_t>(ShaderResources::DrawInputPath::Count);
    static std::array<std::uint64_t, InputPaths> previousInputCount{}, previousInputBytes{};
    static std::array<std::uint64_t, OwnedSnapshotMisses> previousInputMisses{};
    static std::uint64_t previousInputArmed = 0, previousInputRefused = 0, previousVerified = 0, previousDiffering = 0;
    std::array<unsigned long long, InputPaths> inputCount{};
    std::array<double, InputPaths> inputMib{};
    for (std::size_t index = 0; index < InputPaths; ++index) {
        const auto c = counters.inputCount[index].load();
        const auto b = counters.inputBytes[index].load();
        inputCount[index] = static_cast<unsigned long long>(c - previousInputCount[index]);
        inputMib[index] = static_cast<double>(b - previousInputBytes[index]) / (1024.0 * 1024.0);
        previousInputCount[index] = c;
        previousInputBytes[index] = b;
    }
    std::array<unsigned long long, OwnedSnapshotMisses> inputMisses{};
    for (std::size_t index = 0; index < OwnedSnapshotMisses; ++index) {
        const auto m = counters.inputMisses[index].load();
        inputMisses[index] = static_cast<unsigned long long>(m - previousInputMisses[index]);
        previousInputMisses[index] = m;
    }
    const auto inputArmed = counters.inputArmed.load();
    const auto inputRefused = counters.inputRefused.load();
    const auto verify = Recorder::OwnedSnapshotVerifyCounts();
    using Input = ShaderResources::DrawInputPath;
    AgcDriver::ProfilePrint_nid_no_patch("[draw-snapshots] vertex and index inputs, last 10 s (APS5_NO_OWNED_VERTEX_INPUTS %s): tracked reuses %llu (%.1f MiB), owned reuses %llu (%.1f MiB not copied), copies of tracked memory %llu (%.1f MiB), copies of untracked memory %llu (%.1f MiB); owned misses: absent %llu, backing off %llu, words moved %llu, stamps moved %llu, mapping %llu; armed %llu, arms refused %llu; owned verify (both kinds) %llu checked, %llu differing\n", RuntimeSwitches::Set(RuntimeSwitch::NoOwnedVertexInputs) ? "set" : "unset", inputCount[at(Input::TrackedHit)], inputMib[at(Input::TrackedHit)], inputCount[at(Input::OwnedHit)], inputMib[at(Input::OwnedHit)], inputCount[at(Input::TrackedCopy)], inputMib[at(Input::TrackedCopy)], inputCount[at(Input::UntrackedCopy)], inputMib[at(Input::UntrackedCopy)], inputMisses[at(Miss::Absent)], inputMisses[at(Miss::BackingOff)], inputMisses[at(Miss::WordsMoved)], inputMisses[at(Miss::StampsMoved)], inputMisses[at(Miss::Mapping)], static_cast<unsigned long long>(inputArmed - previousInputArmed), static_cast<unsigned long long>(inputRefused - previousInputRefused), static_cast<unsigned long long>(verify[0] - previousVerified), static_cast<unsigned long long>(verify[1] - previousDiffering));
    previousInputArmed = inputArmed;
    previousInputRefused = inputRefused;
    previousVerified = verify[0];
    previousDiffering = verify[1];
}

bool DrawSnapshotsProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

void countDrawSnapshot(DrawSnapshotPath path, std::size_t bytes) {
    if (!DrawSnapshotsProfiled()) return;
    auto& counters = SnapshotPathCounts();
    counters.count[static_cast<std::size_t>(path)].fetch_add(1, std::memory_order_relaxed);
    counters.bytes[static_cast<std::size_t>(path)].fetch_add(bytes, std::memory_order_relaxed);
}

void countOwnedSnapshotMiss(Recorder::OwnedSnapshotMiss miss, bool armed, bool refused) {
    if (!DrawSnapshotsProfiled()) return;
    auto& counters = SnapshotPathCounts();
    counters.misses[static_cast<std::size_t>(miss)].fetch_add(1, std::memory_order_relaxed);
    if (armed) counters.armed.fetch_add(1, std::memory_order_relaxed);
    if (refused) counters.refused.fetch_add(1, std::memory_order_relaxed);
}

void verifyGpuSource(const Context& context, Recorder& recorder, const GuestBufferMemory::GpuSource& source, std::uint64_t address, std::size_t bytes) {
    std::vector<std::byte> expected(bytes);
    std::memcpy(expected.data(), reinterpret_cast<const void*>(address), bytes);
    const bool watched = GuestMemory::Watched(address, bytes);
    if (watched) GuestMemory::CollectWritesUncached(address, bytes);
    const bool raced = watched && GuestMemory::CpuStoredSince(address, bytes, source.since);
    auto readback = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    const auto commands = recorder.Commands();
    CopyBuffer(context, commands, source.buffer->Handle(), source.offset, readback->Handle(), 0, bytes);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    recorder.Keep(source.buffer);
    recorder.OnComplete([readback, expected = std::move(expected), address, raced] {
        const auto actual = readback->Bytes();
        std::size_t first = 0;
        while (first < expected.size() && expected[first] == actual[first]) ++first;
        const bool same = first == expected.size();
        GuestBufferMemory::CountGpuSourceVerify(same);
        if (!same) std::fprintf(stderr, "[gpu-sourced] verify %s: 0x%llx+0x%zx differs first at +0x%zx (import %02x, bound %02x)\n", raced ? "race (a CPU store after the copy-in that the draw's collect had not seen)" : "STALE", static_cast<unsigned long long>(address), expected.size(), first, std::to_integer<unsigned>(expected[first]), std::to_integer<unsigned>(actual[first]));
    });
}

}

void ShaderResources::CountDrawInputPath(DrawInputPath path, std::size_t bytes, std::uint8_t ownedMiss, bool owned, bool armed) {
    if (!DrawSnapshotsProfiled()) return;
    auto& counters = SnapshotPathCounts();
    counters.inputCount[static_cast<std::size_t>(path)].fetch_add(1, std::memory_order_relaxed);
    counters.inputBytes[static_cast<std::size_t>(path)].fetch_add(bytes, std::memory_order_relaxed);
    if (!owned) return;
    if (ownedMiss < OwnedSnapshotMisses) counters.inputMisses[ownedMiss].fetch_add(1, std::memory_order_relaxed);
    if (armed) counters.inputArmed.fetch_add(1, std::memory_order_relaxed);
    else if (ownedMiss != static_cast<std::uint8_t>(Recorder::OwnedSnapshotMiss::BackingOff)) counters.inputRefused.fetch_add(1, std::memory_order_relaxed);
}

std::size_t ShaderResources::MaxDrawSnapshotBytes() {
    const char* text = std::getenv("APS5_DRAW_SNAPSHOT_MAX_MIB");
    const auto mib = text != nullptr ? std::strtoull(text, nullptr, 10) : 64ull;
    return static_cast<std::size_t>(mib * 1024ull * 1024ull);
}

std::size_t ShaderResources::DrawUploadLimit() {
    static const std::size_t limit = [] {
        const char* value = std::getenv("APS5_DRAW_UPLOAD_MAX");
        return value != nullptr ? static_cast<std::size_t>(std::strtoull(value, nullptr, 10)) : std::size_t{4096};
    }();
    return limit;
}

std::shared_ptr<ShaderResources::DrawBindings> ShaderResources::PrepareDrawBindings(Recorder& recorder, std::span<const MovedBuffer> moved, bool movedOnly) const {
    if (_set == VK_NULL_HANDLE && fragment.set == VK_NULL_HANDLE) return {};
    if (usesBda) {
        if (!bdaTemplate || moved.empty()) return {};
        movedOnly = true;
        CountPerUseAddressSet();
    }
    thread_local std::vector<std::pair<std::uint64_t, std::uint64_t>>* readsSlot = nullptr;
    auto& reads = ShaderRecompiler::ThreadOwned(readsSlot);
    guestMemory.InPlaceReads(reads);
    const auto maxSnapshotBytes = maxDrawSnapshotBytes;
    const auto uploadLimit = DrawUploadLimit();
    std::size_t totalSnapshotBytes = 0;
    auto result = std::make_shared<DrawBindings>();
    thread_local std::vector<std::size_t>* selectedSlot = nullptr;
    auto& selected = ShaderRecompiler::ThreadOwned(selectedSlot);
    selected.clear();
    std::array<std::uint64_t, 3> stamp{};
    bool stamped = false;
    const bool gpuSourced = GuestBufferMemory::GpuSourcedDrawBuffers();
    const bool ownedSnapshots = !RuntimeSwitches::Set(RuntimeSwitch::NoOwnedDrawSnapshots);
    const auto keep = [&](DrawBindings::Snapshot snapshot) {
        if (result->snapshots.empty()) result->snapshots.reserve(allocations.size());
        result->snapshots.push_back(std::move(snapshot));
    };
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        const auto& item = allocations[index];
        const auto override = std::find_if(moved.begin(), moved.end(), [&](const MovedBuffer& entry) { return entry.allocation == index; });
        if (override != moved.end() && !override->words.empty()) {
            constexpr std::size_t MovedWordBufferSlots = 8;
            const auto held = std::find_if(movedWordBuffers.begin(), movedWordBuffers.end(), [&](const MovedWordBuffer& entry) { return entry.allocation == index && entry.size == override->size && std::ranges::equal(entry.words, override->words); });
            std::shared_ptr<Buffer> buffer;
            if (held != movedWordBuffers.end()) {
                buffer = held->buffer;
            } else {
                const MemoryStats::OwnerScope memStatsOwner(MemoryStats::Owner::ShaderResources);
                buffer = std::make_shared<Buffer>(context, override->size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                std::memcpy(buffer->Bytes().data(), override->words.data(), override->size);
                for (const auto& patch : dataPatches) {
                    if (patch.allocation == index && patch.byte < override->size) buffer->Bytes()[patch.byte] = static_cast<std::byte>(patch.adjustment);
                }
                MovedWordBuffer entry{index, {override->words.begin(), override->words.end()}, override->size, buffer};
                if (movedWordBuffers.size() < MovedWordBufferSlots) movedWordBuffers.push_back(std::move(entry));
                else movedWordBuffers[movedWordBufferNext++ % MovedWordBufferSlots] = std::move(entry);
            }
            selected.push_back(index);
            keep({0, std::move(buffer), 0, override->size});
            continue;
        }
        std::uint64_t address = item.address;
        std::size_t size = item.size;
        if (override != moved.end()) {
            address = override->address;
            size = override->size;
        } else {
            if (movedOnly || !item.guest || item.written || guestMemory.WritesOverlap(item.address, item.size)) continue;
            const bool direct = std::any_of(reads.begin(), reads.end(), [&](const auto& range) { return item.address >= range.first && item.address < range.second && item.size <= range.second - item.address; });
            if (!direct || recorder.PendingWriteOverlaps(item.address, item.size)) continue;
        }
        const auto begin = address - item.adjustment;
        const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(size, item.adjustment));
        if (override == moved.end() && (bytes > maxSnapshotBytes || totalSnapshotBytes + bytes > maxSnapshotBytes)) continue;
        const auto registryGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        if (!stamped) {
            stamp = {GuestMemory::TrackerGeneration(), GuestMemory::CollectEpoch(), registryGeneration};
            stamped = true;
        }
        VkDeviceSize offset = 0;
        auto buffer = recorder.UnwrittenDrawSnapshot(begin, bytes, stamp, &offset);
        static const bool verifySnapshots = std::getenv("APS5_VERIFY_DRAW_SNAPSHOTS") != nullptr;
        if (verifySnapshots && buffer != nullptr) {
            static std::atomic<std::uint64_t> checked{0}, differing{0};
            const auto fast = buffer->Bytes().subspan(static_cast<std::size_t>(offset), bytes);
            const auto slow = recorder.ReusableDrawSnapshot(begin, bytes, Recorder::SnapshotUse::Storage, nullptr, GuestMemory::CollectWrites(begin, bytes));
            const bool same = slow == buffer || (slow != nullptr && slow->Bytes().size() == fast.size() && std::memcmp(slow->Bytes().data(), fast.data(), fast.size()) == 0) || (slow == nullptr && std::memcmp(fast.data(), reinterpret_cast<const void*>(begin), bytes) == 0);
            if (!same) differing.fetch_add(1);
            if (checked.fetch_add(1) % 200000 == 199999) std::fprintf(stderr, "[draw-snapshots] verify: %llu fast answers checked, %llu differing%c", static_cast<unsigned long long>(checked.load()), static_cast<unsigned long long>(differing.load()), 10);
        }
        bool gpuSide = false;
        if (buffer == nullptr && override == moved.end() && gpuSourced) {
            auto miss = GuestBufferMemory::GpuSourceMiss::NoSource;
            if (auto source = GuestBufferMemory::FindGpuSource(begin, bytes, std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4), miss)) {
                if (GuestBufferMemory::VerifyGpuSourceTurn()) verifyGpuSource(context, recorder, *source, begin, bytes);
                buffer = std::move(source->buffer);
                offset = source->offset;
                gpuSide = true;
            }
            GuestBufferMemory::CountDrawBuffer(gpuSide, bytes, miss);
            if (GuestMemory::TrackerGeneration() != stamp[0]) stamped = false;
        }
        if (buffer == nullptr) {
            const bool uploaded = bytes != 0 && bytes <= uploadLimit;
            if (uploaded) {
                auto slice = recorder.DrawUpload(bytes);
                std::memcpy(slice.bytes, reinterpret_cast<const void*>(begin), bytes);
                buffer = std::move(slice.buffer);
                offset = slice.offset;
                countDrawSnapshot(DrawSnapshotPath::Upload, bytes);
            } else {
                const auto generation = GuestMemory::CollectWrites(begin, bytes);
                const bool owned = generation == 0 && ownedSnapshots;
                bool arm = false;
                auto ownedMiss = Recorder::OwnedSnapshotMiss::None;
                buffer = owned ? recorder.OwnedDrawSnapshot(begin, bytes, arm, ownedMiss) : recorder.ReusableDrawSnapshot(begin, bytes, Recorder::SnapshotUse::Storage, nullptr, generation);
                if (buffer != nullptr) countDrawSnapshot(owned ? DrawSnapshotPath::OwnedHit : DrawSnapshotPath::TrackedHit, bytes);
                if (buffer == nullptr) {
                    std::vector<std::uint64_t> words;
                    bool armed = false;
                    std::uint64_t since = 0;
                    std::uint64_t forgetSerial = 0;
                    if (owned && arm) {
                        words.resize(GuestMemory::HashBlockCount(begin, bytes));
                        armed = GuestMemory::ArmOwnedBlocks(begin, bytes, words, !RuntimeSwitches::Set(RuntimeSwitch::NoOwnedPrivate));
                        if (armed) {
                            forgetSerial = GuestMemory::ForgetSerial();
                            since = GuestMemory::TrackerGeneration();
                        }
                    }
                    const MemoryStats::OwnerScope memStatsOwner(MemoryStats::Owner::ShaderResources);
                    buffer = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                    std::memcpy(buffer->Bytes().data(), reinterpret_cast<const void*>(begin), bytes);
                    if (!owned) recorder.KeepDrawSnapshot(begin, bytes, generation, registryGeneration, buffer);
                    else if (armed) recorder.KeepOwnedDrawSnapshot(begin, bytes, std::move(words), since, registryGeneration, forgetSerial, buffer);
                    else if (arm) recorder.RefuseOwnedDrawSnapshot(begin, bytes);
                    countDrawSnapshot(generation == 0 ? DrawSnapshotPath::UntrackedCopy : DrawSnapshotPath::TrackedCopy, bytes);
                    if (owned) countOwnedSnapshotMiss(ownedMiss, armed, arm && !armed);
                }
            }
            if (GuestMemory::TrackerGeneration() != stamp[0]) stamped = false;
            else if (!uploaded || stamp[1] != 0) recorder.NoteUnwrittenDrawSnapshot(begin, bytes, stamp, buffer, offset);
        }
        selected.push_back(index);
        keep({begin, std::move(buffer), offset, bytes});
        totalSnapshotBytes += bytes;
        CaptureTrace::Log("draw-snapshot batch=%llu address=%llx bytes=%zu", static_cast<unsigned long long>(recorder.Submissions() + 1), static_cast<unsigned long long>(begin), bytes);
    }
    if (selected.empty()) return {};
    Require(context.descriptorCache != nullptr, "draw snapshots require a descriptor cache");
    std::array<bool, 2> described{};
    thread_local std::array<std::vector<VkDescriptorPoolSize>, 2>* countsSlot = nullptr;
    auto& counts = ShaderRecompiler::ThreadOwned(countsSlot);
    for (auto& sizes : counts) sizes.clear();
    for (const auto& binding : bindings) {
        const auto set = setOf(binding);
        described[set] = true;
        const auto count = (binding.flags & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) != 0 ? std::max(1u, static_cast<std::uint32_t>(binding.imageAllocations.size())) : binding.layout.descriptorCount;
        if (count == 0) continue;
        auto& sizes = counts[set];
        const auto held = std::find_if(sizes.begin(), sizes.end(), [&](const VkDescriptorPoolSize& entry) { return entry.type == binding.layout.descriptorType; });
        if (held != sizes.end()) held->descriptorCount += count;
        else sizes.push_back({binding.layout.descriptorType, count});
    }
    for (auto& sizes : counts) std::sort(sizes.begin(), sizes.end(), [](const VkDescriptorPoolSize& a, const VkDescriptorPoolSize& b) { return a.type < b.type; });
    const auto sizesOf = [&](std::size_t set) { return std::span<const VkDescriptorPoolSize>(counts[set]); };
    const auto variableOf = [&](std::size_t set) { return tableBindings[set] != std::numeric_limits<std::size_t>::max() ? std::optional<std::uint32_t>(std::max(1u, static_cast<std::uint32_t>(bindings[tableBindings[set]].imageAllocations.size()))) : std::nullopt; };
    result->cache = context.descriptorCache;
    if (described[0] || splitSets) {
        result->allocation = result->cache->AllocateRewritten(_layout, sizesOf(0), updateAfterBind, variableOf(0));
        result->layout = _layout;
        Require(result->allocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
    }
    if (described[1] || splitSets) {
        result->fragmentAllocation = result->cache->AllocateRewritten(fragment.layout, sizesOf(1), updateAfterBind, variableOf(1));
        result->fragmentLayout = fragment.layout;
        Require(result->fragmentAllocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
    }
    thread_local std::vector<VkDescriptorBufferInfo>* buffersSlot = nullptr;
    auto& buffers = ShaderRecompiler::ThreadOwned(buffersSlot);
    buffers = descriptorBuffers;
    thread_local std::vector<VkWriteDescriptorSet>* writesSlot = nullptr;
    auto& writes = ShaderRecompiler::ThreadOwned(writesSlot);
    writes = descriptorWrites;
    thread_local std::vector<std::size_t>* indicesSlot = nullptr;
    auto& snapshotIndices = ShaderRecompiler::ThreadOwned(indicesSlot);
    snapshotIndices.assign(allocations.size(), selected.size());
    for (std::size_t index = 0; index < selected.size(); ++index) snapshotIndices[selected[index]] = index;
    auto nextWrite = writes.begin();
    for (const auto& binding : bindings) {
        const auto original = setOf(binding) == 0 ? _set : fragment.set;
        while (nextWrite != writes.end() && nextWrite->dstBinding == binding.layout.binding && nextWrite->dstSet == original) {
            auto& write = *nextWrite++;
            write.dstSet = setOf(binding) == 0 ? result->allocation.set : result->fragmentAllocation.set;
            if (binding.allocations.empty()) continue;
            const auto offset = static_cast<std::size_t>(write.pBufferInfo - descriptorBuffers.data());
            write.pBufferInfo = buffers.data() + offset;
            for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
                const auto snapshotIndex = snapshotIndices[binding.allocations[element]];
                if (snapshotIndex == selected.size()) continue;
                const auto& snapshot = result->snapshots[snapshotIndex];
                buffers[offset + element] = {snapshot.buffer->Handle(), snapshot.offset, snapshot.bytes};
            }
        }
    }
    Require(nextWrite == writes.end(), "draw snapshot writes disagree with their descriptor bindings");
    context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    recorder.Keep(result);
    for (const auto& snapshot : result->snapshots) {
        if (snapshot.buffer->Mapped()) recorder.KeepBytes(snapshot.buffer.get(), snapshot.buffer->Bytes().size());
    }
    if (DrawSnapshotsProfiled()) reportDrawSnapshots(recorder);
    return result;
}

void ShaderResources::Bind(VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout layout) const {
    if (_set != VK_NULL_HANDLE) context.Resolved(&DeviceFunctions::cmdBindDescriptorSets, "vkCmdBindDescriptorSets")(commands, bindPoint, layout, 0, 1, &_set, 0, nullptr);
    if (fragment.set != VK_NULL_HANDLE) context.Resolved(&DeviceFunctions::cmdBindDescriptorSets, "vkCmdBindDescriptorSets")(commands, bindPoint, layout, ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet, 1, &fragment.set, 0, nullptr);
}

namespace {
// Debug aid: APS5_GPU_NO_WRITEBACK=1 keeps GPU results out of guest memory.
bool SkipWriteBack() {
    static const bool skip = std::getenv("APS5_GPU_NO_WRITEBACK") != nullptr;
    return skip;
}
}

void ShaderResources::WriteBack() {
    std::exception_ptr unaligned;
    try {
        WriteBackBuffers();
    } catch (const BdaUnalignedAccess&) {
        unaligned = std::current_exception();
    }
    if (!guestMemory.Writes().empty() || std::find(storageWritten.begin(), storageWritten.end(), true) != storageWritten.end() || (usesBda && !bdaReadOnly)) GuestMemory::BreakProofs();
    if (!SkipWriteBack()) {
        for (std::size_t index = 0; index < storageTextures.size(); ++index) {
            if (!storageWritten[index] || storageTextures[index] == nullptr) continue;
            storageTextures[index]->MarkDirty();
            if (!DepthSurfaceStoreBack(context, *storageTextures[index])) MarkStorageDepthWrites(context, storageTextures[index]);
        }
    }
    if (unaligned) std::rethrow_exception(unaligned);
}

void ShaderResources::MarkGpuWritesInRun(Recorder& recorder, const std::shared_ptr<ShaderResources>& self) {
    if (!recorder.RunOpen() || !guestMemory.HasGpuCopies()) {
        MarkGpuWrites(recorder);
        return;
    }
    // As MarkGpuWrites, with the copy-backs (and the marks of the ranges they store) after the run.
    const auto kind = !guestMemory.HoldsLease() ? Recorder::ReadKind::DispatchElement : usesBda && bdaReadOnly && !RuntimeSwitches::Set(RuntimeSwitch::NoReadOnlyLeaseSkip) ? Recorder::ReadKind::AddressBasedReadOnly : Recorder::ReadKind::AddressBased;
    if (auto split = guestMemory.SplitInPlaceReads()) {
        recorder.NotePendingReadList(std::move(split->space), kind);
        recorder.NotePendingReads(split->own, kind);
    } else {
        recorder.NotePendingReads(guestMemory.InPlaceReads(), kind);
    }
    if (SkipWriteBack()) return;
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (!storageWritten[index] || storageTextures[index] == nullptr) continue;
        storageTextures[index]->MarkDirty();
        if (!DepthSurfaceStoreBack(context, *storageTextures[index])) MarkStorageDepthWrites(context, storageTextures[index]);
    }
    // A storage store-back above may have recorded a command, which closed the run: then as MarkGpuWrites.
    if (!recorder.RunOpen()) guestMemory.RecordCopyBacks(recorder);
    recorder.NotePendingWrites(guestMemory.Writes());
    guestMemory.MarkDirectWrites();
    if (!recorder.RunOpen()) return;
    recorder.AfterRun([self](Recorder& after) {
        self->guestMemory.RecordCopyBacks(after);
        self->guestMemory.MarkDirectWrites();
    });
}

void ShaderResources::MarkGpuWrites(Recorder& recorder) {
    // The ranges this use reads in place through their host imports (read-only and written elements
    // alike, and an address-based build's whole leased heaps), before the writes: a CPU store into
    // one of them (the copy HLE) must not land before the recorded work read it.
    const auto kind = !guestMemory.HoldsLease() ? Recorder::ReadKind::DispatchElement : usesBda && bdaReadOnly && !RuntimeSwitches::Set(RuntimeSwitch::NoReadOnlyLeaseSkip) ? Recorder::ReadKind::AddressBasedReadOnly : Recorder::ReadKind::AddressBased;
    if (auto split = guestMemory.SplitInPlaceReads()) {
        recorder.NotePendingReadList(std::move(split->space), kind);
        recorder.NotePendingReads(split->own, kind);
    } else {
        recorder.NotePendingReads(guestMemory.InPlaceReads(), kind);
    }
    if (SkipWriteBack()) return;
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (!storageWritten[index] || storageTextures[index] == nullptr) continue;
        storageTextures[index]->MarkDirty();
        if (!DepthSurfaceStoreBack(context, *storageTextures[index])) MarkStorageDepthWrites(context, storageTextures[index]);
    }
    // Written sub-ranges of buffers the GPU copied out of a host import go back into it by the GPU,
    // recorded here after the work: those regions then need no CPU write-back (HasCopiedWrites),
    // and the note and mark below cover them like direct writes.
    guestMemory.RecordCopyBacks(recorder);
    // Only the written elements' ranges (AddWritable): a read-only element is neither noted here
    // nor marked as a direct write, so CPU reads of its memory never wait for this work.
    recorder.NotePendingWrites(guestMemory.Writes());
    guestMemory.MarkDirectWrites();
    if (!BuildProfiled()) return;
    auto& counters = BufferWrites();
    // Recorded uses only (dispatches and recorded draws): a synchronous draw writes back in
    // WriteBack() and publishes no pending writes, so it has no notes to skip.
    counters.notesSkipped.fetch_add(readOnlyBuffers, std::memory_order_relaxed);
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load();
    if (nowMs - last < 10000 || !counters.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto elements = counters.elements.load();
    const auto readOnly = counters.readOnly.load();
    AgcDriver::ProfilePrint_nid_no_patch("[buffers] descriptor elements bound (cumulative): %llu total, %llu written, %llu read-only; %llu pending-write notes skipped\n", static_cast<unsigned long long>(elements), static_cast<unsigned long long>(elements - readOnly), static_cast<unsigned long long>(readOnly), static_cast<unsigned long long>(counters.notesSkipped.load()));
}

void ShaderResources::CheckFaults(std::uint64_t shaderAddress) const {
    if (bda) bda->CheckFault(shaderAddress);
}

std::shared_ptr<const void> ShaderResources::BeginUse() {
    if (!addressReuse) return {};
    struct Use {
        Use(std::shared_ptr<std::atomic<std::uint32_t>> counter, std::shared_ptr<const void> hold, bool reusedSpace) : uses(std::move(counter)), space(std::move(hold)), reused(reusedSpace) {
            uses->fetch_add(1, std::memory_order_acq_rel);
            if (reused) HoldReusedSpace(true);
        }
        ~Use() {
            if (reused) HoldReusedSpace(false);
            uses->fetch_sub(1, std::memory_order_acq_rel);
        }
        std::shared_ptr<std::atomic<std::uint32_t>> uses;
        std::shared_ptr<const void> space;
        bool reused;
    };
    return std::make_shared<const Use>(usesInFlight, guestMemory.SpaceHold(), std::exchange(reuseProved, false));
}

void ShaderResources::CompleteUse() {
    if (!addressReuse) {
        CheckFaults();
        return;
    }
    try {
        CheckFaults();
    } catch (...) {
        reusable = false;
        throw;
    }
    if (SkipWriteBack()) return;
    for (const auto& [begin, end] : addressWrites) GuestMemory::MarkWritten(begin, static_cast<std::size_t>(end - begin));
}

void ShaderResources::EndUse() noexcept {
    if (!addressReuse) return;
    reuseProved = false;
    guestMemory.SuspendSpace();
}

void ShaderResources::WriteBackBuffers(std::uint64_t shaderAddress) {
    std::exception_ptr unaligned;
    if (bda) {
        try {
            bda->CheckFault(shaderAddress);
        } catch (const BdaUnalignedAccess&) {
            unaligned = std::current_exception();
        }
    }
    if (!SkipWriteBack()) guestMemory.WriteBack();
    if (unaligned) std::rethrow_exception(unaligned);
}

bool ShaderResources::ReadsOverlap(std::uint64_t address, std::size_t bytes) const {
    const auto reads = guestMemory.InPlaceReads();
    return std::any_of(reads.begin(), reads.end(), [&](const auto& range) { return address < range.second && range.first < address + bytes; });
}

void ShaderResources::RunAccesses(std::vector<std::array<std::uint64_t, 3>>& out) const {
    guestMemory.RunAccesses(out);
    if (usesBda) out.push_back({0, std::numeric_limits<std::uint64_t>::max(), Recorder::RunAccessWritten});
    for (const auto& [image, written] : StorageImages()) {
        const auto handle = reinterpret_cast<std::uint64_t>(image);
        out.push_back({handle, handle + 1, Recorder::RunAccessImage | (written ? Recorder::RunAccessWritten : 0u)});
    }
}

std::vector<std::pair<VkImage, bool>> ShaderResources::StorageImages() const {
    std::vector<std::pair<VkImage, bool>> images;
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (storageTextures[index] != nullptr) images.emplace_back(storageTextures[index]->Image(), storageWritten[index]);
    }
    for (const auto& texture : textures) {
        if (texture != nullptr && texture->StorageSource() != nullptr) images.emplace_back(texture->StorageSource()->Image(), false);
    }
    return images;
}

bool ShaderResources::HasCachedStorageImages() const {
    if (std::any_of(storageTextures.begin(), storageTextures.end(), [](const auto& image) { return image != nullptr && !image->Cached(); })) return false;
    return std::none_of(textures.begin(), textures.end(), [](const auto& texture) { return texture != nullptr && texture->StorageSource() != nullptr && !texture->StorageSource()->Cached(); });
}

bool ShaderResources::ReadsImage(const StorageTexture* image) const {
    if (image == nullptr) return false;
    for (const auto& texture : textures) {
        if (texture != nullptr && texture->StorageSource() == image) return true;
    }
    return std::any_of(storageTextures.begin(), storageTextures.end(), [&](const auto& storage) { return storage.get() == image; });
}

bool ShaderResources::BindsReleasedTexture() const {
    return std::any_of(textures.begin(), textures.end(), [](const auto& texture) { return texture != nullptr && texture->SampledCacheState() == Texture::CacheState::Released; });
}

void ShaderResources::PinTextures(bool pin) const {
    for (const auto& texture : textures) {
        if (texture == nullptr) continue;
        if (pin) texture->Pin();
        else texture->Unpin();
    }
}

}
