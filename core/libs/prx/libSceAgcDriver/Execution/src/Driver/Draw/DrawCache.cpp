#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include <algorithm>
#include <bit>
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/WriteEvidence.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <cstdlib>
#include <utility>

namespace AgcDriver::DriverDetail {

bool DrawRecipeRecord::Matches(const std::vector<std::shared_ptr<DispatchVariant>>& variants) const {
    if (stages.size() != variants.size()) return false;
    for (std::size_t i = 0; i < stages.size(); ++i) {
        if (stages[i].owner_before(variants[i]) || variants[i].owner_before(stages[i])) return false;
    }
    return true;
}

const Graphics::AheadDraw* PreparedDrawRecord::For(const std::vector<std::shared_ptr<DispatchVariant>>& variants, std::size_t shaderCount) const {
    if (shaderCount != shaders || variants.size() != stages.size()) return nullptr;
    for (std::size_t i = 0; i < stages.size(); ++i) {
        if (stages[i].owner_before(variants[i]) || variants[i].owner_before(stages[i])) return nullptr;
    }
    return prepared.get();
}

bool DrawRecipeRecord::Expired() const {
    return std::any_of(stages.begin(), stages.end(), [](const std::weak_ptr<const DispatchVariant>& stage) { return stage.expired(); });
}

bool Driver::drawEntries() {
    static const bool entries = std::getenv("APS5_NO_DRAW_SRT_ENTRIES") == nullptr && !stampValidate();
    return entries;
}

bool Driver::drawDataHits() {
    static const bool hits = std::getenv("APS5_NO_DRAW_DATA_HITS") == nullptr;
    return hits;
}

bool Driver::drawStageDataHits() {
    static const bool hits = std::getenv("APS5_NO_DRAW_STAGE_DATA_HITS") == nullptr && !stampValidate();
    return hits;
}

bool Driver::verifyDrawDataHits() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_DATA_HITS") != nullptr;
    return verify;
}

bool Driver::verifyDrawEntries() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_ENTRIES") != nullptr;
    return verify;
}

bool Driver::registerKeyEnabled() {
    static const bool registerKey = std::getenv("APS5_NO_DRAW_KEY") == nullptr;
    return registerKey;
}

bool Driver::verifyDrawRecipe() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_RECIPE") != nullptr;
    return verify;
}

std::size_t Driver::drawCacheEntries() {
    static const std::size_t entries = [] {
        const char* text = std::getenv("APS5_DRAW_CACHE_ENTRIES");
        const auto parsed = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
        return parsed != 0 ? static_cast<std::size_t>(parsed) : std::size_t{4096};
    }();
    return entries;
}

std::uint64_t Driver::DrawCacheBytes() {
    static const std::uint64_t bytes = [] {
        const char* text = std::getenv("APS5_DRAW_CACHE_BYTES");
        const auto parsed = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
        return parsed != 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{256} << 20u;
    }();
    return bytes;
}

void Driver::accountDrawVariant(const DispatchVariant& variant, bool added) {
    if (added) {
        ++drawCacheVariants;
        drawCacheVariantBytes += variantBytes(variant);
    } else {
        --drawCacheVariants;
        drawCacheVariantBytes -= variantBytes(variant);
    }
}

DrawKeySightings::DrawKeySightings(std::size_t keyCount) {
    Graphics::Require(keyCount >= Ways, "draw key sightings need at least one bucket");
    const auto buckets = std::bit_ceil(keyCount / Ways);
    keys.assign(buckets * Ways, 0);
    filled.assign(buckets, 0);
    next.assign(buckets, 0);
    bucketMask = buckets - 1;
}

bool DrawKeySightings::SeenBefore(std::uint64_t key) {
    auto mixed = key;
    mixed = (mixed ^ (mixed >> 30u)) * 0xbf58476d1ce4e5b9ull;
    mixed = (mixed ^ (mixed >> 27u)) * 0x94d049bb133111ebull;
    mixed ^= mixed >> 31u;
    const auto bucket = static_cast<std::size_t>(mixed) & bucketMask;
    auto* ways = keys.data() + bucket * Ways;
    if (std::find(ways, ways + filled[bucket], key) != ways + filled[bucket]) return true;
    ways[next[bucket]] = key;
    next[bucket] = static_cast<std::uint8_t>((next[bucket] + 1u) % Ways);
    if (filled[bucket] < Ways) ++filled[bucket];
    return false;
}

bool Driver::admitDrawKey(std::uint64_t key, std::uint64_t shape) {
    std::lock_guard cacheLock(drawCacheMutex);
    if (drawCache.contains(key)) return true;
    if (!drawKeySightings) drawKeySightings.emplace(drawCacheEntries() * DrawKeySightings::Ways);
    bool seen = drawKeySightings->SeenBefore(key);
    const auto& pending = drawRelocationPending;
    if (pending.key == key && pending.identity != 0) {
        if (pending.alias != 0) {
            const auto trial = drawRelocationTrials.find(pending.identity);
            if (drawKeySightings->SeenBefore(pending.alias ^ 0x9e3779b97f4a7c15ull) && trial != drawRelocationTrials.end()) {
                auto& state = trial->second;
                if (state.productive || state.hits != 0 || state.admissions < DrawRelocationTrial::ProbationAdmissions) {
                    if (!state.productive && state.hits == 0) ++state.admissions;
                    seen = true;
                }
            }
        } else if (!drawRelocations.contains(pending.identity)) {
            if (drawKeySightings->SeenBefore(pending.identity ^ 0xc2b2ae3d27d4eb4full)) seen = true;
        }
    }
    if (seen || shape == 0 || drawShapes.contains(shape)) return seen;
    if (!drawShapeSightings) drawShapeSightings.emplace(drawCacheEntries() * DrawKeySightings::Ways);
    return drawShapeSightings->SeenBefore(shape);
}

std::shared_ptr<DrawEntry> BuildAheadEntry(const std::vector<std::shared_ptr<DispatchVariant>>& fresh, std::shared_ptr<const DrawDecode> decode, std::uint64_t shape, std::shared_ptr<const Graphics::AheadDraw> prepared, std::size_t preparedShaders) {
    auto entry = std::make_shared<DrawEntry>();
    entry->stages.resize(fresh.size());
    entry->decode = std::move(decode);
    entry->shape = shape;
    entry->probation = true;
    for (std::size_t i = 0; i < fresh.size(); ++i) {
        if (fresh[i] != nullptr) entry->stages[i].push_back(fresh[i]);
    }
    if (prepared != nullptr) entry->prepared = std::make_shared<const PreparedDrawRecord>(PreparedDrawRecord{{fresh.begin(), fresh.end()}, preparedShaders, std::move(prepared)});
    return entry;
}

bool Driver::insertAheadEntry(std::uint64_t key, const std::shared_ptr<DrawEntry>& entry, std::vector<std::shared_ptr<DrawEntry>>& released) {
    std::list<std::uint64_t> node{key};
    std::lock_guard aheadLock(drawAheadMutex);
    if (drawAheadEntries.contains(key)) return false;
    const auto limit = drawCacheEntries();
    released.reserve(drawAheadOrder.size() + 1 > limit ? drawAheadOrder.size() + 1 - limit : 0);
    drawAheadEntries.emplace(key, entry);
    entry->order = node.begin();
    drawAheadOrder.splice(drawAheadOrder.begin(), node);
    while (drawAheadOrder.size() > limit) {
        const auto last = drawAheadEntries.find(drawAheadOrder.back());
        released.push_back(std::move(last->second));
        drawAheadEntries.erase(last);
        drawAheadOrder.pop_back();
    }
    return true;
}

std::shared_ptr<DrawEntry> Driver::findAheadEntry(std::uint64_t key) {
    std::lock_guard aheadLock(drawAheadMutex);
    const auto found = drawAheadEntries.find(key);
    return found == drawAheadEntries.end() ? nullptr : found->second;
}

std::shared_ptr<DrawEntry> Driver::takeAheadEntry(std::uint64_t key, const DrawEntry* expected) {
    std::lock_guard aheadLock(drawAheadMutex);
    const auto found = drawAheadEntries.find(key);
    if (found == drawAheadEntries.end() || (expected != nullptr && found->second.get() != expected)) return nullptr;
    auto taken = std::move(found->second);
    drawAheadOrder.erase(taken->order);
    drawAheadEntries.erase(found);
    return taken;
}

void Driver::admitAheadEntry(std::uint64_t key, const DrawEntry& ahead, std::uint64_t shape) {
    std::vector<std::shared_ptr<DrawEntry>> released;
    std::shared_ptr<DrawEntry> taken;
    auto admitted = std::make_shared<DrawEntry>();
    admitted->decode = ahead.decode;
    admitted->prepared = ahead.prepared;
    admitted->stages = ahead.stages;
    admitted->shape = shape;
    admitted->recipes.store(ahead.recipes.load());
    std::list<std::uint64_t> node{key};
    std::lock_guard cacheLock(drawCacheMutex);
    if (drawCache.contains(key)) {
        taken = takeAheadEntry(key, &ahead);
        return;
    }
    prepareEvictions(1, 0);
    const auto inserted = drawCache.emplace(key, admitted).first;
    if (admitted->decode != nullptr && shape != 0) {
        try {
            drawShapes[shape] = key;
        } catch (...) {
            drawCache.erase(inserted);
            throw;
        }
    }
    admitted->touched = drawCacheHits;
    admitted->order = node.begin();
    drawOrder.splice(drawOrder.begin(), node);
    std::uint64_t variants = 0;
    for (const auto& stage : admitted->stages) {
        for (const auto& variant : stage) {
            accountDrawVariant(*variant, true);
            ++variants;
        }
    }
    ++drawEntryCounters.inserts;
    drawEntryCounters.variantsInserted += variants;
    taken = takeAheadEntry(key, &ahead);
    evictDrawEntries();
    released.swap(releasedDrawEntries);
}

void Driver::insertDrawEntry(std::uint64_t key, std::vector<std::shared_ptr<DispatchVariant>>& fresh, std::shared_ptr<const DrawDecode> decode, std::uint64_t shape, bool replaceShape) {
    std::vector<std::shared_ptr<DrawEntry>> released;
    std::shared_ptr<DrawEntry> taken;
    std::lock_guard cacheLock(drawCacheMutex);
    const bool shapeUpdate = replaceShape && shape != 0;
    if (shapeUpdate) {
        if (const auto shapeOf = drawShapes.find(shape); shapeOf != drawShapes.end() && shapeOf->second != key) {
            if (const auto previous = drawCache.find(shapeOf->second); previous != drawCache.end()) {
                for (const auto& variants : previous->second->stages) {
                    for (const auto& variant : variants) accountDrawVariant(*variant, false);
                }
                drawOrder.erase(previous->second->order);
                releasedDrawEntries.push_back(std::move(previous->second));
                drawCache.erase(previous);
            }
            drawShapes.erase(shapeOf);
        }
    }
    const auto found = drawCache.find(key);
    auto replacement = std::make_shared<DrawEntry>();
    replacement->stages.resize(fresh.size());
    replacement->decode = std::move(decode);
    replacement->shape = shape;
    if (found != drawCache.end()) {
        if (found->second->stages.size() == fresh.size()) replacement->stages = found->second->stages;
        if (found->second->decode != nullptr) replacement->decode = found->second->decode;
        replacement->prepared = found->second->prepared;
        replacement->recipes.store(found->second->recipes.load());
    }
    auto resolved = fresh;
    std::vector<std::shared_ptr<DispatchVariant>> added;
    std::vector<std::shared_ptr<DispatchVariant>> dropped;
    std::uint64_t present = 0;
    for (std::size_t i = 0; i < resolved.size(); ++i) {
        if (resolved[i] == nullptr) continue;
        auto& variants = replacement->stages[i];
        const auto kept = std::find_if(variants.begin(), variants.end(), [&](const std::shared_ptr<DispatchVariant>& variant) { return variant->pushOffset == resolved[i]->pushOffset && variant->meshIndexDescriptor == resolved[i]->meshIndexDescriptor && variant->runs == resolved[i]->runs && SameCapturedWords(variant->words, variant->shared, resolved[i]->words, resolved[i]->shared); });
        if (kept != variants.end()) {
            ++present;
            resolved[i] = *kept;
            continue;
        }
        added.push_back(resolved[i]);
        variants.insert(variants.begin(), resolved[i]);
        while (variants.size() > dispatchVariants()) {
            dropped.push_back(std::move(variants.back()));
            variants.pop_back();
        }
    }
    replacement->touched = drawCacheHits;
    const bool ownsShape = replacement->decode != nullptr && shape != 0;
    std::list<std::uint64_t> node;
    if (found == drawCache.end()) node.push_front(key);
    prepareEvictions(found == drawCache.end() ? 1 : 0, found == drawCache.end() ? 0 : 1);
    if (found == drawCache.end()) {
        const auto inserted = drawCache.emplace(key, replacement).first;
        if (ownsShape) {
            try {
                drawShapes[shape] = key;
            } catch (...) {
                drawCache.erase(inserted);
                throw;
            }
        }
        replacement->order = node.begin();
        drawOrder.splice(drawOrder.begin(), node);
    } else {
        if (ownsShape) drawShapes[shape] = key;
        replacement->order = found->second->order;
        drawOrder.splice(drawOrder.begin(), drawOrder, replacement->order);
        releasedDrawEntries.push_back(std::exchange(found->second, std::move(replacement)));
    }
    for (const auto& variant : added) accountDrawVariant(*variant, true);
    for (const auto& variant : dropped) accountDrawVariant(*variant, false);
    auto& counters = drawEntryCounters;
    if (shapeUpdate) ++counters.shapeUpdates;
    else ++counters.inserts;
    counters.present += present;
    counters.variantsInserted += added.size();
    counters.variantsEvicted += dropped.size();
    fresh.swap(resolved);
    taken = takeAheadEntry(key, nullptr);
    evictDrawEntries();
    released.swap(releasedDrawEntries);
}

void Driver::prepareEvictions(std::size_t growth, std::size_t replaced) {
    const auto limit = drawCacheEntries();
    const auto victims = drawOrder.size() + growth > limit ? drawOrder.size() + growth - limit : 0;
    releasedDrawEntries.reserve(releasedDrawEntries.size() + replaced + victims);
    if (!traceDrawCache()) return;
    auto victim = drawOrder.rbegin();
    for (std::size_t i = 0; i < victims && victim != drawOrder.rend(); ++i, ++victim) traceInsert(traceKeysEvicted, *victim);
}

void Driver::evictDrawEntries() {
    while (drawOrder.size() > drawCacheEntries() || (drawOrder.size() > 1 && drawCacheVariantBytes > DrawCacheBytes())) {
        const bool overBytes = drawOrder.size() <= drawCacheEntries();
        const auto last = drawCache.find(drawOrder.back());
        for (const auto& variants : last->second->stages) {
            for (const auto& variant : variants) accountDrawVariant(*variant, false);
        }
        if (const auto shapeOf = drawShapes.find(last->second->shape); shapeOf != drawShapes.end() && shapeOf->second == last->first) drawShapes.erase(shapeOf);
        drawOrder.erase(last->second->order);
        releasedDrawEntries.push_back(std::move(last->second));
        drawCache.erase(last);
        ++drawCacheEvictions;
        ++drawEntryCounters.evictions;
        if (overBytes) {
            ++drawCacheByteEvictions;
            ++drawEntryCounters.byteEvictions;
        }
    }
}

void Driver::eraseDrawEntry(std::uint64_t key, const std::shared_ptr<DrawEntry>& expected) {
    if (expected != nullptr && expected->probation) {
        takeAheadEntry(key, expected.get());
        return;
    }
    const auto found = drawCache.find(key);
    if (found == drawCache.end() || found->second != expected) return;
    for (const auto& variants : found->second->stages) {
        for (const auto& variant : variants) accountDrawVariant(*variant, false);
    }
    if (const auto shapeOf = drawShapes.find(found->second->shape); shapeOf != drawShapes.end() && shapeOf->second == key) drawShapes.erase(shapeOf);
    drawOrder.erase(found->second->order);
    drawCache.erase(found);
}

bool DecodeProgramsCurrent(const DrawDecode& decode, const ShaderRegistry&) {
    return std::all_of(decode.programs.begin(), decode.programs.end(), [&](const DrawProgram& program) {
        if (!program.snapshot->header.empty()) return true;
        return ReadRawShader(program.snapshot->codeAddress + program.codeOffset * sizeof(std::uint32_t)) == program.snapshot &&
            (program.tailContinuation == nullptr || ReadRawShader(program.tailContinuation->codeAddress) == program.tailContinuation);
    });
}

std::shared_ptr<const DrawRecipe> Driver::findDrawRecipe(std::uint64_t key, const std::vector<std::shared_ptr<DispatchVariant>>& stages) {
    std::shared_ptr<const std::vector<DrawRecipeRecord>> records;
    {
        std::lock_guard cacheLock(drawCacheMutex);
        const auto found = drawCache.find(key);
        if (found == drawCache.end()) return nullptr;
        records = found->second->recipes.load();
    }
    if (records == nullptr) return nullptr;
    for (const auto& record : *records) {
        if (record.Matches(stages)) return record.recipe;
    }
    return nullptr;
}

void Driver::attachDrawRecipe(std::uint64_t key, const std::vector<std::shared_ptr<DispatchVariant>>& stages, std::shared_ptr<const DrawRecipe> recipe) {
    std::shared_ptr<DrawEntry> entry;
    {
        std::lock_guard cacheLock(drawCacheMutex);
        const auto found = drawCache.find(key);
        if (found == drawCache.end()) return;
        entry = found->second;
    }
    const auto old = entry->recipes.load();
    auto records = std::make_shared<std::vector<DrawRecipeRecord>>();
    records->push_back({std::vector<std::weak_ptr<const DispatchVariant>>(stages.begin(), stages.end()), std::move(recipe)});
    if (old != nullptr) {
        for (const auto& record : *old) {
            if (record.Matches(stages) || record.Expired() || records->size() >= dispatchVariants()) continue;
            records->push_back(record);
        }
    }
    entry->recipes.store(std::move(records));
    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, VulkanDevice::RecipeKind::Draw);
}

}
