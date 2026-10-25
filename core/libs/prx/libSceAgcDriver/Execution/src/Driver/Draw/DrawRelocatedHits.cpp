#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::size_t PoolTemplates = 64;
constexpr std::size_t MaxTemplates = 16384;
constexpr std::size_t MaxPlans = 4;
constexpr std::size_t MaxValidations = 2;

std::uint64_t Mix(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    return hash;
}

std::uint64_t Aligned(std::uint32_t word) {
    return word & ~std::uint32_t{3};
}

std::uint64_t PoolKey(std::uint64_t shape, std::size_t stage) {
    return Mix(Mix(shape, 0x72656c6f63617465ull), stage);
}

std::uint64_t FetchWords(const std::optional<ShaderRecompiler::ShaderVertexStageInfo>& vertexInfo) {
    if (!vertexInfo) return 0;
    std::uint64_t mask = 0;
    for (const auto reg : {static_cast<std::uint32_t>(vertexInfo->fetchAttribReg), static_cast<std::uint32_t>(vertexInfo->fetchBufferReg)}) mask |= reg + 1u < 64u ? std::uint64_t{3} << reg : ~std::uint64_t{0};
    return mask;
}

struct Tagged {
    std::uint64_t address = 0;
    std::uint32_t words = 0;
    std::int32_t decoder = -1;
};

enum class Origin : std::uint8_t { Unknown, User, Pointer, Conflict };

std::uint64_t IndexHash(std::span<const std::uint32_t> words, std::span<const std::uint32_t> hashed) {
    std::uint64_t hash = 0x2545f4914f6cdd1dull ^ hashed.size();
    for (const auto position : hashed) hash = Mix(hash, words[position]);
    return hash;
}

}

bool Driver::relocatedHitsVerified() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_DATA_HITS") != nullptr;
    return verify;
}

// APS5_VERIFY_RELOCATED_HITS=<n> (default 16 when set without a number): the first n relocated
// stage hits of every 10 s are captured and compiled again and compared with the reused stage.
std::uint64_t Driver::relocatedVerifyQuota() {
    static const std::uint64_t quota = [] {
        const char* text = std::getenv("APS5_VERIFY_RELOCATED_HITS");
        if (text == nullptr) return std::uint64_t{0};
        const auto parsed = std::strtoull(text, nullptr, 10);
        return parsed != 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{16};
    }();
    return quota;
}

bool Driver::takeRelocatedVerify() {
    if (relocatedHitsVerified()) return true;
    const auto quota = relocatedVerifyQuota();
    if (quota == 0) return false;
    std::lock_guard lock(relocatedMutex);
    const auto now = std::chrono::steady_clock::now();
    if (now - relocatedVerifyWindow >= std::chrono::seconds(10)) {
        relocatedVerifyWindow = now;
        relocatedVerifiedInWindow = 0;
    }
    if (relocatedVerifiedInWindow >= quota) return false;
    ++relocatedVerifiedInWindow;
    return true;
}

void Driver::noteRelocatedVerify(bool mismatch) {
    std::lock_guard lock(relocatedMutex);
    ++relocatedCounters.verified;
    if (mismatch) ++relocatedCounters.verifyMismatches;
}

RelocatedCounters Driver::RelocatedHitCounters() {
    std::lock_guard lock(relocatedMutex);
    return relocatedCounters;
}

std::shared_ptr<RelocatedTemplate> Driver::BuildRelocatedTemplate(const std::shared_ptr<const DispatchVariant>& variant, const DrawProgram& program, std::span<const ShaderRecompiler::MemoryRegion> regions, const ShaderRecompiler::SrtUserDataUse& use, std::span<const Graphics::DecodeRead> decodeReads, const std::vector<std::uint32_t>* exports) {
    if (variant == nullptr || variant->compiled == nullptr) return nullptr;
    const auto& words = variant->words;
    std::vector<Tagged> tagged;
    tagged.reserve(regions.size() + decodeReads.size());
    for (const auto& region : regions) tagged.push_back({region.guestAddress, static_cast<std::uint32_t>(region.bytes.size() / sizeof(std::uint32_t)), -1});
    for (std::size_t k = 0; k < decodeReads.size(); ++k) tagged.push_back({decodeReads[k].address, static_cast<std::uint32_t>(decodeReads[k].Bytes().size() / sizeof(std::uint32_t)), static_cast<std::int32_t>(k)});
    std::stable_sort(tagged.begin(), tagged.end(), [](const Tagged& a, const Tagged& b) { return a.address < b.address; });
    {
        std::size_t total = 0, run = 0;
        const auto skipEmpty = [&] {
            while (run < variant->runs.size() && variant->runs[run].second == variant->runs[run].first) ++run;
        };
        for (const auto& region : tagged) {
            skipEmpty();
            auto expected = region.address;
            std::uint32_t covered = 0;
            while (covered < region.words) {
                skipEmpty();
                if (run >= variant->runs.size() || variant->runs[run].first != expected) return nullptr;
                const auto count = static_cast<std::uint32_t>((variant->runs[run].second - variant->runs[run].first) / sizeof(std::uint32_t));
                covered += count;
                expected += static_cast<std::uint64_t>(count) * sizeof(std::uint32_t);
                ++run;
            }
            if (covered != region.words) return nullptr;
            total += region.words;
        }
        skipEmpty();
        if (total != words.size() || run != variant->runs.size()) return nullptr;
    }

    std::vector<std::pair<std::uint64_t, std::uint64_t>> walked;
    for (const auto& region : tagged) {
        if (region.decoder < 0 && region.words != 0) walked.emplace_back(region.address, region.address + static_cast<std::uint64_t>(region.words) * sizeof(std::uint32_t));
    }
    std::sort(walked.begin(), walked.end());
    const auto covered = [&](std::uint64_t address) {
        if ((address & 3u) != 0) return false;
        const auto found = std::upper_bound(walked.begin(), walked.end(), address, [](std::uint64_t value, const auto& range) { return value < range.first; });
        return found != walked.begin() && address + sizeof(std::uint32_t) <= (found - 1)->second;
    };

    auto reads = use.reads;
    std::sort(reads.begin(), reads.end());
    auto nested = use.nested;
    std::sort(nested.begin(), nested.end());
    auto pointerReads = use.pointers;
    std::sort(pointerReads.begin(), pointerReads.end());
    const bool tracked = use.untracked == 0;

    const auto asBase = use.asBase;
    const auto asValue = use.asValue;
    std::uint64_t pinnedUsers = 0;
    std::vector<std::uint64_t> pinnedPointers;
    const auto pinUser = [&](std::int32_t origin) {
        if (origin >= 0 && origin < 64) pinnedUsers |= std::uint64_t{1} << origin;
    };
    for (const auto& [address, origin] : reads) {
        if (origin >= 0 && !covered(address)) pinUser(origin);
    }
    for (const auto& [address, pointer] : nested) {
        if (!covered(address)) pinnedPointers.push_back(pointer);
    }

    struct WordOrigin {
        Origin origin = Origin::Unknown;
        std::uint64_t source = 0;
    };
    std::vector<WordOrigin> origins(words.size());
    {
        std::size_t position = 0;
        for (const auto& region : tagged) {
            if (region.decoder >= 0) {
                position += region.words;
                continue;
            }
            for (std::uint32_t k = 0; k < region.words; ++k, ++position) {
                const auto address = region.address + static_cast<std::uint64_t>(k) * sizeof(std::uint32_t);
                const auto low = std::lower_bound(reads.begin(), reads.end(), std::make_pair(address, std::numeric_limits<std::int32_t>::min()));
                auto high = low;
                std::int32_t user = -1;
                bool users = false, conflict = false;
                std::size_t unbased = 0;
                for (; high != reads.end() && high->first == address; ++high) {
                    if (high->second < 0) {
                        ++unbased;
                        continue;
                    }
                    if (users && high->second != user) conflict = true;
                    users = true;
                    user = high->second;
                }
                if (low == high) continue;
                const auto nestedLow = std::lower_bound(nested.begin(), nested.end(), std::make_pair(address, std::uint64_t{0}));
                std::size_t nestedCount = 0;
                std::uint64_t pointer = 0;
                bool pointers = false;
                for (auto it = nestedLow; it != nested.end() && it->first == address; ++it) {
                    ++nestedCount;
                    if (pointers && it->second != pointer) conflict = true;
                    pointers = true;
                    pointer = it->second;
                }
                const bool unknown = unbased != 0 && (!tracked || nestedCount != unbased);
                const auto kinds = (users ? 1u : 0u) + (pointers && tracked ? 1u : 0u) + (unknown ? 1u : 0u);
                if (conflict || kinds > 1u) {
                    for (auto it = low; it != high; ++it) pinUser(it->second);
                    for (auto it = nestedLow; it != nested.end() && it->first == address; ++it) pinnedPointers.push_back(it->second);
                    origins[position] = {Origin::Conflict, 0};
                } else if (users) {
                    origins[position] = {Origin::User, static_cast<std::uint64_t>(user)};
                } else if (pointers && tracked) {
                    origins[position] = {Origin::Pointer, pointer};
                }
            }
        }
    }
    std::sort(pinnedPointers.begin(), pinnedPointers.end());
    pinnedPointers.erase(std::unique(pinnedPointers.begin(), pinnedPointers.end()), pinnedPointers.end());

    const auto hidden = program.hiddenUserWords;
    std::uint64_t movable = asBase & ~asValue & ~pinnedUsers;
    if (hidden != 0) movable &= hidden >= 64 ? 0 : ~((std::uint64_t{1} << hidden) - 1);
    for (std::size_t u = 0; u < 64; ++u) {
        if (((movable >> u) & 1u) != 0 && (u >= program.userData.size() || program.userData[u] == 0)) movable &= ~(std::uint64_t{1} << u);
    }

    const auto positionOf = [&](std::uint64_t address) -> std::int64_t {
        std::size_t position = 0;
        for (const auto& region : tagged) {
            const auto end = region.address + static_cast<std::uint64_t>(region.words) * sizeof(std::uint32_t);
            if (region.decoder < 0 && address >= region.address && address < end) return static_cast<std::int64_t>(position + (address - region.address) / sizeof(std::uint32_t));
            position += region.words;
        }
        return -1;
    };
    const auto& data = variant->dataPositions;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> movablePointers;
    auto flatSlots = use.pointerSlots;
    std::sort(flatSlots.begin(), flatSlots.end(), [](const auto& a, const auto& b) { return a.second != b.second ? a.second < b.second : a.first < b.first; });
    flatSlots.erase(std::unique(flatSlots.begin(), flatSlots.end()), flatSlots.end());
    std::optional<std::uint32_t> flatBinding;
    for (std::uint32_t b = 0; b < variant->compiled->bindings.size(); ++b) {
        if (variant->compiled->bindings[b].role == ShaderRecompiler::DescriptorRole::FlattenedSrt) {
            flatBinding = b;
            break;
        }
    }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pointerSlots;
    if (tracked) {
        std::vector<std::uint64_t> candidates;
        for (const auto& [address, pointer] : nested) candidates.push_back(pointer);
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        for (const auto pointer : candidates) {
            if (std::binary_search(pinnedPointers.begin(), pinnedPointers.end(), pointer) || !covered(pointer)) continue;
            const auto readCount = static_cast<std::size_t>(std::upper_bound(reads.begin(), reads.end(), std::make_pair(pointer, std::numeric_limits<std::int32_t>::max())) - std::lower_bound(reads.begin(), reads.end(), std::make_pair(pointer, std::numeric_limits<std::int32_t>::min())));
            const auto range = std::equal_range(pointerReads.begin(), pointerReads.end(), pointer);
            if (readCount == 0 || readCount != static_cast<std::size_t>(range.second - range.first)) continue;
            const auto position = positionOf(pointer);
            if (position < 0 || std::binary_search(data.begin(), data.end(), static_cast<std::uint32_t>(position)) || words[static_cast<std::size_t>(position)] == 0) continue;
            const auto first = std::lower_bound(flatSlots.begin(), flatSlots.end(), pointer, [](const auto& entry, std::uint64_t value) { return entry.second < value; });
            const auto slots = std::make_pair(first, std::find_if(first, flatSlots.end(), [&](const auto& entry) { return entry.second != pointer; }));
            bool patchable = true;
            for (auto it = slots.first; it != slots.second && patchable; ++it) {
                const auto* descriptor = flatBinding ? &variant->compiled->bindings[*flatBinding].guestDescriptor : nullptr;
                patchable = descriptor != nullptr && it->first < descriptor->size() && (*descriptor)[it->first] == words[static_cast<std::size_t>(position)];
            }
            if (!patchable) continue;
            for (auto it = slots.first; it != slots.second; ++it) pointerSlots.emplace_back(static_cast<std::uint32_t>(position), it->first);
            movablePointers.emplace_back(pointer, static_cast<std::uint32_t>(position));
        }
    }
    const auto movablePointer = [&](std::uint64_t pointer) -> std::int64_t {
        const auto found = std::lower_bound(movablePointers.begin(), movablePointers.end(), std::make_pair(pointer, std::uint32_t{0}));
        return found != movablePointers.end() && found->first == pointer ? static_cast<std::int64_t>(found->second) : -1;
    };

    auto plan = std::make_shared<RelocatedTemplate>();
    {
        std::size_t position = 0;
        for (const auto& region : tagged) {
            if (region.decoder >= 0) {
                if (region.words != 0) plan->runs.push_back({region.address, 0, static_cast<std::uint32_t>(position), region.words, static_cast<std::uint32_t>(region.decoder), RelocatedRun::Base::Decoder});
                position += region.words;
                continue;
            }
            for (std::uint32_t k = 0; k < region.words; ++k, ++position) {
                const auto address = region.address + static_cast<std::uint64_t>(k) * sizeof(std::uint32_t);
                auto base = RelocatedRun::Base::Fixed;
                std::uint32_t source = 0;
                std::uint64_t offset = 0;
                const auto& origin = origins[position];
                if (origin.origin == Origin::User && origin.source < 64 && ((movable >> origin.source) & 1u) != 0) {
                    base = RelocatedRun::Base::User;
                    source = static_cast<std::uint32_t>(origin.source);
                    offset = address - Aligned(program.userData[source]);
                } else if (origin.origin == Origin::Pointer) {
                    if (const auto pointer = movablePointer(origin.source); pointer >= 0) {
                        base = RelocatedRun::Base::Pointer;
                        source = static_cast<std::uint32_t>(pointer);
                        offset = address - Aligned(words[source]);
                    }
                }
                auto* last = plan->runs.empty() ? nullptr : &plan->runs.back();
                const bool extends = last != nullptr && k != 0 && last->base == base && last->source == source && last->first + last->count == position && (base == RelocatedRun::Base::Fixed ? last->address + static_cast<std::uint64_t>(last->count) * sizeof(std::uint32_t) == address : last->offset + static_cast<std::uint64_t>(last->count) * sizeof(std::uint32_t) == offset);
                if (extends) {
                    ++last->count;
                    continue;
                }
                plan->runs.push_back({address, offset, static_cast<std::uint32_t>(position), 1u, source, base});
            }
        }
    }
    std::vector<std::uint32_t> runOf(words.size(), 0);
    for (std::uint32_t r = 0; r < plan->runs.size(); ++r) {
        for (std::uint32_t k = 0; k < plan->runs[r].count; ++k) runOf[plan->runs[r].first + k] = r;
    }
    {
        std::vector<std::uint8_t> placed(plan->runs.size(), 0u);
        for (std::uint32_t r = 0; r < plan->runs.size(); ++r) {
            if (plan->runs[r].base == RelocatedRun::Base::Pointer) continue;
            plan->order.push_back(r);
            placed[r] = 1u;
        }
        bool progress = true;
        while (plan->order.size() < plan->runs.size() && progress) {
            progress = false;
            for (std::uint32_t r = 0; r < plan->runs.size(); ++r) {
                if (placed[r] != 0u || placed[runOf[plan->runs[r].source]] == 0u) continue;
                plan->order.push_back(r);
                placed[r] = 1u;
                progress = true;
            }
        }
        if (plan->order.size() != plan->runs.size()) return nullptr;
    }
    for (const auto& run : plan->runs) {
        if (run.base == RelocatedRun::Base::Pointer) plan->pointers.push_back(run.source);
    }
    std::sort(plan->pointers.begin(), plan->pointers.end());
    plan->pointers.erase(std::unique(plan->pointers.begin(), plan->pointers.end()), plan->pointers.end());
    for (const auto& entry : pointerSlots) {
        if (std::binary_search(plan->pointers.begin(), plan->pointers.end(), entry.first)) plan->pointerSlots.push_back(entry);
    }
    if (flatBinding) plan->flatBinding = *flatBinding;
    plan->hashed.reserve(words.size());
    for (std::uint32_t position = 0; position < words.size(); ++position) {
        if (std::binary_search(data.begin(), data.end(), position) || std::binary_search(plan->pointers.begin(), plan->pointers.end(), position)) continue;
        plan->hashed.push_back(position);
    }

    std::uint64_t hash = Mix(0x51ed270b27f0a1c3ull, words.size());
    for (const auto& run : plan->runs) {
        hash = Mix(hash, static_cast<std::uint64_t>(run.base) | static_cast<std::uint64_t>(run.source) << 8u | static_cast<std::uint64_t>(run.count) << 40u);
        hash = Mix(hash, run.base == RelocatedRun::Base::Fixed ? run.address : run.offset);
    }
    for (const auto position : data) hash = Mix(hash, 0x100000000ull | position);
    for (const auto position : plan->pointers) hash = Mix(hash, 0x200000000ull | position);
    plan->plan = hash;
    plan->index = IndexHash(words, plan->hashed);
    plan->variant = variant;
    plan->userData = program.userData;
    for (const auto& read : decodeReads) plan->decodeBytes.push_back(static_cast<std::uint32_t>(read.Bytes().size()));
    plan->fragment = program.binary.stage == ShaderRecompiler::ShaderStage::Fragment;
    if (exports != nullptr) plan->exports = *exports;
    plan->shader = program.snapshot;
    plan->movable = movable;
    plan->baseWords = asBase;
    plan->valueWords = asValue;
    plan->fetchWords = variant->vertexInfo != nullptr ? FetchWords(*variant->vertexInfo) : 0;
    plan->hidden = hidden;
    plan->userDataBase = program.userDataBase;
    plan->firstUserSgpr = program.firstUserSgpr;
    plan->vertex = variant->vertexInfo != nullptr;
    plan->tracked = tracked;
    return plan;
}

void Driver::learnRelocatedStage(std::uint64_t shape, std::size_t stage, const DrawProgram& program, const StageCapture& capture, const std::optional<ShaderRecompiler::ShaderVertexStageInfo>& vertexInfo, const std::vector<Graphics::DecodeRead>& decodeReads, const std::vector<std::uint32_t>* exports) {
    const auto started = std::chrono::steady_clock::now();
    auto variant = DrawStageVariant(program, capture, vertexInfo, decodeReads);
    if (drawStageDataHits() && capture.capture != nullptr && !capture.capture->readTrace.leaves.empty()) recordStageDataPositions(*variant, *capture.capture, decodeReads);
    auto plan = BuildRelocatedTemplate(variant, program, capture.regions, capture.userDataUse, decodeReads, exports);
    const auto key = PoolKey(shape, stage);
    std::lock_guard lock(relocatedMutex);
    auto& counters = relocatedCounters;
    counters.learnUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
    if (plan == nullptr) {
        ++counters.learnRefused;
        return;
    }
    ++counters.learned;
    if (!plan->tracked) ++counters.nestedUntracked;
    for (const auto& run : plan->runs) counters.nestedRuns += run.base == RelocatedRun::Base::Pointer ? 1u : 0u;
    auto [found, created] = relocatedPools.try_emplace(key);
    auto& pool = found->second;
    if (created) {
        relocatedOrder.push_front(key);
        pool.order = relocatedOrder.begin();
    } else {
        relocatedOrder.splice(relocatedOrder.begin(), relocatedOrder, pool.order);
    }
    const auto stamp = ++relocatedClock;
    std::size_t slot = pool.templates.size();
    for (std::size_t t = 0; t < pool.templates.size(); ++t) {
        if (pool.templates[t]->plan == plan->plan && pool.templates[t]->index == plan->index) {
            slot = t;
            ++counters.replaced;
            break;
        }
    }
    if (slot == pool.templates.size() && pool.templates.size() >= PoolTemplates) {
        slot = static_cast<std::size_t>(std::min_element(pool.used.begin(), pool.used.end()) - pool.used.begin());
        ++counters.evicted;
    }
    if (slot == pool.templates.size()) {
        pool.templates.push_back(std::move(plan));
        pool.used.push_back(stamp);
        ++relocatedTemplates;
    } else {
        pool.templates[slot] = std::move(plan);
        pool.used[slot] = stamp;
    }
    while (relocatedTemplates > MaxTemplates && relocatedOrder.size() > 1) {
        const auto victim = relocatedPools.find(relocatedOrder.back());
        relocatedTemplates -= victim->second.templates.size();
        counters.evicted += victim->second.templates.size();
        relocatedPools.erase(victim);
        relocatedOrder.pop_back();
    }
}

std::optional<RelocatedHit> Driver::relocatedDrawStage(const RelocatedLookup& lookup) {
    const auto started = std::chrono::steady_clock::now();
    const auto& program = *lookup.program;
    const auto key = PoolKey(lookup.shape, lookup.stage);
    std::vector<std::pair<std::uint64_t, std::shared_ptr<const RelocatedTemplate>>> templates;
    {
        std::lock_guard lock(relocatedMutex);
        ++relocatedCounters.lookups;
        const auto found = relocatedPools.find(key);
        if (found != relocatedPools.end()) {
            relocatedOrder.splice(relocatedOrder.begin(), relocatedOrder, found->second.order);
            templates.reserve(found->second.templates.size());
            for (std::size_t t = 0; t < found->second.templates.size(); ++t) templates.emplace_back(found->second.used[t], found->second.templates[t]);
        }
    }
    std::sort(templates.begin(), templates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    auto furthest = RelocatedFallback::NoTemplate;
    const auto fallback = [&](RelocatedFallback reason) { furthest = std::max(furthest, reason); };
    struct Gathered {
        std::uint64_t plan = 0;
        bool readable = false;
        std::uint64_t index = 0;
        std::vector<std::uint32_t> words;
        std::vector<std::uint64_t> addresses;
    };
    std::vector<Gathered> gathered;
    const auto& vertexInfo = *lookup.vertexInfo;
    const auto& now = program.userData;
    std::optional<RelocatedHit> hit;
    std::shared_ptr<const RelocatedTemplate> chosen;
    std::uint64_t imagesFlushed = 0, runsSynced = 0;
    std::size_t validations = 0;
    std::optional<SampledReadScope> sampling;
    const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> live;
    for (const auto& [stamp, held] : templates) {
        const auto& t = *held;
        const auto& kept = *t.variant;
        if (t.shader != program.snapshot || kept.pushOffset != lookup.pushOffset || t.userData.size() != now.size() || t.hidden != program.hiddenUserWords || t.userDataBase != program.userDataBase || t.firstUserSgpr != program.firstUserSgpr || t.vertex != vertexInfo.has_value() || (t.fragment && (lookup.exports == nullptr ? !t.exports.empty() : *lookup.exports != t.exports))) {
            fallback(RelocatedFallback::Layout);
            continue;
        }
        const auto registers = kept.compiled->userDataRegisterReads;
        const auto& forwarded = kept.compiled->shaderDataUserWords;
        std::optional<RelocatedFallback> refused;
        for (std::size_t w = 0; w < now.size() && !refused; ++w) {
            if (now[w] == t.userData[w]) continue;
            if (w >= 64) {
                refused = RelocatedFallback::ValueWord;
                break;
            }
            const auto bit = std::uint64_t{1} << w;
            if ((kept.userDataMask & bit) == 0) continue;
            if (w < t.hidden || (t.valueWords & bit) != 0) refused = RelocatedFallback::ValueWord;
            else if ((t.movable & bit) != 0) {
                if (now[w] == 0 || ((now[w] ^ t.userData[w]) & 3u) != 0) refused = RelocatedFallback::PinnedBase;
                else if ((registers & bit) != 0 && std::find(forwarded.begin(), forwarded.end(), static_cast<std::uint32_t>(w)) == forwarded.end()) refused = RelocatedFallback::ValueWord;
            } else if ((t.baseWords & bit) != 0) refused = RelocatedFallback::PinnedBase;
            else if ((t.fetchWords & bit) == 0 || (registers & bit) != 0) refused = RelocatedFallback::ValueWord;
        }
        if (refused) {
            fallback(*refused);
            continue;
        }
        if (t.vertex && (!sameVertexInfo(*vertexInfo, *kept.vertexInfo) || lookup.decodeReads.size() != t.decodeBytes.size() || !std::equal(t.decodeBytes.begin(), t.decodeBytes.end(), lookup.decodeReads.begin(), [](std::uint32_t bytes, const Graphics::DecodeRead& read) { return bytes == read.Bytes().size(); }))) {
            fallback(RelocatedFallback::Decode);
            continue;
        }
        auto plan = std::find_if(gathered.begin(), gathered.end(), [&](const Gathered& g) { return g.plan == t.plan; });
        if (plan == gathered.end()) {
            if (gathered.size() >= MaxPlans) {
                fallback(RelocatedFallback::NoIndex);
                continue;
            }
            auto& g = gathered.emplace_back();
            g.plan = t.plan;
            g.words.assign(kept.words.size(), 0u);
            g.addresses.assign(t.runs.size(), 0u);
            g.readable = true;
            for (const auto r : t.order) {
                const auto& run = t.runs[r];
                std::uint64_t address = run.address;
                switch (run.base) {
                    case RelocatedRun::Base::Fixed: break;
                    case RelocatedRun::Base::User: address = Aligned(now[run.source]) + run.offset; break;
                    case RelocatedRun::Base::Pointer: {
                        const auto pointer = g.words[run.source];
                        if (pointer == 0) g.readable = false;
                        address = Aligned(pointer) + run.offset;
                        break;
                    }
                    case RelocatedRun::Base::Decoder: address = lookup.decodeReads[run.source].address + run.offset; break;
                }
                if (!g.readable) break;
                g.addresses[r] = address;
                if (GuestMemory::CopyMapped(address, std::as_writable_bytes(std::span(g.words).subspan(run.first, run.count))) != GuestMemory::Compare::Equal) {
                    g.readable = false;
                    break;
                }
            }
            if (g.readable) g.index = IndexHash(g.words, t.hashed);
            plan = gathered.end() - 1;
        }
        if (!plan->readable) {
            fallback(RelocatedFallback::Unreadable);
            continue;
        }
        if (plan->index != t.index) {
            fallback(RelocatedFallback::NoIndex);
            continue;
        }
        if (!std::all_of(t.pointers.begin(), t.pointers.end(), [&](std::uint32_t p) { return plan->words[p] != 0 && kept.words[p] != 0 && ((plan->words[p] ^ kept.words[p]) & 3u) == 0; })) {
            fallback(RelocatedFallback::PointerBits);
            continue;
        }
        if (validations >= MaxValidations) break;
        ++validations;
        auto copy = std::make_shared<DispatchVariant>();
        copy->runs.reserve(t.runs.size());
        copy->runBases.reserve(t.runs.size());
        copy->relocatableWords = kept.relocatableWords & t.movable;
        for (std::size_t r = 0; r < t.runs.size(); ++r) {
            const auto& run = t.runs[r];
            copy->runs.emplace_back(plan->addresses[r], plan->addresses[r] + static_cast<std::uint64_t>(run.count) * sizeof(std::uint32_t));
            copy->runBases.push_back(run.base == RelocatedRun::Base::User && ((copy->relocatableWords >> run.source) & 1u) != 0 ? static_cast<std::int8_t>(run.source) : std::int8_t{-1});
        }
        copy->words = kept.words;
        for (const auto p : t.pointers) copy->words[p] = plan->words[p];
        copy->forgetSerial = kept.forgetSerial;
        copy->compiled = kept.compiled;
        copy->meshIndexDescriptor = kept.meshIndexDescriptor;
        copy->shader = kept.shader;
        copy->dataPositions = kept.dataPositions;
        copy->dataSlots = kept.dataSlots;
        copy->flatBinding = kept.flatBinding;
        copy->pushOffset = kept.pushOffset;
        copy->userDataMask = kept.userDataMask;
        copy->vertexInfo = kept.vertexInfo;
        RelocatedHit candidate;
        appendEntryRegions(*copy, candidate.regions);
        auto* liveWords = copy->dataPositions.empty() ? nullptr : &live;
        auto result = validateVariant(program.binary.codeAddress, lookup.queue, *copy, candidate.regions, imagesFlushed, runsSynced, sampling, liveWords);
        if (gateRetry() && (result == EntryOutcome::PublishMoved || result == EntryOutcome::PendingMoved)) result = validateVariant(program.binary.codeAddress, lookup.queue, *copy, candidate.regions, imagesFlushed, runsSynced, sampling, liveWords);
        if (result != EntryOutcome::Equal && result != EntryOutcome::EqualData) {
            fallback(result == EntryOutcome::Differing ? RelocatedFallback::Differing : RelocatedFallback::Gate);
            continue;
        }
        if (liveWords != nullptr) {
            for (const auto& [position, value] : live) copy->words[position] = value;
        }
        for (std::size_t w = 0; w < now.size() && w < 64; ++w) {
            if (now[w] == t.userData[w]) continue;
            for (std::uint32_t slot = 0; slot < forwarded.size(); ++slot) {
                if (forwarded[slot] == w) copy->shaderDataPatches.emplace_back(slot, now[w]);
            }
        }
        applyShaderDataPatches(*copy);
        if (liveWords != nullptr && !live.empty() && copy->flatBinding != DispatchVariant::NoFlatBinding) copy->compiled = patchDataWords(*copy, copy->words);
        if (std::any_of(t.pointerSlots.begin(), t.pointerSlots.end(), [&](const auto& entry) { return copy->words[entry.first] != kept.words[entry.first]; })) {
            auto patched = std::make_shared<ShaderRecompiler::RecompileResult>(*copy->compiled);
            auto& descriptor = patched->bindings[t.flatBinding].guestDescriptor;
            for (const auto& [position, slot] : t.pointerSlots) descriptor[slot] = copy->words[position];
            copy->compiled = std::move(patched);
        }
        bool moved = false;
        for (std::size_t r = 0; r < t.runs.size() && !moved; ++r) moved = plan->addresses[r] != t.runs[r].address;
        const bool nestedMoved = std::any_of(t.pointers.begin(), t.pointers.end(), [&](std::uint32_t p) { return plan->words[p] != kept.words[p]; });
        candidate.kind = nestedMoved ? RelocatedClass::Nested : liveWords != nullptr && !live.empty() ? RelocatedClass::Data : moved ? RelocatedClass::Moved : RelocatedClass::Same;
        candidate.regions.clear();
        appendEntryRegions(*copy, candidate.regions);
        candidate.variant = std::move(copy);
        hit = std::move(candidate);
        chosen = held;
        break;
    }
    const auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
    std::lock_guard lock(relocatedMutex);
    auto& counters = relocatedCounters;
    counters.validations += validations;
    counters.synced += runsSynced;
    if (hit) {
        ++counters.hits[static_cast<std::size_t>(hit->kind)];
        counters.hitUs += us;
        const auto found = relocatedPools.find(key);
        if (found != relocatedPools.end()) {
            for (std::size_t t = 0; t < found->second.templates.size(); ++t) {
                if (found->second.templates[t] == chosen) found->second.used[t] = ++relocatedClock;
            }
        }
    } else {
        ++counters.fallbacks[static_cast<std::size_t>(furthest)];
        counters.missUs += us;
    }
    return hit;
}

void Driver::maybeReportRelocatedHits(bool profile) {
    if (!profile) return;
    std::unique_lock lock(relocatedMutex);
    auto& c = relocatedCounters;
    const auto now = std::chrono::steady_clock::now();
    if (now - c.lastReport < std::chrono::seconds(10)) return;
    const auto n = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    const auto hit = [&](RelocatedClass kind) { return n(c.hits[static_cast<std::size_t>(kind)]); };
    const auto miss = [&](RelocatedFallback reason) { return n(c.fallbacks[static_cast<std::size_t>(reason)]); };
    std::uint64_t hits = 0, misses = 0;
    for (const auto value : c.hits) hits += value;
    for (const auto value : c.fallbacks) misses += value;
    const auto seconds = std::chrono::duration<double>(now - c.lastReport).count();
    char line[1400];
    std::snprintf(line, sizeof(line), "[relocated-hits] %.0f s: %llu stage lookups, %llu hits (R0 same %llu, R1 moved %llu, R2 moved+data %llu, R4 nested %llu), %.1f us per hit, %.1f us per fallback; fallbacks: no template %llu, layout %llu, value word %llu, pinned base %llu, unreadable %llu, no index match %llu, pointer bits %llu, vertex decode %llu, differing %llu, gate %llu; %llu validations, %llu runs synced; learned %llu (%.1f us each, %llu refused, %llu replaced, %llu evicted, %llu nested runs, %llu without nested tracking); %zu templates in %zu pools; verified %llu (%llu mismatches)\n", seconds, n(c.lookups), n(hits), hit(RelocatedClass::Same), hit(RelocatedClass::Moved), hit(RelocatedClass::Data), hit(RelocatedClass::Nested), hits != 0 ? c.hitUs / static_cast<double>(hits) : 0.0, misses != 0 ? c.missUs / static_cast<double>(misses) : 0.0, miss(RelocatedFallback::NoTemplate), miss(RelocatedFallback::Layout), miss(RelocatedFallback::ValueWord), miss(RelocatedFallback::PinnedBase), miss(RelocatedFallback::Unreadable), miss(RelocatedFallback::NoIndex), miss(RelocatedFallback::PointerBits), miss(RelocatedFallback::Decode), miss(RelocatedFallback::Differing), miss(RelocatedFallback::Gate), n(c.validations), n(c.synced), n(c.learned), c.learned != 0 ? c.learnUs / static_cast<double>(c.learned) : 0.0, n(c.learnRefused), n(c.replaced), n(c.evicted), n(c.nestedRuns), n(c.nestedUntracked), relocatedTemplates, relocatedPools.size(), n(c.verified), n(c.verifyMismatches));
    c = RelocatedCounters{};
    c.lastReport = now;
    lock.unlock();
    AgcDriver::ProfilePrint_nid_no_patch("%s", line);
}

}
