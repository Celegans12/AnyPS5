#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::size_t MaxPlans = 4;
constexpr std::size_t MaxValidations = 2;
constexpr std::size_t MaxIdentities = 16;
constexpr std::size_t SightingSlots = std::size_t{1} << 16u;

std::size_t EnvCount(const char* name, std::size_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr) return fallback;
    const auto parsed = std::strtoull(text, nullptr, 10);
    return parsed != 0 ? static_cast<std::size_t>(parsed) : fallback;
}

// APS5_RELOCATED_TEMPLATES=<n>: templates kept in all (least recently looked-up pools dropped first).
std::size_t TemplateCap() {
    static const std::size_t cap = EnvCount("APS5_RELOCATED_TEMPLATES", 65536);
    return cap;
}

// APS5_RELOCATED_POOL=<n>: templates kept per (shape key, stage).
std::size_t PoolCap() {
    static const std::size_t cap = EnvCount("APS5_RELOCATED_POOL", 512);
    return cap;
}

// APS5_RELOCATED_LEARN_AFTER=<n>: a stage capture becomes a template on the n-th miss of the same
// sighting (pool, plan and index the miss gathered, fixed user words); 1 learns every miss. A pool
// without templates learns its first miss.
std::uint32_t LearnAfter() {
    static const auto after = static_cast<std::uint32_t>(std::max<std::size_t>(EnvCount("APS5_RELOCATED_LEARN_AFTER", 2), 1));
    return after;
}

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

std::uint64_t IdentityHash(std::span<const std::uint32_t> userData, std::uint64_t fixed) {
    std::uint64_t hash = Mix(0x6a09e667f3bcc909ull, userData.size());
    for (std::size_t w = 0; w < userData.size(); ++w) {
        if (w < 64 && ((fixed >> w) & 1u) == 0) continue;
        hash = Mix(hash, static_cast<std::uint64_t>(w) << 32u | userData[w]);
    }
    return hash;
}

bool FixedWordsEqual(std::span<const std::uint32_t> a, std::span<const std::uint32_t> b, std::uint64_t fixed) {
    if (a.size() != b.size()) return false;
    for (std::size_t w = 0; w < a.size(); ++w) {
        if (a[w] != b[w] && (w >= 64 || ((fixed >> w) & 1u) != 0)) return false;
    }
    return true;
}

std::uint32_t DecodeWord(const Graphics::DecodeRead& read, std::uint32_t word) {
    std::uint32_t value = 0;
    std::memcpy(&value, read.Bytes().data() + static_cast<std::size_t>(word) * sizeof(std::uint32_t), sizeof(value));
    return value;
}

// Equal vertex stage infos but for the base address, stride and record count of each vertex buffer
// (V# words 0-2) of an embedded fetch: the module, its specialization and the input validation
// depend on the V# only through word 3 (format and selectors); the buffers are bound from the
// result's vertex attributes, which the reuse takes from the fresh decode.
bool SameVertexLayout(const ShaderRecompiler::ShaderVertexStageInfo& a, const ShaderRecompiler::ShaderVertexStageInfo& b) {
    if (!a.fetchEmbedded || !b.fetchEmbedded || a.resourcesNum != b.resourcesNum || a.resourcesNum > a.resources.size() || a.fetchAttribReg != b.fetchAttribReg || a.fetchBufferReg != b.fetchBufferReg || a.paClVsOutCntl != b.paClVsOutCntl || a.drawBaseVertexSgpr != b.drawBaseVertexSgpr || a.drawStartInstanceSgpr != b.drawStartInstanceSgpr || a.pointList != b.pointList || a.pointSize != b.pointSize || a.pointSizeMin != b.pointSizeMin || a.pointSizeMax != b.pointSizeMax || a.depthRemapMode != b.depthRemapMode) return false;
    for (std::uint32_t i = 0; i < a.resourcesNum; ++i) {
        if (a.resources[i].fields[3] != b.resources[i].fields[3]) return false;
        const auto& x = a.resourcesDst[i];
        const auto& y = b.resourcesDst[i];
        if (x.registerStart != y.registerStart || x.registersNum != y.registersNum || x.attrId != y.attrId || x.fetchIndex != y.fetchIndex) return false;
    }
    return true;
}

template <typename T>
std::size_t VectorBytes(const std::vector<T>& values) {
    return values.capacity() * sizeof(T);
}

std::size_t CompiledBytes(const ShaderRecompiler::RecompileResult& compiled) {
    std::size_t bytes = sizeof(compiled) + VectorBytes(compiled.pushConstants) + VectorBytes(compiled.vertexAttributes) + VectorBytes(compiled.shaderDataUserWords) + VectorBytes(compiled.parameterExports) + VectorBytes(compiled.specialization) + VectorBytes(compiled.bindings);
    for (const auto& binding : compiled.bindings) bytes += VectorBytes(binding.guestDescriptor);
    return bytes;
}

std::size_t TemplateBytes(const RelocatedTemplate& t) {
    std::size_t bytes = sizeof(t) + VectorBytes(t.runs) + VectorBytes(t.order) + VectorBytes(t.pointers) + VectorBytes(t.pointerSlots) + VectorBytes(t.hashed) + VectorBytes(t.userData) + VectorBytes(t.decodeBytes) + VectorBytes(t.exports) + VectorBytes(t.vertexWords);
    if (t.variant != nullptr) {
        const auto& v = *t.variant;
        bytes += sizeof(v) + VectorBytes(v.runs) + VectorBytes(v.words) + VectorBytes(v.dataPositions) + VectorBytes(v.dataSlots) + VectorBytes(v.runBases);
        if (v.vertexInfo != nullptr) bytes += sizeof(*v.vertexInfo);
    }
    return bytes;
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
    const auto& compiledResult = *variant->compiled;
    const auto registers = compiledResult.userDataRegisterReads;
    std::uint64_t forwarded = 0;
    {
        const auto shaderData = std::find_if(compiledResult.bindings.begin(), compiledResult.bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::ShaderData; });
        std::uint64_t uncovered = 0;
        for (std::uint32_t slot = 0; slot < compiledResult.shaderDataUserWords.size(); ++slot) {
            const auto w = compiledResult.shaderDataUserWords[slot];
            if (w >= 64) continue;
            const bool covered = (static_cast<std::size_t>(slot) + 1u) * sizeof(std::uint32_t) <= compiledResult.pushConstants.size() || (shaderData != compiledResult.bindings.end() && slot < shaderData->guestDescriptor.size());
            (covered ? forwarded : uncovered) |= std::uint64_t{1} << w;
        }
        forwarded &= ~uncovered;
    }
    movable &= ~(registers & ~forwarded);

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
    const auto fetchWords = variant->vertexInfo != nullptr ? FetchWords(*variant->vertexInfo) : 0;
    const auto mask = variant->userDataMask;
    std::uint64_t fixed = 0, patched = 0;
    for (std::size_t w = 0; w < 64 && w < program.userData.size(); ++w) {
        const auto bit = std::uint64_t{1} << w;
        if ((mask & bit) == 0 || (movable & bit) != 0) continue;
        if (w < hidden || (asValue & bit) != 0 || (asBase & bit) != 0) fixed |= bit;
        else if ((registers & bit) != 0) ((forwarded & bit) != 0 ? patched : fixed) |= bit;
        else if ((fetchWords & bit) == 0) fixed |= bit;
    }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> vertexWords;
    if (const auto* info = variant->vertexInfo.get(); info != nullptr && info->fetchEmbedded && info->resourcesNum != 0 && info->resourcesNum <= info->resources.size() && decodeReads.size() == 2u * info->resourcesNum) {
        bool laidOut = true;
        for (std::uint32_t i = 0; i < info->resourcesNum && laidOut; ++i) {
            const auto sharp = 2u * i + 1u;
            const auto run = std::find_if(plan->runs.begin(), plan->runs.end(), [&](const RelocatedRun& entry) { return entry.base == RelocatedRun::Base::Decoder && entry.source == sharp; });
            laidOut = decodeReads[sharp - 1u].Bytes().size() == sizeof(std::uint32_t) && decodeReads[sharp].Bytes().size() == 4u * sizeof(std::uint32_t) && run != plan->runs.end() && run->count == 4u;
            for (std::uint32_t word = 0; word < 3u && laidOut; ++word) vertexWords.emplace_back(run->first + word, sharp * 4u + word);
        }
        if (!laidOut) vertexWords.clear();
    }
    std::sort(vertexWords.begin(), vertexWords.end());
    plan->hashed.reserve(words.size());
    for (std::uint32_t position = 0; position < words.size(); ++position) {
        if (std::binary_search(data.begin(), data.end(), position) || std::binary_search(plan->pointers.begin(), plan->pointers.end(), position)) continue;
        if (std::binary_search(vertexWords.begin(), vertexWords.end(), std::make_pair(position, std::uint32_t{0}), [](const auto& a, const auto& b) { return a.first < b.first; })) continue;
        plan->hashed.push_back(position);
    }

    std::uint64_t hash = Mix(0x51ed270b27f0a1c3ull, words.size());
    for (const auto& run : plan->runs) {
        hash = Mix(hash, static_cast<std::uint64_t>(run.base) | static_cast<std::uint64_t>(run.source) << 8u | static_cast<std::uint64_t>(run.count) << 40u);
        hash = Mix(hash, run.base == RelocatedRun::Base::Fixed ? run.address : run.offset);
    }
    for (const auto position : data) hash = Mix(hash, 0x100000000ull | position);
    for (const auto position : plan->pointers) hash = Mix(hash, 0x200000000ull | position);
    for (const auto& [position, source] : vertexWords) hash = Mix(hash, 0x300000000ull | position);
    hash = Mix(Mix(hash, fixed), patched);
    plan->plan = hash;
    plan->index = IndexHash(words, plan->hashed);
    plan->identity = IdentityHash(program.userData, fixed);
    plan->variant = variant;
    plan->userData = program.userData;
    for (const auto& read : decodeReads) plan->decodeBytes.push_back(static_cast<std::uint32_t>(read.Bytes().size()));
    plan->fragment = program.binary.stage == ShaderRecompiler::ShaderStage::Fragment;
    if (exports != nullptr) plan->exports = *exports;
    plan->shader = program.snapshot;
    plan->tail = program.tailContinuation;
    plan->codeOffset = program.codeOffset;
    plan->movable = movable;
    plan->baseWords = asBase;
    plan->valueWords = asValue;
    plan->fetchWords = fetchWords;
    plan->fixedWords = fixed;
    plan->patchedWords = patched;
    plan->registerWords = registers;
    plan->vertexMovable = !vertexWords.empty();
    plan->vertexWords = std::move(vertexWords);
    plan->hidden = hidden;
    plan->userDataBase = program.userDataBase;
    plan->firstUserSgpr = program.firstUserSgpr;
    plan->vertex = variant->vertexInfo != nullptr;
    plan->tracked = tracked;
    plan->bytes = TemplateBytes(*plan);
    plan->compiledBytes = CompiledBytes(compiledResult);
    return plan;
}

void Driver::learnRelocatedStage(std::uint64_t shape, std::size_t stage, const DrawProgram& program, const StageCapture& capture, const std::optional<ShaderRecompiler::ShaderVertexStageInfo>& vertexInfo, const std::vector<Graphics::DecodeRead>& decodeReads, const std::vector<std::uint32_t>* exports) {
    const auto started = std::chrono::steady_clock::now();
    auto variant = DrawStageVariant(program, capture, vertexInfo, decodeReads);
    if (drawStageDataHits() && capture.capture != nullptr && !capture.capture->readTrace.leaves.empty()) recordStageDataPositions(*variant, *capture.capture, decodeReads);
    std::shared_ptr<const RelocatedTemplate> plan = BuildRelocatedTemplate(variant, program, capture.regions, capture.userDataUse, decodeReads, exports);
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
    if (plan->vertexMovable) ++counters.vertexMovable;
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
    const auto account = [&](const RelocatedTemplate& t, bool add) {
        if (add) {
            relocatedBytes += t.bytes;
            relocatedCompiledBytes += t.compiledBytes;
        } else {
            relocatedBytes -= t.bytes;
            relocatedCompiledBytes -= t.compiledBytes;
        }
    };
    const auto release = [&](RelocatedPool& from, const RelocatedTemplate& t) {
        account(t, false);
        const auto entry = std::find_if(from.plans.begin(), from.plans.end(), [&](const RelocatedPool::Plan& p) { return p.plan == t.plan; });
        if (entry != from.plans.end() && --entry->templates == 0) from.plans.erase(entry);
        --from.templates;
        --relocatedTemplates;
        ++counters.evicted;
    };
    auto& kept = pool.byIndex[Mix(plan->plan, plan->index)];
    const auto same = std::find_if(kept.begin(), kept.end(), [&](const RelocatedPool::Kept& k) { return k.held->identity == plan->identity && FixedWordsEqual(k.held->userData, plan->userData, plan->fixedWords); });
    if (same != kept.end()) {
        account(*same->held, false);
        same->held = plan;
        same->used = stamp;
        account(*plan, true);
        ++counters.replaced;
    } else {
        if (kept.size() >= MaxIdentities) {
            const auto oldest = std::min_element(kept.begin(), kept.end(), [](const RelocatedPool::Kept& a, const RelocatedPool::Kept& b) { return a.used < b.used; });
            release(pool, *oldest->held);
            kept.erase(oldest);
        }
        kept.push_back({plan, stamp});
        account(*plan, true);
        ++pool.templates;
        ++relocatedTemplates;
        const auto entry = std::find_if(pool.plans.begin(), pool.plans.end(), [&](const RelocatedPool::Plan& p) { return p.plan == plan->plan; });
        if (entry != pool.plans.end()) ++entry->templates;
        else pool.plans.push_back({plan->plan, plan, stamp, 1});
    }
    for (auto& entry : pool.plans) {
        if (entry.plan == plan->plan) entry.used = stamp;
    }
    while (pool.templates > PoolCap()) {
        auto oldestList = pool.byIndex.end();
        std::size_t oldestSlot = 0;
        std::uint64_t oldestUsed = std::numeric_limits<std::uint64_t>::max();
        for (auto list = pool.byIndex.begin(); list != pool.byIndex.end(); ++list) {
            for (std::size_t slot = 0; slot < list->second.size(); ++slot) {
                if (list->second[slot].used >= oldestUsed) continue;
                oldestUsed = list->second[slot].used;
                oldestList = list;
                oldestSlot = slot;
            }
        }
        if (oldestList == pool.byIndex.end()) break;
        release(pool, *oldestList->second[oldestSlot].held);
        oldestList->second.erase(oldestList->second.begin() + static_cast<std::ptrdiff_t>(oldestSlot));
        if (oldestList->second.empty()) pool.byIndex.erase(oldestList);
    }
    while (relocatedTemplates > TemplateCap() && relocatedOrder.size() > 1) {
        const auto victim = relocatedPools.find(relocatedOrder.back());
        for (const auto& [index, list] : victim->second.byIndex) {
            for (const auto& entry : list) account(*entry.held, false);
        }
        relocatedTemplates -= victim->second.templates;
        counters.evicted += victim->second.templates;
        relocatedPools.erase(victim);
        relocatedOrder.pop_back();
    }
}

std::optional<RelocatedHit> Driver::relocatedDrawStage(const RelocatedLookup& lookup) {
    const auto started = std::chrono::steady_clock::now();
    const auto& program = *lookup.program;
    const auto key = PoolKey(lookup.shape, lookup.stage);
    const bool vertexBuffers = !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoRelocatedVertexBuffers);
    const bool registerWords = !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoRelocatedRegisterWords);
    if (lookup.learn != nullptr) *lookup.learn = false;
    std::vector<std::shared_ptr<const RelocatedTemplate>> exemplars;
    {
        std::lock_guard lock(relocatedMutex);
        ++relocatedCounters.lookups;
        const auto found = relocatedPools.find(key);
        if (found != relocatedPools.end()) {
            relocatedOrder.splice(relocatedOrder.begin(), relocatedOrder, found->second.order);
            auto plans = found->second.plans;
            std::sort(plans.begin(), plans.end(), [](const RelocatedPool::Plan& a, const RelocatedPool::Plan& b) { return a.used > b.used; });
            for (std::size_t p = 0; p < plans.size() && p < MaxPlans; ++p) exemplars.push_back(plans[p].exemplar);
        }
    }

    auto furthest = RelocatedFallback::NoTemplate;
    const auto fallback = [&](RelocatedFallback reason) { furthest = std::max(furthest, reason); };
    std::optional<RelocatedValueWord> valueWhy;
    struct Gathered {
        std::shared_ptr<const RelocatedTemplate> exemplar;
        bool readable = false;
        std::uint64_t index = 0;
        std::uint64_t fixed = 0;
        std::vector<std::uint32_t> words;
        std::vector<std::uint64_t> addresses;
        std::vector<RelocatedPool::Kept> candidates;
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
    for (const auto& held : exemplars) {
        const auto& e = *held;
        if (e.userData.size() != now.size()) {
            fallback(RelocatedFallback::Layout);
            continue;
        }
        if (std::any_of(e.runs.begin(), e.runs.end(), [&](const RelocatedRun& run) { return run.base == RelocatedRun::Base::Decoder && (run.source >= lookup.decodeReads.size() || lookup.decodeReads[run.source].Bytes().size() != static_cast<std::size_t>(run.count) * sizeof(std::uint32_t)); })) {
            fallback(RelocatedFallback::DecodeReads);
            continue;
        }
        auto& g = gathered.emplace_back();
        g.exemplar = held;
        g.words.assign(e.variant->words.size(), 0u);
        g.addresses.assign(e.runs.size(), 0u);
        g.readable = true;
        for (const auto r : e.order) {
            const auto& run = e.runs[r];
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
        if (!g.readable) {
            fallback(RelocatedFallback::Unreadable);
            continue;
        }
        g.index = IndexHash(g.words, e.hashed);
        g.fixed = IdentityHash(now, e.fixedWords);
    }
    {
        std::lock_guard lock(relocatedMutex);
        const auto found = relocatedPools.find(key);
        for (auto& g : gathered) {
            if (!g.readable || found == relocatedPools.end()) continue;
            const auto list = found->second.byIndex.find(Mix(g.exemplar->plan, g.index));
            if (list == found->second.byIndex.end()) continue;
            g.candidates = list->second;
        }
    }
    const auto classifyValueWord = [&](const RelocatedTemplate& t) {
        for (std::size_t w = 0; w < now.size() && w < t.userData.size(); ++w) {
            if (now[w] == t.userData[w]) continue;
            const auto bit = w < 64 ? std::uint64_t{1} << w : 0;
            if (w < 64 && ((t.fixedWords | (registerWords ? 0 : t.patchedWords)) & bit) == 0) continue;
            return w >= 64 ? RelocatedValueWord::Other : w < t.hidden ? RelocatedValueWord::Hidden : (t.valueWords & bit) != 0 ? RelocatedValueWord::AsValue : (t.baseWords & bit) != 0 ? RelocatedValueWord::Base : (t.registerWords & bit) != 0 ? RelocatedValueWord::Register : RelocatedValueWord::Other;
        }
        return RelocatedValueWord::Other;
    };
    std::vector<std::pair<std::uint32_t, std::uint32_t>> live;
    for (auto& g : gathered) {
        if (!g.readable) continue;
        if (g.candidates.empty()) {
            fallback(RelocatedFallback::NoIndex);
            continue;
        }
        std::sort(g.candidates.begin(), g.candidates.end(), [](const RelocatedPool::Kept& a, const RelocatedPool::Kept& b) { return a.used > b.used; });
        bool identified = false;
        for (const auto& candidate : g.candidates) {
            const auto& t = *candidate.held;
            const auto& kept = *t.variant;
            const auto required = t.fixedWords | (registerWords ? 0 : t.patchedWords);
            if ((registerWords && t.identity != g.fixed) || !FixedWordsEqual(now, t.userData, required)) continue;
            identified = true;
            if (t.runs != g.exemplar->runs || kept.words.size() != g.words.size()) {
                fallback(RelocatedFallback::NoIndex);
                continue;
            }
            if (t.shader != program.snapshot || t.tail != program.tailContinuation || t.codeOffset != program.codeOffset || kept.pushOffset != lookup.pushOffset || t.hidden != program.hiddenUserWords || t.userDataBase != program.userDataBase || t.firstUserSgpr != program.firstUserSgpr || t.vertex != vertexInfo.has_value() || (t.fragment && (lookup.exports == nullptr ? !t.exports.empty() : *lookup.exports != t.exports))) {
                fallback(RelocatedFallback::Layout);
                continue;
            }
            bool pinned = false, patched = false;
            for (std::size_t w = 0; w < now.size() && w < 64; ++w) {
                if (now[w] == t.userData[w]) continue;
                const auto bit = std::uint64_t{1} << w;
                if ((t.movable & bit) != 0 && (now[w] == 0 || ((now[w] ^ t.userData[w]) & 3u) != 0)) pinned = true;
                if ((t.patchedWords & bit) != 0) patched = true;
            }
            if (pinned) {
                fallback(RelocatedFallback::PinnedBase);
                continue;
            }
            bool vertexMoved = false;
            if (t.vertex) {
                const bool sameReads = lookup.decodeReads.size() == t.decodeBytes.size() && std::equal(t.decodeBytes.begin(), t.decodeBytes.end(), lookup.decodeReads.begin(), [](std::uint32_t bytes, const Graphics::DecodeRead& read) { return bytes == read.Bytes().size(); });
                const bool sameInfo = sameReads && sameVertexInfo(*vertexInfo, *kept.vertexInfo);
                vertexMoved = sameReads && !sameInfo && vertexBuffers && t.vertexMovable && SameVertexLayout(*vertexInfo, *kept.vertexInfo);
                if (!sameInfo && !vertexMoved) {
                    fallback(RelocatedFallback::Decode);
                    continue;
                }
            }
            if (!std::all_of(t.pointers.begin(), t.pointers.end(), [&](std::uint32_t p) { return g.words[p] != 0 && kept.words[p] != 0 && ((g.words[p] ^ kept.words[p]) & 3u) == 0; })) {
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
                copy->runs.emplace_back(g.addresses[r], g.addresses[r] + static_cast<std::uint64_t>(run.count) * sizeof(std::uint32_t));
                copy->runBases.push_back(run.base == RelocatedRun::Base::User && ((copy->relocatableWords >> run.source) & 1u) != 0 ? static_cast<std::int8_t>(run.source) : std::int8_t{-1});
            }
            copy->words = kept.words;
            for (const auto p : t.pointers) copy->words[p] = g.words[p];
            for (const auto& [position, source] : t.vertexWords) copy->words[position] = DecodeWord(lookup.decodeReads[source / 4u], source % 4u);
            copy->forgetSerial = kept.forgetSerial;
            copy->compiled = kept.compiled;
            copy->meshIndexDescriptor = kept.meshIndexDescriptor;
            copy->shader = kept.shader;
            copy->dataPositions = kept.dataPositions;
            copy->dataSlots = kept.dataSlots;
            copy->flatBinding = kept.flatBinding;
            copy->pushOffset = kept.pushOffset;
            copy->userDataMask = kept.userDataMask;
            copy->vertexInfo = vertexMoved ? std::make_shared<const ShaderRecompiler::ShaderVertexStageInfo>(*vertexInfo) : kept.vertexInfo;
            RelocatedHit found;
            appendEntryRegions(*copy, found.regions);
            auto* liveWords = copy->dataPositions.empty() ? nullptr : &live;
            auto result = validateVariant(program.binary.codeAddress, lookup.queue, *copy, found.regions, imagesFlushed, runsSynced, sampling, liveWords);
            if (gateRetry() && (result == EntryOutcome::PublishMoved || result == EntryOutcome::PendingMoved)) result = validateVariant(program.binary.codeAddress, lookup.queue, *copy, found.regions, imagesFlushed, runsSynced, sampling, liveWords);
            if (result != EntryOutcome::Equal && result != EntryOutcome::EqualData) {
                fallback(result == EntryOutcome::Differing ? RelocatedFallback::Differing : RelocatedFallback::Gate);
                continue;
            }
            if (liveWords != nullptr) {
                for (const auto& [position, value] : live) copy->words[position] = value;
            }
            const auto& forwarded = kept.compiled->shaderDataUserWords;
            for (std::size_t w = 0; w < now.size() && w < 64; ++w) {
                if (now[w] == t.userData[w]) continue;
                for (std::uint32_t slot = 0; slot < forwarded.size(); ++slot) {
                    if (forwarded[slot] == w) copy->shaderDataPatches.emplace_back(slot, now[w]);
                }
            }
            applyShaderDataPatches(*copy);
            if (liveWords != nullptr && !live.empty() && copy->flatBinding != DispatchVariant::NoFlatBinding) copy->compiled = patchDataWords(*copy, copy->words);
            const bool slotsMoved = std::any_of(t.pointerSlots.begin(), t.pointerSlots.end(), [&](const auto& entry) { return copy->words[entry.first] != kept.words[entry.first]; });
            if (slotsMoved || vertexMoved) {
                auto rebuilt = std::make_shared<ShaderRecompiler::RecompileResult>(*copy->compiled);
                if (slotsMoved) {
                    auto& descriptor = rebuilt->bindings[t.flatBinding].guestDescriptor;
                    for (const auto& [position, slot] : t.pointerSlots) descriptor[slot] = copy->words[position];
                }
                if (vertexMoved) {
                    for (auto& attribute : rebuilt->vertexAttributes) {
                        if (attribute.location < vertexInfo->resourcesNum) attribute.resource = vertexInfo->resources[attribute.location];
                    }
                }
                copy->compiled = std::move(rebuilt);
            }
            bool moved = false;
            for (std::size_t r = 0; r < t.runs.size() && !moved; ++r) moved = g.addresses[r] != t.runs[r].address;
            const bool nestedMoved = std::any_of(t.pointers.begin(), t.pointers.end(), [&](std::uint32_t p) { return g.words[p] != kept.words[p]; });
            found.kind = nestedMoved ? RelocatedClass::Nested : liveWords != nullptr && !live.empty() ? RelocatedClass::Data : moved || vertexMoved ? RelocatedClass::Moved : RelocatedClass::Same;
            found.vertexMoved = vertexMoved;
            found.patched = patched;
            found.regions.clear();
            appendEntryRegions(*copy, found.regions);
            found.variant = std::move(copy);
            hit = std::move(found);
            chosen = candidate.held;
            break;
        }
        if (!identified) {
            fallback(RelocatedFallback::ValueWord);
            if (!valueWhy) valueWhy = classifyValueWord(*g.candidates.front().held);
        }
        if (hit || validations >= MaxValidations) break;
    }
    const auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
    std::lock_guard lock(relocatedMutex);
    auto& counters = relocatedCounters;
    counters.validations += validations;
    counters.synced += runsSynced;
    if (hit) {
        ++counters.hits[static_cast<std::size_t>(hit->kind)];
        if (hit->vertexMoved) ++counters.vertexHits;
        if (hit->patched) ++counters.patchedHits;
        counters.hitUs += us;
        const auto found = relocatedPools.find(key);
        if (found != relocatedPools.end()) {
            const auto stamp = ++relocatedClock;
            const auto list = found->second.byIndex.find(Mix(chosen->plan, chosen->index));
            if (list != found->second.byIndex.end()) {
                for (auto& entry : list->second) {
                    if (entry.held == chosen) entry.used = stamp;
                }
            }
            for (auto& entry : found->second.plans) {
                if (entry.plan == chosen->plan) entry.used = stamp;
            }
        }
        return hit;
    }
    ++counters.fallbacks[static_cast<std::size_t>(furthest)];
    if (furthest == RelocatedFallback::ValueWord && valueWhy) ++counters.valueWords[static_cast<std::size_t>(*valueWhy)];
    counters.missUs += us;
    bool learn = furthest == RelocatedFallback::NoTemplate || LearnAfter() <= 1u;
    if (!learn) {
        const auto readable = std::find_if(gathered.begin(), gathered.end(), [](const Gathered& g) { return g.readable; });
        const auto sighting = readable != gathered.end() ? Mix(Mix(Mix(key, readable->exemplar->plan), readable->index), readable->fixed) : Mix(key, IdentityHash(now, ~std::uint64_t{0}));
        if (relocatedSightings.empty()) relocatedSightings.assign(SightingSlots, {0, 0});
        auto& entry = relocatedSightings[sighting & (SightingSlots - 1u)];
        if (entry.first != sighting) entry = {sighting, 0u};
        if (++entry.second >= LearnAfter()) {
            entry.second = 0;
            learn = true;
        } else {
            ++counters.learnDeferred;
        }
    }
    if (lookup.learn != nullptr) *lookup.learn = learn;
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
    const auto why = [&](RelocatedValueWord reason) { return n(c.valueWords[static_cast<std::size_t>(reason)]); };
    std::uint64_t hits = 0, misses = 0;
    for (const auto value : c.hits) hits += value;
    for (const auto value : c.fallbacks) misses += value;
    const auto seconds = std::chrono::duration<double>(now - c.lastReport).count();
    const auto kib = [&](std::size_t bytes) { return relocatedTemplates != 0 ? static_cast<double>(bytes) / 1024.0 / static_cast<double>(relocatedTemplates) : 0.0; };
    char line[2048];
    std::snprintf(line, sizeof(line), "[relocated-hits] %.0f s: %llu stage lookups, %llu hits (R0 same %llu, R1 moved %llu, R2 moved+data %llu, R4 nested %llu; %llu with moved vertex buffers, %llu with patched register words), %.1f us per hit, %.1f us per fallback; fallbacks: no template %llu, decode reads %llu, unreadable %llu, no index match %llu, value word %llu (as value %llu, hidden %llu, base %llu, register %llu, other %llu), layout %llu, pinned base %llu, vertex decode %llu, pointer bits %llu, differing %llu, gate %llu; %llu validations, %llu runs synced; learned %llu (%.1f us each, %llu refused, %llu deferred, %llu replaced, %llu evicted, %llu nested runs, %llu without nested tracking, %llu with vertex buffer words); %zu templates in %zu pools (cap %zu, %.1f KiB each + %.1f KiB compiled); verified %llu (%llu mismatches)\n", seconds, n(c.lookups), n(hits), hit(RelocatedClass::Same), hit(RelocatedClass::Moved), hit(RelocatedClass::Data), hit(RelocatedClass::Nested), n(c.vertexHits), n(c.patchedHits), hits != 0 ? c.hitUs / static_cast<double>(hits) : 0.0, misses != 0 ? c.missUs / static_cast<double>(misses) : 0.0, miss(RelocatedFallback::NoTemplate), miss(RelocatedFallback::DecodeReads), miss(RelocatedFallback::Unreadable), miss(RelocatedFallback::NoIndex), miss(RelocatedFallback::ValueWord), why(RelocatedValueWord::AsValue), why(RelocatedValueWord::Hidden), why(RelocatedValueWord::Base), why(RelocatedValueWord::Register), why(RelocatedValueWord::Other), miss(RelocatedFallback::Layout), miss(RelocatedFallback::PinnedBase), miss(RelocatedFallback::Decode), miss(RelocatedFallback::PointerBits), miss(RelocatedFallback::Differing), miss(RelocatedFallback::Gate), n(c.validations), n(c.synced), n(c.learned), c.learned != 0 ? c.learnUs / static_cast<double>(c.learned) : 0.0, n(c.learnRefused), n(c.learnDeferred), n(c.replaced), n(c.evicted), n(c.nestedRuns), n(c.nestedUntracked), n(c.vertexMovable), relocatedTemplates, relocatedPools.size(), TemplateCap(), kib(relocatedBytes), kib(relocatedCompiledBytes), n(c.verified), n(c.verifyMismatches));
    c = RelocatedCounters{};
    c.lastReport = now;
    lock.unlock();
    AgcDriver::ProfilePrint_nid_no_patch("%s", line);
}

}
