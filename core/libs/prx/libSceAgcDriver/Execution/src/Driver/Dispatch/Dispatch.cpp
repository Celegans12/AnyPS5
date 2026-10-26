#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/LiveSnapshotRetry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/SlowMaterializations.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/BufferCopy.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DwordPatternFill.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <string>
#include <cstdlib>
#include <functional>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace AgcDriver::DriverDetail {

namespace {

// Whether a compiled compute program binds images or samplers (directly, through image tables, or
// the bindless heap): such a dispatch of the async compute ring is recorded into the main lane.
bool DispatchUsesImages(const ShaderRecompiler::RecompileResult& compiled) {
    if (compiled.bindlessHeap) return true;
    using Role = ShaderRecompiler::DescriptorRole;
    for (const auto& binding : compiled.bindings) {
        switch (binding.role) {
            case Role::GuestImages: case Role::GuestSamplers: case Role::ImageTable: case Role::SamplerTable: case Role::ImageTableMap: return true;
            default: break;
        }
    }
    return false;
}

// APS5_PIPELINE_DISPATCH / APS5_PIPELINE_DISPATCH_MISSES are runtime switches (in-run A/B): set when the variable is
// present (any value); a dispatch packet takes both once (Driver::dispatch, and the gate in PacketExecution).
bool DispatchItemsEnabled() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::PipelineDispatch);
}

bool DispatchMissItemsEnabled() {
    return DispatchItemsEnabled() && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::PipelineDispatchMisses);
}

bool NoDispatchItems() {
    static const bool none = std::getenv("APS5_PIPELINE_DISPATCH_NO_ITEMS") != nullptr;
    return none;
}

bool DrainAfterDispatchItems() {
    static const bool drain = std::getenv("APS5_PIPELINE_DISPATCH_ITEM_DRAIN") != nullptr;
    return drain;
}

enum class ItemFallback : std::size_t { Miss, Reads, NoRecipe, Writes, Verify, Count };
constexpr std::array<const char*, static_cast<std::size_t>(ItemFallback::Count)> ItemFallbackNames{"miss", "reads", "no recipe", "writes", "verify"};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ItemFallback::Count)> itemFallbacks{};
std::atomic<std::uint64_t> itemCommits{0};
std::atomic<std::uint64_t> itemRebuilds{0};
std::atomic<std::uint64_t> missItemCommits{0};
std::atomic<std::uint64_t> missItemRefusals{0};

void NoteItemFallback(ItemFallback reason) {
    itemFallbacks[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

enum class DrainSite : std::size_t { Ahead, ItemsOff, Indirect, Labels, CopyKernel, HtileDepth, HtileBuffer, LookupMiss, LookupVerify, LookupReads, NoRecipe, ItemWrites, Count };
constexpr std::array<const char*, static_cast<std::size_t>(DrainSite::Count)> DrainSiteNames{"ahead", "items off", "indirect", "labels", "copy kernel", "htile depth", "htile buffer", "miss", "verify", "reads", "no recipe", "item writes"};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrainSite::Count)> drainSiteCounts{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrainSite::Count)> drainSiteNs{};
std::atomic<std::uint64_t> profiledAttempts{0}, packetToAttemptNs{0}, attemptToKeyNs{0};

void DrainAt(DrainSite site, bool profile) {
    auto& pipeline = DrawPipeline::Queue0();
    if (!profile) {
        pipeline.Drain(DrawPipeline::DrainReason::Packet, 0x15);
        return;
    }
    const bool busy = pipeline.Busy();
    const auto start = std::chrono::steady_clock::now();
    pipeline.Drain(DrawPipeline::DrainReason::Packet, 0x15);
    if (!busy) return;
    const auto index = static_cast<std::size_t>(site);
    drainSiteCounts[index].fetch_add(1, std::memory_order_relaxed);
    drainSiteNs[index].fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
}

void ReportDispatchFast(bool profile) {
    static std::atomic<std::int64_t> lastReport{0};
    if (!profile) return;
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = lastReport.load(std::memory_order_relaxed);
    if (now - last < 10000 || !lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    std::string drains;
    std::uint64_t decodeDrainNs = 0;
    for (std::size_t i = 0; i < DrainSiteNames.size(); ++i) {
        const auto count = drainSiteCounts[i].exchange(0, std::memory_order_relaxed);
        const auto ns = drainSiteNs[i].exchange(0, std::memory_order_relaxed);
        if (i <= static_cast<std::size_t>(DrainSite::HtileBuffer)) decodeDrainNs += ns;
        if (count == 0) continue;
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu/%.1f", DrainSiteNames[i], static_cast<unsigned long long>(count), static_cast<double>(ns) / 1e6);
        drains += item;
    }
    if (drains.empty()) drains = " none";
    const auto attempts = profiledAttempts.exchange(0, std::memory_order_relaxed);
    const auto entry = packetToAttemptNs.exchange(0, std::memory_order_relaxed);
    const auto decode = attemptToKeyNs.exchange(0, std::memory_order_relaxed);
    const auto perAttempt = [&](std::uint64_t ns) { return attempts != 0 ? static_cast<double>(ns) / 1e3 / static_cast<double>(attempts) : 0.0; };
    std::fprintf(stderr, "[dispatch-fast] (10 s) %llu dispatch attempts; us per attempt: packet start to attempt %.1f, attempt to key %.1f (of it decode-phase drains %.1f); queue-0 dispatch drains by site (count/ms):%s\n", static_cast<unsigned long long>(attempts), perAttempt(entry), perAttempt(decode), perAttempt(decodeDrainNs), drains.c_str());
}

void ReportDispatchItems(bool profile) {
    static auto lastReport = std::chrono::steady_clock::now();
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    std::string drained;
    for (std::size_t i = 0; i < ItemFallbackNames.size(); ++i) {
        char item[48];
        std::snprintf(item, sizeof(item), " %s %llu", ItemFallbackNames[i], static_cast<unsigned long long>(itemFallbacks[i].load(std::memory_order_relaxed)));
        drained += item;
    }
    std::fprintf(stderr, "[dispatch-item] %llu commit items (%llu recipes rebuilt on the committer), %llu miss items (%llu refused on the committer); drained instead, cumulative:%s\n", static_cast<unsigned long long>(itemCommits.load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemRebuilds.load(std::memory_order_relaxed)), static_cast<unsigned long long>(missItemCommits.load(std::memory_order_relaxed)), static_cast<unsigned long long>(missItemRefusals.load(std::memory_order_relaxed)), drained.c_str());
}

bool DispatchSlotsEnabled() {
    return !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoDispatchSlots);
}

bool VerifyDispatchSlots() {
    static const bool verify = std::getenv("APS5_VERIFY_DISPATCH_SLOTS") != nullptr;
    return verify;
}

bool ProfileDispatchSlots() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr || std::getenv("APS5_PROFILE_DISPATCH_SLOTS") != nullptr || VerifyDispatchSlots();
    return profile;
}

enum class SlotKind : std::size_t { HitWithRecipe, Hit, Miss, Count };
constexpr std::array<const char*, static_cast<std::size_t>(SlotKind::Count)> SlotKindNames{"hits with recipe", "hits without recipe", "misses"};
enum class AheadOutcome : std::size_t { Adopted, NotStarted, Failed, ForgetSerial, Registry, NoStageA, KeyChanged, CacheEntry, RecipeAtCommit, VerifyMismatch, Count };
constexpr std::array<const char*, static_cast<std::size_t>(AheadOutcome::Count)> AheadOutcomeNames{"adopted", "not started", "made nothing", "forget serial", "registry", "cache object at prepare", "key changed", "cache entry since", "recipe at commit", "verify mismatch"};
enum class AheadRefusal : std::size_t { Switch, Recipe, Htile, Bindings, PendingWrites, Count };
constexpr std::array<const char*, static_cast<std::size_t>(AheadRefusal::Count)> AheadRefusalNames{"switch", "recipe", "htile", "bindings", "pending writes"};

struct SlotCounterSet {
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(SlotKind::Count)> kinds{};
    std::atomic<std::uint64_t> htileNotes{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrainSite::Count)> avoided{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrainSite::Count)> fallbacks{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(AheadOutcome::Count)> ahead{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(AheadRefusal::Count)> notAhead{};
    std::atomic<std::uint64_t> preparedAhead{0};
    std::atomic<std::uint64_t> prepareNs{0};
    std::atomic<std::uint64_t> verified{0};
    std::atomic<std::uint64_t> verifyMismatches{0};
    std::atomic<std::uint64_t> commits{0};
    std::atomic<std::uint64_t> commitNs{0};
    std::atomic<std::int64_t> lastReport{0};
};

SlotCounterSet& SlotCounters() {
    static auto* counters = new SlotCounterSet();
    return *counters;
}

void NoteSlotFallback(DrainSite site) {
    SlotCounters().fallbacks[static_cast<std::size_t>(site)].fetch_add(1, std::memory_order_relaxed);
}

void ReportDispatchSlots() {
    if (!ProfileDispatchSlots()) return;
    auto& c = SlotCounters();
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = c.lastReport.load(std::memory_order_relaxed);
    if (last == 0) {
        c.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10000 || !c.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    static std::array<std::uint64_t, 64> previous{};
    std::size_t slot = 0;
    const auto delta = [&](std::atomic<std::uint64_t>& value) {
        const auto current = value.load(std::memory_order_relaxed);
        const auto change = current - previous[slot];
        previous[slot++] = current;
        return static_cast<unsigned long long>(change);
    };
    const auto list = [&](auto& values, const auto& names) {
        std::string text;
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto count = delta(values[i]);
            if (count == 0) continue;
            char item[64];
            std::snprintf(item, sizeof(item), " %s %llu", names[i], count);
            text += item;
        }
        return text.empty() ? std::string(" none") : text;
    };
    const auto kinds = list(c.kinds, SlotKindNames);
    const auto htile = delta(c.htileNotes);
    const auto avoided = list(c.avoided, DrainSiteNames);
    const auto fallbacks = list(c.fallbacks, DrainSiteNames);
    const auto prepared = delta(c.preparedAhead);
    const auto prepareMs = static_cast<double>(delta(c.prepareNs)) / 1e6;
    const auto outcomes = list(c.ahead, AheadOutcomeNames);
    const auto refusals = list(c.notAhead, AheadRefusalNames);
    const auto commits = delta(c.commits);
    const auto commitUs = commits != 0 ? static_cast<double>(delta(c.commitNs)) / 1e3 / static_cast<double>(commits) : (static_cast<void>(delta(c.commitNs)), 0.0);
    std::fprintf(stderr, "[dispatch-slots] (10 s) %llu queue-0 dispatch slots (by kind:%s; HTILE notes made in order %llu), %.1f us per slot on the committer; drains avoided by today's site:%s; fallbacks to the drained path by site:%s; stage A ahead: %llu prepared (%.1f ms), at the commit:%s; not prepared ahead:%s; verify (cumulative): %llu checked, %llu differing\n", commits, kinds.c_str(), htile, commitUs, avoided.c_str(), fallbacks.c_str(), prepared, prepareMs, outcomes.c_str(), refusals.c_str(), static_cast<unsigned long long>(c.verified.load(std::memory_order_relaxed)), static_cast<unsigned long long>(c.verifyMismatches.load(std::memory_order_relaxed)));
}

struct HtileMaskClear {
    std::uint64_t address = 0;
    std::size_t bytes = 0;
    std::uint32_t mask = 0;
    std::uint32_t value = 0;
};

struct HtileNote {
    std::uint64_t address = 0;
    std::size_t bytes = 0;
    std::size_t nativeBytes = 0;
};

struct SlotPreparation {
    std::shared_ptr<PreparedDispatch> prepared;
    std::uint64_t forgetSerial = 0;
    std::uint64_t registryGeneration = 0;
    std::uint64_t inputHash = 0;
};

bool AheadBindings(const ShaderRecompiler::RecompileResult& compiled) {
    if (compiled.bindlessHeap || compiled.bdaAbiVersion != 0) return false;
    using Role = ShaderRecompiler::DescriptorRole;
    return std::ranges::all_of(compiled.bindings, [](const auto& binding) {
        switch (binding.role) {
            case Role::GuestBuffers: case Role::GuestImages: case Role::GuestSamplers: case Role::FlattenedSrt: case Role::ShaderData: case Role::ImageTable: case Role::SamplerTable: case Role::ImageTableMap: return true;
            default: return false;
        }
    });
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> SlotInputRanges(const ShaderRecompiler::RecompileResult& compiled, std::span<const ShaderRecompiler::MemoryRegion> captured) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (const auto& binding : compiled.bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers || binding.guestDescriptor.size() < static_cast<std::size_t>(binding.count) * 4u) continue;
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u;
            const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
            const auto base = descriptor.Base48();
            const auto bytes = descriptor.GetSize();
            if (base != 0 && bytes != 0) ranges.emplace_back(base, base + bytes);
        }
    }
    for (const auto& region : captured) {
        if (!region.bytes.empty()) ranges.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
    }
    return ranges;
}

std::uint64_t SlotInputHash(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    constexpr std::uint64_t Limit = 8ull * 1024 * 1024;
    std::uint64_t hash = 0xcbf29ce484222325ull;
    std::uint64_t total = 0;
    for (const auto& [begin, end] : ranges) {
        hash = (hash ^ begin) * 0x100000001b3ull;
        hash = (hash ^ end) * 0x100000001b3ull;
        const auto bytes = end - begin;
        if (total + bytes > Limit || !GuestMemory::Accessible(reinterpret_cast<const void*>(begin), static_cast<std::size_t>(bytes))) continue;
        total += bytes;
        const auto* data = reinterpret_cast<const unsigned char*>(begin);
        for (std::uint64_t i = 0; i < bytes; ++i) hash = (hash ^ data[i]) * 0x100000001b3ull;
    }
    return hash;
}

struct SlotOrigin {
    SlotOrigin(std::uint64_t program, bool kernel, std::uint32_t mask, std::uint32_t value) {
        Graphics::Recorder::SetWriteOrigin(program);
        if (kernel) Graphics::Recorder::SetWriteOriginKernel(mask, value);
    }
    ~SlotOrigin() { Graphics::Recorder::SetWriteOrigin(0); }
    SlotOrigin(const SlotOrigin&) = delete;
    SlotOrigin& operator=(const SlotOrigin&) = delete;
};

}

struct DispatchSlot {
    std::shared_ptr<VulkanDevice> device;
    std::shared_ptr<const ShaderSnapshot> shader;
    std::size_t codeOffset = 0;
    std::vector<std::uint32_t> userData;
    ShaderRecompiler::ShaderComputeStageInfo compute{};
    std::array<std::uint32_t, 5> packet{};
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
    std::shared_ptr<DispatchVariant> keepVariant;
    std::shared_ptr<DispatchVariant> attachVariant;
    std::shared_ptr<ShaderMemory> shaderMemory;
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;
    std::vector<ShaderRecompiler::MemoryRegion> captured;
    std::vector<std::uint32_t> liveWords;
    std::vector<Graphics::GuestMemorySnapshot> snapshots;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> inputRanges;
    std::array<std::uint32_t, 3> groups{};
    std::uint64_t address = 0;
    std::uint64_t dispatchThreads = 0;
    std::uint64_t epoch = 0;
    std::uint32_t queue = 0;
    bool cached = false;
    bool dataHit = false;
    bool partialLastGroup = false;
    bool htileKernel = false;
    std::optional<std::pair<std::uint64_t, std::size_t>> htileClearCompletion;
    HtileMaskClear htileMask;
    std::optional<HtileNote> htileNote;
    std::shared_ptr<DrawPipeline::PrepareSlot> prepareSlot;
};

namespace {

}

bool Driver::pipelineDispatch() {
    return DispatchItemsEnabled() || DispatchSlotsEnabled();
}

Driver::DispatchSlotCounts Driver::DispatchSlotStatistics() {
    auto& c = SlotCounters();
    DispatchSlotCounts counts;
    counts.hitsWithRecipe = c.kinds[static_cast<std::size_t>(SlotKind::HitWithRecipe)].load(std::memory_order_relaxed);
    counts.hits = c.kinds[static_cast<std::size_t>(SlotKind::Hit)].load(std::memory_order_relaxed);
    counts.misses = c.kinds[static_cast<std::size_t>(SlotKind::Miss)].load(std::memory_order_relaxed);
    counts.slots = counts.hitsWithRecipe + counts.hits + counts.misses;
    counts.htileNotes = c.htileNotes.load(std::memory_order_relaxed);
    for (const auto& value : c.avoided) counts.drainsAvoided += value.load(std::memory_order_relaxed);
    for (const auto& value : c.fallbacks) counts.fallbacks += value.load(std::memory_order_relaxed);
    counts.preparedAhead = c.preparedAhead.load(std::memory_order_relaxed);
    counts.adopted = c.ahead[static_cast<std::size_t>(AheadOutcome::Adopted)].load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < c.ahead.size(); ++i) {
        if (i != static_cast<std::size_t>(AheadOutcome::Adopted)) counts.discarded += c.ahead[i].load(std::memory_order_relaxed);
    }
    counts.verified = c.verified.load(std::memory_order_relaxed);
    counts.verifyMismatches = c.verifyMismatches.load(std::memory_order_relaxed);
    return counts;
}

void Driver::commitDispatchSlot(DispatchSlot& slot) {
    auto& counters = SlotCounters();
    if (slot.htileNote.has_value()) {
        if (Graphics::NoteHtileDepthClear(slot.htileNote->address, slot.htileNote->bytes, slot.htileNote->nativeBytes)) slot.htileClearCompletion = std::pair{slot.htileNote->address, slot.htileNote->bytes};
        counters.htileNotes.fetch_add(1, std::memory_order_relaxed);
    }
    const auto& compiled = *slot.compiled;
    const auto code = std::span<const std::uint32_t>(slot.shader->code).subspan(slot.codeOffset);
    std::shared_ptr<RecipeHit> recipeHit;
    if (slot.cached && slot.keepVariant != nullptr && !stampValidate()) recipeHit = slot.htileClearCompletion.has_value() ? nullptr : slot.device->PrepareRecipe(slot.keepVariant->recipe.load(std::memory_order_acquire), false);
    std::shared_ptr<const Recipe> builtRecipe;
    auto* const attachTo = stampValidate() ? nullptr : slot.keepVariant != nullptr ? slot.keepVariant.get() : slot.attachVariant.get();
    bool writersNoted = false;
    const bool noteWrites = writeEvidenceEnabled() || traceCapSync();
    const auto keysFill = !slot.partialLastGroup ? MatchSgprPatternFill(slot.packet, code, slot.userData, slot.compute) : std::nullopt;
    const auto filledKeys = keysFill ? KeysFillMemo(keysFill->pattern, false, true) : std::nullopt;
    std::optional<Graphics::DepthPlaneClear> depthPlaneClear;
    if (!slot.partialLastGroup) {
        if (const auto clear = Graphics::MatchStorageImageClear(code, slot.userData, slot.compute, slot.groups)) {
            const auto clearAddress = Graphics::StorageImageClearAddress(*clear);
            if (clearAddress != 0 && Graphics::DepthSurfaceAt(clearAddress)) {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
                std::lock_guard clearLock(GuestMemory::GpuMutex());
                recordLabelsForPacket(slot.device.get(), slot.queue);
                depthPlaneClear.emplace(*clear);
            }
        }
    }
    const auto adoptAhead = [&](bool wanted) -> std::shared_ptr<PreparedDispatch> {
        if (slot.prepareSlot == nullptr) return nullptr;
        const auto taken = std::static_pointer_cast<SlotPreparation>(slot.prepareSlot->Take());
        auto outcome = AheadOutcome::Adopted;
        if (taken == nullptr) outcome = AheadOutcome::NotStarted;
        else if (!wanted) outcome = AheadOutcome::RecipeAtCommit;
        else if (taken->prepared == nullptr) outcome = (taken->forgetSerial & 1u) != 0 ? AheadOutcome::ForgetSerial : AheadOutcome::Failed;
        else if (GuestMemory::ForgetSerial() != taken->forgetSerial) outcome = AheadOutcome::ForgetSerial;
        else if (GuestAllocations::GuestAllocationsGeneration_nid_postfix() != taken->registryGeneration) outcome = AheadOutcome::Registry;
        else {
            switch (slot.device->AdoptPreparedDispatch(*taken->prepared, compiled)) {
                case VulkanDevice::PreparedAdoption::Adopted: break;
                case VulkanDevice::PreparedAdoption::NoStageA: outcome = AheadOutcome::NoStageA; break;
                case VulkanDevice::PreparedAdoption::KeyChanged: outcome = AheadOutcome::KeyChanged; break;
                default: outcome = AheadOutcome::CacheEntry; break;
            }
        }
        if (outcome == AheadOutcome::Adopted && VerifyDispatchSlots()) {
            counters.verified.fetch_add(1, std::memory_order_relaxed);
            std::string difference;
            if (taken->inputHash != SlotInputHash(slot.inputRanges)) {
                difference = "the bytes the preparation read changed before the commit";
            } else {
                const auto inOrder = slot.device->PrepareDispatch(compiled, slot.snapshots, slot.dispatchThreads, slot.address, true);
                if (inOrder == nullptr) difference = "the in-order preparation made nothing";
                else static_cast<void>(VulkanDevice::SamePreparation(*taken->prepared, *inOrder, difference));
            }
            if (!difference.empty()) {
                counters.verifyMismatches.fetch_add(1, std::memory_order_relaxed);
                std::fprintf(stderr, "[dispatch-slots] verify: compute shader 0x%llx: %s; prepared again in order\n", static_cast<unsigned long long>(slot.address), difference.c_str());
                outcome = AheadOutcome::VerifyMismatch;
            }
        }
        counters.ahead[static_cast<std::size_t>(outcome)].fetch_add(1, std::memory_order_relaxed);
        return outcome == AheadOutcome::Adopted ? taken->prepared : nullptr;
    };
    bool first = true;
    for (;;) {
        std::shared_ptr<PreparedDispatch> prepared;
        const bool wantPrepared = recipeHit == nullptr || VulkanDevice::VerifyRecipes();
        if (first) prepared = adoptAhead(wantPrepared);
        first = false;
        if (wantPrepared && prepared == nullptr) prepared = slot.device->PrepareDispatch(compiled, slot.snapshots, slot.dispatchThreads, slot.address);
        if (!wantPrepared) prepared.reset();
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        std::optional<Graphics::Recorder::LaneSwitch> laneSwitch;
        if (Graphics::Recorder::ThreadLane() != 0 && DispatchUsesImages(compiled)) laneSwitch.emplace(Graphics::Recorder::Active(), 0);
        recordLabelsForPacket(slot.device.get(), slot.queue);
        if (noteWrites && writerKeyedEvidence() && !writersNoted) {
            noteWrittenBuffers(slot.address, slot.queue, compiled);
            writersNoted = true;
        }
        auto* const keysRecorder = filledKeys ? Graphics::Recorder::Active() : nullptr;
        const auto keysNoteBefore = keysRecorder != nullptr ? keysRecorder->LastWriteNote(keysFill->base, keysFill->bytes) : 0;
        if (recipeHit != nullptr) {
            VulkanDevice::IndirectOutcome outcome{0, 0};
            const auto result = slot.device->DispatchRecipe(compiled, slot.groups[0], slot.groups[1], slot.groups[2], 0, slot.address, recipeHit, outcome, VulkanDevice::VerifyRecipes() ? prepared : nullptr, slot.dataHit, {});
            if (result == RecipeOutcome::Rebuild) {
                recipeHit = nullptr;
                VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, false);
                continue;
            }
        } else {
            slot.device->Dispatch(compiled, slot.groups[0], slot.groups[1], slot.groups[2], slot.snapshots, slot.address, std::move(prepared), slot.htileClearCompletion.has_value() ? nullptr : (attachTo != nullptr ? &builtRecipe : nullptr), slot.htileClearCompletion);
        }
        if (slot.htileMask.address != 0) slot.device->NoteHtileMask(slot.htileMask.address, slot.htileMask.bytes, slot.htileMask.mask, slot.htileMask.value);
        if (keysRecorder != nullptr && keysRecorder == Graphics::Recorder::Active()) {
            const auto note = keysRecorder->LastWriteNote(keysFill->base, keysFill->bytes);
            if (note != 0 && note != keysNoteBefore) Graphics::NoteKeysFillOnGpu(keysFill->base, keysFill->bytes, *filledKeys);
        }
        if (builtRecipe != nullptr) {
            if (slot.dataHit) {
                auto own = std::make_shared<Recipe>(*builtRecipe);
                own->dataWordsHash = Graphics::ShaderResources::DataWordsHash({ShaderRecompiler::ShaderStage::Compute, slot.keepVariant->compiled.get(), 0});
                builtRecipe = std::move(own);
            }
            attachTo->recipe.store(std::move(builtRecipe), std::memory_order_release);
            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, false);
        }
        break;
    }
    if (noteWrites && !writerKeyedEvidence()) noteWrittenBuffers(slot.address, slot.queue, compiled);
}

void Driver::dispatch(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::uint64_t indirectArguments) {
    static const bool traceMismatch = std::getenv("APS5_TRACE_SNAPSHOT_MISMATCH") != nullptr;
    const auto initialGpuDepth = traceMismatch ? GuestMemory::GpuMutex().DepthOnThisThread() : 0u;
    const bool initialCompletion = traceMismatch && Graphics::Recorder::InCompletion();
    try {
        RetryLiveSnapshot([&](bool forceFresh) {
            try {
                dispatchAttempt(queue, packet, submission, indirectArguments, forceFresh);
            } catch (const Graphics::LiveSnapshotInvalidated&) {
                [&]() noexcept {
                    if (!traceMismatch) return;
                    static std::atomic<std::uint64_t> records{0};
                    const auto record = records.fetch_add(1, std::memory_order_relaxed);
                    if (record >= 32u) return;
                    const auto low = queue.shader.find(0x20c);
                    const auto high = queue.shader.find(0x20d);
                    const bool addressKnown = low != queue.shader.end() && high != queue.shader.end();
                    const auto address = addressKnown ? (static_cast<std::uint64_t>(low->second) << 8u) | (static_cast<std::uint64_t>(high->second & 0xffu) << 40u) : 0ull;
                    const auto thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
                    std::fprintf(stderr, "[snapshot-mismatch-dispatch] record=%llu thread=%zu shader=0x%llx shader_known=%u queue=%u serial=%llu packet=%zu attempt=%u force_fresh=%u initial_eligible=%u initial_gpu_depth=%u initial_completion=%u unwound_gpu_depth=%u unwound_completion=%u failure=LiveSnapshotInvalidated action=rethrow\n",
                        static_cast<unsigned long long>(record + 1u), thread, static_cast<unsigned long long>(address), addressKnown ? 1u : 0u, submission.queue, static_cast<unsigned long long>(submission.serial), currentPacketOffset(), forceFresh ? 1u : 0u, forceFresh ? 1u : 0u, initialGpuDepth == 0u && !initialCompletion ? 1u : 0u, initialGpuDepth, initialCompletion ? 1u : 0u, GuestMemory::GpuMutex().DepthOnThisThread(), Graphics::Recorder::InCompletion() ? 1u : 0u);
                }();
                throw;
            }
        });
    } catch (const Graphics::LiveSnapshotInvalidated& error) {
        const auto address = (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20c)) << 8u) | (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20d) & 0xffu) << 40u);
        char where[64];
        std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
        throw std::runtime_error(where + std::string(error.what()));
    }
}

void Driver::dispatchAttempt(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::uint64_t indirectArguments, bool forceFresh) {
    PerformanceTimer timing("Driver.Dispatch");
    static const bool profileDrains = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto attemptStart = profileDrains ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // Both switches once per attempt: an A/B flip (at a queue-0 present) never splits one.
    const bool slotsOn = DispatchSlotsEnabled();
    const bool legacyItems = DispatchItemsEnabled();
    const bool legacyMissItems = legacyItems && DispatchMissItemsEnabled();
    const bool itemsOn = slotsOn || legacyItems;
    const bool missItemsOn = slotsOn || legacyMissItems;
    const auto address = (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20c)) << 8u) | (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20d) & 0xffu) << 40u);
    if (address == SkippedShader() && address != 0) return;
    struct OriginScope {
        explicit OriginScope(std::uint64_t program) { Graphics::Recorder::SetWriteOrigin(program); }
        ~OriginScope() { Graphics::Recorder::SetWriteOrigin(0); }
    } originScope(address);
    auto registeredShader = RegisteredProgram(*submission.shaders, address, {0});
    if (!registeredShader) registeredShader = ReadRawShader(address);
    const auto& snapshot = *registeredShader;
    require(snapshot.type == 0, "compute program refers to a non-compute shader");
    const auto userCount = (readRegister(queue.shader, 0x213) >> 1u) & 0x1fu;
    std::vector<std::uint32_t> userData;
    for (std::uint32_t i = 0; i < userCount; ++i) {
        userData.push_back(readUserData(queue.shader, 0x240 + i));
    }
    auto compute = Graphics::DecodeComputeStageInfo(queue.shader, snapshot.header);
    const bool resolving = resolvingAhead();
    if (SkipRefusedShaders() && SkipListedShader(address)) {
        if (resolving) return;
        char where[96];
        std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
        throw ShaderRefusal(std::string(where) + "listed in the diagnostic skip list");
    }
    std::vector<ShaderRecompiler::MemoryRegion> memory{{snapshot.codeAddress, std::as_bytes(std::span(snapshot.code))}};
    if (!snapshot.header.empty()) memory.push_back({snapshot.headerAddress, snapshot.header});

    static const bool unlockedDevice = std::getenv("APS5_NO_UNLOCKED_DEVICE") == nullptr;
    std::shared_ptr<VulkanDevice> localDevice = unlockedDevice ? device.Load() : nullptr;
    if (localDevice == nullptr) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    const auto codeOffset = static_cast<std::size_t>((address - snapshot.codeAddress) / sizeof(std::uint32_t));
    if (userData.size() >= 6 && Graphics::HtileMaskKernelCode(std::span(snapshot.code).subspan(codeOffset))) Graphics::Recorder::SetWriteOriginKernel(userData[5], userData[4]);
    static const bool profileWriters = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (profileWriters && Graphics::HtileWriterProgram(address)) {
        static std::atomic<int> dumped{0};
        if (dumped.fetch_add(1) < 6) {
            std::string text;
            char item[24];
            const auto code = std::span(snapshot.code).subspan(codeOffset);
            for (std::size_t i = 0; i < code.size() && i < 160; ++i) {
                std::snprintf(item, sizeof(item), " %08x", code[i]);
                text += item;
            }
            std::string user;
            for (const auto word : userData) {
                std::snprintf(item, sizeof(item), " %08x", word);
                user += item;
            }
            std::fprintf(stderr, "[htile-writer] queue 0x%x program 0x%llx packet %08x %08x %08x %08x threads %u %u %u user data:%s; code (%zu words):%s\n", submission.queue, static_cast<unsigned long long>(address), packet.size() > 1 ? packet[1] : 0u, packet.size() > 2 ? packet[2] : 0u, packet.size() > 3 ? packet[3] : 0u, packet.size() > 4 ? packet[4] : 0u, compute.numThreads[0], compute.numThreads[1], compute.numThreads[2], user.c_str(), code.size(), text.c_str());
        }
    }
    std::array<std::uint32_t, 5> resolved{};
    ResolvedDispatch ahead;
    bool fromAhead = false;
    auto& groupStats = groupCaptureStats();
    if (!resolving) {
        auto& parked = resolvedAhead();
        if (const auto found = parked.find(currentPacketOffset()); found != parked.end()) {
            ahead = std::move(found->second);
            parked.erase(found);
            fromAhead = !forceFresh && ahead.address == address;
            if (ahead.address != address) ++groupStats.addressMismatches;
        }
    }
    if (fromAhead) {
        resolved = ahead.packet;
        packet = resolved;
        indirectArguments = ahead.indirectArguments;
    } else if (indirectArguments != 0 && !resolving && matchesFillKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute)) {
        indirectSource().path = IndirectFillKernel;
        const auto readStart = std::chrono::steady_clock::now();
        resolved = readDispatchArguments(submission.queue, indirectArguments, packet[4]);
        countIndirect(IndirectFillKernel, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        packet = resolved;
        indirectArguments = 0;
    }
    std::array<std::uint32_t, 3> partialGroups{};
    if (indirectArguments == 0 && (packet[4] & 0x2u) != 0) {
        partialGroups = Pm4::PartialGroupThreads({packet[1], packet[2], packet[3]}, {readRegister(queue.shader, 0x207), readRegister(queue.shader, 0x208), readRegister(queue.shader, 0x209)});
    }
    const bool pipelineCandidate = !resolving && !fromAhead && itemsOn && DrawPipeline::Active() && indirectArguments == 0 && deferredLabels().labels.empty() && !matchesCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) && !MatchesStreamCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) && !MatchesIndexedCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute);
    const bool partialLastGroup = partialGroups != std::array<std::uint32_t, 3>{};
    HtileMaskClear htileMask;
    std::optional<std::pair<std::uint64_t, std::size_t>> htileClearCompletion;
    const bool slotHtile = slotsOn && pipelineCandidate && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoDispatchSlotHtile);
    std::optional<HtileNote> deferredHtile;
    bool busyAtHtile = false;
    auto htileSite = DrainSite::Count;
    const auto settleHtile = [&] {
        if (!deferredHtile.has_value()) return;
        if (Graphics::NoteHtileDepthClear(deferredHtile->address, deferredHtile->bytes, deferredHtile->nativeBytes)) htileClearCompletion = std::pair{deferredHtile->address, deferredHtile->bytes};
        deferredHtile.reset();
    };
    const auto fallBack = [&](DrainSite site) {
        if (slotsOn) NoteSlotFallback(site);
        if (DrawPipeline::Queue0().Busy()) DrainAt(site, profileDrains);
        settleHtile();
    };
    if (resolving) {
        if (indirectArguments == 0 && Graphics::MatchMultisampledSingleClear(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads)) return;
        if (!partialLastGroup && matchesFillKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute)) return;
        if (!partialLastGroup && indirectArguments == 0 && (matchesCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchesStreamCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchesIndexedCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute))) return;
    } else {
        if (indirectArguments == 0 && Graphics::TakeMultisampledSingleClear(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads)) {
            pendingDispatchPhases().outcome = DispatchOutcome::FillHle;
            return;
        }
        if (indirectArguments == 0 && (packet[4] & 0x20u) == 0) {
            if (const auto fill = Graphics::MatchMemoryValueFill(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads, {packet[1], packet[2], packet[3]})) Graphics::NoteMemoryValueFill(*fill);
        }
        if (!partialLastGroup && fillBuffer(queue, submission.queue, packet, std::span(snapshot.code).subspan(codeOffset), userData, compute, localDevice)) {
            pendingDispatchPhases().outcome = DispatchOutcome::FillHle;
            return;
        }
        if (DrawPipeline::Active() && !pipelineCandidate) {
            const auto site = fromAhead ? DrainSite::Ahead : !itemsOn ? DrainSite::ItemsOff : indirectArguments != 0 ? DrainSite::Indirect : !deferredLabels().labels.empty() ? DrainSite::Labels : DrainSite::CopyKernel;
            if (slotsOn) NoteSlotFallback(site);
            DrainAt(site, profileDrains);
        }
        if (const auto htile = Graphics::HtileDepthClearAddress(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads); htile != 0) {
            if (slotHtile) {
                busyAtHtile = DrawPipeline::Queue0().Busy();
                htileSite = DrainSite::HtileDepth;
            } else if (pipelineCandidate && DrawPipeline::Queue0().Busy()) {
                DrainAt(DrainSite::HtileDepth, profileDrains);
            }
            const auto threads = (packet[4] & 0x20u) != 0 ? static_cast<std::uint64_t>(packet[1]) : static_cast<std::uint64_t>(packet[1]) * compute.numThreads[0];
            const auto nativeBytes = Graphics::HtileDepthClearBytes_nid_no_patch(std::span(snapshot.code).subspan(codeOffset), userData, compute, packet, indirectArguments);
            const auto bytes = static_cast<std::size_t>(threads * 16u);
            if (slotHtile) deferredHtile = HtileNote{htile, bytes, nativeBytes};
            else if (Graphics::NoteHtileDepthClear(htile, bytes, nativeBytes)) htileClearCompletion = std::pair{htile, bytes};
            if (nativeBytes != 0u && userData.size() >= 6u) htileMask = {htile, nativeBytes, userData[5], userData[4]};
        } else if (const auto buffered = Graphics::MatchHtileBufferMaskClear(std::span(snapshot.code).subspan(codeOffset), userData, compute, packet, indirectArguments); buffered.has_value() && (buffered->mask & 0xfu) == 0u && (buffered->value & 0xfu) == 0u) {
            // S1: the buffer variant of the mask kernel, read the same way (zmask cleared: a depth clear).
            if (slotHtile) {
                busyAtHtile = DrawPipeline::Queue0().Busy();
                htileSite = DrainSite::HtileBuffer;
                deferredHtile = HtileNote{buffered->address, buffered->bytes, buffered->bytes};
            } else {
                if (pipelineCandidate && DrawPipeline::Queue0().Busy()) DrainAt(DrainSite::HtileBuffer, profileDrains);
                if (Graphics::NoteHtileDepthClear(buffered->address, buffered->bytes, buffered->bytes)) htileClearCompletion = std::pair{buffered->address, buffered->bytes};
            }
            htileMask = {buffered->address, buffered->bytes, buffered->mask, buffered->value};
        }
        if (!partialLastGroup && indirectArguments == 0 && copyBuffer(queue, submission.queue, packet, std::span(snapshot.code).subspan(codeOffset), userData, compute, localDevice, address)) {
            pendingDispatchPhases().outcome = DispatchOutcome::CopyHle;
            return;
        }
    }
    if (indirectArguments == 0 && (packet[4] & 0x20u) != 0) {
        const std::array<std::uint32_t, 3> threads{packet[1], packet[2], packet[3]};
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            if (threads[axis] % compute.numThreads[axis] != 0) compute.partialThreads = threads;
        }
    }
    if (partialLastGroup) compute.partialThreads = partialGroups;
    // One workgroup (direct, 1x1x1 groups, or a thread count inside one group): workgroup-scope
    // coherence for its glc reads (ShaderComputeStageInfo::singleWorkgroup). Opt-in, APS5_SINGLE_GROUP_COHERENCE=1:
    // with the BDA overlays (device-local buffers) it saves ~2 % of 0x80001c0d00 on the request bench, and one
    // of four bench runs gave an unstable output hash, so it is not shown to be exact.
    static const bool singleGroupCoherence = std::getenv("APS5_SINGLE_GROUP_COHERENCE") != nullptr;
    if (singleGroupCoherence && indirectArguments == 0) {
        const bool threads = (packet[4] & 0x20u) != 0;
        bool single = true;
        for (std::uint32_t axis = 0; axis < 3; ++axis) single = single && (threads ? packet[1 + axis] <= compute.numThreads[axis] : packet[1 + axis] == 1u);
        compute.singleWorkgroup = single && packet[1] != 0 && packet[2] != 0 && packet[3] != 0;
    }
    std::array<std::uint32_t, 3> threadGroup{};
    if (indirectArguments != 0 && (packet[4] & 0x20u) != 0) {
        for (std::uint32_t axis = 0; axis < 3; ++axis) threadGroup[axis] = std::max(readRegister(queue.shader, 0x207 + axis) & 0xffffu, 1u);
        compute.partialThreads = {~0u, ~0u, ~0u};
    }
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, address, std::span(snapshot.code).subspan(codeOffset), snapshot.headerAddress, snapshot.header},
        {(packet[4] & 0x8000u) != 0 ? 32u : 64u, 0, userData, compute, std::nullopt, std::nullopt, memory, DispatchFloatMode(snapshot, queue.shader)},
        localDevice->ComputeTarget((packet[4] & 0x8000u) != 0 ? 32u : 64u),
        {0, 0, 0, threadGroup != std::array<std::uint32_t, 3>{} ? 0u : 128u}
    };
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static double captureMs = 0, keyMs = 0, recompileMs = 0, deviceMs = 0;
    static std::uint64_t cacheHits = 0;
    static std::uint64_t dispatches = 0;
    auto lap = std::chrono::steady_clock::now();
    if (profileDrains) {
        profiledAttempts.fetch_add(1, std::memory_order_relaxed);
        if (packetStartedAt() != std::chrono::steady_clock::time_point{} && packetStartedAt() <= attemptStart) packetToAttemptNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(attemptStart - packetStartedAt()).count()), std::memory_order_relaxed);
        attemptToKeyNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(lap - attemptStart).count()), std::memory_order_relaxed);
    }

    std::array<double, DriverPhaseCount> phaseMs{};
    auto phaseLap = lap;
    DispatchPhaseTiming phaseTiming{profile, lap, phaseMs, phaseLap};
    if (profile && packetStartedAt() != std::chrono::steady_clock::time_point{}) phaseMs[PhasePrologue] = std::chrono::duration<double, std::milli>(lap - packetStartedAt()).count();

    static const bool noDispatchCacheEnv = std::getenv("APS5_NO_DISPATCH_CACHE") != nullptr;

    static const std::pair<std::uint64_t, std::uint64_t> probeDispatch = [] {
        const char* text = std::getenv("APS5_PROBE_DISPATCH");
        if (text == nullptr) return std::pair<std::uint64_t, std::uint64_t>{0, 0};
        char* end = nullptr;
        const auto probeAddress = std::strtoull(text, &end, 16);
        const auto index = end != nullptr && *end == ':' ? std::strtoull(end + 1, nullptr, 10) : 0ull;
        return std::pair<std::uint64_t, std::uint64_t>{probeAddress, index};
    }();
    bool probeThis = false;

    if (probeDispatch.first != 0 && (address & 0xfffffffffull) == (probeDispatch.first & 0xfffffffffull)) {
        static std::atomic<std::uint64_t> dispatchesSeen{0};
        probeThis = dispatchesSeen.fetch_add(1) == probeDispatch.second;
        if (probeThis) std::fprintf(stderr, "[gpu] probing dispatch %llu of 0x%llx\n", static_cast<unsigned long long>(probeDispatch.second), static_cast<unsigned long long>(address));
    }

    struct ProbeScope {
        bool active;
        explicit ProbeScope(bool active) : active(active) { if (active) ShaderRecompiler::SetDebugProbeActive(true); }
        ~ProbeScope() { if (active) ShaderRecompiler::SetDebugProbeActive(false); }
    } probeScope{probeThis};
    const bool noDispatchCache = noDispatchCacheEnv || probeThis || forceFresh;
    std::uint64_t key = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
    };
    mix(address);
    mix(packet[4] & 0x8000u);
    for (const auto threads : compute.partialThreads) mix(threads);
    mix(compute.singleWorkgroup ? 0x51u : 0u);
    for (const auto word : userData) mix(word);

    static const bool keyHygiene = std::getenv("APS5_NO_DISPATCH_KEY_HYGIENE") == nullptr;
    if (keyHygiene) {
        for (const auto offset : {0x207u, 0x208u, 0x209u, 0x212u, 0x213u}) {
            const auto found = queue.shader.find(offset);

            mix(found == queue.shader.end() ? (1ull << 32u) : found->second);
        }
    } else {
        for (const auto& [offset, value] : queue.shader) {
            mix(offset);
            mix(value);
        }
    }
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiledResult;

    std::shared_ptr<DispatchVariant> keepVariant;

    std::shared_ptr<DispatchVariant> attachVariant;

    std::shared_ptr<ShaderMemory> shaderMemory;
    std::vector<ShaderRecompiler::MemoryRegion> captured;

    std::vector<std::uint32_t> liveWords;
    bool dataHit = false;
    bool cached = false;
    bool validated = false;

    std::shared_ptr<DispatchEntry> missedEntry;
    bool missedDiffering = false;
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;

    static const bool traceCache = std::getenv("APS5_TRACE_DISPATCH_CACHE") != nullptr;
    if (traceCache) {
        std::lock_guard traceLock(dispatchCacheMutex);
        struct Last { std::vector<std::uint32_t> userData; std::map<std::uint32_t, std::uint32_t> shader; std::uint64_t key; };
        static std::map<std::uint64_t, Last> last;
        static int reports = 0;
        auto& previous = last[address];
        if (previous.key != 0 && previous.key != key && reports < 200) {
            std::string what;
            for (std::size_t i = 0; i < userData.size(); ++i) {
                if (i >= previous.userData.size() || previous.userData[i] != userData[i]) {
                    char text[48];
                    std::snprintf(text, sizeof(text), " user[%zu] %08x->%08x", i, i < previous.userData.size() ? previous.userData[i] : 0u, userData[i]);
                    what += text;
                }
            }
            for (const auto& [offset, value] : queue.shader) {
                const auto old = previous.shader.find(offset);
                if (old == previous.shader.end() || old->second != value) {
                    char text[48];
                    std::snprintf(text, sizeof(text), " sh[%x] %08x->%08x", offset, old == previous.shader.end() ? 0u : old->second, value);
                    what += text;
                }
            }
            ++reports;
            std::fprintf(stderr, "[dispatch-cache] 0x%llx key changed:%s\n", static_cast<unsigned long long>(address), what.c_str());
        }
        previous.userData = userData;
        previous.shader = std::map<std::uint32_t, std::uint32_t>(queue.shader.begin(), queue.shader.end());
        previous.key = key;
    }

    mix(reinterpret_cast<std::uintptr_t>(registeredShader.get()));
    if (fromAhead && key != ahead.key) {
        fromAhead = false;
        ++groupStats.keyMismatches;
    }
    phaseTiming.Phase(PhaseKey);
    timing.Mark("decode_key_device_setup");
    bool readsClear = false;
    bool busyAtLookup = false;
    if (pipelineCandidate) {
        busyAtLookup = DrawPipeline::Queue0().Busy();
        auto lookupSite = DrainSite::LookupMiss;
        std::vector<ShaderRecompiler::MemoryRegion> reads;
        bool entryFound = false;
        {
            std::lock_guard cacheLock(dispatchCacheMutex);
            if (const auto found = dispatchCache.find(key); found != dispatchCache.end()) {
                entryFound = true;
                for (const auto& variant : found->second->variants) appendEntryRegions(*variant, reads);
            }
        }
        if (!entryFound && !noDispatchCache && !stampValidate() && missItemsOn) {
            readsClear = true;
        } else if (!entryFound || noDispatchCache || stampValidate()) {
            NoteItemFallback(ItemFallback::Miss);
        } else if (verifyDataHits()) {
            NoteItemFallback(ItemFallback::Verify);
            lookupSite = DrainSite::LookupVerify;
        } else {
            readsClear = std::none_of(reads.begin(), reads.end(), [](const auto& region) { return DrawPipeline::Queue0().Overlaps(region.guestAddress, region.bytes.size()); });
            if (!readsClear) NoteItemFallback(ItemFallback::Reads);
            lookupSite = DrainSite::LookupReads;
        }
        if (!readsClear) fallBack(lookupSite);
    }
    if (fromAhead) {
        compiledResult = std::move(ahead.compiledResult);
        keepVariant = std::move(ahead.keepVariant);
        attachVariant = std::move(ahead.attachVariant);
        shaderMemory = std::move(ahead.shaderMemory);
        captured = std::move(ahead.captured);
        liveWords = std::move(ahead.liveWords);
        capture = std::move(ahead.capture);
        dataHit = ahead.dataHit;
        cached = ahead.cached;
        validated = ahead.validated;
        ++groupStats.adopted;
        ++dispatchesAdoptedAhead;
        const auto age = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ahead.resolvedAt).count();
        groupStats.ageMs += age;
        groupStats.ageMaxMs = std::max(groupStats.ageMaxMs, age);
        phaseTiming.Phase(PhaseLookup);
    } else {
        lookupDispatch(address, submission, key, noDispatchCache, traceCache, profile, memory, phaseTiming, phaseMs, compiledResult, keepVariant, captured, liveWords, dataHit, cached, validated, missedEntry, missedDiffering);
    }
    timing.Mark("lookup_cache");
    std::vector<DrawPipeline::Range> itemWrites;
    bool pipelined = false;
    bool missItem = false;
    bool slot = false;
    if (readsClear && slotsOn) {
        slot = !NoDispatchItems();
        if (!slot) fallBack(DrainSite::ItemWrites);
    } else if (readsClear) {
        if (cached && keepVariant != nullptr && keepVariant->recipe.load(std::memory_order_acquire) != nullptr) {
            bool bounded = false;
            itemWrites = DispatchWriteRanges(*compiledResult, bounded);
            pipelined = bounded && !NoDispatchItems();
            if (!bounded) NoteItemFallback(ItemFallback::Writes);
        } else if (missItemsOn && !NoDispatchItems()) {
            missItem = true;
        } else {
            NoteItemFallback(ItemFallback::NoRecipe);
        }
        if (!pipelined && !missItem && DrawPipeline::Queue0().Busy()) DrainAt(cached && keepVariant != nullptr && keepVariant->recipe.load(std::memory_order_acquire) != nullptr ? DrainSite::ItemWrites : DrainSite::NoRecipe, profileDrains);
    }
    if (cached) {
        require(keepVariant != nullptr && keepVariant->shader == registeredShader, "dispatch cache belongs to another registered shader");
        captureMs += phaseTiming.Elapsed();
    } else if (!fromAhead) {
        shaderMemory = std::make_shared<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter(), &QueryPendingSpan);
        if (pipelineRequestsEnabled()) shaderMemory->RecordProbes();
        std::uint64_t forgetAtCapture = 0;

        static const bool dumpShaders = std::getenv("APS5_DUMP_SHADERS") != nullptr;
        try {
            auto invocation = InvocationFor(snapshot, codeOffset, request);
            timing.Mark("prepared_invocation");

            const auto waitedBefore = traceCapSync() ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
            const auto captureStarted = std::chrono::steady_clock::now();
            forgetAtCapture = GuestMemory::ForgetSerial();
            capture = [&] {
                const SampledReadScope sampling(evidenceReads);
                return CaptureInvocation(*shaderMemory, snapshot, codeOffset, request, invocation);
            }();
            captured = shaderMemory->Regions();
            request.context.memory = captured;
            if (traceCapSync()) traceCapture("dispatch-capture", address, submission.queue, captured, Graphics::Recorder::ThreadWaitedMs() - waitedBefore);
            captureMs += phaseTiming.Elapsed();
            phaseTiming.Phase(PhaseCapture);
            if (dumpShaders) static_cast<void>(dumpRequest(address, request));
            const auto started = std::chrono::steady_clock::now();
            const auto captureElapsed = std::chrono::duration<double, std::milli>(started - captureStarted).count();


            timing.Mark("capture_resources");
            compiledResult = invocation.Materialize(*capture);
            timing.Mark("materialize");
            if (compiledResult->cacheHit) ++cacheHits;
            const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            static double totalMs = 0;
            totalMs += elapsed;
            if (SlowMaterializationStats()) NoteSlowMaterialization(submission.queue, address, *compiledResult, capture, captureElapsed, elapsed, resolvingAhead());
            if (profile && elapsed > 200) std::fprintf(stderr, "[gpu] compute shader 0x%llx recompile took %.0f ms (%zu SPIR-V words, %zu captured regions, total %.1f s)\n", static_cast<unsigned long long>(address), elapsed, compiledResult->spirv.size(), captured.size(), totalMs / 1000);
            capturePipelineRequest(address, request, *compiledResult, shaderMemory->Probes(), *localDevice);
        } catch (const std::exception& error) {
            std::string dump;
            if (dumpShaders) {
                const auto regions = shaderMemory->Regions();
                try {
                    CaptureTerminalComputeFailure(request, regions, error.what(), submission.queue, submission.serial, currentPacketOffset());
                } catch (...) {}
                auto failed = request;
                failed.context.memory = regions;
                dump = dumpRequest(address, failed, "_failed");
            }

            std::string reason = error.what();
            if (const auto newline = reason.find('\n'); newline != std::string::npos) reason.resize(newline);
            char where[96];
            if (dump.empty()) std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
            else std::snprintf(where, sizeof(where), "compute shader 0x%llx (%s): ", static_cast<unsigned long long>(address), dump.c_str());
            if (!resolving && SkipRefusedShaders() && dynamic_cast<const Graphics::LiveSnapshotInvalidated*>(&error) == nullptr) throw ShaderRefusal(where + reason);
            throw std::runtime_error(where + reason);
        }
        recompileMs += phaseTiming.Elapsed();
        phaseTiming.Phase(PhaseRecompile);
        insertDispatch(address, key, noDispatchCache || !CacheableResult(*compiledResult), profile, registeredShader, forgetAtCapture, memory, shaderMemory, captured, capture, compiledResult, missedEntry, missedDiffering, attachVariant, phaseTiming);
    }
    if (resolving) {
        auto& parked = resolvedAhead()[currentPacketOffset()];
        parked = ResolvedDispatch{};
        parked.address = address;
        parked.key = key;
        parked.indirectArguments = indirectArguments;
        std::copy_n(packet.begin(), std::min(packet.size(), parked.packet.size()), parked.packet.begin());
        parked.registeredShader = registeredShader;
        parked.compiledResult = compiledResult;
        parked.keepVariant = keepVariant;
        parked.attachVariant = attachVariant;
        parked.shaderMemory = shaderMemory;
        parked.captured = std::move(captured);
        parked.liveWords = std::move(liveWords);
        parked.capture = capture;
        parked.dataHit = dataHit;
        parked.cached = cached;
        parked.validated = validated;
        parked.resolvedAt = std::chrono::steady_clock::now();
        ++groupStats.resolved;
        if (cached) ++groupStats.resolvedHits;
        return;
    }
    if (verifyDataHits() && dataHit) verifyDataHit(snapshot, codeOffset, request, memory, address, *keepVariant, liveWords, *compiledResult);

    if (recordQueuedLabelsAfterCapture(submission.queue, captured)) {
        dispatchAttempt(queue, packet, submission, indirectArguments, forceFresh);
        return;
    }
    if (missItem) {
        bool bounded = false;
        itemWrites = DispatchWriteRanges(*compiledResult, bounded);
        if (!bounded || !deferredLabels().labels.empty()) {
            missItem = false;
            NoteItemFallback(ItemFallback::Writes);
            if (DrawPipeline::Queue0().Busy()) DrainAt(DrainSite::ItemWrites, profileDrains);
        }
    }
    if (slot) {
        bool bounded = false;
        itemWrites = DispatchWriteRanges(*compiledResult, bounded);
        if (!bounded || !deferredLabels().labels.empty()) {
            slot = false;
            fallBack(DrainSite::ItemWrites);
        }
    }
    phaseTiming.Phase(PhaseQueuedLabels);
    const auto& compiled = *compiledResult;
    std::vector<Graphics::GuestMemorySnapshot> snapshots;
    for (const auto& region : captured) snapshots.push_back({region.guestAddress, region.bytes});
    std::array<std::uint32_t, 3> groups{packet[1], packet[2], packet[3]};
    if (indirectArguments == 0 && (packet[4] & 0x20u) != 0) {

        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            const auto threads = std::max(readRegister(queue.shader, 0x207 + axis) & 0xffffu, 1u);
            groups[axis] = (groups[axis] + threads - 1) / threads;
        }
    }
    const std::uint64_t dispatchThreads = indirectArguments != 0 ? 0 : std::uint64_t{groups[0]} * groups[1] * groups[2] * compute.numThreads[0] * compute.numThreads[1] * compute.numThreads[2];
    static const bool traceIo = std::getenv("APS5_TRACE_DISPATCH_IO") != nullptr;
    if (traceIo) {

        std::string words;
        if (std::getenv("APS5_TRACE_DISPATCH_IO")[0] == '2') {
            for (const auto word : userData) {
                char text[12];
                std::snprintf(text, sizeof(text), " %08x", word);
                words += text;
            }
        }
        std::fprintf(stderr, "[dispatch-io] queue=0x%x submission=%llu shader 0x%llx%s\n", submission.queue, static_cast<unsigned long long>(submission.serial), static_cast<unsigned long long>(address), words.c_str());
    }

    const auto rethrow = [&](const std::exception& error) {
        std::string message;
        if (fromAhead) {
            ++groupStats.deviceRejects;
            try {
                dispatchAttempt(queue, packet, submission, indirectArguments, forceFresh);
                return true;
            } catch (const Graphics::LiveSnapshotInvalidated& nested) {
                message = nested.what();
            }
        } else {
            CaptureTerminalComputeFailure(request, captured, error.what(), submission.queue, submission.serial, currentPacketOffset());
            message = error.what();
        }
        char where[64];
        std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
        throw std::runtime_error(where + message + describeIndirectSource());
    };
    phaseTiming.Phase(PhaseSnapshots);

    if (slot) {
        auto& counters = SlotCounters();
        const bool hasRecipe = cached && keepVariant != nullptr && keepVariant->recipe.load(std::memory_order_acquire) != nullptr;
        const auto kind = hasRecipe ? SlotKind::HitWithRecipe : cached ? SlotKind::Hit : SlotKind::Miss;
        counters.kinds[static_cast<std::size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
        auto legacySite = DrainSite::Count;
        bool legacyBusy = busyAtLookup;
        if (!legacyItems) {
            legacySite = DrainSite::ItemsOff;
        } else if (htileSite != DrainSite::Count) {
            legacySite = htileSite;
            legacyBusy = busyAtHtile;
        } else if (!legacyMissItems && kind == SlotKind::Miss) {
            legacySite = DrainSite::LookupMiss;
        } else if (!legacyMissItems && kind == SlotKind::Hit) {
            legacySite = DrainSite::NoRecipe;
        }
        if (legacySite != DrainSite::Count && legacyBusy) counters.avoided[static_cast<std::size_t>(legacySite)].fetch_add(1, std::memory_order_relaxed);
        auto state = std::make_shared<DispatchSlot>();
        state->device = localDevice;
        state->shader = registeredShader;
        state->codeOffset = codeOffset;
        state->userData = userData;
        state->compute = compute;
        std::copy_n(packet.begin(), std::min(packet.size(), state->packet.size()), state->packet.begin());
        state->compiled = compiledResult;
        state->keepVariant = keepVariant;
        state->attachVariant = attachVariant;
        state->shaderMemory = shaderMemory;
        state->capture = capture;
        state->captured = std::move(captured);
        state->liveWords = std::move(liveWords);
        state->snapshots = std::move(snapshots);
        state->groups = groups;
        state->address = address;
        state->dispatchThreads = dispatchThreads;
        state->epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
        state->queue = submission.queue;
        state->cached = cached;
        state->dataHit = dataHit;
        state->partialLastGroup = partialLastGroup;
        state->htileKernel = userData.size() >= 6 && Graphics::HtileMaskKernelCode(std::span(snapshot.code).subspan(codeOffset));
        state->htileClearCompletion = htileClearCompletion;
        state->htileMask = htileMask;
        state->htileNote = deferredHtile;
        deferredHtile.reset();
        auto refusal = AheadRefusal::Count;
        if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoDispatchSlotPrepare)) {
            refusal = AheadRefusal::Switch;
        } else if (hasRecipe && !stampValidate()) {
            refusal = AheadRefusal::Recipe;
        } else if (state->htileNote.has_value() || state->htileClearCompletion.has_value()) {
            refusal = AheadRefusal::Htile;
        } else if (!AheadBindings(*compiledResult)) {
            refusal = AheadRefusal::Bindings;
        } else {
            state->inputRanges = SlotInputRanges(*compiledResult, state->captured);
            if (DrawPipeline::Queue0().OverlapsAny(state->inputRanges)) refusal = AheadRefusal::PendingWrites;
        }
        DrawPipeline::Prepare prepareJob;
        if (refusal != AheadRefusal::Count) {
            counters.notAhead[static_cast<std::size_t>(refusal)].fetch_add(1, std::memory_order_relaxed);
        } else {
            state->prepareSlot = std::make_shared<DrawPipeline::PrepareSlot>();
            prepareJob = [state, preparation = state->prepareSlot] {
                auto& work = *state;
                if (!preparation->Begin()) return;
                const auto started = std::chrono::steady_clock::now();
                auto result = std::make_shared<SlotPreparation>();
                try {
                    DrawPipeline::FollowEpoch(work.epoch);
                    GuestMemory::SetCurrentPacket(0x15, work.queue);
                    result->forgetSerial = GuestMemory::ForgetSerial();
                    result->registryGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
                    if (VerifyDispatchSlots()) result->inputHash = SlotInputHash(work.inputRanges);
                    if ((result->forgetSerial & 1u) == 0) result->prepared = work.device->PrepareDispatch(*work.compiled, work.snapshots, work.dispatchThreads, work.address, true);
                } catch (...) {
                    result->prepared = nullptr;
                }
                auto& slotCounters = SlotCounters();
                slotCounters.preparedAhead.fetch_add(1, std::memory_order_relaxed);
                slotCounters.prepareNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
                preparation->Finish(std::move(result));
            };
        }
        DrawPipeline::Queue0().Enqueue([this, state] {
            auto& work = *state;
            DrawPipeline::FollowEpoch(work.epoch);
            GuestMemory::SetCurrentPacket(0x15, work.queue);
            const SlotOrigin origin(work.address, work.htileKernel, work.htileKernel ? work.userData[5] : 0u, work.htileKernel ? work.userData[4] : 0u);
            const auto started = std::chrono::steady_clock::now();
            try {
                commitDispatchSlot(work);
            } catch (const std::exception& error) {
                char where[64];
                std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(work.address));
                throw std::runtime_error(where + std::string(error.what()));
            }
            auto& slotCounters = SlotCounters();
            slotCounters.commits.fetch_add(1, std::memory_order_relaxed);
            slotCounters.commitNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
        }, std::move(itemWrites), 0, {}, {}, std::move(prepareJob));
        if (DrainAfterDispatchItems()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
        ReportDispatchFast(profile);
        ReportDispatchSlots();
        return;
    }
    if (pipelined) {
        const auto recipe = keepVariant->recipe.load(std::memory_order_acquire);
        const auto epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
        const auto queueId = submission.queue;
        itemCommits.fetch_add(1, std::memory_order_relaxed);
        DrawPipeline::Queue0().Enqueue([this, localDevice = std::move(localDevice), result = compiledResult, snapshots = std::move(snapshots), captured = std::move(captured), liveWords = std::move(liveWords), groups, address, queueId, recipe, keepVariant, dataHit, epoch, htileClearCompletion, htileMask] {
            DrawPipeline::FollowEpoch(epoch);
            GuestMemory::SetCurrentPacket(0x15, queueId);
            const auto& compiledCommit = *result;
            const bool noteWrites = writeEvidenceEnabled() || traceCapSync();
            try {
                std::shared_ptr<RecipeHit> recipeHit = htileClearCompletion.has_value() ? nullptr : localDevice->PrepareRecipe(recipe, false);
                std::shared_ptr<const Recipe> builtRecipe;
                bool writersNoted = false;
                for (;;) {
                    std::shared_ptr<PreparedDispatch> prepared;
                    if (recipeHit == nullptr || VulkanDevice::VerifyRecipes() || htileClearCompletion.has_value()) prepared = localDevice->PrepareDispatch(compiledCommit, snapshots);
                    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
                    std::lock_guard gpuLock(GuestMemory::GpuMutex());
                    recordLabelsForPacket(localDevice.get(), queueId);
                    if (noteWrites && writerKeyedEvidence() && !writersNoted) {
                        noteWrittenBuffers(address, queueId, compiledCommit);
                        writersNoted = true;
                    }
                    if (recipeHit != nullptr) {
                        VulkanDevice::IndirectOutcome outcome{0, 0};
                        const auto recipeResult = localDevice->DispatchRecipe(compiledCommit, groups[0], groups[1], groups[2], 0, address, recipeHit, outcome, VulkanDevice::VerifyRecipes() ? prepared : nullptr, dataHit);
                        if (recipeResult == RecipeOutcome::Rebuild) {
                            recipeHit = nullptr;
                            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, false);
                            itemRebuilds.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                    } else {
                        localDevice->Dispatch(compiledCommit, groups[0], groups[1], groups[2], snapshots, address, std::move(prepared), htileClearCompletion.has_value() ? nullptr : &builtRecipe, htileClearCompletion);
                    }
                    if (htileMask.address != 0 && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoHitItemHtileNote)) localDevice->NoteHtileMask(htileMask.address, htileMask.bytes, htileMask.mask, htileMask.value);
                    break;
                }
                if (noteWrites && !writerKeyedEvidence()) noteWrittenBuffers(address, queueId, compiledCommit);
                if (builtRecipe != nullptr) {
                    if (dataHit) {
                        auto own = std::make_shared<Recipe>(*builtRecipe);
                        own->dataWordsHash = Graphics::ShaderResources::DataWordsHash({ShaderRecompiler::ShaderStage::Compute, keepVariant->compiled.get(), 0});
                        builtRecipe = std::move(own);
                    }
                    keepVariant->recipe.store(std::move(builtRecipe), std::memory_order_release);
                    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, false);
                }
            } catch (const std::exception& error) {
                char where[64];
                std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
                throw std::runtime_error(where + std::string(error.what()));
            }
        }, std::move(itemWrites));
        if (DrainAfterDispatchItems()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
        ReportDispatchItems(profile);
        ReportDispatchFast(profile);
        return;
    }
    if (missItem) {
        const auto epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
        const auto queueId = submission.queue;
        auto attachTo = keepVariant != nullptr ? keepVariant : attachVariant;
        missItemCommits.fetch_add(1, std::memory_order_relaxed);
        DrawPipeline::Queue0().Enqueue([this, localDevice = std::move(localDevice), result = compiledResult, captured = std::move(captured), liveWords = std::move(liveWords), capture, shaderMemory, registeredShader, htileMask, htileClearCompletion, groups, address, queueId, dispatchThreads, attachTo = std::move(attachTo), keepVariant, dataHit, epoch] {
            DrawPipeline::FollowEpoch(epoch);
            GuestMemory::SetCurrentPacket(0x15, queueId);
            const auto& compiledCommit = *result;
            std::vector<Graphics::GuestMemorySnapshot> commitSnapshots;
            commitSnapshots.reserve(captured.size());
            for (const auto& region : captured) commitSnapshots.push_back({region.guestAddress, region.bytes});
            const bool noteWrites = writeEvidenceEnabled() || traceCapSync();
            try {
                auto prepared = localDevice->PrepareDispatch(compiledCommit, commitSnapshots, dispatchThreads, address);
                std::shared_ptr<const Recipe> builtRecipe;
                {
                    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
                    std::lock_guard gpuLock(GuestMemory::GpuMutex());
                    recordLabelsForPacket(localDevice.get(), queueId);
                    if (noteWrites && writerKeyedEvidence()) noteWrittenBuffers(address, queueId, compiledCommit);
                    localDevice->Dispatch(compiledCommit, groups[0], groups[1], groups[2], commitSnapshots, address, std::move(prepared), htileClearCompletion.has_value() ? nullptr : (attachTo != nullptr ? &builtRecipe : nullptr), htileClearCompletion);
                    if (htileMask.address != 0) localDevice->NoteHtileMask(htileMask.address, htileMask.bytes, htileMask.mask, htileMask.value);
                }
                if (noteWrites && !writerKeyedEvidence()) noteWrittenBuffers(address, queueId, compiledCommit);
                if (builtRecipe != nullptr) {
                    if (dataHit && keepVariant != nullptr) {
                        auto own = std::make_shared<Recipe>(*builtRecipe);
                        own->dataWordsHash = Graphics::ShaderResources::DataWordsHash({ShaderRecompiler::ShaderStage::Compute, keepVariant->compiled.get(), 0});
                        builtRecipe = std::move(own);
                    }
                    attachTo->recipe.store(std::move(builtRecipe), std::memory_order_release);
                    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, false);
                }
            } catch (const ShaderRefusal&) {
                missItemRefusals.fetch_add(1, std::memory_order_relaxed);
            } catch (const std::exception& error) {
                char where[64];
                std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
                throw std::runtime_error(where + std::string(error.what()));
            }
        }, std::move(itemWrites));
        if (DrainAfterDispatchItems()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
        ReportDispatchItems(profile);
        ReportDispatchFast(profile);
        return;
    }

    settleHtile();
    std::shared_ptr<RecipeHit> recipeHit;
    if (cached && keepVariant != nullptr && !stampValidate()) {
        recipeHit = htileClearCompletion.has_value() ? nullptr : localDevice->PrepareRecipe(keepVariant->recipe.load(std::memory_order_acquire), indirectArguments != 0);
        phaseTiming.Phase(PhaseRecipePrecheck);
    }

    std::shared_ptr<const Recipe> builtRecipe;
    auto* const attachTo = stampValidate() ? nullptr : keepVariant != nullptr ? keepVariant.get() : attachVariant.get();
    bool writersNoted = false;
    const bool noteWrites = writeEvidenceEnabled() || traceCapSync();
    const auto keysFill = indirectArguments == 0 && !partialLastGroup ? MatchSgprPatternFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute) : std::nullopt;
    const auto filledKeys = keysFill ? KeysFillMemo(keysFill->pattern, false, true) : std::nullopt;

    std::optional<Graphics::DepthPlaneClear> depthPlaneClear;
    if (indirectArguments == 0 && !partialLastGroup) {
        if (const auto clear = Graphics::MatchStorageImageClear(std::span(snapshot.code).subspan(codeOffset), userData, compute, groups)) {
            const auto clearAddress = Graphics::StorageImageClearAddress(*clear);
            if (clearAddress != 0 && Graphics::DepthSurfaceAt(clearAddress)) {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
                std::lock_guard clearLock(GuestMemory::GpuMutex());
                recordLabelsForPacket(localDevice.get(), submission.queue);
                depthPlaneClear.emplace(*clear);
            }
        }
    }

    for (;;) {

        std::shared_ptr<PreparedDispatch> prepared;
        if (recipeHit == nullptr || VulkanDevice::VerifyRecipes()) {
            try {
                prepared = localDevice->PrepareDispatch(compiled, snapshots, dispatchThreads, address);
            } catch (const Graphics::LiveSnapshotInvalidated&) {
                throw;
            } catch (const std::exception& error) {
                if (rethrow(error)) return;
            }
            if (profile) {
                const auto now = std::chrono::steady_clock::now();
                const auto prepareMs = std::chrono::duration<double, std::milli>(now - phaseLap).count();
                phaseLap = now;
                double parts = 0;
                if (prepared != nullptr) {
                    const auto phases = VulkanDevice::PreparePhaseMs(*prepared);
                    phaseMs[PhasePrepareKey] += phases[0];
                    phaseMs[PhasePrepareFind] += phases[1];
                    phaseMs[PhasePreparePrecollect] += phases[2];
                    phaseMs[PhasePreparePresync] += phases[3];
                    phaseMs[PhasePrepareStageA] += phases[4];
                    for (const auto part : phases) parts += part;
                }
                phaseMs[PhasePrepareOther] += std::max(0.0, prepareMs - parts);
            }
        }
        GuestMemory::TagGpuLockSite(indirectArguments != 0 ? GuestMemory::GpuLockSite::Indirect : GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        phaseTiming.Phase(PhaseLockWait);
        // The async compute lane takes buffer-only dispatches: one that binds images (texture cache
        // uploads, storage images, bindless tables shared with queue 0's work) is recorded into the
        // main lane, ordered after the ring's earlier work and before its later work (LaneSwitch).
        std::optional<Graphics::Recorder::LaneSwitch> laneSwitch;
        if (Graphics::Recorder::ThreadLane() != 0 && DispatchUsesImages(compiled)) laneSwitch.emplace(Graphics::Recorder::Active(), 0);

        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(PhaseLabels);
        if (noteWrites && writerKeyedEvidence() && !writersNoted) {
            noteWrittenBuffers(address, submission.queue, compiled);
            writersNoted = true;
        }
        phaseTiming.Phase(PhaseNoteWriters);
        auto* const keysRecorder = filledKeys ? Graphics::Recorder::Active() : nullptr;
        const auto keysNoteBefore = keysRecorder != nullptr ? keysRecorder->LastWriteNote(keysFill->base, keysFill->bytes) : 0;
        try {
            if (recipeHit != nullptr) {
                VulkanDevice::IndirectOutcome outcome{0, 0};
                const auto result = localDevice->DispatchRecipe(compiled, groups[0], groups[1], groups[2], indirectArguments, address, recipeHit, outcome, VulkanDevice::VerifyRecipes() ? prepared : nullptr, dataHit, threadGroup);
                if (result == RecipeOutcome::Rebuild) {

                    recipeHit = nullptr;
                    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, indirectArguments != 0);
                    continue;
                }
                if (indirectArguments != 0) countIndirect(outcome.cpuReason, outcome.argumentReadMs);
            } else if (indirectArguments != 0) {
                const auto outcome = localDevice->DispatchIndirect(compiled, indirectArguments, snapshots, address, std::move(prepared), attachTo != nullptr ? &builtRecipe : nullptr, threadGroup);
                countIndirect(outcome.cpuReason, outcome.argumentReadMs);
            } else {
                localDevice->Dispatch(compiled, groups[0], groups[1], groups[2], snapshots, address, std::move(prepared), htileClearCompletion.has_value() ? nullptr : (attachTo != nullptr ? &builtRecipe : nullptr), htileClearCompletion);
            }
        } catch (const Graphics::LiveSnapshotInvalidated&) {
            throw;
        } catch (const std::exception& error) {
            if (rethrow(error)) return;
        }
        if (htileMask.address != 0) localDevice->NoteHtileMask(htileMask.address, htileMask.bytes, htileMask.mask, htileMask.value);
        if (keysRecorder != nullptr && keysRecorder == Graphics::Recorder::Active()) {
            const auto note = keysRecorder->LastWriteNote(keysFill->base, keysFill->bytes);
            if (note != 0 && note != keysNoteBefore) Graphics::NoteKeysFillOnGpu(keysFill->base, keysFill->bytes, *filledKeys);
        }
        if (builtRecipe != nullptr) {
            if (dataHit) {

                auto own = std::make_shared<Recipe>(*builtRecipe);
                own->dataWordsHash = Graphics::ShaderResources::DataWordsHash({ShaderRecompiler::ShaderStage::Compute, keepVariant->compiled.get(), 0});
                builtRecipe = std::move(own);
            }
            attachTo->recipe.store(std::move(builtRecipe), std::memory_order_release);
            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, indirectArguments != 0);
        }
        break;
    }
    phaseTiming.Phase(PhaseDevice);
    if (noteWrites && !writerKeyedEvidence()) noteWrittenBuffers(address, submission.queue, compiled);
    deviceMs += phaseTiming.Elapsed();
    phaseTiming.Phase(PhaseTail);
    ReportDispatchFast(profile);
    if (slotsOn) ReportDispatchSlots();
    if (profile) {
        auto& pending = pendingDispatchPhases();
        pending.outcome = DispatchOutcome::Real;
        pending.phases = true;
        pending.hit = cached;
        pending.validated = validated;
        pending.ms = phaseMs;
        pending.tailAt = phaseLap;
    }
    if (profile && ++dispatches % 100 == 0) AgcDriver::ProfilePrint_nid_no_patch( "[gpu] %llu dispatches (%llu dispatch cache hits, %llu evictions, %llu recompile cache hits): capture %.1f s, cache key %.1f s, recompile %.1f s, device %.1f s\n", static_cast<unsigned long long>(dispatches), static_cast<unsigned long long>(dispatchCacheHits), static_cast<unsigned long long>(dispatchCacheEvictions), static_cast<unsigned long long>(cacheHits), captureMs / 1000, keyMs / 1000, recompileMs / 1000, deviceMs / 1000);
}

}
