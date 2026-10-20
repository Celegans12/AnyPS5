#include "prx/libSceAgcDriver/Graphics/include/BindlessHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DescriptorHeapObjects.hpp"

#include <algorithm>
#include <unordered_map>
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "RuntimeAbi.hpp"
#include <cstring>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <unordered_set>
#include <mutex>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics {

namespace {

namespace Abi = ShaderRecompiler::RuntimeAbi;

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string("AGC bindless heap: ") + what + " failed: " + std::to_string(static_cast<int>(result)));
}

std::uint32_t fragmentBinding(std::uint32_t kind) {
    return Abi::BindingNumber(Abi::Stage::Fragment, static_cast<Abi::Binding>(kind));
}

std::atomic<std::uint64_t> spansWrittenWindow{0};
std::atomic<std::uint64_t> spansKeptWindow{0};

}

bool GpuHeapEnabled() {
    static const bool enabled = std::getenv("APS5_GPU_HEAP") != nullptr;
    return enabled;
}

namespace {
std::atomic<std::uint64_t> heapComputeStages{0};
std::atomic<bool> descriptorHeapDevice{false};
std::atomic<std::uint64_t> imageTableBuilds{0};
std::atomic<std::uint64_t> dataAddressPushes{0};
std::atomic<std::uint64_t> emptyResourceSets{0};
}

void NoteImageTableBuild() {
    imageTableBuilds.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t ImageTableBuilds() {
    return imageTableBuilds.load(std::memory_order_relaxed);
}

void NoteDataAddressPush() {
    dataAddressPushes.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t DataAddressPushes() {
    return dataAddressPushes.load(std::memory_order_relaxed);
}

void NoteEmptyResourceSet() {
    emptyResourceSets.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t EmptyResourceSets() {
    return emptyResourceSets.load(std::memory_order_relaxed);
}

void NoteHeapComputeStage() {
    heapComputeStages.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t HeapComputeStages() {
    return heapComputeStages.load(std::memory_order_relaxed);
}

void SetDescriptorHeapDevice(bool available) {
    descriptorHeapDevice.store(available, std::memory_order_relaxed);
}

bool DescriptorHeapEnabled() {
    return GpuHeapEnabled() && descriptorHeapDevice.load(std::memory_order_relaxed);
}

bool HeapComputeEnabled() {
    return DescriptorHeapEnabled();
}

BindlessHeap::BindlessHeap(const Context& context) : context(context) {
    Require(context.descriptorHeap, "the GPU heap needs VK_EXT_descriptor_heap");
    const auto& heapProperties = context.descriptorHeapProperties;
    const auto align = [](VkDeviceSize value, VkDeviceSize alignment) { return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment; };
    imageHeapStride = align(heapProperties.imageDescriptorSize, heapProperties.imageDescriptorAlignment);
    samplerHeapStride = align(heapProperties.samplerDescriptorSize, heapProperties.samplerDescriptorAlignment);
    resourceHeapUsed = align(static_cast<VkDeviceSize>(static_cast<std::uint32_t>(Abi::Binding::Samplers)) * Abi::GpuImageHeapSlots * imageHeapStride, heapProperties.resourceHeapAlignment);
    samplerHeapUsed = align(static_cast<VkDeviceSize>(Abi::GpuSamplerHeapSlots) * samplerHeapStride, heapProperties.samplerHeapAlignment);
    Require(resourceHeapUsed + heapProperties.minResourceHeapReservedRange <= heapProperties.maxResourceHeapSize, "the GPU heap's image slots exceed the device's resource heap size");
    Require(samplerHeapUsed + heapProperties.minSamplerHeapReservedRangeWithEmbedded <= heapProperties.maxSamplerHeapSize, "the GPU heap's sampler slots exceed the device's sampler heap size");
    constexpr VkBufferUsageFlags heapUsage = VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    resourceHeap = std::make_unique<Buffer>(context, static_cast<std::size_t>(resourceHeapUsed + heapProperties.minResourceHeapReservedRange), heapUsage);
    samplerHeap = std::make_unique<Buffer>(context, static_cast<std::size_t>(samplerHeapUsed + heapProperties.minSamplerHeapReservedRangeWithEmbedded), heapUsage);
    std::fill(resourceHeap->Bytes().begin(), resourceHeap->Bytes().end(), std::byte{0});
    std::fill(samplerHeap->Bytes().begin(), samplerHeap->Bytes().end(), std::byte{0});
    constexpr VkBufferUsageFlags ringUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    try {
        ring = std::make_unique<Buffer>(context, RingBytes, ringUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    } catch (const std::exception&) {
        ring = std::make_unique<Buffer>(context, RingBytes, ringUsage);
    }
    ringAlignment = static_cast<std::uint32_t>(std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 16));
    nextTextureSlot.assign(static_cast<std::uint32_t>(Abi::Binding::Samplers), 0u);
    ringChunks.resize(RingBytes / RingChunkBytes);
    owners.resize(static_cast<std::uint32_t>(Abi::Binding::Samplers));
    freeSlots.resize(static_cast<std::uint32_t>(Abi::Binding::Samplers));
    const char* cap = std::getenv("APS5_HEAP_SLOT_CAP");
    slotCap = cap != nullptr ? std::min<std::uint32_t>(static_cast<std::uint32_t>(std::strtoul(cap, nullptr, 10)), Abi::GpuImageHeapSlots) : Abi::GpuImageHeapSlots;
    if (slotCap == 0u) slotCap = Abi::GpuImageHeapSlots;
}

BindlessHeap::~BindlessHeap() = default;

namespace {

std::mutex& heapsMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<VkDevice, std::unique_ptr<BindlessHeap>>& heaps() {
    static auto& map = *new std::unordered_map<VkDevice, std::unique_ptr<BindlessHeap>>();
    return map;
}

}

namespace {
std::atomic<std::uint64_t> heapsEpoch{0};
}

BindlessHeap& BindlessHeap::For(const Context& context) {
    struct Cached {
        VkDevice device = VK_NULL_HANDLE;
        BindlessHeap* heap = nullptr;
        std::uint64_t epoch = ~std::uint64_t{0};
    };
    thread_local Cached cached;
    const bool trims = CommitTrims();
    if (trims && cached.heap != nullptr && cached.device == context.device && cached.epoch == heapsEpoch.load(std::memory_order_acquire)) {
        return *cached.heap;
    }
    std::lock_guard lock(heapsMutex());
    auto& heap = heaps()[context.device];
    if (heap == nullptr) {
        heap = std::make_unique<BindlessHeap>(context);
        heapsEpoch.fetch_add(1, std::memory_order_acq_rel);
    }
    if (trims) cached = {context.device, heap.get(), heapsEpoch.load(std::memory_order_acquire)};
    return *heap;
}

void BindlessHeap::Release(VkDevice device) {
    std::unique_ptr<BindlessHeap> released;
    {
        std::lock_guard lock(heapsMutex());
        const auto found = heaps().find(device);
        if (found == heaps().end()) return;
        released = std::move(found->second);
        heaps().erase(found);
        heapsEpoch.fetch_add(1, std::memory_order_acq_rel);
    }
}

BindlessHeap* BindlessHeap::Existing(const Context& context) {
    std::lock_guard lock(heapsMutex());
    const auto found = heaps().find(context.device);
    return found == heaps().end() ? nullptr : found->second.get();
}

std::optional<std::uint32_t> BindlessHeap::TextureSlot(std::uint32_t kind, const std::shared_ptr<Texture>& texture, VkImageView view, VkImageLayout layout) {
    if (kind < Abi::FirstImageBinding || kind >= Abi::FirstStorageImageBinding || texture == nullptr || view == VK_NULL_HANDLE) throw std::runtime_error("AGC bindless heap: invalid texture slot request");
    std::vector<std::shared_ptr<Texture>> dropped;
    std::lock_guard lock(mutex);
    const TextureKey key{kind, texture.get(), view, layout};
    if (const auto found = textureSlots.find(key); found != textureSlots.end()) return found->second;
    if (++slotRequests % 1024u == 0u) NoteSweepDue();
    auto slot = takeSlotLocked(kind);
    if (!slot) {
        sweepLocked(dropped);
        slot = takeSlotLocked(kind);
    }
    if (!slot) {
        ++counts.full;
        return std::nullopt;
    }
    const VkDescriptorImageInfo image{VK_NULL_HANDLE, view, layout};
    const VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, fragmentBinding(kind), *slot, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &image, nullptr, nullptr};
    writeDescriptorHeap(write);
    textureSlots.emplace(key, *slot);
    auto& kindOwners = owners[kind];
    if (kindOwners.size() <= *slot) kindOwners.resize(*slot + 1u);
    kindOwners[*slot].texture = texture;
    kindOwners[*slot].key = key;
    texture->AddHeapHold();
    ++counts.textures;
    noteTakenLocked(kind, dropped);
    return slot;
}

std::optional<std::uint32_t> BindlessHeap::StorageSlot(std::uint32_t kind, std::shared_ptr<void> owner, VkImageView view) {
    if (!Abi::GpuHeapStorageKind(kind) || owner == nullptr || view == VK_NULL_HANDLE) throw std::runtime_error("AGC bindless heap: invalid storage slot request");
    std::vector<std::shared_ptr<Texture>> dropped;
    std::lock_guard lock(mutex);
    const TextureKey key{kind, owner.get(), view, VK_IMAGE_LAYOUT_GENERAL};
    if (const auto found = textureSlots.find(key); found != textureSlots.end()) return found->second;
    if (++slotRequests % 1024u == 0u) NoteSweepDue();
    auto slot = takeSlotLocked(kind);
    if (!slot) {
        sweepLocked(dropped);
        slot = takeSlotLocked(kind);
    }
    if (!slot) {
        ++counts.full;
        return std::nullopt;
    }
    const VkDescriptorImageInfo image{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
    const VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, fragmentBinding(kind), *slot, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &image, nullptr, nullptr};
    writeDescriptorHeap(write);
    textureSlots.emplace(key, *slot);
    auto& kindOwners = owners[kind];
    if (kindOwners.size() <= *slot) kindOwners.resize(*slot + 1u);
    kindOwners[*slot].storage = std::move(owner);
    kindOwners[*slot].key = key;
    ++counts.textures;
    return slot;
}

std::optional<std::uint32_t> BindlessHeap::takeSlotLocked(std::uint32_t kind) {
    auto& free = freeSlots[kind];
    if (!free.empty()) {
        const auto slot = free.back();
        free.pop_back();
        ++counts.slotsReused;
        return slot;
    }
    auto& next = nextTextureSlot[kind];
    if (next >= slotCap) return std::nullopt;
    return next++;
}

void BindlessHeap::sweepLocked(std::vector<std::shared_ptr<Texture>>& dropped) {
    ++counts.sweeps;
    const auto firstDropped = dropped.size();
    for (std::uint32_t kind = 0; kind < owners.size(); ++kind) {
        auto& kindOwners = owners[kind];
        for (std::uint32_t slot = 0; slot < kindOwners.size(); ++slot) {
            auto& owner = kindOwners[slot];
            if (owner.storage != nullptr) {
                if (owner.storage.use_count() > 1) continue;
                textureSlots.erase(owner.key);
                owner.storage.reset();
                owner.key = {};
                ++owner.generation;
                slotSerial.fetch_add(1, std::memory_order_release);
                freeSlots[kind].push_back(slot);
                ++counts.slotsFreed;
                continue;
            }
            if (owner.texture == nullptr || owner.texture.use_count() > static_cast<long>(owner.texture->HeapHolds())) continue;
            textureSlots.erase(owner.key);
            dropped.push_back(std::move(owner.texture));
            owner.key = {};
            ++owner.generation;
            slotSerial.fetch_add(1, std::memory_order_release);
            freeSlots[kind].push_back(slot);
            ++counts.slotsFreed;
        }
    }
    for (auto index = firstDropped; index < dropped.size(); ++index) dropped[index]->DropHeapHold();
}

void BindlessHeap::sweepSamplersLocked(std::vector<std::shared_ptr<Sampler>>& dropped) {
    for (std::uint32_t slot = 0; slot < heldSamplers.size(); ++slot) {
        auto& held = heldSamplers[slot];
        if (held == nullptr || held.use_count() != 1) continue;
        samplerSlots.erase(held.get());
        dropped.push_back(std::move(held));
        ++samplerGenerations[slot];
        slotSerial.fetch_add(1, std::memory_order_release);
        freeSamplerSlots.push_back(slot);
        ++counts.slotsFreed;
    }
}

namespace {

constexpr std::uint32_t ImageSlotMargin = 1024u;
constexpr std::uint32_t SamplerSlotMargin = Abi::GpuSamplerHeapSlots / 4u;
std::atomic<bool> nearlyFull{false};

void noteNearlyFull(const char* what, std::uint32_t margin) {
    if (!nearlyFull.exchange(true, std::memory_order_relaxed)) std::fprintf(stderr, "[gpu-heap] fewer than %u %s slots are free; draws take the per-draw path from now on\n", margin, what);
}

}

bool BindlessHeap::NearlyFull() {
    return nearlyFull.load(std::memory_order_relaxed);
}

namespace {

std::atomic<bool> sweepDue{false};

}

void BindlessHeap::NoteSweepDue() {
    sweepDue.store(true, std::memory_order_relaxed);
}

void BindlessHeap::SweepIfDue() {
    if (!sweepDue.load(std::memory_order_relaxed) || !sweepDue.exchange(false, std::memory_order_relaxed)) return;
    std::vector<std::shared_ptr<Texture>> dropped;
    std::vector<std::shared_ptr<Sampler>> droppedSamplers;
    std::lock_guard heapsLock(heapsMutex());
    bool room = true;
    for (auto& [device, heap] : heaps()) {
        if (heap == nullptr) continue;
        std::lock_guard lock(heap->mutex);
        heap->sweepLocked(dropped);
        heap->sweepSamplersLocked(droppedSamplers);
        room = room && heap->roomLocked();
    }
    if (room && nearlyFull.exchange(false, std::memory_order_relaxed)) std::fprintf(stderr, "[gpu-heap] a sweep left twice the margin of slots free; draws take the heap again\n");
}

std::uint32_t BindlessHeap::imageMargin() const {
    return std::max(1u, std::min(ImageSlotMargin, slotCap / 4u));
}

std::size_t BindlessHeap::unusedLocked(std::uint32_t kind) const {
    const auto next = nextTextureSlot[kind];
    return (next < slotCap ? slotCap - next : 0u) + freeSlots[kind].size();
}

bool BindlessHeap::roomLocked() const {
    for (std::uint32_t kind = Abi::FirstImageBinding; kind < Abi::FirstStorageImageBinding; ++kind) {
        if (unusedLocked(kind) < 2u * static_cast<std::size_t>(imageMargin())) return false;
    }
    return Abi::GpuSamplerHeapSlots - heldSamplers.size() + freeSamplerSlots.size() >= 2u * SamplerSlotMargin;
}

void BindlessHeap::noteTakenLocked(std::uint32_t kind, std::vector<std::shared_ptr<Texture>>& dropped) {
    if (nearlyFull.load(std::memory_order_relaxed)) return;
    const auto margin = imageMargin();
    if (unusedLocked(kind) >= margin) return;
    sweepLocked(dropped);
    if (unusedLocked(kind) < margin) noteNearlyFull("image", margin);
}

void BindlessHeap::noteSamplerTakenLocked(std::vector<std::shared_ptr<Sampler>>& dropped) {
    if (nearlyFull.load(std::memory_order_relaxed)) return;
    const auto unused = [&] { return Abi::GpuSamplerHeapSlots - heldSamplers.size() + freeSamplerSlots.size(); };
    if (unused() >= SamplerSlotMargin) return;
    sweepSamplersLocked(dropped);
    if (unused() < SamplerSlotMargin) noteNearlyFull("sampler", SamplerSlotMargin);
}

std::uint32_t BindlessHeap::SamplerGeneration(std::uint32_t slot) const {
    std::lock_guard lock(mutex);
    return slot < samplerGenerations.size() ? samplerGenerations[slot] : 0u;
}

bool BindlessHeap::GenerationsCurrent(std::span<const std::vector<std::uint32_t>> slots, std::span<const std::uint32_t> slotGenerations, std::span<const std::uint32_t> samplerSlots, std::span<const std::uint32_t> samplerGenerations) const {
    std::lock_guard lock(mutex);
    std::size_t generation = 0;
    for (std::uint32_t kind = 0; kind < slots.size(); ++kind) {
        for (const auto slot : slots[kind]) {
            const auto current = kind < owners.size() && slot < owners[kind].size() ? owners[kind][slot].generation : 0u;
            if (generation >= slotGenerations.size() || current != slotGenerations[generation++]) return false;
        }
    }
    for (std::size_t i = 0; i < samplerSlots.size(); ++i) {
        if (i >= samplerGenerations.size() || (samplerSlots[i] < this->samplerGenerations.size() ? this->samplerGenerations[samplerSlots[i]] : 0u) != samplerGenerations[i]) return false;
    }
    return true;
}

std::uint32_t BindlessHeap::SlotGeneration(std::uint32_t kind, std::uint32_t slot) const {
    std::lock_guard lock(mutex);
    return kind < owners.size() && slot < owners[kind].size() ? owners[kind][slot].generation : 0u;
}

const Texture* BindlessHeap::SlotTexture(std::uint32_t kind, std::uint32_t slot) const {
    std::lock_guard lock(mutex);
    return kind < owners.size() && slot < owners[kind].size() ? owners[kind][slot].texture.get() : nullptr;
}

std::uint32_t BindlessHeap::AbsentSlot(std::uint32_t kind, VkImageView view) {
    if (kind < Abi::FirstImageBinding || kind >= static_cast<std::uint32_t>(Abi::Binding::Samplers)) throw std::runtime_error("AGC bindless heap: invalid absent slot request");
    std::lock_guard lock(mutex);
    if (const auto found = absentSlots.find(kind); found != absentSlots.end()) return found->second;
    const auto taken = takeSlotLocked(kind);
    if (!taken) throw std::runtime_error("AGC bindless heap: the image heap is full");
    const auto slot = *taken;
    const VkDescriptorImageInfo image{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
    const VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, fragmentBinding(kind), slot, 1, Abi::GpuHeapStorageKind(kind) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &image, nullptr, nullptr};
    writeDescriptorHeap(write);
    absentSlots.emplace(kind, slot);
    return slot;
}

std::optional<std::uint32_t> BindlessHeap::SamplerSlot(const std::shared_ptr<Sampler>& sampler) {
    if (sampler == nullptr) throw std::runtime_error("AGC bindless heap: invalid sampler slot request");
    std::vector<std::shared_ptr<Sampler>> dropped;
    std::lock_guard lock(mutex);
    if (const auto found = samplerSlots.find(sampler.get()); found != samplerSlots.end()) return found->second;
    if (freeSamplerSlots.empty()) sweepSamplersLocked(dropped);
    std::uint32_t slot = 0;
    if (!freeSamplerSlots.empty()) {
        slot = freeSamplerSlots.back();
        freeSamplerSlots.pop_back();
        ++counts.slotsReused;
    } else if (heldSamplers.size() < Abi::GpuSamplerHeapSlots) {
        slot = static_cast<std::uint32_t>(heldSamplers.size());
        heldSamplers.emplace_back();
        samplerGenerations.push_back(0u);
    } else {
        ++counts.full;
        return std::nullopt;
    }
    const VkDescriptorImageInfo info{sampler->Handle(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    const VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, fragmentBinding(static_cast<std::uint32_t>(Abi::Binding::Samplers)), slot, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &info, nullptr, nullptr};
    writeDescriptorHeap(write);
    samplerSlots.emplace(sampler.get(), slot);
    heldSamplers[slot] = sampler;
    ++counts.samplers;
    noteSamplerTakenLocked(dropped);
    return slot;
}

BindlessHeap::DataSpan BindlessHeap::Allocate(std::size_t words) {
    SpanMark mark;
    return Allocate(words, mark);
}

bool BindlessHeap::Held(const SpanMark& mark) {
    if (mark.entry == 0u || mark.words == 0u) return false;
    const auto first = mark.offset / RingChunkBytes;
    const auto last = (mark.offset + static_cast<std::size_t>(mark.words) * sizeof(std::uint32_t) - 1u) / RingChunkBytes;
    if (first != last) return false;
    std::lock_guard lock(mutex);
    const auto& chunk = ringChunks[first];
    if (chunk.entries != mark.entry || !(chunk.open || first == ringChunk)) return false;
    ++counts.spansKept;
    spansKeptWindow.fetch_add(1, std::memory_order_relaxed);
    return true;
}

BindlessHeap::DataSpan BindlessHeap::Allocate(std::size_t words, SpanMark& mark) {
    const auto bytes = words * sizeof(std::uint32_t);
    if (bytes > RingChunkBytes) throw std::runtime_error("AGC bindless heap: per-draw data exceeds a ring chunk");
    std::lock_guard lock(mutex);
    auto offset = (ringCursor + ringAlignment - 1) / ringAlignment * ringAlignment;
    if (offset + bytes > RingBytes) {
        offset = 0;
        ++counts.ringWraps;
    }
    ringCursor = offset + bytes;
    for (auto chunk = offset / RingChunkBytes; chunk <= (offset + std::max<std::size_t>(bytes, 1u) - 1u) / RingChunkBytes; ++chunk) enterChunk(ringChunks, chunk, ringChunk, "shader data");
    auto* base = reinterpret_cast<std::uint32_t*>(ring->Bytes().data() + offset);
    ++counts.spansWritten;
    spansWrittenWindow.fetch_add(1, std::memory_order_relaxed);
    mark = {static_cast<std::uint32_t>(offset), static_cast<std::uint32_t>(words), ringChunks[offset / RingChunkBytes].entries};
    return {static_cast<std::uint32_t>(offset), std::span<std::uint32_t>(base, words)};
}

void BindlessHeap::enterChunk(std::vector<Chunk>& chunks, std::size_t index, std::size_t& current, const char* what) {
    if (index == current) return;
    current = index;
    auto& chunk = chunks[index];
    if (chunk.open) throw std::runtime_error(std::string("AGC bindless heap: one batch's ") + what + " exceeds its ring");
    if (chunk.serial != 0u) {
        ++counts.chunkWaits;
        if (!chunk.recorder->HasTimeline()) throw std::runtime_error(std::string("AGC bindless heap: the ") + what + " ring needs the recorder timeline to wait for a batch");
        chunk.recorder->WaitSerial(chunk.serial);
        chunk.serial = 0u;
    }
    chunk.open = true;
    ++chunk.entries;
}

VkDeviceSize BindlessHeap::ImageHeapOffset(std::uint32_t kind) const {
    return static_cast<VkDeviceSize>(kind) * Abi::GpuImageHeapSlots * imageHeapStride;
}

void BindlessHeap::writeDescriptorHeap(const VkWriteDescriptorSet& write) {
    constexpr auto bindingCount = static_cast<std::uint32_t>(Abi::Binding::Count);
    const auto kind = write.dstBinding % bindingCount;
    if (write.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER) {
        const auto writeSamplers = context.Function<PFN_vkWriteSamplerDescriptorsEXT>("vkWriteSamplerDescriptorsEXT");
        for (std::uint32_t element = 0; element < write.descriptorCount; ++element) {
            auto record = FindSampler(write.pImageInfo[element].sampler);
            Require(record.has_value(), "a GPU heap sampler has no recorded create info");
            const auto offset = static_cast<VkDeviceSize>(write.dstArrayElement + element) * samplerHeapStride;
            Require(offset + samplerHeapStride <= samplerHeapUsed, "a GPU heap sampler slot lies outside the sampler heap");
            const VkHostAddressRangeEXT target{samplerHeap->Bytes().data() + offset, static_cast<std::size_t>(samplerHeapStride)};
            check(writeSamplers(context.device, 1, record->Chain(), &target), "vkWriteSamplerDescriptorsEXT");
        }
        return;
    }
    Require(write.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, "the GPU heap writes only images and samplers into the descriptor heap");
    const auto writeResources = context.Function<PFN_vkWriteResourceDescriptorsEXT>("vkWriteResourceDescriptorsEXT");
    for (std::uint32_t element = 0; element < write.descriptorCount; ++element) {
        const auto& image = write.pImageInfo[element];
        const auto offset = ImageHeapOffset(kind) + static_cast<VkDeviceSize>(write.dstArrayElement + element) * imageHeapStride;
        Require(offset + imageHeapStride <= resourceHeapUsed, "a GPU heap image slot lies outside the resource heap");
        auto* target = resourceHeap->Bytes().data() + offset;
        if (image.imageView == VK_NULL_HANDLE) {
            std::memset(target, 0, static_cast<std::size_t>(imageHeapStride));
            continue;
        }
        auto record = FindImageView(image.imageView);
        Require(record.has_value(), "a GPU heap image view has no recorded create info");
        const VkImageDescriptorInfoEXT imageInfo{VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT, nullptr, record->Chain(), image.imageLayout};
        VkResourceDescriptorInfoEXT resource{VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT};
        resource.type = write.descriptorType;
        resource.data.pImage = &imageInfo;
        const VkHostAddressRangeEXT range{target, static_cast<std::size_t>(imageHeapStride)};
        check(writeResources(context.device, 1, &resource, &range), "vkWriteResourceDescriptorsEXT");
    }
}

namespace {

const std::uint32_t GeneratedFaultBinding = Abi::BindingNumber(Abi::Stage::TessellationControl, Abi::Binding::FaultBuffer);

bool heapBindings(std::span<const std::uint32_t> spirv, bool workgroupAddress, bool gdsAddress, bool faultAddress, std::vector<std::uint32_t>& bindings, bool& workgroupMemory, std::uint32_t& gdsBinding, bool& faultMemory) {
    bindings.clear();
    workgroupMemory = false;
    gdsBinding = ~0u;
    faultMemory = false;
    if (spirv.size() < 5u) return false;
    constexpr std::uint32_t opDecorate = 71u;
    constexpr std::uint32_t decorationBinding = 33u;
    constexpr std::uint32_t decorationSet = 34u;
    std::unordered_map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> decorated;
    for (std::size_t at = 5; at < spirv.size();) {
        const auto count = spirv[at] >> 16u;
        if (count == 0u || at + count > spirv.size()) return false;
        if ((spirv[at] & 0xffffu) == opDecorate && count >= 4u) {
            auto& entry = decorated.try_emplace(spirv[at + 1u], std::pair<std::uint32_t, std::uint32_t>{~0u, ~0u}).first->second;
            if (spirv[at + 2u] == decorationSet) entry.first = spirv[at + 3u];
            if (spirv[at + 2u] == decorationBinding) entry.second = spirv[at + 3u];
        }
        at += count;
    }
    constexpr auto bindingCount = static_cast<std::uint32_t>(Abi::Binding::Count);
    constexpr auto samplerBinding = static_cast<std::uint32_t>(Abi::Binding::Samplers);
    for (const auto& [id, entry] : decorated) {
        if (entry.first == ~0u && entry.second == ~0u) continue;
        if (workgroupAddress && entry.first == ShaderRecompiler::GpuWorkgroupMemoryDescriptorSet && entry.second == 0u) {
            workgroupMemory = true;
            continue;
        }
        if (gdsAddress && entry.first == Abi::GpuGdsSet && gdsBinding == ~0u) {
            gdsBinding = entry.second;
            continue;
        }
        if (faultAddress && entry.first == Abi::DescriptorSet && entry.second == GeneratedFaultBinding) {
            faultMemory = true;
            continue;
        }
        if (entry.first != Abi::FragmentDescriptorSet || entry.second < bindingCount + Abi::FirstImageBinding || entry.second > bindingCount + samplerBinding) return false;
        if (std::find(bindings.begin(), bindings.end(), entry.second) == bindings.end()) bindings.push_back(entry.second);
    }
    return (!workgroupAddress || workgroupMemory) && (!faultAddress || faultMemory);
}

}

bool DescriptorHeapMappable(std::span<const std::uint32_t> spirv, bool workgroupAddress, bool gdsAddress) {
    std::vector<std::uint32_t> bindings;
    bool workgroupMemory = false;
    std::uint32_t gdsBinding = ~0u;
    bool faultMemory = false;
    return heapBindings(spirv, workgroupAddress, gdsAddress, false, bindings, workgroupMemory, gdsBinding, faultMemory);
}

bool BindlessHeap::DescriptorHeapMappings(std::span<const std::uint32_t> spirv, std::vector<VkDescriptorSetAndBindingMappingEXT>& mappings, std::uint32_t workgroupAddressOffset, bool rootGds, bool rootFault) const {
    mappings.clear();
    std::vector<std::uint32_t> bindings;
    bool workgroupMemory = false;
    std::uint32_t gdsBinding = ~0u;
    bool faultMemory = false;
    if (resourceHeap == nullptr || !heapBindings(spirv, workgroupAddressOffset != ~0u, rootGds, rootFault, bindings, workgroupMemory, gdsBinding, faultMemory)) return false;
    constexpr auto bindingCount = static_cast<std::uint32_t>(Abi::Binding::Count);
    constexpr auto samplerBinding = static_cast<std::uint32_t>(Abi::Binding::Samplers);
    std::sort(bindings.begin(), bindings.end());
    for (const auto binding : bindings) {
        VkDescriptorSetAndBindingMappingEXT mapping{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT};
        mapping.descriptorSet = Abi::FragmentDescriptorSet;
        mapping.firstBinding = binding;
        mapping.bindingCount = 1;
        mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
        const auto kind = binding - bindingCount;
        if (kind == samplerBinding) {
            mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
            mapping.sourceData.constantOffset.heapOffset = 0;
            mapping.sourceData.constantOffset.heapArrayStride = static_cast<std::uint32_t>(samplerHeapStride);
        } else {
            mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT | VK_SPIRV_RESOURCE_TYPE_READ_ONLY_IMAGE_BIT_EXT | VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT;
            mapping.sourceData.constantOffset.heapOffset = static_cast<std::uint32_t>(ImageHeapOffset(kind));
            mapping.sourceData.constantOffset.heapArrayStride = static_cast<std::uint32_t>(imageHeapStride);
        }
        mappings.push_back(mapping);
    }
    if (workgroupMemory) {
        VkDescriptorSetAndBindingMappingEXT mapping{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT};
        mapping.descriptorSet = ShaderRecompiler::GpuWorkgroupMemoryDescriptorSet;
        mapping.firstBinding = 0;
        mapping.bindingCount = 1;
        mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
        mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
        mapping.sourceData.pushAddressOffset = workgroupAddressOffset;
        mappings.push_back(mapping);
    }
    if (gdsBinding != ~0u) {
        VkDescriptorSetAndBindingMappingEXT mapping{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT};
        mapping.descriptorSet = Abi::GpuGdsSet;
        mapping.firstBinding = gdsBinding;
        mapping.bindingCount = 1;
        mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
        mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
        mapping.sourceData.pushAddressOffset = (Abi::RootDword + Abi::RootGdsAddress) * sizeof(std::uint32_t);
        mappings.push_back(mapping);
    }
    if (faultMemory) {
        VkDescriptorSetAndBindingMappingEXT mapping{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT};
        mapping.descriptorSet = Abi::DescriptorSet;
        mapping.firstBinding = GeneratedFaultBinding;
        mapping.bindingCount = 1;
        mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
        mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
        mapping.sourceData.pushAddressOffset = (Abi::RootDword + Abi::RootFaultAddress) * sizeof(std::uint32_t);
        mappings.push_back(mapping);
    }
    return true;
}

namespace {

std::mutex boundHeapsMutex;
std::unordered_set<VkCommandBuffer> boundHeaps;
std::atomic<std::uint64_t> heapBinds{0};
std::atomic<std::uint64_t> heapBindWindow{0};
std::atomic<std::uint64_t> commandBuffersBegun{0};
std::atomic<std::uint64_t> cpuBufferCopies{0};
std::atomic<std::uint64_t> cpuBufferCopyBytes{0};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(SetBindSite::Count)> setBinds{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(HeapFallback::Count)> heapFallbacks{};
std::array<std::atomic<std::uint64_t>, 4> heapRecords{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(HeapStageReason::Count)> heapStageReasons{};
constexpr std::array<const char*, static_cast<std::size_t>(HeapStageReason::Count)> HeapStageReasonNames{"heap", "disabled", "pair refused", "fragment off heap", "stage kind", "no plan", "plan stage", "unresolved image source", "atomic table image", "known ineligible"};
constexpr std::array<const char*, static_cast<std::size_t>(SetBindSite::Count)> SetBindSiteNames{"draw", "dispatch", "detiler", "color transfer", "presentation"};
constexpr std::array<const char*, static_cast<std::size_t>(HeapFallback::Count)> HeapFallbackNames{"draw unmapped", "draw mixed retry", "draw refused pair", "draw mixed reused", "dispatch unmapped"};

std::uint64_t takeCount(std::atomic<std::uint64_t>& counter) {
    return counter.exchange(0, std::memory_order_relaxed);
}

void reportHeapStatistics() {
    static const bool enabled = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!enabled) return;
    static std::atomic<std::uint64_t> calls{0};
    if ((calls.fetch_add(1, std::memory_order_relaxed) & 1023u) != 0u) return;
    static std::mutex reportMutex;
    static auto last = std::chrono::steady_clock::now();
    std::unique_lock lock(reportMutex, std::try_to_lock);
    if (!lock.owns_lock()) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(10)) return;
    last = now;
    std::string binds;
    for (std::size_t i = 0; i < setBinds.size(); ++i) binds += " " + std::string(SetBindSiteNames[i]) + " " + std::to_string(takeCount(setBinds[i]));
    std::string fallbacks;
    for (std::size_t i = 0; i < heapFallbacks.size(); ++i) fallbacks += " " + std::string(HeapFallbackNames[i]) + " " + std::to_string(takeCount(heapFallbacks[i]));
    std::string stageReasons;
    for (std::size_t i = 0; i < heapStageReasons.size(); ++i) stageReasons += " " + std::string(HeapStageReasonNames[i]) + " " + std::to_string(takeCount(heapStageReasons[i]));
    const auto heapDraws = takeCount(heapRecords[1]);
    const auto plainDraws = takeCount(heapRecords[0]);
    const auto heapDispatches = takeCount(heapRecords[3]);
    const auto plainDispatches = takeCount(heapRecords[2]);
    std::fprintf(stderr, "[gpu-heap] (10 s): draws %llu heap / %llu plain, dispatches %llu heap / %llu plain; heap binds %llu over %llu command buffers; set binds:%s; fallbacks:%s; draw stages:%s; CPU buffer copies %llu (%llu MiB); ring spans %llu written / %llu kept\n", static_cast<unsigned long long>(heapDraws), static_cast<unsigned long long>(plainDraws), static_cast<unsigned long long>(heapDispatches), static_cast<unsigned long long>(plainDispatches), static_cast<unsigned long long>(takeCount(heapBindWindow)), static_cast<unsigned long long>(takeCount(commandBuffersBegun)), binds.c_str(), fallbacks.c_str(), stageReasons.c_str(), static_cast<unsigned long long>(takeCount(cpuBufferCopies)), static_cast<unsigned long long>(takeCount(cpuBufferCopyBytes) >> 20u), static_cast<unsigned long long>(takeCount(spansWrittenWindow)), static_cast<unsigned long long>(takeCount(spansKeptWindow)));
}

void forgetHeapBinding(VkCommandBuffer commands) {
    if (!DescriptorHeapEnabled()) return;
    std::lock_guard lock(boundHeapsMutex);
    boundHeaps.erase(commands);
}

}

void NoteHeapRecord(bool compute, bool heap) {
    heapRecords[(compute ? 2u : 0u) + (heap ? 1u : 0u)].fetch_add(1, std::memory_order_relaxed);
    reportHeapStatistics();
}

void NoteHeapStageReason(HeapStageReason reason) {
    heapStageReasons[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void NoteCpuBufferCopy(std::uint64_t bytes) {
    cpuBufferCopies.fetch_add(1, std::memory_order_relaxed);
    cpuBufferCopyBytes.fetch_add(bytes, std::memory_order_relaxed);
}

void NoteHeapFallback(HeapFallback reason) {
    heapFallbacks[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void NoteCommandBufferBegun(VkCommandBuffer commands) {
    commandBuffersBegun.fetch_add(1, std::memory_order_relaxed);
    forgetHeapBinding(commands);
}

void CmdBindDescriptorSets(const Context& context, VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout layout, std::uint32_t firstSet, std::uint32_t setCount, const VkDescriptorSet* sets, SetBindSite site) {
    context.Resolved(&DeviceFunctions::cmdBindDescriptorSets, "vkCmdBindDescriptorSets")(commands, bindPoint, layout, firstSet, setCount, sets, 0, nullptr);
    setBinds[static_cast<std::size_t>(site)].fetch_add(1, std::memory_order_relaxed);
    forgetHeapBinding(commands);
}

std::uint64_t DescriptorHeapBinds() {
    return heapBinds.load(std::memory_order_relaxed);
}

void BindlessHeap::BindDescriptorHeaps(VkCommandBuffer commands) const {
    if (resourceHeap == nullptr) return;
    {
        std::lock_guard lock(boundHeapsMutex);
        if (!boundHeaps.insert(commands).second) return;
    }
    heapBinds.fetch_add(1, std::memory_order_relaxed);
    heapBindWindow.fetch_add(1, std::memory_order_relaxed);
    VkBindHeapInfoEXT info{VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT};
    info.heapRange = {resourceHeap->DeviceAddress(), resourceHeapUsed + context.descriptorHeapProperties.minResourceHeapReservedRange};
    info.reservedRangeOffset = resourceHeapUsed;
    info.reservedRangeSize = context.descriptorHeapProperties.minResourceHeapReservedRange;
    context.Function<PFN_vkCmdBindResourceHeapEXT>("vkCmdBindResourceHeapEXT")(commands, &info);
    info.heapRange = {samplerHeap->DeviceAddress(), samplerHeapUsed + context.descriptorHeapProperties.minSamplerHeapReservedRangeWithEmbedded};
    info.reservedRangeOffset = samplerHeapUsed;
    info.reservedRangeSize = context.descriptorHeapProperties.minSamplerHeapReservedRangeWithEmbedded;
    context.Function<PFN_vkCmdBindSamplerHeapEXT>("vkCmdBindSamplerHeapEXT")(commands, &info);
}

VkDeviceAddress BindlessHeap::RingAddress() const {
    return ring->DeviceAddress();
}

void BindlessHeap::Submitted(Recorder& recorder, std::uint64_t serial) {
    std::lock_guard lock(mutex);
    for (std::size_t index = 0; index < ringChunks.size(); ++index) {
        if (!ringChunks[index].open && index != ringChunk) continue;
        ringChunks[index].recorder = &recorder;
        ringChunks[index].serial = serial;
        ringChunks[index].open = index == ringChunk;
    }
}

BindlessHeap::Counts BindlessHeap::Statistics() const {
    std::lock_guard lock(mutex);
    return counts;
}

}
