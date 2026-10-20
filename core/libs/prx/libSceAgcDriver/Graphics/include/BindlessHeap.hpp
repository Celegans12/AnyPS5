#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BINDLESSHEAP_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BINDLESSHEAP_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "RuntimeAbi.hpp"
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace AgcDriver::Graphics {

class Buffer;
class Recorder;
class Sampler;
class Texture;

class BindlessHeap {
public:
    explicit BindlessHeap(const Context& context);
    ~BindlessHeap();
    BindlessHeap(const BindlessHeap&) = delete;
    BindlessHeap& operator=(const BindlessHeap&) = delete;

    static BindlessHeap& For(const Context& context);
    static BindlessHeap* Existing(const Context& context);
    static void Release(VkDevice device);

    bool DescriptorHeapMode() const { return resourceHeap != nullptr; }
    VkDeviceSize ImageHeapOffset(std::uint32_t kind) const;
    VkDeviceSize ImageHeapStride() const { return imageHeapStride; }
    VkDeviceSize SamplerHeapStride() const { return samplerHeapStride; }
    void BindDescriptorHeaps(VkCommandBuffer commands) const;
    bool DescriptorHeapMappings(std::span<const std::uint32_t> spirv, std::vector<VkDescriptorSetAndBindingMappingEXT>& mappings, std::uint32_t workgroupAddressOffset = ~0u, bool rootGds = false, bool rootFault = false) const;
    VkDeviceAddress RingAddress() const;

    std::optional<std::uint32_t> TextureSlot(std::uint32_t kind, const std::shared_ptr<Texture>& texture, VkImageView view, VkImageLayout layout);
    std::optional<std::uint32_t> StorageSlot(std::uint32_t kind, std::shared_ptr<void> owner, VkImageView view);
    std::uint32_t SlotGeneration(std::uint32_t kind, std::uint32_t slot) const;
    const Texture* SlotTexture(std::uint32_t kind, std::uint32_t slot) const;
    std::optional<std::uint32_t> SamplerSlot(const std::shared_ptr<Sampler>& sampler);
    std::uint32_t SamplerGeneration(std::uint32_t slot) const;
    std::uint64_t SlotSerial() const { return slotSerial.load(std::memory_order_acquire); }
    bool GenerationsCurrent(std::span<const std::vector<std::uint32_t>> slots, std::span<const std::uint32_t> slotGenerations, std::span<const std::uint32_t> samplerSlots, std::span<const std::uint32_t> samplerGenerations) const;
    std::uint32_t AbsentSlot(std::uint32_t kind, VkImageView view);
    static bool NearlyFull();
    static void NoteSweepDue();
    static void SweepIfDue();

    struct DataSpan {
        std::uint32_t offset = 0;
        std::span<std::uint32_t> words;
    };
    DataSpan Allocate(std::size_t words);

    static constexpr std::uint32_t SamplerKind = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Samplers);
    void Submitted(Recorder& recorder, std::uint64_t serial);

    struct Counts {
        std::uint64_t textures = 0, samplers = 0, full = 0, ringWraps = 0;
        std::uint64_t slotsFreed = 0, slotsReused = 0, sweeps = 0, chunkWaits = 0;
    };
    Counts Statistics() const;

    static constexpr std::size_t RingBytes = 64u << 20u;

private:
    Context context;
    std::unique_ptr<Buffer> ring;
    std::unique_ptr<Buffer> resourceHeap;
    std::unique_ptr<Buffer> samplerHeap;
    VkDeviceSize imageHeapStride = 0;
    VkDeviceSize samplerHeapStride = 0;
    VkDeviceSize resourceHeapUsed = 0;
    VkDeviceSize samplerHeapUsed = 0;
    void writeDescriptorHeap(const VkWriteDescriptorSet& write);
    static constexpr std::size_t RingChunkBytes = 4u << 20u;
    struct Chunk {
        Recorder* recorder = nullptr;
        std::uint64_t serial = 0;
        bool open = false;
    };
    std::vector<Chunk> ringChunks;
    std::size_t ringChunk = ~std::size_t{0};
    void enterChunk(std::vector<Chunk>& chunks, std::size_t index, std::size_t& current, const char* what);
    std::size_t ringCursor = 0;
    std::uint32_t ringAlignment = 256;
    mutable std::mutex mutex;
    std::atomic<std::uint64_t> slotSerial{1};
    struct TextureKey {
        std::uint32_t kind;
        const void* texture;
        VkImageView view;
        VkImageLayout layout;
        bool operator==(const TextureKey&) const = default;
    };
    struct TextureKeyHash {
        std::size_t operator()(const TextureKey& key) const { return std::hash<const void*>{}(key.texture) ^ std::hash<const void*>{}(reinterpret_cast<const void*>(key.view)) ^ (static_cast<std::size_t>(key.kind) << 48u) ^ (static_cast<std::size_t>(key.layout) << 40u); }
    };
    std::unordered_map<TextureKey, std::uint32_t, TextureKeyHash> textureSlots;
    std::vector<std::uint32_t> nextTextureSlot;
    struct SlotOwner {
        std::shared_ptr<Texture> texture;
        std::shared_ptr<void> storage;
        TextureKey key{};
        std::uint32_t generation = 0;
    };
    std::vector<std::vector<SlotOwner>> owners;
    std::vector<std::vector<std::uint32_t>> freeSlots;
    std::uint32_t slotCap = 0;
    std::uint64_t slotRequests = 0;
    void sweepLocked(std::vector<std::shared_ptr<Texture>>& dropped);
    std::optional<std::uint32_t> takeSlotLocked(std::uint32_t kind);
    std::unordered_map<const Sampler*, std::uint32_t> samplerSlots;
    std::unordered_map<std::uint32_t, std::uint32_t> absentSlots;
    std::vector<std::shared_ptr<Sampler>> heldSamplers;
    std::vector<std::uint32_t> samplerGenerations;
    std::vector<std::uint32_t> freeSamplerSlots;
    void sweepSamplersLocked(std::vector<std::shared_ptr<Sampler>>& dropped);
    void noteTakenLocked(std::uint32_t kind, std::vector<std::shared_ptr<Texture>>& dropped);
    void noteSamplerTakenLocked(std::vector<std::shared_ptr<Sampler>>& dropped);
    std::uint32_t imageMargin() const;
    std::size_t unusedLocked(std::uint32_t kind) const;
    bool roomLocked() const;
    Counts counts;
};

bool GpuHeapEnabled();
void SetDescriptorHeapDevice(bool available);
bool DescriptorHeapEnabled();
bool HeapComputeEnabled();
bool DescriptorHeapMappable(std::span<const std::uint32_t> spirv, bool workgroupAddress, bool gdsAddress);
void NoteHeapComputeStage();
std::uint64_t HeapComputeStages();
void NoteImageTableBuild();
std::uint64_t ImageTableBuilds();
void NoteDataAddressPush();
void NoteEmptyResourceSet();
std::uint64_t EmptyResourceSets();
std::uint64_t DataAddressPushes();
enum class SetBindSite : std::uint8_t { Draw, Dispatch, Detiler, ColorTransfer, Presentation, Count };
enum class HeapFallback : std::uint8_t { DrawUnmapped, DrawMixedRetry, DrawRefusedPair, DrawMixedReused, DispatchUnmapped, Count };
enum class HeapStageReason : std::uint8_t { Heap, Disabled, PairRefused, FragmentOffHeap, StageKind, NoPlan, PlanStage, UnresolvedImageSource, AtomicTableImage, KnownIneligible, Count };
void NoteHeapStageReason(HeapStageReason reason);
void NoteHeapRecord(bool compute, bool heap);
void NoteHeapFallback(HeapFallback reason);
void NoteCpuBufferCopy(std::uint64_t bytes);
void NoteCommandBufferBegun(VkCommandBuffer commands);
void CmdBindDescriptorSets(const Context& context, VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout layout, std::uint32_t firstSet, std::uint32_t setCount, const VkDescriptorSet* sets, SetBindSite site);
std::uint64_t DescriptorHeapBinds();

}

#endif
