#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SHADERRESOURCES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SHADERRESOURCES_HPP

#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace AgcDriver { class VulkanDevice; }

namespace AgcDriver::Graphics {

class Recorder;

void FlushCachedTextures(VkDevice device);
void ClearCachedTextures(VkDevice device);
std::uint64_t ReleaseTextureCaches(std::uint64_t bytes, bool newImage = false);
std::uint64_t TextureCacheBudget(const VkPhysicalDeviceMemoryProperties& memory);
// What the texture caches hold: every live sampled texture's own image (SampledTextureMemory), the
// part of it the sampled cache's snapshot entries account, the storage image cache's bytes, and the
// part of `sampled` in the overflow type (SampledTextureOverflowMemory): the caches' own, so not
// other usage, but not taken from the sampled ceiling either (the driver moves those pages out of
// the VRAM when it needs the room; the overflow budget bounds them).
struct TextureMemoryUse {
    std::uint64_t sampled = 0;
    std::uint64_t sampledCached = 0;
    std::uint64_t storageCached = 0;
    std::uint64_t overflow = 0;
};
// The texture caches' budgets, out of one share of the device-local heap: VK_EXT_memory_budget's
// budget (or APS5_VRAM_BUDGET_MIB) less the VramBudget headroom and less what everything else uses
// (the heap's usage minus the texture bytes above: the buffer pool, unit shadows, depth surfaces,
// presentation, the driver's own). The storage cache takes up to half of it (never more than
// TextureCacheBudget); the sampled cache admits up to `ceiling`, what is left after the storage
// budget and the sampled images alive outside the cache (in recorded work, in builds), and evicts
// down to `sampled` (four fifths of it). `target` is the share (0 without VK_EXT_memory_budget: the
// static budgets of before, the ceiling a quarter over the sampled budget). `share` is what every
// device-local sampled image together may take, in the cache or not (the ceiling plus the images
// outside the cache when the budgets were read): a new snapshot past it goes to system memory where it can (see Texture).
// APS5_TEXTURE_CACHE_MIB forces both caches' budgets (and the share to the ceiling).
// `previousTarget` (the share of the last read, 0: none) is kept while the new one is within a
// sixty-fourth of it or 64 MiB: the reported usage moves by a few MiB as the driver frees memory
// late (images a retired batch held) or allocates for itself, and the budgets should not follow.
// `overflow` bounds the sampled images in the overflow type (SampledTextureOverflowMemory): the
// driver backs them with system memory once the VRAM is taken, and lean-x24 failed to allocate one
// with heap 1 (system memory) 11.6 of its 15.5 GB budget used and 4.7 GB of them alive, where an
// otherwise idle RTX 3080 Laptop takes 12 GiB of them past a full VRAM (heap 1 nearly empty). So
// the largest heap that is not device-local's budget less its usage less a sixteenth of its budget
// (0: no such heap or no VK_EXT_memory_budget, unbounded); APS5_TEXTURE_OVERFLOW_MIB forces it.
struct TextureCacheBudgets {
    std::uint64_t sampled = 0;
    std::uint64_t ceiling = 0;
    std::uint64_t storage = 0;
    std::uint64_t target = 0;
    std::uint64_t share = 0;
    std::uint64_t overflow = 0;
};
TextureCacheBudgets TextureCacheBudgetsFor(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, const TextureMemoryUse& use, std::uint64_t previousTarget = 0);
std::uint64_t StorageTextureBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, std::uint64_t sampledBytes, std::uint64_t storageBytes);
bool SampledBudgetReportDue(std::uint64_t reported, std::uint64_t budget, std::chrono::steady_clock::duration sinceReport);
std::optional<std::uint32_t> LargestDeviceLocalHeap(const VkPhysicalDeviceMemoryProperties& memory);
std::uint64_t StorageTextureCacheBudget(const Context& context);
inline constexpr std::uint64_t SpareSnapshotPresents = 16;
struct TextureCacheUse {
    std::size_t sampledEntries = 0;
    std::size_t spareEntries = 0;
    std::uint64_t sampledBytes = 0;
    std::uint64_t spareBytes = 0;
    std::uint64_t storageBytes = 0;
};
TextureCacheUse TextureCacheUsage();
void OverrideCacheBudgetForTest(std::uint64_t bytes);
std::uint64_t SampledTextureCacheBudget(const Context& context);
TextureCacheBudgets CurrentTextureCacheBudgets(const Context& context);
// Cumulative sampled-cache counters (the [texture-cache] line reports them per 10 s).
struct SampledCacheCounts {
    std::uint64_t evicted = 0;
    std::uint64_t evictedBytes = 0;
    std::uint64_t refused = 0;
    std::uint64_t refusedBytes = 0;
    std::uint64_t droppedSets = 0;
    std::uint64_t refusedSets = 0;
    std::uint64_t reclaims = 0;
    std::uint64_t reclaimedBytes = 0;
    std::uint64_t spilled = 0;
    std::uint64_t spilledBytes = 0;
    std::uint64_t oneOffReuses = 0;
    std::uint64_t made = 0;
    std::uint64_t spilledReuses = 0;
    std::uint64_t overflowDropped = 0;
    std::uint64_t overflowWaits = 0;
};
SampledCacheCounts SampledCacheCounters();
std::shared_ptr<Texture> CachedSampledTexture(const Context& context, std::span<const std::uint32_t> words, VkImageViewType multisampleViewType = VK_IMAGE_VIEW_TYPE_2D);
std::shared_ptr<Texture> CachedTableTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint64_t guestBytes, bool depthCompare);
bool SampledTexturesShareEntry(std::span<const std::uint32_t> first, std::span<const std::uint32_t> second);
bool AnyPendingCpuWrites(const Context& context);
bool CompletePendingCpuWrites(const Context& context, std::uint64_t address, std::size_t bytes);
bool SpirvStoresThroughAddresses(std::span<const std::uint32_t> words);
bool ShaderStoresThroughAddresses(const ShaderRecompiler::SharedSpirv& spirv);
void LastRecordIndices(std::span<const void* const> logged, std::span<const void* const> objects, std::vector<std::size_t>& indices);

// The cached storage image of a surface (render targets use it as their resident image); brought up
// to date with guest memory before it is returned.
std::shared_ptr<StorageTexture> CachedStorageSurface(const Context& context, const GuestTextureResource& resource);
std::shared_ptr<StorageTexture> CachedStorageLayers(const Context& context, const GuestTextureResource& resource, std::uint32_t firstLayer, std::uint32_t layerCount);
// Whether `image` is still the storage cache's image of its surface (what CachedStorageSurface
// would return); a true answer counts as a use for the cache's eviction order, as a lookup would.
bool StorageImageCached(const Context& context, const StorageTexture* image);
bool StorageImageServesKeys(const StorageTexture& image, std::uint64_t dccAddress);

// Defined in Texture.cpp beside the pending-results registry, for the fast Revalidate below: whether
// a storage image other than `except` has results pending in [address, address + bytes).
bool PendingStorageOverlaps(std::uint64_t address, std::size_t bytes, const StorageTexture* except);

struct PipelineSetLayouts {
    VkDescriptorSetLayout main = VK_NULL_HANDLE;
    VkDescriptorSetLayout fragment = VK_NULL_HANDLE;
    bool split = false;
    std::span<const std::uint32_t> mainKey;
    std::span<const std::uint32_t> fragmentKey;
};

struct TableDescriptors {
    std::uint32_t binding = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> runs;
    std::vector<VkDescriptorImageInfo> images;
    std::shared_ptr<const std::vector<std::weak_ptr<Texture>>> textures;
};

class DescriptorCache {
public:
    explicit DescriptorCache(const Context& context);
    ~DescriptorCache();
    DescriptorCache(const DescriptorCache&) = delete;
    DescriptorCache& operator=(const DescriptorCache&) = delete;

    // The layout for `key` (binding, type, count, stage flags per binding, as ShaderResources builds
    // it from `bindings`), created on first use.
    VkDescriptorSetLayout Layout(std::span<const std::uint32_t> key, std::span<const VkDescriptorSetLayoutBinding> bindings, bool updateAfterBind = false, std::span<const VkDescriptorBindingFlags> flags = {});
    struct SetAllocation {
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::size_t poolIndex = 0;
        bool updateAfterBind = false;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        std::vector<VkDescriptorPoolSize> sizes;
        std::optional<std::uint32_t> variableCount;
        std::shared_ptr<const TableDescriptors> table;
    };
    SetAllocation Allocate(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind = false, std::optional<std::uint32_t> variableCount = std::nullopt);
    void Free(const SetAllocation& allocation, std::shared_ptr<const TableDescriptors> table = {}) noexcept;
    SetAllocation AllocateRewritten(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind = false, std::optional<std::uint32_t> variableCount = std::nullopt);
    void Recycle(VkDescriptorSetLayout layout, const SetAllocation& allocation) noexcept;
    // APS5_PROFILE_DRAW counters: layouts served from the map / created, sets allocated, pools opened.
    struct Stats {
        std::uint64_t layoutHits = 0;
        std::uint64_t layoutMisses = 0;
        std::uint64_t sets = 0;
        std::uint64_t pools = 0;
        std::uint64_t dedicatedPools = 0;
        std::uint64_t keptSets = 0;
        std::uint64_t reusedSets = 0;
    };
    Stats Counters() const;

private:
    Context context;
    PFN_vkDestroyDescriptorSetLayout destroyLayout;
    PFN_vkDestroyDescriptorPool destroyPool;
    PFN_vkResetDescriptorPool resetPool;
    PFN_vkFreeDescriptorSets freeDescriptorSets = nullptr;
    mutable AgcDriver::Mutex mutex;
    std::map<std::vector<std::uint32_t>, VkDescriptorSetLayout> layouts;
    struct Pool {
        VkDescriptorPool handle = VK_NULL_HANDLE;
        std::size_t liveSets = 0;
        bool exhausted = false;
    };
    std::vector<Pool> pools;
    std::vector<Pool> updateAfterBindPools;
    std::vector<VkDescriptorPool> dedicated;
    std::map<std::pair<VkDescriptorSetLayout, std::uint32_t>, std::vector<SetAllocation>> recycled;
    std::vector<SetAllocation> kept;
    Stats stats;
};

enum class ContentKeyComponent : std::size_t { Layout, BufferWord0, BufferWord1, BufferWord2, BufferWord3, Image, Sampler, ShaderData, Other, DrawTarget, Count };

using ContentKeyDiff = std::array<bool, static_cast<std::size_t>(ContentKeyComponent::Count)>;

struct ImageTableReuseCounts {
    std::uint64_t reused = 0;
    std::uint64_t resolved = 0;
    std::uint64_t replayed = 0;
    std::uint64_t writesKept = 0;
    std::uint64_t writesMade = 0;
};
ImageTableReuseCounts ImageTableReuses();
std::uint64_t SampledLookupKeyScans();

struct PresyncSurface {
    std::uint64_t address;
    std::uint64_t bytes;
    bool dccKeys;
};

// S1: the tag of the two-stage draw build (see ShaderResources).
struct DeferredDrawBuild {
    explicit DeferredDrawBuild() = default;
};

class ShaderResources {
public:
    ShaderResources(const Context& context, const ShaderRecompiler::RecompileResult& vertex, const ShaderRecompiler::RecompileResult& fragment, const ColorTarget& target, std::uint64_t indexAddress, std::size_t indexBytes);
    ShaderResources(const Context& context, std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes, std::span<const GuestMemorySnapshot> snapshots = {});
    ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots = {});
    // Two-stage build for dispatches (see build): with `deferred` the constructor runs stage A only,
    // which needs no device lock, and Complete() runs stage B under GuestMemory::GpuMutex; until
    // then no other member may be used. `compute` and `snapshots` must outlive Complete(). An
    // address-based shader (BDA tables) builds entirely in Complete(): its lease acquisition
    // reconciles imports and refreshes mirrors, which needs the lock.
    ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots, bool deferred, std::uint64_t dispatchThreads = 0);
    void Complete();
    // S1 (APS5_NO_SPLIT_DRAW_BUILD=1 off): the draw build in two stages, as for dispatches: the
    // constructor runs stage A only (no device lock; no guest bytes are read: host copies wait for
    // stage B), CompleteDraw() stage B under GuestMemory::GpuMutex. Address-based stages are refused
    // (their lease and mirrors need the lock): callers check UsesAddressTables first. `shaders` must
    // outlive CompleteDraw().
    ShaderResources(DeferredDrawBuild, const Context& context, std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes);
    void CompleteDraw();
    static bool StagesUseAddressTables(std::span<const CompiledShader> shaders);
    bool Completed() const { return completed; }
    ~ShaderResources();
    ShaderResources(const ShaderResources&) = delete;
    ShaderResources& operator=(const ShaderResources&) = delete;
    VkDescriptorSetLayout Layout() const;
    std::vector<VkDescriptorSetLayoutBinding> LayoutBindings() const;
    bool LayoutUpdateAfterBind() const;
    std::span<const VkDescriptorBindingFlags> LayoutBindingFlags() const { return layoutBindingFlags; }
    VkDescriptorSetLayout FragmentLayout() const { return fragment.layout; }
    bool SplitSets() const { return splitSets; }
    PipelineSetLayouts SetLayouts() const { return {_layout, fragment.layout, splitSets, layoutKey, fragmentLayoutKey}; }
    void Bind(VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout layout) const;
    struct DrawBindings {
        DrawBindings() = default;
        DrawBindings(const DrawBindings&) = delete;
        DrawBindings& operator=(const DrawBindings&) = delete;
        struct Snapshot {
            std::uint64_t address;
            std::shared_ptr<Buffer> buffer;
            VkDeviceSize offset = 0;
            std::size_t bytes = 0;
            std::span<std::byte> Bytes() const { return buffer->Bytes().subspan(static_cast<std::size_t>(offset), bytes); }
        };
        DescriptorCache* cache = nullptr;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout fragmentLayout = VK_NULL_HANDLE;
        DescriptorCache::SetAllocation fragmentAllocation;
        DescriptorCache::SetAllocation allocation;
        std::vector<Snapshot> snapshots;
        // Written V#s a compute template binds at a new base for this use (MovedDispatchBuffers with
        // written moves): their own upload of the new ranges, in place, whose writes this use notes
        // and marks (MarkMovedWrites); a snapshot without a buffer stands for such an element.
        std::shared_ptr<GuestBufferMemory> movedWrites;
        std::vector<std::pair<std::size_t, VkDescriptorBufferInfo>> movedWriteViews;
        ~DrawBindings();
    };
    struct MovedBuffer {
        std::size_t allocation;
        std::uint64_t address;
        std::size_t size;
        std::span<const std::uint32_t> words;
        bool written = false;
    };
    std::shared_ptr<DrawBindings> PrepareDrawBindings(Recorder& recorder, std::span<const MovedBuffer> moved = {}, bool movedOnly = false) const;
    std::optional<std::vector<MovedBuffer>> MovedReadOnlyBuffers(std::span<const CompiledShader> shaders, Recorder& recorder) const;
    enum class MoveRefusal : std::size_t { None, Shape, NotGuest, Written, Limit, Inaccessible, Overlap, Bindings, WrittenStaged, WrittenSize, WrittenAdjustment, WrittenOverlap, Count };
    struct MoveProof {
        MoveRefusal refusal = MoveRefusal::None;
        std::size_t buffers = 0;
        std::size_t bytes = 0;
        std::size_t writtenBuffers = 0;
        std::size_t writtenBytes = 0;
    };
    static const char* MoveRefusalName(MoveRefusal refusal);
    std::optional<std::vector<MovedBuffer>> MovedDispatchBuffers(const CompiledShader& shader, Recorder* recorder, std::size_t maxBytes, MoveProof& proof, bool movedWrites = false) const;
    // The pending-read and pending-write notes and the direct-write marks of a use's moved written
    // V#s (DrawBindings::movedWrites), as MarkGpuWrites makes them for the template's own ranges.
    static void MarkMovedWrites(Recorder& recorder, const DrawBindings& bindings);
    static bool MovedWritesOverlap(const DrawBindings& bindings, std::uint64_t address, std::size_t bytes);
    // Every guest buffer element of the build as {address, size, written}, in allocation order (the
    // verify of moved written V#s compares a full build's with the template's plus the moves).
    std::vector<std::array<std::uint64_t, 3>> GuestBufferElements() const;
    // The same with each element's allocation index as a fourth word.
    std::vector<std::array<std::uint64_t, 4>> GuestBufferAllocations() const;
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& GuestWrites() const { return guestMemory.Writes(); }
    static std::size_t MaxDrawSnapshotBytes();
    static std::size_t DrawUploadLimit();
    // The [draw-snapshots] counters of CopyDrawInput (vertex and index inputs, APS5_PROFILE_DRAW).
    enum class DrawInputPath : std::uint8_t { TrackedHit, OwnedHit, TrackedCopy, UntrackedCopy, Count };
    static void CountDrawInputPath(DrawInputPath path, std::size_t bytes, std::uint8_t ownedMiss, bool owned, bool armed);
    void WriteBack();
    // Deferred completion: MarkGpuWrites registers the results the recorded work leaves on the GPU
    // (storage images stay there; buffer ranges are noted so CPU reads wait); WriteBackBuffers runs
    // once the work completed.
    void MarkGpuWrites(Recorder& recorder);
    // As MarkGpuWrites for a dispatch that left its trailing barrier deferred (Recorder::
    // DeferDispatchBarrier): its staged copy-backs (and the direct-write marks they complete) are
    // recorded after the run's deferred barrier (Recorder::AfterRun); `self` keeps the object.
    void MarkGpuWritesInRun(Recorder& recorder, const std::shared_ptr<ShaderResources>& self);
    // The guest memory whose staged shadows this object's uses fill (a dispatch run member, see
    // Recorder::RunCommands).
    const void* MemoryOwner() const { return &guestMemory; }
    // Whether the build copies buffer regions on the GPU (staged shadows among them).
    bool HasGpuCopies() const { return guestMemory.HasGpuCopies(); }
    void WriteBackBuffers(std::uint64_t shaderAddress = 0);
    // Whether WriteBackBuffers has anything the CPU must see (copied written buffers, BDA faults).
    bool NeedsCompletion() const { return (bda != nullptr && faultChecks) || guestMemory.HasCopiedWrites(); }
    // Whether a written buffer was copied (its results reach guest memory by the CPU write-back).
    bool HasCopiedWrites() const { return guestMemory.HasCopiedWrites(); }
    bool HoldsLease() const { return guestMemory.HoldsLease(); }
    bool BdaReadOnly() const { return usesBda && bdaReadOnly; }
    bool UsesAddressTable() const { return usesBda; }
    std::size_t CopiedBytes() const { return guestMemory.CopiedBytes(); }
    bool WritesOverlap(std::uint64_t address, std::size_t bytes) const { return guestMemory.WritesOverlap(address, bytes); }
    bool CopiedWritesOverlap(std::uint64_t address, std::size_t bytes) const { return guestMemory.CopiedWritesOverlap(address, bytes); }
    // Whether a region the recorded work reads in place through a host import overlaps the range.
    bool ReadsOverlap(std::uint64_t address, std::size_t bytes) const;
    // The recorder's hazard tracker inputs (Recorder::NoteAccess): the written elements' guest
    // ranges, the regions read in place, and the storage images with whether each is written.
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& GpuWrites() const { return guestMemory.Writes(); }
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& WriteOnlyRanges() const { return writeOnlyRanges; }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> InPlaceReads() const { return guestMemory.InPlaceReads(); }
    void InPlaceReads(std::vector<std::pair<std::uint64_t, std::uint64_t>>& out) const { guestMemory.InPlaceReads(out); }
    std::vector<std::pair<VkImage, bool>> StorageImages() const;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> DeviceReads() const { return guestMemory.DeviceReads(); }
    void RunAccesses(std::vector<std::array<std::uint64_t, 3>>& out) const;
    void DeviceReads(std::vector<std::pair<std::uint64_t, std::uint64_t>>& out) const { guestMemory.DeviceReads(out); }
    bool ReadsImage(const StorageTexture* image) const;
    bool HasCachedStorageImages() const;
    std::uint32_t TableElements() const { return tableElements[0] + tableElements[1]; }
    const std::vector<std::uint32_t>& LayoutKey() const { return layoutKey; }
    const std::vector<std::uint32_t>& FragmentLayoutKey() const { return fragmentLayoutKey; }
    // Debug aid: each bound guest resource with the fraction of sampled bytes that are nonzero.
    std::string Describe() const;

    // Reuse across dispatches of identical content (see the resource cache in VulkanDevice): the
    // key is everything a compute build consumes from its compiled shader (variant, and per binding
    // kind, role, slot, count, guest descriptor words, written images, depth-compare samplers); push
    // constants are not part of it because every dispatch pushes its own. Reusable says whether this
    // object can serve a later dispatch with the same key: nothing needs completion work and every
    // guest buffer is read in place through a host import. Revalidate brings a reusable object up to
    // date for another use, or says it cannot be reused any more (a buffer's import went away, or a
    // texture or storage image the set references is no longer the one its memory maps to).
    // Without `dataWords` the ShaderData and FlattenedSrt descriptor words stay out of the key
    // (their count and size remain): a compute template then serves dispatches whose constants
    // differ, and the hit refreshes its data buffers with the dispatch's words (RefreshData).
    static std::vector<std::uint32_t> ContentKey(const CompiledShader& shader, bool dataWords = true, bool movableBuffers = false, bool movableWritten = false);
    static void AppendContentKey(const CompiledShader& shader, bool dataWords, bool movableBuffers, std::vector<std::uint32_t>& key, bool movableWritten = false);
    static void AppendContentKeyRoles(const CompiledShader& shader, bool dataWords, bool movableBuffers, std::vector<std::uint8_t>& roles);
    static bool DiffContentKeys(std::span<const std::uint32_t> previous, std::span<const std::uint32_t> key, bool draw, ContentKeyDiff& diff);
    // Records the shader's ShaderData and FlattenedSrt words into this object's data buffers
    // (vkCmdUpdateBuffer, a transfer write the caller's pre-dispatch barrier makes visible; a
    // buffer already holding the words is left alone). Returns whether anything was recorded. With
    // `recorder` the updates are one [gputime] range of class TemplateDataRefresh.
    bool RefreshData(VkCommandBuffer commands, const CompiledShader& shader, Recorder* recorder = nullptr);
    // Walks the write watch over every image surface of a completed object (no device lock: the
    // ranges are fixed by the build), so a Revalidate's collects on the same worker are memo hits.
    void PrecollectSurfaces() const;
    bool Reusable() const { return reusable; }
    std::size_t ReusableRefusal() const { return static_cast<std::size_t>(reusableRefusal); }
    void KeepStorageImageBytes(Recorder& recorder) const;
    const std::vector<std::uint64_t>& MultisampleSources() const { return multisampleSources; }
    static bool NeverReusable(std::span<const CompiledShader> shaders);
    // Whether a stage with a BDA page table stores through guest addresses (its SPIR-V; cached).
    static bool StagesStoreThroughAddresses(std::span<const CompiledShader> shaders);
    // Whether every storing BDA stage stores only through an extended instruction taking a pointer.
    static bool StagesStoreOnlyThroughExtInst(std::span<const CompiledShader> shaders);
    // APS5_PROFILE_DRAW ([bda-template] line): what became of address-table draw builds under
    // APS5_NO_BDA_TEMPLATES unset, by the first reason that kept one from being a template.
    enum class TemplateOutcome : std::size_t { Storing, StoringExtOnly, StorageTable, CopiedWrites, Multisample, LoopGuard, DepthStorage, TableElement, RefreshedPerUse, DataSize, SpaceRegions, Serial, Shared, NotShared, Inserted, Count };
    static const char* TemplateOutcomeName(TemplateOutcome outcome);
    static std::array<std::uint64_t, static_cast<std::size_t>(TemplateOutcome::Count)> TemplateOutcomes();
    static void CountTemplateOutcome(TemplateOutcome outcome);
    static std::uint64_t DepthGuardRefusals();
    static std::uint64_t GuestBufferBytes(const Context& context, std::span<const std::uint32_t> words);
    const std::vector<std::shared_ptr<Texture>>& SampledTextures() const { return textures; }
    // Whether a sampled texture of the set left the sampled cache (evicted, replaced, or made for
    // one use and never admitted): such a set is not kept by the resource cache, whose entries may
    // pin only textures the sampled cache accounts.
    bool BindsReleasedTexture() const;
    // The resource cache's pins on the set's sampled textures (Insert pins, every removal unpins).
    void PinTextures(bool pin) const;
    static std::size_t LookupLogCapacity();
    bool ChecksFaults() const { return bda != nullptr; }
    void CheckFaults(std::uint64_t shaderAddress = 0) const;
    bool InUse() const { return usesInFlight->load(std::memory_order_acquire) != 0; }
    bool ReusesAddressSpace() const { return addressReuse; }
    std::shared_ptr<const void> BeginUse();
    void CompleteUse();
    // Address-table template (APS5_NO_BDA_TEMPLATES unset): a read-only address-table draw build
    // (no store through BDA, no written buffer or image, no GPU copies) that later draws may use
    // while earlier uses are still in flight; each use takes its own descriptor set for changed data
    // words or moved read-only buffers, and the template itself is never written after its build.
    bool SharesUses() const;
    void EndUse() noexcept;
    // `shaders` are the stages the object was built from, in build order (a recorded draw's vertex
    // and fragment stages, or one compute stage): their bindings are walked like the build did.
    // How a Revalidate proved (or refused) the object, for the [recipe] line: the proof path taken
    // and, on a refusal, what failed.
    enum class ProofPath : std::size_t { Fast, OwnRefreshed, Full, Count };
    enum class ProofFailure : std::size_t { None, Imports, Evicted, Pending, Changed, Keys, Other, Count };
    struct ProofReport {
        ProofPath path = ProofPath::Fast;
        ProofFailure failure = ProofFailure::None;
    };
    bool Revalidate(std::span<const CompiledShader> shaders, ProofReport* report = nullptr, std::span<const GuestMemorySnapshot> snapshots = {});
    static bool EpochProofs();
    bool Revalidate(const CompiledShader& shader, ProofReport* report = nullptr) { return Revalidate(std::span<const CompiledShader>(&shader, 1), report); }
    // The proof a recipe hit runs on its template (design_cpu_final M4): Revalidate by that name.
    bool ProveCurrent(std::span<const CompiledShader> shaders, ProofReport* report = nullptr) { return Revalidate(shaders, report); }
    bool ProveCurrent(const CompiledShader& shader, ProofReport* report = nullptr) { return Revalidate(shader, report); }
    // FNV over the ShaderData/FlattenedSrt words, in binding order: of this object's data buffers
    // as the recorded work will find them (maintained by addDataBuffer and RefreshData), and of a
    // compiled shader's descriptors. A recipe hit refreshes the template iff the two differ.
    std::uint64_t DataWordsHash() const { return dataWordsHash; }
    static std::uint64_t DataWordsHash(const CompiledShader& shader);
    void PatchPushConstants(std::span<std::byte, PipelinePushConstantBytes> bytes) const {
        for (const auto& [position, adjustment] : pushPatches) bytes[position] = static_cast<std::byte>(adjustment);
    }
    // Whether RefreshData would record anything for `shader` (the per-word compare; verification).
    bool DataWordsDiffer(const CompiledShader& shader) const;
    // Why the fast proof of a Revalidate left the object to the full walk (the [rescache]
    // revalidate line's reasons); Count: it did not.
    enum class FastFail : std::size_t { NoRecord, Collect, Pending, Evicted, Changed, Keys, ClearedView, StorageKeys, Depth, Committed, Replaced, Audit, Count };
    // Why a Pending failure was left to the full walk instead of the own-object refresh (T1, see
    // refreshOwnObjects); Count: it was not.
    enum class OwnRefreshFallback : std::size_t { Disabled, Snapshot, Keys, ForeignView, SurfaceKey, NotImported, Uncached, Rerun, Count };
    // APS5_PROFILE_DRAW: the build's sub-phases ([resources] totals) and their times in
    // milliseconds (zero when not profiling), so the draw path can total them separately from
    // dispatches. bindingsMs covers the binding plan (stage A) and the image lookups (stage B);
    // prepareMs/completeMs are the two stages' totals.
    enum class BuildPhase : std::size_t { Bindings, Precollect, Upload, Descriptors, StageA, Images, Bda, StageB, Count };
    struct BuildTiming {
        double bindingsMs = 0;
        double uploadMs = 0;
        double descriptorsMs = 0;
        double prepareMs = 0;
        double completeMs = 0;
    };
    const BuildTiming& Timing() const { return timing; }
    // The image surfaces (address, bytes) whose stage-B lookup reads guest memory on the CPU and so
    // would wait, under the device lock, for recorded work writing them: a surface without a host
    // import, or a sampled one that stays a CPU snapshot (block compressed, or the storage-view path
    // is off). VulkanDevice::PrepareDispatch waits for that work before the lock (the pre-sync).
    // Usable right after the deferred constructor, for address-based builds too (it decodes the
    // compiled shader's bindings itself when stage A has not run), and on a completed object from
    // the resource cache (its Revalidate repeats the same lookups), without the device lock.
    std::vector<PresyncSurface> PresyncSurfaces() const;

private:
    friend class AgcDriver::VulkanDevice;
    bool refreshData(VkCommandBuffer commands, const CompiledShader& shader, Recorder* recorder);
    std::optional<std::size_t> shaderDataAllocation(const CompiledShader& shader) const;
    VkBuffer shaderDataBuffer(const CompiledShader& shader) const;
    void forgetShaderDataWords(const CompiledShader& shader);
    struct DescribedRange {
        const char* kind;
        std::uint64_t address;
        std::uint64_t bytes;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t format = 0;
        int tileMode = 0;
        std::uint64_t dccAddress = 0;
        bool imageTable = false;
    };
    std::vector<DescribedRange> describedRanges;
    struct Allocation {
        std::uint64_t address;
        std::size_t size;
        bool guest;
        std::unique_ptr<Buffer> buffer;
        ShaderRecompiler::DescriptorRole role = ShaderRecompiler::DescriptorRole::ShaderData;
        // Guest buffers: whether the shader may store to the range (see addGuestBuffer).
        bool written = true;
        bool writeOnly = false;
        // Data buffers: the words the buffer holds once the recorded work ran (see RefreshData).
        std::vector<std::uint32_t> dataWords;
        std::uint32_t adjustment = 0;
        std::int32_t pushByte = -1;
        std::int64_t dataAllocation = -1;
        std::uint32_t dataByte = 0;
    };
    struct DataPatch {
        std::size_t allocation;
        std::uint32_t byte;
        std::uint32_t adjustment;
    };
    void writeDataWords(VkCommandBuffer commands, std::size_t allocation, std::span<const std::uint32_t> words) const;

    struct Binding {
        VkDescriptorSetLayoutBinding layout;
        std::vector<std::size_t> allocations;
        std::vector<std::size_t> imageAllocations;
        VkDescriptorBindingFlags flags = 0;
        std::int32_t set = -1;
    };

    struct TableStage {
        ShaderRecompiler::ShaderStage stage = ShaderRecompiler::ShaderStage::Compute;
        const ShaderRecompiler::RecompileResult* program = nullptr;
        const ShaderRecompiler::DescriptorBinding* images = nullptr;
        const ShaderRecompiler::DescriptorBinding* storageImages = nullptr;
        std::size_t mapAllocation = std::numeric_limits<std::size_t>::max();
        std::vector<std::uint32_t> map;
        std::vector<ShaderRecompiler::ImageTableEntryPoison> poison;
        std::vector<std::string> messages;
        std::uint32_t elementBase = 0;
        std::uint32_t set = 0;
        std::size_t firstSampler = 0;
        std::size_t samplerCount = 0;
        std::vector<std::size_t> elementTextures;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> samplerRejections;
    };
    std::vector<TableStage> tableStages;
    std::array<std::size_t, 2> tableBindings{std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()};
    std::array<std::size_t, 2> storageTableBindings{std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()};
    std::array<std::uint32_t, 2> sampledTableCapacity{};
    std::array<std::uint32_t, 2> tableElements{};
    TableStage& tableStage(const CompiledShader& shader);
    static const ShaderRecompiler::DescriptorBinding* tableImages(std::span<const CompiledShader> shaders, ShaderRecompiler::ShaderStage stage);
    std::uint32_t rejectTableElement(TableStage& table, std::span<const std::uint32_t> words, const std::string& message);
    void addSamplerTable(const ShaderRecompiler::DescriptorBinding& binding, VkShaderStageFlags flags, TableStage& table);
    void resolveImageTables();
    void resolveStorageImageTables();
    void finishImageTables();

    // A host-imported buffer region the set reads in place, with its import's identity at build time.
    struct DirectRegion {
        std::uint64_t begin;
        std::uint64_t end;
        std::uint64_t serial;
        std::uint64_t regionEnd = 0;
        std::uint64_t committedEnd = 0;
    };

    // The build is two stages. A (buildPrepare, no device lock): the binding plan with its layout
    // entries, samplers, data buffers, the upload's prepare stage (imports already serving the
    // regions, read-only copies), the set layout and the descriptor set. B (buildComplete, under
    // GuestMemory::GpuMutex): the texture and storage image lookups (they refresh, upload and flush
    // through the recorder), the rest of the upload, the BDA objects, the descriptor writes and the
    // reusability record. build runs both.
    void build(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes);
    void buildPrepare(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes);
    void buildComplete();
    // APS5_PROFILE_DRAW: closes the current sub-phase of the build into the [resources] totals.
    double phase(BuildPhase which);
    // `written` is the element's DescriptorBinding::bufferWritten: a read-only element binds the
    // same way but is left out of the write set (no write-back, no pending-write note). `atomic`
    // is its bufferAtomic (see GuestBufferMemory::AddWritable).
    std::size_t addGuestBuffer(std::span<const std::uint32_t> words, const ColorTarget* target, std::uint64_t indexAddress, std::size_t indexBytes, bool written, bool atomic, bool read, std::uint32_t extent);
    std::uint64_t guestBufferBytes(std::span<const std::uint32_t> words) const;
    std::size_t addDataBuffer(std::span<const std::uint32_t> words);
    // Stage A: the layout entry of an image binding (samplers are taken at once, the sampler cache
    // locks itself); stage B looks the sampled textures and storage images up (resolveImageBinding).
    void addImageBinding(const ShaderRecompiler::DescriptorBinding& binding, VkShaderStageFlags flags, bool guardedUnnormalizedWrap);
    // Stage A: one record per planned image element (ImageRecord), in plan order: the decoded
    // descriptor and its surface size (computed once for the build), the write watch walked so stage
    // B's collects are memo hits (APS5_NO_PRECOLLECT=1 skips the pass entirely), and for a sampled
    // element the cache entry it maps to, taken now so stage B needs only the fastRevalidate
    // predicate under the lock (APS5_NO_STAGE_A_IMAGES=1 leaves every element to the full lookup).
    // Whether it ran.
    bool precollectImages();
    struct ImageRecord {
        bool sampled = false;
        // The descriptor decoded and its surface described; false leaves the element to stage B's
        // own decode (which reports the error).
        bool decoded = false;
        std::array<std::uint32_t, 8> words{};
        GuestTextureResource resource{};
        std::uint64_t guestBytes = 0;
        VkComponentMapping components{};
        // Sampled elements: the keys read in stage A and the collect made BEFORE stage B's checks
        // (the generation the entry moves to when they pass).
        DccKeys keys = DccKeys::Uncompressed;
        std::uint64_t generation = 0;
        // The cache entry's object, storage source, keys and generation as of stage A (texture null:
        // no entry, the full lookup makes one).
        std::shared_ptr<Texture> texture;
        std::shared_ptr<StorageTexture> source;
        DccKeys entryKeys = DccKeys::Uncompressed;
        std::uint64_t entryGeneration = 0;
        std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>> entryCommitted;
    };
    std::vector<ImageRecord> imageRecords;
    struct PreviousSampled {
        std::array<std::uint32_t, 8> words{};
        bool depthCompare = false;
        VkImageViewType multisampleViewType = VK_IMAGE_VIEW_TYPE_2D;
        bool firstSample = false;
        std::shared_ptr<Texture> texture;
    };
    PreviousSampled previousSampled;
    // The next record resolveImageBinding consumes (records follow the deferredImages order).
    std::size_t nextImageRecord = 0;
    // Stage B: the record's texture when the fastRevalidate predicate proves it current under the
    // lock and the cache still holds it; null sends the element to cachedTexture.
    std::shared_ptr<Texture> fastTexture(const ImageRecord& record);
    void resolveImageBinding(const ShaderRecompiler::DescriptorBinding& binding, Binding& item, std::span<const std::shared_ptr<Sampler>> shaderSamplers);
    void applyAnisoOverride();
    void forgetDeferredInputs();
    void release() noexcept;
    void prepareAddressBindings(std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots);
    VkDescriptorBufferInfo descriptor(Allocation& allocation);
    void noteReusable();
    bool proveBuild(std::span<const CompiledShader> shaders, ProofReport* report);
    void reportDescriptorCaches() const;
    // What a sampled texture was proved current against when the build (or the last full Revalidate)
    // looked it up, so the next Revalidate can repeat the proof from write stamps and the DCC keys
    // alone instead of repeating the lookup (see fastRevalidate). One record per textures[] entry; an
    // element without a record always takes the full lookup. Storage images need no record: their
    // proof comes from the image itself (its own surface and generation).
    struct ValidatedSurface {
        GuestTextureResource resource{};
        std::uint64_t bytes = 0;
        // Keys read at the lookup (a snapshot texture holds its clear texels while they are unchanged).
        DccKeys keys = DccKeys::Uncompressed;
        // Snapshot textures: write generation the content was known current at, always from a
        // collect issued before the checks that proved it.
        std::uint64_t generation = 0;
        // The collect of the check in progress, moved into `generation` once every element passed.
        std::uint64_t collected = 0;
        // Storage-sourced textures: the image the view follows (kept alive by the texture).
        const StorageTexture* source = nullptr;
        bool depth = false;
        bool valid = false;
        std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>> committed;
        std::array<std::uint64_t, 4> untracked{};
    };
    void captureValidation();
    // A surface of this object the fast proof found a foreign image pending over: the element
    // (index into textures, or into storageTextures when `storage`) whose own object stands in for
    // the full walk's lookup (refreshOwnObjects). `viewUncompressed`: a storage-sourced view under
    // uncompressed keys, which the lookup serves by its own source whenever FindPending names it.
    // `sourceEligible` (set by refreshOwnObjects on a resolved view): the lookup views the cached
    // image of the surface's key, i.e. the view's source, when FindPending names nothing.
    struct PendingOverlap {
        std::size_t element;
        bool storage;
        bool viewUncompressed;
        bool sourceEligible = false;
    };
    // `serialBefore` is StorageTexture::PendingSerial() loaded before any check of this Revalidate;
    // `refreshed` the surfaces refreshOwnObjects resolved earlier in the same Revalidate (empty on
    // the first run): a foreign image still pending over one of them (the own image's alias, an
    // image whose pending units lie outside the surface) is no failure, since the lookup's answer
    // for the surface is the own object its Refresh just brought current, whatever the registry
    // holds; `reason` receives why the proof failed (Count: it passed), `overlapping` the surfaces
    // of a Pending failure in element order (empty otherwise), `accepted` whether a foreign overlap
    // was accepted by either rule (the verify switch then runs the full walk beside the proof).
    bool fastRevalidate(std::uint64_t serialBefore, std::span<const PendingOverlap> refreshed, FastFail& reason, std::vector<PendingOverlap>& overlapping, bool& accepted);
    // T1 (design_cpu_final M3): brings the overlapping surfaces' own objects up to date exactly as
    // the full walk's lookups would, in the walk's binding order, without the lookups: a storage
    // image's Refresh (which stores the foreign images over its surface first), a view's source's
    // Refresh when the lookup would refresh the cached image of the view's surface key, nothing
    // when FindPending names the source itself (the lookup hits on it). Count, or the fallback
    // reason: a surface whose lookup could return another object (a snapshot texture, a view under
    // fast-clear keys, a foreign pending image the lookup would view, a surface key mapping to
    // another cached image, a surface outside the host imports, an object no longer cached), which
    // leaves the whole object to the full walk. Under GuestMemory::GpuMutex, as Revalidate is.
    OwnRefreshFallback refreshOwnObjects(std::span<const CompiledShader> shaders, std::span<PendingOverlap> overlapping);
    // Today's per-element walk (APS5_NO_EPOCH_REVALIDATE=1).
    bool fastRevalidateEach();
    Context context;
    std::vector<std::uint32_t> layoutKey;
    GuestBufferMemory guestMemory;
    std::size_t maxDrawSnapshotBytes = MaxDrawSnapshotBytes();
    std::unique_ptr<BdaResources> bda;
    std::vector<BdaResources::SrtGuardFaults> srtGuardFaults;
    bool usesBda = false;
    bool bdaReadOnly = true;
    std::vector<std::uint64_t> multisampleSources;
    bool usesFaultBuffer = false;
    bool faultChecks = false;
    VkDescriptorSetLayout _layout = VK_NULL_HANDLE;
    // Whether the layout is this object's own (no cache) and destroyed with it.
    bool ownsLayout = false;
    VkDescriptorSet _set = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    // The cache pool the set was allocated from, freed back to it on release.
    DescriptorCache::SetAllocation cacheAllocation;
    struct SetObjects {
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        bool ownsLayout = false;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        DescriptorCache::SetAllocation cacheAllocation;
    };
    bool splitSets = false;
    std::vector<std::uint32_t> fragmentLayoutKey;
    SetObjects fragment;
    struct TableWrites {
        bool recorded = false;
        bool written = false;
        std::uint32_t binding = 0;
        std::size_t firstImage = 0;
        std::size_t images = 0;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> runs;
    };
    std::array<TableWrites, 2> tableWrites;
    std::shared_ptr<const TableDescriptors> keptTable(std::size_t set, std::shared_ptr<const std::vector<std::weak_ptr<Texture>>>& held) noexcept;
    std::uint32_t setOf(const Binding& binding) const;
    std::vector<Allocation> allocations;
    std::vector<VkDescriptorBufferInfo> descriptorBuffers;
    std::vector<VkDescriptorImageInfo> descriptorImages;
    std::vector<VkWriteDescriptorSet> descriptorWrites;
    struct MovedWordBuffer {
        std::size_t allocation;
        std::vector<std::uint32_t> words;
        std::size_t size;
        std::shared_ptr<Buffer> buffer;
    };
    mutable std::vector<MovedWordBuffer> movedWordBuffers;
    mutable std::size_t movedWordBufferNext = 0;
    mutable std::vector<VkDescriptorPoolSize> drawPoolSizes;
    // Guest buffer elements bound read-only: each use of this object skips that many pending-write
    // notes (counted in MarkGpuWrites for the [buffers] line).
    std::size_t readOnlyBuffers = 0;
    std::chrono::steady_clock::time_point lastFullWalk = std::chrono::steady_clock::now();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writeOnlyRanges;
    bool drawBuild = false;
    std::uint64_t dispatchThreads = 0;
    std::vector<std::shared_ptr<Texture>> textures;
    std::vector<bool> textureFirstLayer;
    std::vector<bool> textureBaseLevel;
    std::vector<bool> textureSrgb;
    std::vector<std::shared_ptr<StorageTexture>> storageTextures;
    std::vector<VkImageView> multisampleStorageViews;
    std::vector<std::uint32_t> storageMips;
    std::vector<std::uint64_t> storageKeys;
    std::vector<bool> storageFirstLayer;
    std::vector<bool> storageWritten;
    std::vector<bool> storageAtomic;
    std::vector<bool> storageAtomic64;
    std::vector<std::shared_ptr<Sampler>> samplers;
    std::shared_ptr<Sampler> paddingSampler;
    struct SamplerSource {
        std::array<std::uint32_t, 4> words{};
        bool compareEnable = false;
        bool unnormalized = false;
        bool integerBorder = false;
        bool singleLevelImage = false;
        bool mipmappedImage = false;
    };
    std::vector<SamplerSource> samplerSources;
    bool reusable = false;
    TemplateOutcome reusableRefusal = TemplateOutcome::Count;
    bool addressReuse = false;
    bool bdaTemplate = false;
    bool sharedUses = false;
    bool reuseProved = false;
    std::shared_ptr<std::atomic<std::uint32_t>> usesInFlight = std::make_shared<std::atomic<std::uint32_t>>(0u);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> addressWrites;
    std::vector<DirectRegion> directRegions;
    std::vector<ValidatedSurface> validatedTextures;
    std::vector<std::array<std::uint64_t, 4>> storageUntracked;
    // The pending registry's serial at the last Revalidate that proved this object, taken before
    // its checks and kept only when unchanged after them (0: none): while it is still the serial,
    // no image other than the object's own sources has results pending over its surfaces and
    // regions, so those checks are skipped (see fastRevalidate).
    std::uint64_t pendingSerialSeen = 0;
    std::uint64_t provedEpoch = 0;
    std::uint64_t provedProof = 0;
    std::uint64_t provedPending = 0;
    bool revalidateFull(std::span<const CompiledShader> shaders, ProofReport* report);
    // The import table's identity when the direct regions' serials were last proved.
    HostImportsProof importsProof;
    // FNV-1a offset basis: the hash of no data buffers (DataWordsHash).
    std::uint64_t dataWordsHash = 14695981039346656037ull;
    // Async lane (VulkanDevice's dispatch record, APS5_NO_LANE_TEMPLATE_EDGE): the lane of the
    // last data refresh not yet ordered before the other lane's work (-1: none), and the lanes
    // whose recorded dispatches read the data buffers since (bit per lane).
    int dataWriterLane = -1;
    std::uint8_t dataReaderLanes = 0;
    void rehashDataWords();
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pushPatches;
    std::vector<DataPatch> dataPatches;
    BuildTiming timing;
    // Build state carried from stage A to stage B: the bindings in plan order, the image bindings
    // still to look up (index into `bindings`; the DescriptorBinding lives in the compiled shader),
    // the descriptor counts the set was sized for, and the compute stage of a deferred build.
    std::vector<Binding> bindings;
    std::vector<VkDescriptorBindingFlags> layoutBindingFlags;
    std::vector<VkDescriptorBindingFlags> fragmentBindingFlags;
    std::vector<std::uint32_t> refreshResourceKey;
    struct DeferredImages {
        const ShaderRecompiler::DescriptorBinding* binding;
        std::size_t index;
        std::size_t firstSampler = 0;
        std::size_t samplerCount = 0;
    };
    std::vector<DeferredImages> deferredImages;
    std::uint64_t storageBuffers = 0;
    bool updateAfterBind = false;
    std::uint32_t plannedSampledImages = 0;
    std::uint32_t plannedStorageImages = 0;
    // The compute constructor's shader and captured regions: the caller's objects, valid only until
    // the build (Complete() for a deferred one) is done, and reset then (forgetDeferredInputs).
    CompiledShader deferredCompute{};
    std::span<const GuestMemorySnapshot> deferredSnapshots;
    // The deferred build is address-based: everything runs in Complete().
    bool lockedBuild = false;
    // Stage A runs without the device lock (a deferred, non-address-based build).
    bool unlockedPrepare = false;
    bool deferredDraw = false;
    bool completed = false;
    std::chrono::steady_clock::time_point phaseStart;
    double phaseWaited = 0;
};

// Built ShaderResources by the content the build consumed (see ShaderResources::ContentKey), so a
// dispatch or recorded draw repeating an earlier one binds the earlier descriptor set instead of
// building one (import lookups, sampler and descriptor objects, guest buffer uploads). Only reusable
// objects (nothing to complete, every buffer read in place) are kept, and an entry serves a later
// use only after ShaderResources::Revalidate brought it up to date. Entries hold their object alive;
// the recorder keeps it as well while its batch runs. One LRU bound covers dispatch and draw entries
// together (APS5_RESOURCE_CACHE_ENTRIES, default 1024); the map has its own mutex and locks itself:
// VulkanDevice::PrepareDispatch probes it without GuestMemory::GpuMutex (a hit there is revalidated
// under the lock by the dispatch; inserts and Revalidate stay under it). Compute keys start with the
// stage, draw keys with a marker word, so the two never meet, and both name the device handle: the driver replaces
// the headless device with the windowed one while workers may still hold the old one, so a replaced
// device's entries (whose descriptor sets and pooled buffers belong to it) are unreachable by key
// and dropped by that device's Clear(). APS5_NO_RESOURCE_CACHE=1 (dispatches) and
// APS5_NO_DRAW_RESOURCE_CACHE=1 (draws) build every object as before; the device clears the cache at
// teardown before its descriptor caches go.
class ResourceCache {
public:
    using Key = std::vector<std::uint32_t>;
    std::shared_ptr<ShaderResources> Find(const Key& key);
    // The entry without touching it or counting a miss (the S1 preparer's look ahead).
    std::shared_ptr<ShaderResources> Peek(const Key& key);
    // `evicted`, when given, receives the objects the insert displaces (the entry replaced under the
    // key, the ones over the bound) instead of their being destroyed here: a caller under the GPU
    // mutex hands them to the recorder so the destruction runs off the lock (VulkanDevice::dispatch).
    void Insert(const Key& key, std::shared_ptr<ShaderResources> resources, std::vector<std::shared_ptr<ShaderResources>>* evicted = nullptr);
    // With `object`, erases the entry only while it still holds that object (a replacement made
    // meanwhile by another worker stays).
    void Remove(const Key& key, const ShaderResources* object = nullptr);
    void RemoveImage(const StorageTexture* image, std::vector<std::shared_ptr<ShaderResources>>* evicted = nullptr);
    // Drops every entry binding one of `textures` (sampled textures the sampled cache let go).
    void RemoveTextures(const std::unordered_set<const Texture*>& textures, std::vector<std::shared_ptr<ShaderResources>>* evicted = nullptr);
    bool TryRemoveImage(const StorageTexture* image, std::vector<std::shared_ptr<ShaderResources>>* evicted = nullptr);
    // A compute build the cache was not given (ShaderResources::ReusableRefusal, a TemplateOutcome): the fate its
    // key's next miss is charged to in the [rescache] compute line.
    void NoteNotInserted(const Key& key, std::size_t refusal);
    struct ComputeFates {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t neverSeen = 0;
        std::uint64_t evicted = 0;
        std::uint64_t evictedByDraws = 0;
        std::uint64_t dropped = 0;
        std::uint64_t removed = 0;
        std::uint64_t refusedStorage = 0;
        std::uint64_t refusedTexture = 0;
        std::uint64_t present = 0;
        std::array<std::uint64_t, 16> notReusable{};
        std::size_t entries = 0;
        std::size_t capacity = 0;
        std::uint64_t pinnedBytes = 0;
        std::uint64_t byteBudget = 0;
        bool split = false;
    };
    static bool ComputeBudgetRequested();
    ComputeFates ComputeFateCounts() const;
    static bool ComputeKey(const Key& key) { return !key.empty() && key[0] != 0xffffffffu; }
    // Moves the entry to the front of the LRU (a recipe hit uses its template without Find, and a
    // hot template must not age out under the recipes that depend on it); false when no entry.
    bool Touch(const Key& key);
    // Find and Touch calls so far (the [rescache] line: a recipe hit touches instead of finding).
    static std::uint64_t Finds();
    static std::uint64_t Touches();
    void Clear();
    std::size_t Size() const;
    std::size_t Capacity() const;
    void SetBase(std::size_t entries, std::size_t shrinkWindow);
    std::uint64_t TableElements() const;
    static std::uint64_t TableElementBudget();
    static constexpr std::size_t InitialEntries = 1024;
    static constexpr std::size_t MaximumEntries = 16384;

private:
    void erase(const Key& key);
    void drop(std::list<std::pair<Key, std::shared_ptr<ShaderResources>>>::iterator entry, std::vector<std::shared_ptr<ShaderResources>>* evicted);
    void noteMiss(const Key& key);
    void noteEvicted(const Key& key);
    void noteReturn(const Key& key);
    void noteUse(const Key& key);
    void removeImage(const StorageTexture* image, std::vector<std::shared_ptr<ShaderResources>>* evicted);
    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept {
            constexpr std::uint64_t Prime = 0x9e3779b97f4a7c15ull;
            std::uint64_t lanes[4] = {0x243f6a8885a308d3ull, 0x13198a2e03707344ull, 0xa4093822299f31d0ull, 0x082efa98ec4e6c89ull};
            const auto* words = key.data();
            const auto count = key.size();
            std::size_t at = 0;
            for (; at + 8u <= count; at += 8u) {
                for (std::size_t lane = 0; lane < 4u; ++lane) {
                    const auto value = static_cast<std::uint64_t>(words[at + lane * 2u]) | static_cast<std::uint64_t>(words[at + lane * 2u + 1u]) << 32u;
                    lanes[lane] = (lanes[lane] ^ value) * Prime;
                    lanes[lane] ^= lanes[lane] >> 29u;
                }
            }
            std::uint64_t hash = lanes[0] ^ (lanes[1] << 17u | lanes[1] >> 47u) ^ (lanes[2] << 31u | lanes[2] >> 33u) ^ (lanes[3] << 47u | lanes[3] >> 17u) ^ count;
            for (; at < count; ++at) hash = (hash ^ words[at]) * Prime;
            hash ^= hash >> 33u;
            hash *= 0xff51afd7ed558ccdull;
            hash ^= hash >> 33u;
            hash *= 0xc4ceb9fe1a85ec53ull;
            hash ^= hash >> 33u;
            return static_cast<std::size_t>(hash);
        }
    };
    mutable AgcDriver::Mutex mutex;
    std::list<std::pair<Key, std::shared_ptr<ShaderResources>>> entries;
    std::unordered_map<Key, decltype(entries)::iterator, KeyHash> index;
    // The entries by each sampled texture they bind (once per entry): RemoveTextures without a scan.
    std::unordered_multimap<const Texture*, decltype(entries)::iterator> textureIndex;
    void indexTextures(decltype(entries)::iterator entry, bool add);
    std::size_t capacity = 0;
    bool fixedCapacity = false;
    // Address-table builds (ReusesAddressSpace) have a bound of their own, so the other entries'
    // churn (dispatch keys) does not evict them before the next frame draws them again, and they
    // never displace the others: APS5_ADDRESS_TEMPLATE_ENTRIES (default 2048).
    std::size_t addressEntries = 0;
    std::size_t addressCapacity = 0;
    std::size_t computeEntries = 0;
    std::size_t computeCapacity = 0;
    std::uint64_t computePinnedBytes = 0;
    std::uint64_t computeByteBudget = 0;
    bool computeSplit() const;
    enum Fate : std::uint8_t { FateUnknown, FatePresent, FateEvicted, FateEvictedByDraw, FateDropped, FateRemoved, FateRefusedStorage, FateRefusedTexture, FateNotReusable };
    std::unordered_map<std::size_t, std::uint8_t> computeFates;
    ComputeFates fateCounts;
    std::chrono::steady_clock::time_point fateReport = std::chrono::steady_clock::now();
    void noteFate(const Key& key, std::uint8_t fate);
    void noteComputeLookup(const Key& key, bool hit);
    bool computeEntry(const std::pair<Key, std::shared_ptr<ShaderResources>>& entry) const;
    std::list<std::size_t> evictedOrder;
    std::unordered_map<std::size_t, decltype(evictedOrder)::iterator> evictedKeys;
    std::size_t evictionsSinceCheck = 0;
    std::size_t returnsSinceCheck = 0;
    std::unordered_set<std::size_t> usedKeys;
    std::size_t usesSinceCheck = 0;
    static constexpr std::size_t InitialShrinkWindow = 8;
    static constexpr std::size_t MaximumShrinkWindow = 1024;
    std::size_t baseEntries = InitialEntries;
    std::size_t baseShrinkWindow = InitialShrinkWindow;
    std::size_t shrinkWindow = InitialShrinkWindow;
    bool shrankLast = false;
};

// The process-wide cache the device and the draw path share (keys name the device; a device clears
// it when it goes). Never destroyed: entries belong to the device, not to static teardown.
ResourceCache& SharedResourceCache();

}

#endif
