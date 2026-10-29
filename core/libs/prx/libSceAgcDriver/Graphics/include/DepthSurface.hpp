#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEPTHSURFACE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEPTHSURFACE_HPP

#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/SampleOrder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

class Texture;
class StorageTexture;

VkImageView DepthSurfaceView(const Context& context, const DepthTarget& target, bool depthWritten = true);
// Under GuestMemory::GpuMutex: the HTILE bytes whose store-back read in DepthSurfaceView would wait
// for a pending GPU write (no GPU expansion knows the words), 0 when it would not. The draw's
// HTILE pre-sync (S1) waits for that writer without the mutex first.
std::size_t HtileStoreBackWaits(const Context& context, const DepthTarget& target);
// APS5_PROFILE_DRAW diagnostics of the HTILE writers the pre-sync waits for: the program is remembered
// (HtileWriterProgram: its next dispatches print their code and user data, Dispatch.cpp), and the
// store-back read after the wait reports the words it found for that writer ([htile] line).
void NoteHtileWriter(std::uint64_t program, std::uint64_t htileAddress);
bool HtileWriterProgram(std::uint64_t program);
SampleOrderTarget DepthSurfaceOrderTarget(const Context& context, const DepthTarget& target);
DepthSampleLocations DepthSurfaceSampleLocations(const Context& context, const DepthTarget& target);
enum class DepthPlaneScope { Overlapped, Whole };
std::vector<std::pair<std::uint64_t, std::uint64_t>> FlushDepthPlanes(std::uint64_t address, std::size_t bytes, DepthPlaneScope scope = DepthPlaneScope::Overlapped);
bool DepthPlanesStale();
struct DepthPlaneStoreStatistics {
    std::uint64_t stores = 0;
    std::uint64_t units = 0;
    std::uint64_t bytes = 0;
};
DepthPlaneStoreStatistics DepthPlaneStoreCounts();
void NoteDepthSurfaceWrite(const Context& context, const DepthTarget& target);
bool SampleLocationsCompatibleDepth(const Context& context, VkFormat format, std::uint32_t samples);
std::uint64_t DepthSurfacesSerial();
std::uint64_t DepthSurfacesReleased();
std::uint64_t DepthSliceBytes(VkExtent2D extent, std::uint32_t bytesPerTexel);
bool DirectDepthCopy(const Context& context, VkFormat depthFormat, VkImageAspectFlagBits aspect, VkFormat colorFormat);
void SetDirectDepthCopy(bool enabled);
std::uint64_t DirectDepthCopyCount();
std::uint64_t MultisampledDepthBytes_nid_no_patch(VkExtent2D extent, std::uint32_t bytesPerTexel, std::uint32_t samples);
void RequireNoResidentDepthRead_nid_no_patch(std::uint64_t address, std::size_t bytes);
bool HasResidentDepthGuards_nid_no_patch();
void RetireUnmappedDepth_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void RequireNoNativeDepthFeedback_nid_no_patch(const Context& context, const DepthTarget& target, std::span<const std::shared_ptr<Texture>> sampled);
void ClearDepthSurfaces(VkDevice device);
void RetireDepthSurfaces(VkDevice device, std::uint64_t address, std::uint64_t bytes);
struct WrittenStorageExtent {
    std::uint64_t address;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t bytesPerElement;
    std::array<std::uint32_t, 3> block;
    std::uint32_t blocksPerRow;
};
void RetireCoveredDepthSurfaces(VkDevice device, std::span<const std::pair<std::uint64_t, std::uint64_t>> written, std::span<const WrittenStorageExtent> extents);
bool DepthSurfaceAt(std::uint64_t address);
bool DepthSurfaceOverlaps(VkDevice device, std::uint64_t address, std::uint64_t bytes);
bool DepthSurfaceAt(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource);
void SeedStorageFromDepth(const Context& context, const std::shared_ptr<StorageTexture>& storage);
bool DepthSurfaceServesStorage(const Context& context, const StorageTexture& storage);
void ValidateStorageDepthWrites(const Context& context, std::span<const std::shared_ptr<StorageTexture>> storage);
void MarkStorageDepthWrites(const Context& context, const std::shared_ptr<StorageTexture>& storage);
std::uint64_t HtileDepthClearAddress(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const std::array<std::uint32_t, 3>& numThreads);
// Whether the code starts with the HTILE mask kernel (whatever its user data): the [htile] writer attribution.
bool HtileMaskKernelCode(std::span<const std::uint32_t> code);
// The buffer-descriptor variant of the HTILE mask kernel (the game's 0x141d4c500): each thread loads one 16-byte
// record through a 32_32_32_32_UINT structured buffer (s[0:3], stride 16, index = group * 64 + thread), stores
// (record & mask) | value with the mask in s[8:11] and the value in s[4:7], all four words alike. The range it
// rewrites and its mask/value, when the dispatch is exactly that kernel with a plain descriptor
// (APS5_NO_HTILE_BUFFER_KERNEL=1: never).
struct HtileBufferMaskClear {
    std::uint64_t address = 0;
    std::size_t bytes = 0;
    std::uint32_t mask = 0;
    std::uint32_t value = 0;
};
std::optional<HtileBufferMaskClear> MatchHtileBufferMaskClear(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderComputeStageInfo& compute, std::span<const std::uint32_t> packet, std::uint64_t indirectArguments);
std::size_t HtileDepthClearBytes_nid_no_patch(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderComputeStageInfo& compute, std::span<const std::uint32_t> packet, std::uint64_t indirectArguments);
bool NoteHtileDepthClear(std::uint64_t htileAddress, std::size_t bytes, std::size_t nativeBytes = 0u);
class Recorder;
// The HTILE mask clear kernel (HtileDepthClearAddress) recorded over [address, address + bytes): every word becomes
// (word & mask) | value on the GPU. Noted under the GPU lock right after the dispatch is recorded, with the recorder's
// write note, so a depth store-back can know the HTILE words without reading them back (APS5_NO_HTILE_GPU_EXPANSION=1: off).
void NoteHtileMaskOnGpu(const Context& context, std::uint64_t address, std::size_t bytes, std::uint32_t mask, std::uint32_t value);
// The device's GPU fill of a uniform dword over guest memory (VulkanDevice::FillBuffer), per recorder; null removes it.
using HtileFillOnGpu = std::function<bool(std::uint64_t address, std::size_t bytes, std::uint32_t word)>;
void SetHtileFillOnGpu(const Recorder* recorder, HtileFillOnGpu fill);
void ArmHtileDepthClear(std::uint64_t htileAddress, std::size_t bytes);
void QueueHtileDepthClear(std::uint64_t htileAddress, std::size_t bytes);
void CompleteHtileDepthClear(std::uint64_t htileAddress, std::size_t bytes);
struct MemoryValueFill {
    std::uint64_t address;
    std::size_t bytes;
    std::uint64_t valueAddress;
};
std::optional<MemoryValueFill> MatchMemoryValueFill(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const std::array<std::uint32_t, 3>& numThreads, const std::array<std::uint32_t, 3>& groups);
void NoteMemoryValueFill(const MemoryValueFill& fill);
bool DepthSurfaceHolds(const Context& context, const GuestTextureResource& resource, const Texture* texture);
void NoteDepthMetadataFill(std::uint64_t address, std::size_t bytes, std::uint32_t pattern);
VkImageAspectFlags HtileFillClears(std::uint32_t pattern, bool stencilInHtile);
std::uint64_t HtileMetadataBytes(VkExtent2D extent);
bool HtileFillCovers(std::uint64_t htile, VkExtent2D extent, std::uint64_t address, std::size_t bytes);
std::shared_ptr<Texture> DepthSurfaceTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components, VkImageViewType multisampleViewType = VK_IMAGE_VIEW_TYPE_2D);
std::size_t DepthPlaneCopyCount();
VkDeviceSize DepthPlaneCopyBytes();
VkFormat HostDepthFormat(const Context& context, VkFormat format);
float HostDepthBiasConstant(const Context& context, VkFormat format, float constant);
std::shared_ptr<StorageTexture> DepthSurfaceStorage(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint32_t mip);
bool DepthSurfaceStoreBack(const Context& context, const StorageTexture& storage);
std::array<std::uint32_t, 8> DepthCopySourceWords(const DepthTarget& target);
class DepthCopySourceScope {
public:
    explicit DepthCopySourceScope(bool active);
    ~DepthCopySourceScope();
    DepthCopySourceScope(const DepthCopySourceScope&) = delete;
    DepthCopySourceScope& operator=(const DepthCopySourceScope&) = delete;

private:
    bool previous;
};

struct StorageImageClear {
    std::array<std::uint32_t, 8> image{};
    std::array<std::uint32_t, 4> color{};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};
std::optional<StorageImageClear> MatchStorageImageClear(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderComputeStageInfo& compute, const std::array<std::uint32_t, 3>& groups);
std::uint64_t StorageImageClearAddress(const StorageImageClear& clear);
struct DepthPlaneOverwrite {
    std::array<std::uint32_t, 8> image{};
    GuestTextureResource view{};
    std::uint32_t bits = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t raw = 0;
    std::uint64_t colorAddress = 0;
    std::uint32_t colorBits = 0;
    float depth = 0.0f;
};
class DepthPlaneClear {
public:
    explicit DepthPlaneClear(const StorageImageClear& clear);
    ~DepthPlaneClear();
    DepthPlaneClear(const DepthPlaneClear&) = delete;
    DepthPlaneClear& operator=(const DepthPlaneClear&) = delete;
    bool Active() const { return installed; }

private:
    DepthPlaneOverwrite value;
    DepthPlaneOverwrite* previous = nullptr;
    bool installed = false;
};
struct DepthClearStatistics {
    std::uint64_t matched = 0;
    std::uint64_t taken = 0;
    std::uint64_t copiesSkipped = 0;
    std::uint64_t clears = 0;
    std::uint64_t bytesSkipped = 0;
    std::array<std::uint64_t, 8> declined{};
};
DepthClearStatistics DepthClearCounts();
void SetDepthPlaneClearForTest(bool enabled);

}

#endif
