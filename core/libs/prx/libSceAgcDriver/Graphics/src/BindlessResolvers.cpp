#include "prx/libSceAgcDriver/Graphics/include/BindlessResolvers.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestSamplerResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "DeviceImageTableAbi.hpp"
#include "ImageTableAbi.hpp"
#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics {

namespace {

namespace Abi = ShaderRecompiler::DeviceImageTableAbi;

struct Interfaces {
    std::mutex mutex;
    std::map<std::uint64_t, std::shared_ptr<const BindlessInterface>> entries;
};

Interfaces& Registry() {
    static Interfaces registry;
    return registry;
}

struct Prepared {
    std::array<std::uint32_t, Abi::DescriptorWords> words{};
    GuestTextureResource resource{};
    std::uint64_t guestBytes = 0;
    bool firstLayer = false;
    bool depthCompare = false;
    std::uint32_t mode = 0;
    bool runtimeSampler = false;
    std::array<std::uint32_t, Abi::SamplerWords> sampler{};
};

std::string Hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

ShaderRecompiler::DescriptorImageShape ShapeOf(ShaderRecompiler::RdnaImageDimension dimension) {
    using Dimension = ShaderRecompiler::RdnaImageDimension;
    using Shape = ShaderRecompiler::DescriptorImageShape;
    switch (dimension) {
        case Dimension::Dim1D: return Shape::Image1D;
        case Dimension::Dim1DArray: return Shape::Image1DArray;
        case Dimension::Dim2D: return Shape::Image2D;
        case Dimension::Dim2DArray: return Shape::Image2DArray;
        case Dimension::Dim3D: return Shape::Image3D;
        case Dimension::Dim2DMsaa: return Shape::Image2DMultisample;
        default: break;
    }
    throw std::runtime_error("AGC graphics: a bindless image mode has no descriptor image shape");
}

}

void RegisterBindlessInterface(std::uint64_t id, BindlessInterface registered) {
    auto& registry = Registry();
    std::lock_guard lock(registry.mutex);
    auto& slot = registry.entries[id];
    if (slot != nullptr) {
        Require(slot->image == registered.image && slot->modes == registered.modes, "two different image interfaces share the bindless interface id " + Hex(id));
        return;
    }
    slot = std::make_shared<const BindlessInterface>(std::move(registered));
}

std::shared_ptr<const BindlessInterface> FindBindlessInterface(std::uint64_t id) {
    auto& registry = Registry();
    std::lock_guard lock(registry.mutex);
    const auto found = registry.entries.find(id);
    return found == registry.entries.end() ? nullptr : found->second;
}

bool GuestBindlessMemory::Read(std::uint64_t address, std::span<std::uint32_t> words) {
    const auto bytes = words.size() * sizeof(std::uint32_t);
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) return false;
    GuestMemory::ReadCommittedUnsynced(address, std::as_writable_bytes(words));
    return true;
}

std::uint64_t GuestBindlessMemory::Collect(std::uint64_t address, std::size_t bytes) {
    return GuestMemory::CollectWrites(address, bytes);
}

bool GuestBindlessMemory::Unchanged(std::uint64_t address, std::size_t bytes, std::uint64_t generation) {
    return generation != 0 && GuestMemory::UnchangedSince(address, bytes, generation);
}

std::uint64_t GuestBindlessMemory::MappingSerial() {
    return GuestMemory::ForgetSerial();
}

bool GuestBindlessMemory::Mapped(std::uint64_t address, std::size_t bytes) {
    return GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes) || !GuestMemory::CommittedRanges(address, bytes).empty();
}

DriverBindlessResolver::DriverBindlessResolver(const Context& context, InterfaceLookup lookup) : context(context), lookup(std::move(lookup)) {
    Require(this->lookup != nullptr, "the bindless resolver needs an interface lookup");
}

BindlessPrepared DriverBindlessResolver::Prepare(const DeviceImageKeys::Key& key) {
    using Materializer = ShaderRecompiler::ResourceMaterializer;
    const auto id = static_cast<std::uint64_t>(key[Abi::InterfaceWord]) | (static_cast<std::uint64_t>(key[Abi::InterfaceWord + 1u]) << 32u);
    const auto found = lookup(id);
    if (found == nullptr) throw std::runtime_error("AGC graphics: a bindless key names the image interface " + Hex(id) + ", which no compiled program registered");
    ShaderRecompiler::DescriptorValue descriptor;
    descriptor.dwordCount = Abi::DescriptorWords;
    std::copy_n(key.begin(), Abi::DescriptorWords, descriptor.dwords.begin());
    const auto match = Materializer::TryRuntimeImageMode(found->image, descriptor, found->modes);
    if (!match.mode.has_value()) {
        const auto reason = ShaderRecompiler::ImageTableAbi::PoisonReasonName(match.reason);
        throw std::runtime_error("AGC graphics: the T# matches no runtime mode of its image interface (" + std::string(reason) + (match.detail.empty() ? std::string() : ": " + match.detail) + ")");
    }
    const auto& mode = found->modes.at(*match.mode);
    BindlessPrepared prepared;
    prepared.key = key;
    if (mode.constantSwizzle) {
        prepared.settled = DeviceImageValue{0u, *match.mode, 0u, 0u};
        return prepared;
    }
    if (found->image.resourceClass == ShaderRecompiler::ImageResourceClass::Storage) throw std::runtime_error("AGC graphics: a storage image read or written through the bindless heap is not implemented (storage tables stay on their per-draw binding)");
    if (mode.dimension == ShaderRecompiler::RdnaImageDimension::Dim2DMsaa || mode.dimension == ShaderRecompiler::RdnaImageDimension::Dim2DMsaaArray) throw std::runtime_error("AGC graphics: a multisampled image read through the bindless heap is not implemented");
    if ((mode.emulatedCompare & ShaderRecompiler::EmulatedCompare::Enabled) != 0u) throw std::runtime_error("AGC graphics: comparison sampling of a color texture through the bindless heap is not implemented");
    auto state = std::make_shared<Prepared>();
    std::copy_n(key.begin(), Abi::DescriptorWords, state->words.begin());
    state->resource = DecodeTextureResource(state->words);
    const auto shape = ShapeOf(mode.dimension);
    state->firstLayer = shape == ShaderRecompiler::DescriptorImageShape::Image2D && state->resource.dimension == TextureDimension::k2DArray;
    if (!state->firstLayer && !MatchesGuestDimension(shape, state->resource.dimension)) throw std::runtime_error("AGC graphics: guest texture dimension disagrees with the bindless mode's image shape (shape " + std::to_string(static_cast<int>(shape)) + ", dimension " + std::to_string(static_cast<int>(state->resource.dimension)) + ")");
    static_cast<void>(ResolveTextureFormat(state->resource.format));
    state->guestBytes = DescribeSurface(state->resource).guestBytes;
    const auto address = state->resource.baseAddress;
    const auto bytes = static_cast<std::size_t>(state->guestBytes);
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes) && GuestMemory::CommittedRanges(address, bytes, false).empty()) throw std::runtime_error("AGC graphics: the bindless entry's surface at " + Hex(address) + " (" + Hex(state->guestBytes) + " bytes) is not mapped");
    state->depthCompare = mode.depthCompare;
    state->mode = *match.mode;
    state->runtimeSampler = (key[Abi::SamplerClassWord] & Abi::RuntimeSampler) != 0u;
    std::copy_n(key.begin() + Abi::SamplerWord, Abi::SamplerWords, state->sampler.begin());
    prepared.address = address;
    prepared.bytes = state->guestBytes;
    prepared.generation = GuestMemory::CollectWrites(address, bytes);
    prepared.state = std::move(state);
    return prepared;
}

BindlessResolution DriverBindlessResolver::Commit(const BindlessPrepared& prepared) {
    const auto state = std::static_pointer_cast<Prepared>(prepared.state);
    Require(state != nullptr, "a bindless entry reached commit without its prepared state");
    auto texture = CachedTableTexture(context, state->words, state->resource, state->guestBytes, state->depthCompare);
    Require(texture != nullptr, "the texture cache returned no texture for a bindless entry");
    BindlessResolution resolution;
    resolution.view = state->firstLayer ? texture->FirstLayerView() : texture->View();
    Require(resolution.view != VK_NULL_HANDLE, "a bindless entry's texture has no view of the shape its mode reads");
    resolution.layout = texture->Layout();
    resolution.value = DeviceImageValue{0u, state->mode, 0u, 0u};
    resolution.address = prepared.address;
    resolution.bytes = prepared.bytes;
    resolution.generation = prepared.generation;
    resolution.perUse = texture->IsDepthCopy() || texture->RefreshedPerUse();
    if (state->runtimeSampler) {
        std::shared_ptr<Sampler> sampler;
        if (context.samplerCache != nullptr) {
            sampler = context.samplerCache->Get(context, state->sampler, state->depthCompare);
        } else {
            auto resource = DecodeSamplerResource(state->sampler);
            resource.compareEnable = state->depthCompare;
            sampler = std::make_shared<Sampler>(context, resource);
        }
        resolution.sampler = sampler->Handle();
        resolution.samplerOwner = std::move(sampler);
    }
    resolution.owner = std::move(texture);
    return resolution;
}

bool DriverBindlessResolver::Current(const BindlessResolution& resolution) {
    const auto texture = std::static_pointer_cast<Texture>(resolution.owner);
    return texture == nullptr || texture->SampledCacheState() != Texture::CacheState::Released;
}

}
