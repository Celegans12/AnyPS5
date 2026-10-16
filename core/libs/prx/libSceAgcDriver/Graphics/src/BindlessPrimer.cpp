#include "prx/libSceAgcDriver/Graphics/include/BindlessPrimer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BindlessTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "ImageTableAbi.hpp"
#include <cstdlib>
#include <limits>
#include <vector>
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"

namespace AgcDriver::Graphics {

BindlessPrimer::BindlessPrimer(BindlessTable& table) : table(table) {
}

BindlessPrimer::~BindlessPrimer() = default;

void BindlessPrimer::Prime(std::span<const BindlessPrimedImage> entries) {
    std::lock_guard lock(mutex);
    for (const auto& entry : entries) {
        DeviceImageValue value;
        value.fault = entry.fault;
        if (entry.fault == 0u) {
            Require(entry.texture != nullptr && entry.view != VK_NULL_HANDLE, "a primed bindless image has no view");
            auto image = images.find({entry.view, entry.layout});
            if (image == images.end()) image = images.emplace(std::pair{entry.view, entry.layout}, std::pair{table.Heap().AllocateImage(DescriptorHeap::Array::SampledImages, entry.view, entry.layout), entry.texture}).first;
            value.slot = image->second.first;
            value.mode = entry.mode;
            if (entry.sampler != nullptr) {
                auto sampler = samplers.find(entry.sampler->Handle());
                if (sampler == samplers.end()) sampler = samplers.emplace(entry.sampler->Handle(), std::pair{table.Heap().AllocateSampler(entry.sampler->Handle()), entry.sampler}).first;
                value.samplerSlot = sampler->second.first;
            }
        }
        if (const auto existing = table.Find(entry.key); existing.has_value() && *existing == value) continue;
        Require(table.Insert(entry.key, value) != DeviceImageKeys::Insertion::Full, "the bindless image key table is full, which is not implemented");
    }
}

void ShaderResources::PrimeBindless(BindlessPrimer& primer) const {
    namespace Abi = ShaderRecompiler::ImageTableAbi;
    namespace Bindless = ShaderRecompiler::BindlessImageAbi;
    static const bool skip = std::getenv("APS5_BINDLESS_NO_PRIMER") != nullptr;
    if (skip) return;
    constexpr auto none = std::numeric_limits<std::size_t>::max();
    const auto refused = [](std::uint32_t code) { return (code & Abi::PoisonFlag) != 0u ? (code & ~Abi::PoisonFlag) + 1u : 0u; };
    std::vector<BindlessPrimedImage> entries;
    for (const auto& table : tableStages) {
        if (table.program == nullptr || table.images == nullptr || !table.program->bindlessHeap) continue;
        const Binding* samplerTable = nullptr;
        const Binding* directSamplers = nullptr;
        for (const auto& binding : table.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::SamplerTable && binding.role != ShaderRecompiler::DescriptorRole::GuestSamplers) continue;
            for (const auto& item : bindings) {
                if (item.layout.binding != binding.binding || item.layout.descriptorType != VK_DESCRIPTOR_TYPE_SAMPLER) continue;
                if (binding.role == ShaderRecompiler::DescriptorRole::SamplerTable) samplerTable = &item;
                else directSamplers = &item;
            }
        }
        const auto elementFaults = table.map.at(Abi::HeaderElementFaults);
        const auto samplerFaults = table.map.at(Abi::HeaderSamplerFaults);
        for (const auto& primed : table.program->bindlessPrimer) {
            BindlessPrimedImage entry;
            entry.key = primed.key;
            entry.mode = primed.mode;
            entry.fault = primed.fault;
            if (entry.fault == 0u) {
                Require(primed.element < table.elementTextures.size(), "a bindless primer entry names an image table element the table does not hold");
                entry.fault = refused(table.map.at(elementFaults + primed.element));
            }
            if (entry.fault == 0u) {
                const auto index = table.elementTextures[primed.element];
                Require(index != none && index < textures.size(), "a bindless primer entry names an image table element without a texture");
                entry.texture = textures[index];
                entry.view = textureFirstLayer[index] ? entry.texture->FirstLayerView() : entry.texture->View();
                entry.layout = entry.texture->Layout();
            }
            if (entry.fault == 0u && primed.samplerElement != Bindless::NoSamplerElement && primed.directSampler) {
                Require(directSamplers != nullptr && primed.samplerElement < directSamplers->imageAllocations.size(), "a bindless primer entry names a direct sampler the sampler binding does not hold");
                const auto index = directSamplers->imageAllocations[primed.samplerElement];
                Require(index != none && index < samplers.size(), "a bindless primer entry names a direct sampler without a sampler");
                entry.sampler = samplers[index];
            } else if (entry.fault == 0u && primed.samplerElement != Bindless::NoSamplerElement) {
                Require(samplerTable != nullptr && primed.samplerElement < samplerTable->imageAllocations.size(), "a bindless primer entry names a sampler table element the table does not hold");
                entry.fault = refused(table.map.at(samplerFaults + primed.samplerElement / Abi::SamplerEntryElements));
                if (entry.fault == 0u) {
                    const auto index = samplerTable->imageAllocations[primed.samplerElement];
                    Require(index != none && index < samplers.size(), "a bindless primer entry names a sampler table element without a sampler");
                    entry.sampler = samplers[index];
                }
            }
            if (entry.fault != 0u) {
                entry.texture.reset();
                entry.sampler.reset();
                entry.view = VK_NULL_HANDLE;
                entry.mode = 0u;
            }
            entries.push_back(std::move(entry));
        }
    }
    primer.Prime(entries);
}

}
