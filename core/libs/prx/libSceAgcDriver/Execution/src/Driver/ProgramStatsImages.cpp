#include "prx/libSceAgcDriver/Execution/include/Driver/ProgramStats.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "Recompiler.hpp"

namespace AgcDriver::DriverDetail {

std::vector<StatsImage> ProgramStatsImages(const ShaderRecompiler::RecompileResult& program, bool written) {
    std::vector<StatsImage> images;
    for (const auto& binding : program.bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) continue;
        if (written ? binding.kind != ShaderRecompiler::DescriptorKind::StorageImage : binding.kind != ShaderRecompiler::DescriptorKind::SampledImage && binding.kind != ShaderRecompiler::DescriptorKind::StorageImage) continue;
        for (std::uint32_t element = 0; element < binding.count && images.size() < 64; ++element) {
            if (written && element < binding.imageWritten.size() && !binding.imageWritten[element]) continue;
            if (binding.guestDescriptor.size() < (static_cast<std::size_t>(element) + 1) * 8) continue;
            const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8, 8);
            if (words[0] == 0u && (words[1] & 0xffu) == 0u) continue;
            try {
                const auto resource = Graphics::DecodeTextureResource(words);
                if (resource.baseAddress != 0) images.push_back({resource.baseAddress, resource.width, resource.height, resource.format});
            } catch (const std::exception&) {
            }
        }
    }
    return images;
}

}
