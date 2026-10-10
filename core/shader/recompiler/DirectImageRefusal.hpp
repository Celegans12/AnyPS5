#ifndef CORE_SHADER_RECOMPILER_DIRECTIMAGEREFUSAL_HPP
#define CORE_SHADER_RECOMPILER_DIRECTIMAGEREFUSAL_HPP

#include "IntermediateRepresentation/IrMetadata/Resources.hpp"
#include <cstdint>
#include <cstdlib>

namespace ShaderRecompiler {

inline bool CanDeferDirectImage(const ImageResource& image, IrShaderStage stage) {
    return stage == IrShaderStage::Compute && image.table == NoTable && image.resourceClass == ImageResourceClass::Storage
        && image.dimension == RdnaImageDimension::Dim2DArray && image.written && !image.read && !image.atomic && !image.atomic64
        && image.depthBitsCompatible && !image.depthCompare && !image.r128 && !image.packed && image.byElements == 0u
        && image.byComponents == 0u && image.mipMode == ImageMipMode::None && !image.foldTexelOffsetsOnly
        && !image.cube && !image.depthBits && !image.depthUnorm16 && !image.srgbDecode && !image.constantSwizzle
        && image.conversionFormat == IrBufferFormat::Invalid && image.packedFormat == IrBufferFormat::Invalid && image.emulatedCompare == 0u;
}

inline bool DirectImageLoopGuardEnabled() {
    const auto* value = std::getenv("APS5_LOOP_GUARD");
    return value != nullptr && static_cast<std::uint32_t>(std::strtoul(value, nullptr, 0)) != 0u;
}

}

#endif
