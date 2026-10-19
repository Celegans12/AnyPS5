#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/SrtWalker/SrtFlatSlotClasses.hpp"
#include "Optimization/SrtWalker/SrtEvaluator.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include "RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "RdnaDecoder/RdnaImageOpDecoder.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include "ImageTableAbi.hpp"
#include "DirectImageRefusal.hpp"
#include "UnnormalizedWrap.hpp"
#include "PipelineSpecialization.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

namespace {

using ImageTableAbi::PoisonReason;

std::atomic<std::uint64_t> specializationNanoseconds{0};
std::atomic<std::uint64_t> evaluateNanoseconds{0};
std::atomic<std::uint64_t> materializeNanoseconds{0};

bool MaterializeProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

struct DecodedImage {
    IrTextureNumericClass numericClass = IrTextureNumericClass::Unsupported;
    RdnaImageDimension dimension = RdnaImageDimension::Unknown;
    std::uint32_t mipCount = 1;
    IrBufferFormat conversionFormat = IrBufferFormat::Invalid;
    std::uint32_t shaderSwizzle = ShaderImageIdentitySwizzle;
    bool cube = false;
    bool fmask = false;
    bool depthBits = false;
    bool depthUnorm16 = false;
    IrBufferFormat packedFormat = IrBufferFormat::Invalid;
    bool srgbDecode = false;
};

ShaderBufferResource decodeBufferDescriptor(const DescriptorValue& value) {
    if (value.dwordCount != 4u) {
        throw std::runtime_error("buffer descriptor has an invalid width");
    }
    ShaderBufferResource result;
    for (std::uint32_t i = 0; i < 4u; i++) {
        result.fields[i] = value.dwords[i];
    }
    return result;
}

bool nullImageDescriptor(const DescriptorValue& descriptor) {
    return descriptor.dwords[0] == 0u && (descriptor.dwords[1] & 0xffu) == 0u;
}

ImageType rawImageType(const DescriptorValue& descriptor) {
    return static_cast<ImageType>((descriptor.dwords[3] >> 28u) & 0xfu);
}

bool nullImageDescriptor(const ImageResource& image, const DescriptorValue& descriptor) {
    if (nullImageDescriptor(descriptor)) return true;
    return rawImageType(descriptor) == static_cast<ImageType>(0u) && image.table == NoTable && image.read && !image.written && !image.atomic;
}

IrBufferFormat rawImageFormat(const DescriptorValue& descriptor) {
    return static_cast<IrBufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
}

std::uint32_t descriptorImageSwizzle(const DescriptorValue& descriptor) {
    return descriptor.dwords[3] & 0xfffu;
}

bool descriptorIsCube(const DescriptorValue& descriptor) {
    return rawImageType(descriptor) == ImageType::Cube;
}

RdnaImageDimension descriptorDimension(const DescriptorValue& descriptor, RdnaImageDimension requested) {
    const bool wantArray = requested == RdnaImageDimension::Dim1DArray || requested == RdnaImageDimension::Dim2DArray || requested == RdnaImageDimension::Dim2DMsaaArray;
    switch (rawImageType(descriptor)) {
        case ImageType::Color1D:
            return RdnaImageDimension::Dim1D;
        case ImageType::Color1DArray:
            return wantArray ? RdnaImageDimension::Dim1DArray : RdnaImageDimension::Dim1D;
        case ImageType::Color3D:
            return RdnaImageDimension::Dim3D;
        case ImageType::Cube:
            return RdnaImageDimension::Dim2DArray;
        case ImageType::Color2DArray:
            return wantArray || requested == RdnaImageDimension::Dim3D ? RdnaImageDimension::Dim2DArray : RdnaImageDimension::Dim2D;
        case ImageType::Color2DMsaaArray:
            return wantArray ? RdnaImageDimension::Dim2DMsaaArray : RdnaImageDimension::Dim2DMsaa;
        case ImageType::Color2D:
            return RdnaImageDimension::Dim2D;
        case ImageType::Color2DMsaa:
            return RdnaImageDimension::Dim2DMsaa;
        default:
            throw std::runtime_error("image descriptor has an unsupported type");
    }
}

bool validImageDescriptor(const DescriptorValue& descriptor, bool r128) {
    const auto type = rawImageType(descriptor);
    const auto format = rawImageFormat(descriptor);
    if (type < ImageType::Color1D || format == IrBufferFormat::Invalid) {
        return false;
    }
    if (r128 && type != ImageType::Color1D && type != ImageType::Color2D && type != ImageType::Color2DMsaa) {
        return false;
    }
    if (type == ImageType::Color2DMsaa || type == ImageType::Color2DMsaaArray) {
        const auto baseLevel = (descriptor.dwords[3] >> 12u) & 0xfu;
        const auto fragments = (descriptor.dwords[3] >> 16u) & 0xfu;
        const auto maxMip = (descriptor.dwords[5] >> 4u) & 0xfu;
        return baseLevel == 0u && fragments >= 1u && fragments <= 3u && (r128 || maxMip == fragments);
    }
    return true;
}

bool directRefusalDescriptor(const DescriptorValue& descriptor) {
    if (descriptor.dwordCount != 8u || nullImageDescriptor(descriptor) || rawImageType(descriptor) != ImageType::Color1D
        || rawImageFormat(descriptor) != IrBufferFormat::Format8_8_8_8UNorm) return false;
    const auto& words = descriptor.dwords;
    if (((words[2] >> 14u) & 0x3fffu) != 0u || (words[3] & 0x0ffff000u) != 0u
        || (words[4] & 0x1fff1fffu) != 0u || (words[5] & (0xffu | (1u << 23u) | (1u << 26u))) != 0u
        || (words[6] & ((1u << 10u) | 0xffff8000u)) != 0u || words[7] != 0u) return false;
    for (std::uint32_t component = 0u; component < 4u; ++component) {
        const auto selector = (words[3] >> (3u * component)) & 7u;
        if (selector == 2u || selector == 3u) return false;
    }
    return true;
}

bool directRefusalMemory(const MemoryInfo& memory) {
    return memory.kind == ResourceKind::Image && memory.dataBits == 32u && !memory.d16 && memory.imageSampleFlags == 0u
        && !memory.imageHasMip && !memory.imagePacked && !memory.imageR128 && memory.imageByElements == 0u
        && !memory.gpuDescriptor && memory.imageDimension == RdnaImageDimension::Dim2DArray && memory.imageAddressComponents == 3u
        && memory.dmask != 0u && memory.dmask <= 0xfu && memory.dataDwords == static_cast<std::uint32_t>(std::popcount(memory.dmask))
        && memory.componentCount == memory.dataDwords && memory.componentIndex == 0u;
}

std::uint32_t storageMipCount(const ImageResource& base, const DescriptorValue& descriptor) {
    if (base.mipMode != ImageMipMode::DynamicStorage || nullImageDescriptor(base, descriptor)) {
        return 1u;
    }
    const auto mipBase = (descriptor.dwords[3] >> 12u) & 0xfu;
    const auto mipLast = (descriptor.dwords[3] >> 16u) & 0xfu;
    return mipBase <= mipLast ? mipLast - mipBase + 1u : 0u;
}

struct ImageDecode {
    DecodedImage decoded;
    std::optional<PoisonReason> reason;
    std::string detail;
};

ImageDecode rejectImage(PoisonReason reason, std::string detail) {
    ImageDecode result;
    result.reason = reason;
    result.detail = std::move(detail);
    return result;
}

PoisonReason invalidImageReason(const DescriptorValue& descriptor) {
    const auto type = rawImageType(descriptor);
    if (type < ImageType::Color1D) {
        return PoisonReason::NotImage;
    }
    if (rawImageFormat(descriptor) == IrBufferFormat::Invalid) {
        return PoisonReason::InvalidFormat;
    }
    if (type == ImageType::Color2DMsaa || type == ImageType::Color2DMsaaArray) {
        return PoisonReason::Multisampled;
    }
    return PoisonReason::Dimension;
}

ImageDecode inspectImageDescriptor(const DescriptorValue& descriptor, const ImageResource& base, std::uint32_t srgbDecodeFormats) {
    ImageDecode result;
    auto& decoded = result.decoded;
    decoded.mipCount = storageMipCount(base, descriptor);
    if (decoded.mipCount == 0u) {
        return rejectImage(PoisonReason::InvalidFormat, "storage image descriptor has an invalid mip range");
    }
    if (nullImageDescriptor(base, descriptor)) {
        decoded.numericClass = base.atomic ? IrTextureNumericClass::Uint : IrTextureNumericClass::Float;
        decoded.dimension = RdnaImageDimension::Dim2D;
        decoded.cube = false;
        return result;
    }
    if (base.resourceClass == ImageResourceClass::None || (base.atomic && base.resourceClass != ImageResourceClass::Storage)) {
        throw std::runtime_error("image resource has an invalid class");
    }
    if (!validImageDescriptor(descriptor, base.r128)) {
        return rejectImage(invalidImageReason(descriptor), "image descriptor is invalid");
    }
    decoded.dimension = descriptorDimension(descriptor, base.dimension);
    decoded.cube = descriptorIsCube(descriptor);
    const auto format = rawImageFormat(descriptor);
    if (base.atomic64 && format != IrBufferFormat::Format32_32UInt && format != IrBufferFormat::Format32_32SInt && format != IrBufferFormat::Format32_32Float) {
        return rejectImage(PoisonReason::InvalidFormat, "64-bit atomic image descriptor uses an unsupported format " + std::to_string(static_cast<std::uint32_t>(format)));
    }
    if (base.atomic && !base.atomic64 && format != IrBufferFormat::Format32UInt && format != IrBufferFormat::Format32SInt && format != IrBufferFormat::Format32Float) {
        return rejectImage(PoisonReason::InvalidFormat, "atomic image descriptor uses an unsupported format " + std::to_string(static_cast<std::uint32_t>(format)));
    }
    const bool storage = base.resourceClass == ImageResourceClass::Storage;
    const bool table = base.table != NoTable;
    if (storage && table && rawImageType(descriptor) != ImageType::Color2D) return rejectImage(PoisonReason::Dimension, "storage image tables require a 2D descriptor");
    const auto past = ResourceMaterializer::LevelsPastChain(descriptor);
    const auto uses = base.levelUses;
    if (!storage && past.levels != 0u && ((uses & (ImageLevelUseMipLoad | ImageLevelUseLodQuery)) != 0u || (past.base && (uses & ImageLevelUseLoad) != 0u) || (table && (uses & ImageLevelUseQuery) != 0u))) {
        return rejectImage(PoisonReason::UnsupportedOperation, "image_load_mip, image_get_lod, an image_load at a base level past the chain or an image_get_resinfo through an image table of a T# whose levels reach " + std::to_string(past.levels) + " past its extent's full mip chain is not implemented");
    }
    decoded.fmask = IsFmaskTextureFormat(format);
    if (decoded.fmask && !base.fmaskCompatible) return rejectImage(PoisonReason::Fmask, "FMASK requires a direct 32-bit image load");
    if (decoded.fmask && (storage || base.depthCompare || table)) {
        return rejectImage(PoisonReason::Fmask, "FMASK requires a direct sampled image load");
    }
    if (base.packed) {
        if (table || (!storage && descriptorImageSwizzle(descriptor) != ShaderImageIdentitySwizzle)) {
            return rejectImage(PoisonReason::InvalidFormat, "packed image access requires a direct image, with identity swizzle when sampled");
        }
        decoded.packedFormat = format;
    }
    if (base.byElements != 0u) {
        const bool eightBit = format == IrBufferFormat::Format8UNorm || format == IrBufferFormat::Format8SNorm || format == IrBufferFormat::Format8UInt || format == IrBufferFormat::Format8SInt;
        const bool sixteenBit = format == IrBufferFormat::Format16UNorm || format == IrBufferFormat::Format16SNorm || format == IrBufferFormat::Format16UInt || format == IrBufferFormat::Format16SInt || format == IrBufferFormat::Format16Float;
        const bool eightBitPair = format == IrBufferFormat::Format8_8UNorm || format == IrBufferFormat::Format8_8SNorm || format == IrBufferFormat::Format8_8UInt || format == IrBufferFormat::Format8_8SInt;
        const bool packedEight = format == IrBufferFormat::Format8UNorm || format == IrBufferFormat::Format8UInt || format == IrBufferFormat::Format8SInt;
        const bool packedSixteen = format == IrBufferFormat::Format16UNorm || format == IrBufferFormat::Format16UInt || format == IrBufferFormat::Format16SInt || format == IrBufferFormat::Format8_8UNorm || format == IrBufferFormat::Format8_8UInt || format == IrBufferFormat::Format8_8SInt;
        const bool measured = base.packed ? packedEight || (base.byElements == 2u && packedSixteen)
                                          : base.byElements == 4u ? base.byComponents == 1u && eightBit : base.byElements == 2u && (base.byComponents == 1u ? eightBit || sixteenBit : base.byComponents == 2u && eightBitPair);
        if (!measured || descriptorImageSwizzle(descriptor) != ShaderImageIdentitySwizzle || rawImageType(descriptor) != ImageType::Color2D || table) {
            return rejectImage(PoisonReason::InvalidFormat, base.packed ? "MIMG PCK2/PCK4 requires a direct, identity-swizzled 2D UNORM, UINT or SINT image whose elements fill one dword: R8, R16 or RG8 for PCK2, R8 for PCK4"
                                                                     : "MIMG BY2/BY4 requires a direct, identity-swizzled 2D image whose elements fill one dword: R8, R16 or RG8 for BY2, R8 for BY4");
        }
    }
    const auto remappedFormat = storage ? RemapStorageTextureFormat(format) : RemapTextureFormat(format);
    decoded.conversionFormat = remappedFormat != format ? format : IrBufferFormat::Invalid;
    if (format == IrBufferFormat::Format11_11_10UNorm || format == IrBufferFormat::Format10_11_11Float) {
        const bool floating = format == IrBufferFormat::Format10_11_11Float;
        if (!base.srgbDecodeCompatible) return rejectImage(PoisonReason::Conversion, floating ? "samples or gathers a converted float image, or queries its LOD, which is not implemented" : "sampling, gathering or querying LOD of a converted unorm image is not implemented");
        if (!base.depthBitsCompatible) return rejectImage(PoisonReason::Conversion, floating ? "reads or writes a converted float image with 16-bit data, which is not implemented" : "reads or writes a converted unorm image with 16-bit data, which is not implemented");
        for (std::uint32_t component = 0; component < 4u; ++component) {
            if (((descriptorImageSwizzle(descriptor) >> (component * 3u)) & 7u) == 7u) return rejectImage(PoisonReason::Conversion, "selects a channel the converted image format does not have");
        }
    }
    decoded.srgbDecode = !storage && !base.depthCompare && (srgbDecodeFormats & SrgbDecodeBit(format)) != 0u;
    if (decoded.srgbDecode && !base.srgbDecodeCompatible) {
        return rejectImage(PoisonReason::SrgbUnsupported, "samples or gathers an sRGB image the device cannot sample, which is not implemented");
    }
    if (storage || decoded.conversionFormat != IrBufferFormat::Invalid) {
        decoded.shaderSwizzle = descriptorImageSwizzle(descriptor);
    }
    const bool wideSint = format == IrBufferFormat::Format32SInt || format == IrBufferFormat::Format32_32SInt || format == IrBufferFormat::Format32_32_32_32SInt;
    const bool narrowSint = format == IrBufferFormat::Format16SInt || format == IrBufferFormat::Format8_8SInt || format == IrBufferFormat::Format16_16SInt || format == IrBufferFormat::Format8_8_8_8SInt || format == IrBufferFormat::Format16_16_16_16SInt;
    const bool rawSintStorage = storage && (wideSint || (narrowSint && !base.packed)) && base.written && !base.read && !base.atomic;
    decoded.numericClass = base.atomic ? IrTextureNumericClass::Uint : SampledTextureNumericClass(remappedFormat);
    if (!storage && !base.depthCompare && IsDepthBitsTexture(descriptor.dwords[1], descriptor.dwords[3])) {
        decoded.depthBits = true;
        if (!base.depthBitsCompatible) return rejectImage(PoisonReason::DepthBits16, "runtime image reads depth bits as unsupported 16-bit results");
        decoded.depthUnorm16 = DepthBitsTextureWidth(descriptor.dwords[1], descriptor.dwords[3]) == 16u;
        decoded.numericClass = IrTextureNumericClass::Float;
        decoded.shaderSwizzle = descriptorImageSwizzle(descriptor);
    }
    if (storage) {
        if ((!rawSintStorage && decoded.numericClass == IrTextureNumericClass::Sint) || decoded.numericClass == IrTextureNumericClass::Unsupported) {
            return rejectImage(PoisonReason::NumericClass, "storage image descriptor uses an unsupported format");
        }
        if (rawSintStorage) {
            decoded.numericClass = IrTextureNumericClass::Uint;
        }
        if ((rawSintStorage || (base.atomic && format == IrBufferFormat::Format32SInt)) && !base.packed) {
            if (!base.depthBitsCompatible) return rejectImage(PoisonReason::InvalidFormat, "stores 16-bit data to an image of a SINT format");
            decoded.conversionFormat = format;
        }
    } else if (decoded.numericClass == IrTextureNumericClass::Unsupported || (base.depthCompare && decoded.numericClass != IrTextureNumericClass::Float)) {
        return rejectImage(PoisonReason::NumericClass, "sampled image descriptor uses an unsupported format");
    }
    return result;
}

DecodedImage decodeImageDescriptor(const DescriptorValue& descriptor, const ImageResource& base, std::uint32_t srgbDecodeFormats) {
    auto result = inspectImageDescriptor(descriptor, base, srgbDecodeFormats);
    if (result.reason.has_value()) {
        throw std::runtime_error(result.detail);
    }
    return result.decoded;
}

bool requiresPointSampler(const ImageResource& mode) {
    return mode.numericClass == IrTextureNumericClass::Uint || mode.numericClass == IrTextureNumericClass::Sint || mode.conversionFormat != IrBufferFormat::Invalid || mode.depthBits;
}

bool reducesBetweenTexels(std::uint32_t word0, std::uint32_t filter) {
    return ((word0 >> 29u) & 3u) != 0u && (((filter >> 20u) & 0xfu) != 0u || ((filter >> 26u) & 3u) == 2u);
}

std::string hexText(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "%llx", static_cast<unsigned long long>(value));
    return text;
}

std::uint32_t TableKeyScanLimit() {
    static const std::uint32_t limit = [] {
        const char* text = std::getenv("APS5_IMAGE_TABLE_KEY_SCAN");
        return text != nullptr ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 0)) : 256u;
    }();
    return limit;
}

bool TableStrict() {
    static const bool strict = [] {
        const char* text = std::getenv("APS5_IMAGE_TABLE_STRICT");
        return text != nullptr && std::strcmp(text, "0") != 0;
    }();
    return strict;
}

bool TableTraced() {
    static const bool traced = std::getenv("APS5_TRACE_IMAGE_TABLE") != nullptr;
    return traced;
}

constexpr std::size_t PoisonReasonCount = static_cast<std::size_t>(PoisonReason::Count);

struct TableCounters {
    std::atomic<std::uint64_t> snapshots{0};
    std::atomic<std::uint64_t> tables{0};
    std::atomic<std::uint64_t> keys{0};
    std::array<std::atomic<std::uint64_t>, PoisonReasonCount> poison{};
    std::atomic<std::uint64_t> narrowed{0};
    std::atomic<std::uint64_t> narrowSkipped{0};
    std::atomic<std::uint64_t> memoHits{0};
    std::atomic<std::uint64_t> memoLayouts{0};
    std::atomic<std::uint64_t> memoMisses{0};
    std::atomic<std::uint64_t> memoColumns{0};
    std::atomic<std::uint64_t> memoBytes{0};
    std::atomic<std::uint64_t> nanoseconds{0};
    std::atomic<long long> lastReport{0};
};

TableCounters& tableCounters() {
    static TableCounters counters;
    return counters;
}

struct ColumnMemoTotals {
    std::atomic<std::uint64_t> hits{0};
    std::atomic<std::uint64_t> layoutsReused{0};
    std::atomic<std::uint64_t> misses{0};
};

ColumnMemoTotals& columnMemoTotals() {
    static ColumnMemoTotals totals;
    return totals;
}

void reportTables() {
    if (!MaterializeProfiled()) return;
    auto& counters = tableCounters();
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load(std::memory_order_relaxed);
    if (last == 0) {
        counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000ll || !counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    const auto snapshots = counters.snapshots.exchange(0, std::memory_order_relaxed);
    if (snapshots == 0) return;
    std::string poison;
    for (std::size_t reason = 0; reason < PoisonReasonCount; reason++) {
        const auto count = counters.poison[reason].exchange(0, std::memory_order_relaxed);
        if (count == 0) continue;
        poison += " " + std::string(ImageTableAbi::PoisonReasonName(static_cast<PoisonReason>(reason))) + "=" + std::to_string(count);
    }
    std::fprintf(stderr, "[image-table] (10 s): %llu snapshots, %llu tables, %llu keys, structural poison:%s, narrowing %llu applied / %llu skipped, column memo %llu hits / %llu layouts reused / %llu misses (%llu columns, %.1f MiB kept), %.1f ms\n",
        static_cast<unsigned long long>(snapshots), static_cast<unsigned long long>(counters.tables.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.keys.exchange(0, std::memory_order_relaxed)), poison.empty() ? " none" : poison.c_str(),
        static_cast<unsigned long long>(counters.narrowed.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.narrowSkipped.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.memoHits.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.memoLayouts.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.memoMisses.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.memoColumns.load(std::memory_order_relaxed)), static_cast<double>(counters.memoBytes.load(std::memory_order_relaxed)) / (1024.0 * 1024.0), static_cast<double>(counters.nanoseconds.exchange(0, std::memory_order_relaxed)) / 1e6);
}

enum class RecordState : std::uint8_t { OutsideDomain, Read, Unmapped, Carried };

struct ColumnRecords {
    std::uint64_t base = 0;
    std::uint64_t size = 0;
    std::uint32_t records = 0;
    std::uint32_t keys = 0;
    std::uint32_t fault = 0;
    bool narrowed = false;
    bool outside = false;
    std::vector<RecordState> states;
    std::vector<std::uint32_t> words;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> readRanges;
    std::vector<std::uint64_t> addresses;
};

struct TableTrace {
    std::uint64_t base = 0;
    std::uint64_t size = 0;
    std::uint32_t records = 0;
    std::uint32_t keys = 0;
    std::uint32_t poison = 0;
    bool narrowed = false;

    bool operator==(const TableTrace& other) const = default;
};

void traceTable(const IrResourcePlan& plan, std::uint32_t table, const TableColumn& column, const TableTrace& trace) {
    static std::mutex mutex;
    static std::map<std::pair<std::uint64_t, std::uint32_t>, TableTrace> seen;
    std::lock_guard lock(mutex);
    auto& last = seen[{plan.shaderHash, table}];
    if (last == trace) return;
    last = trace;
    std::fprintf(stderr, "[image-table] shader 0x%llx table %u (%s): base 0x%llx size 0x%llx stride 0x%x addend 0x%x offset 0x%x dwords %u, %u records, %u keys%s, %u structural poison\n", static_cast<unsigned long long>(plan.shaderHash), table, column.sampler ? "sampler" : "image", static_cast<unsigned long long>(trace.base), static_cast<unsigned long long>(trace.size), column.stride, column.addend, column.offset, column.dwordCount, trace.records, trace.keys, trace.narrowed ? " (narrowed)" : "", trace.poison);
}

class TableSnapshotter {
public:
    TableSnapshotter(const IrResourcePlan& plan, const SrtRuntime& runtime, SrtWalker& walker, const std::vector<std::uint8_t>& activeSources, ImageTableSnapshot& tables) : plan(plan), runtime(runtime), walker(walker), activeSources(activeSources), tables(tables) {}

    void Run() {
        tables = ImageTableSnapshot{};
        std::uint32_t count = 0;
        for (const auto& image : plan.info.images) {
            if (image.table != NoTable) count = std::max(count, image.table + 1u);
        }
        for (const auto& sampler : plan.info.samplers) {
            if (sampler.table != NoTable) count = std::max(count, sampler.table + 1u);
        }
        if (count == 0u) {
            return;
        }
        if (runtime.readMemory == nullptr) {
            throw std::runtime_error("image table snapshot requires runtime memory access");
        }
        const auto started = std::chrono::steady_clock::now();
        tables.shader = plan.shaderHash;
        tables.tables.resize(count);
        for (const auto& image : plan.info.images) {
            if (image.table != NoTable) Snapshot(image.source, image.table);
        }
        for (const auto& sampler : plan.info.samplers) {
            if (sampler.table != NoTable) Snapshot(sampler.source, sampler.table);
        }
        auto& counters = tableCounters();
        counters.snapshots.fetch_add(1, std::memory_order_relaxed);
        counters.tables.fetch_add(count, std::memory_order_relaxed);
        counters.nanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
        if (TableStrict()) {
            for (std::uint32_t table = 0; table < tables.tables.size(); table++) {
                const auto& column = tables.tables[table];
                const auto poisoned = std::find_if(column.codes.begin(), column.codes.end(), [](std::uint32_t code) { return (code & ImageTableAbi::PoisonFlag) != 0u && (code & ~ImageTableAbi::PoisonFlag) != static_cast<std::uint32_t>(PoisonReason::OutsideDomain); });
                if (column.fault != 0u || poisoned != column.codes.end()) {
                    const auto reason = static_cast<PoisonReason>((column.fault != 0u ? column.fault : *poisoned) & ~ImageTableAbi::PoisonFlag);
                    throw std::runtime_error("image table " + std::to_string(table) + " at 0x" + hexText(column.base) + " holds a record that cannot be read (" + std::string(ImageTableAbi::PoisonReasonName(reason)) + "), and APS5_IMAGE_TABLE_STRICT is set");
                }
            }
        }
    }

private:
    static constexpr std::uint64_t PageBytes = 4096u;

    bool Accessible(std::uint64_t address, std::uint64_t bytes) const {
        return runtime.accessible == nullptr || runtime.accessible(runtime.userContext, address, bytes);
    }

    bool PageAccessible(std::uint64_t address, std::uint64_t begin, std::uint64_t end) {
        const auto page = address & ~(PageBytes - 1u);
        if (const auto found = pages.find(page); found != pages.end()) return found->second;
        const auto first = std::max(page, begin);
        const auto last = std::min(page + PageBytes, end);
        const bool accessible = first < last && Accessible(first, last - first);
        pages.emplace(page, accessible);
        return accessible;
    }

    bool WholePage(std::uint64_t page) {
        const auto [found, inserted] = wholePages.try_emplace(page, false);
        if (inserted) found->second = Accessible(page, PageBytes);
        return found->second;
    }

    bool AddressAccessible(std::uint64_t address) {
        return WholePage(address & ~(PageBytes - 1u)) || Accessible(address, sizeof(std::uint32_t));
    }

    std::uint32_t Read(std::uint64_t address) const {
        std::uint32_t word = 0;
        if (!runtime.readMemory(runtime.userContext, address, &word)) {
            throw std::runtime_error("image table: failed to read guest memory at 0x" + hexText(address));
        }
        return word;
    }

    ShaderBufferResource EvaluateBuffer(std::uint32_t source) const {
        DescriptorValue value;
        walker.EvaluateDescriptorSource(plan, source, runtime, value);
        return decodeBufferDescriptor(value);
    }

    void Narrow(const TableColumn& column, ColumnRecords& records) {
        const auto skip = [&] {
            tableCounters().narrowSkipped.fetch_add(1, std::memory_order_relaxed);
        };
        std::vector<std::uint32_t> values{0u};
        std::vector<std::pair<std::uint64_t, std::uint64_t>> scanned;
        for (const auto& domain : column.keyDomains) {
            if (domain.value) {
                DescriptorValue key;
                try {
                    walker.EvaluateDescriptorSource(plan, domain.source, runtime, key);
                } catch (const std::exception&) {
                    skip();
                    return;
                }
                values.push_back(key.dwords[0]);
                continue;
            }
            const auto keyBuffer = EvaluateBuffer(domain.source);
            if (keyBuffer.Type() != 0u) {
                skip();
                return;
            }
            const auto base = keyBuffer.Base48();
            const auto size = keyBuffer.GetSize();
            const std::uint64_t first = domain.offset & ~3u;
            const std::uint64_t positions = size >= first + 4u ? (size - first - 4u) / domain.stride + 1u : 0u;
            if (positions > TableKeyScanLimit() || first > 0xffffffffull) {
                skip();
                return;
            }
            if (positions != 0u && runtime.pendingWrite != nullptr && runtime.pendingWrite(runtime.userContext, base + first, size - first)) {
                skip();
                return;
            }
            for (std::uint64_t index = 0; index < positions; index++) {
                const auto address = base + first + index * domain.stride;
                if (!PageAccessible(address, base, base + size)) {
                    skip();
                    return;
                }
                values.push_back(Read(address));
            }
            if (positions != 0u) scanned.emplace_back(base, size);
        }
        records.states.assign(records.keys, RecordState::OutsideDomain);
        records.outside = false;
        for (const auto value : values) {
            const std::uint32_t relative = value * column.stride;
            const bool aligned = relative % column.stride == 0u;
            const auto record = relative / column.stride;
            if (aligned && record < records.keys) {
                records.states[record] = RecordState::Read;
                continue;
            }
            if (column.address || ImageTableAbi::ScalarBufferDword(relative + column.addend, column.offset, records.size).has_value()) records.outside = true;
        }
        records.narrowed = true;
        for (const auto& [base, size] : scanned) tables.ranges.emplace_back(base, size);
        tableCounters().narrowed.fetch_add(1, std::memory_order_relaxed);
    }

    static bool FlatWordsIndex() {
        static const bool flat = std::getenv("APS5_NO_FAST_TABLE_SNAPSHOT") == nullptr;
        return flat;
    }

    static std::size_t WordsSlot(const DescriptorValue& value) {
        return WordsHash{}(WordsHash::Key{value.dwords, value.dwordCount, value.availableDwordMask});
    }

    std::uint32_t WordsIndex(const DescriptorValue& value) {
        if (!FlatWordsIndex()) {
            const auto [found, inserted] = wordsIndex.try_emplace({value.dwords, value.dwordCount, value.availableDwordMask}, static_cast<std::uint32_t>(tables.words.size()));
            if (inserted) tables.words.push_back(value);
            return found->second;
        }
        if ((tables.words.size() + 1u) * 2u > wordSlots.size()) {
            wordSlots.assign(std::max<std::size_t>(64u, wordSlots.size() * 2u), 0u);
            const auto mask = wordSlots.size() - 1u;
            for (std::size_t index = 0; index < tables.words.size(); ++index) {
                auto slot = WordsSlot(tables.words[index]) & mask;
                while (wordSlots[slot] != 0u) slot = (slot + 1u) & mask;
                wordSlots[slot] = static_cast<std::uint32_t>(index + 1u);
            }
        }
        const auto mask = wordSlots.size() - 1u;
        auto slot = WordsSlot(value) & mask;
        while (wordSlots[slot] != 0u) {
            const auto index = wordSlots[slot] - 1u;
            if (tables.words[index] == value) return index;
            slot = (slot + 1u) & mask;
        }
        const auto index = static_cast<std::uint32_t>(tables.words.size());
        tables.words.push_back(value);
        wordSlots[slot] = index + 1u;
        return index;
    }

    ColumnRecords& Column(std::uint32_t source) {
        if (const auto found = columns.find(source); found != columns.end()) {
            return found->second;
        }
        auto& records = columns[source];
        const auto& column = *plan.descriptorSources.at(source).tableColumn;
        if (column.dynamic) return records;
        const bool active = source >= activeSources.size() || activeSources[source] != 0u;
        if (!active) {
            records.fault = ImageTableAbi::PoisonCode(static_cast<std::uint32_t>(PoisonReason::Inactive));
            return records;
        }
        if (column.address) {
            DescriptorValue pointer;
            walker.EvaluateDescriptorSource(plan, column.heapSource, runtime, pointer);
            records.base = (static_cast<std::uint64_t>(pointer.dwords[1]) << 32u) | pointer.dwords[0];
            records.records = ImageTableAbi::ScalarAddressRecords(column.stride);
        } else {
            const auto heap = EvaluateBuffer(column.heapSource);
            if (heap.Type() != 0u) {
                records.fault = ImageTableAbi::PoisonCode(static_cast<std::uint32_t>(PoisonReason::NotBuffer));
                return records;
            }
            records.base = heap.Base48();
            records.size = heap.GetSize();
            records.records = ImageTableAbi::ScalarBufferRecords(column.addend, column.stride, column.offset + (column.indirect ? 4u : 0u), records.size);
            if (records.size != 0u) tables.ranges.emplace_back(records.base, records.size);
        }
        records.records = std::min(records.records, ImageTableAbi::KeyRecords(column.maxKey, column.stride));
        records.keys = std::min(records.records, ImageTableAbi::MaxKeys);
        records.outside = records.keys < records.records;
        records.states.assign(records.keys, RecordState::Read);
        if (!column.keyDomains.empty() && records.keys != 0u) {
            Narrow(column, records);
        }
        records.words.assign(records.keys, 0u);
        const auto end = records.base + records.size;
        if (column.indirect) {
            records.addresses.assign(records.keys, 0u);
            for (std::uint32_t record = 0; record < records.keys; record++) {
                if (records.states[record] != RecordState::Read) continue;
                const auto offset = column.addend + record * column.stride;
                const auto lowWord = ImageTableAbi::ScalarBufferDword(offset, column.offset, records.size);
                const auto highWord = ImageTableAbi::ScalarBufferDword(offset, column.offset + 4u, records.size);
                if (!lowWord.has_value() || !highWord.has_value() || !PageAccessible(records.base + *lowWord, records.base, end) || !PageAccessible(records.base + *highWord, records.base, end)) {
                    records.states[record] = RecordState::Unmapped;
                    continue;
                }
                const std::uint64_t pointer = (static_cast<std::uint64_t>(Read(records.base + *highWord)) << 32u) | Read(records.base + *lowWord);
                DescriptorValue value;
                value.dwordCount = column.sampler ? 4u : 8u;
                std::uint64_t first = std::numeric_limits<std::uint64_t>::max();
                std::uint64_t past = 0;
                for (std::uint32_t dword = 0; dword < column.dwordCount; dword++) {
                    auto address = ImageTableAbi::ScalarAddressDword(pointer, column.indirectOffset, column.indirectImmediate + dword * 4u);
                    if (address.has_value() && (*address & 3u) != 0u) address.reset();
                    if (!address.has_value() || !AddressAccessible(*address)) {
                        records.states[record] = RecordState::Unmapped;
                        break;
                    }
                    value.dwords[dword] = Read(*address);
                    value.availableDwordMask |= static_cast<std::uint8_t>(1u << dword);
                    first = std::min(first, *address);
                    past = std::max(past, *address + sizeof(std::uint32_t));
                }
                if (records.states[record] != RecordState::Read) continue;
                records.words[record] = WordsIndex(value);
                records.addresses[record] = first;
                tables.ranges.emplace_back(first, past - first);
            }
            return records;
        }
        if (records.keys == 0u) return records;
        const auto key = ColumnKey(source, records);
        auto [memo, admitted] = FindColumnMemo(key);
        if (memo != nullptr && !(memo->column == column && memo->narrowed == records.states && ReplayChecks(*memo->layout, column, records))) memo.reset();
        const auto layout = memo != nullptr ? memo->layout : std::make_shared<const ColumnLayout>(Layout(column, records));
        auto narrowed = std::move(records.states);
        records.states = layout->states;
        records.readRanges = layout->readRanges;
        struct ReadScratchStorage {};
        std::vector<std::uint32_t> ownedRead;
        std::span<std::uint32_t> read;
        if (FlatWordsIndex()) {
            auto& scratch = HostThreadLocal<std::vector<std::uint32_t>, ReadScratchStorage>();
            if (scratch.size() < layout->total) scratch.resize(layout->total);
            read = std::span<std::uint32_t>(scratch).first(layout->total);
        } else {
            ownedRead.assign(layout->total, 0u);
            read = ownedRead;
        }
        std::uint32_t at = 0;
        for (const auto& [address, count] : layout->runs) {
            if (runtime.readWords != nullptr) {
                if (!runtime.readWords(runtime.userContext, address, read.data() + at, count)) throw std::runtime_error("image table: failed to read guest memory at 0x" + hexText(address));
            } else {
                for (std::uint32_t word = 0; word < count; word++) read[at + word] = Read(address + word * 4u);
            }
            at += count;
        }
        if (memo != nullptr && std::ranges::equal(memo->read, read)) {
            std::vector<std::uint32_t> global(memo->distinct.size());
            for (std::size_t index = 0; index < memo->distinct.size(); index++) global[index] = WordsIndex(memo->distinct[index]);
            for (std::uint32_t record = 0; record < records.keys; record++) {
                if (records.states[record] == RecordState::Read) records.words[record] = global[memo->local[record]];
            }
            tableCounters().memoHits.fetch_add(1, std::memory_order_relaxed);
            columnMemoTotals().hits.fetch_add(1, std::memory_order_relaxed);
        } else {
            constexpr auto Unseen = std::numeric_limits<std::uint32_t>::max();
            std::shared_ptr<ColumnMemo> stored;
            std::vector<std::uint32_t> locals;
            if (admitted) {
                stored = std::make_shared<ColumnMemo>();
                stored->column = column;
                stored->narrowed = std::move(narrowed);
                stored->layout = layout;
                stored->local.assign(records.keys, 0u);
                locals.assign(tables.words.size() + records.keys, Unseen);
            }
            auto next = layout->split.begin();
            for (std::uint32_t record = 0; record < records.keys; record++) {
                if (records.states[record] != RecordState::Read) continue;
                DescriptorValue value;
                value.dwordCount = column.sampler ? 4u : 8u;
                if (layout->whole[record] != ColumnLayout::Unread) {
                    std::copy_n(read.begin() + layout->whole[record], column.dwordCount, value.dwords.begin());
                    value.availableDwordMask = static_cast<std::uint8_t>((1u << column.dwordCount) - 1u);
                } else {
                    for (std::uint32_t dword = 0; dword < column.dwordCount; dword++) {
                        if (next->second[dword] != ColumnLayout::Unread) {
                            value.dwords[dword] = read[next->second[dword]];
                            value.availableDwordMask |= static_cast<std::uint8_t>(1u << dword);
                        }
                    }
                    ++next;
                }
                const auto global = WordsIndex(value);
                records.words[record] = global;
                if (stored == nullptr) continue;
                if (locals[global] == Unseen) {
                    locals[global] = static_cast<std::uint32_t>(stored->distinct.size());
                    stored->distinct.push_back(value);
                }
                stored->local[record] = locals[global];
            }
            if (stored != nullptr) {
                stored->read.assign(read.begin(), read.end());
                stored->bytes = MemoBytes(*stored);
                StoreColumnMemo(key, std::move(stored));
            }
            if (memo != nullptr) {
                tableCounters().memoLayouts.fetch_add(1, std::memory_order_relaxed);
                columnMemoTotals().layoutsReused.fetch_add(1, std::memory_order_relaxed);
            } else {
                tableCounters().memoMisses.fetch_add(1, std::memory_order_relaxed);
                columnMemoTotals().misses.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (column.address && layout->low < layout->high) tables.ranges.emplace_back(layout->low, layout->high - layout->low);
        return records;
    }

    struct ColumnLayout {
        static constexpr auto Unread = std::numeric_limits<std::uint32_t>::max();

        struct Check {
            std::uint64_t address = 0;
            bool whole = false;
            bool answer = false;

            bool operator==(const Check& other) const = default;
        };

        std::vector<std::uint32_t> whole;
        std::vector<std::pair<std::uint32_t, std::array<std::uint32_t, 8>>> split;
        std::vector<std::pair<std::uint64_t, std::uint32_t>> runs;
        std::vector<RecordState> states;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> readRanges;
        std::vector<Check> checks;
        std::uint64_t low = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t high = 0;
        std::uint32_t total = 0;
    };

    struct ColumnMemo {
        TableColumn column;
        std::vector<RecordState> narrowed;
        std::shared_ptr<const ColumnLayout> layout;
        std::vector<std::uint32_t> read;
        std::vector<DescriptorValue> distinct;
        std::vector<std::uint32_t> local;
        std::size_t bytes = 0;
    };

    struct ColumnMemoKey {
        std::uint64_t shader = 0;
        std::uint32_t source = 0;
        std::uint64_t base = 0;
        std::uint64_t size = 0;
        std::uint32_t keys = 0;
        std::uint64_t states = 0;

        bool operator<(const ColumnMemoKey& other) const {
            return std::tie(shader, source, base, size, keys, states) < std::tie(other.shader, other.source, other.base, other.size, other.keys, other.states);
        }
    };

    static std::size_t ColumnMemoEntries() {
        static const std::size_t entries = [] {
            const char* text = std::getenv("APS5_TABLE_COLUMN_MEMO_ENTRIES");
            if (text == nullptr || *text == '\0') return std::size_t{2048};
            const auto value = std::strtoull(text, nullptr, 10);
            return value == 0u ? std::size_t{512} : static_cast<std::size_t>(value);
        }();
        return entries;
    }
    static constexpr std::size_t ColumnMemoBytes = std::size_t{256} << 20u;
    static constexpr std::size_t ColumnSightings = 4096u;

    struct ColumnMemoStore {
        std::mutex mutex;
        std::map<ColumnMemoKey, std::pair<std::shared_ptr<const ColumnMemo>, std::uint64_t>> entries;
        std::uint64_t clock = 0;
        std::size_t bytes = 0;
        std::array<std::uint64_t, ColumnSightings> sightings{};
        std::size_t nextSighting = 0;
    };

    struct ColumnMemoLookup {
        std::shared_ptr<const ColumnMemo> memo;
        bool admitted = false;
    };

    static ColumnMemoStore& columnMemos() {
        static ColumnMemoStore store;
        return store;
    }

    template<typename T>
    static std::size_t VectorBytes(const std::vector<T>& values) {
        return values.capacity() * sizeof(T);
    }

    static std::size_t MemoBytes(const ColumnMemo& memo) {
        const auto& layout = *memo.layout;
        return sizeof(ColumnMemo) + VectorBytes(memo.column.keyDomains) + VectorBytes(memo.narrowed) + VectorBytes(memo.read) + VectorBytes(memo.distinct) + VectorBytes(memo.local) + sizeof(ColumnLayout) + VectorBytes(layout.whole) + VectorBytes(layout.split) + VectorBytes(layout.runs) + VectorBytes(layout.states) + VectorBytes(layout.readRanges) + VectorBytes(layout.checks);
    }

    ColumnMemoKey ColumnKey(std::uint32_t source, const ColumnRecords& records) const {
        std::uint64_t states = 0xcbf29ce484222325ull;
        for (const auto state : records.states) states = (states ^ static_cast<std::uint8_t>(state)) * 0x100000001b3ull;
        return ColumnMemoKey{plan.shaderHash, source, records.base, records.size, records.keys, states};
    }

    static std::uint64_t SightingOf(const ColumnMemoKey& key) {
        std::uint64_t hash = 0xcbf29ce484222325ull;
        for (const std::uint64_t field : {key.shader, static_cast<std::uint64_t>(key.source), key.base, key.size, static_cast<std::uint64_t>(key.keys), key.states}) hash = (hash ^ field) * 0x100000001b3ull;
        return hash | 1u;
    }

    static ColumnMemoLookup FindColumnMemo(const ColumnMemoKey& key) {
        auto& store = columnMemos();
        std::lock_guard lock(store.mutex);
        if (const auto found = store.entries.find(key); found != store.entries.end()) {
            found->second.second = ++store.clock;
            return {found->second.first, true};
        }
        const auto sighting = SightingOf(key);
        if (std::find(store.sightings.begin(), store.sightings.end(), sighting) != store.sightings.end()) return {nullptr, true};
        store.sightings[store.nextSighting] = sighting;
        store.nextSighting = (store.nextSighting + 1u) % ColumnSightings;
        return {nullptr, false};
    }

    static void StoreColumnMemo(const ColumnMemoKey& key, std::shared_ptr<const ColumnMemo> memo) {
        std::vector<std::shared_ptr<const ColumnMemo>> released;
        auto& store = columnMemos();
        std::lock_guard lock(store.mutex);
        auto& entry = store.entries[key];
        if (entry.first != nullptr) {
            store.bytes -= entry.first->bytes;
            released.push_back(std::move(entry.first));
        }
        store.bytes += memo->bytes;
        entry = {std::move(memo), ++store.clock};
        while (store.entries.size() > ColumnMemoEntries() || store.bytes > ColumnMemoBytes) {
            auto oldest = store.entries.begin();
            for (auto candidate = store.entries.begin(); candidate != store.entries.end(); ++candidate) {
                if (candidate->second.second < oldest->second.second) oldest = candidate;
            }
            store.bytes -= oldest->second.first->bytes;
            released.push_back(std::move(oldest->second.first));
            store.entries.erase(oldest);
        }
        tableCounters().memoColumns.store(store.entries.size(), std::memory_order_relaxed);
        tableCounters().memoBytes.store(store.bytes, std::memory_order_relaxed);
    }

    bool ReplayChecks(const ColumnLayout& layout, const TableColumn& column, const ColumnRecords& records) {
        const auto end = records.base + records.size;
        return std::all_of(layout.checks.begin(), layout.checks.end(), [&](const ColumnLayout::Check& check) {
            const bool answer = check.whole ? WholePage(check.address) : column.address ? Accessible(check.address, sizeof(std::uint32_t)) : PageAccessible(check.address, records.base, end);
            return answer == check.answer;
        });
    }

    ColumnLayout Layout(const TableColumn& column, const ColumnRecords& records) {
        ColumnLayout layout;
        layout.states = records.states;
        layout.whole.assign(records.keys, ColumnLayout::Unread);
        const auto end = records.base + records.size;
        const auto note = [&](std::uint64_t address, bool whole, bool answer) {
            const ColumnLayout::Check check{address, whole, answer};
            if (layout.checks.empty() || layout.checks.back() != check) layout.checks.push_back(check);
            return answer;
        };
        std::uint64_t checkedPage = std::numeric_limits<std::uint64_t>::max();
        bool checkedAccessible = false;
        const auto accessible = [&](std::uint64_t address) {
            const auto page = address & ~(PageBytes - 1u);
            if (column.address) return note(page, true, WholePage(page)) || note(address, false, Accessible(address, sizeof(std::uint32_t)));
            if (page != checkedPage) {
                checkedPage = page;
                checkedAccessible = note(address, false, PageAccessible(address, records.base, end));
            }
            return checkedAccessible;
        };
        const auto wholeAccessible = [&](std::uint64_t address) {
            if (!column.address) return accessible(address);
            const auto page = address & ~(PageBytes - 1u);
            return note(page, true, WholePage(page));
        };
        const auto locate = [&](std::uint32_t offset, std::uint32_t immediate) -> std::optional<std::uint64_t> {
            if (column.address) return ImageTableAbi::ScalarAddressDword(records.base, offset, immediate);
            const auto position = ImageTableAbi::ScalarBufferDword(offset, immediate, records.size);
            return position.has_value() ? std::optional<std::uint64_t>(records.base + *position) : std::nullopt;
        };
        const auto take = [&](std::uint64_t address, std::uint32_t words) {
            if (!layout.runs.empty() && address > layout.runs.back().first && address - layout.runs.back().first == static_cast<std::uint64_t>(layout.runs.back().second) * 4u) {
                layout.runs.back().second += words;
            } else {
                layout.runs.emplace_back(address, words);
            }
            if (column.address) layout.readRanges.emplace_back(address, static_cast<std::uint64_t>(words) * sizeof(std::uint32_t));
            layout.low = std::min(layout.low, address);
            layout.high = std::max(layout.high, address + static_cast<std::uint64_t>(words) * sizeof(std::uint32_t));
            layout.total += words;
            return layout.total - words;
        };
        const auto last = (column.dwordCount - 1u) * 4u;
        for (std::uint32_t record = 0; record < records.keys; record++) {
            if (layout.states[record] != RecordState::Read) continue;
            const auto offset = column.addend + record * column.stride;
            if (offset >= column.foldedAddend && offset > 0xffffffffu - column.foldedCarry) {
                layout.states[record] = RecordState::Carried;
                continue;
            }
            const auto head = locate(offset, column.offset);
            const auto tail = locate(offset, column.offset + last);
            if (head.has_value() && tail.has_value() && (*head & 3u) == 0u && *tail > *head && *tail - *head == last && wholeAccessible(*head) && wholeAccessible(*tail)) {
                layout.whole[record] = take(*head, column.dwordCount);
                continue;
            }
            std::array<std::uint32_t, 8> slots;
            slots.fill(ColumnLayout::Unread);
            for (std::uint32_t dword = 0; dword < column.dwordCount; dword++) {
                auto address = locate(offset, column.offset + dword * 4u);
                if (column.address && address.has_value() && (*address & 3u) != 0u) address.reset();
                if (!address.has_value() && !column.address) continue;
                if (!address.has_value() || !accessible(*address)) {
                    layout.states[record] = RecordState::Unmapped;
                    break;
                }
                slots[dword] = take(*address, 1u);
            }
            if (layout.states[record] == RecordState::Read) layout.split.emplace_back(record, slots);
        }
        std::sort(layout.readRanges.begin(), layout.readRanges.end());
        std::size_t ranges = 0;
        for (const auto& range : layout.readRanges) {
            if (ranges != 0u && range.first - layout.readRanges[ranges - 1u].first <= layout.readRanges[ranges - 1u].second) {
                auto& previous = layout.readRanges[ranges - 1u];
                previous.second = std::max(previous.second, range.first - previous.first + range.second);
            } else {
                layout.readRanges[ranges++] = range;
            }
        }
        layout.readRanges.resize(ranges);
        return layout;
    }

    void Snapshot(std::uint32_t source, std::uint32_t table) {
        const auto& column = *plan.descriptorSources.at(source).tableColumn;
        const auto& records = Column(source);
        auto& out = tables.tables.at(table);
        out.base = records.base;
        out.size = records.size;
        out.records = records.records;
        out.keys = records.keys;
        out.fault = records.fault;
        out.outside = records.outside;
        out.codes.assign(records.keys, 0u);
        out.readRanges = records.readRanges;
        out.pointer = column.address;
        out.indirect = column.indirect;
        out.addend = column.addend;
        out.stride = column.stride;
        out.offset = column.offset;
        out.dwords = column.dwordCount;
        out.addresses = records.addresses;
        TableTrace trace{records.base, records.size, records.records, records.keys, 0u, records.narrowed};
        auto& counters = tableCounters();
        if (records.fault != 0u) counters.poison[records.fault & ~ImageTableAbi::PoisonFlag].fetch_add(1, std::memory_order_relaxed);
        for (std::uint32_t key = 0; key < records.keys; key++) {
            switch (records.states[key]) {
                case RecordState::Read:
                    out.codes[key] = records.words[key];
                    continue;
                case RecordState::OutsideDomain:
                    out.codes[key] = ImageTableAbi::PoisonCode(static_cast<std::uint32_t>(PoisonReason::OutsideDomain));
                    break;
                case RecordState::Unmapped:
                    out.codes[key] = ImageTableAbi::PoisonCode(static_cast<std::uint32_t>(PoisonReason::Unmapped));
                    counters.poison[static_cast<std::size_t>(PoisonReason::Unmapped)].fetch_add(1, std::memory_order_relaxed);
                    break;
                case RecordState::Carried:
                    out.codes[key] = ImageTableAbi::PoisonCode(static_cast<std::uint32_t>(PoisonReason::OffsetCarry));
                    counters.poison[static_cast<std::size_t>(PoisonReason::OffsetCarry)].fetch_add(1, std::memory_order_relaxed);
                    break;
            }
            trace.poison++;
        }
        counters.keys.fetch_add(records.keys, std::memory_order_relaxed);
        if (TableTraced()) traceTable(plan, table, column, trace);
    }

    const IrResourcePlan& plan;
    const SrtRuntime& runtime;
    SrtWalker& walker;
    const std::vector<std::uint8_t>& activeSources;
    ImageTableSnapshot& tables;
    std::map<std::uint32_t, ColumnRecords> columns;
    struct WordsHash {
        using Key = std::tuple<std::array<std::uint32_t, 8>, std::uint32_t, std::uint8_t>;
        std::size_t operator()(const Key& key) const noexcept {
            std::uint64_t hash = 0xcbf29ce484222325ull ^ std::get<1>(key) ^ std::get<2>(key);
            for (const auto word : std::get<0>(key)) hash = (hash ^ word) * 0x100000001b3ull;
            return static_cast<std::size_t>(hash ^ (hash >> 32u));
        }
    };
    std::unordered_map<WordsHash::Key, std::uint32_t, WordsHash> wordsIndex;
    std::vector<std::uint32_t> wordSlots;
    std::unordered_map<std::uint64_t, bool> pages;
    std::unordered_map<std::uint64_t, bool> wholePages;
};

void materializeSnapshot(const IrResourcePlan& plan, const SrtRuntime& runtime, SrtWalker& walker, ResourceSnapshot& snapshot, std::vector<std::uint8_t>& activeSources) {
    snapshot = ResourceSnapshot{};
    if (plan.uniformFill.fill.kind != UniformFillKind::None) {
        const auto words = plan.uniformFill.fill.words;
        if (words == 0u || words > plan.uniformFill.values.size()) {
            throw std::runtime_error("uniform fill plan has an invalid word count");
        }
        std::array<std::uint32_t, 4> stored{};
        walker.EvaluateUniformValues(plan, std::span(plan.uniformFill.values).first(words), runtime, std::span(stored).first(words));
        for (std::uint32_t i = 1; i < words; i++) {
            if (stored[i] != stored[0]) {
                throw std::runtime_error("uniform fill values diverge at runtime");
            }
        }
        snapshot.uniformFill = plan.uniformFill.fill;
        snapshot.uniformFill.value = stored[0];
    }
    if (runtime.userData.size() < plan.userDataCount) {
        throw std::runtime_error("runtime user data is smaller than the shader user data count");
    }
    snapshot.userData.assign(runtime.userData.begin(), runtime.userData.begin() + plan.userDataCount);

    std::vector<DescriptorValue> values;
    const auto evaluateStarted = MaterializeProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    walker.EvaluateRuntimeSources(plan, plan.materializationSources, runtime, values, snapshot.flattenedSrt, plan.cleanFlatSlots, activeSources, &snapshot.srtPoison, &snapshot.nullRootReads);
    if (MaterializeProfiled()) evaluateNanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - evaluateStarted).count()), std::memory_order_relaxed);

    std::size_t cursor = 0;
    if (values.size() < plan.info.buffers.size()) {
        throw std::runtime_error("materialization sources are missing buffer descriptors");
    }
    snapshot.buffers.assign(values.begin(), values.begin() + plan.info.buffers.size());
    cursor += plan.info.buffers.size();

    snapshot.images.resize(plan.info.images.size());
    for (std::uint32_t i = 0; i < plan.info.images.size(); i++) {
        const auto& image = plan.info.images[i];
        if (image.source >= plan.descriptorSources.size()) {
            throw std::runtime_error("image resource references an unknown descriptor source");
        }
        if (image.table != NoTable) {
            snapshot.images[i].dwordCount = 8u;
            continue;
        }
        if (cursor >= values.size()) {
            throw std::runtime_error("materialization sources are missing image descriptors");
        }
        auto descriptor = values[cursor];
        cursor++;
        if (descriptor.dwordCount != 8u) {
            throw std::runtime_error("image descriptor has an invalid width");
        }
        const auto activity = plan.descriptorSources.size() + i;
        if (activity < activeSources.size() && activeSources[activity] == 0u) {
            descriptor = DescriptorValue{};
            descriptor.dwordCount = 8u;
        }
        if (!validImageDescriptor(descriptor, image.r128) && !nullImageDescriptor(image, descriptor)) {
            throw std::runtime_error("runtime image descriptor is invalid");
        }
        snapshot.images[i] = descriptor;
    }

    snapshot.samplers.resize(plan.info.samplers.size());
    for (std::uint32_t i = 0; i < plan.info.samplers.size(); i++) {
        if (plan.info.samplers[i].table != NoTable) {
            snapshot.samplers[i].dwordCount = 4u;
            continue;
        }
        if (cursor >= values.size()) {
            throw std::runtime_error("materialization sources are missing sampler descriptors");
        }
        snapshot.samplers[i] = values[cursor];
        cursor++;
    }
}

void materializeForcedDegamma(const IrResourcePlan& plan, ResourceSnapshot& snapshot) {
    for (std::uint32_t index = 0; index < plan.info.images.size(); ++index) {
        auto& descriptor = snapshot.images.at(index);
        if (((descriptor.dwords[1] >> 20u) & 0x1ffu) != static_cast<std::uint32_t>(IrBufferFormat::Format8_8_8_8UInt)) continue;
        bool forced = false;
        bool plain = false;
        for (const auto& pair : plan.info.sampledPairs) {
            if (pair.image != index) continue;
            const auto& sampler = snapshot.samplers.at(pair.sampler);
            if (sampler.dwordCount != 4u) throw std::runtime_error("FORCE_DEGAMMA requires four sampler descriptor words");
            const bool enabled = (sampler.dwords[0] & (1u << 20u)) != 0u;
            forced = forced || enabled;
            plain = plain || !enabled;
        }
        if (!forced) continue;
        const auto& image = plan.info.images.at(index);
        if (!image.singleMipPlaneBiasCompatible || image.dimension != RdnaImageDimension::Dim2D || image.table != NoTable || image.resourceClass != ImageResourceClass::Sampled || image.depthCompare || image.packed) throw std::runtime_error("FORCE_DEGAMMA on RGBA8_UINT requires a direct plain biased pixel sample");
        if (plain) throw std::runtime_error("FORCE_DEGAMMA on RGBA8_UINT with mixed sampler states is not implemented");
        if ((descriptor.dwords[3] >> 28u) != 10u || (descriptor.dwords[4] & 0x1fffu) != 0u || (descriptor.dwords[3] & 0xff000u) != 0u || (descriptor.dwords[5] & 0xf0u) != 0u) throw std::runtime_error("FORCE_DEGAMMA on RGBA8_UINT requires a one-slice volume at mip zero");
        descriptor.dwords[1] = (descriptor.dwords[1] & ~(0x1ffu << 20u)) | (static_cast<std::uint32_t>(IrBufferFormat::Format8_8_8_8Srgb) << 20u);
    }
}

std::uint32_t colorCompareReference(IrBufferFormat format) {
    switch (format) {
    case IrBufferFormat::Format16_16UNorm:
    case IrBufferFormat::Format11_11_10UNorm: case IrBufferFormat::Format10_11_11UNorm: case IrBufferFormat::Format2_10_10_10UNorm:
    case IrBufferFormat::Format10_10_10_2UNorm: case IrBufferFormat::Format16_16_16_16UNorm:
        return EmulatedCompare::ReferenceUnorm;
    case IrBufferFormat::Format8UNorm: case IrBufferFormat::Format8_8UNorm: case IrBufferFormat::Format8_8_8_8UNorm:
    case IrBufferFormat::Format8Srgb: case IrBufferFormat::Format8_8Srgb: case IrBufferFormat::Format8_8_8_8Srgb:
        return EmulatedCompare::ReferenceZero;
    case IrBufferFormat::Format8SNorm: case IrBufferFormat::Format16SNorm: case IrBufferFormat::Format8_8SNorm: case IrBufferFormat::Format16_16SNorm:
    case IrBufferFormat::Format11_11_10SNorm: case IrBufferFormat::Format10_11_11SNorm: case IrBufferFormat::Format2_10_10_10SNorm:
    case IrBufferFormat::Format10_10_10_2SNorm: case IrBufferFormat::Format8_8_8_8SNorm: case IrBufferFormat::Format16_16_16_16SNorm:
        return EmulatedCompare::ReferenceSnorm;
    case IrBufferFormat::Format16Float: case IrBufferFormat::Format16_16Float: case IrBufferFormat::Format11_11_10Float:
    case IrBufferFormat::Format10_11_11Float: case IrBufferFormat::Format32_32Float: case IrBufferFormat::Format16_16_16_16Float:
        return EmulatedCompare::ReferenceFloat;
    default:
        throw std::runtime_error("comparison sampling of a color texture is implemented only for float, unorm and snorm formats (format " + std::to_string(static_cast<std::uint32_t>(format)) + ")");
    }
}

std::uint32_t emulatedCompareState(const ShaderInfo& info, const ResourceSnapshot& snapshot, std::uint32_t index) {
    const auto& image = info.images[index];
    const auto& descriptor = snapshot.images[index];
    if (!image.depthCompare || descriptor.dwordCount != 8u || nullImageDescriptor(image, descriptor)) return 0u;
    const auto format = rawImageFormat(descriptor);
    if (format == IrBufferFormat::Format32Float || format == IrBufferFormat::Format16UNorm || IsDepthBitsTexture(descriptor.dwords[1], descriptor.dwords[3])) return 0u;
    if (image.table != NoTable || (image.emulatedCompare & EmulatedCompare::Unsupported) != 0u) throw std::runtime_error("unsupported color comparison image instructions");
    if ((image.emulatedCompare & EmulatedCompare::RequiresSingleLevel) != 0u && ((descriptor.dwords[3] >> 12u) & 0xfu) != ((descriptor.dwords[3] >> 16u) & 0xfu)) throw std::runtime_error("color comparison requires a single mip level");
    const auto reference = colorCompareReference(format);
    const auto type = rawImageType(descriptor);
    if (type != ImageType::Color2D && type != ImageType::Color2DArray && type != ImageType::Cube) throw std::runtime_error("comparison sampling of a color texture is implemented only for 2D, 2D array and cube views");
    bool redChannel = false;
    bool compareChannel = false;
    for (std::uint32_t channel = 0u; channel < 4u; ++channel) {
        const auto select = (descriptorImageSwizzle(descriptor) >> (channel * 3u)) & 0x7u;
        if (select == 2u || select == 3u) throw std::runtime_error("comparison sampling of a color texture is implemented only when every view channel is a constant or a color channel");
        redChannel = redChannel || select == 4u;
        compareChannel = compareChannel || select >= 4u;
    }
    if (compareChannel && !redChannel) throw std::runtime_error("comparison sampling of a color texture is implemented only when a view channel selects red");
    std::optional<std::uint32_t> samplerState;
    for (const auto& pair : info.sampledPairs) {
        if (pair.image != index) continue;
        if (pair.sampler >= info.samplers.size() || info.samplers[pair.sampler].table != NoTable) throw std::runtime_error("comparison sampling of a color texture through a sampler table is not implemented");
        if (pair.sampler >= snapshot.samplers.size() || snapshot.samplers[pair.sampler].dwordCount != 4u) throw std::runtime_error("comparison sampling of a color texture has no sampler descriptor");
        const auto& words = snapshot.samplers[pair.sampler].dwords;
        const auto clampX = words[0] & 0x7u;
        const auto clampY = (words[0] >> 3u) & 0x7u;
        const auto function = (words[0] >> 12u) & 0x7u;
        const bool unnormalized = ((words[0] >> 15u) & 0x1u) != 0u;
        if (((words[0] >> 29u) & 0x3u) != 0u) throw std::runtime_error("comparison sampling of a color texture through a min or max reduction sampler is not implemented");
        const auto magFilter = (words[2] >> 20u) & 0x3u;
        const auto minFilter = (words[2] >> 22u) & 0x3u;
        const bool constantZero = reference == EmulatedCompare::ReferenceZero;
        if (type == ImageType::Cube && magFilter == 1u && ((words[0] >> 28u) & 1u) == 0u && !constantZero) {
            throw std::runtime_error("bilinear cube comparison requires DISABLE_CUBE_WRAP to avoid seamless face filtering");
        }
        const auto addressMode = [](std::uint32_t clamp) {
            if (clamp == 0u) return EmulatedCompare::AddressWrap;
            if (clamp == 2u) return EmulatedCompare::AddressEdge;
            if (clamp == 6u) return EmulatedCompare::AddressBorder;
            throw std::runtime_error("comparison sampling of a color texture is implemented only with wrap, clamp-to-edge or clamp-to-border addressing");
        };
        const auto addressX = addressMode(clampX);
        const auto addressY = addressMode(clampY);
        const auto borderType = (words[3] >> 30u) & 0x3u;
        const bool border = addressX == EmulatedCompare::AddressBorder || addressY == EmulatedCompare::AddressBorder;
        if (border && borderType == 3u) throw std::runtime_error("comparison sampling of a color texture with a border color table is not implemented");
        if (!constantZero && (magFilter != minFilter || magFilter > 1u)) throw std::runtime_error("comparison sampling of a color texture is implemented only with equal point or bilinear minification and magnification filters");
        if (unnormalized) throw std::runtime_error("comparison sampling of a color texture does not implement unnormalized coordinates");
        const auto state = EmulatedCompare::Enabled | (function << EmulatedCompare::FunctionShift) | (magFilter == 1u ? EmulatedCompare::Linear : 0u)
            | (addressX << EmulatedCompare::ClampXShift) | (addressY << EmulatedCompare::ClampYShift) | (border && borderType == 2u ? EmulatedCompare::BorderWhite : 0u);
        if (samplerState.has_value() && *samplerState != state) throw std::runtime_error("comparison sampling of a color texture through samplers that disagree is not implemented");
        samplerState = state;
    }
    if (!samplerState.has_value()) throw std::runtime_error("comparison sampling of a color texture has no paired sampler");
    const bool singleLevel = ((descriptor.dwords[3] >> 12u) & 0xfu) == ((descriptor.dwords[3] >> 16u) & 0xfu);
    return *samplerState | (reference << EmulatedCompare::ReferenceShift) | (singleLevel ? EmulatedCompare::SingleLevel : 0u);
}

void materializeTables(const IrResourcePlan& plan, ResourceSnapshot& snapshot, bool allowDirectImageRefusal) {
    for (std::uint32_t i = 0u; i < plan.info.buffers.size(); ++i) {
        const auto& buffer = plan.info.buffers[i];
        const bool scalarOnly = buffer.scalar && !buffer.written && !buffer.atomic && !buffer.formatted && buffer.imageAlias == BufferResource::NoImageAlias
            && std::none_of(plan.memoryInfo.begin(), plan.memoryInfo.end(), [i](const MemoryInfo& memory) { return memory.kind == ResourceKind::Buffer && memory.resource == i; });
        if (scalarOnly) {
            snapshot.buffers.at(i).dwords[1] &= ~0x40000000u;
            snapshot.buffers.at(i).dwords[3] = 0u;
        }
        const auto decoded = decodeBufferDescriptor(snapshot.buffers.at(i));
        if (decoded.Type() != 0u) {
            const auto& words = snapshot.buffers.at(i).dwords;
            char detail[64];
            std::snprintf(detail, sizeof(detail), " (buffer %u V# %08x %08x %08x %08x)", i, words[0], words[1], words[2], words[3]);
            throw std::runtime_error(std::string("buffer descriptor uses an unsupported type") + detail);
        }
        if (plan.stage != IrShaderStage::Compute && decoded.AddTid()) throw std::runtime_error("buffer ADD_TID is only valid for compute shaders");
    }
    if (snapshot.flattenedSrt.size() != plan.srtReads.size()) throw std::runtime_error("runtime SRT size differs from the static interface");
    for (std::uint32_t i = 0u; i < plan.info.images.size(); ++i) {
        const auto& image = plan.info.images[i];
        if (image.table != NoTable) continue;
        const auto mode = ResourceMaterializer::DirectImageMode(plan.info, plan.stage, i, snapshot.images.at(i), allowDirectImageRefusal);
        if (PipelineSpecialization::IsDeferredDirectImageMode(mode)) {
            bool allowed = true;
            bool found = false;
            for (const auto& memory : plan.memoryInfo) {
                if (memory.kind != ResourceKind::Image || memory.resource != i) continue;
                found = true;
                allowed = allowed && directRefusalMemory(memory);
            }
            if (!allowed || !found) static_cast<void>(ResourceMaterializer::RuntimeImageMode(image, snapshot.images.at(i), plan.info.runtimeImageModes.at(i)));
        }
    }
}

void materializeSrtGuards(const IrResourcePlan& plan, ResourceSnapshot& snapshot) {
    const auto& guarded = plan.guardedSrtSlots;
    const auto flags = snapshot.flattenedSrt.size();
    snapshot.flattenedSrt.resize(flags + guarded.size(), 0u);
    for (std::uint32_t record = 0u; record < snapshot.srtPoison.size(); ++record) {
        const auto& poison = snapshot.srtPoison[record];
        const auto found = std::lower_bound(guarded.begin(), guarded.end(), poison.slot);
        if (found == guarded.end() || *found != poison.slot) throw std::runtime_error("SRT read at flat offset " + std::to_string(poison.slot) + " reads inaccessible memory but has no guard");
        snapshot.flattenedSrt[flags + static_cast<std::size_t>(found - guarded.begin())] = record + 1u;
        snapshot.flattenedSrt.insert(snapshot.flattenedSrt.end(), {poison.pc, static_cast<std::uint32_t>(poison.address), static_cast<std::uint32_t>(poison.address >> 32u)});
    }
}

struct DecodeMemoKey {
    std::uint64_t artifact = 0;
    std::uint32_t resource = 0;
    std::array<std::uint32_t, 8> words{};

    bool operator==(const DecodeMemoKey& other) const = default;
};

struct DecodeMemoHash {
    std::size_t operator()(const DecodeMemoKey& key) const {
        std::uint64_t hash = 0xcbf29ce484222325ull;
        const auto mix = [&](std::uint64_t value) {
            hash ^= value;
            hash *= 0x100000001b3ull;
        };
        mix(key.artifact);
        mix(key.resource);
        for (const auto word : key.words) mix(word);
        return static_cast<std::size_t>(hash);
    }
};

struct DecodeMemo {
    std::mutex mutex;
    std::unordered_map<DecodeMemoKey, std::uint32_t, DecodeMemoHash> entries;
};

DecodeMemo& decodeMemo() {
    static DecodeMemo memo;
    return memo;
}

constexpr std::uint32_t DecodeMemoEntries = 1u << 20u;
constexpr std::uint32_t DecodeFailed = 0x80000000u;

bool FastTableResolve() {
    static const bool fast = std::getenv("APS5_NO_FAST_TABLE_RESOLVE") == nullptr;
    return fast;
}

constexpr std::size_t FlatDecodeEntries = std::size_t{1} << 19u;

struct FlatDecodeSlot {
    DecodeMemoKey key;
    std::uint32_t value = 0;
    std::uint32_t used = 0;
};

struct FlatDecodeMemo {
    std::mutex mutex;
    std::vector<FlatDecodeSlot> slots;
    std::size_t count = 0;
};

FlatDecodeMemo& flatDecodeMemo() {
    static FlatDecodeMemo memo;
    return memo;
}

std::uint64_t flatDecodeHash(const DecodeMemoKey& key) {
    const auto mix = [](std::uint64_t value) {
        value ^= value >> 33u;
        value *= 0xff51afd7ed558ccdull;
        value ^= value >> 33u;
        value *= 0xc4ceb9fe1a85ec53ull;
        value ^= value >> 33u;
        return value;
    };
    auto hash = mix(key.artifact ^ (static_cast<std::uint64_t>(key.resource) << 56u));
    for (std::size_t word = 0; word < key.words.size(); word += 2u) hash = mix(hash ^ (static_cast<std::uint64_t>(key.words[word]) | (static_cast<std::uint64_t>(key.words[word + 1u]) << 32u)));
    return hash;
}

const FlatDecodeSlot* flatDecodeFind(const FlatDecodeMemo& memo, const DecodeMemoKey& key, std::uint64_t hash) {
    if (memo.slots.empty()) return nullptr;
    const auto mask = memo.slots.size() - 1u;
    for (auto slot = static_cast<std::size_t>(hash) & mask;; slot = (slot + 1u) & mask) {
        const auto& entry = memo.slots[slot];
        if (entry.used == 0u) return nullptr;
        if (entry.key == key) return &entry;
    }
}

void flatDecodePlace(std::vector<FlatDecodeSlot>& slots, const DecodeMemoKey& key, std::uint64_t hash, std::uint32_t value) {
    const auto mask = slots.size() - 1u;
    auto slot = static_cast<std::size_t>(hash) & mask;
    while (slots[slot].used != 0u) slot = (slot + 1u) & mask;
    slots[slot] = {key, value, 1u};
}

std::uint32_t decodeTableEntry(const ImageResource& image, std::span<const ImageResource> modes, const DescriptorValue& words, std::uint64_t memoKey, std::uint32_t resource) {
    DecodeMemoKey key{memoKey, resource, words.dwords};
    if (memoKey != 0u && FastTableResolve()) {
        const auto hash = flatDecodeHash(key);
        auto& memo = flatDecodeMemo();
        {
            std::lock_guard lock(memo.mutex);
            if (const auto* found = flatDecodeFind(memo, key, hash)) return found->value;
        }
        const auto match = ResourceMaterializer::TryRuntimeImageMode(image, words, modes);
        const auto value = match.mode.has_value() ? *match.mode : DecodeFailed | static_cast<std::uint32_t>(match.reason);
        std::lock_guard lock(memo.mutex);
        if (flatDecodeFind(memo, key, hash) != nullptr) return value;
        if (memo.count >= FlatDecodeEntries) {
            std::fill(memo.slots.begin(), memo.slots.end(), FlatDecodeSlot{});
            memo.count = 0;
        }
        if ((memo.count + 1u) * 2u > memo.slots.size()) {
            std::vector<FlatDecodeSlot> grown(std::max<std::size_t>(4096u, memo.slots.size() * 2u));
            for (const auto& entry : memo.slots) {
                if (entry.used != 0u) flatDecodePlace(grown, entry.key, flatDecodeHash(entry.key), entry.value);
            }
            memo.slots = std::move(grown);
        }
        flatDecodePlace(memo.slots, key, hash, value);
        memo.count++;
        return value;
    }
    if (memoKey != 0u) {
        auto& memo = decodeMemo();
        std::lock_guard lock(memo.mutex);
        if (const auto found = memo.entries.find(key); found != memo.entries.end()) return found->second;
    }
    const auto match = ResourceMaterializer::TryRuntimeImageMode(image, words, modes);
    const auto value = match.mode.has_value() ? *match.mode : DecodeFailed | static_cast<std::uint32_t>(match.reason);
    if (memoKey != 0u) {
        auto& memo = decodeMemo();
        std::lock_guard lock(memo.mutex);
        if (memo.entries.size() >= DecodeMemoEntries) memo.entries.clear();
        memo.entries.emplace(key, value);
    }
    return value;
}

}

ResolvedImageTables ResourceMaterializer::ResolveImageTables(const ShaderInfo& info, const ResourceSnapshot& snapshot, std::uint64_t memoKey) {
    namespace Abi = ImageTableAbi;
    ResolvedImageTables out;
    std::uint32_t count = 0;
    for (const auto& image : info.images) {
        if (image.table != NoTable) count = std::max(count, image.table + 1u);
    }
    for (const auto& sampler : info.samplers) {
        if (sampler.table != NoTable) count = std::max(count, sampler.table + 1u);
    }
    if (count == 0u) return out;
    const auto& tables = snapshot.tables;
    if (tables.tables.size() != count) throw std::runtime_error("image table snapshot disagrees with the static tables");
    if (info.runtimeImageModes.size() != info.images.size()) throw std::runtime_error("prepared runtime image modes are missing");
    out.map.assign(Abi::HeaderWords + Abi::TableWords * count, 0u);
    out.map[Abi::HeaderVersion] = Abi::Version;
    out.poison.push_back({{}, 0u, NoTable, PoisonReason::OutsideSnapshot});
    out.poison.push_back({{}, 0u, NoTable, PoisonReason::OutsideDomain});
    static const bool fastResolve = std::getenv("APS5_NO_FAST_TABLE_RESOLVE") == nullptr;
    using PoisonKey = std::tuple<std::uint32_t, PoisonReason, std::array<std::uint32_t, 8>, std::uint32_t>;
    using ElementKey = std::tuple<std::array<std::uint32_t, 8>, RdnaImageDimension, bool, DescriptorBindingKind>;
    struct ResolveKeyHash {
        static std::uint64_t Mix(std::uint64_t hash, std::uint64_t value) { return (hash ^ value) * 0x100000001b3ull; }
        std::size_t operator()(const PoisonKey& key) const noexcept {
            auto hash = Mix(Mix(0xcbf29ce484222325ull, std::get<0>(key)), static_cast<std::uint64_t>(std::get<1>(key)));
            for (const auto word : std::get<2>(key)) hash = Mix(hash, word);
            return static_cast<std::size_t>(Mix(hash, std::get<3>(key)));
        }
        std::size_t operator()(const ElementKey& key) const noexcept {
            auto hash = 0xcbf29ce484222325ull;
            for (const auto word : std::get<0>(key)) hash = Mix(hash, word);
            hash = Mix(Mix(Mix(hash, static_cast<std::uint64_t>(std::get<1>(key))), std::get<2>(key) ? 1u : 0u), static_cast<std::uint64_t>(std::get<3>(key)));
            return static_cast<std::size_t>(hash ^ (hash >> 29u));
        }
    };
    std::map<PoisonKey, std::uint32_t> poisonIndex;
    std::unordered_map<PoisonKey, std::uint32_t, ResolveKeyHash> poisonHash;
    // Flat poison index (APS5_NO_FLAT_POISON_INDEX=1: the node map above): open addressing over the
    // indices of out.poison that poison() inserted (slot = index + 1, 0 empty), the key compared with
    // the record itself. It answers exactly what poisonHash answers: the first index inserted for an
    // equal (resource, reason, words, dword count), else a new one. Tables of the giant streaming
    // programs (0x80001e5f00: ~40k records per resolve, almost all poisoned) spent ~1/3 of a resolve
    // allocating map nodes.
    static const bool flatPoison = fastResolve && std::getenv("APS5_NO_FLAT_POISON_INDEX") == nullptr;
    std::vector<std::uint32_t> poisonSlots;
    std::size_t poisonInserted = 0;
    const auto poisonKeyHash = [](std::uint32_t resource, PoisonReason reason, const std::array<std::uint32_t, 8>& dwords, std::uint32_t dwordCount) {
        auto hash = ResolveKeyHash::Mix(ResolveKeyHash::Mix(0xcbf29ce484222325ull, resource), static_cast<std::uint64_t>(reason));
        for (const auto word : dwords) hash = ResolveKeyHash::Mix(hash, word);
        hash = ResolveKeyHash::Mix(hash, dwordCount);
        return hash ^ (hash >> 29u) ^ (hash >> 47u);
    };
    const auto flatPoisonIndex = [&](std::uint32_t resource, PoisonReason reason, const DescriptorValue& value, std::uint32_t index) -> std::pair<std::uint32_t, bool> {
        if ((poisonInserted + 1u) * 2u > poisonSlots.size()) {
            std::vector<std::uint32_t> grown(std::max<std::size_t>(1024u, poisonSlots.size() * 2u), 0u);
            const auto mask = grown.size() - 1u;
            for (const auto slot : poisonSlots) {
                if (slot == 0u) continue;
                const auto& record = out.poison[slot - 1u];
                auto at = static_cast<std::size_t>(poisonKeyHash(record.resource, record.reason, record.words, record.dwordCount)) & mask;
                while (grown[at] != 0u) at = (at + 1u) & mask;
                grown[at] = slot;
            }
            poisonSlots = std::move(grown);
        }
        const auto mask = poisonSlots.size() - 1u;
        auto at = static_cast<std::size_t>(poisonKeyHash(resource, reason, value.dwords, value.dwordCount)) & mask;
        while (poisonSlots[at] != 0u) {
            const auto candidate = poisonSlots[at] - 1u;
            const auto& record = out.poison[candidate];
            if (record.resource == resource && record.reason == reason && record.dwordCount == value.dwordCount && record.words == value.dwords) return {candidate, false};
            at = (at + 1u) & mask;
        }
        poisonSlots[at] = index + 1u;
        ++poisonInserted;
        return {index, true};
    };
    std::set<std::uint32_t> expected;
    std::vector<bool> expectedMarks;
    std::size_t expectedCount = 0;
    const auto poison = [&](std::uint32_t resource, PoisonReason reason, const DescriptorValue* words) {
        const auto value = words != nullptr ? *words : DescriptorValue{};
        const auto index = static_cast<std::uint32_t>(out.poison.size());
        // A new index's record is pushed just below, before any later lookup reads it through its slot.
        const auto [found, inserted] = flatPoison ? flatPoisonIndex(resource, reason, value, index) : fastResolve ? [&] { const PoisonKey key{resource, reason, value.dwords, value.dwordCount}; const auto [at, added] = poisonHash.try_emplace(key, index); return std::pair<std::uint32_t, bool>{at->second, added}; }() : [&] { const PoisonKey key{resource, reason, value.dwords, value.dwordCount}; const auto [at, added] = poisonIndex.try_emplace(key, index); return std::pair<std::uint32_t, bool>{at->second, added}; }();
        if (inserted) out.poison.push_back({value.dwords, value.dwordCount, resource, reason});
        if (fastResolve) {
            if (found >= expectedMarks.size()) expectedMarks.resize(static_cast<std::size_t>(found) + 1u, false);
            if (!expectedMarks[found]) {
                expectedMarks[found] = true;
                ++expectedCount;
            }
        } else {
            expected.insert(found);
        }
        return Abi::PoisonCode(found);
    };
    std::map<ElementKey, std::uint32_t> elementIndex;
    std::unordered_map<ElementKey, std::uint32_t, ResolveKeyHash> elementHash;
    std::vector<std::uint32_t> directSamplers(info.images.size(), 0u);
    std::vector<bool> reducingSampler(info.images.size(), false);
    for (const auto& pair : info.sampledPairs) {
        const auto& sampler = info.samplers.at(pair.sampler);
        if (sampler.table != NoTable) continue;
        directSamplers.at(pair.image) |= 1u << pair.sampler;
        const auto& words = snapshot.samplers.at(pair.sampler).dwords;
        if (reducesBetweenTexels(words[0], words[2])) reducingSampler.at(pair.image) = true;
    }
    const auto imageRecord = [&](std::uint32_t resource, const DescriptorValue& words) {
        const auto& image = info.images[resource];
        const auto& modes = info.runtimeImageModes[resource];
        std::array<std::uint32_t, Abi::ImageRecordWords> record{0u, words.dwords[3], 0u};
        if ((words.availableDwordMask & (1u << 3u)) == 0u) {
            record[Abi::ImageRecordFlags] = poison(resource, PoisonReason::OutsideSnapshot, &words);
            const auto offset = static_cast<std::uint32_t>(out.map.size());
            out.map.insert(out.map.end(), record.begin(), record.end());
            return offset;
        }
        const auto decoded = decodeTableEntry(image, modes, words, memoKey, resource);
        if ((decoded & DecodeFailed) != 0u) {
            record[Abi::ImageRecordFlags] = poison(resource, static_cast<PoisonReason>(decoded & ~DecodeFailed), &words);
        } else {
            const auto& mode = modes.at(decoded);
            if (mode.dimension == RdnaImageDimension::Dim2DMsaa || mode.dimension == RdnaImageDimension::Dim2DMsaaArray) {
                record[Abi::ImageRecordFlags] = poison(resource, PoisonReason::Multisampled, &words);
            } else if (requiresPointSampler(mode) && reducingSampler[resource]) {
                record[Abi::ImageRecordFlags] = poison(resource, PoisonReason::Reduction, &words);
            } else {
                auto element = words.dwords;
                if (mode.conversionFormat != IrBufferFormat::Invalid || mode.depthBits) element[3] = (element[3] & ~0xfffu) | ShaderImageIdentitySwizzle;
                const auto kind = DescriptorBindingForImage(mode);
                const ElementKey key{element, mode.dimension, mode.depthCompare, kind};
                const auto next = static_cast<std::uint32_t>(out.elements.size());
                std::uint32_t found = 0;
                bool inserted = false;
                if (fastResolve) {
                    const auto [at, added] = elementHash.try_emplace(key, next);
                    found = at->second;
                    inserted = added;
                    if (inserted && out.elements.size() >= Abi::ImageElementBudget) elementHash.erase(at);
                } else {
                    const auto [at, added] = elementIndex.try_emplace(key, next);
                    found = at->second;
                    inserted = added;
                    if (inserted && out.elements.size() >= Abi::ImageElementBudget) elementIndex.erase(at);
                }
                if (inserted && out.elements.size() >= Abi::ImageElementBudget) {
                    record[Abi::ImageRecordFlags] = poison(resource, PoisonReason::Budget, &words);
                } else {
                    if (inserted) out.elements.push_back({element, mode.dimension, mode.depthCompare, kind, 0u});
                    out.elements[found].samplers |= directSamplers[resource];
                    record[Abi::ImageRecordFlags] = decoded << Abi::ModeShift;
                    record[Abi::ImageRecordElement] = found;
                }
            }
        }
        const auto offset = static_cast<std::uint32_t>(out.map.size());
        out.map.insert(out.map.end(), record.begin(), record.end());
        return offset;
    };
    const auto samplerRecord = [&](std::uint32_t resource, const DescriptorValue& words) {
        std::array<std::uint32_t, Abi::SamplerRecordWords> record{0u, 0u};
        if ((words.dwords[0] & (1u << 15u)) != 0u) {
            record[Abi::SamplerRecordFlags] = poison(resource, PoisonReason::Unnormalized, &words);
        } else if (((words.dwords[0] >> 29u) & 3u) != 0u) {
            record[Abi::SamplerRecordFlags] = poison(resource, PoisonReason::Reduction, &words);
        } else if (out.samplers.size() + Abi::SamplerEntryElements > Abi::SamplerElementBudget) {
            record[Abi::SamplerRecordFlags] = poison(resource, PoisonReason::Budget, &words);
        } else {
            record[Abi::SamplerRecordElement] = static_cast<std::uint32_t>(out.samplers.size());
            for (std::uint32_t variant = 0; variant < 2u; variant++) {
                const bool compare = variant == 1u;
                for (std::uint32_t point = 0; point < 2u; point++) {
                    ImageTableSamplerElement element;
                    std::copy_n(words.dwords.begin(), 4u, element.words.begin());
                    element.compare = compare;
                    element.point = point != 0u;
                    out.samplers.push_back(element);
                }
            }
        }
        const auto offset = static_cast<std::uint32_t>(out.map.size());
        out.map.insert(out.map.end(), record.begin(), record.end());
        return offset;
    };
    std::vector<bool> outside(count, false);
    std::vector<std::uint32_t> codeMapped;
    const auto fill = [&](std::uint32_t resource, std::uint32_t table, bool sampler) {
        const auto& column = tables.tables.at(table);
        const auto header = Abi::TableHeader(table);
        out.map[header + Abi::TableKeyCount] = column.keys;
        out.map[header + Abi::TableSizeLow] = static_cast<std::uint32_t>(column.size);
        out.map[header + Abi::TableSizeHigh] = static_cast<std::uint32_t>(column.size >> 32u);
        out.map[header + Abi::TableBaseLow] = static_cast<std::uint32_t>(column.base);
        out.map[header + Abi::TableBaseHigh] = static_cast<std::uint32_t>(column.base >> 32u);
        if (column.codes.size() != column.keys) throw std::runtime_error("image table snapshot has a truncated map");
        if (column.fault != 0u) {
            out.map[header + Abi::TableFault] = poison(resource, static_cast<PoisonReason>(column.fault & ~Abi::PoisonFlag), nullptr);
        } else if (!sampler && (info.images[resource].tableOperation == TableOperation::Unsupported || info.runtimeImageModes[resource].empty())) {
            out.map[header + Abi::TableFault] = poison(resource, PoisonReason::UnsupportedOperation, nullptr);
        }
        outside[table] = column.outside;
        const auto mapStart = static_cast<std::uint32_t>(out.map.size());
        out.map[header + Abi::TableMapStart] = mapStart;
        out.map.resize(out.map.size() + column.keys, 0u);
        std::unordered_map<std::uint32_t, std::uint32_t> records;
        constexpr std::uint32_t Unmapped = std::numeric_limits<std::uint32_t>::max();
        if (fastResolve) codeMapped.assign(tables.words.size(), Unmapped);
        std::array<std::uint32_t, PoisonReasonCount> reasonCodes{};
        for (std::uint32_t key = 0; key < column.keys; key++) {
            const auto code = column.codes[key];
            std::uint32_t mapped = Abi::NullCode;
            if ((code & Abi::PoisonFlag) != 0u) {
                const auto reason = static_cast<PoisonReason>(code & ~Abi::PoisonFlag);
                const auto slot = static_cast<std::size_t>(reason);
                if (reason == PoisonReason::OutsideDomain) mapped = Abi::PoisonCode(Abi::OutsideDomainPoison);
                else if (fastResolve && slot < reasonCodes.size() && reasonCodes[slot] != 0u) mapped = reasonCodes[slot];
                else {
                    mapped = poison(resource, reason, nullptr);
                    if (fastResolve && slot < reasonCodes.size()) reasonCodes[slot] = mapped;
                }
            } else if (fastResolve && out.map[header + Abi::TableFault] == 0u && code < codeMapped.size() && codeMapped[code] != Unmapped) {
                mapped = codeMapped[code];
            } else if (out.map[header + Abi::TableFault] == 0u) {
                const auto& words = tables.words.at(code);
                if (sampler || (words.availableDwordMask & (1u << 3u)) == 0u || !nullImageDescriptor(info.images[resource], words)) {
                    if (fastResolve && code < codeMapped.size()) {
                        mapped = sampler ? samplerRecord(resource, words) : imageRecord(resource, words);
                    } else {
                        const auto [found, inserted] = records.try_emplace(code, 0u);
                        if (inserted) found->second = sampler ? samplerRecord(resource, words) : imageRecord(resource, words);
                        mapped = found->second;
                    }
                }
                if (fastResolve && code < codeMapped.size()) codeMapped[code] = mapped;
            }
            out.map[mapStart + key] = mapped;
        }
        if (sampler && out.map[header + Abi::TableFault] == 0u) {
            DescriptorValue zero;
            zero.dwordCount = 4u;
            out.map[header + Abi::TableOutsideCode] = samplerRecord(resource, zero);
        }
    };
    for (std::uint32_t resource = 0; resource < info.images.size(); resource++) {
        if (info.images[resource].table != NoTable) fill(resource, info.images[resource].table, false);
    }
    for (std::uint32_t resource = 0; resource < info.samplers.size(); resource++) {
        if (info.samplers[resource].table != NoTable) fill(resource, info.samplers[resource].table, true);
    }
    out.map[Abi::HeaderElementCount] = static_cast<std::uint32_t>(out.elements.size());
    out.map[Abi::HeaderSamplerCount] = static_cast<std::uint32_t>(out.samplers.size());
    out.map[Abi::HeaderPoisonCount] = static_cast<std::uint32_t>(out.poison.size());
    out.map[Abi::HeaderTableCount] = count;
    out.map[Abi::HeaderElementFaults] = static_cast<std::uint32_t>(out.map.size());
    out.map.resize(out.map.size() + out.elements.size(), 0u);
    out.map[Abi::HeaderSamplerFaults] = static_cast<std::uint32_t>(out.map.size());
    out.map.resize(out.map.size() + out.samplers.size() / Abi::SamplerEntryElements, 0u);
    out.faults = static_cast<std::uint32_t>((fastResolve ? expectedCount : expected.size()) + static_cast<std::size_t>(std::count(outside.begin(), outside.end(), true)));
    return out;
}

namespace {

ImageResource bindlessInterfaceImage(const ImageResource& image) {
    auto normalized = image;
    normalized.source = 0u;
    normalized.firstUsePc = 0u;
    if (normalized.table != NoTable) normalized.table = 0u;
    return normalized;
}

std::uint64_t mixBindlessInterface(std::uint64_t hash, const ImageResource& image) {
    const auto mix = [&](std::uint64_t value) {
        hash ^= value;
        hash *= 0x100000001b3ull;
    };
    mix(static_cast<std::uint64_t>(image.resourceClass));
    mix(static_cast<std::uint64_t>(image.numericClass));
    mix(static_cast<std::uint64_t>(image.dimension));
    mix(static_cast<std::uint64_t>(image.mipMode));
    mix(image.mipCount);
    mix(static_cast<std::uint64_t>(image.conversionFormat));
    mix(image.shaderSwizzle);
    mix((image.read ? 1u : 0u) | (image.written ? 2u : 0u) | (image.atomic ? 4u : 0u) | (image.atomic64 ? 8u : 0u) | (image.depthCompare ? 16u : 0u) | (image.cube ? 32u : 0u) | (image.r128 ? 64u : 0u) | (image.srgbDecode ? 128u : 0u));
    mix((image.srgbDecodeCompatible ? 1u : 0u) | (image.depthBits ? 2u : 0u) | (image.depthUnorm16 ? 4u : 0u) | (image.packed ? 8u : 0u) | (image.fmaskCompatible ? 16u : 0u) | (image.depthBitsCompatible ? 32u : 0u) | (image.constantSwizzle ? 64u : 0u) | (image.constantSwizzleCompatible ? 128u : 0u));
    mix((image.flatVolumeCompatible ? 1u : 0u) | (image.flatLineCompatible ? 2u : 0u) | (image.foldTexelOffsetsOnly ? 4u : 0u) | (image.table != NoTable ? 8u : 0u) | (image.cubeFaceCompatible ? 16u : 0u) | (image.singleMipPlaneBiasCompatible ? 32u : 0u) | (image.flatVolumeLineCompatible ? 64u : 0u));
    mix(image.srgbDecodeFormats);
    mix(image.byElements);
    mix(image.byComponents);
    mix(static_cast<std::uint64_t>(image.packedFormat));
    mix(image.emulatedCompare);
    mix(static_cast<std::uint64_t>(image.tableOperation));
    mix(image.levelUses);
    return hash;
}

struct BindlessInterfaceRegistry {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::shared_ptr<const BindlessInterface>> interfaces;
};

BindlessInterfaceRegistry& bindlessInterfaces() {
    static BindlessInterfaceRegistry registry;
    return registry;
}

}

bool ResourceMaterializer::BindlessImage(const ShaderInfo& info, IrShaderStage stage, std::uint32_t resource) {
    if (stage == IrShaderStage::Unknown || stage == IrShaderStage::Fetch || resource >= info.images.size() || resource >= info.runtimeImageModes.size()) return false;
    const auto& image = info.images[resource];
    return image.table != NoTable && image.resourceClass == ImageResourceClass::Sampled && image.tableOperation == TableOperation::Supported && !info.runtimeImageModes[resource].empty();
}

bool ResourceMaterializer::UsesBindlessImages(const ShaderInfo& info, IrShaderStage stage) {
    for (std::uint32_t resource = 0; resource < info.images.size(); ++resource) {
        if (BindlessImage(info, stage, resource)) return true;
    }
    return false;
}

bool ResourceMaterializer::RequiresPointSampler(const ImageResource& mode) {
    return requiresPointSampler(mode);
}

std::uint64_t ResourceMaterializer::BindlessInterfaceId(const ImageResource& image, std::span<const ImageResource> modes) {
    auto hash = mixBindlessInterface(0xcbf29ce484222325ull, bindlessInterfaceImage(image));
    hash = (hash ^ modes.size()) * 0x100000001b3ull;
    for (const auto& mode : modes) hash = mixBindlessInterface(hash, bindlessInterfaceImage(mode));
    return hash;
}

std::uint64_t ResourceMaterializer::RegisterBindlessInterface(const ImageResource& source, std::span<const ImageResource> modes) {
    const auto image = bindlessInterfaceImage(source);
    const auto id = BindlessInterfaceId(image, modes);
    const auto same = [&](const BindlessInterface& entry) {
        if (entry.image != image || entry.modes.size() != modes.size()) return false;
        for (std::size_t index = 0; index < modes.size(); ++index) {
            if (entry.modes[index] != bindlessInterfaceImage(modes[index])) return false;
        }
        return true;
    };
    auto& registry = bindlessInterfaces();
    std::lock_guard lock(registry.mutex);
    if (const auto found = registry.interfaces.find(id); found != registry.interfaces.end()) {
        if (!same(*found->second)) throw std::runtime_error("bindless images: two image interfaces share the id 0x" + hexText(id));
        return id;
    }
    auto entry = std::make_shared<BindlessInterface>();
    entry->id = id;
    entry->image = image;
    for (const auto& mode : modes) entry->modes.push_back(bindlessInterfaceImage(mode));
    registry.interfaces.emplace(id, std::move(entry));
    return id;
}

void ResourceMaterializer::RegisterBindlessInterfaces(const ShaderInfo& info, IrShaderStage stage) {
    for (std::uint32_t resource = 0; resource < info.images.size(); ++resource) {
        if (BindlessImage(info, stage, resource)) static_cast<void>(RegisterBindlessInterface(info.images[resource], info.runtimeImageModes[resource]));
    }
}

std::shared_ptr<const BindlessInterface> ResourceMaterializer::FindBindlessInterface(std::uint64_t id) {
    auto& registry = bindlessInterfaces();
    std::lock_guard lock(registry.mutex);
    const auto found = registry.interfaces.find(id);
    return found != registry.interfaces.end() ? found->second : nullptr;
}

std::vector<BindlessImageAbi::TableRegistration> ResourceMaterializer::BindlessTables(const ShaderInfo& info, IrShaderStage stage, const ResourceSnapshot& snapshot) {
    namespace Abi = ImageTableAbi;
    namespace Bindless = BindlessImageAbi;
    constexpr std::uint32_t DescriptorBytes = Bindless::Table::DescriptorWords * 4u;
    const auto& columns = snapshot.tables.tables;
    const auto readable = [&](const ImageTableColumnSnapshot& column, std::uint32_t key, std::uint32_t dwords) -> const DescriptorValue* {
        const auto code = column.codes[key];
        if ((code & Abi::PoisonFlag) != 0u || code >= snapshot.tables.words.size()) return nullptr;
        const auto& value = snapshot.tables.words[code];
        const auto mask = static_cast<std::uint8_t>((1u << dwords) - 1u);
        return (value.availableDwordMask & mask) == mask ? &value : nullptr;
    };
    const auto address = [](const ImageTableColumnSnapshot& column, std::uint32_t key) -> std::optional<std::uint64_t> {
        if (column.indirect) return key < column.addresses.size() && column.addresses[key] != 0u ? std::optional<std::uint64_t>(column.addresses[key]) : std::nullopt;
        const std::uint32_t offset = column.addend + key * column.stride;
        if (column.pointer) {
            const auto found = Abi::ScalarAddressDword(column.base, offset, column.offset);
            return found.has_value() && (*found & 3u) == 0u ? found : std::nullopt;
        }
        const auto dword = Abi::ScalarBufferDword(offset, column.offset, column.size);
        return dword.has_value() ? std::optional<std::uint64_t>(column.base + *dword) : std::nullopt;
    };
    std::map<std::uint32_t, std::set<Bindless::TableUse>> uses;
    for (std::uint32_t resource = 0; resource < info.images.size(); ++resource) {
        if (!BindlessImage(info, stage, resource)) continue;
        const auto& image = info.images[resource];
        if (image.table >= columns.size() || columns[image.table].fault != 0u) continue;
        const auto id = BindlessInterfaceId(image, info.runtimeImageModes[resource]);
        auto& tableUses = uses[image.table];
        bool paired = false;
        for (const auto& pair : info.sampledPairs) {
            if (pair.image != resource) continue;
            paired = true;
            const auto& sampler = info.samplers.at(pair.sampler);
            if (sampler.table == NoTable) {
                Bindless::TableUse use{id, Bindless::DirectSamplerClass, {}};
                std::copy_n(snapshot.samplers.at(pair.sampler).dwords.begin(), use.sampler.size(), use.sampler.begin());
                tableUses.insert(use);
                continue;
            }
            if (sampler.table >= columns.size() || columns[sampler.table].fault != 0u) continue;
            const auto& samplerColumn = columns[sampler.table];
            for (std::uint32_t key = 0; key < samplerColumn.codes.size(); ++key) {
                const auto* words = readable(samplerColumn, key, Bindless::Table::SamplerWords);
                if (words == nullptr) continue;
                Bindless::TableUse use{id, Bindless::TableSamplerClass, {}};
                std::copy_n(words->dwords.begin(), use.sampler.size(), use.sampler.begin());
                tableUses.insert(use);
            }
        }
        const bool unsampled = (image.levelUses & (ImageLevelUseQuery | ImageLevelUseLoad | ImageLevelUseMipLoad)) != 0u || image.byElements != 0u;
        if (!paired || unsampled) tableUses.insert(Bindless::TableUse{id, Bindless::NoSamplerClass, {}});
    }
    std::map<std::tuple<std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t>, std::set<Bindless::TableUse>> runs;
    for (const auto& [table, tableUses] : uses) {
        if (tableUses.empty()) continue;
        const auto& column = columns[table];
        const auto dwords = column.dwords;
        if (dwords == 0u || dwords > Bindless::Table::DescriptorWords) continue;
        const bool linear = !column.indirect && column.stride >= dwords * 4u && column.stride % 4u == 0u;
        const auto stride = linear ? column.stride : DescriptorBytes;
        std::uint64_t first = 0;
        std::uint32_t count = 0;
        const auto flush = [&] {
            if (count != 0u) runs[{first, count, stride, dwords}].insert(tableUses.begin(), tableUses.end());
            count = 0;
        };
        for (std::uint32_t key = 0; key < column.codes.size(); ++key) {
            const auto at = readable(column, key, dwords) != nullptr ? address(column, key) : std::nullopt;
            if (!at.has_value()) {
                flush();
                continue;
            }
            if (count != 0u && *at == first + static_cast<std::uint64_t>(count) * stride) {
                ++count;
                continue;
            }
            flush();
            first = *at;
            count = 1;
        }
        flush();
    }
    std::vector<Bindless::TableRegistration> out;
    out.reserve(runs.size());
    for (const auto& [run, runUses] : runs) out.push_back({std::get<0>(run), std::get<1>(run), std::get<2>(run), std::vector<Bindless::TableUse>(runUses.begin(), runUses.end()), std::get<3>(run)});
    return out;
}

std::vector<ImageResource> ResourceMaterializer::RuntimeImageModes(const ImageResource& image) {
    const bool table = image.table != NoTable;
    if (table && image.tableOperation == TableOperation::Unsupported) return {};
    std::vector<ImageResource> modes;
    const bool storage = image.resourceClass == ImageResourceClass::Storage;
    const auto append = [&](IrTextureNumericClass numeric, IrBufferFormat conversion, IrBufferFormat packed, bool depth, bool unorm16) {
        auto mode = image;
        mode.numericClass = numeric;
        mode.conversionFormat = conversion;
        mode.packedFormat = packed;
        mode.depthBits = depth;
        mode.depthUnorm16 = unorm16;
        mode.cube = false;
        mode.mipCount = mode.mipMode == ImageMipMode::DynamicStorage ? RuntimeAbi::StorageMipSlots : 1u;
        mode.shaderSwizzle = ShaderImageIdentitySwizzle;
        if (conversion == IrBufferFormat::Format11_11_10UNorm || conversion == IrBufferFormat::Format10_11_11Float) mode.shaderSwizzle = 0x2acu;
        modes.push_back(mode);
        const bool volumeLoad = (image.dimension == RdnaImageDimension::Dim2D || image.dimension == RdnaImageDimension::Dim2DArray) && image.fmaskCompatible;
        const bool volumeStore = image.dimension == RdnaImageDimension::Dim2DArray && storage && image.written && !image.read && !image.atomic && image.mipMode != ImageMipMode::DynamicStorage;
        const bool volumePlaneBias = image.dimension == RdnaImageDimension::Dim2D && image.singleMipPlaneBiasCompatible && !storage && image.table == NoTable && conversion == IrBufferFormat::Invalid;
        if ((volumeLoad || volumeStore || volumePlaneBias) && !depth && packed == IrBufferFormat::Invalid && image.byElements == 0u) {
            auto volume = mode;
            volume.dimension = RdnaImageDimension::Dim3D;
            modes.push_back(volume);
        }
        if (image.dimension == RdnaImageDimension::Dim3D && image.flatVolumeCompatible && !storage && !depth && conversion == IrBufferFormat::Invalid && packed == IrBufferFormat::Invalid) {
            for (const auto dimension : {RdnaImageDimension::Dim2D, RdnaImageDimension::Dim2DArray}) {
                auto plane = mode;
                plane.dimension = dimension;
                modes.push_back(plane);
            }
        }
        if (image.dimension == RdnaImageDimension::Dim3D && image.flatVolumeLineCompatible && !storage && !depth && conversion == IrBufferFormat::Invalid && packed == IrBufferFormat::Invalid) {
            auto line = mode;
            line.dimension = RdnaImageDimension::Dim1D;
            modes.push_back(line);
        }
        if (image.dimension == RdnaImageDimension::Dim2D && storage && !table && image.written && !image.read && !image.atomic && image.mipMode != ImageMipMode::DynamicStorage && !depth && packed == IrBufferFormat::Invalid && image.byElements == 0u) {
            auto line = mode;
            line.dimension = RdnaImageDimension::Dim1D;
            modes.push_back(line);
        }
        if (image.dimension == RdnaImageDimension::Dim2D && image.cubeFaceCompatible && !storage && !depth && conversion == IrBufferFormat::Invalid && packed == IrBufferFormat::Invalid) {
            auto face = mode;
            face.dimension = RdnaImageDimension::Dim2DArray;
            face.cube = true;
            modes.push_back(face);
        }
        if (image.dimension == RdnaImageDimension::Dim2D && image.flatLineCompatible && !storage && !depth && conversion == IrBufferFormat::Invalid && packed == IrBufferFormat::Invalid) {
            auto line = mode;
            line.dimension = RdnaImageDimension::Dim1D;
            modes.push_back(line);
        }
        if (image.dimension == RdnaImageDimension::Dim1D && !storage && !depth && conversion == IrBufferFormat::Invalid && packed == IrBufferFormat::Invalid) {
            auto plane = mode;
            plane.dimension = RdnaImageDimension::Dim2D;
            modes.push_back(plane);
        }
        if (image.dimension == RdnaImageDimension::Dim1DArray || image.dimension == RdnaImageDimension::Dim2DArray || image.dimension == RdnaImageDimension::Dim2DMsaaArray) {
            auto plain = mode;
            plain.dimension = image.dimension == RdnaImageDimension::Dim1DArray ? RdnaImageDimension::Dim1D : image.dimension == RdnaImageDimension::Dim2DArray ? RdnaImageDimension::Dim2D : RdnaImageDimension::Dim2DMsaa;
            modes.push_back(plain);
        }
        if (image.dimension == RdnaImageDimension::Dim2DArray) {
            mode.cube = true;
            modes.push_back(mode);
        }
    };
    if (image.atomic) {
        append(IrTextureNumericClass::Uint, IrBufferFormat::Invalid, IrBufferFormat::Invalid, false, false);
    } else if (image.packed) {
        for (std::uint32_t value = 1u; value <= 77u; ++value) {
            const auto format = static_cast<IrBufferFormat>(value);
            const auto info = GetFormatInfo(format);
            if (info.packedBitfield || info.componentCount == 0u || (storage && info.byteSize == 12u)) continue;
            if (image.byElements != 0u && info.byteSize * 8u * image.byElements > 32u) continue;
            const auto numeric = storage && (format == IrBufferFormat::Format32SInt || format == IrBufferFormat::Format32_32SInt || format == IrBufferFormat::Format32_32_32_32SInt) ? IrTextureNumericClass::Uint : SampledTextureNumericClass(format);
            bool supported = numeric != IrTextureNumericClass::Unsupported && (!storage || numeric != IrTextureNumericClass::Sint);
            for (std::uint32_t component = 0u; component < info.componentCount; ++component) {
                const auto bits = info.componentBits[component];
                const bool exactRead = info.type == SpirvFormatComponentType::Uint || info.type == SpirvFormatComponentType::Sint || (info.type == SpirvFormatComponentType::Unorm && bits <= 16u) || (info.type == SpirvFormatComponentType::Float && bits == 32u);
                const bool exactWrite = info.type == SpirvFormatComponentType::Uint || (bits == 32u && (info.type == SpirvFormatComponentType::Sint || info.type == SpirvFormatComponentType::Float));
                supported &= storage ? exactWrite : exactRead;
            }
            if (supported) append(numeric, IrBufferFormat::Invalid, format, false, false);
        }
    } else {
        if ((image.emulatedCompare & EmulatedCompare::NativeOffsetUnsupported) == 0u) append(IrTextureNumericClass::Float, IrBufferFormat::Invalid, IrBufferFormat::Invalid, false, false);
        if (!image.depthCompare) {
            append(IrTextureNumericClass::Uint, IrBufferFormat::Invalid, IrBufferFormat::Invalid, false, false);
            if (!storage) append(IrTextureNumericClass::Sint, IrBufferFormat::Invalid, IrBufferFormat::Invalid, false, false);
            append(IrTextureNumericClass::Uint, IrBufferFormat::Format11_11_10UInt, IrBufferFormat::Invalid, false, false);
            if (image.srgbDecodeCompatible && image.depthBitsCompatible) {
                append(IrTextureNumericClass::Uint, IrBufferFormat::Format11_11_10UNorm, IrBufferFormat::Invalid, false, false);
                append(IrTextureNumericClass::Uint, IrBufferFormat::Format10_11_11Float, IrBufferFormat::Invalid, false, false);
            }
            if (storage) {
                for (const auto format : {IrBufferFormat::Format5_6_5UNorm, IrBufferFormat::Format5_5_5_1UNorm, IrBufferFormat::Format4_4_4_4UNorm}) {
                    append(IrTextureNumericClass::Uint, format, IrBufferFormat::Invalid, false, false);
                }
            }
            if (!storage) {
                if (image.depthBitsCompatible) {
                    append(IrTextureNumericClass::Float, IrBufferFormat::Invalid, IrBufferFormat::Invalid, true, false);
                    append(IrTextureNumericClass::Float, IrBufferFormat::Invalid, IrBufferFormat::Invalid, true, true);
                }
                if (image.fmaskCompatible && !table) append(IrTextureNumericClass::Float, IrBufferFormat::Invalid, IrBufferFormat::Fmask8_S2_F1, false, false);
            }
        }
    }
    if (storage && image.depthBitsCompatible && !image.packed && !image.atomic64 && ((image.written && !image.read) || image.atomic)) {
        constexpr std::array formats{IrBufferFormat::Format32SInt, IrBufferFormat::Format32_32SInt, IrBufferFormat::Format32_32_32_32SInt, IrBufferFormat::Format16SInt, IrBufferFormat::Format8_8SInt, IrBufferFormat::Format16_16SInt, IrBufferFormat::Format8_8_8_8SInt, IrBufferFormat::Format16_16_16_16SInt};
        for (const auto format : formats) {
            if (image.atomic && format != IrBufferFormat::Format32SInt) continue;
            append(IrTextureNumericClass::Uint, format, IrBufferFormat::Invalid, false, false);
        }
    }
    if (image.depthCompare && !table && (image.emulatedCompare & EmulatedCompare::Unsupported) == 0u && (image.dimension == RdnaImageDimension::Dim2D || image.dimension == RdnaImageDimension::Dim2DArray)) {
        auto mode = image;
        mode.numericClass = IrTextureNumericClass::Float;
        mode.depthCompare = false;
        mode.cube = false;
        mode.emulatedCompare |= EmulatedCompare::Enabled;
        mode.conversionFormat = IrBufferFormat::Invalid;
        mode.packedFormat = IrBufferFormat::Invalid;
        mode.shaderSwizzle = ShaderImageIdentitySwizzle;
        modes.push_back(mode);
        if (image.dimension == RdnaImageDimension::Dim2DArray) {
            auto plain = mode;
            plain.dimension = RdnaImageDimension::Dim2D;
            modes.push_back(plain);
            mode.cube = true;
            modes.push_back(mode);
        }
    }
    if (image.srgbDecodeFormats != 0u && image.srgbDecodeCompatible && !image.depthCompare && !storage && !image.packed) {
        const auto count = modes.size();
        for (std::size_t index = 0; index < count; ++index) {
            const auto& base = modes[index];
            if (base.numericClass != IrTextureNumericClass::Float || base.conversionFormat != IrBufferFormat::Invalid || base.packedFormat != IrBufferFormat::Invalid || base.depthBits) continue;
            auto mode = base;
            mode.srgbDecode = true;
            modes.push_back(mode);
        }
    }
    if (image.constantSwizzleCompatible && !storage && !image.depthCompare && !image.packed && !table) {
        for (const auto numeric : {IrTextureNumericClass::Float, IrTextureNumericClass::Uint, IrTextureNumericClass::Sint}) {
            auto mode = image;
            mode.numericClass = numeric;
            mode.conversionFormat = IrBufferFormat::Invalid;
            mode.packedFormat = IrBufferFormat::Invalid;
            mode.depthBits = false;
            mode.depthUnorm16 = false;
            mode.cube = false;
            mode.mipCount = 1u;
            mode.shaderSwizzle = ShaderImageIdentitySwizzle;
            mode.constantSwizzle = true;
            modes.push_back(mode);
        }
    }
    if (modes.empty() && !table) throw std::runtime_error("image instruction has no supported runtime modes");
    if (image.foldTexelOffsetsOnly && std::ranges::any_of(modes, [](const ImageResource& mode) { return UnnormalizedSampleMismatch(mode) == nullptr; })) {
        std::erase_if(modes, [](const ImageResource& mode) { return UnnormalizedSampleMismatch(mode) != nullptr; });
    }
    return modes;
}

const char* ResourceMaterializer::UnnormalizedSampleMismatch(const ImageResource& image) {
    if (image.table != NoTable) return "samples an image selected at run time";
    if (image.constantSwizzle) return nullptr;
    if ((image.dimension != RdnaImageDimension::Dim1D && image.dimension != RdnaImageDimension::Dim2D) || image.cube) return "samples a 1D-array, 2D-array, 3D, cube or multisampled image";
    if (image.depthCompare) return "is used with depth comparison";
    if (image.conversionFormat != IrBufferFormat::Invalid || image.packed) return "samples an image that needs a format conversion or packed access";
    return nullptr;
}

RuntimeImageModeMatch ResourceMaterializer::TryRuntimeImageMode(const ImageResource& image, const DescriptorValue& descriptor, std::span<const ImageResource> modes) {
    RuntimeImageModeMatch result;
    const auto reject = [&](PoisonReason reason, std::string detail) {
        result.reason = reason;
        result.detail = std::move(detail);
        return result;
    };
    if (descriptor.dwordCount != 8u) throw std::runtime_error("runtime image descriptor must contain eight dwords");
    if (modes.empty()) throw std::runtime_error("prepared runtime image modes are missing");
    if (nullImageDescriptor(image, descriptor)) {
        result.mode = 0u;
        return result;
    }
    const auto inspected = inspectImageDescriptor(descriptor, image, image.srgbDecodeFormats);
    if (inspected.reason.has_value()) return reject(*inspected.reason, inspected.detail);
    result.descriptorInspected = true;
    const auto& decoded = inspected.decoded;
    const auto format = rawImageFormat(descriptor);
    const bool emulated = image.depthCompare && format != IrBufferFormat::Format32Float && format != IrBufferFormat::Format16UNorm && !IsDepthBitsTexture(descriptor.dwords[1], descriptor.dwords[3]);
    if (!emulated && (image.emulatedCompare & EmulatedCompare::NativeOffsetUnsupported) != 0u) return reject(PoisonReason::UnsupportedOperation, "native comparison with a nonconstant texel offset requires VK_KHR_maintenance8 and shaderImageGatherExtended");
    if (image.packed && decoded.packedFormat != IrBufferFormat::Invalid) {
        const auto packedFormat = GetFormatInfo(decoded.packedFormat);
        if (packedFormat.packedBitfield) return reject(PoisonReason::InvalidFormat, "runtime packed image accesses a bitfield format");
        const bool storage = image.resourceClass == ImageResourceClass::Storage;
        if (storage && packedFormat.byteSize == 12u) return reject(PoisonReason::InvalidFormat, "runtime packed image uses a format the hardware does not write");
        for (std::uint32_t component = 0u; component < packedFormat.componentCount; ++component) {
            const auto bits = packedFormat.componentBits[component];
            const bool exact = storage ? packedFormat.type == SpirvFormatComponentType::Uint || (bits == 32u && (packedFormat.type == SpirvFormatComponentType::Sint || packedFormat.type == SpirvFormatComponentType::Float)) : packedFormat.type == SpirvFormatComponentType::Uint || packedFormat.type == SpirvFormatComponentType::Sint || (packedFormat.type == SpirvFormatComponentType::Unorm && bits <= 16u) || (packedFormat.type == SpirvFormatComponentType::Float && bits == 32u);
            if (!exact) return reject(PoisonReason::InvalidFormat, storage ? "runtime packed image bits are not reproducible through the view" : "runtime packed image bits are not recoverable from the view");
        }
    }
    if (decoded.mipCount > (image.mipMode == ImageMipMode::DynamicStorage ? RuntimeAbi::StorageMipSlots : 1u)) return reject(PoisonReason::InvalidFormat, "runtime storage image mip capacity exceeded");
    if ((descriptorImageSwizzle(descriptor) & 06666u) == 0u && !decoded.fmask && !decoded.depthBits && decoded.conversionFormat == IrBufferFormat::Invalid && !decoded.srgbDecode) {
        for (std::uint32_t index = 0u; index < modes.size(); ++index) {
            if (modes[index].constantSwizzle && modes[index].numericClass == decoded.numericClass) {
                result.mode = index;
                return result;
            }
        }
    }
    if (image.singleMipPlaneBiasCompatible && image.dimension == RdnaImageDimension::Dim2D && decoded.dimension == RdnaImageDimension::Dim3D) {
        if ((descriptor.dwords[4] & 0x1fffu) != 0u || (descriptor.dwords[3] & 0xff000u) != 0u || (descriptor.dwords[5] & 0xf0u) != 0u) throw std::runtime_error("2D biased sampling of a volume requires one slice and mip level zero only");
    }
    for (std::uint32_t index = 0u; index < modes.size(); ++index) {
        const auto& mode = modes[index];
        if (mode.constantSwizzle) continue;
        if (((mode.emulatedCompare & EmulatedCompare::Enabled) != 0u) != emulated) continue;
        if (decoded.fmask) {
            if (mode.packedFormat == IrBufferFormat::Fmask8_S2_F1) {
                result.mode = index;
                return result;
            }
            continue;
        }
        if (mode.numericClass == decoded.numericClass && mode.dimension == decoded.dimension && mode.conversionFormat == decoded.conversionFormat && mode.packedFormat == decoded.packedFormat && mode.cube == decoded.cube && mode.depthBits == decoded.depthBits && mode.depthUnorm16 == decoded.depthUnorm16 && mode.srgbDecode == decoded.srgbDecode) {
            result.mode = index;
            return result;
        }
    }
    if (image.dimension == RdnaImageDimension::Dim1D && decoded.dimension != RdnaImageDimension::Dim1D) return reject(PoisonReason::Dimension, "image address has too few coordinate components");
    if (image.foldTexelOffsetsOnly) return reject(PoisonReason::UnsupportedOperation, "image sample with a texel offset that is not a constant takes only images an unnormalized S# can sample without VK_KHR_maintenance8 and shaderImageGatherExtended");
    const auto describe = [](IrTextureNumericClass numeric, RdnaImageDimension dimension, IrBufferFormat conversion, IrBufferFormat packed, bool cube, bool depthBits, bool depthUnorm16, bool srgb) {
        return "{class " + std::to_string(static_cast<int>(numeric)) + ", dim " + std::to_string(static_cast<int>(dimension)) + ", conversion " + std::to_string(static_cast<int>(conversion)) + ", packed " + std::to_string(static_cast<int>(packed)) + ", cube " + std::to_string(cube) + ", depthBits " + std::to_string(depthBits) + ", unorm16 " + std::to_string(depthUnorm16) + ", srgb " + std::to_string(srgb) + "}";
    };
    std::string message = "image descriptor is incompatible with the static runtime image interface: descriptor format " + std::to_string(static_cast<int>(format)) + " decodes to " + describe(decoded.numericClass, decoded.dimension, decoded.conversionFormat, decoded.packedFormat, decoded.cube, decoded.depthBits, decoded.depthUnorm16, decoded.srgbDecode) + (emulated ? " (emulated compare)" : "") + "; modes:";
    for (const auto& mode : modes) message += " " + describe(mode.numericClass, mode.dimension, mode.conversionFormat, mode.packedFormat, mode.cube, mode.depthBits, mode.depthUnorm16, mode.srgbDecode);
    char words[96];
    std::snprintf(words, sizeof(words), "; T# %08x %08x %08x %08x %08x %08x %08x %08x", descriptor.dwords[0], descriptor.dwords[1], descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4], descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]);
    const auto reason = emulated ? PoisonReason::ColorCompare
        : decoded.fmask ? PoisonReason::Fmask
        : decoded.depthBits ? PoisonReason::DepthBits16
        : decoded.srgbDecode ? PoisonReason::SrgbUnsupported
        : decoded.conversionFormat != IrBufferFormat::Invalid ? PoisonReason::Conversion
        : decoded.dimension == RdnaImageDimension::Dim2DMsaa || decoded.dimension == RdnaImageDimension::Dim2DMsaaArray ? PoisonReason::Multisampled
        : std::none_of(modes.begin(), modes.end(), [&](const ImageResource& mode) { return mode.dimension == decoded.dimension && mode.cube == decoded.cube; }) ? PoisonReason::Dimension
        : PoisonReason::NumericClass;
    return reject(reason, message + words);
}

ViewLevelsPastChain ResourceMaterializer::LevelsPastChain(const DescriptorValue& descriptor) {
    if (descriptor.dwordCount != 8u || nullImageDescriptor(descriptor)) return {};
    const auto type = rawImageType(descriptor);
    if (type < ImageType::Color1D || type == ImageType::Color2DMsaa || type == ImageType::Color2DMsaaArray) return {};
    const auto& words = descriptor.dwords;
    const auto width = (((words[1] >> 30u) & 0x3u) | ((words[2] & 0xfffu) << 2u)) + 1u;
    const auto height = ((words[2] >> 14u) & 0x3fffu) + 1u;
    const auto depth = type == ImageType::Color3D ? (words[4] & 0x1fffu) + 1u : 1u;
    const auto base = (words[3] >> 12u) & 0xfu;
    const auto maxMip = (words[5] >> 4u) & 0xfu;
    const auto last = base <= maxMip ? std::min((words[3] >> 16u) & 0xfu, maxMip) : (words[3] >> 16u) & 0xfu;
    const auto chainLast = static_cast<std::uint32_t>(std::bit_width(std::max({width, height, depth}))) - 1u;
    if (last < base || last <= chainLast) return {};
    return {base <= chainLast ? last - chainLast : last - base, base > chainLast};
}

std::uint32_t ResourceMaterializer::RuntimeImageMode(const ImageResource& image, const DescriptorValue& descriptor, std::span<const ImageResource> modes) {
    const auto match = TryRuntimeImageMode(image, descriptor, modes);
    if (!match.mode.has_value()) throw std::runtime_error(match.detail);
    return *match.mode;
}

std::uint32_t ResourceMaterializer::DirectImageMode(const ShaderInfo& info, IrShaderStage stage, std::uint32_t index, const DescriptorValue& descriptor, bool allowRefusal) {
    if (index >= info.images.size()) throw std::runtime_error("ShaderRecompiler: direct image resource is missing");
    if (index >= info.runtimeImageModes.size()) throw std::runtime_error("ShaderRecompiler: direct image runtime modes are missing");
    const auto& image = info.images.at(index);
    const auto match = TryRuntimeImageMode(image, descriptor, info.runtimeImageModes.at(index));
    if (match.mode.has_value()) return *match.mode;
    if (allowRefusal && !DirectImageLoopGuardEnabled() && ShaderRecompiler::CanDeferDirectImage(image, stage)
        && std::none_of(info.sampledPairs.begin(), info.sampledPairs.end(), [&](const auto& pair) { return pair.image == index; })
        && match.descriptorInspected && match.reason == PoisonReason::Dimension && directRefusalDescriptor(descriptor)) {
        return PipelineSpecialization::DeferredDirectImageMode;
    }
    throw std::runtime_error(match.detail);
}

bool ResourceMaterializer::CanGuardUnnormalizedWrap(const IrProgram& program, std::uint32_t index) {
    const auto& resources = program.Resources();
    if (!UnnormalizedWrapCandidate(resources.info.samplers.at(index), resources.stage)) return false;
    bool found = false;
    for (const auto& block : program.Blocks()) {
        for (const auto* instruction : block->Instructions()) {
            if (!ImageOpcodeInfoOf(instruction->Opcode()).needsSampler) continue;
            const auto& memory = resources.memoryInfo.at(instruction->Flags<MemoryFlags>().index);
            if (memory.sampler != index) continue;
            const auto flags = memory.imageSampleFlags;
            if (instruction->Opcode() != IrOpcode::ImageSampleRaw || (flags & RdnaImageSampleFlagLevelZero) == 0u
                || (flags & ~(RdnaImageSampleFlagLevelZero | RdnaImageSampleFlagOffset)) != 0u
                || resources.info.images.at(memory.resource).table != NoTable) return false;
            found = true;
        }
    }
    return found;
}

bool ResourceMaterializer::CanDeferDirectImage(const IrProgram& program, std::uint32_t index) {
    const auto& resources = program.Resources();
    if (!ShaderRecompiler::CanDeferDirectImage(resources.info.images.at(index), resources.stage)
        || std::any_of(resources.info.sampledPairs.begin(), resources.info.sampledPairs.end(), [&](const auto& pair) { return pair.image == index; })) return false;
    bool found = false;
    for (const auto& block : program.Blocks()) {
        for (const auto* instruction : block->Instructions()) {
            if (ImageOpcodeInfoOf(instruction->Opcode()).resourceClass == ImageResourceClass::None) continue;
            const auto& memory = resources.memoryInfo.at(instruction->Flags<MemoryFlags>().index);
            if (memory.resource != index) continue;
            if (instruction->Opcode() != IrOpcode::ImageWrite || !directRefusalMemory(memory)) return false;
            found = true;
        }
    }
    return found;
}

void ResourceMaterializer::PrepareDirectImageRefusals(IrProgram& program, bool faultAbi) {
    if (!faultAbi || DirectImageLoopGuardEnabled()) return;
    for (std::uint32_t index = 0u; index < program.Info().images.size(); ++index) {
        if (CanDeferDirectImage(program, index)) {
            program.Info().usesFaultBuffer = true;
            return;
        }
    }
}

void ResourceMaterializer::ValidateDirectImageRefusals(const ShaderInfo& info, IrShaderStage stage, const ResourceSnapshot& snapshot, bool allowRefusal) {
    if (allowRefusal && !DirectImageLoopGuardEnabled()) return;
    for (std::uint32_t index = 0u; index < info.images.size(); ++index) {
        if (ShaderRecompiler::CanDeferDirectImage(info.images[index], stage)) {
            static_cast<void>(DirectImageMode(info, stage, index, snapshot.images.at(index), false));
        }
    }
}

std::uint32_t ResourceMaterializer::EmulatedCompareState(const ShaderInfo& info, const ResourceSnapshot& snapshot, std::uint32_t index) {
    return emulatedCompareState(info, snapshot, index);
}

void ResourceMaterializer::ApplyStaticInterface(IrProgram& program, bool nativeSampleOffsets) const {
    auto& resources = program.Resources();
    if (!resources.resourceTrackingComplete || !resources.srtPlanComplete) throw std::runtime_error("static resource interface requires a completed resource plan");
    auto images = resources.info.images;
    if (!nativeSampleOffsets) {
        for (const auto& block : program.Blocks()) {
            for (const auto* inst : block->Instructions()) {
                if (inst->Opcode() != IrOpcode::ImageSampleRaw) continue;
                const auto& memory = resources.memoryInfo.at(inst->Flags<MemoryFlags>().index);
                if ((memory.imageSampleFlags & RdnaImageSampleFlagOffset) == 0u) continue;
                const auto* address = inst->Argument(2)->Resolve();
                const auto component = GetRdnaImageAddressComponentLayout(memory.imageSampleFlags, 0u);
                const auto argument = component.bitOffset / 32u;
                if (component.bitWidth == 32u && argument < address->ArgumentCount() && address->Argument(argument)->Resolve()->HasImmediate()) continue;
                if ((memory.imageSampleFlags & RdnaImageSampleFlagCompare) != 0u) {
                    images.at(memory.resource).emulatedCompare |= EmulatedCompare::NativeOffsetUnsupported;
                } else if (resources.info.samplers.at(memory.sampler).table == NoTable) {
                    images.at(memory.resource).foldTexelOffsetsOnly = true;
                    resources.info.samplers.at(memory.sampler).foldTexelOffsetsOnly = true;
                }
            }
        }
    }
    bool tables = false;
    for (auto& image : images) {
        image.srgbDecodeFormats = resources.srgbDecodeFormats;
        image.numericClass = image.atomic ? IrTextureNumericClass::Uint : IrTextureNumericClass::Float;
        image.mipCount = image.mipMode == ImageMipMode::DynamicStorage ? RuntimeAbi::StorageMipSlots : 1u;
        if (image.table == NoTable) continue;
        tables = true;
        const auto modes = RuntimeImageModes(image);
        if (modes.empty() || modes.size() > ImageTableAbi::MaxModes) image.tableOperation = TableOperation::Unsupported;
    }
    for (const auto& pair : resources.info.sampledPairs) {
        if (pair.image >= images.size() || pair.sampler >= resources.info.samplers.size()) throw std::runtime_error("static sampled pair is out of range");
        auto& sampler = resources.info.samplers[pair.sampler];
        sampler.depthCompare = sampler.depthCompare || images[pair.image].depthCompare;
        tables = tables || sampler.table != NoTable;
    }
    resources.info.images = std::move(images);
    resources.guardedSrtSlots = Detail::ComputeGuardedFlatSlots(resources);
    resources.srtGuardOffset = static_cast<std::uint32_t>(resources.srtReads.size());
    if (tables || !resources.guardedSrtSlots.empty()) resources.info.usesFaultBuffer = true;
    PrepareImageModes(resources.info);
}

void ResourceMaterializer::PrepareImageModes(ShaderInfo& info) {
    info.runtimeImageModes.clear();
    info.runtimeImageModes.reserve(info.images.size());
    for (const auto& image : info.images) info.runtimeImageModes.push_back(RuntimeImageModes(image));
}

namespace {

void ownPlanValues(IrResourcePlan& plan) {
    std::vector<IrValue**> roots;
    for (auto& source : plan.descriptorSources) {
        for (auto& dword : source.dwords) roots.push_back(&dword);
    }
    for (auto& read : plan.srtReads) roots.push_back(&read.value);
    for (auto& block : plan.controlFlow) roots.push_back(&block.condition);
    for (auto& value : plan.uniformFill.values) roots.push_back(&value);

    std::unordered_map<const IrValue*, IrValue*> clones;
    std::vector<const IrValue*> order;
    std::vector<const IrValue*> pending;
    for (const auto* root : roots) {
        if (*root != nullptr) pending.push_back(*root);
    }
    while (!pending.empty()) {
        const auto* value = pending.back();
        pending.pop_back();
        if (!clones.emplace(value, nullptr).second) continue;
        order.push_back(value);
        for (const auto* argument : value->Arguments()) {
            if (argument != nullptr) pending.push_back(argument);
        }
    }
    plan.valueStorage.Reserve(order.size());
    for (const auto* value : order) {
        auto* clone = plan.valueStorage.Create(value->Opcode(), value->Type(), static_cast<std::uint32_t>(plan.valueStorage.Size()));
        clone->SetFlags(value->Flags<std::uint64_t>());
        if (value->HasImmediate()) clone->SetImmediateU64(value->ImmediateU64());
        clone->SetRegister(value->Register());
        clones[value] = clone;
    }
    std::unordered_map<const IrBlock*, IrBlock*> blocks;
    const auto blockFor = [&](const IrBlock* block) {
        auto& clone = blocks[block];
        if (clone == nullptr) {
            plan.blockStorage.push_back(std::make_unique<IrBlock>(block->Id()));
            clone = plan.blockStorage.back().get();
        }
        return clone;
    };
    for (const auto* value : order) {
        std::vector<IrValue*> arguments;
        std::vector<IrBlock*> phiBlocks;
        arguments.reserve(value->ArgumentCount());
        if (value->IsPhi()) phiBlocks.reserve(value->ArgumentCount());
        for (std::size_t index = 0; index < value->ArgumentCount(); index++) {
            const auto* argument = value->Argument(index);
            arguments.push_back(argument == nullptr ? nullptr : clones.at(argument));
            if (value->IsPhi()) phiBlocks.push_back(blockFor(value->PhiBlock(index)));
        }
        clones.at(value)->SetDetachedOperands(std::move(arguments), std::move(phiBlocks));
    }
    for (auto* root : roots) {
        if (*root != nullptr) *root = clones.at(*root);
    }
    plan.denseValueIds = true;
}

void buildCompactPlan(IrResourcePlan& plan) {
    if (!plan.denseValueIds || plan.valueStorage.Size() >= CompactResourcePlan::NoValue) return;
    CompactResourcePlan compact;
    const auto idOf = [&](const IrValue* value, std::uint32_t& id) {
        if (value == nullptr) return false;
        const auto* resolved = value->Resolve();
        id = resolved->Id();
        return id < plan.valueStorage.Size() && &plan.valueStorage[id] == resolved;
    };
    try {
        compact.values.reserve(plan.valueStorage.Size());
        for (const auto& stored : plan.valueStorage) {
            const auto* value = &stored;
            if (value->ArgumentCount() > 0xffu) return;
            CompactPlanValue entry;
            entry.immediate = value->ImmediateU64();
            entry.flags = value->Flags<std::uint64_t>();
            entry.firstArgument = static_cast<std::uint32_t>(compact.arguments.size());
            entry.registerIndex = value->Register().index;
            entry.type = value->Type();
            entry.opcode = value->Opcode();
            entry.argumentCount = static_cast<std::uint8_t>(value->ArgumentCount());
            entry.hasImmediate = value->HasImmediate();
            for (std::size_t index = 0; index < value->ArgumentCount(); index++) {
                std::uint32_t id = 0;
                if (!idOf(value->Argument(index), id)) return;
                compact.arguments.push_back(id);
            }
            compact.values.push_back(entry);
        }
        compact.sourceDwords.assign(plan.descriptorSources.size() * 8u, CompactResourcePlan::NoValue);
        for (std::size_t source = 0; source < plan.descriptorSources.size(); source++) {
            const auto& descriptor = plan.descriptorSources[source];
            if (descriptor.dwordCount > 8u) return;
            for (std::uint32_t dword = 0; dword < descriptor.dwordCount; dword++) {
                if (!idOf(descriptor.dwords[dword], compact.sourceDwords[source * 8u + dword])) return;
            }
        }
        compact.srtReads.resize(plan.srtReads.size());
        for (std::size_t slot = 0; slot < plan.srtReads.size(); slot++) {
            if (!idOf(plan.srtReads[slot].value, compact.srtReads[slot])) return;
        }
        compact.conditions.assign(plan.controlFlow.size(), CompactResourcePlan::NoValue);
        for (std::size_t block = 0; block < plan.controlFlow.size(); block++) {
            if (plan.controlFlow[block].condition != nullptr && !idOf(plan.controlFlow[block].condition, compact.conditions[block])) return;
        }
    } catch (const std::exception&) {
        return;
    }
    plan.compact = std::move(compact);
}

}

IrResourcePlan ResourceMaterializer::ExtractPlan(const IrProgram& program) const {
    const IrResourcePlan& source = program.Resources();
    if (!source.resourceTrackingComplete || !source.srtPlanComplete) {
        throw std::runtime_error("ResourceMaterializer::ExtractPlan requires a completed resource and SRT plan");
    }
    IrResourcePlan plan;
    plan.stage = source.stage;
    plan.shaderHash = source.shaderHash;
    plan.userDataBase = source.userDataBase;
    plan.userDataCount = source.userDataCount;
    plan.srgbDecodeFormats = source.srgbDecodeFormats;
    plan.memoryInfo = source.memoryInfo;
    plan.descriptorSources = source.descriptorSources;
    plan.controlFlow = source.controlFlow;
    plan.srtReads = source.srtReads;
    plan.cleanFlatSlots = source.cleanFlatSlots;
    plan.requiresSpecializationMemory = source.requiresSpecializationMemory;
    plan.srtPlanComplete = source.srtPlanComplete;
    plan.resourceTrackingComplete = source.resourceTrackingComplete;
    plan.info = source.info;
    PrepareImageModes(plan.info);
    plan.uniformFill = source.uniformFill;
    for (auto& block : plan.controlFlow) {
        if (block.condition != nullptr && !Detail::Evaluator::MayEvaluate(source, block.condition)) block.condition = nullptr;
    }
    ownPlanValues(plan);
    const auto addSource = [&plan](std::uint32_t index) {
        if (index >= plan.descriptorSources.size()) {
            throw std::runtime_error("ResourceMaterializer::ExtractPlan resource references an unknown descriptor source");
        }
        plan.materializationSources.push_back(index);
    };
    for (const auto& buffer : plan.info.buffers) addSource(buffer.source);
    for (const auto& image : plan.info.images) {
        if (image.source >= plan.descriptorSources.size()) {
            throw std::runtime_error("ResourceMaterializer::ExtractPlan image references an unknown descriptor source");
        }
        if (image.table != NoTable) {
            plan.requiresSpecializationMemory = true;
        } else {
            addSource(image.source);
        }
    }
    for (const auto& sampler : plan.info.samplers) {
        if (sampler.table != NoTable) {
            plan.requiresSpecializationMemory = true;
        } else {
            addSource(sampler.source);
        }
    }
    plan.pureFlatSlots = Detail::ComputePureFlatSlots(plan);
    plan.guardedSrtSlots = Detail::ComputeGuardedFlatSlots(plan);
    buildCompactPlan(plan);
    return plan;
}

void ResourceMaterializer::CompactPlan(IrResourcePlan& plan) {
    plan.compact = {};
    buildCompactPlan(plan);
}

void ResourceMaterializer::Materialize(const IrResourcePlan& program, const SrtRuntime& runtime, ResourceSnapshot& snapshot, bool allowDirectImageRefusal) const {
    const IrResourcePlan& plan = program;
    if (!plan.resourceTrackingComplete) {
        throw std::runtime_error("ResourceMaterializer::Materialize requires a completed resource plan");
    }
    if (plan.requiresSpecializationMemory && runtime.readMemory == nullptr) {
        throw std::runtime_error("ResourceMaterializer::Materialize requires runtime memory access for image tables");
    }
    const auto materializeStarted = MaterializeProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    SrtWalker walker;
    ResourceSnapshot nextSnapshot;
    std::vector<std::uint8_t> activeSources;
    try {
        materializeSnapshot(plan, runtime, walker, nextSnapshot, activeSources);
        TableSnapshotter(plan, runtime, walker, activeSources, nextSnapshot.tables).Run();
    } catch (...) {
        reportTables();
        throw;
    }
    const auto started = MaterializeProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    materializeForcedDegamma(plan, nextSnapshot);
    materializeTables(plan, nextSnapshot, allowDirectImageRefusal);
    materializeSrtGuards(plan, nextSnapshot);
    if (MaterializeProfiled()) specializationNanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
    snapshot = std::move(nextSnapshot);
    reportTables();
    if (MaterializeProfiled()) materializeNanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - materializeStarted).count()), std::memory_order_relaxed);
}

std::uint64_t ResourceMaterializer::SpecializationNanoseconds() {
    return specializationNanoseconds.load(std::memory_order_relaxed);
}

std::uint64_t ResourceMaterializer::EvaluateNanoseconds() {
    return evaluateNanoseconds.load(std::memory_order_relaxed);
}

std::uint64_t ResourceMaterializer::MaterializeNanoseconds() {
    return materializeNanoseconds.load(std::memory_order_relaxed);
}

ImageTableColumnMemoCounts ResourceMaterializer::ColumnMemoCounts() {
    const auto& totals = columnMemoTotals();
    return {totals.hits.load(std::memory_order_relaxed), totals.layoutsReused.load(std::memory_order_relaxed), totals.misses.load(std::memory_order_relaxed)};
}

}
