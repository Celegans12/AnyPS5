#include "ShaderDiskCache.hpp"
#include "EnvironmentGuards.hpp"
#include "BdaAbi.hpp"
#include "CacheKey.hpp"
#include "ShaderCachePack.hpp"
#include "VertexInputSpecialization.hpp"
#include "SpirvBackend/SpirvSpecialization.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Optimization/DescriptorBindingBuilder.hpp"
#include "ShaderCacheDirectory.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "Optimization/RequestMemoryView.hpp"
#include "Optimization/SrtWalker.hpp"
#include "IntermediateRepresentation/IrBlock.hpp"
#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <latch>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

using namespace ShaderRecompiler;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void setEnvironment(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
    ShaderRecompiler::RefreshEnvironmentGuards();
}

void removeCacheDirectory(const std::filesystem::path& directory) {
    std::error_code error;
#ifdef _WIN32
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        if (!entry.is_directory()) files.push_back(entry.path());
    }
    constexpr auto dispositionInfoEx = static_cast<FILE_INFO_BY_HANDLE_CLASS>(21);
    constexpr DWORD deleteFile = 0x00000001, posixSemantics = 0x00000002;
    for (const auto& file : files) {
        const auto handle = CreateFileW(file.c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) error.assign(static_cast<int>(GetLastError()), std::system_category());
        require(!error, "cannot open " + file.string() + " for deletion: " + error.message());
        struct Close { HANDLE file; ~Close() { CloseHandle(file); } } close{handle};
        DWORD flags = deleteFile | posixSemantics;
        if (!SetFileInformationByHandle(handle, dispositionInfoEx, &flags, sizeof(flags))) error.assign(static_cast<int>(GetLastError()), std::system_category());
        require(!error, "cannot delete " + file.string() + ": " + error.message());
    }
#endif
    std::filesystem::remove_all(directory, error);
    require(!error, "cannot remove the shader cache test directory: " + error.message());
}

bool sameBinding(const DescriptorBinding& left, const DescriptorBinding& right) {
    return left.kind == right.kind && left.role == right.role && left.descriptorSet == right.descriptorSet && left.binding == right.binding && left.count == right.count && left.guestDescriptor == right.guestDescriptor && left.readOnly == right.readOnly && left.imageShape == right.imageShape && left.samplerDepthCompare == right.samplerDepthCompare && left.imageWritten == right.imageWritten && left.imageDepthCompare == right.imageDepthCompare && left.imageAtomic == right.imageAtomic && left.imageAtomic64 == right.imageAtomic64 && left.imageShapes == right.imageShapes && left.bufferAtomic == right.bufferAtomic && left.bufferWritten == right.bufferWritten && left.bufferRead == right.bufferRead && left.bufferExtent == right.bufferExtent && left.samplerUnnormalized == right.samplerUnnormalized && left.samplerIntegerBorder == right.samplerIntegerBorder && left.samplerCoordinatesIgnored == right.samplerCoordinatesIgnored && left.imageUnnormalized == right.imageUnnormalized && left.imageUnnormalizedBaseLevel == right.imageUnnormalizedBaseLevel && left.imageSamplers == right.imageSamplers;
}

bool sameBindings(const std::vector<DescriptorBinding>& left, const std::vector<DescriptorBinding>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (!sameBinding(left[i], right[i])) return false;
    }
    return true;
}

void requireSameArtifact(const CompiledShaderArtifact& left, const CompiledShaderArtifact& right, const char* what) {
    const std::string prefix = std::string(what) + ": ";
    require(left.spirv == right.spirv, prefix + "SPIR-V differs");
    require(left.bdaAbiVersion == right.bdaAbiVersion, prefix + "BDA ABI version differs");
    require(left.runtimeAbiVersion == right.runtimeAbiVersion, prefix + "runtime ABI version differs");
    require(left.memoryOffsetDword == right.memoryOffsetDword, prefix + "memory offset differs");
    require(left.shaderDataDwords == right.shaderDataDwords && left.imageMetadataDword == right.imageMetadataDword && left.runtimeImageCount == right.runtimeImageCount && left.runtimeImageResources == right.runtimeImageResources, prefix + "compact data layout differs");
    require(left.hostSubgroupSize == right.hostSubgroupSize && left.fullSubgroups == right.fullSubgroups, prefix + "host subgroup size differs");
    require(left.vertexInputs == right.vertexInputs, prefix + "vertex inputs differ");
    require(left.vertexInputPatches == right.vertexInputPatches, prefix + "vertex type patches differ");
    require(left.registrationModule == right.registrationModule, prefix + "registration module flags differ");
    require(left.vertexOffsetSgpr == right.vertexOffsetSgpr && left.instanceOffsetSgpr == right.instanceOffsetSgpr, prefix + "offset SGPRs differ");
    require(left.vertexOffsetShared == right.vertexOffsetShared && left.instanceOffsetShared == right.instanceOffsetShared && left.vertexOffsetConflict == right.vertexOffsetConflict && left.instanceOffsetConflict == right.instanceOffsetConflict, prefix + "offset flags differ");
    require(left.parameterExports == right.parameterExports, prefix + "parameter exports differ");
    require(left.stencilReferences == right.stencilReferences, prefix + "stencil reference set differs");
    require(left.stencilReplaySafe == right.stencilReplaySafe, prefix + "stencil replay proof differs");
    require(left.stencilReferenceDword == right.stencilReferenceDword, prefix + "stencil replay push offset differs");
    require(left.fragmentParameters.size() == right.fragmentParameters.size(), prefix + "fragment parameter count differs");
    for (std::size_t i = 0; i < left.fragmentParameters.size(); ++i) {
        const auto& a = left.fragmentParameters[i];
        const auto& b = right.fragmentParameters[i];
        require(a.location == b.location && a.sourceLocation == b.sourceLocation && a.flat == b.flat && a.perVertex == b.perVertex && a.custom == b.custom, prefix + "fragment parameter differs");
    }
}

void requireSameResult(const RecompileResult& left, const RecompileResult& right, const char* what) {
    requireSameArtifact(left, right, what);
    const std::string prefix = std::string(what) + ": ";
    require(sameBindings(left.bindings, right.bindings), prefix + "bindings differ");
    require(left.pushConstants == right.pushConstants, prefix + "push constants differ");
    require(left.readsDepthRemapPush == right.readsDepthRemapPush, prefix + "depth remap push reads differ");
    require(left.specialization == right.specialization, prefix + "specialization constants differ");
    require(left.workgroupMemoryDwords == right.workgroupMemoryDwords, prefix + "workgroup memory stride differs");
    require(left.poisonedSrtReads == right.poisonedSrtReads, prefix + "poisoned SRT read counts differ");
    require(left.hasDeferredDirectImage == right.hasDeferredDirectImage, prefix + "deferred direct image flag differs");
    require(left.hasGuardedUnnormalizedWrap == right.hasGuardedUnnormalizedWrap, prefix + "guarded unnormalized wrap flag differs");
    require(left.shaderDataUserWords == right.shaderDataUserWords, prefix + "shader-data user words differ");
    require(left.imageTableRanges == right.imageTableRanges && left.imageTableShader == right.imageTableShader && left.imageTableFaults == right.imageTableFaults && left.imageTablePoison.size() == right.imageTablePoison.size(), prefix + "image table invocation data differs");
    require(left.imageTableReadRanges == right.imageTableReadRanges, prefix + "image table read ranges differ");
    for (std::size_t i = 0; i < left.imageTablePoison.size(); ++i) {
        const auto& a = left.imageTablePoison[i];
        const auto& b = right.imageTablePoison[i];
        require(a.words == b.words && a.dwordCount == b.dwordCount && a.resource == b.resource && a.reason == b.reason, prefix + "image table poison differs");
    }
    require(left.vertexAttributes.size() == right.vertexAttributes.size(), prefix + "vertex attribute count differs");
    for (std::size_t i = 0; i < left.vertexAttributes.size(); ++i) {
        const auto& a = left.vertexAttributes[i];
        const auto& b = right.vertexAttributes[i];
        require(a.location == b.location && a.components == b.components && a.resource.fields == b.resource.fields && a.fetchIndex == b.fetchIndex && a.formatComponents == b.formatComponents, prefix + "vertex attribute differs");
    }
    require(left.barycentricEmulation.active == right.barycentricEmulation.active && left.barycentricEmulation.smooth == right.barycentricEmulation.smooth && left.barycentricEmulation.linear == right.barycentricEmulation.linear, prefix + "barycentric emulation differs");
    require(left.meshBarycentrics == right.meshBarycentrics, prefix + "mesh barycentric layout differs");
}

void requireSameVariant(const CompiledVariant& left, const CompiledVariant& right, const char* what) {
    requireSameArtifact(left.artifact, right.artifact, what);
    const std::string prefix = std::string(what) + ": ";
    const auto& a = left.info;
    const auto& b = right.info;
    require(a.stage == b.stage && a.shaderHash == b.shaderHash && a.waveSize == b.waveSize && a.userDataBase == b.userDataBase && a.userDataCount == b.userDataCount && a.scratchDwords == b.scratchDwords && a.paramExportMask == b.paramExportMask, prefix + "compiled info differs");
    require(a.info == b.info, prefix + "shader info differs");
    require(a.bindings == b.bindings, prefix + "info binding layout differs");
    require(left.bindings.layout == right.bindings.layout, prefix + "allocation layout differs");
    require(left.bindings.pushConstantOffsetBytes == right.bindings.pushConstantOffsetBytes && left.bindings.pushConstantSizeBytes == right.bindings.pushConstantSizeBytes, prefix + "push constant range differs");
}

DescriptorBinding sampleBinding(std::uint32_t seed) {
    DescriptorBinding binding{};
    binding.kind = DescriptorKind::StorageImage;
    binding.role = DescriptorRole::GuestImages;
    binding.descriptorSet = 0;
    binding.binding = 29 + seed;
    binding.count = 3;
    binding.guestDescriptor = {0x11111111u * seed, 0xdeadbeefu, 0x80000000u, 7u, 0u, 0xffffffffu, 42u, seed};
    binding.readOnly = seed % 2 == 0;
    binding.imageShape = DescriptorImageShape::Image2DArray;
    binding.samplerDepthCompare = {true, false, true};
    binding.imageDepthCompare = {false, true, false};
    binding.imageWritten = {false, true, true};
    binding.imageAtomic = {false, true, false};
    binding.imageAtomic64 = {true, false, false};
    binding.imageShapes = {DescriptorImageShape::Image2D, DescriptorImageShape::Image3D};
    binding.bufferAtomic = {true};
    binding.bufferWritten = {false, false, true, true, false};
    binding.samplerUnnormalized = {false, true, true};
    binding.samplerIntegerBorder = {true, false, seed % 2 == 1};
    binding.samplerCoordinatesIgnored = {false, seed % 2 == 0, true};
    binding.imageUnnormalized = {true, false, seed % 2 == 0};
    binding.imageUnnormalizedBaseLevel = {false, true, seed % 3 == 0};
    binding.imageSamplers = {0x5u, 0u, 0x80000000u};
    binding.bufferRead = {true, false, false, true, true, false};
    binding.bufferExtent = {32, 0, 16, 4096, 8, 0};
    return binding;
}

RecompileResult sampleResult() {
    RecompileResult result;
    std::vector<std::uint32_t> words(1000);
    for (std::size_t i = 0; i < words.size(); ++i) words[i] = static_cast<std::uint32_t>(i * 2654435761u);
    words[0] = 0x07230203u;
    result.spirv = std::move(words);
    result.bindings = {sampleBinding(1), sampleBinding(2)};
    result.bindings[1].kind = DescriptorKind::StorageBuffer;
    result.bindings[1].role = DescriptorRole::GuestBuffers;
    result.bindings[1].imageShape.reset();
    DescriptorBinding stores{};
    stores.kind = DescriptorKind::StorageImage;
    stores.role = DescriptorRole::ImageTable;
    stores.binding = RuntimeAbi::StorageImageTableBinding;
    stores.count = 2u;
    stores.guestDescriptor = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0x123400u, 13u << 20u, 0u, 0x9000024cu, 0u, 0u, 0u, 0u};
    stores.imageShapes = {DescriptorImageShape::Image2D, DescriptorImageShape::Image2D};
    stores.imageWritten = {false, true};
    stores.imageAtomic = {false, false};
    result.bindings.push_back(std::move(stores));
    result.pushConstants = {std::byte{1}, std::byte{0xff}, std::byte{0}, std::byte{0x80}, std::byte{7}};
    result.readsDepthRemapPush = true;
    result.bdaAbiVersion = 3;
    result.memoryOffsetDword = 7;
    result.hostSubgroupSize = 32;
    result.fullSubgroups = true;
    result.workgroupMemoryDwords = 32769u;
    result.vertexInputs = {{1, 4, 2}, {5, 2, 0}};
    result.vertexInputPatches = {{1, 20, {7, 8, 9}}, {5, 40, {10, 11, 12}}};
    result.registrationModule = true;
    result.specialization = {{512, 4}, {516, 0x3f800000u}, {517, 0}};
    result.vertexAttributes = {{1, 4, {{0x1000u, 0x20000u, 0x30u, 0x4u}}, 2}, {5, 2, {{9u, 8u, 7u, 6u}}, 0}};
    result.vertexInputs[0].outputMask = 5u;
    result.vertexAttributes[0].formatComponents = 3u;
    result.vertexOffsetSgpr = 12;
    result.instanceOffsetSgpr = -1;
    result.vertexOffsetShared = true;
    result.instanceOffsetShared = false;
    result.vertexOffsetConflict = false;
    result.instanceOffsetConflict = true;
    result.parameterExports = {0, 3, 7};
    result.fragmentParameters = {{0, 1, true, false, true}, {2, 3, false, true}};
    result.stencilReferences = {0u, 17u, 255u};
    result.stencilReplaySafe = true;
    result.stencilReferenceDword = 23u;
    result.poisonedSrtReads = 3;
    result.shaderDataUserWords = {2u, 0xffffffffu, 9u};
    result.barycentricEmulation = {true, false, true};
    result.meshBarycentrics = MeshBarycentricLayout{{true, true, false}, {FragmentParameter{0, 2, false, true}, FragmentParameter{3, 3, false, false}}};
    result.variantId = 99;
    result.imageTablePoison = {{{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}, 8u, 2u, 6u}, {{9u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}, 4u, 0xffffffffu, 19u}};
    result.imageTableRanges = {{0x1000u, 0x200u}, {0x5000000000ull, 48u}};
    result.imageTableReadRanges = {{{0x1000u, 32u}, {0x1040u, 32u}}, {}, {{0x5000000000ull, 16u}}};
    result.imageTableShader = 0x80003aa700ull;
    result.imageTableFaults = 3u;
    result.hasDeferredDirectImage = true;
    result.hasGuardedUnnormalizedWrap = true;
    return result;
}

CompiledVariant sampleVariant() {
    CompiledVariant variant;
    variant.artifact = sampleResult();
    variant.layout = {0, 0, 0, 128};
    auto& info = variant.info;
    info.stage = IrShaderStage::Compute;
    info.shaderHash = 0x123456789abcdefull;
    info.waveSize = 32;
    info.userDataBase = 4;
    info.userDataCount = 16;
    info.scratchDwords = 8;
    info.paramExportMask = 0x5;
    info.info.scratchDwords = 8;
    info.info.sharedMemoryBytes = 4096;
    BufferResource buffer{};
    buffer.source = 3;
    buffer.firstUsePc = 0x40;
    buffer.maxByteExtent = 256;
    buffer.imageAlias = 1;
    buffer.read = true;
    buffer.atomic = true;
    buffer.scalar = true;
    buffer.descriptorFormatted = true;
    buffer.formattedReadMask = 5u;
    buffer.typedAlignment = 2u;
    info.info.buffers = {buffer, BufferResource{}};
    ImageResource image{};
    image.source = 5;
    image.firstUsePc = 0x80;
    image.resourceClass = ImageResourceClass::Storage;
    image.numericClass = IrTextureNumericClass::Uint;
    image.mipMode = ImageMipMode::DynamicStorage;
    image.mipCount = 4;
    image.shaderSwizzle = 0x321u;
    image.written = true;
    image.srgbDecode = true;
    image.cube = true;
    image.r128 = true;
    image.fmaskCompatible = false;
    image.depthBitsCompatible = false;
    image.constantSwizzle = true;
    image.constantSwizzleCompatible = false;
    image.cubeFaceCompatible = false;
    image.flatVolumeCompatible = false;
    image.singleMipPlaneBiasCompatible = true;
    image.volumeArraySampleCompatible = true;
    image.flatLineCompatible = false;
    image.flatVolumeLineCompatible = false;
    image.foldTexelOffsetsOnly = true;
    image.byElements = 4;
    image.byComponents = 1;
    image.emulatedCompare = 0x1a07u;
    image.table = 0;
    image.tableOperation = TableOperation::Unsupported;
    image.levelUses = ImageLevelUseQuery | ImageLevelUseMipLoad;
    info.info.images = {image};
    ResourceMaterializer::PrepareImageModes(info.info);
    info.info.samplers = {{7, 0x10, 3, true, false, SamplerUseExplicitLod | SamplerUseGather | SamplerUseNonZeroLod, true, 1}};
    info.info.usesFaultBuffer = true;
    info.info.sampledPairs = {{0, 0, 0x10}};
    StageInput input{};
    input.kind = StageInputKind::GlobalInvocationId;
    input.location = 2;
    input.componentCount = 3;
    input.debugName = "gid";
    input.perVertex = true;
    info.info.inputs = {input};
    StageOutput output{};
    output.kind = StageOutputKind::Mrt;
    output.index = 1;
    output.location = 4;
    output.debugName = "mrt1";
    info.info.outputs = {output};
    info.info.vertexFetchComponents[3] = 4;
    info.info.vertexOffsetSgpr = 6;
    info.info.hasBitwiseXor = true;
    info.info.usesDma = true;
    info.info.usesFaultBuffer = true;
    info.info.rasterPrimitive = true;
    info.info.layerExported = true;
    info.bindings.pushDataStartDword = 2;
    info.bindings.memoryOffsetDword = 1;
    info.bindings.memoryOffsetCount = 5;
    info.bindings.stencilReferenceDword = 23u;
    info.bindings.userDataRegisters = {4, 5, 9};
    info.bindings.descriptors = {{DescriptorBindingKind::Buffers, {0, 1}}, {DescriptorBindingKind::FlattenedSrt, {}}, {DescriptorBindingKind::StorageImageTable, {}}};
    variant.bindings.layout = info.bindings;
    variant.bindings.pushConstantOffsetBytes = 16;
    variant.bindings.pushConstantSizeBytes = 112;
    return variant;
}

struct SampleRequest {
    std::vector<std::uint32_t> code{0xe0700000u, 0x80000000u, 0xbf810000u, 0x12345678u};
    std::vector<std::uint32_t> userData{0x10000000u, 0x00100000u, 0x40u, 0x00027facu};
    std::array<std::uint32_t, 2> capabilities{1u, 61u};
    std::array<std::string_view, 1> extensions{"SPV_KHR_storage_buffer_storage_class"};
    std::array<std::byte, 16> header{};
    RecompileRequest request{};
    std::uint32_t hostSubgroupSize = 32;

    SampleRequest() {
        request.shader = {ShaderStage::Compute, 0x20000u, code, 0x1f000u, header};
        request.context.waveSize = 64;
        request.context.userDataBaseRegister = 0;
        request.context.userData = userData;
        request.context.compute = ShaderComputeStageInfo{{64u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
        request.target.vulkanVersion = 0x00403000u;
        request.target.spirvVersion = 0x00010600u;
        request.target.subgroupSize = 32;
        request.target.bdaAbiVersion = 1;
        request.target.supportedCapabilities = capabilities;
        request.target.supportedExtensions = extensions;
        request.target.maxWorkgroupSize = {1024u, 1024u, 64u};
        request.target.maxWorkgroupInvocations = 1024;
        request.target.maxWorkgroupSharedMemoryBytes = 49152;
        request.layout = {0, 0, 0, 128};
    }

    std::vector<std::byte> Key() {
        request.shader.code = code;
        request.context.userData = userData;
        request.target.supportedCapabilities = capabilities;
        std::vector<std::byte> key;
        ShaderDiskCache::BuildKey(request, hostSubgroupSize, key);
        return key;
    }
};

void verifyResultRoundTrip() {
    const auto result = sampleResult();
    std::vector<std::byte> bytes;
    ShaderDiskCache::EncodeResult(result, bytes);
    RecompileResult decoded;
    require(ShaderDiskCache::DecodeResult(bytes, decoded), "an encoded result does not decode");
    requireSameResult(result, decoded, "result round trip");
    require(decoded.variantId == 0 && !decoded.cacheHit, "the per-process fields were stored");
    for (std::size_t size = 0; size < bytes.size(); size += size < 256 ? 1 : 97) {
        RecompileResult partial;
        require(!ShaderDiskCache::DecodeResult(std::span(bytes).first(size), partial), "a result truncated to " + std::to_string(size) + " bytes decodes");
    }
    auto longer = bytes;
    longer.push_back(std::byte{0});
    require(!ShaderDiskCache::DecodeResult(longer, decoded), "a result with a trailing byte decodes");
    bytes.clear();
    ShaderDiskCache::EncodeResult(RecompileResult{}, bytes);
    require(ShaderDiskCache::DecodeResult(bytes, decoded), "an empty result does not decode");
    requireSameResult(RecompileResult{}, decoded, "empty result round trip");
    for (const auto abi : {RuntimeAbi::Version - 1u, RuntimeAbi::Version + 1u}) {
        auto incompatible = result;
        incompatible.runtimeAbiVersion = abi;
        bytes.clear();
        ShaderDiskCache::EncodeResult(incompatible, bytes);
        require(!ShaderDiskCache::DecodeResult(bytes, decoded), "an incompatible runtime ABI " + std::to_string(abi) + " result decodes");
    }
}

void verifyElementFlags() {
    ElementFlags flags;
    std::vector<bool> expected;
    for (std::size_t index = 0; index < 70; ++index) {
        flags.push_back(index % 3 == 0);
        expected.push_back(index % 3 == 0);
        require(flags == expected, "element flags differ from a vector after " + std::to_string(index + 1) + " entries");
    }
    const auto copy = flags;
    require(copy == flags && copy.size() == 70 && copy.at(69) && !copy.at(68), "a copy of spilled element flags differs");
    require(std::count(flags.begin(), flags.end(), true) == 24, "spilled element flags do not iterate");
    auto writable = flags;
    writable[69] = false;
    writable.back().flip();
    writable[0] = writable[1];
    require(writable.size() == 70 && writable.back() && !writable[0] && writable[3], "a write through a spilled flag was lost");
    writable.resize(3);
    require(writable == std::vector<bool>{false, false, false}, "shrinking spilled element flags kept the wrong entries");
    writable.resize(5, true);
    writable[1] = true;
    writable.back().flip();
    require(writable == std::vector<bool>{false, true, false, true, false}, "a write through an inline flag was lost");
    auto result = sampleResult();
    result.bindings[1].bufferWritten = flags;
    std::vector<std::byte> bytes;
    ShaderDiskCache::EncodeResult(result, bytes);
    RecompileResult decoded;
    require(ShaderDiskCache::DecodeResult(bytes, decoded) && decoded.bindings[1].bufferWritten == expected, "spilled element flags do not round trip");
    flags.assign(3, true);
    require(flags == std::vector<bool>{true, true, true}, "assigned element flags kept spilled entries");
    flags.clear();
    require(flags.empty() && flags == ElementFlags{}, "cleared element flags are not empty");
}

void verifyEntryRoundTrip() {
    SampleRequest sample;
    const auto key = sample.Key();
    const auto variant = sampleVariant();
    const auto file = ShaderDiskCache::EncodeEntry(key, variant);
    CompiledVariant decoded;
    require(ShaderDiskCache::DecodeEntry(file, key, decoded) == ShaderDiskCache::LoadStatus::Loaded, "an encoded entry does not decode");
    requireSameVariant(variant, decoded, "entry round trip");
    require(decoded.artifact.variantId == 0, "the artifact stored a per-process id");

    for (std::size_t size = 0; size < file.size(); size += size < 512 ? 1 : 131) {
        CompiledVariant partial;
        require(ShaderDiskCache::DecodeEntry(std::span(file).first(size), key, partial) == ShaderDiskCache::LoadStatus::Rejected, "an entry truncated to " + std::to_string(size) + " bytes is not rejected");
    }
    for (std::size_t offset = 0; offset < file.size(); offset += offset < 512 ? 1 : 61) {
        auto damaged = file;
        damaged[offset] ^= std::byte{0x10};
        CompiledVariant partial;
        const auto status = ShaderDiskCache::DecodeEntry(damaged, key, partial);
        require(status == ShaderDiskCache::LoadStatus::Rejected, "an entry with byte " + std::to_string(offset) + " damaged is not rejected");
    }
    auto longer = file;
    longer.push_back(std::byte{0});
    require(ShaderDiskCache::DecodeEntry(longer, key, decoded) == ShaderDiskCache::LoadStatus::Rejected, "an entry with a trailing byte is not rejected");
    auto otherKey = key;
    otherKey.back() ^= std::byte{1};
    require(ShaderDiskCache::DecodeEntry(file, otherKey, decoded) == ShaderDiskCache::LoadStatus::KeyMismatch, "an entry for another key loads");
    auto previousVersion = file;
    previousVersion[4] = static_cast<std::byte>(ShaderDiskCache::FormatVersion - 1u);
    require(ShaderDiskCache::DecodeEntry(previousVersion, key, decoded) == ShaderDiskCache::LoadStatus::Rejected, "an entry with the previous format version loads");
    auto priorStencilVersion = file;
    priorStencilVersion[4] = std::byte{32};
    require(ShaderDiskCache::DecodeEntry(priorStencilVersion, key, decoded) == ShaderDiskCache::LoadStatus::Rejected, "an artifact with the prior stencil format loads");
    auto upstreamVersion = file;
    upstreamVersion[4] = std::byte{33};
    require(ShaderDiskCache::DecodeEntry(upstreamVersion, key, decoded) == ShaderDiskCache::LoadStatus::Rejected, "a stencil artifact with the upstream resource plan format loads");
    for (const auto abi : {RuntimeAbi::Version - 1u, RuntimeAbi::Version + 1u}) {
        auto incompatible = variant;
        incompatible.artifact.runtimeAbiVersion = abi;
        const auto incompatibleFile = ShaderDiskCache::EncodeEntry(key, incompatible);
        require(ShaderDiskCache::DecodeEntry(incompatibleFile, key, decoded) == ShaderDiskCache::LoadStatus::Rejected,
            "an entry with an incompatible runtime ABI " + std::to_string(abi) + " loads");
    }
}

std::vector<std::byte> packBytes(const std::vector<std::pair<std::vector<std::byte>, std::vector<std::byte>>>& entries) {
    using namespace ShaderRecompiler::ShaderDiskCache;
    std::vector<PackRecord> records;
    std::uint64_t offset = sizeof(PackHeader) + entries.size() * sizeof(PackRecord);
    for (const auto& [key, entry] : entries) {
        records.push_back({PackKeyHash(key), offset, entry.size()});
        offset += entry.size();
    }
    std::sort(records.begin(), records.end(), [](const PackRecord& left, const PackRecord& right) { return left.hash < right.hash; });
    PackHeader header;
    header.sourceVersion = SourceVersion();
    header.fileBytes = offset;
    header.entryCount = entries.size();
    header.indexHash = ShaderRecompiler::HashBytes(std::as_bytes(std::span(records)));
    std::vector<std::byte> bytes(offset);
    std::memcpy(bytes.data(), &header, sizeof(header));
    std::memcpy(bytes.data() + sizeof(header), records.data(), records.size() * sizeof(PackRecord));
    std::size_t at = sizeof(header) + records.size() * sizeof(PackRecord);
    for (const auto& [key, entry] : entries) {
        std::memcpy(bytes.data() + at, entry.data(), entry.size());
        at += entry.size();
    }
    return bytes;
}

void refreshIndexHash(std::vector<std::byte>& pack, std::size_t entries) {
    using namespace ShaderRecompiler::ShaderDiskCache;
    const auto index = std::span(pack).subspan(sizeof(PackHeader), entries * sizeof(PackRecord));
    const auto hash = ShaderRecompiler::HashBytes(index);
    std::memcpy(pack.data() + offsetof(PackHeader, indexHash), &hash, sizeof(hash));
}

void verifyPrebuiltPack(const std::filesystem::path& directory) {
    using namespace ShaderRecompiler::ShaderDiskCache;
    SampleRequest sample;
    const auto key = sample.Key();
    const auto variant = sampleVariant();
    const auto file = EncodeEntry(key, variant);
    auto otherKey = key;
    otherKey.back() ^= std::byte{1};
    const auto otherFile = EncodeEntry(otherKey, variant);
    std::filesystem::create_directories(directory);
    const auto writePack = [&](const char* name, const std::vector<std::byte>& bytes) {
        const auto path = directory / name;
        require(WriteFileAtomically(path, bytes), "cannot write the test pack");
        return path;
    };
    PackLoadTimings timings;
    CompiledVariant decoded;
    {
        CachePack pack(writePack("valid.pack", packBytes({{key, file}, {otherKey, otherFile}})));
        require(pack.Valid(), "a valid pack is rejected: " + pack.Error());
        require(pack.Count() == 2, "the pack does not hold both entries");
        require(pack.Load(key, decoded, timings) == LoadStatus::Loaded, "a pack entry does not load");
        requireSameVariant(variant, decoded, "pack round trip");
        auto missingKey = key;
        missingKey.front() ^= std::byte{1};
        require(pack.Load(missingKey, decoded, timings) == LoadStatus::Absent, "an absent key loads from the pack");
    }
    {
        std::vector<std::pair<std::vector<std::byte>, std::vector<std::byte>>> mismatched;
        mismatched.emplace_back(otherKey, file);
        CachePack pack(writePack("mismatch.pack", packBytes(mismatched)));
        require(pack.Valid(), "the key mismatch pack is rejected: " + pack.Error());
        require(pack.Load(otherKey, decoded, timings) != LoadStatus::Loaded, "an entry stored under another key loads");
    }
    {
        auto bytes = packBytes({{key, file}});
        const auto truncated = static_cast<std::uint64_t>(file.size() / 2);
        std::memcpy(bytes.data() + sizeof(PackHeader) + offsetof(PackRecord, size), &truncated, sizeof(truncated));
        refreshIndexHash(bytes, 1);
        CachePack pack(writePack("truncated.pack", bytes));
        require(pack.Valid(), "the truncated-entry pack is rejected: " + pack.Error());
        require(pack.Load(key, decoded, timings) == LoadStatus::Rejected, "a truncated pack entry loads");
    }
    {
        auto bytes = packBytes({{key, file}});
        bytes[sizeof(PackHeader)] ^= std::byte{0x10};
        CachePack pack(writePack("corrupt-index.pack", bytes));
        require(!pack.Valid(), "a pack with a damaged index is accepted");
    }
    {
        auto bytes = packBytes({{key, file}});
        const auto outside = static_cast<std::uint64_t>(bytes.size()) + 16;
        std::memcpy(bytes.data() + sizeof(PackHeader) + offsetof(PackRecord, offset), &outside, sizeof(outside));
        refreshIndexHash(bytes, 1);
        CachePack pack(writePack("corrupt-record.pack", bytes));
        require(!pack.Valid(), "a pack with an out-of-range record is accepted");
    }
    {
        auto bytes = packBytes({{key, file}});
        const auto wrong = SourceVersion() + 1;
        std::memcpy(bytes.data() + offsetof(PackHeader, sourceVersion), &wrong, sizeof(wrong));
        CachePack pack(writePack("wrong-version.pack", bytes));
        require(!pack.Valid(), "a pack from another compiler source version is accepted");
    }
    {
        auto bytes = packBytes({{key, file}});
        bytes.front() ^= std::byte{0x10};
        CachePack pack(writePack("unknown-format.pack", bytes));
        require(!pack.Valid(), "a pack with an unknown format is accepted");
    }
}

SampleRequest prebuiltSample() {
    SampleRequest sample;
    sample.code[3] ^= 0x5a5a5a5au;
    return sample;
}

int runPrebuiltLoadingProcess() {
    using namespace ShaderRecompiler::ShaderDiskCache;
    SampleRequest sample = prebuiltSample();
    const auto key = sample.Key();
    const auto expected = sampleVariant();
    CompiledVariant loaded;
    require(Load(key, loaded), "the prebuilt pack entry does not load");
    requireSameVariant(expected, loaded, "prebuilt load");
    require(Totals().prebuiltHits == 1, "the prebuilt load was not counted");
    const auto afterPack = Totals();
    require(afterPack.prebuiltBytesRead > 0 && afterPack.prebuiltLoadNanoseconds > 0, "the prebuilt load was not timed");
    require(afterPack.hits == 0 && afterPack.loadFailures == 0 && afterPack.bytesRead == 0 && afterPack.loadNanoseconds == 0, "the prebuilt load reached the writable totals");
    auto looseKey = key;
    looseKey.back() ^= std::byte{1};
    CompiledVariant loose;
    require(Load(looseKey, loose), "the loose prebuilt entry does not load");
    requireSameVariant(expected, loose, "loose prebuilt load");
    require(Totals().prebuiltHits == 2, "the loose prebuilt load was not counted");
    require(Totals().hits == 0 && Totals().loadFailures == 0 && Totals().bytesRead == 0, "the loose prebuilt load reached the writable totals");
    auto corruptKey = looseKey;
    corruptKey.front() ^= std::byte{1};
    CompiledVariant corrupt;
    require(!Load(corruptKey, corrupt), "a corrupt loose prebuilt entry loads");
    require(Totals().prebuiltRejected >= 1, "the corrupt loose prebuilt entry was not counted as rejected");
    require(Totals().loadFailures == 0, "the corrupt loose prebuilt entry reached the writable failures");
    auto otherKey = looseKey;
    otherKey.front() ^= std::byte{1};
    CompiledVariant other;
    require(!Load(otherKey, other), "a key absent from prebuilt storage loads");
    Store(otherKey, std::make_shared<const CompiledVariant>(sampleVariant()));
    Flush();
    require(Load(otherKey, other), "the writable cache fall-through does not load");
    std::cout << "prebuilt process loaded from the pack and fell through to the writable cache\n";
    return 0;
}

void verifyPrebuiltAcrossProcesses(const char* self, const std::filesystem::path& directory) {
    using namespace ShaderRecompiler::ShaderDiskCache;
    SampleRequest sample = prebuiltSample();
    const auto prebuilt = directory / "prebuilt";
    std::filesystem::create_directories(prebuilt);
    const auto bytes = packBytes({{sample.Key(), EncodeEntry(sample.Key(), sampleVariant())}});
    require(WriteFileAtomically(prebuilt / "test.pack", bytes), "cannot write the prebuilt pack");
    auto looseKey = sample.Key();
    looseKey.back() ^= std::byte{1};
    require(WriteFileAtomically(prebuilt / EntryName(looseKey), EncodeEntry(looseKey, sampleVariant())), "cannot write the loose prebuilt entry");
    auto corruptKey = looseKey;
    corruptKey.front() ^= std::byte{1};
    auto corruptEntry = EncodeEntry(corruptKey, sampleVariant());
    corruptEntry.resize(corruptEntry.size() / 2);
    require(WriteFileAtomically(prebuilt / EntryName(corruptKey), corruptEntry), "cannot write the corrupt loose prebuilt entry");
    const std::string command = "\"" + std::string(self) + "\" --load-prebuilt";
    require(std::system(command.c_str()) == 0, "the prebuilt loading process failed");
}

void verifyArtifactStorageIsolation() {
    SampleRequest sample;
    const auto key = sample.Key();
    auto result = sampleResult();
    auto variant = sampleVariant();
    variant.artifact = result;
    const auto before = ShaderDiskCache::EncodeEntry(key, variant);
    result.bindings[0].guestDescriptor[0] ^= 0x10000u;
    result.pushConstants[0] ^= std::byte{0xff};
    result.vertexAttributes[0].resource.fields[0] ^= 0x10000u;
    variant.artifact = result;
    require(ShaderDiskCache::EncodeEntry(key, variant) == before, "invocation data changed the stored artifact");
    result.vertexInputs[0].components = 2;
    variant.artifact = result;
    require(ShaderDiskCache::EncodeEntry(key, variant) != before, "the stored artifact ignores vertex input metadata");
}

void verifyMeshBarycentricRequestSerialization() {
    for (const auto& layout : {std::optional<MeshBarycentricLayout>{}, std::optional<MeshBarycentricLayout>{MeshBarycentricLayout{{true, true, true}, {FragmentParameter{0, 2, false, true}, FragmentParameter{4, 4, false, false}}}}}) {
        SampleRequest sample;
        sample.request.shader.stage = ShaderStage::Mesh;
        sample.request.context.compute.reset();
        sample.request.graphics = GraphicsCompileContext{0, {}, MeshConfiguration{4u, 4u, 12u, 12u, 4u, 64u, 1024u, 0u, 4u}, std::nullopt, {}, layout};
        const auto text = RequestSerializer{}.Serialize(sample.request);
        const auto restored = RequestSerializer{}.Deserialize(text);
        require(restored.request.graphics.has_value() && restored.request.graphics->meshBarycentrics == layout, "a captured mesh request lost its barycentric layout");
    }
}

void verifyKeySensitivity() {
    SampleRequest base;
    const auto key = base.Key();
    require(base.Key() == key, "the key is not deterministic");
    require(ShaderDiskCache::EntryName(key).size() == 36, "unexpected entry name length");

    const auto changes = [&](const std::string& what, const std::function<void(SampleRequest&)>& change) {
        SampleRequest sample;
        change(sample);
        const auto changed = sample.Key();
        require(changed != key, "the key ignores " + what);
        require(ShaderDiskCache::EntryName(changed) != ShaderDiskCache::EntryName(key), "the entry name ignores " + what);
    };
    for (std::size_t word = 0; word < base.code.size(); ++word) {
        for (std::uint32_t bit = 0; bit < 32; ++bit) {
            changes("code word " + std::to_string(word) + " bit " + std::to_string(bit), [&](SampleRequest& sample) { sample.code[word] ^= 1u << bit; });
        }
    }
    changes("a code word appended", [](SampleRequest& sample) { sample.code.push_back(0); });
    changes("the stage", [](SampleRequest& sample) { sample.request.shader.stage = ShaderStage::Fragment; sample.request.context.compute.reset(); });
    changes("the wave size", [](SampleRequest& sample) { sample.request.context.waveSize = 32; });
    changes("the user data base", [](SampleRequest& sample) { sample.request.context.userDataBaseRegister = 2; });
    changes("the user data count", [](SampleRequest& sample) { sample.userData.push_back(0); });
    for (std::size_t axis = 0; axis < 3; ++axis) {
        changes("thread count " + std::to_string(axis), [&](SampleRequest& sample) { sample.request.context.compute->numThreads[axis] += 1; });
        changes("group id enable " + std::to_string(axis), [&](SampleRequest& sample) { sample.request.context.compute->groupIdEnable[axis] = true; });
    }
    changes("the LDS size", [](SampleRequest& sample) { sample.request.context.compute->ldsSizeDwords = 64; });
    changes("the thread group size enable", [](SampleRequest& sample) { sample.request.context.compute->tgSizeEnable = true; });
    changes("the thread id component count", [](SampleRequest& sample) { sample.request.context.compute->threadIdComponentCount = 3; });
    changes("the Vulkan version", [](SampleRequest& sample) { sample.request.target.vulkanVersion = 0x00402000u; });
    changes("the SPIR-V version", [](SampleRequest& sample) { sample.request.target.spirvVersion = 0x00010500u; });
    changes("the target subgroup size", [](SampleRequest& sample) { sample.request.target.subgroupSize = 64; });
    changes("the BDA ABI version", [](SampleRequest& sample) { sample.request.target.bdaAbiVersion = 2; });
    changes("a capability", [](SampleRequest& sample) { sample.capabilities[1] = 62u; });
    changes("an extension", [](SampleRequest& sample) { sample.extensions[0] = "SPV_KHR_storage_buffer_storage_clasS"; });
    changes("barycentrics", [](SampleRequest& sample) { sample.request.target.fragmentShaderBarycentricEnabled = true; });
    changes("non-constant texel offsets", [](SampleRequest& sample) { sample.request.target.nonConstantImageOffsets = true; });
    changes("the sRGB formats decoded in the shader", [](SampleRequest& sample) { sample.request.target.srgbDecodeFormats = 2u; });
    changes("the narrow subgroup clock", [](SampleRequest& sample) { sample.request.target.narrowSubgroupClock = true; });
    changes("f64 denormal preservation", [](SampleRequest& sample) { sample.request.target.float64DenormPreserve = true; });
    changes("f16 denormal preservation", [](SampleRequest& sample) { sample.request.target.float16DenormPreserve = true; });
    changes("spanned waves", [](SampleRequest& sample) { sample.request.target.spannedWaves = true; });
    changes("robust buffer access 2", [](SampleRequest& sample) { sample.request.target.robustBufferAccess2 = true; });
    changes("the workgroup size limit", [](SampleRequest& sample) { sample.request.target.maxWorkgroupSize[2] = 128; });
    changes("the invocation limit", [](SampleRequest& sample) { sample.request.target.maxWorkgroupInvocations = 512; });
    changes("the shared memory limit", [](SampleRequest& sample) { sample.request.target.maxWorkgroupSharedMemoryBytes = 32768; });
    changes("the host subgroup size", [](SampleRequest& sample) { sample.hostSubgroupSize = 64; });
    changes("the descriptor set", [](SampleRequest& sample) { sample.request.layout.descriptorSet = 1; });
    changes("the first binding", [](SampleRequest& sample) { sample.request.layout.firstBinding = 1; });
    changes("the push constant offset", [](SampleRequest& sample) { sample.request.layout.pushConstantOffsetBytes = 16; });
    changes("the push constant size", [](SampleRequest& sample) { sample.request.layout.pushConstantSizeBytes = 64; });
    changes("a float mode", [](SampleRequest& sample) { sample.request.context.floatMode = ShaderFloatMode{}; });
    const ShaderFloatMode astroMode{0xc0u, true, false, false};
    const auto withMode = [&](const ShaderFloatMode& mode) {
        SampleRequest sample;
        sample.request.context.floatMode = mode;
        return std::pair{sample.Key(), RecompileCacheKey::ContextHash(sample.request)};
    };
    const auto astro = withMode(astroMode);
    require(withMode(astroMode) == astro, "the float mode key is not deterministic");
    for (const auto& [what, mode] : std::initializer_list<std::pair<const char*, ShaderFloatMode>>{
             {"FLOAT_MODE", {0x00u, true, false, false}}, {"DX10_CLAMP", {0xc0u, false, false, false}},
             {"IEEE_MODE", {0xc0u, true, true, false}}, {"FP16_OVFL", {0xc0u, true, false, true}}}) {
        const auto changed = withMode(mode);
        require(changed.first != astro.first && changed.second != astro.second, std::string("the key ignores ") + what);
    }

    const auto meshKeys = [&](const std::optional<MeshBarycentricLayout>& layout) {
        SampleRequest sample;
        sample.request.shader.stage = ShaderStage::Mesh;
        sample.request.context.compute.reset();
        sample.request.graphics = GraphicsCompileContext{0, {}, MeshConfiguration{4u, 4u, 12u, 12u, 4u, 64u, 1024u, 0u, 4u}, std::nullopt, {}, layout};
        return std::pair{sample.Key(), RecompileCacheKey::ContextHash(sample.request)};
    };
    const std::vector<std::pair<const char*, std::optional<MeshBarycentricLayout>>> meshLayouts{
        {"no mesh barycentric layout", std::nullopt},
        {"smooth mesh barycentrics", MeshBarycentricLayout{{true, true, false}, {}}},
        {"linear mesh barycentrics", MeshBarycentricLayout{{true, false, true}, {}}},
        {"smooth and linear mesh barycentrics", MeshBarycentricLayout{{true, true, true}, {}}},
        {"a per-vertex mesh parameter", MeshBarycentricLayout{{true, true, false}, {FragmentParameter{0, 0, false, true}}}},
        {"another per-vertex mesh parameter source", MeshBarycentricLayout{{true, true, false}, {FragmentParameter{0, 1, false, true}}}},
        {"a flat mesh parameter", MeshBarycentricLayout{{true, true, false}, {FragmentParameter{0, 0, false, false}}}},
    };
    for (std::size_t i = 0; i < meshLayouts.size(); ++i) {
        require(meshKeys(meshLayouts[i].second) == meshKeys(meshLayouts[i].second), std::string("the key of ") + meshLayouts[i].first + " is not deterministic");
        for (std::size_t j = i + 1; j < meshLayouts.size(); ++j) {
            const auto left = meshKeys(meshLayouts[i].second);
            const auto right = meshKeys(meshLayouts[j].second);
            require(left.first != right.first && left.second != right.second, std::string("the mesh key confuses ") + meshLayouts[i].first + " with " + meshLayouts[j].first);
        }
    }

    const auto pixelKeys = [&](bool meshPath) {
        SampleRequest sample;
        sample.request.shader.stage = ShaderStage::Fragment;
        sample.request.context.compute.reset();
        sample.request.graphics = GraphicsCompileContext{0, {}, meshPath ? std::optional(MeshConfiguration{4u, 4u, 12u, 12u, 4u, 64u, 1024u, 0u, 4u}) : std::nullopt, std::nullopt, {}, std::nullopt};
        return std::pair{sample.Key(), RecompileCacheKey::ContextHash(sample.request)};
    };
    require(pixelKeys(true).first != pixelKeys(false).first && pixelKeys(true).second != pixelKeys(false).second, "a pixel shader on the mesh path shares a key with the vertex path");

    SampleRequest moved;
    moved.userData[0] ^= 0x10000u;
    moved.request.shader.codeAddress += 0x1000000u;
    moved.request.shader.headerAddress += 0x1000000u;
    moved.header[3] = std::byte{1};
    require(moved.Key() == key, "the key depends on the user data values or the addresses");
    std::array<std::byte, 32> descriptors{};
    const std::array regions{MemoryRegion{0x100000u, descriptors}};
    moved.request.context.memory = regions;
    std::fill(moved.userData.begin(), moved.userData.end(), 0xffffffffu);
    require(moved.Key() == key, "runtime descriptors changed the artifact key");
    SampleRequest vertex;
    vertex.request.shader.stage = ShaderStage::Vertex;
    vertex.request.context.compute.reset();
    vertex.request.context.vertex.emplace();
    auto& input = *vertex.request.context.vertex;
    input.resourcesNum = 1u;
    input.resources[0].fields = {0x10000u, 16u << 16u, 64u, (22u << 12u) | 0xfacu};
    input.resourcesDst[0] = {0u, 4u, 0u, 0u};
    const auto vertexKey = vertex.Key();
    const auto contextKey = RecompileCacheKey::ContextHash(vertex.request);
    input.resources[0].fields = {0x20000u, 32u << 16u, 128u, (77u << 12u) | 0xfa9u};
    require(vertex.Key() == vertexKey && RecompileCacheKey::ContextHash(vertex.request) == contextKey, "runtime vertex format, stride, swizzle or address changed the static interface key");
    input.resources[0].fields[3] = (20u << 12u) | 0xfacu;
    require(vertex.Key() != vertexKey && RecompileCacheKey::ContextHash(vertex.request) != contextKey, "integer vertex input reused a float interface");
    input.resources[0].fields[3] = (77u << 12u) | 0xfa9u;
    input.resourcesDst[0].registersNum = 2u;
    require(vertex.Key() != vertexKey, "vertex component interface did not change the artifact key");
    input.fetchEmbedded = true;
    const auto embeddedKey = vertex.Key();
    input.resources[0].fields = {};
    input.resourcesDst[0].fetchIndex = 1u;
    require(vertex.Key() == embeddedKey, "runtime vertex fetch descriptors changed the artifact key");
    input.resources[0].fields[3] = (20u << 12u) | 0xfacu;
    require(vertex.Key() == embeddedKey, "embedded numeric class changed the prepared template key");
    input.resourcesNum = 0u;
    require(vertex.Key() != embeddedKey, "embedded semantic interface did not change the artifact key");
    SampleRequest fragment;
    fragment.request.shader.stage = ShaderStage::Fragment;
    fragment.request.context.compute.reset();
    fragment.request.context.pixel.emplace();
    fragment.request.context.pixel->targetOutputMode[0] = 9;
    const auto fragmentKey = fragment.Key();
    const auto fragmentContextKey = RecompileCacheKey::ContextHash(fragment.request);
    fragment.request.context.pixel->targetExportMapping.fill(0x1bu);
    require(fragment.Key() == fragmentKey && RecompileCacheKey::ContextHash(fragment.request) == fragmentContextKey, "runtime export mapping changed the fragment artifact key");
    fragment.request.context.pixel->targetOutputMode[0] = 7;
    require(fragment.Key() != fragmentKey, "integer fragment output reused a float interface");
}

void verifyFloatModeSerialization() {
    const RequestSerializer serializer;
    SampleRequest sample;
    require(!serializer.Deserialize(serializer.Serialize(sample.request)).request.context.floatMode.has_value(), "an unknown float mode came back from serialization");
    const ShaderFloatMode mode{0x04u, false, true, true};
    sample.request.context.floatMode = mode;
    const auto back = serializer.Deserialize(serializer.Serialize(sample.request));
    require(back.request.context.floatMode == mode, "the float mode did not survive serialization");
}

int runLargeFileWrite() {
    constexpr std::size_t size = (std::size_t{2} << 30u) + 4096u;
    const auto directory = std::filesystem::temp_directory_path() / ("aps5-shader-disk-cache-large-file-test-" + std::to_string(std::random_device{}()));
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    std::filesystem::create_directories(directory, error);
    const auto space = std::filesystem::space(directory, error);
    if (error || space.available < size + (std::uintmax_t{1} << 30u)) {
        std::cout << "skipping: not enough free disk space for a file of more than 2 GiB\n";
        return 77;
    }
    std::vector<std::byte> bytes;
    try {
        bytes.resize(size);
    } catch (const std::bad_alloc&) {
        std::cout << "skipping: cannot allocate a buffer of more than 2 GiB\n";
        return 77;
    }
    const std::array<std::size_t, 6> samples{0u, (std::size_t{1} << 30u) - 1u, std::size_t{1} << 30u, (std::size_t{2} << 30u) - 1u, std::size_t{2} << 30u, size - 1u};
    for (std::size_t index = 0; index < samples.size(); ++index) bytes[samples[index]] = static_cast<std::byte>(0xa0u + index);
    const auto path = directory / "pipelines.bin";
    auto written = std::make_shared<std::promise<bool>>();
    auto result = written->get_future();
    std::thread([written, &bytes, path] {
        try {
            written->set_value(WriteFileAtomically(path, bytes));
        } catch (...) {
            written->set_exception(std::current_exception());
        }
    }).detach();
    if (result.wait_for(std::chrono::seconds(200)) != std::future_status::ready) {
        std::cerr << "WriteFileAtomically did not return within 200 s for a file of more than 2 GiB\n";
        std::cerr.flush();
        std::_Exit(1);
    }
    require(result.get(), "cannot write a cache file of more than 2 GiB");
    require(std::filesystem::file_size(path) == size, "the cache file of more than 2 GiB has the wrong size");
    {
        std::ifstream file(path, std::ios::binary);
        require(static_cast<bool>(file), "cannot open the cache file of more than 2 GiB");
        for (std::size_t index = 0; index < samples.size(); ++index) {
            file.seekg(static_cast<std::streamoff>(samples[index]));
            char value = 0;
            require(file.get(value) && static_cast<std::byte>(value) == bytes[samples[index]], "the cache file of more than 2 GiB differs at byte " + std::to_string(samples[index]));
        }
    }
    std::size_t entries = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        static_cast<void>(entry);
        ++entries;
    }
    require(entries == 1u, "the cache file of more than 2 GiB left a temporary file behind");
    std::filesystem::remove_all(directory, error);
    require(!error, "cannot remove the large file test directory: " + error.message());
    std::cout << "large cache file write test passed\n";
    return 0;
}

void verifyAtomicReplacement(const std::filesystem::path& directory) {
    struct Reader {
#ifdef _WIN32
        HANDLE file;
        explicit Reader(const std::filesystem::path& path)
            : file(CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr)) {
            require(file != INVALID_HANDLE_VALUE, "cannot retain the cache file");
        }
        ~Reader() { CloseHandle(file); }
        std::vector<std::byte> Read() {
            require(SetFilePointerEx(file, {}, nullptr, FILE_BEGIN), "cannot seek the retained cache file");
            std::array<std::byte, 32> bytes{};
            DWORD count = 0;
            require(ReadFile(file, bytes.data(), static_cast<DWORD>(bytes.size()), &count, nullptr), "cannot read the retained cache file");
            return {bytes.begin(), bytes.begin() + count};
        }
#else
        std::ifstream file;
        explicit Reader(const std::filesystem::path& path) : file(path, std::ios::binary) {
            require(file.is_open(), "cannot retain the cache file");
        }
        std::vector<std::byte> Read() {
            file.clear();
            file.seekg(0);
            require(!file.fail(), "cannot seek the retained cache file");
            std::array<std::byte, 32> bytes{};
            file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
            require(file.eof() && !file.bad(), "cannot read the retained cache file");
            return {bytes.begin(), bytes.begin() + file.gcount()};
        }
#endif
        Reader(const Reader&) = delete;
        Reader& operator=(const Reader&) = delete;
    };
    const auto root = directory / "atomic-replacement";
    const auto path = root / "entry.bin";
    const std::vector<std::byte> original{std::byte{0x12}, std::byte{0x00}, std::byte{0x34}};
    const std::vector<std::byte> replacement{std::byte{0xab}, std::byte{0xcd}, std::byte{0x00}, std::byte{0xef}};
    require(WriteFileAtomically(path, original), "cannot write the initial cache entry");
#ifdef _WIN32
    {
        const auto renaming = CreateFileW(path.c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                         nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        require(renaming != INVALID_HANDLE_VALUE, "cannot retain the cache entry's rename handle");
        struct Close { HANDLE file; ~Close() { CloseHandle(file); } } close{renaming};
        std::vector<std::byte> bytes;
        require(ReadWholeFile(path, bytes) && bytes == original, "cache reader failed while the published entry retained its rename handle");
    }
#endif
    {
        Reader old(path);
        require(old.Read() == original, "the retained reader did not read the original entry");
        require(WriteFileAtomically(path, replacement), "cannot replace cache entry with a retained reader");
        require(old.Read() == original, "replacement changed the retained original entry");
        std::vector<std::byte> fresh;
        require(ReadWholeFile(path, fresh) && fresh == replacement, "a fresh reader did not read the replacement entry");
        Reader current(path);
        require(current.Read() == replacement, "the replacement reader did not retain its entry");
        require(std::filesystem::remove(path), "cannot delete the replacement while readers retain it");
        require(!ReadWholeFile(path, fresh), "the deleted cache entry remains readable by path");
        require(old.Read() == original && current.Read() == replacement, "deletion invalidated retained cache readers");
    }
    require(std::filesystem::is_empty(root), "atomic replacement left a temporary file behind");
    require(std::filesystem::remove(root), "cannot remove the empty replacement directory");
}

void verifyStore() {
    require(ShaderDiskCache::Enabled(), "the disk cache is not enabled");
    SampleRequest sample;
    const auto key = sample.Key();
    const auto variant = std::make_shared<const CompiledVariant>(sampleVariant());
    const auto before = ShaderDiskCache::Totals();
    CompiledVariant loaded;
    require(!ShaderDiskCache::Load(key, loaded), "an entry loads before it was stored");
    ShaderDiskCache::Store(key, variant);
    ShaderDiskCache::Flush();
    require(ShaderDiskCache::Totals().writes == before.writes + 1, "the store did not write the entry");
    const auto path = ShaderDiskCache::EntryDirectory() / ShaderDiskCache::EntryName(key);
    require(std::filesystem::exists(path), "no entry file at " + path.string());
    require(ShaderDiskCache::Load(key, loaded), "a stored entry does not load");
    requireSameVariant(*variant, loaded, "store round trip");
    require(ShaderDiskCache::Totals().hits == before.hits + 1, "the load was not counted as a hit");

    std::vector<std::byte> file;
    require(ReadWholeFile(path, file), "cannot read the entry back");
    require(WriteFileAtomically(path, std::span(file).first(file.size() / 2)), "cannot truncate the entry");
    require(!ShaderDiskCache::Load(key, loaded), "a truncated entry loads");
    auto corrupt = file;
    corrupt[corrupt.size() - 5] ^= std::byte{0x40};
    require(WriteFileAtomically(path, corrupt), "cannot damage the entry");
    require(!ShaderDiskCache::Load(key, loaded), "a corrupt entry loads");
    require(ShaderDiskCache::Totals().loadFailures == before.loadFailures + 2, "the rejected entries were not counted");
    ShaderDiskCache::Store(key, variant);
    ShaderDiskCache::Flush();
    require(ShaderDiskCache::Load(key, loaded), "a rewritten entry does not load");
}

void verifyPersist() {
    SampleRequest sample;
    const auto key = sample.Key();
    const auto variant = sampleVariant();
    const auto persisted = ShaderDiskCache::Persist(key, variant);
    require(persisted != nullptr, "a persisted entry was not read back");
    const auto path = ShaderDiskCache::EntryDirectory() / ShaderDiskCache::EntryName(key);
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION region{};
    require(VirtualQuery(persisted->artifact.spirv.data(), &region, sizeof(region)) == sizeof(region) && region.Type == MEM_MAPPED, "persisted SPIR-V is not a mapped cache entry");
    const auto close = [](void* handle) { CloseHandle(handle); };
    std::unique_ptr<void, decltype(close)> writer(CreateFileW(path.c_str(), GENERIC_WRITE | DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr), close);
    require(writer.get() != INVALID_HANDLE_VALUE, "cannot retain the published cache writer");
    CompiledVariant loaded;
    require(ShaderDiskCache::Load(key, loaded), "cannot load a cache entry while its writer retains rename access");
    const auto& loadedWords = loaded.artifact.spirv;
    require(VirtualQuery(loadedWords.data(), &region, sizeof(region)) == sizeof(region) && region.Type == MEM_MAPPED, "cache loading with a retained writer did not map the entry");
    writer.reset();
#endif
    requireSameVariant(variant, *persisted, "persist round trip");
    auto detached = persisted->artifact.spirv;
    detached[0] ^= 1u;
    require(detached[0] != persisted->artifact.spirv[0], "changing a copied mapped module changed its owner");
    require(WriteFileAtomically(path, std::vector<std::byte>(64, std::byte{0x5a})), "cannot replace the persisted entry");
    std::error_code error;
    std::filesystem::remove(path, error);
    require(!error, "cannot remove the replaced cache entry");
    requireSameArtifact(variant.artifact, persisted->artifact, "a persisted entry whose file was replaced and removed");
}

struct ComputeRequest {
    std::vector<std::uint32_t> code{0xe0700000u, 0x80000000u, 0xbf810000u};
    std::array<std::uint32_t, 4> userData{0x10000000u, 0x00000000u, 0x40u, 0x00027facu};
    std::array<std::uint32_t, 4> capabilities{1u, 11u, 5347u, 4448u};
    std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    RecompileRequest request{};

    explicit ComputeRequest(bool useCache) {
        request.shader = {ShaderStage::Compute, 0x20000u, code, 0, {}};
        request.context.waveSize = 64;
        request.context.userDataBaseRegister = 0;
        request.context.userData = userData;
        request.context.compute = ShaderComputeStageInfo{{64u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
        request.target.vulkanVersion = 0x00401000u;
        request.target.spirvVersion = 0x00010300u;
        request.target.subgroupSize = 32;
        request.target.supportedCapabilities = capabilities;
        request.target.supportedExtensions = extensions;
        request.target.bdaAbiVersion = 1u;
        request.layout.pushConstantSizeBytes = 128;
        request.useCache = useCache;
    }
};

int runLoadingProcess() {
    ComputeRequest cached(true);
    const auto loaded = Recompile(cached.request);
    const auto totals = ShaderDiskCache::Totals();
    require(totals.hits == 1 && totals.misses == 0 && totals.loadFailures == 0, "the second process did not load the stored variant (hits " + std::to_string(totals.hits) + ", misses " + std::to_string(totals.misses) + ")");
    ComputeRequest fresh(false);
    const auto compiled = Recompile(fresh.request);
    requireSameResult(compiled, loaded, "loaded against compiled");
    require(!loaded.spirv.empty(), "the loaded SPIR-V is empty");
    std::cout << "second process loaded " << loaded.spirv.size() << " SPIR-V words from the disk cache\n";
    return 0;
}

void verifyAcrossProcesses(const char* self) {
    ComputeRequest request(true);
    const auto before = ShaderDiskCache::Totals();
    const auto compiled = Recompile(request.request);
    ShaderDiskCache::Flush();
    const auto after = ShaderDiskCache::Totals();
    require(after.misses == before.misses + 1 && after.writes == before.writes + 1, "the first compile was not looked up and stored");
    require(!compiled.spirv.empty(), "the compile produced no SPIR-V");
    const std::string command = "\"" + std::string(self) + "\" --load";
    require(std::system(command.c_str()) == 0, "the loading process failed");
    for (const auto* mode : {"--load-lazy-plan", "--load-eager-plan"}) {
        const std::string planCommand = "\"" + std::string(self) + "\" " + mode;
        require(std::system(planCommand.c_str()) == 0, std::string(mode) + " process failed");
    }
}

class DirectImageCacheLoopGuard {
public:
    explicit DirectImageCacheLoopGuard(const char* value, const char* variable = "APS5_LOOP_GUARD") : name(variable) {
        if (const auto* current = std::getenv(name)) { present = true; previous = current; }
        Set(value);
    }
    ~DirectImageCacheLoopGuard() { Set(present ? previous.c_str() : nullptr); }
private:
    void Set(const char* value) {
        if (value != nullptr) setEnvironment(name, value);
        else {
#ifdef _WIN32
            _putenv_s(name, "");
#else
            unsetenv(name);
#endif
        }
        ShaderRecompiler::RefreshEnvironmentGuards();
    }
    bool present = false;
    std::string previous;
    const char* name;
};

class PlanIrCapture {
public:
    PlanIrCapture() {
        std::fflush(stderr);
        file = std::tmpfile();
        require(file != nullptr, "lazy plan: cannot create IR capture");
#ifdef _WIN32
        saved = _dup(_fileno(stderr));
        const auto redirected = saved < 0 ? -1 : _dup2(_fileno(file), _fileno(stderr));
#else
        saved = dup(fileno(stderr));
        const auto redirected = saved < 0 ? -1 : dup2(fileno(file), fileno(stderr));
#endif
        if (redirected < 0) {
            Restore();
            std::fclose(file);
            file = nullptr;
            throw std::runtime_error("lazy plan: cannot redirect IR capture");
        }
    }
    ~PlanIrCapture() {
        if (file != nullptr) { Restore(); std::fclose(file); }
    }
    std::string Finish() {
        Restore();
        std::rewind(file);
        std::string text;
        std::array<char, 4096> buffer{};
        for (std::size_t read = 0; (read = std::fread(buffer.data(), 1, buffer.size(), file)) != 0;) text.append(buffer.data(), read);
        const bool failed = std::ferror(file) != 0;
        std::fclose(file);
        file = nullptr;
        require(!failed, "lazy plan: cannot read IR capture");
        return text;
    }
private:
    void Restore() {
        std::fflush(stderr);
        if (saved < 0) return;
#ifdef _WIN32
        _dup2(saved, _fileno(stderr));
        _close(saved);
#else
        dup2(saved, fileno(stderr));
        close(saved);
#endif
        saved = -1;
    }
    std::FILE* file = nullptr;
    int saved = -1;
};

std::size_t planTranslations(std::string_view text) {
    std::size_t count = 0;
    for (std::size_t cursor = 0; (cursor = text.find(" after translate", cursor)) != std::string_view::npos; ++cursor) ++count;
    return count;
}

void verifyLatePlanFailure(bool eager) {
    ComputeRequest sample(true);
    sample.code = {0xf40c0200u, 0xfa000000u, 0xf4000400u, 0xfa000050u, 0xbf8cc07fu, 0xbf068010u, 0xbf850003u,
        0xf4080500u, 0xfa000020u, 0xbf820002u, 0xf4080500u, 0x20000000u, 0xf4080600u, 0xfa000040u,
        0xbf8cc07fu, 0xf09c0f08u, 0x00a20000u, 0xbf8c3f70u, 0xe0700000u, 0x80060000u, 0xbf810000u};
    sample.request.shader = {ShaderStage::Compute, 0xabc58100u, sample.code, 0, {}};
    sample.request.context.waveSize = 32;
    sample.request.context.compute->numThreads = {32u, 1u, 1u};
    sample.request.context.userData = std::span(sample.userData).first(2);
    sample.capabilities[0] = 29u;
    const auto fail = [&] {
        try { static_cast<void>(PrepareShader(sample.request)); }
        catch (const std::runtime_error& error) { return std::string(error.what()); }
        throw std::runtime_error("lazy plan: unsupported descriptor Phi was accepted");
    };
    const auto before = ShaderDiskCache::Totals();
    PlanIrCapture trace;
    const auto expected = fail();
    require(expected.find("GetSamplerResource dword 0 is not a valid runtime value") != std::string::npos, "lazy plan: wrong late planning failure: " + expected);
    require(fail() == expected, "lazy plan: repeated planning failure changed");
    std::barrier start(4);
    std::latch launchReady(1);
    bool launchFailed = false;
    std::array<std::future<std::string>, 4> failures;
    try {
        for (auto& failure : failures) failure = std::async(std::launch::async, [&] {
            launchReady.wait();
            if (launchFailed) return std::string{};
            start.arrive_and_wait();
            return fail();
        });
    } catch (...) {
        launchFailed = true;
        launchReady.count_down();
        throw;
    }
    launchReady.count_down();
    for (auto& failure : failures) require(failure.get() == expected, "lazy plan: concurrent planning failure changed");
    const auto ir = trace.Finish();
    require(planTranslations(ir) == 1u && ir.find(" after fold") != std::string::npos,
        "lazy plan: late failure was not memoized after translation/folding: " + ir);
    ShaderDiskCache::Flush();
    const auto after = ShaderDiskCache::Totals();
    require(after.misses == before.misses + (eager ? 0u : 1u) && after.hits == before.hits && after.writes == before.writes,
        "lazy plan: failed planning repeated disk lookup or stored an artifact");
    std::cout << "late plan failure: one frontend, repeated/concurrent identical refusal, no cache store\n";
}

int runLazyPlanLoadingProcess(bool eager) {
    DirectImageCacheLoopGuard mode(eager ? "0" : nullptr, "APS5_NO_PERF_LAZY_CAPTURE_PLAN");
    DirectImageCacheLoopGuard memo(nullptr, "APS5_NO_FAILURE_MEMO");
    DirectImageCacheLoopGuard dump("all", "APS5_DUMP_IR");
    ComputeRequest registered(true);
    PlanIrCapture preparationTrace;
    const auto handle = PrepareShader(registered.request);
    const auto preparationIr = preparationTrace.Finish();
    const auto totals = ShaderDiskCache::Totals();
    require(totals.hits == 1u && totals.misses == 0u && totals.loadFailures == 0u, "lazy plan: preparation did not verify a disk hit");
    require(planTranslations(preparationIr) == (eager ? 1u : 0u), "lazy plan: wrong warm preparation frontend count: " + preparationIr);
    std::array<std::unique_ptr<ComputeRequest>, 4> current;
    std::array<std::shared_ptr<const ResourceCapture>, 4> captures;
    std::array<std::shared_ptr<const RecompileResult>, 4> results;
    std::array<std::future<void>, 4> workers;
    std::barrier start(4);
    std::latch launchReady(1);
    bool launchFailed = false;
    for (std::size_t index = 0; index < current.size(); ++index) {
        current[index] = std::make_unique<ComputeRequest>(true);
        current[index]->request.shader.codeAddress = 0x30000u;
        current[index]->userData[0] += static_cast<std::uint32_t>(index + 1u) * 0x10000u;
        current[index]->userData[2] += static_cast<std::uint32_t>(index) * 16u;
    }
    PlanIrCapture captureTrace;
    try {
        for (std::size_t index = 0; index < current.size(); ++index) {
            workers[index] = std::async(std::launch::async, [&, index] {
                launchReady.wait();
                if (launchFailed) return;
                start.arrive_and_wait();
                const auto& request = current[index]->request;
                const auto invocation = PreparedShaderInvocation::TryCreate(request, handle);
                require(invocation.has_value(), "lazy plan: relocated invocation did not match the loaded artifact");
                SrtRuntime runtime{};
                runtime.userData = current[index]->userData;
                runtime.shaderBase = request.shader.codeAddress;
                captures[index] = index % 2u == 0u ? invocation->Capture(runtime) : CaptureResources(request, runtime, *handle);
                results[index] = index % 2u == 0u ? invocation->Materialize(*captures[index]) : MaterializeShader(request, *captures[index], *handle);
            });
        }
    } catch (...) {
        launchFailed = true;
        launchReady.count_down();
        throw;
    }
    launchReady.count_down();
    for (auto& worker : workers) worker.get();
    const auto captureIr = captureTrace.Finish();
    require(planTranslations(captureIr) == (eager ? 0u : 1u), "lazy plan: concurrent first capture did not build exactly one plan: " + captureIr);
    PlanIrCapture repeatedTrace;
    for (std::size_t index = 0; index < current.size(); ++index) {
        const auto& snapshot = captures[index]->snapshot;
        require(captures[index]->plan == captures[0]->plan && captures[index]->plan->shaderHash == registered.request.shader.codeAddress,
            "lazy plan: concurrent or relocated capture changed source-plan identity");
        require(snapshot.buffers.size() == 1u && snapshot.buffers[0].dwordCount == 4u
            && std::equal(current[index]->userData.begin(), current[index]->userData.end(), snapshot.buffers[0].dwords.begin()),
            "lazy plan: capture reused another invocation's descriptor words");
        SrtRuntime runtime{};
        runtime.userData = current[index]->userData;
        runtime.shaderBase = current[index]->request.shader.codeAddress;
        const auto repeated = CaptureResources(current[index]->request, runtime, *handle);
        require(repeated->plan == captures[0]->plan, "lazy plan: repeated capture replaced the plan");
        requireSameResult(*results[index], *MaterializeShader(current[index]->request, *repeated, *handle), "lazy plan repeated capture");
    }
    const auto repeatedIr = repeatedTrace.Finish();
    require(planTranslations(repeatedIr) == 0u, "lazy plan: repeated capture reran the frontend: " + repeatedIr);
    for (std::size_t index = 0; index < current.size(); ++index) {
        current[index]->request.useCache = false;
        requireSameResult(Recompile(current[index]->request), *results[index], "lazy plan current descriptor against uncached compile");
    }
    std::cout << (eager ? "eager" : "lazy") << " verified disk hit: preparation frontend=" << planTranslations(preparationIr)
        << ", first concurrent capture frontend=" << planTranslations(captureIr) << ", repeated=0; origin 0x20000, capture 0x30000, four exact current descriptors\n";
    verifyLatePlanFailure(eager);
    return 0;
}

void verifyUnnormalizedWrapTransitions(bool coldLoopOnly = false) {
    DirectImageCacheLoopGuard noLoop(coldLoopOnly ? "8" : nullptr);
    DirectImageCacheLoopGuard policy("0", "APS5_UNNORM_WRAP_GUARD");
    ComputeRequest sample(true);
    sample.code = {0x7e0402ffu, 0x3f000000u, 0x7e0602ffu, 0x3f000000u, 0xf09c0108u, 0x00610802u, 0xe0700000u, 0x80000800u, 0xbf810000u};
    std::array<std::uint32_t, 20> users{0x10000000u, 0u, 256u, 0x00027facu,
        0x20000000u, 0x00700000u, 0x815fc15fu, 0x9050024cu, 0u, 0u, 0u, 0u,
        0x00008000u, 0x00fff000u, 0x05000000u, 0u};
    sample.request.shader = {ShaderStage::Compute, 0xabc57000u, sample.code, 0, {}};
    sample.request.context.userData = users;
    SrtRuntime runtime{};
    runtime.userData = users;
    if (coldLoopOnly) {
        sample.code.insert(sample.code.begin(), 0xbe900382u);
        sample.code.insert(sample.code.end() - 1, {0x80908110u, 0xbf068010u, 0xbf84fff5u, 0x7e3c0280u, 0xdc708000u, 0x0012081eu});
        users[18] = 0x30000000u;
        sample.request.shader = {ShaderStage::Compute, 0xabc57300u, sample.code, 0, {}};
        const auto loopArtifact = PrepareShader(sample.request);
        require(GetPreparedArtifact(*loopArtifact).bdaAbiVersion == BdaAbi::Version, "loop artifact lacks its independent fault interface");
        DirectImageCacheLoopGuard cleared(nullptr);
        setEnvironment("APS5_UNNORM_WRAP_GUARD", "1");
        bool refused = false;
        try { static_cast<void>(MaterializeShader(sample.request, *CaptureResources(sample.request, runtime, *loopArtifact), *loopArtifact)); }
        catch (const std::runtime_error& error) { refused = std::string_view(error.what()) == "unnormalized wrap requires emitted per-sampler guard proof"; }
        require(refused, "loop artifact's fault buffer was mistaken for emitted wrap capability after clearing the loop guard");
        std::cout << "cold loop artifact: independent BDA1 cannot replace wrap capability after clearing policy\n";
        return;
    }
    const auto handle = PrepareShader(sample.request);
    const auto capture = CaptureResources(sample.request, runtime, *handle);
    const auto materialize = [&] { return MaterializeShader(sample.request, *capture, *handle); };
    const auto hasFault = [](const auto& result) { return std::any_of(result.bindings.begin(), result.bindings.end(), [](const auto& binding) { return binding.role == DescriptorRole::FaultBuffer; }); };
    const auto off = materialize();
    require(!off->hasGuardedUnnormalizedWrap && !hasFault(*off), "disabled wrap retained a guard or fault completion");
    setEnvironment("APS5_UNNORM_WRAP_GUARD", "1");
    const auto on = materialize();
    require(on->hasGuardedUnnormalizedWrap && hasFault(*on) && on->PipelineVariantId() != off->PipelineVariantId(), "wrap policy reused the disabled result memo");
    require(materialize() == on, "enabled wrap did not reuse its result memo");
    {
        DirectImageCacheLoopGuard loop("8");
        bool refused = false;
        try { static_cast<void>(materialize()); }
        catch (const std::runtime_error& error) { refused = std::string_view(error.what()) == "unnormalized wrap requires a compute fault buffer without a loop guard"; }
        require(refused, "enabled loop guard reused the guarded wrap result memo");
    }
    require(materialize() == on, "clearing loop guard lost the valid wrap result memo");
    sample.request.target.bdaAbiVersion = 0u;
    bool abiRefused = false;
    try { static_cast<void>(materialize()); }
    catch (const std::runtime_error& error) { abiRefused = std::string_view(error.what()) == "ShaderRecompiler: prepared artifact does not match the static ABI"; }
    require(abiRefused, "changed request ABI reused the guarded wrap result memo");
    sample.request.target.bdaAbiVersion = BdaAbi::Version;
    setEnvironment("APS5_UNNORM_WRAP_GUARD", "0");
    require(materialize() == off, "disabling wrap reused the enabled result memo");
    setEnvironment("APS5_UNNORM_WRAP_GUARD", "1");
    for (const auto word : {0x12u, 0x8012u}) {
        users[12] = word;
        const auto ordinary = MaterializeShader(sample.request, *CaptureResources(sample.request, runtime, *handle), *handle);
        require(!ordinary->hasGuardedUnnormalizedWrap && !hasFault(*ordinary), "normalized or clamp sampling retained wrap fault completion");
    }
    users[12] = 0x8000u;
    sample.code[4] = 0xf0900108u;
    sample.request.shader.codeAddress += 0x100u;
    bool refused = false;
    try {
        const auto unsupported = PrepareShader(sample.request);
        static_cast<void>(MaterializeShader(sample.request, *CaptureResources(sample.request, runtime, *unsupported), *unsupported));
    }
    catch (const std::runtime_error& error) { refused = std::string_view(error.what()) == "unnormalized wrap requires emitted per-sampler guard proof"; }
    require(refused, "non-LZ artifact was accepted without emitted wrap capability");
    std::cout << "unnormalized wrap: policy/result memo, loop guard, pruning and missing capability controls passed\n";
}

struct DirectImageCacheRequest : ComputeRequest {
    std::array<std::uint32_t, 12> imageUsers{0x00010000u, 56u << 20u, 0u, 0xd0000facu};
    std::array<std::uint32_t, 6> imageCapabilities{spv::CapabilityShader, spv::CapabilityInt64,
        spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess,
        spv::CapabilityStorageImageWriteWithoutFormat, spv::CapabilityStorageImageExtendedFormats};

    explicit DirectImageCacheRequest(bool loop = false) : ComputeRequest(true) {
        code = {0x7e3c0280u, 0x7e3e0280u, 0x7e400280u, 0x7e2802ffu, 0x3f800000u,
            0x7e2a0280u, 0x7e2c0280u, 0x7e2e02ffu, 0x3f800000u, 0xbf068008u, 0xbf850000u};
        const auto skip = code.size() - 1u;
        if (loop) code.push_back(0xbe890382u);
        const auto loopStart = code.size();
        code.insert(code.end(), {0xf0202f28u, 0x0000141eu});
        if (loop) {
            code.insert(code.end(), {0x80898109u, 0xbf068009u, 0xbf840000u});
            code.back() |= static_cast<std::uint16_t>(static_cast<std::int32_t>(loopStart) - static_cast<std::int32_t>(code.size()));
        }
        code[skip] |= static_cast<std::uint32_t>(code.size() - skip - 1u);
        if (loop) {
            code.insert(code.end(), {0xdc708000u, 0x000a141eu});
            imageUsers[10] = 0x20000000u;
        }
        code.push_back(0xbf810000u);
        request.shader = {ShaderStage::Compute, loop ? 0xd1aec7200ull : 0xd1aec7100ull, code, 0, {}};
        request.context.waveSize = 32;
        request.context.userData = imageUsers;
        request.context.compute = ShaderComputeStageInfo{{1, 1, 1}, 0, {false, false, false}, false, 1};
        request.target.supportedCapabilities = imageCapabilities;
        request.target.bdaAbiVersion = BdaAbi::Version;
    }
    void Descriptor(bool valid) { imageUsers[3] = (valid ? 0xd0000000u : 0x80000000u) | 0xfacu; }
    std::shared_ptr<const ResourceCapture> Capture(const SourceHandle& handle) const {
        SrtRuntime runtime{};
        runtime.userData = imageUsers;
        return CaptureResources(request, runtime, handle);
    }
    std::shared_ptr<const RecompileResult> Materialize(const SourceHandle& handle) const {
        return MaterializeShader(request, *Capture(handle), handle);
    }
};

bool hasDirectImageRefusalCase(const CompiledShaderArtifact& artifact) {
    for (std::size_t offset = 5; offset < artifact.spirv.size();) {
        const auto words = artifact.spirv[offset] >> 16u;
        require(words != 0 && offset + words <= artifact.spirv.size(), "direct image disk fixture has malformed SPIR-V");
        if ((artifact.spirv[offset] & 0xffffu) == spv::OpSwitch) {
            for (std::size_t word = 3; word + 1 < words; word += 2) {
                if (artifact.spirv[offset + word] == PipelineSpecialization::DeferredDirectImageMode) return true;
            }
        }
        offset += words;
    }
    return false;
}

void verifyDirectImageSelection(const RecompileResult& result, bool valid) {
    std::size_t writes = 0;
    for (std::size_t offset = 5; offset < result.spirv.size(); offset += result.spirv[offset] >> 16u) {
        writes += (result.spirv[offset] & 0xffffu) == spv::OpImageWrite;
    }
    std::size_t images = 0;
    for (const auto& binding : result.bindings) if (binding.role == DescriptorRole::GuestImages) images += binding.count;
    require(writes == (valid ? 1u : 0u) && images == (valid ? 1u : 0u), "direct image disk invocation retained the wrong image accesses or bindings");
    require(std::ranges::all_of(result.specialization, [](const PipelineSpecializationConstant& constant) { return PipelineSpecialization::IsWorkgroupOffset(constant.id) && constant.value == 0u; }) && result.hasDeferredDirectImage == !valid,
        "direct image disk invocation lost its valid/refused completion distinction");
    if (!valid) require(result.bdaAbiVersion == BdaAbi::Version && std::any_of(result.bindings.begin(), result.bindings.end(), [](const auto& binding) {
        return binding.role == DescriptorRole::FaultBuffer;
    }), "direct image disk refusal has no recognized fault-buffer ABI");
}

void directImageCpuRefusal(const std::function<void()>& action, const char* what) {
    try { action(); }
    catch (const std::exception& error) {
        const std::string message = error.what();
        require(message.find("incompatible with the static runtime image interface") != std::string::npos,
            std::string(what) + ": unexpected refusal: " + error.what());
        return;
    }
    throw std::runtime_error(std::string(what) + ": incompatible image escaped CPU refusal");
}

void verifyDirectImageCacheTransitions(DirectImageCacheRequest& sample, const SourceHandle& handle) {
    const auto& artifact = GetPreparedArtifact(handle);
    require(artifact.bdaAbiVersion == BdaAbi::Version && artifact.runtimeAbiVersion == RuntimeAbi::Version && hasDirectImageRefusalCase(artifact),
        "direct image disk artifact was not prepared with refusal capability");
    std::shared_ptr<const RecompileResult> firstValid;
    std::shared_ptr<const RecompileResult> firstRefused;
    for (const bool valid : {true, false, true}) {
        sample.Descriptor(valid);
        for (const auto flag : {0u, 1u, 0u}) {
            sample.imageUsers[8] = flag;
            const auto result = sample.Materialize(handle);
            require(result->cacheHit && result->variantId == artifact.variantId, "direct image disk transition replaced the prepared artifact");
            verifyDirectImageSelection(*result, valid);
            if (valid) require(result->bdaAbiVersion == 0 && std::none_of(result->bindings.begin(), result->bindings.end(), [](const auto& binding) {
                return binding.role == DescriptorRole::FaultBuffer;
            }), "valid image-only disk invocation retained unnecessary fault completion");
            auto& first = valid ? firstValid : firstRefused;
            if (first == nullptr) first = result;
            else require(result->PipelineVariantId() == first->PipelineVariantId() && result->specialization == first->specialization
                && result->spirv == first->spirv,
                "direct image disk transition keyed or folded the runtime branch flag");
            if (flag != 0) require(result->pushConstants != first->pushConstants || !sameBindings(result->bindings, first->bindings),
                "direct image disk invocation failed to capture the changed runtime branch flag");
        }
    }
    require(firstValid->PipelineVariantId() != firstRefused->PipelineVariantId(), "direct image disk valid/refused specializations collided");
    sample.Descriptor(false);
    const auto cachedCapture = sample.Capture(handle);
    {
        DirectImageCacheLoopGuard guarded("8");
        directImageCpuRefusal([&] { static_cast<void>(sample.Capture(handle)); }, "warm disk artifact capture with loop guard");
        directImageCpuRefusal([&] { static_cast<void>(MaterializeShader(sample.request, *cachedCapture, handle)); }, "warm disk result memo with loop guard");
    }
    verifyDirectImageSelection(*sample.Materialize(handle), false);
    sample.Descriptor(true);
    requireSameResult(*firstValid, *sample.Materialize(handle), "direct image disk valid descriptor restored after guard");
}

int runDirectImageLoadingProcess(std::string_view mode) {
    const bool guardedArtifact = mode == "--load-guarded-direct-image";
    const bool guardedNow = guardedArtifact || mode == "--load-direct-image-loop-guard";
    DirectImageCacheRequest sample(guardedArtifact);
    {
        DirectImageCacheLoopGuard baseline(nullptr);
        std::vector<std::byte> baselineKey;
        ShaderDiskCache::BuildKey(sample.request, sample.request.target.subgroupSize, baselineKey);
    }
    std::shared_ptr<const SourceHandle> handle;
    {
        DirectImageCacheLoopGuard guard(guardedNow ? "8" : nullptr);
        handle = PrepareShader(sample.request);
        require(GetPreparedArtifact(*handle).bdaAbiVersion == BdaAbi::Version, "direct image loaded artifact lost its recognized BDA ABI");
        const auto loaded = ShaderDiskCache::Totals();
        require(loaded.hits == 1 && loaded.misses == 0 && loaded.loadFailures == 0 && loaded.writes == 0,
            "direct image child did not load exactly one serialized artifact without recompiling");
        require(sample.Materialize(*handle)->cacheHit, "direct image child did not reuse its loaded artifact");
        if (guardedNow) {
            sample.Descriptor(false);
            directImageCpuRefusal([&] { static_cast<void>(sample.Materialize(*handle)); }, "cold child loading with loop guard");
        } else verifyDirectImageCacheTransitions(sample, *handle);
    }
    if (guardedArtifact) {
        DirectImageCacheLoopGuard disabled(nullptr);
        require(!hasDirectImageRefusalCase(GetPreparedArtifact(*handle)), "guarded disk artifact unexpectedly contains a refusal arm");
        directImageCpuRefusal([&] { static_cast<void>(sample.Materialize(*handle)); }, "guarded disk artifact after loop guard is cleared");
        sample.Descriptor(true);
        verifyDirectImageSelection(*sample.Materialize(*handle), true);
    }
    ShaderDiskCache::Flush();
    const auto after = ShaderDiskCache::Totals();
    require(after.hits == 1 && after.misses == 0 && after.writes == 0 && after.loadFailures == 0,
        "direct image child transitions reloaded or rewrote the serialized artifact");
    std::cout << mode << ": loaded one artifact, no compile or write, cache transitions passed\n";
    return 0;
}

void verifyDirectImageAcrossProcesses(const char* self) {
    DirectImageCacheLoopGuard disabled(nullptr);
    DirectImageCacheRequest sample;
    const auto before = ShaderDiskCache::Totals();
    const auto handle = PrepareShader(sample.request);
    verifyDirectImageCacheTransitions(sample, *handle);
    ShaderDiskCache::Flush();
    const auto normal = ShaderDiskCache::Totals();
    require(normal.misses == before.misses + 1 && normal.writes == before.writes + 1 && normal.hits == before.hits,
        "direct image cold artifact was not compiled and serialized exactly once");
    DirectImageCacheRequest guardedSample(true);
    std::shared_ptr<const SourceHandle> guardedHandle;
    {
        DirectImageCacheLoopGuard guard("8");
        guardedHandle = PrepareShader(guardedSample.request);
        require(GetPreparedArtifact(*guardedHandle).bdaAbiVersion == BdaAbi::Version, "guarded artifact must retain its independent BDA fault buffer");
        require(!hasDirectImageRefusalCase(GetPreparedArtifact(*guardedHandle)), "cold guarded artifact acquired refusal capability");
        guardedSample.Descriptor(false);
        directImageCpuRefusal([&] { static_cast<void>(guardedSample.Materialize(*guardedHandle)); }, "cold compiled artifact with loop guard");
    }
    directImageCpuRefusal([&] { static_cast<void>(guardedSample.Materialize(*guardedHandle)); }, "cold guarded artifact after clearing loop guard");
    ShaderDiskCache::Flush();
    const auto guarded = ShaderDiskCache::Totals();
    require(guarded.misses == normal.misses + 1 && guarded.writes == normal.writes + 1 && guarded.hits == normal.hits,
        "guarded direct image artifact was not compiled and serialized exactly once");
    for (const auto* mode : {"--load-direct-image-refusal", "--load-direct-image-loop-guard", "--load-guarded-direct-image"}) {
        const std::string command = "\"" + std::string(self) + "\" " + mode;
        require(std::system(command.c_str()) == 0, std::string("direct image loading process failed: ") + mode);
    }
}

struct ClockRequest {
    std::vector<std::uint32_t> code{0xf4900100u, 0x00000000u, 0xbf8cc07fu, 0x7e020204u, 0xe0700000u, 0x80000100u, 0xf800180fu, 0x01010101u, 0xbf810000u};
    std::array<std::uint32_t, 4> userData{0x10000000u, 0x00000000u, 0x40u, 0x00027facu};
    std::array<std::uint32_t, 4> capabilities{1u, 11u, 5347u, 4448u};
    std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    RecompileRequest request{};

    explicit ClockRequest(std::uint32_t stride = 0) {
        userData[1] = stride << 16u;
        ShaderPixelStageInfo pixel{};
        pixel.inputAddr = PixelInputBit(PixelInput::PerspectiveCenter);
        pixel.hasPerspectiveCenterVgpr = true;
        pixel.targetOutputMode[0] = 9u;
        pixel.targetExportMapping.fill(0xe4u);
        request.shader = {ShaderStage::Fragment, 0x30000u, code, 0, {}};
        request.context.waveSize = 64;
        request.context.userDataBaseRegister = 0;
        request.context.userData = userData;
        request.context.pixel = pixel;
        request.target.vulkanVersion = 0x00401000u;
        request.target.spirvVersion = 0x00010300u;
        request.target.subgroupSize = 32;
        request.target.supportedCapabilities = capabilities;
        request.target.supportedExtensions = extensions;
        request.target.bdaAbiVersion = 1u;
        request.layout.pushConstantSizeBytes = 128;
    }
};

std::string emissionFailure(const RecompileRequest& request) {
    try {
        static_cast<void>(PrepareShader(request));
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        require(message.find("reads the shader clock, which needs the device's VK_KHR_shader_clock") != std::string::npos, "unexpected failure: " + message.substr(0, message.find("RecompileRequest:")));
        return message;
    }
    throw std::runtime_error("a shader clock read compiled without VK_KHR_shader_clock");
}

std::uint64_t diskMisses() {
    return ShaderDiskCache::Totals().misses;
}

void verifyEmissionFailureMemo() {
    const auto before = ShaderDiskCache::Totals();
    ClockRequest clock;
    const auto first = emissionFailure(clock.request);
    require(diskMisses() == before.misses + 1, "the first compile was not looked up");
    require(emissionFailure(clock.request) == first, "the second compile failed with another message");
    require(diskMisses() == before.misses + 1, "the second compile looked up and emitted the program again");

    ClockRequest layout;
    layout.request.layout = {0, 0, 16, 112};
    static_cast<void>(emissionFailure(layout.request));
    require(diskMisses() == before.misses + 2, "another binding layout reused the failure");
    ClockRequest stride(16);
    static_cast<void>(emissionFailure(stride.request));
    require(diskMisses() == before.misses + 2, "a runtime buffer stride change repeated static preparation");
    ClockRequest relocated;
    relocated.request.shader.codeAddress += 0x1000u;
    static_cast<void>(emissionFailure(relocated.request));
    require(diskMisses() == before.misses + 3, "a relocated copy reused the failure");
    require(emissionFailure(clock.request) == first && diskMisses() == before.misses + 3, "the other requests replaced the failure");

    ClockRequest concurrent;
    concurrent.request.layout.pushConstantSizeBytes = 64;
    std::array<std::future<std::string>, 4> failures;
    for (auto& future : failures) future = std::async(std::launch::async, [&concurrent] { return emissionFailure(concurrent.request); });
    std::vector<std::string> messages;
    for (auto& future : failures) messages.push_back(future.get());
    require(std::all_of(messages.begin(), messages.end(), [&](const std::string& message) { return message == messages.front(); }), "concurrent requests failed with different messages");
    require(diskMisses() == before.misses + 4, "concurrent requests emitted the program " + std::to_string(diskMisses() - before.misses - 3) + " times");

    ShaderDiskCache::Flush();
    require(ShaderDiskCache::Totals().writes == before.writes, "an emission failure was stored in the disk cache");
}

int runWithoutFailureMemo() {
    ClockRequest clock;
    const auto first = emissionFailure(clock.request);
    require(emissionFailure(clock.request) == first, "the second compile failed with another message");
    require(diskMisses() == 2, "APS5_NO_FAILURE_MEMO=1 did not emit the program again (" + std::to_string(diskMisses()) + " lookups)");
    std::cout << "APS5_NO_FAILURE_MEMO=1 emitted the failing program twice\n";
    return 0;
}

void verifyFailureMemoSwitch(const char* self) {
    setEnvironment("APS5_NO_FAILURE_MEMO", "1");
    const std::string command = "\"" + std::string(self) + "\" --no-failure-memo";
    require(std::system(command.c_str()) == 0, "the process with APS5_NO_FAILURE_MEMO=1 failed");
}

void verifyInvocationIsolation() {
    ComputeRequest request(true);
    const auto first = Recompile(request.request);
    request.userData[0] += 0x10000u;
    const auto second = Recompile(request.request);
    require(first.variantId != 0 && first.variantId == second.variantId, "a buffer address change did not reuse the artifact");
    require(second.cacheHit, "the second invocation missed the compiled cache");
    requireSameArtifact(first, second, "invocations sharing an artifact");
    require(first.spirv.data() == second.spirv.data(), "invocations duplicated SPIR-V storage");
    require(!sameBindings(first.bindings, second.bindings) || first.pushConstants != second.pushConstants, "the second invocation reused stale resource data");
    request.userData[0] -= 0x10000u;
    const auto restored = Recompile(request.request);
    requireSameResult(first, restored, "restored invocation");
    const auto original = request.userData;
    const std::array<std::array<std::uint32_t, 4>, 10> descriptors{{
        {original[0] + 0x20000u, original[1], original[2], original[3]},
        {original[0] + 1u, original[1], original[2], original[3]},
        {original[0] + 2u, original[1], original[2], original[3]},
        {original[0] + 3u, original[1], original[2], original[3]},
        {original[0], original[1] | 1u, original[2], original[3]},
        {original[0], original[1], original[2] / 2u, original[3]},
        {original[0], original[1] | (16u << 16u), original[2], original[3]},
        {original[0], original[1], original[2], 0x31004688u},
        {original[0], original[1] | 0x80100000u, original[2], original[3] | 0x00600000u},
        {0u, 0u, 0u, 0u}
    }};
    for (const auto& descriptor : descriptors) {
        request.userData = descriptor;
        const auto changed = Recompile(request.request);
        require(changed.cacheHit && first.variantId == changed.variantId, "buffer metadata changed the compiled variant");
        if ((descriptor[0] & 3u) == (original[0] & 3u) && (descriptor[1] & 0xffff0000u) == (original[1] & 0xffff0000u) && descriptor[3] == original[3] && (descriptor[0] != 0u || (descriptor[1] & 0xffffu) != 0u)) {
            require(first.spirv.data() == changed.spirv.data() && first.PipelineVariantId() == changed.PipelineVariantId(), "buffer address or size rebuilt the specialized module");
        }
        if (descriptor[2] == 0u) {
            require(first.spirv.data() == changed.spirv.data() && first.PipelineVariantId() == changed.PipelineVariantId(), "a null V# did not specialize with the bound slot's substitute");
            continue;
        }
        ComputeRequest fresh(false);
        fresh.userData = descriptor;
        requireSameResult(Recompile(fresh.request), changed, "reused binding plan against fresh compilation");
        const auto repeated = Recompile(request.request);
        require(changed.spirv.data() == repeated.spirv.data() && changed.PipelineVariantId() == repeated.PipelineVariantId(), "buffer specialization did not reuse its module");
    }
    request.userData = original;
    request.userData[3] |= 0x40000000u;
    bool rejected = false;
    try { static_cast<void>(Recompile(request.request)); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "an unsupported buffer descriptor type was accepted");
}

void verifyResultMemoBudget() {
    ComputeRequest request(true);
    const auto handle = PrepareShader(request.request);
    const auto materialize = [&](std::uint32_t address) {
        request.userData[0] = address;
        SrtRuntime runtime{};
        runtime.userData = request.userData;
        return MaterializeShader(request.request, *CaptureResources(request.request, runtime, *handle), *handle);
    };
    const auto budget = ResultMemoSourceBytes();
    const auto first = materialize(0x10000000u);
    require(materialize(0x10000000u) == first, "the result memo did not reuse a result within its budget");
    const auto before = ResultMemoTotals();
    SetResultMemoSourceBytes(1u);
    const auto second = materialize(0x10010000u);
    const auto after = ResultMemoTotals();
    require(after.evictions == before.evictions + 1u && after.evictedBytes - before.evictedBytes == ResultHeapBytes(*first) && after.entries == before.entries, "a source over its result memo budget did not evict exactly its older result");
    const auto again = materialize(0x10000000u);
    require(again != first, "an evicted result was still served from the memo");
    requireSameResult(*first, *again, "a result rematerialized after its memo eviction");
    require(materialize(0x10000000u) == again, "the budget evicted the newest result");
    require(materialize(0x10010000u) != second, "a result evicted under the budget was still served from the memo");
    SetResultMemoSourceBytes(budget);
    bool refused = false;
    try { SetResultMemoSourceBytes(0u); }
    catch (const std::invalid_argument&) { refused = true; }
    require(refused && ResultMemoSourceBytes() == budget, "a zero result memo budget was accepted");
    std::cout << "result memo: per-source byte budget evicts older results, rematerializes them exactly and keeps the newest\n";
}

void verifyBufferAlignmentSpecialization() {
    ComputeRequest request(true);
    request.request.context.waveSize = 32u;
    request.request.context.compute->numThreads = {32u, 1u, 1u};
    request.userData[3] = 0x31027facu;
    static_cast<void>(PrepareShader(request.request));
    const auto misses = diskMisses();
    const auto aligned = Recompile(request.request);
    const auto countAtomics = [](const RecompileResult& result) {
        std::size_t atomics = 0;
        for (std::size_t cursor = 5; cursor < result.spirv.size();) {
            const auto count = result.spirv[cursor] >> 16u;
            require(count != 0u && count <= result.spirv.size() - cursor, "invalid specialized buffer instruction");
            if ((result.spirv[cursor] & 0xffffu) == spv::OpAtomicCompareExchange) ++atomics;
            cursor += count;
        }
        return atomics;
    };
    require(countAtomics(aligned) == 0u, "aligned DWORD store retained unaligned atomic updates");
    for (std::uint32_t offset = 1; offset < 4u; ++offset) {
        request.userData[0] = 0x10000000u + offset;
        const auto unaligned = Recompile(request.request);
        require(unaligned.variantId == aligned.variantId && diskMisses() == misses, "base alignment repeated static shader preparation");
        require(countAtomics(unaligned) == 2u, "unaligned DWORD store did not split into two atomic updates");
        request.userData[0] += 0x10000u;
        const auto relocated = Recompile(request.request);
        require(unaligned.spirv.data() == relocated.spirv.data(), "relocation with unchanged alignment rebuilt the buffer specialization");
    }
    request.userData[0] = 0x10000000u;
    const auto restored = Recompile(request.request);
    require(restored.spirv.data() == aligned.spirv.data(), "unaligned buffers replaced the aligned specialization");
}

struct PresenceRequest : ComputeRequest {
    std::vector<std::uint32_t> presenceCode{0xe0700010u, 0x80000000u, 0xbf800005u, 0xbf810000u};

    explicit PresenceRequest(const std::array<std::uint32_t, 4>& descriptor) : ComputeRequest(true) {
        request.shader.code = presenceCode;
        userData = descriptor;
        request.context.userData = userData;
    }
};

constexpr std::array<std::uint32_t, 4> PresenceBound{0x10000000u, 0x00000000u, 0x40u, 0x00027facu};
constexpr std::array<std::uint32_t, 4> PresenceNull{0x00000000u, 0x00080050u, 0x00000000u, 0x00005204u};

std::vector<std::uint32_t> spirvWords(const RecompileResult& result) {
    require(!result.spirv.empty(), "the result has no SPIR-V");
    return result.spirv.Copy();
}

std::filesystem::path presenceReference() {
    return ShaderDiskCache::EntryDirectory() / "presence-reference.spv";
}

std::size_t substituteRecords() {
    std::size_t records = 0;
    for (const auto& entry : std::filesystem::directory_iterator(ShaderDiskCache::EntryDirectory())) records += entry.path().extension() == ".slots";
    return records;
}

void verifyBufferPresenceSubstitutes(const char* self) {
    const auto recordsBefore = substituteRecords();
    PresenceRequest bound(PresenceBound);
    const auto boundResult = Recompile(bound.request);
    PresenceRequest absent(PresenceNull);
    const auto absentResult = Recompile(absent.request);
    require(absentResult.specializationId == boundResult.specializationId && spirvWords(absentResult) == spirvWords(boundResult), "a null V# did not share the bound slot's specialized module");
    ShaderDiskCache::Flush();
    require(substituteRecords() == recordsBefore + 1u, "the bound slot's substitute was not persisted");
    const auto words = spirvWords(boundResult);
    std::vector<std::byte> bytes(words.size() * sizeof(std::uint32_t));
    std::memcpy(bytes.data(), words.data(), bytes.size());
    require(WriteFileAtomically(presenceReference(), bytes), "cannot write the presence reference SPIR-V");
    const std::string command = "\"" + std::string(self) + "\" --load-presence-substitutes";
    require(std::system(command.c_str()) == 0, "the process loading the persisted substitutes failed");
}

void verifyRawBufferShape() {
    const auto replay = [](const std::array<std::uint32_t, 4>& descriptor) {
        PresenceRequest request(descriptor);
        request.presenceCode[0] = 0xe0700020u;
        request.request.shader.code = request.presenceCode;
        return Recompile(request.request);
    };
    const auto absent = replay(PresenceNull);
    const auto bound = replay(PresenceBound);
    const auto otherFormat = replay({0x10000000u, 0x00000000u, 0x40u, 0x00001204u});
    require(absent.specializationId == bound.specializationId && spirvWords(absent) == spirvWords(bound), "a raw slot's first null V# did not share the bound slot's module");
    require(otherFormat.specializationId == bound.specializationId && spirvWords(otherFormat) == spirvWords(bound), "a raw slot's format and selectors changed its module");
    const auto rawMode = replay({0x10000000u, 0x00000000u, 0x40u, 0x30027facu});
    require(rawMode.specializationId != bound.specializationId && spirvWords(rawMode) != spirvWords(bound), "a raw slot's out-of-range mode did not specialize its module");
    std::cout << "raw buffer slots: a first null V#, other formats and selectors share one module; the out-of-range mode specializes it\n";
}

int runPresenceSubstituteProcess() {
    PresenceRequest absent(PresenceNull);
    const auto result = Recompile(absent.request);
    const auto totals = ShaderDiskCache::Totals();
    require(totals.hits == 1 && totals.misses == 0, "the presence program was not loaded from the disk cache");
    std::vector<std::byte> reference;
    require(ReadWholeFile(presenceReference(), reference), "cannot read the presence reference SPIR-V");
    const auto words = spirvWords(result);
    require(reference.size() == words.size() * sizeof(std::uint32_t) && std::memcmp(reference.data(), words.data(), reference.size()) == 0, "a null V# in a new process did not reproduce the module specialized with the persisted substitute");
    std::cout << "a new process specialized the null V# with the persisted substitute (" << words.size() << " SPIR-V words)\n";
    return 0;
}

void verifyBindingPlanSelection() {
    ShaderInfo info{};
    info.buffers.resize(2);
    info.buffers[0].atomic = true;
    info.buffers[1].read = true;
    CompiledBindingLayout compiled{};
    compiled.layout.descriptors = {{DescriptorBindingKind::Buffers, {0, 1}}, {DescriptorBindingKind::FlattenedSrt, {}}, {DescriptorBindingKind::ShaderData, {}}, {DescriptorBindingKind::Gds, {}}};
    compiled.layout.userDataRegisters = {0};
    compiled.layout.memoryOffsetDword = 1;
    ResourceSnapshot snapshot{};
    snapshot.buffers.resize(2);
    for (auto& buffer : snapshot.buffers) {
        buffer.dwordCount = 4u;
        buffer.dwords = {0x10000000u, 0u, 64u, 0x31016facu};
    }
    snapshot.userData = {17u};
    snapshot.flattenedSrt = {23u};
    const DescriptorBindingBuilder builder;
    const auto plan = builder.Prepare(compiled.layout, info, IrShaderStage::Compute, snapshot);
    BindingAllocationResult full;
    builder.Populate(full, compiled, plan, 0u, snapshot, {});
    require(full.bindings.size() == 4u, "binding plan lost a descriptor before module selection");
    require(full.bindings[0].bufferAtomic == std::vector<bool>{true, false} && full.bindings[0].bufferWritten == std::vector<bool>{true, false} && full.bindings[0].bufferRead == std::vector<bool>{false, true}, "binding plan lost buffer access metadata");
    const std::array liveBindings{full.bindings[0].binding, full.bindings[0].binding, full.bindings[3].binding};
    const auto selected = builder.Select(plan, liveBindings);
    snapshot.userData.clear();
    snapshot.flattenedSrt.clear();
    snapshot.buffers[0].dwords[0] += 0x10000u;
    snapshot.buffers[1].dwords[2] = 128u;
    BindingAllocationResult populated;
    builder.Populate(populated, compiled, selected, 0u, snapshot, {});
    require(populated.bindings.size() == 2u && populated.pushConstants.empty(), "module selection retained an unused descriptor");
    auto expected = full.bindings[0];
    expected.guestDescriptor[0] += 0x10000u;
    expected.guestDescriptor[6] = 128u;
    require(sameBinding(expected, populated.bindings[0]) && sameBinding(full.bindings[3], populated.bindings[1]), "selected plan changed binding order or reused stale buffer data");
    bool rejected = false;
    try { static_cast<void>(builder.Select(plan, std::array{UINT32_MAX})); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "module selection accepted an unknown binding");
    snapshot.buffers[0].dwordCount = 3u;
    rejected = false;
    try { builder.Populate(populated, compiled, selected, 0u, snapshot, {}); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "selected plan accepted an invalid live buffer descriptor");
    snapshot.buffers[0].dwordCount = 4u;
    const auto rejectsAlignment = [&](const char* diagnostic) {
        try { static_cast<void>(builder.Prepare(compiled.layout, info, IrShaderStage::Compute, snapshot)); }
        catch (const std::runtime_error& error) {
            require(std::string_view(error.what()).find(diagnostic) != std::string_view::npos, "alignment check returned an unrelated failure");
            return;
        }
        throw std::runtime_error("binding plan accepted an unsupported buffer alignment");
    };
    snapshot.buffers[0].dwords[0] |= 2u;
    const auto atomicPlan = builder.Prepare(compiled.layout, info, IrShaderStage::Compute, snapshot);
    require(std::ranges::any_of(atomicPlan.specialization, [](const PipelineSpecializationConstant& constant) { return constant.id == PipelineSpecialization::BufferBase + 3u && constant.value == 2u; }), "a buffer atomic V# off a DWORD boundary did not record its misalignment");
    snapshot.buffers[0].dwords[0] &= ~3u;
    info.buffers[1].typedAlignment = 2u;
    snapshot.buffers[1].dwords[0] |= 2u;
    static_cast<void>(builder.Prepare(compiled.layout, info, IrShaderStage::Compute, snapshot));
    snapshot.buffers[1].dwords[0] |= 1u;
    rejectsAlignment("base is not aligned to its element");
    info.buffers[1].typedAlignment = 1u;
    info.buffers[1].descriptorFormatted = true;
    snapshot.buffers[1].dwords[3] = 0x3100bfacu;
    rejectsAlignment("base is not aligned to its element");
    snapshot.buffers[1].dwords[0] &= ~1u;
    static_cast<void>(builder.Prepare(compiled.layout, info, IrShaderStage::Compute, snapshot));
    snapshot.buffers[1].dwords[3] = 0x31014facu;
    rejectsAlignment("base is not aligned to its element");
}

void verifyVertexTypeSpecialization() {
    CompiledShaderArtifact artifact;
    artifact.vertexInputs = {{0, 4, 0, 1}, {1, 4, 0, 1}};
    artifact.spirv = std::vector<std::uint32_t>{
        spv::MagicNumber, 0x00010000u, 0, 28, 0,
        (2u << 16u) | spv::OpCapability, spv::CapabilityShader,
        (3u << 16u) | spv::OpMemoryModel, spv::AddressingModelLogical, spv::MemoryModelGLSL450,
        (7u << 16u) | spv::OpEntryPoint, spv::ExecutionModelVertex, 20, 0x6e69616du, 0, 10, 11,
        (4u << 16u) | spv::OpDecorate, 10, spv::DecorationLocation, 0,
        (4u << 16u) | spv::OpDecorate, 11, spv::DecorationLocation, 1,
        (2u << 16u) | spv::OpTypeVoid, 1,
        (3u << 16u) | spv::OpTypeFloat, 2, 32,
        (4u << 16u) | spv::OpTypeInt, 3, 32, 0,
        (4u << 16u) | spv::OpTypeVector, 4, 2, 4,
        (4u << 16u) | spv::OpTypePointer, 5, spv::StorageClassInput, 4,
        (4u << 16u) | spv::OpTypePointer, 6, spv::StorageClassInput, 2,
        (3u << 16u) | spv::OpTypeFunction, 7, 1,
        (4u << 16u) | spv::OpConstant, 3, 8, 0,
        (4u << 16u) | spv::OpVariable, 5, 10, spv::StorageClassInput,
        (4u << 16u) | spv::OpVariable, 5, 11, spv::StorageClassInput,
        (5u << 16u) | spv::OpFunction, 1, 20, spv::FunctionControlMaskNone, 7,
        (2u << 16u) | spv::OpLabel, 21,
        (5u << 16u) | spv::OpAccessChain, 6, 22, 10, 8,
        (4u << 16u) | spv::OpLoad, 2, 23, 22,
        (4u << 16u) | spv::OpBitcast, 3, 24, 23,
        (5u << 16u) | spv::OpAccessChain, 6, 25, 11, 8,
        (4u << 16u) | spv::OpLoad, 2, 26, 25,
        (4u << 16u) | spv::OpBitcast, 3, 27, 26,
        (1u << 16u) | spv::OpReturn,
        (1u << 16u) | spv::OpFunctionEnd
    };
    PrepareVertexInputSpecialization(artifact);
    const auto original = artifact.spirv.Words();
    for (std::uint32_t first = 0; first < 3u; ++first) {
        for (std::uint32_t second = 0; second < 3u; ++second) {
            const std::array classes{first, second};
            const auto specialized = SpecializeVertexInputTypes(artifact, classes);
            const auto& words = specialized.Words();
            std::map<std::uint32_t, std::uint32_t> scalarKinds;
            std::uint32_t checked = 0;
            for (std::size_t cursor = 5; cursor < words.size(); cursor += words[cursor] >> 16u) {
                const auto op = static_cast<spv::Op>(words[cursor] & 0xffffu);
                if (op == spv::OpTypeFloat) scalarKinds.emplace(words[cursor + 1u], 0u);
                if (op == spv::OpTypeInt) scalarKinds.emplace(words[cursor + 1u], words[cursor + 3u] != 0u ? 1u : 2u);
                if (op == spv::OpLoad) {
                    const auto attribute = words[cursor + 2u] == 23u ? 0u : 1u;
                    require(scalarKinds.at(words[cursor + 1u]) == classes[attribute], "vertex input loaded the wrong numeric type");
                    ++checked;
                }
                if (op == spv::OpBitcast || op == spv::OpCopyObject) {
                    const auto attribute = words[cursor + 2u] == 24u ? 0u : 1u;
                    require(op == (classes[attribute] == 2u ? spv::OpCopyObject : spv::OpBitcast), "vertex input did not preserve raw component bits");
                    ++checked;
                }
            }
            require(checked == 4u, "vertex specialization lost attribute loads");
            require(std::ranges::equal(artifact.spirv.Words(), original), "vertex specialization mutated the shared template");
        }
    }
    auto invalid = artifact;
    invalid.vertexInputPatches.front().word = static_cast<std::uint32_t>(original.size());
    bool rejected = false;
    try { static_cast<void>(SpecializeVertexInputTypes(invalid, std::array{0u, 0u})); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "out-of-range vertex type patch was accepted");
}

void verifyBuiltinSpecialization() {
    std::vector<std::uint32_t> words{spv::MagicNumber, 0x00010300u, 0u, 40u, 0u};
    const auto emit = [&](spv::Op op, std::initializer_list<std::uint32_t> operands) {
        words.push_back((static_cast<std::uint32_t>(operands.size() + 1u) << 16u) | op);
        words.insert(words.end(), operands);
    };
    emit(spv::OpCapability, {spv::CapabilityShader});
    emit(spv::OpMemoryModel, {spv::AddressingModelLogical, spv::MemoryModelGLSL450});
    emit(spv::OpEntryPoint, {spv::ExecutionModelGLCompute, 10u, 0x6e69616du, 0u});
    emit(spv::OpExecutionMode, {10u, spv::ExecutionModeLocalSize, 1u, 1u, 1u});
    emit(spv::OpTypeVoid, {1u});
    emit(spv::OpTypeBool, {2u});
    emit(spv::OpTypeInt, {3u, 32u, 0u});
    emit(spv::OpTypeVector, {4u, 3u, 4u});
    emit(spv::OpTypeFunction, {5u, 1u});
    emit(spv::OpConstant, {3u, 6u, 0u});
    emit(spv::OpConstant, {3u, 7u, 1u});
    emit(spv::OpConstant, {3u, 8u, 2u});
    emit(spv::OpConstant, {3u, 9u, 3u});
    emit(spv::OpConstantComposite, {4u, 30u, 6u, 7u, 8u, 9u});
    emit(spv::OpTypePointer, {32u, spv::StorageClassPrivate, 4u});
    emit(spv::OpVariable, {32u, 31u, spv::StorageClassPrivate});
    emit(spv::OpTypePointer, {33u, spv::StorageClassPrivate, 3u});
    emit(spv::OpVariable, {33u, 34u, spv::StorageClassPrivate});
    emit(spv::OpFunction, {1u, 10u, spv::FunctionControlMaskNone, 5u});
    emit(spv::OpLabel, {11u});
    emit(spv::OpBitFieldUExtract, {3u, 12u, 9u, 7u, 7u});
    emit(spv::OpSelectionMerge, {16u, spv::SelectionControlMaskNone});
    emit(spv::OpSwitch, {12u, 15u, 1u, 14u});
    emit(spv::OpLabel, {14u});
    emit(spv::OpBranch, {16u});
    emit(spv::OpLabel, {15u});
    emit(spv::OpIAdd, {3u, 17u, 8u, 9u});
    emit(spv::OpBranch, {16u});
    emit(spv::OpLabel, {16u});
    emit(spv::OpPhi, {3u, 18u, 7u, 14u, 17u, 15u});
    emit(spv::OpStore, {34u, 18u});
    emit(spv::OpIEqual, {2u, 19u, 18u, 7u});
    emit(spv::OpSelectionMerge, {22u, spv::SelectionControlMaskNone});
    emit(spv::OpBranchConditional, {19u, 20u, 21u});
    emit(spv::OpLabel, {20u});
    emit(spv::OpVectorExtractDynamic, {3u, 23u, 30u, 6u});
    emit(spv::OpVectorExtractDynamic, {3u, 24u, 30u, 7u});
    emit(spv::OpVectorExtractDynamic, {3u, 25u, 30u, 8u});
    emit(spv::OpVectorExtractDynamic, {3u, 26u, 30u, 9u});
    emit(spv::OpCompositeConstruct, {4u, 27u, 23u, 24u, 25u, 26u});
    emit(spv::OpStore, {31u, 27u});
    emit(spv::OpBranch, {22u});
    emit(spv::OpLabel, {21u});
    emit(spv::OpUnreachable, {});
    emit(spv::OpLabel, {22u});
    emit(spv::OpReturn, {});
    emit(spv::OpFunctionEnd, {});
    const auto specialized = SpecializeSpirv(words);
    bool identity = false;
    std::uint32_t phiValue = 0u;
    for (std::size_t cursor = 5; cursor < specialized.size();) {
        const auto count = specialized[cursor] >> 16u;
        require(count != 0u && count <= specialized.size() - cursor, "builtin specialization produced a truncated instruction");
        const auto op = static_cast<spv::Op>(specialized[cursor] & 0xffffu);
        require(op != spv::OpSwitch && op != spv::OpBranchConditional && op != spv::OpPhi && op != spv::OpVectorExtractDynamic, "builtin specialization retained constant control flow or dynamic exports");
        if (op == spv::OpStore && specialized[cursor + 1u] == 31u) identity = specialized[cursor + 2u] == 30u;
        if (op == spv::OpStore && specialized[cursor + 1u] == 34u) phiValue = specialized[cursor + 2u];
        cursor += count;
    }
    bool correctPhi = false;
    for (std::size_t cursor = 5; cursor < specialized.size(); cursor += specialized[cursor] >> 16u) {
        if ((specialized[cursor] & 0xffffu) == spv::OpConstant && specialized[cursor + 2u] == phiValue) correctPhi = specialized[cursor + 3u] == 1u;
    }
    require(identity && correctPhi, "builtin specialization selected the wrong export or phi value");
    require(SpecializeSpirv(specialized) == specialized, "builtin specialization is not stable");
}

void verifySpecializationLiveness() {
    std::vector<std::uint32_t> words{spv::MagicNumber, 0x00010300u, 0u, 64u, 0u};
    const auto emit = [&](spv::Op op, std::initializer_list<std::uint32_t> operands) {
        words.push_back((static_cast<std::uint32_t>(operands.size() + 1u) << 16u) | op);
        words.insert(words.end(), operands);
    };
    emit(spv::OpCapability, {spv::CapabilityShader});
    emit(spv::OpMemoryModel, {spv::AddressingModelLogical, spv::MemoryModelGLSL450});
    emit(spv::OpEntryPoint, {spv::ExecutionModelGLCompute, 10u, 0x6e69616du, 0u});
    emit(spv::OpExecutionMode, {10u, spv::ExecutionModeLocalSize, 1u, 1u, 1u});
    emit(spv::OpDecorate, {27u, spv::DecorationRelaxedPrecision});
    emit(spv::OpTypeVoid, {1u});
    emit(spv::OpTypeInt, {3u, 32u, 0u});
    emit(spv::OpTypePointer, {4u, spv::StorageClassPrivate, 3u});
    emit(spv::OpTypeFunction, {5u, 1u});
    emit(spv::OpConstant, {3u, 6u, 0u});
    emit(spv::OpConstant, {3u, 7u, 1u});
    emit(spv::OpTypeStruct, {24u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u});
    emit(spv::OpConstantComposite, {24u, 25u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u});
    emit(spv::OpVariable, {4u, 8u, spv::StorageClassPrivate, 6u});
    emit(spv::OpFunction, {1u, 10u, spv::FunctionControlMaskNone, 5u});
    emit(spv::OpLabel, {11u});
    emit(spv::OpLoad, {3u, 12u, 8u});
    emit(spv::OpIMul, {3u, 13u, 12u, 7u});
    emit(spv::OpIAdd, {3u, 14u, 13u, 7u});
    emit(spv::OpCopyObject, {3u, 15u, 12u});
    emit(spv::OpCopyObject, {3u, 16u, 15u});
    emit(spv::OpIAdd, {3u, 17u, 16u, 7u});
    emit(spv::OpStore, {8u, 17u});
    emit(spv::OpLoad, {3u, 18u, 8u, spv::MemoryAccessVolatileMask});
    emit(spv::OpFunctionCall, {1u, 20u, 30u});
    emit(spv::OpCompositeExtract, {3u, 26u, 25u, 15u});
    emit(spv::OpStore, {8u, 26u});
    emit(spv::OpCopyObject, {3u, 27u, 12u});
    emit(spv::OpStore, {8u, 27u});
    emit(spv::OpLoad, {3u, 28u, 8u});
    emit(spv::OpReturn, {});
    emit(spv::OpFunctionEnd, {});
    emit(spv::OpFunction, {1u, 30u, spv::FunctionControlMaskNone, 5u});
    emit(spv::OpLabel, {31u});
    emit(spv::OpReturn, {});
    emit(spv::OpFunctionEnd, {});
    const auto specialized = SpecializeSpirv(words);
    std::map<std::uint32_t, std::vector<std::uint32_t>> results;
    std::size_t stores = 0;
    for (std::size_t cursor = 5; cursor < specialized.size();) {
        const auto count = specialized[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(specialized[cursor] & 0xffffu);
        require(count != 0u && count <= specialized.size() - cursor, "dead computation removal truncated an instruction");
        if (op == spv::OpLoad || op == spv::OpCopyObject || op == spv::OpIAdd || op == spv::OpIMul || op == spv::OpCompositeExtract || op == spv::OpFunctionCall) results.emplace(specialized[cursor + 2u], std::vector<std::uint32_t>(specialized.begin() + cursor, specialized.begin() + cursor + count));
        if (op == spv::OpStore) ++stores;
        cursor += count;
    }
    for (const auto id : {13u, 14u, 15u, 16u, 28u}) require(!results.contains(id), "specialization retained a dead computation or copy chain");
    require(results.at(17u).at(3) == 12u, "copy propagation lost the live source");
    require(results.at(26u).at(4) == 15u, "copy propagation rewrote a literal as an ID");
    require(results.contains(18u) && results.contains(20u) && results.contains(27u) && stores == 3u, "specialization removed an effect or decorated copy");
    require(SpecializeSpirv(specialized) == specialized, "dead computation removal is not stable");
}

void verifySpecializationFunctionLiveness() {
    std::vector<std::uint32_t> words{spv::MagicNumber, 0x00010300u, 0u, 64u, 0u};
    const auto emit = [&](spv::Op op, std::initializer_list<std::uint32_t> operands) {
        words.push_back((static_cast<std::uint32_t>(operands.size() + 1u) << 16u) | op);
        words.insert(words.end(), operands);
    };
    emit(spv::OpCapability, {spv::CapabilityShader});
    emit(spv::OpMemoryModel, {spv::AddressingModelLogical, spv::MemoryModelGLSL450});
    emit(spv::OpEntryPoint, {spv::ExecutionModelGLCompute, 10u, 0x6e69616du, 0u});
    emit(spv::OpEntryPoint, {spv::ExecutionModelGLCompute, 40u, 0x00787561u});
    for (const auto entry : {10u, 40u}) emit(spv::OpExecutionMode, {entry, spv::ExecutionModeLocalSize, 1u, 1u, 1u});
    emit(spv::OpName, {20u, 0x64616564u, 0u});
    emit(spv::OpName, {22u, 0x64616f6cu, 0u});
    emit(spv::OpDecorate, {22u, spv::DecorationRelaxedPrecision});
    emit(spv::OpTypeVoid, {1u});
    emit(spv::OpTypeBool, {2u});
    emit(spv::OpTypeInt, {3u, 32u, 0u});
    emit(spv::OpTypePointer, {4u, spv::StorageClassPrivate, 3u});
    emit(spv::OpTypeFunction, {5u, 1u});
    emit(spv::OpConstantFalse, {2u, 6u});
    emit(spv::OpConstant, {3u, 7u, 0u});
    emit(spv::OpVariable, {4u, 8u, spv::StorageClassPrivate, 7u});
    emit(spv::OpFunction, {1u, 10u, spv::FunctionControlMaskNone, 5u});
    emit(spv::OpLabel, {11u});
    emit(spv::OpSelectionMerge, {14u, spv::SelectionControlMaskNone});
    emit(spv::OpBranchConditional, {6u, 12u, 13u});
    emit(spv::OpLabel, {12u});
    emit(spv::OpFunctionCall, {1u, 15u, 20u});
    emit(spv::OpBranch, {14u});
    emit(spv::OpLabel, {13u});
    emit(spv::OpFunctionCall, {1u, 16u, 30u});
    emit(spv::OpBranch, {14u});
    emit(spv::OpLabel, {14u});
    emit(spv::OpReturn, {});
    emit(spv::OpFunctionEnd, {});
    emit(spv::OpFunction, {1u, 20u, spv::FunctionControlMaskNone, 5u});
    emit(spv::OpLabel, {21u});
    emit(spv::OpLoad, {3u, 22u, 8u});
    emit(spv::OpStore, {8u, 22u});
    emit(spv::OpFunctionCall, {1u, 23u, 50u});
    emit(spv::OpReturn, {});
    emit(spv::OpFunctionEnd, {});
    emit(spv::OpFunction, {1u, 30u, spv::FunctionControlMaskNone, 5u});
    emit(spv::OpLabel, {31u});
    emit(spv::OpFunctionCall, {1u, 32u, 60u});
    emit(spv::OpReturn, {});
    emit(spv::OpFunctionEnd, {});
    for (const auto function : {40u, 50u, 60u}) {
        emit(spv::OpFunction, {1u, function, spv::FunctionControlMaskNone, 5u});
        emit(spv::OpLabel, {function + 1u});
        emit(spv::OpStore, {8u, 7u});
        emit(spv::OpReturn, {});
        emit(spv::OpFunctionEnd, {});
    }
    const auto functions = [](const std::vector<std::uint32_t>& module) {
        std::vector<std::uint32_t> result;
        for (std::size_t cursor = 5; cursor < module.size(); cursor += module[cursor] >> 16u) {
            if ((module[cursor] & 0xffffu) == spv::OpFunction) result.push_back(module[cursor + 2u]);
        }
        return result;
    };
    const auto specialized = SpecializeSpirv(words);
    require(functions(specialized) == std::vector<std::uint32_t>{10u, 30u, 40u, 60u}, "specialization retained an uncalled helper or removed a transitive call or entry point");
    std::size_t calls = 0;
    std::size_t stores = 0;
    for (std::size_t cursor = 5; cursor < specialized.size(); cursor += specialized[cursor] >> 16u) {
        const auto op = static_cast<spv::Op>(specialized[cursor] & 0xffffu);
        if (op == spv::OpFunctionCall) {
            require(specialized[cursor + 3u] == 30u || specialized[cursor + 3u] == 60u, "specialization retained a call to a removed function");
            ++calls;
        }
        if (op == spv::OpStore) ++stores;
        if (op == spv::OpName || op == spv::OpDecorate) require(specialized[cursor + 1u] != 20u && specialized[cursor + 1u] != 22u, "specialization retained metadata for a removed function or its results");
    }
    require(calls == 2u && stores == 2u, "specialization lost a reachable helper effect or retained a dead one");
    require(SpecializeSpirv(specialized) == specialized, "function removal is not stable");
    std::vector<std::uint32_t> library(words.begin(), words.begin() + 5);
    for (std::size_t cursor = 5; cursor < words.size(); cursor += words[cursor] >> 16u) {
        const auto count = words[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(words[cursor] & 0xffffu);
        if (op != spv::OpEntryPoint && op != spv::OpExecutionMode) library.insert(library.end(), words.begin() + cursor, words.begin() + cursor + count);
    }
    require(functions(SpecializeSpirv(library)) == functions(words), "specialization removed functions from a module without entry points");
    auto decorated = words;
    for (std::size_t cursor = 5; cursor < decorated.size(); cursor += decorated[cursor] >> 16u) {
        if ((decorated[cursor] & 0xffffu) != spv::OpTypeVoid) continue;
        const std::array<std::uint32_t, 8> group{
            (2u << 16u) | spv::OpDecorationGroup, 63u,
            (3u << 16u) | spv::OpDecorate, 63u, spv::DecorationRelaxedPrecision,
            (3u << 16u) | spv::OpGroupDecorate, 63u, 22u};
        decorated.insert(decorated.begin() + cursor, group.begin(), group.end());
        break;
    }
    require(functions(SpecializeSpirv(decorated)) == functions(words), "specialization removed a helper referenced by unsupported grouped metadata");
}

constexpr std::uint64_t PlanSrtAddress = 0x7f0000010000ull;
constexpr std::uint64_t PlanTableAddress = 0x7f0000020000ull;
constexpr std::uint64_t PlanMaterialAddress = 0x7f0000030000ull;
constexpr std::uint64_t PlanPayloadAddress = 0x7f0000040000ull;
constexpr std::uint64_t PlanPointerAddress = 0x7f0000048000ull;
constexpr std::uint64_t PlanOutputAddress = 0x7f0000050000ull;
constexpr std::uint64_t PlanTextureAddress = 0x7f0000060000ull;
constexpr std::uint64_t PlanUnmappedAddress = 0x7f00000f0000ull;

struct PlanFixture {
    std::string name;
    ShaderStage stage = ShaderStage::Compute;
    std::uint64_t codeAddress = 0;
    std::uint32_t waveSize = 64;
    std::vector<std::uint32_t> code;
    std::vector<std::uint32_t> userData;
    std::vector<std::pair<std::uint64_t, std::vector<std::uint32_t>>> regions;
    std::optional<ShaderPixelStageInfo> pixel;
    std::vector<MemoryRegion> memory;

    RecompileRequest Request(bool useCache) {
        static constexpr std::array<std::uint32_t, 7> capabilities{spv::CapabilityShader, spv::CapabilityImageGatherExtended, spv::CapabilitySampledImageArrayDynamicIndexing, spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityClipDistance};
        static constexpr std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
        memory.clear();
        for (const auto& [address, words] : regions) memory.push_back({address, std::as_bytes(std::span(words))});
        RecompileRequest request{};
        request.shader = {stage, codeAddress, code, 0, {}};
        request.context.waveSize = waveSize;
        request.context.userDataBaseRegister = 0;
        request.context.userData = userData;
        if (stage == ShaderStage::Compute) request.context.compute = ShaderComputeStageInfo{{waveSize, 1u, 1u}, 0u, {false, false, false}, false, 1u};
        request.context.pixel = pixel;
        request.context.memory = memory;
        request.target.vulkanVersion = 0x00401000u;
        request.target.spirvVersion = 0x00010300u;
        request.target.subgroupSize = 32;
        request.target.supportedCapabilities = capabilities;
        request.target.supportedExtensions = extensions;
        request.target.bdaAbiVersion = 1u;
        request.layout.pushConstantSizeBytes = 128;
        request.useCache = useCache;
        return request;
    }
};

std::vector<std::uint32_t> imageWords(std::uint64_t base) {
    return {static_cast<std::uint32_t>(base >> 8u), static_cast<std::uint32_t>((base >> 40u) & 0xffu) | (56u << 20u) | (3u << 30u), 3u << 14u, 0xfacu | (9u << 28u), 0u, 0u, 0u, 0u};
}

std::vector<std::uint32_t> bufferWords(std::uint64_t base, std::uint32_t stride, std::uint32_t records, std::uint32_t format = 0xfacu) {
    return {static_cast<std::uint32_t>(base), static_cast<std::uint32_t>((base >> 32u) & 0xffffu) | (stride << 16u), records, format};
}

void place(std::vector<std::uint32_t>& words, std::size_t offset, const std::vector<std::uint32_t>& values) {
    std::copy(values.begin(), values.end(), words.begin() + static_cast<std::ptrdiff_t>(offset));
}

std::vector<PlanFixture> planFixtures() {
    std::vector<PlanFixture> fixtures;
    const std::vector<std::uint32_t> srtPointer{static_cast<std::uint32_t>(PlanSrtAddress), static_cast<std::uint32_t>(PlanSrtAddress >> 32u)};
    const auto add = [&](std::string name, std::uint32_t waveSize, std::vector<std::uint32_t> code, std::vector<std::uint32_t> userData, std::vector<std::pair<std::uint64_t, std::vector<std::uint32_t>>> regions) {
        PlanFixture fixture;
        fixture.name = std::move(name);
        fixture.codeAddress = 0x100000u + static_cast<std::uint64_t>(fixtures.size()) * 0x1000u;
        fixture.waveSize = waveSize;
        fixture.code = std::move(code);
        fixture.userData = std::move(userData);
        fixture.regions = std::move(regions);
        fixtures.push_back(std::move(fixture));
    };
    add("user data buffer", 64, {0xe0700000u, 0x80000000u, 0xbf810000u}, bufferWords(PlanOutputAddress, 0u, 0x40u, 0x00027facu), {});

    std::vector<std::uint32_t> guarded(32, 0u);
    place(guarded, 0, imageWords(PlanTextureAddress));
    place(guarded, 16, bufferWords(PlanOutputAddress, 0u, 16u));
    guarded[20] = 1u;
    add("guarded image", 32, {0xf4000400u, 0xfa000050u, 0xf4080500u, 0xfa000020u, 0xf4080600u, 0xfa000040u, 0xbf8cc07fu, 0xbf068010u, 0xbf850006u, 0xf40c0200u, 0xfa000000u, 0xbf8cc07fu, 0xf09c0f08u, 0x00a20000u, 0xbf8c3f70u, 0xe0700000u, 0x80060000u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, guarded}});
    add("guarded image stores", 32, {0xf4000400u, 0xfa000050u, 0xbf8cc07fu, 0xbf068010u, 0xbf850006u, 0xf40c0100u, 0xfa000000u, 0xbf8cc07fu, 0xf0200f08u, 0x00010004u, 0xbf820005u, 0xf40c0100u, 0xfa000000u, 0xbf8cc07fu, 0xf0200f10u, 0x00010004u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, guarded}});

    std::vector<std::uint32_t> heap(32, 0u);
    place(heap, 0, imageWords(PlanTextureAddress));
    place(heap, 8, imageWords(PlanTextureAddress + 0x100u));
    place(heap, 24, imageWords(PlanTextureAddress));
    const std::vector<std::uint32_t> materials{0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 3u, 0u, 0u};
    std::vector<std::uint32_t> tables(16, 0u);
    place(tables, 0, bufferWords(PlanTableAddress, 32u, 4u));
    place(tables, 8, bufferWords(PlanMaterialAddress, 16u, 3u));
    place(tables, 12, bufferWords(PlanOutputAddress, 0u, 16u));
    add("bindless material table", 64, {0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080300u, 0xfa000020u, 0xf4080700u, 0xfa000030u, 0x7e200500u, 0x93109010u, 0xf4200406u, 0x20000004u, 0x8f108510u, 0xf42c0502u, 0x20000000u, 0xf09c0f08u, 0x00450000u, 0xe0700000u, 0x80070000u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, tables}, {PlanTableAddress, heap}, {PlanMaterialAddress, materials}});
    add("bindless whole table", 64, {0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080300u, 0xfa000020u, 0xf4080700u, 0xfa000030u, 0x7e200500u, 0x8f108510u, 0xf42c0502u, 0x20000000u, 0xf09c0f08u, 0x00450000u, 0xe0700000u, 0x80070000u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, tables}, {PlanTableAddress, heap}, {PlanMaterialAddress, materials}});
    add("bindless sample in a loop latch", 64, {0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080300u, 0xfa000020u, 0xf4080700u, 0xfa000030u, 0x7e200500u, 0x8f108510u, 0xf42c0502u, 0x20000000u, 0xbea80380u, 0xf09c0f08u, 0x00450000u, 0x80288128u, 0xbf0a8428u, 0xbf85fffbu, 0xe0700000u, 0x80070000u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, tables}, {PlanTableAddress, heap}, {PlanMaterialAddress, materials}});
    auto keyed = tables;
    keyed.resize(20u, 0u);
    keyed[16] = 3u;
    add("bindless selected record", 64, {0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080300u, 0xfa000020u, 0xf4080700u, 0xfa000030u, 0xf4000400u, 0xfa000040u, 0x8f108510u, 0xf42c0502u, 0x20000000u, 0xf09c0f08u, 0x00450000u, 0xe0700000u, 0x80070000u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, keyed}, {PlanTableAddress, heap}, {PlanMaterialAddress, materials}});

    std::vector<std::uint32_t> phis(32, 0u);
    place(phis, 0, imageWords(PlanTextureAddress));
    place(phis, 12, {0x49u, 0u, 0x00500000u, 0u});
    place(phis, 16, bufferWords(PlanOutputAddress, 0u, 16u));
    phis[20] = 1u;
    place(phis, 24, imageWords(PlanTextureAddress + 0x100u));
    add("descriptor phi sampler", 32, {0xf40c0200u, 0xfa000000u, 0xf4000400u, 0xfa000050u, 0xbf8cc07fu, 0xbf068010u, 0xbf850003u, 0xf4080500u, 0xfa000020u, 0xbf820002u, 0xf4080500u, 0xfa000030u, 0xf4080600u, 0xfa000040u, 0xbf8cc07fu, 0xf09c0f08u, 0x00a20000u, 0xbf8c3f70u, 0xe0700000u, 0x80060000u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, phis}});
    add("descriptor phi image", 32, {0xf4000400u, 0xfa000050u, 0xf4080500u, 0xfa000020u, 0xbf8cc07fu, 0xbf068010u, 0xbf850003u, 0xf40c0200u, 0xfa000000u, 0xbf820002u, 0xf40c0200u, 0xfa000060u, 0xf4080600u, 0xfa000040u, 0xbf8cc07fu, 0xf09c0f08u, 0x00a20000u, 0xbf8c3f70u, 0xe0700000u, 0x80060000u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, phis}});
    add("descriptor phi loop", 32, {0xf4080500u, 0xfa000020u, 0xf4080600u, 0xfa000040u, 0xf40c0200u, 0xfa000000u, 0xbe910380u, 0xbf8cc07fu, 0xf09c0f08u, 0x00a20000u, 0xbf8c3f70u, 0xe0700000u, 0x80060000u, 0xbf068011u, 0xbf850003u, 0xf40c0200u, 0xfa000000u, 0xbf820002u, 0xf40c0200u, 0xfa000060u, 0xbf8cc07fu, 0x80118111u, 0xbf0a8211u, 0xbf85fff0u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, phis}});
    add("descriptor phi in sequence", 32, {0xf4000400u, 0xfa000050u, 0xbf8cc07fu, 0xbf068010u, 0xbf850003u, 0xf40c0200u, 0xfa000000u, 0xbf820002u, 0xf40c0200u, 0xfa000060u, 0xbf068010u, 0xbf850003u, 0xf4080500u, 0xfa000020u, 0xbf820002u, 0xf4080500u, 0xfa000030u, 0xf4080600u, 0xfa000040u, 0xbf8cc07fu, 0xf09c0f08u, 0x00a20000u, 0xbf8c3f70u, 0xe0700000u, 0x80060000u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, phis}});

    std::vector<std::uint32_t> root(16, 0u);
    place(root, 0, bufferWords(PlanOutputAddress, 0u, 256u, 0x30027facu));
    root[4] = 1u;
    root[6] = static_cast<std::uint32_t>(PlanPayloadAddress);
    root[7] = static_cast<std::uint32_t>(PlanPayloadAddress >> 32u);
    const std::vector<std::uint32_t> payload{0x3f800000u};
    add("guarded nested read", 64, {0xf4080100u, 0xfa000000u, 0xf4000200u, 0xfa000010u, 0xf4040280u, 0xfa000018u, 0xbe8c03ffu, 0x11111111u, 0xbf8cc07fu, 0xbf068008u, 0xbf850003u, 0xf4000305u, 0xfa000000u, 0xbf8cc07fu, 0x7e02020cu, 0x34040082u, 0xe0701000u, 0x80010102u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, root}, {PlanPayloadAddress, payload}});
    auto chained = root;
    chained[6] = static_cast<std::uint32_t>(PlanPointerAddress);
    chained[7] = static_cast<std::uint32_t>(PlanPointerAddress >> 32u);
    add("guarded pointer chain", 64, {0xf4080100u, 0xfa000000u, 0xf4000200u, 0xfa000010u, 0xf4040280u, 0xfa000018u, 0xbe8c03ffu, 0x11111111u, 0xbf8cc07fu, 0xbf068008u, 0xbf850006u, 0xf4040385u, 0xfa000000u, 0xbf8cc07fu, 0xf4000307u, 0xfa000000u, 0xbf8cc07fu, 0x7e02020cu, 0x34040082u, 0xe0701000u, 0x80010102u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, chained}, {PlanPayloadAddress, payload}, {PlanPointerAddress, {static_cast<std::uint32_t>(PlanPayloadAddress), static_cast<std::uint32_t>(PlanPayloadAddress >> 32u)}}});
    add("descriptor through a pointer", 64, {0xf4040280u, 0xfa000018u, 0xbf8cc07fu, 0xf4080105u, 0xfa000000u, 0xbf8cc07fu, 0x34040082u, 0xe0701000u, 0x80010002u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, chained}, {PlanPointerAddress, bufferWords(PlanOutputAddress, 0u, 256u, 0x30027facu)}});
    add("descriptor invariant phi", 32, {0xf4000400u, 0xfa000050u, 0xf4080500u, 0xfa000020u, 0xbf8cc07fu, 0xbf068010u, 0xbf850003u, 0xf40c0200u, 0xfa000000u, 0xbf820002u, 0xf40c0200u, 0xfa000000u, 0xf4080600u, 0xfa000040u, 0xbf8cc07fu, 0xf09c0f08u, 0x00a20000u, 0xbf8c3f70u, 0xe0700000u, 0x80060000u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, phis}});

    std::vector<std::uint32_t> palette(0x50u + 257u * 8u, 0u);
    place(palette, 0, bufferWords(PlanMaterialAddress, 0u, 32u));
    place(palette, 4, bufferWords(PlanOutputAddress, 0u, 16u));
    palette[12] = static_cast<std::uint32_t>(PlanSrtAddress + 0x140u);
    palette[13] = static_cast<std::uint32_t>((PlanSrtAddress + 0x140u) >> 32u);
    place(palette, 0x10u, imageWords(PlanTextureAddress));
    place(palette, 0x18u, imageWords(PlanTextureAddress + 0x100u));
    add("pointer image table", 64, {0xf4080200u, 0xfa000000u, 0xf4080700u, 0xfa000010u, 0xf4080800u, 0xfa000020u, 0x7e200500u, 0x8f108210u, 0xf4200444u, 0x20000000u, 0x8711ff11u, 0x000000ffu, 0x8f128511u, 0xf40c0900u, 0x24000040u, 0xf09c0f08u, 0x01090000u, 0xe0700000u, 0x80070000u, 0xbf810000u}, srtPointer, {{PlanSrtAddress, palette}, {PlanMaterialAddress, {0u, 1u, 0u, 1u, 0u, 0u, 0u, 0u}}});

    PlanFixture fragment;
    fragment.name = "fragment parameters";
    fragment.stage = ShaderStage::Fragment;
    fragment.codeAddress = 0x100000u + static_cast<std::uint64_t>(fixtures.size()) * 0x1000u;
    fragment.code = {0xc8100000u, 0xc8110001u, 0xc8140402u, 0xc8150403u, 0xf800180fu, 0x05040504u, 0xbf810000u};
    ShaderPixelStageInfo pixel{};
    pixel.interpolatorCount = 2u;
    pixel.interpolatorSettings[1] = 1u;
    pixel.wave32 = true;
    pixel.inputAddr = PixelInputBit(PixelInput::PerspectiveCenter) | PixelInputBit(PixelInput::LinearCenter);
    pixel.hasPerspectiveCenterVgpr = true;
    pixel.noPerspective = true;
    pixel.targetOutputMode[0] = 9u;
    pixel.targetExportMapping[0] = 0xe4u;
    fragment.pixel = pixel;
    fixtures.push_back(std::move(fragment));
    return fixtures;
}

IrResourcePlan handBuiltPlan() {
    IrResourcePlan plan;
    plan.valueStorage.Reserve(64);
    std::uint32_t ids = 0;
    const auto make = [&](IrOpcode opcode, IrType type) -> IrValue& {
        return *plan.valueStorage.Create(opcode, type, ids++);
    };
    const auto constant = [&](std::uint32_t value) -> IrValue& {
        auto& immediate = make(IrOpcode::Void, IrType::U32);
        immediate.SetImmediateU32(value);
        return immediate;
    };
    auto& resource = make(IrOpcode::GetSrtResource, IrType::SrtResource);
    const auto readConst = [&](std::uint32_t slot) -> IrValue& {
        auto& read = make(IrOpcode::ReadConst, IrType::U32);
        read.AddArgument(&resource);
        read.AddArgument(&constant(slot));
        return read;
    };
    const auto userData = [&](std::uint32_t index) -> IrValue& {
        auto& reg = make(IrOpcode::Void, IrType::ScalarReg);
        reg.SetRegister({RegisterBank::Scalar, index});
        auto& read = make(IrOpcode::GetUserData, IrType::U32);
        read.AddArgument(&reg);
        return read;
    };
    auto& handle = make(IrOpcode::CompositeConstructU64, IrType::U64);
    handle.AddArgument(&userData(0));
    handle.AddArgument(&userData(1));
    auto& pointer = make(IrOpcode::LoadAddressU32, IrType::U32);
    pointer.AddArgument(&handle);
    pointer.AddArgument(&constant(8));
    plan.blockStorage.push_back(std::make_unique<IrBlock>(3u));
    plan.blockStorage.push_back(std::make_unique<IrBlock>(7u));
    auto& phi = make(IrOpcode::Phi, IrType::U32);
    phi.AddPhiOperand(plan.blockStorage[0].get(), &readConst(0));
    phi.AddPhiOperand(plan.blockStorage[1].get(), &constant(0));
    auto& condition = make(IrOpcode::Void, IrType::Bool);
    condition.SetImmediateU64(1u);
    condition.SetFlags(std::uint64_t{0x1234u});
    plan.stage = IrShaderStage::Compute;
    plan.shaderHash = 0x123456789abcull;
    plan.userDataBase = 2u;
    plan.userDataCount = 4u;
    plan.srgbDecodeFormats = 5u;
    MemoryInfo memory;
    memory.kind = ResourceKind::Image;
    memory.resource = 1u;
    memory.dmask = 0xfu;
    memory.imageDimension = RdnaImageDimension::Dim2D;
    memory.gpuDescriptor = true;
    memory.flushDenormals = true;
    MemoryInfo every;
    every.kind = ResourceKind::Buffer;
    every.resource = 2u;
    every.sampler = 3u;
    every.offset = 0x14u;
    every.secondaryOffset = 0x28u;
    every.dmask = 0x5u;
    every.dataDwords = 3u;
    every.dataBits = 16u;
    every.componentIndex = 1u;
    every.componentCount = 2u;
    every.dataFormat = 10u;
    every.numberFormat = 7u;
    every.imageSampleFlags = 0x21u;
    every.imageDimension = RdnaImageDimension::Dim3D;
    every.imageAddressComponents = 3u;
    every.addressIsFull = true;
    every.dataSigned = true;
    every.typed = true;
    every.formatted = true;
    every.d16 = true;
    every.imageHasMip = true;
    every.imageR128 = true;
    every.imagePacked = true;
    every.imageByElements = 2u;
    every.idxen = true;
    every.offen = true;
    every.planningOnly = true;
    every.coherent = true;
    every.gpuDescriptor = true;
    every.flushDenormals = true;
    plan.memoryInfo = {memory, every, MemoryInfo{}};
    DescriptorSource buffer;
    buffer.dwordCount = 4u;
    buffer.dwords = {&readConst(1), &readConst(2), &phi, &pointer};
    DescriptorSource table;
    table.dwordCount = 8u;
    table.dwords[0] = &readConst(3);
    table.tableColumn = TableColumn{2u, 32u, 16u, 4u, 8u, 0xffu, false, true, {KeyDomain{3u, 4u, 16u, false}, KeyDomain{2u, 8u, 4u, true}}, false};
    table.tableColumn->selected = true;
    table.tableColumn->selection = 2u;
    table.tableColumn->pointerHops = {PointerHop{0u, 0x10u}, PointerHop{8u, 0x20u}};
    DescriptorSource heap;
    heap.dwordCount = 4u;
    heap.dwords[1] = &readConst(6);
    DescriptorSource material;
    material.dwordCount = 4u;
    material.dwords[2] = &readConst(7);
    auto& wide = make(IrOpcode::Void, IrType::U64);
    wide.SetImmediateU64(0x8000000000000001ull);
    material.dwords[3] = &wide;
    plan.descriptorSources = {buffer, table, heap, material};
    plan.controlFlow = {{&condition, {1u, 2u}, {0u}, {}}, {nullptr, {2u}, {3u}, {0u}}, {nullptr, {}, {}, {}}};
    plan.materializationSources = {0u, 2u};
    plan.srtReads = {{&readConst(4), 2u}, {&pointer, 1u}};
    plan.guardedSrtSlots = {1u};
    plan.srtGuardOffset = 6u;
    plan.cleanFlatSlots = {0u, 1u};
    plan.pureFlatSlots = {1u, 0u};
    plan.requiresSpecializationMemory = true;
    plan.srtPlanComplete = true;
    plan.resourceTrackingComplete = true;
    plan.info.buffers = {BufferResource{}, BufferResource{2u, 0x40u, 0x100u, 0u, true, true, true, true, true, 0x7u, true, 4u}};
    ImageResource image{};
    image.source = 1u;
    image.resourceClass = ImageResourceClass::Sampled;
    image.dimension = RdnaImageDimension::Dim2D;
    plan.info.images = {image};
    ResourceMaterializer::PrepareImageModes(plan.info);
    plan.uniformFill.fill = {UniformFillKind::Buffer, 1u, {4u, 8u, 12u}, 2u, 0x3f800000u};
    auto& fillValue = readConst(5);
    plan.uniformFill.values = {&fillValue, &fillValue, nullptr, nullptr};
    return plan;
}

std::vector<std::byte> encodedPlan(const IrResourcePlan& plan) {
    std::vector<std::byte> bytes;
    require(ShaderDiskCache::EncodePlan(plan, bytes), "a self-contained plan does not encode");
    return bytes;
}

bool samePlan(const IrResourcePlan& left, const IrResourcePlan& right) {
    if (left.valueStorage.Size() != right.valueStorage.Size() || left.blockStorage.size() != right.blockStorage.size()) return false;
    std::map<const IrValue*, std::size_t> leftValues;
    std::map<const IrValue*, std::size_t> rightValues;
    std::map<const IrBlock*, std::size_t> leftBlocks;
    std::map<const IrBlock*, std::size_t> rightBlocks;
    for (std::size_t index = 0; index < left.valueStorage.Size(); ++index) {
        leftValues.emplace(&left.valueStorage[index], index);
        rightValues.emplace(&right.valueStorage[index], index);
    }
    for (std::size_t index = 0; index < left.blockStorage.size(); ++index) {
        leftBlocks.emplace(left.blockStorage[index].get(), index);
        rightBlocks.emplace(right.blockStorage[index].get(), index);
    }
    const auto sameValue = [&](const IrValue* a, const IrValue* b) {
        if (a == nullptr || b == nullptr) return a == b;
        const auto foundA = leftValues.find(a);
        const auto foundB = rightValues.find(b);
        return foundA != leftValues.end() && foundB != rightValues.end() && foundA->second == foundB->second;
    };
    const auto sameBlock = [&](const IrBlock* a, const IrBlock* b) {
        const auto foundA = leftBlocks.find(a);
        const auto foundB = rightBlocks.find(b);
        return foundA != leftBlocks.end() && foundB != rightBlocks.end() && foundA->second == foundB->second;
    };
    for (std::size_t index = 0; index < left.blockStorage.size(); ++index) {
        if (left.blockStorage[index]->Id() != right.blockStorage[index]->Id()) return false;
    }
    for (std::size_t index = 0; index < left.valueStorage.Size(); ++index) {
        const auto& a = left.valueStorage[index];
        const auto& b = right.valueStorage[index];
        if (a.Opcode() != b.Opcode() || a.Type() != b.Type() || a.Id() != b.Id() || a.Flags<std::uint64_t>() != b.Flags<std::uint64_t>() || !(a.Register() == b.Register()) || a.HasImmediate() != b.HasImmediate() || (a.HasImmediate() && a.ImmediateU64() != b.ImmediateU64())) return false;
        if (a.ArgumentCount() != b.ArgumentCount() || a.IsPhi() != b.IsPhi() || a.PhiBlockCount() != b.PhiBlockCount()) return false;
        for (std::size_t operand = 0; operand < a.ArgumentCount(); ++operand) {
            if (!sameValue(a.Argument(operand), b.Argument(operand))) return false;
        }
        for (std::size_t operand = 0; operand < a.PhiBlockCount(); ++operand) {
            if (!sameBlock(a.PhiBlock(operand), b.PhiBlock(operand))) return false;
        }
    }
    if (left.stage != right.stage || left.shaderHash != right.shaderHash || left.userDataBase != right.userDataBase || left.userDataCount != right.userDataCount || left.srgbDecodeFormats != right.srgbDecodeFormats) return false;
    if (left.memoryInfo != right.memoryInfo || left.descriptorSources.size() != right.descriptorSources.size() || left.controlFlow.size() != right.controlFlow.size() || left.srtReads.size() != right.srtReads.size()) return false;
    for (std::size_t index = 0; index < left.descriptorSources.size(); ++index) {
        const auto& a = left.descriptorSources[index];
        const auto& b = right.descriptorSources[index];
        if (a.dwordCount != b.dwordCount || a.tableColumn != b.tableColumn || (a.tableColumn.has_value() && (a.tableColumn->keyDomains != b.tableColumn->keyDomains || a.tableColumn->unkeyed != b.tableColumn->unkeyed || a.tableColumn->maxKey != b.tableColumn->maxKey))) return false;
        for (std::size_t dword = 0; dword < a.dwords.size(); ++dword) {
            if (!sameValue(a.dwords[dword], b.dwords[dword])) return false;
        }
    }
    for (std::size_t index = 0; index < left.controlFlow.size(); ++index) {
        const auto& a = left.controlFlow[index];
        const auto& b = right.controlFlow[index];
        if (!sameValue(a.condition, b.condition) || a.successors != b.successors || a.sources != b.sources || a.images != b.images) return false;
    }
    for (std::size_t index = 0; index < left.srtReads.size(); ++index) {
        if (!sameValue(left.srtReads[index].value, right.srtReads[index].value) || left.srtReads[index].flatOffset != right.srtReads[index].flatOffset) return false;
    }
    for (std::size_t index = 0; index < left.uniformFill.values.size(); ++index) {
        if (!sameValue(left.uniformFill.values[index], right.uniformFill.values[index])) return false;
    }
    return left.materializationSources == right.materializationSources && left.guardedSrtSlots == right.guardedSrtSlots && left.srtGuardOffset == right.srtGuardOffset && left.cleanFlatSlots == right.cleanFlatSlots && left.pureFlatSlots == right.pureFlatSlots &&
           left.requiresSpecializationMemory == right.requiresSpecializationMemory && left.srtPlanComplete == right.srtPlanComplete && left.resourceTrackingComplete == right.resourceTrackingComplete && left.info == right.info && left.uniformFill.fill == right.uniformFill.fill;
}

struct MaterializedPlan {
    bool ok = false;
    std::string error;
    ResourceSnapshot snapshot;
    SrtReadTrace trace;
    std::size_t leaf = 0;
};

MaterializedPlan materializePlan(const IrResourcePlan& plan, SrtRuntime runtime) {
    MaterializedPlan outcome;
    runtime.readTrace = &outcome.trace;
    try {
        ResourceMaterializer{}.Materialize(plan, runtime, outcome.snapshot);
        outcome.ok = true;
    } catch (const std::exception& error) {
        outcome.error = error.what();
    }
    outcome.leaf = static_cast<std::size_t>(std::find_if(plan.valueStorage.begin(), plan.valueStorage.end(), [&](const auto& value) { return &value == outcome.trace.leaf; }) - plan.valueStorage.begin());
    return outcome;
}

bool sameMaterialization(const MaterializedPlan& left, const MaterializedPlan& right) {
    const auto& a = left.snapshot;
    const auto& b = right.snapshot;
    return left.ok == right.ok && left.error == right.error && a.buffers == b.buffers && a.images == b.images && a.samplers == b.samplers && a.flattenedSrt == b.flattenedSrt && a.userData == b.userData && a.uniformFill == b.uniformFill && a.srtPoison == b.srtPoison && a.nullRootReads == b.nullRootReads && a.tables == b.tables &&
           left.leaf == right.leaf && left.trace.leafSlot == right.trace.leafSlot && left.trace.leaves == right.trace.leaves && left.trace.otherReads == right.trace.otherReads;
}

void mutateWords(std::vector<std::uint32_t>& words, std::mt19937& random) {
    static constexpr std::array<std::uint64_t, 8> addresses{PlanSrtAddress, PlanTableAddress, PlanMaterialAddress, PlanPayloadAddress, PlanPointerAddress, PlanOutputAddress, PlanTextureAddress, PlanUnmappedAddress};
    for (auto& word : words) {
        switch (random() % 16u) {
        case 0: word = static_cast<std::uint32_t>(random()); break;
        case 1: word = 0u; break;
        case 2: {
            const auto address = addresses[random() % addresses.size()] + (random() % 8u) * 4u;
            word = random() % 2u == 0u ? static_cast<std::uint32_t>(address) : static_cast<std::uint32_t>(address >> 32u);
            break;
        }
        case 3: word ^= 1u << (random() % 32u); break;
        default: break;
        }
    }
}

std::size_t verifyRoundTripRuntimes(const IrResourcePlan& plan, const IrResourcePlan& decoded, const PlanFixture& fixture, std::uint32_t seed, const std::string& what) {
    std::mt19937 random(seed);
    std::size_t succeeded = 0;
    for (std::uint32_t run = 0; run < 1000u; ++run) {
        auto userData = fixture.userData;
        auto regions = fixture.regions;
        if (userData.size() < plan.userDataCount) userData.resize(plan.userDataCount, 0u);
        if (run != 0u) {
            mutateWords(userData, random);
            for (auto& region : regions) mutateWords(region.second, random);
        }
        std::vector<MemoryRegion> memory;
        for (const auto& [address, words] : regions) memory.push_back({address, std::as_bytes(std::span(words))});
        RequestMemoryView view(memory);
        const auto runtime = view.MakeRuntime(userData, fixture.codeAddress);
        const auto original = materializePlan(plan, runtime);
        const auto loaded = materializePlan(decoded, runtime);
        require(sameMaterialization(original, loaded), what + ": runtime " + std::to_string(run) + " materialized differently from the decoded plan (" + original.error + " / " + loaded.error + ")");
        if (original.ok) ++succeeded;
    }
    return succeeded;
}

void verifyPlanEntryDamage(const std::vector<std::byte>& key, const IrResourcePlan& plan, const std::string& what) {
    const auto file = ShaderDiskCache::EncodePlanEntry(key, plan);
    require(!file.empty(), what + ": the plan entry was not encoded");
    IrResourcePlan decoded;
    require(ShaderDiskCache::DecodePlanEntry(file, key, decoded) == ShaderDiskCache::LoadStatus::Loaded && encodedPlan(decoded) == encodedPlan(plan), what + ": the plan entry does not decode to the plan");
    for (std::size_t size = 0; size < file.size(); size += size < 512 ? 1 : 37) {
        IrResourcePlan partial;
        require(ShaderDiskCache::DecodePlanEntry(std::span(file).first(size), key, partial) == ShaderDiskCache::LoadStatus::Rejected, what + ": a plan entry truncated to " + std::to_string(size) + " bytes is not rejected");
    }
    for (std::size_t offset = 0; offset < file.size(); offset += offset < 512 ? 1 : 29) {
        for (const auto bit : {std::byte{0x01}, std::byte{0x80}}) {
            auto damaged = file;
            damaged[offset] ^= bit;
            IrResourcePlan partial;
            require(ShaderDiskCache::DecodePlanEntry(damaged, key, partial) == ShaderDiskCache::LoadStatus::Rejected, what + ": a plan entry with byte " + std::to_string(offset) + " flipped is not rejected");
        }
    }
    auto longer = file;
    longer.push_back(std::byte{0});
    require(ShaderDiskCache::DecodePlanEntry(longer, key, decoded) == ShaderDiskCache::LoadStatus::Rejected, what + ": a plan entry with a trailing byte is not rejected");
    auto otherKey = key;
    otherKey.back() ^= std::byte{1};
    require(ShaderDiskCache::DecodePlanEntry(file, otherKey, decoded) == ShaderDiskCache::LoadStatus::KeyMismatch, what + ": a plan entry for another key loads");
    auto previousFormat = file;
    previousFormat[4] = static_cast<std::byte>(ShaderDiskCache::FormatVersion - 1u);
    require(ShaderDiskCache::DecodePlanEntry(previousFormat, key, decoded) == ShaderDiskCache::LoadStatus::Rejected, what + ": a plan entry of the previous format loads");
    auto otherSource = file;
    otherSource[8] ^= std::byte{1};
    require(ShaderDiskCache::DecodePlanEntry(otherSource, key, decoded) == ShaderDiskCache::LoadStatus::Rejected, what + ": a plan entry of another source version loads");
    CompiledVariant variant;
    require(ShaderDiskCache::DecodeEntry(file, key, variant) == ShaderDiskCache::LoadStatus::Rejected, what + ": a plan entry decodes as a compiled variant");
}

void verifyPlanPayloadDamage(const IrResourcePlan& plan, const std::string& what) {
    const auto bytes = encodedPlan(plan);
    for (std::size_t size = 0; size < bytes.size(); ++size) {
        IrResourcePlan partial;
        require(!ShaderDiskCache::DecodePlan(std::span(bytes).first(size), partial), what + ": a plan truncated to " + std::to_string(size) + " bytes decodes");
    }
    std::size_t rejected = 0;
    for (std::size_t offset = 0; offset < bytes.size(); ++offset) {
        for (const auto bit : {std::byte{0x01}, std::byte{0x40}}) {
            auto damaged = bytes;
            damaged[offset] ^= bit;
            IrResourcePlan decoded;
            if (!ShaderDiskCache::DecodePlan(damaged, decoded)) {
                ++rejected;
                continue;
            }
            std::vector<std::byte> again;
            require(ShaderDiskCache::EncodePlan(decoded, again), what + ": a plan accepted with byte " + std::to_string(offset) + " flipped is not self-contained");
        }
    }
    require(rejected != 0u, what + ": no damaged plan was rejected");
}

void verifyMalformedPlans() {
    const auto rejects = [](const std::string& what, const std::function<void(IrResourcePlan&)>& damage) {
        auto plan = handBuiltPlan();
        damage(plan);
        std::vector<std::byte> bytes;
        require(ShaderDiskCache::EncodePlan(plan, bytes), "malformed plan: " + what + " did not encode");
        IrResourcePlan decoded;
        require(!ShaderDiskCache::DecodePlan(bytes, decoded), "malformed plan: " + what + " decodes");
    };
    rejects("a branch with one successor", [](IrResourcePlan& plan) { plan.controlFlow[0].successors.pop_back(); });
    rejects("a successor past the blocks", [](IrResourcePlan& plan) { plan.controlFlow[1].successors[0] = 3u; });
    rejects("a block source past the sources", [](IrResourcePlan& plan) { plan.controlFlow[0].sources[0] = 4u; });
    rejects("a block image past the images", [](IrResourcePlan& plan) { plan.controlFlow[1].images[0] = 1u; });
    rejects("a materialized source past the sources", [](IrResourcePlan& plan) { plan.materializationSources.push_back(4u); });
    rejects("a buffer past the sources", [](IrResourcePlan& plan) { plan.info.buffers[0].source = 4u; });
    rejects("an image past the sources", [](IrResourcePlan& plan) { plan.info.images[0].source = 5u; });
    rejects("a descriptor of nine dwords", [](IrResourcePlan& plan) { plan.descriptorSources[1].dwordCount = 9u; });
    rejects("a uniform fill of five words", [](IrResourcePlan& plan) { plan.uniformFill.fill.words = 5u; });
    rejects("an image table heap past the sources", [](IrResourcePlan& plan) { plan.descriptorSources[1].tableColumn->heapSource = 4u; });
    rejects("an image table key past the sources", [](IrResourcePlan& plan) { plan.descriptorSources[1].tableColumn->keyDomains[1].source = 4u; });
    rejects("an image table selection past the sources", [](IrResourcePlan& plan) { plan.descriptorSources[1].tableColumn->selection = 4u; });
    auto foreign = handBuiltPlan();
    IrValue stray(IrOpcode::Void, IrType::U32, 99u);
    foreign.srtReads[0].value = &stray;
    std::vector<std::byte> bytes;
    require(!ShaderDiskCache::EncodePlan(foreign, bytes), "a plan reading a value it does not own was encoded");
    auto unblocked = handBuiltPlan();
    auto& phi = *unblocked.valueStorage.Create(IrOpcode::Phi, IrType::U32, 100u);
    phi.AddArgument(&unblocked.valueStorage[0]);
    unblocked.srtReads[0].value = &phi;
    bytes.clear();
    require(!ShaderDiskCache::EncodePlan(unblocked, bytes), "a phi without predecessor blocks was encoded");
}

void verifyPlanRoundTrip() {
    const auto handBuilt = handBuiltPlan();
    const auto handBytes = encodedPlan(handBuilt);
    IrResourcePlan handDecoded;
    require(ShaderDiskCache::DecodePlan(handBytes, handDecoded), "the hand-built plan does not decode");
    require(encodedPlan(handDecoded) == handBytes && handDecoded.valueStorage.Size() == handBuilt.valueStorage.Size() && handDecoded.blockStorage.size() == handBuilt.blockStorage.size(), "the hand-built plan changed in a round trip");
    const auto& phi = *handDecoded.descriptorSources[0].dwords[2];
    require(phi.IsPhi() && phi.PhiBlockCount() == 2u && phi.PhiBlock(0)->Id() == 3u && phi.PhiBlock(1)->Id() == 7u && phi.Argument(1)->ImmediateU32() == 0u, "the decoded phi lost its blocks or operands");
    require(handDecoded.uniformFill.values[0] == handDecoded.uniformFill.values[1] && handDecoded.uniformFill.values[0] != nullptr && handDecoded.uniformFill.values[2] == nullptr, "the decoded uniform fill lost its shared or null values");
    require(samePlan(handBuilt, handDecoded), "the decoded hand-built plan differs from the plan in a field or a value");
    PlanFixture handFixture;
    handFixture.userData = {static_cast<std::uint32_t>(PlanSrtAddress), static_cast<std::uint32_t>(PlanSrtAddress >> 32u), 0u, 0u};
    std::vector<std::uint32_t> handSrt(16, 0u);
    place(handSrt, 0, bufferWords(PlanOutputAddress, 0u, 16u));
    handFixture.regions = {{PlanSrtAddress, handSrt}, {PlanPayloadAddress, {0x3f800000u}}};
    static_cast<void>(verifyRoundTripRuntimes(handBuilt, handDecoded, handFixture, 7u, "hand-built plan"));
    verifyPlanPayloadDamage(handBuilt, "hand-built plan");
    verifyMalformedPlans();

    auto fixtures = planFixtures();
    std::vector<std::shared_ptr<const IrResourcePlan>> plans;
    std::size_t index = 0;
    for (auto& fixture : fixtures) {
        auto request = fixture.Request(false);
        const auto plan = GetResourcePlan(request);
        plans.push_back(plan);
        const auto bytes = encodedPlan(*plan);
        IrResourcePlan decoded;
        require(ShaderDiskCache::DecodePlan(bytes, decoded), fixture.name + ": the plan does not decode");
        require(encodedPlan(decoded) == bytes, fixture.name + ": the plan changed in a round trip");
        require(samePlan(*plan, decoded), fixture.name + ": the decoded plan differs from the plan in a field or a value");
        require(decoded.valueStorage.Size() == plan->valueStorage.Size() && decoded.blockStorage.size() == plan->blockStorage.size(), fixture.name + ": the decoded plan holds another graph");
        const auto succeeded = verifyRoundTripRuntimes(*plan, decoded, fixture, 0x5eedu + static_cast<std::uint32_t>(index), fixture.name);
        require(succeeded != 0u, fixture.name + ": no runtime materialized");
        std::vector<std::byte> key;
        ShaderDiskCache::BuildPlanKey(request, 32u, key);
        if (index % 4u == 0u) verifyPlanEntryDamage(key, *plan, fixture.name);
        std::cout << "plan round trip: " << fixture.name << ", " << plan->valueStorage.Size() << " values, " << ShaderDiskCache::EncodePlanEntry(key, *plan).size() << " bytes per entry, " << succeeded << " of 1000 runtimes materialized\n";
        ++index;
    }
    auto releaseFixture = fixtures.front();
    releaseFixture.waveSize = 32u;
    auto registeredRequest = releaseFixture.Request(true);
    const auto registeredHandle = PrepareShader(registeredRequest);
    const auto registeredPlan = GetResourcePlan(registeredRequest);
    ShaderDiskCache::Flush();
    ReleaseUndrawnResourcePlan(*registeredHandle);
    const auto beforeReload = ShaderDiskCache::Totals();
    const auto reloadedPlan = GetResourcePlan(registeredRequest);
    require(ShaderDiskCache::Totals().planHits == beforeReload.planHits + 1u, "a released registration plan was not reloaded from the disk cache");
    require(samePlan(*registeredPlan, *reloadedPlan), "a released registration plan changed when reloaded");

    auto reloadFixture = fixtures.front();
    reloadFixture.waveSize = 32u;
    reloadFixture.code.insert(reloadFixture.code.begin(), 0xbf800004u);
    const auto reloadRequest = reloadFixture.Request(true);
    const auto reloadHandle = PrepareShader(reloadRequest);
    static_cast<void>(GetResourcePlan(reloadRequest));
    ShaderDiskCache::Flush();
    std::vector<std::byte> reloadKey;
    ShaderDiskCache::BuildPlanKey(reloadRequest, 32u, reloadKey);
    const auto reloadPath = ShaderDiskCache::EntryDirectory() / ShaderDiskCache::EntryName(reloadKey);
    require(std::filesystem::is_regular_file(reloadPath), "the reload regression plan was not persisted");
    const auto preservedPath = reloadPath.string() + ".reload-preserved";
    std::filesystem::rename(reloadPath, preservedPath);
    ReleaseUndrawnResourcePlan(*reloadHandle);
    static_cast<void>(GetResourcePlan(reloadRequest));
    auto secondLayout = reloadRequest;
    secondLayout.layout.pushConstantSizeBytes = 0u;
    ShaderPreparationContext preparation;
    static_cast<void>(PrepareShader(secondLayout, &preparation));
    require(preparation.ResourcePrograms() == 1u, "a plan reload retained full IR after an artifact for that layout was already available");

    auto relocatedFixture = fixtures.front();
    relocatedFixture.waveSize = 32u;
    relocatedFixture.code.insert(relocatedFixture.code.begin(), 0xbf800005u);
    const auto relocatedRequest = relocatedFixture.Request(true);
    const auto relocatedHandle = PrepareShader(relocatedRequest);
    static_cast<void>(GetResourcePlan(relocatedRequest));
    ShaderDiskCache::Flush();
    std::vector<std::byte> relocatedKey;
    ShaderDiskCache::BuildPlanKey(relocatedRequest, 32u, relocatedKey);
    const auto relocatedPath = ShaderDiskCache::EntryDirectory() / ShaderDiskCache::EntryName(relocatedKey);
    std::filesystem::rename(relocatedPath, relocatedPath.string() + ".reload-preserved");
    ReleaseUndrawnResourcePlan(*relocatedHandle);
    auto drawn = relocatedRequest;
    drawn.layout.pushConstantOffsetBytes = 16u;
    drawn.layout.pushConstantSizeBytes = 112u;
    const auto relocatedInvocation = PreparedShaderInvocation::TryCreate(drawn, relocatedHandle);
    require(relocatedInvocation.has_value(), "the prepared artifact refused a relocated push layout");
    RequestMemoryView drawnMemory(drawn.context.memory);
    static_cast<void>(relocatedInvocation->Materialize(*relocatedInvocation->Capture(drawnMemory.MakeRuntime(drawn.context.userData, drawn.shader.codeAddress))));
    auto relocatedProbe = relocatedRequest;
    relocatedProbe.layout.pushConstantSizeBytes = 0u;
    ShaderPreparationContext relocatedPreparation;
    static_cast<void>(PrepareShader(relocatedProbe, &relocatedPreparation));
    require(relocatedPreparation.ResourcePrograms() == 1u, "a plan reload retained full IR after the prepared artifact served a relocated push layout");

    RequestMemoryView registeredMemory(registeredRequest.context.memory);
    const auto runtimeCapture = CaptureResources(registeredRequest, registeredMemory.MakeRuntime(registeredRequest.context.userData, registeredRequest.shader.codeAddress), *registeredHandle);
    const auto beforeKeep = ShaderDiskCache::Totals();
    ReleaseUndrawnResourcePlan(*registeredHandle);
    require(GetResourcePlan(registeredRequest) == runtimeCapture->plan && ShaderDiskCache::Totals().planHits == beforeKeep.planHits, "a runtime-captured plan was released");
    const auto covers = [&](const std::function<bool(const IrResourcePlan&)>& predicate) {
        return std::any_of(plans.begin(), plans.end(), [&](const std::shared_ptr<const IrResourcePlan>& plan) { return predicate(*plan); });
    };
    require(covers([](const IrResourcePlan& plan) { return std::any_of(plan.descriptorSources.begin(), plan.descriptorSources.end(), [](const DescriptorSource& source) { return source.tableColumn.has_value() && source.tableColumn->keyDomains.empty(); }); }), "no fixture plans a whole image table");
    require(covers([](const IrResourcePlan& plan) { return std::any_of(plan.descriptorSources.begin(), plan.descriptorSources.end(), [](const DescriptorSource& source) { return source.tableColumn.has_value() && !source.tableColumn->keyDomains.empty(); }); }), "no fixture plans an image table with a key domain");
    require(covers([](const IrResourcePlan& plan) { return std::any_of(plan.descriptorSources.begin(), plan.descriptorSources.end(), [](const DescriptorSource& source) { return source.tableColumn.has_value() && source.tableColumn->address; }); }), "no fixture plans an image table through a pointer");
    require(covers([](const IrResourcePlan& plan) { return std::any_of(plan.descriptorSources.begin(), plan.descriptorSources.end(), [](const DescriptorSource& source) { return source.tableColumn.has_value() && source.tableColumn->selected; }); }), "no fixture plans a selected image table record");
    require(covers([](const IrResourcePlan& plan) { return std::any_of(plan.controlFlow.begin(), plan.controlFlow.end(), [](const ResourceBlock& block) { return block.condition != nullptr; }); }), "no fixture plans a control-flow condition");
    require(covers([](const IrResourcePlan& plan) { return std::any_of(plan.valueStorage.begin(), plan.valueStorage.end(), [](const auto& value) { return value.IsPhi(); }); }), "no fixture plans a phi");
    require(covers([](const IrResourcePlan& plan) { return !plan.guardedSrtSlots.empty(); }), "no fixture plans a guarded SRT read");
    require(covers([](const IrResourcePlan& plan) { return std::any_of(plan.pureFlatSlots.begin(), plan.pureFlatSlots.end(), [](std::uint8_t pure) { return pure != 0u; }); }), "no fixture plans a pure flat slot");
}

void verifyPlanKey() {
    auto fixtures = planFixtures();
    auto& fixture = fixtures[1];
    const auto planKey = [&](const std::function<void(RecompileRequest&)>& change, std::uint32_t hostSubgroupSize = 32u) {
        auto request = fixture.Request(true);
        change(request);
        std::vector<std::byte> key;
        ShaderDiskCache::BuildPlanKey(request, hostSubgroupSize, key);
        return key;
    };
    const auto key = planKey([](RecompileRequest&) {});
    require(planKey([](RecompileRequest&) {}) == key, "the plan key is not deterministic");
    std::vector<std::byte> variantKey;
    ShaderDiskCache::BuildKey(fixture.Request(true), 32u, variantKey);
    require(variantKey != key && ShaderDiskCache::EntryName(variantKey) != ShaderDiskCache::EntryName(key), "the plan and the compiled variant share an entry");
    require(planKey([](RecompileRequest& request) { request.layout = {1u, 4u, 16u, 64u}; }) == key, "the binding layout changed the plan key");
    require(planKey([](RecompileRequest& request) { request.shader.codeAddress += 0x1000u; request.context.memory = {}; }) == key, "the code address or the memory changed the plan key");
    auto userData = fixture.userData;
    userData[0] ^= 0x100u;
    require(planKey([&](RecompileRequest& request) { request.context.userData = userData; }) == key, "user data values changed the plan key");
    const auto original = fixture.code;
    for (std::size_t word = 0; word < original.size(); ++word) {
        for (std::uint32_t bit = 0; bit < 32u; ++bit) {
            fixture.code = original;
            fixture.code[word] ^= 1u << bit;
            require(planKey([](RecompileRequest&) {}) != key, "the plan key ignores code word " + std::to_string(word) + " bit " + std::to_string(bit));
        }
    }
    fixture.code = original;
    fixture.code.push_back(0xbf810000u);
    require(planKey([](RecompileRequest&) {}) != key, "the plan key ignores an appended code word");
    fixture.code = original;
    require(planKey([](RecompileRequest&) {}, 64u) != key, "the plan key ignores the host subgroup size");
    require(planKey([](RecompileRequest& request) { request.target.subgroupSize = 64u; }) != key, "the plan key ignores the target");
    require(planKey([](RecompileRequest& request) { request.target.srgbDecodeFormats = 2u; }) != key, "the plan key ignores the sRGB formats decoded in the shader");
    require(planKey([](RecompileRequest& request) { request.context.waveSize = 64u; request.context.compute->numThreads[0] = 64u; }) != key, "the plan key ignores the wave size");
    require(planKey([](RecompileRequest& request) { request.context.floatMode = ShaderFloatMode{0xc0u, true, false, false}; }) != key, "the plan key ignores the float mode");
}

void appendWords(std::vector<std::byte>& out, std::span<const std::uint32_t> words) {
    const auto count = static_cast<std::uint64_t>(words.size());
    const auto countBytes = std::as_bytes(std::span(&count, 1));
    out.insert(out.end(), countBytes.begin(), countBytes.end());
    const auto bytes = std::as_bytes(words);
    out.insert(out.end(), bytes.begin(), bytes.end());
}

void appendSnapshot(std::vector<std::byte>& out, const ResourceCapture& capture) {
    const auto& snapshot = capture.snapshot;
    for (const auto* values : {&snapshot.buffers, &snapshot.images, &snapshot.samplers}) {
        std::vector<std::uint32_t> words;
        for (const auto& value : *values) {
            words.push_back(value.dwordCount);
            words.insert(words.end(), value.dwords.begin(), value.dwords.end());
        }
        appendWords(out, words);
    }
    appendWords(out, snapshot.flattenedSrt);
    appendWords(out, snapshot.userData);
    const auto& fill = snapshot.uniformFill;
    appendWords(out, std::vector<std::uint32_t>{static_cast<std::uint32_t>(fill.kind), fill.resource, fill.groupStride[0], fill.groupStride[1], fill.groupStride[2], fill.words, fill.value, snapshot.nullRootReads});
    std::vector<std::uint32_t> poison;
    for (const auto& entry : snapshot.srtPoison) poison.insert(poison.end(), {entry.slot, entry.pc, static_cast<std::uint32_t>(entry.address), static_cast<std::uint32_t>(entry.address >> 32u)});
    appendWords(out, poison);
    std::vector<std::uint32_t> leaves{capture.readTrace.leafSlot};
    for (const auto& [slot, address] : capture.readTrace.leaves) leaves.insert(leaves.end(), {slot, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u)});
    appendWords(out, leaves);
    std::vector<std::uint32_t> reads;
    for (const auto address : capture.readTrace.otherReads) reads.insert(reads.end(), {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u)});
    appendWords(out, reads);
}

std::vector<std::vector<std::byte>> preparedOutputs(std::vector<PlanFixture>& fixtures, std::vector<std::shared_ptr<const IrResourcePlan>>* plans = nullptr) {
    std::vector<std::vector<std::byte>> outputs;
    for (auto& fixture : fixtures) {
        const auto request = fixture.Request(true);
        const auto handle = PrepareShader(request);
        RequestMemoryView view(request.context.memory);
        const auto capture = CaptureResources(request, view.MakeRuntime(request.context.userData, request.shader.codeAddress), *handle);
        const auto result = MaterializeShader(request, *capture, *handle);
        std::vector<std::byte> bytes;
        appendWords(bytes, GetPreparedArtifact(*handle).spirv.Words());
        ShaderDiskCache::EncodeResult(*result, bytes);
        appendSnapshot(bytes, *capture);
        require(ShaderDiskCache::EncodePlan(*capture->plan, bytes), fixture.name + ": the captured plan does not encode");
        if (plans != nullptr) plans->push_back(capture->plan);
        outputs.push_back(std::move(bytes));
    }
    return outputs;
}

std::filesystem::path planOutputPath(std::size_t index) {
    return ShaderCacheDirectory() / "plan-outputs" / ("fixture-" + std::to_string(index) + ".bin");
}

std::filesystem::path planEntryPath(PlanFixture& fixture) {
    std::vector<std::byte> key;
    ShaderDiskCache::BuildPlanKey(fixture.Request(true), 32u, key);
    return ShaderDiskCache::EntryDirectory() / ShaderDiskCache::EntryName(key);
}

int runPlanLoadingProcess(std::string_view mode) {
    if (mode == "--load-plan-switch") setEnvironment("APS5_TRACE_IMAGE_TABLE", "1");
    if (mode == "--load-plan-off") setEnvironment("APS5_NO_PERF_DISK_PLANS", "1");
    if (mode == "--load-plan-reuse-off") {
        setEnvironment("APS5_NO_TABLE_SNAPSHOT_REUSE", "1");
        setEnvironment("APS5_NO_TABLE_RESOLUTION_REUSE", "1");
    }
    auto fixtures = planFixtures();
    const auto count = fixtures.size();
    std::vector<std::shared_ptr<const IrResourcePlan>> plans;
    const auto outputs = preparedOutputs(fixtures, &plans);
    ShaderDiskCache::Flush();
    const auto totals = ShaderDiskCache::Totals();
    const auto builds = FrontEndBuilds();
    const auto counts = " (front end builds " + std::to_string(builds) + ", plan hits " + std::to_string(totals.planHits) + ", plan misses " + std::to_string(totals.planMisses) + ", plan load failures " + std::to_string(totals.planLoadFailures) + ", variant hits " + std::to_string(totals.hits) + ", variant misses " + std::to_string(totals.misses) + ")";
    if (mode == "--load-plan-switch") {
        require(builds == count && totals.planHits == 0u && totals.planMisses == count && totals.misses == count, "another switch value reused stored plans or variants" + counts);
        std::cout << "a process with another switch value rebuilt all " << count << " plans\n";
        return 0;
    }
    if (mode == "--load-plan") {
        require(builds == 0u && totals.planHits == count && totals.planMisses == 0u && totals.planLoadFailures == 0u && totals.hits == count && totals.misses == 0u, "the second process ran the front end" + counts);
    } else if (mode == "--load-plan-reuse-off") {
        require(builds == 0u && totals.planHits == count && totals.planMisses == 0u && totals.planLoadFailures == 0u && totals.hits == count && totals.misses == 0u, "turning the image table reuse off changed the stored plan or variant keys" + counts);
    } else if (mode == "--load-plan-off") {
        require(builds == count && totals.planHits == 0u && totals.planMisses == 0u && totals.planWrites == 0u && totals.hits == count && totals.misses == 0u, "APS5_NO_PERF_DISK_PLANS=1 used the stored plans or missed the stored variants" + counts);
    } else {
        require(builds == 1u && totals.planHits == count - 1u && totals.planLoadFailures == 1u && totals.planWrites == 1u && totals.hits == count && totals.misses == 0u, "the damaged plan was not rejected and rebuilt alone" + counts);
    }
    for (std::size_t index = 0; index < count; ++index) {
        std::vector<std::byte> expected;
        require(ReadWholeFile(planOutputPath(index), expected), "cannot read the first process's output for " + fixtures[index].name);
        require(outputs[index] == expected, fixtures[index].name + ": the SPIR-V, bindings, push constants, snapshot or plan differ from the first process");
    }
    if (mode == "--load-plan") {
        auto changed = planFixtures();
        auto& fixture = changed.front();
        fixture.code.insert(fixture.code.end() - 1, 0xbf800000u);
        const auto request = fixture.Request(true);
        static_cast<void>(PrepareShader(request));
        const auto after = ShaderDiskCache::Totals();
        require(FrontEndBuilds() == 1u && after.planHits == count && after.planMisses == 1u, "a changed code word reused a stored plan");
        std::cout << "second process prepared " << count << " shaders from stored plans without running the front end\n";
    } else if (mode == "--load-plan-reuse-off") {
        std::cout << "with the image table reuse off, a process loaded all " << count << " stored plans and variants with the same output\n";
    } else if (mode == "--load-plan-off") {
        std::cout << "APS5_NO_PERF_DISK_PLANS=1 ran the front end for all " << count << " shaders with the same output\n";
    } else {
        std::cout << "a damaged plan entry was rejected, rebuilt and stored again with the same output\n";
    }
    for (std::size_t index = 0; index < count; ++index) {
        const auto fresh = GetResourcePlan(fixtures[index].Request(false));
        require(samePlan(*plans[index], *fresh), fixtures[index].name + ": the plan this process used differs from the plan its front end builds");
    }
    ShaderDiskCache::Flush();
    return 0;
}

void verifyPlanAcrossProcesses(const char* self) {
    auto fixtures = planFixtures();
    const auto builds = FrontEndBuilds();
    const auto before = ShaderDiskCache::Totals();
    const auto outputs = preparedOutputs(fixtures);
    ShaderDiskCache::Flush();
    const auto after = ShaderDiskCache::Totals();
    require(FrontEndBuilds() - builds == fixtures.size() && after.planMisses - before.planMisses == fixtures.size() && after.planWrites - before.planWrites == fixtures.size(), "the first process did not build and store every plan");
    std::filesystem::create_directories(planOutputPath(0).parent_path());
    for (std::size_t index = 0; index < outputs.size(); ++index) require(WriteFileAtomically(planOutputPath(index), outputs[index]), "cannot write the first process's output");
    auto refused = fixtures[1];
    refused.codeAddress = 0x200000u;
    refused.code = {0xf40c0200u, 0xfa000000u, 0xf4000400u, 0xfa000050u, 0xbf8cc07fu, 0xbf068010u, 0xbf850003u, 0xf4080500u, 0xfa000020u, 0xbf820002u, 0xf4080500u, 0x20000000u, 0xf4080600u, 0xfa000040u, 0xbf8cc07fu, 0xf09c0f08u, 0x00a20000u, 0xbf8c3f70u, 0xe0700000u, 0x80060000u, 0xbf810000u};
    bool threw = false;
    try {
        static_cast<void>(PrepareShader(refused.Request(true)));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    ShaderDiskCache::Flush();
    require(threw && ShaderDiskCache::Totals().planWrites == after.planWrites && !std::filesystem::exists(planEntryPath(refused)), "a plan the front end refused was stored");
    std::uintmax_t planBytes = 0;
    for (auto& fixture : fixtures) planBytes += std::filesystem::file_size(planEntryPath(fixture));
    std::cout << fixtures.size() << " plan entries, " << planBytes / fixtures.size() << " bytes per entry on average\n";
    const auto run = [&](const char* mode) {
        const std::string command = "\"" + std::string(self) + "\" " + mode;
        require(std::system(command.c_str()) == 0, std::string("the process ") + mode + " failed");
    };
    run("--load-plan");
    run("--load-plan-off");
    run("--load-plan-reuse-off");
    std::vector<std::byte> entry;
    const auto damaged = planEntryPath(fixtures.front());
    require(ReadWholeFile(damaged, entry), "cannot read a stored plan entry");
    entry[entry.size() / 2] ^= std::byte{0x20};
    require(WriteFileAtomically(damaged, entry), "cannot damage a stored plan entry");
    run("--load-plan-corrupt");
    std::vector<std::byte> rewritten;
    std::vector<std::byte> key;
    ShaderDiskCache::BuildPlanKey(fixtures.front().Request(true), 32u, key);
    IrResourcePlan plan;
    require(ReadWholeFile(damaged, rewritten) && ShaderDiskCache::DecodePlanEntry(rewritten, key, plan) == ShaderDiskCache::LoadStatus::Loaded, "the rebuilt plan was not stored again");
    run("--load-plan-switch");
}

void verifySpecializationDropsDeadConstants() {
    const auto module = [](std::uint32_t selector) {
        std::vector<std::uint32_t> words{spv::MagicNumber, 0x00010300u, 0u, 64u, 0u};
        const auto emit = [&](spv::Op op, std::initializer_list<std::uint32_t> operands) {
            words.push_back((static_cast<std::uint32_t>(operands.size() + 1u) << 16u) | op);
            words.insert(words.end(), operands);
        };
        emit(spv::OpCapability, {spv::CapabilityShader});
        emit(spv::OpMemoryModel, {spv::AddressingModelLogical, spv::MemoryModelGLSL450});
        emit(spv::OpEntryPoint, {spv::ExecutionModelGLCompute, 10u, 0x6e69616du, 0u});
        emit(spv::OpExecutionMode, {10u, spv::ExecutionModeLocalSize, 1u, 1u, 1u});
        emit(spv::OpName, {6u, 0x00006c73u});
        emit(spv::OpTypeVoid, {1u});
        emit(spv::OpTypeBool, {2u});
        emit(spv::OpTypeInt, {3u, 32u, 0u});
        emit(spv::OpTypePointer, {4u, spv::StorageClassPrivate, 3u});
        emit(spv::OpTypeFunction, {5u, 1u});
        emit(spv::OpConstant, {3u, 6u, selector});
        emit(spv::OpConstant, {3u, 7u, 4u});
        emit(spv::OpConstant, {3u, 9u, 1u});
        emit(spv::OpConstant, {3u, 21u, 3u});
        emit(spv::OpTypeArray, {22u, 3u, 21u});
        emit(spv::OpTypePointer, {23u, spv::StorageClassPrivate, 22u});
        emit(spv::OpVariable, {4u, 8u, spv::StorageClassPrivate});
        emit(spv::OpVariable, {23u, 24u, spv::StorageClassPrivate});
        emit(spv::OpFunction, {1u, 10u, spv::FunctionControlMaskNone, 5u});
        emit(spv::OpLabel, {11u});
        emit(spv::OpIEqual, {2u, 12u, 6u, 6u});
        emit(spv::OpSelect, {3u, 13u, 12u, 7u, 9u});
        emit(spv::OpStore, {8u, 13u});
        emit(spv::OpReturn, {});
        emit(spv::OpFunctionEnd, {});
        return words;
    };
    struct Contents {
        std::map<std::uint32_t, std::uint32_t> constants;
        std::size_t names = 0;
        std::uint32_t stored = 0;
    };
    const auto contents = [](const std::vector<std::uint32_t>& words) {
        Contents result;
        for (std::size_t cursor = 5; cursor < words.size();) {
            const auto count = words[cursor] >> 16u;
            const auto op = static_cast<spv::Op>(words[cursor] & 0xffffu);
            require(count != 0u && count <= words.size() - cursor, "dead constant removal truncated an instruction");
            if (op == spv::OpConstant && count == 4u) result.constants.emplace(words[cursor + 2u], words[cursor + 3u]);
            if (op == spv::OpName && words[cursor + 1u] == 6u) ++result.names;
            if (op == spv::OpStore && words[cursor + 1u] == 8u) result.stored = words[cursor + 2u];
            cursor += count;
        }
        return result;
    };
    const auto rgba = SpecializeSpirv(module(4u));
    const auto bgra = SpecializeSpirv(module(6u));
    require(rgba == bgra, "variants that differ only in a folded constant specialized to different SPIR-V");
    const auto result = contents(rgba);
    require(!result.constants.contains(6u) && result.names == 0u, "specialization kept a constant the folding consumed, or its name");
    require(result.constants.contains(21u), "specialization removed the constant an array type's length still uses");
    require(result.constants.contains(result.stored) && result.constants.at(result.stored) == 4u, "the store no longer reads the selected value");
    require(SpecializeSpirv(rgba) == rgba, "dead constant removal is not stable");
}

void verifyDefaultDirectory(const char* self) {
    setEnvironment("ANYPS5_SHADER_CACHE_DIR", "");
    setEnvironment("ANYPS5_NO_SHADER_CACHE", "1");
    require(ShaderRecompiler::ShaderCacheDirectory().empty(), "ANYPS5_NO_SHADER_CACHE=1 did not disable the cache");
    setEnvironment("ANYPS5_NO_SHADER_CACHE", "0");
    const auto directory = ShaderRecompiler::ShaderCacheDirectory();
    require(directory.filename() == "shader_cache", "the default cache directory is not named shader_cache");
    std::error_code error;
    require(std::filesystem::equivalent(directory.parent_path(), std::filesystem::absolute(self).parent_path(), error) && !error, "the default cache directory is not beside the executable");
}

}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--resource-plan-codec") {
            verifyResultRoundTrip();
            verifyEntryRoundTrip();
            verifyPlanRoundTrip();
            std::cout << "resource plan codec tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--artifact-codec") {
            verifyResultRoundTrip();
            verifyEntryRoundTrip();
            std::cout << "artifact codec tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--large-file") return runLargeFileWrite();
        if (argc == 2 && std::string_view(argv[1]) == "--load") return runLoadingProcess();
        if (argc == 2 && std::string_view(argv[1]) == "--load-lazy-plan") return runLazyPlanLoadingProcess(false);
        if (argc == 2 && std::string_view(argv[1]) == "--load-eager-plan") return runLazyPlanLoadingProcess(true);
        if (argc == 2 && std::string_view(argv[1]) == "--load-wrap-loop") { verifyUnnormalizedWrapTransitions(true); return 0; }
        if (argc == 2 && (std::string_view(argv[1]) == "--load-direct-image-refusal" || std::string_view(argv[1]) == "--load-direct-image-loop-guard"
                || std::string_view(argv[1]) == "--load-guarded-direct-image")) return runDirectImageLoadingProcess(argv[1]);
        if (argc == 2 && std::string_view(argv[1]) == "--load-prebuilt") return runPrebuiltLoadingProcess();
        if (argc == 2 && std::string_view(argv[1]) == "--no-failure-memo") return runWithoutFailureMemo();
        if (argc == 2 && std::string_view(argv[1]) == "--load-presence-substitutes") return runPresenceSubstituteProcess();
        if (argc == 2 && std::string_view(argv[1]).starts_with("--load-plan")) return runPlanLoadingProcess(argv[1]);
        const auto directory = std::filesystem::temp_directory_path() / ("aps5-shader-disk-cache-test-" + std::to_string(std::random_device{}()));
        std::filesystem::remove_all(directory);
        verifyDefaultDirectory(argv[0]);
        setEnvironment("ANYPS5_NO_SHADER_CACHE", "0");
        setEnvironment("ANYPS5_SHADER_CACHE_DIR", directory.string());
        verifyAtomicReplacement(directory);
        verifyResultRoundTrip();
        verifyElementFlags();
        verifyEntryRoundTrip();
        verifyPrebuiltPack(directory / "packs");
        verifyArtifactStorageIsolation();
        verifyKeySensitivity();
        verifyMeshBarycentricRequestSerialization();
        verifyFloatModeSerialization();
        verifyPlanRoundTrip();
        verifyPlanKey();
        verifyStore();
        verifyPersist();
        verifyAcrossProcesses(argv[0]);
        verifyPrebuiltAcrossProcesses(argv[0], directory);
        verifyPlanAcrossProcesses(argv[0]);
        verifyDirectImageAcrossProcesses(argv[0]);
        verifyUnnormalizedWrapTransitions();
        {
            DirectImageCacheLoopGuard loop("8");
            const std::string command = "\"" + std::string(argv[0]) + "\" --load-wrap-loop";
            require(std::system(command.c_str()) == 0, "cold wrap loop artifact process failed");
        }
        verifyEmissionFailureMemo();
        verifyFailureMemoSwitch(argv[0]);
        verifyInvocationIsolation();
        verifyResultMemoBudget();
        verifyBufferAlignmentSpecialization();
        verifyBufferPresenceSubstitutes(argv[0]);
        verifyRawBufferShape();
        verifyBindingPlanSelection();
        verifyVertexTypeSpecialization();
        verifyBuiltinSpecialization();
        verifySpecializationLiveness();
        verifySpecializationFunctionLiveness();
        verifySpecializationDropsDeadConstants();
        ShaderRecompiler::ShaderDiskCache::Flush();
        removeCacheDirectory(directory);
        std::cout << "shader disk cache tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
