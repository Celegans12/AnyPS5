#include "ShaderDiskCache.hpp"
#include "BdaAbi.hpp"
#include "CacheKey.hpp"
#include "VertexInputSpecialization.hpp"
#include "SpirvBackend/SpirvSpecialization.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Optimization/DescriptorBindingBuilder.hpp"
#include "ShaderCacheDirectory.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <latch>
#include <map>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

thread_local bool mainAllocationThread = false;
std::atomic<bool> failColdWorkerAllocation{false};
std::atomic<unsigned> coldWorkerAllocationFailures{0};

void* allocateTestMemory(std::size_t size) {
    bool fail = size == 128u * 1024u && !mainAllocationThread && failColdWorkerAllocation.exchange(false, std::memory_order_relaxed);
    if (fail) coldWorkerAllocationFailures.fetch_add(1u, std::memory_order_relaxed);
    for (;;) {
        if (void* memory = fail ? nullptr : std::malloc(size == 0u ? 1u : size)) return memory;
        fail = false;
        const auto handler = std::get_new_handler();
        if (handler == nullptr) throw std::bad_alloc();
        handler();
    }
}

}

void* operator new(std::size_t size) { return allocateTestMemory(size); }
void* operator new[](std::size_t size) { return allocateTestMemory(size); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

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
}

bool sameBinding(const DescriptorBinding& left, const DescriptorBinding& right) {
    return left.kind == right.kind && left.role == right.role && left.descriptorSet == right.descriptorSet && left.binding == right.binding && left.count == right.count && left.guestDescriptor == right.guestDescriptor && left.readOnly == right.readOnly && left.imageShape == right.imageShape && left.samplerDepthCompare == right.samplerDepthCompare && left.imageWritten == right.imageWritten && left.imageDepthCompare == right.imageDepthCompare && left.imageAtomic == right.imageAtomic && left.imageShapes == right.imageShapes && left.bufferAtomic == right.bufferAtomic && left.bufferWritten == right.bufferWritten && left.bufferRead == right.bufferRead && left.samplerUnnormalized == right.samplerUnnormalized && left.imageUnnormalized == right.imageUnnormalized && left.imageSamplers == right.imageSamplers;
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
    require(left.spirv.Words() == right.spirv.Words(), prefix + "SPIR-V differs");
    require(left.bdaAbiVersion == right.bdaAbiVersion, prefix + "BDA ABI version differs");
    require(left.runtimeAbiVersion == right.runtimeAbiVersion, prefix + "runtime ABI version differs");
    require(left.memoryOffsetDword == right.memoryOffsetDword, prefix + "memory offset differs");
    require(left.shaderDataDwords == right.shaderDataDwords && left.imageMetadataDword == right.imageMetadataDword && left.runtimeImageCount == right.runtimeImageCount && left.runtimeImageResources == right.runtimeImageResources, prefix + "compact data layout differs");
    require(left.hostSubgroupSize == right.hostSubgroupSize, prefix + "host subgroup size differs");
    require(left.vertexInputs == right.vertexInputs, prefix + "vertex inputs differ");
    require(left.vertexInputPatches == right.vertexInputPatches, prefix + "vertex type patches differ");
    require(left.vertexOffsetSgpr == right.vertexOffsetSgpr && left.instanceOffsetSgpr == right.instanceOffsetSgpr, prefix + "offset SGPRs differ");
    require(left.vertexOffsetShared == right.vertexOffsetShared && left.instanceOffsetShared == right.instanceOffsetShared && left.vertexOffsetConflict == right.vertexOffsetConflict && left.instanceOffsetConflict == right.instanceOffsetConflict, prefix + "offset flags differ");
    require(left.parameterExports == right.parameterExports, prefix + "parameter exports differ");
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
    require(left.specialization == right.specialization, prefix + "specialization constants differ");
    require(left.workgroupMemoryDwords == right.workgroupMemoryDwords, prefix + "workgroup memory stride differs");
    require(left.poisonedSrtReads == right.poisonedSrtReads, prefix + "poisoned SRT read counts differ");
    require(left.hasDeferredDirectImage == right.hasDeferredDirectImage, prefix + "deferred direct image flag differs");
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
    binding.imageShapes = {DescriptorImageShape::Image2D, DescriptorImageShape::Image3D};
    binding.bufferAtomic = {true};
    binding.bufferWritten = {false, false, true, true, false};
    binding.samplerUnnormalized = {false, true, true};
    binding.imageUnnormalized = {true, false, seed % 2 == 0};
    binding.imageSamplers = {0x5u, 0u, 0x80000000u};
    binding.bufferRead = {true, false, false, true, true, false};
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
    result.pushConstants = {std::byte{1}, std::byte{0xff}, std::byte{0}, std::byte{0x80}, std::byte{7}};
    result.bdaAbiVersion = 3;
    result.memoryOffsetDword = 7;
    result.hostSubgroupSize = 32;
    result.workgroupMemoryDwords = 32769u;
    result.vertexInputs = {{1, 4, 2}, {5, 2, 0}};
    result.vertexInputPatches = {{1, 20, {7, 8, 9}}, {5, 40, {10, 11, 12}}};
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
    result.poisonedSrtReads = 3;
    result.barycentricEmulation = {true, false, true};
    result.variantId = 99;
    result.imageTablePoison = {{{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}, 8u, 2u, 6u}, {{9u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}, 4u, 0xffffffffu, 19u}};
    result.imageTableRanges = {{0x1000u, 0x200u}, {0x5000000000ull, 48u}};
    result.imageTableReadRanges = {{{0x1000u, 32u}, {0x1040u, 32u}}, {}, {{0x5000000000ull, 16u}}};
    result.imageTableShader = 0x80003aa700ull;
    result.imageTableFaults = 3u;
    result.hasDeferredDirectImage = true;
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
    image.flatVolumeCompatible = false;
    image.flatLineCompatible = false;
    image.foldTexelOffsetsOnly = true;
    image.byElements = 4;
    image.byComponents = 1;
    image.emulatedCompare = 0x1a07u;
    image.table = 0;
    image.tableOperation = TableOperation::Unsupported;
    info.info.images = {image};
    ResourceMaterializer::PrepareImageModes(info.info);
    info.info.samplers = {{7, 0x10, 3, true, false, SamplerUseExplicitLod | SamplerUseGather, 1, true}};
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
    info.bindings.pushDataStartDword = 2;
    info.bindings.memoryOffsetDword = 1;
    info.bindings.memoryOffsetCount = 5;
    info.bindings.userDataRegisters = {4, 5, 9};
    info.bindings.descriptors = {{DescriptorBindingKind::Buffers, {0, 1}}, {DescriptorBindingKind::FlattenedSrt, {}}};
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

std::vector<std::uint32_t> coldSpirvWords() {
    std::vector<std::uint32_t> words(128u * 1024u / sizeof(std::uint32_t));
    for (std::size_t index = 0; index < words.size(); ++index) words[index] = static_cast<std::uint32_t>(index % 64u);
    words[0] = 0x07230203u;
    return words;
}

void verifyColdSpirvStorage() {
    const auto expected = coldSpirvWords();
    SharedSpirv cold(expected);
    {
        const auto alias = cold;
        require(!cold.TryMakeCold(), "shared SPIR-V storage was demoted");
        require(alias.size() == expected.size(), "a failed demotion changed shared SPIR-V size");
    }
    require(cold.TryMakeCold(), "repetitive 128 KiB SPIR-V did not become cold");
    const auto compressedBytes = cold.ColdBytes();
    require(compressedBytes != 0u && compressedBytes < expected.size() * sizeof(std::uint32_t), "cold SPIR-V did not reduce stored bytes");
    require(!cold.empty() && cold.size() == expected.size(), "cold SPIR-V lost its word count");
    require(cold.ColdBytes() == compressedBytes, "size or empty hydrated cold SPIR-V");
    const auto transient = cold.ReadTransient();
    require(transient && *transient == expected, "transient SPIR-V read changed words");
    require(cold.ColdBytes() == compressedBytes, "transient SPIR-V read warmed storage");
    std::barrier start(4);
    std::latch launchReady(1);
    bool launchFailed = false;
    std::array<std::future<const std::vector<std::uint32_t>*>, 4> readers;
    try {
        for (auto& reader : readers) {
            reader = std::async(std::launch::async, [&]() -> const std::vector<std::uint32_t>* {
                launchReady.wait();
                if (launchFailed) return nullptr;
                start.arrive_and_wait();
                const auto& words = cold.Words();
                require(words == expected, "concurrent SPIR-V hydration changed words");
                return &words;
            });
        }
    } catch (...) {
        launchFailed = true;
        launchReady.count_down();
        throw;
    }
    launchReady.count_down();
    const auto* first = readers.front().get();
    for (std::size_t index = 1; index < readers.size(); ++index) {
        require(readers[index].get() == first, "concurrent SPIR-V readers received different hydrated storage");
    }
    require(first == &cold.Words() && cold.ColdBytes() == 0u, "public words did not permanently hydrate SPIR-V");
    require(!cold.TryMakeCold(), "SPIR-V was demoted after public word access");
    require(*transient == expected, "public hydration invalidated a transient read");

    SharedSpirv original(expected);
    require(original.TryMakeCold(), "copy-on-write fixture did not become cold");
    auto changed = original;
    changed[7] ^= 0x80000000u;
    require(original.ColdBytes() > 0u, "mutating a cold copy hydrated the original storage");
    require(original.Words() == expected, "mutating a cold copy changed the original words");
    require(changed.Words()[7] == (expected[7] ^ 0x80000000u), "cold copy mutation was lost");
    changed.resize(expected.size() + 3u);
    require(changed.size() == expected.size() + 3u && changed.Words().back() == 0u, "SPIR-V growth lost its word count or zero-filled tail");
    changed.resize(17u);
    require(changed.size() == 17u && original.size() == expected.size(), "SPIR-V shrink changed another holder's word count");
    changed.insert(changed.begin() + 1, {0x12345678u, 0xabcdef01u});
    require(changed.size() == 19u && changed.Words()[1] == 0x12345678u && changed.Words()[3] == expected[1], "SPIR-V insertion lost its count or original suffix");

    SharedSpirv pinned(expected);
    const auto pin = pinned.ReadTransient();
    require(pin && *pin == expected && !pinned.TryMakeCold(), "pinned SPIR-V storage was demoted");
    pinned[9] ^= 1u;
    require(*pin == expected && pinned.Words()[9] == (expected[9] ^ 1u), "copy-on-write mutation changed pinned transient words");
    SharedSpirv inserted(expected);
    const auto insertionPin = inserted.ReadTransient();
    inserted.insert(insertionPin->cbegin() + 3, {0x12345678u, 0xabcdef01u});
    auto expectedInserted = expected;
    expectedInserted.insert(expectedInserted.cbegin() + 3, {0x12345678u, 0xabcdef01u});
    require(*insertionPin == expected && inserted.Words() == expectedInserted, "const-iterator insertion changed pinned words or lost its position during copy-on-write");
    SharedSpirv released(expected);
    {
        const auto held = released.ReadTransient();
        require(!released.TryMakeCold() && *held == expected, "an active transient pin did not retain warm SPIR-V");
    }
    require(released.TryMakeCold(), "releasing a transient pin prevented later demotion");

    SharedSpirv tiny(std::vector<std::uint32_t>(16u, 0u));
    require(!tiny.TryMakeCold() && tiny.ColdBytes() == 0u && tiny.size() == 16u, "tiny SPIR-V was demoted or changed");
    std::mt19937 random(0x4c6f6c64u);
    std::vector<std::uint32_t> noise(expected.size());
    for (auto& word : noise) word = random();
    SharedSpirv incompressible(noise);
    require(!incompressible.TryMakeCold(), "incompressible SPIR-V was demoted");
    require(incompressible.ColdBytes() == 0u && incompressible.Words() == noise, "failed compression changed SPIR-V storage");
}

void verifyColdCacheRoundTrips() {
    const auto words = coldSpirvWords();
    SampleRequest sample;
    const auto key = sample.Key();
    auto expectedVariant = sampleVariant();
    expectedVariant.artifact.spirv = words;
    auto variant = sampleVariant();
    variant.artifact.spirv = words;
    require(variant.artifact.spirv.TryMakeCold(), "entry fixture did not become cold");
    const auto frozen = std::make_shared<const CompiledVariant>(std::move(variant));
    const auto entryColdBytes = frozen->artifact.spirv.ColdBytes();
    const auto file = ShaderDiskCache::EncodeEntry(key, *frozen);
    require(frozen->artifact.spirv.ColdBytes() == entryColdBytes, "entry encoding warmed frozen SPIR-V storage");
    require(file == ShaderDiskCache::EncodeEntry(key, expectedVariant), "cold entry encoding changed the cache wire format");
    require(ShaderDiskCache::EncodeEntry(key, *frozen) == file, "repeated cold entry encoding changed bytes");
    CompiledVariant decodedVariant;
    require(ShaderDiskCache::DecodeEntry(file, key, decodedVariant) == ShaderDiskCache::LoadStatus::Loaded, "a cold encoded entry does not decode");
    requireSameVariant(expectedVariant, decodedVariant, "cold entry round trip");
    require(decodedVariant.artifact.variantId == 0u, "a cold entry retained a per-process artifact id");
    require(frozen->artifact.spirv.ColdBytes() == entryColdBytes, "cold entry round trip warmed the frozen artifact");

    auto expectedResult = sampleResult();
    expectedResult.spirv = words;
    auto result = sampleResult();
    result.spirv = words;
    require(result.spirv.TryMakeCold(), "result fixture did not become cold");
    const auto resultColdBytes = result.spirv.ColdBytes();
    std::vector<std::byte> bytes;
    ShaderDiskCache::EncodeResult(result, bytes);
    require(result.spirv.ColdBytes() == resultColdBytes, "result encoding warmed cold SPIR-V storage");
    std::vector<std::byte> warmBytes;
    ShaderDiskCache::EncodeResult(expectedResult, warmBytes);
    require(bytes == warmBytes, "cold result encoding changed the cache wire format");
    RecompileResult decodedResult;
    require(ShaderDiskCache::DecodeResult(bytes, decodedResult), "a cold encoded result does not decode");
    requireSameResult(expectedResult, decodedResult, "cold result round trip");
    require(decodedResult.variantId == 0u && !decodedResult.cacheHit, "a cold result retained per-process fields");
    require(result.spirv.ColdBytes() == resultColdBytes, "cold result round trip warmed the source artifact");
}

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
    for (const auto abi : {RuntimeAbi::Version - 1u, RuntimeAbi::Version + 1u}) {
        auto incompatible = variant;
        incompatible.artifact.runtimeAbiVersion = abi;
        const auto incompatibleFile = ShaderDiskCache::EncodeEntry(key, incompatible);
        require(ShaderDiskCache::DecodeEntry(incompatibleFile, key, decoded) == ShaderDiskCache::LoadStatus::Rejected,
            "an entry with an incompatible runtime ABI " + std::to_string(abi) + " loads");
    }
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

void verifyColdWriterAllocationFailure() {
    const auto words = coldSpirvWords();
    auto expected = sampleVariant();
    expected.artifact.spirv = words;
    auto variant = sampleVariant();
    variant.artifact.spirv = words;
    require(variant.artifact.spirv.TryMakeCold(), "writer failure fixture did not become cold");
    const auto frozen = std::make_shared<const CompiledVariant>(std::move(variant));
    const auto coldBytes = frozen->artifact.spirv.ColdBytes();
    SampleRequest sample;
    sample.code.back() = 0xc01df001u;
    const auto key = sample.Key();
    sample.code.back() = 0xc01df002u;
    const auto nextKey = sample.Key();
    const auto expectedBytes = ShaderDiskCache::EncodeEntry(key, *frozen).size() + ShaderDiskCache::EncodeEntry(nextKey, *frozen).size();
    ShaderDiskCache::Flush();
    const auto before = ShaderDiskCache::Totals();
    struct DisarmInjection {
        ~DisarmInjection() { failColdWorkerAllocation.store(false, std::memory_order_relaxed); }
    } disarmInjection;
    coldWorkerAllocationFailures.store(0u, std::memory_order_relaxed);
    failColdWorkerAllocation.store(true, std::memory_order_relaxed);
    ShaderDiskCache::Store(key, frozen);
    ShaderDiskCache::Flush();
    const bool stillArmed = failColdWorkerAllocation.exchange(false, std::memory_order_relaxed);
    const auto failed = ShaderDiskCache::Totals();
    require(!stillArmed && coldWorkerAllocationFailures.load(std::memory_order_relaxed) == 1u, "the cache worker did not encounter exactly one injected 128 KiB allocation failure");
    require(failed.writeFailures == before.writeFailures + 1u && failed.writes == before.writes && failed.bytesWritten == before.bytesWritten, "a failed cold cache write changed success counters or was not counted once");
    require(frozen->artifact.spirv.ColdBytes() == coldBytes && *frozen->artifact.spirv.ReadTransient() == words && frozen->artifact.spirv.ColdBytes() == coldBytes, "a failed cache write changed or hydrated the cold source");
    require(!std::filesystem::exists(ShaderDiskCache::EntryDirectory() / ShaderDiskCache::EntryName(key)), "a failed cold cache write published an entry");
    ShaderDiskCache::Store(key, frozen);
    ShaderDiskCache::Store(nextKey, frozen);
    ShaderDiskCache::Flush();
    const auto recovered = ShaderDiskCache::Totals();
    require(recovered.writeFailures == failed.writeFailures && recovered.writes == failed.writes + 2u && recovered.bytesWritten == failed.bytesWritten + expectedBytes, "the cache writer did not recover for the retry and following job");
    for (const auto* storedKey : {&key, &nextKey}) {
        CompiledVariant loaded{};
        require(ShaderDiskCache::Load(*storedKey, loaded), "an entry written after worker allocation failure does not load");
        requireSameVariant(expected, loaded, "cache writer allocation recovery");
    }
    require(coldWorkerAllocationFailures.load(std::memory_order_relaxed) == 1u && frozen->artifact.spirv.ColdBytes() == coldBytes && *frozen->artifact.spirv.ReadTransient() == words && frozen->artifact.spirv.ColdBytes() == coldBytes, "successful retries repeated injection or changed the cold source");
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

int runOppositeMaskedSelectLoadingProcess() {
    constexpr auto name = "APS5_NO_PERF_MASKED_SELECT_SCRATCH";
    const bool wasPresent = std::getenv(name) != nullptr;
    if (wasPresent) {
#ifdef _WIN32
        require(_putenv_s(name, "") == 0, "cannot unset the masked-select scratch switch");
#else
        require(unsetenv(name) == 0, "cannot unset the masked-select scratch switch");
#endif
    } else {
        setEnvironment(name, "1");
    }
    require((std::getenv(name) != nullptr) != wasPresent, "the child did not toggle the masked-select scratch switch presence");
    std::cout << "opposite masked-select mode: " << (wasPresent ? "present -> absent" : "absent -> present") << std::endl;
    return runLoadingProcess();
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
    const std::string oppositeCommand = "\"" + std::string(self) + "\" --load-opposite-masked-select";
    require(std::system(oppositeCommand.c_str()) == 0, "the opposite masked-select mode loading process failed");
}

class DirectImageCacheLoopGuard {
public:
    explicit DirectImageCacheLoopGuard(const char* value) {
        if (const auto* current = std::getenv("APS5_LOOP_GUARD")) { present = true; previous = current; }
        Set(value);
    }
    ~DirectImageCacheLoopGuard() { Set(present ? previous.c_str() : nullptr); }
private:
    static void Set(const char* value) {
        if (value != nullptr) setEnvironment("APS5_LOOP_GUARD", value);
        else {
#ifdef _WIN32
            _putenv_s("APS5_LOOP_GUARD", "");
#else
            unsetenv("APS5_LOOP_GUARD");
#endif
        }
    }
    bool present = false;
    std::string previous;
};

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
    require(result.specialization.empty() && result.hasDeferredDirectImage == !valid,
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
    rejectsAlignment("buffer atomic on a V# whose base is not DWORD aligned");
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
            require(artifact.spirv.Words() == original, "vertex specialization mutated the shared template");
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
    mainAllocationThread = true;
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--load") return runLoadingProcess();
        if (argc == 2 && std::string_view(argv[1]) == "--load-opposite-masked-select") return runOppositeMaskedSelectLoadingProcess();
        if (argc == 2 && (std::string_view(argv[1]) == "--load-direct-image-refusal" || std::string_view(argv[1]) == "--load-direct-image-loop-guard"
                || std::string_view(argv[1]) == "--load-guarded-direct-image")) return runDirectImageLoadingProcess(argv[1]);
        if (argc == 2 && std::string_view(argv[1]) == "--no-failure-memo") return runWithoutFailureMemo();
        const auto directory = std::filesystem::temp_directory_path() / ("aps5-shader-disk-cache-test-" + std::to_string(std::random_device{}()));
        std::filesystem::remove_all(directory);
        verifyDefaultDirectory(argv[0]);
        setEnvironment("ANYPS5_NO_SHADER_CACHE", "0");
        setEnvironment("ANYPS5_SHADER_CACHE_DIR", directory.string());
        verifyColdSpirvStorage();
        verifyColdCacheRoundTrips();
        verifyResultRoundTrip();
        verifyEntryRoundTrip();
        verifyArtifactStorageIsolation();
        verifyKeySensitivity();
        verifyFloatModeSerialization();
        verifyStore();
        verifyColdWriterAllocationFailure();
        verifyAcrossProcesses(argv[0]);
        verifyDirectImageAcrossProcesses(argv[0]);
        verifyEmissionFailureMemo();
        verifyFailureMemoSwitch(argv[0]);
        verifyInvocationIsolation();
        verifyBufferAlignmentSpecialization();
        verifyBindingPlanSelection();
        verifyVertexTypeSpecialization();
        verifyBuiltinSpecialization();
        verifySpecializationLiveness();
        verifySpecializationFunctionLiveness();
        ShaderRecompiler::ShaderDiskCache::Flush();
        std::error_code error;
        std::filesystem::remove_all(directory, error);
        require(!error, "cannot remove the shader cache test directory: " + error.message());
        std::cout << "shader disk cache tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
