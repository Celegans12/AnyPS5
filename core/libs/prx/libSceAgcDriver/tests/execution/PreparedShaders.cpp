#include "VulkanTestDevice.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "CompiledVariant.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderMemoryCensus.hpp"
#include "RdnaDecoder/RdnaInstruction.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderPreparation.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <future>
#include <barrier>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <thread>

namespace {

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename TAction>
void ExpectFailure(TAction action, const char* expected) {
    try { action(); }
    catch (const std::exception& error) {
        Require(std::string(error.what()).find(expected) != std::string::npos, error.what());
        return;
    }
    throw std::runtime_error("expected preparation failure");
}

void RunUnregistered(AgcDriver::VulkanDevice& device, ShaderRecompiler::RecompileRequest request) {
    const auto code = request.shader.code;
    for (const bool sourceFirst : {false, true}) {
        AgcDriver::DriverDetail::ShaderSnapshot snapshot{request.shader.codeAddress, 0, 0, {code.begin(), code.end()}, {}};
        request.shader.code = snapshot.code;
        if (sourceFirst) {
            const auto handle = AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request);
            Require(snapshot.prepared->entries.size() == 1 && snapshot.prepared->entries.front().handle == handle, "unregistered compute source was not cached");
        }
        static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 0, request));
        Require(snapshot.prepared->entries.size() == 1, "unregistered compute invocation did not cache exactly one artifact");
        const auto handle = snapshot.prepared->entries.front().handle;
        const auto& artifact = ShaderRecompiler::GetPreparedArtifact(*handle);
        for (const bool useCache : {false, true}) {
            request.useCache = useCache;
            Require(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request) == handle, "unregistered compute source was prepared again");
            const auto repeated = AgcDriver::DriverDetail::InvocationFor(snapshot, 0, request);
            ShaderRecompiler::SrtRuntime runtime{};
            runtime.userData = request.context.userData;
            const auto capture = repeated.Capture(runtime);
            const auto result = repeated.Materialize(*capture);
            Require(snapshot.prepared->entries.size() == 1 && result->variantId == artifact.variantId && result->spirv.data() == artifact.spirv.data(), "unregistered compute invocation replaced its cached artifact");
            device.Dispatch(*result, 1, 1, 1);
        }
        device.WaitIdle();
    }
}

void NullPixelAtDraw(AgcDriver::VulkanDevice& device) {
    std::vector<std::uint32_t> code(64, 0);
    code.front() = 0xbf810000u;
    AgcDriver::DriverDetail::ShaderSnapshot snapshot{AgcDriver::DriverDetail::NullPixelProgramAddress(), 0, 1, code, {}};
    snapshot.header.resize(sizeof(Shader));
    std::array<std::uint32_t, 4> users{};
    const auto pixel = AgcDriver::Graphics::DecodePixelStageInfo({}, {}, true);
    const ShaderRecompiler::RecompileRequest registered{{ShaderRecompiler::ShaderStage::Fragment, snapshot.codeAddress, snapshot.code, 0, {}}, {64, 0, {}, {}, pixel, {}, {}}, device.Target(), {0, 0, 0, 128}};
    snapshot.prepared->entries.push_back({0, ShaderRecompiler::PrepareShader(registered)});
    ShaderRecompiler::RecompileRequest request{{ShaderRecompiler::ShaderStage::Fragment, snapshot.codeAddress, snapshot.code, 0, {}}, {32, 0, users, {}, pixel, {}, {}}, device.Target(), {0, 0, 20, 108}};
    static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 0, request));
    Require(snapshot.prepared->entries.size() == 2, "the null pixel program was not prepared at draw");
    static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 0, request));
    Require(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request) == snapshot.prepared->entries.back().handle && snapshot.prepared->entries.size() == 2, "the null pixel program was prepared again for the same draw");
}

void RegisteredMemoryCensus(const std::shared_ptr<const ShaderRecompiler::SourceHandle>& source) {
    using AgcDriver::DriverDetail::InspectRegisteredShaderMemory;
    using AgcDriver::DriverDetail::ShaderRegistry;
    using AgcDriver::DriverDetail::ShaderSnapshot;
    Require(InspectRegisteredShaderMemory({}) == AgcDriver::DriverDetail::RegisteredShaderMemory{}, "empty registry census is not empty");
    auto raw = std::make_shared<ShaderRecompiler::CompiledVariant>();
    std::vector<std::uint32_t> words{1u, 2u, 3u};
    words.reserve(128u);
    raw->artifact.spirv = std::move(words);
    auto cold = std::make_shared<ShaderRecompiler::CompiledVariant>();
    cold->artifact.spirv = std::vector<std::uint32_t>(16u * 1024u, 0u);
    Require(cold->artifact.spirv.TryMakeCold(), "census fixture could not make storage cold");
    const auto rawBefore = raw->artifact.spirv.InspectStorage();
    const auto coldBefore = cold->artifact.spirv.InspectStorage();
    const auto handleFor = [&](const std::shared_ptr<ShaderRecompiler::CompiledVariant>& artifact) {
        auto handle = std::make_shared<ShaderRecompiler::SourceHandle>();
        handle->source = source->source;
        handle->artifact = artifact;
        return handle;
    };
    const auto rawHandle = handleFor(raw);
    const auto rawAlias = handleFor(raw);
    const auto coldHandle = handleFor(cold);
    auto first = std::make_shared<ShaderSnapshot>(ShaderSnapshot{1u, 11u, 0u, {1u, 2u}, {std::byte{1}}});
    auto second = std::make_shared<ShaderSnapshot>(ShaderSnapshot{2u, 12u, 0u, {3u}, {std::byte{2}}});
    first->code.reserve(64u);
    first->header.reserve(128u);
    second->code.reserve(32u);
    second->header.reserve(64u);
    first->prepared->entries = {{0u, rawHandle}, {0u, rawAlias}, {0u, coldHandle}};
    second->prepared->entries = {{0u, rawHandle}, {0u, nullptr}};
    second->prepared->pending = true;
    second->prepared->failure = std::make_exception_ptr(std::runtime_error("census fixture"));
    second->prepared->rectangles.push_back({1u, 2u, nullptr});
    const ShaderRegistry registry{{1u, first}, {2u, second}, {3u, first}};
    const auto census = InspectRegisteredShaderMemory(registry);
    Require(census.registryEntries == 3u && census.snapshots == 2u && census.pending == 1u && census.failed == 1u, "census lost registration status or counted an aliased snapshot twice");
    Require(census.entryReferences == 5u && census.handles == 3u && census.sources == 1u && census.artifacts == 2u && census.missingArtifacts == 1u && census.excludedRectangles == 1u, "census did not deduplicate shared source/artifact owners or disclose omitted records");
    Require(census.rawArtifacts == 1u && census.coldArtifacts == 1u && census.exposedArtifacts == 0u, "census storage state counts are wrong");
    Require(census.logicalBytes == rawBefore.logicalBytes + coldBefore.logicalBytes && census.rawBytes == rawBefore.residentBytes && census.rawCapacityBytes == rawBefore.residentCapacityBytes && census.coldBytes == coldBefore.coldBytes && census.coldCapacityBytes == coldBefore.coldCapacityBytes, "census conflated logical bytes, backing size or capacity");
    Require(census.rawCapacityBytes > census.rawBytes && census.codeBytes == 3u * sizeof(std::uint32_t) && census.codeCapacityBytes == (first->code.capacity() + second->code.capacity()) * sizeof(std::uint32_t) && census.headerBytes == 2u && census.headerCapacityBytes == first->header.capacity() + second->header.capacity(), "census registry capacity accounting is wrong");
    Require(InspectRegisteredShaderMemory(registry) == census, "repeated census changed the sampled storage");
    const auto coldAfter = cold->artifact.spirv.InspectStorage();
    Require(coldAfter.residentBytes == 0u && coldAfter.coldBytes == coldBefore.coldBytes && coldAfter.coldCapacityBytes == coldBefore.coldCapacityBytes && !coldAfter.exposed, "census hydrated or exposed cold storage");

    // DEBUG: distinct artifact objects can alias the same backing. This upper-bound fixture
    // verifies why the diagnostic must not call its artifact-deduplicated sum physical memory.
    const auto backingAlias = std::make_shared<ShaderRecompiler::CompiledVariant>(*cold);
    first->prepared->entries.push_back({0u, handleFor(backingAlias)});
    const auto aliased = InspectRegisteredShaderMemory(registry);
    Require(aliased.artifacts == 3u && aliased.sources == 1u && aliased.coldArtifacts == 2u && aliased.coldBytes == 2u * coldBefore.coldBytes, "census hid the distinct-artifact shared-backing limit");
    const ShaderRecompiler::SharedSpirv& explicitRead = cold->artifact.spirv;
    Require(explicitRead.Words().size() == 16u * 1024u, "census fixture hydration lost words");
    const auto hydrated = cold->artifact.spirv.InspectStorage();
    const auto exposed = InspectRegisteredShaderMemory(registry);
    Require(exposed.coldArtifacts == 0u && exposed.coldBytes == 0u && exposed.coldCapacityBytes == 0u && exposed.rawArtifacts == 3u && exposed.exposedArtifacts == 2u, "census reports publication-time storage instead of current exposed storage");
    Require(exposed.rawBytes == rawBefore.residentBytes + 2u * hydrated.residentBytes && exposed.rawCapacityBytes == rawBefore.residentCapacityBytes + 2u * hydrated.residentCapacityBytes, "census current raw backing totals are wrong");
    std::cout << "registered shader memory census preserves cold storage and deduplicates artifact references\n";
}

void Run(AgcDriver::VulkanDevice& device) {
    NullPixelAtDraw(device);
    alignas(256) std::array<std::uint32_t, 1> code{0xbf810000u};
    std::array<std::uint32_t, 4> users{};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{1, 1, 1}, 0, {false, false, false}, false, 1, {}};
    ShaderRecompiler::RecompileRequest request{{ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}}, {32, 0, users, compute, {}, {}, {}}, device.ComputeTarget(32), {0, 0, 0, 128}};
    RunUnregistered(device, request);
    AgcDriver::DriverDetail::ShaderSnapshot snapshot{request.shader.codeAddress, 0, 0, {code.begin(), code.end()}, {}};
    snapshot.header.resize(sizeof(Shader));
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request)); }, "artifact is missing");
    const auto handle = ShaderRecompiler::PrepareShader(request);
    RegisteredMemoryCensus(handle);
    snapshot.prepared->entries.push_back({0, handle});
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 1, request)); }, "artifact is missing");
    const auto& artifact = ShaderRecompiler::GetPreparedArtifact(*handle);
    for (std::uint32_t value = 0; value < 8; ++value) {
        users.fill(value);
        request.useCache = value % 2 == 0;
        const auto ready = AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request);
        ShaderRecompiler::SrtRuntime runtime{};
        runtime.userData = users;
        const auto capture = ShaderRecompiler::CaptureResources(request, runtime, *ready);
        const auto result = ShaderRecompiler::MaterializeShader(request, *capture, *ready);
        Require(result->variantId == artifact.variantId && result->spirv.data() == artifact.spirv.data(), "invocation replaced the prepared artifact");
        device.Dispatch(*result, 1, 1, 1);
    }
    device.WaitIdle();
    auto registeredRequest = request;
    registeredRequest.shader.code = snapshot.code;
    registeredRequest.useCache = true;
    const auto invocation = AgcDriver::DriverDetail::InvocationFor(snapshot, 0, registeredRequest);
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 0, request)); }, "does not refer to registered code");
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 1, registeredRequest)); }, "outside the snapshot");
    registeredRequest.context.compute->numThreads[0] = 2;
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 0, registeredRequest)); }, "artifact is missing");
    ShaderRecompiler::SrtRuntime preparedRuntime{};
    preparedRuntime.userData = users;
    const auto preparedCapture = invocation.Capture(preparedRuntime);
    const auto firstResult = invocation.Materialize(*preparedCapture);
    const auto repeatedCapture = invocation.Capture(preparedRuntime);
    Require(invocation.Materialize(*repeatedCapture) == firstResult, "prepared invocation rebuilt an unchanged materialized result");
    users[0] += 1;
    const auto changedCapture = invocation.Capture(preparedRuntime);
    const auto changedResult = invocation.Materialize(*changedCapture);
    Require(changedResult != firstResult && changedResult->spirv.data() == firstResult->spirv.data(), "prepared invocation did not distinguish changed runtime data");
    users[0] -= 1;
    Require(invocation.Materialize(*preparedCapture) == firstResult, "prepared invocation evicted the previous runtime data");
    auto replacement = OpenVulkanTestDevice();
    Require(replacement != nullptr && replacement->Serial() != device.Serial(), "replacement device was not created");
    auto replacementRequest = request;
    replacementRequest.target = replacement->ComputeTarget(32);
    Require(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, replacementRequest) == handle, "device replacement discarded a compatible prepared artifact");
    replacementRequest.shader.code = snapshot.code;
    const auto replacementInvocation = AgcDriver::DriverDetail::InvocationFor(snapshot, 0, replacementRequest);
    const auto replacementCapture = replacementInvocation.Capture(preparedRuntime);
    const auto replacementResult = replacementInvocation.Materialize(*replacementCapture);
    Require(replacementResult->spirv.data() == artifact.spirv.data(), "device replacement recompiled the prepared shader");
    replacement->Dispatch(*replacementResult, 1, 1, 1);
    replacement->WaitIdle();
    replacementRequest.target.nonConstantImageOffsets = !replacementRequest.target.nonConstantImageOffsets;
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 0, replacementRequest)); }, "artifact is missing");
    ShaderRecompiler::SrtRuntime runtime{};
    runtime.userData = users;
    const auto capture = ShaderRecompiler::CaptureResources(request, runtime, *handle);
    request.context.compute->numThreads[0] = 2;
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request)); }, "artifact is missing");
    ExpectFailure([&] { static_cast<void>(ShaderRecompiler::CaptureResources(request, runtime, *handle)); }, "does not match the static ABI");
    ExpectFailure([&] { static_cast<void>(ShaderRecompiler::MaterializeShader(request, *capture, *handle)); }, "does not match the static ABI");
    const auto otherHandle = ShaderRecompiler::PrepareShader(request);
    const auto otherCapture = ShaderRecompiler::CaptureResources(request, runtime, *otherHandle);
    ExpectFailure([&] { static_cast<void>(invocation.Materialize(*otherCapture)); }, "another prepared shader");
    ExpectFailure([&] { static_cast<void>(ShaderRecompiler::MaterializeShader(request, *capture, *otherHandle)); }, "another prepared shader");
    request.context.compute->numThreads[0] = 1;
    request.layout.pushConstantSizeBytes = 124;
    Require(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request) == handle, "compatible push constant capacity discarded the prepared artifact");
    request.layout.pushConstantSizeBytes = 126;
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request)); }, "artifact is missing");
    request.layout.pushConstantSizeBytes = 128;
    code[0] = 0xffffffffu;
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request)); }, "artifact is missing");
}

void FailureCapture(AgcDriver::VulkanDevice& device, bool terminalWriteFailure) {
    using namespace ShaderRecompiler;
    const std::array<std::uint32_t, 1> code{0xffffffffu};
    const std::vector<std::byte> header(2048, std::byte{0x5a});
    RecompileRequest request{{ShaderStage::Compute, 0x12345cafeull, code, 0x20000, header},
        {32, 0, {}, ShaderComputeStageInfo{{1, 1, 1}, 0, {}, false, 1, {}}, {}, {}, {}}, device.ComputeTarget(32), {0, 0, 0, 128}};
    const std::array<std::uint32_t, 2> oldWords{1, 2};
    const std::array<MemoryRegion, 1> oldMemory{{{0x40000, std::as_bytes(std::span(oldWords))}}};
    request.context.memory = oldMemory;
    alignas(4096) std::array<std::uint32_t, 1024> dependencyPage{};
    const auto dependencyAddress = reinterpret_cast<std::uintptr_t>(dependencyPage.data());
    const auto leafAddress = reinterpret_cast<std::uintptr_t>(dependencyPage.data() + 4);
    const std::array<std::uint32_t, 4> terminalWords{static_cast<std::uint32_t>(leafAddress), static_cast<std::uint32_t>(leafAddress >> 32u), 16u, 0x31016facu};
    std::copy(terminalWords.begin(), terminalWords.end(), dependencyPage.begin());
    const std::array<std::uint32_t, 2> users{static_cast<std::uint32_t>(dependencyAddress), static_cast<std::uint32_t>(dependencyAddress >> 32u)};
    const std::array<std::uint32_t, 7> captureCode{0xf4080100u, 0xfa000000u, 0xf4280202u, 0xfa1ffffcu, 0xe0700000u, 0x80020000u, 0xbf810000u};
    auto captureRequest = request;
    captureRequest.shader = {ShaderStage::Compute, request.shader.codeAddress, captureCode, 0, {}};
    captureRequest.context.userData = users;
    captureRequest.context.memory = {};
    captureRequest.useCache = false;
    AgcDriver::ShaderMemory partialMemory({});
    Require(partialMemory.Regions().empty(), "partial capture started with supplied dependency regions");
    std::vector<MemoryRegion> terminalMemory;
    const auto failure = [&](auto action) {
        try { action(); }
        catch (const std::exception& error) { return std::string(error.what()); }
        throw std::runtime_error("invalid shader preparation succeeded");
    };
    const auto expected = failure([&] { static_cast<void>(PrepareShader(request)); });
    const char* previous = std::getenv("APS5_DUMP_SHADERS");
    const std::string saved = previous != nullptr ? previous : "";
    const std::string path = "shader_12345cafe.req";
    std::remove(path.c_str());
#ifdef _WIN32
    Require(_putenv_s("APS5_DUMP_SHADERS", "") == 0, "cannot disable shader capture");
#else
    Require(unsetenv("APS5_DUMP_SHADERS") == 0, "cannot disable shader capture");
#endif
    Require(failure([&] { static_cast<void>(AgcDriver::DriverDetail::PrepareShaderWithDiagnostics(request)); }) == expected, "disabled shader capture changed the preparation failure");
    Require(!std::ifstream(path, std::ios::binary).is_open(), "disabled shader capture created a request");
    Require(!AgcDriver::DriverDetail::CaptureTerminalComputeFailure(request, terminalMemory, "disabled", 0, 1, 2), "disabled terminal capture created a request");
#ifdef _WIN32
    Require(_putenv_s("APS5_DUMP_SHADERS", "1") == 0, "cannot enable shader capture");
#else
    Require(setenv("APS5_DUMP_SHADERS", "1", 1) == 0, "cannot enable shader capture");
#endif
    const auto actual = failure([&] { static_cast<void>(AgcDriver::DriverDetail::PrepareShaderWithDiagnostics(request)); });
    const auto directory = std::filesystem::temp_directory_path() / ("anyps5-terminal-capture-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Require(std::filesystem::create_directory(directory), "cannot create terminal capture test directory");
    const auto destination = (terminalWriteFailure ? directory / "missing" / "child" : directory).string();
    bool written = false;
    std::string captureError;
    const auto terminalError = failure([&] {
        try { static_cast<void>(partialMemory.Capture(captureRequest)); }
        catch (const std::exception& error) {
            captureError = error.what();
            terminalMemory = partialMemory.Regions();
            written = AgcDriver::DriverDetail::CaptureTerminalComputeFailure(captureRequest, terminalMemory, error.what(), 7, 11, 13, destination.c_str());
            throw;
        }
    });
    Require(captureError.find("SrtWalker::EvaluateRuntimeSources") != std::string::npos && captureError.find("ReadConstBuffer") != std::string::npos, "partial capture did not refuse the unsupported negative scalar-buffer offset");
    Require(terminalError == captureError, "terminal capture changed the original exception");
    Require(terminalMemory.size() == 1 && terminalMemory[0].guestAddress == dependencyAddress && terminalMemory[0].bytes.size() == sizeof(terminalWords)
        && std::memcmp(terminalMemory[0].bytes.data(), terminalWords.data(), sizeof(terminalWords)) == 0, "failed capture did not retain its actual descriptor reads");
    Require(written == !terminalWriteFailure, "terminal capture write outcome was incorrect");
    Require(!AgcDriver::DriverDetail::CaptureTerminalComputeFailure(request, oldMemory, "second failure", 7, 12, 13, directory.string().c_str()), "terminal capture exceeded its one-attempt budget");
    std::size_t terminalFiles = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        Require(entry.path().extension() == ".req", "terminal capture left a staging file");
        Require(entry.path().filename().string().starts_with("shader_12345cafe_terminal_"), "terminal capture cannot be collected by the runner");
        std::ifstream terminalFile(entry.path(), std::ios::binary);
        const std::string bytes{std::istreambuf_iterator<char>(terminalFile), std::istreambuf_iterator<char>()};
        auto terminalRequest = captureRequest;
        terminalRequest.context.memory = terminalMemory;
        const RequestSerializer terminalSerializer;
        Require(bytes == terminalSerializer.Serialize(terminalRequest), "terminal capture used stale same-address request bytes");
        const auto decoded = terminalSerializer.Deserialize(bytes);
        Require(decoded.request.context.memory.size() == 1 && decoded.request.context.memory[0].guestAddress == dependencyAddress && decoded.request.context.memory[0].bytes.size() == sizeof(terminalWords)
            && std::memcmp(decoded.request.context.memory[0].bytes.data(), terminalWords.data(), sizeof(terminalWords)) == 0, "terminal captured memory did not roundtrip");
        terminalFile.close();
        Require(std::filesystem::remove(entry.path()), "cannot remove terminal capture test file");
        ++terminalFiles;
    }
    Require(terminalFiles == (terminalWriteFailure ? 0u : 1u), "terminal capture file count was incorrect");
    Require(std::filesystem::remove(directory), "cannot remove terminal capture test directory");
#ifdef _WIN32
    Require(_putenv_s("APS5_DUMP_SHADERS", saved.c_str()) == 0, "cannot restore shader capture");
#else
    Require((previous != nullptr ? setenv("APS5_DUMP_SHADERS", saved.c_str(), 1) : unsetenv("APS5_DUMP_SHADERS")) == 0, "cannot restore shader capture");
#endif
    Require(actual == expected, "shader capture changed the preparation failure");
    std::ifstream file(path, std::ios::binary);
    std::ostringstream text;
    text << file.rdbuf();
    file.close();
    std::remove(path.c_str());
    const RequestSerializer serializer;
    Require(text.str().size() > 1024 && text.str() == serializer.Serialize(request), "shader failure capture was incomplete");
    const auto replay = serializer.Deserialize(text.str());
    Require(serializer.Serialize(replay.request) == text.str(), "shader failure capture changed on replay");
}

void PrepareMultisampledStorage(AgcDriver::VulkanDevice& device) {
    using namespace ShaderRecompiler;
    std::array<std::uint32_t, 13> code{0xd7460000u, 0x0401060cu, 0xd7460001u, 0x0405060du, 0x7e04020eu, 0x7e060280u, 0x7e080208u, 0x7e0a0209u, 0x7e0c020au, 0x7e0e020bu, 0xf0200f38u, 0x00000400u, 0xbf810000u};
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x30000u, code, 0, {}};
    const std::array<std::uint32_t, 16> users{};
    request.context.userData = users;
    request.context.waveSize = 32;
    request.context.compute = ShaderComputeStageInfo{{8, 8, 1}, 0, {true, true, false}, false, 2, {}};
    request.target = device.ComputeTarget(32);
    std::vector<std::uint32_t> multisampleCapabilities(request.target.supportedCapabilities.begin(), request.target.supportedCapabilities.end());
    if (std::ranges::find(multisampleCapabilities, spv::CapabilityStorageImageMultisample) == multisampleCapabilities.end()) multisampleCapabilities.push_back(spv::CapabilityStorageImageMultisample);
    request.target.supportedCapabilities = multisampleCapabilities;
    request.layout.pushConstantSizeBytes = 128;
    request.useCache = false;
    for (std::uint32_t variant = 0; variant < 5; ++variant) {
        if (variant != 0u) {
            code[10] = variant % 2u != 0u ? 0xf0200f38u : 0xf0200f30u;
            if (variant >= 3u) code[10] = variant == 3u ? 0xf03c0138u : 0xf03c0130u;
            code[5] = 0x7e060283u;
            code[4] = 0x7e040282u;
        }
        const auto handle = PrepareShader(request);
        const auto& artifact = GetPreparedArtifact(*handle);
        std::vector<std::uint32_t> samples;
        for (std::size_t offset = 5; offset < artifact.spirv.size(); offset += artifact.spirv[offset] >> 16u) {
            const auto instruction = artifact.spirv[offset];
            if ((instruction & 0xffffu) == spv::OpImageWrite) {
                Require(variant < 3u && (instruction >> 16u) == 6u && artifact.spirv[offset + 4] == spv::ImageOperandsSampleMask, "MSAA storage write lost its sample operand");
                samples.push_back(artifact.spirv[offset + 5]);
            }
            if ((instruction & 0xffffu) == spv::OpImageTexelPointer) {
                Require(variant >= 3u && (instruction >> 16u) == 6u, "MSAA atomic lost its sample operand");
                samples.push_back(artifact.spirv[offset + 5]);
            }
        }
        Require(!samples.empty(), "MSAA storage shader has no image writes or atomics");
        const auto expectedSample = variant == 0u ? 0u : variant % 2u != 0u ? 3u : 2u;
        for (const auto sample : samples) {
            bool found = false;
            for (std::size_t offset = 5; offset < artifact.spirv.size(); offset += artifact.spirv[offset] >> 16u) {
                if ((artifact.spirv[offset] & 0xffffu) != spv::OpConstant || artifact.spirv[offset + 2] != sample) continue;
                Require(artifact.spirv[offset + 3] == expectedSample, "MSAA storage operation addresses the wrong sample");
                found = true;
            }
            Require(found, "MSAA sample constant is missing");
        }
    }
    std::vector<std::uint32_t> capabilities(request.target.supportedCapabilities.begin(), request.target.supportedCapabilities.end());
    std::erase(capabilities, spv::CapabilityStorageImageMultisample);
    request.target.supportedCapabilities = capabilities;
    ExpectFailure([&] { static_cast<void>(PrepareShader(request)); }, "storage image multisampling is unavailable on the target device");
}

void SettleLater(AgcDriver::DriverDetail::PreparedShaders& prepared, std::vector<AgcDriver::DriverDetail::PreparedShaders::Entry> entries, std::exception_ptr failure) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        std::lock_guard lock(prepared.mutex);
        prepared.entries.insert(prepared.entries.end(), entries.begin(), entries.end());
        prepared.failure = failure;
        prepared.pending = false;
    }
    prepared.settled.notify_all();
}

std::exception_ptr InjectedFailure() {
    try {
        throw std::runtime_error("injected registration preparation failure");
    } catch (...) {
        return std::current_exception();
    }
}

void PendingRegistrationPreparation(AgcDriver::VulkanDevice& device) {
    alignas(256) const std::array<std::uint32_t, 1> code{0xbf810000u};
    std::array<std::uint32_t, 4> users{};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{1, 1, 1}, 0, {false, false, false}, false, 1, {}};
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    for (const bool source : {true, false}) {
        AgcDriver::DriverDetail::ShaderSnapshot snapshot{address, 0, 0, {code.begin(), code.end()}, {}};
        snapshot.header.resize(sizeof(Shader));
        ShaderRecompiler::RecompileRequest request{{ShaderRecompiler::ShaderStage::Compute, address, snapshot.code, 0, {}}, {32, 0, users, compute, {}, {}, {}}, device.ComputeTarget(32), {0, 0, 0, 128}};
        const auto handle = ShaderRecompiler::PrepareShader(request);
        snapshot.prepared->pending = true;
        auto settle = std::async(std::launch::async, SettleLater, std::ref(*snapshot.prepared), std::vector<AgcDriver::DriverDetail::PreparedShaders::Entry>{{0, handle}}, nullptr);
        if (source) Require(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request) == handle, "first use did not wait for registration preparation");
        else static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 0, request));
        settle.get();
        Require(snapshot.prepared->entries.size() == 1, "first use prepared a pending registration again");
    }
    AgcDriver::DriverDetail::ShaderSnapshot snapshot{address, 0, 0, {code.begin(), code.end()}, {}};
    snapshot.header.resize(sizeof(Shader));
    snapshot.prepared->pending = true;
    auto settle = std::async(std::launch::async, SettleLater, std::ref(*snapshot.prepared), std::vector<AgcDriver::DriverDetail::PreparedShaders::Entry>{}, InjectedFailure());
    {
        AgcDriver::DriverDetail::ShaderPreparationTransaction transaction;
        ExpectFailure([&] { static_cast<void>(transaction.Read(snapshot)); }, "injected registration preparation failure");
    }
    settle.get();
    ShaderRecompiler::RecompileRequest request{{ShaderRecompiler::ShaderStage::Compute, address, snapshot.code, 0, {}}, {32, 0, users, compute, {}, {}, {}}, device.ComputeTarget(32), {0, 0, 0, 128}};
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request)); }, "injected registration preparation failure");
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 0, request)); }, "injected registration preparation failure");
}

void UnsupportedTypeRegistration() {
    alignas(256) static const std::array<std::uint32_t, 1> code{0xbf810000u};
    struct Header {
        Shader shader{};
        std::array<ShaderRegister, 8> registers{{{0, 0x1218}, {1, 0x40104004}, {2, 0xa0}, {5, 0}, {3, 6}, {4, 0xc}, {6, 0}, {7, 0}}};
    } header;
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(header);
    header.shader.shader_size = sizeof(code);
    header.shader.code = code.data();
    header.shader.sh_registers = header.registers.data();
    header.shader.num_sh_registers = header.registers.size();
    header.shader.type = 8;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

void RegistrationWithoutSpecials() {
    alignas(256) std::array<std::uint32_t, 1> code{0xbf810000u};
    struct Header {
        Shader shader{};
        std::array<ShaderRegister, 7> registers{};
    } header;
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(header);
    header.shader.shader_size = sizeof(code);
    header.shader.code = code.data();
    header.shader.sh_registers = header.registers.data();
    header.shader.num_sh_registers = header.registers.size();
    header.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)}, {0x207, 1}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, 0}}};
    AgcDriverRegisterShader_nid_postfix(&header.shader);
    std::vector<std::uint32_t> commands;
    for (const auto reg : header.registers) commands.insert(commands.end(), {0xc0017600u, reg.offset, reg.value});
    for (const std::uint32_t initiator : {0x8041u, 0x41u}) commands.insert(commands.end(), {0xc0031500u, 1, 1, 1, initiator});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    sceAgcDriverSubmitAcb(0x20, &packet);
    AgcDriverWaitIdle_nid_postfix();
}

void Registration(bool indirect) {
    UnsupportedTypeRegistration();
    alignas(256) std::array<std::uint32_t, 1> code{0xbf810000u};
    struct Header {
        Shader shader{};
        std::array<ShaderRegister, 9> registers{};
        std::array<ShaderRegister, 2> context{{{0x1b6, 0}, {0x1b6, 1}}};
        ShaderSpecialRegs specials{};
        std::array<std::byte, 8> specialTail{};
        std::array<ShaderSemantic, 2> inputs{};
        std::array<ShaderSemantic, 1> outputs{};
        std::array<ShaderSemantic, 2> inputAlias{};
    } header;
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(header);
    header.shader.shader_size = sizeof(code);
    header.shader.code = code.data();
    header.shader.sh_registers = header.registers.data();
    header.shader.num_sh_registers = header.registers.size();
    header.shader.cx_registers = header.context.data();
    header.shader.num_cx_registers = header.context.size();
    header.shader.specials = &header.specials;
    header.shader.input_semantics = header.inputs.data();
    header.shader.num_input_semantics = header.inputs.size();
    header.shader.output_semantics = header.outputs.data();
    header.shader.num_output_semantics = header.outputs.size();
    header.specials.dispatch_modifier = 0x8000;
    header.specialTail.fill(std::byte{0x5a});
    header.inputs[0].semantic = 3;
    header.inputs[0].size_in_elements = 4;
    header.inputs[1].semantic = 7;
    header.inputs[1].hardware_mapping = 1;
    header.outputs[0].semantic = 11;
    header.outputs[0].size_in_elements = 2;
    header.inputAlias = header.inputs;
    header.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)}, {0x207, 2}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, 0}, {0x207, 1}, {0x207, 1}}};
    const auto threadRegisterIndex = header.registers.size() - 1;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
    std::barrier start(4);
    std::vector<std::future<void>> registrations;
    for (unsigned worker = 0; worker < 4; ++worker) {
        registrations.push_back(std::async(std::launch::async, [&] {
            start.arrive_and_wait();
            for (unsigned iteration = 0; iteration < 8; ++iteration) {
                AgcDriverRegisterShader_nid_postfix(&header.shader);
                AgcDriverResolveShaderAbi_nid_postfix(&header.shader, {}, {});
            }
        }));
    }
    for (auto& registration : registrations) registration.get();
    Shader copy = header.shader;
    copy.user_data = reinterpret_cast<ShaderUserData*>(&copy);
    AgcDriverResolveShaderAbi_nid_postfix(&copy, {}, {});
    alignas(Shader) std::array<std::byte, sizeof(Shader)> padded{};
    std::memcpy(padded.data(), &copy, sizeof(Shader));
    padded.back() = std::byte{0x7d};
    AgcDriverResolveShaderAbi_nid_postfix(reinterpret_cast<Shader*>(padded.data()), {}, {});
    copy.target ^= 1u;
    ExpectFailure([&] { AgcDriverResolveShaderAbi_nid_postfix(&copy, {}, {}); }, "replaced shader header");
    Header moved{};
    const auto relocate = [&] {
        moved = header;
        moved.shader.sh_registers = moved.registers.data();
        moved.shader.cx_registers = moved.context.data();
        moved.shader.specials = &moved.specials;
        moved.shader.input_semantics = moved.inputs.data();
        moved.shader.output_semantics = moved.outputs.data();
    };
    relocate();
    AgcDriverResolveShaderAbi_nid_postfix(&moved.shader, {}, {});
    moved.shader.cx_registers = header.context.data();
    moved.shader.output_semantics = header.outputs.data();
    moved.shader.user_data = reinterpret_cast<ShaderUserData*>(&moved);
    AgcDriverResolveShaderAbi_nid_postfix(&moved.shader, {}, {});
    const auto rejectRelocated = [&](auto mutate) {
        relocate();
        mutate(moved);
        ExpectFailure([&] { AgcDriverResolveShaderAbi_nid_postfix(&moved.shader, {}, {}); }, "replaced shader header");
    };
    rejectRelocated([&](Header& changed) { ++changed.registers[threadRegisterIndex].value; });
    rejectRelocated([](Header& changed) { ++changed.context.back().value; });
    rejectRelocated([](Header& changed) { changed.specials.ge_user_vgpr_en.value ^= 1u; });
    rejectRelocated([](Header& changed) { changed.inputs.back().hardware_mapping ^= 1u; });
    rejectRelocated([](Header& changed) { changed.outputs.front().semantic ^= 1u; });
    rejectRelocated([](Header& changed) { ++changed.shader.sh_registers; });
    rejectRelocated([](Header& changed) { changed.shader.input_semantics = changed.inputAlias.data(); });
    rejectRelocated([](Header& changed) { changed.shader.specials = &changed.specials + 1; });
    rejectRelocated([](Header& changed) { changed.shader.sh_registers = nullptr; });
    rejectRelocated([](Header& changed) { changed.shader.input_semantics = nullptr; });
    rejectRelocated([](Header& changed) { changed.shader.specials = nullptr; });
    rejectRelocated([](Header& changed) { changed.shader.target ^= 1u; });
    rejectRelocated([](Header& changed) { changed.shader.header_size = sizeof(Shader); });
    ++header.registers[threadRegisterIndex].value;
    rejectRelocated([](Header&) {});
    --header.registers[threadRegisterIndex].value;
    relocate();
    AgcDriverResolveShaderAbi_nid_postfix(&moved.shader, {}, {});
    header.shader.special_sizes_bytes = sizeof(header.specials) + sizeof(header.specialTail);
    AgcDriverRegisterShader_nid_postfix(&header.shader);
    relocate();
    AgcDriverResolveShaderAbi_nid_postfix(&moved.shader, {}, {});
    rejectRelocated([](Header& changed) { changed.specialTail.back() ^= std::byte{1}; });
    header.shader.special_sizes_bytes = 0;
    header.shader.header_size = offsetof(Header, inputs) + sizeof(ShaderSemantic);
    AgcDriverRegisterShader_nid_postfix(&header.shader);
    Header truncated = header;
    truncated.shader.input_semantics = truncated.inputs.data();
    ExpectFailure([&] { AgcDriverResolveShaderAbi_nid_postfix(&truncated.shader, {}, {}); }, "replaced shader header");
    header.shader.header_size = sizeof(header);
    AgcDriverRegisterShader_nid_postfix(&header.shader);
    header.registers[1].value |= 0x100u;
    ExpectFailure([&] { AgcDriverRegisterShader_nid_postfix(&header.shader); }, "invalid registered program address");
    header.registers[1].value &= 0xffu;
    ++header.registers[0].value;
    ExpectFailure([&] { AgcDriverRegisterShader_nid_postfix(&header.shader); }, "entry point is outside shader code");
    --header.registers[0].value;
    header.shader.num_sh_registers = 255;
    ExpectFailure([&] { AgcDriverRegisterShader_nid_postfix(&header.shader); }, "truncated shader metadata");
    header.shader.num_sh_registers = header.registers.size();
    header.registers[threadRegisterIndex].value = 0;
    ExpectFailure([&] { AgcDriverRegisterShader_nid_postfix(&header.shader); }, "must be nonzero");
    header.registers[threadRegisterIndex].value = 1;
    code[0] = 0xffffffffu;
    ExpectFailure([&] { AgcDriverRegisterShader_nid_postfix(&header.shader); }, "");
    std::vector<std::uint32_t> commands;
    for (const auto reg : header.registers) commands.insert(commands.end(), {0xc0017600u, reg.offset, reg.value});
    const std::array<std::uint32_t, 3> arguments{1, 1, 1};
    const auto argumentAddress = reinterpret_cast<std::uintptr_t>(arguments.data());
    if (indirect) commands.insert(commands.end(), {0xc0021600u, static_cast<std::uint32_t>(argumentAddress), static_cast<std::uint32_t>(argumentAddress >> 32u), 0x8041});
    else commands.insert(commands.end(), {0xc0031500u, 1, 1, 1, 0x8041});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    for (std::uint32_t iteration = 0; iteration < 3; ++iteration) {
        sceAgcDriverSubmitAcb(0x20, &packet);
        AgcDriverWaitIdle_nid_postfix();
    }
    std::uint32_t destination = 0;
    const auto destinationAddress = reinterpret_cast<std::uintptr_t>(&destination);
    commands[threadRegisterIndex * 3 + 2] = 2;
    commands.insert(commands.end(), {0xc0033700u, 0x00100200u, static_cast<std::uint32_t>(destinationAddress), static_cast<std::uint32_t>(destinationAddress >> 32u), 7});
    packet = Packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    sceAgcDriverSubmitAcb(0x20, &packet);
    ExpectFailure([] { AgcDriverWaitIdle_nid_postfix(); }, "artifact is missing");
    ExpectFailure([] { AgcDriverWaitIdle_nid_postfix(); }, "artifact is missing");
    ExpectFailure([] { AgcDriverShutdown_nid_postfix(); }, "artifact is missing");
    Require(destination == 0, "failed dispatch executed a subsequent memory write");
}

void DeferredRegistration(bool preparationFailure = false) {
    alignas(256) std::array<std::uint32_t, 2> code{0xbe842104u, 0xbf810000u};
    if (preparationFailure) code[0] = 0xbe802004u;
    struct Header {
        Shader shader{};
        std::array<ShaderRegister, 8> registers{};
        ShaderSpecialRegs specials{};
    } header;
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(header);
    header.shader.shader_size = sizeof(code);
    header.shader.code = code.data();
    header.shader.sh_registers = header.registers.data();
    header.shader.num_sh_registers = header.registers.size();
    header.shader.specials = &header.specials;
    header.specials.dispatch_modifier = 0x8000;
    header.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)}, {0x207, 1}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, 12}, {0x207, 1}}};
    AgcDriverRegisterShader_nid_postfix(&header.shader);
    std::vector<std::uint32_t> commands;
    for (const auto reg : header.registers) commands.insert(commands.end(), {0xc0017600u, reg.offset, reg.value});
    commands.insert(commands.end(), {0xc0031500u, 1, 1, 1, 0x8041});
    std::uint32_t destination = 0;
    const auto destinationAddress = reinterpret_cast<std::uintptr_t>(&destination);
    commands.insert(commands.end(), {0xc0033700u, 0x00100200u, static_cast<std::uint32_t>(destinationAddress), static_cast<std::uint32_t>(destinationAddress >> 32u), 7u});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    sceAgcDriverSubmitAcb(0x20, &packet);
    const auto reason = preparationFailure ? "unsupported dynamic s_setpc_b64 at program counter 0x0" : "not statically resolvable";
    ExpectFailure([] { AgcDriverWaitIdle_nid_postfix(); }, reason);
    ExpectFailure([] { AgcDriverShutdown_nid_postfix(); }, reason);
    Require(destination == 0u, "failed preparation executed a subsequent memory write");
}

void AperturePreparationClassification(AgcDriver::VulkanDevice& device) {
    using namespace ShaderRecompiler;
    std::array<std::uint32_t, 2> code{0xbeea04ebu, 0xbf810000u};
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x35800u, code, 0, {}};
    request.context.waveSize = 32;
    request.context.compute = ShaderComputeStageInfo{{1, 1, 1}, 0, {false, false, false}, false, 1, {}};
    request.target = device.ComputeTarget(32);
    request.layout.pushConstantSizeBytes = 128;
    request.useCache = false;
    for (std::uint32_t source = 234u; source <= 238u; ++source) {
        code[0] = 0xbeea0400u | source;
        bool rejected = false;
        try { static_cast<void>(GetResourcePlan(request)); }
        catch (const UnsupportedInstructionError& error) {
            Require(source >= 235u, "a reserved scalar source became deferrable");
            Require(std::string(error.what()).find("unsupported scalar source operand code " + std::to_string(source)) != std::string::npos, "aperture preparation lost its scalar source diagnostic");
            rejected = true;
        } catch (const std::invalid_argument& error) {
            Require(source == 234u, "an aperture scalar source remained a malformed-input error");
            Require(std::string(error.what()).find("unsupported scalar source operand code 234") != std::string::npos, "reserved scalar source lost its diagnostic");
            rejected = true;
        }
        Require(rejected, "an unsupported scalar source compiled successfully");
    }
}

void DeferredUndecodableRegistration(bool aperture = false) {
    alignas(256) std::array<std::uint32_t, 3> code{0xbe842104u, 0xf1949f05u, 0x00060018u};
    if (aperture) code = {0xbeea04ebu, 0xbf810000u, 0xbf810000u};
    struct Header {
        Shader shader{};
        std::array<ShaderRegister, 8> registers{};
        ShaderSpecialRegs specials{};
    } header;
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(header);
    header.shader.shader_size = sizeof(code);
    header.shader.code = code.data();
    header.shader.sh_registers = header.registers.data();
    header.shader.num_sh_registers = header.registers.size();
    header.shader.specials = &header.specials;
    header.specials.dispatch_modifier = 0x8000;
    header.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)}, {0x207, 1}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, 12}, {0x207, 1}}};
    AgcDriverRegisterShader_nid_postfix(&header.shader);
    std::vector<std::uint32_t> commands;
    for (const auto reg : header.registers) commands.insert(commands.end(), {0xc0017600u, reg.offset, reg.value});
    commands.insert(commands.end(), {0xc0031500u, 1, 1, 1, 0x8041});
    std::uint32_t destination = 0;
    const auto destinationAddress = reinterpret_cast<std::uintptr_t>(&destination);
    commands.insert(commands.end(), {0xc0033700u, 0x00100200u, static_cast<std::uint32_t>(destinationAddress), static_cast<std::uint32_t>(destinationAddress >> 32u), 7u});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    sceAgcDriverSubmitAcb(0x20, &packet);
    const auto reason = aperture ? "unsupported scalar source operand code 235" : "unsupported MIMG opcode";
    ExpectFailure([] { AgcDriverWaitIdle_nid_postfix(); }, reason);
    ExpectFailure([] { AgcDriverShutdown_nid_postfix(); }, reason);
    Require(destination == 0u, "a deferred unsupported dispatch executed a subsequent memory write");
}

}

int main(int argc, char** argv) {
    try {
        Require(argc == 1 || (argc == 2 && (std::string_view(argv[1]) == "--indirect" || std::string_view(argv[1]) == "--fail-before-registration" || std::string_view(argv[1]) == "--deferred" || std::string_view(argv[1]) == "--deferred-undecodable" || std::string_view(argv[1]) == "--deferred-aperture" || std::string_view(argv[1]) == "--preparation-failure")), "invalid test arguments");
        if (argc == 2 && std::string_view(argv[1]) == "--deferred-aperture") {
            auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
            AperturePreparationClassification(*device);
            device.reset();
            DeferredUndecodableRegistration(true);
            std::cout << "deferred aperture shader tests passed\n";
            return 0;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--deferred" || std::string_view(argv[1]) == "--preparation-failure")) {
            if (!OpenVulkanTestDevice()) return VulkanTestSkipped;
            DeferredRegistration(std::string_view(argv[1]) == "--preparation-failure");
            std::cout << "deferred shader preparation tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--deferred-undecodable") {
            if (!OpenVulkanTestDevice()) return VulkanTestSkipped;
            DeferredUndecodableRegistration();
            std::cout << "deferred undecodable shader tests passed\n";
            return 0;
        }
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device);
        FailureCapture(*device, argc == 2 && std::string_view(argv[1]) == "--indirect");
        PendingRegistrationPreparation(*device);
        Require(argc != 2 || std::string_view(argv[1]) != "--fail-before-registration", "injected failure before registration");
        PrepareMultisampledStorage(*device);
        device.reset();
        RegistrationWithoutSpecials();
        Registration(argc == 2);
        std::cout << "prepared shader and transactional registration tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (const std::exception& shutdownError) {
            std::cerr << shutdownError.what() << '\n';
        }
        return 1;
    }
}
