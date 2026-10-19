// The pipeline-request precheck (Driver::capturePipelineRequest, APS5_PIPELINE_REQUESTS): a kept
// ComputePipelineInputs must match a result only when its fingerprint is the one recorded, so a saved
// request is never skipped for a result that would have been saved under another fingerprint.
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/PipelineFingerprint.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::ComputePipelineFingerprint;
using AgcDriver::ComputePipelineInputs;
using ShaderRecompiler::DescriptorBinding;
using ShaderRecompiler::DescriptorKind;
using ShaderRecompiler::DescriptorRole;
using ShaderRecompiler::RecompileResult;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

DescriptorBinding Binding(DescriptorKind kind, DescriptorRole role, std::uint32_t number, std::uint32_t count) {
    DescriptorBinding binding{};
    binding.kind = kind;
    binding.role = role;
    binding.descriptorSet = 0;
    binding.binding = number;
    binding.count = count;
    return binding;
}

RecompileResult Result() {
    RecompileResult result;
    std::vector<std::uint32_t> words(4096);
    for (std::size_t index = 0; index < words.size(); ++index) words[index] = static_cast<std::uint32_t>(index * 2654435761u);
    words[0] = 0x07230203u;
    result.spirv = std::move(words);
    result.specialization = {{3u, 7u}, {9u, 0x3f800000u}};
    result.hostSubgroupSize = 32u;
    result.workgroupMemoryDwords = 256u;
    result.pushConstants.resize(16);
    result.bindings = {Binding(DescriptorKind::StorageBuffer, DescriptorRole::GuestBuffers, 0u, 4u), Binding(DescriptorKind::StorageBuffer, DescriptorRole::ShaderData, 1u, 1u), Binding(DescriptorKind::SampledImage, DescriptorRole::ImageTable, 2u, 64u)};
    result.variantId = 0x1234u;
    return result;
}

// A layout key standing in for the planned layout: a function of what planComputeLayout reads.
std::vector<std::uint32_t> Layout(const RecompileResult& result) {
    std::vector<std::uint32_t> key;
    for (const auto& binding : result.bindings) key.insert(key.end(), {binding.descriptorSet, binding.binding, static_cast<std::uint32_t>(binding.kind), static_cast<std::uint32_t>(binding.role), binding.count, binding.readOnly ? 1u : 0u});
    return key;
}

std::uint64_t Fingerprint(const RecompileResult& result) {
    const auto layout = Layout(result);
    return ComputePipelineFingerprint(result, layout);
}

template<typename TChange>
void ExpectMiss(const RecompileResult& base, const ComputePipelineInputs& kept, TChange change, const char* what) {
    auto changed = base;
    change(changed);
    Require(!kept.Matches(changed), std::string("precheck matched a result with a changed ") + what);
    Require(Fingerprint(changed) != Fingerprint(base), std::string("test setup: the fingerprint ignores ") + what);
}

template<typename TChange>
void ExpectHit(const RecompileResult& base, const ComputePipelineInputs& kept, TChange change, const char* what) {
    auto changed = base;
    change(changed);
    Require(kept.Matches(changed), std::string("precheck missed a result that differs only in ") + what);
    Require(Fingerprint(changed) == Fingerprint(base), std::string("precheck matched although the fingerprint differs (") + what + ")");
}

void Run() {
    const auto base = Result();
    const auto kept = ComputePipelineInputs::Of(base);
    Require(kept.Matches(base), "precheck missed the result it was taken from");

    // What the fingerprint does not read may differ.
    ExpectHit(base, kept, [](RecompileResult& result) { result.shaderDataUserWords = {1u, 2u, 3u}; }, "shader data user words");
    ExpectHit(base, kept, [](RecompileResult& result) { result.bindings[0].guestDescriptor = {0xdeadbeefu}; }, "a binding's guest descriptor");
    ExpectHit(base, kept, [](RecompileResult& result) { result.pushConstants.assign(32, std::byte{1}); }, "push constant bytes");

    // Everything it reads must not.
    ExpectMiss(base, kept, [](RecompileResult& result) { result.specialization[1].value ^= 1u; }, "specialization value");
    ExpectMiss(base, kept, [](RecompileResult& result) { result.specialization.push_back({11u, 0u}); }, "specialization count");
    ExpectMiss(base, kept, [](RecompileResult& result) { result.hostSubgroupSize = 64u; }, "host subgroup size");
    ExpectMiss(base, kept, [](RecompileResult& result) { result.workgroupMemoryDwords = 512u; }, "workgroup memory");
    ExpectMiss(base, kept, [](RecompileResult& result) { result.pushConstants.clear(); }, "push flag");
    ExpectMiss(base, kept, [](RecompileResult& result) { result.bindings.pop_back(); }, "binding count");
    ExpectMiss(base, kept, [](RecompileResult& result) { result.bindings[2].count = 128u; }, "binding element count");
    ExpectMiss(base, kept, [](RecompileResult& result) { result.bindings[1].role = DescriptorRole::FlattenedSrt; }, "binding role");
    ExpectMiss(base, kept, [](RecompileResult& result) { result.bindings[2].kind = DescriptorKind::StorageImage; }, "binding kind");
    ExpectMiss(base, kept, [](RecompileResult& result) { result.bindings[0].readOnly = true; }, "binding read-only flag");
    ExpectMiss(base, kept, [](RecompileResult& result) { result.bindings[0].binding = 7u; }, "binding number");
    ExpectMiss(base, kept, [](RecompileResult& result) { result.bindings[0].descriptorSet = 1u; }, "descriptor set");
    // A written word copies the shared module first (the kept copy holds the old storage), so it misses.
    ExpectMiss(base, kept, [](RecompileResult& result) { result.spirv[100] ^= 1u; }, "SPIR-V word");
    ExpectMiss(base, kept, [](RecompileResult& result) { result.spirv.resize(result.spirv.size() - 1); }, "SPIR-V length");

    // Equal words in other storage only miss the precheck (the fingerprint path then decides).
    auto copied = base;
    copied.spirv = base.spirv.Copy();
    Require(!kept.Matches(copied), "precheck matched SPIR-V in other storage by address");
    Require(Fingerprint(copied) == Fingerprint(base), "test setup: copied SPIR-V changed the fingerprint");

    // The kept storage stays alive and unchanged when every result that shared it is gone.
    const auto address = base.spirv.data();
    ComputePipelineInputs survivor;
    {
        auto temporary = Result();
        survivor = ComputePipelineInputs::Of(temporary);
        temporary.spirv[5] ^= 1u;
        Require(temporary.spirv.data() != survivor.spirv.data(), "writing a shared module did not copy it first");
    }
    Require(survivor.spirv.size() == base.spirv.size() && survivor.spirv[5] == base.spirv[5], "the kept module changed under the precheck");
    Require(address == base.spirv.data(), "test setup: base storage moved");

    // Cost on a giant module (0x80001c0d00 has 1.27M words): the precheck against the fingerprint it skips.
    auto giant = Result();
    std::vector<std::uint32_t> words(1270000);
    for (std::size_t index = 0; index < words.size(); ++index) words[index] = static_cast<std::uint32_t>(index * 2654435761u);
    giant.spirv = std::move(words);
    const auto giantKept = ComputePipelineInputs::Of(giant);
    const auto time = [](auto action) {
        const auto started = std::chrono::steady_clock::now();
        action();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
    };
    bool matched = false;
    std::uint64_t fingerprint = 0;
    const auto precheckUs = time([&] { for (int repeat = 0; repeat < 100; ++repeat) matched = giantKept.Matches(giant); }) / 100.0;
    const auto fingerprintUs = time([&] { fingerprint = Fingerprint(giant); });
    Require(matched && fingerprint != 0, "giant precheck setup");
    std::printf("pipeline request precheck: 1.27M-word module: precheck %.2f us, layout + fingerprint %.0f us\n", precheckUs, fingerprintUs);
}

}

int main() {
    try {
        Run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "pipeline request precheck: %s\n", error.what());
        return 1;
    }
    std::printf("pipeline request precheck: ok\n");
    return 0;
}
