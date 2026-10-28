// agc_shader_replay <shader_*.req>...: replays compute recompile requests the driver saved with
// APS5_DUMP_SHADERS=1, running the same resource analysis and recompile steps without the game.
#include "ControlFlow/GraphBuilder.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "prx/libSceAgcDriver/tools/RecordedRequests.hpp"
#include "ControlFlow/Structurizer.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Optimization/RequestMemoryView.hpp"
#include "Recompiler.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/PipelineFingerprint.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ResultFootprint.hpp"

#if ANYPS5_ENABLE_SPIRV_TOOLS
#include "spirv-tools/libspirv.hpp"
#endif
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <set>
#include <tuple>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string ReadText(const char* path) {
    std::ifstream file(path, std::ios::binary);
    std::ostringstream stream;
    stream << file.rdbuf();
    auto text = stream.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
    return text;
}

// The error text without the serialized request that Recompile appends.
std::string FirstLine(const std::string& text) {
    const auto request = text.find("\nRecompileRequest:");
    auto shown = request == std::string::npos ? text : text.substr(0, request);
    if (shown.size() > 2000) shown.resize(2000);
    return shown;
}

bool g_disassemble = false;
bool g_assembly = false;
bool g_memory = false;
bool g_spirv = false;
bool g_prepare = false;
bool g_materialize = false;
bool g_graph = false;
bool g_maintenance8 = false;
bool g_robust2 = false;
bool g_waveLdsMemoryBarriers = false;
bool g_deviceScopeLdsMemoryBarriers = false;
bool g_code = false;
bool g_prepareOnly = false;
bool g_pipelineOnly = false;
int g_requests = 0;
bool g_noReference = false;
bool g_variants = false;
bool g_describe = false;
bool g_nativeTarget = false;
bool g_bindless = false;
std::string g_expectedSpirv;

void ConfigurePipelineReplay(int argc, char** argv) {
    std::string cache;
    bool prepare = false;
    bool maintenance = false;
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        if (option == "--pipeline-only") g_pipelineOnly = true;
        if (option == "--no-reference") g_noReference = true;
        if (option == "--fingerprint-variants") g_variants = true;
        if (option == "--describe") g_describe = true;
        if (option == "--native-target") g_nativeTarget = true;
        if (option == "--prepare-only") prepare = true;
        if (option == "--maintenance8") maintenance = true;
        if (option == "--expect-spv") {
            if (!g_expectedSpirv.empty() || i + 1 == argc || std::string(argv[i + 1]).starts_with("--")) throw std::runtime_error("--expect-spv requires one reference file");
            g_expectedSpirv = argv[++i];
            if (g_expectedSpirv.empty()) throw std::runtime_error("--expect-spv file is empty");
        }
        if (option == "--pipeline-cache") {
            if (!cache.empty() || i + 1 == argc || std::string(argv[i + 1]).starts_with("--")) throw std::runtime_error("--pipeline-cache requires one explicit directory");
            cache = argv[++i];
            if (cache.empty()) throw std::runtime_error("--pipeline-cache directory is empty");
        }
    }
    if (!g_pipelineOnly) {
        if (!cache.empty() || !g_expectedSpirv.empty() || g_noReference) throw std::runtime_error("--pipeline-cache, --expect-spv and --no-reference require --pipeline-only");
        return;
    }
    if (prepare || maintenance) throw std::runtime_error("--pipeline-only cannot be combined with --prepare-only or --maintenance8");
    if (cache.empty()) throw std::runtime_error("--pipeline-only requires --pipeline-cache");
    if (std::getenv("ANYPS5_NO_SHADER_CACHE") != nullptr) throw std::runtime_error("--pipeline-only requires ANYPS5_NO_SHADER_CACHE to be unset");
    const std::filesystem::path directory(cache);
    if (!directory.is_absolute() || !std::filesystem::is_directory(directory)) throw std::runtime_error("--pipeline-cache must be an existing absolute private directory");
    const auto* driverCache = std::getenv("__GL_SHADER_DISK_CACHE_PATH");
    if (driverCache == nullptr || *driverCache == '\0') throw std::runtime_error("--pipeline-only requires an explicit fresh __GL_SHADER_DISK_CACHE_PATH");
    const std::filesystem::path driverDirectory(driverCache);
    if (!driverDirectory.is_absolute() || !std::filesystem::is_directory(driverDirectory) || !std::filesystem::is_empty(driverDirectory) || std::filesystem::equivalent(directory, driverDirectory)) throw std::runtime_error("__GL_SHADER_DISK_CACHE_PATH must be a separate existing absolute empty directory");
    const auto selected = std::filesystem::canonical(directory).string();
#ifdef _WIN32
    const auto error = _putenv_s("ANYPS5_SHADER_CACHE_DIR", selected.c_str());
#else
    const auto error = setenv("ANYPS5_SHADER_CACHE_DIR", selected.c_str(), 1);
#endif
    if (error != 0) throw std::runtime_error("could not select the private pipeline cache directory");
    // The tool's device is headless, which since d65cc07a1 starts with an empty VkPipelineCache: load the
    // private directory's file so a pipeline the game already has is a hit, as before.
#ifdef _WIN32
    if (_putenv_s("APS5_PIPELINE_CACHE_LOAD", "1") != 0) throw std::runtime_error("could not request the pipeline cache load");
#else
    if (setenv("APS5_PIPELINE_CACHE_LOAD", "1", 1) != 0) throw std::runtime_error("could not request the pipeline cache load");
#endif
    std::printf("pipeline-only cache: %s\nNVIDIA cache: %s\n", selected.c_str(), driverDirectory.string().c_str());
}

void CheckExpectedSpirv(std::span<const std::uint32_t> words) {
    std::ifstream file(g_expectedSpirv, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(words.size_bytes())) throw std::runtime_error("pipeline-only expected SPIR-V file is unreadable or has a different size");
    file.seekg(0);
    std::vector<std::uint32_t> expected(words.size());
    file.read(reinterpret_cast<char*>(expected.data()), static_cast<std::streamsize>(words.size_bytes()));
    if (!file || !std::equal(words.begin(), words.end(), expected.begin())) throw std::runtime_error("pipeline-only SPIR-V differs from the expected ordinary replay");
    std::printf("  expected SPIR-V matches: %zu bytes\n", words.size_bytes());
}

// The .fp sidecar APS5_PIPELINE_REQUESTS saved next to a request: "key value" lines (fingerprint, the
// per-component hashes, counts) and the game's specialization constants and bindings.
struct Sidecar {
    bool present = false;
    std::map<std::string, std::string> values;
    std::vector<std::string> specs;
    std::vector<std::string> bindings;
    // "probe a|p <address> <bytes> <answer>": the dispatch's accessibility and pending-write probes.
    std::map<std::tuple<char, std::uint64_t, std::uint64_t>, bool> probes;
};

Sidecar ReadSidecar(const char* path) {
    Sidecar sidecar;
    std::filesystem::path file(path);
    file.replace_extension(".fp");
    std::ifstream in(file);
    if (!in) return sidecar;
    sidecar.present = true;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto space = line.find(' ');
        if (space == std::string::npos) continue;
        const auto key = line.substr(0, space);
        if (key == "spec") sidecar.specs.push_back(line.substr(space + 1));
        else if (key == "binding") sidecar.bindings.push_back(line.substr(space + 1));
        else if (key == "probe") {
            std::istringstream fields(line.substr(space + 1));
            char kind = 0;
            std::string address;
            std::string bytes;
            unsigned answer = 0;
            if (fields >> kind >> address >> bytes >> answer) sidecar.probes[{kind, std::stoull(address, nullptr, 16), std::stoull(bytes, nullptr, 16)}] = answer != 0;
        } else sidecar.values[key] = line.substr(space + 1);
    }
    return sidecar;
}

std::string Hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

// The components of `result` that differ from the sidecar's, with the first differing values; empty
// when the sidecar has no component hashes (requests saved before they were recorded).
std::string DescribeDifference(const Sidecar& sidecar, const ShaderRecompiler::RecompileResult& result, const AgcDriver::ComputePipelineComponents& components, std::span<const std::uint32_t> layout) {
    const std::pair<const char*, std::uint64_t> parts[] = {{"spirv", components.spirv}, {"specialization", components.specialization}, {"subgroup", components.subgroup}, {"workgroup_memory", components.workgroupMemory}, {"push", components.push}, {"bindings", components.bindings}};
    std::string text;
    for (const auto& [name, value] : parts) {
        const auto found = sidecar.values.find(std::string("hash_") + name);
        if (found == sidecar.values.end()) return {};
        if (found->second == Hex(value)) continue;
        text += text.empty() ? "" : ", ";
        text += name;
    }
    std::istringstream mine(AgcDriver::ComputePipelineDescription(result, components, layout));
    std::vector<std::string> specs;
    std::vector<std::string> bindings;
    std::map<std::string, std::string> values;
    for (std::string line; std::getline(mine, line);) {
        const auto space = line.find(' ');
        const auto key = line.substr(0, space);
        if (key == "spec") specs.push_back(line.substr(space + 1));
        else if (key == "binding") bindings.push_back(line.substr(space + 1));
        else values[key] = line.substr(space + 1);
    }
    for (const char* key : {"spirv_words", "subgroup", "workgroup_memory", "push", "specialization_id", "variant_id", "layout"}) {
        const auto game = sidecar.values.find(key);
        if (game != sidecar.values.end() && game->second != values[key]) text += std::string("; ") + key + " game " + game->second + " replay " + values[key];
    }
    const auto listDiff = [&](const char* name, const std::vector<std::string>& game, const std::vector<std::string>& replay) {
        if (game == replay) return;
        text += std::string("; ") + name + ": game " + std::to_string(game.size()) + " replay " + std::to_string(replay.size());
        int shown = 0;
        for (std::size_t i = 0; i < std::max(game.size(), replay.size()) && shown < 6; ++i) {
            const auto g = i < game.size() ? game[i] : std::string("-");
            const auto r = i < replay.size() ? replay[i] : std::string("-");
            if (g == r) continue;
            text += " [" + std::to_string(i) + "] game '" + g + "' replay '" + r + "'";
            ++shown;
        }
    };
    listDiff("spec", sidecar.specs, specs);
    listDiff("bindings", sidecar.bindings, bindings);
    return text;
}

void PrintComponents(const char* label, const ShaderRecompiler::RecompileResult& result, const Sidecar& sidecar, AgcDriver::VulkanDevice& device) {
    const auto layout = device.PlannedComputeLayoutKey(result);
    const auto components = AgcDriver::ComputePipelineFingerprintComponents(result, layout);
    const auto game = sidecar.values.find("fingerprint");
    const bool match = game != sidecar.values.end() && game->second == Hex(components.whole);
    std::printf("  %-28s fingerprint %s%s spirv %s spec %s(%zu, id %llx) subgroup %s(%u) wgmem %s(%u) push %s bindings %s(%zu)\n", label, Hex(components.whole).c_str(), match ? " = GAME" : "", Hex(components.spirv).c_str(), Hex(components.specialization).c_str(), result.specialization.size(),
                static_cast<unsigned long long>(result.specializationId), Hex(components.subgroup).c_str(), result.hostSubgroupSize, Hex(components.workgroupMemory).c_str(), result.workgroupMemoryDwords, Hex(components.push).c_str(), Hex(components.bindings).c_str(), result.bindings.size());
    if (!match && sidecar.present) {
        const auto difference = DescribeDifference(sidecar, result, components, layout);
        if (!difference.empty()) std::printf("    differs from the game in: %s\n", difference.c_str());
    }
    if (g_describe) {
        std::istringstream lines(AgcDriver::ComputePipelineDescription(result, components, layout));
        for (std::string line; std::getline(lines, line);) std::printf("    | %s\n", line.c_str());
    }
}

// The 4 KiB guest pages that hold a captured word. The game's accessibility probe (ShaderMemory::accessible)
// asks the guest mapping, page by page; a request only carries the words the capture read, so
// RequestMemoryView's probe (every byte captured) says no to a range of a mapped page with unread words,
// e.g. the gaps of a strided image-table key scan, and the materializer then skips what the game did.
// A page with a captured word was mapped when the game read it.
thread_local const std::set<std::uint64_t>* t_capturedPages = nullptr;
thread_local const std::map<std::tuple<char, std::uint64_t, std::uint64_t>, bool>* t_probes = nullptr;
thread_local std::uint64_t t_unrecordedProbes = 0;

bool CapturedPageAccessible(void*, std::uint64_t address, std::uint64_t bytes) {
    if (t_capturedPages == nullptr || bytes == 0 || address > ~std::uint64_t{0} - bytes) return false;
    for (auto page = address & ~std::uint64_t{0xfff}; page < address + bytes; page += 0x1000) {
        if (!t_capturedPages->contains(page)) return false;
    }
    return true;
}

// mode 4: the answers the game's capture got (the sidecar's probe lines); a probe it did not record falls
// back to the page rule (accessibility) or "no pending write", and is counted.
bool RecordedAccessible(void* context, std::uint64_t address, std::uint64_t bytes) {
    if (t_probes != nullptr) {
        if (const auto found = t_probes->find({'a', address, bytes}); found != t_probes->end()) return found->second;
    }
    ++t_unrecordedProbes;
    return CapturedPageAccessible(context, address, bytes);
}

bool RecordedPending(void*, std::uint64_t address, std::uint64_t bytes) {
    if (t_probes != nullptr) {
        if (const auto found = t_probes->find({'p', address, bytes}); found != t_probes->end()) return found->second;
    }
    ++t_unrecordedProbes;
    return false;
}

std::set<std::uint64_t> CapturedPages(std::span<const ShaderRecompiler::MemoryRegion> regions) {
    std::set<std::uint64_t> pages;
    for (const auto& region : regions) {
        if (region.bytes.empty()) continue;
        for (auto page = region.guestAddress & ~std::uint64_t{0xfff}; page < region.guestAddress + region.bytes.size(); page += 0x1000) pages.insert(page);
    }
    return pages;
}

// The dispatch path's materialization: the shader prepared from the request, a prepared invocation, a
// capture over the request's memory and Materialize (Recompile(request) materializes inside one call).
// mode 0: RequestMemoryView's probes (every byte captured); mode 1: every pending-write probe says yes;
// mode 2: no accessibility probe; mode 3: page-granular accessibility like the game's (CapturedPageAccessible);
// mode 4: the game's recorded probe answers (RecordedAccessible, RecordedPending).
std::shared_ptr<const ShaderRecompiler::RecompileResult> MaterializePrepared(const ShaderRecompiler::RecompileRequest& request, int mode, const Sidecar* sidecar = nullptr) {
    const auto handle = ShaderRecompiler::PrepareShader(request);
    if (!handle) throw std::runtime_error("prepared shader handle is absent");
    auto invocationRequest = request;
    invocationRequest.shader.code = ShaderRecompiler::GetPreparedCode(*handle);
    auto invocation = ShaderRecompiler::PreparedShaderInvocation::TryCreate(invocationRequest, handle);
    if (!invocation) throw std::runtime_error("the prepared shader does not match its invocation");
    ShaderRecompiler::RequestMemoryView memory(request.context.memory);
    auto runtime = memory.MakeRuntime(request.context.userData, request.shader.codeAddress);
    if (mode == 1) runtime.pendingWrite = +[](void*, std::uint64_t, std::uint64_t) { return true; };
    if (mode == 2) runtime.accessible = nullptr;
    const auto pages = CapturedPages(request.context.memory);
    if (mode == 3) runtime.accessible = &CapturedPageAccessible;
    if (mode == 4) {
        runtime.accessible = &RecordedAccessible;
        runtime.pendingWrite = &RecordedPending;
    }
    t_capturedPages = &pages;
    t_probes = sidecar != nullptr ? &sidecar->probes : nullptr;
    t_unrecordedProbes = 0;
    struct Reset {
        ~Reset() {
            t_capturedPages = nullptr;
            t_probes = nullptr;
        }
    } reset;
    const auto capture = invocation->Capture(runtime);
    return invocation->Materialize(*capture);
}

// --pipeline-only: each request needs a reference, the byte-exact SPIR-V of an ordinary replay (--expect-spv,
// one request) or the .fp sidecar the driver saved with it (APS5_PIPELINE_REQUESTS), whose fingerprint covers
// everything the pipeline create info takes from the recompile. Returns a refusal text, empty when it matches.
std::string CheckPipelineReference(const char* path, const ShaderRecompiler::RecompileResult& result, AgcDriver::VulkanDevice& device) {
    const auto sidecar = ReadSidecar(path);
    const auto layout = device.PlannedComputeLayoutKey(result);
    const auto components = AgcDriver::ComputePipelineFingerprintComponents(result, layout);
    std::printf("  pipeline fingerprint: %s\n", Hex(components.whole).c_str());
    if (g_noReference) {
        if (sidecar.present) {
            const auto found = sidecar.values.find("fingerprint");
            const bool same = found != sidecar.values.end() && found->second == Hex(components.whole);
            const auto difference = same ? std::string() : DescribeDifference(sidecar, result, components, layout);
            std::printf("  game sidecar (not checked, --no-reference): %s%s%s\n", same ? "fingerprint matches" : "fingerprint differs", difference.empty() ? "" : ": ", difference.c_str());
        }
        std::printf("  no reference (--no-reference): this replay's own SPIR-V; only a hit in the game's cache proves it is the game's pipeline\n");
        return {};
    }
    if (sidecar.present) {
        const auto found = sidecar.values.find("fingerprint");
        if (found == sidecar.values.end()) return "sidecar of " + std::string(path) + " has no fingerprint";
        if (found->second != Hex(components.whole)) {
            const auto difference = DescribeDifference(sidecar, result, components, layout);
            return "fingerprint differs from the game's (" + found->second + " in the .fp of " + path + ")" + (difference.empty() ? std::string() : ": " + difference);
        }
        std::printf("  game fingerprint matches\n");
        return {};
    }
    if (g_expectedSpirv.empty()) return "no reference: give --expect-spv or --no-reference, or keep the request's .fp sidecar next to it";
    CheckExpectedSpirv(result.spirv);
    return {};
}

std::string TargetDifference(const ShaderRecompiler::SpirvTarget& captured, const ShaderRecompiler::SpirvTarget& native) {
    std::string text;
    const auto differs = [&](const char* name, bool different) {
        if (different) text += std::string(" ") + name;
    };
    differs("vulkanVersion", captured.vulkanVersion != native.vulkanVersion);
    differs("spirvVersion", captured.spirvVersion != native.spirvVersion);
    differs("subgroupSize", captured.subgroupSize != native.subgroupSize);
    differs("bdaAbiVersion", captured.bdaAbiVersion != native.bdaAbiVersion);
    differs("supportedCapabilities", !std::ranges::equal(captured.supportedCapabilities, native.supportedCapabilities));
    differs("supportedExtensions", !std::ranges::equal(captured.supportedExtensions, native.supportedExtensions));
    differs("fragmentShaderBarycentricEnabled", captured.fragmentShaderBarycentricEnabled != native.fragmentShaderBarycentricEnabled);
    differs("maxWorkgroupSize", captured.maxWorkgroupSize != native.maxWorkgroupSize);
    differs("maxWorkgroupInvocations", captured.maxWorkgroupInvocations != native.maxWorkgroupInvocations);
    differs("maxWorkgroupSharedMemoryBytes", captured.maxWorkgroupSharedMemoryBytes != native.maxWorkgroupSharedMemoryBytes);
    differs("mesh", captured.mesh.has_value() != native.mesh.has_value());
    differs("tessellation", captured.tessellation.has_value() != native.tessellation.has_value());
    differs("nonConstantImageOffsets", captured.nonConstantImageOffsets != native.nonConstantImageOffsets);
    differs("srgbDecodeFormats", captured.srgbDecodeFormats != native.srgbDecodeFormats);
    differs("narrowSubgroupClock", captured.narrowSubgroupClock != native.narrowSubgroupClock);
    differs("float64DenormPreserve", captured.float64DenormPreserve != native.float64DenormPreserve);
    differs("bindlessImages", captured.bindlessImages != native.bindlessImages);
    differs("spannedWaves", captured.spannedWaves != native.spannedWaves);
    return text;
}

void InspectSpirv(const char* path, std::span<const std::uint32_t> words) {
    if (g_spirv) {
        std::string name(path);
        name = name.substr(name.find_last_of("/\\") + 1) + ".spv";
        std::ofstream file(name, std::ios::binary);
        file.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size_bytes()));
        if (!file) throw std::runtime_error("could not write " + name);
    }
#if ANYPS5_ENABLE_SPIRV_TOOLS
    if (g_disassemble) {
        spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
        std::string text;
        if (tools.Disassemble(words.data(), words.size(), &text, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)) std::printf("%s\n", text.c_str());
    }
#else
    if (g_disassemble) std::printf("  (--dis needs ANYPS5_ENABLE_SPIRV_TOOLS=ON)\n");
#endif
}

bool Replay(const char* path, std::unique_ptr<AgcDriver::VulkanDevice>& device) {
    auto request = AgcDriver::RecordedRequests::Deserialize(ReadText(path));
    if (g_nativeTarget) {
        if (!device) device = std::make_unique<AgcDriver::VulkanDevice>();
        const auto native = device->ComputeTarget(request.request.context.waveSize);
        const auto difference = TargetDifference(request.request.target, native);
        std::printf("  native target (--native-target): %s%s\n", difference.empty() ? "the captured target matches" : "replaced the captured target's", difference.c_str());
        request.request.target = native;
    }
    // Requests do not carry the lane mode: APS5_SPANNED_WAVE selects it here as the driver does from it at device creation.
    if (std::getenv("APS5_SPANNED_WAVE") != nullptr && !request.request.target.spannedWaves) {
        request.request.target.spannedWaves = true;
        std::printf("  spanned waves (APS5_SPANNED_WAVE): a single-wave wave64 compute program spans two full 32-lane subgroups\n");
    }
    // Requests captured before the single-workgroup flag (serializer version 32) lack it: APS5_REPLAY_SINGLE_GROUP=
    // <program>[,<program>...] (hex addresses, low 36 bits compared) replays those programs' compute requests as the
    // driver records their 1x1x1 dispatches (ShaderComputeStageInfo::singleWorkgroup), e.g. to precompile them.
    if (const char* programs = std::getenv("APS5_REPLAY_SINGLE_GROUP"); programs != nullptr && request.request.context.compute.has_value() && !request.request.context.compute->singleWorkgroup) {
        const auto address = request.request.shader.codeAddress & 0xfffffffffull;
        for (const char* cursor = programs; *cursor != '\0';) {
            char* end = nullptr;
            const auto listed = std::strtoull(cursor, &end, 16);
            if (end == cursor) break;
            if ((listed & 0xfffffffffull) == address) {
                request.compute->singleWorkgroup = true;
                request.request.context.compute = request.compute;
                std::printf("  single workgroup (APS5_REPLAY_SINGLE_GROUP): replayed as a 1x1x1 dispatch\n");
                break;
            }
            cursor = *end == ',' ? end + 1 : end;
        }
    }
    if (g_bindless) {
        request.request.target.bindlessImages = true;
        std::printf("  bindless images (--bindless or APS5_BINDLESS_IMAGES=1): table images read through the descriptor heap\n");
        const auto capabilities = request.request.target.supportedCapabilities;
        if (request.request.shader.stage == ShaderRecompiler::ShaderStage::Fragment && std::ranges::find(capabilities, static_cast<std::uint32_t>(spv::CapabilityGroupNonUniformQuad)) == capabilities.end()) {
            static std::list<std::vector<std::uint32_t>> extended;
            auto& quad = extended.emplace_back(capabilities.begin(), capabilities.end());
            quad.push_back(spv::CapabilityGroupNonUniformQuad);
            request.request.target.supportedCapabilities = quad;
            std::printf("  bindless images: the fragment target gains GroupNonUniformQuad, as a bindless device with fragment quad operations adds it\n");
        }
    }
    if (g_pipelineOnly) {
        if (!g_expectedSpirv.empty() && g_requests > 1) throw std::runtime_error("--expect-spv takes one request; several requests need their .fp sidecars");
        if (request.request.shader.stage != ShaderRecompiler::ShaderStage::Compute || !request.request.context.compute.has_value() || request.request.graphics.has_value()) throw std::runtime_error("pipeline-only requires a compute request");
    }
    std::printf("%s: %zu code words, %zu user data, %zu memory regions, wave%u\n", path, request.request.shader.code.size(), request.request.context.userData.size(), request.request.context.memory.size(), request.request.context.waveSize);
    if (request.request.context.compute.has_value()) {
        const auto& compute = *request.request.context.compute;
        std::printf("  compute: threads %ux%ux%u, lds %u dwords, group ids %d%d%d, tg size %d, thread id components %u\n", compute.numThreads[0], compute.numThreads[1], compute.numThreads[2], compute.ldsSizeDwords, compute.groupIdEnable[0], compute.groupIdEnable[1], compute.groupIdEnable[2], compute.tgSizeEnable, compute.threadIdComponentCount);
        if (compute.PartialGroups()) std::printf("  partial groups: dispatch of %ux%ux%u threads\n", compute.partialThreads[0], compute.partialThreads[1], compute.partialThreads[2]);
    }
    if (request.request.graphics.has_value()) {
        const auto& graphics = *request.request.graphics;
        std::printf("  graphics: %zu linked programs, draw %u indices of %u bytes at 0x%llx, %u instances\n", graphics.linkedPrograms.size(), graphics.draw.indexCount, graphics.draw.indexElementBytes, static_cast<unsigned long long>(graphics.draw.indexAddress), graphics.draw.instanceCount);
        if (graphics.mesh.has_value()) {
            const auto& mesh = *graphics.mesh;
            std::printf("  mesh: input primitive %u, %u primitives / %u vertices per group, max %u vertices / %u primitives, %u threads, lds %u dwords, provoking %u, ESGS item %u\n", mesh.inputPrimitive, mesh.primitivesPerGroup, mesh.verticesPerGroup, mesh.maxVertices, mesh.maxPrimitives, mesh.threadsPerGroup, mesh.ldsSizeDwords, mesh.provokingVertex, mesh.esgsItemSize);
        }
    }
    if (g_memory) {
        // --mem: the captured inputs. User data words are printed; each memory region is written to
        // mem_<code address>_<guest address>.bin next to the request for inspection with other tools.
        const auto& context = request.request.context;
        for (std::size_t i = 0; i < context.userData.size(); ++i) std::printf("  user[%zu] = 0x%08x\n", i, context.userData[i]);
        for (const auto& region : context.memory) {
            char name[96];
            std::snprintf(name, sizeof(name), "mem_%llx_%llx.bin", static_cast<unsigned long long>(request.request.shader.codeAddress), static_cast<unsigned long long>(region.guestAddress));
            if (std::FILE* file = std::fopen(name, "wb")) {
                std::fwrite(region.bytes.data(), 1, region.bytes.size(), file);
                std::fclose(file);
            }
            std::printf("  region 0x%llx + 0x%zx -> %s\n", static_cast<unsigned long long>(region.guestAddress), region.bytes.size(), name);
        }
    }
    if (g_code) {
        std::string name(path);
        name = name.substr(name.find_last_of("/\\") + 1) + ".code";
        const auto& code = request.request.shader.code;
        std::ofstream file(name, std::ios::binary);
        file.write(reinterpret_cast<const char*>(code.data()), static_cast<std::streamsize>(code.size() * sizeof(code[0])));
        if (!file) {
            std::printf("  could not write %s\n", name.c_str());
            return false;
        }
        std::printf("  code -> %s\n", name.c_str());
    }
    if (g_assembly) {
        const auto program = ShaderRecompiler::RdnaInstructionDecoder{}.Decode(request.request.shader.code);
        // Raw first words carry what the text omits (branch offsets, waitcnt fields).
        for (const auto& instruction : program.instructions) std::printf("raw=%08x %s\n", instruction.rawWords[0], ShaderRecompiler::RdnaInstructionToString(instruction).c_str());
    }
    if (g_graph) {
        const auto program = ShaderRecompiler::RdnaInstructionDecoder{}.Decode(request.request.shader.code);
        auto graph = ShaderRecompiler::GraphBuilder{}.Build(program);
        std::printf("control flow graph:\n%s", ShaderRecompiler::GraphToString(graph).c_str());
        try {
            ShaderRecompiler::Structurizer{}.Structurize(graph);
            std::printf("structured control flow graph:\n%s", ShaderRecompiler::GraphToString(graph).c_str());
        } catch (const std::exception& error) {
            std::printf("  structurization failed: %s\n", FirstLine(error.what()).c_str());
            return false;
        }
    }
    if (g_materialize && g_maintenance8) request.request.target.nonConstantImageOffsets = true;
    if (!g_materialize) {
        try {
            if (g_prepareOnly) {
                static_cast<void>(ShaderRecompiler::GetResourcePlan(request.request));
            } else {
                auto program = ShaderRecompiler::PrepareResourceProgram(request.request);
                constexpr ShaderRecompiler::ResourceMaterializer materializer;
                static_cast<void>(materializer.ExtractPlan(program));
            }
        } catch (const std::exception& error) {
            std::printf("  resource analysis failed: %s\n", FirstLine(error.what()).c_str());
            return false;
        }
    }
    if (g_prepare || g_materialize) {
        const char* phase = "prepare";
        try {
            const auto handle = ShaderRecompiler::PrepareShader(request.request);
            std::printf("  prepared: %zu SPIR-V words\n", ShaderRecompiler::GetPreparedArtifact(*handle).spirv.size());
            if (g_materialize) {
                phase = "prepared invocation";
                const auto invocation = ShaderRecompiler::PreparedShaderInvocation::TryCreate(request.request, handle);
                if (!invocation.has_value()) throw std::runtime_error("request does not match its prepared shader");
                ShaderRecompiler::RequestMemoryView memory(request.request.context.memory);
                const auto runtime = memory.MakeRuntime(request.request.context.userData, request.request.shader.codeAddress);
                phase = "capture";
                const auto capture = invocation->Capture(runtime);
                phase = "materialize";
                const auto result = invocation->Materialize(*capture);
                std::printf("  materialized: %zu SPIR-V words\n", result->spirv.size());
                std::uint64_t largest = 0;
                for (const auto& binding : result->bindings) largest = std::max<std::uint64_t>(largest, AgcDriver::DriverDetail::VectorHeapBytes(binding.guestDescriptor));
                std::printf("  result heap: %llu bytes, %zu bindings, largest guest descriptor %llu bytes\n", static_cast<unsigned long long>(ShaderRecompiler::ResultHeapBytes(*result)), result->bindings.size(), static_cast<unsigned long long>(largest));
            }
            return true;
        } catch (const std::exception& error) {
            std::printf("  %s failed: %s\n", phase, FirstLine(error.what()).c_str());
            return false;
        }
    }
    try {
        if (g_maintenance8) request.request.target.nonConstantImageOffsets = true;
        if (g_robust2) request.request.target.robustBufferAccess2 = true;
        if (g_waveLdsMemoryBarriers) request.request.target.waveLdsMemoryBarriers = true;
        if (g_deviceScopeLdsMemoryBarriers) request.request.target.deviceScopeLdsMemoryBarriers = true;
        if (g_prepareOnly) {
            const auto prepared = ShaderRecompiler::PrepareShader(request.request);
            if (!prepared) throw std::runtime_error("prepared shader handle is absent");
            const auto words = ShaderRecompiler::GetPreparedArtifact(*prepared).spirv.Words();
            if (words.empty()) throw std::runtime_error("prepared shader has no SPIR-V");
            std::printf("  prepared only: %zu SPIR-V words; runtime memory not materialized\n", words.size());
            InspectSpirv(path, words);
            return true;
        }
        if (g_variants) {
            // --fingerprint-variants: the pipeline fingerprint components of each way to materialize the
            // request, against the game's sidecar.
            const auto sidecar = ReadSidecar(path);
            if (sidecar.present) std::printf("  game sidecar: fingerprint %s\n", sidecar.values.count("fingerprint") ? sidecar.values.at("fingerprint").c_str() : "?");
            if (!device) device = std::make_unique<AgcDriver::VulkanDevice>();
            PrintComponents("Recompile(request)", ShaderRecompiler::Recompile(request.request), sidecar, *device);
            const char* labels[] = {"prepared+capture", "prepared, all pending", "prepared, all accessible", "prepared, page accessible", "prepared, game's probes"};
            for (int mode = 0; mode < 5; ++mode) {
                if (mode == 4 && sidecar.probes.empty()) continue;
                try {
                    PrintComponents(labels[mode], *MaterializePrepared(request.request, mode, &sidecar), sidecar, *device);
                    if (mode == 4) std::printf("    %llu probes the game did not record\n", static_cast<unsigned long long>(t_unrecordedProbes));
                } catch (const std::exception& error) {
                    std::printf("  %-28s failed: %s\n", labels[mode], FirstLine(error.what()).c_str());
                }
            }
            return true;
        }
        // --pipeline-only materializes the way the dispatch path does (a prepared invocation and a capture),
        // with the game's accessibility and pending-write answers when the sidecar recorded them, else
        // page-granular accessibility like ShaderMemory's. RequestMemoryView's own probe (every byte
        // captured) refuses the unread words of mapped pages, which changed image-table bindings.
        const auto pipelineSidecar = g_pipelineOnly ? ReadSidecar(path) : Sidecar{};
        const auto materialized = g_pipelineOnly ? MaterializePrepared(request.request, pipelineSidecar.probes.empty() ? 3 : 4, &pipelineSidecar) : std::make_shared<const ShaderRecompiler::RecompileResult>(ShaderRecompiler::Recompile(request.request));
        const auto& result = *materialized;
        {
            AgcDriver::PipelineFingerprintDetail::Fnv spirvHash;
            for (const auto word : result.spirv.Words()) spirvHash.Mix(word);
            std::printf("  spirv hash %016llx specialization %016llx\n", static_cast<unsigned long long>(spirvHash.hash), static_cast<unsigned long long>(result.PipelineVariantId()));
        }
        std::printf("  recompiled: %zu SPIR-V words%s\n", result.spirv.size(), !g_pipelineOnly ? "" : pipelineSidecar.probes.empty() ? " (prepared invocation, page-granular accessibility)" : " (prepared invocation, the game's recorded probes)");
        if (result.bindlessHeap) {
            std::set<std::uint32_t> constants;
            const auto words = result.spirv.Words();
            for (std::size_t cursor = 5; cursor < words.size();) {
                const auto count = words[cursor] >> 16u;
                if (count == 0u || cursor + count > words.size()) throw std::runtime_error("bindless images: the SPIR-V is malformed");
                if ((words[cursor] & 0xffffu) == 43u && count == 4u) constants.insert(words[cursor + 3u]);
                cursor += count;
            }
            std::set<std::uint64_t> interfaces;
            std::uint64_t records = 0;
            for (const auto& table : result.bindlessTables) {
                records += table.records;
                for (const auto& use : table.uses) {
                    if (ShaderRecompiler::ResourceMaterializer::FindBindlessInterface(use.interfaceId) == nullptr) throw std::runtime_error("bindless images: the registered interface id " + std::to_string(use.interfaceId) + " was not registered by the compile");
                    if (!constants.contains(static_cast<std::uint32_t>(use.interfaceId)) || !constants.contains(static_cast<std::uint32_t>(use.interfaceId >> 32u))) throw std::runtime_error("bindless images: the SPIR-V does not embed the interface id " + std::to_string(use.interfaceId) + " the driver registers");
                    interfaces.insert(use.interfaceId);
                }
            }
            std::printf("  bindless: %zu table registrations, %llu records, %zu interface ids, every id embedded in the SPIR-V\n", result.bindlessTables.size(), static_cast<unsigned long long>(records), interfaces.size());
        }
        if (g_pipelineOnly) {
            if (!device) device = std::make_unique<AgcDriver::VulkanDevice>();
            if (const auto refusal = CheckPipelineReference(path, result, *device); !refusal.empty()) throw std::runtime_error("pipeline-only " + refusal);
        }
        InspectSpirv(path, result.spirv);
        if (g_pipelineOnly) {
            auto nativeRequest = request.request;
            nativeRequest.target = device->ComputeTarget(request.request.context.waveSize);
            const ShaderRecompiler::RequestSerializer serializer;
            if (serializer.Serialize(request.request) != serializer.Serialize(nativeRequest)) throw std::runtime_error("pipeline-only captured target differs from the native compute target");
            std::printf("  pipeline-only device: %s; all serialized compute target fields match (graphics-only fixedPushSlots is not captured)\n", device->DeviceName().c_str());
            std::printf("  native pipeline begin: %zu SPIR-V words; no dispatch\n", result.spirv.size());
            std::fflush(stdout);
            const auto elapsed = device->WarmComputePipeline(result, request.request.shader.codeAddress);
            std::printf("  native pipeline complete: %.3f ms; no guest resources or dispatch\n", elapsed);
            std::printf("PIPELINE-RESULT %s words=%zu fingerprint=%016llx create_ms=%.3f\n", path, result.spirv.size(), static_cast<unsigned long long>(AgcDriver::ComputePipelineFingerprint(result, device->PlannedComputeLayoutKey(result))), elapsed);
            std::fflush(stdout);
        }
        if (g_memory) {
            for (const auto& binding : result.bindings) {
                if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
                for (std::size_t i = 0; i + 4 <= binding.guestDescriptor.size(); i += 4) {
                    const auto* v = binding.guestDescriptor.data() + i;
                    const bool written = i / 4 >= binding.bufferWritten.size() || binding.bufferWritten[i / 4];
                    std::printf("  buffer %zu: V# %08x %08x %08x %08x%s\n", i / 4, v[0], v[1], v[2], v[3], written ? " written" : "");
                }
            }
        }
        return true;
    } catch (const std::exception& error) {
        std::printf("  %s failed: %s\n", g_pipelineOnly ? "pipeline replay" : g_prepareOnly ? "preparation" : "recompile", FirstLine(error.what()).c_str());
        return false;
    }
}


// --make-request <target.req> <shader.bin> <shader.env> <code-address> <out.req>: a standalone compute
// request from raw guest code and its registration environment (sweep/programs/*.env), with the target
// (device features, lane mode inputs) of a captured compute request of the same wave size. The header
// is omitted and user data is zero, so it measures front-end and planning cost, not a game capture.
int MakeRequest(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: agc_shader_replay --make-request <target.req> <shader.bin> <shader.env> <code-address> <out.req>\n");
        return 2;
    }
    try {
        std::map<std::string, std::string> environment;
        {
            std::ifstream file(argv[4]);
            if (!file) throw std::runtime_error("cannot open shader environment");
            for (std::string line; std::getline(file, line);) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                const auto space = line.find(' ');
                if (space != std::string::npos) environment.emplace(line.substr(0, space), line.substr(space + 1));
            }
        }
        const auto number = [](const std::string& text) {
            std::size_t consumed = 0;
            const auto value = std::stoull(text, &consumed, 0);
            if (text.empty() || text.front() == '-' || consumed != text.size()) throw std::runtime_error("invalid unsigned value " + text);
            return value;
        };
        const auto value = [&](const char* key) { return static_cast<std::uint32_t>(number(environment.at(key))); };
        if (environment.at("stage") != "Compute") throw std::runtime_error("--make-request needs a compute environment");
        std::array<std::uint32_t, 3> threads{};
        std::istringstream threadText(environment.at("threads"));
        if (!(threadText >> threads[0] >> threads[1] >> threads[2]) || threads[0] == 0 || threads[1] == 0 || threads[2] == 0) throw std::runtime_error("invalid compute thread counts");
        std::ifstream file(argv[3], std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("cannot open shader binary");
        const auto bytes = file.tellg();
        if (bytes <= 0 || bytes > 64 * 1024 * 1024 || bytes % 4 != 0) throw std::runtime_error("shader binary must hold whole dwords");
        std::vector<std::uint32_t> code(static_cast<std::size_t>(bytes) / 4u);
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(bytes))) throw std::runtime_error("cannot read shader binary");
        const auto rsrc1 = value("rsrc1");
        const auto rsrc2 = value("rsrc2");
        const auto wave = value("wave");
        const auto userCount = value("user_sgprs");
        if ((wave != 32u && wave != 64u) || userCount > 32u || userCount != ((rsrc2 >> 1u) & 0x1fu)) throw std::runtime_error("invalid compute wave or user SGPR count");
        const auto donor = AgcDriver::RecordedRequests::Deserialize(ReadText(argv[2]));
        if (donor.request.shader.stage != ShaderRecompiler::ShaderStage::Compute || !donor.request.context.compute || donor.request.graphics || donor.request.context.waveSize != wave) throw std::runtime_error("the target request must be a standalone compute request of the same wave size");
        const auto firstUser = value("first_user_sgpr");
        const auto scratch = (rsrc2 & 1u) != 0 ? value("scratch_dw_per_thread") : 0u;
        const auto address = number(argv[5]);
        std::vector<std::uint32_t> userData(userCount, 0u);
        const std::vector<ShaderRecompiler::MemoryRegion> memory{{address, std::as_bytes(std::span<const std::uint32_t>(code))}};
        ShaderRecompiler::RecompileRequest request{};
        request.shader = {ShaderRecompiler::ShaderStage::Compute, address, code, 0, {}};
        request.target = donor.request.target;
        request.context.userData = userData;
        request.context.memory = memory;
        request.context.waveSize = wave;
        request.context.userDataBaseRegister = firstUser;
        request.context.compute = ShaderRecompiler::ShaderComputeStageInfo{threads, std::min(((rsrc2 >> 15u) & 0x1ffu) * 128u, 16384u), {((rsrc2 >> 7u) & 1u) != 0, ((rsrc2 >> 8u) & 1u) != 0, ((rsrc2 >> 9u) & 1u) != 0}, ((rsrc2 >> 10u) & 1u) != 0, ((rsrc2 >> 11u) & 3u) + 1u, {}, scratch};
        request.context.floatMode = ShaderRecompiler::ShaderFloatMode{(rsrc1 >> 12u) & 0xffu, ((rsrc1 >> 21u) & 1u) != 0, ((rsrc1 >> 23u) & 1u) != 0, ((rsrc1 >> 26u) & 1u) != 0};
        request.layout = donor.request.layout;
        request.useCache = true;
        std::ofstream out(argv[6], std::ios::binary);
        out << ShaderRecompiler::RequestSerializer{}.Serialize(request);
        out.close();
        if (!out) throw std::runtime_error("cannot write the request");
        std::printf("request %s: compute 0x%llx, %zu code bytes, wave %u, %ux%ux%u threads, %u user SGPRs, target from %s\n", argv[6], static_cast<unsigned long long>(address), code.size() * 4u, wave, threads[0], threads[1], threads[2], userCount, argv[2]);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "--make-request: %s\n", FirstLine(error.what()).c_str());
        return 1;
    }
}

}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--make-request") return MakeRequest(argc, argv);
    if (argc < 2) {
        std::fprintf(stderr, "usage: agc_shader_replay [--dis] [--asm] [--cfg] [--mem] [--spv] [--prepare] [--materialize] [--code] [--maintenance8] [--robust2] [--wave-lds-memory-barriers] [--device-scope-lds-memory-barriers] [--bindless] [--prepare-only | --pipeline-only --pipeline-cache DIR [--expect-spv FILE | --no-reference]] <shader.req>...\n  --prepare replays the request as a prepared artifact\n  --materialize also captures the resources from the request memory and materializes the prepared invocation\n  --prepare-only plans and compiles without materializing runtime memory\n  --pipeline-only recompiles captured compute inputs and creates the native pipeline without guest resources or dispatch\n  --pipeline-cache selects an existing private cache directory; __GL_SHADER_DISK_CACHE_PATH must select a separate fresh empty directory\n  --expect-spv requires byte equality with an ordinary replay before device creation (one request); without it each\n    request needs the .fp sidecar APS5_PIPELINE_REQUESTS saved next to it, and its pipeline fingerprint must match\n  --no-reference skips that check (requests from older runs); a hit in the game's pipeline cache then proves the match\n  several requests share one device and cache; each success prints a PIPELINE-RESULT line\n  --fingerprint-variants prints the pipeline fingerprint components of each way to materialize a request against its .fp\n    (--describe adds the values: specialization constants, bindings)\n  --native-target replaces a captured compute target with this device's (requests from builds with another target) and names the fields that differed\n  --bindless (or APS5_BINDLESS_IMAGES=1) compiles table images through the bindless descriptor heap; a device made for --pipeline-only enables it too\n  the driver writes shader_<address>.req files when APS5_DUMP_SHADERS is set; shader_<address>.prepare.req holds a request whose preparation failed before any resource capture, so replay it with --prepare\n");
        return 2;
    }
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bindless") g_bindless = true;
    }
    if (g_bindless) {
#ifdef _WIN32
        _putenv_s("APS5_BINDLESS_IMAGES", "1");
#else
        setenv("APS5_BINDLESS_IMAGES", "1", 1);
#endif
    }
    if (const char* value = std::getenv("APS5_BINDLESS_IMAGES"); value != nullptr && std::string(value) == "1") g_bindless = true;
    try {
        ConfigurePipelineReplay(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        if (option == "--pipeline-cache" || option == "--expect-spv") ++i;
        else if (!option.starts_with("--")) ++g_requests;
    }
    std::unique_ptr<AgcDriver::VulkanDevice> device;
    int failures = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--pipeline-only" || std::string(argv[i]) == "--no-reference" || std::string(argv[i]) == "--fingerprint-variants" || std::string(argv[i]) == "--describe" || std::string(argv[i]) == "--native-target" || std::string(argv[i]) == "--bindless") continue;
        if (std::string(argv[i]) == "--pipeline-cache" || std::string(argv[i]) == "--expect-spv") {
            ++i;
            continue;
        }
        if (std::string(argv[i]) == "--prepare-only") {
            g_prepareOnly = true;
            continue;
        }
        if (std::string(argv[i]) == "--dis") {
            g_disassemble = true;
            continue;
        }
        if (std::string(argv[i]) == "--asm") {
            g_assembly = true;
            continue;
        }
        if (std::string(argv[i]) == "--cfg") {
            g_graph = true;
            continue;
        }
        if (std::string(argv[i]) == "--maintenance8") {
            g_maintenance8 = true;
            continue;
        }
        if (std::string(argv[i]) == "--robust2") {
            g_robust2 = true;
            continue;
        }
        if (std::string(argv[i]) == "--wave-lds-memory-barriers") {
            g_waveLdsMemoryBarriers = true;
            continue;
        }
        if (std::string(argv[i]) == "--device-scope-lds-memory-barriers") {
            g_deviceScopeLdsMemoryBarriers = true;
            continue;
        }
        if (std::string(argv[i]) == "--mem") {
            g_memory = true;
            continue;
        }
        if (std::string(argv[i]) == "--code") {
            g_code = true;
            continue;
        }
        if (std::string(argv[i]) == "--prepare") {
            g_prepare = true;
            continue;
        }
        if (std::string(argv[i]) == "--materialize") {
            g_materialize = true;
            continue;
        }
        if (std::string(argv[i]) == "--spv") {
            g_spirv = true;
            continue;
        }
        try {
            if (!Replay(argv[i], device)) ++failures;
        } catch (const std::exception& error) {
            std::printf("%s: could not load request: %s\n", argv[i], FirstLine(error.what()).c_str());
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
