// agc_shader_replay <shader_*.req>...: replays compute recompile requests the driver saved with
// APS5_DUMP_SHADERS=1, running the same resource analysis and recompile steps without the game.
#include "ControlFlow/GraphBuilder.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "ControlFlow/Structurizer.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Recompiler.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"

#if ANYPS5_ENABLE_SPIRV_TOOLS
#include "spirv-tools/libspirv.hpp"
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
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
bool g_graph = false;
bool g_maintenance8 = false;
bool g_code = false;
bool g_prepareOnly = false;
bool g_pipelineOnly = false;
std::string g_expectedSpirv;

void ConfigurePipelineReplay(int argc, char** argv) {
    std::string cache;
    bool prepare = false;
    bool maintenance = false;
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        if (option == "--pipeline-only") g_pipelineOnly = true;
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
        if (!cache.empty() || !g_expectedSpirv.empty()) throw std::runtime_error("--pipeline-cache and --expect-spv require --pipeline-only");
        return;
    }
    if (prepare || maintenance) throw std::runtime_error("--pipeline-only cannot be combined with --prepare-only or --maintenance8");
    if (cache.empty()) throw std::runtime_error("--pipeline-only requires --pipeline-cache");
    if (g_expectedSpirv.empty()) throw std::runtime_error("--pipeline-only requires --expect-spv");
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
    auto request = ShaderRecompiler::RequestSerializer{}.Deserialize(ReadText(path));
    if (g_pipelineOnly) {
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
    try {
        if (g_maintenance8) request.request.target.nonConstantImageOffsets = true;
        if (g_prepareOnly) {
            const auto prepared = ShaderRecompiler::PrepareShader(request.request);
            if (!prepared) throw std::runtime_error("prepared shader handle is absent");
            const auto words = ShaderRecompiler::GetPreparedArtifact(*prepared).spirv.ReadTransient();
            if (!words || words->empty()) throw std::runtime_error("prepared shader has no SPIR-V");
            std::printf("  prepared only: %zu SPIR-V words; runtime memory not materialized\n", words->size());
            InspectSpirv(path, *words);
            return true;
        }
        const auto result = ShaderRecompiler::Recompile(request.request);
        std::printf("  recompiled: %zu SPIR-V words\n", result.spirv.size());
        if (g_pipelineOnly) CheckExpectedSpirv(result.spirv);
        InspectSpirv(path, result.spirv);
        if (g_pipelineOnly) {
            if (!device) device = std::make_unique<AgcDriver::VulkanDevice>();
            auto nativeRequest = request.request;
            nativeRequest.target = device->ComputeTarget(request.request.context.waveSize);
            const ShaderRecompiler::RequestSerializer serializer;
            if (serializer.Serialize(request.request) != serializer.Serialize(nativeRequest)) throw std::runtime_error("pipeline-only captured target differs from the native compute target");
            std::printf("  pipeline-only device: %s; all serialized compute target fields match (graphics-only fixedPushSlots is not captured)\n", device->DeviceName().c_str());
            std::printf("  native pipeline begin: %zu SPIR-V words; no dispatch\n", result.spirv.size());
            std::fflush(stdout);
            const auto elapsed = device->WarmComputePipeline(result, request.request.shader.codeAddress);
            std::printf("  native pipeline complete: %.3f ms; no guest resources or dispatch\n", elapsed);
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

}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: agc_shader_replay [--dis] [--asm] [--cfg] [--mem] [--spv] [--code] [--maintenance8] [--prepare-only | --pipeline-only --pipeline-cache DIR --expect-spv FILE] <shader.req>...\n  --prepare-only plans and compiles without materializing runtime memory\n  --pipeline-only recompiles captured compute inputs and creates the native pipeline without guest resources or dispatch\n  --pipeline-cache selects an existing private cache directory; __GL_SHADER_DISK_CACHE_PATH must select a separate fresh empty directory\n  --expect-spv requires byte equality with an ordinary replay before device creation\n  the driver writes shader_<address>.req files when APS5_DUMP_SHADERS is set\n");
        return 2;
    }
    try {
        ConfigurePipelineReplay(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
    std::unique_ptr<AgcDriver::VulkanDevice> device;
    int failures = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--pipeline-only") continue;
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
        if (std::string(argv[i]) == "--mem") {
            g_memory = true;
            continue;
        }
        if (std::string(argv[i]) == "--code") {
            g_code = true;
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
