#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_SHAPEREPLAY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_SHAPEREPLAY_HPP

#include "prx/libSceAgcDriver/Graphics/include/ShapeReplayProfile.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include <span>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ShaderRecompiler {
struct ResourceCapture;
}

namespace AgcDriver::DriverDetail::ShapeReplay {

struct PatchMap {
    struct BufferElement {
        std::uint32_t binding = 0;
        std::uint32_t element = 0;
        std::vector<std::uint32_t> candidates;
    };
    std::vector<std::uint32_t> flatBindings;
    std::vector<BufferElement> buffers;
    std::int32_t shaderDataBinding = -1;
    bool pushUserWords = false;
    bool vertexAttributes = false;
};

std::optional<PatchMap> Learn(const ShaderRecompiler::RecompileResult& result, const ShaderRecompiler::ResourceSnapshot& snapshot, const ShaderRecompiler::ShaderVertexStageInfo* vertex);
std::shared_ptr<ShaderRecompiler::RecompileResult> Apply(const ShaderRecompiler::RecompileResult& plan, const PatchMap& map, const ShaderRecompiler::ResourceSnapshot& snapshot, const ShaderRecompiler::ShaderVertexStageInfo* vertex);
bool KeyWordsCarry(const ShaderRecompiler::RecompileResult& plan, const ShaderRecompiler::RecompileResult& replayed, const PatchMap& map, bool movableBuffers);
std::string Difference(const ShaderRecompiler::RecompileResult& replayed, const ShaderRecompiler::RecompileResult& full);
std::uint64_t ReplayKey(const Graphics::ShapeReplay::StageRecord& record);

enum class Kind : std::uint8_t { Draw, Dispatch };

bool Active();
bool DispatchActive();
std::uint64_t PlanKey(std::uint64_t shape, std::span<const Graphics::CompiledShader> stages);
std::uint64_t& DrawShape();

struct Counters {
    std::uint64_t captures = 0;
    std::uint64_t eligible = 0;
    std::uint64_t hits = 0;
    std::uint64_t learned = 0;
    std::uint64_t unlearnable = 0;
    std::uint64_t refused = 0;
    std::uint64_t probations = 0;
    std::uint64_t probationMismatches = 0;
    std::uint64_t verified = 0;
    std::uint64_t verifyMismatches = 0;
    std::uint64_t disabled = 0;
    std::uint64_t replayNs = 0;
    std::uint64_t materializeNs = 0;
    std::uint64_t materializes = 0;
    std::uint64_t absentBuffers = 0;
    std::uint64_t keysCarried = 0;
    std::uint64_t keyMismatches = 0;
};
Counters Totals(Kind kind = Kind::Draw);

using FullMaterialize = std::function<std::shared_ptr<const ShaderRecompiler::RecompileResult>()>;
std::shared_ptr<const ShaderRecompiler::RecompileResult> Materialize(std::uint64_t shape, std::size_t stage, ShaderRecompiler::ShaderStage kind, std::uint32_t pushOffset, std::uint64_t program, const ShaderRecompiler::ResourceCapture& capture, const Graphics::ShapeReplay::StageRecord* record, const ShaderRecompiler::ShaderVertexStageInfo* vertex, const FullMaterialize& full, Kind lane = Kind::Draw);

}

#endif
