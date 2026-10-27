#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_COMMANDMIX_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_COMMANDMIX_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace ShaderRecompiler {
struct RecompileResult;
struct ResourceCapture;
struct IrResourcePlan;
}

namespace AgcDriver::Pm4 {
struct DrawParameters;
}

namespace AgcDriver::Graphics {
struct CompiledShader;
}

// APS5_PROFILE_COMMAND_MIX=1: the command mix of each profile window (ProfileWindows; a frame in
// agc_frame_replay, 10 s in the game), as one line per window:
//
//   [command-mix] window=<w> draws <n> (indexed <n>, non-indexed <n>, indirect <n> (multi <n>, count <n>),
//     instanced <n>, recipe <n>); dispatches <n> (direct <n>, indirect <n>, cache hits <n>);
//     draw stages <uses>: <stage mix>; dispatch programs <uses>: <stage mix>; bda-template builds: <outcome> <n>, ...
//   <stage mix> = user-sgprs 0:<n> 1-2:<n> 3-4:<n> 5-8:<n> 9-16:<n> 17+:<n> mean <x> max <n>;
//     srt-depth 0:<n> 1:<n> 2:<n> 3:<n> 4+:<n> unknown:<n>; descriptors ub <n> sb <n> utb <n> stb <n>
//     sampled <n> storage-image <n> sampler <n>; bindless-tables <n>; bda <n>
//
// (one physical line). Draws are counted once per committed draw (Driver::commitDraw): indirect
// (DRAW_INDIRECT / DRAW_INDEX_INDIRECT and their multi forms; multi = more than one record or a
// count read from memory, count = the count read from memory), else indexed or non-indexed;
// instanced = a direct draw with more than one instance; recipe = drawn from a draw recipe.
// Dispatches are counted once per dispatch that reaches the recorder (fill/copy kernels the driver
// replaces are not), cache hits = the compiled program came from the dispatch cache. A "stage use"
// is one shader stage of one draw, or the program of one dispatch: user-sgprs = user data
// registers the program reads (RecompileResult::userDataRegisterReads), srt-depth = the longest
// chain of SRT memory reads its resource walk makes (1 = a table a user SGPR points to, 2 = a
// table read through that one, ...; unknown = the program was compiled before the profile saw its
// resource plan, e.g. from a cache of an earlier run), descriptors = Vulkan descriptors by kind
// (DescriptorKind, array counts summed), bindless-tables = bindless image tables registered, bda =
// programs that address guest memory through the BDA table. bda-template builds: the change of
// ShaderResources::TemplateOutcomes over the window.
namespace AgcDriver::CommandMix {

[[nodiscard]] bool Enabled();

// One stage use, as the line classifies it.
struct StageUse {
    std::uint32_t userSgprs = 0;
    // -1: unknown.
    std::int32_t srtDepth = -1;
    // By the recompiler's DescriptorKind order (uniform buffer, storage buffer, uniform texel buffer,
    // storage texel buffer, sampled image, storage image, sampler).
    std::array<std::uint64_t, 7> descriptors{};
    std::uint32_t bindlessTables = 0;
    bool bda = false;
};

struct DrawUse {
    bool indexed = false;
    bool indirect = false;
    bool multi = false;
    bool countIndirect = false;
    std::uint32_t instances = 1;
    bool recipe = false;
};

// The totals of one window; Format prints the line above.
class Window {
public:
    void AddDraw(const DrawUse& draw);
    void AddDispatch(bool indirect, bool cacheHit);
    void AddStage(bool draw, const StageUse& stage);
    [[nodiscard]] std::string Format(const char* window, const std::string& templates) const;
    [[nodiscard]] std::uint64_t Draws() const { return draws; }
    [[nodiscard]] std::uint64_t Dispatches() const { return dispatches; }

private:
    struct Stages {
        std::uint64_t uses = 0;
        std::array<std::uint64_t, 6> sgprBuckets{};
        std::uint64_t sgprSum = 0;
        std::uint32_t sgprMax = 0;
        std::array<std::uint64_t, 6> depthBuckets{};
        std::array<std::uint64_t, 7> descriptors{};
        std::uint64_t bindlessTables = 0;
        std::uint64_t bda = 0;
    };
    static void formatStages(std::string& line, const Stages& stages);
    std::uint64_t draws = 0, indexed = 0, nonIndexed = 0, indirect = 0, multi = 0, countIndirect = 0, instanced = 0, recipe = 0;
    std::uint64_t dispatches = 0, directDispatches = 0, indirectDispatches = 0, cacheHits = 0;
    Stages drawStages, dispatchStages;
};

// The longest chain of SRT reads in a resource plan (0 when it reads no SRT memory).
[[nodiscard]] std::int32_t SrtDepth(const ShaderRecompiler::IrResourcePlan& plan);
[[nodiscard]] StageUse Classify(const ShaderRecompiler::RecompileResult& program);

// Driver hooks (no-ops when off): a materialized program with its capture (remembers its SRT depth),
// a committed draw with its stages, a dispatch with its program.
void NoteMaterialized(const ShaderRecompiler::RecompileResult& program, const ShaderRecompiler::ResourceCapture& capture);
void NoteDraw(const Pm4::DrawParameters& draw, std::span<const Graphics::CompiledShader> stages, bool recipe);
void NoteDispatch(const ShaderRecompiler::RecompileResult& program, bool indirect, bool cacheHit);

// Prints the window's line and starts the next (ProfileWindows::Close).
void Close(const char* window);

}

#endif
