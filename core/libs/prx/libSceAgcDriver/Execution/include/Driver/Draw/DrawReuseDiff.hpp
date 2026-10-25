#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWREUSEDIFF_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWREUSEDIFF_HPP

#include "UserDataUse.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include <cstdint>
#include <cstdlib>
#include <span>

namespace AgcDriver::DriverDetail {

// Profile-only diagnostic (APS5_DRAW_REUSE_DIFF=1, local): classifies every freshly captured draw
// stage (and dispatch) against a per-thread ring of the last plans of the same (shape key, stage)
// to measure how much of the queue-0 capture work a shape-keyed, pointer-relocating reuse could
// skip (research/structural/reuse/design.md 1.2). Read-only: it copies what the capture already
// recorded and never touches a cache. Reported per 10 s as `[draw-reuse]` / `[dispatch-reuse]`.
inline bool ReuseDiffEnabled() {
    static const bool enabled = std::getenv("APS5_DRAW_REUSE_DIFF") != nullptr;
    return enabled;
}

enum class ReuseKind : std::uint8_t { Draw, Dispatch };

struct ReuseStageInput {
    ReuseKind kind = ReuseKind::Draw;
    std::uint64_t shape = 0;
    std::uint32_t stage = 0;
    std::uint32_t stageType = 0;
    std::span<const std::uint32_t> userData;
    std::uint32_t userDataBase = 0;
    std::uint32_t hiddenUserWords = 0;
    // runs / words in address order (DrawStageVariant), dataPositions (recordStageDataPositions /
    // DataWordPositions), compiled and vertexInfo.
    const DispatchVariant* variant = nullptr;
    const ShaderRecompiler::SrtUserDataUse* use = nullptr;
    std::span<const Graphics::DecodeRead> decodeReads;
    // Driver::sameVertexInfo for the "vertex info equal" count (null for dispatches).
    bool (*sameVertex)(const ShaderRecompiler::ShaderVertexStageInfo&, const ShaderRecompiler::ShaderVertexStageInfo&) = nullptr;
};

void NoteReuseStage(const ReuseStageInput& input);
// Stages of a draw that were not captured: served by a draw-cache hit, by a data-entry candidate,
// or of a draw without a register key (no shape key to group them by).
void NoteReuseUncaptured(ReuseKind kind, std::uint64_t hitStages, std::uint64_t reusedStages, std::uint64_t unkeyedStages);

}

#endif
