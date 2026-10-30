#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_NEWPATH_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_NEWPATH_HPP

#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/Router.hpp"
#include <atomic>
#include <cstdint>
#include <memory>

namespace ShaderRecompiler {
struct RecompileResult;
}

namespace AgcDriver {
class VulkanDevice;
}

namespace AgcDriver::Graphics {
class ShaderResources;
}

namespace AgcDriver::NewPath {

[[nodiscard]] bool Enabled();
[[nodiscard]] bool EmitDispatches();

Decision RouteCommand(CommandKind kind, const VulkanDevice* device = nullptr);

void BeginCommand(CommandKind kind, const VulkanDevice* device);
void EndCommand();

struct EmitCandidate {
    CommandFacts facts;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> twin;
    std::atomic<bool> noted{false};
    bool counted = true;
    bool tableDeferred = false;
    bool addressBased = false;
    Route routed = Route::Fallback;
    mutable struct {
        bool active = false;
        bool cached = false;
        bool committed = false;
        std::uint8_t repeat = 0;
        std::uint8_t cover = 0;
        std::uint32_t shape = 0;
        std::uint32_t regions = 0;
        std::uint64_t decodeNs = 0;
    } census;

    EmitCandidate() = default;
    EmitCandidate(const EmitCandidate&) = delete;
    EmitCandidate& operator=(const EmitCandidate&) = delete;
    ~EmitCandidate();
};

[[nodiscard]] std::shared_ptr<EmitCandidate> ClaimDispatch(const ShaderRecompiler::RecompileResult& compiled);

class EmitScope {
public:
    explicit EmitScope(std::shared_ptr<EmitCandidate> candidate);
    ~EmitScope();
    EmitScope(const EmitScope&) = delete;
    EmitScope& operator=(const EmitScope&) = delete;

private:
    std::shared_ptr<EmitCandidate> candidate;
    std::shared_ptr<EmitCandidate> previous;
};

[[nodiscard]] EmitCandidate* CurrentEmit();
Decision NoteEmit(EmitCandidate& candidate, const CommandFacts& facts);
[[nodiscard]] bool MayEmitWithoutTable(const EmitCandidate& candidate, bool pipelineReady);
// Whether the candidate's build binds its written buffer elements in place (Graphics::InPlaceWritesScope):
// with emission on, for a build whose dispatch may run its resolved twin (APS5_NEW_PATH_IN_PLACE_WRITES=all:
// every candidate's; APS5_NEW_PATH_NO_IN_PLACE_WRITES=1: none).
[[nodiscard]] bool InPlaceWrites(const EmitCandidate& candidate);
// The router's route for these facts, without noting it.
[[nodiscard]] bool WouldEmit(const CommandFacts& facts);
void NoteDispatchCost(const EmitCandidate& candidate, std::uint64_t deviceNs, std::uint64_t buildNs);
void NoteBufferCover(const EmitCandidate& candidate, const Graphics::ShaderResources& resources);

enum class CostStep : std::uint8_t { PreRecord, RecordHead, Ranges, Compare, Route, TwinPipeline, EnsureTable, RecordCommands, TableUpload, RecordPush, DispatchMarks, PostRecord, Count };
void CostBegin();
void CostLap(CostStep step);
void NoteCompletionCost(std::uint64_t ns);

void Close(const char* window);

}

#endif
