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
void NoteDispatchCost(const EmitCandidate& candidate, std::uint64_t deviceNs, std::uint64_t buildNs);
void NoteBufferCover(const EmitCandidate& candidate, const Graphics::ShaderResources& resources);

void Close(const char* window);

}

#endif
