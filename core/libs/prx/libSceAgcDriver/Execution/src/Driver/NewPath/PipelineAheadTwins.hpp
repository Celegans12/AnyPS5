#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_PIPELINEAHEADTWINS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_PIPELINEAHEADTWINS_HPP

#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PipelineAhead.hpp"
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

namespace ShaderRecompiler {
struct RecompileResult;
}

namespace AgcDriver::NewPath {

// What a resolved twin's compute pipeline is built from: the twin module, today's module's bindings (the descriptor
// set the emitted dispatch binds is today's resources' set, so the pipeline layout is described from those bindings),
// and the push stages and program address of the dispatch that asked for it.
struct TwinModule {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> twin;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> layout;
    std::uint32_t pushStages = 0;
    std::uint64_t programAddress = 0;
};

// Creates the pipeline of a twin module (VulkanDevice: DescribeSetLayouts of `layout` + computePipeline, which goes
// through the device's VkPipelineCache, so the next run's creation is a cache hit).
using TwinBuilder = std::function<PipelineHandle(const TwinModule&)>;

[[nodiscard]] PipelineKey TwinPipelineKey(const ShaderRecompiler::RecompileResult& twin, bool pushConstants);
// A module holding only `program`'s bindings (all DescribeSetLayouts reads).
[[nodiscard]] std::shared_ptr<const ShaderRecompiler::RecompileResult> LayoutOf(const ShaderRecompiler::RecompileResult& program);

// One file per key under the manifest's directory: header (magic, format, SourceVersion, key, push stages, program
// address) and the two modules in the shader disk cache's encoding. Another SourceVersion's or key's file is refused.
[[nodiscard]] std::filesystem::path TwinModulePath(const std::filesystem::path& directory, PipelineKey key);
bool WriteTwinModule(const std::filesystem::path& directory, std::uint64_t sourceVersion, PipelineKey key, const TwinModule& module);
[[nodiscard]] std::optional<TwinModule> ReadTwinModule(const std::filesystem::path& directory, std::uint64_t sourceVersion, PipelineKey key);
[[nodiscard]] bool TwinModuleHeaderValid(const std::filesystem::path& directory, std::uint64_t sourceVersion, PipelineKey key);

class TwinPipelines {
public:
    struct Counts {
        std::uint64_t notReady = 0;
        std::uint64_t layoutMismatches = 0;
        std::uint64_t modulesWritten = 0;
        std::uint64_t moduleWriteFailures = 0;
        std::uint64_t prewarmQueued = 0;
        std::uint64_t prewarmSkipped = 0;
        std::uint64_t moduleReadFailures = 0;
        std::uint64_t createdOnWorker = 0;
        std::uint64_t createdOnCaller = 0;
        std::uint64_t callerOverSecond = 0;
        double workerMs = 0;
        double callerMs = 0;
        double maxCallerMs = 0;
    };

    TwinPipelines(PipelineAhead& ahead, std::filesystem::path directory, std::uint64_t sourceVersion);
    ~TwinPipelines();
    TwinPipelines(const TwinPipelines&) = delete;
    TwinPipelines& operator=(const TwinPipelines&) = delete;

    // The dispatch side: the pipeline when it is ready, else null (the caller falls back to today's path). The first
    // request for a key queues its creation (ahead of prewarm work) and writes its module file on the worker; with the
    // gate on nothing is created on the calling thread. `describe` runs once per key, on the first request.
    PipelineHandle Get(PipelineKey key, const std::function<TwinModule()>& describe, const TwinBuilder& build);
    // Queues the manifest's keys whose module files exist, from a thread of its own (the module files are read by the
    // creation workers). Again after Stop only.
    void StartPrewarm(TwinBuilder build);
    // Stops the prewarm thread and resets the creation queue and entries (device teardown).
    void Stop();
    void NoteLayoutMismatch();
    [[nodiscard]] Counts Totals() const;
    [[nodiscard]] const std::filesystem::path& Directory() const { return directory; }

private:
    PipelineFactory factoryFor(PipelineKey key, std::shared_ptr<const TwinModule> module, TwinBuilder build, std::thread::id caller);
    PipelineHandle timed(PipelineKey key, const TwinModule& module, const TwinBuilder& build, bool worker);

    PipelineAhead& ahead;
    std::filesystem::path directory;
    std::uint64_t sourceVersion = 0;
    mutable std::mutex mutex;
    Counts counts;
    std::thread prewarm;
    std::atomic<bool> stopping{false};
    bool prewarmStarted = false;
};

}

#endif
