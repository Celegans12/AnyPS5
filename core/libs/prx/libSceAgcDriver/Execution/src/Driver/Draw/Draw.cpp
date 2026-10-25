#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/CommitPrep.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawProgramTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ParallelProfile.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <algorithm>
#include <spirv/unified1/spirv.hpp>
#include <bit>
#include <cstdlib>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <exception>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <typeinfo>

namespace AgcDriver::DriverDetail {

namespace {

bool TraceShaderCorrelation() {
    static const bool enabled = std::getenv("APS5_TRACE_SHADER_CORRELATION") != nullptr;
    return enabled && AgcDriver::CaptureTrace::Enabled();
}

std::atomic<std::uint64_t>& NextDrawTraceToken() {
    static std::atomic<std::uint64_t> next{0};
    return next;
}

std::string ExceptionReason(const std::exception_ptr& failure) {
    try {
        if (failure != nullptr) std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        return error.what();
    } catch (...) {
    }
    return "unknown exception";
}

const char* ExceptionType(const std::exception_ptr& failure) {
    try {
        if (failure != nullptr) std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        return typeid(error).name();
    } catch (...) {
    }
    return "unknown";
}

void TraceMaterializationFailure(std::uint64_t token, const Submission& submission, std::size_t packetOffset, const char* phase, std::size_t completedSubdraws, const std::exception_ptr& failure) {
    if (token == 0) return;
    try {
        const auto reason = ExceptionReason(failure);
        AgcDriver::CaptureTrace::Log("draw-materialization-failed token=%llu submission=%llu queue=%x offset=%zu phase=%s exception=%s reason=%.256s completedSubdraws=%zu",
                static_cast<unsigned long long>(token), static_cast<unsigned long long>(submission.serial), submission.queue, packetOffset, phase, ExceptionType(failure), reason.c_str(), completedSubdraws);
    } catch (...) {

    }
}

void TraceSubmissionFailure(std::uint64_t token, const Submission& submission, std::size_t packetOffset, std::size_t completedSubdraws, const std::exception_ptr& failure) {
    if (token == 0) return;
    try {
        const auto reason = ExceptionReason(failure);
        AgcDriver::CaptureTrace::Log("draw-submission-failed token=%llu submission=%llu queue=%x offset=%zu exception=%s reason=%.256s completedSubdraws=%zu",
                static_cast<unsigned long long>(token), static_cast<unsigned long long>(submission.serial), submission.queue, packetOffset, ExceptionType(failure), reason.c_str(), completedSubdraws);
    } catch (...) {

    }
}

void TraceDrawAttemptFailure(std::uint64_t token, const char* phase, const char* outcome, std::uint64_t submission, std::uint32_t queue, std::size_t packetOffset, std::string_view reason, std::string_view exceptionType = "none") {
    if (token == 0) return;
        AgcDriver::CaptureTrace::LogBestEffort("draw-attempt-failed token=%llu phase=%s outcome=%s submission=%llu queue=%x offset=%zu exception=%s reason=%.256s",
            static_cast<unsigned long long>(token), phase, outcome, static_cast<unsigned long long>(submission), queue, packetOffset, exceptionType.data(), reason.data());
}

void TraceDrawAttemptException(std::uint64_t token, const char* phase, std::uint64_t submission, std::uint32_t queue, std::size_t packetOffset, const std::exception_ptr& failure) noexcept {
    try {
        TraceDrawAttemptFailure(token, phase, "exception", submission, queue, packetOffset, ExceptionReason(failure), ExceptionType(failure));
    } catch (...) {

    }
}

const char* DrawVerdictName(DrawVerdict verdict) {
    switch (verdict) {
        case DrawVerdict::Drawn: return "Drawn";
        case DrawVerdict::Nothing: return "Nothing";
        case DrawVerdict::Rejected: return "Rejected";
    }
    return "Unknown";
}

std::string_view TrimmedReason(std::string_view reason) {
    if (reason.size() > 200) reason = reason.substr(0, 200);
    return reason;
    }

}

std::uint64_t& AheadDumpKey();

namespace {

// gpuind: GPU-side indirect draw parameters (APS5_NO_GPU_INDIRECT_ARGS=1 turns them off). A
// vertex-path indirect draw whose CP-written base vertex / start instance SGPRs the stage uses
// beyond the folded fetch offset ("patched SGPR not folded") compiles its front stage with those
// SGPRs read from BaseVertex / BaseInstance, so the records stay on the GPU.
bool GpuIndirectArgs() {
    return !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoGpuIndirectArgs);
}

// An indirect draw's draw-cache keys carry its SGPR locations: a cached front stage compiled with
// GPU-side draw parameters must not serve a direct draw with the same registers, or the reverse.
std::uint64_t GpuParameterKey(std::uint64_t key, const Pm4::DrawParameters::IndirectDraw& indirect) {
    if (key == 0) return 0;
    std::uint64_t mixed = key ^ (0x6770756964726177ull + (static_cast<std::uint64_t>(indirect.baseVertexLocation) << 32u) + (static_cast<std::uint64_t>(indirect.startInstanceLocation) << 8u) + 1u);
    mixed *= 0x9e3779b97f4a7c15ull;
    mixed ^= mixed >> 31u;
    return mixed == 0 ? 1 : mixed;
}

// Learned per (front shader, base vertex SGPR, start instance SGPR): 1 = its GPU-parameter stage
// takes the GPU path (compile it so from the start), 2 = it does not (or it was refused).
struct GpuParameterMemo {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::uint8_t> states;
    static std::uint64_t Key(std::uint64_t address, std::int32_t baseVertex, std::int32_t startInstance) { return address * 0x100000001b3ull ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(baseVertex) & 0xffu) << 8u) ^ (static_cast<std::uint32_t>(startInstance) & 0xffu); }
    std::uint8_t Get(std::uint64_t key) {
        std::lock_guard lock(mutex);
        const auto found = states.find(key);
        return found == states.end() ? 0 : found->second;
    }
    void Set(std::uint64_t key, std::uint8_t state) {
        std::lock_guard lock(mutex);
        auto& slot = states[key];
        if (slot != 2) slot = state;
    }
};
GpuParameterMemo& GpuParameterStates() {
    static GpuParameterMemo memo;
    return memo;
}

struct GpuParameterCounters {
    std::atomic<std::uint64_t> compiled{0}, learned{0}, gpuSide{0}, fellBack{0}, refused{0}, notPending{0};
};
GpuParameterCounters& GpuParameterCount() {
    static GpuParameterCounters counters;
    return counters;
}

// Under APS5_PROFILE_DRAW, every 10 s: `[gpuind]` draws whose front stage read the draw parameters
// on the GPU, shapes learned, fallbacks to the CPU path, refusals, memo sizes.
void ReportGpuParameters() {
    static std::atomic<std::int64_t> last{0};
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto previous = last.load(std::memory_order_relaxed);
    if (previous == 0) {
        last.compare_exchange_strong(previous, now);
        return;
    }
    if (now - previous < 10 || !last.compare_exchange_strong(previous, now)) return;
    auto& counters = GpuParameterCount();
    std::size_t good = 0, bad = 0;
    {
        auto& memo = GpuParameterStates();
        std::lock_guard lock(memo.mutex);
        for (const auto& [key, state] : memo.states) (state == 1 ? good : bad) += 1;
    }
    std::fprintf(stderr, "[gpuind] %lld s: gpu-parameter draws %llu (shapes learned %llu), fell back to the CPU path %llu, refused %llu, records without a pending writer (CPU path) %llu; shapes gpu %zu / cpu %zu\n", static_cast<long long>(now - previous), static_cast<unsigned long long>(counters.gpuSide.exchange(0)), static_cast<unsigned long long>(counters.learned.exchange(0)), static_cast<unsigned long long>(counters.fellBack.exchange(0)), static_cast<unsigned long long>(counters.refused.exchange(0)), static_cast<unsigned long long>(counters.notPending.exchange(0)), good, bad);
}

}

namespace {

bool EmptyIndirectSkip() {
    static const bool enabled = std::getenv("APS5_NO_EMPTY_INDIRECT_SKIP") == nullptr;
    return enabled;
}

struct CpuIndirectProgramsStorage {};

std::unordered_map<std::uint64_t, bool>& CpuIndirectPrograms() {
    return HostThreadLocal<std::unordered_map<std::uint64_t, bool>, CpuIndirectProgramsStorage>();
}

}

DrawVerdict Driver::draw(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::size_t packetOffset, std::string& rejected) {
    PerformanceTimer timing("Driver.Draw");
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    std::array<double, DrawDriverPhaseCount> phaseMs{};
    std::uint64_t captures = 0;
    auto phaseLap = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    DrawPhaseTiming phaseTiming{profile, phaseMs, phaseLap};
    if (profile && packetStartedAt() != std::chrono::steady_clock::time_point{}) phaseMs[DrawRowPrologue] = std::chrono::duration<double, std::milli>(phaseLap - packetStartedAt()).count();

    const auto drawn = [&] {
        phaseTiming.Phase(DrawRowVectors);
        if (!profile) return DrawVerdict::Drawn;
        auto& pending = pendingDrawPhases();
        pending.phases = true;
        pending.captures = captures;
        pending.ms = phaseMs;
        pending.tailAt = phaseLap;
        return DrawVerdict::Drawn;
    };
    const std::uint64_t traceToken = TraceShaderCorrelation() ? NextDrawTraceToken().fetch_add(1, std::memory_order_relaxed) + 1 : 0;
    if (traceToken != 0) {
        AgcDriver::CaptureTrace::LogBestEffort("draw-attempt token=%llu submission=%llu queue=%x offset=%zu",
                static_cast<unsigned long long>(traceToken), static_cast<unsigned long long>(submission.serial), submission.queue, packetOffset);
    }
    using Stage = ShaderRecompiler::ShaderStage;
    using Role = ShaderRecompiler::ProgramRole;

    static const std::uint64_t dumpTarget = [] { const char* text = std::getenv("APS5_DUMP_DRAW_SHADERS"); return text ? std::strtoull(text, nullptr, 16) : 0ull; }();

    static const std::uint64_t dumpSlot1 = [] { const char* text = std::getenv("APS5_DUMP_DRAW_SLOT1"); return text ? std::strtoull(text, nullptr, 16) : 0ull; }();

    static const bool lockedPrepare = std::getenv("APS5_LOCKED_DRAW_PREPARE") != nullptr;
    const bool useDrawEntries = drawEntries() && !ShaderRecompiler::DebugProbeActive() && dumpTarget == 0 && dumpSlot1 == 0;
    const bool registerKey = useDrawEntries && registerKeyEnabled();
    std::shared_ptr<VulkanDevice> localDevice;
    const bool pipelined = DrawPipeline::Active();
    static const bool pipelinedIndirect = std::getenv("APS5_NO_PIPELINED_INDIRECT") == nullptr && !DrawPipeline::PipelineIndirect();
    const auto drainBeforeInline = [&] {
        if (pipelined && pipelinedIndirect && DrawPipeline::Queue0().Busy()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Indirect);
    };
    const PublishedDraw* published = nullptr;
    // APS5_NO_DRAW_AHEAD=1 (#6188 off, an APS5_AB switch): the draw neither takes a published record nor an entry compiled ahead.
    const bool useDrawAhead = !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoDrawAhead);
    if (useDrawAhead && drawAhead != nullptr && registerKey && !lockedPrepare && !verifyDrawRecipe() && !verifyDrawEntries() && (localDevice = device.Load()) != nullptr) published = drawAhead->Find(submission, packet, queue, localDevice->Serial());
    Pm4::DrawParameters drawParameters{};
    bool traceIndirect = false;
    bool pipelineIndirect = false;
    if (published != nullptr) {
        drawParameters = published->parameters;
        if (drawParameters.indexed) GuestMemory::CheckRange(reinterpret_cast<const void*>(drawParameters.indexAddress), static_cast<std::size_t>(drawParameters.indexCount) * drawParameters.indexSize, drawParameters.indexSize);
        if (published->precheck == DrawPrecheckOutcome::Nothing) return DrawVerdict::Nothing;
        traceIndirect = TraceIndirectDraws();
    } else {
        try {
            drawParameters = Pm4::ResolveDraw(packet, queue);
        } catch (...) {
            TraceDrawAttemptException(traceToken, "resolve-draw", submission.serial, submission.queue, packetOffset, std::current_exception());
            throw;
        }
        pipelineIndirect = pipelined && drawParameters.indirect && DrawPipeline::PipelineIndirect();
        if (pipelined && drawParameters.indirect && !pipelineIndirect) {
            const auto& indirect = *drawParameters.indirect;
            auto& pipeline = DrawPipeline::Queue0();
            const bool overlaps = pipeline.Busy() && (pipeline.Overlaps(indirect.arguments, static_cast<std::size_t>(std::max<std::uint64_t>(indirect.RangeBytes(), indirect.recordBytes))) || (indirect.countIndirect && pipeline.Overlaps(indirect.countAddress, 4)));
            if (!pipelinedIndirect || overlaps) pipeline.Drain(DrawPipeline::DrainReason::Indirect);
        }
        if (const auto verdict = precheckDraw(queue, submission, packetOffset, packet, drawParameters, rejected, traceIndirect)) {
            TraceDrawAttemptFailure(traceToken, "precheck", "rejected", submission.serial, submission.queue, packetOffset, rejected);
            return *verdict;
        }
    }
    phaseTiming.Phase(DrawRowPrecheck);

    std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::defer_lock);
    if (lockedPrepare) {

        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        gpuLock.lock();
        timing.Mark("gpu_mutex_wait");
        phaseTiming.Phase(DrawRowLockWait);
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;

        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(DrawRowLabels);
    } else if (localDevice == nullptr && (localDevice = device.Load()) == nullptr) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        std::lock_guard createLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    timing.Mark("device_setup");
    phaseTiming.Phase(DrawRowVectors);

    std::uint64_t drawKey = 0;
    DrawKeyComponents drawKeyComponents;
    std::shared_ptr<DrawEntry> entry;
    std::shared_ptr<const DrawDecode> decode;
    std::shared_ptr<DrawEntry> dataEntry;
    std::uint64_t shapeKey = 0;
    std::uint64_t drawIdentity = 0, drawAlias = 0;
    std::vector<DrawUserPatch> userPatches;
    if (registerKey) {
        const auto keyStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        std::shared_ptr<const DrawRelocation> relocation;
        bool aliasFirst = false;
        // One registry pass feeds the identity key, the shader-set key and the register-key tail; the exact key does
        // not depend on any cache state, so it is built before the single drawCacheMutex acquisition of the lookup.
        // A draw-ahead record (#6188) carries the identity, shape and register key the scan built the same way.
        DrawRegistryWalk registryWalk;
        bool walked = false;
        const auto walk = [&]() -> const DrawRegistryWalk* {
            if (!walked) {
                WalkDrawRegistry(queue, *submission.shaders, registryWalk);
                walked = true;
            }
            return &registryWalk;
        };
        const bool relocating = drawRelocationEnabled();
        const bool usePublished = published != nullptr && !traceDrawCache();
        if (relocating) drawIdentity = usePublished ? published->identity : drawIdentityKey(*walk());
        if (usePublished) {
            shapeKey = published->shape;
            drawKey = published->key;
        } else {
            drawKey = drawRegisterKey(queue, *submission.shaders, localDevice->Serial(), &shapeKey, traceDrawCache() || profile ? &drawKeyComponents : nullptr, nullptr, nullptr, walk());
            if (GpuIndirectArgs() && drawParameters.indirect) drawKey = GpuParameterKey(drawKey, *drawParameters.indirect);
        }
        std::unique_lock cacheLock(drawCacheMutex);
        if (relocating) {
            learnPendingRelocation();
            const auto known = drawRelocations.find(drawIdentity);
            if (known != drawRelocations.end()) {
                auto& trial = drawRelocationTrials[drawIdentity];
                if (trial.idle != 0) --trial.idle;
                else relocation = known->second;
                aliasFirst = relocation != nullptr && trial.productive;
            }
        }
        if (aliasFirst) {
            // The alias reads guest memory (relocatedContentKey), so it is computed outside the lock as before.
            cacheLock.unlock();
            drawRegisterKey(queue, *submission.shaders, localDevice->Serial(), nullptr, nullptr, relocation.get(), &drawAlias, walk());
            if (GpuIndirectArgs() && drawParameters.indirect) drawAlias = GpuParameterKey(drawAlias, *drawParameters.indirect);
            cacheLock.lock();
        }
        ++drawEntryCounters.lookups;
        ++drawEntryCounters.registerKeyLookups;
        if (profile) drawEntryCounters.keyUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - keyStart).count();
        const auto found = drawCache.find(drawKey);
        if (found != drawCache.end()) {
            entry = found->second;
            decode = entry->decode;
        } else if (auto ahead = useDrawAhead ? findAheadEntry(drawKey) : nullptr; ahead != nullptr) {
            entry = std::move(ahead);
            decode = entry->decode;
            if (drawIdentity != 0) {
                if (relocation != nullptr && !aliasFirst) {
                    cacheLock.unlock();
                    drawRegisterKey(queue, *submission.shaders, localDevice->Serial(), nullptr, nullptr, relocation.get(), &drawAlias);
                    cacheLock.lock();
                }
                drawRelocationPending = {drawKey, drawAlias, drawIdentity};
            }
        } else {
            ++drawEntryCounters.absent;
            if (relocation != nullptr && !aliasFirst) {
                cacheLock.unlock();
                drawRegisterKey(queue, *submission.shaders, localDevice->Serial(), nullptr, nullptr, relocation.get(), &drawAlias, walk());
                if (GpuIndirectArgs() && drawParameters.indirect) drawAlias = GpuParameterKey(drawAlias, *drawParameters.indirect);
                cacheLock.lock();
            }
            if (drawIdentity != 0) drawRelocationPending = {drawKey, drawAlias, drawIdentity};
            if (drawAlias != 0) relocateDrawEntry(queue, drawAlias, entry, decode, userPatches);
            if (relocation != nullptr) {
                auto& trial = drawRelocationTrials[drawIdentity];
                ++trial.lookups;
                if (trial.lookups >= DrawRelocationTrial::Window) {
                    trial.productive = trial.hits * DrawRelocationTrial::MinimumHitShare >= trial.lookups;
                    if (!trial.productive) trial.idle = DrawRelocationTrial::IdleDraws;
                    trial.lookups = trial.hits = 0;
                }
            }
            if (traceDrawCache()) {
                auto& counters = drawEntryCounters;
                if (traceKeysEvicted.contains(drawKey)) ++counters.absentEvicted;
                else if (traceKeysSeen.contains(drawKey)) ++counters.absentNeverInserted;
                else if (traceShapes.contains(shapeKey)) ++counters.absentUserWords;
                else ++counters.absentNewRegisters;
            }
            if (decode == nullptr && drawDataHits() && !drawParameters.indirect) {
                if (const auto shape = drawShapes.find(shapeKey); shape != drawShapes.end()) {
                    const auto kept = drawCache.find(shape->second);
                    if (kept != drawCache.end() && kept->second->decode != nullptr && kept->second->shape == shapeKey) dataEntry = kept->second;
                }
                if (dataEntry != nullptr) ++drawEntryCounters.dataHits;
            }
        }
        if (traceDrawCache() || profile) traceDrawKeyLookup(queue, drawKeyComponents, found == drawCache.end());
        if (traceDrawCache()) {
            traceInsert(traceKeysSeen, drawKey);
            traceInsert(traceShapes, shapeKey);
        }
        maybeReportDrawCache(profile);
    }
    AheadDumpKey() = drawKey;
    if (decode != nullptr && !DecodeProgramsCurrent(*decode, *submission.shaders)) {
        userPatches.clear();
        std::lock_guard cacheLock(drawCacheMutex);
        eraseDrawEntry(drawKey, entry);
        ++drawEntryCounters.absent;
        entry = nullptr;
        decode = nullptr;
    }
    if (dataEntry != nullptr && !DecodeProgramsCurrent(*dataEntry->decode, *submission.shaders)) dataEntry = nullptr;
    phaseTiming.Phase(DrawRowKeyLookupValidate);

    if (dataEntry != nullptr) {
        auto refreshed = std::make_shared<DrawDecode>(*dataEntry->decode);
        for (auto& program : refreshed->programs) {
            ReadUserWords(queue, program, false);
            if (program.merged) InitializeMerged(queue, program, program.mergedPointer, program.mergedPointerRequired, false);
        }
        if (verifyDrawDataHits()) {
            const auto full = decodeDraw(queue, submission, localDevice->graphicsContext());
            if (!sameDecode(*refreshed, *full)) {
                std::size_t differing = 0;
                while (differing < full->programs.size() && differing < refreshed->programs.size() && full->programs[differing].userData == refreshed->programs[differing].userData) ++differing;
                char message[320];
                std::snprintf(message, sizeof(message), "APS5_VERIFY_DRAW_DATA_HITS: a data-only hit's decode differs from the full decode (key 0x%llx, shape 0x%llx, %zu / %zu programs, first with differing user words %zu, path %u / %u, target 0x%llx / 0x%llx)", static_cast<unsigned long long>(drawKey), static_cast<unsigned long long>(shapeKey), refreshed->programs.size(), full->programs.size(), differing, static_cast<unsigned>(refreshed->state.stages.path), static_cast<unsigned>(full->state.stages.path), static_cast<unsigned long long>(refreshed->state.color.address), static_cast<unsigned long long>(full->state.color.address));
                throw std::runtime_error(message);
            }
        }
        decode = std::move(refreshed);
    }
    try {
        resolveDrawDecode(queue, submission, localDevice->graphicsContext(), decode, registerKey, drawKey, profile);
    } catch (...) {
        TraceDrawAttemptFailure(traceToken, "decode", "exception", submission.serial, submission.queue, packetOffset, ExceptionReason(std::current_exception()));
        throw;
    }
    const auto& graphics = decode->state;
    auto pixel = decode->pixel;
    std::vector<DrawProgram> programs = decode->programs;
    for (const auto& patch : userPatches) programs[patch.program].userData[patch.index] = patch.value;
    auto& writablePrograms = programs;
    if (std::any_of(programs.begin(), programs.end(), [](const DrawProgram& program) { return SkippedShader(program.binary.codeAddress); })) return drawn();
    const bool emptyIndirectSkip = EmptyIndirectSkip();
    auto& cpuIndirectPrograms = CpuIndirectPrograms();
    const auto recordDraws = [&](const Pm4::DrawArguments& arguments) { return arguments.count != 0 && arguments.instances != 0 && (!drawParameters.indexed || !drawParameters.indirect->indexLimitKnown || arguments.firstVertexOrIndex < drawParameters.indexCount); };
    if (emptyIndirectSkip && drawParameters.indirect && !programs.empty()) {
        const auto known = cpuIndirectPrograms.find(programs.front().binary.codeAddress);
        if (known != cpuIndirectPrograms.end() && known->second) {
            const auto& indirect = *drawParameters.indirect;
            recordQueuedLabelsBeforeRead(submission.queue);
            const auto readStart = std::chrono::steady_clock::now();
            const auto count = std::min(indirect.countIndirect ? Pm4::ReadDrawCount(indirect) : indirect.count, indirect.count);
            bool any = false;
            for (std::uint32_t record = 0; record < count && !any; ++record) any = recordDraws(Pm4::ReadDrawArguments(indirect, record));
            if (!any) {
                Graphics::CountEmptyIndirectDraw(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
                timing.Mark("empty_indirect_skipped");
                return drawn();
            }
        }
    }
    const auto setMeshIndexBuffer = [&](const Pm4::DrawParameters& parameters) {
        if (!graphics.stages.mesh) return;
        auto& words = writablePrograms.front().userData;
        require(programs.front().firstUserSgpr == 0 && words.size() >= ShaderRecompiler::MeshIndexBufferUserWord + 4, "mesh program lacks the hidden user words");
        const auto descriptor = Graphics::MeshIndexBufferDescriptor(parameters);
        std::copy(descriptor.begin(), descriptor.end(), words.begin() + ShaderRecompiler::MeshIndexBufferUserWord);
    };
    if (!drawParameters.indirect) setMeshIndexBuffer(drawParameters);
    else if (graphics.stages.mesh) setMeshIndexBuffer(Pm4::DrawParameters{drawParameters.indexAddress, std::max(drawParameters.indexCount, 1u), drawParameters.indexSize, 1, 0, drawParameters.indexed});
    const std::vector<Role>& roles = decode->roles;
    static const bool tracePrograms = std::getenv("APS5_TRACE_DRAW_PROGRAMS") != nullptr;
    if (tracePrograms) {
        std::vector<DrawProgramTraceEntry> identities;
        identities.reserve(programs.size());
        for (std::size_t i = 0; i < programs.size(); ++i) {
            identities.push_back({programs[i].binary.stage, roles[i], programs[i].snapshot->codeAddress, programs[i].binary.codeAddress});
        }
        const auto line = FormatDrawProgramTrace(submission.serial, submission.queue, identities);
        std::fprintf(stderr, "%s\n", line.c_str());
    }
    phaseTiming.Phase(DrawRowDecode);

    const auto locate = [&](std::uint32_t location) -> std::optional<std::pair<std::size_t, std::size_t>> {
        const auto words = UserDataWords(programs, roles, location);
        if (words.empty()) return std::nullopt;
        return words.front();
    };
    if (drawParameters.indirect) {
        auto& indirect = *drawParameters.indirect;
        const auto sgprOf = [&](std::uint32_t location) -> std::int32_t {
            const auto word = locate(location);
            if (!word || word->first != 0) return -1;
            return static_cast<std::int32_t>(programs.front().firstUserSgpr + word->second);
        };
        indirect.baseVertexSgpr = sgprOf(indirect.baseVertexLocation);
        indirect.startInstanceSgpr = sgprOf(indirect.startInstanceLocation);
        indirect.drawIndexSgpr = sgprOf(indirect.drawIndexLocation);
    }
    // gpuind: whether the front stage may be compiled with GPU-side draw parameters, and whether
    // it is (learned per shader and SGPRs; a draw it does not suit falls back below).
    bool gpuParametersEligible = false;
    bool gpuParameters = false;
    std::uint64_t gpuParameterKey = 0;
    if (GpuIndirectArgs() && drawParameters.indirect && !programs.empty() && programs.front().binary.stage == Stage::Vertex && graphics.stages.path == Graphics::ShaderPath::Vertex && !graphics.rectList && !graphics.stages.mesh && !graphics.stages.tessellation && (drawParameters.indirect->baseVertexSgpr >= 0 || drawParameters.indirect->startInstanceSgpr >= 0)) {
        static const bool drawParametersCapability = [&] {
            const auto target = localDevice->Target();
            return std::ranges::find(target.supportedCapabilities, static_cast<std::uint32_t>(spv::CapabilityDrawParameters)) != target.supportedCapabilities.end();
        }();
        // Only a draw whose records recorded or queued GPU work still writes takes the GPU-side
        // parameters: reading those records waits for the GPU (the indirect-args waits). Records
        // nothing pending writes are read at once, and the CPU path's per-record direct draws bind
        // just the index and vertex ranges the records use, where a GPU-side draw binds (and
        // copies or scans) the whole index buffer and vertex descriptor ranges (long34: +2.5 s of
        // draw input work per 10 s when every such draw went GPU side).
        // APS5_GPU_INDIRECT_ARGS_ALWAYS=1: every eligible draw goes GPU side.
        static const bool always = std::getenv("APS5_GPU_INDIRECT_ARGS_ALWAYS") != nullptr;
        const auto& args = *drawParameters.indirect;
        const auto pendingWriter = [&](std::uint64_t address, std::uint64_t bytes) {
            return bytes != 0 && ((pipelined && DrawPipeline::Queue0().Overlaps(address, static_cast<std::size_t>(bytes))) || Graphics::Recorder::SnapshotWriteOverlaps(address, static_cast<std::size_t>(bytes)));
        };
        const bool pending = always || pendingWriter(args.arguments, args.RangeBytes()) || (args.countIndirect && pendingWriter(args.countAddress, 4));
        if (drawParametersCapability && pending) {
            gpuParameterKey = GpuParameterMemo::Key(programs.front().binary.codeAddress, drawParameters.indirect->baseVertexSgpr, drawParameters.indirect->startInstanceSgpr);
            const auto state = GpuParameterStates().Get(gpuParameterKey);
            gpuParametersEligible = state != 2;
            gpuParameters = state == 1;
        } else if (drawParametersCapability) {
            ++GpuParameterCount().notPending;
        }
    }
    const auto applyGpuParameters = [&](std::optional<ShaderRecompiler::ShaderVertexStageInfo>& info) {
        if (!info) return;
        info->drawBaseVertexSgpr = gpuParameters ? drawParameters.indirect->baseVertexSgpr : -1;
        info->drawStartInstanceSgpr = gpuParameters ? drawParameters.indirect->startInstanceSgpr : -1;
    };
    std::vector<ShaderRecompiler::MemoryRegion> memory;
    memory.reserve(2 * programs.size() + 32);
    AppendProgramMemory(memory, programs);
    std::vector<ShaderRecompiler::LinkedProgram> linked;
    linked.reserve(programs.size());
    for (std::size_t i = 0; i < programs.size(); ++i) {
        const auto& program = programs[i];
        linked.push_back({roles[i], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
    }
    timing.Mark("prepare");
    phaseTiming.Phase(DrawRowProgramPrepare);

    std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>> vertexInfos(programs.size());
    std::vector<std::vector<Graphics::DecodeRead>> decodeReads(programs.size());
    const auto decodeVertexInfo = [&](std::size_t i) {
        const auto& program = programs[i];
        if (program.binary.stage == Stage::Fragment || roles[i] == Role::GeometryBack) return;
        decodeReads[i].clear();
        vertexInfos[i] = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, &decodeReads[i], false, 8u - program.firstUserSgpr);
        static const bool unordered = std::getenv("APS5_NO_DECODE_READ_DRAIN") != nullptr;
        if (pipelined && !unordered && std::any_of(decodeReads[i].begin(), decodeReads[i].end(), [](const Graphics::DecodeRead& read) { return DrawPipeline::DrainBeforeRead(read.address, read.size); })) {
            decodeReads[i].clear();
            vertexInfos[i] = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, &decodeReads[i], false, 8u - program.firstUserSgpr);
        }
        if (vertexInfos[i]) vertexInfos[i]->paClVsOutCntl = PositionExportLayout(program.snapshot.get(), program.codeOffset, program.binary.code, readRegister(queue.context, 0x207), program.tailContinuation.get()) | graphics.layerExports;
        if (vertexInfos[i]) {
            vertexInfos[i]->depthRemapMode = graphics.depthRemap.mode;
            Graphics::ApplyPointSize(graphics, *vertexInfos[i]);
        }
        if (i == 0 && gpuParametersEligible) applyGpuParameters(vertexInfos[i]);
    };
    if (!registerKey) {
        for (std::size_t i = 0; i < programs.size(); ++i) decodeVertexInfo(i);
        phaseTiming.Phase(DrawRowDecode);
    }
    auto shaderMemory = std::make_unique<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter(), &QueryPendingSpan);
    std::vector<ShaderRecompiler::RecompileResult> results;
    std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> compiledStages;
    compiledStages.reserve(programs.size());
    std::vector<Graphics::CompiledShader> stages;
    results.reserve(programs.size() + 2u);
    stages.reserve(programs.size());
    std::uint32_t pushCursorBytes = 0;

    std::vector<const ShaderRecompiler::RecompileResult*> programResults(programs.size(), nullptr);

    std::vector<StageCapture> stageCaptures(programs.size());
    std::vector<std::shared_ptr<DispatchVariant>> matched(programs.size());
    std::vector<std::vector<ShaderRecompiler::MemoryRegion>> matchedRegions(programs.size());

    std::vector<std::shared_ptr<DispatchVariant>> fresh(programs.size());

    std::vector<bool> recompiled(programs.size(), false);
    std::vector<std::vector<std::pair<std::shared_ptr<DispatchVariant>, std::vector<ShaderRecompiler::MemoryRegion>>>> dataCandidates(programs.size());
    std::vector<bool> reused(programs.size(), false);
    std::vector<StageDataHit> stageData(programs.size());
    std::uint64_t dataStagesReused = 0, dataStagesCompiled = 0, dataVerified = 0;
    bool drawHit = false;
    bool verifyHit = false;
    const bool stageEntries = useDrawEntries && !(graphics.stages.tessellation && !drawParameters.indirect && drawParameters.instanceCount > 1u);
    try {
        lookupDraw(submission, localDevice, graphics, pixel, programs, roles, vertexInfos, stageEntries, registerKey, profile, drawKey, entry, matched, matchedRegions, drawHit, verifyHit, phaseTiming, phaseMs, dataEntry, dataCandidates, stageData);
    } catch (...) {
        TraceDrawAttemptException(traceToken, "lookup", submission.serial, submission.queue, packetOffset, std::current_exception());
        throw;
    }
    for (std::size_t i = 0; i < matched.size(); ++i) {
        if (matched[i] == nullptr || matched[i]->shaderDataPatches.empty()) continue;
        applyShaderDataPatches(*matched[i]);
        if (stageData[i].compiled != nullptr) stageData[i].compiled = patchDataWords(*matched[i], stageData[i].words);
    }
    const auto relocatedDecode = decode;
    if (!userPatches.empty()) {
        std::lock_guard cacheLock(drawCacheMutex);
        if (drawHit || verifyHit) {
            ++drawRelocationCounters.hits;
            ++drawRelocationTrials[drawIdentity].hits;
            if (const auto alias = drawAliases.find(drawAlias); alias != drawAliases.end()) {
                const auto donor = drawCache.find(alias->second);
                if (donor != drawCache.end() && drawCacheHits - donor->second->touched > drawCacheEntries() / 8) {
                    drawOrder.splice(drawOrder.begin(), drawOrder, donor->second->order);
                    donor->second->touched = drawCacheHits;
                    ++drawEntryCounters.touches;
                }
            }
        }
        if (profile && drawRelocationCounters.lookups >= 100000) reportDrawRelocation();
    }
    if (!userPatches.empty() && !drawHit) {
        auto current = std::make_shared<DrawDecode>(*decode);
        for (const auto& patch : userPatches) current->programs[patch.program].userData[patch.index] = patch.value;
        decode = std::move(current);
    }
    const bool patchedStages = std::any_of(stageData.begin(), stageData.end(), [](const StageDataHit& hit) { return hit.compiled != nullptr; });
    std::uint64_t stageDataVerified = 0;
    const auto target = graphics.hasColorTarget ? graphics.color.address : 0;
    const VulkanDevice::DrawTraceContext trace{traceToken, submission.serial, submission.queue, packetOffset, target};
    const auto* tracePtr = trace.token != 0 ? &trace : nullptr;
    if (trace.token != 0) {
        for (std::size_t i = 0; i < programs.size(); ++i) {
            const auto& program = programs[i];
            AgcDriver::CaptureTrace::LogBestEffort("draw-selected token=%llu submission=%llu queue=%x offset=%zu snapshot=0x%llx code=%llx entryOffset=%zu entry=%llx stage=%u role=%u drawCacheHit=%u variantHit=%u target=%llx",
                    static_cast<unsigned long long>(trace.token), static_cast<unsigned long long>(submission.serial), submission.queue, packetOffset,
                    static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(program.snapshot.get())), static_cast<unsigned long long>(program.snapshot->codeAddress), program.codeOffset,
                    static_cast<unsigned long long>(program.snapshot->codeAddress + program.codeOffset), static_cast<unsigned>(program.binary.stage), static_cast<unsigned>(roles[i]),
                    drawHit ? 1u : 0u, matched[i] != nullptr ? 1u : 0u, static_cast<unsigned long long>(target));
        }
    }

    if (registerKey) {

        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (matched[i] != nullptr && drawHit && !verifyDrawRecipe()) {
                if (matched[i]->vertexInfo != nullptr) vertexInfos[i] = *matched[i]->vertexInfo;
                continue;
            }
            if (!dataCandidates[i].empty() && !verifyDrawDataHits()) {
                const auto& info = dataCandidates[i].front().first->vertexInfo;
                if (info != nullptr) vertexInfos[i] = *info;
                continue;
            }
            decodeVertexInfo(i);
            if (matched[i] != nullptr && verifyDrawRecipe() && (vertexInfos[i].has_value() != (matched[i]->vertexInfo != nullptr) || (vertexInfos[i] && !sameVertexInfo(*vertexInfos[i], *matched[i]->vertexInfo)))) {
                static std::atomic<std::uint64_t> reports{0};
                if (reports.fetch_add(1) < 20) std::fprintf(stderr, "[draw-cache] verify: stage %zu (program 0x%llx) of a hit has a vertex stage info unlike its variant's\n", i, static_cast<unsigned long long>(programs[i].binary.codeAddress));
                std::lock_guard cacheLock(drawCacheMutex);
                ++drawEntryCounters.verifyDecodeMismatches;
            }
        }
        phaseTiming.Phase(DrawRowDecode);
    }

    const auto fold = [&](const ShaderRecompiler::RecompileResult& main, Pm4::DrawParameters& parameters) { FoldDrawOffsets(main, programs.front(), parameters); };

    const auto freshRuns = [&](std::size_t i, std::vector<std::pair<std::uint64_t, std::uint64_t>>& runs, std::vector<std::uint32_t>& words) {
        std::vector<ShaderRecompiler::MemoryRegion> regions(stageCaptures[i].regions.begin(), stageCaptures[i].regions.end());
        for (const auto& read : decodeReads[i]) regions.push_back({read.address, read.Bytes()});
        std::stable_sort(regions.begin(), regions.end(), [](const ShaderRecompiler::MemoryRegion& a, const ShaderRecompiler::MemoryRegion& b) { return a.guestAddress < b.guestAddress; });
        for (const auto& region : regions) {
            runs.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
            const auto count = region.bytes.size() / sizeof(std::uint32_t);
            const auto offset = words.size();
            words.resize(offset + count);
            std::memcpy(words.data() + offset, region.bytes.data(), count * sizeof(std::uint32_t));
        }
    };
    const auto verifyDataStage = [&](std::size_t i, const DispatchVariant& kept) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
        std::vector<std::uint32_t> words;
        freshRuns(i, runs, words);
        const auto& compiled = *stageCaptures[i].compiled;
        const char* what = runs != kept.runs ? "runs" : words != kept.words ? "words" : vertexInfos[i].has_value() != (kept.vertexInfo != nullptr) || (vertexInfos[i] && !sameVertexInfo(*vertexInfos[i], *kept.vertexInfo)) ? "vertex stage info" : compiled.variantId != kept.compiled->variantId ? "variant" : compiled.pushConstants != kept.compiled->pushConstants ? "push constants" : nullptr;
        ++dataVerified;
        if (what == nullptr) return;
        char message[320];
        std::snprintf(message, sizeof(message), "APS5_VERIFY_DRAW_DATA_HITS: stage %zu (program 0x%llx) of a data-only hit differs from its full build in its %s (%zu runs / %zu words kept, %zu / %zu fresh; variant 0x%llx / 0x%llx; key 0x%llx, shape 0x%llx)", i, static_cast<unsigned long long>(programs[i].binary.codeAddress), what, kept.runs.size(), kept.words.size(), runs.size(), words.size(), static_cast<unsigned long long>(kept.compiled->variantId), static_cast<unsigned long long>(compiled.variantId), static_cast<unsigned long long>(drawKey), static_cast<unsigned long long>(shapeKey));
        throw std::runtime_error(message);
    };
    const auto verifyStageData = [&](std::size_t i) {
        const auto& hit = stageData[i];
        const auto& kept = *matched[i];
        std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
        std::vector<std::uint32_t> words;
        freshRuns(i, runs, words);
        const auto& fresh = *stageCaptures[i].compiled;
        const auto& patched = *hit.compiled;
        const bool sameBindings = fresh.bindings.size() == patched.bindings.size() && std::equal(fresh.bindings.begin(), fresh.bindings.end(), patched.bindings.begin(), [](const auto& a, const auto& b) { return a.kind == b.kind && a.role == b.role && a.binding == b.binding && a.count == b.count && a.guestDescriptor == b.guestDescriptor; });
        const char* what = runs != kept.runs ? "runs" : words != hit.words ? "words" : fresh.variantId != patched.variantId ? "variant" : fresh.pushConstants != patched.pushConstants ? "push constants" : !sameBindings ? "bindings" : nullptr;
        ++stageDataVerified;
        if (what == nullptr) return;
        char message[256];
        std::snprintf(message, sizeof(message), "APS5_VERIFY_DRAW_DATA_HITS: stage %zu (program 0x%llx) of a stage data hit differs from its full build in its %s (%zu data positions refreshed)", i, static_cast<unsigned long long>(programs[i].binary.codeAddress), what, kept.dataPositions.size());
        throw std::runtime_error(message);
    };
    std::optional<Graphics::IndirectDrawPath> indirectCpu;
    std::size_t completedSubdraws = 0;
    std::vector<std::uint32_t> pushOffsets(programs.size(), 0);
    std::optional<ShaderRecompiler::MeshBarycentricLayout> meshBarycentrics;
    std::vector<std::size_t> resultIndex(programs.size(), 0);
    for (std::size_t i = 0; i < programs.size(); ++i) {
        if (roles[i] == Role::GeometryBack) continue;
        const auto& program = programs[i];
        pushCursorBytes = Graphics::StagePushOffset(pushCursorBytes, program.binary.stage, localDevice->GraphicsPipelineLibraries());
        pushOffsets[i] = pushCursorBytes;
        if (drawHit) {

            programResults[i] = stageData[i].compiled != nullptr ? stageData[i].compiled.get() : matched[i]->compiled.get();
            memory.insert(memory.end(), matchedRegions[i].begin(), matchedRegions[i].end());
        } else {
            const auto candidate = std::find_if(dataCandidates[i].begin(), dataCandidates[i].end(), [&](const auto& kept) { return kept.first->pushOffset == pushCursorBytes; });
            if (candidate != dataCandidates[i].end() && !verifyDrawDataHits()) {
                matched[i] = candidate->first;
                matchedRegions[i] = std::move(candidate->second);
                reused[i] = true;
                programResults[i] = stageData[i].compiled != nullptr ? stageData[i].compiled.get() : matched[i]->compiled.get();
                memory.insert(memory.end(), matchedRegions[i].begin(), matchedRegions[i].end());
            } else {
                if (!dataCandidates[i].empty() && !verifyDrawDataHits()) decodeVertexInfo(i);
                if (program.binary.stage == ShaderRecompiler::ShaderStage::Fragment) {
                    for (auto previous = i; previous-- > 0;) {
                        if (programResults[previous] == nullptr) continue;
                        pixel = Graphics::DefaultUnexportedInputs(decode->pixel, programResults[previous]->parameterExports);
                        break;
                    }
                }
                compiledStages.push_back(materializeDrawStage(i, pushCursorBytes, queue, submission, packetOffset, trace.token, programs, graphics, pixel, vertexInfos, memory, linked, drawParameters, localDevice, *shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected, meshBarycentrics));
                if (!rejected.empty()) return DrawVerdict::Rejected;
                programResults[i] = compiledStages.back().get();
                if (candidate != dataCandidates[i].end()) verifyDataStage(i, *candidate->first);
                if (stageData[i].compiled != nullptr && verifyDrawDataHits()) verifyStageData(i);
            }
            if (dataEntry != nullptr && reused[i]) ++dataStagesReused;
            else if (dataEntry != nullptr) ++dataStagesCompiled;
        }
        if (i == 0 && drawParameters.indirect) {
            // gpuind: the front stage as compiled (or served by a hit) says whether it reads the
            // draw parameters on the GPU. A GPU-parameter stage only ever draws GPU side (or through
            // the recorder's per-record fallback, which passes the records as BaseVertex /
            // BaseInstance too); the CPU path here patches the user words, so it needs the old stage.
            gpuParameters = vertexInfos[0] && (vertexInfos[0]->drawBaseVertexSgpr >= 0 || vertexInfos[0]->drawStartInstanceSgpr >= 0);
            const auto rematerialize = [&](bool withGpuParameters) {
                gpuParameters = withGpuParameters;
                applyGpuParameters(vertexInfos[0]);
                compiledStages.push_back(materializeDrawStage(0, pushCursorBytes, queue, submission, packetOffset, traceToken, programs, graphics, pixel, vertexInfos, memory, linked, drawParameters, localDevice, *shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected, meshBarycentrics));
                programResults[0] = compiledStages.back().get();
            };
            indirectCpu = ClassifyIndirectDraw(*programResults[0], graphics, programs.front(), localDevice, drawParameters, traceIndirect, gpuParameters);
            auto& counters = GpuParameterCount();
            if (profile) ReportGpuParameters();
            if (gpuParameters && indirectCpu) {
                ++counters.fellBack;
                rematerialize(false);
                if (!rejected.empty()) return DrawVerdict::Rejected;
                indirectCpu = ClassifyIndirectDraw(*programResults[0], graphics, programs.front(), localDevice, drawParameters, traceIndirect, false);
            } else if (gpuParameters) {
                ++counters.gpuSide;
            } else if (gpuParametersEligible && indirectCpu == Graphics::IndirectDrawPath::NotFolded) {
                if (!GpuParametersDrawSide(*programResults[0], graphics, programs.front(), localDevice, drawParameters)) {
                    GpuParameterStates().Set(gpuParameterKey, 2);
                } else {
                    bool refused = false;
                    try {
                        rematerialize(true);
                    } catch (const Graphics::LiveSnapshotInvalidated&) {
                        throw;
                    } catch (const std::exception& error) {
                        refused = true;
                        ++counters.refused;
                        static std::atomic<std::uint64_t> reports{0};
                        if (reports.fetch_add(1) < 16) std::fprintf(stderr, "[gpuind] shader 0x%llx: GPU-side draw parameters refused, its indirect draws read the records on the CPU: %s\n", static_cast<unsigned long long>(programs.front().binary.codeAddress), error.what());
                    }
                    if (!refused && !rejected.empty()) {
                        refused = true;
                        ++counters.refused;
                        static std::atomic<std::uint64_t> reports{0};
                        if (reports.fetch_add(1) < 16) std::fprintf(stderr, "[gpuind] shader 0x%llx: GPU-side draw parameters rejected, its indirect draws read the records on the CPU: %s\n", static_cast<unsigned long long>(programs.front().binary.codeAddress), rejected.c_str());
                        rejected.clear();
                    }
                    const auto fresh = refused ? std::optional<Graphics::IndirectDrawPath>(Graphics::IndirectDrawPath::NotFolded) : ClassifyIndirectDraw(*programResults[0], graphics, programs.front(), localDevice, drawParameters, traceIndirect, true);
                    if (fresh) {
                        GpuParameterStates().Set(gpuParameterKey, 2);
                        rejected.clear();
                        rematerialize(false);
                        if (!rejected.empty()) return DrawVerdict::Rejected;
                        indirectCpu = ClassifyIndirectDraw(*programResults[0], graphics, programs.front(), localDevice, drawParameters, traceIndirect, false);
                    } else {
                        GpuParameterStates().Set(gpuParameterKey, 1);
                        ++counters.learned;
                        ++counters.gpuSide;
                        indirectCpu.reset();
                    }
                }
            }
            if (EmptyIndirectSkip()) CpuIndirectPrograms()[programs.front().binary.codeAddress] = indirectCpu.has_value();
        }
        const auto& result = *programResults[i];
        if (i == 0 && !drawParameters.indirect) fold(result, drawParameters);
        try {
            const auto slotEnd = (pushCursorBytes / Graphics::PipelinePushSlotBytes + 1u) * Graphics::PipelinePushSlotBytes;
            require(result.pushConstants.size() <= slotEnd - pushCursorBytes, "stage push constants exceed the pipeline push constant block");
            stages.push_back({program.binary.stage, &result, result.pushConstants.empty() ? 0u : pushCursorBytes});
            pushCursorBytes += static_cast<std::uint32_t>(result.pushConstants.size());
        } catch (...) {
            TraceMaterializationFailure(trace.token, submission, packetOffset, "push-constant-validation", completedSubdraws, std::current_exception());
            throw;
        }
    }

    if (stageDataVerified != 0) {
        std::lock_guard cacheLock(drawCacheMutex);
        drawEntryCounters.stageDataVerified += stageDataVerified;
    }
    if (dataEntry != nullptr) {
        std::lock_guard cacheLock(drawCacheMutex);
        drawEntryCounters.dataStagesReused += dataStagesReused;
        drawEntryCounters.dataStagesCompiled += dataStagesCompiled;
        drawEntryCounters.dataVerified += dataVerified;
    }
    const bool shapeRefresh = Graphics::DrawPipelineEnabled();
    const auto refreshSeenTwice = [&] {
        std::uint64_t data = shapeKey;
        bool stored = false;
        const auto mix = [&data](std::uint64_t value) {
            data ^= value + 0x9e3779b97f4a7c15ull + (data << 6u) + (data >> 2u);
        };
        for (std::size_t i = 0; i < programs.size(); ++i) {
            const auto& capture = stageCaptures[i];
            if (reused[i] || capture.compiled == nullptr || !CacheableResult(*capture.compiled)) continue;
            stored = true;
            const auto& words = programs[i].userData;
            for (std::size_t w = 0; w < words.size(); ++w) {
                if (w >= 64u || (capture.userDataMask >> w & 1u) != 0) mix((static_cast<std::uint64_t>(i) << 40u) ^ (static_cast<std::uint64_t>(w) << 32u) ^ words[w]);
            }
        }
        if (!stored) return false;
        static std::array<std::atomic<std::uint64_t>, 4096> lastData{};
        return lastData[shapeKey % lastData.size()].exchange(data, std::memory_order_relaxed) == data;
    };
    const bool shapeNeedsEntry = shapeRefresh && registerKey && shapeKey != 0 && drawDataHits() && !drawParameters.indirect && dataEntry != nullptr && dataStagesCompiled != 0 && refreshSeenTwice();
    cacheDrawStages(stageEntries, drawHit, drawParameters, indirectCpu, programs, stageCaptures, vertexInfos, decodeReads, verifyHit, matched, stageData, fresh, drawKey, registerKey, decode, phaseTiming, reused, entry, shapeKey, shapeNeedsEntry);
    timing.Mark("shader_compile_and_link");

    for (const auto& reads : decodeReads) {
        for (const auto& read : reads) memory.push_back({read.address, read.Bytes()});
    }

    const auto regionBounds = [](const ShaderRecompiler::MemoryRegion& region) { return DrawPipeline::Range{region.guestAddress, region.guestAddress + region.bytes.size()}; };
    if (pipelineIndirect && DrawPipeline::Queue0().DrainIfOverlaps(memory, regionBounds, DrawPipeline::DrainReason::Capture)) return draw(queue, packet, submission, packetOffset, rejected);
    if (pipelined && !deferredLabels().labels.empty()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Labels);
    if (recordQueuedLabelsAfterCapture(submission.queue, memory)) return draw(queue, packet, submission, packetOffset, rejected);

    bool rectListBuilt = false;

    std::size_t rectIndex = 0;
    const auto buildRectList = [&] {
        try {
            phaseTiming.Phase(DrawRowVectors);
            require(programs.size() == 2 && programResults[0] != nullptr && programResults[1] != nullptr, "rect-list requires vertex and fragment programs");
            auto rectangle = DrawRectangle(*programs[0].snapshot, programs[1].snapshot, programResults[0]->variantId, programResults[1]->variantId, localDevice->Target());
            auto& control = graphics.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT ? rectangle.lastVertexControl : rectangle.control;
            if (rectListBuilt) {
                results[rectIndex] = std::move(control);
                results[rectIndex + 1] = std::move(rectangle.evaluation);
                phaseTiming.Phase(DrawRowRectList);
                return;
            }
            require(stages.size() == 2, "rect-list requires vertex and fragment programs");
            results.reserve(programs.size() + 2u);
            rectIndex = results.size();
            results.push_back(std::move(control));
            results.push_back(std::move(rectangle.evaluation));
            stages.insert(stages.begin() + 1, {{Stage::TessellationControl, &results[rectIndex], 0}, {Stage::TessellationEvaluation, &results[rectIndex + 1], 0}});
            rectListBuilt = true;
            phaseTiming.Phase(DrawRowRectList);
        } catch (...) {
            TraceMaterializationFailure(trace.token, submission, packetOffset, "rect-list", completedSubdraws, std::current_exception());
            throw;
        }
    };
    if (graphics.rectList) buildRectList();
    if (!programs.empty() && programResults.back() != nullptr && programResults.back()->barycentricEmulation.active && graphics.stages.path == Graphics::ShaderPath::Geometry) {
        require(localDevice->Target().mesh.has_value(), "a pixel shader on the mesh path reads barycentrics, and this device has neither VK_KHR_fragment_shader_barycentric nor mesh shaders (VK_EXT_mesh_shader), so they cannot be provided or emulated");
        require(!graphics.rectList && graphics.stages.mesh && programs.size() == 2 && programResults[0] != nullptr && programResults[1] != nullptr && stages.size() == 2, "a pixel shader that reads barycentrics needs exactly one mesh program before it to emulate them");
        meshBarycentrics = ShaderRecompiler::MeshBarycentricLayoutOf(*programResults[1]);
        if (programResults[0]->meshBarycentrics != meshBarycentrics) {
            const auto pushBytes = programResults[0]->pushConstants.size();
            const auto exports = programResults[0]->parameterExports;
            compiledStages.push_back(materializeDrawStage(0, pushOffsets[0], queue, submission, packetOffset, traceToken, programs, graphics, pixel, vertexInfos, memory, linked, drawParameters, localDevice, *shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected, meshBarycentrics));
            if (!rejected.empty()) return DrawVerdict::Rejected;
            programResults[0] = compiledStages.back().get();
            stages[0].program = programResults[0];
            require(programResults[0]->pushConstants.size() == pushBytes && programResults[0]->parameterExports == exports, "pairing the mesh program with its pixel shader changed the push constants or the parameter exports the pixel shader was built for");
        }
        require(programResults[0]->meshBarycentrics == meshBarycentrics, "the mesh program was compiled for a different barycentric layout than its pixel shader");
    } else if (!graphics.rectList && !programs.empty() && programResults.back() != nullptr && programResults.back()->barycentricEmulation.active) {
        require(graphics.stages.path == Graphics::ShaderPath::Vertex && programs.size() == 2 && programResults[0] != nullptr && stages.size() == 2, "a pixel shader that reads barycentrics without VK_KHR_fragment_shader_barycentric needs a vertex shader before it");
        const auto geometryLimits = localDevice->GeometryLimits();
        results.reserve(programs.size() + 2u);
        if (geometryLimits || !localDevice->Target().tessellation) {
            results.push_back(ShaderRecompiler::BuildBarycentricGeometryShader(*programResults[0], *programResults[1], localDevice->Target(), geometryLimits));
            stages.insert(stages.begin() + 1, Graphics::CompiledShader{Stage::Geometry, &results.back(), 0});
        } else {
            require((graphics.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || graphics.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP) && !graphics.primitiveRestart, "a pixel shader that reads barycentrics without VK_KHR_fragment_shader_barycentric or geometry shaders needs a triangle list or strip without primitive restart");
            auto patches = ShaderRecompiler::BuildBarycentricTessellationShaders(*programResults[0], *programResults[1], localDevice->Target(), localDevice->MaxFragmentInputComponents(), graphics.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT ? 2u : 0u);
            const auto first = results.size();
            results.push_back(std::move(patches.control));
            results.push_back(std::move(patches.evaluation));
            stages.insert(stages.begin() + 1, {{Stage::TessellationControl, &results[first], 0}, {Stage::TessellationEvaluation, &results[first + 1], 0}});
        }
    }
    std::vector<Graphics::GuestMemorySnapshot> snapshots;
    const auto snapshot = [&] {
        snapshots.clear();
        snapshots.reserve(memory.size());
        for (const auto& region : memory) snapshots.push_back({region.guestAddress, region.bytes});
        std::sort(snapshots.begin(), snapshots.end(), [](const Graphics::GuestMemorySnapshot& left, const Graphics::GuestMemorySnapshot& right) { return left.address != right.address ? left.address < right.address : left.bytes.size() < right.bytes.size(); });
    };
    snapshot();
    localDevice->PrimeBindlessDraw(stages);
    timing.Mark("post_compile_prepare");
    const auto lockForDraw = [&] {
        if (gpuLock.owns_lock()) return;
        phaseTiming.Phase(DrawRowVectors);
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        gpuLock.lock();
        timing.Mark("gpu_mutex_wait");
        phaseTiming.Phase(DrawRowLockWait);

        if (auto current = device.Load(); current != nullptr && current != localDevice) {
            static std::atomic<std::uint64_t> replaced{0};
            std::fprintf(stderr, "[draw] device replaced during unlocked preparation (%llu)\n", static_cast<unsigned long long>(++replaced));
            localDevice = std::move(current);
        }

        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(DrawRowLabels);
    };
    if (drawParameters.indirect && indirectCpu && pipelineIndirect && deferredLabels().labels.empty() && !graphics.rectList) {
        const auto indirect = *drawParameters.indirect;
        auto& pipeline = DrawPipeline::Queue0();
        const std::array<DrawPipeline::Range, 2> argumentRanges{DrawPipeline::Range{indirect.arguments, indirect.arguments + indirect.RangeBytes()}, indirect.countIndirect ? DrawPipeline::Range{indirect.countAddress, indirect.countAddress + 4} : DrawPipeline::Range{0, 0}};
        if (pipeline.DrainIfOverlaps(std::span<const DrawPipeline::Range>(argumentRanges), DrawPipeline::DrainReason::Indirect)) pipeline.Note(DrawPipeline::Event::IndirectArgsDrain);
        recordQueuedLabelsBeforeRead(submission.queue);
        const auto readStart = std::chrono::steady_clock::now();
        const auto count = std::min(indirect.countIndirect ? Pm4::ReadDrawCount(indirect) : indirect.count, indirect.count);
        std::vector<Pm4::DrawArguments> records;
        for (std::uint32_t record = 0; record < count; ++record) records.push_back(Pm4::ReadDrawArguments(indirect, record));
        Graphics::CountIndirectDraw(*indirectCpu, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        const auto baseVertexWord = locate(indirect.baseVertexLocation);
        const auto startInstanceWord = locate(indirect.startInstanceLocation);
        const auto drawIndexWord = indirect.drawIndexEnabled ? locate(indirect.drawIndexLocation) : std::nullopt;
        struct ExpandedDraw {
            Pm4::DrawParameters direct;
            std::vector<Graphics::CompiledShader> stages;
            std::vector<Graphics::GuestMemorySnapshot> snapshots;
            std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> patched;
        };
        std::vector<ExpandedDraw> expanded;
        std::vector<DrawPipeline::Range> writes;
        std::vector<DrawPipeline::Range> reads;
        std::vector<const ShaderRecompiler::RecompileResult*> current(programResults.begin(), programResults.end());
        const auto enqueueExpanded = [&] {
            if (expanded.empty()) return;
            phaseTiming.Phase(DrawRowVectors);
            const auto commitQueue = submission.queue;
            const auto commitPacket = (packet[0] >> 8u) & 0xffu;
            const auto epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
            auto commit = [this, localDevice = std::move(localDevice), decode = std::move(decode), expanded = std::move(expanded), commitQueue, commitPacket, epoch, programs = std::move(programs), results = std::move(results), memory = std::move(memory), linked = std::move(linked), stageCaptures = std::move(stageCaptures), matched = std::move(matched), matchedRegions = std::move(matchedRegions), fresh = std::move(fresh), decodeReads = std::move(decodeReads), shaderMemory = std::move(shaderMemory), entry = std::move(entry), dataEntry = std::move(dataEntry), compiledStages = std::move(compiledStages), stageData = std::move(stageData)] {
                DrawPipeline::FollowEpoch(epoch);
                GuestMemory::SetCurrentPacket(commitPacket, commitQueue);
                for (const auto& item : expanded) commitDraw(localDevice, commitQueue, decode->state, item.direct, item.stages, item.snapshots, nullptr, {}, 0);
            };
            auto held = std::make_shared<decltype(commit)>(std::move(commit));
            pipeline.Enqueue([held] { (*held)(); }, std::move(writes), 0, {}, std::move(reads));
            pipeline.Note(DrawPipeline::Event::IndirectCommit);
            phaseTiming.Phase(DrawRowLockWait);
        };
        for (std::uint32_t record = 0; record < records.size(); ++record) {
            const auto& arguments = records[record];
            if (traceIndirect) std::fprintf(stderr, "[draw]   record %u: count %u instances %u first %u vertexOffset %u startInstance %u\n", record, arguments.count, arguments.instances, arguments.firstVertexOrIndex, arguments.vertexOffset, arguments.firstInstance);
            if (arguments.count == 0 || arguments.instances == 0) continue;
            std::set<std::size_t> patched;
            const auto patch = [&](const std::optional<std::pair<std::size_t, std::size_t>>& word, std::uint32_t value) {
                if (!word) return;
                programs[word->first].userData[word->second] = value;
                patched.insert(word->first);
            };
            patch(baseVertexWord, indirect.recordBytes == 20 ? arguments.vertexOffset : arguments.firstVertexOrIndex);
            patch(startInstanceWord, arguments.firstInstance);
            patch(drawIndexWord, record);
            if (graphics.stages.tessellation) {
                for (std::size_t i = 0; i < programs.size(); ++i) {
                    if (programs[i].binary.stage == Stage::TessellationControl || programs[i].binary.stage == Stage::TessellationEvaluation) patched.insert(i);
                }
            }
            Pm4::DrawParameters direct{0, arguments.count, 0, arguments.instances, drawParameters.flags, drawParameters.indexed, 0, 0};
            if (drawParameters.indexed) {
                if (arguments.firstVertexOrIndex >= drawParameters.indexCount) continue;
                direct.indexAddress = drawParameters.indexAddress + static_cast<std::uint64_t>(arguments.firstVertexOrIndex) * drawParameters.indexSize;
                direct.indexCount = std::min(arguments.count, drawParameters.indexCount - arguments.firstVertexOrIndex);
                direct.indexSize = drawParameters.indexSize;
            } else {
                direct.firstVertex = indirect.indxOffset;
            }
            if (graphics.stages.mesh) {
                setMeshIndexBuffer(direct);
                patched.insert(0);
            }
            ExpandedDraw item{direct, stages, {}, {}};
            for (const auto programIndex : patched) {
                if (programResults[programIndex] == nullptr) continue;
                const auto pushBytes = programResults[programIndex]->pushConstants.size();
                decodeVertexInfo(programIndex);
                auto patchedResult = materializeDrawStage(programIndex, pushOffsets[programIndex], queue, submission, packetOffset, traceToken, programs, graphics, pixel, vertexInfos, memory, linked, programs[programIndex].binary.stage == Stage::TessellationControl || programs[programIndex].binary.stage == Stage::TessellationEvaluation ? direct : drawParameters, localDevice, *shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected, meshBarycentrics);
                if (!rejected.empty()) {
                    enqueueExpanded();
                    return DrawVerdict::Rejected;
                }
                require(patchedResult->pushConstants.size() == pushBytes, "patched program changed its push constant layout");
                for (auto& stage : item.stages) {
                    if (stage.program == programResults[programIndex]) stage.program = patchedResult.get();
                }
                current[programIndex] = patchedResult.get();
                item.patched.push_back(std::move(patchedResult));
            }
            fold(*current[0], item.direct);
            for (const auto& region : memory) item.snapshots.push_back({region.guestAddress, region.bytes});
            if (auto known = localDevice->KnownDrawRejection(graphics, item.stages)) {
                enqueueExpanded();
                rejected = std::move(*known);
                return DrawVerdict::Rejected;
            }
            const auto recordWrites = drawWriteRanges(graphics, item.stages);
            writes.insert(writes.end(), recordWrites.begin(), recordWrites.end());
            if (item.direct.indexed && item.direct.indexSize != 0 && item.direct.indexCount != 0) {
                const auto bytes = static_cast<std::uint64_t>(item.direct.indexCount) * item.direct.indexSize;
                if (item.direct.indexAddress <= std::numeric_limits<std::uint64_t>::max() - bytes) reads.emplace_back(item.direct.indexAddress, item.direct.indexAddress + bytes);
            }
            expanded.push_back(std::move(item));
        }
        enqueueExpanded();
        timing.Mark("draw_enqueued");
        return drawn();
    }
    if (drawParameters.indirect && indirectCpu) {
        if (pipelined && !pipelinedIndirect) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Indirect);
        const auto indirect = *drawParameters.indirect;
        results.reserve(programs.size() + 2u);
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (programResults[i] == nullptr) continue;
            resultIndex[i] = results.size();
            results.push_back(ShaderRecompiler::RecompileResult(*programResults[i]));
            for (auto& stage : stages) {
                if (stage.program == programResults[i]) stage.program = &results[resultIndex[i]];
            }
            programResults[i] = &results[resultIndex[i]];
        }
        recordQueuedLabelsBeforeRead(submission.queue);
        const auto readStart = std::chrono::steady_clock::now();
        const auto count = std::min(indirect.countIndirect ? Pm4::ReadDrawCount(indirect) : indirect.count, indirect.count);
        std::vector<Pm4::DrawArguments> records;
        for (std::uint32_t record = 0; record < count; ++record) records.push_back(Pm4::ReadDrawArguments(indirect, record));
        Graphics::CountIndirectDraw(*indirectCpu, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        const auto baseVertexWords = UserDataWords(programs, roles, indirect.baseVertexLocation);
        const auto startInstanceWords = UserDataWords(programs, roles, indirect.startInstanceLocation);
        const auto drawIndexWords = indirect.drawIndexEnabled ? UserDataWords(programs, roles, indirect.drawIndexLocation) : std::vector<std::pair<std::size_t, std::size_t>>{};
        const auto drawable = [&](const Pm4::DrawArguments& arguments) { return arguments.count != 0 && arguments.instances != 0 && (!drawParameters.indexed || !indirect.indexLimitKnown || arguments.firstVertexOrIndex < drawParameters.indexCount); };
        std::size_t lastDrawable = records.size();
        for (std::size_t record = 0; record < records.size(); ++record) {
            if (drawable(records[record])) lastDrawable = record;
        }
        for (std::uint32_t record = 0; record < records.size(); ++record) {
            const auto& arguments = records[record];
            if (traceIndirect) std::fprintf(stderr, "[draw]   record %u: count %u instances %u first %u vertexOffset %u startInstance %u\n", record, arguments.count, arguments.instances, arguments.firstVertexOrIndex, arguments.vertexOffset, arguments.firstInstance);
            if (arguments.count == 0 || arguments.instances == 0) continue;
            std::set<std::size_t> patched;
            const auto patch = [&](const std::vector<std::pair<std::size_t, std::size_t>>& words, std::uint32_t value) {
                for (const auto& [program, word] : words) {
                    writablePrograms[program].userData[word] = value;
                    patched.insert(program);
                }
            };
            patch(baseVertexWords, indirect.recordBytes == 20 ? arguments.vertexOffset : arguments.firstVertexOrIndex);
            patch(startInstanceWords, arguments.firstInstance);
            patch(drawIndexWords, record);
            if (graphics.stages.tessellation) {
                for (std::size_t i = 0; i < programs.size(); ++i) {
                    if (programs[i].binary.stage == Stage::TessellationControl || programs[i].binary.stage == Stage::TessellationEvaluation) patched.insert(i);
                }
            }
            Pm4::DrawParameters direct{0, arguments.count, 0, arguments.instances, drawParameters.flags, drawParameters.indexed, 0, 0};
            if (drawParameters.indexed) {
                if (indirect.indexLimitKnown) {
                    if (arguments.firstVertexOrIndex >= drawParameters.indexCount) continue;
                    direct.indexCount = std::min(arguments.count, drawParameters.indexCount - arguments.firstVertexOrIndex);
                } else {
                    const auto end = static_cast<std::uint64_t>(arguments.firstVertexOrIndex) + arguments.count;
                    require(end <= std::numeric_limits<std::uint32_t>::max(), "indirect draw index range overflow");
                    require(end * drawParameters.indexSize <= std::numeric_limits<std::uint64_t>::max() - drawParameters.indexAddress, "indirect draw index range address overflow");
                    direct.indexCount = arguments.count;
                }
                direct.indexAddress = drawParameters.indexAddress + static_cast<std::uint64_t>(arguments.firstVertexOrIndex) * drawParameters.indexSize;
                direct.indexSize = drawParameters.indexSize;
            } else {
                direct.firstVertex = indirect.indxOffset;
            }
            if (graphics.stages.mesh) {
                setMeshIndexBuffer(direct);
                patched.insert(0);
            }
            for (const auto programIndex : patched) {
                auto& result = results[resultIndex[programIndex]];
                const auto pushBytes = result.pushConstants.size();
                decodeVertexInfo(programIndex);
                result = *materializeDrawStage(programIndex, pushOffsets[programIndex], queue, submission, packetOffset, trace.token, programs, graphics, pixel, vertexInfos, memory, linked, programs[programIndex].binary.stage == Stage::TessellationControl || programs[programIndex].binary.stage == Stage::TessellationEvaluation ? direct : drawParameters, localDevice, *shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected, meshBarycentrics);
                if (!rejected.empty()) return DrawVerdict::Rejected;
                require(result.pushConstants.size() == pushBytes, "patched program changed its push constant layout");
            }
            fold(*programResults[0], direct);
            if (graphics.rectList && patched.contains(0)) buildRectList();
            snapshot();
            if (auto known = localDevice->KnownDrawRejection(graphics, stages)) {
                rejected = std::move(*known);
                return DrawVerdict::Rejected;
            }
            if (pipelined && pipelinedIndirect && record == lastDrawable && !gpuLock.owns_lock() && deferredLabels().labels.empty()) {
                phaseTiming.Phase(DrawRowVectors);
                auto writes = drawWriteRanges(graphics, stages);
                const auto commitQueue = submission.queue;
                const auto commitPacket = (packet[0] >> 8u) & 0xffu;
                const auto epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
                auto commit = [this, localDevice = std::move(localDevice), decode = std::move(decode), direct, stages = std::move(stages), snapshots = std::move(snapshots), commitQueue, commitPacket, epoch, writablePrograms = std::move(writablePrograms), results = std::move(results), memory = std::move(memory), linked = std::move(linked), stageCaptures = std::move(stageCaptures), matched = std::move(matched), matchedRegions = std::move(matchedRegions), fresh = std::move(fresh), decodeReads = std::move(decodeReads), shaderMemory = std::move(shaderMemory), entry = std::move(entry), dataEntry = std::move(dataEntry), compiledStages = std::move(compiledStages), stageData = std::move(stageData)] {
                    DrawPipeline::FollowEpoch(epoch);
                    GuestMemory::SetCurrentPacket(commitPacket, commitQueue);
                    commitDraw(localDevice, commitQueue, decode->state, direct, stages, snapshots, nullptr, {}, 0, nullptr);
                };
                auto held = std::make_shared<decltype(commit)>(std::move(commit));
                DrawPipeline::Queue0().Enqueue([held] { (*held)(); }, std::move(writes));
                phaseTiming.Phase(DrawRowLockWait);
                timing.Mark("draw_enqueued");
                return drawn();
            }
            drainBeforeInline();
            lockForDraw();
            noteDrawWriters(stages, submission.queue);
            phaseTiming.Phase(DrawRowVectors);
            localDevice->Draw(graphics, direct, stages, snapshots, nullptr, nullptr, nullptr, tracePtr);
            phaseTiming.Phase(DrawRowGraphics);
        }
        timing.Mark("draw_and_resource_release");
        return drawn();
    }

    std::vector<std::shared_ptr<DispatchVariant>> recipeStages;
    if (registerKey && !drawParameters.indirect && !patchedStages && Graphics::DrawRecipes()) {
        recipeStages.reserve(programs.size());
        for (std::size_t i = 0; i < programs.size(); ++i) recipeStages.push_back(drawHit ? matched[i] : fresh[i]);
        if (std::all_of(recipeStages.begin(), recipeStages.end(), [](const std::shared_ptr<DispatchVariant>& variant) { return variant == nullptr; })) recipeStages.clear();
    }
    std::shared_ptr<const DrawRecipe> recipe;
    if (drawHit && !recipeStages.empty()) {
        recipe = findDrawRecipe(drawKey, recipeStages);
        if (recipe == nullptr) VulkanDevice::NoteDrawRecipeMiss(VulkanDevice::DrawRecipePrecheck::NoRecipe);
    }
    if (recipe == nullptr) {
        if (auto known = localDevice->KnownDrawRejection(graphics, stages)) {
            rejected = std::move(*known);
            return DrawVerdict::Rejected;
        }
    }
    timing.Mark("recipe_and_rejection");
    const auto* ahead = drawHit && !patchedStages && userPatches.empty() && !drawParameters.indirect && entry != nullptr && entry->prepared != nullptr ? entry->prepared->For(matched, stages.size()) : nullptr;
    if (pipelined && (!drawParameters.indirect || pipelineIndirect) && deferredLabels().labels.empty()) {
        phaseTiming.Phase(DrawRowVectors);
        auto writes = drawWriteRanges(graphics, stages);
        const auto commitQueue = submission.queue;
        const auto commitPacket = (packet[0] >> 8u) & 0xffu;
        const auto epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
        // S1: the preparation the preparer thread may make for this commit (see DrawPipeline::Prepare).
        // It runs with copies of the draw's inputs; the stage pointers stay valid because the
        // preparation holds the commit (which owns the compiled programs).
        const bool splitBuild = Graphics::SplitDrawBuild();
        static const bool htilePresync = std::getenv("APS5_NO_HTILE_PRESYNC") == nullptr;
        const bool prepareDraw = (splitBuild || (htilePresync && decode->state.depth.has_value())) && recipe == nullptr;
        auto slot = prepareDraw ? std::make_shared<DrawPipeline::PrepareSlot>() : nullptr;
        struct PreparedCommit {
            std::shared_ptr<Graphics::PreparedDraw> draw;
            std::uint64_t presyncSerial = 0;
        };
        const auto program = programs.empty() ? std::uint64_t{0} : programs.back().binary.codeAddress;
        auto preparedStages = prepareDraw ? stages : std::vector<Graphics::CompiledShader>{};
        auto prepareDevice = prepareDraw ? localDevice : nullptr;
        auto prepareDecode = prepareDraw ? decode : nullptr;
        auto reads = QueuedDrawReadRanges(drawParameters, stages);
        auto queued = Graphics::ParallelProfile::On() ? CommitPrep::Take(localDevice.get(), reads) : CommitPrep::Stamps{};
        auto commit = [this, localDevice = std::move(localDevice), decode = std::move(decode), drawParameters, stages = std::move(stages), snapshots = std::move(snapshots), recipe = std::move(recipe), recipeStages = std::move(recipeStages), ahead, drawKey, trace = trace, commitQueue, commitPacket, epoch, programs = std::move(programs), results = std::move(results), memory = std::move(memory), linked = std::move(linked), stageCaptures = std::move(stageCaptures), matched = std::move(matched), matchedRegions = std::move(matchedRegions), fresh = std::move(fresh), decodeReads = std::move(decodeReads), shaderMemory = std::move(shaderMemory), entry = std::move(entry), dataEntry = std::move(dataEntry), compiledStages = std::move(compiledStages), stageData = std::move(stageData), slot, program, queued, splitBuild] {
            DrawPipeline::FollowEpoch(epoch);
            GuestMemory::SetCurrentPacket(commitPacket, commitQueue);
            CommitPrep::Stamps atCommit;
            if (queued.taken) {
                atCommit = CommitPrep::Retake(queued, device.Load().get());
                Graphics::ParallelProfile::NoteStamps(Graphics::ParallelProfile::Window::Queue, CommitPrep::Moved(queued, atCommit));
            }
            struct StampScope {
                explicit StampScope(const CommitPrep::Stamps* stamps) { CommitPrep::SetCommitStamps(stamps); }
                ~StampScope() { CommitPrep::SetCommitStamps(nullptr); }
            } stampScope(atCommit.taken ? &atCommit : nullptr);
            struct OriginScope {
                explicit OriginScope(std::uint64_t address) { Graphics::Recorder::SetWriteOrigin(address); }
                ~OriginScope() { Graphics::Recorder::SetWriteOrigin(0); }
            } originScope(program);
            try {
                if (slot != nullptr && splitBuild && CommitPrep::LockSplit() && slot->Begin()) {
                    auto own = std::make_shared<PreparedCommit>();
                    try {
                        own->draw = localDevice->PrepareDraw(decode->state, drawParameters, stages);
                    } catch (...) {
                        own->draw = nullptr;
                    }
                    slot->Finish(std::move(own));
                    CommitPrep::NoteStageA();
                }
                std::shared_ptr<PreparedCommit> prepared = slot != nullptr ? std::static_pointer_cast<PreparedCommit>(slot->Take()) : nullptr;
                std::uint64_t presyncSerial = prepared != nullptr ? prepared->presyncSerial : 0;
                // The pre-sync again, now that every earlier draw is recorded: still without the
                // GPU mutex, so the other queues are not held while this commit waits for the GPU.
                if (htilePresync && decode->state.depth.has_value()) presyncSerial = std::max(presyncSerial, localDevice->PresyncDrawDepth(decode->state));
                commitDraw(localDevice, commitQueue, decode->state, drawParameters, stages, snapshots, recipe, recipeStages, drawKey, prepared != nullptr ? prepared->draw.get() : nullptr, presyncSerial, ahead, trace);
            } catch (const std::runtime_error& error) {
                std::string where = "draw commit (programs";
                for (const auto& program : programs) {
                    char text[24];
                    std::snprintf(text, sizeof(text), " 0x%llx", static_cast<unsigned long long>(program.binary.codeAddress));
                    where += text;
                }
                throw std::runtime_error(where + "): " + error.what());
            }
        };
        auto held = std::make_shared<decltype(commit)>(std::move(commit));
        DrawPipeline::Prepare prepare;
        if (slot != nullptr) {
            prepare = [held, slot, device = std::move(prepareDevice), decode = std::move(prepareDecode), stages = std::move(preparedStages), drawParameters, epoch, commitPacket, commitQueue, splitBuild] {
                if (!slot->Begin()) return;
                auto prepared = std::make_shared<PreparedCommit>();
                try {
                    DrawPipeline::FollowEpoch(epoch);
                    GuestMemory::SetCurrentPacket(commitPacket, commitQueue);
                    if (htilePresync && decode->state.depth.has_value()) prepared->presyncSerial = device->PresyncDrawDepth(decode->state);
                    if (splitBuild) prepared->draw = device->PrepareDraw(decode->state, drawParameters, stages);
                } catch (...) {
                    prepared->draw = nullptr;
                }
                slot->Finish(std::move(prepared));
            };
        }
        timing.Mark("write_ranges_and_commit");
        DrawPipeline::Queue0().Enqueue([held] { (*held)(); }, std::move(writes), 0, {}, std::move(reads), std::move(prepare), true, program * 0x100000001b3ull ^ drawKey);
        if (pipelineIndirect) DrawPipeline::Queue0().Note(DrawPipeline::Event::IndirectCommit);
        phaseTiming.Phase(DrawRowLockWait);
        timing.Mark("draw_enqueued");
        return drawn();
    }
    if (drawParameters.indirect) drainBeforeInline();
    lockForDraw();
    noteDrawWriters(stages, submission.queue);
    phaseTiming.Phase(DrawRowVectors);
    if (recipe != nullptr) {
        if (localDevice->DrawFromRecipe(graphics, drawParameters, stages, snapshots, recipe, tracePtr) == RecipeOutcome::Recorded) {
            phaseTiming.Phase(DrawRowGraphics);
            timing.Mark("draw_and_resource_release");
            return drawn();
        }

        VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
    }
    std::shared_ptr<const DrawRecipe> built;
    localDevice->Draw(graphics, drawParameters, stages, snapshots, recipeStages.empty() ? nullptr : &built, nullptr, ahead, tracePtr);
    phaseTiming.Phase(DrawRowGraphics);
    if (built != nullptr) attachDrawRecipe(drawKey, recipeStages, std::move(built));
    timing.Mark("draw_and_resource_release");
    return drawn();
}

CommitPrep::PreDrain Driver::PreDrainCommit(bool submitOpen) {
    const auto localDevice = Get().device.Load();
    if (localDevice == nullptr) return {};
    const auto drained = localDevice->PreDrainUnlocked(submitOpen);
    return {drained.drained, drained.submitted, drained.waitedUs};
}

void Driver::commitDraw(std::shared_ptr<VulkanDevice> localDevice, std::uint32_t queue, const Graphics::State& graphics, const Pm4::DrawParameters& drawParameters, std::span<const Graphics::CompiledShader> stages, std::span<const Graphics::GuestMemorySnapshot> snapshots, const std::shared_ptr<const DrawRecipe>& recipe, const std::vector<std::shared_ptr<DispatchVariant>>& recipeStages, std::uint64_t drawKey, Graphics::PreparedDraw* prepared, std::uint64_t presyncSerial, const Graphics::AheadDraw* ahead, VulkanDevice::DrawTraceContext trace) {
    const auto* tracePtr = trace.token != 0 ? &trace : nullptr;
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    if (auto current = device.Load(); current != nullptr && current != localDevice) {
        localDevice = std::move(current);
        prepared = nullptr;
        presyncSerial = 0;
    }
    if (const auto* stamps = CommitPrep::CommitStamps(); stamps != nullptr) Graphics::ParallelProfile::NoteStamps(Graphics::ParallelProfile::Window::Commit, CommitPrep::Moved(*stamps, CommitPrep::Retake(*stamps, localDevice.get())));
    // The batches the HTILE pre-sync waited for: their completions land before the store-back reads.
    if (presyncSerial != 0) localDevice->ReapRecorded(presyncSerial);
    struct SubmitDue {
        Driver& driver;
        const std::shared_ptr<VulkanDevice>& device;
        ~SubmitDue() noexcept(false) {
            if (std::uncaught_exceptions() == 0) driver.submitDueAfterCommit(device.get());
        }
    } submitDue{*this, localDevice};
    recordLabelsForPacket(localDevice.get(), queue);
    noteDrawWriters(stages, queue);
    if (recipe != nullptr) {
        if (localDevice->DrawFromRecipe(graphics, drawParameters, stages, snapshots, recipe, tracePtr) == RecipeOutcome::Recorded) return;
        VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
    }
    std::shared_ptr<const DrawRecipe> built;
    localDevice->Draw(graphics, drawParameters, stages, snapshots, recipeStages.empty() ? nullptr : &built, prepared, ahead, tracePtr);
    if (built != nullptr) attachDrawRecipe(drawKey, recipeStages, std::move(built));
}

}
