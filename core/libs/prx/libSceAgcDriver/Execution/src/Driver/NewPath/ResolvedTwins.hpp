#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_RESOLVEDTWINS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_RESOLVEDTWINS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <mutex>
#include <unordered_map>
#include <utility>
#include "BdaAbi.hpp"

namespace ShaderRecompiler {
struct RecompileResult;
}

namespace AgcDriver::NewPath {

enum class TwinOutcome : std::uint8_t { Built, NotAddressBased, Failed, Mismatch, Reused, Skipped };
inline constexpr std::size_t TwinOutcomeCount = 6;

[[nodiscard]] bool UsesAddressTables(const ShaderRecompiler::RecompileResult& compiled);
// S2-share C1: a program that binds guest buffers and no other guest resource (no BDA table, images, samplers,
// bindless tables, GDS or texel buffers; a fault buffer is allowed: the twin's faults go to the command record). Its twin is requested with SpirvTarget::newPathRuntimeBuffers.
[[nodiscard]] bool C1Shape(const ShaderRecompiler::RecompileResult& compiled);
// A resolved twin that reads the buffer V#s at run time: today's build binds guest buffers, the twin binds none.
// It pushes its own words (no per-buffer memory offsets), so TwinDifference compares its layout instead.
[[nodiscard]] bool RuntimeBufferTwin(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin);
[[nodiscard]] bool WalkReplacesTables(const ShaderRecompiler::RecompileResult& twin);
[[nodiscard]] std::string TwinDifference(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin);
[[nodiscard]] bool TwinCompatible(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin);

// C1 F4: whether every range of `sites` (a twin's bounded read or write sites) lies inside `build` (the build's
// guest-buffer ranges, any order), inside ranges with Write where `written`; a site may end up to SiteTailBytes past
// the build range it starts in (the site bound rounds up to an element; the access is out of the V#'s records).
inline constexpr std::uint64_t SiteTailBytes = 16;
[[nodiscard]] bool SitesInside(std::span<const std::pair<std::uint64_t, std::uint64_t>> sites, std::span<const ShaderRecompiler::BdaAbi::Range> build, bool written);

// A program and its resource shape (code address, bindings, push bytes): the key of TwinRoutes.
[[nodiscard]] std::uint64_t TwinProgramKey(std::uint64_t codeAddress, const ShaderRecompiler::RecompileResult& compiled);

// Twin builds ahead of the route (D-combo). A dispatch-cache miss builds today's result, then, before the router sees
// the dispatch, its resolved twin (capture and materialize: about 14 us per dispatch in ab-c1b), once per result, and
// the result is new on almost every dispatch. TwinRoutes keeps, per program key, the routes of the last History
// dispatches that had a twin. A program whose recent twinned dispatches all fell back for a reason of their own
// (cover, layout, push words, cloth guard, a twin that failed or differed; not a pipeline still being made) gets no
// twin, except on every ProbeEvery-th claim. Skipping a twin only sends the dispatch to today's path.
// APS5_NEW_PATH_TWIN_ALWAYS=1 turns the prediction off (every eligible miss builds its twin, as before).
class TwinRoutes {
public:
    static constexpr std::uint32_t History = 16;
    static constexpr std::uint32_t MinObserved = 4;
    static constexpr std::uint32_t ProbeEvery = 32;
    struct Totals {
        std::uint64_t programs = 0, built = 0, skipped = 0, probes = 0, notedNew = 0, notedFallback = 0;
    };
    explicit TwinRoutes(bool always = false) : always(always) {}
    // Whether a dispatch of `program` builds its twin now (counted as built, probe or skipped).
    [[nodiscard]] bool ShouldBuild(std::uint64_t program);
    // The route of a dispatch that had a twin (routedNew false also for a twin that failed or differed).
    void Note(std::uint64_t program, bool routedNew);
    [[nodiscard]] Totals Take() const;

private:
    struct Program {
        std::uint16_t history = 0;
        std::uint8_t observed = 0;
        std::uint32_t skippedSince = 0;
    };
    bool always = false;
    mutable std::mutex mutex;
    std::unordered_map<std::uint64_t, Program> programs;
    Totals totals;
};

TwinRoutes& Routes();

class ResolvedTwins {
public:
    ResolvedTwins();
    void Remember(const std::shared_ptr<const ShaderRecompiler::RecompileResult>& compiled, std::shared_ptr<const ShaderRecompiler::RecompileResult> twin, TwinOutcome outcome, bool c1 = false, std::uint64_t program = 0);
    // The twin of a remembered result; `program` gets its TwinProgramKey and `skipped` whether TwinRoutes skipped it.
    [[nodiscard]] std::shared_ptr<const ShaderRecompiler::RecompileResult> Of(const ShaderRecompiler::RecompileResult& compiled, std::uint64_t* program = nullptr, bool* skipped = nullptr) const;
    [[nodiscard]] bool Known(const ShaderRecompiler::RecompileResult& compiled) const;
    [[nodiscard]] std::array<std::uint64_t, TwinOutcomeCount> Outcomes() const;
    // The outcomes of C1 twin requests (Built, -, Failed, Mismatch, -; also counted in Outcomes) and how many of the
    // built ones bind no guest buffers (RuntimeBufferTwin).
    [[nodiscard]] std::array<std::uint64_t, TwinOutcomeCount> C1Outcomes() const;
    [[nodiscard]] std::uint64_t C1RuntimeBuilt() const;
    [[nodiscard]] std::size_t Size() const;

private:
    struct State;
    std::shared_ptr<State> state;
};

ResolvedTwins& Twins();

}

#endif
