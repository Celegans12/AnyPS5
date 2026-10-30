#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_ROUTER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_ROUTER_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace AgcDriver::NewPath {

enum class CommandKind : std::uint8_t { Draw, Dispatch, DispatchIndirect };
inline constexpr std::size_t CommandKindCount = 3;

enum class Route : std::uint8_t { NewPath, Fallback };

enum class FallbackReason : std::uint8_t { KindSwitch, KindNotEmitted, NoPipeline, RangeNotCovered, PushOverflow, ClothGuard, NonStaticChain, Suspended, NoResolvedModule, NotRecorded, C1NoTwin, C1Switch, C1RangesUnavailable, C1WrittenElsewhere, C1NotCovered, C1Layout, NotClassA, DrawShape, NotLean, TwinSkipped };
inline constexpr std::size_t FallbackReasonCount = 20;

[[nodiscard]] const char* CommandKindName(CommandKind kind);
[[nodiscard]] const char* FallbackReasonName(FallbackReason reason);

struct Decision {
    Route route = Route::Fallback;
    FallbackReason reason = FallbackReason::KindNotEmitted;
};

struct CommandFacts {
    CommandKind kind = CommandKind::Draw;
    bool kindSwitchOff = false;
    bool resolvedProgram = false;
    bool pipelineReady = true;
    bool clothGuard = false;
    bool tableCovers = true;
    bool pushFits = true;
    bool overflowPlaced = false;
    bool staticChains = true;
    bool suspended = false;
    // S2-share C1 (buffer V#s read at run time through the page table, no images): the program has the shape, its
    // twin is not requested while APS5_NEW_PATH_NO_C1 is set, F4's refusal names the reason, and the twin's data
    // layout fits the build's (thread-dimension limit at the same dword).
    bool c1 = false;
    bool c1Switch = false;
    FallbackReason coverReason = FallbackReason::RangeNotCovered;
    bool layoutFits = true;
    // TwinRoutes skipped this dispatch's twin build (its program's recent twinned dispatches all fell back).
    bool twinSkipped = false;
    // Draws (APS5_NEW_PATH_EMIT_DRAWS): an address-based build (class A), a shape the resolved prefix covers
    // (DrawTwins.hpp), recorded into the open batch (a lean draw).
    bool classA = true;
    bool shapeSupported = true;
    bool lean = true;
};

struct RouteCounts {
    std::array<std::uint64_t, CommandKindCount> newPath{};
    std::array<std::array<std::uint64_t, FallbackReasonCount>, CommandKindCount> fallback{};

    [[nodiscard]] std::uint64_t Commands() const;
    [[nodiscard]] std::uint64_t NewPathCommands() const;
};

class Router {
public:
    explicit Router(bool emitDispatches = false, bool emitDraws = false) : emitDispatches(emitDispatches), emitDraws(emitDraws) {}

    [[nodiscard]] Decision Decide(const CommandFacts& facts) const;
    [[nodiscard]] bool EmitsDispatches() const { return emitDispatches; }
    [[nodiscard]] bool EmitsDraws() const { return emitDraws; }
    void Note(CommandKind kind, const Decision& decision);
    [[nodiscard]] RouteCounts Take();

    [[nodiscard]] static std::string Format(const RouteCounts& counts, const char* window);

private:
    bool emitDispatches = false;
    bool emitDraws = false;
    std::array<std::atomic<std::uint64_t>, CommandKindCount> newPath{};
    std::array<std::array<std::atomic<std::uint64_t>, FallbackReasonCount>, CommandKindCount> fallback{};
};

}

#endif
