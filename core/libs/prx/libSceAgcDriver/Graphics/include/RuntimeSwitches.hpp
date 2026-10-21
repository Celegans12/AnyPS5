#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RUNTIMESWITCHES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RUNTIMESWITCHES_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace AgcDriver::Graphics {

// Driver switches that may change while the game runs (the in-run A/B, APS5_AB). Each is an APS5_*
// (mostly APS5_NO_*; APS5_QUIET_PUBLISH and APS5_LANE_SERIALIZE are opt-in)
// variable whose presence selects one of two exact paths; it starts as set when the variable is set.
// Every reader takes it once per operation (a build, a submission, a draw), so an operation never
// sees both settings.
enum class RuntimeSwitch : std::size_t {
    NoSplitDrawBuild,
    NoBdaStaging,
    NoDispatchOverlap,
    NoHtileGpuExpansion,
    NoGpuIndirectArgs,
    NoBarrierBatch,
    NoEarlyRevalidate,
    NoBatchedStaging,
    QuietPublish,
    NoStagedOverlap,
    NoOwedRunClose,
    NoRunAtomicGuard,
    DrawInputsInPlace,
    NoRunStagedReadGuard,
    NoSmallDeviceTier,
    NoExactSizeReuse,
    NoHeldReadSkip,
    NoLaneOpenSerial,
    NoLaneTemplateEdge,
    LaneSerialize,
    NoLaneOverlap,
    Count,
};

namespace RuntimeSwitches {

const char* Name(RuntimeSwitch id);
bool Set(RuntimeSwitch id);
void Assign(RuntimeSwitch id, bool set);
std::optional<RuntimeSwitch> Find(std::string_view name);
// A word for the keys of objects that are built under the current settings and reused later (the
// compute resource cache): 0 while no A/B runs, else one value per phase kind (A or B).
std::uint32_t KeySalt();

}

// The phase plan of APS5_AB=<seconds>:<switch>[,<switch>...]: from APS5_AB_START (a present index, or
// <n>s seconds after the first present seen) phases of <seconds> alternate, A with the switches as
// the environment sets them and B with the listed switches toggled. A phase ends at the first
// present at least <seconds> after its start; APS5_AB_SETTLE (default 2) seconds into a phase it is
// marked settled, and statistics before that point mix the two settings.
class AbSchedule {
public:
    struct Config {
        double phaseSeconds = 0;
        double settleSeconds = 2;
        std::uint64_t startPresent = 0;
        std::optional<double> startSeconds;
        std::vector<RuntimeSwitch> switches;
    };

    enum class Event { None, Begin, Settled };

    struct Step {
        Event event = Event::None;
        std::uint64_t phase = 0;
        bool b = false;
    };

    static Config Parse(std::string_view ab, std::string_view start, std::string_view settle);

    explicit AbSchedule(Config config);

    // `present` counts presents from 1, `seconds` is a monotonic time; at most one event per call.
    Step AtPresent(std::uint64_t present, double seconds);

    const Config& Settings() const { return config; }

private:
    Config config;
    bool started = false;
    bool settled = false;
    std::uint64_t phase = 0;
    double phaseStart = 0;
    std::optional<double> firstSeconds;
};

// The A/B at a present of queue 0 (between frames, the draw pipeline drained): applies a phase
// change to the switches and returns the [ab] lines to print. Throws std::invalid_argument for an
// APS5_AB that names a switch which cannot change while the game runs. Does nothing without APS5_AB.
bool AbConfigured();
std::vector<std::string> AbAtPresent(std::uint64_t present, double seconds);

}

#endif
