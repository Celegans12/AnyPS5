#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_PROGRAMSTATS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_PROGRAMSTATS_HPP

#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ShaderRecompiler { struct RecompileResult; }

namespace AgcDriver::DriverDetail {

// Bisect aids for rendering bugs, all off unless their variable is set:
// - APS5_DRAW_PROGRAM_STATS=<n>: every n-th present, the draws since the previous present grouped by the
//   attachments they render into (pixel program / vertex program and draw count per group), and the compute
//   programs dispatched with the storage images they write (the ones a draw of the frame sampled are marked);
// Lists take addresses and inclusive address ranges.
// - APS5_SKIP_PIXEL_PROGRAMS=<hex[-hex],...>: draws whose pixel program is listed are not drawn;
// - APS5_TINT_PIXEL_PROGRAMS=<hex[-hex],...>: draws whose pixel program is listed write a constant magenta through the
//   blend state (constant colour, zero destination) to every blendable attachment the program writes;
// - APS5_SKIP_COMPUTE_PROGRAMS=<hex[-hex],...>: dispatches of a listed compute program are not dispatched.

using ProgramList = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

ProgramList ParseProgramList(const char* text);
std::uint32_t ParseStatsInterval(const char* text);
bool ListedProgram(const ProgramList& list, std::uint64_t address);

const ProgramList& SkippedPixelPrograms();
const ProgramList& TintedPixelPrograms();
const ProgramList& SkippedComputePrograms();
// APS5_PROGRAM_TARGETS=<hex[-hex],...>: when set, the pixel skip and tint lists only apply to draws with a colour
// target at a listed address.
const ProgramList& ProgramFilterTargets();
// APS5_SKIP_PIXEL_BITS=<frame>:<bits>: a bisect over frame dumps (one every 10 presents). Frame dump <frame> + 1 + b
// (b < bits) skips the pixel programs whose ProgramBisectHash has bit b set (with APS5_PROGRAM_TARGETS, only draws
// into those targets); frame dump <frame> and the ones after the last bit skip nothing. A frame dump's setting holds
// from 5 presents before its present to 5 after, so a present of lag does not mix settings.
std::uint32_t ProgramBisectHash(std::uint64_t address);
std::optional<std::uint32_t> BisectBitForPresent(std::uint64_t present, std::uint32_t frame, std::uint32_t bits);
bool BisectSkipsProgram(std::uint64_t address);
void CountPresent();

bool BlendableAttachmentFormat(VkFormat format);
// Each blend state whose attachment format (formats[i], VK_FORMAT_UNDEFINED for none) is blendable and whose
// write mask is non-zero becomes constant * 1 + destination * 0 with the constant magenta (1, 0, 1, 1).
// Returns the number of attachments tinted.
std::size_t ApplyBlendTint(std::span<VkPipelineColorBlendAttachmentState> blends, std::array<float, 4>& constants, std::span<const VkFormat> formats);

struct StatsTarget {
    std::uint64_t address = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t format = 0;
    std::uint32_t samples = 1;
    auto operator<=>(const StatsTarget&) const = default;
};

struct StatsImage {
    std::uint64_t address = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t format = 0;
    auto operator<=>(const StatsImage&) const = default;
};

std::vector<StatsImage> ProgramStatsImages(const ShaderRecompiler::RecompileResult& program, bool written);

class ProgramStats {
public:
    explicit ProgramStats(std::uint32_t every) : every(every) {}
    bool Enabled() const { return every != 0; }
    void NoteDraw(std::uint64_t pixel, std::uint64_t vertex, std::span<const StatsTarget> colors, std::optional<StatsTarget> depth, bool skipped, bool tinted);
    void NoteDrawSampled(std::uint64_t pixel, std::span<const std::uint64_t> addresses);
    void NoteDispatch(std::uint64_t program, std::span<const StatsImage> written, bool skipped);
    // Counts a present; on every n-th one returns the report lines of the draws and dispatches since the previous
    // present (otherwise none) and starts a new frame.
    std::vector<std::string> Present();

    static ProgramStats& Global();

private:
    struct Group {
        std::vector<StatsTarget> colors;
        std::optional<StatsTarget> depth;
        auto operator<=>(const Group&) const = default;
    };
    struct PixelCount {
        std::uint64_t draws = 0;
        std::uint64_t skipped = 0;
        std::uint64_t tinted = 0;
    };
    struct ComputeCount {
        std::uint64_t dispatches = 0;
        std::uint64_t skipped = 0;
        std::set<StatsImage> written;
    };
    std::uint32_t every;
    std::mutex mutex;
    std::uint64_t presents = 0;
    std::uint64_t draws = 0;
    std::map<Group, std::map<std::pair<std::uint64_t, std::uint64_t>, PixelCount>> groups;
    std::map<std::uint64_t, ComputeCount> compute;
    std::map<std::uint64_t, std::set<std::uint64_t>> sampled;
};

}

#endif
