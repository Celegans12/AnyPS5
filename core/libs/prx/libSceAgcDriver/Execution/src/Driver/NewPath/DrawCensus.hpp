#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_DRAWCENSUS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_DRAWCENSUS_HPP

#include "BdaAbi.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <utility>

namespace ShaderRecompiler {
struct RecompileResult;
enum class ShaderStage;
}

// The draw census of the new path (lane S2-draws; APS5_NEW_PATH=1 + APS5_PROFILE_NEW_PATH=1 +
// APS5_PROFILE_DRAW=1): one [newpath-draws] line per profile window that joins what the old path's
// [draws] line prints apart — the committer's time per draw (Graphics::Draw's phases) by how the
// draw's resources came about (template hit, build, address-based build) and by the draw's shape
// (the binding roles of its stages, its stages, push bytes, user words, indexed/indirect/instanced) —
// and the facts a GPU-resolved draw would be decided on: whether an address-based build's ranges
// are covered by the device page table at the same device address (F4, as for the dispatch twins),
// whether the index buffer and the indirect arguments are mapped by it, and whether today's push
// block leaves room for the resolved prefix (F5). It observes only: nothing it computes changes a
// decision, a binding or a byte the GPU sees.
namespace AgcDriver::NewPath::DrawCensus {

enum ShapeBit : std::uint32_t {
    Bda = 1u << 0,
    Buffers = 1u << 1,
    BuffersWritten = 1u << 2,
    Sampled = 1u << 3,
    StorageImages = 1u << 4,
    Samplers = 1u << 5,
    Bindless = 1u << 6,
    FlattenedSrt = 1u << 7,
    ShaderData = 1u << 8,
    TexelBuffers = 1u << 9,
    Gds = 1u << 10,
    RangesFromMemory = 1u << 11,
    Unbounded = 1u << 12,
    BdaWrites = 1u << 13,
};
inline constexpr std::size_t ShapeBitCount = 14;

enum StageBit : std::uint32_t { StageVertex = 1u << 0, StageLocal = 1u << 1, StageHull = 1u << 2, StageDomain = 1u << 3, StageGeometry = 1u << 4, StagePixel = 1u << 5, StageMesh = 1u << 6, StageCompute = 1u << 7 };
inline constexpr std::size_t StageBitCount = 8;

// Graphics::Draw's phases in Draw.cpp's order (DrawPhase); Draw.cpp asserts the count.
inline constexpr std::size_t PhaseCount = 13;
// Draw.cpp's DrawKind order.
inline constexpr std::size_t KindCount = 4;

// Bytes the resolved prefix takes at the front of the push block (table root, command record:
// NewPath::ResolvedModule words 0-5), so today's words fit behind it only up to 256 - 24.
inline constexpr std::uint32_t PushBlockBytes = 256;
inline constexpr std::uint32_t ResolvedPrefixBytes = 24;

enum class Cover : std::uint8_t { None, Covered, Missing, OtherAddress, Narrower, NoTable };
inline constexpr std::size_t CoverCount = 6;
[[nodiscard]] const char* CoverName(Cover cover);

struct StageFacts {
    std::uint32_t shape = 0;
    std::uint32_t userWords = 0;
    std::uint32_t pushEnd = 0;
    std::uint32_t vertexInputs = 0;
};
[[nodiscard]] StageFacts Facts(const ShaderRecompiler::RecompileResult& program, std::uint32_t pushOffset);
[[nodiscard]] std::uint32_t StageBitOf(ShaderRecompiler::ShaderStage stage);

struct DrawFacts {
    std::uint32_t shape = 0;
    std::uint32_t stages = 0;
    std::uint32_t userWords = 0;
    std::uint32_t pushEnd = 0;
    std::uint32_t vertexInputs = 0;
    bool indexed = false;
    bool indirect = false;
    bool instanced = false;
    Cover bda = Cover::None;
    Cover index = Cover::None;
    Cover arguments = Cover::None;
};
void AddStage(DrawFacts& draw, ShaderRecompiler::ShaderStage stage, const StageFacts& facts);

[[nodiscard]] std::string ShapeName(std::uint32_t shape);
[[nodiscard]] std::string StagesName(std::uint32_t stages);

// Whether [begin, end) is mapped by the (sorted, disjoint) table, at any device address.
[[nodiscard]] bool Mapped(std::span<const ShaderRecompiler::BdaAbi::Range> table, std::uint64_t begin, std::uint64_t end);

// The totals of one window; Format prints the line.
class Window {
public:
    void Add(std::size_t kind, const DrawFacts& draw, std::span<const double> phaseUs);
    [[nodiscard]] std::string Format(const char* window, std::size_t topShapes = 12) const;
    [[nodiscard]] std::uint64_t Draws() const { return draws; }
    [[nodiscard]] std::uint64_t KindDraws(std::size_t kind) const { return kind < KindCount ? kindDraws[kind] : 0; }
    [[nodiscard]] double KindPhaseUs(std::size_t kind, std::size_t phase) const { return kind < KindCount && phase < PhaseCount ? kindPhaseUs[kind][phase] : 0.0; }
    [[nodiscard]] std::uint64_t CoverCountOf(std::size_t which, Cover cover) const { return which < 3 ? covers[which][static_cast<std::size_t>(cover)] : 0; }
    [[nodiscard]] std::uint64_t PushFits() const { return pushFits; }

private:
    struct ShapeStats {
        std::uint64_t draws = 0;
        double us = 0;
        std::array<std::uint64_t, KindCount> kinds{};
        std::uint64_t pushFits = 0;
        std::uint64_t bdaCovered = 0;
    };
    std::uint64_t draws = 0, indexed = 0, indirect = 0, instanced = 0, pushFits = 0;
    std::array<std::uint64_t, KindCount> kindDraws{};
    std::array<std::array<double, PhaseCount>, KindCount> kindPhaseUs{};
    std::array<std::uint64_t, 6> pushBuckets{};
    std::array<std::uint64_t, 6> wordBuckets{};
    std::array<std::array<std::uint64_t, CoverCount>, 3> covers{};
    std::map<std::pair<std::uint32_t, std::uint32_t>, ShapeStats> shapes;
};

// Driver hooks (Graphics/src/Draw.cpp; no-ops unless enabled). Begin stashes the facts of the draw
// the calling thread starts, NoteBuild adds those of its resolved resources, End adds the draw to
// the window with its kind and phases (and prints the line every 10 s in the game).
[[nodiscard]] bool Enabled();

}

#endif
