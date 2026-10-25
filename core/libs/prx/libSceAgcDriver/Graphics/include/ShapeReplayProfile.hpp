#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SHAPEREPLAYPROFILE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SHAPEREPLAYPROFILE_HPP

#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ShaderRecompiler {
struct ResourceCapture;
}

namespace AgcDriver::Graphics::ShapeReplay {

bool EnabledFromEnvironment();

inline bool Enabled() {
    static const bool enabled = EnabledFromEnvironment();
    return enabled;
}

enum class SlotClass : std::uint8_t { Pure, Cone, Base, Structural };

enum Reach : std::uint8_t {
    ReachImage = 1u << 0u,
    ReachSampler = 1u << 1u,
    ReachBufferBase = 1u << 2u,
    ReachBufferShape = 1u << 3u,
    ReachCondition = 1u << 4u,
    ReachUniform = 1u << 5u,
    ReachCone = 1u << 6u,
    ReachOtherSource = 1u << 7u,
};

struct SlotClasses {
    bool eligible = false;
    std::vector<SlotClass> slots;
    std::vector<std::uint8_t> reach;
    std::uint64_t valueUserWords = 0;
    std::uint64_t coneUserWords = 0;
    std::array<std::uint32_t, 4> counts{};
};

SlotClasses ComputeSlotClasses(const ShaderRecompiler::IrResourcePlan& plan);

enum class Diff : std::uint8_t {
    Texture,
    Sampler,
    BufferShape,
    BaseAlignment,
    Condition,
    UniformFill,
    UserValue,
    Vertex,
    Other,
    DrawParameters,
    NoPlan,
    Unrecorded,
    NewShape,
    Count
};
inline constexpr std::size_t HashedClasses = static_cast<std::size_t>(Diff::DrawParameters) + 1u;
const char* DiffName(Diff diff);

struct StageRecord {
    bool eligible = false;
    std::array<std::uint64_t, HashedClasses> hashes{};
    std::array<std::uint32_t, 4> slotCounts{};
    std::uint64_t descriptorPlan = 0;
    std::uint64_t bufferPlan = 0;
    bool buffersPresent = false;
    [[nodiscard]] std::uint64_t Combined() const;
};

StageRecord DescribeStage(const SlotClasses& classes, const ShaderRecompiler::IrResourcePlan& plan, const ShaderRecompiler::ResourceSnapshot& snapshot, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderVertexStageInfo* vertex);
std::shared_ptr<const StageRecord> DescribeCapture(const ShaderRecompiler::ResourceCapture* capture, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderVertexStageInfo* vertex);
std::uint64_t DrawParameterBits(bool indexed, std::uint32_t indexSize, std::uint32_t instanceCount, bool indirect);

enum class Age : std::uint8_t { SameFrame, PreviousFrame, Older, Count };

struct Outcome {
    bool hit = false;
    Age age = Age::SameFrame;
    Diff diff = Diff::Count;
    bool descriptorData = false;
};

struct Tally {
    std::uint64_t lookups = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(Age::Count)> hits{};
    std::array<std::uint64_t, static_cast<std::size_t>(Diff::Count)> misses{};
    std::array<std::uint64_t, 4> slots{};
    std::array<std::uint64_t, 2> descriptorData{};
    void Add(const Outcome& outcome);
    [[nodiscard]] std::uint64_t Hits() const;
};

class Predictor {
public:
    static constexpr std::size_t RingEntries = 64;
    static constexpr std::size_t MaxStages = 6;
    static constexpr std::size_t MaxRings = 1u << 17u;

    Outcome NoteStage(std::uint64_t shape, std::size_t stage, const StageRecord* record, std::uint64_t frame);
    Outcome NoteDraw(std::uint64_t shape, std::span<const StageRecord* const> stages, std::uint64_t drawParameters, std::uint64_t frame);
    void Clear();
    [[nodiscard]] std::size_t Rings() const { return rings.size(); }
    [[nodiscard]] std::uint64_t Resets() const { return resets; }

private:
    struct Ring {
        std::array<std::uint64_t, RingEntries> hashes{};
        std::array<std::uint64_t, RingEntries> frames{};
        std::array<std::uint64_t, HashedClasses> last{};
        std::uint64_t lastPlan = 0;
        std::uint32_t size = 0;
        std::uint32_t next = 0;
    };
    Outcome note(std::uint64_t key, const std::array<std::uint64_t, HashedClasses>& hashes, std::uint64_t combined, std::uint64_t descriptorPlan, std::uint64_t frame);
    std::unordered_map<std::uint64_t, Ring> rings;
    std::uint64_t resets = 0;
};

void NoteDraw(std::uint64_t shape, std::span<const StageRecord* const> stages, std::uint64_t drawParameters, std::uint64_t frame, bool exactHit);
void NoteUnkeyedDraw();
void NoteFrame(std::uint64_t frame);
std::uint64_t Frame();

void NoteResourceLookup(std::span<const std::uint32_t> key, bool hit);

struct DispatchNote {
    std::uint64_t shape = 0;
    const StageRecord* record = nullptr;
    bool exactHit = false;
    bool queue0 = false;
    std::uint64_t variant = 0;
    std::array<std::uint64_t, 3> templates{};
};
void NoteDispatch(const DispatchNote& note);
std::string DispatchReportLine(bool reset);

enum class RecordPart : std::uint8_t { Prelude, BindingsSnapshot, BindingsSet, BindingsUpdate, PassAccess, StartPass, Emit, LeavePass, Count };
void RecordStart();
void RecordLap(RecordPart part);
void RecordEnd();

std::string ReportLines(bool reset);

}

#endif
