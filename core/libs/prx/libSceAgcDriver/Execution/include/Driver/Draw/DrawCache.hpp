#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWCACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWCACHE_HPP

#include "UserDataUse.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include "prx/libc/include/general/AtomicSharedPtr.hpp"

namespace AgcDriver::DriverDetail {

struct DrawProgram {
    ShaderRecompiler::ShaderBinary binary;
    std::uint32_t userDataBase;
    std::uint32_t firstUserSgpr = 8;
    std::vector<std::uint32_t> userData;
    std::array<ShaderRecompiler::MemoryRegion, 2> memory;

    std::shared_ptr<const ShaderSnapshot> snapshot;
    std::size_t codeOffset = 0;
    std::shared_ptr<const ShaderSnapshot> tailContinuation;
    std::uint32_t resourceRegister = 0;
    bool nullPixel = false;
    bool merged = false;
    bool mergedPointerRequired = false;
    std::uint32_t mergedPointer = 0;
    std::uint32_t hiddenUserWords = 0;
};

struct DrawDecode {
    Graphics::State state;
    ShaderRecompiler::ShaderPixelStageInfo pixel;
    std::vector<DrawProgram> programs;
    std::vector<ShaderRecompiler::ProgramRole> roles;
};

void ReadUserWords(const QueueState& queue, DrawProgram& program, bool staticAbi);
void InitializeMerged(const QueueState& queue, DrawProgram& program, std::uint32_t pointerBase, bool pointerRequired, bool staticAbi);
void DecodeGraphicsPrograms(DrawDecode& decoded, const QueueState& queue, const ShaderRegistry& registry, bool staticAbi, bool includeFragment);
ShaderRecompiler::ShaderVertexStageInfo DecodeOrderedVertexStageInfo(std::span<const std::byte> header, std::uint64_t headerAddress, std::span<const std::uint32_t> userData, std::vector<Graphics::DecodeRead>& reads, std::uint32_t hiddenUserWords);

void AppendProgramMemory(std::vector<ShaderRecompiler::MemoryRegion>& memory, const std::vector<DrawProgram>& programs);
std::vector<std::pair<std::size_t, std::size_t>> UserDataWords(const std::vector<DrawProgram>& programs, const std::vector<ShaderRecompiler::ProgramRole>& roles, std::uint32_t location);

struct PreparedGraphicsStage {
    std::shared_ptr<const ShaderSnapshot> snapshot;
    PreparedShaders::Entry entry;
    ShaderRecompiler::ShaderStage stage = ShaderRecompiler::ShaderStage::Vertex;
    std::uint32_t pushOffset = 0;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> registration;
};

std::vector<PreparedGraphicsStage> PrepareGraphicsStages(const DrawDecode& decoded, const ShaderRecompiler::SpirvTarget& target, bool deferUnprepared = false);

struct DrawRecipeRecord {
    std::vector<std::weak_ptr<const DispatchVariant>> stages;
    std::shared_ptr<const DrawRecipe> recipe;
    bool Matches(const std::vector<std::shared_ptr<DispatchVariant>>& variants) const;
    bool Expired() const;
};

struct PreparedDrawRecord {
    std::vector<std::weak_ptr<const DispatchVariant>> stages;
    std::size_t shaders = 0;
    std::shared_ptr<const Graphics::AheadDraw> prepared;
    const Graphics::AheadDraw* For(const std::vector<std::shared_ptr<DispatchVariant>>& variants, std::size_t shaderCount) const;
};

struct DrawEntry {

    std::shared_ptr<const DrawDecode> decode;
    std::shared_ptr<const PreparedDrawRecord> prepared;

    std::vector<std::vector<std::shared_ptr<DispatchVariant>>> stages;

    AtomicSharedPtr<const std::vector<DrawRecipeRecord>> recipes;
    std::uint64_t touched = 0;
    bool probation = false;
    std::list<std::uint64_t>::iterator order;
    std::uint64_t shape = 0;
};

using DrawDataCandidates = std::vector<std::vector<std::pair<std::shared_ptr<DispatchVariant>, std::vector<ShaderRecompiler::MemoryRegion>>>>;

struct StageDataHit {
    std::vector<std::uint32_t> words;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
};

struct DrawKeyComponents {
    std::uint64_t stable = 0;
    std::array<std::uint64_t, 3> userData{};
    std::uint64_t auxiliary = 0;
    std::uint64_t device = 0;
    std::uint64_t shaderSet = 0;
    std::array<std::uint64_t, Graphics::DrawKeyRegisters.size()> ranges{};
};

enum class DrawKeyChange : std::size_t { Stable, PixelUserData, VertexUserData, HullUserData, Auxiliary, Device, NewShaderSet, Unchanged, Count };

enum class DrawMiss : std::size_t { FrontDiffering, FragmentDiffering, OtherDiffering, Layout, Gate, Stages, Count };

struct DrawEntryCounters {
    std::uint64_t lookups = 0, absent = 0, hits = 0, stageValidations = 0, stageEqual = 0, variantsCompared = 0, inserts = 0, shapeUpdates = 0, variantsInserted = 0, variantsEvicted = 0, present = 0, unstable = 0, touches = 0, verifyHits = 0, verifyMismatches = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrawMiss::Count)> misses{};
    std::array<std::uint64_t, MaxDispatchVariants> variantHitsByRank{};
    double validateUs = 0;

    std::uint64_t registerKeyLookups = 0, registerKeyHits = 0, decodeSkipped = 0, decodePartial = 0, facadeMismatches = 0, verifyDecodes = 0, verifyDecodeMismatches = 0;
    double keyUs = 0;
    std::uint64_t absentNewRegisters = 0, absentUserWords = 0, absentEvicted = 0, absentNeverInserted = 0, evictions = 0, byteEvictions = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrawKeyChange::Count)> registerKeyMissChanges{};
    std::array<std::uint64_t, Graphics::DrawKeyRegisters.size()> stableRangeChanges{};
    std::uint64_t differingSameRuns = 0, differingWords = 0, differingRunsChanged = 0;
    std::uint64_t dataHits = 0, dataStagesReused = 0, dataStagesCompiled = 0, dataVerified = 0;
    std::uint64_t stageDataHits = 0, stageDataWords = 0, stageDataVerified = 0, stageDataInserts = 0, stageDataPositions = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

class DrawKeySightings {
public:
    static constexpr std::size_t Ways = 4;
    explicit DrawKeySightings(std::size_t keys);
    bool SeenBefore(std::uint64_t key);

private:
    std::vector<std::uint64_t> keys;
    std::vector<std::uint8_t> filled;
    std::vector<std::uint8_t> next;
    std::size_t bucketMask = 0;
};

enum class DrawVerdict { Drawn, Nothing, Rejected };

struct StageCapture {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    std::vector<std::shared_ptr<const ShaderRecompiler::SrtGuestWords>> supplied;
    std::uint64_t forgetSerial = 0;
    std::uint32_t pushOffset = 0;
    ShaderRecompiler::SrtUserDataUse userDataUse;
    std::uint64_t userDataMask = ~std::uint64_t{0};
};

struct RelocatableUserWord {
    std::uint32_t reg;
    std::vector<std::pair<std::int64_t, std::uint32_t>> reads;
};

struct DrawRelocation {
    std::vector<RelocatableUserWord> words;
    std::vector<std::uint32_t> banned;
};

struct DrawUserPatch {
    std::size_t program;
    std::uint32_t index, value;
};

struct DrawRelocationPending {
    std::uint64_t key = 0, alias = 0, identity = 0;
};

struct DrawRelocationTrial {
    static constexpr std::uint32_t Window = 1024, MinimumHitShare = 16, IdleDraws = 16384, ProbationAdmissions = 2;
    std::uint32_t lookups = 0, hits = 0, idle = 0, admissions = 0;
    bool productive = false;
};

struct DrawRelocationCounters {
    std::uint64_t lookups = 0, donors = 0, refused = 0, hits = 0, evicted = 0;
};

bool DecodeProgramsCurrent(const DrawDecode& decode, const ShaderRegistry& registry);
bool SameCompilation(const ShaderRecompiler::RecompileResult& a, const ShaderRecompiler::RecompileResult& b, std::uint32_t hiddenUserWords = 0);
std::shared_ptr<DrawEntry> BuildAheadEntry(const std::vector<std::shared_ptr<DispatchVariant>>& fresh, std::shared_ptr<const DrawDecode> decode, std::uint64_t shape, std::shared_ptr<const Graphics::AheadDraw> prepared, std::size_t preparedShaders);
ShaderRecompiler::RecompileRequest DrawStageRequest(const DrawProgram& program, const Graphics::State& graphics, const ShaderRecompiler::ShaderPixelStageInfo& pixel, const std::optional<ShaderRecompiler::ShaderVertexStageInfo>& vertexInfo, std::span<const ShaderRecompiler::MemoryRegion> memory, std::span<const ShaderRecompiler::LinkedProgram> linked, const Pm4::DrawParameters& drawParameters, const ShaderRecompiler::SpirvTarget& target, std::uint32_t pushOffset, const std::optional<ShaderRecompiler::MeshBarycentricLayout>& meshBarycentrics = std::nullopt);
std::uint64_t StageUserDataMask(const ShaderRecompiler::ResourceCapture& capture, const ShaderRecompiler::RecompileResult& compiled, const std::optional<ShaderRecompiler::ShaderVertexStageInfo>& vertexInfo);
std::shared_ptr<DispatchVariant> DrawStageVariant(const DrawProgram& program, const StageCapture& capture, const std::optional<ShaderRecompiler::ShaderVertexStageInfo>& vertexInfo, std::span<const Graphics::DecodeRead> decodeReads);

}

#endif
