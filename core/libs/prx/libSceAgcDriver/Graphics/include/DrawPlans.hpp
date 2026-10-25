#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWPLANS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWPLANS_HPP

#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace AgcDriver::Graphics {

class Pipeline;
class ShaderResources;
struct StorageSurfacePlan;

namespace DrawPlans {

bool Active();
bool TakeVerify();
std::uint64_t& CurrentKey();

struct KeyScope {
    explicit KeyScope(std::uint64_t key) : previous(CurrentKey()) { CurrentKey() = key; }
    ~KeyScope() { CurrentKey() = previous; }
    KeyScope(const KeyScope&) = delete;
    KeyScope& operator=(const KeyScope&) = delete;
    std::uint64_t previous;
};

struct PipelinePlan {
    std::weak_ptr<ShaderResources> resources;
    std::weak_ptr<Pipeline> pipeline;
    std::uint64_t hash = 0;
    bool lean = false;
    std::uint32_t fragmentOutputs = 0;
    VertexInputLayout vertexInput;
};

std::shared_ptr<const PipelinePlan> FindPipeline(std::uint64_t key);
void StorePipeline(std::uint64_t key, std::shared_ptr<const PipelinePlan> plan);
void ForgetPipeline(std::uint64_t key);
bool SameVertexInput(const VertexInputLayout& a, const VertexInputLayout& b);

bool PlannedTargets();
std::uint64_t TargetKey(std::uint64_t planKey, std::size_t index);
std::shared_ptr<const StorageSurfacePlan> FindTarget(std::uint64_t key);
void StoreTarget(std::uint64_t key, std::shared_ptr<const StorageSurfacePlan> plan);
void ForgetTarget(std::uint64_t key);

enum class Outcome : std::uint8_t { Hit, NoPlan, Resources, Layout, PipelineGone, StoreMoved, Stored, Unstorable, VerifyMismatch, TargetHit, TargetNoPlan, TargetFallback, TargetVerifyMismatch, Count };
void Count(Outcome outcome);

struct Totals {
    std::array<std::uint64_t, static_cast<std::size_t>(Outcome::Count)> outcomes{};
    std::uint64_t verified = 0;
};
Totals Counters();
void NoteVerified();
std::string ReportLine();

}

}

#endif
