#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_DRIVER_DRAW_DRAWREGISTERKEY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_DRIVER_DRAW_DRAWREGISTERKEY_HPP

#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include <array>

namespace AgcDriver::DriverDetail {

struct RegisterStateKeyParts {
    std::uint64_t stableRegisters = 0;
    std::uint64_t userDataRegisters = 0;
    std::uint64_t auxiliaryRegisters = 0;
    std::array<std::uint64_t, 3> userDataByStage{};
};

std::uint64_t RegisterStateKey(const QueueState& queue, bool allUserWords = false, bool incremental = true, std::uint64_t* shapeKey = nullptr, RegisterStateKeyParts* parts = nullptr);
std::uint64_t DrawShaderSetKey(const QueueState& queue, const ShaderRegistry& registry);

// One pass over the five program addresses of a draw: the identity key, the shader-set key and the register-key tail
// all read the same low/high/RSRC words and the same registry entry per stage, so a draw looks them up once.
struct DrawStageWalk {
    bool present = false;
    std::uint32_t low = 0, high = 0, resources = 0;
    const ShaderRegistry::mapped_type* snapshots = nullptr;
    std::uint64_t offset = 0;
};
using DrawRegistryWalk = std::array<DrawStageWalk, 6>;

void WalkDrawRegistry(const QueueState& queue, const ShaderRegistry& registry, DrawRegistryWalk& walk);
std::uint64_t DrawShaderSetKey(const DrawRegistryWalk& walk);

}

#endif
