#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SHADERS_SHADERMEMORYCENSUS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SHADERS_SHADERMEMORYCENSUS_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"

namespace AgcDriver::DriverDetail {

// DEBUG: current registered snapshots only, sampled without waiting for preparation or hydrating
// SPIR-V. Artifact references are deduplicated, but distinct artifacts may share backing storage.
// Capacities describe owned vector backing, not OS residency or total process memory. Source plans,
// old registries, rectangle modules and materialized/specialized caches are not measured here.
struct RegisteredShaderMemory {
    std::size_t registryEntries = 0, snapshots = 0, pending = 0, failed = 0;
    std::size_t entryReferences = 0, handles = 0, sources = 0, artifacts = 0;
    std::size_t missingArtifacts = 0, excludedRectangles = 0;
    std::size_t rawArtifacts = 0, coldArtifacts = 0, exposedArtifacts = 0;
    std::size_t logicalBytes = 0, rawBytes = 0, rawCapacityBytes = 0;
    std::size_t coldBytes = 0, coldCapacityBytes = 0;
    std::size_t fileArtifacts = 0, fileBytes = 0;
    std::size_t codeBytes = 0, codeCapacityBytes = 0, headerBytes = 0, headerCapacityBytes = 0;

    bool operator==(const RegisteredShaderMemory&) const = default;
};

// DEBUG: the caller pins an immutable registry; individual preparations are sampled under lock.
[[nodiscard]] RegisteredShaderMemory InspectRegisteredShaderMemory(const ShaderRegistry& registry);
[[nodiscard]] bool RegisteredShaderMemoryAfterWorkDue() noexcept;
void ReportRegisteredShaderMemory(const ShaderRegistry& registry, const char* phase) noexcept;

}

#endif
