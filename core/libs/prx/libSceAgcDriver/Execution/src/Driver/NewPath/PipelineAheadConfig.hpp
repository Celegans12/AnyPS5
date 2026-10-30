#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_PIPELINEAHEADCONFIG_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_PIPELINEAHEADCONFIG_HPP

#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PipelineAhead.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PipelineAheadTwins.hpp"
#include <filesystem>

namespace AgcDriver::NewPath {

[[nodiscard]] bool PipelineGateEnabled();
[[nodiscard]] std::filesystem::path DefaultManifestPath();
PipelineAhead& Pipelines();
// The resolved twins' pipelines over Pipelines(); module files in newpath-twins/ beside the manifest.
TwinPipelines& TwinPipelineStore();

}

#endif
