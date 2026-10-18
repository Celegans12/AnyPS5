#ifndef PRX_LIBSCEAGCDRIVER_EXECUTION_DRIVER_DISPATCH_SLOWMATERIALIZATIONS_HPP
#define PRX_LIBSCEAGCDRIVER_EXECUTION_DRIVER_DISPATCH_SLOWMATERIALIZATIONS_HPP

#include "Recompiler.hpp"

#include <cstdint>
#include <memory>

namespace AgcDriver::DriverDetail {

// APS5_MATERIALIZE_STATS=1: a 10 s report of slow dispatch captures + materializations (see the .cpp).
bool SlowMaterializationStats();
void NoteSlowMaterialization(std::uint32_t queue, std::uint64_t address, const ShaderRecompiler::RecompileResult& result, const std::shared_ptr<const ShaderRecompiler::ResourceCapture>& capture, double captureMs, double materializeMs, bool ahead);

}

#endif
