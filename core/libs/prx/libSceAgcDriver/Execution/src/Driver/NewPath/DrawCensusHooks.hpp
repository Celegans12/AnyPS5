#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_DRAWCENSUSHOOKS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_DRAWCENSUSHOOKS_HPP

#include <cstddef>
#include <span>

namespace AgcDriver::Graphics {
struct CompiledShader;
class ShaderResources;
}

namespace AgcDriver::Pm4 {
struct DrawParameters;
}

// The driver side of the draw census (DrawCensus.hpp), called from Graphics::Draw on the thread
// that runs the draw. All are no-ops unless DrawCensus::Enabled(); End without a Begin on the same
// thread (a recipe draw) adds nothing.
namespace AgcDriver::NewPath::DrawCensus {

void Begin(std::span<const Graphics::CompiledShader> shaders, const Pm4::DrawParameters& draw);
void NoteBuild(const Graphics::ShaderResources& resources);
void End(std::size_t kind, std::span<const double> phaseUs);
void Close(const char* window);

}

#endif
