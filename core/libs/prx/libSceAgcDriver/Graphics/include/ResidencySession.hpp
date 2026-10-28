#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYSESSION_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYSESSION_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <string>

namespace AgcDriver::Graphics {

bool ResidencyEnsured(VkDevice device);
void EnsureResidency(const Context& context);
void ReleaseResidency(VkDevice device);
std::string ResidencyReportLine();

}

#endif
