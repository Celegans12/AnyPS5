#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYSESSION_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYSESSION_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstdint>
#include <span>
#include <string>
#include <utility>

namespace AgcDriver::Graphics {

bool ResidencyEnsured(VkDevice device);
void EnsureResidency(const Context& context);
void ReleaseResidency(VkDevice device);
std::string ResidencyReportLine();
bool ResidencyActive();
void NoteStagedRegions(std::span<const std::pair<std::uint64_t, std::uint64_t>> reads, std::span<const std::pair<std::uint64_t, std::uint64_t>> writes);

}

#endif
