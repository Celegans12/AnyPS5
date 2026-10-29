#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYSESSION_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYSESSION_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <memory>
#include <cstdint>
#include <span>
#include <string>
#include <utility>

namespace AgcDriver::Graphics {

struct HostImport;

bool ResidencyEnsured(VkDevice device);
void EnsureResidency(const Context& context);
void ReleaseResidency(VkDevice device);
std::string ResidencyReportLine();
bool ResidencyActive();
std::shared_ptr<HostImport> BindStagedSlot(std::uint64_t address, std::uint64_t bytes, bool written);
std::shared_ptr<HostImport> BindWrittenSlot(std::uint64_t address, std::uint64_t bytes);
std::uint64_t ResidencyBindingSerial();
void NoteStagedOverlays(std::span<const std::pair<std::uint64_t, std::uint64_t>> overlays);
void NoteStagedRegions(std::span<const std::pair<std::uint64_t, std::uint64_t>> reads, std::span<const std::pair<std::uint64_t, std::uint64_t>> writes);

}

#endif
