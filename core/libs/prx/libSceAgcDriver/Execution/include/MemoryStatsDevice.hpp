#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_MEMORYSTATSDEVICE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_MEMORYSTATSDEVICE_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

namespace AgcDriver::MemoryStats {

// DEBUG (APS5_MEM_STATS): wraps the device's vkGetDeviceProcAddr so memory, shader module, pipeline,
// image and buffer creation is counted; returns `real` unchanged when the stats are off.
[[nodiscard]] PFN_vkGetDeviceProcAddr TrackDevice(VkDevice device, VkPhysicalDevice physical, const VkPhysicalDeviceMemoryProperties& memory,
                                                  PFN_vkGetDeviceProcAddr real, PFN_vkGetPhysicalDeviceMemoryProperties2 budget);
// DEBUG (APS5_MEM_STATS + APS5_FAULT_ADDRESSES): the live and recently destroyed addressable buffers
// overlapping [address, address + precision), for a device fault report.
[[nodiscard]] std::vector<std::string> DescribeDeviceAddress(std::uint64_t address, std::uint64_t precision);

}

#endif
