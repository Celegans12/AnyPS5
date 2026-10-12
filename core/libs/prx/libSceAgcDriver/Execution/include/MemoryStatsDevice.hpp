#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_MEMORYSTATSDEVICE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_MEMORYSTATSDEVICE_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

namespace AgcDriver::MemoryStats {

// DEBUG (APS5_MEM_STATS): wraps the device's vkGetDeviceProcAddr so memory, shader module, pipeline,
// image and buffer creation is counted; returns `real` unchanged when the stats are off.
[[nodiscard]] PFN_vkGetDeviceProcAddr TrackDevice(VkDevice device, VkPhysicalDevice physical, const VkPhysicalDeviceMemoryProperties& memory,
                                                  PFN_vkGetDeviceProcAddr real, PFN_vkGetPhysicalDeviceMemoryProperties2 budget);

}

#endif
