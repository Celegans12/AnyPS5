#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_COMMANDLOG_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_COMMANDLOG_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

namespace AgcDriver::CommandLog {

bool Enabled();
[[nodiscard]] PFN_vkGetDeviceProcAddr Wrap(VkDevice device, PFN_vkGetDeviceProcAddr real, bool force = false);

struct Stream {
    std::uint32_t pool = 0;
    std::uint64_t commands = 0;
    std::uint64_t hash = 0;
};
std::vector<Stream> Streams();
std::string Line(std::uint64_t present);
void Reset();

}

#endif
