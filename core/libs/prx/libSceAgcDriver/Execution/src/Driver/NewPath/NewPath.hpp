#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_NEWPATH_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_NEWPATH_HPP

#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/Router.hpp"

namespace AgcDriver {
class VulkanDevice;
}

namespace AgcDriver::NewPath {

[[nodiscard]] bool Enabled();

Decision RouteCommand(CommandKind kind, const VulkanDevice* device = nullptr);

void Close(const char* window);

}

#endif
