#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_RESIDENTCOPYIN_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_RESIDENTCOPYIN_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"

namespace AgcDriver::Graphics {
class Recorder;
}

// Resident staging shadows (APS5_NO_RESIDENT_COPY_IN): 0 passed, 77 skipped (no host imports or
// write watching), throws on a failure. `budget`: the APS5_RESIDENT_MIB fallback variant.
int RunResidentCopyInTests(const AgcDriver::Graphics::Context& context, AgcDriver::Graphics::Recorder& recorder, bool budget);

#endif
