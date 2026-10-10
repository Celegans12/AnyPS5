#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCH_LIVESNAPSHOTRETRY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCH_LIVESNAPSHOTRETRY_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"

namespace AgcDriver::DriverDetail {

template <typename Attempt>
void RetryLiveSnapshot(Attempt&& attempt) {
    const bool allowed = !GuestMemory::GpuMutex().HeldByThisThread() && !Graphics::Recorder::InCompletion();
    try {
        attempt(false);
        return;
    } catch (const Graphics::LiveSnapshotInvalidated&) {
        if (!allowed || GuestMemory::GpuMutex().HeldByThisThread() || Graphics::Recorder::InCompletion()) throw;
    }
    attempt(true);
}

}

#endif
