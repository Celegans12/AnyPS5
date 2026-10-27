#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include "prx/libSceAgcDriver/Execution/include/CommandMix.hpp"
#include "prx/libSceAgcDriver/Execution/include/MemoryStats.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidentCensus.hpp"
#include <atomic>
#include <mutex>

namespace AgcDriver::ProfileWindows {

namespace {

std::atomic<bool> frameWindows{false};

}

void UseFrameWindows(bool on) {
    frameWindows.store(on, std::memory_order_release);
}

bool FrameWindows() {
    return frameWindows.load(std::memory_order_acquire);
}

bool ClockDue(std::chrono::steady_clock::time_point& last) {
    if (FrameWindows()) return false;
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(10)) return false;
    last = now;
    return true;
}

void Close(const char* label) {
    // One closer at a time: the lines of two windows never interleave.
    static std::mutex closing;
    std::lock_guard lock(closing);
    Graphics::Recorder::CloseGpuTimingWindow(label);
    Graphics::CloseDrawProfileWindow(label);
    CommandMix::Close(label);
    Graphics::ResidentCensus::Close(label);
    MemoryStats::ReportNow(label);
    ProfileOutput_nid_no_patch().Flush();
}

}
