#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PROFILEWINDOWS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PROFILEWINDOWS_HPP

#include <chrono>

// The windows of the profile lines that total a stretch of time: [gputime] (APS5_PROFILE_GPU),
// [draws] (APS5_PROFILE_DRAW), [command-mix] (APS5_PROFILE_COMMAND_MIX), [resident-census]
// (APS5_PROFILE_RESIDENT_CENSUS) and [mem] (APS5_MEM_STATS).
//
// By default (the game) each line closes its window on its own wall clock: every 10 s for the first
// four, every APS5_MEM_STATS seconds for [mem], and names it "10 s" as it always did. A frame replay
// switches to frame windows (agc_frame_replay does unless --wall-windows): then no line closes on
// the clock, and the replay closes every window at once with Close, at each replayed frame's flip
// ("frame-<n>") and once more at its end ("replay-end"), so a 3-frame replay prints one line of
// each kind per frame instead of 0-1 per 10 s.
namespace AgcDriver::ProfileWindows {

void UseFrameWindows(bool on);
[[nodiscard]] bool FrameWindows();

// Clock windows: true (and `last` restarted) once 10 s passed since `last`, as each line's own
// check did; never true with frame windows.
[[nodiscard]] bool ClockDue(std::chrono::steady_clock::time_point& last);

// Prints every enabled line now for the window `label` (a token without spaces: "frame-3"),
// starts the next windows, and waits until the profile output has written them.
void Close(const char* label);

}

#endif
