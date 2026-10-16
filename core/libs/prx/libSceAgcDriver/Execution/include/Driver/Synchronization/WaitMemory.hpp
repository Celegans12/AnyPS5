#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_WAITMEMORY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_WAITMEMORY_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/LabelStores.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace AgcDriver::DriverDetail {

inline constexpr const char* LabelRefusalNames[6] = {"none", "trust-off", "queued", "overwritten", "unclosed", "behind-completion"};

inline constexpr std::chrono::microseconds PollTryInterval{1000};

struct LabelStore {
    std::uint64_t stamp = 0;
    std::uint32_t value = 0;
    std::uint32_t queue = 0xffffffffu;
};

struct WriteRecord {
    std::uint64_t target;
    std::uint64_t length;
    std::uint32_t queue;
    std::uint32_t opcode;
};

struct PollStats {
    std::uint64_t ticks = 0, tableHits = 0, tries = 0, triesFailed = 0, submits = 0, reaps = 0, reapsWithWork = 0;

    std::uint64_t reapsBehindCompletion = 0, reapsOther = 0;
    std::array<std::uint64_t, 6> refusals{};
    double longestTryHoldUs = 0;
    double tryHoldUs = 0;
};

struct WaitOutcomes {
    std::uint64_t atEntry = 0, fromRecorder = 0, fromRecorderSameQueue = 0, fromRecorderPolling = 0, polled = 0, timedOut = 0, pollSubmits = 0, pollReaps = 0;
    std::uint64_t fromRecorderUnlocked = 0, entriesUnlocked = 0, entryTriesFailed = 0, fromRecorderLate = 0, lateRefusedCpuStore = 0;
};

struct EpochBumps {
    std::uint64_t submissions = 0, waits = 0, drains = 0, reaps = 0, packets = 0;
};

enum class WaitClass : std::uint8_t { Entry = 0, HeldAtSubmit, HistorySameQueue, HistoryCrossQueue, GpuSameQueue, GpuCrossQueue, GpuLaterSameQueue, GpuLaterCrossQueue, PipelineLabel, MemoryGpu, MemoryCpu, Landed, LandedTimeline, Count };

inline constexpr std::size_t WaitClassCount = static_cast<std::size_t>(WaitClass::Count);

inline constexpr const char* WaitClassNames[WaitClassCount] = {"entry", "held-at-submit", "history-same", "history-cross", "gpu-same", "gpu-cross", "gpu-later-same", "gpu-later-cross", "pipeline-label", "memory-gpu", "memory-cpu", "landed", "landed-timeline"};

struct WaitClassTotals {
    std::array<std::uint64_t, WaitClassCount> count{};
    std::array<double, WaitClassCount> ms{};
    std::uint64_t drainsSkipped = 0, triesSkipped = 0;
};

bool GpuWaits();
void NoteWaitClass(WaitClass which, double ms);
void NoteWaitDrainSkipped();
void NoteWaitTrySkipped();
WaitClassTotals& ThreadWaitClasses();
WaitClassTotals WaitClassCounts();

void PollSleep();

}

#endif
