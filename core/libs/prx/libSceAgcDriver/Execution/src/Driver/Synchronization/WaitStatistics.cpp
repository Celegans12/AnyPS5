#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace AgcDriver::DriverDetail {

PollStats& Driver::pollStats() {
    static thread_local PollStats stats;
    return stats;
}

WaitOutcomes& Driver::waitOutcomes() {
    static thread_local WaitOutcomes outcomes;
    return outcomes;
}

Graphics::Recorder::LateStatistics& Driver::lateCountsSeen() {
    static thread_local Graphics::Recorder::LateStatistics seen{};
    return seen;
}

EpochBumps& Driver::epochBumps() {
    static thread_local EpochBumps bumps;
    return bumps;
}

std::chrono::steady_clock::time_point& Driver::lastEpochBump() {
    static thread_local std::chrono::steady_clock::time_point at{};
    return at;
}

void Driver::bumpEpoch(std::uint64_t EpochBumps::*counter) {
    lastEpochBump() = std::chrono::steady_clock::now();
    GuestMemory::BumpCollectEpoch();
    if (DrawPipeline::Active()) DrawPipeline::EpochToken().fetch_add(1, std::memory_order_relaxed);
    ++(epochBumps().*counter);
}

bool Driver::packetEpoch() {
    static const bool packet = std::getenv("APS5_PACKET_EPOCH") != nullptr;
    return packet;
}

namespace {

std::array<std::atomic<std::uint64_t>, WaitClassCount> waitClassCounts{};
std::array<std::atomic<std::uint64_t>, WaitClassCount> waitClassUs{};
std::atomic<std::uint64_t> waitDrainsSkipped{0}, waitTriesSkipped{0};

}

bool GpuWaits() {
    static const bool enabled = [] { const char* text = std::getenv("APS5_GPU_WAITS"); return text != nullptr && std::strcmp(text, "0") != 0; }();
    return enabled;
}

WaitClassTotals& ThreadWaitClasses() {
    static thread_local WaitClassTotals totals;
    return totals;
}

void NoteWaitClass(WaitClass which, double ms) {
    const auto index = static_cast<std::size_t>(which);
    auto& totals = ThreadWaitClasses();
    ++totals.count[index];
    totals.ms[index] += ms;
    waitClassCounts[index].fetch_add(1, std::memory_order_relaxed);
    waitClassUs[index].fetch_add(static_cast<std::uint64_t>(ms * 1000.0), std::memory_order_relaxed);
}

void NoteWaitDrainSkipped() {
    ++ThreadWaitClasses().drainsSkipped;
    waitDrainsSkipped.fetch_add(1, std::memory_order_relaxed);
}

void NoteWaitTrySkipped() {
    ++ThreadWaitClasses().triesSkipped;
    waitTriesSkipped.fetch_add(1, std::memory_order_relaxed);
}

WaitClassTotals WaitClassCounts() {
    WaitClassTotals totals;
    for (std::size_t i = 0; i < WaitClassCount; ++i) {
        totals.count[i] = waitClassCounts[i].load(std::memory_order_relaxed);
        totals.ms[i] = static_cast<double>(waitClassUs[i].load(std::memory_order_relaxed)) / 1000.0;
    }
    totals.drainsSkipped = waitDrainsSkipped.load(std::memory_order_relaxed);
    totals.triesSkipped = waitTriesSkipped.load(std::memory_order_relaxed);
    return totals;
}

}
