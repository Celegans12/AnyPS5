#include "prx/libSceAgcDriver/Graphics/include/PipelineCreationGate.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace AgcDriver::Graphics {

namespace {

std::uint64_t EnvironmentNumber(const char* name, std::uint64_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') return fallback;
    return std::strtoull(text, nullptr, 10);
}

}

PipelineCreationGate::PipelineCreationGate(Settings settings) : settings(std::move(settings)) {
    stats.budgetBytes = this->settings.budgetBytes;
}

std::uint64_t PipelineCreationGate::PhysicalMemoryBytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? status.ullTotalPhys : 0u;
#else
    const auto pages = sysconf(_SC_PHYS_PAGES);
    const auto page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page) : 0u;
#endif
}

std::uint64_t PipelineCreationGate::AvailableMemoryBytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? status.ullAvailPhys : 0u;
#else
    const auto pages = sysconf(_SC_AVPHYS_PAGES);
    const auto page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page) : 0u;
#endif
}

PipelineCreationGate::Settings PipelineCreationGate::SettingsFromEnvironment() {
    Settings settings;
    settings.bytesPerWord = std::max<std::uint64_t>(1u, EnvironmentNumber("APS5_PIPELINE_COMPILE_BYTES_PER_WORD", settings.bytesPerWord));
    settings.budgetBytes = EnvironmentNumber("APS5_PIPELINE_COMPILE_BUDGET_MIB", PhysicalMemoryBytes() / 8u >> 20u) << 20u;
    settings.smallBytes = EnvironmentNumber("APS5_PIPELINE_COMPILE_SMALL_MIB", settings.smallBytes >> 20u) << 20u;
    if (std::getenv("APS5_PIPELINE_COMPILE_BUDGET_MIB") == nullptr) settings.availableBytes = &PipelineCreationGate::AvailableMemoryBytes;
    return settings;
}

PipelineCreationGate& PipelineCreationGate::Shared() {
    static PipelineCreationGate gate(SettingsFromEnvironment());
    return gate;
}

bool PipelineCreationGate::fits(std::uint64_t bytes) const {
    if (stats.inFlight == 0) return true;
    auto budget = settings.budgetBytes;
    if (settings.availableBytes) {
        const auto available = settings.availableBytes();
        if (available != 0) budget = std::min(budget, stats.inFlightBytes + available / 2u);
    }
    return stats.inFlightBytes + bytes <= budget;
}

PipelineCreationGate::Ticket PipelineCreationGate::Admit(std::size_t words, bool mayWait) {
    if (settings.budgetBytes == 0) return {};
    const auto bytes = EstimateBytes(words);
    std::unique_lock lock(mutex);
    if (!fits(bytes)) {
        if (mayWait && bytes > settings.smallBytes) {
            const auto start = std::chrono::steady_clock::now();
            ++stats.waited;
            while (!fits(bytes)) released.wait_for(lock, std::chrono::milliseconds(250));
            stats.waitNanoseconds += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        } else {
            ++stats.unwaited;
        }
    }
    ++stats.admitted;
    ++stats.inFlight;
    stats.inFlightBytes += bytes;
    stats.peakInFlight = std::max(stats.peakInFlight, stats.inFlight);
    stats.peakBytes = std::max(stats.peakBytes, stats.inFlightBytes);
    return Ticket(this, bytes);
}

void PipelineCreationGate::release(std::uint64_t bytes) noexcept {
    {
        std::lock_guard lock(mutex);
        --stats.inFlight;
        stats.inFlightBytes -= std::min(stats.inFlightBytes, bytes);
    }
    released.notify_all();
}

PipelineCreationGate::Stats PipelineCreationGate::Totals() const {
    std::lock_guard lock(mutex);
    return stats;
}

}
