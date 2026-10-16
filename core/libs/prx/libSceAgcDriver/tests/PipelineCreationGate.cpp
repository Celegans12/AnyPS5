#include "prx/libSceAgcDriver/Graphics/include/PipelineCreationGate.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

using AgcDriver::Graphics::PipelineCreationGate;

namespace {

constexpr std::uint64_t MiB = 1ull << 20u;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

PipelineCreationGate::Settings Settings(std::uint64_t budget, std::uint64_t small = 0) {
    PipelineCreationGate::Settings settings;
    settings.bytesPerWord = 1024;
    settings.budgetBytes = budget;
    settings.smallBytes = small;
    return settings;
}

bool AdmittedWithin(std::atomic<bool>& admitted, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!admitted.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return admitted.load();
}

void aloneAndQueued() {
    PipelineCreationGate gate(Settings(8 * MiB));
    auto first = gate.Admit(16 * 1024, true);
    Require(gate.Totals().inFlight == 1 && gate.Totals().inFlightBytes == 16 * MiB, "a creation over the budget did not run alone");
    std::atomic<bool> admitted{false};
    std::thread second([&] {
        const auto ticket = gate.Admit(4 * 1024, true);
        admitted = true;
    });
    const bool early = AdmittedWithin(admitted, std::chrono::milliseconds(200));
    first = {};
    const bool late = AdmittedWithin(admitted, std::chrono::milliseconds(5000));
    second.join();
    const auto stats = gate.Totals();
    Require(!early, "a creation was admitted while the one in flight took the whole budget");
    Require(late && stats.waited == 1 && stats.admitted == 2 && stats.inFlight == 0 && stats.inFlightBytes == 0, "the waiting creation was not admitted once the first finished");
    Require(stats.peakInFlight == 1 && stats.peakBytes == 16 * MiB, "the peak counts more than one creation at once");
    std::puts("a creation over the budget runs alone; the next one waits for it");
}

void fitsTogether() {
    PipelineCreationGate gate(Settings(8 * MiB));
    const auto first = gate.Admit(3 * 1024, true);
    const auto second = gate.Admit(4 * 1024, true);
    const auto stats = gate.Totals();
    Require(stats.waited == 0 && stats.inFlight == 2 && stats.peakBytes == 7 * MiB, "two creations inside the budget did not run together");
    std::puts("creations that fit the budget run together");
}

void smallAndLocked() {
    PipelineCreationGate gate(Settings(8 * MiB, 2 * MiB));
    const auto big = gate.Admit(8 * 1024, true);
    const auto small = gate.Admit(2 * 1024, true);
    const auto locked = gate.Admit(6 * 1024, false);
    const auto stats = gate.Totals();
    Require(stats.waited == 0 && stats.unwaited == 2 && stats.inFlight == 3 && stats.inFlightBytes == 16 * MiB, "a small creation or one under the GPU mutex waited for the budget");
    std::puts("small creations and creations that may not wait run past the budget and are counted");
}

void availableMemory() {
    auto settings = Settings(1024 * MiB);
    std::atomic<std::uint64_t> available{2 * MiB};
    settings.availableBytes = [&] { return available.load(); };
    PipelineCreationGate gate(settings);
    const auto first = gate.Admit(1024, true);
    std::atomic<bool> admitted{false};
    std::thread second([&] {
        const auto ticket = gate.Admit(2 * 1024, true);
        admitted = true;
    });
    const bool early = AdmittedWithin(admitted, std::chrono::milliseconds(400));
    available = 512 * MiB;
    const bool late = AdmittedWithin(admitted, std::chrono::milliseconds(5000));
    second.join();
    Require(!early, "a creation was admitted past half the available memory");
    Require(late && gate.Totals().waited == 1, "the creation was not admitted once memory became available");
    std::puts("half the available memory bounds the creations in flight");
}

void disabled() {
    PipelineCreationGate gate(Settings(0));
    const auto first = gate.Admit(1u << 20u, true);
    const auto second = gate.Admit(1u << 20u, true);
    const auto stats = gate.Totals();
    Require(stats.admitted == 0 && stats.inFlight == 0, "a zero budget did not turn the gate off");
    std::puts("a zero budget turns the gate off");
}

void defaults() {
    const auto settings = PipelineCreationGate::SettingsFromEnvironment();
    const auto physical = PipelineCreationGate::PhysicalMemoryBytes();
    Require(physical != 0 && PipelineCreationGate::AvailableMemoryBytes() != 0, "physical or available memory reads zero");
    if (std::getenv("APS5_PIPELINE_COMPILE_BUDGET_MIB") == nullptr) Require(settings.budgetBytes == (physical / 8u >> 20u << 20u) && settings.availableBytes != nullptr, "the default budget is not an eighth of physical memory bounded by the available memory");
    std::printf("default budget %llu MiB of %llu MiB physical, %llu bytes per word, small below %llu MiB\n", static_cast<unsigned long long>(settings.budgetBytes >> 20u), static_cast<unsigned long long>(physical >> 20u), static_cast<unsigned long long>(settings.bytesPerWord), static_cast<unsigned long long>(settings.smallBytes >> 20u));
}

}

int main() {
    try {
        aloneAndQueued();
        fitsTogether();
        smallAndLocked();
        availableMemory();
        disabled();
        defaults();
        std::puts("pipeline creation gate tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "pipeline creation gate test failed: %s\n", error.what());
        return 1;
    }
}
