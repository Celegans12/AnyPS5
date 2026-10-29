#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PipelineAhead.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using namespace AgcDriver::NewPath;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

std::filesystem::path TempManifest(const char* name) {
    auto path = std::filesystem::temp_directory_path() / ("aps5-newpath-gate-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())) / name;
    std::filesystem::create_directories(path.parent_path());
    return path;
}

PipelineFactory Make(int value, std::atomic<int>* calls = nullptr, std::thread::id* creator = nullptr, int sleepMs = 0) {
    return [=]() -> PipelineHandle {
        if (sleepMs != 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
        if (calls != nullptr) ++*calls;
        if (creator != nullptr) *creator = std::this_thread::get_id();
        return std::make_shared<int>(value);
    };
}

int ValueOf(const PipelineHandle& handle) {
    return handle != nullptr ? *static_cast<const int*>(handle.get()) : -1;
}

void gateCreatesOffTheCallingThread() {
    PipelineAhead ahead({true, 2}, nullptr);
    std::thread::id creator;
    Expect(ahead.Request(1, Make(11, nullptr, &creator, 20)) == PipelineAhead::Lookup::Pending, "a gated request was not pending");
    ahead.WaitIdle();
    Expect(ValueOf(ahead.Ready(1)) == 11, "the gated pipeline was not created");
    Expect(creator != std::this_thread::get_id() && ahead.CreatedOnWorker(1), "the gated pipeline was created on the calling (decode) thread");
    Expect(ahead.Request(1, Make(99)) == PipelineAhead::Lookup::Ready && ValueOf(ahead.Ready(1)) == 11, "a second request recreated the pipeline");
}

void gateOffCreatesInline() {
    PipelineAhead ahead({false, 2}, nullptr);
    std::thread::id creator;
    Expect(ahead.Request(2, Make(22, nullptr, &creator)) == PipelineAhead::Lookup::Ready, "gate off did not create at once");
    Expect(creator == std::this_thread::get_id() && !ahead.CreatedOnWorker(2) && ahead.Totals().createdInline == 1, "gate off did not create on the calling thread");
    Expect(ValueOf(ahead.Acquire(3, Make(33))) == 33, "gate off: acquire did not return the pipeline");
}

void neverSeenBlocksSeenFallsBack() {
    const auto path = TempManifest("newpath-pipelines.txt");
    {
        auto first = std::make_shared<PipelineManifest>(path, 0x1234);
        PipelineAhead ahead({true, 1}, first);
        Expect(ValueOf(ahead.Acquire(7, Make(77, nullptr, nullptr, 20))) == 77, "a never-seen shape did not block for its pipeline");
        Expect(ahead.Totals().blockedWaits == 1 && ahead.Totals().pendingFallbacks == 0, "a never-seen shape was not counted as blocked");
        ahead.WaitIdle();
    }
    auto second = std::make_shared<PipelineManifest>(path, 0x1234);
    Expect(second->LoadedFromFile() && second->Known(7), "the manifest did not keep the shape for the next run");
    PipelineAhead ahead({true, 1}, second);
    Expect(ahead.Acquire(7, Make(78, nullptr, nullptr, 50)) == nullptr, "a shape a previous run saw blocked instead of falling back while its pipeline is created");
    Expect(ahead.Totals().pendingFallbacks == 1 && ahead.Totals().blockedWaits == 0, "the fallback was not counted");
    ahead.WaitIdle();
    Expect(ValueOf(ahead.Ready(7)) == 78, "the fallen-back shape's pipeline was not created behind the frame");

    auto otherVersion = std::make_shared<PipelineManifest>(path, 0x9999);
    Expect(!otherVersion->LoadedFromFile() && !otherVersion->Known(7), "another SourceVersion's manifest was trusted");
    std::filesystem::remove_all(path.parent_path());
}

void prewarmFromTheManifest() {
    const auto path = TempManifest("newpath-pipelines.txt");
    {
        auto manifest = std::make_shared<PipelineManifest>(path, 0x42);
        Expect(manifest->Record(5) && manifest->Record(6) && !manifest->Record(5), "manifest records are not once per key");
    }
    auto manifest = std::make_shared<PipelineManifest>(path, 0x42);
    PipelineAhead ahead({true, 2}, manifest);
    std::atomic<int> calls{0};
    const auto queued = ahead.Prewarm([&](PipelineKey key) -> PipelineFactory {
        if (key == 6) return {};
        return Make(static_cast<int>(key) * 10, &calls);
    });
    ahead.WaitIdle();
    Expect(queued == 1 && calls == 1 && ValueOf(ahead.Ready(5)) == 50 && ahead.Ready(6) == nullptr, "prewarm did not create exactly the resolvable recorded shapes");
    Expect(ahead.Totals().prewarmed == 1, "prewarmed count is wrong");
    std::filesystem::remove_all(path.parent_path());
}

void failedCreationIsNotReady() {
    PipelineAhead ahead({true, 1}, nullptr);
    ahead.Request(9, [] () -> PipelineHandle { throw std::runtime_error("driver refused"); });
    ahead.WaitIdle();
    Expect(ahead.Ready(9) == nullptr && ahead.Totals().failed == 1, "a failed creation was not reported");
    Expect(ahead.Acquire(9, Make(1)) == nullptr, "a failed shape was handed out");
}

}

int main() {
    try {
        gateCreatesOffTheCallingThread();
        gateOffCreatesInline();
        neverSeenBlocksSeenFallsBack();
        prewarmFromTheManifest();
        failedCreationIsNotReady();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d pipeline gate checks failed\n", failures);
        return 1;
    }
    std::puts("new-path pipeline gate tests passed");
    return 0;
}
