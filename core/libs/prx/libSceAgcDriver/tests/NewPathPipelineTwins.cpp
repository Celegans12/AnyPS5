#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PipelineAhead.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PipelineAheadTwins.hpp"
#include "Recompiler.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <mutex>
#include <set>
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

std::filesystem::path TempDirectory() {
    auto path = std::filesystem::temp_directory_path() / ("aps5-newpath-twins-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
    return path;
}

ShaderRecompiler::DescriptorBinding Binding(ShaderRecompiler::DescriptorRole role, std::uint32_t binding) {
    ShaderRecompiler::DescriptorBinding result{};
    result.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    result.role = role;
    result.binding = binding;
    result.count = 1;
    return result;
}

// Today's module (binds the page table) and its resolved twin (does not), as buildResolvedTwin pairs them.
std::shared_ptr<ShaderRecompiler::RecompileResult> Program(bool resolved, std::uint32_t salt) {
    auto program = std::make_shared<ShaderRecompiler::RecompileResult>();
    program->newPathResolved = resolved;
    program->spirv = std::vector<std::uint32_t>{0x07230203u, 0x00010500u, salt, 64u, 0u, resolved ? 0x1111u : 0x2222u};
    if (!resolved) program->bindings.push_back(Binding(ShaderRecompiler::DescriptorRole::BdaPagetable, 3));
    program->bindings.push_back(Binding(ShaderRecompiler::DescriptorRole::GuestBuffers, 4));
    program->pushConstants = {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    return program;
}

// A builder that records which threads built what, and returns the twin's first salt word as the handle.
struct Builder {
    std::mutex mutex;
    std::set<std::thread::id> threads;
    std::atomic<int> calls{0};
    std::atomic<int> sleepMs{0};
    std::size_t lastLayoutBindings = 0;
    std::size_t lastSpirvWords = 0;

    TwinBuilder Make() {
        return [this](const TwinModule& module) -> PipelineHandle {
            if (sleepMs != 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs.load()));
            std::lock_guard lock(mutex);
            ++calls;
            threads.insert(std::this_thread::get_id());
            lastLayoutBindings = module.layout != nullptr ? module.layout->bindings.size() : 0;
            lastSpirvWords = module.twin->spirv.size();
            return std::make_shared<std::uint32_t>(module.twin->spirv[2]);
        };
    }
};

std::uint32_t ValueOf(const PipelineHandle& handle) {
    return handle != nullptr ? *static_cast<const std::uint32_t*>(handle.get()) : 0;
}

void moduleFilesPerSourceVersion() {
    const auto directory = TempDirectory();
    const auto compiled = Program(false, 7);
    const auto twin = Program(true, 7);
    const auto key = TwinPipelineKey(*twin, true);
    Expect(key != TwinPipelineKey(*twin, false), "the push-constant use is not part of the key");
    Expect(key != TwinPipelineKey(*Program(true, 8), true), "two twins share a key");
    Expect(WriteTwinModule(directory, 0xabc, key, {twin, LayoutOf(*compiled), 1, 0x1234}), "the module file was not written");
    Expect(TwinModuleHeaderValid(directory, 0xabc, key), "the module file header was refused");
    const auto read = ReadTwinModule(directory, 0xabc, key);
    Expect(read.has_value(), "the module file did not read back");
    if (read) {
        Expect(read->twin->newPathResolved && read->twin->spirv.size() == twin->spirv.size() && read->twin->spirv[2] == 7u && read->twin->Identity() == twin->Identity(), "the twin module changed in the file");
        Expect(read->layout->bindings.size() == 2 && read->layout->bindings[0].role == ShaderRecompiler::DescriptorRole::BdaPagetable, "the layout bindings (today's module's) changed in the file");
        Expect(read->pushStages == 1 && read->programAddress == 0x1234, "the dispatch fields changed in the file");
    }
    Expect(!TwinModuleHeaderValid(directory, 0xdef, key) && !ReadTwinModule(directory, 0xdef, key).has_value(), "another SourceVersion's module file was trusted");
    Expect(!ReadTwinModule(directory, 0xabc, key ^ 1).has_value(), "a module file was read for another key");
    // A file renamed to another key's name is refused (the header names its key).
    std::filesystem::copy_file(TwinModulePath(directory, key), TwinModulePath(directory, key ^ 2));
    Expect(!ReadTwinModule(directory, 0xabc, key ^ 2).has_value(), "a module file under another key's name was trusted");
    std::filesystem::remove_all(directory);
}

void notReadyFallsBackAndNothingInline() {
    const auto directory = TempDirectory();
    auto manifest = std::make_shared<PipelineManifest>(directory / "newpath-pipelines.txt", 0x77);
    PipelineAhead ahead({true, 2}, manifest);
    TwinPipelines twins(ahead, directory / "newpath-twins", 0x77);
    Builder builder;
    builder.sleepMs = 50;
    const auto compiled = Program(false, 9);
    const auto twin = Program(true, 9);
    const auto key = TwinPipelineKey(*twin, true);
    int described = 0;
    const auto describe = [&] {
        ++described;
        return TwinModule{twin, LayoutOf(*compiled), 1, 0x99};
    };
    Expect(twins.Get(key, describe, builder.Make()) == nullptr, "a twin whose pipeline is not created yet was handed out (the dispatch must fall back)");
    Expect(twins.Get(key, describe, builder.Make()) == nullptr, "a pending twin pipeline was handed out");
    Expect(described == 1, "the twin module was described more than once");
    ahead.WaitIdle();
    Expect(ValueOf(twins.Get(key, describe, builder.Make())) == 9u, "the twin pipeline was not ready after its creation");
    const auto counts = twins.Totals();
    Expect(builder.calls == 1 && counts.createdOnWorker == 1 && counts.createdOnCaller == 0, "the twin pipeline was not created exactly once, on a worker");
    Expect(!builder.threads.contains(std::this_thread::get_id()), "the twin pipeline was created on the dispatch thread with the gate on");
    Expect(counts.notReady == 2, "the not-ready fallbacks were not counted");
    Expect(counts.modulesWritten == 1 && std::filesystem::exists(TwinModulePath(directory / "newpath-twins", key)), "the module file was not written by the worker");
    Expect(manifest->Known(key), "the twin's key was not recorded in the manifest");
    Expect(builder.lastLayoutBindings == 2, "the pipeline layout was not described from today's module's bindings");
    std::filesystem::remove_all(directory);
}

void nextRunPrewarmsFromTheManifest() {
    const auto directory = TempDirectory();
    const auto compiled = Program(false, 11);
    const auto twin = Program(true, 11);
    const auto key = TwinPipelineKey(*twin, true);
    const auto lost = TwinPipelineKey(*Program(true, 12), true);
    {
        auto manifest = std::make_shared<PipelineManifest>(directory / "newpath-pipelines.txt", 0x55);
        PipelineAhead ahead({true, 1}, manifest);
        TwinPipelines twins(ahead, directory / "newpath-twins", 0x55);
        Builder builder;
        static_cast<void>(twins.Get(key, [&] { return TwinModule{twin, LayoutOf(*compiled), 1, 0x11}; }, builder.Make()));
        ahead.WaitIdle();
        manifest->Record(lost); // a key whose module file was never written (run killed before the worker ran)
    }
    {
        auto manifest = std::make_shared<PipelineManifest>(directory / "newpath-pipelines.txt", 0x55);
        Expect(manifest->LoadedFromFile() && manifest->Known(key) && manifest->Known(lost), "the manifest did not keep the twin keys for the next run");
        PipelineAhead ahead({true, 2}, manifest);
        TwinPipelines twins(ahead, directory / "newpath-twins", 0x55);
        Builder builder;
        twins.StartPrewarm(builder.Make());
        // Wait for the prewarm thread to queue, then for the workers.
        for (int tries = 0; tries < 200 && twins.Totals().prewarmQueued + twins.Totals().prewarmSkipped < 2; ++tries) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        ahead.WaitIdle();
        const auto counts = twins.Totals();
        Expect(counts.prewarmQueued == 1 && counts.prewarmSkipped == 1, "the prewarm did not queue exactly the keys with module files");
        Expect(builder.calls == 1 && builder.lastSpirvWords == twin->spirv.size() && builder.lastLayoutBindings == 2, "the prewarm did not build the twin from its module file");
        Expect(!builder.threads.contains(std::this_thread::get_id()), "the prewarm built on the calling thread");
        int described = 0;
        Expect(ValueOf(twins.Get(key, [&] { ++described; return TwinModule{twin, LayoutOf(*compiled), 1, 0x11}; }, builder.Make())) == 11u, "the prewarmed twin pipeline was not ready at its first dispatch");
        Expect(described == 0 && twins.Totals().notReady == 0, "the first dispatch of a prewarmed twin fell back");
        // The lost key is requested by its first dispatch like a new one.
        const auto lostTwin = Program(true, 12);
        Expect(twins.Get(lost, [&] { return TwinModule{lostTwin, LayoutOf(*Program(false, 12)), 1, 0x12}; }, builder.Make()) == nullptr, "a key without a module file was ready");
        ahead.WaitIdle();
        Expect(ValueOf(twins.Get(lost, [&] { return TwinModule{lostTwin, LayoutOf(*Program(false, 12)), 1, 0x12}; }, builder.Make())) == 12u, "a key without a module file was not created at its first dispatch");
        twins.Stop();
    }
    {
        // Another SourceVersion: nothing is prewarmed.
        auto manifest = std::make_shared<PipelineManifest>(directory / "newpath-pipelines.txt", 0x56);
        PipelineAhead ahead({true, 1}, manifest);
        TwinPipelines twins(ahead, directory / "newpath-twins", 0x56);
        Builder builder;
        twins.StartPrewarm(builder.Make());
        twins.Stop();
        Expect(builder.calls == 0 && twins.Totals().prewarmQueued == 0, "another SourceVersion's manifest was prewarmed");
    }
    std::filesystem::remove_all(directory);
}

void stopForgetsTheOldDevice() {
    const auto directory = TempDirectory();
    PipelineAhead ahead({true, 1}, nullptr);
    TwinPipelines twins(ahead, directory, 0x1);
    Builder builder;
    const auto twin = Program(true, 21);
    const auto key = TwinPipelineKey(*twin, false);
    const auto describe = [&] { return TwinModule{twin, LayoutOf(*Program(false, 21)), 0, 0}; };
    static_cast<void>(twins.Get(key, describe, builder.Make()));
    ahead.WaitIdle();
    Expect(twins.Get(key, describe, builder.Make()) != nullptr, "the pipeline was not ready before the reset");
    twins.Stop();
    Expect(twins.Get(key, describe, builder.Make()) == nullptr, "a pipeline of the torn-down device was handed out after the reset");
    ahead.WaitIdle();
    Expect(builder.calls == 2 && twins.Get(key, describe, builder.Make()) != nullptr, "the pipeline was not created again for the new device");
    std::filesystem::remove_all(directory);
}

void gateOffCreatesOnTheCaller() {
    const auto directory = TempDirectory();
    PipelineAhead ahead({false, 1}, nullptr);
    TwinPipelines twins(ahead, directory, 0x1);
    Builder builder;
    const auto twin = Program(true, 31);
    Expect(ValueOf(twins.Get(TwinPipelineKey(*twin, false), [&] { return TwinModule{twin, LayoutOf(*Program(false, 31)), 0, 0}; }, builder.Make())) == 31u, "gate off: the pipeline was not created at once");
    Expect(builder.threads.contains(std::this_thread::get_id()) && twins.Totals().createdOnCaller == 1 && twins.Totals().modulesWritten == 0, "gate off: not created on the caller (today's inline creation)");
    std::filesystem::remove_all(directory);
}

}

int main() {
    try {
        moduleFilesPerSourceVersion();
        notReadyFallsBackAndNothingInline();
        nextRunPrewarmsFromTheManifest();
        stopForgetsTheOldDevice();
        gateOffCreatesOnTheCaller();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d twin pipeline checks failed\n", failures);
        return 1;
    }
    std::puts("new-path twin pipeline tests passed");
    return 0;
}
