#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PipelineAheadTwins.hpp"
#include "Recompiler.hpp"
#include "ShaderDiskCache.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <system_error>
#include <vector>

namespace AgcDriver::NewPath {

namespace {

constexpr std::uint32_t TwinMagic = 0x57545041u; // "APTW"
constexpr std::uint32_t TwinFormat = 1;

struct Header {
    std::uint32_t magic = TwinMagic;
    std::uint32_t format = TwinFormat;
    std::uint64_t sourceVersion = 0;
    std::uint64_t key = 0;
    std::uint64_t programAddress = 0;
    std::uint32_t pushStages = 0;
    std::uint32_t reserved = 0;
    std::uint64_t twinBytes = 0;
    std::uint64_t layoutBytes = 0;
};
static_assert(sizeof(Header) == 56);

std::optional<Header> readHeader(std::ifstream& in, std::uint64_t sourceVersion, PipelineKey key) {
    Header header;
    if (!in.read(reinterpret_cast<char*>(&header), sizeof(header))) return std::nullopt;
    if (header.magic != TwinMagic || header.format != TwinFormat || header.sourceVersion != sourceVersion || header.key != key) return std::nullopt;
    return header;
}

}

PipelineKey TwinPipelineKey(const ShaderRecompiler::RecompileResult& twin, bool pushConstants) {
    const auto identity = twin.Identity();
    std::uint64_t key = identity.low ^ ((identity.high << 21u) | (identity.high >> 43u));
    key = (key ^ (pushConstants ? 0x9e3779b97f4a7c15ull : 0x2545f4914f6cdd1dull)) * 0xbf58476d1ce4e5b9ull;
    key ^= key >> 31u;
    return key == 0 ? 1 : key;
}

std::shared_ptr<const ShaderRecompiler::RecompileResult> LayoutOf(const ShaderRecompiler::RecompileResult& program) {
    auto layout = std::make_shared<ShaderRecompiler::RecompileResult>();
    layout->bindings = program.bindings;
    return layout;
}

std::filesystem::path TwinModulePath(const std::filesystem::path& directory, PipelineKey key) {
    char name[32];
    std::snprintf(name, sizeof(name), "%016llx.twin", static_cast<unsigned long long>(key));
    return directory / name;
}

bool WriteTwinModule(const std::filesystem::path& directory, std::uint64_t sourceVersion, PipelineKey key, const TwinModule& module) {
    if (directory.empty() || module.twin == nullptr || module.layout == nullptr) return false;
    std::vector<std::byte> twin;
    std::vector<std::byte> layout;
    ShaderRecompiler::ShaderDiskCache::EncodeResult(*module.twin, twin);
    ShaderRecompiler::ShaderDiskCache::EncodeResult(*module.layout, layout);
    Header header;
    header.sourceVersion = sourceVersion;
    header.key = key;
    header.programAddress = module.programAddress;
    header.pushStages = module.pushStages;
    header.twinBytes = twin.size();
    header.layoutBytes = layout.size();
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    const auto path = TwinModulePath(directory, key);
    auto temporary = path;
    temporary += ".tmp" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out.write(reinterpret_cast<const char*>(twin.data()), static_cast<std::streamsize>(twin.size()));
        out.write(reinterpret_cast<const char*>(layout.data()), static_cast<std::streamsize>(layout.size()));
        if (!out) {
            out.close();
            std::filesystem::remove(temporary, error);
            return false;
        }
    }
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return false;
    }
    return true;
}

bool TwinModuleHeaderValid(const std::filesystem::path& directory, std::uint64_t sourceVersion, PipelineKey key) {
    std::ifstream in(TwinModulePath(directory, key), std::ios::binary);
    return in && readHeader(in, sourceVersion, key).has_value();
}

std::optional<TwinModule> ReadTwinModule(const std::filesystem::path& directory, std::uint64_t sourceVersion, PipelineKey key) {
    std::ifstream in(TwinModulePath(directory, key), std::ios::binary);
    if (!in) return std::nullopt;
    const auto header = readHeader(in, sourceVersion, key);
    if (!header || header->twinBytes > (1ull << 31u) || header->layoutBytes > (1ull << 31u)) return std::nullopt;
    std::vector<std::byte> twinBytes(header->twinBytes);
    std::vector<std::byte> layoutBytes(header->layoutBytes);
    if (!in.read(reinterpret_cast<char*>(twinBytes.data()), static_cast<std::streamsize>(twinBytes.size())) || !in.read(reinterpret_cast<char*>(layoutBytes.data()), static_cast<std::streamsize>(layoutBytes.size()))) return std::nullopt;
    auto twin = std::make_shared<ShaderRecompiler::RecompileResult>();
    auto layout = std::make_shared<ShaderRecompiler::RecompileResult>();
    if (!ShaderRecompiler::ShaderDiskCache::DecodeResult(twinBytes, *twin) || !ShaderRecompiler::ShaderDiskCache::DecodeResult(layoutBytes, *layout)) return std::nullopt;
    if (!twin->newPathResolved || TwinPipelineKey(*twin, header->pushStages != 0) != key) return std::nullopt;
    return TwinModule{std::move(twin), std::move(layout), header->pushStages, header->programAddress};
}

TwinPipelines::TwinPipelines(PipelineAhead& ahead, std::filesystem::path directory, std::uint64_t sourceVersion) : ahead(ahead), directory(std::move(directory)), sourceVersion(sourceVersion) {}

TwinPipelines::~TwinPipelines() {
    stopping.store(true, std::memory_order_release);
    if (prewarm.joinable()) prewarm.join();
}

PipelineHandle TwinPipelines::timed(PipelineKey key, const TwinModule& module, const TwinBuilder& build, bool worker) {
    const auto start = std::chrono::steady_clock::now();
    auto handle = build(module);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::uint64_t serial = 0;
    {
        std::lock_guard lock(mutex);
        if (worker) {
            serial = ++counts.createdOnWorker;
            counts.workerMs += ms;
        } else {
            serial = ++counts.createdOnCaller;
            counts.callerMs += ms;
            counts.maxCallerMs = std::max(counts.maxCallerMs, ms);
            if (ms > 1000.0) ++counts.callerOverSecond;
        }
    }
    if (serial <= 512 || ms > 1000.0) std::fprintf(stderr, "[newpath] twin pipeline %016llx created on %s in %.0f ms (%zu SPIR-V words, program 0x%llx)\n", static_cast<unsigned long long>(key), worker ? "a creation worker" : "the calling thread", ms, module.twin->spirv.size(), static_cast<unsigned long long>(module.programAddress));
    return handle;
}

PipelineFactory TwinPipelines::factoryFor(PipelineKey key, std::shared_ptr<const TwinModule> module, TwinBuilder build, std::thread::id caller) {
    return [this, key, module = std::move(module), build = std::move(build), caller]() -> PipelineHandle {
        const bool worker = std::this_thread::get_id() != caller;
        std::optional<TwinModule> loaded;
        const TwinModule* use = module.get();
        if (use != nullptr) {
            if (worker && !std::filesystem::exists(TwinModulePath(directory, key))) {
                const bool written = WriteTwinModule(directory, sourceVersion, key, *use);
                std::lock_guard lock(mutex);
                ++(written ? counts.modulesWritten : counts.moduleWriteFailures);
            }
        } else {
            loaded = ReadTwinModule(directory, sourceVersion, key);
            if (!loaded) {
                std::lock_guard lock(mutex);
                ++counts.moduleReadFailures;
                return nullptr;
            }
            use = &*loaded;
        }
        return timed(key, *use, build, worker);
    };
}

PipelineHandle TwinPipelines::Get(PipelineKey key, const std::function<TwinModule()>& describe, const TwinBuilder& build) {
    if (const auto state = ahead.Find(key); state.has_value()) {
        if (*state == PipelineAhead::Lookup::Ready) return ahead.Ready(key);
        std::lock_guard lock(mutex);
        ++counts.notReady;
        return nullptr;
    }
    auto module = std::make_shared<const TwinModule>(describe());
    // Gate off: PipelineAhead runs the factory here (today's inline creation, counted on the caller).
    const auto state = ahead.Request(key, factoryFor(key, std::move(module), build, std::this_thread::get_id()), {.urgent = true, .forgetOnFailure = false});
    if (state == PipelineAhead::Lookup::Ready) return ahead.Ready(key);
    std::lock_guard lock(mutex);
    ++counts.notReady;
    return nullptr;
}

void TwinPipelines::StartPrewarm(TwinBuilder build) {
    std::lock_guard lock(mutex);
    if (prewarmStarted) return;
    prewarmStarted = true;
    stopping.store(false, std::memory_order_release);
    prewarm = std::thread([this, build = std::move(build)] {
        std::size_t queued = 0;
        std::size_t skipped = 0;
        static_cast<void>(ahead.Prewarm([&](PipelineKey key) -> PipelineFactory {
            if (stopping.load(std::memory_order_acquire) || !TwinModuleHeaderValid(directory, sourceVersion, key)) {
                ++skipped;
                return {};
            }
            ++queued;
            return factoryFor(key, nullptr, build, std::thread::id{});
        }));
        std::lock_guard counted(mutex);
        counts.prewarmQueued += queued;
        counts.prewarmSkipped += skipped;
        if (queued + skipped != 0) std::fprintf(stderr, "[newpath] twin pipelines: %zu queued for creation ahead from %s, %zu manifest keys without a usable module file\n", queued, directory.string().c_str(), skipped);
    });
}

void TwinPipelines::Stop() {
    stopping.store(true, std::memory_order_release);
    std::thread joining;
    {
        std::lock_guard lock(mutex);
        joining = std::move(prewarm);
        prewarmStarted = false;
    }
    if (joining.joinable()) joining.join();
    ahead.Reset();
}

void TwinPipelines::NoteLayoutMismatch() {
    std::lock_guard lock(mutex);
    ++counts.layoutMismatches;
}

TwinPipelines::Counts TwinPipelines::Totals() const {
    std::lock_guard lock(mutex);
    return counts;
}

}
