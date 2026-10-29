#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PipelineAheadConfig.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/NewPath.hpp"
#include "ShaderCacheDirectory.hpp"
#include "ShaderDiskCache.hpp"
#include <cstdio>
#include <cstdlib>

namespace AgcDriver::NewPath {

bool PipelineGateEnabled() {
    static const bool enabled = Enabled() && std::getenv("APS5_NEW_PATH_NO_PIPELINE_GATE") == nullptr;
    return enabled;
}

std::filesystem::path DefaultManifestPath() {
    constexpr const char* name = "newpath-pipelines.txt";
    if (const char* own = std::getenv("APS5_PIPELINE_CACHE_DIR"); own != nullptr && own[0] != '\0') return std::filesystem::path(own) / name;
    const auto directory = ShaderRecompiler::ShaderCacheDirectory();
    if (directory.empty()) return {};
    char version[17];
    std::snprintf(version, sizeof(version), "%016llx", static_cast<unsigned long long>(ShaderRecompiler::ShaderDiskCache::SourceVersion()));
    return directory / version / name;
}

PipelineAhead& Pipelines() {
    static auto* pipelines = [] {
        std::shared_ptr<PipelineManifest> manifest;
        if (Enabled()) manifest = std::make_shared<PipelineManifest>(DefaultManifestPath(), ShaderRecompiler::ShaderDiskCache::SourceVersion());
        return new PipelineAhead(PipelineAhead::Settings{PipelineGateEnabled(), 2}, std::move(manifest));
    }();
    return *pipelines;
}

TwinPipelines& TwinPipelineStore() {
    static auto* twins = [] {
        const auto manifest = DefaultManifestPath();
        return new TwinPipelines(Pipelines(), manifest.empty() ? std::filesystem::path{} : manifest.parent_path() / "newpath-twins", ShaderRecompiler::ShaderDiskCache::SourceVersion());
    }();
    return *twins;
}

}
