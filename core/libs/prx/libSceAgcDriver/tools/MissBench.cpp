// agc_miss_bench [--iterations N] [--bindless] <shader.req>...: local bench of the draw path's per-draw miss cost.
// Each request's memory regions are mapped at their guest addresses, the shader is prepared once, and then every
// iteration runs what Driver::materializeDrawStage does on a draw-cache miss: a fresh ShaderMemory, Capture of the
// prepared invocation (SRT walk + image table snapshot over live memory) and Materialize (bindings, image tables,
// bindless registrations). Prints per-phase times and a fingerprint of the encoded result.
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Recompiler.hpp"
#include "ShaderDiskCache.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {

std::string ReadText(const char* path) {
    std::ifstream file(path, std::ios::binary);
    std::ostringstream stream;
    stream << file.rdbuf();
    auto text = stream.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
    return text;
}

std::uint64_t Fingerprint(const ShaderRecompiler::RecompileResult& result) {
    std::vector<std::byte> bytes;
    ShaderRecompiler::ShaderDiskCache::EncodeResult(result, bytes);
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const auto byte : bytes) hash = (hash ^ static_cast<std::uint8_t>(byte)) * 0x100000001b3ull;
    for (const auto word : result.shaderDataUserWords) hash = (hash ^ word) * 0x100000001b3ull;
    hash = (hash ^ result.specializationId) * 0x100000001b3ull;
    return hash;
}

struct Mapping {
    std::vector<void*> blocks;
    ~Mapping() {
#ifdef _WIN32
        for (auto* block : blocks) VirtualFree(block, 0, MEM_RELEASE);
#endif
    }
};

bool MapRegions(const std::vector<ShaderRecompiler::MemoryRegion>& regions, Mapping& mapping) {
#ifdef _WIN32
    constexpr std::uint64_t Page = 0x10000;
    std::set<std::uint64_t> pages;
    for (const auto& region : regions) {
        if (region.bytes.empty()) continue;
        for (auto page = region.guestAddress & ~(Page - 1u); page < region.guestAddress + region.bytes.size(); page += Page) pages.insert(page);
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
    for (const auto page : pages) {
        if (!runs.empty() && runs.back().second == page) runs.back().second += Page;
        else runs.emplace_back(page, page + Page);
    }
    for (const auto& [begin, end] : runs) {
        void* block = VirtualAlloc(reinterpret_cast<void*>(begin), static_cast<SIZE_T>(end - begin), MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
        if (block == nullptr) {
            using VirtualAlloc2Function = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void*, ULONG);
            static const auto virtualAlloc2 = reinterpret_cast<VirtualAlloc2Function>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernelbase.dll"), "VirtualAlloc2")));
            MEMORY_BASIC_INFORMATION info{};
            if (virtualAlloc2 != nullptr && VirtualQuery(reinterpret_cast<void*>(begin), &info, sizeof(info)) != 0 && info.State == MEM_RESERVE) {
                constexpr ULONG PreservePlaceholder = 0x00000002;
                constexpr ULONG ReplacePlaceholder = 0x00004000;
                VirtualFree(reinterpret_cast<void*>(begin), static_cast<SIZE_T>(end - begin), MEM_RELEASE | PreservePlaceholder);
                block = virtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(begin), static_cast<SIZE_T>(end - begin), MEM_RESERVE | MEM_COMMIT | ReplacePlaceholder | MEM_WRITE_WATCH, PAGE_READWRITE, nullptr, 0);
                if (block == reinterpret_cast<void*>(begin)) {
                    mapping.blocks.push_back(block);
                    continue;
                }
            }
        }
        if (block == nullptr) {
            bool committed = true;
            for (auto page = begin; page < end && committed; page += Page) {
                MEMORY_BASIC_INFORMATION info{};
                committed = VirtualQuery(reinterpret_cast<void*>(page), &info, sizeof(info)) != 0 && (info.State == MEM_COMMIT || VirtualAlloc(reinterpret_cast<void*>(page), Page, MEM_COMMIT, PAGE_READWRITE) != nullptr);
                DWORD old = 0;
                if (committed) committed = VirtualProtect(reinterpret_cast<void*>(page), Page, PAGE_READWRITE, &old) != 0;
            }
            if (committed) continue;
        }
        if (block != reinterpret_cast<void*>(begin)) {
            const auto error = GetLastError();
            MEMORY_BASIC_INFORMATION info{};
            VirtualQuery(reinterpret_cast<void*>(begin), &info, sizeof(info));
            std::printf("  cannot map 0x%llx+0x%llx (error %lu; state %lx type %lx allocation base %p protect %lx region 0x%zx)\n", static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end - begin), error, info.State, info.Type, info.AllocationBase, info.Protect, static_cast<std::size_t>(info.RegionSize));
            if (block != nullptr) VirtualFree(block, 0, MEM_RELEASE);
            return false;
        }
        mapping.blocks.push_back(block);
    }
    for (const auto& region : regions) {
        if (!region.bytes.empty()) std::memcpy(reinterpret_cast<void*>(region.guestAddress), region.bytes.data(), region.bytes.size());
    }
    return true;
#else
    static_cast<void>(regions);
    static_cast<void>(mapping);
    return false;
#endif
}

double Micros(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double, std::micro>(duration).count();
}

double Median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

bool Bench(const char* path, int iterations, bool bindless) {
    auto parsed = ShaderRecompiler::RequestSerializer{}.Deserialize(ReadText(path));
    auto& request = parsed.request;
    if (bindless) request.target.bindlessImages = true;
    if (std::getenv("APS5_SPANNED_WAVE") != nullptr) request.target.spannedWaves = true;
    std::vector<ShaderRecompiler::MemoryRegion> regions(request.context.memory.begin(), request.context.memory.end());
    Mapping mapping;
    if (!MapRegions(regions, mapping)) return false;
    try {
        const auto handle = ShaderRecompiler::PrepareShader(request);
        auto invocationRequest = request;
        invocationRequest.shader.code = ShaderRecompiler::GetPreparedCode(*handle);
        invocationRequest.context.memory = {};
        const auto invocation = ShaderRecompiler::PreparedShaderInvocation::TryCreate(invocationRequest, handle);
        if (!invocation) throw std::runtime_error("the prepared shader does not match its invocation");
        std::vector<double> capture, materialize;
        std::uint64_t first = 0;
        bool stable = true;
        std::size_t tables = 0, keys = 0, words = 0, registrations = 0, mapWords = 0;
        bool heap = false;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            AgcDriver::ShaderMemory memory({});
            const auto started = std::chrono::steady_clock::now();
            const auto captured = memory.Capture(*invocation);
            const auto between = std::chrono::steady_clock::now();
            const auto result = invocation->Materialize(*captured);
            const auto ended = std::chrono::steady_clock::now();
            capture.push_back(Micros(between - started));
            materialize.push_back(Micros(ended - between));
            if (iteration != 0 && iteration + 1 != iterations && iteration % 64 != 0) continue;
            const auto fingerprint = Fingerprint(*result);
            if (iteration == 0) {
                first = fingerprint;
                tables = captured->snapshot.tables.tables.size();
                for (const auto& table : captured->snapshot.tables.tables) {
                    keys += table.keys;
                    words += table.words.size();
                }
                registrations = result->bindlessTables.size();
                heap = result->bindlessHeap;
                for (const auto& binding : result->bindings) {
                    if (binding.role == ShaderRecompiler::DescriptorRole::ImageTableMap) mapWords += binding.guestDescriptor.size();
                }
            } else if (fingerprint != first) {
                stable = false;
            }
        }
        const auto firstCapture = capture.front(), firstMaterialize = materialize.front();
        if (capture.size() > 1) {
            capture.erase(capture.begin());
            materialize.erase(materialize.begin());
        }
        std::printf("BENCH %s stage %u tables %zu keys %zu words %zu map %zu heap %d registrations %zu | first %.1f + %.1f us | median capture %.1f us materialize %.1f us total %.1f us | fingerprint %016llx%s\n",
                    path, static_cast<unsigned>(request.shader.stage), tables, keys, words, mapWords, heap ? 1 : 0, registrations, firstCapture, firstMaterialize, Median(capture), Median(materialize), Median(capture) + Median(materialize),
                    static_cast<unsigned long long>(first), stable ? "" : " UNSTABLE");
        std::fflush(stdout);
        return true;
    } catch (const std::exception& error) {
        std::string text = error.what();
        if (const auto at = text.find('\n'); at != std::string::npos) text.resize(at);
        std::printf("FAIL %s: %s\n", path, text.c_str());
        return false;
    }
}

}

int main(int argc, char** argv) {
    int iterations = 200;
    bool bindless = false;
    std::vector<const char*> paths;
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        if (option == "--iterations" && i + 1 < argc) iterations = std::max(1, std::atoi(argv[++i]));
        else if (option == "--bindless") bindless = true;
        else paths.push_back(argv[i]);
    }
    if (const char* value = std::getenv("APS5_BINDLESS_IMAGES"); value != nullptr && std::string(value) == "1") bindless = true;
    if (paths.empty()) {
        std::fprintf(stderr, "usage: agc_miss_bench [--iterations N] [--bindless] <shader.req>...\n");
        return 2;
    }
    int failures = 0;
    for (const auto* path : paths) failures += Bench(path, iterations, bindless) ? 0 : 1;
    return failures == 0 ? 0 : 1;
}
