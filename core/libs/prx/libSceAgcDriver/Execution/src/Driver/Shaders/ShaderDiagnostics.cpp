#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string_view>

namespace AgcDriver::DriverDetail {

void Driver::reportSkip(const char* kind, const std::string& what) {
    static std::mutex reportedMutex;
    static std::set<std::size_t> reported;
    static std::map<std::string, int> requestDumps;

    static const bool traceSkips = std::getenv("APS5_TRACE_SKIPS") != nullptr;
    static const int dumpLimit = [] { const char* text = std::getenv("APS5_SKIP_REQUEST_DUMPS"); return text ? std::atoi(text) : 8; }();
    static constexpr std::string_view marker = "\nRecompileRequest:\n";
    std::lock_guard lock(reportedMutex);
    std::string line;
    static bool announced = false;
    if (!announced) {
        announced = true;
        char text[160];
        std::snprintf(text, sizeof(text), "[gpu] skip reports: one write each, RecompileRequest dumps capped at %d per reason (APS5_SKIP_REQUEST_DUMPS)\n", dumpLimit);
        line = text;
    }
    const std::string prefix = "[gpu] skipped " + std::string(kind);
    if (reported.insert(std::hash<std::string>{}(what)).second) {
        const auto request = what.find(marker);
        if (request == std::string::npos) {
            line += prefix + ": " + what + "\n";
        } else {
            const auto reason = what.substr(0, request);
            const auto suffix = what.find(' ', request + marker.size());
            auto& dumps = requestDumps[reason];
            if (dumpLimit < 0 || dumps < dumpLimit) {
                ++dumps;
                line += prefix + ": " + what + "\n";
            } else {
                if (dumps == dumpLimit) {
                    ++dumps;
                    line += prefix + ": further RecompileRequest dumps for \"" + reason + "\" are left out\n";
                }
                line += prefix + ": " + reason + " (request left out)" + (suffix == std::string::npos ? std::string() : what.substr(suffix)) + "\n";
            }
        }
    } else if (traceSkips) {
        line += prefix + " again: " + what.substr(0, 100) + "\n";
    }
    if (!line.empty()) std::fwrite(line.data(), 1, line.size(), stderr);
}

std::string Driver::dumpRequest(std::uint64_t address, const ShaderRecompiler::RecompileRequest& request) {
    static std::mutex dumpMutex;
    static std::set<std::uint64_t> dumped;
    char name[64];
    std::snprintf(name, sizeof(name), "shader_%llx.req", static_cast<unsigned long long>(address));
    std::lock_guard lock(dumpMutex);
    if (!dumped.insert(address).second) return name;
    try {
        const auto text = ShaderRecompiler::RequestSerializer{}.Serialize(request);
        if (std::FILE* file = std::fopen(name, "wb")) {
            std::fwrite(text.data(), 1, text.size(), file);
            std::fclose(file);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[gpu] could not serialize request for 0x%llx: %s\n", static_cast<unsigned long long>(address), error.what());
    }
    return name;
}

bool CaptureTerminalComputeFailure(const ShaderRecompiler::RecompileRequest& request, std::span<const ShaderRecompiler::MemoryRegion> captured, const char* error, std::uint32_t queue, std::uint64_t serial, std::size_t packetOffset, const char* directory) noexcept {
    static std::atomic_flag attempted = ATOMIC_FLAG_INIT;
    if (std::getenv("APS5_DUMP_SHADERS") == nullptr || attempted.test_and_set(std::memory_order_relaxed)) return false;
    try {
        if (request.shader.stage != ShaderRecompiler::ShaderStage::Compute || request.graphics || request.context.pixel || request.context.vertex)
            throw std::runtime_error("terminal capture requires a compute request");
        std::size_t remaining = 64u * 1024u * 1024u - 4096u;
        const auto account = [&](std::size_t count, std::size_t width) {
            if (count > remaining / width) throw std::runtime_error("terminal request exceeds the 64 MiB input bound");
            remaining -= count * width;
        };
        account(request.shader.code.size(), sizeof(std::uint32_t));
        account(request.shader.header.size(), 1);
        account(request.context.userData.size(), sizeof(std::uint32_t));
        account(request.target.supportedCapabilities.size(), sizeof(std::uint32_t));
        account(request.target.supportedExtensions.size(), sizeof(std::uint64_t));
        for (const auto extension : request.target.supportedExtensions) account(extension.size(), 1);
        account(captured.size(), 2 * sizeof(std::uint64_t));
        for (const auto& region : captured) account(region.bytes.size(), 1);
        auto current = request;
        current.context.memory = captured;
        const auto text = ShaderRecompiler::RequestSerializer{}.Serialize(current);
        if (text.size() > 96u * 1024u * 1024u) throw std::runtime_error("terminal request exceeds the 96 MiB output bound");

        char name[112];
        std::snprintf(name, sizeof(name), "shader_%llx_terminal_%llx.req", static_cast<unsigned long long>(request.shader.codeAddress), static_cast<unsigned long long>(std::chrono::steady_clock::now().time_since_epoch().count()));
        const auto destination = std::filesystem::path(directory) / name;
        const auto destinationText = destination.string();
        const auto staging = std::filesystem::path(destinationText + ".tmp");
        const auto temporary = staging / "request";
        std::error_code status;
        if (!std::filesystem::create_directory(staging, status) || status) throw std::runtime_error("cannot reserve terminal capture staging directory");
        struct Cleanup {
            const std::filesystem::path& temporary;
            const std::filesystem::path& staging;
            std::FILE* file = nullptr;
            ~Cleanup() noexcept {
                if (file) std::fclose(file);
                try {
                    std::error_code ignored;
                    std::filesystem::remove(temporary, ignored);
                    std::filesystem::remove(staging, ignored);
                } catch (...) {}
            }
        } cleanup{temporary, staging};
        cleanup.file = std::fopen(temporary.string().c_str(), "wb");
        if (!cleanup.file) throw std::runtime_error("cannot open terminal capture staging file");
        const bool complete = std::fwrite(text.data(), 1, text.size(), cleanup.file) == text.size();
        const int closed = std::fclose(cleanup.file);
        cleanup.file = nullptr;
        if (!complete || closed != 0) throw std::runtime_error("cannot finish terminal capture staging file");
        std::filesystem::create_hard_link(temporary, destination, status);
        if (status) throw std::runtime_error("cannot publish terminal capture without replacing a file");
        std::fprintf(stderr, "[gpu] terminal compute request: %s (q=%u serial=%llu packet=%zu regions=%zu)\n[gpu] terminal compute error: %.1024s\n", destinationText.c_str(), queue, static_cast<unsigned long long>(serial), packetOffset, captured.size(), error);
        return true;
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "[gpu] terminal compute capture unavailable: %.256s; original: %.1024s\n", failure.what(), error);
    } catch (...) {
        std::fprintf(stderr, "[gpu] terminal compute capture unavailable; original: %.1024s\n", error);
    }
    return false;
}

}
