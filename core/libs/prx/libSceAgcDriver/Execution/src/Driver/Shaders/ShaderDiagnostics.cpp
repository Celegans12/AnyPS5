#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/PipelineFingerprint.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderPreparation.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace AgcDriver::DriverDetail {

namespace {

std::optional<std::set<std::uint64_t>> selectedPrograms() {
    const char* value = std::getenv("APS5_DUMP_SHADERS");
    if (value == nullptr || std::string_view(value) == "1") return std::nullopt;
    std::set<std::uint64_t> programs;
    std::string_view rest(value);
    while (!rest.empty()) {
        const auto comma = rest.find(',');
        auto token = rest.substr(0, comma);
        rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
        if (token.starts_with("0x") || token.starts_with("0X")) token.remove_prefix(2);
        std::uint64_t address = 0;
        const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), address, 16);
        if (token.empty() || error != std::errc{} || end != token.data() + token.size()) return std::nullopt;
        programs.insert(address);
    }
    if (programs.empty()) return std::nullopt;
    return programs;
}

}

void Driver::reportSkip(const char* kind, const std::string& what) {
    static AgcDriver::Mutex reportedMutex;
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
        if (dumpLimit < 0) {
            std::snprintf(text, sizeof(text), "[gpu] skip reports: one write each, RecompileRequest dumps uncapped (APS5_SKIP_REQUEST_DUMPS)\n");
        } else {
            std::snprintf(text, sizeof(text), "[gpu] skip reports: one write each, RecompileRequest dumps capped at %d per reason (APS5_SKIP_REQUEST_DUMPS)\n", dumpLimit);
        }
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

bool SkipRefusedShaders() {
    static const bool skip = [] { const char* text = std::getenv("APS5_SKIP_REFUSED_SHADERS"); return text != nullptr && std::atoi(text) != 0; }();
    return skip;
}

bool SkipRefusedGraphicsRecompile(std::uint32_t stage, const char* what) {
    static const bool strict = [] { const char* text = std::getenv("APS5_STRICT_SHADER_REFUSALS"); return text != nullptr && std::atoi(text) != 0; }();
    return !strict && stage != static_cast<std::uint32_t>(ShaderRecompiler::ShaderStage::Compute) && what != nullptr && std::strstr(what, "ShaderRecompiler::Recompile: ") != nullptr;
}

bool SkipListedShader(std::uint64_t address) {
    static constexpr char variable[] = "APS5" "_SKIP_SHADER_LIST";
    static const std::vector<std::uint64_t> listed = [] {
        std::vector<std::uint64_t> addresses;
        const char* text = std::getenv(variable);
        if (text == nullptr) return addresses;
        const std::string list(text);
        std::size_t begin = 0;
        while (begin < list.size()) {
            const auto end = std::min(list.find(',', begin), list.size());
            if (end > begin) addresses.push_back(std::strtoull(list.substr(begin, end - begin).c_str(), nullptr, 16));
            begin = end + 1;
        }
        return addresses;
    }();
    return !listed.empty() && std::find(listed.begin(), listed.end(), address) != listed.end();
}

std::string RefusedShaderSubject(std::uint32_t stage, std::uint64_t address) {
    static constexpr const char* stageNames[] = {"compute", "vertex", "hull", "domain", "geometry", "pixel", "local", "mesh"};
    char text[64];
    std::snprintf(text, sizeof(text), "%s shader 0x%llx", stage < std::size(stageNames) ? stageNames[stage] : "graphics", static_cast<unsigned long long>(address));
    return text;
}

void Driver::SkipRefusedPreparation(const ShaderRefusal& refusal) {
    ShaderPreparationTransaction::ContinueAfterRefusal();
    std::string reason = refusal.what();
    if (const auto newline = reason.find('\n'); newline != std::string::npos) reason.resize(newline);
    reportSkip("shader preparation", (refusal.subject.empty() ? std::string("shader") : refusal.subject) + ": " + reason);
}

std::string Driver::dumpRequest(std::uint64_t address, const ShaderRecompiler::RecompileRequest& request, std::string_view suffix) {
    static AgcDriver::Mutex dumpMutex;
    static std::set<std::string> dumped;
    char hex[32];
    std::snprintf(hex, sizeof(hex), "%llx", static_cast<unsigned long long>(address));
    const std::string name = "shader_" + std::string(hex) + std::string(suffix) + ".req";
    if (const auto selected = selectedPrograms(); selected.has_value() && !selected->contains(address)) return {};
    std::lock_guard lock(dumpMutex);
    if (!dumped.insert(name).second) return name;
    try {
        const auto text = ShaderRecompiler::RequestSerializer{}.Serialize(request);
        if (std::FILE* file = std::fopen(name.c_str(), "wb")) {
            const auto written = std::fwrite(text.data(), 1, text.size(), file);
            const auto closed = std::fclose(file);
            if (written != text.size() || closed != 0) {
                dumped.erase(name);
                std::fprintf(stderr, "[gpu] incomplete write to dump file %s (%zu of %zu bytes, close status %d)\n", name.c_str(), written, text.size(), closed);
            }
        } else {
            dumped.erase(name);
            std::fprintf(stderr, "[gpu] could not open dump file %s for writing\n", name.c_str());
        }
    } catch (const std::exception& error) {
        dumped.erase(name);
        std::fprintf(stderr, "[gpu] could not serialize request for 0x%llx: %s\n", static_cast<unsigned long long>(address), error.what());
    }
    return name;
}

namespace {
const char* PipelineRequestDirectory() {
    static const char* directory = [] {
        const char* value = std::getenv("APS5_PIPELINE_REQUESTS");
        return value != nullptr && *value != '\0' ? value : nullptr;
    }();
    return directory;
}
}

bool Driver::pipelineRequestsEnabled() {
    return PipelineRequestDirectory() != nullptr;
}

void Driver::capturePipelineRequest(std::uint64_t address, const ShaderRecompiler::RecompileRequest& request, const ShaderRecompiler::RecompileResult& result, std::span<const ShaderMemory::Probe> probes, VulkanDevice& device) noexcept {
    const char* directory = PipelineRequestDirectory();
    if (directory == nullptr || request.shader.stage != ShaderRecompiler::ShaderStage::Compute || request.graphics.has_value()) return;
    static const std::size_t minimum = [] {
        const char* value = std::getenv("APS5_PIPELINE_REQUESTS_MIN_WORDS");
        return value != nullptr && *value != '\0' ? static_cast<std::size_t>(std::strtoull(value, nullptr, 10)) : std::size_t{500000};
    }();
    // Precheck (APS5_NO_PIPELINE_REQUEST_PRECHECK=1 off): the inputs of every fingerprint already in `saved`,
    // by pipeline variant. A materialization whose inputs match is saved already (its fingerprint would be the
    // one recorded, and `saved` only grows), so it returns before the layout plan and the two SPIR-V hashes.
    static const bool precheck = std::getenv("APS5_NO_PIPELINE_REQUEST_PRECHECK") == nullptr;
    static constexpr std::size_t PrecheckEntries = 1024;
    static std::mutex savedMutex;
    static std::set<std::uint64_t> saved;
    static std::map<std::uint64_t, ComputePipelineInputs> savedInputs;
    try {
        const auto words = result.spirv.size();
        if (words < minimum) return;
        const auto variant = result.PipelineVariantId();
        if (precheck) {
            std::lock_guard lock(savedMutex);
            if (const auto found = savedInputs.find(variant); found != savedInputs.end() && found->second.Matches(result)) return;
        }
        const auto layout = device.PlannedComputeLayoutKey(result);
        const auto components = ComputePipelineFingerprintComponents(result, layout);
        const auto fingerprint = components.whole;
        {
            std::lock_guard lock(savedMutex);
            const bool inserted = saved.insert(fingerprint).second;
            if (precheck) {
                if (savedInputs.size() >= PrecheckEntries && !savedInputs.contains(variant)) savedInputs.clear();
                savedInputs.insert_or_assign(variant, ComputePipelineInputs::Of(result));
            }
            if (!inserted) return;
        }
        char stem[96];
        std::snprintf(stem, sizeof(stem), "c_%07zu_%llx_%016llx", words, static_cast<unsigned long long>(address), static_cast<unsigned long long>(fingerprint));
        const std::filesystem::path folder(directory);
        std::filesystem::create_directories(folder);
        const auto target = folder / (std::string(stem) + ".req");
        if (std::filesystem::exists(target)) return;
        const auto write = [](const std::filesystem::path& path, std::string_view text) {
            const auto temporary = std::filesystem::path(path.string() + ".tmp");
            {
                std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
                file.write(text.data(), static_cast<std::streamsize>(text.size()));
                if (!file) throw std::runtime_error("cannot write " + temporary.string());
            }
            std::filesystem::rename(temporary, path);
        };
        char sidecar[192];
        std::snprintf(sidecar, sizeof(sidecar), "fingerprint %016llx\nwords %zu\naddress 0x%llx\nvariant %016llx\n", static_cast<unsigned long long>(fingerprint), words, static_cast<unsigned long long>(address), static_cast<unsigned long long>(result.PipelineVariantId()));
        std::string text = std::string(sidecar) + ComputePipelineDescription(result, components, layout);
        std::set<std::tuple<bool, std::uint64_t, std::uint64_t>> asked;
        for (const auto& probe : probes) {
            if (!asked.insert({probe.pending, probe.address, probe.bytes}).second) continue;
            char line[96];
            std::snprintf(line, sizeof(line), "probe %c %llx %llx %u\n", probe.pending ? 'p' : 'a', static_cast<unsigned long long>(probe.address), static_cast<unsigned long long>(probe.bytes), probe.answer ? 1u : 0u);
            text += line;
        }
        write(folder / (std::string(stem) + ".fp"), text);
        write(target, ShaderRecompiler::RequestSerializer{}.Serialize(request));
        std::fprintf(stderr, "[pipeline-requests] saved %s.req (%zu SPIR-V words, program 0x%llx)\n", stem, words, static_cast<unsigned long long>(address));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[pipeline-requests] could not save the request of program 0x%llx: %s\n", static_cast<unsigned long long>(address), error.what());
    } catch (...) {
    }
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
