#include "prx/libSceAgcDriver/Graphics/include/ResidencyConfig.hpp"
#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics {

namespace {

constexpr std::uint64_t MiB = std::uint64_t{1} << 20u;

std::uint64_t residentMiB(const char* text) {
    if (text == nullptr || *text == '\0') return 0;
    char* end = nullptr;
    errno = 0;
    const auto value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > (~std::uint64_t{0} >> 20u)) throw std::invalid_argument(std::string("APS5_NEW_PATH_RESIDENT_MIB is not a MiB count: ") + text);
    return value;
}

std::uint32_t chunkPages(const char* text) {
    char* end = nullptr;
    errno = 0;
    const auto value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 || value > 65535) throw std::invalid_argument(std::string("APS5_NEW_PATH_RESIDENT_CHUNK_PAGES is not a page count from 1 to 65535: ") + text);
    return static_cast<std::uint32_t>(value);
}

}

ResidencyConfig ResidencyConfigFrom(ResidencyEnvironment environment) {
    ResidencyConfig config;
    const char* master = environment("APS5_NEW_PATH");
    config.newPath = master != nullptr && *master != '\0' && std::string(master) != "0";
    config.residency = config.newPath && environment("APS5_NEW_PATH_NO_RESIDENCY") == nullptr;
    const char* budget = environment("APS5_NEW_PATH_RESIDENT_MIB");
    if (config.residency && budget != nullptr && *budget != '\0') {
        config.budgetForced = true;
        config.residentBytes = residentMiB(budget) * MiB;
    }
    const char* chunk = environment("APS5_NEW_PATH_RESIDENT_CHUNK_PAGES");
    if (config.residency && chunk != nullptr && *chunk != '\0') config.chunkPages = chunkPages(chunk);
    return config;
}

const ResidencyConfig& CurrentResidencyConfig() {
    static const ResidencyConfig config = ResidencyConfigFrom([](const char* name) -> const char* { return std::getenv(name); });
    return config;
}

}
