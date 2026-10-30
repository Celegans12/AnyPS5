#pragma once

#include <cstdint>

namespace AgcDriver::Graphics {

struct ResidencyConfig {
    bool newPath = false;
    bool residency = false;
    bool budgetForced = false;
    std::uint64_t residentBytes = 0;
    std::uint32_t chunkPages = 0;
    bool slotBinding = false;
    bool eagerCopyIn = false;
};

using ResidencyEnvironment = const char* (*)(const char* name);

ResidencyConfig ResidencyConfigFrom(ResidencyEnvironment environment);
const ResidencyConfig& CurrentResidencyConfig();

}
