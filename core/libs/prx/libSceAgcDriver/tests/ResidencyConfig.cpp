#include "prx/libSceAgcDriver/Graphics/include/ResidencyConfig.hpp"
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

namespace {

using AgcDriver::Graphics::ResidencyConfig;
using AgcDriver::Graphics::ResidencyConfigFrom;

std::map<std::string, std::string> variables;

const char* lookup(const char* name) {
    const auto found = variables.find(name);
    return found == variables.end() ? nullptr : found->second.c_str();
}

int failures = 0;

void check(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

ResidencyConfig with(std::map<std::string, std::string> values) {
    variables = std::move(values);
    return ResidencyConfigFrom(&lookup);
}

}

int main() {
    auto config = with({});
    check(!config.newPath && !config.residency && config.residentBytes == 0, "unset: today's path");

    config = with({{"APS5_NEW_PATH_NO_RESIDENCY", "1"}, {"APS5_NEW_PATH_RESIDENT_MIB", "64"}});
    check(!config.newPath && !config.residency && !config.budgetForced, "stage switches without the master change nothing");

    config = with({{"APS5_NEW_PATH", "0"}});
    check(!config.newPath && !config.residency, "APS5_NEW_PATH=0 is off");

    config = with({{"APS5_NEW_PATH", "1"}});
    check(config.newPath && config.residency && !config.budgetForced, "master on: residency on, budget from the partition");

    config = with({{"APS5_NEW_PATH", "1"}, {"APS5_NEW_PATH_NO_RESIDENCY", ""}});
    check(config.newPath && !config.residency && config.residentBytes == 0, "presence of APS5_NEW_PATH_NO_RESIDENCY turns residency off");

    config = with({{"APS5_NEW_PATH", "1"}, {"APS5_NEW_PATH_RESIDENT_MIB", "1024"}});
    check(config.residency && config.budgetForced && config.residentBytes == (std::uint64_t{1024} << 20u), "budget in MiB");

    config = with({{"APS5_NEW_PATH", "1"}, {"APS5_NEW_PATH_RESIDENT_MIB", "0"}});
    check(config.residency && config.budgetForced && config.residentBytes == 0, "budget 0 is a forced zero budget (matrix row), not the default");

    config = with({{"APS5_NEW_PATH", "1"}});
    check(!config.slotBinding, "slot binding is off unless asked for");
    config = with({{"APS5_NEW_PATH", "1"}, {"APS5_NEW_PATH_SLOT_BINDING", "1"}});
    check(config.slotBinding, "APS5_NEW_PATH_SLOT_BINDING turns slot binding on");
    config = with({{"APS5_NEW_PATH_SLOT_BINDING", "1"}});
    check(!config.slotBinding, "slot binding without the master changes nothing");
    check(!config.eagerCopyIn, "copy-in waits for a reader unless asked for");
    config = with({{"APS5_NEW_PATH", "1"}, {"APS5_NEW_PATH_EAGER_COPY_IN", "1"}});
    check(config.eagerCopyIn, "APS5_NEW_PATH_EAGER_COPY_IN copies in as soon as bytes change");
    config = with({{"APS5_NEW_PATH_EAGER_COPY_IN", "1"}});
    check(!config.eagerCopyIn, "eager copy-in without the master changes nothing");

    config = with({{"APS5_NEW_PATH", "1"}, {"APS5_NEW_PATH_RESIDENT_CHUNK_PAGES", "64"}});
    check(config.residency && config.chunkPages == 64, "chunk pages");

    config = with({{"APS5_NEW_PATH_RESIDENT_CHUNK_PAGES", "64"}});
    check(config.chunkPages == 0, "chunk pages without the master change nothing");

    for (const char* bad : {"0", "65536", "8k"}) {
        bool refused = false;
        try {
            static_cast<void>(with({{"APS5_NEW_PATH", "1"}, {"APS5_NEW_PATH_RESIDENT_CHUNK_PAGES", bad}}));
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused, "a chunk page count outside 1..65535 throws");
    }

    bool threw = false;
    try {
        static_cast<void>(with({{"APS5_NEW_PATH", "1"}, {"APS5_NEW_PATH_RESIDENT_MIB", "12x"}}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a malformed budget throws");

    if (failures != 0) return 1;
    std::printf("agc_residency_config: all checks passed\n");
    return 0;
}
