#ifndef CORE_LIBS_PRX_LIBSCEPAD_PADSCRIPT_HPP
#define CORE_LIBS_PRX_LIBSCEPAD_PADSCRIPT_HPP

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace Pad {

struct ScriptPress {
    std::uint64_t startUs = 0;
    std::uint64_t everyUs = 0;
    std::uint64_t holdUs = 0;
    std::uint32_t buttons = 0;
};

class PadScript {
public:
    static constexpr unsigned MinReads = 2;
    static constexpr std::uint64_t DefaultHoldMs = 200;

    static PadScript Parse(std::string_view presses, std::string_view repeats);
    static PadScript FromEnvironment();

    bool Empty() const { return entries.empty(); }
    std::uint32_t Sample(std::uint64_t nowUs);
    std::string Describe() const;

private:
    struct Entry {
        ScriptPress press;
        bool holding = false;
        std::uint64_t cycle = 0;
        std::uint64_t nextCycle = 0;
        unsigned reads = 0;
    };
    std::vector<Entry> entries;
};

std::uint32_t ParseScriptButtons(std::string_view text);
std::string ScriptButtonNames(std::uint32_t buttons);

enum class TracedCall : std::uint8_t { Read, ReadState, ControllerInformation, ExtControllerInformation };

bool TraceEnabled();
std::uint64_t MicrosSinceProcessStart();
double ProcessSeconds();
void TraceCall(TracedCall call, const void* caller, std::uint32_t buttons);

}

#endif
