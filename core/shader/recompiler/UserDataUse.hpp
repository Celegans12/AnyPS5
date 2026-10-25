#ifndef CORE_SHADER_RECOMPILER_USERDATAUSE_HPP
#define CORE_SHADER_RECOMPILER_USERDATAUSE_HPP

#include <cstdint>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

struct SrtUserDataUse {
    static constexpr std::uint32_t TrackedWords = 64;
    std::uint64_t asBase = 0;
    std::uint64_t asValue = 0;
    std::vector<std::pair<std::uint64_t, std::int32_t>> reads;
    // The tape walk's nested bases: (read address, address of the word whose loaded value is the
    // read's base low half) for reads based directly on a load used only as a read base or a flat
    // slot; `pointers` the addresses such loads read (once per read); `pointerSlots` the flat slots
    // such loads fill (slot, address); `untracked` the reads without a user-word origin made by an
    // evaluator that does not track nested bases.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> nested;
    std::vector<std::uint64_t> pointers;
    std::vector<std::pair<std::uint32_t, std::uint64_t>> pointerSlots;
    std::uint32_t untracked = 0;
};

}

#endif
