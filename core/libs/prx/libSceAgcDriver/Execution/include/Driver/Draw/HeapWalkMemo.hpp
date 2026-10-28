#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_HEAPWALKMEMO_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_HEAPWALKMEMO_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/HeapStages.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

struct HeapWalkEntry {
    std::uint64_t variant = 0;
    std::uint64_t shaderBase = 0;
    std::uint32_t pushOffset = 0;
    std::vector<std::uint32_t> userData;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::vector<std::uint32_t> values;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> accessible;
    ShaderRecompiler::SrtUserDataUse userDataUse;
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
    std::shared_ptr<const ShaderRecompiler::HeapStageWords> words;
};

constexpr std::size_t HeapWalkWays = 4;

constexpr std::size_t HeapWalkMemoKeys = 16384;

constexpr std::size_t HeapWalkDrain = 2;

constexpr std::size_t HeapWalkSeenKeys = 1u << 16;

struct HeapWalkMemo {
    std::unordered_map<std::uint64_t, std::vector<HeapWalkEntry>> current;
    std::unordered_map<std::uint64_t, std::vector<HeapWalkEntry>> previous;
    std::vector<std::uint64_t> seen = std::vector<std::uint64_t>(HeapWalkSeenKeys, 0);

    bool admit(std::uint64_t key) {
        auto& last = seen[(key ^ (key >> 29)) & (HeapWalkSeenKeys - 1u)];
        if (last == key) return true;
        last = key;
        return false;
    }

    std::vector<HeapWalkEntry>* find(std::uint64_t key) {
        if (const auto found = current.find(key); found != current.end()) return &found->second;
        const auto old = previous.find(key);
        if (old == previous.end()) return nullptr;
        auto moved = std::move(old->second);
        previous.erase(old);
        auto& ways = slot(key);
        ways = std::move(moved);
        return &ways;
    }

    std::vector<HeapWalkEntry>& slot(std::uint64_t key) {
        for (std::size_t drained = 0; drained < HeapWalkDrain && !previous.empty(); ++drained) previous.erase(previous.begin());
        if (current.size() >= HeapWalkMemoKeys && !current.contains(key)) {
            previous = std::move(current);
            current.clear();
        }
        return current[key];
    }
};

}

#endif
