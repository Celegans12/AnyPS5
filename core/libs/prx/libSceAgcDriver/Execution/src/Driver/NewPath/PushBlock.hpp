#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_PUSHBLOCK_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_PUSHBLOCK_HPP

#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/NewPathAbi.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace AgcDriver::NewPath {

using PushBytes = std::array<std::byte, Abi::PushBlockBytes>;

struct PushAddresses {
    std::uint64_t pageTable = 0;
    std::uint64_t faultRecord = 0;
    std::uint64_t overflow = 0;
};

[[nodiscard]] bool PushLayoutValid(const Abi::PushLayout& layout);
[[nodiscard]] std::optional<PushBytes> BuildPushBlock(const Abi::PushLayout& layout, std::span<const std::span<const std::uint32_t>, Abi::StageCount> userWords, const PushAddresses& addresses);

}

#endif
