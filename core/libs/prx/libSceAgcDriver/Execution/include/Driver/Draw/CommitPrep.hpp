#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_COMMITPREP_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_COMMITPREP_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace AgcDriver::DriverDetail::CommitPrep {

using Range = std::pair<std::uint64_t, std::uint64_t>;

struct Stamps {
    static constexpr std::size_t MaxReads = 8;
    bool taken = false;
    const void* device = nullptr;
    std::uint64_t forgetSerial = 0;
    std::uint64_t registryGeneration = 0;
    std::uint64_t epoch = 0;
    std::uint64_t proofSerial = 0;
    std::uint64_t pendingSerial = 0;
    std::size_t reads = 0;
    std::array<Range, MaxReads> ranges{};
    std::array<std::uint64_t, MaxReads> cpu{};
    std::uint32_t gpuPending = 0;
};

Stamps Take(const void* device, std::span<const Range> reads);
Stamps Retake(const Stamps& then, const void* device);
std::uint32_t Moved(const Stamps& then, const Stamps& now);

void SetCommitStamps(const Stamps* stamps);
const Stamps* CommitStamps();

}

#endif
