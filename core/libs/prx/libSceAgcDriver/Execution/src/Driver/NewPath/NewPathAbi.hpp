#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_NEWPATHABI_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_NEWPATHABI_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace AgcDriver::NewPath::Abi {

inline constexpr std::uint32_t Version = 1;
inline constexpr std::uint32_t PushBlockBytes = 256;
inline constexpr std::size_t StageCount = 4;
inline constexpr std::uint32_t PageShift = 16;
inline constexpr std::uint32_t SubpageShift = 12;
inline constexpr std::uint32_t FaultOutsideSegment = 16;
inline constexpr std::uint32_t FaultContinuationRounds = 17;

enum class Stage : std::uint8_t { Vertex, Hull, Geometry, Pixel, Compute = Vertex };

struct PushLayout {
    std::uint32_t driverWords = 0;
    std::uint32_t pageTableAddressWord = 0;
    std::array<std::uint32_t, StageCount> userWordOffset{};
    std::array<std::uint32_t, StageCount> userWordCount{};
    bool overflow = false;
    std::uint32_t overflowAddressWord = 0;
};

enum class StoreClass : std::uint8_t { None, Bounded, Unbounded };

enum class ChainTarget : std::uint8_t { BindlessTable, StoreDescriptor, SegmentDescriptor };

struct StaticChain {
    ChainTarget target = ChainTarget::BindlessTable;
    Stage stage = Stage::Vertex;
    bool isStatic = false;
    std::uint32_t userSgpr = 0;
    std::vector<std::uint32_t> loadOffsets;
    std::uint32_t descriptorDwords = 0;
    std::uint32_t site = 0;
};

struct DescriptorLayout {
    std::uint32_t heapSet = 0;
    std::uint32_t heapBinding = 0;
    bool pageTableInPushBlock = true;
};

struct ResolvedProgram {
    std::uint32_t version = Version;
    PushLayout push;
    DescriptorLayout descriptors;
    StoreClass stores = StoreClass::Unbounded;
    bool readsAnything = true;
    bool readSetComplete = false;
    bool transferKernel = false;
    std::vector<StaticChain> chains;
};

[[nodiscard]] constexpr bool FitsPushBlock(const PushLayout& layout) {
    std::uint32_t words = layout.driverWords;
    for (const auto count : layout.userWordCount) words += count;
    return words * 4u <= PushBlockBytes;
}

}

#endif
