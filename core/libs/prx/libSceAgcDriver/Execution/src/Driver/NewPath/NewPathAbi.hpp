#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_NEWPATHABI_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_NEWPATHABI_HPP

#include "RuntimeAbi.hpp"
#include "BdaAbi.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace AgcDriver::NewPath::Abi {

inline constexpr std::uint32_t Version = 2;
inline constexpr std::uint32_t PushBlockBytes = 256;
inline constexpr std::size_t StageCount = 4;
inline constexpr std::uint32_t PageShift = 16;
inline constexpr std::uint32_t SubpageShift = 12;

using Stage = ShaderRecompiler::RuntimeAbi::Stage;

struct PushLayout {
    std::uint32_t driverWords = 0;
    std::uint32_t pageTableAddressWord = 0;
    std::uint32_t faultRecordAddressWord = 2;
    std::array<std::uint32_t, StageCount> userWordOffset{};
    std::array<std::uint32_t, StageCount> userWordCount{};
    bool overflow = false;
    std::uint32_t overflowAddressWord = 4;
};

enum class StoreClass : std::uint8_t { None, Bounded, Unbounded };

enum class ChainTarget : std::uint8_t { BindlessTable, StoreDescriptor, SegmentDescriptor };

struct StaticChain {
    ChainTarget target = ChainTarget::BindlessTable;
    Stage stage = Stage::Main;
    bool isStatic = false;
    std::uint32_t userSgpr = 0;
    std::vector<std::uint32_t> loadOffsets;
    std::uint32_t descriptorDwords = 4;
    std::uint32_t site = 0;
};

struct TransferKernelSite {
    enum class Kind : std::uint8_t { Fill, Copy };
    Kind kind = Kind::Fill;
    std::uint32_t components = 0;
    std::uint32_t destination = 0;
    std::uint32_t source = 0;
    std::array<std::uint32_t, 4> values{};
};

struct DescriptorLayout {
    std::uint32_t heapSet = 0;
    std::uint32_t heapBinding = 0;
};

struct ResolvedProgram {
    std::uint32_t version = Version;
    PushLayout push;
    DescriptorLayout descriptors;
    StoreClass stores = StoreClass::Unbounded;
    bool readsAnything = true;
    bool readSetComplete = false;
    std::vector<StaticChain> chains;
    std::vector<std::uint32_t> storeSites;
    std::vector<std::uint32_t> segmentSites;
    std::optional<TransferKernelSite> transferKernel;
};

[[nodiscard]] constexpr bool FitsPushBlock(const PushLayout& layout) {
    if (layout.pageTableAddressWord % 2u != 0u || layout.faultRecordAddressWord % 2u != 0u) return false;
    if (layout.overflow && layout.overflowAddressWord % 2u != 0u) return false;
    std::uint32_t words = layout.driverWords;
    for (const auto count : layout.userWordCount) words += count;
    return words * 4u <= PushBlockBytes;
}

inline constexpr auto FaultOutsideSegment = static_cast<std::uint32_t>(ShaderRecompiler::BdaAbi::FaultReason::BindlessRefused) + 1u;
inline constexpr auto FaultContinuationRounds = FaultOutsideSegment + 1u;

}

#endif
