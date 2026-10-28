#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_COMMANDWRITES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_COMMANDWRITES_HPP

#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/NewPathAbi.hpp"
#include "BdaAbi.hpp"
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::NewPath {

using AddressRange = std::pair<std::uint64_t, std::uint64_t>;

struct WriteSet {
    bool unbounded = false;
    bool outsideTable = false;
    std::vector<AddressRange> ranges;

    [[nodiscard]] std::uint64_t Bytes() const;
};

[[nodiscard]] std::vector<AddressRange> MergeRanges(std::vector<AddressRange> ranges);
[[nodiscard]] std::vector<AddressRange> WritableRanges(std::span<const ShaderRecompiler::BdaAbi::Range> table);
[[nodiscard]] WriteSet ResolveWrites(std::span<const ShaderRecompiler::BdaAbi::Range> table, Abi::StoreClass stores, std::span<const AddressRange> storeRanges);

}

#endif
