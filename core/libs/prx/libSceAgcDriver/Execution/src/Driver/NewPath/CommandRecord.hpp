#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_COMMANDRECORD_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_COMMANDRECORD_HPP

#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandWrites.hpp"
#include "BdaAbi.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>

namespace AgcDriver::NewPath {

inline constexpr std::size_t CommandRecordWords = ShaderRecompiler::BdaAbi::FaultBufferBytes / sizeof(std::uint32_t);

using MarkWritten = std::function<void(std::uint64_t address, std::size_t bytes)>;

struct RecordOutcome {
    std::uint64_t writtenRanges = 0;
    std::uint64_t writtenBytes = 0;
    bool overflow = false;
    bool invalid = false;
    bool faulted = false;
    ShaderRecompiler::BdaAbi::FaultReason reason{};
    std::string message;
};

void ClearCommandRecord(std::span<std::uint32_t, CommandRecordWords> words);
[[nodiscard]] bool CommandRecordClear(std::span<const std::uint32_t, CommandRecordWords> words);
RecordOutcome CompleteCommandRecord(std::span<std::uint32_t, CommandRecordWords> words, std::span<const AddressRange> leased, const MarkWritten& mark);

}

#endif
