#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandRecord.hpp"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <sstream>

namespace AgcDriver::NewPath {

namespace Bda = ShaderRecompiler::BdaAbi;

namespace {

std::mutex observerMutex;
RecordObserver observer;

}

void SetCommandRecordObserver(RecordObserver next) {
    std::lock_guard lock(observerMutex);
    observer = std::move(next);
}

void ObserveCommandRecord(std::span<const std::uint32_t, CommandRecordWords> words) {
    std::lock_guard lock(observerMutex);
    if (observer) observer(words);
}

void ClearCommandRecord(std::span<std::uint32_t, CommandRecordWords> words) {
    std::fill(words.begin(), words.begin() + Bda::WrittenListWord, 0u);
    words[Bda::SrtGuardWord] = 0u;
}

bool CommandRecordClear(std::span<const std::uint32_t, CommandRecordWords> words) {
    return std::all_of(words.begin(), words.begin() + Bda::WrittenListWord, [](std::uint32_t word) { return word == 0u; }) && words[Bda::SrtGuardWord] == 0u;
}

RecordOutcome CompleteCommandRecord(std::span<std::uint32_t, CommandRecordWords> words, std::span<const AddressRange> leased, const MarkWritten& mark) {
    RecordOutcome outcome;
    Bda::Fault report{};
    std::memcpy(&report, words.data(), sizeof(report));
    if (report.state != Bda::FaultState::Empty) {
        outcome.faulted = true;
        outcome.reason = report.reason;
        std::ostringstream message;
        message << "new-path command fault: reason " << static_cast<std::uint32_t>(report.reason) << " at 0x" << std::hex << report.address << std::dec << " (" << report.bytes << " bytes, stage " << report.stage << ", pc 0x" << std::hex << report.instruction << ")";
        outcome.message = message.str();
    } else if (const auto guard = words[Bda::SrtGuardWord]; guard != 0u) {
        outcome.faulted = true;
        outcome.reason = Bda::FaultReason::Unmapped;
        std::ostringstream message;
        message << "new-path command SRT guard: a guarded SRT read of stage " << (guard >> Bda::SrtGuardStageShift) << " faulted (poison record " << (guard & ((1u << Bda::SrtGuardStageShift) - 1u)) << ")";
        outcome.message = message.str();
    }
    if (words[Bda::WrittenOverflowWord] != 0u) {
        outcome.overflow = true;
        for (const auto& [begin, end] : leased) {
            mark(begin, static_cast<std::size_t>(end - begin));
            ++outcome.writtenRanges;
            outcome.writtenBytes += end - begin;
        }
    } else if (!Bda::ForEachWrittenRange(words.data(), [&](std::uint64_t begin, std::size_t bytes) {
                   mark(begin, bytes);
                   ++outcome.writtenRanges;
                   outcome.writtenBytes += bytes;
               })) {
        outcome.invalid = true;
        for (const auto& [begin, end] : leased) mark(begin, static_cast<std::size_t>(end - begin));
    }
    ClearCommandRecord(words);
    return outcome;
}

}
