#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PushBlock.hpp"
#include <cstring>
#include <vector>

namespace AgcDriver::NewPath {

namespace {

constexpr std::uint32_t BlockWords = Abi::PushBlockBytes / 4u;

void putWords(PushBytes& bytes, std::uint32_t word, const void* source, std::size_t words) {
    std::memcpy(bytes.data() + static_cast<std::size_t>(word) * 4u, source, words * 4u);
}

}

bool PushLayoutValid(const Abi::PushLayout& layout) {
    if (!Abi::FitsPushBlock(layout)) return false;
    std::vector<bool> used(BlockWords, false);
    const auto claim = [&](std::uint32_t first, std::uint32_t count) {
        if (first > BlockWords || count > BlockWords - first) return false;
        for (auto word = first; word < first + count; ++word) {
            if (used[word]) return false;
            used[word] = true;
        }
        return true;
    };
    if (!claim(layout.pageTableAddressWord, 2) || !claim(layout.faultRecordAddressWord, 2)) return false;
    if (layout.overflow && !claim(layout.overflowAddressWord, 2)) return false;
    for (std::size_t stage = 0; stage < Abi::StageCount; ++stage) {
        if (!claim(layout.userWordOffset[stage], layout.userWordCount[stage])) return false;
    }
    return true;
}

std::optional<PushBytes> ResolvedDispatchPush(const Abi::PushLayout& layout, std::span<const std::uint32_t> userWords, std::uint64_t tableRoot, std::uint64_t faultRecord) {
    if (layout.pageTableAddressWord != ResolvedPageTableWord || layout.faultRecordAddressWord != ResolvedFaultRecordWord) return std::nullopt;
    if (tableRoot == 0 || faultRecord == 0 || layout.overflow) return std::nullopt;
    for (std::size_t stage = 1; stage < Abi::StageCount; ++stage) {
        if (layout.userWordCount[stage] != 0) return std::nullopt;
    }
    const std::array<std::span<const std::uint32_t>, Abi::StageCount> words{userWords, {}, {}, {}};
    return BuildPushBlock(layout, words, {tableRoot, faultRecord, 0});
}

std::optional<PushBytes> BuildPushBlock(const Abi::PushLayout& layout, std::span<const std::span<const std::uint32_t>, Abi::StageCount> userWords, const PushAddresses& addresses) {
    if (!PushLayoutValid(layout)) return std::nullopt;
    for (std::size_t stage = 0; stage < Abi::StageCount; ++stage) {
        if (userWords[stage].size() != layout.userWordCount[stage]) return std::nullopt;
    }
    PushBytes bytes{};
    putWords(bytes, layout.pageTableAddressWord, &addresses.pageTable, 2);
    putWords(bytes, layout.faultRecordAddressWord, &addresses.faultRecord, 2);
    if (layout.overflow) putWords(bytes, layout.overflowAddressWord, &addresses.overflow, 2);
    for (std::size_t stage = 0; stage < Abi::StageCount; ++stage) {
        if (!userWords[stage].empty()) putWords(bytes, layout.userWordOffset[stage], userWords[stage].data(), userWords[stage].size());
    }
    return bytes;
}

}
