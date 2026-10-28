#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/HeapWalkMemo.hpp"
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::DriverDetail::HeapWalkEntry;
using AgcDriver::DriverDetail::HeapWalkEntryBytes;
using AgcDriver::DriverDetail::HeapWalkMemo;
using AgcDriver::DriverDetail::HeapWalkMemoKeys;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

HeapWalkEntry sizedEntry(std::size_t words, std::shared_ptr<const ShaderRecompiler::RecompileResult> result = nullptr) {
    HeapWalkEntry entry;
    entry.values.resize(words);
    entry.values.shrink_to_fit();
    entry.result = std::move(result);
    return entry;
}

std::uint64_t heldBytes(const HeapWalkMemo& memo) {
    std::uint64_t bytes = 0;
    for (const auto* generation : {&memo.current, &memo.previous}) {
        for (const auto& [key, ways] : *generation) {
            for (const auto& entry : ways) bytes += HeapWalkEntryBytes(entry);
        }
    }
    return bytes;
}

void requireWithinBudget(const HeapWalkMemo& memo, const std::string& where) {
    const auto held = heldBytes(memo);
    require(held == memo.heldBytes, where + ": the heap walk memo charged " + std::to_string(memo.heldBytes) + " bytes for the " + std::to_string(held) + " it holds");
    require(held <= memo.budget, where + ": the heap walk memo holds " + std::to_string(held) + " bytes, over its budget of " + std::to_string(memo.budget));
}

void oldestEntriesDroppedTests() {
    constexpr std::size_t words = 4096;
    require(HeapWalkEntryBytes(sizedEntry(words)) == sizeof(HeapWalkEntry) + words * sizeof(std::uint32_t), "a heap walk entry is not charged its struct and the capacity of its vectors");
    const auto entryBytes = HeapWalkEntryBytes(sizedEntry(words, std::make_shared<const ShaderRecompiler::RecompileResult>()));
    HeapWalkMemo memo;
    memo.budget = 16 * entryBytes;
    constexpr std::uint64_t keys = 64;
    std::vector<std::weak_ptr<const ShaderRecompiler::RecompileResult>> results;
    for (std::uint64_t key = 1; key <= keys; ++key) {
        auto result = std::make_shared<const ShaderRecompiler::RecompileResult>();
        results.push_back(result);
        memo.remember(key, sizedEntry(words, std::move(result)));
        requireWithinBudget(memo, "after key " + std::to_string(key));
    }
    require(memo.find(1) == nullptr && memo.find(keys / 2) == nullptr, "the oldest heap walk entries stayed past the budget");
    require(results.front().expired() && results[keys / 2 - 1].expired(), "a heap walk entry the budget dropped still holds its result");
    const auto* newest = memo.find(keys);
    require(newest != nullptr && newest->size() == 1 && newest->front().values.size() == words, "the newest heap walk entry is not findable under the budget");
    require(!results.back().expired(), "the newest heap walk entry lost its result");
    std::cout << "the heap walk memo keeps " << memo.current.size() << " of " << keys << " entries within its budget of " << memo.budget << " bytes\n";
}

void previousGenerationDroppedFirstTests() {
    const auto tinyBytes = HeapWalkEntryBytes(sizedEntry(1));
    constexpr std::size_t largeWords = std::size_t{2} << 20u;
    const auto largeBytes = HeapWalkEntryBytes(sizedEntry(largeWords));
    HeapWalkMemo memo;
    memo.budget = largeBytes + 64 * tinyBytes;
    require((HeapWalkMemoKeys + 1) * tinyBytes < memo.budget, "the generation fixture does not fit its budget");
    for (std::uint64_t key = 1; key <= HeapWalkMemoKeys + 1; ++key) memo.remember(key, sizedEntry(1));
    require(memo.previous.size() == HeapWalkMemoKeys && memo.current.size() == 1, "the heap walk memo did not start a generation at its key cap");
    requireWithinBudget(memo, "after the generation swap");
    memo.remember(HeapWalkMemoKeys + 2, sizedEntry(largeWords));
    requireWithinBudget(memo, "after the large entry");
    require(memo.previous.empty(), "the heap walk memo kept its previous generation over the budget");
    require(memo.find(HeapWalkMemoKeys + 1) != nullptr && memo.find(HeapWalkMemoKeys + 2) != nullptr, "the heap walk memo dropped current entries while its previous generation could go");
    std::cout << "the heap walk memo drops its previous generation before its current entries\n";
}

}

int main() {
    try {
        oldestEntriesDroppedTests();
        previousGenerationDroppedFirstTests();
        std::cout << "heap walk memo budget tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "heap walk memo budget tests failed: " << error.what() << '\n';
        return 1;
    }
}
