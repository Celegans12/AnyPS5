#include "Recompiler.hpp"
#include "ColdSpirvCodec.hpp"
#include "ColdSpirvCodecTestHooks.hpp"
#include "ColdSpirvStorageTestHooks.hpp"
#include <barrier>
#include <future>
#include <iostream>
#include <stdexcept>

namespace {

using namespace ShaderRecompiler;
namespace Testing = ColdSpirv::Testing;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class F>
void RequireFailure(F&& action) {
    bool failed = false;
    try { action(); }
    catch (const std::exception&) { failed = true; }
    Testing::ResetAllocationFailure();
    Require(failed, "injected allocation failure did not reach the storage operation");
    Require(Testing::ActiveAllocations() == 0 && Testing::ActiveBytes() == 0, "failed storage operation leaked codec allocations");
}

void AllocationFailurePreservesStorage() {
    const std::vector<std::uint32_t> expected(32u * 1024u, 0x07230203u);
    SharedSpirv raw(expected);
    Testing::FailAllocationAfter(0);
    Require(!raw.TryMakeCold(), "failed compression reported successful demotion");
    Testing::ResetAllocationFailure();
    Require(raw.ColdBytes() == 0 && raw.Words() == expected, "failed compression lost resident words");

    const auto compressed = ColdSpirv::Compress(expected);
    Require(compressed.has_value(), "allocation fixture did not compress");
    Testing::ResetAllocationFailure();
    Require(ColdSpirv::Decompress(*compressed, expected.size()) == expected, "allocation fixture did not decode");
    const auto checks = Testing::AllocationChecks();
    for (std::size_t failAt = 0; failAt < checks; ++failAt) {
        SharedSpirv cold(expected);
        Require(cold.TryMakeCold(), "hydration fixture did not become cold");
        const auto bytes = cold.ColdBytes();
        Testing::FailAllocationAfter(failAt);
        RequireFailure([&] { static_cast<void>(cold.Words()); });
        Require(cold.ColdBytes() == bytes && cold.InspectStorage().residentBytes == 0, "failed hydration discarded its compressed source");
        Require(cold.Words() == expected && cold.ColdBytes() == 0, "hydration could not retry after allocation failure");
    }

    SharedSpirv cold(expected);
    Require(cold.TryMakeCold(), "transient fixture did not become cold");
    const auto bytes = cold.ColdBytes();
    Testing::FailAllocationAfter(0);
    RequireFailure([&] { static_cast<void>(cold.ReadTransient()); });
    Require(cold.ColdBytes() == bytes && !cold.InspectStorage().exposed, "failed transient read changed cold storage");
    Require(*cold.ReadTransient() == expected && cold.ColdBytes() == bytes, "transient read could not retry without hydration");

    auto copy = cold;
    Testing::FailAllocationAfter(0);
    RequireFailure([&] { copy[0] = 1u; });
    Require(copy.ColdBytes() == bytes && cold.ColdBytes() == bytes, "failed cold copy-on-write discarded shared storage");
    copy[0] = 1u;
    Require(cold.ColdBytes() == bytes && copy.Words()[0] == 1u && *cold.ReadTransient() == expected, "copy-on-write retry changed its cold sibling");
}

std::vector<std::uint32_t> FileWords(std::uint32_t salt = 0) {
    std::vector<std::uint32_t> words(32u * 1024u);
    for (std::size_t i = 0; i < words.size(); ++i) words[i] = static_cast<std::uint32_t>(i % 8192u) ^ salt;
    return words;
}

void FilePublicationFailures() {
    const auto words = FileWords();
    const auto encoded = ColdSpirv::Compress(words);
    Require(encoded.has_value(), "file fixture did not compress");
    for (const auto failure : {Testing::FileFailure::Create, Testing::FileFailure::Attach, Testing::FileFailure::Write,
                              Testing::FileFailure::Flush, Testing::FileFailure::Verify}) {
        Testing::ConfigureFileStorage(true);
        {
            SharedSpirv cold(words);
            Testing::FailFileOperation(failure);
            Require(cold.TryMakeCold(), "file publication failure lost RAM compression fallback");
            const auto state = cold.InspectStorage();
            Require(state.fileBytes == 0 && state.coldBytes == encoded->size() && state.residentBytes == 0, "failed file publication discarded the owned compressed payload");
            Require(*cold.ReadTransient() == words && !cold.InspectStorage().exposed, "fallback transient read changed bytes or exposed storage");
            Require(Testing::InspectFileStorage().activeFiles == 0, "failed first append retained a temporary handle");
        }
        const auto path = Testing::LastFilePath();
        Require(path.empty() || !std::filesystem::exists(path), "failed file creation/attachment left a named temporary file");
    }
    for (const bool cap : {false, true}) {
        Testing::ConfigureFileStorage(true, cap ? encoded->size() - 1 : 8ull << 30u, 4096,
                                     cap ? UINT64_MAX : encoded->size() + 4095);
        SharedSpirv cold(words);
        Require(cold.TryMakeCold() && cold.InspectStorage().fileBytes == 0 && cold.InspectStorage().coldBytes == encoded->size(), "cap or reserve refusal did not retain compressed RAM");
        Require(Testing::InspectFileStorage().highWater == 0 && Testing::InspectFileStorage().fallbacks == 1, "refused append consumed spool space");
    }
    Testing::ConfigureFileStorage(true, 8ull << 30u, 4096, encoded->size() + 4096);
    {
        SharedSpirv exact(words);
        Require(exact.TryMakeCold() && exact.InspectStorage().fileBytes == encoded->size(), "exact free-space reserve boundary was rejected");
    }
    Require(Testing::FileRangeFits((1ull << 32u) + 19, 127, 8ull << 30u), "file offset above4GiB was truncated");
    Require(Testing::FileRangeFits((8ull << 30u) - 127, 127, 8ull << 30u), "exact arena-end range was rejected");
    Require(!Testing::FileRangeFits((8ull << 30u) - 127, 128, 8ull << 30u) &&
            !Testing::FileRangeFits(UINT64_MAX - 16, 32, UINT64_MAX), "file range arithmetic wrapped or exceeded its bound");
}

void FileLifetimeAndFailures() {
    const auto words = FileWords();
    Testing::ConfigureFileStorage(true);
    std::filesystem::path path;
    {
        SharedSpirv first(words);
        Require(first.TryMakeCold(), "owned-file fixture did not become cold");
        const auto initial = first.InspectStorage();
        Require(initial.fileBytes > 0 && initial.coldBytes == 0 && initial.coldCapacityBytes == 0 && initial.residentBytes == 0 && !initial.exposed, "file publication retained heap payload or exposed words");
        path = Testing::LastFilePath();
        auto alias = first;
        first = SharedSpirv{};
        Require(Testing::InspectFileStorage().activeFiles == 1 && alias.size() == words.size(), "shared cold owner lost its backing");
        for (const auto failure : {Testing::FileFailure::Read, Testing::FileFailure::ShortRead, Testing::FileFailure::CorruptRead}) {
            Testing::FailFileOperation(failure);
            RequireFailure([&] { static_cast<void>(alias.Words()); });
            Require(alias.InspectStorage().fileBytes == initial.fileBytes && !alias.InspectStorage().exposed, "read failure discarded or exposed file backing");
            Require(*alias.ReadTransient() == words && alias.InspectStorage().fileBytes == initial.fileBytes, "file read could not retry while remaining cold");
        }
        auto changed = alias;
        changed[13] ^= 0x10u;
        Require(alias.InspectStorage().fileBytes == initial.fileBytes && changed.InspectStorage().fileBytes == 0 && changed.Words()[13] == (words[13] ^ 0x10u), "file-backed COW changed its source or lost its edit");
        Require(alias.Words() == words && alias.InspectStorage().fileBytes == 0, "file hydration changed words or kept backing alive");
        Require(Testing::InspectFileStorage().activeFiles == 0 && Testing::InspectFileStorage().liveBytes == 0, "hydration retained the last temporary arena owner");
    }
    Require(!std::filesystem::exists(path), "last file owner did not delete its temporary file");
    Testing::ConfigureFileStorage(true);
    {
        SharedSpirv earlier(words);
        Require(earlier.TryMakeCold(), "first append failed");
        const auto bytes = earlier.InspectStorage().fileBytes;
        Testing::FailFileOperation(Testing::FileFailure::Write);
        SharedSpirv failed(words);
        Require(failed.TryMakeCold() && failed.InspectStorage().fileBytes == 0, "failed tail append replaced RAM fallback");
        Require(Testing::InspectFileStorage().highWater == 2 * bytes && Testing::InspectFileStorage().liveBytes == bytes, "failed tail was not charged separately from live data");
        Require(*earlier.ReadTransient() == words, "failed later write damaged an earlier published slice");
        SharedSpirv after(words);
        Require(after.TryMakeCold() && after.InspectStorage().fileBytes == 0 && Testing::InspectFileStorage().highWater == 2 * bytes, "sealed arena accepted another append after write failure");
    }
    Require(Testing::InspectFileStorage().activeFiles == 0, "sealed arena leaked after its last slice was released");
}

void FileConcurrentReaders() {
    Testing::ConfigureFileStorage(true);
    {
        std::vector<SharedSpirv> cold;
        for (std::uint32_t i = 0; i < 8; ++i) {
            cold.emplace_back(FileWords(i));
            Require(cold.back().TryMakeCold() && cold.back().InspectStorage().fileBytes != 0, "concurrent fixture did not spill");
        }
        std::barrier start(8);
        std::vector<std::future<void>> tasks;
        for (std::uint32_t i = 0; i < 8; ++i) tasks.push_back(std::async(std::launch::async, [&, i] {
            start.arrive_and_wait();
            Require(*cold[i].ReadTransient() == FileWords(i) && !cold[i].InspectStorage().exposed, "concurrent transient restore changed words/state");
            start.arrive_and_wait();
            Require(cold[0].Words() == FileWords(0), "concurrent first readers saw different words");
        }));
        for (auto& task : tasks) task.get();
        const auto stats = Testing::InspectFileStorage();
        Require(stats.maxActiveFiles == 1 && stats.maxRestores == 1 && stats.activeFiles == 1, "file handles or active restore scratch were not bounded");
        Require(stats.highWater > stats.liveBytes, "hydrated dead ranges were not retained in high-water accounting");
    }
    Require(Testing::InspectFileStorage().activeFiles == 0, "concurrent readers retained the arena after release");
}

}

int main() {
    try {
        Testing::ConfigureFileStorage(false);
        AllocationFailurePreservesStorage();
        FilePublicationFailures();
        FileLifetimeAndFailures();
        FileConcurrentReaders();
        Testing::ConfigureFileStorage(true);
        AllocationFailurePreservesStorage();
        Require(Testing::InspectFileStorage().activeFiles == 0, "allocation-failure tests leaked a file owner");
        std::cout << "cold SPIR-V storage allocation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        Testing::ResetAllocationFailure();
        std::cerr << error.what() << '\n';
        return 1;
    }
}
