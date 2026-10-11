#include "Recompiler.hpp"
#include "ColdSpirvCodec.hpp"
#include "ColdSpirvCodecTestHooks.hpp"
#include "ColdSpirvStorageTestHooks.hpp"
#include "ShaderCacheDirectory.hpp"
#include <barrier>
#include <cstring>
#include <future>
#include <iostream>
#include <latch>
#include <random>
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

struct CacheFile {
    static constexpr std::uint64_t Offset = 16;
    std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("aps5-owned-cache-test-" + std::to_string(std::random_device{}()));
    std::filesystem::path path = directory / "entry.bin";
    std::vector<std::uint32_t> words = FileWords();
    std::vector<std::byte> bytes;

    CacheFile() : bytes(Offset + words.size() * sizeof(std::uint32_t) + 16, std::byte{0x35}) {
        Require(std::filesystem::create_directory(directory), "cache fixture directory already exists");
        std::memcpy(bytes.data() + Offset, words.data(), words.size() * sizeof(std::uint32_t));
        Require(WriteFileAtomically(path, bytes), "cannot publish cache fixture");
    }
    ~CacheFile() {
        std::error_code error;
        std::filesystem::remove(path, error);
        std::filesystem::remove(directory, error);
    }
    void WriteInPlace(std::span<const std::byte> data) const {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        Require(static_cast<bool>(file), "cannot open cache fixture for in-place write");
        file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        file.close();
        Require(!file.fail(), "cannot finish in-place cache fixture write");
    }
};

template<class F>
void RequireCacheFailure(F&& action, const char* message) {
    bool failed = false;
    try { action(); }
    catch (const std::runtime_error& error) { failed = std::string_view(error.what()) == message; }
    Require(failed, "cache read did not report the expected failure");
}

void CacheAcquisitionAndMoves() {
    CacheFile file;
    SharedSpirv raw(file.words);
    auto unchanged = [&] {
        const auto info = raw.InspectStorage();
        Require(info.residentBytes == file.words.size() * sizeof(std::uint32_t) && info.cacheBytes == 0 && !info.exposed &&
                *raw.ReadTransient() == file.words, "failed cache acquisition changed resident storage");
    };
    Require(!raw.TryUseCacheFile(file.directory / "absent", CacheFile::Offset, file.bytes), "missing cache file was accepted");
    Require(!raw.TryUseCacheFile(file.path, UINT64_MAX - 8, file.bytes), "overflowing cache range was accepted");
    Require(!raw.TryUseCacheFile(file.path, file.bytes.size(), file.bytes), "out-of-file cache range was accepted");
    Require(!raw.TryUseCacheFile(file.path, CacheFile::Offset + 4, file.bytes), "incorrect in-bounds cache word offset was accepted");
    unchanged();
    auto replacement = file.bytes;
    replacement.front() ^= std::byte{1};
    Require(WriteFileAtomically(file.path, replacement), "cannot replace cache fixture before acquisition");
    Require(!raw.TryUseCacheFile(file.path, CacheFile::Offset, file.bytes), "changed full entry was accepted before acquisition");
    replacement = file.bytes;
    replacement[CacheFile::Offset] ^= std::byte{1};
    Require(WriteFileAtomically(file.path, replacement), "cannot publish different cache words");
    Require(!raw.TryUseCacheFile(file.path, CacheFile::Offset, replacement), "verified different words replaced the original artifact");
    file.WriteInPlace(std::span(file.bytes).first(CacheFile::Offset + 4));
    Require(!raw.TryUseCacheFile(file.path, CacheFile::Offset, file.bytes), "truncated cache file was accepted");
    unchanged();
    file.WriteInPlace(file.bytes);
    {
        const auto pin = raw.ReadTransient();
        Require(!raw.TryUseCacheFile(file.path, CacheFile::Offset, file.bytes), "a live transient resident pin permitted eviction");
        Require(*pin == file.words, "failed pinned acquisition changed borrowed words");
    }
    Require(raw.TryUseCacheFile(file.path, CacheFile::Offset, file.bytes), "cache acquisition did not retry after pin release");
    const auto state = raw.InspectStorage();
    Require(state.cacheBytes == file.words.size() * sizeof(std::uint32_t) && state.residentBytes == 0 && state.residentCapacityBytes == 0 &&
            state.coldBytes == 0 && state.coldCapacityBytes == 0 && state.fileBytes == 0 && !state.exposed,
            "cache publication retained another payload or reported incorrect bytes");
    Require(!raw.TryMakeCold() && !raw.TryUseCacheFile(file.path, CacheFile::Offset, file.bytes), "cache-backed storage was replaced or recompressed");
    Require(*raw.ReadTransient() == file.words && raw.InspectStorage().cacheBytes == state.cacheBytes, "transient cache read exposed or changed backing");
    const auto& held = raw.Words();
    SharedSpirv moved(std::move(raw));
    Require(!moved.TryUseCacheFile(file.path, CacheFile::Offset, file.bytes), "move construction lost the exposed-reference guard");
    SharedSpirv assigned;
    assigned = std::move(moved);
    Require(!assigned.TryUseCacheFile(file.path, CacheFile::Offset, file.bytes), "move assignment lost the exposed-reference guard");
    Require(&assigned.Words() == &held && held == file.words, "move changed a previously exposed word reference");
    for (const bool spill : {false, true}) {
        Testing::ConfigureFileStorage(spill);
        SharedSpirv cold(file.words);
        Require(cold.TryMakeCold(), "cache transition fixture did not become cold");
        Require(cold.TryUseCacheFile(file.path, CacheFile::Offset, file.bytes), "cold storage could not move to its verified cache entry");
        Require(cold.InspectStorage().cacheBytes == state.cacheBytes && cold.InspectStorage().fileBytes == 0 &&
                cold.InspectStorage().coldCapacityBytes == 0 && Testing::InspectFileStorage().activeFiles == 0 &&
                *cold.ReadTransient() == file.words, "cache adoption retained the prior compressed/spool payload");
    }
}

void CacheFailureAndOwnership() {
    CacheFile file;
    SharedSpirv source(file.words);
    Require(source.TryUseCacheFile(file.path, CacheFile::Offset, file.bytes), "cache failure fixture did not acquire its file");
    const auto bytes = source.InspectStorage().cacheBytes;
    for (const auto [failure, message] : {
            std::pair{Testing::CacheFailure::Read, "cache SPIR-V injected read failure"},
            std::pair{Testing::CacheFailure::ShortRead, "cache SPIR-V read failed"},
            std::pair{Testing::CacheFailure::CorruptRead, "cache SPIR-V checksum differs"}}) {
        Testing::FailCacheOperation(failure);
        RequireCacheFailure([&] { static_cast<void>(source.Words()); }, message);
        Require(source.InspectStorage().cacheBytes == bytes && !source.InspectStorage().exposed &&
                *source.ReadTransient() == file.words, "cache read failure lost the retryable original backing");
    }
    auto changed = source;
    Testing::FailCacheOperation(Testing::CacheFailure::Read);
    RequireCacheFailure([&] { changed[13] ^= 0x10u; }, "cache SPIR-V injected read failure");
    Require(changed.InspectStorage().cacheBytes == bytes && source.InspectStorage().cacheBytes == bytes, "failed cache COW detached its source");
    const auto pin = source.ReadTransient();
    changed[13] ^= 0x10u;
    auto expectedChange = file.words;
    expectedChange[13] ^= 0x10u;
    Require(changed.Words() == expectedChange && *pin == file.words && *source.ReadTransient() == file.words &&
            source.InspectStorage().cacheBytes == bytes, "cache COW changed a transient pin or its sibling");
#ifdef _WIN32
    {
        std::ofstream writer(file.path, std::ios::binary | std::ios::trunc);
        Require(!writer, "an owned cache handle permitted in-place truncation");
    }
#else
    auto corrupt = file.bytes;
    corrupt[CacheFile::Offset] ^= std::byte{1};
    file.WriteInPlace(corrupt);
    RequireCacheFailure([&] { static_cast<void>(source.ReadTransient()); }, "cache SPIR-V checksum differs");
    file.WriteInPlace(std::span(file.bytes).first(CacheFile::Offset + 4));
    RequireCacheFailure([&] { static_cast<void>(source.Words()); }, "cache SPIR-V read failed");
    Require(source.InspectStorage().cacheBytes == bytes && !source.InspectStorage().exposed, "in-place damage discarded the cache handle");
    file.WriteInPlace(file.bytes);
    Require(*source.ReadTransient() == file.words, "cache restore could not retry after in-place file repair");
#endif
    auto last = source;
    source = SharedSpirv{};
    Require(*last.ReadTransient() == file.words, "releasing one owner closed the shared cache file");
    Require(last.Words() == file.words && last.InspectStorage().cacheBytes == 0, "hydration failed to release cache backing");
    file.WriteInPlace(file.bytes);
    {
        SharedSpirv finalOwner(file.words);
        Require(finalOwner.TryUseCacheFile(file.path, CacheFile::Offset, file.bytes), "destruction fixture did not acquire cache backing");
    }
    file.WriteInPlace(file.bytes);
}

void CacheConcurrentReaders() {
    CacheFile file;
    SharedSpirv source(file.words);
    Require(source.TryUseCacheFile(file.path, CacheFile::Offset, file.bytes), "concurrent cache fixture did not acquire its file");
    std::barrier start(4);
    std::latch launchReady(1);
    bool launchFailed = false;
    std::vector<std::future<const std::vector<std::uint32_t>*>> tasks;
    tasks.reserve(4);
    try {
        for (unsigned i = 0; i < 4; ++i) tasks.push_back(std::async(std::launch::async, [&, i] {
            launchReady.wait();
            if (launchFailed) return static_cast<const std::vector<std::uint32_t>*>(nullptr);
            start.arrive_and_wait();
            std::exception_ptr failure;
            try {
                const auto transient = source.ReadTransient();
                auto copy = source;
                copy[13] ^= i + 1;
                auto expected = file.words;
                expected[13] ^= i + 1;
                Require(*transient == file.words && copy.Words() == expected && !source.InspectStorage().exposed,
                        "concurrent transient reads or COW changed shared cache storage");
            } catch (...) { failure = std::current_exception(); }
            start.arrive_and_wait();
            if (failure) std::rethrow_exception(failure);
            return &source.Words();
        }));
    } catch (...) {
        launchFailed = true;
        launchReady.count_down();
        throw;
    }
    launchReady.count_down();
    const std::vector<std::uint32_t>* first = nullptr;
    for (auto& task : tasks) {
        const auto* result = task.get();
        Require(result != nullptr && *result == file.words && (first == nullptr || first == result), "concurrent cache hydration published different words");
        first = result;
    }
    Require(source.InspectStorage().cacheBytes == 0, "concurrent hydration retained cache backing");
    file.WriteInPlace(file.bytes);
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
        CacheAcquisitionAndMoves();
        CacheFailureAndOwnership();
        CacheConcurrentReaders();
        std::cout << "cold SPIR-V storage allocation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        Testing::ResetAllocationFailure();
        std::cerr << error.what() << '\n';
        return 1;
    }
}
