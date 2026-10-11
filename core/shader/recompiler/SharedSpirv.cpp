#include "Recompiler.hpp"
#include "ColdSpirvCodec.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include "ShaderCacheDirectory.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <share.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif
#ifdef ANYPS5_COLD_SPIRV_TESTING
#include "ColdSpirvStorageTestHooks.hpp"
#endif

namespace ShaderRecompiler {

namespace {

constexpr std::uint64_t FileLimit = 8ull << 30u;
constexpr std::uint64_t FileReserve = 4ull << 30u;
struct FileArena;
struct FilePool {
    std::mutex mutex;
    std::weak_ptr<FileArena> arena;
    bool occupied = false;
};
FilePool& Pool() {
    static auto* pool = new FilePool;
    return *pool;
}
struct FileCounters {
    std::atomic<std::uint64_t> active{0}, peak{0}, highWater{0}, live{0}, appends{0}, reads{0}, fallbacks{0};
#ifdef ANYPS5_COLD_SPIRV_TESTING
    std::atomic<std::uint64_t> restores{0}, peakRestores{0};
#endif
};
FileCounters& Counters() { static FileCounters counters; return counters; }
#ifdef ANYPS5_COLD_SPIRV_TESTING
std::atomic<int> fileEnabled{-1};
std::atomic<std::uint64_t> fileLimit{FileLimit}, fileReserve{FileReserve}, fileAvailable{UINT64_MAX};
std::atomic<ColdSpirv::Testing::FileFailure> fileFailure{ColdSpirv::Testing::FileFailure::None};
std::atomic<ColdSpirv::Testing::CacheFailure> cacheFailure{ColdSpirv::Testing::CacheFailure::None};
std::filesystem::path lastFilePath;
bool Fail(ColdSpirv::Testing::FileFailure wanted) {
    return fileFailure.compare_exchange_strong(wanted, ColdSpirv::Testing::FileFailure::None);
}
bool Fail(ColdSpirv::Testing::CacheFailure wanted) {
    return cacheFailure.compare_exchange_strong(wanted, ColdSpirv::Testing::CacheFailure::None);
}
#endif
bool FileEnabled() {
#ifdef ANYPS5_COLD_SPIRV_TESTING
    if (fileEnabled.load() >= 0) return fileEnabled.load() != 0;
#endif
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_COLD_SPIRV_FILE");
        return value != nullptr && std::string_view(value) == "1";
    }();
    return enabled;
}
std::uint64_t Limit() {
#ifdef ANYPS5_COLD_SPIRV_TESTING
    return fileLimit.load();
#else
    return FileLimit;
#endif
}
std::uint64_t Reserve() {
#ifdef ANYPS5_COLD_SPIRV_TESTING
    return fileReserve.load();
#else
    return FileReserve;
#endif
}
bool Fits(std::uint64_t offset, std::uint64_t bytes, std::uint64_t limit) {
    return offset <= limit && bytes <= limit - offset && offset <= INT64_MAX && bytes <= INT64_MAX - offset;
}
void Fallback(const char* reason) noexcept {
    auto& c = Counters();
    const auto count = c.fallbacks.fetch_add(1) + 1;
    if (count <= 4 || count % 256 == 0) std::fprintf(stderr, "[cold-spirv-file] fallback=%s count=%llu live_bytes=%llu high_water=%llu cap=%llu reserve=%llu\n", reason,
        static_cast<unsigned long long>(count), static_cast<unsigned long long>(c.live.load()), static_cast<unsigned long long>(c.highWater.load()),
        static_cast<unsigned long long>(Limit()), static_cast<unsigned long long>(Reserve()));
}
bool Seek(std::FILE* file, std::uint64_t offset) {
    if (offset > INT64_MAX) return false;
    std::clearerr(file);
#ifdef _WIN32
    return _fseeki64(file, static_cast<__int64>(offset), SEEK_SET) == 0;
#else
    static_assert(sizeof(off_t) >= sizeof(std::int64_t));
    return fseeko(file, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}
std::FILE* OpenTemporary(const std::filesystem::path& directory) {
#ifdef ANYPS5_COLD_SPIRV_TESTING
    if (Fail(ColdSpirv::Testing::FileFailure::Create)) return nullptr;
#endif
    int descriptor = -1;
    struct DescriptorGuard {
        int& descriptor;
        ~DescriptorGuard() {
            if (descriptor < 0) return;
#ifdef _WIN32
            _close(descriptor);
#else
            close(descriptor);
#endif
        }
    } guard{descriptor};
    std::filesystem::path path;
#ifdef _WIN32
    static std::atomic<std::uint64_t> sequence{0};
    for (unsigned attempt = 0; attempt < 64; ++attempt) {
        path = directory / ("anyps5-cold-spirv-" + std::to_string(_getpid()) + "-" + std::to_string(sequence.fetch_add(1)));
        const int error = _wsopen_s(&descriptor, path.c_str(), _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY | _O_TEMPORARY | _O_NOINHERIT,
                                  _SH_DENYRW, _S_IREAD | _S_IWRITE);
        if (error == 0) break;
        if (error != EEXIST) return nullptr;
    }
    if (descriptor < 0) return nullptr;
#else
    auto pattern = (directory / "anyps5-cold-spirv-XXXXXX").string();
#ifdef __linux__
    descriptor = mkostemp(pattern.data(), O_CLOEXEC);
#else
    descriptor = mkstemp(pattern.data());
#endif
    if (descriptor < 0) return nullptr;
    if (unlink(pattern.c_str()) != 0) {
        // Retry without allocating while the owned descriptor is still valid.
        unlink(pattern.c_str());
        return nullptr;
    }
#ifndef __linux__
    if (fcntl(descriptor, F_SETFD, FD_CLOEXEC) != 0) return nullptr;
#endif
    path = pattern;
#endif
#ifdef ANYPS5_COLD_SPIRV_TESTING
    lastFilePath = path;
    const bool attachFailure = Fail(ColdSpirv::Testing::FileFailure::Attach);
#else
    constexpr bool attachFailure = false;
#endif
#ifdef _WIN32
    auto* file = attachFailure ? nullptr : _fdopen(descriptor, "w+b");
#else
    auto* file = attachFailure ? nullptr : fdopen(descriptor, "w+b");
#endif
    if (file != nullptr) descriptor = -1;
    return file;
}

struct FileArena {
    FileArena(std::FILE* stream, std::filesystem::path root) : stream(stream), directory(std::move(root)) {}
    ~FileArena() {
        std::fclose(stream);
        auto& pool = Pool();
        std::lock_guard lock(pool.mutex);
        pool.occupied = false;
        Counters().active.store(0);
        Counters().highWater.store(0);
    }
    std::FILE* stream;
    std::filesystem::path directory;
    std::mutex ioMutex, restoreMutex;
    std::uint64_t highWater = 0;
    bool writable = true;
};

std::shared_ptr<FileArena> AcquireArena() {
    auto& pool = Pool();
    std::lock_guard lock(pool.mutex);
    if (auto arena = pool.arena.lock()) return arena;
    if (pool.occupied) return {};
    auto directory = std::filesystem::temp_directory_path();
    std::unique_ptr<std::FILE, decltype(&std::fclose)> file(OpenTemporary(directory), &std::fclose);
    if (file == nullptr) return {};
    auto arena = std::make_shared<FileArena>(file.get(), std::move(directory));
    file.release();
    pool.arena = arena;
    pool.occupied = true;
    Counters().active.store(1);
    Counters().peak.store(1);
    return arena;
}

struct FileSlice {
    explicit FileSlice(std::shared_ptr<FileArena> owner) : arena(std::move(owner)) {}
    ~FileSlice() { Counters().live.fetch_sub(bytes); }
    std::vector<std::uint32_t> Read(std::size_t wordCount) const {
        std::lock_guard restore(arena->restoreMutex);
#ifdef ANYPS5_COLD_SPIRV_TESTING
        struct RestoreCount {
            RestoreCount() { const auto count = Counters().restores.fetch_add(1) + 1; Counters().peakRestores.store(std::max(count, Counters().peakRestores.load())); }
            ~RestoreCount() { Counters().restores.fetch_sub(1); }
        } counted;
#endif
        if (bytes == 0 || bytes > ColdSpirv::MaxBytes || wordCount > ColdSpirv::MaxBytes / sizeof(std::uint32_t)) throw std::runtime_error("cold SPIR-V file range is invalid");
        std::vector<std::byte> encoded(static_cast<std::size_t>(bytes));
        {
            std::lock_guard lock(arena->ioMutex);
            if (!Fits(offset, bytes, arena->highWater) || !Seek(arena->stream, offset)) throw std::runtime_error("cold SPIR-V file seek failed");
#ifdef ANYPS5_COLD_SPIRV_TESTING
            if (Fail(ColdSpirv::Testing::FileFailure::Read)) throw std::runtime_error("cold SPIR-V file injected read failure");
            const auto requestBytes = encoded.size() - (Fail(ColdSpirv::Testing::FileFailure::ShortRead) ? 1u : 0u);
#else
            const auto requestBytes = encoded.size();
#endif
            if (std::fread(encoded.data(), 1, requestBytes, arena->stream) != encoded.size()) throw std::runtime_error("cold SPIR-V file read failed");
        }
#ifdef ANYPS5_COLD_SPIRV_TESTING
        if (Fail(ColdSpirv::Testing::FileFailure::CorruptRead)) encoded.front() ^= std::byte{1};
#endif
        if (HashBytes(encoded) != hash) throw std::runtime_error("cold SPIR-V file checksum differs");
        auto words = ColdSpirv::Decompress(encoded, wordCount);
        Counters().reads.fetch_add(1);
        return words;
    }
    std::shared_ptr<FileArena> arena;
    std::uint64_t offset = 0, bytes = 0, hash = 0;
};

std::shared_ptr<const FileSlice> Spill(std::span<const std::byte> encoded) noexcept {
    try {
        auto arena = AcquireArena();
        if (arena == nullptr) { Fallback("create"); return {}; }
        auto slice = std::make_shared<FileSlice>(arena);
        std::lock_guard lock(arena->ioMutex);
        if (!arena->writable) { Fallback("disabled_after_write_failure"); return {}; }
        if (encoded.empty() || encoded.size() > ColdSpirv::MaxBytes || !Fits(arena->highWater, encoded.size(), Limit())) { Fallback("cap"); return {}; }
        std::error_code error;
        auto available = std::filesystem::space(arena->directory, error).available;
#ifdef ANYPS5_COLD_SPIRV_TESTING
        if (fileAvailable.load() != UINT64_MAX) { available = fileAvailable.load(); error.clear(); }
#endif
        if (error || available < Reserve() || encoded.size() > available - Reserve()) { Fallback("reserve"); return {}; }
        const auto offset = arena->highWater;
        arena->highWater += encoded.size();
        Counters().highWater.store(arena->highWater);
        arena->writable = false;
        if (!Seek(arena->stream, offset)) { Fallback("seek"); return {}; }
#ifdef ANYPS5_COLD_SPIRV_TESTING
        if (Fail(ColdSpirv::Testing::FileFailure::Write)) {
            static_cast<void>(std::fwrite(encoded.data(), 1, encoded.size() / 2, arena->stream));
            Fallback("write"); return {};
        }
#endif
        if (std::fwrite(encoded.data(), 1, encoded.size(), arena->stream) != encoded.size() || std::fflush(arena->stream) != 0) { Fallback("write"); return {}; }
#ifdef ANYPS5_COLD_SPIRV_TESTING
        if (Fail(ColdSpirv::Testing::FileFailure::Flush)) { Fallback("flush"); return {}; }
#endif
        if (!Seek(arena->stream, offset)) { Fallback("verify_seek"); return {}; }
        std::array<std::byte, 64u * 1024u> check;
        for (std::size_t done = 0; done < encoded.size();) {
            const auto count = std::min(check.size(), encoded.size() - done);
            if (std::fread(check.data(), 1, count, arena->stream) != count) { Fallback("verify_read"); return {}; }
#ifdef ANYPS5_COLD_SPIRV_TESTING
            if (Fail(ColdSpirv::Testing::FileFailure::Verify)) check.front() ^= std::byte{1};
#endif
            if (std::memcmp(check.data(), encoded.data() + done, count) != 0) { Fallback("verify_bytes"); return {}; }
            done += count;
        }
        slice->offset = offset;
        slice->hash = HashBytes(encoded);
        slice->bytes = encoded.size();
        Counters().live.fetch_add(slice->bytes);
        arena->writable = true;
        const auto count = Counters().appends.fetch_add(1) + 1;
        if (count == 1 || count % 256 == 0) std::fprintf(stderr, "[cold-spirv-file] appends=%llu live_bytes=%llu high_water=%llu cap=%llu reserve=%llu\n",
            static_cast<unsigned long long>(count), static_cast<unsigned long long>(Counters().live.load()), static_cast<unsigned long long>(arena->highWater),
            static_cast<unsigned long long>(Limit()), static_cast<unsigned long long>(Reserve()));
        return slice;
    } catch (...) { Fallback("exception"); return {}; }
}

struct CacheSlice {
    ~CacheSlice() {
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
        if (handle >= 0) close(handle);
#endif
    }
    bool Open(const std::filesystem::path& path, std::span<const std::byte> expected) {
        if (expected.empty() || expected.size() > INT64_MAX) return false;
#ifdef _WIN32
        handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                             nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        LARGE_INTEGER size;
        if (handle == INVALID_HANDLE_VALUE || !GetFileSizeEx(handle, &size) || size.QuadPart < 0 ||
            static_cast<std::uint64_t>(size.QuadPart) != expected.size()) return false;
#else
        handle = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        struct stat status;
        if (handle < 0 || fstat(handle, &status) != 0 || status.st_size < 0 ||
            static_cast<std::uint64_t>(status.st_size) != expected.size()) return false;
#endif
        std::array<std::byte, 64u * 1024u> check;
        for (std::size_t done = 0; done < expected.size();) {
            const auto count = std::min(check.size(), expected.size() - done);
            if (!ReadAt(done, std::span(check).first(count)) ||
                std::memcmp(check.data(), expected.data() + done, count) != 0) return false;
            done += count;
        }
        return true;
    }
    std::vector<std::uint32_t> Read() const {
        static std::mutex restoreMutex;
        std::lock_guard restore(restoreMutex);
        std::vector<std::uint32_t> words(bytes / sizeof(std::uint32_t));
        auto output = std::as_writable_bytes(std::span(words));
#ifdef ANYPS5_COLD_SPIRV_TESTING
        if (Fail(ColdSpirv::Testing::CacheFailure::Read)) throw std::runtime_error("cache SPIR-V injected read failure");
        if (Fail(ColdSpirv::Testing::CacheFailure::ShortRead)) output = output.first(output.size() - 1);
#endif
        if (!ReadAt(offset, output) || output.size() != bytes) throw std::runtime_error("cache SPIR-V read failed");
#ifdef ANYPS5_COLD_SPIRV_TESTING
        if (Fail(ColdSpirv::Testing::CacheFailure::CorruptRead)) output.front() ^= std::byte{1};
#endif
        if (HashBytes(output) != hash) throw std::runtime_error("cache SPIR-V checksum differs");
        return words;
    }
    std::uint64_t offset = 0, bytes = 0, hash = 0;

private:
    bool ReadAt(std::uint64_t position, std::span<std::byte> output) const {
        if (!Fits(position, output.size(), INT64_MAX)) return false;
        std::lock_guard lock(ioMutex);
#ifdef _WIN32
        LARGE_INTEGER location;
        location.QuadPart = static_cast<LONGLONG>(position);
        if (!SetFilePointerEx(handle, location, nullptr, FILE_BEGIN)) return false;
#else
        static_assert(sizeof(off_t) >= sizeof(std::int64_t));
#endif
        while (!output.empty()) {
            const auto count = std::min<std::size_t>(output.size(), 1024u * 1024u);
#ifdef _WIN32
            DWORD read = 0;
            if (!ReadFile(handle, output.data(), static_cast<DWORD>(count), &read, nullptr) || read == 0) return false;
#else
            const auto read = pread(handle, output.data(), count, static_cast<off_t>(position));
            if (read < 0 && errno == EINTR) continue;
            if (read <= 0) return false;
#endif
            position += read;
            output = output.subspan(static_cast<std::size_t>(read));
        }
        return true;
    }
    mutable std::mutex ioMutex;
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int handle = -1;
#endif
};

}

#ifdef ANYPS5_COLD_SPIRV_TESTING
namespace ColdSpirv::Testing {
void ConfigureFileStorage(bool enabled, std::uint64_t limit, std::uint64_t reserve, std::uint64_t available) {
    if (Counters().active.load() != 0) throw std::runtime_error("file storage test reset has live owners");
    fileEnabled.store(enabled); fileLimit.store(limit); fileReserve.store(reserve); fileAvailable.store(available);
    fileFailure.store(FileFailure::None);
    auto& c = Counters();
    c.peak.store(0); c.highWater.store(0); c.live.store(0); c.appends.store(0); c.reads.store(0); c.fallbacks.store(0); c.peakRestores.store(0);
}
void FailFileOperation(FileFailure failure) noexcept { fileFailure.store(failure); }
void FailCacheOperation(CacheFailure failure) noexcept { cacheFailure.store(failure); }
FileStats InspectFileStorage() noexcept {
    auto& c = Counters();
    return {c.active.load(), c.peak.load(), c.highWater.load(), c.live.load(), c.appends.load(), c.reads.load(), c.fallbacks.load(), c.peakRestores.load()};
}
std::filesystem::path LastFilePath() { return lastFilePath; }
bool FileRangeFits(std::uint64_t offset, std::uint64_t bytes, std::uint64_t limit) noexcept { return Fits(offset, bytes, limit); }
}
#endif

struct SharedSpirv::Storage {
    explicit Storage(std::vector<std::uint32_t> value) : words(std::make_shared<std::vector<std::uint32_t>>(std::move(value))) {}

    std::vector<std::uint32_t> ReadCold() const {
        if (cache != nullptr) return cache->Read();
        return file != nullptr ? file->Read(wordCount) : ColdSpirv::Decompress(compressed, wordCount);
    }

    void Hydrate() {
        if (words != nullptr) return;
        auto loaded = std::make_shared<std::vector<std::uint32_t>>(ReadCold());
        words = std::move(loaded);
        std::vector<std::byte>().swap(compressed);
        file.reset();
        cache.reset();
    }

    mutable std::mutex mutex;
    std::shared_ptr<std::vector<std::uint32_t>> words;
    std::vector<std::byte> compressed;
    std::shared_ptr<const FileSlice> file;
    std::shared_ptr<const CacheSlice> cache;
    std::size_t wordCount = 0;
    std::atomic<bool> exposed{false};
    bool attempted = false;
};

SharedSpirv::SharedSpirv(std::vector<std::uint32_t> words) : storage(std::make_shared<Storage>(std::move(words))) {}

SharedSpirv& SharedSpirv::operator=(std::vector<std::uint32_t> other) {
    storage = std::make_shared<Storage>(std::move(other));
    return *this;
}

const std::vector<std::uint32_t>& SharedSpirv::Words() const {
    if (storage == nullptr) return Empty();
    if (storage->exposed.load(std::memory_order_acquire)) return *storage->words;
    std::lock_guard lock(storage->mutex);
    storage->Hydrate();
    storage->exposed.store(true, std::memory_order_release);
    return *storage->words;
}

std::shared_ptr<const std::vector<std::uint32_t>> SharedSpirv::ReadTransient() const {
    if (storage == nullptr) {
        static const auto empty = std::make_shared<const std::vector<std::uint32_t>>();
        return empty;
    }
    std::lock_guard lock(storage->mutex);
    if (storage->words != nullptr) return storage->words;
    return std::make_shared<const std::vector<std::uint32_t>>(storage->ReadCold());
}

bool SharedSpirv::TryUseCacheFile(const std::filesystem::path& path, std::uint64_t spirvOffset,
                                std::span<const std::byte> verifiedEntry) const noexcept {
    try {
        if (storage == nullptr) return false;
        std::lock_guard lock(storage->mutex);
        if (storage->exposed.load(std::memory_order_relaxed) || storage->cache != nullptr ||
            (storage->words != nullptr && storage->words.use_count() != 1)) return false;
        const auto count = storage->words != nullptr ? storage->words->size() : storage->wordCount;
        if (count == 0 || count > ColdSpirv::MaxBytes / sizeof(std::uint32_t) ||
            !Fits(spirvOffset, count * sizeof(std::uint32_t), verifiedEntry.size())) return false;
        const auto expected = verifiedEntry.subspan(static_cast<std::size_t>(spirvOffset), count * sizeof(std::uint32_t));
        auto cache = std::make_shared<CacheSlice>();
        if (!cache->Open(path, verifiedEntry)) return false;
        std::vector<std::uint32_t> restored;
        if (storage->words == nullptr) restored = storage->ReadCold();
        const auto& words = storage->words != nullptr ? *storage->words : restored;
        if (std::memcmp(words.data(), expected.data(), expected.size()) != 0) return false;
        cache->offset = spirvOffset;
        cache->bytes = expected.size();
        cache->hash = HashBytes(expected);
        storage->cache = std::move(cache);
        storage->wordCount = count;
        storage->words.reset();
        std::vector<std::byte>().swap(storage->compressed);
        storage->file.reset();
        return true;
    } catch (...) {
        return false;
    }
}

bool SharedSpirv::TryMakeCold() noexcept {
    try {
        if (storage == nullptr || storage.use_count() != 1) return false;
        std::lock_guard lock(storage->mutex);
        if (storage->exposed.load(std::memory_order_relaxed) || storage->attempted || storage->words == nullptr || storage->words.use_count() != 1) return false;
        storage->attempted = true;
        if (storage->words->size() < 16u * 1024u || storage->words->size() > ColdSpirv::MaxBytes / sizeof(std::uint32_t)) return false;
        auto compressed = ColdSpirv::Compress(*storage->words);
        if (!compressed || compressed->size() >= storage->words->size() * sizeof(std::uint32_t)) return false;
        storage->wordCount = storage->words->size();
        if (FileEnabled()) storage->file = Spill(*compressed);
        if (storage->file == nullptr) storage->compressed = std::move(*compressed);
        storage->words.reset();
        return true;
    } catch (...) {
        return false;
    }
}

std::size_t SharedSpirv::ColdBytes() const {
    if (storage == nullptr) return 0;
    if (storage->exposed.load(std::memory_order_acquire)) return 0;
    std::lock_guard lock(storage->mutex);
    if (storage->cache != nullptr) return static_cast<std::size_t>(storage->cache->bytes);
    return storage->file != nullptr ? static_cast<std::size_t>(storage->file->bytes) : storage->compressed.size();
}

SharedSpirv::StorageInfo SharedSpirv::InspectStorage() const {
    if (storage == nullptr) return {};
    std::lock_guard lock(storage->mutex);
    StorageInfo info;
    info.logicalBytes = (storage->words != nullptr ? storage->words->size() : storage->wordCount) * sizeof(std::uint32_t);
    if (storage->words != nullptr) {
        info.residentBytes = storage->words->size() * sizeof(std::uint32_t);
        info.residentCapacityBytes = storage->words->capacity() * sizeof(std::uint32_t);
    }
    info.coldBytes = storage->compressed.size();
    info.coldCapacityBytes = storage->compressed.capacity();
    info.fileBytes = storage->file != nullptr ? static_cast<std::size_t>(storage->file->bytes) : 0;
    info.cacheBytes = storage->cache != nullptr ? static_cast<std::size_t>(storage->cache->bytes) : 0;
    info.exposed = storage->exposed.load(std::memory_order_relaxed);
    info.eligible = storage.use_count() == 1 && storage->words != nullptr && storage->words.use_count() == 1 && !info.exposed && !storage->attempted && info.logicalBytes >= 64u * 1024u && info.logicalBytes <= ColdSpirv::MaxBytes;
    return info;
}

std::size_t SharedSpirv::size() const {
    if (storage == nullptr) return 0;
    if (storage->exposed.load(std::memory_order_acquire)) return storage->words->size();
    std::lock_guard lock(storage->mutex);
    return storage->words != nullptr ? storage->words->size() : storage->wordCount;
}

std::vector<std::uint32_t>& SharedSpirv::Mutable() {
    if (storage == nullptr || storage.use_count() != 1) storage = std::make_shared<Storage>(*ReadTransient());
    std::lock_guard lock(storage->mutex);
    storage->Hydrate();
    if (storage->words.use_count() != 1) storage->words = std::make_shared<std::vector<std::uint32_t>>(*storage->words);
    storage->exposed.store(true, std::memory_order_release);
    return *storage->words;
}

std::vector<std::uint32_t>::iterator SharedSpirv::insert(std::vector<std::uint32_t>::const_iterator where, std::initializer_list<std::uint32_t> values) {
    const auto offset = where - Words().cbegin();
    auto& words = Mutable();
    return words.insert(words.cbegin() + offset, values);
}

}
