#include "ColdSpirvCodec.hpp"
#include <zlib.h>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

#ifdef ANYPS5_COLD_SPIRV_TESTING
#include "ColdSpirvCodecTestHooks.hpp"
#endif

namespace ShaderRecompiler::ColdSpirv {
namespace {

constexpr std::size_t ScratchLimit = 1024u * 1024u;
static_assert(MaxBytes <= std::numeric_limits<uInt>::max());

#ifdef ANYPS5_COLD_SPIRV_TESTING
struct AllocationControl {
    std::size_t remaining = std::numeric_limits<std::size_t>::max();
    std::size_t checks = 0;
    std::size_t active = 0;
    std::size_t bytes = 0;
};

thread_local AllocationControl allocations;
#endif

bool AllowAllocation() noexcept {
#ifdef ANYPS5_COLD_SPIRV_TESTING
    ++allocations.checks;
    if (allocations.remaining == 0) return false;
    --allocations.remaining;
#endif
    return true;
}

struct Scratch {
    std::size_t bytes = 0;
};

struct alignas(std::max_align_t) AllocationHeader {
    std::size_t bytes;
};

voidpf Allocate(voidpf opaque, uInt count, uInt size) noexcept {
    auto& scratch = *static_cast<Scratch*>(opaque);
    if (size != 0 && count > (ScratchLimit - sizeof(AllocationHeader)) / size) return nullptr;
    const std::size_t bytes = static_cast<std::size_t>(count) * size + sizeof(AllocationHeader);
    if (bytes > ScratchLimit - scratch.bytes || !AllowAllocation()) return nullptr;
    auto* header = static_cast<AllocationHeader*>(std::calloc(1, bytes));
    if (header == nullptr) return nullptr;
    header->bytes = bytes;
    scratch.bytes += bytes;
#ifdef ANYPS5_COLD_SPIRV_TESTING
    ++allocations.active;
    allocations.bytes += bytes;
#endif
    return header + 1;
}

void Release(voidpf opaque, voidpf address) noexcept {
    if (address == nullptr) return;
    auto& scratch = *static_cast<Scratch*>(opaque);
    auto* header = static_cast<AllocationHeader*>(address) - 1;
    scratch.bytes -= header->bytes;
#ifdef ANYPS5_COLD_SPIRV_TESTING
    --allocations.active;
    allocations.bytes -= header->bytes;
#endif
    std::free(header);
}

struct Stream {
    Scratch scratch;
    z_stream value{};
    bool compress;
    bool initialized = false;

    explicit Stream(bool compression) noexcept : compress(compression) {
        value.zalloc = Allocate;
        value.zfree = Release;
        value.opaque = &scratch;
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    ~Stream() {
        if (!initialized) return;
        if (compress) deflateEnd(&value);
        else inflateEnd(&value);
    }
};

template<class T>
std::vector<T> Buffer(std::size_t size) {
    if (!AllowAllocation()) throw std::bad_alloc();
    return std::vector<T>(size);
}

}

std::optional<std::vector<std::byte>> Compress(std::span<const std::uint32_t> words, int level) noexcept {
    if (words.empty() || words.size() > MaxBytes / sizeof(std::uint32_t) || level < 1 || level > 9) {
        return std::nullopt;
    }
    try {
        const auto bytes = words.size_bytes();
        auto output = Buffer<std::byte>(bytes - 1);
        Stream stream(true);
        if (deflateInit(&stream.value, level) != Z_OK) return std::nullopt;
        stream.initialized = true;
        stream.value.next_in = reinterpret_cast<Bytef*>(const_cast<std::uint32_t*>(words.data()));
        stream.value.avail_in = static_cast<uInt>(bytes);
        stream.value.next_out = reinterpret_cast<Bytef*>(output.data());
        stream.value.avail_out = static_cast<uInt>(output.size());
        if (deflate(&stream.value, Z_FINISH) != Z_STREAM_END || stream.value.avail_in != 0 ||
            stream.value.total_out >= bytes) {
            return std::nullopt;
        }
        auto compact = Buffer<std::byte>(static_cast<std::size_t>(stream.value.total_out));
        std::memcpy(compact.data(), output.data(), compact.size());
        return compact;
    } catch (...) {
        return std::nullopt;
    }
}

std::vector<std::uint32_t> Decompress(std::span<const std::byte> bytes, std::size_t wordCount) {
    if (wordCount == 0 || wordCount > MaxBytes / sizeof(std::uint32_t) || bytes.empty() ||
        bytes.size() >= wordCount * sizeof(std::uint32_t)) {
        throw std::runtime_error("Cold SPIR-V has invalid compressed bounds");
    }
    auto output = Buffer<std::uint32_t>(wordCount);
    Stream stream(false);
    if (inflateInit(&stream.value) != Z_OK) throw std::runtime_error("Cold SPIR-V decoder initialization failed");
    stream.initialized = true;
    stream.value.next_in = reinterpret_cast<Bytef*>(const_cast<std::byte*>(bytes.data()));
    stream.value.avail_in = static_cast<uInt>(bytes.size());
    stream.value.next_out = reinterpret_cast<Bytef*>(output.data());
    stream.value.avail_out = static_cast<uInt>(output.size() * sizeof(std::uint32_t));
    if (inflate(&stream.value, Z_FINISH) != Z_STREAM_END || stream.value.avail_in != 0 ||
        stream.value.total_out != output.size() * sizeof(std::uint32_t)) {
        throw std::runtime_error("Cold SPIR-V stream, checksum or decoded size is invalid");
    }
    return output;
}

#ifdef ANYPS5_COLD_SPIRV_TESTING
namespace Testing {

void FailAllocationAfter(std::size_t successfulAllocationChecks) noexcept {
    allocations.remaining = successfulAllocationChecks;
    allocations.checks = 0;
}

void ResetAllocationFailure() noexcept {
    allocations.remaining = std::numeric_limits<std::size_t>::max();
    allocations.checks = 0;
}

std::size_t AllocationChecks() noexcept { return allocations.checks; }
std::size_t ActiveAllocations() noexcept { return allocations.active; }
std::size_t ActiveBytes() noexcept { return allocations.bytes; }

}
#endif

}
