#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_SHARECENSUS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_SHARECENSUS_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace ShaderRecompiler {
struct RecompileResult;
}

namespace AgcDriver::NewPath {
struct EmitCandidate;
}

namespace AgcDriver::NewPath::Census {

enum ShapeBit : std::uint32_t {
    AddressBased = 1u << 0,
    BuffersFromUser = 1u << 1,
    BuffersFromMemory = 1u << 2,
    BuffersWritten = 1u << 3,
    SampledImages = 1u << 4,
    StorageImages = 1u << 5,
    Samplers = 1u << 6,
    Bindless = 1u << 7,
    Gds = 1u << 8,
    Lds = 1u << 9,
    ShaderData = 1u << 10,
    TexelBuffers = 1u << 11,
    FlattenedSrt = 1u << 12,
};

[[nodiscard]] std::uint32_t Classify(const ShaderRecompiler::RecompileResult& compiled);
[[nodiscard]] std::string ShapeName(std::uint32_t shape);

enum class Repeat : std::uint8_t { Hit, Same, PushOnly, DataOnly, BasesOnly, Other, First };
inline constexpr std::size_t RepeatCount = 7;
[[nodiscard]] const char* RepeatName(Repeat repeat);

struct Bindings {
    std::uint64_t shape = 0;
    std::uint64_t push = 0;
    std::uint64_t data = 0;
    std::uint64_t bases = 0;
    std::uint64_t buffers = 0;
    std::uint64_t images = 0;
};
[[nodiscard]] Bindings HashBindings(const ShaderRecompiler::RecompileResult& compiled);
[[nodiscard]] Repeat Compare(const Bindings& previous, const Bindings& now);

enum class BufferCover : std::uint8_t { None, Covered, Missing, OtherAddress, OtherAddressWritten, Narrower, Unavailable };
inline constexpr std::size_t BufferCoverCount = 7;
[[nodiscard]] const char* BufferCoverName(BufferCover cover);

[[nodiscard]] bool Enabled();

void NoteDecode(bool cached, std::size_t capturedRegions, std::chrono::steady_clock::time_point started);
void Fill(EmitCandidate& candidate, const ShaderRecompiler::RecompileResult& compiled);
void NoteCommit(const EmitCandidate& candidate, bool deviceCall, std::uint64_t deviceNs, std::uint64_t buildNs);
void NoteUnclaimed();

[[nodiscard]] std::string Format(const char* window);

}

#endif
