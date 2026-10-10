#include "ColdSpirvCodec.hpp"
#include "ColdSpirvCodecTestHooks.hpp"

#include <algorithm>
#include <array>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <iostream>
#include <latch>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace Codec = ShaderRecompiler::ColdSpirv;
namespace Testing = ShaderRecompiler::ColdSpirv::Testing;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void RequireNoAllocations() {
    Require(Testing::ActiveAllocations() == 0, "codec leaked a zlib allocation");
    Require(Testing::ActiveBytes() == 0, "codec leaked zlib bytes");
}

std::vector<std::uint32_t> Fixture(std::uint32_t seed = 0) {
    std::vector<std::uint32_t> words(16384);
    for (std::size_t index = 0; index < words.size(); ++index) {
        words[index] = 0x00040000u | (static_cast<std::uint32_t>(index % 32) + seed);
    }
    words.front() = 0x07230203u;
    return words;
}

std::vector<std::byte> Encode(const std::vector<std::uint32_t>& words, int level) {
    auto encoded = Codec::Compress(words, level);
    Require(encoded.has_value(), "compressible fixture fell back to raw");
    Require(!encoded->empty() && encoded->size() < words.size() * sizeof(words.front()), "compressed fixture did not save bytes");
    RequireNoAllocations();
    return std::move(*encoded);
}

void ExpectDecodeFailure(std::span<const std::byte> bytes, std::size_t wordCount, const char* message) {
    const std::vector<std::byte> original(bytes.begin(), bytes.end());
    bool failed = false;
    try {
        static_cast<void>(Codec::Decompress(bytes, wordCount));
    } catch (const std::exception&) {
        failed = true;
    }
    Require(failed, message);
    Require(std::equal(bytes.begin(), bytes.end(), original.begin(), original.end()), "failed decoding changed the cold input");
    RequireNoAllocations();
}

void RoundTripsAndFallbacks() {
    Testing::ResetAllocationFailure();
    const auto words = Fixture();
    const auto original = words;
    for (const int level : {1, 6}) {
        const auto encoded = Encode(words, level);
        const auto coldOriginal = encoded;
        Require(Codec::Decompress(encoded, words.size()) == words, "decoded words differ from the original");
        Require(encoded == coldOriginal, "successful decoding changed the cold input");
        Require(words == original, "compression changed the raw input");
        RequireNoAllocations();
    }

    Require(!Codec::Compress(std::span<const std::uint32_t>{}).has_value(), "empty input did not fall back to raw");
    for (const int level : {-2, 10}) {
        Require(!Codec::Compress(words, level).has_value(), "invalid compression level did not fall back to raw");
        Require(words == original, "invalid-level compression changed the raw input");
        RequireNoAllocations();
    }

    std::vector<std::uint32_t> random(65536);
    std::uint32_t state = 0x9e3779b9u;
    for (auto& word : random) {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        word = state;
    }
    const auto randomOriginal = random;
    for (const int level : {1, 6}) {
        Require(!Codec::Compress(random, level).has_value(), "incompressible input did not fall back to raw");
        Require(random == randomOriginal, "incompressible input changed during fallback");
        RequireNoAllocations();
    }

    const std::vector<std::uint32_t> oversized(Codec::MaxBytes / sizeof(std::uint32_t) + 1, 0x6a09e667u);
    Testing::ResetAllocationFailure();
    Require(!Codec::Compress(oversized).has_value(), "oversized input did not fall back to raw");
    Require(Testing::AllocationChecks() == 0, "oversized input attempted a codec allocation");
    Require(std::all_of(oversized.begin(), oversized.end(), [](std::uint32_t word) { return word == 0x6a09e667u; }), "oversized fallback changed its raw input");
    RequireNoAllocations();
}

void MalformedStreams() {
    Testing::ResetAllocationFailure();
    const auto words = Fixture();
    const auto encoded = Encode(words, 1);
    Require(encoded.size() > 8, "fixture stream is too short for corruption tests");
    ExpectDecodeFailure(encoded, 0, "zero decoded length was accepted");
    ExpectDecodeFailure(encoded, words.size() - 1, "short decoded length was accepted");
    ExpectDecodeFailure(encoded, words.size() + 1, "long decoded length was accepted");
    Testing::ResetAllocationFailure();
    ExpectDecodeFailure(encoded, Codec::MaxBytes / sizeof(std::uint32_t) + 1, "oversized decoded length was accepted");
    Require(Testing::AllocationChecks() == 0, "oversized decoded length attempted a codec allocation");
    ExpectDecodeFailure(encoded, std::numeric_limits<std::size_t>::max(), "overflowing decoded length was accepted");
    for (const auto length : {std::size_t{0}, std::size_t{1}, std::size_t{2}, encoded.size() / 2, encoded.size() - 1}) {
        ExpectDecodeFailure(std::span<const std::byte>(encoded).first(length), words.size(), "truncated stream was accepted");
    }

    auto corrupt = encoded;
    corrupt.front() ^= std::byte{0xff};
    ExpectDecodeFailure(corrupt, words.size(), "corrupt zlib header was accepted");
    corrupt = encoded;
    corrupt[corrupt.size() / 2] ^= std::byte{0xff};
    ExpectDecodeFailure(corrupt, words.size(), "corrupt compressed payload was accepted");
    corrupt = encoded;
    corrupt.back() ^= std::byte{1};
    ExpectDecodeFailure(corrupt, words.size(), "corrupt zlib checksum was accepted");
    corrupt = encoded;
    corrupt.push_back(std::byte{0});
    ExpectDecodeFailure(corrupt, words.size(), "trailing garbage was accepted");
    corrupt = encoded;
    corrupt.insert(corrupt.end(), encoded.begin(), encoded.end());
    ExpectDecodeFailure(corrupt, words.size(), "concatenated streams were accepted");
    Require(Codec::Decompress(encoded, words.size()) == words, "valid stream failed after malformed streams");
    RequireNoAllocations();
}

void AllocationFailures() {
    const auto words = Fixture();
    const auto original = words;
    for (const int level : {1, 6}) {
        Testing::ResetAllocationFailure();
        const auto encoded = Encode(words, level);
        const auto compressionChecks = Testing::AllocationChecks();
        Require(compressionChecks > 1, "compression did not exercise vector and zlib allocation checks");
        for (std::size_t failAfter = 0; failAfter < compressionChecks; ++failAfter) {
            Testing::FailAllocationAfter(failAfter);
            const auto failed = Codec::Compress(words, level);
            Require(!failed.has_value(), "allocation failure did not preserve raw fallback");
            Require(Testing::AllocationChecks() > failAfter, "compression did not reach the injected allocation failure");
            Require(words == original, "failed compression changed raw input");
            RequireNoAllocations();
            Testing::ResetAllocationFailure();
            const auto retry = Encode(words, level);
            Require(Codec::Decompress(retry, words.size()) == words, "compression retry did not round trip");
            RequireNoAllocations();
        }

        Testing::ResetAllocationFailure();
        Require(Codec::Decompress(encoded, words.size()) == words, "baseline allocation decode failed");
        const auto decompressionChecks = Testing::AllocationChecks();
        Require(decompressionChecks > 1, "decoding did not exercise vector and zlib allocation checks");
        RequireNoAllocations();
        for (std::size_t failAfter = 0; failAfter < decompressionChecks; ++failAfter) {
            Testing::FailAllocationAfter(failAfter);
            ExpectDecodeFailure(encoded, words.size(), "injected decode allocation failure was accepted");
            Require(Testing::AllocationChecks() > failAfter, "decoding did not reach the injected allocation failure");
            Testing::ResetAllocationFailure();
            Require(Codec::Decompress(encoded, words.size()) == words, "decode retry did not round trip");
            RequireNoAllocations();
        }
        Testing::ResetAllocationFailure();
        ExpectDecodeFailure(encoded, words.size() / 2, "output exhaustion was accepted");
        const auto partialChecks = Testing::AllocationChecks();
        Require(partialChecks > decompressionChecks, "malformed decode did not exercise lazy window allocation");
        for (std::size_t failAfter = 0; failAfter < partialChecks; ++failAfter) {
            Testing::FailAllocationAfter(failAfter);
            ExpectDecodeFailure(encoded, words.size() / 2, "failed partial decode was accepted");
            Require(Testing::AllocationChecks() > failAfter, "partial decoding did not reach the injected allocation failure");
            Testing::ResetAllocationFailure();
            Require(Codec::Decompress(encoded, words.size()) == words, "decode failed after partial allocation failure");
            RequireNoAllocations();
        }
    }
    Testing::ResetAllocationFailure();
}

void ConcurrentCalls() {
    std::array<std::vector<std::uint32_t>, 4> inputs;
    for (std::size_t index = 0; index < inputs.size(); ++index) inputs[index] = Fixture(static_cast<std::uint32_t>(index * 64));
    std::barrier rendezvous(static_cast<std::ptrdiff_t>(inputs.size()));
    std::latch launchReady(1);
    bool launchFailed = false;
    std::array<std::future<void>, 4> workers;
    try {
        for (std::size_t index = 0; index < workers.size(); ++index) {
            workers[index] = std::async(std::launch::async, [&, index] {
                launchReady.wait();
                if (launchFailed) return;
                if (index == 0) Testing::FailAllocationAfter(0);
                else Testing::ResetAllocationFailure();
                rendezvous.arrive_and_wait();
                const auto first = Codec::Compress(inputs[index]);
                const auto allocations = Testing::ActiveAllocations();
                const auto bytes = Testing::ActiveBytes();
                rendezvous.arrive_and_wait();
                Testing::ResetAllocationFailure();
                Require(first.has_value() == (index != 0), "allocation failure state crossed thread boundaries");
                Require(allocations == 0 && bytes == 0, "concurrent compression leaked zlib allocations");
                for (std::size_t repetition = 0; repetition < 8; ++repetition) {
                    const auto encoded = Encode(inputs[index], repetition % 2 == 0 ? 1 : 6);
                    Require(Codec::Decompress(encoded, inputs[index].size()) == inputs[index], "concurrent round trip changed words");
                    RequireNoAllocations();
                }
            });
        }
    } catch (...) {
        launchFailed = true;
        launchReady.count_down();
        throw;
    }
    launchReady.count_down();
    for (auto& worker : workers) worker.get();
    RequireNoAllocations();
}

}

int main() {
    try {
        ConcurrentCalls();
        RoundTripsAndFallbacks();
        MalformedStreams();
        AllocationFailures();
        std::cout << "cold SPIR-V codec tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        Testing::ResetAllocationFailure();
        std::cerr << error.what() << '\n';
        return 1;
    }
}
