#ifndef ANYPS5_COLD_SPIRV_CODEC_HPP
#define ANYPS5_COLD_SPIRV_CODEC_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ShaderRecompiler::ColdSpirv {

inline constexpr std::size_t MaxBytes = 64u * 1024u * 1024u;

std::optional<std::vector<std::byte>> Compress(std::span<const std::uint32_t> words, int level = 1) noexcept;
std::vector<std::uint32_t> Decompress(std::span<const std::byte> bytes, std::size_t wordCount);

}

#endif
