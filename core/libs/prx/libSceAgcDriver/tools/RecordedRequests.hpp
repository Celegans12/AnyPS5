#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TOOLS_RECORDEDREQUESTS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TOOLS_RECORDEDREQUESTS_HPP

// Reading recorded recompile requests in the offline tools (agc_shader_replay, the request bench).
//
// Serialization versions 13-16 were numbered twice: the tree that wrote the 2026-10-09 request corpus used them for
// fields of its own, while the reader follows the upstream numbering, in which v13 adds the guest context's optional
// float mode (a bool after the memory regions) and v16 adds the target's float64DenormPreserve (a trailing bool).
// A compute request from that tree has neither byte, and the reader stops with "truncated data". Measured on the
// corpus (research/speed/newpath/census-c.md): after its memory regions such a request goes straight on to the target,
// and its tail ends at narrowSubgroupClock (v12). The tools insert the two bools as false (neither field existed when
// the request was written) and read it again. Only compute requests are repaired; graphics requests of that tree
// carry pixel or vertex fields under the same numbers and are left as they are. The recompiler's own reader, the shader
// cache key and the game never see this path.

#include "ControlFlow/RequestSerializer.hpp"
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace AgcDriver::RecordedRequests {

namespace Detail {

inline constexpr char Alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

inline std::uint32_t Value(char c) {
    if (c >= 'A' && c <= 'Z') return static_cast<std::uint32_t>(c - 'A');
    if (c >= 'a' && c <= 'z') return static_cast<std::uint32_t>(c - 'a' + 26);
    if (c >= '0' && c <= '9') return static_cast<std::uint32_t>(c - '0' + 52);
    if (c == '+') return 62u;
    if (c == '/') return 63u;
    throw std::runtime_error("recorded request: invalid base64 character");
}

inline std::string Decode(std::string_view text) {
    if (text.size() % 4u != 0u) throw std::runtime_error("recorded request: invalid base64 length");
    std::string bytes;
    bytes.reserve(text.size() / 4u * 3u);
    for (std::size_t i = 0; i < text.size(); i += 4u) {
        const auto chunk = (Value(text[i]) << 18u) | (Value(text[i + 1u]) << 12u) | ((text[i + 2u] == '=' ? 0u : Value(text[i + 2u])) << 6u) | (text[i + 3u] == '=' ? 0u : Value(text[i + 3u]));
        bytes.push_back(static_cast<char>(chunk >> 16u));
        if (text[i + 2u] != '=') bytes.push_back(static_cast<char>((chunk >> 8u) & 0xffu));
        if (text[i + 3u] != '=') bytes.push_back(static_cast<char>(chunk & 0xffu));
    }
    return bytes;
}

inline std::string Encode(std::string_view bytes) {
    std::string text;
    text.reserve((bytes.size() + 2u) / 3u * 4u);
    for (std::size_t i = 0; i < bytes.size(); i += 3u) {
        const auto left = bytes.size() - i;
        const auto b0 = static_cast<std::uint8_t>(bytes[i]);
        const auto b1 = left > 1u ? static_cast<std::uint8_t>(bytes[i + 1u]) : 0u;
        const auto b2 = left > 2u ? static_cast<std::uint8_t>(bytes[i + 2u]) : 0u;
        const auto chunk = (std::uint32_t{b0} << 16u) | (std::uint32_t{b1} << 8u) | b2;
        text.push_back(Alphabet[(chunk >> 18u) & 0x3fu]);
        text.push_back(Alphabet[(chunk >> 12u) & 0x3fu]);
        text.push_back(left > 1u ? Alphabet[(chunk >> 6u) & 0x3fu] : '=');
        text.push_back(left > 2u ? Alphabet[chunk & 0x3fu] : '=');
    }
    return text;
}

// Little-endian reads with bounds; nullopt when the stream ends first.
class Cursor {
public:
    explicit Cursor(std::string_view bytes) : m_bytes(bytes) {}
    std::optional<std::uint64_t> Take(std::size_t width) {
        if (m_bytes.size() - m_offset < width) return std::nullopt;
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < width; ++i) value |= std::uint64_t{static_cast<std::uint8_t>(m_bytes[m_offset + i])} << (8u * i);
        m_offset += width;
        return value;
    }
    bool Skip(std::uint64_t count) {
        if (m_bytes.size() - m_offset < count) return false;
        m_offset += static_cast<std::size_t>(count);
        return true;
    }
    std::size_t Offset() const { return m_offset; }

private:
    std::string_view m_bytes;
    std::size_t m_offset = 0;
};

inline std::uint32_t Version(std::string_view bytes) {
    Cursor cursor(bytes);
    if (!cursor.Take(4u)) return 0u;
    return static_cast<std::uint32_t>(cursor.Take(4u).value_or(0u));
}

// The offset just after the memory regions of a standalone compute request (shader binary, wave size, user data
// base, user data, compute info, no pixel or vertex info, regions), or nullopt for anything else.
inline std::optional<std::size_t> ComputeRegionsEnd(std::string_view bytes) {
    Cursor cursor(bytes);
    if (!cursor.Skip(8u) || !cursor.Skip(1u) || !cursor.Skip(8u)) return std::nullopt;
    const auto code = cursor.Take(8u);
    if (!code || *code > bytes.size() || !cursor.Skip(*code * 4u) || !cursor.Skip(8u)) return std::nullopt;
    const auto header = cursor.Take(8u);
    if (!header || !cursor.Skip(*header) || !cursor.Skip(8u)) return std::nullopt;
    const auto userData = cursor.Take(8u);
    if (!userData || *userData > bytes.size() || !cursor.Skip(*userData * 4u)) return std::nullopt;
    if (cursor.Take(1u).value_or(0u) != 1u || !cursor.Skip(12u + 4u + 3u + 1u + 4u)) return std::nullopt;
    if (cursor.Take(1u).value_or(1u) != 0u || cursor.Take(1u).value_or(1u) != 0u) return std::nullopt;
    const auto regions = cursor.Take(8u);
    if (!regions || *regions > bytes.size()) return std::nullopt;
    for (std::uint64_t i = 0; i < *regions; ++i) {
        const auto length = cursor.Skip(8u) ? cursor.Take(8u) : std::nullopt;
        if (!length || !cursor.Skip(*length)) return std::nullopt;
    }
    return cursor.Offset();
}

}

// RequestSerializer::Deserialize, plus the repair of a version-16 compute request from the 2026-10-09 corpus tree
// (see above). Any other failure is rethrown unchanged.
inline ShaderRecompiler::DeserializedRequest Deserialize(std::string_view text, bool* repaired = nullptr) {
    if (repaired != nullptr) *repaired = false;
    try {
        return ShaderRecompiler::RequestSerializer{}.Deserialize(text);
    } catch (const std::runtime_error& error) {
        if (std::string_view(error.what()) != "RequestSerializer: truncated data") throw;
        auto bytes = Detail::Decode(text);
        const auto regionsEnd = Detail::Version(bytes) == 16u ? Detail::ComputeRegionsEnd(bytes) : std::nullopt;
        if (!regionsEnd) throw;
        bytes.insert(*regionsEnd, 1u, '\0');
        bytes.push_back('\0');
        auto request = ShaderRecompiler::RequestSerializer{}.Deserialize(Detail::Encode(bytes));
        if (repaired != nullptr) *repaired = true;
        return request;
    }
}

}

#endif
