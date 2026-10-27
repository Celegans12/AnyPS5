#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BINDLESSREGISTRATIONS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BINDLESSREGISTRATIONS_HPP

#include "BindlessImageAbi.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

// The bindless image tables a texture service has registered, as 64-bit words: what a frame capture
// records (FrameCallKind::BindlessTables) and agc_frame_replay --bindless-out writes, so a replay can
// prime its heap with the tables the title's warm heap held before the captured frames (it starts
// empty otherwise). Layout: count, then per table base, records, stride, dwords, uses, then per use
// interfaceId, samplerClass and the sampler words.
namespace AgcDriver::Graphics {

using BindlessRegistration = ShaderRecompiler::BindlessImageAbi::TableRegistration;

inline std::vector<std::uint64_t> EncodeBindlessRegistrations(std::span<const BindlessRegistration> tables) {
    std::vector<std::uint64_t> words{tables.size()};
    for (const auto& table : tables) {
        words.insert(words.end(), {table.base, table.records, table.stride, table.dwords, table.uses.size()});
        for (const auto& use : table.uses) {
            words.push_back(use.interfaceId);
            words.push_back(use.samplerClass);
            for (const auto word : use.sampler) words.push_back(word);
        }
    }
    return words;
}

inline std::vector<BindlessRegistration> DecodeBindlessRegistrations(std::span<const std::uint64_t> words) {
    constexpr std::size_t SamplerWords = std::tuple_size_v<decltype(ShaderRecompiler::BindlessImageAbi::TableUse::sampler)>;
    std::size_t at = 0;
    const auto next = [&] {
        if (at >= words.size()) throw std::runtime_error("bindless registrations: truncated");
        return words[at++];
    };
    std::vector<BindlessRegistration> tables;
    if (words.empty()) return tables;
    const auto count = next();
    if (count > words.size()) throw std::runtime_error("bindless registrations: corrupt count");
    tables.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        BindlessRegistration table;
        table.base = next();
        table.records = static_cast<std::uint32_t>(next());
        table.stride = static_cast<std::uint32_t>(next());
        table.dwords = static_cast<std::uint32_t>(next());
        const auto uses = next();
        if (uses > words.size()) throw std::runtime_error("bindless registrations: corrupt use count");
        for (std::uint64_t use = 0; use < uses; ++use) {
            ShaderRecompiler::BindlessImageAbi::TableUse entry;
            entry.interfaceId = next();
            entry.samplerClass = static_cast<std::uint32_t>(next());
            for (std::size_t word = 0; word < SamplerWords; ++word) entry.sampler[word] = static_cast<std::uint32_t>(next());
            table.uses.push_back(entry);
        }
        tables.push_back(std::move(table));
    }
    if (at != words.size()) throw std::runtime_error("bindless registrations: trailing words");
    return tables;
}

}

#endif
