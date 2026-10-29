#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ShareCensus.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/NewPath.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace AgcDriver::NewPath::Census {

namespace {

struct Stats {
    std::uint64_t count = 0, cached = 0, decodeNs = 0, deviceCalls = 0, deviceNs = 0, buildNs = 0, capturedRegions = 0;
    std::array<std::uint64_t, RepeatCount> repeats{};
    std::array<std::uint64_t, BufferCoverCount> covers{};
};

struct ModuleKey {
    std::uint64_t low = 0, high = 0, variant = 0;
    bool operator==(const ModuleKey&) const = default;
};

struct ModuleKeyHash {
    std::size_t operator()(const ModuleKey& key) const { return static_cast<std::size_t>(key.low ^ (key.high * 0x9e3779b97f4a7c15ull) ^ (key.variant * 0xc2b2ae3d27d4eb4full)); }
};

struct State {
    std::mutex mutex;
    std::map<std::uint32_t, Stats> shapes;
    std::unordered_map<ModuleKey, Bindings, ModuleKeyHash> last;
    std::uint64_t unclaimed = 0;
};

State& Get() {
    static auto* state = new State;
    return *state;
}

struct Stash {
    bool valid = false;
    bool cached = false;
    std::size_t regions = 0;
    std::chrono::steady_clock::time_point started;
};

Stash& ThreadStash() {
    thread_local Stash stash;
    return stash;
}

std::uint64_t mix(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    return hash * 0x100000001b3ull;
}

}

bool Enabled() {
    static const bool enabled = [] {
        const auto set = [](const char* name) {
            const char* value = std::getenv(name);
            return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
        };
        return set("APS5_NEW_PATH") && set("APS5_PROFILE_NEW_PATH");
    }();
    return enabled;
}

std::uint32_t Classify(const ShaderRecompiler::RecompileResult& compiled) {
    std::uint32_t shape = 0;
    bool buffers = false;
    for (const auto& binding : compiled.bindings) {
        switch (binding.role) {
        case ShaderRecompiler::DescriptorRole::BdaPagetable: shape |= AddressBased; break;
        case ShaderRecompiler::DescriptorRole::GuestBuffers:
            buffers = true;
            if (binding.kind == ShaderRecompiler::DescriptorKind::UniformTexelBuffer || binding.kind == ShaderRecompiler::DescriptorKind::StorageTexelBuffer) shape |= TexelBuffers;
            if (!binding.readOnly || std::ranges::any_of(binding.bufferWritten, [](bool written) { return written; })) shape |= BuffersWritten;
            break;
        case ShaderRecompiler::DescriptorRole::GuestImages: shape |= binding.kind == ShaderRecompiler::DescriptorKind::StorageImage ? StorageImages : SampledImages; break;
        case ShaderRecompiler::DescriptorRole::GuestSamplers: shape |= Samplers; break;
        case ShaderRecompiler::DescriptorRole::Gds: shape |= Gds; break;
        case ShaderRecompiler::DescriptorRole::FlattenedSrt: shape |= FlattenedSrt; break;
        case ShaderRecompiler::DescriptorRole::ShaderData: shape |= ShaderData; break;
        case ShaderRecompiler::DescriptorRole::ImageTable:
        case ShaderRecompiler::DescriptorRole::SamplerTable:
        case ShaderRecompiler::DescriptorRole::ImageTableMap:
        case ShaderRecompiler::DescriptorRole::ImageStorageTable: shape |= Bindless; break;
        case ShaderRecompiler::DescriptorRole::FaultBuffer: break;
        }
    }
    if (compiled.bindlessHeap || !compiled.bindlessTables.empty()) shape |= Bindless;
    if (buffers) shape |= compiled.rangesFromMemory ? BuffersFromMemory : BuffersFromUser;
    if (compiled.workgroupMemoryDwords != 0) shape |= Lds;
    return shape;
}

std::string ShapeName(std::uint32_t shape) {
    static constexpr std::array<const char*, 13> names{"bda", "buf-user", "buf-mem", "buf-write", "sampled", "storage-img", "sampler", "bindless", "gds", "lds", "shader-data", "texel", "flat-srt"};
    std::string name;
    for (std::size_t bit = 0; bit < names.size(); ++bit) {
        if ((shape & (1u << bit)) == 0) continue;
        if (!name.empty()) name += '+';
        name += names[bit];
    }
    return name.empty() ? std::string("none") : name;
}

const char* RepeatName(Repeat repeat) {
    switch (repeat) {
    case Repeat::Hit: return "cache-hit";
    case Repeat::Same: return "same-inputs";
    case Repeat::PushOnly: return "push-only";
    case Repeat::DataOnly: return "srt-data-only";
    case Repeat::BasesOnly: return "buffer-bases";
    case Repeat::Other: return "descriptors";
    case Repeat::First: return "first";
    }
    return "unknown";
}

const char* BufferCoverName(BufferCover cover) {
    switch (cover) {
    case BufferCover::None: return "n/a";
    case BufferCover::Covered: return "covered";
    case BufferCover::Missing: return "missing";
    case BufferCover::OtherAddress: return "other-address";
    case BufferCover::OtherAddressWritten: return "other-address-writable";
    case BufferCover::Narrower: return "narrower";
    case BufferCover::Unavailable: return "unavailable";
    }
    return "unknown";
}

Bindings HashBindings(const ShaderRecompiler::RecompileResult& compiled) {
    Bindings hashes;
    for (auto* field : {&hashes.shape, &hashes.push, &hashes.data, &hashes.bases, &hashes.buffers, &hashes.images}) *field = 0xcbf29ce484222325ull;
    for (const auto& binding : compiled.bindings) {
        hashes.shape = mix(hashes.shape, static_cast<std::uint64_t>(binding.role) | (static_cast<std::uint64_t>(binding.binding) << 8u) | (static_cast<std::uint64_t>(binding.count) << 32u) | (static_cast<std::uint64_t>(binding.guestDescriptor.size()) << 48u));
        const bool buffer = binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers;
        const bool data = binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt || binding.role == ShaderRecompiler::DescriptorRole::ShaderData;
        for (std::size_t index = 0; index < binding.guestDescriptor.size(); ++index) {
            const auto word = binding.guestDescriptor[index];
            if (data) {
                hashes.data = mix(hashes.data, word);
            } else if (buffer) {
                const auto field = index % 4u;
                hashes.bases = mix(hashes.bases, field == 0 ? word : field == 1 ? (word & 0xffffu) : 0u);
                hashes.buffers = mix(hashes.buffers, field == 0 ? 0u : field == 1 ? (word & 0xffff0000u) : word);
            } else {
                hashes.images = mix(hashes.images, word);
            }
        }
    }
    for (const auto& constant : compiled.specialization) hashes.shape = mix(hashes.shape, constant.id | (static_cast<std::uint64_t>(constant.value) << 32u));
    for (const auto byte : compiled.pushConstants) hashes.push = mix(hashes.push, static_cast<std::uint64_t>(byte));
    hashes.push = mix(hashes.push, compiled.pushConstants.size());
    return hashes;
}

Repeat Compare(const Bindings& previous, const Bindings& now) {
    if (previous.shape != now.shape || previous.buffers != now.buffers || previous.images != now.images) return Repeat::Other;
    if (previous.bases != now.bases) return Repeat::BasesOnly;
    if (previous.data != now.data) return Repeat::DataOnly;
    if (previous.push != now.push) return Repeat::PushOnly;
    return Repeat::Same;
}

void NoteDecode(bool cached, std::size_t capturedRegions, std::chrono::steady_clock::time_point started) {
    if (!Enabled()) return;
    auto& stash = ThreadStash();
    stash.valid = true;
    stash.cached = cached;
    stash.regions = capturedRegions;
    stash.started = started;
}

void Fill(EmitCandidate& candidate, const ShaderRecompiler::RecompileResult& compiled) {
    if (!Enabled()) return;
    auto& stash = ThreadStash();
    candidate.census.active = true;
    candidate.census.shape = Classify(compiled);
    candidate.census.cached = stash.valid && stash.cached;
    candidate.census.regions = stash.valid ? static_cast<std::uint32_t>(stash.regions) : 0u;
    candidate.census.decodeNs = stash.valid ? static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - stash.started).count()) : 0u;
    stash.valid = false;
    auto repeat = Repeat::Hit;
    if (!candidate.census.cached) {
        const auto hash = compiled.spirv.ContentHash();
        const ModuleKey key{hash.low, hash.high, compiled.PipelineVariantId()};
        const auto bindings = HashBindings(compiled);
        auto& state = Get();
        std::lock_guard lock(state.mutex);
        const auto found = state.last.find(key);
        repeat = found == state.last.end() ? Repeat::First : Compare(found->second, bindings);
        if (found == state.last.end() && state.last.size() > 65536u) state.last.clear();
        state.last[key] = bindings;
    }
    candidate.census.repeat = static_cast<std::uint8_t>(repeat);
}

void NoteCommit(const EmitCandidate& candidate, bool deviceCall, std::uint64_t deviceNs, std::uint64_t buildNs) {
    if (!candidate.census.active) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    auto& stats = state.shapes[candidate.census.shape | (candidate.twin != nullptr ? 0x80000000u : 0u)];
    ++stats.count;
    if (candidate.census.cached) ++stats.cached;
    stats.decodeNs += candidate.census.decodeNs;
    stats.capturedRegions += candidate.census.regions;
    ++stats.repeats[candidate.census.repeat];
    ++stats.covers[std::min<std::size_t>(candidate.census.cover, BufferCoverCount - 1)];
    if (deviceCall) {
        ++stats.deviceCalls;
        stats.deviceNs += deviceNs;
        stats.buildNs += buildNs;
    }
}

void NoteUnclaimed() {
    if (!Enabled()) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    ++state.unclaimed;
}

std::string Format(const char* window) {
    auto& state = Get();
    std::vector<std::pair<std::uint32_t, Stats>> rows;
    std::uint64_t unclaimed = 0;
    {
        std::lock_guard lock(state.mutex);
        rows.assign(state.shapes.begin(), state.shapes.end());
        unclaimed = state.unclaimed;
    }
    const auto cost = [](const Stats& stats) { return stats.decodeNs + stats.deviceNs; };
    std::ranges::sort(rows, [&](const auto& left, const auto& right) { return cost(left.second) > cost(right.second); });
    Stats twin, other;
    for (const auto& [shape, stats] : rows) {
        auto& sum = (shape & 0x80000000u) != 0 ? twin : other;
        sum.count += stats.count;
        sum.decodeNs += stats.decodeNs;
        sum.deviceNs += stats.deviceNs;
        for (std::size_t cover = 0; cover < BufferCoverCount; ++cover) sum.covers[cover] += stats.covers[cover];
        if ((shape & 0x80000000u) == 0 && (shape & (SampledImages | StorageImages | Samplers | Bindless)) == 0) sum.cached += stats.covers[static_cast<std::size_t>(BufferCover::Covered)];
    }
    char head[320];
    std::snprintf(head, sizeof(head), "[newpath-share] window=%s claimed direct dispatches by shape (cumulative; ms = decode up to the claim + device call): with twin %llu (%.1f ms), without twin %llu (%.1f ms), unclaimed %llu", window, static_cast<unsigned long long>(twin.count), static_cast<double>(cost(twin)) / 1e6, static_cast<unsigned long long>(other.count), static_cast<double>(cost(other)) / 1e6, static_cast<unsigned long long>(unclaimed));
    std::string line = head;
    line += "; without twin, buffers F4:";
    for (std::size_t cover = 1; cover < BufferCoverCount; ++cover) line += std::string(" ") + BufferCoverName(static_cast<BufferCover>(cover)) + " " + std::to_string(other.covers[cover]);
    line += " (covered without images " + std::to_string(other.cached) + ")";
    std::size_t printed = 0;
    for (const auto& [shape, stats] : rows) {
        if (printed == 16) break;
        ++printed;
        char text[512];
        std::snprintf(text, sizeof(text), "; %s%s: %llu (cached %llu, regions %.1f), decode %.1f us, device %llu x %.1f us (resources %.1f us), %.1f ms; repeat", (shape & 0x80000000u) != 0 ? "twin:" : "", ShapeName(shape & 0x7fffffffu).c_str(), static_cast<unsigned long long>(stats.count), static_cast<unsigned long long>(stats.cached), stats.count != 0 ? static_cast<double>(stats.capturedRegions) / static_cast<double>(stats.count) : 0.0, stats.count != 0 ? static_cast<double>(stats.decodeNs) / 1e3 / static_cast<double>(stats.count) : 0.0, static_cast<unsigned long long>(stats.deviceCalls), stats.deviceCalls != 0 ? static_cast<double>(stats.deviceNs) / 1e3 / static_cast<double>(stats.deviceCalls) : 0.0, stats.deviceCalls != 0 ? static_cast<double>(stats.buildNs) / 1e3 / static_cast<double>(stats.deviceCalls) : 0.0, static_cast<double>(cost(stats)) / 1e6);
        line += text;
        for (std::size_t repeat = 0; repeat < RepeatCount; ++repeat) {
            if (stats.repeats[repeat] != 0) line += std::string(" ") + RepeatName(static_cast<Repeat>(repeat)) + " " + std::to_string(stats.repeats[repeat]);
        }
        if (stats.covers[0] != stats.count) {
            line += "; buffers F4";
            for (std::size_t cover = 1; cover < BufferCoverCount; ++cover) {
                if (stats.covers[cover] != 0) line += std::string(" ") + BufferCoverName(static_cast<BufferCover>(cover)) + " " + std::to_string(stats.covers[cover]);
            }
        }
    }
    if (rows.size() > printed) line += "; " + std::to_string(rows.size() - printed) + " more shapes";
    return line;
}

}
