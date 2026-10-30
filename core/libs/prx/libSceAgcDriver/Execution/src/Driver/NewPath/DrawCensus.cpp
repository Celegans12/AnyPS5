#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DrawCensus.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace AgcDriver::NewPath::DrawCensus {

namespace {

constexpr std::array<const char*, PhaseCount> PhaseNames{"validate", "vertex", "setup", "readTarget", "prepare", "lookup", "resources", "pipeline", "record", "keep", "sync", "writeBack", "describe"};
constexpr std::array<const char*, KindCount> KindNames{"recipe", "template", "build", "bda"};
constexpr std::array<const char*, ShapeBitCount> ShapeNames{"bda", "buf", "buf-write", "sampled", "storage-img", "sampler", "bindless", "flat-srt", "shader-data", "texel", "gds", "ranges-from-mem", "unbounded", "bda-write"};
constexpr std::array<const char*, StageBitCount> StageNames{"vs", "ls/es", "hs", "ds", "gs", "ps", "ms", "cs"};

std::size_t pushBucket(std::uint32_t end) {
    if (end == 0) return 0;
    if (end <= 64) return 1;
    if (end <= 128) return 2;
    if (end <= PushBlockBytes - ResolvedPrefixBytes) return 3;
    if (end <= PushBlockBytes) return 4;
    return 5;
}

std::size_t wordBucket(std::uint32_t words) {
    if (words == 0) return 0;
    if (words <= 8) return 1;
    if (words <= 16) return 2;
    if (words <= 32) return 3;
    if (words <= 48) return 4;
    return 5;
}

void appendf(std::string& line, const char* format, auto... args) {
    char text[512];
    const int n = std::snprintf(text, sizeof(text), format, args...);
    if (n > 0) line.append(text, static_cast<std::size_t>(std::min<int>(n, static_cast<int>(sizeof(text) - 1))));
}

}

const char* CoverName(Cover cover) {
    switch (cover) {
    case Cover::None: return "n/a";
    case Cover::Covered: return "covered";
    case Cover::Missing: return "missing";
    case Cover::OtherAddress: return "other-address";
    case Cover::Narrower: return "narrower";
    case Cover::NoTable: return "no-table";
    }
    return "unknown";
}

StageFacts Facts(const ShaderRecompiler::RecompileResult& program, std::uint32_t pushOffset) {
    StageFacts facts;
    facts.userWords = static_cast<std::uint32_t>(std::popcount(program.userDataRegisterReads));
    facts.pushEnd = program.pushConstants.empty() ? 0u : pushOffset + static_cast<std::uint32_t>(program.pushConstants.size());
    facts.vertexInputs = static_cast<std::uint32_t>(program.vertexInputs.size());
    auto& shape = facts.shape;
    for (const auto& binding : program.bindings) {
        switch (binding.role) {
        case ShaderRecompiler::DescriptorRole::BdaPagetable: shape |= Bda; break;
        case ShaderRecompiler::DescriptorRole::GuestBuffers:
            shape |= Buffers;
            if (binding.kind == ShaderRecompiler::DescriptorKind::UniformTexelBuffer || binding.kind == ShaderRecompiler::DescriptorKind::StorageTexelBuffer) shape |= TexelBuffers;
            if (!binding.readOnly || std::ranges::any_of(binding.bufferWritten, [](bool written) { return written; })) shape |= BuffersWritten;
            break;
        case ShaderRecompiler::DescriptorRole::GuestImages: shape |= binding.kind == ShaderRecompiler::DescriptorKind::StorageImage ? StorageImages : Sampled; break;
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
    if (program.bdaAbiVersion != 0) shape |= Bda;
    if (program.bindlessHeap || !program.bindlessTables.empty()) shape |= Bindless;
    if (program.rangesFromMemory) shape |= RangesFromMemory;
    if (program.unboundedReads || program.unboundedWrites) shape |= Unbounded;
    if (program.bdaWrites) shape |= BdaWrites;
    return facts;
}

std::uint32_t StageBitOf(ShaderRecompiler::ShaderStage stage) {
    switch (stage) {
    case ShaderRecompiler::ShaderStage::Vertex: return StageVertex;
    case ShaderRecompiler::ShaderStage::Local: return StageLocal;
    case ShaderRecompiler::ShaderStage::TessellationControl: return StageHull;
    case ShaderRecompiler::ShaderStage::TessellationEvaluation: return StageDomain;
    case ShaderRecompiler::ShaderStage::Geometry: return StageGeometry;
    case ShaderRecompiler::ShaderStage::Fragment: return StagePixel;
    case ShaderRecompiler::ShaderStage::Mesh: return StageMesh;
    case ShaderRecompiler::ShaderStage::Compute: return StageCompute;
    }
    return 0;
}

void AddStage(DrawFacts& draw, ShaderRecompiler::ShaderStage stage, const StageFacts& facts) {
    draw.shape |= facts.shape;
    draw.stages |= StageBitOf(stage);
    draw.userWords += facts.userWords;
    draw.pushEnd = std::max(draw.pushEnd, facts.pushEnd);
    draw.vertexInputs += facts.vertexInputs;
}

std::string ShapeName(std::uint32_t shape) {
    std::string name;
    for (std::size_t bit = 0; bit < ShapeNames.size(); ++bit) {
        if ((shape & (1u << bit)) == 0) continue;
        if (!name.empty()) name += '+';
        name += ShapeNames[bit];
    }
    return name.empty() ? std::string("none") : name;
}

std::string StagesName(std::uint32_t stages) {
    std::string name;
    for (std::size_t bit = 0; bit < StageNames.size(); ++bit) {
        if ((stages & (1u << bit)) == 0) continue;
        if (!name.empty()) name += '/';
        name += StageNames[bit];
    }
    return name.empty() ? std::string("none") : name;
}

bool Mapped(std::span<const ShaderRecompiler::BdaAbi::Range> table, std::uint64_t begin, std::uint64_t end) {
    if (end <= begin) return true;
    auto cursor = begin;
    auto at = std::upper_bound(table.begin(), table.end(), cursor, [](std::uint64_t value, const ShaderRecompiler::BdaAbi::Range& entry) { return value < entry.end; });
    while (cursor < end) {
        if (at == table.end() || at->begin > cursor) return false;
        cursor = std::min(end, at->end);
        ++at;
    }
    return true;
}

void Window::Add(std::size_t kind, const DrawFacts& draw, std::span<const double> phaseUs) {
    if (kind >= KindCount) kind = 0;
    ++draws;
    ++kindDraws[kind];
    double us = 0;
    for (std::size_t phase = 0; phase < PhaseCount && phase < phaseUs.size(); ++phase) {
        kindPhaseUs[kind][phase] += phaseUs[phase];
        us += phaseUs[phase];
    }
    if (draw.indexed) ++indexed;
    if (draw.indirect) ++indirect;
    if (draw.instanced) ++instanced;
    const bool fits = draw.pushEnd + ResolvedPrefixBytes <= PushBlockBytes;
    if (fits) ++pushFits;
    ++pushBuckets[pushBucket(draw.pushEnd)];
    ++wordBuckets[wordBucket(draw.userWords)];
    ++covers[0][static_cast<std::size_t>(draw.bda)];
    ++covers[1][static_cast<std::size_t>(draw.index)];
    ++covers[2][static_cast<std::size_t>(draw.arguments)];
    auto& stats = shapes[{draw.shape, draw.stages}];
    ++stats.draws;
    stats.us += us;
    ++stats.kinds[kind];
    if (fits) ++stats.pushFits;
    if (draw.bda == Cover::Covered) ++stats.bdaCovered;
}

std::string Window::Format(const char* window, std::size_t topShapes) const {
    std::string line;
    appendf(line, "[newpath-draws] window=%s draws %llu (indexed %llu, indirect %llu, instanced %llu); by kind (count x committer us):", window, static_cast<unsigned long long>(draws), static_cast<unsigned long long>(indexed), static_cast<unsigned long long>(indirect), static_cast<unsigned long long>(instanced));
    for (std::size_t kind = 0; kind < KindCount; ++kind) {
        double us = 0;
        for (const auto phase : kindPhaseUs[kind]) us += phase;
        appendf(line, " %s %llu x %.1f", KindNames[kind], static_cast<unsigned long long>(kindDraws[kind]), kindDraws[kind] != 0 ? us / static_cast<double>(kindDraws[kind]) : 0.0);
    }
    line += "; us per draw by kind and phase:";
    for (std::size_t kind = 0; kind < KindCount; ++kind) {
        if (kindDraws[kind] == 0) continue;
        appendf(line, " %s{", KindNames[kind]);
        bool first = true;
        for (std::size_t phase = 0; phase < PhaseCount; ++phase) {
            const auto us = kindPhaseUs[kind][phase] / static_cast<double>(kindDraws[kind]);
            if (us < 0.05) continue;
            appendf(line, "%s%s %.1f", first ? "" : " ", PhaseNames[phase], us);
            first = false;
        }
        line += '}';
    }
    appendf(line, "; push end bytes 0:%llu 1-64:%llu 65-128:%llu 129-232:%llu 233-256:%llu 257+:%llu (resolved prefix fits %llu)", static_cast<unsigned long long>(pushBuckets[0]), static_cast<unsigned long long>(pushBuckets[1]), static_cast<unsigned long long>(pushBuckets[2]), static_cast<unsigned long long>(pushBuckets[3]), static_cast<unsigned long long>(pushBuckets[4]), static_cast<unsigned long long>(pushBuckets[5]), static_cast<unsigned long long>(pushFits));
    appendf(line, "; user words per draw 0:%llu 1-8:%llu 9-16:%llu 17-32:%llu 33-48:%llu 49+:%llu", static_cast<unsigned long long>(wordBuckets[0]), static_cast<unsigned long long>(wordBuckets[1]), static_cast<unsigned long long>(wordBuckets[2]), static_cast<unsigned long long>(wordBuckets[3]), static_cast<unsigned long long>(wordBuckets[4]), static_cast<unsigned long long>(wordBuckets[5]));
    static constexpr std::array<const char*, 3> coverNames{"bda table F4", "index buffer mapped", "indirect arguments mapped"};
    for (std::size_t which = 0; which < covers.size(); ++which) {
        appendf(line, "; %s:", coverNames[which]);
        for (std::size_t cover = 1; cover < CoverCount; ++cover) {
            if (covers[which][cover] == 0) continue;
            appendf(line, " %s %llu", CoverName(static_cast<Cover>(cover)), static_cast<unsigned long long>(covers[which][cover]));
        }
    }
    std::vector<std::pair<const std::pair<std::uint32_t, std::uint32_t>*, const ShapeStats*>> ranked;
    ranked.reserve(shapes.size());
    for (const auto& [key, stats] : shapes) ranked.emplace_back(&key, &stats);
    std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) { return left.second->us > right.second->us || (left.second->us == right.second->us && *left.first < *right.first); });
    appendf(line, "; shapes %zu, top by committer ms (draws, ms, template/build/bda, push fits, bda covered):", shapes.size());
    double restUs = 0;
    std::uint64_t restDraws = 0;
    for (std::size_t index = 0; index < ranked.size(); ++index) {
        const auto& stats = *ranked[index].second;
        if (index >= topShapes) {
            restUs += stats.us;
            restDraws += stats.draws;
            continue;
        }
        appendf(line, " %s|%s %llu %.1f %llu/%llu/%llu %llu %llu;", StagesName(ranked[index].first->second).c_str(), ShapeName(ranked[index].first->first).c_str(), static_cast<unsigned long long>(stats.draws), stats.us / 1000.0, static_cast<unsigned long long>(stats.kinds[1]), static_cast<unsigned long long>(stats.kinds[2]), static_cast<unsigned long long>(stats.kinds[3]), static_cast<unsigned long long>(stats.pushFits), static_cast<unsigned long long>(stats.bdaCovered));
    }
    if (restDraws != 0) appendf(line, " rest %llu %.1f;", static_cast<unsigned long long>(restDraws), restUs / 1000.0);
    return line;
}

bool Enabled() {
    static const bool enabled = [] {
        const auto set = [](const char* name) {
            const char* value = std::getenv(name);
            return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
        };
        return set("APS5_NEW_PATH") && set("APS5_PROFILE_NEW_PATH") && set("APS5_PROFILE_DRAW");
    }();
    return enabled;
}

}
