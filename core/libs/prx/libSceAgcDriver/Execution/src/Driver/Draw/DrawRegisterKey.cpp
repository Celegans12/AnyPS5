#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRegisterKey.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace AgcDriver::DriverDetail {

namespace {

bool runtimeRegister(std::uint32_t offset) {
    return offset - 0x00cu < 32u || offset - 0x04cu < 16u || offset - 0x08cu < 32u || offset - 0x10cu < 32u || offset == 0x082u || offset == 0x083u || offset == 0x102u || offset == 0x103u;
}

const auto& registerMasks() {
    static const auto masks = [] {
        std::array<std::vector<std::uint64_t>, 3> bits;
        for (const auto& range : Graphics::DrawKeyRegisters) {
            auto& bank = bits[static_cast<std::size_t>(range.bank)];
            bank.resize(std::max(bank.size(), (static_cast<std::size_t>(range.first) + range.count + 63) / 64));
            for (auto offset = range.first; offset < range.first + range.count; ++offset) {
                if (range.bank == Graphics::RegisterBank::Shader && runtimeRegister(offset)) continue;
                bank[offset / 64] |= std::uint64_t{1} << (offset % 64);
            }
        }
        std::array<std::shared_ptr<const Registers::Mask>, 3> result;
        for (std::size_t i = 0; i < bits.size(); ++i) result[i] = std::make_shared<const Registers::Mask>(std::move(bits[i]));
        return result;
    }();
    return masks;
}

template <typename TVisit>
void forEachKeyedUserWord(const QueueState& queue, bool allUserWords, TVisit&& visit) {
    const auto words = [&](std::uint32_t base, std::uint32_t count) {
        for (auto it = queue.shader.lower_bound(base); it != queue.shader.end() && it->first < base + count; ++it) visit(it->first, it->second);
    };
    for (const auto base : {0x00cu, 0x04cu, 0x08cu, 0x10cu}) {
        const auto limit = base == 0x04cu ? 16u : 32u;
        const auto resources = queue.shader.find(base - 1);
        const auto count = resources == queue.shader.end() ? 0u : ((resources->second >> 1u) & 0x1fu) | (((resources->second >> 27u) & 1u) << 5u);
        words(base, allUserWords ? limit : std::min(count, limit));
    }
    words(0x082u, 2);
    words(0x102u, 2);
}

void drawKeyRanges(const QueueState& queue, std::array<std::uint64_t, Graphics::DrawKeyRegisters.size()>& ranges) {
    const auto& masks = registerMasks();
    for (std::size_t r = 0; r < Graphics::DrawKeyRegisters.size(); ++r) {
        const auto& range = Graphics::DrawKeyRegisters[r];
        const auto& registers = range.bank == Graphics::RegisterBank::Context ? queue.context : range.bank == Graphics::RegisterBank::Shader ? queue.shader : queue.userConfig;
        const auto& mask = *masks[static_cast<std::size_t>(range.bank)];
        std::uint64_t hash = 0xcbf29ce484222325ull;
        for (auto it = registers.lower_bound(range.first); it != registers.end() && it->first < range.first + range.count; ++it) {
            if (!mask.Contains(it->first)) continue;
            hash = (hash ^ it->first) * 0x100000001b3ull;
            hash = (hash ^ it->second) * 0x100000001b3ull;
        }
        ranges[r] = hash;
    }
}

}

std::uint64_t RegisterStateKey(const QueueState& queue, bool allUserWords, bool incremental, std::uint64_t* shapeKey, RegisterStateKeyParts* parts) {
    const auto& masks = registerMasks();
    const auto fingerprint = [&](const Registers& registers, Graphics::RegisterBank bank) {
        const auto& mask = masks[static_cast<std::size_t>(bank)];
        return incremental ? registers.Fingerprint(mask) : registers.RecomputeFingerprint(*mask);
    };
    auto shape = (0xcbf29ce484222325ull ^ fingerprint(queue.context, Graphics::RegisterBank::Context)) * 0x100000001b3ull;
    shape = (shape ^ fingerprint(queue.shader, Graphics::RegisterBank::Shader)) * 0x100000001b3ull;
    shape = (shape ^ fingerprint(queue.userConfig, Graphics::RegisterBank::UserConfig)) * 0x100000001b3ull;
    if (shapeKey != nullptr) *shapeKey = shape;
    auto exact = shape;
    std::uint64_t userData = 0xcbf29ce484222325ull;
    std::uint64_t auxiliary = 0xcbf29ce484222325ull;
    const auto mixPart = [](std::uint64_t& part, std::uint64_t value) { part = (part ^ value) * 0x100000001b3ull; };
    const auto words = [&](std::uint32_t base, std::uint32_t count, std::uint64_t& part) {
        for (auto it = queue.shader.lower_bound(base); it != queue.shader.end() && it->first < base + count; ++it) {
            exact = (exact ^ it->first) * 0x100000001b3ull;
            exact = (exact ^ it->second) * 0x100000001b3ull;
            mixPart(part, it->first);
            mixPart(part, it->second);
        }
    };
    std::array<std::uint64_t, 3> byStage{};
    byStage.fill(0xcbf29ce484222325ull);
    for (const auto base : {0x00cu, 0x04cu, 0x08cu, 0x10cu}) {
        const auto limit = base == 0x04cu ? 16u : 32u;
        const auto resources = queue.shader.find(base - 1);
        const auto count = resources == queue.shader.end() ? 0u : ((resources->second >> 1u) & 0x1fu) | (((resources->second >> 27u) & 1u) << 5u);
        words(base, allUserWords ? limit : std::min(count, limit), byStage[base == 0x00cu ? 0u : base == 0x10cu ? 2u : 1u]);
    }
    for (const auto part : byStage) {
        mixPart(userData, part);
    }
    words(0x082u, 2, auxiliary);
    words(0x102u, 2, auxiliary);
    if (parts != nullptr) *parts = {shape, userData, auxiliary, byStage};
    return exact;
}


void WalkDrawRegistry(const QueueState& queue, const ShaderRegistry& registry, DrawRegistryWalk& walk) {
    std::size_t stage = 0;
    for (const auto base : {0x008u, 0x048u, 0x088u, 0x0c8u, 0x108u, 0x148u}) {
        auto& s = walk[stage++];
        s = {};
        const auto low = queue.shader.find(base);
        const auto high = queue.shader.find(base + 1);
        if (low == queue.shader.end() || high == queue.shader.end()) continue;
        s.present = true;
        s.low = low->second;
        s.high = high->second;
        const auto resources = queue.shader.find(base + 3);
        s.resources = resources != queue.shader.end() ? resources->second : 0;
        const auto address = (static_cast<std::uint64_t>(low->second) << 8u) | (static_cast<std::uint64_t>(high->second & 0xffu) << 40u);
        auto it = registry.upper_bound(address);
        if (it == registry.begin()) continue;
        --it;
        s.snapshots = &it->second;
        s.offset = address - it->first;
    }
}

std::uint64_t DrawShaderSetKey(const DrawRegistryWalk& walk) {
    std::uint64_t key = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) { key = (key ^ value) * 0x100000001b3ull; };
    for (const auto& s : walk) {
        if (!s.present) { mix(0); continue; }
        if (s.snapshots == nullptr) { mix(1); continue; }
        for (const auto& snapshot : *s.snapshots) mix(reinterpret_cast<std::uintptr_t>(snapshot.get()));
        mix(s.offset);
    }
    return key;
}

std::uint64_t DrawShaderSetKey(const QueueState& queue, const ShaderRegistry& registry) {
    DrawRegistryWalk walk;
    WalkDrawRegistry(queue, registry, walk);
    return DrawShaderSetKey(walk);
}

std::uint64_t Driver::drawIdentityKey(const DrawRegistryWalk& walk) {
    std::uint64_t key = 0x84222325cbf29ce4ull;
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
    };
    for (const auto& s : walk) {
        if (!s.present) {
            mix(0);
            continue;
        }
        mix(s.low);
        mix(s.high);
        mix(s.resources);
        if (s.snapshots == nullptr) {
            mix(1);
            continue;
        }
        for (const auto& snapshot : *s.snapshots) mix(reinterpret_cast<std::uintptr_t>(snapshot.get()));
        mix(s.offset);
    }
    return key == 0 ? 1 : key;
}

std::uint64_t Driver::drawIdentityKey(const QueueState& queue, const ShaderRegistry& registry) {
    DrawRegistryWalk walk;
    WalkDrawRegistry(queue, registry, walk);
    return drawIdentityKey(walk);
}

std::uint64_t Driver::drawRegisterKey(const QueueState& queue, const ShaderRegistry& registry, std::uint64_t deviceSerial, std::uint64_t* shape, DrawKeyComponents* components, const DrawRelocation* relocation, std::uint64_t* alias, const DrawRegistryWalk* walk) {
    const bool aliased = alias != nullptr && relocation != nullptr && !relocation->words.empty();
    bool readable = true;
    std::uint64_t other = 0x6c62272e07bb0142ull;
    const auto mixOther = [&](std::uint64_t value) {
        other ^= value;
        other *= 0x100000001b3ull;
    };
    static const bool allUserWords = std::getenv("APS5_DRAW_KEY_ALL_USER_WORDS") != nullptr;
    static const bool verify = std::getenv("APS5_VERIFY_REGISTER_KEYS") != nullptr;
    std::uint64_t shapeKey = 0;
    RegisterStateKeyParts parts;
    auto key = RegisterStateKey(queue, allUserWords, true, &shapeKey, components != nullptr ? &parts : nullptr);
    if (verify && key != RegisterStateKey(queue, allUserWords, false)) throw std::runtime_error("incremental register fingerprint disagrees with full state");
    if (aliased) {
        mixOther(shapeKey);
        forEachKeyedUserWord(queue, allUserWords, [&](std::uint32_t offset, std::uint32_t value) {
            mixOther(offset);
            const auto word = std::find_if(relocation->words.begin(), relocation->words.end(), [&](const RelocatableUserWord& candidate) { return candidate.reg == offset; });
            mixOther(word != relocation->words.end() ? relocatedContentKey(*word, value, readable) : value);
        });
        mixOther(deviceSerial);
    }
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
        shapeKey ^= value;
        shapeKey *= 0x100000001b3ull;
    };
    if (components != nullptr) {
        components->stable = parts.stableRegisters;
        components->userData = parts.userDataByStage;
        components->auxiliary = parts.auxiliaryRegisters;
        components->device = deviceSerial;
    }
    DrawRegistryWalk localWalk;
    if (walk == nullptr) {
        WalkDrawRegistry(queue, registry, localWalk);
        walk = &localWalk;
    }
    // Only the cheap parts here: traceDrawKeyLookup builds the per-range breakdown when the stable part changes.
    if (components != nullptr) components->shaderSet = DrawShaderSetKey(*walk);
    mix(deviceSerial);
    for (const auto& s : *walk) {
        if (!s.present) {
            mix(0);
            continue;
        }
        if (s.snapshots == nullptr) {
            mix(1);
            continue;
        }
        for (const auto& snapshot : *s.snapshots) mix(reinterpret_cast<std::uintptr_t>(snapshot.get()));
        mix(s.offset);
        if (aliased) {
            for (const auto& snapshot : *s.snapshots) mixOther(reinterpret_cast<std::uintptr_t>(snapshot.get()));
            mixOther(s.offset);
        }
    }
    if (shape != nullptr) *shape = shapeKey;
    if (aliased && readable) *alias = other == 0 ? 1 : other;
    return key;
}

void Driver::traceDrawKeyLookup(const QueueState& queue, const DrawKeyComponents& components, bool cacheMiss) {
    struct PreviousStorage {};
    auto& previous = HostThreadLocal<std::unordered_map<std::uint64_t, DrawKeyComponents>, PreviousStorage>();
    const auto shaderSet = components.shaderSet;
    if (previous.size() >= 4096 && previous.find(shaderSet) == previous.end()) previous.clear();
    const auto [it, inserted] = previous.try_emplace(shaderSet);
    auto& last = it->second;
    // The range hashes cover exactly the masked registers of the stable fingerprint, so an unchanged stable part means
    // unchanged ranges: walk the ~530 DrawKeyRegisters only for a new shader set or a changed stable part.
    const bool rangesChanged = inserted || last.stable != components.stable;
    std::array<std::uint64_t, Graphics::DrawKeyRegisters.size()> ranges{};
    if (rangesChanged) drawKeyRanges(queue, ranges);
    if (cacheMiss) {
        auto& counters = drawEntryCounters;
        const auto note = [&](DrawKeyChange change) { ++counters.registerKeyMissChanges[static_cast<std::size_t>(change)]; };
        if (inserted) {
            note(DrawKeyChange::NewShaderSet);
        } else {
            bool changed = false;
            const auto compare = [&](bool differs, DrawKeyChange change) {
                if (!differs) return;
                note(change);
                changed = true;
            };
            compare(last.stable != components.stable, DrawKeyChange::Stable);
            compare(last.userData[0] != components.userData[0], DrawKeyChange::PixelUserData);
            compare(last.userData[1] != components.userData[1], DrawKeyChange::VertexUserData);
            compare(last.userData[2] != components.userData[2], DrawKeyChange::HullUserData);
            compare(last.auxiliary != components.auxiliary, DrawKeyChange::Auxiliary);
            compare(last.device != components.device, DrawKeyChange::Device);
            if (!changed) note(DrawKeyChange::Unchanged);
            if (rangesChanged) {
                for (std::size_t r = 0; r < ranges.size(); ++r) {
                    if (last.ranges[r] == ranges[r]) continue;
                    ++counters.stableRangeChanges[r];
                    break;
                }
            }
        }
    }
    last.stable = components.stable;
    last.userData = components.userData;
    last.auxiliary = components.auxiliary;
    last.device = components.device;
    last.shaderSet = components.shaderSet;
    if (rangesChanged) last.ranges = ranges;
}

bool Driver::sameVertexInfo(const ShaderRecompiler::ShaderVertexStageInfo& a, const ShaderRecompiler::ShaderVertexStageInfo& b) {
    if (a.resourcesNum != b.resourcesNum || a.fetchAttribReg != b.fetchAttribReg || a.fetchBufferReg != b.fetchBufferReg || a.fetchEmbedded != b.fetchEmbedded || a.paClVsOutCntl != b.paClVsOutCntl || a.drawBaseVertexSgpr != b.drawBaseVertexSgpr || a.drawStartInstanceSgpr != b.drawStartInstanceSgpr || a.pointList != b.pointList || a.pointSize != b.pointSize || a.pointSizeMin != b.pointSizeMin || a.pointSizeMax != b.pointSizeMax) return false;
    if (a.depthRemapMode != b.depthRemapMode) return false;
    for (std::uint32_t i = 0; i < a.resourcesNum && i < a.resources.size(); ++i) {
        if (a.resources[i].fields != b.resources[i].fields) return false;
        const auto& x = a.resourcesDst[i];
        const auto& y = b.resourcesDst[i];
        if (x.registerStart != y.registerStart || x.registersNum != y.registersNum || x.attrId != y.attrId || x.fetchIndex != y.fetchIndex) return false;
    }
    return true;
}

bool Driver::sameDecode(const DrawDecode& a, const DrawDecode& b) {
    const auto& s = a.state;
    const auto& t = b.state;
    const auto sameColor = [](const Graphics::ColorTarget& x, const Graphics::ColorTarget& y) {
        return x.address == y.address && x.extent.width == y.extent.width && x.extent.height == y.extent.height && x.format == y.format && x.bytes == y.bytes && x.componentMapping == y.componentMapping && x.tileMode == y.tileMode && x.elementBytes == y.elementBytes && x.dccAddress == y.dccAddress && x.dccAlphaOnMsb == y.dccAlphaOnMsb && x.slot == y.slot && x.exportIndex == y.exportIndex && x.partialWriteMask == y.partialWriteMask;
    };
    const auto sameBlend = [](const VkPipelineColorBlendAttachmentState& x, const VkPipelineColorBlendAttachmentState& y) {
        return x.blendEnable == y.blendEnable && x.srcColorBlendFactor == y.srcColorBlendFactor && x.dstColorBlendFactor == y.dstColorBlendFactor && x.colorBlendOp == y.colorBlendOp && x.srcAlphaBlendFactor == y.srcAlphaBlendFactor && x.dstAlphaBlendFactor == y.dstAlphaBlendFactor && x.alphaBlendOp == y.alphaBlendOp && x.colorWriteMask == y.colorWriteMask;
    };
    const auto sameMesh = [](const std::optional<ShaderRecompiler::MeshConfiguration>& x, const std::optional<ShaderRecompiler::MeshConfiguration>& y) {
        if (x.has_value() != y.has_value()) return false;
        if (!x) return true;
        return x->inputPrimitive == y->inputPrimitive && x->primitivesPerGroup == y->primitivesPerGroup && x->verticesPerGroup == y->verticesPerGroup && x->maxVertices == y->maxVertices && x->maxPrimitives == y->maxPrimitives && x->threadsPerGroup == y->threadsPerGroup && x->ldsSizeDwords == y->ldsSizeDwords && x->provokingVertex == y->provokingVertex && x->esgsItemSize == y->esgsItemSize && x->outputPrimitiveSize == y->outputPrimitiveSize;
    };
    const auto sameTess = [](const std::optional<ShaderRecompiler::TessellationConfiguration>& x, const std::optional<ShaderRecompiler::TessellationConfiguration>& y) {
        if (x.has_value() != y.has_value()) return false;
        if (!x) return true;
        return x->inputControlPoints == y->inputControlPoints && x->outputControlPoints == y->outputControlPoints && x->domain == y->domain && x->partitioning == y->partitioning && x->outputTopology == y->outputTopology && x->patchesPerGroup == y->patchesPerGroup && x->primitiveGroupSize == y->primitiveGroupSize && x->breakAtInstance == y->breakAtInstance && x->mergedHull == y->mergedHull;
    };
    if (s.stages.path != t.stages.path || s.stages.registerValue != t.stages.registerValue || s.stages.vertexWaveSize != t.stages.vertexWaveSize || s.stages.fragmentWaveSize != t.stages.fragmentWaveSize || s.stages.hullWaveSize != t.stages.hullWaveSize || !sameMesh(s.stages.mesh, t.stages.mesh) || !sameMesh(s.stages.tessellationGeometry, t.stages.tessellationGeometry) || !sameTess(s.stages.tessellation, t.stages.tessellation)) return false;
    if (!sameColor(s.color, t.color) || s.colors.size() != t.colors.size() || s.blends.size() != t.blends.size()) return false;
    for (std::size_t i = 0; i < s.colors.size(); ++i) {
        if (!sameColor(s.colors[i], t.colors[i])) return false;
    }
    if (s.blendedColors.size() != t.blendedColors.size() || s.targetBlends != t.targetBlends) return false;
    for (std::size_t i = 0; i < s.blendedColors.size(); ++i) {
        if (!sameColor(s.blendedColors[i], t.blendedColors[i])) return false;
    }
    for (std::size_t i = 0; i < s.blends.size(); ++i) {
        if (!sameBlend(s.blends[i], t.blends[i])) return false;
    }
    if (s.colorRop != t.colorRop || s.hasColorTarget != t.hasColorTarget || s.rectList != t.rectList || s.renderExtent.width != t.renderExtent.width || s.renderExtent.height != t.renderExtent.height || s.topology != t.topology || s.negativeOneToOne != t.negativeOneToOne || s.depthClamp != t.depthClamp || s.conservativeRasterization != t.conservativeRasterization || s.perSampleShading != t.perSampleShading || s.cullMode != t.cullMode || s.polygonMode != t.polygonMode || s.frontFace != t.frontFace || !sameBlend(s.blend, t.blend) || s.blendConstants != t.blendConstants) return false;
    if (std::memcmp(&s.viewport, &t.viewport, sizeof(VkViewport)) != 0 || std::memcmp(&s.scissor, &t.scissor, sizeof(VkRect2D)) != 0) return false;
    const auto& p = a.pixel;
    const auto& q = b.pixel;
    if (p.interpolatorCount != q.interpolatorCount || p.interpolatorSettings != q.interpolatorSettings || p.wave32 != q.wave32 || p.inputAddr != q.inputAddr || p.hasPerspectiveCenterVgpr != q.hasPerspectiveCenterVgpr || p.perspectiveCentroid != q.perspectiveCentroid || p.posX != q.posX || p.posY != q.posY || p.posZ != q.posZ || p.posW != q.posW || p.frontFace != q.frontFace || p.frontFaceAllBits != q.frontFaceAllBits || p.ancillary != q.ancillary || p.sampleShading != q.sampleShading || p.noPerspective != q.noPerspective || p.linearCentroid != q.linearCentroid || p.pixelKillEnable != q.pixelKillEnable || p.depthExportEnable != q.depthExportEnable || p.depthExportMin != q.depthExportMin || p.depthExportMax != q.depthExportMax || p.sampleMaskExportEnable != q.sampleMaskExportEnable || p.earlyZ != q.earlyZ || p.executeOnNoop != q.executeOnNoop || p.conservativeZExport != q.conservativeZExport || p.orderedPixelShader != q.orderedPixelShader || p.stencilRefExportEnable != q.stencilRefExportEnable || p.stencilOperationValueExport != q.stencilOperationValueExport || p.targetOutputMode != q.targetOutputMode || p.targetExportMapping != q.targetExportMapping) return false;
    if (a.roles != b.roles || a.programs.size() != b.programs.size()) return false;
    for (std::size_t i = 0; i < a.programs.size(); ++i) {
        const auto& x = a.programs[i];
        const auto& y = b.programs[i];
        if (x.binary.stage != y.binary.stage || x.binary.codeAddress != y.binary.codeAddress || x.userDataBase != y.userDataBase || x.firstUserSgpr != y.firstUserSgpr || x.userData != y.userData || x.snapshot != y.snapshot || x.codeOffset != y.codeOffset || x.tailContinuation != y.tailContinuation) return false;
    }
    if (p.targetExportPacking != q.targetExportPacking || p.targetBlend != q.targetBlend || p.dualSourceBlend != q.dualSourceBlend || p.reversedBlend != q.reversedBlend || p.targetMerges != q.targetMerges || p.alphaToCoverage != q.alphaToCoverage || p.rasterPrimitive != q.rasterPrimitive || p.ditheredAlphaToCoverage != q.ditheredAlphaToCoverage || p.ditheredAlphaSamples != q.ditheredAlphaSamples || p.quadPixelMask != q.quadPixelMask || p.quadSampleMask != q.quadSampleMask || p.iteratesSamples != q.iteratesSamples || p.positionLocation != q.positionLocation || p.sampleLocations != q.sampleLocations || p.sampleOrder != q.sampleOrder) return false;
    if (s.samples != t.samples || s.mergedCoverage != t.mergedCoverage || s.truncatedCoverage != t.truncatedCoverage || s.quadSampleMask != t.quadSampleMask || s.pixelEarlyZ != t.pixelEarlyZ || s.pixelKill != t.pixelKill) return false;
    return true;
}

}
