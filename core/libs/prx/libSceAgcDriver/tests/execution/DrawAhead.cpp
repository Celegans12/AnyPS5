#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "VulkanTestDevice.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::DriverDetail::DrawMiss;

constexpr std::byte Kept{0x40};
constexpr std::size_t Draws = 4;

alignas(256) constexpr std::array<std::uint32_t, 16> VertexCode{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 8> PixelCode{0x7e000200u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 8> TablePixelCode{0xf4000080u, 0xfa000000u, 0xbf8cc07fu, 0x7e000202u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u};
using Target = std::array<std::byte, 1024>;
alignas(256) std::array<Target, Draws> Targets{};
alignas(256) std::array<Target, Draws> References{};
alignas(256) std::array<Target, 1024> Skipped{};
alignas(256) std::array<Target, 4> TableTargets{};
alignas(64) std::atomic<std::uint32_t> Released{0};
alignas(64) std::uint32_t Condition = 0;
alignas(64) std::array<std::uint32_t, 4> IndirectRecord{3u, 1u, 0u, 0u};
alignas(64) std::array<std::uint32_t, 4> Table{};

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header VertexHeader{};
alignas(8) Header PixelHeader{};
alignas(8) Header TablePixelHeader{};

void Registered(Header& header, const std::uint32_t* code, std::size_t bytes, std::uint8_t type) {
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(bytes);
    header.shader.code = code;
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

void Append(std::vector<std::uint32_t>& commands, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload);
}

std::uint32_t Low(const void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint32_t High(const void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer) >> 32u);
}

struct Pass {
    bool hold = true;
    bool table = false;
    const std::uint32_t* condition = nullptr;
    bool indirectDrawIndex = false;
    std::optional<std::uint32_t> written;
    std::vector<std::pair<Target*, std::uint32_t>> draws;
};

std::vector<std::uint32_t> Commands(const Pass& pass) {
    const auto vertex = reinterpret_cast<std::uintptr_t>(VertexCode.data());
    const auto pixel = reinterpret_cast<std::uintptr_t>(pass.table ? TablePixelCode.data() : PixelCode.data());
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> context{
        {0x2d5, 0x2000},
        {0x1b6, 0}, {0x207, 0}, {0x200, 0}, {0x203, 0x800},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, (63u << 14u) | 3u},
        {0x3b8, 0x9000000}, {0x1e0, 0},
        {0xc, 0}, {0xd, 0x40040},
        {0x81, 0x80000000}, {0x82, 0x40040},
        {0x90, 0x80000000}, {0x91, 0x40040},
        {0x94, 0x80000000}, {0x95, 0x40040},
        {0x10f, std::bit_cast<std::uint32_t>(32.0f)}, {0x110, std::bit_cast<std::uint32_t>(32.0f)},
        {0x111, std::bit_cast<std::uint32_t>(-2.0f)}, {0x112, std::bit_cast<std::uint32_t>(2.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0},
        {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
        {0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0},
    };
    std::vector<std::pair<std::uint32_t, std::uint32_t>> shader{
        {0xc8, static_cast<std::uint32_t>(vertex >> 8u)}, {0xc9, static_cast<std::uint32_t>(vertex >> 40u)}, {0x8b, 1u << 1u},
        {0x008, static_cast<std::uint32_t>(pixel >> 8u)}, {0x009, static_cast<std::uint32_t>(pixel >> 40u)},
    };
    if (pass.table) shader.insert(shader.end(), {{0x00b, 2u << 1u}, {0x00c, Low(Table.data())}, {0x00d, High(Table.data())}});
    else shader.insert(shader.end(), {{0x00b, 1u << 1u}, {0x00c, std::bit_cast<std::uint32_t>(1.0f)}});
    std::vector<std::uint32_t> commands;
    Append(commands, 0x79, {0x242, 4});
    if (pass.indirectDrawIndex) {
        Append(commands, 0x79, {0x24a, 0u});
        Append(commands, 0x11, {1u, Low(IndirectRecord.data()), High(IndirectRecord.data())});
    }
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) Append(commands, 0x76, {offset, value});
    if (pass.hold) Append(commands, 0x3c, {0x13u, Low(&Released), High(&Released), 1u, 0xffffffffu, 4u});
    if (pass.written) Append(commands, 0x37, {(5u << 8u) | (1u << 20u), Low(Table.data()), High(Table.data()), *pass.written});
    std::vector<std::uint32_t> body;
    for (const auto& [target, word] : pass.draws) {
        const auto color = reinterpret_cast<std::uintptr_t>(target->data());
        Append(body, 0x69, {0x318, static_cast<std::uint32_t>(color >> 8u)});
        Append(body, 0x69, {0x390, static_cast<std::uint32_t>(color >> 40u)});
        Append(body, 0x76, {0x8c, word});
        if (pass.indirectDrawIndex) Append(body, 0x2c, {0u, 0u, 0u, 0x8cu | (1u << 31u), 1u, 0u, 0u, 16u, 2u});
        else Append(body, 0x2d, {3, 2});
    }
    if (pass.condition != nullptr) Append(commands, 0x22, {Low(pass.condition), High(pass.condition), 0u, static_cast<std::uint32_t>(body.size())});
    commands.insert(commands.end(), body.begin(), body.end());
    return commands;
}

void Submit(const Pass& pass) {
    auto commands = Commands(pass);
    if (pass.hold) Released.store(0, std::memory_order_release);
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "the draw submission failed");
}

void Release() {
    Released.store(1, std::memory_order_release);
    AgcDriverWaitIdle_nid_postfix();
}

template <typename TCondition>
void Await(TCondition&& condition, const std::string& what) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!condition()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            Released.store(1, std::memory_order_release);
            throw std::runtime_error("timed out waiting for " + what);
        }
        std::this_thread::yield();
    }
}

void RenderReference(const Pass& pass) {
    ShaderRecompiler::SetDebugProbeActive(true);
    struct Restore {
        ~Restore() { ShaderRecompiler::SetDebugProbeActive(false); }
    } restore;
    auto reference = pass;
    reference.hold = false;
    Submit(reference);
    AgcDriverWaitIdle_nid_postfix();
}

void DrawsCompiledAheadAreUsed(bool verify) {
    auto& driver = AgcDriver::DriverDetail::Driver::Get();
    Pass reference{false};
    for (std::size_t draw = 0; draw < Draws; ++draw) reference.draws.emplace_back(&References[draw], static_cast<std::uint32_t>(draw));
    for (auto& target : References) target.fill(Kept);
    RenderReference(reference);
    for (auto& target : Targets) target.fill(Kept);
    Pass pass;
    for (std::size_t draw = 0; draw < Draws; ++draw) pass.draws.emplace_back(&Targets[draw], static_cast<std::uint32_t>(draw));
    const auto before = driver.DrawCacheCounters();
    const auto heldBefore = driver.DrawCacheState();
    const auto aheadBefore = driver.DrawAheadStatistics();
    Submit(pass);
    Await([&] { return driver.DrawAheadStatistics().compiled - aheadBefore.compiled >= Draws && driver.DrawAheadStatistics().published - aheadBefore.published >= Draws; }, "the draws behind the wait to be compiled ahead");
    const auto ahead = driver.DrawCacheCounters();
    const auto held = driver.DrawCacheState();
    Release();
    Require(held.probation - heldBefore.probation == Draws && held.admitted == heldBefore.admitted && ahead.inserts == before.inserts && ahead.lookups == before.lookups, "the draws behind the wait were not compiled ahead of the queue and held outside the draw cache: " + std::to_string(held.probation - heldBefore.probation) + " held, " + std::to_string(ahead.inserts - before.inserts) + " inserts and " + std::to_string(ahead.lookups - before.lookups) + " lookups before the release");
    const auto counters = driver.DrawCacheCounters();
    const auto statistics = driver.DrawAheadStatistics();
    Require(statistics.compiled - aheadBefore.compiled == Draws, "the workers compiled " + std::to_string(statistics.compiled - aheadBefore.compiled) + " draws, expected " + std::to_string(Draws));
    if (!verify) Require(statistics.used - aheadBefore.used == Draws, "the queue used " + std::to_string(statistics.used - aheadBefore.used) + " published draw records, expected " + std::to_string(Draws));
    const auto lookups = counters.lookups - before.lookups;
    const auto absent = counters.absent - before.absent;
    const auto hits = counters.hits - before.hits;
    const auto inserts = counters.inserts - before.inserts;
    Require(lookups == Draws && absent == 0 && hits == Draws && inserts == 1, "draws compiled ahead were not reused, or were admitted at their first sight beyond the one key their shape's earlier sighting admits: " + std::to_string(lookups) + " lookups, " + std::to_string(absent) + " absent, " + std::to_string(hits) + " hits, " + std::to_string(inserts) + " inserts");
    for (std::size_t miss = 0; miss < counters.misses.size(); ++miss) Require(counters.misses[miss] == before.misses[miss], "a draw compiled ahead missed with reason " + std::to_string(miss));
    if (verify) Require(counters.verifyHits - before.verifyHits == Draws && counters.verifyMismatches == before.verifyMismatches, "a draw compiled ahead captured other memory than the queue's own compile");
    const auto kept = static_cast<std::size_t>(std::count(References[0].begin(), References[0].end(), Kept));
    Require(kept != 0 && kept != References[0].size(), "the reference draw covered " + std::to_string(References[0].size() - kept) + " of " + std::to_string(References[0].size()) + " bytes, expected an edge through the target");
    for (std::size_t draw = 0; draw < Draws; ++draw) Require(Targets[draw] == References[draw], "draw " + std::to_string(draw) + " compiled ahead wrote other pixels than the uncached path");
    for (std::size_t sight = 0; sight < 2; ++sight) {
        auto again = pass;
        again.hold = false;
        const auto seen = driver.DrawCacheCounters();
        Submit(again);
        AgcDriverWaitIdle_nid_postfix();
        const auto repeated = driver.DrawCacheCounters();
        Require(repeated.hits - seen.hits == Draws && repeated.absent == seen.absent && repeated.inserts - seen.inserts == (verify ? Draws : sight == 0 ? Draws - 1 : 0), "sight " + std::to_string(sight + 2) + " of the draws compiled ahead made " + std::to_string(repeated.hits - seen.hits) + " hits, " + std::to_string(repeated.absent - seen.absent) + " absent and " + std::to_string(repeated.inserts - seen.inserts) + " inserts, expected every draw admitted by its second sight and kept after it");
    }
}

void SkippedDrawsRetireWithTheirSubmission() {
    auto& driver = AgcDriver::DriverDetail::Driver::Get();
    Condition = 0;
    Pass pass;
    pass.condition = &Condition;
    for (std::size_t draw = 0; draw < Skipped.size(); ++draw) {
        Skipped[draw].fill(Kept);
        pass.draws.emplace_back(&Skipped[draw], static_cast<std::uint32_t>(0x100 + draw));
    }
    const auto before = driver.DrawAheadStatistics();
    Submit(pass);
    Await([&] { return driver.DrawAheadStatistics().compiled != before.compiled; }, "a skipped draw to be compiled ahead");
    Release();
    const auto retired = driver.DrawAheadStatistics();
    const auto held = driver.DrawCacheState().probation;
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const auto later = driver.DrawAheadStatistics();
    Require(later.compiled == retired.compiled && driver.DrawCacheState().probation == held, "draw-ahead workers inserted " + std::to_string(later.compiled - retired.compiled) + " entries for draws of a submission that had retired");
    Require(later.published == retired.published, "the draw-ahead scan published draws of a submission that had retired");
    Require(retired.compiled - before.compiled < Skipped.size(), "every skipped draw was compiled ahead before the queue retired them");
    for (const auto& target : Skipped) Require(std::all_of(target.begin(), target.end(), [](std::byte value) { return value == Kept; }), "a draw inside a false condition wrote its target");
}

std::vector<std::uint32_t> RewindCommands;

void RewindContinuationsAreCompiledAhead(bool verify) {
    auto& driver = AgcDriver::DriverDetail::Driver::Get();
    for (auto& target : Targets) target.fill(Kept);
    Pass tail;
    for (std::size_t draw = 0; draw < Draws; ++draw) tail.draws.emplace_back(&Targets[draw], static_cast<std::uint32_t>(0xc00 + draw));
    RewindCommands = Commands(Pass{false});
    Append(RewindCommands, 0x59, {0x80000000u});
    const auto continuation = Commands(tail);
    RewindCommands.insert(RewindCommands.end(), continuation.begin(), continuation.end());
    const auto before = driver.DrawAheadStatistics();
    Released.store(0, std::memory_order_release);
    Packet packet{RewindCommands.data(), static_cast<std::uint32_t>(RewindCommands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "the draw submission failed");
    Await([&] { return driver.DrawAheadStatistics().compiled - before.compiled >= Draws; }, "the draws of a REWIND continuation to be compiled ahead");
    Release();
    const auto after = driver.DrawAheadStatistics();
    Require(after.compiled - before.compiled == Draws, "the workers compiled " + std::to_string(after.compiled - before.compiled) + " draws of a REWIND continuation, expected " + std::to_string(Draws));
    if (!verify) Require(after.used - before.used == Draws, "the queue used " + std::to_string(after.used - before.used) + " published draw records of a REWIND continuation, expected " + std::to_string(Draws));
    for (const auto& target : Targets) Require(std::any_of(target.begin(), target.end(), [](std::byte value) { return value != Kept; }), "a draw of a REWIND continuation wrote nothing");
}

void TablesWrittenBehindTheWaitAreSeen() {
    auto& driver = AgcDriver::DriverDetail::Driver::Get();
    const auto stale = std::bit_cast<std::uint32_t>(0.0f);
    const auto fresh = std::bit_cast<std::uint32_t>(1.0f);
    std::array<Target, 2> expected{};
    for (std::size_t value = 0; value < 2; ++value) {
        Table.fill(value == 0 ? stale : fresh);
        TableTargets[3].fill(Kept);
        Pass reference{false, true};
        reference.draws.emplace_back(&TableTargets[3], 0u);
        RenderReference(reference);
        expected[value] = TableTargets[3];
    }
    Require(expected[0] != expected[1], "the table draw renders the same pixels for both table values");
    for (std::size_t mode = 0; mode < 2; ++mode) {
        Table.fill(stale);
        auto& target = TableTargets[mode];
        target.fill(Kept);
        Pass pass{true, true};
        if (mode == 1) pass.written = fresh;
        pass.draws.emplace_back(&target, static_cast<std::uint32_t>(0x200 + mode));
        const auto before = driver.DrawAheadStatistics();
        Submit(pass);
        Await([&] { return driver.DrawAheadStatistics().compiled != before.compiled; }, "the table draw to be compiled ahead");
        if (mode == 0) Table.fill(fresh);
        Release();
        Require(target == expected[1], mode == 0 ? "a draw compiled ahead ignored a CPU store to its table made while the queue waited" : "a draw compiled ahead ignored a queue write to its table before the draw");
    }
}

void VerificationComparesWholeCompilations() {
    using AgcDriver::DriverDetail::SameCompilation;
    ShaderRecompiler::RecompileResult base;
    base.spirv = std::vector<std::uint32_t>{0x07230203u, 0x00010000u, 0u, 1u, 0u};
    ShaderRecompiler::DescriptorBinding binding{};
    binding.kind = ShaderRecompiler::DescriptorKind::StorageImage;
    binding.count = 1;
    binding.guestDescriptor = {1, 2, 3, 4, 5, 6, 7, 8};
    binding.imageWritten = {false};
    binding.imageShapes = {ShaderRecompiler::DescriptorImageShape{}};
    binding.samplerDepthCompare = {false};
    binding.bufferRead = {true};
    base.bindings.push_back(binding);
    base.vertexAttributes.push_back({0, 4, {{1, 2, 3, 4}}, 0});
    base.pushConstants = {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    base.fragmentParameters.push_back({0, 0, false, false});
    Require(SameCompilation(base, base), "a compilation differed from itself");
    auto cached = base;
    cached.cacheHit = true;
    Require(SameCompilation(base, cached), "a compilation served from a cache differed from the same compilation built anew");
    const auto differs = [&](const char* what, auto&& change) {
        auto other = base;
        change(other);
        Require(!SameCompilation(base, other), std::string("a compilation with another ") + what + " was taken for the same");
    };
    differs("written image", [](auto& result) { result.bindings[0].imageWritten[0] = true; });
    differs("image shape", [](auto& result) { result.bindings[0].imageShapes.push_back(ShaderRecompiler::DescriptorImageShape{}); });
    differs("sampler depth compare", [](auto& result) { result.bindings[0].samplerDepthCompare[0] = true; });
    differs("buffer read", [](auto& result) { result.bindings[0].bufferRead[0] = false; });
    differs("guest descriptor", [](auto& result) { result.bindings[0].guestDescriptor[0] = 9; });
    differs("vertex attribute", [](auto& result) { result.vertexAttributes[0].resource.fields[0] = 5; });
    differs("push constant", [](auto& result) { result.pushConstants[0] = std::byte{9}; });
    differs("SPIR-V module", [](auto& result) { result.spirv = std::vector<std::uint32_t>{0x07230203u, 0x00010000u, 0u, 2u, 0u}; });
    differs("fragment parameter", [](auto& result) { result.fragmentParameters[0].flat = true; });
    differs("specialization", [](auto& result) { result.specialization.push_back({1, 2}); });
    differs("workgroup memory size", [](auto& result) { result.workgroupMemoryDwords = 64; });
}

void PacketDerivedCompilesStayOnTheQueue() {
    using AgcDriver::DriverDetail::DrawAhead;
    AgcDriver::Graphics::State state{};
    AgcDriver::Pm4::DrawParameters draw{0x10000, 3, 2, 1, 0};
    Require(DrawAhead::CompilesAhead(state, draw), "a direct vertex draw is not compiled ahead");
    auto indirect = draw;
    indirect.indirect.emplace();
    Require(!DrawAhead::CompilesAhead(state, indirect), "an indirect draw is compiled ahead");
    auto mesh = state;
    mesh.stages.mesh = ShaderRecompiler::MeshConfiguration{4, 1, 3, 3, 1, 64, 128, 0, 4};
    Require(!DrawAhead::CompilesAhead(mesh, draw), "a mesh draw, whose index buffer descriptor its packet supplies, is compiled ahead under a key of its registers");
    auto tessellation = state;
    tessellation.stages.tessellation = ShaderRecompiler::TessellationConfiguration{3, 3, 1, 0, 2};
    Require(DrawAhead::CompilesAhead(tessellation, draw), "a tessellated draw of one instance is not compiled ahead");
    auto instanced = draw;
    instanced.instanceCount = 2;
    Require(!DrawAhead::CompilesAhead(tessellation, instanced), "an instanced tessellated draw, whose hull patches its packet's counts number, is compiled ahead");
}

void SkippedSpeculationStaysOnProbation() {
    auto& driver = AgcDriver::DriverDetail::Driver::Get();
    for (auto& target : Targets) target.fill(Kept);
    Pass once{false};
    once.draws.emplace_back(&Targets[0], 0x300u);
    AgcDriver::DriverDetail::DrawAhead::SetCompileAhead(false);
    Submit(once);
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::DriverDetail::DrawAhead::SetCompileAhead(true);
    const auto sighted = driver.DrawCacheState();
    const auto aheadBefore = driver.DrawAheadStatistics();
    Condition = 0;
    Pass skipped;
    skipped.condition = &Condition;
    skipped.draws.emplace_back(&Targets[0], 0x300u);
    const auto drawn = Targets[0];
    Submit(skipped);
    Await([&] { return driver.DrawAheadStatistics().compiled != aheadBefore.compiled; }, "the skipped draw to be compiled ahead");
    Release();
    const auto speculated = driver.DrawCacheState();
    Require(speculated.admitted == sighted.admitted && speculated.probation == sighted.probation + 1, "a draw seen once and compiled ahead inside a false condition was admitted without a second execution: " + std::to_string(speculated.admitted - sighted.admitted) + " admitted, " + std::to_string(speculated.probation - sighted.probation) + " on probation");
    Require(Targets[0] == drawn, "a draw inside a false condition wrote its target");
    Submit(once);
    AgcDriverWaitIdle_nid_postfix();
    const auto executed = driver.DrawCacheState();
    Require(executed.admitted == sighted.admitted + 1 && executed.probation == sighted.probation, "the second execution of a draw compiled ahead on probation left " + std::to_string(executed.admitted - sighted.admitted) + " admitted and " + std::to_string(executed.probation - sighted.probation) + " on probation, expected it admitted as without compile-ahead");
}

void UncachedExecutionsAreNoSightings() {
    auto& driver = AgcDriver::DriverDetail::Driver::Get();
    for (auto& target : Targets) target.fill(Kept);
    const auto before = driver.DrawCacheState();
    const auto aheadBefore = driver.DrawAheadStatistics();
    Condition = 0;
    Pass skipped;
    skipped.condition = &Condition;
    skipped.draws.emplace_back(&Targets[2], 0x340u);
    Submit(skipped);
    Await([&] { return driver.DrawAheadStatistics().compiled != aheadBefore.compiled; }, "the skipped draw to be compiled ahead");
    Release();
    const auto speculated = driver.DrawCacheState();
    Require(speculated.admitted == before.admitted && speculated.probation == before.probation + 1, "a skipped draw compiled ahead was not held on probation");
    Pass indirect{false};
    indirect.indirectDrawIndex = true;
    indirect.draws.emplace_back(&Targets[2], 0x340u);
    for (std::size_t sight = 0; sight < 2; ++sight) {
        Submit(indirect);
        AgcDriverWaitIdle_nid_postfix();
    }
    const auto indirected = driver.DrawCacheState();
    Require(indirected.admitted == before.admitted && indirected.probation == before.probation + 1, "two executions of an indirect draw the queue draws on the CPU, which the draw cache does not admit, promoted the draw compiled ahead with its registers: " + std::to_string(indirected.admitted - before.admitted) + " admitted, " + std::to_string(indirected.probation - before.probation) + " on probation");
    Pass direct{false};
    direct.draws.emplace_back(&Targets[2], 0x340u);
    Submit(direct);
    AgcDriverWaitIdle_nid_postfix();
    const auto first = driver.DrawCacheState();
    Require(first.admitted == before.admitted && first.probation == before.probation + 1, "the first execution of the direct draw admitted it");
    Submit(direct);
    AgcDriverWaitIdle_nid_postfix();
    const auto second = driver.DrawCacheState();
    Require(second.admitted == before.admitted + 1 && second.probation == before.probation, "the second execution of the direct draw did not admit it");
}

}

int main() {
    try {
        PacketDerivedCompilesStayOnTheQueue();
        VerificationComparesWholeCompilations();
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        const bool verify = std::getenv("APS5_VERIFY_DRAW_ENTRIES") != nullptr;
        Registered(VertexHeader, VertexCode.data(), sizeof(VertexCode), 2);
        Registered(PixelHeader, PixelCode.data(), sizeof(PixelCode), 1);
        Registered(TablePixelHeader, TablePixelCode.data(), sizeof(TablePixelCode), 1);
        DrawsCompiledAheadAreUsed(verify);
        SkippedDrawsRetireWithTheirSubmission();
        RewindContinuationsAreCompiledAhead(verify);
        TablesWrittenBehindTheWaitAreSeen();
        SkippedSpeculationStaysOnProbation();
        UncachedExecutionsAreNoSightings();
        AgcDriverShutdown_nid_postfix();
        std::cout << "Draws compiled ahead of the queue were reused" << (verify ? " and matched the queue's own compile" : "") << ", retired with their submission and followed table writes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        Released.store(1, std::memory_order_release);
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
