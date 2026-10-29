#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DrawCensus.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace Census = AgcDriver::NewPath::DrawCensus;
using ShaderRecompiler::BdaAbi::Range;

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

ShaderRecompiler::DescriptorBinding Binding(ShaderRecompiler::DescriptorRole role, ShaderRecompiler::DescriptorKind kind, bool readOnly = true) {
    ShaderRecompiler::DescriptorBinding binding{};
    binding.role = role;
    binding.kind = kind;
    binding.count = 1;
    binding.readOnly = readOnly;
    return binding;
}

void stageFactsFromBindings() {
    ShaderRecompiler::RecompileResult program{};
    Require(Census::Facts(program, 0).shape == 0 && Census::Facts(program, 64).pushEnd == 0, "an empty program has a shape or a push end");
    program.userDataRegisterReads = 0b1011'0000'0001ull;
    program.pushConstants.resize(48);
    program.bindings.push_back(Binding(ShaderRecompiler::DescriptorRole::GuestBuffers, ShaderRecompiler::DescriptorKind::StorageBuffer, false));
    program.bindings.push_back(Binding(ShaderRecompiler::DescriptorRole::GuestImages, ShaderRecompiler::DescriptorKind::SampledImage));
    program.bindings.push_back(Binding(ShaderRecompiler::DescriptorRole::GuestSamplers, ShaderRecompiler::DescriptorKind::Sampler));
    program.bindings.push_back(Binding(ShaderRecompiler::DescriptorRole::FlattenedSrt, ShaderRecompiler::DescriptorKind::StorageBuffer));
    program.bindings.push_back(Binding(ShaderRecompiler::DescriptorRole::FaultBuffer, ShaderRecompiler::DescriptorKind::StorageBuffer, false));
    const auto facts = Census::Facts(program, 64);
    Require(facts.userWords == 4, "user words are the read user SGPRs");
    Require(facts.pushEnd == 112, "the push end is the stage's offset plus its bytes");
    Require(facts.shape == (Census::Buffers | Census::BuffersWritten | Census::Sampled | Census::Samplers | Census::FlattenedSrt), "binding roles classified wrong: " + Census::ShapeName(facts.shape));
    ShaderRecompiler::RecompileResult address{};
    address.bdaAbiVersion = 1;
    address.bdaWrites = true;
    address.bindlessHeap = true;
    address.unboundedWrites = true;
    address.bindings.push_back(Binding(ShaderRecompiler::DescriptorRole::ImageStorageTable, ShaderRecompiler::DescriptorKind::StorageImage));
    const auto bda = Census::Facts(address, 0).shape;
    Require(bda == (Census::Bda | Census::BdaWrites | Census::Bindless | Census::Unbounded), "address-based program classified wrong: " + Census::ShapeName(bda));
    Require(Census::ShapeName(0) == "none" && Census::ShapeName(Census::Bda | Census::Sampled) == "bda+sampled", "shape names");
}

void drawFactsJoinStages() {
    Census::DrawFacts draw;
    Census::AddStage(draw, ShaderRecompiler::ShaderStage::Vertex, {Census::Buffers, 12, 64, 3});
    Census::AddStage(draw, ShaderRecompiler::ShaderStage::Fragment, {Census::Sampled, 9, 200, 0});
    Require(draw.shape == (Census::Buffers | Census::Sampled) && draw.userWords == 21 && draw.pushEnd == 200 && draw.vertexInputs == 3, "stages not joined");
    Require(Census::StagesName(draw.stages) == "vs/ps", "stage names: " + Census::StagesName(draw.stages));
}

void mappedWalksContiguousRanges() {
    const std::vector<Range> table{{0x1000, 0x2000, 0xa000, 3, 0}, {0x2000, 0x3000, 0x50000, 1, 0}, {0x4000, 0x5000, 0xb000, 3, 0}};
    Require(Census::Mapped(table, 0x1800, 0x2800), "a range across two adjacent entries is not mapped");
    Require(!Census::Mapped(table, 0x2800, 0x4100), "a range over a hole is mapped");
    Require(!Census::Mapped(table, 0x0, 0x1100), "a range below the table is mapped");
    Require(!Census::Mapped(table, 0x4800, 0x5001), "a range past the last entry is mapped");
    Require(Census::Mapped(table, 0x4000, 0x4000), "an empty range is not mapped");
    Require(!Census::Mapped({}, 0x1000, 0x1001), "an empty table maps something");
}

void windowJoinsKindPhaseAndShape() {
    Census::Window window;
    std::array<double, Census::PhaseCount> phases{};
    phases[6] = 200;
    phases[8] = 30;
    Census::DrawFacts bda;
    bda.shape = Census::Bda | Census::BdaWrites;
    bda.stages = Census::StageVertex | Census::StagePixel;
    bda.pushEnd = 240;
    bda.bda = Census::Cover::Covered;
    bda.indexed = true;
    bda.index = Census::Cover::Covered;
    window.Add(3, bda, phases);
    phases.fill(0);
    phases[5] = 10;
    phases[8] = 20;
    Census::DrawFacts hit;
    hit.shape = Census::Sampled;
    hit.stages = Census::StageVertex | Census::StagePixel;
    hit.pushEnd = 128;
    hit.indirect = true;
    hit.arguments = Census::Cover::Missing;
    window.Add(1, hit, phases);
    window.Add(1, hit, phases);
    Require(window.Draws() == 3 && window.KindDraws(1) == 2 && window.KindDraws(3) == 1, "kind counts");
    Require(window.KindPhaseUs(3, 6) == 200 && window.KindPhaseUs(1, 8) == 40, "kind x phase totals");
    Require(window.PushFits() == 2, "push fit: 128 + 24 fits, 240 + 24 does not");
    Require(window.CoverCountOf(0, Census::Cover::Covered) == 1 && window.CoverCountOf(1, Census::Cover::Covered) == 1 && window.CoverCountOf(2, Census::Cover::Missing) == 2, "cover counts");
    const auto line = window.Format("frame-2");
    Require(line.rfind("[newpath-draws] window=frame-2 draws 3 (indexed 1, indirect 2, instanced 0)", 0) == 0, "line head: " + line);
    Require(line.find("template 2 x 30.0") != std::string::npos && line.find("bda 1 x 230.0") != std::string::npos, "kind averages: " + line);
    Require(line.find("bda{resources 200.0 record 30.0}") != std::string::npos, "kind x phase: " + line);
    Require(line.find("vs/ps|bda+bda-write 1 0.2 0/0/1 0 1;") != std::string::npos, "top shape (most ms first): " + line);
    Require(line.find("vs/ps|sampled 2 0.1 2/0/0 2 0;") != std::string::npos, "second shape: " + line);
    Require(line.find("bda table F4: covered 1") != std::string::npos && line.find("indirect arguments mapped: missing 2") != std::string::npos, "covers: " + line);
    const auto top1 = window.Format("x", 1);
    Require(top1.find(" rest 2 0.1;") != std::string::npos, "the shapes past the top are summed: " + top1);
}

}

int main() {
    const std::array<std::pair<const char*, void (*)()>, 4> tests{{
        {"stage facts from bindings", stageFactsFromBindings},
        {"draw facts join stages", drawFactsJoinStages},
        {"mapped walks contiguous ranges", mappedWalksContiguousRanges},
        {"window joins kind, phase and shape", windowJoinsKindPhaseAndShape},
    }};
    int failed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::printf("PASS %s\n", name);
        } catch (const std::exception& error) {
            ++failed;
            std::printf("FAIL %s: %s\n", name, error.what());
        }
    }
    return failed == 0 ? 0 : 1;
}
