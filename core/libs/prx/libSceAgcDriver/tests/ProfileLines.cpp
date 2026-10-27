// The profile lines a frame replay prints per frame window: [command-mix]
// (APS5_PROFILE_COMMAND_MIX) and [resident-census] (APS5_PROFILE_RESIDENT_CENSUS), and the frame
// windows themselves (ProfileWindows).
#include "prx/libSceAgcDriver/Execution/include/CommandMix.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidentCensus.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include "IntermediateRepresentation/IrValue.hpp"
#include "Recompiler.hpp"
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using AgcDriver::Graphics::ResidentCensus::Class;
using AgcDriver::Graphics::ResidentCensus::Model;
using AgcDriver::Graphics::ResidentCensus::PageBytes;

void check(bool condition, const std::string& reason) {
    if (!condition) throw std::runtime_error(reason);
}

void contains(const std::string& line, const std::string& part) {
    check(line.find(part) != std::string::npos, "'" + part + "' missing from: " + line);
}

void testCommandMixLine() {
    AgcDriver::CommandMix::Window window;
    AgcDriver::CommandMix::DrawUse indexed;
    indexed.indexed = true;
    indexed.instances = 4;
    window.AddDraw(indexed);
    AgcDriver::CommandMix::DrawUse plain;
    plain.recipe = true;
    window.AddDraw(plain);
    AgcDriver::CommandMix::DrawUse indirect;
    indirect.indirect = true;
    indirect.multi = true;
    indirect.countIndirect = true;
    indirect.instances = 0;
    window.AddDraw(indirect);
    window.AddDispatch(false, true);
    window.AddDispatch(true, false);
    AgcDriver::CommandMix::StageUse vertex;
    vertex.userSgprs = 3;
    vertex.srtDepth = 2;
    vertex.descriptors[1] = 2;
    vertex.descriptors[4] = 5;
    window.AddStage(true, vertex);
    AgcDriver::CommandMix::StageUse fragment;
    fragment.userSgprs = 20;
    fragment.bda = true;
    fragment.bindlessTables = 1;
    window.AddStage(true, fragment);
    AgcDriver::CommandMix::StageUse compute;
    compute.srtDepth = 7;
    window.AddStage(false, compute);
    check(window.Draws() == 3 && window.Dispatches() == 2, "command mix counts");
    const auto line = window.Format("frame-2", " Inserted 1");
    contains(line, "[command-mix] window=frame-2 draws 3 (indexed 1, non-indexed 1, indirect 1 (multi 1, count 1), instanced 1, recipe 1)");
    contains(line, "dispatches 2 (direct 1, indirect 1, cache hits 1)");
    contains(line, "draw stages 2: user-sgprs 0:0 1-2:0 3-4:1 5-8:0 9-16:0 17+:1 mean 11.5 max 20; srt-depth 0:0 1:0 2:1 3:0 4+:0 unknown:1; descriptors ub 0 sb 2 utb 0 stb 0 sampled 5 storage-image 0 sampler 0; bindless-tables 1; bda 1");
    contains(line, "dispatch programs 1: user-sgprs 0:1");
    contains(line, "srt-depth 0:0 1:0 2:0 3:0 4+:1 unknown:0");
    contains(line, "; bda-template builds: Inserted 1");
    check(line.find('\n') == std::string::npos, "the command mix is one line");
}

void testClassify() {
    ShaderRecompiler::RecompileResult program;
    program.userDataRegisterReads = 0b1011;
    ShaderRecompiler::DescriptorBinding buffers{};
    buffers.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    buffers.count = 3;
    ShaderRecompiler::DescriptorBinding sampler{};
    sampler.kind = ShaderRecompiler::DescriptorKind::Sampler;
    sampler.count = 0;
    program.bindings = {buffers, sampler};
    program.bdaAbiVersion = 2;
    const auto use = AgcDriver::CommandMix::Classify(program);
    check(use.userSgprs == 3, "user SGPRs are the user data registers read");
    check(use.descriptors[1] == 3 && use.descriptors[6] == 1, "descriptors by kind, array counts summed");
    check(use.bda && use.srtDepth == -1, "a BDA program of unknown depth");
}

void testSrtDepth() {
    ShaderRecompiler::IrResourcePlan plan;
    check(AgcDriver::CommandMix::SrtDepth(plan) == 0, "a plan without SRT reads has depth 0");
    // slot 0: a read through a user SGPR pointer; slot 1: a read through slot 0's value plus an
    // offset; slot 2: a read through slot 1.
    ShaderRecompiler::IrValue user(ShaderRecompiler::IrOpcode::GetUserData, ShaderRecompiler::IrType::U32, 1);
    ShaderRecompiler::IrValue offset0(ShaderRecompiler::IrOpcode::GetUserData, ShaderRecompiler::IrType::U32, 2);
    ShaderRecompiler::IrValue read0(ShaderRecompiler::IrOpcode::ReadConst, ShaderRecompiler::IrType::U32, 3);
    read0.AddArgument(&user);
    read0.AddArgument(&offset0);
    const auto slotValue = [](ShaderRecompiler::IrValue& resource, ShaderRecompiler::IrValue& slot, ShaderRecompiler::IrValue& read, std::uint32_t index) {
        slot.SetImmediateU32(index);
        read.AddArgument(&resource);
        read.AddArgument(&slot);
    };
    ShaderRecompiler::IrValue resource(ShaderRecompiler::IrOpcode::GetSrtResource, ShaderRecompiler::IrType::SrtResource, 4);
    ShaderRecompiler::IrValue slot0(ShaderRecompiler::IrOpcode::GetUserData, ShaderRecompiler::IrType::U32, 5);
    ShaderRecompiler::IrValue flat0(ShaderRecompiler::IrOpcode::ReadConst, ShaderRecompiler::IrType::U32, 6);
    slotValue(resource, slot0, flat0, 0);
    ShaderRecompiler::IrValue sum(ShaderRecompiler::IrOpcode::IAdd32, ShaderRecompiler::IrType::U32, 7);
    sum.AddArgument(&flat0);
    sum.AddArgument(&offset0);
    ShaderRecompiler::IrValue read1(ShaderRecompiler::IrOpcode::ReadConst, ShaderRecompiler::IrType::U32, 8);
    read1.AddArgument(&sum);
    read1.AddArgument(&offset0);
    ShaderRecompiler::IrValue slot1(ShaderRecompiler::IrOpcode::GetUserData, ShaderRecompiler::IrType::U32, 9);
    ShaderRecompiler::IrValue flat1(ShaderRecompiler::IrOpcode::ReadConst, ShaderRecompiler::IrType::U32, 10);
    slotValue(resource, slot1, flat1, 1);
    ShaderRecompiler::IrValue read2(ShaderRecompiler::IrOpcode::ReadConst, ShaderRecompiler::IrType::U32, 11);
    read2.AddArgument(&flat1);
    read2.AddArgument(&offset0);
    plan.srtReads.push_back({&read0, 0});
    check(AgcDriver::CommandMix::SrtDepth(plan) == 1, "one table read through a user SGPR is depth 1");
    plan.srtReads.push_back({&read1, 1});
    check(AgcDriver::CommandMix::SrtDepth(plan) == 2, "a table read through a table is depth 2");
    plan.srtReads.push_back({&read2, 2});
    check(AgcDriver::CommandMix::SrtDepth(plan) == 3, "three nested tables are depth 3");
}

void testCensusSplit() {
    Model model;
    const std::uint64_t base = 0x5000000000ull;
    // Page 0 written by a use, page 1 read after a CPU write, pages 2-3 read only (staged), page 4
    // written by the CPU only after the GPU read it (read-only).
    model.GpuWrite(Class::Staged, base, base + 100);
    model.CpuWrite(base + PageBytes, base + PageBytes + 8);
    model.GpuRead(Class::Staged, base + PageBytes, base + 4 * PageBytes);
    model.GpuRead(Class::StorageImage, base + 4 * PageBytes, base + 5 * PageBytes);
    model.CpuWrite(base + 4 * PageBytes, base + 4 * PageBytes + 4);
    // A group (an address space) noted once, then only touched again.
    const int space = 0;
    check(!model.GroupSeen(&space), "a new group is not yet seen");
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> reach{{base, base + 2 * PageBytes}, {base + 8 * PageBytes, base + 9 * PageBytes}};
    model.GpuReadGroup(&space, Class::BdaLeased, reach);
    check(model.GroupSeen(&space), "a noted group is seen");
    const auto first = model.Close();
    check(first.total.Pages() == 6, "six distinct pages, got " + std::to_string(first.total.Pages()));
    check(first.total.written == 1 && first.total.cpuThenGpu == 1 && first.total.readOnly == 4, "split written/cpu-then-gpu/read-only");
    const auto& staged = first.byClass[static_cast<std::size_t>(Class::Staged)];
    check(staged.written == 1 && staged.cpuThenGpu == 1 && staged.readOnly == 2, "staged class split");
    const auto& leased = first.byClass[static_cast<std::size_t>(Class::BdaLeased)];
    check(leased.Pages() == 3 && leased.written == 1, "a page counts in every class it is in");
    check(first.unionTotal.Pages() == 6, "the union after one window");
    for (const auto evicted : first.evicted) check(evicted == 0, "nothing evicted below the LRU size");
    const auto line = Model::Format("frame-1", first);
    contains(line, "[resident-census] window=frame-1 6 pages = ");
    contains(line, " MiB (gpu-written 0.1, cpu-then-gpu 0.1, read-only ");
    contains(line, "; by class written/cpu-then-gpu/read-only MiB: staged 0.1/0.1/0.1, in-place 0.0/0.0/0.0, bda-leased ");
    contains(line, "lru evictions (pages) 512 MiB 0, 1024 MiB 0, 2048 MiB 0, 4096 MiB 0; union 6 pages");
    // The next window: the group is forgotten, the union keeps the first window's pages.
    check(!model.GroupSeen(&space), "groups are per window");
    model.GpuRead(Class::Indirect, base + 20 * PageBytes, base + 20 * PageBytes + 12);
    const auto second = model.Close();
    check(second.total.Pages() == 1 && second.unionTotal.Pages() == 7 && second.unionTotal.written == 1, "the union grows across windows");
}

void testCensusLru() {
    Model model;
    const std::uint64_t base = 0x5100000000ull;
    // 600 MiB of pages in one window, then the first 100 MiB again: the 512 MiB LRU evicts.
    const std::uint64_t pages = 600u * 1024u * 1024u / PageBytes;
    model.GpuRead(Class::Staged, base, base + pages * PageBytes);
    const auto first = model.Close();
    check(first.evicted[0] == pages - 512u * 1024u * 1024u / PageBytes, "512 MiB evicts the overflow, got " + std::to_string(first.evicted[0]));
    check(first.evicted[1] == 0 && first.evicted[2] == 0 && first.evicted[3] == 0, "larger LRUs keep it all");
    model.GpuRead(Class::Staged, base, base + 1600 * PageBytes);
    const auto second = model.Close();
    check(second.evicted[0] == 1600, "re-reading evicted pages evicts as many, got " + std::to_string(second.evicted[0]));
    check(second.evicted[1] == 0, "1 GiB holds both windows");
}

void testFrameWindows() {
    using AgcDriver::ProfileWindows::ClockDue;
    auto past = std::chrono::steady_clock::now() - std::chrono::seconds(11);
    check(ClockDue(past), "a clock window closes after 10 s");
    check(!ClockDue(past), "and restarts");
    AgcDriver::ProfileWindows::UseFrameWindows(true);
    auto old = std::chrono::steady_clock::now() - std::chrono::seconds(60);
    check(!ClockDue(old), "frame windows never close on the clock");
    // Nothing is enabled in this test: closing prints nothing and returns.
    AgcDriver::ProfileWindows::Close("frame-1");
    AgcDriver::ProfileWindows::UseFrameWindows(false);
    check(ClockDue(old), "clock windows again");
}

}

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "all";
        if (mode == "all" || mode == "command_mix") {
            testCommandMixLine();
            testClassify();
            testSrtDepth();
        }
        if (mode == "all" || mode == "census") {
            testCensusSplit();
            testCensusLru();
        }
        if (mode == "all" || mode == "windows") testFrameWindows();
        std::puts("profile lines test passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
