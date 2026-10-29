#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DrawCensusHooks.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DrawCensus.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DevicePageTable.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "Recompiler.hpp"
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

namespace AgcDriver::NewPath::DrawCensus {

namespace {

using Range = ShaderRecompiler::BdaAbi::Range;

struct State {
    std::mutex mutex;
    Window window;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
    // The device page table's ranges as of `version` (its builds + patches), copied once per change.
    std::mutex tableMutex;
    std::uint64_t version = ~std::uint64_t{0};
    std::shared_ptr<const std::vector<Range>> table;
};

State& Get() {
    static auto* state = new State;
    return *state;
}

struct Stash {
    bool valid = false;
    DrawFacts facts;
};

Stash& ThreadStash() {
    thread_local Stash stash;
    return stash;
}

std::shared_ptr<const std::vector<Range>> tableRanges() {
    auto& state = Get();
    auto& device = DeviceTable();
    const auto totals = device.Totals();
    const auto version = totals.builds + totals.patches;
    std::lock_guard lock(state.tableMutex);
    if (version != state.version || state.table == nullptr) {
        state.table = std::make_shared<const std::vector<Range>>(device.Ranges());
        state.version = version;
    }
    return state.table;
}

Cover mappedCover(const std::vector<Range>& table, std::uint64_t begin, std::uint64_t bytes) {
    if (table.empty()) return Cover::NoTable;
    return Mapped(table, begin, begin + bytes) ? Cover::Covered : Cover::Missing;
}

void print(State& state, const char* window) {
    if (state.window.Draws() == 0) return;
    const auto line = state.window.Format(window);
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
    state.window = Window{};
}

}

void Begin(std::span<const Graphics::CompiledShader> shaders, const Pm4::DrawParameters& draw) {
    if (!Enabled()) return;
    auto& stash = ThreadStash();
    stash.valid = true;
    stash.facts = DrawFacts{};
    auto& facts = stash.facts;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) continue;
        AddStage(facts, shader.stage, Facts(*shader.program, shader.pushConstantOffset));
    }
    facts.indexed = draw.indexed;
    facts.indirect = draw.indirect.has_value();
    facts.instanced = !facts.indirect && draw.instanceCount > 1;
    const auto table = tableRanges();
    const auto indexBytes = static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize;
    if (draw.indexed && draw.indexAddress != 0 && indexBytes != 0) facts.index = mappedCover(*table, draw.indexAddress, indexBytes);
    if (draw.indirect) {
        facts.arguments = mappedCover(*table, draw.indirect->arguments, draw.indirect->RangeBytes());
        if (facts.arguments == Cover::Covered && draw.indirect->countIndirect) facts.arguments = mappedCover(*table, draw.indirect->countAddress, 4);
    }
}

void NoteBuild(const Graphics::ShaderResources& resources) {
    if (!Enabled()) return;
    auto& stash = ThreadStash();
    if (!stash.valid || !resources.UsesAddressTable()) return;
    const auto table = tableRanges();
    const auto ranges = resources.AddressTableRanges();
    if (table->empty()) stash.facts.bda = Cover::NoTable;
    else if (ranges == nullptr) stash.facts.bda = Cover::Missing;
    else {
        switch (DevicePageTable::Compare(*table, *ranges)) {
        case DevicePageTable::Cover::Covered: stash.facts.bda = Cover::Covered; break;
        case DevicePageTable::Cover::Missing: stash.facts.bda = Cover::Missing; break;
        case DevicePageTable::Cover::OtherAddress: stash.facts.bda = Cover::OtherAddress; break;
        case DevicePageTable::Cover::Narrower: stash.facts.bda = Cover::Narrower; break;
        }
    }
}

void End(std::size_t kind, std::span<const double> phaseUs) {
    if (!Enabled()) return;
    auto& stash = ThreadStash();
    if (!stash.valid) return;
    stash.valid = false;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    state.window.Add(kind, stash.facts, phaseUs);
    if (ProfileWindows::ClockDue(state.lastReport)) print(state, "10 s");
}

void Close(const char* window) {
    if (!Enabled()) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    print(state, window);
}

}
