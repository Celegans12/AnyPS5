#include "prx/libSceAgcDriver/Execution/include/CommandLog.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DrawPipeline;
using Counts = AgcDriver::DriverDetail::Driver::DispatchSlotCounts;

constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t BufferBytes = Lanes * 4u;
constexpr std::uint32_t DispatchOpcode = 0x15;
constexpr std::size_t BlockBytes = 65536;

constexpr std::array<std::uint32_t, 8> Program{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u, 0x4a060481u, 0xe0701000u, 0x80010301u, 0xbf810000u};
using Code = std::array<std::uint32_t, 8>;
const Code* CodeBlocks[9]{};

std::uint32_t* Input = nullptr;
std::uint32_t* Output = nullptr;
std::uint32_t* Scratch = nullptr;
std::uint32_t* Chained = nullptr;
std::uint32_t* Other = nullptr;

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("dispatch slots: " + what);
}

std::uint32_t* NewBlock() {
    static std::uintptr_t next = 0x5a0000000ull;
    const auto wanted = next;
    next += BlockBytes;
#ifdef _WIN32
    auto* block = static_cast<std::uint32_t*>(VirtualAlloc(reinterpret_cast<void*>(wanted), BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
    if (block == nullptr) block = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
#else
    static_cast<void>(wanted);
    auto* block = static_cast<std::uint32_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
    check(block != nullptr, "cannot allocate a guest block");
    std::memset(block, 0, BlockBytes);
    GuestAllocations::Mutation().Add(block, BlockBytes, true, true, false);
    return block;
}

template <typename Ready>
void WaitUntil(Ready ready, const std::string& what) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!ready()) {
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("dispatch slots: timed out waiting for " + what);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void Register(Shader& shader, const Code& code) {
    shader = Shader{};
    shader.file_header = 0x34333231;
    shader.version = 0x18;
    shader.header_size = sizeof(Shader);
    shader.shader_size = sizeof(code);
    shader.code = code.data();
    AgcDriverRegisterShader_nid_postfix(&shader);
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

void AppendFill(std::vector<std::uint32_t>& words, std::uint32_t value, const void* destination) {
    const auto address = reinterpret_cast<std::uintptr_t>(destination);
    const std::array<std::uint32_t, 7> fill{0xc0055000u, 2u << 29u, value, 0u, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), BufferBytes};
    words.insert(words.end(), fill.begin(), fill.end());
}

void AppendDispatch(std::vector<std::uint32_t>& words, const Code& code, const void* source, const void* destination) {
    const auto codeAddress = reinterpret_cast<std::uintptr_t>(code.data());
    const auto input = BufferDescriptor(source, BufferBytes);
    const auto output = BufferDescriptor(destination, BufferBytes);
    const std::vector<std::uint32_t> commands{
        0xc0027600u, 0x20cu, static_cast<std::uint32_t>(codeAddress >> 8u), static_cast<std::uint32_t>(codeAddress >> 40u),
        0xc0037600u, 0x207u, Lanes, 1u, 1u,
        0xc0017600u, 0x213u, 8u << 1u,
        0xc0087600u, 0x240u, input[0], input[1], input[2], input[3], output[0], output[1], output[2], output[3],
        0xc0031500u, 1u, 1u, 1u, 0x8041u,
    };
    words.insert(words.end(), commands.begin(), commands.end());
}

void Submit(std::vector<std::uint32_t>& words) {
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&packet) == 0, "submission failed");
}

void Settle() {
    AgcDriverWaitIdle_nid_postfix();
    for (auto* block : {Input, Output, Scratch, Chained, Other}) AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(block), BufferBytes);
}

class Gate {
public:
    Gate() {
        WaitUntil([] { return DrawPipeline::QueuedItems() == 0; }, "the pipeline to empty before a gated round");
        DrawPipeline::Queue0().Enqueue([flag = released] {
            while (!flag->load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }, {});
    }
    ~Gate() { Release(); }
    Gate(const Gate&) = delete;
    Gate& operator=(const Gate&) = delete;
    void Release() { released->store(true, std::memory_order_release); }
private:
    std::shared_ptr<std::atomic<bool>> released = std::make_shared<std::atomic<bool>>(false);
};

bool Env(const char* name) {
    return std::getenv(name) != nullptr;
}

struct Mode {
    bool slots = !AgcDriver::Graphics::RuntimeSwitches::Set(AgcDriver::Graphics::RuntimeSwitch::NoDispatchSlots);
    bool ahead = slots && !AgcDriver::Graphics::RuntimeSwitches::Set(AgcDriver::Graphics::RuntimeSwitch::NoDispatchSlotPrepare);
    bool verify = Env("APS5_VERIFY_DISPATCH_SLOTS");
};

bool GatedRound(std::vector<std::uint32_t> words, std::size_t dispatches, bool waitPrepared) {
    const auto before = AgcDriver::DriverDetail::Driver::DispatchSlotStatistics();
    const auto drainsBefore = DrawPipeline::DrainRequestsByOpcode(DispatchOpcode);
    bool enqueued = false;
    {
        Gate gate;
        Submit(words);
        WaitUntil([&] { return AgcDriver::DriverDetail::Driver::DispatchSlotStatistics().slots >= before.slots + dispatches || DrawPipeline::DrainRequestsByOpcode(DispatchOpcode) > drainsBefore; }, "the round's dispatches enqueued or a drain");
        enqueued = DrawPipeline::DrainRequestsByOpcode(DispatchOpcode) == drainsBefore;
        if (waitPrepared && enqueued) WaitUntil([&] { return AgcDriver::DriverDetail::Driver::DispatchSlotStatistics().preparedAhead > before.preparedAhead; }, "the preparer to prepare a slot ahead");
    }
    Settle();
    return enqueued;
}

void ExpectLanes(const std::uint32_t* block, std::uint32_t expected, const std::string& what) {
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        check(block[lane] == expected, what + ": lane " + std::to_string(lane) + " is " + std::to_string(block[lane]) + ", expected " + std::to_string(expected));
    }
}

void Scenario(const Mode& mode, bool cold, bool expectSlots, std::size_t set = 0) {
    const Code& CodeA = *CodeBlocks[set * 3];
    const Code& CodeB = *CodeBlocks[set * 3 + 1];
    const Code& CodeC = *CodeBlocks[set * 3 + 2];
    for (auto* block : {Input, Output, Scratch, Chained, Other}) std::memset(block, 0, BufferBytes);
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) Input[lane] = 0x1000u;
    {
        std::vector<std::uint32_t> words;
        AppendFill(words, 0x2220u, Other);
        AppendDispatch(words, CodeB, Input, Output);
        const auto before = AgcDriver::DriverDetail::Driver::DispatchSlotStatistics();
        const bool slotted = GatedRound(words, 1, mode.ahead && cold);
        const auto after = AgcDriver::DriverDetail::Driver::DispatchSlotStatistics();
        ExpectLanes(Output, 0x1001u, "ahead round");
        std::printf("dispatch slots: ahead round slotted %d, prepared ahead %llu, adopted %llu, discarded %llu, drains avoided %llu\n", slotted ? 1 : 0, static_cast<unsigned long long>(after.preparedAhead - before.preparedAhead), static_cast<unsigned long long>(after.adopted - before.adopted), static_cast<unsigned long long>(after.discarded - before.discarded), static_cast<unsigned long long>(after.drainsAvoided - before.drainsAvoided));
        if (expectSlots) check(slotted && after.drainsAvoided > before.drainsAvoided, "the dispatch behind the gate drained instead of taking a slot");
        if (mode.slots) check(slotted, "with dispatch slots on, a dispatch drained the pipeline");
        else check(!slotted, "with APS5_NO_DISPATCH_SLOTS and items off, a dispatch did not drain");
        if (mode.ahead && cold) check(after.adopted == before.adopted + 1, "the slot's stage A, made ahead with nothing pending over its inputs, was not adopted");
        if (!mode.ahead) check(after.adopted == before.adopted && after.preparedAhead == before.preparedAhead, "stage A was made ahead with APS5_NO_DISPATCH_SLOT_PREPARE set");
    }
    const std::array<std::uint32_t, 3> fills{0x5150u, 0x6160u, 0x7170u};
    for (std::size_t round = 0; round < fills.size(); ++round) {
        std::vector<std::uint32_t> words;
        AppendFill(words, fills[round], Input);
        AppendDispatch(words, CodeA, Input, Output);
        const auto before = AgcDriver::DriverDetail::Driver::DispatchSlotStatistics();
        const bool slotted = GatedRound(words, 1, false);
        const auto after = AgcDriver::DriverDetail::Driver::DispatchSlotStatistics();
        ExpectLanes(Input, fills[round], "pending fill round " + std::to_string(round) + " input");
        ExpectLanes(Output, fills[round] + 1u, "pending fill round " + std::to_string(round) + " (the slot did not see the fill queued before it)");
        std::printf("dispatch slots: pending fill round %zu slotted %d, prepared ahead %llu\n", round, slotted ? 1 : 0, static_cast<unsigned long long>(after.preparedAhead - before.preparedAhead));
        if (mode.slots) check(slotted && after.preparedAhead == before.preparedAhead, "a slot over a pending fill drained or was prepared ahead");
        if (expectSlots) check(slotted, "the dispatch behind a pending fill drained instead of taking a slot");
    }
    {
        std::vector<std::uint32_t> words;
        AppendDispatch(words, CodeB, Input, Output);
        AppendDispatch(words, CodeC, Output, Chained);
        const bool slotted = GatedRound(words, 2, false);
        ExpectLanes(Output, fills.back() + 1u, "chain round, first dispatch");
        ExpectLanes(Chained, fills.back() + 2u, "chain round (the second slot did not read the first slot's result)");
        std::printf("dispatch slots: chain round slotted %d\n", slotted ? 1 : 0);
        if (mode.slots) check(slotted, "a dispatch reading an earlier slot's output drained");
    }
    {
        const auto old = Input[0];
        std::vector<std::uint32_t> words;
        AppendDispatch(words, CodeB, Input, Output);
        AppendFill(words, 0x9190u, Input);
        AppendDispatch(words, CodeC, Input, Scratch);
        const bool slotted = GatedRound(words, 2, false);
        ExpectLanes(Output, old + 1u, "write-after-read round (the first slot saw the later fill)");
        ExpectLanes(Scratch, 0x9191u, "write-after-read round (the second slot missed the fill before it)");
        std::printf("dispatch slots: write-after-read round slotted %d\n", slotted ? 1 : 0);
        if (mode.slots) check(slotted, "a dispatch in the write-after-read round drained");
    }
}

}

int main(int argc, char** argv) {
#ifdef _WIN32
    _putenv_s("APS5_PIPELINED_DRAWS", "1");
#else
    setenv("APS5_PIPELINED_DRAWS", "1", 1);
#endif
    const bool expectSlots = argc > 1 && std::string_view(argv[1]) == "--expect-slots";
    const bool compareLog = argc > 1 && std::string_view(argv[1]) == "--log";
    try {
        {
            const auto probe = OpenVulkanTestDevice();
            if (!probe) return VulkanTestSkipped;
        }
        auto* codeBlock = NewBlock();
        for (std::size_t program = 0; program < 9; ++program) {
            auto* code = reinterpret_cast<Code*>(codeBlock + program * 64u);
            *code = Program;
            CodeBlocks[program] = code;
        }
        std::array<Shader, 9> shaders{};
        for (std::size_t program = 0; program < 9; ++program) Register(shaders[program], *CodeBlocks[program]);
        Input = NewBlock();
        Output = NewBlock();
        Scratch = NewBlock();
        Chained = NewBlock();
        Other = NewBlock();
        const Mode mode;
        Scenario(mode, true, expectSlots);

        const auto totals = AgcDriver::DriverDetail::Driver::DispatchSlotStatistics();
        std::printf("dispatch slots: totals %llu slots (%llu misses, %llu hits, %llu with recipe), %llu drains avoided, %llu fallbacks, %llu prepared ahead, %llu adopted, %llu discarded, verify %llu checked %llu differing\n", static_cast<unsigned long long>(totals.slots), static_cast<unsigned long long>(totals.misses), static_cast<unsigned long long>(totals.hits), static_cast<unsigned long long>(totals.hitsWithRecipe), static_cast<unsigned long long>(totals.drainsAvoided), static_cast<unsigned long long>(totals.fallbacks), static_cast<unsigned long long>(totals.preparedAhead), static_cast<unsigned long long>(totals.adopted), static_cast<unsigned long long>(totals.discarded), static_cast<unsigned long long>(totals.verified), static_cast<unsigned long long>(totals.verifyMismatches));
        if (mode.verify && mode.ahead) check(totals.verified != 0 && totals.verifyMismatches == 0, "APS5_VERIFY_DISPATCH_SLOTS checked nothing or found a difference");
        if (compareLog) {
            const auto streams = [](const std::string& line) { return line.substr(line.find(':')); };
            for (const bool cold : {false, true}) {
                std::array<std::string, 2> lines;
                for (int pass = 0; pass < 2; ++pass) {
                    AgcDriver::Graphics::RuntimeSwitches::Assign(AgcDriver::Graphics::RuntimeSwitch::NoDispatchSlots, pass == 0);
                    Settle();
                    AgcDriver::CommandLog::Reset();
                    Scenario(Mode{}, false, false, cold ? static_cast<std::size_t>(pass + 1) : 0u);
                    Settle();
                    lines[pass] = AgcDriver::CommandLog::Line(static_cast<std::uint64_t>(pass + 1));
                    std::printf("dispatch slots: %s command log with slots %s: %s", cold ? "cold" : "warm", pass == 0 ? "off" : "on", lines[pass].c_str());
                }
                check(streams(lines[0]) == streams(lines[1]), std::string("the ") + (cold ? "cold" : "warm") + " command streams with dispatch slots on differ from those with them off");
            }
        }
        LibcRunShutdown_nid_postfix();
        std::puts("dispatch slot tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
