#include "prx/libSceAgc/DcbFlow/include/Memory.hpp"
#include "prx/libSceAgc/DcbFlow/include/Sync.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "SceTypes.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using AgcDriver::FrameCallKind;
using AgcDriver::FrameCaptureReader;
using AgcDriver::FrameMemoryKind;

constexpr std::uint32_t Handle = 7;
constexpr std::uint8_t GraphicsMemory = 0x0a;
constexpr std::uint8_t DmaMemory = 0;
constexpr std::size_t BlockBytes = 1u << 20;
constexpr std::size_t CommandBytes = 64u * 1024u;
constexpr std::size_t DataBytes = 4096;
constexpr std::uint32_t Done = 1;

void check(bool condition, const std::string& reason) {
    if (!condition) throw std::runtime_error(reason);
}

std::uint64_t at(const volatile void* pointer) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

class Request final : public AgcDriver::IFlipRequest {
public:
    explicit Request(std::atomic<int>& ready) : ready(ready) {}
    void GpuReady(const std::shared_ptr<AgcDriver::FrameTiming>&) override { ++ready; }
    void Fail(std::exception_ptr error) noexcept override {
        if (!error) std::terminate();
    }

private:
    std::atomic<int>& ready;
};

class Output final : public AgcDriver::IVideoOutput {
public:
    std::shared_ptr<AgcDriver::IFlipRequest> Reserve(const AgcDriver::FlipInfo&) override { return std::make_shared<Request>(ready); }
    void Fail(std::exception_ptr error) noexcept override {
        if (!error) std::terminate();
    }
    std::atomic<int> ready = 0;
};

struct Layout {
    std::byte* base = nullptr;
    std::uint32_t* firstCommands = nullptr;
    std::uint32_t* secondCommands = nullptr;
    std::byte* source = nullptr;
    std::byte* firstTarget = nullptr;
    std::byte* secondTarget = nullptr;
    std::uint32_t* label = nullptr;
};

Layout allocate() {
    auto* const block = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536));
    check(block != nullptr, "cannot allocate guest memory");
    GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, 0x04, BlockBytes);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, BlockBytes, true, true, true);
    }
    Layout layout;
    layout.base = block;
    layout.firstCommands = reinterpret_cast<std::uint32_t*>(block);
    layout.secondCommands = reinterpret_cast<std::uint32_t*>(block + CommandBytes);
    layout.source = block + 2 * CommandBytes;
    layout.firstTarget = layout.source + DataBytes;
    layout.secondTarget = layout.firstTarget + DataBytes;
    layout.label = reinterpret_cast<std::uint32_t*>(layout.secondTarget + DataBytes);
    return layout;
}

std::vector<std::byte> pattern(std::uint8_t seed) {
    std::vector<std::byte> bytes(DataBytes);
    for (std::size_t index = 0; index < bytes.size(); ++index) bytes[index] = static_cast<std::byte>((index * 31 + seed) & 0xff);
    return bytes;
}

void copy(std::uint32_t* words, std::size_t commandBytes, const std::byte* source, std::byte* target, std::uint32_t* label) {
    CommandBuffer buffer{words, words + commandBytes / sizeof(std::uint32_t), words, words + commandBytes / sizeof(std::uint32_t), nullptr, nullptr, 0};
    sceAgcDcbDmaData(&buffer, 0, DmaMemory, 0, at(target), DmaMemory, 0, at(source), DataBytes, 0, 1, 0);
    sceAgcDcbWriteData(&buffer, GraphicsMemory, 0, at(label), &Done, 1, 0, 1);
    Packet packet{words, static_cast<std::uint32_t>(buffer.cursor_up - words), 0, {}};
    check(sceAgcDriverSubmitDcb(&packet) == 0, "submission failed");
}

void copyFrame(std::uint32_t* words, const Layout& layout, std::byte* target) {
    copy(words, CommandBytes, layout.source, target, layout.label);
}

void flip() {
    AgcDriverSubmitFlip_nid_postfix(AgcDriver::FlipInfo{Handle, -1, 1, 0});
}

std::vector<std::byte> readFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    check(file.is_open(), "cannot open " + path.string());
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<std::byte> result(bytes.size());
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}

bool equal(const std::byte* bytes, const std::vector<std::byte>& expected) {
    return std::memcmp(bytes, expected.data(), expected.size()) == 0;
}

void testCaptureAndReplay(const std::filesystem::path& tool) {
    const char* directory = std::getenv("APS5_FRAME_CAPTURE");
    check(directory != nullptr && *directory != '\0', "APS5_FRAME_CAPTURE is not set");
    const std::filesystem::path capture(directory);
    const auto layout = allocate();
    const auto first = pattern(3);
    const auto second = pattern(101);
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(Handle, output);
    AgcDriverUnregisterVideoOutput_nid_postfix(Handle, output);
    output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(Handle, output);

    flip();
    std::memcpy(layout.source, first.data(), first.size());
    copyFrame(layout.firstCommands, layout, layout.firstTarget);
    flip();
    AgcDriverWaitIdle_nid_postfix();
    AgcDriverUnregisterVideoOutput_nid_postfix(Handle, output);
    output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(Handle, output);
    const auto late = allocate();
    std::memcpy(late.source, second.data(), second.size());
    copyFrame(late.firstCommands, late, late.firstTarget);
    flip();
    AgcDriverWaitIdle_nid_postfix();
    check(equal(layout.firstTarget, first) && equal(late.firstTarget, second), "the driver did not run the copies while capturing");

    const FrameCaptureReader reader(capture);
    check(!reader.Truncated(), "the capture is truncated");
    check(reader.Header().find("\"frames\":2") != std::string::npos, "the capture header does not record two frames: " + reader.Header());
    std::size_t submits = 0, flips = 0, memory = 0, starts = 0, registrations = 0, unregistrations = 0;
    for (const auto& call : reader.Calls()) {
        submits += call.kind == FrameCallKind::Submit;
        flips += call.kind == FrameCallKind::Flip;
        memory += call.kind == FrameCallKind::Memory;
        starts += call.kind == FrameCallKind::FrameStart;
        registrations += call.kind == FrameCallKind::RegisterVideoOutput;
        unregistrations += call.kind == FrameCallKind::UnregisterVideoOutput;
    }
    check(starts == 1 && submits == 2 && flips == 2, "the capture holds " + std::to_string(submits) + " submits and " + std::to_string(flips) + " flips after " + std::to_string(starts) + " frame starts");
    check(memory >= 2, "the CPU writes before each copy were not recorded as memory deltas");
    check(registrations == 3 && unregistrations == 2, "video output close and reopen was not recorded before and during capture");
    const auto recorded = [&](const std::byte* base) {
        return std::any_of(reader.Chunks().begin(), reader.Chunks().end(), [&](const auto& chunk) {
            return chunk.kind == FrameMemoryKind::Range && chunk.address <= at(base) && at(base) + BlockBytes <= chunk.address + chunk.bytes;
        });
    };
    check(recorded(layout.base), "the guest block is not in the snapshot ranges");
    check(recorded(late.base), "the block committed during the capture is not in its ranges");
    check(reader.Header().find("\"late_ranges\":1") != std::string::npos, "the capture header does not count one range committed during the capture: " + reader.Header());

    const auto firstDump = capture / "first.bin";
    const auto secondDump = capture / "second.bin";
    const auto command = "\"\"" + tool.string() + "\" \"" + capture.string() + "\" --strict --dump " + std::to_string(at(layout.firstTarget)) + ":" + std::to_string(DataBytes) + ":\"" + firstDump.string() +
                         "\" --dump " + std::to_string(at(late.firstTarget)) + ":" + std::to_string(DataBytes) + ":\"" + secondDump.string() + "\"\"";
    const auto code = std::system(command.c_str());
    check(code == 0, "agc_frame_replay exited with " + std::to_string(code));
    check(readFile(firstDump) == first, "the replay did not reproduce the first copy");
    check(readFile(secondDump) == second, "the replay did not reproduce the second copy");
}

void testConcurrentCopies(const std::filesystem::path& tool) {
    constexpr std::size_t Copies = 64;
    constexpr std::size_t Frames = 8;
    constexpr std::size_t CopyCommandBytes = 1024;
    const char* directory = std::getenv("APS5_FRAME_CAPTURE");
    check(directory != nullptr && *directory != '\0', "APS5_FRAME_CAPTURE is not set");
    const std::filesystem::path capture(directory);
    const auto layout = allocate();
    const auto other = allocate();
    auto* const otherSource = other.base + 128 * 1024;
    auto* const targets = otherSource + DataBytes;
    auto* const otherLabel = reinterpret_cast<std::uint32_t*>(other.base + 512 * 1024);
    const auto copied = pattern(57);
    std::memcpy(otherSource, copied.data(), copied.size());
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(Handle, output);

    flip();
    std::thread copier([&] {
        for (std::size_t index = 0; index < Copies; ++index) {
            copy(reinterpret_cast<std::uint32_t*>(other.base + index * CopyCommandBytes), CopyCommandBytes, otherSource, targets + index * DataBytes, otherLabel);
        }
    });
    for (std::size_t frame = 0; frame + 1 < Frames; ++frame) {
        const auto bytes = pattern(static_cast<std::uint8_t>(frame));
        std::memcpy(layout.source, bytes.data(), bytes.size());
        copyFrame(layout.firstCommands, layout, layout.firstTarget);
        flip();
    }
    copier.join();
    flip();
    AgcDriverWaitIdle_nid_postfix();

    const FrameCaptureReader reader(capture);
    check(!reader.Truncated(), "the capture is truncated");
    std::uint64_t base = 0;
    std::size_t submits = 0, memory = 0;
    for (const auto& call : reader.Calls()) {
        if (call.kind == FrameCallKind::FrameStart) base = call.completed;
        if (call.kind == FrameCallKind::Memory) {
            ++memory;
            check(call.completed >= base + submits, "a memory delta was read with " + std::to_string(base + submits - call.completed) + " of the " + std::to_string(submits) + " submissions recorded before it unfinished");
        }
        submits += call.kind == FrameCallKind::Submit;
    }
    check(submits == Copies + Frames - 1, "the capture holds " + std::to_string(submits) + " submits, not " + std::to_string(Copies + Frames - 1));
    check(memory != 0, "the capture holds no memory deltas");

    const auto dump = capture / "targets.bin";
    const auto command = "\"\"" + tool.string() + "\" \"" + capture.string() + "\" --strict --dump " + std::to_string(at(targets)) + ":" + std::to_string(Copies * DataBytes) + ":\"" + dump.string() + "\"\"";
    const auto code = std::system(command.c_str());
    check(code == 0, "agc_frame_replay exited with " + std::to_string(code));
    const auto replayed = readFile(dump);
    check(replayed.size() == Copies * DataBytes, "the replay dumped " + std::to_string(replayed.size()) + " bytes of the copy targets");
    for (std::size_t index = 0; index < Copies; ++index) {
        check(std::memcmp(replayed.data() + index * DataBytes, copied.data(), DataBytes) == 0, "the replay did not reproduce copy " + std::to_string(index) + " of the second thread");
    }
}


// A submission parked on a label that the title stores only after its next call: the delta before
// that call is read while the work is parked (the capture used to wait 10 s for it and stop), and
// the replay reproduces the copy that ran after the release over the bytes the CPU wrote meanwhile.
void testParkedSubmission(const std::filesystem::path& tool) {
    const char* directory = std::getenv("APS5_FRAME_CAPTURE");
    check(directory != nullptr && *directory != '\0', "APS5_FRAME_CAPTURE is not set");
    const std::filesystem::path capture(directory);
    const auto layout = allocate();
    auto* const gate = layout.label + 16;
    const auto early = pattern(9);
    const auto late = pattern(77);
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(Handle, output);

    flip();
    std::memcpy(layout.source, early.data(), early.size());
    {
        auto* const words = layout.firstCommands;
        CommandBuffer buffer{words, words + CommandBytes / sizeof(std::uint32_t), words, words + CommandBytes / sizeof(std::uint32_t), nullptr, nullptr, 0};
        sceAgcDcbWaitRegMem(&buffer, 0, 3, 0, 0, gate, Done, 0xffffffffu, 0);
        sceAgcDcbDmaData(&buffer, 0, DmaMemory, 0, at(layout.firstTarget), DmaMemory, 0, at(layout.source), DataBytes, 0, 1, 0);
        sceAgcDcbWriteData(&buffer, GraphicsMemory, 0, at(layout.label), &Done, 1, 0, 1);
        Packet packet{words, static_cast<std::uint32_t>(buffer.cursor_up - words), 0, {}};
        check(sceAgcDriverSubmitDcb(&packet) == 0, "submission failed");
    }
    std::memcpy(layout.source, late.data(), late.size());
    const auto began = std::chrono::steady_clock::now();
    flip();
    const auto flipSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    std::atomic_ref<std::uint32_t>(*gate).store(Done, std::memory_order_release);
    copy(layout.secondCommands, CommandBytes, layout.source, layout.secondTarget, layout.label);
    flip();
    AgcDriverWaitIdle_nid_postfix();
    check(equal(layout.firstTarget, late) && equal(layout.secondTarget, late), "the driver did not run the parked copy after the release");
    check(flipSeconds < 5.0, "the flip after the parked submission took " + std::to_string(flipSeconds) + " s");

    const FrameCaptureReader reader(capture);
    check(!reader.Truncated(), "the capture is truncated");
    check(reader.Header().find("\"frames\":2") != std::string::npos, "the capture did not record both frames: " + reader.Header());
    check(reader.Header().find("did not finish") == std::string::npos && reader.Header().find("neither finished") == std::string::npos, "the capture stopped at the parked submission: " + reader.Header());
    check(reader.Header().find("\"blocked_deltas\":0") == std::string::npos, "no delta was read while the submission was parked: " + reader.Header());
    bool parkedDelta = false;
    for (const auto& call : reader.Calls()) parkedDelta = parkedDelta || (call.kind == FrameCallKind::Memory && call.completed < call.accepted);
    check(parkedDelta, "no memory delta was recorded with the parked submission unfinished");

    const auto firstDump = capture / "parked-first.bin";
    const auto secondDump = capture / "parked-second.bin";
    const auto command = "\"\"" + tool.string() + "\" \"" + capture.string() + "\" --strict --dump " + std::to_string(at(layout.firstTarget)) + ":" + std::to_string(DataBytes) + ":\"" + firstDump.string() +
                         "\" --dump " + std::to_string(at(layout.secondTarget)) + ":" + std::to_string(DataBytes) + ":\"" + secondDump.string() + "\"\"";
    const auto code = std::system(command.c_str());
    check(code == 0, "agc_frame_replay exited with " + std::to_string(code));
    check(readFile(firstDump) == late, "the replay did not reproduce the parked copy");
    check(readFile(secondDump) == late, "the replay did not reproduce the copy after the release");
}

// The writer of a parked submission's label lies beyond the replayed window (--frame-limit 1 ends
// at the flip after the submission; the title stores the label only after that flip). Under
// --exact the replay must release the wait once every replay thread is parked and the driver is
// blocked on it, run the parked copy, and end on its own.
void testWriterBeyondWindow(const std::filesystem::path& tool) {
    const char* directory = std::getenv("APS5_FRAME_CAPTURE");
    check(directory != nullptr && *directory != '\0', "APS5_FRAME_CAPTURE is not set");
    const std::filesystem::path capture(directory);
    const auto layout = allocate();
    auto* const gate = layout.label + 16;
    const auto early = pattern(5);
    const auto late = pattern(55);
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(Handle, output);

    flip();
    std::memcpy(layout.source, early.data(), early.size());
    {
        auto* const words = layout.firstCommands;
        CommandBuffer buffer{words, words + CommandBytes / sizeof(std::uint32_t), words, words + CommandBytes / sizeof(std::uint32_t), nullptr, nullptr, 0};
        sceAgcDcbWaitRegMem(&buffer, 0, 3, 0, 0, gate, Done, 0xffffffffu, 0);
        sceAgcDcbDmaData(&buffer, 0, DmaMemory, 0, at(layout.firstTarget), DmaMemory, 0, at(layout.source), DataBytes, 0, 1, 0);
        sceAgcDcbWriteData(&buffer, GraphicsMemory, 0, at(layout.label), &Done, 1, 0, 1);
        Packet packet{words, static_cast<std::uint32_t>(buffer.cursor_up - words), 0, {}};
        check(sceAgcDriverSubmitDcb(&packet) == 0, "submission failed");
    }
    std::memcpy(layout.source, late.data(), late.size());
    flip();
    std::atomic_ref<std::uint32_t>(*gate).store(Done, std::memory_order_release);
    copy(layout.secondCommands, CommandBytes, layout.source, layout.secondTarget, layout.label);
    flip();
    AgcDriverWaitIdle_nid_postfix();
    check(equal(layout.firstTarget, late), "the driver did not run the parked copy after the release");

    const FrameCaptureReader reader(capture);
    check(!reader.Truncated(), "the capture is truncated");
    const auto firstDump = capture / "beyond-first.bin";
    const auto log = capture / "beyond-replay.err";
    const auto command = "\"\"" + tool.string() + "\" \"" + capture.string() + "\" --exact --frame-limit 1 --timeout-s 60 --dump " + std::to_string(at(layout.firstTarget)) + ":" + std::to_string(DataBytes) + ":\"" + firstDump.string() +
                         "\" 2> \"" + log.string() + "\"\"";
    const auto began = std::chrono::steady_clock::now();
    const auto code = std::system(command.c_str());
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    check(code == 0, "agc_frame_replay --exact --frame-limit 1 exited with " + std::to_string(code));
    check(seconds < 45.0, "agc_frame_replay --exact --frame-limit 1 took " + std::to_string(seconds) + " s");
    check(readFile(firstDump) == late, "the replay did not run the parked copy whose writer lies beyond the window");
    const auto text = readFile(log);
    const std::string stderrText(reinterpret_cast<const char*>(text.data()), text.size());
    check(stderrText.find("while parked") != std::string::npos && stderrText.find(" 0 while parked") == std::string::npos, "the wait was not released while the replay was parked");
}
}

int main(int argc, char** argv) {
    if (argc != 2 && !(argc == 3 && (std::string(argv[2]) == "concurrent" || std::string(argv[2]) == "parked" || std::string(argv[2]) == "beyond"))) {
        std::fprintf(stderr, "usage: agc_driver_frame_replay_tests <agc_frame_replay> [concurrent|parked|beyond]\n");
        return 2;
    }
    try {
        if (argc == 3 && std::string(argv[2]) == "parked") testParkedSubmission(argv[1]);
        else if (argc == 3 && std::string(argv[2]) == "beyond") testWriterBeyondWindow(argv[1]);
        else if (argc == 3) testConcurrentCopies(argv[1]);
        else testCaptureAndReplay(argv[1]);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
    std::puts("frame replay tests passed");
    return 0;
}
