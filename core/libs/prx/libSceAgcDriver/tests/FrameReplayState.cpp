#define main frameReplayToolMain
#include "../tools/FrameReplay.cpp"
#undef main
#include "prx/libSceAgcDriver/Execution/include/FrameRecorder.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include <barrier>
#include <sstream>

namespace {

using AgcDriver::FrameCaptureWriter;

constexpr std::uint32_t ReadWrite = 0x04;

void check(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

std::uint64_t address(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer);
}

std::byte* allocate(std::size_t size, bool gpu = true) {
    auto* bytes = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(size, 65536));
    check(bytes != nullptr, "guest allocation failed");
    GuestArena::GuestArenaCommit_nid_postfix(bytes, size, 0x04, size);
#ifdef __linux__
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(bytes, size);
#endif
    GuestAllocations::Mutation mutation;
    mutation.Add(bytes, size, true, true, gpu);
    return bytes;
}

void collectGroups(Replay& replay) {
    for (const auto& chunk : replay.reader.Chunks()) {
        if (chunk.kind == FrameMemoryKind::Delta) replay.groups[chunk.group].push_back(&chunk);
    }
}

FrameCall memoryCall(std::uint64_t group, std::uint64_t thread) {
    FrameCall call;
    call.kind = FrameCallKind::Memory;
    call.values = {group};
    call.thread = thread;
    return call;
}

void testMemoryOrder(const std::filesystem::path& directory) {
    constexpr std::size_t Size = 64u * 1024u * 1024u;
    auto* bytes = allocate(Size);
    const std::vector<std::byte> first(Size, std::byte{0x11});
    const std::vector<std::byte> second(4096, std::byte{0x22});
    FrameCaptureWriter writer(directory);
    writer.WriteDelta(1, 0, address(bytes), first);
    writer.WriteDelta(2, 0, address(bytes + Size - second.size()), second);
    writer.Finish("{}");
    const FrameCaptureReader reader(directory);
    Options options;
    options.release = false;
    Replay replay{options, reader};
    replay.recording = true;
    collectGroups(replay);
    const std::array calls{memoryCall(1, 1), memoryCall(2, 2)};
    replay.runWindow(calls);
    check(std::all_of(bytes, bytes + Size - second.size(), [](std::byte value) { return value == std::byte{0x11}; }), "first delta was not restored");
    check(std::equal(second.begin(), second.end(), bytes + Size - second.size()), "an earlier delta overwrote the later thread's delta");
}

void testReusedMemory(const std::filesystem::path& directory) {
#if defined(_WIN32) || defined(__linux__)
    constexpr std::size_t Size = 2 * FrameCaptureWriter::BlockBytes;
    auto* bytes = allocate(Size);
    auto* cpu = allocate(FrameCaptureWriter::BlockBytes, false);
    std::fill_n(bytes, Size, std::byte{0x7f});
    std::fill_n(cpu, FrameCaptureWriter::BlockBytes, std::byte{0x55});
    std::vector<std::byte> expected(Size);
    std::fill(expected.begin() + FrameCaptureWriter::BlockBytes, expected.end(), std::byte{0x22});
    std::uintptr_t base = 0;
    std::size_t arenaSize = 0;
    GuestArena::GuestArenaRange_nid_postfix(&base, &arenaSize);
    FrameCaptureWriter writer(directory);
    writer.WriteRange(address(bytes), Size, FrameRangeKind::Watched, ReadWrite, FrameRangeAccess::Readable | FrameRangeAccess::Writable | FrameRangeAccess::Gpu);
    writer.WriteRange(address(cpu), FrameCaptureWriter::BlockBytes, FrameRangeKind::Unwatched, ReadWrite, FrameRangeAccess::Readable | FrameRangeAccess::Writable);
    check(writer.WriteSnapshot(address(bytes), expected) == FrameCaptureWriter::BlockBytes, "zero snapshot block was not elided");
    writer.Finish("{\"arena_base\":" + std::to_string(base) + "}");
    const FrameCaptureReader reader(directory);
    const auto skipped = restoreMemory(reader);
    check(std::equal(expected.begin(), expected.end(), bytes), "reused GPU memory did not restore zero and nonzero snapshot blocks");
    check(skipped.bytes == FrameCaptureWriter::BlockBytes, "unexpected skipped memory");
    check(std::all_of(cpu, cpu + FrameCaptureWriter::BlockBytes, [](std::byte value) { return value == std::byte{0x55}; }), "reused CPU memory was overwritten");
#endif
}

void testSharedPage(const std::filesystem::path& directory) {
#ifdef __linux__
    constexpr std::size_t Page = 65536;
    auto* page = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(Page, Page));
    check(page != nullptr, "guest reservation failed");
    GuestArena::GuestArenaRelease_nid_postfix(page, Page);
    const std::vector<std::byte> gpu(0x20, std::byte{0x3c});
    const std::vector<std::byte> cpu(0x40, std::byte{0x5a});
    std::uintptr_t base = 0;
    std::size_t arenaSize = 0;
    GuestArena::GuestArenaRange_nid_postfix(&base, &arenaSize);
    FrameCaptureWriter writer(directory);
    writer.WriteRange(address(page + 0x20), gpu.size(), FrameRangeKind::Watched, ReadWrite, FrameRangeAccess::Readable | FrameRangeAccess::Writable | FrameRangeAccess::Gpu);
    writer.WriteRange(address(page + 0x40), cpu.size(), FrameRangeKind::Unwatched, ReadWrite, FrameRangeAccess::Readable | FrameRangeAccess::Writable);
    writer.WriteSnapshot(address(page + 0x20), gpu);
    writer.WriteSnapshot(address(page + 0x40), cpu);
    writer.Finish("{\"arena_base\":" + std::to_string(base) + "}");
    const FrameCaptureReader reader(directory);
    const auto skipped = restoreMemory(reader);
    check(skipped.bytes == 0, "a range sharing a page with an earlier one was skipped");
    check(std::equal(gpu.begin(), gpu.end(), page + 0x20) && std::equal(cpu.begin(), cpu.end(), page + 0x40), "ranges sharing a page were not restored");
#else
    static_cast<void>(directory);
#endif
}

void testHostNeighbors(const std::filesystem::path& directory) {
#ifdef __linux__
    const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    void* mapped = mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(mapped != MAP_FAILED, "host allocation failed");
    auto* occupied = static_cast<std::byte*>(mapped);
    auto* available = occupied + page;
    check(munmap(available, page) == 0, "host page release failed");
    std::uintptr_t base = 0;
    std::size_t arenaSize = 0;
    GuestArena::GuestArenaRange_nid_postfix(&base, &arenaSize);
    FrameCaptureWriter writer(directory);
    writer.WriteRange(address(occupied), page, FrameRangeKind::Host, ReadWrite, FrameRangeAccess::Readable | FrameRangeAccess::Writable);
    writer.WriteRange(address(available), page, FrameRangeKind::Host, ReadWrite, FrameRangeAccess::Readable | FrameRangeAccess::Writable | FrameRangeAccess::Gpu);
    writer.Finish("{\"arena_base\":" + std::to_string(base) + "}");
    const FrameCaptureReader reader(directory);
    const auto skipped = restoreMemory(reader);
    check(skipped.bytes == page, "an occupied CPU host range was not the only range skipped");
#else
    static_cast<void>(directory);
#endif
}

void testUnwatched() {
    auto* recorder = AgcDriver::FrameRecorder::Active();
    check(recorder != nullptr, "frame recorder is disabled");
    try {
        recorder->FrameBoundary();
    } catch (const std::runtime_error& error) {
        check(std::string_view(error.what()).find("write watch is unavailable") != std::string_view::npos, "a capture without the write watch failed for another reason");
        return;
    }
    check(false, "a capture started without the write watch");
}

void testEopTokens() {
    Frames frames;
    auto output = std::make_shared<ReplayOutput>(frames);
    AgcDriverRegisterVideoOutput_nid_postfix(7, output);
    std::barrier together(2);
    std::atomic<int> wrong{0};
    const auto worker = [&](int lane) {
        for (int round = 0; round < 2000; ++round) {
            const auto requested = AgcDriver::EopFlipTokenBase + round * 8 + lane * 2;
            together.arrive_and_wait();
            try {
                if (AgcDriverReserveEopFlipAt_nid_postfix({7, -1, 1, 0}, requested) != requested) ++wrong;
            } catch (const std::exception&) { ++wrong; }
        }
    };
    std::thread first(worker, 0), second(worker, 1);
    first.join();
    second.join();
    check(wrong.load() == 0, "concurrent explicit EOP reservations returned the wrong token");
    const auto high = AgcDriver::EopFlipTokenBase + 20000;
    check(AgcDriverReserveEopFlipAt_nid_postfix({7, -1, 1, 0}, high) == high, "explicit high token was not reserved");
    check(AgcDriverReserveEopFlipAt_nid_postfix({7, -1, 1, 0}, high - 1) == high - 1, "explicit lower token was not reserved");
    check(AgcDriverReserveEopFlip_nid_postfix({7, -1, 1, 0}) == high + 1, "ordinary reservation reused an explicit token");
    bool rejected = false;
    try { AgcDriverReserveEopFlipAt_nid_postfix({7, -1, 1, 0}, high); }
    catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "duplicate explicit token was not rejected");
    AgcDriverUnregisterVideoOutput_nid_postfix(7, output);
}

class WaitingOutput final : public AgcDriver::IVideoOutput {
public:
    WaitingOutput(Frames& frames, const std::uint32_t* label) : frames(frames), label(label) {}
    std::shared_ptr<AgcDriver::IFlipRequest> Reserve(const AgcDriver::FlipInfo&) override {
        const auto deadline = Clock::now() + std::chrono::seconds(2);
        while (*static_cast<const volatile std::uint32_t*>(label) == 0) {
            check(Clock::now() < deadline, "a blocking driver call prevented the later memory update");
            std::this_thread::yield();
        }
        return std::make_shared<ReplayFlip>(frames);
    }
    void Fail(std::exception_ptr error) noexcept override { frames.Failed(error); }

private:
    Frames& frames;
    const std::uint32_t* label;
};

void testBlockingSubmission(const std::filesystem::path& directory) {
    auto* bytes = allocate(FrameCaptureWriter::BlockBytes);
    auto* label = reinterpret_cast<std::uint32_t*>(bytes);
    const std::uint32_t ready = 1;
    FrameCaptureWriter writer(directory);
    writer.WriteDelta(1, 0, address(label), std::as_bytes(std::span(&ready, 1)));
    writer.Finish("{}");
    const FrameCaptureReader reader(directory);
    Options options;
    options.pace = false;
    options.release = false;
    Replay replay{options, reader};
    collectGroups(replay);
    auto output = std::make_shared<WaitingOutput>(replay.frames, label);
    AgcDriverRegisterVideoOutput_nid_postfix(7, output);
    std::array<std::uint32_t, AgcDriver::FlipPacketWords> words{AgcDriver::FlipPacketHeader, 7, 0xffffffffu, 1, 0, 0};
    for (const auto kind : {FrameCallKind::Submit, FrameCallKind::Flip, FrameCallKind::ReserveEopFlip}) {
        *label = 0;
        FrameCall submit;
        submit.kind = kind;
        submit.thread = 1;
        submit.address = address(words.data());
        submit.size = words.size();
        submit.values = {7, static_cast<std::uint64_t>(-1), 1, 0, AgcDriver::EopFlipTokenBase};
        const std::array calls{submit, memoryCall(1, 2)};
        replay.runWindow(calls);
        check(*label == ready && replay.unfinished == 0, "blocking submission did not complete");
    }
    check(replay.frames.Times().size() == 2, "blocking submissions lost a flip");
    AgcDriverUnregisterVideoOutput_nid_postfix(7, output);
}

void testShaderCode(const std::filesystem::path& directory, bool host) {
    auto* bytes = allocate(1u << 20);
    auto* shader = reinterpret_cast<Shader*>(bytes);
    auto* code = bytes + (1u << 19) + 256;
    constexpr std::size_t CodeBytes = 3 * FrameCaptureWriter::BlockBytes;
#if defined(_WIN32) || defined(__linux__)
    if (host) {
#ifdef _WIN32
        auto* memory = static_cast<std::byte*>(VirtualAlloc(nullptr, CodeBytes + 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        void* mapped = mmap(nullptr, CodeBytes + 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        auto* memory = mapped != MAP_FAILED ? static_cast<std::byte*>(mapped) : nullptr;
#endif
        check(memory != nullptr, "host shader allocation failed");
        GuestAllocations::Mutation mutation;
        mutation.Add(memory, CodeBytes + 4096, true, true, true);
        code = memory + 256;
    }
#endif
    *shader = {};
    shader->file_header = 0x34333231u;
    shader->version = 0x18u;
    shader->header_size = sizeof(Shader);
    shader->shader_size = CodeBytes;
    shader->code = code;
    auto* recorder = AgcDriver::FrameRecorder::Active();
    check(recorder != nullptr, "frame recorder is disabled");
    recorder->FrameBoundary();
    std::vector<std::byte> expected(CodeBytes);
    for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<std::byte>(i * 31 + 17);
    std::memcpy(code, expected.data(), expected.size());
    recorder->RegisterShader(shader);
    recorder->FrameBoundary();
    const FrameCaptureReader reader(directory);
    std::vector<std::byte> captured(CodeBytes);
    const auto apply = [&](const FrameMemoryChunk& chunk) {
        const auto begin = std::max(address(code), chunk.address);
        const auto end = std::min(address(code) + CodeBytes, chunk.address + chunk.bytes);
        if (begin >= end) return;
        const auto data = reader.Read(chunk);
        std::memcpy(captured.data() + begin - address(code), data.data() + begin - chunk.address, end - begin);
    };
    for (const auto& chunk : reader.Chunks()) if (chunk.kind == FrameMemoryKind::Snapshot) apply(chunk);
    bool registered = false;
    for (const auto& call : reader.Calls()) {
        if (call.kind == FrameCallKind::RegisterShader) {
            registered = true;
            break;
        }
        if (call.kind != FrameCallKind::Memory) continue;
        for (const auto& chunk : reader.Chunks()) {
            if (chunk.kind == FrameMemoryKind::Delta && chunk.group == call.values.at(0)) apply(chunk);
        }
    }
    check(registered && captured == expected, "the complete separate shader code was not captured before registration");
}

void testPrologueHeader(const std::filesystem::path& directory) {
    auto* bytes = allocate(1u << 20);
    auto* shader = reinterpret_cast<Shader*>(bytes);
    auto* code = bytes + (1u << 19);
    constexpr std::size_t CodeBytes = 4096;
    for (std::size_t i = 0; i < CodeBytes; ++i) code[i] = static_cast<std::byte>(i * 7 + 3);
    *shader = {};
    shader->file_header = 0x34333231u;
    shader->version = 0x18u;
    shader->header_size = sizeof(Shader);
    shader->shader_size = CodeBytes;
    shader->code = code;
    std::vector<std::byte> registered(sizeof(Shader));
    std::memcpy(registered.data(), shader, sizeof(Shader));
    auto* recorder = AgcDriver::FrameRecorder::Active();
    check(recorder != nullptr, "frame recorder is disabled");
    recorder->RegisterShader(shader);
    std::memset(static_cast<void*>(shader), 0x5a, sizeof(Shader));
    std::vector<std::byte> later(sizeof(Shader));
    std::memcpy(later.data(), shader, sizeof(Shader));
    recorder->FrameBoundary();
    recorder->FrameBoundary();
    const FrameCaptureReader reader(directory);
    std::vector<const FrameMemoryChunk*> rewritten;
    bool restoredFirst = false;
    for (const auto& call : reader.Calls()) {
        if (call.kind == FrameCallKind::RegisterShader) {
            check(call.address == address(shader), "the prologue registration names another header");
            break;
        }
        if (call.kind != FrameCallKind::Memory) continue;
        for (const auto& chunk : reader.Chunks()) {
            if (chunk.kind != FrameMemoryKind::Delta || chunk.group != call.values.at(0)) continue;
            rewritten.push_back(&chunk);
            restoredFirst = restoredFirst || (chunk.address == address(shader) && reader.Read(chunk) == registered);
        }
    }
    check(restoredFirst, "the prologue registration was not preceded by the header it registered");
    std::memcpy(shader, registered.data(), registered.size());
    restoreRewritten(reader, rewritten, Skipped{});
    check(std::memcmp(shader, later.data(), later.size()) == 0, "the snapshot was not restored over the prologue header image");
    check(reader.Header().find("\"prologue_rewrites\":1,") != std::string::npos, "the capture header does not count the prologue rewrite");
    const auto states = std::find_if(reader.Calls().begin(), reader.Calls().end(), [](const FrameCall& call) { return call.kind == FrameCallKind::QueueStates; });
    const auto start = std::find_if(reader.Calls().begin(), reader.Calls().end(), [](const FrameCall& call) { return call.kind == FrameCallKind::FrameStart; });
    check(states != reader.Calls().end() && states < start, "the capture does not record the queue states before the first frame");
    std::vector<std::uint32_t> words;
    for (const auto value : states->values) words.push_back(static_cast<std::uint32_t>(value));
    AgcDriverLoadQueueStates_nid_postfix(words);
    check(AgcDriverSaveQueueStates_nid_postfix() == words, "loaded queue states do not save back unchanged");
}

}

void testCheckpoint(const std::filesystem::path& directory) {
    auto* bytes = allocate(4 * FrameCaptureWriter::BlockBytes);
    std::fill_n(bytes, 4 * FrameCaptureWriter::BlockBytes, std::byte{0x11});
    auto* recorder = AgcDriver::FrameRecorder::Active();
    check(recorder != nullptr, "frame recorder is disabled");
    recorder->FrameBoundary();
    bytes[FrameCaptureWriter::BlockBytes + 5] = std::byte{0x22};
    recorder->FrameBoundary();
    const FrameCaptureReader reader(directory);
    std::size_t blocks = 0;
    for (const auto& chunk : reader.Chunks()) {
        if (chunk.kind != FrameMemoryKind::Checkpoint) continue;
        ++blocks;
        check(chunk.address == address(bytes + FrameCaptureWriter::BlockBytes) && reader.Read(chunk)[5] == std::byte{0x22}, "the checkpoint holds another block");
    }
    check(blocks == 1, "the checkpoint does not hold exactly the changed block");
    check(std::any_of(reader.Calls().begin(), reader.Calls().end(), [](const FrameCall& call) { return call.kind == FrameCallKind::Checkpoint; }), "the capture has no checkpoint call");
    const auto report = directory / "checkpoint.csv";
    const auto lines = [&] {
        std::ifstream file(report);
        std::size_t count = 0;
        for (std::string line; std::getline(file, line);) ++count;
        return count;
    };
    compareCheckpoint(reader, report, Skipped{});
    check(lines() == 1, "an unchanged checkpoint block was reported as differing");
    bytes[FrameCaptureWriter::BlockBytes + 6] = std::byte{0x33};
    compareCheckpoint(reader, report, Skipped{});
    check(lines() == 2, "a differing checkpoint block was not reported");
}

#ifdef _WIN32
void testAliasedPages(const std::filesystem::path& directory) {
    constexpr std::size_t PageBytes = 0x4000;
    auto* section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, PageBytes, nullptr);
    check(section != nullptr, "test section creation failed");
    auto* first = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(PageBytes, 65536));
    auto* second = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(PageBytes, 65536));
    check(first != nullptr && second != nullptr, "guest allocation failed");
    GuestArena::GuestArenaMapDirect_nid_postfix(first, PageBytes, section, 0, PAGE_READWRITE);
    GuestArena::GuestArenaMapDirect_nid_postfix(second, PageBytes, section, 0, PAGE_READWRITE);
    CloseHandle(section);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(first, PageBytes, true, true, true);
        mutation.Add(second, PageBytes, true, true, true);
    }
    first[16] = std::byte{0x5a};
    check(second[16] == std::byte{0x5a}, "the test pages do not alias");
    auto* recorder = AgcDriver::FrameRecorder::Active();
    check(recorder != nullptr, "frame recorder is disabled");
    recorder->FrameBoundary();
    recorder->FrameBoundary();
    const FrameCaptureReader reader(directory);
    const auto aliases = aliasedPages(reader);
    const auto one = aliases.groupOf.find(address(first));
    const auto other = aliases.groupOf.find(address(second));
    check(aliases.pageBytes == PageBytes && one != aliases.groupOf.end() && other != aliases.groupOf.end() && one->second == other->second, "the capture does not list the two views as one backing");
    auto* third = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(PageBytes, 65536));
    auto* fourth = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(PageBytes, 65536));
    AliasedPages replayed;
    replayed.pageBytes = PageBytes;
    replayed.groupOf = {{address(third), 0}, {address(fourth), 0}};
    std::map<std::uint64_t, void*> sections;
    commitWithAliases(third, PageBytes, replayed, sections);
    commitWithAliases(fourth, PageBytes, replayed, sections);
    for (const auto& [group, handle] : sections) CloseHandle(handle);
    third[32] = std::byte{0x77};
    check(fourth[32] == std::byte{0x77}, "replayed aliased pages do not share one backing");
}
#endif

// The ranges a replay cannot map because they lie on a host region of its process (it then starts again).
void testHostCollisions(const std::filesystem::path& directory) {
    FrameCaptureWriter writer(directory);
    writer.WriteRange(0x10000000, 0x20000, FrameRangeKind::Watched, 3, FrameRangeAccess::Readable | FrameRangeAccess::Gpu);
    writer.WriteRange(0x20000000, 0x10000, FrameRangeKind::Watched, 3, FrameRangeAccess::Readable);
    writer.WriteRange(0x30000000, 0x10000, FrameRangeKind::Host, 3, FrameRangeAccess::Readable | FrameRangeAccess::Gpu);
    writer.Finish("{}");
    const FrameCaptureReader reader(directory);
    // A host region at 0x1001f000-0x10021000 and one inside the host-kind range (those are mapped apart, never checked).
    const auto overlaps = [](std::uint64_t address, std::uint64_t bytes) {
        const auto hits = [&](std::uint64_t begin, std::uint64_t end) { return address < end && begin < address + bytes; };
        return hits(0x1001f000, 0x10021000) || hits(0x30000000, 0x30001000);
    };
    const auto found = HostCollisions(reader, overlaps);
    check(found.size() == 1 && found[0].first == 0x10000000 && found[0].second == 0x10020000, "only the guest range on the host region collides");
    check(HostCollisions(reader, [](std::uint64_t, std::uint64_t) { return false; }).empty(), "no host region, no collision");
}

// --dedupe-prologue: only resolutions identical to one since the last registration or memory call.
void testPrologueDedupe() {
    const auto resolve = [](std::uint64_t address, std::uint32_t context) {
        FrameCall call;
        call.kind = FrameCallKind::ResolveShaderAbi;
        call.address = address;
        call.context = {ShaderRegister{0x200, context}};
        return call;
    };
    FrameCall registration;
    registration.kind = FrameCallKind::RegisterShader;
    FrameCall memory = memoryCall(1, 0);
    PrologueDedupe dedupe;
    check(!dedupe.Redundant(resolve(0x1000, 1)), "the first resolution is executed");
    check(dedupe.Redundant(resolve(0x1000, 1)), "an identical repeat is skipped");
    check(!dedupe.Redundant(resolve(0x1000, 2)), "other registers are a different resolution");
    check(!dedupe.Redundant(registration), "a registration is never skipped");
    check(!dedupe.Redundant(resolve(0x1000, 1)), "a repeat after a registration is executed");
    check(dedupe.Redundant(resolve(0x1000, 1)), "and its repeat skipped again");
    check(!dedupe.Redundant(memory), "a memory call is never skipped");
    check(!dedupe.Redundant(resolve(0x1000, 1)), "a repeat after a memory call is executed");
    check(dedupe.Skipped() == 2, "two skipped");
}

// writeDump's chunked write: every byte, in order, with several calls.
void testDumpChunks() {
    std::vector<std::byte> bytes(10000);
    for (std::size_t index = 0; index < bytes.size(); ++index) bytes[index] = static_cast<std::byte>(index * 7u);
    std::ostringstream file(std::ios::binary);
    check(writeChunked(file, bytes.data(), bytes.size(), 4096), "chunked write failed");
    const auto written = file.str();
    check(written.size() == bytes.size() && std::memcmp(written.data(), bytes.data(), bytes.size()) == 0, "chunked write changed the bytes");
    std::ostringstream empty(std::ios::binary);
    check(writeChunked(empty, bytes.data(), 0, 4096) && empty.str().empty(), "an empty dump wrote bytes");
}

int main(int argc, char** argv) {
    try {
        check(argc == 3, "expected a test name and capture directory");
        const std::string mode = argv[1];
        const std::filesystem::path directory = argv[2];
        if (mode == "order") testMemoryOrder(directory);
        else if (mode == "zero") testReusedMemory(directory);
        else if (mode == "shared_page") testSharedPage(directory);
        else if (mode == "host_neighbors") testHostNeighbors(directory);
        else if (mode == "unwatched") testUnwatched();
        else if (mode == "tokens") testEopTokens();
        else if (mode == "dump_chunks") testDumpChunks();
        else if (mode == "dedupe") testPrologueDedupe();
        else if (mode == "host_collisions") testHostCollisions(directory);
        else if (mode == "blocking") testBlockingSubmission(directory);
        else if (mode == "shader" || mode == "shader_host") testShaderCode(directory, mode == "shader_host");
        else if (mode == "shader_prologue") testPrologueHeader(directory);
        else if (mode == "shader_checkpoint") testCheckpoint(directory);
#ifdef _WIN32
        else if (mode == "shader_aliases") testAliasedPages(directory);
#endif
        else throw std::runtime_error("unknown test");
        AgcDriverShutdown_nid_postfix();
        std::puts("frame replay state test passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        try { AgcDriverShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
