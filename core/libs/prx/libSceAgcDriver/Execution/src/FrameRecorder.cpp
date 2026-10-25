#include "prx/libSceAgcDriver/State/include/Configuration.hpp"
#include "prx/libSceAgcDriver/Execution/include/FrameRecorder.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/TraceRevision.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/General.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
extern char** environ;
#endif
#ifdef __linux__
#include "prx/libc/include/specifics/linux/HostMapping.hpp"
#include <sys/uio.h>
#endif

namespace AgcDriver {
namespace {

std::uint64_t readNumber(const char* text, std::uint64_t fallback) {
    if (text == nullptr || *text == '\0') return fallback;
    std::uint64_t value = 0;
    const std::string_view view(text);
    const auto [last, error] = std::from_chars(view.data(), view.data() + view.size(), value);
    if (error != std::errc{} || last != view.data() + view.size()) throw std::runtime_error("frame capture: '" + std::string(view) + "' is not a number");
    return value;
}

std::string jsonString(std::string_view text) {
    std::string result = "\"";
    for (const char character : text) {
        if (character == '"' || character == '\\') {
            result += '\\';
            result += character;
        } else if (static_cast<unsigned char>(character) < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(character)));
            result += escaped;
        } else {
            result += character;
        }
    }
    return result + "\"";
}

std::vector<std::pair<std::string, std::string>> captureEnvironment() {
#ifdef _WIN32
    char** entries = _environ;
#else
    char** entries = environ;
#endif
    std::vector<std::pair<std::string, std::string>> result;
    for (; entries != nullptr && *entries != nullptr; ++entries) {
        const std::string_view entry = *entries;
        const auto equals = entry.find('=');
        if (!entry.starts_with("APS5_") || equals == std::string_view::npos) continue;
        result.emplace_back(std::string(entry.substr(0, equals)), std::string(entry.substr(equals + 1)));
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::uint64_t threadId() {
    return static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

std::vector<ShaderRegister> copyRegisters(std::span<const ShaderRegister> registers) {
    return {registers.begin(), registers.end()};
}

std::uint64_t pointer(const void* value) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(value));
}

std::uint64_t flipValue(std::int64_t value) {
    return static_cast<std::uint64_t>(value);
}

#ifdef __linux__
std::uint32_t windowsProtection(const HostMapping::Mapping& mapping) {
    if (mapping.executable) return mapping.writable ? 0x40u : mapping.readable ? 0x20u : 0x10u;
    if (mapping.writable) return 0x04u;
    return mapping.readable ? 0x02u : 0x01u;
}

bool pageReadable(std::uint64_t address) {
    char byte = 0;
    iovec local{&byte, 1};
    iovec remote{reinterpret_cast<void*>(static_cast<std::uintptr_t>(address)), 1};
    const auto read = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    if (read == 1) return true;
    if (read < 0 && errno == EFAULT) return false;
    throw std::runtime_error(std::string("frame capture: cannot probe a file mapping with process_vm_readv: ") + (read < 0 ? std::strerror(errno) : "nothing was read"));
}

std::uint64_t fileBackedEnd(const HostMapping::Mapping& mapping) {
    if (mapping.inode == 0 || !mapping.readable) return mapping.end;
    const auto page = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    if (pageReadable(mapping.end - page)) return mapping.end;
    if (!pageReadable(mapping.start)) return mapping.start;
    std::uint64_t readable = 0;
    std::uint64_t unreadable = (mapping.end - mapping.start) / page - 1;
    while (unreadable - readable > 1) {
        const auto middle = readable + (unreadable - readable) / 2;
        if (pageReadable(mapping.start + middle * page)) readable = middle;
        else unreadable = middle;
    }
    return mapping.start + unreadable * page;
}
#endif

std::uint64_t contentHash(std::span<const std::byte> bytes) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const auto value : bytes) hash = (hash ^ static_cast<std::uint64_t>(value)) * 0x100000001b3ull;
    return hash == 0 ? 1 : hash;
}

std::uint64_t wideHash(std::span<const std::byte> bytes) {
    std::uint64_t hash = 0xcbf29ce484222325ull ^ (static_cast<std::uint64_t>(bytes.size()) * 0x9e3779b97f4a7c15ull);
    std::size_t at = 0;
    for (; at + 8 <= bytes.size(); at += 8) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes.data() + at, 8);
        hash = (hash ^ word) * 0x100000001b3ull;
        hash ^= hash >> 29;
    }
    for (; at < bytes.size(); ++at) hash = (hash ^ static_cast<std::uint64_t>(bytes[at])) * 0x100000001b3ull;
    return hash == 0 ? 1 : hash;
}

constexpr std::size_t MaximumHeaderImageBytes = 1u << 20;

class ReplayedMemory {
public:
    bool Holds(std::uint64_t address, std::uint64_t bytes, std::uint64_t key) const {
        const auto found = known.find(address);
        return found != known.end() && found->second.first == address + bytes && found->second.second == key;
    }

    bool Rewritten(std::uint64_t address, std::uint64_t bytes) const {
        const auto next = written.upper_bound(address);
        if (next != written.begin() && std::prev(next)->second > address) return true;
        return next != written.end() && next->first < address + bytes;
    }

    void Write(std::uint64_t address, std::uint64_t bytes, std::uint64_t key) {
        const auto end = address + bytes;
        for (auto it = known.begin(); it != known.end();) {
            if (it->first < end && address < it->second.first) it = known.erase(it);
            else ++it;
        }
        known[address] = {end, key};
        auto first = address;
        auto last = end;
        auto it = written.upper_bound(address);
        if (it != written.begin() && std::prev(it)->second >= address) --it;
        while (it != written.end() && it->first <= last) {
            first = std::min(first, it->first);
            last = std::max(last, it->second);
            it = written.erase(it);
        }
        written[first] = last;
    }

    void Note(std::uint64_t address, std::uint64_t bytes, std::uint64_t key) {
        if (!Rewritten(address, bytes)) known[address] = {address + bytes, key};
    }

private:
    std::map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>> known;
    std::map<std::uint64_t, std::uint64_t> written;
};

}

FrameRecorder* FrameRecorder::Active() {
    static FrameRecorder* const recorder = []() -> FrameRecorder* {
        const char* directory = std::getenv("APS5_FRAME_CAPTURE");
        if (directory == nullptr || *directory == '\0') return nullptr;
        const char* trigger = std::getenv("APS5_FRAME_CAPTURE_TRIGGER");
        const auto atBoundary = readNumber(std::getenv("APS5_FRAME_CAPTURE_AT_FLIP"), 0);
        const auto frames = readNumber(std::getenv("APS5_FRAME_CAPTURE_FRAMES"), DefaultFrames);
        if (frames == 0) throw std::runtime_error("frame capture: APS5_FRAME_CAPTURE_FRAMES must be at least 1");
        return new FrameRecorder(directory, trigger != nullptr ? trigger : "", atBoundary, frames);
    }();
    return recorder;
}

FrameRecorder::FrameRecorder(std::filesystem::path directory, std::filesystem::path trigger, std::uint64_t atBoundary, std::uint64_t frames)
    : directory(std::move(directory)), trigger(std::move(trigger)), atBoundary(atBoundary), frames(frames), origin(std::chrono::steady_clock::now()) {}

FrameCall FrameRecorder::make(FrameCallKind kind) {
    FrameCall call;
    call.kind = kind;
    call.timeNs = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - origin).count());
    call.thread = threadId();
    return call;
}

void FrameRecorder::RegisterShader(const Shader* shader) {
    auto call = make(FrameCallKind::RegisterShader);
    call.address = pointer(shader);
    const std::array<const Shader*, 1> shaders{shader};
    record(std::move(call), true, shaders);
}

void FrameRecorder::ResolveShaderAbi(const Shader* shader, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive) {
    auto call = make(FrameCallKind::ResolveShaderAbi);
    call.address = pointer(shader);
    call.context = copyRegisters(context);
    call.primitive = copyRegisters(primitive);
    const std::array<const Shader*, 1> shaders{shader};
    record(std::move(call), true, shaders);
}

void FrameRecorder::ResolveGraphicsStagesAbi(std::span<const Shader* const> stages, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive) {
    auto call = make(FrameCallKind::ResolveGraphicsStagesAbi);
    for (const auto* stage : stages) call.values.push_back(pointer(stage));
    call.context = copyRegisters(context);
    call.primitive = copyRegisters(primitive);
    record(std::move(call), true, stages);
}

void FrameRecorder::ResolveGraphicsAbi(const Shader* vertex, const Shader* pixel, std::uint32_t primitiveType) {
    auto call = make(FrameCallKind::ResolveGraphicsAbi);
    call.values = {pointer(vertex), pointer(pixel), primitiveType};
    const std::array<const Shader*, 2> shaders{vertex, pixel};
    record(std::move(call), true, shaders);
}

void FrameRecorder::RegisterVideoOutput(std::uint32_t handle) {
    auto call = make(FrameCallKind::RegisterVideoOutput);
    call.values = {handle};
    record(std::move(call), true);
}

void FrameRecorder::UnregisterVideoOutput(std::uint32_t handle) {
    auto call = make(FrameCallKind::UnregisterVideoOutput);
    call.values = {handle};
    record(std::move(call), true);
}

void FrameRecorder::Submit(const Packet& packet, std::uint32_t queue, const std::function<void()>& accept) {
    auto call = make(FrameCallKind::Submit);
    call.queue = queue;
    call.address = pointer(packet.addr);
    call.size = packet.dw_num;
    call.flags = packet.flags;
    record(std::move(call), false, {}, accept);
}

void FrameRecorder::Flip(const FlipInfo& info, const std::function<void()>& accept) {
    auto call = make(FrameCallKind::Flip);
    call.cpuFlip = true;
    call.values = {info.handle, flipValue(info.index), info.mode, flipValue(info.argument)};
    record(std::move(call), false, {}, accept);
}

void FrameRecorder::ReserveEopFlip(const FlipInfo& info, std::uint32_t token) {
    auto call = make(FrameCallKind::ReserveEopFlip);
    call.values = {info.handle, flipValue(info.index), info.mode, flipValue(info.argument), token};
    std::lock_guard lock(mutex);
    if (state == State::Waiting) {
        call.ordinal = ++ordinal;
        reservations.push_back(std::move(call));
        if (reservations.size() > ReservationHistory) reservations.pop_front();
        return;
    }
    if (state != State::Recording) return;
    emitDelta();
    if (state != State::Recording) return;
    call.ordinal = ++ordinal;
    std::tie(call.accepted, call.completed) = DriverDetail::Driver::Get().SerialProgress();
    writer->WriteCall(call);
    ++recordedCalls;
}

void FrameRecorder::Suspend(const std::function<void()>& accept) {
    record(make(FrameCallKind::Suspend), false, {}, accept);
}

void FrameRecorder::PresentBuffer(const DisplayBuffer& buffer) {
    std::lock_guard lock(displayMutex);
    displayBuffers[buffer.address] = buffer;
}

void FrameRecorder::writeReferenceFrames() {
    std::map<std::uint64_t, DisplayBuffer> presented;
    {
        std::lock_guard lock(displayMutex);
        presented = displayBuffers;
    }
    if (presented.empty()) return;
    if (!DriverDetail::Driver::Get().WaitIdleFor(std::chrono::seconds(10))) {
        warnings.push_back("no reference frames: the driver did not go idle after the last frame");
        return;
    }
    std::uint64_t index = 0;
    for (const auto& [address, buffer] : presented) {
        try {
            const auto pixels = ReadDisplayBuffer(buffer);
            writer->WriteReference(index, address, pixels);
        } catch (const std::exception& failure) {
            warnings.push_back(std::string("no reference frame for a display buffer: ") + failure.what());
            continue;
        }
        auto call = make(FrameCallKind::DisplayBuffer);
        call.ordinal = ++ordinal;
        call.address = buffer.address;
        call.values = {index, buffer.pixelFormat, buffer.width, buffer.height, buffer.tilingMode, buffer.pitchInPixel, buffer.dccAddress, buffer.dccClearColor, buffer.alignment};
        writer->WriteCall(call);
        ++index;
    }
}

void FrameRecorder::record(FrameCall call, bool prologueCall, std::span<const Shader* const> shaders, const std::function<void()>& accept) {
    std::lock_guard lock(mutex);
    recordLocked(std::move(call), prologueCall, shaders);
    if (accept) accept();
}

void FrameRecorder::recordLocked(FrameCall call, bool prologueCall, std::span<const Shader* const> shaders) {
    if (state == State::Waiting) {
        if (!prologueCall) return;
        call.ordinal = ++ordinal;
        const bool registration = call.kind == FrameCallKind::RegisterShader;
        auto images = prologueShaders(shaders, registration);
        prologue.push_back({std::move(call), std::move(images)});
        return;
    }
    if (state != State::Recording) return;
    if (prologueCall) emitShaderDelta(shaders);
    else emitDelta();
    if (state != State::Recording) return;
    call.ordinal = ++ordinal;
    std::tie(call.accepted, call.completed) = DriverDetail::Driver::Get().SerialProgress();
    writer->WriteCall(call);
    ++recordedCalls;
}

std::vector<FrameRecorder::PrologueShader> FrameRecorder::prologueShaders(std::span<const Shader* const> shaders, bool registration) {
    std::vector<PrologueShader> result;
    for (const auto* shader : shaders) {
        if (shader == nullptr || !GuestMemory::Accessible(shader, sizeof(Shader))) continue;
        Shader fields;
        std::memcpy(&fields, static_cast<const void*>(shader), sizeof(Shader));
        if (fields.file_header != 0x34333231u || fields.header_size < sizeof(Shader) || fields.header_size > MaximumHeaderImageBytes || !GuestMemory::Accessible(shader, fields.header_size)) continue;
        PrologueShader entry;
        entry.header = pointer(shader);
        const std::span header(reinterpret_cast<const std::byte*>(shader), fields.header_size);
        entry.headerHash = wideHash(header);
        if (!headerImages.contains(entry.headerHash)) headerImages.emplace(entry.headerHash, std::vector<std::byte>(header.begin(), header.end()));
        const auto* code = const_cast<const void*>(fields.code);
        if (registration && code != nullptr && fields.shader_size != 0 && GuestMemory::Accessible(code, fields.shader_size)) {
            entry.code = pointer(code);
            entry.codeBytes = fields.shader_size;
            entry.codeHash = wideHash(std::span(static_cast<const std::byte*>(code), fields.shader_size));
        }
        result.push_back(entry);
    }
    return result;
}

void FrameRecorder::writePrologue() {
    const auto registry = DriverDetail::Driver::Get().RegisteredShaders();
    ReplayedMemory replayed;
    std::vector<std::byte> current;
    const auto holdsNow = [&](std::uint64_t address, std::uint64_t bytes, std::uint64_t key) {
        if (replayed.Holds(address, bytes, key)) return true;
        if (replayed.Rewritten(address, bytes)) return false;
        const auto* at = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(address));
        if (!GuestMemory::Accessible(at, static_cast<std::size_t>(bytes))) return false;
        current.resize(static_cast<std::size_t>(bytes));
        std::memcpy(current.data(), at, current.size());
        const auto now = wideHash(current);
        replayed.Note(address, bytes, now);
        return now == key;
    };
    const auto registeredCode = [&](const PrologueShader& shader) -> std::shared_ptr<const DriverDetail::ShaderSnapshot> {
        if (registry == nullptr) return {};
        const auto found = registry->find(shader.code);
        if (found == registry->end()) return {};
        for (const auto& snapshot : found->second) {
            const auto bytes = std::as_bytes(std::span(snapshot->code));
            if (bytes.size() == shader.codeBytes && wideHash(bytes) == shader.codeHash) return snapshot;
        }
        return {};
    };
    struct Write {
        std::uint64_t address;
        std::span<const std::byte> bytes;
        std::uint64_t key;
    };
    std::vector<Write> writes;
    std::vector<std::shared_ptr<const DriverDetail::ShaderSnapshot>> pinned;
    for (const auto& entry : prologue) {
        writes.clear();
        pinned.clear();
        bool stale = false;
        for (const auto& shader : entry.shaders) {
            const auto& image = headerImages.at(shader.headerHash);
            if (!holdsNow(shader.header, image.size(), shader.headerHash)) writes.push_back({shader.header, image, shader.headerHash});
            if (shader.codeBytes == 0 || holdsNow(shader.code, shader.codeBytes, shader.codeHash)) continue;
            auto snapshot = registeredCode(shader);
            if (snapshot == nullptr) {
                stale = true;
                break;
            }
            writes.push_back({shader.code, std::as_bytes(std::span(snapshot->code)), shader.codeHash});
            pinned.push_back(std::move(snapshot));
        }
        if (stale) {
            ++prologueDropped;
            continue;
        }
        if (!writes.empty()) {
            const auto group = ++deltaGroups;
            for (const auto& write : writes) {
                writer->WriteDelta(group, 0, write.address, write.bytes);
                replayed.Write(write.address, write.bytes.size(), write.key);
                ++prologueRewrites;
                prologueRewriteBytes += write.bytes.size();
            }
            auto memory = make(FrameCallKind::Memory);
            memory.ordinal = entry.call.ordinal;
            memory.thread = entry.call.thread;
            memory.timeNs = entry.call.timeNs;
            memory.values = {group};
            writer->WriteCall(memory);
        }
        writer->WriteCall(entry.call);
    }
    prologueCallCount = prologue.size();
    prologue.clear();
    prologue.shrink_to_fit();
    headerImages.clear();
}

void FrameRecorder::FrameBoundary() {
    std::lock_guard lock(mutex);
    ++boundaries;
    if (state == State::Recording) {
        if (++recordedFrames >= frames) finish();
        return;
    }
    if (state == State::Waiting && triggered()) start();
}

bool FrameRecorder::triggered() {
    if (!trigger.empty()) {
        std::error_code error;
        if (!std::filesystem::exists(trigger, error)) return false;
        std::ifstream file(trigger);
        std::uint64_t requested = 0;
        if (file >> requested && requested != 0) frames = requested;
        return true;
    }
    return boundaries >= std::max<std::uint64_t>(atBoundary, 1);
}

void FrameRecorder::emitDelta() {
    std::vector<Span> spans;
    for (const auto& range : ranges) {
        if (range.kind == FrameRangeKind::Watched || range.kind == FrameRangeKind::Unwatched) spans.push_back({range.address, range.bytes, generation});
    }
    const auto mark = GuestMemory::WriteGeneration();
    const auto added = newRanges();
    writeDelta(spans, {}, added);
    generation = mark;
    shaderMarks.clear();
}

std::vector<FrameRecorder::Range> FrameRecorder::newRanges() {
    const auto current = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    if (current == allocationsGeneration) return {};
    allocationsGeneration = current;
    std::map<std::uint64_t, std::uint64_t> covered;
    for (const auto& range : ranges) {
        auto& end = covered[range.address];
        end = std::max(end, range.address + range.bytes);
    }
    std::vector<Range> added;
    for (const auto& range : committedRanges()) {
        const auto end = range.address + range.bytes;
        for (auto at = range.address; at < end;) {
            const auto next = covered.upper_bound(at);
            if (next != covered.begin() && std::prev(next)->second > at) {
                at = std::prev(next)->second;
                continue;
            }
            auto piece = range;
            piece.address = at;
            piece.bytes = (next == covered.end() ? end : std::min(end, next->first)) - at;
            added.push_back(piece);
            at += piece.bytes;
        }
    }
    return added;
}

void FrameRecorder::emitShaderDelta(std::span<const Shader* const> shaders) {
    std::set<std::uint64_t> blocks;
    std::set<std::uint64_t> hostPages;
    std::uintptr_t arenaBase = 0;
    std::size_t arenaBytes = 0;
    GuestArena::GuestArenaRange_nid_postfix(&arenaBase, &arenaBytes);
    const auto note = [&](const void* address, std::uint64_t bytes = 0) {
        if (address == nullptr) return;
        const auto at = pointer(address);
        const bool arena = at >= arenaBase && at - arenaBase < arenaBytes;
        const auto blockBytes = arena ? FrameCaptureWriter::BlockBytes : HostPageBytes;
        const auto begin = at & ~std::uint64_t{blockBytes - 1};
        const auto count = bytes != 0 ? (at - begin + bytes + blockBytes - 1) / blockBytes : 2;
        auto& pages = arena ? blocks : hostPages;
        for (std::uint64_t index = 0; index < count; ++index) pages.insert(begin + index * blockBytes);
    };
    for (const auto* shader : shaders) {
        if (shader == nullptr || !GuestMemory::Accessible(shader, sizeof(Shader))) continue;
        note(shader);
        note(const_cast<const void*>(shader->code), shader->shader_size);
        note(shader->user_data);
        note(shader->cx_registers);
        note(shader->sh_registers);
        note(shader->specials);
        note(shader->input_semantics);
        note(shader->output_semantics);
        if (shader->user_data == nullptr || !GuestMemory::Accessible(shader->user_data, sizeof(ShaderUserData))) continue;
        note(shader->user_data->direct_resource_offset);
        for (const auto* sharps : shader->user_data->sharp_resource_offset) note(sharps);
    }
    if (blocks.empty() && hostPages.empty()) return;
    const auto mark = GuestMemory::WriteGeneration();
    std::vector<Span> spans;
    for (const auto block : blocks) {
        const auto found = shaderMarks.find(block);
        spans.push_back({block, FrameCaptureWriter::BlockBytes, found != shaderMarks.end() ? std::max(found->second, generation) : generation});
    }
    const std::vector<std::uint64_t> pages(hostPages.begin(), hostPages.end());
    writeDelta(spans, pages);
    for (const auto block : blocks) shaderMarks[block] = mark;
}

void FrameRecorder::writeDelta(std::span<const Span> spans, std::span<const std::uint64_t> hostPages, std::span<const Range> added) {
    if (abandonIfDiskLow()) return;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> blocks;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> written;
    for (const auto& span : spans) {
        GuestMemory::CollectWrittenBlocks(span.address, static_cast<std::size_t>(span.bytes), span.since, written);
        blocks.insert(blocks.end(), written.begin(), written.end());
    }
    auto& driver = DriverDetail::Driver::Get();
    auto [accepted, completed] = driver.SerialProgress();
    if (completed < accepted && (!added.empty() || !blocks.empty())) {
        if (!driver.WaitIdleFor(std::chrono::seconds(10))) {
            stop("stopped: the GPU work submitted before a memory delta did not finish within 10 s");
            return;
        }
        std::tie(accepted, completed) = driver.SerialProgress();
    }
    const auto group = deltaGroups + 1;
    bool any = false;
    std::vector<std::byte> bytes;
    for (const auto& range : added) {
        writer->WriteRange(range.address, range.bytes, range.kind, range.protection, range.access);
        ranges.push_back(range);
        ++lateRanges;
        if (!range.readable) {
            unreadableBytes += range.bytes;
            continue;
        }
        for (std::uint64_t at = 0; at < range.bytes;) {
            bytes.resize(static_cast<std::size_t>(std::min<std::uint64_t>(FrameCaptureWriter::BlockBytes, range.bytes - at)));
            GuestMemory::ReadCommittedUnsynced(range.address + at, bytes);
            const bool zero = std::all_of(bytes.begin(), bytes.end(), [](std::byte value) { return value == std::byte{0}; });
            if (!zero) {
                writer->WriteDelta(group, completed, range.address + at, bytes);
                deltaBytes += bytes.size();
                any = true;
            }
            at += bytes.size();
        }
    }
    for (const auto page : hostPages) {
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(page)), HostPageBytes)) continue;
        bytes.resize(HostPageBytes);
        std::memcpy(bytes.data(), reinterpret_cast<const void*>(static_cast<std::uintptr_t>(page)), HostPageBytes);
        const auto hash = contentHash(bytes);
        auto& known = hostPageHashes[page];
        if (known == hash) continue;
        known = hash;
        writer->WriteDelta(group, completed, page, bytes);
        deltaBytes += bytes.size();
        any = true;
    }
    for (const auto& [begin, end] : blocks) {
        bytes.resize(static_cast<std::size_t>(end - begin));
        GuestMemory::ReadCommittedUnsynced(begin, bytes);
        writer->WriteDelta(group, completed, begin, bytes);
        deltaBytes += bytes.size();
        any = true;
    }
    if (!any) return;
    deltaGroups = group;
    auto call = make(FrameCallKind::Memory);
    call.ordinal = ++ordinal;
    call.accepted = accepted;
    call.completed = completed;
    call.values = {group};
    writer->WriteCall(call);
}

void FrameRecorder::snapshot(const Range& range) {
    writer->WriteRange(range.address, range.bytes, range.kind, range.protection, range.access);
    if (!range.readable) {
        unreadableBytes += range.bytes;
        return;
    }
    std::vector<std::byte> bytes;
    for (std::uint64_t at = 0; at < range.bytes; at += SnapshotPieceBytes) {
        if (abandonIfDiskLow()) return;
        bytes.resize(static_cast<std::size_t>(std::min<std::uint64_t>(SnapshotPieceBytes, range.bytes - at)));
        GuestMemory::ReadCommitted(range.address + at, bytes);
        snapshotBytes += writer->WriteSnapshot(range.address + at, bytes);
    }
}

bool FrameRecorder::abandonIfDiskLow() {
    std::error_code error;
    const auto space = std::filesystem::space(directory, error);
    if (error || space.available >= MinimumFreeBytes) return false;
    stop("stopped: less than 4 GiB free on the capture volume");
    return true;
}

void FrameRecorder::stop(const char* reason) {
    warnings.push_back(reason);
    std::fprintf(stderr, "[frame-capture] %s\n", reason);
    try {
        writer->Finish(header());
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "[frame-capture] %s\n", failure.what());
    }
    writer.reset();
    state = State::Done;
}

void FrameRecorder::start() {
    ++startAttempts;
    auto& driver = DriverDetail::Driver::Get();
    if (!driver.WaitIdleFor(std::chrono::seconds(10))) {
        std::fprintf(stderr, "[frame-capture] the driver did not go idle within 10 s at frame boundary %llu (attempt %llu)\n", static_cast<unsigned long long>(boundaries), static_cast<unsigned long long>(startAttempts));
        if (startAttempts >= 5) {
            std::fprintf(stderr, "[frame-capture] giving up: no idle frame boundary\n");
            state = State::Done;
        }
        return;
    }
    if (!GuestMemory::WriteWatched()) throw std::runtime_error("frame capture: the guest write watch is unavailable, so CPU stores during the frames cannot be recorded");
    startedAt = std::chrono::steady_clock::now();
    startBoundary = boundaries;
    allocationsGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    ranges = committedRanges();
    writer = std::make_unique<FrameCaptureWriter>(directory);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> written;
    for (const auto& range : ranges) {
        if (range.kind == FrameRangeKind::Watched || range.kind == FrameRangeKind::Unwatched) GuestMemory::CollectWrittenBlocks(range.address, static_cast<std::size_t>(range.bytes), std::numeric_limits<std::uint64_t>::max(), written);
    }
    generation = GuestMemory::WriteGeneration();
    for (const auto& range : ranges) {
        snapshot(range);
        if (!writer) return;
    }
    writePrologue();
    std::uintptr_t ringBase = 0;
    std::uint32_t ringSize = 0;
    sceAgcDriverGetTFRing(&ringBase, &ringSize);
    auto ring = make(FrameCallKind::SetTessellationFactorRing);
    ring.ordinal = ++ordinal;
    ring.address = ringBase;
    ring.size = ringSize;
    writer->WriteCall(ring);
    const auto pending = driver.PendingEopFlipTokens();
    const std::set<std::uint64_t> open(pending.begin(), pending.end());
    for (const auto& call : reservations) {
        if (open.contains(call.values.back())) writer->WriteCall(call);
    }
    auto begin = make(FrameCallKind::FrameStart);
    begin.ordinal = ++ordinal;
    std::tie(begin.accepted, begin.completed) = driver.SerialProgress();
    begin.values = {boundaries};
    writer->WriteCall(begin);
    state = State::Recording;
    recordedFrames = 0;
    std::fprintf(stderr, "[frame-capture] started at frame boundary %llu: %zu ranges, %.1f MiB snapshot in %.1f s\n", static_cast<unsigned long long>(boundaries), ranges.size(),
                 static_cast<double>(snapshotBytes) / (1024.0 * 1024.0), std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count());
}

void FrameRecorder::finish() {
    emitDelta();
    if (!writer) return;
    auto end = make(FrameCallKind::End);
    end.ordinal = ++ordinal;
    std::tie(end.accepted, end.completed) = DriverDetail::Driver::Get().SerialProgress();
    writer->WriteCall(end);
    writeReferenceFrames();
    writer->Finish(header());
    std::fprintf(stderr, "[frame-capture] %llu frames, %llu calls, %llu memory deltas (%.1f MiB), %.1f MiB raw, %.1f MiB stored in %s\n", static_cast<unsigned long long>(recordedFrames),
                 static_cast<unsigned long long>(recordedCalls), static_cast<unsigned long long>(deltaGroups), static_cast<double>(deltaBytes) / (1024.0 * 1024.0),
                 static_cast<double>(writer->RawBytes()) / (1024.0 * 1024.0), static_cast<double>(writer->StoredBytes()) / (1024.0 * 1024.0), directory.string().c_str());
    writer.reset();
    state = State::Done;
}

std::string FrameRecorder::header() const {
    std::uintptr_t arenaBase = 0;
    std::size_t arenaBytes = 0;
    GuestArena::GuestArenaRange_nid_postfix(&arenaBase, &arenaBytes);
    std::string text = "{\"magic\":\"aps5-frame-capture 1\"";
    text += ",\"commit\":" + jsonString(APS5_TRACE_COMMIT);
    text += std::string(",\"dirty\":") + (APS5_TRACE_DIRTY < 0 ? "null" : APS5_TRACE_DIRTY != 0 ? "true" : "false");
    text += ",\"arena_base\":" + std::to_string(arenaBase) + ",\"arena_bytes\":" + std::to_string(arenaBytes);
    text += ",\"start_boundary\":" + std::to_string(startBoundary) + ",\"frames\":" + std::to_string(recordedFrames);
    text += ",\"calls\":" + std::to_string(recordedCalls) + ",\"prologue_calls\":" + std::to_string(prologueCallCount);
    text += ",\"prologue_rewrites\":" + std::to_string(prologueRewrites) + ",\"prologue_rewrite_bytes\":" + std::to_string(prologueRewriteBytes) + ",\"prologue_dropped\":" + std::to_string(prologueDropped);
    text += ",\"ranges\":" + std::to_string(ranges.size()) + ",\"late_ranges\":" + std::to_string(lateRanges) + ",\"snapshot_bytes\":" + std::to_string(snapshotBytes) + ",\"unreadable_bytes\":" + std::to_string(unreadableBytes);
    text += ",\"memory_groups\":" + std::to_string(deltaGroups) + ",\"delta_bytes\":" + std::to_string(deltaBytes);
    text += ",\"capture_seconds\":" + std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count());
    text += ",\"env\":{";
    bool first = true;
    for (const auto& [name, value] : captureEnvironment()) {
        text += (first ? "" : ",") + jsonString(name) + ":" + jsonString(value);
        first = false;
    }
    text += "},\"warnings\":[";
    first = true;
    for (const auto& warning : warnings) {
        text += (first ? "" : ",") + jsonString(warning);
        first = false;
    }
    return text + "]}\n";
}

std::vector<FrameRecorder::Range> FrameRecorder::committedRanges() {
#ifdef _WIN32
    std::uintptr_t base = 0;
    std::size_t size = 0;
    GuestArena::GuestArenaRange_nid_postfix(&base, &size);
    if (size == 0) throw std::runtime_error("frame capture: the guest arena is not reserved");
    std::vector<Range> result;
    for (const auto& allocation : GuestAllocations::GuestAllocationsAcquireAll_nid_postfix()) {
        if (!allocation->readable) continue;
        const auto first = allocation->address;
        const auto end = first + allocation->bytes;
        const bool host = first < base || end > base + size;
        const auto access = FrameRangeAccess::Readable | (allocation->writable ? FrameRangeAccess::Writable : 0u) | (allocation->gpu ? FrameRangeAccess::Gpu : 0u);
        for (auto address = first; address < end;) {
            MEMORY_BASIC_INFORMATION info{};
            if (VirtualQuery(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(address)), &info, sizeof(info)) == 0) throw std::runtime_error("frame capture: VirtualQuery failed inside the guest arena");
            const auto regionEnd = std::min<std::uint64_t>(end, reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize);
            if (info.State == MEM_COMMIT) {
                Range range;
                range.address = address;
                range.bytes = regionEnd - address;
                range.protection = info.Protect;
                range.access = access;
                const auto page = info.Protect & 0xffu;
                range.readable = (info.Protect & PAGE_GUARD) == 0 && (page == PAGE_READONLY || page == PAGE_READWRITE || page == PAGE_WRITECOPY || page == PAGE_EXECUTE_READ || page == PAGE_EXECUTE_READWRITE || page == PAGE_EXECUTE_WRITECOPY);
                if (host) range.kind = FrameRangeKind::Host;
                else if (info.Type == MEM_PRIVATE) range.kind = GuestMemory::Watched(range.address, static_cast<std::size_t>(range.bytes)) ? FrameRangeKind::Watched : FrameRangeKind::Unwatched;
                else range.kind = FrameRangeKind::Section;
                auto& last = result.empty() ? range : result.back();
                if (!result.empty() && last.address + last.bytes == range.address && last.kind == range.kind && last.protection == range.protection && last.readable == range.readable && last.access == range.access) last.bytes += range.bytes;
                else result.push_back(range);
            }
            address = regionEnd;
        }
    }
    return result;
#elif defined(__linux__)
    std::uintptr_t base = 0;
    std::size_t size = 0;
    GuestArena::GuestArenaRange_nid_postfix(&base, &size);
    if (size == 0) throw std::runtime_error("frame capture: the guest arena is not reserved");
    std::vector<Range> result;
    for (const auto& allocation : GuestAllocations::GuestAllocationsAcquireAll_nid_postfix()) {
        if (!allocation->readable) continue;
        const auto first = allocation->address;
        const auto end = first + allocation->bytes;
        const bool host = first < base || end > base + size;
        const auto access = FrameRangeAccess::Readable | (allocation->writable ? FrameRangeAccess::Writable : 0u) | (allocation->gpu ? FrameRangeAccess::Gpu : 0u);
        HostMapping::Walk(first, end, "frame capture", [&](const HostMapping::Mapping& mapping) {
            if (!mapping.readable && !mapping.writable && !mapping.executable) return;
            const auto add = [&](std::uint64_t address, std::uint64_t regionEnd, bool readable) {
                if (address >= regionEnd) return;
                Range range;
                range.address = address;
                range.bytes = regionEnd - address;
                range.protection = windowsProtection(mapping);
                range.access = access;
                range.readable = readable;
                if (host) range.kind = FrameRangeKind::Host;
                else if (GuestMemory::Watched(range.address, static_cast<std::size_t>(range.bytes))) range.kind = FrameRangeKind::Watched;
                else if (!mapping.shared && mapping.inode == 0) range.kind = FrameRangeKind::Unwatched;
                else range.kind = FrameRangeKind::Section;
                auto& last = result.empty() ? range : result.back();
                if (!result.empty() && last.address + last.bytes == range.address && last.kind == range.kind && last.protection == range.protection && last.readable == range.readable && last.access == range.access) last.bytes += range.bytes;
                else result.push_back(range);
            };
            const auto address = std::max<std::uint64_t>(first, mapping.start);
            const auto regionEnd = std::min<std::uint64_t>(end, mapping.end);
            const auto backed = std::clamp(fileBackedEnd(mapping), address, regionEnd);
            add(address, backed, mapping.readable);
            add(backed, regionEnd, false);
        });
    }
    return result;
#else
    throw std::runtime_error("frame capture: guest memory capture is implemented on Windows and Linux only");
#endif
}

}
