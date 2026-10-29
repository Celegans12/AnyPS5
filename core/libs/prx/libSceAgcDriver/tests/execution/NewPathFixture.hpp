#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_NEWPATHFIXTURE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_NEWPATHFIXTURE_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace NewPath {

using AgcDriver::Graphics::Require;

constexpr std::uint64_t PageBytes = 65536;
constexpr std::uint64_t SubpageBytes = 4096;
constexpr std::uint32_t SubpagesPerPage = 16;
constexpr std::uint32_t WordsPerSubpage = SubpageBytes / 4;
constexpr std::uint32_t WordsPerPage = PageBytes / 4;
constexpr std::uint32_t RawBuffer = 0x31016facu;
constexpr std::uint32_t Format32Sint = 21u;
constexpr std::uint32_t IdentitySwizzle = 0xfacu;
constexpr std::uint32_t Image2D = 9u;

namespace Isa {

constexpr std::uint32_t InlineInteger(std::uint32_t value) { return 128u + value; }
constexpr std::uint32_t Literal = 0xffu;
constexpr std::uint32_t Vgpr(std::uint32_t index) { return 256u + index; }
constexpr std::uint32_t NullSgpr = 0x7du;
constexpr std::uint32_t ZeroOffset = 0x80u;

constexpr std::uint32_t VMovB32 = 0x01u;
constexpr std::uint32_t VReadFirstLaneB32 = 0x02u;
constexpr std::uint32_t VLshlRevB32 = 0x1au;
constexpr std::uint32_t VAddNcU32 = 0x25u;

constexpr std::uint32_t Vop1(std::uint32_t opcode, std::uint32_t vdst, std::uint32_t src0) {
    return (0x3fu << 25u) | (vdst << 17u) | (opcode << 9u) | src0;
}

constexpr std::uint32_t Vop2(std::uint32_t opcode, std::uint32_t vdst, std::uint32_t src0, std::uint32_t vsrc1) {
    return (opcode << 25u) | (vdst << 17u) | (vsrc1 << 9u) | src0;
}

constexpr std::uint32_t BufferLoadDword = 0x0cu;
constexpr std::uint32_t BufferStoreDword = 0x1cu;

constexpr std::array<std::uint32_t, 2> Mubuf(std::uint32_t opcode, std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t resource, bool glc = false, std::uint32_t offset = 0) {
    return {0xe0000000u | (opcode << 18u) | (glc ? 0x4000u : 0u) | 0x1000u | (offset & 0xfffu), (ZeroOffset << 24u) | ((resource / 4u) << 16u) | (vdata << 8u) | vaddr};
}

constexpr std::uint32_t GlobalLoadDword = 0x0cu;
constexpr std::uint32_t GlobalStoreDword = 0x1cu;

constexpr std::array<std::uint32_t, 2> Global(std::uint32_t opcode, std::uint32_t vdst, std::uint32_t vaddr, std::uint32_t vdata, std::uint32_t saddr, std::uint32_t offset = 0) {
    return {0xdc000000u | (opcode << 18u) | (2u << 14u) | (offset & 0xfffu), (vdst << 24u) | (saddr << 16u) | (vdata << 8u) | vaddr};
}

constexpr std::uint32_t SLoadDwordX4 = 0x02u;

constexpr std::array<std::uint32_t, 2> Smem(std::uint32_t opcode, std::uint32_t sdata, std::uint32_t sbase, std::uint32_t offset) {
    return {0xf4000000u | (opcode << 18u) | (sdata << 6u) | (sbase / 2u), (NullSgpr << 25u) | (offset & 0x1fffffu)};
}

constexpr std::array<std::uint32_t, 2> ImageStoreX(std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t resource) {
    return {0xf0201108u, ((resource / 4u) << 16u) | (vdata << 8u) | vaddr};
}

constexpr std::uint32_t WaitAll = 0xbf8c0000u;
constexpr std::uint32_t EndProgram = 0xbf810000u;

}

struct alignas(256) Program {
    std::array<std::uint32_t, 512> words{};
    std::size_t size = 0;

    Program& operator<<(std::uint32_t word) {
        Require(size < words.size(), "new path fixture: a test program does not fit");
        words[size++] = word;
        return *this;
    }

    Program& operator<<(const std::array<std::uint32_t, 2>& pair) { return *this << pair[0] << pair[1]; }

    std::span<const std::uint32_t> Code() const { return {words.data(), size}; }
    std::uint64_t Address() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(words.data())); }
};

inline std::array<std::uint32_t, 4> BufferResource(std::uint64_t address, std::uint32_t bytes, std::uint32_t stride = 0) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), bytes, RawBuffer};
}

inline std::array<std::uint32_t, 8> ImageResource(std::uint64_t address, std::uint32_t format, std::uint32_t width, std::uint32_t height) {
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((width - 1u) & 3u) << 30u),
        ((width - 1u) >> 2u) | ((height - 1u) << 14u),
        IdentitySwizzle | (Image2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

inline std::string Hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

inline std::uint32_t Low(std::uint64_t value) { return static_cast<std::uint32_t>(value); }
inline std::uint32_t High(std::uint64_t value) { return static_cast<std::uint32_t>(value >> 32u); }

class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        file = std::tmpfile();
        Require(file != nullptr, "new path fixture: cannot open a temporary file for stderr");
#ifdef _WIN32
        saved = _dup(_fileno(stderr));
        _dup2(_fileno(file), _fileno(stderr));
#else
        saved = dup(fileno(stderr));
        dup2(fileno(file), fileno(stderr));
#endif
    }

    ~StderrCapture() {
        if (file != nullptr) Finish();
    }

    StderrCapture(const StderrCapture&) = delete;
    StderrCapture& operator=(const StderrCapture&) = delete;

    std::string Finish() {
        std::fflush(stderr);
#ifdef _WIN32
        _dup2(saved, _fileno(stderr));
        _close(saved);
#else
        dup2(saved, fileno(stderr));
        close(saved);
#endif
        std::rewind(file);
        std::string text;
        char buffer[4096];
        for (std::size_t read = 0; (read = std::fread(buffer, 1, sizeof(buffer), file)) != 0;) text.append(buffer, read);
        std::fclose(file);
        file = nullptr;
        std::fwrite(text.data(), 1, text.size(), stderr);
        return text;
    }

private:
    std::FILE* file = nullptr;
    int saved = -1;
};

class Commands {
public:
    void Registers(std::uint32_t first, std::span<const std::uint32_t> values) {
        words.push_back(0xc0007600u | (static_cast<std::uint32_t>(values.size()) << 16u));
        words.push_back(first);
        words.insert(words.end(), values.begin(), values.end());
    }

    void Registers(std::uint32_t first, std::initializer_list<std::uint32_t> values) { Registers(first, std::span<const std::uint32_t>(values.begin(), values.size())); }

    void Dispatch(const Program& program, std::span<const std::uint32_t> userData, std::uint32_t threads, std::uint32_t groups = 1) {
        Require(userData.size() <= 16u, "new path fixture: more than 16 user data registers");
        Require(program.Address() % 256u == 0u, "new path fixture: a program is not 256-byte aligned");
        Registers(0x207, {threads, 1u, 1u});
        Registers(0x20c, {Low(program.Address() >> 8u), Low(program.Address() >> 40u)});
        Registers(0x213, {(static_cast<std::uint32_t>(userData.size()) << 1u) | (1u << 7u)});
        if (!userData.empty()) Registers(0x240, userData);
        words.insert(words.end(), {0xc0031500u, groups, 1u, 1u, 0x41u});
    }

    void WriteData(std::uint64_t address, std::initializer_list<std::uint32_t> values) {
        words.push_back(0xc0003700u | ((2u + static_cast<std::uint32_t>(values.size())) << 16u));
        words.insert(words.end(), {0x00100200u, Low(address), High(address)});
        words.insert(words.end(), values);
    }

    void DmaCopy(std::uint64_t destination, std::uint64_t source, std::uint32_t bytes) {
        words.insert(words.end(), {0xc0055000u, 0u, Low(source), High(source), Low(destination), High(destination), bytes});
    }

    void Wait(std::uint64_t address, std::uint32_t function, std::uint32_t reference, std::uint32_t mask) {
        words.insert(words.end(), {0xc0053c00u, 0x10u | function, Low(address), High(address), reference, mask, 0x19u});
    }

    void Append(std::uint32_t opcode, std::span<const std::uint32_t> payload) {
        Require(!payload.empty(), "new path fixture: a packet has no payload");
        words.push_back(0xc0000000u | ((static_cast<std::uint32_t>(payload.size()) - 1u) << 16u) | (opcode << 8u));
        words.insert(words.end(), payload.begin(), payload.end());
    }

    void Append(std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) { Append(opcode, std::span<const std::uint32_t>(payload.begin(), payload.size())); }

    void Barrier() {
        Append(0x46u, {0x407u});
        Append(0x58u, {0u, 0xffffffffu, 0xffu, 0u, 0u, 10u, 0x4380u});
    }

    void Submit(std::uint32_t queue) {
        Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
        AgcDriver::Submit(&packet, queue);
        words.clear();
    }

private:
    std::vector<std::uint32_t> words;
};

inline void WaitIdle() {
    AgcDriverWaitIdle_nid_postfix();
}

inline void Locked(const std::function<void()>& action) {
    const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    action();
}

inline std::uint64_t PageOf(std::uint64_t address) {
    return address >> 16u;
}

#ifdef _WIN32

class SharedPages {
public:
    explicit SharedPages(std::uint32_t pages, bool cpuWritable = true) : bytes(std::uint64_t{pages} * PageBytes), cpuWritable(cpuWritable) {
        base = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, PageBytes));
        Require(base != nullptr, "new path: cannot reserve guest pages for a section");
        Map();
    }

    ~SharedPages() {
        Unmap();
        GuestArena::GuestArenaRelease_nid_postfix(base, bytes);
    }

    SharedPages(const SharedPages&) = delete;
    SharedPages& operator=(const SharedPages&) = delete;

    void Map() {
        Require(section == nullptr, "new path: a section is mapped twice");
        section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, High(bytes), Low(bytes), nullptr);
        Require(section != nullptr, "new path: cannot create a section");
        GuestArena::GuestArenaMap_nid_postfix(base, bytes, section, 0, PAGE_READWRITE);
        GuestAllocations::Mutation().Add(base, bytes, true, cpuWritable, true);
    }

    void Unmap() {
        if (section == nullptr) return;
        GuestAllocations::Mutation().Remove(base);
        GuestArena::GuestArenaReset_nid_postfix(base, bytes);
        CloseHandle(section);
        section = nullptr;
    }

    std::uint64_t Address(std::uint64_t page = 0, std::uint32_t subpage = 0, std::uint32_t word = 0) const {
        return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(base)) + page * PageBytes + std::uint64_t{subpage} * SubpageBytes + std::uint64_t{word} * 4u;
    }

    volatile std::uint32_t* Words(std::uint64_t page = 0) const { return reinterpret_cast<volatile std::uint32_t*>(base + page * PageBytes); }
    std::uint64_t Bytes() const { return bytes; }

private:
    std::uint8_t* base = nullptr;
    std::uint64_t bytes = 0;
    bool cpuWritable = true;
    HANDLE section = nullptr;
};

class PrivatePages {
public:
    explicit PrivatePages(std::uint32_t pages, bool cpuWritable = true) : bytes(std::uint64_t{pages} * PageBytes) {
        base = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, PageBytes));
        Require(base != nullptr, "new path: cannot reserve private guest pages");
        GuestArena::GuestArenaCommit_nid_postfix(base, bytes, PAGE_READWRITE, 16384);
        GuestAllocations::Mutation().Add(base, bytes, true, cpuWritable, true);
    }

    ~PrivatePages() {
        GuestAllocations::Mutation().Remove(base);
        GuestArena::GuestArenaReset_nid_postfix(base, bytes);
        GuestArena::GuestArenaRelease_nid_postfix(base, bytes);
    }

    PrivatePages(const PrivatePages&) = delete;
    PrivatePages& operator=(const PrivatePages&) = delete;

    std::uint64_t Address(std::uint64_t offset = 0) const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(base)) + offset; }
    std::uint8_t* Data() const { return base; }
    volatile std::uint32_t* Words() const { return reinterpret_cast<volatile std::uint32_t*>(base); }
    std::uint64_t Bytes() const { return bytes; }

private:
    std::uint8_t* base = nullptr;
    std::uint64_t bytes = 0;
};

inline const Program& MirrorReadProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop2(VLshlRevB32, 1, InlineInteger(12), 0)
              << Mubuf(BufferLoadDword, 20, 1, 0)
              << WaitAll
              << Vop1(VMovB32, 30, Vgpr(0))
              << Vop1(VMovB32, 31, InlineInteger(0))
              << ImageStoreX(20, 30, 4)
              << EndProgram;
        return built;
    }();
    return program;
}

class MirrorReader {
public:
    static constexpr std::uint32_t Lanes = 256;
    static constexpr std::uint32_t Pages = Lanes / SubpagesPerPage;
    static constexpr std::uint32_t Sentinel = 0xdeadbeefu;

    MirrorReader() : texels(1) {}

    void Record(Commands& commands, std::uint64_t address, std::uint32_t pages = Pages) {
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) texels.Words()[lane] = Sentinel;
        std::array<std::uint32_t, 12> user{};
        const auto buffer = BufferResource(address, static_cast<std::uint32_t>(std::uint64_t{pages} * PageBytes));
        const auto image = ImageResource(texels.Address(), Format32Sint, Lanes, 1u);
        std::copy(buffer.begin(), buffer.end(), user.begin());
        std::copy(image.begin(), image.end(), user.begin() + 4);
        commands.Dispatch(MirrorReadProgram(), user, Lanes);
    }

    std::vector<std::uint32_t> Collect() {
        WaitIdle();
        AgcDriver::Graphics::StorageTexture::FlushPending(texels.Address(), Lanes * 4u, nullptr, "new path mirror reader");
        WaitIdle();
        AgcDriver::GuestMemory::FlushGpuWrites(texels.Address(), Lanes * 4u);
        const auto layout = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, Format32Sint, Lanes, 1u, 1u).at(0);
        std::vector<std::uint32_t> lanes(Lanes);
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) std::memcpy(&lanes[lane], texels.Data() + layout.tiledOffset + lane * 4u, 4u);
        return lanes;
    }

    std::vector<std::uint32_t> Read(std::uint64_t address, std::uint32_t pages = Pages) {
        Commands commands;
        Record(commands, address, pages);
        commands.Submit(0);
        return Collect();
    }

private:
    PrivatePages texels;
};

#endif

}

#endif
