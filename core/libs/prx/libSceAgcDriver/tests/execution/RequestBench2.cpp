// agc_driver_request_bench_tests <request.req>: runs a captured compute request (APS5_DUMP_SHADERS /
// pipeline_requests) through the real driver dispatch path with synthetic memory, for GPU timing of the
// game's own programs. Design-agent tool (local only, local/design-gpu): the captured regions (user data,
// SRT, constants) are mapped at their guest addresses with their captured bytes; every buffer V# and
// image T# the recompile binds gets memory at its guest address filled by BENCH_FILL. The output hash
// covers every buffer element the program may write, so translation variants can be compared for
// bit-identical results on the same input.
//
// Environment:
//   BENCH_GROUPS=x,y,z (default 1,1,1)   BENCH_REPEATS=n (default 5)   BENCH_RANGES=n extra BDA ranges (default 1200)
//   BENCH_FILL=zero|small|float|mix (default mix)   BENCH_SEED=n   BENCH_TIMEOUT=s (default 600)
//   BENCH_ENDPGM=pc[,pc...]  replace the instruction at each byte offset by s_endpgm (truncation profiling)
//   BENCH_MAX_BUFFER_MIB=n   cap of one synthetic buffer (default 256)
//   BENCH_DUMP=path          write the written buffers after the last repeat
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::uint64_t Env(const char* name, std::uint64_t fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? std::strtoull(value, nullptr, 0) : fallback;
}

std::string EnvText(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? std::string(value) : std::string(fallback);
}

std::string ReadText(const char* path) {
    std::ifstream file(path, std::ios::binary);
    std::ostringstream stream;
    stream << file.rdbuf();
    auto text = stream.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
    return text;
}

constexpr std::uint64_t Granule = 0x10000u;

struct Block {
    std::uint64_t begin;
    std::uint64_t end;
    std::uint8_t* data = nullptr;
    bool borrowed = false;
};

class GuestSpace {
public:
    void Want(std::uint64_t begin, std::uint64_t bytes) {
        if (bytes == 0) return;
        wanted.emplace_back(begin & ~(Granule - 1u), (begin + bytes + Granule - 1u) & ~(Granule - 1u));
    }

    void Map() {
        std::sort(wanted.begin(), wanted.end());
        for (const auto& [begin, end] : wanted) {
            if (!blocks.empty() && begin <= blocks.back().end) {
                blocks.back().end = std::max(blocks.back().end, end);
            } else {
                blocks.push_back({begin, end});
            }
        }
        for (auto& block : blocks) {
#ifdef _WIN32
            block.data = static_cast<std::uint8_t*>(VirtualAlloc(reinterpret_cast<void*>(block.begin), block.end - block.begin, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
            if (block.data == nullptr) {
                // The range may sit inside a reservation (the guest libc reserves its address space): commit in place.
                MEMORY_BASIC_INFORMATION info{};
                VirtualQuery(reinterpret_cast<void*>(block.begin), &info, sizeof(info));
                std::fprintf(stderr, "request bench: 0x%llx is in a region of state 0x%lx type 0x%lx base 0x%llx size 0x%llx; committing in place\n", static_cast<unsigned long long>(block.begin), info.State, info.Type, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info.AllocationBase)), static_cast<unsigned long long>(info.RegionSize));
                auto* pointer = reinterpret_cast<void*>(block.begin);
                const auto bytes = static_cast<std::size_t>(block.end - block.begin);
                if (GuestArena::GuestArenaAvailable_nid_postfix() && GuestArena::GuestArenaContains_nid_postfix(pointer, bytes)) {
                    GuestArena::GuestArenaMarkUsed_nid_postfix(pointer, bytes);
                    GuestArena::GuestArenaCommit_nid_postfix(pointer, bytes, PAGE_READWRITE, Granule);
                    block.data = static_cast<std::uint8_t*>(pointer);
                    block.borrowed = true;
                }
            }
#endif
            if (block.data == nullptr || reinterpret_cast<std::uint64_t>(block.data) != block.begin) {
                std::fprintf(stderr, "request bench: cannot map guest range 0x%llx-0x%llx at its address\n", static_cast<unsigned long long>(block.begin), static_cast<unsigned long long>(block.end));
                std::exit(4);
            }
            std::memset(block.data, 0, block.end - block.begin);
            GuestAllocations::Mutation().Add(block.data, block.end - block.begin, true, true, true);
            mapped += block.end - block.begin;
        }
    }

    std::uint8_t* At(std::uint64_t address, std::uint64_t bytes) {
        for (auto& block : blocks) {
            if (block.begin <= address && address + bytes <= block.end) return block.data + (address - block.begin);
        }
        return nullptr;
    }

    std::size_t Blocks() const { return blocks.size(); }
    std::uint64_t Mapped() const { return mapped; }

private:
    std::vector<std::pair<std::uint64_t, std::uint64_t>> wanted;
    std::vector<Block> blocks;
    std::uint64_t mapped = 0;
};

class ExtraRange {
public:
    ExtraRange() {
#ifdef _WIN32
        block = VirtualAlloc(nullptr, 0x20000u, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#endif
        GuestAllocations::Mutation().Add(block, 0x10000u, true, true, true);
    }
    ~ExtraRange() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#endif
    }

private:
    void* block = nullptr;
};

struct BufferUse {
    std::uint64_t base;
    std::uint64_t bytes;
    std::uint32_t stride;
    std::uint32_t format;
    bool written;
};

std::uint32_t Mix(std::uint64_t value) {
    value ^= value >> 33u;
    value *= 0xff51afd7ed558ccdull;
    value ^= value >> 33u;
    value *= 0xc4ceb9fe1a85ec53ull;
    value ^= value >> 33u;
    return static_cast<std::uint32_t>(value);
}

// Synthetic contents: "small" keeps every dword a small non-negative integer (indices and counts stay in
// range, floats are tiny), "float" every dword a float in [0.25, 1.25), "mix" alternates by 16-byte
// element: even elements small integers, odd elements floats, "zero" leaves the zero fill.
void Fill(std::uint8_t* data, std::uint64_t bytes, const std::string& mode, std::uint64_t seed) {
    if (mode == "zero") return;
    auto* words = reinterpret_cast<std::uint32_t*>(data);
    for (std::uint64_t word = 0; word < bytes / 4u; ++word) {
        const auto random = Mix(seed * 0x9e3779b97f4a7c15ull + word);
        const bool integer = mode == "small" || (mode == "mix" && ((word >> 2u) & 1u) == 0u);
        if (integer) {
            words[word] = random & 0xfu;
        } else {
            const float value = 0.25f + static_cast<float>(random & 0xffffu) / 65536.0f;
            std::memcpy(&words[word], &value, 4);
        }
    }
}

std::uint64_t Hash(const std::uint8_t* data, std::uint64_t bytes, std::uint64_t hash) {
    for (std::uint64_t byte = 0; byte < bytes; ++byte) hash = (hash ^ data[byte]) * 1099511628211ull;
    return hash;
}

}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::fprintf(stderr, "usage: agc_driver_request_bench_tests <request.req>\n");
            return VulkanTestSkipped;
        }
        const auto limit = static_cast<unsigned>(Env("BENCH_TIMEOUT", 600u));
        std::thread([limit] {
            std::this_thread::sleep_for(std::chrono::seconds(limit));
            std::fprintf(stderr, "request bench: exceeded %u s, exiting without waiting for the device\n", limit);
            std::fflush(stderr);
            TerminateProcess(GetCurrentProcess(), 3);
            std::_Exit(3);
        }).detach();
        auto request = ShaderRecompiler::RequestSerializer{}.Deserialize(ReadText(argv[1]));
        if (request.request.shader.stage != ShaderRecompiler::ShaderStage::Compute || !request.request.context.compute.has_value()) {
            std::fprintf(stderr, "request bench: not a compute request\n");
            return 2;
        }
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        request.request.target = device->ComputeTarget(request.request.context.waveSize);
        if (std::getenv("APS5_SPANNED_WAVE") != nullptr) request.request.target.spannedWaves = true;
        if (std::getenv("APS5_BINDLESS_IMAGES") != nullptr) request.request.target.bindlessImages = true;
        // BENCH_SINGLE_GROUP=1: recompile as the driver does for a 1x1x1 dispatch (ShaderComputeStageInfo::singleWorkgroup);
        // only meaningful with BENCH_GROUPS=1,1,1 (the default).
        if (std::getenv("BENCH_SINGLE_GROUP") != nullptr && request.request.context.compute.has_value()) {
            request.compute->singleWorkgroup = true;
            request.request.context.compute = request.compute;
        }
        request.request.useCache = false;
        // Truncation profiling: s_endpgm at the given byte offsets.
        {
            std::stringstream list(EnvText("BENCH_ENDPGM", ""));
            for (std::string item; std::getline(list, item, ',');) {
                const auto pc = std::strtoull(item.c_str(), nullptr, 0);
                if (pc / 4u >= request.shaderCode.size()) throw std::runtime_error("BENCH_ENDPGM past the code");
                request.shaderCode[pc / 4u] = 0xbf810000u;
                for (std::size_t region = 0; region < request.memory.size(); ++region) {
                    const auto& memory = request.memory[region];
                    const auto address = request.request.shader.codeAddress + pc;
                    if (memory.guestAddress <= address && address + 4u <= memory.guestAddress + memory.bytes.size()) {
                        auto* bytes = const_cast<std::byte*>(memory.bytes.data());
                        const std::uint32_t endpgm = 0xbf810000u;
                        std::memcpy(bytes + (address - memory.guestAddress), &endpgm, 4);
                    }
                }
                std::printf("request bench: s_endpgm at 0x%llx\n", static_cast<unsigned long long>(pc));
            }
        }
        // BENCH_PATCH=pc:word[,pc:word...]: replace code dwords (e.g. drop the group id from a single-group program so
        // every workgroup of a multi-group dispatch repeats the same work: a proxy for concurrent independent dispatches).
        {
            std::stringstream list(EnvText("BENCH_PATCH", ""));
            for (std::string item; std::getline(list, item, ',');) {
                const auto colon = item.find(':');
                if (colon == std::string::npos) throw std::runtime_error("BENCH_PATCH wants pc:word");
                const auto pc = std::strtoull(item.substr(0, colon).c_str(), nullptr, 0);
                const auto word = static_cast<std::uint32_t>(std::strtoull(item.substr(colon + 1).c_str(), nullptr, 0));
                if (pc / 4u >= request.shaderCode.size()) throw std::runtime_error("BENCH_PATCH past the code");
                request.shaderCode[pc / 4u] = word;
                for (const auto& memory : request.memory) {
                    const auto address = request.request.shader.codeAddress + pc;
                    if (memory.guestAddress <= address && address + 4u <= memory.guestAddress + memory.bytes.size()) std::memcpy(const_cast<std::byte*>(memory.bytes.data()) + (address - memory.guestAddress), &word, 4);
                }
                std::printf("request bench: code dword 0x%llx = 0x%08x\n", static_cast<unsigned long long>(pc), word);
            }
        }
        const auto compileStart = std::chrono::steady_clock::now();
        const auto shader = ShaderRecompiler::Recompile(request.request);
        const auto compileMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - compileStart).count();
        const auto& compute = *request.request.context.compute;
        std::printf("request bench: %s, %zu code words, threads %ux%ux%u, %zu SPIR-V words (recompile %.0f ms)\n", argv[1], request.request.shader.code.size(), compute.numThreads[0], compute.numThreads[1], compute.numThreads[2], shader.spirv.size(), compileMs);

        GuestSpace space;
        for (const auto& region : request.request.context.memory) space.Want(region.guestAddress, region.bytes.size());
        space.Want(request.request.shader.codeAddress, request.request.shader.code.size() * 4u);
        const auto maxBuffer = Env("BENCH_MAX_BUFFER_MIB", 256u) << 20u;
        std::vector<BufferUse> buffers;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> images;
        std::vector<bool> imageWritten;
        for (const auto& binding : shader.bindings) {
            if (binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers) {
                for (std::size_t i = 0; i + 4 <= binding.guestDescriptor.size(); i += 4) {
                    const auto* v = binding.guestDescriptor.data() + i;
                    const std::uint64_t base = v[0] | (std::uint64_t{v[1] & 0xffffu} << 32u);
                    const std::uint32_t stride = (v[1] >> 16u) & 0x3fffu;
                    const std::uint64_t records = v[2];
                    if (base == 0 || records == 0) continue;
                    auto bytes = stride != 0 ? records * stride : records;
                    bytes = std::min<std::uint64_t>(bytes, maxBuffer);
                    const bool written = i / 4 >= binding.bufferWritten.size() || binding.bufferWritten[i / 4];
                    buffers.push_back({base, bytes, stride, (v[3] >> 12u) & 0x7fu, written});
                    space.Want(base, bytes);
                }
            } else if (binding.role == ShaderRecompiler::DescriptorRole::GuestImages) {
                for (std::size_t i = 0; i + 8 <= binding.guestDescriptor.size(); i += 8) {
                    const std::span<const std::uint32_t> words(binding.guestDescriptor.data() + i, 8);
                    if (AgcDriver::Graphics::IsNullTextureResource(words)) continue;
                    const auto texture = AgcDriver::Graphics::DecodeTextureResource(words);
                    std::uint64_t bytes = std::uint64_t{texture.width} * texture.height * std::max(1u, texture.depthOrLastArray + 1u) * 16u * 2u;
                    bytes = std::min<std::uint64_t>(std::max<std::uint64_t>(bytes, 0x100000u), maxBuffer);
                    images.emplace_back(texture.baseAddress, bytes);
                    imageWritten.push_back(i / 8 >= binding.imageWritten.size() || binding.imageWritten[i / 8]);
                    space.Want(texture.baseAddress, bytes);
                    if (texture.dccAddress != 0) space.Want(texture.dccAddress, bytes / 64u + 0x10000u);
                    std::printf("  image %zu: %ux%u depth %u mips %u format %u tile %u base 0x%llx dcc 0x%llx\n", i / 8, texture.width, texture.height, texture.depthOrLastArray, texture.mipCount, texture.format, static_cast<unsigned>(texture.tileMode), static_cast<unsigned long long>(texture.baseAddress), static_cast<unsigned long long>(texture.dccAddress));
                }
            }
        }
        // BENCH_COPIES=k: k independent instances of the dispatch (k-1 relocated copies of every captured region and
        // bound buffer, BENCH_COPY_STRIDE apart; guest pointers and V# bases inside the captured bytes and the user
        // data are relocated), each recompiled for its own addresses: a proxy for k dispatches of different objects.
        const auto copies = std::max<std::uint64_t>(1u, Env("BENCH_COPIES", 1u));
        const auto copyStride = Env("BENCH_COPY_STRIDE", 0x400000000ull);
        std::vector<std::pair<std::uint64_t, std::uint64_t>> owned;
        const auto codeBegin = request.request.shader.codeAddress;
        const auto codeEnd = codeBegin + request.request.shader.code.size() * 4u;
        for (const auto& region : request.request.context.memory) {
            if (region.guestAddress >= codeBegin && region.guestAddress < codeEnd) continue;
            owned.emplace_back(region.guestAddress & ~(Granule - 1u), (region.guestAddress + region.bytes.size() + Granule - 1u) & ~(Granule - 1u));
        }
        for (const auto& buffer : buffers) owned.emplace_back(buffer.base & ~(Granule - 1u), (buffer.base + buffer.bytes + Granule - 1u) & ~(Granule - 1u));
        const auto isOwned = [&](std::uint64_t address) {
            for (const auto& [begin, end] : owned) if (address >= begin && address < end) return true;
            return false;
        };
        struct Copy {
            std::vector<std::vector<std::byte>> bytes;
            std::vector<ShaderRecompiler::MemoryRegion> regions;
            std::vector<std::uint32_t> userData;
            std::uint64_t offset = 0;
            std::unique_ptr<ShaderRecompiler::RecompileResult> shader;
        };
        std::vector<Copy> copyList(copies - 1u);
        std::uint64_t relocated = 0;
        const auto relocate = [&](std::uint32_t* words, std::size_t count, std::uint64_t offset) {
            for (std::size_t i = 0; i + 1 < count;) {
                const std::uint64_t base = words[i] | (std::uint64_t{words[i + 1] & 0xffffu} << 32u);
                if (base >= 0x1000000000ull && isOwned(base)) {
                    const auto moved = base + offset;
                    words[i] = static_cast<std::uint32_t>(moved);
                    words[i + 1] = (words[i + 1] & 0xffff0000u) | static_cast<std::uint32_t>((moved >> 32u) & 0xffffu);
                    ++relocated;
                    i += 2;
                } else {
                    ++i;
                }
            }
        };
        for (std::uint64_t k = 1; k < copies; ++k) {
            auto& copy = copyList[k - 1u];
            copy.offset = k * copyStride;
            for (const auto& region : request.request.context.memory) {
                const bool code = region.guestAddress >= codeBegin && region.guestAddress < codeEnd;
                copy.bytes.emplace_back(region.bytes.begin(), region.bytes.end());
                auto& bytes = copy.bytes.back();
                if (!code && bytes.size() >= 8u && region.guestAddress % 4u == 0u) relocate(reinterpret_cast<std::uint32_t*>(bytes.data()), bytes.size() / 4u, copy.offset);
            }
            for (std::size_t r = 0; r < copy.bytes.size(); ++r) {
                const auto& region = request.request.context.memory[r];
                const bool code = region.guestAddress >= codeBegin && region.guestAddress < codeEnd;
                copy.regions.push_back({code ? region.guestAddress : region.guestAddress + copy.offset, std::span<const std::byte>(copy.bytes[r].data(), copy.bytes[r].size())});
            }
            copy.userData.assign(request.request.context.userData.begin(), request.request.context.userData.end());
            relocate(copy.userData.data(), copy.userData.size(), copy.offset);
            auto copied = request.request;
            copied.context.memory = std::span<const ShaderRecompiler::MemoryRegion>(copy.regions);
            copied.context.userData = std::span<const std::uint32_t>(copy.userData);
            copy.shader = std::make_unique<ShaderRecompiler::RecompileResult>(ShaderRecompiler::Recompile(copied));
            for (const auto& [begin, end] : owned) space.Want(begin + copy.offset, end - begin);
        }
        if (copies > 1u) std::printf("request bench: %llu relocated copies, %llu guest words relocated per copy\n", static_cast<unsigned long long>(copies - 1u), static_cast<unsigned long long>(relocated / (copies - 1u)));
        space.Map();
        const auto fill = EnvText("BENCH_FILL", "mix");
        const auto seed = Env("BENCH_SEED", 1u);
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            if (auto* data = space.At(buffers[i].base, buffers[i].bytes)) Fill(data, buffers[i].bytes & ~std::uint64_t{3}, fill, seed + i);
            std::printf("  buffer: base 0x%llx bytes %llu stride %u format %u%s\n", static_cast<unsigned long long>(buffers[i].base), static_cast<unsigned long long>(buffers[i].bytes), buffers[i].stride, buffers[i].format, buffers[i].written ? " written" : "");
        }
        for (std::size_t i = 0; i < images.size(); ++i) {
            if (auto* data = space.At(images[i].first, images[i].second)) Fill(data, images[i].second & ~std::uint64_t{3}, fill == "zero" ? fill : std::string("float"), seed + 1000u + i);
        }
        // Captured bytes win over synthetic ones.
        for (const auto& region : request.request.context.memory) {
            if (auto* data = space.At(region.guestAddress, region.bytes.size())) std::memcpy(data, region.bytes.data(), region.bytes.size());
        }
        {
            const auto original = buffers.size();
            for (const auto& copy : copyList) {
                for (std::size_t i = 0; i < original; ++i) {
                    auto moved = buffers[i];
                    moved.base += copy.offset;
                    const auto* from = space.At(buffers[i].base, buffers[i].bytes);
                    auto* to = space.At(moved.base, moved.bytes);
                    if (from != nullptr && to != nullptr) std::memcpy(to, from, moved.bytes);
                    buffers.push_back(moved);
                }
                for (const auto& region : copy.regions) {
                    if (auto* data = space.At(region.guestAddress, region.bytes.size())) std::memcpy(data, region.bytes.data(), region.bytes.size());
                }
            }
        }
        std::vector<std::unique_ptr<ExtraRange>> extras;
        const auto ranges = Env("BENCH_RANGES", 1200u);
        for (std::uint64_t range = 0; range < ranges; ++range) extras.push_back(std::make_unique<ExtraRange>());
        std::printf("request bench: %zu guest blocks, %llu MiB mapped, %zu buffers, %zu images, %llu extra ranges, fill %s\n", space.Blocks(), static_cast<unsigned long long>(space.Mapped() >> 20u), buffers.size(), images.size(), static_cast<unsigned long long>(ranges), fill.c_str());

        // Initial contents of every written buffer, restored before each repeat.
        std::vector<std::vector<std::uint8_t>> initial;
        for (const auto& buffer : buffers) {
            const auto* data = buffer.written ? space.At(buffer.base, buffer.bytes) : nullptr;
            initial.emplace_back(data != nullptr ? std::vector<std::uint8_t>(data, data + buffer.bytes) : std::vector<std::uint8_t>{});
        }
        std::vector<std::vector<std::uint8_t>> initialImages;
        for (std::size_t i = 0; i < images.size(); ++i) {
            const auto* data = imageWritten[i] ? space.At(images[i].first, images[i].second) : nullptr;
            initialImages.emplace_back(data != nullptr ? std::vector<std::uint8_t>(data, data + images[i].second) : std::vector<std::uint8_t>{});
        }
        const auto restore = [&] {
            for (std::size_t i = 0; i < buffers.size(); ++i) {
                if (initial[i].empty()) continue;
                if (auto* data = space.At(buffers[i].base, buffers[i].bytes)) std::memcpy(data, initial[i].data(), initial[i].size());
            }
            for (std::size_t i = 0; i < images.size(); ++i) {
                if (initialImages[i].empty()) continue;
                if (auto* data = space.At(images[i].first, images[i].second)) std::memcpy(data, initialImages[i].data(), initialImages[i].size());
            }
        };
        std::uint32_t groups[3] = {1, 1, 1};
        {
            std::stringstream list(EnvText("BENCH_GROUPS", "1,1,1"));
            std::string item;
            for (int axis = 0; axis < 3 && std::getline(list, item, ','); ++axis) groups[axis] = static_cast<std::uint32_t>(std::strtoul(item.c_str(), nullptr, 0));
        }
        const auto repeats = Env("BENCH_REPEATS", 5u);
        // BENCH_BATCH=k: k dispatches back to back before the wait (with APS5_BENCH_NO_DISPATCH_BARRIERS=1 they may overlap).
        const auto batchCount = std::max<std::uint64_t>(1u, Env("BENCH_BATCH", 1u));
        const auto hashOutputs = [&] {
            std::uint64_t hash = 1469598103934665603ull;
            const auto mix = [&](const std::uint8_t* data, std::uint64_t bytes) {
                const auto* words = reinterpret_cast<const std::uint64_t*>(data);
                for (std::uint64_t word = 0; word < bytes / 8u; ++word) hash = (hash ^ words[word]) * 1099511628211ull;
                for (std::uint64_t byte = bytes & ~std::uint64_t{7}; byte < bytes; ++byte) hash = (hash ^ data[byte]) * 1099511628211ull;
            };
            for (const auto& buffer : buffers) {
                if (!buffer.written) continue;
                if (const auto* data = space.At(buffer.base, buffer.bytes)) mix(data, buffer.bytes);
            }
            for (std::size_t i = 0; i < images.size(); ++i) {
                if (!imageWritten[i]) continue;
                if (const auto* data = space.At(images[i].first, images[i].second)) mix(data, images[i].second);
            }
            return hash;
        };
        // BENCH_OVERLAP=1: the round's dispatches are one unordered run of a guest compute ring (as the packet
        // loop marks dispatches with no packet between them), so the recorder may overlap them.
        const bool overlap = std::getenv("BENCH_OVERLAP") != nullptr;
        const auto dispatchRound = [&] {
            bool first = true;
            const auto one = [&](const ShaderRecompiler::RecompileResult& compiled) {
                if (overlap) AgcDriver::Graphics::Recorder::SetDispatchOrder(0x53, !first);
                first = false;
                device->Dispatch(compiled, groups[0], groups[1], groups[2], {}, request.request.shader.codeAddress);
            };
            for (std::uint64_t batch = 0; batch < batchCount; ++batch) {
                one(shader);
                for (const auto& copy : copyList) one(*copy.shader);
            }
        };
        // BENCH_ASYNC_PROBE=1 (with BENCH_COPIES > 1; APS5_ASYNC_LANE=<ring> on or off): the copies go to the async
        // lane (submitted), then the original dispatch to the main lane, whose result the CPU waits for by its range
        // (Recorder::SyncThrough, the flush hook's wait): with one lane that wait is behind every copy on the queue,
        // with the async lane only behind the original. Prints a PROBE line (median wait).
        const bool asyncProbe = std::getenv("BENCH_ASYNC_PROBE") != nullptr && !copyList.empty();
        std::vector<double> probeTimes;
        const auto probeRound = [&] {
            {
                const AgcDriver::Graphics::Recorder::LaneScope async(1);
                for (const auto& copy : copyList) device->Dispatch(*copy.shader, groups[0], groups[1], groups[2], {}, request.request.shader.codeAddress);
                device->SubmitRecorded(false);
            }
            const AgcDriver::Graphics::Recorder::LaneScope main(0);
            device->Dispatch(shader, groups[0], groups[1], groups[2], {}, request.request.shader.codeAddress);
            device->SubmitRecorded(false);
            const BufferUse* target = nullptr;
            for (const auto& buffer : buffers) {
                if (buffer.written) {
                    target = &buffer;
                    break;
                }
            }
            if (target == nullptr) return;
            const auto start = std::chrono::steady_clock::now();
            {
                std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
                if (auto* recorder = AgcDriver::Graphics::Recorder::Active()) recorder->SyncThrough(target->base, static_cast<std::size_t>(target->bytes));
            }
            probeTimes.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        };
        std::vector<double> times;
        std::uint64_t firstHash = 0;
        bool stable = true;
        // BENCH_WARMUP=s (default 4): dispatch for s seconds before timing, so a laptop GPU that idled through the
        // pipeline compile reaches its clocks (first measurements after a 1-2 min compile ran ~3x slow).
        {
            restore();
            const auto warm = std::chrono::seconds(Env("BENCH_WARMUP", 4u));
            // The first dispatch creates the pipeline (up to ~2 min cold): the warm-up clock starts after it.
            const auto compileStart = std::chrono::steady_clock::now();
            device->Dispatch(shader, groups[0], groups[1], groups[2], {}, request.request.shader.codeAddress);
            device->WaitIdle();
            std::printf("  first dispatch (pipeline create): %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - compileStart).count());
            const auto warmStart = std::chrono::steady_clock::now();
            std::uint64_t warmCount = 0;
            do {
                dispatchRound();
                device->WaitIdle();
                ++warmCount;
            } while (std::chrono::steady_clock::now() - warmStart < warm);
            std::printf("  warm-up: %llu dispatch rounds in %.1f s\n", static_cast<unsigned long long>(warmCount), std::chrono::duration<double>(std::chrono::steady_clock::now() - warmStart).count());
        }
        // BENCH_SECONDS=s: keep repeating until s seconds passed (so an APS5_PROFILE_GPU [gputime] window closes).
        const auto seconds = Env("BENCH_SECONDS", 0u);
        const auto loopStart = std::chrono::steady_clock::now();
        for (std::uint64_t repeat = 0; repeat < repeats || std::chrono::steady_clock::now() - loopStart < std::chrono::seconds(seconds); ++repeat) {
            restore();
            const auto started = std::chrono::steady_clock::now();
            if (asyncProbe) probeRound();
            else dispatchRound();
            device->WaitIdle();
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
            // Hash on the first repeat, then every 16th (images are large); the last one is hashed after the loop.
            if (repeat == 0 || repeat % 16u == 15u) {
                const auto hash = hashOutputs();
                if (repeat == 0) firstHash = hash;
                stable = stable && hash == firstHash;
            }
            std::printf("  repeat %llu: %.3f ms\n", static_cast<unsigned long long>(repeat), times.back());
            std::fflush(stdout);
        }
        {
            const auto hash = hashOutputs();
            stable = stable && hash == firstHash;
        }
        if (const char* dump = std::getenv("BENCH_DUMP"); dump != nullptr && *dump != '\0') {
            std::ofstream file(dump, std::ios::binary);
            for (const auto& buffer : buffers) {
                if (!buffer.written) continue;
                if (const auto* data = space.At(buffer.base, buffer.bytes)) file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(buffer.bytes));
            }
        }
        auto sorted = times;
        std::sort(sorted.begin(), sorted.end());
        std::printf("BENCH batch=%llu copies=%llu request=%s words=%zu groups=%u,%u,%u fill=%s best_ms=%.3f median_ms=%.3f hash=%016llx stable=%d\n", static_cast<unsigned long long>(batchCount), static_cast<unsigned long long>(copies), argv[1], shader.spirv.size(), groups[0], groups[1], groups[2], fill.c_str(), sorted.front(), sorted[sorted.size() / 2], static_cast<unsigned long long>(firstHash), stable ? 1 : 0);
        if (asyncProbe && !probeTimes.empty()) {
            std::sort(probeTimes.begin(), probeTimes.end());
            std::printf("PROBE async_lane=%s main-lane result wait median_ms=%.3f best_ms=%.3f\n", std::getenv("APS5_ASYNC_LANE") != nullptr ? std::getenv("APS5_ASYNC_LANE") : "off", probeTimes[probeTimes.size() / 2], probeTimes.front());
        }
        std::fflush(stdout);
        std::fflush(nullptr);
        TerminateProcess(GetCurrentProcess(), 0);  // no DLL detach: ExitProcess after APS5_PROFILE_GPU hung in teardown
        std::_Exit(0);
    } catch (const std::exception& error) {
        std::cerr << "request bench: " << error.what() << '\n';
        return 1;
    }
}
