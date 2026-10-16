#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::Graphics::Recorder;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

alignas(256) std::array<std::uint32_t, 15> SpinCodeA{0xf4080200u, 0xfa000000u, 0xbe840381u, 0x9304ff04u, 0x0019660du, 0x8004ff04u, 0x3c6ef35fu, 0x80828102u, 0xbf078002u, 0xbf85fff9u, 0xbf8cc07fu, 0x7e000204u, 0xe0700000u, 0x80020000u, 0xbf810000u};
alignas(256) std::array<std::uint32_t, 15> SpinCodeB{0xf4080200u, 0xfa000000u, 0xbe840381u, 0x9304ff04u, 0x0019660du, 0x8004ff04u, 0x3c6ef35fu, 0x80828102u, 0xbf078002u, 0xbf85fff9u, 0xbf8cc07fu, 0x7e000204u, 0xe0700000u, 0x80020000u, 0xbf810000u};
alignas(256) std::array<std::uint32_t, 11> CopyCode{0xf4080200u, 0xfa000000u, 0xf4080300u, 0xfa000010u, 0xbf8cc07fu, 0xe0300000u, 0x80020000u, 0xbf8c3f70u, 0xe0700000u, 0x80030000u, 0xbf810000u};

constexpr std::uint32_t LongSpin = 1u << 22u;
constexpr std::uint32_t ShortSpin = 1u << 10u;
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t Sentinel = 0xdeadbeefu;

struct Slot {
    alignas(256) std::array<std::uint32_t, 4> first{};
    alignas(256) std::array<std::uint32_t, 4> second{};
    alignas(256) std::array<std::uint32_t, 4> copied{};
    alignas(256) std::array<std::uint32_t, 4> firstDescriptor{};
    alignas(256) std::array<std::uint32_t, 4> secondDescriptor{};
    alignas(256) std::array<std::uint32_t, 8> copyDescriptors{};
    alignas(256) volatile std::uint32_t done = 0;
};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "batched barriers: cannot allocate the guest block");
        std::memset(block, 0, BlockBytes);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    Slot& At(std::size_t index) { return *reinterpret_cast<Slot*>(block + index * 8192); }

private:
    std::uint8_t* block = nullptr;
};

std::uint32_t Low(const volatile void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint32_t High(const volatile void* pointer) {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)) >> 32u);
}

std::array<std::uint32_t, 4> Descriptor(const volatile void* buffer, std::uint32_t bytes) {
    return {Low(buffer), High(buffer) & 0xffffu, bytes, 0x31016facu};
}

void SetShaderRegister(std::vector<std::uint32_t>& words, std::uint32_t offset, std::uint32_t value) {
    words.insert(words.end(), {0xc0017600u, offset, value});
}

void Dispatch(std::vector<std::uint32_t>& words, const void* code, std::initializer_list<std::uint32_t> userData) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(code));
    SetShaderRegister(words, 0x20c, static_cast<std::uint32_t>(address >> 8u));
    SetShaderRegister(words, 0x20d, static_cast<std::uint32_t>(address >> 40u) & 0xffu);
    for (const std::uint32_t offset : {0x207u, 0x208u, 0x209u}) SetShaderRegister(words, offset, 1);
    SetShaderRegister(words, 0x212, 0);
    SetShaderRegister(words, 0x213, static_cast<std::uint32_t>(userData.size()) << 1u);
    std::uint32_t offset = 0x240;
    for (const auto value : userData) SetShaderRegister(words, offset++, value);
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x8041u});
}

void Spin(std::vector<std::uint32_t>& words, const std::array<std::uint32_t, 15>& code, std::array<std::uint32_t, 4>& descriptor, std::array<std::uint32_t, 4>& output, std::uint32_t iterations) {
    descriptor = Descriptor(output.data(), sizeof(output));
    Dispatch(words, code.data(), {Low(descriptor.data()), High(descriptor.data()), iterations});
}

void Copy(std::vector<std::uint32_t>& words, Slot& slot, const std::array<std::uint32_t, 4>& source) {
    const auto in = Descriptor(source.data(), sizeof(source));
    const auto out = Descriptor(slot.copied.data(), sizeof(slot.copied));
    std::memcpy(slot.copyDescriptors.data(), in.data(), sizeof(in));
    std::memcpy(slot.copyDescriptors.data() + 4, out.data(), sizeof(out));
    Dispatch(words, CopyCode.data(), {Low(slot.copyDescriptors.data()), High(slot.copyDescriptors.data())});
}

void WriteData(std::vector<std::uint32_t>& words, volatile std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0033700u, 0x00100200u, Low(target), High(target), value});
}

void SubmitAndFinish(std::vector<std::uint32_t>& words, Slot& slot) {
    slot.done = 0;
    WriteData(words, &slot.done, 1);
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    Require(sceAgcDriverSubmitAcb(0x20, &packet) == 0, "compute submit failed");
    const auto start = std::chrono::steady_clock::now();
    while (slot.done == 0) {
        Require(std::chrono::steady_clock::now() - start < std::chrono::seconds(30), "the submission did not finish");
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    AgcDriverWaitIdle_nid_postfix();
}

std::uint32_t SpinResult(std::uint32_t iterations) {
    std::uint32_t value = 1;
    for (std::uint32_t i = 0; i < iterations; ++i) value = value * 0x19660du + 0x3c6ef35fu;
    return value;
}

void Report(const char* name, const Recorder::BatchedBarrierStatistics& before, const Recorder::BatchedBarrierStatistics& after) {
    std::cout << name << ": held " << after.held - before.held << ", recorded " << after.recorded - before.recorded << ", kept across a dispatch " << after.keptAcross - before.keptAcross << "\n";
}

void DependentDispatches(GuestBlock& block) {
    auto& slot = block.At(0);
    slot.first.fill(Sentinel);
    slot.copied.fill(Sentinel);
    std::vector<std::uint32_t> words;
    Spin(words, SpinCodeA, slot.firstDescriptor, slot.first, LongSpin);
    Copy(words, slot, slot.first);
    const auto before = Recorder::BatchedBarrierCounts();
    SubmitAndFinish(words, slot);
    const auto after = Recorder::BatchedBarrierCounts();
    Report("dependent dispatches", before, after);
    Require(slot.first[0] == SpinResult(LongSpin), "dependent: the spin dispatch stored a wrong value");
    Require(slot.copied[0] == SpinResult(LongSpin), "dependent: the copy ran before the spin dispatch it reads finished");
    Require(after.keptAcross == before.keptAcross, "dependent: the copy did not wait for the barrier of the dispatch it reads");
}

void IndependentDispatches(GuestBlock& block) {
    auto& slot = block.At(1);
    slot.first.fill(Sentinel);
    slot.second.fill(Sentinel);
    slot.copied.fill(Sentinel);
    std::vector<std::uint32_t> words;
    Spin(words, SpinCodeA, slot.firstDescriptor, slot.first, LongSpin);
    Spin(words, SpinCodeB, slot.secondDescriptor, slot.second, ShortSpin);
    Copy(words, slot, slot.first);
    const auto before = Recorder::BatchedBarrierCounts();
    SubmitAndFinish(words, slot);
    const auto after = Recorder::BatchedBarrierCounts();
    Report("independent dispatches", before, after);
    Require(slot.first[0] == SpinResult(LongSpin), "independent: the first spin dispatch stored a wrong value");
    Require(slot.second[0] == SpinResult(ShortSpin), "independent: the second spin dispatch stored a wrong value");
    Require(slot.copied[0] == SpinResult(LongSpin), "independent: the copy ran before the spin dispatch it reads finished");
    if (Recorder::BatchBarriers()) {
        Require(after.keptAcross - before.keptAcross >= 1, "independent: the second dispatch did not share the first one's barrier");
        Require(after.held - before.held > after.recorded - before.recorded, "independent: no barrier was batched");
    } else {
        Require(after.held == before.held && after.recorded == before.recorded && after.keptAcross == before.keptAcross, "independent: barriers were held with APS5_BATCH_BARRIERS off");
    }
}

void WriteAfterWrite(GuestBlock& block) {
    auto& slot = block.At(2);
    slot.first.fill(Sentinel);
    std::vector<std::uint32_t> words;
    Spin(words, SpinCodeA, slot.firstDescriptor, slot.first, LongSpin);
    Spin(words, SpinCodeB, slot.secondDescriptor, slot.first, ShortSpin);
    const auto before = Recorder::BatchedBarrierCounts();
    SubmitAndFinish(words, slot);
    const auto after = Recorder::BatchedBarrierCounts();
    Report("write after write", before, after);
    Require(slot.first[0] == SpinResult(ShortSpin), "write after write: the earlier dispatch's store landed last");
    Require(after.keptAcross == before.keptAcross, "write after write: the second store did not wait for the barrier of the first");
}

}

int main() {
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        device.reset();
        GuestBlock block;
        std::cout << "APS5_BATCH_BARRIERS " << (Recorder::BatchBarriers() ? "on" : "off") << "\n";
        DependentDispatches(block);
        IndependentDispatches(block);
        WriteAfterWrite(block);
        AgcDriverShutdown_nid_postfix();
        std::cout << "batched barriers: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "batched barriers: " << error.what() << "\n";
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (const std::exception& shutdownError) {
            std::cerr << shutdownError.what() << "\n";
        }
        return 1;
    }
}
