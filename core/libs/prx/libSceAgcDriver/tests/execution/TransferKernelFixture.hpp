#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_TRANSFERKERNELFIXTURE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_TRANSFERKERNELFIXTURE_HPP

#include "NewPathFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

namespace TransferKernels {

using NewPath::Program;

constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t Groups = 4;
constexpr std::uint32_t Elements = Threads * Groups;
constexpr std::uint32_t Word3 = 0x0004dfacu;
constexpr std::uint32_t StoreDwordX4 = 0x1eu;
constexpr std::uint32_t LoadDwordX4 = 0x0eu;
constexpr std::uint32_t WaitVector = 0xbf8c3f70u;

inline int failures = 0;

inline void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

inline std::array<std::uint32_t, 4> Descriptor(std::uint64_t address, std::uint32_t records) {
    return {NewPath::Low(address), (NewPath::High(address) & 0xffffu) | (16u << 16u), records, Word3};
}

inline std::array<std::uint32_t, 2> Indexed(std::uint32_t opcode, std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t resource) {
    return {0xe0000000u | (opcode << 18u) | 0x2000u, (NewPath::Isa::ZeroOffset << 24u) | ((resource / 4u) << 16u) | (vdata << 8u) | vaddr};
}

inline void GlobalIndex(Program& program, std::uint32_t groupSgpr, std::uint32_t target) {
    using namespace NewPath::Isa;
    program << Vop1(VMovB32, target, groupSgpr);
    program << Vop2(VLshlRevB32, target, InlineInteger(6), target);
    program << Vop2(VAddNcU32, target, Vgpr(target), 0u);
}

inline void Dispatch(const Program& program, std::span<const std::uint32_t> userData) {
    NewPath::Commands commands;
    commands.Dispatch(program, userData, Threads, Groups);
    commands.Submit(0);
    NewPath::WaitIdle();
}

}

#endif
