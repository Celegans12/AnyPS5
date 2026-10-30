#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ProgramScatters: whether a store can take its address from memory (the exact-replay split of
// APS5_EXACT_SERIAL_ALLOCATING applies to such programs).

namespace {

constexpr std::uint32_t EndProgram = 0xbf810000u;
constexpr std::uint32_t LoadV1 = 0xe0300000u, LoadV1b = 0x80000100u;       // buffer_load_dword v1, off, s[0:3], 0
constexpr std::uint32_t LoadV3 = 0xe0300000u, LoadV3b = 0x80000300u;       // buffer_load_dword v3, off, s[0:3], 0
constexpr std::uint32_t LoadV0 = 0xe0300000u, LoadV0b = 0x80000000u;       // buffer_load_dword v0, off, s[0:3], 0
constexpr std::uint32_t WaitVm = 0xbf8c3f70u;                              // s_waitcnt vmcnt(0)
constexpr std::uint32_t ShiftV1 = 0x34020282u;                             // v_lshlrev_b32 v1, 2, v1
constexpr std::uint32_t MoveV2S8 = 0x7e040208u;                            // v_mov_b32 v2, s8
constexpr std::uint32_t MoveV1S8 = 0x7e020208u;                            // v_mov_b32 v1, s8
constexpr std::uint32_t MoveV0S8 = 0x7e000208u;                            // v_mov_b32 v0, s8
constexpr std::uint32_t StoreV2V1 = 0xe0701000u, StoreV2V1b = 0x80010201u; // buffer_store_dword v2, v1, s[4:7], 0 offen
constexpr std::uint32_t StoreOff = 0xe0700000u, StoreOffb = 0x80010200u;   // buffer_store_dword v2, off, s[4:7], 0
constexpr std::uint32_t SkipOne = 0xbf840001u;                             // s_cbranch_scc0 (over one word)
constexpr std::uint32_t CompareV0 = 0x7d880084u;                           // v_cmp_gt_u32 vcc_lo, 4, v0
constexpr std::uint32_t SelectV1 = 0xd5010001u, SelectV1b = 0x01a90280u;   // v_cndmask_b32_e64 v1, 0, 1, vcc_lo
constexpr std::uint32_t ImageStore = 0xf0200108u, ImageStoreb = 0x00020200u; // image_store v2, v[0:1], s[8:15] dmask:0x1 2D
constexpr std::uint32_t ImageLoadV0 = 0xf0000108u, ImageLoadV0b = 0x00020000u; // image_load v0, v[0:1], s[8:15] dmask:0x1 2D

bool scatters(std::initializer_list<std::uint32_t> words) {
    AgcDriver::DriverDetail::ShaderSnapshot snapshot{};
    snapshot.code.assign(words.begin(), words.end());
    snapshot.code.push_back(EndProgram);
    return AgcDriver::DriverDetail::ProgramScatters(snapshot, 0);
}

void expect(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("program scatters: " + what);
}

}

int main() {
    try {
        expect(scatters({LoadV1, LoadV1b, WaitVm, ShiftV1, MoveV2S8, StoreV2V1, StoreV2V1b}), "a buffer store at a loaded offset does not scatter");
        expect(!scatters({LoadV3, LoadV3b, WaitVm, MoveV1S8, ShiftV1, StoreV2V1, StoreV2V1b}), "a store at the workgroup id scatters");
        expect(!scatters({LoadV1, LoadV1b, WaitVm, MoveV1S8, ShiftV1, StoreV2V1, StoreV2V1b}), "a loaded offset overwritten before the store scatters");
        expect(scatters({LoadV1, LoadV1b, WaitVm, SkipOne, MoveV1S8, ShiftV1, StoreV2V1, StoreV2V1b}), "a loaded offset kept on one branch path does not scatter");
        expect(!scatters({LoadV1, LoadV1b, WaitVm, StoreOff, StoreOffb}), "a store without an address register scatters");
        expect(scatters({LoadV0, LoadV0b, WaitVm, CompareV0, SelectV1, SelectV1b, MoveV0S8, ImageStore, ImageStoreb}), "an image store at a coordinate selected by a loaded value does not scatter");
        expect(!scatters({MoveV0S8, ImageLoadV0, ImageLoadV0b, WaitVm, MoveV0S8, ImageStore, ImageStoreb}), "an image store at the workgroup id after an image load into its register scatters");
        expect(scatters({MoveV0S8, ImageLoadV0, ImageLoadV0b, WaitVm, ImageStore, ImageStoreb}), "an image store at a coordinate read from an image does not scatter");
        AgcDriver::DriverDetail::ShaderSnapshot cached{};
        cached.code = {MoveV2S8, LoadV1, LoadV1b, WaitVm, StoreV2V1, StoreV2V1b, EndProgram};
        expect(AgcDriver::DriverDetail::ProgramScatters(cached, 0) && AgcDriver::DriverDetail::ProgramScatters(cached, 0), "the cached answer differs");
        expect(!AgcDriver::DriverDetail::ProgramScatters(cached, 4), "the program from its store (no load before it) scatters");
        AgcDriver::DriverDetail::ShaderSnapshot endless{};
        endless.code = {0x7e000280u};
        expect(!AgcDriver::DriverDetail::ProgramScatters(endless, 0), "code that never ends is classified as scattering");
        LibcRunShutdown_nid_postfix();
        std::puts("program scatters tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
