#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t EndProgram = 0xbf810000u;

bool allocates(std::initializer_list<std::uint32_t> words) {
    AgcDriver::DriverDetail::ShaderSnapshot snapshot{};
    snapshot.code.assign(words.begin(), words.end());
    snapshot.code.push_back(EndProgram);
    return AgcDriver::DriverDetail::ProgramAllocates(snapshot, 0);
}

void expect(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("program allocates: " + what);
}

}

int main() {
    try {
        expect(allocates({0xd8fa0004u, 0x02000000u}), "ds_append offset:4 gds does not allocate");
        expect(allocates({0xd8f40004u, 0x02000000u}), "an LDS ds_consume does not allocate");
        expect(allocates({0xd8fe0304u, 0x00000001u}), "ds_ordered_count gds does not allocate");
        expect(allocates({0xd8660000u, 0x00000001u}), "ds_gws_init does not allocate");
        expect(allocates({0xd8020000u, 0x00000100u}), "a GDS ds_add_u32 does not allocate");
        expect(allocates({0xe0c84000u, 0x80020100u}), "buffer_atomic_add glc does not allocate");
        expect(allocates({0xdcc98000u, 0x017d0402u}), "global_atomic_add glc does not allocate");
        expect(allocates({0xf0442100u, 0x00020102u}), "image_atomic_add glc does not allocate");
        expect(!allocates({0xd8000000u, 0x00000100u}), "an LDS ds_add_u32 allocates");
        expect(!allocates({0xd8800000u, 0x00000201u}), "an LDS ds_add_rtn_u32 allocates");
        expect(!allocates({0xe0c80000u, 0x80020100u}), "buffer_atomic_add without glc allocates");
        expect(!allocates({0xdcc88000u, 0x007d0402u}), "global_atomic_add without glc allocates");
        expect(!allocates({0xf0440100u, 0x00020102u}), "image_atomic_add without glc allocates");
        expect(!allocates({0xe0700000u, 0x80020100u}), "a buffer store allocates");
        AgcDriver::DriverDetail::ShaderSnapshot endless{};
        endless.code = {0x7e000280u};
        expect(AgcDriver::DriverDetail::ProgramAllocates(endless, 0), "code that never ends decodes as not allocating");
        AgcDriver::DriverDetail::ShaderSnapshot append{};
        append.code = {0xe0700000u, 0x80020100u, 0xd8fa0004u, 0x02000000u, EndProgram};
        expect(AgcDriver::DriverDetail::ProgramAllocates(append, 0) && AgcDriver::DriverDetail::ProgramAllocates(append, 0), "the cached answer differs");
        expect(AgcDriver::DriverDetail::ProgramAllocates(append, 2), "the program from the append does not allocate");
        expect(!AgcDriver::DriverDetail::ProgramAllocates(append, 4), "the program from its end allocates");
        LibcRunShutdown_nid_postfix();
        std::puts("program allocates tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
