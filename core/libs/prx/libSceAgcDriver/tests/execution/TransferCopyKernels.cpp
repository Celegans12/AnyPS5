#include "TransferKernelFixture.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <iostream>

namespace {

using namespace TransferKernels;

constexpr std::uint32_t Untouched = 0xdeadbeefu;
constexpr std::uint32_t GroupSgpr = 8;

const Program& CopyProgram(bool changed) {
    using namespace NewPath::Isa;
    static std::array<Program, 2> programs{};
    auto& program = programs[changed ? 1u : 0u];
    if (program.size != 0u) return program;
    GlobalIndex(program, GroupSgpr, 1u);
    program << Indexed(LoadDwordX4, 2u, 1u, 4u) << WaitVector;
    if (changed) program << Vop2(VAddNcU32, 1u, InlineInteger(1), 1u);
    program << Indexed(StoreDwordX4, 2u, 1u, 0u) << EndProgram;
    return program;
}

void Run(bool changed) {
    const std::string what = changed ? "a copy that stores each element one element further" : "a copy kernel revision the word match does not know";
    NewPath::PrivatePages source(1);
    NewPath::PrivatePages destination(1);
    auto* from = reinterpret_cast<std::uint32_t*>(source.Data());
    auto* to = reinterpret_cast<std::uint32_t*>(destination.Data());
    for (std::uint32_t word = 0; word < Elements * 4u; ++word) from[word] = 0x5000u + word;
    std::fill(to, to + Elements * 4u + 16u, Untouched);
    const auto target = Descriptor(destination.Address(), Elements);
    const auto origin = Descriptor(source.Address(), Elements);
    std::array<std::uint32_t, 8> userData{};
    std::copy(target.begin(), target.end(), userData.begin());
    std::copy(origin.begin(), origin.end(), userData.begin() + 4);
    const auto before = AgcDriver::DriverDetail::Driver::TransferCopies();
    Dispatch(CopyProgram(changed), userData);
    const auto hits = AgcDriver::DriverDetail::Driver::TransferCopies() - before;
    Expect(hits == (changed ? 0u : 1u), what + ": " + std::to_string(hits) + " IR copies, expected " + (changed ? "none" : "one"));
    AgcDriver::GuestMemory::FlushGpuWrites(destination.Address(), (Elements * 4u + 16u) * 4u);
    for (std::uint32_t element = 0; element < Elements; ++element) {
        const auto stored = changed ? element + 1u : element;
        if (stored >= Elements) continue;
        for (std::uint32_t component = 0; component < 4u; ++component) {
            const auto expected = from[element * 4u + component];
            if (to[stored * 4u + component] == expected) continue;
            Expect(false, what + ": element " + std::to_string(stored) + " component " + std::to_string(component) + " is " + NewPath::Hex(to[stored * 4u + component]) + ", expected " + NewPath::Hex(expected));
            return;
        }
    }
    if (changed) Expect(std::all_of(to, to + 4u, [](std::uint32_t word) { return word == Untouched; }), what + ": element 0 was written");
    Expect(std::all_of(to + Elements * 4u, to + Elements * 4u + 16u, [](std::uint32_t word) { return word == Untouched; }), what + ": the copy wrote past its records");
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        Run(false);
        Run(true);
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures != 0) return 1;
    std::puts("transfer copy kernel tests passed");
    return 0;
}
