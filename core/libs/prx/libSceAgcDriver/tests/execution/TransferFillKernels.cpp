#include "TransferKernelFixture.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>

namespace {

using namespace TransferKernels;

constexpr std::uint32_t Untouched = 0xdeadbeefu;
constexpr std::array<std::uint32_t, 4> Pattern{0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
constexpr std::uint32_t GroupSgpr = 8;

const Program& FillProgram(bool changed) {
    using namespace NewPath::Isa;
    static std::array<Program, 2> programs{};
    auto& program = programs[changed ? 1u : 0u];
    if (program.size != 0u) return program;
    GlobalIndex(program, GroupSgpr, 1u);
    for (std::uint32_t component = 0; component < 4u; ++component) program << Vop1(VMovB32, 2u + component, 4u + component);
    if (changed) program << Vop2(VAddNcU32, 2u, InlineInteger(1), 2u);
    program << Indexed(StoreDwordX4, 2u, 1u, 0u) << EndProgram;
    return program;
}

struct Revision {
    const char* name;
    std::vector<std::uint32_t> code;
    std::uint32_t userData;
    std::uint32_t valueRegisters;
    std::uint32_t valueMemory;
};

std::vector<Revision> SdkRevisions() {
    return {
        {"the zero fill", {0xd7460004u, 0x04010c04u, 0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e060280u, 0xe01c2000u, 0x80000004u, 0xbf810000u}, 4u, 0x0u, 0x0u},
        {"the user data fill", {0xd7460004u, 0x04010c08u, 0x7e000204u, 0x7e020205u, 0x7e040206u, 0x7e060207u, 0xe01c2000u, 0x80000004u, 0xbf810000u}, 8u, 0xfu, 0x0u},
        {"the buffer pattern fill", {0xd7460004u, 0x04010c08u, 0xf4280302u, 0xfa000000u, 0xbf8cc07fu, 0x7e00020cu, 0x7e02020du, 0x7e04020eu, 0x7e06020fu, 0xe01c2000u, 0x80000004u, 0xbf810000u}, 8u, 0x0u, 0xfu},
    };
}

std::optional<ShaderRecompiler::TransferKernel> Recognized(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, std::uint32_t userDataWords) {
    alignas(256) static std::array<std::uint32_t, 64> program{};
    std::copy(code.begin(), code.end(), program.begin());
    const std::span<const std::uint32_t> words(program.data(), code.size());
    alignas(16) static std::array<std::uint32_t, 4> values{};
    const auto valuesAddress = reinterpret_cast<std::uintptr_t>(values.data());
    std::vector<std::uint32_t> userData(userDataWords, 0u);
    const auto target = Descriptor(0x10000000u, Elements);
    std::copy(target.begin(), target.end(), userData.begin());
    if (userDataWords == 8u) {
        const std::array<std::uint32_t, 4> pattern{static_cast<std::uint32_t>(valuesAddress), static_cast<std::uint32_t>(valuesAddress >> 32u) & 0xffffu, 16u, Word3};
        std::copy(pattern.begin(), pattern.end(), userData.begin() + 4);
    }
    const std::array<ShaderRecompiler::MemoryRegion, 2> memory{{{reinterpret_cast<std::uintptr_t>(program.data()), std::as_bytes(words)}, {valuesAddress, std::as_bytes(std::span(values))}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{64u, 1u, 1u}, 0u, {true, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(program.data()), words, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request).transferKernel;
}

void RecognizesSdkRevisions(AgcDriver::VulkanDevice& device) {
    for (const auto& revision : SdkRevisions()) {
        const auto transfer = Recognized(device, revision.code, revision.userData);
        Expect(transfer.has_value() && transfer->kind == ShaderRecompiler::TransferKernel::Kind::Fill && transfer->components == 4u && transfer->destination == 0u && transfer->valueRegisters == revision.valueRegisters && transfer->valueMemory == revision.valueMemory, std::string(revision.name) + " is not recognized as a fill of its user data");
        auto changed = revision.code;
        const auto store = std::find_if(changed.begin(), changed.end(), [](std::uint32_t word) { return (word & 0xfffffe00u) == 0x7e000200u; });
        Expect(store != changed.end(), std::string(revision.name) + " has no first component move to change");
        if (store == changed.end()) continue;
        *store = 0x7e000304u;
        Expect(!Recognized(device, changed, revision.userData).has_value(), std::string(revision.name) + " with its first component replaced by the lane index is still recognized");
    }
}

void Run(bool changed) {
    const std::string what = changed ? "a fill whose first component is the user data plus one" : "a fill kernel revision the word match does not know";
    NewPath::PrivatePages memory(1);
    auto* words = reinterpret_cast<std::uint32_t*>(memory.Data());
    std::fill(words, words + Elements * 4u + 16u, Untouched);
    const auto destination = Descriptor(memory.Address(), Elements);
    std::array<std::uint32_t, 8> userData{};
    std::copy(destination.begin(), destination.end(), userData.begin());
    std::copy(Pattern.begin(), Pattern.end(), userData.begin() + 4);
    const auto before = AgcDriver::DriverDetail::Driver::TransferFills();
    Dispatch(FillProgram(changed), userData);
    const auto hits = AgcDriver::DriverDetail::Driver::TransferFills() - before;
    Expect(hits == (changed ? 0u : 1u), what + ": " + std::to_string(hits) + " IR fills, expected " + (changed ? "none" : "one"));
    AgcDriver::GuestMemory::FlushGpuWrites(memory.Address(), (Elements * 4u + 16u) * 4u);
    for (std::uint32_t element = 0; element < Elements; ++element) {
        for (std::uint32_t component = 0; component < 4u; ++component) {
            const auto expected = Pattern[component] + (changed && component == 0u ? 1u : 0u);
            if (words[element * 4u + component] == expected) continue;
            Expect(false, what + ": element " + std::to_string(element) + " component " + std::to_string(component) + " is " + NewPath::Hex(words[element * 4u + component]) + ", expected " + NewPath::Hex(expected));
            return;
        }
    }
    Expect(std::all_of(words + Elements * 4u, words + Elements * 4u + 16u, [](std::uint32_t word) { return word == Untouched; }), what + ": the fill wrote past its records");
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
            RecognizesSdkRevisions(*device);
        }
        Run(false);
        Run(true);
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures != 0) return 1;
    std::puts("transfer fill kernel tests passed");
    return 0;
}
