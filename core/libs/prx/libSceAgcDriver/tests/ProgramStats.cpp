#include "prx/libSceAgcDriver/Execution/include/Driver/ProgramStats.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace AgcDriver::DriverDetail;

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool Throws(const char* text) {
    try {
        static_cast<void>(ParseProgramList(text));
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void Lists() {
    Check(ParseProgramList(nullptr).empty() && ParseProgramList("").empty(), "an unset list is not empty");
    const auto list = ParseProgramList("0x80001c0d00, 8000394600,0X141D55D00,0x80001c0d00,");
    Check(list.size() == 3 && list[0] == 0x141d55d00ull && list[1] == 0x80001c0d00ull && list[2] == 0x8000394600ull, "the list is not the sorted unique addresses");
    Check(ListedProgram(list, 0x8000394600ull) && !ListedProgram(list, 0x8000394700ull) && !ListedProgram(list, 0), "membership is wrong");
    Check(Throws("0x80001c0d00,zz") && Throws("0x") && Throws("0") && Throws("12345678901234567"), "a malformed entry is accepted");
    Check(ParseStatsInterval(nullptr) == 0 && ParseStatsInterval("10") == 10, "the interval is misread");
    bool threw = false;
    try {
        static_cast<void>(ParseStatsInterval("1x"));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    Check(threw, "a malformed interval is accepted");
}

void Tint() {
    VkPipelineColorBlendAttachmentState opaque{};
    opaque.colorWriteMask = 0xf;
    auto blended = opaque;
    blended.blendEnable = VK_TRUE;
    blended.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blended.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    VkPipelineColorBlendAttachmentState masked{};
    std::vector<VkPipelineColorBlendAttachmentState> blends{opaque, blended, masked, opaque, opaque};
    const std::vector<VkFormat> formats{VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT, VK_FORMAT_UNDEFINED};
    std::array<float, 4> constants{0.25f, 0.5f, 0.75f, 1.0f};
    Check(ApplyBlendTint(blends, constants, formats) == 2, "the wrong attachments are tinted");
    for (const auto index : {0, 1}) {
        const auto& blend = blends[index];
        Check(blend.blendEnable == VK_TRUE && blend.srcColorBlendFactor == VK_BLEND_FACTOR_CONSTANT_COLOR && blend.dstColorBlendFactor == VK_BLEND_FACTOR_ZERO && blend.colorBlendOp == VK_BLEND_OP_ADD
            && blend.srcAlphaBlendFactor == VK_BLEND_FACTOR_CONSTANT_ALPHA && blend.dstAlphaBlendFactor == VK_BLEND_FACTOR_ZERO && blend.colorWriteMask == 0xf, "a tinted attachment does not write the constant");
    }
    Check(blends[2].blendEnable == VK_FALSE && blends[2].colorWriteMask == 0, "a masked attachment is touched");
    Check(blends[3].blendEnable == VK_FALSE && blends[4].blendEnable == VK_FALSE, "an integer or unknown attachment is blended");
    Check(constants == std::array<float, 4>{1.0f, 0.0f, 1.0f, 1.0f}, "the constant is not magenta");
    std::array<float, 4> untouched{0.25f, 0.5f, 0.75f, 1.0f};
    std::vector<VkPipelineColorBlendAttachmentState> none{masked};
    Check(ApplyBlendTint(none, untouched, formats) == 0 && untouched[0] == 0.25f, "a draw with nothing to tint changes its constants");
}

void Stats() {
    ProgramStats off(0);
    off.NoteDraw(1, 2, {}, std::nullopt, false, false);
    Check(off.Present().empty(), "disabled stats report");
    ProgramStats stats(2);
    const StatsTarget albedo{0x5000000000ull, 1920, 1080, VK_FORMAT_R8G8B8A8_SRGB, 1};
    const StatsTarget normals{0x5001000000ull, 1920, 1080, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 1};
    const StatsTarget depth{0x5002000000ull, 1920, 1080, VK_FORMAT_D32_SFLOAT, 1};
    const StatsTarget gbuffer[]{albedo, normals};
    stats.NoteDraw(0xa00, 0xb00, gbuffer, depth, false, false);
    Check(stats.Present().empty(), "an odd present reports with an interval of 2");
    for (int i = 0; i < 3; ++i) stats.NoteDraw(0xa00, 0xb00, gbuffer, depth, false, false);
    stats.NoteDraw(0xa10, 0xb00, gbuffer, depth, true, false);
    stats.NoteDraw(0xc00, 0xd00, {}, depth, false, false);
    const StatsImage decal{0x6000000000ull, 512, 512, 10};
    const StatsImage other{0x6100000000ull, 64, 64, 10};
    const StatsImage writes[]{decal, other};
    stats.NoteDispatch(0xe00, writes, false);
    stats.NoteDispatch(0xe00, writes, false);
    stats.NoteDispatch(0xf00, {}, true);
    const std::uint64_t reads[]{0x6000000000ull};
    stats.NoteDrawSampled(0xa00, reads);
    const auto lines = stats.Present();
    std::string all;
    for (const auto& line : lines) all += line + "\n";
    Check(lines.size() == 5, "the report has the wrong number of lines");
    Check(lines[0] == "[progstats] present 2: 5 draws into 2 attachment groups, 3 dispatches of 2 compute programs", "the summary line changed");
    Check(lines[1] == "[progstats] present 2 color 0x5000000000:1920x1080:f43 0x5001000000:1920x1080:f64 depth 0x5002000000:1920x1080:f126 | 4 draws: ps 0xa00 (vs 0xb00) x3, ps 0xa10 (vs 0xb00) x1 skipped 1", "the largest group's line changed");
    Check(lines[2] == "[progstats] present 2 color none depth 0x5002000000:1920x1080:f126 | 1 draws: ps 0xc00 (vs 0xd00) x1", "the depth-only group's line changed");
    Check(lines[3] == "[progstats] present 2 compute 0xe00 x2 writes 0x6000000000:512x512:f10 (sampled by ps 0xa00) 0x6100000000:64x64:f10", "the compute writer's line changed");
    Check(lines[4] == "[progstats] present 2 compute without image writes seen: 0xf00 x1 skipped 1", "the plain compute line changed");
    Check(stats.Present().empty(), "an odd present reports");
    const auto empty = stats.Present();
    Check(empty.size() == 1 && empty[0] == "[progstats] present 4: 0 draws into 0 attachment groups, 0 dispatches of 0 compute programs", "a frame does not start empty");
}

}

int main() {
    try {
        Lists();
        Tint();
        Stats();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
