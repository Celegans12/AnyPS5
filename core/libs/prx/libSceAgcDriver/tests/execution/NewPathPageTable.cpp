#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DevicePageTable.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandWrites.hpp"
#include <algorithm>
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "VulkanTestDevice.hpp"
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::NewPath::DevicePageTable;
using Range = DevicePageTable::Range;
namespace Abi = ShaderRecompiler::BdaAbi;

constexpr std::uint64_t Page = std::uint64_t{1} << Abi::PageShift;
constexpr std::uint64_t ImportBase = 0x5000000000ull;
constexpr std::uint64_t ImportDevice = 0x7f0000000000ull;
constexpr std::uint64_t ResidentDevice = 0x7e0000000000ull;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

Range MakeRange(std::uint64_t begin, std::uint64_t end, std::uint64_t device, std::uint32_t permissions = Abi::Read | Abi::Write) {
    return {begin, end, device, permissions, 0u};
}

void OverlayTests() {
    const std::vector<Range> imports{MakeRange(ImportBase, ImportBase + 8 * Page, ImportDevice)};
    const auto plain = DevicePageTable::Overlay(imports, {}, ImportBase + Page, ImportBase + 3 * Page);
    Expect(plain.size() == 1 && plain[0].begin == ImportBase + Page && plain[0].end == ImportBase + 3 * Page && plain[0].deviceAddress == ImportDevice + Page, "an import was not clipped to the asked span with its device offset");
    const std::vector<Range> resident{MakeRange(ImportBase + 2 * Page, ImportBase + 3 * Page, ResidentDevice)};
    const auto split = DevicePageTable::Overlay(imports, resident, ImportBase, ImportBase + 8 * Page);
    Expect(split.size() == 3, "a resident page did not split its import into three pieces");
    if (split.size() == 3) {
        Expect(split[0].end == ImportBase + 2 * Page && split[0].deviceAddress == ImportDevice, "the piece before the resident page is wrong");
        Expect(split[1].begin == ImportBase + 2 * Page && split[1].deviceAddress == ResidentDevice, "the resident page does not map to its mirror");
        Expect(split[2].begin == ImportBase + 3 * Page && split[2].deviceAddress == ImportDevice + 3 * Page && split[2].end == ImportBase + 8 * Page, "the piece after the resident page is wrong");
    }
    const std::vector<Range> readOnly{MakeRange(ImportBase, ImportBase + 2 * Page, ImportDevice, Abi::Read)};
    const std::vector<Range> widening{MakeRange(ImportBase + Page, ImportBase + 3 * Page, ResidentDevice)};
    const auto clamped = DevicePageTable::Overlay(readOnly, widening, ImportBase, ImportBase + 3 * Page);
    Expect(clamped.size() == 3, "a resident piece over a read-only import and past it did not give three pieces");
    if (clamped.size() == 3) {
        Expect(clamped[1].begin == ImportBase + Page && clamped[1].deviceAddress == ResidentDevice && clamped[1].permissions == Abi::Read, "a resident piece widened a read-only import's permissions");
        Expect(clamped[2].begin == ImportBase + 2 * Page && clamped[2].deviceAddress == ResidentDevice + Page && clamped[2].permissions == (Abi::Read | Abi::Write), "a resident piece past every import lost its own permissions");
    }
    const auto none = DevicePageTable::Overlay(imports, {}, ImportBase + 16 * Page, ImportBase + 17 * Page);
    Expect(none.empty(), "a span with no import got a piece");
}

std::uint64_t Translate(const std::shared_ptr<const AgcDriver::Graphics::BdaPageTable>& table, std::uint64_t address) {
    if (table == nullptr) return 0;
    const auto found = table->Find(address);
    return found ? found->deviceAddress + (address - found->begin) : 0;
}

void DeviceTests(const AgcDriver::Graphics::Context& context) {
    std::uint64_t version = 1;
    std::vector<Range> imports{MakeRange(ImportBase, ImportBase + 4 * Page, ImportDevice), MakeRange(ImportBase + 8 * Page, ImportBase + 9 * Page, ImportDevice + 8 * Page, Abi::Read)};
    DevicePageTable table([&](const AgcDriver::Graphics::Context& built) {
        AgcDriver::Graphics::HostImportSnapshot snapshot;
        snapshot.device = built.device;
        snapshot.ranges = imports;
        return snapshot;
    }, [&](VkDevice) { return version; });

    Expect(table.Stale(context.device), "a table never built is not stale");
    const auto first = table.Refresh(context);
    Expect(first != nullptr, "the first refresh built no table");
    Expect(Translate(first, ImportBase + 0x1234) == ImportDevice + 0x1234, "an imported byte did not translate to its import");
    Expect(Translate(first, ImportBase + 8 * Page + 4) == ImportDevice + 8 * Page + 4, "the second import did not translate");
    Expect(Translate(first, ImportBase + 5 * Page) == 0, "a page with no import translated");
    Expect(!table.Stale(context.device) && table.Refresh(context) == first, "an unchanged import version rebuilt the table");
    Expect(table.Totals().builds == 1 && table.Totals().unchanged == 1, "build and unchanged counts are wrong");

    imports.push_back(MakeRange(ImportBase + 5 * Page, ImportBase + 6 * Page, ImportDevice + 5 * Page));
    ++version;
    const auto second = table.Refresh(context);
    Expect(second != first, "a new import version did not make a new table version");
    Expect(Translate(second, ImportBase + 5 * Page + 8) == ImportDevice + 5 * Page + 8, "the new import is not in the new version");
    Expect(Translate(first, ImportBase + 5 * Page + 8) == 0, "the old version changed under its holders");

    table.SetResidentPieces([](std::uint64_t begin, std::uint64_t end) {
        std::vector<Range> pieces;
        const auto page = ImportBase + 2 * Page;
        if (begin < page + Page && page < end) pieces.push_back(MakeRange(page, page + Page, ResidentDevice));
        return pieces;
    });
    const std::vector<std::uint64_t> changed{(ImportBase + 2 * Page) >> Abi::PageShift};
    const auto patched = table.PatchPages(context, changed);
    Expect(patched != nullptr && patched != second, "a patch did not make a new version");
    Expect(Translate(patched, ImportBase + 2 * Page + 0x40) == ResidentDevice + 0x40, "the patched page does not map to its mirror");
    Expect(Translate(patched, ImportBase + Page + 0x40) == ImportDevice + Page + 0x40, "a page next to the patch moved");
    Expect(Translate(patched, ImportBase + 3 * Page + 0x40) == ImportDevice + 3 * Page + 0x40, "the page after the patch moved");
    Expect(Translate(second, ImportBase + 2 * Page + 0x40) == ImportDevice + 2 * Page + 0x40, "the patch changed the version it started from");
    Expect(table.Current() == patched && table.Totals().patches == 1, "the current version or the patch count is wrong");
    const auto listed = table.Ranges();
    Expect(std::any_of(listed.begin(), listed.end(), [](const Range& range) { return range.begin == ImportBase + 2 * Page && range.deviceAddress == ResidentDevice; }), "the patched table's range list lacks the resident page");
    const auto writable = AgcDriver::NewPath::WritableRanges(listed);
    Expect(std::none_of(writable.begin(), writable.end(), [](const AgcDriver::NewPath::AddressRange& range) { return range.first < ImportBase + 9 * Page && ImportBase + 8 * Page < range.second; }), "the read-only import is in the writable ranges");

    ++version;
    const auto rebuilt = table.Refresh(context);
    Expect(Translate(rebuilt, ImportBase + 2 * Page + 0x40) == ResidentDevice + 0x40, "a rebuild dropped the resident page");
    const auto readOnly = rebuilt->Find(ImportBase + 8 * Page);
    Expect(readOnly && readOnly->permissions == Abi::Read, "a read-only import lost its permissions");
}

}

int main() {
    try {
        OverlayTests();
        auto device = OpenVulkanTestDevice();
        if (device == nullptr) {
            if (failures != 0) return 1;
            return VulkanTestSkipped;
        }
        DeviceTests(device->graphicsContext());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d new-path page table checks failed\n", failures);
        return 1;
    }
    std::puts("new-path page table tests passed");
    return 0;
}
