#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DevicePageTable.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandWrites.hpp"
#include <algorithm>
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "VulkanTestDevice.hpp"
#include <cstdio>
#include <exception>
#include <memory>
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

void CompareTests() {
    using Cover = DevicePageTable::Cover;
    const std::vector<Range> table{MakeRange(ImportBase, ImportBase + 4 * Page, ImportDevice), MakeRange(ImportBase + 4 * Page, ImportBase + 6 * Page, ImportDevice + 4 * Page, Abi::Read), MakeRange(ImportBase + 8 * Page, ImportBase + 9 * Page, ResidentDevice)};
    const std::vector<Range> inside{MakeRange(ImportBase + Page, ImportBase + 3 * Page, ImportDevice + Page), MakeRange(ImportBase + 3 * Page, ImportBase + 5 * Page, ImportDevice + 3 * Page, Abi::Read)};
    Expect(DevicePageTable::Compare(table, inside) == Cover::Covered, "a build mapping the table's addresses across two entries was not covered");
    Expect(DevicePageTable::Compare(table, {}) == Cover::Covered, "an empty build was not covered");
    const std::vector<Range> gap{MakeRange(ImportBase + 5 * Page, ImportBase + 7 * Page, ImportDevice + 5 * Page, Abi::Read)};
    Expect(DevicePageTable::Compare(table, gap) == Cover::Missing, "a build reaching past the table was covered");
    const std::vector<Range> staged{MakeRange(ImportBase + 2 * Page, ImportBase + 3 * Page, 0x7d0000000000ull)};
    Expect(DevicePageTable::Compare(table, staged) == Cover::OtherAddress, "a staged copy (another device address) was covered");
    const std::vector<Range> resident{MakeRange(ImportBase + 8 * Page, ImportBase + 9 * Page, ImportDevice + 8 * Page)};
    Expect(DevicePageTable::Compare(table, resident) == Cover::OtherAddress, "a page the table maps to its resident slot was covered for an import build");
    const std::vector<Range> writes{MakeRange(ImportBase + 4 * Page, ImportBase + 5 * Page, ImportDevice + 4 * Page)};
    Expect(DevicePageTable::Compare(table, writes) == Cover::Narrower, "a writable build range over a read-only table entry was covered");
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


constexpr std::uint64_t MirrorBase = 0x213c000ull;
constexpr std::uint64_t MirrorEnd = 0x22b4000ull;
constexpr std::uint64_t MirrorDevice = 0x4fd9c0000ull;

void MirrorMergeTests() {
    const std::vector<Range> imports{MakeRange(ImportBase, ImportBase + 4 * Page, ImportDevice)};
    const std::vector<Range> mirrors{MakeRange(MirrorBase, MirrorEnd, MirrorDevice, Abi::Read)};
    const auto merged = DevicePageTable::WithMirrors(imports, mirrors);
    Expect(merged.size() == 2 && merged[0].begin == MirrorBase && merged[0].end == MirrorEnd && merged[0].deviceAddress == MirrorDevice && merged[0].permissions == Abi::Read && merged[1].begin == ImportBase, "a read-only mirror apart from the imports was not added as its own read-only entry, sorted");
    Expect(DevicePageTable::WithMirrors(imports, {}).size() == 1, "no mirrors changed the imports");
    // A mirror an import partly maps: the import keeps its bytes, the mirror fills around it.
    const std::vector<Range> middle{MakeRange(ImportBase + Page, ImportBase + 2 * Page, ImportDevice + Page)};
    const std::vector<Range> wide{MakeRange(ImportBase, ImportBase + 4 * Page, MirrorDevice, Abi::Read)};
    const auto around = DevicePageTable::WithMirrors(middle, wide);
    Expect(around.size() == 3, "a mirror around an import did not give three entries");
    if (around.size() == 3) {
        Expect(around[0].begin == ImportBase && around[0].end == ImportBase + Page && around[0].deviceAddress == MirrorDevice, "the mirror piece before the import is wrong");
        Expect(around[1].deviceAddress == ImportDevice + Page, "the import lost its bytes to a mirror");
        Expect(around[2].begin == ImportBase + 2 * Page && around[2].end == ImportBase + 4 * Page && around[2].deviceAddress == MirrorDevice + 2 * Page, "the mirror piece after the import lost its device offset");
    }
    std::vector<Range> validated = around;
    AgcDriver::Graphics::ValidateBdaPageRanges(validated);

    // F4 against a table holding the mirror: the build range D-next saw refused in-world is covered
    // when it binds that mirror; another device address or a write is not.
    using Cover = DevicePageTable::Cover;
    const std::vector<Range> build{MakeRange(MirrorBase, MirrorEnd, MirrorDevice, Abi::Read), MakeRange(ImportBase + Page, ImportBase + 3 * Page, ImportDevice + Page)};
    Expect(DevicePageTable::Compare(imports, build) == Cover::Missing, "the exe mirror range was covered by an imports-only table");
    Expect(DevicePageTable::Compare(merged, build) == Cover::Covered, "a build binding the exe mirror was not covered by the table with mirrors");
    const std::vector<Range> copy{MakeRange(MirrorBase, MirrorEnd, MirrorDevice + 0x100000, Abi::Read)};
    Expect(DevicePageTable::Compare(merged, copy) == Cover::OtherAddress, "a build copying the exe range (another device address) was covered");
    const std::vector<Range> written{MakeRange(MirrorBase, MirrorBase + Page, MirrorDevice)};
    Expect(DevicePageTable::Compare(merged, written) == Cover::Narrower, "a writable build range over the read-only mirror was covered");
}

void MirrorDeviceTests(const AgcDriver::Graphics::Context& context) {
    const std::vector<Range> imports{MakeRange(ImportBase, ImportBase + 4 * Page, ImportDevice)};
    std::uint64_t mirrorVersion = 1;
    std::vector<Range> mirrors{MakeRange(MirrorBase, MirrorEnd, MirrorDevice, Abi::Read)};
    auto hold = std::make_shared<int>(7);
    std::weak_ptr<int> held = hold;
    DevicePageTable table([&](const AgcDriver::Graphics::Context& built) {
        AgcDriver::Graphics::HostImportSnapshot snapshot;
        snapshot.device = built.device;
        snapshot.ranges = imports;
        return snapshot;
    }, [](VkDevice) { return std::uint64_t{1}; }, [&](const AgcDriver::Graphics::Context& built) {
        AgcDriver::Graphics::ImageMirrorTableSnapshot snapshot;
        snapshot.device = built.device;
        snapshot.version = mirrorVersion;
        snapshot.ranges = mirrors;
        if (hold != nullptr) snapshot.holds.push_back(hold);
        return snapshot;
    }, [&](VkDevice) { return mirrorVersion; });
    const auto first = table.Refresh(context);
    Expect(Translate(first, MirrorBase + 0x1234) == MirrorDevice + 0x1234, "an exe byte did not translate to its read-only mirror");
    Expect(Translate(first, ImportBase + 8) == ImportDevice + 8, "an import moved when a mirror was added");
    const auto entry = first->Find(MirrorBase);
    Expect(entry && entry->permissions == Abi::Read, "the mirror entry is not read-only");
    Expect(table.Totals().mirrorRanges == 1 && table.Totals().mirrorBytes == MirrorEnd - MirrorBase, "the mirror counts are wrong");
    hold.reset();
    Expect(!held.expired(), "the table does not hold the mapped mirror's buffer");
    const auto writable = AgcDriver::NewPath::WritableRanges(table.Ranges());
    Expect(std::none_of(writable.begin(), writable.end(), [](const AgcDriver::NewPath::AddressRange& range) { return range.first < MirrorEnd && MirrorBase < range.second; }), "the read-only mirror is in the writable ranges");
    const auto build = std::make_shared<const std::vector<Range>>(std::vector<Range>{MakeRange(MirrorBase, MirrorEnd, MirrorDevice, Abi::Read)});
    Expect(table.Covers(build) == DevicePageTable::Cover::Covered, "Covers refused a build binding the mapped mirror");
    const auto hitsBefore = table.Totals().coverHits;
    Expect(table.Probe(*build) == DevicePageTable::Cover::Covered, "Probe refused a build binding the mapped mirror");
    const std::vector<Range> written{MakeRange(MirrorBase, MirrorEnd, MirrorDevice, Abi::Read | Abi::Write)};
    Expect(table.Probe(written) == DevicePageTable::Cover::Narrower, "Probe covered a writable range over the read-only mirror");
    const std::vector<Range> staged{MakeRange(ImportBase, ImportBase + 0x100, ImportDevice + 0x40000000ull, Abi::Read)};
    Expect(table.Probe(staged) == DevicePageTable::Cover::OtherAddress, "Probe covered a staged copy of an import range");
    Expect(table.Totals().coverHits == hitsBefore, "Probe counted in the F4 compare totals");

    // The mirror replaced (a reprotect made a new one at another device address): the version moves,
    // the table is stale, the old build is refused and the new one covered.
    Expect(!table.Stale(context.device), "an unchanged mirror version made the table stale");
    mirrors = {MakeRange(MirrorBase, MirrorEnd, MirrorDevice + 0x10000000ull, Abi::Read)};
    ++mirrorVersion;
    Expect(table.Stale(context.device), "a new mirror version did not make the table stale");
    const auto second = table.Refresh(context);
    Expect(second != first && Translate(second, MirrorBase) == MirrorDevice + 0x10000000ull, "the rebuilt table does not map the replacement mirror");
    Expect(held.expired(), "the replaced mirror's buffer is still held after the rebuild");
    Expect(table.Covers(build) == DevicePageTable::Cover::OtherAddress, "a build binding the replaced mirror was covered");
    // The mirror dropped: the range is missing again.
    mirrors.clear();
    ++mirrorVersion;
    const auto third = table.Refresh(context);
    Expect(Translate(third, MirrorBase) == 0 && table.Totals().mirrorRanges == 0, "a dropped mirror is still mapped");
    table.Release(context.device);
}

void SerialMemoTests(const AgcDriver::Graphics::Context& context) {
    using Cover = DevicePageTable::Cover;
    std::uint64_t version = 1;
    std::vector<Range> imports{MakeRange(ImportBase, ImportBase + 4 * Page, ImportDevice)};
    DevicePageTable table([&](const AgcDriver::Graphics::Context& built) {
        AgcDriver::Graphics::HostImportSnapshot snapshot;
        snapshot.device = built.device;
        snapshot.ranges = imports;
        return snapshot;
    }, [&](VkDevice) { return version; });
    const std::vector<Range> inside{MakeRange(ImportBase + Page, ImportBase + 2 * Page, ImportDevice + Page)};
    const std::vector<Range> outside{MakeRange(ImportBase + 5 * Page, ImportBase + 6 * Page, ImportDevice + 5 * Page)};
    Expect(table.Covers(7, inside) == Cover::Missing && table.Totals().compares == 0, "a table never built covered a build or compared it");
    static_cast<void>(table.Refresh(context));
    Expect(table.Covers(7, inside) == Cover::Covered && table.Totals().compares == 1, "the first use of a space serial was not compared");
    Expect(table.Covers(7, inside) == Cover::Covered && table.Totals().compares == 1 && table.Totals().serialHits == 1, "a second use of the same space serial compared again");
    Expect(table.Covers(8, outside) == Cover::Missing && table.Totals().compares == 2, "another space serial was served from the memo");
    Expect(table.Covers(0, inside) == Cover::Covered && table.Covers(0, outside) == Cover::Missing && table.Totals().compares == 4, "a build without a space serial was served from the memo");
    DevicePageTable::Snapshot admitted;
    Expect(table.Covers(7, inside, &admitted) == Cover::Covered && admitted.table == table.Current() && admitted.lists.writable == table.Lists().writable, "a covered compare did not hand out the table version it compared against");
    const auto admittedTable = admitted.table;
    imports = {MakeRange(ImportBase + 4 * Page, ImportBase + 8 * Page, ImportDevice + 4 * Page)};
    ++version;
    static_cast<void>(table.Refresh(context));
    Expect(admitted.table == admittedTable && table.Current() != admittedTable && Translate(admitted.table, ImportBase + Page + 8) == ImportDevice + Page + 8, "a refresh changed the version a covered build was admitted with");
    Expect(table.Use().table == table.Current(), "Use did not hand out the current version");
    Expect(table.Covers(7, inside) == Cover::Missing && table.Covers(8, outside) == Cover::Covered && table.Totals().compares == 6, "a new table version kept the old version's covers");
    const auto hits = table.Totals().coverHits;
    Expect(hits[static_cast<std::size_t>(Cover::Covered)] == 5 && hits[static_cast<std::size_t>(Cover::Missing)] == 4, "the cover counts are wrong");
    table.Release(context.device);
    Expect(table.Covers(8, outside) == Cover::Missing, "a released table served a memoized cover");
}

}

int main() {
    try {
        OverlayTests();
        CompareTests();
        MirrorMergeTests();
        auto device = OpenVulkanTestDevice();
        if (device == nullptr) {
            if (failures != 0) return 1;
            return VulkanTestSkipped;
        }
        DeviceTests(device->graphicsContext());
        MirrorDeviceTests(device->graphicsContext());
        SerialMemoTests(device->graphicsContext());
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
