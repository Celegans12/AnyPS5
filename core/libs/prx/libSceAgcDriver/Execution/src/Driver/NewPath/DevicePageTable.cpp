#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DevicePageTable.hpp"
#include <algorithm>

namespace AgcDriver::NewPath {

namespace {

using Range = DevicePageTable::Range;

void appendClipped(std::vector<Range>& out, const Range& range, std::uint64_t begin, std::uint64_t end) {
    const auto low = std::max(range.begin, begin);
    const auto high = std::min(range.end, end);
    if (low >= high) return;
    out.push_back({low, high, range.deviceAddress + (low - range.begin), range.permissions, range.reserved});
}

std::uint64_t coveredBytes(std::span<const Range> ranges) {
    std::uint64_t bytes = 0;
    for (const auto& range : ranges) bytes += range.end - range.begin;
    return bytes;
}

}

DevicePageTable::DevicePageTable(ImportSource imports, VersionSource version) : imports(std::move(imports)), version(std::move(version)) {}

std::vector<Range> DevicePageTable::Overlay(std::span<const Range> base, std::span<const Range> resident, std::uint64_t begin, std::uint64_t end) {
    std::vector<Range> out;
    auto cursor = begin;
    const auto fillTo = [&](std::uint64_t limit) {
        if (limit <= cursor) return;
        auto at = std::upper_bound(base.begin(), base.end(), cursor, [](std::uint64_t value, const Range& range) { return value < range.end; });
        for (; at != base.end() && at->begin < limit; ++at) appendClipped(out, *at, cursor, limit);
    };
    const auto appendResident = [&](const Range& piece, std::uint64_t low, std::uint64_t high) {
        auto at = std::upper_bound(base.begin(), base.end(), low, [](std::uint64_t value, const Range& range) { return value < range.end; });
        auto from = low;
        for (; at != base.end() && at->begin < high && from < high; ++at) {
            if (at->begin > from) {
                appendClipped(out, piece, from, at->begin);
                from = at->begin;
            }
            auto clamped = piece;
            clamped.permissions &= at->permissions;
            appendClipped(out, clamped, from, std::min(at->end, high));
            from = std::min(at->end, high);
        }
        if (from < high) appendClipped(out, piece, from, high);
    };
    for (const auto& piece : resident) {
        if (piece.end <= cursor || piece.begin >= end) continue;
        fillTo(std::max(piece.begin, cursor));
        appendResident(piece, std::max(piece.begin, cursor), std::min(piece.end, end));
        cursor = std::min(piece.end, end);
    }
    fillTo(end);
    return out;
}

std::vector<Range> DevicePageTable::piecesLocked(std::uint64_t begin, std::uint64_t end) const {
    std::vector<Range> pieces;
    if (resident) pieces = resident(begin, end);
    std::sort(pieces.begin(), pieces.end(), [](const Range& left, const Range& right) { return left.begin < right.begin; });
    return Overlay(importRanges, pieces, begin, end);
}

std::shared_ptr<const Graphics::BdaPageTable> DevicePageTable::Refresh(const Graphics::Context& context) {
    const auto now = version(context.device);
    std::lock_guard lock(mutex);
    if (built && now == builtVersion) {
        ++counts.unchanged;
        return table;
    }
    if (table != nullptr && tableDevice != context.device) {
        outlivedDevices.push_back(std::move(table));
        ++counts.outlived;
    }
    tableDevice = context.device;
    auto snapshot = imports(context);
    importRanges = std::move(snapshot.ranges);
    std::sort(importRanges.begin(), importRanges.end(), [](const Range& left, const Range& right) { return left.begin < right.begin; });
    auto ranges = piecesLocked(0, ShaderRecompiler::BdaAbi::PageTableEnd);
    Graphics::ValidateBdaPageRanges(ranges);
    table = Graphics::BdaPageTable::Build(context, ranges);
    built = true;
    builtVersion = now;
    ++counts.builds;
    counts.ranges = ranges.size();
    counts.coveredBytes = coveredBytes(ranges);
    counts.importVersion = now;
    tableRanges = std::move(ranges);
    tableLists = ListsOf(tableRanges);
    return table;
}

bool DevicePageTable::Stale(VkDevice device) const {
    const auto now = version(device);
    std::lock_guard lock(mutex);
    return !built || now != builtVersion;
}

std::shared_ptr<const Graphics::BdaPageTable> DevicePageTable::PatchPages(const Graphics::Context& context, std::span<const std::uint64_t> pages) {
    std::lock_guard lock(mutex);
    if (table == nullptr) return nullptr;
    table = Graphics::BdaPageTable::Patch(context, table, pages, [this](std::uint64_t begin, std::uint64_t end) { return piecesLocked(begin, end); });
    ++counts.patches;
    tableRanges = piecesLocked(0, ShaderRecompiler::BdaAbi::PageTableEnd);
    tableLists = ListsOf(tableRanges);
    return table;
}

void DevicePageTable::Release(VkDevice device) {
    std::shared_ptr<const Graphics::BdaPageTable> dropped;
    {
        std::lock_guard lock(mutex);
        if (tableDevice != device || table == nullptr) return;
        dropped = std::move(table);
        tableDevice = VK_NULL_HANDLE;
        importRanges.clear();
        tableRanges.clear();
        tableLists = {};
        built = false;
        builtVersion = 0;
        ++counts.released;
    }
}

void DevicePageTable::SetResidentPieces(ResidentPieces pieces) {
    std::lock_guard lock(mutex);
    resident = std::move(pieces);
}

std::shared_ptr<const Graphics::BdaPageTable> DevicePageTable::Current() const {
    std::lock_guard lock(mutex);
    return table;
}

TableLists DevicePageTable::Lists() const {
    std::lock_guard lock(mutex);
    return tableLists;
}

std::uint64_t DevicePageTable::RootAddress() const {
    std::lock_guard lock(mutex);
    return table != nullptr ? static_cast<std::uint64_t>(table->Address()) : 0u;
}

std::vector<Range> DevicePageTable::Ranges() const {
    std::lock_guard lock(mutex);
    return tableRanges;
}

DevicePageTable::Counts DevicePageTable::Totals() const {
    std::lock_guard lock(mutex);
    return counts;
}

DevicePageTable& DeviceTable() {
    static auto* table = new DevicePageTable([](const Graphics::Context& context) { return Graphics::HostImportRanges(context); }, [](VkDevice device) { return Graphics::HostImportsVersion(device); });
    return *table;
}

}
