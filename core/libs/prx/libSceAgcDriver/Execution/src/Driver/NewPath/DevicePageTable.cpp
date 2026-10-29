#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DevicePageTable.hpp"
#include <algorithm>
#include <cstdio>

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

DevicePageTable::DevicePageTable(ImportSource imports, VersionSource version, MirrorSource mirrors, VersionSource mirrorVersion) : imports(std::move(imports)), version(std::move(version)), mirrors(std::move(mirrors)), mirrorVersion(std::move(mirrorVersion)) {}

std::uint64_t DevicePageTable::mirrorVersionOf(VkDevice device) const {
    return mirrorVersion ? mirrorVersion(device) : 0u;
}

std::vector<Range> DevicePageTable::WithMirrors(std::vector<Range> imports, std::span<const Range> mirrors) {
    if (mirrors.empty()) return imports;
    std::vector<Range> sorted(mirrors.begin(), mirrors.end());
    std::sort(sorted.begin(), sorted.end(), [](const Range& left, const Range& right) { return left.begin < right.begin; });
    std::vector<Range> out = imports;
    std::uint64_t claimed = 0;
    for (const auto& mirror : sorted) {
        // Mirrors overlapping each other: the first one keeps the bytes (never seen; registered
        // ranges are disjoint).
        auto cursor = std::max(mirror.begin, claimed);
        auto at = std::upper_bound(imports.begin(), imports.end(), cursor, [](std::uint64_t value, const Range& range) { return value < range.end; });
        while (cursor < mirror.end) {
            const auto stop = at == imports.end() ? mirror.end : std::min(mirror.end, at->begin);
            if (cursor < stop) appendClipped(out, mirror, cursor, stop);
            if (at == imports.end() || at->begin >= mirror.end) break;
            cursor = std::max(cursor, at->end);
            ++at;
        }
        claimed = std::max(claimed, mirror.end);
    }
    std::sort(out.begin(), out.end(), [](const Range& left, const Range& right) { return left.begin < right.begin; });
    return out;
}

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

DevicePageTable::Cover DevicePageTable::Compare(std::span<const Range> table, std::span<const Range> build, const Range** failed, std::uint64_t* failedAt) {
    const auto fail = [&](const Range& range, std::uint64_t cursor, Cover cover) {
        if (failed != nullptr) *failed = &range;
        if (failedAt != nullptr) *failedAt = cursor;
        return cover;
    };
    for (const auto& range : build) {
        if (range.end <= range.begin) continue;
        auto cursor = range.begin;
        auto at = std::upper_bound(table.begin(), table.end(), cursor, [](std::uint64_t value, const Range& entry) { return value < entry.end; });
        while (cursor < range.end) {
            if (at == table.end() || at->begin > cursor) return fail(range, cursor, Cover::Missing);
            if (at->deviceAddress + (cursor - at->begin) != range.deviceAddress + (cursor - range.begin)) return fail(range, cursor, Cover::OtherAddress);
            if ((at->permissions & range.permissions) != range.permissions) return fail(range, cursor, Cover::Narrower);
            cursor = std::min(range.end, at->end);
            ++at;
        }
    }
    return Cover::Covered;
}

DevicePageTable::Cover DevicePageTable::Covers(const std::shared_ptr<const std::vector<Range>>& build) {
    std::lock_guard lock(mutex);
    if (build == nullptr || !built) {
        ++counts.coverHits[static_cast<std::size_t>(Cover::Missing)];
        return Cover::Missing;
    }
    for (const auto& memo : coverMemo) {
        if (memo.build == build && memo.generation == generation) {
            ++counts.coverHits[static_cast<std::size_t>(memo.cover)];
            return memo.cover;
        }
    }
    const auto cover = compareLocked(*build);
    coverMemo[coverNext] = {build, generation, cover};
    coverNext = (coverNext + 1) % coverMemo.size();
    return cover;
}

DevicePageTable::Cover DevicePageTable::Covers(std::uint64_t serial, std::span<const Range> build) {
    std::lock_guard lock(mutex);
    if (!built) {
        ++counts.coverHits[static_cast<std::size_t>(Cover::Missing)];
        return Cover::Missing;
    }
    if (serialMemoGeneration != generation) {
        serialMemo.clear();
        serialMemoGeneration = generation;
    }
    if (serial != 0) {
        if (const auto found = serialMemo.find(serial); found != serialMemo.end()) {
            ++counts.serialHits;
            ++counts.coverHits[static_cast<std::size_t>(found->second)];
            return found->second;
        }
    }
    const auto cover = compareLocked(build);
    if (serial != 0) {
        if (serialMemo.size() >= 4096) serialMemo.clear();
        serialMemo.emplace(serial, cover);
    }
    return cover;
}

DevicePageTable::Cover DevicePageTable::compareLocked(std::span<const Range> build) {
    ++counts.compares;
    const Range* failed = nullptr;
    std::uint64_t failedAt = 0;
    const auto cover = Compare(tableRanges, build, &failed, &failedAt);
    if (cover != Cover::Covered && failed != nullptr && reportedFailures < 12) {
        ++reportedFailures;
        static constexpr const char* names[] = {"covered", "missing", "other address", "narrower"};
        const auto near = std::upper_bound(tableRanges.begin(), tableRanges.end(), failedAt, [](std::uint64_t value, const Range& entry) { return value < entry.end; });
        std::fprintf(stderr, "[newpath] F4 %s at 0x%llx: build range 0x%llx-0x%llx (device 0x%llx, permissions %u) of %zu; next table entry %s0x%llx-0x%llx (device 0x%llx, permissions %u)\n", names[static_cast<std::size_t>(cover)], static_cast<unsigned long long>(failedAt), static_cast<unsigned long long>(failed->begin), static_cast<unsigned long long>(failed->end), static_cast<unsigned long long>(failed->deviceAddress), static_cast<unsigned>(failed->permissions), build.size(), near == tableRanges.end() ? "none " : "", static_cast<unsigned long long>(near != tableRanges.end() ? near->begin : 0), static_cast<unsigned long long>(near != tableRanges.end() ? near->end : 0), static_cast<unsigned long long>(near != tableRanges.end() ? near->deviceAddress : 0), static_cast<unsigned>(near != tableRanges.end() ? near->permissions : 0));
    }
    ++counts.coverHits[static_cast<std::size_t>(cover)];
    return cover;
}

std::vector<Range> DevicePageTable::piecesLocked(std::uint64_t begin, std::uint64_t end) const {
    std::vector<Range> pieces;
    if (resident) pieces = resident(begin, end);
    std::sort(pieces.begin(), pieces.end(), [](const Range& left, const Range& right) { return left.begin < right.begin; });
    return Overlay(importRanges, pieces, begin, end);
}

std::shared_ptr<const Graphics::BdaPageTable> DevicePageTable::Refresh(const Graphics::Context& context) {
    const auto now = version(context.device);
    const auto mirrorNow = mirrorVersionOf(context.device);
    std::lock_guard lock(mutex);
    if (built && now == builtVersion && mirrorNow == builtMirrorVersion) {
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
    mirrorHolds.clear();
    counts.mirrorRanges = 0;
    counts.mirrorBytes = 0;
    if (mirrors) {
        auto mirrored = mirrors(context);
        if (mirrored.device == context.device && !mirrored.ranges.empty()) {
            const auto before = coveredBytes(importRanges);
            const auto entries = importRanges.size();
            importRanges = WithMirrors(std::move(importRanges), mirrored.ranges);
            counts.mirrorRanges = importRanges.size() - entries;
            counts.mirrorBytes = coveredBytes(importRanges) - before;
            mirrorHolds = std::move(mirrored.holds);
        }
    }
    auto ranges = piecesLocked(0, ShaderRecompiler::BdaAbi::PageTableEnd);
    Graphics::ValidateBdaPageRanges(ranges);
    table = Graphics::BdaPageTable::Build(context, ranges);
    built = true;
    builtVersion = now;
    builtMirrorVersion = mirrorNow;
    counts.mirrorVersion = mirrorNow;
    ++counts.builds;
    counts.ranges = ranges.size();
    counts.coveredBytes = coveredBytes(ranges);
    counts.importVersion = now;
    tableRanges = std::move(ranges);
    tableLists = ListsOf(tableRanges);
    ++generation;
    return table;
}

bool DevicePageTable::Stale(VkDevice device) const {
    const auto now = version(device);
    const auto mirrorNow = mirrorVersionOf(device);
    std::lock_guard lock(mutex);
    return !built || now != builtVersion || mirrorNow != builtMirrorVersion;
}

std::shared_ptr<const Graphics::BdaPageTable> DevicePageTable::PatchPages(const Graphics::Context& context, std::span<const std::uint64_t> pages) {
    std::lock_guard lock(mutex);
    if (table == nullptr) return nullptr;
    table = Graphics::BdaPageTable::Patch(context, table, pages, [this](std::uint64_t begin, std::uint64_t end) { return piecesLocked(begin, end); });
    ++counts.patches;
    tableRanges = piecesLocked(0, ShaderRecompiler::BdaAbi::PageTableEnd);
    tableLists = ListsOf(tableRanges);
    ++generation;
    return table;
}

void DevicePageTable::Release(VkDevice device) {
    std::shared_ptr<const Graphics::BdaPageTable> dropped;
    std::vector<std::shared_ptr<Graphics::Buffer>> droppedRecords;
    {
        std::lock_guard lock(mutex);
        if (recordDevice == device) {
            droppedRecords = std::move(records);
            records.clear();
            recordDevice = VK_NULL_HANDLE;
        }
        if (tableDevice != device || table == nullptr) return;
        dropped = std::move(table);
        tableDevice = VK_NULL_HANDLE;
        importRanges.clear();
        mirrorHolds.clear();
        builtMirrorVersion = 0;
        tableRanges.clear();
        ++generation;
        coverMemo = {};
        serialMemo.clear();
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

std::shared_ptr<Graphics::Buffer> DevicePageTable::TakeRecord(const Graphics::Context& context) {
    {
        std::lock_guard lock(mutex);
        if (recordDevice != context.device) {
            records.clear();
            recordDevice = context.device;
        }
        if (!records.empty()) {
            auto record = std::move(records.back());
            records.pop_back();
            return record;
        }
        ++counts.records;
    }
    auto record = std::make_shared<Graphics::Buffer>(context, ShaderRecompiler::BdaAbi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    auto bytes = record->Bytes();
    ClearCommandRecord(std::span<std::uint32_t, CommandRecordWords>(reinterpret_cast<std::uint32_t*>(bytes.data()), CommandRecordWords));
    return record;
}

void DevicePageTable::ReturnRecord(std::shared_ptr<Graphics::Buffer> record, VkDevice device) {
    std::lock_guard lock(mutex);
    if (record == nullptr || device != recordDevice) return;
    records.push_back(std::move(record));
}

std::size_t DevicePageTable::PooledRecords() const {
    std::lock_guard lock(mutex);
    return records.size();
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
    static auto* table = new DevicePageTable([](const Graphics::Context& context) { return Graphics::HostImportRanges(context); }, [](VkDevice device) { return Graphics::HostImportsVersion(device); }, [](const Graphics::Context& context) { return Graphics::ReadOnlyImageMirrorRanges(context); }, [](VkDevice device) { return Graphics::ReadOnlyImageMirrorsVersion(device); });
    return *table;
}

}
