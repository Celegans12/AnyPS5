#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>

namespace AgcDriver::Graphics {

namespace {

namespace Abi = ShaderRecompiler::BdaAbi;

constexpr VkBufferUsageFlags TableUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
constexpr std::uint64_t PageBytes = std::uint64_t{1} << Abi::PageShift;
constexpr std::uint64_t WindowBytes = std::uint64_t{1} << Abi::WindowShift;

Abi::PageEntry Segment(const Abi::Range& range, std::uint64_t start) {
    const auto first = range.begin > start ? static_cast<std::uint32_t>(range.begin - start) : 0u;
    auto entry = Abi::SegmentEntry(range.deviceAddress - range.begin, first, std::min(range.end - start, Abi::LimitMax), range.permissions);
    if ((range.reserved & Abi::RangeMirrored) != 0) entry.bounds |= 1u << Abi::MirroredShift;
    return entry;
}

Abi::PageEntry Encode(std::vector<Abi::PageEntry>& entries, std::span<const Abi::Range> ranges, std::uint64_t start, std::uint32_t shift, const std::vector<std::size_t>& meeting) {
    if (meeting.size() == 1u) return Segment(ranges[meeting.front()], start);
    Require(shift >= Abi::SplitShift, "two BDA page ranges meet in one byte");
    const auto block = entries.size();
    entries.resize(block + Abi::SplitChildren);
    const auto childShift = shift - Abi::SplitShift;
    for (std::uint32_t child = 0; child < Abi::SplitChildren; ++child) {
        const auto childStart = start + (static_cast<std::uint64_t>(child) << childShift);
        const auto childEnd = childStart + (std::uint64_t{1} << childShift);
        std::vector<std::size_t> inside;
        for (const auto index : meeting) {
            if (ranges[index].begin < childEnd && childStart < ranges[index].end) inside.push_back(index);
        }
        if (inside.empty()) continue;
        const auto entry = Encode(entries, ranges, childStart, childShift, inside);
        entries[block + child] = entry;
    }
    return Abi::SplitEntry(block * sizeof(Abi::PageEntry));
}

std::uint64_t CoveredPages(std::uint32_t window, std::span<const Abi::Range> ranges) {
    const auto start = static_cast<std::uint64_t>(window) << Abi::WindowShift;
    const auto end = start + WindowBytes;
    std::uint64_t pages = 0;
    std::uint64_t counted = 0;
    for (const auto& range : ranges) {
        const auto first = std::max({range.begin, start, counted}) & ~(PageBytes - 1u);
        const auto last = std::min(range.end, end);
        if (last <= first) continue;
        const auto stop = (last + PageBytes - 1u) & ~(PageBytes - 1u);
        pages += (stop - std::max(first, counted)) >> Abi::PageShift;
        counted = stop;
    }
    return pages;
}

bool SameSegment(const Abi::Range& left, const Abi::Range& right) {
    return left.end == right.end && left.deviceAddress - left.begin == right.deviceAddress - right.begin && left.permissions == right.permissions && left.reserved == right.reserved;
}

Abi::PageEntry EncodePage(std::vector<Abi::PageEntry>& entries, std::span<const Abi::Range> ranges, std::uint64_t pageStart) {
    const auto pageEnd = pageStart + PageBytes;
    auto at = std::upper_bound(ranges.begin(), ranges.end(), pageStart, [](std::uint64_t value, const Abi::Range& range) { return value < range.end; });
    std::vector<std::size_t> meeting;
    for (; at != ranges.end() && at->begin < pageEnd; ++at) {
        if (at->begin <= pageStart && pageEnd <= at->end) return Segment(*at, pageStart);
        meeting.push_back(static_cast<std::size_t>(at - ranges.begin()));
    }
    if (meeting.empty()) return {};
    return Encode(entries, ranges, pageStart, Abi::PageShift, meeting);
}

struct Newest {
    std::mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    std::weak_ptr<const BdaPageTable> table;
    BdaPageTable::Counts counts;
};

Newest& NewestTable() {
    static Newest newest;
    return newest;
}

}

void ValidateBdaRangeList(std::span<const Abi::Range> ranges) {
    for (std::size_t index = 1; index < ranges.size(); ++index) Require(ranges[index - 1].end <= ranges[index].begin, "BDA ranges are unsorted or overlap");
}

void ValidateBdaGrants(std::span<const Abi::Grant> grants) {
    for (std::size_t index = 0; index < grants.size(); ++index) {
        Require(grants[index].begin < grants[index].end, "a BDA write grant is empty");
        Require(index == 0 || grants[index - 1].end < grants[index].begin, "BDA write grants are unsorted, overlap or touch");
    }
}

void ValidateBdaPageRanges(std::span<const Abi::Range> ranges) {
    ValidateBdaRangeList(ranges);
    for (const auto& range : ranges) {
        Require(range.begin < range.end && range.end <= Abi::PageTableEnd, "a BDA page range is empty or ends past the page table");
        Require(range.deviceAddress != 0 && range.end - range.begin <= std::numeric_limits<std::uint64_t>::max() - range.deviceAddress, "a BDA page range has no device address or its device range wraps");
        Require((range.reserved & ~Abi::RangeMirrored) == 0 && (range.permissions & ~(Abi::Read | Abi::Write)) == 0, "a BDA page range has unknown permissions or reserved bits");
    }
}

std::vector<Abi::PageEntry> EncodeBdaWindow(std::uint32_t window, std::span<const Abi::Range> ranges) {
    std::vector<Abi::PageEntry> entries(Abi::WindowPages);
    const auto start = static_cast<std::uint64_t>(window) << Abi::WindowShift;
    const auto end = start + WindowBytes;
    std::vector<std::pair<std::uint64_t, std::size_t>> partial;
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        const auto& range = ranges[index];
        Require(range.begin < end && start < range.end, "a BDA page range does not meet its window");
        const auto first = (std::max(range.begin, start) - start) >> Abi::PageShift;
        const auto last = (std::min(range.end, end) - 1u - start) >> Abi::PageShift;
        for (auto page = first; page <= last; ++page) {
            const auto pageStart = start + (page << Abi::PageShift);
            if (range.begin <= pageStart && pageStart + PageBytes <= range.end) entries[page] = Segment(range, pageStart);
            else partial.emplace_back(page, index);
        }
    }
    for (std::size_t at = 0; at < partial.size();) {
        const auto page = partial[at].first;
        std::vector<std::size_t> meeting;
        for (; at < partial.size() && partial[at].first == page; ++at) meeting.push_back(partial[at].second);
        const auto entry = Encode(entries, ranges, start + (page << Abi::PageShift), Abi::PageShift, meeting);
        entries[page] = entry;
    }
    return entries;
}

std::shared_ptr<const BdaPageTable> BdaPageTable::Build(const Context& context, std::span<const Abi::Range> ranges) {
    std::shared_ptr<const BdaPageTable> previous;
    {
        auto& newest = NewestTable();
        std::lock_guard lock(newest.mutex);
        if (newest.device == context.device) previous = newest.table.lock();
    }
    return Build(context, ranges, std::span(&previous, 1));
}

std::shared_ptr<const BdaPageTable> BdaPageTable::Build(const Context& context, std::span<const Abi::Range> ranges, std::span<const std::shared_ptr<const BdaPageTable>> shares) {
    ValidateBdaPageRanges(ranges);
    auto& newest = NewestTable();
    const std::shared_ptr<const BdaPageTable> previous = shares.empty() ? nullptr : shares.front();
    std::map<std::uint32_t, std::vector<Abi::Range>> windows;
    for (const auto& range : ranges) {
        const auto last = static_cast<std::uint32_t>((range.end - 1u) >> Abi::WindowShift);
        for (auto window = static_cast<std::uint32_t>(range.begin >> Abi::WindowShift); window <= last; ++window) windows[window].push_back(range);
    }
    std::shared_ptr<BdaPageTable> table(new BdaPageTable());
    std::vector<std::uint64_t> addresses(Abi::WindowCount, 0u);
    std::uint64_t shared = 0;
    std::uint64_t encoded = 0;
    for (auto& [window, held] : windows) {
        std::shared_ptr<const Leaf> old;
        for (const auto& candidate : shares) {
            auto leaf = candidate != nullptr ? candidate->find(window) : nullptr;
            if (leaf != nullptr && leaf->ranges.size() == held.size() && std::memcmp(leaf->ranges.data(), held.data(), held.size() * sizeof(Abi::Range)) == 0) {
                old = std::move(leaf);
                break;
            }
        }
        if (old != nullptr) {
            table->leaves.push_back(std::move(old));
            ++shared;
        } else {
            auto leaf = std::make_shared<Leaf>();
            leaf->window = window;
            leaf->entries = EncodeBdaWindow(window, held);
            stage(context, leaf->table, leaf->entries.data(), leaf->entries.size() * sizeof(Abi::PageEntry));
            encoded += CoveredPages(window, held);
            leaf->ranges = std::move(held);
            table->leaves.push_back(std::move(leaf));
        }
        const auto address = table->leaves.back()->table.buffer->DeviceAddress();
        Require(address != 0 && address % sizeof(Abi::PageEntry) == 0, "a BDA page table leaf is not aligned for its entries");
        addresses[window] = address;
    }
    if (previous != nullptr && shared == windows.size() && previous->leaves.size() == windows.size()) {
        std::lock_guard lock(newest.mutex);
        ++newest.counts.tables;
        newest.counts.sharedWindows += shared;
        return previous;
    }
    stage(context, table->first, addresses.data(), addresses.size() * sizeof(std::uint64_t));
    Require(table->first.buffer->DeviceAddress() % sizeof(std::uint64_t) == 0, "the BDA page table is not aligned for its entries");
    std::shared_ptr<const BdaPageTable> built = std::move(table);
    std::lock_guard lock(newest.mutex);
    newest.device = context.device;
    newest.table = built;
    ++newest.counts.tables;
    newest.counts.sharedWindows += shared;
    newest.counts.encodedWindows += windows.size() - shared;
    newest.counts.encodedPages += encoded;
    return built;
}

void BdaPageTable::overlapping(std::uint64_t begin, std::uint64_t end, std::vector<Abi::Range>& found) const {
    if (begin >= end || begin >= Abi::PageTableEnd) return;
    const auto firstWindow = static_cast<std::uint32_t>(begin >> Abi::WindowShift);
    const auto lastWindow = static_cast<std::uint32_t>((std::min(end, Abi::PageTableEnd) - 1u) >> Abi::WindowShift);
    auto leaf = std::lower_bound(leaves.begin(), leaves.end(), firstWindow, [](const auto& held, std::uint32_t value) { return held->window < value; });
    for (; leaf != leaves.end() && (*leaf)->window <= lastWindow; ++leaf) {
        const auto& ranges = (*leaf)->ranges;
        auto at = std::upper_bound(ranges.begin(), ranges.end(), begin, [](std::uint64_t value, const Abi::Range& range) { return value < range.end; });
        for (; at != ranges.end() && at->begin < end; ++at) {
            if (found.empty() || found.back().begin < at->begin) found.push_back(*at);
        }
    }
}

std::shared_ptr<const BdaPageTable> BdaPageTable::Patch(const Context& context, const std::shared_ptr<const BdaPageTable>& previous, std::span<const std::uint64_t> changedPages, const Pieces& pieces) {
    Require(previous != nullptr, "a BDA page table patch has no table to patch");
    struct Extent {
        std::uint64_t begin = 0;
        std::uint64_t end = 0;
        std::vector<Abi::Range> fresh;
        std::vector<Abi::Range> old;
    };
    const auto settle = [&](Extent& extent) {
        for (;;) {
            extent.old.clear();
            previous->overlapping(extent.begin, extent.end, extent.old);
            extent.fresh = pieces(extent.begin, extent.end);
            auto low = extent.begin;
            auto high = extent.end;
            for (const auto& range : extent.old) {
                low = std::min(low, range.begin);
                high = std::max(high, range.end);
            }
            for (const auto& range : extent.fresh) {
                low = std::min(low, range.begin);
                high = std::max(high, range.end);
            }
            if (low == extent.begin && high == extent.end) return;
            extent.begin = low;
            extent.end = high;
        }
    };
    std::vector<Extent> extents;
    for (const auto page : changedPages) {
        const auto begin = page << Abi::PageShift;
        if (begin >= Abi::PageTableEnd) continue;
        if (std::any_of(extents.begin(), extents.end(), [&](const Extent& held) { return held.begin <= begin && begin < held.end; })) continue;
        Extent extent{begin, begin + PageBytes, {}, {}};
        settle(extent);
        for (bool merged = true; merged;) {
            merged = false;
            for (auto at = extents.begin(); at != extents.end(); ++at) {
                if (extent.end < at->begin || at->end < extent.begin) continue;
                extent.begin = std::min(extent.begin, at->begin);
                extent.end = std::max(extent.end, at->end);
                extents.erase(at);
                settle(extent);
                merged = true;
                break;
            }
        }
        extents.push_back(std::move(extent));
    }
    std::map<std::uint32_t, std::vector<const Extent*>> windows;
    for (const auto& extent : extents) {
        const auto last = static_cast<std::uint32_t>((std::min(extent.end, Abi::PageTableEnd) - 1u) >> Abi::WindowShift);
        for (auto window = static_cast<std::uint32_t>(extent.begin >> Abi::WindowShift); window <= last; ++window) windows[window].push_back(&extent);
    }
    for (auto& [window, held] : windows) std::sort(held.begin(), held.end(), [](const Extent* left, const Extent* right) { return left->begin < right->begin; });
    std::shared_ptr<BdaPageTable> table(new BdaPageTable());
    std::uint64_t encoded = 0;
    std::uint64_t patched = 0;
    auto kept = previous->leaves.begin();
    for (const auto& [window, held] : windows) {
        for (; kept != previous->leaves.end() && (*kept)->window < window; ++kept) table->leaves.push_back(*kept);
        std::shared_ptr<const Leaf> old;
        if (kept != previous->leaves.end() && (*kept)->window == window) old = *kept++;
        const auto start = static_cast<std::uint64_t>(window) << Abi::WindowShift;
        const auto stop = start + WindowBytes;
        std::vector<Abi::Range> ranges;
        if (old != nullptr) {
            ranges.reserve(old->ranges.size() + 2u * held.size());
            auto range = old->ranges.begin();
            for (const auto* extent : held) {
                for (; range != old->ranges.end() && range->end <= extent->begin; ++range) ranges.push_back(*range);
                for (; range != old->ranges.end() && range->begin < extent->end; ++range) {}
                for (const auto& fresh : extent->fresh) {
                    if (fresh.begin < stop && start < fresh.end) ranges.push_back(fresh);
                }
            }
            for (; range != old->ranges.end(); ++range) ranges.push_back(*range);
        } else {
            for (const auto* extent : held) {
                for (const auto& fresh : extent->fresh) {
                    if (fresh.begin < stop && start < fresh.end) ranges.push_back(fresh);
                }
            }
        }
        ++patched;
        if (ranges.empty()) continue;
        auto leaf = std::make_shared<Leaf>();
        leaf->window = window;
        if (old != nullptr) leaf->entries = old->entries;
        else leaf->entries.resize(Abi::WindowPages);
        for (const auto* extent : held) {
            std::vector<std::pair<std::uint64_t, std::uint64_t>> same;
            for (const auto& fresh : extent->fresh) {
                for (const auto& before : extent->old) {
                    if (!SameSegment(before, fresh) || fresh.end <= before.begin || before.end <= fresh.begin) continue;
                    const auto from = (std::max(fresh.begin, before.begin) + PageBytes - 1u) & ~(PageBytes - 1u);
                    const auto to = fresh.end & ~(PageBytes - 1u);
                    if (from < to) same.emplace_back(from, to);
                }
            }
            std::sort(same.begin(), same.end());
            auto skip = same.begin();
            const auto last = std::min(extent->end, stop);
            for (auto at = std::max(extent->begin & ~(PageBytes - 1u), start); at < last;) {
                while (skip != same.end() && skip->second <= at) ++skip;
                if (skip != same.end() && skip->first <= at) {
                    at = skip->second;
                    continue;
                }
                const auto entry = EncodePage(leaf->entries, ranges, at);
                leaf->entries[(at - start) >> Abi::PageShift] = entry;
                ++encoded;
                at += PageBytes;
            }
        }
        if (leaf->entries.size() > 2u * Abi::WindowPages) {
            leaf->entries = EncodeBdaWindow(window, ranges);
            encoded += CoveredPages(window, ranges);
        }
        stage(context, leaf->table, leaf->entries.data(), leaf->entries.size() * sizeof(Abi::PageEntry));
        leaf->ranges = std::move(ranges);
        table->leaves.push_back(std::move(leaf));
    }
    for (; kept != previous->leaves.end(); ++kept) table->leaves.push_back(*kept);
    if (table->leaves.empty()) return nullptr;
    std::vector<std::uint64_t> addresses(Abi::WindowCount, 0u);
    for (const auto& leaf : table->leaves) {
        const auto address = leaf->table.buffer->DeviceAddress();
        Require(address != 0 && address % sizeof(Abi::PageEntry) == 0, "a BDA page table leaf is not aligned for its entries");
        addresses[leaf->window] = address;
    }
    stage(context, table->first, addresses.data(), addresses.size() * sizeof(std::uint64_t));
    Require(table->first.buffer->DeviceAddress() % sizeof(std::uint64_t) == 0, "the BDA page table is not aligned for its entries");
    std::shared_ptr<const BdaPageTable> built = std::move(table);
    auto& newest = NewestTable();
    std::lock_guard lock(newest.mutex);
    newest.device = context.device;
    newest.table = built;
    ++newest.counts.tables;
    newest.counts.encodedPages += encoded;
    newest.counts.patchedWindows += patched;
    return built;
}

BdaPageTable::Counts BdaPageTable::Totals() {
    auto& newest = NewestTable();
    std::lock_guard lock(newest.mutex);
    return newest.counts;
}

VkDeviceAddress BdaPageTable::Address() const {
    return first.buffer->DeviceAddress();
}

VkDeviceAddress BdaPageTable::LeafAddress(std::uint32_t window) const {
    const auto leaf = find(window);
    return leaf != nullptr ? leaf->table.buffer->DeviceAddress() : 0u;
}

void BdaPageTable::stage(const Context& context, Upload& upload, const void* bytes, std::size_t size) {
    upload.buffer = std::make_shared<Buffer>(context, size, TableUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    upload.staging = std::make_shared<Buffer>(context, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    upload.bytes = size;
    std::memcpy(upload.staging->Bytes().data(), bytes, size);
}

bool BdaPageTable::recordUpload(const Context& context, VkCommandBuffer commands, const Upload& upload, std::uint32_t stream) {
    if (stream != EveryStream) {
        Require(stream < 32u, "a BDA page table upload names a stream past its upload mask");
        const auto bit = 1u << stream;
        if ((upload.streams.fetch_or(bit, std::memory_order_relaxed) & bit) != 0u) return false;
    }
    const VkBufferCopy region{0, 0, upload.bytes};
    context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer")(commands, upload.staging->Handle(), upload.buffer->Handle(), 1, &region);
    return true;
}

VkMemoryPropertyFlags BdaPageTable::Properties() const {
    auto properties = first.buffer->Properties();
    for (const auto& leaf : leaves) properties &= leaf->table.buffer->Properties();
    return properties;
}

bool BdaPageTable::NeedsUploads(std::uint32_t stream) const {
    if (stream == EveryStream) return true;
    const auto bit = 1u << stream;
    const auto missing = [&](const Upload& upload) { return (upload.streams.load(std::memory_order_relaxed) & bit) == 0u; };
    return missing(first) || std::any_of(leaves.begin(), leaves.end(), [&](const auto& leaf) { return missing(leaf->table); });
}

bool BdaPageTable::RecordUploads(const Context& context, VkCommandBuffer commands, std::uint32_t stream) const {
    bool recorded = false;
    for (const auto& leaf : leaves) recorded = recordUpload(context, commands, leaf->table, stream) || recorded;
    return recordUpload(context, commands, first, stream) || recorded;
}

std::optional<Abi::Range> BdaPageTable::Find(std::uint64_t address) const {
    if (address >= Abi::PageTableEnd) return std::nullopt;
    const auto leaf = find(static_cast<std::uint32_t>(address >> Abi::WindowShift));
    if (leaf == nullptr) return std::nullopt;
    const auto found = std::upper_bound(leaf->ranges.begin(), leaf->ranges.end(), address, [](std::uint64_t value, const Abi::Range& range) { return value < range.begin; });
    if (found == leaf->ranges.begin() || address >= std::prev(found)->end) return std::nullopt;
    return *std::prev(found);
}

std::shared_ptr<const BdaPageTable::Leaf> BdaPageTable::find(std::uint32_t window) const {
    const auto found = std::lower_bound(leaves.begin(), leaves.end(), window, [](const auto& leaf, std::uint32_t value) { return leaf->window < value; });
    return found != leaves.end() && (*found)->window == window ? *found : nullptr;
}

}
