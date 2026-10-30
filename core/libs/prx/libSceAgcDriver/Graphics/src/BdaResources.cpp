#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Execution/include/MemoryStats.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureService.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "ImageTableAbi.hpp"
#include "IntermediateRepresentation/IrMetadata/ShaderInfo.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <iomanip>
#include <sstream>

namespace AgcDriver::Graphics {

namespace {

// The page tables built per device, reused by a build whose ranges (guest ranges, device
// addresses, permissions) equal one of them: consecutive address-based builds map the same registered
// set through the same imports and mirrors, so the ~1200-entry table repeats. The shader only reads
// the table (its lookup keeps its state in function variables), and a batch in flight keeps its
// BdaResources, so sharing the buffer is safe. The cache holds the tables weakly: they never outlive
// the builds that own them (a static owner would destroy the buffer after its device, at exit or on
// device replacement, and would fail the tests' leak check), and with deferred lease release the
// previous builds are still in flight when the next one asks, which is what makes the hit rate.
// One table was enough while every build repeated the last one exactly; now the misaligned
// descriptor sub-ranges the GPU copies out of imports sit in pool staging buffers whose device
// addresses cycle through the few buffers the batches in flight leave free, and the three
// address-based programs of one queue interleave with the lease draws, so a table differs from the
// previous one and comes back a few builds later (69% misses in the wave-8 run, 4% before the GPU
// copies). Hence several recent tables, most recently used first, matched by a word hash before
// the byte compare. APS5_BDA_TABLE_CACHE_ENTRIES sets how many (default 8; 1 is the previous
// behaviour); APS5_NO_BDA_TABLE_CACHE=1 builds every table.
// A table built for (or found equal to) the cached address space's ranges carries the space's
// serial: a build served by that space alone binds it by the serial, with no hash and no compare.
struct TableEntry {
    std::uint64_t hash;
    // Shared with the address space's table map when the ranges came from it (no copy).
    std::shared_ptr<const std::vector<ShaderRecompiler::BdaAbi::Range>> ranges;
    std::weak_ptr<Buffer> buffer;
    std::uint64_t spaceSerial = 0;
};

struct TableCache {
    AgcDriver::Mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    std::list<TableEntry> entries;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    BdaResources::TableCacheStats classes;
};

bool sameRanges(const std::vector<ShaderRecompiler::BdaAbi::Range>& left, const std::vector<ShaderRecompiler::BdaAbi::Range>& right);

// APS5_PROFILE_DRAW: why the most recently used entry did not serve this build (see
// TableCacheStats). Under the cache mutex.
void classifyFirstEntry(TableCache& cache, std::uint64_t hash, const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& classes = cache.classes;
    if (cache.entries.empty()) {
        ++classes.firstEmpty;
        return;
    }
    const auto& first = cache.entries.front();
    if (first.hash == hash) {
        if (first.buffer.expired()) ++classes.firstExpired;
        else if (!sameRanges(*first.ranges, ranges)) ++classes.firstSameHash;
        return;
    }
    if (first.buffer.expired()) {
        ++classes.firstExpired;
        return;
    }
    const auto& firstRanges = *first.ranges;
    const auto common = std::min(firstRanges.size(), ranges.size());
    std::size_t at = 0;
    while (at < common && std::memcmp(&firstRanges[at], &ranges[at], sizeof(ranges[at])) == 0) ++at;
    const auto begin = at < ranges.size() ? ranges[at].begin : at < firstRanges.size() ? firstRanges[at].begin : 0;
    ++(begin < 0x10000000ull ? classes.firstDiffersLow : classes.firstDiffersHeap);
}

TableCache& Tables() {
    static TableCache cache;
    return cache;
}

bool tableCacheEnabled() {
    static const bool disabled = std::getenv("APS5_NO_BDA_TABLE_CACHE") != nullptr;
    return !disabled;
}

std::size_t tableCacheEntries() {
    static const std::size_t entries = [] {
        const char* value = std::getenv("APS5_BDA_TABLE_CACHE_ENTRIES");
        const auto parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 8ull;
        return static_cast<std::size_t>(std::clamp<unsigned long long>(parsed, 1, 64));
    }();
    return entries;
}

bool sameRanges(const std::vector<ShaderRecompiler::BdaAbi::Range>& left, const std::vector<ShaderRecompiler::BdaAbi::Range>& right) {
    return left.size() == right.size() && (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(left.front())) == 0);
}

// A 64-bit mix of the table's words: a table is ~1200 ranges of four words, and comparing the bytes
// against every held table would cost as much as building it.
// APS5_NO_LEAN_BDA_TABLES unset: the same words in four independent lanes (each mixed as before), combined at
// the end: the single chain of dependent multiplies took ~60 us for Yotei's ~9,400 ranges (best135). The value
// only keys this process's table cache, and a matching hash is always confirmed by the byte compare.
std::uint64_t hashRangesLanes(const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges) {
    static_assert(sizeof(ShaderRecompiler::BdaAbi::Range) == 4 * sizeof(std::uint64_t));
    std::uint64_t lanes[4] = {0x9e3779b97f4a7c15ull ^ ranges.size(), 0xc2b2ae3d27d4eb4full, 0x165667b19e3779f9ull, 0x27d4eb2f165667c5ull};
    for (const auto& range : ranges) {
        std::uint64_t words[4];
        std::memcpy(words, &range, sizeof(range));
        for (int lane = 0; lane < 4; ++lane) {
            lanes[lane] ^= words[lane];
            lanes[lane] *= 0xff51afd7ed558ccdull;
            lanes[lane] ^= lanes[lane] >> 33u;
        }
    }
    std::uint64_t hash = lanes[0];
    for (int lane = 1; lane < 4; ++lane) {
        hash ^= lanes[lane] + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
        hash *= 0xff51afd7ed558ccdull;
        hash ^= hash >> 33u;
    }
    return hash;
}

std::uint64_t hashRanges(const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges) {
    static_assert(sizeof(ShaderRecompiler::BdaAbi::Range) % sizeof(std::uint64_t) == 0);
    std::uint64_t hash = 0x9e3779b97f4a7c15ull ^ ranges.size();
    for (const auto& range : ranges) {
        std::uint64_t words[sizeof(range) / sizeof(std::uint64_t)];
        std::memcpy(words, &range, sizeof(range));
        for (const auto word : words) {
            hash ^= word;
            hash *= 0xff51afd7ed558ccdull;
            hash ^= hash >> 33u;
        }
    }
    return hash;
}

}

// APS5_PROFILE_DRAW: where a BDA build's time goes ([bda-build], 10 s): the fault buffer, the address
// ranges, and the table by outcome (space-serial hit, hash hit, miss with a new table buffer).
struct BdaBuildTimes {
    std::atomic<std::uint64_t> faults{0}, faultNs{0}, ranges{0}, rangesNs{0};
    std::atomic<std::uint64_t> spaceHits{0}, spaceHitNs{0}, hashHits{0}, hashHitNs{0}, misses{0}, missNs{0};
    // Miss sub-steps (ns): hash + first-entry classification, the candidate scan, the range index, the table
    // buffer, the copies into it, the cache entry (a copy of the ranges when they came from the cached space).
    std::atomic<std::uint64_t> hashNs{0}, scanNs{0}, indexNs{0}, bufferNs{0}, copyNs{0}, entryNs{0};
    std::atomic<long long> lastReport{0};
};
BdaBuildTimes& bdaBuildTimes() {
    static BdaBuildTimes times;
    return times;
}
bool bdaBuildProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}
std::uint64_t nsSince(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}
void reportBdaBuildTimes() {
    auto& t = bdaBuildTimes();
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = t.lastReport.load();
    if (nowMs - last < 10000 || !t.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto line = [](std::atomic<std::uint64_t>& count, std::atomic<std::uint64_t>& ns, char* out, std::size_t size) {
        const auto n = count.exchange(0, std::memory_order_relaxed);
        const auto total = ns.exchange(0, std::memory_order_relaxed);
        std::snprintf(out, size, "%llu x %.1f us = %.0f ms", static_cast<unsigned long long>(n), n != 0 ? static_cast<double>(total) / 1000.0 / static_cast<double>(n) : 0.0, static_cast<double>(total) / 1e6);
    };
    char fault[96], ranges[96], space[96], hashHit[96], miss[96];
    line(t.faults, t.faultNs, fault, sizeof(fault));
    line(t.ranges, t.rangesNs, ranges, sizeof(ranges));
    line(t.spaceHits, t.spaceHitNs, space, sizeof(space));
    line(t.hashHits, t.hashHitNs, hashHit, sizeof(hashHit));
    line(t.misses, t.missNs, miss, sizeof(miss));
    std::fprintf(stderr, "[bda-build] (10 s): fault buffers %s; address ranges %s; table: space hits %s, hash hits %s, misses %s\n", fault, ranges, space, hashHit, miss);
    const auto ms = [](std::atomic<std::uint64_t>& ns) { return static_cast<double>(ns.exchange(0, std::memory_order_relaxed)) / 1e6; };
    std::fprintf(stderr, "[bda-build] (10 s): table build parts (ms): hash %.0f, candidate scan %.0f, index %.0f, buffer %.0f, copies %.0f, cache entry %.0f\n", ms(t.hashNs), ms(t.scanNs), ms(t.indexNs), ms(t.bufferNs), ms(t.copyNs), ms(t.entryNs));
}

BdaResources::BdaResources(const Context& context) : textureService(context.textureService) {
    static_assert(std::endian::native == std::endian::little);
    Require(ShaderRecompiler::BdaAbi::FaultBufferBytes <= context.limits.maxStorageBufferRange, "BDA fault buffer exceeds storage buffer range limit");
    const MemoryStats::OwnerScope memStatsOwner(MemoryStats::Owner::Bda);
    const auto faultStart = bdaBuildProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    fault = std::make_unique<Buffer>(context, ShaderRecompiler::BdaAbi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memset(fault->Bytes().data(), 0, ShaderRecompiler::BdaAbi::WrittenListWord * sizeof(std::uint32_t));
    std::memset(fault->Bytes().data() + ShaderRecompiler::BdaAbi::SrtGuardWord * sizeof(std::uint32_t), 0, sizeof(std::uint32_t));
    if (bdaBuildProfiled()) {
        bdaBuildTimes().faults.fetch_add(1, std::memory_order_relaxed);
        bdaBuildTimes().faultNs.fetch_add(nsSince(faultStart), std::memory_order_relaxed);
    }
}

BdaResources::BdaResources(const Context& context, const GuestBufferMemory& memory) : BdaResources(context) {
    const bool profiled = bdaBuildProfiled();
    // APS5_NO_LEAN_BDA_TABLES=1 (runtime switch): copy the ranges into the table cache and hash them in one
    // chain, as before (see hashRangesLanes and GuestBufferMemory's range-list pool).
    const bool lean = !RuntimeSwitches::Set(RuntimeSwitch::NoLeanBdaTables);
    const auto rangesStart = profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto cached = memory.CachedAddressTable();
    std::vector<ShaderRecompiler::BdaAbi::Range> built;
    if (!cached.has_value()) built = memory.AddressRanges();
    const auto tableStart = profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (profiled) {
        bdaBuildTimes().ranges.fetch_add(1, std::memory_order_relaxed);
        bdaBuildTimes().rangesNs.fetch_add(nsSince(rangesStart), std::memory_order_relaxed);
    }
    // Which way the table came, for [bda-build]: 0 space hit, 1 hash hit, 2 miss.
    int outcome = 2;
    struct NoteOutcome {
        bool profiled;
        const int& outcome;
        std::chrono::steady_clock::time_point start;
        ~NoteOutcome() {
            if (!profiled) return;
            auto& t = bdaBuildTimes();
            const auto ns = nsSince(start);
            (outcome == 0 ? t.spaceHits : outcome == 1 ? t.hashHits : t.misses).fetch_add(1, std::memory_order_relaxed);
            (outcome == 0 ? t.spaceHitNs : outcome == 1 ? t.hashHitNs : t.missNs).fetch_add(ns, std::memory_order_relaxed);
            reportBdaBuildTimes();
        }
    } noteOutcome{profiled, outcome, tableStart};
    const auto& ranges = cached.has_value() ? *cached->ranges : built;
    const auto serial = cached.has_value() ? cached->serial : 0;
    Require(ranges.size() <= std::numeric_limits<std::uint32_t>::max(), "BDA table range count overflow");
    Require(ranges.size() <= (std::numeric_limits<std::size_t>::max() - sizeof(ShaderRecompiler::BdaAbi::Header)) / sizeof(ShaderRecompiler::BdaAbi::Range), "BDA table size overflow");
    // The range index (BdaAbi::IndexedTable) after the ranges; APS5_NO_BDA_INDEX=1 leaves it out.
    static const bool indexDisabled = std::getenv("APS5_NO_BDA_INDEX") != nullptr;
    // [bda-build] index part: on this tree the index is built before the cache lookup (every build, not only misses).
    const auto indexStart = profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto index = indexDisabled ? std::vector<std::uint32_t>{} : ShaderRecompiler::BdaAbi::BuildRangeIndex(ranges.data(), ranges.size());
    if (profiled) bdaBuildTimes().indexNs.fetch_add(nsSince(indexStart), std::memory_order_relaxed);
    tableBytes = sizeof(ShaderRecompiler::BdaAbi::Header) + ranges.size() * sizeof(ShaderRecompiler::BdaAbi::Range) + index.size() * sizeof(std::uint32_t);
    Require(tableBytes <= context.limits.maxStorageBufferRange && ShaderRecompiler::BdaAbi::FaultBufferBytes <= context.limits.maxStorageBufferRange, "BDA descriptors exceed storage buffer range limit");
    auto& cache = Tables();
    std::uint64_t hash = 0;
    if (tableCacheEnabled()) {
        std::lock_guard lock(cache.mutex);
        // Another device's tables are not this one's (their buffers are dead or foreign).
        if (cache.device != context.device) {
            cache.entries.clear();
            cache.device = context.device;
        }
        if (serial != 0) {
            for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
                if (it->spaceSerial != serial) continue;
                if (auto shared = it->buffer.lock(); shared != nullptr) {
                    ++cache.hits;
                    ++cache.classes.spaceTables;
                    outcome = 0;
                    table = std::move(shared);
                    tableRanges = it->ranges;
                    cache.entries.splice(cache.entries.begin(), cache.entries, it);
                    return;
                }
                cache.entries.erase(it);
                break;
            }
        }
        const auto hashStart = profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        hash = lean ? hashRangesLanes(ranges) : hashRanges(ranges);
        classifyFirstEntry(cache, hash, ranges);
        const auto scanStart = profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (profiled) bdaBuildTimes().hashNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(scanStart - hashStart).count()), std::memory_order_relaxed);
        struct NoteScan {
            bool profiled;
            std::chrono::steady_clock::time_point start;
            ~NoteScan() {
                if (profiled) bdaBuildTimes().scanNs.fetch_add(nsSince(start), std::memory_order_relaxed);
            }
        } noteScan{profiled, scanStart};
        for (auto it = cache.entries.begin(); it != cache.entries.end();) {
            // Compare the hash before taking a strong reference: locking a non-matching entry's
            // buffer could make this thread its last owner (a concurrent reap releasing it) and
            // run the pool release under the cache mutex; expired() takes no reference.
            if (it->hash != hash) {
                if (it->buffer.expired()) {
                    it = cache.entries.erase(it);
                } else {
                    ++it;
                }
                continue;
            }
            auto shared = it->buffer.lock();
            if (shared == nullptr) {
                // Every build that owned the table is gone, and its buffer with it.
                it = cache.entries.erase(it);
                continue;
            }
            if (sameRanges(*it->ranges, ranges)) {
                ++cache.hits;
                outcome = 1;
                // An equal table of an earlier space (a rebuild that mapped the same ranges the
                // same way): the next build of this space binds it by the serial.
                if (serial != 0) it->spaceSerial = serial;
                table = std::move(shared);
                tableRanges = it->ranges;
                cache.entries.splice(cache.entries.begin(), cache.entries, it);
                return;
            }
            ++it;
        }
    }
    Require(ShaderRecompiler::BdaAbi::RangesOrdered(ranges.data(), ranges.size()), "BDA table ranges do not ascend by begin");
    for (const auto& range : ranges) Require((range.permissions & ShaderRecompiler::BdaAbi::Write) == 0 || range.end <= ShaderRecompiler::BdaAbi::WrittenAddressLimit, "a writable BDA range ends past the addresses the written-page record can hold");
    const MemoryStats::OwnerScope memStatsOwner(MemoryStats::Owner::Bda);
    const auto bufferStart = profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    table = std::make_shared<Buffer>(context, tableBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto copyStart = profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (profiled) bdaBuildTimes().bufferNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(copyStart - bufferStart).count()), std::memory_order_relaxed);
    const ShaderRecompiler::BdaAbi::Header header{ShaderRecompiler::BdaAbi::Version, static_cast<std::uint32_t>(ranges.size()), sizeof(ShaderRecompiler::BdaAbi::Range), index.empty() ? 0u : ShaderRecompiler::BdaAbi::IndexedTable};
    std::memcpy(table->Bytes().data(), &header, sizeof(header));
    if (!ranges.empty()) std::memcpy(table->Bytes().data() + sizeof(header), ranges.data(), ranges.size() * sizeof(ranges.front()));
    if (!index.empty()) std::memcpy(table->Bytes().data() + sizeof(header) + ranges.size() * sizeof(ranges.front()), index.data(), index.size() * sizeof(index.front()));
    const auto entryStart = profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (profiled) bdaBuildTimes().copyNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(entryStart - copyStart).count()), std::memory_order_relaxed);
    struct NoteEntry {
        bool profiled;
        std::chrono::steady_clock::time_point start;
        ~NoteEntry() {
            if (profiled) bdaBuildTimes().entryNs.fetch_add(nsSince(start), std::memory_order_relaxed);
        }
    } noteEntry{profiled, entryStart};
    if (tableCacheEnabled()) {
        std::lock_guard lock(cache.mutex);
        ++cache.misses;
        if (cache.device != context.device) {
            cache.entries.clear();
            cache.device = context.device;
        }
        auto entryRanges = cached.has_value() && lean && cached->owner != nullptr ? cached->owner : std::make_shared<const std::vector<ShaderRecompiler::BdaAbi::Range>>(cached.has_value() ? ranges : std::move(built));
        tableRanges = entryRanges;
        cache.entries.push_front({hash, std::move(entryRanges), table, serial});
        while (cache.entries.size() > tableCacheEntries()) cache.entries.pop_back();
    }
}

BdaResources::TableCacheStats BdaResources::TableCacheCounters() {
    auto& cache = Tables();
    std::lock_guard lock(cache.mutex);
    auto stats = cache.classes;
    stats.hits = cache.hits;
    stats.misses = cache.misses;
    stats.held = cache.entries.size();
    return stats;
}

VkDescriptorBufferInfo BdaResources::Table() const {
    Require(table != nullptr, "BDA page table was not requested");
    return {table->Handle(), 0, tableBytes};
}

std::shared_ptr<const std::vector<ShaderRecompiler::BdaAbi::Range>> BdaResources::Ranges() const {
    return tableRanges;
}

std::shared_ptr<const std::vector<ShaderRecompiler::BdaAbi::Range>> BdaResources::RangesOf(const GuestBufferMemory& memory) {
    const auto cached = memory.CachedAddressTable();
    if (cached.has_value() && cached->owner != nullptr) return cached->owner;
    if (cached.has_value() && cached->serial != 0 && tableCacheEnabled()) {
        auto& cache = Tables();
        std::lock_guard lock(cache.mutex);
        for (const auto& entry : cache.entries) {
            if (entry.spaceSerial == cached->serial && entry.ranges != nullptr) return entry.ranges;
        }
    }
    return std::make_shared<const std::vector<ShaderRecompiler::BdaAbi::Range>>(cached.has_value() ? *cached->ranges : memory.AddressRanges());
}

VkDescriptorBufferInfo BdaResources::Fault() const {
    return {fault->Handle(), 0, ShaderRecompiler::BdaAbi::FaultBufferBytes};
}

namespace {
std::atomic<bool> loopGuardTripped{false};
}

bool LoopGuardTripped() {
    return loopGuardTripped.load(std::memory_order_relaxed);
}

void BdaResources::CheckFault(std::uint64_t shaderAddress) const {
    markWrittenPages();
    ShaderRecompiler::BdaAbi::Fault report{};
    std::memcpy(&report, fault->Bytes().data(), sizeof(report));
    if (report.state == ShaderRecompiler::BdaAbi::FaultState::Empty) {
        Require(static_cast<std::uint32_t>(report.reason) == 0 && report.address == 0 && report.bytes == 0 && report.stage == 0 && report.instruction == 0 && report.otherFaults == 0, "BDA fault record has data without publication");
        checkSrtGuard();
        return;
    }
    Require(report.state == ShaderRecompiler::BdaAbi::FaultState::Ready && report.otherFaults <= 1, "incomplete or invalid BDA fault record");
    Require(report.reason != ShaderRecompiler::BdaAbi::FaultReason::InvalidRectangle, "rect-list requires finite nondegenerate axis-aligned positions with equal positive W");
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::LoopLimit) {
        // APS5_LOOP_GUARD: the shader left a loop that ran past the guard; the dispatch result is kept.
        loopGuardTripped.store(true, std::memory_order_relaxed);
        std::fprintf(stderr, "[gpu] loop guard: the loop exit at pc 0x%x of shader 0x%llx ran past %u evaluations\n", report.instruction, static_cast<unsigned long long>(report.address), report.bytes);
        std::memset(fault->Bytes().data(), 0, fault->Bytes().size());
        return;
    }
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::UnnormalizedWrap) {
        Require(report.address == 0u, "invalid unnormalized sampler fault payload");
        std::ostringstream message;
        message << "unnormalized sampler: shader ";
        if (shaderAddress != 0u) message << "0x" << std::hex << shaderAddress << std::dec;
        else message << "address unavailable";
        message << " stage " << report.stage << " pc 0x" << std::hex << (report.instruction & 0x7fffffffu) << std::dec
            << " axis " << (report.instruction >> 31u) << " coordinate " << std::bit_cast<float>(report.bytes) << " outside [0.5, size - 0.5]";
        throw std::runtime_error(message.str());
    }
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::DirectImage) {
        const auto resource = ShaderRecompiler::BdaAbi::DirectImageFaultResource(report.bytes);
        Require(report.address == 0u && ShaderRecompiler::BdaAbi::DirectImageFaultReason(report.bytes) == ShaderRecompiler::BdaAbi::DirectImageDimension && resource < ShaderRecompiler::ShaderInfo::MaxImages, "invalid direct image fault payload");
        std::ostringstream message;
        message << "direct image: shader ";
        if (shaderAddress != 0u) message << "0x" << std::hex << shaderAddress << std::dec;
        else message << "address unavailable";
        message << " stage " << report.stage << " pc 0x" << std::hex << report.instruction << std::dec << " resource " << resource << ": dimension";
        throw std::runtime_error(message.str());
    }
    const auto stageTable = std::find_if(imageTables.begin(), imageTables.end(), [&](const ImageTableFaults& candidate) { return candidate.stage == report.stage; });
    const auto* stageTables = stageTable != imageTables.end() ? &*stageTable : imageTables.size() == 1 ? &imageTables.front() : nullptr;
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::ImageTable) {
        std::ostringstream message;
        message << "image table: shader 0x" << std::hex << (stageTables != nullptr ? stageTables->shader : 0u) << std::dec << " stage " << report.stage << " pc 0x" << std::hex << report.instruction;
        if (report.address == ShaderRecompiler::ImageTableAbi::UnknownDescriptorAddress) message << " selected a descriptor whose address was not captured";
        else message << " selected the descriptor at 0x" << report.address;
        message << std::dec;
        if (stageTables != nullptr && report.bytes < stageTables->poison.size()) {
            const auto& poison = stageTables->poison[report.bytes];
            if (poison.dwordCount != 0) {
                message << " (words";
                for (std::uint32_t index = 0; index < poison.dwordCount && index < poison.words.size(); ++index) message << ' ' << std::hex << std::setw(8) << std::setfill('0') << poison.words[index];
                message << std::dec << ")";
            }
            message << ": " << ShaderRecompiler::ImageTableAbi::PoisonReasonName(static_cast<ShaderRecompiler::ImageTableAbi::PoisonReason>(poison.reason));
            if (report.bytes < stageTables->messages.size() && !stageTables->messages[report.bytes].empty()) message << ": " << stageTables->messages[report.bytes];
        } else {
            message << ": poison entry " << report.bytes;
        }
        throw std::runtime_error(message.str());
    }
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::BindlessRefused && textureService != nullptr) {
        if (const auto rejection = textureService->Rejection(report.bytes + 1u); rejection.has_value()) {
            std::ostringstream message;
            message << "image table: shader 0x" << std::hex << shaderAddress << std::dec << " stage " << report.stage << " pc 0x" << std::hex << report.instruction << " selected the descriptor at 0x" << report.address << std::dec << " through the bindless heap " << *rejection;
            throw std::runtime_error(message.str());
        }
    }
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::BindlessMiss || report.reason == ShaderRecompiler::BdaAbi::FaultReason::BindlessRefused) {
        Require(report.reason == ShaderRecompiler::BdaAbi::FaultReason::BindlessRefused || report.address == 0u, "invalid bindless image fault payload");
        std::ostringstream message;
        message << "bindless images: shader ";
        if (shaderAddress != 0u) message << "0x" << std::hex << shaderAddress << std::dec;
        else message << "address unavailable";
        message << " stage " << report.stage << " pc 0x" << std::hex << report.instruction << std::dec;
        if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::BindlessMiss) message << " missed in the device image table while its miss ring was full (" << report.bytes << " records), so the descriptor was not recorded";
        else message << " selected the refused device image table entry " << report.bytes;
        if (report.address != 0u) message << " (descriptor at 0x" << std::hex << report.address << std::dec << ")";
        throw std::runtime_error(message.str());
    }
    std::ostringstream message;
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::UnlinkedCall) {
        message << "shader stage " << report.stage << " called address 0x" << std::hex << report.address << " at program counter 0x" << report.instruction << ", which no linked callee of that call matches";
        throw std::runtime_error(message.str());
    }
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::WindowLaneSpill) {
        message << "shader stage " << report.stage << ": a relative write stored " << report.bytes << " in lane " << (report.address >> 32u) << " of vector window slot " << (report.address & 0xffffffffu) << ", which holds a scalar spilled with v_writelane_b32 that a later v_readlane_b32 was replaced with";
        throw std::runtime_error(message.str());
    }
    message << "BDA access failed: address=0x" << std::hex << report.address << " instruction=0x" << report.instruction << std::dec << " bytes=" << report.bytes << " stage=" << report.stage << " reason=" << static_cast<std::uint32_t>(report.reason);
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::Permission && table != nullptr) {
        const auto bytes = table->Bytes();
        ShaderRecompiler::BdaAbi::Header header{};
        std::memcpy(&header, bytes.data(), sizeof(header));
        for (std::uint32_t index = 0; index < header.count; ++index) {
            ShaderRecompiler::BdaAbi::Range range{};
            std::memcpy(&range, bytes.data() + sizeof(header) + index * sizeof(range), sizeof(range));
            if (report.address < range.begin || report.address >= range.end) continue;
            message << std::hex << "; the store hit 0x" << range.begin << "+0x" << range.end - range.begin << ", read-only in the BDA table: stores through GPU-selected descriptors and FLAT/GLOBAL addresses reach only writable ranges imported in place, not ones served by a mirror or a copy (past APS5_HOST_IMPORT_MIB, or refused by the driver)";
            break;
        }
    }
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::Unaligned) {
        if (report.otherFaults == 0) throw BdaUnalignedAccess(message.str(), report.instruction);
        message << "; a fault of another kind was recorded in the same dispatch";
    }
    throw std::runtime_error(message.str());
}

}

namespace AgcDriver::Graphics {

void BdaResources::SetImageTables(std::vector<ImageTableFaults> tables, std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges) {
    imageTables = std::move(tables);
    imageTableRanges = std::move(ranges);
}

void BdaResources::SetSrtGuards(std::vector<SrtGuardFaults> guards) {
    srtGuards = std::move(guards);
}

void BdaResources::checkSrtGuard() const {
    namespace Abi = ShaderRecompiler::BdaAbi;
    std::uint32_t guard = 0;
    std::memcpy(&guard, fault->Bytes().data() + Abi::SrtGuardWord * sizeof(std::uint32_t), sizeof(guard));
    if (guard == 0) return;
    const auto stage = guard >> Abi::SrtGuardStageShift;
    const auto record = guard & ((1u << Abi::SrtGuardStageShift) - 1u);
    const auto found = std::find_if(srtGuards.begin(), srtGuards.end(), [&](const SrtGuardFaults& candidate) { return candidate.stage == stage; });
    const auto* guards = found != srtGuards.end() ? &*found : srtGuards.size() == 1 ? &srtGuards.front() : nullptr;
    Require(guards != nullptr && record + 2u < guards->flattenedSrt.size(), "a guarded SRT read faulted without its poison record");
    const auto& words = guards->flattenedSrt;
    const auto address = static_cast<std::uint64_t>(words[record + 1u]) | (static_cast<std::uint64_t>(words[record + 2u]) << 32u);
    std::ostringstream message;
    message << "BDA access failed: address=0x" << std::hex << address << " instruction=0x" << words[record] << std::dec << " bytes=4 stage=" << stage << " reason=" << static_cast<std::uint32_t>(Abi::FaultReason::Unmapped);
    throw std::runtime_error(message.str());
}

void BdaResources::markWrittenPages() const {
    namespace Abi = ShaderRecompiler::BdaAbi;
    auto* words = reinterpret_cast<std::uint32_t*>(fault->Bytes().data());
    Require(words[Abi::WrittenOverflowWord] == 0, "more than " + std::to_string(Abi::WrittenPageSlots) + " chunks of " + std::to_string(Abi::WrittenChunkPages) + " pages stored to through the BDA table in one use are not implemented");
    const auto count = words[Abi::WrittenCountWord];
    if (count == 0) return;
    Require(Abi::ForEachWrittenRange(words, [&](std::uint64_t begin, std::size_t bytes) {
        const auto end = begin + bytes;
        for (const auto& [base, size] : imageTableRanges) {
            if (size != 0 && begin < base + size && base < end) {
                std::ostringstream message;
                message << "image table at 0x" << std::hex << base << " is written by its own dispatch";
                Require(false, message.str());
            }
        }
        GuestMemory::MarkPossiblyWritten(begin, bytes);
    }), "invalid BDA written-page record");
    for (std::uint32_t claim = 0; claim < count; ++claim) {
        const auto slot = words[Abi::WrittenListWord + claim];
        words[Abi::WrittenSlotsWord + 2 * slot] = 0;
        words[Abi::WrittenSlotsWord + 2 * slot + 1] = 0;
    }
    words[Abi::WrittenCountWord] = 0;
}

}
