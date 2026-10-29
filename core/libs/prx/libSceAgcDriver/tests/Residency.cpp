#include "prx/libSceAgcDriver/Graphics/include/Residency.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<std::int64_t> heapBytes{0};
constexpr std::size_t HeapHeader = 16;

void* CountedAllocate(std::size_t bytes) {
    auto* block = static_cast<std::byte*>(std::malloc(bytes + HeapHeader));
    if (block == nullptr) throw std::bad_alloc();
    *reinterpret_cast<std::size_t*>(block) = bytes;
    heapBytes += static_cast<std::int64_t>(bytes);
    return block + HeapHeader;
}

void CountedFree(void* pointer) noexcept {
    if (pointer == nullptr) return;
    auto* block = static_cast<std::byte*>(pointer) - HeapHeader;
    heapBytes -= static_cast<std::int64_t>(*reinterpret_cast<std::size_t*>(block));
    std::free(block);
}

}

void* operator new(std::size_t bytes) { return CountedAllocate(bytes); }
void* operator new[](std::size_t bytes) { return CountedAllocate(bytes); }
void operator delete(void* pointer) noexcept { CountedFree(pointer); }
void operator delete[](void* pointer) noexcept { CountedFree(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { CountedFree(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { CountedFree(pointer); }

namespace {

using namespace AgcDriver::Graphics;

int failures = 0;
std::string scope;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    if (failures < 25) std::fprintf(stderr, "FAIL [%s]: %s\n", scope.c_str(), what.c_str());
    ++failures;
}

constexpr std::uint64_t BasePage = 0x4000;
constexpr std::size_t FakeStreams = 3;
constexpr std::uint32_t ChunkSlots = 4;

std::uint64_t Address(std::uint64_t page, std::uint32_t subpage = 0) {
    return (page << ResidencyPageShift) + (std::uint64_t{subpage} << ResidencySubpageShift);
}

ResidencyToken On(std::size_t stream, std::uint64_t value) {
    ResidencyToken token{};
    token[stream] = value;
    return token;
}

ResidencyMask Run(std::uint32_t first, std::uint32_t count) {
    return static_cast<ResidencyMask>(((1u << count) - 1u) << first);
}

using Contents = std::array<std::uint64_t, ResidencySubpages>;

std::string Name(std::uint64_t page) {
    return "page " + std::to_string(page - BasePage);
}

class FakeGpu final : public IResidencyDevice {
public:
    enum class Kind : std::uint8_t { CopyIn, WriteBack, Read, Write };

    struct Op {
        Kind kind = Kind::Read;
        std::size_t stream = 0;
        std::uint64_t batch = 0;
        std::uint64_t seq = 0;
        bool barrier = false;
        bool executed = false;
        bool title = false;
        std::uint64_t page = 0;
        bool mirror = false;
        MirrorSlot slot;
        std::shared_ptr<Contents> import;
        ResidencyMask mask = 0;
        std::uint64_t value = 0;
    };

    struct Batch {
        std::uint64_t value = 0;
        std::vector<std::size_t> ops;
        std::size_t done = 0;
        bool submitted = false;
    };

    struct Stream {
        std::uint64_t last = 0;
        std::uint64_t completed = 0;
        std::deque<Batch> batches;
    };

    struct ImportPage {
        std::shared_ptr<Contents> storage = std::make_shared<Contents>();
        ResidencyImport attributes{true, true, true};
        std::uint64_t stamp = 0;
        std::uint64_t group = 0;
    };

    std::size_t caller = 0;
    std::uint32_t budgetChunks = ~std::uint32_t{0};
    ResidencyManager* manager = nullptr;
    std::map<std::uint64_t, ImportPage> imports;
    std::map<std::uint64_t, bool> armed;
    std::map<std::uint64_t, ResidencyRefusal> guardRefused;
    std::map<std::uint64_t, std::uint64_t> guardCalls;
    bool sampleStaleAtGuard = false;
    std::vector<bool> staleAtGuard;
    std::set<std::uint64_t> reported;
    std::set<std::uint64_t> accessUnguarded;
    std::uint64_t protections = 0;
    std::map<std::uint64_t, std::uint64_t> collected;
    std::uint64_t collectedPages = 0;
    std::vector<std::vector<Contents>> mirrors;
    std::vector<bool> live;
    std::vector<std::uint64_t> slotOwner;
    std::uint64_t generation = 1;
    std::uint64_t waits = 0;
    std::uint64_t faults = 0;
    std::uint64_t mirrorReads = 0;
    std::uint64_t importReads = 0;
    std::vector<ResidencyCopy> copiedIn;
    std::vector<ResidencyCopy> writtenBack;
    std::uint64_t measuredBudget = ~std::uint64_t{0};
    std::uint64_t guestBytes = 0;
    std::vector<ResidencyRange> wholeImports;
    std::uint64_t importsGeneration = 1;
    std::uint64_t collectEpoch = 0;
    std::uint64_t importQueries = 0;
    std::map<std::uint64_t, ResidencyToken> modelPins;
    std::uint64_t hostImports = 0;
    std::map<std::uint64_t, std::uint64_t> hostImported;
    std::map<std::uint64_t, std::uint64_t> hostSettled;
    std::array<Stream, FakeStreams> streams{};
    std::vector<Op> ops;

    bool Reached(const ResidencyToken& token) override {
        for (std::size_t stream = 0; stream < FakeStreams; ++stream) {
            if (token[stream] > streams[stream].completed) return false;
        }
        for (std::size_t stream = FakeStreams; stream < ResidencyStreams; ++stream) {
            if (token[stream] != 0) return false;
        }
        return true;
    }

    void Wait(const ResidencyToken& token) override {
        ++waits;
        WaitFor(token);
    }

    ResidencyToken Newest() override {
        ResidencyToken token{};
        for (std::size_t stream = 0; stream < FakeStreams; ++stream) token[stream] = streams[stream].last;
        return token;
    }

    ResidencyToken Current() override { return CallerToken(caller); }

    ResidencyImport Import(std::uint64_t page) override {
        ++importQueries;
        const auto found = imports.find(page);
        if (found == imports.end()) return {};
        return found->second.attributes;
    }

    std::vector<ResidencyRange> Aliases(std::uint64_t address, std::uint64_t bytes) override {
        std::vector<ResidencyRange> ranges;
        if (bytes == 0) return ranges;
        const auto first = address >> ResidencyPageShift;
        const auto last = (address + bytes - 1) >> ResidencyPageShift;
        for (auto page = first; page <= last; ++page) {
            const auto start = std::max(address, Address(page));
            const auto end = std::min(address + bytes, Address(page + 1));
            for (const auto alias : AliasesOf(page)) {
                if (alias != page) ranges.push_back({Address(alias) + (start - Address(page)), end - start});
            }
        }
        return ranges;
    }

    std::uint64_t Generation() override { return generation; }

    void ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) override {
        for (std::size_t k = 0; k < generations.size(); ++k) {
            if (generations[k] == ResidencyUnchecked) {
                changed[k] = 0;
                continue;
            }
            changed[k] = generations[k] == 0 || imports[firstPage + k].stamp > generations[k] ? 1 : 0;
        }
    }

    std::uint64_t AllocateChunk(std::uint32_t chunk, std::uint32_t slots) override {
        const auto liveCount = static_cast<std::uint32_t>(std::count(live.begin(), live.end(), true));
        Expect(liveCount < budgetChunks, "chunk " + std::to_string(chunk) + " allocated with " + std::to_string(liveCount) + " live chunks and a budget of " + std::to_string(budgetChunks));
        if (chunk >= live.size()) {
            live.resize(chunk + 1, false);
            mirrors.resize(chunk + 1);
            slotOwner.resize((chunk + 1) * slots, 0);
        }
        Expect(!live[chunk], "chunk " + std::to_string(chunk) + " allocated twice");
        live[chunk] = true;
        Contents poison{};
        poison.fill(0xdeadbeefull);
        mirrors[chunk].assign(slots, poison);
        for (std::uint32_t slot = 0; slot < slots; ++slot) slotOwner[chunk * slots + slot] = 0;
        return 0x700000000000ull + (std::uint64_t{chunk} << 32);
    }

    void ReleaseChunk(std::uint32_t chunk) override {
        Expect(chunk < live.size() && live[chunk], "chunk " + std::to_string(chunk) + " released while not live");
        for (const auto& op : inflight()) {
            if (op.mirror && op.slot.chunk == chunk) Expect(false, "chunk " + std::to_string(chunk) + " released with " + Name(op.page) + " work in flight");
        }
        if (chunk < live.size()) live[chunk] = false;
    }

    ResidencyToken CopyIn(std::span<const ResidencyCopy> copies) override {
        for (const auto& copy : copies) {
            checkSlot(copy);
            for (const auto& op : inflight()) {
                if (op.mirror && op.slot == copy.mirror && op.page != copy.page)
                    Expect(false, "slot " + std::to_string(copy.mirror.chunk) + ":" + std::to_string(copy.mirror.slot) + " reused for " + Name(copy.page) + " while " + Name(op.page) + " work on it is in flight");
            }
            if (imports[copy.page].attributes.cpuWritable) Expect(armed[imports[copy.page].group] || manager->HotResident(copy.page), "copy into the mirror of CPU-writable " + Name(copy.page) + " recorded while neither its guard nor write tracking catches later CPU stores");
            auto& owner = slotOwner[copy.mirror.chunk * ChunkSlots + copy.mirror.slot];
            if (owner != copy.page) {
                modelPins.erase(copy.page);
                owner = copy.page;
            }
            Op op;
            op.kind = Kind::CopyIn;
            op.barrier = true;
            op.page = copy.page;
            op.mirror = true;
            op.slot = copy.mirror;
            op.import = imports[copy.page].storage;
            op.mask = MaskOf(copy);
            record(op);
            copiedIn.push_back(copy);
        }
        return CallerToken(caller);
    }

    ResidencyToken WriteBack(std::span<const ResidencyCopy> copies) override {
        for (const auto& copy : copies) {
            checkSlot(copy);
            Op op;
            op.kind = Kind::WriteBack;
            op.barrier = true;
            op.page = copy.page;
            op.mirror = true;
            op.slot = copy.mirror;
            op.import = imports[copy.page].storage;
            op.mask = MaskOf(copy);
            record(op);
            writtenBack.push_back(copy);
        }
        return CallerToken(caller);
    }

    void HostImport(std::uint64_t page) { hostImported[page] = ++hostImports; }

    std::uint64_t ImportStamp() override { return hostImports; }

    bool ImportSettled(std::uint64_t page) override { return hostSettled[page] >= hostImported[page]; }

    void SettleImport(std::uint64_t page, std::uint64_t stamp) override { hostSettled[page] = std::max(hostSettled[page], stamp); }

    ResidencyRefusal Guard(std::uint64_t page) override {
        Expect(imports[page].attributes.section != 0 || (manager != nullptr && manager->ImportSettled(page)), "a write guard armed on private " + Name(page) + " before its import settled");
        Expect(imports[page].attributes.section != 0 || ImportSettled(page), "a write guard armed on private " + Name(page) + " before its newest host import settled");
        ++protections;
        ++guardCalls[page];
        if (const auto refusal = guardRefused.find(page); refusal != guardRefused.end()) return refusal->second;
        armed[imports[page].group] = true;
        return ResidencyRefusal::None;
    }

    void Unguard(std::span<const std::uint64_t> pages) override {
        protections += pages.size();
        for (const auto page : pages) armed[imports[page].group] = false;
    }

    ResidencyRefusal GuardAccess(std::uint64_t page, ResidencyMask) override {
        if (sampleStaleAtGuard) staleAtGuard.push_back(manager->AnyStaleOwned());
        Expect(imports[page].attributes.section != 0 || (manager != nullptr && manager->ImportSettled(page)), "an access guard armed on private " + Name(page) + " before its import settled");
        Expect(imports[page].attributes.section != 0 || ImportSettled(page), "an access guard armed on private " + Name(page) + " before its newest host import settled");
        ++protections;
        const auto refusal = guardRefused.find(page);
        return refusal == guardRefused.end() ? ResidencyRefusal::None : refusal->second;
    }

    void UnguardAccess(std::uint64_t page) override {
        ++protections;
        accessUnguarded.insert(page);
    }

    std::vector<ResidencyRange> WholeImports(std::uint64_t& generation) override {
        if (generation == importsGeneration) return {};
        generation = importsGeneration;
        return wholeImports;
    }

    std::uint64_t Budget() override { return measuredBudget; }

    void CollectWrites(std::uint64_t firstPage, std::uint64_t count) override {
        std::vector<std::uint64_t> written;
        for (auto page = firstPage; page < firstPage + count; ++page) {
            auto& seen = collected[page];
            if (imports[page].stamp <= seen) continue;
            seen = imports[page].stamp;
            written.push_back(page);
        }
        collectedPages += count;
        manager->QueueCpuWritten(written);
    }

    std::uint64_t GuestBytes() override { return guestBytes; }

    std::optional<ResidencyReuseStamp> ReuseStamp() override {
        const auto& batches = streams[caller].batches;
        if (collectEpoch == 0 || batches.empty() || batches.back().submitted) return std::nullopt;
        return ResidencyReuseStamp{generation, collectEpoch, importsGeneration, hostImports, batches.back().value};
    }

    static ResidencyMask MaskOf(const ResidencyCopy& copy) {
        return SubpageMask(copy.page, Address(copy.page) + copy.offset, copy.bytes);
    }

    std::vector<std::uint64_t> AliasesOf(std::uint64_t page) {
        std::vector<std::uint64_t> result;
        const auto group = imports[page].group;
        for (const auto& [other, entry] : imports) {
            if (entry.group == group) result.push_back(other);
        }
        return result;
    }

    void Alias(std::uint64_t page, std::uint64_t alias) {
        auto& owner = imports[page];
        auto& entry = imports[alias];
        entry.storage = owner.storage;
        entry.group = owner.group;
        owner.attributes.aliased = true;
        owner.attributes.section = owner.group + 1;
        owner.attributes.offset = 0;
        entry.attributes = owner.attributes;
    }

    ResidencyToken CallerToken(std::size_t stream) {
        auto& state = streams[stream];
        if (state.batches.empty() || state.batches.back().submitted) state.batches.push_back(Batch{++state.last, {}, 0, false});
        return On(stream, state.batches.back().value);
    }

    void Submit(std::size_t stream) {
        auto& state = streams[stream];
        if (!state.batches.empty()) state.batches.back().submitted = true;
        completeFinished(stream);
    }

    void SubmitAll() {
        for (std::size_t stream = 0; stream < FakeStreams; ++stream) Submit(stream);
    }

    bool Step(std::size_t stream) {
        auto& state = streams[stream];
        for (auto& batch : state.batches) {
            if (batch.done == batch.ops.size()) continue;
            if (!batch.submitted) return false;
            auto& op = ops[batch.ops[batch.done]];
            if (!executable(op)) return false;
            execute(op);
            ++batch.done;
            completeFinished(stream);
            return true;
        }
        return false;
    }

    bool StepAny() {
        bool progress = false;
        for (std::size_t stream = 0; stream < FakeStreams; ++stream) progress = Step(stream) || progress;
        return progress;
    }

    void Drain() {
        SubmitAll();
        while (StepAny()) {}
        for (std::size_t stream = 0; stream < FakeStreams; ++stream) Expect(streams[stream].batches.empty(), "stream " + std::to_string(stream) + " did not drain");
    }

    void WaitFor(const ResidencyToken& token) {
        for (std::size_t stream = 0; stream < FakeStreams; ++stream) {
            auto& state = streams[stream];
            if (token[stream] > state.completed && !state.batches.empty() && state.batches.back().value <= token[stream]) state.batches.back().submitted = true;
            completeFinished(stream);
        }
        while (!Reached(token)) {
            if (!StepAny()) {
                Expect(false, "a wait made no progress");
                return;
            }
        }
    }

    ResidencyToken TitleToken(std::uint64_t page, std::optional<std::size_t> except = std::nullopt) {
        ResidencyToken token{};
        const auto group = imports[page].group;
        for (const auto& op : inflight()) {
            if (!op.title || imports[op.page].group != group) continue;
            if (except && op.stream == *except) continue;
            token[op.stream] = std::max(token[op.stream], op.batch);
        }
        return token;
    }

    std::size_t RecordTitle(Kind kind, std::uint64_t page, ResidencyMask mask, std::uint64_t value, bool granted = false) {
        Op op;
        op.kind = kind;
        op.title = true;
        op.page = page;
        op.mask = mask;
        op.value = value;
        const auto slot = kind == Kind::Write && !granted ? std::nullopt : manager->Slot(page);
        op.mirror = slot.has_value();
        if (slot) op.slot = *slot;
        op.import = imports[page].storage;
        return record(op);
    }

    void CpuWrite(std::uint64_t page, ResidencyMask mask, std::uint64_t value) {
        auto& entry = imports[page];
        Expect(entry.attributes.cpuWritable, "a CPU store into " + Name(page) + ", which the guest cannot write");
        for (std::uint32_t subpage = 0; subpage < ResidencySubpages; ++subpage) {
            if ((mask >> subpage) & 1u) (*entry.storage)[subpage] = value;
        }
        entry.stamp = ++generation;
        for (const auto alias : AliasesOf(page)) imports[alias].stamp = generation;
        auto& guard = armed[entry.group];
        if (!guard) return;
        guard = false;
        ++faults;
        const auto pages = AliasesOf(page);
        reported.insert(pages.begin(), pages.end());
        manager->QueueCpuWritten(pages);
    }

    void Remap(std::uint64_t page, std::uint64_t value) {
        auto& entry = imports[page];
        armed[entry.group] = false;
        entry.storage = std::make_shared<Contents>();
        entry.storage->fill(value);
        entry.stamp = ++generation;
        entry.attributes.serial = generation;
    }

    std::vector<Op> inflight() const {
        std::vector<Op> result;
        for (const auto& [seq, index] : unexecutedOps) result.push_back(ops[index]);
        return result;
    }

private:
    std::uint64_t nextSeq = 1;
    std::set<std::uint64_t> unexecuted;
    std::map<std::uint64_t, std::size_t> unexecutedOps;
    std::set<std::uint64_t> barriers;

    void checkSlot(const ResidencyCopy& copy) {
        Expect(copy.mirror.chunk < live.size() && live[copy.mirror.chunk], "copy into a chunk that is not live");
        Expect(copy.mirror.slot < ChunkSlots, "copy into a slot past the chunk");
        Expect(copy.bytes != 0 && copy.offset + copy.bytes <= ResidencyPageBytes, "copy region is empty or leaves the page");
    }

    std::size_t record(Op op) {
        for (std::size_t other = 0; other < FakeStreams; ++other) {
            if (other == caller) continue;
            auto& stream = streams[other];
            if (stream.batches.empty() || stream.batches.back().submitted || stream.batches.back().ops.empty()) continue;
            const auto& open = stream.batches.back();
            const bool holdsBarrier = std::any_of(open.ops.begin(), open.ops.end(), [&](std::size_t index) { return ops[index].barrier; });
            if (op.barrier || holdsBarrier) stream.batches.back().submitted = true;
        }
        CallerToken(caller);
        auto& batch = streams[caller].batches.back();
        op.stream = caller;
        op.batch = batch.value;
        op.seq = nextSeq++;
        const auto index = ops.size();
        ops.push_back(op);
        batch.ops.push_back(index);
        unexecuted.insert(op.seq);
        unexecutedOps.emplace(op.seq, index);
        if (op.barrier) barriers.insert(op.seq);
        if (op.mirror) {
            auto& pin = modelPins[op.page];
            pin = MergeTokens(pin, On(caller, batch.value));
        }
        return index;
    }

    bool executable(const Op& op) const {
        if (op.barrier && !unexecuted.empty() && *unexecuted.begin() < op.seq) return false;
        if (!barriers.empty() && *barriers.begin() < op.seq) return false;
        return true;
    }

    Contents& mirror(MirrorSlot slot) {
        Expect(slot.chunk < live.size() && live[slot.chunk], "GPU access to a released chunk");
        return mirrors[slot.chunk][slot.slot];
    }

    void execute(Op& op) {
        switch (op.kind) {
            case Kind::CopyIn: {
                auto& target = mirror(op.slot);
                for (std::uint32_t subpage = 0; subpage < ResidencySubpages; ++subpage) {
                    if ((op.mask >> subpage) & 1u) target[subpage] = (*op.import)[subpage];
                }
                break;
            }
            case Kind::WriteBack: {
                const auto& source = mirror(op.slot);
                for (std::uint32_t subpage = 0; subpage < ResidencySubpages; ++subpage) {
                    if ((op.mask >> subpage) & 1u) (*op.import)[subpage] = source[subpage];
                }
                break;
            }
            case Kind::Read: {
                const auto subpage = static_cast<std::uint32_t>(std::countr_zero(static_cast<std::uint32_t>(op.mask)));
                const auto actual = op.mirror ? mirror(op.slot)[subpage] : (*op.import)[subpage];
                Expect(actual == op.value, "GPU read of " + Name(op.page) + " subpage " + std::to_string(subpage) + (op.mirror ? " (mirror)" : " (import)") + " got " + std::to_string(actual) + ", expected " + std::to_string(op.value));
                ++(op.mirror ? mirrorReads : importReads);
                break;
            }
            case Kind::Write: {
                auto& target = op.mirror ? mirror(op.slot) : *op.import;
                for (std::uint32_t subpage = 0; subpage < ResidencySubpages; ++subpage) {
                    if ((op.mask >> subpage) & 1u) target[subpage] = op.value;
                }
                break;
            }
        }
        op.executed = true;
        unexecuted.erase(op.seq);
        unexecutedOps.erase(op.seq);
        barriers.erase(op.seq);
    }

    void completeFinished(std::size_t stream) {
        auto& state = streams[stream];
        while (!state.batches.empty() && state.batches.front().submitted && state.batches.front().done == state.batches.front().ops.size()) {
            const auto batch = state.batches.front();
            for (const auto index : batch.ops) {
                if (ops[index].kind == Kind::Write && !ops[index].mirror) imports[ops[index].page].stamp = ++generation;
            }
            state.completed = batch.value;
            state.batches.pop_front();
        }
    }
};

class NullDevice final : public IResidencyDevice {
public:
    bool Reached(const ResidencyToken&) override { return true; }
    void Wait(const ResidencyToken&) override {}
    ResidencyToken Newest() override { return {}; }
    ResidencyToken Current() override { return {}; }
    ResidencyImport Import(std::uint64_t) override { return {true, true, true}; }
    std::vector<ResidencyRange> Aliases(std::uint64_t, std::uint64_t) override { return {}; }
    std::uint64_t Generation() override { return 1; }
    void ChangedPages(std::uint64_t, std::span<const std::uint64_t>, std::span<std::uint8_t> changed) override { std::fill(changed.begin(), changed.end(), std::uint8_t{0}); }
    std::uint64_t AllocateChunk(std::uint32_t chunk, std::uint32_t) override { return std::uint64_t{chunk + 1} << 36; }
    void ReleaseChunk(std::uint32_t) override {}
    ResidencyToken CopyIn(std::span<const ResidencyCopy>) override { return {}; }
    ResidencyToken WriteBack(std::span<const ResidencyCopy>) override { return {}; }
    ResidencyRefusal Guard(std::uint64_t) override { return ResidencyRefusal::None; }
    void Unguard(std::span<const std::uint64_t>) override {}
    ResidencyRefusal GuardAccess(std::uint64_t, ResidencyMask) override { return ResidencyRefusal::None; }
    void UnguardAccess(std::uint64_t) override {}
};

struct Fixture {
    explicit Fixture(std::uint32_t pageCount, std::uint32_t budgetChunks, std::uint32_t hotFaults = 2, std::uint32_t hotFrames = 2, bool deferCopyIn = false) {
        for (std::uint32_t page = 0; page < pageCount; ++page) {
            auto& entry = gpu.imports[BasePage + page];
            entry.group = page;
            for (std::uint32_t subpage = 0; subpage < ResidencySubpages; ++subpage) (*entry.storage)[subpage] = (std::uint64_t{page} << 8) | subpage;
        }
        gpu.budgetChunks = budgetChunks;
        manager = std::make_unique<ResidencyManager>(gpu, ResidencyPolicy{ChunkSlots, std::uint64_t{budgetChunks} * ChunkSlots * ResidencyPageBytes, hotFaults, hotFrames, ResidencyHotQuietFrames, deferCopyIn});
        gpu.manager = manager.get();
    }

    ~Fixture() { gpu.Drain(); }

    void Use(std::size_t stream, std::uint64_t page, std::uint32_t pages = 1, ResidencyAccess access = ResidencyAccess::Read) {
        gpu.caller = stream;
        manager->Use(Address(page), std::uint64_t{pages} * ResidencyPageBytes, access);
    }

    std::uint64_t Stored(std::uint64_t page, std::uint32_t subpage) {
        return (*gpu.imports[page].storage)[subpage];
    }

    bool Armed(std::uint64_t page) {
        return gpu.armed[gpu.imports[page].group];
    }

    void Share(std::uint64_t page) {
        auto& attributes = gpu.imports[page].attributes;
        attributes.section = gpu.imports[page].group + 1;
        attributes.offset = 0;
    }

    void Grant(std::size_t stream, std::uint64_t page, ResidencyMask mask) {
        gpu.caller = stream;
        manager->Apply();
        std::uint32_t bits = mask;
        while (bits != 0) {
            const auto first = static_cast<std::uint32_t>(std::countr_zero(bits));
            const auto run = static_cast<std::uint32_t>(std::countr_one(bits >> first));
            const auto address = Address(page, first);
            const auto bytes = std::uint64_t{run} * ResidencySubpageBytes;
            manager->NoteGpuWritten(address, bytes);
            manager->Pin(address, bytes, ResidencyAccess::Write, gpu.CallerToken(stream));
            bits &= run >= 32 ? 0u : ~(((1u << run) - 1u) << first);
        }
    }

    FakeGpu gpu;
    std::unique_ptr<ResidencyManager> manager;
};

void BudgetAndLeastRecentOrder() {
    scope = "budget and least-recent order";
    Fixture fixture(16, 2);
    for (std::uint32_t page = 0; page < 8; ++page) {
        fixture.Use(0, BasePage + page);
        fixture.gpu.Drain();
    }
    auto counts = fixture.manager->Counts();
    Expect(counts.chunks == 2 && counts.resident == 8 && counts.free == 0, "8 pages fill exactly two chunks of 4");
    const auto slotOf2 = fixture.manager->Slot(BasePage + 2);
    fixture.Use(0, BasePage + 0);
    fixture.Use(0, BasePage + 1);
    fixture.gpu.Drain();
    fixture.Use(0, BasePage + 8);
    fixture.gpu.Drain();
    Expect(!fixture.manager->Slot(BasePage + 2), "the least recently used page (2) is evicted");
    Expect(fixture.manager->Slot(BasePage + 0) && fixture.manager->Slot(BasePage + 1), "recently used pages 0 and 1 stay");
    Expect(fixture.manager->Slot(BasePage + 8) == slotOf2, "page 8 takes the evicted slot once it retired");
    Expect(!fixture.Armed(BasePage + 2), "the evicted page's guard is disarmed");
    counts = fixture.manager->Counts();
    Expect(counts.chunks == 2 && counts.chunksAllocated == 2, "no chunk past the budget");
    Expect(counts.evicted == 1 && counts.admitted == 9, "one eviction, nine admissions");
    Expect(fixture.manager->Translate(BasePage + 8) == fixture.manager->ChunkAddress(slotOf2->chunk) + (std::uint64_t{slotOf2->slot} << ResidencyPageShift), "Translate gives the slot's device address");
}

void PinnedPagesNeverEvicted() {
    scope = "pinned pages never evicted";
    Fixture fixture(8, 1);
    fixture.Use(1, BasePage + 0, 4);
    for (std::uint32_t page = 0; page < 4; ++page) fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + page, 1, (std::uint64_t{page} << 8));
    fixture.Use(2, BasePage + 4);
    auto counts = fixture.manager->Counts();
    Expect(counts.evicted == 0 && counts.deferred == 1 && counts.resident == 4, "every slot pinned by stream 1's batch: page 4 deferred, nothing evicted");
    Expect(!fixture.manager->Slot(BasePage + 4), "page 4 not resident");
    fixture.gpu.Drain();
    fixture.Use(2, BasePage + 4);
    counts = fixture.manager->Counts();
    Expect(counts.evicted == 1, "after the batch completed, one page is evicted");
    Expect(!fixture.manager->Slot(BasePage + 0), "the least recent page 0 left");
}

void TokenCompletionReleasesPins() {
    scope = "token completion releases pins";
    Fixture fixture(4, 1);
    fixture.Use(0, BasePage + 0);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 1, 0);
    fixture.Use(1, BasePage + 0);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 1, 0);
    fixture.manager->SetBudget(0);
    Expect(fixture.manager->Slot(BasePage + 0).has_value(), "pinned by streams 0 and 1: not evicted when the budget drops");
    Expect(fixture.manager->Counts().chunks == 1, "the chunk stays while its page is pinned");
    fixture.gpu.Submit(0);
    while (fixture.gpu.Step(0)) {}
    fixture.manager->Retire();
    Expect(fixture.manager->Slot(BasePage + 0).has_value(), "stream 1's batch still pins the page");
    fixture.gpu.Drain();
    fixture.manager->Retire();
    Expect(!fixture.manager->Slot(BasePage + 0), "both tokens reached: the page leaves");
    Expect(fixture.manager->Counts().chunks == 0, "the drained chunk is released");
}

void FlushWaitsForWriters() {
    scope = "flush waits for writers";
    Fixture fixture(4, 1);
    fixture.Use(1, BasePage + 0, 1, ResidencyAccess::Write);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 0, 0x8001, 4242);
    const auto waits = fixture.gpu.waits;
    const auto copies = fixture.gpu.copiedIn.size();
    fixture.manager->FlushForCpuRead(Address(BasePage + 0), 16);
    Expect(fixture.gpu.waits == waits + 1, "one wait, for the writer");
    Expect(fixture.gpu.copiedIn.size() == copies, "a flush records no copy");
    Expect(fixture.Stored(BasePage + 0, 0) == 4242 && fixture.Stored(BasePage + 0, 15) == 4242, "the CPU read sees the GPU's write, which went to the import");
    Expect(fixture.manager->Slot(BasePage + 0).has_value(), "a flush keeps the page resident");
    fixture.manager->FlushForCpuRead(Address(BasePage + 0), ResidencyPageBytes);
    Expect(fixture.gpu.waits == waits + 1, "a settled writer needs no wait");
    fixture.Use(0, BasePage + 0);
    Expect(fixture.gpu.copiedIn.size() == copies + 1 && fixture.gpu.copiedIn.back().bytes == ResidencyPageBytes, "the next use refreshes the mirror the write left behind");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x8000, 4242);
    fixture.gpu.Drain();
    Expect(fixture.gpu.mirrorReads == 1, "the read after the refresh went through the mirror");
}

void FaultRefreshOrderedAfterOwnWriters() {
    scope = "a fault's refresh is ordered after the caller's own writers";
    Fixture fixture(4, 1);
    fixture.Use(0, BasePage + 0);
    fixture.gpu.Drain();
    fixture.manager->Prepare();
    fixture.Use(0, BasePage + 0, 1, ResidencyAccess::Write);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 0, 0x0001, 55);
    fixture.gpu.CpuWrite(BasePage + 0, 0x0002, 77);
    const auto waits = fixture.gpu.waits;
    fixture.gpu.copiedIn.clear();
    fixture.manager->Apply();
    Expect(fixture.gpu.waits == waits, "no wait for a writer recorded earlier in the caller's stream");
    Expect(fixture.gpu.copiedIn.size() == 1 && fixture.gpu.copiedIn[0].page == BasePage + 0, "the refresh is recorded behind the writer");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0001, 55);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0002, 77);
    fixture.gpu.Drain();
    fixture.manager->Prepare();
    fixture.Use(0, BasePage + 0, 1, ResidencyAccess::Write);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 0, 0x0004, 56);
    fixture.gpu.CpuWrite(BasePage + 0, 0x0008, 78);
    fixture.gpu.caller = 1;
    const auto crossWaits = fixture.gpu.waits;
    fixture.manager->Apply();
    Expect(fixture.gpu.waits == crossWaits + 1, "a writer in another stream is waited for");
    fixture.gpu.Drain();
}

void ImportWritesRefreshMirrors() {
    scope = "import writes refresh mirrors";
    Fixture fixture(4, 1);
    fixture.Use(0, BasePage + 0);
    fixture.gpu.Drain();
    fixture.gpu.copiedIn.clear();
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 0, 0x0010, 99);
    fixture.manager->NoteImportWritten(Address(BasePage + 0, 4), 100);
    Expect(fixture.gpu.copiedIn.size() == 1 && fixture.gpu.copiedIn[0].offset == 4 * ResidencySubpageBytes && fixture.gpu.copiedIn[0].bytes == 100, "only the written bytes are refreshed");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0010, 99);
    fixture.gpu.Drain();
    fixture.gpu.copiedIn.clear();
    fixture.Use(0, BasePage + 0);
    Expect(fixture.gpu.copiedIn.size() == 1 && fixture.gpu.copiedIn[0].bytes == ResidencyPageBytes, "a partial refresh keeps the copy-in generation, so the stamped page is refreshed whole at its next use");
    fixture.gpu.Drain();
    fixture.gpu.copiedIn.clear();
    fixture.Use(0, BasePage + 0);
    Expect(fixture.gpu.copiedIn.empty(), "an unchanged page is not copied again");
}

std::uint64_t CopiedInBytes(const FakeGpu& gpu) {
    std::uint64_t bytes = 0;
    for (const auto& copy : gpu.copiedIn) bytes += copy.bytes;
    return bytes;
}

void DeferredCopyInUntilAReader() {
    scope = "copy-in deferred until a reader takes the mirror";
    Fixture fixture(4, 1, 2, 2, true);
    const auto page = BasePage + 0;
    fixture.Use(0, page);
    fixture.gpu.Drain();
    Expect(fixture.gpu.copiedIn.empty() && fixture.manager->Behind(page) == ResidencyWholePage, "admission records no copy: the whole mirror is behind its import");
    Expect(fixture.manager->MirroredIn(Address(page), ResidencyPageBytes).empty() && fixture.manager->CurrentPages().empty(), "a mirror behind its import is left out of the device page table");
    for (std::uint64_t round = 0; round < 64; ++round) {
        fixture.gpu.RecordTitle(FakeGpu::Kind::Write, page, 0x0010, 100 + round);
        fixture.manager->RefreshImportWritten(Address(page, 4), 100);
        fixture.gpu.CpuWrite(page, 0x0100, 200 + round);
        fixture.Use(0, page);
        fixture.gpu.Drain();
    }
    auto counts = fixture.manager->Counts();
    Expect(fixture.gpu.copiedIn.empty() && counts.copiedInBytes == 0, "64 rounds of import writes and CPU stores with no reader copy nothing, not " + std::to_string(CopiedInBytes(fixture.gpu)) + " bytes");
    Expect(counts.behindFlips == 1 && counts.behind == 1 && counts.copyInDeferredBytes >= 64 * 100, "the page went behind once and stayed behind");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, page, 0x0010, 163);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, page, 0x0100, 263);
    fixture.gpu.Drain();
    Expect(fixture.gpu.mirrorReads == 2, "both reads went through the mirror");
    Expect(CopiedInBytes(fixture.gpu) == ResidencyPageBytes, "the first reader catches the page up once, whole: " + std::to_string(CopiedInBytes(fixture.gpu)) + " bytes");
    Expect(fixture.manager->Behind(page) == 0 && fixture.manager->CurrentPages() == std::vector<std::uint64_t>{page} && fixture.manager->MirroredIn(Address(page), ResidencyPageBytes).size() == 1, "a caught-up mirror is exposed again");
    fixture.gpu.copiedIn.clear();
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, page, 0x0010, 300);
    fixture.manager->RefreshImportWritten(Address(page, 4), 100);
    Expect(fixture.manager->Behind(page) == 0x0010 && fixture.manager->CurrentPages().empty() && fixture.gpu.copiedIn.empty(), "a 100-byte import write puts its subpage behind and copies nothing");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, page, 0x0010, 300);
    fixture.gpu.Drain();
    Expect(CopiedInBytes(fixture.gpu) == ResidencySubpageBytes, "the reader copies the written subpage only: " + std::to_string(CopiedInBytes(fixture.gpu)) + " bytes");

    const auto shared = BasePage + 1;
    fixture.gpu.imports[shared].attributes.cpuWritable = false;
    fixture.Share(shared);
    fixture.Use(0, shared);
    fixture.gpu.Drain();
    fixture.gpu.copiedIn.clear();
    Expect(fixture.manager->Behind(shared) == ResidencyWholePage, "the shared page is admitted behind");
    fixture.Grant(0, shared, 0x0001);
    Expect(CopiedInBytes(fixture.gpu) == ResidencyPageBytes && fixture.manager->Behind(shared) == 0 && fixture.manager->Dirty(shared) == 0x0001, "a GPU write into a mirror behind its import catches the page up first");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, shared, 0x0001, 900, true);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, shared, 0x0040, 901);
    fixture.gpu.copiedIn.clear();
    fixture.manager->RefreshImportWritten(Address(shared, 6), 100);
    Expect(fixture.gpu.copiedIn.size() == 1 && fixture.gpu.copiedIn[0].bytes == 100 && fixture.manager->Behind(shared) == 0, "a page with GPU-dirty bytes is refreshed at once, the written bytes only");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, shared, 0x0001, 900);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, shared, 0x0040, 901);
    fixture.gpu.Drain();
    counts = fixture.manager->Counts();
    Expect(counts.copiedInBytes <= counts.admitted * ResidencyPageBytes + 2 * ResidencySubpageBytes, "copy-in is bounded by the admitted pages plus the written subpages read since: " + std::to_string(counts.copiedInBytes) + " bytes");
}

void GuardedPagesRefreshAfterFaults() {
    scope = "guarded pages refresh after faults";
    Fixture fixture(4, 1);
    fixture.gpu.imports[BasePage + 1].attributes.cpuWritable = false;
    fixture.Share(BasePage + 0);
    fixture.Share(BasePage + 1);
    fixture.Use(0, BasePage + 0, 2);
    fixture.gpu.Drain();
    Expect(fixture.Armed(BasePage + 0), "a CPU-writable page is guarded before its copy-in");
    Expect(!fixture.Armed(BasePage + 1), "a page the CPU cannot write is not guarded");
    fixture.manager->Prepare();
    fixture.gpu.CpuWrite(BasePage + 0, 0x0002, 7);
    fixture.gpu.CpuWrite(BasePage + 0, 0x0004, 8);
    Expect(fixture.gpu.faults == 1, "only the first store of the epoch faults");
    Expect(!fixture.Armed(BasePage + 0), "the fault disarmed the guard");
    fixture.gpu.copiedIn.clear();
    fixture.manager->Prepare();
    Expect(fixture.Armed(BasePage + 0), "the ordering point re-arms the guard");
    Expect(fixture.gpu.copiedIn.size() == 1 && fixture.gpu.copiedIn[0].bytes == ResidencyPageBytes, "and refreshes the page before the next batch");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0002, 7);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0004, 8);
    fixture.gpu.Drain();
    const auto counts = fixture.manager->Counts();
    Expect(counts.faults == 1 && counts.hot == 0 && counts.refreshed == 1, "one fault, one refresh, no hot page");
}

void OrderingPointsReportQueuedWork() {
    scope = "ordering points report queued work";
    Fixture fixture(4, 1);
    fixture.Share(BasePage + 0);
    fixture.Use(0, BasePage + 0);
    fixture.gpu.Drain();
    const auto epoch = fixture.manager->Epoch();
    Expect(!fixture.manager->Advance() && fixture.manager->Epoch() == epoch + 1, "an ordering point with nothing queued only advances the epoch");
    fixture.gpu.CpuWrite(BasePage + 0, 0x0001, 3);
    Expect(fixture.manager->Advance(), "a queued fault is reported");
    fixture.manager->Apply();
    Expect(!fixture.manager->Advance(), "applying clears the report");
    fixture.gpu.imports[BasePage + 0].attributes.serial = 1;
    fixture.manager->DropRange(Address(BasePage + 0), ResidencyPageBytes);
    Expect(fixture.manager->Advance(), "a queued drop is reported");
    fixture.manager->Apply();
    Expect(!fixture.manager->Advance() && !fixture.manager->Slot(BasePage + 0), "the drop was applied");
}

void HotPagesStayAndCool() {
    scope = "hot pages leave for their imports and cool";
    Fixture fixture(4, 1, 2, 2);
    fixture.Share(BasePage + 0);
    fixture.Share(BasePage + 2);
    fixture.Use(0, BasePage + 0);
    fixture.Use(0, BasePage + 2);
    fixture.gpu.Drain();
    fixture.manager->Prepare();
    for (std::uint32_t round = 0; round < 4; ++round) {
        fixture.gpu.CpuWrite(BasePage + 2, 0x0001, 500 + round);
        fixture.manager->Prepare();
        fixture.manager->NextFrame();
        fixture.manager->NextFrame();
    }
    Expect(fixture.manager->Slot(BasePage + 2).has_value() && fixture.manager->Counts().faults == 4 && !fixture.manager->Hot(BasePage + 2), "a page written every other frame faults every time and stays guarded in its mirror");
    for (std::uint32_t round = 0; round < 3; ++round) {
        fixture.gpu.CpuWrite(BasePage + 0, 0x0001, 1 + round);
        fixture.manager->Prepare();
    }
    Expect(fixture.manager->Slot(BasePage + 0).has_value() && !fixture.manager->Hot(BasePage + 0) && fixture.Armed(BasePage + 0), "faults inside one frame: the page stays in its mirror and guarded");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0001, 3);
    fixture.gpu.Drain();
    fixture.manager->NextFrame();
    fixture.gpu.CpuWrite(BasePage + 0, 0x0001, 4);
    fixture.manager->Prepare();
    fixture.gpu.Drain();
    fixture.manager->Prepare();
    Expect(!fixture.manager->Slot(BasePage + 0).has_value() && fixture.manager->Hot(BasePage + 0) && !fixture.manager->HotResident(BasePage + 0), "faults in consecutive frames: the page turns hot and leaves its mirror for its import");
    Expect(!fixture.Armed(BasePage + 0), "a hot page is unguarded");
    const auto faults = fixture.gpu.faults;
    fixture.gpu.CpuWrite(BasePage + 0, 0x0001, 5);
    Expect(fixture.gpu.faults == faults, "stores to a hot page do not fault");
    fixture.gpu.copiedIn.clear();
    const auto collected = fixture.gpu.collectedPages;
    for (std::uint32_t point = 0; point < 4; ++point) {
        fixture.manager->Prepare();
        fixture.manager->RefreshHot();
    }
    Expect(fixture.gpu.copiedIn.empty() && fixture.gpu.collectedPages == collected, "ordering points copy and collect nothing for a hot page");
    fixture.Use(0, BasePage + 0);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0001, 5);
    fixture.gpu.Drain();
    Expect(!fixture.manager->Slot(BasePage + 0).has_value(), "a dispatch reads the hot page from its import");
    for (std::uint32_t quiet = 1; quiet < ResidencyHotQuietFrames; ++quiet) {
        fixture.manager->NextFrame();
        fixture.Use(0, BasePage + 0);
        fixture.gpu.Drain();
        Expect(!fixture.manager->Slot(BasePage + 0).has_value() && fixture.manager->Hot(BasePage + 0), "quiet for " + std::to_string(quiet) + " frames: still served from its import");
    }
    fixture.manager->NextFrame();
    fixture.Use(0, BasePage + 0);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0001, 5);
    fixture.gpu.Drain();
    Expect(fixture.manager->Slot(BasePage + 0).has_value() && !fixture.manager->Hot(BasePage + 0) && fixture.Armed(BasePage + 0), "quiet for the quiet frames: back in a guarded mirror, read correctly");
    const auto counts = fixture.manager->Counts();
    Expect(counts.hot == 1 && counts.cooled == 1 && counts.hotServed == 1, "one page turned hot, was served from its import and cooled");
}

void UnguardablePages() {
    scope = "unguardable pages";
    Fixture fixture(8, 1);
    fixture.gpu.guardRefused.insert_or_assign(BasePage + 3, ResidencyRefusal::SharedPinned);
    fixture.Share(BasePage + 3);
    fixture.Share(BasePage + 4);
    fixture.Use(0, BasePage + 3, 2);
    fixture.gpu.Drain();
    fixture.manager->Prepare();
    Expect(!fixture.manager->Slot(BasePage + 3) && fixture.manager->Slot(BasePage + 4), "a page whose guard cannot be armed is served from its import");
    Expect(fixture.manager->Refused(BasePage + 3) == ResidencyRefusal::SharedPinned && !fixture.manager->Hot(BasePage + 3) && !fixture.Armed(BasePage + 3), "it is refused for its reason, not hot, and not polled from a mirror");
    fixture.gpu.Drain();
    fixture.gpu.CpuWrite(BasePage + 3, 0x0001, 9);
    Expect(fixture.gpu.faults == 0, "a store into the unguarded page does not fault");
    fixture.Use(0, BasePage + 3);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 3, 0x0001, 9);
    fixture.gpu.Drain();
    fixture.gpu.guardRefused.insert_or_assign(BasePage + 4, ResidencyRefusal::AliasHostWrite);
    fixture.gpu.CpuWrite(BasePage + 4, 0x0001, 10);
    fixture.manager->Prepare();
    fixture.gpu.Drain();
    fixture.manager->Prepare();
    Expect(!fixture.manager->Slot(BasePage + 4) && fixture.manager->Refused(BasePage + 4) == ResidencyRefusal::AliasHostWrite && !fixture.manager->Hot(BasePage + 4), "a page whose re-arm fails leaves its mirror for its import, refused, not hot");
    fixture.Use(0, BasePage + 4);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 4, 0x0001, 10);
    fixture.gpu.Drain();
    const auto counts = fixture.manager->Counts();
    Expect(counts.unguardable == 2 && counts.hot == 0 && counts.refusedPages == 2, "two unguardable events counted, no hot page, two refused pages");
    Expect(counts.refusals[static_cast<std::size_t>(ResidencyRefusal::SharedPinned)] == 1 && counts.refusals[static_cast<std::size_t>(ResidencyRefusal::AliasHostWrite)] == 1, "each refusal is counted under its reason");
}

void RefusedUntilTheirEvent() {
    scope = "refused pages wait for their reason's event";
    Fixture fixture(8, 1);
    fixture.Share(BasePage + 2);
    fixture.gpu.guardRefused.insert_or_assign(BasePage + 2, ResidencyRefusal::SharedPinned);
    fixture.Use(0, BasePage + 2);
    fixture.gpu.Drain();
    Expect(!fixture.manager->Slot(BasePage + 2) && fixture.manager->Refused(BasePage + 2) == ResidencyRefusal::SharedPinned && fixture.gpu.guardCalls[BasePage + 2] == 1, "a page libc refuses to guard is not admitted");
    for (std::uint32_t frame = 0; frame < 2 * ResidencyHotQuietFrames; ++frame) {
        fixture.manager->NextFrame();
        for (std::uint32_t point = 0; point < 4; ++point) {
            fixture.manager->Prepare();
            fixture.manager->RefreshHot();
            fixture.Use(0, BasePage + 2);
            const std::array<std::uint64_t, 1> touched{BasePage + 2};
            fixture.manager->AdmitTouched(touched);
            fixture.gpu.Drain();
        }
    }
    Expect(fixture.gpu.guardCalls[BasePage + 2] == 1 && !fixture.manager->Slot(BasePage + 2), "frames, ordering points, uses and touches retried the guard " + std::to_string(fixture.gpu.guardCalls[BasePage + 2] - 1) + " times");
    fixture.manager->QueueRefusalCleared(Address(BasePage + 2), ResidencyPageBytes, ResidencyRefusalBit(ResidencyRefusal::AliasHostWrite) | ResidencyRefusalBit(ResidencyRefusal::UnsettledImport));
    fixture.manager->Prepare();
    fixture.Use(0, BasePage + 2);
    fixture.gpu.Drain();
    Expect(fixture.gpu.guardCalls[BasePage + 2] == 1 && fixture.manager->Refused(BasePage + 2) == ResidencyRefusal::SharedPinned, "an event that cannot clear the reason retried the guard");
    fixture.manager->QueueRefusalCleared(Address(BasePage + 3), ResidencyPageBytes, ResidencyRefusalBit(ResidencyRefusal::SharedPinned));
    fixture.manager->Prepare();
    fixture.Use(0, BasePage + 2);
    fixture.gpu.Drain();
    Expect(fixture.gpu.guardCalls[BasePage + 2] == 1, "the event of another page retried the guard");
    fixture.gpu.guardRefused.erase(BasePage + 2);
    fixture.manager->QueueRefusalCleared(Address(BasePage + 2), ResidencyPageBytes, ResidencyRefusalBit(ResidencyRefusal::SharedPinned) | ResidencyRefusalBit(ResidencyRefusal::PrivatePinned));
    fixture.manager->Prepare();
    Expect(!fixture.manager->Refused(BasePage + 2).has_value() && fixture.gpu.guardCalls[BasePage + 2] == 1, "the pin release lifts the refusal and costs nothing until the page is used");
    fixture.Use(0, BasePage + 2);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 2, 0x0001, fixture.Stored(BasePage + 2, 0));
    fixture.gpu.Drain();
    Expect(fixture.gpu.guardCalls[BasePage + 2] == 2 && fixture.manager->Slot(BasePage + 2) && fixture.Armed(BasePage + 2), "the next use admits the page into a guarded mirror");
    Expect(fixture.manager->Counts().refusalRetries == 1, "one retry counted");
}

struct WriterCost {
    std::uint64_t protections = 0;
    double microseconds = 0;
};

WriterCost UnboundedWriterCost(std::uint32_t mirrors) {
    constexpr std::uint32_t Writers = 6;
    Fixture fixture(mirrors, (mirrors + ChunkSlots - 1) / ChunkSlots);
    for (std::uint32_t page = 0; page < mirrors; ++page) fixture.Share(BasePage + page);
    fixture.Use(0, BasePage, mirrors, ResidencyAccess::Write);
    fixture.gpu.Drain();
    fixture.manager->Apply();
    Expect(fixture.manager->Counts().resident == mirrors, std::to_string(mirrors) + " writable mirrors resident");
    const auto stored = BasePage + 1;
    WriterCost cost;
    std::uint64_t first = 0;
    for (std::uint32_t writer = 0; writer < Writers; ++writer) {
        if (writer == 1) first = fixture.gpu.protections;
        fixture.gpu.caller = 1;
        const auto start = std::chrono::steady_clock::now();
        fixture.manager->NoteUnboundedWriter(fixture.gpu.CallerToken(1));
        auto spent = std::chrono::steady_clock::now() - start;
        fixture.gpu.RecordTitle(FakeGpu::Kind::Write, stored, 0x0001, 900 + writer, true);
        fixture.gpu.Drain();
        fixture.manager->QueueGpuWritten(Address(stored), ResidencySubpageBytes);
        const auto apply = std::chrono::steady_clock::now();
        fixture.manager->Apply();
        spent += std::chrono::steady_clock::now() - apply;
        if (writer >= 1) cost.microseconds += std::chrono::duration<double, std::micro>(spent).count();
    }
    cost.protections = (fixture.gpu.protections - first) / (Writers - 1);
    cost.microseconds /= Writers - 1;
    const auto counts = fixture.manager->Counts();
    Expect(counts.demoted == 0 && counts.resident == mirrors, "no writer demoted a mirror");
    Expect(fixture.manager->Dirty(stored) == 0x0001, "the stored subpage is dirty in its mirror");
    Expect(fixture.Stored(stored, 0) != 900 + Writers - 1, "the store has not reached guest memory yet");
    fixture.manager->FlushForCpuRead(Address(stored), 8);
    Expect(fixture.Stored(stored, 0) == 900 + Writers - 1, "a CPU read after the writers sees the last store");
    return cost;
}

void UnboundedWriterCostIndependentOfMirrors() {
    scope = "an unbounded writer's guard cost does not grow with the resident mirrors";
    const auto few = UnboundedWriterCost(16);
    const auto many = UnboundedWriterCost(4096);
    std::printf("unbounded writer: %llu protection calls and %.1f us beside 16 mirrors, %llu and %.1f us beside 4096\n", static_cast<unsigned long long>(few.protections), few.microseconds, static_cast<unsigned long long>(many.protections), many.microseconds);
    Expect(few.protections == many.protections, "the same protection calls per writer beside 16 and 4096 writable mirrors");
    Expect(many.protections <= 4, "a writer that stores to one page makes a few protection calls");
}

void DemotionAndDrops() {
    scope = "demotion and drops";
    Fixture fixture(8, 2);
    fixture.gpu.imports[BasePage + 3].attributes.gpuWritable = false;
    fixture.Use(0, BasePage + 0, 4, ResidencyAccess::Write);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 1, 0x0001, 31);
    const auto copies = fixture.gpu.copiedIn.size();
    fixture.manager->Demote(Address(BasePage + 1), 1);
    Expect(!fixture.manager->Slot(BasePage + 1), "a demoted page leaves residency");
    Expect(fixture.Stored(BasePage + 1, 0) == 31, "the demotion waited for the writer, whose store went to the import");
    Expect(fixture.gpu.copiedIn.size() == copies, "a demotion records no copy");
    fixture.gpu.Drain();
    fixture.manager->DemoteWritable();
    Expect(!fixture.manager->Slot(BasePage + 0) && !fixture.manager->Slot(BasePage + 2), "GPU-writable pages demoted");
    Expect(fixture.manager->Slot(BasePage + 3).has_value(), "a page on a read-only import stays");
    fixture.manager->DropRange(Address(BasePage + 3), ResidencyPageBytes);
    Expect(fixture.manager->Slot(BasePage + 3).has_value(), "a drop over a page whose import is unchanged keeps it");
    fixture.gpu.imports[BasePage + 3].attributes.serial = 1;
    fixture.manager->DropRange(Address(BasePage + 3), ResidencyPageBytes);
    Expect(!fixture.manager->Slot(BasePage + 3), "a dropped page is never translated again");
    Expect(!fixture.manager->AnyResident(Address(BasePage + 0), 8 * ResidencyPageBytes), "nothing resident");
    const auto counts = fixture.manager->Counts();
    Expect(counts.dropped == 1 && counts.demoted == 3, "one drop, three demotions");
    fixture.gpu.Drain();
    fixture.manager->Retire();
    Expect(fixture.manager->Counts().chunks == 1 && fixture.manager->Counts().free == ChunkSlots, "every slot retired");
}

void DirtyMirrorsOnSharedPages() {
    scope = "dirty mirrors on shared pages";
    Fixture fixture(8, 1);
    fixture.Share(BasePage + 0);
    fixture.Use(0, BasePage + 0);
    fixture.gpu.Drain();
    fixture.Grant(0, BasePage + 0, 0x000c);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 0, 0x000c, 777, true);
    Expect(fixture.manager->Dirty(BasePage + 0) == 0x000c && fixture.manager->Counts().demoted == 0, "a granted store into a shared page keeps subpages 2 and 3 dirty in its mirror");
    fixture.gpu.Drain();
    Expect(fixture.Stored(BasePage + 0, 2) == 2 && fixture.gpu.writtenBack.empty(), "the store stays in the mirror until a CPU read or an eviction");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0004, 777);
    fixture.gpu.Drain();
    Expect(fixture.manager->Dirty(BasePage + 0) == 0x000c && fixture.gpu.writtenBack.empty(), "a later GPU read reads the mirror and writes nothing back");
    for (std::uint32_t page = 1; page < 4; ++page) fixture.Use(0, BasePage + page);
    fixture.gpu.Drain();
    const auto slot = fixture.manager->Slot(BasePage + 0);
    fixture.Use(0, BasePage + 4);
    Expect(!fixture.manager->Slot(BasePage + 0), "page 0 evicted");
    Expect(fixture.gpu.writtenBack.size() == 1 && fixture.gpu.writtenBack[0].offset == 2 * ResidencySubpageBytes && fixture.gpu.writtenBack[0].bytes == 2 * ResidencySubpageBytes, "exactly the dirty bytes written back");
    Expect(!fixture.manager->Slot(BasePage + 4), "page 4 deferred: the evicted slot is reused only after its write-back completed");
    Expect(fixture.Stored(BasePage + 0, 2) != 777, "the write-back has not run yet");
    fixture.gpu.Drain();
    Expect(fixture.Stored(BasePage + 0, 2) == 777 && fixture.Stored(BasePage + 0, 3) == 777, "the GPU's bytes reached guest memory");
    Expect(fixture.Stored(BasePage + 0, 1) == 1, "subpages the GPU did not write are untouched");
    fixture.Use(0, BasePage + 4);
    Expect(fixture.manager->Slot(BasePage + 4) == slot, "page 4 reuses the slot after the write-back");
}

void CpuReadWritesDirtyBytesBack() {
    scope = "a CPU read writes dirty bytes back";
    Fixture fixture(4, 1);
    fixture.Share(BasePage + 0);
    fixture.Use(1, BasePage + 0, 1, ResidencyAccess::Write);
    fixture.Grant(1, BasePage + 0, 0x8001);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 0, 0x8001, 4242, true);
    const auto waits = fixture.gpu.waits;
    fixture.manager->FlushForCpuRead(Address(BasePage + 0), 16);
    Expect(fixture.gpu.waits == waits + 2, "one wait for the writer, one for the write-back");
    Expect(fixture.gpu.writtenBack.size() == 2, "the two dirty runs written back");
    Expect(fixture.Stored(BasePage + 0, 0) == 4242 && fixture.Stored(BasePage + 0, 15) == 4242, "the CPU read sees the GPU's write");
    Expect(fixture.manager->Dirty(BasePage + 0) == 0 && fixture.manager->Counts().dirty == 0, "the page is clean after the flush");
    Expect(fixture.manager->Slot(BasePage + 0).has_value(), "a flush keeps the page resident");
    fixture.manager->FlushForCpuRead(Address(BasePage + 0), ResidencyPageBytes);
    Expect(fixture.gpu.waits == waits + 2 && fixture.gpu.writtenBack.size() == 2, "a clean page needs no wait and no copy");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x8000, 4242);
    fixture.gpu.Drain();
    Expect(fixture.gpu.mirrorReads == 1, "the read after the flush went through the mirror");
}

void ExactDirtyRanges() {
    scope = "exact dirty ranges";
    Fixture fixture(4, 1);
    fixture.Share(BasePage + 0);
    fixture.Use(0, BasePage + 0);
    fixture.gpu.Drain();
    fixture.manager->NoteGpuWritten(Address(BasePage + 0) + 100, 200);
    Expect(fixture.manager->Dirty(BasePage + 0) == 0x0001, "bytes 100 to 300 dirty subpage 0");
    fixture.gpu.copiedIn.clear();
    fixture.manager->NoteImportWritten(Address(BasePage + 0) + 1000, 50);
    Expect(fixture.gpu.copiedIn.size() == 1 && fixture.gpu.copiedIn[0].offset == 1000 && fixture.gpu.copiedIn[0].bytes == 50, "an import write beside the dirty bytes refreshes only its own bytes");
    fixture.gpu.copiedIn.clear();
    fixture.manager->NoteImportWritten(Address(BasePage + 0) + 150, 100);
    Expect(fixture.gpu.copiedIn.size() == 1 && fixture.gpu.copiedIn[0].offset == 150 && fixture.gpu.copiedIn[0].bytes == 100, "an import write over the dirty bytes refreshes them and they are clean");
    fixture.gpu.writtenBack.clear();
    fixture.manager->FlushForCpuRead(Address(BasePage + 0), ResidencyPageBytes);
    Expect(fixture.gpu.writtenBack.size() == 2 && fixture.gpu.writtenBack[0].offset == 100 && fixture.gpu.writtenBack[0].bytes == 50 && fixture.gpu.writtenBack[1].offset == 250 && fixture.gpu.writtenBack[1].bytes == 50, "the write-back copies the two dirty runs left, not their subpage");
    Expect(fixture.manager->Dirty(BasePage + 0) == 0, "clean after the write-back");
    fixture.gpu.Drain();
}

void DirtyQueryWithoutLock() {
    scope = "dirty query without the lock";
    Fixture fixture(4, 1);
    fixture.Share(BasePage + 0);
    fixture.Use(0, BasePage + 0, 2);
    fixture.gpu.Drain();
    Expect(!fixture.manager->DirtyIn(Address(BasePage), 2 * ResidencyPageBytes), "nothing is dirty before a grant");
    fixture.manager->NoteGpuWritten(Address(BasePage + 0, 5) + 8, 16);
    Expect(fixture.manager->DirtyIn(Address(BasePage + 0, 5), ResidencySubpageBytes), "the subpage holding the granted bytes answers dirty");
    Expect(fixture.manager->DirtyIn(Address(BasePage), 2 * ResidencyPageBytes), "a range over the page answers dirty");
    Expect(!fixture.manager->DirtyIn(Address(BasePage + 0, 4), ResidencySubpageBytes) && !fixture.manager->DirtyIn(Address(BasePage + 0, 6), 2 * ResidencySubpageBytes), "the clean subpages around it answer clean");
    Expect(!fixture.manager->DirtyIn(Address(BasePage + 1), ResidencyPageBytes), "the clean page beside it answers clean");
    fixture.manager->FlushForCpuRead(Address(BasePage + 0), ResidencyPageBytes);
    Expect(!fixture.manager->DirtyIn(Address(BasePage), 2 * ResidencyPageBytes), "a write-back makes the range answer clean");
    fixture.manager->NoteGpuWritten(Address(BasePage + 0, 2), ResidencySubpageBytes);
    fixture.manager->Demote(Address(BasePage + 0), ResidencyPageBytes);
    Expect(!fixture.manager->DirtyIn(Address(BasePage), ResidencyPageBytes), "a page that left residency answers clean");
    fixture.gpu.Drain();
}

void CpuReadSettledWithoutLock() {
    scope = "CPU read settled without the lock";
    Fixture fixture(4, 1);
    fixture.Share(BasePage + 0);
    fixture.Share(BasePage + 1);
    fixture.Use(0, BasePage + 0, 2);
    fixture.gpu.Drain();
    const auto locked = [&](std::uint64_t page, std::uint32_t reads) {
        std::uint32_t count = 0;
        for (std::uint32_t read = 0; read < reads; ++read) {
            const auto address = Address(page, read % ResidencySubpages) + (read * 24u) % ResidencySubpageBytes;
            if (fixture.manager->AnyResident(address, 16) && !fixture.manager->CpuReadSettled(address, 16)) ++count;
        }
        return count;
    };
    Expect(locked(BasePage + 0, 256) == 0 && locked(BasePage + 1, 256) == 0, "reads over clean resident pages take no GPU mutex");
    const auto waits = fixture.gpu.waits;
    fixture.manager->FlushForCpuRead(Address(BasePage + 0), ResidencyPageBytes);
    Expect(fixture.gpu.waits == waits && fixture.gpu.writtenBack.empty(), "a clean resident read has nothing to wait for or write back");

    fixture.Use(1, BasePage + 0, 1, ResidencyAccess::Write);
    fixture.Grant(1, BasePage + 0, 0x0001);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 0, 0x0001, 7171, true);
    Expect(!fixture.manager->CpuReadSettled(Address(BasePage + 0), 16), "a read over a GPU-dirty subpage takes the GPU mutex");
    Expect(locked(BasePage + 1, 64) == 0, "the clean page beside it still takes none");
    fixture.manager->FlushForCpuRead(Address(BasePage + 0), 16);
    Expect(fixture.Stored(BasePage + 0, 0) == 7171, "the read over the dirty page gets the GPU's bytes");
    Expect(fixture.manager->CpuReadSettled(Address(BasePage + 0), ResidencyPageBytes), "the page is settled once its write-back landed");

    fixture.Use(1, BasePage + 1, 1, ResidencyAccess::Write);
    fixture.Grant(1, BasePage + 1, 0x0002);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 1, 0x0002, 7272, true);
    fixture.gpu.Drain();
    fixture.manager->PrepareImportWrite(Address(BasePage + 1, 1), 16);
    Expect(!fixture.manager->DirtyIn(Address(BasePage + 1, 1), 16), "the write-back in flight already published the page clean");
    Expect(!fixture.manager->CpuReadSettled(Address(BasePage + 1, 1), 16), "a read during an in-flight write-back takes the GPU mutex");
    fixture.manager->FlushForCpuRead(Address(BasePage + 1, 1), 16);
    Expect(fixture.Stored(BasePage + 1, 1) == 7272, "the read after the write-back landed gets the GPU's bytes");
    Expect(fixture.manager->CpuReadSettled(Address(BasePage + 1), ResidencyPageBytes), "the page is settled once the in-flight write-back settled");

    fixture.Use(2, BasePage + 0, 1, ResidencyAccess::Write);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 0, 0x0010, 7373);
    Expect(!fixture.manager->DirtyIn(Address(BasePage + 0, 4), 16) && !fixture.manager->CpuReadSettled(Address(BasePage + 0, 4), 16), "a read during a GPU write in flight to the page takes the GPU mutex");
    fixture.manager->FlushForCpuRead(Address(BasePage + 0, 4), 16);
    Expect(fixture.Stored(BasePage + 0, 4) == 7373, "the read after the writer settled gets the GPU's bytes");
    fixture.manager->QueueGpuWritten(Address(BasePage + 1, 4), 32);
    Expect(fixture.manager->CpuReadSettled(Address(BasePage + 1, 4), 32), "a queued recorded write went to guest memory, so its read takes no GPU mutex");
    fixture.manager->Apply();
    Expect(fixture.manager->CpuReadSettled(Address(BasePage + 1, 4), 32), "applying the recorded write refreshes the mirror and leaves the read settled");
    fixture.gpu.Drain();
    fixture.manager->Retire();
    Expect(locked(BasePage + 0, 256) == 0 && locked(BasePage + 1, 256) == 0, "after everything settled, reads over the resident pages take no GPU mutex again");
}

void WriteBackAllAtTeardown() {
    scope = "write-back of every dirty page at teardown";
    Fixture fixture(8, 1);
    for (std::uint32_t page = 0; page < 3; ++page) fixture.Share(BasePage + page);
    fixture.Use(0, BasePage + 0, 3);
    fixture.gpu.Drain();
    fixture.Grant(0, BasePage + 0, 0x0003);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 0, 0x0003, 501, true);
    fixture.Grant(1, BasePage + 2, 0x8000);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 2, 0x8000, 502, true);
    fixture.manager->NoteGpuWritten(Address(BasePage + 1, 4), ResidencySubpageBytes);
    fixture.manager->Demote(Address(BasePage + 1), ResidencyPageBytes);
    Expect(fixture.manager->Counts().dirty == 2, "two resident pages dirty before the teardown");
    fixture.manager->WriteBackAll();
    const auto counts = fixture.manager->Counts();
    Expect(counts.dirty == 0 && counts.accessGuarded == 0, "the teardown write-back left " + std::to_string(counts.dirty) + " dirty pages and " + std::to_string(counts.accessGuarded) + " access-guarded pages");
    Expect(fixture.Stored(BasePage + 0, 0) == 501 && fixture.Stored(BasePage + 0, 1) == 501 && fixture.Stored(BasePage + 2, 15) == 502, "the GPU's bytes reached guest memory before the call returned");
    for (std::uint32_t page = 0; page < 3; ++page) Expect(fixture.gpu.accessUnguarded.contains(BasePage + page), Name(BasePage + page) + " kept its access guard, the evicted one included");
    fixture.gpu.Drain();
}

void GrantDemotesPrivatePage() {
    scope = "a grant demotes a private page whose import has not settled";
    Fixture fixture(4, 1);
    fixture.Use(0, BasePage + 0, 2);
    Expect(!fixture.manager->ImportSettled(BasePage + 0) && !fixture.Armed(BasePage + 0) && fixture.manager->HotResident(BasePage + 0), "a private page admitted before a submission after it completed is unguarded and kept current from the write tracking");
    const auto demoted = fixture.manager->Counts().demoted;
    fixture.Grant(0, BasePage + 0, 0x0001);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 0, 0x0001, 31, true);
    Expect(!fixture.manager->Slot(BasePage + 0) && fixture.manager->Counts().demoted == demoted + 1, "a private page leaves residency before the store");
    Expect(fixture.manager->Slot(BasePage + 1).has_value(), "the page beside it stays");
    Expect(fixture.manager->Counts().dirty == 0, "an unsettled private page is never dirty in a mirror");
    fixture.gpu.Drain();
    Expect(fixture.Stored(BasePage + 0, 0) == 31, "the store went to guest memory");
}

void SettledPrivatePageKeepsDirtyMirror() {
    scope = "a settled private page is guarded and keeps a granted store in its mirror";
    Fixture fixture(4, 1);
    fixture.Use(0, BasePage + 0, 2);
    fixture.manager->Apply();
    Expect(!fixture.manager->ImportSettled(BasePage + 0) && !fixture.Armed(BasePage + 0), "the import is not settled while the submission issued after the admission runs");
    fixture.gpu.Drain();
    Expect(!fixture.manager->ImportSettled(BasePage + 0) && !fixture.Armed(BasePage + 0), "a completed submission is not seen before an ordering point");
    fixture.manager->Apply();
    Expect(fixture.manager->ImportSettled(BasePage + 0) && fixture.manager->ImportSettled(BasePage + 1), "the ordering point after the submission completed settles both imports");
    Expect(fixture.Armed(BasePage + 0) && !fixture.manager->HotResident(BasePage + 0), "the settled CPU-writable page is guarded, no longer kept current from the write tracking");
    Expect(fixture.manager->Counts().importsSettled == 2, "two imports settled, not " + std::to_string(fixture.manager->Counts().importsSettled));
    const auto demoted = fixture.manager->Counts().demoted;
    fixture.Grant(0, BasePage + 0, 0x0001);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 0, 0x0001, 31, true);
    Expect(fixture.manager->Slot(BasePage + 0).has_value() && fixture.manager->Counts().demoted == demoted, "the settled private page stays resident");
    Expect(fixture.manager->Dirty(BasePage + 0) == 0x0001, "the store stays dirty in its mirror");
    fixture.gpu.Drain();
    Expect(fixture.Stored(BasePage + 0, 0) == 0 && fixture.gpu.writtenBack.empty(), "the store has not reached guest memory before a CPU read");
    fixture.manager->FlushForCpuRead(Address(BasePage + 0), 16);
    Expect(fixture.Stored(BasePage + 0, 0) == 31 && fixture.manager->Dirty(BasePage + 0) == 0, "a CPU read writes the store back");
    fixture.gpu.CpuWrite(BasePage + 0, 0x0002, 41);
    Expect(fixture.gpu.faults == 1, "a CPU store into the guarded private page faults");
    fixture.manager->Prepare();
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0002, 41);
    fixture.gpu.Drain();
}

void NewerImportRestartsSettle() {
    scope = "a newer host import of a private page before its settle restarts the settle";
    Fixture fixture(4, 1);
    fixture.gpu.HostImport(BasePage + 0);
    fixture.Use(0, BasePage + 0);
    fixture.manager->Apply();
    fixture.gpu.Drain();
    fixture.gpu.HostImport(BasePage + 0);
    const auto admitted = fixture.manager->Counts().unsettledGuards;
    fixture.manager->Apply();
    Expect(!fixture.manager->ImportSettled(BasePage + 0) && !fixture.Armed(BasePage + 0), "a page imported again after its admission is not settled by the submission issued before that import");
    Expect(fixture.manager->Counts().unsettledGuards == admitted + 1, std::to_string(fixture.manager->Counts().unsettledGuards - admitted) + " settles restarted, expected one");
    Expect(fixture.manager->HotResident(BasePage + 0), "the page whose settle restarted is kept current from the write tracking");
    fixture.manager->Apply();
    Expect(!fixture.manager->ImportSettled(BasePage + 0) && !fixture.Armed(BasePage + 0), "the restarted settle completed with no submission issued after the newer import");
    static_cast<void>(fixture.gpu.CallerToken(0));
    fixture.manager->Apply();
    Expect(!fixture.manager->ImportSettled(BasePage + 0) && !fixture.Armed(BasePage + 0), "the restarted settle completed while the submission issued after the newer import runs");
    fixture.gpu.Drain();
    fixture.manager->Apply();
    Expect(fixture.manager->ImportSettled(BasePage + 0) && fixture.Armed(BasePage + 0), "the page is not settled and guarded after the submission issued after the newer import completed");
    Expect(fixture.gpu.hostSettled[BasePage + 0] == 2, "the page settled at host import stamp " + std::to_string(fixture.gpu.hostSettled[BasePage + 0]) + ", expected the newer import's 2");
    Expect(fixture.manager->Counts().importsSettled == 1 && fixture.manager->Counts().unsettledGuards == admitted + 1, "one import settled after one restart");
}

void GrantThroughAliasDemotesMirror() {
    scope = "a grant through another view demotes the mirrored view";
    Fixture fixture(8, 1);
    fixture.gpu.Alias(BasePage + 0, BasePage + 5);
    fixture.Use(0, BasePage + 0);
    fixture.Use(0, BasePage + 5);
    fixture.gpu.Drain();
    Expect(fixture.manager->Slot(BasePage + 0) && !fixture.manager->Slot(BasePage + 5), "one view holds the mirror");
    fixture.Grant(0, BasePage + 5, 0x0008);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 5, 0x0008, 77, true);
    Expect(!fixture.manager->Slot(BasePage + 0) && fixture.manager->Counts().aliasDemoted == 1, "the mirrored view leaves residency before a store through the other view");
    fixture.gpu.Drain();
    fixture.Use(0, BasePage + 0);
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0008, 77);
    fixture.gpu.Drain();
    Expect(fixture.Stored(BasePage + 0, 3) == 77, "guest memory holds the store for both views");
}

void TouchedPagesAdmitted() {
    scope = "pages touched in a frame are admitted for the next";
    Fixture fixture(8, 1);
    fixture.Use(0, BasePage + 0, 4);
    fixture.gpu.Drain();
    const std::array<std::uint64_t, 3> first{BasePage + 4, BasePage + 5, BasePage + 1};
    fixture.manager->AdmitTouched(first);
    Expect(fixture.manager->Slot(BasePage + 4) && fixture.manager->Slot(BasePage + 5) && fixture.manager->Slot(BasePage + 1), "the touched pages are resident");
    Expect(!fixture.manager->Slot(BasePage + 0) && !fixture.manager->Slot(BasePage + 2), "the least recently used untouched pages left for them");
    Expect(fixture.manager->Counts().touchAdmitted == 2, "two pages admitted from the touch records");
    fixture.gpu.Drain();
    const std::array<std::uint64_t, 5> second{BasePage + 1, BasePage + 4, BasePage + 5, BasePage + 6, BasePage + 7};
    fixture.manager->AdmitTouched(second);
    Expect(fixture.manager->Slot(BasePage + 6) && !fixture.manager->Slot(BasePage + 3), "an untouched page makes room for a touched one");
    Expect(!fixture.manager->Slot(BasePage + 7) && fixture.manager->Counts().touchDeferred == 1, "a page touched in the same frame is never evicted for another touched page");
    Expect(fixture.manager->Slot(BasePage + 1) && fixture.manager->Slot(BasePage + 4) && fixture.manager->Slot(BasePage + 5), "the pages touched again stay");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 6, 0x0002, (std::uint64_t{6} << 8) | 1u);
    fixture.gpu.Drain();
}

void TouchedPagesTakeEvictedSlotsWhenTheyRetire() {
    scope = "touched pages take the slots they evicted once the slots retire";
    Fixture fixture(8, 1);
    fixture.Use(0, BasePage + 0, 4);
    fixture.gpu.Drain();
    fixture.Use(0, BasePage + 3);
    const std::array<std::uint64_t, 2> touched{BasePage + 4, BasePage + 5};
    fixture.manager->AdmitTouched(touched);
    Expect(!fixture.manager->Slot(BasePage + 0) && !fixture.manager->Slot(BasePage + 1), "the two least recently used pages left for the touched pages");
    Expect(fixture.manager->Slot(BasePage + 3).has_value(), "the page in use stays");
    fixture.gpu.Drain();
    fixture.manager->Apply();
    Expect(fixture.manager->Slot(BasePage + 4) && fixture.manager->Slot(BasePage + 5), "both touched pages admitted into the slots they evicted once those retired");
    Expect(fixture.manager->Counts().touchAdmitted == 2 && fixture.manager->Counts().touchDeferred == 0, "two pages admitted from the touch records, none deferred, not " + std::to_string(fixture.manager->Counts().touchAdmitted) + " and " + std::to_string(fixture.manager->Counts().touchDeferred));
    Expect(fixture.manager->Counts().resident == ChunkSlots, "the mirrors stay within the budget");
    fixture.gpu.Drain();
}

void MeasuredBudget() {
    scope = "measured budget";
    Fixture fixture(16, 8);
    fixture.gpu.measuredBudget = 3ull * ChunkSlots * ResidencyPageBytes;
    fixture.manager->SetBudget(ResidencyMeasuredBudget);
    fixture.Use(0, BasePage + 0, 12);
    Expect(fixture.manager->Counts().chunks == 2, "chunks grow to one chunk below the measured budget, not " + std::to_string(fixture.manager->Counts().chunks));
    fixture.gpu.Drain();
    fixture.gpu.measuredBudget = std::uint64_t{ChunkSlots} * ResidencyPageBytes + 1;
    fixture.manager->ApplyBudget();
    fixture.gpu.Drain();
    fixture.manager->Retire();
    Expect(fixture.manager->Counts().chunks == 1 && fixture.manager->Counts().resident <= ChunkSlots, "a lower measurement drains chunks down to it");
    fixture.gpu.measuredBudget = 2ull * ChunkSlots * ResidencyPageBytes;
    fixture.manager->ApplyBudget();
    fixture.manager->Prepare();
    fixture.Use(0, BasePage + 4, 8);
    Expect(fixture.manager->Counts().chunks == 1, "a measurement less than a chunk above the live chunks grows none");
    fixture.gpu.Drain();
    fixture.gpu.measuredBudget = 3ull * ChunkSlots * ResidencyPageBytes;
    fixture.manager->Prepare();
    fixture.Use(0, BasePage + 4, 8);
    Expect(fixture.manager->Counts().chunks == 2, "a higher measurement grows at the next admission");
}

void FullResidencyWhenTheGuestFits() {
    scope = "full residency only when the budget covers the guest";
    Fixture fixture(8, 4);
    for (std::uint32_t page = 0; page < 4; ++page) fixture.Share(BasePage + page);
    fixture.gpu.wholeImports = {ResidencyRange{Address(BasePage), 4 * ResidencyPageBytes}};
    fixture.gpu.importsGeneration = 2;
    fixture.gpu.guestBytes = 32 * ResidencyPageBytes;
    fixture.manager->Prepare();
    Expect(!fixture.manager->FullResidency(), "a budget below the guest size is not full residency");
    for (std::uint32_t page = 0; page < 4; ++page) Expect(fixture.manager->Slot(BasePage + page).has_value(), "shared " + Name(BasePage + page) + " admitted eagerly under a budget that holds it");
    fixture.gpu.guestBytes = 16 * ResidencyPageBytes;
    fixture.manager->Prepare();
    Expect(fixture.manager->FullResidency(), "a budget of the guest size is full residency");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 2, 0x0001, std::uint64_t{2} << 8);
    fixture.gpu.Drain();
}

void EagerAdmissionFillsTheBudget() {
    scope = "eager admission fills the budget, discovery evicts below it";
    Fixture fixture(16, 1);
    for (std::uint32_t page = 0; page < 12; ++page) fixture.Share(BasePage + page);
    fixture.gpu.wholeImports = {ResidencyRange{Address(BasePage), 12 * ResidencyPageBytes}};
    fixture.gpu.importsGeneration = 2;
    fixture.gpu.guestBytes = 64 * ResidencyPageBytes;
    fixture.manager->Prepare();
    fixture.manager->Prepare();
    auto counts = fixture.manager->Counts();
    Expect(!fixture.manager->FullResidency(), "a budget of one chunk for a 64-page guest is not full residency");
    Expect(counts.chunks == 1 && counts.resident == ChunkSlots && counts.free == 0, "eager admission fills the one-chunk budget with " + std::to_string(ChunkSlots) + " shared pages, not " + std::to_string(counts.resident) + " in " + std::to_string(counts.chunks) + " chunks");
    for (std::uint32_t page = 0; page < ChunkSlots; ++page) Expect(fixture.manager->Slot(BasePage + page).has_value(), "shared " + Name(BasePage + page) + " admitted eagerly in commit order");
    fixture.Use(0, BasePage + 1);
    fixture.Use(0, BasePage + 2);
    fixture.Use(0, BasePage + 3);
    fixture.gpu.Drain();
    fixture.Use(0, BasePage + 8);
    fixture.gpu.Drain();
    fixture.manager->Retire();
    fixture.Use(0, BasePage + 8);
    fixture.gpu.Drain();
    fixture.manager->Prepare();
    counts = fixture.manager->Counts();
    Expect(!fixture.manager->Slot(BasePage + 0), "discovery evicts the least recently used eager page (" + Name(BasePage + 0) + ")");
    Expect(fixture.manager->Slot(BasePage + 8).has_value(), "the discovered " + Name(BasePage + 8) + " is admitted under the budget");
    for (std::uint32_t page = 1; page < ChunkSlots; ++page) Expect(fixture.manager->Slot(BasePage + page).has_value(), "recently used " + Name(BasePage + page) + " kept");
    Expect(counts.chunks == 1 && counts.resident == ChunkSlots, "the mirrors stay within the one-chunk budget");
    for (std::uint32_t page = ChunkSlots; page < 12; ++page) {
        if (page != 8) Expect(!fixture.manager->Slot(BasePage + page), "eager admission ended at the budget: " + Name(BasePage + page) + " not admitted into the freed slot");
    }
}

class FakeOwner final : public IResidencyOwner {
public:
    FakeOwner(Fixture& fixture, std::uint64_t value) : fixture(fixture), value(value) {}

    void Retile(std::span<const ResidencyRange> ranges) override {
        for (const auto& range : ranges) {
            retiled.push_back(range);
            for (auto at = range.address; at < range.address + range.bytes;) {
                const auto page = at >> ResidencyPageShift;
                const auto end = std::min(range.address + range.bytes, (page + 1) << ResidencyPageShift);
                const auto mask = SubpageMask(page, at, end - at);
                if (fixture.manager->WriteSlot(page)) {
                    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, page, mask, value, true);
                    fixture.manager->NoteGpuWritten(at, end - at, true);
                } else {
                    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, page, mask, value);
                    fixture.manager->NoteImportWritten(at, end - at);
                }
                at = end;
            }
        }
    }

    Fixture& fixture;
    std::uint64_t value;
    std::vector<ResidencyRange> retiled;
};

void OwnedPagesRetiledBeforeReaders() {
    scope = "owned pages retiled before readers";
    Fixture fixture(8, 1);
    fixture.Share(BasePage + 0);
    FakeOwner owner(fixture, 901);
    fixture.manager->Own(Address(BasePage + 0), ResidencyPageBytes, &owner);
    Expect(fixture.manager->Slot(BasePage + 0).has_value() && fixture.manager->OwnerOf(BasePage + 0) == &owner, "an owned shared page is resident and owned");
    fixture.gpu.Drain();
    fixture.manager->MarkStale(Address(BasePage + 0, 2), 2 * ResidencySubpageBytes, &owner);
    Expect(fixture.manager->Stale(BasePage + 0) == 0x000c && owner.retiled.empty(), "the owner's writes into a resident shared page stay in the owner, stale");
    Expect(fixture.manager->DirtyIn(Address(BasePage + 0, 3), 16), "a stale subpage answers dirty without the lock");
    Expect(fixture.manager->GuardedStaleIn(Address(BasePage + 0), ResidencyPageBytes, &owner) && !fixture.manager->GuardedStaleIn(Address(BasePage + 0), ResidencyPageBytes, nullptr), "the owner's stale subpages of a resident shared page are access-guarded");
    fixture.manager->RetileAllOwned();
    Expect(owner.retiled.size() == 1 && owner.retiled[0].address == Address(BasePage + 0, 2) && owner.retiled[0].bytes == 2 * ResidencySubpageBytes, "the retile before a reader asks the owner for the two stale subpages");
    Expect(fixture.manager->Stale(BasePage + 0) == 0 && fixture.manager->Dirty(BasePage + 0) == 0x000c, "after the retile the subpages are dirty in the mirror, not stale");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0004, 901);
    fixture.gpu.Drain();
    Expect(fixture.Stored(BasePage + 0, 2) == 2, "the rendered bytes stay out of guest memory until a CPU read");
    fixture.manager->FlushForCpuRead(Address(BasePage + 0), ResidencyPageBytes);
    fixture.gpu.Drain();
    Expect(fixture.Stored(BasePage + 0, 2) == 901 && fixture.Stored(BasePage + 0, 3) == 901 && fixture.Stored(BasePage + 0, 4) == 4, "a CPU read writes the rendered subpages back");
}

void OwnedRangeMarkedStaleInOneUpdate() {
    scope = "an owned range marked stale in one update";
    Fixture fixture(8, 1);
    fixture.Share(BasePage + 3);
    fixture.Share(BasePage + 4);
    FakeOwner owner(fixture, 904);
    fixture.manager->Own(Address(BasePage + 3), 2 * ResidencyPageBytes, &owner);
    fixture.gpu.Drain();
    fixture.gpu.sampleStaleAtGuard = true;
    fixture.manager->MarkStale(Address(BasePage + 3, 8), ResidencyPageBytes, &owner);
    fixture.gpu.sampleStaleAtGuard = false;
    Expect(fixture.gpu.staleAtGuard.size() == 2, "both pages of the range are access-guarded");
    Expect(std::none_of(fixture.gpu.staleAtGuard.begin(), fixture.gpu.staleAtGuard.end(), [](bool stale) { return stale; }), "no page of the range is published stale before the whole range is guarded");
    Expect(fixture.manager->Stale(BasePage + 3) == 0xff00 && fixture.manager->Stale(BasePage + 4) == 0x00ff && owner.retiled.empty(), "the range's subpages of both pages are stale in the owner");
}

void CpuReadRetilesStalePages() {
    scope = "a CPU read retiles stale owned pages";
    Fixture fixture(8, 1);
    fixture.Share(BasePage + 1);
    FakeOwner owner(fixture, 902);
    fixture.manager->Own(Address(BasePage + 1), ResidencyPageBytes, &owner);
    fixture.gpu.Drain();
    fixture.manager->MarkStale(Address(BasePage + 1, 7), ResidencySubpageBytes, &owner);
    fixture.manager->FlushForCpuRead(Address(BasePage + 1, 7), 4);
    fixture.gpu.Drain();
    Expect(owner.retiled.size() == 1 && fixture.Stored(BasePage + 1, 7) == 902, "the CPU read of a stale subpage gets the owner's bytes");
    Expect(fixture.manager->Stale(BasePage + 1) == 0 && fixture.manager->Dirty(BasePage + 1) == 0 && fixture.manager->Counts().stalePages == 0, "the page is neither stale nor dirty after the read");
}

void PrivateOwnedPagesHeldForReaders() {
    scope = "owned private pages held for their readers";
    Fixture fixture(8, 1);
    FakeOwner owner(fixture, 903);
    fixture.manager->Own(Address(BasePage + 2), ResidencyPageBytes, &owner);
    fixture.gpu.Drain();
    const auto guarded = fixture.manager->Counts().accessGuarded;
    fixture.manager->MarkStale(Address(BasePage + 2, 1), ResidencySubpageBytes, &owner);
    Expect(owner.retiled.empty() && fixture.manager->Stale(BasePage + 2) == 0x0002 && fixture.manager->Counts().eagerRetiles == 0 && fixture.manager->Counts().accessGuarded == guarded, "a private page holds its stale subpage, unguarded, until a reader");
    Expect(fixture.manager->StaleIn(Address(BasePage + 2, 1), 4) && !fixture.manager->StaleIn(Address(BasePage + 2, 2), 4), "the unlocked stale query names the stale subpage only");
    Expect(!fixture.manager->GuardedStaleIn(Address(BasePage + 2), ResidencyPageBytes, &owner), "a private page's stale subpage is not access-guarded");
    fixture.manager->FlushForCpuRead(Address(BasePage + 2, 1), 4);
    fixture.gpu.Drain();
    Expect(owner.retiled.size() == 1 && fixture.manager->Stale(BasePage + 2) == 0, "a driver read retiles the private page");
    Expect(fixture.Stored(BasePage + 2, 1) == 903 && fixture.manager->Dirty(BasePage + 2) == 0, "the bytes reached guest memory, and the private page is not dirty");
}

void OwnedPagesExemptAndForgotten() {
    scope = "retiles skip exempt owners, and a forgotten owner leaves nothing stale";
    Fixture fixture(8, 1);
    fixture.Share(BasePage + 3);
    fixture.Share(BasePage + 4);
    FakeOwner first(fixture, 905);
    FakeOwner second(fixture, 906);
    fixture.manager->Own(Address(BasePage + 3), ResidencyPageBytes, &first);
    fixture.manager->Own(Address(BasePage + 4), ResidencyPageBytes, &second);
    fixture.gpu.Drain();
    fixture.manager->MarkStale(Address(BasePage + 3), ResidencySubpageBytes, &first);
    fixture.manager->MarkStale(Address(BasePage + 4), ResidencySubpageBytes, &second);
    Expect(fixture.manager->StaleOfOthers(Address(BasePage + 3), 2 * ResidencyPageBytes, &first) && !fixture.manager->StaleOfOthers(Address(BasePage + 3), ResidencyPageBytes, &first), "the stale pages of other owners are told from an owner's own");
    Expect(fixture.manager->HoldOwners(Address(BasePage + 3), 2 * ResidencyPageBytes).size() == 2 && fixture.manager->HoldStaleOwners().size() == 2, "both owners are found by their pages");
    const std::array<const IResidencyOwner*, 1> exempt{&first};
    Expect(fixture.manager->RetileOwned(Address(BasePage + 3), 2 * ResidencyPageBytes, exempt), "the retile over both pages reports a retile");
    Expect(first.retiled.empty() && second.retiled.size() == 1 && fixture.manager->Stale(BasePage + 3) == 0x0001 && fixture.manager->Stale(BasePage + 4) == 0, "the exempt owner keeps its stale subpage, the other is retiled");
    const auto serial = fixture.manager->OwnershipSerial();
    fixture.manager->Forget(&first);
    Expect(fixture.manager->OwnerOf(BasePage + 3) == nullptr && fixture.manager->Stale(BasePage + 3) == 0 && !fixture.manager->StaleIn(Address(BasePage + 3), ResidencyPageBytes) && fixture.manager->OwnershipSerial() != serial, "a forgotten owner leaves no record and nothing stale");
    Expect(!fixture.manager->RetileAllOwned() && first.retiled.empty(), "a forgotten owner is never asked to retile");
    fixture.manager->FlushForCpuRead(Address(BasePage + 3), 2 * ResidencyPageBytes);
    fixture.gpu.Drain();
}

void ClearStaleSparesOtherSubpages() {
    scope = "an owner clears its own stale subpages without a retile";
    Fixture fixture(8, 1);
    fixture.Share(BasePage + 5);
    fixture.Share(BasePage + 6);
    FakeOwner first(fixture, 907);
    FakeOwner second(fixture, 908);
    fixture.manager->Own(Address(BasePage + 5), ResidencyPageBytes, &first);
    fixture.manager->Own(Address(BasePage + 6), ResidencyPageBytes, &second);
    fixture.gpu.Drain();
    fixture.manager->MarkStale(Address(BasePage + 5, 1), 2 * ResidencySubpageBytes, &first);
    fixture.manager->MarkStale(Address(BasePage + 6, 4), ResidencySubpageBytes, &second);
    fixture.manager->ClearStale(Address(BasePage + 5, 1), ResidencySubpageBytes, &first);
    fixture.manager->ClearStale(Address(BasePage + 6), ResidencyPageBytes, &first);
    Expect(fixture.manager->Stale(BasePage + 5) == 0x0004 && fixture.manager->Stale(BasePage + 6) == 0x0010 && first.retiled.empty() && second.retiled.empty(), "only the owner's cleared subpage leaves the stale state, and no owner retiles");
    Expect(!fixture.manager->StaleIn(Address(BasePage + 5, 1), 4) && fixture.manager->StaleIn(Address(BasePage + 5, 2), 4), "the unlocked stale query follows the cleared subpage");
    fixture.manager->ClearStale(Address(BasePage + 5), ResidencyPageBytes, &first);
    Expect(fixture.manager->Stale(BasePage + 5) == 0 && fixture.manager->Counts().stalePages == 1, "clearing the last stale subpage of a page drops it from the stale set");
    fixture.manager->RetileAllOwned();
    Expect(first.retiled.empty() && second.retiled.size() == 1, "a retile asks only the owner that still has stale subpages");
    fixture.manager->FlushForCpuRead(Address(BasePage + 5), 2 * ResidencyPageBytes);
    fixture.gpu.Drain();
}

void UnguardedStalePagesRetiledForSignals() {
    scope = "a completion signal retiles the stale pages no access guard covers";
    Fixture fixture(8, 1);
    fixture.Share(BasePage + 5);
    FakeOwner shared(fixture, 909);
    FakeOwner privateOwner(fixture, 910);
    fixture.manager->Own(Address(BasePage + 5), ResidencyPageBytes, &shared);
    fixture.manager->Own(Address(BasePage + 6), ResidencyPageBytes, &privateOwner);
    fixture.gpu.Drain();
    fixture.manager->MarkStale(Address(BasePage + 5, 2), ResidencySubpageBytes, &shared);
    fixture.manager->MarkStale(Address(BasePage + 6, 3), ResidencySubpageBytes, &privateOwner);
    Expect(fixture.manager->RetileUnguarded(), "the signal retiles the unguarded stale page");
    Expect(shared.retiled.empty() && privateOwner.retiled.size() == 1 && fixture.manager->Stale(BasePage + 5) == 0x0004 && fixture.manager->Stale(BasePage + 6) == 0, "the guarded shared page stays stale, the private page is retiled");
    Expect(!fixture.manager->RetileUnguarded(), "a second signal finds nothing unguarded");
    fixture.gpu.Drain();
    Expect(fixture.Stored(BasePage + 6, 3) == 910, "the private page's bytes reached guest memory");
    fixture.manager->FlushForCpuRead(Address(BasePage + 5), ResidencyPageBytes);
    fixture.gpu.Drain();
}

void StaleOwnedPagesKeptAndDisowned() {
    scope = "stale owned pages are kept, and disowning retiles them";
    Fixture fixture(8, 1);
    FakeOwner owner(fixture, 904);
    for (std::uint32_t page = 0; page < 4; ++page) fixture.Share(BasePage + page);
    fixture.Use(0, BasePage + 0, 4);
    fixture.gpu.Drain();
    fixture.manager->Own(Address(BasePage + 0), ResidencyPageBytes, &owner);
    fixture.manager->MarkStale(Address(BasePage + 0), ResidencySubpageBytes, &owner);
    fixture.Use(0, BasePage + 1, 3);
    fixture.gpu.Drain();
    fixture.Use(0, BasePage + 4);
    Expect(fixture.manager->Slot(BasePage + 0).has_value() && !fixture.manager->Slot(BasePage + 1), "admission past the budget evicts the least recently used page that is not stale");
    fixture.gpu.Drain();
    fixture.manager->Disown(&owner);
    Expect(owner.retiled.size() == 1 && fixture.manager->OwnerOf(BasePage + 0) == nullptr && fixture.manager->Counts().ownedPages == 0, "disowning retiles the stale subpage and forgets the records");
    fixture.manager->FlushForCpuRead(Address(BasePage + 0), ResidencyPageBytes);
    fixture.gpu.Drain();
    Expect(fixture.Stored(BasePage + 0, 0) == 904, "the disowned page's bytes reach guest memory");
}

void DemotionRetilesStalePages() {
    scope = "a demotion retiles stale owned pages first";
    Fixture fixture(8, 1);
    fixture.Share(BasePage + 3);
    FakeOwner owner(fixture, 905);
    fixture.manager->Own(Address(BasePage + 3), ResidencyPageBytes, &owner);
    fixture.gpu.Drain();
    fixture.manager->MarkStale(Address(BasePage + 3, 9), ResidencySubpageBytes, &owner);
    fixture.manager->DemoteWritable();
    fixture.gpu.Drain();
    Expect(!fixture.manager->Slot(BasePage + 3) && owner.retiled.size() == 1 && fixture.Stored(BasePage + 3, 9) == 905, "an unbounded writer's demotion retiled the stale subpage and wrote it back");
}

class DemotingOwner final : public IResidencyOwner {
public:
    explicit DemotingOwner(Fixture& fixture) : fixture(fixture) {}

    void Retile(std::span<const ResidencyRange> ranges) override {
        ++retiles;
        for (const auto& range : ranges) {
            std::thread other([&] { fixture.manager->Demote(range.address & ~(ResidencyPageBytes - 1), ResidencyPageBytes); });
            other.join();
        }
    }

    Fixture& fixture;
    int retiles = 0;
};

void PageLeftDuringDemotionRetileLeavesOnce() {
    scope = "a page another thread demotes while a demotion retiles leaves its mirror once";
    Fixture fixture(8, 1);
    fixture.Share(BasePage + 3);
    DemotingOwner owner(fixture);
    fixture.manager->Own(Address(BasePage + 3), ResidencyPageBytes, &owner);
    fixture.gpu.Drain();
    fixture.manager->MarkStale(Address(BasePage + 3, 9), ResidencySubpageBytes, &owner);
    fixture.manager->DemoteWritable();
    fixture.gpu.Drain();
    Expect(owner.retiles == 1 && !fixture.manager->Slot(BasePage + 3), "the stale page was retiled and left its mirror");
    Expect(fixture.manager->Counts().demoted == 1, "the page left once, not again by the demotion that held its id across the retile: demoted " + std::to_string(fixture.manager->Counts().demoted));
    fixture.Use(0, BasePage + 4);
    fixture.Use(0, BasePage + 5);
    const auto first = fixture.manager->Slot(BasePage + 4);
    const auto second = fixture.manager->Slot(BasePage + 5);
    Expect(first && second && (first->chunk != second->chunk || first->slot != second->slot), "two pages admitted after the demotion hold their own mirror slots");
}

void AdmissionRule() {
    scope = "admission rule";
    Fixture fixture(8, 1);
    fixture.gpu.imports[BasePage + 1].attributes.cpuWritable = false;
    fixture.gpu.imports[BasePage + 2].attributes.whole = false;
    fixture.Use(0, BasePage + 0, 4);
    Expect(fixture.manager->Slot(BasePage + 0) && fixture.manager->Slot(BasePage + 1) && fixture.manager->Slot(BasePage + 3), "whole pages are admitted, CPU-writable or not");
    Expect(!fixture.manager->Slot(BasePage + 2), "a page no single import covers stays on its imports");
    Expect(fixture.manager->Counts().inadmissible == 1, "one inadmissible page counted");
    Expect(fixture.manager->Slot(BasePage + 1)->slot == fixture.manager->Slot(BasePage + 0)->slot + 1, "a range admitted at once takes consecutive slots");
}

void CanonicalAliases() {
    scope = "canonical aliases";
    Fixture fixture(8, 1);
    fixture.gpu.Alias(BasePage + 0, BasePage + 5);
    fixture.Use(0, BasePage + 0);
    fixture.Use(0, BasePage + 5);
    Expect(fixture.manager->Slot(BasePage + 0) && !fixture.manager->Slot(BasePage + 5), "one canonical alias per backing is admitted");
    Expect(fixture.manager->Counts().aliasSkipped == 1, "the second alias is counted");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Write, BasePage + 5, 0x0008, 77);
    fixture.manager->NoteImportWritten(Address(BasePage + 5, 3), ResidencySubpageBytes);
    Expect(!fixture.gpu.copiedIn.empty() && fixture.gpu.copiedIn.back().page == BasePage + 0 && fixture.gpu.copiedIn.back().offset == 3 * ResidencySubpageBytes, "a write through one alias refreshes the other alias's mirror");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0008, 77);
    fixture.gpu.Drain();
    fixture.manager->Prepare();
    fixture.gpu.CpuWrite(BasePage + 5, 0x0001, 66);
    Expect(fixture.gpu.reported.contains(BasePage + 0), "a store through the alias reports the guarded backing's pages");
    fixture.manager->Prepare();
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0001, 66);
    fixture.gpu.Drain();
    fixture.manager->Demote(Address(BasePage + 5), ResidencyPageBytes);
    Expect(!fixture.manager->Slot(BasePage + 0), "demoting one alias demotes the resident one");
    fixture.Use(0, BasePage + 5);
    fixture.Use(0, BasePage + 0);
    Expect(fixture.manager->Slot(BasePage + 5) && !fixture.manager->Slot(BasePage + 0), "after the demotion the other alias can be the canonical one");
}

void EvictedAliasKeepsSharedGuard() {
    scope = "evicted alias keeps the shared guard";
    Fixture fixture(8, 1);
    fixture.gpu.Alias(BasePage + 5, BasePage + 7);
    fixture.Use(0, BasePage + 7);
    fixture.gpu.Drain();
    for (std::uint32_t page = 1; page < 4; ++page) {
        fixture.Use(0, BasePage + page);
        fixture.gpu.Drain();
    }
    Expect(fixture.manager->Slot(BasePage + 7) && fixture.Armed(BasePage + 7), "the alias is resident and its backing guarded");
    fixture.Use(0, BasePage + 4, 2);
    Expect(!fixture.manager->Slot(BasePage + 7) && fixture.manager->Slot(BasePage + 4) && fixture.manager->Slot(BasePage + 5), "one use evicts the alias and admits the other view of its backing");
    Expect(fixture.Armed(BasePage + 5), "the evicted alias's unguard disarmed the guard the admitted view armed");
    fixture.gpu.Drain();
}

void OversizedRangeWithoutMirrorsQueriesNoImports() {
    scope = "a read range over the admission limit without resident pages";
    constexpr std::uint64_t largest = 2 * ResidencyPageBytes;
    Fixture fixture(8, 2);
    for (std::uint32_t page = 0; page < 8; ++page) fixture.Share(BasePage + page);
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> wide{{Address(BasePage + 0), Address(BasePage + 6)}};
    const auto queries = fixture.gpu.importQueries;
    const auto before = fixture.manager->Counts();
    fixture.gpu.caller = 0;
    fixture.manager->UseRanges(wide, largest);
    const auto after = fixture.manager->Counts();
    Expect(fixture.gpu.importQueries == queries, "a build reading 6 import-served pages over a 2-page limit queried " + std::to_string(fixture.gpu.importQueries - queries) + " page imports under the device lock, expected none");
    Expect(after.resident == 0 && after.mixedRanges == before.mixedRanges && after.demotedRanges == before.demotedRanges, "and admitted, demoted or counted nothing");
    fixture.Use(0, BasePage + 2);
    Expect(fixture.manager->Slot(BasePage + 2).has_value(), "page 2 admitted");
    const auto mixed = fixture.manager->Counts().mixedRanges;
    fixture.manager->UseRanges(wide, largest);
    Expect(fixture.manager->Counts().mixedRanges == mixed + 1, "the same range with a resident page is still decided page by page");
    fixture.gpu.Drain();
}

void UnchangedRangesReused() {
    scope = "unchanged ranges reused";
    constexpr std::uint64_t largest = std::uint64_t{8} << 20;
    Fixture fixture(8, 2);
    fixture.Share(BasePage + 4);
    fixture.gpu.collectEpoch = 1;
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges{{Address(BasePage + 0), Address(BasePage + 2)}, {Address(BasePage + 4), Address(BasePage + 5)}};
    const auto draw = [&] {
        fixture.gpu.caller = 0;
        fixture.manager->UseRanges(ranges, largest);
    };
    draw();
    draw();
    const auto queries = fixture.gpu.importQueries;
    const auto before = fixture.manager->Counts();
    for (int index = 0; index < 8; ++index) draw();
    auto counts = fixture.manager->Counts();
    Expect(counts.rangeSetsReused - before.rangeSetsReused == 8 && counts.rangeSetsUsed == before.rangeSetsUsed, "8 draws with unchanged ranges and residency reuse the previous use");
    Expect(fixture.gpu.importQueries == queries, "and do no per-page work");
    Expect(fixture.manager->Slot(BasePage + 0) && fixture.manager->Slot(BasePage + 1) && fixture.manager->Slot(BasePage + 4), "the ranges are resident");

    const auto rebuilds = [&](const char* what) {
        const auto used = fixture.manager->Counts().rangeSetsUsed;
        draw();
        Expect(fixture.manager->Counts().rangeSetsUsed == used + 1, std::string("a draw after ") + what + " rebuilds");
    };
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> other{{Address(BasePage + 0), Address(BasePage + 1)}};
    fixture.manager->UseRanges(other, largest);
    fixture.manager->UseRanges(other, largest);
    {
        const auto used = fixture.manager->Counts().rangeSetsUsed;
        draw();
        Expect(fixture.manager->Counts().rangeSetsUsed == used, "another draw's ranges in between do not invalidate the first");
    }

    fixture.manager->Demote(Address(BasePage + 1), ResidencyPageBytes);
    Expect(!fixture.manager->Slot(BasePage + 1), "the demote took page 1 out");
    rebuilds("a demote");
    Expect(fixture.manager->Slot(BasePage + 1).has_value(), "the rebuild admits page 1 again");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 1, 0x0001, (std::uint64_t{1} << 8) | 0);
    fixture.gpu.Drain();
    draw();
    draw();

    fixture.manager->AdmitTouched(std::vector<std::uint64_t>{BasePage + 6});
    Expect(fixture.manager->Slot(BasePage + 6).has_value(), "page 6 admitted by a touch");
    rebuilds("an admission");
    draw();

    fixture.gpu.CpuWrite(BasePage + 0, 0x0001, 77);
    rebuilds("a CPU write to a page in the range");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 0, 0x0001, 77);
    fixture.gpu.Drain();
    draw();
    draw();

    fixture.manager->Prepare();
    fixture.gpu.CpuWrite(BasePage + 4, 0x0002, 81);
    fixture.manager->Prepare();
    fixture.gpu.Drain();
    fixture.manager->NextFrame();
    fixture.gpu.CpuWrite(BasePage + 4, 0x0002, 82);
    fixture.manager->Prepare();
    fixture.gpu.Drain();
    fixture.manager->Prepare();
    Expect(fixture.manager->Hot(BasePage + 4), "page 4 marked hot by its faults");
    rebuilds("a hot mark");
    fixture.gpu.RecordTitle(FakeGpu::Kind::Read, BasePage + 4, 0x0002, 82);
    fixture.gpu.Drain();

    fixture.gpu.collectEpoch = 0;
    draw();
    const auto used = fixture.manager->Counts().rangeSetsUsed;
    draw();
    Expect(fixture.manager->Counts().rangeSetsUsed == used + 1, "without a reuse stamp every draw does the per-page work");
}

void PublicationAndRanges() {
    scope = "publication and ranges";
    Fixture fixture(8, 2);
    fixture.Use(0, BasePage + 0, 3);
    fixture.Use(0, BasePage + 5);
    const auto changes = fixture.manager->TakeChanges();
    Expect(changes == std::vector<std::uint64_t>{BasePage + 0, BasePage + 1, BasePage + 2, BasePage + 5}, "the change list names the admitted pages");
    Expect(fixture.manager->TakeChanges().empty(), "the change list is taken once");
    const auto ranges = fixture.manager->ResidentRanges();
    Expect(ranges.size() == 2 && ranges[0].address == Address(BasePage) && ranges[0].bytes == 3 * ResidencyPageBytes && ranges[1].address == Address(BasePage + 5), "resident ranges coalesce consecutive pages");
    Expect(fixture.manager->AnyResident(Address(BasePage + 2, 15), 1) && !fixture.manager->AnyResident(Address(BasePage + 3), 2 * ResidencyPageBytes), "AnyResident");
    const auto serial = fixture.manager->Serial();
    fixture.manager->Demote(Address(BasePage + 5), 1);
    Expect(fixture.manager->Serial() > serial && fixture.manager->TakeChanges() == std::vector<std::uint64_t>{BasePage + 5}, "a demotion is a change");
    Expect(fixture.manager->Chunks().size() == 1, "one live chunk");
}

void ConcurrentHooks() {
    scope = "concurrent hooks";
    NullDevice device;
    ResidencyManager manager(device, ResidencyPolicy{64, 64 * ResidencyPageBytes, 2, 2});
    std::atomic<bool> stop{false};
    std::thread hooks([&] {
        std::uint64_t round = 0;
        while (!stop) {
            const std::uint64_t page = BasePage + (round % 128);
            manager.AnyResident(Address(page), ResidencyPageBytes);
            manager.QueueCpuWritten(std::span<const std::uint64_t>(&page, 1));
            if (round % 7 == 0) manager.DropRange(Address(page), ResidencyPageBytes);
            ++round;
        }
    });
    for (std::uint32_t round = 0; round < 20000; ++round) {
        if (round % 16 == 0) manager.Prepare();
        manager.Use(Address(BasePage + (round * 37) % 128), ResidencyPageBytes, ResidencyAccess::Read);
    }
    stop = true;
    hooks.join();
    manager.Retire();
    const auto counts = manager.Counts();
    Expect(counts.resident + counts.retiring + counts.free == counts.chunks * 64, "slots stay accounted with concurrent hooks");
}

void HostOverhead() {
    scope = "host overhead";
    NullDevice device;
    constexpr std::uint32_t Pages = 1u << 16;
    const auto before = heapBytes.load();
    {
        ResidencyManager manager(device, ResidencyPolicy{1024, std::uint64_t{Pages} * ResidencyPageBytes, 2, 2});
        for (std::uint32_t page = 0; page < Pages; page += 256) manager.Use(Address(BasePage + page), 256 * ResidencyPageBytes, ResidencyAccess::Read);
        const auto counts = manager.Counts();
        Expect(counts.resident == Pages, "65536 pages resident");
        manager.TakeChanges();
        const auto bytes = heapBytes.load() - before;
        const auto perPage = static_cast<double>(bytes) / Pages;
        std::printf("host bookkeeping: %lld bytes for %u resident pages (4 GiB of mirrors): %.1f bytes per page, %.4f%% of the mirrored bytes\n", static_cast<long long>(bytes), Pages, perPage, 100.0 * perPage / static_cast<double>(ResidencyPageBytes));
        Expect(perPage < 512.0, "bookkeeping stays under 512 bytes per 64 KiB page");
    }
    const auto left = heapBytes.load() - before;
    Expect(left == 0, "the manager frees its bookkeeping: " + std::to_string(left) + " bytes left");
}

struct Stress {
    static constexpr std::uint32_t Pages = 24;
    static constexpr std::uint32_t AliasFirst = 20;
    static constexpr std::uint32_t AliasOf = 16;

    explicit Stress(std::uint32_t seed) : random(seed), fixture(Pages, 2) {
        for (std::uint32_t page = 0; page < AliasFirst; ++page) {
            auto& attributes = fixture.gpu.imports[BasePage + page].attributes;
            const auto kind = random() % 10;
            attributes.cpuWritable = kind != 0;
            attributes.whole = kind != 1;
            attributes.gpuWritable = kind > 3;
            if (kind >= 7) attributes.section = fixture.gpu.imports[BasePage + page].group + 1;
            if (kind == 2) fixture.gpu.guardRefused.insert_or_assign(BasePage + page, ResidencyRefusal::PrivatePinned);
        }
        for (std::uint32_t page = AliasFirst; page < Pages; ++page) {
            auto& owner = fixture.gpu.imports[BasePage + page - AliasFirst + AliasOf].attributes;
            owner.whole = true;
            owner.cpuWritable = true;
            fixture.gpu.Alias(BasePage + page - AliasFirst + AliasOf, BasePage + page);
        }
        for (std::uint32_t page = 0; page < Pages; ++page) truth[group(BasePage + page)] = *fixture.gpu.imports[BasePage + page].storage;
    }

    std::mt19937 random;
    Fixture fixture;
    std::map<std::uint64_t, Contents> truth;
    std::array<std::uint64_t, Pages> ticks{};
    std::uint64_t tick = 0;
    std::uint64_t value = 1u << 20;
    std::set<std::uint64_t> dropping;
    std::uint64_t operations = 0;

    std::uint32_t pick(std::uint32_t bound) { return static_cast<std::uint32_t>(random() % bound); }
    std::uint64_t group(std::uint64_t page) { return fixture.gpu.imports[page].group; }
    bool aliased(std::uint64_t page) { return fixture.gpu.imports[page].attributes.aliased; }
    Contents& truthOf(std::uint64_t page) { return truth[group(page)]; }

    std::map<std::uint64_t, MirrorSlot> snapshot() {
        std::map<std::uint64_t, MirrorSlot> result;
        for (std::uint32_t page = 0; page < Pages; ++page) {
            if (const auto slot = fixture.manager->Slot(BasePage + page)) result.emplace(BasePage + page, *slot);
        }
        return result;
    }

    void titleWait(std::uint64_t page, std::optional<std::size_t> except) {
        const auto token = fixture.gpu.TitleToken(page, except);
        if (!TokenEmpty(token)) fixture.gpu.WaitFor(token);
    }

    template<typename TCall>
    void call(const char* name, bool explicitLeave, std::uint64_t first, std::uint32_t count, bool lru, TCall&& action) {
        const auto before = snapshot();
        const auto exempt = fixture.gpu.reported;
        std::map<std::uint64_t, bool> pinned;
        std::map<std::uint64_t, ResidencyToken> pins;
        for (const auto& [page, slot] : before) {
            pins[page] = fixture.gpu.modelPins[page];
            pinned[page] = !fixture.gpu.Reached(pins[page]);
        }
        const auto waits = fixture.gpu.waits;
        action();
        fixture.gpu.reported.clear();
        const auto after = snapshot();
        std::vector<std::uint64_t> left;
        for (const auto& [page, slot] : before) {
            const auto now = after.find(page);
            if (now != after.end() && now->second == slot) continue;
            if (dropping.contains(page) || exempt.contains(page)) continue;
            left.push_back(page);
            if (!explicitLeave) {
                const bool stillPinned = !fixture.gpu.Reached(pins[page]);
                Expect(!(pinned[page] && fixture.gpu.waits == waits) && !stillPinned, Name(page) + " evicted while pinned by " + name + " (operation " + std::to_string(operations) + ")");
                Expect(page < first || page >= first + count, Name(page) + " evicted by the use that names it");
            }
        }
        for (const auto& [page, slot] : before) {
            const auto now = after.find(page);
            if (now == after.end() || !(now->second == slot)) {
                fixture.gpu.modelPins.erase(page);
                if (slot.chunk < fixture.gpu.live.size()) fixture.gpu.slotOwner[slot.chunk * ChunkSlots + slot.slot] = 0;
            }
        }
        if (lru && !left.empty()) {
            std::uint64_t newestLeft = 0;
            for (const auto page : left) newestLeft = std::max(newestLeft, ticks[page - BasePage]);
            for (const auto& [page, slot] : before) {
                if (!after.contains(page) || pinned[page] || !fixture.gpu.Reached(fixture.gpu.modelPins[page]) || (page >= first && page < first + count)) continue;
                Expect(ticks[page - BasePage] >= newestLeft, Name(page) + " (used at " + std::to_string(ticks[page - BasePage]) + ") kept while a page used at " + std::to_string(newestLeft) + " was evicted");
            }
        }
        std::map<std::uint64_t, std::uint32_t> residentPerGroup;
        for (const auto& [page, slot] : after) {
            const auto attributes = fixture.gpu.imports[page].attributes;
            Expect(attributes.whole, Name(page) + " admitted although no single import covers it");
            const bool tracked = fixture.gpu.armed[group(page)] || fixture.manager->HotResident(page);
            Expect(!fixture.gpu.guardRefused.contains(page) || !attributes.cpuWritable || tracked, Name(page) + " resident while its guard was refused and its writes are not tracked");
            if (attributes.cpuWritable) Expect(tracked, "CPU-writable " + Name(page) + " resident with neither an armed guard nor write tracking after " + name);
            if (attributes.section == 0 && !fixture.manager->ImportSettled(page)) Expect(fixture.manager->Dirty(page) == 0, "private " + Name(page) + " dirty in its mirror before its import settled, after " + name);
            ++residentPerGroup[group(page)];
        }
        for (const auto& [backing, count2] : residentPerGroup) Expect(count2 == 1, "two aliases of one backing resident");
        for (const auto& [backing, isArmed] : fixture.gpu.armed) {
            if (isArmed) Expect(residentPerGroup.contains(backing), "backing " + std::to_string(backing) + " guarded with no resident page after " + name);
        }
        dropping.clear();
        const auto counts = fixture.manager->Counts();
        Expect(counts.resident + counts.retiring + counts.free == counts.chunks * ChunkSlots, "slot accounting: " + std::to_string(counts.resident) + " resident + " + std::to_string(counts.retiring) + " retiring + " + std::to_string(counts.free) + " free in " + std::to_string(counts.chunks) + " chunks");
        Expect(counts.chunks == static_cast<std::uint32_t>(std::count(fixture.gpu.live.begin(), fixture.gpu.live.end(), true)), "live chunk count matches the device");
        std::uint32_t dirty = 0;
        for (const auto& [page, slot] : after) dirty += fixture.manager->Dirty(page) != 0 ? 1 : 0;
        Expect(counts.dirty == dirty, "dirty page count");
        ++operations;
    }

    void waitAliases(std::uint64_t first, std::uint32_t count, std::optional<std::size_t> except) {
        for (std::uint32_t k = 0; k < count; ++k) titleWait(first + k, except);
    }

    void gpuAccess(bool write) {
        const auto stream = pick(FakeStreams);
        const auto first = BasePage + pick(Pages);
        const auto count = std::min<std::uint32_t>(1 + pick(3), static_cast<std::uint32_t>(BasePage + Pages - first));
        waitAliases(first, count, stream);
        fixture.gpu.caller = stream;
        call("Prepare", false, 0, 0, false, [&] { fixture.manager->Prepare(); });
        call("RefreshHot", false, 0, 0, false, [&] { fixture.manager->RefreshHot(); });
        const bool granted = write && pick(2) == 0;
        if (write && !granted) {
            call("Demote", true, first, count, false, [&] { fixture.manager->Demote(Address(first), std::uint64_t{count} * ResidencyPageBytes); });
        } else if (granted) {
            call("Use", false, first, count, true, [&] { fixture.manager->Use(Address(first), std::uint64_t{count} * ResidencyPageBytes, ResidencyAccess::Write); });
            ++tick;
            for (std::uint32_t k = 0; k < count; ++k) {
                if (fixture.manager->Slot(first + k)) ticks[first + k - BasePage] = tick;
            }
        } else {
            call("Use", false, first, count, true, [&] { fixture.manager->Use(Address(first), std::uint64_t{count} * ResidencyPageBytes, ResidencyAccess::Read); });
            ++tick;
            for (std::uint32_t k = 0; k < count; ++k) {
                if (fixture.manager->Slot(first + k)) ticks[first + k - BasePage] = tick;
            }
        }
        fixture.gpu.caller = stream;
        for (std::uint32_t k = 0; k < count; ++k) {
            const auto page = first + k;
            if (write) {
                const auto mask = static_cast<ResidencyMask>(1 + pick(0xffff));
                const auto stored = ++value;
                if (granted) call("NoteGpuWritten", true, page, 1, false, [&] { fixture.Grant(stream, page, mask); });
                fixture.gpu.caller = stream;
                fixture.gpu.RecordTitle(FakeGpu::Kind::Write, page, mask, stored, granted);
                for (std::uint32_t subpage = 0; subpage < ResidencySubpages; ++subpage) {
                    if ((mask >> subpage) & 1u) truthOf(page)[subpage] = stored;
                }
            } else {
                const auto subpage = pick(ResidencySubpages);
                fixture.gpu.RecordTitle(FakeGpu::Kind::Read, page, static_cast<ResidencyMask>(1u << subpage), truthOf(page)[subpage]);
            }
        }
    }

    void driverWrite() {
        const auto stream = pick(FakeStreams);
        const auto page = BasePage + pick(Pages);
        const auto first = pick(ResidencySubpages);
        const auto count = 1 + pick(ResidencySubpages - first);
        titleWait(page, stream);
        fixture.gpu.caller = stream;
        const auto stored = ++value;
        fixture.gpu.RecordTitle(FakeGpu::Kind::Write, page, Run(first, count), stored);
        for (std::uint32_t subpage = first; subpage < first + count; ++subpage) truthOf(page)[subpage] = stored;
        call("NoteImportWritten", true, page, 1, false, [&] { fixture.manager->NoteImportWritten(Address(page, first), std::uint64_t{count} * ResidencySubpageBytes); });
    }

    void cpuAccess(bool write) {
        auto page = BasePage + pick(Pages);
        if (write && !fixture.gpu.imports[page].attributes.cpuWritable) return;
        const auto first = pick(ResidencySubpages);
        const auto count = 1 + pick(ResidencySubpages - first);
        titleWait(page, std::nullopt);
        fixture.gpu.caller = pick(FakeStreams);
        call("FlushForCpuRead", false, page, 0, false, [&] { fixture.manager->FlushForCpuRead(Address(page, first), std::uint64_t{count} * ResidencySubpageBytes); });
        if (!write) {
            for (std::uint32_t subpage = first; subpage < first + count; ++subpage)
                Expect(fixture.Stored(page, subpage) == truthOf(page)[subpage], "CPU read of " + Name(page) + " subpage " + std::to_string(subpage) + " got " + std::to_string(fixture.Stored(page, subpage)) + ", expected " + std::to_string(truthOf(page)[subpage]) + " (operation " + std::to_string(operations) + ")");
            return;
        }
        const auto stored = ++value;
        fixture.gpu.CpuWrite(page, Run(first, count), stored);
        for (std::uint32_t subpage = first; subpage < first + count; ++subpage) truthOf(page)[subpage] = stored;
    }

    void hammer() {
        const auto page = BasePage + pick(Pages);
        if (!fixture.gpu.imports[page].attributes.cpuWritable) return;
        titleWait(page, std::nullopt);
        for (std::uint32_t round = 0; round < 3; ++round) {
            call("FlushForCpuRead", false, page, 0, false, [&] { fixture.manager->FlushForCpuRead(Address(page), ResidencyPageBytes); });
            const auto stored = ++value;
            fixture.gpu.CpuWrite(page, 0x0001, stored);
            truthOf(page)[0] = stored;
            fixture.manager->NextFrame();
            call("Prepare", false, 0, 0, false, [&] { fixture.manager->Prepare(); });
        }
    }

    void drop() {
        const auto page = BasePage + pick(Pages);
        if (aliased(page)) return;
        titleWait(page, std::nullopt);
        fixture.manager->DropRange(Address(page), ResidencyPageBytes);
        const auto fresh = ++value;
        fixture.gpu.Remap(page, fresh);
        truthOf(page).fill(fresh);
        dropping.insert(page);
    }

    void step() {
        switch (pick(24)) {
            case 0: case 1: case 2: case 3: case 4: gpuAccess(false); break;
            case 5: case 6: case 7: gpuAccess(true); break;
            case 8: case 9: cpuAccess(false); break;
            case 10: case 11: cpuAccess(true); break;
            case 12: drop(); break;
            case 13: fixture.gpu.Submit(pick(FakeStreams)); break;
            case 14: case 15: case 16: {
                const auto steps = 1 + pick(6);
                const auto stream = pick(FakeStreams);
                for (std::uint32_t k = 0; k < steps; ++k) fixture.gpu.Step(stream);
                break;
            }
            case 17: call("Retire", false, 0, 0, false, [&] { fixture.manager->Retire(); }); break;
            case 18: {
                if (pick(4) != 0) {
                    call("DemoteWritable", true, 0, 0, false, [&] { fixture.manager->DemoteWritable(); });
                    break;
                }
                const auto chunks = pick(4);
                fixture.gpu.budgetChunks = chunks;
                call("SetBudget", false, 0, 0, false, [&] { fixture.manager->SetBudget(std::uint64_t{chunks} * ChunkSlots * ResidencyPageBytes); });
                break;
            }
            case 19: driverWrite(); break;
            case 20:
                if (pick(2) == 0) fixture.manager->NextFrame();
                call("Prepare", false, 0, 0, false, [&] { fixture.manager->Prepare(); });
                break;
            case 21: hammer(); break;
            case 22: {
                const auto page = BasePage + pick(AliasFirst);
                if (const auto refusal = fixture.gpu.guardRefused.find(page); refusal != fixture.gpu.guardRefused.end()) {
                    const auto cleared = ResidencyRefusalBit(refusal->second);
                    fixture.gpu.guardRefused.erase(refusal);
                    fixture.manager->QueueRefusalCleared(Address(page), ResidencyPageBytes, cleared);
                } else {
                    fixture.gpu.guardRefused.insert_or_assign(page, pick(2) == 0 ? ResidencyRefusal::SharedPinned : ResidencyRefusal::AliasHostWrite);
                }
                break;
            }
            default: {
                fixture.gpu.SubmitAll();
                while (fixture.gpu.StepAny()) {}
                break;
            }
        }
    }

    void finish() {
        fixture.gpu.Drain();
        fixture.gpu.caller = 0;
        call("FlushForCpuRead", true, 0, 0, false, [&] { fixture.manager->FlushForCpuRead(Address(BasePage), std::uint64_t{Pages} * ResidencyPageBytes); });
        fixture.gpu.Drain();
        for (std::uint32_t page = 0; page < Pages; ++page) {
            for (std::uint32_t subpage = 0; subpage < ResidencySubpages; ++subpage)
                Expect(fixture.Stored(BasePage + page, subpage) == truthOf(BasePage + page)[subpage], "after the drain, " + Name(BasePage + page) + " subpage " + std::to_string(subpage) + " holds " + std::to_string(fixture.Stored(BasePage + page, subpage)) + ", expected " + std::to_string(truthOf(BasePage + page)[subpage]));
        }
        call("Retire", false, 0, 0, false, [&] { fixture.manager->Retire(); });
        const auto counts = fixture.manager->Counts();
        Expect(counts.retiring == 0 && counts.dirty == 0, "an idle GPU leaves nothing retiring or dirty");
    }
};

void RandomSequences() {
    scope = "random sequences";
    ResidencyCounts total;
    std::uint64_t mirrorReads = 0;
    std::uint64_t importReads = 0;
    for (std::uint32_t seed = 1; seed <= 60 && failures == 0; ++seed) {
        Stress stress(seed);
        for (std::uint32_t operation = 0; operation < 2500 && failures == 0; ++operation) stress.step();
        stress.finish();
        if (failures != 0) std::fprintf(stderr, "random sequence seed %u failed\n", seed);
        const auto counts = stress.fixture.manager->Counts();
        total.admitted += counts.admitted;
        total.evicted += counts.evicted;
        total.deferred += counts.deferred;
        total.demoted += counts.demoted;
        total.dropped += counts.dropped;
        total.refreshed += counts.refreshed;
        total.faults += counts.faults;
        total.hot += counts.hot;
        total.cooled += counts.cooled;
        total.unguardable += counts.unguardable;
        total.aliasSkipped += counts.aliasSkipped;
        total.importsSettled += counts.importsSettled;
        total.waits += counts.waits;
        total.writtenBackBytes += counts.writtenBackBytes;
        total.chunksReleased += counts.chunksReleased;
        mirrorReads += stress.fixture.gpu.mirrorReads;
        importReads += stress.fixture.gpu.importReads;
    }
    std::printf("random sequences: %llu admitted, %llu evicted, %llu deferred, %llu demoted, %llu dropped, %llu refreshed, %llu faults, %llu hot, %llu cooled, %llu unguardable, %llu alias skips, %llu waits, %llu KiB written back, %llu chunks released, %llu mirror reads, %llu import reads\n",
        static_cast<unsigned long long>(total.admitted), static_cast<unsigned long long>(total.evicted), static_cast<unsigned long long>(total.deferred), static_cast<unsigned long long>(total.demoted),
        static_cast<unsigned long long>(total.dropped), static_cast<unsigned long long>(total.refreshed), static_cast<unsigned long long>(total.faults), static_cast<unsigned long long>(total.hot),
        static_cast<unsigned long long>(total.cooled), static_cast<unsigned long long>(total.unguardable), static_cast<unsigned long long>(total.aliasSkipped), static_cast<unsigned long long>(total.waits),
        static_cast<unsigned long long>(total.writtenBackBytes >> 10), static_cast<unsigned long long>(total.chunksReleased), static_cast<unsigned long long>(mirrorReads), static_cast<unsigned long long>(importReads));
    Expect(total.evicted > 0 && total.deferred > 0 && total.demoted > 0 && total.refreshed > 0 && total.writtenBackBytes > 0 && total.chunksReleased > 0 && mirrorReads > 0 && importReads > 0, "the sequences exercise eviction, deferral, demotion, refresh, write-back, chunk release and both read paths");
    Expect(total.faults > 0 && total.hot > 0 && total.cooled > 0 && total.unguardable > 0 && total.aliasSkipped > 0 && total.importsSettled > 0, "the sequences exercise faults, hot pages, cooling, refused guards, aliases and settled private imports");
}

}

int main() {
    try {
        BudgetAndLeastRecentOrder();
        PinnedPagesNeverEvicted();
        TokenCompletionReleasesPins();
        FlushWaitsForWriters();
        FaultRefreshOrderedAfterOwnWriters();
        ImportWritesRefreshMirrors();
        DeferredCopyInUntilAReader();
        GuardedPagesRefreshAfterFaults();
        OrderingPointsReportQueuedWork();
        HotPagesStayAndCool();
        UnguardablePages();
        RefusedUntilTheirEvent();
        DemotionAndDrops();
        UnboundedWriterCostIndependentOfMirrors();
        DirtyMirrorsOnSharedPages();
        CpuReadWritesDirtyBytesBack();
        ExactDirtyRanges();
        DirtyQueryWithoutLock();
        CpuReadSettledWithoutLock();
        WriteBackAllAtTeardown();
        GrantDemotesPrivatePage();
        NewerImportRestartsSettle();
        SettledPrivatePageKeepsDirtyMirror();
        GrantThroughAliasDemotesMirror();
        TouchedPagesAdmitted();
        TouchedPagesTakeEvictedSlotsWhenTheyRetire();
        MeasuredBudget();
        FullResidencyWhenTheGuestFits();
        EagerAdmissionFillsTheBudget();
        OwnedPagesRetiledBeforeReaders();
        OwnedRangeMarkedStaleInOneUpdate();
        CpuReadRetilesStalePages();
        PrivateOwnedPagesHeldForReaders();
        OwnedPagesExemptAndForgotten();
        ClearStaleSparesOtherSubpages();
        UnguardedStalePagesRetiledForSignals();
        StaleOwnedPagesKeptAndDisowned();
        DemotionRetilesStalePages();
        PageLeftDuringDemotionRetileLeavesOnce();
        AdmissionRule();
        CanonicalAliases();
        EvictedAliasKeepsSharedGuard();
        PublicationAndRanges();
        UnchangedRangesReused();
        OversizedRangeWithoutMirrorsQueriesNoImports();
        ConcurrentHooks();
        HostOverhead();
        RandomSequences();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL [%s]: exception: %s\n", scope.c_str(), error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d residency checks failed\n", failures);
        return 1;
    }
    std::printf("residency tests passed\n");
    return 0;
}
