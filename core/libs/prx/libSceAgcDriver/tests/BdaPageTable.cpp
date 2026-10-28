#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "BdaPageReference.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace AgcDriver::Graphics;
namespace Abi = ShaderRecompiler::BdaAbi;
namespace Ref = BdaPageReference;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

std::string Hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

Ref::Leaves Encode(std::span<const Abi::Range> ranges) {
    ValidateBdaPageRanges(ranges);
    return Ref::EncodeLeaves(ranges, [](std::uint32_t window, std::span<const Abi::Range> held) { return EncodeBdaWindow(window, held); });
}

Ref::Translation Walked(const Ref::Leaves& leaves, std::uint64_t address, std::uint32_t bytes, std::uint32_t permission = Abi::Read, std::span<const Abi::Grant> grants = {}) {
    const auto walked = Ref::WalkPages(leaves, address, bytes, permission, grants);
    return walked.value_or(Ref::Translation{~std::uint64_t{0}, ~0u});
}

constexpr Ref::Translation Missed{~std::uint64_t{0}, ~0u};

Ref::Translation Mapped(std::uint64_t device) {
    return {device, 0u};
}

void ExpectSame(const Ref::Leaves& leaves, std::span<const Abi::Range> list, std::span<const Abi::Range> held, const std::string& name, std::span<const Abi::Grant> grants = {}) {
    const Ref::Leaves none;
    const auto own = Ref::Without(list, held);
    for (const auto address : Ref::Probes(list)) {
        for (const auto bytes : Ref::ProbeBytes) {
            for (const auto permission : {Abi::Read, Abi::Write, Abi::Read | Abi::Write}) {
                const auto expected = Ref::Translate(none, list, address, bytes, permission, grants);
                const auto actual = Ref::Translate(leaves, own, address, bytes, permission, grants);
                Expect(actual == expected, name + ": " + Hex(address) + " bytes " + std::to_string(bytes) + " permission " + std::to_string(permission) + " gave " + Hex(actual.device) + " reason " + std::to_string(actual.reason) + ", the list " + Hex(expected.device) + " reason " + std::to_string(expected.reason));
                const auto alone = Ref::Translate(leaves, {}, address, bytes, permission, grants);
                Expect(alone == Ref::Translate(none, held, address, bytes, permission, grants), name + ": the page table alone translated " + Hex(address) + " bytes " + std::to_string(bytes) + " permission " + std::to_string(permission) + " unlike its ranges");
            }
        }
    }
}

void ExpectSame(const Ref::Leaves& leaves, std::span<const Abi::Range> list, const std::string& name) {
    ExpectSame(leaves, list, list, name);
}

void WholePageTests() {
    constexpr std::uint64_t base = 0x7f0000000000ull;
    constexpr std::uint64_t device = 0x2000000000ull;
    const std::vector<Abi::Range> ranges{{base, base + 0x40000u, device, Abi::Read | Abi::Write, 0u}};
    const auto leaves = Encode(ranges);
    const auto& entries = leaves.at(static_cast<std::uint32_t>(base >> Abi::WindowShift));
    Expect(entries.size() == Abi::WindowPages, "a range of whole pages needed split blocks");
    for (std::uint32_t page = 0; page < 4u; ++page) {
        const auto& entry = entries.at(((base >> Abi::PageShift) & (Abi::WindowPages - 1u)) + page);
        Expect(Abi::KindOf(entry) == Abi::PageKind::Segment && Abi::FirstOf(entry) == 0u && Abi::LimitOf(entry) == 0x40000u - page * 0x10000u, "a whole page entry does not hold its range's end");
    }
    Expect(Walked(leaves, base, 1u) == Mapped(device), "the first byte of a whole page range was not translated");
    Expect(Walked(leaves, base + 0x1fffeu, 4u, Abi::Write) == Mapped(device + 0x1fffeu), "a span across two pages of one range was not translated");
    Expect(Walked(leaves, base + 0x3ffffu, 1u) == Mapped(device + 0x3ffffu), "the last byte of a whole page range was not translated");
    Expect(Walked(leaves, base + 0x3fffcu, 8u) == Ref::Fault(Abi::FaultReason::Unmapped), "a span past a range's end was not unmapped");
    Expect(Walked(leaves, base + 0x40000u, 1u) == Missed, "the byte after a range was found in the page table");
    Expect(Walked(leaves, base - 1u, 1u) == Missed, "the byte before a range was found in the page table");
    ExpectSame(leaves, ranges, "whole pages");
}

void PartialPageTests() {
    constexpr std::uint64_t base = 0x123450000ull;
    constexpr std::uint64_t device = 0x3000000000ull;
    const std::vector<Abi::Range> ranges{{base + 0x1234u, base + 0x5678u, device, Abi::Read, 0u}};
    const auto leaves = Encode(ranges);
    const auto& entry = leaves.at(static_cast<std::uint32_t>(base >> Abi::WindowShift)).at((base >> Abi::PageShift) & (Abi::WindowPages - 1u));
    Expect(Abi::KindOf(entry) == Abi::PageKind::Segment && Abi::FirstOf(entry) == 0x1234u && Abi::LimitOf(entry) == 0x5678u, "a range inside one page is not held exactly by its entry");
    Expect(Walked(leaves, base + 0x1233u, 1u) == Missed, "the byte before a partial page range was found");
    Expect(Walked(leaves, base + 0x1234u, 1u) == Mapped(device), "the first byte of a partial page range was not translated");
    Expect(Walked(leaves, base + 0x5677u, 1u) == Mapped(device + 0x5678u - 0x1234u - 1u), "the last byte of a partial page range was not translated");
    Expect(Walked(leaves, base + 0x5677u, 2u) == Ref::Fault(Abi::FaultReason::Unmapped), "a span past a partial page range was not unmapped");
    Expect(Walked(leaves, base + 0x5678u, 1u) == Missed, "the byte after a partial page range was found");
    ExpectSame(leaves, ranges, "partial page");
}

void SharedPageTests() {
    constexpr std::uint64_t base = 0x200000000ull;
    const std::vector<Abi::Range> aligned{{base + 0x100u, base + 0x3000u, 0x4000000000ull, Abi::Read, 0u}, {base + 0x3000u, base + 0x9000u, 0x5000000000ull, Abi::Read | Abi::Write, 0u}};
    const auto split = Encode(aligned);
    const auto& entries = split.at(static_cast<std::uint32_t>(base >> Abi::WindowShift));
    Expect(Abi::KindOf(entries.at((base >> Abi::PageShift) & (Abi::WindowPages - 1u))) == Abi::PageKind::Split && entries.size() == Abi::WindowPages + Abi::SplitChildren, "two ranges meeting at a 4 KiB boundary of one page did not take one split level");
    Expect(Walked(split, base + 0x2fffu, 1u) == Mapped(0x4000000000ull + 0x2effu), "the last byte before a split point was not translated");
    Expect(Walked(split, base + 0x3000u, 1u) == Mapped(0x5000000000ull), "the first byte after a split point was not translated");
    Expect(Walked(split, base + 0x2ffeu, 4u) == Ref::Fault(Abi::FaultReason::Unmapped), "a span across two ranges was not unmapped");
    Expect(Walked(split, base + 0xffu, 1u) == Missed, "the gap before the first range of a shared page was found");
    ExpectSame(split, aligned, "4 KiB split");
    const std::vector<Abi::Range> bytes{{base, base + 0x1001u, 0x6000000000ull, Abi::Read, 0u}, {base + 0x1001u, base + 0x2000u, 0x7000000000ull, Abi::Read, 0u}};
    const auto deep = Encode(bytes);
    Expect(deep.at(static_cast<std::uint32_t>(base >> Abi::WindowShift)).size() == Abi::WindowPages + 4u * Abi::SplitChildren, "two ranges meeting at an odd byte did not split down to bytes");
    Expect(Walked(deep, base + 0x1000u, 1u) == Mapped(0x6000000000ull + 0x1000u), "the last byte before an odd split point was not translated");
    Expect(Walked(deep, base + 0x1000u, 2u) == Ref::Fault(Abi::FaultReason::Unmapped), "a span across an odd split point was not unmapped");
    Expect(Walked(deep, base + 0x1001u, 1u) == Mapped(0x7000000000ull), "the first byte after an odd split point was not translated");
    ExpectSame(deep, bytes, "byte split");
    const std::vector<Abi::Range> many{{base + 0x10u, base + 0x11u, 0x10000u, Abi::Read, 0u}, {base + 0x12u, base + 0x20u, 0x20000u, Abi::Read, 0u}, {base + 0x20u, base + 0x30u, 0x30000u, Abi::Read, 0u}, {base + 0x31u, base + 0x10000u, 0x40000u, Abi::Read, 0u}};
    ExpectSame(Encode(many), many, "many ranges in one page");
}

void PermissionTests() {
    constexpr std::uint64_t base = 0x7ff600000000ull;
    const std::vector<Abi::Range> ranges{{base, base + 0x20000u, 0x8000000000ull, Abi::Read, 0u}, {base + 0x20000u, base + 0x30000u, 0x9000000000ull, 0u, 0u}, {base + 0x30000u, base + 0x40000u, 0xa000000000ull, Abi::Read | Abi::Write, 0u}};
    const auto leaves = Encode(ranges);
    Expect(Walked(leaves, base + 8u, 4u, Abi::Write) == Ref::Fault(Abi::FaultReason::Permission), "a write to a read-only page was not a permission fault");
    Expect(Walked(leaves, base + 0x20000u, 1u) == Ref::Fault(Abi::FaultReason::Permission), "a read of a page without permissions was not a permission fault");
    Expect(Walked(leaves, base + 0x30000u, 8u, Abi::Read | Abi::Write) == Mapped(0xa000000000ull), "a writable page did not answer an atomic");
    ExpectSame(leaves, ranges, "permissions");
    const std::vector<Abi::Grant> grants{{base + 0x10u, base + 0x1010u}, {base + 0x1fff0u, base + 0x30010u}};
    Expect(Walked(leaves, base + 0x10u, 4u, Abi::Write, grants) == Mapped(0x8000000010ull), "a write grant did not let a read-only page take a store");
    Expect(Walked(leaves, base + 0x100cu, 4u, Abi::Read | Abi::Write, grants) == Mapped(0x800000100cull), "a write grant did not let a read-only page take an atomic");
    Expect(Walked(leaves, base + 0x100eu, 4u, Abi::Write, grants) == Ref::Fault(Abi::FaultReason::Permission), "a store past the end of its write grant was permitted");
    Expect(Walked(leaves, base + 0xcu, 8u, Abi::Write, grants) == Ref::Fault(Abi::FaultReason::Permission), "a store from before its write grant was permitted");
    Expect(Walked(leaves, base + 0x20000u, 4u, Abi::Write, grants) == Mapped(0x9000000000ull), "a write grant did not add the write permission to a page without permissions");
    Expect(Walked(leaves, base + 0x20000u, 4u, Abi::Read, grants) == Ref::Fault(Abi::FaultReason::Permission), "a write grant made a page without permissions readable");
    Expect(Walked(leaves, base + 0x20000u, 4u, Abi::Read | Abi::Write, grants) == Ref::Fault(Abi::FaultReason::Permission), "a write grant made a page without permissions take an atomic");
    Expect(Walked(leaves, base + 0x1fffcu, 4u, Abi::Write, grants) == Mapped(0x800001fffcull), "a write grant over a range's end did not cover the range's last bytes");
    ExpectSame(leaves, ranges, ranges, "permissions with write grants", grants);
}

void WindowTests() {
    constexpr std::uint64_t boundary = 0x200000000ull;
    constexpr std::uint64_t device = 0xb000000000ull;
    const std::vector<Abi::Range> ranges{{boundary - 0x10000u, boundary + 0x10000u, device, Abi::Read, 0u}};
    const auto leaves = Encode(ranges);
    Expect(leaves.size() == 2u, "a range across a 4 GiB boundary did not reach both windows");
    Expect(Walked(leaves, boundary - 2u, 4u) == Mapped(device + 0xfffeu), "a span across a window boundary was not translated");
    Expect(Walked(leaves, boundary + 0xfffeu, 4u) == Ref::Fault(Abi::FaultReason::Unmapped), "a span past a range in the second window was not unmapped");
    ExpectSame(leaves, ranges, "windows");
    constexpr std::uint64_t large = 0x1000000000ull;
    const std::vector<Abi::Range> huge{{large, large + 0x240000000ull, 0xc000000000ull, Abi::Read, 0u}};
    const auto saturated = Encode(huge);
    Expect(Abi::LimitOf(saturated.at(static_cast<std::uint32_t>(large >> Abi::WindowShift)).at(0)) == Abi::LimitMax, "the limit of a 9 GiB range did not saturate");
    Expect(Walked(saturated, large + 0xffffu, 0xfffffff0u) == Mapped(0xc000000000ull + 0xffffu), "a 4 GiB span inside a 9 GiB range was not translated");
    Expect(Walked(saturated, large + 0x240000000ull - 0xfffffff0u + 1u, 0xfffffff0u) == Ref::Fault(Abi::FaultReason::Unmapped), "a 4 GiB span past a 9 GiB range was not unmapped");
    ExpectSame(saturated, huge, "saturated limit");
}

void UpdateTests() {
    constexpr std::uint64_t base = 0x300000000ull;
    const Abi::Range first{base + 0x100u, base + 0x8000u, 0x100000u, Abi::Read, 0u};
    const Abi::Range second{base + 0x8000u, base + 0x18000u, 0x200000u, Abi::Read, 0u};
    const Abi::Range third{base + 0x20000u, base + 0x20010u, 0x300000u, Abi::Read, 0u};
    const std::vector<Abi::Range> all{first, second, third};
    const std::vector<Abi::Range> removed{first, third};
    auto moved = all;
    moved[1].deviceAddress = 0x900000u;
    const auto before = Encode(all);
    const auto after = Encode(removed);
    const auto remapped = Encode(moved);
    Expect(Walked(before, base + 0x9000u, 4u) == Mapped(0x201000u), "an inserted range was not translated");
    Expect(Walked(after, base + 0x9000u, 4u) == Missed, "a removed range was still translated");
    Expect(Walked(after, base + 0x7ffcu, 8u) == Ref::Fault(Abi::FaultReason::Unmapped), "a span into a removed range was not unmapped");
    Expect(Walked(remapped, base + 0x9000u, 4u) == Mapped(0x901000u), "a remapped range kept its old device address");
    ExpectSame(after, removed, "removed range");
    ExpectSame(remapped, moved, "remapped range");
}

void ValidationTests() {
    const auto rejects = [](std::vector<Abi::Range> ranges, const char* what) {
        bool thrown = false;
        try {
            ValidateBdaPageRanges(ranges);
        } catch (const std::runtime_error&) {
            thrown = true;
        }
        Expect(thrown, what);
    };
    constexpr std::uint64_t base = 0x400000000ull;
    rejects({{base, base + 0x20u, 0x1000u, Abi::Read, 0u}, {base + 0x10u, base + 0x30u, 0x2000u, Abi::Read, 0u}}, "overlapping page ranges were accepted");
    rejects({{base + 0x40u, base + 0x50u, 0x1000u, Abi::Read, 0u}, {base, base + 0x10u, 0x2000u, Abi::Read, 0u}}, "unsorted page ranges were accepted");
    rejects({{base, base, 0x1000u, Abi::Read, 0u}}, "an empty page range was accepted");
    rejects({{Abi::PageTableEnd - 0x10u, Abi::PageTableEnd + 0x10u, 0x1000u, Abi::Read, 0u}}, "a page range past the page table was accepted");
    rejects({{base, base + 0x10u, 0u, Abi::Read, 0u}}, "a page range without a device address was accepted");
    rejects({{base, base + 0x10u, ~std::uint64_t{0} - 4u, Abi::Read, 0u}}, "a page range whose device range wraps was accepted");
    rejects({{base, base + 0x10u, 0x1000u, Abi::Read, 2u}}, "a page range with reserved bits was accepted");
    {
        const std::vector<Abi::Range> mirrored{{base, base + 0x10000u, 0x1000000u, Abi::Read, Abi::RangeMirrored}};
        const auto entries = EncodeBdaWindow(static_cast<std::uint32_t>(base >> Abi::WindowShift), mirrored);
        const bool marked = std::any_of(entries.begin(), entries.end(), [](const Abi::PageEntry& entry) { return Abi::KindOf(entry) == Abi::PageKind::Segment && ((entry.bounds >> Abi::MirroredShift) & 1u) != 0u; });
        Expect(marked, "a mirrored page range did not set the mirrored bit of its segment entry");
        const auto walked = Walked(Encode(mirrored), base + 0x20u, 4u);
        Expect(walked == Mapped(0x1000020u), "a mirrored page range translated " + Hex(walked.device) + ", expected 0x1000020");
    }
    rejects({{base, base + 0x10u, 0x1000u, 4u, 0u}}, "a page range with unknown permissions was accepted");
    const auto listRejects = [](std::vector<Abi::Range> ranges, const char* what) {
        bool thrown = false;
        try {
            ValidateBdaRangeList(ranges);
        } catch (const std::runtime_error&) {
            thrown = true;
        }
        Expect(thrown, what);
    };
    listRejects({{base, base + 0x20u, 0x1000u, Abi::Read, 0u}, {base + 0x1fu, base + 0x30u, 0x2000u, Abi::Read, 0u}}, "an overlapping BDA range list was accepted");
    listRejects({{base + 0x40u, base + 0x50u, 0x1000u, Abi::Read, 0u}, {base, base + 0x10u, 0x2000u, Abi::Read, 0u}}, "an unsorted BDA range list was accepted");
    bool accepted = true;
    try {
        ValidateBdaRangeList(std::vector<Abi::Range>{{base, base + 0x20u, 0x1000u, Abi::Read, 0u}, {base + 0x20u, base + 0x30u, 0u, 0u, 0u}, {base + 0x30u, base + 0x30u, 0x3000u, Abi::Read, 0u}});
        ValidateBdaRangeList({});
    } catch (const std::runtime_error&) {
        accepted = false;
    }
    Expect(accepted, "a sorted, disjoint BDA range list was refused");
}

void RandomTests() {
    std::mt19937_64 random(0x5eed5eedu);
    for (std::uint32_t round = 0; round < 120u; ++round) {
        const auto base = (static_cast<std::uint64_t>(random() % 0x7fffu) << Abi::WindowShift) + (round % 3u == 0u ? 0xffff0000ull : (random() % 0x10000u) << Abi::PageShift);
        const auto table = Ref::RandomTable(random, base, 1u + random() % 40u);
        if (table.list.back().end > Abi::PageTableEnd) continue;
        const auto grants = round % 2u == 0u ? std::vector<Abi::Grant>{} : Ref::RandomGrants(random, table.list, 1u + random() % 6u);
        ExpectSame(Encode(table.pages), table.list, table.pages, "random table " + std::to_string(round), grants);
    }
}

void GrantValidationTests() {
    const auto rejects = [](std::vector<Abi::Grant> grants, const char* what) {
        bool thrown = false;
        try {
            ValidateBdaGrants(grants);
        } catch (const std::runtime_error&) {
            thrown = true;
        }
        Expect(thrown, what);
    };
    constexpr std::uint64_t base = 0x500000000ull;
    rejects({{base, base}}, "an empty write grant was accepted");
    rejects({{base, base + 0x20u}, {base + 0x10u, base + 0x30u}}, "overlapping write grants were accepted");
    rejects({{base, base + 0x20u}, {base + 0x20u, base + 0x30u}}, "touching write grants were accepted");
    rejects({{base + 0x40u, base + 0x50u}, {base, base + 0x10u}}, "unsorted write grants were accepted");
    bool accepted = true;
    try {
        ValidateBdaGrants(std::vector<Abi::Grant>{{base, base + 0x20u}, {base + 0x21u, base + 0x30u}});
        ValidateBdaGrants({});
    } catch (const std::runtime_error&) {
        accepted = false;
    }
    Expect(accepted, "sorted, disjoint write grants were refused");
}

}

int main() {
    try {
        WholePageTests();
        PartialPageTests();
        SharedPageTests();
        PermissionTests();
        WindowTests();
        UpdateTests();
        ValidationTests();
        GrantValidationTests();
        RandomTests();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d BDA page table checks failed\n", failures);
        return 1;
    }
    std::puts("BDA page table tests passed");
    return 0;
}
