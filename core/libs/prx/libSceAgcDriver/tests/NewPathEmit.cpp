#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandWrites.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PushBlock.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/Router.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

using namespace AgcDriver::NewPath;
namespace BdaAbi = ShaderRecompiler::BdaAbi;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

CommandFacts ReadyDispatch() {
    CommandFacts facts;
    facts.kind = CommandKind::Dispatch;
    facts.resolvedProgram = true;
    return facts;
}

void routerDefaultsFallBack() {
    const Router router;
    Expect(!router.EmitsDispatches(), "the router emits by default");
    const auto decision = router.Decide(ReadyDispatch());
    Expect(decision.route == Route::Fallback && decision.reason == FallbackReason::KindNotEmitted, "a ready dispatch left today's path with emission off");
}

void routerEmitOrder() {
    const Router router(true);
    Expect(router.Decide(ReadyDispatch()).route == Route::NewPath, "a ready plain dispatch was not emitted");
    struct Case {
        const char* name;
        void (*set)(CommandFacts&);
        FallbackReason reason;
    };
    const Case cases[] = {
        {"suspended", [](CommandFacts& facts) { facts.suspended = true; facts.kindSwitchOff = true; }, FallbackReason::Suspended},
        {"kind switch", [](CommandFacts& facts) { facts.kindSwitchOff = true; facts.pipelineReady = false; }, FallbackReason::KindSwitch},
        {"draw", [](CommandFacts& facts) { facts.kind = CommandKind::Draw; }, FallbackReason::KindNotEmitted},
        {"indirect dispatch", [](CommandFacts& facts) { facts.kind = CommandKind::DispatchIndirect; }, FallbackReason::KindNotEmitted},
        {"no resolved program", [](CommandFacts& facts) { facts.resolvedProgram = false; facts.tableCovers = false; }, FallbackReason::KindNotEmitted},
        {"no pipeline", [](CommandFacts& facts) { facts.pipelineReady = false; facts.tableCovers = false; }, FallbackReason::NoPipeline},
        {"range", [](CommandFacts& facts) { facts.tableCovers = false; facts.pushFits = false; }, FallbackReason::RangeNotCovered},
        {"push overflow", [](CommandFacts& facts) { facts.pushFits = false; facts.clothGuard = true; }, FallbackReason::PushOverflow},
        {"cloth guard", [](CommandFacts& facts) { facts.clothGuard = true; facts.staticChains = false; }, FallbackReason::ClothGuard},
        {"non-static chain", [](CommandFacts& facts) { facts.staticChains = false; }, FallbackReason::NonStaticChain},
    };
    for (const auto& item : cases) {
        auto facts = ReadyDispatch();
        item.set(facts);
        const auto decision = router.Decide(facts);
        Expect(decision.route == Route::Fallback && decision.reason == item.reason, std::string(item.name) + ": got " + FallbackReasonName(decision.reason));
    }
    auto overflow = ReadyDispatch();
    overflow.pushFits = false;
    overflow.overflowPlaced = true;
    Expect(router.Decide(overflow).route == Route::NewPath, "a placed overflow upload did not allow emission");
}

void pushBlock() {
    Abi::PushLayout layout;
    layout.driverWords = 6;
    layout.pageTableAddressWord = 0;
    layout.faultRecordAddressWord = 2;
    layout.overflowAddressWord = 4;
    layout.overflow = true;
    layout.userWordOffset = {6, 0, 0, 10};
    layout.userWordCount = {3, 0, 0, 2};
    const std::uint32_t vertex[] = {0x11, 0x22, 0x33};
    const std::uint32_t pixel[] = {0x44, 0x55};
    const std::array<std::span<const std::uint32_t>, Abi::StageCount> words{std::span<const std::uint32_t>(vertex), {}, {}, std::span<const std::uint32_t>(pixel)};
    const auto bytes = BuildPushBlock(layout, words, {0x123456789abcull, 0xfeedface0000ull, 0x5000001000ull});
    Expect(bytes.has_value(), "a valid push layout was refused");
    if (bytes) {
        std::uint64_t table = 0, fault = 0, overflowAddress = 0;
        std::uint32_t user[5]{};
        std::memcpy(&table, bytes->data(), 8);
        std::memcpy(&fault, bytes->data() + 8, 8);
        std::memcpy(&overflowAddress, bytes->data() + 16, 8);
        std::memcpy(user, bytes->data() + 24, 12);
        std::memcpy(user + 3, bytes->data() + 40, 8);
        Expect(table == 0x123456789abcull && fault == 0xfeedface0000ull && overflowAddress == 0x5000001000ull, "a driver address is not at its words");
        Expect(user[0] == 0x11 && user[2] == 0x33 && user[3] == 0x44 && user[4] == 0x55, "user words are not at their stage offsets");
    }
    auto overlap = layout;
    overlap.userWordOffset[0] = 2;
    Expect(!BuildPushBlock(overlap, words, {}).has_value(), "user words over the fault-record address were accepted");
    auto wrongCount = layout;
    wrongCount.userWordCount[0] = 4;
    Expect(!BuildPushBlock(wrongCount, words, {}).has_value(), "a stage with fewer words than its layout was accepted");
    auto past = layout;
    past.userWordOffset[3] = 63;
    Expect(!BuildPushBlock(past, words, {}).has_value(), "user words past the 256-byte block were accepted");
}

void resolvedDispatchPush() {
    Expect(PushRangeBytes(true) == 256u && PushRangeBytes(false) == 128u, "the push ranges are not 256 bytes for resolved modules and today's 128 otherwise");
    Abi::PushLayout layout;
    layout.driverWords = 4;
    layout.pageTableAddressWord = ResolvedPageTableWord;
    layout.faultRecordAddressWord = ResolvedFaultRecordWord;
    layout.userWordOffset = {4, 0, 0, 0};
    layout.userWordCount = {16, 0, 0, 0};
    std::vector<std::uint32_t> user(16);
    for (std::uint32_t index = 0; index < 16; ++index) user[index] = 0xa0000000u | index;
    const auto bytes = ResolvedDispatchPush(layout, user, 0x0000123400560000ull, 0x0000777700880000ull);
    Expect(bytes.has_value(), "a resolved dispatch's push block was refused");
    if (bytes) {
        std::uint32_t words[20]{};
        std::memcpy(words, bytes->data(), sizeof(words));
        Expect(words[0] == 0x00560000u && words[1] == 0x00001234u, "the page-table root is not the u64 at words 0-1");
        Expect(words[2] == 0x00880000u && words[3] == 0x00007777u, "the fault-record address is not the u64 at words 2-3");
        Expect(words[4] == 0xa0000000u && words[19] == 0xa000000fu, "user SGPRs do not follow the driver words");
        Expect(std::all_of(bytes->begin() + 80, bytes->end(), [](std::byte value) { return value == std::byte{0}; }), "bytes past the used words are not zero");
    }
    Expect(!ResolvedDispatchPush(layout, user, 0, 0x1000).has_value(), "a resolved dispatch with no page-table root was accepted");
    Expect(!ResolvedDispatchPush(layout, user, 0x1000, 0).has_value(), "a resolved dispatch with no fault record was accepted");
    auto moved = layout;
    moved.pageTableAddressWord = 2;
    moved.faultRecordAddressWord = 0;
    Expect(!ResolvedDispatchPush(moved, user, 0x1000, 0x2000).has_value(), "a resolved layout with the addresses swapped was accepted");
    auto graphics = layout;
    graphics.userWordOffset[3] = 20;
    graphics.userWordCount[3] = 2;
    Expect(!ResolvedDispatchPush(graphics, user, 0x1000, 0x2000).has_value(), "a dispatch layout with pixel words was accepted");
}

BdaAbi::Range MakeRange(std::uint64_t begin, std::uint64_t end, std::uint32_t permissions) {
    return {begin, end, 0x7f0000000000ull + begin, permissions, 0u};
}

void writeSets() {
    const std::vector<BdaAbi::Range> table{
        MakeRange(0x1000, 0x2000, BdaAbi::Read | BdaAbi::Write),
        MakeRange(0x2000, 0x3000, BdaAbi::Read | BdaAbi::Write),
        MakeRange(0x3000, 0x4000, BdaAbi::Read),
        MakeRange(0x8000, 0x9000, BdaAbi::Read | BdaAbi::Write),
    };
    const auto writable = WritableRanges(table);
    Expect(writable.size() == 2 && writable[0] == AddressRange{0x1000, 0x3000} && writable[1] == AddressRange{0x8000, 0x9000}, "writable ranges are not the merged writable table ranges");

    const auto none = ResolveWrites(table, Abi::StoreClass::None, {});
    Expect(!none.unbounded && none.ranges.empty() && !none.outsideTable, "a program without stores has a write set");

    const auto unbounded = ResolveWrites(table, Abi::StoreClass::Unbounded, {});
    Expect(unbounded.unbounded && unbounded.ranges == writable && unbounded.Bytes() == 0x3000, "an unbounded writer is not leased over exactly the writable table ranges");
    Expect(std::none_of(unbounded.ranges.begin(), unbounded.ranges.end(), [](const AddressRange& range) { return range.first == 0 && range.second == ~std::uint64_t{0}; }), "an unbounded writer claimed the whole address space");

    const std::vector<AddressRange> stores{{0x1800, 0x2400}, {0x8000, 0x8100}, {0x2300, 0x2500}};
    const auto bounded = ResolveWrites(table, Abi::StoreClass::Bounded, stores);
    Expect(!bounded.unbounded && !bounded.outsideTable && bounded.ranges.size() == 2 && bounded.ranges[0] == AddressRange{0x1800, 0x2500}, "bounded store ranges were not merged inside the table");

    const std::vector<AddressRange> readOnly{{0x3800, 0x3900}};
    Expect(ResolveWrites(table, Abi::StoreClass::Bounded, readOnly).outsideTable, "a store into a read-only page was not flagged");
    const std::vector<AddressRange> gap{{0x7000, 0x8100}};
    Expect(ResolveWrites(table, Abi::StoreClass::Bounded, gap).outsideTable, "a store into an unmapped gap was not flagged");
}

}

int main() {
    try {
        routerDefaultsFallBack();
        routerEmitOrder();
        pushBlock();
        writeSets();
        resolvedDispatchPush();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d new-path emit checks failed\n", failures);
        return 1;
    }
    std::puts("new-path emit tests passed");
    return 0;
}
