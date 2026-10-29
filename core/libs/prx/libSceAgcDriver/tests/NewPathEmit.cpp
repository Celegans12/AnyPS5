#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandRecord.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandWrites.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PushBlock.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ResolvedPush.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ResolvedTwins.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ShareCensus.hpp"
#include "Recompiler.hpp"
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
        {"no resolved program", [](CommandFacts& facts) { facts.resolvedProgram = false; facts.tableCovers = false; }, FallbackReason::NoResolvedModule},
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

void composedResolvedPush() {
    ShaderRecompiler::RecompileResult program{};
    Expect(!ResolvedModule(program), "a module counts as newPathResolved before the recompiler marks any");
    std::array<std::byte, Abi::PushBlockBytes> legacy{};
    for (std::uint32_t word = 0; word < 32; ++word) {
        const std::uint32_t value = 0xb0000000u | word;
        std::memcpy(legacy.data() + word * 4u, &value, 4);
    }
    const auto bytes = ComposeResolvedPush(legacy, 128, 0x0000222200330000ull, 0x0000444400550000ull);
    Expect(bytes.has_value(), "a resolved push from today's 128-byte block was refused");
    if (bytes) {
        std::uint32_t words[64]{};
        std::memcpy(words, bytes->data(), sizeof(words));
        Expect(words[0] == 0x00330000u && words[1] == 0x00002222u, "the table root is not at words 0-1");
        Expect(words[2] == 0x00550000u && words[3] == 0x00004444u, "the fault record is not at words 2-3");
        Expect(words[4] == 0u && words[5] == 0u, "reserved words 4-5 are not zero");
        bool shifted = true;
        for (std::uint32_t word = 0; word < 32; ++word) shifted = shifted && words[6 + word] == (0xb0000000u | word);
        Expect(shifted, "today's push words are not at 6 + their slot");
        Expect(words[38] == 0u && words[63] == 0u, "words past the shifted block are not zero");
    }
    Expect(!ComposeResolvedPush(legacy, 128, 0, 1).has_value() && !ComposeResolvedPush(legacy, 128, 1, 0).has_value(), "a zero root or fault record was accepted");
    Expect(!ComposeResolvedPush(legacy, 240, 1, 1).has_value(), "a legacy block that cannot shift by six words was accepted");
}

void commandRecords() {
    std::vector<std::uint32_t> storage(CommandRecordWords, 0xffffffffu);
    std::span<std::uint32_t, CommandRecordWords> words(storage.data(), CommandRecordWords);
    ClearCommandRecord(words);
    Expect(CommandRecordClear(words), "a cleared record is not clear");
    std::vector<AddressRange> marked;
    const auto mark = [&](std::uint64_t address, std::size_t bytes) { marked.emplace_back(address, address + bytes); };
    const std::vector<AddressRange> leased{{0x10000000ull, 0x10100000ull}, {0x20000000ull, 0x20010000ull}};

    auto empty = CompleteCommandRecord(words, leased, mark);
    Expect(!empty.faulted && !empty.overflow && !empty.invalid && marked.empty(), "an empty record marked pages or faulted");

    const std::uint64_t chunk = 0x10020000ull;
    words[BdaAbi::WrittenCountWord] = 1;
    words[BdaAbi::WrittenListWord] = 5;
    words[BdaAbi::WrittenSlotsWord + 10] = BdaAbi::WrittenChunkKey(chunk);
    words[BdaAbi::WrittenSlotsWord + 11] = 0b0110u;
    const auto written = CompleteCommandRecord(words, leased, mark);
    Expect(!written.overflow && !written.invalid && written.writtenRanges == 1 && marked.size() == 1 && marked[0] == AddressRange{chunk + 0x1000, chunk + 0x3000}, "a written-page record did not mark exactly its pages");
    Expect(CommandRecordClear(words), "a completed record was not cleared for reuse");

    marked.clear();
    words[BdaAbi::WrittenOverflowWord] = 1;
    const auto overflow = CompleteCommandRecord(words, leased, mark);
    Expect(overflow.overflow && marked == leased, "an overflowed record did not mark every leased range");

    marked.clear();
    words[BdaAbi::WrittenCountWord] = 1;
    words[BdaAbi::WrittenListWord] = BdaAbi::WrittenPageSlots + 3;
    const auto invalid = CompleteCommandRecord(words, leased, mark);
    Expect(invalid.invalid && marked == leased, "an invalid record was trusted instead of marking every leased range");

    marked.clear();
    BdaAbi::Fault fault{};
    fault.state = BdaAbi::FaultState::Ready;
    fault.reason = BdaAbi::FaultReason::Unmapped;
    fault.address = 0x5000123000ull;
    std::memcpy(storage.data(), &fault, sizeof(fault));
    const auto faulted = CompleteCommandRecord(words, leased, mark);
    Expect(faulted.faulted && faulted.reason == BdaAbi::FaultReason::Unmapped && faulted.message.find("5000123000") != std::string::npos, "a fault record was not reported with its reason and address");
    Expect(CommandRecordClear(words), "a faulted record was not cleared");

    words[BdaAbi::SrtGuardWord] = (2u << BdaAbi::SrtGuardStageShift) | 7u;
    const auto guarded = CompleteCommandRecord(words, leased, mark);
    Expect(guarded.faulted && guarded.message.find("SRT guard") != std::string::npos && guarded.message.find("stage 2") != std::string::npos, "an SRT guard trip in the record was not reported");
    Expect(CommandRecordClear(words), "a guard-tripped record was not cleared");
}

std::shared_ptr<ShaderRecompiler::RecompileResult> AddressProgram(bool resolved) {
    auto program = std::make_shared<ShaderRecompiler::RecompileResult>();
    program->newPathResolved = resolved;
    ShaderRecompiler::DescriptorBinding table{};
    table.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    table.role = ShaderRecompiler::DescriptorRole::BdaPagetable;
    table.binding = 3;
    table.count = 1;
    program->bindings.push_back(table);
    program->pushConstants = {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    return program;
}

void resolvedTwins() {
    const auto compiled = AddressProgram(false);
    auto twin = AddressProgram(true);
    Expect(UsesAddressTables(*compiled), "a program with a BDA table binding is not address-based");
    Expect(!UsesAddressTables(ShaderRecompiler::RecompileResult{}), "a program without bindings is address-based");
    Expect(TwinCompatible(*compiled, *twin), "a resolved twin with the same bindings and push words was refused");
    Expect(!TwinCompatible(*compiled, *compiled), "an unresolved module was taken as its own twin");
    auto otherPush = AddressProgram(true);
    otherPush->pushConstants.push_back(std::byte{5});
    Expect(!TwinCompatible(*compiled, *otherPush), "a twin with other push words was accepted");
    auto otherBinding = AddressProgram(true);
    otherBinding->bindings[0].binding = 4;
    Expect(!TwinCompatible(*compiled, *otherBinding), "a twin with another binding number was accepted");
    auto withBuffer = AddressProgram(false);
    ShaderRecompiler::DescriptorBinding buffer{};
    buffer.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    buffer.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    buffer.binding = 5;
    buffer.count = 1;
    buffer.guestDescriptor = {1, 2, 3, 4};
    withBuffer->bindings.push_back(buffer);
    auto walkOnly = AddressProgram(true);
    walkOnly->bindings = {buffer};
    Expect(TwinCompatible(*withBuffer, *walkOnly), "a twin that drops only the BDA table binding was refused");
    auto lostBuffer = AddressProgram(true);
    Expect(!TwinCompatible(*withBuffer, *lostBuffer), "a twin that drops a guest buffer binding was accepted");
    Expect(WalkReplacesTables(*walkOnly), "a twin without table and fault bindings does not let the build skip its table");
    Expect(!WalkReplacesTables(*twin), "a twin that binds the BDA table lets the build skip it");
    Expect(!WalkReplacesTables(*withBuffer), "an unresolved module lets the build skip its table");
    auto withFault = AddressProgram(true);
    withFault->bindings[0].role = ShaderRecompiler::DescriptorRole::FaultBuffer;
    Expect(!WalkReplacesTables(*withFault), "a twin that binds the fault buffer lets the build skip it");

    ResolvedTwins twins;
    twins.Remember(compiled, twin, TwinOutcome::Built);
    Expect(twins.Of(*compiled) == twin, "a remembered twin was not found");
    Expect(twins.Known(*compiled) && !twins.Known(*twin), "a remembered program is not known, or an unremembered one is");
    Expect(twins.Outcomes()[static_cast<std::size_t>(TwinOutcome::Reused)] == 1, "a reuse of a known twin was not counted");
    const auto plain = AddressProgram(false);
    twins.Remember(plain, nullptr, TwinOutcome::NotAddressBased);
    Expect(twins.Of(*plain) == nullptr && twins.Of(*twin) == nullptr, "a program without a twin got one");
    const auto outcomes = twins.Outcomes();
    Expect(outcomes[static_cast<std::size_t>(TwinOutcome::Built)] == 1 && outcomes[static_cast<std::size_t>(TwinOutcome::NotAddressBased)] == 1, "twin outcomes were not counted");
    {
        auto shortLived = std::make_shared<const ShaderRecompiler::RecompileResult>(*compiled);
        twins.Remember(shortLived, twin, TwinOutcome::Built);
        const auto* address = shortLived.get();
        shortLived.reset();
        auto reused = std::shared_ptr<const ShaderRecompiler::RecompileResult>(std::make_shared<ShaderRecompiler::RecompileResult>());
        if (reused.get() == address) Expect(twins.Of(*reused) == nullptr, "a freed program's twin was served to a new program at its address");
    }
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

ShaderRecompiler::DescriptorBinding CensusBinding(ShaderRecompiler::DescriptorRole role, ShaderRecompiler::DescriptorKind kind, std::uint32_t number, std::vector<std::uint32_t> words) {
    ShaderRecompiler::DescriptorBinding binding{};
    binding.role = role;
    binding.kind = kind;
    binding.binding = number;
    binding.count = 1;
    binding.guestDescriptor = std::move(words);
    return binding;
}

void shareCensus() {
    using Role = ShaderRecompiler::DescriptorRole;
    using Kind = ShaderRecompiler::DescriptorKind;
    namespace Census = AgcDriver::NewPath::Census;
    ShaderRecompiler::RecompileResult program;
    program.bindings.push_back(CensusBinding(Role::GuestBuffers, Kind::StorageBuffer, 0, {0x1000, 0x00100000, 64, 0}));
    program.bindings.push_back(CensusBinding(Role::FlattenedSrt, Kind::StorageBuffer, 1, {7, 8}));
    program.rangesFromMemory = true;
    const auto shape = Census::Classify(program);
    Expect(shape == (Census::BuffersFromMemory | Census::BuffersWritten | Census::FlattenedSrt), "census shape " + Census::ShapeName(shape));
    Expect(Census::ShapeName(shape) == "buf-mem+buf-write+flat-srt", "census shape name " + Census::ShapeName(shape));
    program.rangesFromMemory = false;
    program.bindings[0].readOnly = true;
    program.bindings.push_back(CensusBinding(Role::GuestImages, Kind::StorageImage, 2, {1, 2, 3, 4, 5, 6, 7, 8}));
    program.bindings.push_back(CensusBinding(Role::BdaPagetable, Kind::StorageBuffer, 3, {}));
    Expect(Census::Classify(program) == (Census::AddressBased | Census::BuffersFromUser | Census::StorageImages | Census::FlattenedSrt), "census shape with a table, a user-SGPR buffer and a storage image");
    Expect(Census::Classify(ShaderRecompiler::RecompileResult{}) == 0 && Census::ShapeName(0) == "none", "an empty result has a shape");
    program.bindings.resize(2);
    const auto base = Census::HashBindings(program);
    Expect(Census::Compare(base, base) == Census::Repeat::Same, "equal inputs not same");
    auto pushed = program;
    pushed.pushConstants.push_back(std::byte{1});
    Expect(Census::Compare(base, Census::HashBindings(pushed)) == Census::Repeat::PushOnly, "a push change not push-only");
    auto data = program;
    data.bindings[1].guestDescriptor[1] = 9;
    Expect(Census::Compare(base, Census::HashBindings(data)) == Census::Repeat::DataOnly, "an SRT data change not data-only");
    auto moved = data;
    moved.bindings[0].guestDescriptor[0] = 0x2000;
    moved.bindings[0].guestDescriptor[1] = 0x00100001;
    Expect(Census::Compare(base, Census::HashBindings(moved)) == Census::Repeat::BasesOnly, "a moved buffer base not bases-only");
    auto resized = program;
    resized.bindings[0].guestDescriptor[2] = 128;
    Expect(Census::Compare(base, Census::HashBindings(resized)) == Census::Repeat::Other, "a resized buffer not a descriptor change");
    auto stride = program;
    stride.bindings[0].guestDescriptor[1] = 0x00200000;
    Expect(Census::Compare(base, Census::HashBindings(stride)) == Census::Repeat::Other, "a stride change not a descriptor change");
}

int main() {
    try {
        shareCensus();
        routerDefaultsFallBack();
        routerEmitOrder();
        pushBlock();
        writeSets();
        resolvedDispatchPush();
        composedResolvedPush();
        commandRecords();
        resolvedTwins();
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
