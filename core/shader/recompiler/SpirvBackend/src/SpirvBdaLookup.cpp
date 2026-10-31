#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include <bit>
#include <utility>
#include <iterator>
#include <initializer_list>
#include <vector>
#include <cstdlib>
#include <string>

namespace ShaderRecompiler {

namespace {

// (u64 address, u32 bytes, u32 instruction) -> u64 device address of the range holding all the bytes,
std::uint32_t DefineBdaLookup(SpirvEmitterState& state, const char* name, bool recordFaults, std::uint32_t permission = BdaAbi::Read, bool rangeResult = false) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, name);
    const auto rangeType = rangeResult ? state.module.Type(spv::OpTypeVector, u64, 3u) : 0u;
    const auto resultType = rangeResult ? rangeType : u64;
    const auto zero = rangeResult ? state.module.Constant(spv::OpConstantComposite, rangeType, BdaConstant(state, 0u), BdaConstant(state, 0u), BdaConstant(state, 0u)) : BdaConstant(state, 0u);
    state.module.AddFunction(spv::OpFunction, resultType, function, rangeResult ? spv::FunctionControlMaskNone : spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, resultType, u64, u32, u32));
    const auto address = state.module.AllocateId();
    const auto bytes = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto outer = state.module.AllocateId();
    const auto outerBody = state.module.AllocateId();
    const auto outerContinue = state.module.AllocateId();
    const auto outerMerge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, outer);
    EmitLabel(state, outer);
    state.module.AddFunction(spv::OpLoopMerge, outerMerge, outerContinue, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranch, outerBody);
    EmitLabel(state, outerBody);
    std::vector<std::uint32_t> exits{spv::OpPhi, resultType, state.module.AllocateId()};
    // The new path records its fault after the exit loop (a phi of the reason), so its lookup holds no call inside a
    // loop construct; the range-table path keeps recording at each exit.
    const bool faultAfterLoop = state.newPathResolved && recordFaults;
    std::vector<std::uint32_t> reasons;
    if (faultAfterLoop) reasons = {spv::OpPhi, u32, state.module.AllocateId()};
    const auto leave = [&](std::uint32_t value, std::uint32_t reason = 0u) {
        exits.push_back(value);
        exits.push_back(state.currentLabel);
        if (faultAfterLoop) {
            reasons.push_back(reason != 0u ? reason : constant(0u));
            reasons.push_back(state.currentLabel);
        }
        state.module.AddFunction(spv::OpBranch, outerMerge);
    };
    const auto leaveIf = [&](std::uint32_t condition, std::uint32_t reason) {
        const auto failed = state.module.AllocateId();
        const auto next = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
        EmitLabel(state, failed);
        if (recordFaults && !faultAfterLoop) RecordBdaFaultReason(state, address, bytes, instruction, reason);
        leave(zero, faultAfterLoop ? reason : 0u);
        EmitLabel(state, next);
    };
    const auto fail = [&](std::uint32_t condition, BdaAbi::FaultReason reason) { leaveIf(condition, recordFaults ? constant(static_cast<std::uint32_t>(reason)) : 0u); };
    const auto failFirst = [&](std::initializer_list<std::pair<std::uint32_t, BdaAbi::FaultReason>> checks) {
        auto reason = constant(0u);
        for (auto check = std::rbegin(checks); check != std::rend(checks); ++check) {
            reason = Select(state, u32, check->first, constant(static_cast<std::uint32_t>(check->second)), reason);
        }
        leaveIf(binary(spv::OpINotEqual, boolean, reason, constant(0u)), reason);
    };
    const auto closeFunction = [&] {
        EmitLabel(state, outerContinue);
        state.module.AddFunction(spv::OpBranch, outer);
        EmitLabel(state, outerMerge);
        state.module.AddFunction(exits);
        if (faultAfterLoop) {
            state.module.AddFunction(reasons);
            EmitIfCondition(state, binary(spv::OpINotEqual, boolean, reasons[2], constant(0u)), [&] { RecordBdaFaultReason(state, address, bytes, instruction, reasons[2]); });
        }
        state.module.AddFunction(spv::OpReturnValue, exits[2]);
        state.module.AddFunction(spv::OpFunctionEnd);
        return function;
    };
    if (state.newPathResolved) {
        // New path (S4 epoch 1): the device page table instead of the build's range list. A port of the page walk of
        // fable-void's bda_translate (wip/new-gpu-path b6fab39bf, SpirvBdaLookup.cpp:288-440; authors fable-void,
        // EdouardSence, Nesyth) without its touch notes (note_bda_touch, the device-loss site on NVIDIA) and without
        // its range-list fallback: the new path's table is complete, so a miss is Unmapped. The descent loop holds no
        // call and no subgroup operation; faults are recorded at the exits, after the loop.
        const auto u64v = TypeScalarU64(state);
        const auto entryType = TypeU32Vector(state, 4u);
        const auto widen = [&](std::uint32_t value) { return Unary(state, spv::OpUConvert, u64v, value); };
        const auto pair = [&](std::uint32_t low, std::uint32_t high) { return binary(spv::OpBitwiseOr, u64v, widen(low), binary(spv::OpShiftLeftLogical, u64v, widen(high), BdaConstant(state, 32u))); };
        const auto component = [&](std::uint32_t vector, std::uint32_t index) {
            const auto value = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, u32, value, vector, index);
            return value;
        };
        const auto loadDevice = [&](std::uint32_t type, std::uint32_t at, std::uint32_t alignment) {
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, type), pointer, at);
            const auto value = state.module.AllocateId();
            state.module.AddFunction(spv::OpLoad, type, value, pointer, spv::MemoryAccessAlignedMask, alignment);
            return value;
        };
        const auto end = binary(spv::OpIAdd, u64v, address, widen(bytes));
        failFirst({
            {binary(spv::OpIEqual, boolean, bytes, constant(0)), BdaAbi::FaultReason::Overflow},
            {binary(spv::OpULessThanEqual, boolean, end, address), BdaAbi::FaultReason::Overflow},
            {binary(spv::OpUGreaterThanEqual, boolean, address, BdaConstant(state, BdaAbi::PageTableEnd)), BdaAbi::FaultReason::Unmapped},
        });
        const auto pages = NewPathPageTableAddress(state);
        fail(binary(spv::OpIEqual, boolean, pages, BdaConstant(state, 0u)), BdaAbi::FaultReason::InvalidTable);
        const auto window = binary(spv::OpShiftLeftLogical, u64v, binary(spv::OpShiftRightLogical, u64v, address, BdaConstant(state, BdaAbi::WindowShift)), BdaConstant(state, 3u));
        const auto leaf = loadDevice(u64v, binary(spv::OpIAdd, u64v, pages, window), sizeof(std::uint64_t));
        fail(binary(spv::OpIEqual, boolean, leaf, BdaConstant(state, 0u)), BdaAbi::FaultReason::Unmapped);
        const auto page = binary(spv::OpShiftLeftLogical, u64v, binary(spv::OpBitwiseAnd, u64v, binary(spv::OpShiftRightLogical, u64v, address, BdaConstant(state, BdaAbi::PageShift)), BdaConstant(state, BdaAbi::WindowPages - 1u)), BdaConstant(state, 4u));
        const auto initial = loadDevice(entryType, binary(spv::OpIAdd, u64v, leaf, page), sizeof(BdaAbi::PageEntry));
        const auto header = state.module.AllocateId();
        const auto body = state.module.AllocateId();
        const auto continuation = state.module.AllocateId();
        const auto merge = state.module.AllocateId();
        const auto entryLabel = state.currentLabel;
        state.module.AddFunction(spv::OpBranch, header);
        EmitLabel(state, header);
        const auto current = state.module.AllocateId();
        const auto level = state.module.AllocateId();
        const auto nextEntry = state.module.AllocateId();
        const auto nextLevel = state.module.AllocateId();
        state.module.AddFunction(spv::OpPhi, entryType, current, initial, entryLabel, nextEntry, continuation);
        state.module.AddFunction(spv::OpPhi, u32, level, constant(BdaAbi::PageShift), entryLabel, nextLevel, continuation);
        const auto split = binary(spv::OpIEqual, boolean, binary(spv::OpShiftRightLogical, u32, component(current, 2u), constant(BdaAbi::KindShift)), constant(static_cast<std::uint32_t>(BdaAbi::PageKind::Split)));
        const auto descend = binary(spv::OpLogicalAnd, boolean, split, binary(spv::OpUGreaterThanEqual, boolean, level, constant(BdaAbi::SplitShift)));
        state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, descend, body, merge);
        EmitLabel(state, body);
        const auto lowered = binary(spv::OpISub, u32, level, constant(BdaAbi::SplitShift));
        const auto child = binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64v, address, widen(lowered))), constant(BdaAbi::SplitChildren - 1u));
        const auto children = binary(spv::OpIAdd, u64v, leaf, pair(component(current, 0u), component(current, 1u)));
        const auto loaded = loadDevice(entryType, binary(spv::OpIAdd, u64v, children, widen(binary(spv::OpShiftLeftLogical, u32, child, constant(4u)))), sizeof(BdaAbi::PageEntry));
        state.module.AddFunction(spv::OpCopyObject, entryType, nextEntry, loaded);
        state.module.AddFunction(spv::OpCopyObject, u32, nextLevel, lowered);
        state.module.AddFunction(spv::OpBranch, continuation);
        EmitLabel(state, continuation);
        state.module.AddFunction(spv::OpBranch, header);
        EmitLabel(state, merge);
        const auto bounds = component(current, 2u);
        const auto kind = binary(spv::OpShiftRightLogical, u32, bounds, constant(BdaAbi::KindShift));
        const auto mask = binary(spv::OpISub, u32, binary(spv::OpShiftLeftLogical, u32, constant(1u), level), constant(1u));
        const auto offset = widen(binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, address), mask));
        const auto first = widen(binary(spv::OpBitwiseAnd, u32, bounds, constant(BdaAbi::FirstMask)));
        const auto limit = pair(component(current, 3u), binary(spv::OpBitwiseAnd, u32, binary(spv::OpShiftRightLogical, u32, bounds, constant(BdaAbi::LimitHighShift)), constant(1u)));
        const auto permissions = binary(spv::OpBitwiseAnd, u32, binary(spv::OpShiftRightLogical, u32, bounds, constant(BdaAbi::PermissionShift)), constant(BdaAbi::Read | BdaAbi::Write));
        const auto segment = constant(static_cast<std::uint32_t>(BdaAbi::PageKind::Segment));
        failFirst({
            {binary(spv::OpUGreaterThan, boolean, kind, segment), BdaAbi::FaultReason::InvalidTable},
            {binary(spv::OpINotEqual, boolean, kind, segment), BdaAbi::FaultReason::Unmapped},
            {binary(spv::OpLogicalOr, boolean, binary(spv::OpULessThan, boolean, offset, first), binary(spv::OpUGreaterThanEqual, boolean, offset, limit)), BdaAbi::FaultReason::Unmapped},
            {binary(spv::OpUGreaterThan, boolean, binary(spv::OpIAdd, u64v, offset, widen(bytes)), limit), BdaAbi::FaultReason::Unmapped},
            {binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, permissions, constant(permission)), constant(permission)), BdaAbi::FaultReason::Permission},
        });
        const auto delta = pair(component(current, 0u), component(current, 1u));
        if (rangeResult) {
            const auto pageBase = binary(spv::OpISub, u64v, address, offset);
            const auto begin = binary(spv::OpIAdd, u64v, pageBase, first);
            const auto bound = binary(spv::OpIAdd, u64v, pageBase, limit);
            const auto range = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, rangeType, range, begin, bound, binary(spv::OpIAdd, u64v, begin, delta));
            leave(range);
        } else {
            leave(binary(spv::OpIAdd, u64v, address, delta));
        }
        return closeFunction();
    }
    const auto length = state.module.AllocateId();
    state.module.AddFunction(spv::OpArrayLength, u32, length, state.bdaPagetableVariable, 0u);
    fail(binary(spv::OpULessThan, boolean, length, constant(4)), BdaAbi::FaultReason::InvalidTable);
    const auto count = BdaLoadWord(state, constant(1));
    const auto available = binary(spv::OpISub, u32, length, constant(4));
    const auto end = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
    // A plain table is exactly its ranges; an indexed one (BdaAbi::IndexedTable) has its index
    // header and `buckets + 1` start words after them. Every index word read here and below stays
    // inside the table whatever it holds (the reads of a table that fails a check go to word 0).
    const auto reserved = BdaLoadWord(state, constant(3));
    const auto indexed = binary(spv::OpIEqual, boolean, reserved, constant(BdaAbi::IndexedTable));
    const auto countFits = binary(spv::OpULessThanEqual, boolean, count, binary(spv::OpShiftRightLogical, u32, available, constant(3)));
    const auto rangeWords = binary(spv::OpIMul, u32, Select(state, u32, countFits, count, constant(0)), constant(8));
    const auto indexWords = binary(spv::OpISub, u32, available, rangeWords);
    constexpr auto indexHeaderWords = static_cast<std::uint32_t>(sizeof(BdaAbi::IndexHeader) / sizeof(std::uint32_t));
    const auto headerFits = binary(spv::OpLogicalAnd, boolean, binary(spv::OpLogicalAnd, boolean, indexed, countFits), binary(spv::OpUGreaterThanEqual, boolean, indexWords, constant(indexHeaderWords + 1u)));
    const auto indexAt = Select(state, u32, headerFits, binary(spv::OpIAdd, u32, constant(4), rangeWords), constant(0));
    const auto indexBase = BdaLoadAddress(state, indexAt);
    const auto buckets = BdaLoadWord(state, binary(spv::OpIAdd, u32, indexAt, constant(2)));
    const auto shift = BdaLoadWord(state, binary(spv::OpIAdd, u32, indexAt, constant(3)));
    const auto plainBad = binary(spv::OpLogicalOr, boolean, binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, available, constant(7)), constant(0)), binary(spv::OpINotEqual, boolean, count, binary(spv::OpShiftRightLogical, u32, available, constant(3))));
    const auto indexBad = binary(spv::OpLogicalOr, boolean, Unary(state, spv::OpLogicalNot, boolean, headerFits), binary(spv::OpLogicalOr, boolean, binary(spv::OpINotEqual, boolean, binary(spv::OpISub, u32, indexWords, constant(indexHeaderWords + 1u)), buckets), binary(spv::OpLogicalOr, boolean, binary(spv::OpUGreaterThan, boolean, shift, constant(63)), binary(spv::OpIEqual, boolean, buckets, constant(0)))));
    failFirst({
        {binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(0)), constant(BdaAbi::Version)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(2)), constant(sizeof(BdaAbi::Range))), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpLogicalAnd, boolean, binary(spv::OpINotEqual, boolean, reserved, constant(0)), Unary(state, spv::OpLogicalNot, boolean, indexed)), BdaAbi::FaultReason::InvalidTable},
        {Select(state, boolean, indexed, indexBad, plainBad), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpIEqual, boolean, bytes, constant(0)), BdaAbi::FaultReason::Overflow},
        {binary(spv::OpULessThanEqual, boolean, end, address), BdaAbi::FaultReason::Overflow},
    });
    // The search's starting bounds: [0, count) for a plain table, the address's bucket for an
    // indexed one (clamped to [0, count], so even a wrong index reads no entry past the table).
    const auto below = binary(spv::OpULessThan, boolean, address, indexBase);
    const auto bucket = binary(spv::OpShiftRightLogical, u64, binary(spv::OpISub, u64, address, indexBase), Unary(state, spv::OpUConvert, u64, shift));
    const auto beyond = binary(spv::OpUGreaterThanEqual, boolean, bucket, Unary(state, spv::OpUConvert, u64, buckets));
    const auto inBucket = binary(spv::OpLogicalAnd, boolean, indexed, Unary(state, spv::OpLogicalNot, boolean, binary(spv::OpLogicalOr, boolean, below, beyond)));
    const auto startAt = Select(state, u32, inBucket, binary(spv::OpIAdd, u32, binary(spv::OpIAdd, u32, indexAt, constant(indexHeaderWords)), Unary(state, spv::OpUConvert, u32, bucket)), constant(0));
    const auto outside = Select(state, u32, below, constant(0), count);
    const auto startHigh = Select(state, u32, indexed, Select(state, u32, inBucket, BdaLoadWord(state, binary(spv::OpIAdd, u32, startAt, constant(1))), outside), count);
    const auto searchHigh = Select(state, u32, binary(spv::OpULessThan, boolean, count, startHigh), count, startHigh);
    const auto startLow = Select(state, u32, indexed, Select(state, u32, inBucket, BdaLoadWord(state, startAt), outside), constant(0));
    const auto searchLow = Select(state, u32, binary(spv::OpULessThan, boolean, searchHigh, startLow), searchHigh, startLow);
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    const auto entryLabel = state.currentLabel;
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto lower = state.module.AllocateId();
    const auto upper = state.module.AllocateId();
    const auto newLow = state.module.AllocateId();
    const auto newHigh = state.module.AllocateId();
    state.module.AddFunction(spv::OpPhi, u32, lower, searchLow, entryLabel, newLow, continuation);
    state.module.AddFunction(spv::OpPhi, u32, upper, searchHigh, entryLabel, newHigh, continuation);
    const auto search = binary(spv::OpULessThan, boolean, lower, upper);
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, search, body, merge);
    EmitLabel(state, body);
    const auto midpoint = binary(spv::OpIAdd, u32, lower, binary(spv::OpShiftRightLogical, u32, binary(spv::OpISub, u32, upper, lower), constant(1)));
    const auto entry = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, midpoint, constant(8)));
    const auto before = binary(spv::OpULessThan, boolean, address, BdaLoadAddress(state, entry));
    state.module.AddFunction(spv::OpSelect, u32, newLow, before, lower, binary(spv::OpIAdd, u32, midpoint, constant(1)));
    state.module.AddFunction(spv::OpSelect, u32, newHigh, before, midpoint, upper);
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto index = lower;
    fail(binary(spv::OpIEqual, boolean, index, constant(0)), BdaAbi::FaultReason::Unmapped);
    const auto selected = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, index, constant(1)), constant(8)));
    const auto at = [&](std::uint32_t offset) { return binary(spv::OpIAdd, u32, selected, constant(offset)); };
    const auto begin = BdaLoadAddress(state, selected);
    const auto finish = BdaLoadAddress(state, at(2));
    const auto base = BdaLoadAddress(state, at(4));
    const auto permissions = BdaLoadWord(state, at(6));
    const auto offset = binary(spv::OpISub, u64, address, begin);
    const auto result = binary(spv::OpIAdd, u64, base, offset);
    const auto deviceEnd = binary(spv::OpIAdd, u64, result, Unary(state, spv::OpUConvert, u64, bytes));
    failFirst({
        {binary(spv::OpINotEqual, boolean, BdaLoadWord(state, at(7)), constant(0)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpUGreaterThanEqual, boolean, begin, finish), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpUGreaterThan, boolean, end, finish), BdaAbi::FaultReason::Unmapped},
        {binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, permissions, constant(permission)), constant(permission)), BdaAbi::FaultReason::Permission},
        {binary(spv::OpIEqual, boolean, base, BdaConstant(state, 0)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpULessThan, boolean, result, base), BdaAbi::FaultReason::Overflow},
        {binary(spv::OpULessThanEqual, boolean, deviceEnd, result), BdaAbi::FaultReason::Overflow},
    });
    // Diagnostic (request bench overlap prototype, never in the game): APS5_DIAG_WATCH=lo-hi,... (hex guest ranges, read
    // when the module is emitted): a read whose bytes reach a listed range records FaultReason::Watched and goes on.
    if (const char* watch = std::getenv("APS5_DIAG_WATCH"); watch != nullptr && *watch != '\0' && (permission & BdaAbi::Read) != 0u && state.bdaFaultFunction != 0 && state.program.Info().usesDma) {
        std::uint32_t hit = ConstantBool(state, false);
        for (const char* cursor = watch; *cursor != '\0';) {
            char* next = nullptr;
            const auto lo = std::strtoull(cursor, &next, 16);
            if (next == cursor || *next != '-') break;
            const auto hi = std::strtoull(next + 1, &next, 16);
            hit = binary(spv::OpLogicalOr, boolean, hit, binary(spv::OpLogicalAnd, boolean, binary(spv::OpULessThan, boolean, address, BdaConstant(state, hi)), binary(spv::OpUGreaterThan, boolean, end, BdaConstant(state, lo))));
            cursor = *next == ',' ? next + 1 : next;
        }
        EmitIfCondition(state, hit, [&] { RecordBdaFaultReason(state, address, bytes, instruction, constant(static_cast<std::uint32_t>(BdaAbi::FaultReason::Watched))); });
    }
    if (rangeResult) {
        const auto last = binary(spv::OpUGreaterThanEqual, boolean, index, count);
        const auto next = BdaLoadAddress(state, binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, Select(state, u32, last, binary(spv::OpISub, u32, index, constant(1)), index), constant(8))));
        const auto bound = Select(state, u64, binary(spv::OpLogicalOr, boolean, last, binary(spv::OpULessThan, boolean, finish, next)), finish, next);
        const auto range = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, rangeType, range, begin, bound, base);
        leave(range);
    } else {
        leave(result);
    }
    return closeFunction();
}

}

static std::uint32_t DefineBdaNoteWrite(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "note_bda_write");
    state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlMaskNone, state.module.Type(spv::OpTypeFunction, voidType, TypeScalarU64(state)));
    const auto address = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, TypeScalarU64(state), address);
    EmitLabel(state, state.module.AllocateId());
    const auto done = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, boolean), done, spv::StorageClassFunction);
    const auto counter = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, u32), counter, spv::StorageClassFunction);
    state.module.AddFunction(spv::OpStore, done, ConstantBool(state, false));
    state.module.AddFunction(spv::OpStore, counter, constant(0u));
    const auto u64 = TypeScalarU64(state);
    const auto key = binary(spv::OpIAdd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WrittenChunkShift))), constant(1));
    const auto page = binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WrittenPageShift))), constant(BdaAbi::WrittenChunkPages - 1u));
    const auto bit = binary(spv::OpShiftLeftLogical, u32, constant(1u), page);
    const auto hash = binary(spv::OpShiftRightLogical, u32, binary(spv::OpIMul, u32, key, constant(0x9e3779b1u)), constant(32u - static_cast<std::uint32_t>(std::countr_zero(BdaAbi::WrittenPageSlots))));
    const auto scope = constant(spv::ScopeDevice);
    const auto relaxed = constant(spv::MemorySemanticsMaskNone);
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto probe = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, u32, probe, counter);
    const auto found = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, found, done);
    const auto searching = binary(spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, found), binary(spv::OpULessThan, boolean, probe, constant(BdaAbi::WrittenPageProbes)));
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, searching, body, merge);
    EmitLabel(state, body);
    const auto slot = binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, hash, probe), constant(BdaAbi::WrittenPageSlots - 1u));
    const auto keyWord = binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenSlotsWord), binary(spv::OpShiftLeftLogical, u32, slot, constant(1u)));
    const auto current = state.module.AllocateId();
    state.module.AddFunction(spv::OpAtomicLoad, u32, current, BdaWord(state, state.faultBufferVariable, keyWord), scope, relaxed);
    const auto same = binary(spv::OpIEqual, boolean, current, key);
    state.module.AddFunction(spv::OpStore, done, same);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, same), [&] {
        const auto previous = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicCompareExchange, u32, previous, BdaWord(state, state.faultBufferVariable, keyWord), scope, relaxed, relaxed, key, constant(0u));
        const auto claimed = binary(spv::OpIEqual, boolean, previous, constant(0u));
        state.module.AddFunction(spv::OpStore, done, binary(spv::OpLogicalOr, boolean, claimed, binary(spv::OpIEqual, boolean, previous, key)));
        EmitIfCondition(state, claimed, [&] {
            const auto index = state.module.AllocateId();
            state.module.AddFunction(spv::OpAtomicIAdd, u32, index, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenCountWord)), scope, relaxed, constant(1u));
            state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenListWord), index)), scope, relaxed, slot);
        });
    });
    const auto holds = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, holds, done);
    EmitIfCondition(state, holds, [&] {
        const auto maskWord = binary(spv::OpIAdd, u32, keyWord, constant(1u));
        const auto mask = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicLoad, u32, mask, BdaWord(state, state.faultBufferVariable, maskWord), scope, relaxed);
        EmitIfCondition(state, binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, mask, bit), constant(0u)), [&] {
            state.module.AddFunction(spv::OpAtomicOr, u32, state.module.AllocateId(), BdaWord(state, state.faultBufferVariable, maskWord), scope, relaxed, bit);
        });
    });
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpStore, counter, binary(spv::OpIAdd, u32, probe, constant(1u)));
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto noted = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, noted, done);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, noted), [&] {
        state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenOverflowWord)), scope, relaxed, constant(1u));
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

void DefineBdaAnchorFunction(SpirvEmitterState& state) {
    if (state.bdaAnchorFunction == 0) state.bdaAnchorFunction = DefineBdaLookup(state, "bda_anchor_range", false, BdaAbi::Read, true);
}

void DefineGetBdaPointer(SpirvEmitterState& state) {
    if (state.faultBufferVariable != 0) DefineBdaFaultFunction(state);
    if (!state.program.Info().usesDma) return;
    state.bdaPointerFunction = DefineBdaLookup(state, "get_bda_pointer", true);
    if (!BdaByteReadsForced()) state.bdaProbeFunction = DefineBdaLookup(state, "probe_bda_pointer", false);
    DefineBdaDwordReadFunctions(state);
    DefineBdaSpanReadFunctions(state);
    CollectBdaAnchors(state);
    if (!state.bdaAnchors.empty()) DefineBdaAnchorFunction(state);
    if (state.program.Info().bdaWrites) {
        state.bdaWritePointerFunction = DefineBdaLookup(state, "get_bda_write_pointer", true, BdaAbi::Write);
        if (!BdaByteReadsForced()) state.bdaWriteProbeFunction = DefineBdaLookup(state, "probe_bda_write_pointer", false, BdaAbi::Write);
        state.bdaAtomicPointerFunction = DefineBdaLookup(state, "get_bda_atomic_pointer", true, BdaAbi::Read | BdaAbi::Write);
        state.bdaNoteWriteFunction = DefineBdaNoteWrite(state);
        DefineBdaByteWriteFunctions(state);
    }
}

}
