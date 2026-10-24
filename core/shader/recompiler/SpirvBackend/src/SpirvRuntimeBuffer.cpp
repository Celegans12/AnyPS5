#include "SpirvBackend/SpirvRuntimeBuffer.hpp"
#include "SpirvBackend/SpirvEmitterInstructions.hpp"
#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include "SpirvBackend/SpirvMemory/SpirvSubgroup.hpp"
#include "SpirvBackend/SpirvMemory/SpirvInputOutput.hpp"
#include "PipelineSpecialization.hpp"
#include "RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include <algorithm>
#include <cstdlib>
#include <array>
#include <cstring>
#include <type_traits>
#include <span>
#include <utility>
#include <string>
#include <vector>

namespace ShaderRecompiler {
namespace {

struct RuntimeBounds {
    std::uint32_t soffset = 0;
    std::uint32_t indices = 0;
    std::uint32_t swizzleBase = 0;
    std::uint32_t swizzleLow = 0;
    std::uint32_t linearBase = 0;
    std::uint32_t indexInBounds = 0;
    std::uint32_t rawFits = 0;
    std::uint32_t rawIndexInBounds = 0;
    std::uint32_t rawRecords = 0;
    std::uint32_t structuredMode = 0;
    std::uint32_t lowMode = 0;
    std::uint32_t countMode = 0;
    std::uint32_t present = 0;
    std::uint32_t enabled = 0;
};

struct RuntimeDescriptor {
    std::uint32_t base;
    std::uint32_t records;
    std::uint32_t stride;
    std::uint32_t word3;
    std::uint32_t swizzle;
    std::uint32_t index;
    std::uint32_t valid;
    std::uint32_t offset = 0;
    std::uint32_t soffset = 0;
    std::uint32_t invalid = 0;
    RuntimeBounds bounds{};
};

struct DecodedDescriptor {
    std::uint32_t base;
    std::uint32_t records;
    std::uint32_t stride;
    std::uint32_t word3;
    std::uint32_t swizzle;
    std::uint32_t valid;
    std::uint32_t invalid;
    std::uint32_t addTid;
};

struct DescriptorShape {
    std::uint32_t indexShift;
    std::uint32_t indices;
    std::uint32_t structuredMode;
    std::uint32_t lowMode;
    std::uint32_t countMode;
    std::uint32_t present;
    std::uint32_t enabled;
};

struct RuntimeAddress {
    std::uint32_t guest;
    std::uint32_t inBounds;
};

std::uint32_t field(SpirvEmitterState& state, std::uint32_t word, std::uint32_t offset, std::uint32_t count) {
    return EmitBitFieldUExtract(state, word, ConstantU32(state, offset), ConstantU32(state, count));
}

std::uint32_t both(SpirvEmitterState& state, std::uint32_t left, std::uint32_t right) {
    return Binary(state, spv::OpLogicalAnd, TypeBool(state), left, right);
}

std::uint32_t equal(SpirvEmitterState& state, std::uint32_t left, std::uint32_t right) {
    return Binary(state, spv::OpIEqual, TypeBool(state), left, right);
}

std::uint32_t nonzero(SpirvEmitterState& state, std::uint32_t value) {
    return Binary(state, spv::OpINotEqual, TypeBool(state), value, ConstantU32(state, 0u));
}

std::uint32_t gcnWord3(SpirvEmitterState& state, std::uint32_t word1, std::uint32_t word3) {
    const auto u32 = TypeU32(state);
    const auto& formats = GcnBufferFormats();
    std::array<std::uint32_t, 128> words{};
    for (std::size_t index = 0; index < words.size(); ++index) words[index] = ConstantU32(state, formats[index]);
    const auto arrayType = state.module.Type(spv::OpTypeArray, u32, ConstantU32(state, static_cast<std::uint32_t>(words.size())));
    if (state.gcnBufferFormatVariable == 0) state.gcnBufferFormatVariable = state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, arrayType), spv::StorageClassPrivate);
    state.module.AddFunction(spv::OpStore, state.gcnBufferFormatVariable, state.module.Constant(spv::OpConstantComposite, arrayType, std::span<const std::uint32_t>(words)));
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, TypePointer(state, spv::StorageClassPrivate, u32), pointer, state.gcnBufferFormatVariable, field(state, word3, 12u, 7u));
    const auto format = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, u32, format, pointer);
    const auto outOfBounds = Select(state, u32, nonzero(state, field(state, word1, 16u, 14u)), ConstantU32(state, 1u << 28u), ConstantU32(state, 3u << 28u));
    const auto kept = Binary(state, spv::OpBitwiseAnd, u32, word3, ConstantU32(state, 0xc0e00fffu));
    return Binary(state, spv::OpBitwiseOr, u32, Binary(state, spv::OpBitwiseOr, u32, kept, Binary(state, spv::OpShiftLeftLogical, u32, format, ConstantU32(state, 12u))), outOfBounds);
}

void faultIf(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t condition, std::uint32_t address, std::uint32_t bytes, BdaAbi::FaultReason reason) {
    auto& state = context.state;
    EmitIfCondition(state, condition, [&] { RecordBdaFault(state, address, ConstantU32(state, bytes), BdaInstructionPc(state, instruction), reason); });
    StopBdaInvocationIf(state, condition);
}

template<typename TValue, typename TCompute>
TValue cached(SpirvEmitterState& state, std::vector<std::uint32_t> key, TCompute&& compute) {
    static_assert(std::is_trivially_copyable_v<TValue> && sizeof(TValue) % sizeof(std::uint32_t) == 0u);
    const bool hoisted = state.runtimeBufferLabel != 0u && state.currentLabel == state.runtimeBufferLabel;
    if (hoisted) {
        if (const auto found = state.runtimeBufferValues.find(key); found != state.runtimeBufferValues.end()) {
            TValue value;
            std::memcpy(&value, found->second.data(), sizeof(TValue));
            return value;
        }
    }
    const TValue value = compute();
    if (hoisted) {
        std::vector<std::uint32_t> words(sizeof(TValue) / sizeof(std::uint32_t));
        std::memcpy(words.data(), &value, sizeof(TValue));
        state.runtimeBufferValues.emplace(std::move(key), std::move(words));
    }
    return value;
}

std::uint32_t runtimePresent(SpirvValueEmitContext& context, const IrValue& handle) {
    auto& state = context.state;
    auto word1 = context.Arg(handle, 1u);
    if (GcnGuestShaders()) word1 = Binary(state, spv::OpBitwiseAnd, TypeU32(state), word1, ConstantU32(state, GcnBufferWord1Mask));
    const auto based = Binary(state, spv::OpLogicalOr, TypeBool(state), nonzero(state, context.Arg(handle, 0u)), nonzero(state, field(state, word1, 0u, 16u)));
    return both(state, nonzero(state, context.Arg(handle, 2u)), based);
}

const IrValue& descriptorHandle(SpirvValueEmitContext& context, const IrValue& instruction) {
    const auto* handle = instruction.Argument(0)->Resolve();
    if (handle->Opcode() != IrOpcode::GetBufferResource || handle->ArgumentCount() != 4u) context.Fail(instruction, "requires a runtime four-dword V#");
    return *handle;
}

std::vector<std::uint32_t> descriptorKey(SpirvValueEmitContext& context, const IrValue& instruction) {
    const auto& handle = descriptorHandle(context, instruction);
    const auto& memory = context.Memory(instruction);
    const bool based = memory.gpuDescriptor || BufferAccessOf(instruction.Opcode()) == BufferAccess::Atomic;
    return {0u, memory.resource, memory.gpuDescriptor ? 1u : 0u, based ? 1u : 0u, static_cast<std::uint32_t>(memory.kind), context.Arg(handle, 0u), context.Arg(handle, 1u), context.Arg(handle, 2u), context.Arg(handle, 3u)};
}

DecodedDescriptor decode(SpirvValueEmitContext& context, const IrValue& instruction) {
    auto& state = context.state;
    const auto& handle = descriptorHandle(context, instruction);
    const auto& memory = context.Memory(instruction);
    const bool runtime = memory.gpuDescriptor || state.registrationModules;
    const auto first = PipelineSpecialization::BufferBase + memory.resource * PipelineSpecialization::BufferWords;
    const bool runtimeShape = runtime || state.program.Resources().stage == IrShaderStage::Compute;
    auto word1 = runtimeShape ? context.Arg(handle, 1u) : state.module.SpecializationConstant(TypeU32(state), first, 0u);
    if (runtimeShape && GcnGuestShaders()) word1 = Binary(state, spv::OpBitwiseAnd, TypeU32(state), word1, ConstantU32(state, GcnBufferWord1Mask));
    const bool runtimeFormat = memory.gpuDescriptor || (state.registrationModules && !state.program.Info().buffers.at(memory.resource).descriptorFormatted);
    const bool selectsFormat = !runtimeFormat && !state.registrationModules && memory.kind == ResourceKind::Buffer && state.program.Resources().stage == IrShaderStage::Compute && state.program.Info().buffers.at(memory.resource).descriptorFormatted;
    const auto u32 = TypeU32(state);
    auto runtimeWord3 = (runtimeFormat || selectsFormat) ? context.Arg(handle, 3u) : 0u;
    if ((runtimeFormat || selectsFormat) && GcnGuestShaders()) runtimeWord3 = gcnWord3(state, word1, runtimeWord3);
    const auto specializedWord3 = runtimeFormat ? 0u : state.module.SpecializationConstant(u32, first + 1u, 0u);
    const auto word3 = runtimeFormat ? runtimeWord3 : selectsFormat ? Select(state, u32, nonzero(state, state.module.SpecializationConstant(u32, first + PipelineSpecialization::BufferRuntimeFormatWord, 0u)), runtimeWord3, specializedWord3) : specializedWord3;
    auto base = ConstantU32(state, 0u);
    if (memory.gpuDescriptor || BufferAccessOf(instruction.Opcode()) == BufferAccess::Atomic) {
        const auto u64 = TypeScalarU64(state);
        const auto high = Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, field(state, GcnGuestShaders() ? Binary(state, spv::OpBitwiseAnd, u32, context.Arg(handle, 1u), ConstantU32(state, GcnBufferWord1Mask)) : context.Arg(handle, 1u), 0u, 16u)), BdaConstant(state, 32u));
        base = Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, context.Arg(handle, 0u)), high);
    }
    auto invalid = nonzero(state, field(state, word3, 30u, 2u));
    auto addTid = ConstantBool(state, false);
    if (memory.kind == ResourceKind::Buffer) {
        addTid = nonzero(state, field(state, word3, 23u, 1u));
        if (state.program.Resources().stage != IrShaderStage::Compute) invalid = Binary(state, spv::OpLogicalOr, TypeBool(state), invalid, addTid);
    }
    const auto stride = field(state, word1, 16u, 14u);
    const auto valid = memory.gpuDescriptor ? both(state, Unary(state, spv::OpLogicalNot, TypeBool(state), invalid), Binary(state, spv::OpINotEqual, TypeBool(state), base, BdaConstant(state, 0u))) : runtimePresent(context, handle);
    return {base, context.Arg(handle, 2u), stride, word3, both(state, nonzero(state, stride), nonzero(state, field(state, word1, 31u, 1u))), valid, invalid, addTid};
}

RuntimeDescriptor descriptor(SpirvValueEmitContext& context, const IrValue& instruction) {
    auto& state = context.state;
    auto key = descriptorKey(context, instruction);
    const auto decoded = cached<DecodedDescriptor>(state, key, [&] { return decode(context, instruction); });
    auto index = ConstantU32(state, 0u);
    if (context.Memory(instruction).kind == ResourceKind::Buffer) {
        index = context.Arg(instruction, 1u);
        if (state.program.Resources().stage == IrShaderStage::Compute) {
            key.front() = 1u;
            key.insert(key.end(), {index, state.laneHalf});
            index = cached<std::uint32_t>(state, key, [&] {
                const auto u32 = TypeU32(state);
                const auto lane = state.laneCount == 1u && state.hostSubgroupSize < state.program.WaveSize() ?
                    Binary(state, spv::OpBitwiseAnd, u32, EmitLocalInvocationIndex(state), ConstantU32(state, state.program.WaveSize() - 1u)) : EmitSubgroupLocalInvocationId(state);
                return Binary(state, spv::OpIAdd, u32, index, Select(state, u32, decoded.addTid, lane, ConstantU32(state, 0u)));
            });
        }
    }
    return {decoded.base, decoded.records, decoded.stride, decoded.word3, decoded.swizzle, index, decoded.valid, 0u, 0u, decoded.invalid};
}

void checkDescriptor(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource) {
    if (context.Memory(instruction).gpuDescriptor) faultIf(context, instruction, resource.invalid, resource.base, 0u, BdaAbi::FaultReason::InvalidDescriptor);
}

DescriptorShape shape(SpirvEmitterState& state, const RuntimeDescriptor& resource) {
    const auto mode = field(state, resource.word3, 28u, 2u);
    const auto indexShift = Binary(state, spv::OpIAdd, TypeU32(state), field(state, resource.word3, 21u, 2u), ConstantU32(state, 3u));
    return {
        indexShift,
        Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1u), indexShift),
        equal(state, mode, ConstantU32(state, 0u)),
        Binary(state, spv::OpULessThan, TypeBool(state), mode, ConstantU32(state, 2u)),
        equal(state, mode, ConstantU32(state, 2u)),
        nonzero(state, resource.records),
        both(state, resource.valid, nonzero(state, field(state, resource.word3, 12u, 7u))),
    };
}

RuntimeBounds bounds(SpirvEmitterState& state, const RuntimeDescriptor& resource, const DescriptorShape& form, std::uint32_t soffset) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto indexMsb = Binary(state, spv::OpShiftRightLogical, u32, resource.index, form.indexShift);
    const auto indexLsb = Binary(state, spv::OpBitwiseAnd, u32, resource.index, Binary(state, spv::OpISub, u32, form.indices, ConstantU32(state, 1u)));
    const auto rawRecords = Binary(state, spv::OpISub, u32, resource.records, soffset);
    return {
        soffset,
        form.indices,
        Binary(state, spv::OpIMul, u32, indexMsb, resource.stride),
        Binary(state, spv::OpShiftLeftLogical, u32, indexLsb, ConstantU32(state, 2u)),
        Binary(state, spv::OpIMul, u32, resource.index, resource.stride),
        Binary(state, spv::OpULessThan, boolean, resource.index, resource.records),
        Binary(state, spv::OpULessThanEqual, boolean, soffset, resource.records),
        Binary(state, spv::OpULessThan, boolean, resource.index, rawRecords),
        rawRecords,
        form.structuredMode,
        form.lowMode,
        form.countMode,
        form.present,
        form.enabled,
    };
}

RuntimeDescriptor bounded(SpirvValueEmitContext& context, const IrValue& instruction) {
    auto& state = context.state;
    auto resource = descriptor(context, instruction);
    auto key = descriptorKey(context, instruction);
    key.front() = 2u;
    const auto form = cached<DescriptorShape>(state, key, [&] { return shape(state, resource); });
    const auto soffset = context.Arg(instruction, 3u);
    key.front() = 3u;
    key.insert(key.end(), {resource.index, soffset});
    resource.bounds = cached<RuntimeBounds>(state, key, [&] { return bounds(state, resource, form, soffset); });
    return resource;
}

std::uint32_t inBounds(SpirvEmitterState& state, const RuntimeDescriptor& resource, std::uint32_t offset, std::uint32_t bytes) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto& range = resource.bounds;
    const auto less = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpULessThan, boolean, left, right); };
    const auto fits = [&](std::uint32_t start, std::uint32_t size) {
        return both(state, Binary(state, spv::OpUGreaterThanEqual, boolean, size, bytes), Binary(state, spv::OpULessThanEqual, boolean, start, Binary(state, spv::OpISub, u32, size, bytes)));
    };
    const auto structured = both(state, range.indexInBounds, less(offset, resource.stride));
    const auto raw = both(state, range.rawFits, Select(state, boolean, resource.swizzle, both(state, range.rawIndexInBounds, fits(offset, resource.stride)), fits(offset, range.rawRecords)));
    auto result = Select(state, boolean, range.structuredMode, structured, range.indexInBounds);
    result = Select(state, boolean, range.lowMode, result, Select(state, boolean, range.countMode, range.present, raw));
    return both(state, range.enabled, result);
}

std::uint32_t byteOffset(SpirvEmitterState& state, const RuntimeDescriptor& resource, std::uint32_t offset) {
    const auto u32 = TypeU32(state);
    const auto& range = resource.bounds;
    const auto add = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpIAdd, u32, left, right); };
    const auto offsetMsb = Binary(state, spv::OpBitwiseAnd, u32, offset, ConstantU32(state, ~3u));
    const auto offsetLsb = Binary(state, spv::OpBitwiseAnd, u32, offset, ConstantU32(state, 3u));
    const auto swizzled = add(Binary(state, spv::OpIMul, u32, add(range.swizzleBase, offsetMsb), range.indices), add(range.swizzleLow, offsetLsb));
    return add(Select(state, u32, resource.swizzle, swizzled, add(range.linearBase, offset)), range.soffset);
}

RuntimeDescriptor withBounds(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource) {
    auto prepared = resource;
    if (prepared.bounds.enabled == 0u) prepared.bounds = bounds(context.state, resource, shape(context.state, resource), resource.soffset != 0u ? resource.soffset : context.Arg(instruction, 3u));
    return prepared;
}

RuntimeAddress byteAddress(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t offset, std::uint32_t bytes) {
    auto& state = context.state;
    auto prepared = resource;
    if (prepared.bounds.enabled == 0u) prepared.bounds = bounds(state, resource, shape(state, resource), resource.soffset != 0u ? resource.soffset : context.Arg(instruction, 3u));
    const auto byte = byteOffset(state, prepared, offset);
    return {byte, inBounds(state, prepared, offset, bytes)};
}

RuntimeAddress address(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t componentOffset, std::uint32_t bytes) {
    auto& state = context.state;
    const auto offset = Binary(state, spv::OpIAdd, TypeU32(state), resource.offset != 0u ? resource.offset : context.Arg(instruction, 2u), ConstantU32(state, resource.offset != 0u ? componentOffset : context.Memory(instruction).offset + componentOffset));
    const auto bytesAddress = byteAddress(context, instruction, resource, offset, ConstantU32(state, bytes));
    if (!context.Memory(instruction).gpuDescriptor) return bytesAddress;
    return {AddBdaAddress(context, instruction, resource.base, Unary(state, spv::OpUConvert, TypeScalarU64(state), bytesAddress.guest), false), bytesAddress.inBounds};
}

std::uint32_t guestAddress(SpirvEmitterState& state, const RuntimeDescriptor& resource, std::uint32_t byte) {
    return Binary(state, spv::OpIAdd, TypeScalarU64(state), resource.base, Unary(state, spv::OpUConvert, TypeScalarU64(state), byte));
}

MemoryResourceAccess directAccess(SpirvValueEmitContext& context, const IrValue& instruction, bool wide = false, bool plainLoad = false) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    // A single-workgroup dispatch reads glc buffers through the plain, L1-cached binding (workgroup-scope
    // coherence, see ShaderComputeStageInfo::singleWorkgroup); its stores and atomics keep the coherent one.
    const bool plain = plainLoad || (state.singleWorkgroup && BufferAccessOf(instruction.Opcode()) == BufferAccess::Read);
    const auto variable = wide ? state.storageBufferU64Variable : plain || state.storageBufferCoherentVariable == 0u ? state.storageBufferVariable : state.storageBufferCoherentVariable;
    if (variable == 0u) context.Fail(instruction, "direct buffer descriptor array is missing");
    const auto element = ResourceForDescriptor(state, DescriptorBindingKind::Buffers, memory.resource);
    MemoryResourceAccess access;
    access.kind = memory.kind;
    access.objectPointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, wide ? TypeStorageBufferU64Pointer(state) : TypeStorageBufferPointer(state), access.objectPointer, variable, ConstantU32(state, element));
    access.byteOffset = state.memoryByteOffsets.at(element);
    access.memoryAccess = memory.coherent ? spv::MemoryAccessVolatileMask : 0u;
    {
        const auto* descriptor = instruction.Argument(0u)->Resolve();
        std::array<const IrValue*, 4> key{};
        for (std::size_t index = 0; index < key.size(); ++index) key[index] = descriptor->Argument(index)->Resolve();
        const auto found = std::find_if(state.bufferLengths.begin(), state.bufferLengths.end(), [&](const auto& entry) {
            return entry.descriptor == key && entry.variable == variable && entry.element == element && entry.label == state.currentLabel;
        });
        if (found != state.bufferLengths.end()) access.length = found->length;
        else {
            access.length = state.module.AllocateId();
            state.module.AddFunction(spv::OpArrayLength, TypeU32(state), access.length, access.objectPointer, 0u);
            state.bufferLengths.push_back({key, variable, element, state.currentLabel, access.length});
        }
    }
    return access;
}

std::uint32_t baseMisalignment(SpirvValueEmitContext& context, const IrValue& instruction) {
    const auto& memory = context.Memory(instruction);
    if (memory.kind == ResourceKind::ScalarBuffer) return ConstantU32(context.state, 0u);
    auto& state = context.state;
    return state.module.SpecializationConstant(TypeU32(state), PipelineSpecialization::BufferBase + memory.resource * PipelineSpecialization::BufferWords + 3u, 0u);
}

std::uint32_t directWordIndex(SpirvEmitterState& state, const MemoryResourceAccess& access, std::uint32_t address) {
    const auto relative = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address, ConstantU32(state, 2u));
    const auto base = Binary(state, spv::OpShiftRightLogical, TypeU32(state), access.byteOffset, ConstantU32(state, 2u));
    return EmitAddU32(state, relative, base);
}

std::uint32_t readBuffer(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t address, std::uint32_t bits, std::uint32_t enable = 0u) {
    if (context.Memory(instruction).gpuDescriptor) return EmitBdaRead(context, instruction, address, bits);
    auto& state = context.state;
    const auto access = directAccess(context, instruction, false, !context.Memory(instruction).coherent);
    const auto index = directWordIndex(state, access, address);
    const auto misalignment = baseMisalignment(context, instruction);
    const auto unaligned = nonzero(state, misalignment);
    const auto next = EmitAddU32(state, index, ConstantU32(state, 1u));
    const auto last = Select(state, TypeU32(state), unaligned, next, index);
    const auto bounds = EmitMemoryElementInBounds(state, access, last);
    const auto inBounds = enable != 0u ? both(state, enable, bounds) : bounds;
    const auto load = [&](std::uint32_t word) {
        const auto value = state.module.AllocateId();
        const auto pointer = EmitMemoryElementPointer(state, access, word);
        if (access.memoryAccess != 0u) state.module.AddFunction(spv::OpLoad, TypeU32(state), value, pointer, access.memoryAccess);
        else state.module.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
        return value;
    };
    const auto shift = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), misalignment, ConstantU32(state, 3u));
    const auto combine = [&](std::uint32_t low, std::uint32_t high) {
        auto value = EmitOrU32(state, low, high);
        if (bits != 32u) {
            const auto subwordShift = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), EmitAndConstant(state, address, 3u), ConstantU32(state, 3u));
            value = Binary(state, spv::OpShiftRightLogical, TypeU32(state), value, subwordShift);
        }
        return EmitAndConstant(state, value, bits == 32u ? 0xffffffffu : (1u << bits) - 1u);
    };
    const auto highShift = Binary(state, spv::OpISub, TypeU32(state), ConstantU32(state, 32u), shift);
    if (!LoadsSelectable(state, access)) {
        return EmitValueOrZeroIfCondition(state, inBounds, [&] {
            const auto low = Binary(state, spv::OpShiftRightLogical, TypeU32(state), load(index), shift);
            const auto high = EmitValueOrZeroIfCondition(state, unaligned, [&] { return Binary(state, spv::OpShiftLeftLogical, TypeU32(state), load(next), highShift); });
            return combine(low, high);
        });
    }
    const auto lowIndex = GuardedElementIndex(state, access, inBounds, index);
    const auto highIndex = Select(state, TypeU32(state), both(state, inBounds, unaligned), next, lowIndex);
    const auto low = Binary(state, spv::OpShiftRightLogical, TypeU32(state), load(lowIndex), shift);
    const auto high = EmitValueOrZero(state, unaligned, Binary(state, spv::OpShiftLeftLogical, TypeU32(state), load(highIndex), highShift));
    return EmitValueOrZero(state, inBounds, combine(low, high));
}

std::uint32_t byteBuffers(SpirvEmitterState& state) {
    if (state.storageBufferU8Variable != 0u) return state.storageBufferU8Variable;
    if (state.program.Resources().stage != IrShaderStage::Compute) return 0u;
    if (std::find(state.supportedCapabilities.begin(), state.supportedCapabilities.end(), static_cast<std::uint32_t>(spv::CapabilityStorageBuffer8BitAccess)) == state.supportedCapabilities.end()) return 0u;
    const IrBindingLayout& layout = state.program.Metadata().bindings;
    const auto binding = std::find_if(layout.descriptors.begin(), layout.descriptors.end(), [](const IrDescriptorBinding& descriptor) { return descriptor.kind == DescriptorBindingKind::Buffers; });
    if (binding == layout.descriptors.end() || state.storageBufferVariable == 0u) return 0u;
    state.module.EmitCapability(spv::CapabilityStorageBuffer8BitAccess);
    state.module.EmitExtension("SPV_KHR_8bit_storage");
    const auto count = ConstantU32(state, static_cast<std::uint32_t>(binding->resources.size()));
    const auto type = state.module.Type(spv::OpTypeArray, TypeStorageBufferU8Block(state), count);
    const auto variable = state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassStorageBuffer, type), spv::StorageClassStorageBuffer);
    state.module.AddName(variable, "buffers_u8");
    state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet, 0u);
    state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding, NativeBinding(state.program.Resources().stage, DescriptorBindingKind::Buffers));
    state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationAliased);
    if (state.storageBufferU64Variable == 0u && state.storageBufferCoherentVariable == 0u) state.module.AddAnnotation(spv::OpDecorate, state.storageBufferVariable, spv::DecorationAliased);
    if (state.requirements.coherentBuffers) state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationCoherent);
    state.storageBufferU8Variable = variable;
    return variable;
}

void writeBuffer(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t address, std::uint32_t value, std::uint32_t bits = 32u) {
    if (context.Memory(instruction).gpuDescriptor) {
        EmitBdaWrite(context, instruction, address, value, bits);
        return;
    }
    auto& state = context.state;
    const auto access = directAccess(context, instruction);
    const auto u32 = TypeU32(state);
    const auto index = directWordIndex(state, access, address);
    const auto misalignment = baseMisalignment(context, instruction);
    const auto unaligned = nonzero(state, misalignment);
    const auto next = EmitAddU32(state, index, ConstantU32(state, 1u));
    const auto last = Select(state, u32, unaligned, next, index);
    const auto bytes = bits != 32u ? byteBuffers(state) : 0u;
    if (bytes != 0u) {
        const auto element = ResourceForDescriptor(state, DescriptorBindingKind::Buffers, context.Memory(instruction).resource);
        const auto block = state.module.AllocateId();
        state.module.AddFunction(spv::OpAccessChain, TypeStorageBufferU8Pointer(state), block, bytes, ConstantU32(state, element));
        EmitIfCondition(state, EmitMemoryElementInBounds(state, access, last), [&] {
            const auto first = EmitAddU32(state, EmitAddU32(state, Binary(state, spv::OpShiftLeftLogical, u32, index, ConstantU32(state, 2u)), misalignment), EmitAndConstant(state, address, 3u));
            for (std::uint32_t byte = 0; byte < bits / 8u; ++byte) {
                const auto pointer = state.module.AllocateId();
                state.module.AddFunction(spv::OpAccessChain, TypeStorageBufferU8ElementPointer(state), pointer, block, ConstantU32(state, 0u), EmitAddU32(state, first, ConstantU32(state, byte)));
                const auto shifted = byte == 0u ? value : Binary(state, spv::OpShiftRightLogical, u32, value, ConstantU32(state, byte * 8u));
                const auto stored = Unary(state, spv::OpUConvert, TypeU8(state), shifted);
                if (access.memoryAccess != 0u) state.module.AddFunction(spv::OpStore, pointer, stored, access.memoryAccess);
                else state.module.AddFunction(spv::OpStore, pointer, stored);
            }
        });
        return;
    }
    EmitIfCondition(state, EmitMemoryElementInBounds(state, access, last), [&] {
        auto mask = ConstantU32(state, bits == 32u ? 0xffffffffu : (1u << bits) - 1u);
        auto masked = Binary(state, spv::OpBitwiseAnd, u32, value, mask);
        if (bits != 32u) {
            const auto subwordShift = Binary(state, spv::OpShiftLeftLogical, u32, EmitAndConstant(state, address, 3u), ConstantU32(state, 3u));
            mask = Binary(state, spv::OpShiftLeftLogical, u32, mask, subwordShift);
            masked = Binary(state, spv::OpShiftLeftLogical, u32, masked, subwordShift);
        }
        const auto update = [&](std::uint32_t word, std::uint32_t wordMask, std::uint32_t replacement) {
            AtomicUpdate(state, EmitMemoryElementPointer(state, access, word), access.kind, [&](std::uint32_t old) {
                return EmitOrU32(state, Binary(state, spv::OpBitwiseAnd, u32, old, Unary(state, spv::OpNot, u32, wordMask)), replacement);
            });
        };
        if (bits == 32u) {
            EmitIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), [&] {
                const auto pointer = EmitMemoryElementPointer(state, access, index);
                if (access.memoryAccess != 0u) state.module.AddFunction(spv::OpStore, pointer, value, access.memoryAccess);
                else state.module.AddFunction(spv::OpStore, pointer, value);
            });
        }
        EmitIfCondition(state, bits == 32u ? unaligned : ConstantBool(state, true), [&] {
            const auto shift = Binary(state, spv::OpShiftLeftLogical, u32, misalignment, ConstantU32(state, 3u));
            update(index, Binary(state, spv::OpShiftLeftLogical, u32, mask, shift), Binary(state, spv::OpShiftLeftLogical, u32, masked, shift));
            EmitIfCondition(state, unaligned, [&] {
                const auto remaining = Binary(state, spv::OpISub, u32, ConstantU32(state, 32u), shift);
                update(next, Binary(state, spv::OpShiftRightLogical, u32, mask, remaining), Binary(state, spv::OpShiftRightLogical, u32, masked, remaining));
            });
        });
    });
}

std::uint32_t composite(SpirvEmitterState& state, std::uint32_t components, const std::array<std::uint32_t, 4>& values) {
    if (components == 1u) return values[0];
    const auto result = state.module.AllocateId();
    std::vector<std::uint32_t> words{spv::OpCompositeConstruct, TypeU32Composite(state, components), result};
    words.insert(words.end(), values.begin(), values.begin() + components);
    state.module.AddFunction(words);
    return result;
}

std::uint32_t storeData(SpirvValueEmitContext& context, const IrValue& instruction) {
    return context.Arg(instruction, instruction.ArgumentCount() - 2u);
}

std::array<std::uint32_t, 4> storeValues(SpirvEmitterState& state, std::uint32_t data, std::uint32_t components) {
    std::array<std::uint32_t, 4> result{};
    for (std::uint32_t component = 0; component < components; ++component) {
        result[component] = data;
        if (components != 1u) {
            result[component] = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), result[component], data, component);
        }
    }
    return result;
}

std::uint32_t formatted(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& unbounded, const SpirvBufferFormatInfo& info, std::uint32_t components, bool store, std::uint32_t data) {
    auto& state = context.state;
    const auto resource = withBounds(context, instruction, unbounded);
    const auto& memory = context.Memory(instruction);
    const auto u32 = TypeU32(state);
    std::array<std::uint32_t, 4> values{};
    std::array<std::uint32_t, 4> selectors{};
    std::array<std::uint32_t, 4> selected{};
    std::array<RuntimeAddress, 4> addresses{};
    auto inBounds = ConstantBool(state, true);
    for (std::uint32_t output = 0; output < components; ++output) {
        selectors[output] = memory.typed ? ConstantU32(state, output < info.componentCount ? 4u + output : 0u) : field(state, resource.word3, output * 3u, 3u);
        selected[output] = Binary(state, spv::OpUMod, u32, Binary(state, spv::OpISub, u32, selectors[output], ConstantU32(state, 4u)), ConstantU32(state, info.componentCount));
    }
    std::array<std::uint32_t, 4> required{};
    for (std::uint32_t component = 0; component < info.componentCount; ++component) {
        required[component] = ConstantBool(state, store);
        if (!store) {
            for (std::uint32_t output = 0; output < components; ++output) {
                const auto fromMemory = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), selectors[output], ConstantU32(state, 4u));
                required[component] = Binary(state, spv::OpLogicalOr, TypeBool(state), required[component], both(state, fromMemory, equal(state, selected[output], ConstantU32(state, component))));
            }
        }
        addresses[component] = address(context, instruction, resource, GetFormatComponentByteOffset(info, component), info.packedBitfield ? 4u : info.componentBits[component] / 8u);
        inBounds = both(state, inBounds, Binary(state, spv::OpLogicalOr, TypeBool(state), Unary(state, spv::OpLogicalNot, TypeBool(state), required[component]), addresses[component].inBounds));
    }
    if (store) {
        const auto values = storeValues(state, data, components);
        EmitIfCondition(state, inBounds, [&] {
            auto packed = ConstantU32(state, 0u);
            for (std::uint32_t component = 0; component < info.componentCount; ++component) {
                const auto encoded = component >= components ? ConstantU32(state, 0u) : memory.d16 ? EmitD16StoreComponent(state, info, component, values[component]) : EmitFormatStoreComponent(state, info, component, values[component]);
                if (info.packedBitfield) {
                    packed = EmitOrU32(state, packed, Binary(state, spv::OpShiftLeftLogical, u32, EmitAndConstant(state, encoded, (1u << info.componentBits[component]) - 1u), ConstantU32(state, info.componentBitOffset[component])));
                } else {
                    writeBuffer(context, instruction, addresses[component].guest, encoded, info.componentBits[component]);
                }
            }
            if (info.packedBitfield) writeBuffer(context, instruction, addresses[0].guest, packed);
        });
        return ConstantU32(state, 0u);
    }
    for (std::uint32_t component = 0; component < info.componentCount; ++component) {
        values[component] = EmitValueOrZeroIfCondition(state, both(state, inBounds, required[component]), [&] {
            const auto bits = info.componentBits[component];
            auto raw = readBuffer(context, instruction, addresses[component].guest, info.packedBitfield ? 32u : bits);
            if (bits != 32u) {
                const auto signedType = IsSignedFormatComponent(info.type);
                const auto type = signedType ? TypeI32(state) : u32;
                const auto input = signedType ? Unary(state, spv::OpBitcast, type, raw) : raw;
                raw = state.module.AllocateId();
                state.module.AddFunction(signedType ? spv::OpBitFieldSExtract : spv::OpBitFieldUExtract, type, raw, input, ConstantU32(state, info.packedBitfield ? info.componentBitOffset[component] : 0u), ConstantU32(state, bits));
                if (signedType) raw = Unary(state, spv::OpBitcast, u32, raw);
            }
            return memory.d16 ? EmitD16FormatComponent(state, info, component, raw) : NormalizeFormatComponent(state, info, component, raw);
        });
    }
    const bool integer = info.type == SpirvFormatComponentType::Uint || info.type == SpirvFormatComponentType::Sint;
    const auto one = ConstantU32(state, integer ? 1u : memory.d16 ? 0x3c00u : 0x3f800000u);
    std::array<std::uint32_t, 4> outputs{};
    for (std::uint32_t output = 0; output < components; ++output) {
        auto value = values[0];
        for (std::uint32_t component = 1; component < info.componentCount; ++component) value = Select(state, u32, equal(state, selected[output], ConstantU32(state, component)), values[component], value);
        value = Select(state, u32, equal(state, selectors[output], ConstantU32(state, 1u)), one, value);
        outputs[output] = Select(state, u32, equal(state, selectors[output], ConstantU32(state, 0u)), ConstantU32(state, 0u), value);
    }
    return composite(state, components, outputs);
}

std::array<std::uint32_t, 3> FormatTableEntry(std::uint32_t index) {
    std::array<std::uint32_t, 3> entry{};
    if (index == 0u) return entry;
    const auto info = GetFormatInfo(static_cast<IrBufferFormat>(index));
    entry[0] = info.componentCount | (info.packedBitfield ? 8u : 0u) | (static_cast<std::uint32_t>(info.type) << 4u);
    for (std::uint32_t component = 0; component < info.componentCount; ++component) {
        entry[0] |= info.componentBits[component] << (8u + component * 6u);
        entry[1] |= GetFormatComponentByteOffset(info, component) << (component * 8u);
        entry[2] |= info.componentBitOffset[component] << (component * 8u);
    }
    return entry;
}

std::uint32_t genericFormattedLoad(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& unbounded, std::uint32_t components, std::uint32_t format, std::uint32_t formats) {
    auto& state = context.state;
    const auto resource = withBounds(context, instruction, unbounded);
    const auto& memory = context.Memory(instruction);
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    std::array<std::uint32_t, 3> entries{};
    const auto merge = state.module.AllocateId();
    const auto none = state.module.AllocateId();
    std::vector<std::uint32_t> branch{spv::OpSwitch, format, none};
    std::array<std::vector<std::uint32_t>, 3> phis;
    for (std::uint32_t word = 0; word < 3u; ++word) {
        entries[word] = state.module.AllocateId();
        phis[word] = {spv::OpPhi, u32, entries[word]};
    }
    std::vector<std::uint32_t> labels;
    for (std::uint32_t index = 1; index <= formats; ++index) {
        labels.push_back(state.module.AllocateId());
        branch.insert(branch.end(), {index, labels.back()});
    }
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(branch);
    for (std::uint32_t index = 0; index <= formats; ++index) {
        const auto label = index == 0u ? none : labels[index - 1u];
        EmitLabel(state, label);
        const auto words = FormatTableEntry(index);
        for (std::uint32_t word = 0; word < 3u; ++word) phis[word].insert(phis[word].end(), {constant(words[word]), label});
        state.module.AddFunction(spv::OpBranch, merge);
    }
    EmitLabel(state, merge);
    for (const auto& phi : phis) state.module.AddFunction(phi);
    const auto layout = entries[0];
    const auto byteOffsets = entries[1];
    const auto bitOffsets = entries[2];
    const auto componentCount = field(state, layout, 0u, 3u);
    const auto packed = nonzero(state, field(state, layout, 3u, 1u));
    const auto type = field(state, layout, 4u, 4u);
    const auto isType = [&](SpirvFormatComponentType value) { return equal(state, type, constant(static_cast<std::uint32_t>(value))); };
    const auto signedType = Binary(state, spv::OpLogicalOr, boolean, isType(SpirvFormatComponentType::Sint), Binary(state, spv::OpLogicalOr, boolean, isType(SpirvFormatComponentType::Snorm), isType(SpirvFormatComponentType::Sscaled)));
    const auto integer = Binary(state, spv::OpLogicalOr, boolean, isType(SpirvFormatComponentType::Uint), isType(SpirvFormatComponentType::Sint));
    std::array<std::uint32_t, 4> bits{};
    std::array<std::uint32_t, 4> active{};
    for (std::uint32_t component = 0; component < 4u; ++component) {
        bits[component] = field(state, layout, 8u + component * 6u, 6u);
        active[component] = Binary(state, spv::OpULessThan, boolean, constant(component), componentCount);
    }
    std::array<std::uint32_t, 4> selectors{};
    std::array<std::uint32_t, 4> selected{};
    for (std::uint32_t output = 0; output < components; ++output) {
        selectors[output] = field(state, resource.word3, output * 3u, 3u);
        selected[output] = Binary(state, spv::OpUMod, u32, Binary(state, spv::OpISub, u32, selectors[output], constant(4u)), componentCount);
    }
    std::array<std::uint32_t, 4> required{};
    std::array<RuntimeAddress, 4> addresses{};
    auto inBounds = ConstantBool(state, true);
    for (std::uint32_t component = 0; component < 4u; ++component) {
        required[component] = ConstantBool(state, false);
        for (std::uint32_t output = 0; output < components; ++output) {
            const auto fromMemory = Binary(state, spv::OpUGreaterThanEqual, boolean, selectors[output], constant(4u));
            required[component] = Binary(state, spv::OpLogicalOr, boolean, required[component], both(state, fromMemory, equal(state, selected[output], constant(component))));
        }
        required[component] = Select(state, boolean, active[component], required[component], ConstantBool(state, false));
        const auto componentOffset = field(state, byteOffsets, component * 8u, 8u);
        const auto bytes = Select(state, u32, packed, constant(4u), Binary(state, spv::OpShiftRightLogical, u32, bits[component], constant(3u)));
        const auto offset = Binary(state, spv::OpIAdd, u32, resource.offset != 0u ? resource.offset : Binary(state, spv::OpIAdd, u32, context.Arg(instruction, 2u), constant(memory.offset)), componentOffset);
        addresses[component] = byteAddress(context, instruction, resource, offset, bytes);
        if (memory.gpuDescriptor) {
            const auto u64 = TypeScalarU64(state);
            const auto guest = Binary(state, spv::OpIAdd, u64, resource.base, Unary(state, spv::OpUConvert, u64, addresses[component].guest));
            const auto overflow = Select(state, boolean, active[component], Binary(state, spv::OpULessThan, boolean, guest, resource.base), ConstantBool(state, false));
            EmitIfCondition(state, overflow, [&] { RecordBdaFault(state, resource.base, constant(0u), BdaInstructionPc(state, instruction), BdaAbi::FaultReason::Overflow); });
            StopBdaInvocationIf(state, overflow);
            addresses[component].guest = guest;
        }
        inBounds = Select(state, boolean, required[component], both(state, inBounds, addresses[component].inBounds), inBounds);
    }
    // One span read for the element instead of a read (dword or bytes, each with its own table lookup) per component:
    // the reads were over half of the inlined helper (four get_bda_pointer + read_bda_span pairs). The dwords are
    // 4-byte aligned in guest memory, so each holds bytes a component needs and touches no page the per-component reads
    // did not; only a 16-byte element at an unaligned start needs a fifth dword, read on its own. APS5_NO_SINGLE_FORMAT_READ=1
    // reads per component.
    static const bool singleRead = std::getenv("APS5_NO_SINGLE_FORMAT_READ") == nullptr;
    const bool spanRead = singleRead && memory.gpuDescriptor && HasBdaSpanReads(state, instruction);
    std::array<std::uint32_t, 4> spanRaw{};
    if (spanRead) {
        const auto u64 = TypeScalarU64(state);
        const auto start = addresses[0].guest;
        const auto shift = Binary(state, spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, start), constant(3u));
        const auto base = Binary(state, spv::OpISub, u64, start, Unary(state, spv::OpUConvert, u64, shift));
        std::array<std::uint32_t, 4> firstDword{};
        std::array<std::uint32_t, 4> byteShift{};
        std::array<std::uint32_t, 4> straddles{};
        auto used = constant(0u);
        auto fifth = ConstantBool(state, false);
        for (std::uint32_t component = 0; component < 4u; ++component) {
            const auto absolute = Binary(state, spv::OpIAdd, u32, shift, field(state, byteOffsets, component * 8u, 8u));
            const auto bytes = Select(state, u32, packed, constant(4u), Binary(state, spv::OpShiftRightLogical, u32, bits[component], constant(3u)));
            firstDword[component] = Binary(state, spv::OpShiftRightLogical, u32, absolute, constant(2u));
            const auto within = Binary(state, spv::OpBitwiseAnd, u32, absolute, constant(3u));
            byteShift[component] = Binary(state, spv::OpShiftLeftLogical, u32, within, constant(3u));
            straddles[component] = Binary(state, spv::OpUGreaterThan, boolean, Binary(state, spv::OpIAdd, u32, within, bytes), constant(4u));
            const auto own = Binary(state, spv::OpShiftLeftLogical, u32, constant(1u), firstDword[component]);
            const auto next = Select(state, u32, straddles[component], Binary(state, spv::OpShiftLeftLogical, u32, own, constant(1u)), constant(0u));
            used = Binary(state, spv::OpBitwiseOr, u32, used, Select(state, u32, required[component], Binary(state, spv::OpBitwiseOr, u32, own, next), constant(0u)));
            fifth = Binary(state, spv::OpLogicalOr, boolean, fifth, both(state, required[component], both(state, straddles[component], equal(state, firstDword[component], constant(3u)))));
        }
        const auto vector = TypeU32Vector(state, 4);
        const auto zeros = state.module.Constant(spv::OpConstantComposite, vector, constant(0u), constant(0u), constant(0u), constant(0u));
        const auto spanUsed = Binary(state, spv::OpBitwiseAnd, u32, used, constant(0xfu));
        const auto dwords = EmitValueIfElse(state, both(state, inBounds, nonzero(state, spanUsed)), vector, [&] { return EmitBdaSpanRead(context, instruction, base, constant(0u), spanUsed); }, [&] { return zeros; });
        const auto extra = EmitValueOrZeroIfCondition(state, both(state, inBounds, fifth), [&] { return EmitBdaDwordReads(context, instruction, base, 16u, 1u, true)[0]; });
        const auto dwordAt = [&](std::uint32_t index) {
            const auto inVector = state.module.AllocateId();
            state.module.AddFunction(spv::OpVectorExtractDynamic, u32, inVector, dwords, Unary(state, spv::OpBitcast, u32, Binary(state, spv::OpBitwiseAnd, u32, index, constant(3u))));
            return Select(state, u32, equal(state, index, constant(4u)), extra, inVector);
        };
        for (std::uint32_t component = 0; component < 4u; ++component) {
            const auto low = Binary(state, spv::OpShiftRightLogical, u32, dwordAt(firstDword[component]), byteShift[component]);
            const auto highShift = Binary(state, spv::OpISub, u32, constant(32u), byteShift[component]);
            const auto high = Select(state, u32, straddles[component], Binary(state, spv::OpShiftLeftLogical, u32, dwordAt(Binary(state, spv::OpIAdd, u32, firstDword[component], constant(1u))), Binary(state, spv::OpBitwiseAnd, u32, highShift, constant(31u))), constant(0u));
            const auto bytes = Select(state, u32, packed, constant(4u), Binary(state, spv::OpShiftRightLogical, u32, bits[component], constant(3u)));
            const auto mask = Select(state, u32, equal(state, bytes, constant(4u)), constant(0xffffffffu), Binary(state, spv::OpISub, u32, Binary(state, spv::OpShiftLeftLogical, u32, constant(1u), Binary(state, spv::OpShiftLeftLogical, u32, bytes, constant(3u))), constant(1u)));
            spanRaw[component] = Binary(state, spv::OpBitwiseAnd, u32, Binary(state, spv::OpBitwiseOr, u32, low, high), mask);
        }
    }
    std::array<std::uint32_t, 4> values{};
    for (std::uint32_t component = 0; component < 4u; ++component) {
        values[component] = EmitValueOrZeroIfCondition(state, Select(state, boolean, required[component], inBounds, ConstantBool(state, false)), [&] {
            const auto width = Select(state, u32, packed, constant(32u), bits[component]);
            const auto dword = equal(state, width, constant(32u));
            const auto raw = spanRead ? spanRaw[component] : memory.gpuDescriptor
                ? EmitValueIfElse(state, dword, u32, [&] { return readBuffer(context, instruction, addresses[component].guest, 32u); }, [&] {
                    return EmitBdaReadBytes(context, instruction, addresses[component].guest, Binary(state, spv::OpShiftRightLogical, u32, width, constant(3u)));
                })
                : Binary(state, spv::OpShiftRightLogical, u32, readBuffer(context, instruction, addresses[component].guest, 32u), Select(state, u32, dword, constant(0u), Binary(state, spv::OpShiftLeftLogical, u32, EmitAndConstant(state, addresses[component].guest, 3u), constant(3u))));
            const auto bitOffset = Select(state, u32, packed, field(state, bitOffsets, component * 8u, 8u), constant(0u));
            const auto unsignedField = EmitBitFieldUExtract(state, raw, bitOffset, bits[component]);
            const auto signedField = state.module.AllocateId();
            state.module.AddFunction(spv::OpBitFieldSExtract, TypeI32(state), signedField, Unary(state, spv::OpBitcast, TypeI32(state), raw), bitOffset, bits[component]);
            const auto extracted = Select(state, u32, equal(state, bits[component], constant(32u)), raw, Select(state, u32, signedType, Unary(state, spv::OpBitcast, u32, signedField), unsignedField));
            // One straight-line conversion from the format table's type and width instead of a switch over every
            // (type, width) pair the formats use: the switch was ~3/4 of a runtime-format load (~660 SPIR-V instructions a
            // component), and the driver inlines the load helper at every call (0x80002a9000: 112 calls, 1.81M GPU
            // instructions). APS5_NO_COMPACT_FORMAT_CONVERT=1 switches for every load.
            static const bool compact = std::getenv("APS5_NO_COMPACT_FORMAT_CONVERT") == nullptr;
            if (compact) return memory.d16 ? EmitD16FormatComponentRuntime(state, type, bits[component], extracted) : NormalizeFormatComponentRuntime(state, type, bits[component], extracted);
            std::vector<std::pair<SpirvFormatComponentType, std::uint32_t>> conversions;
            std::vector<std::vector<std::uint32_t>> cases;
            for (std::uint32_t index = 1; index <= formats; ++index) {
                const auto info = GetFormatInfo(static_cast<IrBufferFormat>(index));
                if (component >= info.componentCount) continue;
                const std::pair<SpirvFormatComponentType, std::uint32_t> key{info.type, info.componentBits[component]};
                if (std::find(conversions.begin(), conversions.end(), key) == conversions.end()) {
                    conversions.push_back(key);
                    cases.emplace_back();
                }
                cases[static_cast<std::size_t>(std::find(conversions.begin(), conversions.end(), key) - conversions.begin())].push_back(index);
            }
            const auto merge = state.module.AllocateId();
            const auto none = state.module.AllocateId();
            std::vector<std::uint32_t> labels;
            std::vector<std::uint32_t> branch{spv::OpSwitch, format, none};
            for (const auto& indices : cases) {
                labels.push_back(state.module.AllocateId());
                for (const auto index : indices) branch.insert(branch.end(), {index, labels.back()});
            }
            const auto value = state.module.AllocateId();
            std::vector<std::uint32_t> phi{spv::OpPhi, u32, value};
            state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
            state.module.AddFunction(branch);
            for (std::size_t conversion = 0; conversion < conversions.size(); ++conversion) {
                EmitLabel(state, labels[conversion]);
                SpirvBufferFormatInfo info{};
                info.type = conversions[conversion].first;
                info.componentCount = 1u;
                info.componentBits[0] = conversions[conversion].second;
                const auto converted = memory.d16 ? EmitD16FormatComponent(state, info, 0u, extracted) : NormalizeFormatComponent(state, info, 0u, extracted);
                phi.insert(phi.end(), {converted, state.currentLabel});
                state.module.AddFunction(spv::OpBranch, merge);
            }
            EmitLabel(state, none);
            phi.insert(phi.end(), {constant(0u), none});
            state.module.AddFunction(spv::OpBranch, merge);
            EmitLabel(state, merge);
            state.module.AddFunction(phi);
            return value;
        });
    }
    const auto one = Select(state, u32, integer, constant(1u), constant(memory.d16 ? 0x3c00u : 0x3f800000u));
    std::array<std::uint32_t, 4> outputs{};
    for (std::uint32_t output = 0; output < components; ++output) {
        auto value = values[0];
        for (std::uint32_t component = 1; component < 4u; ++component) value = Select(state, u32, equal(state, selected[output], constant(component)), values[component], value);
        value = Select(state, u32, equal(state, selectors[output], constant(1u)), one, value);
        outputs[output] = Select(state, u32, equal(state, selectors[output], constant(0u)), constant(0u), value);
    }
    return composite(state, components, outputs);
}

std::uint32_t formatSwitch(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components, bool store, std::uint32_t data);
std::uint32_t formattedCall(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components, bool store, std::uint32_t data);
std::uint32_t formattedGpuCall(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components, bool store, std::uint32_t data);

std::uint32_t formattedDispatch(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components, bool store, std::uint32_t data) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    if (memory.typed) {
        const auto info = GetFormatInfo(DecodeTBufferFormat(memory.dataFormat, memory.numberFormat));
        if (info.type == SpirvFormatComponentType::Unknown) context.Fail(instruction, "typed buffer instruction has an invalid format");
        return formatted(context, instruction, resource, info, components, store, data);
    }
    if (!store && memory.gpuDescriptor) {
        auto invalid = ConstantBool(state, false);
        for (std::uint32_t component = 0; component < components; ++component) {
            const auto selector = field(state, resource.word3, component * 3u, 3u);
            invalid = Binary(state, spv::OpLogicalOr, TypeBool(state), invalid, Binary(state, spv::OpLogicalOr, TypeBool(state), equal(state, selector, ConstantU32(state, 2u)), equal(state, selector, ConstantU32(state, 3u))));
        }
        faultIf(context, instruction, invalid, resource.base, 0u, BdaAbi::FaultReason::InvalidDescriptor);
    }
    const auto* word = instruction.Argument(0)->Resolve()->Argument(3)->Resolve();
    if (memory.gpuDescriptor && word->HasImmediate()) {
        const auto bits = (word->ImmediateU32() >> 12u) & 0x7fu;
        const auto format = static_cast<IrBufferFormat>(GcnGuestShaders() ? GcnBufferFormats()[bits] : bits);
        if (format == IrBufferFormat::Invalid) context.Fail(instruction, "formatted buffer instruction has an invalid constant format");
        return formatted(context, instruction, resource, GetFormatInfo(format), components, store, data);
    }
    if (!memory.gpuDescriptor) return formattedCall(context, instruction, resource, components, store, data);
    return formattedGpuCall(context, instruction, resource, components, store, data);
}

std::uint32_t formatSwitch(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components, bool store, std::uint32_t data) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto format = field(state, resource.word3, 12u, 7u);
    constexpr std::uint32_t count = static_cast<std::uint32_t>(IrBufferFormat::Format32_32_32_32Float);
    if (!store) {
        const auto type = components == 1u ? TypeU32(state) : TypeU32Composite(state, components);
        const auto zero = components == 1u ? ConstantU32(state, 0u) : ConstantU32CompositeZero(state, components);
        const auto unsupported = Binary(state, spv::OpLogicalOr, TypeBool(state), equal(state, format, ConstantU32(state, 0u)), Binary(state, spv::OpUGreaterThan, TypeBool(state), format, ConstantU32(state, count)));
        const auto result = EmitValueIfElse(state, unsupported, type, [&] {
            if (memory.gpuDescriptor) RecordBdaFault(state, resource.base, ConstantU32(state, 0u), BdaInstructionPc(state, instruction), BdaAbi::FaultReason::InvalidDescriptor);
            return zero;
        }, [&] { return genericFormattedLoad(context, instruction, resource, components, format, count); });
        if (memory.gpuDescriptor) StopBdaInvocationIf(state, unsupported);
        return result;
    }
    const auto merge = state.module.AllocateId();
    const auto invalid = state.module.AllocateId();
    std::array<std::uint32_t, count> labels{};
    std::vector<std::uint32_t> branch{spv::OpSwitch, format, invalid};
    for (std::uint32_t index = 0; index < count; ++index) {
        labels[index] = state.module.AllocateId();
        branch.insert(branch.end(), {index + 1u, labels[index]});
    }
    const auto type = store || components == 1u ? TypeU32(state) : TypeU32Composite(state, components);
    const auto result = state.module.AllocateId();
    std::vector<std::uint32_t> phi{spv::OpPhi, type, result};
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(branch);
    for (std::uint32_t index = 0; index < count; ++index) {
        EmitLabel(state, labels[index]);
        const auto value = formatted(context, instruction, resource, GetFormatInfo(static_cast<IrBufferFormat>(index + 1u)), components, store, data);
        const auto exit = state.module.AllocateId();
        state.module.AddFunction(spv::OpBranch, exit);
        EmitLabel(state, exit);
        state.module.AddFunction(spv::OpBranch, merge);
        phi.insert(phi.end(), {value, exit});
    }
    EmitLabel(state, invalid);
    if (memory.gpuDescriptor) RecordBdaFault(state, resource.base, ConstantU32(state, 0u), BdaInstructionPc(state, instruction), BdaAbi::FaultReason::InvalidDescriptor);
    const auto zero = store || components == 1u ? ConstantU32(state, 0u) : ConstantU32CompositeZero(state, components);
    const auto invalidExit = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, invalidExit);
    EmitLabel(state, invalidExit);
    state.module.AddFunction(spv::OpBranch, merge);
    phi.insert(phi.end(), {zero, invalidExit});
    EmitLabel(state, merge);
    state.module.AddFunction(phi);
    const auto unsupported = Binary(state, spv::OpLogicalOr, TypeBool(state), equal(state, format, ConstantU32(state, 0u)), Binary(state, spv::OpUGreaterThan, TypeBool(state), format, ConstantU32(state, count)));
    if (memory.gpuDescriptor) StopBdaInvocationIf(state, unsupported);
    return result;
}

std::uint32_t formattedCall(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components, bool store, std::uint32_t data) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto u32 = TypeU32(state);
    const auto valueType = components == 1u ? u32 : TypeU32Composite(state, components);
    const auto resultType = store ? state.module.Type(spv::OpTypeVoid) : valueType;
    const auto element = ResourceForDescriptor(state, DescriptorBindingKind::Buffers, memory.resource);
    const std::array<std::uint32_t, 6> key{memory.resource, components, store ? 1u : 0u, memory.d16 ? 1u : 0u, memory.coherent ? 1u : 0u, static_cast<std::uint32_t>(memory.kind)};
    auto function = state.formattedBufferFunctions.find(key);
    if (function == state.formattedBufferFunctions.end()) {
        const auto id = state.module.AllocateId();
        const auto functionType = store ? state.module.Type(spv::OpTypeFunction, resultType, u32, u32, u32, u32, u32, u32, u32, u32, u32, valueType) : state.module.Type(spv::OpTypeFunction, resultType, u32, u32, u32, u32, u32, u32, u32, u32, u32);
        const auto label = state.currentLabel;
        const auto byteOffset = state.memoryByteOffsets.at(element);
        struct Restore {
            SpirvEmitterState& state;
            std::uint32_t element;
            std::uint32_t label;
            std::uint32_t byteOffset;
            ~Restore() {
                state.module.EndHelperFunction();
                state.memoryByteOffsets.at(element) = byteOffset;
                state.currentLabel = label;
            }
        };
        state.module.BeginHelperFunction();
        const Restore restore{state, element, label, byteOffset};
        state.module.AddFunction(spv::OpFunction, resultType, id, spv::FunctionControlDontInlineMask, functionType);
        std::array<std::uint32_t, 10> parameters{};
        for (std::uint32_t index = 0; index < (store ? 10u : 9u); ++index) {
            parameters[index] = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionParameter, index == 9u ? valueType : u32, parameters[index]);
        }
        EmitLabel(state, state.module.AllocateId());
        state.memoryByteOffsets.at(element) = parameters[4];
        const auto first = PipelineSpecialization::BufferBase + memory.resource * PipelineSpecialization::BufferWords;
        RuntimeDescriptor shared{};
        if (state.registrationModules) {
            const auto word3 = state.program.Info().buffers.at(memory.resource).descriptorFormatted ? state.module.SpecializationConstant(u32, first + 1u, 0u) : parameters[6];
            shared = {ConstantU32(state, 0u), parameters[3], parameters[5], word3, nonzero(state, parameters[7]), parameters[0], nonzero(state, parameters[8]), parameters[1], parameters[2]};
        } else if (state.program.Resources().stage == IrShaderStage::Compute) {
            const auto specialized = state.module.SpecializationConstant(u32, first + 1u, 0u);
            const auto word3 = state.program.Info().buffers.at(memory.resource).descriptorFormatted ? Select(state, u32, nonzero(state, state.module.SpecializationConstant(u32, first + PipelineSpecialization::BufferRuntimeFormatWord, 0u)), parameters[6], specialized) : specialized;
            shared = {ConstantU32(state, 0u), parameters[3], parameters[5], word3, nonzero(state, parameters[7]), parameters[0], nonzero(state, parameters[8]), parameters[1], parameters[2]};
        } else {
            const auto word1 = state.module.SpecializationConstant(u32, first, 0u);
            const auto stride = field(state, word1, 16u, 14u);
            const auto swizzle = both(state, nonzero(state, stride), nonzero(state, field(state, word1, 31u, 1u)));
            shared = {ConstantU32(state, 0u), parameters[3], stride, state.module.SpecializationConstant(u32, first + 1u, 0u), swizzle, parameters[0], nonzero(state, parameters[8]), parameters[1], parameters[2]};
        }
        const auto runtimeFormat = state.program.Info().buffers.at(memory.resource).descriptorFormatted && state.program.Resources().stage == IrShaderStage::Compute && !state.registrationModules;
        const auto absentRuntime = runtimeFormat && !store ? both(state, nonzero(state, state.module.SpecializationConstant(u32, first + PipelineSpecialization::BufferRuntimeFormatWord, 0u)), Unary(state, spv::OpLogicalNot, TypeBool(state), shared.valid)) : 0u;
        const auto value = absentRuntime != 0u ? EmitValueIfElse(state, absentRuntime, valueType, [&] { return components == 1u ? ConstantU32(state, 0u) : ConstantU32CompositeZero(state, components); }, [&] { return formatSwitch(context, instruction, shared, components, store, 0u); }) : formatSwitch(context, instruction, shared, components, store, store ? parameters[9] : 0u);
        if (store) state.module.AddFunction(spv::OpReturn);
        else state.module.AddFunction(spv::OpReturnValue, value);
        state.module.AddFunction(spv::OpFunctionEnd);
        function = state.formattedBufferFunctions.emplace(key, id).first;
    }
    const auto offset = Binary(state, spv::OpIAdd, u32, context.Arg(instruction, 2u), ConstantU32(state, memory.offset));
    const auto result = state.module.AllocateId();
    const auto flag = [&](std::uint32_t condition) { return Select(state, u32, condition, ConstantU32(state, 1u), ConstantU32(state, 0u)); };
    const auto swizzle = flag(resource.swizzle);
    const auto valid = flag(resource.valid);
    if (store) state.module.AddFunction(spv::OpFunctionCall, resultType, result, function->second, resource.index, offset, context.Arg(instruction, 3u), resource.records, state.memoryByteOffsets.at(element), resource.stride, resource.word3, swizzle, valid, data);
    else state.module.AddFunction(spv::OpFunctionCall, resultType, result, function->second, resource.index, offset, context.Arg(instruction, 3u), resource.records, state.memoryByteOffsets.at(element), resource.stride, resource.word3, swizzle, valid);
    return store ? ConstantU32(state, 0u) : result;
}

std::uint32_t formattedGpuCall(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components, bool store, std::uint32_t data) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const bool stops = state.bdaStopsInvocations;
    const auto valueType = components == 1u ? u32 : TypeU32Composite(state, components);
    const auto resultType = stops ? (store ? u32 : state.module.Type(spv::OpTypeStruct, valueType, u32)) : (store ? state.module.Type(spv::OpTypeVoid) : valueType);
    const std::array<std::uint32_t, 6> key{components, store ? 1u : 0u, memory.d16 ? 1u : 0u, memory.coherent ? 1u : 0u, static_cast<std::uint32_t>(memory.kind), stops ? 1u : 0u};
    auto function = state.formattedGpuBufferFunctions.find(key);
    if (function == state.formattedGpuBufferFunctions.end()) {
        const auto id = state.module.AllocateId();
        const auto functionType = store ? state.module.Type(spv::OpTypeFunction, resultType, u64, u32, u32, u32, boolean, u32, boolean, u32, u32, u32, valueType)
                                        : state.module.Type(spv::OpTypeFunction, resultType, u64, u32, u32, u32, boolean, u32, boolean, u32, u32, u32);
        struct Restore {
            SpirvEmitterState& state;
            std::uint32_t label;
            std::uint32_t stopValue;
            ~Restore() {
                state.module.EndHelperFunction();
                state.currentLabel = label;
                state.bdaStopValue = stopValue;
                state.bdaPcOverride = 0u;
            }
        };
        state.module.BeginHelperFunction();
        const Restore restore{state, state.currentLabel, state.bdaStopValue};
        state.module.AddName(id, std::string(store ? "store" : "load") + "_formatted_gpu_buffer_x" + std::to_string(components));
        state.module.AddFunction(spv::OpFunction, resultType, id, spv::FunctionControlDontInlineMask, functionType);
        const std::array<std::uint32_t, 11> types{u64, u32, u32, u32, boolean, u32, boolean, u32, u32, u32, valueType};
        std::array<std::uint32_t, 11> parameters{};
        for (std::uint32_t index = 0; index < (store ? 11u : 10u); ++index) {
            parameters[index] = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionParameter, types[index], parameters[index]);
        }
        EmitLabel(state, state.module.AllocateId());
        const auto zero = components == 1u ? ConstantU32(state, 0u) : ConstantU32CompositeZero(state, components);
        if (stops) state.bdaStopValue = store ? ConstantU32(state, 1u) : state.module.Constant(spv::OpConstantComposite, resultType, zero, ConstantU32(state, 1u));
        state.bdaPcOverride = parameters[9];
        const RuntimeDescriptor shared{parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8]};
        const auto value = formatSwitch(context, instruction, shared, components, store, store ? parameters[10] : 0u);
        if (!stops && store) {
            state.module.AddFunction(spv::OpReturn);
        } else {
            auto returned = stops && store ? ConstantU32(state, 0u) : value;
            if (stops && !store) {
                returned = state.module.AllocateId();
                state.module.AddFunction(spv::OpCompositeConstruct, resultType, returned, value, ConstantU32(state, 0u));
            }
            state.module.AddFunction(spv::OpReturnValue, returned);
        }
        state.module.AddFunction(spv::OpFunctionEnd);
        function = state.formattedGpuBufferFunctions.emplace(key, id).first;
    }
    const auto offset = Binary(state, spv::OpIAdd, u32, context.Arg(instruction, 2u), ConstantU32(state, memory.offset));
    std::vector<std::uint32_t> call{spv::OpFunctionCall, resultType, state.module.AllocateId(), function->second, resource.base, resource.records, resource.stride, resource.word3, resource.swizzle, resource.index, resource.valid, offset, context.Arg(instruction, 3u), BdaInstructionPc(state, instruction)};
    if (store) call.push_back(data);
    state.module.AddFunction(call);
    const auto result = call[2];
    if (!stops) return store ? ConstantU32(state, 0u) : result;
    if (store) {
        StopBdaInvocationIf(state, Binary(state, spv::OpINotEqual, boolean, result, ConstantU32(state, 0u)));
        return ConstantU32(state, 0u);
    }
    const auto stopped = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, u32, stopped, result, 1u);
    StopBdaInvocationIf(state, Binary(state, spv::OpINotEqual, boolean, stopped, ConstantU32(state, 0u)));
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, valueType, value, result, 0u);
    return value;
}

std::uint32_t rawGpuCall(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components, std::uint32_t offset) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const bool stops = state.bdaStopsInvocations;
    const auto valueType = TypeU32Composite(state, components);
    const auto resultType = stops ? state.module.Type(spv::OpTypeStruct, valueType, u32) : valueType;
    const std::array<std::uint32_t, 3> key{components, memory.coherent ? 1u : 0u, stops ? 1u : 0u};
    auto function = state.rawGpuBufferFunctions.find(key);
    if (function == state.rawGpuBufferFunctions.end()) {
        const auto id = state.module.AllocateId();
        const std::array<std::uint32_t, 10> types{u64, u32, u32, u32, boolean, u32, boolean, u32, u32, u32};
        const auto functionType = state.module.Type(spv::OpTypeFunction, resultType, types[0], types[1], types[2], types[3], types[4], types[5], types[6], types[7], types[8], types[9]);
        struct Restore {
            SpirvEmitterState& state;
            std::uint32_t label;
            std::uint32_t stopValue;
            ~Restore() {
                state.module.EndHelperFunction();
                state.currentLabel = label;
                state.bdaStopValue = stopValue;
                state.bdaPcOverride = 0u;
            }
        };
        state.module.BeginHelperFunction();
        const Restore restore{state, state.currentLabel, state.bdaStopValue};
        state.module.AddName(id, "load_raw_gpu_buffer_x" + std::to_string(components));
        state.module.AddFunction(spv::OpFunction, resultType, id, spv::FunctionControlDontInlineMask, functionType);
        std::array<std::uint32_t, 10> parameters{};
        for (std::uint32_t index = 0; index < parameters.size(); ++index) {
            parameters[index] = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionParameter, types[index], parameters[index]);
        }
        EmitLabel(state, state.module.AllocateId());
        const auto zero = ConstantU32CompositeZero(state, components);
        if (stops) state.bdaStopValue = state.module.Constant(spv::OpConstantComposite, resultType, zero, ConstantU32(state, 1u));
        state.bdaPcOverride = parameters[9];
        const RuntimeDescriptor shared{parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8]};
        std::array<std::uint32_t, 4> values{};
        for (std::uint32_t component = 0; component < components; ++component) {
            const auto access = address(context, instruction, shared, component * 4u, 4u);
            values[component] = EmitValueOrZeroIfCondition(state, access.inBounds, [&] { return readBuffer(context, instruction, access.guest, 32u); });
        }
        auto returned = composite(state, components, values);
        if (stops) {
            const auto value = returned;
            returned = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, resultType, returned, value, ConstantU32(state, 0u));
        }
        state.module.AddFunction(spv::OpReturnValue, returned);
        state.module.AddFunction(spv::OpFunctionEnd);
        function = state.rawGpuBufferFunctions.emplace(key, id).first;
    }
    std::vector<std::uint32_t> call{spv::OpFunctionCall, resultType, state.module.AllocateId(), function->second, resource.base, resource.records, resource.stride, resource.word3, resource.swizzle, resource.index, resource.valid, offset, resource.bounds.soffset, BdaInstructionPc(state, instruction)};
    state.module.AddFunction(call);
    const auto result = call[2];
    if (!stops) return result;
    const auto stopped = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, u32, stopped, result, 1u);
    StopBdaInvocationIf(state, Binary(state, spv::OpINotEqual, boolean, stopped, ConstantU32(state, 0u)));
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, valueType, value, result, 0u);
    return value;
}

std::uint32_t gpuSpan(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components, std::uint32_t offset, std::uint32_t& byte) {
    auto& state = context.state;
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto span = 4u * (components - 1u);
    const auto limit = ConstantU32(state, 0xffffffffu - span);
    byte = Binary(state, spv::OpIAdd, u32, Binary(state, spv::OpIAdd, u32, resource.bounds.linearBase, offset), resource.bounds.soffset);
    const auto contiguous = both(state, Unary(state, spv::OpLogicalNot, boolean, resource.swizzle), both(state, Binary(state, spv::OpULessThanEqual, boolean, offset, limit), Binary(state, spv::OpULessThanEqual, boolean, byte, limit)));
    return both(state, contiguous, inBounds(state, resource, Binary(state, spv::OpIAdd, u32, offset, ConstantU32(state, span)), ConstantU32(state, 4u)));
}

std::uint32_t gpuLoad(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto offset = Binary(state, spv::OpIAdd, TypeU32(state), context.Arg(instruction, 2u), ConstantU32(state, memory.offset));
    if (components == 1u) {
        const auto inside = inBounds(state, resource, offset, ConstantU32(state, memory.dataBits / 8u));
        const auto guest = guestAddress(state, resource, byteOffset(state, resource, offset));
        return EmitValueOrZeroIfCondition(state, inside, [&] { return readBuffer(context, instruction, guest, memory.dataBits); });
    }
    if (memory.dataBits != 32u) context.Fail(instruction, "multi-dword buffer load must read whole dwords");
    std::uint32_t byte = 0;
    const auto whole = gpuSpan(context, instruction, resource, components, offset, byte);
    return EmitValueIfElse(state, whole, TypeU32Composite(state, components), [&] {
        return composite(state, components, EmitBdaDwordReads(context, instruction, guestAddress(state, resource, byte), 0u, components, true));
    }, [&] { return rawGpuCall(context, instruction, resource, components, offset); });
}

void gpuStore(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto offset = Binary(state, spv::OpIAdd, TypeU32(state), context.Arg(instruction, 2u), ConstantU32(state, memory.offset));
    const auto data = storeData(context, instruction);
    if (components == 1u) {
        const auto inside = inBounds(state, resource, offset, ConstantU32(state, memory.dataBits / 8u));
        const auto guest = guestAddress(state, resource, byteOffset(state, resource, offset));
        EmitIfCondition(state, inside, [&] { writeBuffer(context, instruction, guest, data, memory.dataBits); });
        return;
    }
    if (memory.dataBits != 32u) context.Fail(instruction, "multi-dword buffer store must write whole dwords");
    std::uint32_t byte = 0;
    const auto whole = gpuSpan(context, instruction, resource, components, offset, byte);
    const auto mergeLabel = state.module.AllocateId();
    const auto wholeLabel = state.module.AllocateId();
    const auto partLabel = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, mergeLabel, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, whole, wholeLabel, partLabel);
    EmitLabel(state, wholeLabel);
    EmitBdaDwordWrites(context, instruction, guestAddress(state, resource, byte), 0u, components, data, true);
    state.module.AddFunction(spv::OpBranch, mergeLabel);
    EmitLabel(state, partLabel);
    const auto values = storeValues(state, data, components);
    for (std::uint32_t component = 0; component < components; ++component) {
        const auto access = address(context, instruction, resource, component * 4u, 4u);
        EmitIfCondition(state, access.inBounds, [&] { writeBuffer(context, instruction, access.guest, values[component]); });
    }
    state.module.AddFunction(spv::OpBranch, mergeLabel);
    EmitLabel(state, mergeLabel);
}

bool inlineFormat(SpirvValueEmitContext& context, const IrValue& instruction) {
    const auto& memory = context.Memory(instruction);
    return memory.typed || (memory.gpuDescriptor && descriptorHandle(context, instruction).Argument(3)->Resolve()->HasImmediate());
}

struct CoherentPair {
    MemoryResourceAccess wide;
    std::uint32_t index;
    std::uint32_t usable;
};

std::optional<CoherentPair> coherentPair(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    if (components != 2u || !memory.coherent || memory.gpuDescriptor || memory.dataBits != 32u || state.storageBufferU64Variable == 0u) return std::nullopt;
    const auto low = address(context, instruction, resource, 0u, 4u);
    const auto high = address(context, instruction, resource, 4u, 4u);
    const auto wide = directAccess(context, instruction, true);
    const auto byte = EmitAddU32(state, low.guest, wide.byteOffset);
    const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), byte, ConstantU32(state, 3u));
    const auto aligned = both(state, equal(state, baseMisalignment(context, instruction), ConstantU32(state, 0u)), equal(state, EmitAndConstant(state, byte, 7u), ConstantU32(state, 0u)));
    const auto adjacent = equal(state, high.guest, EmitAddU32(state, low.guest, ConstantU32(state, 4u)));
    const auto inBounds = both(state, both(state, low.inBounds, high.inBounds), EmitMemoryElementInBounds(state, wide, index));
    return CoherentPair{wide, index, both(state, both(state, aligned, adjacent), inBounds)};
}

}

std::uint32_t EmitRuntimeBufferLoad(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t components) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto type = components == 1u ? TypeU32(state) : TypeU32Composite(state, components);
    const auto zero = components == 1u ? ConstantU32(state, 0u) : ConstantU32CompositeZero(state, components);
    const auto resource = memory.formatted && !inlineFormat(context, instruction) ? descriptor(context, instruction) : bounded(context, instruction);
    return EmitValueOrDefaultIfCondition(state, context.Arg(instruction, instruction.ArgumentCount() - 1u), type, zero, [&] {
        checkDescriptor(context, instruction, resource);
        if (memory.formatted) return EmitValueOrDefaultIfCondition(state, resource.valid, type, zero, [&] { return formattedDispatch(context, instruction, resource, components, false, 0u); });
        if (memory.gpuDescriptor) return gpuLoad(context, instruction, resource, components);
        const auto readComponents = [&] {
            std::array<std::uint32_t, 4> values{};
            for (std::uint32_t component = 0; component < components; ++component) {
                const auto access = address(context, instruction, resource, component * 4u, memory.dataBits / 8u);
                values[component] = memory.gpuDescriptor ? EmitValueOrZeroIfCondition(state, access.inBounds, [&] { return readBuffer(context, instruction, access.guest, memory.dataBits); })
                                                         : readBuffer(context, instruction, access.guest, memory.dataBits, access.inBounds);
            }
            return composite(state, components, values);
        };
        const auto pair = coherentPair(context, instruction, resource, components);
        if (!pair) return readComponents();
        return EmitValueIfElse(state, pair->usable, type, [&] {
            const auto pointer = EmitStorageBufferElementPointer(state, pair->wide, pair->index, TypeStorageBufferU64ElementPointer(state));
            const auto value = state.module.AllocateId();
            state.module.AddFunction(spv::OpAtomicLoad, TypeScalarU64(state), value, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone));
            const auto high = Binary(state, spv::OpShiftRightLogical, TypeScalarU64(state), value, BdaConstant(state, 32u));
            return composite(state, 2u, {Unary(state, spv::OpUConvert, TypeU32(state), value), Unary(state, spv::OpUConvert, TypeU32(state), high), 0u, 0u});
        }, readComponents);
    });
}

void EmitRuntimeBufferStore(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t components) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto resource = memory.formatted && !inlineFormat(context, instruction) ? descriptor(context, instruction) : bounded(context, instruction);
    EmitIfCondition(state, context.Arg(instruction, instruction.ArgumentCount() - 1u), [&] {
        checkDescriptor(context, instruction, resource);
        if (memory.formatted) {
            EmitIfCondition(state, resource.valid, [&] { static_cast<void>(formattedDispatch(context, instruction, resource, components, true, storeData(context, instruction))); });
            return;
        }
        if (memory.gpuDescriptor) {
            gpuStore(context, instruction, resource, components);
            return;
        }
        const auto values = storeValues(state, storeData(context, instruction), components);
        const auto writeComponents = [&] {
            for (std::uint32_t component = 0; component < components; ++component) {
                const auto access = address(context, instruction, resource, component * 4u, memory.dataBits / 8u);
                EmitIfCondition(state, access.inBounds, [&] { writeBuffer(context, instruction, access.guest, values[component], memory.dataBits); });
            }
        };
        const auto pair = coherentPair(context, instruction, resource, components);
        if (!pair) {
            writeComponents();
            return;
        }
        EmitIfCondition(state, pair->usable, [&] {
            const auto u64 = TypeScalarU64(state);
            const auto high = Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, values[1]), BdaConstant(state, 32u));
            const auto value = Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, values[0]), high);
            const auto pointer = EmitStorageBufferElementPointer(state, pair->wide, pair->index, TypeStorageBufferU64ElementPointer(state));
            state.module.AddFunction(spv::OpAtomicStore, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone), value);
        });
        EmitIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), pair->usable), writeComponents);
    });
}

// The dwords of one s_buffer_load_dwordxN through a GPU V# that are still in the program: the ReadConstBuffer
// instructions right after `instruction` in its block with the same V#, offset operand and pc, at increasing
// offsets less than 64 bytes past the first.
std::vector<const IrValue*> scalarRun(SpirvValueEmitContext& context, const IrValue& instruction) {
    std::vector<const IrValue*> run{&instruction};
    const IrBlock* block = instruction.Parent();
    if (block == nullptr) return run;
    const auto& instructions = block->Instructions();
    auto at = std::find(instructions.begin(), instructions.end(), &instruction);
    if (at == instructions.end()) return run;
    const auto& first = context.Memory(instruction);
    const auto pc = instruction.Flags<MemoryFlags>().pc;
    for (++at; at != instructions.end(); ++at) {
        const IrValue& next = **at;
        if (next.Opcode() != IrOpcode::ReadConstBuffer || next.ArgumentCount() != 2u || next.Flags<MemoryFlags>().pc != pc) break;
        if (next.Argument(0)->Resolve() != instruction.Argument(0)->Resolve() || next.Argument(1)->Resolve() != instruction.Argument(1)->Resolve()) break;
        const auto& memory = context.Memory(next);
        const auto previous = context.Memory(*run.back()).offset;
        if (memory.kind != ResourceKind::ScalarBuffer || !memory.gpuDescriptor || memory.planningOnly || memory.resource != first.resource || memory.coherent != first.coherent) break;
        if (memory.offset <= previous || memory.offset - first.offset >= 64u || (memory.offset - first.offset) % 4u != 0u) break;
        run.push_back(&next);
    }
    return run;
}

std::uint32_t EmitRuntimeScalarBufferLoad(SpirvValueEmitContext& context, const IrValue& instruction) {
    auto& state = context.state;
    // A later dword of a run the first dword already read (below): its value dominates it when nothing was
    // emitted in between, so it is still the block the run ended in.
    if (const auto found = context.scalarRunValues.find(&instruction); found != context.scalarRunValues.end()) {
        const auto [value, label] = found->second;
        context.scalarRunValues.erase(found);
        if (label == state.currentLabel && state.runtimeBufferLabel == state.currentLabel) return value;
    }
    if (std::ranges::find(state.supportedCapabilities, spv::CapabilityInt64) == state.supportedCapabilities.end()) context.Fail(instruction, "scalar buffer bounds require shaderInt64");
    state.module.EmitCapability(spv::CapabilityInt64);
    // Before the descriptor check, whose fault branch ends the block: whether this load is emitted where its block's
    // instructions are, not inside a guard of its own.
    const bool hoisted = state.runtimeBufferLabel != 0u && state.currentLabel == state.runtimeBufferLabel;
    const auto resource = descriptor(context, instruction);
    checkDescriptor(context, instruction, resource);
    const auto u64 = TypeScalarU64(state);
    const auto size = Select(state, u64, equal(state, resource.stride, ConstantU32(state, 0u)), Unary(state, spv::OpUConvert, u64, resource.records), Binary(state, spv::OpIMul, u64, Unary(state, spv::OpUConvert, u64, resource.stride), Unary(state, spv::OpUConvert, u64, resource.records)));
    struct DwordBounds {
        std::uint32_t byte;
        std::uint32_t offset;
        std::uint32_t inBounds;
    };
    const auto bounds = [&](const IrValue& dword) {
        const auto address = EmitAddU32(state, context.Arg(dword, 1u), ConstantU32(state, context.Memory(dword).offset));
        const auto noCarry = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), address, context.Arg(dword, 1u));
        const auto byte = EmitAndConstant(state, address, ~3u);
        const auto offset = Unary(state, spv::OpUConvert, u64, byte);
        return DwordBounds{byte, offset, both(state, noCarry, both(state, resource.valid, Binary(state, spv::OpULessThanEqual, TypeBool(state), Binary(state, spv::OpIAdd, u64, offset, BdaConstant(state, 4u)), size)))};
    };
    if (context.Memory(instruction).gpuDescriptor && hoisted && HasBdaSpanReads(state, instruction)) {
        const auto run = scalarRun(context, instruction);
        if (run.size() > 1u) {
            // One span read per 4 dwords instead of a guarded read per dword. The in-bounds dwords of the run are a
            // prefix of it (a dword is in bounds when its end fits the V# range and its offset add did not wrap, and
            // both only get harder at higher offsets), and an in-bounds dword sits 4 bytes per dword past the first
            // one's byte, so a run-time mask of the in-bounds dwords reads exactly the dwords the per-dword reads
            // did, in the same order, with the same lookups and faults; the others read 0 as before.
            const auto u32 = TypeU32(state);
            const auto first = context.Memory(instruction).offset;
            std::array<DwordBounds, 16> dwords{};
            for (const IrValue* dword : run) dwords[(context.Memory(*dword).offset - first) / 4u] = bounds(*dword);
            const auto base = Binary(state, spv::OpIAdd, u64, resource.base, dwords[0].offset);
            const auto zeros = state.module.Constant(spv::OpConstantComposite, TypeU32Vector(state, 4), ConstantU32(state, 0u), ConstantU32(state, 0u), ConstantU32(state, 0u), ConstantU32(state, 0u));
            std::array<std::uint32_t, 4> reads{};
            for (std::uint32_t chunk = 0; chunk < 4u; ++chunk) {
                std::uint32_t used = 0;
                for (const IrValue* dword : run) {
                    const auto index = (context.Memory(*dword).offset - first) / 4u;
                    if (index / 4u != chunk) continue;
                    const auto bit = Select(state, u32, dwords[index].inBounds, ConstantU32(state, 1u << (index % 4u)), ConstantU32(state, 0u));
                    used = used == 0u ? bit : Binary(state, spv::OpBitwiseOr, u32, used, bit);
                }
                if (used == 0u) continue;
                reads[chunk] = EmitValueIfElse(state, nonzero(state, used), TypeU32Vector(state, 4), [&] {
                    return EmitBdaSpanRead(context, instruction, base, ConstantU32(state, chunk * 16u), used);
                }, [&] { return zeros; });
            }
            std::uint32_t result = 0;
            for (const IrValue* dword : run) {
                const auto index = (context.Memory(*dword).offset - first) / 4u;
                const auto value = state.module.AllocateId();
                state.module.AddFunction(spv::OpCompositeExtract, u32, value, reads[index / 4u], index % 4u);
                if (dword == &instruction) result = value;
                else context.scalarRunValues[dword] = {value, state.currentLabel};
            }
            return result;
        }
    }
    const auto dword = bounds(instruction);
    if (!context.Memory(instruction).gpuDescriptor) return readBuffer(context, instruction, dword.byte, 32u, dword.inBounds);
    return EmitValueOrZeroIfCondition(state, dword.inBounds, [&] { return readBuffer(context, instruction, AddBdaAddress(context, instruction, resource.base, dword.offset, false), 32u); });
}

std::uint32_t EmitRuntimeBufferAtomic(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t bits, std::uint32_t active, const std::function<std::uint32_t(std::uint32_t)>& operation) {
    auto& state = context.state;
    if (bits != 32u && bits != 64u) context.Fail(instruction, "unsupported runtime buffer atomic width");
    if (context.Memory(instruction).gpuDescriptor && (state.bdaAtomicPointerFunction == 0u || state.bdaNoteWriteFunction == 0u)) context.Fail(instruction, "BDA atomic functions are missing");
    if (!context.Memory(instruction).gpuDescriptor) {
        if (state.program.Resources().stage == IrShaderStage::TessellationControl) context.Fail(instruction, "BDA fault termination requires a barrier-safe tessellation-control execution protocol");
        if (std::ranges::find(state.supportedCapabilities, spv::CapabilityInt64) == state.supportedCapabilities.end()) context.Fail(instruction, "buffer atomic fault addresses require shaderInt64");
        state.module.EmitCapability(spv::CapabilityInt64);
    }
    const auto type = bits == 64u ? TypeU64(state) : TypeU32(state);
    const auto scalar = bits == 64u ? TypeScalarU64(state) : TypeU32(state);
    const auto zero = bits == 64u ? ConstantU64(state, 0u) : ConstantU32(state, 0u);
    const auto resource = bounded(context, instruction);
    return EmitValueOrDefaultIfCondition(state, active, type, zero, [&] {
        checkDescriptor(context, instruction, resource);
        const auto access = address(context, instruction, resource, 0u, bits / 8u);
        return EmitValueOrDefaultIfCondition(state, access.inBounds, type, zero, [&] {
            if (!context.Memory(instruction).gpuDescriptor) {
                const auto storage = directAccess(context, instruction, bits == 64u);
                const auto byte = EmitAddU32(state, access.guest, storage.byteOffset);
                const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), byte, ConstantU32(state, bits == 64u ? 3u : 2u));
                auto valid = EmitMemoryElementInBounds(state, storage, index);
                if (bits == 64u) {
                    const auto unaligned = nonzero(state, EmitAndConstant(state, byte, bits / 8u - 1u));
                    const auto guest = Binary(state, spv::OpIAdd, TypeScalarU64(state), resource.base, Unary(state, spv::OpUConvert, TypeScalarU64(state), access.guest));
                    EmitIfCondition(state, unaligned, [&] {
                        const auto high = Unary(state, spv::OpUConvert, TypeU32(state), Binary(state, spv::OpShiftRightLogical, TypeScalarU64(state), guest, BdaConstant(state, 32u)));
                        RecordBdaFaultWords(state, Unary(state, spv::OpUConvert, TypeU32(state), guest), high, ConstantU32(state, bits / 8u), BdaInstructionPc(state, instruction), BdaAbi::FaultReason::Unaligned);
                    });
                    StopBdaInvocationIf(state, unaligned);
                    valid = both(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), valid);
                }
                return EmitValueOrDefaultIfCondition(state, valid, type, zero, [&] {
                    const auto pointer = EmitStorageBufferElementPointer(state, storage, index, bits == 64u ? TypeStorageBufferU64ElementPointer(state) : TypeStorageBufferElementPointer(state));
                    return operation(pointer);
                });
            }
            const auto atomicAt = [&](std::uint32_t guest) {
                const auto physical = state.module.AllocateId();
                state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaAtomicPointerFunction, guest, ConstantU32(state, bits / 8u), BdaInstructionPc(state, instruction));
                const auto mapped = Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u));
                return EmitValueOrDefaultIfCondition(state, mapped, type, zero, [&] {
                    const auto pointer = state.module.AllocateId();
                    state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, scalar), pointer, physical);
                    const auto result = operation(pointer);
                    state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, guest);
                    return result;
                });
            };
            if (bits == 32u) return atomicAt(Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), access.guest, BdaConstant(state, ~std::uint64_t{3u})));
            const auto unaligned = Binary(state, spv::OpINotEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), access.guest, BdaConstant(state, bits / 8u - 1u)), BdaConstant(state, 0u));
            faultIf(context, instruction, unaligned, access.guest, bits / 8u, BdaAbi::FaultReason::Unaligned);
            return EmitValueOrDefaultIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), type, zero, [&] { return atomicAt(access.guest); });
        });
    });
}

}
