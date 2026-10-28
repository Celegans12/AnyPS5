#include "Optimization/TransferKernel.hpp"

namespace ShaderRecompiler {

namespace {

std::uint32_t Components(IrOpcode opcode) {
    switch (opcode) {
        case IrOpcode::StoreBufferU32:
        case IrOpcode::LoadBufferU32:
            return 1u;
        case IrOpcode::StoreBufferU32x2:
        case IrOpcode::LoadBufferU32x2:
            return 2u;
        case IrOpcode::StoreBufferU32x3:
        case IrOpcode::LoadBufferU32x3:
            return 3u;
        case IrOpcode::StoreBufferU32x4:
        case IrOpcode::LoadBufferU32x4:
            return 4u;
        default:
            return 0u;
    }
}

bool IsLoad(IrOpcode opcode) {
    return opcode == IrOpcode::LoadBufferU32 || opcode == IrOpcode::LoadBufferU32x2 || opcode == IrOpcode::LoadBufferU32x3 || opcode == IrOpcode::LoadBufferU32x4;
}

IrOpcode Construct(std::uint32_t components) {
    switch (components) {
        case 2u: return IrOpcode::CompositeConstructU32x2;
        case 3u: return IrOpcode::CompositeConstructU32x3;
        case 4u: return IrOpcode::CompositeConstructU32x4;
        default: return IrOpcode::Count;
    }
}

IrOpcode Extract(std::uint32_t components) {
    switch (components) {
        case 2u: return IrOpcode::CompositeExtractU32x2;
        case 3u: return IrOpcode::CompositeExtractU32x3;
        case 4u: return IrOpcode::CompositeExtractU32x4;
        default: return IrOpcode::Count;
    }
}

bool ConstantU32(const IrValue* value, std::uint32_t expected) {
    const auto* resolved = value->Resolve();
    return resolved->HasImmediate() && resolved->Type() == IrType::U32 && resolved->ImmediateU32() == expected;
}

bool AllLanes(const IrValue* value) {
    const auto* resolved = value->Resolve();
    return resolved->HasImmediate() && resolved->Type() == IrType::Bool && resolved->ImmediateBool();
}

const IrValue* ThroughBitCasts(const IrValue* value) {
    const auto* resolved = value->Resolve();
    while ((resolved->Opcode() == IrOpcode::BitCastU32F32 || resolved->Opcode() == IrOpcode::BitCastF32U32) && resolved->ArgumentCount() == 1u) resolved = resolved->Argument(0)->Resolve();
    return resolved;
}

std::optional<std::uint32_t> UserDataRegister(const IrValue* value) {
    const auto* resolved = value->Resolve();
    if (resolved->Opcode() != IrOpcode::GetUserData || resolved->ArgumentCount() != 1u) return std::nullopt;
    const auto* operand = resolved->Argument(0);
    if (operand->Type() != IrType::ScalarReg) return std::nullopt;
    return operand->Register().index;
}

std::optional<std::uint32_t> DescriptorRegister(const IrValue* handle) {
    const auto* resolved = handle->Resolve();
    if (resolved->Opcode() != IrOpcode::GetBufferResource || resolved->ArgumentCount() != 4u) return std::nullopt;
    const auto first = UserDataRegister(resolved->Argument(0));
    if (!first) return std::nullopt;
    for (std::uint32_t word = 1; word < 4u; ++word) {
        const auto next = UserDataRegister(resolved->Argument(word));
        if (!next || *next != *first + word) return std::nullopt;
    }
    return first;
}

const MemoryInfo* BufferSite(const IrProgram& program, const IrValue* access) {
    const auto& memory = program.Resources().memoryInfo;
    const auto index = access->Flags<MemoryFlags>().index;
    if (index >= memory.size()) return nullptr;
    const auto& site = memory[index];
    // fable also requires gpuDescriptor: there every runtime V# takes the BDA path. Here a V# in user data that validates
    // is bound instead, with the same meaning (a runtime V#), so the flag is not required.
    if (site.kind != ResourceKind::Buffer || site.offset != 0u || site.typed || site.d16 || site.dataBits != 32u) return nullptr;
    return &site;
}

bool WritesMemory(IrOpcode opcode) {
    const auto buffer = BufferAccessOf(opcode);
    const auto address = AddressOpcodeInfoOf(opcode).access;
    const auto image = ImageOpcodeInfoOf(opcode).access;
    return buffer == BufferAccess::Write || buffer == BufferAccess::Atomic || address == AddressAccess::Write || address == AddressAccess::Atomic || image == ImageAccess::Write || image == ImageAccess::Atomic || opcode == IrOpcode::DataAppend || opcode == IrOpcode::DataConsume;
}

const IrValue* LoadOfAll(const IrValue* data, std::uint32_t components) {
    const auto* resolved = data->Resolve();
    if (components == 1u) return IsLoad(resolved->Opcode()) && Components(resolved->Opcode()) == 1u ? resolved : nullptr;
    if (IsLoad(resolved->Opcode())) return Components(resolved->Opcode()) == components ? resolved : nullptr;
    if (resolved->Opcode() != Construct(components) || resolved->ArgumentCount() != components) return nullptr;
    const IrValue* load = nullptr;
    for (std::uint32_t component = 0; component < components; ++component) {
        const auto* part = resolved->Argument(component)->Resolve();
        if (part->Opcode() != Extract(components) || part->ArgumentCount() != 2u || !ConstantU32(part->Argument(1), component)) return nullptr;
        const auto* whole = part->Argument(0)->Resolve();
        if (!IsLoad(whole->Opcode()) || Components(whole->Opcode()) != components || (load != nullptr && load != whole)) return nullptr;
        load = whole;
    }
    return load;
}

bool FillValues(const IrProgram& program, const IrValue* data, std::uint32_t components, TransferKernelSite& site) {
    const auto* resolved = data->Resolve();
    std::array<const IrValue*, 4> parts{};
    if (components == 1u) {
        parts[0] = resolved;
    } else {
        if (resolved->Opcode() != Construct(components) || resolved->ArgumentCount() != components) return false;
        for (std::uint32_t component = 0; component < components; ++component) parts[component] = resolved->Argument(component);
    }
    for (std::uint32_t component = 0; component < components; ++component) {
        const auto* value = ThroughBitCasts(parts[component]);
        if (value->HasImmediate() && value->Type() == IrType::U32) {
            site.values[component] = value->ImmediateU32();
            continue;
        }
        if (const auto reg = UserDataRegister(value)) {
            site.values[component] = *reg;
            site.valueRegisters |= 1u << component;
            continue;
        }
        if (value->Opcode() != IrOpcode::ReadConstBuffer || value->ArgumentCount() != 2u || !ConstantU32(value->Argument(1), 0u)) return false;
        const auto descriptor = DescriptorRegister(value->Argument(0));
        const auto& memory = program.Resources().memoryInfo;
        const auto index = value->Flags<MemoryFlags>().index;
        if (!descriptor || index >= memory.size() || memory[index].kind != ResourceKind::ScalarBuffer || memory[index].dataDwords != 1u) return false;
        if (site.valueMemory != 0u && site.valueDescriptor != *descriptor) return false;
        site.valueDescriptor = *descriptor;
        site.values[component] = memory[index].offset;
        site.valueMemory |= 1u << component;
    }
    return true;
}

}

std::optional<TransferKernelSite> FindTransferKernel(const IrProgram& program) {
    const auto& info = program.Info();
    if (!info.linearStore || info.unboundedStores) return std::nullopt;
    const IrValue* store = nullptr;
    for (const auto* block : program.BlockOrder()) {
        for (const auto* instruction : block->Instructions()) {
            if (!WritesMemory(instruction->Opcode())) continue;
            if (store != nullptr) return std::nullopt;
            store = instruction;
        }
    }
    if (store == nullptr || store->ArgumentCount() != 6u) return std::nullopt;
    const auto components = Components(store->Opcode());
    const auto* destinationSite = components == 0u ? nullptr : BufferSite(program, store);
    const auto destination = DescriptorRegister(store->Argument(0));
    if (destinationSite == nullptr || !destination || !ConstantU32(store->Argument(2), 0u) || !ConstantU32(store->Argument(3), 0u) || !AllLanes(store->Argument(5))) return std::nullopt;
    TransferKernelSite site;
    site.components = components;
    site.destination = *destination;
    if (const auto* load = LoadOfAll(store->Argument(4), components)) {
        const auto* sourceSite = BufferSite(program, load);
        const auto source = DescriptorRegister(load->Argument(0));
        if (sourceSite == nullptr || !source || load->ArgumentCount() != 5u || sourceSite->formatted != destinationSite->formatted) return std::nullopt;
        if (load->Argument(1)->Resolve() != store->Argument(1)->Resolve() || !ConstantU32(load->Argument(2), 0u) || !ConstantU32(load->Argument(3), 0u) || !AllLanes(load->Argument(4))) return std::nullopt;
        site.kind = TransferKernelSite::Kind::Copy;
        site.source = *source;
        return site;
    }
    if (!FillValues(program, store->Argument(4), components, site)) return std::nullopt;
    site.kind = TransferKernelSite::Kind::Fill;
    return site;
}

}
