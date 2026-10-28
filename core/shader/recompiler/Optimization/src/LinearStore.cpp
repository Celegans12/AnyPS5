#include "Optimization/LinearStore.hpp"
#include "IntermediateRepresentation/IrMetadata/StageIO.hpp"
#include <algorithm>
#include <bit>

namespace ShaderRecompiler {

namespace {

std::uint32_t StoreComponents(IrOpcode opcode) {
    switch (opcode) {
        case IrOpcode::StoreBufferU8:
        case IrOpcode::StoreBufferU16:
        case IrOpcode::StoreBufferU32:
            return 1u;
        case IrOpcode::StoreBufferU32x2:
            return 2u;
        case IrOpcode::StoreBufferU32x3:
            return 3u;
        case IrOpcode::StoreBufferU32x4:
            return 4u;
        default:
            return 0u;
    }
}

bool Constant(const IrValue* value, std::uint32_t expected) {
    const auto* resolved = value->Resolve();
    return resolved->HasImmediate() && resolved->Type() == IrType::U32 && resolved->ImmediateU32() == expected;
}

bool Builtin(const IrValue* value, StageInputKind kind) {
    const auto* resolved = value->Resolve();
    return resolved->Opcode() == IrOpcode::GetBuiltin && resolved->ArgumentCount() == 2u && Constant(resolved->Argument(0), static_cast<std::uint32_t>(kind)) && Constant(resolved->Argument(1), 0u);
}

bool GroupBase(const IrValue* value, std::uint32_t threads) {
    const auto* resolved = value->Resolve();
    if (resolved->ArgumentCount() != 2u) return false;
    const auto* left = resolved->Argument(0);
    const auto* right = resolved->Argument(1);
    if (resolved->Opcode() == IrOpcode::ShiftLeftLogical32) return std::has_single_bit(threads) && Builtin(left, StageInputKind::WorkgroupId) && Constant(right, static_cast<std::uint32_t>(std::countr_zero(threads)));
    if (resolved->Opcode() == IrOpcode::IMul32) return (Builtin(left, StageInputKind::WorkgroupId) && Constant(right, threads)) || (Constant(left, threads) && Builtin(right, StageInputKind::WorkgroupId));
    return false;
}

bool GlobalIndex(const IrValue* value, std::uint32_t threads) {
    const auto* resolved = value->Resolve();
    if (resolved->Opcode() != IrOpcode::IAdd32 || resolved->ArgumentCount() != 2u) return false;
    const auto* left = resolved->Argument(0);
    const auto* right = resolved->Argument(1);
    return (GroupBase(left, threads) && Builtin(right, StageInputKind::LocalInvocationId)) || (Builtin(left, StageInputKind::LocalInvocationId) && GroupBase(right, threads));
}

bool StraightLine(const IrProgram& program) {
    const auto& blocks = program.BlockOrder();
    // fable also refuses programs with guest call functions; this IR has none (no callFunctions metadata).
    if (blocks.empty()) return false;
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        const auto& successors = blocks[index]->Successors();
        const auto& predecessors = blocks[index]->Predecessors();
        const bool last = index + 1u == blocks.size();
        if (successors.size() != (last ? 0u : 1u) || (!last && successors.front() != blocks[index + 1u])) return false;
        if (predecessors.size() != (index == 0u ? 0u : 1u)) return false;
    }
    return true;
}

bool WritesMemory(IrOpcode opcode) {
    const auto buffer = BufferAccessOf(opcode);
    const auto address = AddressOpcodeInfoOf(opcode).access;
    const auto image = ImageOpcodeInfoOf(opcode).access;
    return buffer == BufferAccess::Write || buffer == BufferAccess::Atomic || address == AddressAccess::Write || address == AddressAccess::Atomic || image == ImageAccess::Write || image == ImageAccess::Atomic || opcode == IrOpcode::DataAppend || opcode == IrOpcode::DataConsume;
}

}

std::optional<LinearStoreSite> FindLinearStore(const IrProgram& program, const std::array<std::uint32_t, 3>& threads) {
    const auto& info = program.Info();
    if (program.Resources().stage != IrShaderStage::Compute || threads[0] == 0u || threads[1] != 1u || threads[2] != 1u) return std::nullopt;
    if (info.bufferStores.size() != 1u || info.unboundedStores || info.bufferStores.front().fromMemory || !StraightLine(program)) return std::nullopt;
    // The driver re-reads the live user-data V# to check the linear store, so no site may come from guest memory.
    if (std::ranges::any_of(info.bufferLoads, [](const BufferStoreSite& site) { return site.fromMemory; })) return std::nullopt;
    const IrValue* store = nullptr;
    for (const auto* block : program.BlockOrder()) {
        for (const auto* instruction : block->Instructions()) {
            if (!WritesMemory(instruction->Opcode())) continue;
            if (store != nullptr || StoreComponents(instruction->Opcode()) == 0u) return std::nullopt;
            store = instruction;
        }
    }
    if (store == nullptr || store->ArgumentCount() != 6u) return std::nullopt;
    const auto& memory = program.Resources().memoryInfo;
    const auto index = store->Flags<MemoryFlags>().index;
    if (index >= memory.size()) return std::nullopt;
    const auto& site = memory[index];
    if (site.kind != ResourceKind::Buffer || site.offset != 0u || site.typed || (!site.formatted && site.dataBits == 0u)) return std::nullopt;
    const auto* handle = store->Argument(0)->Resolve();
    if (handle->Opcode() != IrOpcode::GetBufferResource || handle->ArgumentCount() != 4u) return std::nullopt;
    for (std::size_t word = 0; word < 4u; ++word) {
        if (handle->Argument(word)->Resolve()->Opcode() != IrOpcode::GetUserData) return std::nullopt;
    }
    const auto* active = store->Argument(5)->Resolve();
    if (!GlobalIndex(store->Argument(1), threads[0]) || !Constant(store->Argument(2), 0u) || !Constant(store->Argument(3), 0u) || !active->HasImmediate() || active->Type() != IrType::Bool || !active->ImmediateBool()) return std::nullopt;
    const auto components = StoreComponents(store->Opcode());
    LinearStoreSite linear;
    linear.components = components;
    linear.descriptorFormat = site.formatted;
    linear.elementBytes = site.formatted ? 0u : components * site.dataBits / 8u;
    return linear;
}

}
