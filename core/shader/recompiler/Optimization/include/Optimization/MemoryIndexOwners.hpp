#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_MEMORYINDEXOWNERS_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_MEMORYINDEXOWNERS_HPP

#include "IntermediateRepresentation/IrProgram.hpp"

#include <cstdint>
#include <vector>

namespace ShaderRecompiler::Detail {

class MemoryIndexOwners {
public:
    explicit MemoryIndexOwners(const IrProgram& program) : m_entries(program.Resources().memoryInfo.size()) {
        for (const auto& block : program.Blocks()) {
            for (const IrValue* inst : block->Instructions()) {
                const auto op = inst->Opcode();
                if (BufferAccessOf(op) == BufferAccess::None && AddressOpcodeInfoOf(op).access == AddressAccess::None && ImageOpcodeInfoOf(op).access == ImageAccess::None) {
                    continue;
                }
                const auto index = inst->Flags<MemoryFlags>().index;
                if (index >= m_entries.size()) {
                    continue;
                }
                auto& entry = m_entries[index];
                if (entry.owner == nullptr) {
                    entry.owner = inst;
                } else if (entry.owner != inst) {
                    entry.shared = true;
                }
            }
        }
    }

    [[nodiscard]] bool BelongsTo(std::uint32_t index, const IrValue& owner) const {
        const auto& entry = m_entries.at(index);
        return !entry.shared && (entry.owner == nullptr || entry.owner == &owner);
    }

private:
    struct Entry {
        const IrValue* owner = nullptr;
        bool shared = false;
    };
    std::vector<Entry> m_entries;
};

}

#endif
