#ifndef CORE_SHADER_RECOMPILER_COMPILEDVARIANT_HPP
#define CORE_SHADER_RECOMPILER_COMPILEDVARIANT_HPP

#include "Recompiler.hpp"
#include "IntermediateRepresentation/IrMetadata/CompiledShaderInfo.hpp"
#include "Optimization/include/Optimization/BindingAllocator.hpp"
#include "Optimization/include/Optimization/DescriptorBindingBuilder.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

class BufferSubstituteTable {
public:
    using Persist = std::function<void(const std::vector<BufferSubstitute>&)>;

    BufferSubstituteTable(std::vector<BufferSubstitute> slots, Persist persist) : slots(std::move(slots)), persist(std::move(persist)) {}

    [[nodiscard]] std::vector<BufferSubstitute> Slots() const {
        std::lock_guard lock(mutex);
        return slots;
    }

    void Learn(std::size_t slot, const BufferSubstitute& value) {
        std::vector<BufferSubstitute> learned;
        {
            std::lock_guard lock(mutex);
            if (slot >= slots.size() || slots[slot].known) return;
            slots[slot] = value;
            slots[slot].known = true;
            if (!persist) return;
            learned = slots;
        }
        persist(learned);
    }

private:
    mutable std::mutex mutex;
    std::vector<BufferSubstitute> slots;
    Persist persist;
};

struct CompiledVariant {
    BindingLayout layout;
    CompiledShaderInfo info;
    CompiledBindingLayout bindings;
    CompiledShaderArtifact artifact;
    std::shared_ptr<BufferSubstituteTable> substitutes;
};

}

#endif
