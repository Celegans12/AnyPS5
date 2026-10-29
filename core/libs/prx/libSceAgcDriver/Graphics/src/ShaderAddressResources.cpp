#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <unordered_set>
#include <utility>
#include "UnnormalizedWrap.hpp"
#include "DirectImageRefusal.hpp"

namespace AgcDriver::Graphics {

namespace {

std::uint32_t pointerOperandStored(std::uint32_t opcode, std::span<const std::uint32_t> instruction) {
    switch (static_cast<spv::Op>(opcode)) {
    case spv::OpStore:
    case spv::OpCopyMemory:
    case spv::OpCopyMemorySized:
    case spv::OpAtomicStore:
    case spv::OpAtomicFlagClear:
    case spv::OpCooperativeMatrixStoreKHR:
    case spv::OpCooperativeMatrixStoreNV:
    case spv::OpCooperativeMatrixStoreTensorNV:
        return instruction.size() > 1 ? instruction[1] : 0u;
    case spv::OpAtomicExchange:
    case spv::OpAtomicCompareExchange:
    case spv::OpAtomicCompareExchangeWeak:
    case spv::OpAtomicIIncrement:
    case spv::OpAtomicIDecrement:
    case spv::OpAtomicIAdd:
    case spv::OpAtomicISub:
    case spv::OpAtomicSMin:
    case spv::OpAtomicUMin:
    case spv::OpAtomicSMax:
    case spv::OpAtomicUMax:
    case spv::OpAtomicAnd:
    case spv::OpAtomicOr:
    case spv::OpAtomicXor:
    case spv::OpAtomicFlagTestAndSet:
    case spv::OpAtomicFMinEXT:
    case spv::OpAtomicFMaxEXT:
    case spv::OpAtomicFAddEXT:
        return instruction.size() > 3 ? instruction[3] : 0u;
    default:
        return 0u;
    }
}

bool definesNoPointerValue(std::uint32_t opcode) {
    switch (static_cast<spv::Op>(opcode)) {
    case spv::OpTypePointer:
    case spv::OpTypeUntypedPointerKHR:
    case spv::OpTypeForwardPointer:
    case spv::OpName:
    case spv::OpMemberName:
    case spv::OpDecorate:
    case spv::OpMemberDecorate:
    case spv::OpDecorateId:
    case spv::OpDecorateString:
    case spv::OpMemberDecorateString:
        return true;
    default:
        return false;
    }
}

template<typename TVisit>
bool forEachInstruction(std::span<const std::uint32_t> words, const TVisit& visit) {
    for (std::size_t at = 5; at < words.size();) {
        const auto count = words[at] >> 16u;
        if (count == 0 || at + count > words.size()) return false;
        visit(words[at] & 0xffffu, words.subspan(at, count));
        at += count;
    }
    return true;
}

bool storesThroughAddressesCached(const ShaderRecompiler::SharedSpirv& spirv) {
    static std::mutex mutex;
    static std::map<std::pair<std::uint64_t, std::uint64_t>, bool> known;
    const auto hash = spirv.ContentHash();
    const std::pair key{hash.low, hash.high};
    {
        std::lock_guard lock(mutex);
        if (const auto found = known.find(key); found != known.end()) return found->second;
    }
    const bool stores = SpirvStoresThroughAddresses(spirv.Words());
    std::lock_guard lock(mutex);
    known.emplace(key, stores);
    return stores;
}

}

bool ShaderStoresThroughAddresses(const ShaderRecompiler::SharedSpirv& spirv) {
    return storesThroughAddressesCached(spirv);
}

bool SpirvStoresThroughAddresses(std::span<const std::uint32_t> words) {
    if (words.size() < 5 || words[0] != spv::MagicNumber) return true;
    std::unordered_set<std::uint32_t> pointerTypes;
    const bool readable = forEachInstruction(words, [&](std::uint32_t opcode, std::span<const std::uint32_t> instruction) {
        if ((opcode == spv::OpTypePointer || opcode == spv::OpTypeUntypedPointerKHR) && instruction.size() > 2 && instruction[2] == spv::StorageClassPhysicalStorageBuffer) pointerTypes.insert(instruction[1]);
    });
    if (!readable) return true;
    if (pointerTypes.empty()) return false;
    std::unordered_set<std::uint32_t> pointers;
    forEachInstruction(words, [&](std::uint32_t opcode, std::span<const std::uint32_t> instruction) {
        if (instruction.size() > 2 && !definesNoPointerValue(opcode) && pointerTypes.contains(instruction[1])) pointers.insert(instruction[2]);
    });
    bool stores = false;
    forEachInstruction(words, [&](std::uint32_t opcode, std::span<const std::uint32_t> instruction) {
        const auto pointer = pointerOperandStored(opcode, instruction);
        const bool extendedWithPointer = opcode == spv::OpExtInst && std::any_of(instruction.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(5u, instruction.size())), instruction.end(), [&](std::uint32_t operand) { return pointers.contains(operand); });
        stores = stores || extendedWithPointer || (pointer != 0 && pointers.contains(pointer));
    });
    return stores;
}

namespace {

bool storesThroughPlainInstructions(std::span<const std::uint32_t> words) {
    if (words.size() < 5 || words[0] != spv::MagicNumber) return true;
    std::unordered_set<std::uint32_t> pointerTypes;
    forEachInstruction(words, [&](std::uint32_t opcode, std::span<const std::uint32_t> instruction) {
        if ((opcode == spv::OpTypePointer || opcode == spv::OpTypeUntypedPointerKHR) && instruction.size() > 2 && instruction[2] == spv::StorageClassPhysicalStorageBuffer) pointerTypes.insert(instruction[1]);
    });
    if (pointerTypes.empty()) return false;
    std::unordered_set<std::uint32_t> pointers;
    forEachInstruction(words, [&](std::uint32_t opcode, std::span<const std::uint32_t> instruction) {
        if (instruction.size() > 2 && !definesNoPointerValue(opcode) && pointerTypes.contains(instruction[1])) pointers.insert(instruction[2]);
    });
    bool stores = false;
    forEachInstruction(words, [&](std::uint32_t opcode, std::span<const std::uint32_t> instruction) {
        const auto pointer = pointerOperandStored(opcode, instruction);
        stores = stores || (pointer != 0 && pointers.contains(pointer));
    });
    return stores;
}

bool storesOnlyThroughExtInstCached(const ShaderRecompiler::SharedSpirv& spirv) {
    static std::mutex mutex;
    static std::map<std::pair<std::uint64_t, std::uint64_t>, bool> known;
    const auto hash = spirv.ContentHash();
    const std::pair key{hash.low, hash.high};
    {
        std::lock_guard lock(mutex);
        if (const auto found = known.find(key); found != known.end()) return found->second;
    }
    const bool extOnly = storesThroughAddressesCached(spirv) && !storesThroughPlainInstructions(spirv.Words());
    std::lock_guard lock(mutex);
    known.emplace(key, extOnly);
    return extOnly;
}

}

bool ShaderResources::StagesStoreOnlyThroughExtInst(std::span<const CompiledShader> shaders) {
    bool any = false;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr || !std::any_of(shader.program->bindings.begin(), shader.program->bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; })) continue;
        if (!storesThroughAddressesCached(shader.program->spirv)) continue;
        if (!storesOnlyThroughExtInstCached(shader.program->spirv)) return false;
        any = true;
    }
    return any;
}

bool ShaderResources::StagesStoreThroughAddresses(std::span<const CompiledShader> shaders) {
    return std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) {
        return shader.program != nullptr && std::any_of(shader.program->bindings.begin(), shader.program->bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; }) && storesThroughAddressesCached(shader.program->spirv);
    });
}

void ShaderResources::prepareAddressBindings(std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots) {
    static const bool loopGuarded = std::getenv("APS5_LOOP_GUARD") != nullptr;
    for (const auto& shader : shaders) {
        Require(shader.program != nullptr, "missing compiled shader");
        std::uint32_t tables = 0;
        std::uint32_t faults = 0;
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::BdaPagetable && binding.role != ShaderRecompiler::DescriptorRole::FaultBuffer) continue;
            Require(binding.kind == ShaderRecompiler::DescriptorKind::StorageBuffer && binding.count == 1 && binding.guestDescriptor.empty() && !binding.readOnly, "invalid BDA descriptor contract");
            if (binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable) ++tables;
            else ++faults;
        }
        const bool faultOnly = tables == 0 && faults == 1 && shader.program->bdaAbiVersion == ShaderRecompiler::BdaAbi::Version;
        Require((tables == faults || faultOnly) && tables <= 1 && faults <= 1, "invalid BDA table and fault descriptors");
        Require(shader.program->bdaAbiVersion == (faults == 0 ? 0u : ShaderRecompiler::BdaAbi::Version), "incompatible BDA ABI version");
        const bool hasDeferredDirectImage = shader.program->hasDeferredDirectImage;
        Require(!hasDeferredDirectImage || (shader.stage == ShaderRecompiler::ShaderStage::Compute && faults == 1u), "deferred direct image requires a compute fault buffer");
        bool guardedWrap = false;
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestSamplers) continue;
            for (std::size_t element = 0; element < binding.samplerUnnormalized.size(); ++element) {
                if (!binding.samplerUnnormalized[element]) continue;
                Require(binding.guestDescriptor.size() >= (element + 1u) * 4u, "guest sampler binding has incomplete descriptor words");
                guardedWrap |= ShaderRecompiler::UnnormalizedWrapAxes(std::span<const std::uint32_t>(binding.guestDescriptor).subspan(element * 4u, 4u)) != 0u;
            }
        }
        if (ShaderRecompiler::UnnormalizedWrapEnabled() && guardedWrap) Require(shader.program->hasGuardedUnnormalizedWrap && shader.stage == ShaderRecompiler::ShaderStage::Compute && faults == 1u && !ShaderRecompiler::DirectImageLoopGuardEnabled(), "unnormalized wrap requires verified guard proof and a compute fault buffer without a loop guard");
        // Rect-list validation needs a fault buffer, but never accesses guest addresses.
        usesBda = usesBda || tables != 0;
        if (tables != 0 && storesThroughAddressesCached(shader.program->spirv)) bdaReadOnly = false;
        usesFaultBuffer = usesFaultBuffer || faults != 0;
        const bool tableFaultsOnly = std::any_of(shader.program->bindings.begin(), shader.program->bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::ImageTableMap; });
        faultChecks = faultChecks || tables != 0 || (faults != 0 && (!tableFaultsOnly || shader.program->imageTableFaults != 0 || shader.program->poisonedSrtReads != 0 || shader.program->windowLaneGuards != 0 || loopGuarded || hasDeferredDirectImage || guardedWrap));
    }
    if (usesBda) {
        Require(context.bufferDeviceAddress, "buffer device address is not enabled");
        static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
        guestMemory.AcquireRegistered();
        const auto snapshotsStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (addressTableDeferred) guestMemory.SkipLiveSnapshotCompares();
        for (const auto& snapshot : snapshots) guestMemory.AddSnapshot(snapshot);
        if (profile) GuestBufferMemory::CountAddressBuild(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - snapshotsStart).count());
    }
}

std::shared_ptr<const std::vector<ShaderRecompiler::BdaAbi::Range>> ShaderResources::AddressTableRanges() const {
    if (addressTableDeferred && usesBda) {
        if (deferredRanges == nullptr) deferredRanges = BdaResources::RangesOf(guestMemory);
        return deferredRanges;
    }
    return bda != nullptr && usesBda ? bda->Ranges() : nullptr;
}

std::shared_ptr<const std::vector<ShaderRecompiler::BdaAbi::Range>> ShaderResources::GuestBufferRanges() const {
    if (usesBda) return nullptr;
    return std::make_shared<const std::vector<ShaderRecompiler::BdaAbi::Range>>(guestMemory.AddressRanges());
}

VkDescriptorBufferInfo ShaderResources::descriptor(Allocation& allocation) {
    if (allocation.guest) return guestMemory.Descriptor(allocation.address, allocation.size, allocation.adjustment);
    if (allocation.role == ShaderRecompiler::DescriptorRole::GuestBuffers) {
        Require(context.emptyBuffer != VK_NULL_HANDLE, "no placeholder buffer for a null V#");
        return {context.emptyBuffer, 0, allocation.size};
    }
    if (allocation.role == ShaderRecompiler::DescriptorRole::BdaPagetable || allocation.role == ShaderRecompiler::DescriptorRole::FaultBuffer) {
        if (bda == nullptr && addressTableDeferred) {
            Require(context.emptyBuffer != VK_NULL_HANDLE, "no placeholder buffer for a deferred address table");
            return {context.emptyBuffer, 0, VK_WHOLE_SIZE};
        }
        Require(bda != nullptr, "BDA descriptors have no memory owner");
        return allocation.role == ShaderRecompiler::DescriptorRole::BdaPagetable ? bda->Table() : bda->Fault();
    }
    Require(allocation.buffer != nullptr, "shader data has no buffer owner");
    return {allocation.buffer->Handle(), 0, allocation.size};
}

}
