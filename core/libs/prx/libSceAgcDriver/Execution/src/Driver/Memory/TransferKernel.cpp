#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/TransferKernel.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/BufferCopy.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>

#include <cstdlib>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::uint32_t StructuredBounds = 0u;
constexpr std::uint32_t IdentitySelect = 0xfacu;
std::atomic<std::uint64_t> transferFills{0};
std::atomic<std::uint64_t> transferCopies{0};

struct Descriptor {
    std::uint64_t base = 0;
    std::uint32_t stride = 0;
    std::uint32_t records = 0;
    std::uint32_t word3 = 0;
    bool swizzled = false;
};

std::optional<Descriptor> DescriptorAt(std::span<const std::uint32_t> userData, std::uint32_t first) {
    if (first > userData.size() || userData.size() - first < 4u) return std::nullopt;
    const auto words = userData.subspan(first, 4u);
    return Descriptor{words[0] | (static_cast<std::uint64_t>(words[1] & 0xffffu) << 32u), (words[1] >> 16u) & 0x3fffu, words[2], words[3], (words[1] >> 31u) != 0u};
}

bool Contiguous(const Descriptor& descriptor, std::uint32_t elementBytes) {
    const bool addTid = ((descriptor.word3 >> 23u) & 1u) != 0u;
    const auto bounds = (descriptor.word3 >> 28u) & 3u;
    const auto type = descriptor.word3 >> 30u;
    return !descriptor.swizzled && !addTid && type == 0u && bounds == StructuredBounds && descriptor.stride == elementBytes;
}

}

std::optional<TransferRange> ResolveTransfer(const ShaderRecompiler::TransferKernel& transfer, const ShaderRecompiler::LinearStore& destination, std::span<const std::uint32_t> userData, std::uint64_t threads) {
    if (transfer.components == 0u || transfer.components > 4u || threads == 0u) return std::nullopt;
    const auto elementBytes = transfer.components * 4u;
    const auto words = DescriptorAt(userData, transfer.destination);
    if (!words || destination.elementBytes != elementBytes || destination.stride != elementBytes || destination.base != words->base || !Contiguous(*words, elementBytes)) return std::nullopt;
    const auto elements = std::min<std::uint64_t>(threads, std::min<std::uint64_t>(destination.records, words->records));
    TransferRange range;
    range.destination = destination.base;
    range.bytes = static_cast<std::size_t>(elements * elementBytes);
    if (transfer.kind == ShaderRecompiler::TransferKernel::Kind::Copy) {
        const auto source = DescriptorAt(userData, transfer.source);
        if (!source || source->word3 != words->word3 || (source->word3 & 0xfffu) != IdentitySelect || !Contiguous(*source, elementBytes) || source->records < elements) return std::nullopt;
        range.copy = true;
        range.source = source->base;
        return range;
    }
    if (range.destination % 16u != 0u || range.bytes % 16u != 0u) return std::nullopt;
    range.components = transfer.components;
    std::optional<Descriptor> values;
    if (transfer.valueMemory != 0u) {
        values = DescriptorAt(userData, transfer.valueDescriptor);
        if (!values || values->stride != 0u) return std::nullopt;
    }
    for (std::uint32_t component = 0; component < transfer.components; ++component) {
        if ((transfer.valueRegisters >> component) & 1u) {
            if (transfer.values[component] >= userData.size()) return std::nullopt;
            range.pattern[component] = userData[transfer.values[component]];
        } else if ((transfer.valueMemory >> component) & 1u) {
            if (std::uint64_t{transfer.values[component]} + 4u > values->records) return std::nullopt;
            range.reads[component] = values->base + transfer.values[component];
            range.readMask |= 1u << component;
        } else {
            range.pattern[component] = transfer.values[component];
        }
    }
    return range;
}

bool FinishPattern(TransferRange& range) {
    const auto components = range.components;
    if (components == 3u && (range.pattern[1] != range.pattern[0] || range.pattern[2] != range.pattern[0])) return false;
    const auto values = range.pattern;
    for (std::uint32_t word = 0; word < 4u; ++word) range.pattern[word] = components == 3u ? values[0] : values[word % components];
    return true;
}

bool Driver::transferBuffer(std::uint32_t queueId, const ShaderRecompiler::RecompileResult& compiled, std::span<const std::uint32_t> userData, std::uint64_t threads, const std::shared_ptr<VulkanDevice>& localDevice, std::uint64_t programAddress) {
    static const bool disabled = std::getenv("APS5_NO_TRANSFER_KERNELS") != nullptr;
    if (disabled || compiled.rangesFromMemory || !compiled.transferKernel || !compiled.linearStore) return false;
    const auto range = ResolveTransfer(*compiled.transferKernel, *compiled.linearStore, userData, threads);
    if (!range) return false;
    if (range->bytes == 0u) return true;
    if (!range->copy) {
        auto fill = *range;
        std::unique_lock inputLock(GuestMemory::GpuMutex(), std::defer_lock);
        if (fill.readMask != 0u) {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Fill);
            inputLock.lock();
            recordLabelsForPacket(localDevice.get(), queueId);
            for (std::uint32_t component = 0; component < fill.components; ++component) {
                if (((fill.readMask >> component) & 1u) == 0u) continue;
                if (!GuestMemory::Accessible(reinterpret_cast<const void*>(fill.reads[component]), 4)) return false;
                GuestMemory::Read(fill.reads[component], std::as_writable_bytes(std::span(fill.pattern).subspan(component, 1)), 4);
            }
        }
        if (!FinishPattern(fill)) return false;
        storeBufferFill(queueId, fill.destination, fill.bytes, fill.pattern, localDevice, false, true);
        transferFills.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    const auto started = std::chrono::steady_clock::now();
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Copy);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    const auto locked = std::chrono::steady_clock::now();
    recordLabelsForPacket(localDevice.get(), queueId);
    if (queueId == 0) localDevice->ReapRecorded();
    if (!copyRange(queueId, range->source, range->destination, range->bytes, localDevice, programAddress, started, locked)) return false;
    transferCopies.fetch_add(1, std::memory_order_relaxed);
    return true;
}

std::uint64_t Driver::TransferFills() {
    return transferFills.load(std::memory_order_relaxed);
}

std::uint64_t Driver::TransferCopies() {
    return transferCopies.load(std::memory_order_relaxed);
}

}
