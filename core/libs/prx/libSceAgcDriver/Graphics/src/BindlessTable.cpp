#include "prx/libSceAgcDriver/Graphics/include/BindlessTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "DeviceImageTableAbi.hpp"
#include <array>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string_view>
#include <utility>

namespace AgcDriver::Graphics {

namespace {

namespace Abi = ShaderRecompiler::DeviceImageTableAbi;

constexpr VkBufferUsageFlags DeviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
constexpr std::size_t RingWords = Abi::RingRecordsWord + static_cast<std::size_t>(BindlessTable::RingCapacity) * Abi::RecordWords;
constexpr std::size_t UpdateBufferBytes = 65536u;

}

bool BindlessImagesRequested() {
    const char* value = std::getenv("APS5_BINDLESS_IMAGES");
    return value != nullptr && std::string_view(value) == "1";
}

std::optional<std::string> BindlessUnsupported(const BindlessFeatureSupport& support) {
    const std::array<std::pair<bool, const char*>, 9> required{{
        {support.bufferDeviceAddress, "bufferDeviceAddress"},
        {support.nullDescriptor, "robustness2 nullDescriptor"},
        {support.runtimeDescriptorArray, "runtimeDescriptorArray"},
        {support.partiallyBound, "descriptorBindingPartiallyBound"},
        {support.sampledNonUniform, "shaderSampledImageArrayNonUniformIndexing"},
        {support.storageNonUniform, "shaderStorageImageArrayNonUniformIndexing"},
        {support.sampledUpdateAfterBind, "descriptorBindingSampledImageUpdateAfterBind"},
        {support.storageUpdateAfterBind, "descriptorBindingStorageImageUpdateAfterBind"},
        {support.updateUnusedWhilePending, "descriptorBindingUpdateUnusedWhilePending"},
    }};
    std::string missing;
    for (const auto& [present, name] : required) {
        if (present) continue;
        if (!missing.empty()) missing += ", ";
        missing += name;
    }
    if (missing.empty()) return std::nullopt;
    return missing;
}

BindlessTable::BindlessTable(const Context& context, DescriptorHeap::Capacities capacities, std::uint32_t buckets, std::uint32_t maxBuckets) : context(context), heap(context, capacities), keys(buckets, maxBuckets), rings(std::make_shared<Rings>()) {
    Require(context.bufferDeviceAddress, "the bindless image table needs buffer device addresses");
    header = std::make_shared<Buffer>(context, Abi::HeaderWords * sizeof(std::uint32_t), DeviceUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

BindlessTable::~BindlessTable() = default;

std::uint64_t BindlessTable::HeaderAddress() const {
    return header->DeviceAddress();
}

DeviceImageKeys::Insertion BindlessTable::Insert(const Key& key, const DeviceImageValue& value) {
    std::lock_guard lock(mutex);
    return keys.Insert(key, value);
}

std::optional<DeviceImageValue> BindlessTable::Find(const Key& key) const {
    std::lock_guard lock(mutex);
    return keys.Find(key);
}

std::uint32_t BindlessTable::Retire(const std::function<bool(const Key&, const DeviceImageValue&)>& drop) {
    std::lock_guard lock(mutex);
    const auto dropped = keys.Retain([&](const Key& key) {
        const auto value = keys.Find(key);
        Require(value.has_value(), "a live bindless key was not found in its own table");
        return !drop(key, *value);
    });
    totals.retired += dropped;
    return dropped;
}

void BindlessTable::Publish(Recorder& recorder) {
    std::lock_guard lock(mutex);
    heap.Collect(recorder.CompletedThrough());
    if (recorder.HasQueuedKeyStores()) recorder.FlushKeyStores();
    const auto commands = recorder.Commands();
    const auto batch = recorder.OpenSerial();
    const bool assign = ring == nullptr || ringBatch != batch;
    const bool grown = table == nullptr || keys.Buckets() != tableBuckets;
    const bool rebuilt = grown || keys.Rebuilt();
    const auto changed = keys.TakeChanged();
    ++totals.publications;
    if (!assign && !rebuilt && changed.empty()) return;
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    const auto words = keys.Words();
    const auto bytes = words.size() * sizeof(std::uint32_t);
    if (grown) {
        if (table != nullptr) recorder.Keep(table);
        table = std::make_shared<Buffer>(context, bytes, DeviceUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        tableBuckets = keys.Buckets();
    }
    if (rebuilt) {
        auto staging = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memcpy(staging->Bytes().data(), words.data(), bytes);
        const VkBufferCopy region{0, 0, bytes};
        context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer")(commands, staging->Handle(), table->Handle(), 1, &region);
        recorder.Keep(std::move(staging));
        ++totals.uploads;
    } else {
        for (std::size_t first = 0; first < changed.size();) {
            auto last = first;
            while (last + 1u < changed.size() && changed[last + 1u] == changed[last] + 1u && (last + 2u - first) * Abi::BucketBytes <= UpdateBufferBytes) ++last;
            const auto offset = static_cast<VkDeviceSize>(changed[first]) * Abi::BucketBytes;
            context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer")(commands, table->Handle(), offset, static_cast<VkDeviceSize>(last + 1u - first) * Abi::BucketBytes, words.data() + static_cast<std::size_t>(changed[first]) * Abi::BucketWords);
            totals.bucketUpdates += last + 1u - first;
            first = last + 1u;
        }
    }
    if (assign || grown) {
        if (assign) {
            {
                std::lock_guard ringLock(rings->mutex);
                if (!rings->free.empty()) {
                    ring = std::move(rings->free.back());
                    rings->free.pop_back();
                } else {
                    ring = std::make_shared<Buffer>(context, RingWords * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
                }
            }
            std::memset(ring->Bytes().data(), 0, Abi::RingRecordsWord * sizeof(std::uint32_t));
            ringBatch = batch;
            ++totals.rings;
            recorder.OnComplete([shared = rings, held = ring] { drain(shared, held); });
            recorder.OweHostRead();
        }
        std::array<std::uint32_t, Abi::HeaderWords> fields{};
        const auto tableAddress = table->DeviceAddress();
        const auto ringAddress = ring->DeviceAddress();
        fields[Abi::HeaderVersion] = Abi::Version;
        fields[Abi::HeaderTableLow] = static_cast<std::uint32_t>(tableAddress);
        fields[Abi::HeaderTableHigh] = static_cast<std::uint32_t>(tableAddress >> 32u);
        fields[Abi::HeaderMask] = keys.Buckets() - 1u;
        fields[Abi::HeaderRingLow] = static_cast<std::uint32_t>(ringAddress);
        fields[Abi::HeaderRingHigh] = static_cast<std::uint32_t>(ringAddress >> 32u);
        fields[Abi::HeaderRingCapacity] = RingCapacity;
        context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer")(commands, header->Handle(), 0, sizeof(fields), fields.data());
    }
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void BindlessTable::drain(const std::shared_ptr<Rings>& rings, const std::shared_ptr<Buffer>& drained) {
    drained->Invalidate();
    const auto bytes = drained->Bytes();
    const auto decoded = DecodeDeviceImageRing(std::span<const std::uint32_t>(reinterpret_cast<const std::uint32_t*>(bytes.data()), RingWords), RingCapacity);
    std::lock_guard lock(rings->mutex);
    ++rings->drained;
    rings->written += decoded.written;
    rings->dropped += decoded.dropped;
    rings->misses.insert(rings->misses.end(), decoded.misses.begin(), decoded.misses.end());
    rings->free.push_back(drained);
}

std::vector<DeviceImageMiss> BindlessTable::TakeMisses() {
    std::lock_guard lock(rings->mutex);
    const auto misses = std::exchange(rings->misses, {});
    return DistinctDeviceImageMisses(misses);
}

BindlessTable::Counts BindlessTable::Totals() const {
    std::lock_guard lock(mutex);
    auto counts = totals;
    {
        std::lock_guard ringLock(rings->mutex);
        counts.drained = rings->drained;
        counts.misses = rings->written;
        counts.dropped = rings->dropped;
    }
    counts.entries = keys.Entries();
    counts.buckets = keys.Buckets();
    return counts;
}

}
