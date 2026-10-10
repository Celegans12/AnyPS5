#include "Recompiler.hpp"
#include "ColdSpirvCodec.hpp"
#include <atomic>
#include <mutex>

namespace ShaderRecompiler {

struct SharedSpirv::Storage {
    explicit Storage(std::vector<std::uint32_t> value) : words(std::make_shared<std::vector<std::uint32_t>>(std::move(value))) {}

    void Hydrate() {
        if (words != nullptr) return;
        auto loaded = std::make_shared<std::vector<std::uint32_t>>(ColdSpirv::Decompress(compressed, wordCount));
        words = std::move(loaded);
        std::vector<std::byte>().swap(compressed);
    }

    mutable std::mutex mutex;
    std::shared_ptr<std::vector<std::uint32_t>> words;
    std::vector<std::byte> compressed;
    std::size_t wordCount = 0;
    std::atomic<bool> exposed{false};
    bool attempted = false;
};

SharedSpirv::SharedSpirv(std::vector<std::uint32_t> words) : storage(std::make_shared<Storage>(std::move(words))) {}

SharedSpirv& SharedSpirv::operator=(std::vector<std::uint32_t> other) {
    storage = std::make_shared<Storage>(std::move(other));
    return *this;
}

const std::vector<std::uint32_t>& SharedSpirv::Words() const {
    if (storage == nullptr) return Empty();
    if (storage->exposed.load(std::memory_order_acquire)) return *storage->words;
    std::lock_guard lock(storage->mutex);
    storage->Hydrate();
    storage->exposed.store(true, std::memory_order_release);
    return *storage->words;
}

std::shared_ptr<const std::vector<std::uint32_t>> SharedSpirv::ReadTransient() const {
    if (storage == nullptr) {
        static const auto empty = std::make_shared<const std::vector<std::uint32_t>>();
        return empty;
    }
    std::lock_guard lock(storage->mutex);
    if (storage->words != nullptr) return storage->words;
    return std::make_shared<const std::vector<std::uint32_t>>(ColdSpirv::Decompress(storage->compressed, storage->wordCount));
}

bool SharedSpirv::TryMakeCold() noexcept {
    try {
        if (storage == nullptr || storage.use_count() != 1) return false;
        std::lock_guard lock(storage->mutex);
        if (storage->exposed.load(std::memory_order_relaxed) || storage->attempted || storage->words == nullptr || storage->words.use_count() != 1) return false;
        storage->attempted = true;
        if (storage->words->size() < 16u * 1024u || storage->words->size() > ColdSpirv::MaxBytes / sizeof(std::uint32_t)) return false;
        auto compressed = ColdSpirv::Compress(*storage->words);
        if (!compressed || compressed->size() >= storage->words->size() * sizeof(std::uint32_t)) return false;
        storage->wordCount = storage->words->size();
        storage->compressed = std::move(*compressed);
        storage->words.reset();
        return true;
    } catch (...) {
        return false;
    }
}

std::size_t SharedSpirv::ColdBytes() const {
    if (storage == nullptr) return 0;
    if (storage->exposed.load(std::memory_order_acquire)) return 0;
    std::lock_guard lock(storage->mutex);
    return storage->compressed.size();
}

SharedSpirv::StorageInfo SharedSpirv::InspectStorage() const {
    if (storage == nullptr) return {};
    std::lock_guard lock(storage->mutex);
    StorageInfo info;
    info.logicalBytes = (storage->words != nullptr ? storage->words->size() : storage->wordCount) * sizeof(std::uint32_t);
    if (storage->words != nullptr) {
        info.residentBytes = storage->words->size() * sizeof(std::uint32_t);
        info.residentCapacityBytes = storage->words->capacity() * sizeof(std::uint32_t);
    }
    info.coldBytes = storage->compressed.size();
    info.coldCapacityBytes = storage->compressed.capacity();
    info.exposed = storage->exposed.load(std::memory_order_relaxed);
    info.eligible = storage.use_count() == 1 && storage->words != nullptr && storage->words.use_count() == 1 && !info.exposed && !storage->attempted && info.logicalBytes >= 64u * 1024u && info.logicalBytes <= ColdSpirv::MaxBytes;
    return info;
}

std::size_t SharedSpirv::size() const {
    if (storage == nullptr) return 0;
    if (storage->exposed.load(std::memory_order_acquire)) return storage->words->size();
    std::lock_guard lock(storage->mutex);
    return storage->words != nullptr ? storage->words->size() : storage->wordCount;
}

std::vector<std::uint32_t>& SharedSpirv::Mutable() {
    if (storage == nullptr || storage.use_count() != 1) storage = std::make_shared<Storage>(*ReadTransient());
    std::lock_guard lock(storage->mutex);
    storage->Hydrate();
    if (storage->words.use_count() != 1) storage->words = std::make_shared<std::vector<std::uint32_t>>(*storage->words);
    storage->exposed.store(true, std::memory_order_release);
    return *storage->words;
}

std::vector<std::uint32_t>::iterator SharedSpirv::insert(std::vector<std::uint32_t>::const_iterator where, std::initializer_list<std::uint32_t> values) {
    const auto offset = where - Words().cbegin();
    auto& words = Mutable();
    return words.insert(words.cbegin() + offset, values);
}

}
