#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PipelineAhead.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <system_error>

namespace AgcDriver::NewPath {

namespace {

constexpr const char* ManifestMagic = "aps5-newpath-pipelines";

}

PipelineManifest::PipelineManifest(std::filesystem::path path, std::uint64_t sourceVersion) : path(std::move(path)), sourceVersion(sourceVersion) {
    load();
}

void PipelineManifest::load() {
    if (path.empty()) return;
    std::ifstream in(path);
    if (!in) return;
    std::string magic;
    std::uint64_t version = 0;
    in >> magic >> std::hex >> version;
    if (magic != ManifestMagic || version != sourceVersion) return;
    PipelineKey key = 0;
    while (in >> std::hex >> key) keys.insert(key);
    loaded = true;
}

bool PipelineManifest::Known(PipelineKey key) const {
    std::lock_guard lock(mutex);
    return keys.contains(key);
}

std::vector<PipelineKey> PipelineManifest::Keys() const {
    std::lock_guard lock(mutex);
    return {keys.begin(), keys.end()};
}

bool PipelineManifest::Record(PipelineKey key) {
    std::lock_guard lock(mutex);
    if (keys.contains(key) || path.empty()) return false;
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    const bool fresh = !std::filesystem::exists(path, error) || !loaded;
    std::ofstream out(path, fresh ? std::ios::trunc : std::ios::app);
    if (!out) return false;
    if (fresh) {
        out << ManifestMagic << ' ' << std::hex << sourceVersion << '\n';
        for (const auto known : keys) out << std::hex << known << '\n';
        loaded = true;
    }
    out << std::hex << key << '\n';
    keys.insert(key);
    return true;
}

PipelineAhead::PipelineAhead(Settings settings, std::shared_ptr<PipelineManifest> manifest) : settings(settings), manifest(std::move(manifest)) {}

PipelineAhead::~PipelineAhead() {
    {
        std::lock_guard lock(mutex);
        stopping = true;
    }
    work.notify_all();
    for (auto& thread : threads) thread.join();
}

void PipelineAhead::startLocked() {
    if (!threads.empty()) return;
    const auto count = settings.workers == 0 ? 1u : settings.workers;
    for (unsigned index = 0; index < count; ++index) {
        threads.emplace_back([this] { run(); });
        workerIds.insert(threads.back().get_id());
    }
}

void PipelineAhead::run() {
    std::unique_lock lock(mutex);
    for (;;) {
        work.wait(lock, [&] { return stopping || !queue.empty(); });
        if (queue.empty()) return;
        auto [key, factory, forgetOnFailure] = std::move(queue.front());
        queue.pop_front();
        ++running;
        lock.unlock();
        PipelineHandle handle;
        bool failed = false;
        try {
            handle = factory();
            failed = handle == nullptr;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[newpath] pipeline 0x%llx was not created ahead: %s\n", static_cast<unsigned long long>(key), error.what());
            failed = true;
        }
        lock.lock();
        ++(failed ? counts.failed : counts.created);
        --running;
        if (failed && forgetOnFailure) {
            entries.erase(key);
        } else {
            auto& entry = entries[key];
            entry.handle = std::move(handle);
            entry.done = true;
            entry.failed = failed;
            entry.onWorker = true;
        }
        finished.notify_all();
    }
}

PipelineAhead::Lookup PipelineAhead::Request(PipelineKey key, PipelineFactory factory) {
    return Request(key, std::move(factory), Options{});
}

PipelineAhead::Lookup PipelineAhead::Request(PipelineKey key, PipelineFactory factory, Options options) {
    std::unique_lock lock(mutex);
    if (const auto found = entries.find(key); found != entries.end()) return found->second.done ? (found->second.failed ? Lookup::Unknown : Lookup::Ready) : Lookup::Pending;
    ++counts.requested;
    if (!settings.gate) {
        entries[key];
        lock.unlock();
        PipelineHandle handle;
        try {
            handle = factory();
        } catch (...) {
            lock.lock();
            entries[key].done = true;
            entries[key].failed = true;
            ++counts.failed;
            finished.notify_all();
            throw;
        }
        lock.lock();
        auto& done = entries[key];
        done.handle = std::move(handle);
        done.done = true;
        done.failed = done.handle == nullptr;
        ++counts.createdInline;
        finished.notify_all();
        return done.failed ? Lookup::Unknown : Lookup::Ready;
    }
    entries[key];
    startLocked();
    if (options.urgent) queue.push_front({key, std::move(factory), options.forgetOnFailure});
    else queue.push_back({key, std::move(factory), options.forgetOnFailure});
    lock.unlock();
    work.notify_one();
    if (manifest != nullptr) static_cast<void>(manifest->Record(key));
    return Lookup::Pending;
}

std::optional<PipelineAhead::Lookup> PipelineAhead::Find(PipelineKey key) const {
    std::lock_guard lock(mutex);
    const auto found = entries.find(key);
    if (found == entries.end()) return std::nullopt;
    return found->second.done ? (found->second.failed ? Lookup::Unknown : Lookup::Ready) : Lookup::Pending;
}

PipelineHandle PipelineAhead::Ready(PipelineKey key) const {
    std::lock_guard lock(mutex);
    const auto found = entries.find(key);
    return found != entries.end() && found->second.done ? found->second.handle : nullptr;
}

PipelineHandle PipelineAhead::Acquire(PipelineKey key, PipelineFactory factory) {
    const bool seenBefore = manifest != nullptr && manifest->Known(key);
    const auto state = Request(key, std::move(factory));
    if (state == Lookup::Ready) return Ready(key);
    if (state == Lookup::Unknown) return nullptr;
    std::unique_lock lock(mutex);
    if (seenBefore) {
        ++counts.pendingFallbacks;
        return nullptr;
    }
    ++counts.blockedWaits;
    finished.wait(lock, [&] { return entries[key].done; });
    const auto& entry = entries[key];
    return entry.failed ? nullptr : entry.handle;
}

std::size_t PipelineAhead::Prewarm(const std::function<PipelineFactory(PipelineKey)>& resolve) {
    if (manifest == nullptr) return 0;
    std::size_t queued = 0;
    for (const auto key : manifest->Keys()) {
        auto factory = resolve(key);
        if (!factory) continue;
        if (Request(key, std::move(factory)) != Lookup::Ready) ++queued;
    }
    std::lock_guard lock(mutex);
    counts.prewarmed += queued;
    return queued;
}

void PipelineAhead::WaitIdle() {
    std::unique_lock lock(mutex);
    finished.wait(lock, [&] { return queue.empty() && running == 0; });
}

void PipelineAhead::Reset() {
    std::unique_lock lock(mutex);
    for (const auto& queued : queue) entries.erase(queued.key);
    queue.clear();
    finished.wait(lock, [&] { return running == 0; });
    entries.clear();
    finished.notify_all();
}

PipelineAhead::Counts PipelineAhead::Totals() const {
    std::lock_guard lock(mutex);
    return counts;
}

bool PipelineAhead::CreatedOnWorker(PipelineKey key) const {
    std::lock_guard lock(mutex);
    const auto found = entries.find(key);
    return found != entries.end() && found->second.onWorker;
}

}
