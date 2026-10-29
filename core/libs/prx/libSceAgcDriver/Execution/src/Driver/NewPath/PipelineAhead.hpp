#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_PIPELINEAHEAD_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_PIPELINEAHEAD_HPP

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace AgcDriver::NewPath {

using PipelineKey = std::uint64_t;
using PipelineHandle = std::shared_ptr<void>;
using PipelineFactory = std::function<PipelineHandle()>;

class PipelineManifest {
public:
    PipelineManifest(std::filesystem::path path, std::uint64_t sourceVersion);

    [[nodiscard]] bool Known(PipelineKey key) const;
    [[nodiscard]] std::vector<PipelineKey> Keys() const;
    bool Record(PipelineKey key);
    [[nodiscard]] bool LoadedFromFile() const { return loaded; }
    [[nodiscard]] const std::filesystem::path& Path() const { return path; }

private:
    void load();

    std::filesystem::path path;
    std::uint64_t sourceVersion = 0;
    mutable std::mutex mutex;
    std::set<PipelineKey> keys;
    bool loaded = false;
};

class PipelineAhead {
public:
    struct Settings {
        bool gate = true;
        unsigned workers = 2;
    };
    enum class Lookup : std::uint8_t { Ready, Pending, Unknown };
    // urgent: queued ahead of earlier requests (a dispatch waiting for it, not a prewarm).
    // forgetOnFailure: a failed creation leaves no entry, so a later request tries again.
    struct Options {
        bool urgent = false;
        bool forgetOnFailure = false;
    };
    struct Counts {
        std::uint64_t requested = 0;
        std::uint64_t created = 0;
        std::uint64_t failed = 0;
        std::uint64_t createdInline = 0;
        std::uint64_t blockedWaits = 0;
        std::uint64_t pendingFallbacks = 0;
        std::uint64_t prewarmed = 0;
    };

    PipelineAhead(Settings settings, std::shared_ptr<PipelineManifest> manifest);
    ~PipelineAhead();
    PipelineAhead(const PipelineAhead&) = delete;
    PipelineAhead& operator=(const PipelineAhead&) = delete;

    Lookup Request(PipelineKey key, PipelineFactory factory);
    Lookup Request(PipelineKey key, PipelineFactory factory, Options options);
    // The state of a key without requesting it (nullopt: never requested, or forgotten).
    [[nodiscard]] std::optional<Lookup> Find(PipelineKey key) const;
    [[nodiscard]] PipelineHandle Ready(PipelineKey key) const;
    PipelineHandle Acquire(PipelineKey key, PipelineFactory factory);
    std::size_t Prewarm(const std::function<PipelineFactory(PipelineKey)>& resolve);
    void WaitIdle();
    // Drops the queued requests, waits for the running ones and forgets every entry (device teardown: the handles
    // belong to the old device).
    void Reset();

    [[nodiscard]] Counts Totals() const;
    [[nodiscard]] bool CreatedOnWorker(PipelineKey key) const;

private:
    struct Entry {
        PipelineHandle handle;
        bool done = false;
        bool failed = false;
        bool onWorker = false;
    };
    void run();
    void startLocked();

    Settings settings;
    std::shared_ptr<PipelineManifest> manifest;
    mutable std::mutex mutex;
    std::condition_variable work;
    std::condition_variable finished;
    struct Queued {
        PipelineKey key;
        PipelineFactory factory;
        bool forgetOnFailure;
    };
    std::deque<Queued> queue;
    std::unordered_map<PipelineKey, Entry> entries;
    std::vector<std::thread> threads;
    std::set<std::thread::id> workerIds;
    std::size_t running = 0;
    bool stopping = false;
    Counts counts;
};

}

#endif
