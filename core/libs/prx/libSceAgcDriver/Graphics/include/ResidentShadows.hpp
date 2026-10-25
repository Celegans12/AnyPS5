#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENTSHADOWS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENTSHADOWS_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"

namespace AgcDriver::Graphics {

class Buffer;

// Resident staging shadows (phase 1 stage 1, APS5_NO_RESIDENT_COPY_IN; see GuestBufferMemory.cpp).
// A staged region's device-local shadow is kept in a registry keyed by its guest range, so later
// builds and later uses of the same range share it, and a use copies in only the 64 KiB block parts
// of the range anything but the shadow's own uses stored into since the shadow last equalled the
// import. The state of an entry changes under the registry mutex only, always with the GPU mutex
// held as well (UploadFinish, RecordStagingCopies, MarkDirectWrites), except a note of a recorded
// GPU write (NoteWrite, from the recorder) and a build's destructor, which take the registry mutex.
namespace ResidentShadows {

struct Entry {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    bool addressable = false;
    std::shared_ptr<Buffer> shadow;
    // The import the shadow copies (HostImport::serial) and the guest mapping stamp of the range
    // (GuestMemory::ForgetStamp) when the last use began.
    std::uint64_t importSerial = 0;
    std::uint64_t forgetStamp = 0;
    // Registered (an evicted or replaced entry is not, and is never trusted again).
    bool live = true;
    // The shadow equals the import's bytes as of tracker generation residentSince: the last use
    // ended with nothing else storing into the range during it (see GuestMemory::ResidentMarkOwnWrites).
    bool clean = false;
    std::uint64_t residentSince = 0;
    // The build (its GuestBufferMemory) whose use is open: copy-in decided, copy-back not yet marked.
    const void* user = nullptr;
    std::uint64_t useSince = 0;
    int lane = 0;
    // A recorded GPU write by anyone but the open use's own build noted over the range since the
    // open use began (or since the last use ended).
    bool foreignNote = false;
    std::uint64_t lastUse = 0;
};

using EntryPtr = std::shared_ptr<Entry>;

// Whether the registry is used at all (APS5_RESIDENT_MIB=0 turns it off for the whole run).
bool Enabled();
// The live entry of exactly [begin, end) with that buffer usage, or null.
EntryPtr Find(std::uint64_t begin, std::uint64_t end, bool addressable);
// Registers a shadow for [begin, end) (replacing a live entry of the same key, which is dropped).
// Null when the budget (APS5_RESIDENT_MIB, default 1024) cannot hold it after evicting idle entries.
EntryPtr Register(std::uint64_t begin, std::uint64_t end, bool addressable, std::shared_ptr<Buffer> shadow, std::uint64_t importSerial);
// Drops an entry from the registry (its shadow stays with the builds holding it, as a private one).
void Drop(const EntryPtr& entry);
// Every entry (device teardown, ClearHostImports).
void Clear();
// The registry mutex, for the callers that change an entry's state.
std::unique_lock<AgcDriver::Mutex> Lock();
void Touch(Entry& entry);

// A recorded GPU write into guest memory (Recorder's write notes): every live entry it overlaps whose
// open use is not the noting build's own (OwnNotes) has its shadow differ from the import.
void NoteWrite(std::uint64_t begin, std::uint64_t end);
class OwnNotes {
public:
    explicit OwnNotes(const void* owner);
    ~OwnNotes();
    OwnNotes(const OwnNotes&) = delete;
    OwnNotes& operator=(const OwnNotes&) = delete;

private:
    const void* previous;
};

// APS5_PROFILE_DRAW counters, reported as the [resident] line every 10 s (Report).
enum class Full : std::size_t { NewEntry, Off, Unclean, Import, Mapping, Note, Lane, Lease, Unwatched, Count };
struct Stats {
    std::atomic<std::uint64_t> uses{0};
    std::atomic<std::uint64_t> skipped{0};
    std::atomic<std::uint64_t> partial{0};
    std::atomic<std::uint64_t> full{0};
    std::atomic<std::uint64_t> fullBy[static_cast<std::size_t>(Full::Count)]{};
    std::atomic<std::uint64_t> savedBytes{0};
    std::atomic<std::uint64_t> copiedBytes{0};
    std::atomic<std::uint64_t> privateBusy{0};
    std::atomic<std::uint64_t> privateBudget{0};
    std::atomic<std::uint64_t> privateLane{0};
    std::atomic<std::uint64_t> privateCovered{0};
    std::atomic<std::uint64_t> reuseRefused{0};
    std::atomic<std::uint64_t> takeovers{0};
    std::atomic<std::uint64_t> runsClosed{0};
    std::atomic<std::uint64_t> endsClean{0};
    std::atomic<std::uint64_t> endsForeign{0};
    std::atomic<std::uint64_t> foreignNotes{0};
    std::atomic<std::uint64_t> evictions{0};
    std::atomic<std::uint64_t> verified{0};
    std::atomic<std::uint64_t> verifyDiffers{0};
    std::atomic<std::uint64_t> faults{0};
};
Stats& Counters();
// Live entries and their shadow bytes.
std::pair<std::uint64_t, std::uint64_t> Held();
void Report();
// APS5_VERIFY_RESIDENT=<n>: whether this skipped copy-in is one of the first n of the current 10 s
// (it then also copies the import and the shadow into a host buffer and compares them).
bool VerifyThisUse();

}

}

#endif
