#include "ResidentCopyIn.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidentShadows.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include "prx/libc/include/GuestWriteWatch.hpp"
#include <sys/mman.h>
#endif

namespace {

using namespace AgcDriver::Graphics;
namespace GM = AgcDriver::GuestMemory;

constexpr std::uint64_t Unit = 65536;
constexpr std::size_t UnitWords = Unit / 4;
constexpr std::uint32_t Initial = 0x11111111u;

void* allocateWatched(std::size_t bytes) {
    if (!GM::WriteWatched()) return nullptr;
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, Unit);
    if (block == nullptr) return nullptr;
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* raw = mmap(nullptr, bytes + Unit, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return nullptr;
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (begin + Unit - 1) & ~std::uintptr_t{Unit - 1};
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + Unit) munmap(reinterpret_cast<void*>(aligned + bytes), begin + Unit - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    return block;
}

void releaseWatched(void* block, std::size_t bytes) {
#ifdef _WIN32
    GuestArena::GuestArenaReset_nid_postfix(block, bytes);
    GuestArena::GuestArenaRelease_nid_postfix(block, bytes);
#else
    munmap(block, bytes);
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, bytes);
#endif
}

void* allocateUnwatched(std::size_t bytes) {
#ifdef _WIN32
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    return std::aligned_alloc(Unit, bytes);
#endif
}

void releaseUnwatched(void* block) {
#ifdef _WIN32
    VirtualFree(block, 0, MEM_RELEASE);
#else
    std::free(block);
#endif
}

struct Fixture {
    const Context& context;
    Recorder& recorder;
    std::uint64_t element = 0;
    std::size_t elementBytes = 0;
    volatile std::uint32_t* words = nullptr;
    PFN_vkCmdFillBuffer fill = nullptr;
    bool diff = false;
};

// One build of the staged element: its guest memory (the copy-in of its first use is recorded by
// Upload) and the shadow's descriptor view.
struct Build {
    std::unique_ptr<GuestBufferMemory> memory;
    VkDescriptorBufferInfo view{};
    std::uint32_t adjustment = 0;
};

Build makeBuild(Fixture& fx) {
    Build build;
    build.memory = std::make_unique<GuestBufferMemory>(fx.context);
    build.memory->AllowDeviceStaging();
    build.memory->AddWritable(fx.element, fx.elementBytes, false, true);
    build.memory->Upload(false);
    Require(build.memory->InPlaceReads().empty(), "the resident test element was not staged");
    build.view = build.memory->Descriptor(fx.element, fx.elementBytes, build.adjustment);
    return build;
}

void fillShadow(Fixture& fx, const Build& build, std::size_t word, std::uint32_t value) {
    auto commands = fx.recorder.Commands();
    fx.fill(commands, build.view.buffer, build.view.offset + build.adjustment + word * 4u, 4, value);
    RecordMemoryBarrier(fx.context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
}

// The work of a use (stores into the shadow), its copy-back, notes and marks, as MarkGpuWrites
// records them after a dispatch, then the batch run to completion.
void work(Fixture& fx, Build& build, std::vector<std::pair<std::size_t, std::uint32_t>> stores) {
    for (const auto& [word, value] : stores) fillShadow(fx, build, word, value);
    build.memory->RecordCopyBacks(fx.recorder);
    build.memory->NoteOwnWrites(fx.recorder);
    build.memory->MarkDirectWrites();
    fx.recorder.Sync();
}

// A store into the shadow that nothing tracks: it reaches the import with the next use's whole
// copy-back only when that use skipped the copy-in of its 64 KiB unit (diff off).
void secret(Fixture& fx, const Build& build, std::size_t word, std::uint32_t value) {
    fillShadow(fx, build, word, value);
    fx.recorder.Sync();
}

void reuse(Fixture& fx, Build& build) {
    Require(build.memory->RecordStagingCopies(fx.recorder), "a reused build with a free resident shadow was refused");
}

std::uint32_t guest(Fixture& fx, std::size_t word) {
    return fx.words[word];
}

void expectWord(Fixture& fx, std::size_t word, std::uint32_t expected, const char* what) {
    const auto actual = guest(fx, word);
    if (actual == expected) return;
    std::cout << "resident copy-in: element dword " << word << " holds 0x" << std::hex << actual << ", expected 0x" << expected << std::dec << " (" << what << ")\n";
    Require(false, std::string("resident copy-in: ") + what);
}

std::uint64_t counter(const std::atomic<std::uint64_t>& value) {
    return value.load(std::memory_order_relaxed);
}

struct Restore {
    const Context& context;
    ImportWatch decided;
    ~Restore() { SetImportWatch(context, decided); }
};

int unwatchedTests(Fixture& base) {
    // An import of memory the tracker does not watch: nothing proves the range, so every use
    // copies it in whole (a stale shadow is never taken).
    constexpr std::size_t bytes = 1u << 20u;
    void* block = allocateUnwatched(bytes);
    Require(block != nullptr, "cannot allocate the unwatched block");
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    GuestAllocations::Mutation().Add(block, bytes, true, true, true);
    if (HostImportFor(base.context, address, bytes) == nullptr) {
        GuestAllocations::Mutation().Remove(block);
        releaseUnwatched(block);
        std::cout << "host import of the unwatched block refused: not tested\n";
        return 0;
    }
    Fixture fx{base.context, base.recorder, address + Unit, 4 * Unit, static_cast<volatile std::uint32_t*>(block) + UnitWords, base.fill, base.diff};
    auto build = makeBuild(fx);
    work(fx, build, {{3, 0x21212121u}});
    secret(fx, build, 9, 0xdead0101u);
    reuse(fx, build);
    work(fx, build, {});
    expectWord(fx, 3, 0x21212121u, "a use's store into an unwatched element was lost");
    if (!fx.diff) expectWord(fx, 9, Initial, "a use of an unwatched element skipped its copy-in");
    build.memory.reset();
    GuestAllocations::Mutation().Remove(block);
    HostImportFor(base.context, address, bytes);
    releaseUnwatched(block);
    std::cout << "resident copy-in: unwatched imports copy in whole\n";
    return 0;
}

}

int RunResidentCopyInTests(const Context& context, Recorder& recorder, bool budget) {
    if (context.hostImportAlignment == 0 || !GM::WriteWatched()) {
        std::cout << "host imports or write watching unavailable: resident copy-in not tested\n";
        return 77;
    }
    if (!ResidentShadows::Enabled()) {
        std::cout << "APS5_RESIDENT_MIB=0: resident copy-in not tested\n";
        return 77;
    }
    const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const bool verifying = std::getenv("APS5_VERIFY_RESIDENT") != nullptr;
    const Restore restore{context, PrepareImportWatch(context)};
    SetImportWatch(context, ImportWatch::Watch);
    constexpr std::size_t bytes = 4u << 20u;
    void* block = allocateWatched(bytes);
    if (block == nullptr) {
        std::cout << "cannot allocate watched guest memory: resident copy-in not tested\n";
        return 77;
    }
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    GuestAllocations::Mutation().Add(block, bytes, true, true, true);
    struct Cleanup {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Cleanup() {
            GuestAllocations::Mutation().Remove(block);
            HostImportFor(context, address, bytes);
            releaseWatched(block, bytes);
        }
    } cleanup{context, block, address};
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr) {
        std::cout << "host import of the watched block refused: resident copy-in not tested\n";
        return 77;
    }
    // The element: 8 units (2 MiB in the budget variant, over its 1 MiB budget) from unit 2.
    const std::size_t elementBytes = budget ? 32 * Unit : 8 * Unit;
    Fixture fx{context, recorder, address + 2 * Unit, elementBytes, static_cast<volatile std::uint32_t*>(block) + 2 * UnitWords, context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer"), std::getenv("APS5_STAGING_DIFF") != nullptr && context.bufferDeviceAddress};
    if (!GM::Watched(fx.element, fx.elementBytes)) {
        std::cout << "the imported block left the write watch: resident copy-in not tested\n";
        return 77;
    }
    const bool detect = !fx.diff;
    auto& stats = ResidentShadows::Counters();
    const auto word = [](std::size_t unit, std::size_t index) { return unit * UnitWords + index; };

    if (budget) {
        // A shadow over the budget stays private: every use copies in whole, and the results land.
        const auto refusedBefore = counter(stats.privateBudget);
        auto build = makeBuild(fx);
        Require(counter(stats.privateBudget) > refusedBefore, "a shadow over APS5_RESIDENT_MIB was registered");
        work(fx, build, {{word(0, 1), 0x22222222u}});
        secret(fx, build, word(3, 5), 0xdead0001u);
        reuse(fx, build);
        work(fx, build, {{word(1, 2), 0x23232323u}});
        expectWord(fx, word(0, 1), 0x22222222u, "a private shadow's first use lost its store");
        expectWord(fx, word(1, 2), 0x23232323u, "a private shadow's second use lost its store");
        if (detect) expectWord(fx, word(3, 5), Initial, "a private shadow skipped its copy-in");
        std::cout << "resident copy-in budget tests passed\n";
        return 0;
    }

    // (1) A new range: the first use registers its shadow and copies in whole.
    auto a = makeBuild(fx);
    work(fx, a, {{word(0, 1), 0x22222222u}});
    expectWord(fx, word(0, 1), 0x22222222u, "the first use's store did not reach the import");
    expectWord(fx, word(0, 2), Initial, "the first use changed a byte it did not store");
    Require(ResidentShadows::Find(fx.element, fx.element + fx.elementBytes, false) != nullptr, "the first use registered no resident shadow");

    // (2) No write between two uses: the copy-in is skipped (the untracked shadow store survives).
    const auto skippedBefore = counter(stats.skipped);
    const auto verifiedBefore = counter(stats.verified);
    const auto differsBefore = counter(stats.verifyDiffers);
    reuse(fx, a);
    work(fx, a, {{word(0, 3), 0x23232323u}});
    expectWord(fx, word(0, 3), 0x23232323u, "a skipped use lost its store");
    expectWord(fx, word(0, 1), 0x22222222u, "a skipped use lost the previous use's store");
    if (profile) Require(counter(stats.skipped) > skippedBefore, "an unchanged range was copied in");
    if (verifying) Require(counter(stats.verified) > verifiedBefore && counter(stats.verifyDiffers) == differsBefore, "APS5_VERIFY_RESIDENT reported a clean skipped shadow as different");
    secret(fx, a, word(0, 10), 0xdead0001u);
    reuse(fx, a);
    work(fx, a, {});
    if (detect) expectWord(fx, word(0, 10), 0xdead0001u, "an unchanged range was copied in (the shadow's own bytes were replaced)");
    if (verifying) Require(counter(stats.verifyDiffers) > differsBefore, "APS5_VERIFY_RESIDENT did not see a shadow that differs from its import");

    // (3) A CPU store between two uses: its unit is copied in (the store survives the copy-back,
    // the untracked shadow store in that unit is replaced), the other units are not.
    fx.words[word(2, 5)] = 0x77777777u;
    secret(fx, a, word(2, 9), 0xdead0002u);
    secret(fx, a, word(4, 9), 0xdead0003u);
    const auto partialBefore = counter(stats.partial);
    reuse(fx, a);
    work(fx, a, {{word(5, 1), 0x24242424u}});
    expectWord(fx, word(2, 5), 0x77777777u, "a CPU store made between two uses was rolled back");
    expectWord(fx, word(5, 1), 0x24242424u, "a partly copied use lost its store");
    if (detect) expectWord(fx, word(2, 9), Initial, "the unit of a CPU store was not copied in");
    if (detect) expectWord(fx, word(4, 9), 0xdead0003u, "a unit no CPU store touched was copied in");
    if (profile) Require(counter(stats.partial) > partialBefore, "a CPU store into one unit did not make a partial copy-in");

    // (4) A new build of the same range shares the shadow and skips the copy-in.
    secret(fx, a, word(5, 7), 0xdead0004u);
    auto b = makeBuild(fx);
    Require(b.view.buffer == a.view.buffer, "a new build of a resident range did not take its shadow");
    work(fx, b, {});
    if (detect) expectWord(fx, word(5, 7), 0xdead0004u, "a new build of an unchanged resident range copied it in");

    // (4b) APS5_STAGING_DIFF: the pristine copy is the entry's and its diff copy-back keeps it equal
    // to the shadow, so a use that copies nothing in takes no pristine copy either. Shown with an
    // untracked store into the pristine copy: taken over, the kernel sees the shadow's dword as
    // changed and stores it over a CPU store the work left alone; refreshed (or with
    // APS5_NO_RESIDENT_PRISTINE set), the CPU store survives as without residency.
    if (fx.diff) {
        const auto entry = ResidentShadows::Find(fx.element, fx.element + fx.elementBytes, false);
        Require(entry != nullptr && entry->pristine != nullptr, "the resident entry has no pristine copy under APS5_STAGING_DIFF");
        const auto pristineCase = [&](std::size_t target, bool kept) {
            const auto savedBefore = counter(stats.pristineSavedBytes);
            {
                auto commands = recorder.Commands();
                fx.fill(commands, entry->pristine->Handle(), target * 4u, 4, 0xdeadbeefu);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                recorder.Sync();
            }
            reuse(fx, a);
            recorder.Sync();
            fx.words[target] = 0x99999999u;
            work(fx, a, {});
            if (kept) {
                expectWord(fx, target, Initial, "a pristine copy that equals the shadow was copied again (the marker went)");
                if (profile) Require(counter(stats.pristineSavedBytes) > savedBefore, "a skipped pristine copy was not counted");
            } else {
                expectWord(fx, target, 0x99999999u, "a refreshed pristine copy lost a CPU store the work left alone");
            }
            // The CPU store made during the use: the next use copies its unit (and the pristine part).
            reuse(fx, a);
            work(fx, a, {});
        };
        pristineCase(word(0, 40), true);
        RuntimeSwitches::Assign(RuntimeSwitch::NoResidentPristine, true);
        pristineCase(word(0, 41), false);
        RuntimeSwitches::Assign(RuntimeSwitch::NoResidentPristine, false);
        // The first use after the switch cleared copies the whole pristine in again, then it holds.
        reuse(fx, a);
        work(fx, a, {});
        pristineCase(word(0, 42), true);
    }

    // (5) A recorded GPU write by other work into the import (noted, not yet marked): the next
    // use copies in whole, after it.
    {
        const auto* entry = HostImportFor(context, fx.element, fx.elementBytes);
        Require(entry != nullptr, "the test import went away");
        const auto target = fx.element + word(6, 2) * 4u;
        auto commands = recorder.Commands();
        fx.fill(commands, entry->buffer, HostImportOffset(*entry, target), 4, 0x66666666u);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
        recorder.NotePendingWrite(target, 4);
        secret(fx, a, word(7, 3), 0xdead0005u);
        reuse(fx, a);
        work(fx, a, {});
        expectWord(fx, word(6, 2), 0x66666666u, "a noted GPU write into the import was rolled back");
        if (detect) expectWord(fx, word(7, 3), Initial, "a use after a foreign GPU write did not copy in whole");
    }

    // (6) A driver store (MarkWritten from another path) into the range: its unit is copied in.
    fx.words[word(1, 11)] = 0x55555555u;
    GM::MarkWritten(fx.element + word(1, 11) * 4u, 4);
    secret(fx, a, word(1, 12), 0xdead0006u);
    reuse(fx, a);
    work(fx, a, {});
    expectWord(fx, word(1, 11), 0x55555555u, "a driver store between two uses was rolled back");
    if (detect) expectWord(fx, word(1, 12), Initial, "the unit of a driver store was not copied in");

    // (6b) A CPU store made during a use (after its copy-in decision, before its end): the use
    // ends without collecting it, and the next use's decision finds it and copies its unit in.
    reuse(fx, a);
    fx.words[word(6, 20)] = 0x56565656u;
    work(fx, a, {});
    secret(fx, a, word(6, 21), 0xdead000cu);
    reuse(fx, a);
    work(fx, a, {});
    if (detect) expectWord(fx, word(6, 21), Initial, "a CPU store made during the previous use did not get its unit copied in");
    if (detect) expectWord(fx, word(6, 20), Initial, "the whole copy-back of the use the CPU store raced did not store the shadow's bytes");

    // (7) An open use of another build: a new build takes a private shadow, and the open use, with
    // the other build's writes noted during it, ends unclean (the next use copies in whole).
    {
        const auto busyBefore = counter(stats.privateBusy);
        reuse(fx, a);
        auto c = makeBuild(fx);
        Require(c.view.buffer != a.view.buffer, "a new build took a resident shadow with an open use");
        if (profile) Require(counter(stats.privateBusy) > busyBefore, "the private shadow of a busy range was not counted");
        work(fx, c, {});
        const auto foreignBefore = counter(stats.endsForeign);
        work(fx, a, {});
        Require(counter(stats.endsForeign) > foreignBefore, "a use with another build's writes noted during it ended clean");
        secret(fx, a, word(0, 30), 0xdead0007u);
        reuse(fx, a);
        work(fx, a, {});
        if (detect) expectWord(fx, word(0, 30), Initial, "a use after an unclean end skipped its copy-in");
    }

    // (8) APS5_NO_RESIDENT_COPY_IN toggled while the game runs: set, the next use copies in whole;
    // cleared again, the use after skips (the uses made while set kept the entry's state).
    RuntimeSwitches::Assign(RuntimeSwitch::NoResidentCopyIn, true);
    secret(fx, a, word(1, 7), 0xdead0008u);
    reuse(fx, a);
    work(fx, a, {});
    if (detect) expectWord(fx, word(1, 7), Initial, "a use with APS5_NO_RESIDENT_COPY_IN set skipped its copy-in");
    RuntimeSwitches::Assign(RuntimeSwitch::NoResidentCopyIn, false);
    secret(fx, a, word(1, 8), 0xdead0009u);
    reuse(fx, a);
    work(fx, a, {});
    if (detect) expectWord(fx, word(1, 8), 0xdead0009u, "a use after APS5_NO_RESIDENT_COPY_IN was cleared copied in");

    // (9) A use that never ended (a reused build not dispatched): the next one copies in whole.
    reuse(fx, a);
    secret(fx, a, word(3, 4), 0xdead000au);
    reuse(fx, a);
    work(fx, a, {});
    if (detect) expectWord(fx, word(3, 4), Initial, "a use after one that never ended skipped its copy-in");

    // (10) The build destroyed with its use open leaves the entry unclean; a new build copies in.
    reuse(fx, a);
    secret(fx, a, word(4, 4), 0xdead000bu);
    const auto shadow = a.view.buffer;
    a.memory.reset();
    b.memory.reset();
    auto d = makeBuild(fx);
    Require(d.view.buffer == shadow, "the resident shadow did not outlive its builds");
    work(fx, d, {});
    if (detect) expectWord(fx, word(4, 4), Initial, "a use after a build destroyed mid-use skipped its copy-in");
    d.memory.reset();

    // The cost of the copy-in decision's walk (printed only): a GetWriteWatch over the range's pages.
    {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> stored;
        for (const std::size_t span : {std::size_t{64} << 10u, fx.elementBytes, bytes - 2 * Unit}) {
            const auto since = GM::ResidentUseGeneration(fx.element, span);
            const auto started = std::chrono::steady_clock::now();
            constexpr int rounds = 200;
            for (int round = 0; round < rounds; ++round) GM::ResidentStoredSince(fx.element, span, since, stored);
            const auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count() / rounds;
            std::cout << "resident copy-in: decision walk of " << (span >> 10u) << " KiB takes " << us << " us\n";
        }
    }
    unwatchedTests(fx);
    std::cout << "resident copy-in tests passed (" << (fx.diff ? "diff copy-backs" : "whole copy-backs") << (verifying ? ", verify" : "") << ")\n";
    return 0;
}
