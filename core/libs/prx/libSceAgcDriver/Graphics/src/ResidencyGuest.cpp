#include "prx/libSceAgcDriver/Graphics/include/ResidencyGuest.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include <algorithm>

namespace AgcDriver::Graphics {

namespace {

ResidencyRefusal RefusalOf(std::uint8_t code) {
    switch (code) {
    case 0:
        return ResidencyRefusal::None;
    case 2:
        return ResidencyRefusal::AliasHostWrite;
    case 3:
        return ResidencyRefusal::PrivatePinned;
    case 4:
        return ResidencyRefusal::PrivateHostWrite;
    case 5:
        return ResidencyRefusal::ResidentView;
    case 8:
        return ResidencyRefusal::UnsettledImport;
    default:
        return ResidencyRefusal::Unavailable;
    }
}

}

ArenaResidencyGuest::ArenaResidencyGuest(const Context& context) : context(context) {}

std::optional<ResidencyGuestImport> ArenaResidencyGuest::ImportOf(std::uint64_t page) {
    const auto address = page << ResidencyPageShift;
    const auto* import = ExistingHostImportFor(context, address, ResidencyPageBytes);
    if (import == nullptr && GuestMemory::GpuMutex().HeldByThisThread()) import = HostImportFor(context, address, ResidencyPageBytes);
    if (import == nullptr) return std::nullopt;
    if (import->unwatched && GuestMemory::GpuMutex().HeldByThisThread() && WatchedAfterSettle(context, address, ResidencyPageBytes)) {
        const auto* settled = ExistingHostImportFor(context, address, ResidencyPageBytes);
        if (settled != nullptr) import = settled;
    }
    ResidencyGuestImport located;
    const bool whole = import->base <= address && address + ResidencyPageBytes <= import->base + import->bytes;
    located.attributes.whole = whole && !import->unwatched && !import->dmaBuf;
    located.attributes.cpuWritable = true;
    located.attributes.gpuWritable = true;
    located.attributes.serial = import->serial;
    located.buffer = import->buffer;
    located.offset = HostImportOffset(*import, address);
    located.address = import->address + located.offset;
    return located;
}

std::vector<ResidencyRange> ArenaResidencyGuest::Aliases(std::uint64_t, std::uint64_t) {
    return {};
}

std::uint64_t ArenaResidencyGuest::Generation() {
    return GuestMemory::TrackerGeneration();
}

void ArenaResidencyGuest::ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) {
    if (generations.empty()) return;
    static_cast<void>(GuestMemory::CollectWritesUncached(firstPage << ResidencyPageShift, generations.size() << ResidencyPageShift));
    for (std::size_t k = 0; k < generations.size(); ++k) {
        if (generations[k] == ResidencyUnchecked) {
            changed[k] = 0;
            continue;
        }
        const auto stamp = GuestMemory::NewestStamp((firstPage + k) << ResidencyPageShift, ResidencyPageBytes);
        changed[k] = generations[k] == 0 || stamp > generations[k] ? 1 : 0;
    }
}

ResidencyRefusal ArenaResidencyGuest::Guard(std::uint64_t) {
    return ResidencyRefusal::Unavailable;
}

void ArenaResidencyGuest::Unguard(std::span<const std::uint64_t>) {}

ResidencyRefusal ArenaResidencyGuest::GuardAccess(std::uint64_t page, ResidencyMask mask) {
    const auto base = static_cast<std::uintptr_t>(page << ResidencyPageShift);
    std::vector<std::uintptr_t> armed;
    for (std::uint32_t subpage = 0; subpage < ResidencySubpages; ++subpage) {
        if ((mask & (1u << subpage)) == 0) continue;
        const auto address = base + (std::uintptr_t{subpage} << ResidencySubpageShift);
        const auto code = GuestArena::GuestArenaGuardAccess_nid_no_patch(address, ResidencySubpageBytes);
        if (code == 0) {
            armed.push_back(address);
            continue;
        }
        guardRefusals[std::min<std::size_t>(code, guardRefusals.size() - 1)].fetch_add(1, std::memory_order_relaxed);
        for (const auto undo : armed) GuestArena::GuestArenaUnguardAccess_nid_no_patch(undo, ResidencySubpageBytes);
        return RefusalOf(code);
    }
    return ResidencyRefusal::None;
}

void ArenaResidencyGuest::UnguardAccess(std::uint64_t page) {
    GuestArena::GuestArenaUnguardAccess_nid_no_patch(static_cast<std::uintptr_t>(page << ResidencyPageShift), ResidencyPageBytes);
}

std::uint64_t ArenaResidencyGuest::GuestBytes() {
    std::uintptr_t base = 0;
    std::size_t bytes = 0;
    GuestArena::GuestArenaRange_nid_postfix(&base, &bytes);
    return bytes;
}

}
