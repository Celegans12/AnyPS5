#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYGUEST_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYGUEST_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include <array>
#include <atomic>
#include <cstdint>

namespace AgcDriver::Graphics {

inline constexpr std::size_t ResidencyGuardRefusalCodes = 10;

class ArenaResidencyGuest final : public IResidencyGuestMemory {
public:
    explicit ArenaResidencyGuest(const Context& context);

    std::optional<ResidencyGuestImport> ImportOf(std::uint64_t page) override;
    std::vector<ResidencyRange> Aliases(std::uint64_t address, std::uint64_t bytes) override;
    std::uint64_t Generation() override;
    void ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) override;
    ResidencyRefusal Guard(std::uint64_t page) override;
    void Unguard(std::span<const std::uint64_t> pages) override;
    ResidencyRefusal GuardAccess(std::uint64_t page, ResidencyMask mask) override;
    void UnguardAccess(std::uint64_t page) override;
    std::uint64_t GuestBytes() override;
    ResidencyWrites PageWrites(std::uint64_t page, std::uint64_t generation) override;
    std::vector<std::uint64_t> WrittenSince(std::span<const std::pair<std::uint64_t, std::uint64_t>> resident) override;

    std::uint64_t UnwatchedImports() const { return unwatchedImports.load(std::memory_order_relaxed); }
    std::uint64_t GuardRefusals(std::size_t code) const { return code < guardRefusals.size() ? guardRefusals[code].load(std::memory_order_relaxed) : 0; }

private:
    Context context;
    std::array<std::atomic<std::uint64_t>, ResidencyGuardRefusalCodes> guardRefusals{};
    std::atomic<std::uint64_t> unwatchedImports{0};
};

}

#endif
