#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/BindlessPriming.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BindlessRegistrations.hpp"
#include <memory>
#include <mutex>

namespace AgcDriver::DriverDetail {

std::vector<std::uint64_t> Driver::BindlessRegistrationWords() {
    const auto localDevice = device.Load();
    if (localDevice == nullptr) return {};
    const auto tables = localDevice->BindlessRegistrations();
    return Graphics::EncodeBindlessRegistrations(tables);
}

void Driver::PrimeBindlessTables(std::span<const std::uint64_t> words) {
    const auto tables = Graphics::DecodeBindlessRegistrations(words);
    if (tables.empty()) return;
    auto localDevice = device.Load();
    if (localDevice == nullptr) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
        std::lock_guard createLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    localDevice->PrimeBindlessRegistrations(tables);
}

}

namespace AgcDriver {

std::vector<std::uint64_t> BindlessRegistrationWords() {
    return DriverDetail::Driver::Get().BindlessRegistrationWords();
}

void PrimeBindlessTables(std::span<const std::uint64_t> words) {
    DriverDetail::Driver::Get().PrimeBindlessTables(words);
}

}
