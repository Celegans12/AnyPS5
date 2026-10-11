#include "prx/libc/include/general/VabiMacros.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {

struct UsbdTimeval {
    std::int64_t seconds;
    std::int64_t microseconds;
};

}

extern "C" {
std::int32_t APS5_VABI sceUsbdInit();
void APS5_VABI sceUsbdExit();
std::int64_t APS5_VABI sceUsbdGetDeviceList(void*** list);
void APS5_VABI sceUsbdFreeDeviceList(void** list, std::int32_t unrefDevices);
std::int32_t APS5_VABI sceUsbdHandleEventsTimeout(const UsbdTimeval* timeout);
int APS5_VABI sceUsbdOpen();
void* APS5_VABI sceUsbdAllocTransfer(std::int32_t isoPackets);
void APS5_VABI sceUsbdFreeTransfer(void* transfer);
void APS5_VABI sceUsbdFillInterruptTransfer(void* transfer, void* deviceHandle, std::uint8_t endpoint, std::uint8_t* buffer, std::int32_t length, void* callback, void* userData, std::uint32_t timeout);
std::int32_t APS5_VABI sceUsbdCancelTransfer(void* transfer);
std::int32_t APS5_VABI sceUsbdEventHandlingOk();
void APS5_VABI sceUsbdFreeConfigDescriptor(void* config);
}

namespace {

constexpr std::int32_t invalidArgument = static_cast<std::int32_t>(0x80240002);

void Require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "USBD: %s\n", message);
        std::abort();
    }
}

}

int main() {
    Require(sceUsbdInit() == 0, "initialization failed");
    void** list = nullptr;
    Require(sceUsbdGetDeviceList(&list) == 0, "a device was listed");
    Require(list != nullptr && list[0] == nullptr, "device list is not an empty null-terminated array");
    sceUsbdFreeDeviceList(list, 1);
    Require(sceUsbdGetDeviceList(nullptr) == invalidArgument, "null list accepted");
    Require(sceUsbdHandleEventsTimeout(nullptr) == invalidArgument, "null timeout accepted");
    const UsbdTimeval invalid{0, 1000000};
    Require(sceUsbdHandleEventsTimeout(&invalid) == invalidArgument, "out-of-range microseconds accepted");
    const UsbdTimeval timeout{0, 50000};
    const auto start = std::chrono::steady_clock::now();
    Require(sceUsbdHandleEventsTimeout(&timeout) == 0, "event handling failed");
    Require(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(45), "event handling returned before the timeout");
    bool threw = false;
    try {
        sceUsbdOpen();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    Require(threw, "opening a device did not throw");
    auto* transfer = static_cast<std::uint8_t*>(sceUsbdAllocTransfer(2));
    Require(transfer != nullptr, "transfer allocation failed");
    Require(*reinterpret_cast<std::int32_t*>(transfer + 56) == 2, "iso packet count not stored");
    Require(sceUsbdAllocTransfer(-1) == nullptr, "negative iso packet count accepted");
    std::uint8_t buffer[8] = {};
    int user = 0;
    sceUsbdFillInterruptTransfer(transfer, nullptr, 0x81, buffer, 8, reinterpret_cast<void*>(&Require), &user, 250);
    Require(transfer[9] == 0x81 && transfer[10] == 3, "endpoint or interrupt type not filled");
    Require(*reinterpret_cast<std::uint32_t*>(transfer + 12) == 250 && *reinterpret_cast<std::int32_t*>(transfer + 20) == 8, "timeout or length not filled");
    Require(*reinterpret_cast<void**>(transfer + 40) == &user && *reinterpret_cast<std::uint8_t**>(transfer + 48) == buffer, "user data or buffer not filled");
    Require(sceUsbdCancelTransfer(transfer) == static_cast<std::int32_t>(0x80240005), "cancelling an unsubmitted transfer did not report not found");
    sceUsbdFreeTransfer(transfer);
    sceUsbdFreeTransfer(nullptr);
    Require(sceUsbdEventHandlingOk() == 1, "event handling not allowed");
    sceUsbdFreeConfigDescriptor(nullptr);
    sceUsbdExit();
    std::puts("USBD tests passed");
    return 0;
}
