#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DevicePageTable.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/NewPath.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "VulkanTestDevice.hpp"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

void SetEnvironment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

}

int main() {
    try {
        SetEnvironment("APS5_NEW_PATH", "1");
        Expect(AgcDriver::NewPath::Enabled(), "APS5_NEW_PATH=1 did not turn the new path on");
        auto& table = AgcDriver::NewPath::DeviceTable();
        for (int round = 0; round < 2; ++round) {
            auto device = OpenVulkanTestDevice();
            if (device == nullptr) return failures != 0 ? 1 : VulkanTestSkipped;
            const auto before = table.Totals();
            const auto context = device->graphicsContext();
            std::weak_ptr<const AgcDriver::Graphics::BdaPageTable> held = table.Refresh(context);
            Expect(!held.expired(), "round " + std::to_string(round) + ": the device built no page table");
            Expect(table.Totals().builds == before.builds + 1, "round " + std::to_string(round) + ": the refresh did not build");
            device.reset();
            Expect(held.expired(), "round " + std::to_string(round) + ": the page table outlived its device");
            Expect(table.Current() == nullptr, "round " + std::to_string(round) + ": a table is still current after its device was destroyed");
            Expect(table.Totals().released == before.released + 1, "round " + std::to_string(round) + ": the device's teardown did not release its table");
            Expect(table.Totals().outlived == 0, "round " + std::to_string(round) + ": a table was kept past its device");
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d page table teardown checks failed\n", failures);
        return 1;
    }
    std::puts("new-path page table teardown tests passed");
    return 0;
}
