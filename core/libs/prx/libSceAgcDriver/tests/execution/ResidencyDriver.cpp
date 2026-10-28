#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyConfig.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "VulkanTestDevice.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <stdlib.h>
#endif

namespace {

using namespace AgcDriver::Graphics;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

constexpr std::uint64_t GuestBase = 0x7000000000ull;
constexpr std::uint64_t BasePage = GuestBase >> ResidencyPageShift;

class BufferGuest final : public IResidencyGuestMemory {
public:
    explicit BufferGuest(const Context& context) : buffer(std::make_shared<Buffer>(context, 2 * ResidencyPageBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {}

    std::optional<ResidencyGuestImport> ImportOf(std::uint64_t page) override {
        if (page < BasePage || page >= BasePage + 2) return std::nullopt;
        ResidencyGuestImport import;
        import.attributes = ResidencyImport{true, true, true};
        import.buffer = buffer->Handle();
        import.offset = (page - BasePage) << ResidencyPageShift;
        return import;
    }
    std::vector<ResidencyRange> Aliases(std::uint64_t, std::uint64_t) override { return {}; }
    std::uint64_t Generation() override { return 1; }
    void ChangedPages(std::uint64_t, std::span<const std::uint64_t>, std::span<std::uint8_t> changed) override { std::fill(changed.begin(), changed.end(), std::uint8_t{0}); }
    ResidencyRefusal Guard(std::uint64_t) override { return ResidencyRefusal::Unavailable; }
    void Unguard(std::span<const std::uint64_t>) override {}
    ResidencyRefusal GuardAccess(std::uint64_t, ResidencyMask) override { return ResidencyRefusal::Unavailable; }
    void UnguardAccess(std::uint64_t) override {}
    std::uint64_t GuestBytes() override { return 2 * ResidencyPageBytes; }

    std::uint32_t* Words(std::uint32_t page) { return reinterpret_cast<std::uint32_t*>(buffer->Bytes().data() + (std::size_t{page} << ResidencyPageShift)); }

    std::shared_ptr<Buffer> buffer;
};

void RefusingGuestKeepsEverythingOnStaging(const Context& context, Recorder& recorder) {
    RefusingResidencyGuest guest;
    ResidencyDriver driver(context, recorder, guest, ResidencyPolicy{4, std::uint64_t{4} << ResidencyPageShift});
    Expect(ResidencyDriver::Active() == &driver, "refusing guest: the driver is not the active one");
    bool second = false;
    try {
        RefusingResidencyGuest other;
        ResidencyDriver duplicate(context, recorder, other, ResidencyPolicy{4, std::uint64_t{4} << ResidencyPageShift});
    } catch (const std::logic_error&) {
        second = true;
    }
    Expect(second, "refusing guest: a second active driver was allowed");
    Expect(ResidencyDriver::Active() == &driver, "refusing guest: the failed second driver cleared the active one");
    {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        driver.Manager().Use(GuestBase, 4 * ResidencyPageBytes, ResidencyAccess::Read);
        driver.Manager().Use(GuestBase, 4 * ResidencyPageBytes, ResidencyAccess::Write);
    }
    driver.OrderingPoint();
    const auto counts = driver.Manager().Counts();
    Expect(counts.resident == 0 && counts.chunks == 0 && counts.admitted == 0 && counts.copiedInBytes == 0, "refusing guest: a page became resident (" + std::to_string(counts.resident) + " resident, " + std::to_string(counts.chunks) + " chunks)");
    Expect(driver.Mirrors().DeviceBytes() == 0, "refusing guest: a mirror chunk was allocated");
    for (std::uint64_t page = BasePage; page < BasePage + 4; ++page) Expect(!driver.Manager().Slot(page).has_value(), "refusing guest: page " + std::to_string(page - BasePage) + " has a mirror slot");
}

void CopyRoundTripOnLane(const Context& context, Recorder& recorder, AgcDriver::VulkanDevice& device, int lane) {
    BufferGuest guest(context);
    ResidencyDriver driver(context, recorder, guest, ResidencyPolicy{4, std::uint64_t{4} << ResidencyPageShift});
    for (std::uint32_t word = 0; word < ResidencyPageBytes / 4; ++word) {
        guest.Words(0)[word] = 0x1000000u + word + static_cast<std::uint32_t>(lane) * 7u;
        guest.Words(1)[word] = 0;
    }
    {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        Expect(driver.AllocateChunk(0, 4) != 0, "lane " + std::to_string(lane) + ": no mirror chunk address");
    }
    ResidencyToken in{};
    {
        const Recorder::LaneScope scope(lane);
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        const std::vector<ResidencyCopy> copies{{BasePage, MirrorSlot{0, 2}, 0, static_cast<std::uint32_t>(ResidencyPageBytes)}};
        in = driver.CopyIn(copies);
    }
    Expect(in[static_cast<std::size_t>(lane)] != 0 && in[static_cast<std::size_t>(1 - lane)] == 0, "lane " + std::to_string(lane) + ": the copy-in token does not name its lane alone");
    Expect(!driver.Reached(in), "lane " + std::to_string(lane) + ": an unsubmitted copy-in is reached");
    ResidencyToken out{};
    {
        const Recorder::LaneScope scope(lane);
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        const std::vector<ResidencyCopy> copies{{BasePage + 1, MirrorSlot{0, 2}, 0, static_cast<std::uint32_t>(ResidencyPageBytes)}};
        out = driver.WriteBack(copies);
    }
    {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        driver.Wait(out);
    }
    Expect(driver.Reached(in) && driver.Reached(out), "lane " + std::to_string(lane) + ": Wait did not reach both tokens");
    std::uint32_t wrong = 0;
    for (std::uint32_t word = 0; word < ResidencyPageBytes / 4; ++word) wrong += guest.Words(1)[word] != guest.Words(0)[word];
    Expect(wrong == 0, "lane " + std::to_string(lane) + ": " + std::to_string(wrong) + " words differ after the import -> mirror -> import round trip");
    {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        device.SubmitRecorded(false);
        device.WaitIdle();
        driver.ReleaseChunk(0);
    }
}

}

int main(int argc, char** argv) {
    const bool asyncLane = argc > 1 && std::string(argv[1]) == "--async-lane";
#ifdef _WIN32
    if (asyncLane) _putenv_s("APS5_ASYNC_LANE", "51");
#else
    if (asyncLane) setenv("APS5_ASYNC_LANE", "51", 1);
#endif
    try {
        Expect(!CurrentResidencyConfig().residency, "the test runs without APS5_NEW_PATH, so the switch must read off");
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto context = device->graphicsContext();
        auto* recorder = Recorder::Active();
        if (recorder == nullptr) throw std::runtime_error("no active recorder");
        RefusingResidencyGuest refusing;
        Expect(ResidencyDriver::CreateIfEnabled(context, *recorder, refusing) == nullptr, "CreateIfEnabled built a driver with the switch off");
        RefusingGuestKeepsEverythingOnStaging(context, *recorder);
        Expect(ResidencyDriver::Active() == nullptr, "the driver stayed active after its destruction");
        CopyRoundTripOnLane(context, *recorder, *device, 0);
        if (asyncLane) {
            if (!recorder->AsyncLaneActive()) {
                std::puts("skipped: the device has no third queue for the async lane");
                return VulkanTestSkipped;
            }
            CopyRoundTripOnLane(context, *recorder, *device, 1);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        return 1;
    }
    if (failures != 0) return 1;
    std::printf("residency driver tests passed%s\n", asyncLane ? " (both lanes)" : "");
    return 0;
}
