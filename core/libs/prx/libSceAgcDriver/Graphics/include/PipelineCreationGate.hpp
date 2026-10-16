#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINECREATIONGATE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINECREATIONGATE_HPP

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>

namespace AgcDriver::Graphics {

// Admits pipeline creations while the driver compiler memory they are estimated to take (SPIR-V
// words times bytes per word) fits a budget. A creation alone always runs, and so do small ones
// and ones that may not wait (the caller holds the GPU mutex); those only count.
class PipelineCreationGate {
public:
    struct Settings {
        std::uint64_t bytesPerWord = 3072;
        std::uint64_t budgetBytes = 0;
        std::uint64_t smallBytes = 256ull << 20u;
        std::function<std::uint64_t()> availableBytes;
    };

    struct Stats {
        std::uint64_t admitted = 0;
        std::uint64_t waited = 0;
        std::uint64_t waitNanoseconds = 0;
        std::uint64_t unwaited = 0;
        std::uint64_t inFlight = 0;
        std::uint64_t inFlightBytes = 0;
        std::uint64_t peakInFlight = 0;
        std::uint64_t peakBytes = 0;
        std::uint64_t budgetBytes = 0;
    };

    class Ticket {
    public:
        Ticket() = default;
        Ticket(PipelineCreationGate* gate, std::uint64_t bytes) : gate(gate), bytes(bytes) {}
        Ticket(const Ticket&) = delete;
        Ticket& operator=(const Ticket&) = delete;
        Ticket(Ticket&& other) noexcept : gate(other.gate), bytes(other.bytes) { other.gate = nullptr; }
        Ticket& operator=(Ticket&& other) noexcept {
            if (this != &other) {
                release();
                gate = other.gate;
                bytes = other.bytes;
                other.gate = nullptr;
            }
            return *this;
        }
        ~Ticket() { release(); }

    private:
        void release() noexcept {
            if (gate != nullptr) gate->release(bytes);
            gate = nullptr;
        }

        PipelineCreationGate* gate = nullptr;
        std::uint64_t bytes = 0;
    };

    explicit PipelineCreationGate(Settings settings);

    static PipelineCreationGate& Shared();
    static Settings SettingsFromEnvironment();
    static std::uint64_t PhysicalMemoryBytes();
    static std::uint64_t AvailableMemoryBytes();

    std::uint64_t EstimateBytes(std::size_t words) const { return static_cast<std::uint64_t>(words) * settings.bytesPerWord; }
    Ticket Admit(std::size_t words, bool mayWait);
    Stats Totals() const;

private:
    bool fits(std::uint64_t bytes) const;
    void release(std::uint64_t bytes) noexcept;

    Settings settings;
    mutable std::mutex mutex;
    std::condition_variable released;
    Stats stats;
};

}

#endif
