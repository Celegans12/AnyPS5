#pragma once
#include <cstdint>

// Local diagnostic for upstream #9860 (clear hand-over -> glare in Yotei), never upstream as is.
// APS5_HANDOVER_TRACE=<firstPresent>:<count> writes handover-trace-<pid>.log (into APS5_TRACE_DIR or the
// working directory): every clearHandOver() with its receiver and sources, every supersedeHandedOver(),
// receivers refreshed before anything wrote them, and the first readers of a handed range (refresh,
// upload, borrow, write-back, guest read, bind). Lines are written only inside the present window; the
// hand-over bookkeeping runs from the start so a hand-over made earlier is still named.
// APS5_HANDOVER_TRACE_ADDR=<lo>-<hi>[,<lo>-<hi>...] keeps only events overlapping those ranges and logs
// their readers even without a hand-over (the run with the hand-over off).
namespace AgcDriver::Graphics::HandOverTrace {

bool Enabled();
// Driver::Present: counts presents for the window (present 0 is the first present).
void NotePresent();
// A sampled or storage image view bound by a draw or dispatch resource build.
void NoteBind(const char* kind, std::uint64_t address, std::uint64_t bytes, std::uint32_t format);

// The draw or dispatch the calling thread executes, named on every line it logs.
class CommandScope {
public:
    CommandScope(const char* kind, std::uint64_t address);
    ~CommandScope();
    CommandScope(const CommandScope&) = delete;
    CommandScope& operator=(const CommandScope&) = delete;

private:
    const char* previousKind;
    std::uint64_t previousAddress;
};

}
