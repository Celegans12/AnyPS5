#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HANDOFFSPIN_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HANDOFFSPIN_HPP

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace AgcDriver {

inline void SpinPause() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

inline std::chrono::nanoseconds HandoffSpinWindowFor(const char* configured, unsigned logicalProcessors) {
    constexpr std::chrono::microseconds automaticWindow{150};
    constexpr unsigned automaticProcessors = 8;
    constexpr long long longestWindowUs = 10000;
    if (configured != nullptr) return std::chrono::microseconds(std::clamp<long long>(std::strtoll(configured, nullptr, 10), 0, longestWindowUs));
    return logicalProcessors >= automaticProcessors ? std::chrono::nanoseconds(automaticWindow) : std::chrono::nanoseconds::zero();
}

inline std::chrono::nanoseconds HandoffSpinWindow() {
    static const std::chrono::nanoseconds window = HandoffSpinWindowFor(std::getenv("APS5_HANDOFF_SPIN_US"), std::thread::hardware_concurrency());
    return window;
}

template<typename TReady>
bool SpinUntil(std::chrono::nanoseconds window, TReady&& ready) {
    if (ready()) return true;
    if (window.count() <= 0) return false;
    const auto until = std::chrono::steady_clock::now() + window;
    do {
        SpinPause();
        if (ready()) return true;
    } while (std::chrono::steady_clock::now() < until);
    return false;
}

template<typename TMutex>
void LockSpinning(TMutex& mutex, std::chrono::nanoseconds window) {
    if (window.count() > 0 && SpinUntil(window, [&] { return mutex.try_lock(); })) return;
    mutex.lock();
}

}

#endif
