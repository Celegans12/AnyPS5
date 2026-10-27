#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_BINDLESSPRIMING_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_BINDLESSPRIMING_HPP

#include <cstdint>
#include <span>
#include <vector>

// The bindless heap primer of a frame replay (Graphics/include/BindlessRegistrations.hpp has the
// word layout): the tables the driver's texture service registered so far, and priming the heap
// with such tables before the replayed frames.
namespace AgcDriver {

std::vector<std::uint64_t> BindlessRegistrationWords();
void PrimeBindlessTables(std::span<const std::uint64_t> words);

}

#endif
