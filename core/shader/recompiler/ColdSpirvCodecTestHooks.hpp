#ifndef ANYPS5_COLD_SPIRV_CODEC_TEST_HOOKS_HPP
#define ANYPS5_COLD_SPIRV_CODEC_TEST_HOOKS_HPP

#include <cstddef>

namespace ShaderRecompiler::ColdSpirv::Testing {

void FailAllocationAfter(std::size_t successfulAllocationChecks) noexcept;
void ResetAllocationFailure() noexcept;
std::size_t AllocationChecks() noexcept;
std::size_t ActiveAllocations() noexcept;
std::size_t ActiveBytes() noexcept;

}

#endif
