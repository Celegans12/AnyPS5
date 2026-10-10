#ifndef ANYPS5_COLD_SPIRV_STORAGE_TEST_HOOKS_HPP
#define ANYPS5_COLD_SPIRV_STORAGE_TEST_HOOKS_HPP

#include <cstdint>
#include <filesystem>

namespace ShaderRecompiler::ColdSpirv::Testing {

enum class FileFailure { None, Create, Attach, Write, Flush, Verify, Read, ShortRead, CorruptRead };
struct FileStats {
    std::uint64_t activeFiles = 0, maxActiveFiles = 0, highWater = 0, liveBytes = 0;
    std::uint64_t appends = 0, reads = 0, fallbacks = 0, maxRestores = 0;
};
void ConfigureFileStorage(bool enabled, std::uint64_t limit = 8ull << 30u,
                          std::uint64_t reserve = 4ull << 30u,
                          std::uint64_t available = UINT64_MAX);
void FailFileOperation(FileFailure failure) noexcept;
FileStats InspectFileStorage() noexcept;
std::filesystem::path LastFilePath();
bool FileRangeFits(std::uint64_t offset, std::uint64_t bytes, std::uint64_t limit) noexcept;

}

#endif
