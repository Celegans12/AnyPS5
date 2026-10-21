#include <io/FileIdentity.hpp>
#include <system_error>
#include <tuple>

#ifdef _WIN32
#include <windows.h>
#endif

namespace Io {

#ifdef _WIN32
namespace {

std::tuple<DWORD, DWORD, DWORD> FileIdentity(const std::filesystem::path& path) {
    const auto handle = CreateFileW(path.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
        throw std::filesystem::filesystem_error("Cannot open file identity", path, std::error_code(static_cast<int>(GetLastError()), std::system_category()));
    BY_HANDLE_FILE_INFORMATION information{};
    const bool read = GetFileInformationByHandle(handle, &information) != 0;
    const auto error = GetLastError();
    CloseHandle(handle);
    if (!read) throw std::filesystem::filesystem_error("Cannot read file identity", path, std::error_code(static_cast<int>(error), std::system_category()));
    return {information.dwVolumeSerialNumber, information.nFileIndexHigh, information.nFileIndexLow};
}

}
#endif

bool SameFile(const std::filesystem::path& first, const std::filesystem::path& second) {
#ifdef _WIN32
    return FileIdentity(first) == FileIdentity(second);
#else
    return std::filesystem::equivalent(first, second);
#endif
}

}
