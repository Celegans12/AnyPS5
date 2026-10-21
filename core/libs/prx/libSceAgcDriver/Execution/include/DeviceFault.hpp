#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DEVICEFAULT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DEVICEFAULT_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace AgcDriver {

inline const char* DeviceFaultAddressKind(VkDeviceFaultAddressTypeEXT type) {
    switch (type) {
    case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT: return "none";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT: return "read invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT: return "write invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT: return "execute invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT: return "instruction pointer unknown";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT: return "instruction pointer invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT: return "instruction pointer fault";
    default: return "unknown";
    }
}

inline std::string DeviceFaultText(const char (&text)[VK_MAX_DESCRIPTION_SIZE]) {
    return std::string(text, strnlen(text, VK_MAX_DESCRIPTION_SIZE));
}

inline std::vector<std::string> DeviceFaultReport(VkDevice device, PFN_vkGetDeviceFaultInfoEXT faultInfo, VkQueue queue, PFN_vkGetQueueCheckpointDataNV checkpointData, std::vector<std::string> (*describe)(std::uint64_t, std::uint64_t) = nullptr) {
    std::vector<std::string> lines;
    char line[VK_MAX_DESCRIPTION_SIZE + 128];
    if (checkpointData != nullptr && queue != VK_NULL_HANDLE) {
        std::uint32_t count = 0;
        checkpointData(queue, &count, nullptr);
        std::vector<VkCheckpointDataNV> points(count, VkCheckpointDataNV{VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
        if (count != 0) checkpointData(queue, &count, points.data());
        points.resize(count);
        if (points.empty()) lines.emplace_back("[device-fault] no checkpoint reached");
        for (const auto& point : points) {
            const auto marker = static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(point.pCheckpointMarker));
            const auto kind = marker >> 56u;
            const char* name = kind == 0x1 ? "draw program" : kind == 0xf1 ? "copy-in of guest" : kind == 0xf2 ? "copy-back of guest" : kind == 0xf3 ? "diff copy-back of guest" : kind == 0xf4 ? "bindless publish" : kind == 0xf5 ? "DMA copy to guest" : kind == 0xf6 ? "GPU copies" : kind == 0xb0 ? "batch begin (lane << 48 | opened at)" : kind == 0xb1 ? "batch end (lane << 48 | serial)" : kind == 0xb2 ? "async lane dispatch of program" : "program";
            std::snprintf(line, sizeof(line), "[device-fault] checkpoint: %s 0x%llx at stage 0x%x", name, kind == 0x1 || kind >= 0xf1 || kind == 0xb0 || kind == 0xb1 || kind == 0xb2 ? marker & ((1ull << 56u) - 1u) : marker, static_cast<unsigned>(point.stage));
            lines.emplace_back(line);
        }
    }
    if (faultInfo == nullptr || device == VK_NULL_HANDLE) return lines;
    VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    if (faultInfo(device, &counts, nullptr) != VK_SUCCESS) {
        lines.emplace_back("[device-fault] vkGetDeviceFaultInfoEXT reported nothing");
        return lines;
    }
    std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
    counts.vendorBinarySize = 0;
    VkDeviceFaultInfoEXT info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
    info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
    info.pVendorInfos = vendors.empty() ? nullptr : vendors.data();
    const auto result = faultInfo(device, &counts, &info);
    addresses.resize(std::min<std::size_t>(addresses.size(), counts.addressInfoCount));
    vendors.resize(std::min<std::size_t>(vendors.size(), counts.vendorInfoCount));
    std::snprintf(line, sizeof(line), "[device-fault] result %d: %s", static_cast<int>(result), DeviceFaultText(info.description).c_str());
    lines.emplace_back(line);
    for (const auto& address : addresses) {
        std::snprintf(line, sizeof(line), "[device-fault] %s at 0x%llx (precision 0x%llx)", DeviceFaultAddressKind(address.addressType), static_cast<unsigned long long>(address.reportedAddress), static_cast<unsigned long long>(address.addressPrecision));
        lines.emplace_back(line);
        if (describe != nullptr) {
            for (auto& described : describe(address.reportedAddress, address.addressPrecision)) lines.push_back("[device-fault]   " + described);
        }
    }
    for (const auto& vendor : vendors) {
        std::snprintf(line, sizeof(line), "[device-fault] vendor: %s (code 0x%llx, data 0x%llx)", DeviceFaultText(vendor.description).c_str(), static_cast<unsigned long long>(vendor.vendorFaultCode), static_cast<unsigned long long>(vendor.vendorFaultData));
        lines.emplace_back(line);
    }
    return lines;
}

}

#endif
