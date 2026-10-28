#include "VulkanTestDevice.hpp"
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::unique_ptr<AgcDriver::VulkanDevice> OpenBeside(const char* which) {
    try {
        return std::make_unique<AgcDriver::VulkanDevice>();
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(which) + " could not be created while another device is live in the process: " + error.what());
    }
}

}

int main() {
    try {
        auto first = OpenVulkanTestDevice();
        if (!first) return VulkanTestSkipped;
        auto second = OpenBeside("a second device");
        Require(second->Serial() != first->Serial(), "the second device has the first device's serial");
        const auto firstSerial = first->Serial();
        first.reset();
        auto replacement = OpenBeside("a replacement device");
        Require(replacement->Serial() != firstSerial && replacement->Serial() != second->Serial(), "the replacement device has the serial of an earlier device");
        std::puts("sparse residency device tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
