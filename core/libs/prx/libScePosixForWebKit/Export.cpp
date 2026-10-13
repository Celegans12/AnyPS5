#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libc/include/HostEntropy.hpp"
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" {

char* APS5_VABI strcasestr_nid_postfix(const char* text, const char* needle) {
    if (*needle == '\0') return const_cast<char*>(text);
    for (; *text != '\0'; ++text) {
        const char* candidate = text;
        const char* match = needle;
        while (*candidate != '\0' && *match != '\0' &&
               std::tolower(static_cast<unsigned char>(*candidate)) ==
               std::tolower(static_cast<unsigned char>(*match))) {
            ++candidate;
            ++match;
        }
        if (*match == '\0') return const_cast<char*>(text);
    }
    return nullptr;
}

void APS5_VABI arc4random_buf_nid_postfix(void* buffer, std::size_t size) {
    HostEntropy_nid_no_patch(buffer, size);
}

std::uint32_t APS5_VABI arc4random_nid_postfix() {
    std::uint32_t value;
    arc4random_buf_nid_postfix(&value, sizeof(value));
    return value;
}

}
