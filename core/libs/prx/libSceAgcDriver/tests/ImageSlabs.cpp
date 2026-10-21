#include "prx/libSceAgcDriver/Graphics/include/ImageSlabs.hpp"
#include <cstdio>

namespace {

constexpr VkDeviceSize KiB = 1024;
constexpr VkDeviceSize MiB = 1024 * KiB;

struct Case {
    VkDeviceSize size;
    VkDeviceSize alignment;
    VkDeviceSize slabClass;
};

constexpr Case Cases[]{
    {3 * MiB, 64 * KiB, 4 * MiB},
    {4 * MiB, 64 * KiB, 4 * MiB},
    {2 * MiB + 100 * KiB, 64 * KiB, 0},
    {3 * MiB - 1, 64 * KiB, 0},
    {5 * MiB, 64 * KiB, 0},
    {48 * KiB, 1 * KiB, 64 * KiB},
    {40 * KiB, 1 * KiB, 0},
    {16 * KiB, 64 * KiB, 0},
    {64 * KiB, 64 * KiB, 64 * KiB},
    {0, 1 * KiB, 0},
};

}

int main() {
    using namespace AgcDriver::Graphics;
    int failures = 0;
    for (const auto& item : Cases) {
        const auto got = ImageSlabClass(item.size, item.alignment);
        if (got != item.slabClass) {
            std::fprintf(stderr, "image of %llu bytes (alignment %llu): slab class %llu, expected %llu\n", static_cast<unsigned long long>(item.size), static_cast<unsigned long long>(item.alignment), static_cast<unsigned long long>(got), static_cast<unsigned long long>(item.slabClass));
            ++failures;
        }
        if (got != 0 && (ImageSlabAccountedBytes(got) != got || ImageSlabAccountedBytes(got) < item.size || ImageSlabAccountedBytes(got) * 3u > item.size * 4u)) {
            std::fprintf(stderr, "image of %llu bytes: a slot of %llu bytes is counted as %llu\n", static_cast<unsigned long long>(item.size), static_cast<unsigned long long>(got), static_cast<unsigned long long>(ImageSlabAccountedBytes(got)));
            ++failures;
        }
    }
    if (failures == 0) std::puts("image slab tests passed");
    return failures == 0 ? 0 : 1;
}
