#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_IMAGESLABS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_IMAGESLABS_HPP

#include <vulkan/vulkan.h>
#include <algorithm>
#include <bit>

namespace AgcDriver::Graphics {

inline constexpr VkDeviceSize ImageSlabLimit = VkDeviceSize{4} << 20u;

// The BufferPool slab class (a power-of-two slot size) a device-local sampled snapshot of `size` bytes takes, or 0 when
// it gets an allocation of its own: past ImageSlabLimit, or when the image fills less than 3/4 of its class, so a slot
// wastes at most a quarter of what the sampled texture cache counts for it.
inline VkDeviceSize ImageSlabClass(VkDeviceSize size, VkDeviceSize alignment) {
    if (size == 0) return 0;
    const auto capacity = std::bit_ceil(std::max(size, alignment));
    if (capacity > ImageSlabLimit || alignment > capacity) return 0;
    if (size * 4u < capacity * 3u) return 0;
    return capacity;
}

// What a snapshot in a slab slot counts against the sampled texture budget: the whole slot, which no other image uses.
inline VkDeviceSize ImageSlabAccountedBytes(VkDeviceSize capacity) {
    return capacity;
}

}

#endif
