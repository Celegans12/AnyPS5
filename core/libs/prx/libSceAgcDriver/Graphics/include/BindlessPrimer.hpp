#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BINDLESSPRIMER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BINDLESSPRIMER_HPP

#include "prx/libSceAgcDriver/Graphics/include/DeviceImageKeys.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <utility>

namespace AgcDriver::Graphics {

class BindlessTable;
class Sampler;
class Texture;

struct BindlessPrimedImage {
    DeviceImageKeys::Key key{};
    std::shared_ptr<Texture> texture;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::shared_ptr<Sampler> sampler;
    std::uint32_t mode = 0;
    std::uint32_t fault = 0;
};

class BindlessPrimer {
public:
    explicit BindlessPrimer(BindlessTable& table);
    ~BindlessPrimer();
    BindlessPrimer(const BindlessPrimer&) = delete;
    BindlessPrimer& operator=(const BindlessPrimer&) = delete;

    void Prime(std::span<const BindlessPrimedImage> entries);

private:
    BindlessTable& table;
    std::mutex mutex;
    std::map<std::pair<VkImageView, VkImageLayout>, std::pair<std::uint32_t, std::shared_ptr<Texture>>> images;
    std::map<VkSampler, std::pair<std::uint32_t, std::shared_ptr<Sampler>>> samplers;
};

}

#endif
