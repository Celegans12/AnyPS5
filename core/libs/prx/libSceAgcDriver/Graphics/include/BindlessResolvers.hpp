#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BINDLESSRESOLVERS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BINDLESSRESOLVERS_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureService.hpp"
#include "IntermediateRepresentation/IrMetadata/Resources.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace AgcDriver::Graphics {

struct BindlessInterface {
    ShaderRecompiler::ImageResource image;
    std::vector<ShaderRecompiler::ImageResource> modes;
};

void RegisterBindlessInterface(std::uint64_t id, BindlessInterface registered);
std::shared_ptr<const BindlessInterface> FindBindlessInterface(std::uint64_t id);

class GuestBindlessMemory final : public BindlessMemory {
public:
    bool Read(std::uint64_t address, std::span<std::uint32_t> words) override;
    std::uint64_t Collect(std::uint64_t address, std::size_t bytes) override;
    bool Unchanged(std::uint64_t address, std::size_t bytes, std::uint64_t generation) override;
    std::uint64_t MappingSerial() override;
    bool Mapped(std::uint64_t address, std::size_t bytes) override;
};

class DriverBindlessResolver final : public BindlessResolver {
public:
    using InterfaceLookup = std::function<std::shared_ptr<const BindlessInterface>(std::uint64_t)>;
    explicit DriverBindlessResolver(const Context& context, InterfaceLookup lookup = &FindBindlessInterface);
    BindlessPrepared Prepare(const DeviceImageKeys::Key& key) override;
    BindlessResolution Commit(const BindlessPrepared& prepared) override;
    bool Current(const BindlessResolution& resolution) override;

private:
    Context context;
    InterfaceLookup lookup;
};

}

#endif
