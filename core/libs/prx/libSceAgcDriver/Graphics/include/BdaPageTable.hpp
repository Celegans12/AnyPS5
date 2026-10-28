#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BDAPAGETABLE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BDAPAGETABLE_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "BdaAbi.hpp"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace AgcDriver::Graphics {

class Buffer;

void ValidateBdaRangeList(std::span<const ShaderRecompiler::BdaAbi::Range> ranges);
void ValidateBdaGrants(std::span<const ShaderRecompiler::BdaAbi::Grant> grants);
void ValidateBdaPageRanges(std::span<const ShaderRecompiler::BdaAbi::Range> ranges);
std::vector<ShaderRecompiler::BdaAbi::PageEntry> EncodeBdaWindow(std::uint32_t window, std::span<const ShaderRecompiler::BdaAbi::Range> ranges);

class BdaPageTable {
public:
    struct Counts {
        std::uint64_t tables = 0;
        std::uint64_t encodedWindows = 0;
        std::uint64_t sharedWindows = 0;
        std::uint64_t encodedPages = 0;
        std::uint64_t patchedWindows = 0;
    };
    using Pieces = std::function<std::vector<ShaderRecompiler::BdaAbi::Range>(std::uint64_t begin, std::uint64_t end)>;
    static std::shared_ptr<const BdaPageTable> Build(const Context& context, std::span<const ShaderRecompiler::BdaAbi::Range> ranges);
    static std::shared_ptr<const BdaPageTable> Build(const Context& context, std::span<const ShaderRecompiler::BdaAbi::Range> ranges, std::span<const std::shared_ptr<const BdaPageTable>> shares);
    static std::shared_ptr<const BdaPageTable> Patch(const Context& context, const std::shared_ptr<const BdaPageTable>& previous, std::span<const std::uint64_t> changedPages, const Pieces& pieces);
    static Counts Totals();
    static constexpr std::uint32_t EveryStream = ~0u;
    VkDeviceAddress Address() const;
    VkDeviceAddress LeafAddress(std::uint32_t window) const;
    std::optional<ShaderRecompiler::BdaAbi::Range> Find(std::uint64_t address) const;
    bool RecordUploads(const Context& context, VkCommandBuffer commands, std::uint32_t stream) const;
    bool NeedsUploads(std::uint32_t stream) const;
    VkMemoryPropertyFlags Properties() const;

private:
    struct Upload {
        std::shared_ptr<Buffer> buffer;
        std::shared_ptr<Buffer> staging;
        std::size_t bytes = 0;
        mutable std::atomic<std::uint32_t> streams{0};
    };
    struct Leaf {
        std::uint32_t window = 0;
        std::vector<ShaderRecompiler::BdaAbi::Range> ranges;
        std::vector<ShaderRecompiler::BdaAbi::PageEntry> entries;
        Upload table;
    };
    BdaPageTable() = default;
    static void stage(const Context& context, Upload& upload, const void* bytes, std::size_t size);
    void overlapping(std::uint64_t begin, std::uint64_t end, std::vector<ShaderRecompiler::BdaAbi::Range>& found) const;
    static bool recordUpload(const Context& context, VkCommandBuffer commands, const Upload& upload, std::uint32_t stream);
    std::shared_ptr<const Leaf> find(std::uint32_t window) const;
    std::vector<std::shared_ptr<const Leaf>> leaves;
    Upload first;
};

}

#endif
