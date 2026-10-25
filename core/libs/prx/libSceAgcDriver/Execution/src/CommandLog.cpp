#include "prx/libSceAgcDriver/Execution/include/CommandLog.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <unordered_map>

namespace AgcDriver::CommandLog {

namespace {

constexpr std::uint64_t FnvOffset = 14695981039346656037ull;
constexpr std::uint64_t FnvPrime = 1099511628211ull;

struct Buffer {
    std::uint64_t hash = FnvOffset;
    std::uint64_t commands = 0;
    std::uint32_t pool = 0;
};

struct StreamState {
    std::uint64_t hash = FnvOffset;
    std::uint64_t commands = 0;
};

struct State {
    std::mutex mutex;
    PFN_vkGetDeviceProcAddr real = nullptr;
    std::unordered_map<VkCommandBuffer, Buffer> buffers;
    std::unordered_map<std::uint64_t, std::uint32_t> pools;
    std::map<std::uint32_t, StreamState> streams;
    std::unordered_map<std::uint64_t, std::uint64_t> ordinals;
    std::unordered_map<std::uint64_t, std::uint64_t> sets;
    std::uint64_t presents = 0;
    std::FILE* out = nullptr;
    bool outOpened = false;
};

State& state() {
    static auto* created = new State();
    return *created;
}

template<typename T>
std::uint64_t raw(T handle) {
    if constexpr (std::is_pointer_v<T>) return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(handle));
    else return static_cast<std::uint64_t>(handle);
}

struct Hasher {
    State& log;
    std::uint64_t value;
    void word(std::uint64_t word) {
        for (int byte = 0; byte < 8; ++byte) {
            value ^= (word >> (byte * 8)) & 0xffu;
            value *= FnvPrime;
        }
    }
    void real(float number) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &number, sizeof(bits));
        word(bits);
    }
    template<typename T>
    void handle(T object) {
        const auto key = raw(object);
        if (key == 0) {
            word(0);
            return;
        }
        const auto [it, inserted] = log.ordinals.try_emplace(key, log.ordinals.size() + 1);
        word(it->second);
    }
    void bytes(const void* data, std::size_t size) {
        const auto* p = static_cast<const std::uint8_t*>(data);
        word(size);
        for (std::size_t index = 0; index < size; ++index) {
            value ^= p[index];
            value *= FnvPrime;
        }
    }
    void layers(const VkImageSubresourceLayers& layers) {
        word(layers.aspectMask);
        word(layers.mipLevel);
        word(layers.baseArrayLayer);
        word(layers.layerCount);
    }
    void range(const VkImageSubresourceRange& range) {
        word(range.aspectMask);
        word(range.baseMipLevel);
        word(range.levelCount);
        word(range.baseArrayLayer);
        word(range.layerCount);
    }
    void offset(const VkOffset3D& offset) {
        word(static_cast<std::uint32_t>(offset.x));
        word(static_cast<std::uint32_t>(offset.y));
        word(static_cast<std::uint32_t>(offset.z));
    }
    void extent(const VkExtent3D& extent) {
        word(extent.width);
        word(extent.height);
        word(extent.depth);
    }
    void rect(const VkRect2D& rect) {
        word(static_cast<std::uint32_t>(rect.offset.x));
        word(static_cast<std::uint32_t>(rect.offset.y));
        word(rect.extent.width);
        word(rect.extent.height);
    }
    void attachment(const VkRenderingAttachmentInfo* info) {
        if (info == nullptr) {
            word(0);
            return;
        }
        handle(info->imageView);
        word(info->imageLayout);
        word(info->resolveMode);
        handle(info->resolveImageView);
        word(info->resolveImageLayout);
        word(info->loadOp);
        word(info->storeOp);
        bytes(&info->clearValue, sizeof(info->clearValue));
    }
};

struct Scope {
    std::unique_lock<std::mutex> lock;
    Buffer& buffer;
    Hasher hash;
    Scope(VkCommandBuffer commands, std::uint32_t id) : lock(state().mutex), buffer(state().buffers[commands]), hash{state(), buffer.hash} { hash.word(id); }
    ~Scope() {
        buffer.hash = hash.value;
        ++buffer.commands;
    }
};

#define APS5_LOG_REAL(name) PFN_vk##name real##name = nullptr

APS5_LOG_REAL(CmdPipelineBarrier);
APS5_LOG_REAL(CmdBindPipeline);
APS5_LOG_REAL(CmdCopyBuffer);
APS5_LOG_REAL(CmdPushConstants);
APS5_LOG_REAL(CmdBindDescriptorSets);
APS5_LOG_REAL(CmdCopyImageToBuffer);
APS5_LOG_REAL(CmdDispatch);
APS5_LOG_REAL(CmdCopyBufferToImage);
APS5_LOG_REAL(CmdUpdateBuffer);
APS5_LOG_REAL(CmdFillBuffer);
APS5_LOG_REAL(CmdClearColorImage);
APS5_LOG_REAL(CmdCopyImage);
APS5_LOG_REAL(CmdDraw);
APS5_LOG_REAL(CmdBeginRenderPass);
APS5_LOG_REAL(CmdSetViewport);
APS5_LOG_REAL(CmdSetScissor);
APS5_LOG_REAL(CmdEndRenderPass);
APS5_LOG_REAL(CmdCopyQueryPoolResults);
APS5_LOG_REAL(CmdSetStencilReference);
APS5_LOG_REAL(CmdResetQueryPool);
APS5_LOG_REAL(CmdBlitImage);
APS5_LOG_REAL(CmdWriteTimestamp);
APS5_LOG_REAL(CmdSetDepthBounds);
APS5_LOG_REAL(CmdSetDepthBias);
APS5_LOG_REAL(CmdDrawIndexed);
APS5_LOG_REAL(CmdSetStencilWriteMask);
APS5_LOG_REAL(CmdEndQuery);
APS5_LOG_REAL(CmdDrawIndirect);
APS5_LOG_REAL(CmdDrawIndexedIndirect);
APS5_LOG_REAL(CmdClearDepthStencilImage);
APS5_LOG_REAL(CmdBindVertexBuffers);
APS5_LOG_REAL(CmdBindIndexBuffer);
APS5_LOG_REAL(CmdBeginQuery);
APS5_LOG_REAL(CmdSetStencilTestEnableEXT);
APS5_LOG_REAL(CmdSetStencilOpEXT);
APS5_LOG_REAL(CmdSetStencilCompareMask);
APS5_LOG_REAL(CmdSetFrontFaceEXT);
APS5_LOG_REAL(CmdSetDepthWriteEnableEXT);
APS5_LOG_REAL(CmdSetDepthTestEnableEXT);
APS5_LOG_REAL(CmdSetDepthCompareOpEXT);
APS5_LOG_REAL(CmdSetDepthBoundsTestEnableEXT);
APS5_LOG_REAL(CmdSetCullModeEXT);
APS5_LOG_REAL(CmdDispatchIndirect);
APS5_LOG_REAL(CmdBeginRenderingKHR);
APS5_LOG_REAL(CmdEndRenderingKHR);
APS5_LOG_REAL(CmdDrawMeshTasksIndirectEXT);
APS5_LOG_REAL(CmdDrawMeshTasksEXT);
APS5_LOG_REAL(CmdDrawIndirectCountKHR);
APS5_LOG_REAL(CmdDrawIndexedIndirectCountKHR);
APS5_LOG_REAL(AllocateCommandBuffers);
APS5_LOG_REAL(BeginCommandBuffer);
APS5_LOG_REAL(ResetCommandBuffer);
APS5_LOG_REAL(QueueSubmit);
APS5_LOG_REAL(QueuePresentKHR);
APS5_LOG_REAL(UpdateDescriptorSets);

#undef APS5_LOG_REAL

void writeLine(const std::string& line) {
    auto& log = state();
    const char* target = std::getenv("APS5_COMMAND_LOG");
    if (!log.outOpened) {
        log.outOpened = true;
        if (target != nullptr && std::strcmp(target, "1") != 0 && *target != '\0') log.out = std::fopen(target, "a");
    }
    if (log.out != nullptr) {
        std::fputs(line.c_str(), log.out);
        std::fflush(log.out);
    } else {
        AgcDriver::ProfilePrint_nid_no_patch("%s", line.c_str());
    }
}

std::string lineLocked(State& log, std::uint64_t present) {
    std::string line = "[cmdlog] present " + std::to_string(present) + ":";
    std::uint64_t total = FnvOffset;
    std::uint64_t commands = 0;
    char text[96];
    for (const auto& [pool, stream] : log.streams) {
        std::snprintf(text, sizeof(text), " pool%u %llu/%016llx", pool, static_cast<unsigned long long>(stream.commands), static_cast<unsigned long long>(stream.hash));
        line += text;
        Hasher mix{log, total};
        mix.word(pool);
        mix.word(stream.hash);
        total = mix.value;
        commands += stream.commands;
    }
    std::snprintf(text, sizeof(text), "; total %llu/%016llx\n", static_cast<unsigned long long>(commands), static_cast<unsigned long long>(total));
    line += text;
    return line;
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(VkCommandBuffer commands, VkPipelineStageFlags source, VkPipelineStageFlags destination, VkDependencyFlags dependency, std::uint32_t memoryCount, const VkMemoryBarrier* memory, std::uint32_t bufferCount, const VkBufferMemoryBarrier* buffers, std::uint32_t imageCount, const VkImageMemoryBarrier* images) {
    {
        Scope scope(commands, 1);
        auto& h = scope.hash;
        h.word(source);
        h.word(destination);
        h.word(dependency);
        h.word(memoryCount);
        for (std::uint32_t index = 0; index < memoryCount; ++index) {
            h.word(memory[index].srcAccessMask);
            h.word(memory[index].dstAccessMask);
        }
        h.word(bufferCount);
        for (std::uint32_t index = 0; index < bufferCount; ++index) {
            const auto& barrier = buffers[index];
            h.word(barrier.srcAccessMask);
            h.word(barrier.dstAccessMask);
            h.word(barrier.srcQueueFamilyIndex);
            h.word(barrier.dstQueueFamilyIndex);
            h.handle(barrier.buffer);
            h.word(barrier.offset);
            h.word(barrier.size);
        }
        h.word(imageCount);
        for (std::uint32_t index = 0; index < imageCount; ++index) {
            const auto& barrier = images[index];
            h.word(barrier.srcAccessMask);
            h.word(barrier.dstAccessMask);
            h.word(barrier.oldLayout);
            h.word(barrier.newLayout);
            h.word(barrier.srcQueueFamilyIndex);
            h.word(barrier.dstQueueFamilyIndex);
            h.handle(barrier.image);
            h.range(barrier.subresourceRange);
        }
    }
    realCmdPipelineBarrier(commands, source, destination, dependency, memoryCount, memory, bufferCount, buffers, imageCount, images);
}

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer commands, VkPipelineBindPoint point, VkPipeline pipeline) {
    {
        Scope scope(commands, 2);
        scope.hash.word(point);
        scope.hash.handle(pipeline);
    }
    realCmdBindPipeline(commands, point, pipeline);
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer commands, VkBuffer source, VkBuffer destination, std::uint32_t count, const VkBufferCopy* regions) {
    {
        Scope scope(commands, 3);
        auto& h = scope.hash;
        h.handle(source);
        h.handle(destination);
        h.word(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            h.word(regions[index].srcOffset);
            h.word(regions[index].dstOffset);
            h.word(regions[index].size);
        }
    }
    realCmdCopyBuffer(commands, source, destination, count, regions);
}

VKAPI_ATTR void VKAPI_CALL CmdPushConstants(VkCommandBuffer commands, VkPipelineLayout layout, VkShaderStageFlags stages, std::uint32_t offset, std::uint32_t size, const void* values) {
    {
        Scope scope(commands, 4);
        scope.hash.handle(layout);
        scope.hash.word(stages);
        scope.hash.word(offset);
        scope.hash.bytes(values, size);
    }
    realCmdPushConstants(commands, layout, stages, offset, size, values);
}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer commands, VkPipelineBindPoint point, VkPipelineLayout layout, std::uint32_t first, std::uint32_t count, const VkDescriptorSet* sets, std::uint32_t dynamicCount, const std::uint32_t* dynamic) {
    {
        Scope scope(commands, 5);
        auto& h = scope.hash;
        h.word(point);
        h.handle(layout);
        h.word(first);
        h.word(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            h.handle(sets[index]);
            const auto content = state().sets.find(raw(sets[index]));
            h.word(content != state().sets.end() ? content->second : 0);
        }
        h.word(dynamicCount);
        for (std::uint32_t index = 0; index < dynamicCount; ++index) h.word(dynamic[index]);
    }
    realCmdBindDescriptorSets(commands, point, layout, first, count, sets, dynamicCount, dynamic);
}

void hashBufferImageCopies(Hasher& h, std::uint32_t count, const VkBufferImageCopy* regions) {
    h.word(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto& region = regions[index];
        h.word(region.bufferOffset);
        h.word(region.bufferRowLength);
        h.word(region.bufferImageHeight);
        h.layers(region.imageSubresource);
        h.offset(region.imageOffset);
        h.extent(region.imageExtent);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer(VkCommandBuffer commands, VkImage image, VkImageLayout layout, VkBuffer buffer, std::uint32_t count, const VkBufferImageCopy* regions) {
    {
        Scope scope(commands, 6);
        scope.hash.handle(image);
        scope.hash.word(layout);
        scope.hash.handle(buffer);
        hashBufferImageCopies(scope.hash, count, regions);
    }
    realCmdCopyImageToBuffer(commands, image, layout, buffer, count, regions);
}

VKAPI_ATTR void VKAPI_CALL CmdDispatch(VkCommandBuffer commands, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    {
        Scope scope(commands, 7);
        scope.hash.word(x);
        scope.hash.word(y);
        scope.hash.word(z);
    }
    realCmdDispatch(commands, x, y, z);
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(VkCommandBuffer commands, VkBuffer buffer, VkImage image, VkImageLayout layout, std::uint32_t count, const VkBufferImageCopy* regions) {
    {
        Scope scope(commands, 8);
        scope.hash.handle(buffer);
        scope.hash.handle(image);
        scope.hash.word(layout);
        hashBufferImageCopies(scope.hash, count, regions);
    }
    realCmdCopyBufferToImage(commands, buffer, image, layout, count, regions);
}

VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, const void* data) {
    {
        Scope scope(commands, 9);
        scope.hash.handle(buffer);
        scope.hash.word(offset);
        scope.hash.bytes(data, static_cast<std::size_t>(size));
    }
    realCmdUpdateBuffer(commands, buffer, offset, size, data);
}

VKAPI_ATTR void VKAPI_CALL CmdFillBuffer(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, std::uint32_t data) {
    {
        Scope scope(commands, 10);
        scope.hash.handle(buffer);
        scope.hash.word(offset);
        scope.hash.word(size);
        scope.hash.word(data);
    }
    realCmdFillBuffer(commands, buffer, offset, size, data);
}

VKAPI_ATTR void VKAPI_CALL CmdClearColorImage(VkCommandBuffer commands, VkImage image, VkImageLayout layout, const VkClearColorValue* color, std::uint32_t count, const VkImageSubresourceRange* ranges) {
    {
        Scope scope(commands, 11);
        scope.hash.handle(image);
        scope.hash.word(layout);
        scope.hash.bytes(color, sizeof(*color));
        scope.hash.word(count);
        for (std::uint32_t index = 0; index < count; ++index) scope.hash.range(ranges[index]);
    }
    realCmdClearColorImage(commands, image, layout, color, count, ranges);
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImage(VkCommandBuffer commands, VkImage source, VkImageLayout sourceLayout, VkImage destination, VkImageLayout destinationLayout, std::uint32_t count, const VkImageCopy* regions) {
    {
        Scope scope(commands, 12);
        auto& h = scope.hash;
        h.handle(source);
        h.word(sourceLayout);
        h.handle(destination);
        h.word(destinationLayout);
        h.word(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            h.layers(regions[index].srcSubresource);
            h.offset(regions[index].srcOffset);
            h.layers(regions[index].dstSubresource);
            h.offset(regions[index].dstOffset);
            h.extent(regions[index].extent);
        }
    }
    realCmdCopyImage(commands, source, sourceLayout, destination, destinationLayout, count, regions);
}

VKAPI_ATTR void VKAPI_CALL CmdDraw(VkCommandBuffer commands, std::uint32_t vertices, std::uint32_t instances, std::uint32_t firstVertex, std::uint32_t firstInstance) {
    {
        Scope scope(commands, 13);
        scope.hash.word(vertices);
        scope.hash.word(instances);
        scope.hash.word(firstVertex);
        scope.hash.word(firstInstance);
    }
    realCmdDraw(commands, vertices, instances, firstVertex, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL CmdBeginRenderPass(VkCommandBuffer commands, const VkRenderPassBeginInfo* begin, VkSubpassContents contents) {
    {
        Scope scope(commands, 14);
        auto& h = scope.hash;
        h.handle(begin->renderPass);
        h.handle(begin->framebuffer);
        h.rect(begin->renderArea);
        h.word(begin->clearValueCount);
        for (std::uint32_t index = 0; index < begin->clearValueCount; ++index) h.bytes(&begin->pClearValues[index], sizeof(VkClearValue));
        h.word(contents);
    }
    realCmdBeginRenderPass(commands, begin, contents);
}

VKAPI_ATTR void VKAPI_CALL CmdSetViewport(VkCommandBuffer commands, std::uint32_t first, std::uint32_t count, const VkViewport* viewports) {
    {
        Scope scope(commands, 15);
        auto& h = scope.hash;
        h.word(first);
        h.word(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            h.real(viewports[index].x);
            h.real(viewports[index].y);
            h.real(viewports[index].width);
            h.real(viewports[index].height);
            h.real(viewports[index].minDepth);
            h.real(viewports[index].maxDepth);
        }
    }
    realCmdSetViewport(commands, first, count, viewports);
}

VKAPI_ATTR void VKAPI_CALL CmdSetScissor(VkCommandBuffer commands, std::uint32_t first, std::uint32_t count, const VkRect2D* scissors) {
    {
        Scope scope(commands, 16);
        scope.hash.word(first);
        scope.hash.word(count);
        for (std::uint32_t index = 0; index < count; ++index) scope.hash.rect(scissors[index]);
    }
    realCmdSetScissor(commands, first, count, scissors);
}

VKAPI_ATTR void VKAPI_CALL CmdEndRenderPass(VkCommandBuffer commands) {
    {
        Scope scope(commands, 17);
    }
    realCmdEndRenderPass(commands);
}

VKAPI_ATTR void VKAPI_CALL CmdCopyQueryPoolResults(VkCommandBuffer commands, VkQueryPool pool, std::uint32_t first, std::uint32_t count, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize stride, VkQueryResultFlags flags) {
    {
        Scope scope(commands, 18);
        auto& h = scope.hash;
        h.handle(pool);
        h.word(first);
        h.word(count);
        h.handle(buffer);
        h.word(offset);
        h.word(stride);
        h.word(flags);
    }
    realCmdCopyQueryPoolResults(commands, pool, first, count, buffer, offset, stride, flags);
}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilReference(VkCommandBuffer commands, VkStencilFaceFlags faces, std::uint32_t reference) {
    {
        Scope scope(commands, 19);
        scope.hash.word(faces);
        scope.hash.word(reference);
    }
    realCmdSetStencilReference(commands, faces, reference);
}

VKAPI_ATTR void VKAPI_CALL CmdResetQueryPool(VkCommandBuffer commands, VkQueryPool pool, std::uint32_t first, std::uint32_t count) {
    {
        Scope scope(commands, 20);
        scope.hash.handle(pool);
        scope.hash.word(first);
        scope.hash.word(count);
    }
    realCmdResetQueryPool(commands, pool, first, count);
}

VKAPI_ATTR void VKAPI_CALL CmdBlitImage(VkCommandBuffer commands, VkImage source, VkImageLayout sourceLayout, VkImage destination, VkImageLayout destinationLayout, std::uint32_t count, const VkImageBlit* regions, VkFilter filter) {
    {
        Scope scope(commands, 21);
        auto& h = scope.hash;
        h.handle(source);
        h.word(sourceLayout);
        h.handle(destination);
        h.word(destinationLayout);
        h.word(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            h.layers(regions[index].srcSubresource);
            h.offset(regions[index].srcOffsets[0]);
            h.offset(regions[index].srcOffsets[1]);
            h.layers(regions[index].dstSubresource);
            h.offset(regions[index].dstOffsets[0]);
            h.offset(regions[index].dstOffsets[1]);
        }
        h.word(filter);
    }
    realCmdBlitImage(commands, source, sourceLayout, destination, destinationLayout, count, regions, filter);
}

VKAPI_ATTR void VKAPI_CALL CmdWriteTimestamp(VkCommandBuffer commands, VkPipelineStageFlagBits stage, VkQueryPool pool, std::uint32_t query) {
    {
        Scope scope(commands, 22);
        scope.hash.word(stage);
        scope.hash.handle(pool);
        scope.hash.word(query);
    }
    realCmdWriteTimestamp(commands, stage, pool, query);
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthBounds(VkCommandBuffer commands, float minimum, float maximum) {
    {
        Scope scope(commands, 23);
        scope.hash.real(minimum);
        scope.hash.real(maximum);
    }
    realCmdSetDepthBounds(commands, minimum, maximum);
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthBias(VkCommandBuffer commands, float constant, float clamp, float slope) {
    {
        Scope scope(commands, 24);
        scope.hash.real(constant);
        scope.hash.real(clamp);
        scope.hash.real(slope);
    }
    realCmdSetDepthBias(commands, constant, clamp, slope);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexed(VkCommandBuffer commands, std::uint32_t indices, std::uint32_t instances, std::uint32_t firstIndex, std::int32_t vertexOffset, std::uint32_t firstInstance) {
    {
        Scope scope(commands, 25);
        scope.hash.word(indices);
        scope.hash.word(instances);
        scope.hash.word(firstIndex);
        scope.hash.word(static_cast<std::uint32_t>(vertexOffset));
        scope.hash.word(firstInstance);
    }
    realCmdDrawIndexed(commands, indices, instances, firstIndex, vertexOffset, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilWriteMask(VkCommandBuffer commands, VkStencilFaceFlags faces, std::uint32_t mask) {
    {
        Scope scope(commands, 26);
        scope.hash.word(faces);
        scope.hash.word(mask);
    }
    realCmdSetStencilWriteMask(commands, faces, mask);
}

VKAPI_ATTR void VKAPI_CALL CmdEndQuery(VkCommandBuffer commands, VkQueryPool pool, std::uint32_t query) {
    {
        Scope scope(commands, 27);
        scope.hash.handle(pool);
        scope.hash.word(query);
    }
    realCmdEndQuery(commands, pool, query);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirect(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset, std::uint32_t count, std::uint32_t stride) {
    {
        Scope scope(commands, 28);
        scope.hash.handle(buffer);
        scope.hash.word(offset);
        scope.hash.word(count);
        scope.hash.word(stride);
    }
    realCmdDrawIndirect(commands, buffer, offset, count, stride);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirect(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset, std::uint32_t count, std::uint32_t stride) {
    {
        Scope scope(commands, 29);
        scope.hash.handle(buffer);
        scope.hash.word(offset);
        scope.hash.word(count);
        scope.hash.word(stride);
    }
    realCmdDrawIndexedIndirect(commands, buffer, offset, count, stride);
}

VKAPI_ATTR void VKAPI_CALL CmdClearDepthStencilImage(VkCommandBuffer commands, VkImage image, VkImageLayout layout, const VkClearDepthStencilValue* value, std::uint32_t count, const VkImageSubresourceRange* ranges) {
    {
        Scope scope(commands, 30);
        scope.hash.handle(image);
        scope.hash.word(layout);
        scope.hash.real(value->depth);
        scope.hash.word(value->stencil);
        scope.hash.word(count);
        for (std::uint32_t index = 0; index < count; ++index) scope.hash.range(ranges[index]);
    }
    realCmdClearDepthStencilImage(commands, image, layout, value, count, ranges);
}

VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers(VkCommandBuffer commands, std::uint32_t first, std::uint32_t count, const VkBuffer* buffers, const VkDeviceSize* offsets) {
    {
        Scope scope(commands, 31);
        scope.hash.word(first);
        scope.hash.word(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            scope.hash.handle(buffers[index]);
            scope.hash.word(offsets[index]);
        }
    }
    realCmdBindVertexBuffers(commands, first, count, buffers, offsets);
}

VKAPI_ATTR void VKAPI_CALL CmdBindIndexBuffer(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
    {
        Scope scope(commands, 32);
        scope.hash.handle(buffer);
        scope.hash.word(offset);
        scope.hash.word(type);
    }
    realCmdBindIndexBuffer(commands, buffer, offset, type);
}

VKAPI_ATTR void VKAPI_CALL CmdBeginQuery(VkCommandBuffer commands, VkQueryPool pool, std::uint32_t query, VkQueryControlFlags flags) {
    {
        Scope scope(commands, 33);
        scope.hash.handle(pool);
        scope.hash.word(query);
        scope.hash.word(flags);
    }
    realCmdBeginQuery(commands, pool, query, flags);
}

#define APS5_LOG_STATE1(name, id, type)                                          \
    VKAPI_ATTR void VKAPI_CALL name(VkCommandBuffer commands, type value) {      \
        {                                                                        \
            Scope scope(commands, id);                                           \
            scope.hash.word(static_cast<std::uint64_t>(value));                  \
        }                                                                        \
        real##name(commands, value);                                             \
    }

APS5_LOG_STATE1(CmdSetStencilTestEnableEXT, 34, VkBool32)
APS5_LOG_STATE1(CmdSetFrontFaceEXT, 35, VkFrontFace)
APS5_LOG_STATE1(CmdSetDepthWriteEnableEXT, 36, VkBool32)
APS5_LOG_STATE1(CmdSetDepthTestEnableEXT, 37, VkBool32)
APS5_LOG_STATE1(CmdSetDepthCompareOpEXT, 38, VkCompareOp)
APS5_LOG_STATE1(CmdSetDepthBoundsTestEnableEXT, 39, VkBool32)
APS5_LOG_STATE1(CmdSetCullModeEXT, 40, VkCullModeFlags)

#undef APS5_LOG_STATE1

VKAPI_ATTR void VKAPI_CALL CmdSetStencilOpEXT(VkCommandBuffer commands, VkStencilFaceFlags faces, VkStencilOp fail, VkStencilOp pass, VkStencilOp depthFail, VkCompareOp compare) {
    {
        Scope scope(commands, 41);
        scope.hash.word(faces);
        scope.hash.word(fail);
        scope.hash.word(pass);
        scope.hash.word(depthFail);
        scope.hash.word(compare);
    }
    realCmdSetStencilOpEXT(commands, faces, fail, pass, depthFail, compare);
}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilCompareMask(VkCommandBuffer commands, VkStencilFaceFlags faces, std::uint32_t mask) {
    {
        Scope scope(commands, 42);
        scope.hash.word(faces);
        scope.hash.word(mask);
    }
    realCmdSetStencilCompareMask(commands, faces, mask);
}

VKAPI_ATTR void VKAPI_CALL CmdDispatchIndirect(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset) {
    {
        Scope scope(commands, 43);
        scope.hash.handle(buffer);
        scope.hash.word(offset);
    }
    realCmdDispatchIndirect(commands, buffer, offset);
}

VKAPI_ATTR void VKAPI_CALL CmdBeginRenderingKHR(VkCommandBuffer commands, const VkRenderingInfo* info) {
    {
        Scope scope(commands, 44);
        auto& h = scope.hash;
        h.word(info->flags);
        h.rect(info->renderArea);
        h.word(info->layerCount);
        h.word(info->viewMask);
        h.word(info->colorAttachmentCount);
        for (std::uint32_t index = 0; index < info->colorAttachmentCount; ++index) h.attachment(&info->pColorAttachments[index]);
        h.attachment(info->pDepthAttachment);
        h.attachment(info->pStencilAttachment);
    }
    realCmdBeginRenderingKHR(commands, info);
}

VKAPI_ATTR void VKAPI_CALL CmdEndRenderingKHR(VkCommandBuffer commands) {
    {
        Scope scope(commands, 45);
    }
    realCmdEndRenderingKHR(commands);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawMeshTasksIndirectEXT(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset, std::uint32_t count, std::uint32_t stride) {
    {
        Scope scope(commands, 46);
        scope.hash.handle(buffer);
        scope.hash.word(offset);
        scope.hash.word(count);
        scope.hash.word(stride);
    }
    realCmdDrawMeshTasksIndirectEXT(commands, buffer, offset, count, stride);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawMeshTasksEXT(VkCommandBuffer commands, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    {
        Scope scope(commands, 47);
        scope.hash.word(x);
        scope.hash.word(y);
        scope.hash.word(z);
    }
    realCmdDrawMeshTasksEXT(commands, x, y, z);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirectCountKHR(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countOffset, std::uint32_t maximum, std::uint32_t stride) {
    {
        Scope scope(commands, 48);
        scope.hash.handle(buffer);
        scope.hash.word(offset);
        scope.hash.handle(countBuffer);
        scope.hash.word(countOffset);
        scope.hash.word(maximum);
        scope.hash.word(stride);
    }
    realCmdDrawIndirectCountKHR(commands, buffer, offset, countBuffer, countOffset, maximum, stride);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirectCountKHR(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countOffset, std::uint32_t maximum, std::uint32_t stride) {
    {
        Scope scope(commands, 49);
        scope.hash.handle(buffer);
        scope.hash.word(offset);
        scope.hash.handle(countBuffer);
        scope.hash.word(countOffset);
        scope.hash.word(maximum);
        scope.hash.word(stride);
    }
    realCmdDrawIndexedIndirectCountKHR(commands, buffer, offset, countBuffer, countOffset, maximum, stride);
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* buffers) {
    const auto result = realAllocateCommandBuffers(device, info, buffers);
    if (result != VK_SUCCESS) return result;
    auto& log = state();
    std::lock_guard lock(log.mutex);
    const auto [pool, inserted] = log.pools.try_emplace(raw(info->commandPool), static_cast<std::uint32_t>(log.pools.size() + 1));
    for (std::uint32_t index = 0; index < info->commandBufferCount; ++index) log.buffers[buffers[index]] = Buffer{FnvOffset, 0, pool->second};
    return result;
}

void restart(VkCommandBuffer commands) {
    auto& log = state();
    std::lock_guard lock(log.mutex);
    auto& buffer = log.buffers[commands];
    buffer.hash = FnvOffset;
    buffer.commands = 0;
}

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer commands, const VkCommandBufferBeginInfo* info) {
    restart(commands);
    return realBeginCommandBuffer(commands, info);
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(VkCommandBuffer commands, VkCommandBufferResetFlags flags) {
    restart(commands);
    return realResetCommandBuffer(commands, flags);
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
    {
        auto& log = state();
        std::lock_guard lock(log.mutex);
        for (std::uint32_t submit = 0; submit < count; ++submit) {
            for (std::uint32_t index = 0; index < submits[submit].commandBufferCount; ++index) {
                const auto& buffer = log.buffers[submits[submit].pCommandBuffers[index]];
                auto& stream = log.streams[buffer.pool];
                Hasher mix{log, stream.hash};
                mix.word(buffer.hash);
                mix.word(buffer.commands);
                stream.hash = mix.value;
                stream.commands += buffer.commands;
            }
        }
    }
    return realQueueSubmit(queue, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* info) {
    std::string line;
    {
        auto& log = state();
        std::lock_guard lock(log.mutex);
        line = lineLocked(log, ++log.presents);
        log.streams.clear();
        log.ordinals.clear();
    }
    writeLine(line);
    return realQueuePresentKHR(queue, info);
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(VkDevice device, std::uint32_t writeCount, const VkWriteDescriptorSet* writes, std::uint32_t copyCount, const VkCopyDescriptorSet* copies) {
    {
        auto& log = state();
        std::lock_guard lock(log.mutex);
        for (std::uint32_t index = 0; index < writeCount; ++index) {
            const auto& write = writes[index];
            auto& content = log.sets[raw(write.dstSet)];
            Hasher h{log, content == 0 ? FnvOffset : content};
            h.word(write.dstBinding);
            h.word(write.dstArrayElement);
            h.word(write.descriptorCount);
            h.word(write.descriptorType);
            for (std::uint32_t element = 0; element < write.descriptorCount; ++element) {
                if (write.pImageInfo != nullptr) {
                    h.handle(write.pImageInfo[element].sampler);
                    h.handle(write.pImageInfo[element].imageView);
                    h.word(write.pImageInfo[element].imageLayout);
                } else if (write.pBufferInfo != nullptr) {
                    h.handle(write.pBufferInfo[element].buffer);
                    h.word(write.pBufferInfo[element].offset);
                    h.word(write.pBufferInfo[element].range);
                } else if (write.pTexelBufferView != nullptr) {
                    h.handle(write.pTexelBufferView[element]);
                }
            }
            content = h.value;
        }
        for (std::uint32_t index = 0; index < copyCount; ++index) {
            const auto& copy = copies[index];
            const auto source = log.sets.find(raw(copy.srcSet));
            auto& content = log.sets[raw(copy.dstSet)];
            Hasher h{log, content == 0 ? FnvOffset : content};
            h.word(source != log.sets.end() ? source->second : 0);
            h.word(copy.srcBinding);
            h.word(copy.srcArrayElement);
            h.word(copy.dstBinding);
            h.word(copy.dstArrayElement);
            h.word(copy.descriptorCount);
            content = h.value;
        }
    }
    realUpdateDescriptorSets(device, writeCount, writes, copyCount, copies);
}

struct Hook {
    const char* name;
    PFN_vkVoidFunction thunk;
    PFN_vkVoidFunction* real;
};

#define APS5_LOG_HOOK(name) Hook{"vk" #name, reinterpret_cast<PFN_vkVoidFunction>(&name), reinterpret_cast<PFN_vkVoidFunction*>(&real##name)}

const Hook Hooks[] = {
    APS5_LOG_HOOK(CmdPipelineBarrier), APS5_LOG_HOOK(CmdBindPipeline), APS5_LOG_HOOK(CmdCopyBuffer), APS5_LOG_HOOK(CmdPushConstants),
    APS5_LOG_HOOK(CmdBindDescriptorSets), APS5_LOG_HOOK(CmdCopyImageToBuffer), APS5_LOG_HOOK(CmdDispatch), APS5_LOG_HOOK(CmdCopyBufferToImage),
    APS5_LOG_HOOK(CmdUpdateBuffer), APS5_LOG_HOOK(CmdFillBuffer), APS5_LOG_HOOK(CmdClearColorImage), APS5_LOG_HOOK(CmdCopyImage),
    APS5_LOG_HOOK(CmdDraw), APS5_LOG_HOOK(CmdBeginRenderPass), APS5_LOG_HOOK(CmdSetViewport), APS5_LOG_HOOK(CmdSetScissor),
    APS5_LOG_HOOK(CmdEndRenderPass), APS5_LOG_HOOK(CmdCopyQueryPoolResults), APS5_LOG_HOOK(CmdSetStencilReference), APS5_LOG_HOOK(CmdResetQueryPool),
    APS5_LOG_HOOK(CmdBlitImage), APS5_LOG_HOOK(CmdWriteTimestamp), APS5_LOG_HOOK(CmdSetDepthBounds), APS5_LOG_HOOK(CmdSetDepthBias),
    APS5_LOG_HOOK(CmdDrawIndexed), APS5_LOG_HOOK(CmdSetStencilWriteMask), APS5_LOG_HOOK(CmdEndQuery), APS5_LOG_HOOK(CmdDrawIndirect),
    APS5_LOG_HOOK(CmdDrawIndexedIndirect), APS5_LOG_HOOK(CmdClearDepthStencilImage), APS5_LOG_HOOK(CmdBindVertexBuffers), APS5_LOG_HOOK(CmdBindIndexBuffer),
    APS5_LOG_HOOK(CmdBeginQuery), APS5_LOG_HOOK(CmdSetStencilTestEnableEXT), APS5_LOG_HOOK(CmdSetStencilOpEXT), APS5_LOG_HOOK(CmdSetStencilCompareMask),
    APS5_LOG_HOOK(CmdSetFrontFaceEXT), APS5_LOG_HOOK(CmdSetDepthWriteEnableEXT), APS5_LOG_HOOK(CmdSetDepthTestEnableEXT), APS5_LOG_HOOK(CmdSetDepthCompareOpEXT),
    APS5_LOG_HOOK(CmdSetDepthBoundsTestEnableEXT), APS5_LOG_HOOK(CmdSetCullModeEXT), APS5_LOG_HOOK(CmdDispatchIndirect), APS5_LOG_HOOK(CmdBeginRenderingKHR),
    APS5_LOG_HOOK(CmdEndRenderingKHR), APS5_LOG_HOOK(CmdDrawMeshTasksIndirectEXT), APS5_LOG_HOOK(CmdDrawMeshTasksEXT), APS5_LOG_HOOK(CmdDrawIndirectCountKHR),
    APS5_LOG_HOOK(CmdDrawIndexedIndirectCountKHR), APS5_LOG_HOOK(AllocateCommandBuffers), APS5_LOG_HOOK(BeginCommandBuffer), APS5_LOG_HOOK(ResetCommandBuffer),
    APS5_LOG_HOOK(QueueSubmit), APS5_LOG_HOOK(QueuePresentKHR), APS5_LOG_HOOK(UpdateDescriptorSets),
};

#undef APS5_LOG_HOOK

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Proc(VkDevice device, const char* name) {
    const auto real = state().real(device, name);
    if (name == nullptr || real == nullptr) return real;
    const std::string_view wanted(name);
    for (const auto& hook : Hooks) {
        if (wanted != hook.name) continue;
        *hook.real = real;
        return hook.thunk;
    }
    return real;
}

}

bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_COMMAND_LOG");
        return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

PFN_vkGetDeviceProcAddr Wrap(VkDevice device, PFN_vkGetDeviceProcAddr real, bool force) {
    if ((!Enabled() && !force) || real == nullptr || device == VK_NULL_HANDLE) return real;
    state().real = real;
    return &Proc;
}

std::vector<Stream> Streams() {
    auto& log = state();
    std::lock_guard lock(log.mutex);
    std::vector<Stream> streams;
    for (const auto& [pool, stream] : log.streams) streams.push_back({pool, stream.commands, stream.hash});
    return streams;
}

std::string Line(std::uint64_t present) {
    auto& log = state();
    std::lock_guard lock(log.mutex);
    return lineLocked(log, present);
}

void Reset() {
    auto& log = state();
    std::lock_guard lock(log.mutex);
    log.streams.clear();
    log.ordinals.clear();
    log.sets.clear();
}

}
