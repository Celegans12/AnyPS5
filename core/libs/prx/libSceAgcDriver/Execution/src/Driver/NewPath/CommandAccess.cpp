#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandAccess.hpp"
#include <vector>

namespace AgcDriver::NewPath {

TableLists ListsOf(std::span<const ShaderRecompiler::BdaAbi::Range> ranges) {
    std::vector<AddressRange> mapped;
    mapped.reserve(ranges.size());
    for (const auto& range : ranges) mapped.emplace_back(range.begin, range.end);
    TableLists lists;
    lists.mapped = std::make_shared<const Graphics::Recorder::ReadList>(MergeRanges(std::move(mapped)));
    lists.writable = std::make_shared<const Graphics::Recorder::ReadList>(WritableRanges(ranges));
    return lists;
}

AccessNotes NoteCommandAccess(Graphics::Recorder& recorder, const WriteSet& writes, const TableLists& lists, bool readsAnything) {
    AccessNotes notes;
    if (readsAnything && lists.mapped != nullptr && !lists.mapped->empty()) {
        recorder.NotePendingReadList(lists.mapped, Graphics::Recorder::ReadKind::AddressBasedReadOnly);
        notes.notedReads = true;
    }
    if (writes.unbounded) {
        if (lists.writable != nullptr && !lists.writable->empty()) {
            recorder.NotePendingReadList(lists.writable, Graphics::Recorder::ReadKind::AddressBased);
            notes.leasedWritable = true;
        }
        return notes;
    }
    if (!writes.ranges.empty()) {
        recorder.NotePendingWrites(writes.ranges);
        notes.boundedRanges = writes.ranges.size();
    }
    return notes;
}

bool UploadTableForLane(const Graphics::Context& context, Graphics::Recorder& recorder, const std::shared_ptr<const Graphics::BdaPageTable>& table, std::uint32_t lane) {
    if (table == nullptr || !table->NeedsUploads(lane)) return false;
    auto commands = recorder.PreludeCommands();
    const bool prelude = commands != VK_NULL_HANDLE;
    if (!prelude) commands = recorder.Commands();
    if (!table->RecordUploads(context, commands, lane)) return false;
    recorder.Keep(std::const_pointer_cast<Graphics::BdaPageTable>(table));
    if (prelude) return true;
    VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkAccessFlags sourceAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
    VkAccessFlags destinationAccess = VK_ACCESS_SHADER_READ_BIT;
    Graphics::Recorder::FilterLaneBarrier(sourceStage, destinationStage, sourceAccess, destinationAccess);
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, sourceAccess, destinationAccess};
    context.Resolved(&Graphics::DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, sourceStage, destinationStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return true;
}

}
