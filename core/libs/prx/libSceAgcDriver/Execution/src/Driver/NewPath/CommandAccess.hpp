#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_COMMANDACCESS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_COMMANDACCESS_HPP

#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandWrites.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace AgcDriver::NewPath {

struct TableLists {
    std::shared_ptr<const Graphics::Recorder::ReadList> mapped;
    std::shared_ptr<const Graphics::Recorder::ReadList> writable;
};

[[nodiscard]] TableLists ListsOf(std::span<const ShaderRecompiler::BdaAbi::Range> ranges);

struct AccessNotes {
    std::uint64_t boundedRanges = 0;
    bool leasedWritable = false;
    bool notedReads = false;
};

AccessNotes NoteCommandAccess(Graphics::Recorder& recorder, const WriteSet& writes, const TableLists& lists, bool readsAnything);

bool UploadTableForLane(const Graphics::Context& context, Graphics::Recorder& recorder, const std::shared_ptr<const Graphics::BdaPageTable>& table, std::uint32_t lane);

}

#endif
