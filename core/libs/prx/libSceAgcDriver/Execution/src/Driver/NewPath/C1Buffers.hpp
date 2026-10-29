#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_C1BUFFERS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_C1BUFFERS_HPP

#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/DevicePageTable.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/NewPath.hpp"

// S2-share C1 on the new path (driver side): F4 over the guest-buffer ranges of a build whose twin reads its buffer
// V#s at run time through the device page table.
namespace AgcDriver::NewPath {

// The build's guest-buffer ranges onto the candidate (after the build, before the record); null when the build
// cannot give them.
void NoteBufferRanges(EmitCandidate& candidate, const Graphics::ShaderResources& resources);
// F4: every guest-buffer range of the build maps in the device page table to the address the build binds, with its
// permissions (Write where the build writes in place). Fills the facts' cover and reason; `used` as Covers.
// threadLimit: a thread-dimension dispatch, whose limit the build writes into today's ShaderData (the twin must
// bind it, so its layout is today's: TwinDifference), else C1Layout.
void CoverBuffers(const EmitCandidate& candidate, const Graphics::ShaderResources& resources, CommandFacts& facts, DevicePageTable::Snapshot* used, bool threadLimit);

}

#endif
