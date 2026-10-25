#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PARALLELPROFILE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PARALLELPROFILE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace AgcDriver::Graphics::ParallelProfile {

bool On();
void MarkCommitter();
bool OnCommitter();

void NoteDraw(std::span<const double> phaseUs, std::span<const double> movable, std::span<const char* const> names);
void NotePassDraw(bool continued);
void NoteSerialWork();

enum class Item : std::uint8_t { Draw, Label, Other, Count };
struct ItemSample {
    Item kind = Item::Other;
    double wallUs = 0;
    double heldUs = 0;
    double lockedWaitUs = 0;
    double preDrainUs = 0;
    bool preDrained = false;
    bool stageA = false;
};
void NoteItem(const ItemSample& sample);

enum class Stamp : std::uint8_t { Forget, Registry, Device, Epoch, Proof, Pending, GpuWrite, CpuWrite, Count };
enum class Window : std::uint8_t { Queue, Commit, Count };
constexpr std::uint32_t StampBit(Stamp stamp) { return 1u << static_cast<std::uint32_t>(stamp); }
void NoteStamps(Window window, std::uint32_t moved);

struct GateInputs {
    double movableShare = 0;
    double survival = 0;
    double chunkP50 = 0;
};
struct GateVerdict {
    bool movable = false;
    bool survival = false;
    bool chunks = false;
    bool Pass() const { return movable && survival && chunks; }
};
GateVerdict Gate(const GateInputs& inputs);
double Percentile(std::span<const std::uint64_t> histogram, double fraction);

void Report();

}

#endif
