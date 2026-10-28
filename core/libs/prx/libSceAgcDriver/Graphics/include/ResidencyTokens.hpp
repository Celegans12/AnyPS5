#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYTOKENS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYTOKENS_HPP

#include "prx/libSceAgcDriver/Graphics/include/Residency.hpp"
#include <cstdint>
#include <memory>

namespace AgcDriver::Graphics {

class Recorder;

inline constexpr std::size_t ResidencyLaneSlots = 2;

class IResidencyLanes {
public:
    virtual ~IResidencyLanes() = default;
    virtual int CallerLane() const = 0;
    virtual std::uint64_t OpenSerial() const = 0;
    virtual std::uint64_t SubmittedSerial() const = 0;
    virtual std::uint64_t CompletedThrough() const = 0;
    virtual void SubmitAll() = 0;
    virtual void WaitThrough(std::uint64_t serial) = 0;
    virtual void Complete(std::uint64_t serial) = 0;
};

class ResidencyTokens {
public:
    explicit ResidencyTokens(IResidencyLanes& lanes) : lanes(lanes) {}

    static std::uint64_t Through(const ResidencyToken& token);
    ResidencyToken Caller() const;
    ResidencyToken All() const;
    bool Reached(const ResidencyToken& token) const;
    void Finish(const ResidencyToken& token);
    bool WaitUnlocked(const ResidencyToken& token);

private:
    IResidencyLanes& lanes;
};

std::unique_ptr<IResidencyLanes> RecorderLanes(Recorder& recorder);

}

#endif
