#include "prx/libSceAgcDriver/Graphics/include/ResidencyTokens.hpp"
#include <algorithm>
#include <stdexcept>

namespace AgcDriver::Graphics {

static_assert(ResidencyLaneSlots <= ResidencyStreams);

std::uint64_t ResidencyTokens::Through(const ResidencyToken& token) {
    std::uint64_t through = 0;
    for (std::size_t slot = 0; slot < ResidencyLaneSlots; ++slot) through = std::max(through, token[slot]);
    for (std::size_t slot = ResidencyLaneSlots; slot < token.size(); ++slot) {
        if (token[slot] != 0) throw std::logic_error("residency token names a stream past the recorder's two lanes");
    }
    return through;
}

ResidencyToken ResidencyTokens::Caller() const {
    if (lanes.NativeTokens()) return lanes.CallerToken();
    const auto lane = lanes.CallerLane();
    if (lane < 0 || static_cast<std::size_t>(lane) >= ResidencyLaneSlots) throw std::logic_error("residency: the calling thread's lane has no token slot");
    ResidencyToken token{};
    token[static_cast<std::size_t>(lane)] = lanes.OpenSerial();
    return token;
}

ResidencyToken ResidencyTokens::All() const {
    if (lanes.NativeTokens()) return lanes.AllToken();
    ResidencyToken token{};
    const auto open = lanes.OpenSerial();
    for (std::size_t slot = 0; slot < ResidencyLaneSlots; ++slot) token[slot] = open;
    return token;
}

bool ResidencyTokens::Reached(const ResidencyToken& token) const {
    if (lanes.NativeTokens()) return lanes.TokenReached(token);
    return Through(token) <= lanes.CompletedThrough();
}

void ResidencyTokens::Finish(const ResidencyToken& token) {
    if (lanes.NativeTokens()) {
        lanes.FinishToken(token);
        return;
    }
    if (Reached(token)) return;
    lanes.SubmitAll();
    const auto submitted = lanes.SubmittedSerial();
    lanes.WaitThrough(submitted);
    lanes.Complete(submitted);
    if (!Reached(token)) throw std::logic_error("residency: a token not reached after every submitted batch completed");
}

bool ResidencyTokens::WaitUnlocked(const ResidencyToken& token) {
    return Reached(token);
}

}
