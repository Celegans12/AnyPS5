#include "Ngs2Test.hpp"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>

static constexpr float SoundSpeed = 343.0f;

static void TestLayouts() {
    Require(sizeof(Ngs2GeomVector) == 12);
    Require(sizeof(Ngs2GeomCone) == 16);
    Require(sizeof(Ngs2GeomRolloff) == 16);
    Require(sizeof(Ngs2GeomListenerParam) == 60);
    Require(sizeof(Ngs2GeomListenerWork) == 96);
    Require(sizeof(Ngs2GeomSourceParam) == 108);
    Require(sizeof(Ngs2GeomA3dAttribute) == 32);
    Require(sizeof(Ngs2GeomAttribute) == 308);
}

static void TestResetListenerParam() {
    Ngs2GeomListenerParam param;
    std::memset(&param, 0x5a, sizeof(param));
    Require(sceNgs2GeomResetListenerParam(&param) == SCE_NGS2_OK);
    Require(param.position.x == 0.0f && param.position.y == 0.0f && param.position.z == 0.0f);
    Require(param.velocity.x == 0.0f && param.velocity.y == 0.0f && param.velocity.z == 0.0f);
    Require(param.orient_front.x == 0.0f && param.orient_front.y == 0.0f && param.orient_front.z == 1.0f);
    Require(param.orient_up.x == 0.0f && param.orient_up.y == 1.0f && param.orient_up.z == 0.0f);
    Require(param.sound_speed == SoundSpeed);
    Require(param.reserved[0] == 0u && param.reserved[1] == 0u);
    Require(sceNgs2GeomResetListenerParam(nullptr) == SCE_NGS2_ERROR_INVALID_OUT_ADDRESS);
}

static void TestResetSourceParam() {
    Ngs2GeomSourceParam param;
    std::memset(&param, 0x5a, sizeof(param));
    Require(sceNgs2GeomResetSourceParam(&param) == SCE_NGS2_OK);
    Require(param.position.x == 0.0f && param.position.y == 0.0f && param.position.z == 0.0f);
    Require(param.velocity.x == 0.0f && param.velocity.y == 0.0f && param.velocity.z == 0.0f);
    Require(param.direction.x == 0.0f && param.direction.y == 0.0f && param.direction.z == 1.0f);
    Require(param.cone.inner_level == 1.0f && param.cone.inner_angle == 360.0f);
    Require(param.cone.outer_level == 1.0f && param.cone.outer_angle == 360.0f);
    Require(param.rolloff.model == 0u);
    Require(param.rolloff.max_distance == 1000000.0f);
    Require(param.rolloff.rolloff_factor == 1.0f);
    Require(param.rolloff.reference_distance == 1.0f);
    Require(param.doppler_factor == 1.0f && param.fbw_level == 1.0f && param.lfe_level == 1.0f);
    Require(param.max_level == 1.0f && param.min_level == 0.0f && param.radius == 0.0f);
    Require(param.num_speakers == 2u && param.matrix_format == 2u);
    Require(param.reserved[0] == 0u && param.reserved[1] == 0u);
    Require(sceNgs2GeomResetSourceParam(nullptr) == SCE_NGS2_ERROR_INVALID_OUT_ADDRESS);
}

static Ngs2GeomListenerParam Listener() {
    Ngs2GeomListenerParam param{};
    Require(sceNgs2GeomResetListenerParam(&param) == SCE_NGS2_OK);
    return param;
}

static void RequireMatrix(const Ngs2GeomListenerWork& work, const float (&expected)[4][4]) {
    for (std::size_t row = 0; row < 4; row++) {
        for (std::size_t column = 0; column < 4; column++) Require(work.matrix[row][column] == expected[row][column]);
    }
}

static void TestDefaultListener() {
    auto param = Listener();
    Ngs2GeomListenerWork work;
    std::memset(&work, 0x5a, sizeof(work));
    Require(sceNgs2GeomCalcListener(&param, &work, 0) == SCE_NGS2_OK);
    RequireMatrix(work, {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}});
    Require(work.velocity.x == 0.0f && work.velocity.y == 0.0f && work.velocity.z == 0.0f);
    Require(work.sound_speed == SoundSpeed && work.coordinate == 0);
    Require(work.reserved[0] == 0 && work.reserved[1] == 0 && work.reserved[2] == 0);

    param.velocity = {2.0f, -3.0f, 4.0f};
    param.sound_speed = 1500.0f;
    std::memset(&work, 0x5a, sizeof(work));
    Require(sceNgs2GeomCalcListener(&param, &work, 0) == SCE_NGS2_OK);
    RequireMatrix(work, {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}});
    Require(work.velocity.x == 2.0f && work.velocity.y == -3.0f && work.velocity.z == 4.0f);
    Require(work.sound_speed == 1500.0f && work.coordinate == 0);
    Require(work.reserved[0] == 0 && work.reserved[1] == 0 && work.reserved[2] == 0);

    param.position = {-0.0f, -0.0f, -0.0f};
    param.orient_front = {-0.0f, -0.0f, 1.0f};
    param.orient_up = {-0.0f, 1.0f, -0.0f};
    Require(sceNgs2GeomCalcListener(&param, &work, 0) == SCE_NGS2_OK);
    RequireMatrix(work, {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}});
}

static void RejectListener(const Ngs2GeomListenerParam* param, std::uint32_t flags) {
    Ngs2GeomListenerWork work;
    std::memset(&work, 0x5a, sizeof(work));
    unsigned char before[sizeof(work)];
    std::memcpy(before, &work, sizeof(work));
    bool rejected = false;
    try {
        sceNgs2GeomCalcListener(param, &work, flags);
    } catch (const std::exception&) {
        rejected = true;
    }
    Require(rejected);
    Require(std::memcmp(before, &work, sizeof(work)) == 0);
}

static void TestUnsupportedListener() {
    auto param = Listener();
    for (const auto flags : {1u, 2u, 0xffffffffu}) RejectListener(&param, flags);
    for (const auto vector : {&Ngs2GeomListenerParam::position, &Ngs2GeomListenerParam::orient_front, &Ngs2GeomListenerParam::orient_up}) {
        for (const auto component : {&Ngs2GeomVector::x, &Ngs2GeomVector::y, &Ngs2GeomVector::z}) {
            for (const float value : {-1.0f, 0.0f, 1.0f, 2.0f, std::numeric_limits<float>::min(), std::numeric_limits<float>::max()}) {
                param = Listener();
                auto& field = (param.*vector).*component;
                if (field == value) continue;
                field = value;
                RejectListener(&param, 0);
            }
            param = Listener();
            auto& field = (param.*vector).*component;
            field = std::nextafter(field, 2.0f);
            RejectListener(&param, 0);
        }
    }
}

static void TestInvalidListener() {
    auto param = Listener();
    Require(sceNgs2GeomCalcListener(&param, nullptr, 0) == SCE_NGS2_ERROR_INVALID_OUT_ADDRESS);
    RejectListener(nullptr, 0);
    for (const float value : {0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        param.sound_speed = value;
        RejectListener(&param, 0);
    }
    param = Listener();
    param.orient_front = {};
    RejectListener(&param, 0);
    param = Listener();
    param.orient_up = param.orient_front;
    RejectListener(&param, 0);
    param = Listener();
    param.orient_up = {};
    RejectListener(&param, 0);
    for (const auto vector : {&Ngs2GeomListenerParam::position, &Ngs2GeomListenerParam::orient_front,
                              &Ngs2GeomListenerParam::orient_up, &Ngs2GeomListenerParam::velocity}) {
        for (const auto component : {&Ngs2GeomVector::x, &Ngs2GeomVector::y, &Ngs2GeomVector::z}) {
            for (const float value : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
                param = Listener();
                (param.*vector).*component = value;
                RejectListener(&param, 0);
            }
        }
    }
}

int main() {
    TestLayouts();
    TestResetListenerParam();
    TestResetSourceParam();
    TestDefaultListener();
    TestUnsupportedListener();
    TestInvalidListener();
    return 0;
}
