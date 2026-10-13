#include <cmath>
#include <cstdint>

#include "prx/libc/include/General.hpp"
#include "Ngs2Internal.hpp"

static constexpr float DEFAULT_SOUND_SPEED = 343.0f;

int Ngs2GeomCalcListener(const Ngs2GeomListenerParam* param, Ngs2GeomListenerWork* outWork, std::uint32_t flags) {
    if (param == nullptr) APS5_INVALID_ARG_EX;
    if (outWork == nullptr) return SCE_NGS2_ERROR_INVALID_OUT_ADDRESS;
    if (flags != 0) throw std::runtime_error("NGS2: listener coordinate flags are unsupported");
    if (param->position.x != 0.0f || param->position.y != 0.0f || param->position.z != 0.0f)
        throw std::runtime_error("NGS2: listener translation is unsupported");
    if (param->orient_front.x != 0.0f || param->orient_front.y != 0.0f || param->orient_front.z != 1.0f ||
        param->orient_up.x != 0.0f || param->orient_up.y != 1.0f || param->orient_up.z != 0.0f)
        throw std::runtime_error("NGS2: non-default listener orientation is unsupported");
    if (!std::isfinite(param->velocity.x) || !std::isfinite(param->velocity.y) || !std::isfinite(param->velocity.z)) APS5_INVALID_ARG_EX;
    if (!std::isfinite(param->sound_speed) || param->sound_speed <= 0.0f) APS5_INVALID_ARG_EX;
    Ngs2GeomListenerWork work{};
    for (std::size_t i = 0; i < 4; i++) work.matrix[i][i] = 1.0f;
    work.velocity = param->velocity;
    work.sound_speed = param->sound_speed;
    *outWork = work;
    return SCE_NGS2_OK;
}

int Ngs2GeomResetListenerParam(Ngs2GeomListenerParam* out_listener_param) {
    if (out_listener_param == nullptr) return SCE_NGS2_ERROR_INVALID_OUT_ADDRESS;
    *out_listener_param = {};
    out_listener_param->orient_front.z = 1.0f;
    out_listener_param->orient_up.y = 1.0f;
    out_listener_param->sound_speed = DEFAULT_SOUND_SPEED;
    return SCE_NGS2_OK;
}

int Ngs2GeomResetSourceParam(Ngs2GeomSourceParam* out_source_param) {
    if (out_source_param == nullptr) return SCE_NGS2_ERROR_INVALID_OUT_ADDRESS;
    *out_source_param = {};
    out_source_param->direction.z = 1.0f;
    out_source_param->cone.inner_level = 1.0f;
    out_source_param->cone.inner_angle = 360.0f;
    out_source_param->cone.outer_level = 1.0f;
    out_source_param->cone.outer_angle = 360.0f;
    out_source_param->rolloff.max_distance = 1000000.0f;
    out_source_param->rolloff.rolloff_factor = 1.0f;
    out_source_param->rolloff.reference_distance = 1.0f;
    out_source_param->doppler_factor = 1.0f;
    out_source_param->fbw_level = 1.0f;
    out_source_param->lfe_level = 1.0f;
    out_source_param->max_level = 1.0f;
    out_source_param->num_speakers = 2;
    out_source_param->matrix_format = 2;
    return SCE_NGS2_OK;
}
