#pragma once

#include "SceTypes.hpp"

extern "C" {

int APS5_VABI sceNpSessionSignalingCrossplayCreateContext(void);
int APS5_VABI sceNpSessionSignalingCrossplayDestroyContext(void);
int APS5_VABI sceNpSessionSignalingCrossplayTerminate(void);
int APS5_VABI sceNpSessionSignalingCrossplayGetMemoryInfo(void);
int APS5_VABI sceNpSessionSignalingCrossplayInitialize(void);

}
