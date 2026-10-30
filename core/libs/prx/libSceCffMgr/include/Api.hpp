#pragma once

#include "SceTypes.hpp"

extern "C" {

int APS5_VABI sceConsoleFeatureFlagManagerDebugInit(void);
int APS5_VABI sceConsoleFeatureFlagManagerEventClose(void);
int APS5_VABI sceConsoleFeatureFlagManagerGetVersion(void);
int APS5_VABI sceConsoleFeatureFlagManagerEventOpen(void);
int APS5_VABI sceConsoleFeatureFlagManagerSetOn(void);
int APS5_VABI sceConsoleFeatureFlagManagerIsOn(void);
int APS5_VABI sceConsoleFeatureFlagManagerIsWaitingReboot(void);

}
