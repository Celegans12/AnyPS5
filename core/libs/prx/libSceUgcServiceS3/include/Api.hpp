#pragma once

#include "SceTypes.hpp"

extern "C" {

int APS5_VABI sceUgcServiceS3DownloadFileWithGmsResponse(void);
int APS5_VABI sceUgcServiceS3CreateRequest(void);
int APS5_VABI sceUgcServiceS3UploadFileWithGmsResponse(void);
int APS5_VABI sceUgcServiceS3CancelRequest(void);

}
