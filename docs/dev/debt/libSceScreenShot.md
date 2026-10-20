# libSceScreenShot technical debt

### Unknown function info

- [sceScreenShotSetOverlayImage and sceScreenShotSetOverlayImageWithOrigin](../../../core/libs/prx/libSceScreenShot/Export.cpp) (libSceScreenShot) - names and NIDs are present in the [PS5 3.20 export inventory](https://github.com/DNNDHH/PS5-3.20_Libs/blob/b2a77c09cd71c01f34c790cfbe98c751a13262ad/libSceScreenShot.c). The PS4 compatibility implementation also exposes the names but declares no parameters; no PS5 prototype was found. The declarations have no parameters solely to resolve known imports and make no ABI claim
