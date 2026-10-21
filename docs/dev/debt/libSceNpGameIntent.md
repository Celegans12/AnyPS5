# libSceNpGameIntent technical debt

### Silent stubs

- [sceNpGameIntentVshInitialize, sceNpGameIntentVshTerminate](../../../core/libs/prx/libSceNpGameIntent/Export.cpp) (libSceNpGameIntent) - VSH initialization and termination return 0 without performing shell operations, matching `sceNpGameIntentInitialize` and `sceNpGameIntentTerminate`

### Unknown function info

- [sceNpGameIntentInitParamInit](../../../core/libs/prx/libSceNpGameIntent/Export.cpp) (libSceNpGameIntent) - struct layout matches [fpPS4](https://github.com/red-prig/fpPS4/blob/master/src/np/ps4_libscenpgameintent.pas#L17) `SceNpGameIntentInitParam`; setting `size`, zeroing reserved bytes, and throwing on null pointer follow Sony SDK `*ParamInit` conventions but are unverified on retail hardware
- [sceNpGameIntentInfoInit](../../../core/libs/prx/libSceNpGameIntent/Export.cpp) (libSceNpGameIntent) - struct layout matches [fpPS4](https://github.com/red-prig/fpPS4/blob/master/src/np/ps4_libscenpgameintent.pas#L28) `SceNpGameIntentInfo`; setting `size`, `user_id = -1`, zeroing fields, and throwing on null pointer match `sceNpGameIntentReceiveIntent` defaults but are unverified on retail hardware
