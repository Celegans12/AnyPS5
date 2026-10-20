# libSceAutoMounterClient technical debt

### Unknown function info

- [libSceAutoMounterClient](../../../core/libs/prx/libSceAutoMounterClient/Export.cpp) - the three names and NIDs are in the [PS5 3.20 export inventory](https://github.com/DNNDHH/PS5-3.20_Libs/blob/b2a77c09cd71c01f34c790cfbe98c751a13262ad/libSceAutoMounterClient.c), but no credible prototypes were found. The declarations have no parameters solely to resolve known imports; this is not an ABI claim and the exports must not be called until their signatures are verified
