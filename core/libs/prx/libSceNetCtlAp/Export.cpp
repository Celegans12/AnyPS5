#include <cstdint>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

extern "C" {

int APS5_VABI sceNetCtlApInit() {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

void APS5_VABI sceNetCtlApTerm() {
    NotImplemented_nid_no_patch(__func__);
}

int APS5_VABI sceNetCtlApCheckCallback() {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceNetCtlApClearEvent(int eventType) {
    (void)eventType;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceNetCtlApRegisterCallback(void (*func)(int, void*), void* arg, int* cid) {
    (void)func;
    (void)arg;
    (void)cid;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceNetCtlApUnregisterCallback(int cid) {
    (void)cid;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceNetCtlApGetResult(int eventType, int* errorCode) {
    (void)eventType;
    (void)errorCode;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceNetCtlApGetInfo(void* info) {
    (void)info;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceNetCtlApGetState(int* state) {
    (void)state;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceNetCtlApStop() {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceNetCtlApGetConnectInfo(std::uint8_t* optStr, void* connectInfo) {
    (void)optStr;
    (void)connectInfo;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

}
