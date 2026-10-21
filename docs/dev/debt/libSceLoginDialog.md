# libSceLoginDialog technical debt

### Silent stubs

- [libSceLoginDialog](../../../core/libs/prx/libSceLoginDialog/Export.cpp) - the login dialog runs the common-dialog state machine but shows nothing and reports the user as having cancelled
- [sceLoginDialogClose](../../../core/libs/prx/libSceLoginDialog/Export.cpp) - sceLoginDialogClose transitions an initialized dialog directly to COMMON_DIALOG_STATUS_FINISHED even when called before Open, following the Signin / NetCtlAp / PlayerSelection convention; this behavior is unverified against real hardware.
