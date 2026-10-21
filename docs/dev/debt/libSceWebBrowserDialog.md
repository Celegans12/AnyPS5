# libSceWebBrowserDialog technical debt

### Silent stubs

- [libSceWebBrowserDialog](../../../core/libs/prx/libSceWebBrowserDialog/Export.cpp) - `sceWebBrowserDialogOpen` and `sceWebBrowserDialogOpenForPredeterminedContent` run the common-dialog state machine without displaying an interactive browser, and `sceWebBrowserDialogGetResult` reports that the user cancelled. Minecraft Bedrock (PPSA17221) imports `sceWebBrowserDialogOpenForPredeterminedContent` (#5744). The real export takes a second argument, the allow-list of up to twenty addresses (SharpProspero `docs/dialogs.md`); it is not read here, so no address filtering is modelled, and the exact validation order and codes with a null second argument are unverified.
- [libSceWebBrowserDialog](../../../core/libs/prx/libSceWebBrowserDialog/Export.cpp) - `sceWebBrowserDialogClose` transitions an initialized dialog directly to `COMMON_DIALOG_STATUS_FINISHED` even when called before `Open`, following the Signin / NetCtlAp / PlayerSelection convention; this behavior is unverified against real hardware.
