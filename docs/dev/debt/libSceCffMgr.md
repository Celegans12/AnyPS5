# libSceCffMgr technical debt

### Unknown function info

- [libSceCffMgr inventory exports](../../../core/libs/prx/libSceCffMgr/Export.cpp) (libSceCffMgr) - parameter lists, return types and behaviour are unverified, so every entry throws through `NotImplemented_nid_no_patch`; the parameterless `int` declarations only resolve imports. The names and NIDs come from the [PS5 13.60 export inventory](https://github.com/xdCloudy/PS5-13.60-Function-Stub-Library/blob/226383a7fa4b1275d6cebf61eb1c1e7ca7b58bcb/data/firmware-13.60/libscecffmgr-cdffaf2f9a.jsonl), and the pinned [module map](https://github.com/boykopovar/ps5-nid-map/blob/cc1d586fddab576a520f226da6b2632b031e6d09/data/Modules/libSceCffMgr.inc) assigns them to the physical `libSceCffMgr.sprx` provider. Seven more inventory NIDs are unresolved and not declared.
