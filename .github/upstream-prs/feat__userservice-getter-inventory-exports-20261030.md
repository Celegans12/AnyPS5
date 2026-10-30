### What
Adds 150 throwing export placeholders to `libSceUserService`: the 13 lifecycle and status exports (`sceUserServiceLogin`, `sceUserServiceIsLoggedIn`, `sceUserServiceRegisterEventCallback` and others) and the first 137 of the 257 getters, alphabetically, that the PS5 13.60 export inventory names and `main` does not define. The remaining getters and the 242 setters follow in later PRs to stay under 150 stubs per PR. Each NID was recomputed from its name and is listed for `libSceUserService` in the pinned ps5-nid-map module map. Signatures are unknown, so each placeholder is a parameterless `int` that throws through `NotImplemented_nid_no_patch`, and `docs/dev/debt/libSceUserService.md` records the gap.

### Tested
Not tested on a title. `python3 tools/check_conventions.py --base origin/main` passes, and `clang++ -std=c++20 -fsyntax-only` on the library's `Export.cpp` is clean. Searched `main` and the heads of PRs #11000 to #13406 for the names and NIDs, and none define them.

### Checklist
<!-- Rules behind each item: https://github.com/boykopovar/AnyPS5/blob/main/CONTRIBUTING.md -->
- [x] Based on current `main`; no other open PR implements the same functions
- [x] One topic per PR; follow-ups go in a new PR
- [x] No comments in code except [technical debt](https://github.com/boykopovar/AnyPS5/blob/main/docs/dev/CONVENTIONS.md)
- [x] Unimplemented paths throw (`NotImplemented_nid_no_patch`); silent stubs are listed under Silent stubs in their library's [technical debt](https://github.com/boykopovar/AnyPS5/tree/main/docs/dev/debt) file
- [x] No notes, investigation `.md` files or images added to the repository (attach them to this PR)
- [x] New third-party code is a submodule built from source
- [x] Generic behaviour, not specific to one title
- [x] Depends on: none
- [x] Fixes: none
- [x] Unblocks: none
- [x] GPUs tested: n/a
- [x] AI-assisted: yes
