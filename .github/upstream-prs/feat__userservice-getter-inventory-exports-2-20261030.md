### What
Adds the remaining 120 throwing `libSceUserService` getter placeholders that the PS5 13.60 export inventory names and `main` does not define, from `sceUserServiceGetNotificationSettings2` onward. Stacked on #13422, so this branch also carries its commit; review only the second commit, `feat(libSceUserService): add remaining 120 getter export placeholders`. The 242 setters follow in later PRs.

### Tested
Not tested on a title. `python3 tools/check_conventions.py --base origin/main` passes, and `clang++ -std=c++20 -fsyntax-only` on the library's `Export.cpp` is clean. Searched `main` and the heads of PRs #11000 to #13434 for the names and NIDs, and none define them.

### Checklist
<!-- Rules behind each item: https://github.com/boykopovar/AnyPS5/blob/main/CONTRIBUTING.md -->
- [x] Based on current `main`; no other open PR implements the same functions
- [x] One topic per PR; follow-ups go in a new PR
- [x] No comments in code except [technical debt](https://github.com/boykopovar/AnyPS5/blob/main/docs/dev/CONVENTIONS.md)
- [x] Unimplemented paths throw (`NotImplemented_nid_no_patch`); silent stubs are listed under Silent stubs in their library's [technical debt](https://github.com/boykopovar/AnyPS5/tree/main/docs/dev/debt) file
- [x] No notes, investigation `.md` files or images added to the repository (attach them to this PR)
- [x] New third-party code is a submodule built from source
- [x] Generic behaviour, not specific to one title
- [x] Depends on: #13422
- [x] Fixes: none
- [x] Unblocks: none
- [x] GPUs tested: n/a
- [x] AI-assisted: yes
