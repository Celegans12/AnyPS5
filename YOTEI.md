# Ghost of Yōtei (PPSA26344) on AnyPS5: test branch

This branch, `yotei/best` on Celegans12/AnyPS5, is the tree behind the Yōtei numbers in
[discussion #675](https://github.com/boykopovar/AnyPS5/discussions/675). It moves forward only when a build
matches the 60-frame title reference exactly and runs faster in the open world than the build before it.

**It is a work-in-progress test branch, not upstream code.** It carries local rewrites that haven't been reviewed
upstream (bindless image tables, an async compute queue, a split draw committer and more), plus diagnostics. It's
based on `main` from 2026-10-15 (`a5e2a80c6`) and is not synced with current `main`. Fixes that apply to `main` go
upstream as separate PRs.

## Current state (2026-10-19)

- Build `8ba98278a`, shader-cache SourceVersion `63a364723a2373f6`.
- From a save near Teshio Ridge, the open world (Nupur river) appears about 10 minutes after launch with a warm
  shader cache. The first run on a new SourceVersion compiles shaders cold and can take 30-60 minutes to get there.
- In the world: about **5.0 presents per 10 s (~0.5 fps)**, about 2.5 s per frame. Earlier builds on this line ran
  25 minutes in the world without a crash.
- Tested on Windows 11, RTX 3080 Laptop GPU (16 GB, driver 610.62), 32 GB RAM, MinGW-w64 GCC 15.2.0, Release.

## Run it with

```
APS5_SPANNED_WAVE=1 APS5_BINDLESS_IMAGES=1 APS5_BINDLESS_LAZY=1 APS5_NO_RESULT_MEMO=1
APS5_STAGING_DIFF=1 APS5_PIPELINE_INDIRECT=1 APS5_NO_ADDRESS_DRAW_REUSE=1 APS5_ASYNC_LANE=51
```

- `APS5_ASYNC_LANE=51` records the cloth compute ring (queue 0x51) on a second Vulkan queue.
- Don't set `APS5_PIPELINE_REQUESTS` when you measure: it costs about 12 % of a frame on this branch.
- Every local change has an off switch (`APS5_NO_...`); the list is in the commit messages.

## Known issues

- Black shapes on some rocks and terrain: the depth pre-pass writes depth that the G-buffer draw of the same mesh
  then fails its EQUAL test against. Cause not found yet.
- Process private memory reaches about 34 GB in the world, so 32 GB of RAM is tight.
- One compute program (0x8000361f00) is still refused by the recompiler at registration (non-fatal so far).

## What it carries on top of `main` 2026-10-15

- Upstream PRs adopted before they merged, among them #4943, #7183, #4288, #3557, #6771, #7044, #6617, #6566, #1896,
  #6763, #6632, #6631, #6501, #7134, #6187, #6192, #6636, #7212 and #7218.
- Local work:
  - image tables resolved on the GPU through a bindless descriptor heap, including T#s loaded from run-time addresses;
  - a cheaper draw-miss path (resolve memo, faster table snapshots);
  - texture revalidation that walks only the readable parts of partly committed memory;
  - the HTILE mask-clear kernel's buffer-descriptor form recognized;
  - a draw preparer thread with a two-phase draw build;
  - device-local buffers for address-based compute builds (the cloth programs);
  - copy-backs that store only the changed dwords;
  - dispatch overlap;
  - the async compute queue.

## Updates

- 2026-10-19: branch created at `8ba98278a` (async compute queue), 5.0 presents per 10 s in the world.
