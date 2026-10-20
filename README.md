<h1 align="center">Ghost of Yōtei on AnyPS5</h1>

<p align="center"><b>A test branch of <a href="https://github.com/boykopovar/AnyPS5">AnyPS5</a> that runs Ghost of Yōtei (PPSA26344) into the open world.</b><br>
<sub>Branch <code>yotei/best</code> on Celegans12/AnyPS5 · progress thread: <a href="https://github.com/boykopovar/AnyPS5/discussions/675">discussion #675</a> · <a href="#run-it">run it</a> · <a href="#updates">updates</a></sub></p>

<p align="center"><img src="https://raw.githubusercontent.com/Celegans12/AnyPS5/bringup-media/yotei/2026-10-19/open-world-batchstage.png" alt="The Nupur river and the Ainu village in the open world of Ghost of Yōtei on AnyPS5, frame read back 1,571 s after launch" width="880"></p>

> [!NOTE]
> This is a work-in-progress test branch, not upstream code. It is based on `main` from 2026-10-15 (`a5e2a80c6`), is not
> synced with current `main`, and carries local rewrites that haven't been reviewed upstream, plus diagnostics. It moves
> forward only when a build renders our 60-frame title reference exactly and runs faster in the open world than the build
> before it. Fixes that apply to `main` go upstream as separate PRs.

## Where it stands (2026-10-20)

| | |
|---|---|
| **Furthest point** | Open world, the Nupur river by the Ainu village, with HUD and quest |
| **Time to the world** | about 10 minutes from a save near Teshio Ridge (warm shader cache) |
| **Frame rate in the world** | about **0.5 fps** (4.4-5.6 presents per 10 s; this laptop's GPU clocks drift by ~14 % between sessions, so builds are compared back to back in one session) |
| **Stability** | 25 minutes in the world without a crash on earlier builds of this line |
| **Tested on** | Windows 11, RTX 3080 Laptop GPU 16 GB (driver 610.62), 32 GB RAM, MinGW-w64 GCC 15.2.0, Release |
| **Build** | `3cbec3212` (combo19 + batched staging + in-run A/B + overlap staging fixes + import re-entry fix + sparse page table) + docs, shader-cache SourceVersion `68f2c78b5d5316b4` |

<table>
<tr>
<td width="50%"><img src="https://raw.githubusercontent.com/Celegans12/AnyPS5/bringup-media/yotei/2026-10-18/open-world-nupur-river.png" alt="The Nupur river in daylight, 636 s after launch"></td>
<td width="50%"><img src="https://raw.githubusercontent.com/Celegans12/AnyPS5/bringup-media/yotei/2026-10-18/open-world-rain.png" alt="The same spot with rain coming in, 1,986 s after launch"></td>
</tr>
<tr>
<td align="center"><sub>Daylight at the river, 636 s after launch</sub></td>
<td align="center"><sub>The weather turns: rain, 33 minutes in</sub></td>
</tr>
<tr>
<td colspan="2"><img src="https://raw.githubusercontent.com/Celegans12/AnyPS5/bringup-media/yotei/2026-10-15/opening-cinematic-on-main.png" alt="The burning tree from the opening cinematic"></td>
</tr>
<tr>
<td colspan="2" align="center"><sub>The opening cinematic, the first world scene that rendered on current main (2026-10-15)</sub></td>
</tr>
</table>

## How the frame rate got here

| Date | Presents per 10 s in the world | What changed |
|---|---|---|
| 10-17 | 2.35 | first time in the open world |
| 10-18 | 2.97 | cheaper draw-cache keys |
| 10-18 | 3.9 | texture revalidation walks only the readable parts of partly committed memory |
| 10-18 | 4.2 | hand-off PRs from upstream, copy-backs of changed dwords only |
| 10-18 | 4.6 | draw preparer thread, device-local buffers for the cloth compute programs |
| 10-19 | 5.0 | the cloth compute ring on its own Vulkan queue |
| 10-19 | +5 % presents, -8 % frame time vs the build above, same session | indirect-draw records read on the GPU, exact f32 path for f16 mix math (one pass 101 -> 27 ms GPU), presenter and barrier trims |
| 10-19 | **5.1** (+9 % in an in-run A/B, ~2.7 SE) | a dispatch run's staged copy-ins in one prelude with one barrier instead of a barrier per dispatch: the cloth queue's GPU time halves, GPU busy -26 % |

## Run it

Build as usual ([build instructions](docs/dev/BUILD.md)), then run the game with:

```
APS5_SPANNED_WAVE=1 APS5_BINDLESS_IMAGES=1 APS5_BINDLESS_LAZY=1 APS5_NO_RESULT_MEMO=1
APS5_STAGING_DIFF=1 APS5_PIPELINE_INDIRECT=1 APS5_NO_ADDRESS_DRAW_REUSE=1 APS5_ASYNC_LANE=51
```

- `APS5_ASYNC_LANE=51` records the cloth compute ring (queue 0x51) on a second Vulkan queue. If the GPU hangs or the device is lost (see known issues), leave it out.
- Off switches added on this line: `APS5_NO_GPU_INDIRECT_ARGS=1`, `APS5_NO_MIX_F32=1`, `APS5_NO_BARRIER_BATCH=1`, `APS5_NO_EARLY_REVALIDATE=1`, `APS5_NO_FLAT_POISON_INDEX=1`, `APS5_NO_BATCHED_STAGING=1`, and in the latest build `APS5_NO_OWED_RUN_CLOSE=1`, `APS5_NO_RUN_ATOMIC_GUARD=1`, `APS5_NO_RUN_STAGED_READ_GUARD=1` (the three overlap fixes) and `APS5_NO_SPARSE_PAGE_STATES=1`.
- `APS5_AB=<seconds>:<switch>[,...]` (with `APS5_AB_START=<present>`) alternates two configurations of driver switches in one run, so a comparison needs one run and no run-to-run noise.
- The first run on a new shader-cache version compiles shaders cold; it can take 30-60 minutes to reach the world.
- Leave `APS5_PIPELINE_REQUESTS` unset when you measure: it costs about 12 % of a frame here.
- Every local change has an off switch (`APS5_NO_...`), named in its commit message.

## Known issues

- **GPU hangs with `APS5_ASYNC_LANE=51`.** On 2026-10-20 this build hung twice in the world (a queue waits 40 s for the GPU, then the device is lost), 400-510 s after launch. Every hang or GPU write fault we've seen with dispatch overlap on happened with the second queue on. The two runs we've done without it so far (on our merge build) were clean. If you hit this, drop `APS5_ASYNC_LANE=51` from the env. A fix is being worked on.
- Black shapes on some rocks and terrain (the depth pre-pass and the G-buffer pass disagree on depth). Cause not found yet.
- Process memory reaches about 34 GB in the world, so 32 GB of RAM is tight.
- The overlap fixes keep conflicting dispatches out of overlapped runs, which costs about 115 ms of GPU time per frame (+14 %). Frames are CPU-bound, so presents barely move (about 1 % in an in-run A/B), but it will matter once the CPU side gets faster.
- One compute program (0x8000361f00) is still refused by the recompiler at registration (non-fatal so far).
- About 0.5 fps is far from playable. What limits a frame now, and what's next, is in [discussion #675](https://github.com/boykopovar/AnyPS5/discussions/675).

## What it carries on top of `main` 2026-10-15

- Upstream PRs adopted before they merged, among them #4943, #7183, #4288, #3557, #6771, #7044, #6617, #6566, #1896,
  #6763, #6632, #6631, #6501, #7134, #6187, #6192, #6636, #7212, #7218 and #7489, plus upstream 5c716de97.
- Local work:
  - image tables resolved on the GPU through a bindless descriptor heap, including T#s loaded from run-time addresses;
  - a cheaper draw-miss path (resolve memo, faster table snapshots);
  - texture revalidation that walks only the readable parts of partly committed memory;
  - the HTILE mask-clear kernel's buffer-descriptor form recognized;
  - a draw preparer thread with a two-phase draw build;
  - device-local buffers for address-based compute builds (the cloth programs);
  - copy-backs that store only the changed dwords;
  - dispatch overlap;
  - the async compute queue;
  - indirect-draw parameters read on the GPU for draws whose records are still being written;
  - an exact f32 path for `v_fma_mix` behind a fused-FMA startup check;
  - presenter revalidation prepared off the GPU lock, and a cheaper image-table miss path;
  - batched staging: a dispatch run's copy-ins in one prelude ahead of it and its copy-backs behind one barrier (on NVIDIA any barrier between dispatches serializes them);
  - an in-run A/B switch registry;
  - dispatch runs that keep staged results and order: a run that still owes a build's copy-backs closes before that build is reused, atomics on one staged range don't share a run, and a read of a run member's staged output closes the run;
  - from contributors in #675: FardinHaque70's fix for an import unwatch hook that re-entered the import lookup (from hritikutekar's diagnosis), and Izamaru's page-state table allocated on demand (about 250 MB less commit charge).

## Updates

- 2026-10-19: branch created at `8ba98278a` (async compute queue), 5.0 presents per 10 s in the world.
- 2026-10-19: moved to combo19 (`75dd18755`): matches the title reference exactly; against the previous build in the same session +5 % presents, -8 % frame time, -40 % CPU waits for GPU results. New shader-cache version, so the first run compiles cold.
- 2026-10-19: added batched staging (`ace5c14cd`): T1 exact; in-run A/B in the world 5.12 vs 4.68 presents per 10 s, GPU busy 810 vs 1,096 ms per frame. Same shader cache as combo19.
- 2026-10-20: added three fixes for dispatch overlap with staged buffers (`3cbec3212`). Each had a test that failed on the previous build: a reused build lost its results, two atomic adds became one, and a read saw data from before the run. Also FardinHaque70's import re-entry fix and Izamaru's sparse page table. T1 exact, same shader cache. Speed is within noise by presents (in-run A/B: about 1 %), GPU busy +14 %. In the same A/B, the only GPU hang came in a phase with the fixes off.
- 2026-10-21: added a known issue: GPU hangs with `APS5_ASYNC_LANE=51` (twice in the world on this build). Leave it out if you hit them.

---

# About AnyPS5

Tool for automatic executables porting to Linux and Windows.

[**Web converter**](https://boykopovar.github.io/AnyPS5/) - files stay on your device.

![anyps5_collage1](https://github.com/user-attachments/assets/ed2f0b2d-33d3-46d7-ab88-767ca872fc99)

Includes a [relinker](core/relinker) that converts executable to the target system's native format and implementations of [system prx libraries](core/libs/prx) suitable for dynamic linking. No emulation or separate runtime process.

[**Discord**](https://discord.gg/BHFztBPUe), [Usage](docs/user/USAGE.md), [Build instructions](docs/dev/BUILD.md), [Architecture](docs/dev/ARCHITECTURE.md), [Technical debt of the project](docs/dev/TechnicalDebt.md), [code style conventions](docs/dev/CONVENTIONS.md), [contributing](CONTRIBUTING.md)

## Status

[![libraries](https://boykopovar.github.io/AnyPS5/badge-libraries.svg)](https://boykopovar.github.io/AnyPS5/progress.html) [![shaders](https://boykopovar.github.io/AnyPS5/badge-shaders.svg)](https://boykopovar.github.io/AnyPS5/progress.html)

[![progress map](https://boykopovar.github.io/AnyPS5/progress.svg)](https://boykopovar.github.io/AnyPS5/progress.html)

[Progress history](https://boykopovar.github.io/AnyPS5/history.html)

<sub>* System libraries: percentage of the functions known to the project so far (declared in [core/libs/prx](core/libs/prx)), not of every PS5 system function. The total grows as more functions are declared.</sub>

Unsupported or unexpected states strictly throw `std::runtime_error`. `what()` is printed to stderr and the process terminates.

The [shader recompiler](core/shader/recompiler/Recompiler.cpp) successfully produces SPIR-V (validated via [Spirv-Tools](3rdparty/SPIRV-Tools) when built with `ANYPS5_ENABLE_SPIRV_TOOLS`).

## Compatibility

Per-game progress (stage reached, open blockers, libraries and GPU coverage) is tracked on the [blocker map](https://alxcbtw.github.io/anyps5-roadmap/), a community project built from public GitHub data ([source](https://github.com/AlxcBTW/anyps5-roadmap)). Test reports go in the [Game bring-up](https://github.com/boykopovar/AnyPS5/discussions/categories/game-bring-up) discussions, one thread per title.

## Input mapping

SDL-mapped game controllers are supported, including analog sticks and triggers. Keyboard and mouse controls can be configured with an `anyps5-input.ini` file. See [input mapping](docs/user/INPUT_MAPPING.md) for the supported devices and configuration format.

## Disclaimer

This project is intended for interoperability, research, preservation, and compatibility purposes. It does not include, distribute, or require copyrighted software, firmware, cryptographic keys, or proprietary libraries. Users are responsible for ensuring that any binaries used with this project are obtained and used in accordance with applicable laws and their respective license terms.

## License

This project is licensed under the GNU General Public License version 2 only.
