<h1 align="center">Ghost of Yōtei on AnyPS5</h1>

<p align="center"><b>A test branch of <a href="https://github.com/boykopovar/AnyPS5">AnyPS5</a> that runs Ghost of Yōtei (PPSA26344) into the open world.</b><br>
<sub>Branch <code>yotei/best</code> on Celegans12/AnyPS5-Ghost-of-Yotei · progress thread: <a href="https://github.com/boykopovar/AnyPS5/discussions/675">discussion #675</a> · <a href="#run-it">run it</a> · <a href="#updates">updates</a></sub></p>

<p align="center"><img src="https://raw.githubusercontent.com/Celegans12/AnyPS5-Ghost-of-Yotei/bringup-media/yotei/2026-10-21/open-world-merge19.png" alt="Atsu at the Nupur river by the Ainu village in the open world of Ghost of Yōtei on AnyPS5 (2026-10-21)" width="880"></p>

> [!NOTE]
> This is a work-in-progress test branch, not upstream code. Since 2026-10-27 it is based on `main` from 2026-10-26
> (`88795ed448`) and carries local rewrites that haven't been reviewed upstream, plus diagnostics. It moves forward only
> when a build renders our 60-frame title reference exactly and runs faster in the open world than the build before it.
> Fixes that apply to `main` go upstream as separate PRs.

## Where it stands (2026-10-27)

| | |
|---|---|
| **Furthest point** | Open world, the Nupur river by the Ainu village, with HUD and quest |
| **Time to the world** | about 5-7 minutes from a save near Teshio Ridge (warm shader cache) |
| **Frame rate in the world** | about **0.6-0.8 fps** (6-8.4 presents per 10 s depending on the spot and the session; about 7.8-7.9 at the river village on this build (merged with `main` 2026-10-26; 7.9 against 7.75 for the build before, back to back); this laptop's GPU clocks drift by ~14 % between sessions, so builds are compared back to back in one session) |
| **Stability** | with the second Vulkan queue on its own compute-only queue family, world runs are clean (our measured runs last 25-40 minutes in the world: 2,400 s on 10-26, three 1,500 s runs on 10-27); the one crash we saw on 10-25 (a second-queue label bug) is fixed since `5becc82a97`. One Linux tester (RTX 3080 Ti 12 GB, 3840x2160 guest) had a GPU hang right after the menu with video memory nearly full; not reproduced here yet |
| **Tested on** | Windows 11, RTX 3080 Laptop GPU 16 GB (driver 610.62), 32 GB RAM, MinGW-w64 GCC 15.2.0, Release |
| **Build** | `416ce3c368`: our tree (`d3e0db3032`, owned answers from per-block checks within a collect epoch, and everything listed under [What it carries](#what-it-carries-on-top-of-main-2026-10-26)) merged with `main` 2026-10-26 `88795ed448` (which now includes #9957, #9181, #9567 and #11120), plus Pipodi's absent runtime formats (Celegans12/AnyPS5-Ghost-of-Yotei#1) and two of hritikutekar's runtime-format reads from #675, + docs, shader-cache SourceVersion `22c401ea7ca3796f`, title reference T1 exact |

<table>
<tr>
<td width="50%"><img src="https://raw.githubusercontent.com/Celegans12/AnyPS5-Ghost-of-Yotei/bringup-media/yotei/2026-10-18/open-world-nupur-river.png" alt="The Nupur river in daylight, 636 s after launch"></td>
<td width="50%"><img src="https://raw.githubusercontent.com/Celegans12/AnyPS5-Ghost-of-Yotei/bringup-media/yotei/2026-10-18/open-world-rain.png" alt="The same spot with rain coming in, 1,986 s after launch"></td>
</tr>
<tr>
<td align="center"><sub>Daylight at the river, 636 s after launch</sub></td>
<td align="center"><sub>The weather turns: rain, 33 minutes in</sub></td>
</tr>
<tr>
<td colspan="2"><img src="https://raw.githubusercontent.com/Celegans12/AnyPS5-Ghost-of-Yotei/bringup-media/yotei/2026-10-15/opening-cinematic-on-main.png" alt="The burning tree from the opening cinematic"></td>
</tr>
<tr>
<td colspan="2" align="center"><sub>The opening cinematic, the first world scene that rendered on `main` (2026-10-15)</sub></td>
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
| 10-21 | **5.0** (the previous build got 4.19 in the same session: +19 % presents, -13 % frame time) | merged with `main` 2026-10-20; three fixes for slowdowns `main` added in the draw committer (buffer pool tiers, exact-size buffer reuse, held-read checks: +18 % in an in-run A/B); the second Vulkan queue off |
| 10-21 | **5.9** (lane off 5.45 in the same session: +8 % presents, -16 % median frame; provisional) | the cloth compute ring's second Vulkan queue back on, now on a compute-only queue family |
| 10-21 | +11 % in an in-run A/B (6.26 vs 5.62) | hritikutekar's two committer fixes: a count of prepared bindless tables instead of a full walk every 4 ms, and small texture snapshots in pooled slabs |
| 10-22 | **5.2** (the previous build with hritikutekar's fixes: 5.13 in the same session) | merged with `main` 2026-10-21; same speed, newer `main` underneath |
| 10-24 | +2.8 % / +2 % / +3 % in in-run A/Bs | hritikutekar's second committer batch; #7514; queue-0 dispatches through the draw pipeline (`APS5_PIPELINE_DISPATCH=1`) |
| 10-24 | even with the build before | merged with `main` 2026-10-24 |
| 10-25 | **7.6** (7.2 with them off in the same run: +8.5 %, frames ~100 ms shorter) | owned texture proofs: the committer's texture checks become a few word reads instead of a memory walk |
| 10-25 | +1.8 % in an in-run A/B | shahulmubeen-a's HTILE note for cache-hit dispatch items (#2) |
| 10-26 | **8.05** (7.70 with them off in the same run: +4.4 %, frames ~55 ms shorter) | owned reuse of untracked vertex/index inputs and draw snapshots; storage flushes wait only for the lanes that write the range |
| 10-26 | **7.8** (7.3 with all owned features off in the same run: +6.4 %, frames ~80 ms shorter) | the owned features made fault-free: write watch instead of write protection, answered from per-block checks within each collect epoch |
| 10-27 | **7.9** (the build before: 7.75 between two runs of this one, same script) | merged with `main` 2026-10-26; same speed, newer `main` underneath |

## Run it

Build as usual ([build instructions](docs/dev/BUILD.md)), then run the game with:

```
APS5_SPANNED_WAVE=1 APS5_BINDLESS_IMAGES=1 APS5_BINDLESS_LAZY=1 APS5_NO_RESULT_MEMO=1
APS5_STAGING_DIFF=1 APS5_PIPELINE_INDIRECT=1 APS5_NO_ADDRESS_DRAW_REUSE=1
APS5_NO_PIPELINE_JOURNAL=1 APS5_RESOURCE_CACHE_ENTRIES=1024 APS5_ASYNC_LANE=51
APS5_NO_DRAW_AHEAD=1 APS5_DEFER_REFRESH_FLUSH=1 APS5_PIPELINE_DISPATCH=1
```

- **This build has a new shader-cache version (since 2026-10-27, `22c401ea7ca3796f`), so the first run compiles cold**: about 54 minutes to the world on our machine (frame 90 at 3,218 s). The second run can still compile a few very large pipelines (about 3 minutes on the last loading screen for us, if the first run was stopped before the driver saved them); after that, runs start warm (frame 90 at about 400 s).
- `APS5_ASYNC_LANE=51` runs the cloth compute ring on a second Vulkan queue. That queue now comes from a compute-only queue family by default; `APS5_LANE_GRAPHICS_FAMILY=1` goes back to a third queue of the graphics family, which faulted the GPU for us. Without `APS5_ASYNC_LANE`, everything runs on one queue as before.
- `APS5_DEFER_REFRESH_FLUSH=1` lets a storage refresh record its GPU-direct write-backs behind the pending writer instead of waiting for it; hritikutekar's GPU slice copies for reflection probes need it.
- `APS5_NO_DRAW_AHEAD=1` keeps mitul72's draw-ahead (#6188) off: worker threads that capture and compile upcoming draws ahead of queue 0. Leave it out to try it: in our A/B it was +4 % presents (not significant), and one of four title checks with it on stalled at boot and rendered wrong, which we haven't reproduced or explained yet.
- `APS5_NO_PIPELINE_JOURNAL=1` turns off `main`'s startup pipeline replay. `APS5_RESOURCE_CACHE_ENTRIES=1024` is the resource-cache size all our measurements used.
- `main`'s paths that broke Yōtei on our tree are off here and can be turned back on:
  - `APS5_RESIDENT_LINEAR=1`: resident linear color targets. Wrong spawn, T-pose, camera under water or in the sky (#8152).
  - `APS5_DEFERRED_SETTLES=1`: deferred import settles. GPU write faults.
  - `APS5_HOST_IMPORT_HALF_HEAP_RESERVE=1`: `main`'s halved host-import budget. The world arrived late.
  - `APS5_TABLE_COUNT_NEUTRAL_PIPELINES=1`: broke world rendering.
  - `APS5_CLEAR_HAND_OVER=1`: `main`'s clear hand-over (#7427). A white glare with coloured sprites over the upper half of the world.
  - `APS5_PLAYGO_SCENARIO_CHUNKS=1`: `main`'s PlayGo chunk listing from a scenario file (#7726). The ATRAC9 crash it used to cause is gone since #6721 (on this base too), but with the listing the boot crawls: on `main` of 10-26 the title sequence reached frame 23 by 370 s, against the title at 324 s with the listing off. Since `main` of 10-26, a chunk ID outside the listing throws (or follows `APS5_PLAYGO_UNLISTED_CHUNKS`); with the listing off, as here, every chunk answers installed.
  - `APS5_IMAGE_POOL=1`: `main`'s image memory pool (#7870), not tested on Yōtei.
  - Where `main` and our tree differ, `APS5_RESULT_MEMO_TABLE=1`, `APS5_RELOCATED_HEADER_COPIES=1` and `APS5_SAMPLED_LATER_USE_REFUSAL=1` select `main`'s version.
- Off switches for the newest local changes:
  - `APS5_NO_SMALL_DEVICE_TIER=1`, `APS5_NO_EXACT_SIZE_REUSE=1`, `APS5_NO_HELD_READ_SKIP=1`, `APS5_NO_HELD_READ_BOUNDS=1`: committer fixes.
  - `APS5_NO_OWED_RUN_CLOSE=1`, `APS5_NO_RUN_ATOMIC_GUARD=1`, `APS5_NO_RUN_STAGED_READ_GUARD=1`: overlap fixes.
  - `APS5_NO_LANE_OPEN_SERIAL=1`, `APS5_NO_LANE_TEMPLATE_EDGE=1`, `APS5_NO_LANE_LABEL_SCOPE=1`, `APS5_NO_LANE_FORWARD_EDGE=1`, `APS5_NO_LANE_LEASE_WAIT=1`: second-queue fixes.
  - `APS5_NO_IMAGE_SLABS=1`, `APS5_NO_PREPARED_TABLE_COUNT=1`: hritikutekar's committer fixes; `APS5_NO_PRESENT_FRAME_AGE=1`: his texture-cache frame age.
  - `APS5_NO_INCREMENTAL_BDA_RANGES=1`, `APS5_NO_SLICE_COPY=1`, `APS5_NO_DEPTH_INDEX=1`, `APS5_NO_DEFERRED_DRAW_EVICTION=1`: hritikutekar's four committer changes (`APS5_VERIFY_BDA_INCREMENTAL=1` checks the incremental merge against the full one).
  - `APS5_NO_BDA_TEMPLATES=1`: address-table templates (cached builds for read-only draws that read memory through pointers).
  - `APS5_NO_LEAN_BDA_TABLES=1`, `APS5_NO_DEPTH_LOOKUP_FAST=1`, `APS5_NO_REFUSAL_MEMO=1`, `APS5_NO_DISTINCT_WRITE_SCANS=1`, `APS5_NO_COLLECT_HEAP=1`, `APS5_NO_LARGER_SMALL_POOLS=1`, `APS5_NO_DEFER_IMPORTED_STORES=1`: hritikutekar's second batch (`APS5_VERIFY_COLLECT_HEAP=1` checks the heap merge against the old one; `APS5_DEFER_FLIP_STORE=1` is his opt-in flip-store deferral, off here).
  - `APS5_FLUSH_WAIT_WHOLE_RANGE=1`: a storage flush waits for writers over its whole range again instead of only the image layers it stores (SleglJan's #8887); `APS5_NO_REFRESH_LOOKUP_CACHE=1`: texture refreshes walk the pending registry and look up host imports in full again (AlxcBTW's #7514).
  - `APS5_NO_READ_ONLY_LEASE_SKIP=1`: an address-table build whose shaders never store through pointers counts as a writer of its leased memory again, so texture refreshes wait for it (#10382).
  - `APS5_NO_OWNED_VERTEX_INPUTS=1`, `APS5_NO_OWNED_DRAW_SNAPSHOTS=1`: owned reuse of vertex/index inputs and draw snapshots in untracked memory (`APS5_VERIFY_OWNED_SNAPSHOTS=<n>` compares the first n reuses per 10 s against guest memory). `APS5_GPU_SOURCED_DRAW_BUFFERS=1` / `APS5_IMPORT_SOURCED_COPY_IN=1` are opt-in experiments (slower here).
  - `APS5_NO_FLUSH_LANE_WAIT=1`: a storage flush waits for every batch on both queues again instead of only the lanes that write or lease the stored range (`APS5_VERIFY_FLUSH_LANE_WAIT=1` checks it).
  - `APS5_NO_OWNED_PROOFS=1`: owned texture proofs (`APS5_VERIFY_OWNED_PROOFS=<n>` re-runs the old proof for the first n armed proofs every 10 s and reports any disagreement). Windows only; elsewhere the old proof runs.
  - `APS5_NO_OWNED_EPOCH=1`: owned proofs ask Windows for the written pages on every proof again instead of answering from the per-block check of the current collect epoch (`APS5_VERIFY_OWNED_EPOCH=<n>` also queries the first n epoch answers every 10 s and reports any disagreement).
  - `APS5_NO_ABSENT_RUNTIME_FORMAT=1` (Pipodi's absent runtime formats), `APS5_NO_COMPACT_FORMAT_CONVERT=1` and `APS5_NO_SINGLE_FORMAT_READ=1` (hritikutekar's runtime-format reads).
  - `APS5_NO_HIT_ITEM_HTILE_NOTE=1`: a depth clear recorded as a cached dispatch item (`APS5_PIPELINE_DISPATCH`) no longer notes its HTILE mask, so the next depth draw waits for the whole GPU before its store-back again.
  - `APS5_NO_OVERLAP_RANGE_CHECK=1`: overlapped dispatches no longer need disjoint access ranges (the old rule).
  - `APS5_NO_DISPATCH_OVERLAP=1`, `APS5_NO_BATCHED_STAGING=1`, `APS5_NO_SPARSE_PAGE_STATES=1`.
  - Every local change has an off switch (`APS5_NO_...`), named in its commit message.
- Diagnostics: `APS5_GPU_CHECKPOINTS=1` (`=batch` marks only batch begin/end per queue), `APS5_NV_DIAGNOSTICS=1` (NVIDIA device diagnostics config, so an Nsight Aftermath crash dump can name the fault), `APS5_FAULT_ADDRESSES=1` (with `APS5_MEM_STATS=1`; also reports a buffer destroyed while a pending batch writes it), `APS5_LANE_SERIALIZE=1`, `APS5_NO_LANE_OVERLAP=1`.
- `APS5_AB=<seconds>:<switch>[,...]` (with `APS5_AB_START=<present>`) alternates two configurations of driver switches in one run, so a comparison needs one run and no run-to-run noise.
- Leave `APS5_PIPELINE_REQUESTS` unset when you measure: it costs about 12 % of a frame here.

## Known issues

- Fixed on 10-26: the owned proofs no longer write-protect memory (no guest-thread write faults), and this build carries upstream's #9181, so resident runs aren't write-protected either. The first write-watch versions measured neutral; with the per-epoch checks (`d3e0db3032`) they are +6.4 % again and back on in the env.

- Fixed on 10-25: a rare crash with the second queue (`AGC graphics: no batch to append a completion label to`): a queue-0 label waiting for a copy-back found the copied writer only in the second queue's batch. The label now goes into that lane's batch (`5becc82a97`, with a test).
- A silver sheen sometimes appears on Atsu and the boat after entering the world, and in a few runs over the whole scene. It comes and goes: in one run it cleared by itself after about ten minutes, and the three runs of 10-27 didn't show it, so it may depend on the weather or time of day. Under investigation (one lead: a GPU fill over texture memory that the next texture read doesn't see).

- **The second queue needs a compute-only queue family.** On a third queue of the graphics family (the old setup, or a GPU without a compute-only family, where the driver falls back to it), 8 of 9 world runs lost the GPU within 4-14 minutes. An Nsight Aftermath dump showed a DMA page fault inside the second queue's batch. On the compute-only family, 4 of 4 runs were clean. If you hit GPU hangs, leave `APS5_ASYNC_LANE` out.
- The second queue is tested only on Windows with NVIDIA (RTX 3080 Laptop, driver 610.62). AMD and Linux are untested.
- A GPU hang can reset NVIDIA's shader store for the game. After that, the saved pipelines no longer load and every run compiles again (long stalls on the last loading screen). If that happens after a hang, expect one slow run.
- Black shapes on some rocks and terrain (the depth pre-pass and the G-buffer pass disagree on depth). Cause not found yet.
- Process memory is about 30-35 GB in the world, so 32 GB of RAM is tight. `main` 2026-10-21 dropped the 2 GiB reserve of the sampled-texture budget (#7862), with no switch; our cold run went down to 0.2 GB of free RAM (paging) without failing.
- Once, during loading on a build of 10-21 or 10-22, the game stopped with `missing static shader ABI register 531`. It did not come back in any later run; if you see it, please post the log in #675.
- A dispatch joins an overlapped run only when its access ranges are disjoint from the run's; address-based, BDA and bindless dispatches count as conflicting. That costs about 5 % GPU time per frame against the old rule. Frames are CPU-bound, so presents don't move.
- One compute program (0x8000361f00) is still refused by the recompiler at registration (non-fatal so far).
- About 0.6-0.8 fps is far from playable. What limits a frame now, and what's next, is in [discussion #675](https://github.com/boykopovar/AnyPS5/discussions/675).

## What it carries on top of `main` 2026-10-26

- Most of the upstream PRs this branch adopted earlier are part of `main` now, including #9957, #9181, #9567, #9558, #9600 and #11120. Still carried here because they're open upstream: mrfelek's #10228, SleglJan's #8887, AlxcBTW's #7514, mitul72's #6188 (off in the env) and our own PRs (#10382 and the dispatch-overlap and staging series).
- Changes on `main` after 2026-10-26 (over 100 PRs, including #11365 per-draw GPU timing, #8896, #10837 and #10980) aren't in this build yet.
- Local work:
  - image tables resolved on the GPU through a bindless descriptor heap, including T#s loaded from run-time addresses;
  - a cheaper draw-miss path (resolve memo, faster table snapshots);
  - texture revalidation that walks only the readable parts of partly committed memory;
  - the HTILE mask-clear kernel's buffer-descriptor form recognized;
  - a draw preparer thread with a two-phase draw build;
  - device-local buffers for address-based compute builds (the cloth programs);
  - copy-backs that store only the changed dwords;
  - dispatch overlap, and dispatch runs that keep staged results and order;
  - the async compute queue on a compute-only queue family, with buffers shared across both families, and fixes for its cross-queue ordering (consumed-label edges, edges that never hold back earlier work, lease waits that cover both queues);
  - indirect-draw parameters read on the GPU for draws whose records are still being written;
  - an exact f32 path for `v_fma_mix` behind a fused-FMA startup check;
  - presenter revalidation prepared off the GPU lock, and a cheaper image-table miss path;
  - batched staging: a dispatch run's copy-ins in one prelude ahead of it and its copy-backs behind one barrier (on NVIDIA any barrier between dispatches serializes them);
  - an in-run A/B switch registry;
  - Pipodi's absent runtime formats (Celegans12/AnyPS5-Ghost-of-Yotei#1) and hritikutekar's straight-line format conversion and single span reads for runtime-format buffer loads;
  - the PipelineJournal and shader-cache size checks set for GCC/glibc, for Linux builds (the build before this one built with GCC 16 for a tester; the values for this `main` are computed, not yet built on Linux);
  - owned reuse of vertex/index inputs and draw snapshots in host-imported memory the write tracker doesn't watch (same per-block word proof as the texture checks);
  - storage write-backs before a label wait only for the queue lanes whose batches write or lease the stored range;
  - shahulmubeen-a's HTILE note for cache-hit dispatch items (Celegans12/AnyPS5-Ghost-of-Yotei#2) and a fix for second-queue labels whose copied writer sits only in the async lane's batch;
  - owned texture proofs: the 64 KiB blocks under the committer's untracked texture snapshots are watched for writes (write watch, or, when unmapped, every map and commit path), each check good for the rest of its collect epoch, so a snapshot proof is a few word reads instead of a memory walk; no write protection and no guest faults;
  - fixes for three slowdowns that `main` added in the draw committer:
    - small device buffers get their own pool tier;
    - an exact-size buffer is reused instead of rebinding another one;
    - a flush with nothing to store skips the held-read check;
  - mitul72's draw-ahead from #6188 (off in the env above);
  - Izamaru's #9558 (pipeline-cache save temporaries released after use, about 324 MiB lower peak commit during a save);
  - address-table templates: read-only draws that read memory through pointers reuse their cached build, with the pointer part built fresh each time;
  - the read-only lease skip (#10382): a texture refresh no longer waits for address-table builds that only read the memory they leased;
  - SleglJan's #8887 (a storage flush waits only for writers over the layers it stores) and AlxcBTW's #7514 (texture refreshes answer registry and host-import lookups from an index and a cache);
  - from hritikutekar in #675: BDA address ranges merged incrementally, GPU copies for reflection-probe slices, an address index for depth surfaces, and draw-cache evictions destroyed on the release thread; then shared and recycled BDA range lists, faster depth-surface lookups, a memo for refused preparations, pending-write scans over distinct ranges, a heap merge in the shader-memory collect, small pool tiers that keep 4x their budget, and no CPU writer wait before storing host-imported storage images;
  - from contributors in #675: shahulmubeen-a's register lists read at any byte address (#9600, an open-world crash); hritikutekar's texture-cache frame age (entries age by presents while a frame takes seconds) and his second queue on a compute-only family when the graphics family has one queue (RADV); hritikutekar's prepared-table count in the texture service and slab-backed texture snapshots (here only for images that fill at least 3/4 of their slot, with slots counted at full size against the budget; unrestricted slabs made the texture cache thrash on this tree); FardinHaque70's fix for an import unwatch hook that re-entered the import lookup (from hritikutekar's diagnosis), and Izamaru's page-state table allocated on demand (about 250 MB less commit charge).

## Updates

- 2026-10-19: branch created at `8ba98278a` (async compute queue), 5.0 presents per 10 s in the world.
- 2026-10-19: moved to combo19 (`75dd18755`): matches the title reference exactly; against the previous build in the same session +5 % presents, -8 % frame time, -40 % CPU waits for GPU results. New shader-cache version, so the first run compiles cold.
- 2026-10-19: added batched staging (`ace5c14cd`): T1 exact; in-run A/B in the world 5.12 vs 4.68 presents per 10 s, GPU busy 810 vs 1,096 ms per frame. Same shader cache as combo19.
- 2026-10-20: added three fixes for dispatch overlap with staged buffers (`3cbec3212`). Each had a test that failed on the previous build: a reused build lost its results, two atomic adds became one, and a read saw data from before the run. Also FardinHaque70's import re-entry fix and Izamaru's sparse page table. T1 exact, same shader cache. Speed is within noise by presents (in-run A/B: about 1 %), GPU busy +14 %. In the same A/B, the only GPU hang came in a phase with the fixes off.
- 2026-10-21: added a known issue: GPU hangs with `APS5_ASYNC_LANE=51` (twice in the world on this build). Leave it out if you hit them.
- 2026-10-21: moved to our tree merged with `main` 2026-10-20 (`020d736b15`). In a same-session pair with the second queue off on both, 4.99 vs 4.19 presents per 10 s (+19 %) and a 2,070 vs 2,372 ms median frame (-13 %). It reaches the world sooner too: frame 85 at 334 s vs 478 s. T1 exact. **New shader-cache version, so the first run compiles cold.** `APS5_ASYNC_LANE=51` is out of the env until its GPU hang is fixed.
- 2026-10-21: the second queue is back (`63700c68ef`, same shader cache, T1 exact). Its "hang" was a DMA page fault on the GPU tied to the graphics family's third queue: the same work on a compute-only queue family ran clean in 4 of 4 world runs, up to 16 minutes each. The queue now comes from a compute-only family by default, and `APS5_ASYNC_LANE=51` is back in the env. Same-session pair, second queue on vs off: 5.89 vs 5.45 presents per 10 s (+8 %), 1,564 vs 1,852 ms median frame (-16 %). These numbers are **provisional**: an Nsight Aftermath monitor was running during both runs. Also three more cross-queue fixes, each with a test that fails without it.
- 2026-10-22: moved to our tree merged with `main` 2026-10-21 (`557682ca44`), with hritikutekar's two committer fixes from #675. In a same-session pair against the previous build plus the same fixes: 5.20 vs 5.13 presents per 10 s and a 2,065 vs 2,071 ms median frame, so the same speed on a newer `main`. On the previous build the two fixes were worth +11 % in an in-run A/B (6.26 vs 5.62). T1 exact. **New shader-cache version, so the first run compiles cold.** Three `main` paths that broke Yōtei are off here and listed under [Run it](#run-it). These numbers were measured with no debugging tools running.
- 2026-10-22: dispatch overlap now checks access ranges (`4212ed0d36`, same shader cache, T1 exact): a dispatch joins an overlapped run only when what it reads and writes doesn't overlap what the run reads and writes, the rule CnRJay asked for on #9176. Without it, about 40 % of the joins in the world (28,969 of 42,749) had an access conflict. In an in-run A/B the check costs no presents (-0.15 ± 0.27 per 10 s) and about 43 ms of GPU time per frame; frames were clean either way. `APS5_NO_OVERLAP_RANGE_CHECK=1` goes back to the old rule.
- 2026-10-22: three fixes from #675 (`18bffe373a`, same shader cache, T1 exact): shahulmubeen-a's #9600 (register lists at any byte address; an open-world crash), hritikutekar's texture-cache frame age (in-run A/B 6.09 vs 5.86 presents per 10 s, about 1 SE; his big gain was with the cache over budget, which ours no longer is), and his second-queue fix for GPUs whose graphics family has one queue (RADV; NVIDIA unchanged).
- 2026-10-22: added mitul72's draw-ahead from #6188 (`3957893917`, same shader cache), **off by default** (`APS5_NO_DRAW_AHEAD=1` in the env). With it on, queue 0's capture drops from 82 to 31 µs per draw, but most of that comes back as waiting for the committer: in an in-run A/B, 5.74 vs 5.44 presents per 10 s, within noise. The title check matched with it off every time and with it on in 3 of 4 runs; the one miss stalled at boot while the workers compiled about 10k entries and hasn't come back.
- 2026-10-23: next round of committer work (`65427e8b2b`, same shader cache, T1 exact; new env line `APS5_DEFER_REFRESH_FLUSH=1`): address-table templates, Izamaru's #9558 and hritikutekar's four changes from #675. In-run A/Bs on this build, change on minus off: incremental BDA ranges +0.16 ± 0.19 presents per 10 s, slice copies + depth index + deferred evictions +0.24 ± 0.25, templates +0.25 ± 0.27. Each is within noise on its own and none is negative; frames clean in every phase. The incremental BDA merge was checked against the full merge for ~1,060 s in the world with 0 differences.
- 2026-10-23: added the read-only lease skip (`085bd8f203`, same shader cache, T1 exact; upstream as #10382). A texture refresh used to wait for every address-table build that leased its memory, even builds that only read it; in the world that was about six waits of 35-200 ms per frame. In-run A/B on this build (1,300 s, 9 phases each): +0.19 ± 0.15 presents per 10 s (+2.8 %) and frames about 38 ms shorter, within noise; on the build before the templates it was +0.41 ± 0.33. Frames clean in both phases. `APS5_NO_READ_ONLY_LEASE_SKIP=1` turns it off.
- 2026-10-24: hritikutekar's second committer batch from #675 (`98e1a55cf2`, same shader cache, T1 exact): BDA range lists shared and recycled instead of copied, depth-surface lookups that skip addresses no surface starts at, refused preparations answered from a memo, pending-write scans over the distinct ranges, a heap merge in the shader-memory collect, small pool tiers that keep 4x their budget, and host-imported storage images stored behind their writer on the GPU without a CPU wait. In-run A/B (all seven on vs off, 9 phases each): +0.18 ± 0.06 presents per 10 s (about +2.8 %, 3 SE) and frames about 40 ms shorter; frames identical in both phases, and the heap merge matched the old merge in about 10.5 million collects.
- 2026-10-24: added SleglJan's #8887 and AlxcBTW's #7514 (`06ef2cbc1f`, same shader cache, T1 exact). In-run A/Bs on this build (1,300 s, 9 phases each, change on minus off): #8887 +0.03 ± 0.06 presents per 10 s (neutral), #7514 +0.15 ± 0.12 (about +2 %, frames ~25 ms shorter). Frames clean in every phase. Our perf-evidence runs for the six PRs we have open are linked from those PRs; the round showed that separate runs drift too much (up to 22 % between two unchanged runs) to resolve changes this size, so the switches are now measured inside one run.
- 2026-10-24: new env line item `APS5_PIPELINE_DISPATCH=1` (same build, same shader cache, T1 exact with it). A dispatch whose program is already in the dispatch cache no longer makes queue 0 drain the draw pipeline first; it goes through the pipeline as an item instead. Two in-run A/Bs (9 phases each): +0.18 ± 0.16 and +0.54 ± 0.53 presents per 10 s with it on, pooled about +0.21 ± 0.15 (+3 %), frames 31 and 78 ms shorter. Leave `APS5_PIPELINE_DISPATCH_MISSES` off: with misses pipelined too, the committer becomes the limit and frames get slower (-0.26 ± 0.10).
- 2026-10-24: moved to our tree merged with `main` 2026-10-24 (`a39be6252d`, includes #6721 and #8992) plus mrfelek's #9957 (buffer V# presence read at run time) and #10228 (image table for a T# read at a first-lane offset) (`4aae313e38`, **new shader cache**, T1 exact cold and warm). Linux: the PipelineJournal `State` size check is now right for GCC/glibc (ported from hritikutekar's linux branch; not yet built on Linux by us). Two same-session pairs against the previous build, both with the README env: frame p50 1,917 / 1,818 ms against 1,896 / 1,994 ms, frame 90 at 521 / 405 s against 428 / 427 s (the first new-build run came right after the cold run). Pooled that's even (median frame about 4 % shorter, GPU time about the same), so this move is about being on a newer `main`. Upstream's new sparse heap mirrors stay on: `APS5_NO_SPARSE_HEAP_MIRROR=1` measured the same.
- 2026-10-25: owned texture proofs (`9d0ee8e1af`, same shader cache, T1 exact). The draw committer used to re-prove every "untracked" texture snapshot by walking its memory, about 3,000 times a frame with almost nothing ever changed. Now the 64 KiB blocks under those textures are armed: write-protected (or, for unmapped blocks, watched through every map and commit path), and any CPU store or mapping change bumps a per-block word before it lands. A proof is then a few word reads. Driver, DMA and GPU writes are still caught by the write trackers as before. About 97 % of the proofs take the fast path. In-run A/B (`APS5_NO_OWNED_PROOFS` as B, 9 phases each): **+0.61 ± 0.06 presents per 10 s** with them on (about +8.5 %), frames about 100 ms shorter, the committer 4 points less busy; the verify mode (`APS5_VERIFY_OWNED_PROOFS`) found 0 disagreements and there were 0 faults. `APS5_NO_OWNED_PROOFS=1` turns it off.
- 2026-10-25: shahulmubeen-a's HTILE note for cache-hit dispatch items (Celegans12/AnyPS5-Ghost-of-Yotei#2) and our fix for the second-queue label crash (`5becc82a97`, same shader cache, T1 exact, focused ctest 388/390 with the two known failures). In a 4-arm in-run A/B the HTILE note was +0.13 ± 0.05 presents per 10 s (+1.8 %, frames ~23 ms shorter); in the same run hritikutekar's in-place join (`8106f78f5`) cost 0.18 ± 0.09 presents although it saved ~47 ms of GPU time, so it isn't in this build, and `APS5_DISPATCH_ITEM_BDA_WRITES` and the GPU-ordered write-backs were neutral here.
- 2026-10-26: owned reuse of draw inputs and a lane-targeted storage flush (`6debc980ee`, same shader cache, T1 exact with both verify modes, focused ctest 391/394: the two known failures plus `agc_write_tracking_metadata_commit`, which failed once under the full parallel suite and passes 30 of 30 repeats on this and the previous build). Vertex/index inputs and draw snapshots in host-imported memory that the write tracker doesn't watch were copied again on every draw; they are now kept and reused while per-block words armed on their memory show no store (the same proof as the texture checks). Storage write-backs before a label used to wait for every batch on both queues; they now wait only for the lanes whose batches write or lease the stored range. In-run A/B (6 pairs per arm): all three off -0.34 ± 0.07 presents per 10 s (+4.4 % with them on, frames ~55 ms shorter); vertex inputs alone +3.9 %, the flush alone +2.0 %. 2,035 reuse verifications found no difference.
- 2026-10-26: correctness move (`3243b61759`, same shader cache, T1 exact, focused ctest 392/394 with the two known failures). The owned proofs and owned draw/vertex inputs relied on guest-thread write faults, which upstream removed (#5596, #9181) because the exception frame overwrites the System V red zone (#361). They now use write watch instead, and upstream's #9181 is cherry-picked (resident runs are no longer write-protected). In an in-run A/B the write-watch versions were neutral (each arm within ±0.1 presents per 10 s once two stalled phases are left out; 0 faults, 0 verify differences), so the README env turns them off. In that session the build ran at about 7.8-8.2 presents per 10 s with the owned features on or off; the fault-based versions had measured +8.5 % and +4.4 % in earlier sessions, so some of that is likely lost, but separate sessions can't show how much. A cheaper fault-free proof is the next thing to try.
- 2026-10-26: owned answers from per-block checks within a collect epoch (`d3e0db3032`, same shader cache, T1 exact). The write-watch proofs asked Windows for the written pages on every proof, about 15 queries each. Now each collect epoch takes a mark from a shared clock, and a block that was checked after the mark is answered from its word alone, so most proofs are a few atomic loads (0.04-0.2 µs instead of up to 125 µs). A CPU store before the thread's ordering point is seen by the next query; a store racing a submission waits for the next mark, which is the rule the collect memo already had. **The owned features are back on in the env.** In-run A/B (2,400 s, 6 phases per arm): all owned features off -0.50 ± 0.04 presents per 10 s (**+6.4 %** with them on, frames ~80 ms shorter), the epoch checks alone off -0.19 ± 0.09 (+2.5 %), owned vertex inputs alone off +0.01 (neutral). 0 guest faults, 0 epoch disagreements. `APS5_NO_OWNED_EPOCH=1` goes back to one query per proof; `APS5_VERIFY_OWNED_EPOCH=<n>` also queries the first n epoch answers every 10 s and reports any disagreement.
- 2026-10-27: moved to our tree merged with `main` 2026-10-26 (`416ce3c368`, **new shader cache**, T1 exact cold and warm). `main` now carries #9957 (buffer presence read at run time) and #9181, so those collapse into `main`'s versions; #10228 stays ours. Also Pipodi's absent runtime formats (Celegans12/AnyPS5-Ghost-of-Yotei#1, off with `APS5_NO_ABSENT_RUNTIME_FORMAT=1`) and hritikutekar's straight-line format conversion and single span reads from #675 (`APS5_NO_COMPACT_FORMAT_CONVERT=1`, `APS5_NO_SINGLE_FORMAT_READ=1`). `main`'s PlayGo now throws on chunk IDs outside the scenario list; without `APS5_PLAYGO_SCENARIO_CHUNKS` every chunk answers installed again, as before. Three 1,500 s world runs back to back (this build, the build before, this build): 7.87 / 7.75 / 7.92 presents per 10 s over the steady phases (one 60 s phase of the first run, a compile stall, left out), 1,742 against 1,743 presents in total, so the same speed. Frames clean, no fatal. Focused ctest 448/451 (one fails on plain `main` too, one only with the opt-in `APS5_LAYER_REFRESH=1`, one known NVIDIA-only).

---

# About AnyPS5

Tool for automatic executables porting to Linux and Windows.

The relinker-only build runs on macOS, including Apple silicon.

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

Per-game status and test reports live in the [Game bring-up](https://github.com/boykopovar/AnyPS5/discussions/categories/game-bring-up) discussions, one thread per title.

## Input mapping

SDL-mapped game controllers are supported, including analog sticks and triggers. Keyboard and mouse controls can be configured with an `anyps5-input.ini` file. See [input mapping](docs/user/INPUT_MAPPING.md) for the supported devices and configuration format.

## Disclaimer

This project is intended for interoperability, research, preservation, and compatibility purposes. It does not include, distribute, or require copyrighted software, firmware, cryptographic keys, or proprietary libraries. Users are responsible for ensuring that any binaries used with this project are obtained and used in accordance with applicable laws and their respective license terms.

AnyPS5 is an independent project and is not affiliated with, authorized or endorsed by Sony Interactive Entertainment. "PlayStation" and "PS5" are trademarks of Sony Interactive Entertainment Inc.

## License

This project is licensed under the GNU General Public License version 2 only.
