# Performance comparison

| Capture | Commit | Trace | Presents | Window (ms) |
| --- | --- | --- | ---: | ---: |
| baseline | 085bd8f20325 | aps5-trace-085bd8f20325-21492.json.gz | 16 | 35189.383 |
| candidate | 085bd8f20325 | aps5-trace-085bd8f20325-22740.json.gz | 16 | 31077.601 |

- Compare captures from the same machine, title and scene; summaries do not record the CPU, title or scene.
- Changes are candidate minus baseline. A negative frame-time change means a shorter frame interval.
- A missing top phase is unreported, not zero; its change is unavailable.
- Recorded configuration differs; these measurements do not isolate the code change.

## Configuration differences

| Setting | Baseline | Candidate |
| --- | --- | --- |
| env.APS5_NO_DISPATCH_OVERLAP | 1 | null |

## Frame and GPU times (ms)

| Metric | Baseline | Candidate | Change | Change (%) |
| --- | ---: | ---: | ---: | ---: |
| frame_ms.p50 | 2177.066 | 1931.814 | -245.252 | -11.265 |
| frame_ms.p90 | 2314.313 | 2173.967 | -140.346 | -6.064 |
| frame_ms.p99 | 2782.240 | 2187.742 | -594.498 | -21.368 |
| frame_ms.max | 2782.240 | 2187.742 | -594.498 | -21.368 |
| mean_frame_ms | 2199.336 | 1942.350 | -256.986 | -11.685 |
| gpu_busy_ms_per_present | 951.726 | 845.322 | -106.404 | -11.180 |

## Top phases (ms per frame)

| Metric | Baseline | Candidate | Change | Change (%) |
| --- | ---: | ---: | ---: | ---: |
| DrawPipeline.commit | 1396.893 | 1193.178 | -203.715 | -14.583 |
| Driver.Dispatch | n/a | 987.341 | n/a | n/a |
| Driver.Draw | 1243.574 | 1038.498 | -205.076 | -16.491 |
| Driver.Packet | 11075.374 | 9712.659 | -1362.715 | -12.304 |
| Driver.Packet.dispatch | 1350.259 | 1237.273 | -112.986 | -8.368 |
| Driver.Packet.draw | 1249.199 | 1043.430 | -205.769 | -16.472 |
| Driver.Packet.memory_wait | 7858.848 | 6865.413 | -993.435 | -12.641 |
| Graphics.Draw | 1267.176 | 1087.744 | -179.432 | -14.160 |
| Graphics.Draw.draw_and_resource_release | 1090.075 | n/a | n/a | n/a |
