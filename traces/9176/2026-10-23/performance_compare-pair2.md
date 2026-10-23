# Performance comparison

| Capture | Commit | Trace | Presents | Window (ms) |
| --- | --- | --- | ---: | ---: |
| baseline | 085bd8f20325 | aps5-trace-085bd8f20325-13716.json.gz | 16 | 35180.498 |
| candidate | 085bd8f20325 | aps5-trace-085bd8f20325-14880.json.gz | 16 | 31385.074 |

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
| frame_ms.p50 | 2158.311 | 1937.559 | -220.752 | -10.228 |
| frame_ms.p90 | 2470.962 | 2060.037 | -410.925 | -16.630 |
| frame_ms.p99 | 2544.854 | 2211.216 | -333.638 | -13.110 |
| frame_ms.max | 2544.854 | 2211.216 | -333.638 | -13.110 |
| mean_frame_ms | 2198.781 | 1961.567 | -237.214 | -10.788 |
| gpu_busy_ms_per_present | 1059.102 | 856.278 | -202.824 | -19.151 |

## Top phases (ms per frame)

| Metric | Baseline | Candidate | Change | Change (%) |
| --- | ---: | ---: | ---: | ---: |
| DrawPipeline.commit | 1395.699 | 1242.586 | -153.113 | -10.970 |
| Driver.Dispatch | 1155.940 | 982.310 | -173.630 | -15.021 |
| Driver.Draw | 1230.033 | 1079.260 | -150.773 | -12.258 |
| Driver.Packet | 11117.026 | 9884.848 | -1232.178 | -11.084 |
| Driver.Packet.dispatch | 1456.715 | 1271.660 | -185.055 | -12.704 |
| Driver.Packet.draw | 1235.832 | 1084.085 | -151.747 | -12.279 |
| Driver.Packet.memory_wait | 7757.510 | 6963.016 | -794.494 | -10.242 |
| Graphics.Draw | 1246.875 | 1125.270 | -121.605 | -9.753 |
