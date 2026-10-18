<h1 align="center">Ghost of Yōtei on AnyPS5</h1>

<p align="center"><b>A test branch of <a href="https://github.com/boykopovar/AnyPS5">AnyPS5</a> that runs Ghost of Yōtei (PPSA26344) into the open world.</b><br>
<sub>Branch <code>yotei/best</code> on Celegans12/AnyPS5 · progress thread: <a href="https://github.com/boykopovar/AnyPS5/discussions/675">discussion #675</a> · details: <a href="YOTEI.md">YOTEI.md</a></sub></p>

<p align="center"><img src="https://raw.githubusercontent.com/Celegans12/AnyPS5/bringup-media/yotei/2026-10-19/open-world-async-lane.png" alt="Atsu at the Nupur river in the open world of Ghost of Yōtei on AnyPS5, frame read back 747 s after launch" width="880"></p>

> [!NOTE]
> This is a work-in-progress test branch, not upstream code. It is based on `main` from 2026-10-15 and carries local
> rewrites that haven't been reviewed upstream. It moves forward only when a build renders our 60-frame title reference
> exactly and runs faster in the open world than the build before it.

## Where it stands (2026-10-19)

| | |
|---|---|
| **Furthest point** | Open world, the Nupur river by the Ainu village, with HUD and quest |
| **Time to the world** | about 10 minutes from a save near Teshio Ridge (warm shader cache) |
| **Frame rate in the world** | about **0.5 fps** (5.0 presents per 10 s) |
| **Stability** | 25 minutes in the world without a crash on earlier builds of this line |
| **Tested on** | Windows 11, RTX 3080 Laptop GPU 16 GB (driver 610.62), 32 GB RAM, MinGW-w64 GCC 15.2.0 |

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
| 10-19 | **5.0** | the cloth compute ring on its own Vulkan queue |

## Run it

Build as usual ([build instructions](docs/dev/BUILD.md)), then run the game with:

```
APS5_SPANNED_WAVE=1 APS5_BINDLESS_IMAGES=1 APS5_BINDLESS_LAZY=1 APS5_NO_RESULT_MEMO=1
APS5_STAGING_DIFF=1 APS5_PIPELINE_INDIRECT=1 APS5_NO_ADDRESS_DRAW_REUSE=1 APS5_ASYNC_LANE=51
```

- The first run on a new shader-cache version compiles shaders cold; it can take 30-60 minutes to reach the world.
- Leave `APS5_PIPELINE_REQUESTS` unset when you measure: it costs about 12 % of a frame here.
- Every local change has an off switch (`APS5_NO_...`), named in its commit message.

## Known issues

- Black shapes on some rocks and terrain (the depth pre-pass and the G-buffer pass disagree on depth). Cause not found yet.
- Process memory reaches about 34 GB in the world, so 32 GB of RAM is tight.
- About 0.5 fps is far from playable. What limits a frame now, and what's next, is in [YOTEI.md](YOTEI.md) and #675.

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
