ZLUDA is a drop-in replacement for CUDA on non-NVIDIA GPUs. ZLUDA allows running unmodified CUDA applications using non-NVIDIA GPUs with near-native performance

<div align="center">

<!-- 80x28 104.75x28  62x28-->
[<img src="https://img.shields.io/badge/quick start-green?style=for-the-badge&logo=readthedocs&logoColor=white" width="267.5" height="56">](https://zluda.readthedocs.io) [<img src="https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white" width="209.5" height="56">](https://discord.gg/sg6BNzXuc7) [<img src="https://img.shields.io/badge/news-red?style=for-the-badge&logo=book&logoColor=white" width="124" height="56">](https://vosen.github.io/ZLUDA/)

<div/>

## Development practices in this fork (Apple GPU / AGX work)

This fork adds an Apple GPU target (PTX -> AIR -> AGX, Asahi Linux, M2 Max) and develops it against real CUDA
applications: Blender Cycles, llama.cpp and Geekbench 7. Practices followed during that work, recorded for
transparency (2026-11-07):

* **Reading application PTX to find failures and performance problems.** This is what upstream ZLUDA's
  `zluda_trace` module dumps (`module_*.ptx`, see the troubleshooting docs) are for.
* **Generic optimizations motivated by what real applications' kernels do.** Optimizations are written for any
  program, not for one application: for example constant-memory specialization in `zluda_agx`, and AGX backend
  passes (load hoisting for all-pairs loops, base + offset threadgroup addresses with 16-byte loads, byte-to-float
  conversion, nested-guard flattening) that were first developed on Geekbench 7's kernels.
* **Timing named applications and tuning defaults on them.** Upstream does the same: its cuDNN tests use PyTorch's
  convolution shapes and its docs cover individual applications.
* **Extracting PTX from an application's files.** In October 2026 the PTX modules that Geekbench 7 ships in its
  workload data file (`geekbench-workload.plxr`) were extracted from that file directly to compile and study them,
  rather than only as the application loaded them through the driver API.

Since 2026-11-06 the project has a stricter rule: PTX produced by NVIDIA's proprietary compilers, and code derived
from it, is no longer inspected or analyzed; such applications are only run and timed. Code analysis uses PTX from
open-source compilers (for example Cycles built with clang), hand-written kernels and ZLUDA's own library kernels.
ZLUDA contains no application-specific code paths.
