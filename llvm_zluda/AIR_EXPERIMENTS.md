# .ptx → .air (→ .agx): experiments and dead ends

A log of what was tried on the Air target (`compile_air.rs`, `lib.cpp`, the AIR/AGX backends it drives) and did
not work, or worked differently than expected, so that later attempts start from it. Newest first within each
section. GPU times are from the M2 Max behind the AIR service; "suite" is PTX-formalization's tests through
`ptx/tools/air_ptx_formalization_check.py` (default mode baseline 245–247 PASS, strict subset 18/18).

## Open problems

- **(Solved with real calls, below.) Blender Cycles to AGX code fully inlined does not fit in ~7.9 GB**, on
  either route. Whole-module: the old route
  (`--target agx`) reached 5.7 GB and the AIR route 6.5 GB before the watchdog stopped them (2026-10-30, current
  trees). The user recalls AGX code generation working at one point; no record of a Cycles AGX build was found in
  /scratch or the commit history (the per-kernel commit 3435e7a already says shade_shadow needs over 6.5 GB), so
  the working build may have been an earlier commit, another machine, or smaller programs. If it was an earlier
  commit, it is a regression to bisect (candidates: the AGX always-inline policy, the texture and soft-f64
  libraries, the AIR inliner's keep-function changes of 2026-10-29) rather than a limit to work around.
  Rerun alone (nothing else on the container, 2026-10-30): the old route still stops at 6.4 GB within 73 s,
  with or without `--object-only` (the IR text dumps are not the difference). Cycles' PTX only reached /scratch on
  2026-10-28 23:14, and the first AGX attempt after it (3435e7a) already ran out, so in this container Cycles has
  never compiled to AGX; a working build would have been on a machine with more memory. A kernel at a time (`ZLUDA_AGX_PER_KERNEL=1`), 12 kernels compile, then
  `kernel_gpu_integrator_shade_shadow` (about 5M instructions with everything inlined, as the AGX backend inlines
  every non-kernel function) runs out. Measured on that kernel alone with `llc`: straight-line strength reduction's
  scalar evolution took it from 0.9 to 4.4 GB (now skipped above 200k instructions,
  `-agx-straight-line-sr-max-instructions`; 2.8 GB without it), the load/store vectorizer adds ~1.2 GB, and
  StructurizeCFG then ran for over 25 minutes. Inside `zoc`, RSS only grew across the per-kernel loop (1 → 5.6 GB
  over the first 12 kernels) although each kernel has its own context. Not the allocator keeping freed memory:
  `malloc_trim(0)` after each kernel (with its module, diagnostics and context dropped first) left the curve the
  same (6.6 GB at shade_shadow); reverted. Nothing accumulates either: traced per kernel, RSS falls back to
  0.2–0.5 GB between kernels (shade_background peaks at 2.2 GB, shade_light_* at 1.2 GB). shade_shadow alone grows
  in steps: 0.5 → 1.9 GB in its first minute (IR passes), flat for 7 minutes (StructurizeCFG on one 5M-instruction
  function), 4.2 → 5.0 GB at ~8 minutes (instruction selection, by the pass order), 5.6 → 6.4 GB at ~17 minutes
  (the machine passes). It is the size of one fully inlined function in machine IR; pass tweaks will not bring
  it under 7.9 GB. What would: not inlining everything into the kernel (real calls in the AGX backend, now only
  `-agx-experimental-leaf-calls`), or a machine with more memory.
- One Cycles kernel, `kernel_gpu_integrator_shade_surface_mnee` (3.7 MB of AIR packaged alone), exceeds the AIR
  service's 120 s Metal compile cap. With the .ftz and texture fixes the other five of the six that used to time
  out now build (2026-10-30, wall time with the queue: shade_background 129 s, shade_surface 63 s,
  shade_surface_raytrace 110 s, shade_volume_ray_marching 38 s, shader_eval_curve_shadow_transparency 19 s; their
  AIR 40-55% smaller), so 74 of Cycles' 75 pipelines build. Metal's compile time for these kernels, not ZLUDA's,
  is now the slow part. Not over-inlining: shade_surface_mnee packaged alone keeps 162 functions, and
  `-air-keep-function-cost` 20000 / 5000 / 2000 left it byte-identical (its large functions are kept already).

## 2026-11-06: regressions on master, constant-memory specialization, occupancy defaults (M2 Max)

Blender Open Data samples per minute, classroom / junkshop / monster (sum; Metal's median for this chip is 1777):
2026-11-03 204 / 247 / 290 (741) -> 283 / 344 / 425 (1052) with the defaults of branch claude/perf-2026-11-06 and
llvm-project claude/occupancy-default. Runs on an 18 W charger with the battery supplying the rest; configurations
alternate, repeats within 0.3%. llama.cpp pp128 547-573, tg64 118-137, 4/4 exact throughout.

What master did first (fixed, all generic):
- **Cycles took over 20 minutes and 7 GB to compile** (2026-11-03: 23 s, 1.16 GB). 74334b13 inlined every
  mode-dependent callee, transitively, into every function that sets the floating point mode; Cycles' PTX switches
  between .ftz and other f32 operations everywhere (each switch is a set of the mode), so its whole call tree went into
  its kernels: 0.48 M -> 21 M instructions after the first pipeline. Bisected with a worktree and the instruction count
  of NAME.linked.ll (zoc without --object-only writes it). Now (LLVMZludaAirStaticRounding): the mode reaching each call
  is known bit by bit; a call needs its caller's mode only where a bit its callee can see differs; calls at round to
  nearest in other denormal modes go to a copy of the callee with those modes (at most four per function); functions
  given the mode as an argument start in the bits every caller passes (without that, 40 000 directed-rounding
  emulations where every call runs at round to nearest, and 223 s of AGX code). ~45 s, object 12.2 MB; PTX suite and
  the FP-mode regressions identical to master in both modes.
- **llama.cpp did not start**: zoc's air-verify (e082ab02) applied Metal's 31-buffer limit to portable AIR.
- **Cycles ran at 15 samples per minute**: kernels loaded on request registered the whole 13 MB module for launch-time
  specialization, rewritten and recompiled for every new set of scalar arguments.

What gained:
- **Constant-memory specialization** (docs/src/agx.md): `kernel_params` folded into the kernels, +10% (classroom
  210 -> 231). Images within 0.0007 mean of the generic code.
- **Register cap on call trees**: kernels with real calls declared 128. Cap 72 / 80 / 80 with callee-saved 8 (with
  callee-saved 16, scheduler target 48, 16 KB stack, blocks <= 192): 1054 / 1034 / 1036. Defaults now: the backend
  caps call trees (-agx-call-tree-max-gprs=72), zoc caps every kernel of a large module that defines functions
  (junkshop 321 -> 337), blocks up to 192 (+2.4%).
- **Refuted for leaf kernels**: the same cap on every kernel of llama.cpp: pp128 547 -> 309, tg64 122 -> 80. Leaf kernels
  declare what they use; a cap only spills there.

Open: Geekbench 25.85 -> 26.87 s of GPU time against 2026-11-03, mostly convolve_horizontal / vertical (+21%),
blur_gray (+37%), depthwise_convolution_2d (+25%); not the block size (the same with ZLUDA_AGX_MAX_BLOCK=1024) and
not the register cap (no calls there): something else changed since 2026-11-03 (other sessions' AGX backend commits
are the first suspects). Compile time: the rounding stage 8.7 s (0.4 s on 2026-11-03), O3 13.8 s (6.9 s).

## Compile time, memory and host costs on Asahi (2026-11-01)

No AIR service this session: GPU numbers are of the AGX code on the M2 Max through zluda_agx (Asahi DRM). Corpus:
Geekbench's 15 modules and llama.cpp's 17 (`/scratch/perf-goal/ctime.py`: one zoc per module as the driver runs it,
wall time, peak RSS, object md5).

- **The Metal copy is not needed on Asahi** (`zoc --agx-only`, the driver and every served request): the clone of the
  module for Metal, its passes and bitcode, and the portable AIR output. Objects byte-identical (30/30), corpus
  51.5 -> 49.3 s.
- **llama.cpp's 10 MB module (317 kernels): 28.6 s, 2.1 GB.** Serial: the first pipeline 9.8 s (InstCombine 55%, the
  first of its three runs 3.7 s; SROA 19%: 311k allocas), O3 in AIR's address spaces 9.5 s; the per-kernel AGX code
  6.7 s on 12 threads. The time is spread over 332 functions (median 3k lines), no outlier: what would shorten it is
  optimizing the module in parallel parts (the IPO passes, globalopt and ipsccp, would then see a part's globals only:
  unsound without care). Dead end: an early-cse / instsimplify / simplifycfg before the first InstCombine changes
  nothing (7.98-8.19 s for the prefix). `ZLUDA_AIR_DUMP_FIRST=FILE` writes the module before the first pipeline for
  `opt -time-passes`.
- **"Cannot select: store (s64) into %stack.0"** (that module as a whole, and per kernel with O3): two zero
  <2 x float> stores into one alloca, the DAG rebuilding a memory operand from the frame index with the stack's
  pseudo source value in address space 0. Fixed in the AGX backend (getAddressSpaceForPseudoSourceKind, llvm-project
  claude/perf-air-agx); llvm-reduce from the kernel to 6 lines. The default objects are unchanged by the fix.
- **Host costs that libcudart programs hit** (llama.cpp b11706 links the CUDA runtime statically, so its start-up is
  the runtime's): `ZLUDA_AGX_API_STATS` showed 1.75 of 2.5 s of start-up in cuMemsetD8Async, two 384 MiB memsets at
  0.45 GB/s (a pattern copy per element, length known at run time; a micro-benchmark with a constant pattern length
  had looked fast: measure the real call). Fill as typed slices: 866 -> 8 ms, start-up 0.72 s. Generation: per token
  ~4.7 ms waiting for the logits (GPU time) and ~3 ms in cuGraphLaunch overlapping it; GPU-bound.
- **AGX O3 per kernel for large modules** (`ZLUDA_AIR_AGX_PIPELINE=o3`, possible once the stack fix above was in):
  llama.cpp's 10 MB module +0.9 s (26.5 -> 27.4 s, the stage runs in parallel), same peak memory; llama.cpp
  tg64 133.1 -> 134.5 t/s (two runs each, run-to-run spread +-1.3), pp128 equal. Within noise: left off (Cycles' compile
  paid 31 -> 44 s for it before).
- **Small private arrays read at a run-time offset in registers** (LLVMZludaPromoteDynamicPrivateLoads, after the
  first pipeline): ggml's mul_mat_vec_f<half> keeps a float[8] written at constant offsets and read once at
  `[depot + 4 * lane]`, so the array went to the stack (80 bytes): stack setup ~7.5 us a launch plus scattered stack
  traffic (playground stack-launch, stack-memory). As a select over the 8 constant-offset loads SROA keeps it in
  registers: 46.1 -> 14.6 us and 53.8 -> 25.3 us per launch, llama.cpp pp128 445.6 -> 566.0 t/s, tg64 131.9 -> 133.9;
  completions exact, formalization suite unchanged. Found from the simulator's worst capture (-72%), whose gap
  turned out to be the stack.
  Then also stores at run-time offsets (arrays of at most 8 elements), the `or disjoint` InstCombine makes of an
  aligned array's address plus an offset, and phis of pointers of one offset (GVN's PRE): Geekbench's blend kernel
  loses its 64-byte stack; of the corpus' 589 kernels 3 keep one (trace_bvh / trace_bvh_shadow: BVH traversal stacks;
  calculate_descriptors: a 256-byte array). (The commit message of 59957a9 says 1,774 kernels: wrong, 589.)
  Geekbench with it: blend 725.5 -> 702.4 us per launch (-3%), summed GPU time 26.94 -> 26.87 s, the same 10 expected
  errors.
  Geekbench (2 of its modules change): every workload completes, the same 10 expected errors, summed GPU time 27.13
  -> 26.94 s; its changed modules' kernels within their run-to-run spread (keypoint kernels are data-dependent).
- Dead end, software-pipelined tile GEMM: loading the next k-step's 8 fragments before this step's 16 products
  (gen_tile_gemm.py, the variants that load all fragments first, and the f16 `_tb` one) passes the kernel tests
  but is 9-14% slower on the M2 Max (gemmbench, f16 A/B: 4096^3 6.99 -> 6.35 TFLOPS, f16 C 7.18 -> 6.20, 4096 x
  512 x 4096 6.26 -> 5.66): 8 more fragments live per SIMD-group, while AGX already overlaps the loads with the
  products. Large GEMMs run at ~7 TFLOPS (about half the FMA peak); more needs a different tile, not pipelining.
- Dead end, wider SIMD-group tiles: 32 x 64 per SIMD-group (4 x 8 matrices, 32 accumulators), two SIMD-groups per
  64 x 64 workgroup tile (GEN_TILE_J=8 in a trial generator, 64-thread launches): correct (1024^3 max relative error
  3.6e-6) but 7-25% slower (4096^3 6.99 -> 6.49 TFLOPS, 1024^3 5.38 -> 4.05, f16 C 7.18 -> 5.45): half the SIMD-groups
  per workgroup hide less latency than the better load-to-product ratio gains. A larger tile would have to keep
  four SIMD-groups per workgroup (a 128 x 64 or 128 x 128 workgroup tile and a new plan).
  Tried too: 128 x 64 workgroup tiles, four SIMD-groups of 32 x 64 (the plan cutting M' in 128s): as slow (4096^3
  6.48, 1024^3 4.02, f16 C 5.39 TFLOPS). So the 32 accumulators per SIMD-group (register pressure, fewer resident
  SIMD-groups) cost more than the reuse gains, whatever the workgroup shape: 16 accumulators (32 x 32) is the
  better point for these kernels.
- Dead end (as built), threadgroup-staged tile GEMM: agx_hgemm_tile64_cf{16,32}_tb with K in chunks of 32 staged
  in 8 KiB of threadgroup memory by the 128 threads (16 scalar f16 loads of A and of B each, zero past K), a barrier,
  the 4 x 4 fragments per SIMD-group from threadgroup memory (air.simdgroup_matrix_load from workgroup memory
  compiles and is correct: hgemm kernel tests 21/21, 1024^3 max relative error 3.6e-6), a barrier: 2.5x slower
  (4096^3 6.99 -> 2.75 TFLOPS, 512 x 512 x 4096 4.97 -> 1.30). The staging is the cost (scalar loads behind a bound
  branch, B's transposed stores into threadgroup memory, two barriers per 32 of K); fragments loaded straight from
  device memory are already efficient here. A competitive version needs vector loads, a conflict-free layout and
  double buffering (generator: /scratch/perf-goal/tggemm/gen_tg.py, not kept).
- Dead end, dp4a: a signed dp4a is 18 AGX instructions (4 imadd; the bytes extracted by bfi + asr pairs, asr 24 for
  byte 3). The free 16-bit sign extension (MOD_SX) exists only on add / multiply-add sources (IADD_I32_FROM16,
  IMADD_*_FROM32), not on shifts, so byte 1 (sext of the low half, then asr 8) is two instructions as now: no
  saving there. A real gain needs packed 8-bit arithmetic, which AGX does not have (see 2026-10-31 above).
- Geekbench 7 GPU with all of the above: every workload completes (the block-size search's out-of-resources answers
  as before), summed GPU time 27.13 s against 25.93-26.85 s of earlier AIR-route runs; its largest difference,
  particle 2.39 s, is inside those runs' 2.04-2.42 s.
- Blender default cube (1024x1024, 128 spp) with all of the above: first render (compiling Cycles) 52.1 -> 45.2 s,
  second 1.67 -> 1.63 s, the image the same (max difference 1/255, none over 2).
- **A module of independent kernels compiled in parts** (zoc, `parts`): llama.cpp's 10 MB module has 317 kernels and
  no other function, so its PTX splits into 12 parts (every part with the module's variables, the kernels balanced by
  size), each compiled on its own thread with its own context, the objects merged (the parts' global data are equal;
  the merger checks it, and any failure compiles the module whole). 22.1 -> 6.1 s, peak 2.4 -> 1.2 GB; the 7 MB one
  6.3 -> 1.4 s; the corpus 40.6 -> 19.3 s. A part takes the module's light+ / per-kernel O3 decision (by its own
  ratio, few kernels next to the linked helpers, it would have taken O3: the Q5_K kernel back at 122 us); with it the
  Q5_K kernel 116.7-117.2 us, the Q5_1 one 25.9-26.3 us (controlled replays), completions 4/4 exact. Cycles (383
  .func) and the other 28 modules are byte-identical. ZLUDA_AGX_PARTS=0 / N.
- **Light cleanup in AIR's address spaces, O3 per kernel** (`ZLUDA_AIR_AS_PIPELINE=light+` with
  `ZLUDA_AIR_AGX_PIPELINE=o3`, experiment): llama.cpp's modules 44.1 -> 36.7 s (the 10 MB one 26.5 -> 22.1 s; peak
  2.08 -> 2.44 GB, every compiling thread running O3), completions 4/4 exact; but GPU time not shown to be equal
  (tg64 134.6 / 123.7 against 132.8, pp128 -2%; per kernel +1-4% everywhere, even in modules it does not change,
  i.e. within the clock's variation). Controlled (captures of both builds replayed alternately, 1000 back to back,
  three times): mul_mat_vec_q<Q5_K> 121.7-122.6 -> 117.1-117.5 us (-4%), mul_mat_vec_q<Q5_1> 26.0 -> 26.2 us. But
  Blender Cycles (75 kernels sharing many functions, each kernel's O3 repeating them) gains nothing: 43.4-49.7 ->
  49.5-57.3 s, peak up to 6.9 GB. Left opt-in: it suits modules of independent kernels (llama.cpp's), not Cycles'.
- Dead end: compiling a lazily loaded library in the background when the runtime registers it (cuLibraryLoadData),
  to overlap llama.cpp's 17 module compilations on an empty cache (41.6 s): the same 41.6 s. The CUDA runtime that
  llama-app links asks for each library's module (cuLibraryGetModule) right after registering it, before it registers
  the next, so the compilations stay one after the other. Overlapping them would need module and kernel handles
  before compiling (compilation at the first launch), a change of the driver's module model. Reverted.
- Dead end: reading each kernel's module lazily from the portable AIR (getLazyBitcodeModule, materializing only the
  functions the kernel reaches, instead of parsing all 13 MB of Cycles per kernel): the same object, peak 6.1-6.3 GB
  against 6.3-6.7, the AGX stage 30.6-32.6 s against 26.6-32.1 (two runs each): within noise. Cycles' peak is the
  12 kernels being compiled at once (the largest ~1-2 GB each), not the parses. Reverted.
- cuDNN (agx): temporaries from a per-handle pool instead of an allocation, a synchronization and a free per call:
  3x3 convolutions 1.4-4x faster (8x64x56x56 4.52 -> 1.11 ms), the CPU no longer blocked on each call
  (`zluda_dnn_agx/tools/convbench.c`, checked against a CPU convolution).
- cuBLAS (agx) split K now also for f16 A/B with f32 C from K = 512: 896 x 64 x 896 98 -> 86 us, 128 x 512 x 896
  99 -> 82 us (`zluda_blas_agx/tools/gemmbench.c`). Skinny n <= 8 products stay slow (0.2-0.3 TFLOPS: the dot kernel
  reads A once per column), but ggml uses its own kernels there.

## What went wrong, 2026-11-03 (Blender Cycles session): read before repeating it

Policy, from the user: running nvcc-generated PTX through ZLUDA is fine; decompiling or reverse engineering it, or
disassembling cubins, is not. A regular Blender must work against the driver, with no application-specific code in
ZLUDA or LLVM.

Mistakes in how the work was done:

- **Replaced Blender's kernel out of an over-strict reading of the EULA note.** Stock PTX was set aside for a kernel
  built with clang, which cost most of an hour and changed the benchmark's input. Ask, or check the policy above,
  before changing what is being measured. The clang build still exists (ptx/tools/cycles_clang, /scratch/cycles-clang)
  but is not the benchmark.
- **Reported a comparison that had not run.** This container has no `diff`, and `diff -rq ... 2>/dev/null | grep -v`
  printed nothing, which read as "identical". Never send a checking tool's stderr to /dev/null. Use
  `git diff --no-index` or Python's filecmp. (Redone with filecmp: Blender 5.1's bundled kernel/ and util/ are
  identical to Cycles tag v5.1.0, apart from one extra OptiX file.) Also missing here: `which`, `/usr/bin/time`
  (a pipeline ending in grep then shows nothing at all), numpy outside Blender's Python, gdb, lldb.
- **Hid files in the Blender install to work around a driver problem.** Moving the cubins to lib-hidden-cubins hid
  the bug for weeks: a stock install failed. Fixed in the driver: it reports compute capability 8.0, which Blender ships
  no cubin for. **First try, 8.7 (no GPU has it), broke llama.cpp**: cublasCreate fails on an unknown architecture, found
  only by running the other benchmarks. Run llama.cpp and Geekbench too before calling a driver change done.
- **Theorized about the instanced-mesh BVH leaves from reading the source**, with no new evidence. Reading bvh2.cpp,
  build.cpp and geometry_bvh.cpp finds no path to (2, 4, 0, 0): the source is right. The answer was in the compiled
  host (below).
- **Three wrong guesses at compile memory, each stated before it was measured**: (1) every job parsing the whole
  module: lazy bitcode loading gave only 6.53 -> 5.92 GB; (2) allocator arenas keeping freed memory:
  MALLOC_ARENA_MAX=1 gave 5.89 GB at twice the CPU; (3) the same called functions compiled in every kernel:
  whole-module compilation with a module-wide frame layout (AGXFrameLowering, reverted) took 188 s at 1.48 GB, no
  better than ZLUDA_AGX_JOBS=1 (188 s, 1.26 GB), and the object was still 37 MB. The old "1338 copies, 5.3 MB unique"
  figure no longer describes this build. Look at symbol sizes first.
- **A wrong suspicion, the seed**: Cycles' seed seemed to do nothing (at most 1-2/255 at 16 samples, CPU and GPU), and this was
  put down to the host. The classroom file forces it: set to 12345 it reads back as 1 after the render (the log line that
  said otherwise printed before that), while the default cube responds normally (max 69). Read a setting back after the
  render before trusting it. No drivers or render handlers: the value holds through a view layer update and reverts only in
  the render, and the scene has an action (SceneAction), so almost certainly keyframed; the keyframe itself was not listed.

## Instanced geometry missing: a miscompiled host, not ZLUDA (2026-11-03)

The aarch64 Blender 5.1.0 build (ae6d847d66fa, .comment names GCC 14.2.1) miscompiles `BVH2::pack_instances`:

    41e2168: ldr q30, [x0], #16             // int4 data = leaf_nodes_offset[i]
    41e216c: add v30.2s, v31.2s, v30.2s     // data.x += prim_offset; data.y += prim_offset
    41e2170: str q30, [x1], #16

A .2s operation clears the upper half of the register, so every instanced mesh's leaf loses its visibility and primitive
type: the (2, 4, 0, 0) above, skipped by the traversal. Cycles' source is correct. Top-level leaves do not pass through
this loop, which is why only instances disappeared. GCC 15.2 compiles a small replica correctly (ldp d30, d29), and it
was not reproduced with a compiler here, so it is not established as a GCC bug. A scan of the whole .text finds this
one instance of the pattern.

Nothing belongs in ZLUDA or LLVM for this. Workaround: /scratch/.../blender-instfix, the same binary with bit 30 set in
the word at 0x41e216c (file offset 0x3de216c, 0x0ebe87fe -> 0x4ebe87fe, add .4s; v31's upper half is already 0). The
original binary is kept and the bench scripts use the copy. A rebuild with a fixed compiler is the real fix.
**Every Blender score from before this is for scenes with geometry missing.** Against the CPU (16 samples, 25%):
linked and collection instances now max 0; classroom mean 19.3 -> 0.34. Open Data, samples per minute (stock PTX):
monster 245 (was 466), junkshop 193 (160), classroom 146 (133).

Classroom GPU against CPU by samples, at the same seed: 16 mean 0.34 / max 139, 128 0.28 / 54, 1024 0.17 / 13, 4096
0.098 / 4, signed error about 0 throughout: no measured bias. ZLUDA_ROUNDING_STRICT=1 does not reduce the difference
(0.3455; 0.1712 against 0.1665 at 1024), and the CPU on BVH2 against Embree is 0.0004, so it is neither rounding nor BVH
layout. The faint trace along chair edges in the 4096 map is noise: the signed error averaged over blocks of 1, 4, 8, 16
and 32 pixels has rms 0.28, 0.071, 0.036, 0.017, 0.009, halving as uncorrelated noise does (blockdiff.py), a tenth of
the image's own sampling noise (1024 against 4096 samples: rms 3.0014 on the GPU, 3.0006 on the CPU). Junkshop and monster were compared at 16 samples only (mean
differences small, signed error about 0).

## Compile memory, the result (2026-11-03, Blender Cycles' stock kernel)

6.53 GB 44 s, object 44 MB -> 1.16 GB 23 s, object 9.5 MB, 51 s of CPU, with GPU time improving, not traded. In order of
effect: the software texture fallback out of line (`sample_f32_outlined`, 3.65 -> 1.86 GB, then 1.16 GB with the unit filtering: it was inlined at every tex,
and only textures the unit cannot sample reach it now), no 64-bit remainders in the sampler (4.96 -> 3.65), jobs on the
fast cores only (5.92 -> 4.96), lazy bitcode (6.53 -> 5.92). The cost was the code expanded per tex, not LLVM. An
out-pointer cannot be passed across a real call ("Cannot select: FrameIndex"): the fallback returns its four floats.

## GPU time of Blender Cycles, what moved it (2026-11-03, stock PTX, samples per minute)

Baseline with instances fixed: junkshop 193, classroom 146, monster 245. Same images and PTX suite at each step.

- **Call registers no longer reserved**: r2..r36 were out of register allocation in all code with calls. Calls and returns
  now use their argument and result registers explicitly (glued copies, implicit uses), and a function keeps only the
  links of its depth and shallower. 193 -> 198, 146 -> 151, 242 -> 246, repeated twice. The pop after a call was marked
  isBarrier, which it is not: invisible until a result stayed in its register and the pop ended a block.
- **Min / max read only by compares**: the slab test's took four instructions each (fcmpsel, or, fcmpsel eq, fmul by 1.0
  to flush), about 36 extra per BVH node. A compare cannot tell -0 from +0 or see a NaN's payload, and the GPU's compares
  flush. -> 204, 161, 261. **The existing LLVMZludaAirDropMinMaxFixups had silently stopped working**: it ran before the
  fix-ups took the form it matches (light+ replaced O3 as the first pipeline) and matched llvm.fabs, not air.fabs. A pass
  that returns a count should have that count checked by a test.
- **Linear filtering on the texture unit in default mode, exact filtering in strict mode** (the user's decision). Decoding each
  channel's format at run time was the cost. Tried first, then removed: the unit's point samples with software weights
  (228, 176, 266; bit-identical to the f32 software sampler, 11 forms). With the unit's filtering 246, 204, 286, and
  compile 1.16 GB, 9.5 MB. **"Hardware differs up to 0.3%" in older notes compared it with the f32 software sampler, not the
  model**: against strict mode the unit is within 1.2e-4, and the f32 sampler's far-out positions are what was 0.3% off.
  Compare with the reference, not the nearest other implementation. See docs/src/agx.md for the numbers per format.
- **A 64-bit remainder per axis per texel** in the sampler's mod_pos: no speed change, as wrapping is rare at run time, but
  compile 4.96 -> 3.65 GB. Static size and dynamic cost are separate questions: check both.
- **Called functions keep the top 48 registers** (llvm-project, -agx-callee-saved): no register survived a call, so everything
  live across one went to memory. 0 / 16 / 48 / 64 / 80: classroom 176 / 191 / 197 / 196 / 193, and 15% less compile CPU.
  The first try faulted on the GPU (Bad address, 8 recorded in gpu_safety): a kernel's register count came from its
  register masks, which no longer showed what its callees use. A kernel with calls now declares 128.
- **Result**: junkshop 193 -> 247, classroom 146 -> 204, monster 245 -> 290 samples per minute (+28%, +40%, +18%). Strict mode:
  junkshop 179, classroom 148. Compile 1.16 GB, 23 s, object 9.5 MB. llama.cpp pp128 425-441, tg64 125-135 t/s across runs,
  4/4 completions as on the CPU. Geekbench, alternating the compiler from before these changes with this one on the same
  day: 26.98, 26.12 s -> 25.60, 25.75 s of GPU time (later runs 25.6-26.8). Older sessions' 24.8-25.3 s are not comparable.
- **WRONG CONCLUSION, later reversed: register cap (-agx-default-max-gprs)**: recorded here as "occupancy is not the limit" from 128: 194, 96: 161, 72: 131, 56: 96. That test was flawed: the call ABI then reserved r1-r36, so a cap of 96 left about 59 usable registers and spills swamped the gain. With only r0, r1 and the links reserved a cap of 80 gives +15-20%: see "Occupancy" below. Lesson: a refuted result is only as good as what else the test changed.
- **Refuted, more inlining**: -inline-threshold 500 and 2000 are within noise and cost compile time. -air-keep-function-cost
  does not change AGX code at all (17.9129 MB at 5000, 80000, 320000): what O3 leaves uninlined stays a real call.
- **Refuted, backend options**: on classroom and junkshop -agx-uniform-constants, -agx-max-unskipped-instrs=2 and 8,
  -agx-atomic-optimizer, -agx-element-aligned-vectors, -agx-mem-cluster=32 and -agx-launch-divisors are all within about 1%.
- **Refuted, register allocator options**: -exhaustive-register-search, -enable-local-reassign, -greedy-reverse-local-assignment and
  -agx-postra-cluster-wait=0 are within 1%; -split-spill-mode=size is 1-2% worse; -enable-misched=false costs 7%, so the
  machine scheduler earns its time. -consider-local-interval-cost and -disable-spill-hoist gave no result (not looked into).
- **Refuted, stack spacing**: 12 KB against 32 KB per thread is the same speed (a memory gain, above).
- **What is left is spread out**: replaying shade_surface (junkshop, 30.8M instructions, agx_run --trace-pcs), the largest
  part is mov_imm 9%, fmul 9%, stack_load 7%, jmp_exec_none 7%, wait 7%, and no function but the interpreter over 15%.
- **Refuted, the GPU clock in a render**: a lone launch runs 2.4-3.2x slower than with the GPU kept busy, but a render is not held back by
  that. A small kernel kept running in another process: classroom 204 -> 198, junkshop 245 -> 238, and back without it. Keeping
  the clock up from the driver would not help this workload; it may help applications with long gaps between launches.
- **Refuted, allocator cost of callee-saved registers**: -regalloc-csr-first-time-cost 5, 50 and 500 are within 1%.
- **Refuted, launch overhead**: launch batching changes nothing (197.9 vs 197.4). 18.8 s GPU in about 19.7 s of launches:
  this is GPU-bound. An earlier "19% not GPU" was wrong, its span included 3.5 s of scene setup.

## Compile memory: what it is (2026-11-03, stock PTX, per kernel with real calls)

Of 36.4 MB of AGX code, 34 MB is 14 functions of about 2.4 MB each: Cycles' shader interpreter with its cases inlined, one
per template variant, already inlined in the PTX. Compiling one holds about 0.5 GB, so the peak is simply how many run
at once. Time (llc, whole module, 131 s): instruction selection 22%, StructurizeCFG 19%, register allocation 16%, of
which global splitting 11 s. No pass dominates.

Jobs against peak and wall: 1: 1.26 GB 188 s; 2: 1.71 GB 101 s; 4: 2.84 GB 63 s; 6: 3.94 GB 54 s; 8: 4.97 GB 45 s;
12: 5.89 GB 48 s. Past the 8 performance cores, jobs only hold memory for longer on efficiency cores, so the default is
now the fastest cores' count: 6.53 GB 44 s -> 4.96 GB 44.8 s, the object byte-identical. For less memory still,
ZLUDA_AGX_JOBS trades it for time as above. Merging identical functions at IR level is untested: `opt -passes=mergefunc`
on stock.opt.ll did not finish in 280 s.

Also a dead end: llc cannot read stock.portable.air ("Invalid call record"), and -time-passes through
ZLUDA_AGX_LLVM_ARGS prints nothing. Profile with llc on zoc's stock.opt.ll (run zoc without --object-only).

Dead ends and traps:

- **Standalone Cycles host: not buildable here.** Its required OpenImageIO, OpenEXR, Imath, pugixml and TBB headers
  are absent, and Blender's lib/ has only the .so files. No network to fetch them: https to github.com,
  projects.blender.org, pypi.org and Fedora mirrors fails, ssh.github.com:443 is refused, sudo needs a password.
  `dnf repoquery` still answers, from its cache, and `dnf download` then fails, so a repoquery result does not mean
  packages can be had. Sources arrive only when the user puts them in /scratch.
- **`pgrep -f NAME` and `pkill -f NAME` match their own shell's command line**: a wait loop never ended (a finished test
  suite sat waited on for several minutes), and pkill killed the command running it. Use -x, or wait on the output.
- **Pushing was refused all session** (all four keys). /scratch/save-repos.sh after each commit, and retry.
- **A commit needs an identity**: none is configured. Use GIT_AUTHOR_* / GIT_COMMITTER_* for one command, not git config.
- **Copying ext/HiGHS from /scratch/ZLUDA brings its .git file**, which breaks `git status` ("not a git repository:
  ext/HiGHS/../../.git/modules/ext/HiGHS"). Delete that file. The .bc files in a fresh checkout are LFS pointers: copy
  them from /scratch/air-pipeline/lfs and mark them skip-worktree.
- **A fix can exist only in the build tree.** /scratch/llvm-agx-main had the stack pseudo source value fix that
  llvm-project lacked. Compare the two trees before relying on either.
- **Building Cycles' kernel with clang, what failed first**: __clang_cuda_intrinsics.h needs CUDA's own definitions;
  system headers break it (-nostdlibinc, with empty stubs); `restrict` is not C++ (__restrict); ballot needs
  `-Xclang -target-feature -Xclang +ptx87`; `sat.f32` is not a PTX instruction; ZLUDA's parser takes
  `tex.2d... {r,g,b,a}, [t, {x, y}]` only. The prelude's math is our own approximations, so a difference from the CPU
  render may come from it.
- **Unfinished: compile memory of the clang-built kernel.** The AGX stage reached 8.3 GB on one remaining kernel, with
  the other threads done, when it was stopped (55 of 75 timed in the second run, which was also stopped; which kernel
  is not known). The stock PTX compiles in 53 s. Do not compare clang-kernel numbers with stock ones.

## Real calls in the AGX backend (`-agx-experimental-leaf-calls`), 2026-10-30

The way out of Cycles' AGX memory: shade_shadow with its large helpers kept as called functions compiles in 14 s at
156 MB (fully inlined: 6.4 GB and over 25 minutes in StructurizeCFG without finishing). What was in the way, in
the order it was found:

- **A crash at once on Cycles**: AGXAlwaysInline's call-depth computation held a DenseMap iterator across an insert
  into the same map (fine for the small lit tests, not for 120 functions).
- **63 of the 71 refused helpers read global data** (`kernel_params`, lookup tables, `switch.table.*`), which only a
  kernel can address (a uniform register). AGXGlobalDataArgs now passes the buffer's address as an extra i64
  parameter to the kept functions that need it and their callers; their global references become address + the
  variable's offset in computeGlobalDataLayout's layout. (A first test module made with `llvm-extract --recursive`
  had every global as a declaration, which gives the same "unsupported external global" error: extract with
  `--rglob='.*'`.)
- **Callees before the kernel in .text**: modules from AIR list functions before kernels, and a kernel's program
  runs from its symbol on, so a call jumped backwards out of it (indirect.ptx: "targets -52 outside the program").
  Called functions are now moved after every kernel.
- **The runtime and the harness loaded only the kernel symbol's own bytes**; with called functions in the module,
  a kernel's code now extends to the end of .text in agx_runtime and agx_ptx_formalization_check.py.
- **The simulator rejects `ret`** unless `AGX_EXPERIMENTAL_RET=buffer-pop2` (the return model measured on 570
  hardware cases); call timing is not calibrated. Most PTX suite programs keep no calls (the AIR inliner inlines
  functions under 100 instructions), so passing tests there say little about calls: indirect.ptx is the one that
  exercised them.

- **Stack frames over the fixed 4 KB per call depth** (whole-module Cycles: shade_volume's kernel frame 6.3 KB;
  per kernel: integrate_surface's callee frame 6.9 KB). Frame offsets are 15-bit, so all depths must fit in 32 KB.
  `-agx-measured-call-frames` (a module of one kernel, so per-kernel compilation): the kernel's region is its
  frame, each depth's is the largest frame at that depth, recorded as called functions compile in increasing
  depth after the kernel; the kernel reserves all 32 KB. A larger kernel region alone, with fixed 4 KB depths,
  was not enough (callee frames over 4 KB too).
- **AGXGlobalDataArgs appended the rewritten functions to the module**, which undid the depth order measured frames
  rely on ("called functions out of depth order"): each now takes its original's place.
- **The per-kernel object merger refused duplicate function names**: each kernel's object now carries its own copy
  of what it calls (local symbols, calls resolved within the object). It accepts duplicate locals and writes them
  with the other local symbols.
- Result, Cycles per kernel with `-agx-experimental-leaf-calls -agx-measured-call-frames`: all 75 kernels, a
  28.9 MB object (1338 local called functions), 1.0 GB peak, 119 s in all (AGX stage 83 s). Fully inlined it ran
  out at 6.4 GB on shade_shadow. Call timing on the GPU is not calibrated; no GPU time measured yet.

- **The AGX object is mostly duplicates**: Cycles' 28.9 MB holds 0.7 MB of kernels and 28.0 MB of called
  functions, 1338 copies of 213 functions (5.3 MB with one copy each). 151 of the 213 are byte-identical in
  every kernel; the others differ by their call offsets and (measured) frame offsets. Sharing one copy needs
  the backend to emit relocations for calls to local functions (they are resolved PC-relative when each
  kernel's object is written), so that the merger can point every kernel's calls at the one copy; and one
  frame layout for the shared functions. Done (ceafb92, `-agx-call-relocations`): 28.9 -> 16.7 MB, 516 copies
  kept; the remaining duplicates differ by their kernels' measured frame offsets, so a frame layout common to
  the kernels would be the next step (one copy each: about 6 MB).
- **A single-kernel module never reaches the merger**, so with `-agx-call-relocations` its calls stayed unresolved
  and the simulator crashed decoding them (17 tests, "index out of bounds" in fsem.rs). Any object with call
  relocations now goes through the merger, on both routes.
- **Small modules forced through the per-kernel path** (`ZLUDA_AGX_PER_KERNEL=1`): tma-red.ptx fails to merge,
  "the kernels' objects have different global data", with or without calls; Cycles (and the default, over 4 MB)
  does not hit it.
- **zoc was not deterministic** (until 649b737's parent): the Metal copy's functions came out in heap-address
  order (LLVMZludaAirGlobalsToBuffer), almost a million lines of difference between two runs of Cycles. Compare
  outputs of two runs (`llvm-dis`, or md5sum now that they match) before attributing a difference to a change.
- **Device-flaky PTX tests**: ldm-0, ldm2-st0 and occupancy pass or fail from one run to the next with the same
  zoc; rerun them with the previous build before calling a change a regression.

## Strict FP on the AIR route, and the zoc server, 2026-10-31 (second session, later)

- **zoc --serve took the direct route** (fixed 455ba04): every module under 4 MiB that the driver compiled went
  through `compile_agx` whatever `--target`. Small-module numbers before that commit (Geekbench, llama.cpp's
  smaller modules, agx_bench_kernel timings) are of the direct route; Cycles, llama.cpp's 10 MB module and the
  PTX-formalization harness were AIR. Found from strict f32 division timing differently through the server and
  the command line. AIR route, corrected: Geekbench GPU score 54807 (direct with batching: 52974 / 53752).
- **Strict mode (ZLUDA_ROUNDING_STRICT=1), AIR route, measured**: f32 add / mul / fma 3.3x the default mode's time
  (the denormal test of each operation's run), div.rn 1.25x, sqrt.rn 1.8x. Geekbench 38260 -> 45168 after the
  fixes below (default 54807); llama.cpp tg 41.7 vs 43.8 t/s, pp 378 vs 430 (completions exact both ways);
  Blender default cube GPU time +6.7%; Open Data monster 261 / classroom 61.7 / junkshop 67.5 vs 466 / 133 / 162.
  PTX-formalization: strict 397 files / 1928 checks, default 377 / 1818.
- What helped strict mode: approximate divisions kept approximate (llvm b602fa44: div.approx / rcp.approx were
  computed exactly, slower than div.rn); exact zero results without soft-f64 (c3f5e644: a zero result went to
  soft-f64, 24-30x); bit-form .ftz flushes dropped where the ALU flushes (96c88d8f, 0cb73d79); canonical NaNs
  only where observed (c15b98b, e8592d6); .ftz comparisons left to the hardware in AGX code (ce75cdb: Cycles is
  .ftz throughout, ~15 instructions per comparison; classroom 42.8 -> 60.1).
- **Dead ends**: an `!fpmath` error margin (ZLUDA_FP_ULP) for division and square root: on the AIR route
  AIREmulateF64's fast path is already the hardware's division, 1.15-1.25x default, nothing left to relax
  (AGXStrictFP, which it targeted, only serves the direct route). The canonical NaN test as a float compare
  (`select(x uno x, NaN, x)`, one fcmpsel instead of and + icmpsel; intersect_closest 28016 -> 23590
  instructions): LLVM folds the float select of a NaN into x (the payload is not kept: rounding-diff-f32 got
  0x7fc00000); as an integer select on the float compare it lowered to 42923 instructions. Reverted.
- What remains of strict mode's cost in Cycles is mostly the canonical NaN of values that are observed (stored,
  passed through integers): ~5000 and + icmpsel pairs in intersect_closest (15076 instructions in the default mode,
  28016 strict). What worked instead of the float compare: fadd32 / fmul32 / fmadd32 return 0x7fc00000 for every
  NaN (playground 2026-10-11-float-semantics), so their canonical NaN is one integer comparison with it
  (llvm 315e9fba): 28016 -> 24283 instructions, monster 261 -> 271, classroom 61.7 -> 63.4, the same image. Other
  producers (rcp, rsqrt, exp2, conversions) keep the two instructions: their NaN output is not established.
  Then established (playground 2026-10-31-special-nan: rcp, rsqrt, exp2, log2 give 0x7fc00000 too; llvm
  ca3980ea). Strict mode with all of it: Geekbench 48702 (default 54807), Open Data junkshop 76.6 (default 162),
  llama.cpp pp128 382.5 t/s (default 430), completions exact.
- The harness has error margins now: `--ulp N` and `--flush-ok` (float-typed checks only), PASS~ in its output.
- **llama.cpp generation, where the time goes (AIR route, graph batching)**: 44 t/s, the main thread waiting on
  the GPU 80% of the time; kernels 14.4 ms per token (mul_mat_vec_q<Q5_1> 6.6 ms, 62 us a launch; the Q8_0 output
  projection 1.7 ms in one launch, ~5x its bandwidth bound), an empty dispatch in a graph 2.7-3.3 us (~490 per
  token), cuGraphLaunch 20.3 ms and the logits' read-back 2.3 ms per token. Without the CDM barrier after each
  batched dispatch generation ran at 61 t/s, with garbage output: the barriers are needed. Leaving out only the
  barriers without a hazard needs each dispatch's read and write ranges: ggml keeps every intermediate tensor in
  one allocation, so a test at allocation granularity finds a hazard everywhere. AGX has no signed bitfield
  extract (bfeil / extr are unsigned), so dp4a's signed bytes stay two instructions each.

## Application benchmarks, AIR route, 2026-10-31 (second session)

- **Blender: launch-time specialization cost 85% of a warm render.** Cycles' scalar arguments change at every
  launch, and each new set made `specialized()` rewrite and hash all 13 MB of its PTX (~40 ms per launch against
  ~2 ms of GPU work). Not specialized above 4 MiB: default cube 512x512 64 spp 5.77 -> 0.92 s, same image.
  Specializing large modules only once arguments repeat (llama.cpp's do) gave no gain there (tg 41.3 -> 41.6 t/s,
  pp 335 -> 314): left off.
- **Cycles' compile (cold first render)**: per-kernel AGX code on every core memory allows (was 4 threads) 72.5 ->
  44.2 s; no second O3 in each kernel's module (it repeated O3 on the functions the kernels share) 44 -> 31 s, the
  first render at 1024x1024 67 -> 50 s, GPU time unchanged. With that O3, llama.cpp's 10 MB module failed to
  compile ("Cannot select: store ... into %stack.0"), another reason it stays off.
- **llama.cpp b11706 needed three fixes to start on the AIR route**: CUDA_MODULE_LOADING lazy by default (a
  native-only start-up image), cvt.*.e2m1x2 with a .b8 register (FP4 quantize kernels), %envreg1:%envreg2 as the
  grid-sync workspace (soft_max_f32_parallelize_cols traps on null). Greedy completions match the CPU's.
  Qwen2.5 0.5B Q5_K_M, -fa 0 -b 8 -ub 8: pp128 335-337 t/s, tg 41-42 t/s. Generation is GPU-bound (the main
  thread waits on the GPU 66% of the time): ~490 kernels per token, 14 ms of GPU time;
  mul_mat_vec_q<Q5_1> is 45% of it and latency-bound (68.5 us for 3.3 MB, 82 registers, 18 SIMD-groups per core).
  Its PTX is unrolled already: unroll, scheduler and memory-clustering options changed nothing (within 2%).
- **Graph and launch batching (ZLUDA_AGX_GRAPH_BATCH=1 ZLUDA_AGX_LAUNCH_BATCH=1, still opt-in)**: llama.cpp pp128
  335 -> 413-453 t/s, tg unchanged; Geekbench GPU score 51865 -> 52974; Blender 1024x1024 128 spp second render
  1.70 -> 1.69 s, the same images and completions.
- **Blender Open Data benchmark (script 3.1.2, scenes monster / junkshop / classroom)**: rendered once the device
  reported CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT (0 made every 2D texture's pitch 0): 464.6 / 156.6 / 133.3
  samples per minute. **Their images are wrong**: instanced objects (linked duplicates, collection instances) are
  missing. Traced one ray through the BVH (agx_run --trace-pcs, AGX_TRACE_LANE): the instance push, transforms,
  slab tests and stack are right; every leaf of an instanced mesh's own BVH holds visibility 0 and primitive type
  0 in its .z / .w (a non-instanced scene's leaves hold 1693 and 1), so the traversal skips all their triangles.
  Cycles' pack_instances copies those words from the mesh's BVH as they are: the host data, built on the CPU,
  already has them. Not the calls (inlined, the same), not the generic-pointer bug below (fixed, the same), not a
  backend option. Open: why this Blender build's mesh BVH2 leaves get no visibility / type (a Cycles issue of the
  CUDA BVH2 path, or something Cycles reads from the driver). Later: a log of the driver's host-to-device copies
  shows Cycles itself uploads the mesh leaves as (2, 4, 0, 0) (the instance leaves written by pack_leaf directly
  have theirs); Cycles' spatial-split, compact-BVH and BVH-type settings change nothing; and the build's commit
  (ae6d847d66fa) does not exist in Blender's repository ("commit doesn't exist"), so the aarch64 build is a
  third party's, with no official Linux ARM64 build to compare (blender.org publishes x64 only).
- **Generic pointers across real calls** (fixed, llvm-project claude/air-calls-generic-pointers): a called
  function given a generic pointer to its caller's private memory accessed it as device memory
  (ptx/tools/agx_tests/private-ptr-call.ptx).

## Application benchmarks, AIR route (driver default), 2026-10-31

- **Geekbench 7 GPU (CUDA)**: every workload completes; summed GPU time 25.26 s (air06) vs 25.08 / 25.34 s on the
  direct route (direct40/41). Per-kernel differences are run-to-run noise: `build_patches_padding` measured
  0.89–2.40 s with identical AGX code on both routes. The Preview cannot show scores (upload host unreachable).
- **llama.cpp (b11706 llama-app, Qwen2.5 0.5B Q5_K_M, `-fa 0 -b 8 -ub 8`, 3 repetitions)**: pp128 347.1 ± 1.0 t/s,
  tg32 43.8 ± 0.8 t/s, vs 93.9 / 39.7 for the staged 2026-10-24 direct-route driver (older driver too, so not a
  pure route comparison). Greedy completions of the four llama_cpp.md prompts match the CPU output exactly.
  Without `-fa 0 -b 8 -ub 8` completions reach cuBLAS (no PTX, error 209) on either route.

## Metal GPU time of f32 math (M2 Max, 256 x 256 threads, 2048 dependent operations each; scratchpad bench/math32.ptx)

Relative to an fma (0.161 ms): lg2.approx 1.0, ex2.approx.ftz 1.0 (was 1.6: a software flush the GPU makes
itself), sin.approx 2.8, rcp.approx 2.3, div.approx 2.9, rsqrt.approx 1.3 (was 3.3: 1 / sqrt instead of AIR's
fast_rsqrt), sqrt.approx 1.7 (was 4.0), sqrt.rn 3.0, div.full / div.rn 5.1. Dead ends: `fdiv arcp afn` for rcp
(no change: the reciprocal is the hardware's already); a faster div.full from Metal's fast divide (2.5 ulp, PTX
allows 2). The AIR service caps a request's buffers at 16 MiB in all (and rejects large request files), which
fits in the M2 Max's 48 MB system cache: memory bandwidth cannot be measured through it.

## Dead ends and surprises

- **The PTX suite passing is not enough: compile Cycles and Geekbench too.** 1a856ba (.ftz compare marking)
  passed the whole suite and made Blender Cycles' compilation assert (a constant compare operand, which no suite
  test has); three more commits went in on top before it was noticed (fixed in 17f4490). Before committing a
  front-end or pass change: `zoc --target air --object-only --ignore-errors` on Cycles' PTX and on
  scratchpad gb7/mod*.ptx, both with and without `--arch agx-g14g`.

- **The AIR device harness measured the AGX-mode AIR**: air_ptx_formalization_check.py always passed
  `--arch agx-g14g`, so every GPU time and suite result before 2026-10-30 evening was of the AIR made with AGX
  code (its Metal copy from the O3 path), not of a Metal-only compilation. `--arch none` measures the Metal-only
  path; the suite gives the same results both ways. A change to one path only (as the division expansion first
  was) shows no effect in the other.
- **Integer division**: Metal's udiv i32 costs about nine mul.wide; AMDGPU's expansion (reciprocal estimate, one
  Newton step, two corrections) is exact on the M2 Max and, with the divisor's part hoisted, 2.8x faster in a
  loop; recomputed every time it is only 13% faster. Integer-op timings in a dependent chain (x = x / y + c)
  measure latency, where the expansion gains nothing: use independent work per iteration.
- **Half arithmetic in a .ftz function costs nothing extra** (h_plain vs h_ftz, the same time).

- **The GPU's denormal behaviour, measured** (M2 Max, 2026-10-30): f32 compares flush denormals, f32 min / max
  return a denormal operand unchanged, 16-bit denormals are kept. So setp.ftz.f32 needs no flush outside strict
  mode and min/max.ftz.f32 always do.
- **Tricks that rely on the GPU's flushing compares fail on constants**: LLVM folds compares as IEEE does.
  Dropping setp.ftz's flush in the front end broke half-oob-set (constants moved through registers, so not
  constants when emitted); a min/max flush as `(r == 0) ? copysign(0, r) : r` broke minmax3. The flushes are
  now removed after the first optimization only (marked !zluda.ftz.compare), and min/max keep the bit flush.
- **Constrained mode costs GPU time**: any mode change in a module (a .rz anywhere) makes every function's f32
  .ftz compares flush in software; the same f32 loop ran 61% slower in such a module. Look for other per-module
  switches that tax unrelated code.

- **The scratch `guard.py` (compile a kernel's pipeline without running it) assumed a named entry block**; zoc's
  current Metal AIR has unnamed ones. Run `opt -passes=instnamer` before it.

- **"Cannot select: load (s64) ... addrspace 2" from AGX instruction selection** after regenerating the texture
  library was not about the loads: the newer clang made switch lookup tables (`@switch.table.*`), globals AIR and
  AGX do not have, and the first error (`global switch.table... has a type AIR does not have`) is easy to miss
  above it. `gen_agx_tex.py` now passes `-fno-jump-tables`. Same class of failure as the first pipeline's
  `simplifycfg` lookup tables: keep `switch-to-lookup` away from anything AIR or AGX compiles.
- **Regenerating `zluda_ptx_impl_agx.ll` with a different clang rewrites every generated section** (the library was
  made with Ubuntu clang 18.1.3; the tree's clang 24 changes cvt, heap and math too, and `llvm.lifetime.*` lost its
  size argument). Only the texture section was regenerated and spliced in; the others keep clang 18's output.
- **Soft-f64 was blamed on Cycles' 113 `.rz` f32 operations; wrong.** Directed-rounding operations are already
  outlined into shared `__air_hint.*` functions (one call site each). All of Cycles' soft-f64 (720 multiplies, 40
  divides) came from the software texture sampler, which filters in double to match the formal model exactly and
  is inlined at every `tex`. Count the callers of `__air_f64_*` before assuming where doubles come from: Cycles'
  PTX has no `.f64` at all.
- **Software subnormal flushing for `.ftz` round-to-nearest f32 operations doubled Cycles' IR** (1.46M vs 0.53M
  instructions; 49 s vs 18 s .ptx → .air). Metal's f32 hardware already flushes outside strict mode; flush in
  software only in strict mode and for directed or unknown roundings.
- **Release `zoc` discards value names**, so anything identified by a value name breaks only in release builds:
  the globals buffer's and the atomics lock table's `air.arg_name` were empty (the harness then reports a flood of
  `UNSUPPORTED: more AIR buffers than PTX parameters`, and the lock table would get an ordinary location index).
  Interface names now go through an `air-arg-name` attribute. Earlier, parsing the soft-f64 library in a
  name-discarding context failed the same way. Test release builds, not only debug ones.
- **Regenerating the soft-f64 library with clang 24 plus a branch-free `fma` / `clz128`** (no branches in
  `__air_f64_fma`, 282 → 258 instructions, host-tested on 2M cases): fma -4.8% GPU time, but mul +7.9%, div +4.6%
  (the new clang's code for the rest). Reverted; the library stays as generated before. `-mlzcnt` did not remove
  `fma`'s branches either (they came from `clz128` branching between two counts, not from the cost model).
- **Pipeline choice does not change Metal GPU time**: O3 vs light vs light+, the old route vs the direct one, GEP vs
  integer address arithmetic all measured equal within noise; Metal's compiler re-optimizes. Choose pipelines for
  compile time and memory. (The GEP-addressing change to AIRLegalizeUnsupportedIR was reverted for this reason.)
- **A light first pipeline crashed Metal's compiler** (`XPC_ERROR_CONNECTION_INTERRUPTED`) on 125 SIMD tests: SIMD
  operations left inside loops and under branches. The first stage needs full unrolling, lower-constant-intrinsics
  and tailcallelim (recursion in calls.ptx). Check new pipelines offline before sending them to the device; a
  crashing or hung request can freeze the Mac running the service.
- **InstCombine's fixpoint verification aborts** in the light pipelines: use `instcombine<no-verify-fixpoint>`.
- **Metal's pipeline cache ignores metadata**: a per-build salt in metadata did not defeat it; an unused volatile
  stack array per kernel does (`ZLUDA_AIR_SALT`).
- **The globals buffer as a `ptr` parameter of non-kernel functions was rejected by Metal**; it is passed as i64.
- **Per-kernel AGX compilation** first copied every function into each kernel's module (reachable functions only
  now) and shared one context (a fresh context per kernel now: the shared one held Cycles at 4 GB).
- **`llvm-extract` of one kernel from portable AIR**: `llc` rejects its bitcode ("Invalid call record") and the
  other kernels' `!air.kernel` entries remain ("an !air.kernel entry names no function"). Round-trip through
  `.ll` and prune `!air.kernel` to the kept kernel.
- **Never run Geekbench's BVH kernels with random (gbgen) inputs**: they hang the GPU.

## Dependency-aware barriers (2026-10-31)

Every batched dispatch ends with the CDM barrier (cache invalidate/flush, playground
`2026-10-16-l2-command-boundary`). Leaving out every barrier gave llama tg 61 t/s but wrong output, so the
question was how many barriers a dependency analysis could drop.

- `ZLUDA_AGX_HAZARD_BARRIERS=1` (driver, off): barrier only before a dispatch whose argument allocations overlap
  those since the last barrier. Correct (llama completions 4/4 exact) but no speed-up (tg 46.8 vs 48.6, pp 441 vs
  447, within spread): ggml keeps all intermediate tensors in one allocation.
- `ZLUDA_AIR_RANGE_REPORT=1` (compiler, report only; `_WHY=1` prints the blocking term): how many device accesses
  scalar evolution writes as pointer argument + a function of launch values. llama.cpp's 1157 kernels: 2755 of
  36961 accesses, no kernel entirely; blockers are grid-stride / data-dependent loops (trip counts not
  computable), selects and phis, and 23903 accesses whose base is not a pointer argument.
- `ZLUDA_AGX_HAZARD_STATS=1` (driver): the ceiling, assuming perfect byte ranges and taking only identical pointer
  values as conflicts (optimistic). llama tg: 24366 of 28672 dispatches still need a barrier (85%): its graph is a
  chain (norm -> mul -> matmul -> rope -> ...). At most ~15% of the ~6 ms/token barrier cost could go, about
  44 -> 47 t/s, for a multi-day compiler + runtime project. Not pursued; the allocation-level flag stays as an
  experiment.

## Strict FP: canonical NaNs through chains of .ftz flushes (2026-10-31)

`ZLUDA_AIR_NAN_WHY=1` (zoc) counts what observes each canonicalized result. In Cycles nearly all were .ftz operand
flushes on the bits, select((b & 0x7f800000) == 0, sign, b), applied again to an already flushed value (the next
.ftz operation) with LLVM sharing the sign bits between the two flushes; the pass only accepted a single flush.
A NaN passes every flush of such a family unchanged (its exponent is not zero), so the whole family is now accepted.

- Cycles intersect_closest, strict: 23765 -> 20135 instructions (default 15069).
- PTX-formalization strict 397 / 1928, default 377 / 1818 (unchanged); new test agx_tests/ftz-nan-chain.ptx passes
  strict (default mode does not canonicalize, by design).
- Blender Open Data strict: monster 271 -> 279, junkshop 76.6 -> 80.0, classroom 63.4 -> 65.5 samples/min.
- Geekbench strict 48702 -> 49641. llama strict completions 4/4 exact.

## Geekbench "errors" in default mode (2026-10-31)

The 10 errors of a default Geekbench run: 6 are CUDA 701 (launch out of resources) at 1024-thread blocks of an
image filter, from Geekbench's block-size search (the kernel's registers allow fewer threads per core; correct,
the search moves on). The rest were launch-time specializations of the ray tracer (trace_bvh, hit_surface,
generate_rays, ...) that failed to load: "the modules lay out their global data differently". Their only global
data is LLVM's switch lookup tables (switch.table.*), which a recompilation names and sizes differently. A
module whose globals are only such compiler-made constants now keeps its own copy. All specializations load (0
failures); Geekbench GPU time 26.85 -> 26.09 s, score 53249 (within the 53441-54812 spread of earlier runs).
- Then: the flush family also passes through integer selects and phis of the flushed bits, and unsigned
  comparisons below the NaNs (x < C, C <= 0x7f800001: false for every NaN) do not observe them. intersect_closest
  strict 20135 -> 19921. The remaining observers are real bit uses (sign tests, xor of the sign, or with sign bits):
  NaN canonicalization is no longer the main strict cost; the rest is denormal emulation (air.ieee-fp32) and
  comparisons on the bits. Test agx_tests/ftz-nan-phi.ptx.

## Strict FP: .ftz min / max to the hardware (2026-10-31)

LLVMZludaAirCompareFloatBits rewrote every f32 min / max on the bits (about 20 integer instructions), ignoring
its `functions` argument, which the comparisons respect: AGX's f32 compare (fcmpsel ltn / gtn) already orders a
denormal as zero, PTX's .ftz order, and the backend's lowering of minimumnum / maximumnum handles signed zeros and
NaN operands. Now .ftz f32 min / max go to the hardware in AGX code (still on the bits for IEEE functions and for
Metal). Cycles is .ftz throughout (3284 max.ftz, 1644 min.ftz).

- intersect_closest strict 19921 -> 16732 instructions (default 15069).
- Blender Open Data strict: monster 279 -> 345, junkshop 79.4 -> 97.7, classroom 65.5 -> 81.9 samples/min.
- Geekbench strict 49641 -> 53084. llama strict 4/4 exact. Suites unchanged (strict 1928, default 1818).
- New test agx_tests/ftz-minmax.ptx (denormals, NaNs, signed zeros): passes on the simulator and on the GPU
  (agx_run_hw), as do ftz-nan-chain and ftz-nan-phi.

## Strict FP: calls in the shading kernels (2026-10-31, open)

Blender junkshop per-launch GPU time, strict vs default: intersect kernels now equal; shade_surface x2.3,
shade_light_nee x3.5. Strict code keeps many more real calls (shade_light_nee: 1629 calls, 12332 stack loads vs
156 / 3783 in default): the module-level O3 inlines less. ZLUDA_AGX_INLINE_KERNELS (now also "all"):
- shade_light_nee and shade_surface inlined, strict junkshop 97.3 -> 122.1 samples/min (shade_surface 43.7 -> 27.0
  ms, shade_light_nee 23.8 -> 12.1 ms), warm-up with compile 107 -> 144 s; default 159.7 -> 161.7.
- "all": a single Cycles compile ran 21+ minutes at 26 GB and was stopped. Undoes the compile-cost work; not a
  default. Next: why strict's O3 inlines less (constrained-FP intrinsics or strictfp attributes in the inline
  cost), to get default's inlining without forcing it.

## Strict FP: the texture sampler (2026-10-31)

The calls in strict shading kernels (1629 in shade_light_nee vs 156) are not Cycles functions left uninlined: they
are the soft-f64 routines of the exact software texture sampler (__air_f64_mul, _from_u64, _floor, ...), which
default mode runs as GPU floats. Skipping the strict-mode bit flushes of .ftz operations in functions that flush
f32 denormals (the IEEE f32 emulation leaves those to the hardware) changed little (1629 -> 1615 calls; kept, suites
unchanged).

zoc --strict-fp-f32-textures (driver: ZLUDA_ROUNDING_STRICT_TEXTURES=f32, in the cache key) keeps strict arithmetic
but filters tex.1d / tex.2d in f32 as default mode does. Blender junkshop strict 97.3 -> 144 samples/min (default
160; shade_surface 43.7 -> 19.6 ms, shade_light_nee 23.8 -> 7.1 ms), compile warm-up 107 -> 92 s. Against the exact
sampler (generated linear-filter tests, 256 coordinates each, simulator): f32 textures with one channel exact,
two channels with border addressing within 2 ulp. Strict formalization suite unchanged with it (1928). Off by
default: an error margin for textures only.

## llama generation: device copies inside the graph's batch (2026-10-31)

ZLUDA_AGX_GRAPH_TIMING=1 (driver) prints each graph's shape and its host time. llama.cpp's token graph is 506
kernel launches with 24 device-to-device copies between the layers (L13 D1 L20 D1 ... L13). Each copy waited for
the GPU (sync_all) and copied on the CPU: 25 commands per token and 13 ms of host waiting, nearly all of the token's
time. Device copies in a graph are now a dispatch of built-in copy kernels (16-, 4- or 1-byte units, compiled once
through the AIR route) inside the batch, ordered by its barriers; module globals and copies outside graphs as before.

- llama tg128 47.3 -> 124.9 t/s (same session, -b 8 -ub 8); pp128 436 -> 441. Strict: tg128 130.8, pp128 401.6.
- Completions 4/4 exact in both modes; driver tests pass.
- (llama-bench without -b 8 -ub 8 aborts in cublasCreate: this llama-app links cuBLAS statically, whose fatbin has
  no PTX; not a driver regression.)

## A larger model: Qwen2.5-7B-Instruct Q4_K_M (2026-10-31)

Apache-2.0 GGUF (bartowski/Qwen2.5-7B-Instruct-GGUF, 4.36 GiB) in /scratch/models/qwen2.5-7b; the bundled
llama-app (cuBLAS linked statically, so -b 8 -ub 8; a cudart replacement is out of ZLUDA's scope).

- pp128 64.0 t/s, tg64 27.1 t/s (AIR route, default mode, after the in-batch device copies).
- tg reads all 4.36 GiB of weights per token: ~118 GiB/s, about 30% of the M2 Max's rated bandwidth.
- ZLUDA_AGX_PROFILE is no guide here: every command shows a 50-60 us floor whatever its size (a 4-block copy as
  long as an 18944-row matrix-vector product).
- The Q4_K matrix-vector kernel (mul_mat_vec_q<12>), captured (512 rows x 3584, 1 MB) and replayed on the GPU: 20.9
  us back to back (58 us cold), ~49 GB/s. Its capture's regs.txt says 124, but in 16-bit halves: 62-63 general
  purpose registers, 832 threads per core, so occupancy is not what limits it. .maxnreg 56 or less makes it spill,
  and the replay (no stack memory) fails with "Bad address"; recompiled without a limit it runs as fast (21.3 us).
  This launch is small (512 rows); the 18944-row FFN projections are the ones to measure next.
- The 18944 x 3584 Q4_K launch (38.2 MB of weights), captured and replayed on the GPU: 474 us back to back (cold
  1.41 ms), ~80 GB/s, about a fifth of the rated bandwidth; two per layer, 28 layers: most of the 27 t/s.
- dp4a is the cost: compiled with dp4a replaced by one multiply-add (wrong results, an upper bound only), the same
  launch takes 277 us instead of 446 (-38%; 574 -> 491 instructions, 63 -> 49 registers). Each exact dp4a is four
  byte multiply-adds with every signed byte extended by a shift pair (AGX has no signed bitfield extract); bias or
  sign-correction rewrites cost more instructions, not fewer. A real gain needs packed 8- or 16-bit multiply-adds,
  which nothing measured so far establishes on AGX: a hardware probe first.
- Packed integer operations for dp4a (asked for a probe): what is measured or documented already says there are
  none. The playground's sub-op sweeps (2026-10-11-subop-spaces: intl, bitrev, popcount, simd_op reductions,
  shuffles, convert) found no dot product or packed integer arithmetic; applegpu's documentation lists none, and its
  device_load i8 format zero-extends only (snorm8 loads are lossy floats: -128 and -127 both give -1.0). The only
  packed arithmetic is simd_matrix_fmadd16/32 (float, cooperative across the SIMD group), wasted on a matrix-vector
  product. What 32-bit integer ops do have is free sign extension of 16-bit register halves (MOD_SX), which saves
  about one instruction per signed operand of a dp4a (7 -> 6), not the 38% upper bound. Not pursued further; the
  unexplored part (imadd's unused encoding bits) is speculative and earlier undefined encodings hung the GPU.

## Default mode: the float sampler for every software texture sample (2026-11-04)

The default mode filtered only plain tex.1d / tex.2d in float (or on the texture unit); every other tex form
(mipmaps, tex.level / .grad, 3D, cubemaps, arrays, offsets, depth compare) ran the exact sampler in software
double precision: ~7600 AGX instructions a mipmapped sample (PTX-formalization tex-mip, one thread, 32 samples:
243,945 instructions, 556,017 cycles in the AGX Performance Simulator). zluda_ptx_impl_agx_tex.c is now also
compiled with float for double (gen_agx_tex.py: -DTEX_F32 -cl-single-precision-constant, its types and internal
functions prefixed apart), and the default mode's tex instructions call that build (mode 2 of the last argument;
strict mode keeps the exact sampler): tex-mip 39,561 instructions, 206,839 cycles, compiled in 7.3 s (193 MB).
The texture tests still match the model exactly in default mode; strict mode, the default-mode results of the
other tests and the strict compile of the 413 test modules (320 s, 541 MB) are unchanged. The generated section
is now clang 18's output throughout.

## Strict FP: exact texture filtering in integers (2026-10-31)

The exact double precision sampler, the remaining strict-mode cost in Cycles, now has an integer path for unsigned
normalized textures (u8 / u16 read as floats, level 0, tex.1d / tex.2d), bit-identical to the double sampler (see
the commit and zluda_ptx_impl_agx_tex.c's exact_unorm_sample; 70 generated reference cases, suites unchanged).
Blender Open Data strict: junkshop 97 -> 153 (default 160), classroom 82 -> 128 (133), monster 345 -> 429 (466)
samples/min. The opt-in ZLUDA_ROUNDING_STRICT_TEXTURES=f32 is no longer needed for speed on these scenes.
Regenerating the texture block needs a clang: /scratch/clang-build (clang and NVPTX built from the tree); splice
only that block (the other blocks keep clang 18's output) and drop repeated declarations.

## Blender Cycles drops instanced geometry on the GPU, 2026-11-03 (solved: a miscompiled host, see above)

- **Cycles renders run but are wrong.** 16-sample renders against the CPU in Blender 5.1.0: classroom loses its desks,
  chairs and ceiling lamps (mean difference 19.3 of 255), monster loses the monster and the child's face (13.1).
  Unique geometry is right. Earlier checks compared GPU images with each other, so they missed it: the October 31
  strict classroom image has the same holes. **Open Data scores recorded before this fix (monster, classroom,
  junkshop) are of scenes with geometry missing; do not use them as render performance.**
- **Minimal case:** a scene of a plane and three unique spheres matches the CPU exactly (difference 0). With the spheres
  sharing one mesh they all disappear. Scripts, which take `-- DEVICE MODE OUT` with MODE `uniq`, `inst` or `scale`:
  `docker-scratch/blender-src/instancing-repro/scene.py`, and `audit.py` to count a scene's instances.
- **Ruled out:** the compiler version (October 30 `zoc` gives the same image), the driver version (October 31, November 1
  and HEAD builds), and with `ZLUDA_AGX_SPECIALIZE=0`, `ZLUDA_AGX_UNIFORM_CONSTANTS=0`, `ZLUDA_AGX_LAUNCH_BATCH=0`,
  `ZLUDA_AGX_GRAPH_BATCH=0`, `ZLUDA_AGX_SOFTWARE_TEXTURES=1`, `ZLUDA_AGX_HAZARD_BARRIERS=1` and
  `ZLUDA_AGX_USC_BARRIER=1` the image does not change at all.
- **Next:** the path entered only for instances in Cycles' BVH2 traversal (`kernel/bvh/traversal.h`: the negative
  `prim_addr` leaf, `object_node`, `bvh_instance_push`, its pointers to the ray's private copies). Debug with kernels
  built from Cycles' source (`Builds/blender-cycles-5.1.1`), not by inspecting PTX bundled in a prebuilt application.
- **Compile time:** `zludaModeField` solved a function's web of modes again for each operation, over 18 minutes on Cycles;
  solved once per function, 47 s. Against the October 30 `zoc` (37.8 s, 1.26 GB, 16.0 MB object) it still takes 7.0 GB and
  writes 44.3 MB. `85ef7ed`'s entry mode is not the cause; the cause is not known.
- **Geekbench:** with `libtensorrt_rtx.so` absent all eleven workloads run on the CUDA fallback, 26.29 s of GPU time. With
  it present Background Blur, Face Tracking and Super Resolution still fail to deserialize their engines. Reported scores:
  68805 through ZLUDA, 41822 on the same chip through Mesa's OpenCL. No workload's output has been verified.

## Occupancy (register cap), 2026-11-03

How kernels came to declare 128: the greedy allocator pays nothing for r127 over r40, the backend learned the count only
afterwards, and a kernel with calls declared everything since a callee may use anything. The per-function limit existed, but
only PTX that asked for it (.maxnreg) got one.

Samples/min at -agx-default-max-gprs (callee-saved 32, 30 s of Blender Open Data; 0 GPU faults):

| Cap | classroom | junkshop | monster |
| --- | --- | --- | --- |
| 128 | 204 | 247 | 290 |
| 96 | 217 | 276 | 325 |
| 80 | 233 | 283 | 349 |
| 72 | 234 | 265 | 355 |
| 64 | 234 | 264 | 354 |
| 56 | 226 | 240 | 333 |
| 48 | 201 | 227 | 293 |

- Callee-saved 16 against 32: within noise at every cap.
- A kernel must declare the cap (not 128) for this to work: an early try of the cap did nothing because kernels with calls
  still declared 128 whatever the cap.
- The callee-saved set must sit under the cap, built at run time; with the old fixed top block it was above it.
- Frames grow as registers shrink: the 12 KB stack first try fails at cap 80 (sample_f32_outlined), and the driver
  then compiles twice (warm-up 26 s to 52 s). 16 KB fits.
- Block size (cuOccupancyMaxPotentialBlockSize capped at 32-192 threads): no gain at 128 registers; 32 was slightly worse.

### Options at cap 80 (2026-11-03, sweep8)

Samples/min, classroom / junkshop / monster, cap 80 and 16 KB stack unless stated, 0 GPU faults. **Power caveat:** an 18 W USB-C
charger against about 60 W of load, battery draining from 46% to 21%: compare within this table, do not compare it with other
tables or take it as the final number. Base ran first, and its monster (332) may be low: a monster gain under about 4% may be noise.

| Option | classroom | junkshop | monster | Verdict |
| --- | --- | --- | --- | --- |
| base: -agx-callee-saved=32 | 230 | 269 | 332 | |
| -agx-callee-saved=48 (the default) | 221 | 254 | 323 | worse: 48 leaves almost nothing free under a cap of 80 |
| -agx-callee-saved=16 | 236 | 278 | 335 | best on classroom, junkshop; values under 16 not tried |
| ZLUDA_AGX_STACK_TRY=0 (full stack) | 230 | 270 | 331 | no gain over 16 KB |
| ZLUDA_AGX_MAX_BLOCK=64 | 229 | 273 | 337 | |
| ZLUDA_AGX_MAX_BLOCK=128 | 232 | 274 | 344 | |
| ZLUDA_AGX_MAX_BLOCK=192 | 233 | 275 | 345 | best block size |
| ZLUDA_AGX_MAX_BLOCK=320 | 233 | 276 | 339 | |
| -agx-sched-target-gprs=48 | 237 | 271 | 352 | a real gain; much less than hoped |
| -agx-enable-sink-remat | 231 | 272 | 333 | **refuted**, no gain |
| -agx-callee-saved-stride=8 | 230 | 274 | 346 | unclear: may be noise |
| -agx-uniform-constants | 230 | 270 | 328 | **refuted** at cap 80, no gain |

The combination has not been run. -agx-declare-tree-gprs has had no render test: every kernel's tree still reaches the cap, so it
cannot change anything until the worst functions come down. check_tree_gprs.py (/scratch/air-pipeline) checks it offline first.

### What went wrong in these sweeps (for the next agent)

- **sweep6 forgot -agx-callee-saved**, so ran at the default of 48, and made every option look wrong against the earlier sweep.
  Pin every option you are not testing.
- **The laptop lost AC at about 17:50 UTC with nobody noticing**, then a weak charger replaced it. sweep6 and sweep7 are invalid
  (the last 9 runs of sweep7 fell to 75-170 samples/min), and two conclusions drawn from them were withdrawn. run.sh now refuses on
  battery, but **"AC online" does not mean enough power**: macsmc-ac/online was 1 and status "Charging" while the battery supplied
  41 W. Read power2_input (AC input) during a run; it was about 56 W with the good charger, 18 W with the weak one.
  Total system power and AC input: /sys/class/hwmon/hwmon1/power{1,2}_input.
- **A wrong "refuted" outlived its test**: see the register cap note above. A flag recorded as refuted at 128 registers has not been
  refuted at 80. Occupancy changes what matters.

### Reloading spills instead of storing them: small (2026-11-03, static counts, cap 80, callee-saved 48)

Over all 75 kernel modules (1,323 function instances): 34,730 spill stores, of which **24,739 (71%) are callee-save traffic** (exactly 1:1 save/restore, inflated by callee-saved 48), leaving 9,991 real spills.

| Candidate | Stores | % of real | Needs |
| --- | --- | --- | --- |
| kernel_params load through a uniform base | 170 | 1.7% | isReMaterializable on device loads |
| + callee kernel_params load, base still live | 57 | 2.3% cumulative | accept invariant loads without dereferenceable (AGXGlobalDataArgs loses it) |
| + scene array load, operands live | 125 | 3.5% cumulative | invariant info the IR does not have; no clean general proof found |
| + callees read global data from a uniform | 652 | 10.0% cumulative | calling-convention change; whether uniforms stay readable across call is unverified |
| ALU with operands live at every reload | 897 | 9.0% | flag cheap ALU ops rematerializable; removes the memory access entirely |
| mov_imm, single definition | 4 | 0.04% | nothing: 415 more are phi inputs that cannot be rematerialized |

Load remat saves only the store, since each reload becomes a device load: about 3 of 49 cycles per pair on the schedule model (a model, not a measurement). No remat change has been built or tested.
Order to try: lower callee-saved first, then ALU remat, then the kernel_params tiers. Scripts and data: /scratch/NOTES/spill-analysis (spillan.py). An early version dropped block labels with trailing comments and made mov_imm look like 621 removable stores: check a classifier against MIR.

