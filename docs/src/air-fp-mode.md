# AIR floating-point mode preservation

The AIR static-rounding lowering now initializes denormal-mode bits from the function's entry attributes.
Those attributes describe entry/default behavior, not a rule that overrides every subsequent PTX mode change.
In strict mode, if a float operation can restore IEEE denormal behavior inside an otherwise FTZ function,
the function's f32 environment is changed to IEEE and explicit per-operation flushes preserve FTZ instructions.
All-FTZ strict functions retain their existing hardware path. No AGX instruction, target hint or PTX parser
change is involved; the fix is in ZLUDA's portable AIR preparation.

This fixes a reproduced mixed-mode failure: an FTZ multiply caused a later non-FTZ correctly rounded division
to flush a subnormal result to zero. It also makes the existing cvt-0 conversion suite pass. Default-mode
behavior is intentionally still the documented hardware-oriented approximation, not strict IEEE compliance.

## Evidence

Tests use authored/open-source PTX-formalization fixtures, our compiler and AGX-Performance-Simulator.
No vendor-generated binary or engine was inspected. The target was agx-g14c (M2 Max).

| Check | Baseline | Candidate |
| --- | --- | --- |
| 184 imported ZLUDA formalization files, strict | 179 PASS | 180 PASS |
| Same 184 files, default mode | 178 PASS | 178 PASS |
| cvt-0, strict | FAIL (O22[15]) | PASS, all 27 checks |
| div_noftz, strict | FAIL (expected 0x26, got zero) | PASS |
| New interleaved-mode regression, strict | FAIL | PASS |
| call_rnd, strict | FAIL | FAIL |

The strict imported-suite status change is only div_noftz FAIL -> PASS. Default-mode statuses are unchanged.
Remaining strict outcomes are call_rnd FAIL, const_ident COMPILE-ERROR, ldmatrix SIM-ERROR, and
bar_red_and_pred UNSUPPORTED. Thus this is progress, not full PTX compliance. At that checkpoint directed rounding across function calls still failed; the following change addresses
the reproduced nonrecursive direct-call case.

Raw comparison results remain in scratch as ptx-pipeline-zluda-{baseline,after,default-baseline,default-after}.json
and ptx-pipeline-focus-{baseline,after}.json. Baseline compiler: /scratch/air-pipeline/lib-fam/zoc; candidate:
/scratch/zluda-air-target/release/zoc. No representative application throughput/compile-time claim is made.
Strict mixed-mode functions may need more software arithmetic to preserve the required results.

## Reproduce

Build with the existing offline dependencies and LLVM AGX build:

```sh
cargo build --release -p compiler --no-default-features --features agx --offline
python3 ptx/tools/agx_ptx_formalization_check.py \
  --zoc /path/to/new/zoc --agx-run /path/to/agx_run \
  --target air --arch agx-g14c --strict-fp -j 4 \
  ptx/tools/tests/air_mixed_denorm.ptx /path/to/PTX-formalization/tests/cvt-0.ptx \
  /path/to/PTX-formalization/tests/zluda/*.ptx
```

Set CARGO_HOME, RUSTUP_HOME, CARGO_TARGET_DIR and ZLUDA_LLVM_BUILD_DIR for the local toolchain as needed;
put LLVM readelf/objcopy tools on PATH. The regression mixes non-FTZ and FTZ division with an FTZ multiply,
checking preserved subnormal output and explicit flushing independently. TensorRT remains inactive.


## Directed rounding across direct calls

Static rounding used to lower a mode-dependent callee independently from its caller. A caller selecting
round-up could therefore enter a helper that was converted as nearest-even. The AIR preparation now identifies
dynamic-rounding functions and their transitive callers, and inlines eligible helpers before resolving static
rounding metadata. It only forces this when the caller explicitly changes FP mode. Recursion is detected and
excluded, and noinline functions are respected. Inlining preserves the caller's reaching mode even through
an intermediate wrapper without arithmetic. Default nearest-only callers are not forced to inline. Modules with no explicit mode-setting calls skip
the added call-graph analysis entirely.

This is a correctness change with possible code-size/compile-time cost in mode-changing functions. No
application throughput improvement is claimed. Recursive, indirect or noinline calls that require inherited
mode still need a general interprocedural solution; this is not a claim of complete dynamic FP-environment
support. No new GPU instruction or target hint is introduced.

Validation on agx-g14c using authored/open-source formalization fixtures and AGX-Performance-Simulator:

| Suite | Previous checkpoint | Candidate |
| --- | --- | --- |
| 184 imported ZLUDA tests, strict | 180 PASS, 1 numerical FAIL | 181 PASS, no numerical FAIL |
| Same tests, default | 178 PASS | 179 PASS |
| call_rnd, strict/default | FAIL | PASS |
| New nested-call rounding regression, strict | FAIL with original baseline compiler | PASS |
| Mixed denormal regression and cvt-0, strict | PASS | PASS |

Only call_rnd changes status in each 184-test suite. Strict remaining outcomes are const_ident COMPILE-ERROR,
ldmatrix SIM-ERROR and bar_red_and_pred UNSUPPORTED. Those are not counted as passes. Default mode still has
two numerical failures, as expected for its non-strict floating-point scope. Raw results are in
/scratch/ptx-call-rounding-{strict,default}.json. The nested fixture retains upstream MIT/Apache provenance
and adds an authored arithmetic-free wrapper; it is not a vendor-generated artifact.


### Broader core-suite comparison

The final candidate was compared with the original saved compiler across 413 top-level PTX-formalization
fixtures plus the two new local regressions (415 total), strict mode, agx-g14c, four compiler/simulator workers
and a 30-second per-operation timeout. This comparison includes both denormal and call-rounding fixes:

* Original saved compiler: 396 PASS, 16 FAIL, 2 COMPILE-ERROR, 1 PARTIAL.
* Final candidate: 407 PASS, 5 FAIL, 2 COMPILE-ERROR, 1 PARTIAL.
* Eleven FAIL -> PASS transitions and no status regressions. Besides the local regressions, the improved
  files are cvt-0/1/4/5/8/9 and half-0/2/3.

Remaining numerical failures: fmisc-0/1 (NaN payload canonicalization), shared-per-kernel (shared-memory
addressing), sub-borrow (borrow results), and transc-edge (one-ULP difference). addr32-generic still fails
translation; tex-mip hits the compiler timeout; perctamemory is partial after a rejected kernel. None is
counted as a pass. Raw results: /scratch/ptx-call-rounding-core-{baseline,after}.json.
This establishes neither full PTX coverage nor representative application performance.


## Strict NaN canonicalization of AIR math calls

Strict-mode canonicalization now recognizes the five ordinary AIR math declarations used by PTX exp2,
log2, sine, cosine and reciprocal square root. They had been omitted because they are not LLVM intrinsic
functions after remapping, allowing the backend's quiet-NaN payload to escape instead of PTX's canonical
0x7fffffff. Arbitrary loads, stores and calls retain their existing bit behavior.

The new authored air_math_nan regression covers quiet/signaling/negative NaNs, infinities, zero and raw
load/store payload preservation. It fails with the original saved compiler and passes with the candidate.
Both fmisc-0 and fmisc-1 now pass all eleven checks each. The imported strict suite remains 181 PASS out of
184 files, with unchanged nonpassing statuses. Across 413 core fixtures plus three local regressions, the
candidate has 410 PASS, 3 FAIL, 2 COMPILE-ERROR and 1 PARTIAL. Existing core status changes from the preceding
checkpoint are only fmisc-0/1 FAIL -> PASS. Raw results: /scratch/ptx-math-nan-{focus,imported,core}.json.

For the authored math fixture, default-mode AGX object output is byte-identical between the original saved
compiler and candidate (SHA-256 08dac2ebca8cce2d6a0babf9f27942ff0ff3ec89cc709b5fcbeaebbb0ab6ddce).
That is a code-identity check for one fixture, not a performance benchmark. Requested performance testing was
not started before the user's request to wrap up. No throughput or compile-time improvement is claimed.
