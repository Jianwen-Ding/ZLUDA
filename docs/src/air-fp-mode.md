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
bar_red_and_pred UNSUPPORTED. Thus this is progress, not full PTX compliance. Directed rounding across
function calls still needs work; the generated AIR retains a nearest-rounded add_rp function in that case.

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
