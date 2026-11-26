#!/bin/sh
# A large generated test suite for the AGX pipeline: PTX-formalization's data-driven generators
# (scripts/gen_*.py; expected values from Python transcriptions of the PTX ISA, big integers and an
# exact-rational IEEE reference - no GPU, no vendor tool) with new seeds and more cases per launch than
# its tests/stress, written to OUT. Run the result with ../ptxconf/ptxconf.py.
#
#   tests/stress/gen.sh OUT [N] [SEEDS...]     (default N 512, seeds 1001 .. 1010; the matrix
#                                               generators with the first two seeds)
#   python3 tests/ptxconf/ptxconf.py --backend agx -j 2 --timeout 900 OUT
set -e
OUT=${1:?usage: gen.sh OUT [N] [SEEDS...]}; shift
N=${1:-512}; [ $# -gt 0 ] && shift
SEEDS=${*:-1001 1002 1003 1004 1005 1006 1007 1008 1009 1010}
HERE=$(cd "$(dirname "$0")" && pwd)
PF=${PTX_FORMALIZATION:-$(cd "$(dirname "$0")/../../../.." && pwd)/PTX-formalization}
S=$PF/scripts
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
l() { # name, generator of gen/, args...
    name=$1; shift
    python3 "$HERE/gen/$@" > "$OUT/$name.ptx" 2>/dev/null || { echo "generator failed: $*" >&2; rm -f "$OUT/$name.ptx"; }
}
g() { # name, generator, args...
    name=$1; shift
    python3 "$S/$@" > "$OUT/$name.ptx" 2>/dev/null || { echo "generator failed: $*" >&2; rm -f "$OUT/$name.ptx"; }
}
for s in $SEEDS; do
    g bits-s$s gen_bits.py "$N" "$s"
    g carry-s$s gen_carry.py "$N" "$s"
    g intx-s$s gen_intx.py "$N" "$s"
    g fmisc-s$s gen_fmisc.py "$N" "$s"
    g lop3-s$s gen_lop3.py "$N" "$s"
    g video-s$s gen_video.py "$N" "$s"
    g ieee-diff-s$s gen_ieee_diff.py "$N" "$s"
    for w in 16 32 64; do g rounding-diff-f$w-s$s gen_round_diff.py "$N" "$s" "$w"; done
    for p in 0 1 2 3 4 5 6 7 8 9 10 11; do g cvt-$p-s$s gen_cvt.py "$N" "$s" "$p" 12; done
    for p in 0 1 2 3; do g half-$p-s$s gen_half.py "$N" "$s" "$p" 4; done
    for p in 0 1 2 3 4 5; do g pred-$p-s$s gen_pred.py "$N" "$s" "$p" 6; done
    for p in 0 1 2 3; do g mini-$p-s$s gen_mini.py "$N" "$s" "$p" 4; done
    for p in 0 1 2 3; do g warp-$p-s$s gen_warp.py "$s" "$p" 4; done
    # this directory's generators: sin / cos .approx.f32 near the limits of the helpers' binary32 fast path
    for fn in sin cos; do l sincos-$fn-s$s gen_sincos.py "$N" "$s" "$fn"; done
    # ex2 / lg2 / rsqrt / sqrt / rcp .approx.f32 against the model's correctly rounded definitions (.ftz on odd seeds)
    for op in ex2 lg2 rsqrt sqrt rcp; do l transc-$op-s$s gen_transc.py "$N" "$s" "$op" $([ $((s % 2)) = 1 ] && echo ftz); done
done
# matrix generators: they write tests/<name>.ptx in their working directory
set -- $SEEDS
for ms in $1 ${2:-}; do
    TMP=$(mktemp -d)
    mkdir "$TMP/tests"
    for p in 0 1 2 3 4 5 6 7; do python3 "$S/gen_mma.py" "$ms" "$p" 8 > "$TMP/tests/mma-$p.ptx" || true; done
    for gen in gen_mmasp gen_wgmma gen_wgmma_sp gen_tcgen05 gen_mma_bs; do
        (cd "$TMP" && python3 "$S/$gen.py" "$ms" > /dev/null 2>&1) || echo "generator failed: $gen" >&2
    done
    for f in "$TMP"/tests/*.ptx; do [ -s "$f" ] && mv "$f" "$OUT/$(basename "$f" .ptx)-s$ms.ptx"; done
    rm -rf "$TMP"
done
ls "$OUT" | wc -l
