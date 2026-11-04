#!/bin/sh
# Writes the generated emulation tests into DIR (several MB): the software f64 arithmetic and conversions, directed
# float fma, add, sub and mul, 64-bit division, every f16 to fp8 and every fp8 pair to f16. Expected values come from Python (IEEE doubles, exact rationals, integers), not from
# any GPU. Check them with agx_ptx_formalization_check.py --target air (and --strict-fp).
#   gen_all.sh DIR
set -e
H=$(dirname "$0")/gen; mkdir -p "$1"
for s in 1 2 3; do
  python3 "$H/f64_addmul.py" $s "$1"; python3 "$H/f64_fma.py" $s "$1"; python3 "$H/f64_divsqrt.py" $s "$1"
  python3 "$H/f32_fma_directed.py" $s "$1"; python3 "$H/f32_addmul_directed.py" $s "$1"; python3 "$H/f64_conversions.py" $s "$1"; python3 "$H/f64_integral.py" $s "$1"
done
for g in f64_addmul f64_fma f64_divsqrt; do ZEROS=1 python3 "$H/$g.py" 7 "$1"; done
python3 "$H/div64.py" "$1"
python3 "$H/mini_encode.py" "$1/mini_encode_f16.ptx" f16; python3 "$H/mini_encode.py" "$1/mini_decode.ptx" decode
