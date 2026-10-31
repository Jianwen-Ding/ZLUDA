#!/usr/bin/env python3
# agx_texture_compare.py HW SW TYPE: the texture unit's results against the software sampler's (agx_texture_check outputs)
import sys, struct, math
hw = open(sys.argv[1], 'rb').read(); sw = open(sys.argv[2], 'rb').read(); ty = sys.argv[3]
n = len(hw) // 4
if ty != 'f32':
    bad = sum(1 for i in range(0, len(hw), 4) if hw[i:i+4] != sw[i:i+4])
    print(f'{"EXACT" if bad == 0 else "DIFF"} int: {bad}/{n} values differ')
    sys.exit()
h = struct.unpack(f'<{n}f', hw); s = struct.unpack(f'<{n}f', sw)
hb = struct.unpack(f'<{n}i', hw); sb = struct.unpack(f'<{n}i', sw)
def key(b): return b if b >= 0 else -(b & 0x7fffffff)
bad = maxabs = maxulp = nan = 0
for i in range(n):
    if hb[i] == sb[i]: continue
    if math.isnan(h[i]) or math.isnan(s[i]):
        nan += 1; bad += 1; continue
    bad += 1
    maxabs = max(maxabs, abs(h[i] - s[i]))
    maxulp = max(maxulp, abs(key(hb[i]) - key(sb[i])))
rel = max((abs(h[i]-s[i]) / max(abs(s[i]), 1e-30) for i in range(n) if hb[i] != sb[i] and not math.isnan(h[i]) and not math.isnan(s[i])), default=0)
print(f'{"EXACT" if bad == 0 else "DIFF"} f32: {bad}/{n} differ, max abs {maxabs:.3g}, max rel {rel:.3g}, max ulp {maxulp}, nan mismatches {nan}')
