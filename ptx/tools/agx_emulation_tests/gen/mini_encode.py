# usage: mini_encode.py OUT.ptx [f16|decode]  (cvt.rn.satfinite of f32 pairs to e4m3x2, e5m2x2, e2m3x2, e3m2x2, e2m1x2 and
# relu.e4m3x2 at the rounding ties, the subnormal/normal boundary, saturation, infinities, NaNs and f32 subnormals;
# expected from exact rationals rounded to nearest even; with f16: every f16 to e4m3 and e5m2, from f16x2;
# with decode: cvt.rn.f16x2 of every pair of codes of each format)
import random, struct, sys
from fractions import Fraction as F
random.seed(8)
FMT = {'e4m3': (4, 3, 0x7e, 0x7f, 8), 'e5m2': (5, 2, 0x7b, 0x7f, 8), 'e2m3': (2, 3, 31, 31, 6),
       'e3m2': (3, 2, 31, 31, 6), 'e2m1': (2, 1, 7, 7, 4)}
def f32(b): return struct.unpack('<f', struct.pack('<I', b))[0]
def decode(c, mb, bias):
    e, m = c >> mb, c & ((1 << mb) - 1)
    return F(m, 1 << mb) * F(2) ** (1 - bias) if e == 0 else (1 + F(m, 1 << mb)) * F(2) ** (e - bias)
def encode(b, fmt, relu=False):
    eb, mb, top, nan, el = FMT[fmt]
    bias = (1 << (eb - 1)) - 1
    mag = b & 0x7fffffff
    if mag > 0x7f800000: return nan
    if relu and b >> 31: return 0
    sign = (b >> 31) << (el - 1)
    if mag == 0x7f800000: return top | sign
    x = F(f32(mag))
    best = min(range(top + 1), key=lambda c: (abs(decode(c, mb, bias) - x), c % 2))
    return best | sign
def edges(fmt):
    eb, mb, top, nan, el = FMT[fmt]
    bias = (1 << (eb - 1)) - 1
    out = []
    for c in range(top + 1):
        v = decode(c, mb, bias)
        nxt = decode(c + 1, mb, bias) if c < top else v * 2
        mid = (v + nxt) / 2
        for t in (v, mid, (v + mid) / 2, (mid + nxt) / 2):
            b = struct.unpack('<I', struct.pack('<f', float(t)))[0]
            out += [b, b - 1, b + 1]
    return out
if len(sys.argv) > 2 and sys.argv[2] == 'f16':
    def f16(h): return struct.unpack('<e', struct.pack('<H', h))[0]
    def encode16(h, fmt):
        eb, mb, top, nan, el = FMT[fmt]
        bias = (1 << (eb - 1)) - 1
        mag = h & 0x7fff
        if mag > 0x7c00: return nan
        sign = (h >> 15) << (el - 1)
        if mag == 0x7c00: return top | sign
        x = F(f16(mag))
        return min(range(top + 1), key=lambda c: (abs(decode(c, mb, bias) - x), c % 2)) | sign
    H = list(range(65536)); random.shuffle(H)
    A = [H[i] | (H[i + 1] << 16) for i in range(0, 65536, 2)]
    E = []
    for a in A:
        for f in ('e4m3', 'e5m2'):
            E.append((encode16(a >> 16, f) << 8) | encode16(a & 0xffff, f))
    ptx = '''// cvt.rn.satfinite.{e4m3x2,e5m2x2}.f16x2 of every f16 (pairs in a shuffled order); expected from exact
// rationals rounded to nearest even.
.version 8.7
.target sm_100a
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 po)
{
  .reg .u64 %x, %o, %t2; .reg .u32 %t, %t1, %t3; .reg .b32 %a, %r0, %r1; .reg .b16 %h0, %h1;
  ld.param.u64 %x, [pa]; ld.param.u64 %o, [po];
  mov.u32 %t, %tid.x; mov.u32 %t1, %ctaid.x; mov.u32 %t3, %ntid.x; mad.lo.u32 %t, %t1, %t3, %t;
  mul.wide.u32 %t2, %t, 4; add.u64 %x, %x, %t2;
  mul.wide.u32 %t2, %t, 8; add.u64 %o, %o, %t2;
  ld.global.b32 %a, [%x];
  cvt.rn.satfinite.e4m3x2.f16x2 %h0, %a; cvt.u32.u16 %r0, %h0;
  cvt.rn.satfinite.e5m2x2.f16x2 %h1, %a; cvt.u32.u16 %r1, %h1;
  st.global.v2.u32 [%o], {%r0, %r1};
  ret;
}
'''
    w = lambda xs: ', '.join(map(str, xs))
    ptx += '.buffer A .u32 = { %s };\n.buffer O .u32 [%d];\n.launch k <<< %d, 128 >>> (A, O);\n.expect O .u32 = { %s };\n' % (
        w(A), len(E), len(A) // 128, w(E))
    open(sys.argv[1], 'w').write(ptx)
    sys.exit(0)
if len(sys.argv) > 2 and sys.argv[2] == 'decode':
    def dec16(code, fmt):
        eb, mb, top, nan, el = FMT[fmt]
        bias = (1 << (eb - 1)) - 1
        code &= (1 << el) - 1
        mag = code & ((1 << (el - 1)) - 1)
        if (fmt == 'e4m3' and mag == 0x7f) or (fmt == 'e5m2' and mag > 0x7c): return 0x7fff
        sign = (code >> (el - 1)) << 15
        if fmt == 'e5m2' and mag == 0x7c: return 0x7c00 | sign
        v = float(decode(mag, mb, bias))
        return struct.unpack('<H', struct.pack('<e', v))[0] | sign
    def decbf(code, fmt):
        h = dec16(code, fmt)
        if h == 0x7fff: return 0x7fff
        if h & 0x7fff == 0x7c00: return 0x7f80 | (h & 0x8000)
        return struct.unpack('<I', struct.pack('<f', struct.unpack('<e', struct.pack('<H', h))[0]))[0] >> 16
    A = list(range(65536)); random.shuffle(A)
    E = []
    for a in A:
        for d in (dec16, decbf):
            for f in ('e4m3', 'e5m2', 'e2m3', 'e3m2'):
                E.append((d(a >> 8, f) << 16) | d(a & 0xff, f))
            E.append((d((a >> 4) & 0xf, 'e2m1') << 16) | d(a & 0xf, 'e2m1'))
    ptx = '''// cvt.rn.{f16x2,bf16x2}.{e4m3x2,e5m2x2,e2m3x2,e3m2x2} of every pair of codes, and .e2m1x2 of every byte (of the same
// words, in a shuffled order), expected from exact rationals.
.version 8.7
.target sm_100a
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 po)
{
  .reg .u64 %x, %o, %t2; .reg .u32 %t, %t1, %t3, %a; .reg .b32 %r<10>; .reg .b16 %h; .reg .b8 %q;
  ld.param.u64 %x, [pa]; ld.param.u64 %o, [po];
  mov.u32 %t, %tid.x; mov.u32 %t1, %ctaid.x; mov.u32 %t3, %ntid.x; mad.lo.u32 %t, %t1, %t3, %t;
  mul.wide.u32 %t2, %t, 4; add.u64 %x, %x, %t2;
  mul.wide.u32 %t2, %t, 40; add.u64 %o, %o, %t2;
  ld.global.u32 %a, [%x]; cvt.u16.u32 %h, %a; cvt.u8.u32 %q, %a;
  cvt.rn.f16x2.e4m3x2 %r0, %h;
  cvt.rn.f16x2.e5m2x2 %r1, %h;
  cvt.rn.f16x2.e2m3x2 %r2, %h;
  cvt.rn.f16x2.e3m2x2 %r3, %h;
  cvt.rn.f16x2.e2m1x2 %r4, %q;
  st.global.u32 [%o], %r0; st.global.u32 [%o+4], %r1; st.global.u32 [%o+8], %r2; st.global.u32 [%o+12], %r3;
  st.global.u32 [%o+16], %r4;
  cvt.rn.bf16x2.e4m3x2 %r5, %h;
  cvt.rn.bf16x2.e5m2x2 %r6, %h;
  cvt.rn.bf16x2.e2m3x2 %r7, %h;
  cvt.rn.bf16x2.e3m2x2 %r8, %h;
  cvt.rn.bf16x2.e2m1x2 %r9, %q;
  st.global.u32 [%o+20], %r5; st.global.u32 [%o+24], %r6; st.global.u32 [%o+28], %r7; st.global.u32 [%o+32], %r8;
  st.global.u32 [%o+36], %r9;
  ret;
}
'''
    w = lambda xs: ', '.join(map(str, xs))
    ptx += '.buffer A .u32 = { %s };\n.buffer O .u32 [%d];\n.launch k <<< %d, 128 >>> (A, O);\n.expect O .u32 = { %s };\n' % (
        w(A), len(E), len(A) // 128, w(E))
    open(sys.argv[1], 'w').write(ptx)
    sys.exit(0)
special = [0, 1, 0x7fffff, 0x800000, 0x7f7fffff, 0x7f800000, 0x7fc00000, 0x7f800001, 0x7fffffff]
vals = set()
for fmt in FMT:
    vals.update(edges(fmt))
vals.update(special)
vals = sorted(v & 0x7fffffff for v in vals)
vals = vals + [v | 0x80000000 for v in vals] + [random.getrandbits(32) for _ in range(256)]
random.shuffle(vals)
if len(vals) % 2: vals.append(0)
A = vals[0::2]; B = vals[1::2]
n = len(A)
blocks = (n + 127) // 128
while len(A) < blocks * 128: A.append(0); B.append(0)
order = ['e4m3', 'e5m2', 'e2m3', 'e3m2', 'e2m1']
E = []
for a, b in zip(A, B):
    lane = lambda f: 4 if FMT[f][4] == 4 else 8
    for f in order:
        E.append((encode(a, f) << lane(f)) | encode(b, f))
    E.append((encode(a, 'e4m3', True) << 8) | encode(b, 'e4m3', True))
ptx = '''// cvt.rn.satfinite to the narrow float formats (and .relu e4m3) of f32 pairs at the rounding ties, the
// subnormal/normal boundary, saturation, infinities, NaNs and f32 subnormals; expected from exact rationals.
.version 8.7
.target sm_100a
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 po)
{
  .reg .u64 %x, %y, %o, %t2; .reg .u32 %t, %t1, %t3; .reg .f32 %a, %b; .reg .b16 %h<6>; .reg .b8 %q; .reg .b32 %r<6>;
  ld.param.u64 %x, [pa]; ld.param.u64 %y, [pb]; ld.param.u64 %o, [po];
  mov.u32 %t, %tid.x; mov.u32 %t1, %ctaid.x; mov.u32 %t3, %ntid.x; mad.lo.u32 %t, %t1, %t3, %t;
  mul.wide.u32 %t2, %t, 4; add.u64 %x, %x, %t2; add.u64 %y, %y, %t2;
  mul.wide.u32 %t2, %t, 24; add.u64 %o, %o, %t2;
  ld.global.f32 %a, [%x]; ld.global.f32 %b, [%y];
  cvt.rn.satfinite.e4m3x2.f32 %h0, %a, %b; cvt.u32.u16 %r0, %h0;
  cvt.rn.satfinite.e5m2x2.f32 %h1, %a, %b; cvt.u32.u16 %r1, %h1;
  cvt.rn.satfinite.e2m3x2.f32 %h2, %a, %b; cvt.u32.u16 %r2, %h2;
  cvt.rn.satfinite.e3m2x2.f32 %h3, %a, %b; cvt.u32.u16 %r3, %h3;
  cvt.rn.satfinite.e2m1x2.f32 %q, %a, %b; cvt.u32.u8 %r4, %q;
  cvt.rn.satfinite.relu.e4m3x2.f32 %h5, %a, %b; cvt.u32.u16 %r5, %h5;
  st.global.v2.u32 [%o], {%r0, %r1}; st.global.v2.u32 [%o+8], {%r2, %r3}; st.global.v2.u32 [%o+16], {%r4, %r5};
  ret;
}
'''
w = lambda xs: ', '.join(map(str, xs))
ptx += '.buffer A .u32 = { %s };\n.buffer B .u32 = { %s };\n.buffer O .u32 [%d];\n.launch k <<< %d, 128 >>> (A, B, O);\n.expect O .u32 = { %s };\n' % (
    w(A), w(B), len(E), blocks, w(E))
open(sys.argv[1], 'w').write(ptx)
