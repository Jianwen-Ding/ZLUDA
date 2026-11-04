# usage: f32_addmul_directed.py SEED OUTDIR  (writes OUTDIR/addmul-directed-SEED.ptx: add / sub / mul .rz / .rp / .rm.f32; expected values from Python: IEEE doubles / exact rationals / integers)
import os, sys, random, struct, math, sys
from fractions import Fraction as F
seed=int(sys.argv[1]); random.seed(seed)
def fb(x): return struct.unpack('<I', struct.pack('<f', x))[0]
def bf(u): return struct.unpack('<f', struct.pack('<I', u))[0]
def f32(x): return bf(fb(x))
MAXF=F(bf(0x7f7fffff)); TINY=F(2)**-149
def round32(e, mode):   # e Fraction, exact; returns bits
    if e == 0: return None
    s = -1 if e < 0 else 1; a = abs(e)
    # find quantum: exponent
    ex = a.numerator.bit_length() - a.denominator.bit_length()
    if F(2)**ex > a: ex -= 1
    q = max(ex - 23, -149)
    u = F(2)**q
    n = a / u; lo = n.numerator // n.denominator
    exact = (lo * u == a)
    if mode == 'rz': k = lo
    elif mode == 'rp': k = lo if exact or s < 0 else lo + 1
    elif mode == 'rm': k = lo if exact or s > 0 else lo + 1
    v = k * u
    if v > MAXF:
        if mode == 'rz' or (mode == 'rp' and s < 0) or (mode == 'rm' and s > 0): v = MAXF
        else: return fb(math.inf) | (0x80000000 if s < 0 else 0)
    return fb(float(v)) | (0x80000000 if s < 0 else 0)
def rnd():
    r = random.random()
    if r < .30: return f32(random.uniform(-8, 8) * 2.0 ** random.randint(-40, 40))
    # around the bounds of the error-free fast paths (2^-102, 2^-70, 2^126, 2^127) and the ends of the range
    if r < .60: return bf(random.getrandbits(23) | (random.choice([1, 2, 3, 24, 25, 26, 56, 57, 58, 252, 253, 254]) << 23) | (random.getrandbits(1) << 31))
    if r < .67: return bf(random.getrandbits(23) | (random.getrandbits(1) << 31))   # subnormal
    if r < .71: return random.choice([0.0, -0.0, 1.0, -1.0, bf(0x00800000), bf(0x7f7fffff), bf(0x80000001)])
    return bf(random.getrandbits(32) & 0x7f7fffff | (random.getrandbits(1) << 31))
def sgn(x): return math.copysign(1, x) < 0
modes = ['rz', 'rp', 'rm']
A=[];B=[];E=[]
while len(A) < 2048:
    a = rnd(); r = random.random()
    if r < .15: b = -a                                                       # exact zero sum
    else:
        try:
            if r < .35: b = -f32(a * (1 + random.choice([2**-23, -2**-23, 2**-22, 2**-12])))   # cancellation
            elif r < .45: b = f32(a * 2.0 ** -random.randint(20, 30))             # far apart
            else: b = rnd()
        except OverflowError:
            b = rnd()
    if not all(math.isfinite(v) for v in (a, b)): continue
    outs = []
    for op in ('add', 'sub', 'mul'):
        bb = -b if op == 'sub' else b
        e = F(a) * F(bb) if op == 'mul' else F(a) + F(bb)
        for m in modes:
            v = round32(e, m)
            if v is None:   # exact zero: sign rules
                if op == 'mul': v = 0x80000000 if sgn(a) != sgn(bb) else 0
                elif a == 0 and bb == 0 and sgn(a) == sgn(bb): v = 0x80000000 if sgn(a) else 0
                else: v = 0x80000000 if m == 'rm' else 0
            outs.append(v)
    A.append(fb(a)); B.append(fb(b)); E += outs
ptx = """.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 po)
{
  .reg .u64 %p<8>; .reg .u32 %t<4>; .reg .f32 %a, %b, %r<9>;
  ld.param.u64 %p0, [pa]; ld.param.u64 %p1, [pb]; ld.param.u64 %p2, [po];
  mov.u32 %t0, %tid.x; mov.u32 %t1, %ctaid.x; mov.u32 %t2, %ntid.x; mad.lo.u32 %t0, %t1, %t2, %t0;
  mul.wide.u32 %p4, %t0, 4; add.u64 %p0, %p0, %p4; add.u64 %p1, %p1, %p4;
  mul.wide.u32 %p5, %t0, 36; add.u64 %p2, %p2, %p5;
  ld.global.f32 %a, [%p0]; ld.global.f32 %b, [%p1];
  add.rz.f32 %r0, %a, %b; add.rp.f32 %r1, %a, %b; add.rm.f32 %r2, %a, %b;
  sub.rz.f32 %r3, %a, %b; sub.rp.f32 %r4, %a, %b; sub.rm.f32 %r5, %a, %b;
  mul.rz.f32 %r6, %a, %b; mul.rp.f32 %r7, %a, %b; mul.rm.f32 %r8, %a, %b;
  st.global.f32 [%p2], %r0; st.global.f32 [%p2+4], %r1; st.global.f32 [%p2+8], %r2;
  st.global.f32 [%p2+12], %r3; st.global.f32 [%p2+16], %r4; st.global.f32 [%p2+20], %r5;
  st.global.f32 [%p2+24], %r6; st.global.f32 [%p2+28], %r7; st.global.f32 [%p2+32], %r8;
  ret;
}
"""
w = lambda xs: ', '.join(map(str, xs))
ptx += '.buffer A .u32 = { %s };\n.buffer B .u32 = { %s };\n.buffer O .u32 [%d];\n.launch k <<< 16, 128 >>> (A, B, O);\n.expect O .u32 = { %s };\n' % (w(A), w(B), len(E), w(E))
open(os.path.join(sys.argv[2], 'addmul-directed-%s.ptx' % sys.argv[1]), 'w').write(ptx)
