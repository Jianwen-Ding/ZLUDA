# usage: f32_fma_directed.py SEED OUTDIR  (writes OUTDIR/fma-directed-SEED.ptx; expected values from Python: IEEE doubles / exact rationals / integers)
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
    if r < .55: return f32(random.uniform(-8, 8) * 2.0 ** random.randint(-40, 40))
    if r < .65: return bf(random.getrandbits(32) & 0x807fffff | (random.choice([1, 2, 254, 253, 127, 30, 60]) << 23))
    if r < .72: return bf(random.getrandbits(23) | (random.getrandbits(1) << 31))   # subnormal
    if r < .76: return random.choice([0.0, -0.0, 1.0, -1.0, bf(0x00800000), bf(0x7f7fffff)])
    return bf(random.getrandbits(32) & 0x7f7fffff | (random.getrandbits(1) << 31))
modes = ['rz', 'rp', 'rm']
A=[];B=[];C=[];E=[]
while len(A) < 2048:
    a, b = rnd(), rnd(); r = random.random()
    try:
        c = -f32(a * b * (1 + random.choice([0, 2**-23, -2**-23, 2**-12]))) if r < .25 else rnd()
    except OverflowError:
        c = rnd()
    if not all(math.isfinite(v) for v in (a, b, c)): continue
    e = F(a) * F(b) + F(c)
    outs = []
    ok = True
    for m in modes:
        v = round32(e, m)
        if v is None:   # exact zero: sign rules
            pz = (a == 0 or b == 0); sp = (math.copysign(1, a) * math.copysign(1, b) < 0); sz = math.copysign(1, c) < 0
            if pz and c == 0: v = 0x80000000 if (sp and sz) or (m == 'rm' and (sp or sz)) else 0
            else: v = 0x80000000 if m == 'rm' else 0
        outs.append(v)
    A.append(fb(a)); B.append(fb(b)); C.append(fb(c)); E += outs
ptx = '''.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 po)
{
  .reg .u64 %p<8>; .reg .u32 %t<4>; .reg .f32 %a, %b, %c, %x, %y, %z;
  ld.param.u64 %p0, [pa]; ld.param.u64 %p1, [pb]; ld.param.u64 %p3, [pc]; ld.param.u64 %p2, [po];
  mov.u32 %t0, %tid.x; mov.u32 %t1, %ctaid.x; mov.u32 %t2, %ntid.x; mad.lo.u32 %t0, %t1, %t2, %t0;
  mul.wide.u32 %p4, %t0, 4; add.u64 %p0, %p0, %p4; add.u64 %p1, %p1, %p4; add.u64 %p3, %p3, %p4;
  mul.wide.u32 %p5, %t0, 12; add.u64 %p2, %p2, %p5;
  ld.global.f32 %a, [%p0]; ld.global.f32 %b, [%p1]; ld.global.f32 %c, [%p3];
  fma.rz.f32 %x, %a, %b, %c; fma.rp.f32 %y, %a, %b, %c; fma.rm.f32 %z, %a, %b, %c;
  st.global.f32 [%p2], %x; st.global.f32 [%p2+4], %y; st.global.f32 [%p2+8], %z;
  ret;
}
'''
w = lambda xs: ', '.join(map(str, xs))
ptx += '.buffer A .u32 = { %s };\n.buffer B .u32 = { %s };\n.buffer C .u32 = { %s };\n.buffer O .u32 [%d];\n.launch k <<< 16, 128 >>> (A, B, C, O);\n.expect O .u32 = { %s };\n' % (w(A), w(B), w(C), len(E), w(E))
open(os.path.join(sys.argv[2], 'fma-directed-%s.ptx' % sys.argv[1]), 'w').write(ptx)
