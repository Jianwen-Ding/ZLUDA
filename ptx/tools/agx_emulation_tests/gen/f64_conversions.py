# usage: f64_conversions.py SEED OUTDIR  (writes OUTDIR/conversions-SEED.ptx; expected values from Python: IEEE doubles / exact rationals / integers)
import os, sys, random, struct, math, sys
seed=int(sys.argv[1]); random.seed(seed)
d2u=lambda x: struct.unpack('<Q',struct.pack('<d',x))[0]
u2d=lambda u: struct.unpack('<d',struct.pack('<Q',u))[0]
f2u=lambda x: struct.unpack('<I',struct.pack('<f',x))[0]
u2f=lambda u: struct.unpack('<f',struct.pack('<I',u))[0]
M=(1<<64)-1
def rndd():
    r=random.random()
    if r<.3: return random.uniform(-1e6,1e6)
    if r<.45: return float(random.randint(-2**40,2**40))+random.choice([0,.5,.25,.75,-.5])
    if r<.55: return random.uniform(-1,1)*2.0**random.randint(-1074,1023)
    if r<.62: return u2d(random.getrandbits(52)|(random.getrandbits(1)<<63))
    if r<.7: return random.choice([0.0,-0.0,1.0,-1.0,2.0**31,-2.0**31,2.0**31-1,2.0**63,-2.0**63,2.0**64,2.0**53+1,3.4028235677973366e38,3.4028236e38,1.1754943508222875e-38,1.1754942e-38,1e-45,-1e300,math.inf,-math.inf])
    return u2d(random.getrandbits(64)&0x7fefffffffffffff|(random.getrandbits(1)<<63))
def tof32(x):
    try: return f2u(x)   # struct rounds to nearest even, with denormals
    except OverflowError: return 0x7f800000 | (0x80000000 if x<0 else 0)
def sat(x,lo,hi):
    if math.isnan(x): return 0
    if x<=lo: return lo
    if x>=hi: return hi
    return int(x)
D=[];I64=[];E=[]
while len(D)<2048:
    x=rndd()
    if math.isnan(x): continue
    i=random.choice([random.getrandbits(64), random.getrandbits(random.randint(1,64)), random.randint(-2**53,2**53)&M, 0, M, 1<<63, (1<<53)+1, (1<<63)-1])
    s=i-(1<<64) if i>>63 else i
    f=tof32(x)
    # PTX cvt.rzi.s64.f64 / u64 saturate; NaN -> 0
    e=[f, d2u(u2f(f)) if (f&0x7f800000)!=0x7f800000 or True else 0,
       sat(x,-2**63,2**63-1)&M, sat(x,0,2**64-1)&M, sat(x,-2**31,2**31-1)&0xffffffff,
       d2u(float(s)), d2u(float(i)), d2u(float(s&0xffffffff if False else ((s+2**31)%2**32)-2**31))]
    D.append(d2u(x)); I64.append(i); E+=e
# f32->f64 of the converted float: math via python float(u2f) exact
ptx='''.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 pd, .param .u64 pi, .param .u64 po)
{
  .reg .u64 %p<12>; .reg .u32 %t; .reg .f64 %d, %e, %g, %h, %k; .reg .f32 %f; .reg .s64 %s, %si; .reg .u64 %u, %ui; .reg .s32 %w, %wi;
  ld.param.u64 %p0, [pd]; ld.param.u64 %p1, [pi]; ld.param.u64 %p2, [po];
  mov.u32 %t, %tid.x; { .reg .u32 c, n; mov.u32 c, %ctaid.x; mov.u32 n, %ntid.x; mad.lo.u32 %t, c, n, %t; }
  mul.wide.u32 %p4, %t, 8; add.u64 %p5, %p0, %p4; add.u64 %p6, %p1, %p4;
  ld.global.f64 %d, [%p5]; ld.global.s64 %si, [%p6]; ld.global.u64 %ui, [%p6]; cvt.u32.u64 %wi, %ui;
  cvt.rn.f32.f64 %f, %d; cvt.f64.f32 %e, %f;
  cvt.rzi.s64.f64 %s, %d; cvt.rzi.u64.f64 %u, %d; cvt.rzi.s32.f64 %w, %d;
  cvt.rn.f64.s64 %g, %si; cvt.rn.f64.u64 %h, %ui; cvt.rn.f64.s32 %k, %wi;
  mul.wide.u32 %p7, %t, 64; add.u64 %p8, %p2, %p7;
  st.global.f32 [%p8], %f; st.global.f64 [%p8+8], %e; st.global.s64 [%p8+16], %s; st.global.u64 [%p8+24], %u;
  st.global.s32 [%p8+32], %w; st.global.f64 [%p8+40], %g; st.global.f64 [%p8+48], %h; st.global.f64 [%p8+56], %k;
  ret;
}
'''
# pack expected into u32 words matching layout: f32 (4 bytes) + pad 4, f64, s64, u64, s32 + pad, f64, f64, f64
W=[]
for j in range(len(D)):
    e=E[8*j:8*j+8]
    W+= [e[0], None]
    for v in (e[1], e[2], e[3]): W += [v & 0xffffffff, v >> 32]
    W += [e[4], None]
    for v in (e[5], e[6], e[7]): W += [v & 0xffffffff, v >> 32]
w=lambda xs: ', '.join(map(str,xs))
# padding words: initial zeros in O
W=[0 if v is None else v for v in W]
ptx+='.buffer D .u64 = { %s };\n.buffer I .u64 = { %s };\n.buffer O .u32 [%d];\n.launch k <<< 16, 128 >>> (D, I, O);\n.expect O .u32 = { %s };\n' % (w(D), w(I64), len(W), w(W))
open(os.path.join(sys.argv[2], 'conversions-%s.ptx' % sys.argv[1]), 'w').write(ptx)
