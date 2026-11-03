# usage: f64_integral.py SEED OUTDIR  (writes OUTDIR/integral-SEED.ptx; expected values from Python)
import os, random, struct, math, sys
seed=int(sys.argv[1]); random.seed(seed)
d2u=lambda x: struct.unpack('<Q',struct.pack('<d',x))[0]
u2d=lambda u: struct.unpack('<d',struct.pack('<Q',u))[0]
def rnd():
    r=random.random()
    if r<.3: return random.uniform(-100,100)
    if r<.45: return float(random.randint(-10**6,10**6))+random.choice([0,.5,-.5,.25,.75,1e-9])
    if r<.55: return random.uniform(-1,1)
    if r<.6: return random.choice([0.0,-0.0,0.5,-0.5,1.5,-1.5,2.5,-2.5,0.49999999999999994,-0.49999999999999994,1.0,-1.0,2.0**52,2.0**52+1,-(2.0**52)-1,2.0**52-0.5,4503599627370495.5,math.inf,-math.inf,5e-324,-5e-324])
    if r<.7: return u2d(random.getrandbits(52)|(random.getrandbits(1)<<63))
    return u2d(random.getrandbits(64)&0x7fefffffffffffff|(random.getrandbits(1)<<63))
def rne(x):
    if math.isinf(x) or x == 0: return x
    r = round(x)   # Python round: ties to even
    return math.copysign(float(r), x) if r == 0 else float(r)
def away(x):
    if math.isinf(x): return x
    r = math.floor(abs(x) + 0.5) if abs(x) < 2**52 else abs(x)
    return math.copysign(float(r), x)
def fl(x): return x if math.isinf(x) else (math.copysign(0.0, x) if x == 0 else float(math.floor(x)) if math.floor(x) != 0 or x >= 0 else -0.0)
def ce(x):
    if math.isinf(x): return x
    c = math.ceil(x)
    return math.copysign(0.0, x) if c == 0 else float(c)
def tr(x):
    if math.isinf(x): return x
    t = math.trunc(x)
    return math.copysign(0.0, x) if t == 0 else float(t)
D=[];E=[]
while len(D)<2048:
    x=rnd(); D.append(d2u(x))
    def sat(v,lo,hi): return lo if v<=lo else hi if v>=hi else int(v)
    ri = 0 if math.isnan(x) else sat(rne(x), -2**31, 2**31-1) & 0xffffffff
    mi = 0 if math.isnan(x) else sat(fl(x), -2**63, 2**63-1) & ((1<<64)-1)
    E+=[d2u(fl(x)), d2u(ce(x)), d2u(tr(x)), d2u(rne(x)), ri, mi]
ptx='''.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 pd, .param .u64 po)
{
  .reg .u64 %p<12>; .reg .u32 %t; .reg .f64 %d, %a, %b, %c, %e; .reg .s32 %ri; .reg .s64 %mi, %ri64;
  ld.param.u64 %p0, [pd]; ld.param.u64 %p2, [po];
  mov.u32 %t, %tid.x; { .reg .u32 c, n; mov.u32 c, %ctaid.x; mov.u32 n, %ntid.x; mad.lo.u32 %t, c, n, %t; }
  mul.wide.u32 %p4, %t, 8; add.u64 %p5, %p0, %p4; ld.global.f64 %d, [%p5];
  cvt.rmi.f64.f64 %a, %d; cvt.rpi.f64.f64 %b, %d; cvt.rzi.f64.f64 %c, %d; cvt.rni.f64.f64 %e, %d;
  cvt.rni.s32.f64 %ri, %d; cvt.rmi.s64.f64 %mi, %d; cvt.u64.u32 %ri64, %ri;
  mul.wide.u32 %p7, %t, 48; add.u64 %p8, %p2, %p7;
  st.global.f64 [%p8], %a; st.global.f64 [%p8+8], %b; st.global.f64 [%p8+16], %c; st.global.f64 [%p8+24], %e;
  st.global.u64 [%p8+32], %ri64; st.global.s64 [%p8+40], %mi;
  ret;
}
'''
w=lambda xs: ', '.join(map(str,xs))
ptx+='.buffer D .u64 = { %s };\n.buffer O .u64 [%d];\n.launch k <<< 16, 128 >>> (D, O);\n.expect O .u64 = { %s };\n' % (w(D), len(E), w(E))
open(os.path.join(sys.argv[2], 'integral-%s.ptx' % sys.argv[1]), 'w').write(ptx)
