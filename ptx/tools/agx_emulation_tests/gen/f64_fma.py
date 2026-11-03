# usage: f64_fma.py SEED OUTDIR  (writes OUTDIR/fma-SEED.ptx; expected values from Python: IEEE doubles / exact rationals / integers)
import os, sys, random, struct, math, sys
from fractions import Fraction as F
seed=int(sys.argv[1]); random.seed(seed)
def b(x): return struct.unpack('<Q', struct.pack('<d', x))[0]
def d(u): return struct.unpack('<d', struct.pack('<Q', u))[0]
def rnd():
    if os.environ.get('ZEROS') and random.random() < .4: return random.choice([0.0, -0.0])
    return rnd0()
def rnd0():
    r=random.random()
    if r<.5: return random.uniform(-10,10)*2.0**random.randint(-30,30)
    if r<.6: return d(random.getrandbits(64) & ~(0x7ff<<52) | (random.choice([1,2,2046,2045,1023])<<52))
    if r<.68: return d(random.getrandbits(52) | (random.getrandbits(1)<<63))
    if r<.72: return random.choice([0.0,-0.0,math.inf,-math.inf,1.0,-1.0,2.0**-1022,1.7976931348623157e308])
    if r<.85: return d(random.getrandbits(64) & 0x7fefffffffffffff | (random.getrandbits(1)<<63))
    return random.uniform(-1,1)*2.0**random.randint(-1074,1023)
def fma(x,y,z):
    if any(math.isinf(v) or math.isnan(v) for v in (x,y,z)):
        r=x*y+z if not (math.isinf(x*y) and math.isinf(z) and (x*y)!=z) else math.nan
        return r
    e=F(x)*F(y)+F(z)
    if e==0:
        # exact zero: sign rules (RNE): +0 unless both x*y and z are -0 or negative zero sum
        p0 = (x==0 or y==0); 
        sp = math.copysign(1,x)*math.copysign(1,y) < 0
        sz = math.copysign(1,z) < 0
        if p0 and z==0: return -0.0 if (sp and sz) else 0.0
        return 0.0
    try: return e.numerator/e.denominator
    except OverflowError: return math.inf if e>0 else -math.inf
A=[];B=[];C=[];E=[]
while len(A)<4096:
    x=rnd(); y=rnd(); r=random.random()
    if r<.25:
        p=x*y
        if math.isinf(p) or math.isnan(p): continue
        z=-p*(1+random.choice([0,2**-52,-2**-52,2**-40,2**-60]))   # cancellation
    elif r<.4: z=x*y*2.0**random.randint(-80,80) if not math.isinf(x*y) else rnd()
    else: z=rnd()
    if math.isinf(z) or math.isnan(z): z=rnd()
    v=fma(x,y,z)
    if isinstance(v,float) and math.isnan(v): continue
    A.append(b(x)); B.append(b(y)); C.append(b(z)); E.append(b(v))
ptx='''.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 po)
{
  .reg .u64 %p<8>; .reg .u32 %t<4>; .reg .f64 %a, %b, %c, %r;
  ld.param.u64 %p0, [pa]; ld.param.u64 %p1, [pb]; ld.param.u64 %p3, [pc]; ld.param.u64 %p2, [po];
  mov.u32 %t0, %tid.x; mov.u32 %t1, %ctaid.x; mov.u32 %t2, %ntid.x; mad.lo.u32 %t0, %t1, %t2, %t0;
  mul.wide.u32 %p4, %t0, 8; add.u64 %p0, %p0, %p4; add.u64 %p1, %p1, %p4; add.u64 %p3, %p3, %p4; add.u64 %p2, %p2, %p4;
  ld.global.f64 %a, [%p0]; ld.global.f64 %b, [%p1]; ld.global.f64 %c, [%p3];
  fma.rn.f64 %r, %a, %b, %c;
  st.global.f64 [%p2], %r;
  ret;
}
'''
w=lambda xs: ', '.join(map(str,xs))
ptx+='.buffer A .u64 = { %s };\n.buffer B .u64 = { %s };\n.buffer C .u64 = { %s };\n.buffer O .u64 [%d];\n.launch k <<< 32, 128 >>> (A, B, C, O);\n.expect O .u64 = { %s };\n' % (w(A),w(B),w(C),len(E),w(E))
open(os.path.join(sys.argv[2], 'fma-%s.ptx' % sys.argv[1]), 'w').write(ptx)
