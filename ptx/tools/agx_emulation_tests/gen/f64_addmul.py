# usage: f64_addmul.py SEED OUTDIR  (writes OUTDIR/addmul-SEED.ptx; expected values from Python: IEEE doubles / exact rationals / integers)
import os, sys, random, struct, math, sys
random.seed(int(sys.argv[1]) if len(sys.argv)>1 else 1)
def b(x): return struct.unpack('<Q', struct.pack('<d', x))[0]
def d(u): return struct.unpack('<d', struct.pack('<Q', u))[0]
def rnd():
    if os.environ.get('ZEROS') and random.random() < .4: return random.choice([0.0, -0.0])
    return rnd0()
def rnd0():
    r=random.random()
    if r<.5: return random.uniform(-10,10)*2.0**random.randint(-30,30)
    if r<.6: return d(random.getrandbits(64) & ~(0x7ff<<52) | (random.choice([1,2,2046,2045,1023])<<52))  # edge exponents
    if r<.7: return d(random.getrandbits(52) | (random.getrandbits(1)<<63))  # subnormal
    if r<.75: return random.choice([0.0,-0.0,math.inf,-math.inf,1.0,-1.0,2.0**-1022,-2.0**-1022,1.7976931348623157e308])
    if r<.85: return d(random.getrandbits(64) & 0x7fefffffffffffff | (random.getrandbits(1)<<63))
    return random.uniform(-1,1)*2.0**random.randint(-1074,1023) if random.random()<.5 else 2.0**random.randint(-1074,1023)
A=[];B=[];E=[]
N=4096
while len(A)<N:
    x=rnd(); r=random.random()
    if r<.2: y=-x*(1+random.choice([0,2**-52,-2**-52,2**-30]))   # cancellation
    elif r<.3: y=x*2.0**random.randint(-60,60)
    else: y=rnd()
    try:
        s=x+y; m=x*y; t=x-y
    except OverflowError: continue
    if any(math.isnan(v) for v in (s,m,t)) or math.isnan(x) or math.isnan(y): continue
    A.append(b(x)); B.append(b(y)); E+= [b(s), b(t), b(m)]
ptx='''.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 po)
{
  .reg .u64 %p<8>; .reg .u32 %t<4>; .reg .f64 %a, %b, %s, %d, %m;
  ld.param.u64 %p0, [pa]; ld.param.u64 %p1, [pb]; ld.param.u64 %p2, [po];
  mov.u32 %t0, %tid.x; mov.u32 %t1, %ctaid.x; mov.u32 %t2, %ntid.x; mad.lo.u32 %t0, %t1, %t2, %t0;
  mul.wide.u32 %p4, %t0, 8; add.u64 %p0, %p0, %p4; add.u64 %p1, %p1, %p4;
  mul.wide.u32 %p5, %t0, 24; add.u64 %p2, %p2, %p5;
  ld.global.f64 %a, [%p0]; ld.global.f64 %b, [%p1];
  add.rn.f64 %s, %a, %b; sub.rn.f64 %d, %a, %b; mul.rn.f64 %m, %a, %b;
  st.global.f64 [%p2], %s; st.global.f64 [%p2+8], %d; st.global.f64 [%p2+16], %m;
  ret;
}
'''
w=lambda xs: ', '.join(map(str,xs))
ptx+='.buffer A .u64 = { %s };\n.buffer B .u64 = { %s };\n.buffer O .u64 [%d];\n.launch k <<< 32, 128 >>> (A, B, O);\n.expect O .u64 = { %s };\n' % (w(A),w(B),len(E),w(E))
open(os.path.join(sys.argv[2], 'addmul-%s.ptx' % sys.argv[1]), 'w').write(ptx)
