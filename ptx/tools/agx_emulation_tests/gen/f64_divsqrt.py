# usage: f64_divsqrt.py SEED OUTDIR  (writes OUTDIR/divsqrt-SEED.ptx; expected values from Python: IEEE doubles / exact rationals / integers)
import os, sys, random, struct, math, sys
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
    if r<.72: return random.choice([0.0,-0.0,math.inf,-math.inf,1.0,-1.0,2.0**-1022,1.7976931348623157e308,4.0,2.0])
    if r<.85: return d(random.getrandbits(64) & 0x7fefffffffffffff | (random.getrandbits(1)<<63))
    return random.uniform(-1,1)*2.0**random.randint(-1074,1023)
A=[];B=[];E=[]
while len(A)<4096:
    x=rnd(); y=rnd()
    if random.random()<.15: y=x*random.choice([1,2,0.5,3,1+2**-52])  # exact-ish quotients
    try: q=x/y
    except ZeroDivisionError: q=math.copysign(math.inf,x)*math.copysign(1,y) if x!=0 else math.nan
    except OverflowError: q=math.copysign(math.inf, x)*math.copysign(1,y)
    ax=abs(x)
    s=math.sqrt(ax)
    if math.isnan(q): continue
    A.append(b(x)); B.append(b(y)); E+=[b(q), b(s)]
ptx='''.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 po)
{
  .reg .u64 %p<8>; .reg .u32 %t<4>; .reg .f64 %a, %b, %q, %s;
  ld.param.u64 %p0, [pa]; ld.param.u64 %p1, [pb]; ld.param.u64 %p2, [po];
  mov.u32 %t0, %tid.x; mov.u32 %t1, %ctaid.x; mov.u32 %t2, %ntid.x; mad.lo.u32 %t0, %t1, %t2, %t0;
  mul.wide.u32 %p4, %t0, 8; add.u64 %p0, %p0, %p4; add.u64 %p1, %p1, %p4;
  mul.wide.u32 %p5, %t0, 16; add.u64 %p2, %p2, %p5;
  ld.global.f64 %a, [%p0]; ld.global.f64 %b, [%p1];
  div.rn.f64 %q, %a, %b; abs.f64 %a, %a; sqrt.rn.f64 %s, %a;
  st.global.f64 [%p2], %q; st.global.f64 [%p2+8], %s;
  ret;
}
'''
w=lambda xs: ', '.join(map(str,xs))
ptx+='.buffer A .u64 = { %s };\n.buffer B .u64 = { %s };\n.buffer O .u64 [%d];\n.launch k <<< 32, 128 >>> (A, B, O);\n.expect O .u64 = { %s };\n' % (w(A),w(B),len(E),w(E))
open(os.path.join(sys.argv[2], 'divsqrt-%s.ptx' % sys.argv[1]), 'w').write(ptx)
