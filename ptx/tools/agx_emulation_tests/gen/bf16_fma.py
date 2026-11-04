# usage: bf16_fma.py OUT.ptx  (fma.rn.bf16x2 of 4096 bfloat triples, expected from exact rationals; strict mode only)
import random, struct, math, sys
from fractions import Fraction as F
random.seed(21)
def bfv(h): return struct.unpack('<f',struct.pack('<I',h<<16))[0]
def rn_bf16(e):
    s=1 if e<0 else 0; a=abs(e)
    ex=a.numerator.bit_length()-a.denominator.bit_length()
    if F(2)**ex>a: ex-=1
    q=max(ex-7,-133)
    u=F(2)**q; n=a/u; lo=n.numerator//n.denominator; rem=n-lo
    k=lo+1 if rem>F(1,2) or (rem==F(1,2) and lo%2==1) else lo
    v=k*u
    if v>F(2)**127*(2-F(1,128)): return 0x7f80|(s<<15)
    return (struct.unpack('<I',struct.pack('<f',float(v)))[0]>>16)|(s<<15)
def rnd():
    r=random.random()
    if r<.4: x=random.uniform(1,2)*2.0**random.randint(-12,12)*random.choice([1,-1]); return struct.unpack('<I',struct.pack('<f',x))[0]>>16
    if r<.55: return random.getrandbits(16)&0x807f
    if r<.62: return random.choice([0,0x8000,0x3f80,0xbf80,0x0080,0x7f7f,0xff7f])
    if r<.75: return (random.getrandbits(7)|(random.choice([1,2,3,20,30,60,63,64,65,200,250,253,254])<<7)|(random.getrandbits(1)<<15))
    return random.getrandbits(16)&0x7f7f|(random.getrandbits(1)<<15)
A=[];B=[];C=[];E=[]
while len(A)<4096:
    a,b=rnd(),rnd()
    va,vb=bfv(a),bfv(b)
    if random.random()<.2 and va*vb!=0 and abs(va*vb)<3e38:
        c=(struct.unpack('<I',struct.pack('<f',-va*vb))[0]>>16)^random.choice([0,1,2])
    else: c=rnd()
    vc=bfv(c)
    if any(math.isinf(v) or math.isnan(v) for v in (va,vb,vc)): continue
    e=F(va)*F(vb)+F(vc)
    if e==0:
        if (va==0 or vb==0) and vc==0:
            sp=math.copysign(1,va)*math.copysign(1,vb)<0; sz=math.copysign(1,vc)<0
            v=0x8000 if (sp and sz) else 0
        else: v=0
    else: v=rn_bf16(e)
    A.append(a);B.append(b);C.append(c);E.append(v)
P=lambda L: [L[i]|(L[i+1]<<16) for i in range(0,len(L),2)]
ptx='''// fma.rn.bf16x2 of 4096 bfloat triples (realistic, subnormal, edge exponents, cancellation), expected from exact
// rationals rounded to nearest even bfloat.
.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 po)
{
  .reg .u64 %x, %y, %z, %o, %t2; .reg .u32 %t, %t1, %t3; .reg .b32 %a, %b, %c, %r;
  ld.param.u64 %x, [pa]; ld.param.u64 %y, [pb]; ld.param.u64 %z, [pc]; ld.param.u64 %o, [po];
  mov.u32 %t, %tid.x; mov.u32 %t1, %ctaid.x; mov.u32 %t3, %ntid.x; mad.lo.u32 %t, %t1, %t3, %t;
  mul.wide.u32 %t2, %t, 4; add.u64 %x, %x, %t2; add.u64 %y, %y, %t2; add.u64 %z, %z, %t2; add.u64 %o, %o, %t2;
  ld.global.b32 %a, [%x]; ld.global.b32 %b, [%y]; ld.global.b32 %c, [%z];
  fma.rn.bf16x2 %r, %a, %b, %c;
  st.global.b32 [%o], %r;
  ret;
}
'''
w=lambda xs: ', '.join(map(str,xs))
EP=P(E)
ptx+='.buffer A .u32 = { %s };\n.buffer B .u32 = { %s };\n.buffer C .u32 = { %s };\n.buffer O .u32 [%d];\n.launch k <<< 16, 128 >>> (A, B, C, O);\n.expect O .u32 = { %s };\n'%(w(P(A)),w(P(B)),w(P(C)),len(EP),w(EP))
open(sys.argv[1],'w').write(ptx)
