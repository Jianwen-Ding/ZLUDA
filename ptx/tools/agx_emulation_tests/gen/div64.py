# usage: div64.py OUTDIR  (writes OUTDIR/div64.ptx)
import os, sys, random
random.seed(11)
M=(1<<64)-1
def s(x): return x-(1<<64) if x>>63 else x
cases=[]
edges=[0,1,2,3,7,(1<<32)-1,1<<32,(1<<32)+1,(1<<63)-1,1<<63,(1<<63)+1,M,M-1,0xFFFFFFFF00000000,0x8000000000000001,12345678901234567]
for a in edges:
    for b in edges: cases.append((a,b))
while len(cases)<2048:
    a=random.getrandbits(64)>>random.choice([0,0,0,1,7,20,31,32,33,40,50,60])
    b=random.getrandbits(64)>>random.randint(0,63)
    if random.random()<.1: b=1<<random.randint(0,63)
    if random.random()<.1: b=(1<<random.randint(1,64))-1
    if random.random()<.05: a=b*random.getrandbits(random.randint(0,20))+random.randint(0,3)
    cases.append((a&M,b&M))
cases=cases[:2048]
def udiv(a,b): return M if b==0 else a//b
def urem(a,b): return a if b==0 else a%b
def sdiv(a,b):
    A,Bv=s(a),s(b)
    if Bv==0: return M
    q=abs(A)//abs(Bv); q=-q if (A<0)!=(Bv<0) else q
    return q&M
def srem(a,b):
    A,Bv=s(a),s(b)
    if Bv==0: return a
    r=abs(A)%abs(Bv); r=-r if A<0 else r
    return r&M
ptx='''.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 po)
{
  .reg .u64 %p<8>; .reg .u32 %t<4>; .reg .u64 %a, %b, %q, %r, %sq, %sr;
  ld.param.u64 %p0, [pa]; ld.param.u64 %p1, [pb]; ld.param.u64 %p2, [po];
  mov.u32 %t0, %tid.x; mov.u32 %t1, %ctaid.x; mov.u32 %t2, %ntid.x; mad.lo.u32 %t0, %t1, %t2, %t0;
  mul.wide.u32 %p4, %t0, 8; add.u64 %p0, %p0, %p4; add.u64 %p1, %p1, %p4;
  mul.wide.u32 %p5, %t0, 32; add.u64 %p2, %p2, %p5;
  ld.global.u64 %a, [%p0]; ld.global.u64 %b, [%p1];
  div.u64 %q, %a, %b; rem.u64 %r, %a, %b; div.s64 %sq, %a, %b; rem.s64 %sr, %a, %b;
  st.global.u64 [%p2], %q; st.global.u64 [%p2+8], %r; st.global.u64 [%p2+16], %sq; st.global.u64 [%p2+24], %sr;
  ret;
}
'''
w=lambda xs: ', '.join(str(x) for x in xs)
A=[a for a,b in cases]; Bs=[b for a,b in cases]
exp=[]
for a,b in cases: exp+= [udiv(a,b),urem(a,b),sdiv(a,b),srem(a,b)]
ptx+='.buffer A .u64 = { %s };\n.buffer B .u64 = { %s };\n.buffer O .u64 [%d];\n.launch k <<< 16, 128 >>> (A, B, O);\n.expect O .u64 = { %s };\n' % (w(A),w(Bs),len(exp),w(exp))
open(os.path.join(sys.argv[1], 'div64.ptx'),'w').write(ptx)
