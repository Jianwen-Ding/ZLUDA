#!/usr/bin/env python3
"""Numerical graph-runtime tests; see support.py for the driver harness."""
from support import *
import support

rng=np.random.default_rng(1234)
# All elementwise modes, scalar and multidimensional broadcasting, and fused ReLU.
for kind in ['add','mul','sub','div','max','min']:
    g=graph(); a=tensor(g,'a',[2,1,5]); b=tensor(g,'b',[1,4,1]); y=node(g,[2,4,5],'element_wise',a=a,b=b,kind=kind)
    g['outputs']=[y]; av=rng.normal(size=[2,1,5]).astype('f'); bv=rng.uniform(.5,2,size=[1,4,1]).astype('f')
    result={'add':lambda:av+bv,'mul':lambda:av*bv,'sub':lambda:av-bv,'div':lambda:av/bv,'max':lambda:np.maximum(av,bv),'min':lambda:np.minimum(av,bv)}[kind]()
    run(g,{'a':av,'b':bv},{f't{y}':result})
g=graph(); a=tensor(g,'a',[2,3,4]); b=tensor(g,'b',[3,1],rng.normal(size=[3,1])); y=node(g,[2,3,4],'element_wise',a=a,b=b,kind='add'); z=node(g,[2,3,4],'activation',input=y,kind='relu'); g['outputs']=[z]
av=rng.normal(size=[2,3,4]).astype('f'); bv=np.asarray(g['tensors'][b]['data']).reshape(3,1)
assert run(g,{'a':av},{f't{z}':np.maximum(av+bv,0)},roundtrip=True)['fused']==1
for kind in ['relu','sigmoid','tanh','identity']:
    g=graph(); x=tensor(g,'x',[3,7]); y=node(g,[3,7],'activation',input=x,kind=kind); g['outputs']=[y]
    v=np.linspace(-12,12,21,dtype='f').reshape(3,7)
    ref={'relu':lambda:np.maximum(v,0),'sigmoid':lambda:1/(1+np.exp(-v)),'tanh':lambda:np.tanh(v),'identity':lambda:v}[kind]()
    run(g,{'x':v},{f't{y}':ref})
# Unary extensions preserve serialized graph identity and write non-workgroup tails.
for kind,fn in [('sqrt',np.sqrt),('exp',np.exp),('log',np.log),('abs',np.abs),('neg',np.negative),('reciprocal',np.reciprocal)]:
    value=np.linspace(.125,8,257,dtype='f')
    if kind in ['abs','neg']:value-=4
    g=graph();x=tensor(g,'x',[257]);y=node(g,[257],'activation',input=x,kind=kind);g['outputs']=[y]
    run(g,{'x':value},{f't{y}':fn(value)},roundtrip=True)
# Domain boundaries, infinities and NaNs must classify like the CPU references.
with np.errstate(all='ignore'):
 for kind,fn in [('sqrt',np.sqrt),('exp',np.exp),('log',np.log),('abs',np.abs),('neg',np.negative),('reciprocal',np.reciprocal)]:
    value=np.array([-np.inf,-100.,-1.,-0.,0.,1.,100.,np.inf,np.nan],dtype='f')
    g=graph();x=tensor(g,'x',[9]);y=node(g,[9],'activation',input=x,kind=kind);g['outputs']=[y]
    run(g,{'x':value},{f't{y}':fn(value)})
# Stable softmax along every axis, including non-contiguous reduction rows.
for axis in [0,1,2,-1]:
    g=graph(); x=tensor(g,'x',[2,3,5]); y=node(g,[2,3,5],'softmax',input=x,axis=axis); g['outputs']=[y]
    v=(rng.normal(size=[2,3,5])*20+1000).astype('f'); ex=np.exp(v-np.max(v,axis=axis,keepdims=True)); ref=ex/np.sum(ex,axis=axis,keepdims=True)
    run(g,{'x':v},{f't{y}':ref},rtol=5e-4)
# Subgroup rows include non-multiples of the SIMD width and a partially filled final workgroup.
for channels in [37,4096]:
    g=graph();x=tensor(g,'x',[3,channels]);y=node(g,[3,channels],'softmax',input=x,axis=-1);g['outputs']=[y]
    v=rng.normal(size=[3,channels]).astype('f');ex=np.exp(v-v.max(axis=1,keepdims=True));ref=ex/ex.sum(axis=1,keepdims=True)
    run(g,{'x':v},{f't{y}':ref},rtol=5e-4)
# Every transpose combination, rectangular tails and the tiled GEMM path.
for m,n,k in [(3,4,5),(65,70,17),(128,128,64)]:
    for ta,tb in [(False,False),(True,False),(False,True),(True,True)]:
        ash=[k,m] if ta else [m,k]; bsh=[n,k] if tb else [k,n]
        g=graph(); a=tensor(g,'a',ash); b=tensor(g,'b',bsh); y=node(g,[m,n],'mat_mul',a=a,b=b,trans_a=ta,trans_b=tb); g['outputs']=[y]
        av=rng.normal(size=ash).astype('f'); bv=rng.normal(size=bsh).astype('f')
        run(g,{'a':av,'b':bv},{f't{y}':(av.T if ta else av)@(bv.T if tb else bv)},atol=2e-4)

# Broadcast batches use strided GEMM when possible and distinct offsets for mixed axes.
for ap,bp in [([3],[3]),([3],[]),([],[3]),([2,1],[1,3]),([1,3],[2,1]),([2,1,1,1,1,1],[1,1,1,1,1,3])]:
 for ta,tb in [(False,False),(True,False),(False,True),(True,True)]:
    ash=ap+([5,3] if ta else [3,5]);bsh=bp+([4,5] if tb else [5,4])
    av=rng.normal(size=ash).astype('f');bv=rng.normal(size=bsh).astype('f')
    ref=np.matmul(av.swapaxes(-1,-2) if ta else av,bv.swapaxes(-1,-2) if tb else bv)
    g=graph();a=tensor(g,'a',ash);b=tensor(g,'b',bsh);y=node(g,ref.shape,'mat_mul',a=a,b=b,trans_a=ta,trans_b=tb);g['outputs']=[y]
    run(g,{'a':av,'b':bv},{f't{y}':ref},roundtrip=True)

# Dedicated reductions cover serial and subgroup paths, tails and strided axes.
for shape,axis in [([3,7],1),([3,37],1),([3,4096],1),([2,37,5],1),([2,3,5],0)]:
 for mean in [False,True]:
    g=graph();x=tensor(g,'x',shape);value=rng.normal(size=shape).astype('f')
    ref=(np.mean if mean else np.sum)(value,axis=axis)
    y=node(g,ref.shape,'reduce',input=x,axis=axis,mean=mean,keepdims=False);g['outputs']=[y]
    info=run(g,{'x':value},{f't{y}':ref},roundtrip=True,atol=2e-4,rtol=5e-4);assert info['workspace']==0

def conv_ref(x,w,b,stride,pad,dil,groups):
    N,C,H,W=x.shape; K,_,R,S_=w.shape
    P_=(H+2*pad[0]-dil[0]*(R-1)-1)//stride[0]+1
    Q_=(W+2*pad[1]-dil[1]*(S_-1)-1)//stride[1]+1
    out=np.zeros((N,K,P_,Q_),dtype='f')
    for n in range(N):
      for k in range(K):
       group=k//(K//groups)
       for p in range(P_):
        for q in range(Q_):
         acc=np.float32(0)
         for c in range(C//groups):
          for r in range(R):
           for s in range(S_):
            h=p*stride[0]-pad[0]+r*dil[0]; ww=q*stride[1]-pad[1]+s*dil[1]
            if 0<=h<H and 0<=ww<W: acc=np.float32(acc+x[n,group*(C//groups)+c,h,ww]*w[k,c,r,s])
         out[n,k,p,q]=acc+(b[k] if b is not None else 0)
    return out
for xsh,wsh,groups,stride,pad,dil,bias,relu in [([1,2,5,6],[4,1,3,2],2,[2,1],[1,0],[1,2],True,True),([2,3,4,5],[3,1,3,3],3,[1,1],[1,1],[1,1],False,True),([1,2,5,5],[3,2,3,3],1,[1,1],[0,0],[1,1],True,False)]:
    g=graph(); x=tensor(g,'x',xsh); wv=rng.normal(size=wsh).astype('f'); w=tensor(g,'w',wsh,wv)
    bv=rng.normal(size=[wsh[0]]).astype('f') if bias else None; b=tensor(g,'bias',[wsh[0]],bv) if bias else None
    xv=rng.normal(size=xsh).astype('f'); ref=conv_ref(xv,wv,bv,stride,pad,dil,groups)
    y=node(g,ref.shape,'conv2d',input=x,weights=w,bias=b,stride=stride,padding=pad,dilation=dil,groups=groups)
    if relu: y=node(g,ref.shape,'activation',input=y,kind='relu'); ref=np.maximum(ref,0)
    g['outputs']=[y]; info=run(g,{'x':xv},{f't{y}':ref},roundtrip=True)
    if relu: assert info['fused']==1
# Pooling, including padding accounting.
for kind in ['max','average']:
 for include in [False,True]:
    g=graph(); x=tensor(g,'x',[2,2,5,4]); y=node(g,[2,2,3,3],'pool2d',input=x,kind=kind,window=[3,2],stride=[2,2],padding=[1,1],include_padding=include); g['outputs']=[y]
    v=rng.normal(size=[2,2,5,4]).astype('f'); ref=np.empty([2,2,3,3],dtype='f')
    for n in range(2):
     for c in range(2):
      for p in range(3):
       for q in range(3):
        vals=[v[n,c,h,w] for h in range(p*2-1,p*2+2) for w in range(q*2-1,q*2+1) if 0<=h<5 and 0<=w<4]
        ref[n,c,p,q]=max(vals) if kind=='max' else sum(vals)/(6 if include else len(vals))
    run(g,{'x':v},{f't{y}':ref})
# Transpose, reshape, concat along a middle dimension, multiple outputs and branched liveness.
g=graph(); x=tensor(g,'x',[2,3,4]); t=node(g,[3,2,4],'transpose',input=x,permutation=[1,0,2]); r=node(g,[3,8],'reshape',input=t,shape=[3,8]); u=node(g,[3,8],'activation',input=r,kind='tanh'); v=node(g,[3,16],'concat',inputs=[r,u],axis=1); g['outputs']=[u,v]
xv=rng.normal(size=[2,3,4]).astype('f'); tr=xv.transpose(1,0,2).reshape(3,8)
run(g,{'x':xv},{f't{u}':np.tanh(tr),f't{v}':np.concatenate([tr,np.tanh(tr)],axis=1)},roundtrip=True)
# A long chain exercises slot reuse after the preceding value's last use.
g=graph(); x=tensor(g,'x',[257]); out=x; value=rng.normal(size=[257]).astype('f'); ref=value.copy()
for i in range(8): out=node(g,[257],'activation',input=out,kind='tanh'); ref=np.tanh(ref)
g['outputs']=[out]; info=run(g,{'x':value},{f't{out}':ref}); assert info['workspace']<=2*1280,info
# Scalar broadcasts, rank-8 metadata and constant-only graphs have distinct output bindings.
for ash,bsh in [([],[]),([2,3],[]),([1,1,2,1,1,1,1,3],[1,1,1,1,1,1,1,3])]:
    g=graph();av=np.asarray(rng.normal(size=ash),dtype='f');bv=np.asarray(rng.normal(size=bsh),dtype='f')
    a=tensor(g,'a',ash);b=tensor(g,'b',bsh,bv);ref=av+bv;y=node(g,ref.shape,'element_wise',a=a,b=b,kind='add');g['outputs']=[y]
    run(g,{'a':av},{f't{y}':ref},roundtrip=True)
g=graph();v=np.arange(7,dtype='f');y=tensor(g,'constant_output',[7],v);g['outputs']=[y];run(g,{}, {'constant_output':v},roundtrip=True)
# Missing, undersized, overlapping bindings and foreign serialization fail explicitly.
g=graph(); x=tensor(g,'x',[4]); out=node(g,[4],'activation',input=x,kind='identity'); g['outputs']=[out]
data=json.dumps(g).encode(); storage=C.create_string_buffer(data); e=ck(build(storage,len(data),1024)); c=ck(create(e)); p=alloc(np.zeros(4,dtype='f'))
try:
    assert enqueue(c,None)==-1
    assert bind(c,b'x',p,4)==-1
    assert bind(c,b'x',p,16)==0 and bind(c,f't{out}'.encode(),p,16)==0
    assert enqueue(c,None)==-1
finally:
    close(c); destroy(e); dc('cuMemFree_v2',allocations.pop())
assert not allocations
print(json.dumps({'passed_graphs':support.cases,'max_absolute_error':support.max_error,'backend':'MLIR -> AIR -> AGX','vendor_engines_used':False},indent=2))
