#!/usr/bin/env python3
"""Public-schema ONNX fixtures, independently encoded without any vendor library.
If the optional open-source onnx package is installed, also validate each fixture with its checker.
"""
from support import *
import support
import struct
try:
    import onnx
except ImportError:
    onnx=None
build_onnx=api('ztrtBuildOnnx',P,P,S,S)

def vi(v):
    v=int(v)&((1<<64)-1);b=bytearray()
    while v>=128:b.append((v&127)|128);v>>=7
    b.append(v);return bytes(b)
def var(i,v):return vi(i<<3)+vi(v)
def blob(i,v):return vi((i<<3)|2)+vi(len(v))+v
def text(i,v):return blob(i,v.encode())
def info(name,shape):
    dims=b''.join(blob(1,var(1,d)) for d in shape)
    ty=var(1,1)+blob(2,dims)
    return text(1,name)+blob(2,blob(1,ty))
def constant(name,value,dtype=1):
    value=np.asarray(value,dtype=np.float32 if dtype==1 else np.int64)
    return blob(1,b''.join(vi(d)for d in value.shape))+var(2,dtype)+text(8,name)+blob(9,value.tobytes())
def attr(name,value):
    if isinstance(value,float):return text(1,name)+var(20,1)+vi((2<<3)|5)+struct.pack('<f',value)
    if isinstance(value,int):return text(1,name)+var(20,2)+var(3,value)
    if isinstance(value,str):return text(1,name)+var(20,3)+text(4,value)
    return text(1,name)+var(20,7)+blob(8,b''.join(vi(v) for v in value))
def op(kind,inputs,output,**attrs):
    return b''.join(text(1,n)for n in inputs)+text(2,output)+text(4,kind)+b''.join(blob(5,attr(k,v)) for k,v in attrs.items())
def model(inputs,outputs,nodes,initializers=(),opset=17):
    graph=b''.join(blob(1,n)for n in nodes)+b''.join(blob(5,t)for t in initializers)+b''.join(blob(11,info(n,s))for n,s in inputs)+b''.join(blob(12,info(n,s))for n,s in outputs)
    return var(1,8)+blob(7,graph)+blob(8,var(2,opset))
def execute(data,values,expected):
    if onnx is not None:onnx.checker.check_model(onnx.load_model_from_string(data))
    source=C.create_string_buffer(data);e=ck(build_onnx(source,len(data),1<<30))
    size=serialize(e,None,0);native=(C.c_ubyte*size)();assert serialize(e,native,size)==size
    g=decode_engine(bytes(native))
    run(g,values,expected,engine_override=e)

rng=np.random.default_rng(4321)
x=rng.normal(size=(2,3)).astype('f');w=rng.normal(size=(3,4)).astype('f');b=rng.normal(size=(4,)).astype('f')
data=model([('x',[2,3])],[('y',[2,4])],[op('MatMul',['x','w'],'mm'),op('Add',['mm','b'],'sum'),op('Relu',['sum'],'y')],[constant('w',w),constant('b',b)])
execute(data,{'x':x},{'y':np.maximum(x@w+b,0)})
if os.environ.get('ZTRT_ONNX_FIXTURE'):
    from pathlib import Path
    Path(os.environ['ZTRT_ONNX_FIXTURE']).write_bytes(data)
# ONNX MatMul batch broadcasting preserves leading batch dimensions.
for ash,bsh in [((3,4,5),(3,5,7)),((2,1,4,5),(1,3,5,7)),((2,3,4,5),(5,7)),((5,),(5,)),((5,),(5,7)),((3,5),(5,)),((5,),(2,3,5,7)),((2,3,4,5),(5,))]:
    a=rng.normal(size=ash).astype('f');b=rng.normal(size=bsh).astype('f');ref=a@b
    execute(model([('a',ash),('b',bsh)],[('y',ref.shape)],[op('MatMul',['a','b'],'y')]),{'a':a,'b':b},{'y':ref})
# Gemm transposition, scaling, broadcast bias.
x=rng.normal(size=(3,2)).astype('f');w=rng.normal(size=(4,3)).astype('f');b=rng.normal(size=(4,)).astype('f')
data=model([('x',[3,2])],[('y',[2,4])],[op('Gemm',['x','w','b'],'y',transA=1,transB=1,alpha=0.5,beta=0.25)],[constant('w',w),constant('b',b)])
execute(data,{'x':x},{'y':.5*(x.T@w.T)+.25*b})
# A convolutional image network through pooling and reshape with a constant i64 shape tensor.
x=np.arange(16,dtype='f').reshape(1,1,4,4)-6;w=np.ones((1,1,2,2),dtype='f');b=np.array([.5],dtype='f')
c=np.array([x[0,0,i:i+2,j:j+2].sum()+.5 for i in range(3)for j in range(3)],dtype='f').reshape(1,1,3,3)
r=np.maximum(c,0);pooled=np.array([r[0,0,i:i+2,j:j+2].mean()for i in range(2)for j in range(2)],dtype='f').reshape(1,4)
data=model([('x',[1,1,4,4])],[('y',[1,4])],[op('Conv',['x','w','b'],'c'),op('Relu',['c'],'r'),op('AveragePool',['r'],'p',kernel_shape=[2,2]),op('Reshape',['p','shape'],'y')],[constant('w',w),constant('b',b),constant('shape',np.array([1,-1]),7)])
execute(data,{'x':x},{'y':pooled})
# Complete image classifier: convolution -> ReLU -> pooling -> reshape -> dense -> softmax.
dense=rng.normal(size=(4,5)).astype('f');dense_bias=rng.normal(size=(5,)).astype('f')
logits=pooled@dense+dense_bias;probs=np.exp(logits-logits.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True)
classifier=model([('x',[1,1,4,4])],[('probabilities',[1,5])],[op('Conv',['x','w','b'],'c'),op('Relu',['c'],'r'),op('AveragePool',['r'],'p',kernel_shape=[2,2]),op('Reshape',['p','shape'],'flat'),op('Gemm',['flat','dense','dense_bias'],'logits'),op('Softmax',['logits'],'probabilities',axis=-1)],[constant('w',w),constant('b',b),constant('shape',np.array([1,-1]),7),constant('dense',dense),constant('dense_bias',dense_bias)])
execute(classifier,{'x':x},{'probabilities':probs})
# Every activation, elementwise and softmax path with multidimensional broadcast.
for kind in ['Sigmoid','Tanh','Identity']:
    x=rng.normal(size=(2,5)).astype('f')
    ref={'Sigmoid':lambda:1/(1+np.exp(-x)),'Tanh':lambda:np.tanh(x),'Identity':lambda:x}[kind]()
    execute(model([('x',[2,5])],[('y',[2,5])],[op(kind,['x'],'y')]),{'x':x},{'y':ref})
for kind in ['Add','Sub','Mul','Div','Max','Min']:
    x=rng.normal(size=(2,3,4)).astype('f');b=rng.uniform(.5,2,size=(3,1)).astype('f')
    ref={'Add':lambda:x+b,'Sub':lambda:x-b,'Mul':lambda:x*b,'Div':lambda:x/b,'Max':lambda:np.maximum(x,b),'Min':lambda:np.minimum(x,b)}[kind]()
    execute(model([('x',[2,3,4])],[('y',[2,3,4])],[op(kind,['x','b'],'y')],[constant('b',b)]),{'x':x},{'y':ref})
x=rng.normal(size=(2,3,4)).astype('f');e=np.exp(x-x.max(axis=1,keepdims=True));ref=e/e.sum(axis=1,keepdims=True)
execute(model([('x',[2,3,4])],[('y',[2,3,4])],[op('Softmax',['x'],'y',axis=1)]),{'x':x},{'y':ref})
x=rng.normal(size=(2,3,4)).astype('f');tr=x.transpose(1,0,2).reshape(3,8)
execute(model([('x',[2,3,4])],[('y',[3,16])],[op('Transpose',['x'],'t',perm=[1,0,2]),op('Flatten',['t'],'f',axis=1),op('Concat',['f','f'],'y',axis=1)]),{'x':x},{'y':np.concatenate([tr,tr],axis=1)})
x=rng.normal(size=(2,3,4,5)).astype('f')
execute(model([('x',[2,3,4,5])],[('y',[2,3,1,1])],[op('GlobalAveragePool',['x'],'y')]),{'x':x},{'y':x.mean(axis=(2,3),keepdims=True)})
# Constant-only models use our normal output binding path.
value=constant('',np.array([1.,2.,3.],dtype='f'))
constant_node=text(2,'y')+text(4,'Constant')+blob(5,text(1,'value')+var(20,4)+blob(5,value))
execute(model([],[('y',[3])],[constant_node]),{}, {'y':np.array([1.,2.,3.],dtype='f')})
# Inference BatchNormalization across ranks, with non-default epsilon and fused ReLU.
for sh in [(2,3),(2,3,7),(2,3,4,5),(1,3,1,2,1)]:
    x=rng.normal(size=sh).astype('f');scale=rng.normal(size=3).astype('f');bias=rng.normal(size=3).astype('f')
    mean=rng.normal(size=3).astype('f');variance=rng.uniform(.01,2,size=3).astype('f');eps=.002
    broad=[1,3]+[1]*(len(sh)-2)
    ref=np.maximum((x-mean.reshape(broad))/np.sqrt(variance.reshape(broad)+eps)*scale.reshape(broad)+bias.reshape(broad),0)
    data=model([('x',sh)],[('y',sh)],[op('BatchNormalization',['x','s','b','m','v'],'bn',epsilon=eps),op('Relu',['bn'],'y')],[constant('s',scale),constant('b',bias),constant('m',mean),constant('v',variance)])
    execute(data,{'x':x},{'y':ref})
for training,var_values in [(1,np.ones(3,dtype='f')),(0,-np.ones(3,dtype='f'))]:
    data=model([('x',[2,3])],[('y',[2,3])],[op('BatchNormalization',['x','s','b','m','v'],'y',training_mode=training)],[constant('s',np.ones(3)),constant('b',np.zeros(3)),constant('m',np.zeros(3)),constant('v',var_values)])
    buf=C.create_string_buffer(data);assert not build_onnx(buf,len(data),1<<30)
# Versioned reductions: axis attributes before Mean-18, constant axis inputs thereafter.
for kind,version in [('ReduceSum',13),('ReduceSum',18),('ReduceMean',13),('ReduceMean',18)]:
 for axes in [[0],[1],[-1],[0,2],[]]:
  for keep in [0,1]:
    x=rng.normal(size=(2,3,5)).astype('f');modern=kind=='ReduceSum' or version>=18
    nodes=[op(kind,['x','axes'] if modern else ['x'],'y',keepdims=keep,**({} if modern else {'axes':axes}))]
    init=[constant('axes',np.asarray(axes,dtype='int64'),7)] if modern else []
    ref=(np.sum if kind=='ReduceSum' else np.mean)(x,axis=tuple(axes) if axes else None,keepdims=bool(keep))
    execute(model([('x',x.shape)],[('y',ref.shape)],nodes,init,opset=version),{'x':x},{'y':ref})
for kind in ['ReduceSum','ReduceMean']:
    x=rng.normal(size=(2,3)).astype('f')
    execute(model([('x',x.shape)],[('y',x.shape)],[op(kind,['x'],'y',noop_with_empty_axes=1)],opset=18),{'x':x},{'y':x})
    x=np.asarray(3.25,dtype='f');execute(model([('x',[])],[('y',[])],[op(kind,['x'],'y')],opset=18),{'x':x},{'y':x})
# Reject duplicate/out-of-range axes, invalid flags, dynamic axes and wrong-version attributes.
for axes,flags in [([1,1],{}),([3],{}),([1],{'keepdims':2}),([1],{'noop_with_empty_axes':2})]:
    data=model([('x',[2,3])],[('y',[2,1])],[op('ReduceSum',['x','axes'],'y',**flags)],[constant('axes',np.asarray(axes),7)],opset=18)
    buf=C.create_string_buffer(data);assert not build_onnx(buf,len(data),1<<30)
data=model([('x',[2,3])],[('y',[2,1])],[op('ReduceMean',['x'],'y',axes=[1])],opset=18)
buf=C.create_string_buffer(data);assert not build_onnx(buf,len(data),1<<30)
# Malformed metadata and unsupported operators/attributes are explicit failures.
for bad in [b'\0',b'\x80'*11,b'\x3a\xff\xff\xff\xff\x7f',model([('x',[2,3])],[('y',[2,3])],[op('Relu',['x'],'y',unknown=1)]),model([('x',[2,3])],[('y',[2,3])],[op('Erf',['x'],'y')]),model([('x',[2,3])],[('y',[2,3])],[op('Relu',['x'],'y')],opset=12)]:
    buf=C.create_string_buffer(bad);assert not build_onnx(buf,len(bad),1<<30),bad
print(json.dumps({'onnx_graphs_passed':support.cases,'max_absolute_error':support.max_error,'official_onnx_checker':onnx is not None},indent=2))
