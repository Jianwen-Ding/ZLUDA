#!/usr/bin/env python3
"""Stream ordering, CUDA graph capture, context ownership and serialization-budget regression tests."""
from support import *

g=graph();x=tensor(g,'x',[2,3]);wv=np.arange(1,7,dtype='f').reshape(3,2);w=tensor(g,'w',[3,2],wv);b=tensor(g,'b',[2],np.array([1,-1],dtype='f'))
view=node(g,[2,3],'reshape',input=x,shape=[2,3]);y=node(g,[2,2],'mat_mul',a=view,b=w);y=node(g,[2,2],'element_wise',a=y,b=b,kind='add');y=node(g,[2,2],'activation',input=y,kind='relu');g['outputs']=[y]
data=json.dumps(g).encode();source=C.create_string_buffer(data)
assert not build(source,len(data),0)
e=ck(build(source,len(data),256));assert workspace(e)==256
size=serialize(e,None,0);payload=(C.c_ubyte*size)();assert serialize(e,payload,size)==size
assert decode_engine(bytes(payload))['workspace_limit']==256
r=ck(runtime())
assert not restore(r,payload,size,128)
restored=ck(restore(r,payload,size,512));destroy(restored);free_runtime(r)
c=ck(create(e))
xv=np.array([[1,-2,3],[4,5,-6]],dtype='f');xp=alloc(xv);yp=alloc(np.full((2,2),np.nan,dtype='f'))
assert bind(c,b'x',xp,xv.nbytes)==0 and bind(c,f't{y}'.encode(),yp,16)==0
assert enqueue(c,None)==0 and sync(c)==0,err()
# Rebinding invalidates the cached overlap check, including after a successful enqueue.
assert bind(c,f't{y}'.encode(),xp,16)==0
assert enqueue(c,None)==-1
assert bind(c,f't{y}'.encode(),yp,16)==0
assert enqueue(c,None)==0 and sync(c)==0,err()
# A view must refresh its root pointer after an input is rebound.
x2=np.ascontiguousarray(xv+1,dtype='f');xp2=alloc(x2)
assert bind(c,b'x',xp2,x2.nbytes)==0 and enqueue(c,None)==0 and sync(c)==0,err()
out=np.empty((2,2),dtype='f');dc('cuMemcpyDtoH_v2',P(out.ctypes.data),yp,S(out.nbytes))
np.testing.assert_allclose(out,np.maximum(x2@wv+np.array([1,-1]),0),atol=1e-5)
assert bind(c,b'x',xp,xv.nbytes)==0
stream=P();dc('cuStreamCreate',C.byref(stream),I(1))
graph_handle=P();exec_handle=P()
dc('cuStreamBeginCapture_v2',stream,I(0));assert enqueue(c,stream)==0,err()
dc('cuStreamEndCapture',stream,C.byref(graph_handle))
dc('cuGraphInstantiateWithFlags',C.byref(exec_handle),graph_handle,C.c_uint64(0))
for scale in [1.,2.]:
    v=np.ascontiguousarray(xv*scale,dtype='f');dc('cuMemcpyHtoD_v2',xp,P(v.ctypes.data),S(v.nbytes))
    dc('cuGraphLaunch',exec_handle,P());dc('cuCtxSynchronize')
    out=np.empty((2,2),dtype='f');dc('cuMemcpyDtoH_v2',P(out.ctypes.data),yp,S(out.nbytes))
    np.testing.assert_allclose(out,np.maximum(v@wv+np.array([1,-1]),0),atol=1e-5)
assert sync(c)==0,err()
dc('cuGraphExecDestroy',exec_handle);dc('cuGraphDestroy',graph_handle);dc('cuStreamDestroy_v2',stream)
# Changing streams must not let two enqueues overwrite the same activation workspace concurrently.
s1=P();s2=P();dc('cuStreamCreate',C.byref(s1),I(1));dc('cuStreamCreate',C.byref(s2),I(1))
assert enqueue(c,s1)==0 and enqueue(c,s2)==0 and sync(c)==0,err()
dc('cuStreamDestroy_v2',s1);dc('cuStreamDestroy_v2',s2)
# Independent workspaces share the engine's immutable device resources safely.
c2=ck(create(e));yp2=alloc(np.full((2,2),np.nan,dtype='f'))
assert bind(c2,b'x',xp,xv.nbytes)==0 and bind(c2,f't{y}'.encode(),yp2,16)==0
assert enqueue(c,None)==0 and enqueue(c2,None)==0 and sync(c)==0 and sync(c2)==0,err()
out1=np.empty((2,2),dtype='f');out2=np.empty_like(out1)
dc('cuMemcpyDtoH_v2',P(out1.ctypes.data),yp,S(out1.nbytes));dc('cuMemcpyDtoH_v2',P(out2.ctypes.data),yp2,S(out2.nbytes))
np.testing.assert_array_equal(out1,out2);close(c2)
# A caller may synchronize/destroy a stream without clearing the runtime's last-stream tracking.
s3=P();dc('cuStreamCreate',C.byref(s3),I(1));assert enqueue(c,s3)==0,err()
dc('cuStreamSynchronize',s3);dc('cuStreamDestroy_v2',s3)
assert enqueue(c,None)==0 and sync(c)==0,err()
# A foreign current CUDA context rejects enqueue. Destruction restores that foreign context after freeing ours.
other=P();dc('cuCtxCreate_v2',C.byref(other),I(0),I(0));assert enqueue(c,None)==-1
close(c);c=None
current=P();dc('cuCtxGetCurrent',C.byref(current));assert current.value==other.value
primary=ctx;dc('cuCtxSetCurrent',primary);dc('cuCtxDestroy_v2',other)
# A C-ABI execution context owns its plan even if the engine handle is released first.
c=ck(create(e));destroy(e);e=None
assert bind(c,b'x',xp,xv.nbytes)==0 and bind(c,f't{y}'.encode(),yp,16)==0
assert enqueue(c,None)==0 and sync(c)==0,err();close(c)
while allocations:dc('cuMemFree_v2',allocations.pop())
# Null object errors stay on the C boundary, without panics or device accesses.
assert api('ztrtGetNbIOTensors',I,P)(None)==-1
assert api('ztrtGetLayerCount',I,P)(None)==-1
assert serialize(None,None,0)==0
print('PASS: CUDA graph replay, cross-stream workspace ordering, context restoration, shared resources, engine lifetime, budget persistence and null handles')
