#!/usr/bin/env python3
"""Numerical tests of our graph runtime and kernels. No vendor runtime or engine is used.
Run with Python + NumPy, ZTRT_LIBRARY and ZLUDA_AGX_LIBCUDA pointing to our builds.
"""
import ctypes as C
import json
import os
import time
import numpy as np

P=C.c_void_p; I=C.c_int32; S=C.c_size_t; Q=C.c_int64
lib=C.CDLL(os.environ['ZTRT_LIBRARY'])
cuda=C.CDLL(os.environ['ZLUDA_AGX_LIBCUDA'])
def api(name, ret, *args):
    fn=getattr(lib,name); fn.restype=ret; fn.argtypes=list(args); return fn
err=api('ztrtGetLastError',C.c_char_p)
build=api('ztrtBuildJson',P,P,S,S)
destroy=api('ztrtDestroyEngine',None,P)
create=api('ztrtCreateExecutionContext',P,P)
close=api('ztrtDestroyExecutionContext',None,P)
bind=api('ztrtSetTensorAddress',I,P,C.c_char_p,P,S)
enqueue=api('ztrtEnqueueV3',I,P,P)
sync=api('ztrtSynchronize',I,P)
serialize=api('ztrtSerializeEngine',S,P,P,S)
runtime=api('ztrtCreateRuntime',P)
restore=api('ztrtDeserializeEngine',P,P,P,S,S)
free_runtime=api('ztrtDestroyRuntime',None,P)
workspace=api('ztrtGetWorkspaceSize',S,P)
fused=api('ztrtGetFusedActivationCount',S,P)

def ck(v):
    assert v, (err() or b'no runtime error').decode(); return v

def dc(name, *args):
    fn=getattr(cuda,name); fn.restype=C.c_int
    rc=fn(*args)
    assert rc==0,(name,rc)

dc('cuInit',I(0))
ctx=P(); dc('cuDevicePrimaryCtxRetain',C.byref(ctx),I(0)); dc('cuCtxSetCurrent',ctx)
allocations=[]
def alloc(a):
    a=np.ascontiguousarray(a,dtype=np.float32)
    p=P(); dc('cuMemAlloc_v2',C.byref(p),S(a.nbytes)); allocations.append(p)
    dc('cuMemcpyHtoD_v2',p,P(a.ctypes.data),S(a.nbytes)); return p

def graph(): return {'tensors':[],'inputs':[],'outputs':[],'nodes':[]}
def tensor(g,name,shape,data=None):
    i=len(g['tensors']); t={'name':name,'shape':list(shape)}
    if data is not None: t['data']=np.asarray(data,dtype=np.float32).reshape(-1).tolist()
    else: g['inputs'].append(i)
    g['tensors'].append(t); return i

def node(g,output_shape,op,**kwargs):
    i=len(g['tensors']); g['tensors'].append({'name':f't{i}','shape':list(output_shape)})
    g['nodes'].append({'output':i,'op':op,**kwargs}); return i

cases=0
max_error=0.0

def run(g,values,expected,rtol=3e-4,atol=3e-5,roundtrip=False,repeats=2,engine_override=None):
    global cases,max_error
    data=json.dumps(g,separators=(',',':')).encode(); storage=C.create_string_buffer(data)
    e=ck(build(storage,len(data),1<<30)) if engine_override is None else engine_override; c=None; count=len(allocations)
    try:
        if roundtrip:
            size=serialize(e,None,0); ck(size)
            blob=(C.c_ubyte*size)(); assert serialize(e,blob,size)==size
            r=ck(runtime())
            try:
                assert not restore(r,C.c_char_p(b'not our format'),14,1<<30)
                other=ck(restore(r,blob,size,1<<30))
            finally: free_runtime(r)
            destroy(e); e=other
        c=ck(create(e))
        # Every result buffer starts as NaNs to expose missing/tail writes.
        pointers={}
        for i in g['inputs']:
            name=g['tensors'][i]['name']; value=np.asarray(values[name],dtype=np.float32)
            assert list(value.shape)==g['tensors'][i]['shape']
            p=alloc(value); pointers[name]=p
            assert bind(c,name.encode(),p,value.nbytes)==0,(err(),name)
        for i in g['outputs']:
            name=g['tensors'][i]['name']; shape=g['tensors'][i]['shape']
            output=np.full(shape,np.nan,dtype=np.float32); p=alloc(output); pointers[name]=p
            assert bind(c,name.encode(),p,output.nbytes)==0,err()
        for _ in range(repeats):
            assert enqueue(c,None)==0,err()
        assert sync(c)==0,err()
        for i in g['outputs']:
            name=g['tensors'][i]['name']; shape=g['tensors'][i]['shape']
            output=np.empty(shape,dtype=np.float32)
            dc('cuMemcpyDtoH_v2',P(output.ctypes.data),pointers[name],S(output.nbytes))
            reference=np.asarray(expected[name],dtype=np.float32)
            np.testing.assert_allclose(output,reference,rtol=rtol,atol=atol,err_msg=name)
            max_error=max(max_error,float(np.max(np.abs(output-reference))))
        cases+=1
        return {'workspace':workspace(e),'fused':fused(e)}
    finally:
        if c: close(c)
        if e: destroy(e)
        while len(allocations)>count: dc('cuMemFree_v2',allocations.pop())


def decode_engine(data):
    import struct
    if data[:8] == b'ZTRTAGX1': return json.loads(data[16:])
    assert data[:8] == b'ZTRTAGX2'
    size=struct.unpack_from('<Q',data,16)[0];g=json.loads(data[24:24+size]);at=24+size
    while at<len(data):
        i,n=struct.unpack_from('<IQ',data,at);at+=12
        g['tensors'][i]['data']=list(struct.unpack_from('<'+'f'*n,data,at));at+=n*4
    assert at==len(data)
    return g
