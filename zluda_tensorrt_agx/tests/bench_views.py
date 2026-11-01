#!/usr/bin/env python3
"""Isolate internal reshape-copy elimination; compare library versions via ZTRT_LIBRARY."""
from support import *
import statistics
results=[]
for width in [16,256,1024]:
    g=graph();x=tensor(g,'x',[width,width]);v=node(g,[width*width],'reshape',input=x,shape=[width*width]);y=node(g,[width*width],'activation',input=v,kind='tanh');g['outputs']=[y]
    data=json.dumps(g).encode();storage=C.create_string_buffer(data);e=ck(build(storage,len(data),1<<30));c=ck(create(e))
    values=np.linspace(-2,2,width*width,dtype='f');xp=alloc(values);yp=alloc(np.full(width*width,np.nan,dtype='f'))
    assert bind(c,b'x',xp,values.nbytes)==0 and bind(c,f't{y}'.encode(),yp,values.nbytes)==0
    for _ in range(5):assert enqueue(c,None)==0
    assert sync(c)==0;samples=[]
    for _ in range(30):
        start=time.perf_counter_ns()
        for _ in range(4):assert enqueue(c,None)==0
        assert sync(c)==0;samples.append((time.perf_counter_ns()-start)/4000)
    output=np.empty_like(values);dc('cuMemcpyDtoH_v2',P(output.ctypes.data),yp,S(output.nbytes))
    np.testing.assert_allclose(output,np.tanh(values),atol=3e-5,rtol=3e-4)
    results.append({'elements':width*width,'workspace_bytes':workspace(e),'median_enqueue_sync_us':statistics.median(samples)})
    close(c);destroy(e)
    while allocations:dc('cuMemFree_v2',allocations.pop())
print(json.dumps({'results':results},indent=2))
