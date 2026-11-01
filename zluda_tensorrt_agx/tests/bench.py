#!/usr/bin/env python3
"""Small independent inference benchmarks; reports end-to-end enqueue+sync latency, not a TensorRT comparison."""
from support import *
import resource
import statistics
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--samples',type=int,default=12);parser.add_argument('--batch',type=int,default=4);args=parser.parse_args()
rng=np.random.default_rng(789)
cases=[]
g=graph();x=tensor(g,'x',[64,4096]);y=node(g,[64,4096],'softmax',input=x,axis=-1);g['outputs']=[y];cases.append(('softmax_64x4096',g))
g=graph();x=tensor(g,'x',[1024,1024]);b=tensor(g,'bias',[1024],rng.normal(size=[1024]));y=node(g,[1024,1024],'element_wise',a=x,b=b,kind='add');y=node(g,[1024,1024],'activation',input=y,kind='relu');g['outputs']=[y];cases.append(('bias_relu_1m',g))
g=graph();x=tensor(g,'x',[256,256]);w=tensor(g,'weights',[256,256],rng.normal(size=[256,256]));y=node(g,[256,256],'mat_mul',a=x,b=w);g['outputs']=[y];cases.append(('matmul_256',g))
g=graph();x=tensor(g,'x',[1,8,32,32]);w=tensor(g,'weights',[16,8,3,3],rng.normal(size=[16,8,3,3]));b=tensor(g,'bias',[16],rng.normal(size=[16]));y=node(g,[1,16,32,32],'conv2d',input=x,weights=w,bias=b,padding=[1,1]);y=node(g,[1,16,32,32],'activation',input=y,kind='relu');g['outputs']=[y];cases.append(('conv_8to16_32x32',g))
g=graph();x=tensor(g,'x',[16,16,16,16]);b=tensor(g,'b',[16,16,16,16]);y=node(g,[16,16,16,16],'element_wise',a=x,b=b,kind='add');g['outputs']=[y];cases.append(('equal_shape_add_64k',g))
g=graph();x=tensor(g,'x',[8,64,64]);w=tensor(g,'weights',[64,64],rng.normal(size=[64,64]));y=node(g,[8,64,64],'mat_mul',a=x,b=w);g['outputs']=[y];cases.append(('broadcast_matmul_8x64',g))
g=graph();x=tensor(g,'x',[64,4096]);w=tensor(g,'coefficients',[4096,1],np.ones((4096,1),dtype='f'));y=node(g,[64,1],'mat_mul',a=x,b=w);y=node(g,[64],'reshape',input=y,shape=[64]);g['outputs']=[y];cases.append(('lowered_reduce_sum_64x4096',g))
g=graph();x=tensor(g,'x',[64,4096]);y=node(g,[64],'reduce',input=x,axis=1,mean=False,keepdims=False);g['outputs']=[y];cases.append(('native_reduce_sum_64x4096',g))
g=graph();x=tensor(g,'x',[64,1024]);scale=tensor(g,'scale',[1024],np.ones(1024,dtype='f'));eps=tensor(g,'epsilon',[],np.array(.003,dtype='f'))
mean=node(g,[64,1],'reduce',input=x,axis=1,mean=True,keepdims=True);center=node(g,[64,1024],'element_wise',a=x,b=mean,kind='sub');sq=node(g,[64,1024],'element_wise',a=center,b=center,kind='mul')
var=node(g,[64,1],'reduce',input=sq,axis=1,mean=True,keepdims=True);var=node(g,[64,1],'element_wise',a=var,b=eps,kind='add');std=node(g,[64,1],'activation',input=var,kind='sqrt');inv=node(g,[64,1],'activation',input=std,kind='reciprocal')
y=node(g,[64,1024],'element_wise',a=center,b=inv,kind='mul');y=node(g,[64,1024],'element_wise',a=y,b=scale,kind='mul');g['outputs']=[y];cases.append(('layer_norm_64x1024',g))
g=graph();x=tensor(g,'x',[64,1024]);scale=tensor(g,'scale',[1024],np.ones(1024,dtype='f'));core=node(g,[64,1024],'layer_norm',input=x,axis=1,epsilon=.003);y=node(g,[64,1024],'element_wise',a=core,b=scale,kind='mul');g['outputs']=[y];cases.append(('fused_layer_norm_64x1024',g))
g=graph();x=tensor(g,'x',[1024,1024]);indices=list(range(1023,-1,-2));y=node(g,[1024,512],'gather',input=x,axis=1,indices=indices,index_shape=[512]);g['outputs']=[y];cases.append(('gather_columns_1024x512',g))
results=[]
for name,g in cases:
    data=json.dumps(g,separators=(',',':')).encode();source=C.create_string_buffer(data);start=time.perf_counter_ns();e=ck(build(source,len(data),1<<30));build_ms=(time.perf_counter_ns()-start)/1e6
    start=time.perf_counter_ns();c=ck(create(e));prepare_ms=(time.perf_counter_ns()-start)/1e6
    start=time.perf_counter_ns();shared_context=ck(create(e));shared_prepare_ms=(time.perf_counter_ns()-start)/1e6;close(shared_context)
    count=len(allocations);outs=[]
    try:
        for i in g['inputs']+g['outputs']:
            t=g['tensors'][i];value=rng.uniform(-1,1,size=t['shape']).astype('f') if i in g['inputs'] else np.full(t['shape'],np.nan,dtype='f')
            p=alloc(value);assert bind(c,t['name'].encode(),p,value.nbytes)==0,err()
            if i in g['outputs']:outs.append((p,value))
        start=time.perf_counter_ns();assert enqueue(c,None)==0 and sync(c)==0,err();cold_ms=(time.perf_counter_ns()-start)/1e6
        for _ in range(3):assert enqueue(c,None)==0,err()
        assert sync(c)==0,err();samples=[]
        for _ in range(args.samples):
            start=time.perf_counter_ns()
            for _ in range(args.batch):assert enqueue(c,None)==0,err()
            assert sync(c)==0,err();samples.append((time.perf_counter_ns()-start)/args.batch/1e3)
        for p,v in outs:
            dc('cuMemcpyDtoH_v2',P(v.ctypes.data),p,S(v.nbytes));assert np.isfinite(v).all(),name
        results.append({'case':name,'build_ms':build_ms,'prepare_ms':prepare_ms,'shared_prepare_ms':shared_prepare_ms,'first_enqueue_ms':cold_ms,'median_enqueue_sync_us':statistics.median(samples),'workspace_bytes':workspace(e),'constant_bytes':api('ztrtGetConstantBytes',S,P)(e),'fused_activations':fused(e)})
    finally:
        close(c);destroy(e)
        while len(allocations)>count:dc('cuMemFree_v2',allocations.pop())
print(json.dumps({'backend':'portable MLIR -> AIR -> AGX','samples':args.samples,'enqueues_per_sample':args.batch,'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'results':results},indent=2))
