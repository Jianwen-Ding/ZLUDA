#!/usr/bin/env python3
"""Compare native buffer/reader loading in fresh processes using only a generated ZTRT engine."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import tempfile
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',required=True,type=Path)
p.add_argument('--weights-mib',type=int,default=32)
p.add_argument('--prepare-context',action='store_true',help='Also prepare two GPU contexts; requires our ZLUDA_AGX_LIBCUDA driver')
p.add_argument('--repetitions',type=int,default=1)
p.add_argument('--worker',choices=['buffer','reader'])
p.add_argument('--engine',type=Path)
a=p.parse_args()
if not 1<=a.repetitions<=100:p.error('--repetitions must be between 1 and 100')
if a.worker and a.engine is None:p.error('--worker requires --engine')
if a.worker:
    lib=C.CDLL(str(a.library.resolve()))
    lib.ztrtCreateRuntime.restype=C.c_void_p
    lib.ztrtGetLastError.restype=C.c_char_p
    lib.ztrtDeserializeEngine.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_size_t]
    lib.ztrtDeserializeEngine.restype=C.c_void_p
    callback_type=C.CFUNCTYPE(C.c_ssize_t,C.c_void_p,C.c_void_p,C.c_size_t)
    lib.ztrtDeserializeEngineReader.argtypes=[C.c_void_p,C.c_void_p,callback_type,C.c_size_t]
    lib.ztrtDeserializeEngineReader.restype=C.c_void_p
    lib.ztrtDestroyEngine.argtypes=[C.c_void_p]
    lib.ztrtDestroyRuntime.argtypes=[C.c_void_p]
    runtime=lib.ztrtCreateRuntime();assert runtime
    start=time.perf_counter()
    if a.worker=='buffer':
        payload=bytearray(a.engine.stat().st_size)
        with a.engine.open('rb',buffering=0) as source:
            assert source.readinto(payload)==len(payload)
        pointer=(C.c_ubyte*len(payload)).from_buffer(payload)
        engine=lib.ztrtDeserializeEngine(runtime,pointer,len(payload),1<<30)
    else:
        with a.engine.open('rb',buffering=0) as source:
            @callback_type
            def read(_,destination,count):
                try:
                    return source.readinto((C.c_ubyte*count).from_address(destination))
                except OSError:
                    return -1
            engine=lib.ztrtDeserializeEngineReader(runtime,None,read,1<<30)
    assert engine,lib.ztrtGetLastError()
    seconds=time.perf_counter()-start
    preparation={}
    if a.prepare_context:
        driver=C.CDLL(os.environ['ZLUDA_AGX_LIBCUDA'])
        def call(name,*arguments):
            code=getattr(driver,name)(*arguments)
            assert code==0,(name,code)
        call('cuInit',C.c_uint(0));primary=C.c_void_p()
        call('cuDevicePrimaryCtxRetain',C.byref(primary),C.c_int(0));call('cuCtxSetCurrent',primary)
        lib.ztrtCreateExecutionContext.argtypes=[C.c_void_p];lib.ztrtCreateExecutionContext.restype=C.c_void_p
        lib.ztrtDestroyExecutionContext.argtypes=[C.c_void_p]
        contexts=[]
        for name in ['first_context_seconds','cached_context_seconds']:
            began=time.perf_counter();context=lib.ztrtCreateExecutionContext(engine);assert context,lib.ztrtGetLastError()
            preparation[name]=time.perf_counter()-began;contexts.append(context)
        for context in contexts:lib.ztrtDestroyExecutionContext(context)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    lib.ztrtDestroyEngine(engine);lib.ztrtDestroyRuntime(runtime)
    if a.prepare_context:call('cuDevicePrimaryCtxRelease_v2',C.c_int(0))
    print(json.dumps({'mode':a.worker,'seconds':seconds,'peak_rss_kib':peak,**preparation}))
else:
    if not 1<=a.weights_mib<=128:p.error('--weights-mib must be between 1 and 128')
    count=a.weights_mib*(1<<18)
    graph={'tensors':[{'name':'input','shape':[count]},{'name':'weights','shape':[count]},{'name':'output','shape':[count]}],
           'inputs':[0],'outputs':[2],'nodes':[{'output':2,'op':'element_wise','a':0,'b':1,'kind':'add'}]}
    metadata=json.dumps(graph,separators=(',',':')).encode()
    with tempfile.TemporaryDirectory(prefix='ztrt-load-') as directory:
        path=Path(directory)/'generated.ztrt'
        with path.open('wb') as output:
            output.write(b'ZTRTAGX2'+struct.pack('<QQ',8+len(metadata)+12+count*4,len(metadata))+metadata+struct.pack('<IQ',1,count))
            chunk=struct.pack('<f',1.25)*(1<<18)
            for _ in range(a.weights_mib):output.write(chunk)
        results=[]
        for _ in range(a.repetitions):
            for mode in ['buffer','reader']:
                command=[sys.executable,__file__,'--library',str(a.library),'--worker',mode,'--engine',str(path)]
                if a.prepare_context:command.append('--prepare-context')
                result=subprocess.check_output(command,text=True)
                results.append(json.loads(result))
        print(json.dumps({'generated_weights_mib':a.weights_mib,'results':results},indent=2))
