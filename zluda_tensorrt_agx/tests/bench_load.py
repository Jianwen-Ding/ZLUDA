#!/usr/bin/env python3
"""Compare native buffer/reader loading in fresh processes using only a generated ZTRT engine."""
import argparse
import ctypes as C
import json
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
p.add_argument('--worker',choices=['buffer','reader'])
p.add_argument('--engine',type=Path)
a=p.parse_args()
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
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    lib.ztrtDestroyEngine(engine);lib.ztrtDestroyRuntime(runtime)
    print(json.dumps({'mode':a.worker,'seconds':seconds,'peak_rss_kib':peak}))
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
        for mode in ['buffer','reader']:
            result=subprocess.check_output([sys.executable,__file__,'--library',str(a.library),'--worker',mode,'--engine',str(path)],text=True)
            results.append(json.loads(result))
        print(json.dumps({'generated_weights_mib':a.weights_mib,'results':results},indent=2))
