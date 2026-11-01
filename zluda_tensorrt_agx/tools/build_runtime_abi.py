#!/usr/bin/env python3
"""Build/test the isolated runtime/engine/context ABI prototype; no NVIDIA SDK/runtime is used."""
import argparse
from pathlib import Path
import subprocess
import shutil
import os

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--cxx',default='c++')
p.add_argument('--native-library',type=Path,default=Path(os.environ.get('ZTRT_LIBRARY',str(Path(__file__).resolve().parents[2]/'target/release/libtensorrt_agx.so'))),help='Our built native inference library')
p.add_argument('--gpu',action='store_true',help='Also run the context-dispatch GPU probe using ZLUDA_AGX_LIBCUDA')
p.add_argument('--sanitize',action='store_true',help='Instrument adapter and probe with AddressSanitizer')
a=p.parse_args();crate=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
native=a.native_library.resolve()
if not native.is_file():p.error(f'missing native inference library: {native}; build it or pass --native-library')
flags=['-std=c++17','-Wall','-Wextra','-Werror','-I',str(crate/'include')]
if a.sanitize:flags+=['-fsanitize=address','-fno-omit-frame-pointer','-g']
library=out/'libtensorrt_agx_runtime_abi.so';probe=out/'ztrt-runtime-abi-probe'
subprocess.run([a.cxx,*flags,'-fPIC','-fvisibility=hidden','-shared',str(crate/'abi/runtime.cpp'),str(crate/'abi/engine.cpp'),str(crate/'abi/context.cpp'),str(native),'-Wl,-rpath,'+str(native.parent),'-o',str(library)],check=True)
subprocess.run([a.cxx,*flags,str(crate/'tests/test_runtime_abi.cpp'),'-ldl','-o',str(probe)],check=True)
subprocess.run([str(probe),str(library)],check=True)
engine_probe=out/'ztrt-engine-abi-probe'
subprocess.run([a.cxx,*flags,str(crate/'tests/test_engine_abi.cpp'),str(native),'-Wl,-rpath,'+str(native.parent),'-ldl','-o',str(engine_probe)],check=True)
subprocess.run([str(engine_probe),str(library)],check=True)
if a.gpu:
    if not os.environ.get('ZLUDA_AGX_LIBCUDA'):p.error('--gpu requires ZLUDA_AGX_LIBCUDA pointing to our driver')
    context_probe=out/'ztrt-context-abi-probe'
    subprocess.run([a.cxx,*flags,str(crate/'tests/test_context_abi.cpp'),str(native),'-Wl,-rpath,'+str(native.parent),'-ldl','-o',str(context_probe)],check=True)
    subprocess.run([str(context_probe),str(library)],check=True)
shutil.copy2(crate.parent/'LICENSE-APACHE',out/'LICENSE-APACHE')
shutil.copy2(crate/'abi/README.md',out/'ABI_README.md')
print(f'Built isolated runtime/engine/context ABI prototype: {library}')
