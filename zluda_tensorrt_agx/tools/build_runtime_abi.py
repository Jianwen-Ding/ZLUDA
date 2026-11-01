#!/usr/bin/env python3
"""Build/test the isolated runtime configuration ABI prototype; no NVIDIA SDK/runtime is used."""
import argparse
from pathlib import Path
import subprocess
import shutil

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--cxx',default='c++')
p.add_argument('--sanitize',action='store_true',help='Instrument adapter and probe with AddressSanitizer')
a=p.parse_args();crate=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
flags=['-std=c++17','-Wall','-Wextra','-Werror','-I',str(crate/'include')]
if a.sanitize:flags+=['-fsanitize=address','-fno-omit-frame-pointer','-g']
library=out/'libtensorrt_agx_runtime_abi.so';probe=out/'ztrt-runtime-abi-probe'
subprocess.run([a.cxx,*flags,'-fPIC','-fvisibility=hidden','-shared',str(crate/'abi/runtime.cpp'),'-o',str(library)],check=True)
subprocess.run([a.cxx,*flags,str(crate/'tests/test_runtime_abi.cpp'),'-ldl','-o',str(probe)],check=True)
subprocess.run([str(probe),str(library)],check=True)
shutil.copy2(crate.parent/'LICENSE-APACHE',out/'LICENSE-APACHE')
shutil.copy2(crate/'abi/README.md',out/'ABI_README.md')
print(f'Built isolated configuration-only ABI prototype: {library}')
