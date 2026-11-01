#!/usr/bin/env python3
"""Compile/run an independent public-header client; never silently fall back to local subset headers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference-dir',required=True,type=Path)
p.add_argument('--manifest',required=True,type=Path)
p.add_argument('--abi-library',required=True,type=Path)
p.add_argument('--native-library',required=True,type=Path)
p.add_argument('--output-dir',required=True,type=Path)
p.add_argument('--type-include',action='append',default=[],type=Path,help='Optional separately reviewed CUDA type declarations, not a runtime library')
p.add_argument('--cxx',default='c++')
p.add_argument('--sanitize',action='store_true')
a=p.parse_args();crate=Path(__file__).resolve().parents[1];reference=a.reference_dir.resolve()
manifest=json.loads(a.manifest.read_text())
if manifest.get('target')!='TensorRT-RTX 1.7.1':p.error('manifest target must be TensorRT-RTX 1.7.1')
required={'NvInferRuntime.h','NvInferImpl.h','NvInferRuntimeBase.h','NvInferRuntimeCommon.h','NvInferPluginBase.h','NvInferLegacyDims.h','NvInferVersion.h'}
if not required<=set(manifest.get('headers',{})):p.error('manifest is missing required reference headers')
for name,record in manifest['headers'].items():
    if Path(name).name!=name or not name.endswith('.h'):p.error('invalid manifest header path')
    path=reference/name
    if not path.is_file():p.error(f'missing reference header: {name}')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=record['sha256']:p.error(f'reference hash mismatch: {name}')
    if b'SPDX-License-Identifier: Apache-2.0' not in data:p.error(f'unreviewed header license: {name}')
# No -I for our C++ subset. Copy only the independent native C declarations beside the client.
a.output_dir.mkdir(parents=True,exist_ok=True);out=a.output_dir.resolve()
source=(crate/'tests/test_reference_runtime.cpp').read_text().replace('#include "zluda_tensorrt.h"','#include "native_api.h"')
(out/'client.cpp').write_text(source);(out/'native_api.h').write_bytes((crate/'include/zluda_tensorrt.h').read_bytes())
flags=['-std=c++17','-Wall','-Wextra','-Werror','-I',str(reference)]
for directory in a.type_include:flags+=['-I',str(directory.resolve())]
if a.sanitize:flags+=['-fsanitize=address','-fno-omit-frame-pointer','-g']
client=out/'reference-client';native=a.native_library.resolve();abi=a.abi_library.resolve()
subprocess.run([a.cxx,*flags,str(out/'client.cpp'),str(native),'-Wl,-rpath,'+str(native.parent),'-ldl','-o',str(client)],check=True)
subprocess.run([str(client),str(abi)],check=True)
print('PASS: reference-header client (runtime subset exercised; no full ABI or GPU-context claim)')
