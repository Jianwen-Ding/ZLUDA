#!/usr/bin/env python3
"""Emit portable AIR and/or AGX objects from our portable inference MLIR. Uses open-source LLVM tools only."""
import argparse
from pathlib import Path
import subprocess
import tempfile
p=argparse.ArgumentParser()
p.add_argument('--llvm-build',type=Path,required=True)
p.add_argument('--llvm-source',type=Path)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--emit',choices=['air','agx','both'],default='both')
p.add_argument('--chips',default='agx-g14c')
p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]/'agx_libs/kernels/inference.mlir')
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
bin=a.llvm_build/'bin'
def run(*args):subprocess.run([str(v)for v in args],check=True)
if a.emit in ['air','both']:
    with tempfile.TemporaryDirectory()as d:
        d=Path(d);lowered=d/'lowered.mlir'
        run(bin/'mlir-opt',a.source,'--pass-pipeline=builtin.module(gpu.module(arith-emulate-f64,convert-gpu-to-air{use-bare-ptr-memref-call-conv=1 index-bitwidth=32}),reconcile-unrealized-casts)','-o',lowered)
        lines=lowered.read_text().splitlines();start=next(i for i,l in enumerate(lines)if l.lstrip().startswith('gpu.module'))
        depth=lines[start].count('{')-lines[start].count('}');end=start
        while depth:end+=1;depth+=lines[end].count('{')-lines[end].count('}')
        body=d/'body.mlir';body.write_text('module {\n'+'\n'.join(lines[start+1:end])+'\n}\n')
        ir=d/'input.ll';run(bin/'mlir-translate','--mlir-to-llvmir',body,'-o',ir)
        air=a.output_dir/'inference.air';run(bin/'opt','-mtriple=air','-passes=air-portable',ir,'-o',air)
        run(bin/'opt','-S',air,'-o',a.output_dir/'inference.air.ll')
if a.emit in ['agx','both']:
    source=a.llvm_source
    if source is None:
        text=(a.llvm_build/'CMakeCache.txt').read_text()
        root=next(l.split('=',1)[1]for l in text.splitlines()if l.startswith('CMAKE_HOME_DIRECTORY:INTERNAL='))
        source=Path(root).parent
    script=source/'mlir/utils/agx-kernels/agx-build-fatbin.py'
    run('python3',script,'--mlir-opt',bin/'mlir-opt','--mlir-translate',bin/'mlir-translate','--llc',bin/'llc',
        '--chips',a.chips,'--index-bitwidth=32','--via-air',a.source,'-o',a.output_dir/'inference.agxfat')
print('emitted',a.emit,'to',a.output_dir)
