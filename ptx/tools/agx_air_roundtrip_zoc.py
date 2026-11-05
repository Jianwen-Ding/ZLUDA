#!/usr/bin/env python3
"""zoc whose AGX code comes from the .air file on disk: `.agx -> .air -> .agx` with no state kept in memory.

zoc compiles its AGX object from the module in memory, so reading the file it wrote is never tested by zoc alone.
This stands in for zoc in the checks (--zoc): it runs zoc, rebuilds NAME.o and NAME.asm from NAME.portable.air with
llc, as zoc would have made them, and fails if the object differs. --target agx needs ZLUDA_AGX_VIA_AIR=1.

    ZLUDA_AGX_VIA_AIR=1 ZOC=target/release/zoc LLVM_BIN=$LLVM/bin python3 ptx/tools/agx_ptx_formalization_check.py \\
        --zoc ptx/tools/agx_air_roundtrip_zoc.py --agx-run .../agx_run PTX-formalization/tests/*.ptx
"""

import os
import shlex
import subprocess
import sys


def main():
    bin_dir, args = os.environ['LLVM_BIN'], sys.argv[1:]
    r = subprocess.run([os.environ['ZOC'], *args], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    value = lambda name: next((args[i + 1] for i, a in enumerate(args[:-1]) if a == name), None)
    arch, out = value('--arch'), value('--output-dir') or '.'
    stem = os.path.splitext(os.path.basename(args[-1]))[0]
    path = lambda ext: os.path.join(out, stem + ext)
    if r.returncode != 0 or not arch or not arch.startswith('agx') or not os.path.exists(path('.portable.air')):
        return r.returncode
    flags = ['-mtriple=agx', '-mcpu=' + arch, *shlex.split(os.environ.get('ZLUDA_AGX_LLVM_ARGS', ''))]
    if os.environ.get('ZLUDA_AGX_UNIFORM_CONSTANTS') != '0':
        flags.append('-agx-uniform-constants')
    if os.environ.get('ZLUDA_AGX_LAUNCH_DIVISORS') != '0':
        flags.append('-agx-launch-divisors')
    if '--strict-fp' in args and value('--target') != 'air':
        flags.append('-mattr=+ieee-fp32,+canonical-nan')
    source = path('.portable.air')
    if value('--target') == 'air':
        # (zoc --target air optimizes for AGX first, ZLUDA_AIR_AGX_PIPELINE)
        source = path('.agx.bc')
        subprocess.run([os.path.join(bin_dir, 'opt'), *flags, '-passes=default<O3>', path('.portable.air'), '-o', source],
                       check=True)
    llc = os.path.join(bin_dir, 'llc')
    subprocess.run([llc, *flags, '-filetype=obj', source, '-o', path('.rt.o')], check=True)
    subprocess.run([llc, *flags, source, '-o', path('.asm')], check=True)
    with open(path('.o'), 'rb') as a, open(path('.rt.o'), 'rb') as b:
        if a.read() != b.read():
            print('error: the object from ' + path('.portable.air') + ' differs from zoc\'s', file=sys.stderr)
            return 1
    os.replace(path('.rt.o'), path('.o'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
