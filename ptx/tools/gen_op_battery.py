#!/usr/bin/env python3
"""PTX programs for checking the routes against each other: one operation on ordinary and edge values.

The PTX-formalization tests often exercise math on special values only, which is how a wrong sin (scaled by pi / 2 on
one route, right on 0, infinities and NaN) went unseen. Each program here stores one operation's results for 32
threads, with no expected values: air_agx_diff.py compares the routes.

    python3 ptx/tools/gen_op_battery.py OUT_DIR
    python3 ptx/tools/air_agx_diff.py --tools ... --direct OUT_DIR/*.ptx
"""
import math
import os
import random
import struct
import sys

random.seed(1234)
F32 = [0.5, 1.0, 1.5, 3.0, -0.75, 2.5, 100.25, -33.5, 0.1, 7.0, 1e-3, -1e3, 3.14159, 0.7071, 12345.678, -0.001,
       1e10, -1e-10, 1.0e-30, 42.0, 0.25, 8.0, 16.5, -2.0, 0.0, -0.0, 1.1754944e-38, 3.0e38, float('inf'),
       -float('inf'), float('nan'), 9.5]
INTS = [0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 100, 255, 256, 1000, 65535, 65536, 0x7fffffff, 0x80000000, 0xffffffff,
        0xfffffffe, 0x12345678, 0xdeadbeef, 0x55555555, 0xaaaaaaaa, 0x80000001, 0x00ff00ff, 0xff00ff00, 12345, 54321,
        0x7ffffffe, 0x40000000, 0xc0000000]
BITS = {'8': 8, '16': 16, '32': 32, '64': 64}
HEAD = ".version 8.0\n.target sm_90\n.address_size 64\n"


def f32bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def f64bits(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def values(width, kind, n):
    """n values of the type as unsigned integers of the width."""
    out = []
    for i in range(32):
        if kind == 'f' and width == 32:
            out.append(f32bits(F32[(i * n) % 32]))
        elif kind == 'f' and width == 64:
            out.append(f64bits(F32[(i * n) % 32] * 1.000000001))
        elif kind == 'f' and width == 16:
            x = F32[(i * n) % 32]
            try:
                out.append(struct.unpack('<H', struct.pack('<e', x))[0])
            except OverflowError:
                out.append(0x7c00 if x > 0 else 0xfc00)
        else:
            v = INTS[(i * n) % 32]
            if width == 64:
                v = (v << 32) | INTS[(i * n + 11) % 32]
            out.append(v & ((1 << width) - 1))
    return out


def program(name, ops, srcs, dst, setup=''):
    """ops: PTX lines over %a, %b, %c (of types srcs) leaving the result in %d (type dst)."""
    regs = ''
    loads = ''
    bufs = ''
    args = []
    params = []
    for i, (reg, ty) in enumerate(zip('abc', srcs)):
        width = int(ty[1:])
        regtype = 'b' + ty[1:] if ty[0] != 'f' or ty == 'f16' else ty
        regs += f'    .reg .{regtype} %{reg};\n'
        size = max(width // 8, 1)
        params.append(f'.param .u64 p{i}')
        loads += (f'    mul.wide.u32 %o, %t, {size}; ld.param.u64 %q, [p{i}]; add.u64 %q, %q, %o; '
                  f'ld.global.{regtype} %{reg}, [%q];\n')
        vals = values(width, ty[0], 3 * i + 1)
        bufs += f'.buffer I{i} .u{width} = {{ {", ".join(str(v) for v in vals)} }};\n'
        args.append(f'I{i}')
    width = int(dst[1:])
    size = width // 8
    rd = 'b' + dst[1:] if dst[0] != 'f' or dst == 'f16' else dst
    regs += f'    .reg .{rd} %d;\n'
    params.append('.param .u64 po')
    return (HEAD + f'.visible .entry k({", ".join(params)})\n{{\n    .reg .b32 %t, %n; .reg .b64 %o, %q; .reg .pred %p;\n'
            + regs + '    mov.u32 %t, %tid.x;\n' + loads + setup + ''.join(f'    {o};\n' for o in ops)
            + f'    mul.wide.u32 %o, %t, {size}; ld.param.u64 %q, [po]; add.u64 %q, %q, %o; st.global.{rd} [%q], %d;\n'
            + '    ret;\n}\n' + bufs + f'.buffer O .u{width} [32];\n.launch k <<< 1, 32 >>> ({", ".join(args + ["O"])});\n')


def battery():
    P = {}
    def add(name, op, srcs, dst):
        P[name] = program(name, [op], srcs, dst)
    for f in ('f32', 'f64'):
        for o in ('add', 'sub', 'mul', 'min', 'max'):
            add(f'{o}-{f}', f'{o}.{f} %d, %a, %b', (f, f), f)
        add(f'div-{f}', f'div.{"full." if f == "f32" else "rn."}{f} %d, %a, %b', (f, f), f)
        add(f'fma-{f}', f'fma.rn.{f} %d, %a, %b, %c', (f, f, f), f)
        for o in ('abs', 'neg'):
            add(f'{o}-{f}', f'{o}.{f} %d, %a', (f,), f)
        add(f'sqrt-{f}', f'sqrt.{"approx." if f == "f32" else "rn."}{f} %d, %a', (f,), f)
        add(f'rcp-{f}', f'rcp.{"approx." if f == "f32" else "rn."}{f} %d, %a', (f,), f)
        for r in ('rni', 'rzi', 'rmi', 'rpi'):
            add(f'cvtrnd-{r}-{f}', f'cvt.{r}.{f}.{f} %d, %a', (f,), f)
        for cmp_ in ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'equ', 'neu', 'ltu', 'num', 'nan'):
            P[f'setp-{cmp_}-{f}'] = program(f'setp-{cmp_}-{f}', [f'setp.{cmp_}.{f} %p, %a, %b', 'selp.b32 %d, 1, 0, %p'],
                                            (f, f), 'b32')
    for o in ('sin', 'cos', 'lg2', 'ex2', 'rsqrt'):
        add(f'{o}-approx-f32', f'{o}.approx.f32 %d, %a', ('f32',), 'f32')
    add('tanh-approx-f16', 'tanh.approx.f16 %d, %a', ('f16',), 'f16')
    add('rsqrt-approx-f64', 'rsqrt.approx.f64 %d, %a', ('f64',), 'f64')
    add('rcp-approx-f64', 'rcp.approx.ftz.f64 %d, %a', ('f64',), 'f64')
    for o in ('add', 'sub', 'mul', 'min', 'max', 'neg', 'abs'):
        n = 1 if o in ('neg', 'abs') else 2
        srcs = ('f16',) * n
        add(f'{o}-f16', f'{o}.f16 %d, %a' + (', %b' if n == 2 else ''), srcs, 'f16')
    for w in ('32', '64'):
        add(f'add-u{w}', f'add.u{w} %d, %a, %b', (f'u{w}', f'u{w}'), f'b{w}')
        add(f'sub-u{w}', f'sub.u{w} %d, %a, %b', (f'u{w}', f'u{w}'), f'b{w}')
        add(f'mul-lo-u{w}', f'mul.lo.u{w} %d, %a, %b', (f'u{w}', f'u{w}'), f'b{w}')
        add(f'mul-hi-u{w}', f'mul.hi.u{w} %d, %a, %b', (f'u{w}', f'u{w}'), f'b{w}')
        add(f'mul-hi-s{w}', f'mul.hi.s{w} %d, %a, %b', (f's{w}', f's{w}'), f'b{w}')
        add(f'div-u{w}', f'div.u{w} %d, %a, %b', (f'u{w}', f'u{w}'), f'b{w}')
        add(f'rem-u{w}', f'rem.u{w} %d, %a, %b', (f'u{w}', f'u{w}'), f'b{w}')
        add(f'div-s{w}', f'div.s{w} %d, %a, %b', (f's{w}', f's{w}'), f'b{w}')
        add(f'rem-s{w}', f'rem.s{w} %d, %a, %b', (f's{w}', f's{w}'), f'b{w}')
        for o in ('min', 'max'):
            add(f'{o}-u{w}', f'{o}.u{w} %d, %a, %b', (f'u{w}', f'u{w}'), f'b{w}')
            add(f'{o}-s{w}', f'{o}.s{w} %d, %a, %b', (f's{w}', f's{w}'), f'b{w}')
        for o in ('and', 'or', 'xor'):
            add(f'{o}-b{w}', f'{o}.b{w} %d, %a, %b', (f'b{w}', f'b{w}'), f'b{w}')
        add(f'not-b{w}', f'not.b{w} %d, %a', (f'b{w}',), f'b{w}')
        add(f'neg-s{w}', f'neg.s{w} %d, %a', (f's{w}',), f'b{w}')
        add(f'abs-s{w}', f'abs.s{w} %d, %a', (f's{w}',), f'b{w}')
        add(f'brev-b{w}', f'brev.b{w} %d, %a', (f'b{w}',), f'b{w}')
        add(f'popc-b{w}', f'popc.b{w} %d, %a', (f'b{w}',), 'b32')
        add(f'clz-b{w}', f'clz.b{w} %d, %a', (f'b{w}',), 'b32')
        add(f'bfind-u{w}', f'bfind.u{w} %d, %a', (f'u{w}',), 'b32')
        P[f'shl-b{w}'] = program(f'shl-b{w}', ['and.b32 %n, %t, 31', f'shl.b{w} %d, %a, %n'], (f'b{w}',), f'b{w}')
        P[f'shr-u{w}'] = program(f'shr-u{w}', ['and.b32 %n, %t, 31', f'shr.u{w} %d, %a, %n'], (f'u{w}',), f'b{w}')
        P[f'shr-s{w}'] = program(f'shr-s{w}', ['and.b32 %n, %t, 31', f'shr.s{w} %d, %a, %n'], (f's{w}',), f'b{w}')
        P[f'bfe-u{w}'] = program(f'bfe-u{w}', ['and.b32 %n, %t, 15', f'bfe.u{w} %d, %a, %n, 8'], (f'u{w}',), f'b{w}')
        P[f'bfe-s{w}'] = program(f'bfe-s{w}', ['and.b32 %n, %t, 15', f'bfe.s{w} %d, %a, %n, 8'], (f's{w}',), f'b{w}')
        P[f'bfi-b{w}'] = program(f'bfi-b{w}', ['and.b32 %n, %t, 15', f'bfi.b{w} %d, %a, %b, %n, 8'], (f'b{w}', f'b{w}'),
                                 f'b{w}')
        for c in ('eq', 'lt', 'le', 'gt', 'ge', 'ne'):
            P[f'setp-{c}-s{w}'] = program(f'setp-{c}-s{w}', [f'setp.{c}.s{w} %p, %a, %b', 'selp.b32 %d, 1, 0, %p'],
                                          (f's{w}', f's{w}'), 'b32')
        for c in ('lt', 'gt'):
            P[f'setp-{c}-u{w}'] = program(f'setp-{c}-u{w}', [f'setp.{c}.u{w} %p, %a, %b', 'selp.b32 %d, 1, 0, %p'],
                                          (f'u{w}', f'u{w}'), 'b32')
    add('mad-lo-s32', 'mad.lo.s32 %d, %a, %b, %c', ('s32', 's32', 's32'), 'b32')
    add('mad-hi-u32', 'mad.hi.u32 %d, %a, %b, %c', ('u32', 'u32', 'u32'), 'b32')
    add('mul-wide-u32', 'mul.wide.u32 %d, %a, %b', ('u32', 'u32'), 'b64')
    add('mul-wide-s32', 'mul.wide.s32 %d, %a, %b', ('s32', 's32'), 'b64')
    add('prmt-b32', 'prmt.b32 %d, %a, %b, 0x3120', ('b32', 'b32'), 'b32')
    add('dp4a', 'dp4a.u32.u32 %d, %a, %b, %c', ('u32', 'u32', 'u32'), 'b32')
    # conversions
    ints = ['u8', 's8', 'u16', 's16', 'u32', 's32', 'u64', 's64']
    for src in ('f32', 'f64', 'f16'):
        for dst in ints:
            if dst.endswith('8'):
                continue
            for r in ('rzi', 'rni', 'rmi', 'rpi'):
                add(f'cvt-{r}-{dst}-{src}', f'cvt.{r}.{dst}.{src} %d, %a', (src,), 'b' + dst[1:])
        for dst in ints:
            if dst.endswith('8'):
                continue
            add(f'cvt-{src}-{dst}', f'cvt.{"rn." if src != "f64" or dst in ("u64", "s64") else ""}{src}.{dst} %d, %a'
                if src != 'f64' else f'cvt.rn.f64.{dst} %d, %a', (dst,), src)
    add('cvt-f64-f32', 'cvt.f64.f32 %d, %a', ('f32',), 'f64')
    add('cvt-f32-f64', 'cvt.rn.f32.f64 %d, %a', ('f64',), 'f32')
    add('cvt-f16-f32', 'cvt.rn.f16.f32 %d, %a', ('f32',), 'f16')
    add('cvt-f32-f16', 'cvt.f32.f16 %d, %a', ('f16',), 'f32')
    for a, b in (('u32', 'u16'), ('s32', 's16'), ('u16', 'u32'), ('s16', 's32'), ('u64', 'u32'), ('s64', 's32'),
                 ('u32', 'u64'), ('s32', 's64')):
        add(f'cvt-{a}-{b}', f'cvt.{a}.{b} %d, %a', (b,), 'b' + a[1:])
    return P


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    progs = battery()
    for name, text in progs.items():
        with open(os.path.join(out, name + '.ptx'), 'w') as f:
            f.write(text)
    print(len(progs), 'programs in', out)


if __name__ == '__main__':
    main()
