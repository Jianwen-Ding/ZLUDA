#!/usr/bin/env python3
"""Generate agx_tests/approx-bounds.ptx: the approximate f32 functions on random inputs, checked against the error
bounds PTX ISA 9.4 states for them (section 9.7.3) with .expectbound, and their special-value tables.

    rcp.approx         1 ulp                 sqrt.approx   2^-23 relative     rsqrt.approx  2^-22.9 relative
    sin.approx         2^-20.5 absolute on [-2pi, 2pi], 2^-14.7 on [-100pi, 100pi]
    cos.approx         2^-20.4 absolute on [-2pi, 2pi], 2^-14.7 on [-100pi, 100pi]
    lg2.approx         2^-22 absolute on (0.5, 2), 2^-22 relative elsewhere
    ex2.approx         2 ulp                 tanh.approx   2^-11 relative

Without .ftz subnormal inputs and results are supported; with .ftz they are flushed to sign-preserving zero.
The exact values are computed in double precision, far closer than the bounds.
    usage: gen_approx_bounds.py [N] [SEED] > agx_tests/approx-bounds.ptx
"""
import math
import random
import struct
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 7)
INF, NAN = math.inf, math.nan


def f2b(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def b2f(b):
    return struct.unpack('<f', struct.pack('<I', b))[0]


def subnormal(x):
    return x != 0 and abs(x) < 2.0 ** -126


def flush(x):
    return math.copysign(0.0, x) if subnormal(x) else x


def rand_exp(lo, hi, sign=True):
    """A random f32 with an exponent in [lo, hi] (lo < -126: subnormals too)."""
    e = random.randint(lo, hi)
    if e < -126:
        b = random.randint(1, (1 << 23) - 1) >> min(23, -126 - e)
        b = max(b, 1)
    else:
        b = (e + 127) << 23 | random.getrandbits(23)
    if sign and random.random() < .5:
        b |= 1 << 31
    return b2f(b)


def uniform(lo, hi):
    return b2f(f2b(random.uniform(lo, hi)))


def rcp(x):
    return math.copysign(INF, x) if x == 0 else (math.copysign(0.0, x) if math.isinf(x) else 1 / x)


def sqrt_(x):
    return x if x == 0 else (NAN if x < 0 or x != x else math.sqrt(x))


def rsqrt(x):
    return math.copysign(INF, x) if x == 0 else (NAN if x < 0 or x != x else (0.0 if math.isinf(x) else x ** -0.5))


def sin_(x):
    return x if x == 0 else (NAN if math.isinf(x) or x != x else math.sin(x))


def cos_(x):
    return NAN if math.isinf(x) or x != x else math.cos(x)


def lg2(x):
    return -INF if x == 0 else (NAN if x < 0 or x != x else (INF if math.isinf(x) else math.log2(x)))


def ex2(x):
    if x != x:
        return NAN
    if x == -INF:
        return 0.0
    if x > 128:
        return INF
    return 2.0 ** x


def tanh_(x):
    return math.tanh(x)


SPECIAL = [0.0, -0.0, INF, -INF, NAN]
# (instruction, function, [(kind, bound, inputs)])
CASES = [
    ('rcp.approx.f32', rcp, [('ulp', '1', SPECIAL + [rand_exp(-149, 127) for _ in range(N)])]),
    ('rcp.approx.ftz.f32', rcp, [('ulp', '1', SPECIAL + [rand_exp(-149, 127) for _ in range(N)])]),
    ('sqrt.approx.f32', sqrt_, [('rel', '2^-23', SPECIAL + [-1.0, -2.0 ** -140] +
                                 [rand_exp(-149, 127, False) for _ in range(N)])]),
    ('sqrt.approx.ftz.f32', sqrt_, [('rel', '2^-23', SPECIAL + [rand_exp(-149, 127, False) for _ in range(N)])]),
    ('rsqrt.approx.f32', rsqrt, [('rel', '2^-22.9', SPECIAL + [-1.0] + [rand_exp(-149, 127, False) for _ in range(N)])]),
    ('rsqrt.approx.ftz.f32', rsqrt, [('rel', '2^-22.9', SPECIAL + [rand_exp(-149, 127, False) for _ in range(N)])]),
    ('sin.approx.f32', sin_, [('abs', '2^-20.5', SPECIAL + [uniform(-2 * math.pi, 2 * math.pi) for _ in range(N)]),
                              ('abs', '2^-14.7', [uniform(-100 * math.pi, 100 * math.pi) for _ in range(N)])]),
    ('cos.approx.f32', cos_, [('abs', '2^-20.4', SPECIAL + [uniform(-2 * math.pi, 2 * math.pi) for _ in range(N)]),
                              ('abs', '2^-14.7', [uniform(-100 * math.pi, 100 * math.pi) for _ in range(N)])]),
    ('lg2.approx.f32', lg2, [('abs', '2^-22', [1.0] + [uniform(0.5, 2.0) for _ in range(N)]),
                             ('rel', '2^-22', SPECIAL + [-1.0] + [x for x in (rand_exp(-149, 127, False)
                                                                              for _ in range(2 * N))
                                                                  if not 0.5 <= x <= 2.0][:N])]),
    ('lg2.approx.ftz.f32', lg2, [('rel', '2^-22', [x for x in (rand_exp(-149, 127, False) for _ in range(2 * N))
                                                   if not 0.5 <= x <= 2.0][:N])]),
    ('ex2.approx.f32', ex2, [('ulp', '2', SPECIAL + [uniform(-150, 128) for _ in range(N)] +
                              [float(k) for k in range(-149, 128, 7)])]),
    ('ex2.approx.ftz.f32', ex2, [('ulp', '2', SPECIAL + [uniform(-150, 128) for _ in range(N)])]),
    ('tanh.approx.f32', tanh_, [('rel', '2^-11', [0.0, -0.0, INF, -INF] + [uniform(-12, 12) for _ in range(N)] +
                                 [rand_exp(-149, -1) for _ in range(N // 4)])]),
]


def lit(x):
    return 'nan' if x != x else ('-inf' if x == -INF else ('inf' if x == INF else repr(x)))


out = ['// Generated by ptx/tools/gen_approx_bounds.py %d %s: the approximate f32 functions within the error'
       % (N, sys.argv[2] if len(sys.argv) > 2 else '7'),
       '// bounds of PTX ISA 9.4, section 9.7.3 (.expectbound: agx_ptx_formalization_check.py only)',
       '.version 8.0', '.target sm_80', '.address_size 64', '']
harness = []
for k, (insn, fun, groups) in enumerate(CASES):
    ftz = '.ftz' in insn
    out += ['.visible .entry k%d(.param .u64 px, .param .u64 py)' % k, '{',
            '    .reg .b32 %r0; .reg .f32 %f<3>; .reg .b64 %rd<4>;',
            '    mov.u32 %r0, %tid.x; mul.wide.u32 %rd1, %r0, 4;',
            '    ld.param.u64 %rd2, [px]; add.s64 %rd2, %rd2, %rd1; ld.global.f32 %f1, [%rd2];',
            '    %s %%f2, %%f1;' % insn,
            '    ld.param.u64 %rd3, [py]; add.s64 %rd3, %rd3, %rd1; st.global.f32 [%rd3], %f2;',
            '    ret;', '}', '']
    xs = [x for _, _, inputs in groups for x in inputs]
    xs += [1.0] * (-len(xs) % 32)   # whole warps (unchecked)
    harness.append('// %s' % insn)
    harness.append('.buffer X%d .u32 = { %s };' % (k, ', '.join(str(f2b(x)) for x in xs)))
    harness.append('.buffer Y%d .u32 [%d];' % (k, len(xs)))
    assert len(xs) <= 1024
    harness.append('.launch k%d <<< 1, %d >>> (X%d, Y%d);' % (k, len(xs), k, k))
    off = 0
    for kind, bound, inputs in groups:
        exact = []
        for x in inputs:
            y = fun(flush(x) if ftz else x)
            if ftz and subnormal(b2f(f2b(y)) if not (y != y or math.isinf(y)) and abs(y) < 3.4e38 else y):
                y = math.copysign(0.0, y)
            exact.append(lit(y))
        harness.append('.expectbound Y%d .f32 @ %d %s %s = { %s };' % (k, off, kind, bound, ', '.join(exact)))
        off += len(inputs)
print('\n'.join(out + harness))
