#!/usr/bin/env python3
"""ex2 / lg2 / rsqrt / rcp / sqrt .approx{.ftz}.f32 stress program: expected values from an exact evaluation of
PTX-formalization's definitions (sail/ptx/ptx_math.sail: ex2 = 2^x and lg2 = log2 x correctly rounded, rsqrt =
1 / sqrt x with both operations rounded, rcp and sqrt correctly rounded; .ftz flushes the operand and the result; NaN results 0x7fffffff). The
hardware's approximations (within an ulp) differ from these on many operands, so this catches a lowering to them;
operands also cover the limits of the helper library's binary32 fast paths (ptx/lib/metal_impl/float.inc
f_exp2_fast, f_lg2_fast): subnormal results and operands, x next to integers, to -125 / 127.5 and to 1.
Exact arithmetic: decimal exp / ln at 80 digits, rationals for the roundings.
   usage: gen_transc.py N SEED ex2|lg2|rsqrt|rcp|sqrt [ftz] > out.ptx"""
import decimal
import math
import random
import struct
import sys
from fractions import Fraction

D = decimal.Decimal
decimal.getcontext().prec = 80
LN2 = D(2).ln()


def f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def rn32(q):
    """nearest-even binary32 of the rational q, with subnormals and overflow to infinity"""
    if q == 0:
        return 0
    s = q < 0
    a = -q if s else q
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    e = max(e, -126)
    m = a / Fraction(2) ** (e - 23)
    fl = m.numerator // m.denominator
    rem = m - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl & 1):
        fl += 1
    if fl == 1 << 24:
        fl >>= 1
        e += 1
    if e > 127:
        return 0x7F800000 | (s << 31)
    if fl < 1 << 23:  # subnormal
        return fl | (s << 31)
    return (((e + 127) << 23) | (fl & 0x7FFFFF)) | (s << 31)


def flush(b):
    return b & 0x80000000 if (b & 0x7F800000) == 0 else b


def rsqrt_root(fx):
    """RN(sqrt x): a 70-bit integer root of x scaled by an even power of two, rounded once (never a tie)"""
    k = 140 - (fx.numerator.bit_length() - fx.denominator.bit_length())
    k += k & 1
    n = fx * Fraction(2) ** k
    rt = math.isqrt(n.numerator // n.denominator)
    sticky = rt * rt * n.denominator != n.numerator
    return rn32(Fraction(2 * rt + (1 if sticky else 0), 2) / Fraction(2) ** (k // 2))


def model(op, b, ftz):
    if ftz:
        b = flush(b)
    x = f32(b)
    if x != x:
        return 0x7FFFFFFF
    if op == "ex2":
        if math.isinf(x):
            r = 0 if x < 0 else 0x7F800000
        elif x == 0:
            r = 0x3F800000
        elif x >= 128:
            r = 0x7F800000
        elif x < -151:
            r = 0
        else:
            r = rn32(Fraction((D(Fraction(x).numerator) / D(Fraction(x).denominator) * LN2).exp()))
            if x == math.floor(x):
                r = rn32(Fraction(2) ** int(x))
    elif op == "lg2":
        if x < 0:
            return 0x7FFFFFFF
        if x == 0:
            return 0xFF800000
        if math.isinf(x):
            return 0x7F800000
        fx = Fraction(x)
        r = rn32(Fraction((D(fx.numerator).ln() - D(fx.denominator).ln()) / LN2))
        m, e = math.frexp(x)
        if m == 0.5:
            r = bits(float(e - 1))
    elif op == "rcp":  # correctly rounded 1 / x
        if math.isinf(x):
            return flush(b & 0x80000000) if ftz else b & 0x80000000
        if x == 0:
            return 0x7F800000 | (b & 0x80000000)
        r = rn32(1 / Fraction(x))
    elif op == "sqrt":
        if x < 0:
            return 0x7FFFFFFF
        if x == 0 or math.isinf(x):
            return b
        r = rsqrt_root(Fraction(x))
    else:  # rsqrt
        if x < 0:
            return 0x7FFFFFFF
        if x == 0:
            return 0x7F800000 | (b & 0x80000000)
        if math.isinf(x):
            return 0
        r = rn32(1 / Fraction(f32(rsqrt_root(Fraction(x)))))
    return flush(r) if ftz else r


def main():
    n, seed, op = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    ftz = len(sys.argv) > 4 and sys.argv[4] == "ftz"
    rnd = random.Random(seed * 7 + len(op) + ftz)
    xs = []
    for i in range(n):
        k = i % 6
        if op == "ex2":
            if k == 0:
                x = rnd.uniform(-150, 128)
            elif k == 1:  # next to integers and to the fast path's limits
                x = rnd.choice([rnd.randint(-150, 128), -125.0, 127.5, -126.0, -149.0, 0.0, 1.0])
                x = f32(max(bits(x) + rnd.randint(-3, 3), 0)) if x else rnd.choice([1, -1]) * 2.0 ** rnd.randint(-70, -20)
            elif k == 2:
                x = rnd.uniform(-1, 1) * 2.0 ** rnd.randint(-30, 0)
            elif k == 3:  # subnormal results
                x = rnd.uniform(-151, -125)
            elif k == 4:
                x = rnd.uniform(-8, 8)
            else:
                x = f32(rnd.getrandbits(32))
        else:
            if k == 0:
                x = rnd.uniform(0, 4)
            elif k == 1:  # next to 1 and to powers of two
                x = f32(bits(2.0 ** rnd.randint(-126, 127)) + rnd.randint(-40, 40))
            elif k == 2:  # subnormals
                x = f32(rnd.randint(1, 0x7FFFFF))
            elif k == 3:
                x = rnd.uniform(1, 2) * 2.0 ** rnd.randint(-126, 127)
            elif k == 4:
                x = rnd.choice([0.75, 1.5, 1.0, 0.5]) * (1 + rnd.uniform(-1, 1) * 2.0 ** -rnd.randint(1, 23))
            else:
                x = f32(rnd.getrandbits(32))
        xs.append(bits(float(x)))
    xs = [b & 0xFFFFFFFF for b in xs]
    want = [model(op, b, ftz) for b in xs]
    print(""".version 8.0
.target sm_90
.address_size 64
.visible .entry k(.param .u64 pi, .param .u64 po)
{
    .reg .b32 %%r<4>; .reg .f32 %%f<3>; .reg .b64 %%rd<6>;
    ld.param.u64 %%rd1, [pi]; ld.param.u64 %%rd2, [po];
    mov.u32 %%r1, %%ctaid.x; mov.u32 %%r2, %%ntid.x; mov.u32 %%r3, %%tid.x;
    mad.lo.s32 %%r1, %%r1, %%r2, %%r3;
    mul.wide.u32 %%rd3, %%r1, 4;
    add.s64 %%rd4, %%rd1, %%rd3; ld.global.f32 %%f1, [%%rd4];
    %s.approx%s.f32 %%f2, %%f1;
    add.s64 %%rd5, %%rd2, %%rd3; st.global.f32 [%%rd5], %%f2;
    ret;
}""" % (op, ".ftz" if ftz else ""))
    print(".buffer I .u32 = { %s };" % ", ".join(str(b) for b in xs))
    print(".buffer O .u32 [%d];" % n)
    print(".launch k <<< %d, 256 >>> (I, O);" % (n // 256))
    print(".expect O .u32 = { %s };" % ", ".join(str(b) for b in want))


main()
