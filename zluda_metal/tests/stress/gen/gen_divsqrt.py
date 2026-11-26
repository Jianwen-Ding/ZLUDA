#!/usr/bin/env python3
"""div.rn.f32 and sqrt.rn.f32 stress program, concentrated where a fast path's validity matters: quotients within
a few ulps of the smallest normal 2^-126 (both sides), denormal operands, divisors near 2^127, square roots of the
smallest normals and of denormals, and random normal pairs. Expected values: exact rationals rounded to nearest even
binary32 with IEEE subnormals (PTX .rn without .ftz).   usage: gen_divsqrt.py N SEED > out.ptx"""
import math
import random
import struct
import sys
from fractions import Fraction


def f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def frac(b):
    e, m, s = (b >> 23) & 255, b & 0x7FFFFF, -1 if b >> 31 else 1
    if e == 0:
        return s * Fraction(m, 1 << 149)
    return s * Fraction(m | 0x800000, 1 << 23) * Fraction(2) ** (e - 127)


def rn(q):
    """nearest-even binary32 bits of the rational q (subnormals, overflow to inf)"""
    s = 0x80000000 if q < 0 else 0
    a = -q if q < 0 else q
    if a == 0:
        return s
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    e = max(e, -126)
    m = a / Fraction(2) ** (e - 23)
    fl = m.numerator // m.denominator
    r = m - fl
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and fl & 1):
        fl += 1
    v = Fraction(fl) * Fraction(2) ** (e - 23)
    if v >= Fraction(2) ** 128:
        return s | 0x7F800000
    return s | bits(float(v)) if v >= Fraction(2) ** -126 else s | int(v * (1 << 149))


def isqrt_frac(q):
    """sqrt of a non-negative rational to nearest-even binary32 (exact: compare squares)"""
    if q == 0:
        return 0
    approx = bits(math.sqrt(float(q)))
    cands = list(range(max(1, approx - 3), approx + 4))
    # exact: the correctly rounded root r has (mid(r-), mid(r+)) bracketing sqrt(q): compare squares of midpoints
    for c in cands:
        lo = (frac(c - 1) + frac(c)) / 2 if c > 0 else Fraction(0)
        hi = (frac(c) + frac(c + 1)) / 2
        if lo * lo <= q <= hi * hi:
            if lo * lo == q and c & 1:
                return c - 1
            if hi * hi == q and c & 1:
                return c + 1
            return c
    raise ValueError("sqrt rounding")


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rnd = random.Random(seed)
    a_s, b_s = [], []
    for i in range(n):
        k = i % 6
        if k == 0:  # quotient near 2^-126
            b = rnd.getrandbits(23) | (rnd.randint(110, 140) << 23)
            target = Fraction(2) ** -126 * (1 + Fraction(rnd.randint(-8, 8), 1 << 24))
            a = bits(float(target * frac(b)))
        elif k == 1:  # denormal operand
            a = rnd.getrandbits(23) or 1
            b = rnd.getrandbits(23) | (rnd.randint(1, 140) << 23)
            if rnd.random() < 0.5:
                a, b = b, a
        elif k == 2:  # huge divisor
            a = rnd.getrandbits(23) | (rnd.randint(100, 254) << 23)
            b = rnd.getrandbits(23) | (rnd.randint(250, 254) << 23)
        elif k == 3:  # tiny divisor / overflow side
            a = rnd.getrandbits(23) | (rnd.randint(200, 254) << 23)
            b = rnd.getrandbits(23) | (rnd.randint(1, 10) << 23)
        else:
            a = rnd.getrandbits(23) | (rnd.randint(1, 254) << 23)
            b = rnd.getrandbits(23) | (rnd.randint(1, 254) << 23)
        a |= rnd.getrandbits(1) << 31
        b |= rnd.getrandbits(1) << 31
        a_s.append(a)
        b_s.append(b)
    want = []
    for a, b in zip(a_s, b_s):
        q = rn(frac(a) / frac(b))
        x = a & 0x7FFFFFFF  # sqrt of |a|: mostly small normals and denormals by construction of k 0 / 1
        if (a >> 23) & 255 > 4 and rnd.random() < 0.5:
            x = rnd.getrandbits(23) | (rnd.randint(1, 3) << 23)
        want.append((q, isqrt_frac(frac(x)), x))
    print(""".version 8.0
.target sm_90
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 px, .param .u64 pq, .param .u64 ps)
{
    .reg .b32 %r<4>; .reg .f32 %f<6>; .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [pa]; ld.param.u64 %rd2, [pb]; ld.param.u64 %rd3, [px];
    ld.param.u64 %rd4, [pq]; ld.param.u64 %rd5, [ps];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;
    mul.wide.u32 %rd6, %r1, 4;
    add.s64 %rd7, %rd1, %rd6; ld.global.f32 %f1, [%rd7];
    add.s64 %rd8, %rd2, %rd6; ld.global.f32 %f2, [%rd8];
    add.s64 %rd9, %rd3, %rd6; ld.global.f32 %f3, [%rd9];
    div.rn.f32 %f4, %f1, %f2;
    sqrt.rn.f32 %f5, %f3;
    add.s64 %rd10, %rd4, %rd6; st.global.f32 [%rd10], %f4;
    add.s64 %rd11, %rd5, %rd6; st.global.f32 [%rd11], %f5;
    ret;
}""")
    print(".buffer A .u32 = { %s };" % ", ".join(map(str, a_s)))
    print(".buffer B .u32 = { %s };" % ", ".join(map(str, b_s)))
    print(".buffer X .u32 = { %s };" % ", ".join(str(w[2]) for w in want))
    print(".buffer Q .u32 [%d];" % n)
    print(".buffer S .u32 [%d];" % n)
    print(".launch k <<< %d, 256 >>> (A, B, X, Q, S);" % (n // 256))
    print(".expect Q .u32 = { %s };" % ", ".join(str(w[0]) for w in want))
    print(".expect S .u32 = { %s };" % ", ".join(str(w[1]) for w in want))


main()
