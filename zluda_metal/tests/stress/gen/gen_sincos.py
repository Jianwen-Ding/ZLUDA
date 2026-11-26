#!/usr/bin/env python3
"""sin.approx.f32 / cos.approx.f32 stress program: expected values from an exact evaluation of PTX-formalization's
definition (sail/ptx/ptx_math.sail sincos_r: correctly rounded to nearest binary32; NaN and infinity give NaN).
Operands concentrated where the helper library's binary32 fast path (|x| < 200, ptx/lib/metal_impl/float.inc
f_sincos_fast) is closest to its limits: next to multiples of pi/32 and of pi (small reduced arguments, results near
zero), at 2^-12, at 200 and pi/64, plus random operands in and beyond its range and binary32's extremes. Exact
arithmetic: decimal Taylor series at 140 digits after a reduction by a 140-digit pi, rationals for the rounding.
   usage: gen_sincos.py N SEED [sin|cos] > out.ptx"""
import decimal
import random
import struct
import sys
from fractions import Fraction

D = decimal.Decimal
decimal.getcontext().prec = 140


def pi_dec():
    # Machin: pi = 16 atan(1/5) - 4 atan(1/239)
    def atan_inv(n):
        s, p, k, n2 = D(0), D(1) / n, 0, n * n
        while True:
            t = p / (2 * k + 1)
            if t < D(10) ** -150:
                return s
            s += -t if k & 1 else t
            p /= n2
            k += 1
    return 16 * atan_inv(5) - 4 * atan_inv(239)


PI = pi_dec()


def f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def rn32(q):
    """nearest-even binary32 of the rational q (normal range)"""
    if q == 0:
        return 0
    s = q < 0
    a = -q if s else q
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    m = a / Fraction(2) ** (e - 23)
    fl = m.numerator // m.denominator
    rem = m - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl & 1):
        fl += 1
    if fl == 1 << 24:
        fl >>= 1
        e += 1
    return (((e + 127) << 23) | (fl & 0x7FFFFF)) | (s << 31)


def sincos_dec(x, issin):
    # reduce x by pi/2 exactly enough (x is a binary32: at most ~2^128, 140 digits leave > 90 after reduction)
    k = int((x / (PI / 2)).to_integral_value(rounding=decimal.ROUND_HALF_EVEN))
    r = x - k * (PI / 2)
    q = (k if issin else k + 1) % 4
    # q 0: sin r, 1: cos r, 2: -sin r, 3: -cos r
    r2 = r * r
    if q & 1:
        s, t, n = D(0), D(1), 0
    else:
        s, t, n = D(0), r, 1
    while abs(t) > D(10) ** -145:
        s += t
        t = -t * r2 / ((n + 1) * (n + 2))
        n += 2
    return -s if q & 2 else s


def model(b, issin):
    x = f32(b)
    if x != x or x in (float("inf"), float("-inf")):
        return 0x7FFFFFFF
    if x == 0:
        return b if issin else 0x3F800000
    if abs(x) < 2.0 ** -60:  # sin x rounds to x (also subnormals), cos x to 1
        return b if issin else 0x3F800000
    return rn32(Fraction(sincos_dec(D(x), issin)))


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    fn = sys.argv[3] if len(sys.argv) > 3 else "sin"
    issin = fn == "sin"
    rnd = random.Random(seed * 2 + issin)
    xs = []
    for i in range(n):
        k = i % 8
        if k == 0:
            x = rnd.uniform(-200, 200)
        elif k == 1:  # next to a multiple of pi/32 (small reduced argument), and of pi/2 (results near 0 or 1)
            m = rnd.randint(-2040, 2040)
            x = m * float(PI) / 32 * (1 + rnd.uniform(-1, 1) * 2.0 ** -rnd.randint(18, 24))
        elif k == 2:
            m = rnd.randint(-127, 127)
            x = m * float(PI) / 2
            x = f32(bits(x) + rnd.randint(-3, 3)) if x else rnd.uniform(-1e-3, 1e-3)
        elif k == 3:  # magnitudes from 2^-14 to 2^8
            x = rnd.choice([1, -1]) * rnd.uniform(1, 2) * 2.0 ** rnd.randint(-14, 7)
        elif k == 4:  # the limits: 2^-12, pi/64, 200
            x = rnd.choice([2.0 ** -12, float(PI) / 64, 200.0]) * rnd.choice([1, -1])
            x = f32(bits(x) + rnd.randint(-4, 4))
        elif k == 5:  # beyond the fast path
            x = rnd.choice([1, -1]) * rnd.uniform(1, 2) * 2.0 ** rnd.randint(8, 127)
        elif k == 6:  # tiny and subnormal
            x = rnd.choice([1, -1]) * rnd.uniform(1, 2) * 2.0 ** rnd.randint(-149, -12)
        else:
            x = rnd.uniform(-4, 4)
        xs.append(bits(x))
    want = [model(b, issin) for b in xs]
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
    %s.approx.f32 %%f2, %%f1;
    add.s64 %%rd5, %%rd2, %%rd3; st.global.f32 [%%rd5], %%f2;
    ret;
}""" % fn)
    print(".buffer I .u32 = { %s };" % ", ".join(str(b) for b in xs))
    print(".buffer O .u32 [%d];" % n)
    print(".launch k <<< %d, 256 >>> (I, O);" % (n // 256))
    print(".expect O .u32 = { %s };" % ", ".join(str(b) for b in want))


main()
