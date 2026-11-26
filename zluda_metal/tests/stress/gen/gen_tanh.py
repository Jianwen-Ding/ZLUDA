#!/usr/bin/env python3
"""tanh.approx.f32 stress program: expected values from an exact evaluation of PTX-formalization's definition
(sail/ptx/ptx_math.sail "tanh"): |x| < 2^-12 -> x; |x| >= 20 -> +-1; else (e1 - e2) / (e1 + e2) with e1, e2 the
correctly rounded binary32 e^x and e^-x and each operation rounded to nearest binary32. Exact arithmetic: decimal
exp at 60 digits, rationals for the roundings.   usage: gen_tanh.py N SEED > out.ptx"""
import decimal, random, struct, sys
from fractions import Fraction

decimal.getcontext().prec = 60


def f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def rn32(q):
    """nearest-even binary32 of the rational q (normal range only)"""
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
    bits = ((e + 127) << 23) | (fl & 0x7FFFFF)
    return bits | (s << 31)


def val(b):
    return Fraction(f32(b))


def tanh_model(b):
    x = f32(b)
    if x != x:
        return 0x7FFFFFFF
    ax = abs(x)
    if ax < 2.0 ** -12:
        return b
    if ax >= 20:
        return 0xBF800000 if x < 0 else 0x3F800000
    e1 = rn32(Fraction(decimal.Decimal(x).exp()))
    e2 = rn32(Fraction((-decimal.Decimal(x)).exp()))
    num = rn32(val(e1) - val(e2))
    den = rn32(val(e1) + val(e2))
    return rn32(val(num) / val(den))


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rnd = random.Random(seed)
    xs = []
    for i in range(n):
        k = i % 4
        if k == 0:
            x = rnd.uniform(-20, 20)
        elif k == 1:
            x = rnd.uniform(-1, 1) * 2.0 ** rnd.randint(-13, 0)
        elif k == 2:
            x = rnd.choice([0.25, 20.0, 2.0 ** -12, 1.0, 10.0]) * rnd.choice([1, -1]) * (1 + rnd.uniform(-1e-6, 1e-6))
        else:
            x = rnd.uniform(-5, 5)
        xs.append(struct.unpack("<I", struct.pack("<f", x))[0])
    want = [tanh_model(b) for b in xs]
    print(""".version 8.0
.target sm_90
.address_size 64
.visible .entry k(.param .u64 pi, .param .u64 po)
{
    .reg .b32 %r<4>; .reg .f32 %f<3>; .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [pi]; ld.param.u64 %rd2, [po];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd1, %rd3; ld.global.f32 %f1, [%rd4];
    tanh.approx.f32 %f2, %f1;
    add.s64 %rd5, %rd2, %rd3; st.global.f32 [%rd5], %f2;
    ret;
}""")
    print(".buffer I .u32 = { %s };" % ", ".join(str(b) for b in xs))
    print(".buffer O .u32 [%d];" % n)
    print(".launch k <<< %d, 256 >>> (I, O);" % (n // 256))
    print(".expect O .u32 = { %s };" % ", ".join(str(b) for b in want))


main()
