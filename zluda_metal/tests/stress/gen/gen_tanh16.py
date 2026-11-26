#!/usr/bin/env python3
"""tanh.approx.f16 over every binary16 input (PTX-formalization, sail/ptx/ptx_math.sail tanh_r: halves correctly
rounded; NaN -> canonical NaN, +-inf -> +-1, +-0 -> itself). Exact: decimal tanh at 60 digits, rounded as a rational.
usage: gen_tanh16.py PART PARTS > out.ptx   (inputs PART*65536/PARTS ... of the 65536 bit patterns)"""
import decimal
import struct
import sys
from fractions import Fraction

decimal.getcontext().prec = 60


def h2frac(b):
    return Fraction(struct.unpack("<e", struct.pack("<H", b))[0])


def rn16(q):
    """nearest-even binary16 bits of the rational q, |q| <= 1"""
    s = 0x8000 if q < 0 else 0
    a = -q if q < 0 else q
    if a == 0:
        return s
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    e = max(e, -14)  # subnormals: quantum 2^-24
    m = a / Fraction(2) ** (e - 10)
    fl = m.numerator // m.denominator
    rem = m - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl & 1):
        fl += 1
    v = Fraction(fl) * Fraction(2) ** (e - 10)
    return s | struct.unpack("<H", struct.pack("<e", float(v)))[0]


def model(b):
    e, m = (b >> 10) & 31, b & 1023
    if e == 31:
        return 0x7FFF if m else (b & 0x8000) | 0x3C00
    if b & 0x7FFF == 0:
        return b
    x = h2frac(b)
    d = decimal.Decimal(x.numerator) / decimal.Decimal(x.denominator)
    t = (d * 2).exp()
    t = (t - 1) / (t + 1)
    return rn16(Fraction(t))


def main():
    part, parts = int(sys.argv[1]), int(sys.argv[2])
    n = 65536 // parts
    xs = list(range(part * n, (part + 1) * n))
    want = [model(b) for b in xs]
    print(""".version 8.0
.target sm_90
.address_size 64
.visible .entry k(.param .u64 pi, .param .u64 po)
{
    .reg .b32 %r<4>; .reg .b16 %h<3>; .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [pi]; ld.param.u64 %rd2, [po];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;
    mul.wide.u32 %rd3, %r1, 2;
    add.s64 %rd4, %rd1, %rd3; ld.global.b16 %h1, [%rd4];
    tanh.approx.f16 %h2, %h1;
    add.s64 %rd5, %rd2, %rd3; st.global.b16 [%rd5], %h2;
    ret;
}""")
    print(".buffer I .u16 = { %s };" % ", ".join(str(b) for b in xs))
    print(".buffer O .u16 [%d];" % n)
    print(".launch k <<< %d, 256 >>> (I, O);" % (n // 256))
    print(".expect O .u16 = { %s };" % ", ".join(str(b) for b in want))


main()
