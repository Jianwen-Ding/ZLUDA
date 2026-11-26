#!/usr/bin/env python3
"""ZLUDA's cuBLAS (src/blas.rs: MLIR kernels on the AGX backend) against numpy references.

Each case calls the cublas* entry point of libzluda_metal.so through ctypes, with random data, and compares the
result with numpy's (float64) result: relative error (max |got - want| / max |want|) within a tolerance per type
(f32 1e-5 * sqrt(k), f64 1e-12, f16 / bf16 outputs their precision). Cases cover transposes, leading dimensions
larger than the rows, negative and non-unit increments, alpha / beta = 0 and 1, both pointer modes, batches.

  test_blas.py [--lib LIB] [-k SUBSTR] [-v]
"""
import argparse
import ctypes
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ptxconf"))
import ptxconf  # noqa: E402

c_int, c_void_p, c_int64 = ctypes.c_int, ctypes.c_void_p, ctypes.c_int64
N, T = 0, 1
rng = np.random.default_rng(1234)


class Blas:
    def __init__(self, lib):
        os.environ["ZLUDA_METAL_BACKEND"] = "agx"
        sib = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "llvm-project", "build", "bin", "llvm-air2agx"))
        if os.path.exists(sib):
            os.environ.setdefault("ZLUDA_AGX_AIR2AGX", sib)
        self.cu = ptxconf.Cuda(lib)
        self.l = self.cu.l
        self.h = c_void_p()
        self.ok(self.l.cublasCreate_v2(ctypes.byref(self.h)), "cublasCreate")

    def ok(self, r, what):
        if r != 0:
            raise RuntimeError("%s returned %d" % (what, r))

    def dev(self, a):
        a = np.ascontiguousarray(a)
        return self.cu.alloc(a.tobytes()), a

    def get(self, p, like):
        return np.frombuffer(self.cu.read(p, like.nbytes), dtype=like.dtype).reshape(like.shape)

    def call(self, name, *args):
        f = getattr(self.l, name)
        conv = []
        for a in args:
            if isinstance(a, float):
                raise TypeError("pass scalars by pointer")
            conv.append(a)
        self.ok(f(*conv), name)

    def set_pointer_mode(self, m):
        self.ok(self.l.cublasSetPointerMode_v2(self.h, m), "cublasSetPointerMode")


def ptr(v, t):
    x = t(v)
    return ctypes.byref(x), x


def rel(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    den = max(np.max(np.abs(want)), 1e-30)
    return float(np.max(np.abs(got - want)) / den)


CASES = []


def case(f):
    CASES.append(f)
    return f


def ctype(dt):
    return ctypes.c_float if dt == np.float32 else ctypes.c_double


def tol(dt, k=1):
    return 2e-5 * max(1, np.sqrt(k)) if dt == np.float32 else 1e-12 * max(1, np.sqrt(k))


def P(dt):
    return "S" if dt == np.float32 else "D"


def strided(n, inc, dt):
    """a vector of n elements at increment inc (storage, logical view as numpy indexes)"""
    store = rng.standard_normal(max(1, 1 + (n - 1) * abs(inc))).astype(dt)
    idx = np.arange(n) * abs(inc)
    if inc < 0:
        idx = idx[::-1]
    return store, idx


for dt in (np.float32, np.float64):
    for n, incx, incy in ((1, 1, 1), (1000, 1, 1), (777, 2, -3), (5000, -1, 2)):
        @case
        def axpy(b, dt=dt, n=n, incx=incx, incy=incy):
            x, ix = strided(n, incx, dt)
            y, iy = strided(n, incy, dt)
            alpha = 0.75
            px, _ = b.dev(x)
            py, y0 = b.dev(y)
            a, _keep = ptr(alpha, ctype(dt))
            b.call("cublas%saxpy_v2" % P(dt), b.h, n, a, c_void_p(px), incx, c_void_p(py), incy)
            want = y0.astype(np.float64).copy()
            want[iy] += alpha * x.astype(np.float64)[ix]
            return rel(b.get(py, y0), want), tol(dt)

        @case
        def dot(b, dt=dt, n=n, incx=incx, incy=incy):
            x, ix = strided(n, incx, dt)
            y, iy = strided(n, incy, dt)
            px, _ = b.dev(x)
            py, _ = b.dev(y)
            r = ctype(dt)()
            b.call("cublas%sdot_v2" % P(dt), b.h, n, c_void_p(px), incx, c_void_p(py), incy, ctypes.byref(r))
            want = np.dot(x.astype(np.float64)[ix], y.astype(np.float64)[iy])
            return abs(r.value - want) / max(1.0, np.sum(np.abs(x[ix] * y[iy]))), tol(dt, n)

    for n, inc in ((1, 1), (3000, 1), (999, 3)):
        @case
        def scal_nrm2_asum_amax(b, dt=dt, n=n, inc=inc):
            x, ix = strided(n, inc, dt)
            px, x0 = b.dev(x)
            r = ctype(dt)()
            b.call("cublas%snrm2_v2" % P(dt), b.h, n, c_void_p(px), inc, ctypes.byref(r))
            e = abs(r.value - np.linalg.norm(x.astype(np.float64)[ix])) / np.linalg.norm(x[ix])
            b.call("cublas%sasum_v2" % P(dt), b.h, n, c_void_p(px), inc, ctypes.byref(r))
            e = max(e, abs(r.value - np.sum(np.abs(x.astype(np.float64)[ix]))) / np.sum(np.abs(x[ix])))
            i = c_int()
            b.call("cublasI%samax_v2" % P(dt).lower(), b.h, n, c_void_p(px), inc, ctypes.byref(i))
            if i.value != int(np.argmax(np.abs(x[ix]))) + 1:
                return 1.0, 0
            b.call("cublasI%samin_v2" % P(dt).lower(), b.h, n, c_void_p(px), inc, ctypes.byref(i))
            if i.value != int(np.argmin(np.abs(x[ix]))) + 1:
                return 1.0, 0
            a, _keep = ptr(-2.0, ctype(dt))
            b.call("cublas%sscal_v2" % P(dt), b.h, n, a, c_void_p(px), inc)
            want = x0.astype(np.float64).copy()
            want[ix] *= -2.0
            return max(e, rel(b.get(px, x0), want)), tol(dt, n)

    @case
    def copy_swap(b, dt=dt):
        n = 1234
        x, ix = strided(n, 2, dt)
        y, iy = strided(n, -1, dt)
        px, x0 = b.dev(x)
        py, y0 = b.dev(y)
        b.call("cublas%sswap_v2" % P(dt), b.h, n, c_void_p(px), 2, c_void_p(py), -1)
        wx, wy = x0.copy(), y0.copy()
        wx[ix], wy[iy] = y0[iy], x0[ix]
        e = max(rel(b.get(px, x0), wx), rel(b.get(py, y0), wy))
        b.call("cublas%scopy_v2" % P(dt), b.h, n, c_void_p(px), 2, c_void_p(py), -1)
        wy2 = wy.copy()
        wy2[iy] = wx[ix]
        return max(e, rel(b.get(py, y0), wy2)), 0 if e == 0 else 1e-30

    for trans, m, n, lda in ((N, 100, 70, 100), (T, 64, 129, 80), (N, 1, 300, 5)):
        @case
        def gemv(b, dt=dt, trans=trans, m=m, n=n, lda=lda):
            A = rng.standard_normal((n, lda)).astype(dt)  # column-major m x n with leading dimension lda: A[j, i]
            xs, ys = (n, m) if trans == N else (m, n)
            x, ix = strided(xs, 1, dt)
            y, iy = strided(ys, -2, dt)
            pa, _ = b.dev(A)
            px, _ = b.dev(x)
            py, y0 = b.dev(y)
            al, _k1 = ptr(1.5, ctype(dt))
            be, _k2 = ptr(-0.5, ctype(dt))
            b.call("cublas%sgemv_v2" % P(dt), b.h, trans, m, n, al, c_void_p(pa), lda, c_void_p(px), 1, be,
                   c_void_p(py), -2)
            Am = A.astype(np.float64)[:, :m].T  # m x n
            op = Am if trans == N else Am.T
            want = y0.astype(np.float64).copy()
            want[iy] = 1.5 * op @ x.astype(np.float64)[ix] - 0.5 * want[iy]
            return rel(b.get(py, y0), want), tol(dt, max(m, n))

    for ta, tb, m, n, k, beta in ((N, N, 64, 64, 64, 0.0), (T, N, 100, 37, 81, 1.0), (N, T, 129, 130, 3, -0.5),
                                  (T, T, 7, 200, 65, 0.25)):
        @case
        def gemm(b, dt=dt, ta=ta, tb=tb, m=m, n=n, k=k, beta=beta):
            lda = (m if ta == N else k) + 3
            ldb = (k if tb == N else n) + 1
            ldc = m + 2
            A = rng.standard_normal(((k if ta == N else m), lda)).astype(dt)
            B = rng.standard_normal(((n if tb == N else k), ldb)).astype(dt)
            C = rng.standard_normal((n, ldc)).astype(dt)
            pa, _ = b.dev(A)
            pb, _ = b.dev(B)
            pc, c0 = b.dev(C)
            al, _k1 = ptr(0.5, ctype(dt))
            be, _k2 = ptr(beta, ctype(dt))
            b.call("cublas%sgemm_v2" % P(dt), b.h, ta, tb, m, n, k, al, c_void_p(pa), lda, c_void_p(pb), ldb, be,
                   c_void_p(pc), ldc)
            Am = A.astype(np.float64)[:, :(m if ta == N else k)].T
            Bm = B.astype(np.float64)[:, :(k if tb == N else n)].T
            opA = Am if ta == N else Am.T
            opB = Bm if tb == N else Bm.T
            want = c0.astype(np.float64).copy()
            want[:, :m] = (0.5 * opA @ opB + beta * want[:, :m].T).T
            return rel(b.get(pc, c0), want), tol(dt, k)

    @case
    def gemm_strided_batched_device_scalars(b, dt=dt):
        m, n, k, batch = 33, 17, 40, 3
        A = rng.standard_normal((batch, k, m)).astype(dt)
        B = rng.standard_normal((batch, n, k)).astype(dt)
        C = rng.standard_normal((batch, n, m)).astype(dt)
        pa, _ = b.dev(A)
        pb, _ = b.dev(B)
        pc, c0 = b.dev(C)
        pal, _ = b.dev(np.array([2.0], dt))
        pbe, _ = b.dev(np.array([1.0], dt))
        b.set_pointer_mode(1)
        try:
            b.call("cublas%sgemmStridedBatched" % P(dt), b.h, N, N, m, n, k, c_void_p(pal), c_void_p(pa), m,
                   c_int64(m * k), c_void_p(pb), k, c_int64(k * n), c_void_p(pbe), c_void_p(pc), m, c_int64(m * n), batch)
        finally:
            b.set_pointer_mode(0)
        want = np.stack([(2.0 * A[i].T.astype(np.float64) @ B[i].T.astype(np.float64) + C[i].T).T for i in range(batch)])
        return rel(b.get(pc, c0), want), tol(dt, k)


def f16(x):
    return x.astype(np.float16)


def bf16_bits(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_val(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


for at, ct in (("f16", "f16"), ("f16", "f32"), ("bf16", "bf16"), ("bf16", "f32")):
    @case
    def gemm_ex(b, at=at, ct=ct):
        m, n, k = 72, 40, 96
        codes = {"f16": 2, "bf16": 14, "f32": 0}
        Af = rng.standard_normal((k, m)).astype(np.float32)
        Bf = rng.standard_normal((n, k)).astype(np.float32)
        Cf = rng.standard_normal((n, m)).astype(np.float32)
        conv = (lambda x: f16(x)) if at == "f16" else bf16_bits
        val = (lambda x: x.astype(np.float64)) if at == "f16" else bf16_val
        A, B = conv(Af), conv(Bf)
        C = Cf if ct == "f32" else conv(Cf)
        cval = (lambda x: x.astype(np.float64)) if ct == "f32" else val
        pa, _ = b.dev(A)
        pb, _ = b.dev(B)
        pc, c0 = b.dev(C)
        al, _k1 = ptr(1.0, ctypes.c_float)
        be, _k2 = ptr(0.5, ctypes.c_float)
        b.call("cublasGemmEx", b.h, N, N, m, n, k, al, c_void_p(pa), codes[at], m, c_void_p(pb), codes[at], k, be,
               c_void_p(pc), codes[ct], m, 68, 0)
        want = (val(A).T @ val(B).T + 0.5 * cval(c0).T).T
        got = cval(b.get(pc, c0))
        return rel(got, want), (2e-5 * np.sqrt(k) if ct == "f32" else (2e-3 if at == "f16" else 1.6e-2))


@case
def hgemm(b):
    m, n, k = 50, 30, 64
    A = f16(rng.standard_normal((k, m)))
    B = f16(rng.standard_normal((n, k)))
    C = f16(rng.standard_normal((n, m)))
    pa, _ = b.dev(A)
    pb, _ = b.dev(B)
    pc, c0 = b.dev(C)
    al = np.array([1.0], np.float16).view(np.uint16)
    be = np.array([1.0], np.float16).view(np.uint16)
    alp, _k1 = ptr(int(al[0]), ctypes.c_uint16)
    bep, _k2 = ptr(int(be[0]), ctypes.c_uint16)
    b.call("cublasHgemm", b.h, N, N, m, n, k, alp, c_void_p(pa), m, c_void_p(pb), k, bep, c_void_p(pc), m)
    want = (A.astype(np.float64).T @ B.astype(np.float64).T + c0.astype(np.float64).T).T
    return rel(b.get(pc, c0).astype(np.float64), want), 2e-3


@case
def set_get_matrix(b):
    rows, cols = 7, 5
    a = rng.standard_normal((cols, 9)).astype(np.float32)  # lda 9
    p = c_void_p(b.cu.alloc(bytes(4 * 8 * cols)))  # ldb 8
    b.call("cublasSetMatrix", rows, cols, 4, a.ctypes.data_as(c_void_p), 9, p, 8)
    out = np.zeros((cols, 6), np.float32)  # ldc 6... rows 7 do not fit: use 7
    out = np.zeros((cols, 7), np.float32)
    b.call("cublasGetMatrix", rows, cols, 4, p, 8, out.ctypes.data_as(c_void_p), 7)
    return rel(out[:, :rows], a[:, :rows]), 1e-30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.path.join(HERE, "..", "..", "target", "release", "libzluda_metal.so"))
    ap.add_argument("-k")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    b = Blas(a.lib)
    npass = nfail = 0
    for i, f in enumerate(CASES):
        d = f.__defaults__ or ()
        label = "%s%s" % (f.__name__, tuple(x.__name__ if isinstance(x, type) else x for x in d if x is not None))
        if a.k and a.k not in label:
            continue
        try:
            err, t = f(b)
            good = err <= t
            msg = "rel %.3g (tol %.3g)" % (err, t)
        except Exception as e:  # noqa: BLE001
            good, msg = False, "".join(traceback.format_exception_only(type(e), e)).strip()
        npass += good
        nfail += not good
        if a.v or not good:
            print("%-5s %s  %s" % ("PASS" if good else "FAIL", label, msg), flush=True)
    print("---- PASS %d, FAIL %d of %d" % (npass, nfail, npass + nfail))
    sys.exit(1 if nfail else 0)


main()
