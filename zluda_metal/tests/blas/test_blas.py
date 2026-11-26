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

    # (the shapes with m, n multiples of 16 and k of 8 take the SIMD-group matrix kernel for f32)
    for ta, tb, m, n, k, beta in ((N, N, 64, 64, 64, 0.0), (T, N, 100, 37, 81, 1.0), (N, T, 129, 130, 3, -0.5),
                                  (T, T, 7, 200, 65, 0.25), (N, N, 48, 32, 40, 1.0), (T, N, 32, 64, 16, -1.0),
                                  (N, T, 16, 16, 8, 0.5), (T, T, 80, 48, 24, 2.0)):
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
        return batched(b, dt, m, n, k, batch)

    @case
    def gemm_strided_batched_mma(b, dt=dt):
        return batched(b, dt, 32, 48, 24, 4)

    def batched(b, dt, m, n, k, batch):
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


# ------------------------------------------------------------------------------------------- level 1 / 2 / 3
LOWER, UPPER = 0, 1
LEFT, RIGHT = 0, 1


def colmajor(rows, cols, ld, dt):
    """storage of a column-major rows x cols matrix with leading dimension ld (numpy [col, row]) and its matrix"""
    st = rng.standard_normal((max(cols, 1), ld)).astype(dt)
    return st, st.astype(np.float64)[:cols, :rows].T


def vec(n, inc, dt):
    x, ix = strided(n, inc, dt)
    return x, ix, x.astype(np.float64)[ix]


def tri_of(M, uplo, diag):
    T = np.triu(M) if uplo == UPPER else np.tril(M)
    if diag == 1:
        np.fill_diagonal(T, 1.0)
    return T


def sym_of(M, uplo):
    T = np.triu(M) if uplo == UPPER else np.tril(M)
    return T + T.T - np.diag(np.diag(M))


def band_store(M, kl, ku, ld, dt):
    """LAPACK band storage of M (m x n): AB[ku + i - j, j]"""
    m, n = M.shape
    st = rng.standard_normal((n, ld)).astype(dt)  # garbage outside the band
    for j in range(n):
        for i in range(max(0, j - ku), min(m, j + kl + 1)):
            st[j, ku + i - j] = M[i, j]
    return st


def packed_store(M, uplo, dt):
    n = M.shape[0]
    out = []
    for j in range(n):
        out += list(M[:j + 1, j]) if uplo == UPPER else list(M[j:, j])
    return np.array(out, dt)


def tri_band_store(M, uplo, k, ld, dt):
    """symmetric / triangular band storage: upper AB[k + i - j, j], lower AB[i - j, j]"""
    n = M.shape[0]
    st = rng.standard_normal((n, ld)).astype(dt)
    for j in range(n):
        rng_i = range(max(0, j - k), j + 1) if uplo == UPPER else range(j, min(n, j + k + 1))
        for i in rng_i:
            st[j, (k + i - j) if uplo == UPPER else (i - j)] = M[i, j]
    return st


def banded(M, kl, ku):
    m, n = M.shape
    i, j = np.indices((m, n))
    return np.where((i - j <= kl) & (j - i <= ku), M, 0.0)


for dt in (np.float32, np.float64):

    @case
    def rot_rotm(b, dt=dt):
        n = 300
        x, ix, xv = vec(n, 2, dt)
        y, iy, yv = vec(n, -1, dt)
        px, x0 = b.dev(x)
        py, y0 = b.dev(y)
        c, s = 0.6, 0.8
        b.call("cublas%srot_v2" % P(dt), b.h, n, c_void_p(px), 2, c_void_p(py), -1, ptr(c, ctype(dt))[0], ptr(s, ctype(dt))[0])
        wx, wy = x0.astype(np.float64).copy(), y0.astype(np.float64).copy()
        wx[ix], wy[iy] = c * xv + s * yv, c * yv - s * xv
        e = max(rel(b.get(px, x0), wx), rel(b.get(py, y0), wy))
        for flag in (-1.0, 0.0, 1.0, -2.0):
            hp = np.array([flag, 0.5, -0.25, 1.5, 2.0], dt)
            x1, y1 = b.get(px, x0).astype(np.float64).copy(), b.get(py, y0).astype(np.float64).copy()
            b.call("cublas%srotm_v2" % P(dt), b.h, n, c_void_p(px), 2, c_void_p(py), -1, hp.ctypes.data_as(c_void_p))
            h11, h21, h12, h22 = {-1.0: (0.5, -0.25, 1.5, 2.0), 0.0: (1, -0.25, 1.5, 1), 1.0: (0.5, -1, 1, 2.0),
                                  -2.0: (1, 0, 0, 1)}[flag]
            a_, b_ = x1[ix].copy(), y1[iy].copy()
            x1[ix], y1[iy] = h11 * a_ + h12 * b_, h21 * a_ + h22 * b_
            e = max(e, rel(b.get(px, x0), x1), rel(b.get(py, y0), y1))
        return e, tol(dt, 4)

    @case
    def rotg_rotmg(b, dt=dt):
        ct = ctype(dt)
        e = 0.0
        for a0, b0 in ((3.0, 4.0), (-5.0, 2.0), (0.0, 0.0), (1.0, -7.0)):
            a, bb, c, s = ct(a0), ct(b0), ct(), ct()
            b.call("cublas%srotg_v2" % P(dt), b.h, ctypes.byref(a), ctypes.byref(bb), ctypes.byref(c), ctypes.byref(s))
            r = np.hypot(a0, b0)
            if r == 0:
                e = max(e, abs(c.value - 1) + abs(s.value) + abs(a.value))
                continue
            r = np.copysign(r, a0 if abs(a0) > abs(b0) else b0)
            e = max(e, abs(a.value - r) / abs(r), abs(c.value - a0 / r), abs(s.value - b0 / r))
            # (c s; -s c) (a b)^T = (r 0)^T
            e = max(e, abs(-s.value * a0 + c.value * b0) / abs(r))
        for d1v, d2v, x1v, y1v in ((2.0, 3.0, 1.5, 0.5), (1.0, 1.0, 0.25, 2.0), (4.0, 0.5, -1.0, 3.0)):
            d1, d2, x1, y1 = ct(d1v), ct(d2v), ct(x1v), ct(y1v)
            prm = (ct * 5)()
            b.call("cublas%srotmg_v2" % P(dt), b.h, ctypes.byref(d1), ctypes.byref(d2), ctypes.byref(x1), ctypes.byref(y1), prm)
            flag = prm[0]
            H = {-1.0: [[prm[1], prm[3]], [prm[2], prm[4]]], 0.0: [[1, prm[3]], [prm[2], 1]],
                 1.0: [[prm[1], 1], [-1, prm[4]]], -2.0: [[1, 0], [0, 1]]}[flag]
            # H (sqrt(d1) x1, sqrt(d2) y1) in the scaled basis: the second component vanishes
            v = np.array(H) @ np.array([x1v, y1v])
            e = max(e, abs(v[1]) / max(abs(v[0]), 1e-30), abs(v[0] - x1.value) / abs(v[0]))
            # d1' x1'^2 + ... preserves the weighted norm: d1 x1^2 + d2 y1^2 = d1' (x1')^2
            e = max(e, abs(d1v * x1v ** 2 + d2v * y1v ** 2 - d1.value * x1.value ** 2) / (d1v * x1v ** 2 + d2v * y1v ** 2))
        return e, tol(dt, 16)

    for m, n, incx, incy in ((70, 50, 1, 1), (33, 97, -2, 3)):
        @case
        def ger(b, dt=dt, m=m, n=n, incx=incx, incy=incy):
            lda = m + 3
            A, Am = colmajor(m, n, lda, dt)
            x, ix, xv = vec(m, incx, dt)
            y, iy, yv = vec(n, incy, dt)
            pa, a0 = b.dev(A)
            px, _ = b.dev(x)
            py, _ = b.dev(y)
            b.call("cublas%sger_v2" % P(dt), b.h, m, n, ptr(0.75, ctype(dt))[0], c_void_p(px), incx, c_void_p(py), incy, c_void_p(pa), lda)
            want = a0.astype(np.float64).copy()
            want[:n, :m] += (0.75 * np.outer(xv, yv)).T
            return rel(b.get(pa, a0), want), tol(dt)

    for uplo in (LOWER, UPPER):
        @case
        def syr_syr2(b, dt=dt, uplo=uplo):
            n, lda = 45, 50
            A, Am = colmajor(n, n, lda, dt)
            x, ix, xv = vec(n, 1, dt)
            y, iy, yv = vec(n, -2, dt)
            pa, a0 = b.dev(A)
            px, _ = b.dev(x)
            py, _ = b.dev(y)
            b.call("cublas%ssyr_v2" % P(dt), b.h, uplo, n, ptr(0.5, ctype(dt))[0], c_void_p(px), 1, c_void_p(pa), lda)
            b.call("cublas%ssyr2_v2" % P(dt), b.h, uplo, n, ptr(-1.5, ctype(dt))[0], c_void_p(px), 1, c_void_p(py), -2, c_void_p(pa), lda)
            upd = 0.5 * np.outer(xv, xv) - 1.5 * (np.outer(xv, yv) + np.outer(yv, xv))
            mask = np.triu(np.ones((n, n))) if uplo == UPPER else np.tril(np.ones((n, n)))
            want = a0.astype(np.float64).copy()
            want[:n, :n] += (upd * mask).T
            return rel(b.get(pa, a0), want), tol(dt, 4)

        @case
        def symv_sbmv_spmv(b, dt=dt, uplo=uplo):
            n, lda, k = 61, 64, 5
            A, Am = colmajor(n, n, lda, dt)
            x, ix, xv = vec(n, 2, dt)
            e = 0.0
            for kind in ("symv", "sbmv", "spmv"):
                y, iy, yv = vec(n, -1, dt)
                py, y0 = b.dev(y)
                px, _ = b.dev(x)
                if kind == "symv":
                    S_ = sym_of(Am, uplo)
                    pa, _ = b.dev(A)
                    b.call("cublas%ssymv_v2" % P(dt), b.h, uplo, n, ptr(1.25, ctype(dt))[0], c_void_p(pa), lda, c_void_p(px), 2,
                           ptr(0.5, ctype(dt))[0], c_void_p(py), -1)
                elif kind == "sbmv":
                    S_ = banded(sym_of(Am, uplo), k, k)
                    pa, _ = b.dev(tri_band_store(S_, uplo, k, k + 2, dt))
                    b.call("cublas%ssbmv_v2" % P(dt), b.h, uplo, n, k, ptr(1.25, ctype(dt))[0], c_void_p(pa), k + 2, c_void_p(px), 2,
                           ptr(0.5, ctype(dt))[0], c_void_p(py), -1)
                else:
                    S_ = sym_of(Am, uplo)
                    pa, _ = b.dev(packed_store(S_, uplo, dt))
                    b.call("cublas%sspmv_v2" % P(dt), b.h, uplo, n, ptr(1.25, ctype(dt))[0], c_void_p(pa), c_void_p(px), 2,
                           ptr(0.5, ctype(dt))[0], c_void_p(py), -1)
                S_ = S_.astype(dt).astype(np.float64)
                want = y0.astype(np.float64).copy()
                want[iy] = 1.25 * S_ @ xv + 0.5 * yv
                e = max(e, rel(b.get(py, y0), want))
            return e, tol(dt, n)

        for trans in (N, T):
            for diag in (0, 1):
                @case
                def trmv_trsv(b, dt=dt, uplo=uplo, trans=trans, diag=diag):
                    n, lda, k = 57, 60, 4
                    A, Am = colmajor(n, n, lda, dt)
                    A *= dt(2.0 / n)  # well conditioned, with the unit diagonal too
                    for i in range(n):
                        A[i, i] += 1
                    Am = A.astype(np.float64)[:n, :n].T
                    e = 0.0
                    for kind in ("tr", "tb", "tp"):
                        Tm = tri_of(Am, uplo, diag)
                        if kind == "tb":
                            Tm = banded(Tm, k, k)
                            store, lda_ = tri_band_store(Tm, uplo, k, k + 1, dt), k + 1
                        elif kind == "tp":
                            store, lda_ = packed_store(Tm, uplo, dt), None
                        else:
                            store, lda_ = A, lda
                        op = Tm if trans == N else Tm.T
                        x, ix, xv = vec(n, -3, dt)
                        pa, _ = b.dev(store)
                        for nm in ("mv", "sv"):
                            px, x0 = b.dev(x)
                            args = (b.h, uplo, trans, diag, n) + ((k,) if kind == "tb" else ()) + (c_void_p(pa),) + \
                                (() if kind == "tp" else (lda_,)) + (c_void_p(px), -3)
                            b.call("cublas%s%s%s_v2" % (P(dt), kind, nm), *args)
                            want = x0.astype(np.float64).copy()
                            want[ix] = op @ xv if nm == "mv" else np.linalg.solve(op, xv)
                            e = max(e, rel(b.get(px, x0), want))
                    return e, tol(dt, n)

    for trans, m, n, kl, ku in ((N, 50, 40, 3, 6), (T, 37, 64, 0, 2)):
        @case
        def gbmv(b, dt=dt, trans=trans, m=m, n=n, kl=kl, ku=ku):
            M = banded(rng.standard_normal((m, n)), kl, ku).astype(dt).astype(np.float64)
            lda = kl + ku + 2
            pa, _ = b.dev(band_store(M, kl, ku, lda, dt))
            xs, ys = (n, m) if trans == N else (m, n)
            x, ix, xv = vec(xs, 1, dt)
            y, iy, yv = vec(ys, 2, dt)
            px, _ = b.dev(x)
            py, y0 = b.dev(y)
            b.call("cublas%sgbmv_v2" % P(dt), b.h, trans, m, n, kl, ku, ptr(2.0, ctype(dt))[0], c_void_p(pa), lda, c_void_p(px), 1,
                   ptr(-1.0, ctype(dt))[0], c_void_p(py), 2)
            want = y0.astype(np.float64).copy()
            want[iy] = 2.0 * (M if trans == N else M.T) @ xv - yv
            return rel(b.get(py, y0), want), tol(dt, n)

    for trans in (N, T):
        @case
        def gemv_strided_batched(b, dt=dt, trans=trans):
            m, n, lda, batch = 40, 75, 41, 3
            sa = lda * n + 7
            A = rng.standard_normal(sa * batch).astype(dt)
            xs, ys = (n, m) if trans == N else (m, n)
            sx, sy = 2 * xs + 1, ys + 3
            x = rng.standard_normal(sx * batch).astype(dt)
            y = rng.standard_normal(sy * batch).astype(dt)
            pa, _ = b.dev(A)
            px, _ = b.dev(x)
            py, y0 = b.dev(y)
            b.call("cublas%sgemvStridedBatched" % P(dt), b.h, trans, m, n, ptr(1.0, ctype(dt))[0], c_void_p(pa), lda, c_int64(sa),
                   c_void_p(px), 2, c_int64(sx), ptr(0.25, ctype(dt))[0], c_void_p(py), 1, c_int64(sy), batch)
            want = y0.astype(np.float64).copy()
            for q in range(batch):
                Am = A[q * sa:q * sa + lda * n].astype(np.float64).reshape(n, lda)[:, :m].T
                op = Am if trans == N else Am.T
                xv = x[q * sx:q * sx + 2 * xs:2].astype(np.float64)
                want[q * sy:q * sy + ys] = op @ xv + 0.25 * want[q * sy:q * sy + ys]
            # gemvBatched on the same data through pointer arrays (device memory)
            pyb, y1 = b.dev(y)
            arr = lambda base, stride, nbytes: b.dev(np.array([base + q * stride * nbytes for q in range(batch)], np.uint64))[0]
            sz = np.dtype(dt).itemsize
            b.call("cublas%sgemvBatched" % P(dt), b.h, trans, m, n, ptr(1.0, ctype(dt))[0], c_void_p(arr(pa, sa, sz)), lda,
                   c_void_p(arr(px, sx, sz)), 2, ptr(0.25, ctype(dt))[0], c_void_p(arr(pyb, sy, sz)), 1, batch)
            return max(rel(b.get(py, y0), want), rel(b.get(pyb, y1), want)), tol(dt, n)

    for side in (LEFT, RIGHT):
        for uplo in (LOWER, UPPER):
            @case
            def symm(b, dt=dt, side=side, uplo=uplo):
                m, n = 40, 24
                k = m if side == LEFT else n
                A, Am = colmajor(k, k, k + 1, dt)
                B, Bm = colmajor(m, n, m + 2, dt)
                C, Cm = colmajor(m, n, m, dt)
                pa, _ = b.dev(A)
                pb, _ = b.dev(B)
                pc, c0 = b.dev(C)
                b.call("cublas%ssymm_v2" % P(dt), b.h, side, uplo, m, n, ptr(0.5, ctype(dt))[0], c_void_p(pa), k + 1, c_void_p(pb),
                       m + 2, ptr(2.0, ctype(dt))[0], c_void_p(pc), m)
                S_ = sym_of(Am, uplo)
                want = 0.5 * (S_ @ Bm if side == LEFT else Bm @ S_) + 2.0 * Cm
                return rel(b.get(pc, c0).astype(np.float64)[:n, :m].T, want), tol(dt, k)

            for trans in (N, T):
                for diag in (0, 1):
                    @case
                    def trmm_trsm(b, dt=dt, side=side, uplo=uplo, trans=trans, diag=diag):
                        m, n = 33, 20
                        k = m if side == LEFT else n
                        A, Am = colmajor(k, k, k + 3, dt)
                        A *= dt(2.0 / k)
                        for i in range(k):
                            A[i, i] += 1
                        Am = A.astype(np.float64)[:k, :k].T
                        Tm = tri_of(Am, uplo, diag)
                        op = Tm if trans == N else Tm.T
                        B, Bm = colmajor(m, n, m + 1, dt)
                        pa, _ = b.dev(A)
                        pb, b0 = b.dev(B)
                        C, _ = colmajor(m, n, m + 4, dt)
                        pc, c0 = b.dev(C)
                        b.call("cublas%strmm_v2" % P(dt), b.h, side, uplo, trans, diag, m, n, ptr(1.5, ctype(dt))[0], c_void_p(pa),
                               k + 3, c_void_p(pb), m + 1, c_void_p(pc), m + 4)
                        want = 1.5 * (op @ Bm if side == LEFT else Bm @ op)
                        e = rel(b.get(pc, c0).astype(np.float64)[:n, :m].T, want)
                        # in place (C = B)
                        b.call("cublas%strmm_v2" % P(dt), b.h, side, uplo, trans, diag, m, n, ptr(1.5, ctype(dt))[0], c_void_p(pa),
                               k + 3, c_void_p(pb), m + 1, c_void_p(pb), m + 1)
                        e = max(e, rel(b.get(pb, b0).astype(np.float64)[:n, :m].T, want))
                        # trsm undoes it: op(A) X = (1 / 1.5) B'
                        b.call("cublas%strsm_v2" % P(dt), b.h, side, uplo, trans, diag, m, n, ptr(1 / 1.5, ctype(dt))[0], c_void_p(pa),
                               k + 3, c_void_p(pb), m + 1)
                        e = max(e, rel(b.get(pb, b0).astype(np.float64)[:n, :m].T, Bm))
                        return e, tol(dt, k) * 4

    for uplo in (LOWER, UPPER):
        for trans in (N, T):
            @case
            def syrk_syr2k_syrkx(b, dt=dt, uplo=uplo, trans=trans):
                n, k = 48, 29
                r, c = (n, k) if trans == N else (k, n)
                A, Am = colmajor(r, c, r + 1, dt)
                B, Bm = colmajor(r, c, r + 2, dt)
                opA, opB = (Am, Bm) if trans == N else (Am.T, Bm.T)
                pa, _ = b.dev(A)
                pb, _ = b.dev(B)
                mask = np.triu(np.ones((n, n))) if uplo == UPPER else np.tril(np.ones((n, n)))
                e = 0.0
                for nm, upd, beta in (("syrk_v2", opA @ opA.T, 0.5), ("syr2k_v2", opA @ opB.T + opB @ opA.T, -1.0),
                                      ("syrkx", opA @ opB.T, 0.0)):
                    C, Cm = colmajor(n, n, n + 3, dt)
                    pc, c0 = b.dev(C)
                    args = (b.h, uplo, trans, n, k, ptr(0.75, ctype(dt))[0], c_void_p(pa), r + 1) + \
                        (() if nm == "syrk_v2" else (c_void_p(pb), r + 2)) + (ptr(beta, ctype(dt))[0], c_void_p(pc), n + 3)
                    b.call("cublas%s%s" % (P(dt), nm), *args)
                    want = c0.astype(np.float64).copy()
                    want[:n, :n] = np.where(mask, 0.75 * upd + beta * Cm, Cm).T
                    e = max(e, rel(b.get(pc, c0), want))
                return e, tol(dt, k)

    for ta, tb in ((N, N), (T, N), (N, T), (T, T)):
        @case
        def geam(b, dt=dt, ta=ta, tb=tb):
            m, n = 37, 53
            A, Am = colmajor(*((m, n) if ta == N else (n, m)), (m if ta == N else n) + 1, dt)
            B, Bm = colmajor(*((m, n) if tb == N else (n, m)), (m if tb == N else n) + 2, dt)
            pa, _ = b.dev(A)
            pb, _ = b.dev(B)
            C, _ = colmajor(m, n, m + 1, dt)
            pc, c0 = b.dev(C)
            b.call("cublas%sgeam" % P(dt), b.h, ta, tb, m, n, ptr(0.5, ctype(dt))[0], c_void_p(pa), A.shape[1], ptr(-2.0, ctype(dt))[0],
                   c_void_p(pb), B.shape[1], c_void_p(pc), m + 1)
            want = 0.5 * (Am if ta == N else Am.T) - 2.0 * (Bm if tb == N else Bm.T)
            e = rel(b.get(pc, c0).astype(np.float64)[:n, :m].T, want)
            # beta 0: B is not read (a null pointer)
            b.call("cublas%sgeam" % P(dt), b.h, ta, tb, m, n, ptr(0.5, ctype(dt))[0], c_void_p(pa), A.shape[1], ptr(0.0, ctype(dt))[0],
                   c_void_p(0), B.shape[1], c_void_p(pc), m + 1)
            e = max(e, rel(b.get(pc, c0).astype(np.float64)[:n, :m].T, 0.5 * (Am if ta == N else Am.T)))
            return e, tol(dt)

    for side in (LEFT, RIGHT):
        @case
        def dgmm(b, dt=dt, side=side):
            m, n = 30, 44
            A, Am = colmajor(m, n, m + 2, dt)
            x, ix, xv = vec(m if side == LEFT else n, -2, dt)
            pa, _ = b.dev(A)
            px, _ = b.dev(x)
            C, _ = colmajor(m, n, m, dt)
            pc, c0 = b.dev(C)
            b.call("cublas%sdgmm" % P(dt), b.h, side, m, n, c_void_p(pa), m + 2, c_void_p(px), -2, c_void_p(pc), m)
            want = (np.diag(xv) @ Am) if side == LEFT else (Am @ np.diag(xv))
            return rel(b.get(pc, c0).astype(np.float64)[:n, :m].T, want), tol(dt)


@case
def argument_checks(b):
    """CUBLAS_STATUS_INVALID_VALUE (7) for bad enums and dimensions"""
    f = b.l
    one = ctypes.c_float(1.0)
    bad = [
        f.cublasSgemv_v2(b.h, 5, 4, 4, ctypes.byref(one), None, 4, None, 1, ctypes.byref(one), None, 1),
        f.cublasSgemv_v2(b.h, N, 4, 4, ctypes.byref(one), None, 3, None, 1, ctypes.byref(one), None, 1),
        f.cublasStrsv_v2(b.h, 2, N, 0, 4, None, 4, None, 1),
        f.cublasStrsm_v2(b.h, LEFT, LOWER, N, 3, 4, 4, ctypes.byref(one), None, 4, None, 4),
        f.cublasSsyrk_v2(b.h, UPPER, N, 4, 4, ctypes.byref(one), None, 3, ctypes.byref(one), None, 4),
        f.cublasSger_v2(b.h, 4, 4, ctypes.byref(one), None, 0, None, 1, None, 4),
        f.cublasSgeam(b.h, N, N, 4, 4, ctypes.byref(one), None, 2, ctypes.byref(one), None, 4, None, 4),
        f.cublasSdgmm(b.h, 2, 4, 4, None, 4, None, 1, None, 4),
        f.cublasSgbmv_v2(b.h, N, 4, 4, 1, 1, ctypes.byref(one), None, 2, None, 1, ctypes.byref(one), None, 1),
    ]
    return sum(r != 7 for r in bad), 0


# ------------------------------------------------------------------------------------------- complex
CPX = {np.complex64: ("C", ctypes.c_float, np.float32), np.complex128: ("Z", ctypes.c_double, np.float64)}


def cstrided(n, inc, dt):
    st = (rng.standard_normal(max(1, 1 + (n - 1) * abs(inc))) + 1j * rng.standard_normal(max(1, 1 + (n - 1) * abs(inc)))).astype(dt)
    idx = np.arange(n) * abs(inc)
    if inc < 0:
        idx = idx[::-1]
    return st, idx


def cptr(v, rt):
    x = (rt * 2)(v.real, v.imag)
    return ctypes.byref(x), x


def crel(got, want):
    return float(np.max(np.abs(np.asarray(got) - want)) / max(np.max(np.abs(want)), 1e-30))


def ctol(dt, k=1):
    return (4e-5 if dt == np.complex64 else 2e-12) * max(1, np.sqrt(k))


for dt in (np.complex64, np.complex128):
    @case
    def complex_level1(b, dt=dt):
        Pc, rt, real = CPX[dt]
        n = 513
        x, ix = cstrided(n, 2, dt)
        y, iy = cstrided(n, -3, dt)
        px, x0 = b.dev(x)
        py, y0 = b.dev(y)
        al = 0.5 - 1.25j
        b.call("cublas%saxpy_v2" % Pc, b.h, n, cptr(al, rt)[0], c_void_p(px), 2, c_void_p(py), -3)
        want = y0.astype(np.complex128).copy()
        want[iy] += al * x0.astype(np.complex128)[ix]
        e = crel(b.get(py, y0), want)
        yv = b.get(py, y0).astype(np.complex128)
        xv = x0.astype(np.complex128)
        r = (rt * 2)()
        for nm, wantd in (("dotu", np.sum(xv[ix] * yv[iy])), ("dotc", np.sum(np.conj(xv[ix]) * yv[iy]))):
            b.call("cublas%s%s_v2" % (Pc, nm), b.h, n, c_void_p(px), 2, c_void_p(py), -3, r)
            e = max(e, abs(complex(r[0], r[1]) - wantd) / np.sum(np.abs(xv[ix] * yv[iy])))
        rr = rt()
        nn = "Sc" if Pc == "C" else "Dz"
        for inc in (1, 2):
            sel = xv[::inc][:n // inc]
            b.call("cublas%snrm2_v2" % nn, b.h, n // inc, c_void_p(px), inc, ctypes.byref(rr))
            e = max(e, abs(rr.value - np.linalg.norm(sel)) / np.linalg.norm(sel))
            b.call("cublas%sasum_v2" % nn, b.h, n // inc, c_void_p(px), inc, ctypes.byref(rr))
            w = np.sum(np.abs(sel.real) + np.abs(sel.imag))
            e = max(e, abs(rr.value - w) / w)
        # scal, sscal, rot, swap, copy
        b.call("cublas%sscal_v2" % Pc, b.h, n, cptr(2 - 1j, rt)[0], c_void_p(px), 2)
        xv2 = xv.copy()
        xv2[ix] *= 2 - 1j
        b.call("cublas%s%sscal_v2" % (Pc, "s" if Pc == "C" else "d"), b.h, n, ctypes.byref(rt(-0.5)), c_void_p(px), 2)
        xv2[ix] *= -0.5
        e = max(e, crel(b.get(px, x0), xv2))
        c_, s_ = 0.8, 0.6
        b.call("cublas%srot_v2" % ("Cs" if Pc == "C" else "Zd"), b.h, n, c_void_p(px), 2, c_void_p(py), -3,
               ctypes.byref(rt(c_)), ctypes.byref(rt(s_)))
        wx, wy = xv2.copy(), yv.copy()
        wx[ix], wy[iy] = c_ * xv2[ix] + s_ * yv[iy], c_ * yv[iy] - s_ * xv2[ix]
        e = max(e, crel(b.get(px, x0), wx), crel(b.get(py, y0), wy))
        b.call("cublas%sswap_v2" % Pc, b.h, n, c_void_p(px), 2, c_void_p(py), -3)
        wx2, wy2 = wx.copy(), wy.copy()
        wx2[ix], wy2[iy] = wy[iy], wx[ix]
        b.call("cublas%scopy_v2" % Pc, b.h, n, c_void_p(py), -3, c_void_p(px), 2)
        wx2[ix] = wy2[iy]
        e = max(e, crel(b.get(px, x0), wx2), crel(b.get(py, y0), wy2))
        return e, ctol(dt, n)

    for trans in (N, T, 2):
        @case
        def complex_gemv(b, dt=dt, trans=trans):
            Pc, rt, real = CPX[dt]
            m, n, lda = 45, 70, 47
            A = (rng.standard_normal((n, lda)) + 1j * rng.standard_normal((n, lda))).astype(dt)
            Am = A.astype(np.complex128)[:, :m].T
            op = {N: Am, T: Am.T, 2: Am.conj().T}[trans]
            x, ix = cstrided(op.shape[1], 1, dt)
            y, iy = cstrided(op.shape[0], -2, dt)
            pa, _ = b.dev(A)
            px, _ = b.dev(x)
            py, y0 = b.dev(y)
            al, be = 1 + 0.5j, -0.25 + 1j
            b.call("cublas%sgemv_v2" % Pc, b.h, trans, m, n, cptr(al, rt)[0], c_void_p(pa), lda, c_void_p(px), 1,
                   cptr(be, rt)[0], c_void_p(py), -2)
            want = y0.astype(np.complex128).copy()
            want[iy] = al * op @ x.astype(np.complex128)[ix] + be * want[iy]
            return crel(b.get(py, y0), want), ctol(dt, n)

    for ta, tb, m, n, k, beta in ((N, N, 40, 33, 20, 0j), (2, T, 17, 64, 50, 1 - 1j), (T, 2, 64, 48, 32, 0.5j)):
        @case
        def complex_gemm(b, dt=dt, ta=ta, tb=tb, m=m, n=n, k=k, beta=beta):
            Pc, rt, real = CPX[dt]
            ra, ca = (m, k) if ta == N else (k, m)
            rb, cb = (k, n) if tb == N else (n, k)
            A = (rng.standard_normal((ca, ra + 1)) + 1j * rng.standard_normal((ca, ra + 1))).astype(dt)
            B = (rng.standard_normal((cb, rb + 2)) + 1j * rng.standard_normal((cb, rb + 2))).astype(dt)
            C = (rng.standard_normal((n, m + 3)) + 1j * rng.standard_normal((n, m + 3))).astype(dt)
            pa, _ = b.dev(A)
            pb, _ = b.dev(B)
            pc, c0 = b.dev(C)
            al = 0.5 + 2j
            for nm in ("gemm_v2", "gemm3m"):
                pc, c0 = b.dev(C)
                b.call("cublas%s%s" % (Pc, nm), b.h, ta, tb, m, n, k, cptr(al, rt)[0], c_void_p(pa), ra + 1, c_void_p(pb), rb + 2,
                       cptr(beta, rt)[0], c_void_p(pc), m + 3)
            opf = lambda M, t: {N: M, T: M.T, 2: M.conj().T}[t]
            Am = A.astype(np.complex128)[:, :ra].T
            Bm = B.astype(np.complex128)[:, :rb].T
            want = c0.astype(np.complex128).copy()
            want[:, :m] = (al * opf(Am, ta) @ opf(Bm, tb) + beta * want[:, :m].T).T
            return crel(b.get(pc, c0), want), ctol(dt, k)

    @case
    def complex_gemm_strided_batched(b, dt=dt):
        Pc, rt, real = CPX[dt]
        m, n, k, batch = 20, 18, 9, 3
        A = (rng.standard_normal((batch, k, m)) + 1j * rng.standard_normal((batch, k, m))).astype(dt)
        B = (rng.standard_normal((batch, n, k)) + 1j * rng.standard_normal((batch, n, k))).astype(dt)
        C = (rng.standard_normal((batch, n, m)) + 1j * rng.standard_normal((batch, n, m))).astype(dt)
        pa, _ = b.dev(A)
        pb, _ = b.dev(B)
        pc, c0 = b.dev(C)
        b.call("cublas%sgemmStridedBatched" % Pc, b.h, N, N, m, n, k, cptr(1j, rt)[0], c_void_p(pa), m, c_int64(m * k),
               c_void_p(pb), k, c_int64(k * n), cptr(1 + 0j, rt)[0], c_void_p(pc), m, c_int64(m * n), batch)
        want = np.stack([(1j * A[i].T.astype(np.complex128) @ B[i].T + C[i].T).T for i in range(batch)])
        return crel(b.get(pc, c0), want), ctol(dt, k)


@case
def misc_api_and_64(b):
    """GetProperty, atomics mode, the Async copies, the _64 entry points"""
    f = b.l
    e = 0.0
    v = c_int()
    for kind, want in ((0, 12), (1, 4), (2, 0)):
        b.ok(f.cublasGetProperty(kind, ctypes.byref(v)), "cublasGetProperty")
        e += v.value != want
    b.ok(f.cublasSetAtomicsMode(b.h, 1), "cublasSetAtomicsMode")
    b.ok(f.cublasGetAtomicsMode(b.h, ctypes.byref(v)), "cublasGetAtomicsMode")
    e += v.value != 1
    e += f.cublasSetAtomicsMode(b.h, 5) != 7
    b.ok(f.cublasSetAtomicsMode(b.h, 0), "cublasSetAtomicsMode")
    n = 100
    x = rng.standard_normal(2 * n).astype(np.float32)
    p = c_void_p(b.cu.alloc(bytes(4 * n)))
    b.ok(f.cublasSetVectorAsync(n, 4, x.ctypes.data_as(c_void_p), 2, p, 1, None), "cublasSetVectorAsync")
    out = np.zeros(n, np.float32)
    b.ok(f.cublasGetVectorAsync_64(c_int64(n), 4, p, c_int64(1), out.ctypes.data_as(c_void_p), c_int64(1), None), "GetVectorAsync_64")
    e += rel(out, x[::2])
    m2 = np.zeros((5, 3), np.float32)
    b.ok(f.cublasGetMatrixAsync(3, 5, 4, p, 10, m2.ctypes.data_as(c_void_p), 3, None), "cublasGetMatrixAsync")
    e += rel(m2.ravel(), np.concatenate([x[::2][10 * j:10 * j + 3] for j in range(5)]))
    # _64 compute routines
    y = rng.standard_normal(n).astype(np.float32)
    py, y0 = b.dev(y)
    al = ctypes.c_float(2.0)
    b.ok(f.cublasSaxpy_v2_64(b.h, c_int64(n), ctypes.byref(al), p, c_int64(1), py, c_int64(1)), "cublasSaxpy_v2_64")
    e += rel(b.get(py, y0), y.astype(np.float64) + 2.0 * x[::2])
    i64 = c_int64()
    b.ok(f.cublasIsamax_v2_64(b.h, c_int64(n), p, c_int64(1), ctypes.byref(i64)), "cublasIsamax_v2_64")
    e += i64.value != int(np.argmax(np.abs(x[::2]))) + 1
    A = rng.standard_normal((8, 8)).astype(np.float32)
    pa, _ = b.dev(A)
    pc, c0 = b.dev(np.zeros((8, 8), np.float32))
    zero = ctypes.c_float(0.0)
    b.ok(f.cublasSgemm_v2_64(b.h, N, N, c_int64(8), c_int64(8), c_int64(8), ctypes.byref(al), c_void_p(pa), c_int64(8),
                             c_void_p(pa), c_int64(8), ctypes.byref(zero), c_void_p(pc), c_int64(8)), "cublasSgemm_v2_64")
    e += rel(b.get(pc, c0), (2.0 * A.T.astype(np.float64) @ A.T).T)
    e += f.cublasSscal_v2_64(b.h, c_int64(1 << 40), ctypes.byref(al), p, c_int64(1)) != 15
    return e, 1e-5


def f16(x):
    return x.astype(np.float16)


def bf16_bits(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_val(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


for at, ct, m, n, k in (("f16", "f16", 72, 40, 96), ("f16", "f32", 72, 40, 96), ("bf16", "bf16", 72, 40, 96),
                        ("bf16", "f32", 72, 40, 96), ("f16", "f32", 64, 48, 32), ("bf16", "f32", 32, 16, 64)):
    @case
    def gemm_ex(b, at=at, ct=ct, m=m, n=n, k=k):
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
