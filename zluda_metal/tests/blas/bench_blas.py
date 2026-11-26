#!/usr/bin/env python3
"""Predicted times of representative cuBLAS calls (src/blas.rs) on the AGX Performance Simulator.

Each case runs one cuBLAS call with ZLUDA_AGX_STATS set and sums the predicted time (time_us) of the launches it
made (one JSON line per launch). Results are not checked here (tests/blas/test_blas.py does that).

  bench_blas.py [--lib LIB] [-k SUBSTR] [--json OUT]
"""
import argparse
import ctypes
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ptxconf"))
import ptxconf  # noqa: E402

c_void_p = ctypes.c_void_p
STATS = os.path.join(tempfile.mkdtemp(prefix="zluda-blas-bench-"), "stats.jsonl")
os.environ["ZLUDA_AGX_STATS"] = STATS
rng = np.random.default_rng(7)


def cases():
    def gemm(dt, m, n, k, ta=0, tb=0):
        def run(b):
            lda = m if ta == 0 else k
            ldb = k if tb == 0 else n
            pa = b.dev(rng.standard_normal(m * k).astype(dt))
            pb = b.dev(rng.standard_normal(k * n).astype(dt))
            pc = b.dev(np.zeros(m * n, dt))
            ct = ctypes.c_float if dt == np.float32 else ctypes.c_double
            al, be = ct(1.0), ct(0.0)
            f = "cublasSgemm_v2" if dt == np.float32 else "cublasDgemm_v2"
            return getattr(b.l, f)(b.h, ta, tb, m, n, k, ctypes.byref(al), c_void_p(pa), lda, c_void_p(pb), ldb,
                                   ctypes.byref(be), c_void_p(pc), m)
        return ("%sgemm %dx%dx%d" % ("s" if dt == np.float32 else "d", m, n, k), run)

    def hgemm(m, n, k):
        def run(b):
            pa = b.dev(rng.standard_normal(m * k).astype(np.float16))
            pb = b.dev(rng.standard_normal(k * n).astype(np.float16))
            pc = b.dev(np.zeros(m * n, np.float16))
            one = ctypes.c_uint16(0x3C00)
            zero = ctypes.c_uint16(0)
            return b.l.cublasHgemm(b.h, 0, 0, m, n, k, ctypes.byref(one), c_void_p(pa), m, c_void_p(pb), k,
                                   ctypes.byref(zero), c_void_p(pc), m)
        return ("hgemm %dx%dx%d" % (m, n, k), run)

    def gemv(dt, m, n, t):
        def run(b):
            pa = b.dev(rng.standard_normal(m * n).astype(dt))
            px = b.dev(rng.standard_normal(n if t == 0 else m).astype(dt))
            py = b.dev(np.zeros(m if t == 0 else n, dt))
            ct = ctypes.c_float if dt == np.float32 else ctypes.c_double
            al, be = ct(1.0), ct(0.0)
            f = "cublasSgemv_v2" if dt == np.float32 else "cublasDgemv_v2"
            return getattr(b.l, f)(b.h, t, m, n, ctypes.byref(al), c_void_p(pa), m, c_void_p(px), 1,
                                   ctypes.byref(be), c_void_p(py), 1)
        return ("%sgemv %s %dx%d" % ("s" if dt == np.float32 else "d", "NT"[t], m, n), run)

    def dot(n):
        def run(b):
            px = b.dev(rng.standard_normal(n).astype(np.float32))
            py = b.dev(rng.standard_normal(n).astype(np.float32))
            r = ctypes.c_float()
            return b.l.cublasSdot_v2(b.h, n, c_void_p(px), 1, c_void_p(py), 1, ctypes.byref(r))
        return ("sdot %d" % n, run)

    def nrm2(n):
        def run(b):
            px = b.dev(rng.standard_normal(n).astype(np.float32))
            r = ctypes.c_float()
            return b.l.cublasSnrm2_v2(b.h, n, c_void_p(px), 1, ctypes.byref(r))
        return ("snrm2 %d" % n, run)

    def amax(n):
        def run(b):
            px = b.dev(rng.standard_normal(n).astype(np.float32))
            r = ctypes.c_int()
            return b.l.cublasIsamax_v2(b.h, n, c_void_p(px), 1, ctypes.byref(r))
        return ("isamax %d" % n, run)

    out = []
    for s in (256, 512, 1024):
        out.append(gemm(np.float32, s, s, s))
    out += [gemm(np.float32, 250, 250, 250), gemm(np.float32, 500, 500, 500), gemm(np.float32, 1000, 1000, 1000),
            gemm(np.float32, 255, 257, 129), gemm(np.float64, 128, 128, 128), hgemm(256, 256, 256),
            hgemm(512, 512, 512), gemv(np.float32, 1000, 1000, 0), gemv(np.float32, 1000, 1000, 1),
            gemv(np.float64, 1000, 1000, 0), dot(1000000), nrm2(1000000), amax(1000000)]
    return out


class B:
    def __init__(self, lib):
        os.environ["ZLUDA_METAL_BACKEND"] = "agx"
        self.cu = ptxconf.Cuda(lib)
        self.l = self.cu.l
        self.h = c_void_p()
        assert self.l.cublasCreate_v2(ctypes.byref(self.h)) == 0

    def dev(self, a):
        return self.cu.alloc(np.ascontiguousarray(a).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.path.join(HERE, "..", "..", "target", "release", "libzluda_metal.so"))
    ap.add_argument("-k")
    ap.add_argument("--json")
    a = ap.parse_args()
    b = B(a.lib)
    res = {}
    for name, run in cases():
        if a.k and a.k not in name:
            continue
        open(STATS, "w").close()
        st = run(b)
        lines = [json.loads(x) for x in open(STATS)]
        t = sum(x["time_us"] or 0 for x in lines)
        res[name] = t
        print("%-24s %10.1f us  (%d launches: %s)%s" % (name, t, len(lines), ", ".join(sorted({x["kernel"] for x in lines})),
                                                         "" if st == 0 else "  status %d" % st), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


main()
