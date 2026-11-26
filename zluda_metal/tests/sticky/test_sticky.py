#!/usr/bin/env python3
"""Sticky kernel faults, as CUDA has them (src/cuda.rs): a kernel that faults (an illegal address, a trap, a failed
assert, a hang) launches successfully, and from then on every call that uses its context returns the fault's error;
destroying the context (or resetting the primary one) clears it. Launch errors (an impossible block size) are not
sticky. Each case runs in its own process on the AGX backend.

  test_sticky.py [--lib LIB] [-v]
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ptxconf"))

HDR = ".version 8.0\n.target sm_80\n.address_size 64\n"
KERNELS = {
    "oob": HDR + """.visible .entry k(.param .u64 p) {
    .reg .b64 %rd<2>; .reg .b32 %r<2>;
    mov.u64 %rd1, 16; mov.u32 %r1, 7; st.global.u32 [%rd1], %r1; ret; }""",
    "trap": HDR + """.visible .entry k(.param .u64 p) {
    .reg .b64 %rd<2>; .reg .b32 %r<2>; .reg .pred %p<2>;
    ld.param.u64 %rd1, [p]; mov.u32 %r1, %tid.x; setp.eq.u32 %p1, %r1, 3;
    @%p1 trap; st.global.u32 [%rd1], %r1; ret; }""",
    "assert": HDR + """.extern .func __assertfail(.param .b64 a, .param .b64 b, .param .b32 c, .param .b64 d, .param .b64 e);
.global .align 1 .b8 msg[11] = {120, 32, 62, 32, 53, 32, 38, 38, 32, 49, 0};
.global .align 1 .b8 file[8] = {107, 101, 114, 110, 46, 99, 117, 0};
.global .align 1 .b8 fn[7] = {107, 101, 114, 110, 101, 108, 0};
.visible .entry k(.param .u64 p) {
    .reg .b64 %rd<5>; .reg .b32 %r<3>; .reg .pred %p<2>;
    mov.u32 %r1, %tid.x; setp.ne.u32 %p1, %r1, 5; @%p1 bra DONE;
    mov.u64 %rd1, msg; mov.u64 %rd2, file; mov.u64 %rd3, fn; mov.u32 %r2, 42;
    { .param .b64 a; .param .b64 b; .param .b32 c; .param .b64 d; .param .b64 e;
      st.param.b64 [a], %rd1; st.param.b64 [b], %rd2; st.param.b32 [c], %r2; st.param.b64 [d], %rd3;
      st.param.b64 [e], 1;
      call.uni __assertfail, (a, b, c, d, e); }
    trap;
DONE: ret; }""",
    "hang": HDR + """.visible .entry k(.param .u64 p) {
    .reg .b32 %r<2>;
    mov.u32 %r1, 0;
LOOP: add.u32 %r1, %r1, 1; bra LOOP; }""",
    "good": HDR + """.visible .entry k(.param .u64 p) {
    .reg .b64 %rd<3>; .reg .b32 %r<2>;
    ld.param.u64 %rd1, [p]; mov.u32 %r1, %tid.x; mul.wide.u32 %rd2, %r1, 4; add.s64 %rd1, %rd1, %rd2;
    st.global.u32 [%rd1], %r1; ret; }""",
}


def child(case, lib):
    """one case in this process: a JSON line of the results"""
    os.environ["ZLUDA_METAL_BACKEND"] = "agx"
    import ptxconf
    cu = ptxconf.Cuda(lib)
    l = cu.l
    for f in ("cuCtxDestroy_v2", "cuCtxCreate_v2", "cuDevicePrimaryCtxRetain", "cuCtxSetCurrent",
              "cuDevicePrimaryCtxReset"):
        getattr(l, f)
    out = {}

    def load(src):
        mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
        r = l.cuModuleLoadData(ctypes.byref(mod), src.encode() + b"\0")
        assert r == 0, ("load", r, cu.last_error())
        assert l.cuModuleGetFunction(ctypes.byref(fn), mod, b"k") == 0
        return fn

    def launch(fn, block=32):
        buf = cu.alloc(bytes(4096))
        h = ctypes.c_uint64(buf)
        arr = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(h), ctypes.c_void_p))
        return l.cuLaunchKernel(fn, 1, 1, 1, block, 1, 1, 0, None, arr, None)

    p = ctypes.c_uint64()
    if case == "config":
        fn = load(KERNELS["good"])
        out["launch"] = launch(fn, block=2048)
        out["sync"] = l.cuCtxSynchronize()
        out["alloc"] = l.cuMemAlloc_v2(ctypes.byref(p), 64)
    elif case == "primary":
        # the primary context: faulted, then cleared by cuDevicePrimaryCtxReset
        ctx = ctypes.c_void_p()
        l.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), 0)
        l.cuCtxSetCurrent(ctx)
        fn = load(KERNELS["oob"])
        out["launch"] = launch(fn)
        out["sync"] = l.cuCtxSynchronize()
        out["reset"] = l.cuDevicePrimaryCtxReset(0)
        out["sync_after_reset"] = l.cuCtxSynchronize()
    else:
        fn = load(KERNELS[case])
        out["launch"] = launch(fn)
        out["sync"] = l.cuCtxSynchronize()
        out["alloc"] = l.cuMemAlloc_v2(ctypes.byref(p), 64)
        out["memcpy"] = l.cuMemcpyHtoD_v2(0, b"x" * 4, 4)
        name = ctypes.c_char_p()
        l.cuGetErrorName(out["sync"], ctypes.byref(name))
        out["name"] = (name.value or b"").decode()
        # a new context is clean
        cur = ctypes.c_void_p()
        l.cuCtxGetCurrent(ctypes.byref(cur))
        out["destroy"] = l.cuCtxDestroy_v2(cur)
        ctx = ctypes.c_void_p()
        out["create"] = l.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0)
        g = load(KERNELS["good"])
        out["good_launch"] = launch(g)
        out["good_sync"] = l.cuCtxSynchronize()
    print(json.dumps(out), flush=True)
    os._exit(0)


EXPECT = {
    "oob": {"launch": 0, "sync": 700, "alloc": 700, "memcpy": 700, "name": "CUDA_ERROR_ILLEGAL_ADDRESS"},
    "trap": {"launch": 0, "sync": 719, "alloc": 719, "memcpy": 719, "name": "CUDA_ERROR_LAUNCH_FAILED"},
    "assert": {"launch": 0, "sync": 710, "alloc": 710, "memcpy": 710, "name": "CUDA_ERROR_ASSERT"},
    "hang": {"launch": 0, "sync": 702, "alloc": 702, "memcpy": 702, "name": "CUDA_ERROR_LAUNCH_TIMEOUT"},
    "config": {"sync": 0, "alloc": 0},
    "primary": {"launch": 0, "sync": 700, "reset": 0, "sync_after_reset": 0},
}
CLEAN = {"destroy": 0, "create": 0, "good_launch": 0, "good_sync": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.path.join(HERE, "..", "..", "target", "release", "libzluda_metal.so"))
    ap.add_argument("--child")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    if a.child:
        child(a.child, a.lib)
    env = dict(os.environ, ZLUDA_METAL_BACKEND="agx", ZLUDA_AGX_MAX_STEPS="100000")
    sib = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "llvm-project", "build", "bin", "llvm-air2agx"))
    if os.path.exists(sib):
        env.setdefault("ZLUDA_AGX_AIR2AGX", sib)
    fails = 0
    for case, want in EXPECT.items():
        r = subprocess.run([sys.executable, __file__, "--child", case, "--lib", a.lib], capture_output=True, text=True,
                           env=env, timeout=600)
        line = [x for x in r.stdout.splitlines() if x.startswith("{")]
        got = json.loads(line[-1]) if line else {}
        w = dict(want, **(CLEAN if case in ("oob", "trap", "assert", "hang") else {}))
        if case == "config":
            ok = got.get("launch", 0) != 0 and all(got.get(k) == v for k, v in w.items())
        else:
            ok = all(got.get(k) == v for k, v in w.items())
        if case == "assert":
            ok = ok and "kern.cu:42: kernel: Assertion `x > 5 && 1` failed." in r.stderr
        fails += not ok
        print("%-5s %-8s %s" % ("PASS" if ok else "FAIL", case, json.dumps(got)), flush=True)
        if (a.v or not ok) and r.stderr.strip():
            print("      stderr:", r.stderr.strip()[-400:].replace("\n", " | "))
    print("---- %s" % ("PASS" if not fails else "FAIL %d" % fails))
    sys.exit(1 if fails else 0)


main()
