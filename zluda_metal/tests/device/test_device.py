#!/usr/bin/env python3
"""CUDA device runtime behavior on the AGX backend (src/device_rt.rs, ptx/lib/metal_impl/devrt.inc): printf (format
semantics, flush points, the FIFO size limit and its circular overwrite), device malloc / free (16-byte alignment,
reuse, persistence across launches, CU_LIMIT_MALLOC_HEAP_SIZE), clocks (%clock64 / %globaltimer monotonic,
__nanosleep advancing them), special register ranges, shared memory out of bounds (a sticky illegal address),
cooperative launch limits. Each case runs in its own process (its stdout is the device's printf output).

kernels.ptx is clang's output for kernels.cu (`--regen` rebuilds it:
clang++ -x cuda --cuda-device-only -nocudainc -nocudalib); other kernels are hand-written PTX below.

  test_device.py [--lib LIB] [-v] [-k CASE] [--regen]
"""
import argparse
import ctypes
import json
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ptxconf"))

HDR = ".version 8.0\n.target sm_80\n.address_size 64\n"
PTX = {
    # a shared array of 16 words indexed by %tid.x * 64 (out of bounds for every thread but 0)
    "smem_oob": HDR + """.visible .entry k(.param .u64 p) {
    .shared .align 4 .b32 s[16];
    .reg .b32 %r<4>; .reg .b64 %rd<3>;
    mov.u32 %r1, %tid.x; shl.b32 %r2, %r1, 14; mov.u32 %r3, s; add.u32 %r3, %r3, %r2;
    st.shared.u32 [%r3], %r1; bar.sync 0; ret; }""",
    # dynamic shared memory: its address modulo 16 (CUDA aligns it to at least 16 bytes) after a static .b8
    "dyn_align": HDR + """.extern .shared .align 16 .b8 dyn[];
.visible .entry k(.param .u64 p) {
    .shared .align 1 .b8 one[3];
    .reg .b32 %r<4>; .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [p]; mov.u32 %r1, dyn; and.b32 %r2, %r1, 15; mov.u32 %r3, one;
    st.shared.u8 [%r3], 1; st.shared.u32 [%r1], 5; ld.shared.u32 %r3, [%r1];
    st.global.u32 [%rd1], %r2; st.global.u32 [%rd1+4], %r3; ret; }""",
    # lane masks and warp ids of a 96-thread block (3 warps)
    "sregs": HDR + """.visible .entry k(.param .u64 p) {
    .reg .b32 %r<10>; .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [p]; mov.u32 %r1, %tid.x; mul.wide.u32 %rd2, %r1, 32; add.s64 %rd1, %rd1, %rd2;
    mov.u32 %r2, %lanemask_eq; mov.u32 %r3, %lanemask_lt; mov.u32 %r4, %lanemask_le; mov.u32 %r5, %lanemask_gt;
    mov.u32 %r6, %lanemask_ge; mov.u32 %r7, %warpid; mov.u32 %r8, %nwarpid; mov.u32 %r9, %laneid;
    st.global.v4.u32 [%rd1], {%r2, %r3, %r4, %r5}; st.global.v4.u32 [%rd1+16], {%r6, %r7, %r8, %r9}; ret; }""",
    # a grid-wide spin barrier (cooperative style): one block can pass it
    "coop": HDR + """.visible .entry k(.param .u64 p) {
    .reg .b32 %r<4>; .reg .b64 %rd<2>; .reg .pred %q;
    ld.param.u64 %rd1, [p]; bar.sync 0; mov.u32 %r1, %tid.x; setp.ne.u32 %q, %r1, 0; @%q bra W;
    atom.global.add.u32 %r2, [%rd1], 1;
W:  ld.volatile.global.u32 %r3, [%rd1]; mov.u32 %r1, %nctaid.x; setp.lt.u32 %q, %r3, %r1; @%q bra W;
    bar.sync 0; ret; }""",
    # __threadfence message passing between blocks: block 0 publishes data then a flag, block 1 waits for the flag
    "fence": HDR + """.visible .entry k(.param .u64 p) {
    .reg .b32 %r<4>; .reg .b64 %rd<2>; .reg .pred %q;
    ld.param.u64 %rd1, [p]; mov.u32 %r1, %tid.x; setp.ne.u32 %q, %r1, 0; @%q bra DONE;
    mov.u32 %r1, %ctaid.x; setp.ne.u32 %q, %r1, 0; @%q bra CONSUME;
    st.global.u32 [%rd1+8], 42; membar.gl; st.volatile.global.u32 [%rd1], 1; bra DONE;
CONSUME: ld.volatile.global.u32 %r2, [%rd1]; setp.eq.u32 %q, %r2, 0; @%q bra CONSUME;
    membar.gl; ld.global.u32 %r3, [%rd1+8]; st.global.u32 [%rd1+4], %r3;
DONE: ret; }""",
}


def child(case, lib):
    os.environ["ZLUDA_METAL_BACKEND"] = "agx"
    import ptxconf
    cu = ptxconf.Cuda(lib)
    l = cu.l
    l.cuCtxSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    l.cuCtxGetLimit.argtypes = [ctypes.c_void_p, ctypes.c_int]
    l.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    l.cuLaunchCooperativeKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 2
    out = {}

    def load(src):
        mod = ctypes.c_void_p()
        r = l.cuModuleLoadData(ctypes.byref(mod), src.encode() + b"\0")
        assert r == 0, ("load", r, cu.last_error())
        return mod

    def fn(mod, name):
        f = ctypes.c_void_p()
        assert l.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()) == 0, name
        return f

    def launch(f, grid, block, *args, shared=0, coop=False):
        hold = []
        for a in args:
            # ctypes values as given; Python ints: small ones are int parameters, others pointers
            hold.append(a if not isinstance(a, int) else ctypes.c_int32(a) if -2**31 <= a < 2**20 else
                        ctypes.c_uint64(a))
        arr = (ctypes.c_void_p * max(1, len(hold)))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in hold])
        if coop:
            return l.cuLaunchCooperativeKernel(f, grid, 1, 1, block, 1, 1, shared, None, arr)
        return l.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, shared, None, arr, None)

    def limit(which):
        v = ctypes.c_size_t()
        r = l.cuCtxGetLimit(ctypes.byref(v), which)
        return v.value if r == 0 else -r

    def say(s):
        sys.stdout.flush()
        os.write(1, s.encode())

    if case.startswith("print") or case in ("heap", "clocks"):
        mod = load(open(os.path.join(HERE, "kernels.ptx")).read())
    if case == "print_basic":
        ret = cu.alloc(bytes(16))
        out["launch"] = launch(fn(mod, "print_basic"), 1, 4, ret)
        say("--before sync\n")
        out["sync"] = l.cuCtxSynchronize()
        say("--after sync\n")
        out["ret"] = list(struct.unpack("4i", cu.read(ret, 16)))
    elif case == "print_misc":
        ret = cu.alloc(bytes(8))
        out["launch"] = launch(fn(mod, "print_misc"), 1, 2, ret)
        out["ret"] = list(struct.unpack("2i", cu.read(ret, 8)))
    elif case == "print_flush":
        # output appears at the next flush point: the next launch, a blocking copy, module unload
        f = fn(mod, "print_many")
        launch(f, 1, 1, 1)
        say("--after launch 1\n")
        launch(f, 2, 1, 1)
        say("--after launch 2\n")
        cu.read(cu.alloc(bytes(4)), 4)
        say("--after memcpy\n")
        launch(f, 1, 1, 2)
        say("--before unload\n")
        out["unload"] = l.cuModuleUnload(mod)
        say("--after unload\n")
    elif case == "print_fifo":
        out["default"] = limit(1)
        out["set"] = l.cuCtxSetLimit(1, 256)
        out["get"] = limit(1)
        # 20 lines of 32-byte records in a 256-byte FIFO: the oldest are overwritten, the last 8 printed
        launch(fn(mod, "print_many"), 1, 1, 20)
        out["sync"] = l.cuCtxSynchronize()
        out["bad"] = l.cuCtxSetLimit(99, 1)
    elif case == "heap":
        out["default"] = limit(2)
        out["set"] = l.cuCtxSetLimit(2, 1 << 20)
        n = 64
        buf = cu.alloc(bytes(24 * n))
        out["launch"] = launch(fn(mod, "heap_basic"), 2, 32, buf, 100)
        out["sync"] = l.cuCtxSynchronize()
        v = struct.unpack("%dQ" % (3 * n), cu.read(buf, 24 * n))
        ptrs, again, ok = v[0::3], v[1::3], v[2::3]
        out["all_ok"] = all(ok)
        out["nonnull"] = all(ptrs) and all(again)
        out["aligned16"] = all(p % 16 == 0 for p in ptrs + again)
        out["distinct"] = len(set(ptrs)) == n and len(set(again)) == n
        out["reused"] = len(set(ptrs) & set(again)) == n  # the freed blocks came back
        out["set_after_use"] = l.cuCtxSetLimit(2, 2 << 20)
        slot = cu.alloc(bytes(8))
        res = cu.alloc(bytes(4))
        launch(fn(mod, "heap_keep"), 1, 1, slot, 1234)
        launch(fn(mod, "heap_read"), 1, 1, slot, res)
        out["persist"] = struct.unpack("i", cu.read(res, 4))[0]
        big = cu.alloc(bytes(8))
        launch(fn(mod, "heap_big"), 1, 1, big, ctypes.c_uint64(1 << 21))
        out["too_big_null"] = struct.unpack("Q", cu.read(big, 8))[0] == 0
        launch(fn(mod, "heap_big"), 1, 1, big, ctypes.c_uint64(600 << 10))
        out["fits"] = struct.unpack("Q", cu.read(big, 8))[0] != 0
        out["final_sync"] = l.cuCtxSynchronize()
    elif case == "clocks":
        buf = cu.alloc(bytes(64 * 128))
        f = fn(mod, "clocks")
        launch(f, 2, 64, buf)
        a = struct.unpack("%dQ" % (8 * 128), cu.read(buf, 64 * 128))
        launch(f, 2, 64, buf)
        b = struct.unpack("%dQ" % (8 * 128), cu.read(buf, 64 * 128))
        rows = [a[8 * i:8 * i + 8] for i in range(128)]
        rows2 = [b[8 * i:8 * i + 8] for i in range(128)]
        out["clock_increasing"] = all(r[1] > r[0] for r in rows)
        # __nanosleep(100 us) advances the clocks by about that much
        out["sleep_ns_min"] = min(r[3] - r[2] for r in rows)
        out["timer_increasing"] = all(r[3] > r[2] for r in rows)
        out["across_launches"] = min(r[0] for r in rows2) > max(r[1] for r in rows) and \
            min(r[2] for r in rows2) > max(r[3] for r in rows)
        import time
        out["timer_is_ns_epoch"] = abs(rows[0][2] / 1e9 - time.time()) < 3600
        out["smid_in_range"] = all(r[5] < r[6] for r in rows)
        out["warpid"] = sorted(set(r[7] for r in rows))
        out["smids"] = sorted(set(r[5] for r in rows))
    elif case == "smem_oob":
        mod = load(PTX["smem_oob"])
        out["launch"] = launch(fn(mod, "k"), 1, 4, cu.alloc(bytes(16)))
        out["sync"] = l.cuCtxSynchronize()
    elif case == "dyn_align":
        mod = load(PTX["dyn_align"])
        buf = cu.alloc(bytes(8))
        out["launch"] = launch(fn(mod, "k"), 1, 1, buf, shared=64)
        out["mod16"], out["value"] = struct.unpack("2I", cu.read(buf, 8))
    elif case == "sregs":
        mod = load(PTX["sregs"])
        buf = cu.alloc(bytes(32 * 96))
        out["launch"] = launch(fn(mod, "k"), 1, 96, buf)
        v = struct.unpack("%dI" % (8 * 96), cu.read(buf, 32 * 96))
        ok = True
        for t in range(96):
            eq, lt, le, gt, ge, wid, nw, lane = v[8 * t:8 * t + 8]
            L = t % 32
            ok &= lane == L and eq == 1 << L and lt == (1 << L) - 1 and le == (2 << L) - 1 and \
                gt == (0xFFFFFFFF << (L + 1)) & 0xFFFFFFFF and ge == (0xFFFFFFFF << L) & 0xFFFFFFFF and \
                wid == t // 32 and wid < nw
        out["ok"] = bool(ok)
    elif case == "fence":
        mod = load(PTX["fence"])
        buf = cu.alloc(bytes(12))
        out["launch"] = launch(fn(mod, "k"), 2, 32, buf)
        out["sync"] = l.cuCtxSynchronize()
        out["seen"] = struct.unpack("3I", cu.read(buf, 12))[1]
    elif case == "coop":
        mod = load(PTX["coop"])
        f = fn(mod, "k")
        out["one_block"] = launch(f, 1, 64, cu.alloc(bytes(4)), coop=True)
        out["sync"] = l.cuCtxSynchronize()
        out["two_blocks"] = launch(f, 2, 64, cu.alloc(bytes(4)), coop=True)
    sys.stdout.flush()
    print("\n" + json.dumps(out), flush=True)
    os._exit(0)


BASIC = ["tid %d of 4: %x  3.14 1.234568e+04 0.0001 %c str -1234567890123 18446744073709551615 %% 0002.500|7   |"
         % (t - 2, 255 + t, chr(65 + t)) for t in range(4)]
EXPECT = {
    "print_basic": ({"launch": 0, "sync": 0, "ret": [12] * 4},
                    ["--before sync"] + BASIC + ["--after sync"]),
    "print_misc": ({"launch": 0, "ret": [0, 10]},
                   ["no args", "no args", "even/(null) 0x1234     42|1.23|-1 -32768 -5",
                    "odd/(null) 0x1234     42|1.23|-1 -32768 -5"]),
    "print_flush": ({"unload": 0},
                    ["--after launch 1", "line 0 block 0", "--after launch 2", "line 0 block 0", "line 0 block 1",
                     "--after memcpy", "--before unload", "line 0 block 0", "line 1 block 0", "--after unload"]),
    "print_fifo": ({"default": 1 << 20, "set": 0, "get": 256, "sync": 0, "bad": 215},
                   ["line %d block 0" % i for i in range(12, 20)]),
    "heap": ({"default": 8 << 20, "set": 0, "launch": 0, "sync": 0, "all_ok": True, "nonnull": True,
              "aligned16": True, "distinct": True, "reused": True, "set_after_use": 1, "persist": 1234,
              "too_big_null": True, "fits": True, "final_sync": 0}, None),
    "clocks": ({"clock_increasing": True, "timer_increasing": True, "across_launches": True,
                "timer_is_ns_epoch": True, "smid_in_range": True, "warpid": [0, 1], "smids": [0, 1]}, None),
    "smem_oob": ({"launch": 0, "sync": 700}, None),
    "dyn_align": ({"launch": 0, "mod16": 0, "value": 5}, None),
    "sregs": ({"launch": 0, "ok": True}, None),
    "fence": ({"launch": 0, "sync": 0, "seen": 42}, None),
    "coop": ({"one_block": 0, "sync": 0, "two_blocks": 720}, None),
}


# CUDA behavior the AGX path does not reproduce (reported, not counted as failures)
KNOWN = {
    "smem_oob": "the AGX functional model (as the hardware it models) drops out-of-bounds threadgroup stores and reads "
                "0, so the kernel completes; CUDA reports CUDA_ERROR_ILLEGAL_ADDRESS",
}


def regen():
    subprocess.check_call(["clang++", "-x", "cuda", "--cuda-device-only", "-nocudainc", "-nocudalib",
                           "--cuda-gpu-arch=sm_70", "-Xclang", "-target-feature", "-Xclang", "+ptx70", "-O2", "-S",
                           "-o", os.path.join(HERE, "kernels.ptx"), os.path.join(HERE, "kernels.cu")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.path.join(HERE, "..", "..", "target", "release", "libzluda_metal.so"))
    ap.add_argument("--child")
    ap.add_argument("-k", action="append")
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    if a.child:
        child(a.child, a.lib)
    if a.regen:
        regen()
    env = dict(os.environ, ZLUDA_METAL_BACKEND="agx", ZLUDA_AGX_MAX_STEPS="2000000")
    sib = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "llvm-project", "build", "bin", "llvm-air2agx"))
    if os.path.exists(sib):
        env.setdefault("ZLUDA_AGX_AIR2AGX", sib)
    fails = 0
    for case, (want, lines) in EXPECT.items():
        if a.k and case not in a.k:
            continue
        r = subprocess.run([sys.executable, __file__, "--child", case, "--lib", a.lib], capture_output=True, text=True,
                           env=env, timeout=1200)
        text = r.stdout.splitlines()
        js = [x for x in text if x.startswith("{")]
        got = json.loads(js[-1]) if js else {}
        printed = [x for x in text if x and not x.startswith("{")]
        ok = all(got.get(k) == v for k, v in want.items())
        if case == "clocks":
            ok = ok and got.get("sleep_ns_min", 0) >= 90000
        if lines is not None:
            ok = ok and printed == lines
        status = "PASS" if ok else "FAIL"
        if case in KNOWN:
            status = "XPASS" if ok else "XFAIL"
        else:
            fails += not ok
        print("%-5s %-12s %s" % (status, case, json.dumps(got)), flush=True)
        if case in KNOWN and not ok:
            print("      known:", KNOWN[case])
        if a.v or (not ok and case not in KNOWN):
            if lines is not None:
                print("      printed:", printed)
            if r.stderr.strip():
                print("      stderr:", r.stderr.strip()[-600:].replace("\n", " | "))
    print("---- %s" % ("PASS" if not fails else "FAIL %d" % fails))
    sys.exit(1 if fails else 0)


main()
