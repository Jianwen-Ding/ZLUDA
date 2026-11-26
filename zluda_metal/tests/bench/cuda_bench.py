#!/usr/bin/env python3
"""Realistic CUDA kernels through ZLUDA -> AGX, timed by the AGX Performance Simulator's model.

The kernels are the LLVM fork's agx-bench CUDA sources (llvm/utils/agx-bench/cuda/*.cu, open sources compiled with
clang's NVPTX target: no CUDA toolkit, no nvcc). Each is compiled to PTX, loaded through ZLUDA's CUDA driver API
(libzluda_metal.so, ZLUDA_METAL_BACKEND=agx) and launched at a realistic size:

* the scalar arguments come from the source's `// agx-cost:` line (`--uniform I=V`: argument words in order, a
  pointer or a 64-bit integer taking two, aligned);
* the grid is the line's `--grid`, or 64 workgroups of its `--wg` (8 x 8 for 2D workgroups);
* every pointer argument gets its own 16 MiB buffer: float-like data (`float`, `half`, `f4`, `double`, ...) a
  deterministic pattern of small values, integer data zeros (index arrays then stay in bounds), CSR row pointers a
  regular pattern.

ZLUDA_AGX_STATS gives each launch's predicted time. Output buffers are hashed: --update writes the hashes as the
reference (bench_ref.json next to this script by default), later runs report OUTPUT-CHANGED when an optimization
changes a result.

  cuda_bench.py [--clang clang] [--lib LIB] [--json OUT] [--compare OLD.json] [--update] [--fp-mode strict|fast|fast-ftz]
                [NAME ...]

--fp-mode is ZLUDA_FP_MODE (zluda_metal/src/compile.rs FpMode; default strict here, the mode whose outputs
bench_ref.json holds; the other modes have their own reference, bench_ref_<mode>.json).
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "llvm-project", "llvm", "utils", "agx-bench", "cuda")
sys.path.insert(0, os.path.join(HERE, "..", "ptxconf"))
import ptxconf  # noqa: E402

BUF = 16 << 20


def compile_ptx(src, clang):
    with tempfile.TemporaryDirectory() as t:
        out = os.path.join(t, "k.ptx")
        r = subprocess.run([clang, "-x", "cuda", "--cuda-device-only", "-nocudainc", "-nocudalib", "-O3",
                            "--cuda-gpu-arch=sm_80", "-Xclang", "-target-feature", "-Xclang", "+ptx70", "-S", "-I", SRC,
                            src, "-o", out], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-400:])
        return open(out).read()


def signature(src):
    s = open(src).read()
    m = re.search(r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)", s)
    params = [p.strip() for p in m.group(2).split(",") if p.strip()]
    return m.group(1), params


def cost_line(src):
    for line in open(src):
        if line.startswith("// agx-cost:"):
            return line[len("// agx-cost:"):].strip()
    return ""


def plan(src):
    name, params = signature(src)
    line = cost_line(src)
    uni = {int(k, 0): int(v, 0) for k, v in re.findall(r"--uniform (\w+)=(\w+)", line)}
    wg = [int(x) for x in re.search(r"--wg ([\d,]+)", line).group(1).split(",")] if "--wg" in line else [256]
    wg = (wg + [1, 1])[:3]
    g = re.search(r"--grid ([\d,]+)", line)
    if g:
        grid_threads = (list(map(int, g.group(1).split(","))) + [1, 1])[:3]
        grid = [max(1, -(-grid_threads[i] // wg[i])) for i in range(3)]
    else:
        grid = [8, 8, 1] if wg[1] > 1 else [64, 1, 1]
    args, word = [], 0
    for p in params:
        if "*" in p:
            word += word % 2
            base = p.split("*")[0].replace("const", "").strip()
            args.append(("ptr", base, re.findall(r"\w+$", p)[0]))
            word += 2
        elif re.match(r"(long|size_t|unsigned long)", p.replace("const ", "")):
            word += word % 2
            args.append(("i64", uni.get(word, 0) | (uni.get(word + 1, 0) << 32)))
            word += 2
        else:
            args.append(("i32", uni.get(word, 0)))
            word += 1
    return name, args, grid, wg


def buffer_data(base, pname):
    n = BUF
    if "rowptr" in pname:
        rows = n // 4
        per = 16 if "vector" not in pname else 64
        return struct.pack("<%di" % rows, *[min(per * i, 1 << 20) for i in range(rows)])
    if base in ("float", "f4"):
        return struct.pack("<%df" % (n // 4), *[1.0 + (i % 7) / 8.0 for i in range(n // 4)])
    if base == "double":
        return struct.pack("<%dd" % (n // 8), *[1.0 + (i % 7) / 8.0 for i in range(n // 8)])
    if base in ("half", "half2"):
        return struct.pack("<%de" % (n // 2), *[1.0 + (i % 7) / 8.0 for i in range(n // 2)])
    return bytes(n)


def run_one(src, lib, clang):
    name, args, grid, wg = plan(src)
    ptx = compile_ptx(src, clang)
    stats = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
    os.environ["ZLUDA_AGX_STATS"] = stats
    cuda = ptxconf.Cuda(lib)
    mod = ctypes.c_void_p()
    if cuda.cuModuleLoadData(ctypes.byref(mod), ptx.encode() + b"\0"):
        raise RuntimeError("load: " + cuda.last_error()[-600:])
    fn = ctypes.c_void_p()
    cuda.check(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()), "get function")
    holders, bufs = [], []
    for a in args:
        if a[0] == "ptr":
            p = cuda.alloc(buffer_data(a[1], a[2]))
            bufs.append(p)
            v = p.to_bytes(8, "little")
        else:
            v = (a[1] & ((1 << 64) - 1)).to_bytes(8, "little")
        holders.append(ctypes.create_string_buffer(v, 16))
    arr = (ctypes.c_void_p * len(holders))(*[ctypes.cast(h, ctypes.c_void_p) for h in holders])
    r = cuda.cuLaunchKernel(fn, grid[0], grid[1], grid[2], wg[0], wg[1], wg[2], 0, None, arr, None)
    if r or cuda.cuCtxSynchronize():
        raise RuntimeError("launch %d: %s" % (r, cuda.last_error()[-600:]))
    h = hashlib.sha256()
    for p in bufs:
        h.update(cuda.read(p, BUF))
    launches = [json.loads(l) for l in open(stats) if l.strip()]
    os.unlink(stats)
    l = launches[-1]
    return {"time_us": l["time_us"], "instructions": l["instructions"], "gprs": l["gprs"],
            "code_bytes": l["code_bytes"], "hash": h.hexdigest()[:16], "grid": grid, "wg": wg,
            "timing_error": l["timing_error"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--clang", default="clang")
    ap.add_argument("--lib", default=os.path.join(HERE, "..", "..", "target", "release", "libzluda_metal.so"))
    ap.add_argument("--json")
    ap.add_argument("--compare")
    ap.add_argument("--ref")
    ap.add_argument("--fp-mode", dest="fp_mode", choices=["strict", "fast", "fast-ftz"], default="strict")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--one")
    a = ap.parse_args()
    if a.one:  # child process: one kernel
        os.environ["ZLUDA_METAL_BACKEND"] = "agx"
        try:
            print(json.dumps(run_one(a.one, a.lib, a.clang)))
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"error": str(e)[-500:]}))
        sys.stdout.flush()
        os._exit(0)
    if not a.ref:
        a.ref = os.path.join(HERE, "bench_ref.json" if a.fp_mode == "strict" else "bench_ref_%s.json" % a.fp_mode)
    names = a.names or sorted(f[:-3] for f in os.listdir(SRC) if f.endswith(".cu"))
    ref = json.load(open(a.ref)) if os.path.exists(a.ref) else {}
    old = json.load(open(a.compare)) if a.compare else {}
    env = dict(os.environ, ZLUDA_FP_MODE=a.fp_mode)
    sib = os.path.join(ROOT, "llvm-project", "build", "bin", "llvm-air2agx")
    if "ZLUDA_AGX_AIR2AGX" not in env and os.path.exists(sib):
        env["ZLUDA_AGX_AIR2AGX"] = sib
    res = {}
    print("%-16s %10s %10s %5s %6s  %s" % ("kernel", "time_us", "instr", "gprs", "bytes", "status"))
    for n in names:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", os.path.join(SRC, n + ".cu"),
                            "--lib", a.lib, "--clang", a.clang], capture_output=True, text=True, env=env,
                           timeout=3600)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        x = json.loads(line[-1]) if line else {"error": (r.stdout + r.stderr)[-300:]}
        res[n] = x
        if "error" in x:
            print("%-16s %s" % (n, "ERROR " + x["error"].replace("\n", " ")[:150]), flush=True)
            continue
        st = "ok"
        if n in ref and ref[n] != x["hash"]:
            st = "OUTPUT-CHANGED"
        if n in old and "time_us" in old[n] and old[n]["time_us"] and x["time_us"]:
            st += "  %+.1f%%" % (100 * (x["time_us"] / old[n]["time_us"] - 1))
        print("%-16s %10.2f %10d %5d %6d  %s" % (n, x["time_us"] or float("nan"), x["instructions"], x["gprs"],
                                                 x["code_bytes"], st), flush=True)
    ok = [n for n in res if "time_us" in res[n] and res[n]["time_us"]]
    if ok:
        print("geomean time %.3f us over %d kernels" % (
            math.exp(sum(math.log(res[n]["time_us"]) for n in ok) / len(ok)), len(ok)))
    if old:
        common = [n for n in ok if n in old and old[n].get("time_us")]
        if common:
            print("geomean ratio vs old: %.4f" % math.exp(
                sum(math.log(res[n]["time_us"] / old[n]["time_us"]) for n in common) / len(common)))
    if a.update:
        json.dump({n: res[n]["hash"] for n in res if "hash" in res[n]}, open(a.ref, "w"), indent=1)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
