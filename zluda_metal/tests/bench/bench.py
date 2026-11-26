#!/usr/bin/env python3
"""Performance of the AGX pipeline on real PTX programs, by the AGX Performance Simulator's model.

Runs each program through ptxconf (so results are still checked bit for bit) with ZLUDA_AGX_STATS set: every launch
is also timed by the simulator's dispatch model (src/agx.rs time_dispatch: M2 Max, CUDA-runtime launch profile,
paths from the functional model over the launch's memory). Prints per program the predicted time, dynamic
instructions, GPRs and code bytes, and the totals; --json writes them, --compare OLD.json shows the change.

  bench.py [--lib LIB] [--json OUT] [--compare OLD.json] [FILES...]   (default: the BENCH programs below)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PF = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "PTX-formalization", "tests"))
BENCH = ["corpus-dot", "corpus-gcd", "corpus-histogram", "corpus-matmul", "corpus-reduce", "corpus-saxpy",
         "corpus-scan", "corpus-transpose", "corpus-vecadd", "corpus-warpscan", "matmul", "reduce", "vecadd",
         "atomics", "barrier", "calls", "dynshared", "mma-0", "ldm-0", "wg-0", "testfloat", "cvt-0", "half-0",
         "video-0", "lop3-shf", "bulk", "tma"]


def run(path, lib, fp_mode):
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as t:
        stats = t.name
    env = dict(os.environ, ZLUDA_AGX_STATS=stats)
    r = subprocess.run([sys.executable, os.path.join(HERE, "..", "ptxconf", "ptxconf.py"), "--backend", "agx",
                        "-j", "1", "--lib", lib, "--fp-mode", fp_mode, path], capture_output=True, text=True, env=env)
    last = r.stdout.strip().splitlines()[-1].split() if r.stdout.strip() else ["", "CRASH"]
    status = last[1]  # "---- PASS 1 of 1", APPROX, FAIL, ...
    launches = [json.loads(l) for l in open(stats) if l.strip()]
    os.unlink(stats)
    return {"status": status, "launches": len(launches),
            "time_us": sum(l["time_us"] or 0 for l in launches),
            "instructions": sum(l["instructions"] for l in launches),
            "gprs": max([l["gprs"] for l in launches] or [0]),
            "code_bytes": sum(l["code_bytes"] for l in launches),
            "timing_errors": sum(1 for l in launches if l["timing_error"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--lib", default=os.path.join(HERE, "..", "..", "target", "release", "libzluda_metal.so"))
    ap.add_argument("--json")
    ap.add_argument("--compare")
    ap.add_argument("--fp-mode", dest="fp_mode", choices=["strict", "fast", "fast-ftz"], default="strict")
    a = ap.parse_args()
    files = a.files or [os.path.join(PF, b + ".ptx") for b in BENCH]
    old = json.load(open(a.compare)) if a.compare else {}
    res = {}
    print("%-26s %6s %12s %12s %5s %7s  %s" % ("program", "status", "time_us", "instructions", "gprs", "bytes",
                                                "change"))
    for f in files:
        name = os.path.basename(f)[:-4]
        r = res[name] = run(f, a.lib, a.fp_mode)
        ch = ""
        if name in old and old[name]["time_us"]:
            ch = "%+.1f%% time, %+.1f%% instr" % (100 * (r["time_us"] / old[name]["time_us"] - 1),
                                                 100 * (r["instructions"] / max(old[name]["instructions"], 1) - 1))
        print("%-26s %6s %12.2f %12d %5d %7d  %s" % (name, r["status"][:6], r["time_us"], r["instructions"],
                                                   r["gprs"], r["code_bytes"], ch), flush=True)
    tot = sum(r["time_us"] for r in res.values())
    print("total predicted time %.2f us over %d programs" % (tot, len(res)))
    if old:
        common = [k for k in res if k in old]
        print("geomean time ratio vs old: %.4f" % (
            __import__("math").prod(res[k]["time_us"] / old[k]["time_us"] for k in common
                                    if old[k]["time_us"] and res[k]["time_us"]) ** (1 / max(len(common), 1))))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
