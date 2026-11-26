#!/usr/bin/env python3
"""Translation validation of the AGX optimization pipeline on real PTX kernels, with alive2.

For each PTX program (PTX-formalization's harness programs: the module part only), ZLUDA compiles it to portable
AIR (ptx2metallib, ZLUDA_PORTABLE_AIR=1); llvm-air2agx lowers the AIR to AGX IR at -O0 (agx-lower-air alone) and
-O3 (the whole optimization pipeline); alive-tv checks that every kernel at -O3 refines the same kernel at -O0, or
with --src air that it refines ZLUDA's AIR itself (alive2's GPU semantics gives AIR and AGX IR one meaning).

  opt_check.py [--alive-tv PATH] [--air2agx PATH] [--ptx2metallib PATH] [--timeout S] [-j N] [--json OUT]
               [--exact] [--per-pass] [--unroll N] FILE_OR_DIR...

--per-pass checks each pass of the -O3 pipeline that changes the module (llvm-air2agx -print-changed
-print-module-scope): every step must refine the one before, so the whole pipeline is proved by small queries
that the solver answers where one -O0 / -O3 query times out. --unroll N lets alive2 unroll loops N times
(bounded: a loop that needs more iterations is not checked beyond them).

By default floating point operations are uninterpreted functions of their operands (alive-tv --fp-uf, this
repository's alive2 fork): sound and fast, but a transformation that rewrites a floating point operation is
reported "unknown"; --exact uses IEEE semantics instead (much slower).

Results per kernel: correct, INCORRECT (alive2 found a counterexample: an optimization bug, or a gap in alive2's
model of an AGX intrinsic, to be read), unknown (timeout / failed to prove), error (alive2 could not translate the
IR, e.g. an unsupported construct), skip (the program does not compile).
No proprietary tool is used.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ptxconf"))
import ptxconf  # noqa: E402


def defaults():
    return {
        "alive_tv": os.path.join(ROOT, "alive2", "build", "alive-tv"),
        "air2agx": os.path.join(ROOT, "llvm-project", "build", "bin", "llvm-air2agx"),
        "ptx2metallib": os.path.join(HERE, "..", "..", "target", "release", "ptx2metallib"),
    }


def kernels_of(ll):
    return re.findall(r"^define agx_kernel [^@]*@([\w.$]+)\(", ll, re.M)


PASSES = []  # per-pass results (--per-pass)


def alive(a, k, src, tgt):
    """alive-tv on kernel k: (status, detail, output)"""
    cmd = [a.alive_tv, "--smt-to=%d" % (a.timeout * 1000), "--src-fn=" + k, "--tgt-fn=" + k, src, tgt]
    if not a.exact:
        cmd.insert(1, "--fp-uf")
    if a.unroll:
        cmd[1:1] = ["--src-unroll=%d" % a.unroll, "--tgt-unroll=%d" % a.unroll]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout * 4 + 60)
        txt = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "unknown", "process timeout", ""
    if re.search(r"^\s*[1-9]\d* incorrect transformations", txt, re.M):
        st = "INCORRECT"
    elif re.search(r"^\s*[1-9]\d* correct transformations", txt, re.M):
        st = "correct"
    elif re.search(r"^\s*[1-9]\d* failed-to-prove", txt, re.M):
        st = "unknown"
    else:
        st = "error"
    detail = ""
    if st != "correct":
        m = re.search(r"(ERROR:[^\n]*|Transformation doesn't verify![^\n]*\n[^\n]*)", txt)
        detail = m.group(1) if m else txt[-300:]
    return st, detail.strip(), txt


STRIP = lambda text: re.sub(r"\bmemory\([^)]*\)", "", text)


def check_per_pass(path, a, lib, t):
    """every module-changing pass of -O3, checked against the module before it"""
    r = subprocess.run([a.air2agx, "-mcpu=agx-g14c", "-mattr=+canonical-nan", "-O3", "--emit=llvm", lib,
                        "--output-dir", os.path.join(t, "pp"), "-print-changed=quiet", "-print-module-scope"],
                       capture_output=True, text=True, timeout=600)
    if r.returncode:
        return [{"program": path, "kernel": "*", "status": "skip", "detail": "air2agx: " + r.stderr[-200:]}]
    # one module per metallib file: the dumps of each run in order (a dump restarts at AGXLowerAIRPass)
    parts = re.split(r"^(?:; )?\*\*\* IR Dump After (.*?) \*\*\*\n", r.stderr, flags=re.M)
    chains, cur = [], None
    for i in range(1, len(parts), 2):
        name, text = parts[i], STRIP(parts[i + 1])
        if name.startswith("AGXLowerAIRPass") or cur is None:
            cur = []
            chains.append(cur)
        cur.append((name, text))
    out = []
    n = 0
    for chain in chains:
        kernels = kernels_of(chain[0][1])
        worst = {k: ("correct", "") for k in kernels}
        for j in range(1, len(chain)):
            pname = chain[j][0]
            fn = re.search(r" on (\S+)$", pname)
            src, tgt = os.path.join(t, "s%d.ll" % n), os.path.join(t, "t%d.ll" % n)
            n += 1
            open(src, "w").write(chain[j - 1][1])
            open(tgt, "w").write(chain[j][1])
            for k in kernels:
                # a function pass on another function does not change this kernel
                if fn and not fn.group(1).startswith(("[", "(")) and fn.group(1) != k:
                    continue
                if k not in kernels_of(chain[j][1]):
                    continue
                st, detail, txt = alive(a, k, src, tgt)
                PASSES.append({"program": path, "kernel": k, "pass": pname, "status": st, "detail": detail[:200]})
                if a.v:
                    print("  %-9s %s %s %s  %s" % (st, os.path.basename(path), k, pname, detail[:120]), flush=True)
                rank = {"correct": 0, "error": 1, "unknown": 2, "INCORRECT": 3}
                if rank[st] > rank[worst[k][0]]:
                    worst[k] = (st, pname + ": " + detail)
                if st == "INCORRECT" and a.keep:
                    os.makedirs(a.keep, exist_ok=True)
                    base = os.path.join(a.keep, "%s.%s.%d" % (os.path.basename(path), k, j))
                    open(base + ".src.ll", "w").write(open(src).read())
                    open(base + ".tgt.ll", "w").write(open(tgt).read())
                    open(base + ".txt", "w").write(txt)
        for k, (st, d) in worst.items():
            out.append({"program": path, "kernel": k, "status": st, "detail": d})
    return out


def check_program(path, a):
    out = []
    with tempfile.TemporaryDirectory() as t:
        module, _ = ptxconf.split_program(open(path, errors="replace").read())
        ptx = os.path.join(t, "m.ptx")
        open(ptx, "w").write(module)
        lib = os.path.join(t, "m.metallib")
        env = dict(os.environ, ZLUDA_PORTABLE_AIR="1")
        air_dir = os.path.join(t, "air")
        r = subprocess.run([a.ptx2metallib, ptx, "-o", lib, "--emit-air", air_dir], capture_output=True, text=True,
                           env=env, timeout=600)
        if r.returncode:
            return [{"program": path, "kernel": "*", "status": "skip", "detail": "front end: " + r.stderr[-200:]}]
        if a.per_pass:
            return check_per_pass(path, a, lib, t)
        ll = {}
        for lvl in ("0", "3"):
            d = os.path.join(t, "O" + lvl)
            os.makedirs(d)
            r = subprocess.run([a.air2agx, "-mcpu=agx-g14c", "-mattr=+canonical-nan", "-O" + lvl, "--emit=llvm",
                                lib, "--output-dir", d], capture_output=True, text=True, timeout=600)
            if r.returncode:
                return [{"program": path, "kernel": "*", "status": "skip", "detail": "air2agx: " + r.stderr[-200:]}]
            ll[lvl] = {f: open(os.path.join(d, f)).read() for f in sorted(os.listdir(d))}
            # memory(...) attributes inferred while ignoring reads of AGX constant memory (address
            # space 4, which LLVM's alias analysis knows to be constant and alive2 does not) would
            # make those reads UB for alive2: drop them (this only stops checking the inference)
            for f, text in ll[lvl].items():
                text = re.sub(r"\bmemory\([^)]*\)", "", text)
                ll[lvl][f] = text
                open(os.path.join(d, f), "w").write(text)
        if a.src == "air":
            # ZLUDA's AIR itself (alive2's GPU semantics compares AIR with AGX IR): agx-lower-air and the
            # optimizer together
            dis = os.path.join(os.path.dirname(a.air2agx), "llvm-dis")
            for f in ll["3"]:
                ks = kernels_of(ll["3"][f])
                for k in ks:
                    air = os.path.join(air_dir, k + ".air")
                    if os.path.exists(air):
                        subprocess.run([dis, air, "-o", os.path.join(t, "O0", f)], check=True)
        for f in ll["0"]:
            src, tgt = os.path.join(t, "O0", f), os.path.join(t, "O3", f)
            if f not in ll["3"]:
                continue
            for k in kernels_of(ll["3"][f] if a.src == "air" else ll["0"][f]):
                st, detail, txt = alive(a, k, src, tgt)
                out.append({"program": path, "kernel": k, "status": st, "detail": detail})
                if st == "INCORRECT" and a.keep:
                    os.makedirs(a.keep, exist_ok=True)
                    base = os.path.join(a.keep, os.path.basename(path) + "." + k)
                    open(base + ".src.ll", "w").write(open(src).read())
                    open(base + ".tgt.ll", "w").write(open(tgt).read())
                    open(base + ".txt", "w").write(txt)
    return out


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--alive-tv", dest="alive_tv", default=d["alive_tv"])
    ap.add_argument("--air2agx", default=d["air2agx"])
    ap.add_argument("--ptx2metallib", default=d["ptx2metallib"])
    ap.add_argument("--timeout", type=int, default=60, help="SMT timeout per kernel (s)")
    ap.add_argument("-j", type=int, default=2)
    ap.add_argument("-k", action="append")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--per-pass", dest="per_pass", action="store_true")
    ap.add_argument("--unroll", type=int, default=0)
    ap.add_argument("-v", action="store_true", help="print every per-pass result")
    ap.add_argument("--src", choices=["o0", "air"], default="o0",
                    help="the source: AGX IR at -O0 (checks the optimizer) or ZLUDA's AIR (agx-lower-air too)")
    ap.add_argument("--keep", help="directory for the IR and alive2 output of INCORRECT kernels")
    ap.add_argument("--json")
    a = ap.parse_args()
    files = ptxconf.collect(a.paths, a.k)
    res = []
    with cf.ThreadPoolExecutor(a.j) as ex:
        for rs in ex.map(lambda f: check_program(f, a), files):
            for r in rs:
                res.append(r)
                if r["status"] != "correct":
                    print("%-9s %s %s  %s" % (r["status"], os.path.relpath(r["program"]), r["kernel"],
                                             r["detail"].replace("\n", " ")[:200]), flush=True)
    counts = {}
    for r in res:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("----", ", ".join("%s %d" % kv for kv in sorted(counts.items())), "kernels of", len(files), "programs")
    if PASSES:
        pc = {}
        for r in PASSES:
            pc[r["status"]] = pc.get(r["status"], 0) + 1
        print("---- per pass:", ", ".join("%s %d" % kv for kv in sorted(pc.items())), "pass checks")
    if a.json:
        json.dump({"kernels": res, "passes": PASSES}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
