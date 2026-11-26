#!/usr/bin/env python3
"""Translation validation of the AGX optimization pipeline on real PTX kernels, with alive2.

For each PTX program (PTX-formalization's harness programs: the module part only), ZLUDA compiles it to portable
AIR (ptx2metallib, ZLUDA_PORTABLE_AIR=1); llvm-air2agx lowers the AIR to AGX IR at -O0 (agx-lower-air alone) and
-O3 (the whole optimization pipeline); alive-tv checks that every kernel at -O3 refines the same kernel at -O0.

  opt_check.py [--alive-tv PATH] [--air2agx PATH] [--ptx2metallib PATH] [--timeout S] [-j N] [--json OUT]
               [--exact] FILE_OR_DIR...

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


def check_program(path, a):
    out = []
    with tempfile.TemporaryDirectory() as t:
        module, _ = ptxconf.split_program(open(path, errors="replace").read())
        ptx = os.path.join(t, "m.ptx")
        open(ptx, "w").write(module)
        lib = os.path.join(t, "m.metallib")
        env = dict(os.environ, ZLUDA_PORTABLE_AIR="1")
        r = subprocess.run([a.ptx2metallib, ptx, "-o", lib], capture_output=True, text=True, env=env, timeout=600)
        if r.returncode:
            return [{"program": path, "kernel": "*", "status": "skip", "detail": "front end: " + r.stderr[-200:]}]
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
        for f in ll["0"]:
            src, tgt = os.path.join(t, "O0", f), os.path.join(t, "O3", f)
            if f not in ll["3"]:
                continue
            for k in kernels_of(ll["0"][f]):
                cmd = [a.alive_tv, "--smt-to=%d" % (a.timeout * 1000), "--src-fn=" + k, "--tgt-fn=" + k, src, tgt]
                if not a.exact:
                    cmd.insert(1, "--fp-uf")
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout * 4 + 60)
                    txt = r.stdout + r.stderr
                except subprocess.TimeoutExpired:
                    out.append({"program": path, "kernel": k, "status": "unknown", "detail": "process timeout"})
                    continue
                if re.search(r"^\s*[1-9]\d* incorrect transformations", txt, re.M):
                    st = "INCORRECT"
                elif re.search(r"^\s*[1-9]\d* correct transformations", txt, re.M):
                    st = "correct"
                elif re.search(r"^\s*[1-9]\d* failed-to-prove", txt, re.M):
                    st = "unknown"
                else:
                    st = "error"
                detail = ""
                if st in ("INCORRECT", "error", "unknown"):
                    m = re.search(r"(ERROR:[^\n]*|Transformation doesn't verify![^\n]*\n[^\n]*)", txt)
                    detail = m.group(1) if m else txt[-300:]
                out.append({"program": path, "kernel": k, "status": st, "detail": detail.strip()})
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
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
