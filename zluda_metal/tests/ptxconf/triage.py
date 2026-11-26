#!/usr/bin/env python3
"""Ranks what blocks the most ptxconf results: triage.py results.json [N]"""
import collections, json, re, sys

d = json.load(open(sys.argv[1]))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
c, ex = collections.Counter(), {}
for k, v in d.items():
    if v["status"] == "PASS":
        continue
    det = v["detail"]
    items = re.findall(r'Unrecognized(?:Statement|Directive)\("((?:[^"\\]|\\.)*)"\)', det)
    keys = set()
    for it in items:
        it = re.sub(r"^@!?%?\w+\s+", "", it.strip())
        if not it:
            continue
        op = it.split()[0]
        parts = re.split(r"\.", op)
        keys.add(".".join(parts[:2]) if len(parts) > 1 and not parts[1][:1].isdigit() else parts[0])
    if not items:
        m = re.search(r"unresolved external function: (__zluda_ptx_impl_\w+)", det)
        keys.add(v["status"] + ": " + (m.group(1) if m else re.sub(r"\d+", "N", det)[:90]))
    for kk in keys:
        c[kk] += 1
        ex.setdefault(kk, k.split("/")[-1])
for k, cnt in c.most_common(n):
    print(cnt, k, "|", ex[k])
