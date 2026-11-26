#!/usr/bin/env python3
"""Writes the object of the first gpu.binary in an MLIR file (gpu-module-to-binary's output) to a file: MLIR prints
it as a string literal with \\XX hex escapes.   extract_binary.py IN.mlir OUT"""
import re
import sys

text = open(sys.argv[1], encoding="latin-1").read()
m = re.search(r'#gpu\.object<#\w+\.target(?:<(?:[^>"]|"[^"]*")*>)?\s*,\s*"((?:[^"\\]|\\.)*)"', text, re.S)
if not m:
    sys.exit("no gpu.object in " + sys.argv[1])
s, out, i = m.group(1), bytearray(), 0
while i < len(s):
    c = s[i]
    if c == "\\":
        n = s[i + 1]
        if n in '\\"':
            out.append(ord(n))
            i += 2
        elif n == "n":
            out.append(10)
            i += 2
        elif n == "t":
            out.append(9)
            i += 2
        else:
            out.append(int(s[i + 1:i + 3], 16))
            i += 3
    else:
        out.append(ord(c))
        i += 1
open(sys.argv[2], "wb").write(out)
