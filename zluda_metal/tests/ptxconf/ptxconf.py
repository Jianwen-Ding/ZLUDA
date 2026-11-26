#!/usr/bin/env python3
"""PTX-formalization conformance through ZLUDA's Metal pipeline.

Runs the harness programs of PTX-formalization (`tests/*.ptx`: a PTX module
followed by `.buffer` / `.launch` / `.expect` directives) through ZLUDA's CUDA
driver API (`libzluda_metal.so`): PTX -> ZLUDA LLVM IR -> AIR -> .metallib, then
with ZLUDA_METAL_BACKEND=agx  .metallib -> llvm-air2agx -> AGX machine code run
on the AGX Performance Simulator. Every `.expect` is checked bit for bit
(floats are parsed at the buffer's type, as the formal model does).

  ptxconf.py [--lib target/release/libzluda_metal.so] [--backend agx|metal]
             [--json out.json] [-j N] [-k SUBSTR] FILE_OR_DIR...

Each program runs in its own process (a crash or a hang is one result).
Results: PASS, FAIL (an .expect differs), COMPILE (the module did not load),
LAUNCH (a launch failed), UNSUPPORTED (a harness directive this runner does not
implement: tensor maps, textures, multimem, ...), HARNESS (could not parse),
TIMEOUT, CRASH.

No proprietary tool is involved: no nvcc / ptxas, no Apple Metal compiler.
"""
import argparse
import ctypes
import json
import math
import os
import re
import struct
import subprocess
import sys
import concurrent.futures as cf

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIB = os.path.join(HERE, "..", "..", "target", "release", "libzluda_metal.so")

HARNESS = ("buffer", "launch", "print", "expect", "patch", "append", "tensormap", "texture", "sampler", "surface",
           "multimem", "fabric", "memmodel", "ub", "occupancy", "cooplaunch", "convergence", "isacheck")
IGNORED = ("print", "memmodel", "isacheck", "convergence", "occupancy")
UNSUPPORTED = ("tensormap", "texture", "sampler", "surface", "multimem", "fabric", "ub")

SIZES = {"u8": 1, "s8": 1, "b8": 1, "u16": 2, "s16": 2, "b16": 2, "f16": 2, "bf16": 2, "u32": 4, "s32": 4,
         "b32": 4, "f32": 4, "u64": 8, "s64": 8, "b64": 8, "f64": 8, "pred": 1, "b128": 16}


class Unsupported(Exception):
    pass


def f_to_bits(x, ty):
    if ty == "f32":
        return struct.unpack("<I", struct.pack("<f", x))[0]
    if ty == "f64":
        return struct.unpack("<Q", struct.pack("<d", x))[0]
    if ty == "f16":
        return struct.unpack("<H", struct.pack("<e", x))[0]
    if ty == "bf16":
        b = struct.unpack("<I", struct.pack("<f", x))[0]
        return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    return int(x)


def parse_value(v, ty):
    v = v.strip()
    size = SIZES[ty]
    mask = (1 << (8 * size)) - 1
    m = re.fullmatch(r"([-+]?)0[fF]([0-9a-fA-F]{8})", v)
    if m:
        b = int(m.group(2), 16)
        if ty == "f64":
            return f_to_bits(struct.unpack("<f", struct.pack("<I", b))[0], ty) ^ ((1 << 63) if m.group(1) == "-" else 0)
        return (b ^ (0x80000000 if m.group(1) == "-" else 0)) & mask
    m = re.fullmatch(r"([-+]?)0[dD]([0-9a-fA-F]{16})", v)
    if m:
        b = int(m.group(2), 16)
        if ty == "f32":
            return f_to_bits(struct.unpack("<d", struct.pack("<Q", b))[0], ty)
        return (b ^ ((1 << 63) if m.group(1) == "-" else 0)) & mask
    m = re.fullmatch(r"([-+]?)0[xX]([0-9a-fA-F]+)[uU]?[lL]?", v)
    if m:
        n = int(m.group(2), 16)
        return (-n if m.group(1) == "-" else n) & mask
    if re.fullmatch(r"[-+]?\d+[uU]?", v) and not ty.startswith(("f", "bf")):
        return int(v.rstrip("uU")) & mask
    low = v.lower().lstrip("+")
    if low in ("inf", "-inf", "nan", "-nan", "infinity", "-infinity") or re.fullmatch(
            r"[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?", low):
        x = float(low)
        if ty.startswith(("f", "bf")):
            return f_to_bits(x, ty) & mask
        return int(x) & mask
    # constant expressions (3*3, WARP_SZ/4): evaluate integer arithmetic only
    if re.fullmatch(r"[\d\s+\-*/()%<>&|^~]+", v):
        return int(eval(v.replace("/", "//"))) & mask
    raise Unsupported("value %r" % v)


def split_list(s):
    s = s.strip()
    if s.startswith("{"):
        s = s[1:]
    if s.endswith("}"):
        s = s[:-1]
    return [x.strip() for x in s.split(",") if x.strip()]


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def split_program(text):
    """(module text, [(directive, body)]) in order."""
    clean = strip_comments(text)
    out_lines, dirs = [], []
    lines = clean.split("\n")
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        m = re.match(r"\.(\w+)\b", s)
        if m and m.group(1) in HARNESS and not (m.group(1) in ("texture", "sampler", "surface") and
                                                re.match(r"\.(texture|sampler|surface)\s+\.\w+", s) is None and
                                                False):
            stmt = s
            while not stmt.rstrip().endswith(";") and i + 1 < len(lines):
                i += 1
                stmt += " " + lines[i].strip()
            dirs.append((m.group(1), stmt[len(m.group(0)):].strip().rstrip(";").strip()))
            out_lines.append("")
        else:
            out_lines.append(lines[i])
        i += 1
    return "\n".join(out_lines), dirs


def dim(s):
    nums = [int(x, 0) for x in re.findall(r"0x[0-9a-fA-F]+|\d+", s)]
    return tuple((nums + [1, 1, 1])[:3])


def parse_launch(body):
    m = re.match(r"(\S+)\s*<<<(.*)>>>\s*\((.*)\)\s*$", body, re.S)
    if not m:
        raise Unsupported("launch syntax")
    name, cfg, args = m.groups()
    parts, depth, cur = [], 0, ""
    for c in cfg:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if c == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    parts.append(cur)
    grid, block = dim(parts[0]), dim(parts[1])
    shared = int(parts[2].strip(), 0) if len(parts) > 2 else 0
    return name, grid, block, shared, split_list(args)


def kernel_params(module, name):
    """Declared .param types of an .entry, in order."""
    m = re.search(r"\.entry\s+" + re.escape(name) + r"\s*\((.*?)\)", module, re.S)
    if not m:
        return []
    out = []
    for p in m.group(1).split(","):
        t = re.search(r"\.param\s+(?:\.align\s+\d+\s+)?(?:\.ptr\s+(?:\.\w+\s+)?(?:\.align\s+\d+\s+)?)?\.(\w+)", p)
        arr = re.search(r"\[(\d+)\]", p)
        if t:
            out.append((t.group(1), int(arr.group(1)) if arr else None))
    return out


# --------------------------------------------------------------------------- CUDA driver API
class Cuda:
    def __init__(self, lib):
        self.l = ctypes.CDLL(lib)
        for f in ("cuInit", "cuDeviceGet", "cuCtxCreate_v2", "cuModuleLoadData", "cuModuleGetFunction",
                  "cuMemAlloc_v2", "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2", "cuLaunchKernel", "cuCtxSynchronize",
                  "cuModuleGetGlobal_v2", "cuMemsetD8_v2"):
            setattr(self, f, getattr(self.l, f))
        self.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        self.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        self.cuMemsetD8_v2.argtypes = [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t]
        self.check(self.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self.check(self.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        self.check(self.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

    def last_error(self):
        try:
            f = self.l.zludaMetalGetLastError
            f.restype = ctypes.c_size_t
            f.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
            buf = ctypes.create_string_buffer(1 << 16)
            f(buf, len(buf))
            return buf.value.decode(errors="replace")
        except AttributeError:
            return ""

    def check(self, r, what):
        if r != 0:
            raise RuntimeError("%s failed (%d): %s" % (what, r, self.last_error()))

    def alloc(self, data):
        p = ctypes.c_uint64()
        self.check(self.cuMemAlloc_v2(ctypes.byref(p), max(len(data), 1)), "cuMemAlloc")
        if data:
            self.check(self.cuMemcpyHtoD_v2(p.value, data, len(data)), "cuMemcpyHtoD")
        return p.value

    def read(self, p, n):
        b = ctypes.create_string_buffer(n)
        self.check(self.cuMemcpyDtoH_v2(b, p, n), "cuMemcpyDtoH")
        return b.raw


class Buffer:
    def __init__(self, name, ty, data):
        self.name, self.ty, self.data = name, ty, bytearray(data)
        self.ptr = None


def pack_vals(vals, ty):
    size = SIZES[ty]
    return b"".join(v.to_bytes(size, "little") for v in vals)


def run_program(path, lib):
    text = open(path, errors="replace").read()
    module, dirs = split_program(text)
    for d, _ in dirs:
        if d in UNSUPPORTED:
            return {"status": "UNSUPPORTED", "detail": "." + d}
    if not any(d in ("launch", "cooplaunch") for d, _ in dirs):
        return {"status": "UNSUPPORTED", "detail": "no .launch"}
    try:
        cuda = Cuda(lib)
    except Exception as e:
        return {"status": "CRASH", "detail": "init: %s" % e}
    mod = ctypes.c_void_p()
    src = module.encode() + b"\0"
    r = cuda.cuModuleLoadData(ctypes.byref(mod), src)
    if r != 0:
        return {"status": "COMPILE", "detail": cuda.last_error()[-1500:]}
    bufs, last, checks, nexp = {}, None, [], 0
    try:
        for d, body in dirs:
            if d == "buffer":
                m = re.match(r"(\w+)\s+\.(\w+)\s*(?:\[\s*(\w+)\s*\])?\s*(?:=\s*(\{.*\}))?\s*$", body, re.S)
                if not m:
                    raise Unsupported("buffer syntax")
                name, ty, n, init = m.groups()
                if ty not in SIZES:
                    raise Unsupported("buffer type " + ty)
                vals = [parse_value(v, ty) for v in split_list(init)] if init else []
                count = int(n, 0) if n else len(vals)
                data = pack_vals(vals, ty).ljust(count * SIZES[ty], b"\0")
                bufs[name] = last = Buffer(name, ty, data)
            elif d == "append":
                m = re.match(r"(\w+)\s+\.(\w+)\s*=\s*(\{.*\})\s*$", body, re.S)
                b = bufs[m.group(1)]
                b.data += pack_vals([parse_value(v, m.group(2)) for v in split_list(m.group(3))], m.group(2))
            elif d == "patch":
                m = re.match(r"(\w+)\s+\.(\w+)\s*@\s*(\w+)\s*=\s*(\{.*\})\s*$", body, re.S)
                name, ty, off, init = m.groups()
                b = bufs[name]
                data = pack_vals([parse_value(v, ty) for v in split_list(init)], ty)
                o = int(off, 0) * SIZES[ty]
                if b.ptr is None:
                    if len(b.data) < o + len(data):
                        b.data += bytes(o + len(data) - len(b.data))
                    b.data[o:o + len(data)] = data
                else:
                    cuda.check(cuda.cuMemcpyHtoD_v2(b.ptr + o, bytes(data), len(data)), "patch")
            elif d in ("launch", "cooplaunch"):
                name, grid, block, shared, args = parse_launch(body)
                fn = ctypes.c_void_p()
                cuda.check(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()), "cuModuleGetFunction " + name)
                ptys = kernel_params(module, name)
                holders = []
                for i, a in enumerate(args):
                    if a in bufs:
                        b = bufs[a]
                        if b.ptr is None:
                            b.ptr = cuda.alloc(bytes(b.data))
                        v = b.ptr.to_bytes(8, "little")
                    else:
                        ty = ptys[i][0] if i < len(ptys) else "u64"
                        if ty not in SIZES:
                            raise Unsupported("param type " + ty)
                        v = parse_value(a, ty).to_bytes(SIZES[ty], "little").ljust(16, b"\0")
                    holders.append(ctypes.create_string_buffer(bytes(v), max(len(v), 16)))
                arr = (ctypes.c_void_p * max(len(holders), 1))(*[ctypes.cast(h, ctypes.c_void_p) for h in holders])
                r = cuda.cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared, None,
                                        arr, None)
                if r != 0 or cuda.cuCtxSynchronize() != 0:
                    return {"status": "LAUNCH", "detail": "%s: %s" % (name, cuda.last_error()[-1500:])}
            elif d == "expect":
                m = re.match(r"(\w+)\s+\.(\w+)\s*(?:@\s*(\w+))?\s*=\s*(\{.*\})\s*$", body, re.S)
                if not m:
                    raise Unsupported("expect syntax")
                name, ty, off, init = m.groups()
                want = [parse_value(v, ty) for v in split_list(init)]
                o = int(off, 0) * SIZES[ty] if off else 0
                n = len(want) * SIZES[ty]
                if name in bufs:
                    b = bufs[name]
                    got = cuda.read(b.ptr, n + o)[o:] if b.ptr is not None else bytes(b.data[o:o + n])
                else:
                    gp, gs = ctypes.c_uint64(), ctypes.c_size_t()
                    cuda.check(cuda.cuModuleGetGlobal_v2(ctypes.byref(gp), ctypes.byref(gs), mod, name.encode()),
                               "cuModuleGetGlobal " + name)
                    got = cuda.read(gp.value + o, n)
                gv = [int.from_bytes(got[i * SIZES[ty]:(i + 1) * SIZES[ty]], "little") for i in range(len(want))]
                nexp += 1
                if gv != want:
                    bad = [(i, hex(g), hex(w)) for i, (g, w) in enumerate(zip(gv, want)) if g != w]
                    checks.append("%s: %d/%d differ, first %s" % (name, len(bad), len(want), bad[:4]))
    except Unsupported as e:
        return {"status": "UNSUPPORTED", "detail": str(e)}
    except (KeyError, AttributeError, ValueError, RuntimeError) as e:
        return {"status": "HARNESS" if not isinstance(e, RuntimeError) else "LAUNCH", "detail": repr(e)[-800:]}
    if checks:
        return {"status": "FAIL", "detail": "; ".join(checks)[-1500:]}
    return {"status": "PASS", "detail": "%d expects" % nexp}


def child(path, lib, timeout, env):
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", path, "--lib", lib],
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "detail": "> %ds" % timeout}
    for line in reversed(r.stdout.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                pass
    return {"status": "CRASH", "detail": (r.stdout + r.stderr)[-800:]}


def collect(paths, k):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".ptx"))
        else:
            out.append(p)
    return [p for p in out if not k or any(s in p for s in k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--lib", default=DEFAULT_LIB)
    ap.add_argument("--backend", default="agx")
    ap.add_argument("--one")
    ap.add_argument("--json")
    ap.add_argument("-j", type=int, default=os.cpu_count())
    ap.add_argument("-k", action="append")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    if a.one:
        # the translator recurses deeply: run on a thread with a large stack
        import threading
        out = {}

        def body():
            try:
                out["r"] = run_program(a.one, a.lib)
            except Unsupported as e:
                out["r"] = {"status": "UNSUPPORTED", "detail": str(e)}
            except Exception as e:  # noqa: BLE001
                out["r"] = {"status": "HARNESS", "detail": repr(e)[-800:]}
        threading.stack_size(512 << 20)
        t = threading.Thread(target=body)
        t.start()
        t.join()
        print(json.dumps(out.get("r", {"status": "CRASH", "detail": "no result"})))
        sys.stdout.flush()
        os._exit(0)
    env = dict(os.environ, ZLUDA_METAL_BACKEND=a.backend)
    # a sibling llvm-project checkout's build (the AGX backend's fork)
    sibling = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "llvm-project", "build", "bin",
                                            "llvm-air2agx"))
    if "ZLUDA_AGX_AIR2AGX" not in env and os.path.exists(sibling):
        env["ZLUDA_AGX_AIR2AGX"] = sibling
    files = collect(a.paths, a.k)
    results = {}
    with cf.ThreadPoolExecutor(a.j) as ex:
        futs = {ex.submit(child, f, a.lib, a.timeout, env): f for f in files}
        for fu in cf.as_completed(futs):
            f = futs[fu]
            results[f] = fu.result()
            r = results[f]
            if a.v or r["status"] not in ("PASS",):
                print("%-11s %s  %s" % (r["status"], os.path.relpath(f), r["detail"].replace("\n", " ")[:300]),
                      flush=True)
    counts = {}
    for r in results.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("----", ", ".join("%s %d" % kv for kv in sorted(counts.items())), "of", len(results))
    if a.json:
        json.dump({os.path.relpath(k): v for k, v in sorted(results.items())}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
