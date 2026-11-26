#!/usr/bin/env python3
"""Generates kernels.mlir: the device code of ZLUDA's cuBLAS for Apple GPUs, in MLIR's GPU dialect.

The kernels are compiled by the LLVM fork's MLIR (GPU -> AIR, convert-gpu-to-air, then gpu-module-to-binary with
#air.target<chip = "agx-g14c">: AIR's portable passes, which emulate doubles, and the AGX backend, giving an AGX
object), see build.sh and build.rs. One template per BLAS routine, instantiated per element type: s (f32),
d (f64: AIR has no double, llvm-air2agx emulates it), and for GEMM the mixed forms of cublasGemmEx (f16 / bf16
inputs, f32 accumulation).

Conventions (src/blas.rs launches them):
* every memref is a bare device pointer (use-bare-ptr-memref-call-conv), every scalar an i32 or the element type;
  each is one kernel argument, in order (the AGX record: a pointer's address or the scalar's bytes in uniforms);
* vectors are (pointer, offset, increment) in elements: element i is x[off + i * inc] (the host turns BLAS's
  negative increments into an offset);
* matrices are column-major (pointer, offset, row stride, column stride): A(i, j) = A[off + i * rs + j * cs], so
  op(A) = A^T is the same kernel with the strides swapped;
* reductions are two launches: blocks of one SIMD group (32 threads) write partial results, then one block reduces
  them (deterministic: the same order for a given n).

  gen_kernels.py > kernels.mlir   (build.sh runs it)
"""

TYPES = {"s": "f32", "d": "f64"}


def hdr():
    return "module attributes {gpu.container_module} {\n  gpu.module @zluda_blas [#air.target<chip = \"agx-g14c\">] {\n"


def ftr():
    return "  }\n}\n"


def gid():
    """global thread index (i32) and the grid's thread count"""
    return """      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %b0 = arith.muli %bid, %bdim : index
      %gi = arith.addi %b0, %tid : index
      %gn = arith.muli %gdim, %bdim : index
      %i = arith.index_cast %gi : index to i32
      %stride = arith.index_cast %gn : index to i32
"""


def idx(name, off, i, inc):
    """%name = index of off + i * inc"""
    return """      %{n}_m = arith.muli %{i}, %{inc} : i32
      %{n}_a = arith.addi %{off}, %{n}_m : i32
      %{n} = arith.index_cast %{n}_a : i32 to index
""".format(n=name, off=off, i=i, inc=inc)


def axpy(p, t):
    """y = alpha x + y"""
    return """    gpu.func @{p}axpy(%n: i32, %alpha: {t}, %x: memref<?x{t}, 1>, %xo: i32, %xi: i32, %y: memref<?x{t}, 1>, %yo: i32, %yi: i32) kernel {{
{gid}      scf.for %k = %i to %n step %stride : i32 {{
{ix}{iy}        %xv = memref.load %x[%ix] : memref<?x{t}, 1>
        %yv = memref.load %y[%iy] : memref<?x{t}, 1>
        %r = math.fma %alpha, %xv, %yv : {t}
        memref.store %r, %y[%iy] : memref<?x{t}, 1>
      }}
      gpu.return
    }}
""".format(p=p, t=t, gid=gid(), ix=idx("ix", "xo", "k", "xi"), iy=idx("iy", "yo", "k", "yi"))


def scal(p, t):
    """x = alpha x"""
    return """    gpu.func @{p}scal(%n: i32, %alpha: {t}, %x: memref<?x{t}, 1>, %xo: i32, %xi: i32) kernel {{
{gid}      scf.for %k = %i to %n step %stride : i32 {{
{ix}        %xv = memref.load %x[%ix] : memref<?x{t}, 1>
        %r = arith.mulf %alpha, %xv : {t}
        memref.store %r, %x[%ix] : memref<?x{t}, 1>
      }}
      gpu.return
    }}
""".format(p=p, t=t, gid=gid(), ix=idx("ix", "xo", "k", "xi"))


def copy_swap(p, t):
    """y = x (mode 0), or exchange x and y (mode 1)"""
    return """    gpu.func @{p}copy(%n: i32, %mode: i32, %x: memref<?x{t}, 1>, %xo: i32, %xi: i32, %y: memref<?x{t}, 1>, %yo: i32, %yi: i32) kernel {{
{gid}      %c0 = arith.constant 0 : i32
      %sw = arith.cmpi ne, %mode, %c0 : i32
      scf.for %k = %i to %n step %stride : i32 {{
{ix}{iy}        %xv = memref.load %x[%ix] : memref<?x{t}, 1>
        scf.if %sw {{
          %yv = memref.load %y[%iy] : memref<?x{t}, 1>
          memref.store %yv, %x[%ix] : memref<?x{t}, 1>
        }}
        memref.store %xv, %y[%iy] : memref<?x{t}, 1>
      }}
      gpu.return
    }}
""".format(p=p, t=t, gid=gid(), ix=idx("ix", "xo", "k", "xi"), iy=idx("iy", "yo", "k", "yi"))


def reduce(p, t):
    """partial sums of one SIMD group per block: op 0 dot (x . y), 1 sum of squares, 2 sum of |x|; out[block].
    The final pass is the same kernel on the partials (op 3: plain sum, increments 1)."""
    return """    gpu.func @{p}reduce(%n: i32, %op: i32, %x: memref<?x{t}, 1>, %xo: i32, %xi: i32, %y: memref<?x{t}, 1>, %yo: i32, %yi: i32, %out: memref<?x{t}, 1>) kernel {{
{gid}      %zero = arith.constant 0.0 : {t}
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %acc = scf.for %k = %i to %n step %stride iter_args(%a = %zero) -> ({t}) : i32 {{
{ix}        %xv = memref.load %x[%ix] : memref<?x{t}, 1>
        %isdot = arith.cmpi eq, %op, %c0 : i32
        %v = scf.if %isdot -> ({t}) {{
{iy}          %yv = memref.load %y[%iy] : memref<?x{t}, 1>
          %m = arith.mulf %xv, %yv : {t}
          scf.yield %m : {t}
        }} else {{
          %issq = arith.cmpi eq, %op, %c1 : i32
          %sq = arith.mulf %xv, %xv : {t}
          %isabs = arith.cmpi eq, %op, %c2 : i32
          %ab = math.absf %xv : {t}
          %w0 = arith.select %isabs, %ab, %xv : {t}
          %w = arith.select %issq, %sq, %w0 : {t}
          scf.yield %w : {t}
        }}
        %na = arith.addf %a, %v : {t}
        scf.yield %na : {t}
      }}
      %s = gpu.subgroup_reduce add %acc : ({t}) -> {t}
      %lane = gpu.lane_id
      %c0i = arith.constant 0 : index
      %first = arith.cmpi eq, %lane, %c0i : index
      scf.if %first {{
        memref.store %s, %out[%bid] : memref<?x{t}, 1>
      }}
      gpu.return
    }}
""".format(p=p, t=t, gid=gid(), ix=idx("ix", "xo", "k", "xi"), iy=idx("iy", "yo", "k", "yi"))


def amax(p, t):
    """index of the first element of largest (op 0) / smallest (op 1) |x|, per block of one SIMD group: the
    partial value in out[block], its 1-based index in outi[block]. The final pass (op 2 / 3) reads the partials as
    (value, index) pairs: x = values, y = indices (as the element type: exact below 2^24 / 2^53)."""
    return """    gpu.func @{p}amax(%n: i32, %op: i32, %x: memref<?x{t}, 1>, %xo: i32, %xi: i32, %y: memref<?x{t}, 1>, %out: memref<?x{t}, 1>, %outi: memref<?x{t}, 1>) kernel {{
{gid}      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %opm = arith.remui %op, %c2 : i32
      %ismin = arith.cmpi eq, %opm, %c1 : i32
      %final = arith.cmpi uge, %op, %c2 : i32
      %inf = arith.constant 0x7F800000 : i32
      %pinf = arith.bitcast %inf : i32 to f32
      %pinft = {conv_inf}
      %neg1 = arith.constant -1.0 : {t}
      %start = arith.select %ismin, %pinft, %neg1 : {t}
      %big = arith.constant 2147483647 : i32
      %r:2 = scf.for %k = %i to %n step %stride iter_args(%bv = %start, %bi = %big) -> ({t}, i32) : i32 {{
{ix}        %xv = memref.load %x[%ix] : memref<?x{t}, 1>
        %av0 = math.absf %xv : {t}
        %av = arith.select %final, %xv, %av0 : {t}
        %ki = arith.index_cast %k : i32 to index
        %pi = scf.if %final -> (i32) {{
          %yv = memref.load %y[%ki] : memref<?x{t}, 1>
          %yi = arith.fptosi %yv : {t} to i32
          scf.yield %yi : i32
        }} else {{
          %k1 = arith.addi %k, %c1 : i32
          scf.yield %k1 : i32
        }}
        %gt = arith.cmpf ogt, %av, %bv : {t}
        %lt = arith.cmpf olt, %av, %bv : {t}
        %better = arith.select %ismin, %lt, %gt : i1
        %nv = arith.select %better, %av, %bv : {t}
        %ni = arith.select %better, %pi, %bi : i32
        scf.yield %nv, %ni : {t}, i32
      }}
      // the best value of the SIMD group, then the smallest index that has it
      %mx = gpu.subgroup_reduce maxnumf %r#0 : ({t}) -> {t}
      %mn = gpu.subgroup_reduce minnumf %r#0 : ({t}) -> {t}
      %bestv = arith.select %ismin, %mn, %mx : {t}
      %has = arith.cmpf oeq, %r#0, %bestv : {t}
      %cand = arith.select %has, %r#1, %big : i32
      %besti = gpu.subgroup_reduce minsi %cand : (i32) -> i32
      %lane = gpu.lane_id
      %c0i = arith.constant 0 : index
      %first = arith.cmpi eq, %lane, %c0i : index
      scf.if %first {{
        memref.store %bestv, %out[%bid] : memref<?x{t}, 1>
        %bif = arith.sitofp %besti : i32 to {t}
        memref.store %bif, %outi[%bid] : memref<?x{t}, 1>
      }}
      gpu.return
    }}
""".format(p=p, t=t, gid=gid(), ix=idx("ix", "xo", "k", "xi"),
           conv_inf="arith.extf %pinf : f32 to f64" if t == "f64" else "arith.addf %pinf, %pinf : f32")


def gemv(p, t):
    """y = alpha op(A) x + beta y; op(A) is m x n with A(i, j) = A[ao + i rs + j cs]; one thread per y element"""
    return """    gpu.func @{p}gemv(%m: i32, %n: i32, %alpha: {t}, %a: memref<?x{t}, 1>, %ao: i32, %ars: i32, %acs: i32, %x: memref<?x{t}, 1>, %xo: i32, %xi: i32, %beta: {t}, %y: memref<?x{t}, 1>, %yo: i32, %yi: i32) kernel {{
{gid}      %zero = arith.constant 0.0 : {t}
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      scf.for %r = %i to %m step %stride : i32 {{
        %rr = arith.muli %r, %ars : i32
        %rowbase = arith.addi %ao, %rr : i32
        %acc = scf.for %j = %c0 to %n step %c1 iter_args(%s = %zero) -> ({t}) : i32 {{
{ia}{ix}          %av = memref.load %a[%ia] : memref<?x{t}, 1>
          %xv = memref.load %x[%ix] : memref<?x{t}, 1>
          %ns = math.fma %av, %xv, %s : {t}
          scf.yield %ns : {t}
        }}
{iy}        %bz = arith.cmpf oeq, %beta, %zero : {t}
        %yv = scf.if %bz -> ({t}) {{
          scf.yield %zero : {t}
        }} else {{
          %y0 = memref.load %y[%iy] : memref<?x{t}, 1>
          %by = arith.mulf %beta, %y0 : {t}
          scf.yield %by : {t}
        }}
        %res = math.fma %alpha, %acc, %yv : {t}
        memref.store %res, %y[%iy] : memref<?x{t}, 1>
      }}
      gpu.return
    }}
""".format(p=p, t=t, gid=gid(), ia=idx("ia", "rowbase", "j", "acs").replace("      %", "          %"),
           ix=idx("ix", "xo", "j", "xi").replace("      %", "          %"), iy=idx("iy", "yo", "r", "yi"))


def gemm(name, ta, tb, tc, tacc):
    """C = alpha op(A) op(B) + beta C, column-major, batched over grid z (strides in elements between matrices).
    op(A) is m x k: A(i, l) = A[ao + i ars + l acs]; op(B) k x n: B(l, j) = B[bo + l brs + j bcs];
    C(i, j) = C[co + i + j ldc]. 2D blocks; each thread computes a 4 x 4 tile of C (rows i0 + 0..3 * by, columns
    j0 + 0..3 * bx... see below), accumulating in tacc."""
    def ld(mem, t, i):
        v = "memref.load %{m}[%{i}] : memref<?x{t}, 1>".format(m=mem, i=i, t=t)
        return v
    ext = lambda t: "" if t == tacc else ("arith.extf %{v} : " + t + " to " + tacc)
    lines = []
    w = lines.append
    w("""    gpu.func @{n}(%m: i32, %n: i32, %k: i32, %alpha: {acc}, %a: memref<?x{ta}, 1>, %ao: i32, %ars: i32, %acs: i32, %sa: i32, %b: memref<?x{tb}, 1>, %bo: i32, %brs: i32, %bcs: i32, %sb: i32, %beta: {acc}, %c: memref<?x{tc}, 1>, %co: i32, %ldc: i32, %sc: i32) kernel {{
      %tx = gpu.thread_id x
      %ty = gpu.thread_id y
      %bx = gpu.block_id x
      %byi = gpu.block_id y
      %bz = gpu.block_id z
      %dx = gpu.block_dim x
      %dy = gpu.block_dim y
      %txi = arith.index_cast %tx : index to i32
      %tyi = arith.index_cast %ty : index to i32
      %bxi = arith.index_cast %bx : index to i32
      %byy = arith.index_cast %byi : index to i32
      %bzi = arith.index_cast %bz : index to i32
      %dxi = arith.index_cast %dx : index to i32
      %dyi = arith.index_cast %dy : index to i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c4 = arith.constant 4 : i32
      %zero = arith.constant 0.0 : {acc}
      // the block covers 4 dx rows and 4 dy columns; thread (tx, ty) rows i0 + r dx, columns j0 + q dy
      %rb = arith.muli %bxi, %dxi : i32
      %rb4 = arith.muli %rb, %c4 : i32
      %i0 = arith.addi %rb4, %txi : i32
      %cb = arith.muli %byy, %dyi : i32
      %cb4 = arith.muli %cb, %c4 : i32
      %j0 = arith.addi %cb4, %tyi : i32
      %zsa = arith.muli %bzi, %sa : i32
      %abase = arith.addi %ao, %zsa : i32
      %zsb = arith.muli %bzi, %sb : i32
      %bbase = arith.addi %bo, %zsb : i32
      %zsc = arith.muli %bzi, %sc : i32
      %cbase = arith.addi %co, %zsc : i32""".format(n=name, ta=ta, tb=tb, tc=tc, acc=tacc))
    # rows and columns, clamped for the loads (results stored only in range)
    w("      %mm1 = arith.subi %m, %c1 : i32\n      %nm1 = arith.subi %n, %c1 : i32")
    for r in range(4):
        w("""      %cr{r} = arith.constant {r} : i32
      %ro{r} = arith.muli %cr{r}, %dxi : i32
      %row{r} = arith.addi %i0, %ro{r} : i32
      %rowc{r} = arith.minsi %row{r}, %mm1 : i32
      %rofs{r} = arith.muli %rowc{r}, %ars : i32
      %arow{r} = arith.addi %abase, %rofs{r} : i32
      %co{r} = arith.muli %cr{r}, %dyi : i32
      %col{r} = arith.addi %j0, %co{r} : i32
      %colc{r} = arith.minsi %col{r}, %nm1 : i32
      %cofs{r} = arith.muli %colc{r}, %bcs : i32
      %bcol{r} = arith.addi %bbase, %cofs{r} : i32""".format(r=r))
    accs = ["%%acc%d%d" % (r, q) for r in range(4) for q in range(4)]
    w("      %%res:16 = scf.for %%l = %%c0 to %%k step %%c1 iter_args(%s) -> (%s) : i32 {" % (
        ", ".join("%s = %%zero" % a.replace("%acc", "%s") for a in accs), ", ".join([tacc] * 16)))
    w("        %la = arith.muli %l, %acs : i32\n        %lb = arith.muli %l, %brs : i32")
    for r in range(4):
        w("        %%ai%d = arith.addi %%arow%d, %%la : i32\n        %%aix%d = arith.index_cast %%ai%d : i32 to index" % (r, r, r, r))
        w("        %%av%d_ = %s" % (r, ld("a", ta, "aix%d" % r)))
        w("        %%av%d = %s" % (r, ext(ta).format(v="av%d_" % r)) if ext(ta) else "        %%av%d = arith.addf %%av%d_, %%zero : %s" % (r, r, tacc))
        w("        %%bi%d = arith.addi %%bcol%d, %%lb : i32\n        %%bix%d = arith.index_cast %%bi%d : i32 to index" % (r, r, r, r))
        w("        %%bv%d_ = %s" % (r, ld("b", tb, "bix%d" % r)))
        w("        %%bv%d = %s" % (r, ext(tb).format(v="bv%d_" % r)) if ext(tb) else "        %%bv%d = arith.addf %%bv%d_, %%zero : %s" % (r, r, tacc))
    outs = []
    for r in range(4):
        for q in range(4):
            w("        %%n%d%d = math.fma %%av%d, %%bv%d, %%s%d%d : %s" % (r, q, r, q, r, q, tacc))
            outs.append("%%n%d%d" % (r, q))
    w("        scf.yield %s : %s\n      }" % (", ".join(outs), ", ".join([tacc] * 16)))
    w("      %bzero = arith.cmpf oeq, %beta, %zero : " + tacc)
    for r in range(4):
        for q in range(4):
            j = r * 4 + q
            w("""      %inr{r}{q} = arith.cmpi slt, %row{r}, %m : i32
      %inc{r}{q} = arith.cmpi slt, %col{q}, %n : i32
      %in{r}{q} = arith.andi %inr{r}{q}, %inc{r}{q} : i1
      scf.if %in{r}{q} {{
        %cl{r}{q} = arith.muli %col{q}, %ldc : i32
        %cc{r}{q} = arith.addi %cbase, %cl{r}{q} : i32
        %ci{r}{q} = arith.addi %cc{r}{q}, %row{r} : i32
        %cix{r}{q} = arith.index_cast %ci{r}{q} : i32 to index
        %old{r}{q} = scf.if %bzero -> ({acc}) {{
          scf.yield %zero : {acc}
        }} else {{
          %o_{r}{q} = memref.load %c[%cix{r}{q}] : memref<?x{tc}, 1>
          {oext}
          %bo{r}{q} = arith.mulf %beta, %oe{r}{q} : {acc}
          scf.yield %bo{r}{q} : {acc}
        }}
        %v{r}{q} = math.fma %alpha, %res#{j}, %old{r}{q} : {acc}
        {vtrunc}
        memref.store %vt{r}{q}, %c[%cix{r}{q}] : memref<?x{tc}, 1>
      }}""".format(r=r, q=q, j=j, acc=tacc, tc=tc,
                   oext=("%%oe%d%d = arith.extf %%o_%d%d : %s to %s" % (r, q, r, q, tc, tacc)) if tc != tacc
                   else ("%%oe%d%d = arith.addf %%o_%d%d, %%zero : %s" % (r, q, r, q, tacc)),
                   vtrunc=("%%vt%d%d = arith.truncf %%v%d%d : %s to %s" % (r, q, r, q, tacc, tc)) if tc != tacc
                   else ("%%vt%d%d = arith.addf %%v%d%d, %%zero : %s" % (r, q, r, q, tacc))))
    w("      gpu.return\n    }\n")
    return "\n".join(lines)


def main():
    out = [hdr()]
    for p, t in TYPES.items():
        for f in (axpy, scal, copy_swap, reduce, amax, gemv):
            out.append(f(p, t))
        out.append(gemm(p + "gemm", t, t, t, t))
    # cublasGemmEx / Hgemm: half or bfloat inputs, float accumulation, the output in either
    for a in ("f16", "bf16"):
        for c in (a, "f32"):
            out.append(gemm("gemm_%s_%s" % (a, c), a, a, c, "f32"))
    out.append(ftr())
    print("".join(out))


main()
