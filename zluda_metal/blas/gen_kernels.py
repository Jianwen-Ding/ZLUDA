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


BIG = "1073741824"  # static size of the pointer arguments a reinterpret_cast reads (bare pointers: the size is not used)


def gemm_mma(name, ta, rb=2, qb=2):
    """The SIMD-group matrix GEMM (air.simdgroup_matrix_*, AGX's 8 x 8 matrix multiply): C (f32) = alpha op(A)
    op(B) + beta C for m a multiple of 8 rb, n of 8 qb and k of 8 (src/blas.rs chooses it). One SIMD group (the
    threadgroup) computes an 8 rb x 8 qb tile of C as rb x qb 8 x 8 accumulators over k in steps of 8. Matrices
    cannot be scaled elementwise, so alpha and beta are products with alpha I and beta I (`diag`: a 16 x 8 f32
    buffer, rows 0-7 alpha I, rows 8-15 beta I); C is not read when beta is 0."""
    mt = "memref<?x?x{t}, strided<[?, ?], offset: ?>, 1>"
    am, cm = mt.format(t=ta), mt.format(t="f32")
    dm = "memref<16x8xf32, strided<[8, 1]>, 1>"
    v = "vector<64x%s>" % ta
    mmt = v + ", vector<64xf32>" if ta != "f32" else "vector<64xf32>"
    f = "vector<64xf32>"
    w = []
    w.append("""    gpu.func @{n}(%m: i32, %n: i32, %k: i32, %diag: memref<128xf32, 1>, %a: memref<{big}x{ta}, 1>, %ao: i32, %ars: i32, %acs: i32, %sa: i32, %b: memref<{big}x{ta}, 1>, %bo: i32, %brs: i32, %bcs: i32, %sb: i32, %bnz: i32, %c: memref<{big}xf32, 1>, %co: i32, %ldc: i32, %sc: i32) kernel {{
      %bx = gpu.block_id x
      %by = gpu.block_id y
      %bz = gpu.block_id z
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %th = arith.constant {th} : index
      %tw = arith.constant {tw} : index
      %i0 = arith.muli %bx, %th : index
      %j0 = arith.muli %by, %tw : index
      %bzi = arith.index_cast %bz : index to i32
      %mi = arith.index_cast %m : i32 to index
      %ni = arith.index_cast %n : i32 to index
      %ki = arith.index_cast %k : i32 to index
      %zsa = arith.muli %bzi, %sa : i32
      %ab = arith.addi %ao, %zsa : i32
      %zsb = arith.muli %bzi, %sb : i32
      %bb = arith.addi %bo, %zsb : i32
      %zsc = arith.muli %bzi, %sc : i32
      %cb = arith.addi %co, %zsc : i32
      %abi = arith.index_cast %ab : i32 to index
      %bbi = arith.index_cast %bb : i32 to index
      %cbi = arith.index_cast %cb : i32 to index
      %arsi = arith.index_cast %ars : i32 to index
      %acsi = arith.index_cast %acs : i32 to index
      %brsi = arith.index_cast %brs : i32 to index
      %bcsi = arith.index_cast %bcs : i32 to index
      %ldci = arith.index_cast %ldc : i32 to index
      %A = memref.reinterpret_cast %a to offset: [%abi], sizes: [%mi, %ki], strides: [%arsi, %acsi] : memref<{big}x{ta}, 1> to {am}
      %B = memref.reinterpret_cast %b to offset: [%bbi], sizes: [%ki, %ni], strides: [%brsi, %bcsi] : memref<{big}x{ta}, 1> to {am}
      %C = memref.reinterpret_cast %c to offset: [%cbi], sizes: [%mi, %ni], strides: [%c1, %ldci] : memref<{big}xf32, 1> to {cm}
      %D = memref.reinterpret_cast %diag to offset: [0], sizes: [16, 8], strides: [8, 1] : memref<128xf32, 1> to {dm}
      %zf = arith.constant 0.0 : f32
      %z = air.simdgroup_matrix_fill %zf : {f}""".format(n=name, ta=ta, big=BIG, am=am, cm=cm, dm=dm, f=f,
                                                         th=8 * rb, tw=8 * qb))
    for r in range(rb):
        w.append("      %%ro%d = arith.constant %d : index\n      %%i%d = arith.addi %%i0, %%ro%d : index" % (r, 8 * r, r + 1, r) if r else "      %i1 = arith.addi %i0, %c0 : index")
    for q in range(qb):
        w.append("      %%qo%d = arith.constant %d : index\n      %%j%d = arith.addi %%j0, %%qo%d : index" % (q, 8 * q, q + 1, q) if q else "      %j1 = arith.addi %j0, %c0 : index")
    tiles = [(r, q) for r in range(rb) for q in range(qb)]
    w.append("      %%r:%d = scf.for %%l = %%c0 to %%ki step %%c8 iter_args(%s) -> (%s) {" % (
        len(tiles), ", ".join("%%s%d_%d = %%z" % t for t in tiles), ", ".join([f] * len(tiles))))
    for r in range(rb):
        w.append("        %%a%d = air.simdgroup_matrix_load %%A[%%i%d, %%l] : %s -> %s" % (r, r + 1, am, v))
    for q in range(qb):
        w.append("        %%b%d = air.simdgroup_matrix_load %%B[%%l, %%j%d] : %s -> %s" % (q, q + 1, am, v))
    for r, q in tiles:
        w.append("        %%n%d_%d = air.simdgroup_matrix_multiply_accumulate %%a%d, %%b%d, %%s%d_%d : %s" % (r, q, r, q, r, q, mmt))
    w.append("        scf.yield %s : %s\n      }" % (", ".join("%%n%d_%d" % t for t in tiles), ", ".join([f] * len(tiles))))
    w.append("""      %da = air.simdgroup_matrix_load %D[%c0, %c0] : {dm} -> {f}
      %db = air.simdgroup_matrix_load %D[%c8, %c0] : {dm} -> {f}
      %c0i = arith.constant 0 : i32
      %withc = arith.cmpi ne, %bnz, %c0i : i32""".format(dm=dm, f=f))
    for withc in (True, False):
        w.append("      scf.if %withc {" if withc else "      } else {")
        for idx, (r, q) in enumerate(tiles):
            t = "%d_%d" % (r, q)
            sfx = "c" if withc else "n"
            w.append("        %%x%s%s = air.simdgroup_matrix_multiply_accumulate %%r#%d, %%da, %%z : %s" % (t, sfx, idx, f))
            if withc:
                w.append("        %%o%s = air.simdgroup_matrix_load %%C[%%i%d, %%j%d] : %s -> %s" % (t, r + 1, q + 1, cm, f))
                w.append("        %%y%s = air.simdgroup_matrix_multiply_accumulate %%o%s, %%db, %%x%sc : %s" % (t, t, t, f))
                w.append("        air.simdgroup_matrix_store %%y%s, %%C[%%i%d, %%j%d] : %s -> %s" % (t, r + 1, q + 1, f, cm))
            else:
                w.append("        air.simdgroup_matrix_store %%x%sn, %%C[%%i%d, %%j%d] : %s -> %s" % (t, r + 1, q + 1, f, cm))
    w.append("      }\n      gpu.return\n    }\n")
    return "\n".join(w)


def g2d():
    """2D element loop setup: %i (rows, grid x, one per thread) and %j0 / %js (columns, grid y stride loop)"""
    return """      %tx = gpu.thread_id x
      %ty = gpu.thread_id y
      %bx = gpu.block_id x
      %by = gpu.block_id y
      %dx = gpu.block_dim x
      %dy = gpu.block_dim y
      %gy = gpu.grid_dim y
      %rx = arith.muli %bx, %dx : index
      %ri = arith.addi %rx, %tx : index
      %i = arith.index_cast %ri : index to i32
      %cy = arith.muli %by, %dy : index
      %cj = arith.addi %cy, %ty : index
      %j0 = arith.index_cast %cj : index to i32
      %gyd = arith.muli %gy, %dy : index
      %js = arith.index_cast %gyd : index to i32
"""


def rot(p, t):
    """the plane rotation / modified rotation as a 2 x 2 transform: x' = h11 x + h12 y, y' = h21 x + h22 y"""
    return """    gpu.func @{p}rot(%n: i32, %h11: {t}, %h21: {t}, %h12: {t}, %h22: {t}, %x: memref<?x{t}, 1>, %xo: i32, %xi: i32, %y: memref<?x{t}, 1>, %yo: i32, %yi: i32) kernel {{
{gid}      scf.for %k = %i to %n step %stride : i32 {{
{ix}{iy}        %xv = memref.load %x[%ix] : memref<?x{t}, 1>
        %yv = memref.load %y[%iy] : memref<?x{t}, 1>
        %py = arith.mulf %h12, %yv : {t}
        %nx = math.fma %h11, %xv, %py : {t}
        %qy = arith.mulf %h22, %yv : {t}
        %ny = math.fma %h21, %xv, %qy : {t}
        memref.store %nx, %x[%ix] : memref<?x{t}, 1>
        memref.store %ny, %y[%iy] : memref<?x{t}, 1>
      }}
      gpu.return
    }}
""".format(p=p, t=t, gid=gid(), ix=idx("ix", "xo", "k", "xi"), iy=idx("iy", "yo", "k", "yi"))


def ger(p, t):
    """A += alpha x y^T (A(i, j) = a[ao + i + j lda]); mask 1 updates only i <= j, 2 only i >= j (syr, syr2)"""
    return """    gpu.func @{p}ger(%m: i32, %n: i32, %mask: i32, %alpha: {t}, %x: memref<?x{t}, 1>, %xo: i32, %xi: i32, %y: memref<?x{t}, 1>, %yo: i32, %yi: i32, %a: memref<?x{t}, 1>, %ao: i32, %lda: i32) kernel {{
{g2d}      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %mu = arith.cmpi eq, %mask, %c1 : i32
      %ml = arith.cmpi eq, %mask, %c2 : i32
      %inm = arith.cmpi slt, %i, %m : i32
      scf.if %inm {{
{ix}        %xv = memref.load %x[%ix] : memref<?x{t}, 1>
        scf.for %j = %j0 to %n step %js : i32 {{
          %le = arith.cmpi sle, %i, %j : i32
          %ge = arith.cmpi sge, %i, %j : i32
          %nle = arith.xori %le, %true : i1
          %nge = arith.xori %ge, %true : i1
          %s1 = arith.andi %mu, %nle : i1
          %s2 = arith.andi %ml, %nge : i1
          %skip = arith.ori %s1, %s2 : i1
          %go = arith.xori %skip, %true : i1
          scf.if %go {{
{iy}            %yv = memref.load %y[%iy] : memref<?x{t}, 1>
            %ay = arith.mulf %alpha, %yv : {t}
            %jl = arith.muli %j, %lda : i32
            %a0 = arith.addi %ao, %jl : i32
            %a1 = arith.addi %a0, %i : i32
            %ia = arith.index_cast %a1 : i32 to index
            %av = memref.load %a[%ia] : memref<?x{t}, 1>
            %r = math.fma %xv, %ay, %av : {t}
            memref.store %r, %a[%ia] : memref<?x{t}, 1>
          }}
        }}
      }}
      gpu.return
    }}
""".format(p=p, t=t, g2d=g2d() + "      %true = arith.constant true\n",
           ix=idx("ix", "xo", "i", "xi").replace("      %", "        %"),
           iy=idx("iy", "yo", "j", "yi").replace("      %", "            %"))


def geam(p, t):
    """C = alpha A~ + beta op(B) elementwise (mode bit 6 clear) or C = alpha A~ * op(B) (bit 6, dgmm: B a vector
    with a zero stride), m x n; C(i, j) = c[co + i crs + j ccs]; with mode bit 7 only i <= j is written, bit 8 only
    i >= j. A~(i, j) is A in the storage of mode & 3: 0 general (a[ao + i ars + j acs]); otherwise (lda = acs) the
    structured matrix of bits 2 symmetric, 3 triangular, 4 upper (the stored triangle), 5 unit diagonal, as a
    dense matrix: 1 full storage, 2 banded (kl sub- / ku superdiagonals, LAPACK band storage; kl for symmetric /
    triangular), 3 packed. A is not read where alpha is 0 (except products), B where beta is 0."""
    L = lambda s: s.format(t=t)
    return """    gpu.func @{p}geam(%m: i32, %n: i32, %mode: i32, %alpha: {t}, %a: memref<?x{t}, 1>, %ao: i32, %ars: i32, %acs: i32, %kl: i32, %ku: i32, %beta: {t}, %b: memref<?x{t}, 1>, %bo: i32, %brs: i32, %bcs: i32, %c: memref<?x{t}, 1>, %co: i32, %crs: i32, %ccs: i32) kernel {{
{g2d}      %true = arith.constant true
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %c3 = arith.constant 3 : i32
      %zero = arith.constant 0.0 : {t}
      %one = arith.constant 1.0 : {t}
      %bit = arith.constant 1 : i32
      %kind = arith.andi %mode, %c3 : i32
      %m4 = arith.constant 4 : i32
      %m8 = arith.constant 8 : i32
      %m16 = arith.constant 16 : i32
      %m32 = arith.constant 32 : i32
      %m64 = arith.constant 64 : i32
      %m128 = arith.constant 128 : i32
      %m256 = arith.constant 256 : i32
      %b4 = arith.andi %mode, %m4 : i32
      %sym = arith.cmpi ne, %b4, %c0 : i32
      %b8 = arith.andi %mode, %m8 : i32
      %tri = arith.cmpi ne, %b8, %c0 : i32
      %b16 = arith.andi %mode, %m16 : i32
      %up = arith.cmpi ne, %b16, %c0 : i32
      %b32 = arith.andi %mode, %m32 : i32
      %unit = arith.cmpi ne, %b32, %c0 : i32
      %b64 = arith.andi %mode, %m64 : i32
      %prod = arith.cmpi ne, %b64, %c0 : i32
      %b128 = arith.andi %mode, %m128 : i32
      %wu = arith.cmpi ne, %b128, %c0 : i32
      %b256 = arith.andi %mode, %m256 : i32
      %wl = arith.cmpi ne, %b256, %c0 : i32
      %isgen = arith.cmpi eq, %kind, %c0 : i32
      %isband = arith.cmpi eq, %kind, %c2 : i32
      %ispk = arith.cmpi eq, %kind, %c3 : i32
      %structd = arith.ori %sym, %tri : i1
      %az = arith.cmpf oeq, %alpha, %zero : {t}
      %naz = arith.xori %az, %true : i1
      %reada = arith.ori %naz, %prod : i1
      %bz = arith.cmpf oeq, %beta, %zero : {t}
      %nbz = arith.xori %bz, %true : i1
      %readb = arith.ori %nbz, %prod : i1
      %n2 = arith.addi %n, %n : i32
      %inm = arith.cmpi slt, %i, %m : i32
      scf.if %inm {{
        scf.for %j = %j0 to %n step %js : i32 {{
          %le = arith.cmpi sle, %i, %j : i32
          %ge = arith.cmpi sge, %i, %j : i32
          %nle = arith.xori %le, %true : i1
          %nge = arith.xori %ge, %true : i1
          %s1 = arith.andi %wu, %nle : i1
          %s2 = arith.andi %wl, %nge : i1
          %skip = arith.ori %s1, %s2 : i1
          %go = arith.xori %skip, %true : i1
          scf.if %go {{
            %av = scf.if %reada -> ({t}) {{
              %v = scf.if %isgen -> ({t}) {{
                %g0 = arith.muli %i, %ars : i32
                %g1 = arith.muli %j, %acs : i32
                %g2 = arith.addi %g0, %g1 : i32
                %g3 = arith.addi %ao, %g2 : i32
                %gi = arith.index_cast %g3 : i32 to index
                %gv = memref.load %a[%gi] : memref<?x{t}, 1>
                scf.yield %gv : {t}
              }} else {{
                %intri = arith.select %up, %le, %ge : i1
                %notin = arith.xori %intri, %true : i1
                %swap = arith.andi %sym, %notin : i1
                %si = arith.select %swap, %j, %i : i32
                %sj = arith.select %swap, %i, %j : i32
                %zo = arith.andi %tri, %notin : i1
                %dg = arith.cmpi eq, %i, %j : i32
                %u1 = arith.andi %tri, %unit : i1
                %isone = arith.andi %u1, %dg : i1
                // banded: the distance from the diagonal and the row in band storage
                %dup = arith.subi %sj, %si : i32
                %dlo = arith.subi %si, %sj : i32
                %d = arith.select %up, %dup, %dlo : i32
                %klsi = arith.addi %kl, %si : i32
                %rup = arith.subi %klsi, %sj : i32
                %rs = arith.select %up, %rup, %dlo : i32
                %imj = arith.subi %i, %j : i32
                %jmi = arith.subi %j, %i : i32
                %gb1 = arith.cmpi sle, %imj, %kl : i32
                %gb2 = arith.cmpi sle, %jmi, %ku : i32
                %gbv = arith.andi %gb1, %gb2 : i1
                %gbr = arith.addi %ku, %imj : i32
                %sbv = arith.cmpi sle, %d, %kl : i32
                %bv = arith.select %structd, %sbv, %gbv : i1
                %br = arith.select %structd, %rs, %gbr : i32
                %bc = arith.select %structd, %sj, %j : i32
                %bcl = arith.muli %bc, %acs : i32
                %bidx = arith.addi %br, %bcl : i32
                // packed
                %sj1 = arith.addi %sj, %c1 : i32
                %pu0 = arith.muli %sj, %sj1 : i32
                %pu1 = arith.divsi %pu0, %c2 : i32
                %pu = arith.addi %si, %pu1 : i32
                %nsj = arith.subi %n2, %sj : i32
                %nsj1 = arith.subi %nsj, %c1 : i32
                %pl0 = arith.muli %sj, %nsj1 : i32
                %pl1 = arith.divsi %pl0, %c2 : i32
                %pl = arith.addi %si, %pl1 : i32
                %pidx = arith.select %up, %pu, %pl : i32
                %fl = arith.muli %sj, %acs : i32
                %fidx = arith.addi %si, %fl : i32
                %x1 = arith.select %ispk, %pidx, %fidx : i32
                %x2 = arith.select %isband, %bidx, %x1 : i32
                %x3 = arith.addi %ao, %x2 : i32
                %nzo = arith.xori %zo, %true : i1
                %nb = arith.xori %isband, %true : i1
                %bok = arith.ori %nb, %bv : i1
                %valid = arith.andi %nzo, %bok : i1
                %sv = scf.if %valid -> ({t}) {{
                  %xi_ = arith.index_cast %x3 : i32 to index
                  %lv = memref.load %a[%xi_] : memref<?x{t}, 1>
                  scf.yield %lv : {t}
                }} else {{
                  scf.yield %zero : {t}
                }}
                %sv1 = arith.select %isone, %one, %sv : {t}
                scf.yield %sv1 : {t}
              }}
              scf.yield %v : {t}
            }} else {{
              scf.yield %zero : {t}
            }}
            %bval = scf.if %readb -> ({t}) {{
              %h0 = arith.muli %i, %brs : i32
              %h1 = arith.muli %j, %bcs : i32
              %h2 = arith.addi %h0, %h1 : i32
              %h3 = arith.addi %bo, %h2 : i32
              %hi = arith.index_cast %h3 : i32 to index
              %hv = memref.load %b[%hi] : memref<?x{t}, 1>
              scf.yield %hv : {t}
            }} else {{
              scf.yield %zero : {t}
            }}
            %pa = arith.mulf %alpha, %av : {t}
            %pr = arith.mulf %pa, %bval : {t}
            %sb = arith.mulf %beta, %bval : {t}
            %sm = math.fma %alpha, %av, %sb : {t}
            %sm1 = arith.select %bz, %pa, %sm : {t}
            %res = arith.select %prod, %pr, %sm1 : {t}
            %k0 = arith.muli %i, %crs : i32
            %k1 = arith.muli %j, %ccs : i32
            %k2 = arith.addi %k0, %k1 : i32
            %k3 = arith.addi %co, %k2 : i32
            %ki = arith.index_cast %k3 : i32 to index
            memref.store %res, %c[%ki] : memref<?x{t}, 1>
          }}
        }}
      }}
      gpu.return
    }}
""".format(p=p, t=t, g2d=g2d())


def trsv(p, t):
    """solves M x = alpha b in place (x holds b), M(i, j) = a[ao + i ars + j acs] lower (lower != 0) or upper
    triangular, unit diagonal (unit != 0) or not; one SIMD group per vector (block x: vector x + vb * block, the
    columns or rows of trsm's B). Lane l owns the elements i = l mod 32: each step the owner of the next unknown
    solves it and broadcasts it (a shuffle), every lane updates its remaining elements: no memory is shared."""
    return """    gpu.func @{p}trsv(%n: i32, %a: memref<?x{t}, 1>, %ao: i32, %ars: i32, %acs: i32, %lower: i32, %unit: i32, %alpha: {t}, %x: memref<?x{t}, 1>, %xo: i32, %xi: i32, %xb: i32) kernel {{
      %bid = gpu.block_id x
      %tid = gpu.thread_id x
      %bi = arith.index_cast %bid : index to i32
      %lane = arith.index_cast %tid : index to i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c31 = arith.constant 31 : i32
      %c32 = arith.constant 32 : i32
      %zero = arith.constant 0.0 : {t}
      %vb = arith.muli %bi, %xb : i32
      %base = arith.addi %xo, %vb : i32
      %lw = arith.cmpi ne, %lower, %c0 : i32
      %un = arith.cmpi ne, %unit, %c0 : i32
      scf.for %r = %lane to %n step %c32 : i32 {{
        %q0 = arith.muli %r, %xi : i32
        %q1 = arith.addi %base, %q0 : i32
        %qi = arith.index_cast %q1 : i32 to index
        %qv = memref.load %x[%qi] : memref<?x{t}, 1>
        %qa = arith.mulf %alpha, %qv : {t}
        memref.store %qa, %x[%qi] : memref<?x{t}, 1>
      }}
      %nm1 = arith.subi %n, %c1 : i32
      scf.for %s = %c0 to %n step %c1 : i32 {{
        %back = arith.subi %nm1, %s : i32
        %j = arith.select %lw, %s, %back : i32
        %owner = arith.andi %j, %c31 : i32
        %own = arith.cmpi eq, %lane, %owner : i32
        %jr = arith.muli %j, %ars : i32
        %jc = arith.muli %j, %acs : i32
        %v = scf.if %own -> ({t}) {{
          %x0 = arith.muli %j, %xi : i32
          %x1 = arith.addi %base, %x0 : i32
          %xx = arith.index_cast %x1 : i32 to index
          %xv = memref.load %x[%xx] : memref<?x{t}, 1>
          %r = scf.if %un -> ({t}) {{
            scf.yield %xv : {t}
          }} else {{
            %d0 = arith.addi %jr, %jc : i32
            %d1 = arith.addi %ao, %d0 : i32
            %dx = arith.index_cast %d1 : i32 to index
            %dv = memref.load %a[%dx] : memref<?x{t}, 1>
            %q = arith.divf %xv, %dv : {t}
            scf.yield %q : {t}
          }}
          memref.store %r, %x[%xx] : memref<?x{t}, 1>
          scf.yield %r : {t}
        }} else {{
          scf.yield %zero : {t}
        }}
        %xj, %ok = gpu.shuffle idx %v, %owner, %c32 : {t}
        %j1 = arith.addi %j, %c1 : i32
        %st = arith.select %lw, %j1, %c0 : i32
        %en = arith.select %lw, %n, %j : i32
        %ls = arith.subi %lane, %st : i32
        %lm = arith.andi %ls, %c31 : i32
        %i0 = arith.addi %st, %lm : i32
        %acol = arith.addi %ao, %jc : i32
        scf.for %i = %i0 to %en step %c32 : i32 {{
          %y0 = arith.muli %i, %xi : i32
          %y1 = arith.addi %base, %y0 : i32
          %yx = arith.index_cast %y1 : i32 to index
          %yv = memref.load %x[%yx] : memref<?x{t}, 1>
          %e0 = arith.muli %i, %ars : i32
          %e1 = arith.addi %acol, %e0 : i32
          %ex = arith.index_cast %e1 : i32 to index
          %ev = memref.load %a[%ex] : memref<?x{t}, 1>
          %ne = arith.negf %ev : {t}
          %u = math.fma %ne, %xj, %yv : {t}
          memref.store %u, %x[%yx] : memref<?x{t}, 1>
        }}
      }}
      gpu.return
    }}
""".format(p=p, t=t)


def gemv_ybeta(t, ind):
    """y[%iy] = alpha %sum + beta y[%iy] (y not read where beta is 0)"""
    s = """%bz = arith.cmpf oeq, %beta, %zero : {t}
%yv = scf.if %bz -> ({t}) {{
  scf.yield %zero : {t}
}} else {{
  %y0 = memref.load %y[%iy] : memref<?x{t}, 1>
  %by = arith.mulf %beta, %y0 : {t}
  scf.yield %by : {t}
}}
%res = math.fma %alpha, %sum, %yv : {t}
memref.store %res, %y[%iy] : memref<?x{t}, 1>
""".format(t=t)
    return "".join(ind + l + "\n" for l in s.splitlines())


GEMV_ARGS = ("%m: i32, %n: i32, %alpha: {t}, %a: memref<?x{t}, 1>, %ao: i32, %ars: i32, %acs: i32, %sa: i32, "
             "%x: memref<?x{t}, 1>, %xo: i32, %xi: i32, %sx: i32, %beta: {t}, %y: memref<?x{t}, 1>, %yo: i32, "
             "%yi: i32, %sy: i32")


def gemv_head(t):
    return """      %bz_ = gpu.block_id y
      %bzi = arith.index_cast %bz_ : index to i32
      %zsa = arith.muli %bzi, %sa : i32
      %ab = arith.addi %ao, %zsa : i32
      %zsx = arith.muli %bzi, %sx : i32
      %xb = arith.addi %xo, %zsx : i32
      %zsy = arith.muli %bzi, %sy : i32
      %yb = arith.addi %yo, %zsy : i32
      %zero = arith.constant 0.0 : {t}
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %mm1 = arith.subi %m, %c1 : i32
""".format(t=t)


def gemv_n(p, t):
    """y = alpha op(A) x + beta y, op(A) m x n (A(i, j) = a[ao + i ars + j acs]), for row-contiguous columns
    (ars 1): blocks of 32 x 8 threads, 32 rows (a SIMD group's lanes, coalesced) by 8 SIMD groups that split the
    columns; their partial sums are added in threadgroup memory (in a fixed order). Batched over grid y."""
    return """    gpu.func @{p}gemv_n({args}) workgroup(%part: memref<256x{t}, 3>) kernel {{
{head}      %tx = gpu.thread_id x
      %ty = gpu.thread_id y
      %bx = gpu.block_id x
      %gx = gpu.grid_dim x
      %txi = arith.index_cast %tx : index to i32
      %tyi = arith.index_cast %ty : index to i32
      %bxi = arith.index_cast %bx : index to i32
      %gxi = arith.index_cast %gx : index to i32
      %c8 = arith.constant 8 : i32
      %c32 = arith.constant 32 : i32
      %r0b = arith.muli %bxi, %c32 : i32
      %r0 = arith.addi %r0b, %txi : i32
      %rstep = arith.muli %gxi, %c32 : i32
      %ty32 = arith.muli %tyi, %c32 : i32
      %slot0 = arith.addi %ty32, %txi : i32
      %slot = arith.index_cast %slot0 : i32 to index
      %is0 = arith.cmpi eq, %tyi, %c0 : i32
      // the rows of all blocks: every thread of a block runs the same number of iterations (barriers)
      scf.for %rr = %r0b to %m step %rstep : i32 {{
        %r = arith.addi %rr, %txi : i32
        %rc = arith.minsi %r, %mm1 : i32
        %rofs = arith.muli %rc, %ars : i32
        %abase = arith.addi %ab, %rofs : i32
        %acc = scf.for %j = %tyi to %n step %c8 iter_args(%s = %zero) -> ({t}) : i32 {{
          %ja = arith.muli %j, %acs : i32
          %ia_ = arith.addi %abase, %ja : i32
          %ia = arith.index_cast %ia_ : i32 to index
          %jx = arith.muli %j, %xi : i32
          %ix_ = arith.addi %xb, %jx : i32
          %ix = arith.index_cast %ix_ : i32 to index
          %av = memref.load %a[%ia] : memref<?x{t}, 1>
          %xv = memref.load %x[%ix] : memref<?x{t}, 1>
          %ns = math.fma %av, %xv, %s : {t}
          scf.yield %ns : {t}
        }}
        memref.store %acc, %part[%slot] : memref<256x{t}, 3>
        gpu.barrier
        %inm = arith.cmpi slt, %r, %m : i32
        %fin = arith.andi %is0, %inm : i1
        scf.if %fin {{
          %sum = scf.for %q = %c0 to %c8 step %c1 iter_args(%s2 = %zero) -> ({t}) : i32 {{
            %q32 = arith.muli %q, %c32 : i32
            %qs_ = arith.addi %q32, %txi : i32
            %qs = arith.index_cast %qs_ : i32 to index
            %pv = memref.load %part[%qs] : memref<256x{t}, 3>
            %ns2 = arith.addf %s2, %pv : {t}
            scf.yield %ns2 : {t}
          }}
          %ry = arith.muli %r, %yi : i32
          %iy_ = arith.addi %yb, %ry : i32
          %iy = arith.index_cast %iy_ : i32 to index
{ybeta}        }}
        gpu.barrier
      }}
      gpu.return
    }}
""".format(p=p, t=t, args=GEMV_ARGS.format(t=t), head=gemv_head(t), ybeta=gemv_ybeta(t, "          "))


def gemv_t(p, t):
    """the same for column-contiguous rows (acs 1) or few rows: a SIMD group per row (lanes split the row,
    coalesced), 8 rows per block of 256 threads; the lanes' sums are reduced across the SIMD group."""
    return """    gpu.func @{p}gemv_t({args}) kernel {{
{head}      %tid = gpu.thread_id x
      %bx = gpu.block_id x
      %gx = gpu.grid_dim x
      %sg = gpu.subgroup_id : index
      %lanex = gpu.lane_id
      %lane = arith.index_cast %lanex : index to i32
      %sgi = arith.index_cast %sg : index to i32
      %bxi = arith.index_cast %bx : index to i32
      %gxi = arith.index_cast %gx : index to i32
      %c8 = arith.constant 8 : i32
      %c32 = arith.constant 32 : i32
      %r0b = arith.muli %bxi, %c8 : i32
      %r0 = arith.addi %r0b, %sgi : i32
      %rstep = arith.muli %gxi, %c8 : i32
      %is0 = arith.cmpi eq, %lane, %c0 : i32
      scf.for %r = %r0 to %m step %rstep : i32 {{
        %rofs = arith.muli %r, %ars : i32
        %abase = arith.addi %ab, %rofs : i32
        %acc = scf.for %j = %lane to %n step %c32 iter_args(%s = %zero) -> ({t}) : i32 {{
          %ja = arith.muli %j, %acs : i32
          %ia_ = arith.addi %abase, %ja : i32
          %ia = arith.index_cast %ia_ : i32 to index
          %jx = arith.muli %j, %xi : i32
          %ix_ = arith.addi %xb, %jx : i32
          %ix = arith.index_cast %ix_ : i32 to index
          %av = memref.load %a[%ia] : memref<?x{t}, 1>
          %xv = memref.load %x[%ix] : memref<?x{t}, 1>
          %ns = math.fma %av, %xv, %s : {t}
          scf.yield %ns : {t}
        }}
        %sum = gpu.subgroup_reduce add %acc : ({t}) -> {t}
        scf.if %is0 {{
          %ry = arith.muli %r, %yi : i32
          %iy_ = arith.addi %yb, %ry : i32
          %iy = arith.index_cast %iy_ : i32 to index
{ybeta}        }}
      }}
      gpu.return
    }}
""".format(p=p, t=t, args=GEMV_ARGS.format(t=t), head=gemv_head(t), ybeta=gemv_ybeta(t, "          "))


# --------------------------------------------------------------------------------------------- complex (c, z)
# A complex vector or matrix is its real type's memref, interleaved: element k is (re, im) at 2 k, 2 k + 1; offsets,
# increments and strides are in complex elements.

def cidx(name, off, i, inc, ind="      "):
    """%name_r / %name_i: the real and imaginary indexes of element off + i * inc"""
    return ("""{ind}%{n}_m = arith.muli %{i}, %{inc} : i32
{ind}%{n}_a = arith.addi %{off}, %{n}_m : i32
{ind}%{n}_2 = arith.addi %{n}_a, %{n}_a : i32
{ind}%{n}_3 = arith.addi %{n}_2, %c1_ : i32
{ind}%{n}_r = arith.index_cast %{n}_2 : i32 to index
{ind}%{n}_i = arith.index_cast %{n}_3 : i32 to index
""").format(n=name, off=off, i=i, inc=inc, ind=ind)


def cld(v, mem, n, t, ind="      "):
    return ("{ind}%{v}r = memref.load %{m}[%{n}_r] : memref<?x{t}, 1>\n"
            "{ind}%{v}i = memref.load %{m}[%{n}_i] : memref<?x{t}, 1>\n").format(v=v, m=mem, n=n, t=t, ind=ind)


def cst(v, mem, n, t, ind="      "):
    return ("{ind}memref.store %{v}r, %{m}[%{n}_r] : memref<?x{t}, 1>\n"
            "{ind}memref.store %{v}i, %{m}[%{n}_i] : memref<?x{t}, 1>\n").format(v=v, m=mem, n=n, t=t, ind=ind)


def cfma(o, ar, ai, br, bi, cr, ci, t, ind="      "):
    """%o r/i = a * b + c (complex)"""
    return ("""{ind}%{o}_p = arith.mulf %{ai}, %{bi} : {t}
{ind}%{o}_q = arith.subf %{cr}, %{o}_p : {t}
{ind}%{o}r = math.fma %{ar}, %{br}, %{o}_q : {t}
{ind}%{o}_s = arith.mulf %{ai}, %{br} : {t}
{ind}%{o}_u = arith.addf %{ci}, %{o}_s : {t}
{ind}%{o}i = math.fma %{ar}, %{bi}, %{o}_u : {t}
""").format(o=o, ar=ar, ai=ai, br=br, bi=bi, cr=cr, ci=ci, t=t, ind=ind)


def caxpy(p, t):
    """y = alpha x + y (mode 0), x = alpha x (mode 1: scal, y unused)"""
    return """    gpu.func @{p}axpy(%n: i32, %mode: i32, %ar: {t}, %ai: {t}, %x: memref<?x{t}, 1>, %xo: i32, %xi: i32, %y: memref<?x{t}, 1>, %yo: i32, %yi: i32) kernel {{
{gid}      %c1_ = arith.constant 1 : i32
      %c0_ = arith.constant 0 : i32
      %zero = arith.constant 0.0 : {t}
      %sc = arith.cmpi ne, %mode, %c0_ : i32
      scf.for %k = %i to %n step %stride : i32 {{
{ix}{lx}        scf.if %sc {{
{s}{sx}        }} else {{
{iy}{ly}{a}{sy}        }}
      }}
      gpu.return
    }}
""".format(p=p, t=t, gid=gid(), ix=cidx("ix", "xo", "k", "xi", "        "), lx=cld("X", "x", "ix", t, "        "),
           iy=cidx("iy", "yo", "k", "yi", "          "), ly=cld("Y", "y", "iy", t, "          "),
           a=cfma("R", "ar", "ai", "Xr", "Xi", "Yr", "Yi", t, "          "), sy=cst("R", "y", "iy", t, "          "),
           s=cfma("s", "ar", "ai", "Xr", "Xi", "zero", "zero", t, "          "), sx=cst("s", "x", "ix", t, "          "))


def creduce(p, t):
    """partial complex sums per block of one SIMD group: op 0 x . y, 1 conj(x) . y, 3 the sum of x (final pass)"""
    return """    gpu.func @{p}reduce(%n: i32, %op: i32, %x: memref<?x{t}, 1>, %xo: i32, %xi: i32, %y: memref<?x{t}, 1>, %yo: i32, %yi: i32, %out: memref<?x{t}, 1>) kernel {{
{gid}      %c1_ = arith.constant 1 : i32
      %c3 = arith.constant 3 : i32
      %zero = arith.constant 0.0 : {t}
      %one = arith.constant 1.0 : {t}
      %fin = arith.cmpi eq, %op, %c3 : i32
      %cj = arith.cmpi eq, %op, %c1_ : i32
      %r:2 = scf.for %k = %i to %n step %stride iter_args(%sr = %zero, %si = %zero) -> ({t}, {t}) : i32 {{
{ix}{lx}        %xin = arith.negf %Xi : {t}
        %xc = arith.select %cj, %xin, %Xi : {t}
        %v:2 = scf.if %fin -> ({t}, {t}) {{
          %a1 = arith.addf %sr, %Xr : {t}
          %a2 = arith.addf %si, %Xi : {t}
          scf.yield %a1, %a2 : {t}, {t}
        }} else {{
{iy}{ly}{f}          scf.yield %qr, %qi : {t}, {t}
        }}
        scf.yield %v#0, %v#1 : {t}, {t}
      }}
      %s0 = gpu.subgroup_reduce add %r#0 : ({t}) -> {t}
      %s1 = gpu.subgroup_reduce add %r#1 : ({t}) -> {t}
      %lane = gpu.lane_id
      %c0i = arith.constant 0 : index
      %first = arith.cmpi eq, %lane, %c0i : index
      scf.if %first {{
        %b2 = arith.addi %bid, %bid : index
        %b3 = arith.addi %b2, %c1i : index
        memref.store %s0, %out[%b2] : memref<?x{t}, 1>
        memref.store %s1, %out[%b3] : memref<?x{t}, 1>
      }}
      gpu.return
    }}
""".format(p=p, t=t, gid=gid() + "      %c1i = arith.constant 1 : index\n", ix=cidx("ix", "xo", "k", "xi", "        "),
           lx=cld("X", "x", "ix", t, "        "), iy=cidx("iy", "yo", "k", "yi", "          "),
           ly=cld("Y", "y", "iy", t, "          "), f=cfma("q", "Xr", "xc", "Yr", "Yi", "sr", "si", t, "          "))


def cgemv(p, t):
    """y = alpha op(A) x + beta y, op(A)(i, j) = A[ao + i ars + j acs], conjugated if conj; a thread per row"""
    return """    gpu.func @{p}gemv(%m: i32, %n: i32, %conj: i32, %ar: {t}, %ai: {t}, %a: memref<?x{t}, 1>, %ao: i32, %ars: i32, %acs: i32, %x: memref<?x{t}, 1>, %xo: i32, %xi: i32, %br: {t}, %bi: {t}, %y: memref<?x{t}, 1>, %yo: i32, %yi: i32) kernel {{
{gid}      %c1_ = arith.constant 1 : i32
      %c0 = arith.constant 0 : i32
      %zero = arith.constant 0.0 : {t}
      %cj = arith.cmpi ne, %conj, %c0 : i32
      %bzr = arith.cmpf oeq, %br, %zero : {t}
      %bzi = arith.cmpf oeq, %bi, %zero : {t}
      %bz = arith.andi %bzr, %bzi : i1
      scf.for %r = %i to %m step %stride : i32 {{
        %rr = arith.muli %r, %ars : i32
        %rowbase = arith.addi %ao, %rr : i32
        %acc:2 = scf.for %j = %c0 to %n step %c1_ iter_args(%sr = %zero, %si = %zero) -> ({t}, {t}) : i32 {{
{ia}{la}          %ain = arith.negf %Ai : {t}
          %ac = arith.select %cj, %ain, %Ai : {t}
{ix}{lx}{f}          scf.yield %qr, %qi : {t}, {t}
        }}
{iy}        %old:2 = scf.if %bz -> ({t}, {t}) {{
          scf.yield %zero, %zero : {t}, {t}
        }} else {{
{ly}{g}          scf.yield %or, %oi : {t}, {t}
        }}
{h}{sy}      }}
      gpu.return
    }}
""".format(p=p, t=t, gid=gid(), ia=cidx("ia", "rowbase", "j", "acs", "          "),
           la=cld("A", "a", "ia", t, "          "),
           ix=cidx("ix", "xo", "j", "xi", "          "), lx=cld("X", "x", "ix", t, "          "),
           f=cfma("q", "Ar", "ac", "Xr", "Xi", "sr", "si", t, "          "),
           iy=cidx("iy", "yo", "r", "yi", "        "), ly=cld("Y", "y", "iy", t, "          "),
           g=cfma("o", "br", "bi", "Yr", "Yi", "zero", "zero", t, "          "),
           h=cfma("w", "ar", "ai", "acc#0", "acc#1", "old#0", "old#1", t, "        "), sy=cst("w", "y", "iy", t, "        "))


def cgemm(p, t, tile):
    """C = alpha op(A) op(B) + beta C (complex), batched over grid z; op(A)(i, l) = A[ao + i ars + l acs] conjugated
    if conja (likewise B); C(i, j) = C[co + i + j ldc]. Blocks of 16 x 16 threads, each a tile x tile block of C
    (rows i0 + r 16, columns j0 + q 16)."""
    w = []
    w.append("""    gpu.func @{p}gemm(%m: i32, %n: i32, %k: i32, %ar: {t}, %ai: {t}, %a: memref<?x{t}, 1>, %ao: i32, %ars: i32, %acs: i32, %sa: i32, %conja: i32, %b: memref<?x{t}, 1>, %bo: i32, %brs: i32, %bcs: i32, %sb: i32, %conjb: i32, %br: {t}, %bi: {t}, %c: memref<?x{t}, 1>, %co: i32, %ldc: i32, %sc: i32) kernel {{
      %tx = gpu.thread_id x
      %ty = gpu.thread_id y
      %bx = gpu.block_id x
      %by = gpu.block_id y
      %bz = gpu.block_id z
      %txi = arith.index_cast %tx : index to i32
      %tyi = arith.index_cast %ty : index to i32
      %bxi = arith.index_cast %bx : index to i32
      %byi = arith.index_cast %by : index to i32
      %bzi = arith.index_cast %bz : index to i32
      %c0 = arith.constant 0 : i32
      %c1_ = arith.constant 1 : i32
      %c16 = arith.constant 16 : i32
      %cT = arith.constant {bt} : i32
      %zero = arith.constant 0.0 : {t}
      %ca = arith.cmpi ne, %conja, %c0 : i32
      %cb = arith.cmpi ne, %conjb, %c0 : i32
      %rb = arith.muli %bxi, %cT : i32
      %i0 = arith.addi %rb, %txi : i32
      %cbk = arith.muli %byi, %cT : i32
      %j0 = arith.addi %cbk, %tyi : i32
      %zsa = arith.muli %bzi, %sa : i32
      %abase = arith.addi %ao, %zsa : i32
      %zsb = arith.muli %bzi, %sb : i32
      %bbase = arith.addi %bo, %zsb : i32
      %zsc = arith.muli %bzi, %sc : i32
      %cbase = arith.addi %co, %zsc : i32
      %mm1 = arith.subi %m, %c1_ : i32
      %nm1 = arith.subi %n, %c1_ : i32""".format(p=p, t=t, bt=16 * tile))
    for r in range(tile):
        w.append("""      %cr{r} = arith.constant {o} : i32
      %row{r} = arith.addi %i0, %cr{r} : i32
      %rowc{r} = arith.minsi %row{r}, %mm1 : i32
      %rofs{r} = arith.muli %rowc{r}, %ars : i32
      %arow{r} = arith.addi %abase, %rofs{r} : i32
      %col{r} = arith.addi %j0, %cr{r} : i32
      %colc{r} = arith.minsi %col{r}, %nm1 : i32
      %cofs{r} = arith.muli %colc{r}, %bcs : i32
      %bcol{r} = arith.addi %bbase, %cofs{r} : i32""".format(r=r, o=16 * r))
    tiles = [(r, q) for r in range(tile) for q in range(tile)]
    nacc = 2 * len(tiles)
    w.append("      %%res:%d = scf.for %%l = %%c0 to %%k step %%c1_ iter_args(%s) -> (%s) : i32 {" % (
        nacc, ", ".join("%%s%d%dr = %%zero, %%s%d%di = %%zero" % (r, q, r, q) for r, q in tiles), ", ".join([t] * nacc)))
    for r in range(tile):
        w.append(cidx("ai%d" % r, "arow%d" % r, "l", "acs", "        ").rstrip("\n"))
        w.append(cld("a%d" % r, "a", "ai%d" % r, t, "        ").rstrip("\n"))
        w.append("        %%a%din = arith.negf %%a%di : %s\n        %%a%dc = arith.select %%ca, %%a%din, %%a%di : %s" % (r, r, t, r, r, r, t))
        w.append(cidx("bi%d" % r, "bcol%d" % r, "l", "brs", "        ").rstrip("\n"))
        w.append(cld("b%d" % r, "b", "bi%d" % r, t, "        ").rstrip("\n"))
        w.append("        %%b%din = arith.negf %%b%di : %s\n        %%b%dc = arith.select %%cb, %%b%din, %%b%di : %s" % (r, r, t, r, r, r, t))
    outs = []
    for r, q in tiles:
        w.append(cfma("n%d%d" % (r, q), "a%dr" % r, "a%dc" % r, "b%dr" % q, "b%dc" % q, "s%d%dr" % (r, q), "s%d%di" % (r, q), t, "        ").rstrip("\n"))
        outs += ["%%n%d%dr" % (r, q), "%%n%d%di" % (r, q)]
    w.append("        scf.yield %s : %s\n      }" % (", ".join(outs), ", ".join([t] * nacc)))
    w.append("""      %bzr = arith.cmpf oeq, %br, %zero : {t}
      %bzi_ = arith.cmpf oeq, %bi, %zero : {t}
      %bzero = arith.andi %bzr, %bzi_ : i1""".format(t=t))
    for idx_, (r, q) in enumerate(tiles):
        tq = "%d%d" % (r, q)
        w.append("""      %inr{tq} = arith.cmpi slt, %row{r}, %m : i32
      %inc{tq} = arith.cmpi slt, %col{q}, %n : i32
      %in{tq} = arith.andi %inr{tq}, %inc{tq} : i1
      scf.if %in{tq} {{
        %cl{tq} = arith.muli %col{q}, %ldc : i32
        %cc{tq} = arith.addi %cbase, %cl{tq} : i32
{ci}        %old{tq}:2 = scf.if %bzero -> ({t}, {t}) {{
          scf.yield %zero, %zero : {t}, {t}
        }} else {{
{lc}{bc}          scf.yield %e{tq}r, %e{tq}i : {t}, {t}
        }}
{fin}{st}      }}""".format(tq=tq, r=r, q=q, t=t, ci=cidx("x" + tq, "cc" + tq, "row%d" % r, "c1_", "        "),
                    lc=cld("o" + tq, "c", "x" + tq, t, "          "),
                    bc=cfma("e" + tq, "br", "bi", "o%sr" % tq, "o%si" % tq, "zero", "zero", t, "          "),
                    fin=cfma("v" + tq, "ar", "ai", "res#%d" % (2 * idx_), "res#%d" % (2 * idx_ + 1), "old%s#0" % tq, "old%s#1" % tq, t, "        "),
                    st=cst("v" + tq, "c", "x" + tq, t, "        ")))
    w.append("      gpu.return\n    }\n")
    return "\n".join(w)


def main():
    out = [hdr()]
    for p, t in TYPES.items():
        for f in (axpy, scal, copy_swap, reduce, amax, rot, ger, geam, trsv, gemv_n, gemv_t):
            out.append(f(p, t))
        out.append(gemm(p + "gemm", t, t, t, t))
    for p, t, tile in (("c", "f32", 2), ("z", "f64", 1)):
        for f in (caxpy, creduce, cgemv):
            out.append(f(p, t))
        out.append(cgemm(p, t, tile))
    # cublasGemmEx / Hgemm: half or bfloat inputs, float accumulation, the output in either
    for a in ("f16", "bf16"):
        for c in (a, "f32"):
            out.append(gemm("gemm_%s_%s" % (a, c), a, a, c, "f32"))
    for a in ("f32", "f16", "bf16"):
        out.append(gemm_mma("gemm_mma_%s" % a, a))
        out.append(gemm_mma("gemm_mma32_%s" % a, a, 4, 4))
    out.append(ftr())
    print("".join(out))


main()
