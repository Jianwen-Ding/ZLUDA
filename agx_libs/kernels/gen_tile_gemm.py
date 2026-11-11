#!/usr/bin/env python3
"""Generates the tile GEMM kernels of sgemm.mlir and hgemm.mlir (written out, as MLIR has no loop over SSA names):
each SIMD-group's 32 x 32 tile is 4 x 4 air 8x8 matrices accumulated along K in registers. The kernels are
row-major D = A B with variants that read A or B transposed (`_ta`, `_tb`, `_tatb`): A[r, k] at a[k * lda + r]
and B[k, c] at b[c * ldb + k]. Run it after changing the kernel shape:

  gen_tile_gemm.py sgemm.mlir hgemm.mlir

Each gpu.func @agx_[sh]gemm_tile64* of the files is replaced in place (the comments before it stay); missing
variants are appended at the end of the module.
"""

import re
import sys

KERNELS = {  # file: [(name, A and B type, C type)]
    "sgemm.mlir": [("agx_sgemm_tile64", "f32", "f32")],
    "hgemm.mlir": [("agx_hgemm_tile64_cf16", "f16", "f16"), ("agx_hgemm_tile64_cf32", "f16", "f32")],
}
VARIANTS = [("", False, False), ("_ta", True, False), ("_tb", False, True), ("_tatb", True, True)]
# `_u` kernels make no alignment assumption (odd leading dimensions or strides, matrices not 8-byte aligned)
ALIGNMENTS = [("", True), ("_u", False)]
# The next step's fragments loaded while the current step's products run (see kernel()), f16 and f32 kernels
PREFETCH = True
# The K span of the outer loop (see kernel())
CHUNK = 1 << 20


def kernel(name, ab, c, ta, tb, aligned=True):
    M = lambda t: f"vector<64x{t}>"
    acc, mab = M("f32"), M(ab)
    mem = lambda t: f"memref<?x?x{t}, strided<[?, 1], offset: ?>, 1>"
    o = []
    w = o.append
    w(f"  gpu.func @{name}(%a0: memref<1x{ab}, 1>, %b0: memref<1x{ab}, 1>,")
    w(f"      %c0: memref<1x{c}, 1>, %M: index, %N: index, %K: index, %lda: index,")
    w(f"      %ldb: index, %ldc: index, %bsa: index, %bsb: index, %bsc: index, %alpha: f32,")
    w(f"      %beta: f32) kernel")
    w("      attributes {known_block_size = array<i32: 128, 1, 1>} {")
    for v in [0, 1, 2, 8, 16, 24, 32, 64, 128]:
        w(f"    %i{v} = arith.constant {v} : index")
    w("    %m16 = arith.constant -16 : index")
    # K is a multiple of 8 (the plan's): said here, every fragment's step index is a multiple of 8, so the two
    # elements of a fragment row that a lane loads are one aligned 32-bit access (the prefetch's clamped index hid it)
    w("    %m8 = arith.constant -8 : index")
    w("    %K8 = arith.andi %K, %m8 : index")
    # The plan launches the aligned kernel with 8-byte aligned matrices and even leading dimensions and batch
    # strides: said here, the two elements of a matrix row a lane loads are one aligned 32-bit access even for
    # 16-bit types. The `_u` kernel takes them as they are.
    e = "_e" if aligned else ""  # the values the kernel uses: masked even, or as passed
    mat = (lambda n: f"%{n}0a") if aligned else (lambda n: f"%{n}0")
    if aligned:
        w("    %even = arith.constant -2 : index")
        for n in ["a", "b", "c"]:
            w(f"    %{n}0a = memref.assume_alignment %{n}0, 8 : memref<1x{ab if n != 'c' else c}, 1>")
        for v in ["lda", "ldb", "ldc", "bsa", "bsb", "bsc"]:
            w(f"    %{v}_e = arith.andi %{v}, %even : index")
    # the matrices of batch entry z (grid y) are bsa, bsb and bsc elements on
    w("    %z = gpu.block_id y")
    for n in ["a", "b", "c"]:
        w(f"    %o{n} = arith.muli %z, %bs{n}{e} : index")
    # a transposed operand is stored as its transpose, row-major
    for n, t, sz, ld in [("A", ab, "%K, %M" if ta else "%M, %K", "%lda" + e),
                         ("B", ab, "%N, %K" if tb else "%K, %N", "%ldb" + e), ("C", c, "%M, %N", "%ldc" + e)]:
        w(f"    %{n} = memref.reinterpret_cast {mat(n.lower())} to offset: [%o{n.lower()}], sizes: [{sz}], strides: [{ld}, 1]")
        w(f"        : memref<1x{t}, 1> to {mem(t)}")
    w("""
    // Grouped tile order: groups of 8 tile rows, column-major inside a group.
    %tiles_m = arith.divui %M, %i64 : index
    %tiles_n = arith.divui %N, %i64 : index
    %id = gpu.block_id x
    %per_group = arith.muli %tiles_n, %i8 : index
    %group = arith.divui %id, %per_group : index
    %first_m = arith.muli %group, %i8 : index
    %left_m = arith.subi %tiles_m, %first_m : index
    %group_m = arith.minui %left_m, %i8 : index
    %in_group = arith.remui %id, %per_group : index
    %rem_m = arith.remui %in_group, %group_m : index
    %tm = arith.addi %first_m, %rem_m : index
    %tn = arith.divui %in_group, %group_m : index

    // This SIMD-group's 32 x 32 tile: 4 x 4 matrices.
    %sg = gpu.subgroup_id : index
    %sr = arith.divui %sg, %i2 : index
    %sc = arith.remui %sg, %i2 : index
    %tm64 = arith.muli %tm, %i64 : index
    %tn64 = arith.muli %tn, %i64 : index
    %sr32 = arith.muli %sr, %i32 : index
    %sc32 = arith.muli %sc, %i32 : index
    %row = arith.addi %tm64, %sr32 : index
    %col = arith.addi %tn64, %sc32 : index""")
    for i in range(1, 4):
        w(f"    %row{i} = arith.addi %row, %i{8 * i} : index")
        w(f"    %col{i} = arith.addi %col, %i{8 * i} : index")
    rows = ["%row"] + [f"%row{i}" for i in range(1, 4)]
    cols = ["%col"] + [f"%col{i}" for i in range(1, 4)]
    w("    %fz = arith.constant 0.0 : f32")
    w(f"    %zero = air.simdgroup_matrix_fill %fz : {acc}")
    ts = ", ".join([acc] * 16)
    # K in chunks of CHUNK (one trip for any K the libraries see): the SIMD-groups run independently, without a
    # barrier per chunk. Measured on G14C (gemmbench, f16 and f32, llama shapes and 4096^3): chunks of 128 with a
    # barrier (the earlier kernel) -> no barrier -4..-7%, chunks of 896 without barrier -7..-9%.
    w(f"    %chunk = arith.constant {CHUNK} : index")
    w("    %acc:16 = scf.for %kk = %i0 to %K8 step %chunk")
    w("        iter_args(" + ", ".join(f"%o{i}{j} = %zero" for i in range(4) for j in range(4)) + f") -> ({ts}) {{")
    w("      %kk_end0 = arith.addi %kk, %chunk : index")
    w("      %kk_end = arith.minui %kk_end0, %K8 : index")
    prefetch = PREFETCH
    if prefetch:
        # Software pipelining: the fragments of step k + 8 are loaded while step k's products run; the last step
        # loads its own again (clamped) instead of branching. The loop runs two steps per trip with the two
        # fragment sets taking turns (double buffering): with one set carried, the prefetched fragments were
        # copied into the carried registers at the end of each trip, and the copies waited for the loads.
        w("      %kk_last = arith.subi %kk_end, %i8 : index")
        def loads(k, pre, out):
            for i in range(4):
                idx = f"{k}, {rows[i]}] transpose" if ta else f"{rows[i]}, {k}]"
                w(f"      {pre}%{out}a{i} = air.simdgroup_matrix_load %A[{idx} : {mem(ab)} -> {mab}")
            for j in range(4):
                idx = f"{cols[j]}, {k}] transpose" if tb else f"{k}, {cols[j]}]"
                w(f"      {pre}%{out}b{j} = air.simdgroup_matrix_load %B[{idx} : {mem(ab)} -> {mab}")
        tys_p = f"{mab}, {acc}"
        def products(fa, fb, accs, out, pre):
            for i in range(4):
                for j in range(4):
                    w(f"      {pre}%{out}{i}{j} = air.simdgroup_matrix_multiply_accumulate {fa(i)}, {fb(j)}, "
                      f"{accs(i, j)} : {tys_p}")
        loads("%kk", "", "p")
        frag = ", ".join([mab] * 8)
        w("      %span = arith.subi %kk_end, %kk : index")
        w("      %span16 = arith.andi %span, %m16 : index")
        w("      %kk_end16 = arith.addi %kk, %span16 : index")
        # The loop counts steps of 8 and each fragment index is (step clamped) * 8, so that the alignment of the
        # fragment addresses is plain to LLVM (through the loop phis, umin and geps it is past the known-bits depth,
        # and the vectorizer then keeps the two 16-bit elements of a lane as two loads)
        w("      %s0 = arith.divui %kk, %i8 : index")
        w("      %s_end = arith.divui %kk_end16, %i8 : index")
        w("      %s_last = arith.divui %kk_last, %i8 : index")
        w("      %in:24 = scf.for %s = %s0 to %s_end step %i2")
        w("          iter_args(" + ", ".join(f"%a{i}{j} = %o{i}{j}" for i in range(4) for j in range(4))
          + ", " + ", ".join(f"%ca{i} = %pa{i}" for i in range(4)) + ", " + ", ".join(f"%cb{j} = %pb{j}" for j in range(4))
          + f") -> ({ts}, {frag}) {{")
        w("        %sn0 = arith.addi %s, %i1 : index")
        w("        %sn = arith.minui %sn0, %s_last : index")
        w("        %kn = arith.muli %sn, %i8 : index")
        loads("%kn", "  ", "n")
        products(lambda i: f"%ca{i}", lambda j: f"%cb{j}", lambda i, j: f"%a{i}{j}", "p", "  ")
        w("        %sm0 = arith.addi %s, %i2 : index")
        w("        %sm = arith.minui %sm0, %s_last : index")
        w("        %km = arith.muli %sm, %i8 : index")
        loads("%km", "  ", "q")
        products(lambda i: f"%na{i}", lambda j: f"%nb{j}", lambda i, j: f"%p{i}{j}", "n", "  ")
        w("        scf.yield " + ", ".join(f"%n{i}{j}" for i in range(4) for j in range(4)) + ", "
          + ", ".join(f"%qa{i}" for i in range(4)) + ", " + ", ".join(f"%qb{j}" for j in range(4)) + f" : {ts}, {frag}")
        w("      }")
        # an odd last step: the carried fragments are its own
        w("      %odd = arith.cmpi ne, %span16, %span : index")
        w(f"      %fin:16 = scf.if %odd -> ({ts}) {{")
        products(lambda i: f"%in#{16 + i}", lambda j: f"%in#{20 + j}", lambda i, j: f"%in#{4 * i + j}", "r", "  ")
        w("        scf.yield " + ", ".join(f"%r{i}{j}" for i in range(4) for j in range(4)) + f" : {ts}")
        w("      } else {")
        w("        scf.yield " + ", ".join(f"%in#{k}" for k in range(16)) + f" : {ts}")
        w("      }")
        w("      scf.yield " + ", ".join(f"%fin#{k}" for k in range(16)) + f" : {ts}")
        w("    }")
    else:
      w("      %in:16 = scf.for %k = %kk to %kk_end step %i8")
      w("          iter_args(" + ", ".join(f"%a{i}{j} = %o{i}{j}" for i in range(4) for j in range(4))
        + f") -> ({ts}) {{")
    if not prefetch:
        # With B transposed (and A not) in 16-bit, each A fragment is instead loaded just before its four products
        # (agx_cost: hgemm _tb -9..-13% on G14G; G13G moves both ways, sgemm _tb loses).
        a_late = tb and not ta and ab == "f16"
        if not a_late:
            for i in range(4):
                idx = f"%k, {rows[i]}] transpose" if ta else f"{rows[i]}, %k]"
                w(f"        %ma{i} = air.simdgroup_matrix_load %A[{idx} : {mem(ab)} -> {mab}")
        tys = acc if ab == "f32" else f"{mab}, {acc}"
        # With A transposed (and B not), each B fragment is loaded just before its four products: fewer fragments live at
        # once (agx_cost at K = 256: -8% on G14G, -19% on G13G for agx_sgemm_tile64_ta); the other variants lose or
        # move both ways with it, so they load all of B first.
        interleave = ta and not tb
        for j in range(4):
            idx = f"{cols[j]}, %k] transpose" if tb else f"%k, {cols[j]}]"
            w(f"        %mb{j} = air.simdgroup_matrix_load %B[{idx} : {mem(ab)} -> {mab}")
            if interleave:
                for i in range(4):
                    w(f"        %n{i}{j} = air.simdgroup_matrix_multiply_accumulate %ma{i}, %mb{j}, %a{i}{j} : {tys}")
        if not interleave:
            for i in range(4):
                if a_late:
                    idx = f"%k, {rows[i]}] transpose" if ta else f"{rows[i]}, %k]"
                    w(f"        %ma{i} = air.simdgroup_matrix_load %A[{idx} : {mem(ab)} -> {mab}")
                for j in range(4):
                    w(f"        %n{i}{j} = air.simdgroup_matrix_multiply_accumulate %ma{i}, %mb{j}, %a{i}{j} : {tys}")
        w("        scf.yield " + ", ".join(f"%n{i}{j}" for i in range(4) for j in range(4)) + f" : {ts}")
        w("      }")
        w("      scf.yield " + ", ".join(f"%in#{k}" for k in range(16)) + f" : {ts}")
        w("    }")
    w("")
    w("    // C = alpha * acc + beta * C, C not read when beta is 0 (one branch for the tile).")
    w("    %no_c = arith.cmpf oeq, %beta, %fz : f32")
    w(f"    %malpha = air.simdgroup_matrix_fill %alpha : {acc}")
    w(f"    %mbeta = air.simdgroup_matrix_fill %beta : {acc}")
    mc = M(c)

    def store(v, i, j, s, ind):
        if c == "f32":
            w(f"{ind}air.simdgroup_matrix_store {v}, %C[{rows[i]}, {cols[j]}] : {acc} -> {mem(c)}")
        else:
            w(f"{ind}%h{s} = arith.truncf {v} : {acc} to {mc}")
            w(f"{ind}air.simdgroup_matrix_store %h{s}, %C[{rows[i]}, {cols[j]}] : {mc} -> {mem(c)}")

    w("    scf.if %no_c {")
    for i in range(4):
        for j in range(4):
            s = f"_{i}{j}"
            w(f"      %s{s} = arith.mulf %acc#{4 * i + j}, %malpha : {acc}")
            store(f"%s{s}", i, j, s, "      ")
    w("    } else {")
    for i in range(4):
        for j in range(4):
            s = f"_{i}{j}"
            w(f"      %s{s} = arith.mulf %acc#{4 * i + j}, %malpha : {acc}")
            if c == "f32":
                w(f"      %old{s} = air.simdgroup_matrix_load %C[{rows[i]}, {cols[j]}] : {mem(c)} -> {mc}")
            else:
                w(f"      %oh{s} = air.simdgroup_matrix_load %C[{rows[i]}, {cols[j]}] : {mem(c)} -> {mc}")
                w(f"      %old{s} = arith.extf %oh{s} : {mc} to {acc}")
            w(f"      %r{s} = math.fma %old{s}, %mbeta, %s{s} : {acc}")
            store(f"%r{s}", i, j, s, "      ")
    w("    }")
    w("    gpu.return")
    w("  }")
    return o


def main():
    for path in sys.argv[1:]:
        base = path.rsplit("/", 1)[-1]
        L = open(path).read().split("\n")
        for name, ab, c in KERNELS[base]:
            for (asuffix, aligned), (suffix, ta, tb) in [(al, v) for al in ALIGNMENTS for v in VARIANTS]:
                fn = name + asuffix + suffix
                body = kernel(fn, ab, c, ta, tb, aligned)
                start = next((i for i, l in enumerate(L) if re.match(r"  gpu\.func @%s\(" % fn, l)), None)
                if start is None:
                    end = max(i for i, l in enumerate(L) if l == "}")  # the module's end
                    which = {(False, False): "A and B as they are",
                             (True, False): "A[r, k] at a[k * lda + r]", (False, True): "B[k, c] at b[c * ldb + k]",
                             (True, True): "A[r, k] at a[k * lda + r] and B[k, c] at b[c * ldb + k]"}[(ta, tb)]
                    doc = ["", f"  // {name} with {which} (transposed)" + ("." if aligned else ", without alignment.")]
                    L[end:end] = doc + body
                else:
                    end = next(i for i in range(start, len(L)) if L[i] == "  }")
                    L[start:end + 1] = body
        open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
