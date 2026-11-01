// FFT kernels of zluda_fft_agx (cuFFT), single precision, for every AGX
// generation like sgemm.mlir. Complex numbers are interleaved (re, im) f32
// pairs. A transform of length n runs over `batch` sequences: element i of
// sequence b is at offset b * dist + i * stride (in complex elements). tw holds
// the n twiddles exp(-2 pi i t / n), t = 0 ... n - 1, computed on the host in
// double precision; inverse transforms use their conjugates. Like cuFFT, the
// inverse transform is not normalized.

gpu.module @agx_fft {
  // One radix-2 pass of the Stockham autosort FFT for n a power of two: with
  // span ns (1, 2, 4, ..., n / 2), butterfly j of n / 2 reads src[j] and
  // src[j + n / 2] and writes dst[(j / ns) 2 ns + j mod ns] and that + ns.
  // log2(n) passes, alternating two buffers, give the transform in order.
  // Launch an (n / 2) x batch grid.
  gpu.func @agx_fft_radix2(%src0: memref<1xf32, 1>, %dst0: memref<1xf32, 1>,
                           %tw0: memref<1xf32, 1>, %n_: i32, %ns_: i32,
                           %batch_: i32, %stride_: i32, %dist_: i32,
                           %inverse_: i32) kernel {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %big = arith.constant 4611686018427387904 : index
    %n = arith.index_cast %n_ : i32 to index
    %ns = arith.index_cast %ns_ : i32 to index
    %batch = arith.index_cast %batch_ : i32 to index
    %stride = arith.index_cast %stride_ : i32 to index
    %dist = arith.index_cast %dist_ : i32 to index
    %inverse = arith.index_cast %inverse_ : i32 to index
    %Src = memref.reinterpret_cast %src0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %Dst = memref.reinterpret_cast %dst0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %Tw = memref.reinterpret_cast %tw0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %half = arith.divui %n, %i2 : index
    %j = gpu.global_id x
    %b = gpu.global_id y
    %in_j = arith.cmpi ult, %j, %half : index
    %in_b = arith.cmpi ult, %b, %batch : index
    %live = arith.andi %in_j, %in_b : i1
    scf.if %live {
      %base = arith.muli %b, %dist : index
      // element offsets in floats: 2 * (base + i * stride)
      %ia0 = arith.muli %j, %stride : index
      %ia1 = arith.addi %base, %ia0 : index
      %ia = arith.muli %ia1, %i2 : index
      %jh = arith.addi %j, %half : index
      %ib0 = arith.muli %jh, %stride : index
      %ib1 = arith.addi %base, %ib0 : index
      %ib = arith.muli %ib1, %i2 : index
      %ia_im = arith.addi %ia, %i1 : index
      %ib_im = arith.addi %ib, %i1 : index
      %ar = memref.load %Src[%ia] : memref<?xf32, strided<[1]>, 1>
      %ai = memref.load %Src[%ia_im] : memref<?xf32, strided<[1]>, 1>
      %br = memref.load %Src[%ib] : memref<?xf32, strided<[1]>, 1>
      %bi = memref.load %Src[%ib_im] : memref<?xf32, strided<[1]>, 1>
      // w = exp(-+ 2 pi i k / (2 ns)) = tw[k n / (2 ns)]
      %k = arith.remui %j, %ns : index
      %ns2 = arith.muli %ns, %i2 : index
      %step = arith.divui %n, %ns2 : index
      %t = arith.muli %k, %step : index
      %t2 = arith.muli %t, %i2 : index
      %t2i = arith.addi %t2, %i1 : index
      %wr = memref.load %Tw[%t2] : memref<?xf32, strided<[1]>, 1>
      %wi0 = memref.load %Tw[%t2i] : memref<?xf32, strided<[1]>, 1>
      %is_inv = arith.cmpi ne, %inverse, %i0 : index
      %wi_neg = arith.negf %wi0 : f32
      %wi = arith.select %is_inv, %wi_neg, %wi0 : f32
      // v = b * w
      %bwr0 = arith.mulf %br, %wr : f32
      %vr2 = arith.mulf %bi, %wi : f32
      %vr_ok = arith.subf %bwr0, %vr2 : f32
      %bwi0 = arith.mulf %br, %wi : f32
      %vi = math.fma %bi, %wr, %bwi0 : f32
      %sr = arith.addf %ar, %vr_ok : f32
      %si = arith.addf %ai, %vi : f32
      %dr = arith.subf %ar, %vr_ok : f32
      %di = arith.subf %ai, %vi : f32
      %jd0 = arith.divui %j, %ns : index
      %jd1 = arith.muli %jd0, %ns2 : index
      %jd = arith.addi %jd1, %k : index
      %oa0 = arith.muli %jd, %stride : index
      %oa1 = arith.addi %base, %oa0 : index
      %oa = arith.muli %oa1, %i2 : index
      %jdn = arith.addi %jd, %ns : index
      %ob0 = arith.muli %jdn, %stride : index
      %ob1 = arith.addi %base, %ob0 : index
      %ob = arith.muli %ob1, %i2 : index
      %oa_im = arith.addi %oa, %i1 : index
      %ob_im = arith.addi %ob, %i1 : index
      memref.store %sr, %Dst[%oa] : memref<?xf32, strided<[1]>, 1>
      memref.store %si, %Dst[%oa_im] : memref<?xf32, strided<[1]>, 1>
      memref.store %dr, %Dst[%ob] : memref<?xf32, strided<[1]>, 1>
      memref.store %di, %Dst[%ob_im] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }

  // One radix-4 pass of the Stockham FFT (two radix-2 passes in one, half the
  // memory traffic): with span ns (n / ns a multiple of 4), butterfly j of n / 4
  // reads x_k = src[j + k n / 4], k = 0 ... 3, and with q = j mod ns and
  // t_k = x_k w^(q k), w = exp(-+ 2 pi i / (4 ns)) = tw[n / (4 ns)], writes the
  // 4-point DFT of t to dst[(j / ns) 4 ns + q + m ns], m = 0 ... 3. Launch an
  // (n / 4) x batch grid.
  gpu.func @agx_fft_radix4(%src0: memref<1xf32, 1>, %dst0: memref<1xf32, 1>,
                           %tw0: memref<1xf32, 1>, %n_: i32, %ns_: i32,
                           %batch_: i32, %stride_: i32, %dist_: i32,
                           %inverse_: i32) kernel {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i4 = arith.constant 4 : index
    %big = arith.constant 4611686018427387904 : index
    %n = arith.index_cast %n_ : i32 to index
    %ns = arith.index_cast %ns_ : i32 to index
    %batch = arith.index_cast %batch_ : i32 to index
    %stride = arith.index_cast %stride_ : i32 to index
    %dist = arith.index_cast %dist_ : i32 to index
    %inverse = arith.index_cast %inverse_ : i32 to index
    %Src = memref.reinterpret_cast %src0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %Dst = memref.reinterpret_cast %dst0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %Tw = memref.reinterpret_cast %tw0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %quarter = arith.divui %n, %i4 : index
    %j = gpu.global_id x
    %b = gpu.global_id y
    %in_j = arith.cmpi ult, %j, %quarter : index
    %in_b = arith.cmpi ult, %b, %batch : index
    %live = arith.andi %in_j, %in_b : i1
    scf.if %live {
      %base = arith.muli %b, %dist : index
      %q = arith.remui %j, %ns : index
      %ns4 = arith.muli %ns, %i4 : index
      %step = arith.divui %n, %ns4 : index
      %qs = arith.muli %q, %step : index
      %is_inv = arith.cmpi ne, %inverse, %i0 : index
      %c0 = arith.constant 0 : index
      %off00 = arith.muli %quarter, %c0 : index
      %jj0 = arith.addi %j, %off00 : index
      %e00 = arith.muli %jj0, %stride : index
      %e01 = arith.addi %base, %e00 : index
      %e0 = arith.muli %e01, %i2 : index
      %e0i = arith.addi %e0, %i1 : index
      %xr0 = memref.load %Src[%e0] : memref<?xf32, strided<[1]>, 1>
      %xi0 = memref.load %Src[%e0i] : memref<?xf32, strided<[1]>, 1>
      %c1 = arith.constant 1 : index
      %off10 = arith.muli %quarter, %c1 : index
      %jj1 = arith.addi %j, %off10 : index
      %e10 = arith.muli %jj1, %stride : index
      %e11 = arith.addi %base, %e10 : index
      %e1 = arith.muli %e11, %i2 : index
      %e1i = arith.addi %e1, %i1 : index
      %xr1 = memref.load %Src[%e1] : memref<?xf32, strided<[1]>, 1>
      %xi1 = memref.load %Src[%e1i] : memref<?xf32, strided<[1]>, 1>
      %tq1 = arith.muli %qs, %c1 : index
      %tq12 = arith.muli %tq1, %i2 : index
      %tq12i = arith.addi %tq12, %i1 : index
      %wr1 = memref.load %Tw[%tq12] : memref<?xf32, strided<[1]>, 1>
      %wi10 = memref.load %Tw[%tq12i] : memref<?xf32, strided<[1]>, 1>
      %wi1n = arith.negf %wi10 : f32
      %wi1 = arith.select %is_inv, %wi1n, %wi10 : f32
      %a1r0 = arith.mulf %xr1, %wr1 : f32
      %a1r1 = arith.mulf %xi1, %wi1 : f32
      %tr1 = arith.subf %a1r0, %a1r1 : f32
      %a1i0 = arith.mulf %xr1, %wi1 : f32
      %ti1 = math.fma %xi1, %wr1, %a1i0 : f32
      %c2 = arith.constant 2 : index
      %off20 = arith.muli %quarter, %c2 : index
      %jj2 = arith.addi %j, %off20 : index
      %e20 = arith.muli %jj2, %stride : index
      %e21 = arith.addi %base, %e20 : index
      %e2 = arith.muli %e21, %i2 : index
      %e2i = arith.addi %e2, %i1 : index
      %xr2 = memref.load %Src[%e2] : memref<?xf32, strided<[1]>, 1>
      %xi2 = memref.load %Src[%e2i] : memref<?xf32, strided<[1]>, 1>
      %tq2 = arith.muli %qs, %c2 : index
      %tq22 = arith.muli %tq2, %i2 : index
      %tq22i = arith.addi %tq22, %i1 : index
      %wr2 = memref.load %Tw[%tq22] : memref<?xf32, strided<[1]>, 1>
      %wi20 = memref.load %Tw[%tq22i] : memref<?xf32, strided<[1]>, 1>
      %wi2n = arith.negf %wi20 : f32
      %wi2 = arith.select %is_inv, %wi2n, %wi20 : f32
      %a2r0 = arith.mulf %xr2, %wr2 : f32
      %a2r1 = arith.mulf %xi2, %wi2 : f32
      %tr2 = arith.subf %a2r0, %a2r1 : f32
      %a2i0 = arith.mulf %xr2, %wi2 : f32
      %ti2 = math.fma %xi2, %wr2, %a2i0 : f32
      %c3 = arith.constant 3 : index
      %off30 = arith.muli %quarter, %c3 : index
      %jj3 = arith.addi %j, %off30 : index
      %e30 = arith.muli %jj3, %stride : index
      %e31 = arith.addi %base, %e30 : index
      %e3 = arith.muli %e31, %i2 : index
      %e3i = arith.addi %e3, %i1 : index
      %xr3 = memref.load %Src[%e3] : memref<?xf32, strided<[1]>, 1>
      %xi3 = memref.load %Src[%e3i] : memref<?xf32, strided<[1]>, 1>
      %tq3 = arith.muli %qs, %c3 : index
      %tq32 = arith.muli %tq3, %i2 : index
      %tq32i = arith.addi %tq32, %i1 : index
      %wr3 = memref.load %Tw[%tq32] : memref<?xf32, strided<[1]>, 1>
      %wi30 = memref.load %Tw[%tq32i] : memref<?xf32, strided<[1]>, 1>
      %wi3n = arith.negf %wi30 : f32
      %wi3 = arith.select %is_inv, %wi3n, %wi30 : f32
      %a3r0 = arith.mulf %xr3, %wr3 : f32
      %a3r1 = arith.mulf %xi3, %wi3 : f32
      %tr3 = arith.subf %a3r0, %a3r1 : f32
      %a3i0 = arith.mulf %xr3, %wi3 : f32
      %ti3 = math.fma %xi3, %wr3, %a3i0 : f32
      // 4-point DFT: s02 = t0 + t2, d02 = t0 - t2, s13 = t1 + t3, d13 = t1 - t3;
      // y0 = s02 + s13, y2 = s02 - s13, y1 = d02 - i d13, y3 = d02 + i d13 (forward;
      // the inverse swaps y1 and y3)
      %s02r = arith.addf %xr0, %tr2 : f32
      %s02i = arith.addf %xi0, %ti2 : f32
      %d02r = arith.subf %xr0, %tr2 : f32
      %d02i = arith.subf %xi0, %ti2 : f32
      %s13r = arith.addf %tr1, %tr3 : f32
      %s13i = arith.addf %ti1, %ti3 : f32
      %d13r = arith.subf %tr1, %tr3 : f32
      %d13i = arith.subf %ti1, %ti3 : f32
      %y0r = arith.addf %s02r, %s13r : f32
      %y0i = arith.addf %s02i, %s13i : f32
      %y2r = arith.subf %s02r, %s13r : f32
      %y2i = arith.subf %s02i, %s13i : f32
      // - i d13 = (d13i, -d13r)
      %ma_r = arith.addf %d02r, %d13i : f32
      %ma_i = arith.subf %d02i, %d13r : f32
      %mb_r = arith.subf %d02r, %d13i : f32
      %mb_i = arith.addf %d02i, %d13r : f32
      %y1r = arith.select %is_inv, %mb_r, %ma_r : f32
      %y1i = arith.select %is_inv, %mb_i, %ma_i : f32
      %y3r = arith.select %is_inv, %ma_r, %mb_r : f32
      %y3i = arith.select %is_inv, %ma_i, %mb_i : f32
      %jd0 = arith.divui %j, %ns : index
      %jd1 = arith.muli %jd0, %ns4 : index
      %jd = arith.addi %jd1, %q : index
      %o00 = arith.muli %ns, %c0 : index
      %o01 = arith.addi %jd, %o00 : index
      %o02 = arith.muli %o01, %stride : index
      %o03 = arith.addi %base, %o02 : index
      %o0 = arith.muli %o03, %i2 : index
      %o0i = arith.addi %o0, %i1 : index
      memref.store %y0r, %Dst[%o0] : memref<?xf32, strided<[1]>, 1>
      memref.store %y0i, %Dst[%o0i] : memref<?xf32, strided<[1]>, 1>
      %o10 = arith.muli %ns, %c1 : index
      %o11 = arith.addi %jd, %o10 : index
      %o12 = arith.muli %o11, %stride : index
      %o13 = arith.addi %base, %o12 : index
      %o1 = arith.muli %o13, %i2 : index
      %o1i = arith.addi %o1, %i1 : index
      memref.store %y1r, %Dst[%o1] : memref<?xf32, strided<[1]>, 1>
      memref.store %y1i, %Dst[%o1i] : memref<?xf32, strided<[1]>, 1>
      %o20 = arith.muli %ns, %c2 : index
      %o21 = arith.addi %jd, %o20 : index
      %o22 = arith.muli %o21, %stride : index
      %o23 = arith.addi %base, %o22 : index
      %o2 = arith.muli %o23, %i2 : index
      %o2i = arith.addi %o2, %i1 : index
      memref.store %y2r, %Dst[%o2] : memref<?xf32, strided<[1]>, 1>
      memref.store %y2i, %Dst[%o2i] : memref<?xf32, strided<[1]>, 1>
      %o30 = arith.muli %ns, %c3 : index
      %o31 = arith.addi %jd, %o30 : index
      %o32 = arith.muli %o31, %stride : index
      %o33 = arith.addi %base, %o32 : index
      %o3 = arith.muli %o33, %i2 : index
      %o3i = arith.addi %o3, %i1 : index
      memref.store %y3r, %Dst[%o3] : memref<?xf32, strided<[1]>, 1>
      memref.store %y3i, %Dst[%o3i] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }

  // A whole power-of-two transform in threadgroup memory (4 <= n, n x
  // (threads y) <= 4096; single precision only, see agx_libs/build.rs):
  // one SIMD-group row of n / 4 threads (x) per sequence, (threads y)
  // sequences per workgroup, so each element is read and written in device
  // memory once. Thread j loads src[j + k n / 4], k < 4, and when log2(n) is
  // odd does the twiddle-free first radix-2 pass on them (the butterflies j and
  // j + n / 4 of agx_fft_radix2 with ns = 1); then radix-4 passes as
  // agx_fft_radix4, through threadgroup memory (real parts, then imaginary
  // parts), the last one storing to dst. In place is allowed. Launch a
  // ((n / 4) x (batch / threads y, rounded up)) grid of (n / 4, threads y)
  // groups.
  gpu.func @agx_fft_shared(%src0: memref<1xf32, 1>, %dst0: memref<1xf32, 1>,
                           %tw0: memref<1xf32, 1>, %n_: i32, %batch_: i32,
                           %stride_: i32, %dist_: i32, %inverse_: i32)
      workgroup(%sm: memref<8192xf32, #gpu.address_space<workgroup>>) kernel {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index
    %i4 = arith.constant 4 : index
    %big = arith.constant 4611686018427387904 : index
    %n = arith.index_cast %n_ : i32 to index
    %batch = arith.index_cast %batch_ : i32 to index
    %stride = arith.index_cast %stride_ : i32 to index
    %dist = arith.index_cast %dist_ : i32 to index
    %inverse = arith.index_cast %inverse_ : i32 to index
    %Src = memref.reinterpret_cast %src0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %Dst = memref.reinterpret_cast %dst0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %Tw = memref.reinterpret_cast %tw0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %is_inv = arith.cmpi ne, %inverse, %i0 : index
    %j = gpu.thread_id x
    %s = gpu.thread_id y
    %per = gpu.block_dim y
    %grp = gpu.block_id x
    %b0 = arith.muli %grp, %per : index
    %b = arith.addi %b0, %s : index
    %live = arith.cmpi ult, %b, %batch : index
    %base = arith.muli %b, %dist : index
    %slot = arith.muli %s, %n : index
    %im0 = arith.muli %per, %n : index
    %quarter = arith.shrui %n, %i2 : index
    %half = arith.shrui %n, %i1 : index
    %jk0 = arith.addi %j, %i0 : index
    %jk1 = arith.addi %j, %quarter : index
    %jk2 = arith.addi %j, %half : index
    %jk3 = arith.addi %jk1, %half : index
    // log2(n) odd: 0xAAAAAAAA has the odd bits
    %oddbits = arith.constant 2863311530 : index
    %ob = arith.andi %n, %oddbits : index
    %odd = arith.cmpi ne, %ob, %i0 : index
    scf.if %live {
      %v0 = arith.muli %jk0, %stride : index
      %v1 = arith.addi %base, %v0 : index
      %v2 = arith.muli %v1, %i2 : index
      %v3 = arith.addi %v2, %i1 : index
      %xr4 = memref.load %Src[%v2] : memref<?xf32, strided<[1]>, 1>
      %xi5 = memref.load %Src[%v3] : memref<?xf32, strided<[1]>, 1>
      %v6 = arith.muli %jk1, %stride : index
      %v7 = arith.addi %base, %v6 : index
      %v8 = arith.muli %v7, %i2 : index
      %v9 = arith.addi %v8, %i1 : index
      %xr10 = memref.load %Src[%v8] : memref<?xf32, strided<[1]>, 1>
      %xi11 = memref.load %Src[%v9] : memref<?xf32, strided<[1]>, 1>
      %v12 = arith.muli %jk2, %stride : index
      %v13 = arith.addi %base, %v12 : index
      %v14 = arith.muli %v13, %i2 : index
      %v15 = arith.addi %v14, %i1 : index
      %xr16 = memref.load %Src[%v14] : memref<?xf32, strided<[1]>, 1>
      %xi17 = memref.load %Src[%v15] : memref<?xf32, strided<[1]>, 1>
      %v18 = arith.muli %jk3, %stride : index
      %v19 = arith.addi %base, %v18 : index
      %v20 = arith.muli %v19, %i2 : index
      %v21 = arith.addi %v20, %i1 : index
      %xr22 = memref.load %Src[%v20] : memref<?xf32, strided<[1]>, 1>
      %xi23 = memref.load %Src[%v21] : memref<?xf32, strided<[1]>, 1>
      scf.if %odd {
        %v24 = arith.addf %xr4, %xr16 : f32
        %v25 = arith.addf %xi5, %xi17 : f32
        %v26 = arith.subf %xr4, %xr16 : f32
        %v27 = arith.subf %xi5, %xi17 : f32
        %v28 = arith.addf %xr10, %xr22 : f32
        %v29 = arith.addf %xi11, %xi23 : f32
        %v30 = arith.subf %xr10, %xr22 : f32
        %v31 = arith.subf %xi11, %xi23 : f32
        %v32 = arith.muli %j, %i2 : index
        %v33 = arith.addi %v32, %i1 : index
        %v34 = arith.muli %jk1, %i2 : index
        %v35 = arith.addi %v34, %i1 : index
        %v36 = arith.addi %slot, %v32 : index
        %v37 = arith.addi %im0, %v36 : index
        memref.store %v24, %sm[%v36] : memref<8192xf32, #gpu.address_space<workgroup>>
        memref.store %v25, %sm[%v37] : memref<8192xf32, #gpu.address_space<workgroup>>
        %v38 = arith.addi %slot, %v33 : index
        %v39 = arith.addi %im0, %v38 : index
        memref.store %v26, %sm[%v38] : memref<8192xf32, #gpu.address_space<workgroup>>
        memref.store %v27, %sm[%v39] : memref<8192xf32, #gpu.address_space<workgroup>>
        %v40 = arith.addi %slot, %v34 : index
        %v41 = arith.addi %im0, %v40 : index
        memref.store %v28, %sm[%v40] : memref<8192xf32, #gpu.address_space<workgroup>>
        memref.store %v29, %sm[%v41] : memref<8192xf32, #gpu.address_space<workgroup>>
        %v42 = arith.addi %slot, %v35 : index
        %v43 = arith.addi %im0, %v42 : index
        memref.store %v30, %sm[%v42] : memref<8192xf32, #gpu.address_space<workgroup>>
        memref.store %v31, %sm[%v43] : memref<8192xf32, #gpu.address_space<workgroup>>
      } else {
        %v44 = arith.addi %slot, %jk0 : index
        %v45 = arith.addi %im0, %v44 : index
        memref.store %xr4, %sm[%v44] : memref<8192xf32, #gpu.address_space<workgroup>>
        memref.store %xi5, %sm[%v45] : memref<8192xf32, #gpu.address_space<workgroup>>
        %v46 = arith.addi %slot, %jk1 : index
        %v47 = arith.addi %im0, %v46 : index
        memref.store %xr10, %sm[%v46] : memref<8192xf32, #gpu.address_space<workgroup>>
        memref.store %xi11, %sm[%v47] : memref<8192xf32, #gpu.address_space<workgroup>>
        %v48 = arith.addi %slot, %jk2 : index
        %v49 = arith.addi %im0, %v48 : index
        memref.store %xr16, %sm[%v48] : memref<8192xf32, #gpu.address_space<workgroup>>
        memref.store %xi17, %sm[%v49] : memref<8192xf32, #gpu.address_space<workgroup>>
        %v50 = arith.addi %slot, %jk3 : index
        %v51 = arith.addi %im0, %v50 : index
        memref.store %xr22, %sm[%v50] : memref<8192xf32, #gpu.address_space<workgroup>>
        memref.store %xi23, %sm[%v51] : memref<8192xf32, #gpu.address_space<workgroup>>
      }
    }
    gpu.barrier
    %ns0 = arith.select %odd, %i2, %i1 : index
    %lns0 = arith.select %odd, %i1, %i0 : index
    %eighth = arith.shrui %quarter, %i1 : index
    %step0 = arith.select %odd, %eighth, %quarter : index
    %nsl:3 = scf.while (%ns = %ns0, %lns = %lns0, %step = %step0) : (index, index, index) -> (index, index, index) {
      %more = arith.cmpi ugt, %step, %i1 : index
      scf.condition(%more) %ns, %lns, %step : index, index, index
    } do {
      ^bb0(%ns: index, %lns: index, %step: index):
      %v52 = arith.subi %ns, %i1 : index
      %v53 = arith.andi %j, %v52 : index
      %v54 = arith.muli %ns, %i4 : index
      %v55 = arith.muli %v53, %step : index
      %v56 = arith.addi %slot, %jk0 : index
      %v57 = arith.addi %im0, %v56 : index
      %xr58 = memref.load %sm[%v56] : memref<8192xf32, #gpu.address_space<workgroup>>
      %xi59 = memref.load %sm[%v57] : memref<8192xf32, #gpu.address_space<workgroup>>
      %v60 = arith.addi %slot, %jk1 : index
      %v61 = arith.addi %im0, %v60 : index
      %xr62 = memref.load %sm[%v60] : memref<8192xf32, #gpu.address_space<workgroup>>
      %xi63 = memref.load %sm[%v61] : memref<8192xf32, #gpu.address_space<workgroup>>
      %v64 = arith.addi %slot, %jk2 : index
      %v65 = arith.addi %im0, %v64 : index
      %xr66 = memref.load %sm[%v64] : memref<8192xf32, #gpu.address_space<workgroup>>
      %xi67 = memref.load %sm[%v65] : memref<8192xf32, #gpu.address_space<workgroup>>
      %v68 = arith.addi %slot, %jk3 : index
      %v69 = arith.addi %im0, %v68 : index
      %xr70 = memref.load %sm[%v68] : memref<8192xf32, #gpu.address_space<workgroup>>
      %xi71 = memref.load %sm[%v69] : memref<8192xf32, #gpu.address_space<workgroup>>
      %v72 = arith.muli %v55, %i1 : index
      %v73 = arith.muli %v72, %i2 : index
      %v74 = arith.addi %v73, %i1 : index
      %v75 = memref.load %Tw[%v73] : memref<?xf32, strided<[1]>, 1>
      %v76 = memref.load %Tw[%v74] : memref<?xf32, strided<[1]>, 1>
      %v77 = arith.negf %v76 : f32
      %v78 = arith.select %is_inv, %v77, %v76 : f32
      %v79 = arith.mulf %xr62, %v75 : f32
      %v80 = arith.mulf %xi63, %v78 : f32
      %v81 = arith.subf %v79, %v80 : f32
      %v82 = arith.mulf %xr62, %v78 : f32
      %v83 = math.fma %xi63, %v75, %v82 : f32
      %v84 = arith.muli %v55, %i2 : index
      %v85 = arith.muli %v84, %i2 : index
      %v86 = arith.addi %v85, %i1 : index
      %v87 = memref.load %Tw[%v85] : memref<?xf32, strided<[1]>, 1>
      %v88 = memref.load %Tw[%v86] : memref<?xf32, strided<[1]>, 1>
      %v89 = arith.negf %v88 : f32
      %v90 = arith.select %is_inv, %v89, %v88 : f32
      %v91 = arith.mulf %xr66, %v87 : f32
      %v92 = arith.mulf %xi67, %v90 : f32
      %v93 = arith.subf %v91, %v92 : f32
      %v94 = arith.mulf %xr66, %v90 : f32
      %v95 = math.fma %xi67, %v87, %v94 : f32
      %v96 = arith.muli %v55, %i3 : index
      %v97 = arith.muli %v96, %i2 : index
      %v98 = arith.addi %v97, %i1 : index
      %v99 = memref.load %Tw[%v97] : memref<?xf32, strided<[1]>, 1>
      %v100 = memref.load %Tw[%v98] : memref<?xf32, strided<[1]>, 1>
      %v101 = arith.negf %v100 : f32
      %v102 = arith.select %is_inv, %v101, %v100 : f32
      %v103 = arith.mulf %xr70, %v99 : f32
      %v104 = arith.mulf %xi71, %v102 : f32
      %v105 = arith.subf %v103, %v104 : f32
      %v106 = arith.mulf %xr70, %v102 : f32
      %v107 = math.fma %xi71, %v99, %v106 : f32
      %v108 = arith.addf %xr58, %v93 : f32
      %v109 = arith.addf %xi59, %v95 : f32
      %v110 = arith.subf %xr58, %v93 : f32
      %v111 = arith.subf %xi59, %v95 : f32
      %v112 = arith.addf %v81, %v105 : f32
      %v113 = arith.addf %v83, %v107 : f32
      %v114 = arith.subf %v81, %v105 : f32
      %v115 = arith.subf %v83, %v107 : f32
      %v116 = arith.addf %v108, %v112 : f32
      %v117 = arith.addf %v109, %v113 : f32
      %v118 = arith.subf %v108, %v112 : f32
      %v119 = arith.subf %v109, %v113 : f32
      %v120 = arith.addf %v110, %v115 : f32
      %v121 = arith.subf %v111, %v114 : f32
      %v122 = arith.subf %v110, %v115 : f32
      %v123 = arith.addf %v111, %v114 : f32
      %v124 = arith.select %is_inv, %v122, %v120 : f32
      %v125 = arith.select %is_inv, %v123, %v121 : f32
      %v126 = arith.select %is_inv, %v120, %v122 : f32
      %v127 = arith.select %is_inv, %v121, %v123 : f32
      %v128 = arith.shrui %j, %lns : index
      %v129 = arith.muli %v128, %v54 : index
      %v130 = arith.addi %v129, %v53 : index
      %v131 = arith.muli %ns, %i0 : index
      %v132 = arith.addi %v130, %v131 : index
      %v133 = arith.muli %ns, %i1 : index
      %v134 = arith.addi %v130, %v133 : index
      %v135 = arith.muli %ns, %i2 : index
      %v136 = arith.addi %v130, %v135 : index
      %v137 = arith.muli %ns, %i3 : index
      %v138 = arith.addi %v130, %v137 : index
      gpu.barrier
      %v139 = arith.addi %slot, %v132 : index
      %v140 = arith.addi %im0, %v139 : index
      memref.store %v116, %sm[%v139] : memref<8192xf32, #gpu.address_space<workgroup>>
      memref.store %v117, %sm[%v140] : memref<8192xf32, #gpu.address_space<workgroup>>
      %v141 = arith.addi %slot, %v134 : index
      %v142 = arith.addi %im0, %v141 : index
      memref.store %v124, %sm[%v141] : memref<8192xf32, #gpu.address_space<workgroup>>
      memref.store %v125, %sm[%v142] : memref<8192xf32, #gpu.address_space<workgroup>>
      %v143 = arith.addi %slot, %v136 : index
      %v144 = arith.addi %im0, %v143 : index
      memref.store %v118, %sm[%v143] : memref<8192xf32, #gpu.address_space<workgroup>>
      memref.store %v119, %sm[%v144] : memref<8192xf32, #gpu.address_space<workgroup>>
      %v145 = arith.addi %slot, %v138 : index
      %v146 = arith.addi %im0, %v145 : index
      memref.store %v126, %sm[%v145] : memref<8192xf32, #gpu.address_space<workgroup>>
      memref.store %v127, %sm[%v146] : memref<8192xf32, #gpu.address_space<workgroup>>
      gpu.barrier
      %nsn = arith.muli %ns, %i4 : index
      %lnsn = arith.addi %lns, %i2 : index
      %stepn = arith.shrui %step, %i2 : index
      scf.yield %nsn, %lnsn, %stepn : index, index, index
    }
    scf.if %live {
      %v147 = arith.subi %nsl#0, %i1 : index
      %v148 = arith.andi %j, %v147 : index
      %v149 = arith.muli %nsl#0, %i4 : index
      %v150 = arith.muli %v148, %nsl#2 : index
      %v151 = arith.addi %slot, %jk0 : index
      %v152 = arith.addi %im0, %v151 : index
      %xr153 = memref.load %sm[%v151] : memref<8192xf32, #gpu.address_space<workgroup>>
      %xi154 = memref.load %sm[%v152] : memref<8192xf32, #gpu.address_space<workgroup>>
      %v155 = arith.addi %slot, %jk1 : index
      %v156 = arith.addi %im0, %v155 : index
      %xr157 = memref.load %sm[%v155] : memref<8192xf32, #gpu.address_space<workgroup>>
      %xi158 = memref.load %sm[%v156] : memref<8192xf32, #gpu.address_space<workgroup>>
      %v159 = arith.addi %slot, %jk2 : index
      %v160 = arith.addi %im0, %v159 : index
      %xr161 = memref.load %sm[%v159] : memref<8192xf32, #gpu.address_space<workgroup>>
      %xi162 = memref.load %sm[%v160] : memref<8192xf32, #gpu.address_space<workgroup>>
      %v163 = arith.addi %slot, %jk3 : index
      %v164 = arith.addi %im0, %v163 : index
      %xr165 = memref.load %sm[%v163] : memref<8192xf32, #gpu.address_space<workgroup>>
      %xi166 = memref.load %sm[%v164] : memref<8192xf32, #gpu.address_space<workgroup>>
      %v167 = arith.muli %v150, %i1 : index
      %v168 = arith.muli %v167, %i2 : index
      %v169 = arith.addi %v168, %i1 : index
      %v170 = memref.load %Tw[%v168] : memref<?xf32, strided<[1]>, 1>
      %v171 = memref.load %Tw[%v169] : memref<?xf32, strided<[1]>, 1>
      %v172 = arith.negf %v171 : f32
      %v173 = arith.select %is_inv, %v172, %v171 : f32
      %v174 = arith.mulf %xr157, %v170 : f32
      %v175 = arith.mulf %xi158, %v173 : f32
      %v176 = arith.subf %v174, %v175 : f32
      %v177 = arith.mulf %xr157, %v173 : f32
      %v178 = math.fma %xi158, %v170, %v177 : f32
      %v179 = arith.muli %v150, %i2 : index
      %v180 = arith.muli %v179, %i2 : index
      %v181 = arith.addi %v180, %i1 : index
      %v182 = memref.load %Tw[%v180] : memref<?xf32, strided<[1]>, 1>
      %v183 = memref.load %Tw[%v181] : memref<?xf32, strided<[1]>, 1>
      %v184 = arith.negf %v183 : f32
      %v185 = arith.select %is_inv, %v184, %v183 : f32
      %v186 = arith.mulf %xr161, %v182 : f32
      %v187 = arith.mulf %xi162, %v185 : f32
      %v188 = arith.subf %v186, %v187 : f32
      %v189 = arith.mulf %xr161, %v185 : f32
      %v190 = math.fma %xi162, %v182, %v189 : f32
      %v191 = arith.muli %v150, %i3 : index
      %v192 = arith.muli %v191, %i2 : index
      %v193 = arith.addi %v192, %i1 : index
      %v194 = memref.load %Tw[%v192] : memref<?xf32, strided<[1]>, 1>
      %v195 = memref.load %Tw[%v193] : memref<?xf32, strided<[1]>, 1>
      %v196 = arith.negf %v195 : f32
      %v197 = arith.select %is_inv, %v196, %v195 : f32
      %v198 = arith.mulf %xr165, %v194 : f32
      %v199 = arith.mulf %xi166, %v197 : f32
      %v200 = arith.subf %v198, %v199 : f32
      %v201 = arith.mulf %xr165, %v197 : f32
      %v202 = math.fma %xi166, %v194, %v201 : f32
      %v203 = arith.addf %xr153, %v188 : f32
      %v204 = arith.addf %xi154, %v190 : f32
      %v205 = arith.subf %xr153, %v188 : f32
      %v206 = arith.subf %xi154, %v190 : f32
      %v207 = arith.addf %v176, %v200 : f32
      %v208 = arith.addf %v178, %v202 : f32
      %v209 = arith.subf %v176, %v200 : f32
      %v210 = arith.subf %v178, %v202 : f32
      %v211 = arith.addf %v203, %v207 : f32
      %v212 = arith.addf %v204, %v208 : f32
      %v213 = arith.subf %v203, %v207 : f32
      %v214 = arith.subf %v204, %v208 : f32
      %v215 = arith.addf %v205, %v210 : f32
      %v216 = arith.subf %v206, %v209 : f32
      %v217 = arith.subf %v205, %v210 : f32
      %v218 = arith.addf %v206, %v209 : f32
      %v219 = arith.select %is_inv, %v217, %v215 : f32
      %v220 = arith.select %is_inv, %v218, %v216 : f32
      %v221 = arith.select %is_inv, %v215, %v217 : f32
      %v222 = arith.select %is_inv, %v216, %v218 : f32
      %v223 = arith.shrui %j, %nsl#1 : index
      %v224 = arith.muli %v223, %v149 : index
      %v225 = arith.addi %v224, %v148 : index
      %v226 = arith.muli %nsl#0, %i0 : index
      %v227 = arith.addi %v225, %v226 : index
      %v228 = arith.muli %nsl#0, %i1 : index
      %v229 = arith.addi %v225, %v228 : index
      %v230 = arith.muli %nsl#0, %i2 : index
      %v231 = arith.addi %v225, %v230 : index
      %v232 = arith.muli %nsl#0, %i3 : index
      %v233 = arith.addi %v225, %v232 : index
      %v234 = arith.muli %v227, %stride : index
      %v235 = arith.addi %base, %v234 : index
      %v236 = arith.muli %v235, %i2 : index
      %v237 = arith.addi %v236, %i1 : index
      memref.store %v211, %Dst[%v236] : memref<?xf32, strided<[1]>, 1>
      memref.store %v212, %Dst[%v237] : memref<?xf32, strided<[1]>, 1>
      %v238 = arith.muli %v229, %stride : index
      %v239 = arith.addi %base, %v238 : index
      %v240 = arith.muli %v239, %i2 : index
      %v241 = arith.addi %v240, %i1 : index
      memref.store %v219, %Dst[%v240] : memref<?xf32, strided<[1]>, 1>
      memref.store %v220, %Dst[%v241] : memref<?xf32, strided<[1]>, 1>
      %v242 = arith.muli %v231, %stride : index
      %v243 = arith.addi %base, %v242 : index
      %v244 = arith.muli %v243, %i2 : index
      %v245 = arith.addi %v244, %i1 : index
      memref.store %v213, %Dst[%v244] : memref<?xf32, strided<[1]>, 1>
      memref.store %v214, %Dst[%v245] : memref<?xf32, strided<[1]>, 1>
      %v246 = arith.muli %v233, %stride : index
      %v247 = arith.addi %base, %v246 : index
      %v248 = arith.muli %v247, %i2 : index
      %v249 = arith.addi %v248, %i1 : index
      memref.store %v221, %Dst[%v248] : memref<?xf32, strided<[1]>, 1>
      memref.store %v222, %Dst[%v249] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }

  // One Stockham pass of any radix r (n / ns a multiple of r), one thread per
  // output: thread g of n, with m = g / (n / r) and j = g mod (n / r), q = j mod
  // ns, writes dst[(j / ns) r ns + q + m ns] = sum over k < r of
  // src[j + k n / r] tw[(k (q n / (r ns) + m n / r)) mod n] (conjugated when
  // inverse): the twiddle of the pass times that of the r-point transform. With
  // r = n and ns = 1 it is the direct transform. Launch an n x batch grid; src
  // and dst must not overlap.
  gpu.func @agx_fft_radixr(%src0: memref<1xf32, 1>, %dst0: memref<1xf32, 1>,
                           %tw0: memref<1xf32, 1>, %n_: i32, %ns_: i32, %r_: i32,
                           %batch_: i32, %stride_: i32, %dist_: i32,
                           %inverse_: i32) kernel {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %f0 = arith.constant 0.0 : f32
    %big = arith.constant 4611686018427387904 : index
    %n = arith.index_cast %n_ : i32 to index
    %ns = arith.index_cast %ns_ : i32 to index
    %r = arith.index_cast %r_ : i32 to index
    %batch = arith.index_cast %batch_ : i32 to index
    %stride = arith.index_cast %stride_ : i32 to index
    %dist = arith.index_cast %dist_ : i32 to index
    %inverse = arith.index_cast %inverse_ : i32 to index
    %Src = memref.reinterpret_cast %src0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %Dst = memref.reinterpret_cast %dst0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %Tw = memref.reinterpret_cast %tw0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %g = gpu.global_id x
    %b = gpu.global_id y
    %in_g = arith.cmpi ult, %g, %n : index
    %in_b = arith.cmpi ult, %b, %batch : index
    %live = arith.andi %in_g, %in_b : i1
    scf.if %live {
      %base = arith.muli %b, %dist : index
      %is_inv = arith.cmpi ne, %inverse, %i0 : index
      %nr = arith.divui %n, %r : index
      %m = arith.divui %g, %nr : index
      %j = arith.remui %g, %nr : index
      %q = arith.remui %j, %ns : index
      %rns = arith.muli %r, %ns : index
      %step = arith.divui %n, %rns : index
      %qs = arith.muli %q, %step : index
      %mn = arith.muli %m, %nr : index
      // below n: q step < n / r and m n / r <= n - n / r
      %inc = arith.addi %qs, %mn : index
      %sum:4 = scf.for %k = %i0 to %r step %i1 iter_args(%sr = %f0, %si = %f0, %t = %i0, %jk = %j) -> (f32, f32, index, index) {
        %ij0 = arith.muli %jk, %stride : index
        %ij1 = arith.addi %base, %ij0 : index
        %ij = arith.muli %ij1, %i2 : index
        %ij_im = arith.addi %ij, %i1 : index
        %xr = memref.load %Src[%ij] : memref<?xf32, strided<[1]>, 1>
        %xi = memref.load %Src[%ij_im] : memref<?xf32, strided<[1]>, 1>
        %t2 = arith.muli %t, %i2 : index
        %t2i = arith.addi %t2, %i1 : index
        %wr = memref.load %Tw[%t2] : memref<?xf32, strided<[1]>, 1>
        %wi0 = memref.load %Tw[%t2i] : memref<?xf32, strided<[1]>, 1>
        %wi_neg = arith.negf %wi0 : f32
        %wi = arith.select %is_inv, %wi_neg, %wi0 : f32
        // (xr + i xi)(wr + i wi)
        %p0 = math.fma %xr, %wr, %sr : f32
        %ni = arith.negf %xi : f32
        %sr1 = math.fma %ni, %wi, %p0 : f32
        %q0 = math.fma %xr, %wi, %si : f32
        %si1 = math.fma %xi, %wr, %q0 : f32
        // next twiddle index (k + 1) inc mod n, next element j + (k + 1) n / r
        %t_n0 = arith.addi %t, %inc : index
        %wrap = arith.cmpi uge, %t_n0, %n : index
        %t_n1 = arith.subi %t_n0, %n : index
        %t_n = arith.select %wrap, %t_n1, %t_n0 : index
        %jk_n = arith.addi %jk, %nr : index
        scf.yield %sr1, %si1, %t_n, %jk_n : f32, f32, index, index
      }
      %jd0 = arith.divui %j, %ns : index
      %jd1 = arith.muli %jd0, %rns : index
      %jd2 = arith.addi %jd1, %q : index
      %mns = arith.muli %m, %ns : index
      %jd = arith.addi %jd2, %mns : index
      %ok0 = arith.muli %jd, %stride : index
      %ok1 = arith.addi %base, %ok0 : index
      %ok = arith.muli %ok1, %i2 : index
      %ok_im = arith.addi %ok, %i1 : index
      memref.store %sum#0, %Dst[%ok] : memref<?xf32, strided<[1]>, 1>
      memref.store %sum#1, %Dst[%ok_im] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }

  // Real to complex input: dst (complex, n per sequence, contiguous) =
  // src (real, n per sequence, idist apart) with zero imaginary parts.
  // Launch an n x batch grid.
  gpu.func @agx_fft_real_to_complex(%src0: memref<1xf32, 1>, %dst0: memref<1xf32, 1>,
                                    %n_: i32, %batch_: i32, %idist_: i32) kernel {
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %f0 = arith.constant 0.0 : f32
    %big = arith.constant 4611686018427387904 : index
    %n = arith.index_cast %n_ : i32 to index
    %batch = arith.index_cast %batch_ : i32 to index
    %idist = arith.index_cast %idist_ : i32 to index
    %Src = memref.reinterpret_cast %src0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %Dst = memref.reinterpret_cast %dst0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %i = gpu.global_id x
    %b = gpu.global_id y
    %in_i = arith.cmpi ult, %i, %n : index
    %in_b = arith.cmpi ult, %b, %batch : index
    %live = arith.andi %in_i, %in_b : i1
    scf.if %live {
      %s0 = arith.muli %b, %idist : index
      %s = arith.addi %s0, %i : index
      %x = memref.load %Src[%s] : memref<?xf32, strided<[1]>, 1>
      %d0 = arith.muli %b, %n : index
      %d1 = arith.addi %d0, %i : index
      %d = arith.muli %d1, %i2 : index
      %d_im = arith.addi %d, %i1 : index
      memref.store %x, %Dst[%d] : memref<?xf32, strided<[1]>, 1>
      memref.store %f0, %Dst[%d_im] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }

  // Copies the first m complex elements of each of `batch` sequences:
  // dst[b odist + i ostride] = src[b idist + i istride] (complex elements).
  // Launch an m x batch grid. Used for the n / 2 + 1 outputs of real to
  // complex transforms and to finish in-place transforms exactly (the
  // elements between strided ones are left alone).
  gpu.func @agx_fft_copy(%src0: memref<1xf32, 1>, %dst0: memref<1xf32, 1>,
                         %m_: i32, %batch_: i32, %istride_: i32, %idist_: i32,
                         %ostride_: i32, %odist_: i32) kernel {
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %big = arith.constant 4611686018427387904 : index
    %m = arith.index_cast %m_ : i32 to index
    %batch = arith.index_cast %batch_ : i32 to index
    %istride = arith.index_cast %istride_ : i32 to index
    %idist = arith.index_cast %idist_ : i32 to index
    %ostride = arith.index_cast %ostride_ : i32 to index
    %odist = arith.index_cast %odist_ : i32 to index
    %Src = memref.reinterpret_cast %src0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %Dst = memref.reinterpret_cast %dst0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %i = gpu.global_id x
    %b = gpu.global_id y
    %in_i = arith.cmpi ult, %i, %m : index
    %in_b = arith.cmpi ult, %b, %batch : index
    %live = arith.andi %in_i, %in_b : i1
    scf.if %live {
      %s0 = arith.muli %b, %idist : index
      %si = arith.muli %i, %istride : index
      %s1 = arith.addi %s0, %si : index
      %s = arith.muli %s1, %i2 : index
      %s_im = arith.addi %s, %i1 : index
      %d0 = arith.muli %b, %odist : index
      %di = arith.muli %i, %ostride : index
      %d1 = arith.addi %d0, %di : index
      %d = arith.muli %d1, %i2 : index
      %d_im = arith.addi %d, %i1 : index
      %xr = memref.load %Src[%s] : memref<?xf32, strided<[1]>, 1>
      %xi = memref.load %Src[%s_im] : memref<?xf32, strided<[1]>, 1>
      memref.store %xr, %Dst[%d] : memref<?xf32, strided<[1]>, 1>
      memref.store %xi, %Dst[%d_im] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }

  // Complex to real input: dst (complex, n per sequence, contiguous) is the
  // full Hermitian spectrum of src (n / 2 + 1 complex elements per sequence,
  // idist apart): dst[k] = src[k] for k <= n / 2, conj(src[n - k]) above.
  // Launch an n x batch grid.
  gpu.func @agx_fft_hermitian(%src0: memref<1xf32, 1>, %dst0: memref<1xf32, 1>,
                              %n_: i32, %batch_: i32, %idist_: i32) kernel {
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %big = arith.constant 4611686018427387904 : index
    %n = arith.index_cast %n_ : i32 to index
    %batch = arith.index_cast %batch_ : i32 to index
    %idist = arith.index_cast %idist_ : i32 to index
    %Src = memref.reinterpret_cast %src0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %Dst = memref.reinterpret_cast %dst0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %k = gpu.global_id x
    %b = gpu.global_id y
    %in_k = arith.cmpi ult, %k, %n : index
    %in_b = arith.cmpi ult, %b, %batch : index
    %live = arith.andi %in_k, %in_b : i1
    scf.if %live {
      %h = arith.divui %n, %i2 : index
      %low = arith.cmpi ule, %k, %h : index
      %mirror = arith.subi %n, %k : index
      %j = arith.select %low, %k, %mirror : index
      %s0 = arith.muli %b, %idist : index
      %s1 = arith.addi %s0, %j : index
      %s = arith.muli %s1, %i2 : index
      %s_im = arith.addi %s, %i1 : index
      %xr = memref.load %Src[%s] : memref<?xf32, strided<[1]>, 1>
      %xi0 = memref.load %Src[%s_im] : memref<?xf32, strided<[1]>, 1>
      %xin = arith.negf %xi0 : f32
      %xi = arith.select %low, %xi0, %xin : f32
      %d0 = arith.muli %b, %n : index
      %d1 = arith.addi %d0, %k : index
      %d = arith.muli %d1, %i2 : index
      %d_im = arith.addi %d, %i1 : index
      memref.store %xr, %Dst[%d] : memref<?xf32, strided<[1]>, 1>
      memref.store %xi, %Dst[%d_im] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }

  // The real parts of src (complex, n per sequence, contiguous) into dst
  // (real, odist apart). Launch an n x batch grid.
  gpu.func @agx_fft_real_part(%src0: memref<1xf32, 1>, %dst0: memref<1xf32, 1>,
                              %n_: i32, %batch_: i32, %odist_: i32) kernel {
    %i2 = arith.constant 2 : index
    %big = arith.constant 4611686018427387904 : index
    %n = arith.index_cast %n_ : i32 to index
    %batch = arith.index_cast %batch_ : i32 to index
    %odist = arith.index_cast %odist_ : i32 to index
    %Src = memref.reinterpret_cast %src0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %Dst = memref.reinterpret_cast %dst0 to offset: [0], sizes: [%big], strides: [1]
        : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %i = gpu.global_id x
    %b = gpu.global_id y
    %in_i = arith.cmpi ult, %i, %n : index
    %in_b = arith.cmpi ult, %b, %batch : index
    %live = arith.andi %in_i, %in_b : i1
    scf.if %live {
      %s0 = arith.muli %b, %n : index
      %s1 = arith.addi %s0, %i : index
      %s = arith.muli %s1, %i2 : index
      %x = memref.load %Src[%s] : memref<?xf32, strided<[1]>, 1>
      %d0 = arith.muli %b, %odist : index
      %d = arith.addi %d0, %i : index
      memref.store %x, %Dst[%d] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }
}
