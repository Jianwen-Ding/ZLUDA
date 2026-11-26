module attributes {gpu.container_module} {
  gpu.module @zluda_blas [#air.target] {
    gpu.func @saxpy(%n: i32, %alpha: f32, %x: memref<?xf32, 1>, %xo: i32, %xi: i32, %y: memref<?xf32, 1>, %yo: i32, %yi: i32) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %b0 = arith.muli %bid, %bdim : index
      %gi = arith.addi %b0, %tid : index
      %gn = arith.muli %gdim, %bdim : index
      %i = arith.index_cast %gi : index to i32
      %stride = arith.index_cast %gn : index to i32
      scf.for %k = %i to %n step %stride : i32 {
      %ix_m = arith.muli %k, %xi : i32
      %ix_a = arith.addi %xo, %ix_m : i32
      %ix = arith.index_cast %ix_a : i32 to index
      %iy_m = arith.muli %k, %yi : i32
      %iy_a = arith.addi %yo, %iy_m : i32
      %iy = arith.index_cast %iy_a : i32 to index
        %xv = memref.load %x[%ix] : memref<?xf32, 1>
        %yv = memref.load %y[%iy] : memref<?xf32, 1>
        %r = math.fma %alpha, %xv, %yv : f32
        memref.store %r, %y[%iy] : memref<?xf32, 1>
      }
      gpu.return
    }
    gpu.func @sscal(%n: i32, %alpha: f32, %x: memref<?xf32, 1>, %xo: i32, %xi: i32) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %b0 = arith.muli %bid, %bdim : index
      %gi = arith.addi %b0, %tid : index
      %gn = arith.muli %gdim, %bdim : index
      %i = arith.index_cast %gi : index to i32
      %stride = arith.index_cast %gn : index to i32
      scf.for %k = %i to %n step %stride : i32 {
      %ix_m = arith.muli %k, %xi : i32
      %ix_a = arith.addi %xo, %ix_m : i32
      %ix = arith.index_cast %ix_a : i32 to index
        %xv = memref.load %x[%ix] : memref<?xf32, 1>
        %r = arith.mulf %alpha, %xv : f32
        memref.store %r, %x[%ix] : memref<?xf32, 1>
      }
      gpu.return
    }
    gpu.func @scopy(%n: i32, %mode: i32, %x: memref<?xf32, 1>, %xo: i32, %xi: i32, %y: memref<?xf32, 1>, %yo: i32, %yi: i32) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %b0 = arith.muli %bid, %bdim : index
      %gi = arith.addi %b0, %tid : index
      %gn = arith.muli %gdim, %bdim : index
      %i = arith.index_cast %gi : index to i32
      %stride = arith.index_cast %gn : index to i32
      %c0 = arith.constant 0 : i32
      %sw = arith.cmpi ne, %mode, %c0 : i32
      scf.for %k = %i to %n step %stride : i32 {
      %ix_m = arith.muli %k, %xi : i32
      %ix_a = arith.addi %xo, %ix_m : i32
      %ix = arith.index_cast %ix_a : i32 to index
      %iy_m = arith.muli %k, %yi : i32
      %iy_a = arith.addi %yo, %iy_m : i32
      %iy = arith.index_cast %iy_a : i32 to index
        %xv = memref.load %x[%ix] : memref<?xf32, 1>
        scf.if %sw {
          %yv = memref.load %y[%iy] : memref<?xf32, 1>
          memref.store %yv, %x[%ix] : memref<?xf32, 1>
        }
        memref.store %xv, %y[%iy] : memref<?xf32, 1>
      }
      gpu.return
    }
    gpu.func @sreduce(%n: i32, %op: i32, %x: memref<?xf32, 1>, %xo: i32, %xi: i32, %y: memref<?xf32, 1>, %yo: i32, %yi: i32, %out: memref<?xf32, 1>) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %b0 = arith.muli %bid, %bdim : index
      %gi = arith.addi %b0, %tid : index
      %gn = arith.muli %gdim, %bdim : index
      %i = arith.index_cast %gi : index to i32
      %stride = arith.index_cast %gn : index to i32
      %zero = arith.constant 0.0 : f32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %acc = scf.for %k = %i to %n step %stride iter_args(%a = %zero) -> (f32) : i32 {
      %ix_m = arith.muli %k, %xi : i32
      %ix_a = arith.addi %xo, %ix_m : i32
      %ix = arith.index_cast %ix_a : i32 to index
        %xv = memref.load %x[%ix] : memref<?xf32, 1>
        %isdot = arith.cmpi eq, %op, %c0 : i32
        %v = scf.if %isdot -> (f32) {
      %iy_m = arith.muli %k, %yi : i32
      %iy_a = arith.addi %yo, %iy_m : i32
      %iy = arith.index_cast %iy_a : i32 to index
          %yv = memref.load %y[%iy] : memref<?xf32, 1>
          %m = arith.mulf %xv, %yv : f32
          scf.yield %m : f32
        } else {
          %issq = arith.cmpi eq, %op, %c1 : i32
          %sq = arith.mulf %xv, %xv : f32
          %isabs = arith.cmpi eq, %op, %c2 : i32
          %ab = math.absf %xv : f32
          %w0 = arith.select %isabs, %ab, %xv : f32
          %w = arith.select %issq, %sq, %w0 : f32
          scf.yield %w : f32
        }
        %na = arith.addf %a, %v : f32
        scf.yield %na : f32
      }
      %s = gpu.subgroup_reduce add %acc : (f32) -> f32
      %lane = gpu.lane_id
      %c0i = arith.constant 0 : index
      %first = arith.cmpi eq, %lane, %c0i : index
      scf.if %first {
        memref.store %s, %out[%bid] : memref<?xf32, 1>
      }
      gpu.return
    }
    gpu.func @samax(%n: i32, %op: i32, %x: memref<?xf32, 1>, %xo: i32, %xi: i32, %y: memref<?xf32, 1>, %out: memref<?xf32, 1>, %outi: memref<?xf32, 1>) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %b0 = arith.muli %bid, %bdim : index
      %gi = arith.addi %b0, %tid : index
      %gn = arith.muli %gdim, %bdim : index
      %i = arith.index_cast %gi : index to i32
      %stride = arith.index_cast %gn : index to i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %opm = arith.remui %op, %c2 : i32
      %ismin = arith.cmpi eq, %opm, %c1 : i32
      %final = arith.cmpi uge, %op, %c2 : i32
      %inf = arith.constant 0x7F800000 : i32
      %pinf = arith.bitcast %inf : i32 to f32
      %pinft = arith.addf %pinf, %pinf : f32
      %neg1 = arith.constant -1.0 : f32
      %start = arith.select %ismin, %pinft, %neg1 : f32
      %big = arith.constant 2147483647 : i32
      %r:2 = scf.for %k = %i to %n step %stride iter_args(%bv = %start, %bi = %big) -> (f32, i32) : i32 {
      %ix_m = arith.muli %k, %xi : i32
      %ix_a = arith.addi %xo, %ix_m : i32
      %ix = arith.index_cast %ix_a : i32 to index
        %xv = memref.load %x[%ix] : memref<?xf32, 1>
        %av0 = math.absf %xv : f32
        %av = arith.select %final, %xv, %av0 : f32
        %ki = arith.index_cast %k : i32 to index
        %pi = scf.if %final -> (i32) {
          %yv = memref.load %y[%ki] : memref<?xf32, 1>
          %yi = arith.fptosi %yv : f32 to i32
          scf.yield %yi : i32
        } else {
          %k1 = arith.addi %k, %c1 : i32
          scf.yield %k1 : i32
        }
        %gt = arith.cmpf ogt, %av, %bv : f32
        %lt = arith.cmpf olt, %av, %bv : f32
        %better = arith.select %ismin, %lt, %gt : i1
        %nv = arith.select %better, %av, %bv : f32
        %ni = arith.select %better, %pi, %bi : i32
        scf.yield %nv, %ni : f32, i32
      }
      // the best value of the SIMD group, then the smallest index that has it
      %mx = gpu.subgroup_reduce maxnumf %r#0 : (f32) -> f32
      %mn = gpu.subgroup_reduce minnumf %r#0 : (f32) -> f32
      %bestv = arith.select %ismin, %mn, %mx : f32
      %has = arith.cmpf oeq, %r#0, %bestv : f32
      %cand = arith.select %has, %r#1, %big : i32
      %besti = gpu.subgroup_reduce minsi %cand : (i32) -> i32
      %lane = gpu.lane_id
      %c0i = arith.constant 0 : index
      %first = arith.cmpi eq, %lane, %c0i : index
      scf.if %first {
        memref.store %bestv, %out[%bid] : memref<?xf32, 1>
        %bif = arith.sitofp %besti : i32 to f32
        memref.store %bif, %outi[%bid] : memref<?xf32, 1>
      }
      gpu.return
    }
    gpu.func @sgemv(%m: i32, %n: i32, %alpha: f32, %a: memref<?xf32, 1>, %ao: i32, %ars: i32, %acs: i32, %x: memref<?xf32, 1>, %xo: i32, %xi: i32, %beta: f32, %y: memref<?xf32, 1>, %yo: i32, %yi: i32) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %b0 = arith.muli %bid, %bdim : index
      %gi = arith.addi %b0, %tid : index
      %gn = arith.muli %gdim, %bdim : index
      %i = arith.index_cast %gi : index to i32
      %stride = arith.index_cast %gn : index to i32
      %zero = arith.constant 0.0 : f32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      scf.for %r = %i to %m step %stride : i32 {
        %rr = arith.muli %r, %ars : i32
        %rowbase = arith.addi %ao, %rr : i32
        %acc = scf.for %j = %c0 to %n step %c1 iter_args(%s = %zero) -> (f32) : i32 {
          %ia_m = arith.muli %j, %acs : i32
          %ia_a = arith.addi %rowbase, %ia_m : i32
          %ia = arith.index_cast %ia_a : i32 to index
          %ix_m = arith.muli %j, %xi : i32
          %ix_a = arith.addi %xo, %ix_m : i32
          %ix = arith.index_cast %ix_a : i32 to index
          %av = memref.load %a[%ia] : memref<?xf32, 1>
          %xv = memref.load %x[%ix] : memref<?xf32, 1>
          %ns = math.fma %av, %xv, %s : f32
          scf.yield %ns : f32
        }
      %iy_m = arith.muli %r, %yi : i32
      %iy_a = arith.addi %yo, %iy_m : i32
      %iy = arith.index_cast %iy_a : i32 to index
        %bz = arith.cmpf oeq, %beta, %zero : f32
        %yv = scf.if %bz -> (f32) {
          scf.yield %zero : f32
        } else {
          %y0 = memref.load %y[%iy] : memref<?xf32, 1>
          %by = arith.mulf %beta, %y0 : f32
          scf.yield %by : f32
        }
        %res = math.fma %alpha, %acc, %yv : f32
        memref.store %res, %y[%iy] : memref<?xf32, 1>
      }
      gpu.return
    }
    gpu.func @sgemm(%m: i32, %n: i32, %k: i32, %alpha: f32, %a: memref<?xf32, 1>, %ao: i32, %ars: i32, %acs: i32, %sa: i32, %b: memref<?xf32, 1>, %bo: i32, %brs: i32, %bcs: i32, %sb: i32, %beta: f32, %c: memref<?xf32, 1>, %co: i32, %ldc: i32, %sc: i32) kernel {
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
      %zero = arith.constant 0.0 : f32
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
      %cbase = arith.addi %co, %zsc : i32
      %mm1 = arith.subi %m, %c1 : i32
      %nm1 = arith.subi %n, %c1 : i32
      %cr0 = arith.constant 0 : i32
      %ro0 = arith.muli %cr0, %dxi : i32
      %row0 = arith.addi %i0, %ro0 : i32
      %rowc0 = arith.minsi %row0, %mm1 : i32
      %rofs0 = arith.muli %rowc0, %ars : i32
      %arow0 = arith.addi %abase, %rofs0 : i32
      %co0 = arith.muli %cr0, %dyi : i32
      %col0 = arith.addi %j0, %co0 : i32
      %colc0 = arith.minsi %col0, %nm1 : i32
      %cofs0 = arith.muli %colc0, %bcs : i32
      %bcol0 = arith.addi %bbase, %cofs0 : i32
      %cr1 = arith.constant 1 : i32
      %ro1 = arith.muli %cr1, %dxi : i32
      %row1 = arith.addi %i0, %ro1 : i32
      %rowc1 = arith.minsi %row1, %mm1 : i32
      %rofs1 = arith.muli %rowc1, %ars : i32
      %arow1 = arith.addi %abase, %rofs1 : i32
      %co1 = arith.muli %cr1, %dyi : i32
      %col1 = arith.addi %j0, %co1 : i32
      %colc1 = arith.minsi %col1, %nm1 : i32
      %cofs1 = arith.muli %colc1, %bcs : i32
      %bcol1 = arith.addi %bbase, %cofs1 : i32
      %cr2 = arith.constant 2 : i32
      %ro2 = arith.muli %cr2, %dxi : i32
      %row2 = arith.addi %i0, %ro2 : i32
      %rowc2 = arith.minsi %row2, %mm1 : i32
      %rofs2 = arith.muli %rowc2, %ars : i32
      %arow2 = arith.addi %abase, %rofs2 : i32
      %co2 = arith.muli %cr2, %dyi : i32
      %col2 = arith.addi %j0, %co2 : i32
      %colc2 = arith.minsi %col2, %nm1 : i32
      %cofs2 = arith.muli %colc2, %bcs : i32
      %bcol2 = arith.addi %bbase, %cofs2 : i32
      %cr3 = arith.constant 3 : i32
      %ro3 = arith.muli %cr3, %dxi : i32
      %row3 = arith.addi %i0, %ro3 : i32
      %rowc3 = arith.minsi %row3, %mm1 : i32
      %rofs3 = arith.muli %rowc3, %ars : i32
      %arow3 = arith.addi %abase, %rofs3 : i32
      %co3 = arith.muli %cr3, %dyi : i32
      %col3 = arith.addi %j0, %co3 : i32
      %colc3 = arith.minsi %col3, %nm1 : i32
      %cofs3 = arith.muli %colc3, %bcs : i32
      %bcol3 = arith.addi %bbase, %cofs3 : i32
      %res:16 = scf.for %l = %c0 to %k step %c1 iter_args(%s00 = %zero, %s01 = %zero, %s02 = %zero, %s03 = %zero, %s10 = %zero, %s11 = %zero, %s12 = %zero, %s13 = %zero, %s20 = %zero, %s21 = %zero, %s22 = %zero, %s23 = %zero, %s30 = %zero, %s31 = %zero, %s32 = %zero, %s33 = %zero) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) : i32 {
        %la = arith.muli %l, %acs : i32
        %lb = arith.muli %l, %brs : i32
        %ai0 = arith.addi %arow0, %la : i32
        %aix0 = arith.index_cast %ai0 : i32 to index
        %av0_ = memref.load %a[%aix0] : memref<?xf32, 1>
        %av0 = arith.addf %av0_, %zero : f32
        %bi0 = arith.addi %bcol0, %lb : i32
        %bix0 = arith.index_cast %bi0 : i32 to index
        %bv0_ = memref.load %b[%bix0] : memref<?xf32, 1>
        %bv0 = arith.addf %bv0_, %zero : f32
        %ai1 = arith.addi %arow1, %la : i32
        %aix1 = arith.index_cast %ai1 : i32 to index
        %av1_ = memref.load %a[%aix1] : memref<?xf32, 1>
        %av1 = arith.addf %av1_, %zero : f32
        %bi1 = arith.addi %bcol1, %lb : i32
        %bix1 = arith.index_cast %bi1 : i32 to index
        %bv1_ = memref.load %b[%bix1] : memref<?xf32, 1>
        %bv1 = arith.addf %bv1_, %zero : f32
        %ai2 = arith.addi %arow2, %la : i32
        %aix2 = arith.index_cast %ai2 : i32 to index
        %av2_ = memref.load %a[%aix2] : memref<?xf32, 1>
        %av2 = arith.addf %av2_, %zero : f32
        %bi2 = arith.addi %bcol2, %lb : i32
        %bix2 = arith.index_cast %bi2 : i32 to index
        %bv2_ = memref.load %b[%bix2] : memref<?xf32, 1>
        %bv2 = arith.addf %bv2_, %zero : f32
        %ai3 = arith.addi %arow3, %la : i32
        %aix3 = arith.index_cast %ai3 : i32 to index
        %av3_ = memref.load %a[%aix3] : memref<?xf32, 1>
        %av3 = arith.addf %av3_, %zero : f32
        %bi3 = arith.addi %bcol3, %lb : i32
        %bix3 = arith.index_cast %bi3 : i32 to index
        %bv3_ = memref.load %b[%bix3] : memref<?xf32, 1>
        %bv3 = arith.addf %bv3_, %zero : f32
        %n00 = math.fma %av0, %bv0, %s00 : f32
        %n01 = math.fma %av0, %bv1, %s01 : f32
        %n02 = math.fma %av0, %bv2, %s02 : f32
        %n03 = math.fma %av0, %bv3, %s03 : f32
        %n10 = math.fma %av1, %bv0, %s10 : f32
        %n11 = math.fma %av1, %bv1, %s11 : f32
        %n12 = math.fma %av1, %bv2, %s12 : f32
        %n13 = math.fma %av1, %bv3, %s13 : f32
        %n20 = math.fma %av2, %bv0, %s20 : f32
        %n21 = math.fma %av2, %bv1, %s21 : f32
        %n22 = math.fma %av2, %bv2, %s22 : f32
        %n23 = math.fma %av2, %bv3, %s23 : f32
        %n30 = math.fma %av3, %bv0, %s30 : f32
        %n31 = math.fma %av3, %bv1, %s31 : f32
        %n32 = math.fma %av3, %bv2, %s32 : f32
        %n33 = math.fma %av3, %bv3, %s33 : f32
        scf.yield %n00, %n01, %n02, %n03, %n10, %n11, %n12, %n13, %n20, %n21, %n22, %n23, %n30, %n31, %n32, %n33 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
      }
      %bzero = arith.cmpf oeq, %beta, %zero : f32
      %inr00 = arith.cmpi slt, %row0, %m : i32
      %inc00 = arith.cmpi slt, %col0, %n : i32
      %in00 = arith.andi %inr00, %inc00 : i1
      scf.if %in00 {
        %cl00 = arith.muli %col0, %ldc : i32
        %cc00 = arith.addi %cbase, %cl00 : i32
        %ci00 = arith.addi %cc00, %row0 : i32
        %cix00 = arith.index_cast %ci00 : i32 to index
        %old00 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_00 = memref.load %c[%cix00] : memref<?xf32, 1>
          %oe00 = arith.addf %o_00, %zero : f32
          %bo00 = arith.mulf %beta, %oe00 : f32
          scf.yield %bo00 : f32
        }
        %v00 = math.fma %alpha, %res#0, %old00 : f32
        %vt00 = arith.addf %v00, %zero : f32
        memref.store %vt00, %c[%cix00] : memref<?xf32, 1>
      }
      %inr01 = arith.cmpi slt, %row0, %m : i32
      %inc01 = arith.cmpi slt, %col1, %n : i32
      %in01 = arith.andi %inr01, %inc01 : i1
      scf.if %in01 {
        %cl01 = arith.muli %col1, %ldc : i32
        %cc01 = arith.addi %cbase, %cl01 : i32
        %ci01 = arith.addi %cc01, %row0 : i32
        %cix01 = arith.index_cast %ci01 : i32 to index
        %old01 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_01 = memref.load %c[%cix01] : memref<?xf32, 1>
          %oe01 = arith.addf %o_01, %zero : f32
          %bo01 = arith.mulf %beta, %oe01 : f32
          scf.yield %bo01 : f32
        }
        %v01 = math.fma %alpha, %res#1, %old01 : f32
        %vt01 = arith.addf %v01, %zero : f32
        memref.store %vt01, %c[%cix01] : memref<?xf32, 1>
      }
      %inr02 = arith.cmpi slt, %row0, %m : i32
      %inc02 = arith.cmpi slt, %col2, %n : i32
      %in02 = arith.andi %inr02, %inc02 : i1
      scf.if %in02 {
        %cl02 = arith.muli %col2, %ldc : i32
        %cc02 = arith.addi %cbase, %cl02 : i32
        %ci02 = arith.addi %cc02, %row0 : i32
        %cix02 = arith.index_cast %ci02 : i32 to index
        %old02 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_02 = memref.load %c[%cix02] : memref<?xf32, 1>
          %oe02 = arith.addf %o_02, %zero : f32
          %bo02 = arith.mulf %beta, %oe02 : f32
          scf.yield %bo02 : f32
        }
        %v02 = math.fma %alpha, %res#2, %old02 : f32
        %vt02 = arith.addf %v02, %zero : f32
        memref.store %vt02, %c[%cix02] : memref<?xf32, 1>
      }
      %inr03 = arith.cmpi slt, %row0, %m : i32
      %inc03 = arith.cmpi slt, %col3, %n : i32
      %in03 = arith.andi %inr03, %inc03 : i1
      scf.if %in03 {
        %cl03 = arith.muli %col3, %ldc : i32
        %cc03 = arith.addi %cbase, %cl03 : i32
        %ci03 = arith.addi %cc03, %row0 : i32
        %cix03 = arith.index_cast %ci03 : i32 to index
        %old03 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_03 = memref.load %c[%cix03] : memref<?xf32, 1>
          %oe03 = arith.addf %o_03, %zero : f32
          %bo03 = arith.mulf %beta, %oe03 : f32
          scf.yield %bo03 : f32
        }
        %v03 = math.fma %alpha, %res#3, %old03 : f32
        %vt03 = arith.addf %v03, %zero : f32
        memref.store %vt03, %c[%cix03] : memref<?xf32, 1>
      }
      %inr10 = arith.cmpi slt, %row1, %m : i32
      %inc10 = arith.cmpi slt, %col0, %n : i32
      %in10 = arith.andi %inr10, %inc10 : i1
      scf.if %in10 {
        %cl10 = arith.muli %col0, %ldc : i32
        %cc10 = arith.addi %cbase, %cl10 : i32
        %ci10 = arith.addi %cc10, %row1 : i32
        %cix10 = arith.index_cast %ci10 : i32 to index
        %old10 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_10 = memref.load %c[%cix10] : memref<?xf32, 1>
          %oe10 = arith.addf %o_10, %zero : f32
          %bo10 = arith.mulf %beta, %oe10 : f32
          scf.yield %bo10 : f32
        }
        %v10 = math.fma %alpha, %res#4, %old10 : f32
        %vt10 = arith.addf %v10, %zero : f32
        memref.store %vt10, %c[%cix10] : memref<?xf32, 1>
      }
      %inr11 = arith.cmpi slt, %row1, %m : i32
      %inc11 = arith.cmpi slt, %col1, %n : i32
      %in11 = arith.andi %inr11, %inc11 : i1
      scf.if %in11 {
        %cl11 = arith.muli %col1, %ldc : i32
        %cc11 = arith.addi %cbase, %cl11 : i32
        %ci11 = arith.addi %cc11, %row1 : i32
        %cix11 = arith.index_cast %ci11 : i32 to index
        %old11 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_11 = memref.load %c[%cix11] : memref<?xf32, 1>
          %oe11 = arith.addf %o_11, %zero : f32
          %bo11 = arith.mulf %beta, %oe11 : f32
          scf.yield %bo11 : f32
        }
        %v11 = math.fma %alpha, %res#5, %old11 : f32
        %vt11 = arith.addf %v11, %zero : f32
        memref.store %vt11, %c[%cix11] : memref<?xf32, 1>
      }
      %inr12 = arith.cmpi slt, %row1, %m : i32
      %inc12 = arith.cmpi slt, %col2, %n : i32
      %in12 = arith.andi %inr12, %inc12 : i1
      scf.if %in12 {
        %cl12 = arith.muli %col2, %ldc : i32
        %cc12 = arith.addi %cbase, %cl12 : i32
        %ci12 = arith.addi %cc12, %row1 : i32
        %cix12 = arith.index_cast %ci12 : i32 to index
        %old12 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_12 = memref.load %c[%cix12] : memref<?xf32, 1>
          %oe12 = arith.addf %o_12, %zero : f32
          %bo12 = arith.mulf %beta, %oe12 : f32
          scf.yield %bo12 : f32
        }
        %v12 = math.fma %alpha, %res#6, %old12 : f32
        %vt12 = arith.addf %v12, %zero : f32
        memref.store %vt12, %c[%cix12] : memref<?xf32, 1>
      }
      %inr13 = arith.cmpi slt, %row1, %m : i32
      %inc13 = arith.cmpi slt, %col3, %n : i32
      %in13 = arith.andi %inr13, %inc13 : i1
      scf.if %in13 {
        %cl13 = arith.muli %col3, %ldc : i32
        %cc13 = arith.addi %cbase, %cl13 : i32
        %ci13 = arith.addi %cc13, %row1 : i32
        %cix13 = arith.index_cast %ci13 : i32 to index
        %old13 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_13 = memref.load %c[%cix13] : memref<?xf32, 1>
          %oe13 = arith.addf %o_13, %zero : f32
          %bo13 = arith.mulf %beta, %oe13 : f32
          scf.yield %bo13 : f32
        }
        %v13 = math.fma %alpha, %res#7, %old13 : f32
        %vt13 = arith.addf %v13, %zero : f32
        memref.store %vt13, %c[%cix13] : memref<?xf32, 1>
      }
      %inr20 = arith.cmpi slt, %row2, %m : i32
      %inc20 = arith.cmpi slt, %col0, %n : i32
      %in20 = arith.andi %inr20, %inc20 : i1
      scf.if %in20 {
        %cl20 = arith.muli %col0, %ldc : i32
        %cc20 = arith.addi %cbase, %cl20 : i32
        %ci20 = arith.addi %cc20, %row2 : i32
        %cix20 = arith.index_cast %ci20 : i32 to index
        %old20 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_20 = memref.load %c[%cix20] : memref<?xf32, 1>
          %oe20 = arith.addf %o_20, %zero : f32
          %bo20 = arith.mulf %beta, %oe20 : f32
          scf.yield %bo20 : f32
        }
        %v20 = math.fma %alpha, %res#8, %old20 : f32
        %vt20 = arith.addf %v20, %zero : f32
        memref.store %vt20, %c[%cix20] : memref<?xf32, 1>
      }
      %inr21 = arith.cmpi slt, %row2, %m : i32
      %inc21 = arith.cmpi slt, %col1, %n : i32
      %in21 = arith.andi %inr21, %inc21 : i1
      scf.if %in21 {
        %cl21 = arith.muli %col1, %ldc : i32
        %cc21 = arith.addi %cbase, %cl21 : i32
        %ci21 = arith.addi %cc21, %row2 : i32
        %cix21 = arith.index_cast %ci21 : i32 to index
        %old21 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_21 = memref.load %c[%cix21] : memref<?xf32, 1>
          %oe21 = arith.addf %o_21, %zero : f32
          %bo21 = arith.mulf %beta, %oe21 : f32
          scf.yield %bo21 : f32
        }
        %v21 = math.fma %alpha, %res#9, %old21 : f32
        %vt21 = arith.addf %v21, %zero : f32
        memref.store %vt21, %c[%cix21] : memref<?xf32, 1>
      }
      %inr22 = arith.cmpi slt, %row2, %m : i32
      %inc22 = arith.cmpi slt, %col2, %n : i32
      %in22 = arith.andi %inr22, %inc22 : i1
      scf.if %in22 {
        %cl22 = arith.muli %col2, %ldc : i32
        %cc22 = arith.addi %cbase, %cl22 : i32
        %ci22 = arith.addi %cc22, %row2 : i32
        %cix22 = arith.index_cast %ci22 : i32 to index
        %old22 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_22 = memref.load %c[%cix22] : memref<?xf32, 1>
          %oe22 = arith.addf %o_22, %zero : f32
          %bo22 = arith.mulf %beta, %oe22 : f32
          scf.yield %bo22 : f32
        }
        %v22 = math.fma %alpha, %res#10, %old22 : f32
        %vt22 = arith.addf %v22, %zero : f32
        memref.store %vt22, %c[%cix22] : memref<?xf32, 1>
      }
      %inr23 = arith.cmpi slt, %row2, %m : i32
      %inc23 = arith.cmpi slt, %col3, %n : i32
      %in23 = arith.andi %inr23, %inc23 : i1
      scf.if %in23 {
        %cl23 = arith.muli %col3, %ldc : i32
        %cc23 = arith.addi %cbase, %cl23 : i32
        %ci23 = arith.addi %cc23, %row2 : i32
        %cix23 = arith.index_cast %ci23 : i32 to index
        %old23 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_23 = memref.load %c[%cix23] : memref<?xf32, 1>
          %oe23 = arith.addf %o_23, %zero : f32
          %bo23 = arith.mulf %beta, %oe23 : f32
          scf.yield %bo23 : f32
        }
        %v23 = math.fma %alpha, %res#11, %old23 : f32
        %vt23 = arith.addf %v23, %zero : f32
        memref.store %vt23, %c[%cix23] : memref<?xf32, 1>
      }
      %inr30 = arith.cmpi slt, %row3, %m : i32
      %inc30 = arith.cmpi slt, %col0, %n : i32
      %in30 = arith.andi %inr30, %inc30 : i1
      scf.if %in30 {
        %cl30 = arith.muli %col0, %ldc : i32
        %cc30 = arith.addi %cbase, %cl30 : i32
        %ci30 = arith.addi %cc30, %row3 : i32
        %cix30 = arith.index_cast %ci30 : i32 to index
        %old30 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_30 = memref.load %c[%cix30] : memref<?xf32, 1>
          %oe30 = arith.addf %o_30, %zero : f32
          %bo30 = arith.mulf %beta, %oe30 : f32
          scf.yield %bo30 : f32
        }
        %v30 = math.fma %alpha, %res#12, %old30 : f32
        %vt30 = arith.addf %v30, %zero : f32
        memref.store %vt30, %c[%cix30] : memref<?xf32, 1>
      }
      %inr31 = arith.cmpi slt, %row3, %m : i32
      %inc31 = arith.cmpi slt, %col1, %n : i32
      %in31 = arith.andi %inr31, %inc31 : i1
      scf.if %in31 {
        %cl31 = arith.muli %col1, %ldc : i32
        %cc31 = arith.addi %cbase, %cl31 : i32
        %ci31 = arith.addi %cc31, %row3 : i32
        %cix31 = arith.index_cast %ci31 : i32 to index
        %old31 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_31 = memref.load %c[%cix31] : memref<?xf32, 1>
          %oe31 = arith.addf %o_31, %zero : f32
          %bo31 = arith.mulf %beta, %oe31 : f32
          scf.yield %bo31 : f32
        }
        %v31 = math.fma %alpha, %res#13, %old31 : f32
        %vt31 = arith.addf %v31, %zero : f32
        memref.store %vt31, %c[%cix31] : memref<?xf32, 1>
      }
      %inr32 = arith.cmpi slt, %row3, %m : i32
      %inc32 = arith.cmpi slt, %col2, %n : i32
      %in32 = arith.andi %inr32, %inc32 : i1
      scf.if %in32 {
        %cl32 = arith.muli %col2, %ldc : i32
        %cc32 = arith.addi %cbase, %cl32 : i32
        %ci32 = arith.addi %cc32, %row3 : i32
        %cix32 = arith.index_cast %ci32 : i32 to index
        %old32 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_32 = memref.load %c[%cix32] : memref<?xf32, 1>
          %oe32 = arith.addf %o_32, %zero : f32
          %bo32 = arith.mulf %beta, %oe32 : f32
          scf.yield %bo32 : f32
        }
        %v32 = math.fma %alpha, %res#14, %old32 : f32
        %vt32 = arith.addf %v32, %zero : f32
        memref.store %vt32, %c[%cix32] : memref<?xf32, 1>
      }
      %inr33 = arith.cmpi slt, %row3, %m : i32
      %inc33 = arith.cmpi slt, %col3, %n : i32
      %in33 = arith.andi %inr33, %inc33 : i1
      scf.if %in33 {
        %cl33 = arith.muli %col3, %ldc : i32
        %cc33 = arith.addi %cbase, %cl33 : i32
        %ci33 = arith.addi %cc33, %row3 : i32
        %cix33 = arith.index_cast %ci33 : i32 to index
        %old33 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_33 = memref.load %c[%cix33] : memref<?xf32, 1>
          %oe33 = arith.addf %o_33, %zero : f32
          %bo33 = arith.mulf %beta, %oe33 : f32
          scf.yield %bo33 : f32
        }
        %v33 = math.fma %alpha, %res#15, %old33 : f32
        %vt33 = arith.addf %v33, %zero : f32
        memref.store %vt33, %c[%cix33] : memref<?xf32, 1>
      }
      gpu.return
    }
    gpu.func @daxpy(%n: i32, %alpha: f64, %x: memref<?xf64, 1>, %xo: i32, %xi: i32, %y: memref<?xf64, 1>, %yo: i32, %yi: i32) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %b0 = arith.muli %bid, %bdim : index
      %gi = arith.addi %b0, %tid : index
      %gn = arith.muli %gdim, %bdim : index
      %i = arith.index_cast %gi : index to i32
      %stride = arith.index_cast %gn : index to i32
      scf.for %k = %i to %n step %stride : i32 {
      %ix_m = arith.muli %k, %xi : i32
      %ix_a = arith.addi %xo, %ix_m : i32
      %ix = arith.index_cast %ix_a : i32 to index
      %iy_m = arith.muli %k, %yi : i32
      %iy_a = arith.addi %yo, %iy_m : i32
      %iy = arith.index_cast %iy_a : i32 to index
        %xv = memref.load %x[%ix] : memref<?xf64, 1>
        %yv = memref.load %y[%iy] : memref<?xf64, 1>
        %r = math.fma %alpha, %xv, %yv : f64
        memref.store %r, %y[%iy] : memref<?xf64, 1>
      }
      gpu.return
    }
    gpu.func @dscal(%n: i32, %alpha: f64, %x: memref<?xf64, 1>, %xo: i32, %xi: i32) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %b0 = arith.muli %bid, %bdim : index
      %gi = arith.addi %b0, %tid : index
      %gn = arith.muli %gdim, %bdim : index
      %i = arith.index_cast %gi : index to i32
      %stride = arith.index_cast %gn : index to i32
      scf.for %k = %i to %n step %stride : i32 {
      %ix_m = arith.muli %k, %xi : i32
      %ix_a = arith.addi %xo, %ix_m : i32
      %ix = arith.index_cast %ix_a : i32 to index
        %xv = memref.load %x[%ix] : memref<?xf64, 1>
        %r = arith.mulf %alpha, %xv : f64
        memref.store %r, %x[%ix] : memref<?xf64, 1>
      }
      gpu.return
    }
    gpu.func @dcopy(%n: i32, %mode: i32, %x: memref<?xf64, 1>, %xo: i32, %xi: i32, %y: memref<?xf64, 1>, %yo: i32, %yi: i32) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %b0 = arith.muli %bid, %bdim : index
      %gi = arith.addi %b0, %tid : index
      %gn = arith.muli %gdim, %bdim : index
      %i = arith.index_cast %gi : index to i32
      %stride = arith.index_cast %gn : index to i32
      %c0 = arith.constant 0 : i32
      %sw = arith.cmpi ne, %mode, %c0 : i32
      scf.for %k = %i to %n step %stride : i32 {
      %ix_m = arith.muli %k, %xi : i32
      %ix_a = arith.addi %xo, %ix_m : i32
      %ix = arith.index_cast %ix_a : i32 to index
      %iy_m = arith.muli %k, %yi : i32
      %iy_a = arith.addi %yo, %iy_m : i32
      %iy = arith.index_cast %iy_a : i32 to index
        %xv = memref.load %x[%ix] : memref<?xf64, 1>
        scf.if %sw {
          %yv = memref.load %y[%iy] : memref<?xf64, 1>
          memref.store %yv, %x[%ix] : memref<?xf64, 1>
        }
        memref.store %xv, %y[%iy] : memref<?xf64, 1>
      }
      gpu.return
    }
    gpu.func @dreduce(%n: i32, %op: i32, %x: memref<?xf64, 1>, %xo: i32, %xi: i32, %y: memref<?xf64, 1>, %yo: i32, %yi: i32, %out: memref<?xf64, 1>) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %b0 = arith.muli %bid, %bdim : index
      %gi = arith.addi %b0, %tid : index
      %gn = arith.muli %gdim, %bdim : index
      %i = arith.index_cast %gi : index to i32
      %stride = arith.index_cast %gn : index to i32
      %zero = arith.constant 0.0 : f64
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %acc = scf.for %k = %i to %n step %stride iter_args(%a = %zero) -> (f64) : i32 {
      %ix_m = arith.muli %k, %xi : i32
      %ix_a = arith.addi %xo, %ix_m : i32
      %ix = arith.index_cast %ix_a : i32 to index
        %xv = memref.load %x[%ix] : memref<?xf64, 1>
        %isdot = arith.cmpi eq, %op, %c0 : i32
        %v = scf.if %isdot -> (f64) {
      %iy_m = arith.muli %k, %yi : i32
      %iy_a = arith.addi %yo, %iy_m : i32
      %iy = arith.index_cast %iy_a : i32 to index
          %yv = memref.load %y[%iy] : memref<?xf64, 1>
          %m = arith.mulf %xv, %yv : f64
          scf.yield %m : f64
        } else {
          %issq = arith.cmpi eq, %op, %c1 : i32
          %sq = arith.mulf %xv, %xv : f64
          %isabs = arith.cmpi eq, %op, %c2 : i32
          %ab = math.absf %xv : f64
          %w0 = arith.select %isabs, %ab, %xv : f64
          %w = arith.select %issq, %sq, %w0 : f64
          scf.yield %w : f64
        }
        %na = arith.addf %a, %v : f64
        scf.yield %na : f64
      }
      %s = gpu.subgroup_reduce add %acc : (f64) -> f64
      %lane = gpu.lane_id
      %c0i = arith.constant 0 : index
      %first = arith.cmpi eq, %lane, %c0i : index
      scf.if %first {
        memref.store %s, %out[%bid] : memref<?xf64, 1>
      }
      gpu.return
    }
    gpu.func @damax(%n: i32, %op: i32, %x: memref<?xf64, 1>, %xo: i32, %xi: i32, %y: memref<?xf64, 1>, %out: memref<?xf64, 1>, %outi: memref<?xf64, 1>) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %b0 = arith.muli %bid, %bdim : index
      %gi = arith.addi %b0, %tid : index
      %gn = arith.muli %gdim, %bdim : index
      %i = arith.index_cast %gi : index to i32
      %stride = arith.index_cast %gn : index to i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %opm = arith.remui %op, %c2 : i32
      %ismin = arith.cmpi eq, %opm, %c1 : i32
      %final = arith.cmpi uge, %op, %c2 : i32
      %inf = arith.constant 0x7F800000 : i32
      %pinf = arith.bitcast %inf : i32 to f32
      %pinft = arith.extf %pinf : f32 to f64
      %neg1 = arith.constant -1.0 : f64
      %start = arith.select %ismin, %pinft, %neg1 : f64
      %big = arith.constant 2147483647 : i32
      %r:2 = scf.for %k = %i to %n step %stride iter_args(%bv = %start, %bi = %big) -> (f64, i32) : i32 {
      %ix_m = arith.muli %k, %xi : i32
      %ix_a = arith.addi %xo, %ix_m : i32
      %ix = arith.index_cast %ix_a : i32 to index
        %xv = memref.load %x[%ix] : memref<?xf64, 1>
        %av0 = math.absf %xv : f64
        %av = arith.select %final, %xv, %av0 : f64
        %ki = arith.index_cast %k : i32 to index
        %pi = scf.if %final -> (i32) {
          %yv = memref.load %y[%ki] : memref<?xf64, 1>
          %yi = arith.fptosi %yv : f64 to i32
          scf.yield %yi : i32
        } else {
          %k1 = arith.addi %k, %c1 : i32
          scf.yield %k1 : i32
        }
        %gt = arith.cmpf ogt, %av, %bv : f64
        %lt = arith.cmpf olt, %av, %bv : f64
        %better = arith.select %ismin, %lt, %gt : i1
        %nv = arith.select %better, %av, %bv : f64
        %ni = arith.select %better, %pi, %bi : i32
        scf.yield %nv, %ni : f64, i32
      }
      // the best value of the SIMD group, then the smallest index that has it
      %mx = gpu.subgroup_reduce maxnumf %r#0 : (f64) -> f64
      %mn = gpu.subgroup_reduce minnumf %r#0 : (f64) -> f64
      %bestv = arith.select %ismin, %mn, %mx : f64
      %has = arith.cmpf oeq, %r#0, %bestv : f64
      %cand = arith.select %has, %r#1, %big : i32
      %besti = gpu.subgroup_reduce minsi %cand : (i32) -> i32
      %lane = gpu.lane_id
      %c0i = arith.constant 0 : index
      %first = arith.cmpi eq, %lane, %c0i : index
      scf.if %first {
        memref.store %bestv, %out[%bid] : memref<?xf64, 1>
        %bif = arith.sitofp %besti : i32 to f64
        memref.store %bif, %outi[%bid] : memref<?xf64, 1>
      }
      gpu.return
    }
    gpu.func @dgemv(%m: i32, %n: i32, %alpha: f64, %a: memref<?xf64, 1>, %ao: i32, %ars: i32, %acs: i32, %x: memref<?xf64, 1>, %xo: i32, %xi: i32, %beta: f64, %y: memref<?xf64, 1>, %yo: i32, %yi: i32) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %b0 = arith.muli %bid, %bdim : index
      %gi = arith.addi %b0, %tid : index
      %gn = arith.muli %gdim, %bdim : index
      %i = arith.index_cast %gi : index to i32
      %stride = arith.index_cast %gn : index to i32
      %zero = arith.constant 0.0 : f64
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      scf.for %r = %i to %m step %stride : i32 {
        %rr = arith.muli %r, %ars : i32
        %rowbase = arith.addi %ao, %rr : i32
        %acc = scf.for %j = %c0 to %n step %c1 iter_args(%s = %zero) -> (f64) : i32 {
          %ia_m = arith.muli %j, %acs : i32
          %ia_a = arith.addi %rowbase, %ia_m : i32
          %ia = arith.index_cast %ia_a : i32 to index
          %ix_m = arith.muli %j, %xi : i32
          %ix_a = arith.addi %xo, %ix_m : i32
          %ix = arith.index_cast %ix_a : i32 to index
          %av = memref.load %a[%ia] : memref<?xf64, 1>
          %xv = memref.load %x[%ix] : memref<?xf64, 1>
          %ns = math.fma %av, %xv, %s : f64
          scf.yield %ns : f64
        }
      %iy_m = arith.muli %r, %yi : i32
      %iy_a = arith.addi %yo, %iy_m : i32
      %iy = arith.index_cast %iy_a : i32 to index
        %bz = arith.cmpf oeq, %beta, %zero : f64
        %yv = scf.if %bz -> (f64) {
          scf.yield %zero : f64
        } else {
          %y0 = memref.load %y[%iy] : memref<?xf64, 1>
          %by = arith.mulf %beta, %y0 : f64
          scf.yield %by : f64
        }
        %res = math.fma %alpha, %acc, %yv : f64
        memref.store %res, %y[%iy] : memref<?xf64, 1>
      }
      gpu.return
    }
    gpu.func @dgemm(%m: i32, %n: i32, %k: i32, %alpha: f64, %a: memref<?xf64, 1>, %ao: i32, %ars: i32, %acs: i32, %sa: i32, %b: memref<?xf64, 1>, %bo: i32, %brs: i32, %bcs: i32, %sb: i32, %beta: f64, %c: memref<?xf64, 1>, %co: i32, %ldc: i32, %sc: i32) kernel {
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
      %zero = arith.constant 0.0 : f64
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
      %cbase = arith.addi %co, %zsc : i32
      %mm1 = arith.subi %m, %c1 : i32
      %nm1 = arith.subi %n, %c1 : i32
      %cr0 = arith.constant 0 : i32
      %ro0 = arith.muli %cr0, %dxi : i32
      %row0 = arith.addi %i0, %ro0 : i32
      %rowc0 = arith.minsi %row0, %mm1 : i32
      %rofs0 = arith.muli %rowc0, %ars : i32
      %arow0 = arith.addi %abase, %rofs0 : i32
      %co0 = arith.muli %cr0, %dyi : i32
      %col0 = arith.addi %j0, %co0 : i32
      %colc0 = arith.minsi %col0, %nm1 : i32
      %cofs0 = arith.muli %colc0, %bcs : i32
      %bcol0 = arith.addi %bbase, %cofs0 : i32
      %cr1 = arith.constant 1 : i32
      %ro1 = arith.muli %cr1, %dxi : i32
      %row1 = arith.addi %i0, %ro1 : i32
      %rowc1 = arith.minsi %row1, %mm1 : i32
      %rofs1 = arith.muli %rowc1, %ars : i32
      %arow1 = arith.addi %abase, %rofs1 : i32
      %co1 = arith.muli %cr1, %dyi : i32
      %col1 = arith.addi %j0, %co1 : i32
      %colc1 = arith.minsi %col1, %nm1 : i32
      %cofs1 = arith.muli %colc1, %bcs : i32
      %bcol1 = arith.addi %bbase, %cofs1 : i32
      %cr2 = arith.constant 2 : i32
      %ro2 = arith.muli %cr2, %dxi : i32
      %row2 = arith.addi %i0, %ro2 : i32
      %rowc2 = arith.minsi %row2, %mm1 : i32
      %rofs2 = arith.muli %rowc2, %ars : i32
      %arow2 = arith.addi %abase, %rofs2 : i32
      %co2 = arith.muli %cr2, %dyi : i32
      %col2 = arith.addi %j0, %co2 : i32
      %colc2 = arith.minsi %col2, %nm1 : i32
      %cofs2 = arith.muli %colc2, %bcs : i32
      %bcol2 = arith.addi %bbase, %cofs2 : i32
      %cr3 = arith.constant 3 : i32
      %ro3 = arith.muli %cr3, %dxi : i32
      %row3 = arith.addi %i0, %ro3 : i32
      %rowc3 = arith.minsi %row3, %mm1 : i32
      %rofs3 = arith.muli %rowc3, %ars : i32
      %arow3 = arith.addi %abase, %rofs3 : i32
      %co3 = arith.muli %cr3, %dyi : i32
      %col3 = arith.addi %j0, %co3 : i32
      %colc3 = arith.minsi %col3, %nm1 : i32
      %cofs3 = arith.muli %colc3, %bcs : i32
      %bcol3 = arith.addi %bbase, %cofs3 : i32
      %res:16 = scf.for %l = %c0 to %k step %c1 iter_args(%s00 = %zero, %s01 = %zero, %s02 = %zero, %s03 = %zero, %s10 = %zero, %s11 = %zero, %s12 = %zero, %s13 = %zero, %s20 = %zero, %s21 = %zero, %s22 = %zero, %s23 = %zero, %s30 = %zero, %s31 = %zero, %s32 = %zero, %s33 = %zero) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) : i32 {
        %la = arith.muli %l, %acs : i32
        %lb = arith.muli %l, %brs : i32
        %ai0 = arith.addi %arow0, %la : i32
        %aix0 = arith.index_cast %ai0 : i32 to index
        %av0_ = memref.load %a[%aix0] : memref<?xf64, 1>
        %av0 = arith.addf %av0_, %zero : f64
        %bi0 = arith.addi %bcol0, %lb : i32
        %bix0 = arith.index_cast %bi0 : i32 to index
        %bv0_ = memref.load %b[%bix0] : memref<?xf64, 1>
        %bv0 = arith.addf %bv0_, %zero : f64
        %ai1 = arith.addi %arow1, %la : i32
        %aix1 = arith.index_cast %ai1 : i32 to index
        %av1_ = memref.load %a[%aix1] : memref<?xf64, 1>
        %av1 = arith.addf %av1_, %zero : f64
        %bi1 = arith.addi %bcol1, %lb : i32
        %bix1 = arith.index_cast %bi1 : i32 to index
        %bv1_ = memref.load %b[%bix1] : memref<?xf64, 1>
        %bv1 = arith.addf %bv1_, %zero : f64
        %ai2 = arith.addi %arow2, %la : i32
        %aix2 = arith.index_cast %ai2 : i32 to index
        %av2_ = memref.load %a[%aix2] : memref<?xf64, 1>
        %av2 = arith.addf %av2_, %zero : f64
        %bi2 = arith.addi %bcol2, %lb : i32
        %bix2 = arith.index_cast %bi2 : i32 to index
        %bv2_ = memref.load %b[%bix2] : memref<?xf64, 1>
        %bv2 = arith.addf %bv2_, %zero : f64
        %ai3 = arith.addi %arow3, %la : i32
        %aix3 = arith.index_cast %ai3 : i32 to index
        %av3_ = memref.load %a[%aix3] : memref<?xf64, 1>
        %av3 = arith.addf %av3_, %zero : f64
        %bi3 = arith.addi %bcol3, %lb : i32
        %bix3 = arith.index_cast %bi3 : i32 to index
        %bv3_ = memref.load %b[%bix3] : memref<?xf64, 1>
        %bv3 = arith.addf %bv3_, %zero : f64
        %n00 = math.fma %av0, %bv0, %s00 : f64
        %n01 = math.fma %av0, %bv1, %s01 : f64
        %n02 = math.fma %av0, %bv2, %s02 : f64
        %n03 = math.fma %av0, %bv3, %s03 : f64
        %n10 = math.fma %av1, %bv0, %s10 : f64
        %n11 = math.fma %av1, %bv1, %s11 : f64
        %n12 = math.fma %av1, %bv2, %s12 : f64
        %n13 = math.fma %av1, %bv3, %s13 : f64
        %n20 = math.fma %av2, %bv0, %s20 : f64
        %n21 = math.fma %av2, %bv1, %s21 : f64
        %n22 = math.fma %av2, %bv2, %s22 : f64
        %n23 = math.fma %av2, %bv3, %s23 : f64
        %n30 = math.fma %av3, %bv0, %s30 : f64
        %n31 = math.fma %av3, %bv1, %s31 : f64
        %n32 = math.fma %av3, %bv2, %s32 : f64
        %n33 = math.fma %av3, %bv3, %s33 : f64
        scf.yield %n00, %n01, %n02, %n03, %n10, %n11, %n12, %n13, %n20, %n21, %n22, %n23, %n30, %n31, %n32, %n33 : f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64
      }
      %bzero = arith.cmpf oeq, %beta, %zero : f64
      %inr00 = arith.cmpi slt, %row0, %m : i32
      %inc00 = arith.cmpi slt, %col0, %n : i32
      %in00 = arith.andi %inr00, %inc00 : i1
      scf.if %in00 {
        %cl00 = arith.muli %col0, %ldc : i32
        %cc00 = arith.addi %cbase, %cl00 : i32
        %ci00 = arith.addi %cc00, %row0 : i32
        %cix00 = arith.index_cast %ci00 : i32 to index
        %old00 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_00 = memref.load %c[%cix00] : memref<?xf64, 1>
          %oe00 = arith.addf %o_00, %zero : f64
          %bo00 = arith.mulf %beta, %oe00 : f64
          scf.yield %bo00 : f64
        }
        %v00 = math.fma %alpha, %res#0, %old00 : f64
        %vt00 = arith.addf %v00, %zero : f64
        memref.store %vt00, %c[%cix00] : memref<?xf64, 1>
      }
      %inr01 = arith.cmpi slt, %row0, %m : i32
      %inc01 = arith.cmpi slt, %col1, %n : i32
      %in01 = arith.andi %inr01, %inc01 : i1
      scf.if %in01 {
        %cl01 = arith.muli %col1, %ldc : i32
        %cc01 = arith.addi %cbase, %cl01 : i32
        %ci01 = arith.addi %cc01, %row0 : i32
        %cix01 = arith.index_cast %ci01 : i32 to index
        %old01 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_01 = memref.load %c[%cix01] : memref<?xf64, 1>
          %oe01 = arith.addf %o_01, %zero : f64
          %bo01 = arith.mulf %beta, %oe01 : f64
          scf.yield %bo01 : f64
        }
        %v01 = math.fma %alpha, %res#1, %old01 : f64
        %vt01 = arith.addf %v01, %zero : f64
        memref.store %vt01, %c[%cix01] : memref<?xf64, 1>
      }
      %inr02 = arith.cmpi slt, %row0, %m : i32
      %inc02 = arith.cmpi slt, %col2, %n : i32
      %in02 = arith.andi %inr02, %inc02 : i1
      scf.if %in02 {
        %cl02 = arith.muli %col2, %ldc : i32
        %cc02 = arith.addi %cbase, %cl02 : i32
        %ci02 = arith.addi %cc02, %row0 : i32
        %cix02 = arith.index_cast %ci02 : i32 to index
        %old02 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_02 = memref.load %c[%cix02] : memref<?xf64, 1>
          %oe02 = arith.addf %o_02, %zero : f64
          %bo02 = arith.mulf %beta, %oe02 : f64
          scf.yield %bo02 : f64
        }
        %v02 = math.fma %alpha, %res#2, %old02 : f64
        %vt02 = arith.addf %v02, %zero : f64
        memref.store %vt02, %c[%cix02] : memref<?xf64, 1>
      }
      %inr03 = arith.cmpi slt, %row0, %m : i32
      %inc03 = arith.cmpi slt, %col3, %n : i32
      %in03 = arith.andi %inr03, %inc03 : i1
      scf.if %in03 {
        %cl03 = arith.muli %col3, %ldc : i32
        %cc03 = arith.addi %cbase, %cl03 : i32
        %ci03 = arith.addi %cc03, %row0 : i32
        %cix03 = arith.index_cast %ci03 : i32 to index
        %old03 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_03 = memref.load %c[%cix03] : memref<?xf64, 1>
          %oe03 = arith.addf %o_03, %zero : f64
          %bo03 = arith.mulf %beta, %oe03 : f64
          scf.yield %bo03 : f64
        }
        %v03 = math.fma %alpha, %res#3, %old03 : f64
        %vt03 = arith.addf %v03, %zero : f64
        memref.store %vt03, %c[%cix03] : memref<?xf64, 1>
      }
      %inr10 = arith.cmpi slt, %row1, %m : i32
      %inc10 = arith.cmpi slt, %col0, %n : i32
      %in10 = arith.andi %inr10, %inc10 : i1
      scf.if %in10 {
        %cl10 = arith.muli %col0, %ldc : i32
        %cc10 = arith.addi %cbase, %cl10 : i32
        %ci10 = arith.addi %cc10, %row1 : i32
        %cix10 = arith.index_cast %ci10 : i32 to index
        %old10 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_10 = memref.load %c[%cix10] : memref<?xf64, 1>
          %oe10 = arith.addf %o_10, %zero : f64
          %bo10 = arith.mulf %beta, %oe10 : f64
          scf.yield %bo10 : f64
        }
        %v10 = math.fma %alpha, %res#4, %old10 : f64
        %vt10 = arith.addf %v10, %zero : f64
        memref.store %vt10, %c[%cix10] : memref<?xf64, 1>
      }
      %inr11 = arith.cmpi slt, %row1, %m : i32
      %inc11 = arith.cmpi slt, %col1, %n : i32
      %in11 = arith.andi %inr11, %inc11 : i1
      scf.if %in11 {
        %cl11 = arith.muli %col1, %ldc : i32
        %cc11 = arith.addi %cbase, %cl11 : i32
        %ci11 = arith.addi %cc11, %row1 : i32
        %cix11 = arith.index_cast %ci11 : i32 to index
        %old11 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_11 = memref.load %c[%cix11] : memref<?xf64, 1>
          %oe11 = arith.addf %o_11, %zero : f64
          %bo11 = arith.mulf %beta, %oe11 : f64
          scf.yield %bo11 : f64
        }
        %v11 = math.fma %alpha, %res#5, %old11 : f64
        %vt11 = arith.addf %v11, %zero : f64
        memref.store %vt11, %c[%cix11] : memref<?xf64, 1>
      }
      %inr12 = arith.cmpi slt, %row1, %m : i32
      %inc12 = arith.cmpi slt, %col2, %n : i32
      %in12 = arith.andi %inr12, %inc12 : i1
      scf.if %in12 {
        %cl12 = arith.muli %col2, %ldc : i32
        %cc12 = arith.addi %cbase, %cl12 : i32
        %ci12 = arith.addi %cc12, %row1 : i32
        %cix12 = arith.index_cast %ci12 : i32 to index
        %old12 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_12 = memref.load %c[%cix12] : memref<?xf64, 1>
          %oe12 = arith.addf %o_12, %zero : f64
          %bo12 = arith.mulf %beta, %oe12 : f64
          scf.yield %bo12 : f64
        }
        %v12 = math.fma %alpha, %res#6, %old12 : f64
        %vt12 = arith.addf %v12, %zero : f64
        memref.store %vt12, %c[%cix12] : memref<?xf64, 1>
      }
      %inr13 = arith.cmpi slt, %row1, %m : i32
      %inc13 = arith.cmpi slt, %col3, %n : i32
      %in13 = arith.andi %inr13, %inc13 : i1
      scf.if %in13 {
        %cl13 = arith.muli %col3, %ldc : i32
        %cc13 = arith.addi %cbase, %cl13 : i32
        %ci13 = arith.addi %cc13, %row1 : i32
        %cix13 = arith.index_cast %ci13 : i32 to index
        %old13 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_13 = memref.load %c[%cix13] : memref<?xf64, 1>
          %oe13 = arith.addf %o_13, %zero : f64
          %bo13 = arith.mulf %beta, %oe13 : f64
          scf.yield %bo13 : f64
        }
        %v13 = math.fma %alpha, %res#7, %old13 : f64
        %vt13 = arith.addf %v13, %zero : f64
        memref.store %vt13, %c[%cix13] : memref<?xf64, 1>
      }
      %inr20 = arith.cmpi slt, %row2, %m : i32
      %inc20 = arith.cmpi slt, %col0, %n : i32
      %in20 = arith.andi %inr20, %inc20 : i1
      scf.if %in20 {
        %cl20 = arith.muli %col0, %ldc : i32
        %cc20 = arith.addi %cbase, %cl20 : i32
        %ci20 = arith.addi %cc20, %row2 : i32
        %cix20 = arith.index_cast %ci20 : i32 to index
        %old20 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_20 = memref.load %c[%cix20] : memref<?xf64, 1>
          %oe20 = arith.addf %o_20, %zero : f64
          %bo20 = arith.mulf %beta, %oe20 : f64
          scf.yield %bo20 : f64
        }
        %v20 = math.fma %alpha, %res#8, %old20 : f64
        %vt20 = arith.addf %v20, %zero : f64
        memref.store %vt20, %c[%cix20] : memref<?xf64, 1>
      }
      %inr21 = arith.cmpi slt, %row2, %m : i32
      %inc21 = arith.cmpi slt, %col1, %n : i32
      %in21 = arith.andi %inr21, %inc21 : i1
      scf.if %in21 {
        %cl21 = arith.muli %col1, %ldc : i32
        %cc21 = arith.addi %cbase, %cl21 : i32
        %ci21 = arith.addi %cc21, %row2 : i32
        %cix21 = arith.index_cast %ci21 : i32 to index
        %old21 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_21 = memref.load %c[%cix21] : memref<?xf64, 1>
          %oe21 = arith.addf %o_21, %zero : f64
          %bo21 = arith.mulf %beta, %oe21 : f64
          scf.yield %bo21 : f64
        }
        %v21 = math.fma %alpha, %res#9, %old21 : f64
        %vt21 = arith.addf %v21, %zero : f64
        memref.store %vt21, %c[%cix21] : memref<?xf64, 1>
      }
      %inr22 = arith.cmpi slt, %row2, %m : i32
      %inc22 = arith.cmpi slt, %col2, %n : i32
      %in22 = arith.andi %inr22, %inc22 : i1
      scf.if %in22 {
        %cl22 = arith.muli %col2, %ldc : i32
        %cc22 = arith.addi %cbase, %cl22 : i32
        %ci22 = arith.addi %cc22, %row2 : i32
        %cix22 = arith.index_cast %ci22 : i32 to index
        %old22 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_22 = memref.load %c[%cix22] : memref<?xf64, 1>
          %oe22 = arith.addf %o_22, %zero : f64
          %bo22 = arith.mulf %beta, %oe22 : f64
          scf.yield %bo22 : f64
        }
        %v22 = math.fma %alpha, %res#10, %old22 : f64
        %vt22 = arith.addf %v22, %zero : f64
        memref.store %vt22, %c[%cix22] : memref<?xf64, 1>
      }
      %inr23 = arith.cmpi slt, %row2, %m : i32
      %inc23 = arith.cmpi slt, %col3, %n : i32
      %in23 = arith.andi %inr23, %inc23 : i1
      scf.if %in23 {
        %cl23 = arith.muli %col3, %ldc : i32
        %cc23 = arith.addi %cbase, %cl23 : i32
        %ci23 = arith.addi %cc23, %row2 : i32
        %cix23 = arith.index_cast %ci23 : i32 to index
        %old23 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_23 = memref.load %c[%cix23] : memref<?xf64, 1>
          %oe23 = arith.addf %o_23, %zero : f64
          %bo23 = arith.mulf %beta, %oe23 : f64
          scf.yield %bo23 : f64
        }
        %v23 = math.fma %alpha, %res#11, %old23 : f64
        %vt23 = arith.addf %v23, %zero : f64
        memref.store %vt23, %c[%cix23] : memref<?xf64, 1>
      }
      %inr30 = arith.cmpi slt, %row3, %m : i32
      %inc30 = arith.cmpi slt, %col0, %n : i32
      %in30 = arith.andi %inr30, %inc30 : i1
      scf.if %in30 {
        %cl30 = arith.muli %col0, %ldc : i32
        %cc30 = arith.addi %cbase, %cl30 : i32
        %ci30 = arith.addi %cc30, %row3 : i32
        %cix30 = arith.index_cast %ci30 : i32 to index
        %old30 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_30 = memref.load %c[%cix30] : memref<?xf64, 1>
          %oe30 = arith.addf %o_30, %zero : f64
          %bo30 = arith.mulf %beta, %oe30 : f64
          scf.yield %bo30 : f64
        }
        %v30 = math.fma %alpha, %res#12, %old30 : f64
        %vt30 = arith.addf %v30, %zero : f64
        memref.store %vt30, %c[%cix30] : memref<?xf64, 1>
      }
      %inr31 = arith.cmpi slt, %row3, %m : i32
      %inc31 = arith.cmpi slt, %col1, %n : i32
      %in31 = arith.andi %inr31, %inc31 : i1
      scf.if %in31 {
        %cl31 = arith.muli %col1, %ldc : i32
        %cc31 = arith.addi %cbase, %cl31 : i32
        %ci31 = arith.addi %cc31, %row3 : i32
        %cix31 = arith.index_cast %ci31 : i32 to index
        %old31 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_31 = memref.load %c[%cix31] : memref<?xf64, 1>
          %oe31 = arith.addf %o_31, %zero : f64
          %bo31 = arith.mulf %beta, %oe31 : f64
          scf.yield %bo31 : f64
        }
        %v31 = math.fma %alpha, %res#13, %old31 : f64
        %vt31 = arith.addf %v31, %zero : f64
        memref.store %vt31, %c[%cix31] : memref<?xf64, 1>
      }
      %inr32 = arith.cmpi slt, %row3, %m : i32
      %inc32 = arith.cmpi slt, %col2, %n : i32
      %in32 = arith.andi %inr32, %inc32 : i1
      scf.if %in32 {
        %cl32 = arith.muli %col2, %ldc : i32
        %cc32 = arith.addi %cbase, %cl32 : i32
        %ci32 = arith.addi %cc32, %row3 : i32
        %cix32 = arith.index_cast %ci32 : i32 to index
        %old32 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_32 = memref.load %c[%cix32] : memref<?xf64, 1>
          %oe32 = arith.addf %o_32, %zero : f64
          %bo32 = arith.mulf %beta, %oe32 : f64
          scf.yield %bo32 : f64
        }
        %v32 = math.fma %alpha, %res#14, %old32 : f64
        %vt32 = arith.addf %v32, %zero : f64
        memref.store %vt32, %c[%cix32] : memref<?xf64, 1>
      }
      %inr33 = arith.cmpi slt, %row3, %m : i32
      %inc33 = arith.cmpi slt, %col3, %n : i32
      %in33 = arith.andi %inr33, %inc33 : i1
      scf.if %in33 {
        %cl33 = arith.muli %col3, %ldc : i32
        %cc33 = arith.addi %cbase, %cl33 : i32
        %ci33 = arith.addi %cc33, %row3 : i32
        %cix33 = arith.index_cast %ci33 : i32 to index
        %old33 = scf.if %bzero -> (f64) {
          scf.yield %zero : f64
        } else {
          %o_33 = memref.load %c[%cix33] : memref<?xf64, 1>
          %oe33 = arith.addf %o_33, %zero : f64
          %bo33 = arith.mulf %beta, %oe33 : f64
          scf.yield %bo33 : f64
        }
        %v33 = math.fma %alpha, %res#15, %old33 : f64
        %vt33 = arith.addf %v33, %zero : f64
        memref.store %vt33, %c[%cix33] : memref<?xf64, 1>
      }
      gpu.return
    }
    gpu.func @gemm_f16_f16(%m: i32, %n: i32, %k: i32, %alpha: f32, %a: memref<?xf16, 1>, %ao: i32, %ars: i32, %acs: i32, %sa: i32, %b: memref<?xf16, 1>, %bo: i32, %brs: i32, %bcs: i32, %sb: i32, %beta: f32, %c: memref<?xf16, 1>, %co: i32, %ldc: i32, %sc: i32) kernel {
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
      %zero = arith.constant 0.0 : f32
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
      %cbase = arith.addi %co, %zsc : i32
      %mm1 = arith.subi %m, %c1 : i32
      %nm1 = arith.subi %n, %c1 : i32
      %cr0 = arith.constant 0 : i32
      %ro0 = arith.muli %cr0, %dxi : i32
      %row0 = arith.addi %i0, %ro0 : i32
      %rowc0 = arith.minsi %row0, %mm1 : i32
      %rofs0 = arith.muli %rowc0, %ars : i32
      %arow0 = arith.addi %abase, %rofs0 : i32
      %co0 = arith.muli %cr0, %dyi : i32
      %col0 = arith.addi %j0, %co0 : i32
      %colc0 = arith.minsi %col0, %nm1 : i32
      %cofs0 = arith.muli %colc0, %bcs : i32
      %bcol0 = arith.addi %bbase, %cofs0 : i32
      %cr1 = arith.constant 1 : i32
      %ro1 = arith.muli %cr1, %dxi : i32
      %row1 = arith.addi %i0, %ro1 : i32
      %rowc1 = arith.minsi %row1, %mm1 : i32
      %rofs1 = arith.muli %rowc1, %ars : i32
      %arow1 = arith.addi %abase, %rofs1 : i32
      %co1 = arith.muli %cr1, %dyi : i32
      %col1 = arith.addi %j0, %co1 : i32
      %colc1 = arith.minsi %col1, %nm1 : i32
      %cofs1 = arith.muli %colc1, %bcs : i32
      %bcol1 = arith.addi %bbase, %cofs1 : i32
      %cr2 = arith.constant 2 : i32
      %ro2 = arith.muli %cr2, %dxi : i32
      %row2 = arith.addi %i0, %ro2 : i32
      %rowc2 = arith.minsi %row2, %mm1 : i32
      %rofs2 = arith.muli %rowc2, %ars : i32
      %arow2 = arith.addi %abase, %rofs2 : i32
      %co2 = arith.muli %cr2, %dyi : i32
      %col2 = arith.addi %j0, %co2 : i32
      %colc2 = arith.minsi %col2, %nm1 : i32
      %cofs2 = arith.muli %colc2, %bcs : i32
      %bcol2 = arith.addi %bbase, %cofs2 : i32
      %cr3 = arith.constant 3 : i32
      %ro3 = arith.muli %cr3, %dxi : i32
      %row3 = arith.addi %i0, %ro3 : i32
      %rowc3 = arith.minsi %row3, %mm1 : i32
      %rofs3 = arith.muli %rowc3, %ars : i32
      %arow3 = arith.addi %abase, %rofs3 : i32
      %co3 = arith.muli %cr3, %dyi : i32
      %col3 = arith.addi %j0, %co3 : i32
      %colc3 = arith.minsi %col3, %nm1 : i32
      %cofs3 = arith.muli %colc3, %bcs : i32
      %bcol3 = arith.addi %bbase, %cofs3 : i32
      %res:16 = scf.for %l = %c0 to %k step %c1 iter_args(%s00 = %zero, %s01 = %zero, %s02 = %zero, %s03 = %zero, %s10 = %zero, %s11 = %zero, %s12 = %zero, %s13 = %zero, %s20 = %zero, %s21 = %zero, %s22 = %zero, %s23 = %zero, %s30 = %zero, %s31 = %zero, %s32 = %zero, %s33 = %zero) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) : i32 {
        %la = arith.muli %l, %acs : i32
        %lb = arith.muli %l, %brs : i32
        %ai0 = arith.addi %arow0, %la : i32
        %aix0 = arith.index_cast %ai0 : i32 to index
        %av0_ = memref.load %a[%aix0] : memref<?xf16, 1>
        %av0 = arith.extf %av0_ : f16 to f32
        %bi0 = arith.addi %bcol0, %lb : i32
        %bix0 = arith.index_cast %bi0 : i32 to index
        %bv0_ = memref.load %b[%bix0] : memref<?xf16, 1>
        %bv0 = arith.extf %bv0_ : f16 to f32
        %ai1 = arith.addi %arow1, %la : i32
        %aix1 = arith.index_cast %ai1 : i32 to index
        %av1_ = memref.load %a[%aix1] : memref<?xf16, 1>
        %av1 = arith.extf %av1_ : f16 to f32
        %bi1 = arith.addi %bcol1, %lb : i32
        %bix1 = arith.index_cast %bi1 : i32 to index
        %bv1_ = memref.load %b[%bix1] : memref<?xf16, 1>
        %bv1 = arith.extf %bv1_ : f16 to f32
        %ai2 = arith.addi %arow2, %la : i32
        %aix2 = arith.index_cast %ai2 : i32 to index
        %av2_ = memref.load %a[%aix2] : memref<?xf16, 1>
        %av2 = arith.extf %av2_ : f16 to f32
        %bi2 = arith.addi %bcol2, %lb : i32
        %bix2 = arith.index_cast %bi2 : i32 to index
        %bv2_ = memref.load %b[%bix2] : memref<?xf16, 1>
        %bv2 = arith.extf %bv2_ : f16 to f32
        %ai3 = arith.addi %arow3, %la : i32
        %aix3 = arith.index_cast %ai3 : i32 to index
        %av3_ = memref.load %a[%aix3] : memref<?xf16, 1>
        %av3 = arith.extf %av3_ : f16 to f32
        %bi3 = arith.addi %bcol3, %lb : i32
        %bix3 = arith.index_cast %bi3 : i32 to index
        %bv3_ = memref.load %b[%bix3] : memref<?xf16, 1>
        %bv3 = arith.extf %bv3_ : f16 to f32
        %n00 = math.fma %av0, %bv0, %s00 : f32
        %n01 = math.fma %av0, %bv1, %s01 : f32
        %n02 = math.fma %av0, %bv2, %s02 : f32
        %n03 = math.fma %av0, %bv3, %s03 : f32
        %n10 = math.fma %av1, %bv0, %s10 : f32
        %n11 = math.fma %av1, %bv1, %s11 : f32
        %n12 = math.fma %av1, %bv2, %s12 : f32
        %n13 = math.fma %av1, %bv3, %s13 : f32
        %n20 = math.fma %av2, %bv0, %s20 : f32
        %n21 = math.fma %av2, %bv1, %s21 : f32
        %n22 = math.fma %av2, %bv2, %s22 : f32
        %n23 = math.fma %av2, %bv3, %s23 : f32
        %n30 = math.fma %av3, %bv0, %s30 : f32
        %n31 = math.fma %av3, %bv1, %s31 : f32
        %n32 = math.fma %av3, %bv2, %s32 : f32
        %n33 = math.fma %av3, %bv3, %s33 : f32
        scf.yield %n00, %n01, %n02, %n03, %n10, %n11, %n12, %n13, %n20, %n21, %n22, %n23, %n30, %n31, %n32, %n33 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
      }
      %bzero = arith.cmpf oeq, %beta, %zero : f32
      %inr00 = arith.cmpi slt, %row0, %m : i32
      %inc00 = arith.cmpi slt, %col0, %n : i32
      %in00 = arith.andi %inr00, %inc00 : i1
      scf.if %in00 {
        %cl00 = arith.muli %col0, %ldc : i32
        %cc00 = arith.addi %cbase, %cl00 : i32
        %ci00 = arith.addi %cc00, %row0 : i32
        %cix00 = arith.index_cast %ci00 : i32 to index
        %old00 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_00 = memref.load %c[%cix00] : memref<?xf16, 1>
          %oe00 = arith.extf %o_00 : f16 to f32
          %bo00 = arith.mulf %beta, %oe00 : f32
          scf.yield %bo00 : f32
        }
        %v00 = math.fma %alpha, %res#0, %old00 : f32
        %vt00 = arith.truncf %v00 : f32 to f16
        memref.store %vt00, %c[%cix00] : memref<?xf16, 1>
      }
      %inr01 = arith.cmpi slt, %row0, %m : i32
      %inc01 = arith.cmpi slt, %col1, %n : i32
      %in01 = arith.andi %inr01, %inc01 : i1
      scf.if %in01 {
        %cl01 = arith.muli %col1, %ldc : i32
        %cc01 = arith.addi %cbase, %cl01 : i32
        %ci01 = arith.addi %cc01, %row0 : i32
        %cix01 = arith.index_cast %ci01 : i32 to index
        %old01 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_01 = memref.load %c[%cix01] : memref<?xf16, 1>
          %oe01 = arith.extf %o_01 : f16 to f32
          %bo01 = arith.mulf %beta, %oe01 : f32
          scf.yield %bo01 : f32
        }
        %v01 = math.fma %alpha, %res#1, %old01 : f32
        %vt01 = arith.truncf %v01 : f32 to f16
        memref.store %vt01, %c[%cix01] : memref<?xf16, 1>
      }
      %inr02 = arith.cmpi slt, %row0, %m : i32
      %inc02 = arith.cmpi slt, %col2, %n : i32
      %in02 = arith.andi %inr02, %inc02 : i1
      scf.if %in02 {
        %cl02 = arith.muli %col2, %ldc : i32
        %cc02 = arith.addi %cbase, %cl02 : i32
        %ci02 = arith.addi %cc02, %row0 : i32
        %cix02 = arith.index_cast %ci02 : i32 to index
        %old02 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_02 = memref.load %c[%cix02] : memref<?xf16, 1>
          %oe02 = arith.extf %o_02 : f16 to f32
          %bo02 = arith.mulf %beta, %oe02 : f32
          scf.yield %bo02 : f32
        }
        %v02 = math.fma %alpha, %res#2, %old02 : f32
        %vt02 = arith.truncf %v02 : f32 to f16
        memref.store %vt02, %c[%cix02] : memref<?xf16, 1>
      }
      %inr03 = arith.cmpi slt, %row0, %m : i32
      %inc03 = arith.cmpi slt, %col3, %n : i32
      %in03 = arith.andi %inr03, %inc03 : i1
      scf.if %in03 {
        %cl03 = arith.muli %col3, %ldc : i32
        %cc03 = arith.addi %cbase, %cl03 : i32
        %ci03 = arith.addi %cc03, %row0 : i32
        %cix03 = arith.index_cast %ci03 : i32 to index
        %old03 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_03 = memref.load %c[%cix03] : memref<?xf16, 1>
          %oe03 = arith.extf %o_03 : f16 to f32
          %bo03 = arith.mulf %beta, %oe03 : f32
          scf.yield %bo03 : f32
        }
        %v03 = math.fma %alpha, %res#3, %old03 : f32
        %vt03 = arith.truncf %v03 : f32 to f16
        memref.store %vt03, %c[%cix03] : memref<?xf16, 1>
      }
      %inr10 = arith.cmpi slt, %row1, %m : i32
      %inc10 = arith.cmpi slt, %col0, %n : i32
      %in10 = arith.andi %inr10, %inc10 : i1
      scf.if %in10 {
        %cl10 = arith.muli %col0, %ldc : i32
        %cc10 = arith.addi %cbase, %cl10 : i32
        %ci10 = arith.addi %cc10, %row1 : i32
        %cix10 = arith.index_cast %ci10 : i32 to index
        %old10 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_10 = memref.load %c[%cix10] : memref<?xf16, 1>
          %oe10 = arith.extf %o_10 : f16 to f32
          %bo10 = arith.mulf %beta, %oe10 : f32
          scf.yield %bo10 : f32
        }
        %v10 = math.fma %alpha, %res#4, %old10 : f32
        %vt10 = arith.truncf %v10 : f32 to f16
        memref.store %vt10, %c[%cix10] : memref<?xf16, 1>
      }
      %inr11 = arith.cmpi slt, %row1, %m : i32
      %inc11 = arith.cmpi slt, %col1, %n : i32
      %in11 = arith.andi %inr11, %inc11 : i1
      scf.if %in11 {
        %cl11 = arith.muli %col1, %ldc : i32
        %cc11 = arith.addi %cbase, %cl11 : i32
        %ci11 = arith.addi %cc11, %row1 : i32
        %cix11 = arith.index_cast %ci11 : i32 to index
        %old11 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_11 = memref.load %c[%cix11] : memref<?xf16, 1>
          %oe11 = arith.extf %o_11 : f16 to f32
          %bo11 = arith.mulf %beta, %oe11 : f32
          scf.yield %bo11 : f32
        }
        %v11 = math.fma %alpha, %res#5, %old11 : f32
        %vt11 = arith.truncf %v11 : f32 to f16
        memref.store %vt11, %c[%cix11] : memref<?xf16, 1>
      }
      %inr12 = arith.cmpi slt, %row1, %m : i32
      %inc12 = arith.cmpi slt, %col2, %n : i32
      %in12 = arith.andi %inr12, %inc12 : i1
      scf.if %in12 {
        %cl12 = arith.muli %col2, %ldc : i32
        %cc12 = arith.addi %cbase, %cl12 : i32
        %ci12 = arith.addi %cc12, %row1 : i32
        %cix12 = arith.index_cast %ci12 : i32 to index
        %old12 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_12 = memref.load %c[%cix12] : memref<?xf16, 1>
          %oe12 = arith.extf %o_12 : f16 to f32
          %bo12 = arith.mulf %beta, %oe12 : f32
          scf.yield %bo12 : f32
        }
        %v12 = math.fma %alpha, %res#6, %old12 : f32
        %vt12 = arith.truncf %v12 : f32 to f16
        memref.store %vt12, %c[%cix12] : memref<?xf16, 1>
      }
      %inr13 = arith.cmpi slt, %row1, %m : i32
      %inc13 = arith.cmpi slt, %col3, %n : i32
      %in13 = arith.andi %inr13, %inc13 : i1
      scf.if %in13 {
        %cl13 = arith.muli %col3, %ldc : i32
        %cc13 = arith.addi %cbase, %cl13 : i32
        %ci13 = arith.addi %cc13, %row1 : i32
        %cix13 = arith.index_cast %ci13 : i32 to index
        %old13 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_13 = memref.load %c[%cix13] : memref<?xf16, 1>
          %oe13 = arith.extf %o_13 : f16 to f32
          %bo13 = arith.mulf %beta, %oe13 : f32
          scf.yield %bo13 : f32
        }
        %v13 = math.fma %alpha, %res#7, %old13 : f32
        %vt13 = arith.truncf %v13 : f32 to f16
        memref.store %vt13, %c[%cix13] : memref<?xf16, 1>
      }
      %inr20 = arith.cmpi slt, %row2, %m : i32
      %inc20 = arith.cmpi slt, %col0, %n : i32
      %in20 = arith.andi %inr20, %inc20 : i1
      scf.if %in20 {
        %cl20 = arith.muli %col0, %ldc : i32
        %cc20 = arith.addi %cbase, %cl20 : i32
        %ci20 = arith.addi %cc20, %row2 : i32
        %cix20 = arith.index_cast %ci20 : i32 to index
        %old20 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_20 = memref.load %c[%cix20] : memref<?xf16, 1>
          %oe20 = arith.extf %o_20 : f16 to f32
          %bo20 = arith.mulf %beta, %oe20 : f32
          scf.yield %bo20 : f32
        }
        %v20 = math.fma %alpha, %res#8, %old20 : f32
        %vt20 = arith.truncf %v20 : f32 to f16
        memref.store %vt20, %c[%cix20] : memref<?xf16, 1>
      }
      %inr21 = arith.cmpi slt, %row2, %m : i32
      %inc21 = arith.cmpi slt, %col1, %n : i32
      %in21 = arith.andi %inr21, %inc21 : i1
      scf.if %in21 {
        %cl21 = arith.muli %col1, %ldc : i32
        %cc21 = arith.addi %cbase, %cl21 : i32
        %ci21 = arith.addi %cc21, %row2 : i32
        %cix21 = arith.index_cast %ci21 : i32 to index
        %old21 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_21 = memref.load %c[%cix21] : memref<?xf16, 1>
          %oe21 = arith.extf %o_21 : f16 to f32
          %bo21 = arith.mulf %beta, %oe21 : f32
          scf.yield %bo21 : f32
        }
        %v21 = math.fma %alpha, %res#9, %old21 : f32
        %vt21 = arith.truncf %v21 : f32 to f16
        memref.store %vt21, %c[%cix21] : memref<?xf16, 1>
      }
      %inr22 = arith.cmpi slt, %row2, %m : i32
      %inc22 = arith.cmpi slt, %col2, %n : i32
      %in22 = arith.andi %inr22, %inc22 : i1
      scf.if %in22 {
        %cl22 = arith.muli %col2, %ldc : i32
        %cc22 = arith.addi %cbase, %cl22 : i32
        %ci22 = arith.addi %cc22, %row2 : i32
        %cix22 = arith.index_cast %ci22 : i32 to index
        %old22 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_22 = memref.load %c[%cix22] : memref<?xf16, 1>
          %oe22 = arith.extf %o_22 : f16 to f32
          %bo22 = arith.mulf %beta, %oe22 : f32
          scf.yield %bo22 : f32
        }
        %v22 = math.fma %alpha, %res#10, %old22 : f32
        %vt22 = arith.truncf %v22 : f32 to f16
        memref.store %vt22, %c[%cix22] : memref<?xf16, 1>
      }
      %inr23 = arith.cmpi slt, %row2, %m : i32
      %inc23 = arith.cmpi slt, %col3, %n : i32
      %in23 = arith.andi %inr23, %inc23 : i1
      scf.if %in23 {
        %cl23 = arith.muli %col3, %ldc : i32
        %cc23 = arith.addi %cbase, %cl23 : i32
        %ci23 = arith.addi %cc23, %row2 : i32
        %cix23 = arith.index_cast %ci23 : i32 to index
        %old23 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_23 = memref.load %c[%cix23] : memref<?xf16, 1>
          %oe23 = arith.extf %o_23 : f16 to f32
          %bo23 = arith.mulf %beta, %oe23 : f32
          scf.yield %bo23 : f32
        }
        %v23 = math.fma %alpha, %res#11, %old23 : f32
        %vt23 = arith.truncf %v23 : f32 to f16
        memref.store %vt23, %c[%cix23] : memref<?xf16, 1>
      }
      %inr30 = arith.cmpi slt, %row3, %m : i32
      %inc30 = arith.cmpi slt, %col0, %n : i32
      %in30 = arith.andi %inr30, %inc30 : i1
      scf.if %in30 {
        %cl30 = arith.muli %col0, %ldc : i32
        %cc30 = arith.addi %cbase, %cl30 : i32
        %ci30 = arith.addi %cc30, %row3 : i32
        %cix30 = arith.index_cast %ci30 : i32 to index
        %old30 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_30 = memref.load %c[%cix30] : memref<?xf16, 1>
          %oe30 = arith.extf %o_30 : f16 to f32
          %bo30 = arith.mulf %beta, %oe30 : f32
          scf.yield %bo30 : f32
        }
        %v30 = math.fma %alpha, %res#12, %old30 : f32
        %vt30 = arith.truncf %v30 : f32 to f16
        memref.store %vt30, %c[%cix30] : memref<?xf16, 1>
      }
      %inr31 = arith.cmpi slt, %row3, %m : i32
      %inc31 = arith.cmpi slt, %col1, %n : i32
      %in31 = arith.andi %inr31, %inc31 : i1
      scf.if %in31 {
        %cl31 = arith.muli %col1, %ldc : i32
        %cc31 = arith.addi %cbase, %cl31 : i32
        %ci31 = arith.addi %cc31, %row3 : i32
        %cix31 = arith.index_cast %ci31 : i32 to index
        %old31 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_31 = memref.load %c[%cix31] : memref<?xf16, 1>
          %oe31 = arith.extf %o_31 : f16 to f32
          %bo31 = arith.mulf %beta, %oe31 : f32
          scf.yield %bo31 : f32
        }
        %v31 = math.fma %alpha, %res#13, %old31 : f32
        %vt31 = arith.truncf %v31 : f32 to f16
        memref.store %vt31, %c[%cix31] : memref<?xf16, 1>
      }
      %inr32 = arith.cmpi slt, %row3, %m : i32
      %inc32 = arith.cmpi slt, %col2, %n : i32
      %in32 = arith.andi %inr32, %inc32 : i1
      scf.if %in32 {
        %cl32 = arith.muli %col2, %ldc : i32
        %cc32 = arith.addi %cbase, %cl32 : i32
        %ci32 = arith.addi %cc32, %row3 : i32
        %cix32 = arith.index_cast %ci32 : i32 to index
        %old32 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_32 = memref.load %c[%cix32] : memref<?xf16, 1>
          %oe32 = arith.extf %o_32 : f16 to f32
          %bo32 = arith.mulf %beta, %oe32 : f32
          scf.yield %bo32 : f32
        }
        %v32 = math.fma %alpha, %res#14, %old32 : f32
        %vt32 = arith.truncf %v32 : f32 to f16
        memref.store %vt32, %c[%cix32] : memref<?xf16, 1>
      }
      %inr33 = arith.cmpi slt, %row3, %m : i32
      %inc33 = arith.cmpi slt, %col3, %n : i32
      %in33 = arith.andi %inr33, %inc33 : i1
      scf.if %in33 {
        %cl33 = arith.muli %col3, %ldc : i32
        %cc33 = arith.addi %cbase, %cl33 : i32
        %ci33 = arith.addi %cc33, %row3 : i32
        %cix33 = arith.index_cast %ci33 : i32 to index
        %old33 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_33 = memref.load %c[%cix33] : memref<?xf16, 1>
          %oe33 = arith.extf %o_33 : f16 to f32
          %bo33 = arith.mulf %beta, %oe33 : f32
          scf.yield %bo33 : f32
        }
        %v33 = math.fma %alpha, %res#15, %old33 : f32
        %vt33 = arith.truncf %v33 : f32 to f16
        memref.store %vt33, %c[%cix33] : memref<?xf16, 1>
      }
      gpu.return
    }
    gpu.func @gemm_f16_f32(%m: i32, %n: i32, %k: i32, %alpha: f32, %a: memref<?xf16, 1>, %ao: i32, %ars: i32, %acs: i32, %sa: i32, %b: memref<?xf16, 1>, %bo: i32, %brs: i32, %bcs: i32, %sb: i32, %beta: f32, %c: memref<?xf32, 1>, %co: i32, %ldc: i32, %sc: i32) kernel {
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
      %zero = arith.constant 0.0 : f32
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
      %cbase = arith.addi %co, %zsc : i32
      %mm1 = arith.subi %m, %c1 : i32
      %nm1 = arith.subi %n, %c1 : i32
      %cr0 = arith.constant 0 : i32
      %ro0 = arith.muli %cr0, %dxi : i32
      %row0 = arith.addi %i0, %ro0 : i32
      %rowc0 = arith.minsi %row0, %mm1 : i32
      %rofs0 = arith.muli %rowc0, %ars : i32
      %arow0 = arith.addi %abase, %rofs0 : i32
      %co0 = arith.muli %cr0, %dyi : i32
      %col0 = arith.addi %j0, %co0 : i32
      %colc0 = arith.minsi %col0, %nm1 : i32
      %cofs0 = arith.muli %colc0, %bcs : i32
      %bcol0 = arith.addi %bbase, %cofs0 : i32
      %cr1 = arith.constant 1 : i32
      %ro1 = arith.muli %cr1, %dxi : i32
      %row1 = arith.addi %i0, %ro1 : i32
      %rowc1 = arith.minsi %row1, %mm1 : i32
      %rofs1 = arith.muli %rowc1, %ars : i32
      %arow1 = arith.addi %abase, %rofs1 : i32
      %co1 = arith.muli %cr1, %dyi : i32
      %col1 = arith.addi %j0, %co1 : i32
      %colc1 = arith.minsi %col1, %nm1 : i32
      %cofs1 = arith.muli %colc1, %bcs : i32
      %bcol1 = arith.addi %bbase, %cofs1 : i32
      %cr2 = arith.constant 2 : i32
      %ro2 = arith.muli %cr2, %dxi : i32
      %row2 = arith.addi %i0, %ro2 : i32
      %rowc2 = arith.minsi %row2, %mm1 : i32
      %rofs2 = arith.muli %rowc2, %ars : i32
      %arow2 = arith.addi %abase, %rofs2 : i32
      %co2 = arith.muli %cr2, %dyi : i32
      %col2 = arith.addi %j0, %co2 : i32
      %colc2 = arith.minsi %col2, %nm1 : i32
      %cofs2 = arith.muli %colc2, %bcs : i32
      %bcol2 = arith.addi %bbase, %cofs2 : i32
      %cr3 = arith.constant 3 : i32
      %ro3 = arith.muli %cr3, %dxi : i32
      %row3 = arith.addi %i0, %ro3 : i32
      %rowc3 = arith.minsi %row3, %mm1 : i32
      %rofs3 = arith.muli %rowc3, %ars : i32
      %arow3 = arith.addi %abase, %rofs3 : i32
      %co3 = arith.muli %cr3, %dyi : i32
      %col3 = arith.addi %j0, %co3 : i32
      %colc3 = arith.minsi %col3, %nm1 : i32
      %cofs3 = arith.muli %colc3, %bcs : i32
      %bcol3 = arith.addi %bbase, %cofs3 : i32
      %res:16 = scf.for %l = %c0 to %k step %c1 iter_args(%s00 = %zero, %s01 = %zero, %s02 = %zero, %s03 = %zero, %s10 = %zero, %s11 = %zero, %s12 = %zero, %s13 = %zero, %s20 = %zero, %s21 = %zero, %s22 = %zero, %s23 = %zero, %s30 = %zero, %s31 = %zero, %s32 = %zero, %s33 = %zero) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) : i32 {
        %la = arith.muli %l, %acs : i32
        %lb = arith.muli %l, %brs : i32
        %ai0 = arith.addi %arow0, %la : i32
        %aix0 = arith.index_cast %ai0 : i32 to index
        %av0_ = memref.load %a[%aix0] : memref<?xf16, 1>
        %av0 = arith.extf %av0_ : f16 to f32
        %bi0 = arith.addi %bcol0, %lb : i32
        %bix0 = arith.index_cast %bi0 : i32 to index
        %bv0_ = memref.load %b[%bix0] : memref<?xf16, 1>
        %bv0 = arith.extf %bv0_ : f16 to f32
        %ai1 = arith.addi %arow1, %la : i32
        %aix1 = arith.index_cast %ai1 : i32 to index
        %av1_ = memref.load %a[%aix1] : memref<?xf16, 1>
        %av1 = arith.extf %av1_ : f16 to f32
        %bi1 = arith.addi %bcol1, %lb : i32
        %bix1 = arith.index_cast %bi1 : i32 to index
        %bv1_ = memref.load %b[%bix1] : memref<?xf16, 1>
        %bv1 = arith.extf %bv1_ : f16 to f32
        %ai2 = arith.addi %arow2, %la : i32
        %aix2 = arith.index_cast %ai2 : i32 to index
        %av2_ = memref.load %a[%aix2] : memref<?xf16, 1>
        %av2 = arith.extf %av2_ : f16 to f32
        %bi2 = arith.addi %bcol2, %lb : i32
        %bix2 = arith.index_cast %bi2 : i32 to index
        %bv2_ = memref.load %b[%bix2] : memref<?xf16, 1>
        %bv2 = arith.extf %bv2_ : f16 to f32
        %ai3 = arith.addi %arow3, %la : i32
        %aix3 = arith.index_cast %ai3 : i32 to index
        %av3_ = memref.load %a[%aix3] : memref<?xf16, 1>
        %av3 = arith.extf %av3_ : f16 to f32
        %bi3 = arith.addi %bcol3, %lb : i32
        %bix3 = arith.index_cast %bi3 : i32 to index
        %bv3_ = memref.load %b[%bix3] : memref<?xf16, 1>
        %bv3 = arith.extf %bv3_ : f16 to f32
        %n00 = math.fma %av0, %bv0, %s00 : f32
        %n01 = math.fma %av0, %bv1, %s01 : f32
        %n02 = math.fma %av0, %bv2, %s02 : f32
        %n03 = math.fma %av0, %bv3, %s03 : f32
        %n10 = math.fma %av1, %bv0, %s10 : f32
        %n11 = math.fma %av1, %bv1, %s11 : f32
        %n12 = math.fma %av1, %bv2, %s12 : f32
        %n13 = math.fma %av1, %bv3, %s13 : f32
        %n20 = math.fma %av2, %bv0, %s20 : f32
        %n21 = math.fma %av2, %bv1, %s21 : f32
        %n22 = math.fma %av2, %bv2, %s22 : f32
        %n23 = math.fma %av2, %bv3, %s23 : f32
        %n30 = math.fma %av3, %bv0, %s30 : f32
        %n31 = math.fma %av3, %bv1, %s31 : f32
        %n32 = math.fma %av3, %bv2, %s32 : f32
        %n33 = math.fma %av3, %bv3, %s33 : f32
        scf.yield %n00, %n01, %n02, %n03, %n10, %n11, %n12, %n13, %n20, %n21, %n22, %n23, %n30, %n31, %n32, %n33 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
      }
      %bzero = arith.cmpf oeq, %beta, %zero : f32
      %inr00 = arith.cmpi slt, %row0, %m : i32
      %inc00 = arith.cmpi slt, %col0, %n : i32
      %in00 = arith.andi %inr00, %inc00 : i1
      scf.if %in00 {
        %cl00 = arith.muli %col0, %ldc : i32
        %cc00 = arith.addi %cbase, %cl00 : i32
        %ci00 = arith.addi %cc00, %row0 : i32
        %cix00 = arith.index_cast %ci00 : i32 to index
        %old00 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_00 = memref.load %c[%cix00] : memref<?xf32, 1>
          %oe00 = arith.addf %o_00, %zero : f32
          %bo00 = arith.mulf %beta, %oe00 : f32
          scf.yield %bo00 : f32
        }
        %v00 = math.fma %alpha, %res#0, %old00 : f32
        %vt00 = arith.addf %v00, %zero : f32
        memref.store %vt00, %c[%cix00] : memref<?xf32, 1>
      }
      %inr01 = arith.cmpi slt, %row0, %m : i32
      %inc01 = arith.cmpi slt, %col1, %n : i32
      %in01 = arith.andi %inr01, %inc01 : i1
      scf.if %in01 {
        %cl01 = arith.muli %col1, %ldc : i32
        %cc01 = arith.addi %cbase, %cl01 : i32
        %ci01 = arith.addi %cc01, %row0 : i32
        %cix01 = arith.index_cast %ci01 : i32 to index
        %old01 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_01 = memref.load %c[%cix01] : memref<?xf32, 1>
          %oe01 = arith.addf %o_01, %zero : f32
          %bo01 = arith.mulf %beta, %oe01 : f32
          scf.yield %bo01 : f32
        }
        %v01 = math.fma %alpha, %res#1, %old01 : f32
        %vt01 = arith.addf %v01, %zero : f32
        memref.store %vt01, %c[%cix01] : memref<?xf32, 1>
      }
      %inr02 = arith.cmpi slt, %row0, %m : i32
      %inc02 = arith.cmpi slt, %col2, %n : i32
      %in02 = arith.andi %inr02, %inc02 : i1
      scf.if %in02 {
        %cl02 = arith.muli %col2, %ldc : i32
        %cc02 = arith.addi %cbase, %cl02 : i32
        %ci02 = arith.addi %cc02, %row0 : i32
        %cix02 = arith.index_cast %ci02 : i32 to index
        %old02 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_02 = memref.load %c[%cix02] : memref<?xf32, 1>
          %oe02 = arith.addf %o_02, %zero : f32
          %bo02 = arith.mulf %beta, %oe02 : f32
          scf.yield %bo02 : f32
        }
        %v02 = math.fma %alpha, %res#2, %old02 : f32
        %vt02 = arith.addf %v02, %zero : f32
        memref.store %vt02, %c[%cix02] : memref<?xf32, 1>
      }
      %inr03 = arith.cmpi slt, %row0, %m : i32
      %inc03 = arith.cmpi slt, %col3, %n : i32
      %in03 = arith.andi %inr03, %inc03 : i1
      scf.if %in03 {
        %cl03 = arith.muli %col3, %ldc : i32
        %cc03 = arith.addi %cbase, %cl03 : i32
        %ci03 = arith.addi %cc03, %row0 : i32
        %cix03 = arith.index_cast %ci03 : i32 to index
        %old03 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_03 = memref.load %c[%cix03] : memref<?xf32, 1>
          %oe03 = arith.addf %o_03, %zero : f32
          %bo03 = arith.mulf %beta, %oe03 : f32
          scf.yield %bo03 : f32
        }
        %v03 = math.fma %alpha, %res#3, %old03 : f32
        %vt03 = arith.addf %v03, %zero : f32
        memref.store %vt03, %c[%cix03] : memref<?xf32, 1>
      }
      %inr10 = arith.cmpi slt, %row1, %m : i32
      %inc10 = arith.cmpi slt, %col0, %n : i32
      %in10 = arith.andi %inr10, %inc10 : i1
      scf.if %in10 {
        %cl10 = arith.muli %col0, %ldc : i32
        %cc10 = arith.addi %cbase, %cl10 : i32
        %ci10 = arith.addi %cc10, %row1 : i32
        %cix10 = arith.index_cast %ci10 : i32 to index
        %old10 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_10 = memref.load %c[%cix10] : memref<?xf32, 1>
          %oe10 = arith.addf %o_10, %zero : f32
          %bo10 = arith.mulf %beta, %oe10 : f32
          scf.yield %bo10 : f32
        }
        %v10 = math.fma %alpha, %res#4, %old10 : f32
        %vt10 = arith.addf %v10, %zero : f32
        memref.store %vt10, %c[%cix10] : memref<?xf32, 1>
      }
      %inr11 = arith.cmpi slt, %row1, %m : i32
      %inc11 = arith.cmpi slt, %col1, %n : i32
      %in11 = arith.andi %inr11, %inc11 : i1
      scf.if %in11 {
        %cl11 = arith.muli %col1, %ldc : i32
        %cc11 = arith.addi %cbase, %cl11 : i32
        %ci11 = arith.addi %cc11, %row1 : i32
        %cix11 = arith.index_cast %ci11 : i32 to index
        %old11 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_11 = memref.load %c[%cix11] : memref<?xf32, 1>
          %oe11 = arith.addf %o_11, %zero : f32
          %bo11 = arith.mulf %beta, %oe11 : f32
          scf.yield %bo11 : f32
        }
        %v11 = math.fma %alpha, %res#5, %old11 : f32
        %vt11 = arith.addf %v11, %zero : f32
        memref.store %vt11, %c[%cix11] : memref<?xf32, 1>
      }
      %inr12 = arith.cmpi slt, %row1, %m : i32
      %inc12 = arith.cmpi slt, %col2, %n : i32
      %in12 = arith.andi %inr12, %inc12 : i1
      scf.if %in12 {
        %cl12 = arith.muli %col2, %ldc : i32
        %cc12 = arith.addi %cbase, %cl12 : i32
        %ci12 = arith.addi %cc12, %row1 : i32
        %cix12 = arith.index_cast %ci12 : i32 to index
        %old12 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_12 = memref.load %c[%cix12] : memref<?xf32, 1>
          %oe12 = arith.addf %o_12, %zero : f32
          %bo12 = arith.mulf %beta, %oe12 : f32
          scf.yield %bo12 : f32
        }
        %v12 = math.fma %alpha, %res#6, %old12 : f32
        %vt12 = arith.addf %v12, %zero : f32
        memref.store %vt12, %c[%cix12] : memref<?xf32, 1>
      }
      %inr13 = arith.cmpi slt, %row1, %m : i32
      %inc13 = arith.cmpi slt, %col3, %n : i32
      %in13 = arith.andi %inr13, %inc13 : i1
      scf.if %in13 {
        %cl13 = arith.muli %col3, %ldc : i32
        %cc13 = arith.addi %cbase, %cl13 : i32
        %ci13 = arith.addi %cc13, %row1 : i32
        %cix13 = arith.index_cast %ci13 : i32 to index
        %old13 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_13 = memref.load %c[%cix13] : memref<?xf32, 1>
          %oe13 = arith.addf %o_13, %zero : f32
          %bo13 = arith.mulf %beta, %oe13 : f32
          scf.yield %bo13 : f32
        }
        %v13 = math.fma %alpha, %res#7, %old13 : f32
        %vt13 = arith.addf %v13, %zero : f32
        memref.store %vt13, %c[%cix13] : memref<?xf32, 1>
      }
      %inr20 = arith.cmpi slt, %row2, %m : i32
      %inc20 = arith.cmpi slt, %col0, %n : i32
      %in20 = arith.andi %inr20, %inc20 : i1
      scf.if %in20 {
        %cl20 = arith.muli %col0, %ldc : i32
        %cc20 = arith.addi %cbase, %cl20 : i32
        %ci20 = arith.addi %cc20, %row2 : i32
        %cix20 = arith.index_cast %ci20 : i32 to index
        %old20 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_20 = memref.load %c[%cix20] : memref<?xf32, 1>
          %oe20 = arith.addf %o_20, %zero : f32
          %bo20 = arith.mulf %beta, %oe20 : f32
          scf.yield %bo20 : f32
        }
        %v20 = math.fma %alpha, %res#8, %old20 : f32
        %vt20 = arith.addf %v20, %zero : f32
        memref.store %vt20, %c[%cix20] : memref<?xf32, 1>
      }
      %inr21 = arith.cmpi slt, %row2, %m : i32
      %inc21 = arith.cmpi slt, %col1, %n : i32
      %in21 = arith.andi %inr21, %inc21 : i1
      scf.if %in21 {
        %cl21 = arith.muli %col1, %ldc : i32
        %cc21 = arith.addi %cbase, %cl21 : i32
        %ci21 = arith.addi %cc21, %row2 : i32
        %cix21 = arith.index_cast %ci21 : i32 to index
        %old21 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_21 = memref.load %c[%cix21] : memref<?xf32, 1>
          %oe21 = arith.addf %o_21, %zero : f32
          %bo21 = arith.mulf %beta, %oe21 : f32
          scf.yield %bo21 : f32
        }
        %v21 = math.fma %alpha, %res#9, %old21 : f32
        %vt21 = arith.addf %v21, %zero : f32
        memref.store %vt21, %c[%cix21] : memref<?xf32, 1>
      }
      %inr22 = arith.cmpi slt, %row2, %m : i32
      %inc22 = arith.cmpi slt, %col2, %n : i32
      %in22 = arith.andi %inr22, %inc22 : i1
      scf.if %in22 {
        %cl22 = arith.muli %col2, %ldc : i32
        %cc22 = arith.addi %cbase, %cl22 : i32
        %ci22 = arith.addi %cc22, %row2 : i32
        %cix22 = arith.index_cast %ci22 : i32 to index
        %old22 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_22 = memref.load %c[%cix22] : memref<?xf32, 1>
          %oe22 = arith.addf %o_22, %zero : f32
          %bo22 = arith.mulf %beta, %oe22 : f32
          scf.yield %bo22 : f32
        }
        %v22 = math.fma %alpha, %res#10, %old22 : f32
        %vt22 = arith.addf %v22, %zero : f32
        memref.store %vt22, %c[%cix22] : memref<?xf32, 1>
      }
      %inr23 = arith.cmpi slt, %row2, %m : i32
      %inc23 = arith.cmpi slt, %col3, %n : i32
      %in23 = arith.andi %inr23, %inc23 : i1
      scf.if %in23 {
        %cl23 = arith.muli %col3, %ldc : i32
        %cc23 = arith.addi %cbase, %cl23 : i32
        %ci23 = arith.addi %cc23, %row2 : i32
        %cix23 = arith.index_cast %ci23 : i32 to index
        %old23 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_23 = memref.load %c[%cix23] : memref<?xf32, 1>
          %oe23 = arith.addf %o_23, %zero : f32
          %bo23 = arith.mulf %beta, %oe23 : f32
          scf.yield %bo23 : f32
        }
        %v23 = math.fma %alpha, %res#11, %old23 : f32
        %vt23 = arith.addf %v23, %zero : f32
        memref.store %vt23, %c[%cix23] : memref<?xf32, 1>
      }
      %inr30 = arith.cmpi slt, %row3, %m : i32
      %inc30 = arith.cmpi slt, %col0, %n : i32
      %in30 = arith.andi %inr30, %inc30 : i1
      scf.if %in30 {
        %cl30 = arith.muli %col0, %ldc : i32
        %cc30 = arith.addi %cbase, %cl30 : i32
        %ci30 = arith.addi %cc30, %row3 : i32
        %cix30 = arith.index_cast %ci30 : i32 to index
        %old30 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_30 = memref.load %c[%cix30] : memref<?xf32, 1>
          %oe30 = arith.addf %o_30, %zero : f32
          %bo30 = arith.mulf %beta, %oe30 : f32
          scf.yield %bo30 : f32
        }
        %v30 = math.fma %alpha, %res#12, %old30 : f32
        %vt30 = arith.addf %v30, %zero : f32
        memref.store %vt30, %c[%cix30] : memref<?xf32, 1>
      }
      %inr31 = arith.cmpi slt, %row3, %m : i32
      %inc31 = arith.cmpi slt, %col1, %n : i32
      %in31 = arith.andi %inr31, %inc31 : i1
      scf.if %in31 {
        %cl31 = arith.muli %col1, %ldc : i32
        %cc31 = arith.addi %cbase, %cl31 : i32
        %ci31 = arith.addi %cc31, %row3 : i32
        %cix31 = arith.index_cast %ci31 : i32 to index
        %old31 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_31 = memref.load %c[%cix31] : memref<?xf32, 1>
          %oe31 = arith.addf %o_31, %zero : f32
          %bo31 = arith.mulf %beta, %oe31 : f32
          scf.yield %bo31 : f32
        }
        %v31 = math.fma %alpha, %res#13, %old31 : f32
        %vt31 = arith.addf %v31, %zero : f32
        memref.store %vt31, %c[%cix31] : memref<?xf32, 1>
      }
      %inr32 = arith.cmpi slt, %row3, %m : i32
      %inc32 = arith.cmpi slt, %col2, %n : i32
      %in32 = arith.andi %inr32, %inc32 : i1
      scf.if %in32 {
        %cl32 = arith.muli %col2, %ldc : i32
        %cc32 = arith.addi %cbase, %cl32 : i32
        %ci32 = arith.addi %cc32, %row3 : i32
        %cix32 = arith.index_cast %ci32 : i32 to index
        %old32 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_32 = memref.load %c[%cix32] : memref<?xf32, 1>
          %oe32 = arith.addf %o_32, %zero : f32
          %bo32 = arith.mulf %beta, %oe32 : f32
          scf.yield %bo32 : f32
        }
        %v32 = math.fma %alpha, %res#14, %old32 : f32
        %vt32 = arith.addf %v32, %zero : f32
        memref.store %vt32, %c[%cix32] : memref<?xf32, 1>
      }
      %inr33 = arith.cmpi slt, %row3, %m : i32
      %inc33 = arith.cmpi slt, %col3, %n : i32
      %in33 = arith.andi %inr33, %inc33 : i1
      scf.if %in33 {
        %cl33 = arith.muli %col3, %ldc : i32
        %cc33 = arith.addi %cbase, %cl33 : i32
        %ci33 = arith.addi %cc33, %row3 : i32
        %cix33 = arith.index_cast %ci33 : i32 to index
        %old33 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_33 = memref.load %c[%cix33] : memref<?xf32, 1>
          %oe33 = arith.addf %o_33, %zero : f32
          %bo33 = arith.mulf %beta, %oe33 : f32
          scf.yield %bo33 : f32
        }
        %v33 = math.fma %alpha, %res#15, %old33 : f32
        %vt33 = arith.addf %v33, %zero : f32
        memref.store %vt33, %c[%cix33] : memref<?xf32, 1>
      }
      gpu.return
    }
    gpu.func @gemm_bf16_bf16(%m: i32, %n: i32, %k: i32, %alpha: f32, %a: memref<?xbf16, 1>, %ao: i32, %ars: i32, %acs: i32, %sa: i32, %b: memref<?xbf16, 1>, %bo: i32, %brs: i32, %bcs: i32, %sb: i32, %beta: f32, %c: memref<?xbf16, 1>, %co: i32, %ldc: i32, %sc: i32) kernel {
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
      %zero = arith.constant 0.0 : f32
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
      %cbase = arith.addi %co, %zsc : i32
      %mm1 = arith.subi %m, %c1 : i32
      %nm1 = arith.subi %n, %c1 : i32
      %cr0 = arith.constant 0 : i32
      %ro0 = arith.muli %cr0, %dxi : i32
      %row0 = arith.addi %i0, %ro0 : i32
      %rowc0 = arith.minsi %row0, %mm1 : i32
      %rofs0 = arith.muli %rowc0, %ars : i32
      %arow0 = arith.addi %abase, %rofs0 : i32
      %co0 = arith.muli %cr0, %dyi : i32
      %col0 = arith.addi %j0, %co0 : i32
      %colc0 = arith.minsi %col0, %nm1 : i32
      %cofs0 = arith.muli %colc0, %bcs : i32
      %bcol0 = arith.addi %bbase, %cofs0 : i32
      %cr1 = arith.constant 1 : i32
      %ro1 = arith.muli %cr1, %dxi : i32
      %row1 = arith.addi %i0, %ro1 : i32
      %rowc1 = arith.minsi %row1, %mm1 : i32
      %rofs1 = arith.muli %rowc1, %ars : i32
      %arow1 = arith.addi %abase, %rofs1 : i32
      %co1 = arith.muli %cr1, %dyi : i32
      %col1 = arith.addi %j0, %co1 : i32
      %colc1 = arith.minsi %col1, %nm1 : i32
      %cofs1 = arith.muli %colc1, %bcs : i32
      %bcol1 = arith.addi %bbase, %cofs1 : i32
      %cr2 = arith.constant 2 : i32
      %ro2 = arith.muli %cr2, %dxi : i32
      %row2 = arith.addi %i0, %ro2 : i32
      %rowc2 = arith.minsi %row2, %mm1 : i32
      %rofs2 = arith.muli %rowc2, %ars : i32
      %arow2 = arith.addi %abase, %rofs2 : i32
      %co2 = arith.muli %cr2, %dyi : i32
      %col2 = arith.addi %j0, %co2 : i32
      %colc2 = arith.minsi %col2, %nm1 : i32
      %cofs2 = arith.muli %colc2, %bcs : i32
      %bcol2 = arith.addi %bbase, %cofs2 : i32
      %cr3 = arith.constant 3 : i32
      %ro3 = arith.muli %cr3, %dxi : i32
      %row3 = arith.addi %i0, %ro3 : i32
      %rowc3 = arith.minsi %row3, %mm1 : i32
      %rofs3 = arith.muli %rowc3, %ars : i32
      %arow3 = arith.addi %abase, %rofs3 : i32
      %co3 = arith.muli %cr3, %dyi : i32
      %col3 = arith.addi %j0, %co3 : i32
      %colc3 = arith.minsi %col3, %nm1 : i32
      %cofs3 = arith.muli %colc3, %bcs : i32
      %bcol3 = arith.addi %bbase, %cofs3 : i32
      %res:16 = scf.for %l = %c0 to %k step %c1 iter_args(%s00 = %zero, %s01 = %zero, %s02 = %zero, %s03 = %zero, %s10 = %zero, %s11 = %zero, %s12 = %zero, %s13 = %zero, %s20 = %zero, %s21 = %zero, %s22 = %zero, %s23 = %zero, %s30 = %zero, %s31 = %zero, %s32 = %zero, %s33 = %zero) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) : i32 {
        %la = arith.muli %l, %acs : i32
        %lb = arith.muli %l, %brs : i32
        %ai0 = arith.addi %arow0, %la : i32
        %aix0 = arith.index_cast %ai0 : i32 to index
        %av0_ = memref.load %a[%aix0] : memref<?xbf16, 1>
        %av0 = arith.extf %av0_ : bf16 to f32
        %bi0 = arith.addi %bcol0, %lb : i32
        %bix0 = arith.index_cast %bi0 : i32 to index
        %bv0_ = memref.load %b[%bix0] : memref<?xbf16, 1>
        %bv0 = arith.extf %bv0_ : bf16 to f32
        %ai1 = arith.addi %arow1, %la : i32
        %aix1 = arith.index_cast %ai1 : i32 to index
        %av1_ = memref.load %a[%aix1] : memref<?xbf16, 1>
        %av1 = arith.extf %av1_ : bf16 to f32
        %bi1 = arith.addi %bcol1, %lb : i32
        %bix1 = arith.index_cast %bi1 : i32 to index
        %bv1_ = memref.load %b[%bix1] : memref<?xbf16, 1>
        %bv1 = arith.extf %bv1_ : bf16 to f32
        %ai2 = arith.addi %arow2, %la : i32
        %aix2 = arith.index_cast %ai2 : i32 to index
        %av2_ = memref.load %a[%aix2] : memref<?xbf16, 1>
        %av2 = arith.extf %av2_ : bf16 to f32
        %bi2 = arith.addi %bcol2, %lb : i32
        %bix2 = arith.index_cast %bi2 : i32 to index
        %bv2_ = memref.load %b[%bix2] : memref<?xbf16, 1>
        %bv2 = arith.extf %bv2_ : bf16 to f32
        %ai3 = arith.addi %arow3, %la : i32
        %aix3 = arith.index_cast %ai3 : i32 to index
        %av3_ = memref.load %a[%aix3] : memref<?xbf16, 1>
        %av3 = arith.extf %av3_ : bf16 to f32
        %bi3 = arith.addi %bcol3, %lb : i32
        %bix3 = arith.index_cast %bi3 : i32 to index
        %bv3_ = memref.load %b[%bix3] : memref<?xbf16, 1>
        %bv3 = arith.extf %bv3_ : bf16 to f32
        %n00 = math.fma %av0, %bv0, %s00 : f32
        %n01 = math.fma %av0, %bv1, %s01 : f32
        %n02 = math.fma %av0, %bv2, %s02 : f32
        %n03 = math.fma %av0, %bv3, %s03 : f32
        %n10 = math.fma %av1, %bv0, %s10 : f32
        %n11 = math.fma %av1, %bv1, %s11 : f32
        %n12 = math.fma %av1, %bv2, %s12 : f32
        %n13 = math.fma %av1, %bv3, %s13 : f32
        %n20 = math.fma %av2, %bv0, %s20 : f32
        %n21 = math.fma %av2, %bv1, %s21 : f32
        %n22 = math.fma %av2, %bv2, %s22 : f32
        %n23 = math.fma %av2, %bv3, %s23 : f32
        %n30 = math.fma %av3, %bv0, %s30 : f32
        %n31 = math.fma %av3, %bv1, %s31 : f32
        %n32 = math.fma %av3, %bv2, %s32 : f32
        %n33 = math.fma %av3, %bv3, %s33 : f32
        scf.yield %n00, %n01, %n02, %n03, %n10, %n11, %n12, %n13, %n20, %n21, %n22, %n23, %n30, %n31, %n32, %n33 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
      }
      %bzero = arith.cmpf oeq, %beta, %zero : f32
      %inr00 = arith.cmpi slt, %row0, %m : i32
      %inc00 = arith.cmpi slt, %col0, %n : i32
      %in00 = arith.andi %inr00, %inc00 : i1
      scf.if %in00 {
        %cl00 = arith.muli %col0, %ldc : i32
        %cc00 = arith.addi %cbase, %cl00 : i32
        %ci00 = arith.addi %cc00, %row0 : i32
        %cix00 = arith.index_cast %ci00 : i32 to index
        %old00 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_00 = memref.load %c[%cix00] : memref<?xbf16, 1>
          %oe00 = arith.extf %o_00 : bf16 to f32
          %bo00 = arith.mulf %beta, %oe00 : f32
          scf.yield %bo00 : f32
        }
        %v00 = math.fma %alpha, %res#0, %old00 : f32
        %vt00 = arith.truncf %v00 : f32 to bf16
        memref.store %vt00, %c[%cix00] : memref<?xbf16, 1>
      }
      %inr01 = arith.cmpi slt, %row0, %m : i32
      %inc01 = arith.cmpi slt, %col1, %n : i32
      %in01 = arith.andi %inr01, %inc01 : i1
      scf.if %in01 {
        %cl01 = arith.muli %col1, %ldc : i32
        %cc01 = arith.addi %cbase, %cl01 : i32
        %ci01 = arith.addi %cc01, %row0 : i32
        %cix01 = arith.index_cast %ci01 : i32 to index
        %old01 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_01 = memref.load %c[%cix01] : memref<?xbf16, 1>
          %oe01 = arith.extf %o_01 : bf16 to f32
          %bo01 = arith.mulf %beta, %oe01 : f32
          scf.yield %bo01 : f32
        }
        %v01 = math.fma %alpha, %res#1, %old01 : f32
        %vt01 = arith.truncf %v01 : f32 to bf16
        memref.store %vt01, %c[%cix01] : memref<?xbf16, 1>
      }
      %inr02 = arith.cmpi slt, %row0, %m : i32
      %inc02 = arith.cmpi slt, %col2, %n : i32
      %in02 = arith.andi %inr02, %inc02 : i1
      scf.if %in02 {
        %cl02 = arith.muli %col2, %ldc : i32
        %cc02 = arith.addi %cbase, %cl02 : i32
        %ci02 = arith.addi %cc02, %row0 : i32
        %cix02 = arith.index_cast %ci02 : i32 to index
        %old02 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_02 = memref.load %c[%cix02] : memref<?xbf16, 1>
          %oe02 = arith.extf %o_02 : bf16 to f32
          %bo02 = arith.mulf %beta, %oe02 : f32
          scf.yield %bo02 : f32
        }
        %v02 = math.fma %alpha, %res#2, %old02 : f32
        %vt02 = arith.truncf %v02 : f32 to bf16
        memref.store %vt02, %c[%cix02] : memref<?xbf16, 1>
      }
      %inr03 = arith.cmpi slt, %row0, %m : i32
      %inc03 = arith.cmpi slt, %col3, %n : i32
      %in03 = arith.andi %inr03, %inc03 : i1
      scf.if %in03 {
        %cl03 = arith.muli %col3, %ldc : i32
        %cc03 = arith.addi %cbase, %cl03 : i32
        %ci03 = arith.addi %cc03, %row0 : i32
        %cix03 = arith.index_cast %ci03 : i32 to index
        %old03 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_03 = memref.load %c[%cix03] : memref<?xbf16, 1>
          %oe03 = arith.extf %o_03 : bf16 to f32
          %bo03 = arith.mulf %beta, %oe03 : f32
          scf.yield %bo03 : f32
        }
        %v03 = math.fma %alpha, %res#3, %old03 : f32
        %vt03 = arith.truncf %v03 : f32 to bf16
        memref.store %vt03, %c[%cix03] : memref<?xbf16, 1>
      }
      %inr10 = arith.cmpi slt, %row1, %m : i32
      %inc10 = arith.cmpi slt, %col0, %n : i32
      %in10 = arith.andi %inr10, %inc10 : i1
      scf.if %in10 {
        %cl10 = arith.muli %col0, %ldc : i32
        %cc10 = arith.addi %cbase, %cl10 : i32
        %ci10 = arith.addi %cc10, %row1 : i32
        %cix10 = arith.index_cast %ci10 : i32 to index
        %old10 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_10 = memref.load %c[%cix10] : memref<?xbf16, 1>
          %oe10 = arith.extf %o_10 : bf16 to f32
          %bo10 = arith.mulf %beta, %oe10 : f32
          scf.yield %bo10 : f32
        }
        %v10 = math.fma %alpha, %res#4, %old10 : f32
        %vt10 = arith.truncf %v10 : f32 to bf16
        memref.store %vt10, %c[%cix10] : memref<?xbf16, 1>
      }
      %inr11 = arith.cmpi slt, %row1, %m : i32
      %inc11 = arith.cmpi slt, %col1, %n : i32
      %in11 = arith.andi %inr11, %inc11 : i1
      scf.if %in11 {
        %cl11 = arith.muli %col1, %ldc : i32
        %cc11 = arith.addi %cbase, %cl11 : i32
        %ci11 = arith.addi %cc11, %row1 : i32
        %cix11 = arith.index_cast %ci11 : i32 to index
        %old11 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_11 = memref.load %c[%cix11] : memref<?xbf16, 1>
          %oe11 = arith.extf %o_11 : bf16 to f32
          %bo11 = arith.mulf %beta, %oe11 : f32
          scf.yield %bo11 : f32
        }
        %v11 = math.fma %alpha, %res#5, %old11 : f32
        %vt11 = arith.truncf %v11 : f32 to bf16
        memref.store %vt11, %c[%cix11] : memref<?xbf16, 1>
      }
      %inr12 = arith.cmpi slt, %row1, %m : i32
      %inc12 = arith.cmpi slt, %col2, %n : i32
      %in12 = arith.andi %inr12, %inc12 : i1
      scf.if %in12 {
        %cl12 = arith.muli %col2, %ldc : i32
        %cc12 = arith.addi %cbase, %cl12 : i32
        %ci12 = arith.addi %cc12, %row1 : i32
        %cix12 = arith.index_cast %ci12 : i32 to index
        %old12 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_12 = memref.load %c[%cix12] : memref<?xbf16, 1>
          %oe12 = arith.extf %o_12 : bf16 to f32
          %bo12 = arith.mulf %beta, %oe12 : f32
          scf.yield %bo12 : f32
        }
        %v12 = math.fma %alpha, %res#6, %old12 : f32
        %vt12 = arith.truncf %v12 : f32 to bf16
        memref.store %vt12, %c[%cix12] : memref<?xbf16, 1>
      }
      %inr13 = arith.cmpi slt, %row1, %m : i32
      %inc13 = arith.cmpi slt, %col3, %n : i32
      %in13 = arith.andi %inr13, %inc13 : i1
      scf.if %in13 {
        %cl13 = arith.muli %col3, %ldc : i32
        %cc13 = arith.addi %cbase, %cl13 : i32
        %ci13 = arith.addi %cc13, %row1 : i32
        %cix13 = arith.index_cast %ci13 : i32 to index
        %old13 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_13 = memref.load %c[%cix13] : memref<?xbf16, 1>
          %oe13 = arith.extf %o_13 : bf16 to f32
          %bo13 = arith.mulf %beta, %oe13 : f32
          scf.yield %bo13 : f32
        }
        %v13 = math.fma %alpha, %res#7, %old13 : f32
        %vt13 = arith.truncf %v13 : f32 to bf16
        memref.store %vt13, %c[%cix13] : memref<?xbf16, 1>
      }
      %inr20 = arith.cmpi slt, %row2, %m : i32
      %inc20 = arith.cmpi slt, %col0, %n : i32
      %in20 = arith.andi %inr20, %inc20 : i1
      scf.if %in20 {
        %cl20 = arith.muli %col0, %ldc : i32
        %cc20 = arith.addi %cbase, %cl20 : i32
        %ci20 = arith.addi %cc20, %row2 : i32
        %cix20 = arith.index_cast %ci20 : i32 to index
        %old20 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_20 = memref.load %c[%cix20] : memref<?xbf16, 1>
          %oe20 = arith.extf %o_20 : bf16 to f32
          %bo20 = arith.mulf %beta, %oe20 : f32
          scf.yield %bo20 : f32
        }
        %v20 = math.fma %alpha, %res#8, %old20 : f32
        %vt20 = arith.truncf %v20 : f32 to bf16
        memref.store %vt20, %c[%cix20] : memref<?xbf16, 1>
      }
      %inr21 = arith.cmpi slt, %row2, %m : i32
      %inc21 = arith.cmpi slt, %col1, %n : i32
      %in21 = arith.andi %inr21, %inc21 : i1
      scf.if %in21 {
        %cl21 = arith.muli %col1, %ldc : i32
        %cc21 = arith.addi %cbase, %cl21 : i32
        %ci21 = arith.addi %cc21, %row2 : i32
        %cix21 = arith.index_cast %ci21 : i32 to index
        %old21 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_21 = memref.load %c[%cix21] : memref<?xbf16, 1>
          %oe21 = arith.extf %o_21 : bf16 to f32
          %bo21 = arith.mulf %beta, %oe21 : f32
          scf.yield %bo21 : f32
        }
        %v21 = math.fma %alpha, %res#9, %old21 : f32
        %vt21 = arith.truncf %v21 : f32 to bf16
        memref.store %vt21, %c[%cix21] : memref<?xbf16, 1>
      }
      %inr22 = arith.cmpi slt, %row2, %m : i32
      %inc22 = arith.cmpi slt, %col2, %n : i32
      %in22 = arith.andi %inr22, %inc22 : i1
      scf.if %in22 {
        %cl22 = arith.muli %col2, %ldc : i32
        %cc22 = arith.addi %cbase, %cl22 : i32
        %ci22 = arith.addi %cc22, %row2 : i32
        %cix22 = arith.index_cast %ci22 : i32 to index
        %old22 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_22 = memref.load %c[%cix22] : memref<?xbf16, 1>
          %oe22 = arith.extf %o_22 : bf16 to f32
          %bo22 = arith.mulf %beta, %oe22 : f32
          scf.yield %bo22 : f32
        }
        %v22 = math.fma %alpha, %res#10, %old22 : f32
        %vt22 = arith.truncf %v22 : f32 to bf16
        memref.store %vt22, %c[%cix22] : memref<?xbf16, 1>
      }
      %inr23 = arith.cmpi slt, %row2, %m : i32
      %inc23 = arith.cmpi slt, %col3, %n : i32
      %in23 = arith.andi %inr23, %inc23 : i1
      scf.if %in23 {
        %cl23 = arith.muli %col3, %ldc : i32
        %cc23 = arith.addi %cbase, %cl23 : i32
        %ci23 = arith.addi %cc23, %row2 : i32
        %cix23 = arith.index_cast %ci23 : i32 to index
        %old23 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_23 = memref.load %c[%cix23] : memref<?xbf16, 1>
          %oe23 = arith.extf %o_23 : bf16 to f32
          %bo23 = arith.mulf %beta, %oe23 : f32
          scf.yield %bo23 : f32
        }
        %v23 = math.fma %alpha, %res#11, %old23 : f32
        %vt23 = arith.truncf %v23 : f32 to bf16
        memref.store %vt23, %c[%cix23] : memref<?xbf16, 1>
      }
      %inr30 = arith.cmpi slt, %row3, %m : i32
      %inc30 = arith.cmpi slt, %col0, %n : i32
      %in30 = arith.andi %inr30, %inc30 : i1
      scf.if %in30 {
        %cl30 = arith.muli %col0, %ldc : i32
        %cc30 = arith.addi %cbase, %cl30 : i32
        %ci30 = arith.addi %cc30, %row3 : i32
        %cix30 = arith.index_cast %ci30 : i32 to index
        %old30 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_30 = memref.load %c[%cix30] : memref<?xbf16, 1>
          %oe30 = arith.extf %o_30 : bf16 to f32
          %bo30 = arith.mulf %beta, %oe30 : f32
          scf.yield %bo30 : f32
        }
        %v30 = math.fma %alpha, %res#12, %old30 : f32
        %vt30 = arith.truncf %v30 : f32 to bf16
        memref.store %vt30, %c[%cix30] : memref<?xbf16, 1>
      }
      %inr31 = arith.cmpi slt, %row3, %m : i32
      %inc31 = arith.cmpi slt, %col1, %n : i32
      %in31 = arith.andi %inr31, %inc31 : i1
      scf.if %in31 {
        %cl31 = arith.muli %col1, %ldc : i32
        %cc31 = arith.addi %cbase, %cl31 : i32
        %ci31 = arith.addi %cc31, %row3 : i32
        %cix31 = arith.index_cast %ci31 : i32 to index
        %old31 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_31 = memref.load %c[%cix31] : memref<?xbf16, 1>
          %oe31 = arith.extf %o_31 : bf16 to f32
          %bo31 = arith.mulf %beta, %oe31 : f32
          scf.yield %bo31 : f32
        }
        %v31 = math.fma %alpha, %res#13, %old31 : f32
        %vt31 = arith.truncf %v31 : f32 to bf16
        memref.store %vt31, %c[%cix31] : memref<?xbf16, 1>
      }
      %inr32 = arith.cmpi slt, %row3, %m : i32
      %inc32 = arith.cmpi slt, %col2, %n : i32
      %in32 = arith.andi %inr32, %inc32 : i1
      scf.if %in32 {
        %cl32 = arith.muli %col2, %ldc : i32
        %cc32 = arith.addi %cbase, %cl32 : i32
        %ci32 = arith.addi %cc32, %row3 : i32
        %cix32 = arith.index_cast %ci32 : i32 to index
        %old32 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_32 = memref.load %c[%cix32] : memref<?xbf16, 1>
          %oe32 = arith.extf %o_32 : bf16 to f32
          %bo32 = arith.mulf %beta, %oe32 : f32
          scf.yield %bo32 : f32
        }
        %v32 = math.fma %alpha, %res#14, %old32 : f32
        %vt32 = arith.truncf %v32 : f32 to bf16
        memref.store %vt32, %c[%cix32] : memref<?xbf16, 1>
      }
      %inr33 = arith.cmpi slt, %row3, %m : i32
      %inc33 = arith.cmpi slt, %col3, %n : i32
      %in33 = arith.andi %inr33, %inc33 : i1
      scf.if %in33 {
        %cl33 = arith.muli %col3, %ldc : i32
        %cc33 = arith.addi %cbase, %cl33 : i32
        %ci33 = arith.addi %cc33, %row3 : i32
        %cix33 = arith.index_cast %ci33 : i32 to index
        %old33 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_33 = memref.load %c[%cix33] : memref<?xbf16, 1>
          %oe33 = arith.extf %o_33 : bf16 to f32
          %bo33 = arith.mulf %beta, %oe33 : f32
          scf.yield %bo33 : f32
        }
        %v33 = math.fma %alpha, %res#15, %old33 : f32
        %vt33 = arith.truncf %v33 : f32 to bf16
        memref.store %vt33, %c[%cix33] : memref<?xbf16, 1>
      }
      gpu.return
    }
    gpu.func @gemm_bf16_f32(%m: i32, %n: i32, %k: i32, %alpha: f32, %a: memref<?xbf16, 1>, %ao: i32, %ars: i32, %acs: i32, %sa: i32, %b: memref<?xbf16, 1>, %bo: i32, %brs: i32, %bcs: i32, %sb: i32, %beta: f32, %c: memref<?xf32, 1>, %co: i32, %ldc: i32, %sc: i32) kernel {
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
      %zero = arith.constant 0.0 : f32
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
      %cbase = arith.addi %co, %zsc : i32
      %mm1 = arith.subi %m, %c1 : i32
      %nm1 = arith.subi %n, %c1 : i32
      %cr0 = arith.constant 0 : i32
      %ro0 = arith.muli %cr0, %dxi : i32
      %row0 = arith.addi %i0, %ro0 : i32
      %rowc0 = arith.minsi %row0, %mm1 : i32
      %rofs0 = arith.muli %rowc0, %ars : i32
      %arow0 = arith.addi %abase, %rofs0 : i32
      %co0 = arith.muli %cr0, %dyi : i32
      %col0 = arith.addi %j0, %co0 : i32
      %colc0 = arith.minsi %col0, %nm1 : i32
      %cofs0 = arith.muli %colc0, %bcs : i32
      %bcol0 = arith.addi %bbase, %cofs0 : i32
      %cr1 = arith.constant 1 : i32
      %ro1 = arith.muli %cr1, %dxi : i32
      %row1 = arith.addi %i0, %ro1 : i32
      %rowc1 = arith.minsi %row1, %mm1 : i32
      %rofs1 = arith.muli %rowc1, %ars : i32
      %arow1 = arith.addi %abase, %rofs1 : i32
      %co1 = arith.muli %cr1, %dyi : i32
      %col1 = arith.addi %j0, %co1 : i32
      %colc1 = arith.minsi %col1, %nm1 : i32
      %cofs1 = arith.muli %colc1, %bcs : i32
      %bcol1 = arith.addi %bbase, %cofs1 : i32
      %cr2 = arith.constant 2 : i32
      %ro2 = arith.muli %cr2, %dxi : i32
      %row2 = arith.addi %i0, %ro2 : i32
      %rowc2 = arith.minsi %row2, %mm1 : i32
      %rofs2 = arith.muli %rowc2, %ars : i32
      %arow2 = arith.addi %abase, %rofs2 : i32
      %co2 = arith.muli %cr2, %dyi : i32
      %col2 = arith.addi %j0, %co2 : i32
      %colc2 = arith.minsi %col2, %nm1 : i32
      %cofs2 = arith.muli %colc2, %bcs : i32
      %bcol2 = arith.addi %bbase, %cofs2 : i32
      %cr3 = arith.constant 3 : i32
      %ro3 = arith.muli %cr3, %dxi : i32
      %row3 = arith.addi %i0, %ro3 : i32
      %rowc3 = arith.minsi %row3, %mm1 : i32
      %rofs3 = arith.muli %rowc3, %ars : i32
      %arow3 = arith.addi %abase, %rofs3 : i32
      %co3 = arith.muli %cr3, %dyi : i32
      %col3 = arith.addi %j0, %co3 : i32
      %colc3 = arith.minsi %col3, %nm1 : i32
      %cofs3 = arith.muli %colc3, %bcs : i32
      %bcol3 = arith.addi %bbase, %cofs3 : i32
      %res:16 = scf.for %l = %c0 to %k step %c1 iter_args(%s00 = %zero, %s01 = %zero, %s02 = %zero, %s03 = %zero, %s10 = %zero, %s11 = %zero, %s12 = %zero, %s13 = %zero, %s20 = %zero, %s21 = %zero, %s22 = %zero, %s23 = %zero, %s30 = %zero, %s31 = %zero, %s32 = %zero, %s33 = %zero) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) : i32 {
        %la = arith.muli %l, %acs : i32
        %lb = arith.muli %l, %brs : i32
        %ai0 = arith.addi %arow0, %la : i32
        %aix0 = arith.index_cast %ai0 : i32 to index
        %av0_ = memref.load %a[%aix0] : memref<?xbf16, 1>
        %av0 = arith.extf %av0_ : bf16 to f32
        %bi0 = arith.addi %bcol0, %lb : i32
        %bix0 = arith.index_cast %bi0 : i32 to index
        %bv0_ = memref.load %b[%bix0] : memref<?xbf16, 1>
        %bv0 = arith.extf %bv0_ : bf16 to f32
        %ai1 = arith.addi %arow1, %la : i32
        %aix1 = arith.index_cast %ai1 : i32 to index
        %av1_ = memref.load %a[%aix1] : memref<?xbf16, 1>
        %av1 = arith.extf %av1_ : bf16 to f32
        %bi1 = arith.addi %bcol1, %lb : i32
        %bix1 = arith.index_cast %bi1 : i32 to index
        %bv1_ = memref.load %b[%bix1] : memref<?xbf16, 1>
        %bv1 = arith.extf %bv1_ : bf16 to f32
        %ai2 = arith.addi %arow2, %la : i32
        %aix2 = arith.index_cast %ai2 : i32 to index
        %av2_ = memref.load %a[%aix2] : memref<?xbf16, 1>
        %av2 = arith.extf %av2_ : bf16 to f32
        %bi2 = arith.addi %bcol2, %lb : i32
        %bix2 = arith.index_cast %bi2 : i32 to index
        %bv2_ = memref.load %b[%bix2] : memref<?xbf16, 1>
        %bv2 = arith.extf %bv2_ : bf16 to f32
        %ai3 = arith.addi %arow3, %la : i32
        %aix3 = arith.index_cast %ai3 : i32 to index
        %av3_ = memref.load %a[%aix3] : memref<?xbf16, 1>
        %av3 = arith.extf %av3_ : bf16 to f32
        %bi3 = arith.addi %bcol3, %lb : i32
        %bix3 = arith.index_cast %bi3 : i32 to index
        %bv3_ = memref.load %b[%bix3] : memref<?xbf16, 1>
        %bv3 = arith.extf %bv3_ : bf16 to f32
        %n00 = math.fma %av0, %bv0, %s00 : f32
        %n01 = math.fma %av0, %bv1, %s01 : f32
        %n02 = math.fma %av0, %bv2, %s02 : f32
        %n03 = math.fma %av0, %bv3, %s03 : f32
        %n10 = math.fma %av1, %bv0, %s10 : f32
        %n11 = math.fma %av1, %bv1, %s11 : f32
        %n12 = math.fma %av1, %bv2, %s12 : f32
        %n13 = math.fma %av1, %bv3, %s13 : f32
        %n20 = math.fma %av2, %bv0, %s20 : f32
        %n21 = math.fma %av2, %bv1, %s21 : f32
        %n22 = math.fma %av2, %bv2, %s22 : f32
        %n23 = math.fma %av2, %bv3, %s23 : f32
        %n30 = math.fma %av3, %bv0, %s30 : f32
        %n31 = math.fma %av3, %bv1, %s31 : f32
        %n32 = math.fma %av3, %bv2, %s32 : f32
        %n33 = math.fma %av3, %bv3, %s33 : f32
        scf.yield %n00, %n01, %n02, %n03, %n10, %n11, %n12, %n13, %n20, %n21, %n22, %n23, %n30, %n31, %n32, %n33 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
      }
      %bzero = arith.cmpf oeq, %beta, %zero : f32
      %inr00 = arith.cmpi slt, %row0, %m : i32
      %inc00 = arith.cmpi slt, %col0, %n : i32
      %in00 = arith.andi %inr00, %inc00 : i1
      scf.if %in00 {
        %cl00 = arith.muli %col0, %ldc : i32
        %cc00 = arith.addi %cbase, %cl00 : i32
        %ci00 = arith.addi %cc00, %row0 : i32
        %cix00 = arith.index_cast %ci00 : i32 to index
        %old00 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_00 = memref.load %c[%cix00] : memref<?xf32, 1>
          %oe00 = arith.addf %o_00, %zero : f32
          %bo00 = arith.mulf %beta, %oe00 : f32
          scf.yield %bo00 : f32
        }
        %v00 = math.fma %alpha, %res#0, %old00 : f32
        %vt00 = arith.addf %v00, %zero : f32
        memref.store %vt00, %c[%cix00] : memref<?xf32, 1>
      }
      %inr01 = arith.cmpi slt, %row0, %m : i32
      %inc01 = arith.cmpi slt, %col1, %n : i32
      %in01 = arith.andi %inr01, %inc01 : i1
      scf.if %in01 {
        %cl01 = arith.muli %col1, %ldc : i32
        %cc01 = arith.addi %cbase, %cl01 : i32
        %ci01 = arith.addi %cc01, %row0 : i32
        %cix01 = arith.index_cast %ci01 : i32 to index
        %old01 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_01 = memref.load %c[%cix01] : memref<?xf32, 1>
          %oe01 = arith.addf %o_01, %zero : f32
          %bo01 = arith.mulf %beta, %oe01 : f32
          scf.yield %bo01 : f32
        }
        %v01 = math.fma %alpha, %res#1, %old01 : f32
        %vt01 = arith.addf %v01, %zero : f32
        memref.store %vt01, %c[%cix01] : memref<?xf32, 1>
      }
      %inr02 = arith.cmpi slt, %row0, %m : i32
      %inc02 = arith.cmpi slt, %col2, %n : i32
      %in02 = arith.andi %inr02, %inc02 : i1
      scf.if %in02 {
        %cl02 = arith.muli %col2, %ldc : i32
        %cc02 = arith.addi %cbase, %cl02 : i32
        %ci02 = arith.addi %cc02, %row0 : i32
        %cix02 = arith.index_cast %ci02 : i32 to index
        %old02 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_02 = memref.load %c[%cix02] : memref<?xf32, 1>
          %oe02 = arith.addf %o_02, %zero : f32
          %bo02 = arith.mulf %beta, %oe02 : f32
          scf.yield %bo02 : f32
        }
        %v02 = math.fma %alpha, %res#2, %old02 : f32
        %vt02 = arith.addf %v02, %zero : f32
        memref.store %vt02, %c[%cix02] : memref<?xf32, 1>
      }
      %inr03 = arith.cmpi slt, %row0, %m : i32
      %inc03 = arith.cmpi slt, %col3, %n : i32
      %in03 = arith.andi %inr03, %inc03 : i1
      scf.if %in03 {
        %cl03 = arith.muli %col3, %ldc : i32
        %cc03 = arith.addi %cbase, %cl03 : i32
        %ci03 = arith.addi %cc03, %row0 : i32
        %cix03 = arith.index_cast %ci03 : i32 to index
        %old03 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_03 = memref.load %c[%cix03] : memref<?xf32, 1>
          %oe03 = arith.addf %o_03, %zero : f32
          %bo03 = arith.mulf %beta, %oe03 : f32
          scf.yield %bo03 : f32
        }
        %v03 = math.fma %alpha, %res#3, %old03 : f32
        %vt03 = arith.addf %v03, %zero : f32
        memref.store %vt03, %c[%cix03] : memref<?xf32, 1>
      }
      %inr10 = arith.cmpi slt, %row1, %m : i32
      %inc10 = arith.cmpi slt, %col0, %n : i32
      %in10 = arith.andi %inr10, %inc10 : i1
      scf.if %in10 {
        %cl10 = arith.muli %col0, %ldc : i32
        %cc10 = arith.addi %cbase, %cl10 : i32
        %ci10 = arith.addi %cc10, %row1 : i32
        %cix10 = arith.index_cast %ci10 : i32 to index
        %old10 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_10 = memref.load %c[%cix10] : memref<?xf32, 1>
          %oe10 = arith.addf %o_10, %zero : f32
          %bo10 = arith.mulf %beta, %oe10 : f32
          scf.yield %bo10 : f32
        }
        %v10 = math.fma %alpha, %res#4, %old10 : f32
        %vt10 = arith.addf %v10, %zero : f32
        memref.store %vt10, %c[%cix10] : memref<?xf32, 1>
      }
      %inr11 = arith.cmpi slt, %row1, %m : i32
      %inc11 = arith.cmpi slt, %col1, %n : i32
      %in11 = arith.andi %inr11, %inc11 : i1
      scf.if %in11 {
        %cl11 = arith.muli %col1, %ldc : i32
        %cc11 = arith.addi %cbase, %cl11 : i32
        %ci11 = arith.addi %cc11, %row1 : i32
        %cix11 = arith.index_cast %ci11 : i32 to index
        %old11 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_11 = memref.load %c[%cix11] : memref<?xf32, 1>
          %oe11 = arith.addf %o_11, %zero : f32
          %bo11 = arith.mulf %beta, %oe11 : f32
          scf.yield %bo11 : f32
        }
        %v11 = math.fma %alpha, %res#5, %old11 : f32
        %vt11 = arith.addf %v11, %zero : f32
        memref.store %vt11, %c[%cix11] : memref<?xf32, 1>
      }
      %inr12 = arith.cmpi slt, %row1, %m : i32
      %inc12 = arith.cmpi slt, %col2, %n : i32
      %in12 = arith.andi %inr12, %inc12 : i1
      scf.if %in12 {
        %cl12 = arith.muli %col2, %ldc : i32
        %cc12 = arith.addi %cbase, %cl12 : i32
        %ci12 = arith.addi %cc12, %row1 : i32
        %cix12 = arith.index_cast %ci12 : i32 to index
        %old12 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_12 = memref.load %c[%cix12] : memref<?xf32, 1>
          %oe12 = arith.addf %o_12, %zero : f32
          %bo12 = arith.mulf %beta, %oe12 : f32
          scf.yield %bo12 : f32
        }
        %v12 = math.fma %alpha, %res#6, %old12 : f32
        %vt12 = arith.addf %v12, %zero : f32
        memref.store %vt12, %c[%cix12] : memref<?xf32, 1>
      }
      %inr13 = arith.cmpi slt, %row1, %m : i32
      %inc13 = arith.cmpi slt, %col3, %n : i32
      %in13 = arith.andi %inr13, %inc13 : i1
      scf.if %in13 {
        %cl13 = arith.muli %col3, %ldc : i32
        %cc13 = arith.addi %cbase, %cl13 : i32
        %ci13 = arith.addi %cc13, %row1 : i32
        %cix13 = arith.index_cast %ci13 : i32 to index
        %old13 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_13 = memref.load %c[%cix13] : memref<?xf32, 1>
          %oe13 = arith.addf %o_13, %zero : f32
          %bo13 = arith.mulf %beta, %oe13 : f32
          scf.yield %bo13 : f32
        }
        %v13 = math.fma %alpha, %res#7, %old13 : f32
        %vt13 = arith.addf %v13, %zero : f32
        memref.store %vt13, %c[%cix13] : memref<?xf32, 1>
      }
      %inr20 = arith.cmpi slt, %row2, %m : i32
      %inc20 = arith.cmpi slt, %col0, %n : i32
      %in20 = arith.andi %inr20, %inc20 : i1
      scf.if %in20 {
        %cl20 = arith.muli %col0, %ldc : i32
        %cc20 = arith.addi %cbase, %cl20 : i32
        %ci20 = arith.addi %cc20, %row2 : i32
        %cix20 = arith.index_cast %ci20 : i32 to index
        %old20 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_20 = memref.load %c[%cix20] : memref<?xf32, 1>
          %oe20 = arith.addf %o_20, %zero : f32
          %bo20 = arith.mulf %beta, %oe20 : f32
          scf.yield %bo20 : f32
        }
        %v20 = math.fma %alpha, %res#8, %old20 : f32
        %vt20 = arith.addf %v20, %zero : f32
        memref.store %vt20, %c[%cix20] : memref<?xf32, 1>
      }
      %inr21 = arith.cmpi slt, %row2, %m : i32
      %inc21 = arith.cmpi slt, %col1, %n : i32
      %in21 = arith.andi %inr21, %inc21 : i1
      scf.if %in21 {
        %cl21 = arith.muli %col1, %ldc : i32
        %cc21 = arith.addi %cbase, %cl21 : i32
        %ci21 = arith.addi %cc21, %row2 : i32
        %cix21 = arith.index_cast %ci21 : i32 to index
        %old21 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_21 = memref.load %c[%cix21] : memref<?xf32, 1>
          %oe21 = arith.addf %o_21, %zero : f32
          %bo21 = arith.mulf %beta, %oe21 : f32
          scf.yield %bo21 : f32
        }
        %v21 = math.fma %alpha, %res#9, %old21 : f32
        %vt21 = arith.addf %v21, %zero : f32
        memref.store %vt21, %c[%cix21] : memref<?xf32, 1>
      }
      %inr22 = arith.cmpi slt, %row2, %m : i32
      %inc22 = arith.cmpi slt, %col2, %n : i32
      %in22 = arith.andi %inr22, %inc22 : i1
      scf.if %in22 {
        %cl22 = arith.muli %col2, %ldc : i32
        %cc22 = arith.addi %cbase, %cl22 : i32
        %ci22 = arith.addi %cc22, %row2 : i32
        %cix22 = arith.index_cast %ci22 : i32 to index
        %old22 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_22 = memref.load %c[%cix22] : memref<?xf32, 1>
          %oe22 = arith.addf %o_22, %zero : f32
          %bo22 = arith.mulf %beta, %oe22 : f32
          scf.yield %bo22 : f32
        }
        %v22 = math.fma %alpha, %res#10, %old22 : f32
        %vt22 = arith.addf %v22, %zero : f32
        memref.store %vt22, %c[%cix22] : memref<?xf32, 1>
      }
      %inr23 = arith.cmpi slt, %row2, %m : i32
      %inc23 = arith.cmpi slt, %col3, %n : i32
      %in23 = arith.andi %inr23, %inc23 : i1
      scf.if %in23 {
        %cl23 = arith.muli %col3, %ldc : i32
        %cc23 = arith.addi %cbase, %cl23 : i32
        %ci23 = arith.addi %cc23, %row2 : i32
        %cix23 = arith.index_cast %ci23 : i32 to index
        %old23 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_23 = memref.load %c[%cix23] : memref<?xf32, 1>
          %oe23 = arith.addf %o_23, %zero : f32
          %bo23 = arith.mulf %beta, %oe23 : f32
          scf.yield %bo23 : f32
        }
        %v23 = math.fma %alpha, %res#11, %old23 : f32
        %vt23 = arith.addf %v23, %zero : f32
        memref.store %vt23, %c[%cix23] : memref<?xf32, 1>
      }
      %inr30 = arith.cmpi slt, %row3, %m : i32
      %inc30 = arith.cmpi slt, %col0, %n : i32
      %in30 = arith.andi %inr30, %inc30 : i1
      scf.if %in30 {
        %cl30 = arith.muli %col0, %ldc : i32
        %cc30 = arith.addi %cbase, %cl30 : i32
        %ci30 = arith.addi %cc30, %row3 : i32
        %cix30 = arith.index_cast %ci30 : i32 to index
        %old30 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_30 = memref.load %c[%cix30] : memref<?xf32, 1>
          %oe30 = arith.addf %o_30, %zero : f32
          %bo30 = arith.mulf %beta, %oe30 : f32
          scf.yield %bo30 : f32
        }
        %v30 = math.fma %alpha, %res#12, %old30 : f32
        %vt30 = arith.addf %v30, %zero : f32
        memref.store %vt30, %c[%cix30] : memref<?xf32, 1>
      }
      %inr31 = arith.cmpi slt, %row3, %m : i32
      %inc31 = arith.cmpi slt, %col1, %n : i32
      %in31 = arith.andi %inr31, %inc31 : i1
      scf.if %in31 {
        %cl31 = arith.muli %col1, %ldc : i32
        %cc31 = arith.addi %cbase, %cl31 : i32
        %ci31 = arith.addi %cc31, %row3 : i32
        %cix31 = arith.index_cast %ci31 : i32 to index
        %old31 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_31 = memref.load %c[%cix31] : memref<?xf32, 1>
          %oe31 = arith.addf %o_31, %zero : f32
          %bo31 = arith.mulf %beta, %oe31 : f32
          scf.yield %bo31 : f32
        }
        %v31 = math.fma %alpha, %res#13, %old31 : f32
        %vt31 = arith.addf %v31, %zero : f32
        memref.store %vt31, %c[%cix31] : memref<?xf32, 1>
      }
      %inr32 = arith.cmpi slt, %row3, %m : i32
      %inc32 = arith.cmpi slt, %col2, %n : i32
      %in32 = arith.andi %inr32, %inc32 : i1
      scf.if %in32 {
        %cl32 = arith.muli %col2, %ldc : i32
        %cc32 = arith.addi %cbase, %cl32 : i32
        %ci32 = arith.addi %cc32, %row3 : i32
        %cix32 = arith.index_cast %ci32 : i32 to index
        %old32 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_32 = memref.load %c[%cix32] : memref<?xf32, 1>
          %oe32 = arith.addf %o_32, %zero : f32
          %bo32 = arith.mulf %beta, %oe32 : f32
          scf.yield %bo32 : f32
        }
        %v32 = math.fma %alpha, %res#14, %old32 : f32
        %vt32 = arith.addf %v32, %zero : f32
        memref.store %vt32, %c[%cix32] : memref<?xf32, 1>
      }
      %inr33 = arith.cmpi slt, %row3, %m : i32
      %inc33 = arith.cmpi slt, %col3, %n : i32
      %in33 = arith.andi %inr33, %inc33 : i1
      scf.if %in33 {
        %cl33 = arith.muli %col3, %ldc : i32
        %cc33 = arith.addi %cbase, %cl33 : i32
        %ci33 = arith.addi %cc33, %row3 : i32
        %cix33 = arith.index_cast %ci33 : i32 to index
        %old33 = scf.if %bzero -> (f32) {
          scf.yield %zero : f32
        } else {
          %o_33 = memref.load %c[%cix33] : memref<?xf32, 1>
          %oe33 = arith.addf %o_33, %zero : f32
          %bo33 = arith.mulf %beta, %oe33 : f32
          scf.yield %bo33 : f32
        }
        %v33 = math.fma %alpha, %res#15, %old33 : f32
        %vt33 = arith.addf %v33, %zero : f32
        memref.store %vt33, %c[%cix33] : memref<?xf32, 1>
      }
      gpu.return
    }
  }
}

