// Independent inference kernels. Standard MLIR only: lowerable through AIR or directly to AGX.
// No target-specific intrinsics or performance hints. f32 tensors use contiguous storage.
gpu.module @inference {
  gpu.func @agx_infer_unary(%x0: memref<1xf32, 1>, %y0: memref<1xf32, 1>, %n: i32, %op: i32) kernel {
    %big = arith.constant 2147483647 : index
    %x = memref.reinterpret_cast %x0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %y = memref.reinterpret_cast %y0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %id = gpu.global_id x
    %N = arith.index_cast %n : i32 to index
    %live = arith.cmpi ult, %id, %N : index
    scf.if %live {
      %v = memref.load %x[%id] : memref<?xf32, strided<[1]>, 1>
      %i0 = arith.constant 0 : i32
      %i1 = arith.constant 1 : i32
      %i2 = arith.constant 2 : i32
      %f0 = arith.constant 0.0 : f32
      %f1 = arith.constant 1.0 : f32
      %relu = arith.cmpi eq, %op, %i0 : i32
      %sigmoid = arith.cmpi eq, %op, %i1 : i32
      %tanh = arith.cmpi eq, %op, %i2 : i32
      %r = scf.if %relu -> f32 {
        %a = arith.maxnumf %v, %f0 : f32
        scf.yield %a : f32
      } else {
        %a = scf.if %sigmoid -> f32 {
          %neg = arith.negf %v : f32
          %e = math.exp %neg : f32
          %den = arith.addf %f1, %e : f32
          %s = arith.divf %f1, %den : f32
          scf.yield %s : f32
        } else {
          %b = scf.if %tanh -> f32 {
            %t = math.tanh %v : f32
            scf.yield %t : f32
          } else {
            %tag4 = arith.constant 4 : i32
            %is4 = arith.cmpi eq, %op, %tag4 : i32
            %result4 = scf.if %is4 -> f32 {
              %value4 = math.sqrt %v : f32
              scf.yield %value4 : f32
            } else {
              %tag5 = arith.constant 5 : i32
            %is5 = arith.cmpi eq, %op, %tag5 : i32
            %result5 = scf.if %is5 -> f32 {
              %value5 = math.exp %v : f32
              scf.yield %value5 : f32
            } else {
              %tag6 = arith.constant 6 : i32
            %is6 = arith.cmpi eq, %op, %tag6 : i32
            %result6 = scf.if %is6 -> f32 {
              %value6 = math.log %v : f32
              scf.yield %value6 : f32
            } else {
              %tag7 = arith.constant 7 : i32
            %is7 = arith.cmpi eq, %op, %tag7 : i32
            %result7 = scf.if %is7 -> f32 {
              %value7 = math.absf %v : f32
              scf.yield %value7 : f32
            } else {
              %tag8 = arith.constant 8 : i32
            %is8 = arith.cmpi eq, %op, %tag8 : i32
            %result8 = scf.if %is8 -> f32 {
              %value8 = arith.negf %v : f32
              scf.yield %value8 : f32
            } else {
              %tag9 = arith.constant 9 : i32
            %is9 = arith.cmpi eq, %op, %tag9 : i32
            %result9 = scf.if %is9 -> f32 {
              %value9 = arith.divf %f1, %v : f32
              scf.yield %value9 : f32
            } else {
              scf.yield %v : f32
            }
            scf.yield %result9 : f32
            }
            scf.yield %result8 : f32
            }
            scf.yield %result7 : f32
            }
            scf.yield %result6 : f32
            }
            scf.yield %result5 : f32
            }
            scf.yield %result4 : f32
          }
          scf.yield %b : f32
        }
        scf.yield %a : f32
      }
      memref.store %r, %y[%id] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }

  // Metadata: output dimensions, A broadcast strides, B broadcast strides (rank entries each).
  gpu.func @agx_infer_binary(%a0: memref<1xf32, 1>, %b0: memref<1xf32, 1>, %y0: memref<1xf32, 1>,
      %m0: memref<1xi32, 1>, %n: i32, %rank_: i32, %op: i32, %relu_: i32) kernel {
    %big = arith.constant 2147483647 : index
    %a = memref.reinterpret_cast %a0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %b = memref.reinterpret_cast %b0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %y = memref.reinterpret_cast %y0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %m = memref.reinterpret_cast %m0 to offset: [0], sizes: [%big], strides: [1] : memref<1xi32, 1> to memref<?xi32, strided<[1]>, 1>
    %id = gpu.global_id x
    %N = arith.index_cast %n : i32 to index
    %live = arith.cmpi ult, %id, %N : index
    scf.if %live {
      %z = arith.constant 0 : index
      %one = arith.constant 1 : index
      %rank = arith.index_cast %rank_ : i32 to index
      %rank2 = arith.addi %rank, %rank : index
      %aa, %bb, %qq = scf.for %d = %z to %rank step %one iter_args(%ai = %z, %bi = %z, %q = %id) -> (index, index, index) {
        %rev = arith.subi %rank, %d : index
        %axis = arith.subi %rev, %one : index
        %ad = arith.addi %axis, %rank : index
        %bd = arith.addi %axis, %rank2 : index
        %dim_ = memref.load %m[%axis] : memref<?xi32, strided<[1]>, 1>
        %as_ = memref.load %m[%ad] : memref<?xi32, strided<[1]>, 1>
        %bs_ = memref.load %m[%bd] : memref<?xi32, strided<[1]>, 1>
        %dim = arith.index_cast %dim_ : i32 to index
        %as = arith.index_cast %as_ : i32 to index
        %bs = arith.index_cast %bs_ : i32 to index
        %coord = arith.remui %q, %dim : index
        %next = arith.divui %q, %dim : index
        %ax = arith.muli %coord, %as : index
        %bx = arith.muli %coord, %bs : index
        %an = arith.addi %ai, %ax : index
        %bn = arith.addi %bi, %bx : index
        scf.yield %an, %bn, %next : index, index, index
      }
      %av = memref.load %a[%aa] : memref<?xf32, strided<[1]>, 1>
      %bv = memref.load %b[%bb] : memref<?xf32, strided<[1]>, 1>
      %i0 = arith.constant 0 : i32
      %i1 = arith.constant 1 : i32
      %i2 = arith.constant 2 : i32
      %i3 = arith.constant 3 : i32
      %i4 = arith.constant 4 : i32
      %add = arith.cmpi eq, %op, %i0 : i32
      %mul = arith.cmpi eq, %op, %i1 : i32
      %sub = arith.cmpi eq, %op, %i2 : i32
      %div = arith.cmpi eq, %op, %i3 : i32
      %max = arith.cmpi eq, %op, %i4 : i32
      %r = scf.if %add -> f32 {
        %v = arith.addf %av, %bv : f32
        scf.yield %v : f32
      } else {
        %v = scf.if %mul -> f32 {
          %w = arith.mulf %av, %bv : f32
          scf.yield %w : f32
        } else {
          %w = scf.if %sub -> f32 {
            %s = arith.subf %av, %bv : f32
            scf.yield %s : f32
          } else {
            %s = scf.if %div -> f32 {
              %t = arith.divf %av, %bv : f32
              scf.yield %t : f32
            } else {
              %t = scf.if %max -> f32 {
                %u = arith.maxnumf %av, %bv : f32
                scf.yield %u : f32
              } else {
                %u = arith.minnumf %av, %bv : f32
                scf.yield %u : f32
              }
              scf.yield %t : f32
            }
            scf.yield %s : f32
          }
          scf.yield %w : f32
        }
        scf.yield %v : f32
      }
      %f0 = arith.constant 0.0 : f32
      %rr = arith.maxnumf %r, %f0 : f32
      %relu = arith.cmpi ne, %relu_, %i0 : i32
      %out = arith.select %relu, %rr, %r : f32
      memref.store %out, %y[%id] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }

  // Each thread reduces one axis row, retaining a stable maximum before exponentiation.
  gpu.func @agx_infer_softmax(%x0: memref<1xf32, 1>, %y0: memref<1xf32, 1>, %rows_: i32, %channels_: i32, %inner_: i32) kernel {
    %big = arith.constant 2147483647 : index
    %x = memref.reinterpret_cast %x0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %y = memref.reinterpret_cast %y0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %row = gpu.global_id x
    %rows = arith.index_cast %rows_ : i32 to index
    %live = arith.cmpi ult, %row, %rows : index
    scf.if %live {
      %z = arith.constant 0 : index
      %one = arith.constant 1 : index
      %channels = arith.index_cast %channels_ : i32 to index
      %inner = arith.index_cast %inner_ : i32 to index
      %outer = arith.divui %row, %inner : index
      %in = arith.remui %row, %inner : index
      %span = arith.muli %channels, %inner : index
      %base0 = arith.muli %outer, %span : index
      %base = arith.addi %base0, %in : index
      %ninf = arith.constant 0xFF800000 : f32
      %f0 = arith.constant 0.0 : f32
      %max = scf.for %c = %z to %channels step %one iter_args(%m = %ninf) -> f32 {
        %ci = arith.muli %c, %inner : index
        %i = arith.addi %base, %ci : index
        %v = memref.load %x[%i] : memref<?xf32, strided<[1]>, 1>
        %next = arith.maxnumf %v, %m : f32
        scf.yield %next : f32
      }
      %sum = scf.for %c = %z to %channels step %one iter_args(%s = %f0) -> f32 {
        %ci = arith.muli %c, %inner : index
        %i = arith.addi %base, %ci : index
        %v = memref.load %x[%i] : memref<?xf32, strided<[1]>, 1>
        %shift = arith.subf %v, %max : f32
        %e = math.exp %shift : f32
        %next = arith.addf %s, %e : f32
        scf.yield %next : f32
      }
      scf.for %c = %z to %channels step %one {
        %ci = arith.muli %c, %inner : index
        %i = arith.addi %base, %ci : index
        %v = memref.load %x[%i] : memref<?xf32, strided<[1]>, 1>
        %shift = arith.subf %v, %max : f32
        %e = math.exp %shift : f32
        %p = arith.divf %e, %sum : f32
        memref.store %p, %y[%i] : memref<?xf32, strided<[1]>, 1>
      }
    }
    gpu.return
  }

  // Output dimensions and permuted input strides, rank entries each.
  gpu.func @agx_infer_transpose(%x0: memref<1xf32, 1>, %y0: memref<1xf32, 1>, %m0: memref<1xi32, 1>, %n: i32, %rank_: i32) kernel {
    %big = arith.constant 2147483647 : index
    %x = memref.reinterpret_cast %x0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %y = memref.reinterpret_cast %y0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %m = memref.reinterpret_cast %m0 to offset: [0], sizes: [%big], strides: [1] : memref<1xi32, 1> to memref<?xi32, strided<[1]>, 1>
    %id = gpu.global_id x
    %N = arith.index_cast %n : i32 to index
    %live = arith.cmpi ult, %id, %N : index
    scf.if %live {
      %z = arith.constant 0 : index
      %one = arith.constant 1 : index
      %rank = arith.index_cast %rank_ : i32 to index
      %addr, %qq = scf.for %d = %z to %rank step %one iter_args(%a = %z, %q = %id) -> (index, index) {
        %rev = arith.subi %rank, %d : index
        %axis = arith.subi %rev, %one : index
        %si = arith.addi %axis, %rank : index
        %dim_ = memref.load %m[%axis] : memref<?xi32, strided<[1]>, 1>
        %s_ = memref.load %m[%si] : memref<?xi32, strided<[1]>, 1>
        %dim = arith.index_cast %dim_ : i32 to index
        %s = arith.index_cast %s_ : i32 to index
        %coord = arith.remui %q, %dim : index
        %next = arith.divui %q, %dim : index
        %offset = arith.muli %coord, %s : index
        %a1 = arith.addi %a, %offset : index
        scf.yield %a1, %next : index, index
      }
      %v = memref.load %x[%addr] : memref<?xf32, strided<[1]>, 1>
      memref.store %v, %y[%id] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }

  gpu.func @agx_infer_concat(%x0: memref<1xf32, 1>, %y0: memref<1xf32, 1>, %n: i32, %chunk_: i32, %outspan_: i32, %offset_: i32) kernel {
    %big = arith.constant 2147483647 : index
    %x = memref.reinterpret_cast %x0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %y = memref.reinterpret_cast %y0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %id = gpu.global_id x
    %N = arith.index_cast %n : i32 to index
    %live = arith.cmpi ult, %id, %N : index
    scf.if %live {
      %chunk = arith.index_cast %chunk_ : i32 to index
      %outspan = arith.index_cast %outspan_ : i32 to index
      %offset = arith.index_cast %offset_ : i32 to index
      %outer = arith.divui %id, %chunk : index
      %inner = arith.remui %id, %chunk : index
      %base0 = arith.muli %outer, %outspan : index
      %base = arith.addi %base0, %offset : index
      %dest = arith.addi %base, %inner : index
      %v = memref.load %x[%id] : memref<?xf32, strided<[1]>, 1>
      memref.store %v, %y[%dest] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }

  gpu.func @agx_infer_pool(%x0: memref<1xf32, 1>, %y0: memref<1xf32, 1>, %n_: i32,
      %H_: i32, %W_: i32, %P_: i32, %Q_: i32, %R_: i32, %S_: i32,
      %ph_: i32, %pw_: i32, %sh_: i32, %sw_: i32, %kind: i32, %include_: i32) kernel {
    %big = arith.constant 2147483647 : index
    %x = memref.reinterpret_cast %x0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %y = memref.reinterpret_cast %y0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %id = gpu.global_id x
    %N = arith.index_cast %n_ : i32 to index
    %live = arith.cmpi ult, %id, %N : index
    scf.if %live {
      %z = arith.constant 0 : index
      %one = arith.constant 1 : index
      %H = arith.index_cast %H_ : i32 to index
      %W = arith.index_cast %W_ : i32 to index
      %P = arith.index_cast %P_ : i32 to index
      %Q = arith.index_cast %Q_ : i32 to index
      %R = arith.index_cast %R_ : i32 to index
      %S = arith.index_cast %S_ : i32 to index
      %ph = arith.index_cast %ph_ : i32 to index
      %pw = arith.index_cast %pw_ : i32 to index
      %sh = arith.index_cast %sh_ : i32 to index
      %sw = arith.index_cast %sw_ : i32 to index
      %q = arith.remui %id, %Q : index
      %tmp = arith.divui %id, %Q : index
      %p = arith.remui %tmp, %P : index
      %nc = arith.divui %tmp, %P : index
      %hp = arith.muli %p, %sh : index
      %wp = arith.muli %q, %sw : index
      %h0 = arith.subi %hp, %ph : index
      %w0 = arith.subi %wp, %pw : index
      %hw = arith.muli %H, %W : index
      %base = arith.muli %nc, %hw : index
      %i0 = arith.constant 0 : i32
      %ismax = arith.cmpi eq, %kind, %i0 : i32
      %include = arith.cmpi ne, %include_, %i0 : i32
      %f0 = arith.constant 0.0 : f32
      %ninf = arith.constant 0xFF800000 : f32
      %init = arith.select %ismax, %ninf, %f0 : f32
      %acc, %count = scf.for %r = %z to %R step %one iter_args(%acc0 = %init, %cnt0 = %z) -> (f32, index) {
        %h = arith.addi %h0, %r : index
        %acc1, %cnt1 = scf.for %s = %z to %S step %one iter_args(%acc2 = %acc0, %cnt2 = %cnt0) -> (f32, index) {
          %w = arith.addi %w0, %s : index
          %hlive = arith.cmpi ult, %h, %H : index
          %wlive = arith.cmpi ult, %w, %W : index
          %valid = arith.andi %hlive, %wlive : i1
          %next, %cntnext = scf.if %valid -> (f32, index) {
            %hi = arith.muli %h, %W : index
            %i0_ = arith.addi %base, %hi : index
            %i = arith.addi %i0_, %w : index
            %v = memref.load %x[%i] : memref<?xf32, strided<[1]>, 1>
            %max = arith.maxnumf %acc2, %v : f32
            %sum = arith.addf %acc2, %v : f32
            %out = arith.select %ismax, %max, %sum : f32
            %n = arith.addi %cnt2, %one : index
            scf.yield %out, %n : f32, index
          } else {
            scf.yield %acc2, %cnt2 : f32, index
          }
          scf.yield %next, %cntnext : f32, index
        }
        scf.yield %acc1, %cnt1 : f32, index
      }
      %area = arith.muli %R, %S : index
      %den = arith.select %include, %area, %count : index
      %safe = arith.maxui %den, %one : index
      %deni = arith.index_cast %safe : index to i32
      %denf = arith.uitofp %deni : i32 to f32
      %avg = arith.divf %acc, %denf : f32
      %result = arith.select %ismax, %acc, %avg : f32
      memref.store %result, %y[%id] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }

  // Contiguous long rows: a portable SIMD-group cooperates on each row.
  gpu.func @agx_infer_softmax_simd(%x0: memref<1xf32, 1>, %y0: memref<1xf32, 1>, %rows_: i32, %channels_: i32) kernel {
    %big = arith.constant 2147483647 : index
    %x = memref.reinterpret_cast %x0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %y = memref.reinterpret_cast %y0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %id = gpu.global_id x
    %width = gpu.subgroup_size : index
    %lane = gpu.lane_id
    %row = arith.divui %id, %width : index
    %rows = arith.index_cast %rows_ : i32 to index
    %live = arith.cmpi ult, %row, %rows : index
    scf.if %live {
      %channels = arith.index_cast %channels_ : i32 to index
      %base = arith.muli %row, %channels : index
      %ninf = arith.constant 0xFF800000 : f32
      %f0 = arith.constant 0.0 : f32
      %localmax = scf.for %c = %lane to %channels step %width iter_args(%m = %ninf) -> f32 {
        %i = arith.addi %base, %c : index
        %v = memref.load %x[%i] : memref<?xf32, strided<[1]>, 1>
        %next = arith.maxnumf %v, %m : f32
        scf.yield %next : f32
      }
      %max = gpu.subgroup_reduce maxnumf %localmax : (f32) -> f32
      %localsum = scf.for %c = %lane to %channels step %width iter_args(%s = %f0) -> f32 {
        %i = arith.addi %base, %c : index
        %v = memref.load %x[%i] : memref<?xf32, strided<[1]>, 1>
        %shift = arith.subf %v, %max : f32
        %e = math.exp %shift : f32
        %next = arith.addf %s, %e : f32
        scf.yield %next : f32
      }
      %sum = gpu.subgroup_reduce add %localsum : (f32) -> f32
      scf.for %c = %lane to %channels step %width {
        %i = arith.addi %base, %c : index
        %v = memref.load %x[%i] : memref<?xf32, strided<[1]>, 1>
        %shift = arith.subf %v, %max : f32
        %e = math.exp %shift : f32
        %p = arith.divf %e, %sum : f32
        memref.store %p, %y[%i] : memref<?xf32, strided<[1]>, 1>
      }
    }
    gpu.return
  }

  // Packed NCHW grouped cross-correlation with bias and ReLU fused into the output store.
  gpu.func @agx_infer_conv2d(%x0: memref<1xf32, 1>, %w0: memref<1xf32, 1>, %b0: memref<1xf32, 1>, %y0: memref<1xf32, 1>,
      %N_: i32, %C_: i32, %H_: i32, %W_: i32, %K_: i32, %R_: i32, %S_: i32, %P_: i32, %Q_: i32,
      %ph_: i32, %pw_: i32, %sh_: i32, %sw_: i32, %dh_: i32, %dw_: i32, %G_: i32, %bias_: i32, %relu_: i32) kernel {
    %big = arith.constant 2147483647 : index
    %x = memref.reinterpret_cast %x0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %w = memref.reinterpret_cast %w0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %b = memref.reinterpret_cast %b0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %y = memref.reinterpret_cast %y0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %N = arith.index_cast %N_ : i32 to index
    %C = arith.index_cast %C_ : i32 to index
    %H = arith.index_cast %H_ : i32 to index
    %W = arith.index_cast %W_ : i32 to index
    %K = arith.index_cast %K_ : i32 to index
    %R = arith.index_cast %R_ : i32 to index
    %S = arith.index_cast %S_ : i32 to index
    %P = arith.index_cast %P_ : i32 to index
    %Q = arith.index_cast %Q_ : i32 to index
    %ph = arith.index_cast %ph_ : i32 to index
    %pw = arith.index_cast %pw_ : i32 to index
    %sh = arith.index_cast %sh_ : i32 to index
    %sw = arith.index_cast %sw_ : i32 to index
    %dh = arith.index_cast %dh_ : i32 to index
    %dw = arith.index_cast %dw_ : i32 to index
    %G = arith.index_cast %G_ : i32 to index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %i0 = arith.constant 0 : i32
    %f0 = arith.constant 0.0 : f32
    %id = gpu.global_id x
    %q = arith.remui %id, %Q : index
    %t1 = arith.divui %id, %Q : index
    %p = arith.remui %t1, %P : index
    %t2 = arith.divui %t1, %P : index
    %k = arith.remui %t2, %K : index
    %n = arith.divui %t2, %K : index
    %live = arith.cmpi ult, %n, %N : index
    scf.if %live {
      %cpg = arith.divui %C, %G : index
      %kpg = arith.divui %K, %G : index
      %group = arith.divui %k, %kpg : index
      %cb = arith.muli %group, %cpg : index
      %hp = arith.muli %p, %sh : index
      %wp = arith.muli %q, %sw : index
      %h0 = arith.subi %hp, %ph : index
      %w_0 = arith.subi %wp, %pw : index
      %nh0 = arith.subi %zero, %h0 : index
      %hh0 = arith.subi %H, %h0 : index
      %rlo0 = arith.ceildivsi %nh0, %dh : index
      %rhi0 = arith.ceildivsi %hh0, %dh : index
      %rlo = arith.maxsi %rlo0, %zero : index
      %rhi = arith.minsi %rhi0, %R : index
      %nw0 = arith.subi %zero, %w_0 : index
      %ww0 = arith.subi %W, %w_0 : index
      %slo0 = arith.ceildivsi %nw0, %dw : index
      %shi0 = arith.ceildivsi %ww0, %dw : index
      %slo = arith.maxsi %slo0, %zero : index
      %shi = arith.minsi %shi0, %S : index
      %nc = arith.muli %n, %C : index
      %wkc = arith.muli %k, %cpg : index
      %acc = scf.for %cc = %zero to %cpg step %one iter_args(%acc0 = %f0) -> f32 {
        %c = arith.addi %cb, %cc : index
        %xc = arith.addi %nc, %c : index
        %xch = arith.muli %xc, %H : index
        %wc = arith.addi %wkc, %cc : index
        %wcr = arith.muli %wc, %R : index
        %acc1 = scf.for %r = %rlo to %rhi step %one iter_args(%acc2 = %acc0) -> f32 {
          %rdh = arith.muli %r, %dh : index
          %h = arith.addi %h0, %rdh : index
          %xh = arith.addi %xch, %h : index
          %xhw = arith.muli %xh, %W : index
          %wr = arith.addi %wcr, %r : index
          %wrs = arith.muli %wr, %S : index
          %acc3 = scf.for %ss = %slo to %shi step %one iter_args(%acc4 = %acc2) -> f32 {
            %sdw = arith.muli %ss, %dw : index
            %ww = arith.addi %w_0, %sdw : index
            %xi = arith.addi %xhw, %ww : index
            %wi = arith.addi %wrs, %ss : index
            %xv = memref.load %x[%xi] : memref<?xf32, strided<[1]>, 1>
            %wv = memref.load %w[%wi] : memref<?xf32, strided<[1]>, 1>
            %v = math.fma %xv, %wv, %acc4 : f32
            scf.yield %v : f32
          }
          scf.yield %acc3 : f32
        }
        scf.yield %acc1 : f32
      }
      %bias = arith.cmpi ne, %bias_, %i0 : i32
      %biased = scf.if %bias -> f32 {
        %bv = memref.load %b[%k] : memref<?xf32, strided<[1]>, 1>
        %v = arith.addf %acc, %bv : f32
        scf.yield %v : f32
      } else {
        scf.yield %acc : f32
      }
      %relu = arith.cmpi ne, %relu_, %i0 : i32
      %r = arith.maxnumf %biased, %f0 : f32
      %out = arith.select %relu, %r, %biased : f32
      memref.store %out, %y[%id] : memref<?xf32, strided<[1]>, 1>
    }
    gpu.return
  }

  // Standard MLIR reduction. Inner-strided rows avoid transpose and coefficient buffers.
  gpu.func @agx_infer_reduce(%x0: memref<1xf32, 1>, %y0: memref<1xf32, 1>, %rows_: i32, %channels_: i32, %inner_: i32, %mean_: i32) kernel {
    %big = arith.constant 2147483647 : index
    %x = memref.reinterpret_cast %x0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %y = memref.reinterpret_cast %y0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %zero = arith.constant 0 : index
    %fzero = arith.constant 0.0 : f32
    %izero = arith.constant 0 : i32
    %id = gpu.global_id x
    %width = arith.constant 1 : index
    %lane = arith.constant 0 : index
    %row = arith.addi %id, %zero : index
    %rows = arith.index_cast %rows_ : i32 to index
    %channels = arith.index_cast %channels_ : i32 to index
    %inner = arith.index_cast %inner_ : i32 to index
    %live = arith.cmpi ult, %row, %rows : index
    scf.if %live {
      %outer = arith.divui %row, %inner : index
      %tail = arith.remui %row, %inner : index
      %span = arith.muli %channels, %inner : index
      %head = arith.muli %outer, %span : index
      %base = arith.addi %head, %tail : index
      %local = scf.for %c = %lane to %channels step %width iter_args(%acc = %fzero) -> f32 {
        %offset = arith.muli %c, %inner : index
        %at = arith.addi %base, %offset : index
        %value = memref.load %x[%at] : memref<?xf32, strided<[1]>, 1>
        %next = arith.addf %acc, %value : f32
        scf.yield %next : f32
      }
      %sum = arith.addf %local, %fzero : f32
      %first = arith.cmpi eq, %lane, %zero : index
      scf.if %first {
        %average = arith.cmpi ne, %mean_, %izero : i32
        %result = scf.if %average -> f32 {
          %count = arith.sitofp %channels_ : i32 to f32
          %mean = arith.divf %sum, %count : f32
          scf.yield %mean : f32
        } else { scf.yield %sum : f32 }
        memref.store %result, %y[%row] : memref<?xf32, strided<[1]>, 1>
      }
    }
    gpu.return
  }

  // Standard MLIR reduction. Inner-strided rows avoid transpose and coefficient buffers.
  gpu.func @agx_infer_reduce_simd(%x0: memref<1xf32, 1>, %y0: memref<1xf32, 1>, %rows_: i32, %channels_: i32, %inner_: i32, %mean_: i32) kernel {
    %big = arith.constant 2147483647 : index
    %x = memref.reinterpret_cast %x0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %y = memref.reinterpret_cast %y0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %zero = arith.constant 0 : index
    %fzero = arith.constant 0.0 : f32
    %izero = arith.constant 0 : i32
    %id = gpu.global_id x
    %width = gpu.subgroup_size : index
    %lane = gpu.lane_id
    %row = arith.divui %id, %width : index
    %rows = arith.index_cast %rows_ : i32 to index
    %channels = arith.index_cast %channels_ : i32 to index
    %inner = arith.index_cast %inner_ : i32 to index
    %live = arith.cmpi ult, %row, %rows : index
    scf.if %live {
      %outer = arith.divui %row, %inner : index
      %tail = arith.remui %row, %inner : index
      %span = arith.muli %channels, %inner : index
      %head = arith.muli %outer, %span : index
      %base = arith.addi %head, %tail : index
      %local = scf.for %c = %lane to %channels step %width iter_args(%acc = %fzero) -> f32 {
        %offset = arith.muli %c, %inner : index
        %at = arith.addi %base, %offset : index
        %value = memref.load %x[%at] : memref<?xf32, strided<[1]>, 1>
        %next = arith.addf %acc, %value : f32
        scf.yield %next : f32
      }
      %sum = gpu.subgroup_reduce add %local : (f32) -> f32
      %first = arith.cmpi eq, %lane, %zero : index
      scf.if %first {
        %average = arith.cmpi ne, %mean_, %izero : i32
        %result = scf.if %average -> f32 {
          %count = arith.sitofp %channels_ : i32 to f32
          %mean = arith.divf %sum, %count : f32
          scf.yield %mean : f32
        } else { scf.yield %sum : f32 }
        memref.store %result, %y[%row] : memref<?xf32, strided<[1]>, 1>
      }
    }
    gpu.return
  }

  // Fused two-pass centered statistics and normalization, using only standard MLIR.
  gpu.func @agx_infer_layer_norm(%x0: memref<1xf32, 1>, %y0: memref<1xf32, 1>, %rows_: i32, %channels_: i32, %epsilon: f32) kernel {
    %big = arith.constant 2147483647 : index
    %x = memref.reinterpret_cast %x0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %y = memref.reinterpret_cast %y0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %zero = arith.constant 0 : index
    %fzero = arith.constant 0.0 : f32
    %fone = arith.constant 1.0 : f32
    %id = gpu.global_id x
    %width = arith.constant 1 : index
    %lane = arith.constant 0 : index
    %row = arith.addi %id, %zero : index
    %rows = arith.index_cast %rows_ : i32 to index
    %channels = arith.index_cast %channels_ : i32 to index
    %count = arith.sitofp %channels_ : i32 to f32
    %live = arith.cmpi ult, %row, %rows : index
    scf.if %live {
      %base = arith.muli %row, %channels : index
      %local = scf.for %c = %lane to %channels step %width iter_args(%acc = %fzero) -> f32 {
        %at = arith.addi %base, %c : index
        %v = memref.load %x[%at] : memref<?xf32, strided<[1]>, 1>
        %next = arith.addf %acc, %v : f32
        scf.yield %next : f32
      }
      %sum = arith.addf %local, %fzero : f32
      %mean = arith.divf %sum, %count : f32
      %localvar = scf.for %c = %lane to %channels step %width iter_args(%acc = %fzero) -> f32 {
        %at = arith.addi %base, %c : index
        %v = memref.load %x[%at] : memref<?xf32, strided<[1]>, 1>
        %d = arith.subf %v, %mean : f32
        %squared = arith.mulf %d, %d : f32
        %next = arith.addf %acc, %squared : f32
        scf.yield %next : f32
      }
      %totalvar = arith.addf %localvar, %fzero : f32
      %variance = arith.divf %totalvar, %count : f32
      %regularized = arith.addf %variance, %epsilon : f32
      %std = math.sqrt %regularized : f32
      %inverse = arith.divf %fone, %std : f32
      scf.for %c = %lane to %channels step %width {
        %at = arith.addi %base, %c : index
        %v = memref.load %x[%at] : memref<?xf32, strided<[1]>, 1>
        %d = arith.subf %v, %mean : f32
        %normalized = arith.mulf %d, %inverse : f32
        memref.store %normalized, %y[%at] : memref<?xf32, strided<[1]>, 1>
      }
    }
    gpu.return
  }

  // Fused two-pass centered statistics and normalization, using only standard MLIR.
  gpu.func @agx_infer_layer_norm_simd(%x0: memref<1xf32, 1>, %y0: memref<1xf32, 1>, %rows_: i32, %channels_: i32, %epsilon: f32) kernel {
    %big = arith.constant 2147483647 : index
    %x = memref.reinterpret_cast %x0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %y = memref.reinterpret_cast %y0 to offset: [0], sizes: [%big], strides: [1] : memref<1xf32, 1> to memref<?xf32, strided<[1]>, 1>
    %zero = arith.constant 0 : index
    %fzero = arith.constant 0.0 : f32
    %fone = arith.constant 1.0 : f32
    %id = gpu.global_id x
    %width = gpu.subgroup_size : index
    %lane = gpu.lane_id
    %row = arith.divui %id, %width : index
    %rows = arith.index_cast %rows_ : i32 to index
    %channels = arith.index_cast %channels_ : i32 to index
    %count = arith.sitofp %channels_ : i32 to f32
    %live = arith.cmpi ult, %row, %rows : index
    scf.if %live {
      %base = arith.muli %row, %channels : index
      %local = scf.for %c = %lane to %channels step %width iter_args(%acc = %fzero) -> f32 {
        %at = arith.addi %base, %c : index
        %v = memref.load %x[%at] : memref<?xf32, strided<[1]>, 1>
        %next = arith.addf %acc, %v : f32
        scf.yield %next : f32
      }
      %sum = gpu.subgroup_reduce add %local : (f32) -> f32
      %mean = arith.divf %sum, %count : f32
      %localvar = scf.for %c = %lane to %channels step %width iter_args(%acc = %fzero) -> f32 {
        %at = arith.addi %base, %c : index
        %v = memref.load %x[%at] : memref<?xf32, strided<[1]>, 1>
        %d = arith.subf %v, %mean : f32
        %squared = arith.mulf %d, %d : f32
        %next = arith.addf %acc, %squared : f32
        scf.yield %next : f32
      }
      %totalvar = gpu.subgroup_reduce add %localvar : (f32) -> f32
      %variance = arith.divf %totalvar, %count : f32
      %regularized = arith.addf %variance, %epsilon : f32
      %std = math.sqrt %regularized : f32
      %inverse = arith.divf %fone, %std : f32
      scf.for %c = %lane to %channels step %width {
        %at = arith.addi %base, %c : index
        %v = memref.load %x[%at] : memref<?xf32, strided<[1]>, 1>
        %d = arith.subf %v, %mean : f32
        %normalized = arith.mulf %d, %inverse : f32
        memref.store %normalized, %y[%at] : memref<?xf32, strided<[1]>, 1>
      }
    }
    gpu.return
  }
}
