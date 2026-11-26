#!/usr/bin/env python3
"""CUDA driver API conformance of libzluda_metal.so (src/cuda.rs, src/cuda/*.rs).

Every check asserts a behavior the CUDA Driver API reference documents (quoted or paraphrased next to it): error
codes for invalid values, handles and contexts, the context stack, primary context reference counting, memory
(pitched, 2D / 3D copies, sets, pinned / registered / managed memory, pointer attributes, the stream ordered
allocator), streams and events (ordering, host functions, NOT_READY, timing), modules and functions (attributes,
parameter info, enumeration), launches (validation, cooperative, cuLaunchKernelEx), occupancy, the CUDA 12
library API, the linker, graphs (explicit and captured) and cuGetProcAddress. Each group runs in its own process
(the driver's state is per process) on the AGX backend.

  test_driver.py [--lib LIB] [-k SUBSTR] [-v]
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
P, U64, SZ, I, U, F = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int, ctypes.c_uint, ctypes.c_float
byref = ctypes.byref

# CUresult values (the documented ABI)
SUCCESS, INVALID_VALUE, OUT_OF_MEMORY, NOT_INITIALIZED, DEINITIALIZED = 0, 1, 2, 3, 4
INVALID_DEVICE, INVALID_CONTEXT, NO_BINARY_FOR_GPU, UNSUPPORTED_LIMIT, INVALID_PTX = 101, 201, 209, 215, 218
UNSUPPORTED_EXEC_AFFINITY, FILE_NOT_FOUND, INVALID_HANDLE, ILLEGAL_STATE, NOT_FOUND, NOT_READY = 224, 301, 400, 401, 500, 600
PEER_ACCESS_NOT_ENABLED, CONTEXT_IS_DESTROYED = 705, 709
HOST_MEMORY_ALREADY_REGISTERED, HOST_MEMORY_NOT_REGISTERED, COOPERATIVE_LAUNCH_TOO_LARGE = 712, 713, 720
NOT_SUPPORTED = 801
CAPTURE_UNSUPPORTED, CAPTURE_INVALIDATED, CAPTURE_UNMATCHED, CAPTURE_UNJOINED = 900, 901, 903, 904
CAPTURE_ISOLATION, CAPTURE_IMPLICIT, CAPTURED_EVENT = 905, 906, 907

HDR = ".version 8.0\n.target sm_80\n.address_size 64\n"
# fill(p, v): p[i] = v + i for the block's threads (i = global thread index)
FILL = HDR + """.visible .entry fill(.param .u64 p, .param .u32 v) {
    .reg .b64 %rd<4>; .reg .b32 %r<6>;
    ld.param.u64 %rd1, [p]; ld.param.u32 %r1, [v];
    mov.u32 %r2, %tid.x; mov.u32 %r3, %ctaid.x; mov.u32 %r4, %ntid.x; mad.lo.u32 %r2, %r3, %r4, %r2;
    add.u32 %r5, %r1, %r2; mul.wide.u32 %rd2, %r2, 4; add.s64 %rd3, %rd1, %rd2; st.global.u32 [%rd3], %r5; ret; }
.visible .entry add1(.param .u64 p) {
    .reg .b64 %rd<4>; .reg .b32 %r<4>;
    ld.param.u64 %rd1, [p]; mov.u32 %r1, %tid.x; mul.wide.u32 %rd2, %r1, 4; add.s64 %rd3, %rd1, %rd2;
    ld.global.u32 %r2, [%rd3]; add.u32 %r3, %r2, 1; st.global.u32 [%rd3], %r3; ret; }
.visible .entry shared(.param .u64 p) {
    .shared .align 4 .b8 buf[1024];
    .reg .b64 %rd<4>; .reg .b32 %r<4>;
    ld.param.u64 %rd1, [p]; mov.u32 %r1, %tid.x; mov.u32 %r2, buf; mul.lo.u32 %r3, %r1, 4; add.u32 %r2, %r2, %r3;
    st.shared.u32 [%r2], %r1; bar.sync 0; ld.shared.u32 %r3, [%r2];
    mul.wide.u32 %rd2, %r1, 4; add.s64 %rd3, %rd1, %rd2; st.global.u32 [%rd3], %r3; ret; }
.visible .global .align 4 .u32 counter = 7;
"""
OOB = HDR + """.visible .entry k(.param .u64 p) {
    .reg .b64 %rd<2>; .reg .b32 %r<2>;
    mov.u64 %rd1, 16; mov.u32 %r1, 7; st.global.u32 [%rd1], %r1; ret; }"""

# argument types of the entry points the checks call (all return CUresult)
ARGS = {
    "cuInit": [U], "cuDriverGetVersion": [P], "cuDeviceGet": [P, I], "cuDeviceGetCount": [P],
    "cuDeviceGetName": [P, I, I], "cuDeviceTotalMem_v2": [P, I], "cuDeviceGetAttribute": [P, I, I],
    "cuDeviceComputeCapability": [P, P, I], "cuDeviceGetUuid": [P, I], "cuDeviceGetPCIBusId": [P, I, I],
    "cuDeviceGetByPCIBusId": [P, ctypes.c_char_p], "cuDeviceCanAccessPeer": [P, I, I],
    "cuDeviceGetP2PAttribute": [P, I, I, I],
    "cuCtxCreate_v2": [P, U, I], "cuCtxCreate_v3": [P, P, I, U, I], "cuCtxDestroy_v2": [P],
    "cuCtxPushCurrent_v2": [P], "cuCtxPopCurrent_v2": [P], "cuCtxSetCurrent": [P], "cuCtxGetCurrent": [P],
    "cuCtxGetDevice": [P], "cuCtxSynchronize": [], "cuCtxGetApiVersion": [P, P], "cuCtxGetId": [P, P],
    "cuCtxGetFlags": [P], "cuCtxSetFlags": [U], "cuCtxGetLimit": [P, I], "cuCtxSetLimit": [I, SZ],
    "cuCtxGetCacheConfig": [P], "cuCtxSetCacheConfig": [I], "cuCtxGetStreamPriorityRange": [P, P],
    "cuCtxEnablePeerAccess": [P, U], "cuCtxDisablePeerAccess": [P],
    "cuDevicePrimaryCtxRetain": [P, I], "cuDevicePrimaryCtxRelease_v2": [I], "cuDevicePrimaryCtxReset_v2": [I],
    "cuDevicePrimaryCtxGetState": [I, P, P], "cuDevicePrimaryCtxSetFlags_v2": [I, U],
    "cuMemAlloc_v2": [P, SZ], "cuMemFree_v2": [U64], "cuMemGetInfo_v2": [P, P], "cuMemGetAddressRange_v2": [P, P, U64],
    "cuMemAllocPitch_v2": [P, P, SZ, SZ, U], "cuMemAllocHost_v2": [P, SZ], "cuMemHostAlloc": [P, SZ, U],
    "cuMemFreeHost": [P], "cuMemHostGetDevicePointer_v2": [P, P, U], "cuMemHostGetFlags": [P, P],
    "cuMemHostRegister_v2": [P, SZ, U], "cuMemHostUnregister": [P], "cuMemAllocManaged": [P, SZ, U],
    "cuMemPrefetchAsync": [U64, SZ, I, P], "cuMemAdvise": [U64, SZ, I, I],
    "cuMemcpyHtoD_v2": [U64, P, SZ], "cuMemcpyDtoH_v2": [P, U64, SZ], "cuMemcpyDtoD_v2": [U64, U64, SZ],
    "cuMemcpy": [U64, U64, SZ], "cuMemcpyHtoDAsync_v2": [U64, P, SZ, P], "cuMemcpyDtoHAsync_v2": [P, U64, SZ, P],
    "cuMemcpyAsync": [U64, U64, SZ, P], "cuMemcpy2D_v2": [P], "cuMemcpy2DAsync_v2": [P, P], "cuMemcpy3D_v2": [P],
    "cuMemcpyPeer": [U64, P, U64, P, SZ],
    "cuMemsetD8_v2": [U64, ctypes.c_ubyte, SZ], "cuMemsetD16_v2": [U64, ctypes.c_ushort, SZ],
    "cuMemsetD32_v2": [U64, U, SZ], "cuMemsetD32Async": [U64, U, SZ, P],
    "cuMemsetD2D8_v2": [U64, SZ, ctypes.c_ubyte, SZ, SZ], "cuMemsetD2D32_v2": [U64, SZ, U, SZ, SZ],
    "cuPointerGetAttribute": [P, I, U64], "cuPointerGetAttributes": [U, P, P, U64], "cuPointerSetAttribute": [P, I, U64],
    "cuMemAllocAsync": [P, SZ, P], "cuMemFreeAsync": [U64, P], "cuDeviceGetDefaultMemPool": [P, I],
    "cuMemPoolGetAttribute": [P, I, P], "cuMemPoolSetAttribute": [P, I, P], "cuMemAllocFromPoolAsync": [P, SZ, P, P],
    "cuModuleLoadData": [P, ctypes.c_char_p], "cuModuleLoad": [P, ctypes.c_char_p], "cuModuleUnload": [P],
    "cuModuleGetFunction": [P, P, ctypes.c_char_p], "cuModuleGetGlobal_v2": [P, P, P, ctypes.c_char_p],
    "cuModuleGetFunctionCount": [P, P], "cuModuleEnumerateFunctions": [P, U, P], "cuModuleGetLoadingMode": [P],
    "cuFuncGetAttribute": [P, I, P], "cuFuncSetAttribute": [P, I, I], "cuFuncSetCacheConfig": [P, I],
    "cuFuncGetModule": [P, P], "cuFuncGetName": [P, P], "cuFuncGetParamInfo": [P, SZ, P, P], "cuFuncIsLoaded": [P, P],
    "cuLaunchKernel": [P, U, U, U, U, U, U, U, P, P, P], "cuLaunchCooperativeKernel": [P, U, U, U, U, U, U, U, P, P],
    "cuLaunchKernelEx": [P, P, P, P],
    "cuOccupancyMaxActiveBlocksPerMultiprocessor": [P, P, I, SZ],
    "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": [P, P, I, SZ, U],
    "cuOccupancyMaxPotentialBlockSize": [P, P, P, P, SZ, I], "cuOccupancyAvailableDynamicSMemPerBlock": [P, P, I, I],
    "cuStreamCreate": [P, U], "cuStreamCreateWithPriority": [P, U, I], "cuStreamDestroy_v2": [P],
    "cuStreamSynchronize": [P], "cuStreamQuery": [P], "cuStreamWaitEvent": [P, P, U], "cuStreamGetFlags": [P, P],
    "cuStreamGetPriority": [P, P], "cuStreamGetCtx": [P, P], "cuStreamGetId": [P, P],
    "cuStreamAddCallback": [P, P, P, U], "cuLaunchHostFunc": [P, P, P],
    "cuEventCreate": [P, U], "cuEventDestroy_v2": [P], "cuEventRecord": [P, P], "cuEventQuery": [P],
    "cuEventSynchronize": [P], "cuEventElapsedTime": [P, P, P],
    "cuGetErrorName": [U, P], "cuGetErrorString": [U, P], "cuGetProcAddress_v2": [ctypes.c_char_p, P, I, U64, P],
    "cuLibraryLoadData": [P, ctypes.c_char_p, P, P, U, P, P, U], "cuLibraryUnload": [P],
    "cuLibraryGetKernel": [P, P, ctypes.c_char_p], "cuLibraryGetModule": [P, P], "cuLibraryGetKernelCount": [P, P],
    "cuLibraryGetGlobal": [P, P, P, ctypes.c_char_p], "cuKernelGetFunction": [P, P],
    "cuKernelGetAttribute": [P, I, P, I], "cuKernelGetName": [P, P], "cuKernelGetLibrary": [P, P],
    "cuLinkCreate_v2": [U, P, P, P], "cuLinkAddData_v2": [P, I, P, SZ, ctypes.c_char_p, U, P, P],
    "cuLinkComplete": [P, P, P], "cuLinkDestroy": [P],
    "cuGraphCreate": [P, U], "cuGraphDestroy": [P], "cuGraphAddKernelNode_v2": [P, P, P, SZ, P],
    "cuGraphAddMemsetNode": [P, P, P, SZ, P, P], "cuGraphAddMemcpyNode": [P, P, P, SZ, P, P],
    "cuGraphAddHostNode": [P, P, P, SZ, P], "cuGraphAddEmptyNode": [P, P, P, SZ], "cuGraphGetNodes": [P, P, P],
    "cuGraphGetRootNodes": [P, P, P], "cuGraphGetEdges": [P, P, P, P], "cuGraphNodeGetType": [P, P],
    "cuGraphInstantiateWithFlags": [P, P, U64], "cuGraphLaunch": [P, P], "cuGraphExecDestroy": [P],
    "cuStreamBeginCapture_v2": [P, I], "cuStreamEndCapture": [P, P], "cuStreamIsCapturing": [P, P],
    "cuStreamGetCaptureInfo_v2": [P, P, P, P, P, P], "cuThreadExchangeStreamCaptureMode": [P],
}


class Driver:
    def __init__(self, lib):
        self.l = ctypes.CDLL(lib)

    def __getattr__(self, name):
        f = getattr(self.l, name)
        f.restype = U
        if name in ARGS:
            f.argtypes = ARGS[name]
        setattr(self, name, f)
        return f


class Checks:
    def __init__(self, group, verbose):
        self.group, self.verbose, self.results = group, verbose, []

    def eq(self, got, want, what):
        """`what` names the behavior and the documentation it follows"""
        ok = got == want
        self.results.append((ok, what, got, want))
        if self.verbose or not ok:
            print("  %s %s: %s: got %r, want %r" % ("ok  " if ok else "FAIL", self.group, what, got, want), flush=True)
        return ok

    def true(self, cond, what):
        return self.eq(bool(cond), True, what)


def u32s(b):
    return list(memoryview(bytes(b)).cast("I"))


class Env:
    """a context and helpers"""

    def __init__(self, c, t, make_ctx=True):
        self.c, self.t = c, t
        assert c.cuInit(0) == 0
        if make_ctx:
            self.ctx = P()
            assert c.cuCtxCreate_v2(byref(self.ctx), 0, 0) == 0

    def alloc(self, n, data=None):
        p = U64()
        assert self.c.cuMemAlloc_v2(byref(p), n) == 0
        if data is not None:
            assert self.c.cuMemcpyHtoD_v2(p.value, data, len(data)) == 0
        return p.value

    def read(self, p, n):
        b = ctypes.create_string_buffer(n)
        r = self.c.cuMemcpyDtoH_v2(b, p, n)
        assert r == 0, r
        return b.raw

    def module(self, src=FILL):
        m = P()
        r = self.c.cuModuleLoadData(byref(m), src.encode())
        assert r == 0, r
        return m

    def func(self, m, name):
        f = P()
        assert self.c.cuModuleGetFunction(byref(f), m, name.encode()) == 0
        return f

    @staticmethod
    def params(*vals):
        """kernel parameter array from ctypes values"""
        keep = list(vals)
        arr = (P * len(keep))(*[ctypes.cast(byref(v), P) for v in keep])
        arr._keep = keep
        return arr

    def launch(self, f, grid, block, args, stream=None, shared=0):
        return self.c.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, shared, stream, self.params(*args), None)


# ---------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------
def g_init(c, t):
    # "cuInit must be called before any other function from the driver API"; before it, calls return
    # CUDA_ERROR_NOT_INITIALIZED
    p, d, ctx = U64(), I(), P()
    t.eq(c.cuDeviceGetCount(byref(d)), NOT_INITIALIZED, "cuDeviceGetCount before cuInit")
    t.eq(c.cuDeviceGet(byref(d), 0), NOT_INITIALIZED, "cuDeviceGet before cuInit")
    t.eq(c.cuCtxCreate_v2(byref(ctx), 0, 0), NOT_INITIALIZED, "cuCtxCreate before cuInit")
    t.eq(c.cuMemAlloc_v2(byref(p), 16), NOT_INITIALIZED, "cuMemAlloc before cuInit")
    t.eq(c.cuCtxGetCurrent(byref(ctx)), NOT_INITIALIZED, "cuCtxGetCurrent before cuInit")
    t.eq(c.cuEventCreate(byref(ctx), 0), NOT_INITIALIZED, "cuEventCreate before cuInit")
    t.eq(c.cuMemAllocHost_v2(byref(ctx), 16), NOT_INITIALIZED, "cuMemAllocHost before cuInit")
    t.eq(c.cuModuleLoadData(byref(ctx), FILL.encode()), NOT_INITIALIZED, "cuModuleLoadData before cuInit")
    # cuDriverGetVersion and the error functions work before cuInit
    t.eq(c.cuDriverGetVersion(byref(d)), SUCCESS, "cuDriverGetVersion before cuInit")
    t.true(d.value >= 12000, "driver version is a CUDA 12 version")
    s = ctypes.c_char_p()
    t.eq(c.cuGetErrorName(NOT_INITIALIZED, byref(s)), SUCCESS, "cuGetErrorName before cuInit")
    # "Currently, the Flags parameter must be 0"
    t.eq(c.cuInit(1), INVALID_VALUE, "cuInit flags must be 0")
    t.eq(c.cuInit(0), SUCCESS, "cuInit")
    t.eq(c.cuInit(0), SUCCESS, "cuInit twice")
    t.eq(c.cuDeviceGetCount(byref(d)), SUCCESS, "cuDeviceGetCount after cuInit")
    t.eq(d.value, 1, "one device")
    # without a current context, context-dependent calls return CUDA_ERROR_INVALID_CONTEXT
    t.eq(c.cuMemAlloc_v2(byref(p), 16), INVALID_CONTEXT, "cuMemAlloc without a current context")
    t.eq(c.cuCtxSynchronize(), INVALID_CONTEXT, "cuCtxSynchronize without a current context")
    t.eq(c.cuCtxGetCurrent(byref(ctx)), SUCCESS, "cuCtxGetCurrent without a context")
    t.eq(ctx.value, None, "cuCtxGetCurrent: NULL when no context is current")


def g_errors(c, t):
    Env(c, t, make_ctx=False)
    s = ctypes.c_char_p()
    for code, name in [(0, "CUDA_SUCCESS"), (1, "CUDA_ERROR_INVALID_VALUE"), (201, "CUDA_ERROR_INVALID_CONTEXT"),
                       (400, "CUDA_ERROR_INVALID_HANDLE"), (600, "CUDA_ERROR_NOT_READY"),
                       (700, "CUDA_ERROR_ILLEGAL_ADDRESS"), (709, "CUDA_ERROR_CONTEXT_IS_DESTROYED"),
                       (712, "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED"), (901, "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED"),
                       (999, "CUDA_ERROR_UNKNOWN")]:
        t.eq(c.cuGetErrorName(code, byref(s)), SUCCESS, "cuGetErrorName(%d)" % code)
        t.eq(s.value.decode(), name, "cuGetErrorName(%d) is the enumerator's name" % code)
    # "returns CUDA_ERROR_INVALID_VALUE ... if error is not a valid CUresult; *pStr is set to NULL"
    t.eq(c.cuGetErrorName(12345, byref(s)), INVALID_VALUE, "cuGetErrorName of an unknown code")
    t.eq(s.value, None, "cuGetErrorName of an unknown code sets NULL")
    t.eq(c.cuGetErrorString(2, byref(s)), SUCCESS, "cuGetErrorString")
    t.true(s.value and b"memory" in s.value and s.value != b"CUDA_ERROR_OUT_OF_MEMORY",
           "cuGetErrorString is a description, not the name")
    t.eq(c.cuGetErrorString(12345, byref(s)), INVALID_VALUE, "cuGetErrorString of an unknown code")


def g_device(c, t):
    Env(c, t, make_ctx=False)
    d, v, mj, mn = I(), I(), I(), I()
    t.eq(c.cuDeviceGet(byref(d), 1), INVALID_DEVICE, "cuDeviceGet: an ordinal past the count is INVALID_DEVICE")
    t.eq(c.cuDeviceGet(None, 0), INVALID_VALUE, "cuDeviceGet: NULL output")
    name = ctypes.create_string_buffer(256)
    t.eq(c.cuDeviceGetName(name, 256, 0), SUCCESS, "cuDeviceGetName")
    t.true(len(name.value) > 0, "the device has a name")
    t.eq(c.cuDeviceGetName(name, 256, 3), INVALID_DEVICE, "cuDeviceGetName of an invalid device")
    t.eq(c.cuDeviceGetAttribute(byref(v), 10, 0), SUCCESS, "cuDeviceGetAttribute WARP_SIZE")
    t.eq(v.value, 32, "WARP_SIZE is 32")
    t.eq(c.cuDeviceGetAttribute(byref(v), 0, 0), INVALID_VALUE, "cuDeviceGetAttribute: 0 is no attribute")
    t.eq(c.cuDeviceGetAttribute(byref(v), 100000, 0), INVALID_VALUE, "cuDeviceGetAttribute: unknown attribute")
    t.eq(c.cuDeviceGetAttribute(byref(v), 1, 5), INVALID_DEVICE, "cuDeviceGetAttribute of an invalid device")
    t.eq(c.cuDeviceGetAttribute(None, 1, 0), INVALID_VALUE, "cuDeviceGetAttribute: NULL output")
    vals = {}
    for a in range(1, 135):
        r = c.cuDeviceGetAttribute(byref(v), a, 0)
        if r != 0:
            t.eq(r, SUCCESS, "attribute %d is answered" % a)
        vals[a] = v.value
    t.eq(c.cuDeviceComputeCapability(byref(mj), byref(mn), 0), SUCCESS, "cuDeviceComputeCapability")
    t.eq((vals[75], vals[76]), (mj.value, mn.value), "COMPUTE_CAPABILITY_MAJOR/MINOR match cuDeviceComputeCapability")
    t.true(vals[1] >= vals[2] and vals[2] >= 1 and vals[4] >= 1, "block limits are consistent")
    t.true(vals[97] >= vals[8], "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN >= MAX_SHARED_MEMORY_PER_BLOCK")
    t.eq(vals[41], 1, "UNIFIED_ADDRESSING")
    t.eq(vals[83], 1, "MANAGED_MEMORY (AGX backend)")
    t.eq(vals[115], 1, "MEMORY_POOLS_SUPPORTED")
    t.true(vals[16] > 0 and vals[39] >= vals[1], "MULTIPROCESSOR_COUNT and MAX_THREADS_PER_MULTIPROCESSOR")
    total = SZ()
    t.eq(c.cuDeviceTotalMem_v2(byref(total), 0), SUCCESS, "cuDeviceTotalMem")
    t.true(total.value > 0, "total memory")
    bus = ctypes.create_string_buffer(32)
    t.eq(c.cuDeviceGetPCIBusId(bus, 32, 0), SUCCESS, "cuDeviceGetPCIBusId")
    t.eq(c.cuDeviceGetByPCIBusId(byref(d), bus.value), SUCCESS, "cuDeviceGetByPCIBusId round trip")
    t.eq(d.value, 0, "cuDeviceGetByPCIBusId finds device 0")
    # peer access: one device
    t.eq(c.cuDeviceCanAccessPeer(byref(v), 0, 0), SUCCESS, "cuDeviceCanAccessPeer")
    t.eq(v.value, 0, "a device is not its own peer")
    t.eq(c.cuDeviceCanAccessPeer(byref(v), 0, 1), INVALID_DEVICE, "cuDeviceCanAccessPeer of an invalid peer")


def g_context_stack(c, t):
    Env(c, t, make_ctx=False)
    a, b, cur, popped = P(), P(), P(), P()
    v, dev = U(), I()
    # "Creates a new CUDA context and associates it with the calling thread": it is pushed
    t.eq(c.cuCtxCreate_v2(byref(a), 0, 0), SUCCESS, "cuCtxCreate")
    c.cuCtxGetCurrent(byref(cur))
    t.eq(cur.value, a.value, "a new context is current")
    t.eq(c.cuCtxCreate_v2(byref(b), 0, 0), SUCCESS, "cuCtxCreate (second)")
    c.cuCtxGetCurrent(byref(cur))
    t.eq(cur.value, b.value, "the newest context is current")
    # cuCtxPopCurrent: "Pops the current CUDA context from the CPU thread and passes back the old context handle"
    t.eq(c.cuCtxPopCurrent_v2(byref(popped)), SUCCESS, "cuCtxPopCurrent")
    t.eq(popped.value, b.value, "cuCtxPopCurrent returns the popped context")
    c.cuCtxGetCurrent(byref(cur))
    t.eq(cur.value, a.value, "the previous context is current again (a real stack)")
    t.eq(c.cuCtxPushCurrent_v2(b), SUCCESS, "cuCtxPushCurrent")
    c.cuCtxGetCurrent(byref(cur))
    t.eq(cur.value, b.value, "the pushed context is current")
    t.eq(c.cuCtxPushCurrent_v2(None), INVALID_CONTEXT, "cuCtxPushCurrent(NULL)")
    t.eq(c.cuCtxPushCurrent_v2(P(0x1234)), INVALID_CONTEXT, "cuCtxPushCurrent of a bogus handle")
    # cuCtxSetCurrent: "this will replace the top of that stack with ctx"; NULL pops
    t.eq(c.cuCtxSetCurrent(a), SUCCESS, "cuCtxSetCurrent")
    c.cuCtxPopCurrent_v2(byref(popped))
    t.eq(popped.value, a.value, "cuCtxSetCurrent replaced the top")
    c.cuCtxGetCurrent(byref(cur))
    t.eq(cur.value, a.value, "below it is the first push of a")
    t.eq(c.cuCtxSetCurrent(None), SUCCESS, "cuCtxSetCurrent(NULL) pops")
    c.cuCtxGetCurrent(byref(cur))
    t.eq(cur.value, None, "the stack is empty")
    t.eq(c.cuCtxSetCurrent(None), SUCCESS, "cuCtxSetCurrent(NULL) on an empty stack is a no-op")
    t.eq(c.cuCtxPopCurrent_v2(byref(popped)), INVALID_CONTEXT, "cuCtxPopCurrent on an empty stack")
    t.eq(c.cuCtxGetDevice(byref(dev)), INVALID_CONTEXT, "cuCtxGetDevice without a context")
    # cuCtxGetApiVersion / cuCtxGetId accept a context or NULL (the current one)
    t.eq(c.cuCtxGetApiVersion(a, byref(v)), SUCCESS, "cuCtxGetApiVersion")
    t.eq(v.value, 3020, "API version 3020")
    t.eq(c.cuCtxGetApiVersion(None, byref(v)), INVALID_CONTEXT, "cuCtxGetApiVersion(NULL) without a current context")
    ia, ib = U64(), U64()
    c.cuCtxGetId(a, byref(ia))
    c.cuCtxGetId(b, byref(ib))
    t.true(ia.value != ib.value, "cuCtxGetId: contexts have unique ids")
    # cuCtxDestroy: "If ctx is current to the calling thread then ctx will also be popped"
    c.cuCtxPushCurrent_v2(a)
    c.cuCtxPushCurrent_v2(b)
    t.eq(c.cuCtxDestroy_v2(b), SUCCESS, "cuCtxDestroy of the current context")
    c.cuCtxGetCurrent(byref(cur))
    t.eq(cur.value, a.value, "the destroyed current context was popped")
    t.eq(c.cuCtxDestroy_v2(b), INVALID_CONTEXT, "cuCtxDestroy of a destroyed context")
    t.eq(c.cuCtxPushCurrent_v2(b), INVALID_CONTEXT, "cuCtxPushCurrent of a destroyed context")
    t.eq(c.cuCtxDestroy_v2(None), INVALID_CONTEXT, "cuCtxDestroy(NULL)")
    # destroyed below the top: when it becomes current again, CUDA_ERROR_CONTEXT_IS_DESTROYED
    x = P()
    c.cuCtxCreate_v2(byref(x), 0, 0)  # stack: a, x
    c.cuCtxPushCurrent_v2(a)  # a, x, a
    c.cuCtxDestroy_v2(x)  # x is not current: stays in the stack
    c.cuCtxPopCurrent_v2(None)
    t.eq(c.cuCtxSynchronize(), CONTEXT_IS_DESTROYED, "a destroyed context that is current: CONTEXT_IS_DESTROYED")
    c.cuCtxPopCurrent_v2(None)
    t.eq(c.cuCtxSynchronize(), SUCCESS, "back to a live context")
    # flags
    t.eq(c.cuCtxCreate_v2(byref(x), 3, 0), INVALID_VALUE, "cuCtxCreate: two scheduling flags at once")
    t.eq(c.cuCtxCreate_v2(byref(x), 0x10000, 0), INVALID_VALUE, "cuCtxCreate: unknown flags")
    t.eq(c.cuCtxCreate_v2(byref(x), 0, 1), INVALID_DEVICE, "cuCtxCreate: invalid device")
    t.eq(c.cuCtxCreate_v2(byref(x), 4, 0), SUCCESS, "cuCtxCreate with CU_CTX_SCHED_BLOCKING_SYNC")
    t.eq(c.cuCtxGetFlags(byref(v)), SUCCESS, "cuCtxGetFlags")
    t.eq(v.value & 7, 4, "cuCtxGetFlags returns the scheduling flag")
    t.eq(c.cuCtxSetFlags(2), SUCCESS, "cuCtxSetFlags")
    c.cuCtxGetFlags(byref(v))
    t.eq(v.value & 7, 2, "cuCtxSetFlags took effect")
    t.eq(c.cuCtxCreate_v3(byref(x), byref(U(0)), 1, 0, 0), UNSUPPORTED_EXEC_AFFINITY,
         "cuCtxCreate_v3 with execution affinity parameters")
    # per-thread stacks: another thread starts with an empty stack
    seen = []

    def other():
        o = P()
        c.cuCtxGetCurrent(byref(o))
        seen.append(o.value)
        seen.append(c.cuCtxSetCurrent(a))
        c.cuCtxGetCurrent(byref(o))
        seen.append(o.value)
    th = threading.Thread(target=other)
    th.start()
    th.join()
    t.eq(seen, [None, SUCCESS, a.value], "the context stack is per thread")
    # peer access: one device
    t.eq(c.cuCtxEnablePeerAccess(a, 1), INVALID_VALUE, "cuCtxEnablePeerAccess: flags must be 0")
    t.eq(c.cuCtxEnablePeerAccess(a, 0), INVALID_DEVICE, "cuCtxEnablePeerAccess: the device cannot access a peer")
    t.eq(c.cuCtxDisablePeerAccess(a), PEER_ACCESS_NOT_ENABLED, "cuCtxDisablePeerAccess: never enabled")


def g_primary(c, t):
    e = Env(c, t, make_ctx=False)
    p1, p2 = P(), P()
    fl, act = U(), I()
    t.eq(c.cuDevicePrimaryCtxGetState(0, byref(fl), byref(act)), SUCCESS, "cuDevicePrimaryCtxGetState")
    t.eq(act.value, 0, "the primary context is inactive before it is retained")
    t.eq(c.cuDevicePrimaryCtxRelease_v2(0), INVALID_CONTEXT, "releasing a primary context that is not retained")
    t.eq(c.cuDevicePrimaryCtxSetFlags_v2(0, 4), SUCCESS, "cuDevicePrimaryCtxSetFlags")
    t.eq(c.cuDevicePrimaryCtxSetFlags_v2(0, 3), INVALID_VALUE, "cuDevicePrimaryCtxSetFlags: invalid flags")
    t.eq(c.cuDevicePrimaryCtxRetain(byref(p1), 0), SUCCESS, "cuDevicePrimaryCtxRetain")
    t.eq(c.cuDevicePrimaryCtxRetain(byref(p2), 0), SUCCESS, "cuDevicePrimaryCtxRetain again")
    t.eq(p1.value, p2.value, "the primary context is one context")
    c.cuDevicePrimaryCtxGetState(0, byref(fl), byref(act))
    t.eq((fl.value, act.value), (4, 1), "retained: active, with the flags set")
    t.eq(c.cuCtxDestroy_v2(p1), INVALID_CONTEXT, "a primary context is released, not destroyed")
    c.cuCtxSetCurrent(p1)
    d = e.alloc(64, b"\1" * 64)
    m = e.module()
    t.eq(c.cuDevicePrimaryCtxRelease_v2(0), SUCCESS, "release (count 2 -> 1)")
    t.eq(c.cuMemFree_v2(d), SUCCESS, "still retained: its memory is alive")
    d = e.alloc(64)
    # "Destroy all allocations and reset all state on the primary context" ... "does not release it"
    t.eq(c.cuDevicePrimaryCtxReset_v2(0), SUCCESS, "cuDevicePrimaryCtxReset")
    c.cuDevicePrimaryCtxGetState(0, byref(fl), byref(act))
    t.eq(act.value, 0, "a reset primary context is inactive")
    rng, base = SZ(), U64()
    t.eq(c.cuMemGetAddressRange_v2(byref(base), byref(rng), d), NOT_FOUND, "reset freed the context's allocations")
    f = P()
    t.eq(c.cuModuleGetFunction(byref(f), m, b"fill"), INVALID_HANDLE, "reset unloaded the context's modules")
    t.eq(c.cuCtxSynchronize(), SUCCESS, "still retained: the context is usable after a reset")
    c.cuDevicePrimaryCtxGetState(0, byref(fl), byref(act))
    t.eq(act.value, 1, "used again: active")
    d = e.alloc(64)
    t.eq(c.cuDevicePrimaryCtxRelease_v2(0), SUCCESS, "release to 0: the primary context is destroyed")
    c.cuDevicePrimaryCtxGetState(0, byref(fl), byref(act))
    t.eq(act.value, 0, "released: inactive")
    # CUDA_ERROR_CONTEXT_IS_DESTROYED: "...or is a primary context which has not yet been initialized"
    t.eq(c.cuCtxSynchronize(), CONTEXT_IS_DESTROYED, "using a released primary context that is still current")
    c.cuDevicePrimaryCtxRetain(byref(p1), 0)
    t.eq(c.cuMemGetAddressRange_v2(byref(base), byref(rng), d), NOT_FOUND, "release freed the allocations")
    t.eq(c.cuCtxSynchronize(), SUCCESS, "retained again: usable")
    # a sticky fault is cleared by the reset
    k = e.func(e.module(OOB), "k")
    t.eq(e.launch(k, 1, 1, [U64(0)]), SUCCESS, "a faulting launch succeeds (asynchronous)")
    t.eq(c.cuCtxSynchronize(), 700, "then the context has the sticky error")
    c.cuDevicePrimaryCtxReset_v2(0)
    t.eq(c.cuCtxSynchronize(), SUCCESS, "the reset cleared it")


def g_limits(c, t):
    Env(c, t)
    v = SZ()
    for lim, default in [(0, 1024), (1, 1 << 20), (2, 8 << 20)]:
        t.eq(c.cuCtxGetLimit(byref(v), lim), SUCCESS, "cuCtxGetLimit(%d)" % lim)
        t.eq(v.value, default, "limit %d default" % lim)
    t.eq(c.cuCtxSetLimit(2, 32 << 20), SUCCESS, "cuCtxSetLimit MALLOC_HEAP_SIZE")
    c.cuCtxGetLimit(byref(v), 2)
    t.eq(v.value, 32 << 20, "the limit was set")
    t.eq(c.cuCtxSetLimit(1, 4 << 20), SUCCESS, "cuCtxSetLimit PRINTF_FIFO_SIZE")
    t.eq(c.cuCtxGetLimit(byref(v), 99), UNSUPPORTED_LIMIT, "cuCtxGetLimit of an unknown limit")
    t.eq(c.cuCtxSetLimit(99, 1), UNSUPPORTED_LIMIT, "cuCtxSetLimit of an unknown limit")
    t.eq(c.cuCtxSetLimit(5, 256), INVALID_VALUE, "MAX_L2_FETCH_GRANULARITY is at most 128 bytes")
    cfg = I()
    t.eq(c.cuCtxSetCacheConfig(1), SUCCESS, "cuCtxSetCacheConfig")
    c.cuCtxGetCacheConfig(byref(cfg))
    t.eq(cfg.value, 1, "cuCtxGetCacheConfig")
    t.eq(c.cuCtxSetCacheConfig(9), INVALID_VALUE, "cuCtxSetCacheConfig: invalid value")
    lo, hi = I(), I()
    t.eq(c.cuCtxGetStreamPriorityRange(byref(lo), byref(hi)), SUCCESS, "cuCtxGetStreamPriorityRange")
    t.true(hi.value <= lo.value, "greatest priority <= least (lower numbers are higher priorities)")


def g_memory(c, t):
    e = Env(c, t)
    p, base = U64(), U64()
    n, free0, free1, total = SZ(), SZ(), SZ(), SZ()
    # "If bytesize is 0, cuMemAlloc() returns CUDA_ERROR_INVALID_VALUE"
    t.eq(c.cuMemAlloc_v2(byref(p), 0), INVALID_VALUE, "cuMemAlloc of 0 bytes")
    t.eq(c.cuMemAlloc_v2(None, 16), INVALID_VALUE, "cuMemAlloc: NULL output")
    c.cuMemGetInfo_v2(byref(free0), byref(total))
    d = e.alloc(1 << 20)
    c.cuMemGetInfo_v2(byref(free1), byref(total))
    t.eq(free0.value - free1.value, 1 << 20, "cuMemGetInfo: free memory drops by the allocation")
    t.eq(d % 256, 0, "allocations are aligned to at least 256 bytes")
    t.eq(c.cuMemAlloc_v2(byref(p), 1 << 60), OUT_OF_MEMORY, "an allocation beyond the device's memory")
    t.eq(c.cuMemGetAddressRange_v2(byref(base), byref(n), d + 1000), SUCCESS, "cuMemGetAddressRange inside")
    t.eq((base.value, n.value), (d, 1 << 20), "base and size of the allocation")
    t.eq(c.cuMemGetAddressRange_v2(byref(base), byref(n), 8), NOT_FOUND, "cuMemGetAddressRange of no allocation")
    t.eq(c.cuMemFree_v2(d + 4), INVALID_VALUE, "cuMemFree of an interior pointer")
    t.eq(c.cuMemFree_v2(d), SUCCESS, "cuMemFree")
    t.eq(c.cuMemFree_v2(d), INVALID_VALUE, "cuMemFree twice")
    t.eq(c.cuMemFree_v2(0), SUCCESS, "cuMemFree(0)")
    # copies
    d = e.alloc(64, bytes(range(64)))
    t.eq(e.read(d, 64), bytes(range(64)), "HtoD then DtoH")
    t.eq(c.cuMemcpyHtoD_v2(d + 60, b"x" * 8, 8), INVALID_VALUE, "cuMemcpyHtoD past the allocation")
    t.eq(c.cuMemcpyHtoD_v2(0x10, b"x" * 8, 8), INVALID_VALUE, "cuMemcpyHtoD to no device memory")
    t.eq(c.cuMemcpyHtoD_v2(d, None, 8), INVALID_VALUE, "cuMemcpyHtoD from NULL")
    d2 = e.alloc(64)
    t.eq(c.cuMemcpyDtoD_v2(d2, d, 64), SUCCESS, "cuMemcpyDtoD")
    t.eq(e.read(d2, 64), bytes(range(64)), "DtoD copied")
    hb = ctypes.create_string_buffer(64)
    t.eq(c.cuMemcpy(ctypes.addressof(hb), d, 64), SUCCESS, "cuMemcpy (unified addressing) device to host")
    t.eq(hb.raw, bytes(range(64)), "cuMemcpy read the device memory")
    t.eq(c.cuMemcpyPeer(d2, e.ctx, d, e.ctx, 64), SUCCESS, "cuMemcpyPeer between contexts of the device")
    # sets
    t.eq(c.cuMemsetD32_v2(d, 0xdeadbeef, 16), SUCCESS, "cuMemsetD32")
    t.eq(u32s(e.read(d, 64)), [0xdeadbeef] * 16, "cuMemsetD32 value")
    t.eq(c.cuMemsetD16_v2(d, 0x1234, 32), SUCCESS, "cuMemsetD16")
    t.eq(e.read(d, 4), b"\x34\x12\x34\x12", "cuMemsetD16 value")
    t.eq(c.cuMemsetD32_v2(d + 2, 0, 4), INVALID_VALUE, "cuMemsetD32 of a misaligned pointer")
    t.eq(c.cuMemsetD8_v2(d, 1, 65), INVALID_VALUE, "cuMemsetD8 past the allocation")


def g_pitch_2d_3d(c, t):
    e = Env(c, t)
    dp, pitch = U64(), SZ()
    # "ElementSizeBytes ... may be 4, 8 or 16"
    t.eq(c.cuMemAllocPitch_v2(byref(dp), byref(pitch), 100, 10, 3), INVALID_VALUE, "cuMemAllocPitch: element size 3")
    t.eq(c.cuMemAllocPitch_v2(byref(dp), byref(pitch), 0, 10, 4), INVALID_VALUE, "cuMemAllocPitch: width 0")
    t.eq(c.cuMemAllocPitch_v2(byref(dp), byref(pitch), 100, 10, 4), SUCCESS, "cuMemAllocPitch")
    t.true(pitch.value >= 100 and pitch.value % 256 == 0, "the pitch covers a row and is aligned")
    W, H = 100, 10

    class M2(ctypes.Structure):  # CUDA_MEMCPY2D
        _fields_ = [("srcX", SZ), ("srcY", SZ), ("srcType", U), ("srcHost", P), ("srcDevice", U64), ("srcArray", P),
                    ("srcPitch", SZ), ("dstX", SZ), ("dstY", SZ), ("dstType", U), ("dstHost", P),
                    ("dstDevice", U64), ("dstArray", P), ("dstPitch", SZ), ("Width", SZ), ("Height", SZ)]
    host = bytes((i * 7) & 255 for i in range(W * H))
    hbuf = ctypes.create_string_buffer(host, W * H)
    m = M2(srcType=1, srcHost=ctypes.addressof(hbuf), srcPitch=W, dstType=2, dstDevice=dp.value,
           dstPitch=pitch.value, Width=W, Height=H)
    t.eq(c.cuMemcpy2D_v2(byref(m)), SUCCESS, "cuMemcpy2D host to pitched device memory")
    raw = e.read(dp.value, pitch.value * H)
    t.eq(b"".join(raw[r * pitch.value:r * pitch.value + W] for r in range(H)), host, "rows landed at the pitch")
    out = ctypes.create_string_buffer(W * H)
    back = M2(srcType=2, srcDevice=dp.value, srcPitch=pitch.value, dstType=1, dstHost=ctypes.addressof(out),
              dstPitch=W, Width=W, Height=H)
    t.eq(c.cuMemcpy2D_v2(byref(back)), SUCCESS, "cuMemcpy2D back")
    t.eq(out.raw, host, "cuMemcpy2D round trip")
    sub = ctypes.create_string_buffer(4 * 3)
    s2 = M2(srcX=10, srcY=2, srcType=2, srcDevice=dp.value, srcPitch=pitch.value, dstType=1,
            dstHost=ctypes.addressof(sub), dstPitch=4, Width=4, Height=3)
    t.eq(c.cuMemcpy2D_v2(byref(s2)), SUCCESS, "cuMemcpy2D of a sub-rectangle (srcXInBytes, srcY)")
    t.eq(sub.raw, b"".join(host[(2 + r) * W + 10:(2 + r) * W + 14] for r in range(3)), "sub-rectangle contents")
    bad = M2(srcType=1, srcHost=ctypes.addressof(hbuf), srcPitch=W - 1, dstType=2, dstDevice=dp.value,
             dstPitch=pitch.value, Width=W, Height=H)
    t.eq(c.cuMemcpy2D_v2(byref(bad)), INVALID_VALUE, "cuMemcpy2D: a pitch below WidthInBytes")
    big = M2(srcType=1, srcHost=ctypes.addressof(hbuf), srcPitch=W, dstType=2, dstDevice=dp.value,
             dstPitch=pitch.value, Width=W, Height=H + 50)
    t.eq(c.cuMemcpy2D_v2(byref(big)), INVALID_VALUE, "cuMemcpy2D past the device allocation")
    arr = M2(srcType=3, dstType=2, dstDevice=dp.value, dstPitch=pitch.value, Width=W, Height=1)
    t.eq(c.cuMemcpy2D_v2(byref(arr)), NOT_SUPPORTED, "cuMemcpy2D from a CUDA array (arrays are not supported)")
    # 2D memset
    t.eq(c.cuMemsetD2D8_v2(dp.value, pitch.value, 0xAB, 4, H), SUCCESS, "cuMemsetD2D8")
    raw = e.read(dp.value, pitch.value * H)
    t.true(all(raw[r * pitch.value:r * pitch.value + 4] == b"\xab" * 4 and raw[r * pitch.value + 4] == host[r * W + 4]
               for r in range(H)), "cuMemsetD2D8 set only the rectangle")
    t.eq(c.cuMemsetD2D32_v2(dp.value, 8, 0, 4, 2), INVALID_VALUE, "cuMemsetD2D32: pitch below the row")

    class M3(ctypes.Structure):  # CUDA_MEMCPY3D
        _fields_ = [("srcX", SZ), ("srcY", SZ), ("srcZ", SZ), ("srcLOD", SZ), ("srcType", U), ("srcHost", P),
                    ("srcDevice", U64), ("srcArray", P), ("r0", P), ("srcPitch", SZ), ("srcHeight", SZ),
                    ("dstX", SZ), ("dstY", SZ), ("dstZ", SZ), ("dstLOD", SZ), ("dstType", U), ("dstHost", P),
                    ("dstDevice", U64), ("dstArray", P), ("r1", P), ("dstPitch", SZ), ("dstHeight", SZ),
                    ("Width", SZ), ("Height", SZ), ("Depth", SZ)]
    X, Y, Z = 8, 4, 3
    vol = bytes(range(X * Y * Z))
    vb = ctypes.create_string_buffer(vol, len(vol))
    d3 = e.alloc(16 * 5 * Z)
    c.cuMemsetD8_v2(d3, 0, 16 * 5 * Z)
    m3 = M3(srcType=1, srcHost=ctypes.addressof(vb), srcPitch=X, srcHeight=Y, dstType=2, dstDevice=d3, dstPitch=16,
            dstHeight=5, Width=X, Height=Y, Depth=Z)
    t.eq(c.cuMemcpy3D_v2(byref(m3)), SUCCESS, "cuMemcpy3D host to device with padded pitch and slice height")
    raw = e.read(d3, 16 * 5 * Z)
    ok = all(raw[(z * 5 + y) * 16:(z * 5 + y) * 16 + X] == vol[(z * Y + y) * X:(z * Y + y) * X + X]
             for z in range(Z) for y in range(Y))
    t.true(ok, "cuMemcpy3D placed every row of every slice")
    m3.srcHeight = Y - 1
    t.eq(c.cuMemcpy3D_v2(byref(m3)), INVALID_VALUE, "cuMemcpy3D: srcHeight below Height")


def g_host_memory(c, t):
    e = Env(c, t)
    hp, dptr = P(), U64()
    fl = U()
    # pinned memory is mapped at the same address (unified addressing)
    t.eq(c.cuMemHostAlloc(byref(hp), 4096, 0x100), INVALID_VALUE, "cuMemHostAlloc: unknown flags")
    t.eq(c.cuMemHostAlloc(byref(hp), 4096, 2), SUCCESS, "cuMemHostAlloc DEVICEMAP")
    t.eq(c.cuMemHostGetFlags(byref(fl), hp), SUCCESS, "cuMemHostGetFlags")
    t.eq(fl.value, 2, "cuMemHostGetFlags returns the allocation flags")
    t.eq(c.cuMemHostGetDevicePointer_v2(byref(dptr), hp, 1), INVALID_VALUE, "cuMemHostGetDevicePointer: flags must be 0")
    t.eq(c.cuMemHostGetDevicePointer_v2(byref(dptr), hp, 0), SUCCESS, "cuMemHostGetDevicePointer")
    t.eq(dptr.value, hp.value, "with unified addressing the device pointer is the host pointer")
    f = e.func(e.module(), "fill")
    t.eq(e.launch(f, 1, 64, [U64(dptr.value), U(100)]), SUCCESS, "a kernel writes pinned host memory")
    c.cuCtxSynchronize()
    t.eq(u32s(ctypes.string_at(hp.value, 256)), list(range(100, 164)), "the host sees the kernel's writes (zero copy)")
    t.eq(c.cuMemFreeHost(hp), SUCCESS, "cuMemFreeHost")
    t.eq(c.cuMemFreeHost(hp), INVALID_VALUE, "cuMemFreeHost twice")
    buf = ctypes.create_string_buffer(8192)
    t.eq(c.cuMemFreeHost(buf), INVALID_VALUE, "cuMemFreeHost of memory the driver did not allocate")
    t.eq(c.cuMemHostGetDevicePointer_v2(byref(dptr), buf, 0), INVALID_VALUE,
         "cuMemHostGetDevicePointer of unregistered memory")
    # registration
    t.eq(c.cuMemHostRegister_v2(buf, 0, 0), INVALID_VALUE, "cuMemHostRegister of 0 bytes")
    t.eq(c.cuMemHostRegister_v2(buf, 8192, 0x40), INVALID_VALUE, "cuMemHostRegister: unknown flags")
    t.eq(c.cuMemHostRegister_v2(buf, 8192, 2), SUCCESS, "cuMemHostRegister")
    t.eq(c.cuMemHostRegister_v2(ctypes.addressof(buf) + 4096, 16, 0), HOST_MEMORY_ALREADY_REGISTERED,
         "registering part of a registered range")
    t.eq(c.cuMemHostGetDevicePointer_v2(byref(dptr), ctypes.addressof(buf) + 16, 0), SUCCESS,
         "cuMemHostGetDevicePointer inside a registered range")
    t.eq(dptr.value, ctypes.addressof(buf) + 16, "the same offset")
    t.eq(e.launch(f, 1, 32, [U64(ctypes.addressof(buf)), U(7)]), SUCCESS, "a kernel writes registered memory")
    c.cuCtxSynchronize()
    t.eq(u32s(buf.raw[:128]), list(range(7, 39)), "the application's buffer has the kernel's writes")
    t.eq(c.cuMemHostUnregister(ctypes.addressof(buf) + 16), HOST_MEMORY_NOT_REGISTERED,
         "cuMemHostUnregister of an interior pointer")
    t.eq(c.cuMemHostUnregister(buf), SUCCESS, "cuMemHostUnregister")
    t.eq(c.cuMemHostUnregister(buf), HOST_MEMORY_NOT_REGISTERED, "cuMemHostUnregister twice")
    # managed memory
    t.eq(c.cuMemAllocManaged(byref(dptr), 256, 0), INVALID_VALUE, "cuMemAllocManaged: flags must be GLOBAL or HOST")
    t.eq(c.cuMemAllocManaged(byref(dptr), 0, 1), INVALID_VALUE, "cuMemAllocManaged of 0 bytes")
    t.eq(c.cuMemAllocManaged(byref(dptr), 256, 1), SUCCESS, "cuMemAllocManaged")
    ctypes.memmove(dptr.value, (U * 64)(*range(64)), 256)  # the host writes it directly
    t.eq(c.cuMemPrefetchAsync(dptr.value, 256, 0, None), SUCCESS, "cuMemPrefetchAsync to the device")
    t.eq(c.cuMemPrefetchAsync(dptr.value + 128, 256, 0, None), INVALID_VALUE, "cuMemPrefetchAsync past the range")
    t.eq(c.cuMemAdvise(dptr.value, 256, 1, 0), SUCCESS, "cuMemAdvise SET_READ_MOSTLY")
    add1 = e.func(e.module(), "add1")
    e.launch(add1, 1, 64, [U64(dptr.value)])
    c.cuCtxSynchronize()
    t.eq(list((U * 64).from_address(dptr.value)), list(range(1, 65)), "host and kernel share managed memory")
    t.eq(c.cuMemFree_v2(dptr.value), SUCCESS, "cuMemFree of managed memory")


def g_pointer_attributes(c, t):
    e = Env(c, t)
    d = e.alloc(4096)
    v32, v64, vp = U(), U64(), P()
    t.eq(c.cuPointerGetAttribute(byref(vp), 1, d + 8), SUCCESS, "CONTEXT")
    t.eq(vp.value, e.ctx.value, "CONTEXT is the allocating context")
    c.cuPointerGetAttribute(byref(v32), 2, d)
    t.eq(v32.value, 2, "MEMORY_TYPE of device memory is CU_MEMORYTYPE_DEVICE")
    c.cuPointerGetAttribute(byref(v64), 3, d + 8)
    t.eq(v64.value, d + 8, "DEVICE_POINTER")
    # "If the memory referenced by ptr is not accessible from the host then CUDA_ERROR_INVALID_VALUE"
    t.eq(c.cuPointerGetAttribute(byref(vp), 4, d), INVALID_VALUE, "HOST_POINTER of device memory")
    c.cuPointerGetAttribute(byref(v64), 11, d + 100)
    t.eq(v64.value, d, "RANGE_START_ADDR")
    sz = SZ()
    c.cuPointerGetAttribute(byref(sz), 12, d + 100)
    t.eq(sz.value, 4096, "RANGE_SIZE")
    c.cuPointerGetAttribute(byref(v32), 8, d)
    t.eq(v32.value, 0, "IS_MANAGED of device memory")
    id1, id2 = U64(), U64()
    d2 = e.alloc(16)
    c.cuPointerGetAttribute(byref(id1), 7, d)
    c.cuPointerGetAttribute(byref(id2), 7, d2)
    t.true(id1.value != id2.value, "BUFFER_ID is unique per allocation")
    t.eq(c.cuPointerGetAttribute(byref(v32), 2, 0x1234), INVALID_VALUE, "an unknown pointer")
    t.eq(c.cuPointerGetAttribute(byref(v32), 999, d), INVALID_VALUE, "an unknown attribute")
    hp = P()
    c.cuMemAllocHost_v2(byref(hp), 64)
    c.cuPointerGetAttribute(byref(v32), 2, hp.value)
    t.eq(v32.value, 1, "MEMORY_TYPE of pinned memory is CU_MEMORYTYPE_HOST")
    c.cuPointerGetAttribute(byref(vp), 4, hp.value)
    t.eq(vp.value, hp.value, "HOST_POINTER of pinned memory")
    m = U64()
    c.cuMemAllocManaged(byref(m), 64, 1)
    c.cuPointerGetAttribute(byref(v32), 8, m.value)
    t.eq(v32.value, 1, "IS_MANAGED of managed memory")
    # cuPointerGetAttributes: "will not return an error when the ptr encountered is not a valid CUDA pointer.
    # Instead, the attributes are assigned default NULL values"
    attrs = (I * 3)(2, 1, 8)
    outs = (U(7), P(7), U(7))
    data = (P * 3)(*[ctypes.cast(byref(o), P) for o in outs])
    t.eq(c.cuPointerGetAttributes(3, attrs, data, 0x1234), SUCCESS, "cuPointerGetAttributes of an unknown pointer")
    t.eq([outs[0].value, outs[1].value, outs[2].value], [0, None, 0], "defaults for an unknown pointer")
    t.eq(c.cuPointerGetAttributes(3, attrs, data, d), SUCCESS, "cuPointerGetAttributes of device memory")
    t.eq([outs[0].value, outs[1].value, outs[2].value], [2, e.ctx.value, 0], "memory type, context, managed")
    one = U(1)
    t.eq(c.cuPointerSetAttribute(byref(one), 6, d), SUCCESS, "cuPointerSetAttribute SYNC_MEMOPS")
    c.cuPointerGetAttribute(byref(v32), 6, d)
    t.eq(v32.value, 1, "SYNC_MEMOPS was set")
    t.eq(c.cuPointerSetAttribute(byref(one), 2, d), INVALID_VALUE, "only SYNC_MEMOPS is settable")


def g_mempool(c, t):
    e = Env(c, t)
    pool, d = P(), U64()
    s = P()
    c.cuStreamCreate(byref(s), 0)
    t.eq(c.cuDeviceGetDefaultMemPool(byref(pool), 0), SUCCESS, "cuDeviceGetDefaultMemPool")
    t.eq(c.cuMemAllocAsync(byref(d), 1024, s), SUCCESS, "cuMemAllocAsync")
    vp = P()
    c.cuPointerGetAttribute(byref(vp), 17, d.value)
    t.eq(vp.value, pool.value, "MEMPOOL_HANDLE of a stream-ordered allocation")
    used = U64()
    t.eq(c.cuMemPoolGetAttribute(pool, 7, byref(used)), SUCCESS, "cuMemPoolGetAttribute USED_MEM_CURRENT")
    t.eq(used.value, 1024, "the pool's used memory")
    thr = U64(1 << 20)
    t.eq(c.cuMemPoolSetAttribute(pool, 4, byref(thr)), SUCCESS, "cuMemPoolSetAttribute RELEASE_THRESHOLD")
    c.cuMemPoolGetAttribute(pool, 4, byref(used))
    t.eq(used.value, 1 << 20, "the release threshold")
    t.eq(c.cuMemFreeAsync(d.value, s), SUCCESS, "cuMemFreeAsync")
    c.cuStreamSynchronize(s)
    base, n = U64(), SZ()
    t.eq(c.cuMemGetAddressRange_v2(byref(base), byref(n), d.value), NOT_FOUND, "freed when the stream reached it")
    t.eq(c.cuMemAllocFromPoolAsync(byref(d), 64, P(0x1234), s), INVALID_VALUE, "cuMemAllocFromPoolAsync of no pool")
    t.eq(c.cuMemAllocAsync(byref(d), 64, P(0x1234)), INVALID_HANDLE, "cuMemAllocAsync on an invalid stream")


def g_streams(c, t):
    e = Env(c, t)
    s, s2, cx = P(), P(), P()
    fl, pr = U(), I()
    t.eq(c.cuStreamCreate(byref(s), 2), INVALID_VALUE, "cuStreamCreate: unknown flags")
    t.eq(c.cuStreamCreate(byref(s), 1), SUCCESS, "cuStreamCreate NON_BLOCKING")
    c.cuStreamGetFlags(s, byref(fl))
    t.eq(fl.value, 1, "cuStreamGetFlags")
    # "priority ... outside the range ... will automatically be clamped"
    t.eq(c.cuStreamCreateWithPriority(byref(s2), 0, -100), SUCCESS, "cuStreamCreateWithPriority")
    c.cuStreamGetPriority(s2, byref(pr))
    lo, hi = I(), I()
    c.cuCtxGetStreamPriorityRange(byref(lo), byref(hi))
    t.eq(pr.value, hi.value, "an out-of-range priority is clamped to the greatest")
    t.eq(c.cuStreamGetCtx(s, byref(cx)), SUCCESS, "cuStreamGetCtx")
    t.eq(cx.value, e.ctx.value, "the stream's context")
    i1, i2 = U64(), U64()
    c.cuStreamGetId(s, byref(i1))
    c.cuStreamGetId(s2, byref(i2))
    t.true(i1.value != i2.value, "cuStreamGetId: unique ids")
    t.eq(c.cuStreamQuery(s), SUCCESS, "cuStreamQuery of an idle stream")
    t.eq(c.cuStreamQuery(P(0x1234)), INVALID_HANDLE, "cuStreamQuery of an invalid stream")
    t.eq(c.cuStreamQuery(None), SUCCESS, "cuStreamQuery of the legacy stream")
    t.eq(c.cuStreamQuery(P(1)), SUCCESS, "cuStreamQuery(CU_STREAM_LEGACY)")
    t.eq(c.cuStreamQuery(P(2)), SUCCESS, "cuStreamQuery(CU_STREAM_PER_THREAD)")
    # host functions and NOT_READY
    gate = threading.Event()
    order = []
    HOSTFN = ctypes.CFUNCTYPE(None, P)
    CB = ctypes.CFUNCTYPE(None, P, U, P)
    hf = HOSTFN(lambda _: (gate.wait(30), order.append("host")))
    cb = CB(lambda st, status, data: order.append(("cb", status, data)))
    d = e.alloc(256)
    f = e.func(e.module(), "fill")
    t.eq(c.cuLaunchHostFunc(s, hf, None), SUCCESS, "cuLaunchHostFunc")
    t.eq(e.launch(f, 1, 64, [U64(d), U(5)], stream=s), SUCCESS, "a launch queued behind the host function")
    ev = P()
    c.cuEventCreate(byref(ev), 0)
    t.eq(c.cuEventRecord(ev, s), SUCCESS, "cuEventRecord behind the host function")
    t.eq(c.cuStreamAddCallback(s, cb, P(42), 1), INVALID_VALUE, "cuStreamAddCallback: flags must be 0")
    t.eq(c.cuStreamAddCallback(s, cb, P(42), 0), SUCCESS, "cuStreamAddCallback")
    time.sleep(0.2)
    # cuStreamQuery: "CUDA_ERROR_NOT_READY if not" all operations have completed
    t.eq(c.cuStreamQuery(s), NOT_READY, "cuStreamQuery while a host function blocks the stream")
    t.eq(c.cuEventQuery(ev), NOT_READY, "cuEventQuery of a record not yet reached")
    t.eq(order, [], "nothing after the host function ran")
    # a non-blocking stream's work does not wait for the other stream
    s3 = P()
    c.cuStreamCreate(byref(s3), 1)
    d3 = e.alloc(256)
    t.eq(e.launch(f, 1, 64, [U64(d3), U(9)], stream=s3), SUCCESS, "a launch on an independent stream")
    t.eq(c.cuStreamSynchronize(s3), SUCCESS, "the independent stream completes")
    hb = ctypes.create_string_buffer(256)
    c.cuMemcpyDtoHAsync_v2(hb, d3, 256, s3)
    c.cuStreamSynchronize(s3)
    t.eq(u32s(hb.raw)[:4], [9, 10, 11, 12], "the independent stream ran while the other was blocked")
    gate.set()
    t.eq(c.cuStreamSynchronize(s), SUCCESS, "cuStreamSynchronize waits for the host function and the work after it")
    t.eq(order, ["host", ("cb", 0, 42)], "host function, then the callback with CUDA_SUCCESS and its user data")
    t.eq(c.cuEventQuery(ev), SUCCESS, "the event completed")
    t.eq(c.cuStreamQuery(s), SUCCESS, "the stream is idle")
    c.cuMemcpyDtoHAsync_v2(hb, d, 256, s)
    c.cuStreamSynchronize(s)
    t.eq(u32s(hb.raw)[:4], [5, 6, 7, 8], "the queued kernel ran after the host function")
    # cross-stream ordering through events: s2 waits for an event of s
    gate.clear()
    hf2 = HOSTFN(lambda _: (gate.wait(30), order.append("host2")))
    c.cuLaunchHostFunc(s, hf2, None)
    c.cuEventRecord(ev, s)
    t.eq(c.cuStreamWaitEvent(s2, ev, 0), SUCCESS, "cuStreamWaitEvent")
    t.eq(c.cuStreamWaitEvent(s2, ev, 4), INVALID_VALUE, "cuStreamWaitEvent: unknown flags")
    after = HOSTFN(lambda _: order.append("after"))
    c.cuLaunchHostFunc(s2, after, None)
    time.sleep(0.2)
    t.eq(c.cuStreamQuery(s2), NOT_READY, "the waiting stream is not ready")
    gate.set()
    c.cuStreamSynchronize(s2)
    t.eq(order[-2:], ["host2", "after"], "the waiting stream's work ran after the event")
    # legacy stream implicit synchronization with blocking streams
    gate.clear()
    b1 = P()
    c.cuStreamCreate(byref(b1), 0)
    hb1 = HOSTFN(lambda _: (gate.wait(30), order.append("blocking")))
    hl = HOSTFN(lambda _: order.append("legacy"))
    c.cuLaunchHostFunc(b1, hb1, None)
    c.cuLaunchHostFunc(None, hl, None)
    time.sleep(0.2)
    t.eq(c.cuStreamQuery(None), NOT_READY, "the legacy stream waits for a blocking stream")
    gate.set()
    c.cuCtxSynchronize()
    t.eq(order[-2:], ["blocking", "legacy"], "legacy stream work ran after the blocking stream's")
    # cuStreamDestroy
    t.eq(c.cuStreamDestroy_v2(s3), SUCCESS, "cuStreamDestroy")
    t.eq(c.cuStreamDestroy_v2(s3), INVALID_HANDLE, "cuStreamDestroy twice")
    t.eq(c.cuStreamDestroy_v2(None), INVALID_HANDLE, "the legacy stream cannot be destroyed")
    t.eq(c.cuLaunchHostFunc(s3, hf, None), INVALID_HANDLE, "a destroyed stream")
    keep = [hf, cb, hf2, after, hb1, hl]  # noqa: F841


def g_events(c, t):
    e = Env(c, t)
    a, b, nt = P(), P(), P()
    ms = F()
    t.eq(c.cuEventCreate(byref(a), 8), INVALID_VALUE, "cuEventCreate: unknown flags")
    t.eq(c.cuEventCreate(byref(a), 4), INVALID_VALUE, "INTERPROCESS requires DISABLE_TIMING")
    c.cuEventCreate(byref(a), 0)
    c.cuEventCreate(byref(b), 0)
    c.cuEventCreate(byref(nt), 2)
    # "If cuEventRecord() has not been called on this event, the function returns CUDA_SUCCESS"
    t.eq(c.cuEventQuery(a), SUCCESS, "cuEventQuery of a never recorded event")
    t.eq(c.cuEventSynchronize(a), SUCCESS, "cuEventSynchronize of a never recorded event")
    # "If cuEventRecord() has not been called on either event then CUDA_ERROR_INVALID_HANDLE is returned"
    t.eq(c.cuEventElapsedTime(byref(ms), a, b), INVALID_HANDLE, "cuEventElapsedTime of unrecorded events")
    d = e.alloc(4096 * 4)
    f = e.func(e.module(), "fill")
    c.cuEventRecord(a, None)
    for i in range(3):
        e.launch(f, 16, 256, [U64(d), U(i)])
    c.cuEventRecord(b, None)
    c.cuEventRecord(nt, None)
    t.eq(c.cuEventSynchronize(b), SUCCESS, "cuEventSynchronize")
    t.eq(c.cuEventElapsedTime(byref(ms), a, b), SUCCESS, "cuEventElapsedTime")
    t.true(0.0 < ms.value < 10.0, "simulated GPU time of three launches: positive and plausible (%g ms)" % ms.value)
    t.eq(c.cuEventElapsedTime(byref(ms), b, a) == SUCCESS and ms.value < 0, True, "reversed events: negative time")
    # "If either event was created with the CU_EVENT_DISABLE_TIMING flag, then this function will return
    # CUDA_ERROR_INVALID_HANDLE"
    t.eq(c.cuEventElapsedTime(byref(ms), a, nt), INVALID_HANDLE, "cuEventElapsedTime of a DISABLE_TIMING event")
    t.eq(c.cuEventElapsedTime(None, a, b), INVALID_VALUE, "cuEventElapsedTime: NULL output")
    # NOT_READY: an event behind a blocked host function
    s = P()
    c.cuStreamCreate(byref(s), 0)
    gate = threading.Event()
    hf = ctypes.CFUNCTYPE(None, P)(lambda _: gate.wait(30))
    c.cuLaunchHostFunc(s, hf, None)
    c.cuEventRecord(b, s)
    t.eq(c.cuEventElapsedTime(byref(ms), a, b), NOT_READY, "cuEventElapsedTime of an incomplete event")
    gate.set()
    c.cuEventSynchronize(b)
    t.eq(c.cuEventQuery(b), SUCCESS, "complete after cuEventSynchronize")
    t.eq(c.cuEventDestroy_v2(a), SUCCESS, "cuEventDestroy")
    t.eq(c.cuEventDestroy_v2(a), INVALID_HANDLE, "cuEventDestroy twice")
    t.eq(c.cuEventRecord(a, None), INVALID_HANDLE, "cuEventRecord of a destroyed event")
    t.eq(c.cuEventQuery(P(0x1234)), INVALID_HANDLE, "cuEventQuery of an invalid event")
    keep = [hf]  # noqa: F841


def g_modules(c, t):
    e = Env(c, t)
    m, f, g, m2 = P(), P(), P(), P()
    v, n = I(), U()
    t.eq(c.cuModuleLoadData(byref(m), b"this is not PTX"), INVALID_PTX, "cuModuleLoadData of invalid PTX")
    t.eq(c.cuModuleLoadData(byref(m), b"\x7fELF\x02\x01\x01"), NO_BINARY_FOR_GPU, "a cubin (no binary for the GPU)")
    t.eq(c.cuModuleLoad(byref(m), b"/nonexistent/x.ptx"), FILE_NOT_FOUND, "cuModuleLoad of a missing file")
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as tf:
        tf.write(FILL)
    t.eq(c.cuModuleLoad(byref(m2), tf.name.encode()), SUCCESS, "cuModuleLoad of a PTX file")
    os.unlink(tf.name)
    m = e.module()
    t.eq(c.cuModuleGetFunction(byref(f), m, b"nope"), NOT_FOUND, "cuModuleGetFunction of an unknown name")
    t.eq(c.cuModuleGetFunction(byref(f), P(0x1234), b"fill"), INVALID_HANDLE, "cuModuleGetFunction: invalid module")
    t.eq(c.cuModuleGetFunction(None, m, b"fill"), INVALID_VALUE, "cuModuleGetFunction: NULL output")
    t.eq(c.cuModuleGetFunction(byref(f), m, b"fill"), SUCCESS, "cuModuleGetFunction")
    t.eq(c.cuModuleGetFunctionCount(byref(n), m), SUCCESS, "cuModuleGetFunctionCount")
    t.eq(n.value, 3, "three kernels")
    fs = (P * 3)()
    t.eq(c.cuModuleEnumerateFunctions(fs, 3, m), SUCCESS, "cuModuleEnumerateFunctions")
    names = []
    for h in fs:
        s = ctypes.c_char_p()
        c.cuFuncGetName(byref(s), h)
        names.append(s.value.decode())
    t.eq(sorted(names), ["add1", "fill", "shared"], "the enumerated functions' names (cuFuncGetName)")
    t.eq(c.cuModuleEnumerateFunctions(fs, 4, m), INVALID_VALUE, "cuModuleEnumerateFunctions of more than exist")
    mo = P()
    c.cuFuncGetModule(byref(mo), f)
    t.eq(mo.value, m.value, "cuFuncGetModule")
    off, sz = SZ(), SZ()
    t.eq(c.cuFuncGetParamInfo(f, 1, byref(off), byref(sz)), SUCCESS, "cuFuncGetParamInfo")
    t.eq((off.value, sz.value), (8, 4), "the second parameter (u32) at offset 8")
    t.eq(c.cuFuncGetParamInfo(f, 2, byref(off), byref(sz)), INVALID_VALUE, "cuFuncGetParamInfo past the last")
    t.eq(c.cuFuncIsLoaded(byref(v), f), SUCCESS, "cuFuncIsLoaded")
    t.eq(v.value, 1, "loaded")
    dp, nb = U64(), SZ()
    t.eq(c.cuModuleGetGlobal_v2(byref(dp), byref(nb), m, b"counter"), SUCCESS, "cuModuleGetGlobal")
    t.eq(nb.value, 4, "the variable's size")
    t.eq(e.read(dp.value, 4), b"\x07\0\0\0", "the variable's initializer")
    t.eq(c.cuModuleGetGlobal_v2(byref(dp), byref(nb), m, b"nope"), NOT_FOUND, "cuModuleGetGlobal: unknown name")
    vt = U()
    c.cuMemcpyHtoD_v2(dp.value, b"\x09\0\0\0", 4)
    # function attributes
    for a in range(16):
        r = c.cuFuncGetAttribute(byref(v), a, f)
        if r != 0:
            t.eq(r, SUCCESS, "cuFuncGetAttribute(%d)" % a)
    t.eq(c.cuFuncGetAttribute(byref(v), 16, f), INVALID_VALUE, "cuFuncGetAttribute: unknown attribute")
    sh = e.func(m, "shared")
    c.cuFuncGetAttribute(byref(v), 1, sh)
    t.eq(v.value, 1024, "SHARED_SIZE_BYTES of a kernel with 1 KiB of .shared")
    c.cuFuncGetAttribute(byref(v), 4, f)
    t.true(0 < v.value <= 255, "NUM_REGS")
    c.cuFuncGetAttribute(byref(v), 5, f)
    t.eq(v.value, 80, "PTX_VERSION of a .target sm_80 module")
    c.cuFuncGetAttribute(byref(v), 0, f)
    t.true(32 <= v.value <= 1024, "MAX_THREADS_PER_BLOCK")
    c.cuFuncGetAttribute(byref(v), 8, f)
    dflt = v.value
    t.eq(dflt, 48 << 10, "MAX_DYNAMIC_SHARED_SIZE_BYTES defaults to 48 KiB minus the static shared memory")
    t.eq(c.cuFuncSetAttribute(f, 8, 60 << 10), SUCCESS, "opting in to more dynamic shared memory")
    c.cuFuncGetAttribute(byref(v), 8, f)
    t.eq(v.value, 60 << 10, "the opt-in took effect")
    t.eq(c.cuFuncSetAttribute(f, 8, 1 << 20), INVALID_VALUE, "beyond MAX_SHARED_MEMORY_PER_BLOCK_OPTIN")
    t.eq(c.cuFuncSetAttribute(f, 9, 101), INVALID_VALUE, "PREFERRED_SHARED_MEMORY_CARVEOUT above 100")
    t.eq(c.cuFuncSetAttribute(f, 4, 10), INVALID_VALUE, "NUM_REGS is read-only")
    t.eq(c.cuFuncSetCacheConfig(f, 2), SUCCESS, "cuFuncSetCacheConfig")
    t.eq(c.cuFuncSetCacheConfig(f, 7), INVALID_VALUE, "cuFuncSetCacheConfig: invalid value")
    t.eq(c.cuFuncGetAttribute(byref(v), 0, P(0x1234)), INVALID_HANDLE, "cuFuncGetAttribute of an invalid function")
    # unload
    t.eq(c.cuModuleUnload(m2), SUCCESS, "cuModuleUnload")
    t.eq(c.cuModuleUnload(m2), INVALID_HANDLE, "cuModuleUnload twice")
    mode = I()
    t.eq(c.cuModuleGetLoadingMode(byref(mode)), SUCCESS, "cuModuleGetLoadingMode")
    t.eq(mode.value, 1, "eager loading")
    del vt


def g_launch(c, t):
    e = Env(c, t)
    m = e.module()
    f = e.func(m, "fill")
    d = e.alloc(4096 * 4)
    pa = Env.params(U64(d), U(3))
    L = c.cuLaunchKernel
    t.eq(L(f, 2, 1, 1, 128, 1, 1, 0, None, pa, None), SUCCESS, "cuLaunchKernel")
    c.cuCtxSynchronize()
    t.eq(u32s(e.read(d, 256 * 4)), [3 + i for i in range(256)], "the kernel's results")
    t.eq(L(f, 0, 1, 1, 32, 1, 1, 0, None, pa, None), INVALID_VALUE, "a zero grid dimension")
    t.eq(L(f, 1, 1, 1, 0, 1, 1, 0, None, pa, None), INVALID_VALUE, "a zero block dimension")
    t.eq(L(f, 1, 1, 1, 2048, 1, 1, 0, None, pa, None), INVALID_VALUE, "a block beyond MAX_THREADS_PER_BLOCK")
    t.eq(L(f, 1, 1, 1, 1, 1, 128, 0, None, pa, None), INVALID_VALUE, "blockDimZ beyond MAX_BLOCK_DIM_Z")
    t.eq(L(f, 1, 70000, 1, 32, 1, 1, 0, None, pa, None), INVALID_VALUE, "gridDimY beyond MAX_GRID_DIM_Y")
    t.eq(L(f, 1, 1, 1, 32, 1, 1, 50 << 10, None, pa, None), INVALID_VALUE,
         "dynamic shared memory beyond the function's MAX_DYNAMIC_SHARED_SIZE_BYTES")
    t.eq(L(f, 1, 1, 1, 32, 1, 1, 0, None, None, None), INVALID_VALUE, "no arguments for a kernel that has some")
    t.eq(L(P(0x1234), 1, 1, 1, 32, 1, 1, 0, None, pa, None), INVALID_HANDLE, "an invalid function")
    t.eq(L(f, 1, 1, 1, 32, 1, 1, 0, P(0x1234), pa, None), INVALID_HANDLE, "an invalid stream")
    # the "extra" argument buffer: CU_LAUNCH_PARAM_BUFFER_POINTER 1, _SIZE 2, _END 0
    buf = (ctypes.c_uint8 * 16)()
    ctypes.memmove(buf, bytes(U64(d)) + bytes(U(50)) + b"\0" * 4, 16)
    size = SZ(16)
    extra = (P * 5)(1, ctypes.cast(buf, P), 2, ctypes.cast(byref(size), P), 0)
    t.eq(L(f, 1, 1, 1, 32, 1, 1, 0, None, None, extra), SUCCESS, "cuLaunchKernel with an extra parameter buffer")
    c.cuCtxSynchronize()
    t.eq(u32s(e.read(d, 8)), [50, 51], "the extra buffer's arguments")
    size.value = 8
    t.eq(L(f, 1, 1, 1, 32, 1, 1, 0, None, None, extra), INVALID_VALUE, "an extra buffer smaller than the parameters")
    # cooperative launches: the grid must be co-resident
    t.eq(c.cuLaunchCooperativeKernel(f, 1, 1, 1, 64, 1, 1, 0, None, pa), SUCCESS, "cuLaunchCooperativeKernel")
    nb = I()
    c.cuOccupancyMaxActiveBlocksPerMultiprocessor(byref(nb), f, 1024, 0)
    sms = I()
    c.cuDeviceGetAttribute(byref(sms), 16, 0)
    t.eq(c.cuLaunchCooperativeKernel(f, nb.value * sms.value + 1, 1, 1, 1024, 1, 1, 0, None, pa),
         COOPERATIVE_LAUNCH_TOO_LARGE, "a cooperative grid larger than the device holds")
    # the functional model runs blocks one after another: only a one-block grid can synchronize grid-wide
    t.eq(c.cuLaunchCooperativeKernel(f, 2, 1, 1, 32, 1, 1, 0, None, pa), COOPERATIVE_LAUNCH_TOO_LARGE,
         "a cooperative grid the simulator cannot run co-resident")

    class LaunchAttr(ctypes.Structure):
        _fields_ = [("id", U), ("pad", U), ("val", U64 * 8)]

    class LaunchConfig(ctypes.Structure):
        _fields_ = [("grid", U * 3), ("block", U * 3), ("shared", U), ("stream", P), ("attrs", P), ("n", U)]
    attr = LaunchAttr(id=2)
    attr.val[0] = 1
    cfg = LaunchConfig(grid=(U * 3)(1, 1, 1), block=(U * 3)(32, 1, 1), shared=0, stream=None,
                       attrs=ctypes.cast(byref(attr), P), n=1)
    pa2 = Env.params(U64(d), U(77))
    t.eq(c.cuLaunchKernelEx(byref(cfg), f, pa2, None), SUCCESS, "cuLaunchKernelEx (cooperative attribute)")
    c.cuCtxSynchronize()
    t.eq(u32s(e.read(d, 4)), [77], "cuLaunchKernelEx ran the kernel")
    attr.id = 4
    attr.val[0] = 2 | (1 << 32)
    attr.val[1] = 1
    cfg.grid = (U * 3)(2, 1, 1)
    t.eq(c.cuLaunchKernelEx(byref(cfg), f, pa2, None), NOT_SUPPORTED, "launch-time cluster dimensions")
    t.eq(c.cuLaunchKernelEx(None, f, pa2, None), INVALID_VALUE, "cuLaunchKernelEx: NULL config")


def g_occupancy(c, t):
    e = Env(c, t)
    f = e.func(e.module(), "fill")
    nb, mg, bs = I(), I(), I()
    t.eq(c.cuOccupancyMaxActiveBlocksPerMultiprocessor(byref(nb), f, 256, 0), SUCCESS,
         "cuOccupancyMaxActiveBlocksPerMultiprocessor")
    mt = I()
    c.cuDeviceGetAttribute(byref(mt), 39, 0)
    t.true(1 <= nb.value and nb.value * 256 <= mt.value, "resident blocks fit MAX_THREADS_PER_MULTIPROCESSOR")
    small = I()
    c.cuOccupancyMaxActiveBlocksPerMultiprocessor(byref(small), f, 64, 0)
    t.true(small.value >= nb.value, "smaller blocks: at least as many resident")
    c.cuOccupancyMaxActiveBlocksPerMultiprocessor(byref(nb), f, 32, 40 << 10)
    t.eq(nb.value, 1, "40 KiB of shared memory per block: one block per 64 KiB multiprocessor")
    t.eq(c.cuOccupancyMaxActiveBlocksPerMultiprocessor(byref(nb), f, 0, 0), INVALID_VALUE, "a block size of 0")
    t.eq(c.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(byref(nb), f, 32, 0, 4), INVALID_VALUE,
         "unknown occupancy flags")
    t.eq(c.cuOccupancyMaxPotentialBlockSize(byref(mg), byref(bs), f, None, 0, 0), SUCCESS,
         "cuOccupancyMaxPotentialBlockSize")
    t.true(bs.value % 32 == 0 and 32 <= bs.value <= 1024, "the block size is a multiple of the warp size")
    c.cuOccupancyMaxActiveBlocksPerMultiprocessor(byref(nb), f, bs.value, 0)
    sms = I()
    c.cuDeviceGetAttribute(byref(sms), 16, 0)
    t.eq(mg.value, nb.value * sms.value, "minGridSize = resident blocks x multiprocessors")
    c.cuOccupancyMaxPotentialBlockSize(byref(mg), byref(bs), f, None, 0, 96)
    t.true(bs.value <= 96, "blockSizeLimit is respected")
    B2D = ctypes.CFUNCTYPE(SZ, I)
    cb = B2D(lambda b: b * 64)
    t.eq(c.cuOccupancyMaxPotentialBlockSize(byref(mg), byref(bs), f, cb, 0, 0), SUCCESS,
         "cuOccupancyMaxPotentialBlockSize with a block-size-to-shared-memory callback")
    t.true(bs.value * 64 <= 48 << 10, "the callback's shared memory fits")
    dyn = SZ()
    t.eq(c.cuOccupancyAvailableDynamicSMemPerBlock(byref(dyn), f, 2, 128), SUCCESS,
         "cuOccupancyAvailableDynamicSMemPerBlock")
    t.eq(dyn.value, 32 << 10, "two blocks share the 64 KiB")


def g_library(c, t):
    Env(c, t, make_ctx=False)
    lib, k, fn, mod = P(), P(), P(), P()
    n, v = U(), I()
    # cuLibraryLoadData needs no context
    t.eq(c.cuLibraryLoadData(byref(lib), FILL.encode(), None, None, 0, None, None, 0), SUCCESS,
         "cuLibraryLoadData without a context")
    t.eq(c.cuLibraryGetKernelCount(byref(n), lib), SUCCESS, "cuLibraryGetKernelCount")
    t.eq(n.value, 3, "three kernels")
    t.eq(c.cuLibraryGetKernel(byref(k), lib, b"fill"), SUCCESS, "cuLibraryGetKernel")
    t.eq(c.cuLibraryGetKernel(byref(k), lib, b"nope"), NOT_FOUND, "cuLibraryGetKernel of an unknown name")
    c.cuLibraryGetKernel(byref(k), lib, b"fill")
    t.eq(c.cuKernelGetFunction(byref(fn), k), INVALID_CONTEXT, "cuKernelGetFunction needs a current context")
    lb = P()
    t.eq(c.cuKernelGetLibrary(byref(lb), k), SUCCESS, "cuKernelGetLibrary")
    t.eq(lb.value, lib.value, "the kernel's library")
    t.eq(c.cuKernelGetAttribute(byref(v), 1, k, 0), SUCCESS, "cuKernelGetAttribute")
    for ctx_i in range(2):
        e = Env(c, t)
        t.eq(c.cuKernelGetFunction(byref(fn), k), SUCCESS, "cuKernelGetFunction in context %d" % ctx_i)
        d = e.alloc(256)
        t.eq(e.launch(fn, 1, 32, [U64(d), U(10 * ctx_i)]), SUCCESS, "a library kernel launches in context %d" % ctx_i)
        t.eq(e.launch(k, 1, 32, [U64(d + 4), U(10 * ctx_i)]), SUCCESS, "cuLaunchKernel takes a CUkernel")
        c.cuCtxSynchronize()
        t.eq(u32s(e.read(d, 8)), [10 * ctx_i, 10 * ctx_i], "the library kernel's results")
        c.cuCtxDestroy_v2(e.ctx)
    e = Env(c, t)
    t.eq(c.cuLibraryGetModule(byref(mod), lib), SUCCESS, "cuLibraryGetModule")
    dp, sz = U64(), SZ()
    t.eq(c.cuLibraryGetGlobal(byref(dp), byref(sz), lib, b"counter"), SUCCESS, "cuLibraryGetGlobal")
    t.eq(sz.value, 4, "the global's size")
    t.eq(c.cuLibraryUnload(lib), SUCCESS, "cuLibraryUnload")
    t.eq(c.cuLibraryUnload(lib), INVALID_HANDLE, "cuLibraryUnload twice")
    # the linker: one PTX input
    st, img, isz = P(), P(), SZ()
    t.eq(c.cuLinkCreate_v2(0, None, None, byref(st)), SUCCESS, "cuLinkCreate")
    src = FILL.encode()
    t.eq(c.cuLinkAddData_v2(st, 1, src, len(src), b"fill.ptx", 0, None, None), SUCCESS, "cuLinkAddData PTX")
    t.eq(c.cuLinkAddData_v2(st, 0, src, len(src), b"x.cubin", 0, None, None), NO_BINARY_FOR_GPU, "a cubin input")
    t.eq(c.cuLinkComplete(st, byref(img), byref(isz)), SUCCESS, "cuLinkComplete")
    m = P()
    t.eq(c.cuModuleLoadData(byref(m), ctypes.cast(img, ctypes.c_char_p)), SUCCESS, "the linked image loads")
    t.eq(c.cuLinkDestroy(st), SUCCESS, "cuLinkDestroy")


def g_graphs(c, t):
    e = Env(c, t)
    f = e.func(e.module(), "fill")
    add1 = e.func(e.module(), "add1")
    d = e.alloc(256)
    g, ex = P(), P()
    t.eq(c.cuGraphCreate(byref(g), 1), INVALID_VALUE, "cuGraphCreate: flags must be 0")
    t.eq(c.cuGraphCreate(byref(g), 0), SUCCESS, "cuGraphCreate")

    class KP(ctypes.Structure):  # CUDA_KERNEL_NODE_PARAMS_v2
        _fields_ = [("func", P), ("grid", U * 3), ("block", U * 3), ("shared", U), ("params", P), ("extra", P),
                    ("kern", P), ("ctx", P)]

    class MS(ctypes.Structure):  # CUDA_MEMSET_NODE_PARAMS
        _fields_ = [("dst", U64), ("pitch", SZ), ("value", U), ("esize", U), ("width", SZ), ("height", SZ)]
    ms = MS(dst=d, pitch=0, value=0, esize=4, width=64, height=1)
    n_set, n_k1, n_k2, n_host = P(), P(), P(), P()
    t.eq(c.cuGraphAddMemsetNode(byref(n_set), g, None, 0, byref(ms), e.ctx), SUCCESS, "cuGraphAddMemsetNode")
    pa = Env.params(U64(d), U(100))
    kp = KP(func=f, grid=(U * 3)(1, 1, 1), block=(U * 3)(64, 1, 1), shared=0, params=ctypes.cast(pa, P))
    dep = (P * 1)(n_set)
    t.eq(c.cuGraphAddKernelNode_v2(byref(n_k1), g, dep, 1, byref(kp)), SUCCESS, "cuGraphAddKernelNode")
    pa2 = Env.params(U64(d))
    kp2 = KP(func=add1, grid=(U * 3)(1, 1, 1), block=(U * 3)(64, 1, 1), shared=0, params=ctypes.cast(pa2, P))
    dep = (P * 1)(n_k1)
    t.eq(c.cuGraphAddKernelNode_v2(byref(n_k2), g, dep, 1, byref(kp2)), SUCCESS, "a dependent kernel node")
    hits = []
    HOSTFN = ctypes.CFUNCTYPE(None, P)
    hf = HOSTFN(lambda data: hits.append(data))

    class HP(ctypes.Structure):
        _fields_ = [("fn", HOSTFN), ("data", P)]
    hp = HP(fn=hf, data=P(5))
    dep = (P * 1)(n_k2)
    t.eq(c.cuGraphAddHostNode(byref(n_host), g, dep, 1, byref(hp)), SUCCESS, "cuGraphAddHostNode")
    t.eq(c.cuGraphAddKernelNode_v2(byref(n_k2), g, (P * 1)(P(0x1234)), 1, byref(kp2)), INVALID_VALUE,
         "a dependency that is not a node of the graph")
    cnt, ty = SZ(), I()
    t.eq(c.cuGraphGetNodes(g, None, byref(cnt)), SUCCESS, "cuGraphGetNodes count")
    t.eq(cnt.value, 4, "four nodes")
    c.cuGraphGetRootNodes(g, None, byref(cnt))
    t.eq(cnt.value, 1, "one root (the memset)")
    c.cuGraphGetEdges(g, None, None, byref(cnt))
    t.eq(cnt.value, 3, "three edges")
    c.cuGraphNodeGetType(n_host, byref(ty))
    t.eq(ty.value, 3, "CU_GRAPH_NODE_TYPE_HOST")
    c.cuGraphNodeGetType(n_set, byref(ty))
    t.eq(ty.value, 2, "CU_GRAPH_NODE_TYPE_MEMSET")
    t.eq(c.cuGraphInstantiateWithFlags(byref(ex), g, 0), SUCCESS, "cuGraphInstantiateWithFlags")
    for i in range(2):
        t.eq(c.cuGraphLaunch(ex, None), SUCCESS, "cuGraphLaunch %d" % i)
    c.cuCtxSynchronize()
    t.eq(u32s(e.read(d, 16)), [101, 102, 103, 104], "memset, fill, add1 in dependency order (each launch)")
    t.eq(hits, [5, 5], "the host node ran once per launch")
    t.eq(c.cuGraphExecDestroy(ex), SUCCESS, "cuGraphExecDestroy")
    t.eq(c.cuGraphLaunch(ex, None), INVALID_VALUE, "launching a destroyed executable graph")
    c.cuGraphDestroy(g)
    # stream capture, with a fork and join through events
    s1, s2, ev1, ev2, cg = P(), P(), P(), P(), P()
    c.cuStreamCreate(byref(s1), 0)
    c.cuStreamCreate(byref(s2), 0)
    c.cuEventCreate(byref(ev1), 0)
    c.cuEventCreate(byref(ev2), 0)
    st = I()
    d2 = e.alloc(256)
    t.eq(c.cuStreamBeginCapture_v2(None, 0), CAPTURE_UNSUPPORTED, "capture may not begin on the legacy stream")
    t.eq(c.cuStreamBeginCapture_v2(s1, 0), SUCCESS, "cuStreamBeginCapture")
    t.eq(c.cuStreamBeginCapture_v2(s1, 0), ILLEGAL_STATE, "cuStreamBeginCapture of a capturing stream")
    c.cuStreamIsCapturing(s1, byref(st))
    t.eq(st.value, 1, "CU_STREAM_CAPTURE_STATUS_ACTIVE")
    t.eq(c.cuMemsetD32Async(d, 0, 64, s1), SUCCESS, "a captured memset")
    t.eq(e.launch(f, 1, 32, [U64(d), U(1000)], stream=s1), SUCCESS, "a captured launch")
    c.cuEventRecord(ev1, s1)
    t.eq(c.cuEventQuery(ev1), CAPTURED_EVENT, "cuEventQuery of an event recorded in a capture")
    t.eq(c.cuStreamWaitEvent(s2, ev1, 0), SUCCESS, "fork: s2 waits for a captured event")
    c.cuStreamIsCapturing(s2, byref(st))
    t.eq(st.value, 1, "the forked stream joined the capture")
    e.launch(f, 1, 32, [U64(d2), U(2000)], stream=s2)
    c.cuEventRecord(ev2, s2)
    status, cid, gr, deps, nd = I(), U64(), P(), P(), SZ()
    t.eq(c.cuStreamGetCaptureInfo_v2(s1, byref(status), byref(cid), byref(gr), byref(deps), byref(nd)), SUCCESS,
         "cuStreamGetCaptureInfo_v2")
    t.eq((status.value, nd.value), (1, 1), "active, one dependency node")
    t.eq(c.cuStreamEndCapture(s2, byref(cg)), CAPTURE_UNMATCHED, "ending the capture on a stream it did not begin on")
    c.cuStreamWaitEvent(s1, ev2, 0)  # join
    t.eq(c.cuStreamEndCapture(s1, byref(cg)), CAPTURE_INVALIDATED,
         "an illegal end of capture invalidated the capture")
    # again, properly
    c.cuStreamBeginCapture_v2(s1, 0)
    c.cuMemsetD32Async(d, 0, 64, s1)
    e.launch(f, 1, 32, [U64(d), U(1000)], stream=s1)
    c.cuEventRecord(ev1, s1)
    c.cuStreamWaitEvent(s2, ev1, 0)
    e.launch(add1, 1, 32, [U64(d)], stream=s2)
    c.cuEventRecord(ev2, s2)
    c.cuStreamWaitEvent(s1, ev2, 0)
    e.launch(add1, 1, 32, [U64(d)], stream=s1)
    hb = ctypes.create_string_buffer(16)
    t.eq(c.cuStreamEndCapture(s1, byref(cg)), SUCCESS, "cuStreamEndCapture after the join")
    c.cuStreamIsCapturing(s2, byref(st))
    t.eq(st.value, 0, "the forked stream's capture ended too")
    c.cuGraphGetNodes(cg, None, byref(cnt))
    t.eq(cnt.value, 4, "the captured graph has the four operations")
    t.eq(u32s(e.read(d, 4)) != [1002], True, "nothing ran during the capture")
    c.cuGraphInstantiateWithFlags(byref(ex), cg, 0)
    t.eq(c.cuGraphLaunch(ex, s1), SUCCESS, "launching the captured graph")
    c.cuStreamSynchronize(s1)
    c.cuMemcpyDtoH_v2(hb, d, 16)
    t.eq(u32s(hb.raw), [1002, 1003, 1004, 1005], "memset, fill, add1 (forked), add1 (joined)")
    # unjoined fork
    c.cuStreamBeginCapture_v2(s1, 0)
    c.cuEventRecord(ev1, s1)
    c.cuStreamWaitEvent(s2, ev1, 0)
    e.launch(add1, 1, 32, [U64(d)], stream=s2)
    t.eq(c.cuStreamEndCapture(s1, byref(cg)), CAPTURE_UNJOINED, "a forked stream that did not join back")
    # prohibited calls during capture
    c.cuStreamBeginCapture_v2(s1, 0)
    t.eq(c.cuStreamSynchronize(s1), CAPTURE_UNSUPPORTED, "cuStreamSynchronize of a capturing stream")
    t.eq(c.cuStreamEndCapture(s1, byref(cg)), CAPTURE_INVALIDATED, "which invalidated the capture")
    t.eq(cg.value, None, "an invalidated capture returns no graph")
    c.cuStreamBeginCapture_v2(s1, 0)
    p = U64()
    t.eq(c.cuMemAlloc_v2(byref(p), 16), CAPTURE_UNSUPPORTED, "cuMemAlloc during a global-mode capture")
    c.cuStreamEndCapture(s1, byref(cg))
    c.cuStreamBeginCapture_v2(s1, 0)
    mode = I(2)
    t.eq(c.cuThreadExchangeStreamCaptureMode(byref(mode)), SUCCESS, "cuThreadExchangeStreamCaptureMode")
    t.eq(mode.value, 0, "the previous mode was global")
    t.eq(c.cuMemAlloc_v2(byref(p), 16), SUCCESS, "cuMemAlloc in relaxed mode during a capture")
    c.cuThreadExchangeStreamCaptureMode(byref(mode))
    t.eq(c.cuMemcpyHtoD_v2(p.value, b"1234", 4), CAPTURE_IMPLICIT,
         "a synchronous (legacy stream) copy while a blocking stream captures")
    c.cuStreamEndCapture(s1, byref(cg))
    t.eq(c.cuStreamEndCapture(s1, byref(cg)), ILLEGAL_STATE, "cuStreamEndCapture of a stream that is not capturing")
    keep = [hf, hp]  # noqa: F841


def g_proc_address(c, t):
    Env(c, t, make_ctx=False)
    pfn = P()
    st = I()
    G = c.cuGetProcAddress_v2
    t.eq(G(b"cuMemAlloc", byref(pfn), 12000, 0, byref(st)), SUCCESS, "cuGetProcAddress cuMemAlloc")
    t.eq(pfn.value, ctypes.cast(c.l.cuMemAlloc_v2, P).value, "cuMemAlloc at CUDA 12 is cuMemAlloc_v2")
    G(b"cuCtxCreate", byref(pfn), 11000, 0, byref(st))
    t.eq(pfn.value, ctypes.cast(c.l.cuCtxCreate_v2, P).value, "cuCtxCreate at 11.0 is _v2")
    G(b"cuCtxCreate", byref(pfn), 12000, 0, byref(st))
    t.eq(pfn.value, ctypes.cast(c.l.cuCtxCreate_v3, P).value, "cuCtxCreate at 12.0 is _v3 (11.4)")
    G(b"cuGraphInstantiate", byref(pfn), 12000, 0, byref(st))
    t.eq(pfn.value, ctypes.cast(c.l.cuGraphInstantiateWithFlags, P).value,
         "cuGraphInstantiate at 12.0 is cuGraphInstantiateWithFlags")
    t.eq(G(b"cuNoSuchFunction", byref(pfn), 12000, 0, byref(st)), NOT_FOUND, "an unknown symbol")
    t.eq((pfn.value, st.value), (None, 1), "NULL and CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND")
    t.eq(G(b"cuGetProcAddress", byref(pfn), 11000, 0, byref(st)), NOT_FOUND, "a symbol newer than the version")
    t.eq(st.value, 2, "CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT")
    t.eq(G(b"cuInit", byref(pfn), 12000, 8, byref(st)), INVALID_VALUE, "unknown flags")
    t.eq(G(b"cublasCreate_v2", byref(pfn), 12000, 0, byref(st)), NOT_FOUND, "cuBLAS is not a driver entry point")
    # every exported driver entry point is found by its own name
    syms = subprocess.run(["nm", "-D", "--defined-only", c.lib], capture_output=True, text=True).stdout.split()
    exported = sorted({s for s in syms if s.startswith("cu") and not s.startswith("cublas")})
    missing = []
    for s in exported:
        if G(s.encode(), byref(pfn), 12080, 0, byref(st)) != 0:
            missing.append(s)
            continue
        # a base name of a versioned family resolves to the family's newest version
        same = [ctypes.cast(getattr(c.l, x), P).value for x in (s, s + "_v2", s + "_v3", s + "_v4", s + "WithFlags")
                if hasattr(c.l, x)]
        if pfn.value not in same:
            missing.append(s)
    t.true(len(exported) > 200, "the driver API surface (%d entry points)" % len(exported))


DEINIT_C = r"""
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
typedef unsigned (*init_t)(unsigned);
typedef unsigned (*alloc_t)(unsigned long long *, size_t);
static alloc_t alloc_fn;
static void late(void) { unsigned long long p; printf("late %u\n", alloc_fn(&p, 16)); fflush(stdout); }
int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { printf("dlopen failed\n"); return 1; }
    init_t init = (init_t)dlsym(h, "cuInit");
    alloc_fn = (alloc_t)dlsym(h, "cuMemAlloc_v2");
    atexit(late);   /* runs after the driver's own exit handler (registered later, by cuInit) */
    printf("init %u\n", init(0));
    return 0;
}
"""


def g_deinit(c, t):
    # CUDA_ERROR_DEINITIALIZED: "the CUDA driver is in the process of shutting down"
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "deinit.c"), os.path.join(d, "deinit")
        open(src, "w").write(DEINIT_C)
        r = subprocess.run(["clang", "-o", exe, src, "-ldl"], capture_output=True, text=True)
        if not t.eq(r.returncode, 0, "build the exit-handler program with clang"):
            return
        r = subprocess.run([exe, c.lib], capture_output=True, text=True, timeout=120)
        t.eq(r.stdout.split(), ["init", "0", "late", "4"], "a call from an exit handler after the driver's teardown")


def g_context_resources(c, t):
    e = Env(c, t)
    d = e.alloc(1024)
    m = e.module()
    s, ev = P(), P()
    c.cuStreamCreate(byref(s), 0)
    c.cuEventCreate(byref(ev), 0)
    f = e.func(m, "fill")
    other = P()
    c.cuCtxCreate_v2(byref(other), 0, 0)
    t.eq(e.launch(f, 1, 32, [U64(d), U(0)], stream=s), INVALID_HANDLE, "a stream of another context")
    c.cuCtxPopCurrent_v2(None)
    t.eq(c.cuCtxDestroy_v2(e.ctx), SUCCESS, "cuCtxDestroy")
    c.cuCtxPushCurrent_v2(other)
    base, n = U64(), SZ()
    t.eq(c.cuMemGetAddressRange_v2(byref(base), byref(n), d), NOT_FOUND, "the destroyed context's memory is freed")
    t.eq(c.cuModuleUnload(m), INVALID_HANDLE, "its modules are unloaded")
    t.eq(c.cuStreamQuery(s), INVALID_HANDLE, "its streams are destroyed")
    t.eq(c.cuEventQuery(ev), INVALID_HANDLE, "its events are destroyed")


GROUPS = {k[2:]: v for k, v in globals().items() if k.startswith("g_")}


def child(group, lib, verbose):
    c = Driver(lib)
    c.lib = lib
    t = Checks(group, verbose)
    try:
        GROUPS[group](c, t)
    except Exception as ex:  # an exception is a failed check
        import traceback
        traceback.print_exc()
        t.results.append((False, "exception: %r" % ex, None, None))
    print("RESULT " + json.dumps([sum(1 for r in t.results if r[0]), len(t.results)]), flush=True)
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.path.join(HERE, "..", "..", "target", "release", "libzluda_metal.so"))
    ap.add_argument("--child")
    ap.add_argument("-k", default="")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    a.lib = os.path.abspath(a.lib)
    if a.child:
        child(a.child, a.lib, a.v)
    env = dict(os.environ, ZLUDA_METAL_BACKEND="agx")
    sib = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "llvm-project", "build", "bin", "llvm-air2agx"))
    if os.path.exists(sib):
        env.setdefault("ZLUDA_AGX_AIR2AGX", sib)
    passed = total = 0
    failed = []
    for g in GROUPS:
        if a.k not in g:
            continue
        r = subprocess.run([sys.executable, __file__, "--child", g, "--lib", a.lib] + (["-v"] if a.v else []),
                           capture_output=True, text=True, env=env, timeout=1800)
        res = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
        ok, n = json.loads(res[-1][7:]) if res else (0, 1)
        passed += ok
        total += n
        print("%-5s %-20s %d/%d" % ("PASS" if ok == n else "FAIL", g, ok, n), flush=True)
        if ok != n or a.v:
            out = "\n".join(x for x in r.stdout.splitlines() if not x.startswith("RESULT "))
            if out:
                print(out)
            if ok != n and r.stderr.strip():
                print("      stderr:", r.stderr.strip()[-1500:])
        if ok != n:
            failed.append(g)
    print("---- %s: %d/%d checks" % ("PASS" if not failed else "FAIL (%s)" % ", ".join(failed), passed, total))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
