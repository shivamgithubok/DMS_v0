"""Native TensorRT engine runner (no PyCUDA / torch dependency).

Talks to libcudart through ctypes, which is always present on JetPack. The stream
is created with the *highest* CUDA priority so the real-time detector's kernels
are scheduled ahead of lower-priority GPU work in this process; the VLM runs in a
separate server process (its own CUDA context), which keeps it from ever holding
the GIL or the detector's stream.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

_H2D = 1
_D2H = 2


def import_tensorrt():
    try:
        import tensorrt as trt
    except ModuleNotFoundError:
        for site in ("/usr/lib/python3.10/dist-packages", "/usr/lib/python3/dist-packages",
                     "/usr/lib/python3.12/dist-packages"):
            if Path(site).exists() and site not in sys.path:
                sys.path.append(site)
        import tensorrt as trt
    return trt


def _load_cudart():
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.13", "libcudart.so.11.0",
                 "/usr/local/cuda/lib64/libcudart.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError("libcudart not found")


class _CudaRT:
    def __init__(self) -> None:
        lib = _load_cudart()
        vp = ctypes.c_void_p
        lib.cudaMalloc.argtypes = [ctypes.POINTER(vp), ctypes.c_size_t]
        lib.cudaFree.argtypes = [vp]
        lib.cudaMemcpyAsync.argtypes = [vp, vp, ctypes.c_size_t, ctypes.c_int, vp]
        lib.cudaStreamSynchronize.argtypes = [vp]
        lib.cudaStreamDestroy.argtypes = [vp]
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        lib.cudaDeviceGetStreamPriorityRange.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        lib.cudaStreamCreateWithPriority.argtypes = [ctypes.POINTER(vp), ctypes.c_uint, ctypes.c_int]
        self.lib = lib

    def check(self, code: int, what: str) -> None:
        if code != 0:
            msg = self.lib.cudaGetErrorString(code)
            raise RuntimeError(f"{what} failed: {msg.decode() if msg else code}")

    def high_priority_stream(self):
        least, greatest = ctypes.c_int(), ctypes.c_int()
        self.check(self.lib.cudaDeviceGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest)),
                   "cudaDeviceGetStreamPriorityRange")
        stream = ctypes.c_void_p()
        # flag 1 = cudaStreamNonBlocking; greatest is numerically the lowest value = highest priority
        self.check(self.lib.cudaStreamCreateWithPriority(ctypes.byref(stream), 1, greatest.value),
                   "cudaStreamCreateWithPriority")
        return stream


class EngineSession:
    """Run a serialized TensorRT engine with static input shapes."""

    provider = "TensorRT"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.trt = import_tensorrt()
        self.cuda = _CudaRT()
        logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.runtime = self.trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize TensorRT engine {self.path} (built for another TRT version?)")
        self.context = self.engine.create_execution_context()
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.shapes: dict[str, tuple] = {}
        self.dtypes: dict[str, np.dtype] = {}
        self.device: dict[str, int] = {}
        self.host: dict[str, np.ndarray] = {}
        mode_in = self.trt.TensorIOMode.INPUT
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            (self.inputs if self.engine.get_tensor_mode(name) == mode_in else self.outputs).append(name)
        for name in self.inputs:
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):  # dynamic: pin to the max profile shape
                shape = tuple(self.engine.get_tensor_profile_shape(name, 0)[2])
                self.context.set_input_shape(name, shape)
            self._alloc(name, shape)
        for name in self.outputs:
            self._alloc(name, tuple(self.context.get_tensor_shape(name)))
        self.stream = self.cuda.high_priority_stream()
        self.fp16 = any(self.dtypes[n] == np.float16 for n in self.inputs + self.outputs) or "fp16" in self.path.name

    def _np_dtype(self, trt_dtype):
        table = {self.trt.DataType.FLOAT: np.float32, self.trt.DataType.HALF: np.float16,
                 self.trt.DataType.INT32: np.int32, self.trt.DataType.INT8: np.int8}
        if trt_dtype not in table:
            raise RuntimeError(f"unsupported TensorRT dtype {trt_dtype}")
        return table[trt_dtype]

    def _alloc(self, name: str, shape: tuple) -> None:
        dtype = self._np_dtype(self.engine.get_tensor_dtype(name))
        host = np.empty(shape, dtype=dtype)
        ptr = ctypes.c_void_p()
        self.cuda.check(self.cuda.lib.cudaMalloc(ctypes.byref(ptr), max(1, host.nbytes)), "cudaMalloc")
        self.device[name] = ptr.value
        self.host[name] = host
        self.shapes[name] = shape
        self.dtypes[name] = dtype
        self.context.set_tensor_address(name, ptr.value)

    def input_shape(self, name: str | None = None) -> tuple:
        return self.shapes[name or self.inputs[0]]

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        lib = self.cuda.lib
        for name in self.inputs:
            arr = np.ascontiguousarray(feeds[name], dtype=self.dtypes[name])
            if arr.shape != self.shapes[name]:
                raise ValueError(f"{name}: expected {self.shapes[name]}, got {arr.shape}")
            self.host[name] = arr  # keep alive until the copy completes
            self.cuda.check(lib.cudaMemcpyAsync(self.device[name], arr.ctypes.data, arr.nbytes, _H2D, self.stream),
                            "cudaMemcpyAsync H2D")
        if not self.context.execute_async_v3(self.stream.value):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        for name in self.outputs:
            host = self.host[name]
            self.cuda.check(lib.cudaMemcpyAsync(host.ctypes.data, self.device[name], host.nbytes, _D2H, self.stream),
                            "cudaMemcpyAsync D2H")
        self.cuda.check(lib.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        return {name: self.host[name].astype(np.float32, copy=True) for name in self.outputs}

    def close(self) -> None:
        for ptr in self.device.values():
            self.cuda.lib.cudaFree(ptr)
        self.device.clear()
        if self.stream:
            self.cuda.lib.cudaStreamDestroy(self.stream)
            self.stream = None

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass
