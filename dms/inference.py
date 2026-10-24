"""Model session factory: native TensorRT engine first, ONNX Runtime second.

ONNX Runtime is configured with the TensorRT execution provider (FP16, on-disk engine
cache) when available, then CUDA, then CPU, so the same code runs on the Orin and on
a development workstation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("dms.inference")


class OrtSession:
    def __init__(self, path: Path, cache_dir: Path | None = None, fp16: bool = True,
                 threads: int = 2, allow_trt: bool = True, allow_cuda: bool = True) -> None:
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        available = ort.get_available_providers()
        providers: list = []
        if allow_trt and "TensorrtExecutionProvider" in available:
            cache = str(cache_dir or path.parent / "trt_cache")
            Path(cache).mkdir(parents=True, exist_ok=True)
            providers.append(("TensorrtExecutionProvider", {
                "trt_fp16_enable": fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache,
                "trt_timing_cache_enable": True,
                "trt_max_workspace_size": 256 << 20,
            }))
        if allow_cuda and "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
        providers.append("CPUExecutionProvider")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.outputs = [o.name for o in self.session.get_outputs()]
        self._shapes = {i.name: tuple(d if isinstance(d, int) else 1 for d in i.shape)
                        for i in self.session.get_inputs()}
        active = self.session.get_providers()[0]
        self.provider = {"TensorrtExecutionProvider": "ORT-TensorRT", "CUDAExecutionProvider": "ORT-CUDA",
                         "CPUExecutionProvider": "ORT-CPU"}.get(active, active)
        self.fp16 = fp16 and active == "TensorrtExecutionProvider"

    def input_shape(self, name: str | None = None) -> tuple:
        return self._shapes[name or self.inputs[0]]

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        values = self.session.run(self.outputs, feeds)
        return dict(zip(self.outputs, values))

    def close(self) -> None:
        self.session = None


def create_session(onnx_path: Path | None, engine_path: Path | None = None, *, threads: int = 2,
                   allow_trt: bool = True, allow_cuda: bool = True):
    """Return (session, label). Engine wins if it loads; any failure falls back to ORT."""
    if engine_path and Path(engine_path).exists():
        try:
            from .trt_session import EngineSession

            session = EngineSession(engine_path)
            log.info("loaded TensorRT engine %s", engine_path)
            return session
        except Exception as exc:
            log.warning("TensorRT engine %s unusable (%s); falling back to ONNX Runtime", engine_path, exc)
    if onnx_path is None or not Path(onnx_path).exists():
        raise FileNotFoundError(f"model not found: {onnx_path}")
    session = OrtSession(Path(onnx_path), threads=threads, allow_trt=allow_trt, allow_cuda=allow_cuda)
    log.info("loaded %s with %s", Path(onnx_path).name, session.provider)
    return session


class Timer:
    """Rolling latency statistics in milliseconds."""

    def __init__(self, size: int = 120) -> None:
        self.size = size
        self.samples: list[float] = []

    def add(self, ms: float) -> None:
        self.samples.append(ms)
        if len(self.samples) > self.size:
            del self.samples[0]

    def measure(self):
        return _Span(self)

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.samples, 95)) if self.samples else 0.0


class _Span:
    def __init__(self, timer: Timer) -> None:
        self.timer = timer

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000.0
        self.timer.add(self.ms)
        return False
