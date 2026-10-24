"""Jetson system telemetry from sysfs/procfs: temperatures, GPU load, unified memory.

No jtop dependency. Paths are probed once; anything missing reads as None so the same
code runs on a workstation.
"""

from __future__ import annotations

import glob
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass

log = logging.getLogger("dms.telemetry")

_GPU_LOAD_CANDIDATES = (
    "/sys/devices/platform/bus@0/17000000.gpu/load",  # Orin, JetPack 6
    "/sys/devices/platform/17000000.ga10b/load",
    "/sys/devices/17000000.ga10b/load",
    "/sys/devices/platform/gpu.0/load",
    "/sys/devices/gpu.0/load",
)


@dataclass
class SystemStats:
    gpu_temp_c: float | None = None
    cpu_temp_c: float | None = None
    soc_temp_c: float | None = None
    gpu_load_pct: float | None = None
    mem_used_mb: float = 0.0
    mem_total_mb: float = 0.0
    swap_used_mb: float = 0.0
    process_rss_mb: float = 0.0
    timestamp: float = 0.0

    @property
    def max_temp_c(self) -> float | None:
        temps = [t for t in (self.gpu_temp_c, self.cpu_temp_c, self.soc_temp_c) if t is not None]
        return max(temps) if temps else None


def hardware_label() -> str:
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(path, "rb") as handle:
                text = handle.read().decode(errors="ignore").strip("\x00 \n")
            if text:
                text = text.replace(" Developer Kit", "").replace(" Engineering Reference", "")
                return text
        except OSError:
            continue
    cpu = platform.processor() or platform.machine()
    import re
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    cpu = re.sub(r"\((?:R|TM)\)|\bProcessor\b|\bCPU\b|@.*$", "", cpu)
    return re.sub(r"\s+", " ", cpu).strip()


def is_jetson() -> bool:
    return os.path.exists("/etc/nv_tegra_release") or "tegra" in platform.release()


class Telemetry:
    def __init__(self, period_s: float = 1.0) -> None:
        self.period_s = period_s
        self.stats = SystemStats()
        self._zones = self._find_zones()
        self._gpu_load = next((p for p in _GPU_LOAD_CANDIDATES if os.path.exists(p)), None)
        if self._gpu_load is None:
            found = glob.glob("/sys/devices/platform/bus@0/*.gpu/load") + glob.glob("/sys/devices/platform/*.gpu/load")
            self._gpu_load = found[0] if found else None
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)

    @staticmethod
    def _find_zones() -> dict[str, str]:
        zones = {}
        for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
            try:
                with open(os.path.join(zone, "type")) as handle:
                    kind = handle.read().strip().lower()
            except OSError:
                continue
            if "gpu" in kind:
                zones.setdefault("gpu", os.path.join(zone, "temp"))
            elif "cpu" in kind:
                zones.setdefault("cpu", os.path.join(zone, "temp"))
            elif kind.startswith(("soc", "tj")) or "x86_pkg" in kind:
                zones.setdefault("soc", os.path.join(zone, "temp"))
        return zones

    @staticmethod
    def _read_float(path: str | None, scale: float = 1.0) -> float | None:
        if not path:
            return None
        try:
            with open(path) as handle:
                return float(handle.read().strip()) / scale
        except (OSError, ValueError):
            return None

    def sample(self) -> SystemStats:
        stats = SystemStats(timestamp=time.time())
        stats.gpu_temp_c = self._read_float(self._zones.get("gpu"), 1000.0)
        stats.cpu_temp_c = self._read_float(self._zones.get("cpu"), 1000.0)
        stats.soc_temp_c = self._read_float(self._zones.get("soc"), 1000.0)
        load = self._read_float(self._gpu_load)
        stats.gpu_load_pct = None if load is None else load / 10.0  # sysfs reports per-mille
        meminfo = {}
        try:
            with open("/proc/meminfo") as handle:
                for line in handle:
                    key, value = line.split(":", 1)
                    meminfo[key] = float(value.split()[0]) / 1024.0
        except OSError:
            pass
        stats.mem_total_mb = meminfo.get("MemTotal", 0.0)
        stats.mem_used_mb = stats.mem_total_mb - meminfo.get("MemAvailable", stats.mem_total_mb)
        stats.swap_used_mb = meminfo.get("SwapTotal", 0.0) - meminfo.get("SwapFree", 0.0)
        try:
            with open("/proc/self/status") as handle:
                for line in handle:
                    if line.startswith("VmRSS"):
                        stats.process_rss_mb = float(line.split()[1]) / 1024.0
        except OSError:
            pass
        self.stats = stats
        return stats

    def start(self) -> "Telemetry":
        self.sample()
        self._thread.start()
        return self

    def stop(self) -> None:
        self._halt.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        last_log = 0.0
        while not self._halt.wait(self.period_s):
            stats = self.sample()
            if time.monotonic() - last_log > 30:
                last_log = time.monotonic()
                log.info("telemetry gpu=%sC cpu=%sC gpu_load=%s%% mem=%.0f/%.0fMB rss=%.0fMB",
                         _fmt(stats.gpu_temp_c), _fmt(stats.cpu_temp_c), _fmt(stats.gpu_load_pct),
                         stats.mem_used_mb, stats.mem_total_mb, stats.process_rss_mb)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.0f}"
