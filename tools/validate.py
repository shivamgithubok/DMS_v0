#!/usr/bin/env python3
"""Run the full system headless for N seconds and check the spec's validation criteria.

    python tools/validate.py --duration 120                       # on the Jetson, real sensors
    python tools/validate.py --driver assets/demo/driver_reference.jpg \
        --audio-device tests/data/cabin_events.wav --duration 60   # bench run

Criteria
  real-time detector >= 15 FPS on the 640x480 driver feed (5th percentile of 1 s samples)
  VLM answer <= 3 s per inference (p95)
  audio alert latency < 500 ms (max over alerts raised)
  MQTT publish success >= 99 % and empty outbox at the end (when a broker is reachable)
  unified memory < 7.5 GB (peak)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dms.bootstrap import ensure_venv  # noqa: E402

ensure_venv()

import main as dms_main  # noqa: E402
from dms import alerts as A  # noqa: E402
from dms.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=8.0)
    ap.add_argument("--report", default="data/validation_report.json")
    known, rest = ap.parse_known_args()
    args = dms_main.parse_args(rest + ["--headless"])
    cfg_path = args.config if (ROOT / args.config).exists() else None
    cfg = load_config(cfg_path, dms_main.build_overrides(args))
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname).1s %(name)s: %(message)s")
    system = dms_main.System(cfg)
    system.start()
    fps, mem = [], []
    t_end = time.monotonic() + known.warmup + known.duration
    started = time.monotonic()
    try:
        while time.monotonic() < t_end:
            time.sleep(1.0)
            if time.monotonic() - started < known.warmup:
                continue
            fps.append(system.shared.driver.fps)
            if system.shared.system is not None:
                mem.append(system.shared.system.mem_used_mb)
            print(f"\r{time.monotonic() - started:5.0f}s  fps={fps[-1]:5.1f}  alerts={len(system.bus.history)}", end="")
        time.sleep(2.0)  # let the outbox drain
    finally:
        print()
        vlm_worker = next((w for w in system.workers if w.name == "vlm-worker"), None)
        mqtt = system.shared.mqtt
        system.stop()

    audio_lat = [a.extra.get("detection_latency_ms", 0) for a in system.bus.history if a.category == A.AUDIO
                 and a.type != A.ABUSIVE_LANGUAGE]
    vlm_lat = list(vlm_worker.latencies) if vlm_worker else []
    checks = []

    def check(name, ok, value, target):
        checks.append({"criterion": name, "pass": ok, "value": value, "target": target})

    p5 = float(np.percentile(fps, 5)) if fps else 0.0
    check("real-time FPS (p5)", p5 >= 15.0, round(p5, 1), ">= 15")
    if cfg.vlm.enabled:
        p95 = float(np.percentile(vlm_lat, 95)) if vlm_lat else None
        check("VLM latency p95 [s]", p95 is not None and p95 <= 3.0, None if p95 is None else round(p95, 2), "<= 3.0")
    if cfg.audio.enabled:
        worst = max(audio_lat) if audio_lat else None
        check("audio alert latency max [ms]", worst is not None and worst < 500, worst,
              "< 500 (needs an audio event during the run)")
    if cfg.mqtt.enabled:
        check("MQTT success rate [%]", mqtt.connected and mqtt.success_rate >= 99.0, round(mqtt.success_rate, 2), ">= 99")
        check("MQTT outbox drained", mqtt.queued == 0, mqtt.queued, "0")
    peak = max(mem) if mem else None
    check("peak unified memory [MB]", peak is not None and peak < 7500, None if peak is None else round(peak), "< 7500")

    width = max(len(c["criterion"]) for c in checks)
    for c in checks:
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['criterion']:<{width}}  {c['value']}  (target {c['target']})")
    report = {"hardware": system.shared.hardware, "duration_s": known.duration, "fps_mean": round(float(np.mean(fps)), 1)
              if fps else 0, "alerts": dict(system.bus.counts), "checks": checks}
    out = ROOT / known.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"report: {out}")
    return 0 if all(c["pass"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
