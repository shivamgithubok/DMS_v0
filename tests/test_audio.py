import time
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dms import alerts as A
from dms.audio_detector import AudioDetector, RingBuffer
from dms.config import load_config
from dms.state import SharedState
from synth_audio import SR, mix, place, road_noise, scream, shout


def _feed(det, signal, block=800):
    """Push audio in real time (50 ms blocks) like the microphone callback does."""
    t0 = time.monotonic()
    for i in range(0, len(signal), block):
        target = t0 + (i + block) / SR
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        det.ingest(signal[i:i + block], time.monotonic())
    return t0


def _detector():
    cfg = load_config(None, {"audio": {"backend": "dsp", "kws": {"enabled": False}, "save_snippets": False}})
    bus = A.AlertBus("test", cooldown_s=1.0)
    det = AudioDetector(cfg, SharedState(), bus)
    det.classifier_thread.start()
    return det, bus


def test_ring_buffer_wraps():
    rb = RingBuffer(1.0, 10)
    rb.write(np.arange(7, dtype=np.float32), 1.0)
    rb.write(np.arange(7, 14, dtype=np.float32), 2.0)
    out, t = rb.latest(10)
    assert list(out) == list(range(4, 14)) and t == 2.0


def test_scream_alert_latency_under_500ms():
    det, bus = _detector()
    try:
        audio = place(road_noise(6.0, seed=3), scream(1.5), 4.0)
        t0 = _feed(det, audio)
        time.sleep(0.6)
        screams = [a for a in bus.history if a.type == A.SCREAM]
        assert screams, "no SCREAM alert"
        onset = t0 + 4.0
        latency = screams[0].timestamp - (time.time() - (time.monotonic() - onset))
        assert latency < 0.5, f"latency {latency:.3f}s"
        assert screams[0].extra["detection_latency_ms"] < 500
        assert not [a for a in bus.history if a.timestamp < screams[0].timestamp - 0.6]  # no false alarm on noise
    finally:
        det.stop()


def test_altercation_from_overlapping_shouts():
    det, bus = _detector()
    try:
        base = road_noise(9.0, seed=4)
        base = place(base, shout(4.0, 180, -19, seed=9), 4.0)
        base = place(base, shout(4.0, 360, -19, seed=10), 4.3)
        _feed(det, base)
        time.sleep(0.4)
        assert any(a.type == A.ALTERCATION for a in bus.history)
    finally:
        det.stop()


def test_quiet_cabin_is_silent():
    det, bus = _detector()
    try:
        _feed(det, mix(road_noise(5.0, seed=5)))
        time.sleep(0.3)
        assert not bus.history
    finally:
        det.stop()
