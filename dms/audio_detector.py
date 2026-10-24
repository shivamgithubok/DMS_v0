"""Cabin audio events (spec Thread 4): SCREAM, ALTERCATION, ABUSIVE_LANGUAGE.

Capture: sounddevice (PortAudio/ALSA) callback, mono float32, 16 kHz (22.05 kHz when the
Cochl.Sense backend is used, which requires the model rate). A .wav path as ``device``
replays the file in real time for bench tests.

Sound-event backends (``audio.backend``):
  cochl  - Cochl.Sense Edge SDK stream processor (needs the SDK wheel + project key).
  yamnet - YAMNet (AudioSet, 521 classes) as TFLite or ONNX; 0.96 s window.
  dsp    - built-in signal-processing classifier (loudness, pitch, voicing, overlapping
           voices). Always available; less accurate than a learned model.
  auto   - cochl if configured, else yamnet if the model file exists, else dsp.

Foul language: Vosk keyword spotting with a closed grammar (keywords + [unk]) on the
16 kHz stream - small-vocabulary decoding is cheap and robust on the A78 cores.

Latency: classification runs every ``hop_s`` (0.25 s) on the newest window, so an event
is scored within one hop plus inference after enough of it has entered the window.
Each alert carries the measured onset-to-alert latency.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from . import alerts as A
from .config import data_path, model_path, resolve_path
from .state import AudioState

log = logging.getLogger("dms.audio")

SCREAM_LABELS = ("Screaming",)
SHOUT_LABELS = ("Shout", "Yell", "Bellow", "Children shouting", "Battle cry")
SPEECH_LABELS = ("Speech", "Conversation", "Narration, monologue", "Babbling", "Hubbub, speech noise, speech babble")
CROWD_LABELS = ("Conversation", "Hubbub, speech noise, speech babble", "Babbling", "Chatter")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32, copy=False)
    n_out = int(round(len(x) * sr_out / sr_in))
    if sr_out < sr_in:  # crude anti-alias: moving average over the decimation ratio
        k = int(np.ceil(sr_in / sr_out))
        if k > 1:
            x = np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")
    t_in = np.arange(len(x)) / sr_in
    t_out = np.arange(n_out) / sr_out
    return np.interp(t_out, t_in, x).astype(np.float32)


def dbfs(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-6))


class RingBuffer:
    def __init__(self, seconds: float, rate: int) -> None:
        self.rate = rate
        self.data = np.zeros(int(seconds * rate), dtype=np.float32)
        self.total = 0  # samples written since start
        self.lock = threading.Lock()
        self.t_last = 0.0  # monotonic time of the newest sample

    def write(self, chunk: np.ndarray, t_end: float) -> None:
        n = len(chunk)
        size = len(self.data)
        with self.lock:
            if n >= size:
                self.data[:] = chunk[-size:]
            else:
                idx = self.total % size
                first = min(n, size - idx)
                self.data[idx:idx + first] = chunk[:first]
                if first < n:
                    self.data[:n - first] = chunk[first:]
            self.total += n
            self.t_last = t_end

    def latest(self, n: int) -> tuple[np.ndarray, float]:
        size = len(self.data)
        n = min(n, size, self.total)
        with self.lock:
            end = self.total % size
            if n <= end:
                out = self.data[end - n:end].copy()
            else:
                out = np.concatenate([self.data[size - (n - end):], self.data[:end]])
            return out, self.t_last


# --------------------------------------------------------------------------- classifiers
class DspClassifier:
    """Heuristic scores for Screaming / Shout / Speech / Conversation from one window."""

    name = "dsp"
    rate = 16000

    def __init__(self) -> None:
        self.floor = deque(maxlen=240)  # ~60 s of hop levels for a noise-floor estimate

    @staticmethod
    def _pitch_track(x: np.ndarray, sr: int, frame: int = 640, hop: int = 320):
        """Normalized-autocorrelation f0 (60-1600 Hz) per 40 ms frame -> (f0, voicing, level)."""
        f0s, voiced, levels = [], [], []
        lag_min, lag_max = int(sr / 1600), int(sr / 60)
        window = np.hanning(frame).astype(np.float32)
        for start in range(0, len(x) - frame + 1, hop):
            seg = x[start:start + frame] * window
            levels.append(dbfs(x[start:start + frame]))
            seg = seg - seg.mean()
            spec = np.fft.rfft(seg, 2 * frame)
            ac = np.fft.irfft(spec * np.conj(spec))[:frame]
            if ac[0] <= 1e-9:
                f0s.append(0.0)
                voiced.append(0.0)
                continue
            ac = ac / ac[0]
            region = ac[lag_min:lag_max]
            lag = int(np.argmax(region)) + lag_min
            strength = float(ac[lag])
            # prefer the shortest lag with a comparable peak (avoid octave errors downward)
            for div in (4, 3, 2):
                cand = int(round(lag / div))
                if cand >= lag_min and ac[cand] > 0.85 * strength:
                    lag, strength = cand, float(ac[cand])
                    break
            f0s.append(sr / lag)
            voiced.append(strength)
        return np.array(f0s), np.array(voiced), np.array(levels)

    def predict(self, x: np.ndarray) -> dict:
        level = dbfs(x)
        self.floor.append(level)
        floor = float(np.percentile(self.floor, 15)) if len(self.floor) > 8 else -60.0
        snr = level - floor
        f0, voicing, levels = self._pitch_track(x, self.rate)
        active = levels > max(floor + 10.0, -55.0)
        voiced = (voicing > 0.45) & active
        vf = float(voiced.mean()) if len(voiced) else 0.0
        f0v = f0[voiced] if voiced.any() else np.array([0.0])
        median_f0 = float(np.median(f0v))
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1.0 / self.rate)
        total = spec.sum() + 1e-9
        hf_ratio = float(spec[(freqs > 1500) & (freqs < 5000)].sum() / total)
        loud = 1.0 / (1.0 + np.exp(-(level + 26.0) / 3.0))  # ~0.5 at -26 dBFS
        above = 1.0 / (1.0 + np.exp(-(snr - 15.0) / 3.0))
        sustained = min(1.0, vf / 0.5)
        high_pitch = 1.0 / (1.0 + np.exp(-(median_f0 - 520.0) / 60.0))
        mid_pitch = float(np.exp(-((median_f0 - 300.0) / 140.0) ** 2))
        speech_pitch = float(np.exp(-((median_f0 - 170.0) / 90.0) ** 2))
        scream = loud * above * sustained * high_pitch * min(1.0, 0.4 + hf_ratio * 2.0)
        shout = loud * above * sustained * mid_pitch * min(1.0, 0.5 + hf_ratio * 2.0)
        speech = above * min(1.0, vf / 0.3) * max(speech_pitch, mid_pitch * 0.7)
        # overlapping voices: two separated f0 clusters inside the window
        # Two talkers in one window show up as two compact f0 clusters (the frame-wise
        # tracker locks onto whichever voice dominates each 40 ms frame).
        overlap = 0.0
        speechy = f0v[(f0v > 70) & (f0v < 700)]
        if len(speechy) >= 8:
            v = np.sort(np.log2(speechy))
            n = len(v)
            min_count = max(3, int(0.2 * n))
            best = 0.0
            for k in range(min_count, n - min_count + 1):
                lower, upper = v[:k], v[k:]
                sep = float(np.median(upper) - np.median(lower))
                spread = max(np.subtract(*np.percentile(lower, [75, 25])), np.subtract(*np.percentile(upper, [75, 25])))
                if spread < 0.25:
                    best = max(best, sep)
            overlap = float(np.clip((best - 0.3) / 0.4, 0.0, 1.0))  # 0.3 octave (~23 %) apart -> 0
        conversation = speech * max(overlap, 0.0)
        return {"Screaming": float(scream), "Shout": float(shout), "Speech": float(speech),
                "Conversation": float(conversation), "_level": level, "_floor": floor, "_f0": median_f0}


class YamnetClassifier:
    name = "yamnet"
    rate = 16000
    window = 15600

    def __init__(self, model_file: Path, class_map: Path) -> None:
        import csv

        with open(class_map, newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.labels = [row["display_name"] for row in rows]
        self.kind = model_file.suffix.lower()
        if self.kind == ".onnx":
            from .inference import OrtSession

            self.session = OrtSession(model_file, threads=2)
            self.input = self.session.inputs[0]
            self.input_shape = self.session.input_shape()
        else:
            interpreter_cls = None
            for mod, attr in (("ai_edge_litert.interpreter", "Interpreter"), ("tflite_runtime.interpreter", "Interpreter"),
                              ("tensorflow.lite", "Interpreter")):
                try:
                    module = __import__(mod, fromlist=[attr])
                    interpreter_cls = getattr(module, attr)
                    break
                except ImportError:
                    continue
            if interpreter_cls is None:
                raise RuntimeError("no TFLite runtime (pip install ai-edge-litert)")
            self.interp = interpreter_cls(model_path=str(model_file), num_threads=2)
            inp = self.interp.get_input_details()[0]
            # TF-Hub export takes [15600]; the MediaPipe audio-classifier export takes [1, 15600]
            self.inp_shape = [int(d) for d in inp["shape"]]
            if int(np.prod(self.inp_shape)) != self.window:
                self.inp_shape = [self.window]
                self.interp.resize_tensor_input(inp["index"], self.inp_shape, strict=False)
            self.interp.allocate_tensors()
            self.inp_index = inp["index"]
            self.out_index = next(o["index"] for o in self.interp.get_output_details() if o["shape"][-1] == 521)

    def predict(self, x: np.ndarray) -> dict:
        wav = x[-self.window:]
        if len(wav) < self.window:
            wav = np.pad(wav, (self.window - len(wav), 0))
        wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
        if self.kind == ".onnx":
            shape = self.input_shape
            feed = wav.reshape(shape) if len(shape) > 1 and np.prod(shape) == self.window else wav
            out = self.session.run({self.input: feed})
            scores = next(v for v in out.values() if v.shape[-1] == 521)
        else:
            self.interp.set_tensor(self.inp_index, wav.reshape(self.inp_shape))
            self.interp.invoke()
            scores = self.interp.get_tensor(self.out_index)
        scores = np.asarray(scores).reshape(-1, 521).mean(axis=0)
        result = {label: float(scores[i]) for i, label in enumerate(self.labels)}
        result["_level"] = dbfs(x)
        return result


class CochlBackend:
    """Cochl.Sense Edge SDK stream processor (API per docs.cochl.ai Edge SDK samples)."""

    name = "cochl"

    def __init__(self, cfg, on_tags) -> None:
        import sense  # Cochl.Sense Edge SDK wheel

        self.sense = sense
        with open(resolve_path(cfg.config_file), encoding="utf-8") as handle:
            config = json.load(handle)
        outer = self

        class _Listener(sense.ResultListener):
            def on_result(self, frame) -> None:
                if not frame.is_ok():
                    log.warning("cochl error: %s", frame.error)
                    return
                on_tags({tag.name: float(tag.probability) for tag in frame.tags}, frame)

        self._session_ctx = sense.session(cfg.project_key, config)
        self._session_ctx.__enter__()
        self.listener = _Listener()
        self.processor = sense.create_stream_processor(self.listener)
        self.rate = int(self.processor.model_sample_rate())
        self.processor.__enter__()
        self._outer = outer

    def push(self, chunk_f32: np.ndarray) -> None:
        pcm = (np.clip(chunk_f32, -1.0, 1.0) * 32767).astype(np.int16)
        self.processor.push(pcm.tobytes(), 1, self.sense.SampleFormat.INT16, self.rate)

    def close(self) -> None:
        try:
            self.processor.__exit__(None, None, None)
        finally:
            self._session_ctx.__exit__(None, None, None)


# --------------------------------------------------------------------------- keyword spotting
class VoskKws(threading.Thread):
    def __init__(self, cfg, on_keyword) -> None:
        super().__init__(name="audio-kws", daemon=True)
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)
        model_dir = model_path(cfg.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Vosk model not found at {model_dir}")
        words = []
        for line in Path(resolve_path(cfg.keywords_file)).read_text(encoding="utf-8").splitlines():
            line = line.strip().lower()
            if line and not line.startswith("#"):
                words.append(line)
        self.keywords = set(words)
        self.model = Model(str(model_dir))
        self.recognizer = KaldiRecognizer(self.model, 16000, json.dumps(sorted(self.keywords) + ["[unk]"]))
        self.recognizer.SetWords(True)
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self.on_keyword = on_keyword
        self.min_conf = float(cfg.min_confidence)
        self._halt = threading.Event()
        self._reported: set = set()

    def feed(self, chunk_16k: np.ndarray) -> None:
        try:
            self.queue.put_nowait(chunk_16k)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                chunk = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            if self.recognizer.AcceptWaveform(pcm):
                result = json.loads(self.recognizer.Result())
                for word in result.get("result", []):
                    w = word.get("word", "")
                    if w in self.keywords and word.get("conf", 0.0) >= self.min_conf and w not in self._reported:
                        self.on_keyword(w, float(word.get("conf", 0.0)))
                self._reported.clear()
            else:
                partial = json.loads(self.recognizer.PartialResult()).get("partial", "")
                for w in partial.split():
                    if w in self.keywords and w not in self._reported:
                        self._reported.add(w)  # fire early on the partial for latency
                        self.on_keyword(w, self.min_conf)


# --------------------------------------------------------------------------- detector
class AudioDetector:
    def __init__(self, cfg, shared, bus: A.AlertBus) -> None:
        self.cfg = cfg.audio
        self.shared = shared
        self.bus = bus
        self._halt = threading.Event()
        self.backend_name = "off"
        self.classifier = None
        self.cochl = None
        self.kws = None
        self.stream = None
        self.replay_thread = None
        self._cochl_scores: deque = deque(maxlen=8)
        self.hops: deque = deque()  # (t, loud_voice, shout, overlap)
        self.capture_rate = int(self.cfg.sample_rate)
        self.last_tags: dict = {}
        self.last_event = ""
        self.last_event_ts = 0.0
        self.latency_ms = 0.0
        self.error = ""
        self._kws_status = "off"
        self._pick_backend()
        self.ring = RingBuffer(6.0, self.capture_rate)
        self.ring16 = RingBuffer(6.0, 16000) if self.capture_rate != 16000 else self.ring
        self.classifier_thread = threading.Thread(target=self._classify_loop, name="audio-classifier", daemon=True)

    # ------------------------------------------------------------------ setup
    def _pick_backend(self) -> None:
        wanted = self.cfg.backend
        yamnet_file = model_path(self.cfg.yamnet_model)
        if wanted in ("auto", "cochl") and self.cfg.cochl.project_key:
            try:
                self.cochl = CochlBackend(self.cfg.cochl, self._on_cochl)
                self.capture_rate = self.cochl.rate
                self.backend_name = "cochl"
                return
            except Exception as exc:
                log.warning("Cochl.Sense unavailable (%s)", exc)
                if wanted == "cochl":
                    self.error = f"cochl: {exc}"
        if wanted in ("auto", "yamnet", "cochl") and yamnet_file.exists():
            try:
                self.classifier = YamnetClassifier(yamnet_file, model_path(self.cfg.yamnet_class_map))
                self.backend_name = "yamnet"
                return
            except Exception as exc:
                log.warning("YAMNet unavailable (%s)", exc)
        self.classifier = DspClassifier()
        self.backend_name = "dsp"

    def start(self) -> "AudioDetector":
        if self.cfg.kws.enabled:
            try:
                self.kws = VoskKws(self.cfg.kws, self._on_keyword)
                self.kws.start()
                self._kws_status = "vosk"
            except Exception as exc:
                self._kws_status = "off"
                log.warning("keyword spotting disabled: %s", exc)
        self.classifier_thread.start()
        device = self.cfg.device
        if isinstance(device, str) and device.lower().endswith(".wav"):
            self.replay_thread = threading.Thread(target=self._replay, args=(resolve_path(device, "data"),),
                                                  name="audio-capture", daemon=True)
            self.replay_thread.start()
        else:
            self._open_mic(device)
        log.info("audio: backend=%s rate=%d kws=%s", self.backend_name, self.capture_rate, self._kws_status)
        self._publish(-120.0, {})
        return self

    def _open_mic(self, device) -> None:
        try:
            import sounddevice as sd

            block = int(self.capture_rate * self.cfg.block_ms / 1000)
            self.stream = sd.InputStream(samplerate=self.capture_rate, channels=1, dtype="float32", blocksize=block,
                                         device=device, latency="low", callback=self._on_audio)
            self.stream.start()
            info = sd.query_devices(self.stream.device, "input")
            log.info("microphone: %s @ %d Hz", info.get("name", device), self.capture_rate)
        except Exception as exc:
            self.error = f"microphone: {exc}"
            log.error("audio capture failed: %s\n%s", exc, self._device_help())
        threading.Thread(target=self._watch_input, name="audio-watch", daemon=True).start()

    @staticmethod
    def _device_help() -> str:
        try:
            import sounddevice as sd

            rows = [f"    [{i}] {d['name']} ({d['max_input_channels']} ch, {int(d['default_samplerate'])} Hz)"
                    for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
        except Exception as exc:
            return f"  (cannot list audio devices: {exc})"
        if not rows:
            return "  no audio input devices found (plug in a USB microphone; check arecord -l)"
        return ("  input devices (set audio.device to a name or index, or --audio-device):\n" + "\n".join(rows))

    def _watch_input(self) -> None:
        """Warn once if the microphone delivers no samples (wrong device / muted / ALSA busy)."""
        if self._halt.wait(4.0) or self.ring.total > 0 or self.replay_thread is not None:
            return
        self.error = self.error or "no audio from the microphone"
        log.warning("no audio received from the microphone after 4 s\n%s", self._device_help())
        self._publish(-120.0, {})

    def stop(self) -> None:
        self._halt.set()
        if self.stream is not None:
            # abort() drops buffers instead of draining them; stop() can block forever on an
            # input device that never delivers. Bound it either way.
            stream = self.stream

            def _close():
                try:
                    stream.abort()
                    stream.close()
                except Exception:
                    pass
            closer = threading.Thread(target=_close, name="audio-close", daemon=True)
            closer.start()
            closer.join(timeout=2.0)
            if closer.is_alive():
                log.warning("audio device did not close within 2 s; leaving it to process exit")
        if self.kws:
            self.kws.stop()
        if self.cochl:
            try:
                self.cochl.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ capture
    def _on_audio(self, indata, frames, time_info, status) -> None:  # PortAudio callback thread
        self.ingest(indata[:, 0].copy(), time.monotonic())

    def ingest(self, chunk: np.ndarray, t_end: float) -> None:
        self.ring.write(chunk, t_end)
        chunk16 = chunk if self.capture_rate == 16000 else resample(chunk, self.capture_rate, 16000)
        if self.ring16 is not self.ring:
            self.ring16.write(chunk16, t_end)
        if self.kws:
            self.kws.feed(chunk16)
        if self.cochl:
            try:
                self.cochl.push(chunk)
            except Exception as exc:
                self.error = f"cochl push: {exc}"

    def _replay(self, path: Path) -> None:
        with wave.open(str(path), "rb") as wav:
            rate, width, channels = wav.getframerate(), wav.getsampwidth(), wav.getnchannels()
            raw = wav.readframes(wav.getnframes())
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if width == 1:
            data = (data - 128.0) / 128.0
        else:
            data /= float(np.iinfo(dtype).max)
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        data = resample(data, rate, self.capture_rate)
        block = int(self.capture_rate * self.cfg.block_ms / 1000)
        t_next = time.monotonic()
        pos = 0
        self.replay_started = t_next
        while not self._halt.is_set():
            chunk = data[pos:pos + block]
            if len(chunk) < block:
                pos = 0
                chunk = np.concatenate([chunk, data[:block - len(chunk)]])
            pos += block
            t_next += block / self.capture_rate
            delay = t_next - time.monotonic()
            if delay > 0:
                self._halt.wait(delay)
            self.ingest(chunk, time.monotonic())

    # ------------------------------------------------------------------ classification
    def _classify_loop(self) -> None:
        hop = float(self.cfg.hop_s)
        window = int(float(self.cfg.window_s) * 16000)
        next_t = time.monotonic() + hop
        while not self._halt.is_set():
            delay = next_t - time.monotonic()
            if delay > 0 and self._halt.wait(delay):
                break
            next_t += hop
            if next_t < time.monotonic():
                next_t = time.monotonic() + hop  # fell behind; skip rather than burst
            if self.ring16.total < 16000 * 0.5:
                continue
            x, t_last = self.ring16.latest(window)
            level = dbfs(x[-1600:])
            if self.cochl:
                scores = self._merged_cochl_scores()
                scores["_level"] = dbfs(x)
            else:
                try:
                    scores = self.classifier.predict(x)
                except Exception as exc:
                    self.error = f"classifier: {exc}"
                    log.exception("audio classifier failed")
                    continue
            self._decide(scores, x, t_last)
            self._publish(level, scores)

    def _on_cochl(self, tags: dict, frame) -> None:
        self._cochl_scores.append((time.monotonic(), tags))

    def _merged_cochl_scores(self) -> dict:
        now = time.monotonic()
        cfg = self.cfg.cochl
        merged: dict = {}
        for ts, tags in list(self._cochl_scores):
            if now - ts > 1.2:
                continue
            for name, prob in tags.items():
                merged[name] = max(merged.get(name, 0.0), prob)
        out = dict(merged)
        out["Screaming"] = max([merged.get(t, 0.0) for t in cfg.scream_tags] + [0.0])
        out["Speech"] = max([merged.get(t, 0.0) for t in cfg.speech_tags] + [0.0])
        out["Shout"] = max([merged.get(t, 0.0) for t in ("Shout", "Yell")] + [0.0])
        return out

    def _onset_latency_ms(self, x: np.ndarray, t_last: float, now: float) -> float:
        """Estimate onset as the first 20 ms frame above (floor + 12 dB) in the latest burst."""
        frame = 320
        levels = np.array([dbfs(x[i:i + frame]) for i in range(0, len(x) - frame + 1, frame)])
        if not len(levels):
            return 0.0
        floor = np.percentile(levels, 10)
        hot = levels > max(floor + 12.0, -50.0)
        if not hot.any():
            return 0.0
        # walk back over the contiguous loud run that ends nearest "now"
        last_hot = np.nonzero(hot)[0][-1]
        start = last_hot
        while start > 0 and hot[start - 1]:
            start -= 1
        onset = t_last - (len(x) - start * frame) / 16000.0
        return max(0.0, (now - onset) * 1000.0)

    def _decide(self, scores: dict, x: np.ndarray, t_last: float) -> None:
        cfg = self.cfg
        now = time.monotonic()
        scream = max(scores.get(k, 0.0) for k in SCREAM_LABELS)
        shout = max([scores.get(k, 0.0) for k in SHOUT_LABELS] + [0.0])
        speech = max([scores.get(k, 0.0) for k in SPEECH_LABELS] + [0.0])
        crowd = max([scores.get(k, 0.0) for k in CROWD_LABELS] + [0.0])
        level = scores.get("_level", dbfs(x))
        if scream >= cfg.scream_threshold:
            self._alert(A.SCREAM, scream, "Screaming detected in cabin", x, t_last, now)

        loud_voice = (speech >= 0.3 or shout >= cfg.shout_threshold) and level >= cfg.altercation_loud_dbfs
        self.hops.append((now, loud_voice, shout >= cfg.shout_threshold, crowd >= 0.25))
        while self.hops and now - self.hops[0][0] > cfg.altercation_window_s:
            self.hops.popleft()
        hop = float(cfg.hop_s)
        loud_s = sum(1 for h in self.hops if h[1]) * hop
        shouts = sum(1 for h in self.hops if h[2])
        overlapping = sum(1 for h in self.hops if h[3])
        if loud_s >= cfg.altercation_min_s and (shouts >= 2 or overlapping >= 2):
            conf = min(1.0, 0.5 + 0.1 * shouts + 0.1 * overlapping)
            self._alert(A.ALTERCATION, conf, "Loud overlapping voices / shouting in cabin", x, t_last, now)

    def _on_keyword(self, word: str, conf: float) -> None:
        x, t_last = self.ring16.latest(16000)
        masked = word[0] + "*" * max(0, len(word) - 1)
        self._alert(A.ABUSIVE_LANGUAGE, conf, f"Abusive language detected ({masked})", x, t_last, time.monotonic(),
                    extra={"keyword": masked})

    def _alert(self, kind: str, conf: float, text: str, x: np.ndarray, t_last: float, now: float, extra=None) -> None:
        latency = self._onset_latency_ms(x, t_last, now)
        alert = A.Alert(type=kind, category=A.AUDIO, confidence=float(conf), description=text, reason=kind.lower(),
                        extra={"detection_latency_ms": round(latency), "backend": self.backend_name, **(extra or {})})
        last = self.bus._last.get((alert.type, alert.reason))
        if last is not None and now - last < self.cfg.alert_cooldown_s:
            return
        if self.cfg.save_snippets:
            alert.audio_path = self._save_snippet(kind)
        if self.bus.emit(alert, cooldown_s=self.cfg.alert_cooldown_s):
            self.latency_ms = latency
            self.last_event, self.last_event_ts = kind, time.time()

    def _save_snippet(self, kind: str) -> str:
        audio, _ = self.ring.latest(int(self.capture_rate * 1.0))
        path = data_path("audio", f"{time.strftime('%Y%m%d_%H%M%S')}_{kind.lower()}.wav")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.capture_rate)
            wav.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        return str(path)

    def _publish(self, level: float, scores: dict) -> None:
        tags = {k: round(v, 2) for k, v in sorted(((k, v) for k, v in scores.items() if not k.startswith("_")),
                                                   key=lambda kv: -kv[1])[:3]}
        self.shared.audio = AudioState(enabled=True, backend=self.backend_name, level_dbfs=level, tags=tags,
                                       last_event=self.last_event, last_event_ts=self.last_event_ts,
                                       kws=self._kws_status, latency_ms=self.latency_ms, error=self.error)
