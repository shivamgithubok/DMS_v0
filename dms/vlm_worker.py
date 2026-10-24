"""Low-frequency cabin scene understanding with a small VLM (spec Thread 3).

The VLM is served by a separate process on the same Jetson - llama.cpp ``llama-server``
or vLLM, both exposing the OpenAI chat-completions API - and this thread is only an
HTTP client. That split is deliberate:

* the VLM has its own CUDA context, so the detector's high-priority TensorRT stream is
  never queued behind a 1.5-3B-parameter decode in this process;
* the decode never holds this process's GIL, so the real-time loop and UI stay smooth;
* the server can be restarted or swapped (Qwen2.5-VL-3B, VILA1.5-3B, ...) without
  touching the DMS.

Scheduling: one request every ``interval_s`` (default 2.5 s = 0.4 Hz). The interval is
multiplied by ``thermal_backoff`` while any SoC zone is above ``thermal_limit_c`` (85 C),
doubled while the real-time loop is below ``min_realtime_fps``, and requests are skipped
entirely when unified memory is above ``memory_limit_mb``.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import alerts as A
from .config import data_path
from .state import VlmState

log = logging.getLogger("dms.vlm")

CATEGORY_TERMS = {
    "driver_distracted": [
        r"(?:holding|using|looking at|on) (?:a |the |his |her |their )?(?:mobile |cell )?phone", r"\btexting\b",
        r"\bphone\b", r"looking away", r"not looking at the road", r"eyes (?:are )?closed", r"\basleep\b",
        r"\bdrows", r"\bdistract", r"\beating\b", r"hands? (?:are )?off the (?:steering )?wheel",
    ],
    "passenger_aggressive": [
        r"\baggress", r"\bfight", r"\bhitting\b", r"\bpunch", r"\bgrabb", r"\bshov", r"\bthreaten",
        r"leaning (?:into|over|towards?) (?:the )?(?:front|driver)", r"reaching (?:for|towards?) the (?:driver|wheel)",
        r"\baltercation\b", r"\bviolen",
    ],
    "child_unattended": [
        r"(?:child|kid|baby|infant|toddler)[^.]{0,40}(?:alone|unattended|unsupervised|without (?:an )?adult)",
        r"(?:alone|unattended)[^.]{0,30}(?:child|kid|baby|infant|toddler)",
        r"(?:child|baby|infant)[^.]{0,30}(?:not in|without) (?:a )?(?:car )?seat",
    ],
    "objects_blocking_driver": [
        r"\bblock(?:ing|s|ed)?\b", r"\bobstruct", r"covering the (?:windshield|view|mirror)",
    ],
    "other": [r"\bweapon\b", r"\bknife\b", r"\bgun\b", r"\bsmok", r"\bfire\b", r"no seat ?belt", r"unbuckled"],
}
_NEGATORS = re.compile(r"(?:\bno\b|\bnot\b|\bnor\b|\bwithout\b|n't\b|\bnone\b|\bneither\b|\bno signs? of\b|"
                       r"\bfree of\b|\babsent\b)[^.;,]*$")
_SAFE_PHRASES = re.compile(r"^\W*(?:safe\b|no unsafe|nothing unsafe|there is no unsafe|no (?:visible |obvious )?"
                           r"(?:unsafe|dangerous) behaviou?r|the cabin (?:appears|looks|is) safe|all clear)")


@dataclass
class VlmVerdict:
    unsafe: bool
    categories: list = field(default_factory=list)
    summary: str = ""
    confidence: float | None = None  # None means a prose answer, so the confidence gate is skipped
    scene: str = ""


@dataclass
class PolicyDecision:
    """What the safety policy kept from one scene answer."""

    kept: list = field(default_factory=list)
    suppressed: list = field(default_factory=list)  # "hazard: reason"
    scene: str = ""
    line: str = ""


# The real-time detector owns driver state. The VLM may only add cabin hazards.
DRIVER_VLM_CATEGORIES = {"driver_distracted", "other"}
CABIN_HAZARDS = ("passenger_aggressive", "child_unattended", "objects_blocking_driver")
_HAZARD_NAMES = set(CABIN_HAZARDS) | DRIVER_VLM_CATEGORIES
CMS_VLM_CATEGORIES = {
    "passenger_aggressive": (A.PASSENGER_AGGRESSIVE, "passenger_aggressive"),
    "child_unattended": (A.CHILD_PRESENT, "vlm_child"),
    "objects_blocking_driver": (A.OBJECT_ON_SEAT, "vlm_blocking"),
}
_HAZARD_LABEL = {
    "passenger_aggressive": "passenger aggressive",
    "child_unattended": "child unattended",
    "objects_blocking_driver": "object blocking driver",
    "driver_distracted": "driver distracted",
    "other": "other",
}


def verdict_alerts(verdict: VlmVerdict) -> list[tuple[str, str, str]]:
    """(alert type, reason, channel) for one parsed answer. Empty when the scene is safe."""
    if not verdict.unsafe:
        return []
    routed = []
    driver = [name for name in verdict.categories if name in DRIVER_VLM_CATEGORIES]
    if driver:
        routed.append((A.VLM_ALERT, ",".join(driver), "dms"))
    for name in verdict.categories:
        spec = CMS_VLM_CATEGORIES.get(name)
        if spec is not None:
            routed.append((spec[0], spec[1], "cms"))
    return routed


def _label(name: str) -> str:
    return _HAZARD_LABEL.get(name, name.replace("_", " "))


def state_card(driver, cabin, audio) -> str:
    """One line of measurements the scene prompt must trust."""
    bits = []
    if driver is not None:
        yaw = "unknown" if getattr(driver, "yaw", None) is None else f"{float(driver.yaw):.0f}"
        bits.append(
            "driver eyes={eye} yaw_deg={yaw} phone={phone} belt={belt} distracted={dist} drowsy={drowsy}".format(
                eye=getattr(driver, "eye_state", None) or "--",
                yaw=yaw,
                phone="yes" if getattr(driver, "phone", False) else "no",
                belt=getattr(driver, "seatbelt", None) or "Unknown",
                dist="yes" if getattr(driver, "distracted", False) else "no",
                drowsy="yes" if getattr(driver, "drowsy", False) else "no",
            )
        )
    known = cabin is not None and bool(
        getattr(cabin, "connected", False) or getattr(cabin, "seats", None) or getattr(cabin, "faces", None)
    )
    if not known:
        bits.append("cabin=unavailable")
    else:
        seats = ",".join(f"{s.name}={s.occupant}" for s in cabin.seats) or "none"
        objects = ",".join(f"{o.label}@{o.seat or 'cabin'}" for o in list(cabin.objects)[:4]) or "none"
        bits.append(
            f"cabin occupants={cabin.occupants} adults={cabin.adults} children={cabin.children} "
            f"child_present={'yes' if cabin.child_present else 'no'} seats={seats} objects={objects}"
        )
    if audio is not None and getattr(audio, "enabled", False):
        bits.append(f"audio={getattr(audio, 'last_event', '') or 'none'}")
    return "Measurements: " + " | ".join(bits) + "."


def scene_prompt(base: str, hint: str, driver, cabin, audio) -> str:
    parts = [p.strip() for p in (base, hint, state_card(driver, cabin, audio)) if p and str(p).strip()]
    return " ".join(parts)


def _json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or ("scene" not in data and "hazards" not in data):
        return None
    return data


def _hazard_name(value) -> str:
    name = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return name if name in _HAZARD_NAMES else ""


def _cabin_has(cabin, kind: str) -> bool:
    flag = "child_present" if kind == "child" else "adult_present"
    if getattr(cabin, flag, False):
        return True
    return any(getattr(seat, "occupant", "") == kind for seat in getattr(cabin, "seats", []) or [])


def _cabin_known(cabin) -> bool:
    return cabin is not None and bool(
        getattr(cabin, "connected", False) or getattr(cabin, "seats", None) or getattr(cabin, "faces", None)
    )


def apply_policy(verdict: VlmVerdict, driver=None, cabin=None, confidence_min: float = 0.55) -> PolicyDecision:
    """Keep cabin hazards the measurements do not contradict. Driver state stays with the detector.

    ``driver`` is accepted so a future rule can read it; driver hazards are dropped regardless.
    """
    del driver  # detector owns driver alerts; the argument documents the call site
    kept = []
    suppressed = []
    floor = float(confidence_min)
    for name in verdict.categories:
        if name in DRIVER_VLM_CATEGORIES:
            suppressed.append(f"{_label(name)}: detector owns driver state")
            continue
        if name not in CMS_VLM_CATEGORIES:
            continue
        if verdict.confidence is not None and verdict.confidence < floor:
            suppressed.append(f"{_label(name)}: confidence {verdict.confidence:.2f} below {floor:.2f}")
            continue
        if name == "child_unattended" and _cabin_known(cabin) and _cabin_has(cabin, "adult") and not _cabin_has(cabin, "child"):
            suppressed.append(f"{_label(name)}: adult in cabin, no child measured")
            continue
        kept.append(name)
    scene = (verdict.scene or verdict.summary or "").strip()
    if kept and suppressed:
        line = "Policy: kept " + ", ".join(_label(n) for n in kept) + " · suppressed " + "; ".join(suppressed)
    elif kept:
        line = "Policy: kept " + ", ".join(_label(n) for n in kept)
    elif suppressed:
        line = "Policy: suppressed " + "; ".join(suppressed)
    else:
        line = "Policy: no extra cabin hazard"
    return PolicyDecision(kept, suppressed, scene[:160], line)


def assess(text: str, driver=None, cabin=None, confidence_min: float = 0.55):
    """Parse one answer, apply the safety policy, and list the alerts to emit."""
    verdict = parse_vlm(text)
    decision = apply_policy(verdict, driver, cabin, confidence_min)
    routed = verdict_alerts(VlmVerdict(bool(decision.kept), list(decision.kept), decision.scene, verdict.confidence, decision.scene))
    return verdict, decision, routed


def parse_vlm(text: str) -> VlmVerdict:
    """Turn a VLM answer into a scene, hazards, and an optional confidence.

    JSON is preferred. Older SAFE/UNSAFE prose still parses, with confidence left unset.
    """
    raw = (text or "").strip()
    raw = re.sub(r"<think>.*?</think>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    data = _json_object(raw)
    if data is not None:
        scene = re.sub(r"\s+", " ", str(data.get("scene") or "")).strip()[:160]
        found = []
        hazards = data.get("hazards") or []
        if isinstance(hazards, str):
            hazards = [hazards]
        if isinstance(hazards, list):
            for item in hazards:
                name = _hazard_name(item)
                if name and name not in found:
                    found.append(name)
        confidence = 0.0
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        summary = scene or re.sub(r"\s+", " ", raw)[:160]
        return VlmVerdict(bool(found), found, summary, confidence, scene or summary)
    clean = re.sub(r"\s+", " ", raw)
    lower = clean.lower()
    explicit_unsafe = bool(re.match(r"^\W*unsafe\b", lower))
    explicit_safe = not explicit_unsafe and bool(_SAFE_PHRASES.match(lower))
    summary = re.sub(r"^\W*(?:un)?safe\W*", "", clean, flags=re.I).strip() or clean
    summary = summary[:160]
    if explicit_safe:
        summary = summary or "SAFE"
        return VlmVerdict(False, [], summary, None, summary)
    found = []
    for category, patterns in CATEGORY_TERMS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, lower):
                prefix = lower[max(0, match.start() - 40):match.start()]
                sentence_prefix = re.split(r"[.;!?]", prefix)[-1]
                if _NEGATORS.search(sentence_prefix):
                    continue
                found.append(category)
                break
            if category in found:
                break
    if explicit_unsafe and not found:
        found = ["other"]
    return VlmVerdict(bool(found), found, summary, None, summary)


def compose(driver: np.ndarray | None, passenger: np.ndarray | None, width: int) -> np.ndarray | None:
    """Side-by-side composite (driver | passenger), each tile width/2 wide when both exist."""
    frames = [f for f in (driver, passenger) if f is not None]
    if not frames:
        return None
    tile_w = width // len(frames)
    tiles = []
    for f in frames:
        h, w = f.shape[:2]
        tiles.append(cv2.resize(f, (tile_w, int(round(h * tile_w / w))), interpolation=cv2.INTER_AREA))
    height = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    return np.hstack(tiles)


class VlmClient:
    def __init__(self, endpoint: str, model: str = "vlm", api_key: str = "", timeout: float = 3.5) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.server_model = ""

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def probe(self) -> bool:
        """GET /v1/models (llama.cpp and vLLM both serve it). Picks up the served model id."""
        url = self.endpoint.split("/v1/")[0] + "/v1/models"
        try:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                body = json.loads(resp.read().decode())
            models = [m.get("id") for m in body.get("data", []) if m.get("id")]
            if models:
                self.server_model = models[0]
                if self.model in ("", "vlm", "auto"):
                    self.model = models[0]
            return True
        except Exception as exc:
            log.debug("VLM probe failed: %s", exc)
            return False

    def ask(self, jpeg: bytes, prompt: str, max_tokens: int = 80, temperature: float = 0.0) -> str:
        image_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            # Qwen3.5 spends the token budget on hidden reasoning unless this is set.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(), headers=self._headers(),
                                     method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"] or ""


class VlmWorker(threading.Thread):
    def __init__(self, cfg, capture, shared, bus: A.AlertBus) -> None:
        super().__init__(name="vlm-worker", daemon=True)
        self.cfg = cfg.vlm
        self.capture = capture
        self.shared = shared
        self.bus = bus
        self.client = VlmClient(self.cfg.endpoint, self.cfg.model, self.cfg.api_key, self.cfg.timeout_s)
        self.base_prompt = self.cfg.prompt.strip()
        self.format_hint = (self.cfg.format_hint or "").strip()
        self.confidence_min = float(self.cfg.get("confidence_min", 0.55))
        self._halt = threading.Event()
        self.calls = 0
        self.errors = 0
        self.latencies: list[float] = []

    def stop(self) -> None:
        self._halt.set()

    def _latency_avg(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def _publish(self, **changes) -> None:
        current = self.shared.vlm
        values = {k: getattr(current, k) for k in current.__dataclass_fields__}
        values.update(changes)
        values.update(enabled=True, calls=self.calls, errors=self.errors, model=self.client.server_model or self.client.model)
        self.shared.vlm = VlmState(**values)

    def _interval(self) -> tuple[float, str]:
        interval = float(self.cfg.interval_s)
        reason = ""
        system = self.shared.system
        if system is not None:
            hottest = system.max_temp_c
            if hottest is not None and hottest > self.cfg.thermal_limit_c:
                interval *= float(self.cfg.thermal_backoff)
                reason = f"thermal {hottest:.0f}C"
        fps = self.shared.driver.fps
        if 0 < fps < self.cfg.min_realtime_fps:
            interval *= 2.0
            reason = reason or f"realtime {fps:.0f} fps"
        return interval, reason

    def run(self) -> None:
        self._publish(status="waiting", interval_s=self.cfg.interval_s)
        warned = False
        while not self._halt.is_set() and not self.client.probe():
            self._publish(status="waiting", text=f"VLM server not reachable at {self.cfg.endpoint}")
            if not warned:
                warned = True
                log.warning("VLM server not reachable at %s; scene understanding is paused until it is up. "
                            "Start it with tools/vlm_server.sh (model: python3 tools/download_models.py --vlm), "
                            "or run with --no-vlm.", self.cfg.endpoint)
            self._halt.wait(5.0)
        if self._halt.is_set():
            return
        log.info("VLM ready: %s at %s", self.client.model, self.cfg.endpoint)
        self._publish(status="ok", text="")
        next_at = time.monotonic()
        while not self._halt.is_set():
            interval, reason = self._interval()
            wait = next_at - time.monotonic()
            if wait > 0 and self._halt.wait(wait):
                break
            next_at = time.monotonic() + interval
            system = self.shared.system
            if system is not None and system.mem_used_mb > self.cfg.memory_limit_mb:
                self._publish(status="throttled", text=f"paused: memory {system.mem_used_mb:.0f} MB",
                              interval_s=interval)
                log.warning("VLM skipped: unified memory %.0f MB > %d MB", system.mem_used_mb, self.cfg.memory_limit_mb)
                continue
            if reason:
                log.info("VLM interval %.1fs (%s)", interval, reason)
            self._cycle(interval, reason)

    def _cycle(self, interval: float, reason: str) -> None:
        drv = self.capture.driver.latest()
        pas = self.capture.passenger.latest() if self.capture.passenger else None
        if drv is None and pas is None:
            return
        if self.cfg.composite:
            image = compose(drv.image if drv else None, pas.image if pas else None, int(self.cfg.image_width) * 2)
        else:
            image = compose(drv.image if drv else None, None, int(self.cfg.image_width))
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.jpeg_quality)])
        if not ok:
            return
        self._publish(status="running", interval_s=interval)
        started = time.monotonic()
        try:
            prompt = scene_prompt(self.base_prompt, self.format_hint, self.shared.driver, self.shared.passenger,
                                   self.shared.audio)
            text = self.client.ask(buf.tobytes(), prompt, int(self.cfg.max_tokens), float(self.cfg.temperature))
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
            self.errors += 1
            latency = time.monotonic() - started
            log.warning("VLM request failed after %.2fs: %s", latency, exc)
            self._publish(status="error", text=f"request failed: {exc}", latency_s=latency,
                          latency_avg_s=self._latency_avg())
            if not self.client.probe():
                self._publish(status="waiting", text="VLM server not reachable")
            return
        latency = time.monotonic() - started
        self.calls += 1
        self.latencies = (self.latencies + [latency])[-50:]
        _parsed, decision, routed = assess(
            text, self.shared.driver, self.shared.passenger, self.confidence_min)
        confidence = 0.0 if _parsed.confidence is None else float(_parsed.confidence)
        self._publish(status="throttled" if reason else "ok", text=decision.scene, scene=decision.scene,
                      policy=decision.line, unsafe=bool(decision.kept), categories=list(decision.kept),
                      suppressed=list(decision.suppressed), confidence=confidence, latency_s=latency,
                      latency_avg_s=self._latency_avg(), last_ts=time.time(), interval_s=interval)
        log.info("VLM %.2fs %s | %s", latency, decision.line, decision.scene)
        if routed:
            path = data_path("snapshots", f"{time.strftime('%Y%m%d_%H%M%S')}_vlm.jpg")
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            for kind, alert_reason, channel in routed:
                self.bus.emit(A.Alert(
                    type=kind, category=A.VISION, channel=channel,
                    confidence=confidence or 0.7, reason=alert_reason,
                    description=decision.scene, snapshot_path=str(path),
                    extra={"categories": list(decision.kept), "suppressed": list(decision.suppressed),
                           "policy": decision.line, "raw": text[:400], "latency_s": round(latency, 2),
                           "model": self.client.model}),
                    cooldown_s=max(10.0, 3 * interval))
