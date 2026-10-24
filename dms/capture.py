"""Camera capture for the driver and passenger channels (Thread 1).

Sources
-------
``auto:<n>``         CSI sensor n if the kernel has it registered, else the n-th USB camera
                      (also used when a registered CSI sensor delivers no frames). Default.
``usb:<n>``          n-th USB (UVC) camera, resolved by name in sysfs.
``csi:<sensor-id>``   MIPI CSI sensor through nvarguscamerasrc (Jetson ISP, NVMM zero-copy
                      scale/convert in nvvidconv). Uses OpenCV's GStreamer backend when the
                      installed cv2 has it, otherwise PyGObject + appsink directly. The pip
                      OpenCV wheels are built without GStreamer, so the PyGObject path is what
                      runs in the .venv set up by setup_jetson.sh.
``v4l2:<n|/dev/videoN>`` USB/UVC camera.
``gst:<pipeline>``    Any GStreamer pipeline ending in ``appsink`` (it is renamed to ``appsink``).
``rtsp://...``        Network stream.
``<file.mp4>``        Video file, looped and paced at its native rate.
``<file.jpg|png>``    Still image published at the configured fps (bench testing).

Each camera has its own grabber thread so a stalled sensor never delays the other one.
Consumers always get the *latest* frame (size-1 drop-oldest queue): the real-time
detector must never work through a backlog.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT

log = logging.getLogger("dms.capture")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm", ".h264", ".h265"}


def _media_path(src: str) -> Path:
    """Resolve a file source from the working directory, then from the repo root."""
    path = Path(src)
    if path.is_absolute() or path.exists():
        return path
    rooted = ROOT / path
    return rooted if rooted.exists() else path


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float  # time.monotonic() at capture
    wall_time: float  # time.time() at capture
    seq: int
    source: str


@dataclass
class CameraStats:
    fps: float = 0.0
    frames: int = 0
    dropped: int = 0
    reconnects: int = 0
    connected: bool = False
    backend: str = ""
    error: str = ""
    _times: list = field(default_factory=list)


def _opencv_has_gstreamer() -> bool:
    info = cv2.getBuildInformation()
    for line in info.splitlines():
        if "GStreamer" in line:
            return "YES" in line
    return False


def csi_pipeline(sensor_id: int, width: int, height: int, fps: int, flip: int = 0,
                 sensor_width: int | None = 1280, sensor_height: int | None = 720, bgr: bool = True) -> str:
    """nvarguscamerasrc -> NVMM -> nvvidconv (HW scale/flip) -> CPU BGR(x) appsink.

    The appsink must be *named* "appsink": OpenCV locates the sink of a manual pipeline by
    element name ("appsink"/"opencvsink"), not by type. With ``sensor_width=None`` the
    sensor mode is left to Argus (works for any sensor, at the cost of a larger capture).
    """
    mode = (f"width={sensor_width},height={sensor_height},"
            if sensor_width and sensor_height else "")
    tail = "videoconvert ! video/x-raw,format=BGR ! " if bgr else ""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),{mode}framerate={fps}/1,format=NV12 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw,width={width},height={height},format=BGRx ! "
        f"{tail}appsink name=appsink drop=true max-buffers=1 sync=false"
    )


def _name_appsink(pipeline: str) -> str:
    """Make sure the final appsink is called "appsink" (OpenCV finds it by name)."""
    head, sep, tail = pipeline.rpartition("appsink")
    if not sep:
        return pipeline
    elem, bang, rest = tail.partition("!")
    elem = re.sub(r"\bname=\S+", "", elem)
    return f"{head}appsink name=appsink{elem}{bang}{rest}"


class CameraUnavailable(RuntimeError):
    """The requested sensor does not exist on this system (retrying soon will not help)."""


SYSFS_V4L = Path("/sys/class/video4linux")

CSI_HELP = (
    "No CSI camera is registered with the kernel/Argus. Check:\n"
    "  1. ribbon cable seated (contacts facing the right way) on CAM0/CAM1, board powered off when connecting\n"
    "  2. camera overlay enabled: sudo /opt/nvidia/jetson-io/jetson-io.py -> Configure Jetson 24pin CSI "
    "Connector -> select your sensor (e.g. IMX219 Dual) -> save and reboot\n"
    "  3. after reboot: ls /dev/video*  and  v4l2-ctl --list-devices  must show 'vi-output' nodes\n"
    "  4. sudo dmesg | grep -iE 'imx|nvcsi|vi5'   and   sudo systemctl restart nvargus-daemon\n"
    "  5. test outside the DMS: gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=30 ! fakesink\n"
    "USB cameras are used automatically when present (cameras.usb_fallback); or pass --driver v4l2:0."
)


def v4l2_nodes(root: Path = SYSFS_V4L) -> list[dict]:
    """Video nodes from sysfs: [{"index": n, "name": str, "csi": bool, "capture": bool}]."""
    out = []
    if not root.exists():
        return out
    for node in sorted(root.glob("video*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        try:
            name = (node / "name").read_text().strip()
        except OSError:
            continue
        try:
            sub_index = int((node / "index").read_text().strip())
        except (OSError, ValueError):
            sub_index = 0
        csi = name.startswith("vi-output")
        out.append({"index": int(re.sub(r"\D", "", node.name) or 0), "name": name, "csi": csi,
                    # UVC creates two nodes per camera; sysfs index 0 is the image stream
                    "capture": csi or sub_index == 0})
    return out


def csi_sensors(root: Path = SYSFS_V4L) -> list[dict] | None:
    """CSI sensors bound to the Jetson VI driver, or None when sysfs is not available."""
    if not root.exists():
        return None
    return [n for n in v4l2_nodes(root) if n["csi"]]


def usb_cameras(root: Path = SYSFS_V4L) -> list[dict]:
    return [n for n in v4l2_nodes(root) if not n["csi"] and n["capture"]]


def resolve_usb(n: int) -> int:
    """/dev/video index of the n-th USB camera (sysfs order); /dev/video{2n} if sysfs is unavailable."""
    cams = usb_cameras()
    if cams:
        if n >= len(cams):
            raise CameraUnavailable(f"USB camera {n} not found ({len(cams)} connected: "
                                    + ", ".join(f"/dev/video{c['index']} {c['name']}" for c in cams) + ")")
        return cams[n]["index"]
    if SYSFS_V4L.exists():
        raise CameraUnavailable("no USB camera connected (check lsusb and v4l2-ctl --list-devices)")
    return 2 * n


def _gi_available() -> bool:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # noqa: F401
        return True
    except Exception:
        return False


class _GiAppSink:
    """Minimal GStreamer appsink reader that does not depend on OpenCV's build."""

    def __init__(self, pipeline: str) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        if not Gst.is_initialized():
            Gst.init(None)
        self.Gst = Gst
        self.pipeline = Gst.parse_launch(pipeline)
        self.sink = self.pipeline.get_by_name("appsink") or self.pipeline.get_by_name("sink")
        if self.sink is None:
            raise RuntimeError("GStreamer pipeline needs an 'appsink name=appsink'")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"GStreamer pipeline failed to start: {self._bus_error()}")
        ret, _, _ = self.pipeline.get_state(5 * Gst.SECOND)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"GStreamer pipeline failed to start: {self._bus_error()}")

    def _bus_error(self) -> str:
        bus = self.pipeline.get_bus()
        msg = bus.timed_pop_filtered(0, self.Gst.MessageType.ERROR) if bus else None
        if msg is None:
            return "no error message (is nvargus-daemon running and the sensor connected?)"
        err, debug = msg.parse_error()
        return f"{err.message} ({debug.splitlines()[0] if debug else ''})"

    def read(self, timeout_s: float = 1.0):
        sample = self.sink.emit("try-pull-sample", int(timeout_s * 1e9))
        if sample is None:
            return False, None
        caps = sample.get_caps().get_structure(0)
        width = caps.get_value("width")
        height = caps.get_value("height")
        fmt = caps.get_value("format")
        buf = sample.get_buffer()
        ok, info = buf.map(self.Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            channels = 4 if fmt in ("BGRx", "BGRA", "RGBA", "RGBx") else 3
            stride = info.size // height
            arr = np.ndarray((height, stride), dtype=np.uint8, buffer=info.data)
            arr = arr[:, : width * channels].reshape(height, width, channels)
            if channels == 4:
                image = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            else:
                image = arr.copy()
        finally:
            buf.unmap(info)
        return True, image

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return True

    def release(self) -> None:
        self.pipeline.set_state(self.Gst.State.NULL)


class Camera:
    """One source with its own grabber thread and a latest-frame slot."""

    def __init__(self, name: str, source: str, width: int = 640, height: int = 480,
                 fps: int = 30, flip: int = 0, sensor_size=None, usb_fallback: bool = True,
                 usb_index: int | None = None) -> None:
        self.name = name
        self.usb_fallback = usb_fallback
        self.usb_index = usb_index  # which USB camera to take over if CSI is absent
        self.requested = None  # original source when a fallback replaced it
        self._explained = False
        self.sensor_size = tuple(sensor_size) if sensor_size else None
        self.source = str(source)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.flip = int(flip)
        self.stats = CameraStats()
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._seq = 0
        self._consumed = 0
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cam-{name}", daemon=True)
        self._cap = None
        self._still: np.ndarray | None = None
        self._pace = 0.0  # seconds per frame for file/still sources

    # ------------------------------------------------------------------ open
    def _open(self):
        src = self.source
        lower = src.lower()
        self._pace = 0.0
        if lower.startswith(("csi:", "auto:")):
            return self._open_csi(int(src.split(":", 1)[1] or 0))
        if lower.startswith("usb:"):
            return self._open_v4l2(resolve_usb(int(src.split(":", 1)[1] or 0)))
        if lower.startswith("gst:"):
            pipeline = src.split(":", 1)[1]
            pipeline = _name_appsink(pipeline)
            if _opencv_has_gstreamer():
                self.stats.backend = "gstreamer/opencv"
                return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            self.stats.backend = "gstreamer/gi"
            return _GiAppSink(pipeline)
        if lower.startswith("v4l2:"):
            dev = src.split(":", 1)[1]
            return self._open_v4l2(int(dev) if dev.isdigit() else dev)
        if lower.startswith(("rtsp://", "http://", "https://")):
            self.stats.backend = "network"
            return cv2.VideoCapture(src)
        path = _media_path(src)
        src = str(path)
        if path.suffix.lower() in IMAGE_EXT:
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"cannot read image {path}")
            self._still = self._fit(image)
            self._pace = 1.0 / self.fps
            self.stats.backend = "still"
            return "still"
        if path.suffix.lower() in VIDEO_EXT or path.exists():
            cap = cv2.VideoCapture(str(path))
            native = cap.get(cv2.CAP_PROP_FPS) or self.fps
            self._pace = 1.0 / max(1.0, min(native, 60.0))
            self.stats.backend = "file"
            return cap
        raise RuntimeError(f"unrecognised camera source {src!r}")

    def _open_csi(self, sensor: int):
        """Try sensor modes x backends until one delivers a frame.

        Modes: the configured one, then 1280x720 and 1920x1080 (IMX219/IMX477 both have
        them), then no explicit mode (Argus picks). Backends: OpenCV's GStreamer build
        (JetPack's system OpenCV) and PyGObject appsink (pip OpenCV has no GStreamer).
        """
        sensors = csi_sensors()
        if sensors is not None and sensor >= len(sensors):
            found = ", ".join(n["name"] for n in sensors) or "none"
            raise CameraUnavailable(f"CSI sensor {sensor} not registered (CSI sensors found: {found})")
        modes = []
        if self.sensor_size:
            modes.append(tuple(self.sensor_size))
        modes += [(1280, 720), (1920, 1080), (None, None)]
        modes = list(dict.fromkeys(modes))
        backends = []
        if _opencv_has_gstreamer():
            backends.append("opencv")
        if _gi_available():
            backends.append("gi")
        if not backends:
            raise CameraUnavailable("no GStreamer access for CSI: install python3-gi (sudo apt install python3-gi "
                               "gir1.2-gst-plugins-base-1.0) or use a GStreamer-enabled OpenCV")
        errors = []
        for sw, sh in modes:
            for backend in backends:
                if self._halt.is_set():
                    raise RuntimeError("stopping")
                cap = None
                try:
                    if backend == "opencv":
                        cap = cv2.VideoCapture(
                            csi_pipeline(sensor, self.width, self.height, self.fps, self.flip, sw, sh),
                            cv2.CAP_GSTREAMER)
                        if not cap.isOpened():
                            raise RuntimeError("OpenCV could not open the pipeline")
                    else:
                        cap = _GiAppSink(csi_pipeline(sensor, self.width, self.height, self.fps, self.flip,
                                                      sw, sh, bgr=False))
                    deadline = time.monotonic() + 4.0
                    ok = False
                    while time.monotonic() < deadline and not ok and not self._halt.is_set():
                        ok, _ = cap.read()
                    if not ok:
                        raise RuntimeError("no frame within 4 s")
                    self.stats.backend = f"nvarguscamerasrc/{backend}" + (f" {sw}x{sh}" if sw else " auto")
                    return cap
                except Exception as exc:
                    errors.append(f"{backend} {sw or 'auto'}x{sh or ''}: {exc}")
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
        raise CameraUnavailable(f"CSI sensor {sensor} gave no frames (is nvargus-daemon running? "
                                f"sudo systemctl restart nvargus-daemon); tried " + "; ".join(errors))

    def _open_v4l2(self, target):
        cap = cv2.VideoCapture(target, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open V4L2 device {target} (busy, or permissions: user in 'video' group?)")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # USB2 bandwidth for 2 cameras
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = False
        deadline = time.monotonic() + 3.0
        while not ok and time.monotonic() < deadline and not self._halt.is_set():
            ok, _ = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"V4L2 device {target} opened but delivered no frames")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.stats.backend = f"v4l2 /dev/video{target} {w}x{h}" if isinstance(target, int) else f"v4l2 {target}"
        return cap

    def _switch_to_usb(self) -> bool:
        """CSI sensor absent: take over a USB (UVC) camera if one is plugged in."""
        src = self.source.lower()
        if not (src.startswith("auto:") or (self.usb_fallback and src.startswith("csi:"))):
            return False
        cams = usb_cameras()
        want = self.usb_index if self.usb_index is not None else int(self.source.split(":", 1)[1] or 0)
        if want >= len(cams):
            return False
        cam = cams[want]
        log.warning("camera %s: CSI sensor not present, using USB camera /dev/video%d (%s)",
                    self.name, cam["index"], cam["name"])
        self.requested = self.requested or self.source
        self.source = f"v4l2:{cam['index']}"
        return True

    def _fit(self, image: np.ndarray) -> np.ndarray:
        """Center-crop to the target aspect ratio, then resize."""
        h, w = image.shape[:2]
        target = self.width / self.height
        if abs(w / h - target) > 0.01:
            if w / h > target:
                new_w = int(h * target)
                x0 = (w - new_w) // 2
                image = image[:, x0:x0 + new_w]
            else:
                new_h = int(w / target)
                y0 = (h - new_h) // 2
                image = image[y0:y0 + new_h]
        if image.shape[1] != self.width or image.shape[0] != self.height:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return image

    def _read(self):
        if self._cap == "still":
            return True, self._still.copy()
        cap = self._cap
        ok, image = cap.read()
        if not ok and self.stats.backend == "file":
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            ok, image = cap.read()
        if ok and image is not None and (image.shape[1] != self.width or image.shape[0] != self.height):
            image = self._fit(image)
        return ok, image

    # ------------------------------------------------------------------ thread
    def start(self) -> "Camera":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._halt.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            self._close()

    def _close(self) -> None:
        if self._cap not in (None, "still"):
            try:
                self._cap.release()
            except Exception:  # pragma: no cover - driver specific
                pass
        self._cap = None
        self.stats.connected = False

    def _run(self) -> None:
        backoff = 0.5
        failures = 0
        while not self._halt.is_set():
            if self._cap is None:
                try:
                    self._cap = self._open()
                    if self._cap != "still" and not self._cap.isOpened():
                        raise RuntimeError("source did not open")
                    self.stats.connected = True
                    self.stats.error = ""
                    backoff = 0.5
                    failures = 0
                    log.info("camera %s opened %s via %s", self.name, self.source, self.stats.backend)
                except CameraUnavailable as exc:
                    self.stats.error = str(exc)
                    self.stats.connected = False
                    self._cap = None
                    if self._switch_to_usb():
                        continue
                    if not self._explained:
                        self._explained = True
                        help_text = CSI_HELP if self.source.lower().startswith(("csi:", "auto:")) else \
                            "Plug in a USB camera or choose another source (--driver/--passenger, cameras.*.source)."
                        log.error("camera %s: %s\n%s", self.name, exc, help_text)
                    else:
                        log.warning("camera %s: %s (retry in 30s)", self.name, exc)
                    self._halt.wait(30.0)  # hardware absent: check rarely, keep the log readable
                    continue
                except Exception as exc:  # keep retrying, cameras do come back
                    self.stats.error = str(exc)
                    self.stats.connected = False
                    self._cap = None
                    if self._halt.is_set():
                        break
                    log.warning("camera %s open failed: %s (retry in %.1fs)", self.name, exc, backoff)
                    self._halt.wait(backoff)
                    backoff = min(backoff * 2, 30.0)
                    self.stats.reconnects += 1
                    continue
            started = time.monotonic()
            ok, image = self._read()
            if not ok or image is None:
                failures += 1
                if failures >= 30:
                    log.warning("camera %s lost, reopening", self.name)
                    self._close()
                    self.stats.reconnects += 1
                else:
                    self._halt.wait(0.01)
                continue
            failures = 0
            if self.flip and not self.stats.backend.startswith("nvarguscamerasrc"):  # nvvidconv already flipped
                image = _apply_flip(image, self.flip)
            now = time.monotonic()
            with self._cond:
                self._seq += 1
                if self._latest is not None and self._consumed < self._latest.seq:
                    self.stats.dropped += 1  # overwritten before any consumer took it
                self._latest = Frame(image, now, time.time(), self._seq, self.name)
                self._cond.notify_all()
            self._tick(now)
            if self._pace:
                remaining = self._pace - (time.monotonic() - started)
                if remaining > 0:
                    self._halt.wait(remaining)

    def _tick(self, now: float) -> None:
        stats = self.stats
        stats.frames += 1
        stats._times.append(now)
        while stats._times and now - stats._times[0] > 2.0:
            stats._times.pop(0)
        if len(stats._times) > 1:
            stats.fps = (len(stats._times) - 1) / (stats._times[-1] - stats._times[0])

    # ------------------------------------------------------------------ consumer API
    def latest(self) -> Frame | None:
        with self._cond:
            return self._latest

    def wait_next(self, after_seq: int, timeout: float = 1.0) -> Frame | None:
        """Block until a frame newer than ``after_seq`` exists, return the newest one."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._halt.is_set():
                if self._latest is not None and self._latest.seq > after_seq:
                    self._consumed = self._latest.seq
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
        return None


def _apply_flip(image: np.ndarray, method: int) -> np.ndarray:
    # nvvidconv flip-method semantics: 0 none, 1 ccw90, 2 rot180, 3 cw90, 4 hflip, 6 vflip
    if method == 1:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if method == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    if method == 3:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if method == 4:
        return cv2.flip(image, 1)
    if method == 6:
        return cv2.flip(image, 0)
    return image


class CaptureService:
    """Owns both cabin cameras (spec Thread 1)."""

    def __init__(self, cfg) -> None:
        cams = cfg.cameras
        self.driver = Camera("driver", **_cam_args(cams.driver))
        self.passenger = Camera("passenger", **_cam_args(cams.passenger)) if cams.passenger.get("source") else None

    def start(self) -> "CaptureService":
        self.driver.start()
        if self.passenger:
            self.passenger.start()
        return self

    def stop(self) -> None:
        self.driver.stop()
        if self.passenger:
            self.passenger.stop()


def _cam_args(section) -> dict:
    return {
        "source": section.get("source"),
        "width": section.get("width", 640),
        "height": section.get("height", 480),
        "fps": section.get("fps", 30),
        "flip": section.get("flip", 0),
        "sensor_size": section.get("sensor_size"),
        "usb_fallback": section.get("usb_fallback", True),
        "usb_index": section.get("usb_index"),
    }


if __name__ == "__main__":  # quick sensor check: python -m dms.capture csi:0
    import sys

    logging.basicConfig(level=logging.INFO)
    cam = Camera("probe", sys.argv[1] if len(sys.argv) > 1 else "csi:0").start()
    seq = 0
    end = time.monotonic() + 5
    while time.monotonic() < end:
        frame = cam.wait_next(seq, 2.0)
        if frame:
            seq = frame.seq
    print(f"{cam.source}: {cam.stats.fps:.1f} fps, backend={cam.stats.backend}, frames={cam.stats.frames}")
    cam.stop()
