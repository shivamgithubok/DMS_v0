import time

import dms.capture as C


def _sysfs(tmp_path, nodes):
    root = tmp_path / "video4linux"
    for i, (name, index) in enumerate(nodes):
        d = root / f"video{i}"
        d.mkdir(parents=True)
        (d / "name").write_text(name + "\n")
        (d / "index").write_text(f"{index}\n")
    return root


def test_sysfs_classification(tmp_path):
    root = _sysfs(tmp_path, [("vi-output, imx219 9-0010", 0), ("vi-output, imx219 10-0010", 0),
                             ("USB Camera: USB Camera", 0), ("USB Camera: USB Camera", 1)])
    assert [n["index"] for n in C.csi_sensors(root)] == [0, 1]
    assert [n["index"] for n in C.usb_cameras(root)] == [2]  # metadata node (index 1) skipped


def test_no_sysfs_means_unknown(tmp_path):
    assert C.csi_sensors(tmp_path / "missing") is None


def test_absent_csi_switches_to_usb(monkeypatch):
    monkeypatch.setattr(C, "csi_sensors", lambda root=None: [])
    monkeypatch.setattr(C, "usb_cameras", lambda root=None: [{"index": 7, "name": "UVC", "csi": False, "capture": True}])
    cam = C.Camera("driver", "csi:0").start()
    deadline = time.monotonic() + 3
    while cam.source == "csi:0" and time.monotonic() < deadline:
        time.sleep(0.05)
    t0 = time.monotonic()
    cam.stop()
    assert cam.source == "v4l2:7"
    assert time.monotonic() - t0 < 6


def test_absent_csi_without_usb_explains_once_and_stops_fast(monkeypatch, caplog):
    monkeypatch.setattr(C, "csi_sensors", lambda root=None: [])
    monkeypatch.setattr(C, "usb_cameras", lambda root=None: [])
    cam = C.Camera("passenger", "csi:1").start()
    time.sleep(0.5)
    t0 = time.monotonic()
    cam.stop()  # it is sleeping in the 30 s back-off: stop must not wait for it
    assert time.monotonic() - t0 < 1.0
    assert "not registered" in cam.stats.error
    assert sum("jetson-io" in r.getMessage() for r in caplog.records) == 1


class _FakeV4L2:
    def __init__(self, target):
        import numpy as np
        self.target, self.frame = target, np.full((480, 640, 3), 80, np.uint8)

    def read(self):
        time.sleep(0.01)
        return True, self.frame.copy()

    def isOpened(self):  # noqa: N802
        return True

    def release(self):
        pass


UVC = [{"index": 2, "name": "Logitech C920", "csi": False, "capture": True},
       {"index": 4, "name": "Arducam USB", "csi": False, "capture": True}]


def _fake_open(self, target):
    self.stats.backend = f"v4l2 /dev/video{target}"
    return _FakeV4L2(target)


def _wait(cam, cond, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.05)
    return cond()


def test_auto_uses_usb_and_delivers_frames(monkeypatch):
    monkeypatch.setattr(C, "csi_sensors", lambda root=None: [])
    monkeypatch.setattr(C, "usb_cameras", lambda root=None: UVC)
    monkeypatch.setattr(C.Camera, "_open_v4l2", _fake_open)
    drv, pas = C.Camera("driver", "auto:0").start(), C.Camera("passenger", "auto:1", fps=15).start()
    try:
        assert _wait(drv, lambda: drv.latest() is not None) and _wait(pas, lambda: pas.latest() is not None)
        assert (drv.source, pas.source) == ("v4l2:2", "v4l2:4")
        assert drv.requested == "auto:0" and drv.stats.connected
    finally:
        drv.stop(), pas.stop()


def test_auto_falls_back_when_registered_csi_gives_no_frames(monkeypatch):
    monkeypatch.setattr(C, "usb_cameras", lambda root=None: UVC)
    monkeypatch.setattr(C.Camera, "_open_v4l2", _fake_open)

    def dead_csi(self, sensor):
        raise C.CameraUnavailable(f"CSI sensor {sensor} gave no frames")
    monkeypatch.setattr(C.Camera, "_open_csi", dead_csi)
    cam = C.Camera("driver", "auto:0").start()
    try:
        assert _wait(cam, lambda: cam.latest() is not None)
        assert cam.source == "v4l2:2"
    finally:
        cam.stop()


def test_usb_source_resolves_nth_camera(monkeypatch):
    monkeypatch.setattr(C, "usb_cameras", lambda root=None: UVC)
    assert C.resolve_usb(0) == 2 and C.resolve_usb(1) == 4
    try:
        C.resolve_usb(2)
        raise AssertionError("expected CameraUnavailable")
    except C.CameraUnavailable as exc:
        assert "2 connected" in str(exc)


def test_csi_only_mode_never_switches(monkeypatch):
    monkeypatch.setattr(C, "csi_sensors", lambda root=None: [])
    monkeypatch.setattr(C, "usb_cameras", lambda root=None: UVC)
    cam = C.Camera("driver", "csi:0", usb_fallback=False).start()
    time.sleep(0.4)
    cam.stop()
    assert cam.source == "csi:0" and not cam.stats.connected


def test_cli_camera_flag():
    import main
    o = main.build_overrides(main.parse_args(["--camera", "usb"]))
    assert o["cameras"]["driver"]["source"] == "usb:0" and o["cameras"]["passenger"]["source"] == "usb:1"
    o = main.build_overrides(main.parse_args(["--camera", "csi"]))
    assert o["cameras"]["driver"]["usb_fallback"] is False
