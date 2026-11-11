"""One Guardian API call at a time.

Video stills, alert clips, talk, and the recording poll were opening the API
together. Those calls sat until the timeout, and the MQTT keepalive died
while the sockets were stuck.
"""

from __future__ import annotations

import json
import threading
import urllib.request

_lock = threading.Lock()


def post_json(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _lock:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode() or "{}")
