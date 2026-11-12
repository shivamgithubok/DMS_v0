"""LiveKit-compatible access tokens (HS256). Works with a real SFU or dummy mode."""

from __future__ import annotations

import time
import uuid

import jwt

from .config import settings


def room_name(ride_id: str) -> str:
    return ride_id


def mint(
    identity: str,
    room: str,
    *,
    can_publish: bool,
    can_subscribe: bool,
    ttl_s: int,
    name: str = "",
    publish_sources: list[str] | None = None,
) -> tuple[str, int]:
    now = int(time.time())
    exp = now + int(ttl_s)
    video = {
        "room": room,
        "roomJoin": True,
        "canPublish": bool(can_publish),
        "canSubscribe": bool(can_subscribe),
        "canPublishData": False,
    }
    if publish_sources:
        video["canPublishSources"] = list(publish_sources)
    payload = {
        "iss": settings.livekit_api_key,
        "sub": identity,
        "jti": uuid.uuid4().hex,
        "iat": now,
        "nbf": now - 5,
        "exp": exp,
        "name": name or identity,
        "video": video,
    }
    token = jwt.encode(payload, settings.livekit_api_secret, algorithm="HS256")
    if isinstance(token, bytes):
        token = token.decode()
    return token, exp * 1000


def decode(token: str) -> dict:
    return jwt.decode(
        token,
        settings.livekit_api_secret,
        algorithms=["HS256"],
        issuer=settings.livekit_api_key,
    )


def viewer_token(ride_id: str, viewer_id: str) -> tuple[str, int]:
    return mint(
        f"viewer-{viewer_id}",
        room_name(ride_id),
        can_publish=False,
        can_subscribe=True,
        ttl_s=settings.viewer_ttl_s,
        name=viewer_id,
    )


def talker_token(ride_id: str) -> tuple[str, int]:
    """Phone mic only. The app publishes one audio track named guardian-mic."""
    return mint(
        f"guardian-{uuid.uuid4().hex[:10]}",
        room_name(ride_id),
        can_publish=True,
        can_subscribe=False,
        ttl_s=settings.viewer_ttl_s,
        name="guardian",
        publish_sources=["microphone"],
    )


def speaker_token(ride_id: str, device_id: str) -> tuple[str, int]:
    """Car listens for the guardian mic. It does not publish on this identity."""
    return mint(
        f"speaker-{device_id}",
        room_name(ride_id),
        can_publish=False,
        can_subscribe=True,
        ttl_s=settings.publisher_ttl_s,
        name=device_id,
    )


def publisher_token(ride_id: str, device_id: str, *, can_publish: bool = False) -> tuple[str, int]:
    return mint(
        f"camera-{device_id}",
        room_name(ride_id),
        can_publish=can_publish,
        can_subscribe=False,
        ttl_s=settings.publisher_ttl_s,
        name=device_id,
    )
