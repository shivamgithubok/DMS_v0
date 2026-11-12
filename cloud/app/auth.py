"""Guardian session for the phone API.

The vehicle keeps using the device key. The phone uses a bearer token minted
by POST /v1/session. When GUARDIAN_AUTH_REQUIRED is off, phone routes stay
open so the current local demo keeps working.
"""

from __future__ import annotations

import time

import jwt
from fastapi import HTTPException

from .config import settings

DEV_MEDIA_KEY = "dev-media-key"
_ISSUER = "vms-guardian"


def session_configured() -> bool:
    return bool(settings.guardian_api_key and settings.session_secret)


def issue_session(guardian_id: str) -> tuple[str, int]:
    now = int(time.time())
    exp = now + int(settings.session_ttl_s)
    token = jwt.encode(
        {"iss": _ISSUER, "sub": guardian_id, "iat": now, "exp": exp},
        settings.session_secret,
        algorithm="HS256",
    )
    if isinstance(token, bytes):
        token = token.decode()
    return token, exp * 1000


def guardian_id_from_token(token: str) -> str:
    payload = jwt.decode(token, settings.session_secret, algorithms=["HS256"], issuer=_ISSUER)
    guardian_id = str(payload.get("sub") or "").strip()
    if not guardian_id:
        raise jwt.InvalidTokenError("missing subject")
    return guardian_id


def bearer_token(header: str | None) -> str:
    if not header:
        return ""
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return ""
    return value.strip()


def check_phone_session(header: str | None) -> str | None:
    """Return the guardian id, or None when auth is off. Raise 401 when it is on."""
    if not settings.auth_required:
        return None
    token = bearer_token(header)
    if not token or not settings.session_secret:
        raise HTTPException(401, "sign in required")
    try:
        return guardian_id_from_token(token)
    except jwt.PyJWTError as exc:
        raise HTTPException(401, "sign in required") from exc


def check_device_key(key: str) -> None:
    expected = settings.media_device_key or ""
    if settings.require_device_key and (not expected or expected == DEV_MEDIA_KEY or key == DEV_MEDIA_KEY):
        raise HTTPException(403, "bad device key")
    if not expected or key != expected:
        raise HTTPException(403, "bad device key")
