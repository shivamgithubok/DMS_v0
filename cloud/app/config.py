"""Runtime settings for the guardian API. Secrets come from the environment."""

from __future__ import annotations

import os


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    def __init__(self) -> None:
        raw = os.environ.get("DATABASE_URL") or "sqlite:///./data.db"
        if raw.startswith("postgres://"):
            raw = "postgresql://" + raw[len("postgres://") :]
        if raw.startswith("postgresql://") and "+psycopg" not in raw:
            raw = "postgresql+psycopg://" + raw[len("postgresql://") :]
        self.database_url = raw
        self.mqtt_host = os.environ.get("MQTT_HOST", "127.0.0.1")
        self.mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
        self.mqtt_username = os.environ.get("MQTT_USERNAME", "")
        self.mqtt_password = os.environ.get("MQTT_PASSWORD", "")
        self.mqtt_tls = _bool("MQTT_TLS")
        self.default_ride_id = os.environ.get("DEFAULT_RIDE_ID", "ride-1842")
        self.default_device_id = os.environ.get("DEFAULT_DEVICE_ID", "orin-nano-01")
        self.default_passenger = os.environ.get("DEFAULT_PASSENGER", "Meera Shah")
        self.livekit_url = os.environ.get("LIVEKIT_URL", "dummy://local")
        self.livekit_api_key = os.environ.get("LIVEKIT_API_KEY", "devkey")
        self.livekit_api_secret = os.environ.get("LIVEKIT_API_SECRET", "devsecret-change-me-please-32b")
        self.media_device_key = os.environ.get("MEDIA_DEVICE_KEY", "dev-media-key")
        # Desk default is open so the current phone build and local curl keep working.
        # Production (render.yaml) turns these on and supplies the two secrets.
        self.auth_required = _bool("GUARDIAN_AUTH_REQUIRED")
        self.require_device_key = _bool("GUARDIAN_REQUIRE_DEVICE_KEY")
        self.seed_demo = _bool("SEED_DEMO")
        self.guardian_api_key = os.environ.get("GUARDIAN_API_KEY", "")
        self.session_secret = os.environ.get("GUARDIAN_SESSION_SECRET", "")
        self.session_ttl_s = int(os.environ.get("GUARDIAN_SESSION_TTL_S", "86400"))
        self.cors_origins = [item.strip() for item in os.environ.get("CORS_ORIGINS", "").split(",") if item.strip()]
        self.viewer_ttl_s = int(os.environ.get("LIVEKIT_VIEWER_TTL_S", "900"))
        self.publisher_ttl_s = int(os.environ.get("LIVEKIT_PUBLISHER_TTL_S", "1800"))
        self.publisher_online_s = int(os.environ.get("PUBLISHER_ONLINE_S", "60"))
        self.link_grace_s = int(os.environ.get("LINK_GRACE_S", "900"))
        self.media_dir = os.environ.get("MEDIA_DIR", "data/media")
        self.clip_ttl_days = int(os.environ.get("CLIP_TTL_DAYS", "30"))
        self.fcm_server_key = os.environ.get("FCM_SERVER_KEY", "")


settings = Settings()
