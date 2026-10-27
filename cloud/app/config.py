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


settings = Settings()
