#!/usr/bin/env python3
"""Tiny MQTT broker for laptop testing. Not used in production."""

from __future__ import annotations

import asyncio
import os

from amqtt.broker import Broker

HOST = os.environ.get("MQTT_HOST", "127.0.0.1")
PORT = int(os.environ.get("MQTT_PORT", "1883"))

CONFIG = {
    "listeners": {"default": {"type": "tcp", "bind": f"{HOST}:{PORT}"}},
    "sys_interval": 0,
    "auth": {"allow-anonymous": True},
    "topic-check": {"enabled": False},
}


async def main() -> None:
    broker = Broker(CONFIG)
    await broker.start()
    print(f"local MQTT broker mqtt://{HOST}:{PORT}", flush=True)
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
