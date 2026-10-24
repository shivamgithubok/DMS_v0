"""Synthetic cabin audio for tests: road noise, speech-like voices, shouts, screams."""

from __future__ import annotations

import numpy as np

SR = 16000


def road_noise(seconds: float, level_db: float = -42.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    white = rng.standard_normal(n)
    # brown-ish: integrate and high-pass slightly
    brown = np.cumsum(white)
    brown -= np.convolve(brown, np.ones(400) / 400, mode="same")
    brown /= np.sqrt(np.mean(brown ** 2)) + 1e-9
    return (brown * 10 ** (level_db / 20)).astype(np.float32)


def voice(seconds: float, f0: float, level_db: float, jitter: float = 0.04, harmonics: int = 12,
          syllable_hz: float = 4.0, seed: int = 1, duty: float = 0.7) -> np.ndarray:
    """Harmonic source with pitch wobble and syllable-rate amplitude gating."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    wobble = 1 + jitter * np.sin(2 * np.pi * 5.5 * t) + 0.02 * np.cumsum(rng.standard_normal(n)) / np.sqrt(n)
    phase = 2 * np.pi * np.cumsum(f0 * wobble) / SR
    sig = sum((1.0 / k) * np.sin(k * phase) for k in range(1, harmonics + 1) if k * f0 < SR / 2 - 200)
    gate = (0.5 + 0.5 * np.sin(2 * np.pi * syllable_hz * t + rng.uniform(0, 6))) > (1 - duty)
    env = np.convolve(gate.astype(float), np.ones(240) / 240, mode="same")
    sig = sig * env
    sig /= np.sqrt(np.mean(sig ** 2)) + 1e-9
    return (sig * 10 ** (level_db / 20)).astype(np.float32)


def scream(seconds: float, level_db: float = -14.0, seed: int = 3) -> np.ndarray:
    return voice(seconds, 1050.0, level_db, jitter=0.06, harmonics=6, syllable_hz=0.5, seed=seed, duty=0.98)


def shout(seconds: float, f0: float = 330.0, level_db: float = -18.0, seed: int = 4) -> np.ndarray:
    return voice(seconds, f0, level_db, jitter=0.05, harmonics=14, syllable_hz=3.0, seed=seed, duty=0.8)


def mix(*parts: np.ndarray) -> np.ndarray:
    n = max(len(p) for p in parts)
    out = np.zeros(n, dtype=np.float32)
    for p in parts:
        out[:len(p)] += p
    return np.clip(out, -1, 1)


def place(base: np.ndarray, event: np.ndarray, at_s: float) -> np.ndarray:
    out = base.copy()
    i = int(at_s * SR)
    j = min(len(out), i + len(event))
    out[i:j] += event[:j - i]
    return np.clip(out, -1, 1)
