"""Lightweight audio-visual sync check: does the mouth open when the voice is loud?

We correlate the mouth-opening signal (from the face check, one value per video frame) with the
speech loudness envelope (RMS per video frame) over small time shifts. A real talking head gives a
clear positive correlation near zero shift. Dubbed audio, voice-over on B-roll, or a different person
talking off-screen give a weak or shifted correlation and are dropped.

This is a cheap stand-in for SyncNet (joonson/syncnet_python), which can replace it later for
higher precision on large runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pvdata.config import SyncConfig


@dataclass
class SyncResult:
    ok: bool
    corr: float
    offset_frames: int
    reason: str = ""


def loudness_envelope(audio: np.ndarray, sr: int, fps: int, n_frames: int) -> np.ndarray:
    hop = sr / fps
    env = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        a, b = int(round(i * hop)), int(round((i + 1) * hop))
        seg = audio[a:b]
        env[i] = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if len(seg) else 0.0
    return env


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def sync_score(openings: np.ndarray, envelope: np.ndarray, cfg: SyncConfig) -> SyncResult:
    openings = np.asarray(openings, dtype=np.float64)
    n = min(len(openings), len(envelope))
    o, e = openings[:n], np.asarray(envelope, dtype=np.float64)[:n]
    valid = ~np.isnan(o)
    if valid.sum() < max(10, n // 2):
        return SyncResult(False, 0.0, 0, "too_few_mouth_frames")
    o = np.where(valid, o, np.nanmean(o))
    if o.std() < 1e-6:
        return SyncResult(False, 0.0, 0, "mouth_not_moving")
    if e.std() < 1e-9:
        return SyncResult(False, 0.0, 0, "no_audio_energy")
    best_corr, best_lag = -1.0, 0
    for lag in range(-cfg.max_lag_frames, cfg.max_lag_frames + 1):
        a = o[max(0, lag) : n + min(0, lag)]
        b = e[max(0, -lag) : n - max(0, lag)]
        if len(a) < 8:
            continue
        c = float(np.mean(_zscore(a) * _zscore(b)))
        if c > best_corr:
            best_corr, best_lag = c, lag
    if best_corr < cfg.min_corr:
        return SyncResult(False, best_corr, best_lag, "weak_av_correlation")
    if abs(best_lag) > cfg.max_abs_offset_frames:
        return SyncResult(False, best_corr, best_lag, "av_offset")
    return SyncResult(True, best_corr, best_lag)
