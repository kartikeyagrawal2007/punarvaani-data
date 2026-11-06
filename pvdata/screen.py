"""Automatic review of a YouTube video before it is downloaded in full (replaces checking every row by hand).

    3 x 20 s samples at 360p (a few MB) ─► face: one frontal, big-enough face in >= 50% of frames (2 fps)
                                        ─► audio: Whisper language ID says Hindi (Hindi + Urdu >= 0.6)
                                                  and >= 85% of transcribed words are pure Devanagari
                                                  and someone talks (>= 1 word/s)

The face rules are the clip rules from ``FaceConfig`` (yaw, pitch, face size, one face) with the mouth-size
limit scaled to the sample's resolution. A video passes only if all checks pass; the scores are kept so
`pvdata auto` can mine the most promising videos first.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from pvdata import media
from pvdata.asr import Transcriber, Word
from pvdata.config import Config, FaceConfig, ScreenConfig
from pvdata.face import FaceObs, frame_problem

log = logging.getLogger("pvdata")
LATIN = re.compile(r"[A-Za-z]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


@dataclass
class ScreenResult:
    ok: bool
    reason: str = ""
    face_ratio: float = 0.0
    hindi_prob: float | None = None
    devanagari_ratio: float = 0.0
    words_per_s: float = 0.0
    face_problems: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Rough share of the video likely to become clips; used to mine the best videos first."""
        return round(self.face_ratio * self.devanagari_ratio, 3)

    def to_dict(self) -> dict:
        return {**asdict(self), "score": self.score}


def sample_windows(duration: float, cfg: ScreenConfig) -> list[tuple[float, float]]:
    """Evenly spread windows that avoid the first and last 10% (intros, outros, ads)."""
    if duration <= 0:
        return [(0.0, cfg.sample_s)]
    n = max(1, cfg.samples)
    fracs = [0.5] if n == 1 else [0.2 + 0.6 * i / (n - 1) for i in range(n)]
    out = []
    for f in fracs:
        a = max(0.0, min(duration - cfg.sample_s, f * duration - cfg.sample_s / 2))
        out.append((round(a, 1), round(min(duration, a + cfg.sample_s), 1)))
    return out


def face_cfg_for(height: int, base: FaceConfig) -> FaceConfig:
    """Same face rules at a lower resolution: the mouth-size limit (in pixels at 720p) scales with height."""
    return replace(base, min_mouth_px=max(8, round(base.min_mouth_px * height / 720)))


def is_devanagari_word(text: str) -> bool:
    return bool(DEVANAGARI.search(text)) and not LATIN.search(text)


def decide(obs: list[FaceObs], frame_h: int, words: list[Word], speech_s: float, hindi_prob: float | None,
           cfg: Config) -> ScreenResult:  # fmt: skip
    """Turn the face observations and transcript of the samples into a pass/fail decision."""
    sc = cfg.screen
    fcfg = face_cfg_for(frame_h, cfg.face)
    problems = [frame_problem(o, fcfg) for o in obs]
    face_ratio = sum(p is None for p in problems) / len(problems) if problems else 0.0
    tally: dict[str, int] = {}
    for p in problems:
        if p:
            tally[p] = tally.get(p, 0) + 1
    deva = sum(is_devanagari_word(w.text) for w in words) / len(words) if words else 0.0
    wps = len(words) / speech_s if speech_s > 0 else 0.0
    res = ScreenResult(True, "", round(face_ratio, 3), None if hindi_prob is None else round(hindi_prob, 3),
                       round(deva, 3), round(wps, 2), tally)  # fmt: skip
    if face_ratio < sc.min_face_ratio:
        worst = max(tally, key=tally.get) if tally else "no_face"
        res.ok, res.reason = False, f"screen_face_{worst}"
    elif wps < sc.min_words_per_s:
        res.ok, res.reason = False, "screen_no_speech"
    elif hindi_prob is not None and hindi_prob < sc.min_hindi_prob:
        res.ok, res.reason = False, "screen_not_hindi"
    elif deva < sc.min_devanagari_ratio:
        res.ok, res.reason = False, "screen_too_much_english"
    return res


class Screener:
    """Downloads the samples (``fetch``) and scores them (``score``); split so downloads can run ahead."""

    def __init__(self, cfg: Config, work: str | Path, transcriber: Transcriber,
                 face_factory: Callable[[float], Callable[[np.ndarray], FaceObs]], cookies: str | None = None):  # fmt: skip
        self.cfg, self.work = cfg, Path(work) / "screen"
        self.transcriber, self.face_factory, self.cookies = transcriber, face_factory, cookies

    def fetch(self, video_id: str, duration: float) -> list[Path]:
        url = f"https://www.youtube.com/watch?v={video_id}"
        windows = sample_windows(duration, self.cfg.screen)
        return media.download_sections(url, self.work / video_id, windows, self.cfg.screen.height, self.cookies)

    def score(self, samples: list[Path]) -> ScreenResult:
        if not samples:
            return ScreenResult(False, "screen_download_failed")
        sc = self.cfg.screen
        obs: list[FaceObs] = []
        words: list[Word] = []
        speech_s, frame_h, probs = 0.0, sc.height, []
        for path in samples:
            frames = media.read_frames(path, 0.0, None, sc.fps)  # type: ignore[arg-type]
            if len(frames):
                frame_h = frames.shape[1]
                detector = self.face_factory(sc.fps)
                try:
                    obs += [detector(f) for f in frames]
                finally:
                    if hasattr(detector, "close"):
                        detector.close()
            wav = path.with_suffix(".wav")
            media.extract_audio(path, wav)
            speech_s += len(media.read_wav(wav)) / media.SAMPLE_RATE
            words += self.transcriber.transcribe(wav)
            lp = self.transcriber.language_probs(wav)
            if lp is not None:
                probs.append(lp.get("hi", 0.0) + lp.get("ur", 0.0))
        hindi = float(np.mean(probs)) if probs else None
        return decide(obs, frame_h, words, speech_s, hindi, self.cfg)

    def screen(self, video_id: str, duration: float, samples: list[Path] | None = None) -> ScreenResult:
        """Fetch (unless ``samples`` were fetched ahead) and score one video, then delete the samples."""
        try:
            samples = samples if samples is not None else self.fetch(video_id, duration)
            return self.score(samples)
        except Exception as e:  # noqa: BLE001 - a broken video is a rejection, not a crash
            log.warning("screen %s failed: %s", video_id, e)
            return ScreenResult(False, f"screen_error:{type(e).__name__}")
        finally:
            shutil.rmtree(self.work / video_id, ignore_errors=True)
