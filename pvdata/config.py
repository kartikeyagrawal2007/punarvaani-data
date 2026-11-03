"""All thresholds in one place, loaded from YAML (configs/default.yaml) with dataclass defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ClipConfig:
    # TEAM_GUIDE: 1.5-2.5 s per clip (about 2-6 words), so Stage 1's 16 frames -> 32 CTC steps can align.
    min_s: float = 1.5
    max_s: float = 2.5
    min_words: int = 2
    max_words: int = 6
    pad_s: float = 0.08  # context kept before the first and after the last word
    max_gap_s: float = 0.35  # a pause longer than this ends a clip (don't glue sentences together)
    ctc_steps: int = 32  # T_ctc of the Stage-1 model; a clip needs U + R <= this
    fps: int = 25


@dataclass
class TextConfig:
    spell_out_numbers: bool = True
    reject_latin: bool = True  # TEAM_GUIDE: pure Devanagari only; clips with English words are dropped
    min_asr_word_prob: float = 0.45  # drop a clip if any word's ASR confidence is below this


@dataclass
class FaceConfig:
    sample_every: int = 2  # check every Nth frame (speed)
    min_detect_ratio: float = 0.95  # TEAM_GUIDE: crop success >= 95%
    max_faces: int = 1
    max_yaw_deg: float = 20.0  # frontal: head turned left/right
    max_pitch_deg: float = 20.0  # frontal: head tilted up/down
    min_face_height_frac: float = 0.18  # face height / frame height (small faces give useless mouth crops)
    min_mouth_px: int = 40  # mouth width in pixels in the source frame


@dataclass
class SyncConfig:
    enabled: bool = True
    max_lag_frames: int = 5  # search +-0.2 s for the best audio/mouth alignment
    min_corr: float = 0.15  # correlation of mouth opening vs speech loudness
    max_abs_offset_frames: int = 3  # best alignment must be near zero lag (dubbing / B-roll fail this)


@dataclass
class VideoConfig:
    min_height: int = 720  # TEAM_GUIDE: at least 720p
    out_height: int = 720
    out_fps: int = 25
    keep_audio: bool = True  # Stage 1 ignores audio; keeping it lets us re-check sync later


@dataclass
class ASRConfig:
    backend: str = "faster-whisper"  # faster-whisper | whisperx | hf
    model: str = "large-v3"
    language: str = "hi"
    device: str = "auto"
    compute_type: str = "auto"


@dataclass
class Config:
    clip: ClipConfig = field(default_factory=ClipConfig)
    text: TextConfig = field(default_factory=TextConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 0

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        cfg = cls()
        if path is None:
            return cfg
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            current = getattr(cfg, f.name)
            if hasattr(current, "__dataclass_fields__"):
                unknown = set(value) - {x.name for x in fields(current)}
                if unknown:
                    raise ValueError(f"unknown keys in '{f.name}': {sorted(unknown)}")
                for k, v in value.items():
                    setattr(current, k, v)
            else:
                setattr(cfg, f.name, value)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
