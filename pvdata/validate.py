"""Check a raw_videos/ folder against the TEAM_GUIDE rules before handing it to Stage-1 training.

Works on any folder, including clips recorded by hand (volunteers), not only mined ones.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pvdata import media
from pvdata.config import Config
from pvdata.ctc import ctc_steps_for, fits_ctc
from pvdata.export import CLIP_NAME

ALLOWED = re.compile(r"^[ऀ-ॿ ]+$")


@dataclass
class ValidationReport:
    clips: int = 0
    speakers: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    by_speaker: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(raw_videos: str | Path, cfg: Config | None = None, check_media: bool = True) -> ValidationReport:
    cfg = cfg or Config()
    root = Path(raw_videos)
    rep = ValidationReport()
    if not root.is_dir():
        rep.errors.append(f"{root} is not a folder")
        return rep
    per_spk: Counter = Counter()
    for spk_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not re.match(r"^spk\d{2,}$", spk_dir.name):
            rep.warnings.append(f"{spk_dir.name}: folder name should look like spk01")
        stems = {p.stem for p in spk_dir.iterdir() if p.suffix.lower() in {".mp4", ".txt"}}
        for stem in sorted(stems):
            mp4, txt = spk_dir / f"{stem}.mp4", spk_dir / f"{stem}.txt"
            where = f"{spk_dir.name}/{stem}"
            m = CLIP_NAME.match(stem)
            if not m or m.group(1) != spk_dir.name:
                rep.errors.append(f"{where}: name must be {spk_dir.name}_NNN")
            if not mp4.exists():
                rep.errors.append(f"{where}: .txt without .mp4")
                continue
            if not txt.exists():
                rep.errors.append(f"{where}: .mp4 without .txt")
                continue
            rep.clips += 1
            per_spk[spk_dir.name] += 1
            try:
                text = txt.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                rep.errors.append(f"{where}: transcript is not UTF-8")
                continue
            if not text or not ALLOWED.match(text.replace("।", "")):
                rep.errors.append(f"{where}: transcript must be Devanagari, spaces and । only")
            info = None
            if check_media:
                try:
                    info = media.probe(mp4)
                except Exception as e:
                    rep.errors.append(f"{where}: unreadable video ({type(e).__name__})")
                    continue
            if cfg.clip.ctc_steps > 0:
                steps = cfg.clip.ctc_steps
            elif info is not None:
                steps = ctc_steps_for(info["duration"], info["fps"] or 25)
            else:  # no media check: assume the shortest allowed clip at 25 fps (strictest)
                steps = ctc_steps_for(cfg.clip.min_s, 25)
            ok, need = fits_ctc(text, steps)
            if not ok:
                rep.errors.append(f"{where}: needs {need} CTC steps > {steps} (clip too long for Stage 1)")
            if info is not None:
                if not (cfg.clip.min_s - 0.15 <= info["duration"] <= cfg.clip.max_s + 0.15):
                    rep.warnings.append(
                        f"{where}: duration {info['duration']:.2f}s outside {cfg.clip.min_s}-{cfg.clip.max_s}s"
                    )
                if info["height"] < cfg.video.min_height:
                    rep.warnings.append(f"{where}: {info['height']}p is below {cfg.video.min_height}p")
                if info["fps"] < 23.5:
                    rep.warnings.append(f"{where}: {info['fps']:.1f} fps (need 24-30)")
    rep.by_speaker = dict(sorted(per_spk.items()))
    rep.speakers = len(per_spk)
    if rep.clips == 0:
        rep.errors.append("no clips found")
    return rep
