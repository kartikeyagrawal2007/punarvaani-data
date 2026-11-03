"""The list of videos to mine: one row per video, with the speaker it shows.

CSV columns (header required):
    url,speaker,start,end,notes
- url:     YouTube (or any yt-dlp) link, or a local file path
- speaker: short id for the person on screen (e.g. ``anchor_ravish``). It becomes ``spk01``, ``spk02``, ...
           in the output, in order of first appearance. Choose videos with ONE speaker facing the camera.
- start, end: optional section in seconds (or mm:ss / hh:mm:ss) to keep, e.g. skip intros and B-roll.
- notes:   free text (licence, channel, why chosen).
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path


def parse_time(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    parts = [float(p) for p in str(value).strip().split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


@dataclass
class Source:
    url: str
    speaker: str
    start: float | None = None
    end: float | None = None
    notes: str = ""

    @property
    def source_id(self) -> str:
        """Stable short id for caching (same URL + section -> same id)."""
        key = f"{self.url}|{self.start}|{self.end}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    @property
    def is_local(self) -> bool:
        return not self.url.startswith(("http://", "https://"))


def read_sources(path: str | Path) -> list[Source]:
    rows: list[Source] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        missing = {"url", "speaker"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for i, row in enumerate(reader, start=2):
            url, speaker = (row.get("url") or "").strip(), (row.get("speaker") or "").strip()
            if not url:
                continue
            if not speaker:
                raise ValueError(f"{path}:{i}: every row needs a speaker id")
            src = Source(
                url, speaker, parse_time(row.get("start")), parse_time(row.get("end")), (row.get("notes") or "").strip()
            )
            if src.start is not None and src.end is not None and src.end <= src.start:
                raise ValueError(f"{path}:{i}: end must be after start")
            rows.append(src)
    return rows


def speaker_ids(sources: list[Source]) -> dict[str, str]:
    """Map free-form speaker names to spk01, spk02, ... in order of first appearance."""
    mapping: dict[str, str] = {}
    for s in sources:
        if s.speaker not in mapping:
            mapping[s.speaker] = f"spk{len(mapping) + 1:02d}"
    return mapping
