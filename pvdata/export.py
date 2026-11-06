"""Output layout (exactly what TEAM_GUIDE.md asks for) plus bookkeeping files.

<out>/raw_videos/spk01/spk01_001.mp4      clip (720p, 25 fps, H.264, audio kept)
<out>/raw_videos/spk01/spk01_001.txt      UTF-8 Devanagari transcript
<out>/manifest.csv                        one row per candidate clip, accepted or rejected (and why)
<out>/report.json                         counts per stage and rejection reason
<out>/splits.json                         speaker-disjoint train / val / test speaker lists
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

CLIP_NAME = re.compile(r"^(spk\d{2,})_(\d{3,})$")
MANIFEST_FIELDS = [
    "clip_id", "speaker", "speaker_name", "source_url", "source_id", "license", "channel", "title", "upload_date", "start", "end", "duration", "text",
    "raw_asr_text", "ctc_need", "asr_min_prob", "face_ratio", "yaw", "pitch", "sync_corr", "sync_offset",
    "status", "reason",
]  # fmt: skip


def clip_paths(out: Path, speaker: str, index: int) -> tuple[Path, Path]:
    d = out / "raw_videos" / speaker
    stem = f"{speaker}_{index:03d}"
    return d / f"{stem}.mp4", d / f"{stem}.txt"


def next_index(out: Path, speaker: str) -> int:
    d = out / "raw_videos" / speaker
    nums = [int(m.group(2)) for p in d.glob(f"{speaker}_*.mp4") if (m := CLIP_NAME.match(p.stem))] if d.exists() else []
    return max(nums, default=0) + 1


def write_transcript(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


class Manifest:
    """Append-only CSV, so an interrupted run keeps everything it already decided."""

    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, MANIFEST_FIELDS).writeheader()

    def add(self, row: dict) -> None:
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, MANIFEST_FIELDS, extrasaction="ignore").writerow(row)

    def drop_source(self, source_id: str) -> list[str]:
        """Remove the rows of one source (from an interrupted run) and return the clip ids they had written."""
        rows = self.rows()
        keep = [r for r in rows if r.get("source_id") != source_id]
        if len(keep) == len(rows):
            return []
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, MANIFEST_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(keep)
        return [r["clip_id"] for r in rows if r.get("source_id") == source_id and r.get("clip_id")]

    def rows(self) -> list[dict]:
        with open(self.path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))


def is_youtube(url: str) -> bool:
    return any(h in (url or "") for h in ("youtube.com/", "youtu.be/"))


def license_of(row: dict) -> str:
    """yt-dlp only reports a licence for Creative Commons videos; any other YouTube video is under YouTube's
    standard licence (release IDs + timestamps only)."""
    if row.get("license"):
        return row["license"]
    return "youtube_standard" if is_youtube(row.get("source_url", "")) else "unknown"


def build_report(rows: list[dict]) -> dict:
    accepted = [r for r in rows if r["status"] == "accepted"]
    hours = sum(float(r["duration"]) for r in accepted) / 3600
    per_speaker = Counter(r["speaker"] for r in accepted)
    return {
        "candidates": len(rows),
        "accepted": len(accepted),
        "acceptance_rate_pct": round(100 * len(accepted) / len(rows), 1) if rows else 0.0,
        "accepted_hours": round(hours, 3),
        "speakers": len(per_speaker),
        "clips_per_speaker": dict(sorted(per_speaker.items())),
        "accepted_by_license": dict(Counter(license_of(r) for r in accepted).most_common()),
        "rejected_by_reason": dict(Counter(r["reason"] for r in rows if r["status"] != "accepted").most_common()),
    }


def speaker_splits(speakers: list[str], val_frac: float, test_frac: float, seed: int = 0,
                   clips: dict[str, int] | None = None) -> dict[str, list[str]]:  # fmt: skip
    """Speaker-disjoint split: every speaker's clips go to exactly one of train / val / test.

    The fractions set how many *speakers* go to val and test (at least one each). With ``clips`` (clips per
    speaker), val/test speakers are drawn only from mid-sized ones (2-15% of all clips) when there are enough:
    a speaker with half the data never becomes the test set, and a 4-clip speaker never becomes a useless one.
    """
    s = sorted(set(speakers))
    random.Random(seed).shuffle(s)
    n = len(s)
    n_test = max(1, round(n * test_frac)) if n >= 3 and test_frac > 0 else 0
    n_val = max(1, round(n * val_frac)) if n >= 2 and val_frac > 0 else 0
    if clips:
        total = sum(clips.get(x, 0) for x in s) or 1
        share = {x: clips.get(x, 0) / total for x in s}
        mid = [x for x in s if 0.02 <= share[x] <= 0.15]
        not_big = [x for x in s if share[x] <= 0.15 and x not in mid]
        rest = [x for x in s if x not in mid and x not in not_big]
        s = mid + sorted(not_big, key=lambda x: -share[x]) + rest  # preferred val/test speakers first
    return {
        "test": sorted(s[:n_test]),
        "val": sorted(s[n_test : n_test + n_val]),
        "train": sorted(s[n_test + n_val :]),
    }


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def materialise_splits(out: Path, splits: dict[str, list[str]], link: bool = True) -> dict[str, int]:
    """Create <out>/splits/{train,val,test}/spkXX -> raw_videos/spkXX (symlinks by default) for the trainer's
    dedicated-split mode. Returns clip counts per split."""
    counts: dict[str, int] = {}
    for split, spks in splits.items():
        base = out / "splits" / split
        base.mkdir(parents=True, exist_ok=True)
        n = 0
        for spk in spks:
            src = (out / "raw_videos" / spk).resolve()
            dst = base / spk
            if not src.exists():
                continue
            if dst.is_symlink() or dst.exists():
                if dst.is_symlink():
                    dst.unlink()
                else:
                    import shutil

                    shutil.rmtree(dst)
            if link:
                try:
                    dst.symlink_to(src, target_is_directory=True)
                except OSError:  # Windows without Developer Mode can't create symlinks: copy instead
                    import shutil

                    shutil.copytree(src, dst)
            else:
                import shutil

                shutil.copytree(src, dst)
            n += len(list(src.glob("*.mp4")))
        counts[split] = n
    return counts
