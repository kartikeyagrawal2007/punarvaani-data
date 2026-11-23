"""Re-check clips already in a dataset with the strict active-speaker checks (one shot + SyncNet).

Clips mined before SyncNet was added passed only the light correlation check, which lets some voice-over
B-roll through (a news anchor talking over footage of someone else). This re-scores every clip in place:

    pvdata resync --out dataset            -> dataset/resync/resync.csv, one row per clip (resumable)
    pvdata resync --out dataset --apply    -> failing clips moved to dataset/excluded_sync/<spk>/, manifest
                                              rows marked, and every clip of a video dropped when most fail

Stop `pvdata auto` before --apply (both write the manifest). Afterwards re-run identify, split and validate.
"""

from __future__ import annotations

import csv
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from pvdata import media
from pvdata.config import Config
from pvdata.export import MANIFEST_FIELDS, Manifest
from pvdata.shots import change_scores, find_cuts
from pvdata.syncnet import FPS

FIELDS = ["clip_id", "speaker", "frames", "cut", "face_cov", "sync_conf", "syncnet_offset", "verdict"]
SYNC_VERDICTS = {"ok", "low_syncnet_conf", "syncnet_offset"}  # clips that reached the SyncNet decision


def check_file(mp4: Path, cfg: Config, face_factory: Callable, scorer: Callable, tmp: Path) -> dict:
    frames = media.read_frames(mp4, 0.0, None, FPS)
    row = {"frames": len(frames), "cut": 0}
    if cfg.shot.enabled and find_cuts(change_scores(frames), cfg.shot.min_score, cfg.shot.ratio):
        return {**row, "cut": 1, "verdict": "shot_cut"}
    detector = face_factory()
    try:
        obs = [detector(f) for f in frames]
    finally:
        if hasattr(detector, "close"):
            detector.close()
    boxes = [o.box if o.n_faces == 1 else None for o in obs]
    cov = sum(b is not None for b in boxes) / max(len(boxes), 1)
    row["face_cov"] = round(cov, 3)
    if cov < 0.9:
        return {**row, "verdict": "face_coverage"}
    wav = tmp / "clip.wav"
    try:
        media.extract_audio(mp4, wav)
        audio = media.read_wav(wav)
    except Exception:
        return {**row, "verdict": "no_audio"}
    res = scorer(frames, boxes, audio)
    if res is None:
        return {**row, "verdict": "too_short"}
    row.update(sync_conf=round(res.conf, 2), syncnet_offset=res.offset)
    if res.conf < cfg.sync.min_conf:
        return {**row, "verdict": "low_syncnet_conf"}
    if abs(res.offset) > cfg.sync.max_syncnet_offset:
        return {**row, "verdict": "syncnet_offset"}
    return {**row, "verdict": "ok"}


def source_rule(rows: list[dict], cfg: Config) -> set[str]:
    """Speaker folders (one per video) where too few clips pass: all their clips go."""
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["verdict"] in SYNC_VERDICTS:
            by_spk[r["speaker"]].append(r)
    bad = set()
    for spk, rs in by_spk.items():
        if len(rs) >= cfg.sync.min_source_checked:
            rate = sum(r["verdict"] == "ok" for r in rs) / len(rs)
            if rate < cfg.sync.min_source_pass_rate:
                bad.add(spk)
    return bad


def resync(out: str | Path, cfg: Config, face_factory: Callable, scorer: Callable, apply: bool = False,
           progress: Callable | None = None) -> dict:  # fmt: skip
    out = Path(out)
    clips = sorted((out / "raw_videos").glob("*/*.mp4"))
    csv_path = out / "resync" / "resync.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as f:
            rows = {r["clip_id"]: r for r in csv.DictReader(f)}
    else:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()
    todo = [c for c in clips if c.stem not in rows]
    with tempfile.TemporaryDirectory() as tmp:
        for mp4 in progress(todo) if progress else todo:
            row = {"clip_id": mp4.stem, "speaker": mp4.parent.name,
                   **check_file(mp4, cfg, face_factory, scorer, Path(tmp))}  # fmt: skip
            rows[mp4.stem] = row
            with open(csv_path, "a", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, FIELDS, extrasaction="ignore").writerow(row)

    present = {c.stem: c for c in clips}
    current = [r for cid, r in rows.items() if cid in present]
    bad_sources = source_rule(current, cfg)
    failing = {r["clip_id"]: ("source_low_sync_rate" if r["speaker"] in bad_sources and r["verdict"] == "ok"
                              else r["verdict"])
               for r in current if r["verdict"] != "ok" or r["speaker"] in bad_sources}  # fmt: skip
    summary = {"clips": len(current), "pass": len(current) - len(failing), "fail": len(failing),
               "videos_dropped": len(bad_sources), "reasons": dict(Counter(failing.values()).most_common()),
               "applied": apply}  # fmt: skip
    if apply and failing:
        _apply(out, present, failing, rows)
    return summary


def _apply(out: Path, present: dict[str, Path], failing: dict[str, str], rows: dict[str, dict]) -> None:
    for cid in failing:
        mp4 = present[cid]
        dst = out / "excluded_sync" / mp4.parent.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in (mp4, mp4.with_suffix(".txt")):
            if f.exists():
                shutil.move(str(f), str(dst / f.name))
    for folder in {present[cid].parent for cid in failing}:
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    mpath = out / "manifest.csv"
    if not mpath.exists():
        return
    mrows = Manifest(mpath).rows()
    for r in mrows:
        res = rows.get(r.get("clip_id", ""))
        if not res:
            continue
        r["sync_conf"], r["syncnet_offset"] = res.get("sync_conf", ""), res.get("syncnet_offset", "")
        if r["clip_id"] in failing and r.get("status") == "accepted":
            r["status"], r["reason"] = "rejected", f"resync_{failing[r['clip_id']]}"
    tmp = out / "manifest.tmp.csv"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(mrows)
    tmp.replace(mpath)
