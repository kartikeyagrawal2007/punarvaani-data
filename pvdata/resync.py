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
import os
import shutil
import tempfile
import threading
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pvdata import media
from pvdata.config import Config
from pvdata.export import MANIFEST_FIELDS, Manifest
from pvdata.shots import change_scores, face_jump_cuts, find_cuts
from pvdata.syncnet import FPS

FIELDS = ["clip_id", "speaker", "frames", "cut", "face_cov", "sync_conf", "syncnet_offset", "verdict"]
SYNC_VERDICTS = {"ok", "low_syncnet_conf", "syncnet_offset"}  # clips that reached the SyncNet decision


def check_file(mp4: Path, cfg: Config, detector: Callable, scorer: Callable, tmp: Path) -> dict:
    frames = media.read_frames(mp4, 0.0, None, FPS)
    row = {"frames": len(frames), "cut": 0}
    obs = [detector(f) for f in frames]
    boxes = [o.box if o.n_faces == 1 else None for o in obs]
    if cfg.shot.enabled:
        cuts = find_cuts(change_scores(frames), cfg.shot.min_score, cfg.shot.ratio)
        if face_jump_cuts(cuts, boxes, cfg.shot.same_face_iou):
            return {**row, "cut": 1, "verdict": "shot_cut"}
    cov = sum(b is not None for b in boxes) / max(len(boxes), 1)
    row["face_cov"] = round(cov, 3)
    if cov < 0.9:
        return {**row, "verdict": "face_coverage"}
    wav = tmp / f"{mp4.stem}.wav"
    try:
        media.extract_audio(mp4, wav)
        audio = media.read_wav(wav)
    except Exception:
        return {**row, "verdict": "no_audio"}
    finally:
        wav.unlink(missing_ok=True)
    res = scorer(frames, boxes, audio)
    if res is None:
        return {**row, "verdict": "too_short"}
    row.update(sync_conf=round(res.conf, 2), syncnet_offset=res.offset)
    return {**row, "verdict": sync_verdict(res.conf, res.offset, cfg)}


def sync_verdict(conf: float, offset: int, cfg: Config) -> str:
    if conf < cfg.sync.min_conf:
        return "low_syncnet_conf"
    if abs(offset) > cfg.sync.max_syncnet_offset:
        return "syncnet_offset"
    return "ok"


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
           progress: Callable | None = None, workers: int = 0, recheck: tuple[str, ...] = ()) -> dict:  # fmt: skip
    """``workers`` clips are checked at once (default: half the CPU cores, at most 8). Each worker keeps one face
    detector for all its clips (loading MediaPipe per clip cost ~1 s); SyncNet runs on the GPU one batch at a time.

    Saved SyncNet scores are re-judged with the current thresholds every run, so changing ``sync.min_conf`` needs
    no re-scoring. ``recheck`` names verdicts (e.g. ``shot_cut``) whose clips are checked again from scratch."""
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
    for cid in [cid for cid, r in rows.items() if r["verdict"] in recheck]:
        del rows[cid]
    for r in rows.values():
        if r["verdict"] in SYNC_VERDICTS and r.get("sync_conf") not in (None, ""):
            r["verdict"] = sync_verdict(float(r["sync_conf"]), int(float(r["syncnet_offset"] or 0)), cfg)
    todo = [c for c in clips if c.stem not in rows]
    workers = workers if workers > 0 else max(1, min(8, (os.cpu_count() or 2) // 2))
    local = threading.local()
    detectors: list = []

    def one(mp4: Path, tmp: Path) -> dict:
        if not hasattr(local, "detector"):
            local.detector = face_factory()
            detectors.append(local.detector)
        try:
            res = check_file(mp4, cfg, local.detector, scorer, tmp)
        except Exception as e:  # an unreadable clip must not stop a run of hours
            res = {"verdict": f"error_{type(e).__name__}"}
        return {"clip_id": mp4.stem, "speaker": mp4.parent.name, **res}

    def in_order(pool: ThreadPoolExecutor, tmp: Path):
        """Results in clip order with only a few clips queued, so Ctrl+C stops within seconds."""
        it, pending = iter(todo), deque()
        try:
            for c in it:
                pending.append(pool.submit(one, c, tmp))
                if len(pending) >= 2 * workers:
                    break
            while pending:
                row = pending.popleft().result()
                nxt = next(it, None)
                if nxt is not None:
                    pending.append(pool.submit(one, nxt, tmp))
                yield row
        finally:  # Ctrl+C or an error: drop the queue; only the clips being checked right now finish
            for f in pending:
                f.cancel()

    errors = 0
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(workers) as pool:
        results = in_order(pool, Path(tmp))
        for row in progress(results, total=len(todo)) if progress else results:
            if row["verdict"].startswith("error_"):  # not saved, so the next run tries this clip again
                errors += 1
                continue
            rows[row["clip_id"]] = row
            with open(csv_path, "a", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, FIELDS, extrasaction="ignore").writerow(row)
    for d in detectors:
        if hasattr(d, "close"):
            d.close()

    present = {c.stem: c for c in clips}
    current = [r for cid, r in rows.items() if cid in present]
    bad_sources = source_rule(current, cfg)
    failing = {r["clip_id"]: ("source_low_sync_rate" if r["speaker"] in bad_sources and r["verdict"] == "ok"
                              else r["verdict"])
               for r in current if r["verdict"] != "ok" or r["speaker"] in bad_sources}  # fmt: skip
    summary = {"clips": len(current), "pass": len(current) - len(failing), "fail": len(failing),
               "videos_dropped": len(bad_sources), "reasons": dict(Counter(failing.values()).most_common()),
               "unreadable_retry_next_run": errors, "applied": apply}  # fmt: skip
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
