"""Copy a balanced, finished subset of a dataset (e.g. 2 hours for a teammate) while mining keeps running.

- Only clips that are complete: listed as accepted in manifest.csv, both .mp4 and .txt present, and the .mp4
  not modified in the last few minutes (so a clip being written right now is never copied half-done).
- Balanced over people: clips are taken from every person up to the same cap ("water filling"), so one
  prolific speaker can't fill the snapshot. Without face identity yet, the person is approximated by the
  channel (``<channel>_<videoid>`` speaker names from `pvdata auto`) or the speaker name (manual runs).
- Clips already handed over before can be excluded (``exclude``: an earlier snapshot's manifest.csv).

The snapshot is a normal dataset folder (raw_videos/, manifest.csv, speakers.json, report.json), so
`pvdata identify`, `pvdata split` and `pvdata validate` work on it directly. File names stay the same as in
the source dataset, so every clip can be traced back.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import time
from pathlib import Path

from pvdata.export import Manifest, build_report, write_json

AUTO_NAME = re.compile(r"^(.*)_[\w-]{11}$")  # <channel slug>_<YouTube video id>


def person_key(speaker_name: str) -> str:
    m = AUTO_NAME.match(speaker_name or "")
    return m.group(1) if m and m.group(1) not in ("", "yt") else (speaker_name or "?")


def water_fill(totals: dict[str, float], target: float) -> float:
    """The per-group cap c such that sum(min(total, c)) is about ``target``."""
    if sum(totals.values()) <= target:
        return max(totals.values(), default=0.0)
    lo, hi = 0.0, max(totals.values())
    for _ in range(50):
        mid = (lo + hi) / 2
        if sum(min(t, mid) for t in totals.values()) < target:
            lo = mid
        else:
            hi = mid
    return hi


def snapshot(src: str | Path, dst: str | Path, hours: float, seed: int = 0, settle_s: float = 300,
             exclude: str | Path | None = None) -> dict:  # fmt: skip
    src, dst = Path(src), Path(dst)
    if dst.resolve() == src.resolve():
        raise ValueError("the snapshot folder must be different from the dataset folder")
    names = json.loads((src / "speakers.json").read_text(encoding="utf-8")) if (src / "speakers.json").exists() else {}
    skip = set()
    if exclude:
        ex = Path(exclude)
        ex = ex / "manifest.csv" if ex.is_dir() else ex
        skip = {r["clip_id"] for r in Manifest(ex).rows() if r.get("clip_id")}
    now = time.time()
    rows = []
    for r in Manifest(src / "manifest.csv").rows():
        cid, spk = r.get("clip_id"), r.get("speaker")
        if r.get("status") != "accepted" or not cid or cid in skip:
            continue
        mp4 = src / "raw_videos" / spk / f"{cid}.mp4"
        if not (mp4.exists() and mp4.with_suffix(".txt").exists()) or now - mp4.stat().st_mtime < settle_s:
            continue
        rows.append(r)

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(person_key(names.get(r["speaker"], r.get("speaker_name", ""))), []).append(r)
    totals = {g: sum(float(r["duration"]) for r in rs) for g, rs in groups.items()}
    cap = water_fill(totals, hours * 3600)
    rng = random.Random(seed)
    chosen = []
    for g in sorted(groups):
        rs = groups[g][:]
        rng.shuffle(rs)  # spread over the person's videos and over time
        got = 0.0
        for r in rs:
            if got >= cap:
                break
            chosen.append(r)
            got += float(r["duration"])

    out = Manifest(dst / "manifest.csv")
    have = {r["clip_id"] for r in out.rows()}
    for r in chosen:
        if r["clip_id"] in have:
            continue
        for ext in (".mp4", ".txt"):
            a = src / "raw_videos" / r["speaker"] / f"{r['clip_id']}{ext}"
            b = dst / "raw_videos" / r["speaker"] / a.name
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(a, b)
        out.add(r)
    used = {r["speaker"] for r in out.rows()}
    write_json(dst / "speakers.json", {k: v for k, v in names.items() if k in used})
    report = build_report(out.rows())
    write_json(dst / "report.json", report)
    return {"people": len({person_key(names.get(s, s)) for s in used}), "speaker_folders": len(used),
            "clips": report["accepted"], "hours": report["accepted_hours"], "cap_minutes_per_person": round(cap / 60, 1),
            "eligible_hours": round(sum(totals.values()) / 3600, 2)}  # fmt: skip
