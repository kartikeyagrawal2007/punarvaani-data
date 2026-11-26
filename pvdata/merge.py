"""Merge a dataset mined on another PC into the main dataset.

    pvdata merge --from D:/dataset_pc2 --into dataset

Each lab PC runs ``pvdata auto`` into its own folder with its own seed searches. Every PC numbers its speaker
folders from spk01, so the clips of each merged video move into a new speaker folder (the next free number)
and their clip ids follow. A video that the main dataset already has (mined on two PCs) is skipped, so merging
the same folder twice adds nothing. The other PC's autopilot state comes along too: the main PC will not mine
those videos again, and channel counts add up so channel caps still hold.

Stop ``pvdata auto`` on the main dataset first: it keeps its state in memory and would overwrite the merged
state when it next saves. Afterwards run ``pvdata identify`` (the same person mined on two PCs becomes one
person) and ``pvdata split``.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from pvdata.export import Manifest, build_report, write_json

BUSY_SECONDS = 120


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _next_speaker_number(out: Path, names: dict) -> int:
    used = [int(s[3:]) for s in names if s[3:].isdigit()]
    raw = out / "raw_videos"
    if raw.exists():
        used += [int(p.name[3:]) for p in raw.iterdir() if p.is_dir() and p.name[3:].isdigit()]
    return max(used, default=0) + 1


def _known_videos(rows: list[dict], state: dict) -> set[str]:
    known = {r["source_id"] for r in rows if r.get("source_id")}
    known |= {vid for vid, v in state.get("videos", {}).items() if v.get("status") in ("mined", "mining", "failed")}
    return known


def merge(src: str | Path, dst: str | Path, force: bool = False) -> dict:
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("--from and --into are the same folder")
    if not (src / "manifest.csv").exists():
        raise FileNotFoundError(f"no manifest.csv in {src}: is this a pvdata dataset folder?")
    dst_manifest = dst / "manifest.csv"
    if not force and dst_manifest.exists() and time.time() - dst_manifest.stat().st_mtime < BUSY_SECONDS:
        raise RuntimeError(f"{dst_manifest} changed in the last {BUSY_SECONDS} s: stop `pvdata auto` on {dst} first "
                           "(Ctrl+C), wait 2 minutes, then merge (or pass --force)")  # fmt: skip

    out_manifest = Manifest(dst_manifest)
    dst_rows = out_manifest.rows()
    dst_names = _load_json(dst / "speakers.json")  # spkNN -> speaker name
    dst_state = _load_json(dst / "auto" / "state.json")
    src_rows = Manifest(src / "manifest.csv").rows()
    src_names = _load_json(src / "speakers.json")
    src_state = _load_json(src / "auto" / "state.json")

    known = _known_videos(dst_rows, dst_state)
    known_names = set(dst_names.values())

    # Group the other PC's rows by speaker folder; every auto video is its own folder.
    by_speaker: dict[str, list[dict]] = {}
    for r in src_rows:
        if r.get("speaker"):
            by_speaker.setdefault(r["speaker"], []).append(r)

    n = _next_speaker_number(dst, dst_names)
    spk_map: dict[str, str] = {}
    added_videos: set[str] = set()
    stats = {"speaker_folders": 0, "clips": 0, "seconds": 0.0, "skipped_videos": 0, "missing_files": 0}
    for old_spk in sorted(by_speaker, key=lambda s: (len(s), s)):
        rows = by_speaker[old_spk]
        name = src_names.get(old_spk) or next((r["speaker_name"] for r in rows if r.get("speaker_name")), old_spk)
        videos = {r["source_id"] for r in rows if r.get("source_id")}
        if name in known_names or (videos and videos <= known):
            stats["skipped_videos"] += max(1, len(videos))
            continue
        new_spk = f"spk{n:02d}"
        n += 1
        spk_map[old_spk] = new_spk
        dst_names[new_spk] = name
        stats["speaker_folders"] += 1
        for r in rows:
            row = dict(r, speaker=new_spk)
            cid = r.get("clip_id") or ""
            if r.get("status") == "accepted" and cid:
                new_cid = new_spk + cid[len(old_spk) :] if cid.startswith(old_spk + "_") else f"{new_spk}_{cid}"
                files = [src / "raw_videos" / old_spk / f"{cid}{ext}" for ext in (".mp4", ".txt")]
                if all(f.exists() for f in files):
                    for f in files:
                        target = dst / "raw_videos" / new_spk / f"{new_cid}{f.suffix}"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, target)
                    row["clip_id"] = new_cid
                    stats["clips"] += 1
                    stats["seconds"] += float(r.get("duration") or 0)
                else:  # moved away on the other PC (resync / exclude): keep the row, but not as accepted
                    row.update(clip_id="", status="rejected", reason="merge_missing_file")
                    stats["missing_files"] += 1
            out_manifest.add(row)
        added_videos |= videos

    write_json(dst / "speakers.json", dst_names)
    _merge_state(dst_state, src_state, added_videos, spk_map)
    if dst_state:
        (dst / "auto").mkdir(parents=True, exist_ok=True)
        tmp = dst / "auto" / "state.tmp"
        tmp.write_text(json.dumps(dst_state, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(dst / "auto" / "state.json")

    log = _load_json(dst / "merged.json")
    log.setdefault(str(src.resolve()), []).append({"at": time.strftime("%Y-%m-%d %H:%M"), **stats})
    write_json(dst / "merged.json", log)
    report = build_report(Manifest(dst_manifest).rows())
    write_json(dst / "report.json", report)
    stats["hours"] = round(stats.pop("seconds") / 3600, 3)
    stats["dataset_hours_now"] = report["accepted_hours"]
    return stats


def _merge_state(dst: dict, src: dict, added: set[str], spk_map: dict[str, str]) -> None:
    """Bring the other PC's decisions over so the main autopilot skips those videos and channel caps add up."""
    if not src:
        return
    videos = dst.setdefault("videos", {})
    for vid, v in src.get("videos", {}).items():
        if vid in videos:
            continue
        if vid in added:
            videos[vid] = dict(v, speaker=spk_map.get(v.get("speaker", ""), v.get("speaker", "")))
        elif v.get("status") == "rejected":  # screened out there: no need to screen it again here
            videos[vid] = dict(v)
    channels = dst.setdefault("channels", {})
    for cid, c in src.get("channels", {}).items():
        if cid not in channels:
            channels[cid] = dict(c)
            continue
        d = channels[cid]
        d["mined"] = d.get("mined", 0) + c.get("mined", 0)
        d["accepted_s"] = d.get("accepted_s", 0.0) + c.get("accepted_s", 0.0)
        for flag in ("expanded", "blocked"):
            if c.get(flag):
                d[flag] = c[flag]
        if not d.get("name"):
            d["name"] = c.get("name", "")
    queries = dst.setdefault("queries", {})
    for q, pool in src.get("queries", {}).items():
        queries[q] = max(queries.get(q, 0), pool)
