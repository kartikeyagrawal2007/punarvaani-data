"""Human check at the right level: one thumbnail strip per speaker folder (= per video), not per clip.

Automatic checks can't tell a real person from an animated avatar, a cartoon or a dubbed puppet: they all
have "one frontal face whose lips move with the speech". Looking at ~400 strips takes minutes.

    pvdata review  --out dataset           -> dataset/review/index.html (tick the bad ones, copy the command)
    pvdata exclude --out dataset spk385 …  -> clips moved to dataset/excluded/, manifest rows marked
                                              manual_exclude, and the video's channel blocked in the autopilot

Stop `pvdata auto` before `pvdata exclude` (both write the manifest and the autopilot state), then restart it.
"""

from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path

from pvdata.export import MANIFEST_FIELDS, Manifest


def _strip(clips: list[Path], dst: Path, n: int = 3, width: int = 220) -> bool:
    import cv2
    import numpy as np

    picks = [clips[round(i * (len(clips) - 1) / max(n - 1, 1))] for i in range(min(n, len(clips)))]
    tiles = []
    for clip in picks:
        cap = cv2.VideoCapture(str(clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2))
        ok, frame = cap.read()
        cap.release()
        if ok:
            h = round(frame.shape[0] * width / frame.shape[1])
            tiles.append(cv2.resize(frame, (width, h)))
    if not tiles:
        return False
    h = min(t.shape[0] for t in tiles)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), np.hstack([t[:h] for t in tiles]), [cv2.IMWRITE_JPEG_QUALITY, 80]))


def make_review(out: str | Path, refresh: bool = False) -> Path:
    out = Path(out)
    names = json.loads((out / "speakers.json").read_text(encoding="utf-8")) if (out / "speakers.json").exists() else {}
    folders = sorted((p for p in (out / "raw_videos").iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name))
    cards = []
    for f in folders:
        clips = sorted(f.glob("*.mp4"))
        if not clips:
            continue
        thumb = out / "review" / "thumbs" / f"{f.name}.jpg"
        if refresh or not thumb.exists():
            _strip(clips, thumb)
        name = html.escape(names.get(f.name, ""))
        cards.append(
            f'<label class="card"><input type="checkbox" value="{f.name}">'
            f'<img loading="lazy" src="thumbs/{f.name}.jpg" alt="{f.name}">'
            f"<b>{f.name}</b> <span>{len(clips)} clips</span><small>{name}</small></label>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Dataset review</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#f6f8f8;color:#14262c}}
h1{{font-size:20px}} p{{max-width:900px}}
#bar{{position:sticky;top:0;background:#0b3c49;color:#fff;padding:10px;border-radius:8px;z-index:2}}
#cmd{{width:100%;height:56px;font-family:monospace;font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px;margin-top:12px}}
.card{{background:#fff;border-radius:8px;padding:8px;display:block;cursor:pointer;border:3px solid transparent}}
.card:has(input:checked){{border-color:#e8833a;background:#fff4ea}}
.card img{{width:100%;display:block;border-radius:4px;margin:6px 0}}
.card small{{display:block;color:#5b6b70;word-break:break-all}} .card span{{color:#5b6b70}}
</style></head><body>
<h1>Dataset review: {len(cards)} speaker folders</h1>
<p>Each strip shows 3 clips of one video. Tick every folder that is <b>not a real person speaking Hindi to the
camera</b>: animation or avatars, cartoons, slides, wrong person, heavy filters, dubbing. Then stop
<code>pvdata auto</code>, paste the command below into PowerShell, and restart the autopilot.</p>
<div id="bar">Ticked: <span id="n">0</span><br><textarea id="cmd" readonly></textarea></div>
<div class="grid">{"".join(cards)}</div>
<script>
const out = {json.dumps(str(out))};
function upd(){{const v=[...document.querySelectorAll('input:checked')].map(e=>e.value);
document.getElementById('n').textContent=v.length;
document.getElementById('cmd').value=v.length?`pvdata exclude --out "${{out}}" ${{v.join(' ')}}`:'(tick folders above)';}}
document.querySelectorAll('input').forEach(e=>e.addEventListener('change',upd));upd();
</script></body></html>"""
    path = out / "review" / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def exclude(out: str | Path, speakers: list[str], reason: str = "manual_exclude") -> dict:
    """Move whole speaker folders out of raw_videos, mark their clips in the manifest, block their channels."""
    out = Path(out)
    want = set(speakers)
    moved = 0
    for spk in sorted(want):
        src = out / "raw_videos" / spk
        if not src.is_dir():
            continue
        dst = out / "excluded" / spk
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.move(str(f), str(dst / f.name))
        src.rmdir()
        moved += 1

    rows = Manifest(out / "manifest.csv").rows()
    clips = 0
    for r in rows:
        if r.get("speaker") in want and r.get("status") == "accepted":
            r["status"], r["reason"] = "rejected", reason
            clips += 1
    tmp = out / "manifest.tmp.csv"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out / "manifest.csv")

    blocked: set[str] = set()
    state_path = out / "auto" / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for v in state.get("videos", {}).values():
            if v.get("speaker") in want and v.get("channel_id"):
                blocked.add(v["channel_id"])
        for cid in blocked:
            state.setdefault("channels", {}).setdefault(cid, {"name": "", "mined": 0, "accepted_s": 0.0,
                                                              "expanded": False})["blocked"] = True  # fmt: skip
        for v in state.get("videos", {}).values():
            if v.get("channel_id") in blocked and v.get("status") in ("new", "ok"):
                v["status"], v["reason"] = "rejected", "channel_blocked"
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(state_path)

    log_path = out / "excluded" / "excluded.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    done.append({"speakers": sorted(want), "reason": reason, "channels_blocked": sorted(blocked)})
    log_path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"folders_moved": moved, "clips_marked": clips, "channels_blocked": len(blocked)}
