"""pvdata snapshot: balanced, finished clips only, excludes earlier handovers, traceable names."""

import json
import os
import time

from pvdata.export import MANIFEST_FIELDS, Manifest
from pvdata.snapshot import person_key, snapshot, water_fill


def make_dataset(root, layout, recent=()):
    """layout: speaker folder -> (speaker name, number of 2-s clips)."""
    m = Manifest(root / "manifest.csv")
    names = {}
    old = time.time() - 3600
    for spk, (name, n) in layout.items():
        names[spk] = name
        for i in range(1, n + 1):
            cid = f"{spk}_{i:03d}"
            mp4 = root / "raw_videos" / spk / f"{cid}.mp4"
            mp4.parent.mkdir(parents=True, exist_ok=True)
            mp4.write_bytes(b"v")
            mp4.with_suffix(".txt").write_text("नमस्ते भारत", encoding="utf-8")
            if cid not in recent:
                os.utime(mp4, (old, old))
            m.add({k: "" for k in MANIFEST_FIELDS} | {"clip_id": cid, "speaker": spk, "speaker_name": name,
                                                     "duration": "2.0", "status": "accepted"})  # fmt: skip
        m.add({k: "" for k in MANIFEST_FIELDS} | {"speaker": spk, "status": "rejected", "reason": "face_no_face"})
    (root / "speakers.json").write_text(json.dumps(names), encoding="utf-8")


def test_person_key_groups_a_channels_videos():
    assert person_key("ravish_kumar_official_dQw4w9WgXcQ") == "ravish_kumar_official"
    assert person_key("punya_prasun") == "punya_prasun"
    assert person_key("yt_dQw4w9WgXcQ") == "yt_dQw4w9WgXcQ"  # channel name had no latin letters: keep per video


def test_water_fill():
    assert abs(water_fill({"a": 100, "b": 10, "c": 10}, 60) - 40) < 0.01
    assert water_fill({"a": 5, "b": 5}, 60) == 5


def test_snapshot_is_balanced_finished_and_excludes_earlier(tmp_path):
    src = tmp_path / "dataset"
    layout = {"spk01": ("punya_prasun", 600), "spk02": ("ravish", 40), "spk03": ("chan_x_aaaaaaaaaaa", 30),
              "spk04": ("chan_x_bbbbbbbbbbb", 30), "spk05": ("meera", 25)}  # fmt: skip
    make_dataset(src, layout, recent={"spk05_025.mp4"[:-4]})
    (src / "raw_videos" / "spk02" / "spk02_040.txt").unlink()  # half-written clip: no transcript yet

    s = snapshot(src, tmp_path / "v1", hours=0.1)  # 360 s = 180 clips of 2 s
    rows = Manifest(tmp_path / "v1" / "manifest.csv").rows()
    per = {}
    for r in rows:
        per[r["speaker"]] = per.get(r["speaker"], 0) + 1
    assert s["people"] == 4 and abs(s["hours"] - 0.1) < 0.01
    assert per["spk01"] < 100  # punya is capped, not 600 clips
    assert per["spk05"] == 24  # meera: all her finished clips (one is still being written)
    assert per["spk03"] + per["spk04"] <= per["spk01"] + 1  # one channel = one person's share
    ids = {r["clip_id"] for r in rows}
    assert "spk02_040" not in ids and "spk05_025" not in ids
    assert all((tmp_path / "v1" / "raw_videos" / r["speaker"] / f"{r['clip_id']}.mp4").exists() for r in rows)
    assert set(json.loads((tmp_path / "v1" / "speakers.json").read_text(encoding="utf-8"))) == set(per)

    s2 = snapshot(src, tmp_path / "v2", hours=0.1, exclude=tmp_path / "v1")
    ids2 = {r["clip_id"] for r in Manifest(tmp_path / "v2" / "manifest.csv").rows()}
    assert ids2 and not ids & ids2 and s2["clips"] == len(ids2)


def test_gold_snapshot_keeps_only_the_best_clips(tmp_path):
    src = tmp_path / "dataset"
    make_dataset(src, {"spk01": ("anchor_a", 20), "spk02": ("anchor_b", 20)})
    rows = Manifest(src / "manifest.csv").rows()
    for i, r in enumerate(rows):  # half the clips are borderline (weak sync), one is turned away
        good = i % 2 == 0
        r.update(face_ratio="1.0", yaw="-30" if i == 4 else "3", pitch="2", sync_corr="0.6" if good else "0.18",
                 sync_offset="0", asr_min_prob="0.9", text="आज हम बात करेंगे")  # fmt: skip
    import csv

    with open(src / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    from pvdata.snapshot import GOLD

    s = snapshot(src, tmp_path / "gold", hours=1.0, quality_min=GOLD, best_first=True)
    kept = Manifest(tmp_path / "gold" / "manifest.csv").rows()
    assert kept and all(float(r["sync_corr"]) >= 0.3 and abs(float(r["yaw"])) <= 15 for r in kept)
    assert s["clips"] == len(kept) == 19  # 20 strong-sync clips minus the one turned away
