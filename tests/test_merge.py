"""pvdata merge: clips from another PC get new speaker numbers, duplicates are skipped, state comes along."""

import json
import os
import time

import pytest

from pvdata.export import MANIFEST_FIELDS, Manifest
from pvdata.merge import merge


def make(root, videos, state_videos=None):
    """videos: spk -> (speaker name, video id, accepted clips, rejected rows)."""
    m = Manifest(root / "manifest.csv")
    names = {}
    for spk, (name, vid, n_ok, n_bad) in videos.items():
        names[spk] = name
        for i in range(1, n_ok + 1):
            cid = f"{spk}_{i:03d}"
            mp4 = root / "raw_videos" / spk / f"{cid}.mp4"
            mp4.parent.mkdir(parents=True, exist_ok=True)
            mp4.write_bytes(f"{vid}-{i}".encode())
            mp4.with_suffix(".txt").write_text("नमस्ते भारत", encoding="utf-8")
            m.add(
                {k: "" for k in MANIFEST_FIELDS}
                | {
                    "clip_id": cid,
                    "speaker": spk,
                    "speaker_name": name,
                    "source_id": vid,
                    "duration": "2.0",
                    "status": "accepted",
                }
            )
        for _ in range(n_bad):
            m.add(
                {k: "" for k in MANIFEST_FIELDS}
                | {
                    "speaker": spk,
                    "speaker_name": name,
                    "source_id": vid,
                    "status": "rejected",
                    "reason": "face_no_face",
                }
            )
    (root / "speakers.json").write_text(json.dumps(names), encoding="utf-8")
    if state_videos is not None:
        (root / "auto").mkdir(parents=True, exist_ok=True)
        (root / "auto" / "state.json").write_text(json.dumps(state_videos), encoding="utf-8")
    old = time.time() - 3600
    os.utime(root / "manifest.csv", (old, old))


def test_merge_renumbers_skips_duplicates_and_is_idempotent(tmp_path):
    main, pc2 = tmp_path / "dataset", tmp_path / "dataset_pc2"
    make(
        main,
        {"spk01": ("chanA_vid1", "vid1", 3, 1), "spk02": ("chanB_vid2", "vid2", 2, 0)},
        {
            "videos": {"vid1": {"status": "mined", "channel_id": "A"}, "vid2": {"status": "mined", "channel_id": "B"}},
            "channels": {"A": {"name": "chanA", "mined": 1, "accepted_s": 6.0, "expanded": True}},
            "queries": {"q1": 20},
        },
    )
    make(
        pc2,
        {
            "spk01": ("chanC_vid3", "vid3", 4, 2),  # new video
            "spk02": ("chanA_vid1", "vid1", 3, 1),  # mined on both PCs: skip
            "spk03": ("chanA_vid5", "vid5", 1, 0),
        },  # same channel, new video
        {
            "videos": {
                "vid3": {"status": "mined", "speaker": "spk01", "channel_id": "C"},
                "vid1": {"status": "mined", "speaker": "spk02", "channel_id": "A"},
                "vid5": {"status": "mined", "speaker": "spk03", "channel_id": "A"},
                "vid9": {"status": "rejected", "reason": "screen_no_face"},
                "vid8": {"status": "ok"},
            },
            "channels": {
                "A": {"name": "chanA", "mined": 2, "accepted_s": 8.0, "expanded": False, "blocked": True},
                "C": {"name": "chanC", "mined": 1, "accepted_s": 8.0},
            },
            "queries": {"q1": 10, "q2": 30},
        },
    )

    s = merge(pc2, main)
    assert s["speaker_folders"] == 2 and s["clips"] == 5 and s["skipped_videos"] == 1
    names = json.loads((main / "speakers.json").read_text(encoding="utf-8"))
    assert names["spk03"] == "chanC_vid3" and names["spk04"] == "chanA_vid5"
    assert sorted(p.name for p in (main / "raw_videos" / "spk03").glob("*.mp4")) == [
        f"spk03_00{i}.mp4" for i in range(1, 5)
    ]
    assert (main / "raw_videos" / "spk04" / "spk04_001.txt").exists()
    rows = Manifest(main / "manifest.csv").rows()
    assert {r["clip_id"] for r in rows if r["speaker"] == "spk03" and r["status"] == "accepted"} == {
        f"spk03_00{i}" for i in range(1, 5)
    }
    assert (
        sum(r["status"] == "rejected" and r["speaker"] == "spk03" for r in rows) == 2
    )  # rejections kept for the report
    assert sum(r["source_id"] == "vid1" for r in rows) == 4  # the duplicate video added nothing

    state = json.loads((main / "auto" / "state.json").read_text(encoding="utf-8"))
    assert state["videos"]["vid3"]["speaker"] == "spk03" and state["videos"]["vid5"]["speaker"] == "spk04"
    assert state["videos"]["vid9"]["status"] == "rejected" and "vid8" not in state["videos"]
    assert state["channels"]["A"]["mined"] == 3 and state["channels"]["A"]["blocked"] is True
    assert state["queries"] == {"q1": 20, "q2": 30}
    report = json.loads((main / "report.json").read_text(encoding="utf-8"))
    assert report["accepted"] == 10

    old = time.time() - 3600
    os.utime(main / "manifest.csv", (old, old))
    again = merge(pc2, main)
    assert again["clips"] == 0 and again["speaker_folders"] == 0
    assert len(Manifest(main / "manifest.csv").rows()) == len(rows)


def test_merge_refuses_while_the_main_miner_is_writing(tmp_path):
    main, pc2 = tmp_path / "dataset", tmp_path / "pc2"
    make(main, {"spk01": ("a_v1", "v1", 1, 0)})
    make(pc2, {"spk01": ("b_v2", "v2", 1, 0)})
    os.utime(main / "manifest.csv", None)  # just written
    with pytest.raises(RuntimeError, match="stop `pvdata auto`"):
        merge(pc2, main)
    assert merge(pc2, main, force=True)["clips"] == 1


def test_clip_moved_away_on_the_other_pc_is_not_counted(tmp_path):
    main, pc2 = tmp_path / "dataset", tmp_path / "pc2"
    make(main, {"spk01": ("a_v1", "v1", 1, 0)})
    make(pc2, {"spk01": ("b_v2", "v2", 2, 0)})
    (pc2 / "raw_videos" / "spk01" / "spk01_002.mp4").unlink()
    s = merge(pc2, main)
    assert s["clips"] == 1 and s["missing_files"] == 1
    rows = [r for r in Manifest(main / "manifest.csv").rows() if r["speaker"] == "spk02"]
    assert sorted(r["status"] for r in rows) == ["accepted", "rejected"]
