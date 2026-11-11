"""Frame rate without stutter, the trainer's per-clip CTC rule, and the review / exclude tools."""

import json
import shutil
import subprocess

import pytest

from pvdata import media
from pvdata.auto import State
from pvdata.config import Config
from pvdata.ctc import ctc_steps_for, fits_ctc
from pvdata.export import Manifest, build_report
from pvdata.pipeline import Pipeline
from pvdata.sources import Source

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def test_smooth_fps_never_drops_frames_unevenly():
    assert [media.smooth_fps(f) for f in (24, 25, 29.97, 30, 50, 59.94, 60, 15)] == [24, 25, 30, 30, 25, 30, 30, 25]


def test_ctc_steps_match_the_trainer():
    assert ctc_steps_for(2.0, 25) == 100  # 50 frames -> ceil(50/2) * 4
    assert ctc_steps_for(3.0, 25) == 152  # 75 frames -> 38 * 4, as in docs/ARCHITECTURE.md
    long_sentence = "आज हम बात करेंगे देश की राजनीति और अर्थव्यवस्था के बारे में"
    assert not fits_ctc(long_sentence, 32)[0]  # the old fixed 16-frame rule rejected this
    assert fits_ctc(long_sentence, ctc_steps_for(3.0, 25))[0]  # the trainer accepts it in a 3 s clip


@needs_ffmpeg
def test_30_fps_source_gives_30_fps_clips(tmp_path):
    from tests.test_pipeline_e2e import BrightnessFace, FakeASR, make_video

    base = make_video(tmp_path / "base.mp4")
    v30 = tmp_path / "talk30.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(base), "-vf", "fps=30", "-c:a", "copy", str(v30)],
                   check=True)  # fmt: skip
    pipe = Pipeline(Config(), tmp_path / "out", tmp_path / "work", FakeASR(["नमस्ते भारत आज मौसम"]), BrightnessFace)
    pipe.run([Source(str(v30), "ravi")])
    clips = sorted((tmp_path / "out" / "raw_videos" / "spk01").glob("*.mp4"))
    assert clips and round(media.probe(clips[0])["fps"]) == 30


@needs_ffmpeg
def test_review_page_and_exclude_block_the_channel(tmp_path):
    from pvdata.review import exclude, make_review
    from tests.test_pipeline_e2e import BrightnessFace, FakeASR, make_video

    out = tmp_path / "dataset"
    video = make_video(tmp_path / "talk.mp4")
    for name in ("good", "cartoon"):
        shutil.copy(video, tmp_path / f"{name}.mp4")
    pipe = Pipeline(Config(), out, tmp_path / "work", FakeASR(["नमस्ते भारत आज मौसम", "पानी दो मुझे अभी"]),
                    BrightnessFace)  # fmt: skip
    pipe.run([Source(str(tmp_path / "good.mp4"), "chan_good_aaaaaaaaaaa"),
              Source(str(tmp_path / "cartoon.mp4"), "chan_toon_bbbbbbbbbbb")])  # fmt: skip
    hours_before = build_report(Manifest(out / "manifest.csv").rows())["accepted_hours"]

    st = State(out / "auto" / "state.json")  # the autopilot's memory of these two videos
    st.videos = {"a": {"id": "a", "channel_id": "UCgood", "status": "mined", "speaker": "spk01"},
                 "b": {"id": "b", "channel_id": "UCtoon", "status": "mined", "speaker": "spk02"},
                 "c": {"id": "c", "channel_id": "UCtoon", "status": "ok"},
                 "d": {"id": "d", "channel_id": "UCtoon", "status": "new"}}  # fmt: skip
    st.save()

    page = make_review(out)
    text = page.read_text(encoding="utf-8")
    assert "spk01" in text and "spk02" in text and (out / "review" / "thumbs" / "spk02.jpg").stat().st_size > 1000

    res = exclude(out, ["spk02"])
    assert res == {"folders_moved": 1, "clips_marked": res["clips_marked"], "channels_blocked": 1}
    assert res["clips_marked"] > 0
    assert not (out / "raw_videos" / "spk02").exists() and any((out / "excluded" / "spk02").glob("*.mp4"))
    after = build_report(Manifest(out / "manifest.csv").rows())
    assert after["accepted_hours"] < hours_before and "spk02" not in after["clips_per_speaker"]
    state = json.loads((out / "auto" / "state.json").read_text(encoding="utf-8"))
    assert state["channels"]["UCtoon"]["blocked"] and not state["channels"].get("UCgood", {}).get("blocked")
    assert state["videos"]["c"]["reason"] == state["videos"]["d"]["reason"] == "channel_blocked"
    assert state["videos"]["b"]["status"] == "mined"
