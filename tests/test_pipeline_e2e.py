"""End to end on a synthetic video: no YouTube, no real ASR, no real face needed.

The video's brightness and the audio's loudness follow the same "speech" envelope, and a fake face
detector reads mouth opening from brightness, so the sync check behaves as on a real talking head.
"""

import shutil
import subprocess
import wave

import cv2
import numpy as np
import pytest

from pvdata.asr import Word
from pvdata.config import Config
from pvdata.export import speaker_splits
from pvdata.face import FaceObs
from pvdata.pipeline import Pipeline
from pvdata.sources import Source
from pvdata.validate import validate

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

FPS, SR, SECONDS = 25, 16000, 12.0


def envelope(t):
    return 0.1 + 0.9 * np.clip(np.sin(2 * np.pi * 1.7 * t), 0, None)


def make_video(path, height=720):
    width = height * 16 // 9
    n = int(SECONDS * FPS)
    silent = path.with_suffix(".silent.mp4")
    vw = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    for i in range(n):
        level = int(40 + 200 * envelope(i / FPS))
        vw.write(np.full((height, width, 3), level, np.uint8))
    vw.release()
    t = np.arange(int(SECONDS * SR)) / SR
    audio = (0.5 * envelope(t) * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    wav = path.with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes(audio.tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)  # fmt: skip
    return path


class FakeASR:
    """Words every 0.4 s; the text of each group decides what the checks should do with it."""

    def __init__(self, texts):
        self.texts = texts

    def transcribe(self, wav):
        words, t = [], 0.3
        for group in self.texts:
            for tok in group.split():
                words.append(Word(tok, t, t + 0.35, 0.9))
                t += 0.4
            t += 0.6  # pause between groups -> separate clips
        return words


class BrightnessFace:
    def __call__(self, frame):
        return FaceObs(1, yaw=1.0, pitch=0.0, face_h_frac=0.4, mouth_w_px=150, opening=float(frame.mean()) / 255)


def run(tmp_path, texts, height=720):
    video = make_video(tmp_path / "talk.mp4", height)
    cfg = Config()
    pipe = Pipeline(cfg, tmp_path / "out", tmp_path / "work", FakeASR(texts), BrightnessFace)
    report = pipe.run([Source(str(video), "ravi")])
    return report, pipe


def test_end_to_end_writes_team_guide_layout(tmp_path):
    texts = ["नमस्ते भारत आज मौसम", "School bus में बच्चे", "हम ५ लोग साथ हैं", "पानी दो मुझे अभी"]
    report, pipe = run(tmp_path, texts)
    out = tmp_path / "out"
    clips = sorted((out / "raw_videos" / "spk01").glob("*.mp4"))
    assert report["accepted"] == len(clips) >= 2
    assert clips[0].name == "spk01_001.mp4"
    for mp4 in clips:
        txt = mp4.with_suffix(".txt").read_text(encoding="utf-8").strip()
        assert txt and all(("ऀ" <= ch <= "ॿ") or ch == " " for ch in txt)
    rows = pipe.manifest.rows()
    reasons = {r["reason"] for r in rows if r["status"] == "rejected"}
    assert "text_contains_latin" in reasons  # the English-word group is dropped
    texts_out = {r["text"] for r in rows if r["status"] == "accepted"}
    assert any("पाँच" in t for t in texts_out)  # digit spelled out
    assert validate(out / "raw_videos").ok
    assert (out / "report.json").exists() and (out / "speakers.json").exists()


def test_rerun_skips_finished_sources(tmp_path):
    texts = ["नमस्ते भारत आज मौसम", "पानी दो मुझे अभी"]
    first, _ = run(tmp_path, texts)
    second, _ = run(tmp_path, texts)
    assert second["accepted"] == first["accepted"]
    assert len(list((tmp_path / "out" / "raw_videos" / "spk01").glob("*.mp4"))) == first["accepted"]


def test_low_resolution_source_is_rejected(tmp_path):
    report, pipe = run(tmp_path, ["नमस्ते भारत आज मौसम"], height=360)
    assert report["accepted"] == 0
    assert pipe.manifest.rows()[0]["reason"] == "source_below_720p"


def test_speaker_disjoint_split():
    s = speaker_splits([f"spk{i:02d}" for i in range(1, 11)], 0.1, 0.1, seed=0)
    all_ = s["train"] + s["val"] + s["test"]
    assert len(all_) == len(set(all_)) == 10 and s["val"] and s["test"]


def test_interrupted_source_is_redone_without_duplicates(tmp_path):
    texts = ["नमस्ते भारत आज मौसम", "पानी दो मुझे अभी"]
    first, pipe = run(tmp_path, texts)
    for p in (tmp_path / "work" / "done").glob("*.json"):
        p.unlink()  # as if the run had been stopped before this source finished
    second, pipe = run(tmp_path, texts)
    clips = list((tmp_path / "out" / "raw_videos" / "spk01").glob("*.mp4"))
    assert second["accepted"] == first["accepted"] == len(clips)
    assert len(pipe.manifest.rows()) == first["candidates"]


def test_parallel_judging_matches_serial(tmp_path):
    texts = ["नमस्ते भारत आज मौसम", "School bus में बच्चे", "हम ५ लोग साथ हैं", "पानी दो मुझे अभी"]
    video = make_video(tmp_path / "talk.mp4")
    rows = {}
    for workers in (1, 4):
        out = tmp_path / f"out{workers}"
        pipe = Pipeline(Config(), out, out / "work", FakeASR(texts), BrightnessFace, workers=workers)
        pipe.run([Source(str(video), "ravi")])
        rows[workers] = [(r["start"], r["status"], r["reason"], r["clip_id"]) for r in pipe.manifest.rows()]
    assert rows[1] == rows[4]


def test_split_by_clips_keeps_a_dominant_speaker_in_train():
    counts = [499, 149, 52, 1425, 58, 34, 23, 45, 123, 13, 4, 20]  # the first real run: spk04 has 58% of clips
    clips = {f"spk{i:02d}": c for i, c in enumerate(counts, 1)}
    total = sum(counts)
    for seed in range(20):
        s = speaker_splits(list(clips), 0.1, 0.1, seed, clips)
        assert sorted(s["train"] + s["val"] + s["test"]) == sorted(clips)
        assert "spk01" in s["train"] and "spk04" in s["train"] and len(s["train"]) == 10
        for split in ("val", "test"):
            assert 0.02 * total <= sum(clips[x] for x in s[split]) <= 0.15 * total


def test_youtube_rows_without_licence_are_youtube_standard():
    from pvdata.export import license_of

    assert license_of({"license": "", "source_url": "https://www.youtube.com/watch?v=x"}) == "youtube_standard"
    assert license_of({"license": "Creative Commons Attribution license (reuse allowed)"}).startswith("Creative")
    assert license_of({"source_url": "C:/videos/local.mp4"}) == "unknown"
