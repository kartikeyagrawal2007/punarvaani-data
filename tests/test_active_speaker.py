"""Active-speaker checks: shot cuts, SyncNet (network, crop, offset/confidence), per-video rule, resync."""

import csv
import shutil

import numpy as np
import pytest

from pvdata.config import Config
from pvdata.export import MANIFEST_FIELDS, Manifest
from pvdata.face import FaceObs
from pvdata.pipeline import Decision, Pipeline
from pvdata.shots import change_scores, find_cuts, shot_bounds
from pvdata.syncnet import SyncNetResult, crop_faces, fill_boxes, median_filter, offset_confidence

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


# ------------------------------------------------------------------ shot cuts
def _scene(seed, n, h=72, w=128):
    import cv2

    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    base = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)  # smooth blobs, like a real scene
    frames = []
    for i in range(n):  # small camera shake: the same scene shifted by a pixel
        frames.append(np.roll(base, i % 2, axis=1))
    return frames


def test_cut_between_two_scenes_is_found():
    frames = _scene(1, 20) + _scene(2, 20)
    assert find_cuts(change_scores(frames)) == [20]


def test_motion_and_flash_are_not_cuts():
    frames = _scene(1, 40)
    flashed = [np.clip(f.astype(int) + (60 if i == 20 else 0), 0, 255).astype(np.uint8) for i, f in enumerate(frames)]
    assert find_cuts(change_scores(frames)) == []
    assert find_cuts(change_scores(flashed)) == []


def test_shot_bounds_picks_the_shot_around_the_clip():
    assert shot_bounds([10, 60], 80, 20, 50) == (10, 60)
    assert shot_bounds([], 80, 20, 50) == (0, 80)


# ------------------------------------------------------------------ SyncNet pieces
def test_offset_confidence_finds_the_shift_and_is_confident():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(60, 32))
    shifted = np.roll(v, 3, axis=0) + 0.05 * rng.normal(size=v.shape)
    offset, conf, _ = offset_confidence(v, shifted)
    assert offset == -3 and conf > 2.0
    _, conf_random, _ = offset_confidence(v, rng.normal(size=v.shape))
    assert conf_random < conf / 3


def test_median_filter_keeps_edges_and_fill_boxes_uses_nearest():
    x = np.array([10.0] * 5 + [50.0] + [10.0] * 5)
    assert np.allclose(median_filter(x, 3), 10.0)
    assert np.allclose(median_filter(np.array([5.0, 5.0]), 13), 5.0)
    filled = fill_boxes([None, (0, 0, 10, 10), None, None, (5, 5, 15, 15)])
    assert filled[0].tolist() == [0, 0, 10, 10] and filled[3].tolist() == [5, 5, 15, 15]
    assert fill_boxes([None, None]) is None


def test_crop_is_224_bgr_with_grey_padding():
    frame = np.zeros((360, 640, 3), np.uint8)
    frame[...] = (255, 0, 0)  # red in RGB
    crops = crop_faces(np.stack([frame] * 3), np.array([[270, 130, 370, 230]] * 3))
    assert crops.shape == (3, 224, 224, 3)
    assert tuple(crops[0, 112, 112]) == (0, 0, 255)  # BGR
    edge = crop_faces(np.stack([frame]), np.array([[0, 0, 100, 100]]))
    assert tuple(edge[0, 2, 2]) == (110, 110, 110)


def test_network_shapes_and_weight_names_match_syncnet_python(tmp_path):
    torch = pytest.importorskip("torch")
    from pvdata.syncnet import SyncNetScorer, build_model

    model = build_model().eval()
    with torch.no_grad():
        assert model.forward_lip(torch.zeros(2, 3, 5, 224, 224)).shape == (2, 1024)
        assert model.forward_aud(torch.zeros(2, 1, 13, 20)).shape == (2, 1024)
    keys = set(model.state_dict())
    assert {"netcnnlip.0.weight", "netcnnaud.0.weight", "netfclip.3.weight", "netfcaud.3.bias"} <= keys
    path = tmp_path / "syncnet_v2.model"
    state = {k: v for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
    torch.save(state, path)
    SyncNetScorer._load(path)  # loads like the published file (old torch: no num_batches_tracked)
    torch.save({"wrong.weight": torch.zeros(1)}, path)
    with pytest.raises(ValueError):
        SyncNetScorer._load(path)


def test_scorer_runs_end_to_end_on_cpu():
    pytest.importorskip("torch")
    pytest.importorskip("python_speech_features")
    from pvdata.syncnet import SyncNetScorer, build_model

    scorer = SyncNetScorer(model=build_model(), device="cpu", batch=8)
    frames = np.random.default_rng(0).integers(0, 255, (20, 120, 160, 3), dtype=np.uint8)
    res = scorer(frames, [(40, 20, 120, 100)] * 20, np.zeros(int(16000 * 21 / 25), np.float32))
    assert isinstance(res, SyncNetResult) and res.windows == 15 and abs(res.offset) <= 15


# ------------------------------------------------------------------ pipeline integration
class BoxFace:
    def __call__(self, frame):
        return FaceObs(1, yaw=1.0, pitch=0.0, face_h_frac=0.4, mouth_w_px=150, opening=float(frame.mean()) / 255,
                       box=(500.0, 200.0, 780.0, 480.0))  # fmt: skip


def _fake_syncnet(conf, offset=0):
    calls = []

    def score(frames, boxes, audio):
        calls.append((len(frames), len(audio)))
        return SyncNetResult(conf, offset, 1.0, max(0, len(frames) - 5))

    score.calls = calls
    return score


def _run(tmp_path, syncnet):
    from pvdata.sources import Source
    from tests.test_pipeline_e2e import FakeASR, make_video

    tmp_path.mkdir(parents=True, exist_ok=True)
    video = make_video(tmp_path / "talk.mp4")
    texts = ["नमस्ते भारत आज मौसम", "हम पाँच लोग साथ हैं"]
    pipe = Pipeline(Config(), tmp_path / "out", tmp_path / "work", FakeASR(texts), BoxFace, syncnet=syncnet)
    pipe.run([Source(str(video), "ravi")])
    return pipe.manifest.rows()


@needs_ffmpeg
def test_syncnet_rejects_voice_over_and_keeps_talking_heads(tmp_path):
    low = _run(tmp_path / "low", _fake_syncnet(1.2))
    assert {r["reason"] for r in low} == {"sync_low_syncnet_conf"}
    good = _fake_syncnet(7.5)
    high = _run(tmp_path / "high", good)
    assert all(r["status"] == "accepted" and float(r["sync_conf"]) == 7.5 for r in high)
    frames, samples = good.calls[0]
    assert frames > 40 and abs(samples / 16000 - frames / 25) < 0.05  # clip + context, audio matches video


@needs_ffmpeg
def test_large_syncnet_offset_is_rejected(tmp_path):
    rows = _run(tmp_path, _fake_syncnet(8.0, offset=9))
    assert {r["reason"] for r in rows} == {"sync_syncnet_offset"}


def test_video_where_most_clips_fail_sync_is_dropped():
    pipe = Pipeline.__new__(Pipeline)
    pipe.cfg = Config()
    ok = [Decision("accepted", "", "x", {}, True) for _ in range(3)]
    bad = [Decision("rejected", "sync_low_syncnet_conf", "x", {}, True) for _ in range(7)]
    text = [Decision("rejected", "text_contains_latin") for _ in range(20)]
    assert pipe.drop_voiceover_sources(ok + bad + text)
    assert all(d.status == "rejected" for d in ok) and ok[0].reason == "source_low_sync_rate"
    good = [Decision("accepted", "", "x", {}, True) for _ in range(8)] + bad[:2]
    assert not pipe.drop_voiceover_sources(good)
    few = [Decision("accepted", "", "x", {}, True)] + [Decision("rejected", "r", "x", {}, True) for _ in range(4)]
    assert not pipe.drop_voiceover_sources(few)  # too few clips to judge the video


def test_old_manifest_gets_new_columns_when_appended(tmp_path):
    path = tmp_path / "manifest.csv"
    old = [f for f in MANIFEST_FIELDS if f not in ("sync_conf", "syncnet_offset")]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, old)
        w.writeheader()
        w.writerow({"clip_id": "spk01_000", "status": "accepted", "reason": ""})
    m = Manifest(path)
    assert next(csv.reader(open(path, encoding="utf-8"))) == old  # reading alone never rewrites the file
    m.add({"clip_id": "spk01_001", "status": "accepted", "sync_conf": "6.1"})
    rows = Manifest(path).rows()
    assert next(csv.reader(open(path, encoding="utf-8"))) == MANIFEST_FIELDS
    assert [r["clip_id"] for r in rows] == ["spk01_000", "spk01_001"] and rows[1]["sync_conf"] == "6.1"


@needs_ffmpeg
def test_resync_moves_failing_clips_and_marks_the_manifest(tmp_path):
    from pvdata.resync import resync

    rows = _run(tmp_path, None)  # mined with the old checks only
    out = tmp_path / "out"
    assert sum(r["status"] == "accepted" for r in rows) == 2
    spk = next((out / "raw_videos").iterdir())
    (spk / f"{spk.name}_999.mp4").write_bytes(b"not a video")  # e.g. a clip still being written
    made = []

    def factory():
        made.append(1)
        return BoxFace()

    dry = resync(out, Config(), factory, _fake_syncnet(1.0), workers=2)
    assert dry == {"clips": 2, "pass": 0, "fail": 2, "videos_dropped": 0, "reasons": {"low_syncnet_conf": 2},
                   "unreadable_retry_next_run": 1, "applied": False}  # fmt: skip
    assert len(made) <= 2  # one face detector per worker, not per clip
    (spk / f"{spk.name}_999.mp4").unlink()
    assert len(list((out / "raw_videos").glob("*/*.mp4"))) == 2
    done = resync(out, Config(), BoxFace, _fake_syncnet(9.0), apply=True)  # cached: the first scores are kept
    assert done["fail"] == 2 and done["applied"]
    assert not list((out / "raw_videos").glob("*/*.mp4"))
    assert len(list((out / "excluded_sync").glob("*/*.mp4"))) == 2
    marked = Manifest(out / "manifest.csv").rows()
    assert {r["reason"] for r in marked if r.get("clip_id")} == {"resync_low_syncnet_conf"}
