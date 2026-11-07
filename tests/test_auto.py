"""Autopilot pieces: CTC-aware segmentation, stable speaker ids, metadata filter, screening decisions,
person-grouped splits, face-identity clustering, and the whole discover -> screen -> mine loop offline."""

import json
import shutil

import numpy as np
import pytest

from pvdata.asr import Word
from pvdata.auto import AutoPilot
from pvdata.config import ClipConfig, Config
from pvdata.export import grouped_splits
from pvdata.face import FaceObs
from pvdata.find import Hit, parse_hits
from pvdata.identity import identify, leader_cluster, load_persons
from pvdata.pipeline import Pipeline
from pvdata.screen import ScreenResult, decide, face_cfg_for, sample_windows
from pvdata.segment import segment_words
from pvdata.sources import Source, speaker_ids

GOOD = FaceObs(1, yaw=2, pitch=1, face_h_frac=0.4, mouth_w_px=30)  # 30 px mouth is fine at 360p (limit 20)


def words_every(texts, step=0.4):
    out, t = [], 0.3
    for tok in texts.split():
        out.append(Word(tok, t, t + 0.35, 0.9))
        t += step
    return out


# ------------------------------------------------------------------ segmentation
def test_segment_takes_longest_run_that_passes_accept():
    words = words_every("एक दो तीन चार पाँच")  # 5 words in ~2.0 s: one candidate of all five by default
    cfg = ClipConfig()
    assert len(segment_words(words, cfg)[0].words) == 5
    shorter = segment_words(words, cfg, accept=lambda text: len(text.split()) <= 4)
    assert [len(c.words) for c in shorter][0] == 4
    # nothing passes -> the longest run is still produced, so the checks can log why it failed
    assert len(segment_words(words, cfg, accept=lambda text: False)[0].words) == 5


# ------------------------------------------------------------------ ids
def test_speaker_ids_keep_existing_numbers():
    srcs = [Source("a", "ravish"), Source("b", "new_person"), Source("c", "abhisar")]
    ids = speaker_ids(srcs, {"ravish": "spk01", "abhisar": "spk02", "punya": "spk04"})
    assert ids["ravish"] == "spk01" and ids["abhisar"] == "spk02" and ids["new_person"] == "spk05"


def test_parse_hits_reads_channel_id():
    h = parse_hits("abc12345678\t600\tTitle\tChan\tUC123\nxyz\tNA\tT2\tC2\tNA\n")
    assert h[0].channel_id == "UC123" and h[1].channel_id == "" and h[1].duration == 0.0


# ------------------------------------------------------------------ screening
def test_sample_windows_spread_and_inside_video():
    w = sample_windows(1000, Config().screen)
    assert len(w) == 3 and w[0][0] >= 150 and w[-1][1] <= 1000 and all(b - a == 20 for a, b in w)
    assert sample_windows(15, Config().screen)[0][0] == 0


def test_screen_decisions():
    cfg = Config()
    hindi = words_every("आज हम बात करेंगे देश की राजनीति पर " * 5)
    ok = decide([GOOD] * 40, 360, hindi, 20.0, 0.9, cfg)
    assert ok.ok and ok.score > 0.9
    assert face_cfg_for(360, cfg.face).min_mouth_px == 20
    assert decide([FaceObs(0)] * 30 + [GOOD] * 10, 360, hindi, 20, 0.9, cfg).reason == "screen_face_no_face"
    assert decide([FaceObs(2)] * 40, 360, hindi, 20, 0.9, cfg).reason == "screen_face_multiple_faces"
    english = words_every("today we will talk about politics in the country " * 4)
    assert decide([GOOD] * 40, 360, english, 20, None, cfg).reason == "screen_too_much_english"
    assert decide([GOOD] * 40, 360, hindi, 20, 0.2, cfg).reason == "screen_not_hindi"
    assert decide([GOOD] * 40, 360, hindi[:5], 20, 0.9, cfg).reason == "screen_no_speech"


# ------------------------------------------------------------------ splits by person
def test_grouped_splits_keep_a_persons_folders_together():
    folders = [f"spk{i:02d}" for i in range(1, 21)]
    clips = {f: 50 for f in folders}
    groups = {"p1": ["spk01", "spk02", "spk03"], "p2": ["spk04", "spk05"]}
    for seed in range(10):
        s = grouped_splits(folders, clips, groups, 0.1, 0.1, seed)
        where = {f: k for k, v in s.items() for f in v}
        assert sorted(where) == folders
        assert len({where[f] for f in groups["p1"]}) == 1 and len({where[f] for f in groups["p2"]}) == 1


# ------------------------------------------------------------------ identity
def test_leader_cluster_separates_two_faces():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=128), rng.normal(size=128)
    vecs = [x / np.linalg.norm(x) for x in [a + 0.3 * rng.normal(size=128) for _ in range(5)]
            + [b + 0.3 * rng.normal(size=128) for _ in range(3)]]  # fmt: skip
    labels = leader_cluster(np.stack(vecs), 0.4)
    assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1 and labels[0] != labels[5]


def test_identify_merges_folders_and_quarantines_guests(tmp_path):
    rng = np.random.default_rng(1)
    faces = {k: rng.normal(size=128) for k in ("ravi", "guest", "meera")}
    layout = {"spk01": ["ravi"] * 6 + ["guest"], "spk02": ["ravi"] * 4, "spk03": ["meera"] * 5}
    who = {}
    for folder, people in layout.items():
        for i, person in enumerate(people, 1):
            clip = tmp_path / "raw_videos" / folder / f"{folder}_{i:03d}.mp4"
            clip.parent.mkdir(parents=True, exist_ok=True)
            clip.write_bytes(b"")
            clip.with_suffix(".txt").write_text("नमस्ते", encoding="utf-8")
            who[clip.stem] = person

    def embed(clip):
        v = faces[who[clip.stem]] + 0.2 * rng.normal(size=128)
        return v / np.linalg.norm(v)

    summary = identify(tmp_path, embed, 0.4, quarantine=True)
    assert summary["persons"] == 2 and summary["other_face"] == 1 and summary["quarantined"] == 1
    persons = load_persons(tmp_path)
    assert sorted(map(sorted, persons.values())) == [["spk01", "spk02"], ["spk03"]]
    assert (tmp_path / "quarantine" / "spk01" / "spk01_007.mp4").exists()
    again = identify(tmp_path, lambda c: pytest.fail("cached clips must not be embedded again"), 0.4)
    assert again["persons"] == 2


# ------------------------------------------------------------------ the loop, offline
class FakeScreener:
    def __init__(self, reject=()):
        self.reject, self.seen = set(reject), []

    def fetch(self, vid, duration):
        return []

    def screen(self, vid, duration, samples=None):
        self.seen.append(vid)
        return (
            ScreenResult(False, "screen_face_no_face") if vid in self.reject else ScreenResult(True, "", 0.9, 0.9, 1.0)
        )


def test_autopilot_discovers_screens_mines_expands_and_caps(tmp_path):
    from tests.test_pipeline_e2e import BrightnessFace, FakeASR, make_video

    base = make_video(tmp_path / "base.mp4")
    files = {}
    for vid in ("a1", "a2", "a3", "a4", "b1"):
        files[vid] = tmp_path / f"{vid}.mp4"
        shutil.copy(base, files[vid])
    searched = []

    def search(q, pool):
        searched.append(pool)
        return [Hit("a1", 600, "Anchor talk 1", "Chan A", "UCa"), Hit("a2", 600, "Anchor talk 2", "Chan A", "UCa"),
                Hit("b1", 600, "Studio", "Chan B", "UCb"), Hit("long1", 3 * 3600, "Marathon", "Chan C", "UCc"),
                Hit("deb1", 900, "Big Debate tonight", "Chan D", "UCd")]  # fmt: skip

    def channel_list(cid, n):
        assert cid == "UCa"
        # a3 and a5 have no length in the listing (0): it is looked up before screening
        return [Hit("a3", 0, "Anchor talk 3", "Chan A", "UCa"), Hit("a4", 600, "Anchor talk 4", "Chan A", "UCa"),
                Hit("a5", 0, "Anchor talk 5", "Chan A", "UCa")]  # fmt: skip

    cfg = Config()
    cfg.auto.max_videos_per_channel, cfg.auto.expand_min_minutes_per_video = 3, 0.0
    texts = ["नमस्ते भारत आज मौसम", "पानी दो मुझे अभी"]

    def make(tmp):
        pipe = Pipeline(cfg, tmp / "out", tmp / "work", FakeASR(texts), BrightnessFace, keep_downloads=False)
        mk = lambda v: Source(str(files[v["id"]]), f"{v['channel_id']}_{v['id']}")  # noqa: E731
        ap = AutoPilot(cfg, tmp / "out", ["anchor hindi"], pipe, FakeScreener(reject={"b1"}), search, channel_list, mk)
        ap.duration_of = lambda vid: {"a3": 700.0, "a5": 50.0}[vid]
        return ap

    pilot = make(tmp_path)
    summary = pilot.run(target_hours=10, max_videos=10)
    st = pilot.state.videos
    assert [st[v]["status"] for v in ("a1", "a2", "a3")] == ["mined"] * 3
    assert st["a4"]["reason"] == "channel_cap" and st["b1"]["reason"] == "screen_face_no_face"
    assert st["a3"]["duration"] == 700.0 and st["a5"]["reason"] == "meta_duration"
    assert st["long1"]["reason"] == "meta_duration" and st["deb1"]["reason"] == "meta_title"
    assert summary["mined_this_session"] == 3 and summary["hours"] > 0
    assert searched == [50, 100, 200, 400]  # each round searches deeper, then stops
    speakers = json.loads((tmp_path / "out" / "speakers.json").read_text(encoding="utf-8"))
    assert sorted(speakers) == ["spk01", "spk02", "spk03"]
    assert files["a1"].exists()  # local sources are never deleted

    resumed = make(tmp_path).run(target_hours=10, max_videos=10)  # restart: nothing is redone
    assert resumed["mined_this_session"] == 0
    clips = list((tmp_path / "out" / "raw_videos").glob("spk*/*.mp4"))
    assert len(clips) == sum(st[v]["accepted"] for v in ("a1", "a2", "a3")) > 0


def test_release_deletes_full_download_but_keeps_asr(tmp_path):
    pipe = Pipeline(Config(), tmp_path / "out", tmp_path / "work", None, None, keep_downloads=False)
    src = Source("https://www.youtube.com/watch?v=abc", "x")
    for sub, ext in (("downloads", "mp4"), ("audio", "wav"), ("asr", "json")):
        (tmp_path / "work" / sub / f"{src.source_id}.{ext}").write_bytes(b"x")
    pipe.release(src)
    assert not (tmp_path / "work" / "downloads" / f"{src.source_id}.mp4").exists()
    assert not (tmp_path / "work" / "audio" / f"{src.source_id}.wav").exists()
    assert (tmp_path / "work" / "asr" / f"{src.source_id}.json").exists()


def test_screening_waits_while_enough_videos_are_ready_to_mine(tmp_path):
    from tests.test_pipeline_e2e import BrightnessFace, FakeASR, make_video

    base = make_video(tmp_path / "base.mp4")

    def mk(v):
        f = tmp_path / f"{v['id']}.mp4"
        if not f.exists():
            shutil.copy(base, f)
        return Source(str(f), v["id"])

    hits = [Hit(f"v{i:02d}", 600, f"Talk {i}", f"Chan {i}", f"UC{i}") for i in range(40)]
    cfg = Config()
    pipe = Pipeline(cfg, tmp_path / "out", tmp_path / "work", FakeASR(["नमस्ते भारत आज मौसम"]), BrightnessFace)
    screener = FakeScreener()
    pilot = AutoPilot(cfg, tmp_path / "out", ["q"], pipe, screener, lambda q, n: hits, lambda c, n: [], mk)
    pilot.run(target_hours=10, max_videos=3)
    # samples arrive instantly here; before the fix every one of the 40 videos was screened before mining
    assert len(screener.seen) <= 3 + cfg.auto.ok_buffer + 3
