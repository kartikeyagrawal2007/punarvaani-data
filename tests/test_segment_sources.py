from pvdata.asr import Word
from pvdata.config import ClipConfig
from pvdata.segment import segment_words
from pvdata.sources import parse_time, read_sources, speaker_ids


def _words(spec):
    """spec: list of (start, end) -> Words named w0, w1, ..."""
    return [Word(f"w{i}", s, e) for i, (s, e) in enumerate(spec)]


def test_clips_are_1_5_to_2_5_s_and_disjoint():
    words = _words([(i * 0.4, i * 0.4 + 0.35) for i in range(30)])  # steady speech, 2.5 words/s
    cfg = ClipConfig()
    clips = segment_words(words, cfg)
    assert len(clips) >= 4
    for c in clips:
        assert cfg.min_s <= c.duration <= cfg.max_s + 1e-9
        assert cfg.min_words <= len(c.words) <= cfg.max_words
    for a, b in zip(clips, clips[1:]):
        assert a.end <= b.start + 1e-9
    used = [w.text for c in clips for w in c.words]
    assert len(used) == len(set(used))


def test_long_pause_breaks_a_clip():
    words = _words([(0.0, 0.4), (0.45, 0.85), (2.5, 2.9), (2.95, 3.4), (3.45, 3.9), (3.95, 4.4)])
    clips = segment_words(words, ClipConfig())
    for c in clips:
        starts = [w.start for w in c.words]
        assert not (0.45 in starts and 2.5 in starts)


def test_sources_csv(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("# comment\nurl,speaker,start,end,notes\nhttps://y/1,ravi,0:30,1:02:03,a\n/local/v.mp4,meena,,,\nhttps://y/2,ravi,,,\n",
                 encoding="utf-8")  # fmt: skip
    rows = read_sources(p)
    assert [r.speaker for r in rows] == ["ravi", "meena", "ravi"]
    assert rows[0].start == 30 and rows[0].end == 3723 and rows[1].is_local
    assert speaker_ids(rows) == {"ravi": "spk01", "meena": "spk02"}
    assert parse_time("") is None and parse_time("1:30") == 90
