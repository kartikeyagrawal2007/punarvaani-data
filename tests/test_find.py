from pvdata.find import parse_hits


def test_parse_hits():
    out = "abc123\t600.0\tटाइटल एक\tChannel A\nbad line\nxyz789\tNA\tTitle\tChannel B\n"
    hits = parse_hits(out)
    assert [h.video_id for h in hits] == ["abc123", "xyz789"]
    assert hits[0].duration == 600 and hits[1].duration == 0
    assert hits[0].url == "https://www.youtube.com/watch?v=abc123"


def test_per_video_speakers(tmp_path, monkeypatch):
    from pvdata import find

    q = tmp_path / "q.csv"
    q.write_text("speaker,query\n@tedx,TEDx Hindi\nravi,Ravi Hindi\n", encoding="utf-8")
    fake = {"TEDx Hindi": [find.Hit("a1", 600, "t", "TEDx"), find.Hit("b2", 700, "t", "TEDx")],
            "Ravi Hindi": [find.Hit("c3", 900, "t", "Ravi")]}  # fmt: skip
    monkeypatch.setattr(find, "search", lambda query, n, lo, hi: fake[query][:n])
    out = tmp_path / "s.csv"
    assert find.find_sources(q, out, per_speaker=2) == 3
    from pvdata.sources import read_sources

    assert [s.speaker for s in read_sources(out)] == ["tedx_a1", "tedx_b2", "ravi"]
