from pvdata.find import parse_hits


def test_parse_hits():
    out = "abc123\t600.0\tटाइटल एक\tChannel A\nbad line\nxyz789\tNA\tTitle\tChannel B\n"
    hits = parse_hits(out)
    assert [h.video_id for h in hits] == ["abc123", "xyz789"]
    assert hits[0].duration == 600 and hits[1].duration == 0
    assert hits[0].url == "https://www.youtube.com/watch?v=abc123"
