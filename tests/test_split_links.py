"""Split folders link to raw_videos (never copies by default) and re-running split never deletes clips."""

from pvdata.export import _is_link, materialise_splits


def test_split_links_and_rerun_keeps_the_clips(tmp_path):
    raw = tmp_path / "raw_videos" / "spk01"
    raw.mkdir(parents=True)
    (raw / "spk01_000.mp4").write_bytes(b"x")
    (raw / "spk01_000.txt").write_text("नमस्ते", encoding="utf-8")
    splits = {"train": ["spk01"], "val": [], "test": []}
    assert materialise_splits(tmp_path, splits) == {"train": 1, "val": 0, "test": 0}
    link = tmp_path / "splits" / "train" / "spk01"
    assert _is_link(link) and (link / "spk01_000.mp4").exists()
    materialise_splits(tmp_path, splits)  # a second run replaces the link, not the clips behind it
    assert (raw / "spk01_000.mp4").exists() and (raw / "spk01_000.txt").exists()
    assert _is_link(link)
