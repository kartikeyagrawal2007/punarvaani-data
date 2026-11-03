"""``pvdata`` command line.

pvdata run      --sources sources/my_sources.csv --out dataset/      # mine clips
pvdata validate dataset/raw_videos                                   # check the TEAM_GUIDE rules
pvdata split    --out dataset/                                       # speaker-disjoint train/val/test
pvdata report   --out dataset/                                       # counts + rejection reasons
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pvdata.config import Config


def _cmd_run(args) -> int:
    from tqdm import tqdm

    from pvdata.asr import make_transcriber
    from pvdata.face import MediaPipeFaces
    from pvdata.pipeline import Pipeline
    from pvdata.sources import read_sources

    cfg = Config.load(args.config)
    if args.no_sync:
        cfg.sync.enabled = False
    sources = read_sources(args.sources)
    if args.limit:
        sources = sources[: args.limit]
    a = cfg.asr
    transcriber = make_transcriber(a.backend, args.asr_model or a.model, a.language, a.device, a.compute_type)
    pipe = Pipeline(cfg, args.out, args.work or Path(args.out) / "work", transcriber,
                    lambda: MediaPipeFaces(args.face_model, cfg.clip.fps), cookies=args.cookies)  # fmt: skip
    report = pipe.run(sources, progress=lambda s: tqdm(s, desc="sources"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _cmd_validate(args) -> int:
    from pvdata.validate import validate

    rep = validate(args.raw_videos, Config.load(args.config), check_media=not args.no_media)
    for w in rep.warnings[: args.show]:
        print("WARN ", w)
    for e in rep.errors[: args.show]:
        print("ERROR", e)
    print(f"\n{rep.clips} clips, {rep.speakers} speakers, {len(rep.errors)} errors, {len(rep.warnings)} warnings")
    print("clips per speaker:", rep.by_speaker)
    print("OK" if rep.ok else "NOT OK: fix the errors above")
    return 0 if rep.ok else 1


def _cmd_split(args) -> int:
    from pvdata.export import materialise_splits, speaker_splits, write_json

    cfg = Config.load(args.config)
    out = Path(args.out)
    speakers = sorted(p.name for p in (out / "raw_videos").iterdir() if p.is_dir())
    splits = speaker_splits(speakers, cfg.val_frac, cfg.test_frac, cfg.seed)
    write_json(out / "splits.json", splits)
    counts = materialise_splits(out, splits, link=not args.copy)
    print(json.dumps({"speakers": splits, "clips": counts}, ensure_ascii=False, indent=2))
    return 0


def _cmd_report(args) -> int:
    from pvdata.export import Manifest, build_report, write_json

    out = Path(args.out)
    report = build_report(Manifest(out / "manifest.csv").rows())
    write_json(out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="pvdata", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None, help="YAML config (default: built-in values = configs/default.yaml)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="mine clips from a sources CSV")
    r.add_argument("--sources", required=True)
    r.add_argument("--out", required=True, help="dataset folder (raw_videos/ is created inside)")
    r.add_argument("--work", help="cache folder for downloads, audio and ASR (default: <out>/work)")
    r.add_argument("--asr-model", help="override the ASR model name / Hugging Face id")
    r.add_argument("--face-model", help="path to face_landmarker.task (downloaded automatically if omitted)")
    r.add_argument("--cookies", help="cookies.txt for yt-dlp if YouTube asks to sign in")
    r.add_argument("--no-sync", action="store_true", help="skip the audio-mouth sync check")
    r.add_argument("--limit", type=int, help="only the first N sources (for a quick test)")
    r.set_defaults(fn=_cmd_run)

    v = sub.add_parser("validate", help="check a raw_videos/ folder against the TEAM_GUIDE rules")
    v.add_argument("raw_videos")
    v.add_argument("--no-media", action="store_true", help="skip ffprobe checks (faster)")
    v.add_argument("--show", type=int, default=50, help="max errors/warnings to print")
    v.set_defaults(fn=_cmd_validate)

    s = sub.add_parser("split", help="speaker-disjoint train/val/test split")
    s.add_argument("--out", required=True)
    s.add_argument("--copy", action="store_true", help="copy folders instead of symlinking")
    s.set_defaults(fn=_cmd_split)

    p = sub.add_parser("report", help="recompute report.json from manifest.csv")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_report)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
