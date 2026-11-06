"""``pvdata`` command line.

pvdata run      --sources sources/my_sources.csv --out dataset/      # mine clips
pvdata auto     --seeds sources/seeds.csv --out dataset/           # find + screen + mine until --target-hours
pvdata screen   https://www.youtube.com/watch?v=XXXX                 # would auto accept this video? (dry run)
pvdata identify --out dataset/                                       # face ids: merge persons, flag guests
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
                    lambda: MediaPipeFaces(args.face_model, cfg.clip.fps), cookies=args.cookies,
                    workers=args.workers)  # fmt: skip
    report = pipe.run(sources, progress=lambda s: tqdm(s, desc="sources"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _models(args, cfg):
    from pvdata.asr import make_transcriber

    a = cfg.asr
    return make_transcriber(a.backend, getattr(args, "asr_model", None) or a.model, a.language, a.device,
                            a.compute_type)  # fmt: skip


def _cmd_auto(args) -> int:
    from pvdata.auto import AutoPilot, read_seeds
    from pvdata.face import MediaPipeFaces
    from pvdata.find import channel_videos, search_raw
    from pvdata.pipeline import Pipeline
    from pvdata.screen import Screener

    cfg = Config.load(args.config)
    if args.keep_downloads:
        cfg.auto.keep_downloads = True
    work = args.work or Path(args.out) / "work"
    transcriber = _models(args, cfg)
    pipe = Pipeline(cfg, args.out, work, transcriber, lambda: MediaPipeFaces(args.face_model, cfg.clip.fps),
                    cookies=args.cookies, workers=args.workers, keep_downloads=cfg.auto.keep_downloads)  # fmt: skip
    screener = Screener(cfg, work, transcriber, lambda fps: MediaPipeFaces(args.face_model, fps), args.cookies)
    pilot = AutoPilot(cfg, args.out, read_seeds(args.seeds), pipe, screener, search_raw, channel_videos)
    summary = pilot.run(args.target_hours, args.max_videos)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _cmd_screen(args) -> int:
    import re

    from pvdata.face import MediaPipeFaces
    from pvdata.find import search_raw
    from pvdata.screen import Screener

    cfg = Config.load(args.config)
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", args.video)
    vid = m.group(1) if m else args.video
    hits = [h for h in search_raw(vid, 5) if h.video_id == vid]  # cheapest way to get the duration
    duration = args.duration or (hits[0].duration if hits else 600.0)
    screener = Screener(cfg, Path(args.work), _models(args, cfg), lambda fps: MediaPipeFaces(None, fps), args.cookies)
    res = screener.screen(vid, duration)
    print(json.dumps({"video": vid, "duration": duration, **res.to_dict()}, ensure_ascii=False, indent=2))
    return 0 if res.ok else 1


def _cmd_identify(args) -> int:
    from pvdata.identity import SFaceEmbedder, identify

    cfg = Config.load(args.config)
    summary = identify(args.out, SFaceEmbedder(cfg.identity.frames_per_clip), cfg.identity.same_person,
                       quarantine=args.quarantine)  # fmt: skip
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_find(args) -> int:
    from pvdata.find import find_sources

    n = find_sources(args.queries, args.out, args.per_speaker, args.min_minutes, args.max_minutes)
    print(f"wrote {n} videos to {args.out}. Review every row before `pvdata run`.")
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
    from pvdata.export import grouped_splits, materialise_splits, write_json
    from pvdata.identity import load_persons

    cfg = Config.load(args.config)
    out = Path(args.out)
    speakers = sorted(p.name for p in (out / "raw_videos").iterdir() if p.is_dir())
    clips = {spk: len(list((out / "raw_videos" / spk).glob("*.mp4"))) for spk in speakers}
    persons = load_persons(out)
    if persons:
        print(f"using identity/persons.json: {len(persons)} persons over {len(speakers)} speaker folders")
    splits = grouped_splits(speakers, clips, persons, cfg.val_frac, cfg.test_frac, cfg.seed)
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
    r.add_argument(
        "--workers",
        type=int,
        default=0,
        help="candidates checked in parallel (default: number of CPU cores, at most 8)",
    )
    r.set_defaults(fn=_cmd_run)

    au = sub.add_parser("auto", help="discover, screen and mine YouTube videos until --target-hours is reached")
    au.add_argument("--seeds", required=True, help="CSV with a 'query' column (see sources/seeds.csv)")
    au.add_argument("--out", required=True)
    au.add_argument("--work", help="cache folder (default: <out>/work)")
    au.add_argument("--target-hours", type=float, help="stop at this many accepted hours (default: config, 100)")
    au.add_argument("--max-videos", type=int, help="stop after mining this many videos (for a test)")
    au.add_argument("--keep-downloads", action="store_true", help="keep full videos after mining (needs disk)")
    au.add_argument("--asr-model")
    au.add_argument("--face-model")
    au.add_argument("--cookies")
    au.add_argument("--workers", type=int, default=0)
    au.set_defaults(fn=_cmd_auto)

    sc = sub.add_parser("screen", help="screen one YouTube video (face + Hindi checks on 3 short samples)")
    sc.add_argument("video", help="YouTube URL or video id")
    sc.add_argument("--duration", type=float, help="video length in seconds (looked up if omitted)")
    sc.add_argument("--work", default="screen_test")
    sc.add_argument("--asr-model")
    sc.add_argument("--cookies")
    sc.set_defaults(fn=_cmd_screen)

    idf = sub.add_parser("identify", help="face embeddings -> persons.json (merge speakers, flag other faces)")
    idf.add_argument("--out", required=True)
    idf.add_argument("--quarantine", action="store_true", help="move clips of other faces to <out>/quarantine/")
    idf.set_defaults(fn=_cmd_identify)

    f = sub.add_parser("find", help="build a sources CSV from YouTube searches (speaker,query per row)")
    f.add_argument("--queries", required=True, help="CSV with columns speaker,query")
    f.add_argument("--out", required=True, help="sources CSV to write")
    f.add_argument("--per-speaker", type=int, default=3)
    f.add_argument("--min-minutes", type=float, default=4)
    f.add_argument("--max-minutes", type=float, default=40)
    f.set_defaults(fn=_cmd_find)

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
