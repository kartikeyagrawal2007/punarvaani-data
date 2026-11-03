"""The whole data engine, one source video at a time.

    source (URL or file) ─► download (yt-dlp) ─► audio 16 kHz (ffmpeg)
      ─► ASR with word timestamps (Whisper / WhisperX / IndicWhisper)
      ─► cut into 1.5–2.5 s word-aligned candidates
      ─► per candidate:  transcript rules ─► ASR confidence ─► CTC length (U+R ≤ 32)
                          ─► face checks (one frontal face, big enough, mouth visible, ≥95% of frames)
                          ─► audio–mouth sync
      ─► accepted: raw_videos/spkXX/spkXX_NNN.mp4 + .txt      rejected: logged with the reason

Everything is cached under <work>/ so a crashed or interrupted run continues where it stopped.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pvdata import media
from pvdata.asr import Transcriber, Word
from pvdata.config import Config
from pvdata.ctc import fits_ctc
from pvdata.export import Manifest, build_report, clip_paths, next_index, write_json, write_transcript
from pvdata.face import FaceObs, check_clip
from pvdata.segment import Candidate, segment_words
from pvdata.sources import Source, speaker_ids
from pvdata.sync import loudness_envelope, sync_score
from pvdata.text_norm import normalize_transcript

log = logging.getLogger("pvdata")


@dataclass
class Decision:
    status: str  # accepted | rejected
    reason: str = ""
    text: str = ""
    info: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg: Config, out: str | Path, work: str | Path, transcriber: Transcriber,
                 face_factory: Callable[[], Callable[[np.ndarray], FaceObs]], cookies: str | None = None):  # fmt: skip
        self.cfg, self.out, self.work = cfg, Path(out), Path(work)
        self.transcriber, self.face_factory, self.cookies = transcriber, face_factory, cookies
        for d in ("downloads", "audio", "asr", "done"):
            (self.work / d).mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest(self.out / "manifest.csv")

    # ---------------------------------------------------------------- stages
    def fetch(self, src: Source) -> Path:
        if src.is_local:
            return Path(src.url)
        cached = sorted((self.work / "downloads").glob(f"{src.source_id}.*"))
        if cached:
            return cached[0]
        path = media.download(src.url, self.work / "downloads" / "tmp", self.cfg.video.min_height, self.cookies)
        final = self.work / "downloads" / f"{src.source_id}{path.suffix}"
        path.rename(final)
        return final

    def transcribe(self, src: Source, video: Path, start: float, end: float | None) -> tuple[Path, list[Word]]:
        wav = self.work / "audio" / f"{src.source_id}.wav"
        if not wav.exists():
            media.extract_audio(video, wav, start, end)
        cache = self.work / "asr" / f"{src.source_id}.json"
        if cache.exists():
            words = [Word(**w) for w in json.loads(cache.read_text(encoding="utf-8"))]
        else:
            words = self.transcriber.transcribe(wav)
            cache.write_text(json.dumps([w.__dict__ for w in words], ensure_ascii=False), encoding="utf-8")
        return wav, words

    def judge(self, cand: Candidate, video: Path, offset: float, audio: np.ndarray) -> Decision:
        """Run all checks on one candidate. ``offset`` = section start in the source video."""
        c = self.cfg
        norm = normalize_transcript(cand.text, c.text.spell_out_numbers, c.text.reject_latin)
        if not norm.ok:
            return Decision("rejected", f"text_{norm.reason}")
        info = {"asr_min_prob": round(cand.min_prob, 3)}
        if cand.min_prob < c.text.min_asr_word_prob:
            return Decision("rejected", "low_asr_confidence", norm.text, info)
        ok, need = fits_ctc(norm.text, c.clip.ctc_steps)
        info["ctc_need"] = need
        if not ok:
            return Decision("rejected", "too_many_phonemes", norm.text, info)
        frames = media.read_frames(video, offset + cand.start, offset + cand.end, c.clip.fps)
        detector = self.face_factory()
        try:
            obs = [detector(f) for f in frames]
        finally:
            if hasattr(detector, "close"):
                detector.close()
        face = check_clip(obs, c.face)
        info.update(
            face_ratio=round(face.good_ratio, 3), yaw=round(face.median_yaw, 1), pitch=round(face.median_pitch, 1)
        )
        if not face.ok:
            return Decision("rejected", f"face_{face.reason}", norm.text, info)
        if c.sync.enabled:
            sr = media.SAMPLE_RATE
            seg = audio[int(cand.start * sr) : int(cand.end * sr)]
            env = loudness_envelope(seg, sr, c.clip.fps, len(face.openings))
            sync = sync_score(np.array(face.openings), env, c.sync)
            info.update(sync_corr=round(sync.corr, 3), sync_offset=sync.offset_frames)
            if not sync.ok:
                return Decision("rejected", f"sync_{sync.reason}", norm.text, info)
        return Decision("accepted", "", norm.text, info)

    # ---------------------------------------------------------------- driver
    def run_source(self, src: Source, spk: str) -> dict:
        done = self.work / "done" / f"{src.source_id}.json"
        if done.exists():
            log.info("skip %s (already processed)", src.url)
            return json.loads(done.read_text(encoding="utf-8"))
        video = self.fetch(src)
        info = media.probe(video)
        counts = {"accepted": 0, "rejected": 0}
        base_row = {"speaker": spk, "speaker_name": src.speaker, "source_url": src.url, "source_id": src.source_id}
        if info["height"] < self.cfg.video.min_height:
            self.manifest.add(
                {**base_row, "status": "rejected", "reason": f"source_below_{self.cfg.video.min_height}p"}
            )
            counts["rejected"] += 1
        else:
            start = src.start or 0.0
            end = min(src.end, info["duration"]) if src.end else info["duration"]
            wav, words = self.transcribe(src, video, start, end)
            audio = media.read_wav(wav)
            for cand in segment_words(words, self.cfg.clip, end - start):
                d = self.judge(cand, video, start, audio)
                row = {**base_row, "start": round(start + cand.start, 3), "end": round(start + cand.end, 3),
                       "duration": round(cand.duration, 3), "text": d.text, "raw_asr_text": cand.text,
                       "status": d.status, "reason": d.reason, **d.info}  # fmt: skip
                if d.status == "accepted":
                    mp4, txt = clip_paths(self.out, spk, next_index(self.out, spk))
                    media.cut_clip(video, mp4, start + cand.start, start + cand.end, self.cfg.video.out_height,
                                   self.cfg.video.out_fps, self.cfg.video.keep_audio)  # fmt: skip
                    write_transcript(txt, d.text)
                    row["clip_id"] = mp4.stem
                self.manifest.add(row)
                counts[d.status] += 1
        write_json(done, counts)
        return counts

    def run(self, sources: list[Source], progress: Callable | None = None) -> dict:
        ids = speaker_ids(sources)
        write_json(self.out / "speakers.json", {v: k for k, v in ids.items()})
        it = progress(sources) if progress else sources
        for src in it:
            try:
                c = self.run_source(src, ids[src.speaker])
                log.info("%s -> %s", src.url, c)
            except Exception as e:  # one bad video must not stop a long run
                log.exception("failed: %s", src.url)
                self.manifest.add({"speaker": ids[src.speaker], "speaker_name": src.speaker, "source_url": src.url,
                                   "source_id": src.source_id, "status": "rejected",
                                   "reason": f"source_error:{type(e).__name__}"})  # fmt: skip
        report = build_report(self.manifest.rows())
        write_json(self.out / "report.json", report)
        return report
