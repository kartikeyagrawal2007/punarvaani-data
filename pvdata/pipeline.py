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
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pvdata import media
from pvdata.asr import Transcriber, Word
from pvdata.config import Config
from pvdata.ctc import fits_ctc
from pvdata.export import Manifest, build_report, clip_paths, is_youtube, next_index, write_json, write_transcript
from pvdata.face import FaceObs, check_clip, scan_frames
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
                 face_factory: Callable[[], Callable[[np.ndarray], FaceObs]], cookies: str | None = None,
                 workers: int = 1, keep_downloads: bool = True):  # fmt: skip
        self.cfg, self.out, self.work = cfg, Path(out), Path(work)
        self.transcriber, self.face_factory, self.cookies = transcriber, face_factory, cookies
        self.workers = workers if workers > 0 else min(8, os.cpu_count() or 2)
        self.keep_downloads = keep_downloads
        self._downloader = ThreadPoolExecutor(1)  # fetches the next video while the current one is processed
        self._pending: dict[str, Future] = {}
        for d in ("downloads", "audio", "asr", "done"):
            (self.work / d).mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest(self.out / "manifest.csv")

    # ---------------------------------------------------------------- stages
    def prefetch(self, src: Source) -> None:
        """Start downloading ``src`` in the background (no-op if it is local, finished or already pending)."""
        sid = src.source_id
        if src.is_local or sid in self._pending or (self.work / "done" / f"{sid}.json").exists():
            return
        self._pending[sid] = self._downloader.submit(self._fetch, src)

    def fetch(self, src: Source) -> Path:
        pending = self._pending.pop(src.source_id, None)
        return pending.result() if pending is not None else self._fetch(src)

    def _fetch(self, src: Source) -> Path:
        if src.is_local:
            return Path(src.url)
        final = self.work / "downloads" / f"{src.source_id}.mp4"
        if final.exists():
            return final
        path = media.download(src.url, self.work / "downloads" / "tmp", self.cfg.video.min_height, self.cookies)
        meta = media.read_info(path)
        (self.work / "meta").mkdir(exist_ok=True)
        write_json(self.work / "meta" / f"{src.source_id}.json", meta)
        path.with_suffix(".info.json").unlink(missing_ok=True)
        path.rename(final)
        return final

    def release(self, src: Source) -> None:
        """Delete the full download and audio of a finished source (the ASR cache and metadata stay)."""
        if self.keep_downloads or src.is_local:
            return
        for p in (self.work / "downloads" / f"{src.source_id}.mp4", self.work / "audio" / f"{src.source_id}.wav"):
            p.unlink(missing_ok=True)

    def accept_text(self, text: str) -> bool:
        """Cheap pre-check used to choose clip boundaries: would this text pass the transcript and CTC rules?"""
        t = self.cfg.text
        norm = normalize_transcript(text, t.spell_out_numbers, t.reject_latin)
        return norm.ok and fits_ctc(norm.text, self.cfg.clip.ctc_steps)[0]

    def provenance(self, src: Source) -> dict:
        """Licence/channel/title/upload date for the manifest (needed for the dataset paper's datasheet)."""
        p = self.work / "meta" / f"{src.source_id}.json"
        meta = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        prov = {k: meta.get(k) or "" for k in ("license", "channel", "title", "upload_date")}
        if not prov["license"] and is_youtube(src.url):
            prov["license"] = "youtube_standard"  # yt-dlp only reports a licence for Creative Commons videos
        return prov

    def transcribe(self, src: Source, video: Path, start: float, end: float | None) -> tuple[Path, list[Word]]:
        wav = self.work / "audio" / f"{src.source_id}.wav"
        if not wav.exists():
            media.extract_audio(video, wav, start, end)
        cache = self.work / "asr" / f"{src.source_id}.json"
        if cache.exists():
            words = [Word(**w) for w in json.loads(cache.read_text(encoding="utf-8"))]
        else:
            t0 = time.time()
            words = self.transcriber.transcribe(wav)
            log.info("ASR %s: %d words in %.0f s", src.url, len(words), time.time() - t0)
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
        frames = media.read_frames(video, offset + cand.start, offset + cand.end, c.clip.fps, c.video.out_height)
        detector = self.face_factory()
        try:
            obs = scan_frames(detector, frames, c.face)
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
        # An interrupted earlier attempt may have written some clips of this source: remove them, then redo it.
        for clip_id in self.manifest.drop_source(src.source_id):
            for ext in (".mp4", ".txt"):
                (self.out / "raw_videos" / spk / f"{clip_id}{ext}").unlink(missing_ok=True)
        video = self.fetch(src)
        info = media.probe(video)
        counts = {"accepted": 0, "rejected": 0, "accepted_s": 0.0}
        base_row = {"speaker": spk, "speaker_name": src.speaker, "source_url": src.url, "source_id": src.source_id,
                    **self.provenance(src)}  # fmt: skip
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
            cands = segment_words(words, self.cfg.clip, end - start, accept=self.accept_text)
            t0 = time.time()
            # Judging (frame decoding + face + sync) runs in parallel; writing clips stays in order, one at a time.
            with ThreadPoolExecutor(self.workers) as pool:
                decisions = pool.map(lambda c: self.judge(c, video, start, audio), cands)
                for cand, d in zip(cands, decisions):
                    self._write(src, spk, video, start, cand, d, base_row, counts)
            log.info("judged %d candidates of %s in %.0f s (%d workers)", len(cands), src.url, time.time() - t0,
                     self.workers)  # fmt: skip
        counts["accepted_s"] = round(counts["accepted_s"], 2)
        write_json(done, counts)
        self.release(src)
        return counts

    def _write(self, src: Source, spk: str, video: Path, start: float, cand: Candidate, d: Decision,
               base_row: dict, counts: dict) -> None:  # fmt: skip
        row = {**base_row, "start": round(start + cand.start, 3), "end": round(start + cand.end, 3),
               "duration": round(cand.duration, 3), "text": d.text, "raw_asr_text": cand.text,
               "status": d.status, "reason": d.reason, **d.info}  # fmt: skip
        if d.status == "accepted":
            mp4, txt = clip_paths(self.out, spk, next_index(self.out, spk))
            v = self.cfg.video
            media.cut_clip(video, mp4, start + cand.start, start + cand.end, v.out_height, v.out_fps, v.keep_audio)
            write_transcript(txt, d.text)
            row["clip_id"] = mp4.stem
            counts["accepted_s"] += cand.duration
        self.manifest.add(row)
        counts[d.status] += 1

    def assign_ids(self, sources: list[Source]) -> dict[str, str]:
        """speaker name -> spkNN, keeping the numbers already in speakers.json so ids never change between runs."""
        path = self.out / "speakers.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        ids = speaker_ids(sources, {name: spk for spk, name in existing.items()})
        write_json(path, {v: k for k, v in ids.items()})
        return ids

    def run_safe(self, src: Source, spk: str) -> dict | None:
        """run_source, but a failure is logged and recorded in the manifest instead of stopping a long run."""
        try:
            c = self.run_source(src, spk)
            log.info("%s -> %s", src.url, c)
            return c
        except Exception as e:  # one bad video must not stop a long run
            log.error("failed: %s: %s: %s", src.url, type(e).__name__, e)
            log.debug("traceback", exc_info=True)
            self.manifest.add({"speaker": spk, "speaker_name": src.speaker, "source_url": src.url,
                               "source_id": src.source_id, "status": "rejected",
                               "reason": f"source_error:{type(e).__name__}"})  # fmt: skip
            return None

    def run(self, sources: list[Source], progress: Callable | None = None) -> dict:
        ids = self.assign_ids(sources)
        it = progress(sources) if progress else sources
        for i, src in enumerate(it):
            if i + 1 < len(sources):
                self.prefetch(sources[i + 1])
            self.run_safe(src, ids[src.speaker])
        report = build_report(self.manifest.rows())
        write_json(self.out / "report.json", report)
        return report
