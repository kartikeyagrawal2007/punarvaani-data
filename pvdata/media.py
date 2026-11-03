"""Downloading and cutting media with yt-dlp and ffmpeg (both called as programs)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise RuntimeError(f"'{binary}' not found on PATH. Install it (see README) and try again.")
    return path


def download(url: str, out_dir: str | Path, min_height: int = 720, cookies: str | None = None) -> Path:
    """Download one video as mp4 (best stream of at least ``min_height`` if available)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "%(id)s.%(ext)s")
    fmt = (
        f"bv*[height>={min_height}][ext=mp4]+ba[ext=m4a]/bv*[height>={min_height}]+ba/b[height>={min_height}]/bv*+ba/b"
    )
    cmd = [_require("yt-dlp"), "-f", fmt, "--merge-output-format", "mp4", "--no-playlist", "--write-info-json", "-o", template,
           "--print", "after_move:filepath", url]  # fmt: skip
    if cookies:
        cmd[1:1] = ["--cookies", cookies]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().splitlines()
    return Path(out[-1])


def read_info(video_path: str | Path) -> dict:
    """Licence and provenance from the yt-dlp .info.json written next to a download (empty if absent)."""
    p = Path(video_path).with_suffix(".info.json")
    if not p.exists():
        return {}
    info = json.loads(p.read_text(encoding="utf-8"))
    return {k: info.get(k) or "" for k in ("id", "license", "channel", "title", "upload_date")}


def probe(path: str | Path) -> dict:
    """Width, height, fps and duration of the first video stream."""
    cmd = [_require("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)]  # fmt: skip
    info = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    stream = info["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(num) / float(den) if float(den) else 0.0,
        "duration": float(info["format"]["duration"]),
    }


def extract_audio(video: str | Path, wav: str | Path, start: float | None = None, end: float | None = None) -> Path:
    cmd = [_require("ffmpeg"), "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video)]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0.0):.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav)]
    subprocess.run(cmd, check=True)
    return Path(wav)


def read_wav(path: str | Path) -> np.ndarray:
    """Mono 16 kHz PCM wav -> float32 array in [-1, 1] (no extra dependency)."""
    import wave

    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("expected mono 16-bit wav (use extract_audio)")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def cut_clip(video: str | Path, out: str | Path, start: float, end: float, height: int = 720, fps: int = 25,
             keep_audio: bool = True) -> Path:  # fmt: skip
    """Re-encode [start, end] as H.264 mp4 at ``fps`` and ``height`` (frame-accurate, even dimensions)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_require("ffmpeg"), "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(video),
           "-t", f"{end - start:.3f}", "-vf", f"scale=-2:{height},fps={fps}", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]  # fmt: skip
    cmd += ["-c:a", "aac", "-ar", str(SAMPLE_RATE), "-ac", "1"] if keep_audio else ["-an"]
    cmd += ["-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return out


def read_frames(video: str | Path, start: float = 0.0, end: float | None = None, fps: int = 25) -> np.ndarray:
    """RGB frames of [start, end] resampled to ``fps`` via ffmpeg -> (T, H, W, 3) uint8."""
    info = probe(video)
    w, h = info["width"], info["height"]
    cmd = [_require("ffmpeg"), "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(video)]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8)
    return frames.reshape(-1, h, w, 3)
