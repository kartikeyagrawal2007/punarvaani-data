"""Downloading and cutting media with yt-dlp and ffmpeg (both called as programs)."""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class ToolError(RuntimeError):
    """An external program (yt-dlp, ffmpeg, ffprobe) failed; the message ends with its last stderr lines."""


def _run(cmd: list[str], text: bool = False) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, capture_output=True, text=text)
    if res.returncode != 0:
        err = res.stderr if text else res.stderr.decode("utf-8", "replace")
        tail = " | ".join(line.strip() for line in err.strip().splitlines()[-3:])
        raise ToolError(f"{Path(cmd[0]).stem} exited with {res.returncode}: {tail[:500]}")
    return res


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise RuntimeError(f"'{binary}' not found on PATH. Install it (see README) and try again.")
    return path


def download(url: str, out_dir: str | Path, min_height: int = 720, cookies: str | None = None,
             prefer_height: int | None = 720) -> Path:  # fmt: skip
    """Download one video as mp4: a stream of at least ``min_height``, preferring ``prefer_height`` (clips are
    written at 720p, so a 1080p download only costs bandwidth)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "%(id)s.%(ext)s")
    fmt = (
        f"bv*[height>={min_height}][ext=mp4]+ba[ext=m4a]/bv*[height>={min_height}]+ba/b[height>={min_height}]/bv*+ba/b"
    )
    cmd = [_require("yt-dlp"), "-f", fmt, "--merge-output-format", "mp4", "--no-playlist", "--write-info-json", "-o", template,
           "--print", "after_move:filepath", url]  # fmt: skip
    if prefer_height:
        cmd[1:1] = ["-S", f"res:{prefer_height}"]
    if cookies:
        cmd[1:1] = ["--cookies", cookies]
    out = _run(cmd, text=True).stdout.strip().splitlines()
    return Path(out[-1])


def video_duration(url: str, cookies: str | None = None) -> float:
    """Length in seconds from YouTube metadata (nothing is downloaded); 0 if unknown."""
    cmd = [_require("yt-dlp"), "--skip-download", "--no-warnings", "--no-playlist", "--print", "%(duration)s", url]
    if cookies:
        cmd[1:1] = ["--cookies", cookies]
    try:
        return float(_run(cmd, text=True).stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 0.0


def download_sections(url: str, out_dir: str | Path, sections: list[tuple[float, float]], max_height: int = 360,
                      cookies: str | None = None) -> list[Path]:  # fmt: skip
    """Download only short pieces of a video (low resolution, with audio): cheap samples for screening."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = f"bv*[height<={max_height}]+ba/b[height<={max_height}]/wv*+ba/w"
    cmd = [_require("yt-dlp"), "-f", fmt, "-S", f"res:{max_height}", "--merge-output-format", "mp4", "--no-playlist",
           "-o", str(out_dir / "%(id)s.%(section_start)d.%(ext)s"), "--print", "after_move:filepath"]  # fmt: skip
    for a, b in sections:
        cmd += ["--download-sections", f"*{a:.0f}-{b:.0f}"]
    if cookies:
        cmd[1:1] = ["--cookies", cookies]
    out = _run(cmd + [url], text=True).stdout.strip().splitlines()
    return [Path(line) for line in out if line.strip() and Path(line.strip()).exists()]


def read_info(video_path: str | Path) -> dict:
    """Licence and provenance from the yt-dlp .info.json written next to a download (empty if absent)."""
    p = Path(video_path).with_suffix(".info.json")
    if not p.exists():
        return {}
    info = json.loads(p.read_text(encoding="utf-8"))
    return {k: info.get(k) or "" for k in ("id", "license", "channel", "title", "upload_date")}


@functools.lru_cache(maxsize=64)
def _probe(path: str, mtime: float) -> dict:
    cmd = [_require("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)]  # fmt: skip
    info = json.loads(_run(cmd, text=True).stdout)
    stream = info["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(num) / float(den) if float(den) else 0.0,
        "duration": float(info["format"]["duration"]),
    }


def probe(path: str | Path) -> dict:
    """Width, height, fps and duration of the first video stream (cached per file)."""
    p = Path(path)
    return dict(_probe(str(p.resolve()), p.stat().st_mtime))


def extract_audio(video: str | Path, wav: str | Path, start: float | None = None, end: float | None = None) -> Path:
    cmd = [_require("ffmpeg"), "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video)]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0.0):.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav)]
    _run(cmd)
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
    _run(cmd)
    return out


def read_frames(video: str | Path, start: float = 0.0, end: float | None = None, fps: int = 25,
                max_height: int | None = None) -> np.ndarray:  # fmt: skip
    """RGB frames of [start, end] resampled to ``fps`` via ffmpeg -> (T, H, W, 3) uint8.

    ``max_height`` scales taller sources down (e.g. 1080p -> 720p, the clip output size), which keeps the face
    checks measuring what the trainer will see and makes decoding much cheaper."""
    info = probe(video)
    w, h = info["width"], info["height"]
    vf = f"fps={fps}"
    if max_height and h > max_height:
        w, h = int(round(w * max_height / h / 2)) * 2, max_height
        vf += f",scale={w}:{h}"
    cmd = [_require("ffmpeg"), "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(video)]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = _run(cmd).stdout
    frames = np.frombuffer(raw, dtype=np.uint8)
    return frames.reshape(-1, h, w, 3)
