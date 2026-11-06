"""Speech recognition with word timestamps. Every backend returns the same list of ``Word``s.

Backends (choose with ``asr.backend`` in the config):
- ``faster-whisper`` (default): Whisper large-v3 with word timestamps; fast on GPU, works on CPU.
- ``whisperx``: Whisper + wav2vec2 forced alignment, the most accurate word boundaries.
- ``hf``: any Hugging Face Whisper checkpoint, e.g. AI4Bharat IndicWhisper (Vistaar) for better Hindi text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


class Transcriber:
    def transcribe(self, wav: str | Path) -> list[Word]:  # pragma: no cover - interface
        raise NotImplementedError

    def language_probs(self, wav: str | Path) -> dict[str, float] | None:
        """Spoken-language probabilities (e.g. {"hi": 0.8, "ur": 0.1, ...}), or None if the backend can't tell."""
        return None


class FasterWhisper(Transcriber):
    def __init__(self, model: str = "large-v3", language: str = "hi", device: str = "auto", compute_type: str = "auto"):
        from faster_whisper import WhisperModel

        self.language = language
        self.model = WhisperModel(
            model, device=device, compute_type="default" if compute_type == "auto" else compute_type
        )
        logging.getLogger("pvdata").info("faster-whisper %s on %s", model, getattr(self.model.model, "device", "?"))

    def transcribe(self, wav):
        segments, _ = self.model.transcribe(str(wav), language=self.language, word_timestamps=True,
                                            vad_filter=True, condition_on_previous_text=False)  # fmt: skip
        return [Word(w.word.strip(), float(w.start), float(w.end), float(w.probability))
                for seg in segments for w in (seg.words or []) if w.word.strip()]  # fmt: skip

    def language_probs(self, wav):
        _, info = self.model.transcribe(str(wav), language=None, vad_filter=True)  # detection runs eagerly
        probs = dict(getattr(info, "all_language_probs", None) or [])
        return probs or {info.language: float(info.language_probability)}


class WhisperX(Transcriber):
    def __init__(self, model: str = "large-v3", language: str = "hi", device: str = "auto", compute_type: str = "auto"):
        import whisperx

        self.wx, self.language = whisperx, language
        self.device = ("cuda" if _cuda() else "cpu") if device == "auto" else device
        ct = ("float16" if self.device == "cuda" else "int8") if compute_type == "auto" else compute_type
        self.model = whisperx.load_model(model, self.device, compute_type=ct, language=language)
        self.align_model, self.meta = whisperx.load_align_model(language_code=language, device=self.device)

    def transcribe(self, wav):
        audio = self.wx.load_audio(str(wav))
        result = self.model.transcribe(audio, batch_size=8, language=self.language)
        aligned = self.wx.align(result["segments"], self.align_model, self.meta, audio, self.device)
        words = []
        for w in aligned.get("word_segments", []):
            if "start" in w and "end" in w and w.get("word", "").strip():
                words.append(Word(w["word"].strip(), float(w["start"]), float(w["end"]), float(w.get("score", 1.0))))
        return words


class HFWhisper(Transcriber):
    """Hugging Face pipeline (e.g. an IndicWhisper checkpoint). Word timestamps come from cross-attention."""

    def __init__(self, model: str, language: str = "hi", device: str = "auto", compute_type: str = "auto"):
        from transformers import pipeline

        dev = (0 if _cuda() else -1) if device == "auto" else device
        self.pipe = pipeline("automatic-speech-recognition", model=model, device=dev, chunk_length_s=30)
        self.language = language

    def transcribe(self, wav):
        out = self.pipe(
            str(wav), return_timestamps="word", generate_kwargs={"language": self.language, "task": "transcribe"}
        )
        words = []
        for c in out.get("chunks", []):
            start, end = c["timestamp"]
            if c["text"].strip() and start is not None and end is not None:
                words.append(Word(c["text"].strip(), float(start), float(end)))
        return words


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def make_transcriber(backend: str, model: str, language: str = "hi", device: str = "auto", compute_type: str = "auto"):
    backends = {"faster-whisper": FasterWhisper, "whisperx": WhisperX, "hf": HFWhisper}
    if backend not in backends:
        raise ValueError(f"unknown ASR backend '{backend}', choose from {sorted(backends)}")
    return backends[backend](model=model, language=language, device=device, compute_type=compute_type)
