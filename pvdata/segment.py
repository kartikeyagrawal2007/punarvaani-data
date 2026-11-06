"""Cut a long transcript into short clips at word boundaries.

Each clip is a run of consecutive words that:
- lasts ``min_s``-``max_s`` seconds (1.5-2.5 s by default, as in TEAM_GUIDE), padding included;
- has ``min_words``-``max_words`` words;
- contains no pause longer than ``max_gap_s`` (a long pause usually means a new sentence or a cut);
- never shares a word with another clip.

The greedy scan prefers the longest valid run starting at each position, then jumps past it. If ``accept``
is given (e.g. "the normalised text fits the CTC limit"), it takes the longest run that ``accept`` passes
instead, so a fast speaker or a trailing English word costs one word rather than the whole clip. If no run
passes, the longest one is still returned, and the checks reject it with the reason.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pvdata.asr import Word
from pvdata.config import ClipConfig


@dataclass
class Candidate:
    start: float
    end: float
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def min_prob(self) -> float:
        return min((w.prob for w in self.words), default=1.0)


def segment_words(words: list[Word], cfg: ClipConfig, total_duration: float | None = None,
                  accept: Callable[[str], bool] | None = None) -> list[Candidate]:  # fmt: skip
    words = sorted((w for w in words if w.end > w.start), key=lambda w: w.start)
    out: list[Candidate] = []
    i = 0
    while i < len(words):
        valid: list[Candidate] = []
        for j in range(i, min(i + cfg.max_words, len(words))):
            if j > i and words[j].start - words[j - 1].end > cfg.max_gap_s:
                break
            n = j - i + 1
            start = max(0.0, words[i].start - cfg.pad_s)
            end = words[j].end + cfg.pad_s
            if total_duration is not None:
                end = min(end, total_duration)
            dur = end - start
            if dur > cfg.max_s:
                break
            if n >= cfg.min_words and dur >= cfg.min_s:
                valid.append(Candidate(start, end, words[i : j + 1]))
        best = valid[-1] if valid else None
        if accept is not None and len(valid) > 1:
            best = next((c for c in reversed(valid) if accept(c.text)), best)
        if best is not None:
            # don't let padding overlap the previous clip
            if out and best.start < out[-1].end:
                best.start = out[-1].end
            if best.duration >= cfg.min_s:
                out.append(best)
                i += len(best.words)
                continue
        i += 1
    return out
