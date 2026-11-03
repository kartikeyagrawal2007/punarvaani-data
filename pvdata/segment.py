"""Cut a long transcript into short clips at word boundaries.

Each clip is a run of consecutive words that:
- lasts ``min_s``-``max_s`` seconds (1.5-2.5 s by default, as in TEAM_GUIDE), padding included;
- has ``min_words``-``max_words`` words;
- contains no pause longer than ``max_gap_s`` (a long pause usually means a new sentence or a cut);
- never shares a word with another clip.

The greedy scan prefers the longest valid run starting at each position, then jumps past it.
"""

from __future__ import annotations

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


def segment_words(words: list[Word], cfg: ClipConfig, total_duration: float | None = None) -> list[Candidate]:
    words = sorted((w for w in words if w.end > w.start), key=lambda w: w.start)
    out: list[Candidate] = []
    i = 0
    while i < len(words):
        best: Candidate | None = None
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
                best = Candidate(start, end, words[i : j + 1])
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
