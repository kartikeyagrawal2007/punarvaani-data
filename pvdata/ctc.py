"""The Stage-1 CTC length rule, computed exactly like the trainer (Data/dataset.py in the main repo).

The trainer reads all frames of a clip and gives it T = ceil(frames / tubelet) x upsample CTC steps
(tubelet 2, upsample 4: a 2 s clip at 25 fps -> 50 frames -> 100 steps). It drops any clip whose
phoneme labels need more: U (labels, word separators '|' included) + R (adjacent repeats, which need
a blank between them) must be <= T. Clips that would be dropped at training time are dropped here.
(The old trainer sampled a fixed 16 frames -> 32 steps; set clip.ctc_steps: 32 to get that rule back.)
"""

from __future__ import annotations

import math

from pvdata.g2p import devanagari_to_phonemes


def phonemes(text: str) -> list[str]:
    return devanagari_to_phonemes(text.replace("।", " "))


def ctc_min_length(labels: list[str]) -> int:
    u = len(labels)
    r = sum(1 for a, b in zip(labels, labels[1:]) if a == b)
    return u + r


def ctc_steps_for(duration_s: float, fps: float, tubelet: int = 2, upsample: int = 4) -> int:
    """CTC steps the trainer gives a clip of this length (all frames read)."""
    frames = max(1, round(duration_s * fps))
    return math.ceil(frames / tubelet) * upsample


def fits_ctc(text: str, ctc_steps: int = 32) -> tuple[bool, int]:
    need = ctc_min_length(phonemes(text))
    return need <= ctc_steps, need
