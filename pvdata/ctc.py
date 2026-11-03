"""The Stage-1 CTC length rule, computed exactly like the trainer (Data/dataset.py in the main repo).

The trainer turns 16 frames into T = 32 CTC steps and drops any clip whose phoneme labels need
more: U (number of labels, word separators '|' included) + R (adjacent repeats, which need a blank
between them) must be <= T. Clips that would be dropped at training time are dropped here instead.
"""

from __future__ import annotations

from pvdata.g2p import devanagari_to_phonemes


def phonemes(text: str) -> list[str]:
    return devanagari_to_phonemes(text.replace("।", " "))


def ctc_min_length(labels: list[str]) -> int:
    u = len(labels)
    r = sum(1 for a, b in zip(labels, labels[1:]) if a == b)
    return u + r


def fits_ctc(text: str, ctc_steps: int = 32) -> tuple[bool, int]:
    need = ctc_min_length(phonemes(text))
    return need <= ctc_steps, need
