"""Shot-cut detection: a clip must be one continuous camera shot.

News packages cut between the anchor, B-roll and the person being talked about. A cut inside a clip means
the mouth on screen changes person or disappears while the voice goes on. We use the adaptive rule of
PySceneDetect's AdaptiveDetector: the mean HSV change between consecutive (downscaled) frames must be large
in absolute terms AND several times larger than every change in the frames around it (a cut is an isolated
spike), so steady motion (a speaker gesturing, a slow zoom, a fade, a flash) does not count as a cut.
"""

from __future__ import annotations

import numpy as np


def change_scores(frames_rgb, width: int = 96) -> np.ndarray:
    """Mean absolute HSV difference to the previous frame, per frame (0 for the first).

    Brightness is compared after removing each frame's mean, so a camera flash or auto-exposure step (common at
    press conferences) is not a cut, while a change of scene, framing or person still is."""
    import cv2

    small = []
    for f in frames_rgb:
        h, w = f.shape[:2]
        s = cv2.resize(f, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(s, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 2] -= hsv[..., 2].mean()
        small.append(hsv)
    scores = np.zeros(len(small), dtype=np.float64)
    for i in range(1, len(small)):
        scores[i] = float(np.mean(np.abs(small[i] - small[i - 1])))
    return scores


def find_cuts(scores: np.ndarray, min_score: float = 15.0, ratio: float = 3.0, window: int = 2) -> list[int]:
    """Indices i where a new shot starts (frame i differs sharply from frame i-1)."""
    cuts = []
    n = len(scores)
    for i in range(1, n):
        neigh = [scores[j] for j in range(max(1, i - window), min(n, i + window + 1)) if j != i]
        base = max(neigh + [1.0])
        if scores[i] >= min_score and scores[i] >= ratio * base:
            cuts.append(i)
    return cuts


def shot_bounds(cuts: list[int], n: int, lo: int, hi: int) -> tuple[int, int]:
    """[start, end) of the shot that contains frames lo..hi-1 (assumes no cut inside that range)."""
    start = max([c for c in cuts if c <= lo], default=0)
    end = min([c for c in cuts if c >= hi], default=n)
    return start, end
