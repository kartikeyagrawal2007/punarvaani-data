"""Per-clip face checks with MediaPipe Face Landmarker.

For every frame we measure: how many faces, head yaw/pitch (from MediaPipe's face transformation
matrix), face height relative to the frame, mouth width in pixels, and mouth opening (inner-lip gap /
mouth width, which the sync check uses). A clip passes only if, in at least ``min_detect_ratio`` of its
frames, there is exactly one face that is frontal, big enough and has a visible mouth.
"""

from __future__ import annotations

import math
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pvdata.config import FaceConfig

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MOUTH_LEFT, MOUTH_RIGHT, LIP_TOP, LIP_BOTTOM = 61, 291, 13, 14
FOREHEAD, CHIN = 10, 152


@dataclass
class FaceObs:
    n_faces: int
    yaw: float = 0.0  # degrees
    pitch: float = 0.0
    face_h_frac: float = 0.0
    mouth_w_px: float = 0.0
    opening: float = 0.0


def ensure_model(path: str | Path | None = None) -> Path:
    target = Path(path) if path else Path.home() / ".cache" / "pvdata" / "face_landmarker.task"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, target)
    return target


def euler_from_matrix(m: np.ndarray) -> tuple[float, float]:
    """(yaw, pitch) in degrees from a 4x4 (or 3x3) head pose matrix."""
    r = np.asarray(m)[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    return yaw, pitch


def observe(landmarks_px: np.ndarray, matrix: np.ndarray | None, frame_h: int, n_faces: int) -> FaceObs:
    """Build a FaceObs from one face's (478, 2) pixel landmarks."""
    mouth_w = float(np.linalg.norm(landmarks_px[MOUTH_RIGHT] - landmarks_px[MOUTH_LEFT]))
    gap = float(np.linalg.norm(landmarks_px[LIP_BOTTOM] - landmarks_px[LIP_TOP]))
    face_h = float(np.linalg.norm(landmarks_px[CHIN] - landmarks_px[FOREHEAD]))
    yaw, pitch = euler_from_matrix(matrix) if matrix is not None else (0.0, 0.0)
    return FaceObs(n_faces, yaw, pitch, face_h / max(frame_h, 1), mouth_w, gap / max(mouth_w, 1e-6))


class MediaPipeFaces:
    """Callable: RGB frame -> FaceObs (n_faces=0 when nothing is found). Uses VIDEO mode, so feed frames in order."""

    def __init__(self, model_path: str | Path | None = None, fps: float = 25.0):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=3,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(options)
        self._step_ms = 1000.0 / fps
        self._t = 0

    def __call__(self, frame_rgb: np.ndarray) -> FaceObs:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        res = self._lm.detect_for_video(image, int(self._t * self._step_ms))
        self._t += 1
        if not res.face_landmarks:
            return FaceObs(0)
        h, w = frame_rgb.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
        matrix = np.array(res.facial_transformation_matrixes[0]) if res.facial_transformation_matrixes else None
        return observe(pts, matrix, h, len(res.face_landmarks))

    def close(self) -> None:
        self._lm.close()


@dataclass
class FaceReport:
    ok: bool
    reason: str
    good_ratio: float
    openings: list[float] = field(default_factory=list)  # per frame, NaN where no usable face
    median_yaw: float = 0.0
    median_pitch: float = 0.0


def frame_problem(o: FaceObs, cfg: FaceConfig) -> str | None:
    if o.n_faces == 0:
        return "no_face"
    if o.n_faces > cfg.max_faces:
        return "multiple_faces"
    if abs(o.yaw) > cfg.max_yaw_deg or abs(o.pitch) > cfg.max_pitch_deg:
        return "not_frontal"
    if o.face_h_frac < cfg.min_face_height_frac:
        return "face_too_small"
    if o.mouth_w_px < cfg.min_mouth_px:
        return "mouth_too_small"
    return None


def scan_frames(detector, frames, cfg: FaceConfig) -> list[FaceObs]:
    """Run the detector over a clip, stopping as soon as too many frames have failed for the clip to pass.

    With ``min_detect_ratio`` 0.95 a 50-frame clip may fail at most 2 frames, so stock footage and cutaways
    are rejected after about 3 frames instead of 50. The returned partial list still fails ``check_clip``.
    """
    budget = int(len(frames) * (1 - cfg.min_detect_ratio) + 1e-9)
    obs, bad = [], 0
    for f in frames:
        o = detector(f)
        obs.append(o)
        if frame_problem(o, cfg) is not None:
            bad += 1
            if bad > budget:
                break
    return obs


def check_clip(observations: list[FaceObs], cfg: FaceConfig) -> FaceReport:
    if not observations:
        return FaceReport(False, "no_frames", 0.0)
    problems = [frame_problem(o, cfg) for o in observations]
    good = [p is None for p in problems]
    ratio = sum(good) / len(good)
    openings = [o.opening if g else float("nan") for o, g in zip(observations, good)]
    usable = [o for o, g in zip(observations, good) if g]
    yaw = float(np.median([o.yaw for o in usable])) if usable else 0.0
    pitch = float(np.median([o.pitch for o in usable])) if usable else 0.0
    if ratio >= cfg.min_detect_ratio:
        return FaceReport(True, "", ratio, openings, yaw, pitch)
    worst = Counter(p for p in problems if p is not None).most_common(1)[0][0]
    return FaceReport(False, worst, ratio, openings, yaw, pitch)
