"""SyncNet active-speaker check: is the person on screen the one whose voice we hear?

The light correlation check in ``sync.py`` asks only "does the mouth move when the audio is loud?". On a
1.5-3 s clip, two unrelated speech signals reach a correlation of 0.25-0.30 by chance, so a news anchor's
voice-over on B-roll of a politician who is himself talking passes it. SyncNet (Chung & Zisserman, 2016,
the model used to build LRS2/LRS3 and VoxCeleb) embeds 5-frame lip windows and 0.2 s of MFCC audio in a
shared space and measures how well they match at every shift of +-15 frames:

* ``conf`` = median distance - minimum distance. A real talking head has one sharp minimum (conf ~5-10);
  voice-over, dubbing or another speaker gives a flat curve (conf ~0-2).
* ``offset`` = the shift of that minimum, in video frames at 25 fps.

This file is a dependency-light port of joonson/syncnet_python (MIT): the same network (``S``), the same
face crop (box from the face check, 40 % margin, 224x224, BGR) and the same MFCC front end, so the
published ``syncnet_v2.model`` weights load unchanged. torch and python_speech_features are imported only
when a scorer is built (``pip install -e .[syncnet]``).
"""

from __future__ import annotations

import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WEIGHTS_URL = "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model"
FPS = 25  # SyncNet was trained on 25 fps video: one video frame = 4 MFCC frames (100 Hz)
SAMPLE_RATE = 16000
WINDOW = 5  # video frames per lip window (0.2 s)


@dataclass
class SyncNetResult:
    conf: float
    offset: int  # best audio/video shift in video frames at 25 fps (0 = in sync)
    min_dist: float
    windows: int


def ensure_weights(path: str | Path | None = None) -> Path:
    """Path to syncnet_v2.model, downloading it to ~/.cache/pvdata on first use (about 52 MB)."""
    target = Path(path) if path else Path.home() / ".cache" / "pvdata" / "syncnet_v2.model"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)
        tmp.replace(target)
    return target


def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """1-D median filter with edge replication (scipy's medfilt pads with zeros, which shrinks short tracks)."""
    x = np.asarray(x, dtype=np.float64)
    k = max(1, min(k, len(x) if len(x) % 2 else len(x) - 1))
    if k <= 1:
        return x.copy()
    r = k // 2
    p = np.pad(x, r, mode="edge")
    return np.array([np.median(p[i : i + k]) for i in range(len(x))])


def fill_boxes(boxes: list) -> np.ndarray | None:
    """Replace missing (None) boxes by the nearest known one; None if no frame has a box."""
    idx = [i for i, b in enumerate(boxes) if b is not None]
    if not idx:
        return None
    out = np.zeros((len(boxes), 4), dtype=np.float64)
    for i in range(len(boxes)):
        j = min(idx, key=lambda k: abs(k - i))
        out[i] = boxes[j]
    return out


def crop_faces(frames_rgb: np.ndarray, boxes: np.ndarray, crop_scale: float = 0.4, size: int = 224) -> np.ndarray:
    """syncnet_python's crop: smoothed box centre and size, 40 % margin (more below the chin), grey padding.

    Returns (T, size, size, 3) uint8 in BGR order, which is what the published weights were trained on."""
    import cv2

    boxes = np.asarray(boxes, dtype=np.float64)
    s = median_filter(np.maximum(boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]) / 2, 13)
    x = median_filter((boxes[:, 0] + boxes[:, 2]) / 2, 13)
    y = median_filter((boxes[:, 1] + boxes[:, 3]) / 2, 13)
    out = np.zeros((len(frames_rgb), size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(frames_rgb):
        bs, cs = max(s[i], 1.0), crop_scale
        pad = int(bs * (1 + 2 * cs))
        img = cv2.copyMakeBorder(frame[..., ::-1], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(110, 110, 110))
        my, mx = y[i] + pad, x[i] + pad
        y1, y2 = int(my - bs), int(my + bs * (1 + 2 * cs))
        x1, x2 = int(mx - bs * (1 + cs)), int(mx + bs * (1 + cs))
        out[i] = cv2.resize(img[y1:y2, x1:x2], (size, size))
    return out


def mfcc_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """13 x N MFCC at 100 Hz from float audio in [-1, 1] (scaled to int16 range, as syncnet_python reads wavs)."""
    import python_speech_features

    return np.asarray(python_speech_features.mfcc(np.asarray(audio, dtype=np.float64) * 32768.0, sr)).T


def offset_confidence(vfeat: np.ndarray, afeat: np.ndarray, vshift: int = 15) -> tuple[int, float, float]:
    """SyncNet's calc_pdist + decision: mean distance per shift -> (offset, conf, min_dist)."""
    vfeat, afeat = np.asarray(vfeat, np.float64), np.asarray(afeat, np.float64)
    win = 2 * vshift + 1
    ap = np.pad(afeat, ((vshift, vshift), (0, 0)))
    dists = np.stack([np.linalg.norm(vfeat[i][None, :] - ap[i : i + win] + 1e-6, axis=1) for i in range(len(vfeat))])
    mdist = dists.mean(axis=0)
    minidx = int(np.argmin(mdist))
    return vshift - minidx, float(np.median(mdist) - mdist[minidx]), float(mdist[minidx])


def build_model(num_out: int = 1024):
    """The SyncNet 'S' network with syncnet_python's module names, so its state dict loads as is."""
    from torch import nn

    class S(nn.Module):
        def __init__(self):
            super().__init__()
            self.netcnnaud = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
                nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
                nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            self.netfcaud = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, num_out))
            self.netfclip = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, num_out))
            self.netcnnlip = nn.Sequential(
                nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
                nn.BatchNorm3d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
                nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
                nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
                nn.BatchNorm3d(512),
                nn.ReLU(inplace=True),
            )

        def forward_aud(self, x):
            mid = self.netcnnaud(x)
            return self.netfcaud(mid.view(mid.size(0), -1))

        def forward_lip(self, x):
            mid = self.netcnnlip(x)
            return self.netfclip(mid.view(mid.size(0), -1))

    return S()


class SyncNetScorer:
    """Callable: (frames_rgb at 25 fps, face boxes, 16 kHz audio) -> SyncNetResult. Thread safe."""

    def __init__(self, weights: str | Path | None = None, device: str = "auto", batch: int = 32, vshift: int = 15,
                 model=None):  # fmt: skip
        import torch

        self._torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.batch, self.vshift = batch, vshift
        self.model = model if model is not None else self._load(ensure_weights(weights))
        self.model.to(self.device).eval()
        self._lock = threading.Lock()

    @staticmethod
    def _load(path: Path):
        import torch

        model = build_model()
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            state = state.state_dict()
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        if missing or unexpected:
            raise ValueError(f"{path} is not a SyncNet v2 state dict (missing {missing[:3]}, unexpected "
                             f"{list(unexpected)[:3]})")  # fmt: skip
        return model

    def embed(self, crops_bgr: np.ndarray, mfcc: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        torch = self._torch
        n = min(len(crops_bgr), mfcc.shape[1] // 4) - WINDOW
        if n < 1:
            return None
        video = torch.from_numpy(np.ascontiguousarray(crops_bgr.transpose(3, 0, 1, 2))).float()  # 3,T,H,W
        audio = torch.from_numpy(np.ascontiguousarray(mfcc)).float()[None]  # 1,13,N
        vf, af = [], []
        with self._lock, torch.no_grad():
            for i in range(0, n, self.batch):
                idx = range(i, min(n, i + self.batch))
                v = torch.stack([video[:, j : j + WINDOW] for j in idx]).to(self.device)
                a = torch.stack([audio[:, :, 4 * j : 4 * j + 4 * WINDOW] for j in idx]).to(self.device)
                vf.append(self.model.forward_lip(v).cpu())
                af.append(self.model.forward_aud(a).cpu())
        return torch.cat(vf).numpy(), torch.cat(af).numpy()

    def __call__(self, frames_rgb: np.ndarray, boxes: list, audio: np.ndarray) -> SyncNetResult | None:
        filled = fill_boxes(boxes)
        if filled is None:
            return None
        feats = self.embed(crop_faces(frames_rgb, filled), mfcc_features(audio))
        if feats is None:
            return None
        offset, conf, dmin = offset_confidence(*feats, vshift=self.vshift)
        return SyncNetResult(conf, offset, dmin, len(feats[0]))
