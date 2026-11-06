"""Who is in each clip, from faces: merges one person's videos and catches clips of somebody else.

`pvdata auto` mines every video into its own speaker folder (one video = one provisional speaker), because
it can't know who is on screen. This step fixes that afterwards:

    every clip ─► 3 frames ─► OpenCV YuNet face ─► SFace 128-d embedding (averaged, L2-normalised)
    per folder:  cluster the clips; the biggest cluster is the folder's speaker, the rest are "other faces"
                 (a guest, a cut to another anchor) and can be quarantined
    across folders: cluster the folder speakers -> persons; persons.json lists each person's folders

`pvdata split` then keeps all folders of one person in the same split, so the same face can never be in both
train and test. Embeddings are cached in <out>/identity/, so reruns only embed new clips.

Models (Apache-2.0, OpenCV Zoo) are downloaded once to ~/.cache/pvdata.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np

log = logging.getLogger("pvdata")

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "yunet": (f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx", "face_detection_yunet_2023mar.onnx"),
    "sface": (f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx", "face_recognition_sface_2021dec.onnx"),
}  # fmt: skip


def ensure_model(name: str, cache: Path | None = None) -> Path:
    url, fname = MODELS[name]
    target = (cache or Path.home() / ".cache" / "pvdata") / fname
    if not target.exists() or target.stat().st_size < 10_000:  # a tiny file is an error page, not a model
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target)
    return target


class SFaceEmbedder:
    """Callable: clip path -> unit-length face embedding (or None if no face was found)."""

    def __init__(self, frames_per_clip: int = 3):
        import cv2

        self.cv2, self.n = cv2, frames_per_clip
        self.det = cv2.FaceDetectorYN.create(str(ensure_model("yunet")), "", (320, 320), 0.7, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(ensure_model("sface")), "")

    def frame_embedding(self, bgr: np.ndarray) -> np.ndarray | None:
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])  # the largest face
        feat = self.rec.feature(self.rec.alignCrop(bgr, face)).reshape(-1).astype(np.float32)
        n = float(np.linalg.norm(feat))
        return feat / n if n > 0 else None

    def __call__(self, clip: Path) -> np.ndarray | None:
        cap = self.cv2.VideoCapture(str(clip))
        total = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT)) or 1
        feats = []
        for k in range(self.n):
            cap.set(self.cv2.CAP_PROP_POS_FRAMES, int(total * (k + 1) / (self.n + 1)))
            ok, frame = cap.read()
            if ok:
                f = self.frame_embedding(frame)
                if f is not None:
                    feats.append(f)
        cap.release()
        if not feats:
            return None
        m = np.mean(feats, axis=0)
        return m / max(float(np.linalg.norm(m)), 1e-9)


def leader_cluster(vectors: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy clustering of unit vectors: join the most similar cluster if cosine >= threshold, else start one.
    O(n * clusters), fine for the clips of one video or the speakers of a whole dataset."""
    labels = np.full(len(vectors), -1)
    sums: list[np.ndarray] = []
    for i, v in enumerate(vectors):
        if sums:
            cents = np.stack([s / max(float(np.linalg.norm(s)), 1e-9) for s in sums])
            sims = cents @ v
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                labels[i] = j
                sums[j] = sums[j] + v
                continue
        labels[i] = len(sums)
        sums.append(v.copy())
    return labels


class EmbeddingCache:
    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / "embeddings.npz"
        self.data: dict[str, np.ndarray | None] = {}
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            ids, vecs, has = list(z["ids"]), z["vecs"], z["has"]
            self.data = {str(i): (vecs[k] if has[k] else None) for k, i in enumerate(ids)}

    def save(self) -> None:
        ids = list(self.data)
        dim = next((v.shape[0] for v in self.data.values() if v is not None), 128)
        vecs = np.stack([v if v is not None else np.zeros(dim, np.float32) for v in self.data.values()]) if ids \
            else np.zeros((0, dim), np.float32)  # fmt: skip
        has = np.array([v is not None for v in self.data.values()], bool)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.array(ids, dtype=str), vecs=vecs, has=has)
        tmp.replace(self.path)


def identify(out: str | Path, embedder: Callable[[Path], np.ndarray | None], same_person: float = 0.40,
             quarantine: bool = False, save_every: int = 500) -> dict:  # fmt: skip
    """Embed all clips, find each folder's speaker, merge folders into persons, flag other faces."""
    out = Path(out)
    raw = out / "raw_videos"
    cache = EmbeddingCache(out / "identity")
    clips = sorted(raw.glob("spk*/*.mp4"))
    new = [c for c in clips if c.stem not in cache.data]
    for k, clip in enumerate(new, 1):
        cache.data[clip.stem] = embedder(clip)
        if k % save_every == 0:
            cache.save()
            log.info("identity: embedded %d / %d new clips", k, len(new))
    cache.save()

    rows, folder_vec, folder_n = [], {}, {}
    for folder in sorted(p for p in raw.iterdir() if p.is_dir()):
        ids = [c.stem for c in sorted(folder.glob("*.mp4"))]
        have = [i for i in ids if cache.data.get(i) is not None]
        for i in ids:
            if cache.data.get(i) is None:
                rows.append({"clip_id": i, "folder": folder.name, "status": "no_face", "similarity": ""})
        if not have:
            continue
        vecs = np.stack([cache.data[i] for i in have])
        labels = leader_cluster(vecs, same_person)
        main = int(np.bincount(labels).argmax())
        cent = vecs[labels == main].mean(axis=0)
        cent /= max(float(np.linalg.norm(cent)), 1e-9)
        folder_vec[folder.name], folder_n[folder.name] = cent, int((labels == main).sum())
        for i, lab, v in zip(have, labels, vecs):
            rows.append({"clip_id": i, "folder": folder.name, "status": "speaker" if lab == main else "other_face",
                         "similarity": round(float(v @ cent), 3)})  # fmt: skip

    # biggest folders first, so each person's centroid is seeded by their best-covered video
    names = sorted(folder_vec, key=lambda f: -folder_n[f])
    persons: dict[str, dict] = {}
    if names:
        labels = leader_cluster(np.stack([folder_vec[f] for f in names]), same_person)
        order = {lab: k for k, lab in enumerate(dict.fromkeys(labels))}
        for f, lab in zip(names, labels):
            pid = f"person{order[lab] + 1:04d}"
            p = persons.setdefault(pid, {"folders": [], "clips": 0})
            p["folders"].append(f)
            p["clips"] += folder_n[f]
    for p in persons.values():
        p["folders"].sort()

    moved = 0
    if quarantine:
        for r in rows:
            if r["status"] in ("other_face", "no_face"):
                for ext in (".mp4", ".txt"):
                    src = raw / r["folder"] / f"{r['clip_id']}{ext}"
                    if src.exists():
                        dst = out / "quarantine" / r["folder"] / src.name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(src), str(dst))
                moved += 1

    with open(out / "identity" / "clips.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["clip_id", "folder", "status", "similarity"])
        w.writeheader()
        w.writerows(rows)
    (out / "identity" / "persons.json").write_text(json.dumps(persons, indent=2) + "\n", encoding="utf-8")
    status = [r["status"] for r in rows]
    return {
        "clips": len(rows),
        "speaker": status.count("speaker"),
        "other_face": status.count("other_face"),
        "no_face": status.count("no_face"),
        "folders": len(folder_vec),
        "persons": len(persons),
        "quarantined": moved,
    }


def load_persons(out: str | Path) -> dict[str, list[str]] | None:
    """person -> folders, from an earlier `pvdata identify` (None if it hasn't been run)."""
    p = Path(out) / "identity" / "persons.json"
    if not p.exists():
        return None
    return {k: v["folders"] for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
