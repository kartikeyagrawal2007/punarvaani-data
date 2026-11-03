import numpy as np

from pvdata.config import FaceConfig, SyncConfig
from pvdata.face import FaceObs, check_clip, euler_from_matrix
from pvdata.sync import loudness_envelope, sync_score


def good(opening=0.2):
    return FaceObs(1, yaw=2, pitch=-3, face_h_frac=0.4, mouth_w_px=120, opening=opening)


def test_clip_passes_with_one_frontal_face():
    rep = check_clip([good() for _ in range(50)], FaceConfig())
    assert rep.ok and rep.good_ratio == 1.0


def test_clip_fails_for_turned_head_or_two_faces():
    turned = [FaceObs(1, yaw=40, pitch=0, face_h_frac=0.4, mouth_w_px=120) for _ in range(10)]
    rep = check_clip([good() for _ in range(40)] + turned, FaceConfig())
    assert not rep.ok and rep.reason == "not_frontal"
    two = [FaceObs(2, face_h_frac=0.4, mouth_w_px=120) for _ in range(50)]
    assert check_clip(two, FaceConfig()).reason == "multiple_faces"


def test_euler_identity_is_frontal():
    assert euler_from_matrix(np.eye(4)) == (0.0, 0.0)


def _signals(n=60, shift=0):
    t = np.arange(n)
    env = np.clip(np.sin(t / 3.0), 0, None) + 0.05
    mouth = np.roll(env, shift) * 0.3
    return mouth, env


def test_sync_accepts_aligned_and_rejects_shifted():
    cfg = SyncConfig()
    mouth, env = _signals()
    r = sync_score(mouth, env, cfg)
    assert r.ok and r.offset_frames == 0 and r.corr > 0.9
    mouth, env = _signals(shift=5)
    r = sync_score(mouth, env, SyncConfig(max_lag_frames=8))
    assert not r.ok and r.reason == "av_offset"


def test_sync_rejects_still_mouth_and_silence():
    cfg = SyncConfig()
    _, env = _signals()
    assert sync_score(np.full(60, 0.2), env, cfg).reason == "mouth_not_moving"
    mouth, _ = _signals()
    assert sync_score(mouth, np.zeros(60), cfg).reason == "no_audio_energy"


def test_loudness_envelope_length():
    audio = np.random.default_rng(0).normal(size=16000).astype(np.float32)
    assert loudness_envelope(audio, 16000, 25, 25).shape == (25,)
