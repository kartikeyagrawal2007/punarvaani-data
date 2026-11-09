# punarvaani-data

> 📘 **New here? Read [`docs/DATA_ENGINE_GUIDE.md`](docs/DATA_ENGINE_GUIDE.md)**: aim, pipeline diagrams, what's done and what isn't, your step-by-step tasks, improvements and interview Q&A.

**The Hindi lip-reading dataset builder for [Punarvaani](https://github.com/kartikeyagrawal2007/punarvaani).**
It turns YouTube videos of one person talking to the camera into short, clean, word-aligned clips with Devanagari transcripts, in exactly the format the Stage-1 trainer expects (`TEAM_GUIDE.md`, Part 1).

```
sources.csv ─► download (yt-dlp) ─► 16 kHz audio (ffmpeg) ─► ASR + word timestamps (Whisper / WhisperX / IndicWhisper)
            ─► 1.5–2.5 s word-aligned candidates
            ─► transcript rules ─► ASR confidence ─► CTC length (U+R ≤ 32)
            ─► one frontal face, big enough, mouth visible (MediaPipe) ─► audio–mouth sync
            ─► raw_videos/spk01/spk01_001.mp4 + spk01_001.txt   (+ manifest, report, speaker-disjoint splits)
```

## What you get (the contract with the main repo)

```
dataset/
├── raw_videos/                  ← give this to scripts/preprocess_lips.py in the main repo
│   ├── spk01/
│   │   ├── spk01_001.mp4        720p, 25 fps, H.264, 1.5–2.5 s, audio kept
│   │   ├── spk01_001.txt        UTF-8, Devanagari only, numbers spelled out, no punctuation except ।
│   │   └── ...
│   └── spk02/ ...
├── manifest.csv                 every candidate clip: source, times, texts, scores, accepted/rejected + reason
├── report.json                  totals, hours, clips per speaker, rejections by reason
├── speakers.json                spk01 → the speaker name you used in sources.csv
├── splits.json                  speaker-disjoint train / val / test
└── splits/{train,val,test}/     symlinks to the speaker folders (for the trainer's dedicated-split mode)
```

Rules enforced, from `TEAM_GUIDE.md`:

| Rule | How it is enforced |
|---|---|
| 1.5–2.5 s per clip, about 2–6 words | `segment.py` cuts at word boundaries, never through a long pause |
| Stage 1 maps 16 frames → 32 CTC steps; a clip needs **U + R ≤ 32** | `ctc.py` uses the **same G2P file as the trainer** (`pvdata/g2p/hindi_g2p.py`, vendored) |
| Pure Devanagari, numbers spelled out, no punctuation except । | `text_norm.py` (`२०२५` → `दो हज़ार पच्चीस`); clips with English words are dropped |
| Frontal face, mouth unobstructed, crop success ≥ 95% | `face.py`: one face, yaw/pitch ≤ 20°, face ≥ 18% of frame height, mouth ≥ 40 px, in ≥ 95% of frames |
| ≥ 720p, 25/30 fps | sources below 720p are skipped; clips are written at 720p, 25 fps |
| No speaker leakage between train and val | `pvdata split` assigns whole speakers to train/val/test |

## Install

```bash
git clone <this repo> && cd punarvaani-data
python -m venv .venv && source .venv/bin/activate
pip install -e ".[download,asr,dev]"     # yt-dlp + faster-whisper
# ffmpeg must be installed:  macOS: brew install ffmpeg   Ubuntu/Colab: apt-get install -y ffmpeg
pytest -q                                 # 46 tests, ~45 s, no internet needed
```

ASR is the slow part: use a GPU (Colab/Kaggle) for real runs. Cutting clips and face checks run fine on a laptop.

## Use it

**1. List your videos** in a CSV (copy `sources/example_sources.csv`):

```csv
url,speaker,start,end,notes
https://www.youtube.com/watch?v=XXXX,anchor_a,0:30,12:00,DD News bulletin - anchor only
https://www.youtube.com/watch?v=YYYY,lecturer_b,,,NPTEL Hindi lecture
```
- One row per video. `speaker` is your short name for the person on screen. Use the **same name** for the same person across videos; they become `spk01`, `spk02`, ... in order.
- `start`/`end` (optional, seconds or mm:ss) keep only the part where that person faces the camera.

Or let it search YouTube for you: put one person per row in `sources/speakers.csv` (`speaker,query`), then
```bash
pvdata find --queries sources/speakers.csv --out sources/my_sources.csv --per-speaker 3
```
and **review every row**: delete panels, compilations, other people and mostly-English videos.

**2. Mine clips:**
```bash
pvdata run --sources sources/my_sources.csv --out dataset/            # add --limit 2 for a quick first test
```
Everything is cached in `dataset/work/` (downloads, audio, ASR), so you can stop and rerun at any time; finished videos are skipped.

**3. Check and split:**
```bash
pvdata validate dataset/raw_videos      # must end with "OK"
pvdata split --out dataset/             # speaker-disjoint train/val/test
pvdata report --out dataset/            # how many clips, and why the others were dropped
```

**4. Hand over to Stage 1** (main repo):
```bash
python scripts/preprocess_lips.py --input_dir /path/to/dataset/raw_videos --output_dir ./hindi_dataset --num_frames 16 --frame_size 224
```

### Colab (GPU) in four cells
```python
!git clone <this repo> && cd punarvaani-data && pip install -q -e ".[download,asr]" && apt-get -qq install -y ffmpeg
%cd punarvaani-data
!pvdata run --sources sources/my_sources.csv --out /content/drive/MyDrive/punarvaani_dataset
!pvdata validate /content/drive/MyDrive/punarvaani_dataset/raw_videos && pvdata split --out /content/drive/MyDrive/punarvaani_dataset
```

## Autopilot: grow to 100 hours without reviewing videos (`pvdata auto`)

```bash
pvdata screen "https://www.youtube.com/watch?v=XXXX"            # dry run: would autopilot take this video?
pvdata auto --seeds sources/seeds.csv --out dataset/ --max-videos 5   # short test
pvdata auto --seeds sources/seeds.csv --out dataset/               # until 100 h (auto.target_hours)
pvdata identify --out dataset/ --quarantine                        # who is who: merge persons, remove guests
pvdata split --out dataset/                                        # now keeps each person in one split
```

```
seed queries ─► discover: YouTube search, length 4–60 min, title filter (debate, panel, song, shorts ... dropped)
             ─► screen:   3 × 20 s samples at 360p (a few MB): one frontal face in ≥ 50% of frames,
                          Whisper says Hindi, ≥ 85% Devanagari words, someone talking
             ─► mine:     the normal pipeline (720p download, next video downloads in the background,
                          full video deleted afterwards)
             ─► learn:    a channel that gives ≥ 1 accepted min/video after 2 videos gets its other uploads queued
caps per channel (25 videos, 3 h) keep any one person from dominating; best-scoring videos are mined first
```

- Every video is its own speaker folder (`<channel>_<videoid>` in `speakers.json`). **`pvdata identify`** embeds the faces (OpenCV YuNet + SFace), merges folders of the same person into one entry of `identity/persons.json`, and with `--quarantine` moves clips of other faces (guests, co-anchors) to `quarantine/`. `pvdata split` then never puts one person in two splits.
- **Hand over part of it while mining runs:** `pvdata snapshot --out dataset/ --to handover/ --hours 2` copies a balanced subset of finished clips (every person up to the same cap; clips still being written are skipped; `--exclude` an earlier handover to send only new clips). Then run `identify`, `split` and `validate` on `handover/`.
- Stop with Ctrl+C at any time; rerunning continues. Everything decided is in `dataset/auto/state.json` (status and reason for every video seen).
- Add seed queries any time (especially for women speakers and regional accents). Each round searches deeper (top 50 → 100 → 200 → 400).
- Disk: clips only (the full videos are deleted). Check `raw_videos` size ÷ number of clips × the clips you expect.

## Choosing videos (this decides the quality more than any setting)

- **One person, facing the camera, mouth always visible:** news anchors in studio, lecturers at a desk, talking-head vlogs, close-up interview answers.
- Avoid: panels and debates (several faces), B-roll and voice-over, heavy background music, side angles, masks or mics covering the mouth.
- Many speakers beat many hours of one speaker: aim for **10+ speakers**. Prototype: 300–500 clips; benchmark: 1,500+ (TEAM_GUIDE).
- Use `start`/`end` to cut out intros, ads and B-roll before processing.
- Check each channel's terms of use and note it in `notes`. **Never upload the clips publicly**: faces and voices are personal data. Share them only privately within the team.

## Building it for the dataset paper

We plan to publish the dataset itself (like LRS3, which was built from TED talks). That changes three things:

**1. Prefer openly licensed, many-speaker sources.** `pvdata find` supports `@name` rows (see `sources/speakers.csv`): every video becomes its own speaker, which is ideal for TEDx, NPTEL and Josh Talks.

| Source | Why | Licence (verify per video/course) |
|---|---|---|
| **NPTEL** Hindi lectures | Hundreds of teachers, clear Hindi, long takes | Creative Commons (NPTEL states CC BY-SA); check each course page |
| **TED / TEDx** Hindi talks | Many speakers, diverse ages, accents, genders | TED usage policy: CC BY-NC-ND 4.0 (non-commercial research) |
| Josh Talks (Hindi) | Many speakers, one person on stage | Standard YouTube licence: release IDs and timestamps only |
| News/journalist channels | Long, clean talking-head takes | Standard YouTube licence: release IDs and timestamps only |

NPTEL often shows slides with a small inset of the teacher, and TEDx cuts to wide stage shots and the audience. The face checks drop those parts automatically, so expect a lower acceptance rate.

**2. Provenance is recorded for every clip.** `manifest.csv` stores `license`, `channel`, `title` and `upload_date` (from yt-dlp metadata), and `report.json` counts accepted clips per licence. That's the raw material for the datasheet.

**3. Release format.** Publish **video IDs + start/end times + transcripts + this code**, not the video files, as LRS3, AVSpeech and VoxCeleb did. Users rebuild the clips themselves. Never publish volunteer or patient recordings without written consent covering release.

For the paper, also report:
- the filter funnel (`rejected_by_reason`);
- hours and clips per speaker;
- speaker counts per split;
- a hand-checked sample of about 300 transcripts (transcript accuracy);
- baseline Stage-1 results on the fixed speaker-disjoint test split.

## Why clips get rejected (`report.json`)

| Reason | Meaning | What to do |
|---|---|---|
| `text_contains_latin` | English words in the transcript | Normal for Hinglish; fine to lose. Set `text.reject_latin: false` to strip them instead (less clean) |
| `low_asr_confidence` | Whisper unsure of a word | Try `--asr-model` with IndicWhisper (`asr.backend: hf`) for better Hindi |
| `too_many_phonemes` | Too much speech for 32 CTC steps | Expected for fast speakers |
| `face_no_face` / `face_multiple_faces` / `face_not_frontal` / `face_face_too_small` / `face_mouth_too_small` | Fails the face rules in ≥ 5% of frames | Pick tighter `start`/`end`, or better videos |
| `sync_weak_av_correlation` / `sync_av_offset` | Mouth doesn't match the audio (voice-over, dubbing, B-roll) | Usually correct to drop |
| `source_below_720p` | Video is below 720p | Find an HD upload |

Thresholds are in `configs/default.yaml`; copy it and pass `--config my.yaml` to change them.

## Code map

| File | What |
|---|---|
| `pvdata/sources.py` | reads the sources CSV, assigns spk01, spk02, ... |
| `pvdata/media.py` | yt-dlp download, ffprobe, audio extraction, frame reading, clip cutting (ffmpeg) |
| `pvdata/asr.py` | faster-whisper / WhisperX / Hugging Face Whisper (IndicWhisper) with word timestamps |
| `pvdata/segment.py` | words → 1.5–2.5 s candidates |
| `pvdata/text_norm.py` | TEAM_GUIDE transcript rules, Hindi number words |
| `pvdata/ctc.py` + `pvdata/g2p/` | the trainer's CTC length rule with the trainer's own G2P |
| `pvdata/face.py` | MediaPipe face count, head pose, face/mouth size, mouth opening |
| `pvdata/sync.py` | audio-loudness vs mouth-opening correlation (a light stand-in for SyncNet) |
| `pvdata/pipeline.py` | runs everything per video, resumable; downloads the next video in the background |
| `pvdata/find.py` | YouTube search and channel listing (metadata only) |
| `pvdata/screen.py` | automatic review of a video from 3 short low-res samples |
| `pvdata/auto.py` | the autopilot loop: discover → screen → mine → expand good channels, with caps |
| `pvdata/identity.py` | face embeddings → persons, other-face clips, cached |
| `pvdata/export.py`, `pvdata/validate.py` | output layout, manifest, report, splits, validation |

## Notes

- `pvdata/g2p/hindi_g2p.py` is a copy of `Data/hindi_g2p.py` from the main repo. If that file changes there, copy it again so the CTC check stays identical to training.
- The sync check is deliberately light. For a large final run, [SyncNet](https://github.com/joonson/syncnet_python) can replace it.
- Licence: MIT for this code. The videos you mine keep their own copyright.
