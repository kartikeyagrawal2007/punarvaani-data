# punarvaani-data

> 📘 **New here? Read [`docs/DATA_ENGINE_GUIDE.md`](docs/DATA_ENGINE_GUIDE.md)**: aim, pipeline diagrams, what's done and what isn't, your step-by-step tasks, improvements and interview Q&A.

**The Hindi lip-reading dataset builder for [Punarvaani](https://github.com/kartikeyagrawal2007/punarvaani).**
It turns YouTube videos of one person talking to the camera into short, clean, word-aligned clips with Devanagari transcripts, in exactly the format the Stage-1 trainer expects (`TEAM_GUIDE.md`, Part 1).

```
sources.csv ─► download (yt-dlp) ─► 16 kHz audio (ffmpeg) ─► ASR + word timestamps (Whisper / WhisperX / IndicWhisper)
            ─► 1.5–3 s word-aligned candidates
            ─► transcript rules ─► ASR confidence ─► CTC length (U+R ≤ 32)
            ─► one frontal face, big enough, mouth visible (MediaPipe) ─► audio–mouth sync
            ─► raw_videos/spk01/spk01_001.mp4 + spk01_001.txt   (+ manifest, report, speaker-disjoint splits)
```

## What you get (the contract with the main repo)

```
dataset/
├── raw_videos/                  ← give this to scripts/preprocess_lips.py in the main repo
│   ├── spk01/
│   │   ├── spk01_001.mp4        720p, source frame rate (24–30 fps), H.264, 1.5–3 s, audio kept
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
| 1.5–3 s per clip (trainer reads all frames), 2–10 words | `segment.py` cuts at word boundaries, never through a long pause |
| Stage 1 gives a clip ceil(frames/2)×4 CTC steps; it needs **U + R ≤ that** | `ctc.py` uses the **same G2P file as the trainer** (`pvdata/g2p/hindi_g2p.py`, vendored) |
| Pure Devanagari, numbers spelled out, no punctuation except । | `text_norm.py` (`२०२५` → `दो हज़ार पच्चीस`); clips with English words are dropped |
| Frontal face, mouth unobstructed, crop success ≥ 95% | `face.py`: one face, yaw/pitch ≤ 20°, face ≥ 18% of frame height, mouth ≥ 40 px, in ≥ 95% of frames |
| ≥ 720p, 25/30 fps | sources below 720p are skipped; clips are written at 720p, 25 fps |
| No speaker leakage between train and val | `pvdata split` assigns whole speakers to train/val/test |

## Install

```bash
git clone <this repo> && cd punarvaani-data
python -m venv .venv && source .venv/bin/activate
pip install -e ".[download,asr,syncnet,dev]"   # yt-dlp + faster-whisper + SyncNet
# ffmpeg must be installed:  macOS: brew install ffmpeg   Ubuntu/Colab: apt-get install -y ffmpeg
pytest -q                                 # 51 tests, ~60 s, no internet needed
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
!git clone <this repo> && cd punarvaani-data && pip install -q -e ".[download,asr,syncnet]" && apt-get -qq install -y ffmpeg
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
- **Remove bad videos (animation, avatars, wrong person):** `pvdata review --out dataset/` writes `dataset/review/index.html` with one thumbnail strip per video; tick the bad ones, stop `auto`, run the `pvdata exclude ...` command the page shows, restart `auto`. Their clips move to `excluded/` and their channels are never mined again.
- **Hand over part of it while mining runs:** `pvdata snapshot --out dataset/ --to handover/ --hours 2` copies a balanced subset of finished clips (every person up to the same cap; clips still being written are skipped; `--exclude` an earlier handover to send only new clips). Then run `identify`, `split` and `validate` on `handover/`.
- Stop with Ctrl+C at any time; rerunning continues. Everything decided is in `dataset/auto/state.json` (status and reason for every video seen).
- Add seed queries any time (especially for women speakers and regional accents). Each round searches deeper (top 50 → 100 → 200 → 400).
- Disk: clips only (the full videos are deleted). Check `raw_videos` size ÷ number of clips × the clips you expect.
- **Several PCs at once:** give every PC the same seeds and a `--shard`: this PC's number / number of PCs. Each PC then owns a fixed share of YouTube channels (a checksum of the channel id), so two PCs never mine the same channel or video, and they need no connection to each other. The main PC keeps its `dataset/` and adds `--shard 1/4`; the others mine into their own folder:
  ```bash
  pvdata auto --seeds sources/seeds.csv --out dataset/ --shard 1/4          # main PC
  pvdata auto --seeds sources/seeds.csv --out dataset_pc2/ --shard 2/4      # PC 2 (and 3/4, 4/4 on the others)
  ```
  Every few days copy each PC's folder to the main PC (USB drive, AnyDesk or a shared folder), stop `auto` there, and merge. Each merged video gets a new speaker number, a video the main dataset already has is skipped (merging the same folder twice adds nothing), and the other PC's autopilot decisions come along so the main PC never mines those videos. Then identify (one person mined on two PCs becomes one person) and split:
  ```bash
  pvdata merge --from D:/dataset_pc2 --into dataset/
  pvdata identify --out dataset/ --quarantine && pvdata split --out dataset/
  ```
  A PC keeps mining into its own folder after a merge; the next merge only adds what is new.

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
| `shot_cut` | The clip crosses a camera cut where the face also jumps or vanishes (anchor → B-roll, another person); tickers, banners and flashes don't count | Correct to drop |
| `sync_weak_av_correlation` / `sync_av_offset` | Cheap check: mouth doesn't move with the audio | Usually correct to drop |
| `sync_low_syncnet_conf` / `sync_syncnet_offset` | SyncNet: the face on screen is not the voice we hear (voice-over, a news anchor talking over footage of someone else, dubbing) | Correct to drop |
| `source_low_sync_rate` | Fewer than half of this video's clips passed the sync checks: a news package or B-roll montage, so its few "passes" are chance matches | Correct to drop |
| `source_below_720p` | Video is below 720p | Find an HD upload |

Thresholds are in `configs/default.yaml`; copy it and pass `--config my.yaml` to change them.

## Code map

| File | What |
|---|---|
| `pvdata/sources.py` | reads the sources CSV, assigns spk01, spk02, ... |
| `pvdata/media.py` | yt-dlp download, ffprobe, audio extraction, frame reading, clip cutting (ffmpeg) |
| `pvdata/asr.py` | faster-whisper / WhisperX / Hugging Face Whisper (IndicWhisper) with word timestamps |
| `pvdata/segment.py` | words → 1.5–3 s candidates |
| `pvdata/text_norm.py` | TEAM_GUIDE transcript rules, Hindi number words |
| `pvdata/ctc.py` + `pvdata/g2p/` | the trainer's CTC length rule with the trainer's own G2P |
| `pvdata/face.py` | MediaPipe face count, head pose, face/mouth size, mouth opening |
| `pvdata/sync.py` | cheap pre-filter: audio loudness vs mouth opening correlation |
| `pvdata/syncnet.py` | SyncNet active-speaker check (port of joonson/syncnet_python, same weights) |
| `pvdata/shots.py` | shot-cut detection (a clip must be one camera shot) |
| `pvdata/resync.py` | `pvdata resync`: re-check clips already in a dataset with SyncNet + shot cuts |
| `pvdata/pipeline.py` | runs everything per video, resumable; downloads the next video in the background |
| `pvdata/find.py` | YouTube search and channel listing (metadata only) |
| `pvdata/screen.py` | automatic review of a video from 3 short low-res samples |
| `pvdata/auto.py` | the autopilot loop: discover → screen → mine → expand good channels, with caps |
| `pvdata/identity.py` | face embeddings → persons, other-face clips, cached |
| `pvdata/export.py`, `pvdata/validate.py` | output layout, manifest, report, splits, validation |

## Notes

- `pvdata/g2p/hindi_g2p.py` is a copy of `Data/hindi_g2p.py` from the main repo. If that file changes there, copy it again so the CTC check stays identical to training.
- **Active speaker (is the face on screen the voice we hear?)** is checked in three layers: one camera shot per clip, a cheap mouth/loudness correlation, then [SyncNet](https://github.com/joonson/syncnet_python) on the clip plus up to 1 s of the same shot. SyncNet needs `pip install -e .[syncnet]` (torch + python_speech_features) and downloads `syncnet_v2.model` (~52 MB) on first use; a GPU makes it fast. `--no-syncnet` mines with the cheap check only. A video where fewer than half of the clips pass is dropped whole.
- **Clips mined before SyncNet** can be re-checked in place: `pvdata resync --out dataset/` (dry run, writes `resync/resync.csv`), look at a few clips near the threshold (changing `sync.min_conf` re-judges the saved scores, no re-scoring; `--recheck shot_cut` re-runs only those clips), then `pvdata resync --out dataset/ --apply` (stop `auto` first) and re-run `identify`, `split`, `validate`.
- Licence: MIT for this code. The videos you mine keep their own copyright.
