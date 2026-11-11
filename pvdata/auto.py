"""`pvdata auto`: grow the dataset to a target number of hours without reviewing videos by hand.

    seed queries ─► discover (YouTube search, metadata filter) ─► queue
         ▲                                                          │
         │ channels that yield well: list their other uploads       ▼
         └──────────────── mine (full pipeline) ◄─ screen (3 x 20 s samples: face + Hindi)

- Every video becomes its own provisional speaker (``<channel>_<video id>``); run `pvdata identify` afterwards
  to merge each person's videos and catch guests.
- Speaker diversity: at most ``max_videos_per_channel`` videos and ``max_hours_per_channel`` hours per channel.
- Downloads run ahead in the background (screening samples and the next full video) while the GPU/CPU work
  on the current one; full videos are deleted after mining unless ``keep_downloads``.
- All decisions live in <out>/auto/state.json, so the loop can be stopped (Ctrl+C) and restarted any time.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from pvdata.config import Config
from pvdata.export import Manifest, build_report, write_json
from pvdata.find import Hit
from pvdata.screen import ScreenResult
from pvdata.sources import Source

log = logging.getLogger("pvdata")


def read_seeds(path: str | Path) -> list[str]:
    """Seed queries, one per row (column ``query``); ``#`` lines are comments."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        return [r["query"].strip() for r in rows if (r.get("query") or "").strip()]


def slug(text: str, n: int = 24) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return (s or "yt")[:n]


class State:
    """Everything `auto` has decided, saved after every change (atomic replace)."""

    def __init__(self, path: Path):
        self.path = path
        d = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.videos: dict[str, dict] = d.get("videos", {})  # id -> {title, channel, channel_id, duration, status...}
        self.channels: dict[str, dict] = d.get("channels", {})  # id -> {name, mined, accepted_s, expanded}
        self.queries: dict[str, int] = d.get("queries", {})  # query -> search pool already used

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        data = {"videos": self.videos, "channels": self.channels, "queries": self.queries}
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def channel(self, cid: str, name: str = "") -> dict:
        c = self.channels.setdefault(cid or "unknown", {"name": name, "mined": 0, "accepted_s": 0.0, "expanded": False})
        if name and not c["name"]:
            c["name"] = name
        return c

    def count(self, status: str) -> int:
        return sum(v["status"] == status for v in self.videos.values())


class AutoPilot:
    def __init__(self, cfg: Config, out: str | Path, seeds: list[str], pipeline, screener,
                 search: Callable[[str, int], list[Hit]], channel_list: Callable[[str, int], list[Hit]],
                 make_source: Callable[[dict], Source] | None = None):  # fmt: skip
        self.cfg, self.out, self.seeds = cfg, Path(out), seeds
        self.pipe, self.screener, self.search, self.channel_list = pipeline, screener, search, channel_list
        self.make_source = make_source or (lambda v: Source(f"https://www.youtube.com/watch?v={v['id']}",
                                                            f"{slug(v.get('channel', ''))}_{v['id']}"))  # fmt: skip
        self.state = State(self.out / "auto" / "state.json")
        self.exclude = re.compile(cfg.auto.exclude_title_regex) if cfg.auto.exclude_title_regex else None
        self.io = ThreadPoolExecutor(2)  # screening samples download ahead
        self.samples: dict[str, Future] = {}
        for v in self.state.videos.values():  # a crash mid-screen or mid-mine: just try that video again
            if v["status"] in ("screening", "mining"):
                v["status"] = "new" if v["status"] == "screening" else "ok"

    # ------------------------------------------------------------- discover
    def hours(self) -> float:
        return build_report(Manifest(self.out / "manifest.csv").rows())["accepted_hours"]

    def channel_yield(self, cid: str) -> float:
        """Accepted seconds per mined video so far (0 for an untried channel)."""
        c = self.state.channel(cid)
        return c["accepted_s"] / c["mined"] if c["mined"] else 0.0

    def channel_done(self, cid: str) -> bool:
        """The channel has reached its cap of mined videos or accepted hours."""
        a, c = self.cfg.auto, self.state.channel(cid)
        if c.get("blocked"):  # set by `pvdata exclude` (e.g. an animation channel)
            return True
        return c["mined"] >= a.max_videos_per_channel or c["accepted_s"] >= a.max_hours_per_channel * 3600

    def channel_full(self, cid: str) -> bool:
        """...or would reach it with the videos already waiting to be mined."""
        queued = sum(v.get("channel_id") == cid and v["status"] == "ok" for v in self.state.videos.values())
        return (
            self.channel_done(cid) or self.state.channel(cid)["mined"] + queued >= self.cfg.auto.max_videos_per_channel
        )

    def add_hits(self, hits: list[Hit]) -> int:
        a, n = self.cfg.auto, 0
        for h in hits:
            if h.video_id in self.state.videos:
                continue
            reason = ""
            if h.duration and not a.min_minutes * 60 <= h.duration <= a.max_minutes * 60:  # 0 = unknown, look up later
                reason = "meta_duration"
            elif self.exclude is not None and self.exclude.search(h.title):
                reason = "meta_title"
            self.state.videos[h.video_id] = {"id": h.video_id, "title": h.title, "channel": h.channel,
                                             "channel_id": h.channel_id, "duration": h.duration,
                                             "status": "rejected" if reason else "new", "reason": reason}  # fmt: skip
            self.state.channel(h.channel_id, h.channel)
            n += 0 if reason else 1
        return n

    def discover(self) -> int:
        """One round: expand good channels, then search each seed query a bit deeper. Returns new videos."""
        a, found = self.cfg.auto, 0
        for cid, c in self.state.channels.items():
            if c["expanded"] or not cid or cid == "unknown" or c["mined"] < a.expand_after_videos:
                continue
            if c["accepted_s"] / max(c["mined"], 1) >= a.expand_min_minutes_per_video * 60 and not self.channel_full(
                cid
            ):
                c["expanded"] = True
                try:
                    n = self.add_hits(self.channel_list(cid, a.channel_list_size))
                    log.info("auto: channel %s (%.1f min/video) -> %d new videos", c["name"], c["accepted_s"] / 60
                             / max(c["mined"], 1), n)  # fmt: skip
                    found += n
                except Exception as e:  # noqa: BLE001
                    log.warning("auto: listing channel %s failed: %s", cid, e)
        for q in self.seeds:
            used = self.state.queries.get(q, 0)
            pool = a.search_pool if used == 0 else min(used * 2, a.max_search_pool)
            if pool <= used:
                continue
            self.state.queries[q] = pool
            try:
                n = self.add_hits(self.search(q, pool))
                log.info("auto: search %r (top %d) -> %d new videos", q, pool, n)
                found += n
            except Exception as e:  # noqa: BLE001
                log.warning("auto: search %r failed: %s", q, e)
        self.state.save()
        return found

    # --------------------------------------------------------------- screen
    def need_screening(self) -> bool:
        """Screening is only worth its downloads while the queue of videos ready to mine is short."""
        return self.state.count("ok") < self.cfg.auto.ok_buffer

    def queue_samples(self, ahead: int = 3) -> None:
        if not self.need_screening():
            return
        new = [v for v in self.state.videos.values() if v["status"] == "new" and v["id"] not in self.samples]
        new.sort(key=lambda v: -self.channel_yield(v.get("channel_id", "")))  # proven channels first
        for v in new[: max(0, ahead - len(self.samples))]:
            if self.channel_full(v.get("channel_id", "")):
                blocked = self.state.channel(v.get("channel_id", "")).get("blocked")
                v["status"], v["reason"] = "rejected", "channel_blocked" if blocked else "channel_cap"
                continue
            self.samples[v["id"]] = self.io.submit(self._fetch_samples, v)

    def _fetch_samples(self, v: dict) -> list | None:
        """Runs in a download thread. None means the video turned out too short or too long."""
        if not v.get("duration"):  # some listings have no length: ask YouTube before sampling
            v["duration"] = self.duration_of(v["id"])
            a = self.cfg.auto
            if not a.min_minutes * 60 <= v["duration"] <= a.max_minutes * 60:
                return None
        return self.screener.fetch(v["id"], v["duration"])

    def duration_of(self, video_id: str) -> float:
        from pvdata.media import video_duration

        return video_duration(f"https://www.youtube.com/watch?v={video_id}")

    def screen_one(self, block: bool) -> bool:
        """Score one video whose samples have arrived (waits for one if ``block``). True if one was screened."""
        if not self.samples:
            return False
        ready = [vid for vid, f in self.samples.items() if f.done()]
        if not ready and not block:
            return False
        vid = ready[0] if ready else next(iter(self.samples))
        fut = self.samples.pop(vid)
        v = self.state.videos[vid]
        v["status"] = "screening"
        try:
            samples = fut.result()
            res = (ScreenResult(False, "meta_duration") if samples is None
                   else self.screener.screen(vid, v["duration"], samples=samples))  # fmt: skip
        except Exception as e:  # noqa: BLE001 - sample download failed
            log.warning("auto: samples of %s failed: %s", vid, e)
            res = ScreenResult(False, f"screen_error:{type(e).__name__}")
        v.update(status="ok" if res.ok else "rejected", reason=res.reason, screen=res.to_dict())
        log.info("auto: screen %s %s %s (face %.2f, deva %.2f)", vid, "OK  " if res.ok else "SKIP", res.reason,
                 res.face_ratio, res.devanagari_ratio)  # fmt: skip
        self.state.save()
        return True

    # ----------------------------------------------------------------- mine
    def next_to_mine(self) -> list[dict]:
        ok = [v for v in self.state.videos.values() if v["status"] == "ok"]
        return sorted(ok, key=lambda v: -v.get("screen", {}).get("score", 0))

    def mine_one(self) -> bool:
        todo = []
        for v in self.next_to_mine():
            if self.channel_done(v.get("channel_id", "")):
                blocked = self.state.channel(v.get("channel_id", "")).get("blocked")
                v.update(status="rejected", reason="channel_blocked" if blocked else "channel_cap")
            else:
                todo.append(v)
        if not todo:
            return False
        v = todo[0]
        if len(todo) > 1:
            self.pipe.prefetch(self.make_source(todo[1]))
        src = self.make_source(v)
        spk = self.pipe.assign_ids([src])[src.speaker]
        v["status"] = "mining"
        self.state.save()
        counts = self.pipe.run_safe(src, spk)
        c = self.state.channel(v.get("channel_id", ""), v.get("channel", ""))
        if counts is None:
            v.update(status="failed")
        else:
            got = counts.get("accepted_s", 0.0)
            v.update(status="mined", accepted=counts["accepted"], accepted_s=got, speaker=spk)
            c["mined"] += 1
            c["accepted_s"] += got
            self._hours += got / 3600
        self.state.save()
        return True

    # ----------------------------------------------------------------- loop
    def run(self, target_hours: float | None = None, max_videos: int | None = None) -> dict:
        target = target_hours if target_hours is not None else self.cfg.auto.target_hours
        mined, t0 = 0, time.time()
        self._hours = hours = self.hours()
        while hours < target and (max_videos is None or mined < max_videos):
            if self.state.count("new") < 5 and not self.samples:
                if self.discover() == 0 and self.state.count("new") == 0 and self.state.count("ok") == 0:
                    log.info("auto: nothing left to try; add seed queries to find more")
                    break
            self.queue_samples()
            for _ in range(3):  # score samples that have arrived, but never starve mining
                if not self.need_screening() or not self.screen_one(block=False):
                    break
                self.queue_samples()
            t1 = time.time()
            if self.mine_one():
                mined += 1
                hours = self._hours
                log.info("auto: %.2f / %.0f h after %d videos this session (%.0f min; this video %.1f min)", hours,
                         target, mined, (time.time() - t0) / 60, (time.time() - t1) / 60)  # fmt: skip
            elif not self.screen_one(block=True) and self.state.count("new") == 0:
                if self.discover() == 0:
                    log.info("auto: nothing left to try; add seed queries to find more")
                    break
        report = build_report(Manifest(self.out / "manifest.csv").rows())
        write_json(self.out / "report.json", report)
        return {"hours": report["accepted_hours"], "mined_this_session": mined, **self.summary()}

    def summary(self) -> dict:
        reasons: dict[str, int] = {}
        for v in self.state.videos.values():
            key = v["status"] if v["status"] != "rejected" else f"rejected:{v.get('reason') or '?'}"
            reasons[key] = reasons.get(key, 0) + 1
        return {"videos": dict(sorted(reasons.items(), key=lambda kv: -kv[1])), "channels": len(self.state.channels)}
