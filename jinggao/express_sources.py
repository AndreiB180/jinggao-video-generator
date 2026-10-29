"""Public footage supply for the express pipeline.

Bilibili supplies long real footage (search API + free whole-video preview
sprites + DASH streams through its overseas mirror); Pixabay supplies stock
clips when a key is configured. Everything here is free: no model calls.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import re
import struct
import threading
import time
from pathlib import Path

import httpx
from PIL import Image

from . import config, gpu, media, paths, throttle

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
BILI_REFERER = "https://www.bilibili.com/"
MIRRORS = ("upos-hz-mirrorakam.akamaized.net",)
SEARCH_TTL = 24 * 3600
TILE = (256, 144)


def _cache(name: str) -> Path:
    path = paths.DATA / "cache" / "express" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:40]


def _read_json(path: Path, ttl: float):
    try:
        if time.time() - path.stat().st_mtime < ttl:
            return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        pass
    return None


def _write_json(path: Path, value) -> None:
    tmp = paths.temporary(path)
    tmp.write_text(json.dumps(value, ensure_ascii=False), "utf-8")
    paths.publish(tmp, path)


def _seconds(text) -> float:
    try:
        parts = [int(p) for p in str(text).split(":")]
    except ValueError:
        return 0.0
    total = 0
    for part in parts:
        total = total * 60 + part
    return float(total)


def _plain(text) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", str(text or ""))).strip()


WBI_TABLE = [46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14,
             39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6,
             63, 57, 62, 11, 36, 20, 34, 44, 52]
COOKIE_FILE = "bilibili-cookies.json"


def cookie_path() -> Path:
    return paths.DATA / COOKIE_FILE


def saved_cookies() -> dict:
    try:
        data = json.loads(cookie_path().read_text("utf-8"))
        return {k: str(v) for k, v in data.items() if isinstance(k, str)} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_cookies(cookies: dict) -> None:
    import os
    paths.prepare()
    target = cookie_path()
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(cookies, ensure_ascii=False), "utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(target)
    paths.keep_sync()


class Bilibili:
    def __init__(self):
        self.client = httpx.Client(headers={"User-Agent": UA, "Referer": BILI_REFERER}, timeout=15,
                                   follow_redirects=True)
        self.lock = threading.Lock()
        self.ready = False
        self.search_gate = threading.Semaphore(3)
        self.api_gate = threading.Semaphore(3)
        self.blocked = 0              # signed API calls (_api): stream() etc.
        self.cooled = 0.0             # seconds this client waited for B站 to accept calls again
        self.search_refused = 0       # searches refused in a row (risk control or network)
        self.shots_blocked = 0        # videoshot (preview) calls refused in a row
        self.shots_blocked_at = 0.0
        self.offline_search = False   # index builds switch this on when search keeps being refused
        self.counts = threading.Lock()
        self.mixin = None
        self.cookies = saved_cookies()
        self.logged_in = False
        if self.cookies.get("SESSDATA"):
            for name, value in self.cookies.items():
                self.client.cookies.set(name, value, domain=".bilibili.com")
            try:
                nav = self.client.get("https://api.bilibili.com/x/web-interface/nav").json().get("data") or {}
                self.logged_in = bool(nav.get("isLogin"))
            except (httpx.HTTPError, ValueError):
                # The check itself was refused (B站 throttling three computers on one line): keep the saved
                # login rather than make the whole film anonymous (480p, most videos skipped).
                self.logged_in = True
            if not self.logged_in:
                self.client.cookies.clear()

    def close(self):
        self.client.close()

    def _warm(self, force=False):
        with self.lock:
            if force and not self.logged_in:
                self.client.cookies.clear()
                self.ready = False
            if not self.ready:
                self.client.get(BILI_REFERER)
                self.ready = True

    def _sign(self, params: dict) -> dict:
        import hashlib as _h
        import urllib.parse as _u
        with self.lock:
            if self.mixin is None:
                nav = self.client.get("https://api.bilibili.com/x/web-interface/nav").json()["data"]["wbi_img"]
                raw = (nav["img_url"].rsplit("/", 1)[1].split(".")[0] + nav["sub_url"].rsplit("/", 1)[1].split(".")[0])
                self.mixin = "".join(raw[i] for i in WBI_TABLE)[:32]
        params = dict(params, wts=int(time.time()))
        params = {k: "".join(ch for ch in str(v) if ch not in "!'()*") for k, v in params.items()}
        query = _u.urlencode(sorted(params.items()))
        params["w_rid"] = _h.md5((query + self.mixin).encode()).hexdigest()
        return params

    @staticmethod
    def _dm() -> dict:
        import base64 as _b
        import random
        import string
        r1, r2 = random.randint(0, 113), random.randint(0, 513)
        return {"dm_img_list": "[]",
                "dm_img_str": _b.b64encode("".join(random.choices(string.printable, k=random.randint(16, 64))).encode())[:-2].decode(),
                "dm_cover_img_str": _b.b64encode("".join(random.choices(string.printable, k=random.randint(32, 128))).encode())[:-2].decode(),
                "dm_img_inter": json.dumps({"ds": [], "wh": [2 * 1920 + 2 * 1080 + 3 * r1, 4 * 1920 - 1080 + r1, r1],
                                            "of": [30 + r2, 40 + 2 * r2, r2]}, separators=(",", ":"))}

    COOL_DOWN = 60          # seconds a refused film waits before asking again (it keeps everything it has)
    PATIENCE = 600          # ...and at most this long in all, then it gives up on B站 for this round

    def _api(self, url: str, params: dict) -> dict:
        if self.blocked >= 3:
            # Refused again and again: wait for B站 to calm down and go on, instead of giving up on B站 for the
            # round (the film then filled its lines with clips it already showed).
            if self.cooled >= self.PATIENCE:
                raise ValueError("B站接口持续风控，本轮跳过B站来源")
            throttle.bili_refused(self.COOL_DOWN)
            throttle.bili_wait()
            self.cooled += self.COOL_DOWN
            self.blocked = 0
        with self.api_gate:
            for attempt in range(3):
                self._warm()
                with throttle.bili_call():
                    r = self.client.get(url, params=self._sign(params))
                if r.status_code == 412:
                    throttle.bili_refused(2.0 * (attempt + 1))
                    time.sleep(2.0 * (attempt + 1))
                    continue
                body = r.json()
                if body.get("code") in (-352, -412, -509, -799):
                    throttle.bili_refused(2.0 * (attempt + 1))
                    time.sleep(2.0 * (attempt + 1))
                    continue
                self.blocked = 0
                throttle.bili_answered()
                return body
        self.blocked += 1
        raise ValueError("B站接口暂时限制访问（风控），请稍后重试或扫码登录B站")

    def cached(self, keyword: str, page: int = 1) -> bool:
        """This search was made today and its answer is on disk (no request needed)."""
        return _read_json(_cache("bili-search") / f"{_key(keyword, page)}.json", SEARCH_TTL) is not None

    def search(self, keyword: str, page: int = 1) -> list[dict]:
        path = _cache("bili-search") / f"{_key(keyword, page)}.json"
        cached = _read_json(path, SEARCH_TTL)
        if cached is not None:
            return cached
        if self.offline_search:      # risk control: only searches already cached today
            return []
        rows = None
        with self.search_gate:
            for attempt in range(3):
                try:
                    self._warm(force=attempt > 0)
                    with throttle.bili_call():
                        r = self.client.get("https://api.bilibili.com/x/web-interface/search/type",
                                            params={"search_type": "video", "keyword": keyword, "page": page})
                    if r.status_code == 412:
                        throttle.bili_refused(1.5 * (attempt + 1))
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    body = r.json()
                    if body.get("code") != 0:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    rows = body.get("data", {}).get("result") or []
                    break
                except (httpx.HTTPError, ValueError):
                    time.sleep(1.0 * (attempt + 1))
        with self.counts:
            self.search_refused = self.search_refused + 1 if rows is None else 0
        if rows is None:
            return []
        results = []
        for row in rows:
            if row.get("type") != "video" or not row.get("bvid") or str(row.get("is_pay")) == "1":
                continue
            if str(row.get("is_charge_video")) == "1" or str(row.get("badgepay")) == "True":
                continue
            results.append({
                "provider": "bilibili", "id": row["bvid"],
                "url": f"https://www.bilibili.com/video/{row['bvid']}",
                "title": _plain(row.get("title"))[:200], "tags": _plain(row.get("tag"))[:300],
                "description": _plain(row.get("description"))[:300],
                "duration": _seconds(row.get("duration")), "author": _plain(row.get("author"))[:60],
                "typename": _plain(row.get("typename")), "play": int(row.get("play") or 0),
                "pubdate": int(row.get("pubdate") or 0), "query": keyword,
            })
        if results:  # an empty reply may be risk control; never pin it for a day
            _write_json(path, results)
        return results

    def previews(self, bvid: str, duration: float) -> list[dict]:
        """Whole-video preview thumbnails: [{t, sprite, cell}] with sprite paths on disk."""
        root = _cache("bili-shots") / bvid
        meta_path = root / "meta.json"
        meta = _read_json(meta_path, 30 * 24 * 3600)
        if meta is None:
            with self.counts:
                # Three refusals in a row trip the breaker; after two minutes one call may probe again.
                if self.shots_blocked >= 3 and time.monotonic() - self.shots_blocked_at < 120:
                    raise ValueError("B站预览接口持续风控，暂停读取预览")
            self._warm()
            body = {}
            for attempt in range(3):
                with throttle.bili_call():
                    r = self.client.get("https://api.bilibili.com/x/player/videoshot", params={"bvid": bvid, "index": 1})
                body = r.json() if r.status_code == 200 else {}
                if r.status_code != 412 and body.get("code") not in (-352, -412, -509, -799):
                    with self.counts:
                        self.shots_blocked = 0
                    break
                throttle.bili_refused(2.0 * (attempt + 1))
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
            else:
                with self.counts:
                    self.shots_blocked += 1
                    self.shots_blocked_at = time.monotonic()
                raise ValueError("B站预览接口暂时限制访问（风控）")
            data = body.get("data") or {}
            images = [("https:" + u if u.startswith("//") else u) for u in data.get("image") or []]
            if not images:
                return []
            times = []
            if data.get("pvdata"):
                pv = data["pvdata"]
                pv = "https:" + pv if pv.startswith("//") else pv
                response = self.client.get(pv)
                raw = response.content if response.status_code == 200 else b""
                values = struct.unpack(f">{len(raw) // 2}H", raw[:len(raw) // 2 * 2])
                times = list(values[1:])
                if times and any(b < a for a, b in zip(times, times[1:], strict=False)):
                    times = []   # not a timestamp table; fall back to uniform spacing
            meta = {"images": images, "x": int(data.get("img_x_len") or 10), "y": int(data.get("img_y_len") or 10),
                    "w": int(data.get("img_x_size") or 160), "h": int(data.get("img_y_size") or 90), "times": times}
            root.mkdir(parents=True, exist_ok=True)
            _write_json(meta_path, meta)
        per = meta["x"] * meta["y"]
        times = meta["times"]
        capacity = per * len(meta["images"])
        if not times:
            count = capacity
            times = [duration * i / max(1, count) for i in range(count)]
        return [{"t": float(t), "sheet": index // per, "cell": index % per, "meta": meta, "root": str(root)}
                for index, t in enumerate(times[:capacity])]

    def sprite(self, root: str, meta: dict, sheet: int) -> Path:
        path = Path(root) / f"sprite-{sheet}.jpg"
        if not path.exists():
            self._warm()
            response = self.client.get(meta["images"][sheet])
            response.raise_for_status()
            Image.open(io.BytesIO(response.content)).verify()
            tmp = paths.temporary(path)
            tmp.write_bytes(response.content)
            paths.publish(tmp, path)
        return path

    VIEW_TTL = 30 * 24 * 3600   # a video's cid, length and title do not change
    STREAM_TTL = 80 * 60        # play addresses carry a deadline about two hours out
    LOWRES_TTL = 24 * 3600

    def view(self, bvid: str) -> dict:
        """The video's info (cid, pages, length, title), cached: every play address needs it, and library-first
        films ask for many more videos, which B站 answered with risk control (10-29: 76 of 76 refused)."""
        path = _cache("bili-view") / f"{bvid}.json"
        data = _read_json(path, self.VIEW_TTL)
        if data is None:
            view = self._api("https://api.bilibili.com/x/web-interface/wbi/view", {"bvid": bvid})
            if view.get("code") != 0:
                raise ValueError(f"B站视频信息不可用：{view.get('message', '')}")
            full = view["data"]
            data = {"cid": full["cid"], "pages": [{"duration": p.get("duration")} for p in (full.get("pages") or [])[:1]],
                    "duration": full.get("duration"), "title": full.get("title", ""), "desc": (full.get("desc") or "")[:500],
                    "owner": {"name": (full.get("owner") or {}).get("name", "")}}
            _write_json(path, data)
        return data

    def stream(self, bvid: str, min_height: int = 720) -> dict | None:
        """Smallest AVC video-only DASH stream with short edge >= min_height (max 1080). Answers are kept for a
        while (a film made again after a refusal asks for the same videos)."""
        path = _cache("bili-stream") / f"{bvid}-{min_height}-{int(bool(self.logged_in))}.json"
        kept = _read_json(path, self.STREAM_TTL)
        if kept is None:
            kept = _read_json(path, self.LOWRES_TTL)
            kept = kept if kept is not None and not kept.get("available") else None
        if kept is not None:
            return kept
        found = self._stream(bvid, min_height)
        if found and (found.get("available") or not found.get("voucher")):
            _write_json(path, found)
        return found

    def _stream(self, bvid: str, min_height: int) -> dict | None:
        data = self.view(bvid)
        params = {"bvid": bvid, "cid": data["cid"], "qn": 80, "fnval": 4048, "fourk": 1, **self._dm()}
        if not self.logged_in:
            params["try_look"] = 1
        play = self._api("https://api.bilibili.com/x/player/wbi/playurl", params)
        if play.get("code") != 0:
            raise ValueError(f"B站播放地址不可用：{play.get('message', '')}")
        videos = ((play.get("data") or {}).get("dash") or {}).get("video") or []
        rows = []
        for v in videos:
            short = min(int(v.get("width") or 0), int(v.get("height") or 0))
            if short < min_height or short > 1080:
                continue
            rows.append((0 if str(v.get("codecs", "")).startswith("avc") else 1, short, v.get("bandwidth") or 0, v))
        best_short = max((min(int(v.get("width") or 0), int(v.get("height") or 0)) for v in videos), default=0)
        if not rows:
            # No DASH video at all under load comes with a v_voucher (risk control), not a verdict on the video.
            return {"available": False, "best_short_edge": best_short,
                    "voucher": bool((play.get("data") or {}).get("v_voucher"))}
        rows.sort(key=lambda row: row[:3])
        v = rows[0][3]
        urls = [v.get("baseUrl") or v.get("base_url")] + list(v.get("backupUrl") or v.get("backup_url") or [])
        urls = [u for u in urls if u]
        urls.sort(key=lambda u: 0 if any(m in u for m in MIRRORS) else 1)
        segment = v.get("SegmentBase") or v.get("segment_base") or {}
        part = (data.get("pages") or [{}])[0]
        return {"available": True, "urls": urls, "width": int(v.get("width") or 0), "height": int(v.get("height") or 0),
                "headers": {"User-Agent": UA, "Referer": BILI_REFERER},
                "duration": float(part.get("duration") or data.get("duration") or 0), "title": data.get("title", ""),
                "description": (data.get("desc") or "")[:500], "uploader": (data.get("owner") or {}).get("name", ""),
                "segment_base": {"init": segment.get("Initialization") or segment.get("initialization"),
                                 "index": segment.get("indexRange") or segment.get("index_range")}}


    def audio(self, bvid: str) -> dict | None:
        """The smallest audio-only DASH stream of a video ({urls, headers, duration}), for transcribing what a
        news report says (news_pool); None when the video has none."""
        data = self.view(bvid)
        params = {"bvid": bvid, "cid": data["cid"], "qn": 16, "fnval": 16, **self._dm()}
        if not self.logged_in:
            params["try_look"] = 1
        tracks = []
        for attempt in range(3):
            play = self._api("https://api.bilibili.com/x/player/wbi/playurl", params)
            tracks = ((play.get("data") or {}).get("dash") or {}).get("audio") or []
            if play.get("code") == 0 and tracks:
                break
            # Under load B站 answers code 0 with a v_voucher and no streams: back off and ask again.
            throttle.bili_refused(3.0 * (attempt + 1))
            time.sleep(3.0 * (attempt + 1))
        else:
            raise ValueError(f"B站没有给出音频流（code {play.get('code')}{'，风控' if (play.get('data') or {}).get('v_voucher') else ''}）")
        track = min(tracks, key=lambda a: a.get("bandwidth") or 1e12)
        urls = [track.get("baseUrl") or track.get("base_url")] + list(track.get("backupUrl") or track.get("backup_url") or [])
        urls = sorted((u for u in urls if u), key=lambda u: 0 if any(m in u for m in MIRRORS) else 1)
        part = (data.get("pages") or [{}])[0]
        return {"urls": urls, "headers": {"User-Agent": UA, "Referer": BILI_REFERER},
                "duration": float(part.get("duration") or data.get("duration") or 0)}


def qr_login_start() -> dict:
    """Start a Bilibili app QR login. Returns {url, key, png}; scan within ~3 minutes."""
    import qrcode
    with httpx.Client(headers={"User-Agent": UA, "Referer": BILI_REFERER}, timeout=15) as client:
        data = client.get("https://passport.bilibili.com/x/passport-login/web/qrcode/generate").json()["data"]
    image = qrcode.make(data["url"], box_size=8, border=3)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return {"url": data["url"], "key": data["qrcode_key"], "png": buffer.getvalue()}


def qr_login_poll(key: str) -> dict:
    """Poll once. On success the session cookies are saved locally (never committed)."""
    with httpx.Client(headers={"User-Agent": UA, "Referer": BILI_REFERER}, timeout=15) as client:
        r = client.get("https://passport.bilibili.com/x/passport-login/web/qrcode/poll", params={"qrcode_key": key})
        data = r.json().get("data") or {}
        code = data.get("code")
        if code == 0:
            cookies = {c.name: c.value for c in client.cookies.jar if c.name in
                       ("SESSDATA", "bili_jct", "DedeUserID", "DedeUserID__ckMd5", "sid", "buvid3", "buvid4", "b_nut")}
            if cookies.get("SESSDATA"):
                save_cookies(cookies)
                return {"status": "success"}
            return {"status": "error", "message": "登录成功但未取得会话"}
    return {"status": {86101: "waiting", 86090: "scanned", 86038: "expired"}.get(code, "error"),
            "message": data.get("message", "")}


def login_status() -> dict:
    cookies = saved_cookies()
    if not cookies.get("SESSDATA"):
        return {"logged_in": False}
    with httpx.Client(headers={"User-Agent": UA, "Referer": BILI_REFERER}, timeout=15, cookies=cookies) as client:
        try:
            data = client.get("https://api.bilibili.com/x/web-interface/nav").json().get("data") or {}
        except (httpx.HTTPError, ValueError):
            return {"logged_in": True, "verified": False}
    return {"logged_in": bool(data.get("isLogin")), "verified": True, "name": data.get("uname", ""),
            "vip": bool((data.get("vip") or {}).get("status"))}


_file_locks: dict = {}
_file_locks_guard = threading.Lock()


def _file_lock(name: str) -> threading.Lock:
    with _file_locks_guard:
        return _file_locks.setdefault(name, threading.Lock())


class Pixabay:
    def __init__(self, key: str):
        self.key = key
        self.client = httpx.Client(timeout=20, follow_redirects=True, headers={"User-Agent": UA})

    def close(self):
        self.client.close()

    def search(self, query: str) -> list[dict]:
        path = _cache("pixabay-search") / f"{_key(query)}.json"
        cached = _read_json(path, SEARCH_TTL)
        if cached is not None:
            return cached
        try:
            r = self.client.get("https://pixabay.com/api/videos/",
                                params={"key": self.key, "q": query[:100], "per_page": 20, "safesearch": "true"})
            r.raise_for_status()
            hits = r.json().get("hits") or []
        except (httpx.HTTPError, ValueError):
            return []
        results = []
        for hit in hits:
            if hit.get("type") not in (None, "film"):
                continue
            renditions = []
            for name, v in (hit.get("videos") or {}).items():
                if not v.get("url"):
                    continue
                short = min(int(v.get("width") or 0), int(v.get("height") or 0))
                renditions.append({"name": name, "url": v["url"], "width": int(v.get("width") or 0),
                                   "height": int(v.get("height") or 0), "short": short,
                                   "size": int(v.get("size") or 0), "thumbnail": v.get("thumbnail", "")})
            usable = sorted((r for r in renditions if 720 <= r["short"] <= 1080), key=lambda r: r["size"] or 1e12)
            if not usable:
                continue
            thumb = next((r["thumbnail"] for r in sorted(renditions, key=lambda r: r["short"]) if r["thumbnail"]), "")
            results.append({
                "provider": "pixabay", "id": str(hit["id"]), "url": hit.get("pageURL", ""),
                "title": hit.get("tags", ""), "tags": hit.get("tags", ""), "description": "",
                "duration": float(hit.get("duration") or 0), "author": hit.get("user", ""),
                "rendition": usable[0], "thumbnail": thumb, "query": query,
                "license": "Pixabay Content License", "license_url": "https://pixabay.com/service/license-summary/",
            })
        _write_json(path, results)
        return results

    def thumbnail(self, row: dict) -> Image.Image | None:
        if not row.get("thumbnail"):
            return None
        path = _cache("pixabay-thumbs") / f"{row['id']}.jpg"
        if not path.exists():
            try:
                data = self.client.get(row["thumbnail"]).content
                Image.open(io.BytesIO(data)).verify()
                path.write_bytes(data)
            except (httpx.HTTPError, OSError, ValueError):
                return None
        try:
            with Image.open(path) as im:
                return im.convert("RGB")
        except OSError:
            return None

    def download(self, row: dict, check=None) -> Path:
        rendition = row["rendition"]
        path = _cache("pixabay-files") / f"{row['id']}-{rendition['name']}.mp4"
        with _file_lock(str(path)):
            return self._download(row, path, check)

    def _download(self, row: dict, path: Path, check=None) -> Path:
        rendition = row["rendition"]
        if path.exists() and path.stat().st_size > 10000:
            return path
        tmp = paths.temporary(path)
        try:
            with self.client.stream("GET", rendition["url"], timeout=60) as r:
                r.raise_for_status()
                with tmp.open("wb") as out:
                    for chunk in r.iter_bytes(1 << 20):
                        if check:
                            check()
                        out.write(chunk)
            paths.publish(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return path


def bilibili_tile(client: Bilibili, preview: dict) -> Image.Image:
    meta = preview["meta"]
    sprite = client.sprite(preview["root"], meta, preview["sheet"])
    with Image.open(sprite) as im:
        cell = preview["cell"]
        x, y = (cell % meta["x"]) * meta["w"], (cell // meta["x"]) * meta["h"]
        return im.crop((x, y, x + meta["w"], y + meta["h"])).convert("RGB")


def scale_filter(short_edge: int = 720) -> str:
    # Landscape -> N rows, portrait -> N columns (N = project minimum, at least 720).
    n = max(720, int(short_edge))
    return (f"scale='if(gte(iw,ih),-2,{n})':'if(gte(iw,ih),{n},-2)':flags=bicubic,"
            "fps=25,format=yuv420p")


def cuda_scale_filter(short_edge: int = 720) -> str:
    """scale_filter on the card: frames stay in GPU memory from the decoder to the encoder."""
    n = max(720, int(short_edge))
    return f"scale_cuda=w='if(gte(iw,ih),-2,{n})':h='if(gte(iw,ih),{n},-2)'"


def clip_args(source: str, start: float, duration: float, out: Path, way: str, *, headers: dict | None = None,
              short_edge: int = 720) -> list:
    """FFmpeg arguments that decode exactly [start, start+duration) into an H.264 clip, made one way:
    full (NVDEC + CUDA scaling + NVENC), nvenc (CPU decode and scale, NVENC) or cpu (libx264)."""
    args = [paths.binary("ffmpeg"), "-v", "error", "-y"]
    if way == "full":
        args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    else:
        args += ["-threads", "2", "-filter_threads", "1"]      # many clips run at once: few threads each
    if headers:
        args += ["-reconnect", "1", "-reconnect_on_network_error", "1", "-reconnect_streamed", "1",
                 "-reconnect_delay_max", "3", "-rw_timeout", "15000000",
                 "-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    args += ["-ss", f"{max(0.0, start):.3f}", "-i", source, "-t", f"{duration:.3f}", "-an"]
    if way == "full":
        args += ["-vf", cuda_scale_filter(short_edge), "-fps_mode", "cfr", "-r", "25", *gpu.NVENC_CLIP]
    else:
        args += ["-vf", scale_filter(short_edge), *(gpu.NVENC_CLIP if way == "nvenc" else gpu.X264_CLIP)]
    return args + ["-movflags", "+faststart", str(out)]


def cut_window(source: str, start: float, duration: float, out: Path, *, headers: dict | None = None,
               check=None, timeout: int = 90, short_edge: int = 720, mode: str | None = None) -> str:
    """Decode exactly [start, start+duration) into an H.264 clip; partial output never survives.

    Returns how it was encoded (full / nvenc / cpu). With an NVIDIA card the clip goes to the card (waiting a
    moment for a free session); a GPU failure is redone on the CPU. mode forces one way, without fallback."""
    final = out
    out = out.with_name(out.stem + ".part.mp4")
    def make(way):
        media.run(clip_args(source, start, duration, out, way, headers=headers, short_edge=short_edge),
                  timeout=timeout, check=check)
    try:
        if mode:
            make(mode)
            used = mode
        else:
            used = _encode_clip(make, check)
        out.replace(final)
        return used
    finally:
        out.unlink(missing_ok=True)


def _encode_clip(make, check) -> str:
    way = gpu.mode("clip")
    if way != "cpu":
        with throttle.either(check) as lane:
            if lane == "cpu":
                make("cpu")
                gpu.worked("cpu")
                return "cpu"
            try:
                make(way)
                gpu.worked(way)
                return way
            except ValueError as exc:      # the card refused this clip: the CPU makes it below
                gpu.failed(way, exc)
    with throttle.slot("cpu", check):
        make("cpu")
    gpu.worked("cpu")
    return "cpu"


def fetch_range(client: httpx.Client, url: str, start: int, end: int, check=None, attempts: int = 8) -> bytes:
    """Inclusive byte range with resume; the overseas mirror often closes long responses early."""
    buffer = bytearray()
    want = end - start + 1
    for attempt in range(attempts):
        if check:
            check()
        try:
            with client.stream("GET", url, headers={"Range": f"bytes={start + len(buffer)}-{end}"}) as r:
                if r.status_code not in (200, 206):
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                skip = start + len(buffer) if r.status_code == 200 else 0
                for chunk in r.iter_bytes(256 * 1024):
                    if check:
                        check()   # cancellation or an abandoned round stops a slow transfer mid-stream
                    if skip:
                        cut = min(skip, len(chunk))
                        chunk, skip = chunk[cut:], skip - cut
                    buffer.extend(chunk)
                    if len(buffer) >= want:
                        break
            if len(buffer) >= want:
                return bytes(buffer[:want])
        except httpx.HTTPError:
            time.sleep(0.3 * (attempt + 1))
    raise ValueError(f"分段下载未完成（{len(buffer)}/{want} 字节）")


def _boxes(data: bytes):
    pos = 0
    while pos + 8 <= len(data):
        size, kind = struct.unpack(">I4s", data[pos:pos + 8])
        header = 8
        if size == 1 and pos + 16 <= len(data):
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            header = 16
        if size < header:
            return
        yield kind.decode("latin1"), pos, size, header
        pos += size


def dash_index(client: httpx.Client, url: str, check=None, segment_base: dict | None = None) -> dict:
    limit = 65535
    try:
        limit = max(4095, int(str((segment_base or {}).get("index", "")).split("-")[1]))
    except (IndexError, ValueError):
        pass
    head = fetch_range(client, url, 0, limit, check)
    init_end, refs = None, None
    for kind, pos, size, header in _boxes(head):
        if kind == "moov":
            init_end = pos + size
        if kind == "sidx":
            if pos + size > len(head):
                head = head + fetch_range(client, url, len(head), pos + size - 1, check)
            body = head[pos + header:pos + size]
            version, timescale = body[0], struct.unpack(">I", body[8:12])[0]
            if version == 0:
                earliest, first = struct.unpack(">II", body[12:20])
                rest = body[20:]
            else:
                earliest, first = struct.unpack(">QQ", body[12:28])
                rest = body[28:]
            count = struct.unpack(">H", rest[2:4])[0]
            offset, t = pos + size + first, earliest / timescale
            refs = []
            for i in range(count):
                word, duration, _ = struct.unpack(">III", rest[4 + 12 * i:16 + 12 * i])
                length = word & 0x7FFFFFFF
                refs.append((offset, length, t, duration / timescale))
                offset += length
                t += duration / timescale
            break
    if init_end is None or not refs:
        raise ValueError("视频流缺少分段索引")
    return {"init": head[:init_end], "refs": refs}


def dash_window(stream: dict, start: float, duration: float, out: Path, check=None, short_edge: int = 720) -> str:
    """Fetch only the DASH segments covering the window (holding a download slot), then trim locally (holding
    a gpu or cpu slot instead). Returns how the clip was encoded."""
    last = None
    for url in stream["urls"]:
        local = out.with_suffix(".src.mp4")
        try:
            with throttle.slot("bili_download", check):
                refs = _fetch_segments(stream, url, start, duration, local, check)
            try:
                return cut_window(str(local), start - refs[0][2], duration, out, check=check, timeout=60,
                                  short_edge=short_edge)
            finally:
                local.unlink(missing_ok=True)
        except (ValueError, httpx.HTTPError, OSError) as exc:
            last = exc
            local.unlink(missing_ok=True)
            out.unlink(missing_ok=True)
    raise ValueError(f"片段下载失败：{last}")


def _fetch_segments(stream: dict, url: str, start: float, duration: float, local: Path, check=None) -> list:
    """Write the init segment plus the media segments covering the window to local; returns their refs."""
    with httpx.Client(headers=stream["headers"], timeout=httpx.Timeout(10, read=8), follow_redirects=True) as client:
        index = stream.setdefault("_index", {}).get(url) or dash_index(client, url, check, stream.get("segment_base"))
        stream["_index"][url] = index
        refs = [r for r in index["refs"] if r[2] + r[3] > start - 0.05 and r[2] < start + duration + 0.05]
        if not refs:
            raise ValueError("时间窗超出视频长度")
        parts = [None] * len(refs)
        def get(i):
            offset, length, _, _ = refs[i]
            parts[i] = fetch_range(client, url, offset, offset + length - 1, check)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(refs))) as pool:
            list(pool.map(get, range(len(refs))))
    local.write_bytes(index["init"] + b"".join(parts))
    return refs


def probe_duration(path: Path) -> float:
    info = media.probe(path)
    return float(info.get("format", {}).get("duration") or 0)


def stock_key() -> str:
    return config.read().get("pixabay_api_key", "").strip()
