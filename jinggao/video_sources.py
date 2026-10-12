"""Public video discovery, metadata-first inspection and bounded clip retrieval.

Discovery is not permission to reuse a video. All providers retain their source URL
and publisher evidence. This module never supplies cookies, defeats a challenge,
deciphers protected streams, or calls a local ML model.
"""
from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time
from collections.abc import Callable
from concurrent.futures import as_completed
from .execution import ThreadPoolExecutor
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlencode, urljoin, urlparse

import httpx

from . import execution, media, network, paths, store

AGENT = "Jinggao/0.1 (public video research; metadata and bounded previews)"
METADATA_LIMIT = 4 * 1024 * 1024
PREVIEW_LIMIT = 32 * 1024 * 1024
CLIP_LIMIT = 180 * 1024 * 1024
Check = Callable[[], None]
SEARCH_PROVIDER_SLOTS = threading.BoundedSemaphore(8)
_search_diagnostics = threading.local()
UNAVAILABLE_TTLS = {"preview_only": 86400, "blocked": 300, "unsupported": 1800}


class TencentQueryFallback:
    """Per-match feedback for later, already-budgeted Tencent searches.

    The caller supplies the complete safe topic/entity query. This class never
    infers one, trims names or changes a shot's content/identity requirements.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._empty_long_query = False

    @staticmethod
    def explicit_short(query: str, hint: str) -> str:
        if not isinstance(hint, str):
            return ""
        hint = " ".join(hint.split())
        original = " ".join(query.split())
        return hint if hint and len(hint) < len(original) and len(hint) <= 80 else ""

    def select(self, query: str, hint: str) -> str:
        short = self.explicit_short(query, hint)
        with self._lock:
            return short if self._empty_long_query and short else query

    def observe_empty(self, query: str, effective: str, hint: str) -> bool:
        if effective != query or not self.explicit_short(query, hint):
            return False
        with self._lock:
            self._empty_long_query = True
        return True


def _clean(value: object, limit=1200) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", str(value or ""))).strip()[:limit]


def _number(value: object, default=0.0) -> float:
    try:
        n = float(value)
        return n if math.isfinite(n) and n >= 0 else default
    except (ValueError, TypeError):
        return default


def _duration(value: object) -> float:
    if isinstance(value, (float, int)):
        return _number(value)
    text = str(value or "")
    if re.fullmatch(r"\d+(?::\d{1,2}){1,2}", text):
        total = 0
        for part in text.split(":"):
            total = total * 60 + int(part)
        return float(total)
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", text)
    return sum(_number(v) * scale for v, scale in zip(m.groups(), (3600, 60, 1), strict=True)) if m else _number(text)


def _http_url(value: object) -> str:
    text = html.unescape(str(value or ""))
    p = urlparse(text)
    return text if p.scheme in ("https", "http") and p.hostname and not p.username and not p.password else ""


def _host(url: str, suffix: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == suffix or host.endswith("." + suffix)


def _fetch(url: str, *, limit=METADATA_LIMIT, check: Check | None = None,
           headers: dict | None = None, target: Path | None = None,
           json_payload: dict | None = None) -> tuple[bytes, dict]:
    with execution.network_slot(check):
        return _fetch_response(url, limit=limit, check=check, headers=headers, target=target,
                               json_payload=json_payload)


def _fetch_response(url: str, *, limit=METADATA_LIMIT, check: Check | None = None,
                    headers: dict | None = None, target: Path | None = None,
                    json_payload: dict | None = None) -> tuple[bytes, dict]:
    """Validate each redirect; cap transferred bytes, not just Content-Length."""
    from .sources import public_url, download_slot
    out = bytearray()
    original = url
    stream_file = None
    started = time.monotonic()
    try:
        if target:
            stream_file = target.open("wb")
        with network.client(timeout=20, follow_redirects=False,
                          headers={"User-Agent": AGENT, **(headers or {})}) as client:
            for _ in range(5):
                if check:
                    check()
                public_url(url)
                cooldown = download_slot(url)
                options = {"json": json_payload} if json_payload is not None else {}
                with client.stream("POST" if json_payload is not None else "GET", url, **options) as response:
                    if response.is_redirect:
                        if json_payload is not None:
                            raise ValueError("素材搜索接口发生重定向，已停止此渠道")
                        redirected = str(response.url.join(response.headers.get("location", "")))
                        if headers and any(k.lower() == "authorization" for k in headers) and (
                                urlparse(redirected).netloc != urlparse(url).netloc or urlparse(redirected).scheme != "https"):
                            raise ValueError("鉴权接口发生跨域重定向，已停止以保护本地密钥")
                        url = redirected
                        continue
                    if response.status_code == 429:
                        retry = response.headers.get("Retry-After", "60")
                        delay = min(86400, max(1, int(retry))) if retry.isdigit() else 60
                        cooldown.write_text(json.dumps({"until": time.time() + delay}), "utf-8")
                        raise ValueError("来源暂时限流，已遵守 Retry-After；稍后再试")
                    if response.status_code in (401, 403, 412, 451):
                        raise ValueError(f"来源限制公开访问（HTTP {response.status_code}），请在原页确认或导入已获授权文件")
                    response.raise_for_status()
                    length = _number(response.headers.get("Content-Length"))
                    if length > limit:
                        raise ValueError(f"资源超过本次 {limit // (1024 * 1024)} MB 下载预算")
                    size = 0
                    for chunk in response.iter_bytes(64 * 1024):
                        if check:
                            check()
                        if time.monotonic() - started > 100:
                            raise ValueError("来源响应过慢，已停止本次下载")
                        size += len(chunk)
                        if size > limit:
                            raise ValueError(f"资源超过本次 {limit // (1024 * 1024)} MB 下载预算")
                        if stream_file:
                            stream_file.write(chunk)
                        else:
                            out.extend(chunk)
                    return bytes(out), {"bytes": size, "url": str(response.url),
                                        "content_type": response.headers.get("Content-Type", ""),
                                        "original_url": original}
            raise ValueError("来源重定向过多")
    except httpx.HTTPError:
        raise ValueError("公开视频来源暂时无法读取，请检查网络或从原页导入授权文件") from None
    finally:
        if stream_file:
            stream_file.close()


def _json(url: str, *, check: Check | None = None, headers=None) -> dict:
    data, _ = _fetch(url, check=check, headers=headers)
    try:
        body = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        raise ValueError("来源没有返回公开数据，可能需要验证或页面已变化") from None
    if not isinstance(body, dict):
        raise ValueError("来源返回的数据格式已变化")
    return body


def _cached(provider: str, query: str, fetch: Callable[[], list[dict]]) -> list[dict]:
    root = paths.DATA / "cache"
    root.mkdir(parents=True, exist_ok=True)
    cache = root / ("video-" + provider + "-" + hashlib.sha256(query.encode()).hexdigest() + ".json")
    began = time.monotonic()
    diagnostic = {"provider": provider, "cache_hit": False}
    try:
        with execution.keyed_lock(("video-search", str(cache))):
            diagnostic["cache_wait_seconds"] = round(time.monotonic()-began, 3)
            if cache.exists() and time.time() - cache.stat().st_mtime < 1800:
                try:
                    result = json.loads(cache.read_text("utf-8"))
                    diagnostic["cache_hit"] = True
                    return result
                except (ValueError, OSError):
                    pass
            result = fetch()
            temporary = cache.with_name(cache.name + "." + store.uid() + ".tmp")
            try:
                temporary.write_text(json.dumps(result, ensure_ascii=False), "utf-8")
                temporary.replace(cache)
            finally:
                temporary.unlink(missing_ok=True)
            return result
    finally:
        records = getattr(_search_diagnostics, "caches", None)
        if records is not None:
            records.append(diagnostic | {"elapsed_seconds": round(time.monotonic()-began, 3)})


def _candidate(provider: str, title: str, source: str, thumbnail: str, **extra) -> dict:
    from .sources import candidate
    row = candidate(provider, title, source, thumbnail, source, kind="video",
                    duration=0, width=0, height=0, publisher="", publisher_url="",
                    source_verified=False, metadata_checked=False,
                    preview_urls=[{"url": thumbnail, "at": None, "kind": "publisher_thumbnail"}] if thumbnail else [],
                    download_status="unchecked", download_reason="先读来源和小预览，再决定是否下载片段")
    return row | extra


def _evidence(source: str, title: str, description: str, publisher: str, publisher_url: str,
              published_at: str, origin: str) -> dict:
    return {"url": source, "title": _clean(title), "description": _clean(description),
            "publisher": _clean(publisher), "publisher_url": _http_url(publisher_url),
            "published_at": _clean(published_at, 100), "readable": True, "evidence_origin": origin}


def parse_cctv_search(body: dict) -> list[dict]:
    result = []
    for item in body.get("list", [])[:24]:
        source, thumb = _http_url(item.get("urllink")), _http_url(item.get("imglink"))
        if not source or not _host(source, "cctv.com"):
            continue
        title = _clean(item.get("all_title") or item.get("title"))
        publisher, date = _clean(item.get("channel") or "央视网"), _clean(item.get("uploadtime"))
        row = _candidate("cctv", title, source, thumb)
        row.update(duration=_duration(item.get("durations")), publisher=publisher,
                   publisher_url="https://www.cctv.com/", published_at=date, source_verified=True,
                   license="央视网视频，公开可看不代表可转载；使用许可待核实",
                   source_evidence=_evidence(source, title, "", publisher, "https://www.cctv.com/",
                                             date, "publisher_search"))
        result.append(row)
    return result


def cctv_search(query: str, page: int = 1) -> list[dict]:
    def fetch():
        url = "https://search.cctv.com/ifsearch.php?" + urlencode({
            "page": page, "qtext": query, "sort": "relevance", "type": "video", "vtime": "-1"})
        return parse_cctv_search(_json(url))
    return _cached("cctv", query if page == 1 else f"{query}\npage={page}", fetch)


def _json_assignment(markup: str, name: str) -> dict:
    match = re.search(r"(?:var\s+)?" + re.escape(name) + r"\s*=\s*", markup)
    if not match:
        return {}
    try:
        value, _ = json.JSONDecoder().raw_decode(markup[match.end():])
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}


def _walk(value, key: str):
    if isinstance(value, dict):
        if key in value:
            yield value[key]
        for item in value.values():
            yield from _walk(item, key)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item, key)


def _yt_text(value) -> str:
    if not isinstance(value, dict):
        return ""
    return value.get("simpleText") or "".join(r.get("text", "") for r in value.get("runs", []))


def parse_youtube_search(markup: str) -> list[dict]:
    data = _json_assignment(markup, "ytInitialData")
    if not data:
        raise ValueError("YouTube 公开检索不可读，可能需要同意/验证；未继续请求")
    result, seen = [], set()
    # Limit to real search contents; never treat recommendations as an exact match.
    contents = data.get("contents", {}).get("twoColumnSearchResultsRenderer", {}).get("primaryContents", {})
    for item in _walk(contents, "videoRenderer"):
        ident = item.get("videoId", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", ident) or ident in seen:
            continue
        seen.add(ident)
        source = "https://www.youtube.com/watch?v=" + ident
        title = _yt_text(item.get("title", {}))
        thumb = max(item.get("thumbnail", {}).get("thumbnails", []),
                    key=lambda t: _number(t.get("width")), default={}).get("url", "")
        author = item.get("ownerText") or item.get("longBylineText") or {}
        publisher = _yt_text(author)
        runs = author.get("runs", [])
        end = runs[0].get("navigationEndpoint", {}) if runs else {}
        publisher_url = end.get("commandMetadata", {}).get("webCommandMetadata", {}).get("url", "")
        publisher_url = urljoin("https://www.youtube.com/", publisher_url) if publisher_url else ""
        desc = " ".join(_yt_text(d.get("snippetText", {})) for d in item.get("detailedMetadataSnippets", []))
        published = _yt_text(item.get("publishedTimeText", {}))
        row = _candidate("youtube", title, source, _http_url(thumb))
        row.update(video_id=ident, publisher=publisher, publisher_url=publisher_url,
                   duration=_duration(_yt_text(item.get("lengthText", {}))), description=_clean(desc),
                   published_at=published, source_verified=True,
                   publisher_verified=any("VERIFIED" in b.get("metadataBadgeRenderer", {}).get("style", "")
                                          for b in item.get("ownerBadges", [])),
                   license="YouTube 来源视频，下载与转载许可待核实",
                   source_evidence=_evidence(source, title, desc, publisher, publisher_url,
                                             published, "publisher_search"))
        result.append(row)
        if len(result) >= 16:
            break
    return result


def youtube_search(query: str) -> list[dict]:
    def fetch():
        data, _ = _fetch("https://www.youtube.com/results?" + urlencode({"search_query": query}))
        return parse_youtube_search(data.decode("utf-8", errors="replace"))
    return _cached("youtube", query, fetch)


def parse_dailymotion_search(body: dict) -> list[dict]:
    result = []
    for item in body.get("list", [])[:12]:
        source, thumb = _http_url(item.get("url")), _http_url(item.get("thumbnail_url"))
        if not source or not _host(source, "dailymotion.com"):
            continue
        title, description = _clean(item.get("title")), _clean(item.get("description"))
        publisher, pub_url = _clean(item.get("owner.screenname")), _http_url(item.get("owner.url"))
        created = _number(item.get("created_time"))
        date = datetime.fromtimestamp(created, UTC).isoformat() if created else ""
        row = _candidate("dailymotion", title, source, thumb)
        row.update(video_id=str(item.get("id", "")), duration=_duration(item.get("duration")),
                   description=description, publisher=publisher, publisher_url=pub_url, published_at=date,
                   source_verified=True, license="Dailymotion 来源视频，下载与转载许可待核实",
                   source_evidence=_evidence(source, title, description, publisher, pub_url, date, "publisher_search"))
        result.append(row)
    return result


def dailymotion_search(query: str) -> list[dict]:
    def fetch():
        url = "https://api.dailymotion.com/videos?" + urlencode({"search": query, "limit": 10,
            "fields": "id,title,description,url,thumbnail_url,duration,owner.screenname,owner.url,created_time"})
        return parse_dailymotion_search(_json(url))
    return _cached("dailymotion", query, fetch)


def nasa_search(query: str) -> list[dict]:
    def fetch():
        body = _json("https://images-api.nasa.gov/search?" + urlencode({
            "q": query, "media_type": "video", "page_size": 8}))
        result = []
        for item in body.get("collection", {}).get("items", [])[:8]:
            data = item.get("data", [{}])[0]
            ident = str(data.get("nasa_id", ""))
            if not ident or data.get("media_type") != "video":
                continue
            source = "https://images.nasa.gov/details/" + quote(ident, safe="")
            thumb = next((_http_url(x.get("href")) for x in item.get("links", []) if x.get("rel") == "preview"), "")
            title, desc = _clean(data.get("title")), _clean(data.get("description"))
            publisher = "NASA / " + _clean(data.get("center", ""))
            result.append(_candidate("nasa", title, source, thumb,
                collection_url=_http_url(item.get("href")), description=desc,
                publisher=publisher, publisher_url="https://www.nasa.gov/", source_verified=True,
                published_at=_clean(data.get("date_created")),
                license="NASA Media Usage Guidelines；需核对第三方版权、署名及标志/肖像使用条件",
                license_url="https://www.nasa.gov/nasa-brand-center/images-and-media/",
                source_evidence=_evidence(source, title, desc, publisher, "https://www.nasa.gov/",
                                          data.get("date_created", ""), "publisher_search")))
        return result
    return _cached("nasa", query, fetch)


def archive_search(query: str) -> list[dict]:
    def fetch():
        # Quote user terms: do not let a generated search string become Solr syntax.
        terms = re.sub(r'["\\:()\[\]{}]', " ", query).strip()
        url = "https://archive.org/advancedsearch.php?" + urlencode({
            "q": f'mediatype:movies AND (title:("{terms}") OR description:("{terms}"))',
            "output": "json", "rows": 8, "fl[]": "identifier,title,description,creator,date,licenseurl"})
        body = _json(url)
        result = []
        for item in body.get("response", {}).get("docs", []):
            ident = item.get("identifier", "")
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,150}", ident):
                continue
            source = "https://archive.org/details/" + ident
            title, desc = _clean(item.get("title")), _clean(item.get("description"))
            publisher = _clean(item.get("creator") or "Internet Archive 上传者，待核实")
            license_url = item.get("licenseurl", "")
            if isinstance(license_url, list):
                license_url = license_url[0] if license_url else ""
            result.append(_candidate("archive", title, source, "https://archive.org/services/img/" + ident,
                archive_id=ident, description=desc, publisher=publisher, publisher_url=source,
                source_verified=True, published_at=_clean(item.get("date")),
                license="Internet Archive 条目上传者声明；需核对原作者及许可", license_url=_http_url(license_url),
                source_evidence=_evidence(source, title, desc, publisher, source,
                                          item.get("date", ""), "publisher_search")))
        return result
    return _cached("archive", query, fetch)


def commons_video_search(query: str) -> list[dict]:
    from .sources import commons
    result = []
    for item in commons(query + " filetype:video"):
        if item.get("kind") != "video":
            continue
        source = item.get("source_url", "")
        result.append(item | {"source_verified": True, "publisher": item.get("author", ""),
                              "source_evidence": _evidence(source, item["title"], item.get("description", ""),
                                  item.get("author", ""), source, "", "publisher_search"),
                              "download_status": "unchecked"})
    return result


def pexels_search(query: str, key: str) -> list[dict]:
    if not key:
        return []
    def fetch():
        body = _json("https://api.pexels.com/videos/search?" + urlencode({
            "query": query, "per_page": 10}), headers={"Authorization": key})
        result = []
        for item in body.get("videos", [])[:10]:
            source = _http_url(item.get("url"))
            if not source:
                continue
            user = item.get("user", {})
            title = _clean(urlparse(source).path.rsplit("/", 2)[-2].replace("-", " "))
            streams = [{"url": _http_url(f.get("link")), "format": "file", "width": int(_number(f.get("width"))),
                        "height": int(_number(f.get("height")))} for f in item.get("video_files", [])
                       if f.get("file_type") == "video/mp4" and _http_url(f.get("link"))]
            row = _candidate("pexels", title, source, _http_url(item.get("image")),
                             duration=_duration(item.get("duration")), width=int(_number(item.get("width"))),
                             height=int(_number(item.get("height"))), media_streams=streams,
                             publisher=_clean(user.get("name")), publisher_url=_http_url(user.get("url")),
                             source_verified=True, author=_clean(user.get("name")),
                             license="Pexels License；不得暗示人物代言，仍需逐项核对",
                             license_url="https://www.pexels.com/license/")
            row["source_evidence"] = _evidence(source, title, "Pexels 视频素材", row["publisher"],
                                                row["publisher_url"], "", "publisher_search")
            result.append(row)
        return result
    # Results are independent of the secret; never cache credentials or headers.
    return _cached("pexels", query, fetch)


def pixabay_video_search(query: str, key: str) -> list[dict]:
    if not key:
        return []
    from .sources import pixabay
    # The existing adapter owns the required 24-hour API-response cache.
    # Do not request the photo endpoint for a video search.
    return [row for row in pixabay(query, key, video_only=True) if row.get("kind") == "video"]


def _commons(candidate: dict, check: Check) -> dict:
    source = candidate.get("source_url", "")
    if not _host(source, "commons.wikimedia.org"):
        raise ValueError("Commons 素材缺少可核对的文件页面")
    title = unquote(urlparse(source).path.removeprefix("/wiki/"))
    if not title.startswith("File:"):
        title = "File:" + candidate.get("title", "")
    body = _json("https://commons.wikimedia.org/w/api.php?" + urlencode({
        "action": "query", "format": "json", "prop": "videoinfo", "viprop": "url|size|derivatives", "titles": title}), check=check)
    streams = []
    duration = candidate.get("duration", 0)
    for page in body.get("query", {}).get("pages", {}).values():
        for info in page.get("videoinfo", []):
            duration = _duration(info.get("duration")) or duration
            for item in info.get("derivatives", []):
                url = _http_url(item.get("src"))
                if not url or not item.get("type", "").startswith("video/"):
                    continue
                streams.append({"url": url, "format": "file", "width": int(_number(item.get("width"))),
                                "height": int(_number(item.get("height"))),
                                "bandwidth": _number(item.get("bandwidth"))})
    if not streams:
        return _generic(candidate, check)
    return candidate | {"media_streams": streams, "duration": duration}


def _nasa(candidate: dict, check: Check) -> dict:
    url = candidate.get("collection_url", "")
    if _host(url, "images-assets.nasa.gov"):
        data, _ = _fetch(url, check=check)
        try:
            files = json.loads(data)
        except ValueError:
            raise ValueError("NASA 返回的资源清单格式已变化") from None
    else:
        # A source restored from an older index may lack the stable manifest.
        # Resolve the original NASA id via the documented official asset API.
        source = urlparse(candidate.get("source_url", ""))
        ident = candidate.get("video_id") or unquote(source.path.removeprefix("/details/"))
        if source.hostname != "images.nasa.gov" or not source.path.startswith("/details/") or not ident:
            raise ValueError("NASA 素材缺少官方资源清单或条目编号")
        body = _json("https://images-api.nasa.gov/asset/" + quote(str(ident), safe=""), check=check)
        files = [r.get("href") for r in body.get("collection", {}).get("items", []) if isinstance(r, dict)]
    if not isinstance(files, list):
        raise ValueError("NASA 返回的资源清单格式已变化")
    streams = []
    for entry in files:
        if not isinstance(entry, str) or not _host(entry, "nasa.gov") or not re.search(r"\.mp4$", entry, re.I):
            continue
        # NASA gives standard derivative names. These are selection hints, never
        # a verified resolution; actual width/height come from the clip probe.
        name = urlparse(entry).path.lower()
        hint = 180 if "~preview" in name or "~mobile" in name else 360 if "~small" in name else 720 if "~medium" in name else 2160 if "~orig" in name else 1080
        streams.append({"url": entry.replace("http://", "https://", 1), "format": "file",
                        "width": 0, "height": 0, "quality_hint": hint})
    return candidate | {"media_streams": streams}


def _stock_video(candidate: dict, check: Check) -> dict:
    """Refresh official rendition metadata; no ephemeral URLs enter the index."""
    from . import config, stock_sources
    provider = candidate["provider"]
    cfg = config.read()
    access = stock_sources.registry(cfg)[provider]
    if access["status"] != "configured":
        raise ValueError(access["reason"])
    source = candidate.get("source_url", "")
    if not stock_sources._url(source, stock_sources.SOURCE_HOSTS[provider]):
        raise ValueError("素材来源与提供方不一致")
    ident = str(candidate.get("video_id") or "")
    if not ident:
        match = re.search(r"(?:-|/)(\d+)/?$", urlparse(source).path)
        ident = match[1] if match else ""
    if not re.fullmatch(r"\d{1,20}", ident):
        raise ValueError("素材缺少可核对的视频编号")
    return candidate | stock_sources.details(provider, ident, source, settings=cfg, check=check)


def _archive(candidate: dict, check: Check) -> dict:
    ident = candidate.get("archive_id") or urlparse(candidate["source_url"]).path.rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,150}", ident):
        raise ValueError("Internet Archive 条目标识无效")
    body = _json("https://archive.org/metadata/" + ident, check=check)
    metadata = body.get("metadata", {})
    if body.get("is_dark") or metadata.get("access-restricted-item") == "true":
        raise ValueError("Internet Archive 条目访问受限，未下载")
    files = {str(f.get("name", "")): f for f in body.get("files", [])}
    families = {}
    for name, f in files.items():
        if not re.search(r"\.(mp4|webm|ogv)$", name, re.I) or f.get("private") == "true":
            continue
        ancestor = f
        seen = set()
        while ancestor.get("original") in files and ancestor.get("original") not in seen:
            seen.add(ancestor["original"])
            ancestor = files[ancestor["original"]]
        family = str(ancestor.get("name") or name)
        families.setdefault(family, []).append({
            "url": "https://archive.org/download/" + ident + "/" + quote(name, safe=""),
            "format": "file", "width": int(_number(f.get("width"))),
            "height": int(_number(f.get("height"))), "filesize": int(_number(f.get("size"))),
            "source_duration": _duration(f.get("length")), "media_family": family,
            "quality_hint": 720 if "720" in name or "HD" in f.get("format", "") else 480})
    # An Archive item can contain several different movies. Keep preview and
    # final variants tied to one recorded original, never mix arbitrary files.
    selected_family = candidate.get("archive_media_family")
    if selected_family not in families and families:
        selected_family = max(families, key=lambda f: max(_short_edge(s) for s in families[f]))
    streams = families.get(selected_family, [])
    duration = max((s["source_duration"] for s in streams), default=0) or candidate.get("duration", 0)
    return candidate | {"media_streams": streams, "archive_media_family": selected_family, "duration": duration}


def _recent_unavailable(candidates, minimum=720):
    """Apply URL availability after merging live metadata, with expiring states."""
    from . import library
    by_source = {}
    for candidate in candidates:
        try:
            sid, _ = library.source_record(candidate)
        except ValueError:
            continue
        by_source.setdefault(sid, []).append(candidate)
    if not by_source:
        return candidates, {}
    unavailable, counts = set(), {}
    now = time.time()
    with library.connection() as con:
        ids = list(by_source)
        for start in range(0, len(ids), 400):
            batch = ids[start:start+400]
            rows = con.execute("SELECT source_id,state,checked,reason FROM source_catalog WHERE source_id IN ("
                               + ",".join("?" for _ in batch) + ")", batch)
            for sid, state, checked, reason in rows:
                if state == "preview_only":
                    dimensions = re.search(r"(\d+)\s*[×x]\s*(\d+)", reason)
                    threshold = re.search(r"低于\s*(\d+)\s*p", reason)
                    # A previous 1080p project can reject a real 720p stream.
                    # Skip only when its recorded size/upper bound is too low
                    # for this request; missing dimensions need re-inspection.
                    if dimensions:
                        if min(map(int, dimensions.groups())) >= minimum:
                            continue
                    elif not threshold or int(threshold[1]) > minimum:
                        continue
                if 0 <= now-checked < UNAVAILABLE_TTLS.get(state, 0):
                    unavailable.add(sid)
                    counts[state] = counts.get(state, 0)+len(by_source[sid])
    skipped = {id(c) for sid in unavailable for c in by_source[sid]}
    return [c for c in candidates if id(c) not in skipped], counts


def search_videos(query: str, *, include_stock: bool = True, pexels_key: str = "",
                  pixabay_key: str = "", channel_names: set[str] | None = None,
                  library_query: str | None = None, use_index: bool = True,
                  stock_query: str | None = None,
                  check: Check | None = None, diagnostics: dict | None = None,
                  min_video_height: int = 720, tencent_short_query: str = "",
                  tencent_query_state: TencentQueryFallback | None = None) -> tuple[list[dict], list[str]]:
    batches, failures = {}, []
    check = check or (lambda: None)
    check()
    from .publisher_videos import chinanews_search
    from .platform_videos import tencent_search, bilibili_search
    from .portal_videos import iqiyi_search, sohu_search, youku_search
    from .baidu_videos import baidu_video_search
    from . import source_index, stock_queries
    from .sources import tokens
    indexed = source_index.recall(library_query or query, channel_names=channel_names) if use_index else []
    suggestion = stock_queries.suggested_query(query, stock_query) if include_stock else None
    stock_names = {source_index.PROVIDERS[p] for p in stock_queries.STOCK_PROVIDERS}
    stock_index_channels = stock_names if channel_names is None else stock_names & channel_names
    if use_index and suggestion and stock_index_channels:
        # An English hint expands only the stock metadata index. Requiring all
        # words here prevents a lone "rolling" hit becoming a tea opportunity.
        extra = source_index.recall(suggestion, channel_names=stock_index_channels)
        indexed += [c for c in extra if c.get("provider") in stock_queries.STOCK_PROVIDERS and
                    stock_queries.metadata_matches(suggestion, c.get("title", "") + " " + c.get("description", ""))]
        if diagnostics is not None:
            diagnostics["stock_index_query"] = suggestion
    if not include_stock:
        indexed = [r for r in indexed if r.get("provider") not in {"pexels", "pixabay", "coverr", "commons", "nasa"}]
    # A large title index is not an available footage library. Only recently
    # playable, strongly related sources may suppress ordinary online discovery.
    requested = tokens(query)
    strong = [c for c in indexed if c.get("index_coverage", 0) >= .8 and c.get("thumbnail_url")
              and len(requested & tokens(c["title"] + " " + c.get("description", ""))) / max(1, len(requested)) >= .8
              and c.get("index_state") in ("available", "cached")
              and time.time() - c.get("checked_at", 0) < 86400]
    if len(strong) >= 16:
        if diagnostics is not None:
            diagnostics.update(index_only=True, providers=[])
        return [c | {"index_only_search": True} for c in strong[:40]], []
    # Ordinary publisher channels remain available. Stock providers add
    # metadata candidates when requested; a missing optional key never triggers
    # an HTTP call and no source result becomes an editorial approval.
    channels = [("腾讯视频", tencent_search), ("哔哩哔哩", bilibili_search),
                ("央视", cctv_search), ("中新网", chinanews_search),
                ("爱奇艺", iqiyi_search), ("搜狐", sohu_search), ("优酷", youku_search),
                ("百度视频", baidu_video_search)]
    stock_effective = query
    stock_reason = "original"
    if include_stock:
        from . import config, stock_sources
        cfg = config.read()
        if pexels_key:
            cfg = cfg | {"pexels_api_key": pexels_key}
        if pixabay_key:
            cfg = cfg | {"pixabay_api_key": pixabay_key}
        profile = stock_queries.query_profile(query)
        stock_effective = suggestion or (" ".join(profile["queries"]) if profile["complete"] else query)
        stock_reason = "model_group_stock_query" if suggestion else "static_bilingual_query" if profile["complete"] else "original"
        for provider, access in stock_sources.registry(cfg).items():
            if access["status"] == "configured":
                channels.append((source_index.PROVIDERS[provider],
                    lambda effective, provider=provider: stock_sources.search_items(
                        provider, effective, settings=cfg, check=check)))
    if channel_names is not None:
        channels = [(name, fn) for name, fn in channels if name in channel_names]
    def search_channel(item):
        name, function = item
        queued = time.monotonic()
        while not SEARCH_PROVIDER_SLOTS.acquire(timeout=.2):
            check()
        began = time.monotonic()
        diagnostic = {"provider": name, "queue_seconds": round(began-queued, 3)}
        _search_diagnostics.caches = []
        try:
            check()
            # Check shared feedback after the provider slot is acquired: queued
            # work can use an earlier empty result without another request.
            effective = (tencent_query_state.select(query, tencent_short_query)
                         if name == "腾讯视频" and tencent_query_state is not None else query)
            if name in stock_names:
                effective = stock_effective
            diagnostic.update(requested_query=query, effective_query=effective,
                              query_reason=stock_reason if name in stock_names else
                              "explicit_short_after_empty" if effective != query else "original")
            rows = function(effective)
            if name == "Pixabay" and suggestion:
                # One bounded public-metadata fallback for ordinary group
                # hints. Keep the original query and every approval condition.
                # Do not recursively broaden results or discard primary hits.
                fallback = stock_queries.missing_stock_term(effective, rows)
                if fallback:
                    check()
                    diagnostic["stock_term_fallback"] = {"query": fallback, "metadata_only": True}
                    try:
                        additional = function(fallback)
                        seen = {row["id"] for row in rows}
                        extra = [row for row in additional if row["id"] not in seen]
                        rows = [*rows, *extra]
                        diagnostic["stock_term_fallback"].update(status="ok", additional_candidates=len(extra))
                    except (ValueError, httpx.HTTPError) as exc:
                        # A failed second lookup must not erase valid primary
                        # results. Cancellation/deadline still propagates.
                        check()
                        diagnostic["stock_term_fallback"].update(status="error", error=str(exc)[:160])
            if name == "腾讯视频" and tencent_query_state is not None and not rows:
                diagnostic["empty_query_trigger"] = tencent_query_state.observe_empty(
                    query, effective, tencent_short_query)
            diagnostic.update(candidates=len(rows), status="ok")
            return rows, None, diagnostic
        except (ValueError, httpx.HTTPError) as exc:
            error = name+"公开检索未完成："+str(exc)[:160]
            diagnostic.update(candidates=0, status="error", error=error)
            return [], error, diagnostic
        finally:
            diagnostic["elapsed_seconds"] = round(time.monotonic()-began, 3)
            diagnostic["caches"] = _search_diagnostics.caches
            del _search_diagnostics.caches
            SEARCH_PROVIDER_SLOTS.release()
    provider_diagnostics = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(search_channel, item): item[0] for item in channels}
        for future in as_completed(futures):
            name = futures[future]
            rows, error, diagnostic = future.result()
            batches[name] = rows
            provider_diagnostics[name] = diagnostic
            if error:
                failures.append(error)
    check()
    if diagnostics is not None:
        diagnostics.update(index_only=False, providers=[provider_diagnostics[name] for name, _ in channels])
    result = list(indexed)
    for name, _ in channels:
        result.extend(batches[name])
    result = list({c.get("source_url") or c["id"]: c for c in result}.values())
    from . import library
    try:
        library.remember_sources(result)
        result, skipped = _recent_unavailable(result, min_video_height)
        if diagnostics is not None:
            diagnostics["skipped_recent_unavailable"] = skipped
    except (OSError, ValueError, sqlite3.Error):
        failures.append("来源索引暂时无法写入，本轮检索结果仍可使用")
    return result, failures


class _PageMetadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta: dict[str, str] = {}
        self.json_scripts: list[str] = []
        self._json_script = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            key = attrs.get("property") or attrs.get("name")
            if key and attrs.get("content"):
                self.meta[key.lower()] = attrs["content"]
        if tag == "script" and attrs.get("type", "").lower() == "application/ld+json":
            self._json_script = True

    def handle_endtag(self, tag):
        if tag == "script":
            self._json_script = False

    def handle_data(self, data):
        if self._json_script:
            self.json_scripts.append(data)


def parse_hls(markup: str, url: str) -> dict:
    """Only ordinary finite VOD manifests; never follow key/DRM/map/file URLs."""
    if not markup.lstrip().startswith("#EXTM3U"):
        raise ValueError("播放地址未返回 HLS 清单，可能被来源限制")
    if any(tag in markup for tag in ("#EXT-X-KEY:", "#EXT-X-SESSION-KEY:", "#EXT-X-MAP:",
                                    "#EXT-X-BYTERANGE:", "#EXT-X-GAP", "#EXT-X-I-FRAMES-ONLY")):
        raise ValueError("此播放清单含加密或暂不支持的分段格式，未尝试破解或下载")
    variants, segments, pending, elapsed = [], [], None, 0.0
    duration = None
    for line in markup.splitlines():
        line = line.strip()
        if line.startswith("#EXT-X-STREAM-INF:"):
            attrs = dict(re.findall(r'([A-Z-]+)=("[^"]*"|[^,]+)', line.split(":", 1)[1]))
            wh = attrs.get("RESOLUTION", "0x0").split("x")
            pending = {"width": int(_number(wh[0])), "height": int(_number(wh[-1])),
                       "bandwidth": _number(attrs.get("BANDWIDTH")), "format": "hls"}
        elif line.startswith("#EXTINF:"):
            duration = _number(line.split(":", 1)[1].split(",", 1)[0])
        elif line and not line.startswith("#"):
            resolved = _http_url(urljoin(url, line))
            if not resolved:
                raise ValueError("播放清单包含非公开 HTTP(S) 资源，已停止")
            if pending is not None:
                variants.append(pending | {"url": resolved})
                pending = None
            elif duration is not None:
                if duration <= 0 or duration > 120:
                    raise ValueError("来源单个视频分段过长，不符合小片段下载策略")
                segments.append({"url": resolved, "duration": duration, "start": elapsed})
                elapsed += duration
                duration = None
    if segments and "#EXT-X-ENDLIST" not in markup:
        raise ValueError("暂不自动截取直播流，请使用已完成的公开视频")
    if len(segments) > 7200 or elapsed > 12 * 3600:
        raise ValueError("来源视频过长，请选择短集锦或提供具体片段")
    if not variants and not segments:
        raise ValueError("播放清单没有可用视频片段")
    return {"variants": variants, "segments": segments, "duration": elapsed}


def _hls(url: str, check: Check, referer: str = "") -> dict:
    data, info = _fetch(url, limit=1024 * 1024, check=check,
                        headers={"Referer": referer} if referer else None)
    return parse_hls(data.decode("utf-8", errors="replace"), info["url"])


def _cctv(candidate: dict, check: Check) -> dict:
    source = candidate["source_url"]
    data, _ = _fetch(source, check=check)
    page = data.decode("utf-8", errors="replace")
    parser = _PageMetadata()
    parser.feed(page)
    match = re.search(r'\b(?:guid|videoCenterId)\s*=\s*["\x27]([a-fA-F0-9]{32})["\x27]', page)
    if not match:
        raise ValueError("央视原页未提供可读的公开视频标识")
    body = _json("https://vdn.apps.cntv.cn/api/getHttpVideoInfo.do?" + urlencode({"pid": match[1]}), check=check)
    if str(body.get("is_protected", "0")) != "0" or str(body.get("is_invalid_copyright", "0")) != "0" \
            or str(body.get("public", "1")) != "1" or body.get("ack") != "yes":
        raise ValueError("央视标记此视频为受保护或不可公开访问，未下载")
    title = _clean(body.get("title") or parser.meta.get("og:title") or candidate["title"])
    desc = _clean(parser.meta.get("description") or parser.meta.get("og:description") or body.get("tag"))
    publisher = _clean(body.get("play_channel") or candidate.get("publisher") or "央视网")
    date = _clean(body.get("f_pgmtime") or candidate.get("published_at", ""))
    streams = []
    # The alternative enc/h5e manifests contain proprietary protected TS payloads;
    # a listed 720p size is not proof that ordinary FFmpeg can decode the footage.
    hls_url = _http_url(body.get("hls_url"))
    if hls_url:
        parsed = _hls(hls_url, check, source)
        streams = parsed["variants"] or [{"url": hls_url, "format": "hls", "height": 0, "width": 0}]
    if not streams:
        for chapters in (body.get("video", {}).get("chapters3", []), body.get("video", {}).get("chapters", [])):
            if len(chapters) == 1 and _http_url(chapters[0].get("url")):
                streams = [{"url": chapters[0]["url"], "format": "file", "width": 0, "height": 0}]
                break
    thumbnail = _http_url(body.get("image")) or candidate.get("thumbnail_url", "")
    return candidate | {"title": title, "description": desc, "publisher": publisher,
                        "published_at": date, "publisher_url": "https://www.cctv.com/", "source_verified": True,
                        "duration": _duration(body.get("video", {}).get("totalLength")),
                        "thumbnail_url": thumbnail, "preview_urls": [{"url": thumbnail, "at": None,
                                                                       "kind": "publisher_thumbnail"}],
                        "source_evidence": _evidence(source, title, desc, publisher, "https://www.cctv.com/",
                                                      date, "publisher_page"), "media_streams": streams}


def _ytdlp_command() -> list[str]:
    bundled = paths.ROOT / "runtime" / "yt-dlp" / ("yt-dlp.exe" if os.name == "nt" else "yt-dlp")
    if bundled.is_file():
        return [str(bundled)]
    if not getattr(sys, "frozen", False) and importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    return []


def _youtube_js_runtime() -> str:
    bundled = paths.ROOT / "runtime" / "deno" / ("deno.exe" if os.name == "nt" else "deno")
    if bundled.is_file():
        return "deno:" + str(bundled)
    for runtime in ("deno", "node"):
        located = shutil.which(runtime)
        if located:
            return runtime + ":" + located
    return ""


def _stop_public_extractor(proc: subprocess.Popen) -> None:
    """Terminate this extractor and its own JS children, never a user browser."""
    if os.name == "nt":
        taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
        try:
            subprocess.run([str(taskkill), "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _run_ytdlp_json(args: list[str], check: Check, *, timeout: float = 75,
                    output_limit: int = METADATA_LIMIT, label: str = "YouTube") -> dict:
    """Bound JSON and diagnostics in memory; never include URLs in an error."""
    buffers = [bytearray(), bytearray()]
    overflow = threading.Event()

    def collect(pipe, dest, limit):
        try:
            while chunk := pipe.read(64 * 1024):
                if len(dest) + len(chunk) > limit:
                    overflow.set()
                    return
                dest.extend(chunk)
        except (OSError, ValueError):
            pass

    # The child only needs runtime/OS variables. Do not pass local API keys,
    # Python/Node injection settings, proxy settings, or a user's HOME profile.
    allowed_env = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR",
                   "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed_env}
    with tempfile.TemporaryDirectory(prefix="jinggao-video-metadata-") as td:
        env.update(PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1", DENO_NO_UPDATE_CHECK="1",
                   DENO_DIR=str(Path(td) / "deno-cache"), NO_COLOR="1")
        try:
            proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, cwd=td, env=env,
                                    start_new_session=os.name != "nt",
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError:
            raise ValueError(f"{label}解析器无法启动，请检查完整运行包") from None
        readers = [threading.Thread(target=collect, args=(pipe, dest, limit), daemon=True)
                   for pipe, dest, limit in ((proc.stdout, buffers[0], output_limit),
                                             (proc.stderr, buffers[1], 256 * 1024))]
        for reader in readers:
            reader.start()
        started = time.monotonic()
        try:
            while proc.poll() is None:
                check()
                if overflow.is_set():
                    raise ValueError(f"{label}来源元数据超过读取上限，已停止")
                if time.monotonic() - started > timeout:
                    raise ValueError(f"{label}公开解析超时，未继续重试")
                time.sleep(0.1)
            check()
            for reader in readers:
                reader.join(timeout=2)
            if any(reader.is_alive() for reader in readers):
                _stop_public_extractor(proc)
                raise ValueError(f"{label}公开解析未正常结束，已停止")
            if overflow.is_set():
                raise ValueError(f"{label}来源元数据超过读取上限，已停止")
            if proc.returncode:
                raise ValueError(f"{label}公开解析未完成，可能需要验证或访问权限")
            try:
                body = json.loads(buffers[0])
            except (ValueError, UnicodeDecodeError):
                raise ValueError(f"{label}公开解析没有返回有效资料") from None
            if not isinstance(body, dict):
                raise ValueError(f"{label}公开解析返回了不支持的资料格式")
            return body
        finally:
            if proc.poll() is None:
                _stop_public_extractor(proc)
            for pipe in (proc.stdout, proc.stderr):
                pipe.close()
            for reader in readers:
                reader.join(timeout=1)


def _youtube_ytdlp(candidate: dict, check: Check) -> dict:
    """Optional official extractor for one ordinary public YouTube watch page."""
    parsed = urlparse(candidate.get("source_url", ""))
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("公开视频解析仅支持标准 YouTube HTTPS 视频页面")
    if host == "youtu.be":
        video_id = parsed.path.strip("/")
    elif host in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
        video_id = parse_qs(parsed.query).get("v", [""])[0] if parsed.path == "/watch" else ""
        if not video_id and re.fullmatch(r"/(?:shorts|embed|live)/[A-Za-z0-9_-]{11}/?", parsed.path):
            video_id = parsed.path.strip("/").split("/")[1]
    else:
        video_id = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("公开视频解析只接受单个 YouTube 视频，不接受搜索页、列表或其他网站")
    command, runtime = _ytdlp_command(), _youtube_js_runtime()
    if not command or not runtime:
        raise ValueError("可选 YouTube 解析器或 JavaScript 运行时尚未安装，当前只能检索预览")
    source = "https://www.youtube.com/watch?v=" + video_id
    args = [*command, "--ignore-config", "--no-cache-dir", "--no-cookies", "--no-cookies-from-browser",
            "--no-plugin-dirs", "--no-remote-components", "--no-update", "--no-playlist", "--skip-download",
            "--dump-single-json", "--no-check-formats", "--use-extractors", "youtube", "--proxy", "",
            "--socket-timeout", "20", "--retries", "0", "--fragment-retries", "0", "--extractor-retries", "0",
            "--file-access-retries", "0", "--no-geo-bypass", "--no-js-runtimes", "--js-runtimes", runtime,
            "--", source]
    body = _run_ytdlp_json(args, check)
    if body.get("id") != video_id or body.get("_type", "video") not in (None, "video") or body.get("entries"):
        raise ValueError("YouTube 解析返回了不同视频或列表，已停止")
    if body.get("availability") != "public" or body.get("is_live") or body.get("has_drm") or _number(body.get("age_limit")) >= 18:
        raise ValueError("YouTube 资料并非普通公开视频，未读取登录态或受保护媒体")
    by_resolution = {}
    for f in body.get("formats", []):
        url = _http_url(f.get("url"))
        protocol, container = f.get("protocol"), f.get("ext")
        width, height = int(_number(f.get("width"))), int(_number(f.get("height")))
        if not url or not _host(url, "googlevideo.com") or urlparse(url).scheme != "https" or f.get("has_drm") \
                or f.get("vcodec") in (None, "none") or not 0 < width <= 16384 or not 0 < height <= 16384 \
                or container not in ("mp4", "webm") or protocol not in ("https", "m3u8", "m3u8_native"):
            continue
        kind = "file" if protocol == "https" else "hls"
        stream = {"url": url, "format": kind, "container": container, "width": width, "height": height,
                  "filesize": int(_number(f.get("filesize"))), "bandwidth": _number(f.get("tbr")) * 1000,
                  "fps": _number(f.get("fps")), "vcodec": _clean(f.get("vcodec"), 80),
                  "format_id": _clean(f.get("format_id"), 80)}
        # Prefer a seekable H.264 file at each size. HLS remains subject to the
        # same encrypted/unsupported-manifest checks as every other provider.
        preference = (kind == "file", stream["vcodec"].startswith("avc1"), stream["fps"], stream["bandwidth"])
        old = by_resolution.get((width, height))
        if not old or preference > old[0]:
            by_resolution[(width, height)] = (preference, stream)
    streams = [value[1] for value in by_resolution.values()]
    if not streams:
        raise ValueError("YouTube 公开解析未提供普通可用视频流，已停止")
    title, desc = _clean(body.get("title")), _clean(body.get("description"))
    publisher, channel_id = _clean(body.get("channel") or body.get("uploader")), str(body.get("channel_id", ""))
    if not title or not publisher or not re.fullmatch(r"[A-Za-z0-9_-]{24}", channel_id):
        raise ValueError("YouTube 来源缺少可核对的标题或发布者，未采信")
    pub_url = "https://www.youtube.com/channel/" + channel_id
    date = str(body.get("upload_date") or "")
    date = f"{date[:4]}-{date[4:6]}-{date[6:]}" if re.fullmatch(r"\d{8}", date) else ""
    thumbnail = _http_url(body.get("thumbnail")) or candidate.get("thumbnail_url", "")
    return candidate | {"url": source, "source_url": source, "title": title, "description": desc,
                        "publisher": publisher, "publisher_url": pub_url, "published_at": date,
                        "publisher_verified": bool(body.get("channel_is_verified")), "source_verified": True,
                        "duration": _duration(body.get("duration")), "media_streams": streams,
                        "media_extractor": "yt-dlp-public", "thumbnail_url": thumbnail,
                        "preview_urls": [{"url": thumbnail, "at": None, "kind": "publisher_thumbnail"}] if thumbnail else [],
                        "source_evidence": _evidence(source, title, desc, publisher, pub_url, date, "publisher_page")}


def _youtube(candidate: dict, check: Check) -> dict:
    source = candidate["source_url"]
    data, _ = _fetch(source, check=check)
    page = data.decode("utf-8", errors="replace")
    body = _json_assignment(page, "ytInitialPlayerResponse")
    if not body:
        raise ValueError("YouTube 原页播放器资料不可读，未尝试登录或验证")
    details = body.get("videoDetails", {})
    micro = body.get("microformat", {}).get("playerMicroformatRenderer", {})
    title, desc = _clean(details.get("title") or candidate["title"]), _clean(details.get("shortDescription"))
    channel_id = details.get("channelId", "")
    pub_url = "https://www.youtube.com/channel/" + channel_id if re.fullmatch(r"[A-Za-z0-9_-]{24}", channel_id) else candidate.get("publisher_url", "")
    publisher = _clean(details.get("author") or candidate.get("publisher", ""))
    published = _clean(micro.get("publishDate") or candidate.get("published_at", ""))
    candidate = candidate | {"title": title, "description": desc, "duration": _duration(details.get("lengthSeconds")),
                              "publisher": publisher, "publisher_url": pub_url, "published_at": published,
                              "source_verified": True,
                              "source_evidence": _evidence(source, title, desc, publisher, pub_url, published, "publisher_page")}
    status = body.get("playabilityStatus", {})
    if status.get("status") != "OK":
        return candidate | {"media_streams": [], "download_reason": "YouTube 要求登录、地区权限或验证；已停止自动取流"}
    streams = []
    data = body.get("streamingData", {})
    for f in [*data.get("formats", []), *data.get("adaptiveFormats", [])]:
        url = _http_url(f.get("url"))
        if url and f.get("width") and f.get("height") and not f.get("drmFamilies") and not f.get("signatureCipher"):
            streams.append({"url": url, "format": "file", "width": int(_number(f.get("width"))),
                            "height": int(_number(f.get("height"))), "filesize": int(_number(f.get("contentLength"))),
                            "container": "webm" if "video/webm" in f.get("mimeType", "") else "mp4"})
    if data.get("hlsManifestUrl") and not streams:
        parsed = _hls(data["hlsManifestUrl"], check, source)
        streams = parsed["variants"] or [{"url": data["hlsManifestUrl"], "format": "hls", "width": 0, "height": 0}]
    candidate["available_format_heights"] = sorted({int(_number(f.get("height"))) for f in data.get("adaptiveFormats", [])
                                                    if f.get("height")})
    candidate["media_streams"] = streams
    if not streams:
        if any(f.get("drmFamilies") for f in [*data.get("formats", []), *data.get("adaptiveFormats", [])]):
            candidate["download_reason"] = "YouTube 仅提供受保护格式，已停止自动取流"
        else:
            try:
                return _youtube_ytdlp(candidate, check)
            except ValueError as exc:
                candidate["download_reason"] = str(exc)
    return candidate


def _dailymotion(candidate: dict, check: Check) -> dict:
    source = candidate["source_url"]
    ident = candidate.get("video_id") or urlparse(source).path.rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Za-z0-9]+", ident):
        raise ValueError("Dailymotion 视频标识无效")
    body = _json("https://www.dailymotion.com/player/metadata/video/" + ident, check=check)
    if body.get("error") or body.get("private") or body.get("is_password_protected") or body.get("protected_delivery"):
        raise ValueError("Dailymotion 视频需要权限或受保护，已停止取流")
    owner = body.get("owner", {})
    title, desc = _clean(body.get("title") or candidate["title"]), _clean(body.get("description"))
    publisher, pub_url = _clean(owner.get("screenname")), _http_url(owner.get("url"))
    streams = []
    for formats in body.get("qualities", {}).values():
        for f in formats:
            url = _http_url(f.get("url"))
            if not url:
                continue
            if "mpegurl" in f.get("type", "").lower():
                parsed = _hls(url, check, source)
                streams.extend(parsed["variants"] or [{"url": url, "format": "hls", "width": 0, "height": 0}])
            elif f.get("type", "").startswith("video/"):
                streams.append({"url": url, "format": "file", "width": 0, "height": 0})
    return candidate | {"title": title, "description": desc, "publisher": publisher, "publisher_url": pub_url,
                        "duration": _duration(body.get("duration")), "source_verified": True, "media_streams": streams,
                        "source_evidence": _evidence(source, title, desc, publisher, pub_url,
                                                      candidate.get("published_at", ""), "publisher_page")}


def _generic(candidate: dict, check: Check) -> dict:
    url = _http_url(candidate.get("url"))
    if re.search(r"\.(mp4|mov|webm|ogv|ogg)(?:$|\?)", url, re.I):
        return candidate | {"media_streams": [{"url": url, "format": "file", "width": candidate.get("width", 0),
                                                "height": candidate.get("height", 0), "filesize": candidate.get("filesize", 0)}]}
    if ".m3u8" in url:
        parsed = _hls(url, check, candidate.get("source_url", ""))
        return candidate | {"media_streams": parsed["variants"] or [{"url": url, "format": "hls", "width": 0, "height": 0}]}
    source = candidate.get("source_url") or url
    data, _ = _fetch(source, check=check)
    parser = _PageMetadata()
    parser.feed(data.decode("utf-8", errors="replace"))
    for script in parser.json_scripts:
        try:
            body = json.loads(script)
        except ValueError:
            continue
        objects = body if isinstance(body, list) else [body]
        if isinstance(body, dict):
            objects += body.get("@graph", [])
        for item in objects:
            if not isinstance(item, dict) or item.get("@type") != "VideoObject":
                continue
            content = _http_url(item.get("contentUrl"))
            if not content:
                continue
            title, desc = _clean(item.get("name") or candidate["title"]), _clean(item.get("description"))
            publisher = item.get("publisher") or {}
            publisher = publisher.get("name", "") if isinstance(publisher, dict) else str(publisher)
            pub_url = urlparse(source)._replace(path="/", query="", fragment="").geturl()
            row = candidate | {"title": title, "description": desc, "url": content,
                                "duration": _duration(item.get("duration")), "publisher": _clean(publisher),
                                "source_verified": True,
                                "source_evidence": _evidence(source, title, desc, publisher, pub_url,
                                                              item.get("uploadDate", ""), "publisher_page")}
            if not _http_url(row.get("thumbnail_url")):
                thumbnail = item.get("thumbnailUrl")
                if isinstance(thumbnail, list):
                    thumbnail = next((value for value in thumbnail if isinstance(value, str) and value), "")
                thumbnail = thumbnail if isinstance(thumbnail, str) and thumbnail else parser.meta.get("og:image", "")
                thumbnail = _http_url(urljoin(source, thumbnail)) if thumbnail else ""
                if thumbnail:
                    row["thumbnail_url"] = thumbnail
                    row["preview_urls"] = [{"url": thumbnail, "at": None, "kind": "publisher_thumbnail"}]
            if ".m3u8" in content:
                parsed = _hls(content, check, source)
                streams = parsed["variants"] or [{"url": content, "format": "hls", "width": 0, "height": 0}]
            elif re.search(r"\.(mp4|mov|webm|ogg|ogv)(?:$|\?)", content, re.I):
                streams = [{"url": content, "format": "file", "width": 0, "height": 0}]
            else:
                raise ValueError("原页没有明确的普通视频文件或 HLS 地址，未跟随嵌套页面")
            return row | {"media_streams": streams}
    raise ValueError("原页未公开提供支持的视频地址，请从原页获取授权片段后导入")


_inspection_cache = {}
_inspection_guard = threading.Lock()
INSPECTION_FIELDS = ("title", "description", "duration", "publisher", "publisher_url", "published_at",
                     "source_verified", "source_evidence", "available_format_heights", "media_streams",
                     "metadata_checked", "width", "height", "download_status", "download_reason",
                     "quality_verified", "inspection_at", "extractor", "media_extractor",
                     "stock_detail_expires_at", "stock_detail_cache_hit")


def inspect_video_candidate(candidate: dict, check: Check | None = None, *, force: bool = False) -> dict:
    """Share short-lived public metadata, never another shot's review or identity."""
    check = check or (lambda: None)
    check()
    source = candidate.get("source_url") or candidate.get("url", "")
    key = (str(paths.DATA), candidate.get("provider"), source)
    with execution.keyed_lock(("inspection", key), check=check):
        check()
        with _inspection_guard:
            cached = _inspection_cache.get(key)
        row = candidate
        if cached and not force:
            ttl = 600 if cached.get("media_streams") else 20
            if time.time()-cached.get("inspection_at", 0) < ttl:
                row = candidate | cached
        result = _inspect_video_candidate(row, check, force=force)
        from . import source_index
        source_index.record_check(result)
        with _inspection_guard:
            if len(_inspection_cache) >= 256 and key not in _inspection_cache:
                _inspection_cache.pop(next(iter(_inspection_cache)))
            _inspection_cache[key] = {k: result[k] for k in INSPECTION_FIELDS if k in result}
        return result


def _inspect_video_candidate(candidate: dict, check: Check | None = None, *, force: bool = False) -> dict:
    """Read only metadata/manifests: cover image is not proof of segment identity."""
    check = check or (lambda: None)
    check()
    row = dict(candidate)
    age = time.time() - _number(row.get("inspection_at"))
    stock_detail = row.get("provider") in ("pexels", "pixabay") and "stock_detail_expires_at" in row
    # Official stock details check their own credential-bound 24h cache. A
    # short inspection hit must not extend that deadline or reuse another key.
    if not force and not stock_detail and row.get("metadata_checked") and age < 600 and row.get("media_streams"):
        best = max(row["media_streams"], key=lambda s: (_short_edge(s), _number(s.get("bandwidth"))))
        minimum = int(_number(row.get("min_video_height"), 720))
        low = _short_edge(best) > 0 and _short_edge(best) < minimum
        row["download_status"] = "preview_only" if low else "available"
        row["download_reason"] = "已复用公开流资料，实际样段仍需审核" if not low else f"公开流短边低于 {minimum}p；可低清预览"
        return row
    if not force and row.get("metadata_checked") and age < 20 and row.get("download_status") in ("blocked", "unsupported"):
        return row
    source = row.get("source_url") or row.get("url", "")
    try:
        if _host(source, "cctv.com"):
            row = _cctv(row, check)
        elif _host(source, "chinanews.com.cn") or _host(source, "chinanews.com"):
            from .publisher_videos import chinanews
            row = chinanews(row, check)
        elif any(_host(source, host) for host in ("v.qq.com", "bilibili.com", "iqiyi.com", "v.youku.com", "tv.sohu.com")):
            from .platform_videos import inspect_platform
            row = inspect_platform(row, check)
        elif _host(source, "youtube.com") or _host(source, "youtu.be"):
            row = _youtube(row, check)
        elif _host(source, "dailymotion.com"):
            row = _dailymotion(row, check)
        elif row.get("provider") == "commons" or _host(source, "commons.wikimedia.org"):
            row = _commons(row, check)
        elif row.get("provider") == "nasa" or _host(source, "images.nasa.gov"):
            row = _nasa(row, check)
        elif row.get("provider") == "archive":
            row = _archive(row, check)
        elif row.get("provider") in ("pexels", "pixabay"):
            if (not row.get("media_streams") or force or stock_detail
                    or (row.get("metadata_checked") and age >= 600)):
                row = _stock_video(row, check)
        elif row.get("provider") == "coverr":
            raise ValueError("Coverr 服务集成授权尚未确认，当前仅展示来源状态")
        else:
            row = _generic(row, check)
        streams = row.get("media_streams", [])
        best = max(streams, key=lambda s: (_short_edge(s), _number(s.get("bandwidth"))), default={})
        row.update(metadata_checked=True, width=int(_number(best.get("width"))), height=int(_number(best.get("height"))))
        if streams:
            height = row["height"]
            minimum = int(_number(row.get("min_video_height"), 720))
            low = _short_edge(best) > 0 and _short_edge(best) < minimum
            row.update(download_status="preview_only" if low else "available",
                       download_reason=f"公开流为 {row['width']}×{height}，短边低于 {minimum}p 成片门槛；可低清预览" if low
                       else "已找到公开流；仍需小样段解码、人物与画面审核", quality_verified=False)
        else:
            row.update(download_status="unsupported", download_reason=row.get("download_reason") or "未找到公开可下载流")
    except ValueError as exc:
        row.update(metadata_checked=True, download_status="blocked", download_reason=str(exc), media_streams=[])
    row["inspection_at"] = time.time()
    row["kind"] = "video"
    return row


def _short_edge(stream: dict) -> float:
    width, height = _number(stream.get("width")), _number(stream.get("height"))
    return min(width, height) if width and height else height or _number(stream.get("quality_hint"))


def _choose_stream(candidate: dict, preview: bool) -> dict:
    streams = candidate.get("media_streams", [])
    if not streams:
        raise ValueError(candidate.get("download_reason") or "请先读取视频来源资料")
    if preview:
        small = [s for s in streams if 0 < _number(s.get("height") or s.get("quality_hint")) <= 480]
        return max(small, key=lambda s: _number(s.get("height") or s.get("quality_hint"))) if small else min(
            streams, key=lambda s: (_number(s.get("height") or s.get("quality_hint"), 720), _number(s.get("bandwidth"))))
    minimum = int(_number(candidate.get("min_video_height"), 720))
    adequate = [s for s in streams if not _short_edge(s) or _short_edge(s) >= minimum]
    if not adequate:
        raise ValueError(f"公开视频低于 {minimum}p 成片门槛，请换来源或明确降低项目清晰度要求")
    capped = [s for s in adequate if _short_edge(s) <= 1080]
    if candidate.get("prefer_efficient_stream"):
        return min(capped or adequate, key=lambda s: (_short_edge(s), _number(s.get("bandwidth"))))
    return max(capped or adequate, key=lambda s: (_short_edge(s), _number(s.get("bandwidth"))))


def select_segments(segments: list[dict], clip_start: float, duration: float) -> list[dict]:
    end = clip_start + duration
    selected = [s for s in segments if s["start"] < end - 0.001 and s["start"] + s["duration"] > clip_start + 0.001]
    if not selected:
        raise ValueError("指定时间范围不在来源视频内")
    if selected[-1]["start"] + selected[-1]["duration"] < end - 0.1:
        raise ValueError("来源剩余视频不足以覆盖所需镜头时长")
    if len(selected) > 30:
        raise ValueError("本次需要下载的分段过多，请缩短镜头")
    return selected


@contextmanager
def _range_proxy(url: str, limit: int, check: Check, referer: str):
    """A local, one-resource proxy lets FFmpeg seek without an unlimited downloader.

    Every upstream redirect is checked. The shared byte budget counts actual
    transferred data, including data read before FFmpeg closes a range request.
    """
    from .sources import public_url, download_slot
    state = {"bytes": 0, "error": "", "ranges": 0, "range_responses": 0, "source_size": 0}
    guard = threading.Lock()
    stopped = threading.Event()
    active_clients = set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path != "/media":
                self.send_error(404)
                return
            requested_range = self.headers.get("Range", "")
            if requested_range and not re.fullmatch(r"bytes=\d+-\d*", requested_range):
                self.send_error(416)
                return
            headers = {"User-Agent": AGENT, "Referer": str(httpx.URL(referer)) if referer else ""}
            if requested_range:
                headers["Range"] = requested_range
                with guard:
                    state["ranges"] += 1
            source = url
            sent_headers = False
            try:
                with network.client(timeout=15, follow_redirects=False, headers=headers) as client:
                    with guard:
                        active_clients.add(client)
                    for _ in range(5):
                        check()
                        public_url(source)
                        cooldown = download_slot(source)
                        with client.stream("GET", source) as response:
                            if response.is_redirect:
                                source = str(response.url.join(response.headers.get("location", "")))
                                continue
                            if response.status_code == 429:
                                retry = response.headers.get("Retry-After", "60")
                                delay = min(86400, max(1, int(retry))) if retry.isdigit() else 60
                                cooldown.write_text(json.dumps({"until": time.time() + delay}), "utf-8")
                                raise ValueError("来源 HTTP 429 限流，已遵守 Retry-After，稍后再试")
                            if response.status_code not in (200, 206):
                                raise ValueError(f"来源片段读取受限（HTTP {response.status_code}）")
                            # A source ignoring seek must not be repeatedly read from byte zero.
                            if requested_range and requested_range != "bytes=0-" and response.status_code != 206:
                                raise ValueError("来源不支持 HTTP Range 定位；未继续下载整部视频")
                            if response.status_code == 206:
                                content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)",
                                                             response.headers.get("Content-Range", ""))
                                offset = int(requested_range.split("=", 1)[1].split("-", 1)[0]) if requested_range else 0
                                if not content_range or int(content_range[1]) != offset:
                                    raise ValueError("来源返回不一致的 HTTP Range 数据，已停止")
                                with guard:
                                    state["range_responses"] += 1
                                    if content_range[3] != "*":
                                        state["source_size"] = int(content_range[3])
                            elif response.headers.get("Content-Length", "").isdigit():
                                with guard:
                                    state["source_size"] = int(response.headers["Content-Length"])
                            self.send_response(response.status_code)
                            for key in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"):
                                if response.headers.get(key):
                                    self.send_header(key, response.headers[key])
                            self.send_header("Connection", "close")
                            self.end_headers()
                            sent_headers = True
                            for chunk in response.iter_bytes(32 * 1024):
                                check()
                                if stopped.is_set():
                                    return
                                with guard:
                                    state["bytes"] += len(chunk)
                                    over = state["bytes"] > limit
                                if over:
                                    raise ValueError(f"片段读取超过 {limit // (1024 * 1024)} MB 预算，已停止")
                                self.wfile.write(chunk)
                            return
                    raise ValueError("来源重定向过多")
            except (BrokenPipeError, ConnectionResetError):
                # Normal FFmpeg seeking closes the old range before opening a new one.
                return
            except Exception as exc:
                with guard:
                    state["error"] = str(exc) if isinstance(exc, ValueError) else "公开媒体读取失败"
                if not sent_headers:
                    self.send_error(502)
            finally:
                if "client" in locals():
                    with guard:
                        active_clients.discard(client)
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/media", state
    finally:
        stopped.set()
        with guard:
            clients = list(active_clients)
        for client in clients:
            client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def probe_stream(stream: dict, referer: str, check: Check) -> dict:
    """Read the media header and duration without downloading or decoding a movie."""
    with _range_proxy(stream["url"], 6*1024*1024, check, referer) as (local, state):
        try:
            body = json.loads(media.run([paths.binary("ffprobe"), "-v", "error",
                "-protocol_whitelist", "http,tcp", "-rw_timeout", "10000000", "-f", "mov",
                "-show_entries", "format=duration:stream=codec_type,width,height,duration", "-of", "json", local],
                timeout=20, check=check))
        except ValueError:
            if state["error"]:
                raise ValueError(state["error"]) from None
            raise
        track = next((s for s in body.get("streams", []) if s.get("codec_type") == "video"), {})
        duration = _duration(body.get("format", {}).get("duration")) or _duration(track.get("duration"))
        if not track or not duration:
            raise ValueError(state["error"] or "视频头信息缺少有效时长或画面")
        return {"duration": duration, "width": int(track.get("width", 0)),
                "height": int(track.get("height", 0)), "filesize": state["source_size"]}


def _clip(candidate: dict, target: Path, clip_start: float, duration: float,
          check: Check | None, preview: bool) -> dict:
    check = check or (lambda: None)
    if not math.isfinite(clip_start) or clip_start < 0 or not math.isfinite(duration) or not 0.5 <= duration <= (12 if preview else 120):
        raise ValueError("小样段为 0.5–12 秒，成片镜头为 0.5–120 秒；开始时间不能小于零")
    check()
    if not candidate.get("metadata_checked") or time.time() - _number(candidate.get("inspection_at")) > 600:
        candidate = inspect_video_candidate(candidate, check)
    stream = _choose_stream(candidate, preview)
    total_budget = PREVIEW_LIMIT if preview else CLIP_LIMIT
    headers = {"Referer": str(httpx.URL(candidate.get("source_url", "")))}
    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    source_error = ""
    with tempfile.TemporaryDirectory(prefix="jinggao-video-") as td:
        temp = Path(td)
        start = clip_start
        mode = "whole_file_then_trim"
        if stream["format"] == "hls":
            playlist = _hls(stream["url"], check, candidate.get("source_url", ""))
            if playlist["variants"]:
                raise ValueError("播放来源嵌套清单格式发生变化，请重新检索")
            selected = select_segments(playlist["segments"], clip_start, duration)
            names = []
            for i, segment in enumerate(selected):
                check()
                part = temp / f"{i:03}.ts"
                segment_url = segment["url"]
                if candidate.get("media_extractor") == "iqiyi-public":
                    from .portal_videos import resolve_iqiyi_segment
                    segment_url = resolve_iqiyi_segment(segment, stream, candidate.get("source_url", ""), check)
                _, info = _fetch(segment_url, limit=total_budget - downloaded,
                                 check=check, headers=headers, target=part)
                downloaded += info["bytes"]
                names.append(part)
            # All network input is fetched and size checked above. FFmpeg sees only local files.
            input_file = temp / "segments.ts"
            with input_file.open("wb") as f:
                for path in names:
                    with path.open("rb") as source:
                        while chunk := source.read(1024 * 1024):
                            f.write(chunk)
            start = clip_start - selected[0]["start"]
            mode = "hls_selected_segments"
        output = temp / "clip.mp4"
        filters = "scale='min(iw,640)':-2" if preview else "scale='min(iw,1920)':-2"

        def encode(input_url: str, network=False):
            # Force the expected demuxer for remote data: an HTML/playlist response
            # cannot make FFmpeg open arbitrary nested network resources.
            demuxer = "matroska" if stream.get("container") == "webm" or re.search(r"\.webm(?:$|\?)", stream["url"], re.I) else "ogg" if re.search(
                r"\.(ogg|ogv)(?:$|\?)", stream["url"], re.I) else "mov"
            # Input-seeking a stand-alone TS chunk can skip to a missing next
            # keyframe and silently produce a 0-second clip. Decode locally up
            # to the requested offset. Sparse WebM cue tables need a short preroll.
            preroll = min(start, 10) if network and demuxer in ("matroska", "ogg") else start if not network else 0
            input_seek = start - preroll
            media.run([paths.binary("ffmpeg"), "-v", "error", "-xerror", "-err_detect", "explode", "-y",
                       "-protocol_whitelist", "http,tcp" if network else "file",
                       *(["-f", demuxer, "-rw_timeout", "15000000"] if network else []),
                       *(["-ss", str(round(input_seek, 3))] if input_seek > 0 else []), "-i", input_url,
                       *(["-ss", str(round(preroll, 3))] if preroll > 0 else []), "-t", str(round(duration, 3)), "-map", "0:v:0", "-an", "-vf", filters,
                       "-c:v", "libx264", "-preset", "veryfast" if candidate.get("prefer_efficient_stream") else "fast",
                       "-threads", "2", "-crf", "29" if preview else "19",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], timeout=180, check=check)

        try:
            if stream["format"] == "hls":
                encode(str(input_file))
            else:
                with _range_proxy(stream["url"], total_budget, check, headers["Referer"]) as (local_url, state):
                    try:
                        encode(local_url, network=True)
                    except ValueError:
                        if state["error"]:
                            raise ValueError(state["error"]) from None
                        raise
                    downloaded = state["bytes"]
                    source_error = state["error"]
                    mode = "http_range_bounded" if state["range_responses"] else "bounded_progressive_read"
        except ValueError as exc:
            if any(word in str(exc) for word in ("预算", "HTTP", "超过", "超时", "限流")):
                raise
            raise ValueError("视频小样段无法正常解码，可能是受保护编码或损坏数据；已停止，未当作合格素材") from None
        info = media.probe(output)
        video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        actual_duration = _number(info.get("format", {}).get("duration"))
        if not video or actual_duration < duration - 0.3:
            if source_error:
                raise ValueError(source_error)
            raise ValueError(f"实际下载片段只有 {actual_duration:.2f} 秒（需要 {duration:.2f} 秒），未将尾帧冻结伪装成动态素材")
        if not preview and min(int(video.get("height", 0)), int(video.get("width", 0))) < int(_number(candidate.get("min_video_height"), 720)):
            raise ValueError("实际视频清晰度低于项目门槛，未用放大图片冒充高清视频")
        target.write_bytes(output.read_bytes())
    return {"path": str(target), "source_clip_start": clip_start, "clip_start": 0,
            "duration": actual_duration, "width": int(video["width"]), "height": int(video["height"]),
            "download_mode": mode, "downloaded_bytes": downloaded, "preview": preview,
            "source_url": candidate.get("source_url", ""), "source_duration": candidate.get("duration", 0),
            "source_evidence": candidate.get("source_evidence", {}), "publisher": candidate.get("publisher", ""),
            "publisher_url": candidate.get("publisher_url", ""), "source_verified": candidate.get("source_verified", False),
            "published_at": candidate.get("published_at", ""), "quality_verified": True}


def preview_video_candidate(candidate: dict, target: Path, clip_start: float = 0,
                            duration: float = 8, check: Check | None = None) -> dict:
    return _clip(candidate, target, clip_start, duration, check, True)


def download_video_candidate(candidate: dict, target: Path, clip_start: float = 0,
                             duration: float = 8, check: Check | None = None) -> dict:
    return _clip(candidate, target, clip_start, duration, check, False)
