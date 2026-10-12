from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
import socket
import time
import threading
from .execution import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from urllib.parse import urlparse

import httpx

from . import config, execution, network, paths, store

AGENT = "Jinggao/0.1 (local video editor; Wikimedia Commons search)"
_download_lock = threading.Lock()
_last_commons_download = 0.0


def cooldown_path(url: str) -> Path:
    host = urlparse(url).hostname or ""
    cache = paths.DATA / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / ("cooldown-"+hashlib.sha256(host.encode()).hexdigest()+".json")


def cooldown_remaining(url: str) -> float:
    cooldown = cooldown_path(url)
    if cooldown.exists():
        return max(0, json.loads(cooldown.read_text())["until"]-time.time())
    return 0


def download_slot(url: str) -> Path:
    global _last_commons_download
    host = urlparse(url).hostname or ""
    remaining = cooldown_remaining(url)
    if remaining > 0:
        raise ValueError(f"素材来源限流（HTTP 429），请约 {max(1,round(remaining/60))} 分钟后重试")
    if host == "upload.wikimedia.org":
        with _download_lock:
            time.sleep(max(0, 1.2-(time.monotonic()-_last_commons_download)))
            _last_commons_download = time.monotonic()
    return cooldown_path(url)


def public_url(url: str) -> None:
    from .domestic import require
    require(url)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("素材链接必须为公开 HTTP(S) 链接")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except OSError:
        raise ValueError("无法解析素材域名") from None
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("素材下载不接受本机、内网或保留地址")


def download(url: str, target: Path, *, limit: int = 300 * 1024 * 1024, check=None) -> None:
    with execution.network_slot(check):
        _download(url, target, limit=limit, check=check)


def _download(url: str, target: Path, *, limit: int = 300 * 1024 * 1024, check=None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # Multiple shots can need the same thumbnail concurrently, including jobs
    # in separate processes. Never share a mutable .part file between them.
    tmp = target.with_name(target.name + "." + store.uid() + ".part")
    try:
        began = time.monotonic()
        with network.client(timeout=httpx.Timeout(12, connect=6), headers={"User-Agent": AGENT}, follow_redirects=False) as client:
            for _ in range(6):
                if check:
                    check()
                cooldown = download_slot(url)
                public_url(url)
                with client.stream("GET", url) as response:
                    if response.is_redirect:
                        url = str(response.url.join(response.headers["location"]))
                        continue
                    if response.status_code == 429:
                        retry = response.headers.get("Retry-After", "60")
                        delay = max(1, int(retry)) if retry.isdigit() else 60
                        cooldown.write_text(json.dumps({"until": time.time()+delay}))
                        raise ValueError(f"素材来源限流（HTTP 429），请约 {max(1,round(delay/60))} 分钟后重试")
                    response.raise_for_status()
                    size = 0
                    with tmp.open("wb") as f:
                        for chunk in response.iter_bytes(256 * 1024):
                            if check:
                                check()
                            if time.monotonic()-began > 30:
                                raise ValueError("图片来源下载超过 30 秒，已跳过该候选")
                            size += len(chunk)
                            if size > limit:
                                raise ValueError("素材超过下载大小限制（300 MB；大文件请本地导入）")
                            f.write(chunk)
                    tmp.replace(target)
                    return
            raise ValueError("素材链接重定向过多")
    except httpx.HTTPError:
        raise ValueError("素材下载失败，来源可能限制直链；可从原始页面获取后本地导入") from None
    finally:
        tmp.unlink(missing_ok=True)


def clean(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value or "")).strip()[:600]


def candidate(provider: str, title: str, url: str, thumbnail: str, source: str, **extra) -> dict:
    return {"id": hashlib.sha256((provider+url).encode()).hexdigest()[:24], "provider": provider,
            "title": clean(title), "url": url, "thumbnail_url": thumbnail, "source_url": source,
            "license": "待确认", "author": "", "kind": "image", "status": "unverified", **extra}


def tokens(s: str) -> set[str]:
    s = s.lower()
    return set(re.findall(r"[a-z0-9]{2,}", s)) | {s[i:i+2] for i in range(len(s)-1)
            if all("\u4e00" <= ch <= "\u9fff" for ch in s[i:i+2])}


def local_candidates(project: dict, query: str) -> list[dict]:
    q = tokens(query)
    scored = []
    for a in project["assets"]:
        if a["kind"] not in ("image", "video"):
            continue
        overlap = len(q & tokens(a["title"] + " " + a.get("description", "")))
        c = {"id": "local-"+a["id"], "provider": "local", "asset_id": a["id"], "title": a["title"],
             "kind": a["kind"], "thumbnail_path": a.get("thumbnail", a["path"]),
             "source_url": a.get("source_url", ""), "license": a.get("license", "待确认"),
             "description": a.get("description", ""), "status": "unverified",
             "duration": a.get("duration", 0), "width": a.get("width", 0), "height": a.get("height", 0),
             "source_evidence": a.get("source_evidence", {}), "publisher": a.get("publisher", ""),
             "publisher_url": a.get("publisher_url", ""), "published_at": a.get("published_at", ""),
             "source_verified": a.get("source_verified", False), "source_clip_start": a.get("source_clip_start", 0)}
        scored.append((overlap, c))
    return [c for _, c in sorted(scored, key=lambda x: x[0], reverse=True)[:18]]


def commons(query: str) -> list[dict]:
    with execution.keyed_lock(("commons-search", str(paths.DATA), query)):
        return _commons(query)


def _commons(query: str) -> list[dict]:
    cache = paths.DATA / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    file = cache / ("commons-1280-"+hashlib.sha256(query.encode()).hexdigest()+".json")
    if file.exists() and time.time()-file.stat().st_mtime < 86400:
        body = json.loads(file.read_text("utf-8"))
    else:
        for attempt in range(2):
            try:
                response = httpx.get("https://commons.wikimedia.org/w/api.php", params={
                    "action": "query", "format": "json", "generator": "search", "gsrsearch": query,
                    "gsrnamespace": 6, "gsrlimit": 14, "prop": "imageinfo", "iiprop": "url|mime|extmetadata|size",
                    "iiurlwidth": 1280}, headers={"User-Agent": AGENT}, timeout=10)
                response.raise_for_status()
                body = response.json()
                break
            except httpx.HTTPStatusError as exc:
                if attempt == 1 or exc.response.status_code not in (429, 502, 503, 504):
                    raise
                retry = exc.response.headers.get("Retry-After", "")
                if retry.isdigit() and int(retry) > 15:
                    raise ValueError("Commons 暂时限制请求，请稍后重试或更换渠道") from None
                time.sleep(max(1+attempt, int(retry) if retry.isdigit() else 0))
            except httpx.RequestError:
                if attempt == 1:
                    raise
                time.sleep(1+attempt)
        if not body.get("error"):
            temporary = file.with_name(file.name+"."+store.uid()+".tmp")
            temporary.write_text(json.dumps(body), "utf-8")
            temporary.replace(file)
    if body.get("error"):
        raise ValueError("Commons 搜索服务返回错误")
    result = []
    for page in sorted(body.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0)):
        for item in page.get("imageinfo", [])[:1]:
            mime = item.get("mime", "")
            if mime not in ("image/jpeg", "image/png", "image/webp", "video/webm", "video/ogg", "video/mp4"):
                continue
            meta = item.get("extmetadata", {})
            def get(key, meta=meta):
                return clean(meta.get(key, {}).get("value", ""))
            title, source = page["title"].removeprefix("File:"), item.get("descriptionurl", "")
            image_url = item.get("thumburl", item["url"]) if mime.startswith("image/") else item["url"]
            result.append(candidate("commons", title, image_url,
                          item.get("thumburl", item["url"]), item.get("descriptionurl", ""),
                          kind="video" if mime.startswith("video") else "image",
                          license=get("LicenseShortName"), license_url=get("LicenseUrl"), author=get("Artist"),
                          description=get("ImageDescription"), width=item.get("width"), height=item.get("height"),
                          duration=item.get("duration", 0), filesize=item.get("size", 0), source_verified=True,
                          source_evidence={"url": source, "title": title, "description": get("ImageDescription"),
                              "readable": True, "evidence_origin": "platform_metadata", "image_url": image_url,
                              "original_url": item["url"]} ))
    return result


def tavily(query: str, key: str) -> list[dict]:
    if not key:
        raise ValueError("请在设置中填写 Tavily API Key")
    r = httpx.post("https://api.tavily.com/search", headers={"Authorization": "Bearer "+key},
                   json={"query": query, "search_depth": "basic", "include_images": True,
                         "include_image_descriptions": True, "max_results": 8}, timeout=35)
    r.raise_for_status()
    result, body = [], r.json()
    images = []
    for item in body.get("results", []):
        for im in item.get("images", []):
            if isinstance(im, str):
                im = {"url": im}
            images.append(im | {"source_url": item.get("url", "")})
    images += body.get("images", [])
    seen = set()
    for im in images:
        if isinstance(im, str):
            im = {"url": im, "description": query}
        url = im.get("url", "")
        if url and url not in seen:
            seen.add(url)
            result.append(candidate("tavily", im.get("description") or query, url, url,
                          im.get("source_url", ""), description=im.get("description", "")))
        if len(result) >= 14:
            break
    return result


def parse_bing_images(markup: str) -> list[dict]:
    result, seen = [], set()
    for value in re.findall(r'<a\b[^>]*\bm="([^"]+)"', markup):
        try:
            item = json.loads(html.unescape(value))
        except ValueError:
            continue
        if not isinstance(item, dict):
            continue
        url, source, thumb = (item.get(k, "") for k in ("murl", "purl", "turl"))
        if any(not isinstance(v, str) or urlparse(v).scheme not in ("http", "https")
               for v in (url, source, thumb)) or url in seen:
            continue
        seen.add(url)
        title = item.get("t")
        result.append(candidate("bing", title if isinstance(title, str) and title else "网页图片", url, thumb, source,
                      description="网页搜图候选；原始出处、图注和使用许可待核实",
                      license="网页图片，使用许可待核实"))
        if len(result) == 14:
            break
    return result


def bing_images(query: str) -> list[dict]:
    """Read public image search results; stop on access checks or page changes."""
    cache = paths.DATA / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    file = cache / ("bing-"+hashlib.sha256(query.encode()).hexdigest()+".json")
    if file.exists() and time.time()-file.stat().st_mtime < 3600:
        return json.loads(file.read_text("utf-8"))
    response = httpx.get("https://www.bing.com/images/search", params={"q": query, "form": "HDRSC2"},
                        headers={"User-Agent": AGENT}, timeout=10, follow_redirects=True)
    response.raise_for_status()
    result = parse_bing_images(response.text)
    if not result:
        raise ValueError("网页搜图没有可读结果，可能需要验证或页面已变化；请稍后重试或切换渠道")
    file.write_text(json.dumps(result, ensure_ascii=False), "utf-8")
    return result


def pixabay(query: str, key: str, *, video_only: bool = False) -> list[dict]:
    if not key:
        raise ValueError("请在设置中填写 Pixabay API Key")
    # Cache API responses for at least 24 hours, per the provider's usage requirements.
    cache = paths.DATA / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cooldown = cache / ("pixabay-limit-" + hashlib.sha256(key.encode()).hexdigest() + ".json")
    result = []
    for kind in (("video",) if video_only else ("image", "video")):
        name = hashlib.sha256((query+kind+key).encode()).hexdigest()
        p = cache / f"pixabay-{name}.json"
        if p.exists() and time.time()-p.stat().st_mtime < 86400:
            body = json.loads(p.read_text("utf-8"))
        else:
            if cooldown.exists() and json.loads(cooldown.read_text("utf-8")).get("until", 0) > time.time():
                raise ValueError("Pixabay 暂时限流，已遵守服务端等待时间")
            r = httpx.get("https://pixabay.com/api/" + ("videos/" if kind == "video" else ""),
                          params={"key": key, "q": query[:100], "per_page": 12, "safesearch": "true"}, timeout=25)
            if r.status_code == 429:
                delays = [int(value) for value in (r.headers.get("Retry-After", ""),
                                                   r.headers.get("X-RateLimit-Reset", "")) if value.isdigit()]
                cooldown.write_text(json.dumps({"until": time.time() + min(86400, max([60, *delays]))}), "utf-8")
                raise ValueError("Pixabay 暂时限流，已遵守服务端等待时间")
            r.raise_for_status()
            body = r.json()
            p.write_text(json.dumps(body), "utf-8")
        for hit in body.get("hits", []):
            if kind == "video":
                formats = hit.get("videos", {})
                video = formats.get("medium", {})
                streams = [{"url": f["url"], "format": "file", "width": f.get("width", 0),
                            "height": f.get("height", 0), "filesize": f.get("size", 0)}
                           for f in formats.values() if isinstance(f, dict) and
                           urlparse(str(f.get("url", ""))).scheme in ("https", "http")]
                if not streams:
                    continue
                url = video.get("url") or streams[0]["url"]
                thumbnail = video.get("thumbnail") or next((f.get("thumbnail", "") for f in formats.values()
                                                              if isinstance(f, dict) and f.get("thumbnail")), "")
                from .video_sources import _evidence
                source, title, author = hit.get("pageURL", ""), hit.get("tags", query), hit.get("user", "")
                result.append(candidate("pixabay", title, url, thumbnail, source, kind="video", author=author,
                    width=video.get("width", 0), height=video.get("height", 0), duration=hit.get("duration", 0),
                    filesize=video.get("size", 0), media_streams=streams, publisher=author, publisher_url=source,
                    source_verified=True, download_status="unchecked",
                    preview_urls=[{"url": thumbnail, "at": None, "kind": "publisher_thumbnail"}] if thumbnail else [],
                    source_evidence=_evidence(source, title, "Pixabay 视频素材：" + title,
                                              author, source, "", "platform_metadata"),
                    license="Pixabay Content License，需逐项检查使用条件",
                    license_url="https://pixabay.com/service/license-summary/"))
            else:
                url = hit.get("largeImageURL", hit.get("webformatURL", ""))
                thumbnail = hit.get("webformatURL", url)
                if url:
                    result.append(candidate("pixabay", hit.get("tags", query), url, thumbnail,
                                            hit.get("pageURL", ""), kind=kind, author=hit.get("user", ""),
                                            license="Pixabay Content License，需逐项检查使用条件"))
    return result


def search_images(query: str) -> tuple[list[dict], list[str]]:
    """Image fallback uses domestic publisher search, including legacy settings."""
    if config.read()["search_provider"] == "local":
        return [], []
    from .publisher_videos import chinanews_images
    try:
        return chinanews_images(query), []
    except (ValueError, httpx.HTTPError):
        return [], ["国内图片检索未完成"]


def search(project: dict, query: str) -> tuple[list[dict], str]:
    query = query.strip()
    if not query:
        raise ValueError("请输入素材搜索词")
    local = local_candidates(project, query)
    if config.read()["search_provider"] == "local":
        return local, "已搜索本项目素材"
    from .video_sources import search_videos
    videos, failures = search_videos(query)
    images, errors = search_images(query) if config.read()["search_provider"] != "video" else ([], [])
    result = list({c["id"]: c for c in [*local, *videos, *images]}.values())
    return result, f"找到 {len(result)} 个国内及本地候选；实际画面仍待核验" + (
        "；" + "、".join(failures + errors) if failures or errors else "")


def _legacy_search_shot(project: dict, shot: dict) -> tuple[list[dict], str]:
    queries = list(dict.fromkeys(q.strip() for q in [shot["query"], *shot.get("search_queries", [])] if q.strip()))[:3]
    if len(queries) <= 1 or config.read()["search_provider"] == "local":
        return search(project, shot["query"])
    batches, failures = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(search, project, query) for query in queries]
        for query, future in zip(queries, futures, strict=True):
            try:
                rows, message = future.result()
                batches.append([c | {"retrieved_query": query} for c in rows])
                if "未完成" in message:
                    failures.append(message)
            except ValueError as exc:
                failures.append(str(exc))
    by_id = {}
    for group in zip_longest(*batches):
        for candidate in group:
            if candidate and candidate["id"] not in by_id:
                by_id[candidate["id"]] = candidate
    if not by_id and failures:
        raise ValueError(failures[0])
    note = f"已检索 {len(queries)} 组关键词，去重后 {len(by_id)} 个候选，展示 {min(48,len(by_id))} 个；尚未核对实际画面"
    if failures:
        note += f"；{len(failures)} 组检索未完成"
    return list(by_id.values())[:48], note


def search_shot(project: dict, shot: dict) -> tuple[list[dict], str]:
    """Video-first projects never silently turn a missing video into a photo."""
    # Existing saved projects have no policy. Their old search behavior remains
    # available; new projects explicitly carry video_preferred from the store.
    policy = project.get("media_policy", "mixed")
    if "media_policy" not in project:
        return _legacy_search_shot(project, shot)
    cfg = config.read()
    queries = list(dict.fromkeys(q.strip() for q in [shot.get("query", ""), *shot.get("search_queries", [])]
                                 if isinstance(q, str) and q.strip()))[:4]
    if not queries:
        raise ValueError("请先填写素材搜索词")
    from . import identity
    people = shot.get("persons") or shot.get("required_people") or shot.get("people") or []
    people = identity.persons([{"name": person, "aliases": []} if isinstance(person, str) else person for person in people])
    names = list(dict.fromkeys(name for person in people for name in [person["name"], *person["aliases"]]
                               if identity.mentions(name, person)))
    # A precise person query supplements the scene query without widening it to
    # generic stock footage. The model also supplies separate Chinese/English queries.
    for name in names[:2]:
        if name not in queries and len(queries) < 4:
            queries.append(name)
    local = local_candidates(project, queries[0])
    if cfg["search_provider"] == "local":
        rows = [r for r in local if policy != "video_only" or r.get("kind") == "video"]
        rows.sort(key=lambda r: r.get("kind") != "video")
        count = sum(r.get("kind") == "video" for r in rows)
        return rows, f"本项目有 {count} 个视频候选；" + ("没有可用视频，请导入视频或开启在线检索" if not count else "实际片段仍待复核")
    from .video_sources import search_videos
    batches, failures = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(search_videos, query, include_stock=not bool(names)) for query in queries]
        for query, future in zip(queries, futures, strict=True):
            try:
                rows, errors = future.result()
                if names:
                    # Most search engines OR multiple names. Do not turn an
                    # unrelated recommendation into evidence about the person.
                    rows = [r for r in rows if any(identity.mentions(r.get("title", "") + " " +
                                                                   r.get("description", ""), person) for person in people)]
                batches.append([r | {"retrieved_query": query,
                                     "min_video_height": project.get("min_video_height", 720)} for r in rows])
                failures.extend(errors)
            except (ValueError, httpx.HTTPError) as exc:
                failures.append(str(exc) if isinstance(exc, ValueError) else "视频检索未完成")
    # Keep multiple publishers visible even when the first provider has many hits.
    grouped = {}
    for batch in batches:
        for row in batch:
            grouped.setdefault(row.get("provider", "unknown"), []).append(row)
    online = [r for group in zip_longest(*grouped.values()) for r in group if r]
    images = []
    if policy != "video_only" and cfg["search_provider"] not in ("video", "pexels"):
        try:
            fallback, _ = _legacy_search_shot(project, shot)
            # Video-first retains a small, explicit photo fallback. It is never
            # automatically selected merely because the video channels failed.
            images = [r for r in fallback if r.get("kind") != "video"][:10 if policy == "video_preferred" else 18]
            online.extend(r for r in fallback if r.get("kind") == "video")
        except (ValueError, httpx.HTTPError):
            failures.append("图片补充检索未完成")
    by_id = {}
    for row in sorted(local, key=lambda r: r.get("kind") != "video") + online:
        if row.get("kind") == "video" and row["id"] not in by_id:
            by_id[row["id"]] = row
    videos = list(by_id.values())[:48]
    if policy != "video_only":
        for row in images:
            if row["id"] not in by_id:
                videos.append(row)
                by_id[row["id"]] = row
    count = sum(r.get("kind") == "video" for r in videos)
    note = f"已检索 {len(queries)} 组关键词，找到 {count} 个视频和 {len(videos)-count} 个图片补充候选；先预检来源和样段"
    if not count:
        note += "；视频缺口尚未解决，图片不能替代视频配额"
    if failures:
        note += "；未完成渠道：" + "、".join(dict.fromkeys(failures))
    return videos, note
