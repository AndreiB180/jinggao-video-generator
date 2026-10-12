"""Official, paginated video metadata search; never an editorial approval.

Registry configuration and observed reachability are separate. Missing keys and
Coverr's unresolved service-integration terms never trigger speculative requests.
Only normalized public metadata is cached: no headers, API keys or signed media
URLs. This module neither downloads footage nor crawls public catalog pages.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from urllib.parse import parse_qsl, quote, urlencode, urlparse

import httpx

from . import config, execution, paths, store, video_sources

VERSION = "official-video-index-v2"
PAGE_SIZE = 20
CACHE_TTL = 86400  # Pixabay explicitly requires at least a 24-hour search cache.
PROVIDERS = {
    "pexels": {
        "name": "Pexels", "key_field": "pexels_api_key",
        "catalog_url": "https://www.pexels.com/videos/",
        "docs_url": "https://www.pexels.com/api/documentation/",
        "api_url": "https://api.pexels.com/v1/videos/search",
        "license_url": "https://www.pexels.com/license/",
        "license": "Pexels License；需核对人物、商标及使用条件，并保留来源",
        "public_index": "官网公开视频目录；搜索 API 需要独立 Key，未自动采集目录",
        "bulk_index_allowed": False,
        "index_policy": "按需索引：随用户文稿或明确搜索查询并累计；通用素材库补全不扫描，查询缓存 24 小时",
    },
    "pixabay": {
        "name": "Pixabay", "key_field": "pixabay_api_key",
        "catalog_url": "https://pixabay.com/videos/",
        "docs_url": "https://pixabay.com/api/docs/",
        "api_url": "https://pixabay.com/api/videos/",
        "license_url": "https://pixabay.com/service/license-summary/",
        "license": "Pixabay Content License；需逐条核对署名、人物及商标等使用条件",
        "public_index": "官网公开视频目录；API 需要独立 Key，不进行系统性批量下载",
        "bulk_index_allowed": False,
        "index_policy": "按需索引：仅随用户文稿或明确搜索查询并累计；通用素材库补全不扫描，查询缓存 24 小时",
    },
    "coverr": {
        "name": "Coverr", "key_field": "coverr_api_key",
        "catalog_url": "https://coverr.co/stock-video-footage/categories",
        "docs_url": "https://api.coverr.co/docs/",
        "api_url": "https://api.coverr.co/videos",
        "license_url": "https://coverr.co/license",
        "license": "Coverr License / API 服务条款；本服务集成授权尚未确认",
        "public_index": "官网类别与合集目录；所有 API 均需 Key，未自动采集目录",
    },
    "nasa": {
        "name": "NASA Image and Video Library", "key_field": "",
        "catalog_url": "https://images.nasa.gov/",
        "docs_url": "https://images.nasa.gov/docs/images.nasa.gov_api_docs.pdf",
        "api_url": "https://images-api.nasa.gov/search",
        "license_url": "https://www.nasa.gov/nasa-brand-center/images-and-media/",
        "license": "NASA Media Usage Guidelines；需核对第三方版权、署名及标志/肖像使用条件",
        "public_index": "官方无 Key 视频元数据 API；仅覆盖 NASA 收录题材",
    },
    "commons": {
        "name": "Wikimedia Commons", "key_field": "",
        "catalog_url": "https://commons.wikimedia.org/wiki/Category:Videos",
        "docs_url": "https://commons.wikimedia.org/wiki/Commons:API",
        "api_url": "https://commons.wikimedia.org/w/api.php",
        "license_url": "https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia",
        "license": "以每个 Commons 文件页面的许可及署名要求为准",
        "public_index": "官方无 Key MediaWiki 搜索 API；仅接受返回 MIME 为视频的文件",
    },
}
SOURCE_HOSTS = {
    "pexels": {"pexels.com", "www.pexels.com"},
    "pixabay": {"pixabay.com", "www.pixabay.com"},
    "coverr": {"coverr.co", "www.coverr.co", "api.coverr.co"},
    "nasa": {"images.nasa.gov"},
    "commons": {"commons.wikimedia.org"},
}
PREVIEW_HOSTS = {
    "pexels": {"images.pexels.com", "static-videos.pexels.com"},
    "pixabay": {"cdn.pixabay.com"},
    "coverr": {"storage.coverr.co"},
    "nasa": {"images-assets.nasa.gov"},
    "commons": {"upload.wikimedia.org", "thumb.wikimedia.org"},
}
_PRIVATE_PARAMS = {"key", "api_key", "token", "access_token", "authorization", "signature", "jwt"}
_detail_memory = {}
_detail_guard = threading.Lock()
_DETAIL_MEDIA_HOSTS = {"pexels": {"videos.pexels.com", "static-videos.pexels.com"},
                       "pixabay": {"cdn.pixabay.com"}}


class SourceUnavailable(ValueError):
    def __init__(self, provider, status, reason):
        self.provider, self.status, self.reason = provider, status, reason
        super().__init__(f"{PROVIDERS[provider]['name']}：{reason}")


def registry(settings=None):
    """No network calls; ``configured`` does not mean a successful live probe."""
    settings = config.read() if settings is None else settings
    result = {}
    for provider, definition in PROVIDERS.items():
        key_field = definition["key_field"]
        has_key = bool(str(settings.get(key_field) or "").strip()) if key_field else False
        configured = not key_field or has_key
        state = "configured" if configured else "needs_key"
        reason = ("接口条件已具备，尚未验证当前可达性" if configured
                  else "官方 API 需要另行配置素材站 Key；Qwen Key 不能用于此站")
        if provider == "coverr":
            state = "restricted"
            reason = "本服务的 Coverr 集成授权尚未确认；配置 Key 也不自动解除限制"
        bulk_allowed = definition.get("bulk_index_allowed", True)
        policy = definition.get("index_policy", "")
        if policy:
            reason += "；" + policy
        result[provider] = {"provider": provider, **definition,
            "website": definition["catalog_url"],
            "key_required": bool(key_field), "key_configured": has_key,
            "configuration_status": "configured" if configured else "needs_key",
            "status": state, "reason": reason, "checked": False,
            "live_checked": False, "observed_at": None, "metadata_only": True,
            "bulk_index_allowed": bulk_allowed,
            "index_mode": "catalogue" if bulk_allowed else "on_demand", "index_policy": policy}
    return result


def _url(value, hosts=None):
    if not isinstance(value, str):
        return ""
    try:
        parsed = urlparse(value)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.port not in (None, 443) or (hosts is not None and parsed.hostname.lower() not in hosts)
                or any(k.lower() in _PRIVATE_PARAMS for k, _ in parse_qsl(parsed.query))):
            return ""
        return parsed._replace(fragment="").geturl()
    except ValueError:
        return ""


def _text(value, limit=1200):
    return video_sources._clean(value, limit) if isinstance(value, str) else ""


def _number(value):
    return 0.0 if isinstance(value, bool) else video_sources._number(value)


def _nasa_collection(value):
    url = _url(value, {"images-assets.nasa.gov"})
    parsed = urlparse(url)
    return url if parsed.path.endswith("/collection.json") and not parsed.query else ""


def _record(provider, ident, title, source, preview="", *, description="", duration=0,
            author="", publisher="", publisher_url="", published_at="", license_name="",
            license_url="", width=0, height=0, **extra):
    source = _url(source, SOURCE_HOSTS[provider])
    if not source or not ident or not _text(title):
        return None
    preview = _url(preview, PREVIEW_HOSTS[provider])
    definition = PROVIDERS[provider]
    row = video_sources._candidate(provider, _text(title), source, preview,
        video_id=str(ident), metadata_preview_url=preview, description=_text(description),
        duration=_number(duration), width=int(_number(width)), height=int(_number(height)),
        author=_text(author), publisher=_text(publisher or author),
        publisher_url=_url(publisher_url), published_at=_text(published_at, 100),
        license=_text(license_name) or definition["license"],
        license_url=_url(license_url) or definition["license_url"],
        source_verified=True, metadata_only=True, **extra)
    row["source_evidence"] = video_sources._evidence(source, row["title"], row["description"],
        row["publisher"], row["publisher_url"], row["published_at"], "platform_metadata")
    return row


def _items(body, field):
    rows = body.get(field)
    if not isinstance(rows, list):
        raise ValueError("官方响应结构不完整")
    return rows[:PAGE_SIZE]


def parse(provider, body):
    """Strict envelope, per-item video filtering; no signed rendition URLs."""
    if provider not in PROVIDERS or not isinstance(body, dict) or body.get("error"):
        raise ValueError("官方响应结构无效")
    result = []
    if provider == "nasa":
        collection = body.get("collection")
        if not isinstance(collection, dict):
            raise ValueError("NASA 未返回元数据集合")
        for item in _items(collection, "items"):
            if not isinstance(item, dict) or not isinstance(item.get("data"), list):
                continue
            data = next((d for d in item["data"] if isinstance(d, dict) and d.get("media_type") == "video"), {})
            ident = data.get("nasa_id")
            if not isinstance(ident, str) or not ident or len(ident) > 180:
                continue
            links = item.get("links") if isinstance(item.get("links"), list) else []
            preview = next((p.get("href") for p in links if isinstance(p, dict) and p.get("rel") == "preview"), "")
            result.append(_record(provider, ident, data.get("title"),
                "https://images.nasa.gov/details/" + quote(ident, safe=""), preview,
                description=data.get("description"), publisher="NASA / " + _text(data.get("center")),
                publisher_url="https://www.nasa.gov/", published_at=data.get("date_created"),
                collection_url=_nasa_collection(item.get("href"))))
    elif provider == "commons":
        query = body.get("query")
        # MediaWiki legitimately omits query/pages on an empty generator result.
        if query is None and "batchcomplete" in body:
            return []
        if not isinstance(query, dict) or not isinstance(query.get("pages"), dict):
            raise ValueError("Commons 未返回文件元数据")
        for page in sorted(query["pages"].values(), key=lambda p: _number(p.get("index")) if isinstance(p, dict) else 0):
            if not isinstance(page, dict) or not isinstance(page.get("imageinfo"), list):
                continue
            for info in page["imageinfo"][:1]:
                if not isinstance(info, dict) or info.get("mime") not in ("video/webm", "video/ogg", "video/mp4"):
                    continue
                meta = info.get("extmetadata") if isinstance(info.get("extmetadata"), dict) else {}
                def value(name, meta=meta):
                    entry = meta.get(name)
                    return entry.get("value", "") if isinstance(entry, dict) else ""
                title = _text(page.get("title")).removeprefix("File:")
                result.append(_record(provider, page.get("pageid") or title, title,
                    info.get("descriptionurl"), info.get("thumburl"),
                    description=value("ImageDescription"), author=value("Artist"),
                    license_name=value("LicenseShortName"), license_url=value("LicenseUrl"),
                    duration=info.get("duration"), width=info.get("width"), height=info.get("height")))
    else:
        for item in _items(body, "videos" if provider == "pexels" else "hits"):
            if not isinstance(item, dict) or not item.get("id"):
                continue
            if provider == "pexels":
                source = _url(item.get("url"), SOURCE_HOSTS[provider])
                if not source or not urlparse(source).path.startswith("/video/"):
                    continue
                user = item.get("user") if isinstance(item.get("user"), dict) else {}
                slug = urlparse(source).path.rstrip("/").rsplit("/", 1)[-1]
                result.append(_record(provider, item["id"], item.get("title") or slug.replace("-", " "),
                    source, item.get("image"), duration=item.get("duration"), width=item.get("width"),
                    height=item.get("height"), author=user.get("name"), publisher_url=user.get("url")))
            elif provider == "pixabay":
                formats = item.get("videos")
                if item.get("type") not in ("film", "animation") or not isinstance(formats, dict):
                    continue
                renditions = [v for v in formats.values() if isinstance(v, dict) and _url(v.get("url"), {"cdn.pixabay.com"})]
                if not renditions:
                    continue
                preview = next((v.get("thumbnail") for v in renditions if v.get("thumbnail")), "")
                result.append(_record(provider, item["id"], item.get("tags"), item.get("pageURL"), preview,
                    duration=item.get("duration"), author=item.get("user"), description=item.get("tags"),
                    width=max(_number(v.get("width")) for v in renditions),
                    height=max(_number(v.get("height")) for v in renditions)))
            else:
                ident = str(item["id"])
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", ident):
                    continue
                source = _url(item.get("canonical_url") or item.get("url"), SOURCE_HOSTS[provider])
                result.append(_record(provider, ident, item.get("title"),
                    source or "https://api.coverr.co/videos/" + ident,
                    item.get("thumbnail") or item.get("poster"), description=item.get("description"),
                    duration=item.get("duration"), width=item.get("max_width"), height=item.get("max_height"),
                    published_at=item.get("published_at"), source_access="public_page" if source else "api_key"))
    unique = {}
    for row in result:
        if row is not None:
            unique.setdefault(row["source_url"], row)
    return list(unique.values())[:PAGE_SIZE]


def _request(provider, query, page, key, check):
    headers = None
    chinese = bool(re.search(r"[\u3400-\u9fff]", query))
    if provider == "pexels":
        params = {"query": query, "page": page, "per_page": PAGE_SIZE, "locale": "zh-CN" if chinese else "en-US"}
        headers = {"Authorization": key}
    elif provider == "pixabay":
        params = {"key": key, "q": query, "page": page, "per_page": PAGE_SIZE,
                  "safesearch": "true", "lang": "zh" if chinese else "en"}
    elif provider == "coverr":
        params = {"query": query, "page": page - 1, "page_size": PAGE_SIZE, "urls": "false"}
        headers = {"Authorization": "Bearer " + key}
    elif provider == "nasa":
        params = {"q": query, "media_type": "video", "page": page, "page_size": PAGE_SIZE}
    else:
        params = {"action": "query", "format": "json", "generator": "search",
                  "gsrsearch": "(" + query + ") filetype:video", "gsrnamespace": 6,
                  "gsrlimit": PAGE_SIZE, "gsroffset": (page - 1) * PAGE_SIZE,
                  "prop": "imageinfo", "iiprop": "url|mime|extmetadata|size", "iiurlwidth": 640}
    return video_sources._json(PROVIDERS[provider]["api_url"] + "?" + urlencode(params), check=check, headers=headers)


def _failure(exc):
    # Never return arbitrary transport text: Pixabay's request URL contains a key.
    code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
    text = str(exc)
    if (code in (401, 403, 412, 451) or re.search(r"\b(?:401|403|412|451)\b", text)
            or any(word in text for word in ("验证", "登录", "challenge"))):
        return "restricted", "来源限制访问或鉴权未通过；未绕过登录、验证或权限限制"
    if code == 429 or any(word in text for word in ("429", "限流", "冷却")):
        return "restricted", "来源限流；已停止本次请求，未自动重试"
    if any(word in text for word in ("国内素材", "已接入", "鉴权接口发生跨域")):
        return "restricted", "当前来源或重定向未获网络访问策略允许"
    return "unavailable", "官方元数据暂不可用、连接失败或响应格式不完整；未自动重试"


def search(provider, query, page=1, *, settings=None, check=None):
    """One demand-driven official page, or its 24h cache.

    Called for a user's current script, explicit search or explicit connection
    check. Topic prewarming is separately disabled by source_index.run for
    on-demand providers. Empty success is not a source failure.
    """
    if provider not in PROVIDERS:
        raise ValueError("未知外部素材来源")
    if not isinstance(query, str) or not query.strip() or len(query) > (100 if provider == "pixabay" else 200):
        raise ValueError("素材查询为空或超过官方接口长度限制")
    if type(page) is not int or not 1 <= page <= 1000:
        raise ValueError("素材页码必须为 1 到 1000")
    settings = config.read() if settings is None else settings
    state = registry(settings)[provider]
    result = {"provider": provider, "query": query, "page": page, "items": [],
        "status": state["status"], "configuration_status": state["configuration_status"],
        "reason": state["reason"], "cache_hit": False, "checked": False, "observed_at": None,
        "live_checked": False, "model_requests": 0,
        "index_mode": state["index_mode"], "bulk_index_allowed": state["bulk_index_allowed"]}
    if check:
        check()
    if state["status"] != "configured":
        return result
    key = str(settings.get(PROVIDERS[provider]["key_field"]) or "").strip()
    binding = [VERSION, provider, PROVIDERS[provider]["api_url"], query, page, PAGE_SIZE,
               hashlib.sha256(key.encode()).hexdigest()]
    digest = hashlib.sha256(json.dumps(binding, ensure_ascii=False).encode()).hexdigest()
    folder = paths.DATA / "cache" / "stock-sources"
    cache = folder / (digest + ".json")
    live_checked = False
    try:
        with execution.keyed_lock(("stock-search", str(cache)), check=check):
            try:
                saved = json.loads(cache.read_text("utf-8"))
                valid = (isinstance(saved, dict) and saved.get("binding") == digest
                         and type(saved.get("observed_at")) in (int, float)
                         and 0 <= time.time() - saved["observed_at"] < CACHE_TTL
                         and isinstance(saved.get("items"), list)
                         and all(isinstance(r, dict) and r.get("kind") == "video"
                                 and r.get("provider") == provider and r.get("status") == "unverified"
                                 and r.get("metadata_only") is True and r.get("download_status") == "unchecked"
                                 and not {"approved", "verification", "preview", "media_streams"}.intersection(r)
                                 and _url(r.get("source_url"), SOURCE_HOSTS[provider])
                                 for r in saved["items"]))
                if valid:
                    return result | {"items": saved["items"], "checked": True, "cache_hit": True,
                        "observed_at": saved["observed_at"], "reason": "复用上次官方查询元数据；未重新探测当前可达性"}
            except (OSError, ValueError, TypeError):
                pass
            live_checked = True
            body = _request(provider, query, page, key, check)
            if check:
                check()
            rows = parse(provider, body)
            now = time.time()
            saved = {"binding": digest, "observed_at": now, "items": rows}
            folder.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_name(cache.name + "." + store.uid() + ".tmp")
            try:
                temporary.write_text(json.dumps(saved, ensure_ascii=False), "utf-8")
                temporary.replace(cache)
            finally:
                temporary.unlink(missing_ok=True)
            return result | {"items": rows, "checked": True, "live_checked": True, "observed_at": now,
                "reason": f"官方查询返回 {len(rows)} 条视频元数据；未下载或核验画面"}
    except (ValueError, httpx.HTTPError, OSError) as exc:
        status, reason = _failure(exc)
        return result | {"status": status, "reason": reason, "checked": live_checked,
                         "live_checked": live_checked, "observed_at": time.time() if live_checked else None}


def search_items(provider, query, page=1, *, settings=None, check=None):
    """Adapter for source_index.routes; unavailable is never a fabricated empty page."""
    result = search(provider, query, page, settings=settings, check=check)
    if result["status"] != "configured":
        raise SourceUnavailable(provider, result["status"], result["reason"])
    return result["items"]


def _detail_stream(value, provider):
    if not isinstance(value, dict):
        return None
    url = value.get("url")
    if not isinstance(url, str):
        return None
    try:
        parsed = urlparse(url)
        if (parsed.scheme != "https" or parsed.hostname not in _DETAIL_MEDIA_HOSTS[provider]
                or parsed.username or parsed.password or parsed.port not in (None, 443)
                or any(k.lower() in {"key", "api_key", "authorization"} for k, _ in parse_qsl(parsed.query))):
            return None
    except ValueError:
        return None
    return {"url": parsed._replace(fragment="").geturl(), "format": "file", "width": int(_number(value.get("width"))),
            "height": int(_number(value.get("height")))}


def details(provider, ident, source, *, settings=None, check=None):
    """Cache one official ID lookup for 24h, independently of inspection TTL.

    Only normalized public metadata and query-free rendition URLs reach disk.
    Query-bearing media URLs are volatile: usable in this process, never saved
    as permanent links. A restart does not authorize another same-day API call
    just to recover a volatile URL. No caller option bypasses this cache.
    """
    if provider not in _DETAIL_MEDIA_HOSTS or not re.fullmatch(r"\d{1,20}", str(ident)):
        raise ValueError("素材缺少可核对的视频编号")
    if not _url(source, SOURCE_HOSTS[provider]):
        raise ValueError("素材来源与提供方不一致")
    ident = str(ident)
    settings = config.read() if settings is None else settings
    access = registry(settings)[provider]
    if access["status"] != "configured":
        raise ValueError(access["reason"])
    key = str(settings.get(PROVIDERS[provider]["key_field"]) or "").strip()
    binding = ["official-video-detail-v1", provider, ident, source.rstrip("/"),
               hashlib.sha256(key.encode()).hexdigest()]
    digest = hashlib.sha256(json.dumps(binding, ensure_ascii=False).encode()).hexdigest()
    folder = paths.DATA / "cache" / "stock-details"
    target = folder / (digest + ".json")
    memory_key = str(target)

    def valid(saved):
        if (not isinstance(saved, dict) or saved.get("binding") != digest
                or type(saved.get("observed_at")) not in (int, float)
                or not 0 <= time.time() - saved["observed_at"] < CACHE_TTL):
            return False
        row = saved.get("metadata")
        return (isinstance(row, dict) and row.get("provider") == provider
                and str(row.get("video_id")) == ident
                and _url(row.get("source_url"), SOURCE_HOSTS[provider]).rstrip("/") == source.rstrip("/")
                and row.get("kind") == "video" and row.get("status") == "unverified"
                and row.get("metadata_only") is True and row.get("download_status") == "unchecked"
                and not {"approved", "verification", "media_streams", "headers", "api_key"}.intersection(row)
                and isinstance(saved.get("streams"), list)
                and all(_detail_stream(stream, provider) == stream for stream in saved["streams"]))

    def result(saved, cache_hit):
        # Callers may mutate candidates without altering this request cache.
        saved = json.loads(json.dumps(saved))
        row = saved["metadata"] | {"media_streams": saved["streams"],
            "stock_detail_expires_at": saved["observed_at"] + CACHE_TTL,
            "stock_detail_cache_hit": cache_hit}
        if not saved["streams"] and saved.get("volatile_streams_omitted"):
            row["download_reason"] = "已复用 24 小时元数据缓存；临时视频地址未持久化，当前无可用流，未重复请求"
        return row

    with execution.keyed_lock(("stock-detail", memory_key), check=check):
        if check:
            check()
        with _detail_guard:
            saved = _detail_memory.get(memory_key)
        if valid(saved):
            return result(saved, True)
        with _detail_guard:
            _detail_memory.pop(memory_key, None)
        try:
            saved = json.loads(target.read_text("utf-8"))
            # Disk entries must never contain a signed/query-bearing stream.
            if valid(saved) and all(not urlparse(r["url"]).query for r in saved["streams"]):
                return result(saved, True)
        except (OSError, ValueError, TypeError):
            pass
        try:
            if provider == "pexels":
                item = video_sources._json("https://api.pexels.com/v1/videos/videos/" + ident,
                    check=check, headers={"Authorization": key})
                raw_streams = [{"url": f.get("link"), "width": f.get("width"), "height": f.get("height")}
                               for f in item.get("video_files", []) if isinstance(f, dict) and f.get("file_type") == "video/mp4"]
                rows = parse(provider, {"videos": [item]})
            else:
                body = video_sources._json("https://pixabay.com/api/videos/?" + urlencode({"key": key, "id": ident}), check=check)
                item = next((r for r in body.get("hits", []) if isinstance(r, dict) and str(r.get("id")) == ident), {})
                raw_streams = [f for f in item.get("videos", {}).values() if isinstance(f, dict)]
                rows = parse(provider, {"hits": [item]})
        except (ValueError, httpx.HTTPError, OSError, TypeError, AttributeError) as exc:
            raise ValueError(_failure(exc)[1]) from None
        if not rows or str(item.get("id")) != ident or rows[0]["source_url"].rstrip("/") != source.rstrip("/"):
            raise ValueError("官方视频元数据与原始条目不一致")
        streams = [stream for raw in raw_streams if (stream := _detail_stream(raw, provider))]
        saved = {"binding": digest, "observed_at": time.time(), "metadata": rows[0], "streams": streams}
        # Do not allow even an unexpected API echo to persist the credential.
        if key in json.dumps(saved, ensure_ascii=False):
            raise ValueError("官方视频响应包含敏感鉴权信息，未保存或使用")
        if check:
            check()
        durable = saved | {"streams": [r for r in streams if not urlparse(r["url"]).query],
                           "volatile_streams_omitted": any(urlparse(r["url"]).query for r in streams)}
        temporary = target.with_name(target.name + "." + store.uid() + ".tmp")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(durable, ensure_ascii=False), "utf-8")
            temporary.replace(target)
        except OSError:
            raise ValueError("官方视频资料缓存无法保存；本次未自动重试") from None
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        with _detail_guard:
            for old_key, old in list(_detail_memory.items()):
                if time.time() - old["observed_at"] >= CACHE_TTL:
                    _detail_memory.pop(old_key, None)
            if len(_detail_memory) >= 256:
                _detail_memory.pop(next(iter(_detail_memory)))
            _detail_memory[memory_key] = saved
        return result(saved, False)
