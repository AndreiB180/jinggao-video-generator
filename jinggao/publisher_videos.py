"""Video discovery from publishers' ordinary public search/player pages."""
from __future__ import annotations

import json
import re
from urllib.parse import urlencode

from . import video_sources as video


def _chinanews_rows(markup: str) -> list[dict]:
    match = re.search(r"\bvar\s+docArr\s*=\s*", markup)
    if not match:
        raise ValueError("中新网视频检索页格式已变化")
    try:
        items, _ = json.JSONDecoder().raw_decode(markup[match.end():])
    except ValueError:
        raise ValueError("中新网视频检索数据不可读") from None
    if not isinstance(items, list):
        raise ValueError("中新网视频检索数据格式无效")
    return items


def parse_chinanews_search(markup: str) -> list[dict]:
    result = []
    for item in _chinanews_rows(markup)[:20]:
        source = video._http_url(item.get("url", "")).replace("http://", "https://", 1)
        if not any(video._host(source, d) for d in ("chinanews.com.cn", "chinanews.com")) or "/shipin/" not in source:
            continue
        title = item.get("title", "")
        title = " ".join(title) if isinstance(title, list) else title
        title, desc = video._clean(title), video._clean(item.get("content_without_tag", ""))
        date = video._clean(item.get("pubtime", ""))
        result.append(video._candidate("chinanews", title, source,
            video._http_url(item.get("thumbnail_file", "")), description=desc,
            publisher="中国新闻网", publisher_url="https://www.chinanews.com.cn/",
            publisher_verified=True, source_verified=True, published_at=date,
            license="中国新闻网公开视频；转载及商用许可需核实",
            source_evidence=video._evidence(source, title, desc, "中国新闻网",
                "https://www.chinanews.com.cn/", date, "publisher_search")))
    return result


def chinanews_search(query: str, page: int = 1) -> list[dict]:
    def fetch():
        data, _ = video._fetch("https://sou.chinanews.com.cn/search/video?" + urlencode({
            "q": query, "sortType": "_score", "pageNum": page}))
        return parse_chinanews_search(data.decode("utf-8", errors="replace"))
    return video._cached("chinanews-relevance", query if page == 1 else f"{query}\npage={page}", fetch)


def chinanews_images(query: str) -> list[dict]:
    from .sources import candidate
    def fetch():
        raw, _ = video._fetch("https://sou.chinanews.com.cn/search/pic?" + urlencode({"q": query, "sortType": "_score"}))
        result = []
        for item in _chinanews_rows(raw.decode("utf-8", errors="replace"))[:20]:
            source = video._http_url(item.get("url")).replace("http://", "https://", 1)
            image = video._http_url(item.get("hd_media_file") or item.get("md_media_file")).replace("http://", "https://", 1)
            if not source or not image:
                continue
            title = item.get("title", "")
            title = video._clean(" ".join(title) if isinstance(title, list) else title)
            result.append(candidate("chinanews_image", title, image,
                video._http_url(item.get("thumbnail_file")), source,
                publisher="中国新闻网", published_at=item.get("pubtime", ""),
                width=int(video._number(item.get("frame_width"))), height=int(video._number(item.get("frame_height"))),
                license="中国新闻网图片；转载及商用许可需核实"))
        return result
    return video._cached("chinanews-images", query, fetch)


def chinanews_articles(query: str, check) -> list[tuple[str, str]]:
    data, _ = video._fetch("https://sou.chinanews.com.cn/search/news?" + urlencode({"q": query, "sortType": "_score"}), check=check)
    rows = []
    for item in _chinanews_rows(data.decode("utf-8", errors="replace"))[:3]:
        title = item.get("title", "")
        title = video._clean(" ".join(title) if isinstance(title, list) else title)
        source = video._http_url(item.get("url")).replace("http://", "https://", 1)
        if source and any(video._host(source, d) for d in ("chinanews.com.cn", "chinanews.com")):
            rows.append((source, title))
    return rows


def chinanews(candidate: dict, check) -> dict:
    source = candidate["source_url"]
    data, _ = video._fetch(source, check=check)
    markup = data.decode("utf-8", errors="replace")
    parser = video._PageMetadata()
    parser.feed(markup)
    # Bind to the article's player configuration. Never harvest unrelated
    # recommendation or advertising videos merely because they occur in HTML.
    variants = re.findall(r'\{\s*name\s*:\s*["\']([^"\']+)["\']\s*,\s*url\s*:\s*["\']([^"\']+)["\']\s*\}', markup)
    selected = re.search(r'\bsource\s*:\s*["\']([^"\']+\.mp4(?:\?[^"\']*)?)["\']', markup)
    if not selected:
        selected = re.search(r'addVariable\(["\'](?:vInfo|vsrc|videoUrl)["\']\s*,\s*["\']([^"\']+)["\']', markup)
    if not variants and selected:
        variants = [("默认", selected[1])]
    streams, seen = [], set()
    for label, url in variants:
        url = video._http_url(url).replace("http://", "https://", 1)
        if (not any(video._host(url, host) for host in ("cns.com.cn", "cnsimg.net", "chinanews.com.cn", "chinanews.com"))
                or not re.search(r"\.mp4(?:$|\?)", url, re.I) or url in seen):
            continue
        seen.add(url)
        streams.append({"url": url, "format": "file", "width": 0, "height": 0,
                        "quality_hint": 1080 if "超清" in label else 720 if "高清" in label else 360})
    if not streams:
        raise ValueError("中新网原页没有公开的普通视频播放地址")
    # Probe only metadata; dimensions on a thumbnail never establish video quality.
    best = max(streams, key=video._short_edge)
    facts = video.probe_stream(best, source, check)
    best.update(width=facts["width"], height=facts["height"], filesize=facts["filesize"])
    title = video._clean(parser.meta.get("og:title") or candidate["title"])
    desc = video._clean(candidate.get("description") or parser.meta.get("description", ""))
    date = candidate.get("published_at", "")
    return candidate | {"media_streams": streams, "duration": facts["duration"],
        "title": title, "description": desc, "source_verified": True,
        "publisher": "中国新闻网", "publisher_url": "https://www.chinanews.com.cn/",
        "source_evidence": video._evidence(source, title, desc, "中国新闻网",
            "https://www.chinanews.com.cn/", date, "publisher_page")}
