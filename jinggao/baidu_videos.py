"""Baidu's ordinary HTML video search, with original domestic video links.

Search cards are discovery evidence only. They never certify the publisher,
identity, stream quality, or permission to reuse a clip. The existing player
inspectors must open the original source before a result can supply footage.
No account, challenge handling, alternate endpoint, or overseas relay is used.
"""
from __future__ import annotations

import json
import re
import time
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from . import domestic, execution, network, paths, sources, store, video_sources as video

SEARCH_URL = "https://www.baidu.com/sf/vsearch"
COOLDOWN_SECONDS = 1800
REQUEST_INTERVAL = 5.0
REQUEST_BUDGET = 20.0
MAX_BYTES = 4 * 1024 * 1024
MAX_PAGE = 10
_last_request: dict[str, float] = {}


def _original(value: str) -> tuple[str, str, str]:
    """Accept individual pages supported by existing domestic player readers."""
    url = video._http_url(value.strip())
    if not url or len(url) > 2048 or not domestic.allowed(url):
        return "", "", ""
    p = urlsplit(url)
    if p.port or "\\" in url:
        return "", "", ""
    host, path = (p.hostname or "").lower(), p.path
    query = parse_qs(p.query)
    if host in ("www.bilibili.com", "bilibili.com"):
        match = re.fullmatch(r"/video/(BV[0-9A-Za-z]{10})/?", path)
        if match and query.get("p", ["1"]) == ["1"]:
            return "https://www.bilibili.com/video/" + match[1], "bilibili", match[1]
    elif host == "v.qq.com":
        match = re.fullmatch(r"/(?:x/(?:page|cover/\w+)|boke/page/\w/\w/\w)/([A-Za-z0-9]{11})\.html", path)
        if match:
            return "https://v.qq.com/x/page/" + match[1] + ".html", "tencent", match[1]
    elif host == "v.youku.com":
        match = re.fullmatch(r"/v_show/id_([A-Za-z0-9]+={0,2})\.html", path)
        if match:
            return "https://v.youku.com" + path, "youku", match[1].rstrip("=")
    elif host in ("www.iqiyi.com", "iqiyi.com"):
        if re.fullmatch(r"/v_[A-Za-z0-9]+\.html", path):
            return "https://www.iqiyi.com" + path, "iqiyi", ""
    elif host == "tv.sohu.com":
        from .portal_videos import sohu_video_id
        canonical = "https://tv.sohu.com" + path
        if ident := sohu_video_id(canonical):
            return canonical, "sohu", ident
    elif host.endswith(".cctv.com"):
        if re.fullmatch(r"/\d{4}/\d{2}/\d{2}/VIDE[A-Za-z0-9]+\.shtml", path):
            return urlunsplit(("https", host, path, "", "")), "cctv", ""
    elif host in ("www.chinanews.com.cn", "www.chinanews.com"):
        # Public article pages can retain a single editorial channel prefix,
        # for example /life/shipin/cns-d/2026/01-01/article.shtml.
        if re.fullmatch(r"/(?:[a-z][a-z0-9_-]{0,31}/)?shipin/\S+\.(?:shtml|html)", path):
            return urlunsplit(("https", host, path, "", "")), "chinanews", ""
    return "", "", ""


class _Cards(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.pending = ""
        self.card = None
        self.depth = 0
        self.stack = []
        self.rows = []
        self.saw_results = False

    def handle_comment(self, text):
        if self.card is None:
            self.pending = text.strip() if text.strip().startswith(("http://", "https://")) else ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if "video_list_container" in classes:
            self.saw_results = True
        if tag == "div" and "video_list" in classes:
            self.saw_results = True
            self.card = {"original": self.pending, "title": "", "duration": "", "description": "",
                         "source_label": "", "text": "", "thumbnail": ""}
            self.pending, self.depth, self.stack = "", 0, []
        if self.card is None:
            self.pending = ""
            return
        if tag == "div":
            self.depth += 1
        if tag == "img" and not self.card["thumbnail"]:
            self.card["thumbnail"] = attrs.get("src") or attrs.get("data-src") or ""
        field = self.stack[-1][1] if self.stack else ""
        if tag == "a" and "video-title" in classes:
            field = "title"
        elif "video_play_timer" in classes:
            field = "duration"
        elif "wetSource" in classes:
            field = "source_label"
        elif tag == "div" and "c-color-text" in classes:
            field = "description"
        if tag not in ("img", "br", "hr", "meta", "link", "input", "source", "wbr"):
            self.stack.append((tag, field))

    def handle_data(self, text):
        if self.card is None:
            return
        self.card["text"] += text
        if self.stack and self.stack[-1][1]:
            self.card[self.stack[-1][1]] += text

    def handle_endtag(self, tag):
        if self.card is None:
            return
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break
        if tag == "div":
            self.depth -= 1
            if self.depth == 0:
                self.rows.append(self.card)
                self.card, self.stack = None, []


def _challenge(markup: str) -> bool:
    title = re.search(r"<title\b[^>]*>(.*?)</title>", markup, re.I | re.S)
    return bool(title and re.search(r"安全验证|访问验证|百度验证码", video._clean(title[1]))) or any(
        marker in markup for marker in ("wappass.baidu.com/static/captcha", "Forbid spider access", 'id="captcha"'))


def parse_baidu_video_search(markup: str, search_url: str = SEARCH_URL) -> list[dict]:
    if _challenge(markup):
        raise ValueError("百度视频要求安全验证，已停止该渠道")
    parser = _Cards()
    parser.feed(markup)
    if not parser.saw_results:
        raise ValueError("百度视频网页未返回可读取的结果结构，已停止该渠道")
    result, seen = [], set()
    for item in parser.rows:
        try:
            source, platform, ident = _original(item["original"])
        except ValueError:  # malformed ports or IPv6 in an untrusted card
            continue
        title = video._clean(item["title"])
        thumb = video._http_url(item["thumbnail"].strip())
        if (not source or source in seen or not title or not thumb or len(thumb) > 4096
                or not domestic.allowed(thumb)):
            continue
        seen.add(source)
        duration = video._duration(item["duration"].strip())
        description = video._clean(item["description"]).removeprefix("简介：").strip()
        date = re.search(r"发布时间[：:]\s*(\d{4}-\d{2}-\d{2})", item["text"])
        discovery = {"url": search_url, "title": title, "description": description,
                     "source_url": source, "original_url": item["original"], "duration": duration,
                     "thumbnail_url": thumb, "source_label": video._clean(item["source_label"]),
                     "published_at": date[1] if date else "", "readable": True,
                     "evidence_origin": "aggregator_search"}
        result.append(video._candidate("baidu_video", title, source, thumb,
            duration=duration, video_id=ident, platform_provider=platform, description=description,
            publisher="", publisher_url="", publisher_verified=False, published_at="", source_verified=False,
            discovery_evidence=discovery,
            source_evidence={"url": source, "readable": False, "evidence_origin": "unread_original_page"},
            license="百度视频聚合发现；原发布者、内容及转载许可均待核实"))
    return result[:20]


def _fetch_html(url: str, check) -> str:
    """Stop before following any verification or non-search redirect."""
    started = time.monotonic()
    with execution.network_slot(check):
        with network.client(timeout=httpx.Timeout(8, connect=5), follow_redirects=False,
                            trust_env=False, headers={"User-Agent": video.AGENT}) as client:
            for _ in range(3):
                check()
                remaining = REQUEST_BUDGET - (time.monotonic() - started)
                if remaining <= 0:
                    raise ValueError("百度视频检索超过本次时间预算，已停止该渠道")
                sources.public_url(url)
                with client.stream("GET", url, timeout=min(8, remaining)) as response:
                    if response.is_redirect:
                        target = urljoin(url, response.headers.get("location", ""))
                        p = urlsplit(target)
                        if (p.scheme != "https" or p.netloc != "www.baidu.com"
                                or p.path != "/sf/vsearch"):
                            raise ValueError("百度视频跳转至验证或非检索页面，已停止该渠道")
                        url = target
                        continue
                    if response.status_code in (401, 403, 412, 429, 451):
                        raise ValueError(f"百度视频限制公开访问（HTTP {response.status_code}），已停止该渠道")
                    response.raise_for_status()
                    if video._number(response.headers.get("Content-Length")) > MAX_BYTES:
                        raise ValueError("百度视频结果超过 4 MB 读取预算")
                    body = bytearray()
                    for chunk in response.iter_bytes(65536):
                        check()
                        body.extend(chunk)
                        if len(body) > MAX_BYTES or time.monotonic() - started > REQUEST_BUDGET:
                            raise ValueError("百度视频结果超过本次读取预算，已停止该渠道")
                    markup = bytes(body).decode("utf-8", errors="replace")
                    if _challenge(markup):
                        raise ValueError("百度视频要求安全验证，已停止该渠道")
                    return markup
    raise ValueError("百度视频检索重定向过多，已停止该渠道")


def _stop_path():
    root = paths.DATA / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root / "baidu-video-public-stop.json"


def _write_stop(path, reason):
    temporary = path.with_name(path.name + "." + store.uid() + ".tmp")
    try:
        temporary.write_text(json.dumps({"until": time.time() + COOLDOWN_SECONDS,
                                         "reason": str(reason)[:200]}), "utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def baidu_video_search(query: str, page: int = 1, *, check=None) -> list[dict]:
    """Metadata-only discovery; caches success for 30 minutes and stops on denial."""
    query = re.sub(r"\s+", " ", str(query)).strip()[:200]
    if not query:
        return []
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("百度视频页码必须为正整数")
    if page > MAX_PAGE:
        return []
    check = check or (lambda: None)
    check()
    params = {"pd": "video", "wd": query}
    if page > 1:
        params["pn"] = (page - 1) * 10
    url = SEARCH_URL + "?" + urlencode(params)

    def fetch():
        stop = _stop_path()
        key = str(stop.resolve())
        with execution.keyed_lock(("baidu-public-search", key)):
            check()
            try:
                saved = json.loads(stop.read_text("utf-8"))
            except (OSError, ValueError):
                saved = {}
            if not isinstance(saved, dict):
                saved = {}
            if video._number(saved.get("until")) > time.time():
                raise ValueError("百度视频公开检索已进入 30 分钟冷却；未继续请求验证页面")
            delay = max(0.0, REQUEST_INTERVAL - (time.monotonic() - _last_request.get(key, 0)))
            while delay > 0:
                check()
                time.sleep(min(delay, .2))
                delay = max(0.0, REQUEST_INTERVAL - (time.monotonic() - _last_request.get(key, 0)))
            _last_request[key] = time.monotonic()
            try:
                return parse_baidu_video_search(_fetch_html(url, check), url)
            except (ValueError, httpx.HTTPError) as exc:
                _write_stop(stop, exc)
                raise ValueError("百度视频公开检索停止：" + str(exc)[:180]) from None

    return video._cached("baidu-video-v1", f"{query}\npage={page}", fetch)
