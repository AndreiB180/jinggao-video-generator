"""News topics: articles from the news crawler, a pasted link, or pasted text.

Each item can become a project whose produce job first rewrites the article (rewrite.py)
and then makes the whole video. Items are kept in DATA/news.json (newest first).
"""
from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
import socket
import threading
import time
from html.parser import HTMLParser

import httpx
from urllib.parse import urljoin, urlparse

from . import network, paths, store

MAX_ITEMS = 500
MAX_TEXT = 30000
_lock = threading.Lock()


def _file():
    paths.prepare()
    return paths.DATA / "news.json"


def items() -> list[dict]:
    try:
        rows = json.loads(_file().read_text("utf-8"))
    except (OSError, ValueError):
        return []
    return [r for r in rows if isinstance(r, dict) and r.get("id")] if isinstance(rows, list) else []


def _save(rows: list[dict]) -> None:
    tmp = _file().with_suffix(".tmp")
    tmp.write_text(json.dumps(rows[:MAX_ITEMS], ensure_ascii=False), "utf-8")
    tmp.replace(_file())


def clean_text(text: str) -> str:
    lines = [re.sub(r"[ \t　\xa0]+", " ", line).strip() for line in str(text or "").replace("\r", "").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()[:MAX_TEXT]


def normalize(row: dict, origin: str) -> dict | None:
    """One stored item from a crawler record or user input; None when it has no usable text."""
    title = re.sub(r"\s+", " ", str(row.get("title") or "")).strip()[:200]
    text = clean_text(row.get("text") or row.get("content") or row.get("body") or "")
    if len(re.sub(r"\s+", "", text)) < 30:
        return None
    url = str(row.get("url") or row.get("link") or "").strip()[:2000]
    if url and urlparse(url).scheme not in ("http", "https"):
        url = ""
    key = url or (title + text[:200])
    return {"id": hashlib.sha1(key.encode()).hexdigest()[:16], "title": title or text[:24],
            "text": text, "url": url, "source": str(row.get("source") or urlparse(url).hostname or "")[:80],
            "published": str(row.get("published") or row.get("publish_time") or row.get("time") or "")[:40],
            "origin": origin, "added": time.time(), "chars": len(re.sub(r"\s+", "", text)),
            "projects": [],
            # The crawler's own AI 口播稿 (its 洗稿), used as the script instead of a new rewrite.
            "koubo": clean_text(row.get("koubo") or "") if isinstance(row.get("koubo"), str) else "",
            "koubo_title": str(row.get("koubo_title") or "")[:80]}


def add(rows: list[dict], origin: str) -> tuple[list[dict], int]:
    """Store new items (an item already present keeps its history). Returns (stored items, new count)."""
    fresh = all_fresh = [item for item in (normalize(r, origin) for r in rows if isinstance(r, dict)) if item]
    known = {r["id"] for r in items()}
    fresh = [item for item in fresh if item["id"] not in known]       # cap only what is really new
    if len(fresh) > MAX_ITEMS:
        # Keep the newest by real publish time when most records carry one; otherwise keep the batch's own
        # order (crawlers usually list newest first).
        times = [_timestamp(r["published"]) for r in fresh]
        if sum(t is not None for t in times) >= 0.8 * len(fresh):
            order = sorted(range(len(fresh)), key=lambda i: (times[i] is not None, times[i] or 0), reverse=True)
            fresh = [fresh[i] for i in order[:MAX_ITEMS]]
        else:
            fresh = fresh[:MAX_ITEMS]
    with _lock:
        existing = items()
        known = {r["id"] for r in existing}
        scripts = {item["id"]: item for item in all_fresh if item.get("koubo")}
        for r in existing:
            if r["id"] in scripts and not r.get("koubo"):
                r["koubo"], r["koubo_title"] = scripts[r["id"]]["koubo"], scripts[r["id"]]["koubo_title"]
        new = []
        for item in fresh:
            if item["id"] not in known:
                known.add(item["id"])
                new.append(item)
        _save(new + existing)
    stored = items()
    ids, kept = {item["id"] for item in all_fresh}, {r["id"] for r in stored}
    return [r for r in stored if r["id"] in ids], sum(1 for item in new if item["id"] in kept)


def _timestamp(value: str) -> float | None:
    """Seconds for the publish-time formats crawlers emit: ISO, epoch, 2026年10月13日 10:20, RFC 822."""
    from datetime import datetime, timezone
    from email.utils import parsedate_to_datetime
    text = str(value or "").strip()
    if not text:
        return None
    now = time.time()
    relative = re.fullmatch(r"(\d+)\s*(秒|分钟|小时|天)前", text)
    if relative:
        return now - int(relative.group(1)) * {"秒": 1, "分钟": 60, "小时": 3600, "天": 86400}[relative.group(2)]
    if text in ("刚刚", "刚才"):
        return now
    day = re.fullmatch(r"(今天|昨天|前天)\s*(\d{1,2}):(\d{2})", text)
    if day:
        base = datetime.now().replace(hour=int(day.group(2)), minute=int(day.group(3)), second=0, microsecond=0)
        return base.timestamp() - {"今天": 0, "昨天": 1, "前天": 2}[day.group(1)] * 86400
    short = re.fullmatch(r"(\d{1,2})[-/月](\d{1,2})日?\s*(\d{1,2}):(\d{2})", text)
    if short:
        try:
            return datetime(datetime.now().year, int(short.group(1)), int(short.group(2)),
                            int(short.group(3)), int(short.group(4))).timestamp()
        except ValueError:
            return None
    if re.fullmatch(r"\d{10}(\.\d+)?|\d{13}", text):
        number = float(text)
        return number / 1000 if number > 1e12 else number
    match = re.match(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?(?:[ T]*(\d{1,2}):(\d{2})(?::(\d{2}))?)?", text)
    if match:
        y, mo, d, h, mi, se = (int(g or 0) for g in match.groups())
        try:
            return datetime(y, mo, d, h, mi, se, tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None
    try:
        return parsedate_to_datetime(text).timestamp()
    except (TypeError, ValueError, IndexError):
        return None


def get(item_id: str) -> dict:
    found = next((r for r in items() if r["id"] == item_id), None)
    if not found:
        raise KeyError("新闻不存在或已删除")
    return found


def remove(item_id: str) -> None:
    with _lock:
        _save([r for r in items() if r["id"] != item_id])


def _mark(item_id: str, project_id: str) -> None:
    with _lock:
        rows = items()
        for r in rows:
            if r["id"] == item_id:
                r["projects"] = (r.get("projects") or []) + [project_id]
        _save(rows)


def make_project(item_id: str, *, aspect: str = "16:9", minutes: float | None = None,
                 resolution: str = "720p", auto: bool = True) -> dict:
    """A project holding the article. With auto, a produce job rewrites it and makes the video;
    otherwise the raw article is the draft script, to be rewritten from the editor."""
    item = get(item_id)
    title = (item["title"] or "新闻选题")[:100]
    from .rewrite import clean_script
    draft = clean_script(item["koubo"]) if item.get("koubo") else item["text"]
    project = store.new_project(title, "" if auto else draft, aspect)
    news = {key: item.get(key, "") for key in ("id", "title", "text", "url", "source", "published", "koubo", "koubo_title")} \
        | {"minutes": minutes}
    project = store.mutate(project["id"], lambda p: p.update(news=news))
    job = store.enqueue(project["id"], "produce", {"resolution": resolution, "mode": "model"}) if auto else None
    _mark(item_id, project["id"])
    return {"project_id": project["id"], "title": project["title"], "job_id": job["id"] if job else None}


# --- a pasted link -----------------------------------------------------------------------------

class _Article(HTMLParser):
    """Paragraph text grouped by its parent block; the block with the most text is the article."""
    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "iframe", "svg", "button",
            "select", "textarea"}
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
            "track", "wbr", "keygen"}
    # A new block ends an unclosed <p> (HTML's implied end tag), so its text is not lost.
    CLOSES_P = {"p", "div", "section", "article", "ul", "ol", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6",
                "blockquote", "pre", "figure", "dl", "hr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.skip, self.blocks, self.title, self.meta = [], 0, {}, "", {}
        self.ids, self.serial = [], 0       # one number per open element: blocks group by container, not tag path
        self.sig, self.paras = {}, []       # element number -> "tag.class"; paragraphs in document order
        self.link_chars = 0                 # text inside <a> in the current paragraph (link lists are not prose)
        self.current, self.in_title, self.in_h1, self.h1 = None, False, False, ""

    def _close_p(self):
        if "p" not in self.stack:
            return
        while self.stack:
            top = self.stack.pop()
            self.ids.pop()
            if top in self.SKIP:
                self.skip -= 1
            if top == "p":
                self._flush()
                return

    def _flush(self):
        if self.current is not None:
            text = re.sub(r"\s+", " ", "".join(self.current)).strip()
            if text and self.link_chars > .5 * len(text):
                text = ""
            if text:
                self.blocks.setdefault(tuple(self.ids), []).append(text)
                self.paras.append((tuple(self.ids), text))
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and attrs.get("content"):
            name = (attrs.get("property") or attrs.get("name") or "").lower()
            if name in ("og:title", "og:site_name", "article:published_time", "publishdate", "pubdate"):
                self.meta[name] = attrs["content"]
        if tag == "br" and self.current is not None:
            self.current.append("\n")
        if tag in self.CLOSES_P:
            self._close_p()
        if tag in self.VOID:
            return
        self.stack.append(tag)
        self.serial += 1
        self.ids.append(self.serial)
        self.sig[self.serial] = tag + "." + " ".join(sorted((attrs.get("class") or "").split()))
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self.in_title = True
        elif tag == "h1" and not self.h1:
            self.in_h1 = True
        elif tag == "p" and not self.skip:
            self.current, self.link_chars = [], 0

    def handle_endtag(self, tag):
        if tag not in self.stack:
            return
        while self.stack:
            top = self.stack.pop()
            self.ids.pop()
            if top in self.SKIP:
                self.skip -= 1
            if top == "title":
                self.in_title = False
            if top == "h1":
                self.in_h1 = False
            if top == "p":
                self._flush()
            if top == tag:
                break

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        if self.in_h1:
            self.h1 += data
        if self.current is not None and not self.skip:
            self.current.append(data)
            if "a" in self.stack:
                self.link_chars += len(data.strip())

    def close(self):
        super().close()
        self._close_p()


def extract(page: str) -> dict:
    parser = _Article()
    parser.feed(page)
    parser.close()
    # The article is the container with the most prose (sentences with punctuation), not the longest list.
    prose = lambda ps: sum(len(p) for p in ps if len(p) > 12 and re.search(r"[。，！？；,.!?;]", p))
    best_key = max(parser.blocks, key=lambda k: prose(parser.blocks[k]), default=None)
    best = []
    if best_key == ():
        best = parser.blocks[()]
    elif best_key is not None:
        # Pages that wrap each paragraph in its own element (百家号 textWrap, WeChat sections): take every
        # sibling wrapper of the same kind under the same parent, in document order. Side columns differ.
        sig = parser.sig.get(best_key[-1])
        best = [text for key, text in parser.paras
                if key == best_key or (len(key) == len(best_key) and key[:-1] == best_key[:-1]
                                       and parser.sig.get(key[-1]) == sig)]
    paragraphs = [p for p in best if not re.fullmatch(r"(责任编辑|编辑|作者|来源|记者|图片来源)[:：].{0,30}|[【\[].{0,20}[】\]]", p)]
    title = re.sub(r"\s+", " ", parser.meta.get("og:title") or parser.h1 or parser.title).strip()
    site = parser.meta.get("og:site_name", "").strip()
    # Only a site-name suffix is cut ("标题 - 中新网", "标题_新浪新闻"), never part of the headline ("1-2负于泰国").
    suffix = re.search(r"\s*(?:\s[-|｜]\s|_|｜)\s*([^-_|｜]{2,20})$", title)
    if suffix and len(title[:suffix.start()].strip()) >= len(suffix.group(1)) and (
            site and site in suffix.group(1) or re.search(r"(网|新闻|日报|晚报|报|频道|在线|客户端|News)$", suffix.group(1).strip())):
        title = title[:suffix.start()]
    text = "\n".join(paragraphs)
    text = re.sub(r"\s*[（(]完[)）]\s*|\s*[【\[](责任)?编辑[:：][^】\]]{0,20}[】\]]\s*", "\n", text).strip()
    return {"title": html.unescape(title)[:200], "text": text,
            "source": parser.meta.get("og:site_name", ""), "published": parser.meta.get("article:published_time", "")}


FAKE_IP = ipaddress.ip_network("198.18.0.0/15")    # proxy "fake-ip" answers (Clash/Surge TUN mode)


def _usable(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])
    return ip.is_global or ip in FAKE_IP


def _resolve(host: str) -> str:
    """One checked address for host: every answer must be public, and the request then goes to exactly
    this address (so a second DNS answer cannot point the connection at a local service)."""
    try:
        ipaddress.ip_address(host)
        literal = True
    except ValueError:
        literal = False
    if literal:
        if not _usable(host):
            raise ValueError("只能读取公网新闻页面")
        return host
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError:
        raise ValueError("无法解析新闻网址的域名，请检查链接") from None
    addresses = [info[4][0] for info in infos]
    if not addresses or not all(_usable(a) for a in addresses):
        raise ValueError("只能读取公网新闻页面")
    return next((a for a in addresses if ":" not in a), addresses[0])


def _proxied(url: str) -> bool:
    import os
    from urllib.request import getproxies
    scheme = urlparse(url).scheme
    proxies = getproxies()
    return bool(proxies.get(scheme) or proxies.get("all")
                or any(os.environ.get(k) for k in (f"{scheme}_proxy", f"{scheme.upper()}_PROXY", "all_proxy", "ALL_PROXY")))


def _public(host: str) -> bool:
    try:
        _resolve(host)
        return True
    except ValueError:
        return False


MAX_PAGE = 8_000_000


def fetch(url: str) -> dict:
    try:
        return _fetch(url)
    except httpx.InvalidURL:
        raise ValueError("请粘贴有效的新闻链接（网址里不能有全角字母或空白）") from None
    except httpx.HTTPError as exc:
        raise ValueError(f"新闻页面无法打开：{type(exc).__name__}") from None


def _host(url: str) -> str:
    """The host exactly as httpx will connect to it (IDNA 2008), so the guard checks the same name."""
    return httpx.URL(url).raw_host.decode("ascii").strip("[]")





def _fetch(url: str) -> dict:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("请粘贴以 http:// 或 https:// 开头的新闻链接")
    with network.client(timeout=20, follow_redirects=False, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9"}) as client:
        for _ in range(5):          # follow redirects by hand so every hop is checked
            host = _host(url)
            address = _resolve(host)
            # Behind an HTTP proxy the proxy resolves and tunnels the name; pinning to an IP would break TLS.
            pinned = httpx.URL(url) if _proxied(url) else httpx.URL(url).copy_with(host=address)
            # Connect to the checked address; the name stays in the Host header and the TLS certificate check.
            with client.stream("GET", pinned, headers={"Host": httpx.URL(url).netloc.decode("ascii")},
                               extensions={"sni_hostname": host}) as response:
                if response.is_redirect and response.headers.get("location"):
                    url = urljoin(url, response.headers["location"])
                    continue
                if response.status_code >= 400:
                    raise ValueError(f"新闻页面无法打开（HTTP {response.status_code}）")
                kind = response.headers.get("content-type", "text/html").lower()
                if not any(t in kind for t in ("html", "xml", "text/plain")):
                    raise ValueError("链接不是网页（可能是视频或文件），请粘贴新闻页面链接")
                body = bytearray()
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) > MAX_PAGE:
                        raise ValueError("页面过大，无法读取")
                encoding = (response.charset_encoding or "").lower()
            break
        else:
            raise ValueError("新闻链接重定向次数过多")
    content = bytes(body)
    declared = re.search(rb'<meta[^>]+charset=["\']?([\w-]+)', content[:4000], re.I)
    declared = declared.group(1).decode().lower() if declared else ""
    # GB2312/GBK pages often contain rarer characters (names like 镕) only GB18030 decodes.
    if encoding in ("gbk", "gb2312", "gb18030") or declared in ("gbk", "gb2312", "gb18030"):
        page = content.decode("gb18030", "replace")
    else:
        try:
            page = content.decode(encoding or declared or "utf-8", "replace")
        except LookupError:
            page = content.decode("utf-8", "replace")
    found = extract(page)
    if len(re.sub(r"\s+", "", found["text"])) < 80:
        raise ValueError("没有从页面中识别出正文，请直接复制新闻原文粘贴")
    return found | {"url": url, "source": found["source"] or urlparse(url).hostname or ""}
