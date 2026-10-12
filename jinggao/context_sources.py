"""Bounded, public text corroboration for dates absent from video titles."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from . import execution, identity, video_sources


PHOTO_PUBLISHERS = ("news.cn", "xinhuanet.com", "people.com.cn", "chinanews.com.cn", "chinanews.com",
                    "cctv.com", "cntv.cn", "peopleapp.com")


def photo_publisher(url):
    host = urlparse(url).hostname or ""
    return any(host == domain or host.endswith("."+domain) for domain in PHOTO_PUBLISHERS)


class PhotoCaptions(HTMLParser):
    """Bind image alt text or its immediately following paragraph, not article titles."""
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.source, self.pending, self.in_paragraph = source, None, False
        self.text, self.rows = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img":
            self.pending = urljoin(self.source, attrs.get("src", ""))
            self.text = []
            alt = attrs.get("alt", "").strip()
            if alt:
                self.rows.append({"image_url": self.pending, "caption": alt[:500], "method": "image_alt"})
        elif tag in ("p", "figcaption"):
            self.in_paragraph = True
            self.text = []
        elif tag in ("script", "style", "h1", "h2", "section", "hr"):
            self.pending = None

    def handle_data(self, value):
        if self.in_paragraph:
            self.text.append(value)

    def handle_endtag(self, tag):
        if tag in ("p", "figcaption"):
            caption = re.sub(r"\s+", " ", "".join(self.text)).strip()
            if self.pending and caption:
                # A short, adjacent photo caption can bind a name. A long
                # article paragraph or a nearby headline cannot lend identity.
                if 5 <= len(caption) <= 180:
                    self.rows.append({"image_url": self.pending, "caption": caption, "method": "adjacent_caption"})
                self.pending = None
            self.in_paragraph = False
            self.text = []


def image_caption(candidate, check):
    source, target = candidate.get("source_url", ""), candidate.get("url", "")
    if not photo_publisher(source):
        return candidate
    def fetch():
        data, _ = video_sources._fetch(source, limit=2*1024*1024, check=check)
        parser = PhotoCaptions(source)
        parser.feed(data.decode("utf-8", errors="replace"))
        return parser.rows
    rows = video_sources._cached("photo-caption", source, fetch)
    bound = [r for r in rows if r["image_url"] == target]
    if not bound:
        return candidate
    caption = "\n".join(dict.fromkeys(r["caption"] for r in bound))
    return candidate | {"provider": "publisher_image", "source_verified": True,
        "source_evidence": {"url": source, "title": caption, "description": "",
            "article_title": candidate["title"], "readable": True, "evidence_origin": "publisher_page",
            "image_url": target, "photo_attribution": "publisher_caption"}}


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts, self.current = [], []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1
        if tag in {"p", "h1", "h2", "br", "li"} and self.current:
            self.parts.append("".join(self.current))
            self.current = []

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.hidden = max(0, self.hidden-1)
        if tag in {"p", "h1", "h2", "li"} and self.current:
            self.parts.append("".join(self.current))
            self.current = []

    def handle_data(self, text):
        if not self.hidden:
            self.current.append(text)


def search(query, check):
    def fetch():
        from .publisher_videos import chinanews_articles
        links = chinanews_articles(query, check)
        def read(item):
            url, title = item
            try:
                data, _ = video_sources._fetch(url, limit=2*1024*1024, check=check)
                parser = Paragraphs()
                parser.feed(data.decode("utf-8", errors="replace"))
                paragraphs = [re.sub(r"\s+", " ", p).strip() for p in parser.parts]
                paragraphs = [p[:1600] for p in paragraphs if 30 <= len(p) <= 1600 and identity.event_dates(p)]
                return {"url": url, "title": title, "paragraphs": paragraphs[:12],
                        "readable": True, "evidence_origin": "publisher_page"}
            except ValueError:
                return None
        return [r for r in execution.ordered_map(read, links[:3], 3) if r and r["paragraphs"]]
    return video_sources._cached("context-text-domestic", query, fetch)
