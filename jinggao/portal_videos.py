"""Public domestic portal discovery. No account, challenge or DRM workarounds.

Sohu's search endpoint is currently blocked; its independently public editorial
directories are a smaller source and are explicitly labelled as such.
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse

from . import domestic, sources, video_sources as video

_locks = {p: threading.Lock() for p in ("iqiyi", "youku", "sohu")}
_cooldown = {}
SOHU_FEEDS = (
    "life/meishi/", "life/", "health/", "culture/", "popular-science/", "edu/parent/",
    "tec/", "life/journey/", "life/family/", "fun/", "edu/", "pgc-mil/", "auto/", "game/", "musician/",
)


def _public(provider, action):
    with _locks[provider]:
        if time.monotonic() < _cooldown.get(provider, 0):
            raise ValueError("该平台公开检索处于验证冷却期，已跳过")
        try:
            return action()
        except ValueError as exc:
            if any(s in str(exc) for s in ("验证", "403", "412", "429", "限流")):
                _cooldown[provider] = time.monotonic()+1800
            raise


def iqiyi_video_id(url):
    """The public player's reversible URL ID, not an album/redirect identifier."""
    match = re.fullmatch(
        r"https?://(?:www\.iqiyi\.com/v_([a-zA-Z0-9]{1,11})\.html|"
        r"[a-z]+\.iqiyi\.com/resource/pcw/play/([a-zA-Z0-9]{1,11})(?:\?playertype=immersiveShort)?)", url)
    if not match:
        return ""
    value = int(match[1] or match[2], 36) ^ 0x75706971676C
    if value < 900000:
        value = 100 * (value + 900000)
    # Entity/album IDs ending in 01/14 require a separate episode redirect.
    # The player accepts ordinary video IDs ending in 00, 07 or 09.
    return str(value) if 0 < value < 2**53 and value % 100 in (0, 7, 9) else ""


def parse_iqiyi_search(body):
    if not isinstance(body,dict):
        raise ValueError("爱奇艺公开检索格式已变化")
    if body.get("code") != 0:
        raise ValueError("爱奇艺公开检索暂不可用："+str(body.get("code")))
    found = {}
    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            url = video._http_url(value.get("pageUrl"))
            # Album/person cards can link to a sample episode. Only use actual
            # video cards with their own duration and ordinary public play flags.
            ident = iqiyi_video_id(url)
            if (ident and str(value.get("qipuId", "")) == ident
                    and value.get("duration") and value.get("title") and not value.get("payMarkUrl")
                    and not value.get("firstVideoIsVip") and "ischarge=true" not in str(value.get("playUrl") or "")):
                title = video._clean(value["title"])
                thumb = video._http_url(value.get("img"))
                if thumb and domestic.allowed(thumb):
                    # Never inherit the enclosing star/album's biography as
                    # evidence for unrelated videos in that person's search card.
                    def label(key):
                        item = value.get(key)
                        return video._clean(item.get("value") if isinstance(item,dict) else item)
                    desc = video._clean(value.get("description") or value.get("subTitle") or label("brief"))
                    publisher,date = label("uploader"),label("releaseTime")
                    found.setdefault(url, video._candidate("iqiyi", title, url, thumb,
                        duration=video._number(value["duration"])/1000,
                        video_id=str(value.get("qipuId", "")), description=desc,
                        publisher=publisher, published_at=date, source_verified=True,
                        license="爱奇艺公开内容；转载及商用许可需核实",
                        source_evidence=video._evidence(url,title,desc,publisher,"",date,"publisher_search")))
            for child in value.values():
                if isinstance(child,(list,dict)):
                    walk(child)
    walk(body.get("data",{}))
    return list(found.values())[:60]


def iqiyi_search(query, page=1):
    def fetch():
        body = video._json("https://mesh.if.iqiyi.com/portal/lw/search/homePageV3?"+urlencode({
            "key":query,"pageNum":page,"pageSize":20}), headers={"Referer":"https://www.iqiyi.com/"})
        return parse_iqiyi_search(body)
    return video._cached("iqiyi", f"{query}\npage={page}", lambda:_public("iqiyi",fetch))


def parse_iqiyi_playback(body, video_id):
    """Parse the same anonymous startup response used by the public web player.

    `ev` is a JSON envelope encoded by a fixed XOR in the published player JS;
    it is not a media key or a DRM license. No media decryption is performed.
    """
    if not isinstance(body, dict):
        raise ValueError("爱奇艺公开播放数据格式已变化")
    info = body.get("videoInfo") or {}
    if not isinstance(info, dict) or any(str(value) != video_id for value in (
            body.get("tvid"), body.get("tvId"), info.get("tvId"))):
        raise ValueError("爱奇艺返回的视频编号与请求不一致，未采用广告或推荐视频")
    if (info.get("payMark") != 0 or info.get("effective") is not True
            or info.get("isLock") is not False or info.get("pagePublishStatus") != "PAGE_ONLINE"
            or info.get("isEnabledInteraction") or info.get("downloadAllowed") is False):
        raise ValueError("此爱奇艺视频需要额外权限、不可下载或含交互，已跳过")
    envelope = body.get("ev")
    if not isinstance(envelope, str) or not envelope or len(envelope) > 2*1024*1024:
        raise ValueError("爱奇艺没有返回普通公开播放资料")
    try:
        dash = json.loads("".join(chr(ord(char) ^ 90) for char in envelope))
    except (ValueError, TypeError):
        raise ValueError("爱奇艺公开播放资料格式已变化") from None
    data = dash.get("data") if isinstance(dash, dict) else None
    if not isinstance(data, dict) or str(data.get("tvid")) != video_id:
        raise ValueError("爱奇艺内容流编号与请求不一致")
    if dash.get("code") != "A00000" or not 100 < video._number(data.get("st")) < 200:
        raise ValueError("爱奇艺限制此视频的公开播放，未重试其他权限或地区")
    content = data.get("content") or {}
    boss = data.get("boss_ts") or data.get("boss") or {}
    vip_types = content.get("vipTypes") or [] if isinstance(content, dict) else None
    if (not isinstance(content, dict) or content.get("bossStatus") != 0
            or not isinstance(vip_types, list) or any(str(tier) != "0" for tier in vip_types)
            or boss or data.get("is_preview") or data.get("is_drm")):
        raise ValueError("此爱奇艺内容需要付费、鉴权或仅可试看，已跳过")
    program = data.get("program") or {}
    renditions = program.get("video", []) if isinstance(program, dict) else []
    streams = []
    for item in renditions if isinstance(renditions, list) else []:
        if not isinstance(item, dict) or item.get("_selected") is not True:
            continue
        # The published player labels -1/1 as clear; 7/8/9 are DRM systems.
        # Unknown future types also fail closed, even if a URL is present.
        if (item.get("drmType") not in (-1, 1) or item.get("drm") or item.get("license_host")
                or item.get("has_drm") or item.get("isPreview") or item.get("ptt")
                or item.get("ff") != "ts" or item.get("code") != 2):
            continue
        size = re.fullmatch(r"(\d{1,5})x(\d{1,5})", str(item.get("scrsz", "")))
        url = video._http_url(item.get("ml") or item.get("m3u8Url"))
        if not size or not domestic.allowed(url):
            continue
        width, height = map(int, size.groups())
        if not 0 < min(width, height) <= max(width, height) <= 16384:
            continue
        streams.append({"url": url, "format": "hls", "container": "mp4", "width": width,
            "height": height, "duration": video._duration(item.get("duration")),
            "iqiyi_video_id": video_id, "iqiyi_dispatch_common": item.get("ms", ""),
            "iqiyi_dispatch_files": item.get("mk") or {}})
    if not streams:
        raise ValueError("爱奇艺没有返回普通可解码的国内视频流；DRM、试看和未知格式已排除")
    return info, streams


def iqiyi_playback(candidate, check):
    source = candidate.get("source_url", "")
    ident = iqiyi_video_id(source)
    if not ident or candidate.get("video_id") and str(candidate["video_id"]) != ident:
        raise ValueError("爱奇艺链接与视频编号不一致，或不是单个公开视频")

    def fetch():
        body = video._json("https://mesh.if.iqiyi.com/player/lw/lwplay/accelerator.js?" + urlencode({
            "tvid": ident, "apiVer": 3, "format": "json",
            # Public web player's versioned response uses external VOD manifests.
            "lwaver": "17.094.26512", "appver": "17.094.26512"}),
            check=check, headers={"Referer": source})
        info, streams = parse_iqiyi_playback(body, ident)
        for stream in streams:
            manifest = video._hls(stream["url"], check, source)
            if manifest["variants"] or not manifest["segments"]:
                raise ValueError("爱奇艺当前清单不是单个普通点播视频")
            if any(not domestic.allowed(part["url"]) for part in manifest["segments"]):
                raise ValueError("爱奇艺清单包含未接入的媒体来源，已停止")
            duration = manifest["duration"]
            if stream["duration"] and abs(duration-stream["duration"]) > max(2, stream["duration"]*0.02):
                raise ValueError("爱奇艺清单时长与原视频不一致，未采用试看或广告")
            stream["duration"] = duration
        title = video._clean(info.get("title"))
        if not title:
            raise ValueError("爱奇艺公开播放资料缺少原视频标题")
        evidence = video._evidence(source, title, candidate.get("description", ""),
            candidate.get("publisher", ""), candidate.get("publisher_url", ""),
            candidate.get("published_at", ""), "platform_metadata")
        evidence["video_id"] = ident
        return candidate | {"title": title, "video_id": ident, "source_verified": True,
            "source_evidence": evidence, "media_extractor": "iqiyi-public", "media_streams": streams,
            "duration": max(stream["duration"] for stream in streams)}
    return _public("iqiyi", fetch)


def resolve_iqiyi_segment(segment, stream, referer, check):
    """Follow the public player's per-file dispatch to the ordinary TS bytes.

    The server already supplies all query values in `ms`/`mk`; never generate
    signatures, request new permissions, decrypt media, or retry another region.
    """
    ident = iqiyi_video_id(referer)
    original = video._http_url(segment.get("url"))
    parsed = urlparse(original)
    if (not ident or stream.get("iqiyi_video_id") != ident
            or parsed.hostname != "data.video.iqiyi.com" or not parsed.path.endswith(".ts")):
        raise ValueError("爱奇艺分段调度来源与原视频不一致")
    per_file = stream.get("iqiyi_dispatch_files") or {}
    suffix = stream.get("iqiyi_dispatch_common") or (
        per_file.get(parsed.path.rsplit("/", 1)[-1]) if isinstance(per_file, dict) else "")
    if not isinstance(suffix, str) or not suffix or len(suffix) > 8192 or any(c in suffix for c in "\r\n#"):
        raise ValueError("爱奇艺分段缺少官方公开调度资料")
    query = parse_qs(parsed.query + "&" + suffix)
    if not query.get("qd_tvid") or any(value != ident for value in query["qd_tvid"]):
        raise ValueError("爱奇艺分段编号与原视频不一致")
    dispatch = original + ("&" if parsed.query else "?") + suffix + "&pv=0.1&cross-domain=1&mss=3&stauto=0"
    body = _public("iqiyi", lambda: video._json(dispatch, check=check, headers={"Referer": referer}))
    nodes = body.get("d") or []
    if (body.get("code") not in (None, "A00000") or not isinstance(nodes, list) or not nodes
            or not isinstance(nodes[0], dict) or urlparse(str(body.get("uri", ""))).path != parsed.path):
        raise ValueError("爱奇艺没有返回匹配此片段的公开分发地址")
    target = video._http_url(nodes[0].get("URL"))
    resolved = urlparse(target)
    target_query = parse_qs(resolved.query)
    if (not domestic.allowed(target) or resolved.path != parsed.path
            or target_query.get("qd_tvid") != [ident]
            or any(target_query.get(key) != query.get(key) for key in ("start", "end", "sd"))):
        raise ValueError("爱奇艺分发地址的来源、视频或片段范围不一致")
    return target


def sohu_video_id(url):
    match = re.fullmatch(r"https?://tv\.sohu\.com/v/([\w=-]+)\.html",url)
    if not match:
        return ""
    try:
        path = base64.urlsafe_b64decode(match[1]+"="*(-len(match[1])%4)).decode()
    except (ValueError,UnicodeError):
        return ""
    match = re.fullmatch(r"(?:(?:us|pl)/\d+/|\d+/n)(\d+)\.shtml",path)
    return match[1] if match else ""


class _Cards(HTMLParser):
    def __init__(self, provider):
        super().__init__(convert_charrefs=True)
        self.provider,self.anchor,self.cards = provider,None,{}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a":
            url = urljoin("https://tv.sohu.com/" if self.provider == "sohu" else "https://www.youku.com/", a.get("href",""))
            valid = sohu_video_id(url) if self.provider == "sohu" else re.fullmatch(r"https?://v\.youku\.com/v_show/id_([A-Za-z0-9=]+)\.html",url)
            self.anchor = {"url":url,"title":a.get("title", ""),"text":"","thumb":""} if valid else None
        elif tag == "img" and self.anchor is not None:
            self.anchor["thumb"] = urljoin(self.anchor["url"],a.get("src") or a.get("data-src") or "")
            self.anchor["title"] = self.anchor["title"] or a.get("alt", "")

    def handle_data(self, text):
        if self.anchor is not None:
            self.anchor["text"] += text

    def handle_endtag(self, tag):
        if tag != "a" or self.anchor is None:
            return
        a,self.anchor = self.anchor,None
        title = video._clean(a["title"] or a["text"])
        if not title or not a["thumb"] or not domestic.allowed(a["thumb"]):
            return
        url = a["url"]
        self.cards[url] = video._candidate(self.provider,title,url,a["thumb"],
            video_id=sohu_video_id(url) if self.provider == "sohu" else url.split("id_")[-1].split(".")[0],
            source_verified=True, license="公开节目目录；转载及商用许可需核实",
            source_evidence=video._evidence(url,title,"","","","","publisher_search"))


def parse_cards(markup, provider):
    if any(t in markup for t in ("anti-reptile", "_____tmd_____", "x5secdata", "验证码")):
        raise ValueError("原站要求验证，停止该渠道采集")
    parser = _Cards(provider)
    parser.feed(markup)
    return list(parser.cards.values())


def sohu_directory():
    def fetch():
        result = {}
        for feed in SOHU_FEEDS:
            raw,_ = video._fetch("https://tv.sohu.com/ugc/"+feed)
            charset = "gb18030" if re.search(br"charset\s*=\s*[\"']?gb",raw[:3000],re.I) else "utf-8"
            for c in parse_cards(raw.decode(charset,errors="replace"),"sohu"):
                result[c["source_url"]] = c
        return list(result.values())
    return video._cached("sohu-directory", "public-channels-v1", lambda:_public("sohu", fetch))


def sohu_search(query, page=1):
    tokens = sources.tokens(query)
    matches = [(len(tokens & sources.tokens(c["title"])),c) for c in sohu_directory()]
    matches = sorted((pair for pair in matches if pair[0]),key=lambda p:-p[0])
    return [c for _,c in matches[(page-1)*30:page*30]]


def youku_search(query, page=1):
    def fetch():
        raw,_ = video._fetch("https://so.youku.com/search_video/q_"+quote(query)+"?pg="+str(page))
        result = parse_cards(raw.decode("utf-8",errors="replace"),"youku")
        if not result:
            raise ValueError("优酷网页未返回可读取的单视频条目；开放平台另需开发者应用")
        return result
    return video._cached("youku", f"{query}\npage={page}",lambda:_public("youku",fetch))
