"""Domestic platform search and ordinary, unencrypted playback extraction.

Search uses the requests exposed by the public web clients. Playback uses public
Bilibili metadata/playback or the bundled Tencent extractor for one ordinary video.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from urllib.parse import parse_qsl, urlencode

from . import domestic, video_sources as video

_bili_lock = threading.Lock()
_bili_blocked_until = 0.0


def parse_tencent_search(body: dict) -> list[dict]:
    if body.get("ret") != 0 or body.get("data", {}).get("errcode") != 0:
        raise ValueError("腾讯视频搜索暂不可用")
    data = body["data"]
    result, seen = [], set()
    # Covers, paid series and advertisements are not individual footage items.
    for item in video._walk({"normalList": data.get("normalList"),
                             "areaBoxList": data.get("areaBoxList")}, "videoInfo"):
        if not isinstance(item, dict) or not isinstance(item.get("videoDoc"), dict):
            continue
        doc = item["videoDoc"]
        key = dict(parse_qsl(doc.get("dataKey", "")))
        vid = key.get("vid", "")
        if not re.fullmatch(r"[A-Za-z0-9]{11}", vid) or vid in seen or doc.get("payStatus"):
            continue
        seen.add(vid)
        title, description = video._clean(item.get("title")), video._clean(item.get("descrip"))
        publisher, date = video._clean(doc.get("uploader")), video._clean(item.get("checkupTime"))
        source = "https://v.qq.com/x/page/" + vid + ".html"
        thumbnail = video._http_url(item.get("imgUrl")).replace("http://", "https://", 1)
        if not title or not thumbnail or not domestic.allowed(thumbnail):
            continue
        result.append(video._candidate("tencent", title, source, thumbnail,
            duration=video._duration(doc.get("timeLong")), video_id=vid, description=description,
            publisher=publisher, publisher_verified=False, published_at=date, source_verified=True,
            license="腾讯视频公开内容；转载及商用许可需核实",
            source_evidence=video._evidence(source, title, description, publisher, "", date, "publisher_search")))
    return result[:40]


def tencent_search(query: str, page: int = 1) -> list[dict]:
    def fetch():
        payload = {"version": "26022601", "clientType": 1, "query": query, "pagenum": max(0, page-1),
            "pagesize": 30, "queryFrom": 0, "filterValue": "", "uuid": str(uuid.uuid4()), "retry": 0,
            "isPrefetch": True, "searchDatakey": "", "transInfo": "", "isneedQc": True,
            "preQid": "", "adClientInfo": "", "extraInfo": {"multi_terminal_pc": "1",
                "isNewMarkLabel": "0", "themeType": "0", "sugRelatedIds": "{}", "appVersion": "",
                "frontVersion": "26060108"},
            "featureList": ["DEFAULT_FEFEATURE", "PC_SHORT_VIDEOS_WATERFALL", "PC_WANT_EPISODE_V2", "PC_WANT_EPISODE"]}
        raw, _ = video._fetch("https://pbaccess.video.qq.com/trpc.videosearch.mobile_search.MultiTerminalSearch/MbSearch?vversion_platform=2",
            headers={"Origin": "https://v.qq.com", "Referer": "https://v.qq.com/"}, json_payload=payload)
        try:
            return parse_tencent_search(json.loads(raw))
        except (TypeError, KeyError, json.JSONDecodeError):
            raise ValueError("腾讯视频搜索数据格式已变化") from None
    return video._cached("tencent", query if page == 1 else f"{query}\npage={page}", fetch)


def bilibili_search(query: str, page: int = 1) -> list[dict]:
    from . import bilibili_web_search
    deadline = time.monotonic() + bilibili_web_search.BROWSER_TIMEOUT
    def check_backoff():
        now = time.monotonic()
        if now < bilibili_web_search.blocked_until:
            raise ValueError("B站普通匿名网页需要验证，本轮已跳过该渠道")
        if now < bilibili_web_search.unavailable_until:
            raise bilibili_web_search.PublicSearchUnavailable("B站普通网页加载暂不可用，60秒短暂退避中，已跳过本次检索")
    def fetch():
        with _bili_lock:
            check_backoff()
        # At most two independent anonymous browsers. Do not hold the state
        # lock through navigation. Queue wait and navigation share one budget.
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not bilibili_web_search.SEARCH_SLOTS.acquire(timeout=remaining):
            raise ValueError("B站公开检索繁忙，本次已跳过")
        try:
            with _bili_lock:
                check_backoff()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("B站公开检索等待已达到时限，本次已跳过")
            try:
                return bilibili_web_search.read_public_page(query, page, timeout=remaining)
            except bilibili_web_search.PublicSearchBlocked:
                with _bili_lock:
                    bilibili_web_search.blocked_until = time.monotonic() + 1800
                raise
            except bilibili_web_search.PublicSearchUnavailable:
                with _bili_lock:
                    bilibili_web_search.unavailable_until = time.monotonic() + bilibili_web_search.TRANSIENT_BACKOFF
                raise
        finally:
            bilibili_web_search.SEARCH_SLOTS.release()
    # A protocol-specific key does not inherit empty/old API search responses.
    return video._cached("bilibili-web-v1", query if page == 1 else f"{query}\npage={page}", fetch)


def inspect_platform(candidate: dict, check) -> dict:
    source = candidate.get("source_url", "")
    if re.fullmatch(r"https://v\.qq\.com/x/(?:page|cover/\w+)/([A-Za-z0-9]{11})\.html", source):
        extractor, label = "vqq:video", "腾讯视频"
        video_id = source.rsplit("/", 1)[-1].split(".")[0]
    elif re.fullmatch(r"https://(?:www\.)?bilibili\.com/video/(BV[0-9A-Za-z]{10})/?", source):
        return _bilibili_public(candidate, check)
    elif video._host(source, "iqiyi.com"):
        from .portal_videos import iqiyi_playback
        return iqiyi_playback(candidate, check)
    elif re.fullmatch(r"https?://v\.youku\.com/v_show/id_([A-Za-z0-9]+={0,2})\.html",source):
        extractor,label = "Youku","优酷"
        video_id = source.split("id_")[-1].split(".")[0].rstrip("=")
    elif source.startswith(("https://tv.sohu.com/v/", "http://tv.sohu.com/v/")):
        from .portal_videos import sohu_video_id
        video_id = sohu_video_id(source)
        if not video_id:
            raise ValueError("搜狐链接不是单个视频，未展开合集")
        extractor,label = "SohuV,Sohu","搜狐"
    else:
        raise ValueError("解析器只接受国内平台的单个公开视频链接")
    command = video._ytdlp_command()
    if not command:
        raise ValueError("视频解析组件缺失，请使用完整运行包")
    args = [*command, "--ignore-config", "--no-cache-dir", "--no-cookies", "--no-cookies-from-browser",
        "--no-plugin-dirs", "--no-remote-components", "--no-update", "--no-playlist", "--skip-download",
        "--dump-single-json", "--no-check-formats", "--use-extractors", extractor, "--proxy", "",
        "--socket-timeout", "8", "--retries", "0", "--fragment-retries", "0", "--extractor-retries", "0",
        "--file-access-retries", "0", "--no-geo-bypass", "--no-js-runtimes", "--", source]
    body = video._run_ytdlp_json(args, check, timeout=24, label=label)
    same = str(body.get("id")) == video_id if video_id else body.get("webpage_url") == source and bool(body.get("id"))
    if not same or body.get("_type", "video") != "video" or body.get("entries"):
        raise ValueError("解析结果不是请求的单个视频")
    if body.get("is_live") or body.get("has_drm") or body.get("availability") in {
            "private", "premium_only", "subscriber_only", "needs_auth"}:
        raise ValueError("此视频需要额外权限或仍在直播，已跳过")
    formats = {}
    for f in body.get("formats", []):
        url, protocol = video._http_url(f.get("url")), f.get("protocol")
        if (not domestic.allowed(url) or f.get("has_drm") or f.get("vcodec") == "none"
                or protocol not in ("https", "http", "m3u8", "m3u8_native") or f.get("ext") != "mp4"):
            continue
        width, height = int(video._number(f.get("width"))), int(video._number(f.get("height")))
        if not 0 <= min(width, height) <= max(width, height) <= 16384:
            continue
        # Sohu labels every rendition with the original upload's dimensions.
        # Keep its normal/high/super streams until actual header inspection.
        key = (width, height, f.get("format_id")) if label == "搜狐" else (width,height)
        stream = {"url": url, "width": width, "height": height, "container": "mp4",
            "format": "hls" if protocol.startswith("m3u8") else "file",
            "filesize": int(video._number(f.get("filesize"))), "bandwidth": video._number(f.get("tbr"))*1000}
        # Extractor order prefers the platform's main CDN. One rendition per
        # resolution avoids probing the same video on four mirrors.
        formats.setdefault(key, stream)
    if not formats:
        raise ValueError("没有普通可解码的国内视频流")
    for stream in formats.values():
        if label == "搜狐" or not stream["width"] or not stream["height"]:
            probe = video.probe_stream(stream, source, check)
            stream.update({k:probe[k] for k in ("width","height")})
    formats = {(s["width"],s["height"]):s for s in formats.values()}
    title = video._clean(body.get("title"))
    if not title or " video #" in title:
        title = candidate.get("title", "")
    publisher = video._clean(body.get("uploader")) or candidate.get("publisher", "")
    description = video._clean(body.get("description")) or candidate.get("description", "")
    duration = video._duration(body.get("duration")) or video._duration(candidate.get("duration"))
    if not duration:
        best = max(formats.values(), key=video._short_edge)
        if best["format"] == "hls":
            manifest = video._hls(best["url"], check, source)
            duration = sum(s["duration"] for s in manifest["segments"])
        else:
            duration = video.probe_stream(best, source, check)["duration"]
    evidence = dict(candidate.get("source_evidence") or {})
    if body.get("description") and " video #" not in str(body.get("title", "")):
        evidence = video._evidence(source, title, description, publisher,
            candidate.get("publisher_url", ""), candidate.get("published_at", ""), "publisher_page")
    return candidate | {"title": title, "publisher": publisher, "description": description,
        "duration": duration, "media_streams": list(formats.values()), "media_extractor": "yt-dlp-domestic",
        "source_evidence": evidence, "source_verified": bool(evidence.get("readable"))}


def _bilibili_public(candidate: dict, check) -> dict:
    """Read one public BV's first part; search throttling is independent of playback."""
    source = candidate["source_url"]
    bvid = source.rstrip("/").rsplit("/", 1)[-1]
    headers = {"Referer": "https://www.bilibili.com/"}
    check()
    body = video._json("https://api.bilibili.com/x/web-interface/view?" + urlencode({"bvid": bvid}),
                       headers=headers)
    info = body.get("data") or {}
    if body.get("code") != 0 or info.get("bvid") != bvid:
        raise ValueError("B站视频信息不可读或与请求编号不一致")
    rights = info.get("rights") or {}
    if info.get("is_upower_exclusive") or rights.get("pay") or rights.get("is_stein_gate"):
        raise ValueError("此 B站视频需要额外权限或交互选择，已跳过")
    aid, cid = int(video._number(info.get("aid"))), int(video._number(info.get("cid")))
    pages = info.get("pages") or []
    if aid <= 0 or cid <= 0 or not pages or int(video._number(pages[0].get("cid"))) != cid:
        raise ValueError("B站视频分 P 编号不一致，未选择不明片段")
    check()
    body = video._json("https://api.bilibili.com/x/player/playurl?" + urlencode({
        "avid": aid, "cid": cid, "qn": 64, "fnval": 0, "fourk": 0}), headers=headers)
    playback = body.get("data") or {}
    if body.get("code") != 0 or playback.get("is_drm") or playback.get("is_preview"):
        raise ValueError("B站没有返回普通公开播放流")
    streams = playback.get("durl") or []
    if len(streams) != 1:
        raise ValueError("B站当前未返回支持的单文件公开流")
    url = video._http_url(streams[0].get("url"))
    domestic.require(url)
    stream = {"url": url, "format": "file", "container": "mp4"}
    # A posting's original dimension can differ from the anonymous playback
    # rendition. Probe actual bytes instead of relabelling a 480p stream as HD.
    probe = video.probe_stream(stream, source, check)
    stream.update({k: probe[k] for k in ("width", "height", "filesize")})
    owner = info.get("owner") or {}
    title, description = video._clean(info.get("title")), video._clean(info.get("desc"))
    publisher = video._clean(owner.get("name"))
    publisher_url = "https://space.bilibili.com/" + str(owner.get("mid", ""))
    published = time.strftime("%Y-%m-%d", time.gmtime(video._number(info.get("pubdate"))))
    evidence = video._evidence(source, title, description, publisher, publisher_url, published, "platform_metadata")
    evidence.update(video_id=bvid, aid=aid, cid=cid)
    return candidate | {"title": title, "description": description, "publisher": publisher,
        "publisher_url": publisher_url, "published_at": published, "duration": probe["duration"],
        "media_streams": [stream], "media_extractor": "bilibili-public", "source_verified": True,
        "source_evidence": evidence}
