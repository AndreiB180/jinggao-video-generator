import json

import pytest

from jinggao import bilibili_web_search, domestic, paths, platform_videos as platform, publisher_videos, video_sources as video


def search_item(vid="abcdefghijk", pay=""):
    return {"videoInfo": {"title": "<em>马龙</em>乒乓球比赛", "descrip": "比赛集锦",
        "imgUrl": "http://puui.qpic.cn/preview.jpg", "checkupTime": "2026-10-08",
        "videoDoc": {"dataKey": "vid="+vid, "uploader": "发布者", "timeLong": 180, "payStatus": pay}}}


def test_tencent_search_retains_single_video_source_and_excludes_paid_and_series():
    body = {"ret": 0, "data": {"errcode": 0, "normalList": {"itemList": [search_item(), search_item(),
        search_item("otheritemid", "paid"), {"videoInfo": {"title": "付费剧集", "coverDoc": {}}}]}}}
    rows = platform.parse_tencent_search(body)
    assert len(rows) == 1
    row = rows[0]
    assert row["source_url"] == "https://v.qq.com/x/page/abcdefghijk.html"
    assert row["title"] == "马龙乒乓球比赛" and row["duration"] == 180
    assert row["publisher"] == "发布者" and not row["publisher_verified"]
    assert row["thumbnail_url"].startswith("https://puui.qpic.cn/")
    assert row["height"] == 0  # A cover is not a decoded video.
    assert row["source_evidence"]["evidence_origin"] == "publisher_search"


def test_platform_extractor_rejects_drm_foreign_and_duplicate_cdn_formats(monkeypatch):
    monkeypatch.setattr(video, "_ytdlp_command", lambda: ["yt-dlp"])
    source = "https://v.qq.com/x/page/abcdefghijk.html"
    def read(args, check, **kwargs):
        assert "vqq:video" in args and "--no-geo-bypass" in args and args[-1] == source
        assert "--no-cookies-from-browser" in args
        base = {"width": 1280, "height": 720, "ext": "mp4", "protocol": "m3u8_native"}
        return {"id": "abcdefghijk", "title": "vqq-video video #abcdefghijk", "formats": [
            base | {"url": "https://ltsyz.gtimg.com/first.m3u8"},
            base | {"url": "https://ltsyz.gtimg.com/mirror.m3u8"},
            base | {"url": "https://rr.googlevideo.com/blocked.m3u8"},
            base | {"width": 1920, "height": 1080, "has_drm": True, "url": "https://ltsyz.gtimg.com/drm.m3u8"}]}
    monkeypatch.setattr(video, "_run_ytdlp_json", read)
    evidence = {"readable": True, "evidence_origin": "publisher_search"}
    row = platform.inspect_platform({"source_url": source, "title": "原搜索标题", "duration": 180,
                                     "source_evidence": evidence}, lambda: None)
    assert row["title"] == "原搜索标题" and row["source_evidence"] == evidence
    assert len(row["media_streams"]) == 1 and "first.m3u8" in row["media_streams"][0]["url"]


def test_bili_challenge_stops_repeated_queries_for_other_shots(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(bilibili_web_search, "blocked_until", 0)
    calls = []
    def read(*a, **kw):
        calls.append(a[0])
        raise bilibili_web_search.PublicSearchBlocked("B站普通匿名网页要求验证")
    monkeypatch.setattr(bilibili_web_search, "read_public_page", read)
    for query in ("人物比赛", "番茄种植", "城市航拍"):
        with pytest.raises(ValueError):
            platform.bilibili_search(query)
    assert len(calls) == 1


def test_source_boundary_rejects_unintegrated_sources_and_lookalike_domains():
    for url in ("https://v.qq.com/x/page/a.html", "https://i0.hdslb.com/cover.jpg",
                "https://poss-videocloud.cns.com.cn/clip.mp4"):
        assert domestic.allowed(url)
    for url in ("https://youtube.com/watch?v=x", "https://duckduckgo.com/", "https://bing.com/",
                "https://qq.com.evil.example/a", "https://evilqq.com/a"):
        with pytest.raises(ValueError, match="已接入"):
            domestic.require(url)


def test_chinanews_video_search_never_counts_related_non_video_articles():
    data = [{"url": "http://www.chinanews.com.cn/ty/shipin/cns-d/2026/10-08/news1.shtml",
             "title": ["<em>球员</em>训练"], "thumbnail_file": "https://i2.chinanews.com.cn/a.jpg"},
            {"url": "https://www.chinanews.com.cn/ty/article.shtml", "title": "报道"}]
    rows = publisher_videos.parse_chinanews_search("var docArr = " + json.dumps(data) + ";")
    assert len(rows) == 1 and rows[0]["source_url"].startswith("https://")
    assert rows[0]["provider"] == "chinanews" and rows[0]["title"] == "球员训练"


def test_chinanews_legacy_and_current_vinfo_are_bound_to_main_player(monkeypatch):
    source = "https://www.chinanews.com.cn/shipin/2026/03-22/news1049605.shtml"
    stream = "http://v-oss.cnsimg.net/cspimp/2026/03/22/main.mp4"
    markup = ('so.addVariable("vInfo", "'+stream+'");'
              'adVideoUrl="https://v-oss.cnsimg.net/ad.mp4";'
              'var recommendations=["https://v-oss.cnsimg.net/other.mp4"];')
    monkeypatch.setattr(video, "_fetch", lambda *a, **kw: (markup.encode(), {}))
    monkeypatch.setattr(video, "probe_stream", lambda *a: {"width": 1080, "height": 1920, "duration": 50, "filesize": 10000})
    result = publisher_videos.chinanews({"source_url": source, "title": "主视频"}, lambda: None)
    assert [s["url"] for s in result["media_streams"]] == [stream.replace("http:", "https:")]
    assert result["duration"] == 50 and result["source_evidence"]["url"] == source


def bili_fixture():
    return {"bvid": "BV1oZTszNEJw", "aid": 114630059886031, "cid": 30333472461,
        "pages": [{"cid": 30333472461}], "title": "公开投稿", "desc": "比赛说明",
        "dimension": {"width": 1920, "height": 1080}, "rights": {"pay": 0},
        "owner": {"mid": 123, "name": "发布者"}, "pubdate": 1749116172}


def test_bili_public_binds_first_part_provenance_and_probes_real_resolution(monkeypatch):
    from jinggao.identity import SOURCE_ORIGINS
    calls = []
    def read(url, **kw):
        calls.append(url)
        if "/view?" in url:
            return {"code": 0, "data": bili_fixture()}
        assert "avid=114630059886031" in url and "cid=30333472461" in url
        return {"code": 0, "data": {"durl": [{"url": "https://upos.bilivideo.com/video.mp4"}]}}
    monkeypatch.setattr(video, "_json", read)
    monkeypatch.setattr(video, "probe_stream", lambda *a: {
        "width": 854, "height": 480, "duration": 128.366, "filesize": 10000})
    monkeypatch.setattr(video, "_ytdlp_command", lambda: pytest.fail("Public BV should not need a browser/extractor"))
    row = platform.inspect_platform({"source_url": "https://www.bilibili.com/video/BV1oZTszNEJw/"}, lambda: None)
    assert len(calls) == 2 and row["media_streams"][0]["height"] == 480
    assert row["source_evidence"]["cid"] == 30333472461
    assert row["source_evidence"]["evidence_origin"] in SOURCE_ORIGINS
    assert row["duration"] == 128.366 and not row.get("quality_verified")


@pytest.mark.parametrize("change", [
    {"bvid": "BV1xxxxxxxxx"}, {"pages": [{"cid": 123}]}, {"rights": {"pay": 1}},
    {"is_upower_exclusive": True},
])
def test_bili_rejects_wrong_source_part_or_restricted_content_before_playback(monkeypatch, change):
    calls = []
    def read(url, **kw):
        calls.append(url)
        return {"code": 0, "data": bili_fixture() | change}
    monkeypatch.setattr(video, "_json", read)
    with pytest.raises(ValueError):
        platform.inspect_platform({"source_url": "https://www.bilibili.com/video/BV1oZTszNEJw"}, lambda: None)
    assert len(calls) == 1


@pytest.mark.parametrize("playback", [
    {"is_drm": True}, {"is_preview": True},
    {"durl": [{"url": "https://foreign.example/video.mp4"}]},
])
def test_bili_does_not_probe_protected_or_foreign_playback(monkeypatch, playback):
    def read(url, **kw):
        return {"code": 0, "data": bili_fixture() if "/view?" in url else playback}
    monkeypatch.setattr(video, "_json", read)
    monkeypatch.setattr(video, "probe_stream", lambda *a: pytest.fail("Rejected playback must not be fetched"))
    with pytest.raises(ValueError):
        platform.inspect_platform({"source_url": "https://www.bilibili.com/video/BV1oZTszNEJw"}, lambda: None)
