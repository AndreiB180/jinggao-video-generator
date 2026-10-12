from __future__ import annotations

import json
import re
import time

import httpx
import pytest

from jinggao import config, media, paths, sources, video_sources as video


def test_video_only_reports_gap_without_photo_fallback(monkeypatch):
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})
    monkeypatch.setattr(video, "search_videos", lambda *a, **k: ([], ["公开视频未完成"]))
    monkeypatch.setattr(sources, "_legacy_search_shot", lambda *a: pytest.fail("video_only must not search photos"))
    rows, note = sources.search_shot({"media_policy": "video_only", "assets": []}, {"query": "城市街景"})
    assert rows == []
    assert "视频缺口" in note and "图片不能替代视频配额" in note


def test_person_query_filters_unrelated_recommendations_and_keeps_video_metadata(monkeypatch):
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "video"})
    calls = []

    def search(query, **kwargs):
        calls.append((query, kwargs))
        return [{"id": "correct", "kind": "video", "provider": "cctv", "title": "梁靖崑比赛", "duration": 30},
                {"id": "noise", "kind": "video", "provider": "youtube", "title": "gaming stream"}], []

    monkeypatch.setattr(video, "search_videos", search)
    rows, note = sources.search_shot({"media_policy": "video_preferred", "assets": [], "min_video_height": 720},
                                    {"query": "乒乓球", "persons": [{"name": "梁靖崑", "aliases": ["Liang Jingkun"]}]})
    assert [r["id"] for r in rows] == ["correct"]
    assert rows[0]["duration"] == 30 and rows[0]["min_video_height"] == 720
    assert all(not k["include_stock"] for _, k in calls)
    assert "梁靖崑" in [q for q, _ in calls]
    assert "1 个视频" in note


def test_local_video_keeps_publisher_evidence():
    evidence = {"url": "https://sports.cctv.com/item", "readable": True}
    rows = sources.local_candidates({"assets": [{"id": "asset", "title": "球员采访", "path": "assets/a.mp4",
                                                "kind": "video", "duration": 8, "width": 1280, "height": 720,
                                                "source_evidence": evidence, "publisher": "央视网"}]}, "球员")
    assert rows[0]["duration"] == 8 and rows[0]["source_evidence"] == evidence


def test_youtube_search_preserves_real_publisher_and_ignores_other_sections():
    item = {"videoId": "abcdefghijk", "title": {"runs": [{"text": "Liang Jingkun vs Kuo Guan-Hong"}]},
            "lengthText": {"simpleText": "9:09"},
            "ownerText": {"runs": [{"text": "World Table Tennis", "navigationEndpoint": {
                "commandMetadata": {"webCommandMetadata": {"url": "/@wttglobal"}}}}]},
            "ownerBadges": [{"metadataBadgeRenderer": {"style": "BADGE_STYLE_TYPE_VERIFIED"}}],
            "thumbnail": {"thumbnails": [{"url": "https://i.ytimg.com/frame.jpg", "width": 720, "height": 404}]}}
    body = {"contents": {"twoColumnSearchResultsRenderer": {"primaryContents": {
        "sectionListRenderer": {"contents": [{"videoRenderer": item}]}}}},
        "recommendations": [{"videoRenderer": item | {"videoId": "mnopqrstuvw"}}]}
    rows = video.parse_youtube_search("var ytInitialData = " + json.dumps(body) + ";")
    assert len(rows) == 1 and rows[0]["duration"] == 549
    assert rows[0]["publisher_url"] == "https://www.youtube.com/@wttglobal"
    assert rows[0]["height"] == 0  # Search thumbnail dimensions are not video dimensions.
    assert rows[0]["source_evidence"]["readable"] and rows[0]["publisher_verified"]


def test_stream_spec_without_url_is_not_downloadable(monkeypatch):
    body = {"playabilityStatus": {"status": "OK"}, "videoDetails": {"title": "Actual video", "lengthSeconds": "50"},
            "streamingData": {"adaptiveFormats": [{"height": 1080}], "serverAbrStreamingUrl": "https://example.test/stream"}}
    monkeypatch.setattr(video, "_fetch", lambda *a, **k: (("var ytInitialPlayerResponse = " + json.dumps(body)).encode(), {}))
    def unavailable(*args):
        raise ValueError("可选公开解析器未安装")
    monkeypatch.setattr(video, "_youtube_ytdlp", unavailable)
    row = video.inspect_video_candidate({"title": "video", "source_url": "https://www.youtube.com/watch?v=abcdefghijk"})
    assert row["download_status"] == "unsupported"
    assert row["height"] == 0 and row["available_format_heights"] == [1080]
    assert not row["media_streams"]


def _public_youtube_metadata():
    return {"id": "NOfcBrVMNGw", "title": "Liang Jingkun vs Kuo Guan-Hong | MS R32 | China Smash 2026",
            "description": "China Smash 2026 official match", "availability": "public", "duration": 549,
            "channel": "World Table Tennis", "channel_id": "UC9ckyA_A3MfXUa0ttxMoIZw",
            "channel_is_verified": True, "upload_date": "20261007", "is_live": False,
            "thumbnail": "https://i.ytimg.com/vi/NOfcBrVMNGw/hqdefault.jpg",
            "formats": [{"format_id": "134", "url": "https://rr1.googlevideo.com/videoplayback?itag=134",
                         "protocol": "https", "ext": "mp4", "vcodec": "avc1.4D401E",
                         "width": 640, "height": 360, "fps": 25, "has_drm": False},
                        {"format_id": "312", "url": "https://manifest.googlevideo.com/api/manifest.m3u8",
                         "protocol": "m3u8_native", "ext": "mp4", "vcodec": "avc1.64002A",
                         "width": 1920, "height": 1080, "fps": 50, "has_drm": False},
                        {"format_id": "299", "url": "https://rr1.googlevideo.com/videoplayback?itag=299",
                         "protocol": "https", "ext": "mp4", "vcodec": "avc1.64002A",
                         "width": 1920, "height": 1080, "fps": 50, "has_drm": False}]}


def test_public_youtube_fallback_preserves_evidence_and_uses_only_public_streams(monkeypatch):
    player = {"playabilityStatus": {"status": "OK"}, "videoDetails": {"title": "Match", "lengthSeconds": "549"},
              "streamingData": {"adaptiveFormats": [{"height": 1080}]}}
    monkeypatch.setattr(video, "_fetch", lambda *a, **k: (("var ytInitialPlayerResponse = " + json.dumps(player)).encode(), {}))
    monkeypatch.setattr(video, "_ytdlp_command", lambda: ["bundled/yt-dlp.exe"])
    monkeypatch.setattr(video, "_youtube_js_runtime", lambda: "deno:bundled/deno.exe")
    body = _public_youtube_metadata()
    body["formats"] += [body["formats"][-1] | {"format_id": "protected", "height": 2160, "has_drm": True},
                         body["formats"][-1] | {"format_id": "private-host", "url": "https://127.0.0.1/private"},
                         body["formats"][-1] | {"format_id": "audio-only", "vcodec": "none"}]
    calls = []

    def discover(args, check):
        calls.append(args)
        return body

    monkeypatch.setattr(video, "_run_ytdlp_json", discover)
    row = video.inspect_video_candidate({"title": "Match", "source_url": "https://www.youtube.com/watch?v=NOfcBrVMNGw"}, force=True)
    assert row["download_status"] == "available" and row["height"] == 1080
    assert row["media_extractor"] == "yt-dlp-public" and row["duration"] == 549
    assert row["source_evidence"]["url"] == row["source_url"]
    assert row["source_evidence"]["publisher"] == "World Table Tennis"
    assert row["source_evidence"]["evidence_origin"] == "publisher_page"
    assert row["published_at"] == "2026-10-07" and row["publisher_verified"]
    assert [s["format_id"] for s in row["media_streams"]] == ["134", "299"]
    assert video._choose_stream(row, preview=False)["format_id"] == "299"
    args = calls[0]
    for flag in ("--ignore-config", "--no-cache-dir", "--no-cookies", "--no-cookies-from-browser",
                 "--no-plugin-dirs", "--no-remote-components", "--skip-download", "--no-check-formats"):
        assert flag in args
    assert args[args.index("--proxy") + 1] == ""
    assert args[args.index("--extractor-retries") + 1] == "0"
    assert args[-2:] == ["--", row["source_url"]]
    video.inspect_video_candidate(row)
    assert len(calls) == 1  # Reuse the normal ten-minute metadata cache.


@pytest.mark.parametrize("change", [{"availability": "private"}, {"availability": "unlisted"},
                                     {"is_live": True}, {"has_drm": True}, {"age_limit": 18},
                                     {"id": "differentid"}, {"_type": "playlist", "entries": [{}]}])
def test_youtube_fallback_rejects_restricted_or_different_content(monkeypatch, change):
    monkeypatch.setattr(video, "_ytdlp_command", lambda: ["yt-dlp"])
    monkeypatch.setattr(video, "_youtube_js_runtime", lambda: "node:/runtime/node")
    monkeypatch.setattr(video, "_run_ytdlp_json", lambda *a: _public_youtube_metadata() | change)
    with pytest.raises(ValueError):
        video._youtube_ytdlp({"source_url": "https://youtu.be/NOfcBrVMNGw"}, lambda: None)


@pytest.mark.parametrize("status,formats", [("LOGIN_REQUIRED", []), ("UNPLAYABLE", []),
    ("OK", [{"width": 1920, "height": 1080, "drmFamilies": ["WIDEVINE"]}]),
    ("OK", [{"width": 1280, "height": 720, "url": "https://rr1.googlevideo.com/videoplayback?itag=298"}])])
def test_youtube_does_not_fallback_past_restrictions_or_repeat_work(monkeypatch, status, formats):
    player = {"playabilityStatus": {"status": status}, "videoDetails": {"title": "Match"},
              "streamingData": {"adaptiveFormats": formats}}
    monkeypatch.setattr(video, "_fetch", lambda *a, **k: (("var ytInitialPlayerResponse = " + json.dumps(player)).encode(), {}))
    monkeypatch.setattr(video, "_youtube_ytdlp", lambda *a: pytest.fail("must not invoke optional extractor"))
    row = video.inspect_video_candidate({"title": "Match", "source_url": "https://www.youtube.com/watch?v=NOfcBrVMNGw"}, force=True)
    assert bool(row["media_streams"]) == bool(formats and formats[0].get("url"))


def test_youtube_runtime_prefers_bundled_tools(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    suffix = ".exe" if video.os.name == "nt" else ""
    extractor = tmp_path / "runtime" / "yt-dlp" / ("yt-dlp" + suffix)
    deno = tmp_path / "runtime" / "deno" / ("deno" + suffix)
    for path in (extractor, deno):
        path.parent.mkdir(parents=True)
        path.touch()
    monkeypatch.setattr(video.shutil, "which", lambda *a: pytest.fail("bundled runtime must win"))
    assert video._ytdlp_command() == [str(extractor)]
    assert video._youtube_js_runtime() == "deno:" + str(deno)


def test_public_extractor_bounds_output_scrubs_environment_and_redacts_errors(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://should-not-be-used.invalid")
    monkeypatch.setenv("TEST_API_SECRET", "not-a-real-key")
    script = "import os,json; print(json.dumps({'proxy': os.getenv('HTTPS_PROXY'), 'secret': os.getenv('TEST_API_SECRET')}))"
    result = video._run_ytdlp_json([video.sys.executable, "-c", script], lambda: None)
    assert result == {"proxy": None, "secret": None}
    with pytest.raises(ValueError, match="读取上限"):
        video._run_ytdlp_json([video.sys.executable, "-c", "print('x'*9000)"], lambda: None, output_limit=100)
    with pytest.raises(ValueError) as error:
        video._run_ytdlp_json([video.sys.executable, "-c",
                               "import sys; print('https://example.test/?token=private-value',file=sys.stderr);sys.exit(1)"], lambda: None)
    assert "private-value" not in str(error.value) and "https://" not in str(error.value)


def test_public_extractor_cancellation_and_timeout_terminate_child(monkeypatch):
    children = []
    popen = video.subprocess.Popen

    def spawn(*a, **k):
        proc = popen(*a, **k)
        children.append(proc)
        return proc

    monkeypatch.setattr(video.subprocess, "Popen", spawn)
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise media.Cancelled()

    args = [video.sys.executable, "-c", "import time;time.sleep(30)"]
    with pytest.raises(media.Cancelled):
        video._run_ytdlp_json(args, cancel)
    assert children[0].poll() is not None
    with pytest.raises(ValueError, match="超时"):
        video._run_ytdlp_json(args, lambda: None, timeout=0.15)
    assert all(proc.poll() is not None for proc in children)


def test_hls_only_selects_overlapping_segments_and_rejects_protected_or_live():
    text = "#EXTM3U\n#EXTINF:10,\n0.ts\n#EXTINF:10,\n1.ts\n#EXTINF:10,\n2.ts\n#EXT-X-ENDLIST\n"
    parsed = video.parse_hls(text, "https://media.example/video/main.m3u8")
    assert [s["url"] for s in video.select_segments(parsed["segments"], 9, 3)] == [
        "https://media.example/video/0.ts", "https://media.example/video/1.ts"]
    assert len(video.select_segments(parsed["segments"], 10, 10)) == 1
    with pytest.raises(ValueError, match="不足"):
        video.select_segments(parsed["segments"], 29, 2)
    for unsafe in (text.replace("0.ts", "file:///etc/passwd"),
                   text.replace("#EXTINF:10,", '#EXT-X-KEY:METHOD=AES-128,URI="key"\n#EXTINF:10,', 1),
                   text.replace("#EXT-X-ENDLIST", "")):
        with pytest.raises(ValueError):
            video.parse_hls(unsafe, "https://media.example/video/main.m3u8")


def test_low_resolution_stream_is_only_preview(monkeypatch):
    monkeypatch.setattr(video, "_generic", lambda candidate, check: candidate | {
        "media_streams": [{"url": "https://media.example/low.mp4", "format": "file", "width": 480, "height": 270}]})
    row = video.inspect_video_candidate({"title": "clip", "url": "https://media.example/low.mp4", "min_video_height": 720})
    assert row["download_status"] == "preview_only" and row["height"] == 270
    with pytest.raises(ValueError, match="720p"):
        video._choose_stream(row, preview=False)
    assert video._choose_stream(row, preview=True)["height"] == 270


def test_metadata_fetch_caps_actual_body_and_stops_private_redirect(monkeypatch, tmp_path):
    client = httpx.Client
    monkeypatch.setattr(sources, "download_slot", lambda url: tmp_path / "cooldown.json")
    visited = []

    def public(url):
        visited.append(url)
        if "127.0.0.1" in url:
            raise ValueError("内网")

    monkeypatch.setattr(sources, "public_url", public)
    monkeypatch.setattr(video.network, "client", lambda **kw: client(transport=httpx.MockTransport(
        lambda r: httpx.Response(302, headers={"Location": "http://127.0.0.1/private"})), **kw))
    with pytest.raises(ValueError, match="内网"):
        video._fetch("https://media.example/open")
    assert visited == ["https://media.example/open", "http://127.0.0.1/private"]
    monkeypatch.setattr(video.network, "client", lambda **kw: client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, content=b"x" * 9000)), **kw))
    with pytest.raises(ValueError, match="预算"):
        video._fetch("https://media.example/large", limit=1000)


def test_real_ffmpeg_range_preview_keeps_source_timing_and_budget(monkeypatch, tmp_path):
    original = tmp_path / "source.mp4"
    media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
               "testsrc2=size=1280x720:rate=10", "-t", "4", "-c:v", "libx264", "-preset", "ultrafast",
               "-movflags", "+faststart", str(original)], timeout=30)
    content = original.read_bytes()
    requests = []
    client = httpx.Client

    def respond(request):
        requests.append(request.headers.get("range", ""))
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", request.headers.get("range", ""))
        start = int(match[1]) if match else 0
        end = min(len(content) - 1, int(match[2]) if match and match[2] else len(content) - 1)
        return httpx.Response(206 if match else 200, content=content[start:end+1], headers={
            "Content-Type": "video/mp4", "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{len(content)}"})

    monkeypatch.setattr(sources, "public_url", lambda url: None)
    monkeypatch.setattr(sources, "download_slot", lambda url: tmp_path / "cooldown.json")
    monkeypatch.setattr(video.network, "client", lambda **kw: client(transport=httpx.MockTransport(respond), **kw))
    candidate = {"metadata_checked": True, "inspection_at": time.time(), "min_video_height": 720,
                 "source_url": "https://media.example/page", "duration": 4,
                 "media_streams": [{"url": "https://media.example/source.mp4", "format": "file",
                                    "width": 1280, "height": 720}]}
    out = tmp_path / "preview.mp4"
    result = video.preview_video_candidate(candidate, out, 1.25, 1.5)
    assert result["source_clip_start"] == 1.25 and result["clip_start"] == 0
    assert result["download_mode"] == "http_range_bounded" and any(requests)
    assert result["width"] == 640 and abs(result["duration"] - 1.5) < 0.12
    assert result["downloaded_bytes"] > 0 and out.exists()
    monkeypatch.setattr(video, "PREVIEW_LIMIT", 1024)
    with pytest.raises(ValueError, match="预算"):
        video.preview_video_candidate(candidate, tmp_path / "over-limit.mp4", 1.25, 1.5)
    assert not (tmp_path / "over-limit.mp4").exists()


def test_authorized_api_request_never_forwards_key_across_redirect(monkeypatch, tmp_path):
    client = httpx.Client
    calls = []

    def reply(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://different.example/collect"})

    monkeypatch.setattr(sources, "public_url", lambda url: None)
    monkeypatch.setattr(sources, "download_slot", lambda url: tmp_path / "cooldown.json")
    monkeypatch.setattr(video.network, "client", lambda **kw: client(transport=httpx.MockTransport(reply), **kw))
    with pytest.raises(ValueError, match="跨域重定向"):
        video._fetch("https://api.example/data", headers={"Authorization": "test-only-placeholder"})
    assert calls == ["https://api.example/data"]


def test_range_proxy_honors_retry_after_across_attempts(monkeypatch, tmp_path):
    import urllib.error
    import urllib.request
    client = httpx.Client
    calls = []

    def reply(request):
        calls.append(str(request.url))
        return httpx.Response(429, headers={"Retry-After": "600"})

    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(sources, "public_url", lambda url: None)
    monkeypatch.setattr(video.network, "client", lambda **kw: client(transport=httpx.MockTransport(reply), **kw))
    for _ in range(2):
        with video._range_proxy("https://media.example/movie.mp4", 10000, lambda: None, "") as (url, state):
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(url, timeout=5)
            assert "429" in state["error"]
    assert len(calls) == 1


def test_real_hls_download_fetches_only_the_requested_time_window(monkeypatch, tmp_path):
    media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
               "testsrc2=size=1280x720:rate=10", "-t", "6", "-c:v", "libx264", "-preset", "ultrafast",
               "-g", "20", "-sc_threshold", "0", "-f", "hls", "-hls_time", "2", "-hls_list_size", "0",
               "-hls_segment_filename", str(tmp_path / "%d.ts"), str(tmp_path / "index.m3u8")], timeout=30)
    client = httpx.Client
    requested = []

    def reply(request):
        name = request.url.path.rsplit("/", 1)[-1]
        requested.append(name)
        return httpx.Response(200, content=(tmp_path / name).read_bytes())

    monkeypatch.setattr(sources, "public_url", lambda url: None)
    monkeypatch.setattr(sources, "download_slot", lambda url: tmp_path / "cooldown.json")
    monkeypatch.setattr(video.network, "client", lambda **kw: client(transport=httpx.MockTransport(reply), **kw))
    candidate = {"metadata_checked": True, "inspection_at": time.time(), "min_video_height": 720,
                 "source_url": "https://media.example/article", "duration": 6,
                 "media_streams": [{"url": "https://media.example/index.m3u8", "format": "hls", "width": 1280, "height": 720}]}
    result = video.download_video_candidate(candidate, tmp_path / "clip.mp4", 2.3, 1)
    assert requested == ["index.m3u8", "1.ts"]
    assert result["download_mode"] == "hls_selected_segments" and result["source_clip_start"] == 2.3
    assert result["height"] == 720 and abs(result["duration"] - 1) < 0.1
    assert result["downloaded_bytes"] == (tmp_path / "1.ts").stat().st_size


def test_successful_inspection_reuses_metadata_but_force_refreshes(monkeypatch):
    calls = []

    def inspect(candidate, check):
        calls.append(1)
        return candidate | {"media_streams": [{"url": "https://media.example/a.mp4", "format": "file", "width": 1280, "height": 720}]}

    monkeypatch.setattr(video, "_generic", inspect)
    row = video.inspect_video_candidate({"url": "https://media.example/a.mp4", "title": "clip"})
    assert video.inspect_video_candidate(row)["download_status"] == "available"
    assert len(calls) == 1
    assert video.inspect_video_candidate(row | {"min_video_height": 1080})["download_status"] == "preview_only"
    video.inspect_video_candidate(row, force=True)
    assert len(calls) == 2


def test_archive_preview_and_final_stay_with_the_same_original_movie(monkeypatch):
    body = {"metadata": {}, "files": [
        {"name": "first.mp4", "width": "1280", "height": "720", "length": "25"},
        {"name": "first-low.mp4", "original": "first.mp4", "width": "640", "height": "360", "length": "25"},
        {"name": "second.mp4", "width": "1920", "height": "1080", "length": "15"},
        {"name": "second-low.mp4", "original": "second.mp4", "width": "320", "height": "180", "length": "15"}]}
    monkeypatch.setattr(video, "_json", lambda *a, **k: body)
    row = video.inspect_video_candidate({"provider": "archive", "title": "Two unrelated movies",
                                         "source_url": "https://archive.org/details/example"})
    assert row["duration"] == 15
    assert video._choose_stream(row, preview=True)["media_family"] == "second.mp4"
    assert video._choose_stream(row, preview=False)["media_family"] == "second.mp4"


def test_schema_video_content_url_does_not_recursively_refetch_its_page(monkeypatch):
    data = '<script type="application/ld+json">' + json.dumps({"@type": "VideoObject", "name": "clip",
                                                           "contentUrl": "https://media.example/page"}) + '</script>'
    calls = []

    def fetch(*args, **kwargs):
        calls.append(1)
        return data.encode(), {}

    monkeypatch.setattr(video, "_fetch", fetch)
    row = video.inspect_video_candidate({"source_url": "https://media.example/page", "title": "clip"})
    assert row["download_status"] == "blocked" and len(calls) == 1


def test_person_search_accepts_complete_punctuation_variants_not_surnames(monkeypatch):
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "video"})
    rows = [{"id": name, "kind": "video", "provider": "youtube", "title": title} for name, title in (
        ("hyphen", "Kuo Guan-Hong match"), ("space", "Interview with KUO GUAN HONG"),
        ("surname", "Kuo interview"), ("partial", "Kuo Guan Honge training"),
        ("different", "Kuo Guan-Min table tennis"))]
    monkeypatch.setattr(video, "search_videos", lambda *a, **k: (rows, []))
    result, _ = sources.search_shot({"media_policy": "video_only", "assets": []},
                                   {"query": "郭冠宏", "persons": [{"name": "郭冠宏", "aliases": ["Kuo Guan Hong", "Kuo"]}]})
    assert {r["id"] for r in result} == {"hyphen", "space"}


def test_person_channels_are_domestic_and_never_fall_back_overseas(monkeypatch, tmp_path):
    from jinggao import baidu_videos, platform_videos, portal_videos, publisher_videos
    monkeypatch.setattr(video.paths, "DATA", tmp_path)
    monkeypatch.setattr(video.network, "client", lambda *a, **kw: pytest.fail("unmocked live provider request"))
    names = []
    for module, name in ((video, "cctv"), (platform_videos, "tencent"),
                         (platform_videos, "bilibili"), (publisher_videos, "chinanews"),
                         (portal_videos, "iqiyi"), (portal_videos, "sohu"), (portal_videos, "youku"),
                         (baidu_videos, "baidu_video")):
        def search(query, name=name):
            names.append(name)
            return [{"id": name, "provider": name, "kind": "video"}]
        monkeypatch.setattr(module, name + "_search", search)
    for name in ("youtube", "dailymotion", "archive", "commons_video", "nasa", "pexels", "pixabay_video"):
        monkeypatch.setattr(video, name + "_search", lambda *a: pytest.fail("overseas source called"))
    rows, errors = video.search_videos("郭冠宏", include_stock=False, pexels_key="test-placeholder")
    assert not errors
    assert set(names) == {"cctv", "tencent", "bilibili", "chinanews", "iqiyi", "sohu", "youku", "baidu_video"}
    assert len(rows) == 8


def test_generic_page_thumbnail_is_a_cover_not_video_resolution(monkeypatch):
    data = '<meta property="og:image" content="/covers/preview.jpg"><meta property="og:image:width" content="1920">'
    data += '<script type="application/ld+json">' + json.dumps({"@type": "VideoObject", "name": "clip",
        "contentUrl": "https://media.example/clip.mp4", "duration": "PT30S"}) + '</script>'
    monkeypatch.setattr(video, "_fetch", lambda *a, **k: (data.encode(), {}))
    row = video.inspect_video_candidate({"source_url": "https://media.example/watch", "title": "clip"})
    assert row["thumbnail_url"] == "https://media.example/covers/preview.jpg"
    assert row["preview_urls"] == [{"url": row["thumbnail_url"], "at": None, "kind": "publisher_thumbnail"}]
    assert row["width"] == 0 and row["height"] == 0


def test_old_foreign_keys_do_not_enable_overseas_video_sources(monkeypatch):
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "video", "pixabay_api_key": "test-placeholder"})
    calls = []
    def search(query, **kwargs):
        calls.append(query)
        return [{"id": "video", "provider": "tencent", "kind": "video", "title": "city street"}], []
    monkeypatch.setattr(video, "search_videos", search)
    monkeypatch.setattr(sources, "pixabay", lambda *a, **kw: pytest.fail("overseas source called"))
    rows, _ = sources.search_shot({"media_policy": "video_only", "assets": []}, {"query": "city"})
    assert [r["id"] for r in rows] == ["video"]
    assert calls == ["city"]
    rows, note = sources.search_shot({"media_policy": "video_preferred", "assets": []},
                                    {"query": "city", "persons": [{"name": "郭冠宏", "aliases": ["Kuo Guan-Hong"]}]})
    assert rows == [] and "视频缺口" in note
    assert calls.count("city") == 2


def test_pixabay_video_endpoint_cache_and_renditions(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    calls = []
    hit = {"pageURL": "https://pixabay.com/videos/id-125/", "tags": "city, street", "duration": 12,
           "user": "Example Author", "videos": {
               "large": {"url": "https://cdn.pixabay.com/large.mp4", "width": 1920, "height": 1080, "size": 9000},
               "medium": {"url": "https://cdn.pixabay.com/medium.mp4", "width": 1280, "height": 720,
                          "size": 6000, "thumbnail": "https://cdn.pixabay.com/medium.jpg"},
               "tiny": {"url": "https://cdn.pixabay.com/tiny.mp4", "width": 640, "height": 360, "size": 2000}}}

    def get(url, **kwargs):
        calls.append(url)
        return httpx.Response(200, json={"hits": [hit]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(sources.httpx, "get", get)
    rows = video.pixabay_video_search("city", "test-placeholder")
    assert video.pixabay_video_search("city", "test-placeholder") == rows
    assert calls == ["https://pixabay.com/api/videos/"]
    row = video.inspect_video_candidate(rows[0])
    assert row["duration"] == 12 and row["source_evidence"]["evidence_origin"] == "platform_metadata"
    assert row["source_url"] == hit["pageURL"] and row["thumbnail_url"] == hit["videos"]["medium"]["thumbnail"]
    assert video._choose_stream(row, preview=True)["height"] == 360
    assert video._choose_stream(row, preview=False)["height"] == 1080


def test_pixabay_429_prevents_further_query_requests_until_reset(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return httpx.Response(429, headers={"X-RateLimit-Reset": "120"}, request=httpx.Request("GET", url))

    monkeypatch.setattr(sources.httpx, "get", get)
    for query in ("city", "cooking"):
        with pytest.raises(ValueError, match="限流"):
            video.pixabay_video_search(query, "test-placeholder")
    assert len(calls) == 1
