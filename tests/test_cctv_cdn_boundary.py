import socket

import httpx
import pytest

from jinggao import domestic, paths, sources, video_sources as video


@pytest.mark.parametrize("url", [
    "https://hls.cntv.cdn20.com/asp/hls/main/video/main.m3u8?maxbr=2048",
    "http://HLS.CNTV.CDN20.COM./asp/hls/main/video/main.m3u8",
])
def test_cctv_official_hls_host_is_allowed(url):
    domestic.require(url)


@pytest.mark.parametrize("host", [
    "cdn20.com",
    "other.cdn20.com",
    "cntv.cdn20.com",
    "other.cntv.cdn20.com",
    "sub.hls.cntv.cdn20.com",
    "evilhls.cntv.cdn20.com",
    "hls.cntv.cdn20.com.evil.example",
])
def test_cctv_shared_cdn_exception_does_not_allow_other_hosts(host):
    with pytest.raises(ValueError, match="已接入"):
        domestic.require(f"https://{host}/main.m3u8")


def test_cctv_allowed_host_reports_upstream_403_and_stops(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(sources.socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
    ])
    calls = []

    def respond(request):
        calls.append(str(request.url))
        return httpx.Response(403, text="Forbidden")

    def client(**kwargs):
        return httpx.Client(**(kwargs | {"transport": httpx.MockTransport(respond)}))

    monkeypatch.setattr(video.network, "client", client)
    url = "https://hls.cntv.cdn20.com/asp/hls/main/video/main.m3u8?maxbr=2048"
    with pytest.raises(ValueError, match=r"HTTP 403"):
        video._fetch(url, limit=1024)
    assert calls == [url]
