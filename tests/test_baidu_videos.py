import json
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from jinggao import baidu_videos as baidu, network, paths, sources


def card(source="http://www.bilibili.com/video/BV13H4y1s72y", title="公开<em>演讲</em>",
         thumb="https://gimg4.baidu.com/poster/example.jpg", duration="11:02"):
    return f'''<!-- {source} -->
      <div class="video_list video_short" srcid="4185">
       <a href="http://www.baidu.com/link?url=opaque"><div class="c-img">
        <img src="{thumb}"><span class="video_play_timer">{duration}</span></div></a>
       <div class="video_small_intro"><a class="video-title c-link"
         href="http://www.baidu.com/link?url=opaque">{title}</a>
        <div class="c-color-text c-font-normal">简介：原标题可供检索</div>
        <div><span class="wetSource c-font-normal">来源：哔哩哔哩</span>
         <span>发布时间：2024-03-17</span></div></div>
      </div>'''


def page(*cards):
    return '<html><title>普通查询_百度视频搜索</title><div class="video_list_container">' + ''.join(cards) + '</div></html>'


def test_real_html_card_links_are_discovery_not_publisher_proof():
    url = baidu.SEARCH_URL + '?pd=video&wd=公开演讲'
    row, = baidu.parse_baidu_video_search(page(card()), url)
    assert row['source_url'] == 'https://www.bilibili.com/video/BV13H4y1s72y'
    assert row['title'] == '公开演讲' and row['duration'] == 662
    assert row['provider'] == 'baidu_video' and row['platform_provider'] == 'bilibili'
    assert row['video_id'] == 'BV13H4y1s72y'
    assert row['thumbnail_url'] == 'https://gimg4.baidu.com/poster/example.jpg'
    assert not row['source_verified'] and not row['publisher_verified']
    assert not row['publisher'] and not row['publisher_url'] and not row['published_at']
    assert row['source_evidence'] == {'url':row['source_url'], 'readable':False,
                                       'evidence_origin':'unread_original_page'}
    assert row['discovery_evidence']['url'] == url
    assert row['discovery_evidence']['evidence_origin'] == 'aggregator_search'
    assert row['discovery_evidence']['published_at'] == '2024-03-17'
    assert row['discovery_evidence']['source_label'] == '来源：哔哩哔哩'
    assert row['download_status'] == 'unchecked' and not row['metadata_checked']
    assert row['width'] == row['height'] == 0


@pytest.mark.parametrize('source',[
    'https://www.youtube.com/watch?v=abcdefghijk',
    'https://www.bilibili.com.evil.test/video/BV13H4y1s72y',
    'https://www.baidu.com/link?url=opaque',
    'https://www.baidu.com/sf/vsearch?pd=video&wd=视频',
    'https://www.bilibili.com/video/BV13H4y1s72y?p=2',
    'https://user:password@www.bilibili.com/video/BV13H4y1s72y',
    'https://www.bilibili.com:invalid/video/BV13H4y1s72y',
    'http://127.0.0.1/video/BV13H4y1s72y',
    'https://haokan.baidu.com/web/search/page?query=视频',
    'https://download.example.cn/video.mp4',
    'https://www.iqiyi.com/a_collection.html',
    'https://v.qq.com/x/cover/collection.html',
    'https://space.bilibili.com/123456',
    'https://www.chinanews.com.cn/one/two/shipin/cns-d/2026/01-01/news123.shtml',
])
def test_cross_border_redirects_accounts_downloads_and_collections_are_ignored(source):
    assert baidu.parse_baidu_video_search(page(card(source=source))) == []


def test_card_source_cannot_leak_from_other_elements_or_previous_result():
    stale = '<!-- https://www.bilibili.com/video/BV13H4y1s72y --><p>unrelated card</p>'
    orphan = card().split('-->', 1)[1]
    assert baidu.parse_baidu_video_search(page(stale + orphan)) == []
    rows = baidu.parse_baidu_video_search(page(card(), orphan, card()))
    assert len(rows) == 1


@pytest.mark.parametrize('thumb',[
    'https://foreign.example/cover.jpg', 'file:///tmp/cover.png',
    'https://gimg4.baidu.com.evil.test/cover.jpg', '',
])
def test_untrusted_or_missing_thumbnail_is_not_downloaded(thumb):
    assert baidu.parse_baidu_video_search(page(card(thumb=thumb))) == []


@pytest.mark.parametrize(('source', 'expected', 'platform'), [
    ('http://v.qq.com/boke/page/d/0/j/d0184vz0d7j.html#vfrm=2-3-0-1',
     'https://v.qq.com/x/page/d0184vz0d7j.html','tencent'),
    ('https://v.youku.com/v_show/id_Xabcd==.html?spm=source',
     'https://v.youku.com/v_show/id_Xabcd==.html', 'youku'),
    ('https://www.iqiyi.com/v_u4dmtp3hxk.html?src=web',
     'https://www.iqiyi.com/v_u4dmtp3hxk.html','iqiyi'),
    ('https://tv.cctv.com/2025/01/02/VIDEabcd0123.shtml?from=search',
     'https://tv.cctv.com/2025/01/02/VIDEabcd0123.shtml','cctv'),
    ('https://www.chinanews.com.cn/shipin/cns-d/2026/01-01/news123.shtml',
     'https://www.chinanews.com.cn/shipin/cns-d/2026/01-01/news123.shtml','chinanews'),
    ('https://www.chinanews.com.cn/life/shipin/cns-d/2026/01-01/news123.shtml',
     'https://www.chinanews.com.cn/life/shipin/cns-d/2026/01-01/news123.shtml','chinanews'),
])
def test_only_known_individual_video_paths_are_canonicalized(source, expected, platform):
    row, = baidu.parse_baidu_video_search(page(card(source=source)))
    assert row['source_url'] == expected and row['platform_provider'] == platform


@pytest.mark.parametrize('html',[
    '<html><title>百度安全验证</title><body>请完成验证</body></html>',
    '<html><script src="https://wappass.baidu.com/static/captcha/tuxing.js"></script></html>',
    '{"antiFlag":1,"message":"Forbid spider access"}',
])
def test_challenge_page_never_becomes_empty_success(html):
    with pytest.raises(ValueError, match='验证'):
        baidu.parse_baidu_video_search(html)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'DATA', tmp_path)
    monkeypatch.setattr(baidu, 'REQUEST_INTERVAL', 0)
    monkeypatch.setattr(baidu, '_last_request', {})
    return tmp_path


def test_success_cache_and_provider_stop_are_distinct_and_persistent(isolated, monkeypatch):
    calls = []
    def fetch(url, check):
        calls.append(url)
        if len(calls) > 1:
            raise ValueError('百度视频要求安全验证')
        return page(card())
    monkeypatch.setattr(baidu, '_fetch_html', fetch)
    good = baidu.baidu_video_search('普通 演讲')
    assert baidu.baidu_video_search(' 普通   演讲 ') == good
    with pytest.raises(ValueError, match='安全验证'):
        baidu.baidu_video_search('第二个查询')
    assert len(calls) == 2
    assert json.loads(baidu._stop_path().read_text())['until'] > baidu.time.time()
    monkeypatch.setattr(baidu, '_last_request', {})  # simulated process restart
    with pytest.raises(ValueError, match='冷却'):
        baidu.baidu_video_search('第三个查询')
    assert baidu.baidu_video_search('普通 演讲') == good
    assert len(calls) == 2


def test_pages_are_cached_separately_bounded_and_have_only_public_query_parameters(isolated, monkeypatch):
    calls = []
    monkeypatch.setattr(baidu, '_fetch_html', lambda url, check: calls.append(url) or page(card()))
    baidu.baidu_video_search('厨房 收纳', 1)
    baidu.baidu_video_search('厨房 收纳', 2)
    baidu.baidu_video_search('厨房 收纳', 2)
    assert len(calls) == 2
    assert parse_qs(urlsplit(calls[0]).query) == {'pd':['video'], 'wd':['厨房 收纳']}
    assert parse_qs(urlsplit(calls[1]).query) == {'pd':['video'], 'wd':['厨房 收纳'], 'pn':['10']}
    assert baidu.baidu_video_search('厨房 收纳', baidu.MAX_PAGE + 1) == []
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            baidu.baidu_video_search('厨房 收纳', invalid)
    assert len(calls) == 2


def mocked_client(monkeypatch, handler):
    seen_options = []
    def client(**kwargs):
        seen_options.append(kwargs)
        return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(network, 'client', client)
    monkeypatch.setattr(sources, 'public_url', lambda url: None)
    return seen_options


@pytest.mark.parametrize('target',[
    'https://wappass.baidu.com/static/captcha/tuxing_v2.html?challenge=not-followed',
    'https://foreign.example/search', 'http://www.baidu.com/sf/vsearch',
    'https://www.baidu.com/link?url=not-followed',
])
def test_public_request_never_follows_verification_or_other_entrypoints(isolated, monkeypatch, target):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={'Location': target})
    options = mocked_client(monkeypatch, handler)
    with pytest.raises(ValueError, match='停止'):
        baidu.baidu_video_search('公开查询')
    assert len(requests) == 1
    assert options[0]['trust_env'] is False and options[0]['follow_redirects'] is False
    assert not any(k.lower() in ('cookie', 'authorization') for k in requests[0].headers)
    with pytest.raises(ValueError, match='冷却'):
        baidu.baidu_video_search('另一个查询')
    assert len(requests) == 1


def test_http_429_stops_the_lane_without_followups(isolated, monkeypatch):
    requests = []
    mocked_client(monkeypatch, lambda request: requests.append(request) or httpx.Response(429))
    with pytest.raises(ValueError, match='429'):
        baidu.baidu_video_search('公开查询')
    with pytest.raises(ValueError, match='冷却'):
        baidu.baidu_video_search('其他查询')
    assert len(requests) == 1


def test_declared_and_streamed_size_are_both_bounded(isolated, monkeypatch):
    monkeypatch.setattr(baidu, 'MAX_BYTES', 100)
    mocked_client(monkeypatch, lambda request: httpx.Response(200, headers={'Content-Length':'101'}))
    with pytest.raises(ValueError, match='预算'):
        baidu._fetch_html(baidu.SEARCH_URL, lambda: None)
    class Body(httpx.SyncByteStream):
        def __iter__(self):
            yield b'x' * 80
            yield b'x' * 21
    mocked_client(monkeypatch, lambda request: httpx.Response(200, stream=Body()))
    with pytest.raises(ValueError, match='预算'):
        baidu._fetch_html(baidu.SEARCH_URL, lambda: None)


def test_response_time_budget_is_enforced_without_retry(isolated, monkeypatch):
    clock = [100.0]
    requests = []
    monkeypatch.setattr(baidu.time, 'monotonic', lambda: clock[0])
    def handler(request):
        requests.append(request)
        clock[0] += baidu.REQUEST_BUDGET + 1
        return httpx.Response(200, text=page(card()))
    mocked_client(monkeypatch, handler)
    with pytest.raises(ValueError, match='预算'):
        baidu._fetch_html(baidu.SEARCH_URL, lambda: None)
    assert len(requests) == 1


def test_cancellation_before_start_does_not_make_network_request_or_cooldown(isolated, monkeypatch):
    monkeypatch.setattr(baidu, '_fetch_html', lambda *args: pytest.fail('network must not be called'))
    def cancelled():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        baidu.baidu_video_search('公开查询', check=cancelled)
    assert not baidu._stop_path().exists()
