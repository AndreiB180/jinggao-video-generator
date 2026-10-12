import gzip
import json
import sqlite3
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from jinggao import baidu_videos, library, library_collect, media, paths, planner, platform_videos, portal_videos, publisher_videos, source_index, stock_sources, store, video_sources
from jinggao.api import create_app


@pytest.fixture
def empty(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'DATA', tmp_path / 'catalogue')
    store.init()
    monkeypatch.setattr(stock_sources, "search_items", lambda *a, **kw: [])


def row(n, title='火箭发射', provider='tencent'):
    return video_sources._candidate(provider, title, f'https://v.qq.com/x/page/{n:011d}.html',
        'https://puui.qpic.cn/test.jpg', description='实际来源简介', publisher='发布者')


def test_legacy_metadata_is_migrated_and_discovery_does_not_replace_evidence(empty):
    con = sqlite3.connect(library.root() / 'index.sqlite3')
    con.execute('CREATE TABLE sources (id TEXT PRIMARY KEY,url TEXT UNIQUE,body TEXT,updated REAL)')
    old = row(1) | {'source_evidence': {'evidence_origin': 'publisher_page', 'title': '原始证据'}}
    sid, body = library.source_record(old)
    body.pop('thumbnail_url')
    con.execute('INSERT INTO sources VALUES (?,?,?,?)', (sid, body['source_url'], json.dumps(body), 1))
    con.commit()
    con.close()
    assert source_index.search('火箭')['total'] == 1
    assert library.remember_sources([row(1) | {'source_evidence': {'title': '新封面文字'}}], topic='航天国防', query='火箭发射') == 0
    hit = source_index.search('火箭')['items'][0]
    assert hit['source_evidence']['title'] == '原始证据'
    assert hit['thumbnail_url'] and hit['index_state'] == 'indexed'
    assert source_index.overview()['topics']['航天国防'] == 1


def test_source_search_ranks_before_limiting_and_paginates_without_duplicates(empty):
    library.remember_sources([row(1, '马龙 乒乓球')])
    library.remember_sources([row(n, '每日生活视频') for n in range(2, 1002)])
    library.remember_sources([row(2000, '城市消防') | {'description': '马龙区消防队'}])
    assert source_index.search('马龙')['items'][0]['title'] == '马龙 乒乓球'
    assert source_index.search('每日生活视频')['total'] == 1000
    a = source_index.search('每日生活视频', limit=40)['items']
    b = source_index.search('每日生活视频', limit=40, offset=40)['items']
    assert len(a) == len(b) == 40
    assert not {x['source_id'] for x in a} & {x['source_id'] for x in b}
    assert source_index.search('每日生活视频', provider='cctv')['total'] == 0


def test_index_recall_remains_unchecked_and_strong_hits_skip_network(empty, monkeypatch):
    candidates = [row(n, '火箭发射现场') | {'approved': True, 'media_streams': ['private'],
                  'verification': {'verdict': 'match'}} for n in range(20)]
    library.remember_sources(candidates)
    for candidate in candidates:
        source_index.record_check(candidate | {'download_status': 'available'})
    def forbidden(*args, **kwargs):
        pytest.fail('strong index hit used network')
    for module, name in ((video_sources, 'cctv_search'), (platform_videos, 'tencent_search'),
                         (platform_videos, 'bilibili_search'), (publisher_videos, 'chinanews_search'),
                         (portal_videos, 'iqiyi_search'), (portal_videos, 'sohu_search'), (portal_videos, 'youku_search'),
                         (baidu_videos, 'baidu_video_search')):
        monkeypatch.setattr(module, name, forbidden)
    rows, failures = video_sources.search_videos('火箭发射')
    assert len(rows) == 20 and not failures
    assert all(c['from_source_index'] and c['download_status'] == 'unchecked' and not c['metadata_checked'] for c in rows)
    assert all('approved' not in c and 'media_streams' not in c and 'verification' not in c for c in rows)
    assert source_index.recall('火箭发射', channel_names={'央视'}) == []


def test_unchecked_title_matches_cannot_suppress_new_usable_providers(empty, monkeypatch):
    library.remember_sources([row(n, '火箭发射现场') for n in range(20)])
    calls = []
    def adapter(query):
        calls.append(query)
        return []
    for module, name in ((video_sources, 'cctv_search'), (platform_videos, 'tencent_search'),
                         (platform_videos, 'bilibili_search'), (publisher_videos, 'chinanews_search'),
                         (portal_videos, 'iqiyi_search'), (portal_videos, 'sohu_search'), (portal_videos, 'youku_search'),
                         (baidu_videos, 'baidu_video_search')):
        monkeypatch.setattr(module, name, adapter)
    rows, _ = video_sources.search_videos('火箭发射')
    assert len(rows) == 20 and len(calls) == 8
    assert not any(r.get('index_only_search') for r in rows)


def test_sparse_index_still_uses_all_domestic_providers(empty, monkeypatch):
    library.remember_sources([row(1)])
    seen = []
    def adapter(q):
        seen.append(q)
        return [row(2)]
    for module, name in ((video_sources, 'cctv_search'), (platform_videos, 'tencent_search'),
                         (platform_videos, 'bilibili_search'), (publisher_videos, 'chinanews_search'),
                         (portal_videos, 'iqiyi_search'), (portal_videos, 'sohu_search'), (portal_videos, 'youku_search'),
                         (baidu_videos, 'baidu_video_search')):
        monkeypatch.setattr(module, name, adapter)
    rows, _ = video_sources.search_videos('火箭')
    assert len(seen) == 8 and len(rows) == 2


def test_entity_only_index_cannot_suppress_exact_event_search(empty, monkeypatch):
    library.remember_sources([row(n, '马龙日常训练') for n in range(20)])
    seen = []
    def adapter(q):
        seen.append(q)
        return []
    for module, name in ((video_sources, 'cctv_search'), (platform_videos, 'tencent_search'),
                         (platform_videos, 'bilibili_search'), (publisher_videos, 'chinanews_search'),
                         (portal_videos, 'iqiyi_search'), (portal_videos, 'sohu_search'), (portal_videos, 'youku_search'),
                         (baidu_videos, 'baidu_video_search')):
        monkeypatch.setattr(module, name, adapter)
    video_sources.search_videos('马龙东京奥运会决赛', library_query='马龙')
    assert len(seen) == 8


def test_partial_chinese_name_is_not_a_person_candidate(empty):
    library.remember_sources([row(1, '陈冠宏采访'), row(2, '郭冠宏比赛')])
    assert [c['title'] for c in source_index.recall('郭冠宏')] == ['郭冠宏比赛']


def test_aggregator_discovery_is_preserved_without_becoming_publisher_proof(empty):
    candidate = row(1, '公开演讲', 'baidu_video') | {
        'platform_provider': 'tencent', 'source_verified': False, 'publisher': '',
        'source_evidence': {'readable': False, 'evidence_origin': 'unread_original_page'},
        'discovery_evidence': {'url': 'https://www.baidu.com/sf/vsearch?pd=video&wd=演讲',
                               'title': '公开演讲', 'evidence_origin': 'aggregator_search'}}
    library.remember_sources([candidate])
    found = source_index.search('公开演讲', provider='baidu_video')['items'][0]
    assert found['discovery_evidence'] == candidate['discovery_evidence']
    assert found['platform_provider'] == 'tencent'
    assert not found['source_verified'] and not found['source_evidence']['readable']
    assert found['publisher'] == ''


def test_bulk_is_metadata_only_resumable_and_stops_challenged_provider(empty, monkeypatch):
    monkeypatch.setattr(source_index, 'TOPICS', {'生活家居': '家庭生活', '航天国防': '火箭发射'})
    calls = []
    def adapter(q, page):
        calls.append((q, page))
        return [row(page * 10 + (q == '火箭发射'), q)]
    def blocked(q, page):
        raise ValueError('HTTP 412 验证')
    monkeypatch.setattr(source_index, 'routes', lambda: {'tencent': adapter, 'bilibili': blocked})
    monkeypatch.setattr(source_index, 'REQUEST_INTERVAL', 0)
    def forbidden(*a, **kw):
        pytest.fail('metadata build downloaded or reviewed video')
    monkeypatch.setattr(video_sources, 'inspect_video_candidate', forbidden)
    monkeypatch.setattr(video_sources, 'download_video_candidate', forbidden)
    monkeypatch.setattr(planner, 'chat', forbidden)
    pid = library_collect.project_id()
    options = {'index_target': 100, 'index_entities': False, 'index_providers': ['tencent', 'bilibili']}
    result = source_index.run(pid, options, lambda: None, lambda *a: None)
    assert result['total'] == 6 and result['pages'] == 7
    assert 'bilibili' in result['provider_stops'] and result['downloads'] == result['model_requests'] == 0
    assert calls[:2] == [('家庭生活', 1), ('火箭发射', 1)]
    calls.clear()
    again = source_index.run(pid, options, lambda: None, lambda *a: None)
    assert not calls and again['skipped_pages'] == 6 and again['added'] == 0


def test_cancel_retains_completed_pages(empty, monkeypatch):
    monkeypatch.setattr(source_index, 'TOPICS', {'航天国防': '火箭发射'})
    monkeypatch.setattr(source_index, 'routes', lambda: {'tencent': lambda q, p: [row(p)]})
    pid = library_collect.project_id()
    def check():
        if library.stats()['sources']:
            raise media.Cancelled()
    with pytest.raises(media.Cancelled):
        source_index.run(pid, {'index_target': 100, 'index_entities': False, 'index_providers': ['tencent']}, check, lambda *a: None)
    assert library.stats()['sources'] == 1
    with library.connection() as c:
        assert c.execute("SELECT count(*) FROM index_pages WHERE status='completed'").fetchone()[0] == 1


def test_seed_contains_no_projects_accounts_files_or_reviews_and_installs_once(empty, tmp_path, monkeypatch):
    library.remember_sources([row(1) | {'context_sources': [{'private': 'context'}], 'approved': True}],
                             topic='航天国防', query='火箭发射')
    destination = tmp_path / 'public.jsonl.gz'
    assert source_index.export_seed(destination) == 1
    content = gzip.decompress(destination.read_bytes()).decode()
    assert 'private' not in content and 'context_sources' not in content and 'approved' not in content
    monkeypatch.setattr(paths, 'DATA', tmp_path / 'clean-client')
    assert source_index.install_seed(destination) == 1
    assert source_index.install_seed(destination) == 0
    assert source_index.search('火箭')['items'][0]['index_state'] == 'indexed'
    assert source_index.overview()['topics']['航天国防'] == 1
    assert library.stats()['bytes'] == 0


def test_pagination_parameters_and_cache_keys_are_distinct(empty, monkeypatch):
    calls = []
    def fetch(url, **kw):
        calls.append((url, kw))
        return b'{"ret":0,"data":{"errcode":0}}', {}
    monkeypatch.setattr(video_sources, '_fetch', fetch)
    platform_videos.tencent_search('厨房', 1)
    platform_videos.tencent_search('厨房', 2)
    assert [k['json_payload']['pagenum'] for _, k in calls] == [0, 1]
    calls.clear()
    monkeypatch.setattr(video_sources, '_json', lambda url, **kw: calls.append(url) or {'list': []})
    video_sources.cctv_search('厨房', 1)
    video_sources.cctv_search('厨房', 2)
    assert [parse_qs(urlparse(u).query)['page'] for u in calls] == [['1'], ['2']]


def test_catalog_api_validates_bounds(empty):
    with TestClient(create_app()) as client:
        assert client.get('/api/library/sources?topic=未知分类').status_code == 400
        assert client.get('/api/library/sources?offset=-1').status_code == 400
        assert client.post('/api/library/build', json={'action': 'index', 'index_target': 999999}).status_code == 422
        assert client.post('/api/library/build', json={'action': 'index', 'index_topics': ['未知分类']}).status_code == 400
        assert client.post('/api/library/build', json={'action': 'index'}, headers={'Origin': 'https://example.com'}).status_code == 403


def test_entity_catalogue_counts_mentions_not_seed_names_or_identity_approval(empty):
    from jinggao import catalog_taxonomy
    library.remember_sources([row(1,'马龙回顾2024年巴黎奥运会'),row(2,'辛巴直播片段'),row(3,'陈冠宏讲话')])
    source_index.prepare_entities()
    names=source_index.entity_search('辛有志')
    assert names[0]['sources']==1 and names[0]['evidence']=='metadata_mention'
    assert source_index.search(entity=names[0]['id'])['items'][0]['title']=='辛巴直播片段'
    assert source_index.search('辛有志')['items'][0]['title']=='辛巴直播片段'
    assert source_index.recall('辛有志')[0]['title']=='辛巴直播片段'
    hit=source_index.entity_search('2024年巴黎奥运会马龙')[0]
    assert hit['sources']==1
    assert source_index.overview()['planned_entities']['person']>source_index.overview()['entities']['person']
    assert not catalog_taxonomy.mentions('盘子里的饺子已经煮好了')
    record=source_index.search(entity=names[0]['id'])['items'][0]
    assert 'approved' not in record and record['index_state']=='indexed'


def test_entity_api_rejects_invalid_kind_and_filters_sources(empty):
    source_index.prepare_entities()
    with TestClient(create_app()) as client:
        assert client.get('/api/library/entities?kind=faces').status_code==400
        records=client.get('/api/library/entities?query=雷军&kind=person').json()['items']
        assert len(records)==1 and records[0]['sources']==0
