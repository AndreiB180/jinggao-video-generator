"""A missing tag gets at most one lookup; it never becomes approval evidence."""
from copy import deepcopy

import pytest

from jinggao import config, library, pooled_match, stock_queries, stock_sources, video_sources


@pytest.mark.parametrize(('query', 'title', 'expected'), [
    ('pruning plant leaves', 'plant, green, leaves', 'pruning'),
    ('washing vegetables', 'vegetable, basket', 'washing'),
    ('plant leaf cleaning', 'plant, leaves, green', 'cleaning'),
    ('metal grinding', 'metal, grind, workshop', None),
    ('plant leaf', 'plant, leaves', None),
    ('pruning', 'leaf', None),
    ('2023 C919 flight', 'aircraft flight', None),
    ('office walking', 'snow mountain', None),
    ('home plant tutorial', 'plant', None),
])
def test_missing_term_is_observed_absence_not_an_invented_translation(query, title, expected):
    assert stock_queries.missing_stock_term(query, [{'title': title}]) == expected
    assert stock_queries.missing_stock_term(query, []) is None


def test_single_word_hint_survives_but_named_and_stale_group_gates_remain():
    shot = {'id': 's', 'narration': '修剪植物', 'duration': 8, 'persons': [], 'event': '',
            'event_scope': 'background', 'query': '修剪植物'}
    project = {'id': 'test', 'script': '修剪植物', 'requirements': '', 'shots': [shot]}
    group = {'indices': [0], 'queries': ['修剪植物'], 'stock_queries': ['pruning'],
             'stock_query_context': stock_queries.group_context(project, [shot])}
    before = deepcopy((project, group))
    assert pooled_match.stock_search_query(project, [group], [shot], '修剪植物') == 'pruning'
    assert (project, group) == before
    assert stock_queries.suggested_query('2023 C919首飞', 'pruning') is None
    assert stock_queries.suggested_query('植物', '123') is None
    shot['persons'] = [{'name': '指定人物', 'aliases': []}]
    group['stock_query_context'] = stock_queries.group_context(project, [shot])
    assert pooled_match.stock_search_query(project, [group], [shot], '修剪植物') is None
    shot['persons'] = []
    project['requirements'] = '新的具体要求'
    assert pooled_match.stock_search_query(project, [group], [shot], '修剪植物') is None


def test_rare_activity_match_is_not_buried_by_common_object_tags(monkeypatch):
    monkeypatch.setattr(pooled_match, 'stock_search_query', lambda *a: 'pruning plant leaves')
    rows = [{'id': str(i), 'provider': 'pixabay', 'title': 'plant, leaves, green', 'metadata_only': True}
            for i in range(20)]
    rows.append({'id': 'activity', 'provider': 'pixabay', 'title': 'rose, pruning, plant', 'metadata_only': True})
    before = deepcopy(rows)
    selected, notes = pooled_match.stock_candidate_opportunities({}, {'queries': ['修剪植物']}, [],
        rows, rows[:12], [{'candidate_id': row['id']} for row in rows[:12]], 12, [0])
    assert selected[2]['id'] == 'activity'
    assert notes[2]['stock_queries'][0]['matched_tag_terms'] == ['plant', 'pruning']
    assert len(selected) == 12 and len({row['id'] for row in selected}) == 12
    assert rows == before and all(not row.get('approved') for row in selected)


@pytest.mark.parametrize('fallback_fails', [False, True])
def test_real_search_path_keeps_original_hits_and_stops_after_one_fallback(monkeypatch, fallback_fails):
    monkeypatch.setattr(config, 'read', lambda: {})
    monkeypatch.setattr(stock_sources, 'registry', lambda *a: {'pixabay': {'status': 'configured'}})
    monkeypatch.setattr(library, 'remember_sources', lambda rows: None)
    monkeypatch.setattr(video_sources, '_recent_unavailable', lambda rows, height: (rows, {}))
    primary = {'id': 'p', 'provider': 'pixabay', 'title': 'plant, leaves',
               'source_url': 'https://pixabay.com/videos/id-1/', 'metadata_only': True, 'status': 'unverified'}
    additional = {'id': 'a', 'provider': 'pixabay', 'title': 'rose, pruning, plant',
                  'source_url': 'https://pixabay.com/videos/id-2/', 'metadata_only': True, 'status': 'unverified'}
    before = deepcopy((primary, additional))
    calls = []
    def lookup(provider, query, **kwargs):
        calls.append(query)
        if query == 'pruning plant leaves':
            return [primary]
        assert query == 'pruning'
        if fallback_fails:
            raise ValueError('ordinary public lookup unavailable')
        return [primary, additional]
    monkeypatch.setattr(stock_sources, 'search_items', lookup)
    diagnostic = {}
    rows, errors = video_sources.search_videos('修剪植物', stock_query='pruning plant leaves',
        channel_names={'Pixabay'}, use_index=False, diagnostics=diagnostic)
    assert calls == ['pruning plant leaves', 'pruning']
    assert {c['id'] for c in rows} == ({'p'} if fallback_fails else {'p', 'a'})
    assert not errors
    assert (primary, additional) == before
    assert all(c['metadata_only'] and not c.get('approved') for c in rows)
    report = diagnostic['providers'][0]
    assert report['requested_query'] == '修剪植物' and report['effective_query'] == 'pruning plant leaves'
    assert report['stock_term_fallback']['query'] == 'pruning'
    assert report['stock_term_fallback']['status'] == ('error' if fallback_fails else 'ok')
