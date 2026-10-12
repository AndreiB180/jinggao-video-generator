from collections import defaultdict
from copy import deepcopy

import pytest

from jinggao import pooled_match


def fixture(count=3):
    shots = [{'id': str(i), 'duration': 10, 'narration': '原文不改'} for i in range(count)]
    candidate = {'id': 'c', 'kind': 'video', 'source_url': 'https://example.com/video',
                 'download_status': 'available', 'duration': 150, 'width': 1280, 'height': 720}
    suggestions = {s['id']: [{'candidate_id': 'c', 'sample_at': 60, 'clip_start': 55,
                             'duration': 10, 'sampling_only': True}] for s in shots}
    return shots, {'c': candidate}, suggestions, defaultdict(list)


def test_sparse_collision_gets_distinct_unapproved_windows_without_mutation():
    shots, candidates, suggestions, used = fixture()
    original = deepcopy((shots, candidates, suggestions, used))
    plan = pooled_match.collision_sampling_plan(shots, suggestions, candidates, used,
                                                sampled_sources=14, sampled_frames=34)
    assert len(plan) == 1
    assert [s['id'] for s in plan[0]['shots']] == ['1', '2']
    assert len(plan[0]['windows']) == 2
    assert plan[0]['sampling_only']
    assert not {'approved', 'verification'}.intersection(plan[0]['candidate'])
    assert (shots, candidates, suggestions, used) == original
    # All three can now get distinct *sampled* intervals if the independent
    # observer actually recommends both new frames. Nothing approves content.
    assert pooled_match.reserve_sampled(candidates['c'], suggestions['0'][0], 10, used)
    for row in plan[0]['windows']:
        assert pooled_match.reserve_sampled(candidates['c'],
            {'sample_at': row['at'], 'clip_start': row['start']}, 10, used)
    assert len(used[candidates['c']['source_url']]) == 3


def test_no_collision_or_no_proposal_does_not_add_a_probe():
    shots, candidates, suggestions, used = fixture(1)
    assert pooled_match.collision_sampling_plan(shots, suggestions, candidates, used) == []
    assert pooled_match.collision_sampling_plan(shots, {}, candidates, used) == []


@pytest.mark.parametrize('counts', [dict(sampled_sources=24), dict(sampled_frames=96),
                                   dict(sampled_sources=25, sampled_frames=100)])
def test_original_wave_resource_limits_are_preserved(counts):
    shots, candidates, suggestions, used = fixture()
    assert pooled_match.collision_sampling_plan(shots, suggestions, candidates, used, **counts) == []


def test_read_limits_do_not_expand_to_short_or_unavailable_sources():
    shots, candidates, suggestions, used = fixture()
    candidates['c']['duration'] = 10.12
    for rows in suggestions.values():
        rows[0].update(sample_at=5, clip_start=0)
    assert pooled_match.collision_sampling_plan(shots, suggestions, candidates, used) == []
    candidates['c'].update(duration=150, download_status='blocked')
    assert pooled_match.collision_sampling_plan(shots, suggestions, candidates, used) == []


def test_one_gap_is_not_observed_again_for_every_alternative_source():
    shots, candidates, suggestions, used = fixture(4)
    candidates['d'] = candidates['c'] | {'id': 'd', 'source_url': 'https://example.com/another'}
    for rows in suggestions.values():
        rows.append(rows[0] | {'candidate_id': 'd'})
    plan = pooled_match.collision_sampling_plan(shots, suggestions, candidates, used)
    ids = [shot['id'] for row in plan for shot in row['shots']]
    assert sorted(ids) == ['2', '3']
    assert len(plan) == 1


def test_known_occupied_and_previously_probed_intervals_are_not_requested_again():
    shots, candidates, suggestions, used = fixture()
    key = candidates['c']['source_url']
    used[key].append((65.12, 80))
    old = {key: [(40, 55)]}
    plan = pooled_match.collision_sampling_plan(shots, suggestions, candidates, used, probed=old)
    assert plan
    for window in plan[0]['windows']:
        assert not any(window['start'] < end and window['end'] > start
                       for start, end in [*used[key], *old[key], (55, 65.12)])


def test_multi_source_probe_groups_can_enter_the_production_shared_ranker():
    shots, candidates, suggestions, used = fixture(4)
    candidates['d'] = candidates['c'] | {'id': 'd', 'source_url': 'https://example.com/another'}
    for sid in ['2', '3']:
        suggestions[sid] = [suggestions[sid][0] | {'candidate_id': 'd'}]
    plan = pooled_match.collision_sampling_plan(shots, suggestions, candidates, used)
    rows = [(row['shots'], [row['candidate']]) for row in plan]
    batches, deferred = pooled_match.shared_rank_batches({'shots': shots}, rows)
    assert {s['id'] for batch in batches for s in batch} == {'1', '3'}
    assert not deferred


def test_normal_match_reads_collision_window_but_never_approves_without_full_review(tmp_path, monkeypatch):
    from jinggao import (config, library, library_match, media, montage_candidates, paths,
                         plan_reselection, planner, project_pool, scene_recall, scene_windows,
                         store, video_sources)
    monkeypatch.setattr(paths, 'DATA', tmp_path)
    store.init()
    config.update({'search_provider': 'video'})
    project = store.new_project('相邻片段仍须审核', '工人加工零件。工人继续加工零件。', media_policy='video_only')
    shots = [planner.new_shot(text, query='工人加工', duration=10)
             for text in ['工人加工零件。', '工人继续加工零件。']]
    for shot in shots:
        shot['duration'] = 10
    store.mutate(project['id'], lambda p: p.update(shots=shots))
    project = store.get_project(project['id'])
    candidate = fixture()[1]['c'] | {'provider': 'iqiyi', 'title': '工人加工', 'description': ''}
    monkeypatch.setattr(project_pool, 'collect', lambda *a: ([], {}))
    monkeypatch.setattr(library_match, 'match', lambda *a, **kw: (set(), {
        'library_model_requests': 0, 'library_failures': []}))
    monkeypatch.setattr(pooled_match, 'retrieval_groups', lambda *a: ([{
        'label': '工人加工', 'indices': [0, 1], 'queries': ['工人加工']}], {'cache_hit': True}))
    monkeypatch.setattr(scene_recall, 'candidates', lambda *a: {})
    monkeypatch.setattr(video_sources, 'search_videos', lambda *a, **kw: ([deepcopy(candidate)], []))
    monkeypatch.setattr(video_sources, 'inspect_video_candidate', lambda c, check: c)
    monkeypatch.setattr(planner, 'rank_candidates', lambda p, s, rows, check:
                        [{'candidate_id': c['id']} for c in rows])
    waves, downloads, reviews, reviewed_windows = [], [], [], []
    def discover(p, groups, *args):
        waves.append(deepcopy(groups))
        suggestions = {}
        for targets, candidates in groups:
            c = candidates[0]
            points = c.get('sample_points', [60])
            for shot in targets:
                suggestions[shot['id']] = [{'candidate_id': c['id'], 'sample_at': at,
                                           'clip_start': at-shot['duration']/2} for at in points]
        return suggestions, {}, {'model_requests': 1, 'sampled_bytes': 1,
            'sampled_sources': 1, 'sampled_frames': 4, 'failures': []}
    monkeypatch.setattr(pooled_match, 'discover_windows', discover)
    def download(c, target, start, duration, check):
        downloads.append((start, start+duration))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'fixture')
        return {'source_clip_start': start, 'download_mode': 'fixture', 'downloaded_bytes': 7}
    monkeypatch.setattr(video_sources, 'download_video_candidate', download)
    monkeypatch.setattr(media, 'inspect_asset', lambda pid, target, title: {
        'id': target.stem, 'path': str(target), 'kind': 'video', 'duration': downloads[-1][1]-downloads[-1][0],
        'width': 1280, 'height': 720})
    monkeypatch.setattr(scene_windows, 'align', lambda p, block, asset, used, check:
                        (block['items'], {'adjusted': [], 'error': ''}))
    monkeypatch.setattr(plan_reselection, 'existing_windows', lambda *a: {})
    monkeypatch.setattr(plan_reselection, 'remember', lambda *a: None)
    monkeypatch.setattr(library, 'remember_window', lambda *a, **kw: None)
    def review(p, rows, check):
        reviews.extend(row['shot']['id'] for row in rows)
        reviewed_windows.extend((row['shot']['clip_start'], row['shot']['clip_start']+row['shot']['duration'])
                                for row in rows)
        verdict = {'verdict': 'uncertain', 'scene_alignment': 'uncertain', 'reason': '独立审核未完成',
                   'selection_blocked': True, 'needs_source_check': True,
                   'identity_evidence': {'reason': '通用活动'},
                   'identity_audit': {'failure_stage': 'pixel_observation'}}
        return [row | {'verification': verdict, 'preview': {}} for row in rows], [], {'model_requests': 1}
    monkeypatch.setattr(pooled_match, 'review_batch', review)
    monkeypatch.setattr(montage_candidates, 'match', lambda *a, **kw: (set(), {'model_requests': 0, 'failures': []}))
    stats = pooled_match.match(project, shots, lambda: None, lambda *a: None, workers=1)
    assert len(waves) == 2 and waves[1][0][1][0].get('sample_points')
    assert waves[1][0][0][0]['id'] == shots[1]['id']
    # Close windows share one download, but each has a separate full review.
    assert len(downloads) == 1 and sorted(reviews) == sorted(s['id'] for s in shots)
    a, b = reviewed_windows
    assert a[1] <= b[0] or b[1] <= a[0]
    assert stats['collision_sampling'][0]['sampling_only']
    assert stats['selected'] == 0
    final = store.get_project(project['id'])
    assert not any(s.get('selected_asset') or s.get('approved') for s in final['shots'])
