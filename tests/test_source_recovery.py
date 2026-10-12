"""Recovery keeps independently valid footage opportunities, never approvals."""
from copy import deepcopy
import hashlib
import json

import pytest
from PIL import Image

from jinggao import config, editorial_contract, paths, planner, pooled_match
from jinggao import sample_catalog as catalog, segment_discovery as discovery


@pytest.fixture
def frames(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(config, 'read', lambda: {
        'vision_model': 'fixture', 'model_base_url': 'https://example.test/v1'})
    result = []
    for ref in range(3):
        path = tmp_path / f'{ref}.jpg'
        Image.new('RGB', (1280, 720), (ref*80, 60, 40)).save(path)
        result.append({'ref': ref, 'path': str(path),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


def observation(ref):
    return {'ref': ref, 'objects': '公园步道与行人', 'action': '行人走过步道', 'text': []}


@pytest.mark.parametrize('rows', [
    [observation(0), observation(0)], [observation(-1)], [observation(3)],
    [observation(True)], [{'objects': '公园'}, observation(1)],
])
def test_ambiguous_numbering_discards_entire_paid_response(frames, tmp_path, monkeypatch, rows):
    monkeypatch.setattr(planner, 'chat', lambda *a, **k: ({'frames': rows}, {'request_count': 1}))
    with pytest.raises(ValueError, match='编号') as caught:
        catalog.describe(frames, lambda: None, cache_dir=tmp_path / 'cache')
    assert caught.value.model_requests == 1
    assert not list((tmp_path / 'cache').glob('*.json'))


def test_partial_direct_rank_cannot_select_failed_or_renumbered_frame(frames, tmp_path, monkeypatch):
    calls = []
    def chat(system, content, **options):
        calls.append(options['task'])
        if options['vision']:
            return {'frames': [observation(0) | {'action': ''}, observation(2)]}, {}
        request = json.loads(content)
        assert [row['ref'] for row in request['frames']] == [2]
        return {'rankings': {'0': [0, 1, 2]}}, {}
    monkeypatch.setattr(planner, 'chat', chat)
    result = discovery.rank_windows_batch([{'id': 'shot', 'duration': 6, 'narration': '在公园散步。'}],
        [{'candidate_id': 'source', 'source_url': 'https://v.qq.com/x/page/test.html', 'duration': 60,
          'frames': [{'id': f'original-{i}', 'at': i*10+5, 'path': frame['path']}
                     for i, frame in enumerate(frames)]}], summary_dir=tmp_path / 'ranks')
    assert calls == ['pixel_observation', 'scene_ranking']
    assert [row['frame_id'] for row in result['proposals']['shot']] == ['original-2']
    assert all(row['sampling_only'] for row in result['proposals']['shot'])


def test_no_valid_frame_stops_before_text_ranking(frames, tmp_path, monkeypatch):
    calls = []
    def chat(*args, **options):
        calls.append(options['task'])
        assert options['vision']
        return {'frames': []}, {}
    monkeypatch.setattr(planner, 'chat', chat)
    result = discovery.rank_windows_batch([{'id': 'shot', 'duration': 6}],
        [{'candidate_id': 'source', 'source_url': 'https://v.qq.com/x/page/test.html', 'duration': 60,
          'frames': [{'id': 'frame', 'at': 5, 'path': frames[0]['path']}]}],
        summary_dir=tmp_path / 'ranks')
    assert calls == ['pixel_observation'] and result['model_requests'] == 1
    assert result['status'] == 'no_observed_frames' and result['proposals'] == {'shot': []}


def test_shared_partial_facts_still_bind_model_and_actual_pixels(frames, monkeypatch):
    monkeypatch.setattr(planner, 'chat', lambda *a, **k: ({'frames': [observation(2)]}, {}))
    packet, _ = catalog.prepare(frames, lambda: None)
    monkeypatch.setattr(planner, 'chat', lambda *a, **k: pytest.fail('No retries'))
    facts, info = catalog.prepared_facts(frames, packet, lambda: None, allow_partial=True)
    assert set(facts) == {2} and set(info['failed_refs']) == {0, 1}
    with pytest.raises(ValueError, match='模型或协议'):
        catalog.prepared_facts(frames, packet | {'model_key': 'changed'}, lambda: None, allow_partial=True)
    from pathlib import Path
    Path(frames[0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='变化'):
        catalog.prepared_facts(frames, packet, lambda: None, allow_partial=True)


def park():
    project = {'script': '在公园散步，观察树木和行人。', 'requirements': ''}
    shots = [planner.new_shot(project['script'], query='公园散步', visual_origin='generated')]
    bundle = editorial_contract.compile_responses(project, shots, [{'contracts': [{
        'ref': 's1', 'use': 'illustration', 'refs': [], 'facts': [], 'must': [],
        'queries': ['公园散步过程实拍', '公园跑步跟拍']}]}])
    shots = editorial_contract.attach(project, shots, bundle)
    return project, shots, {'indices': [0], 'queries': ['公园散步过程实拍', '公园跑步跟拍']}


def test_subject_sources_survive_twelve_row_cut_without_description_only_topics():
    project, shots, group = park()
    def video(cid, title, provider, **extra):
        return {'id': cid, 'kind': 'video', 'title': title, 'provider': provider,
                'duration': 180, 'width': 1280, 'height': 720, 'status': 'unverified', **extra}
    rows = [video(f'news-{i}', '公园散步跑步过程实拍新闻', 'chinanews') for i in range(30)]
    rows += [video(f'walk-{i}', '城市公园实拍', 'bilibili') for i in range(6)]
    rows += [video(f'stock-{i}', '公园, 树木, 自然', 'pixabay') for i in range(6)]
    rows += [video('unrelated', '探访汽车生产车间', 'iqiyi', description='公园散步过程实拍'),
             video('low', '公园实拍', 'sohu', height=480)]
    before = deepcopy((project, shots, rows))
    selected, audit = pooled_match.primary_candidate_pool(project, group, shots, rows, 24)
    assert len(selected) == 24
    assert {row['provider'] for row in selected[:12]} == {'bilibili', 'chinanews', 'pixabay'}
    assert sum(row['provider'] == 'chinanews' for row in selected[:12]) <= 4
    assert not {'unrelated', 'low'} & {row['id'] for row in selected[:12]}
    assert all(row['metadata_only'] for row in audit)
    assert (project, shots, rows) == before


@pytest.mark.parametrize('change', [{'persons': [{'name': '张三', 'aliases': []}]},
    {'event': '2026年开幕式', 'event_scope': 'exact'}, {'visual_origin': 'manual'}, {'locked': True}])
def test_special_requirements_never_enter_subject_diversification(monkeypatch, change):
    project, shots, group = park()
    shots[0].update(change)
    monkeypatch.setattr(pooled_match, 'source_candidate_pool', lambda *a, **k: pytest.fail('Keep exact requirements'))
    pooled_match.primary_candidate_pool(project, group, shots,
        [{'id': 'candidate', 'kind': 'video', 'title': '公园'}], 12)


@pytest.mark.parametrize('raw', ['{"frames":[],"frames":[]}', '{"frames":[{"ref":0,"ref":1}]}'])
def test_pixel_json_rejects_duplicate_fields_before_binding(raw):
    with pytest.raises(ValueError, match='重复字段'):
        planner.json_response(raw, strict=True)
