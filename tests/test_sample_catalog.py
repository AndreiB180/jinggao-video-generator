import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from jinggao import config, costs, media, paths, planner, pooled_match, sample_catalog as catalog, segment_discovery as discovery


@pytest.fixture
def frames(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(config, 'read', lambda: {'vision_model': 'fixture-vision', 'model_base_url': 'https://api.example.cn'})
    rows = []
    for ref in range(2):
        path = tmp_path / f'{ref}.jpg'
        Image.new('RGB', (1280, 720), (ref * 100, 60, 140)).save(path)
        rows.append({'ref': ref, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'at': ref * 10 + 1, 'source_url': 'https://v.qq.com/x/page/test.html', 'title': '不应给观察员的标题'})
    return rows


def response(count=2):
    return {'frames': [{'ref': i, 'objects': '工作台与包裹', 'action': '手抓着包裹', 'text': []}
                       for i in range(count)]}


def test_observer_sees_pixels_only_and_reuses_facts_for_other_sources(frames, tmp_path, monkeypatch):
    calls = []
    def chat(system, content, **options):
        calls.append(content)
        encoded = json.dumps(content, ensure_ascii=False)
        assert '不应给观察员的标题' not in encoded and 'v.qq.com' not in encoded
        assert '孤立采样点' in system and '不得' in system
        assert options['reasoning'] is False and options['vision'] is True
        return response(), {'model': 'fixture'}
    monkeypatch.setattr(planner, 'chat', chat)
    first, info = catalog.describe(frames, lambda: None, cache_dir=tmp_path / 'cache')
    assert info['model_requests'] == 1 and info['cache_hits'] == 0
    second, info = catalog.describe([r | {'source_url': 'https://www.bilibili.com/video/other', 'ref': r['ref']+5}
                                     for r in frames], lambda: None, cache_dir=tmp_path / 'cache')
    assert info['model_requests'] == 0 and info['cache_hits'] == 2 and len(calls) == 1
    assert second[5] == first[0] and second[6] == first[1]
    assert not any(key in first[0] for key in ('approved', 'identity', 'rankings'))


def test_partial_response_keeps_only_complete_facts_and_preserves_request_count(frames, tmp_path, monkeypatch):
    monkeypatch.setattr(planner, 'chat', lambda *a, **k: (response(1), {}))
    facts, info = catalog.describe(frames, lambda: None, cache_dir=tmp_path / 'cache')
    assert set(facts) == {0} and set(info['failed_refs']) == {1}
    assert info['model_requests'] == 1 and len(list((tmp_path / 'cache').glob('*.json'))) == 1
    diagnostic = json.loads(next((tmp_path / 'cache' / 'failures').glob('*.json')).read_text())
    assert diagnostic['accepted_refs'] == [0] and diagnostic['response'] == response(1)


def test_changed_pixels_cannot_reuse_an_old_reference(frames, tmp_path, monkeypatch):
    Path(frames[0]['path']).write_bytes(b'changed')
    monkeypatch.setattr(planner, 'chat', lambda *a, **k: pytest.fail('No request for stale frame hashes'))
    with pytest.raises(ValueError, match='变化'):
        catalog.describe(frames, lambda: None, cache_dir=tmp_path / 'cache')


def test_overlapping_concurrent_requests_only_observe_pixels_once(frames, tmp_path, monkeypatch):
    guard, calls = threading.Lock(), []
    def chat(*args, **kwargs):
        with guard:
            calls.append(1)
        time.sleep(.05)
        return response(), {}
    monkeypatch.setattr(planner, 'chat', chat)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(catalog.describe, frames, lambda: None, cache_dir=tmp_path / 'cache') for _ in range(2)]
        results = [f.result(timeout=2) for f in futures]
    assert len(calls) == 1 and sum(info['model_requests'] for _, info in results) == 1


def sample(frames):
    return [{'candidate_id': 'candidate', 'source_url': frames[0]['source_url'], 'duration': 30,
             'title': '公开物流记录', 'source_evidence': {'title': '公开物流记录'},
             'frames': [{'id': f'f{r["ref"]}', 'path': r['path'], 'at': r['at']} for r in frames]}]


def test_rank_sees_only_blind_facts_and_failed_response_retains_replay_input(frames, tmp_path, monkeypatch):
    calls = []
    def chat(system, content, **options):
        calls.append(options)
        if options['vision']:
            assert '全新文稿' not in json.dumps(content, ensure_ascii=False)
            return response(), {'model': 'observer'}
        parsed = json.loads(content)
        assert parsed['shots'][0]['narration'] == '全新文稿'
        assert parsed['frames'][0]['objects'] == '工作台与包裹'
        assert 'data:image' not in content
        error = planner.ModelResponseError('truncated response', model_info={'model': 'fixture', 'usage': {}})
        error.request_count = 1
        raise error
    monkeypatch.setattr(planner, 'chat', chat)
    with pytest.raises(planner.ModelResponseError) as error:
        discovery.rank_windows_batch([{'id': 'one', 'duration': 6, 'narration': '全新文稿'}], sample(frames),
                                     summary_dir=tmp_path / 'ranks')
    assert len(calls) == 2 and error.value.model_requests == 2
    record = json.loads(Path(error.value.summary_path).read_text())
    assert record['status'] == 'failed' and record['proposals']['one'] == []
    assert record['frame_observations']['0']['action'] == '手抓着包裹'
    assert record['request']['frames'][0]['sha256'] == frames[0]['sha256']
    assert record['request']['shots'][0]['id'] == 'one'


def test_observer_account_failure_is_not_converted_to_empty_proposals(frames, tmp_path, monkeypatch):
    def chat(*args, **kwargs):
        raise planner.ModelUnavailable('额度不足')
    monkeypatch.setattr(planner, 'chat', chat)
    with pytest.raises(planner.ModelUnavailable) as error:
        discovery.rank_windows_batch([{'id': 'one', 'duration': 6}], sample(frames), summary_dir=tmp_path / 'ranks')
    assert error.value.model_requests == 1
    assert json.loads(Path(error.value.summary_path).read_text())['status'] == 'failed'


@pytest.mark.parametrize('bad_observation', [False, True])
def test_discovery_wave_observes_once_before_all_text_batches(frames, tmp_path, monkeypatch, bad_observation):
    shots = [{'id': f's{i}', 'duration': 6, 'narration': '工作人员整理包裹。'} for i in range(18)]
    sampled = sample(frames)[0]
    candidate = {'id': 'candidate'}
    monkeypatch.setattr(discovery, 'sample_candidate', lambda *a, **kw: sampled)
    calls = []
    def chat(system, content, **options):
        calls.append(options['task'])
        if options['vision']:
            result = response()
            if bad_observation:
                result['frames'][0]['action'] = ''
            return result, {'model': 'fixture', 'usage': {'prompt_tokens': 123, 'completion_tokens': 45}}
        assert calls[0] == 'pixel_observation'
        request = json.loads(content)
        return {'rankings': {str(i): [0] for i in range(len(request['shots']))}}, {'model': 'fixture'}
    monkeypatch.setattr(planner, 'chat', chat)
    proposals, reviewed, info = pooled_match.discover_windows({}, [(shots, [candidate])], lambda: None, lambda *a: None)
    assert calls.count('pixel_observation') == 1
    assert info['source_observation']['model_requests'] == 1
    assert len(info['summaries']) == 1
    assert calls == ['pixel_observation', 'scene_ranking'] and info['model_requests'] == 2
    assert all(proposals[s['id']] for s in shots)
    assert set(reviewed) == {s['id'] for s in shots}
    if bad_observation:
        assert info['failures']
        assert all(p['frame_id'] == 'f1' for rows in proposals.values() for p in rows)
        assert info['source_observation']['batches'][0]['model_info']['usage']['prompt_tokens'] == 123
        assert len(list((tmp_path / 'data' / 'cache' / 'sample-catalog').glob('*.json'))) == 1
    else:
        assert not info['failures']
    assert not any(s.get('selected_asset') or s.get('approved') for s in shots)


def test_prepared_facts_reject_changed_absent_or_different_model_without_request(frames, monkeypatch):
    monkeypatch.setattr(planner, 'chat', lambda *a, **k: (response(), {}))
    packet, info = catalog.prepare(frames, lambda: None)
    assert info['model_requests'] == 1
    monkeypatch.setattr(planner, 'chat', lambda *a, **k: pytest.fail('Prepared failures never retry'))
    with pytest.raises(ValueError, match='缺少'):
        catalog.prepared_facts(frames, packet | {'observations': {}}, lambda: None)
    with pytest.raises(ValueError, match='模型或协议'):
        catalog.prepared_facts(frames, packet | {'model_key': 'stale'}, lambda: None)
    Path(frames[0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='变化'):
        catalog.prepared_facts(frames, packet, lambda: None)


def test_wave_deduplicates_identical_pixels_before_failed_batch(frames, monkeypatch):
    calls = []
    def chat(*args, **kwargs):
        calls.append(1)
        return {'frames': []}, {}
    monkeypatch.setattr(planner, 'chat', chat)
    repeated = [frames[i % 2] | {'ref': i} for i in range(20)]
    packet, info = catalog.prepare(repeated, lambda: None)
    assert len(calls) == info['model_requests'] == 1 and info['unique_frames'] == 2
    assert len(packet['errors']) == 2 and not packet['observations']


def test_large_wave_keeps_twelve_image_limit_and_does_not_retry_failed_images(frames, tmp_path, monkeypatch):
    rows = []
    for ref in range(16):
        path = tmp_path / f'large-{ref}.jpg'
        Image.new('RGB', (1280, 720), (ref*12, 30, 70)).save(path)
        rows.append({'ref': ref, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    sizes = []
    def chat(system, content, **options):
        count = sum(row.get('type') == 'image_url' for row in content)
        sizes.append(count)
        result = response(count)
        if len(sizes) == 1:
            result['frames'][0]['action'] = ''
        return result, {'model': 'fixture'}
    monkeypatch.setattr(planner, 'chat', chat)
    packet, info = catalog.prepare([*rows, *rows[:4]], lambda: None)
    assert sizes == [12, 4] and info['model_requests'] == 2
    assert len(packet['errors']) == 1 and len(packet['observations']) == 15
    assert len(list((tmp_path / 'data' / 'cache' / 'sample-catalog').glob('*.json'))) == 15


def test_larger_cohorts_do_not_multiply_observer_concurrency(frames, tmp_path, monkeypatch):
    rows = []
    for ref in range(25):
        path = tmp_path / f'parallel-{ref}.jpg'
        Image.new('RGB', (1280, 720), (ref*9, 40, 60)).save(path)
        rows.append({'ref': ref, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    lock, counts = threading.Lock(), {'active': 0, 'peak': 0, 'calls': 0}
    def describe(batch, check):
        assert len(batch) <= 12
        with lock:
            counts['active'] += 1
            counts['calls'] += 1
            counts['peak'] = max(counts['peak'], counts['active'])
        time.sleep(.03)
        with lock:
            counts['active'] -= 1
        return {r['ref']: {'objects': '机器', 'action': '静止', 'text': []} for r in batch}, {'model_requests': 1}
    monkeypatch.setattr(catalog, 'describe', describe)
    packet, info = catalog.prepare(rows, lambda: None, workers=8)
    assert counts['calls'] == info['model_requests'] == 3 and counts['peak'] == 2
    assert len(packet['observations']) == 25 and not packet['errors']


def test_discovery_wave_cancellation_does_not_rank(frames, monkeypatch):
    class Cancelled(RuntimeError):
        pass
    cancelled = False
    def check():
        if cancelled:
            raise Cancelled('cancelled')
    def chat(*args, **kwargs):
        nonlocal cancelled
        cancelled = True
        return response(), {}
    monkeypatch.setattr(planner, 'chat', chat)
    monkeypatch.setattr(discovery, 'sample_candidate', lambda *a, **kw: sample(frames)[0])
    with pytest.raises(Cancelled):
        pooled_match.discover_windows({}, [([{'id': 'one', 'duration': 6}], [{'id': 'candidate'}])], check, lambda *a: None)


def wave_frames(tmp_path):
    rows = []
    for ref in range(13):
        path = tmp_path / f'wave-{ref}.jpg'
        Image.new('RGB', (1280, 720), (ref * 15, 70, 30)).save(path)
        rows.append({'ref': ref, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows


def observed(batch):
    return {row['ref']: {'objects': '工作台', 'action': '手移动包裹', 'text': []} for row in batch}


@pytest.mark.parametrize('sent_count', [0, 1])
def test_observer_uses_sender_count_when_budget_or_account_rejects(frames, monkeypatch, sent_count):
    error_type = costs.BudgetExceeded if sent_count == 0 else planner.ModelUnavailable
    def chat(*args, **kwargs):
        exc = error_type('sender stopped')
        exc.request_count = sent_count
        exc.model_info = {'request_count': sent_count}
        raise exc
    monkeypatch.setattr(planner, 'chat', chat)
    with pytest.raises(error_type) as error:
        catalog.prepare(frames, lambda: None)
    assert error.value.model_requests == sent_count
    assert error.value.observation_info['batches'][0]['model_requests'] == sent_count


@pytest.mark.parametrize('error_type', [planner.ModelUnavailable, costs.BudgetExceeded, media.Cancelled])
def test_interrupted_wave_keeps_completed_and_failed_batch_usage(frames, tmp_path, monkeypatch, error_type):
    calls = []
    def describe(batch, check):
        calls.append(len(batch))
        info = {'model_requests': 1, 'model_info': {'usage': {'prompt_tokens': len(batch) * 100}}}
        if len(batch) == 1:
            exc = error_type('stop this wave')
            exc.model_requests = 1
            exc.model_info = info['model_info']
            raise exc
        return observed(batch), info
    monkeypatch.setattr(catalog, 'describe', describe)
    with pytest.raises(error_type) as error:
        catalog.prepare(wave_frames(tmp_path), lambda: None, workers=1)
    assert calls == [12, 1] and error.value.model_requests == 2
    info = error.value.observation_info
    assert info['interrupted'] and info['model_requests'] == 2
    assert [row['model_info']['usage']['prompt_tokens'] for row in info['batches']] == [1200, 100]


def test_configuration_change_keeps_prior_wave_request_count(frames, tmp_path, monkeypatch):
    def describe(batch, check):
        monkeypatch.setattr(config, 'read', lambda: {'vision_model': 'changed', 'model_base_url': 'https://api.example.cn'})
        return observed(batch), {'model_requests': 1, 'model_info': {'usage': {'prompt_tokens': 1200}}}
    monkeypatch.setattr(catalog, 'describe', describe)
    with pytest.raises(ValueError, match='配置') as error:
        catalog.prepare(wave_frames(tmp_path), lambda: None, workers=1)
    assert error.value.model_requests == 1
    assert [row['model_requests'] for row in error.value.observation_info['batches']] == [1, 0]
    assert error.value.observation_info['batches'][0]['model_info']['usage']['prompt_tokens'] == 1200


@pytest.mark.parametrize('consumer_cancel', [False, True])
def test_parallel_wave_joins_other_paid_batch_before_reporting_abort(frames, tmp_path, monkeypatch, consumer_cancel):
    barrier, cancelled = threading.Barrier(2), threading.Event()
    def check():
        if cancelled.is_set():
            raise media.Cancelled('cancelled after first response')
    def describe(batch, check):
        barrier.wait(timeout=2)
        if len(batch) == 12:
            time.sleep(.05)  # This already-started request finishes after the other batch aborts.
            return observed(batch), {'model_requests': 1, 'model_info': {'usage': {'prompt_tokens': 1200}}}
        if consumer_cancel:
            cancelled.set()
            return observed(batch), {'model_requests': 1, 'model_info': {'usage': {'prompt_tokens': 100}}}
        exc = planner.ModelUnavailable('account rejected second request')
        exc.model_requests = 1
        exc.model_info = {'usage': {'prompt_tokens': 100}}
        raise exc
    monkeypatch.setattr(catalog, 'describe', describe)
    expected = media.Cancelled if consumer_cancel else planner.ModelUnavailable
    with pytest.raises(expected) as error:
        catalog.prepare(wave_frames(tmp_path), check, workers=2)
    assert error.value.model_requests == 2
    info = error.value.observation_info
    assert info['interrupted'] and len(info['batches']) == 2
    assert sum(row['model_info']['usage']['prompt_tokens'] for row in info['batches']) == 1300
