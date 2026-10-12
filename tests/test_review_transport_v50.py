"""The judge must supply complete decisions; an empty OCR record is not missing data."""
from copy import deepcopy
import hashlib
import json

import httpx
import pytest

from jinggao import config, costs, editorial_contract, frame_contract, network, paths, planner
from jinggao import review_evidence, review_packet, review_schema


def fixture():
    project = {'script': '在公园步道散步。', 'requirements': ''}
    shot = planner.new_shot(project['script'], query='公园步道', visual_origin='generated')
    bundle = editorial_contract.compile_responses(project, [shot], [{'contracts': [{
        'ref': 's1', 'use': 'illustration', 'facts': [], 'must': [], 'refs': [], 'queries': ['公园步道']}]}])
    shot = editorial_contract.attach(project, [shot], bundle)[0]
    frames = [{'frame_index': i, 'objects': '公园小径和树木', 'action': '行人沿小径步行', 'text': []} for i in range(5)]
    refs, _ = review_evidence.references(shot['id'], frames)
    record = {'shot_id': shot['id'], 'narration': shot['narration'], 'source_context': shot['editorial_contract'],
        'selected_visual_requirements': frame_contract.requirements(shot, frames),
        'frame_observations': frames, 'observation_references': refs, 'requested_people': [],
        'evidence_binding': {'frame_sha256': [hashlib.sha256(str(i).encode()).hexdigest() for i in range(5)],
                             'clip_start': 2, 'duration': 6}}
    content = [{'type': 'text', 'text': json.dumps(record, ensure_ascii=False)}]
    packet = review_packet.Packet(content)
    return shot, frames, content, packet


def answer():
    return {'reviews': {'r0': {'status': 'supported', 'reason': '实际公园散步活动与本句相符。', 'labels': [],
        'evidence': [{'ref': 'r0:f0:action'}], 'primary_checks': {
            'source_alignment': {'required_activity': '公园步行', 'observed_activity': '行人在公园步行',
                                 'relationship': 'direct_match', 'reason': '实际行人在公园步行。',
                                 'evidence': [{'ref': 'r0:f0:action'}]},
            'caption_context': {'status': 'supported', 'reason': '所有采样帧均未记录到可读原文字，资料说明无冲突。',
                                'evidence': [{'ref': 'r0:text_absence'}]}}}}}


def test_explicit_absence_is_limited_to_caption_context_and_bound_to_all_frames():
    shot, frames, _, packet = fixture()
    raw = review_schema.parse(packet, json.dumps(answer()))
    restored = packet.restore(raw)['reviews'][0]
    _, lookup = review_evidence.references(shot['id'], frames)
    resolved = review_evidence.resolve(restored, lookup)
    assert frame_contract.validate(shot, resolved['primary_checks'], frames)[0] == 'supported'
    absence = resolved['primary_checks'][1]['evidence'][0]
    assert absence['frame_indices'] == [0, 1, 2, 3, 4] and absence['quote'] == []
    assert not review_evidence.condition_grounded('source_alignment', absence, frames)
    assert not review_evidence.condition_grounded('caption_context', absence | {'frame_indices': [0]}, frames)
    changed = deepcopy(frames)
    changed[4]['text'] = ['另一事件的新闻字幕']
    assert not review_evidence.condition_grounded('caption_context', absence, changed)
    assert packet.valid_response(raw)


@pytest.mark.parametrize('change', ['missing', 'nonempty', 'bad_type', 'missing_action', 'duplicate_index'])
def test_incomplete_or_nonempty_ocr_never_creates_absence_evidence(change):
    _, frames, _, _ = fixture()
    if change == 'missing':
        frames[1].pop('text')
    elif change == 'nonempty':
        frames[1]['text'] = ['CCTV']
    elif change == 'bad_type':
        frames[1]['text'] = ''
    elif change == 'missing_action':
        frames[1]['action'] = ''
    else:
        frames[1]['frame_index'] = 0
    assert review_evidence.text_absence(frames) is None


@pytest.mark.parametrize('change', ['missing_check', 'foreign_ref', 'wrong_condition', 'as_identity', 'missing_shot'])
def test_keyed_review_does_not_repair_invalid_provider_output(change):
    _, _, _, packet = fixture()
    body = answer()
    row = body['reviews']['r0']
    if change == 'missing_check':
        row['primary_checks'].pop('caption_context')
    elif change == 'foreign_ref':
        row['evidence'] = [{'ref': 'r1:f0:action'}]
    elif change == 'wrong_condition':
        row['primary_checks']['source_alignment']['evidence'] = [{'ref': 'r0:text_absence'}]
    elif change == 'as_identity':
        row['labels'] = [{'ref': 'r0:text_absence', 'relation': 'on_subject'}]
    else:
        body['reviews'] = {}
    with pytest.raises(ValueError):
        review_schema.parse(packet, json.dumps(body))


def test_missing_evidence_is_not_filled_and_unknown_is_never_upgraded():
    shot, frames, _, packet = fixture()
    body = answer()
    body['reviews']['r0']['primary_checks']['caption_context']['evidence'] = []
    raw = review_schema.parse(packet, json.dumps(body))
    assert not packet.valid_response(raw)
    _, lookup = review_evidence.references(shot['id'], frames)
    row = review_evidence.resolve(packet.restore(raw)['reviews'][0], lookup)
    assert frame_contract.validate(shot, row['primary_checks'], frames)[0] == 'unknown'
    body['reviews']['r0']['status'] = 'unknown'
    body['reviews']['r0']['primary_checks']['caption_context']['status'] = 'unknown'
    parsed = review_schema.parse(packet, json.dumps(body))
    assert parsed['reviews'][0]['status'] == 'unknown'


def test_payload_preserves_all_inputs_and_prices_the_complete_schema():
    _, _, content, packet = fixture()
    cfg = config.DEFAULTS | {'model_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'}
    prepared = planner._prepare_chat_payload(cfg, 'qwen3.7-flash', 'Complete review rules', content,
        task='evidence_review', vision=False, max_tokens=4096, reasoning=False, reasoning_effort=None)
    assert prepared['packet'].unpack() == packet.original
    payload = prepared['payload']
    schema = payload['response_format']['json_schema']['schema']
    assert set(schema['properties']['reviews']['properties']) == {'r0'}
    caption_refs = schema['$defs']['r0_caption_evidence']['items']['properties']['ref']['enum']
    assert 'r0:text_absence' in caption_refs
    assert 'r0:text_absence' not in schema['$defs']['r0_evidence']['items']['properties']['ref']['enum']
    legacy = payload | {'response_format': {'type': 'json_object'}}
    assert costs._input_bound(cfg, payload)-costs._input_bound(cfg, legacy) == len(
        json.dumps(payload['response_format'], ensure_ascii=False, separators=(',', ':')).encode())


def test_output_allowance_follows_actual_checks_not_remaining_money():
    _, _, content, _ = fixture()
    cfg = config.DEFAULTS | {'model_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'}
    def prepare(rows, settings=cfg, maximum=4096):
        return planner._prepare_chat_payload(settings, 'qwen3.7-flash', 'review', rows,
            task='evidence_review', vision=False, max_tokens=maximum, reasoning=False, reasoning_effort=None)['payload']['max_tokens']
    assert prepare(content) == 2048
    assert prepare(content, cfg | {'model_project_budget_cny': '0'}) == 2048
    assert prepare(content, maximum=1000) == 1000
    larger = json.loads(content[0]['text'])
    larger['selected_visual_requirements'].update({f'must_{i}': f'明确原文条件{i}' for i in range(8)})
    assert prepare([{'type': 'text', 'text': json.dumps(larger)}], maximum=12000) > 4096


@pytest.mark.parametrize('provider,vision,task', [
    ('https://other.example/v1', False, 'evidence_review'),
    ('https://dashscope.aliyuncs.com/compatible-mode/v1', True, 'pixel_observation'),
    ('https://dashscope.aliyuncs.com/compatible-mode/v1', False, 'planning')])
def test_schema_is_only_enabled_for_documented_text_review(provider, vision, task):
    _, _, content, _ = fixture()
    prepared = planner._prepare_chat_payload(config.DEFAULTS | {'model_base_url': provider},
        'qwen3.7-flash', 'protocol', content, task=task, vision=vision, max_tokens=4096,
        reasoning=False, reasoning_effort=None)
    assert prepared['payload']['response_format'] == {'type': 'json_object'}


def test_normal_chat_sends_once_and_reuses_only_complete_bound_result(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'DATA', tmp_path)
    cfg = config.DEFAULTS | {'model_api_key': 'fake-test-key', 'model_name': 'qwen3.7-flash',
        'review_model': 'qwen3.7-flash', 'model_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'}
    monkeypatch.setattr(config, 'read', lambda: cfg.copy())
    _, _, content, _ = fixture()
    calls = []
    def respond(url, **kw):
        calls.append(kw['json'])
        assert kw['json']['response_format']['type'] == 'json_schema'
        return httpx.Response(200, request=httpx.Request('POST', url), json={
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer())}}],
            'usage': {'prompt_tokens': 300, 'completion_tokens': 100}})
    monkeypatch.setattr(network, 'post', respond)
    with costs.scope('fixture', 'test'):
        first, info = planner.chat('review', content, task='evidence_review', max_tokens=4096)
        second, cached = planner.chat('review', content, task='evidence_review', max_tokens=4096)
    assert len(calls) == 1 and info['request_count'] == 1 and cached['request_count'] == 0
    assert first == second and costs.summary('fixture')['project']['requests'] == 1


def test_duplicate_json_keys_fail_before_normalization():
    _, _, _, packet = fixture()
    with pytest.raises(ValueError, match='重复字段'):
        review_schema.parse(packet, '{"reviews":{},"reviews":{}}')
