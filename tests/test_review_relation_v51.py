"""An explicit same-topic judgment must not be converted into visual approval."""
import json

import pytest

from jinggao import curation, frame_contract, planner, review_evidence, review_schema
from tests.test_review_transport_v50 import answer, fixture


@pytest.mark.parametrize(("relation", "expected"), [
    ("direct_match", "supported"), ("listed_example", "supported"),
    ("topic_only", "conflict"), ("different_activity", "conflict"), ("insufficient", "unknown"),
])
def test_declared_relation_controls_condition_even_when_summary_says_supported(relation, expected):
    shot, frames, _, packet = fixture()
    body = answer()
    source = body['reviews']['r0']['primary_checks']['source_alignment']
    source['relationship'] = relation
    parsed = review_schema.parse(packet, json.dumps(body))
    restored = packet.restore(parsed)['reviews'][0]
    _, lookup = review_evidence.references(shot['id'], frames)
    resolved = review_evidence.resolve(restored, lookup)
    status, _, checks = frame_contract.validate(shot, resolved['primary_checks'], frames)
    assert status == expected
    assert checks[0]['source_relation']['relationship'] == relation
    assert checks[0]['source_relation']['required_activity'] == '公园步行'
    assert body['reviews']['r0']['status'] == 'supported'  # An optimistic summary cannot override a mismatch.


def test_schema_requires_comparison_before_summary_and_scopes_evidence():
    _, _, _, packet = fixture()
    schema = review_schema.response_format(packet)['json_schema']['schema']
    assert next(iter(schema['properties']['reviews']['properties']['r0']['properties'])) == 'primary_checks'
    alignment = schema['$defs']['r0_alignment']
    assert 'status' not in alignment['properties']
    assert set(alignment['properties']['relationship']['enum']) == set(review_schema.relation_options(packet.original[0]))
    assert alignment['properties']['evidence']['$ref'] == '#/$defs/r0_evidence'


@pytest.mark.parametrize('field', ['required_activity', 'observed_activity', 'relationship'])
def test_missing_or_unknown_comparison_is_not_repaired(field):
    _, _, _, packet = fixture()
    body = answer()
    body['reviews']['r0']['primary_checks']['source_alignment'].pop(field)
    with pytest.raises(ValueError):
        review_schema.parse(packet, json.dumps(body))


def test_old_automatic_source_review_cannot_keep_its_approval():
    shot, _, _, _ = fixture()
    shot.update(selected_asset='asset', approved=True, verification={
        'verdict': 'match', 'scene_alignment': 'match', 'selection_blocked': False,
        'needs_source_check': False, 'identity_audit': {'version': planner.FRAME_AUDIT_VERSION-1}})
    project = {'script': shot['narration'], 'requirements': '', 'shots': [shot],
               'assets': [{'id': 'asset', 'kind': 'video'}]}
    assert '规则已更新' in curation.selection_error(project, shot, project['assets'][0])
    assert curation.approve_verified(project) == 0
    assert shot['approved'] is False and shot['selected_asset'] == 'asset'
