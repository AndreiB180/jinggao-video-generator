"""Actual-frame second opinions must not inherit a first review's mistakes."""
import json
from threading import Barrier, Lock

from PIL import Image
import pytest
import httpx

from jinggao import config, media, neutral_observer, planner, review_evidence

pytestmark = pytest.mark.usefixtures("isolated_pixel_quality", "separate_pixel_observer")


@pytest.fixture
def record(tmp_path):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (1280, 720), "white").save(frame)
    return {"shot": {"id": "shot", "persons": [{"name": "刘翔", "aliases": []}],
                     "narration": "刘翔参加男子110米栏。", "event_scope": "background"},
            "asset": {"kind": "video", "title": "比赛", "source_url": "https://www.iqiyi.com/v_test.html"},
            "frame_paths": [frame], "verification": {"verdict": "match", "reason": "不应泄露的一审意见"}}


@pytest.fixture(autouse=True)
def observation(monkeypatch):
    observed = {"scene_description": "跑道上运动员跨越栏架，旁有刘翔姓名条。", "frames": [
        {"frame_index": 0, "objects": "跑道上运动员，旁有刘翔姓名条", "action": "跨越栏架", "text": ["刘翔"]}]}
    monkeypatch.setattr(neutral_observer, "observe", lambda records, check:
        ({r["shot"]["id"]: observed for r in records}, 1))
    return observed


def evidence(quote="跨越栏架", index=0, field="action"):
    return [{"frame_index": index, "field": field, "quote": quote}]


def caption_check():
    return {"key": "caption_c1", "status": "supported", "reason": "核验实际姓名文字",
            "evidence": evidence("刘翔", field="text")}


@pytest.mark.parametrize("relation,expected", [("on_subject", "supported"), ("scoreboard_participant", "supported"),
                                             ("headline", "unknown"), ("reference_only", "unknown")])
def test_name_headline_is_not_current_subject_evidence(record, monkeypatch, relation, expected):
    def chat(system, content, **kwargs):
        text = json.dumps(content, ensure_ascii=False)
        assert "不应泄露的一审意见" not in text and '"verification"' not in text
        assert all(r["type"] == "text" for r in content)
        assert kwargs["vision"] is False
        return {"reviews": [{"shot_id": "shot", "status": "supported", "reason": "读取字幕", "evidence": evidence(), "labels": [
            {"frame_index": 0, "quote": "刘翔", "relation": relation}], "primary_checks": [caption_check()]}]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    assert planner.audit_selected_frames([record], lambda: None)["shot"]["status"] == expected


def test_invalid_frame_cannot_provide_person_proof(record, monkeypatch):
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"reviews": [{
        "shot_id": "shot", "status": "supported", "reason": "读取字幕", "evidence": evidence(), "labels": [
            {"frame_index": 2, "quote": "刘翔", "relation": "on_subject"}]}]}, {}))
    assert planner.audit_selected_frames([record], lambda: None)["shot"]["status"] == "unknown"


@pytest.mark.parametrize("status,verdict", [("conflict", "mismatch"), ("unknown", "uncertain")])
def test_wrong_place_or_subject_vetoes_first_pass(status, verdict):
    first = {"verdict": "match", "selection_blocked": False, "needs_source_check": False, "reason": "题材相符"}
    reviewed = planner.apply_identity_audit(first, {"status": status, "reason": "实际为香港故宫，不是北京故宫"})
    assert reviewed["verdict"] == verdict and reviewed["selection_blocked"] and reviewed["needs_source_check"]
    assert first["verdict"] == "match"


def test_generic_shot_needs_no_invented_person_label(record, monkeypatch, observation):
    record["shot"].update(persons=[], narration="包裹通过分拣流水线。")
    observation["frames"][0].update(objects="流水线上多个包裹", action="包裹沿分拣线流转", text=[])
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"reviews": [{
        "shot_id": "shot", "status": "supported", "reason": "真实分拣流水线", "evidence": evidence("包裹沿分拣线流转"), "labels": []}]}, {}))
    assert planner.audit_selected_frames([record], lambda: None)["shot"]["status"] == "supported"


def test_pronouns_keep_source_context_without_leaking_initial_verdict(record, monkeypatch, observation):
    observation["frames"][0]["text"] = ["2023年以来"]
    before = {"id": "previous", "narration": "二〇二四年快递业务量为一千七百五十点八亿件。"}
    record["shot"].update(persons=[], narration="这说的是那一年的业务量。")
    context = planner.narrative_context({"shots": [before, record["shot"]]}, record["shot"])
    record["narrative_context"] = context
    assert context[0] == {"position": "before", "narration": before["narration"]}
    def chat(system, content, **kwargs):
        data = json.loads(content[0]["text"])
        assert data["narrative_context"] == context
        assert "不应泄露的一审意见" not in json.dumps(content, ensure_ascii=False)
        return {"reviews": [{"shot_id": "shot", "status": "conflict", "reason": "片内统计字幕年份不同", "evidence": evidence("2023年以来", field="text"), "labels": [
            {"frame_index": 0, "quote": "2023年以来", "relation": "headline"}]}]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    audit = planner.audit_selected_frames([record], lambda: None)["shot"]
    assert audit["status"] == "conflict"
    assert planner.apply_identity_audit(record["verification"], audit)["selection_blocked"]


def test_data_caption_year_cannot_be_excused_as_generic_footage(record, monkeypatch, observation):
    record["shot"].update(persons=[], narration="这说的是那一年的业务量。")
    record["narrative_context"] = [{"position": "before", "narration": "二〇二四年快递业务量增长。"}]
    observation["frames"][0]["text"] = ["据国家邮政局消息 2023年以来 快递业整体呈企稳回升态势"]
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"reviews": [{
        "shot_id": "shot", "status": "supported", "reason": "模型仅按物体匹配忽略统计年份", "evidence": evidence(), "labels": [
            {"frame_index": 0, "quote": "据国家邮政局消息 2023年以来 快递业整体呈企稳回升态势", "relation": "headline"}]}]}, {}))
    reviewed = planner.audit_selected_frames([record], lambda: None)["shot"]
    assert reviewed["status"] == "unknown" and "2024" in reviewed["reason"]
    labels = [{"quote": "拍摄时间 2023年"}]
    assert planner.statistical_caption_conflict(record["shot"], record["narrative_context"], labels) == ""
    assert planner.statistical_caption_conflict({"narration": "包裹在分拣线上流转。"}, [], [{"quote": "2023年业务量增长"}]) == ""
    assert planner.statistical_caption_conflict(record["shot"], record["narrative_context"], [{"quote": "2024年业务量增长"}]) == ""


def test_publisher_caption_bound_to_exact_photo_remains_valid(record, monkeypatch):
    source, url = "https://www.chinanews.com.cn/photo/example.shtml", "https://i2.chinanews.com.cn/photo/example.jpg"
    record["shot"].update(event_scope="background", overlay_label="资料画面")
    record["asset"] = {"kind": "image", "provider": "publisher_image", "source_url": source,
        "origin_image_url": url, "source_evidence": {"url": source, "readable": True, "title": "刘翔在比赛中跨栏。",
            "evidence_origin": "publisher_page", "photo_attribution": "publisher_caption", "image_url": url}}
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"reviews": [{
        "shot_id": "shot", "status": "supported", "reason": "逐图图注对应本图，画面为跨栏", "evidence": evidence(),
        "labels": [], "primary_checks": [caption_check()]}]}, {}))
    assert planner.audit_selected_frames([record], lambda: None)["shot"]["status"] == "supported"
    record["asset"]["origin_image_url"] = "https://i2.chinanews.com.cn/photo/different.jpg"
    assert planner.audit_selected_frames([record], lambda: None)["shot"]["status"] == "unknown"


def test_unrelated_id_cannot_borrow_another_shots_audit(record, monkeypatch):
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"reviews": [{
        "shot_id": "another", "status": "supported", "reason": "匹配", "labels": []}]}, {}))
    audit = planner.audit_selected_frames([record], lambda: None)["shot"]
    assert audit["status"] == "unknown" and "编号" in audit["reason"]
    assert audit["request_count"] == 2


def test_generic_shots_cannot_ignore_explicit_context_conflicts():
    result = planner.review_result({"persons": [], "evidence_mode": "generic"}, {"kind": "video"}, {
        "verdict": "match", "reason": "主题相符", "conflicts": [], "needs_source_check": False,
        "context_conflicts": ["香港博物馆与北京地点要求冲突"], "scene_alignment": "match"}, 5, {})
    assert result["verdict"] == "mismatch" and result["selection_blocked"]


@pytest.mark.parametrize("status", [401, 402, 403, 429])
def test_account_errors_abort_instead_of_becoming_bad_materials(monkeypatch, tmp_path, status):
    monkeypatch.setattr(planner.paths, "DATA", tmp_path)
    monkeypatch.setattr(planner.config, "read", lambda: {"model_name": "test", "vision_model": "test",
        "model_api_key": "secret-never-printed", "model_base_url": "https://models.test",
        "cost_input_per_million": "2", "cost_cached_per_million": "0.04", "cost_output_per_million": "8"})
    request = httpx.Request("POST", "https://models.test/chat/completions")
    monkeypatch.setattr(planner.network, "post", lambda *a, **kw: httpx.Response(status, request=request))
    with pytest.raises(planner.ModelUnavailable, match=f"HTTP {status}") as error:
        planner.chat("test", "test", max_tokens=1000)
    assert "secret-never-printed" not in str(error.value)
    assert not isinstance(error.value, ValueError)


@pytest.mark.parametrize("status", ["supported", "conflict"])
def test_text_judge_cannot_borrow_a_quote_from_another_frame(record, observation, monkeypatch, status):
    observation["frames"].append({"frame_index": 1, "objects": "运动员站立", "action": "站立等候", "text": []})
    record["frame_paths"] *= 2
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"reviews": [{
        "shot_id": "shot", "status": status, "reason": "错误借用前一帧动作", "labels": [],
        "evidence": evidence(index=1)}]}, {}))
    result = planner.audit_selected_frames([record], lambda: None)["shot"]
    assert result["status"] == "unknown" and result["evidence"] == []
    assert result["frame_observations"] == observation["frames"]
    assert result["request_count"] == 2


def test_label_must_come_from_ocr_even_when_objects_mentions_a_name(record, observation, monkeypatch):
    observation["frames"][0]["text"] = []
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"reviews": [{
        "shot_id": "shot", "status": "supported", "reason": "姓名只在主体概括里", "evidence": evidence(),
        "labels": [{"frame_index": 0, "quote": "刘翔", "relation": "on_subject"}]}]}, {}))
    result = planner.audit_selected_frames([record], lambda: None)["shot"]
    assert result["status"] == "unknown" and result["labels"] == []


def test_all_requested_people_need_full_subject_labels(record, monkeypatch):
    record["shot"]["persons"].append({"name": "另一选手", "aliases": []})
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"reviews": [{
        "shot_id": "shot", "status": "supported", "reason": "只标一名参赛者", "evidence": evidence(),
        "labels": [{"frame_index": 0, "quote": "刘翔", "relation": "scoreboard_participant"}]}]}, {}))
    assert planner.audit_selected_frames([record], lambda: None)["shot"]["status"] == "unknown"


def test_judge_cannot_replace_neutral_description_or_hide_statistical_caption(record, observation, monkeypatch):
    record["shot"].update(persons=[], narration="2024年业务量增长。")
    observation["frames"][0]["text"] = ["2023年业务量增长"]
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"reviews": [{
        "shot_id": "shot", "status": "supported", "reason": "只挑所需动作", "evidence": evidence(), "labels": [],
        "scene_description": "按原稿改写场景", "frame_observations": []}]}, {}))
    result = planner.audit_selected_frames([record], lambda: None)["shot"]
    assert result["status"] == "unknown" and "2024" in result["reason"]
    assert result["scene_description"] == observation["scene_description"]
    assert result["frame_observations"] == observation["frames"]


@pytest.mark.parametrize("observer_calls", [0, 1, 2])
def test_every_result_reports_total_calls_including_cached_observations(record, observation, monkeypatch, observer_calls):
    other = record | {"shot": record["shot"] | {"id": "other", "persons": []}}
    record["shot"]["persons"] = []
    monkeypatch.setattr(neutral_observer, "observe", lambda records, check:
        ({r["shot"]["id"]: observation for r in records}, observer_calls))
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"reviews": [
        {"shot_id": sid, "status": "supported", "reason": "实际动作", "labels": [], "evidence": evidence()}
        for sid in ("shot", "other")]}, {}))
    results = planner.audit_selected_frames([record, other], lambda: None)
    assert {r["request_count"] for r in results.values()} == {observer_calls + 1}


@pytest.mark.parametrize("base_url,model,limits", [
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.7-flash", {1: 3072, 2: 4608}),
    ("https://api.deepseek.com", "deepseek-flash", {1: 12000, 2: 12000}),
])
def test_large_audit_uses_two_shot_subbatches_and_totals_actual_calls(
        record, observation, monkeypatch, base_url, model, limits):
    # Output sizing is provider-specific; do not inherit the developer's saved
    # account configuration or depend on whether a test data directory is empty.
    monkeypatch.setattr(config, "read", lambda: config.DEFAULTS | {
        "model_base_url": base_url, "model_name": model, "review_model": model,
        "pixel_observer_mode": "separate"})
    records = [record | {"shot": record["shot"] | {"id": f"shot-{i}", "visual_plan": {
        "mode": "single", "primary": {"subject": "运动员", "action": "跨越栏架", "scene": "跑道"}}},
        "requirements": "保留当前主体姓名证据", "narrative_context": [{"position": "before", "narration": "本场比赛"}]}
        for i in range(5)]
    lock, barrier = Lock(), Barrier(2, timeout=5)
    calls, observed_batches, active, peak = [], [], 0, 0
    def observe(rows, check):
        check()
        ids = [r["shot"]["id"] for r in rows]
        observed_batches.append(ids)
        return {sid: observation for sid in ids}, 0 if ids[0] == "shot-0" else 1
    def chat(system, content, **kwargs):
        nonlocal active, peak
        rows = [json.loads(item["text"]) for item in content]
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append((system, rows))
        try:
            if rows[0]["shot_id"] in {"shot-0", "shot-2"}:
                barrier.wait()
            assert 1 <= len(rows) <= 2
            # v60 sizes the judge's output with review_evidence.output_limit (provider-independent at
            # this layer; per-provider sub-batch sizing happens later in planner._prepare_chat_payload).
            assert kwargs == {"vision": False, "max_tokens": review_evidence.output_limit(content), "timeout": 60,
                              "reasoning": True, "reasoning_effort": "low", "task": "evidence_review"}
            for row in rows:
                assert row["requested_people"] == record["shot"]["persons"]
                assert row["requirements"] == "保留当前主体姓名证据"
                assert row["narrative_context"] == records[0]["narrative_context"]
                assert row["frame_observations"] == observation["frames"]
                # v60 has no per-caption caption_cN keys; captions go through the shared text review.
                assert row["selected_visual_requirements"] == {
                    "subject": "运动员", "action": "跨越栏架", "scene": "跑道"}
            return {"reviews": [{"shot_id": row["shot_id"], "status": "supported", "reason": "真实帧依据",
                "labels": [{"frame_index": 0, "quote": "刘翔", "relation": "on_subject"}], "evidence": evidence(),
                "primary_checks": [{"key": key, "status": "supported", "reason": "实际观察", "evidence": evidence()}
                                   for key in ("subject", "action", "scene")]} for row in rows]}, {}
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(neutral_observer, "observe", observe)
    monkeypatch.setattr(planner, "chat", chat)
    results = planner.audit_selected_frames(records, lambda: None)
    assert peak == 2 and active == 0 and len(calls) == len(observed_batches) == 3
    assert len({system for system, _ in calls}) == 1
    assert list(results) == [r["shot"]["id"] for r in records]
    assert {r["request_count"] for r in results.values()} == {5}  # Three judges, two uncached observers.
    assert all(r["status"] == "supported" and len(r["primary_checks"]) == 3 for r in results.values())


@pytest.mark.parametrize("failure", [planner.ModelUnavailable("HTTP 402"), media.Cancelled()])
def test_split_audit_propagates_fatal_failures_without_retry(record, monkeypatch, failure):
    records = [record | {"shot": record["shot"] | {"id": f"shot-{i}"}} for i in range(6)]
    called = []
    def chat(*args, **kwargs):
        called.append(True)
        raise failure
    monkeypatch.setattr(planner, "chat", chat)
    with pytest.raises(type(failure)):
        planner.audit_selected_frames(records, lambda: None)
    assert 1 <= len(called) <= 2  # At most the two already admitted children, never a retry.


def test_observer_schema_error_preserves_call_count_without_requesting_judge(record, monkeypatch):
    def observe(*args):
        raise neutral_observer.ObservationError("组编号不完整", 2)
    monkeypatch.setattr(neutral_observer, "observe", observe)
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: pytest.fail("invalid facts must not reach judge"))
    result = planner.audit_selected_frames([record], lambda: None)["shot"]
    assert result["status"] == "unknown" and result["request_count"] == 2
    assert result["frame_observations"] == [] and result["scene_description"] == ""


def test_text_schema_error_keeps_neutral_facts_and_both_calls(record, observation, monkeypatch):
    def chat(*args, **kwargs):
        raise ValueError("JSON不完整")
    monkeypatch.setattr(planner, "chat", chat)
    result = planner.audit_selected_frames([record], lambda: None)["shot"]
    assert result["status"] == "unknown" and result["request_count"] == 2
    assert result["frame_observations"] == observation["frames"]


@pytest.mark.parametrize("stage", ["observe", "judge"])
def test_both_audit_stages_propagate_account_failure(record, monkeypatch, stage):
    def unavailable(*args, **kwargs):
        raise planner.ModelUnavailable("HTTP 402")
    monkeypatch.setattr(neutral_observer if stage == "observe" else planner,
                        "observe" if stage == "observe" else "chat", unavailable)
    with pytest.raises(planner.ModelUnavailable, match="402"):
        planner.audit_selected_frames([record], lambda: None)


def test_empty_audit_makes_no_requests(monkeypatch):
    monkeypatch.setattr(neutral_observer, "observe", lambda *a: pytest.fail("empty audit"))
    assert planner.audit_selected_frames([], lambda: None) == {}


def test_global_requirements_reach_text_judge_without_reaching_blind_observer(record, observation, monkeypatch):
    record["requirements"] = "不能出现促销购物教程"
    record["shot"]["persons"] = []
    def chat(system, content, **kwargs):
        assert json.loads(content[0]["text"])["requirements"] == record["requirements"]
        return {"reviews": [{"shot_id": "shot", "status": "supported", "reason": "实际动作",
                              "labels": [], "evidence": evidence(), "primary_checks": [caption_check()]}]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    assert planner.audit_selected_frames([record], lambda: None)["shot"]["status"] == "supported"


def test_statistical_year_on_adjacent_ocr_line_requires_review():
    shot = {"narration": "二〇二四年快递业务量完成一千七百五十点八亿件。"}
    labels = [{"frame_index": 0, "quote": "2023年"},
              {"frame_index": 0, "quote": "全国快递业务量突破400亿件"}]
    assert "2023" in planner.statistical_caption_conflict(shot, [], labels)
    labels[1]["frame_index"] = 1
    assert not planner.statistical_caption_conflict(shot, [], labels)
    labels[1]["frame_index"] = 0
    labels[0]["quote"] = "拍摄时间2023年"
    assert not planner.statistical_caption_conflict(shot, [], labels)
    labels = [{"frame_index": 0, "quote": "拍摄时间"}, {"frame_index": 0, "quote": "2023年"},
              {"frame_index": 0, "quote": "全国快递业务量完成1750.8亿件"}]
    assert not planner.statistical_caption_conflict(shot, [], labels)
