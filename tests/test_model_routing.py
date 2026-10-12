import pytest

from jinggao import config, costs, model_routing


def test_text_routes_can_be_cheaper_without_changing_visual_model():
    cfg = config.DEFAULTS | {"model_name": "default", "planning_model": "small-planner",
                             "requirements_model": "small-extractor", "review_model": "small-reviewer",
                             "vision_model": "visual"}
    assert model_routing.model(cfg, "planning") == "small-planner"
    assert model_routing.model(cfg, "requirements") == "small-extractor"
    assert model_routing.model(cfg, "evidence_review") == "small-reviewer"
    assert model_routing.model(cfg, "candidate_ranking") == "visual"
    cfg["planning_model"] = ""
    assert model_routing.model(cfg, "planning") == "default"
    for task in ("captions", "audio_alignment", "deduplication", "render"):
        with pytest.raises(ValueError, match="本地工具"):
            model_routing.model(cfg, task)


def test_cheapest_text_model_depends_on_input_output_mix():
    cfg = config.DEFAULTS | {"model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"}
    flash = costs._rates(cfg, "qwen3.7-flash", 10000)
    turbo = costs._rates(cfg, "qwen-turbo", 10000)
    assert costs._amount(flash, 10000, 1000) < costs._amount(turbo, 10000, 1000)
    assert costs._amount(turbo, 1000, 10000) < costs._amount(flash, 1000, 10000)


def test_planning_cache_reuses_paid_batch_but_invalidates_changed_model_or_request(tmp_path, monkeypatch):
    from jinggao import paths
    from jinggao.plan_cache import PlanningBatch
    monkeypatch.setattr(paths, "DATA", tmp_path)
    cfg = config.DEFAULTS | {"planning_model": "cheap-v1"}
    monkeypatch.setattr(config, "read", lambda: cfg)
    args = ({"id": "project"}, "planning protocol", {"script": "原稿", "requirements": "指定人物"}, {"task": "planning"})
    cache = PlanningBatch(*args)
    cache.save({"shots": [{"source_unit_id": "u1", "subject": "指定人物"}]})
    assert PlanningBatch(*args).read()["shots"][0]["subject"] == "指定人物"
    changed = (args[0], args[1], args[2] | {"requirements": "另一人物"}, args[3])
    assert PlanningBatch(*changed).read() is None
    cfg["planning_model"] = "cheap-v2"
    assert PlanningBatch(*args).read() is None
