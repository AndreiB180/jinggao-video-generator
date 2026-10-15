"""Regression: a relevant stock clip must reach the bounded metadata prefix."""
from copy import deepcopy
import json

from jinggao import config, model_routing, paths, pooled_match


def test_existing_stock_at_pool_tail_gets_inspection_slot_without_duplication(monkeypatch):
    monkeypatch.setattr(pooled_match, "stock_search_query", lambda *args: "chopping vegetables")
    domestic = [{"id": f"domestic-{i}", "provider": "bilibili", "title": "厨房"} for i in range(20)]
    stocks = [{"id": f"stock-{i}", "provider": "pixabay", "title": "chopping vegetables",
               "approved": False, "status": "unverified"} for i in range(4)]
    selected = domestic + stocks
    audit = [{"candidate_id": row["id"], "reason": "original"} for row in selected]
    before = deepcopy((selected, audit))
    result, notes = pooled_match.stock_candidate_opportunities({}, {"queries": ["切菜"]}, [],
        selected + stocks, selected, audit, 24, [0])
    assert len(result) == len({row["id"] for row in result}) == 24
    assert [row["id"] for row in result[:12] if row["provider"] == "pixabay"] == [r["id"] for r in stocks]
    assert (selected, audit) == before
    assert all(row["id"] == note["candidate_id"] for row, note in zip(result, notes, strict=True))
    assert all(note["metadata_only"] for note in notes if note["reason"] != "original")
    assert not any(row.get("approved") for row in result)


def test_no_hint_leaves_existing_order_and_audit_untouched(monkeypatch):
    monkeypatch.setattr(pooled_match, "stock_search_query", lambda *args: None)
    rows = [{"id": "stock", "provider": "pixabay", "title": "chopping vegetables"}]
    audit = [{"candidate_id": "stock", "reason": "original"}]
    result = pooled_match.stock_candidate_opportunities({}, {"queries": ["指定赛事"]}, [],
        rows, rows, audit, 24, [0])
    assert result == (rows, audit)


def test_fresh_install_uses_one_qwen_key_for_models_and_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    cfg = config.read()
    assert cfg["model_base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert {model_routing.model(cfg, task) for task in (
        "planning", "requirements", "evidence_review", "retrieval_grouping", "pixel_observation",
        "candidate_ranking")} == {"qwen3.7-flash"}
    assert cfg["tts_provider"] == "edge_tts" and cfg["edge_tts_voice"] == "zh-CN-YunxiNeural"
    assert cfg["source_plan_mode"] == "unified" and cfg["pixel_observer_mode"] == "joint"
    assert cfg["model_project_budget_cny"] == cfg["model_job_budget_cny"] == "0.10"
    assert all(not cfg[key] for key in config.SECRETS)
    assert not (tmp_path / "settings.json").exists()


def test_existing_explicit_settings_survive_changed_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    saved = {"model_base_url": "https://example.invalid/v1", "model_name": "chosen-text-model",
             "vision_model": "chosen-vision-model", "tts_provider": "import",
             "source_plan_mode": "legacy", "pixel_observer_mode": "separate"}
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(saved))
    before = path.read_bytes()
    assert {k: config.read()[k] for k in saved} == saved
    assert path.read_bytes() == before
