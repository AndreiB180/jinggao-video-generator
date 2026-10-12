"""Whole-record text review batching at the ledger's conservative price tier."""
from __future__ import annotations

from copy import deepcopy

from . import costs


def input_limit(cfg, model):
    # Manual prices have no automatic tier. Other models retain their current
    # request shape; this is not a substitute token estimator or budget bypass.
    manual = any(cfg.get(key) for key in (
        "cost_input_per_million", "cost_cached_per_million", "cost_output_per_million"))
    if not manual and costs._provider(cfg) == "dashscope.aliyuncs.com" and model in costs.QWEN_FLASH_MODELS:
        return 32_000
    return None


def partition(content, prepare, first, cfg, limit):
    """Measure each final packed payload, preserving contiguous whole records."""
    if limit is None or len(content) <= 1 or costs._input_bound(cfg, first["payload"]) <= limit:
        return [first]
    batches, current, prepared = [], [], None
    for record in content:
        trial = prepare([*current, record])
        if current and costs._input_bound(cfg, trial["payload"]) > limit:
            batches.append(prepared)
            current = [record]
            prepared = prepare(current)
        else:
            current.append(record)
            prepared = trial
    if prepared:
        batches.append(prepared)
    # An over-limit singleton is deliberately kept intact. The normal ledger
    # still selects its higher price and may stop it before any network send.
    return batches


def unknown_rows(packet, reason):
    return [{"shot_id": record["shot_id"], "status": "unknown", "reason": str(reason)[:1000],
             "labels": [], "evidence": [], "primary_checks": [
                 {"key": key, "status": "unknown", "reason": "本批文本复核未完成", "evidence": []}
                 for key in record.get("selected_visual_requirements", {})]}
            for record in packet.original]


def scoped_rows(packet, response):
    """Never let a missing, duplicate or foreign ID become another shot's row."""
    rows = response.get("reviews") if isinstance(response, dict) else None
    expected = [record["shot_id"] for record in packet.original]
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or any(not isinstance(row, dict) or not isinstance(row.get("shot_id"), str) for row in rows)
            or {row["shot_id"] for row in rows} != set(expected)):
        return unknown_rows(packet, "本批文本复核的镜头编号缺失、重复或不属于本批"), False
    by_id = {row["shot_id"]: row for row in rows}
    return [by_id[sid] for sid in expected], True


def summary(model, infos):
    usage = {}
    def add(target, source):
        for key, value in source.items():
            if isinstance(value, dict):
                if key not in target:
                    target[key] = {}
                if isinstance(target[key], dict):
                    add(target[key], value)
            elif type(value) in (int, float):
                if type(target.get(key, 0)) in (int, float):
                    target[key] = target.get(key, 0) + value
    for info in infos:
        add(usage, info.get("usage", {}))
    hits = sum(bool(info.get("cache_hit")) for info in infos)
    return {"model": model, "usage": usage,
            "request_count": sum(info["request_count"] for info in infos),
            "batch_count": len(infos), "cache_hits": hits,
            "cache_hit": bool(infos) and hits == len(infos),
            "batches": deepcopy(infos)}
