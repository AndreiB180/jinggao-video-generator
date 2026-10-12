"""Explicit task routing; local work never falls back to a paid model."""

ROUTES = {
    "planning": "planning_model",
    "requirements": "requirements_model",
    "evidence_review": "review_model",
}
VISUAL_TASKS = {"candidate_ranking", "visual_review", "pixel_observation", "pixel_quality"}


def model(cfg, task, *, vision=False):
    if task in {"captions", "audio_alignment", "deduplication", "render"}:
        raise ValueError("此环节仅使用本地工具，不允许转为付费模型请求")
    if vision or task in VISUAL_TASKS:
        return cfg.get("vision_model", "")
    return cfg.get(ROUTES.get(task, ""), "").strip() or cfg.get("model_name", "")


def describe(cfg):
    return [
        {"task": task, "model": model(cfg, task), "billing": "tokens"}
        for task in (*ROUTES, "pixel_observation", "pixel_quality", "candidate_ranking")
    ] + [{"task": task, "model": "local", "billing": "no_api"}
         for task in ("captions", "audio_alignment", "deduplication", "render")]
