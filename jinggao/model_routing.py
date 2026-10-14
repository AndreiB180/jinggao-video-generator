"""Explicit task routing; local work never falls back to a paid model."""
from urllib.parse import urlparse

ROUTES = {
    "planning": "planning_model",
    "requirements": "requirements_model",
    "evidence_review": "review_model",
    "rewrite": "rewrite_model",
}
# Rewriting is the one step where writing quality shows; a stronger model costs about 0.01 CNY an article.
REWRITE_DEFAULT = {"dashscope.aliyuncs.com": "qwen3.7-plus"}
VISUAL_TASKS = {"candidate_ranking", "visual_review", "pixel_observation", "pixel_quality"}


def model(cfg, task, *, vision=False):
    if task in {"captions", "audio_alignment", "deduplication", "render"}:
        raise ValueError("此环节仅使用本地工具，不允许转为付费模型请求")
    if vision or task in VISUAL_TASKS:
        return cfg.get("vision_model", "")
    # Manual unit prices are entered for the configured model; never switch to a pricier one under them.
    manual_prices = any(str(cfg.get(k, "")).strip() for k in ("cost_input_per_million", "cost_output_per_million"))
    if task == "rewrite" and not cfg.get("rewrite_model", "").strip() and not manual_prices:
        default = REWRITE_DEFAULT.get(urlparse(cfg.get("model_base_url", "")).hostname or "")
        if default:
            return default
    return cfg.get(ROUTES.get(task, ""), "").strip() or cfg.get("model_name", "")


def describe(cfg):
    return [
        {"task": task, "model": model(cfg, task), "billing": "tokens"}
        for task in (*ROUTES, "pixel_observation", "pixel_quality", "candidate_ranking")
    ] + [{"task": task, "model": "local", "billing": "no_api"}
         for task in ("captions", "audio_alignment", "deduplication", "render")]
