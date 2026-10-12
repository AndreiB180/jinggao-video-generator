from __future__ import annotations

import json
import os
import threading
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse

from . import paths

DEFAULTS = {
    "model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model_api_key": "", "model_name": "qwen3.7-flash",
    "vision_model": "qwen3.7-flash", "tavily_api_key": "", "pixabay_api_key": "", "pexels_api_key": "", "coverr_api_key": "",
    "planning_model": "", "requirements_model": "", "review_model": "",
    "tts_provider": "baidu_http", "tts_base_url": "https://api.openai.com/v1",
    "tts_api_key": "", "tts_model": "", "tts_voice": "alloy",
    "tts_pricing_json": "",
    "encoder": "auto", "search_provider": "mixed",
    "baidu_voice_name": "磁性专业很灵动",
    "plan_workers": "3", "match_workers": "3", "render_workers": "2",
    "plan_retries": "1",
    "source_plan_mode": "unified", "pixel_observer_mode": "joint",
    "model_project_budget_cny": "0.10", "model_daily_budget_cny": "3.00",
    "model_job_budget_cny": "0.10",
    "model_provider_budget_cny": "",
    "cost_input_per_million": "", "cost_cached_per_million": "", "cost_output_per_million": "",
    "cost_image_tokens": "0",
}
SECRETS = {k for k in DEFAULTS if k.endswith("api_key")}
_lock = threading.Lock()


def read() -> dict:
    p = paths.DATA / "settings.json"
    data = DEFAULTS | json.loads(p.read_text("utf-8")) if p.exists() else DEFAULTS.copy()
    if data["search_provider"] not in ("mixed", "video", "local"):
        data["search_provider"] = "mixed"
    return data


def public() -> dict:
    data = read()
    return {k: v for k, v in data.items() if k not in SECRETS} | {
        k + "_set": bool(data[k]) for k in SECRETS
    }


def update(values: dict) -> dict:
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError("未知设置字段")
    for key, value in values.items():
        if not isinstance(value, str) or len(value) > 4000:
            raise ValueError("设置内容无效")
        if key.endswith("base_url"):
            parsed = urlparse(value)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username:
                raise ValueError("服务地址必须是 HTTP(S) 地址，不能包含用户名密码")
    if values.get("encoder", "auto") not in ("auto", "cpu", "nvenc"):
        raise ValueError("编码方式无效")
    if values.get("tts_provider", "system") not in ("system", "compatible", "import", "baidu_web", "baidu_http"):
        raise ValueError("配音方式无效")
    if values.get("search_provider", "mixed") not in ("mixed", "video", "local"):
        raise ValueError("素材检索仅支持多源检索、视频检索或本地素材")
    for key in ("plan_workers", "match_workers", "render_workers"):
        if key in values and values[key] not in ("1", "2", "3", "4"):
            raise ValueError("并发数须为 1 到 4")
    if values.get("plan_retries", "1") not in ("0", "1"):
        raise ValueError("分镜格式重试次数须为 0 或 1")
    if values.get("source_plan_mode", "legacy") not in ("legacy", "unified"):
        raise ValueError("源文规划方式无效")
    if values.get("pixel_observer_mode", "separate") not in ("separate", "joint"):
        raise ValueError("画面观察方式无效")
    paths.prepare()
    with _lock:
        data = read() | values
        from .tts_costs import validate_pricing
        validate_pricing(data["tts_pricing_json"])
        for key in ("model_project_budget_cny", "model_daily_budget_cny", "model_provider_budget_cny", "model_job_budget_cny"):
            if key == "model_provider_budget_cny" and data[key] == "":
                continue
            try:
                limit = Decimal(data[key])
                if not limit.is_finite() or not 0 <= limit <= 1000 or limit != limit.quantize(Decimal("0.01")):
                    raise ValueError
            except (ValueError, InvalidOperation):
                raise ValueError("模型预算须为 0 到 1000 元，最多两位小数；0 表示暂停付费请求") from None
        rates = [data[k] for k in ("cost_input_per_million", "cost_cached_per_million", "cost_output_per_million")]
        if any(rates):
            try:
                parsed = [Decimal(v) for v in rates]
                if (not all(v.is_finite() and 0 <= v <= 100000 for v in parsed)
                        or parsed[0] <= 0 or parsed[2] <= 0 or parsed[1] > parsed[0]):
                    raise ValueError
            except (ValueError, InvalidOperation):
                raise ValueError("请完整填写输入、缓存输入和输出单价；输入与输出须大于零，缓存价不能高于输入价") from None
        if not data["cost_image_tokens"].isdigit() or not 0 <= int(data["cost_image_tokens"]) <= 1000000:
            raise ValueError("每图 token 预留须为 0 到 1000000 的整数；0 表示仅使用已知模型规则")
        target = paths.DATA / "settings.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(target)
    return public()
