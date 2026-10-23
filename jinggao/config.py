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
    "planning_model": "", "requirements_model": "", "review_model": "", "rewrite_model": "",
    "tts_provider": "edge_tts", "edge_tts_voice": "zh-CN-YunxiNeural", "qwen_tts_voice": "Ethan", "tts_speed": "1.1", "tts_daily_budget_cny": "5.00", "index_daily_budget_cny": "2.00",
    "tts_migrated": "", "tts_migrated_edge": "",
    "export_dir": "", "auto_save_exports": "1",
    "tts_base_url": "https://api.openai.com/v1",
    "tts_api_key": "", "tts_model": "", "tts_voice": "alloy",
    "tts_pricing_json": "",
    "encoder": "auto", "search_provider": "mixed",
    "baidu_voice_name": "磁性专业很灵动", "baidu_browser": "auto",
    "plan_workers": "3", "match_workers": "3", "render_workers": "3", "video_workers": "3",
    "plan_retries": "1",
    "source_plan_mode": "unified", "pixel_observer_mode": "joint",
    "model_project_budget_cny": "0.10", "model_daily_budget_cny": "3.00",
    "model_job_budget_cny": "0.10",
    "model_provider_budget_cny": "",
    "cost_input_per_million": "", "cost_cached_per_million": "", "cost_output_per_million": "",
    "cost_image_tokens": "0",
    "pipeline": "express", "express_final_check": "1", "shot_length": "long", "news_mode": "1", "news_migrated": "",
    "subtitle_font": "black", "subtitle_size": "84", "subtitle_color": "white",
    "news_crawler_dir": "", "news_crawler_command": "", "news_crawler_output": "",
    "news_crawler_limit": "5", "news_crawler_platforms": "",
}
SECRETS = {k for k in DEFAULTS if k.endswith("api_key")}
_lock = threading.Lock()


# A package may carry keys for free services the vendor provides (defaults.json beside launch.py);
# anything the user saves in Settings takes precedence.
BUNDLED_KEYS = {"pixabay_api_key"}
_bundled_cache = {}


def bundled() -> dict:
    path = paths.ROOT / "defaults.json"
    if path not in _bundled_cache:
        try:
            raw = json.loads(path.read_text("utf-8")) if path.is_file() else {}
        except (OSError, ValueError):
            raw = {}
        _bundled_cache[path] = {k: v.strip() for k, v in raw.items()
                                if k in BUNDLED_KEYS and isinstance(v, str) and v.strip()} if isinstance(raw, dict) else {}
    return _bundled_cache[path]


def read() -> dict:
    p = paths.DATA / "settings.json"
    data = DEFAULTS | bundled() | (json.loads(p.read_text("utf-8")) if p.exists() else {})
    for key, value in bundled().items():
        if not data.get(key):       # an empty saved field does not hide the bundled key
            data[key] = value
    if data["search_provider"] not in ("mixed", "video", "local"):
        data["search_provider"] = "mixed"
    return data


def migrate() -> None:
    """Once per change of a default: a Baidu 一镜 choice saved before 2026-10-14 (when it could not be used on
    Windows) and then a saved 千问配音 (paid per character) move to the free voice; 新闻模式, off while it was an
    experiment (0.4.3), is turned on once (0.4.4). Settings saved by this version carry the flags (update()),
    so a choice made afterwards, 百度一镜 or 新闻模式 off included, is kept."""
    p = paths.DATA / "settings.json"
    if not p.exists():
        return
    with _lock:
        data = json.loads(p.read_text("utf-8"))
        if data.get("tts_migrated") and data.get("tts_migrated_edge") and data.get("news_migrated"):
            return
        if not data.get("tts_migrated") and data.get("tts_provider") in ("baidu_web", "baidu_http"):
            data["tts_provider"] = "qwen_tts"
        if not data.get("tts_migrated_edge") and data.get("tts_provider") == "qwen_tts":
            data["tts_provider"] = "edge_tts"
        if not data.get("news_migrated"):
            data["news_mode"] = "1"
        data["tts_migrated"] = data["tts_migrated_edge"] = data["news_migrated"] = "1"
        temporary = p.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(p)


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
    if values.get("tts_provider", "system") not in ("system", "edge_tts", "qwen_tts", "compatible", "import", "baidu_web", "baidu_http"):
        raise ValueError("配音方式无效")
    if values.get("baidu_browser", "auto") not in ("auto", "edge", "chrome"):
        raise ValueError("一镜专用浏览器须为 auto、edge 或 chrome")
    if len(values.get("baidu_voice_name", "").strip()) > 60:
        raise ValueError("一镜音色名称太长")
    from . import edge_voice, qwen_tts
    if values.get("edge_tts_voice", edge_voice.DEFAULT_VOICE) not in edge_voice.NAMES:
        raise ValueError("免费配音音色无效")
    if values.get("qwen_tts_voice", qwen_tts.DEFAULT_VOICE) not in qwen_tts.NAMES:
        raise ValueError("千问音色无效")
    if values.get("tts_speed", "1.0") not in qwen_tts.SPEEDS:
        raise ValueError("语速须为 1.0、1.1、1.2 或 1.3")
    if values.get("search_provider", "mixed") not in ("mixed", "video", "local"):
        raise ValueError("素材检索仅支持多源检索、视频检索或本地素材")
    for key in ("plan_workers", "match_workers", "render_workers"):
        if key in values and values[key] not in ("1", "2", "3", "4"):
            raise ValueError("并发数须为 1 到 4")
    if "video_workers" in values and values["video_workers"] not in ("1", "2", "3", "4", "5", "6"):
        raise ValueError("同时制作的视频须为 1 到 6 条")
    if values.get("news_mode", "0") not in ("0", "1"):
        raise ValueError("新闻模式开关须为 0 或 1")
    if values.get("shot_length", "long") not in ("long", "medium", "line"):
        raise ValueError("镜头长度须为 long、medium 或 line")
    if values.get("pipeline", "express") not in ("express", "classic"):
        raise ValueError("制作流程须为 express 或 classic")
    if values.get("auto_save_exports", "1") not in ("0", "1"):
        raise ValueError("自动保存开关须为 0 或 1")
    if values.get("export_dir", "") and not os.path.isabs(os.path.expanduser(values["export_dir"])):
        raise ValueError("成片保存位置须是完整的文件夹路径")
    if values.get("express_final_check", "1") not in ("0", "1"):
        raise ValueError("成片抽检开关须为 0 或 1")
    from . import subtitle_style
    if values.get("subtitle_font", "black") not in subtitle_style.FONTS:
        raise ValueError("字幕字体无效")
    if values.get("subtitle_size", "84") not in subtitle_style.SIZES:
        raise ValueError("字幕字号无效")
    if values.get("subtitle_color", "white") not in subtitle_style.COLOURS:
        raise ValueError("字幕颜色无效")
    if values.get("plan_retries", "1") not in ("0", "1"):
        raise ValueError("分镜格式重试次数须为 0 或 1")
    if values.get("source_plan_mode", "legacy") not in ("legacy", "unified"):
        raise ValueError("源文规划方式无效")
    if values.get("pixel_observer_mode", "separate") not in ("separate", "joint"):
        raise ValueError("画面观察方式无效")
    paths.prepare()
    with _lock:
        data = read() | values
        # Whatever is saved now is the user's current choice: the one-time migration must never undo it
        # (without the flags, a 百度一镜 or 千问 choice saved before the next start would be moved to the free voice).
        data["tts_migrated"] = data["tts_migrated_edge"] = data["news_migrated"] = "1"
        from .tts_costs import validate_pricing
        validate_pricing(data["tts_pricing_json"])
        for key in ("model_project_budget_cny", "model_daily_budget_cny", "model_provider_budget_cny", "model_job_budget_cny",
                    "tts_daily_budget_cny", "index_daily_budget_cny"):
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
