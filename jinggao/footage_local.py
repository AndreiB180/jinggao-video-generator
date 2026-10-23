"""Describe preview frames with a local vision model, for overnight index builds on a GPU machine.

The model runs in llama.cpp's server (OpenAI-compatible /v1/chat/completions) on the same machine; see
scripts/footage_local.py. One request per video: its frames in time order, each answered with a line,
a picture type and a few search words. No model fees, so every frame of every video gets a full answer.
"""
from __future__ import annotations

import json
import re

import httpx

from .footage_index import _kind

PROMPT = """这些图片是同一个 B 站视频按时间顺序截的预览帧，依次是图1、图2……。逐张独立地看，每张给三项：
1. 描述：最多16个字，写主体、动作、场景。人物写清几个人、男女老少、表情；能看出时代（古代/民国/现代）、地方（中国/外国、城市/农村、医院/法庭/客厅/办公室……）、事物的写上。
2. 类型，只能是：实拍、影视剧、动画、图表文字、主持人出镜、黑屏或模糊。
   影视剧=电视剧、电影、短剧画面；主持人出镜=有人正对镜头讲话（主播、演播室、UP主口播）；图表文字=画面以文字、图表、截图、地图为主；动画=动画、CG、游戏画面。
3. 关键词：3到5个检索词，空格隔开，带常见的同义说法（例如：吵架 争执 夫妻矛盾 愤怒）。
只写画面里看得见的东西。标题只用来认出画面里确实出现的剧名、地名、人物、事物；不要根据标题或字幕编剧情、猜情绪（“怀念过去”“机会出现”这类不要写），不抄画面上的字幕。
只输出 JSON：{"1":["描述","类型","关键词"],"2":[...],...}，每张图都要有。"""
MAX_TEXT = 60
LABEL = re.compile(r"^\s*(描述|类型|关键词)\s*[:：]\s*")
ENTRY = re.compile(r'"[^"\d]*(\d+)"\s*:\s*\[\s*"([^"]*)"\s*,\s*"([^"]*)"\s*(?:,\s*"([^"]*)"\s*)?\]')


def _jpeg(image, quality=85) -> str:
    import base64
    import io
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def parse(answer: str, count: int) -> dict:
    """The model's JSON answer -> {frame index (from 0): (text, kind)}; text is the line and its search
    words ("描述｜关键词"), both searchable. Unusable entries are left out."""
    text = re.sub(r"<think>.*?</think>", "", answer or "", flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1]) if 0 <= start < end else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        # Cut off at the length limit or otherwise broken: keep the entries that are whole.
        data = {m[0]: [m[1], m[2], m[3]] for m in ENTRY.findall(text)}
    out = {}
    for key, value in data.items() if isinstance(data, dict) else []:
        number = re.sub(r"\D", "", str(key))
        if not number or not 1 <= int(number) <= count or not value:
            continue
        if isinstance(value, dict):      # {"描述": …, "类型": …, "关键词": …}
            value = [value.get(k, "") for k in ("描述", "类型", "关键词")]
        value = value if isinstance(value, list) else [str(value)]
        value = [LABEL.sub("", " ".join(v) if isinstance(v, list) else str(v)) for v in value]
        line = re.sub(r"[“”\"\[\]|｜]", "", value[0]).strip(" ，,。")[:24]
        kind = _kind(value[1]) if len(value) > 1 else ""
        words = " ".join(re.split(r"[\s,，、;；/]+", value[2].strip()))[:MAX_TEXT] if len(value) > 2 else ""
        if line:
            out[int(number) - 1] = ((f"{line}｜{words}" if words else line)[:MAX_TEXT], kind)
    return out


def describer(url: str, model: str = "local", timeout: float = 240):
    """describer(row, images) -> {frame index: (text, kind)} through the server at url (…/v1)."""
    client = httpx.Client(timeout=timeout, trust_env=False)
    def describe(row, images):
        content = [{"type": "text", "text": f"视频标题：{str(row.get('title') or '')[:80]}\n共 {len(images)} 张。"}]
        for n, image in enumerate(images, 1):
            content += [{"type": "text", "text": f"图{n}"}, {"type": "image_url", "image_url": {"url": _jpeg(image)}}]
        body = {"model": model, "temperature": 0.2, "max_tokens": 120 + 70 * len(images),
                "chat_template_kwargs": {"enable_thinking": False}, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": content}]}
        response = client.post(url.rstrip("/") + "/chat/completions", json=body)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        answer = parse(message.get("content") or "", len(images))
        if not answer:
            raise ValueError(f"本地模型没有给出可用的描述：{(message.get('content') or '')[:120]}")
        return answer
    return describe
