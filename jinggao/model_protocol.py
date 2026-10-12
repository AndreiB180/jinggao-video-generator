"""Provider options for bounded JSON tasks, shared by all model transports."""
import base64
import io
from urllib.parse import urlparse


def _bound_inline_images(payload):
    """Declare a real server pixel cap without resizing the encoded picture.

    Aliyun's Qwen3.7 grid is 32 pixels; rounding each side upward preserves
    the provider's normal rounded dimensions. Oversized images retain its
    existing default cap. See https://help.aliyun.com/zh/model-studio/vision.
    """
    from PIL import Image
    for message in payload.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_url" or "max_pixels" in block:
                continue
            details = block.get("image_url", {})
            if (not isinstance(details, dict) or any(part.get("vl_high_resolution_images")
                    for part in (payload, block, details))):
                continue
            url = details.get("url", "")
            if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                continue
            try:
                raw = base64.b64decode(url.split(";base64,", 1)[1], validate=True)
                with Image.open(io.BytesIO(raw)) as picture:
                    if getattr(picture, "n_frames", 1) != 1:
                        continue
                    width, height = picture.size
                if width <= 10 or height <= 10:
                    continue
                rounded_area = ((width + 31) // 32) * ((height + 31) // 32) * 1024
                block["max_pixels"] = min(2_621_440, max(4_096, rounded_area))
            except (ValueError, OSError, Image.DecompressionBombError):
                # Unknown image contents retain the normal conservative cap.
                continue


def apply(payload, cfg, *, reasoning=False, reasoning_effort=None):
    host = (urlparse(cfg["model_base_url"]).hostname or "").lower()
    if host.removeprefix("api.") == "deepseek.com":
        payload["thinking"] = {"type": "enabled" if reasoning else "disabled"}
        if reasoning and reasoning_effort is not None:
            if reasoning_effort not in {"low", "high", "max"}:
                raise ValueError("模型思考强度须为low、high或max")
            payload["reasoning_effort"] = reasoning_effort
    elif host == "dashscope.aliyuncs.com" and payload["model"].startswith("qwen"):
        # Qwen defaults to thinking. The paid A/B evaluation did not improve
        # these bounded JSON reviews, so retain the lower-latency mode.
        # DeepSeek-specific reasoning flags must not leak to this provider.
        payload["enable_thinking"] = False
        # These transports extract structured facts and make editorial checks.
        # The provider's creative default makes repeat reviews fluctuate.
        payload["temperature"] = 0
        if payload["model"] in {"qwen3.7-flash", "qwen3.7-flash-2026-07-15"}:
            _bound_inline_images(payload)
