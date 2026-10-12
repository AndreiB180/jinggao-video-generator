import base64
from copy import deepcopy
import io
import math

from PIL import Image
import pytest

from jinggao import config, costs, model_protocol


CFG = config.DEFAULTS | {"model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"}


def request(size=(960, 540), count=1):
    data = io.BytesIO()
    Image.new("RGB", size).save(data, "JPEG")
    url = "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()
    return {"model": "qwen3.7-flash", "max_tokens": 3500, "messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}
                                     for _ in range(count)]}]}


def reference_resize(size, maximum):
    # Aliyun's documented smart_resize, used as an independent size check.
    width, height = size
    h, w = round(height / 32) * 32, round(width / 32) * 32
    if h * w > maximum:
        factor = math.sqrt(height * width / maximum)
        h, w = math.floor(height / factor / 32) * 32, math.floor(width / factor / 32) * 32
    elif h * w < 4096:
        factor = math.sqrt(4096 / (height * width))
        h, w = math.ceil(height * factor / 32) * 32, math.ceil(width * factor / 32) * 32
    return h, w


@pytest.mark.parametrize("size", [(960, 540), (1280, 720), (720, 1280), (853, 480), (960, 960),
                                  (1280, 1280), (17, 17), (11, 2000), (3840, 2160)])
def test_declared_cap_keeps_image_bytes_and_normal_provider_dimensions(size):
    payload = request(size)
    before = deepcopy(payload["messages"])
    model_protocol.apply(payload, CFG)
    image = payload["messages"][0]["content"][0]
    assert image["image_url"] == before[0]["content"][0]["image_url"]
    assert reference_resize(size, image["max_pixels"]) == reference_resize(size, 2_621_440)
    assert image["max_pixels"] % 1024 == 0


@pytest.mark.parametrize("mode", ["remote", "invalid", "explicit", "high", "other_model", "other_provider"])
def test_unknown_or_explicit_image_policy_keeps_conservative_reservation(mode):
    payload = request()
    image = payload["messages"][0]["content"][0]
    cfg = CFG
    if mode == "remote":
        image["image_url"]["url"] = "https://example.org/preview.jpg"
    elif mode == "invalid":
        image["image_url"]["url"] = "data:image/jpeg;base64,invalid!"
    elif mode == "explicit":
        image["max_pixels"] = 1_310_720
    elif mode == "high":
        payload["vl_high_resolution_images"] = True
    elif mode == "other_model":
        payload["model"] = "qwen-vl-plus"
    else:
        cfg = CFG | {"model_base_url": "https://example.org/v1"}
    before = deepcopy(payload["messages"])
    model_protocol.apply(payload, cfg)
    assert payload["messages"] == before


def test_thumbnail_batch_guard_uses_the_same_explicit_cap_sent_to_provider():
    payload = request(count=12)
    before = costs._input_bound(CFG, payload)
    model_protocol.apply(payload, CFG)
    after = costs._input_bound(CFG, payload)
    assert after < before
    assert all(block["max_pixels"] == 522_240 for block in payload["messages"][0]["content"])
    assert costs._amount(costs._rates(CFG, payload["model"], after), after, 3500) < costs._amount(
        costs._rates(CFG, payload["model"], before), before, 3500)
