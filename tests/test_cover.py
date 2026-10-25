"""封面: title layout and the 标题 line taken out of narration."""
from PIL import Image

from jinggao import cover, script_norm


def test_the_headline_line_is_not_narrated():
    body, title = script_norm.split_title("第一句。\n第二句。\n\n**标题：菲船闯仁爱礁被拦，这出戏还能演多久？**")
    assert body == "第一句。\n第二句。" and title == "菲船闯仁爱礁被拦，这出戏还能演多久？"
    assert script_norm.split_title("他说标题：不重要。") == ("他说标题：不重要。", "")
    script, info = script_norm.normalize("今天下雨了。\n标题：下雨")
    assert "标题" not in script and info["title"] == "下雨"


def test_titles_break_at_clauses_and_long_ones_end_with_an_ellipsis():
    fits = lambda text: len(text) <= 10
    assert cover.lines("伊朗导弹齐射，美军航母压境，这仗打不打？", fits, 4) == (["伊朗导弹齐射", "美军航母压境", "这仗打不打？"], False)
    rows, broken = cover.lines("一二三四五六七八九十一二三四五", fits, 4)
    assert rows == ["一二三四五六七八九十", "一二三四五"] and broken
    rows, broken = cover.lines("第一句话，第二句话，第三句话，第四句话", lambda t: len(t) <= 5, 2)
    assert rows == ["第一句话", "第二句话…"] and broken
    assert cover.headline("标题：“凌晨导弹来袭。”") == "凌晨导弹来袭"


def test_the_title_is_split_for_the_cover_without_a_model():
    assert cover.plan_text("标题：伊朗30枚导弹齐射，美军三航母压境，中东这仗到底打不打得起来？") == {
        "tag": "深度解读", "main": "伊朗30枚导弹齐射", "sub": "美军三航母压境", "ask": "中东这仗到底打不打得起来？"}
    assert cover.plan_text("一句话") == {"tag": "深度解读", "main": "一句话", "sub": "", "ask": ""}
    rows, _ = cover._halves("伊朗30枚导弹齐射", 1150, 200, 60)
    assert rows == ["伊朗30枚", "导弹齐射"]                       # never between 30 and 枚
    assert cover._halves("六枚导弹｜砸向也门总统府", 1150, 200, 60)[0] == ["六枚导弹", "砸向也门总统府"]


def test_designed_covers_in_both_shapes():
    picture = Image.new("RGB", (1920, 1080), (90, 120, 150))
    parts = {"tag": "国际焦点", "main": "伊朗30枚导弹齐射", "sub": "美军三航母压境", "ask": "中东这仗到底打不打得起来？"}
    land, port = cover.design(picture, parts, "landscape"), cover.design(picture, parts, "portrait")
    assert land.size == (1920, 1080) and port.size == (1080, 1920)
    assert land.getpixel((1900, 540)) != land.getpixel((60, 540))        # text side darkened, picture side not


def _fake_service(monkeypatch, outcome="succeeded"):
    import io
    import json
    import httpx
    from jinggao import network
    seen = []
    png = io.BytesIO()
    Image.new("RGB", (64, 36), (200, 30, 30)).save(png, "PNG")
    def handler(request):
        seen.append((request.method, request.url.path, request.headers.get("authorization"),
                     json.loads(request.content) if request.content else None))
        if request.url.path == "/v1/api/generate":
            return httpx.Response(200, json={"id": "t1", "status": "running", "progress": 0})
        if request.url.path == "/v1/api/result":
            return httpx.Response(200, json={"id": "t1", "status": outcome, "error": "bad words",
                                             "results": [{"url": "https://files.example/c.png"}]})
        return httpx.Response(200, content=png.getvalue())
    monkeypatch.setattr(network, "client", lambda **kw: httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(cover.time, "sleep", lambda s: None)
    return seen


def test_ai_covers_are_drawn_in_both_shapes_with_the_title(tmp_path, monkeypatch):
    seen = _fake_service(monkeypatch)
    cfg = {"cover_mode": "ai", "cover_api_key": "k", "cover_sizes": "both", "cover_reference": "1",
           "cover_prompt_landscape": "百家号封面：{标题}", "cover_prompt_portrait": ""}
    picture = Image.new("RGB", (1920, 1080), (9, 9, 9))
    result = cover.render(cfg, picture, cover.plan_text("伊朗导弹齐射？"), "标题：伊朗导弹齐射？", tmp_path, "片名")
    assert [p.split("/")[-1] for p in result["files"]] == ["片名-封面-横版.jpg", "片名-封面-竖版.jpg"] and not result["error"]
    asks = [body for method, path, auth, body in seen if path == "/v1/api/generate"]
    assert all(auth == "Bearer k" for _, path, auth, _ in seen if path.startswith("/v1/api"))
    assert asks[0]["prompt"].startswith("百家号封面：伊朗导弹齐射？") and asks[0]["aspectRatio"] == "16:9"
    assert asks[1]["aspectRatio"] == "9:16" and cover.PROMPT in asks[1]["prompt"] and asks[1]["images"][0].startswith("data:image/jpeg")


def test_a_refused_ai_cover_is_designed_locally_instead(tmp_path, monkeypatch):
    _fake_service(monkeypatch, outcome="violation")
    cfg = {"cover_mode": "ai", "cover_api_key": "k", "cover_sizes": "landscape", "cover_reference": "0"}
    picture = Image.new("RGB", (1920, 1080), (90, 120, 150))
    result = cover.render(cfg, picture, cover.plan_text("伊朗导弹齐射"), "伊朗导弹齐射", tmp_path, "片名")
    assert [p.split("/")[-1] for p in result["files"]] == ["片名-封面-横版.jpg"] and "审核" in result["error"]
    result = cover.render({"cover_mode": "design", "cover_sizes": "portrait"}, picture, cover.plan_text("标题"), "标题",
                          tmp_path, "片2")
    assert [p.split("/")[-1] for p in result["files"]] == ["片2-封面-竖版.jpg"] and not result["error"]


def test_the_model_picks_the_frame_and_words_and_its_absence_falls_back(monkeypatch):
    from jinggao import planner
    frames = [Image.new("RGB", (320, 180), (20 * i, 60, 90)) for i in range(1, 5)]
    monkeypatch.setattr(planner, "chat", lambda *a, **k: ({"frame": 3, "tag": "国际焦点", "main": "六枚导弹｜砸向总统府",
                                                           "sub": "内阁险遭波及", "ask": "还能绷多久？"}, {}))
    picture, parts = cover.choose("凌晨六枚导弹砸向总统府，内阁险遭波及，还能绷多久？", frames)
    assert picture is frames[2] and parts["main"] == "六枚导弹｜砸向总统府" and parts["tag"] == "国际焦点"
    def down(*a, **k):
        raise planner.ModelUnavailable("没有额度")
    monkeypatch.setattr(planner, "chat", down)
    picture, parts = cover.choose("凌晨六枚导弹砸向总统府，还能绷多久？", frames)
    assert picture in frames and parts == {"tag": "深度解读", "main": "凌晨六枚导弹砸向总统府", "sub": "", "ask": "还能绷多久？"}
    assert cover.choose("标题", []) == (None, cover.plan_text("标题"))


def test_one_ai_background_is_lettered_locally_in_both_shapes(tmp_path, monkeypatch):
    seen = _fake_service(monkeypatch)
    cfg = {"cover_mode": "ai_bg", "cover_api_key": "k", "cover_sizes": "both", "cover_reference": "1"}
    picture = Image.new("RGB", (1920, 1080), (9, 9, 9))
    result = cover.render(cfg, picture, cover.plan_text("伊朗导弹齐射？"), "伊朗导弹齐射？", tmp_path, "片名")
    asks = [body for method, path, auth, body in seen if path == "/v1/api/generate"]
    assert len(asks) == 1 and asks[0]["model"] == "nano-banana-2-lite" and asks[0]["imageSize"] == "1K"
    assert "不要出现任何文字" in asks[0]["prompt"] and "伊朗导弹齐射？" in asks[0]["prompt"]
    assert [p.split("/")[-1] for p in result["files"]] == ["片名-封面-横版.jpg", "片名-封面-竖版.jpg"] and not result["error"]
    landscape = Image.open(result["files"][0])
    assert landscape.size == (1920, 1080) and landscape.getpixel((1800, 300))[0] > 100   # the AI picture (red) behind


def test_the_older_interface_is_tried_when_the_new_one_refuses(tmp_path, monkeypatch):
    import io
    import json
    import httpx
    from jinggao import network
    png = io.BytesIO()
    Image.new("RGB", (64, 36), (200, 30, 30)).save(png, "PNG")
    paths_seen = []
    def handler(request):
        paths_seen.append(request.url.path)
        if request.url.path == "/v1/api/generate":
            return httpx.Response(400, json={"id": "", "status": "failed", "error": "unknown parameter images"})
        if request.url.path == "/v1/draw/completions":
            body = json.loads(request.content)
            assert body["webHook"] == "-1" and body["size"] == "16:9" and body["urls"]
            return httpx.Response(200, json={"code": 0, "data": {"id": "old1"}, "msg": "success"})
        if request.url.path == "/v1/draw/result":
            return httpx.Response(200, json={"code": 0, "data": {"id": "old1", "status": "succeeded",
                                                                "results": [{"url": "https://files.example/c.png"}]}})
        return httpx.Response(200, content=png.getvalue())
    monkeypatch.setattr(network, "client", lambda **kw: httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(cover.time, "sleep", lambda s: None)
    image = cover.ai_picture({"cover_api_key": "k", "cover_mode": "ai"}, "画", "landscape", [Image.new("RGB", (32, 18))])
    assert image.size == (64, 36) and paths_seen[:3] == ["/v1/api/generate", "/v1/draw/completions", "/v1/draw/result"]


def test_a_timed_out_background_is_drawn_again_by_gpt_image(tmp_path, monkeypatch):
    calls = []
    def drawn(cfg, prompt, shape, references):
        calls.append(cover.model_for(cfg))
        if len(calls) == 1:
            raise cover.CoverError("封面生成失败：google gemini timeout")
        return Image.new("RGB", (1672, 941), (200, 30, 30))
    monkeypatch.setattr(cover, "ai_picture", drawn)
    cfg = {"cover_mode": "ai_bg", "cover_api_key": "k", "cover_sizes": "landscape", "cover_reference": "0"}
    result = cover.render(cfg, Image.new("RGB", (1920, 1080)), cover.plan_text("标题"), "标题", tmp_path, "片")
    assert calls == ["nano-banana-2-lite", "gpt-image-2"] and not result["error"] and len(result["files"]) == 1
