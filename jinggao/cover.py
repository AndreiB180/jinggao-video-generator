"""封面: covers for a finished film, saved next to its MP4.

Setting cover_mode:
- "design" (default, about ¥0.001): the low-cost vision model picks the background among frames of the
  opening shots (taken from the footage itself, so no burned-in subtitles) and splits the title into a
  tag, a big line, a second line and a question; the cover is then drawn locally in the 百家号 look: red
  tag, gold main line, silver second line, the question on a yellow hazard bar.
- "ai_bg": the image model draws one picture without lettering (the cheapest model, about ¥0.02), and both
  shapes are designed on it: an AI-drawn cover for a cent or two a film, with a title that cannot be misspelt.
- "ai": the image model draws each cover whole, title included (GrsAI gpt-image-2, ¥0.03-0.06 a picture),
  from the user's prompts.
Both AI ways get the chosen frame as a reference picture, run in the background, and fall back to the
design above for whatever fails.
- "off".
Landscape 16:9 and/or portrait 9:16 (cover_sizes).
"""
from __future__ import annotations

import base64
import io
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat

from . import paths

FONT = "NotoSansSC-Black.otf"
PROMPT = ("这个是我的自媒体选题标题，需要展示出主题文字钩子，这个封面是用在百家号这个平台的，"
          "整体风格喜好要对应这个平台的人群")
SHAPES = {"landscape": ("16:9", "横版", (1920, 1080)), "portrait": ("9:16", "竖版", (1080, 1920))}
BACKGROUND = ("为这期视频画一张百家号封面的背景图。画面里不要出现任何文字、字母、数字、标志和水印。"
              "主体人物或场景放在画面右侧，左侧留出较暗、干净的区域放标题。写实、电影感、冲击力强，"
              "符合百家号读者的喜好。")
AI_WAIT = 240          # seconds one AI cover may take
SAMPLES = 5            # opening shots looked at for the picture
YELLOW, WHITE, BLACK = (255, 222, 0), (255, 255, 255), (0, 0, 0)


def headline(title: str) -> str:
    """The title without a "标题：" label, quotes or a trailing full stop."""
    text = re.sub(r"^\s*(标题|题目)\s*[:：]\s*", "", title or "").strip()
    return text.strip("“”\"'《》 ").rstrip("。.")


def lines(text: str, fits, most: int) -> tuple[list[str], bool]:
    """Clauses packed into lines that fit (fits(text) -> bool, by measured width), breaking inside a clause
    only when it is wider than a line; the comma closing a line is dropped. More than most lines: the
    rest is cut with an ellipsis. Returns (lines, whether a clause had to be broken or text cut)."""
    clauses = [c for c in re.findall(r"[^，,、；;：:！!？?。]+[，,、；;：:！!？?。]?", text) if c.strip()]
    out, current, broken = [], "", False
    for clause in clauses:
        trial = current + clause
        if fits(trial.rstrip("，,、；;：:")):
            current = trial
            continue
        if current:
            out.append(current)
            current = ""
        while not fits(clause.rstrip("，,、；;：:")):
            broken = True
            cut = len(clause) - 1
            while cut > 1 and not fits(clause[:cut]):
                cut -= 1
            out.append(clause[:cut])
            clause = clause[cut:]
        current = clause
    if current:
        out.append(current)
    out = [line.rstrip("，,、；;：: ") for line in out]
    if len(out) > most:
        broken = True
        last = out[most - 1]
        while last and not fits(last + "…"):
            last = last[:-1]
        out = out[:most - 1] + [last.rstrip("，,、；;：: ") + "…"]
    return [line for line in out if line], broken


def _frame(video: Path, at: float, size: tuple[int, int]) -> Image.Image | None:
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "frame.jpg"
        done = subprocess.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", str(video),
                               "-frames:v", "1", "-vf", f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=increase,"
                               f"crop={size[0]}:{size[1]}", str(target)], capture_output=True, timeout=60)
        if done.returncode != 0 or not target.exists():
            return None
        with Image.open(target) as image:
            return image.convert("RGB")


def _score(image: Image.Image) -> float:
    """Sharp, bright and colourful; dark or flat frames score nothing."""
    small = image.resize((320, round(320 * image.height / image.width)))
    light = ImageStat.Stat(small.convert("L")).mean[0]
    if light < 45:
        return 0.0
    edges = ImageStat.Stat(small.convert("L").filter(ImageFilter.FIND_EDGES)).mean[0]
    colour = ImageStat.Stat(small.convert("HSV").split()[1]).mean[0]
    return edges * (1 + colour / 255) * min(1.0, light / 90)


def pick_frame(video: Path, starts: list[float], size: tuple[int, int]) -> Image.Image | None:
    best = None
    for at in starts[:SAMPLES]:
        image = _frame(video, at, size)
        if image is not None and (best is None or _score(image) > best[0]):
            best = (_score(image), image)
    return best[1] if best else None


TAGS = ("深度解读", "国际焦点", "深度揭秘", "真相调查", "历史档案", "财经观察", "社会热点", "人物故事", "军事前沿",
        "娱乐速递", "体育风云", "独家")
GOLD = ((255, 246, 196), (255, 196, 40), (236, 92, 10))
SILVER = ((255, 255, 255), (224, 228, 235), (170, 178, 190))


def plan_text(title: str) -> dict:
    """Title split into the cover's parts without a model: the first clause is the big line, a question
    goes on the yellow bar, what is left is the second line."""
    text = headline(title)
    clauses = [c.strip("，,、；;：: ") for c in re.findall(r"[^，,、；;：:！!？?。]+[！!？?]?", text) if c.strip("，,、；;：: ")]
    if not clauses:
        return {"tag": "深度解读", "main": "精彩内容", "sub": "", "ask": ""}
    ask = clauses.pop() if len(clauses) > 1 and clauses[-1][-1:] in "？?！!" else ""
    main = clauses.pop(0)
    return {"tag": "深度解读", "main": main, "sub": "，".join(clauses), "ask": ask}


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(paths.ROOT / "assets" / "fonts" / FONT), size)


def _fit(text: str, width: int, start: int, smallest: int) -> ImageFont.FreeTypeFont:
    size = start
    while size > smallest and _font(size).getlength(text) > width:
        size -= 4
    return _font(size)


def _wrap(text: str, width: int, start: int, smallest: int, most: int) -> tuple[list[str], ImageFont.FreeTypeFont]:
    """text in at most `most` lines at the biggest size from start down that fits width."""
    size = start
    while True:
        font = _font(size)
        rows, broken = lines(text, lambda t: font.getlength(t) <= width, most)
        if not broken or size <= smallest:
            return rows, font
        size -= 6


def _cuts(text: str) -> list[int]:
    """Where a line may break: between words (jieba when installed), never inside a number and the word
    after it (30枚, 8.55亿)."""
    try:
        import jieba
        words = jieba.lcut(text)
    except Exception:   # noqa: BLE001 — no segmenter: any character boundary
        words = list(text)
    cuts, at = [], 0
    for word in words[:-1]:
        at += len(word)
        if not (text[at - 1].isdigit() or text[at - 1] in ".．"):
            cuts.append(at)
    return cuts


def _halves(text: str, width: int, size: int, smallest: int) -> tuple[list[str], ImageFont.FreeTypeFont]:
    """The big line at about `size`: one line when it fits, else two halves split between words near the
    middle (a comma first), as large as the width allows up to `size`."""
    if "｜" in text:     # the break the model marked
        rows = [r for r in text.split("｜", 1) if r]
        return rows, _fit(max(rows, key=lambda r: _font(100).getlength(r)), width, size, smallest)
    font = _fit(text, width, size, smallest)
    if font.size >= size * 0.9 or len(text) < 6:
        return [text], font
    middle = len(text) / 2
    cuts = _cuts(text) or [round(middle)]
    cut = min(cuts, key=lambda i: abs(i - middle) - (2 if text[i - 1] in "，,、：:" else 0))
    rows = [text[:cut].rstrip("，,、：: "), text[cut:].lstrip("，,、：: ")]
    longest = max(rows, key=lambda r: _font(100).getlength(r))
    return rows, _fit(longest, width, size, smallest)


def _shiny(base: Image.Image, xy, text: str, font, colours, stroke: int, outline=(30, 8, 0), shadow=True):
    """Text with a vertical colour gradient, a thick outline and a soft drop shadow (the 营销号 look)."""
    x, y = xy
    box = font.getbbox(text, stroke_width=stroke)
    w, h = box[2] + stroke * 2, box[3] + stroke * 2
    if shadow:
        glow = Image.new("L", base.size, 0)
        ImageDraw.Draw(glow).text((x + stroke // 2 + 6, y + stroke // 2 + 8), text, font=font, fill=210,
                                  stroke_width=stroke + 4, stroke_fill=210)
        base.paste((0, 0, 0), (0, 0), glow.filter(ImageFilter.GaussianBlur(max(4, stroke))))
    edge = Image.new("L", base.size, 0)
    ImageDraw.Draw(edge).text((x, y), text, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
    base.paste(outline, (0, 0), edge)
    face = Image.new("L", base.size, 0)
    ImageDraw.Draw(face).text((x, y), text, font=font, fill=255)
    ramp = Image.new("RGB", (1, max(2, h)))
    top, mid, low = colours
    for i in range(ramp.height):
        f = i / (ramp.height - 1)
        a, b, t = (top, mid, f * 2) if f < 0.5 else (mid, low, f * 2 - 1)
        ramp.putpixel((0, i), tuple(int(a[k] + (b[k] - a[k]) * t) for k in range(3)))
    fill = Image.new("RGB", base.size)
    fill.paste(ramp.resize((base.width, h)), (0, max(0, y + box[1] - stroke)))
    base.paste(fill, (0, 0), face)
    return w, h


def _bottom_fade(base: Image.Image, top: int, strength: int = 235) -> Image.Image:
    """Darken from `top` to the bottom edge: the subtitle line that some footage has burned in sits there."""
    w, h = base.size
    column = Image.new("L", (1, h), 0)
    for y in range(top, h):
        column.putpixel((0, y), int(strength * min(1.0, (y - top) / max(1, (h - top) * 0.55))))
    return Image.composite(Image.new("RGB", (w, h)), base, column.resize((w, h)))


def _grade(picture: Image.Image) -> Image.Image:
    """A little more contrast and colour, a little darker: the dramatic look of these covers."""
    from PIL import ImageEnhance
    picture = ImageEnhance.Contrast(picture).enhance(1.15)
    picture = ImageEnhance.Color(picture).enhance(1.15)
    return ImageEnhance.Brightness(picture).enhance(0.9)


def _shade(size, stops) -> Image.Image:
    """A mask darkening along x (or y): stops [(position 0-1, alpha 0-255), ...]."""
    length = size[0]
    row = Image.new("L", (length, 1))
    for i in range(length):
        f = i / max(1, length - 1)
        for (p0, a0), (p1, a1) in zip(stops, stops[1:], strict=False):
            if p0 <= f <= p1:
                row.putpixel((i, 0), int(a0 + (a1 - a0) * (f - p0) / max(1e-6, p1 - p0)))
                break
    return row


def _tag(base, xy, text, size):
    font = _font(size)
    x, y = xy
    w = int(font.getlength(text)) + size
    h = int(size * 1.45)
    canvas = ImageDraw.Draw(base)
    canvas.polygon([(x, y), (x + w + size // 2, y), (x + w, y + h), (x - size // 2, y + h)], fill=(214, 20, 28))
    canvas.text((x + size // 2, y + (h - size) // 2 - size // 12), text, font=font, fill=(255, 255, 255))
    return h


def _bar(base, xy, text, width, size):
    """The question on a yellow bar with black hazard stripes at its ends."""
    font = _fit(text, width - size, size, 40)
    x, y = xy
    w = int(font.getlength(text)) + font.size
    h = int(font.size * 1.5)
    canvas = ImageDraw.Draw(base)
    canvas.rectangle((x, y, x + w, y + h), fill=(255, 212, 0))
    stripe = max(8, h // 4)
    for sx in range(-h, w, stripe * 2):
        for edge in (y, y + h - stripe // 2):
            canvas.polygon([(x + sx, edge + stripe // 2), (x + sx + stripe, edge), (x + sx + stripe * 2, edge),
                            (x + sx + stripe, edge + stripe // 2)], fill=(20, 20, 20))
    canvas.rectangle((x, y, x + w, y + h), outline=(20, 20, 20), width=max(3, h // 18))
    canvas.text((x + font.size // 2, y + (h - font.size) // 2 - font.size // 12), text, font=font, fill=(10, 10, 10))
    return h


def _blocks(parts: dict, width: int, scale: float, sizes: dict):
    """The title's lines at this scale: [(rows, font, kind)] and their total height."""
    blocks = []
    rows, font = _halves(parts.get("main") or "精彩内容", width, int(sizes["main"] * scale), 60)
    blocks.append((rows, font, "main"))
    if parts.get("sub"):
        rows, font = _wrap(parts["sub"], width, int(sizes["sub"] * scale), 48, 2)
        blocks.append((rows, font, "sub"))
    height = sum(len(rows) * font.size * 1.14 for rows, font, _ in blocks) + 0.18 * sizes["sub"] * scale * len(blocks)
    return blocks, height


def _draw_blocks(base, blocks, left: int | None, y: int, width: int) -> int:
    for rows, font, kind in blocks:
        stroke = max(5, font.size // 12)
        for row in rows:
            x = left if left is not None else (base.width - int(font.getlength(row))) // 2
            _shiny(base, (x, y), row, font, GOLD if kind == "main" else SILVER, stroke,
                   outline=(30, 8, 0) if kind == "main" else (0, 0, 0))
            y += int(font.size * 1.14)
        y += int(font.size * 0.18)
    return y


def design(picture: Image.Image, parts: dict, shape: str = "landscape") -> Image.Image:
    """The cover drawn from a frame and the title's parts {tag, main, sub, ask}: a red tag, the main line in
    big gold type, the second line in silver, the question on a yellow hazard bar."""
    w, h = SHAPES[shape][2]
    frame = _grade(picture.convert("RGB"))
    ask_size = 76 if shape == "landscape" else 70
    if shape == "landscape":
        base = ImageOps.fit(frame, (w, h))
        dark = _shade((w, h), [(0, 235), (0.45, 190), (0.72, 40), (1, 20)]).resize((w, h))
        base = Image.composite(Image.new("RGB", (w, h)), base, dark)
        base = _bottom_fade(base, int(h * 0.74))
        left, width = int(w * 0.045), int(w * 0.6)
        y = int(h * 0.07)
        if parts.get("tag"):
            y += _tag(base, (left + 20, y), parts["tag"], 54) + int(h * 0.035)
        room = int(h * (0.74 if parts.get("ask") else 0.93)) - y
        scale = 1.0
        blocks, height = _blocks(parts, width, scale, {"main": 200, "sub": 110})
        while height > room and scale > 0.5:
            scale -= 0.06
            blocks, height = _blocks(parts, width, scale, {"main": 200, "sub": 110})
        y = _draw_blocks(base, blocks, left, y, width)
        if parts.get("ask"):
            _bar(base, (left, max(y + int(h * 0.01), int(h * 0.76))), parts["ask"], int(w * 0.62), ask_size)
        return base
    # Portrait: the frame across the lower part on a blurred, darkened copy of itself; the text above it.
    base = ImageOps.fit(frame, (w, h)).filter(ImageFilter.GaussianBlur(28))
    base = Image.blend(base, Image.new("RGB", (w, h)), 0.45)
    y = int(h * 0.05)
    if parts.get("tag"):
        y += _tag(base, ((w - int(_font(58).getlength(parts["tag"])) - 58) // 2, y), parts["tag"], 58) + int(h * 0.025)
    ask_h = int(ask_size * 1.5 + h * 0.03) if parts.get("ask") else 0
    room = int(h * 0.6) - ask_h - y
    scale = 1.0
    blocks, height = _blocks(parts, int(w * 0.92), scale, {"main": 175, "sub": 100})
    while height > room and scale > 0.5:
        scale -= 0.06
        blocks, height = _blocks(parts, int(w * 0.92), scale, {"main": 175, "sub": 100})
    y = _draw_blocks(base, blocks, None, y, int(w * 0.92))
    if parts.get("ask"):
        font = _fit(parts["ask"], int(w * 0.86), ask_size, 40)
        bar_w = int(font.getlength(parts["ask"])) + font.size
        y += _bar(base, ((w - bar_w) // 2, y + int(h * 0.01)), parts["ask"], int(w * 0.92), ask_size) + int(h * 0.02)
    top = max(y + int(h * 0.02), int(h * 0.45))
    shown = ImageOps.fit(frame, (w, h - top))       # the picture fills the rest down to the bottom edge
    base.paste(shown, (0, top))
    fade = Image.new("L", (1, shown.height))
    for i in range(shown.height):
        fade.putpixel((0, i), int(255 * min(1.0, i / max(1, shown.height * 0.2))))
    base.paste(Image.new("RGB", (w, shown.height)), (0, top), Image.eval(fade.resize((w, shown.height)), lambda v: 255 - v))
    return _bottom_fade(base, int(h * 0.84))


PLAN_SYSTEM = """图片是一条视频开头几个画面的联系表，每格左上角有黄色编号。要为这条视频做百家号封面：
1. frame：挑最适合做封面背景的一格编号。有人物正脸、冲突、爆炸、标志性地点或物件的优先；不要文字多、模糊、黑屏、表情包或只有空镜风景的。
2. 把标题拆成封面上的字，全部用标题里的原词，不编新内容：
   tag：栏目标签，从 国际焦点、深度揭秘、历史档案、财经观察、社会热点、人物故事、军事前沿、娱乐速递、体育风云、真相调查 里选最贴的一个；
   main：最抓人的一句（不超过14个字），超过6个字时用“｜”标出分两行的位置，断在两个词之间，最好断在意思的停顿处；sub：第二句（不超过16个字，可以为空）；ask：标题里的问句或悬念（不超过18个字，可以为空）。
   每一项都是完整的词句，不能截断词语（“另一种颜色”不能写成“另一种颜”）；字数超了就换一句更短的，不要硬截。
   一个短语不要拆到两项里（“这出戏还能演多久？”整句放进 ask）；三项不重复同一段话，main 里有的话 sub、ask 里不再出现。
只输出 JSON：{"frame":1,"tag":"","main":"","sub":"","ask":""}"""


def source_frames(project: dict, count: int = 8) -> list[Image.Image]:
    """A frame from the middle of each of the first shots, taken from the downloaded footage itself (cropped as
    the film shows it): no burned-in subtitles, unlike frames of the finished film."""
    from . import express_clean
    assets = {a.get("id"): a for a in project.get("assets") or []}
    frames = []
    for shot in (project.get("shots") or [])[:count]:
        asset = assets.get(shot.get("selected_asset"))
        if not asset or asset.get("kind") != "video":
            continue
        try:
            source = paths.file_path(project["id"], asset["path"])
        except ValueError:
            continue
        at = float(shot.get("clip_start") or 0) + float(shot.get("duration") or 4) / 2
        crop = express_clean.video_filter(shot.get("frame_crop")) if shot.get("frame_crop") else ""
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "frame.jpg"
            done = subprocess.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", str(source),
                                   "-frames:v", "1", "-vf", (crop + "," if crop else "") + "scale=1920:-2", str(target)],
                                  capture_output=True, timeout=60)
            if done.returncode == 0 and target.exists():
                with Image.open(target) as image:
                    picture = image.convert("RGB")
                if ImageStat.Stat(picture.convert("L")).mean[0] >= 30:
                    frames.append(picture)
    return frames


def choose(title: str, frames: list[Image.Image]) -> tuple[Image.Image | None, dict]:
    """The background frame and the title's parts, chosen by the low-cost vision model (about ¥0.001); without
    a model or when it fails: the sharpest frame and the title split at its commas."""
    parts = plan_text(title)
    if not frames:
        return None, parts
    best = max(frames, key=_score)
    try:
        from . import express_match as em, footage_index, planner
        sheet = footage_index._sheet([{"image": f} for f in frames], 4)
        content = [{"type": "text", "text": f"标题：{headline(title)}\n共 {len(frames)} 格。"},
                   {"type": "image_url", "image_url": {"url": em._jpeg(sheet)}}]
        answer, _ = planner.chat(PLAN_SYSTEM, content, vision=True, max_tokens=200, timeout=60, task="express_pick")
    except Exception:   # noqa: BLE001 — no key, no budget, a refusal: the plain choice
        return best, parts
    if not isinstance(answer, dict):
        return best, parts
    index = answer.get("frame")
    picture = frames[index - 1] if isinstance(index, int) and 1 <= index <= len(frames) else best
    chosen = tidy(answer)
    if not chosen["main"]:
        return picture, parts
    return picture, chosen


PUNCT = "，,、；;：:！!？?。"
QUESTION = re.compile(r"(.)[不没]\1|到底|吗|呢|什么|怎么|为何|为什么|多久|多少|哪|谁|难道|是否|能否")


def _clauses(text: str) -> list[str]:
    return [c for c in re.findall(rf"[^{PUNCT}]+[{PUNCT}]?", text) if c.strip(PUNCT + " ")]


def _shared(a: str, b: str) -> int:
    """Length of the longest run of characters a and b have in common."""
    best = 0
    for i in range(len(a)):
        for j in range(i + best + 1, len(a) + 1):
            if a[i:j] in b:
                best = j - i
            else:
                break
    return best


def _whole(text: str, most: int) -> str:
    """text when it fits most characters, else as many whole clauses from the start as fit ("" when even the
    first is longer): a word is never cut in half (the client got "另一种颜" for "另一种颜色")."""
    if len(text) <= most:
        return text
    kept = ""
    for clause in _clauses(text):
        if len((kept + clause).rstrip(PUNCT)) > most:
            break
        kept += clause
    return kept.rstrip(PUNCT + " ")


def _not_again(text: str, *others: str) -> str:
    """text without the clauses that repeat what the bigger lines already say (4 characters or more in a
    row); a question left without its mark gets it back."""
    kept = "".join(c for c in _clauses(text) if not any(_shared(c.rstrip(PUNCT), o) >= 4 for o in others if o))
    kept = kept.rstrip("，,、；;：: ")
    if len(kept.rstrip(PUNCT)) < 3:
        return ""
    if kept[-1] not in PUNCT and text.rstrip()[-1:] in "？?" or kept[-1] not in PUNCT and QUESTION.search(kept):
        kept += "？"
    return kept


def tidy(answer: dict) -> dict:
    """The model's cover words made safe to draw: whole words and clauses only, the main line's break moved
    off a word's middle, no part repeating another, a known tag."""
    clean = {k: re.sub(r"[“”\"|\s]", "", str(answer.get(k) or "")) for k in ("tag", "main", "sub", "ask")}
    tag = clean["tag"] if clean["tag"] in TAGS else "深度解读"
    main = clean["main"].replace("｜", "|", 1).replace("｜", "")
    left, _, right = main.partition("|")
    plain = (left + right).strip(PUNCT + " ")
    if len(plain) > 20:
        plain = _whole(plain, 20) or plain
        left, right = plain, ""
    if right and plain.startswith(left):
        at = len(left.rstrip(PUNCT + " "))
        fine = left[-1:] in PUNCT or at in _cuts(plain)
        main = (left.rstrip(PUNCT + " ") + "｜" + right.lstrip(PUNCT + " ")) if fine else plain
    else:
        main = plain
    sub = _not_again(_whole(clean["sub"].replace("｜", ""), 18), plain)
    ask = _not_again(_whole(clean["ask"].replace("｜", ""), 22), plain, sub.rstrip(PUNCT))
    return {"tag": tag, "main": main, "sub": sub, "ask": ask}


class CoverError(RuntimeError):
    pass


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def prompt_for(template: str, title: str, aspect_label: str, references: bool) -> str:
    """The user's prompt with the title: in place of {标题} if it has one, else after it."""
    template = (template or PROMPT).strip()
    text = template.replace("{标题}", title) if "{标题}" in template else f"{template}\n\n标题：{title}"
    text += f"\n画面比例：{aspect_label}。"
    if references:
        text += "\n参考图是这期视频里的画面，可以用其中的人物和场景。"
    return text


def model_for(cfg: dict) -> str:
    """The chosen model, else the cheapest that does the job: a background needs no lettering
    (nano-banana-2-lite, about ¥0.02); a whole cover with its Chinese title needs gpt-image-2."""
    return cfg.get("cover_model") or ("nano-banana-2-lite" if cfg.get("cover_mode") == "ai_bg" else "gpt-image-2")


def ai_picture(cfg: dict, prompt: str, shape: str, references: list) -> Image.Image:
    """One picture drawn by the image model (GrsAI's /v1/api/generate, then polling /v1/api/result)."""
    from . import network
    key = (cfg.get("cover_api_key") or "").strip()
    if not key:
        raise CoverError("没有填写封面 API Key")
    aspect = SHAPES[shape][0]
    host = (cfg.get("cover_base_url") or "https://grsai.dakka.com.cn").rstrip("/")
    body = {"model": model_for(cfg), "aspectRatio": aspect, "replyType": "async", "prompt": prompt}
    if body["model"].startswith("nano-banana"):
        body["imageSize"] = "1K"
    if references:
        body["images"] = [_data_url(image) for image in references]
    headers = {"Authorization": f"Bearer {key}"}
    with network.client(timeout=60) as client:
        try:
            url = _drawn(client, host, headers, body)
        except CoverError as exc:
            if "审核" in str(exc):
                raise
            url = _drawn_legacy(client, host, headers, body)    # the older per-model interface
        picture = client.get(url, timeout=120)
        picture.raise_for_status()
    with Image.open(io.BytesIO(picture.content)) as image:
        return image.convert("RGB")


def _finished(answer: dict) -> str:
    if answer.get("status") != "succeeded" or not answer.get("results"):
        reason = "内容审核没有通过" if answer.get("status") == "violation" or "moderation" in str(
            answer.get("failure_reason")) else answer.get("error") or "生成失败"
        raise CoverError(f"封面生成失败：{str(reason)[:120]}")
    return answer["results"][0]["url"]


def _drawn(client, host, headers, body) -> str:
    """POST /v1/api/generate (async), then GET /v1/api/result until it is done: the picture's URL."""
    response = client.post(host + "/v1/api/generate", json=body, headers=headers)
    try:
        answer = response.json()
    except ValueError:
        raise CoverError(f"封面服务返回 HTTP {response.status_code}") from None
    began = time.monotonic()
    while answer.get("status") not in ("succeeded", "failed", "violation"):
        if not answer.get("id"):
            raise CoverError(str(answer.get("error") or answer.get("msg") or "封面服务没有返回任务")[:120])
        if time.monotonic() - began > AI_WAIT:
            raise CoverError("封面生成超时")
        time.sleep(4)
        answer = client.get(host + "/v1/api/result", params={"id": answer["id"]}, headers=headers).json() \
            | {"id": answer["id"]}
    return _finished(answer)


def _drawn_legacy(client, host, headers, body) -> str:
    """The older interface: /v1/draw/nano-banana or /v1/draw/completions with webHook "-1", polled at
    /v1/draw/result."""
    nano = body["model"].startswith("nano-banana")
    old = {"model": body["model"], "prompt": body["prompt"], "urls": body.get("images", []), "webHook": "-1",
           "shutProgress": True} | ({"aspectRatio": body["aspectRatio"], "imageSize": "1K"} if nano
                                    else {"size": body["aspectRatio"]})
    answer = client.post(host + ("/v1/draw/nano-banana" if nano else "/v1/draw/completions"), json=old,
                         headers=headers).json()
    if answer.get("code") != 0 or not (answer.get("data") or {}).get("id"):
        raise CoverError(str(answer.get("msg") or "封面服务没有返回任务")[:120])
    task, began = answer["data"]["id"], time.monotonic()
    while True:
        if time.monotonic() - began > AI_WAIT:
            raise CoverError("封面生成超时")
        time.sleep(4)
        data = (client.post(host + "/v1/draw/result", json={"id": task}, headers=headers).json() or {}).get("data") or {}
        if data.get("status") in ("succeeded", "failed"):
            return _finished(data)


def _shapes(cfg: dict) -> list[str]:
    return {"both": ["landscape", "portrait"], "landscape": ["landscape"],
            "portrait": ["portrait"]}.get(cfg.get("cover_sizes") or "both", ["landscape", "portrait"])


def render(cfg: dict, picture: Image.Image, parts: dict, title: str, folder: Path, stem: str) -> dict:
    """Save the covers (<stem>-封面-横版.jpg / -竖版.jpg). AI mode draws them with the image model, any shape it
    fails on is designed locally instead. Returns {"files": [...], "error": "..."}."""
    files, error = [], ""
    references = [picture] if cfg.get("cover_reference", "1") == "1" else []
    mode = cfg.get("cover_mode")
    if mode == "ai_bg":
        # One picture without lettering, then both shapes are designed on it: a cent or two a film, and the
        # title cannot come out misspelt.
        try:
            prompt = (cfg.get("cover_prompt_background") or BACKGROUND).strip() + f"\n视频主题：{headline(title)}"
            if references:
                prompt += "\n参考图是这期视频里的画面，可以用其中的人物和场景。"
            try:
                picture = ai_picture(cfg, prompt, "landscape", references)
            except CoverError as exc:
                # The cheapest model times out now and then ("google gemini timeout"): gpt-image-2 once more.
                if "审核" in str(exc) or not model_for(cfg).startswith("nano-banana"):
                    raise
                picture = ai_picture(cfg | {"cover_model": "gpt-image-2"}, prompt, "landscape", references)
        except Exception as exc:   # noqa: BLE001 — the footage frame instead
            error = str(exc)[:160]
    for shape in _shapes(cfg):
        target = folder / f"{stem}-封面-{SHAPES[shape][1]}.jpg"
        picture_out = None
        if mode == "ai":
            try:
                prompt = prompt_for(cfg.get(f"cover_prompt_{shape}"), headline(title), SHAPES[shape][1], bool(references))
                picture_out = ai_picture(cfg, prompt, shape, references)
            except Exception as exc:   # noqa: BLE001 — designed locally instead
                error = str(exc)[:160]
        (picture_out or design(picture, parts, shape)).save(target, "JPEG", quality=92)
        files.append(str(target))
    return {"files": files, "error": error}


def start(project: dict, video: Path, folder: Path, stem: str, title: str) -> dict | None:
    """Covers for a saved film. The background frame and the text are chosen now (one low-cost vision
    request, in the export job); designed covers are drawn at once, AI ones in the background (they take a
    minute or two). The result is noted on the project (cover.files / cover.error)."""
    from . import config, store
    cfg = config.read()
    if (cfg.get("cover_mode") or "design") == "off":
        return None
    starts, at = [], 0.0
    for shot in (project.get("shots") or [])[:SAMPLES]:
        length = float(shot.get("duration") or 0)
        starts.append(at + length / 2)
        at += length
    frames = source_frames(project)
    if not frames:
        fallback = pick_frame(video, starts, (1920, 1080))
        frames = [fallback] if fallback is not None else []
    picture, parts = choose(title, frames)
    def note(result):
        try:
            store.mutate(project["id"], lambda p: p.update(cover=result), internal=True)
        except Exception:   # noqa: BLE001 — the project may be gone
            pass
    if picture is None:
        note({"files": [], "error": "没有可用的画面"})
        return None
    if cfg.get("cover_mode") in ("ai", "ai_bg") and (cfg.get("cover_api_key") or "").strip():
        threading.Thread(target=lambda: note(render(cfg, picture, parts, title, folder, stem)),
                         name="cover", daemon=True).start()
        return {"files": [], "pending": True}
    result = render(cfg | {"cover_mode": "design"}, picture, parts, title, folder, stem)
    note(result)
    return result
