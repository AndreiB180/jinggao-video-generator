"""封面: a cover picture for a finished film, saved next to its MP4.

A clear, bright frame from the opening shots (sharpest of a few, never a dark one) with the title in big
yellow-and-white Black type over a dark gradient on the lower part of the picture, the usual look of
营销号 covers. The gradient and title cover the burned-in subtitle line of the frame.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from . import paths

FONT = "NotoSansSC-Black.otf"
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


def draw(picture: Image.Image, title: str) -> Image.Image:
    w, h = picture.size
    portrait = h > w
    most = 4 if portrait else 3
    measure = ImageDraw.Draw(picture)
    # The biggest type at which every clause keeps to one line (a little smaller rather than a broken phrase).
    for per_line in ((8, 9, 10) if portrait else (13, 14, 15, 16)):
        size = int(w / (per_line + 1.2))
        font = ImageFont.truetype(str(paths.ROOT / "assets" / "fonts" / FONT), size)
        stroke = max(4, size // 11)
        rows, broken = lines(headline(title), lambda text: measure.textlength(text, font=font) + 2 * stroke <= w * 0.92,
                             most)
        if not broken:
            break
    rows = rows or ["精彩内容"]
    gap = int(size * 0.18)
    block = len(rows) * size + (len(rows) - 1) * gap
    bottom = int(h * (0.88 if portrait else 0.92))
    top = bottom - block
    # The film's own subtitle line sits in the lower part of the frame: smear it before the title goes on.
    band = int(h * 0.7)
    picture = picture.copy()
    picture.paste(picture.crop((0, band, w, h)).filter(ImageFilter.GaussianBlur(radius=max(8, w // 60))), (0, band))
    # Dark gradient from a little above the title to the bottom edge.
    shade = Image.new("L", (1, h), 0)
    begin = max(0, top - int(h * 0.18))
    for y in range(begin, h):
        shade.putpixel((0, y), 200 if y >= top else int(200 * (y - begin) / max(1, top - begin)))
    picture = Image.composite(Image.new("RGB", (w, h), BLACK), picture, shade.resize((w, h)))
    canvas = ImageDraw.Draw(picture)
    y = top
    for i, row in enumerate(rows):
        width = canvas.textlength(row, font=font)
        x = (w - width) / 2
        canvas.text((x, y), row, font=font, fill=YELLOW if i == 0 else WHITE, stroke_width=stroke, stroke_fill=BLACK)
        y += size + gap
    return picture


def make(video: Path, target: Path, title: str, starts: list[float], portrait: bool = False) -> Path | None:
    """Save the cover of video as target (JPEG). starts: times in the film where its first shots begin."""
    size = (1080, 1920) if portrait else (1920, 1080)
    picture = pick_frame(video, starts, size)
    if picture is None:
        return None
    draw(picture, title).save(target, "JPEG", quality=92)
    return target
