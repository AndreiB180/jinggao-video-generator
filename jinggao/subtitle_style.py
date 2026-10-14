"""Burned-in subtitle look: font, size, colour, and how much text fits on one line.

Fonts are SIL OFL files fetched by scripts/fetch_font.py into assets/fonts. A missing
optional font falls back to the bundled regular Noto Sans SC.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

from . import config, paths

# key -> (file, ASS font name, label shown in settings)
FONTS = {
    "black": ("NotoSansSC-Black.otf", "Noto Sans SC Black", "思源黑体 粗 · 经典营销号字幕"),
    "serif": ("NotoSerifSC-Black.otf", "Noto Serif SC Black", "思源宋体 粗 · 纪录片风格"),
    "huangyou": ("ZCOOLQingKeHuangYou-Regular.ttf", "ZCOOL QingKe HuangYou", "站酷庆科黄油体 · 圆润醒目"),
    "kuaile": ("ZCOOLKuaiLe-Regular.ttf", "ZCOOL KuaiLe", "站酷快乐体 · 轻松活泼"),
    "regular": ("NotoSansSC-Regular.otf", "Noto Sans SC", "思源黑体 常规 · 旧版细字"),
}
SIZES = {"60": "小", "72": "中", "84": "大（默认）", "96": "特大"}
COLOURS = {"white": ("&H00FFFFFF", "#FFFFFF", "白字黑边"), "yellow": ("&H0000E6FF", "#FFE600", "黄字黑边")}
DEFAULTS = {"subtitle_font": "black", "subtitle_size": "84", "subtitle_color": "white"}


def font_file(key: str) -> Path | None:
    path = paths.ROOT / "assets" / "fonts" / FONTS[key][0]
    return path if path.exists() else None


def available() -> dict:
    return {k: font_file(k) is not None for k in FONTS}


def width(text: str) -> float:
    """Display width in full-width characters (Latin letters and digits are about half as wide)."""
    return sum(0.55 if ord(ch) < 0x2E80 and not ch.isspace() else (0.3 if ch.isspace() else 1.0) for ch in text)


def settings(values: dict | None = None) -> dict:
    values = values or config.read()
    font = values.get("subtitle_font", DEFAULTS["subtitle_font"])
    size = values.get("subtitle_size", DEFAULTS["subtitle_size"])
    colour = values.get("subtitle_color", DEFAULTS["subtitle_color"])
    return {"subtitle_font": font if font in FONTS else DEFAULTS["subtitle_font"],
            "subtitle_size": size if size in SIZES else DEFAULTS["subtitle_size"],
            "subtitle_color": colour if colour in COLOURS else DEFAULTS["subtitle_color"]}


def style(w: int, h: int, values: dict | None = None) -> dict:
    chosen = settings(values)
    key = chosen["subtitle_font"]
    if font_file(key) is None:
        key = "regular"   # optional fonts not fetched yet
    fallback = "black" if font_file("black") else "regular"
    size = round(int(chosen["subtitle_size"]) * min(w, h) / 1080)
    primary, css, _ = COLOURS[chosen["subtitle_color"]]
    return {"font": key, "file": font_file(key) or paths.font_path(), "family": FONTS[key][1],
            "fallback_file": font_file(fallback) or paths.font_path(), "fallback_family": FONTS[fallback][1],
            "size": size, "outline": max(2, round(size / 11)), "shadow": max(1, round(size / 28)),
            "primary": primary, "css_colour": css, "margin_v": round(h * .07),
            "line_chars": line_chars(w, h, chosen)}


def line_chars(w: int, h: int, values: dict | None = None) -> int:
    """Full-width characters that fit one subtitle line (independent of the output resolution)."""
    size = int(settings(values)["subtitle_size"]) * min(w, h) / 1080
    return max(6, math.floor(w * .9 / size))


def line_chars_for(aspect: str, values: dict | None = None) -> int:
    return line_chars(*((1920, 1080) if aspect != "9:16" else (1080, 1920)), values)


@lru_cache(maxsize=8)
def em_ratio(path: Path) -> float:
    """CSS em size per ASS font size: libass sizes the ascent+descent box, CSS sizes the em square."""
    ascent, descent = ImageFont.truetype(str(path), 100).getmetrics()
    return 100 / max(1, ascent + descent)


@lru_cache(maxsize=8)
def _font(path: str):
    return ImageFont.truetype(path, 32)


@lru_cache(maxsize=8)
def _notdef(path: str) -> bytes:
    return bytes(_font(path).getmask("￿"))


def missing(path: Path, ch: str) -> bool:
    """The font has no glyph for ch (it would render as an empty box)."""
    if ch.isspace() or ord(ch) < 0x80:
        return False
    return bytes(_font(str(path)).getmask(ch)) == _notdef(str(path))


def ass_text(text: str, st: dict) -> str:
    """Escape untrusted narration for ASS and switch to the fallback font for glyphs the style font lacks."""
    out = []
    for ch in text:
        if ch == "\n":
            out.append(r"\N")
            continue
        safe = {"\\": "＼", "{": "｛", "}": "｝", "\r": " "}.get(ch, ch)
        if st["file"] != st["fallback_file"] and missing(st["file"], safe):
            out.append("{\\fn" + st["fallback_family"] + "}" + safe + "{\\fn" + st["family"] + "}")
        else:
            out.append(safe)
    return "".join(out)
