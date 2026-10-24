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


def test_a_cover_is_drawn_on_the_picture():
    picture = Image.new("RGB", (1920, 1080), (90, 120, 150))
    out = cover.draw(picture, "凌晨六枚导弹砸向也门总统府，内阁会议险遭波及，红海局势还能绷多久？")
    assert out.size == (1920, 1080) and out.getpixel((960, 100)) == (90, 120, 150)   # top untouched
    assert out.getpixel((10, 1070)) != (90, 120, 150)                                # bottom darkened
