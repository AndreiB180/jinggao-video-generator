import pytest

from jinggao import editorial_contract


@pytest.mark.parametrize("separator", ["\n\n", "\r\n\r\n", "\n \t\n\n"])
def test_source_span_starting_in_separator_selects_current_paragraph(separator):
    previous = "这是前一段，不属于当前句。"
    current = "先看一个数字。这里给出统计年度和数据。"
    script = previous + separator + current + separator + "这是下一段。"
    span = {"start": len(previous), "end": script.index("这里")}
    context = editorial_contract.local_context(script, span)
    assert context["source_span"] == span
    assert context["narration"] == script[span["start"]:span["end"]]
    assert [p["quote"] for p in context["paragraphs"]] == [current]
    for paragraph in context["paragraphs"]:
        start, end = paragraph["source_span"].values()
        assert script[start:end] == paragraph["quote"]


def test_end_separator_does_not_absorb_following_paragraph():
    script = "第一段的前句。第一段的后句。\n\n第二段。"
    span = {"start": script.index("第一段的后句"), "end": script.index("第二段")}
    context = editorial_contract.local_context(script, span)
    assert [p["quote"] for p in context["paragraphs"]] == ["第一段的前句。第一段的后句。"]


def test_genuinely_crossing_span_keeps_both_paragraphs():
    script = "开头。第一段末。\n\n第二段初。第二段末。\n\n第三段。"
    span = {"start": script.index("第一段末"), "end": script.index("第二段末")}
    context = editorial_contract.local_context(script, span)
    assert [p["quote"] for p in context["paragraphs"]] == ["开头。第一段末。", "第二段初。第二段末。"]


@pytest.mark.parametrize("span", [{"start": -1, "end": 2}, {"start": 4, "end": 2},
    {"start": True, "end": 2}, {"start": 0, "end": 200}, {"start": 3, "end": 5}])
def test_invalid_or_whitespace_only_span_does_not_invent_local_scope(span):
    assert editorial_contract.local_context("原文。\n\n后文。", span) == {}
