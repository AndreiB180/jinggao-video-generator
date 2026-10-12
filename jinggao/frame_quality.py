"""Narrow quality rejections grounded only in neutral observations' actual OCR.

This module neither approves a clip nor judges identity, source rights or whether
an image is AI generated. Missing/unrecognised OCR is not evidence of good quality.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

VERSION = 1
MIN_REPEATED_FRAMES = 2

_SALE = r"(?:下单|购买|付费|联系客服)"
_MEDIA = r"(?:高清(?:无水印)?(?:版本?|素材|原片|视频)|无水印(?:版本?|素材|原片|视频)?|(?:视频)?素材|原片)"
# Only direct acquisition language may connect the sale and the media object.
# Unbounded wildcards could turn ordinary shopping/news captions into a CTA.
_BRIDGE = r"(?:后|即可|可|并|请|获取|获得|下载|领取|提供|解锁|购买|下单|付费|[:,;]){0,6}"
_PROMPT = re.compile(
    rf"{_SALE}{_BRIDGE}{_MEDIA}"
    rf"|{_MEDIA}[:,;]{{0,2}}(?:请|需要|需)[:,;]{{0,2}}{_SALE}"
)
_NON_OFFER = re.compile(
    r"(?:无需|无须|不需|不必|不用|不下单|不购买|不付费|免费|"
    r"不要|请勿|切勿|禁止购买|禁止付费|禁止下单|警惕|谨防|骗局|诈骗|被骗|骗子)"
)
_DISCUSSION = re.compile(r"(?:如何|怎样|怎么|教程|购买成本|购买渠道|购买流程)")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", "", text)


def _is_offer(text: str) -> bool:
    for match in _PROMPT.finditer(text):
        # A warning/negation near the quoted phrase is not a sales offer.
        # This deliberately errs against hard-blocking ambiguous reportage.
        context = text[max(0, match.start() - 16):match.end() + 16]
        if not _NON_OFFER.search(context) and not _DISCUSSION.search(context):
            return True
    return False


def _frame_evidence(frame: Mapping) -> list[dict]:
    visible = frame.get("text")
    if not isinstance(visible, list):
        return []
    # An OCR engine may split one overlay over adjacent lines. Never combine
    # different frames, and prefer the smallest complete, attributable quote.
    for width in (1, 2, 3):
        for start in range(len(visible) - width + 1):
            lines = visible[start:start + width]
            if any(not isinstance(line, str) or not line.strip() for line in lines):
                continue
            text = _normalise("".join(lines))
            if len(text) > 180 or not _is_offer(text):
                continue
            return [{"frame_index": frame["frame_index"], "field": "text",
                     "text_index": start + offset, "quote": line}
                    for offset, line in enumerate(lines)]
    return []


def assess_sales_preview(frame_observations: Sequence[Mapping]) -> dict:
    """Reject repeated explicit sale overlays; all other results remain unjudged.

    Pass the neutral observer's ``frames`` list (or persisted
    ``frame_observations``), not a title, source description or model verdict.
    A rejection requires acquisition language and a media object in at least two
    distinct nonnegative integer frame indices. Evidence preserves the exact OCR
    strings and their indices, including any original whitespace/punctuation.
    Duplicate/invalid frame indices cannot inflate repetition. The result never
    returns a match/approval, and ``not_detected`` must continue normal review.
    """
    evidence, seen, matched = [], set(), set()
    if not isinstance(frame_observations, (list, tuple)):
        frame_observations = []
    for frame in frame_observations:
        if not isinstance(frame, Mapping):
            continue
        index = frame.get("frame_index")
        if type(index) is not int or index < 0 or index in seen:
            continue
        seen.add(index)
        quotes = _frame_evidence(frame)
        if quotes:
            matched.add(index)
            evidence.extend(quotes)
    rejected = len(matched) >= MIN_REPEATED_FRAMES
    return {
        "version": VERSION,
        "status": "reject" if rejected else "not_detected",
        "code": "repeated_sales_preview" if rejected else "no_repeated_sales_preview_detected",
        "reason": ("多个实际帧重复出现购买/付费获取高清或无水印素材的售卖预览提示，当前片段质量不合格。"
                   if rejected else "未检出达到重复帧门槛的明确售卖预览文字；不代表素材质量或内容已通过。"),
        "matched_frame_count": len(matched),
        "required_frame_count": MIN_REPEATED_FRAMES,
        "evidence": evidence,
    }
