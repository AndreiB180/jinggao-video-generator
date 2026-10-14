"""The standard form every script takes before narration and shot planning.

Pasted scripts, AI rewrites and crawler 口播稿 all become: full-width Chinese punctuation that follows
the meaning, spoken list numbering (第一，第二…), obvious homophone typos corrected, one sentence per
line. A model is asked only when the text needs it (missing punctuation, list markers, spaces used as
pauses); its answer is accepted only if nothing but punctuation, list numbering and a few short
corrections changed, and every correction is reported.
"""
from __future__ import annotations

import difflib
import re

from . import planner
from .execution import ThreadPoolExecutor     # keeps the job's cost scope in the threads

PUNCT = "，。！？；：、,.!?;:"
CJK = r"㐀-鿿"
NORMALIZE = """你是中文口播稿校对。把下面的口播稿整理成标准、读起来顺的朗读文本：
1. 补全并改正标点：按语义断句，句末用。？！，句中停顿用，、：；，引语用“”。原文里用空格或换行代替的停顿也改成标点；换行可能断在词语中间，要按语义接起来。
2. 列表编号改成口语：如“一赶紧开仓放粮……2 派衙役上街……3 全城戒严”改成“第一，赶紧开仓放粮……第二，派衙役上街……第三，全城戒严”。
3. 只改正明显的同音错别字、漏字或多余的字（如“七七枕路”应为“饿殍枕路”），每处不超过4个字；拿不准的不改。
4. 除此之外不得增加、删除、改写或调换任何文字、数字、人名和事实；不要润色、不要缩写、不要加内容。
只输出 JSON：{"text":"整理后的全文（不要换行）"}"""
CHUNK = 1200          # characters per request (cut at the original line breaks)
MAX_EDIT = 4          # characters one correction may touch on either side
MAX_SHARE = 0.03      # all corrections together, as a share of the text


def _plain(text: str) -> str:
    return re.sub(r"[\s，。！？；：、,.!?;:“”\"'‘’（）()【】《》\-—…·]+", "", text)


def needs_model(script: str) -> bool:
    """Punctuation missing or replaced by spaces/line breaks, or bare list numbers in the text."""
    body = re.sub(r"\s+", "", script)
    if len(body) < 40:
        return False
    marks = sum(script.count(p) for p in PUNCT)
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    open_lines = sum(1 for line in lines if not re.search(r"[。！？!?…”」）)]$", line))
    spaced = len(re.findall(rf"[{CJK}] +[{CJK}0-9]", script))
    listed = len(re.findall(rf"(?:^|[{CJK}])\s*\d\s+[{CJK}]|[{CJK}]\s+\d\s*$", script, re.M))
    return marks * 25 < len(body) or (len(lines) >= 6 and open_lines > 0.5 * len(lines)) or spaced >= 3 or listed >= 2


def local(script: str) -> str:
    """Rules that need no model: full-width punctuation between Chinese text, spaces between Chinese
    words become commas, no markdown or emoji, one sentence per line."""
    from .rewrite import clean_script
    text = script.replace("\r", "")
    text = re.sub(rf"(?<=[{CJK}])[ \t\u3000]+(?=[{CJK}])", "，", text)
    # "卖110 我就卖100": a number right after Chinese text, then a space, is a pause (not "iPhone 15 发布").
    text = re.sub(rf"([{CJK}][0-9][0-9.]*)[ \t\u3000]+(?=[{CJK}])", r"\1，", text)
    for half, full in ((",", "，"), (";", "；"), (":", "："), ("?", "？"), ("!", "！")):
        text = re.sub(rf"(?<=[{CJK}]){re.escape(half)}|{re.escape(half)}(?=[{CJK}])", full, text)
    text = re.sub(rf"(?<=[{CJK}])\.(?=\s|$)", "。", text)
    return clean_script(text)


def _accepted(original: str, revised: str) -> tuple[bool, list[dict]]:
    """Only punctuation, list numbering and a few short corrections may differ."""
    a, b = _plain(original), _plain(revised)
    changes, changed = [], 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        old, new = a[i1:i2], b[j1:j2]
        numbering = re.fullmatch(r"[第一二三四五六七八九十0-9]*", old) and re.fullmatch(r"[第一二三四五六七八九十0-9]*", new)
        if not numbering:
            if max(len(old), len(new)) > MAX_EDIT:
                return False, []
            changed += max(len(old), len(new))
            changes.append({"from": a[max(0, i1 - 4):i2 + 4], "to": a[max(0, i1 - 4):i1] + new + a[i2:i2 + 4]})
    return changed <= max(6, MAX_SHARE * len(a)), changes


def _simplified(ch: str) -> bool:
    try:
        ch.encode("gb2312")
        return True
    except UnicodeEncodeError:
        return False


def _keep_simplified(original: str, revised: str) -> str:
    """Undo one-for-one character changes that only turn a simplified character traditional (里→裡)."""
    a, b = _plain(original), _plain(revised)
    positions = [i for i, ch in enumerate(revised) if _plain(ch)]      # plain index -> index in revised
    chars = list(revised)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "replace" and i2 - i1 == j2 - j1:
            for k in range(j2 - j1):
                if not _simplified(b[j1 + k]) and _simplified(a[i1 + k]) and j1 + k < len(positions):
                    chars[positions[j1 + k]] = a[i1 + k]
    return "".join(chars)


def _ask(chunk: str, check=None) -> tuple[str | None, list[dict], dict]:
    if check:
        check()
    try:
        result, info = planner.chat(NORMALIZE, chunk, max_tokens=min(8000, 400 + len(chunk) * 2), timeout=150,
                                    task="planning")
    except ValueError as exc:
        if "HTTP 400" in str(exc) and len(chunk) > 200:
            # Usually the provider's content check refusing one passage: try the halves on their own.
            cut = len(chunk) // 2
            left, right = _ask(chunk[:cut], check), _ask(chunk[cut:], check)
            if left[0] is None or right[0] is None:
                return None, [], {}
            return left[0] + right[0], left[1] + right[1], {"usage": _sum(left[2], right[2])}
        if "HTTP 400" in str(exc):
            return None, [], {}
        raise
    text = result.get("text") if isinstance(result, dict) else None
    if not isinstance(text, str):
        return None, [], info
    text = _keep_simplified(chunk, text)
    ok, changes = _accepted(chunk, text)
    return (text if ok else None), (changes if ok else []), info


def _sum(*infos) -> dict:
    usage = {}
    for info in infos:
        for key, value in ((info or {}).get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
    return usage


def normalize(script: str, check=None) -> tuple[str, dict]:
    """The standard form of a script, plus what was done: {"model": bool, "changes": [...], ...}."""
    script = str(script or "").strip()
    if not needs_model(script):
        return local(script), {"model": False, "changes": [], "chunks": 0, "kept": 0}
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    chunks, current = [], []
    for line in lines:
        current.append(line)
        if sum(len(x) for x in current) >= CHUNK:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    # Lines are joined with a space so the model sees where the original breaks were.
    texts = [" ".join(chunk) for chunk in chunks]
    with ThreadPoolExecutor(max_workers=min(4, len(texts))) as pool:
        answers = list(pool.map(lambda text: _ask(text, check), texts))
    out, changes, kept = [], [], 0
    for text, (revised, found, _) in zip(texts, answers, strict=True):
        if revised is None:
            out.append(text)           # this part stays as it was (local rules only)
            continue
        kept += 1
        out.append(revised)
        changes += found
    normalized = local("\n".join(out))
    return normalized, {"model": True, "changes": changes[:60], "chunks": len(texts), "kept": kept,
                        "usage": _sum(*(info for _, _, info in answers))}
