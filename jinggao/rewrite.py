"""AI rewrite (洗稿): turn a news article or rough draft into a spoken narration script.

One text request to the configured planning model (Qwen by default). Facts stay as in the
source; the wording, order and rhythm change. The result is laid out one short sentence per
line, which the express planner turns into one subtitle line per shot.
"""
from __future__ import annotations

import contextlib
import random
import re
import uuid

from . import costs, planner, subtitle_style

CHARS_PER_MINUTE = 235          # measured narration speed (about 3.9 characters a second)
MAX_SOURCE = 12000

SYSTEM = """你是中文短视频（资讯解说/营销号）口播文案作者。把用户给的素材改写成一篇可以直接配音的口播稿。
只输出 JSON：{"title":"..","script":".."}
要求：
1. 事实准确：人名、地名、机构、数字、比分、日期、引语及其出处必须与素材一致；不得编造素材里没有的事实、数据、引语或细节；素材里没有直接写出的年数、人数、金额、倍数不要自己推算（素材写“2017年”，不要改成“七年前”）；素材中的传闻或单方说法保留“据报道”“某某称”等限定。
2. 重新表达：不要照抄原文句子，用口语化、有节奏的说法重新讲一遍（换句式、调整叙述顺序、补充必要的背景解释），但不能改变事实含义。
3. 结构：开头1-3句直接抛出最抓人的结果、冲突或悬念；中间按时间或因果把来龙去脉讲清楚；结尾1-2句总结，最后用一个问题引导观众在评论区讨论。
4. 格式：每行一句，以短句为主，每句尽量不超过{line}个字；按意思分段，段与段之间空一行。不要小标题、序号、括号注释、markdown、表情符号、话题标签，不要写“各位观众”“点赞关注”“本期视频”。数字、年份、比分、金额一律写阿拉伯数字（如“95后”“41秒”“1.4万”），与素材保持一致。script 里只放口播的正文，不要写 \\n、$、#、* 这类符号，也不要夹英文字母（素材里本来就有的英文名称除外）。
5. 篇幅：约{target}字（可上下浮动15%）。素材信息不够时宁可短一些，不要注水或编造。
6. title：10-20字的视频标题，可以有悬念，但不能夸大或歪曲事实，不要沿用素材标题的说法；{title_style}
7. 不要和素材逐句对应：素材的开头一句不能拿来当开头，句子不要一句换一句地同义改写，至少把两块内容换个先后顺序讲，过渡和比喻用你自己的。
8. 这一篇的讲法：{style}讲法只管顺序和语气，内容只能来自素材，不要虚构人物、场景、对话。"""


META = re.compile(r"素材|原文|标题里|口播稿|核对|未提及|没提(到|过)|无法(确认|核实)")

CHECK = """你是新闻事实核对编辑。给你“素材”和据此改写的“口播稿”（每行一句）。只找口播稿里的事实错误：
数字、比分、日期、年份、排名、人名、地名、机构名写错，张冠李戴（把甲的事说成乙的），因果或结果说反，编造了素材里完全没有的事实、数据或引语。
以下都不算错误，不要改：措辞和句式不同、语序调整、省略了素材里的细节、口语化的概括、开头的悬念和结尾的提问。
素材开头的“标题”也是素材，标题里有的数字和说法不算编造。fix 必须是能直接念出来的口播句子，不要写对素材的说明。
只输出 JSON：{"fixes":[{"line":"口播稿中有错的那一行（逐字照抄）","fix":"只改错处、保持口播风格的新句子"}]}；同一行有多处错误时合并成一条（每行最多一条）；没有错误时 fixes 为空数组。"""


MAX_MINUTES = 15

# One of these is drawn at random for every rewrite, and the model writes at a creative temperature: the same
# source rewritten on several computers (or twice on one) came out word for word the same (11-07), because every
# Qwen request was sent at temperature 0.
STYLES = (
    "开头用一个反常识的结论把人留住，中间按时间顺序讲，结尾点出这件事背后的门道。",
    "开头先抛出一个具体的数字或场面，再倒过来交代前因，最后讲各方的反应。",
    "像跟朋友聊天一样讲，多用“你想想”“说白了”这类口语，开头直接问观众一个问题。",
    "开头先讲结果，再分两三层拆开原因，每层用一句话收住。",
    "从素材里写到的某一方（国家、机构、公司或人物）的处境切进去讲，再拉远看全局，结尾留一个悬念。不要编人物和场景。",
    "开头用对比（以前和现在、说的和做的、这一方和那一方）制造反差，再顺着因果讲清楚。",
    "开头用一句短促的判断定调，正文多用短句和设问推进节奏。",
    "开头先摆出外界最常见的一种看法，再用素材里的事实一步步讲出另一面。",
)
TITLE_STYLES = (
    "标题用一个问句。", "标题先摆出最关键的数字或事实。", "标题用前后反差。", "标题用一句判断加一个悬念。",
    "标题点出事件里的一方。", "标题说出结果，留一个“为什么”。",
)
WRITE_TEMPERATURE = 0.7     # 0.9 made up a year count ("七年时间" for a source saying "2017年4月") in a 11-07 test
COPIED = 0.5           # share of the draft's 6-character runs found in the source above which it is written again (11-07: a
                       # second try brought 0.72 to 0.60 and 0.34 to 0.35, so only the worst are tried again)
COPY_CHECK_FROM = 300  # characters of source: a short one is mostly names and numbers that any rewrite repeats


def copied(script: str, source: str, n: int = 6) -> float:
    """How much of script repeats source word for word: the share of its n-character runs (letters and digits
    only) that also occur in source."""
    flat = lambda text: re.sub(r"[\W_]+", "", text or "")
    a, b = flat(script), flat(source)
    runs = {a[i:i + n] for i in range(max(0, len(a) - n + 1))}
    if not runs:
        return 0.0
    have = {b[i:i + n] for i in range(max(0, len(b) - n + 1))}
    return len(runs & have) / len(runs)


def _today() -> str:
    """The date for the model: without it, "九年前" from a 2017 event was "corrected" to 七年 (its own idea of today)."""
    import time
    t = time.localtime()
    return f"{t.tm_year}年{t.tm_mon}月{t.tm_mday}日"


def target_chars(source: str, minutes: float | None = None) -> int:
    """About the source length (a rewrite neither pads nor drops facts), capped by the wanted duration
    (no duration given: only the overall 15-minute cap)."""
    body = len(re.sub(r"\s+", "", source))
    cap = round((minutes or MAX_MINUTES) * CHARS_PER_MINUTE)
    return max(150, min(cap, round(body * 0.9)))


CJK = r"\u3000-\u303f\u3400-\u9fff\uff00-\uffef"


def clean_script(text: str) -> str:
    """Plain narration: no markdown, list markers, emoji or hashtags; one sentence a line. A line break the model
    wrote as the two characters \\n (seen 11-07: every line of a draft began with "\\n"), $ signs ("$$"), and
    spaces between Chinese characters are taken out."""
    text = str(text or "").replace("\r", "")
    text = re.sub(r"\\+[rn]", "\n", text)            # "\\n" written out instead of a line break
    text = re.sub(r"\\+t", " ", text)
    text = re.sub(r"\\+(?=[\"'“”])", "", text)
    lines = []
    for raw in text.split("\n"):
        line = re.sub(r"^\s*(#+|[-*•·](?=\s)|\d{1,2}[、)）](?!\d)|\d{1,2}\.(?=\s))\s*", "", raw)
        line = re.sub(r"[*_`#$^~|\\{}<>]|\[[^\]]*\]\([^)]*\)", "", line)
        line = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", line)
        line = re.sub(r"[ \t　]+", " ", line).strip()
        line = re.sub(rf"(?<=[{CJK}]) (?=\S)|(?<=\S) (?=[{CJK}])", "", line)
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        # A long line holding several sentences becomes one sentence per line.
        lines += [s.strip() for s in re.findall(r"[^。！？!?]+[。！？!?”」]*", line) if s.strip()] or [line]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def foreign_words(script: str, source: str) -> str:
    """script without the runs of Latin letters that the source never has (a stray "n" left of a "\\n", English
    the model slipped in); names the source itself writes in English (NBA, iPhone) stay."""
    have = {w.lower() for w in re.findall(r"[A-Za-z]+", source or "")}

    def keep(match):
        return match.group(0) if match.group(0).lower() in have else ""
    lines = [re.sub(r"[A-Za-z]+", keep, line) for line in script.split("\n")]
    lines = [re.sub(r"[ \t]{2,}", " ", re.sub(rf"(?<=[{CJK}]) | (?=[{CJK}])", "", line)).strip() for line in lines]
    out = []
    for line in lines:
        if line and not re.search(r"[\w\u3400-\u9fff]", line):
            continue        # nothing left but punctuation
        out.append(line)
    return "\n".join(out).strip()


def unsupported_numbers(script: str, source: str) -> list[str]:
    """Numbers in the rewrite that never occur in the source (to check by hand)."""
    have = set(re.findall(r"\d+(?:\.\d+)?", source))
    found = []
    for number in re.findall(r"\d+(?:\.\d+)?", script):
        if number not in have and number not in found:
            found.append(number)
    return found


def _own(project_id: str) -> str:
    """A shared label (news, rewrite, benchmark) gets a fresh id for each rewrite: the per-project model cap
    (¥0.30) summed every rewrite ever made under the label, and 对标采集 stopped for good after about twenty
    scripts (11-05). Rewrites stay under the daily cap."""
    project_id = str(project_id or "rewrite")
    return project_id if re.fullmatch(r"[0-9a-f]{32}", project_id) else f"{project_id}-{uuid.uuid4().hex[:12]}"


def _scope(project_id: str, stage: str):
    """Inside a worker job the job's own cost scope (and budget) applies; elsewhere bill the project."""
    if costs._scope.get()[0] != "unassigned":
        return contextlib.nullcontext()
    return costs.scope(project_id, "", stage)


def fact_check(script: str, source: str, *, project_id: str = "news", check=None) -> dict:
    """Second pass: find factual slips against the source and correct only those lines. If the check
    itself fails, the script stays as it is and is reported as unchecked."""
    fixes, unmatched, usage, checked_ok = [], 0, {}, True
    try:
        with _scope(project_id, "rewrite_check"):
            if check:
                check()
            checked, check_info = planner.chat(CHECK, f"今天是{_today()}。\n\n素材：\n{source}\n\n口播稿：\n{script}",
                                               max_tokens=3000, timeout=150, task="rewrite")
    except (costs.BudgetExceeded, costs.PricingUnavailable):
        raise
    except Exception as exc:
        from .media import Cancelled
        if isinstance(exc, Cancelled):
            raise
        checked, check_info, checked_ok = {}, {}, False
    lines = script.split("\n")
    key = lambda text: re.sub(r"[\W_]+", "", text)
    def find(text):
        return next((i for i, line in enumerate(lines) if line and key(line) == key(text)), None)
    # Every fix row quotes the ORIGINAL line. The prompt asks for one row per line; if the checker still
    # sends two different corrections for one line, the first is applied and the other is reported.
    edits, fixed_line = {}, {}     # line index -> changes; line index -> its (first) corrected text
    rows = checked.get("fixes") if isinstance(checked, dict) and isinstance(checked.get("fixes"), list) else []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("line"), str):
            unmatched += 1
            continue
        fix = row.get("fix")
        fix = "".join(map(str, fix)) if isinstance(fix, list) else fix
        at = find(row["line"].strip()) if row["line"].strip() else None
        if at is None or not isinstance(fix, str):
            unmatched += 1
            continue
        if fix.strip() == "":      # explicitly empty: an invented sentence with nothing to keep
            edits.setdefault(at, []).append((0, len(lines[at]), None))
            continue
        good = clean_script(fix).replace("\n", "")
        if not good or META.search(good):
            unmatched += 1      # a remark about the source ("素材里根本没提过投资金额"), not a line to read out
            continue
        if re.sub(r"\s+", "", good) == re.sub(r"\s+", "", lines[at]):
            continue     # no change (compare the real text: a fix may move only a decimal point or sign)
        if at in fixed_line and fixed_line[at] != good:
            # A second, different correction of the same line: two ways to fix one error can combine
            # into something neither meant, so only the first is applied and this one is flagged.
            unmatched += 1
            continue
        fixed_line[at] = good
        edits.setdefault(at, []).append((0, 0, good))
    for at, changes in edits.items():
        original = lines[at]
        if any(r is None for _, _, r in changes):
            fixes.append(f"{original} → （删除）")
            lines[at] = None
            continue
        lines[at] = fixed_line[at]
        fixes.append(f"{original} → {fixed_line[at]}")
    script = clean_script("\n".join(line for line in lines if line is not None))
    for name, value in (check_info.get("usage") or {}).items():
        if isinstance(value, (int, float)):
            usage[name] = usage.get(name, 0) + value
    return {"script": script, "fixes": fixes, "unmatched_fixes": unmatched, "checked": checked_ok, "usage": usage}


def from_crawler(koubo: str, source: str, *, title: str = "", project_id: str = "news", check=None) -> dict:
    """The news crawler's own 口播稿 (its AI rewrite): laid out one sentence per line and fact-checked."""
    script = clean_script(koubo)
    if len(re.sub(r"\s+", "", script)) < 50:
        raise ValueError("爬虫口播稿太短")
    project_id = _own(project_id)
    checked = fact_check(script, str(source or "")[:MAX_SOURCE], project_id=project_id, check=check)
    script = checked["script"]
    return {"title": title.strip()[:40], "script": script, "target_chars": len(re.sub(r"\s+", "", koubo)),
            "chars": len(re.sub(r"\s+", "", script)), "check_numbers": unsupported_numbers(script, source),
            "fixes": checked["fixes"], "unmatched_fixes": checked["unmatched_fixes"], "checked": checked["checked"],
            "model": "crawler", "usage": checked["usage"]}


def rewrite(source: str, *, title: str = "", minutes: float | None = None, aspect: str = "16:9",
            project_id: str = "news", check=None) -> dict:
    source = str(source or "").strip()
    if len(re.sub(r"\s+", "", source)) < 30:
        raise ValueError("素材太短，至少需要 30 个字")
    source = source[:MAX_SOURCE]
    project_id = _own(project_id)
    target = target_chars(source, minutes)
    line = subtitle_style.line_chars_for(aspect)
    styles = random.SystemRandom().sample(STYLES, len(STYLES))
    content = (f"今天是{_today()}。\n\n" + (f"素材标题：{title.strip()}\n\n" if title.strip() else "")
               + "素材正文：\n" + source)
    result, info, script, usage, best = {}, {}, "", {}, None
    for attempt in range(2):        # a retry when the answer has no usable script or copies the source
        system = (SYSTEM.replace("{line}", str(line)).replace("{target}", str(target))
                  .replace("{style}", styles[attempt % len(styles)])
                  .replace("{title_style}", random.SystemRandom().choice(TITLE_STYLES)))
        asked = content
        if best is not None:
            asked += ("\n\n上一稿和素材太像了（很多句子只是换了几个字）。这次换一种讲法，开头、句子和顺序都重新写，"
                      "事实不变。")
        with _scope(project_id, "rewrite"):
            if check:
                check()
            result, info = planner.chat(system, asked, max_tokens=min(8000, 600 + target * 2),
                                        timeout=150, task="rewrite", temperature=WRITE_TEMPERATURE)
        for name, value in (info.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[name] = usage.get(name, 0) + value
        body = result.get("script") if isinstance(result, dict) else ""
        script = clean_script("\n".join(map(str, body)) if isinstance(body, list) else body)
        script = foreign_words(script, f"{title}\n{source}")
        if len(re.sub(r"\s+", "", script)) < 50:
            continue
        overlap = copied(script, source)
        if best is None or overlap < best[0]:
            best = (overlap, script, result, info)
        if overlap <= COPIED or len(re.sub(r"[\W_]+", "", source)) < COPY_CHECK_FROM:
            break
    if best is None:
        raise ValueError("模型没有返回可用的口播稿，请稍后重试")
    _, script, result, info = best
    # The title goes with the source: a number only the title gives (6700亿) was "corrected" into a remark (11-07)
    checked = fact_check(script, (f"标题：{title.strip()}\n" if title.strip() else "") + source, project_id=project_id,
                         check=check)
    script, fixes, unmatched, checked_ok = checked["script"], checked["fixes"], checked["unmatched_fixes"], checked["checked"]
    script = foreign_words(script, f"{title}\n{source}")
    for name, value in checked["usage"].items():
        usage[name] = usage.get(name, 0) + value
    new_title = re.sub(r"\s+", " ", str(result.get("title") or "")).strip()[:40] or title.strip()[:40]
    script = re.sub(r"\n{3,}", "\n\n", script)
    rows = script.split("\n")
    if rows.count("") >= 0.6 * (len(rows) - rows.count("")):    # a blank line after every sentence
        script = "\n".join(r for r in rows if r)
    return {"title": new_title, "script": script, "target_chars": target,
            "chars": len(re.sub(r"\s+", "", script)), "check_numbers": unsupported_numbers(script, source),
            "fixes": fixes, "unmatched_fixes": unmatched, "checked": checked_ok,
            "model": info.get("model"), "usage": usage, "copied": round(copied(script, source), 3)}
