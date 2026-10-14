"""Planning for the express pipeline.

The script is split locally into one subtitle line per shot (every character kept),
then a cheap text request (one per ~100 shots) groups shots that can share footage
searches and returns Chinese (Bilibili) and English (stock) queries, a short visual
target per shot, and the named people that must really appear.
"""
from __future__ import annotations

import re

from . import captions, identity, planner, subtitle_style

MAX_SHOTS = 400
BATCH = 100

SYSTEM = """你是营销号短视频的选材编辑。给你按口播切好的镜头，请为找“实拍视频素材”做检索规划。
只输出 JSON：
{"domain":"..","ancient":[],"topic":{"zh":["..",".."],"en":["..",".."]},
 "groups":[{"id":"G1","zh":["..",".."],"en":["..",".."],"person":""}],
 "shots":[{"id":"S1","g":"G1","v":"..","p":[]}]}
规则：
1. groups：把画面内容相近、可以共用同一批素材的镜头归为一组（通常每组2-4个镜头，可以不相邻），组数不超过镜头数的一半且不超过14。
2. zh：2个B站检索词，每个2-3个短词、用空格分隔，写具体可见的画面主体，并尽量加“航拍/实拍/空镜/纪录片/历史影像/素材”之一，例如“集装箱码头 航拍”“岸桥 吊装 集装箱”。不要写抽象概念、观点或整句。
3. en：2个Pixabay英文检索词，2-4个英文单词，写通用可见画面，如“container port aerial”。若该组是中国特定历史事件、具名人物、特定中国文化物件等外国图库拍不到的内容，en 给空数组。
4. person：只有当该组镜头的口播在讲某个具体真实人物、画面理应出现此人时，填写原文中的完整姓名，否则填空字符串。有人物的组只放讲这个人的镜头，zh 检索词必须包含这个姓名（如“某某 历史影像”）。
5. shots：每个镜头都要出现且顺序不变。g=所属组；v=6-14字的通用画面建议（场景+主体，容易在实拍素材中找到，不写细节动作、特写或文字内容；不要写截图、网页、评论区、弹幕、聊天记录这类屏幕画面，讲网传视频或网友热议时写与事件相关的实拍场景），普通画面只需大致相关；p=画面必须出现的真实具名人物姓名（必须与口播原文逐字一致），没有就空数组。
6. topic：整篇文稿的通用画面检索词（zh 2个、en 2个），用于补位，要选不出现具体人物也能用的环境/场景画面。
7. 若有“用户画面要求”，必须体现在检索词和画面建议中。
9. ancient：口播里出现的、生活在照相术出现（约1840年）以前的具名古人姓名（如范仲淹、包拯、李白），没有就空数组；这些人不可能有真实影像。
8. domain：整篇所属的宽泛领域，2-6字（如“乒乓球”“集装箱港口”“兰州牛肉面”“宋朝历史”“人工智能”）。不要用文章标题、具体事件或人物（讲范仲淹救灾写“宋朝历史”，不能写“范仲淹荒政”）。除带具体人名的检索词外，每个 zh 检索词都要包含这个领域词，避免搜到同名的其他领域（例如讲乒乓球世界排名，要写“乒乓球 世界排名”，不能只写“世界排名”）。
10. 古人（ancient 里的人）：他们没有真实影像，组的 zh 检索词写能搜到影视剧、纪录片片段的词，如“范仲淹 电视剧 片段”“范仲淹 纪录片”；普通历史画面的组写朝代场景，如“宋朝 市井 影视剧”“古代 粮仓 纪录片”“宋代 西湖 画舫”，尽量是有动作的视频画面，少用静态插画。"""


def _text(value, limit) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _queries(values, limit, ascii_only=False) -> list[str]:
    result = []
    for value in values if isinstance(values, list) else []:
        text = _text(value, limit)
        if not text or (ascii_only and not re.fullmatch(r"[A-Za-z0-9 ,'&-]+", text)):
            continue
        if text not in result:
            result.append(text)
    return result[:2]


# A shot shorter than about this many full-width characters can last under 1 s with a fast voice
# (Baidu 一镜 reads about 5 characters a second); such pieces join a neighbour when the line allows.
MIN_WIDTH = 6


def line_shots(script: str, line_chars: int) -> list[dict]:
    """One subtitle line per shot: sentences, split at clauses (then words) when wider than a line.
    A piece too short to speak for about a second joins its narrower neighbour in the same paragraph
    (lines up to a blank line), as long as the result still fits one subtitle line."""
    pieces, paragraph = [], 0
    for line in script.splitlines():
        if not line.strip():
            paragraph += 1
            continue
        for sentence in re.findall(r"[^。！？!?]+[。！？!?]*|[。！？!?]+", line):
            if sentence.strip():
                pieces += [[paragraph, piece] for piece in captions.parts(sentence.strip(), line_chars) if piece.strip()]
    width = subtitle_style.width
    merged = True
    while merged:
        merged = False
        for i, (n, text) in enumerate(pieces):
            if width(text) >= MIN_WIDTH:
                continue
            options = [(width(pieces[j][1]), j) for j in (i - 1, i + 1)
                       if 0 <= j < len(pieces) and pieces[j][0] == n and width(pieces[j][1]) + width(text) <= line_chars]
            if options:
                a, b = sorted((i, min(options)[1]))
                pieces[a][1] += pieces[b][1]
                del pieces[b]
                merged = True
                break
    if len(pieces) > MAX_SHOTS:
        raise ValueError(f"稿件按字幕行拆分后超过 {MAX_SHOTS} 个镜头，请分成多个项目")
    if planner.normalized("".join(p for _, p in pieces)) != planner.normalized(script):
        raise ValueError("文稿包含无法自动切分的结构，请整理为口播正文")
    return [planner.new_shot(text) for _, text in pieces]


def merge_short(project: dict, indices: list[int]) -> None:
    """After real alignment: join each too-short shot to its shorter neighbour. Narration texts are
    concatenated in order (the full audio is unchanged); the kept shot keeps its footage and plan."""
    shots = project["shots"]
    for i in sorted(set(indices), reverse=True):
        if len(shots) < 2 or not 0 <= i < len(shots):
            continue
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(shots)]
        j = min(neighbours, key=lambda k: (subtitle_style.width(shots[k]["narration"]), k))
        a, b = sorted((i, j))
        keep, gone = shots[a], shots[b]
        keep["narration"] += gone["narration"]
        keep["duration"] = round(float(keep["duration"]) + float(gone["duration"]), 3)
        if not keep.get("selected_asset") and gone.get("selected_asset"):
            for key in ("selected_asset", "verification", "frame_crop", "crop_asset", "crop_bottom", "fit", "clip_start", "approved"):
                if key in gone:
                    keep[key] = gone[key]
        people = {p["name"]: {"name": p["name"], "aliases": p.get("aliases", [])}
                  for p in (keep.get("persons") or []) + (gone.get("persons") or [])}
        keep["persons"] = identity.persons(list(people.values())[:3])
        if keep["persons"]:
            keep["evidence_mode"] = "person"
        keep["approved"] = False
        del shots[b]
    info = project.get("plan_info") or {}
    for group in info.get("groups", []):
        group["shots"] = [sid for sid in group.get("shots", []) if any(s["id"] == sid for s in shots)]


def _ask(shots: list[dict], requirements: str, check=None) -> tuple[dict, dict]:
    lines = [f"S{i+1}：{shot['narration']}" for i, shot in enumerate(shots)]
    content = ("用户画面要求：" + (requirements.strip() or "无") + "\n\n镜头：\n" + "\n".join(lines))
    max_tokens = min(8000, 900 + 70 * len(shots))
    if check:
        check()
    return planner.chat(SYSTEM, content, max_tokens=max_tokens, timeout=120, task="express_plan")


def annotate(shots: list[dict], script: str, requirements: str = "", check=None) -> dict:
    """Mutates shots in place with express fields; returns plan_info. Long scripts are planned in
    batches of about BATCH shots (cut at paragraph-like boundaries); group ids get a batch prefix."""
    batches, start = [], 0
    while start < len(shots):
        end = min(len(shots), start + BATCH)
        if end < len(shots):   # avoid leaving a tiny last batch
            end = min(end, len(shots) - BATCH // 3) if len(shots) - end < BATCH // 3 else end
        batches.append((start, end))
        start = end
    rows, groups, topic, usage, model, domain = {}, {}, {"zh": [], "en": []}, {}, None, ""
    if len(batches) > 1:   # independent requests: ask them together
        from .execution import ThreadPoolExecutor   # keeps the job's cost scope in the threads
        with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
            answers = list(pool.map(lambda b: _ask(shots[b[0]:b[1]], requirements, check), batches))
    else:
        answers = [_ask(shots, requirements, check)]
    # One domain word for the whole script (the first batch that names one).
    domain = next((_text(r.get("domain"), 8) for r, _ in answers if isinstance(r, dict) and _text(r.get("domain"), 8)), "")
    # People from before photography: no real footage can exist, so portraits and dramatisations stand in.
    ancient = sorted({name for r, _ in answers if isinstance(r, dict) and isinstance(r.get("ancient"), list)
                      for name in (_text(n, 40) for n in r["ancient"]) if len(name) >= 2 and name in script})
    for number, ((begin, end), (result, info)) in enumerate(zip(batches, answers, strict=True)):
        model = model or info.get("model")
        for key, value in (info.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
        prefix = f"B{number+1}" if len(batches) > 1 else ""
        for r in result.get("shots", []) if isinstance(result.get("shots"), list) else []:
            if isinstance(r, dict) and re.fullmatch(r"S\d+", str(r.get("id", "")).strip()):
                local = int(str(r["id"]).strip()[1:])
                if 1 <= local <= end - begin:
                    rows[f"S{begin + local}"] = r | {"g": prefix + str(r.get("g", "")).strip()[:12]}
        for row in result.get("groups", []) if isinstance(result.get("groups"), list) else []:
            if not isinstance(row, dict) or not str(row.get("id", "")).strip():
                continue
            gid = prefix + str(row["id"]).strip()[:12]
            person = _text(row.get("person"), 40)
            if person and person not in script:
                person = ""
            zh = _queries(row.get("zh"), 40)
            if person:
                zh = [q if person in q else f"{person} {q}" for q in zh] or [f"{person} 历史影像"]
            elif domain:
                zh = [q if domain in q.replace(" ", "") else f"{domain} {q}" for q in zh]
            groups[gid] = {"id": gid, "zh": zh, "en": [] if person else _queries(row.get("en"), 60, ascii_only=True),
                           "person": person, "shots": []}
        found = result.get("topic") if isinstance(result.get("topic"), dict) else {}
        if not topic["zh"]:
            topic = {"zh": _queries(found.get("zh"), 40), "en": _queries(found.get("en"), 60, ascii_only=True)}
    for i, shot in enumerate(shots):
        row = rows.get(f"S{i+1}", {})
        gid = str(row.get("g", "")).strip()[:16]
        if gid not in groups:
            gid = f"X{i+1}"
            groups[gid] = {"id": gid, "zh": [_text(shot["narration"], 30)], "en": [], "person": "", "shots": []}
        group = groups[gid]
        names = []
        for name in row.get("p", []) if isinstance(row.get("p"), list) else []:
            name = _text(name, 40)
            if 2 <= len(name) and name in shot["narration"] and name not in names:
                names.append(name)
        if group["person"] and group["person"] in shot["narration"] and group["person"] not in names:
            names.insert(0, group["person"])
        visual = _text(row.get("v"), 60) or _text(shot["narration"], 60)
        group["shots"].append(shot["id"])
        shot.update(
            visual=visual, visual_origin="express", query=(group["zh"] or [visual])[0],
            search_queries=list(group["zh"]), persons=identity.persons([{"name": n} for n in names[:3]]),
            evidence_mode="person" if names else "generic", event="", event_scope="exact",
            express={"group": gid, "en": list(group["en"]), "zh": list(group["zh"])})
    known = {g["person"] for g in groups.values() if g["person"]}
    known |= {p["name"] for shot in shots for p in shot.get("persons") or []}
    for shot in shots:
        names = [p["name"] for p in shot.get("persons") or []]
        missing = [n for n in sorted(known, key=len, reverse=True) if n in shot["narration"] and n not in names]
        if missing:
            shot["persons"] = identity.persons([{"name": n} for n in (names + missing)[:3]])
            shot["evidence_mode"] = "person"
    groups = {gid: g for gid, g in groups.items() if g["shots"]}
    if not topic["zh"]:
        topic["zh"] = [q for g in groups.values() if not g["person"] for q in g["zh"]][:2]
    if domain:
        topic["zh"] = [q if domain in q.replace(" ", "") else f"{domain} {q}" for q in topic["zh"]]
    return {"mode": "express", "groups": list(groups.values()), "topic": topic, "domain": domain, "ancient": ancient,
            "model": model, "usage": usage, "batches": len(batches)}


def plan(project: dict, check=None) -> tuple[list[dict], dict]:
    """Normalise the script (script_norm: punctuation, spoken numbering, obvious typos, one sentence per
    line), then one subtitle line per shot. info["script"] is the standard text narration will read."""
    from . import script_norm
    script = project.get("script", "").strip()
    if not script:
        raise ValueError("请先输入文稿")
    script, normalized = script_norm.normalize(script, check)
    shots = line_shots(script, subtitle_style.line_chars_for(project.get("aspect", "16:9")))
    info = annotate(shots, script, project.get("requirements", ""), check)
    return shots, info | {"script": script, "normalize": normalized}
