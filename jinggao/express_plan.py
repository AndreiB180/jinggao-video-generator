"""Planning for the express pipeline.

The script is split locally into one subtitle line per shot (every character kept),
then a cheap text request (one per ~100 shots) groups shots that can share footage
searches and returns Chinese (Bilibili) and English (stock) queries, a short visual
target per shot, and the named people that must really appear.
"""
from __future__ import annotations

import math
import re

from . import captions, identity, planner, subtitle_style

MAX_SHOTS = 400
BATCH = 100

SYSTEM = """你是营销号短视频的选材编辑。给你按口播切好的镜头，请为找“实拍视频素材”做检索规划。
只输出 JSON：
{"domain":"..","ancient":[],"event":[],"topic":{"zh":["..",".."],"en":["..",".."]},
 "groups":[{"id":"G1","zh":["..",".."],"en":["..",".."],"person":""}],
 "shots":[{"id":"S1","g":"G1","v":"..","p":[]}]}
规则：
1. groups：把画面内容相近、可以共用同一批素材的镜头归为一组（每组2-4个镜头，可以不相邻），组数不少于镜头数的四分之一、不超过镜头数的一半，最多20组。组分得太粗，全片只有几批素材，画面会翻来覆去重复。
2. zh：2个B站检索词，每个2-4个短词、用空格分隔。必须带上这组口播里出现的具体名称（国家、地区、城市、组织、人名、品牌、型号之一），再写具体可见的画面主体，例如“也门 胡塞 导弹 袭击”“沙特 联军 空袭”“红海 驱逐舰”“集装箱码头 岸桥 吊装”。时事新闻（战争冲突、外交、政治、社会事件）加“新闻”或“现场”；历史题材可加“纪录片/历史影像”；只有泛泛的场景（城市、自然、生活）才用“航拍/空镜/素材”。讲人的镜头（官员、领导人、民众、士兵、当事人、开会、讲话、抗议）检索词写人物和场景（如“也门 总理 讲话”“萨那 街头 民众”“沙特 王储 会见”），不要写武器装备；同一篇里各组的检索词要互相错开，不要几组都搜同一类画面。情感、婚姻、家庭、职场、社会故事（没有真实新闻画面的个人经历）的镜头，检索词写情节加“电视剧片段”或“短剧”，如“夫妻 吵架 电视剧片段”“妻子 查看 丈夫手机 短剧”“婆媳 争执 电视剧”，这类片段有真人有情绪，比空镜好。不要写抽象概念、观点或整句，也不要只写“导弹”“军舰”“港口”这种不带名称的泛词（会搜到别国别处的画面）。
3. en：2个Pixabay英文检索词，2-4个英文单词，写通用可见画面，如“container port aerial”。外国图库只有通用的风景、城市、生活、行业画面：若该组讲的是具体国家地区正在发生的新闻事件（战争冲突、外交、政治、灾难、社会新闻）、中国特定历史事件、具名人物或特定中国文化物件，en 给空数组（图库里只会是不相干的游艇、海景、街道）；画面主体是中国人的生活场景（家庭、夫妻、父母孩子、职场、街坊邻里、农村）时也给空数组（图库里是外国人，观众一看就不对）。
4. person：只有当该组镜头的口播在讲某个具体真实人物、画面理应出现此人时，填写原文中的完整姓名，否则填空字符串。有人物的组只放讲这个人的镜头，zh 检索词必须包含这个姓名（如“某某 历史影像”）。
5. shots：每个镜头都要出现且顺序不变。g=所属组；v=6-14字的通用画面建议（场景+主体，容易在实拍素材中找到，不写细节动作、特写或文字内容；不要写截图、网页、评论区、弹幕、聊天记录这类屏幕画面，讲网传视频或网友热议时写与事件相关的实拍场景），普通画面只需大致相关；p=画面必须出现的真实具名人物姓名（必须与口播原文逐字一致），没有就空数组；社会故事、情感故事里的化名和代称（张女士、李先生、老周、小李、王某、刘阿姨）不是具名人物，不写进 p，检索词里也不要写这些名字。
6. topic：整篇文稿的通用画面检索词（zh 2个、en 2个），用于补位，要选不出现具体人物也能用的环境/场景画面。
7. 若有“用户画面要求”，必须体现在检索词和画面建议中。
9. ancient：口播里出现的、生活在照相术出现（约1840年）以前的具名古人姓名（如范仲淹、包拯、李白），没有就空数组；这些人不可能有真实影像。
8. domain：整篇所属的宽泛领域，2-6字（如“乒乓球”“集装箱港口”“兰州牛肉面”“宋朝历史”“人工智能”）。不要用文章标题、具体事件或人物（讲范仲淹救灾写“宋朝历史”，不能写“范仲淹荒政”）。时事新闻写事件所在的国家或地区（如“也门”“俄乌”“台海”“日本”），不要写“局势”“形势”“问题”“风云”这类抽象词（搜不到画面）。除带具体人名的检索词外，每个 zh 检索词都要包含这个领域词，避免搜到同名的其他领域（例如讲乒乓球世界排名，要写“乒乓球 世界排名”，不能只写“世界排名”）。
10. 古人（ancient 里的人）：他们没有真实影像，组的 zh 检索词写能搜到影视剧、纪录片片段的词，如“范仲淹 电视剧 片段”“范仲淹 纪录片”；普通历史画面的组写朝代场景，如“宋朝 市井 影视剧”“古代 粮仓 纪录片”“宋代 西湖 画舫”，尽量是有动作的视频画面，少用静态插画。
11. event：若整篇讲的是一件具体的新闻事件（有明确的当事方、地点或时间，例如某次袭击、某场会谈、某起案件），给 2-3 个能在 B 站搜到这件事新闻报道视频的检索词，每个 3-5 个短词，写当事方 + 地点 + 事件（如“胡塞 导弹 袭击 亚丁 总统府”“也门 内阁 会议 遇袭”“沙特 联军 空袭 萨那”）；讲的是泛泛的话题、个人故事、历史或科普，给空数组。"""


def _text(value, limit) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


# A story's stand-in for an ordinary person (张女士, 老周, 小李, 王某): no one to find footage of, and as a
# search word it finds strangers' vlogs with the same nickname.
# 老X/小X/阿X is one only when the plan names it as a person (老挝, 小米 and 阿里 are not people).
TITLED = re.compile(r"[\u4e00-\u9fff](?:女士|先生|小姐|某某|某|阿姨|大爷|大妈|大娘|大姐|大哥|师傅|奶奶|爷爷|叔|婶)")
NICKNAME = re.compile(r"(?:老|小|阿)[\u4e00-\u9fff]")


def placeholder(name: str, person: bool = True) -> bool:
    return bool(TITLED.fullmatch(name or "") or (person and NICKNAME.fullmatch(name or "")))


def _queries(values, limit, ascii_only=False) -> list[str]:
    result = []
    for value in values if isinstance(values, list) else []:
        text = _text(value, limit)
        if not text or (ascii_only and not re.fullmatch(r"[A-Za-z0-9 ,'&-]+", text)):
            continue
        if text not in result:
            result.append(text)
    return result[:2]


# A shot shorter than about this many full-width characters flashes past (about 2 s at 4.5 characters a
# second; Baidu 一镜 reads about 5); such pieces join a neighbour when the line allows.
MIN_WIDTH = 9
# After the real voice is aligned, a shot shorter than this joins a neighbour (before any footage is chosen).
MIN_SECONDS = 2.0
# Setting shot_length: how much narration one shot covers, as full-width characters (about 4.8 a second with
# the default voice): a shot takes subtitle lines until it reaches the first number and never passes the
# second. Similar channels cut every 6-7 s; one line per shot (about 2.5 s) needs 2-3 times the clips,
# so a long script takes that much longer to match and repeats more.
SHOT_WIDTHS = {"long": (28, 44), "medium": (19, 30), "line": None}


def shot_widths(values: dict | None = None):
    from . import config
    choice = str((values if values is not None else config.read()).get("shot_length") or "long")
    return SHOT_WIDTHS.get(choice, SHOT_WIDTHS["long"])


def line_shots(script: str, line_chars: int, widths: tuple[int, int] | None = None) -> list[dict]:
    """Subtitle lines (sentences, split at clauses, then words, when wider than a line), grouped into shots.
    A piece too short to speak for about a second joins its narrower neighbour in the same paragraph
    (lines up to a blank line), as long as the result still fits one subtitle line. With widths
    (target, most) neighbouring lines of a paragraph then share a shot until it is about target characters
    wide; the subtitles still show one line at a time. Without, one line per shot."""
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
    if widths:
        target, most = widths
        shots = []
        for n, text in pieces:
            if shots and shots[-1][0] == n and width(shots[-1][1]) < target and width(shots[-1][1]) + width(text) <= most:
                shots[-1][1] += text
            else:
                shots.append([n, text])
        pieces = shots
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


OPENING = ("\nS1 是全片开头的钩子，观众看不看下去就看它：S1 单独成一组，检索词写开头这件事最具体、最有冲击力的现场画面"
           "（事件现场、冲突场面、当事人、标志性地点），不要写概念或泛泛的场景。")


def _ask(shots: list[dict], requirements: str, check=None, opening: bool = False) -> tuple[dict, dict]:
    lines = [f"S{i+1}：{shot['narration']}" for i, shot in enumerate(shots)]
    least, most = max(1, math.ceil(len(shots) / 4)), min(20, max(1, len(shots) // 2))
    content = ("用户画面要求：" + (requirements.strip() or "无") + f"\n\n共 {len(shots)} 个镜头，groups 分 {least}-{most} 组。"
               + (OPENING if opening else "") + "\n镜头：\n" + "\n".join(lines))
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
    stand_ins = set()
    if len(batches) > 1:   # independent requests: ask them together
        from .execution import ThreadPoolExecutor   # keeps the job's cost scope in the threads
        with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
            answers = list(pool.map(lambda b: _ask(shots[b[0]:b[1]], requirements, check, opening=b[0] == 0), batches))
    else:
        answers = [_ask(shots, requirements, check, opening=True)]
    # One domain word for the whole script (the first batch that names one).
    domain = next((_text(r.get("domain"), 8) for r, _ in answers if isinstance(r, dict) and _text(r.get("domain"), 8)), "")
    # A concrete news event: queries for news reports about it (news_pool cuts their footage where they say the same).
    event = next((_queries(r.get("event"), 40) for r, _ in answers if isinstance(r, dict) and _queries(r.get("event"), 40)), [])
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
            if placeholder(person):
                stand_ins.add(person)
            if person and (person not in script or placeholder(person)):
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
            if placeholder(name):
                stand_ins.add(name)
            elif 2 <= len(name) and name in shot["narration"] and name not in names:
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
    for group in groups.values():
        # A story's nicknames found by the model, and any 张女士/王某 form, are not words to search for.
        kept = [" ".join(w for w in q.split() if w not in stand_ins and not placeholder(w, person=False))
                for q in group["zh"]]
        group["zh"] = [q for q in kept if q] or group["zh"]
    for shot in shots:
        group = groups.get(shot["express"]["group"])
        if group:
            shot.update(query=(group["zh"] or [shot["visual"]])[0], search_queries=list(group["zh"]))
            shot["express"]["zh"] = list(group["zh"])
    if not topic["zh"]:
        topic["zh"] = [q for g in groups.values() if not g["person"] for q in g["zh"]][:2]
    if domain:
        topic["zh"] = [q if domain in q.replace(" ", "") else f"{domain} {q}" for q in topic["zh"]]
    return {"mode": "express", "groups": list(groups.values()), "topic": topic, "domain": domain, "ancient": ancient,
            "event": event, "model": model, "usage": usage, "batches": len(batches)}


def plan(project: dict, check=None) -> tuple[list[dict], dict]:
    """Normalise the script (script_norm: punctuation, spoken numbering, obvious typos, one sentence per
    line), then subtitle lines grouped into shots (setting shot_length). info["script"] is the standard text
    narration will read."""
    from . import script_norm
    script = project.get("script", "").strip()
    if not script:
        raise ValueError("请先输入文稿")
    script, normalized = script_norm.normalize(script, check)
    shots = line_shots(script, subtitle_style.line_chars_for(project.get("aspect", "16:9")), shot_widths())
    info = annotate(shots, script, project.get("requirements", ""), check)
    return shots, info | {"script": script, "normalize": normalized}
