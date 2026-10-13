"""One-call planning for the express pipeline.

The script is split locally (every character kept, sentence boundaries), then
a single cheap text request groups shots that can share footage searches and
returns Chinese (Bilibili) and English (stock) queries, a short visual target
per shot, and the named people that must really appear.
"""
from __future__ import annotations

import re

from . import identity, planner

SYSTEM = """你是营销号短视频的选材编辑。给你按口播切好的镜头，请为找“实拍视频素材”做检索规划。
只输出 JSON：
{"topic":{"zh":["..",".."],"en":["..",".."]},
 "groups":[{"id":"G1","zh":["..",".."],"en":["..",".."],"person":""}],
 "shots":[{"id":"S1","g":"G1","v":"..","p":[]}]}
规则：
1. groups：把画面内容相近、可以共用同一批素材的镜头归为一组（通常每组2-4个镜头，可以不相邻），组数不超过镜头数的一半且不超过14。
2. zh：2个B站检索词，每个2-3个短词、用空格分隔，写具体可见的画面主体，并尽量加“航拍/实拍/空镜/纪录片/历史影像/素材”之一，例如“集装箱码头 航拍”“岸桥 吊装 集装箱”。不要写抽象概念、观点或整句。
3. en：2个Pixabay英文检索词，2-4个英文单词，写通用可见画面，如“container port aerial”。若该组是中国特定历史事件、具名人物、特定中国文化物件等外国图库拍不到的内容，en 给空数组。
4. person：只有当该组镜头的口播在讲某个具体真实人物、画面理应出现此人时，填写原文中的完整姓名，否则填空字符串。有人物的组只放讲这个人的镜头，zh 检索词必须包含这个姓名（如“某某 历史影像”）。
5. shots：每个镜头都要出现且顺序不变。g=所属组；v=6-14字的通用画面建议（场景+主体，容易在实拍素材中找到，不写细节动作、特写或文字内容），普通画面只需大致相关；p=画面必须出现的真实具名人物姓名（必须与口播原文逐字一致），没有就空数组。
6. topic：整篇文稿的通用画面检索词（zh 2个、en 2个），用于补位，要选不出现具体人物也能用的环境/场景画面。
7. 若有“用户画面要求”，必须体现在检索词和画面建议中。"""


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


def annotate(shots: list[dict], script: str, requirements: str = "", check=None) -> dict:
    """Mutates shots in place with express fields; returns plan_info."""
    lines = [f"S{i+1}：{shot['narration']}" for i, shot in enumerate(shots)]
    content = ("用户画面要求：" + (requirements.strip() or "无") + "\n\n镜头：\n" + "\n".join(lines))
    max_tokens = min(8000, 900 + 70 * len(shots))
    if check:
        check()
    result, info = planner.chat(SYSTEM, content, max_tokens=max_tokens, timeout=120, task="express_plan")
    rows = {str(r.get("id", "")).strip(): r for r in result.get("shots", []) if isinstance(r, dict)}
    groups = {}
    for row in result.get("groups", []) if isinstance(result.get("groups"), list) else []:
        if not isinstance(row, dict) or not str(row.get("id", "")).strip():
            continue
        gid = str(row["id"]).strip()[:12]
        person = _text(row.get("person"), 40)
        if person and person not in script:
            person = ""
        zh = _queries(row.get("zh"), 40)
        if person:
            zh = [q if person in q else f"{person} {q}" for q in zh] or [f"{person} 历史影像"]
        groups[gid] = {"id": gid, "zh": zh, "en": [] if person else _queries(row.get("en"), 60, ascii_only=True),
                       "person": person, "shots": []}
    topic = result.get("topic") if isinstance(result.get("topic"), dict) else {}
    topic = {"zh": _queries(topic.get("zh"), 40), "en": _queries(topic.get("en"), 60, ascii_only=True)}
    for i, shot in enumerate(shots):
        row = rows.get(f"S{i+1}", {})
        gid = str(row.get("g", "")).strip()[:12]
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
    usage = info.get("usage") or {}
    return {"mode": "express", "groups": list(groups.values()), "topic": topic,
            "model": info.get("model"), "usage": usage}


def plan(project: dict, check=None) -> tuple[list[dict], dict]:
    script = project.get("script", "").strip()
    if not script:
        raise ValueError("请先输入文稿")
    shots = planner.rule_shots(script)
    info = annotate(shots, script, project.get("requirements", ""), check)
    return shots, info
