"""Static stock-media retrieval hints, never translations of source evidence.

Topic seeds find catalogue entries; they are not used for query coverage.
Explicit phrase mappings preserve their Chinese originals and keep aliases
separate. Unknown names, dates, places and qualifiers stay unknown. No model,
network request or external tokenizer is needed.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import re


COMMON_PROVIDERS = frozenset({"pexels", "pixabay", "coverr", "commons"})
STOCK_PROVIDERS = COMMON_PROVIDERS | {"nasa"}
TOPIC_QUERIES = {
    "生活家居": ("home cleaning", "home renovation"),
    "美食烹饪": ("food preparation", "bread baking"),
    "城市街景": ("city streets", "city skyline"),
    "家庭情感": ("family gathering", "parents and children"),
    "职场劳动": ("office workers", "factory workers"),
    "教育校园": ("school classroom", "university campus"),
    "社会民生": ("community services", "elderly care"),
    "时政公共事务": ("press conference", "public meeting"),
    "历史档案": ("archival footage", "archaeological excavation"),
    "传统文化": ("traditional crafts", "calligraphy"),
    "商业财经": ("stock exchange", "retail store"),
    "科技互联网": ("data center", "robotics"),
    "工业制造": ("manufacturing assembly line", "metal machining"),
    "农业乡村": ("crop harvesting", "tea picking", "greenhouse farming"),
    "医疗健康": ("hospital care", "physical therapy"),
    "自然环境": ("forest", "ocean waves", "wetlands"),
    "动物宠物": ("pet care", "wildlife"),
    "旅游地理": ("mountain landscape", "coastal landscape"),
    "交通汽车": ("railway transport", "road traffic"),
    "体育赛事": ("table tennis match", "football match"),
    "娱乐文艺": ("dance performance", "theater rehearsal"),
    "人物访谈": ("interview", "public speaking"),
    "航天国防": ("rocket launch", "space station", "astronaut training"),
    "灾害救援": ("firefighters", "flood rescue"),
    "互联网人物": ("video creator", "live streaming"),
    "互联网事件": ("technology conference", "product launch event"),
    "游戏电竞": ("esports tournament", "video game development"),
    "动漫二次元": ("animation production", "comic convention"),
    "时尚美妆": ("fashion show", "makeup application"),
    "母婴育儿": ("baby care", "children reading"),
    "搞笑喜剧": ("comedy performance", "circus performance"),
    "影视幕后": ("film production", "film set"),
    "音乐演出": ("live music", "orchestra rehearsal"),
    "名人动态": ("red carpet event", "awards ceremony"),
    "法律社会": ("courthouse", "traffic police"),
    "国际见闻": ("international exhibition", "world heritage"),
    "人文阅读": ("bookstore", "library reading"),
    "手工收藏": ("woodworking", "pottery making"),
    "运动户外": ("hiking", "rock climbing"),
    "房产消费": ("furniture manufacturing", "home interior"),
}
# NASA's space, Earth science and technology collection does not imply that
# its media covers every general stock category (or any requested identity).
NASA_QUERIES = {
    "自然环境": ("Earth observation", "ocean", "climate"),
    "科技互联网": ("robotics", "aeronautics", "research laboratory"),
    "航天国防": ("rocket launch", "space station", "astronaut training"),
}

# Only explicit concepts enter live bilingual retrieval. In particular, these
# are not generated from TOPIC_QUERIES or inferred from a query's broad class.
PHRASES = {
    "火箭发射": "rocket launch",
    "火箭": "rocket",
    "空间站": "space station",
    "航天员训练": "astronaut training",
    "卫星发射": "satellite launch",
    "地球观测": "Earth observation",
    "茶叶摊放": "spreading tea leaves",
    "茶叶采摘": "tea picking",
    "茶叶揉捻": "tea rolling",
    "茶叶干燥": "tea drying",
    "茶叶杀青": "tea fixation",
    "茶园": "tea plantation",
    "茶叶": "tea leaves",
    "温室种植": "greenhouse farming",
    "水稻收割": "rice harvesting",
    "果园采摘": "fruit picking",
    "农田灌溉": "crop irrigation",
    "食物制作": "food preparation",
    "烘焙面包": "bread baking",
    "城市街景": "city streets",
    "城市天际线": "city skyline",
    "家庭团聚": "family gathering",
    "办公室办公": "office workers",
    "工厂工人": "factory workers",
    "学校课堂": "school classroom",
    "大学校园": "university campus",
    "社区服务": "community services",
    "养老护理": "elderly care",
    "新闻发布会": "press conference",
    "考古发掘": "archaeological excavation",
    "书法创作": "calligraphy",
    "股票交易": "stock trading",
    "零售商店": "retail store",
    "数据中心": "data center",
    "机器人": "robot",
    "工厂流水线": "factory assembly line",
    "金属加工": "metal machining",
    "医院护理": "hospital care",
    "康复训练": "physical therapy",
    "森林": "forest",
    "海浪": "ocean waves",
    "湿地": "wetlands",
    "宠物护理": "pet care",
    "野生动物": "wildlife",
    "山地风景": "mountain landscape",
    "铁路运输": "railway transport",
    "道路交通": "road traffic",
    "机场": "airport",
    "乒乓球比赛": "table tennis match",
    "足球比赛": "football match",
    "舞蹈表演": "dance performance",
    "戏剧排练": "theater rehearsal",
    "人物访谈": "interview",
    "公众演讲": "public speaking",
    "消防救援": "firefighting",
    "洪水救援": "flood rescue",
    "直播": "live streaming",
    "科技大会": "technology conference",
    "产品发布会": "product launch event",
    "电子竞技比赛": "esports tournament",
    "游戏开发": "video game development",
    "动画制作": "animation production",
    "动漫展": "comic convention",
    "时装秀": "fashion show",
    "化妆": "makeup application",
    "婴儿护理": "baby care",
    "儿童阅读": "children reading",
    "喜剧表演": "comedy performance",
    "电影拍摄": "film production",
    "乐队排练": "band rehearsal",
    "红毯活动": "red carpet event",
    "颁奖典礼": "awards ceremony",
    "法院": "courthouse",
    "交通警察": "traffic police",
    "国际展览": "international exhibition",
    "世界遗产": "world heritage",
    "书店": "bookstore",
    "木工制作": "woodworking",
    "陶器制作": "pottery making",
    "徒步": "hiking",
    "攀岩": "rock climbing",
    "家具制造": "furniture manufacturing",
    "家居收纳": "home organization",
    "房屋装修": "home renovation",
}
CHINESE_ALIASES = {
    "火箭发射": ("发射火箭",),
    "航天员训练": ("宇航员训练",),
    "茶叶摊放": ("摊放茶叶", "鲜叶摊放", "摊放鲜叶", "摊晾茶叶", "茶叶摊晾"),
    "茶叶采摘": ("采茶", "采摘茶叶"),
    "茶叶揉捻": ("揉捻茶叶",),
    "茶叶干燥": ("干燥茶叶",),
    "茶叶杀青": ("绿茶杀青",),
    "食物制作": ("美食制作",),
    "电子竞技比赛": ("电竞比赛",),
    "交通警察": ("交警",),
    "陶器制作": ("制陶",),
}
ENGLISH_ALIASES = {
    "火箭发射": ("rocket launches", "launching a rocket", "launching rockets"),
    "茶叶摊放": ("spread tea leaves", "tea leaves spreading"),
    "茶叶采摘": ("picking tea leaves", "tea plucking", "plucking tea leaves"),
    "茶叶揉捻": ("rolling tea leaves",),
    "茶叶干燥": ("drying tea leaves",),
    "茶叶": ("tea leaf",),
    "机器人": ("robots",),
    "工厂流水线": ("manufacturing assembly line",),
    "数据中心": ("data centre",),
    "城市街景": ("city street",),
    "机场": ("airports",),
    "法院": ("courthouses",),
}


def queries(topic: str, provider: str) -> tuple[str, ...]:
    """Return bounded topic seeds; unknown providers/topics have no fallback."""
    provider = provider.lower().strip()
    if provider == "nasa":
        return NASA_QUERIES.get(topic, ())
    return TOPIC_QUERIES.get(topic, ()) if provider in COMMON_PROVIDERS else ()


@lru_cache(maxsize=256)
def _english_pattern(phrase):
    # Preserve word boundaries: rocket != rocketry, launch != launcher, and
    # tea != teacher. Hyphens may separate the words of one known phrase.
    words = r"[\s-]+".join(re.escape(word) for word in phrase.split())
    return re.compile(r"(?<![A-Za-z0-9_])" + words + r"(?![A-Za-z0-9_])", re.I)


def _variants(concept):
    return (PHRASES[concept], *ENGLISH_ALIASES.get(concept, ()))


def query_profile(text: str) -> dict:
    """Explain recognized retrieval hints without altering the original text.

    A longest phrase masks overlapping generic words. Only query wrappers are
    ignored; unrecognized constraints prevent strong bilingual recall.
    """
    matches = []
    for concept in PHRASES:
        for phrase in (concept, *CHINESE_ALIASES.get(concept, ())):
            matches.extend((m.start(), m.end(), concept) for m in re.finditer(re.escape(phrase), text))
        for phrase in _variants(concept):
            matches.extend((m.start(), m.end(), concept) for m in _english_pattern(phrase).finditer(text))
    accepted, end = [], -1
    for start, stop, concept in sorted(matches, key=lambda row: (row[0], -(row[1] - row[0]))):
        if start >= end:
            accepted.append((start, stop, concept))
            end = stop
    remaining = list(text)
    for start, stop, _ in accepted:
        remaining[start:stop] = " " * (stop - start)
    residual = "".join(remaining)
    # Do not ignore names, years, geography, negation, failure/success, or a
    # broad category. Those constraints still require the original query.
    residual = re.sub(r"视频素材|视频片段|视频画面|高清视频|纪录片|视频|素材|画面|片段|实拍|高清|现场", " ", residual)
    residual = re.sub(r"(?<![A-Za-z0-9_])(?:footage|video|clip|4k|1080p|720p)(?![A-Za-z0-9_])", " ", residual, flags=re.I)
    unknown = tuple(re.findall(r"[\w]+", residual))
    concepts = tuple(dict.fromkeys(concept for _, _, concept in accepted))
    return {"original": text, "concepts": concepts, "queries": tuple(PHRASES[c] for c in concepts)[:4],
            "unmapped": unknown, "complete": bool(concepts) and not unknown}


def query_terms(text: str) -> tuple[str, ...]:
    """At most four explicit English phrases; never a guessed name translation."""
    return query_profile(text)["queries"]


def coverage(query: str, candidate_text: str) -> float:
    """Conservative phrase coverage for retrieval, not visual/person evidence.

    An unknown constraint gives zero bilingual coverage even when another
    mapped phrase occurs. The caller may still search the unchanged original.
    """
    profile = query_profile(query)
    if not profile["complete"]:
        return 0.0
    matched = sum(any(_english_pattern(phrase).search(candidate_text) for phrase in _variants(concept))
                  for concept in profile["concepts"])
    return matched / len(profile["concepts"])


def suggested_query(original, english):
    """Validate an optional search hint, never a translation/evidence verdict."""
    if not isinstance(original, str) or not isinstance(english, str):
        return None
    english = " ".join(english.split())
    if not 1 <= len(english) <= 100 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 '-]*", english):
        return None
    if not re.search(r"[A-Za-z]", english):
        return None
    # Never lose an explicit year, model number or Latin name from the query.
    # Chinese names cannot be proved by this function; the caller blocks named
    # people/exact events and all actual content still passes the usual gates.
    literals = re.findall(r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*", original)
    if any(not re.search(r"(?<![A-Za-z0-9_])" + re.escape(v) + r"(?![A-Za-z0-9_])", english, re.I)
           for v in literals):
        return None
    return english


def paired_queries(queries, english):
    """Pair before deduplication; ambiguous or missing entries stay unknown."""
    if not isinstance(queries, list) or any(not isinstance(q, str) for q in queries):
        return {}
    values = english if isinstance(english, list) and len(english) == len(queries) else [None] * len(queries)
    pairs = {}
    for query, value in zip(queries, values, strict=True):
        query = query.strip()
        clean = suggested_query(query, value)
        pairs.setdefault(query, set()).add(clean)
    return {q: next(iter(values)) if len(values) == 1 else None for q, values in pairs.items()}


def group_context(project, shots):
    """Bind ephemeral hints to source text and shot constraints, not approval."""
    fields = ("id", "narration", "persons", "event_persons", "event", "event_scope", "query",
              "search_queries", "visual_plan", "editorial_contract", "visual_origin", "locked")
    data = {"script": project.get("script"), "requirements": project.get("requirements"),
            "shots": [{k: shot.get(k) for k in fields} for shot in shots]}
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def metadata_matches(english, text):
    """Every literal word must occur; metadata can only earn a search slot."""
    words = re.findall(r"[A-Za-z0-9]+", english or "")
    return bool(words) and all(re.search(r"(?<![A-Za-z0-9_])" + re.escape(word) +
                                         r"(?![A-Za-z0-9_])", text, re.I) for word in words)


def metadata_overlap(english, text):
    """Partial stock tags earn inspection only, never query/content coverage.

    APIs return tag lists, not sentences: 'on', 'the', camera words, and common
    inflections should not exclude a potentially useful clip before a model
    sees even its metadata. Keep metadata_matches strict for its other users.
    """
    weak = {"a", "an", "the", "in", "on", "at", "of", "and", "with", "to", "for", "by",
            "video", "videos", "footage", "clip", "clips", "close", "up", "view", "shot"}
    def words(value):
        return set(re.findall(r"[a-z]+", (value or "").lower())) - weak
    def forms(word):
        variants = {word}
        if len(word) > 4 and word.endswith("ies"):
            variants.add(word[:-3]+"y")
        if len(word) > 4 and word.endswith("ves"):
            variants.update((word[:-3]+"f", word[:-3]+"fe"))
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            variants.add(word[:-1])
        if len(word) > 5 and word.endswith("ing"):
            stem = word[:-3]
            variants.update((stem, stem+"e"))
            if len(stem) > 2 and stem[-1] == stem[-2]:
                variants.add(stem[:-1])
        return variants
    query_words = words(english)
    observed = set().union(*(forms(word) for word in words(text)))
    matched = sorted(word for word in query_words if forms(word) & observed)
    return matched if query_words and len(matched) >= min(2, len(query_words)) else []


def missing_stock_term(english, rows):
    """One extra lookup when a phrase's result page omits a key term entirely.

    Providers may return common-tag matches while dropping the rarer activity.
    This only proposes another metadata lookup, never relaxes shot conditions.
    Numeric/special-name phrases are not shortened here. Named/exact groups
    are already excluded by the caller's source-bound stock hint gate.
    """
    if not rows or not isinstance(english, str) or not re.fullmatch(r"[a-z]+(?: [a-z]+)+", english):
        return None
    context = {"process", "tutorial", "scene", "scenes", "home", "indoor", "outdoor", "people"}
    terms = list(dict.fromkeys(word for word in english.split()
                              if len(word) > 2 and word not in context and metadata_overlap(word, word)))
    if len(terms) < 2:
        return None
    present = {word for word in terms if any(metadata_overlap(word,
        row.get("title", "") + " " + row.get("description", "")) for row in rows)}
    # With no relevant tags at all, do not guess which term was discarded.
    if not present:
        return None
    return next((word for word in terms if word not in present), None)
