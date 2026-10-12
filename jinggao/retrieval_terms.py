"""Spend on local candidates only when observations contain a specific clue.

This is a retrieval gate, not a content judgment. Misses continue to the online
source search; they are never persisted as rejected or approved footage.
"""
from functools import lru_cache
import re

from . import identity


WEAK = frozenset("""画面 场景 镜头 展示 示意 示意图 资料 纪录片 片段 特写 视角 内部 过程 主题
工作 运行 系统 进行 相关 实际 重要 不同 主要 发生 继续 看到 体现 时刻 照片 图像 人员 团队 数据
标准 问题 日常 生活 应用 普通 使用 融入 背后 现场 合作 作业 地面 历史 首次 长期 衡量 记录 成果
进步 说明 标识 蒙太奇 意象 全景 近景 细节 感受 服务 完成 时间 显示 流程 外观 页面 截图 图表
中国 功能 内容 阶段 部分 节点 公众 开始 介绍 持续 界面 实物
scene frame footage video photo image related background process work working system activity daily people
team view""".split())
TEXT_CLAUSE = re.compile(r"字幕|文字|台标|水印|大标题|标题|标识|标志")
ABSENT = re.compile(r"^.{0,12}?(?:未出现|未观察到|未展示|没有出现|未见|看不到)")

# These words can describe almost any industry. Keep them in search queries,
# but they cannot by themselves justify spending on a source-bound local clip.
# This is deliberately not a list of allowed industries or visual subjects.
SOURCE_GENERIC = WEAK | frozenset("""动作 设备 机械 手工 自动化 现代化 操作 处理 制作 制造 加工
生产 产品 商品 食品 包装 标签 名称 信息 质量 品质 效果 技巧 环境 家庭 表面 变化
工艺 条件 程度 原料 材料 工具 机器 结构 技术 整理 检查 特征 状态 需要 可以 应该
machine equipment action operation production product food label environment change surface""".split())
SOURCE_VERSION = "source_observation_subjects_v2"
SOURCE_ABSENT = re.compile(r"未出现|未观察到|未展示|没有出现|未见|看不到|未发现|没有|不包含|并非|不是")


def _query_words(query):
    try:
        import jieba
        jieba.setLogLevel(30)
        return jieba.lcut(query, HMM=False)
    except ImportError:
        return re.findall(r"[A-Za-z0-9]+", query)


@lru_cache(maxsize=256)
def _source_terms(queries, source_text):
    """Query vocabulary is evidence only when it occurs in the actual source.

    Unknown Han compounds can be split into single characters by jieba. Recover
    only contiguous runs of those characters that also occur verbatim in the
    source; never restore arbitrary bigrams across otherwise segmented words.
    """
    source = source_text.lower()
    proposed = set()
    for query in queries:
        words = _query_words(query)
        proposed.update(word.lower() for word in words if len(word) >= 2
                        and re.fullmatch(r"[\w\u4e00-\u9fff]+", word))
        run = ""
        for word in [*words, ""]:
            if re.fullmatch(r"[\u4e00-\u9fff]", word):
                run += word
                continue
            for start in range(len(run)):
                for end in range(start + 2, min(len(run), start + 12) + 1):
                    term = run[start:end]
                    if term in source:
                        proposed.add(term)
            run = ""
    bound = {term for term in proposed if _contains(term, source)}
    return (frozenset(bound - SOURCE_GENERIC), frozenset(proposed - bound),
            frozenset(bound & SOURCE_GENERIC))


def source_profile(queries, source_text, *, people=()):
    """Build a local spending hint from validated original text, not model prose."""
    terms, query_only, generic = _source_terms(tuple(queries), source_text)
    bound_people = [{"name": p["name"], "aliases": [a for a in p.get("aliases", []) if a in source_text]}
                    for p in people if p.get("name") and p["name"] in source_text]
    return {"version": SOURCE_VERSION, "source_terms": sorted(terms),
            "query_only_terms": sorted(query_only), "generic_terms": sorted(generic),
            # A blind description must not infer a requested person's name.
            # Named identities retain the existing source/OCR retrieval path.
            "subject_terms": [] if bound_people else sorted(_source_subjects(tuple(queries), source_text)),
            "people": bound_people}


@lru_cache(maxsize=256)
def _source_subjects(queries, source_text):
    """Keep source-bound query subjects ahead of incidental shared actions.

    A noun is a retrieval hint, never a claim that the footage depicts it. Each
    alternative query contributes at most one subject. Queries without a
    concrete source-bound noun retain the existing action-clue behaviour.
    """
    try:
        import jieba
        import jieba.posseg
    except ImportError:
        return frozenset()
    jieba.setLogLevel(30)
    source = source_text.lower()
    subjects = set()
    for query in queries:
        for word, tag in jieba.posseg.cut(query, HMM=False):
            word = word.lower()
            if (not tag.startswith("n") or tag in {"nr", "ns", "nt"}
                    or len(word) < 2 or word in SOURCE_GENERIC):
                continue
            # Search segmentation knows compound subwords. A repeatedly named
            # source subject such as 公交 can occur inside 城市公交/公交车; do not
            # require the observer to repeat the query's exact compound.
            parts = {part.lower() for part in jieba.cut_for_search(word, HMM=False)
                     if len(part) >= 2 and part.lower() not in SOURCE_GENERIC
                     and _contains(part.lower(), source)}
            if not parts:
                continue
            subject = max(parts, key=lambda part: (source.count(part), len(part)))
            subjects.add(subject)
            break
    return frozenset(subjects)


def source_support(profile, description, visible_text=""):
    """Return inspection clues only; a miss defers local spending, never footage."""
    scene = " ".join(clause for clause in re.split(r"[。；;，,\n]", description)
                     if not TEXT_CLAUSE.search(clause) and not SOURCE_ABSENT.search(clause)).lower()
    terms = profile["source_terms"]
    subjects = profile.get("subject_terms", [])
    observed = [term for term in sorted(set(terms) | set(subjects)) if _contains(term, scene)]
    observed_subjects = [term for term in subjects if _contains(term, scene)]
    identifiers = [term for term in terms if re.search(r"[a-z]", term) and re.search(r"[0-9]", term)
                   and _contains(term, visible_text.lower())]
    names = [p["name"] for p in profile["people"] if identity.mentions(visible_text, p)]
    return {"version": SOURCE_VERSION,
            "eligible": bool(observed and (not subjects or observed_subjects) or identifiers or names),
            "observed_terms": observed, "observed_subjects": observed_subjects,
            "ocr_identifiers": identifiers, "ocr_names": names}


@lru_cache(maxsize=2048)
def anchors(query):
    try:
        import jieba
        jieba.setLogLevel(30)
        words = jieba.lcut(query, HMM=False)
    except ImportError:
        # Do not restore broad Han bigrams when the optional segmenter is
        # unavailable. Fresh source search still has the original query.
        words = re.findall(r"[A-Za-z0-9]+", query)
    return frozenset(word.lower() for word in words
                     if len(word) >= 2 and re.fullmatch(r"[\w\u4e00-\u9fff]+", word)) - WEAK


def _contains(term, text):
    if re.fullmatch(r"[a-z0-9]+", term):
        return bool(re.search(r"(?<![a-z0-9])"+re.escape(term)+r"(?![a-z0-9])", text))
    return term in text


def observed_match(queries, description, visible_text="", *, people=()):
    terms = anchors(" ".join(queries))
    scene = " ".join(clause for clause in re.split(r"[。；;，,\n]", description)
                     if not TEXT_CLAUSE.search(clause) and not ABSENT.search(clause)).lower()
    if any(_contains(term, scene) for term in terms):
        return True
    # A literal model/flight code or requested full name can earn inspection,
    # but the existing identity, action and provenance gates still apply.
    text = visible_text.lower()
    identifiers = [term for term in terms if re.search(r"[a-z]", term) and re.search(r"[0-9]", term)]
    return (any(_contains(term, text) for term in identifiers)
            or any(identity.mentions(visible_text, person) for person in people))
