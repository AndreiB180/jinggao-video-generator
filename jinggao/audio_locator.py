"""Use free local speech timestamps to choose image probes, never to approve footage.

The ordinary blind image observer does not receive the transcript. Each chosen
time still has to yield actual pixels and the full selected-window review.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import math
import re
import time

from . import media

MAX_SOURCES = 2
MAX_TARGETED_POINTS = 3
# Exclude connective language from the retrieval score, not from the script.
STOP = set("一个 一种 一些 一下 这些 那些 这个 那个 自己 我们 你们 他们 大家 什么 怎么 这样 那样 可以 需要 时候 然后 就是 也是 不是 还有 没有 不要 不能 已经 还是 因为 所以 如果 但是 不过 其实 可能 一定 真正 开始 之后 之前 最后 今天 每天 每次 不同 完整 过程 视频 教程 画面 进行 当中 里面 上面 下面 比较 非常 觉得 看到 知道 注意 看看 有人 每个 每一 所有 很多 越来 一点 一会".split())
STOP.update(("时间", "位置", "东西", "方面", "部分", "内容"))
GRAM_PREFIX = set("的了着过和与而也就都又再把让从对给在被于很最更将是有不")
GRAM_SUFFIX = set("的了着和与而也就都把从给被于很要去得是有不")


def terms(text):
    """Keep local Chinese bigrams / whole Latin words; no generated synonyms."""
    result = set()
    for token in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z][a-zA-Z0-9_-]+", str(text).lower()):
        if re.search(r"[\u3400-\u9fff]", token):
            result.update(token[i:i+2] for i in range(len(token)-1) if token[i:i+2] not in STOP
                          and token[i] not in GRAM_PREFIX and token[i+1] not in GRAM_SUFFIX)
        elif len(token) >= 3:
            result.add(token)
    return result


@lru_cache(maxsize=512)
def _keywords(text):
    # Jieba is already shipped with the local subtitle component. Whole words
    # avoid treating pronoun fragments (e.g. 的车/回到) as specific objects.
    import jieba.posseg
    return {word: 1.0 if flag.startswith("n") else .4
            for word, flag in jieba.posseg.cut(text.lower())
            if 2 <= len(word) <= 32 and word not in STOP
            and (flag.startswith("n") or flag in {"v", "vn", "eng"})}


def candidates(group_rows):
    """Choose at most two already-inspected ordinary sources per initial wave.

    Explicit neighbour/collision probes keep their own positions. Long POVs,
    exact people/events, and inaccessible sources do not trigger transcription.
    """
    from . import curation, editorial_contract, source_audio
    rows, targets = {}, {}
    for shots, choices in group_rows:
        for candidate in choices:
            if (candidate.get("sample_points") or not source_audio.eligible(candidate)
                    or any(curation.needs_identity(s) for s in shots)):
                continue
            cid = candidate["id"]
            rows[cid] = candidate
            targets.setdefault(cid, {}).update({s["id"]: s for s in shots})

    def priority(cid):
        title = terms(rows[cid].get("title", ""))
        hits = []
        for shot in targets[cid].values():
            # Retrieval already resolved paraphrases such as 看看轮胎 ->
            # 自行车检查. Requiring the title to repeat narration literally
            # otherwise disables audio localization for those same sources.
            hints = editorial_contract.retrieval_fields(shot).get("query_hints", [])
            wanted = terms(" ".join([shot.get("narration", ""), *hints]))
            hits.append(len(title & wanted))
        return (-max(hits, default=0), -sum(h > 0 for h in hits), cid)

    chosen = [cid for cid in sorted(rows, key=priority) if priority(cid)[0] < 0][:MAX_SOURCES]
    return {cid: list(targets[cid].values()) for cid in chosen}


def _windows(words, duration):
    # Overlapping eight-second speech windows retain a short local description
    # and the timestamp of each word; pauses never become invented words.
    for begin in range(0, math.ceil(duration), 4):
        chosen = [w for w in words if begin <= (w["start"]+w["end"])/2 < begin+8]
        if chosen:
            text = "".join(w["text"] for w in chosen)
            yield {"start": chosen[0]["start"], "end": chosen[-1]["end"],
                   "text": text, "words": _keywords(text)}


def locate(words, shots, duration, *, count=4):
    """Replace up to three uniform probes with distinct speech-relevant times.

    A source saying something is not visual evidence. Keep one ordinary probe
    for non-speech material, and never increase the number of paid image probes.
    """
    from . import segment_discovery
    if count != 4 or not 1 <= duration <= 420:
        raise ValueError("音轨定位仅支持最多420秒来源的原有4个采样机会")
    uniform = segment_discovery._points(duration, count)
    windows = list(_windows(words, duration))
    shot_terms = {s["id"]: _keywords(s.get("narration", "")) for s in shots}
    wanted = {t for ts in shot_terms.values() for t in ts}
    for window in windows:
        window["matches"] = {t: 1.0 for t in wanted if t in window["text"]}
        # A weak noun-ending match can recall 轮胎/外胎 or 茶杯/水杯. It only
        # earns a probe and is never expanded into a claimed visual action.
        for t in wanted-window["matches"].keys():
            if (len(t) == 2 and any(ts.get(t) == 1 for ts in shot_terms.values())
                    and any(len(w) == 2 and weight == 1 and w[-1] == t[-1]
                            for w, weight in window["words"].items())):
                window["matches"][t] = .4
    frequency = Counter(t for w in windows for t in w["matches"])
    # Common vocabulary across the transcript should not crowd out a concrete
    # operation that occurs only briefly.
    weights = {t: math.log1p(len(windows)/(1+n)) for t, n in frequency.items()}
    options = []
    for window in windows:
        hits = {sid: sorted(window["matches"].keys() & ts.keys()) for sid, ts in shot_terms.items()}
        scores = {sid: sum(weights[t]*shot_terms[sid][t]*window["matches"][t] for t in ts)
                  for sid, ts in hits.items() if ts}
        if scores:
            at = min(duration-.08, max(.04, (window["start"]+window["end"])/2))
            options.append(window | {"at": round(at, 3), "scores": scores, "hits": hits})
    selected, served = [], Counter()
    # Different speech windows about the same operation are retrieval choices,
    # not proof that a long continuous video interval matches the narration.
    while options and len(selected) < MAX_TARGETED_POINTS:
        row = max(options, key=lambda r: (max(v/(1+served[sid]) for sid, v in r["scores"].items()),
                                         sum(v/(1+served[sid]) for sid, v in r["scores"].items()), -r["at"]))
        selected.append(row)
        served.update(row["scores"].keys())
        options = [r for r in options if abs(r["at"]-row["at"]) >= 8]
    if not selected:
        return [], []
    points = [r["at"] for r in selected]
    # Fill from the unchanged baseline, choosing the widest uncovered location.
    remaining = list(uniform)
    while remaining and len(points) < count:
        at = max(remaining, key=lambda p: (min(abs(p-q) for q in points), -p))
        remaining.remove(at)
        if all(abs(at-q) >= .04 for q in points):
            points.append(at)
    if len(points) != count:
        return [], []
    evidence = [{"at": r["at"], "speech_start": r["start"], "speech_end": r["end"],
                 "speech_text": r["text"], "matched_terms": {k: v for k, v in r["hits"].items() if v}}
                for r in selected]
    return sorted(points), evidence


def refine(candidate, shots, check):
    """A failed free index keeps ordinary image sampling; cancellation propagates."""
    from . import source_audio
    began = time.monotonic()
    info = {"candidate_id": candidate["id"], "source_url": candidate["source_url"],
            "model_requests": 0, "status": "uniform", "content_approved": False}
    try:
        transcript = source_audio.transcript(candidate, check)
        check()
        points, evidence = locate(transcript["words"], shots, candidate["duration"])
        info.update({k: transcript[k] for k in ("cache_hit", "downloaded_bytes", "audio_sha256", "cache_path")
                     if k in transcript})
        if points:
            info.update(status="speech_guided", sample_points=points, hints=evidence)
            return candidate | {"sample_points": points}, info
        info["reason"] = "音轨没有提供与当前口播重合的具体定位词，保留普通抽帧"
    except media.Cancelled:
        raise
    except (ValueError, OSError) as exc:
        check()  # Do not turn a real cancellation/budget stop into a fallback.
        info["reason"] = str(exc)
    finally:
        info["elapsed_seconds"] = round(time.monotonic()-began, 3)
    return candidate, info
