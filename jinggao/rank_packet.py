"""Share repeated candidate text without changing any shot or candidate identity."""
from collections import Counter
from copy import deepcopy
import json


PROTOCOL = """\n候选资料采用无损共享文本表 candidate_texts。候选的 source_title、observed_scene、visible_text 字段若为 {\"$candidate_text\":\"T0\"}，其值就是 candidate_texts.T0 的完整原文；请先展开再排序，不能把引用当成空白或缺少观察。其他字段和输出格式不变。candidate_ref 仍属于其所在镜头，禁止返回共享文本表的编号。\n"""
FIELDS = ("source_title", "observed_scene", "visible_text")


def encoded(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def unpack(data):
    original = deepcopy(data)
    table = original.pop("candidate_texts", {})
    for shot in original["shots"]:
        for candidate in shot["candidates"]:
            for key in FIELDS:
                value = candidate.get(key)
                if isinstance(value, dict) and set(value) == {"$candidate_text"}:
                    candidate[key] = table[value["$candidate_text"]]
    return original


def pack(data):
    """Transport only; original evidence must round-trip byte for byte as JSON."""
    if "candidate_texts" in data:
        raise ValueError("Candidate text table already supplied")
    counts = Counter()
    for shot in data["shots"]:
        for candidate in shot["candidates"]:
            for key in FIELDS:
                value = candidate.get(key)
                if not isinstance(value, str):
                    raise ValueError("Candidate evidence must be original text")
                if len(value) >= 80:
                    counts[value] += 1
    table = {f"T{i}": text for i, (text, count) in enumerate(counts.items()) if count >= 2}
    if not table:
        return data, ""
    shared = deepcopy(data)
    references = {value: {"$candidate_text": key} for key, value in table.items()}
    for shot in shared["shots"]:
        for candidate in shot["candidates"]:
            for key in FIELDS:
                candidate[key] = references.get(candidate[key], candidate[key])
    shared["candidate_texts"] = table
    if unpack(shared) != data:
        raise ValueError("Candidate packet changed source evidence")
    # Small savings can be erased by the extra protocol text. Keep the ordinary
    # request and its existing cache identity unless transport is clearly less.
    if len(encoded(data).encode()) - len(encoded(shared).encode()) <= len(PROTOCOL.encode()) + 128:
        return data, ""
    return shared, PROTOCOL
