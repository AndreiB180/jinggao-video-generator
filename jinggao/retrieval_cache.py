"""Resume complete retrieval group proposals; never transfer scene approval."""
import hashlib
import json
from copy import deepcopy

from . import config, model_protocol, model_routing, paths
from .plan_cache import Batch

# Selection/review/progress are intentionally absent. Even constraints not
# repeated in the grouping prompt must invalidate reuse when the user changes
# them; the model request itself keeps its existing complete, bounded shape.
CONTEXT_FIELDS = ("id", "narration", "duration", "subject", "persons", "event_persons", "event",
                  "event_scope", "evidence_mode", "location", "era", "exclusions", "visual",
                  "visual_plan", "visual_origin", "locked", "overlay_label", "preferred_media",
                  "query", "search_queries", "source_unit_id", "editorial_contract")


def shot_context(shot):
    return {key: deepcopy(shot.get(key)) for key in CONTEXT_FIELDS}


def valid(response, count):
    # Optional stock hints cannot invalidate an otherwise reusable grouping:
    # sanitization drops bad hints without another paid repair request.
    groups = response.get("groups") if isinstance(response, dict) else None
    if not isinstance(groups, list) or not 1 <= len(groups) <= 16:
        return False
    seen = []
    for group in groups:
        if (not isinstance(group, dict) or not isinstance(group.get("label"), str)
                or not 1 <= len(group["label"].strip()) <= 200):
            return False
        indices, queries = group.get("indices"), group.get("queries")
        if (not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < count for i in indices)
                or not isinstance(queries, list) or not 1 <= len(queries) <= 2
                or any(not isinstance(q, str) or not 1 <= len(q.strip()) <= 200 for q in queries)):
            return False
        seen.extend(indices)
    return sorted(seen) == list(range(count))


def sanitize(response):
    from . import stock_queries
    response = deepcopy(response)
    groups = response.get("groups") if isinstance(response, dict) else None
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, dict) or "stock_queries" not in group:
            continue
        queries = group.get("queries")
        if not isinstance(queries, list) or any(not isinstance(q, str) for q in queries):
            group.pop("stock_queries", None)
            continue
        pairs = stock_queries.paired_queries(queries, group["stock_queries"])
        group["stock_queries"] = [pairs.get(q.strip()) for q in queries]
    return response


def context_shots(project, shots):
    """Use one complete plan for grouping; pending membership is only a view."""
    complete = project.get("shots") or shots
    for rows in (complete, shots):
        if (not isinstance(rows, list) or not rows or len(rows) > 160
                or any(not isinstance(s, dict) or not isinstance(s.get("id"), str)
                       or not s["id"] for s in rows)
                or len({s["id"] for s in rows}) != len(rows)):
            raise ValueError("素材池需要不重复的完整分镜编号")
    if not {s["id"] for s in shots} <= {s["id"] for s in complete}:
        raise ValueError("待配镜头不属于当前完整分镜，未请求分组")
    return complete


def normalize(response, shots):
    """Complete the existing deterministic retrieval fallback before caching.

    This assigns search opportunities only. It cannot modify shot conditions,
    identities, source evidence, selected assets or approval.
    """
    from . import stock_queries
    response = sanitize(response)
    rows = response.get("groups") if isinstance(response, dict) else None
    if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
        raise ValueError("素材池分组格式无效")
    groups = []
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("label"), str)
                or not 1 <= len(row["label"].strip()) <= 200
                or not isinstance(row.get("indices"), list) or not row["indices"]
                or any(type(i) is not int or not 0 <= i < len(shots) for i in row["indices"])):
            raise ValueError("素材池包含无效分组标签或镜头编号")
        row = deepcopy(row)
        queries = row.get("queries")
        if (not isinstance(queries, list) or not 1 <= len(queries) <= 2
                or any(not isinstance(q, str) or not 1 <= len(q.strip()) <= 200 for q in queries)):
            original = shots[row["indices"][0]]
            queries = [original.get("query", ""), *(original.get("search_queries") or [])]
            queries = list(dict.fromkeys(q.strip() for q in queries
                if isinstance(q, str) and 1 <= len(q.strip()) <= 200))[:2]
            row.pop("stock_queries", None)
            if not queries:
                raise ValueError("素材池缺少有效检索词")
        pairs = stock_queries.paired_queries(queries, row.get("stock_queries"))
        group = {"label": row["label"].strip(), "indices": row["indices"],
                 "queries": list(dict.fromkeys(q.strip() for q in queries))}
        if "stock_queries" in row:
            group["stock_queries"] = [pairs.get(q) for q in group["queries"]]
        groups.append(group)
    assigned = set()
    for group in groups:
        group["indices"] = list(dict.fromkeys(i for i in group["indices"] if i not in assigned))
        assigned.update(group["indices"])
    groups = [g for g in groups if g["indices"]]
    for missing in sorted(set(range(len(shots))) - assigned):
        nearest = min(groups, key=lambda g: min(abs(missing - i) for i in g["indices"]))
        nearest["indices"].append(missing)
    for group in groups:
        group["indices"].sort()
    result = {"groups": groups}
    if not valid(result, len(shots)):
        raise ValueError("素材池规范化后未完整覆盖原分镜，未继续检索")
    return result


def project_groups(groups, complete, shots):
    """Translate full-plan indices by immutable shot ID, never by position."""
    context_shots({"shots": complete}, shots)
    selected = {s["id"]: i for i, s in enumerate(shots)}
    result = []
    for group in groups:
        indices = sorted(selected[complete[i]["id"]] for i in group["indices"]
                         if complete[i]["id"] in selected)
        if indices:
            result.append(deepcopy(group) | {"indices": indices})
    if not valid({"groups": result}, len(shots)):
        raise ValueError("素材池投影未完整覆盖待配镜头，未继续检索")
    return result


class Groups(Batch):
    def __init__(self, project, protocol, payload, options):
        self.inputs = deepcopy((project, protocol, payload, options))
        self.to_stable = self.to_current = {}
        cfg = config.read()
        self.model = model_routing.model(cfg, options["task"])
        request = {"model": self.model, "system": protocol,
                   "payload": payload, "options": options}
        model_protocol.apply(request, cfg)
        key = {"version": 2, "request": request, "endpoint": cfg["model_base_url"],
               "context": [shot_context(shot) for shot in project.get("shots", [])]}
        digest = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.path = paths.DATA / "cache" / "retrieval-groups" / (digest+".json") if project.get("id") else None

    def current(self):
        return Groups(*self.inputs).path == self.path

    def save(self, response):
        super().save(response)
        if self.path is not None and self.read() != response:
            # Do not report a resumable success if the best-effort base writer
            # swallowed an I/O failure. No automatic second model call.
            error = ValueError("检索分组已生成但缓存未完整保存，已停止后续检索；请检查缓存目录")
            error.grouping_response = deepcopy(response)
            raise error
