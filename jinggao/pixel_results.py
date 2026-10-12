"""Keep independently complete pixel groups, never repair an incomplete group."""
from __future__ import annotations

import json
import re


def unique_object(pairs):
    """Reject duplicate fields before a JSON decoder can silently overwrite."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("像素观察JSON含重复字段")
        result[key] = value
    return result


def complete_groups(raw, expected, parse):
    """Validate whole JSON first, then each unambiguous, expected group.

    A missing/invalid group's facts are never guessed from siblings. Duplicate,
    contradictory or foreign IDs make the entire response ambiguous, so none
    of its groups are reusable. Callers must reject provider truncation before
    using this helper and still report the original batch failure.
    """
    if not isinstance(raw, str):
        return {}
    try:
        body = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()),
                          object_pairs_hook=unique_object)
    except ValueError:
        return {}
    groups = body.get("groups") if isinstance(body, dict) else None
    if not isinstance(groups, list):
        return {}
    indexed = {}
    for group in groups:
        if not isinstance(group, dict):
            return {}
        ident = group.get("id", group.get("group"))
        if (not isinstance(ident, str) or ident not in expected or ident in indexed
                or "id" in group and "group" in group and group["id"] != group["group"]):
            return {}
        indexed[ident] = group
    complete = {}
    for ident, group in indexed.items():
        try:
            complete.update(parse(json.dumps({"groups": [group]}, ensure_ascii=False),
                                  {ident: expected[ident]}))
        except (ValueError, TypeError, KeyError):
            continue
    return complete
