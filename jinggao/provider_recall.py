"""Bounded publisher-only search opportunities after a real empty response.

Short terms are literal retrieval hints, never complete semantic descriptions
or permission to accept a clip. Original narration and constraints stay intact.
"""
from __future__ import annotations

from collections import defaultdict
import re

from . import editorial_contract, identity, retrieval_terms

PROVIDER = "中新网"


def _normal(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def _words(query, source, *, local=False):
    profile = retrieval_terms.source_profile([query], source)
    terms = set(profile["source_terms"])
    # Packaging/production are insufficient subject anchors, but remain useful
    # local hints when paired with a separately grounded, specific subject.
    if local:
        terms.update(set(profile["generic_terms"]) - retrieval_terms.WEAK - {"需要", "可以", "应该"})
    text = query.lower()
    return sorted(terms, key=lambda term: (text.find(term), -len(term), term))


def _empty_queries(diagnostics):
    result = set()
    for task in diagnostics:
        query = _normal(task.get("query"))
        for row in task.get("providers", []):
            if (query and row.get("provider") == PROVIDER and row.get("status") == "ok"
                    and type(row.get("candidates")) is int and row["candidates"] == 0
                    and _normal(row.get("requested_query")) == query
                    and _normal(row.get("effective_query")) == query):
                result.add(query)
    return result


def _bound_group(project, group, shots):
    contracts = {}
    for index in group["indices"]:
        shot = shots[index]
        contract = editorial_contract.for_shot(shot, project)
        if (not contract or contract["use"] != "illustration" or contract["must"]
                or identity.persons(shot.get("persons", []))
                or (shot.get("event") and shot.get("event_scope", "exact") == "exact")):
            return {}
        contracts[index] = contract
    return contracts


def _query_hints(shot, contract):
    """Use grouping hints only after the caller validates the source contract."""
    if contract["query_hints"]:
        return contract["query_hints"]
    hints = shot.get("retrieval_hints")
    if (not isinstance(hints, dict)
            or hints.get("source_checksum") != shot["editorial_contract"]["checksum"]):
        return []
    queries = hints.get("queries")
    if (not isinstance(queries, list) or not 1 <= len(queries) <= 2
            or any(not isinstance(q, str) or not 1 <= len(q.strip()) <= 200 for q in queries)):
        return []
    return [_normal(q) for q in queries]


def _options(project, groups, shots, pending, empty, searched):
    options = {}
    for group in groups:
        triggers = [q for q in map(_normal, group["queries"]) if q in empty]
        if not triggers:
            continue
        contracts = _bound_group(project, group, shots)
        if not contracts:
            continue
        members = [shots[i] for i in group["indices"]]
        context = "\n".join(s["narration"] for s in members)
        group_query = " ".join(group["queries"])
        anchors = _words(group_query, context)
        occurrences = {term: sum(retrieval_terms._contains(term, s["narration"].lower()) for s in members)
                       for term in anchors}
        # Repetition supplies a shared retrieval anchor, not proof of a noun,
        # identity or an action. POS tags are not reliable for these phrases.
        anchors = [term for term in anchors if occurrences[term] >= min(2, len(members))]
        if not anchors:
            continue
        anchor = max(anchors, key=lambda term: (occurrences[term], -anchors.index(term)))
        for index, contract in contracts.items():
            shot = shots[index]
            if shot["id"] not in pending:
                continue
            hints = _query_hints(shot, contract)
            current = shot["narration"]
            local = _words(current, current, local=True)
            # Take whole words from the original sentence, then locate them in
            # its existing hints. This can recover 烘干 from 烘干机 without
            # inventing a new step or paying for another query-generation call.
            ordered = list(dict.fromkeys(term for hint in hints
                for term in sorted(local, key=lambda value: (hint.lower().find(value), -len(value)))
                if retrieval_terms._contains(term, hint.lower()) and term != anchor))
            value = shot["editorial_contract"]
            local_source = " ".join(value["full_script"][ref["start"]:ref["end"]]
                                    for ref in value["source_refs"])
            # Preserve identifiers even when generated hints omitted them, but
            # never copy an unrelated sibling's aircraft/product code. A bound
            # reference may legitimately supply the identifier for a pronoun.
            literals = list(dict.fromkeys(re.findall(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", local_source)))
            for term in ordered:
                parts, seen = [], set()
                for word in [*literals, anchor, term]:
                    if word.lower() not in seen:
                        seen.add(word.lower())
                        parts.append(word)
                query = " ".join(parts)
                if (query in searched or len(query) >= max(map(len, triggers)) or len(query) > 80):
                    continue
                row = options.setdefault(query, {"groups": [], "indices": set(), "fallback_of": [],
                    "terms": parts, "source_bindings": {}})
                if group not in row["groups"]:
                    row["groups"].append(group)
                row["indices"].add(index)
                row["fallback_of"] = list(dict.fromkeys([*row["fallback_of"], *triggers]))
                row["source_bindings"][shot["id"]] = shot["editorial_contract"]["checksum"]
    return options


def plan(project, groups, shots, remaining, found, diagnostics, broad, *, limit):
    """Share existing rescue slots between broad searches and publisher gaps.

    Returns query->original group objects plus metadata for publisher-only
    entries. No network, model calls, or mutations belong in this planner.
    """
    limit = max(0, int(limit))
    if not limit:
        return {}, {}
    pending = {shot["id"] for shot in remaining}
    searched = {_normal(query) for query in found} | {_normal(query) for query in broad}
    options = _options(project, groups, shots, pending, _empty_queries(diagnostics), searched)
    broad_rows = list(broad.items())
    selected, details, opportunities = {}, {}, defaultdict(int)
    prefer_short = True
    while len(selected) < limit and (options or broad_rows):
        if options and (prefer_short or not broad_rows):
            def priority(item):
                query, row = item
                fresh = [i for i in row["indices"] if not opportunities[i]]
                seconds = sum(max(.1, float(shots[i].get("duration") or 0))/(1+opportunities[i])
                              for i in row["indices"])
                return (not bool(fresh), -seconds, -len(fresh), min(row["indices"]), query)
            query, row = min(options.items(), key=priority)
            selected[query] = row["groups"]
            indices = sorted(row["indices"])
            details[query] = {"provider": PROVIDER, "query_reason": "source_terms_after_empty",
                "fallback_of": row["fallback_of"], "terms": row["terms"],
                "shot_ids": [shots[i]["id"] for i in indices], "source_bindings": row["source_bindings"],
                "metadata_only": True}
            del options[query]
        else:
            query, target_groups = broad_rows.pop(0)
            selected[query] = target_groups
            # A group-wide retry does not establish a local search opportunity
            # for every sentence. Rotate the short queries using their own IDs.
            indices = []
        for index in set(indices):
            opportunities[index] += 1
        prefer_short = not prefer_short
    return selected, details
