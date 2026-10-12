"""Immutable source IDs for planning transport, not semantic classification.

Full sentences remain in the request. A selected ID expands to its complete
literal part, never a model-written paraphrase. Semicolon clauses may supply an
atomic event anchor; commands and presentation always retain the full sentence.
"""
from copy import deepcopy

VERSION = "source_ids_v2"
COMPILER_VERSION = "source_parent_resolution_v1"


def _parts(text, separators="。！？!?\n"):
    """Keep quoted commands intact and lose no whitespace."""
    pairs = {"“": "”", "‘": "’", "「": "」", "『": "』", "（": "）", "(": ")"}
    stack, ranges, start = [], [], 0
    for index, char in enumerate(text):
        if stack and char == stack[-1]:
            stack.pop()
        elif char in pairs:
            stack.append(pairs[char])
        elif char == '"':
            stack.append(char)
        elif not stack and char in separators:
            end = index + 1
            if text[start:end].strip():
                ranges.append([start, end])
            elif ranges:
                ranges[-1][1] = end
            start = end
    if start < len(text):
        if text[start:].strip():
            ranges.append([start, len(text)])
        elif ranges:
            ranges[-1][1] = len(text)
    if ranges:
        ranges[0][0] = 0
    elif text:
        # Whitespace-only customer requirements still remain in the request;
        # they cannot be cited as a nonempty condition by the old compiler.
        ranges = [[0, len(text)]]
    assert "".join(text[start:end] for start, end in ranges) == text
    return ranges


def build(project, shots):
    from . import editorial_contract as contract

    units, _ = contract._inputs(project, shots)
    lookup = {}

    def parts(ref, text, base, source):
        result = []
        for index, (start, end) in enumerate(_parts(text), 1):
            key = f"{ref}.{index}"
            quote = text[start:end]
            item = {"id": key, "text": quote}
            result.append(item)
            lookup[key] = {"source_ref": ref, "text": quote, "source": source,
                           "start": base + start, "end": base + end}
            clauses = _parts(quote, "；;")
            if len(clauses) > 1:
                item["clauses"] = []
                for clause_index, (left, right) in enumerate(clauses, 1):
                    child = f"{key}c{clause_index}"
                    clause_text = quote[left:right]
                    item["clauses"].append({"id": child, "text": clause_text})
                    lookup[child] = {"source_ref": ref, "text": clause_text, "source": source,
                                     "start": base + start + left, "end": base + start + right,
                                     "parent_id": key, "parent_text": quote,
                                     "parent_span": {"start": base + start, "end": base + end}}
        return result

    script = [{"ref": unit["ref"], "parts": parts(unit["ref"], unit["text"], unit["start"], "script")}
              for unit in units]
    requirements = parts("r", project.get("requirements", ""), 0, "requirements")
    assert "".join(p["text"] for unit in script for p in unit["parts"]) == project["script"]
    assert "".join(p["text"] for p in requirements) == project.get("requirements", "")
    return {"script_units": script, "requirements": requirements}, lookup


def expand(project, shots, responses):
    """Resolve IDs without changing kinds, source refs or literal constraints.

    Non-event clauses retain their entire parent. Only identical citations
    created by that lifting are coalesced, with original slot pointers mapped.
    The model response itself stays untouched and every resolution is traced.
    """
    _, lookup = build(project, shots)
    if not isinstance(responses, list):
        raise ValueError("源ID规划响应必须为完整批次数组")
    result, traces = deepcopy(responses), []

    def citation(key, row, *, field, index, kind=None, name=None):
        if not isinstance(key, str) or key not in lookup:
            raise ValueError(f"{field}引用未知原文ID")
        source = lookup[key]
        refs = row.get("refs")
        if (not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs)
                or source["source_ref"] not in {row.get("ref"), *refs, "r"}):
            raise ValueError(f"{field}原文ID未绑定当前单元或明确依赖")
        selected_quote = source["text"]
        if name is not None:
            if (kind != "person" or field != "facts" or not isinstance(name, str)
                    or not name.strip() or name != name.strip() or len(name) > 80 or name not in selected_quote):
                raise ValueError("仅person事实允许来源ID内逐字短姓名/别名对，不能借选择值改写原句")
        lifted = bool(source.get("parent_id") and not (field == "facts" and kind in {"event", "event_date"}))
        resolved_id = source["parent_id"] if lifted else key
        resolved = lookup[resolved_id]
        quote = name if name is not None else resolved["text"]
        trace = {"ref": row["ref"], "field": field, "index": index, "source_id": key,
                       "source_ref": source["source_ref"], "source": source["source"],
                       "source_span": {"start": source["start"], "end": source["end"]},
                       "selector": name, "compiler_version": COMPILER_VERSION,
                       "resolved_source_id": resolved_id,
                       "resolved_source_span": {"start": resolved["start"], "end": resolved["end"]},
                       "resolution": "complete_parent" if lifted else "literal_id"}
        if source.get("parent_id"):
            trace.update(parent_id=source["parent_id"], parent_text=source["parent_text"],
                         parent_span=deepcopy(source["parent_span"]))
        traces.append(trace)
        value = [kind, source["source_ref"], quote] if kind is not None else [source["source_ref"], quote]
        return value, trace, source.get("parent_id", key), lifted

    def append_citation(expanded, seen, value, trace, family, lifted):
        previous = seen.get(tuple(value))
        if previous and previous[1] == family and (lifted or previous[2]):
            trace.update(compiled_index=previous[0], coalesced=True)
            return previous[0]
        expanded.append(value)
        index = len(expanded)
        seen.setdefault(tuple(value), (index, family, lifted))
        trace.update(compiled_index=index, coalesced=False)
        return index

    def remap(values, slots, kinds, field, *, nonempty=False):
        # Validate original pointers first. Do not make an invalid reference
        # valid merely because a later citation moved into that slot.
        if (not isinstance(values, list) or len(values) > 24 or nonempty and not values
                or any(not isinstance(value, str) or value not in slots for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{field}引用缺失、重复或越权")
        if any(slots[value][1] not in kinds for value in values):
            raise ValueError(f"{field}引用的事实类型无效")
        return list(dict.fromkeys(slots[value][0] for value in values))

    for response in result:
        if not isinstance(response, dict) or set(response) != {"contracts", "presentation"}:
            raise ValueError("源ID规划须完整返回contracts和presentation")
        rows = response["contracts"]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("源ID contracts无效")
        by_ref, slot_maps = {}, {}
        for row in rows:
            if not isinstance(row.get("ref"), str) or row["ref"] in by_ref:
                raise ValueError("源ID契约编号缺失或重复")
            by_ref[row["ref"]] = row
            slots = slot_maps[row["ref"]] = {}
            for field in ("facts", "must"):
                values = row.get(field)
                if not isinstance(values, list) or len(values) > 12:
                    raise ValueError(f"源ID {field}必须完整给出至多12项数组")
                expanded, seen, original = [], {}, set()
                for index, value in enumerate(values, 1):
                    if (not isinstance(value, list) or len(value) not in (2, 3)
                            or not isinstance(value[0], str)):
                        raise ValueError(f"源ID {field}须为[kind,id]，person事实可另带短姓名")
                    if len(value) == 3 and (field != "facts" or value[0] != "person"):
                        raise ValueError("长原文只能引用ID；不得改写或截短条件")
                    if len(value) == 3 and not isinstance(value[2], str):
                        raise ValueError("person事实短姓名必须逐字来自所选原文ID")
                    resolved, trace, family, lifted = citation(value[1], row, field=field, index=index, kind=value[0],
                                                               name=value[2] if len(value) == 3 else None)
                    signature = tuple(value)
                    if signature in original:
                        raise ValueError(f"{field}条件重复")
                    original.add(signature)
                    target = append_citation(expanded, seen, resolved, trace, family, lifted)
                    prefix = "f" if field == "facts" else "m"
                    slots[f"{prefix}{index}"] = (f"{prefix}{target}", value[0])
                    trace.update(original_pointer=f"{prefix}{index}", compiled_pointer=f"{prefix}{target}")
                row[field] = expanded
        presentation = response["presentation"]
        if not isinstance(presentation, list):
            raise ValueError("源ID presentation无效")
        for row in presentation:
            if (not isinstance(row, dict) or not isinstance(row.get("ref"), str)
                    or row["ref"] not in by_ref):
                raise ValueError("源ID呈现编号无效")
            values = row.get("sources")
            if not isinstance(values, list) or not 1 <= len(values) <= 4:
                raise ValueError("源ID呈现sources必须完整给出1–4项数组")
            expanded, seen, original = [], {}, set()
            for index, key in enumerate(values, 1):
                resolved, trace, family, lifted = citation(key, by_ref[row["ref"]],
                                                           field="presentation.sources", index=index)
                if key in original:
                    raise ValueError("presentation.sources条件重复")
                original.add(key)
                append_citation(expanded, seen, resolved, trace, family, lifted)
            row["sources"] = expanded
            slots = slot_maps[row["ref"]]
            if any(old != target for old, (target, _) in slots.items()):
                for field in ("persons", "event_persons"):
                    people = row.get(field)
                    if not isinstance(people, list) or any(not isinstance(person, dict) for person in people):
                        raise ValueError(f"{field}必须逐项给出姓名、别名和原文依据")
                    for person in people:
                        person["refs"] = remap(person.get("refs"), slots, {"person"}, field, nonempty=True)
                row["event"] = remap(row.get("event"), slots, {"event", "event_date"}, "event")
    return result, traces
