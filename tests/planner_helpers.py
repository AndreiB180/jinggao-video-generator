def source_contract_response(request, *, rows=None, queries=("相关说明资料",)):
    """Source-contract schema adapter, without inferring semantic obligations.

    Tests supply any facts/must explicitly. Generated visual suggestions never
    become conditions merely because this fixture received a planner request.
    """
    assert request["stage"] == "source_edit_contract_v1"
    rows = rows or {}
    return {"contracts": [
        {"ref": ref, "use": "illustration", "refs": [], "facts": [], "must": [],
         "queries": list(queries)} | rows.get(ref, {})
        for ref in request["targets"]
    ]}


def consistency_response(request, *, patches=None, purposes=None):
    """Minimal response adapter for existing planner.chat test doubles.

    Branch on request['stage'] == 'plan_consistency' before handling source_units.
    This makes fixtures structurally coherent; it does not simulate a semantic model.
    """
    patches, purposes = patches or {}, purposes or {}
    reviews = []
    for shot in request["shots"]:
        sid = shot["shot_id"]
        purpose = purposes.get(sid, "event_record" if shot.get("event") and shot.get("event_scope") == "exact"
                               else "illustration")
        patch = {}
        if purpose == "event_record":
            if shot.get("evidence_mode") != "event":
                patch["evidence_mode"] = "event"
        else:
            if shot.get("event_scope") != "background":
                patch["event_scope"] = "background"
            mode = "person" if shot.get("persons") else "generic"
            if shot.get("evidence_mode") != mode:
                patch["evidence_mode"] = mode
            if not any(word in shot.get("overlay_label", "") for word in ("资料", "示意")):
                patch["overlay_label"] = "相关资料画面"
        patch.update(patches.get(sid, {}))
        if request.get("visual_contract") and purpose == "illustration":
            merged = shot | patch
            if "visual_plan" not in merged or ("visual_plan" not in patch
                    and any(key in patch for key in ("visual", "subject", "query", "search_queries"))):
                # Fixtures supply their intended literal action here. This is
                # a schema adapter, not a production semantic shortcut.
                patch["visual_plan"] = {"mode": "single", "primary": {
                    "scene": "", "subject": merged.get("subject") or "资料主体",
                    "action": merged.get("visual") or shot["narration"],
                    "query": merged.get("query") or "资料 主动作",
                    "search_queries": merged.get("search_queries", []),
                }, "alternatives": []}
        quote = shot["narration"][:120]
        if quote not in request["full_script"]:
            quote = request["full_script"][:120]
        reviews.append({"shot_id": sid, "status": "revised" if patch else "consistent", "purpose": purpose,
                        "reason": "按原文限定用途并核对视觉条件。",
                        "evidence": [{"source": "script", "quote": quote}], "patch": patch})
    return {"requirements_preserved": True, "reviews": reviews}
