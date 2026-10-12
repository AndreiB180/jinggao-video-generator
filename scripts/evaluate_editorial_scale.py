"""Recheck saved pixel-bound windows without buying vision or editing projects.

Default: measure complete production text payloads offline. --paid makes one
bounded pass through the normal text judge. Expected labels are only read after
the pass, never included in requests. This is a regression, not a video trial.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from jinggao import (config, costs, editorial_contract, model_routing, network, neutral_observer, paths,
                     planner, review_batches, visual_quality)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/recovery-20261010"
TEA_ID = "939304dd46ff4758b3b97eacbaa8c8da"
EVAL_ID = "aliyun-evaluation-20261009"


def bind_original_source(row):
    """Restore the fixture's omitted unit ID from its hash-bound original.

    No contract, source quote, observation, or expected outcome is rewritten.
    """
    if editorial_contract.for_shot(row["shot"], row["project_context"]):
        return
    manifest = row["evidence_manifest"]
    snapshot = Path(manifest["snapshot_path"]).read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == manifest["snapshot_sha256"]
    original = json.loads(snapshot)["shots"][row["shot_number"]-1]
    assert original["id"] == row["shot"]["id"]
    assert original["editorial_contract"] == row["shot"]["editorial_contract"]
    assert "source_unit_id" not in row["shot"]
    row["shot"]["source_unit_id"] = original["source_unit_id"]
    assert editorial_contract.for_shot(row["shot"], row["project_context"])


def load_sets():
    holdout = json.loads((BASE / "editorial-scale-holdout-01/ready-records.json").read_text())["records"]
    project = json.loads((BASE / "fresh-pipeline-pilot/project-result.json").read_text())
    assets = {a["id"]: a for a in project["assets"]}
    tea = []
    for shot in project["shots"]:
        if not shot.get("selected_asset"):
            continue
        audit = shot["verification"]["identity_audit"]
        tea.append({"case_id": shot["id"], "shot": shot,
                    "asset": assets[shot["selected_asset"]],
                    "project_context": project, "requirements": project.get("requirements", ""),
                    "narrative_context": planner.narrative_context(project, shot),
                    "frame_paths": [str(paths.DATA / "projects" / TEA_ID / f["path"])
                                    for f in shot["match_preview"]["frames"]],
                    "observation": {"scene_description": audit["scene_description"],
                                    "frames": audit["frame_observations"]},
                    "visual_quality": shot["verification"]["visual_quality"]})
    assert len(tea) == 2 and len(holdout) == 7
    for row in [*tea, *holdout]:
        bind_original_source(row)
        row["frame_paths"] = [Path(p) for p in row["frame_paths"]]
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in row["frame_paths"]]
        assert hashes == row["visual_quality"]["frame_sha256"], "Saved facts no longer bind these pixels"
        assert len(hashes) == len(row["observation"]["frames"])
    return [(TEA_ID, tea), (EVAL_ID, holdout)]


def run_set(project_id, rows, *, paid, label, strict_review=False):
    cfg = config.read() | {"pixel_observer_mode": "separate", "model_job_budget_cny": "0.02",
                          "_strict_text_review_trial": strict_review}
    model = model_routing.model(cfg, "evidence_review", vision=False)
    before = costs.summary(project_id)["project"]
    original_chat, original_post = planner.chat, network.post
    measurements, calls, wire_calls = [], [], []
    observations = {r["shot"]["id"]: r["observation"] for r in rows}
    qualities = {r["shot"]["id"]: r["visual_quality"] for r in rows}
    # Only these production fields enter the auditor, excluding historical
    # reviews and independent expected outcomes stored alongside the samples.
    records = [{k: deepcopy(r[k]) for k in (
        "shot", "asset", "project_context", "requirements", "narrative_context", "frame_paths")}
        for r in rows]

    def capture_post(url, **kwargs):
        # Preserve the provider's actual answer before normalization/vetoes.
        # Credentials and request/response headers never enter the report.
        if not paid:
            raise AssertionError("Offline evaluation must not send a model request")
        if url != cfg["model_base_url"].rstrip("/") + "/chat/completions":
            raise AssertionError("Unexpected endpoint during text-only evaluation")
        entry = {"payload": deepcopy(kwargs["json"])}
        wire_calls.append(entry)
        sent_at = time.monotonic()
        try:
            response = original_post(url, **kwargs)
            entry["http_status"] = response.status_code
            try:
                body = response.json()
                if isinstance(body, dict):
                    entry["model_response"] = {k: body[k] for k in ("model", "choices", "usage") if k in body}
            except ValueError:
                entry["non_json_response"] = True
            return response
        except Exception as exc:
            entry["transport_error_type"] = type(exc).__name__
            raise
        finally:
            entry["elapsed_seconds"] = round(time.monotonic()-sent_at, 3)

    def invoke(system, content, **kwargs):
        def prepare(items):
            return planner._prepare_chat_payload(cfg, model, system, items,
                task=kwargs["task"], vision=kwargs.get("vision", False),
                max_tokens=kwargs.get("max_tokens"), reasoning=kwargs.get("reasoning", False),
                reasoning_effort=kwargs.get("reasoning_effort"))
        first = prepare(content)
        batches = review_batches.partition(content, prepare, first, cfg,
                                           review_batches.input_limit(cfg, model))
        measurements.append({"record_count": len(content),
            "combined_bound": costs._input_bound(cfg, first["payload"]),
            "combined_packet": first["packet"].stats,
            "batches": [{"record_count": len(b["packet"].original),
                         "input_bound": costs._input_bound(cfg, b["payload"]),
                         "packet": b["packet"].stats} for b in batches]})
        if not paid:
            return {"reviews": []}, {"cache_hit": True, "request_count": 0}
        answer, info = original_chat(system, content, **kwargs)
        calls.append(info)
        return answer, info

    started = time.monotonic()
    answers, error = {}, ""
    try:
        with patch.object(config, "read", lambda: cfg.copy()), \
            patch.object(neutral_observer, "observe", lambda batch, check: (
                {r["shot"]["id"]: observations[r["shot"]["id"]] for r in batch}, 0)), \
            patch.object(visual_quality, "observe", lambda batch, check: (
                {r["shot"]["id"]: qualities[r["shot"]["id"]] for r in batch}, {"model_requests": 0})), \
            patch.object(planner, "chat", invoke), \
            patch.object(network, "post", capture_post), \
            costs.scope(project_id, f"editorial-scale-{label}", "editorial_scale_regression"):
            answers = planner.audit_selected_frames(records, lambda: None)
    except Exception as exc:
        # Preserve failed paid attempts and the resulting ledger totals even
        # when the batch cannot return a complete set of verdicts.
        error = type(exc).__name__ + ": " + str(exc)[:500]
    after = costs.summary(project_id)["project"]
    return {"project_id": project_id, "paid": paid, "strict_review_trial": strict_review,
            "elapsed_seconds": round(time.monotonic()-started, 3),
            "new_requests": after["requests"]-before["requests"],
            "new_accounted_cny": round(after["total_cny"]-before["total_cny"], 6),
            "measurements": measurements, "calls": calls, "wire_calls": wire_calls, "answers": answers,
            "completed": not error, "error": error,
            "before": before, "after": after,
            "case_ids": {r["shot"]["id"]: r["case_id"] for r in rows}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paid", action="store_true")
    parser.add_argument("--label", default="v23")
    parser.add_argument("--strict-review", action="store_true",
                        help="Opt into the unpromoted structured-review trial for this process only")
    parser.add_argument("--sets", choices=("all", "tea", "holdout"), default="all")
    parser.add_argument("--cases", nargs="+", help="Evaluate only these case IDs; unknown IDs fail before any API call")
    args = parser.parse_args()
    output = BASE / "editorial-scale-regression" / f"{args.label}-{'paid' if args.paid else 'offline'}.json"
    if output.exists():
        raise SystemExit(f"Preserve existing evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = []
    for project_id, records in load_sets():
        if args.sets == "tea" and project_id != TEA_ID or args.sets == "holdout" and project_id != EVAL_ID:
            continue
        selected.append((project_id, records))
    if args.cases:
        known = {r["case_id"] for _, records in selected for r in records}
        if missing := set(args.cases) - known:
            raise SystemExit("Unknown case IDs in the selected set: " + ", ".join(sorted(missing)))
        selected = [(pid, [r for r in records if r["case_id"] in args.cases]) for pid, records in selected]
    report = {"scope": "saved pixel-bound historical regression with original source contracts; fixture unit IDs restored from hash-bound original snapshots; not a complete-video or fresh-script test",
              "production_modified": False, "vision_calls": 0, "selected_sets": args.sets,
              "selected_cases": args.cases, "sets": []}
    for project_id, records in selected:
        if not records:
            continue
        result = run_set(project_id, records, paid=args.paid, label=args.label, strict_review=args.strict_review)
        report["sets"].append(result)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({k: result[k] for k in (
            "project_id", "paid", "elapsed_seconds", "new_requests", "new_accounted_cny", "measurements")},
            ensure_ascii=False), flush=True)
        if not result["completed"]:
            break
    report["completed"] = bool(report["sets"]) and all(s["completed"] for s in report["sets"])
    if args.paid:
        report["scoring_note"] = "Matches stored historical labels only; reasons and pixels need independent audit before any quality claim."
        expected = json.loads((BASE / "editorial-scale-holdout-01/expected.json").read_text())
        accepted = {r["shot_id"]: r["accepted_review_statuses"] for r in expected["samples"]}
        checks = [{"shot_id": sid, "case_id": result["case_ids"][sid],
                   "status": row["status"], "passed": row["status"] in accepted.get(sid, ["unknown", "conflict"]),
                   "reason": row["reason"]}
                  for result in report["sets"] for sid, row in result["answers"].items()]
        report["checks"] = checks
        expected_count = sum(len(s["case_ids"]) for s in report["sets"])
        report["passed"] = report["completed"] and len(checks) == expected_count and all(c["passed"] for c in checks)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({"passed": report["passed"], "checks": checks}, ensure_ascii=False), flush=True)
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
