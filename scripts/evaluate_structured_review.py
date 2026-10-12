"""Bounded paid regression: same saved, pixel-bound facts; new text transport.

No project or video is modified. Replaying the identical audit should issue no
second text request. Existing per-project/provider ledgers remain in force.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from jinggao import config, costs, editorial_contract, neutral_observer, planner, visual_quality

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/recovery-20261009/structured-review"
SAMPLES = ROOT / "artifacts/recovery-20261009/aliyun-evaluation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="v1", choices=("v1", "v2", "v3"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SAMPLES / "fixed-sample-manifest.json").read_text())
    saved = json.loads((SAMPLES / "audit-responses-v3.json").read_text())
    records, observations, qualities = [], {}, {}
    for case in manifest["cases"]:
        name, script = case["case"], case["narration"]
        paths = [Path(f["path"]) for f in case["frames"]]
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        assert hashes == saved[name]["visual_quality"]["frame_sha256"], "Changed pixels: do not reuse saved facts"
        assert len(paths) == len(saved[name]["frame_observations"])
        project = {"script": script, "requirements": ""}
        shot = planner.new_shot(narration=script, visual="", query=script)
        shot.update(id=name, source_unit_id="u1")
        reply = {"contracts": [{"ref": "s1", "use": "illustration", "refs": [],
                  "facts": [["activity", "s1", script]], "must": [], "queries": [script]}]}
        bundle = editorial_contract.compile_responses(project, [shot], [reply])
        project["shots"] = editorial_contract.attach(project, [shot], bundle)
        records.append({"shot": project["shots"][0], "asset": {"id": "sample-"+name, "kind": "video"},
                        "project_context": project, "frame_paths": paths})
        observations[name] = {"scene_description": saved[name]["scene_description"],
                              "frames": saved[name]["frame_observations"]}
        qualities[name] = saved[name]["visual_quality"]
    pid = "aliyun-evaluation-20261009"
    before = costs.summary(pid)["project"]
    cfg = config.read() | {"model_job_budget_cny": "0.02"}
    original_chat = planner.chat
    calls = []
    def capture(system, content, **kwargs):
        result, info = original_chat(system, content, **kwargs)
        calls.append({"task": kwargs.get("task"), "info": info})
        return result, info
    started = time.monotonic()
    with patch.object(config, "read", lambda: cfg.copy()), \
            patch.object(neutral_observer, "observe", lambda rows, check: (
                {r["shot"]["id"]: observations[r["shot"]["id"]] for r in rows}, 0)), \
            patch.object(visual_quality, "observe", lambda rows, check: (
                {r["shot"]["id"]: qualities[r["shot"]["id"]] for r in rows}, {"model_requests": 0})), \
            patch.object(planner, "chat", capture), \
            costs.scope(pid, "structured_review_v1", "structured_review_regression"):
        first = planner.audit_selected_frames(deepcopy(records), lambda: None)
        middle = costs.summary(pid)["project"]
        repeat = planner.audit_selected_frames(deepcopy(records), lambda: None)
    after = costs.summary(pid)["project"]
    checks = [{"case": c["case"], "expected": c["expected"], "actual": first[c["case"]]["status"],
               "passed": (first[c["case"]]["status"] == "supported") == (c["expected"] == "supported"),
               "repeat_same": first[c["case"]]["status"] == repeat[c["case"]]["status"]}
              for c in manifest["cases"]]
    result = {"checks": checks, "elapsed_seconds": round(time.monotonic()-started, 3),
              "new_estimated_cost_cny": round(after["total_cny"]-before["total_cny"], 6),
              "first_paid_requests": middle["requests"]-before["requests"],
              "repeat_paid_requests": after["requests"]-middle["requests"],
              "before": before, "after": after, "calls": calls, "results": first,
              "observations_reused_and_pixel_hash_checked": True, "production_project_modified": False}
    (OUT / f"fixed-sample-regression-{args.label}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in {"results", "before", "after"}}, ensure_ascii=False))
    assert all(c["passed"] and c["repeat_same"] for c in checks)
    assert result["repeat_paid_requests"] == 0


if __name__ == "__main__":
    main()
