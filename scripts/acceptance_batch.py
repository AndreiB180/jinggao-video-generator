"""Record real API batch production without approving or repairing its output.

Usage: python scripts/acceptance_batch.py ARTIFACT_DIRECTORY [--resume]
The supplied directory contains independently authored scripts/manifest.json.
No secrets, browser state or settings files are copied into the artifacts.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time

import httpx


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:8766")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", default="", help="Comma-separated frozen script numbers for a diagnostic pilot")
    parser.add_argument("--title-prefix", default="独立验收")
    args = parser.parse_args()
    root = args.directory.resolve()
    runfile = root / "baseline-run.json"
    with httpx.Client(base_url=args.base, timeout=30, trust_env=False) as client:
        if args.resume:
            run = json.loads(runfile.read_text("utf-8"))
        else:
            if runfile.exists():
                raise SystemExit("Existing baseline exists; use --resume or a new artifact directory")
            files = sorted((root / "scripts").glob("[0-9][0-9].txt"))
            if len(files) != 6:
                raise SystemExit("Require all six independently authored scripts before submitting")
            if args.only:
                requested = {f"{int(n):02d}" for n in args.only.split(",")}
                if not requested <= {file.stem for file in files}:
                    raise SystemExit("Unknown frozen script number")
                files = [file for file in files if file.stem in requested]
            names = ["雷军宣布造车", "刘翔雅典夺冠", "快递末端劳动", "故宫开放", "C919商业首航", "唐宫夜宴"]
            rows = []
            for file in files:
                i = int(file.stem)-1
                script = file.read_text("utf-8").strip()
                rows.append({"title": f"{args.title_prefix} {file.stem} · {names[i]}", "script": script,
                             "aspect": "9:16" if i == 2 else "16:9", "media_policy": "video_preferred",
                             "min_video_ratio": .8, "min_video_height": 720})
            submitted = time.time()
            response = client.post("/api/batch", json={"projects": rows, "resolution": "720p"})
            response.raise_for_status()
            run = {"submitted_at": submitted, "submitted_utc": datetime.now(UTC).isoformat(),
                   "protocol": "Frozen independent scripts; default 80% real video / 720p; no manual approvals; existing general library allowed; no script-tailored prefetch",
                   "projects": [item | {"script_file": str(file.relative_to(root)),
                       "script_sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "first_running_at": None,
                       "terminal_at": None} for item, file in zip(response.json()["projects"], files, strict=True)]}
            save(runfile, run)
        last = {}
        while True:
            pending = False
            for item in run["projects"]:
                if item["terminal_at"] is not None:
                    continue
                r = client.get("/api/jobs/"+item["job_id"])
                r.raise_for_status()
                job = r.json()
                now = time.time()
                if job["status"] == "running" and item["first_running_at"] is None:
                    item["first_running_at"] = now
                signature = (job["status"], job["message"])
                if last.get(item["job_id"]) != signature:
                    entry = {"at": now, "job_id": item["job_id"], "project_id": item["project_id"],
                             **{k: job.get(k) for k in ("status", "progress", "message", "error")}}
                    with (root / "baseline-events.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps(entry, ensure_ascii=False)+"\n")
                    print(item["title"], job["status"], round(job["progress"], 1), job["message"], flush=True)
                    last[item["job_id"]] = signature
                if job["status"] in ("queued", "running"):
                    pending = True
                else:
                    # The observer can be suspended after the worker releases
                    # its power assertion. Do not charge that post-completion
                    # sleep to production; retain both clocks for comparison.
                    finished = float(job.get("updated") or now)
                    if not run["submitted_at"] <= finished <= now:
                        finished = now
                    item["terminal_at"] = finished
                    item["terminal_observed_at"] = now
                    item["terminal_observation_delay_seconds"] = round(now-finished, 3)
                    item["status"] = job["status"]
                    item["error"] = job.get("error", "")
                    item["wall_seconds_including_queue"] = round(finished-run["submitted_at"], 3)
                    item["observed_wall_seconds_including_queue"] = round(now-run["submitted_at"], 3)
                    item["active_seconds_observed"] = round(now-(item["first_running_at"] or job["created"]), 3)
                    project = client.get("/api/projects/"+item["project_id"]).json()
                    item["workflow_info"] = project.get("workflow_info")
                    item["execution_info"] = project.get("execution_info")
                    item["composition"] = project.get("composition")
                    item["exports"] = project.get("exports")
                    save(root / (item["project_id"]+"-baseline.json"), project)
                    save(root / (item["job_id"]+"-job.json"), job)
                save(runfile, run)
            if not pending:
                break
            time.sleep(5)
    print("Baseline finished; inspect recorded gaps and exported media before acceptance.", flush=True)


if __name__ == "__main__":
    main()
