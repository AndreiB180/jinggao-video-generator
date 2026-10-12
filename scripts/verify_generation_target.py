"""Run one existing project under its real cumulative budget, preserving failures.

This measures the actual application, with the installed general material index.
It neither fabricates approvals nor treats pre-existing paid work as a cold run.
An interrupted observer resumes the saved job handle; it never resubmits it.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time

import httpx


def save(file, value):
    temporary = file.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(file)


def previous_attempts(file, project_id, script_sha256):
    """Keep failed work in the same experiment instead of restarting its clock."""
    if file is None:
        return []
    previous = json.loads(file.read_text("utf-8"))
    if previous["project_id"] != project_id or previous["script_sha256"] != script_sha256:
        raise ValueError("Retry evidence must belong to the same project and unchanged script")
    if previous.get("status") not in ("failed", "cancelled", "needs_review"):
        raise ValueError("Retry evidence must describe a finished unsuccessful attempt")
    attempts = previous.get("prior_attempts", []) + [{
        "evidence": str(file.resolve()), "job_id": previous["job_id"],
        "started_at": previous["started_at"], "elapsed_seconds": previous["elapsed_seconds"],
        "status": previous["status"], "paid_delta_cny": previous["paid_delta_cny"],
        "external_processing_seconds": sum(row["elapsed_seconds"]
                                           for row in previous.get("external_processing", [])),
    }]
    if len({row["job_id"] for row in attempts}) != len(attempts):
        raise ValueError("Duplicate job in retry evidence")
    return attempts


def timing_result(run, ended_at):
    elapsed = max(0, ended_at-run["started_at"])
    previous = run.get("prior_attempts", [])
    active = elapsed + sum(row["elapsed_seconds"] + row.get("external_processing_seconds", 0)
                           for row in previous)
    first_started = min([run["started_at"], *[row["started_at"] for row in previous]])
    wall = max(0, ended_at-first_started)
    return {"elapsed_seconds": round(elapsed, 3), "cumulative_active_seconds": round(active, 3),
            "end_to_end_wall_seconds": round(wall, 3),
            "active_time_pass": active <= run["max_seconds"] and not run["cancel_requested"],
            "time_pass": wall <= min(600, run["max_seconds"]) and not run["cancel_requested"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:8766")
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--retry-of", type=Path, help="Previous failed run.json; preserve its cost and time")
    args = parser.parse_args()
    if not 0 < args.seconds <= 600:
        parser.error("The generation target is fixed at 600 seconds; only a shorter stop limit is allowed")
    args.directory.mkdir(parents=True, exist_ok=True)
    record = args.directory / "run.json"
    with httpx.Client(base_url=args.base, trust_env=False, timeout=30) as client:
        def get(url):
            response = client.get(url)
            response.raise_for_status()
            return response.json()
        if record.exists():
            run = json.loads(record.read_text("utf-8"))
            if run["project_id"] != args.project_id:
                raise SystemExit("Existing evidence belongs to another project; refusing to replace it")
        else:
            project = get("/api/projects/"+args.project_id)
            jobs = get("/api/projects/"+args.project_id+"/jobs")
            if any(j["status"] in ("running", "queued") for j in jobs):
                raise SystemExit("Project already has a live job; inspect that handle first")
            save(args.directory / "before.json", project)
            before = project["model_costs"]["project"]
            if before["total_cny"] >= .1:
                raise SystemExit("Existing project has consumed its one-mao budget; no reset or duplicate project")
            if before["requests"] and args.retry_of is None:
                raise SystemExit("Project has paid history; supply --retry-of instead of restarting its clock")
            script_sha256 = hashlib.sha256(project["script"].encode()).hexdigest()
            attempts = previous_attempts(args.retry_of, args.project_id, script_sha256)
            for previous in attempts:
                old_job = get("/api/jobs/"+previous["job_id"])
                if old_job["project_id"] != args.project_id or old_job["status"] != previous["status"]:
                    raise SystemExit("Saved retry evidence disagrees with the actual job history")
            remaining = args.seconds-sum(row["elapsed_seconds"] + row.get("external_processing_seconds", 0)
                                         for row in attempts)
            if remaining <= 0:
                raise SystemExit("Earlier attempts have exhausted the active generation time budget")
            started = time.time()
            response = client.post("/api/projects/"+args.project_id+"/jobs", json={"kind": "produce", "resolution": "720p"})
            response.raise_for_status()
            job = response.json()
            run = {"project_id": args.project_id, "job_id": job["id"], "started_at": started,
                   "started_utc": datetime.now(UTC).isoformat(), "max_seconds": args.seconds, "max_total_cny": .1,
                   "script_sha256": script_sha256, "prior_attempts": attempts,
                   "attempt_time_budget_seconds": remaining,
                   "before_cost": before, "before_shots": len(project["shots"]),
                   "before_assets": len(project["assets"]), "before_full_audio": bool(project.get("full_audio")),
                   "before_exports": [e["id"] for e in project.get("exports", [])],
                   "library_scope": "Existing general library and caches retained; no project-specific prefilling",
                   "cancel_requested": False}
            save(record, run)
        last = None
        while True:
            job = get("/api/jobs/"+run["job_id"])
            current = {k: job.get(k) for k in ("status", "progress", "message", "error")}
            if current != last:
                event = {"at": time.time(), "elapsed_seconds": round(time.time()-run["started_at"], 3), **current}
                with (args.directory / "events.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(event, ensure_ascii=False)+"\n")
                print(json.dumps(event, ensure_ascii=False), flush=True)
                last = current
            if job["status"] not in ("queued", "running"):
                break
            if (time.time()-run["started_at"] > run.get("attempt_time_budget_seconds", run["max_seconds"])
                    and not run["cancel_requested"]):
                response = client.post("/api/jobs/"+run["job_id"]+"/cancel")
                response.raise_for_status()
                run["cancel_requested"] = True
                save(record, run)
            time.sleep(3)
        project = get("/api/projects/"+args.project_id)
        save(args.directory / "after.json", project)
        save(args.directory / "job.json", job)
        total = project["model_costs"]["project"]
        run["cancel_requested"] = run["cancel_requested"] or bool(job.get("cancel"))
        timing = timing_result(run, float(job.get("updated", time.time())))
        exports = [e for e in project.get("exports", []) if e["id"] not in run["before_exports"]]
        own_export = next((e for e in exports if e["id"] == run["job_id"]), None)
        delivery = ({"duration_seconds": own_export["duration"],
                     "at_least_four_minutes": own_export["duration"] >= 240,
                     "at_least_720p": min(own_export["width"], own_export["height"]) >= 720,
                     "with_audio": not own_export.get("silent_preview", True),
                     "caption_alignment": own_export.get("caption_alignment"),
                     "composition": own_export.get("composition")} if own_export else None)
        run.update(status=job["status"], error=job.get("error"), **timing,
                   total_cost=total, new_exports=exports, composition=project.get("composition"),
                   delivery_checks=delivery,
                   workflow_info=project.get("workflow_info"),
                   paid_delta_cny=round(total["total_cny"]-run["before_cost"]["total_cny"], 6),
                   cost_pass=bool(total.get("cost_complete")) and total["total_cny"] <= .1,
                   full_video_exported=bool(own_export) and job["status"] == "completed",
                   independent_quality_review="pending; export and unit tests alone do not prove editorial quality")
        save(record, run)
        print(json.dumps({k: run[k] for k in ("status", "elapsed_seconds", "paid_delta_cny", "cost_pass",
                                             "time_pass", "full_video_exported")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
