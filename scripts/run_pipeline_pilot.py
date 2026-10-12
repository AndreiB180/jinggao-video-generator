"""Run one new-content production pilot with the normal API and durable budget.

No prefetched footage, imported narration, manual approval or budget reset.
The temporary pipeline settings are restored after the job reaches a terminal
state. All project data, failed requests and unconfirmed charges remain saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import httpx


MODES = {"source_plan_mode": "unified", "pixel_observer_mode": "joint"}
ACTIVE = {"running", "queued"}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(path)


def request(client, method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:8766")
    parser.add_argument("--title", default="新管线诊断 · 绿茶制作")
    parser.add_argument("--resume", action="store_true", help="Continue the same saved project; never reset its fees")
    args = parser.parse_args()
    root = args.directory.resolve()
    runfile = root / "pilot-run.json"
    if runfile.exists() and not args.resume:
        raise SystemExit("This pilot already ran; retain its project and costs instead of creating a clone")
    previous = json.loads(runfile.read_text()) if args.resume else None
    if previous and previous.get("status") not in {"failed", "cancelled", "interrupted", "needs_review"}:
        raise SystemExit("Only a terminal unfinished pilot can resume")
    script = (root / "script.txt").read_text("utf-8").strip()
    if not script:
        raise SystemExit("Missing new script")
    normalized = "".join(script.split())
    run = {"script_sha256": hashlib.sha256(script.encode()).hexdigest(),
           "normalized_script_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
           "character_count": len(normalized), "modes": MODES, "approval": "pending_independent_review",
           "protocol": "One new script; normal produce API; existing general library allowed; no tailored prefetch, budget reset or manual approval. First attempt starts without audio; resumed attempts reuse saved work and retain all earlier fees and elapsed time."}
    if previous:
        if previous["normalized_script_sha256"] != run["normalized_script_sha256"]:
            raise SystemExit("The script changed; this is not an unchanged-project resume")
        attempt = previous.get("attempt_number", 1)
        archive = root / f"attempt-{attempt}-run.json"
        if archive.exists():
            raise SystemExit("Prior attempt archive already exists; inspect recovery before another resume")
        save(archive, previous)
        run.update(attempt_number=attempt + 1, previous_attempt=str(archive), resumed=True,
                   first_submitted_at=previous.get("first_submitted_at", previous["submitted_at"]),
                   cold_start_passed=False)
    with httpx.Client(base_url=args.base, trust_env=False, timeout=30) as client:
        if any(j["status"] in ACTIVE for j in request(client, "GET", "/api/queue")):
            raise SystemExit("Another job is active; pipeline settings left unchanged")
        for project in request(client, "GET", "/api/projects"):
            existing = request(client, "GET", "/api/projects/" + project["id"])
            if "".join(existing["script"].split()) == normalized and not (previous and project["id"] == previous["project_id"]):
                raise SystemExit("Script already belongs to " + project["id"] + "; do not duplicate its budget")
        settings = request(client, "GET", "/api/settings")
        if any(not 0 < float(settings[k]) <= .10 for k in ("model_project_budget_cny", "model_job_budget_cny")):
            raise SystemExit("The pilot requires existing positive project and job limits of at most 0.10 CNY")
        if not all(key in settings for key in MODES):
            raise SystemExit("Running server lacks the opt-in pipeline; no settings or projects changed")
        original = {key: settings[key] for key in MODES}
        run["original_modes"] = original
        run["tts_provider_unchanged"] = settings["tts_provider"]
        job_id = None
        terminal = False
        try:
            request(client, "PATCH", "/api/settings", json=MODES)
            started = time.time()
            run["submitted_at"] = started
            save(runfile, run)
            if previous:
                current = request(client, "GET", "/api/projects/" + previous["project_id"])
                if "".join(current["script"].split()) != normalized:
                    raise ValueError("Saved project source differs from the frozen pilot")
                run["saved_audio_reused"] = bool(current.get("full_audio"))
                run["saved_shot_count"] = len(current.get("shots", []))
                resumed_job = request(client, "POST", "/api/projects/" + previous["project_id"] + "/jobs",
                                      json={"kind": "produce", "mode": "model", "resolution": "720p"})
                result = {"project_id": previous["project_id"], "job_id": resumed_job["id"], "title": current["title"]}
            else:
                result = request(client, "POST", "/api/batch", json={"projects": [{
                    "title": args.title, "script": script, "aspect": "16:9", "media_policy": "video_preferred",
                    "min_video_ratio": .8, "min_video_height": 720}], "resolution": "720p"})["projects"][0]
            run.update(result)
            job_id = result["job_id"]
            save(runfile, run)
            last = None
            while True:
                job = request(client, "GET", "/api/jobs/" + job_id)
                now = time.time()
                signature = (job["status"], job["message"])
                if signature != last:
                    event = {"at": now, **{key: job.get(key) for key in ("id", "status", "progress", "message", "error")}}
                    with (root / "pilot-events.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                    last = signature
                if job["status"] not in ACTIVE:
                    terminal = True
                    break
                if now - started >= 600 and not run.get("deadline_cancel_requested_at"):
                    run["deadline_cancel_requested_at"] = now
                    request(client, "POST", "/api/jobs/" + job_id + "/cancel")
                    save(runfile, run)
                time.sleep(3)
            project = request(client, "GET", "/api/projects/" + result["project_id"])
            save(root / "project-result.json", project)
            save(root / "job-result.json", job)
            run.update(status=job["status"], error=job.get("error", ""), finished_at=job["updated"],
                       wall_seconds_including_queue=round(job["updated"] - started, 3),
                       exports=project["exports"], model_costs=project["model_costs"],
                       composition=project["composition"], caption_alignment=project["caption_alignment"],
                       workflow_info=project.get("workflow_info"), execution_info=project.get("execution_info"))
            run["total_wall_seconds_since_first_submission"] = round(
                job["updated"] - run.get("first_submitted_at", started), 3)
            save(runfile, run)
        finally:
            # Do not change the observation strategy underneath an active job.
            # An interrupted harness asks the ordinary worker to stop and leaves
            # a visible recovery record if it has not yet acknowledged cancellation.
            if job_id and not terminal:
                job = request(client, "GET", "/api/jobs/" + job_id)
                terminal = job["status"] not in ACTIVE
                if not terminal:
                    request(client, "POST", "/api/jobs/" + job_id + "/cancel")
                    run["harness_interrupted_job_cancellation_requested"] = True
            current = request(client, "GET", "/api/settings")
            if (not job_id or terminal) and all(current[k] == v for k, v in MODES.items()):
                request(client, "PATCH", "/api/settings", json=original)
                run["settings_restored"] = True
            else:
                run["settings_restored"] = False
            save(runfile, run)


if __name__ == "__main__":
    main()
