import json

import pytest

from scripts.verify_generation_target import previous_attempts, timing_result


def test_retry_keeps_failed_job_time_and_debugging_gap(tmp_path):
    evidence = tmp_path / "run.json"
    evidence.write_text(json.dumps({"project_id": "p", "script_sha256": "original",
        "job_id": "j1", "status": "failed", "started_at": 1000, "elapsed_seconds": 53.647,
        "paid_delta_cny": .012712}))
    prior = previous_attempts(evidence, "p", "original")
    timing = timing_result({"started_at": 1600, "prior_attempts": prior,
                           "max_seconds": 600, "cancel_requested": False}, 2100)
    assert timing["elapsed_seconds"] == 500
    assert timing["cumulative_active_seconds"] == 553.647
    assert timing["end_to_end_wall_seconds"] == 1100
    assert timing["active_time_pass"] is True
    assert timing["time_pass"] is False
    assert prior[0]["paid_delta_cny"] == .012712


@pytest.mark.parametrize("changes", [{"project_id": "different"}, {"script_sha256": "rewritten"},
                                     {"status": "running"}, {"status": "completed"}])
def test_retry_rejects_unrelated_or_live_evidence(tmp_path, changes):
    evidence = tmp_path / "run.json"
    evidence.write_text(json.dumps({"project_id": "p", "script_sha256": "original",
        "job_id": "j1", "status": "failed", "started_at": 1000, "elapsed_seconds": 10,
        "paid_delta_cny": .001, **changes}))
    with pytest.raises(ValueError):
        previous_attempts(evidence, "p", "original")


def test_prior_failures_can_exhaust_target_without_this_attempt_exceeding_it():
    timing = timing_result({"started_at": 100, "prior_attempts": [
        {"started_at": 0, "elapsed_seconds": 54}], "max_seconds": 600, "cancel_requested": False}, 650)
    assert timing["elapsed_seconds"] == 550
    assert timing["cumulative_active_seconds"] == 604
    assert timing["active_time_pass"] is False
    assert timing["time_pass"] is False


def test_cold_run_and_cancelled_run():
    run = {"started_at": 100, "max_seconds": 600, "cancel_requested": False}
    assert timing_result(run, 300) == {"elapsed_seconds": 200, "cumulative_active_seconds": 200,
        "end_to_end_wall_seconds": 200, "active_time_pass": True, "time_pass": True}
    assert timing_result(run | {"cancel_requested": True}, 300)["time_pass"] is False
