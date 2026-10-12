"""Unfinished frame work survives reviewer updates; completed decisions do not."""

import pytest

from jinggao import pending_windows as pending, pooled_match, store, window_attempts
from test_project_pool import environment as _environment, video as _video

environment = _environment
video = _video


@pytest.fixture
def receipt(environment, monkeypatch):
    p, asset, target, *_ = environment
    monkeypatch.setattr(window_attempts, "policy", lambda: {"reviewer": "before"})
    p = store.mutate(
        p["id"], lambda p: p["assets"][0].update(source_url="https://v.qq.com/x/page/local.html")
    )
    shot, asset = p["shots"][0], p["assets"][0]
    frames, preview = pooled_match.sample_frames(p["id"], shot, asset, lambda: None)
    saved = pending.capture(p, {"shot": shot, "asset": asset}, frames, preview, lambda: None)
    monkeypatch.setattr(window_attempts, "policy", lambda: {"reviewer": "after"})
    return p, shot, asset, target, frames, saved


def test_pending_pixels_resume_under_current_reviewer_without_any_approval(receipt):
    p, shot, asset, target, frames, saved = receipt
    rows, stats = pending.recover(p, [shot], lambda: None)
    assert stats["candidates"] == 1 and rows[0]["pending_receipt"].key == saved.key
    actual, _ = rows[0]["pending_receipt"].replay(p, shot, asset, lambda: None)
    assert actual == frames
    assert not rows[0].get("verification") and not rows[0]["shot"]["approved"]
    fresh = pending.capture(p, rows[0], frames, saved.body["preview"], lambda: None)
    assert fresh.key != saved.key and fresh.body["binding"]["context"]["review_policy"] == {
        "reviewer": "after"
    }


@pytest.mark.parametrize("state", ["reviewed", "terminal"])
def test_completed_decision_cannot_cross_reviewer_change(receipt, state):
    p, shot, asset, target, frames, saved = receipt
    saved.finish(state, "old completed decision")
    assert not pending.recover(p, [shot], lambda: None)[0]


@pytest.mark.parametrize("change", ["script", "source", "media", "frames", "duration"])
def test_changed_requirements_or_bytes_still_reject_pending_work(receipt, change):
    p, shot, asset, target, frames, saved = receipt
    if change == "script":
        p["script"] += "新要求"
    elif change == "source":
        asset["source_url"] = "https://v.qq.com/x/page/other.html"
    elif change == "duration":
        shot["duration"] += 1
    elif change == "media":
        target.write_bytes(target.read_bytes() + b"changed")
    else:
        frames[0].write_bytes(b"changed pixels")
    assert not pending.recover(p, [shot], lambda: None)[0]
