import pytest

from jinggao import visual_quality


@pytest.fixture
def isolated_pixel_quality(monkeypatch):
    """Text-judgment unit tests isolate the separately tested pixel observer.

    Opt in explicitly; production integration/quality tests use their own
    controlled pixel responses. This fixture never runs automatically.
    """
    def observe(records, check):
        check()
        return {r["shot"]["id"]: {"representation": "live_action", "quality_status": "not_detected",
            "quality_reason": "isolated pixel-quality stub", "frames": [], "frame_sha256": []}
                for r in records}, {"model_requests": 0}
    monkeypatch.setattr(visual_quality, "observe", observe)

