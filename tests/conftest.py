import pytest

from jinggao import visual_quality

# These import jinggao.review_instructions / review_view / local_scene_candidates, which never
# existed in this repository (no git history, absent from docs/source-baseline.json and the
# v60 egg-info SOURCES). They were collected from the pre-v60 mixed workspace. Delete or port.
collect_ignore = ["test_caption_review.py", "test_local_scene_candidates.py", "test_review_instructions.py",
                  "test_review_schema.py", "test_review_view.py", "test_typed_review.py"]


@pytest.fixture(autouse=True)
def private_output_folder(tmp_path_factory, monkeypatch):
    """Finished videos are copied to the user's Videos folder; in tests they go to a temporary one."""
    from jinggao import config
    monkeypatch.setitem(config.DEFAULTS, "export_dir", str(tmp_path_factory.mktemp("output")))


@pytest.fixture(autouse=True)
def no_graphics_card(monkeypatch):
    """Tests never run the real NVIDIA check: the card is absent unless a test says otherwise."""
    from jinggao import gpu
    monkeypatch.setattr(gpu, "_cache", {"mtime": None, "value": {"nvenc": False, "full": False, "nvidia": False,
                                                                  "sessions": 0, "reason": "测试环境"}})
    monkeypatch.setattr(gpu, "_disabled", {})
    monkeypatch.setattr(gpu, "_failures", {"full": 0, "nvenc": 0})
    monkeypatch.setattr(gpu, "_counts", {})


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



@pytest.fixture
def separate_pixel_observer(monkeypatch):
    """v60 defaults to pixel_observer_mode=joint (one vision request for facts+quality).

    Text-judge unit tests mock the two separate observers instead; opt in explicitly.
    """
    from jinggao import config
    read = config.read
    monkeypatch.setattr(config, "read", lambda *a, **k: read(*a, **k) | {"pixel_observer_mode": "separate"})


_RETIRED = {}
for _line in (__import__("pathlib").Path(__file__).with_name("retired_tests.txt")).read_text("utf-8").splitlines():
    if _line.strip() and not _line.startswith("#"):
        _node, _reason = _line.split("\t", 1)
        _RETIRED[_node] = _reason


def pytest_collection_modifyitems(config, items):
    """Tests of removed classic-pipeline behaviour stay visible as skips with the reason (retired_tests.txt)."""
    for item in items:
        reason = _RETIRED.get(item.nodeid)
        if reason:
            item.add_marker(pytest.mark.skip(reason="已退役的旧协议测试：" + reason))
