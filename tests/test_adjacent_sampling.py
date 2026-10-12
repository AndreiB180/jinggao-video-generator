"""Extra probes consume existing source slots and cannot inherit approval."""
from collections import defaultdict
from copy import deepcopy

import pytest

from jinggao import pooled_match, segment_discovery


def test_neighbour_probes_fit_free_space_and_do_not_reserve_unseen_ranges():
    source = {"id": "source", "source_url": "https://v.qq.com/x/page/demo.html", "duration": 143.7}
    verified = [(7.4, 15.64), (46.76, 56.76)]
    used = defaultdict(list, {source["source_url"]: deepcopy(verified)})
    before = deepcopy(used)
    points = pooled_match.adjacent_sample_points(source, verified, used, [11.16, 7.08, 10.0])
    assert 2 <= len(points) <= 4 and len(set(points)) == len(points)
    assert used == before
    assert all(0 < at < source["duration"] for at in points)
    # The actual neighbour still has to be sampled and judged. A synthetic
    # proposal here exercises only non-overlap, never editorial acceptance.
    for at in points:
        proposal = {"sample_at": at, "clip_start": at-7.08/2}
        assert pooled_match.reserve_sampled(source, proposal, 7.08, used) is not None


def test_no_verified_window_or_no_free_neighbour_means_no_extra_probe():
    source = {"id": "short", "source_url": "https://v.qq.com/x/page/short.html", "duration": 12}
    used = defaultdict(list, {source["source_url"]: [(0, 12)]})
    assert pooled_match.adjacent_sample_points(source, [], used, [4]) == []
    assert pooled_match.adjacent_sample_points(source, [(2, 10)], used, [4]) == []
    assert pooled_match.adjacent_sample_points(source, [(2, 10)], {}, [30]) == []


def test_shorter_windows_can_fill_neighbours_too_small_for_longest_shot():
    source = {"id": "short", "duration": 20}
    points = pooled_match.adjacent_sample_points(source, [(8, 12)], {"short": [(8, 12)]}, [12, 4])
    assert len(points) == 2
    assert points[0] > 12 and points[1] < 8


def test_discovery_passes_explicit_points_through_same_time_and_byte_budgets(monkeypatch):
    received = []

    def sample(candidate, root, check, **kwargs):
        received.append(kwargs)
        return {"candidate_id": candidate["id"], "frames": [], "downloaded_bytes": 0}

    monkeypatch.setattr(segment_discovery, "sample_candidate", sample)
    monkeypatch.setattr(segment_discovery, "rank_windows_batch", lambda *a, **kw: {"proposals": {}, "model_info": {}})
    candidate = {"id": "good-source", "sample_points": [17, 41]}
    proposed, _, stats = pooled_match.discover_windows({}, [([{"id": "needs"}], [candidate])],
        lambda: None, lambda *a: None)
    assert not proposed and stats["sampled_frames"] == 0
    assert received == [{"count": 4, "max_seconds": 18, "max_bytes": 16*1024*1024,
                         "min_height": 720, "sample_points": [17, 41]}]


@pytest.mark.parametrize("points", [[float("nan")], [True], [100], [-1], [1, 1], [], list(range(7))])
def test_invalid_explicit_probes_fail_before_network(monkeypatch, tmp_path, points):
    candidate = {"id": "test", "source_url": "https://v.qq.com/x/page/example.html", "duration": 100,
                 "metadata_checked": True, "download_status": "available",
                 "media_streams": [{"url": "https://video.qq.com/valid.m3u8", "format": "hls",
                                    "width": 1280, "height": 720}]}
    monkeypatch.setattr(segment_discovery, "_read", lambda *a, **kw: pytest.fail("must not contact media"))
    with pytest.raises(ValueError, match="补采"):
        segment_discovery.sample_candidate(candidate, tmp_path, sample_points=points)
