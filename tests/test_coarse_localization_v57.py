"""Small official renditions locate scenes; only HD production assets may export."""
from contextlib import contextmanager
import hashlib
from pathlib import Path

from PIL import Image
import pytest

from jinggao import curation, paths, pooled_match, segment_discovery as discovery, video_sources


def candidate():
    return {"id": "source", "source_url": "https://pixabay.com/videos/id-47978/",
            "title": "Pruning", "metadata_checked": True, "download_status": "available", "duration": 20,
            "media_streams": [
                {"url": "https://cdn.pixabay.com/small.mp4", "format": "file", "width": 640, "height": 360},
                {"url": "https://cdn.pixabay.com/hd.mp4", "format": "file", "width": 1280, "height": 720}]}


@pytest.fixture
def decode(monkeypatch):
    calls = []
    @contextmanager
    def proxy(url, budget, source):
        calls.append(url)
        yield url, {"error": ""}
    def extract(url, at, target, budget, minimum, **kwargs):
        Image.new("RGB", (minimum * 16 // 9, minimum)).save(target)
        return at, minimum * 16 // 9, minimum
    def sequence(url, points, folder, budget, minimum):
        rows = []
        for i, at in enumerate(points):
            target = folder / f"frame-{i}.jpg"
            actual, width, height = extract(url, at, target, budget, minimum)
            rows.append({"id": f"f{i}", "at": actual, "path": str(target), "width": width, "height": height})
        return rows
    monkeypatch.setattr(discovery, "_proxy", proxy)
    monkeypatch.setattr(discovery, "_extract", extract)
    monkeypatch.setattr(discovery, "_extract_sequence", sequence)
    return calls


def test_small_localization_preserves_hd_download_and_export_requirement(tmp_path, decode):
    source = candidate()
    result = discovery.sample_candidate(source, tmp_path, coarse=True)
    assert result["status"] == "sampled" and result["decoder_strategy"] == "small_rendition_single_pass"
    assert result["sampling_only"] and result["localization_min_height"] == 360
    assert result["output_min_height"] == 720 and decode == [source["media_streams"][0]["url"]]
    assert all(min(f["width"], f["height"]) == 360 for f in result["frames"])
    assert video_sources._choose_stream(source, False)["height"] == 720
    p = {"shots": [{"id": "shot", "duration": 8, "selected_asset": "a"}],
         "assets": [{"id": "a", "kind": "video", "duration": 10, "width": 640, "height": 360}],
         "min_video_height": 720}
    assert curation.composition(p)["low_resolution_shots"] == [1]
    assert not any(k in result for k in ("approved", "quality_verified", "usable_windows"))


def test_default_is_hd_and_complete_old_hd_cache_is_reused(tmp_path, decode):
    first = discovery.sample_candidate(candidate(), tmp_path)
    second = discovery.sample_candidate(candidate(), tmp_path, coarse=True)
    assert first["status"] == "sampled" and second["cache_hit"]
    assert decode == [candidate()["media_streams"][1]["url"]]
    assert all(f["height"] == 720 for f in second["frames"])


def test_coarse_cache_cannot_satisfy_hd_sampling(tmp_path, decode):
    first = discovery.sample_candidate(candidate(), tmp_path, coarse=True)
    second = discovery.sample_candidate(candidate(), tmp_path)
    assert not second["cache_hit"] and first["frames"][0]["path"] != second["frames"][0]["path"]
    assert second["frames"][0]["height"] == 720


@pytest.mark.parametrize("mode", ["low_only", "protected_hd", "no_small", "unknown_hd"])
def test_coarse_requires_unprotected_known_hd_rendition(tmp_path, decode, mode):
    source = candidate()
    if mode == "low_only":
        source["media_streams"].pop()
    elif mode == "protected_hd":
        source["media_streams"][1]["has_drm"] = True
    elif mode == "no_small":
        source["media_streams"].pop(0)
    else:
        source["media_streams"][1].update(width=0, height=0)
    if mode in ("low_only", "protected_hd"):
        with pytest.raises(ValueError):
            discovery.sample_candidate(source, tmp_path, coarse=True)
        assert not decode
    else:
        result = discovery.sample_candidate(source, tmp_path, coarse=True)
        assert "localization_min_height" not in result and result["frames"][0]["height"] == 720


def test_rank_and_shared_facts_preserve_coarse_provenance(tmp_path, decode):
    result = discovery.sample_candidate(candidate(), tmp_path, coarse=True)
    facts = {hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest(): {} for f in result["frames"]}
    shared, failures = pooled_match.shared_observed_samples([result], {"observations": facts}, lambda: None)
    assert not failures and shared[0]["localization_min_height"] == 360
    _, snapshot, _ = discovery._rank_payload([], shared, lambda: None)
    assert snapshot["candidates"][0]["sampling_only"]
    assert snapshot["candidates"][0]["output_min_height"] == 720
    assert all(f["height"] == 360 for f in snapshot["frames"])
    result.pop("localization_min_height")
    with pytest.raises(ValueError, match="720p"):
        discovery._rank_payload([], [result], lambda: None)


@pytest.mark.parametrize("identity", [False, True])
def test_discovery_uses_coarse_only_for_ordinary_footage(tmp_path, monkeypatch, identity):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    requested = []
    def sample(source, *args, **kwargs):
        requested.append(kwargs["coarse"])
        return {"frames": [], "errors": [], "downloaded_bytes": 0}
    monkeypatch.setattr(discovery, "sample_candidate", sample)
    monkeypatch.setattr(discovery, "observe_samples", lambda *a: ({}, {"model_requests": 0, "failures": []}))
    shot = {"id": "one", "persons": ["姓名"] if identity else []}
    pooled_match.discover_windows({"shots": [shot]}, [([shot], [candidate()])], lambda: None, lambda *a: None)
    assert requested == [not identity] and not shot.get("approved")
