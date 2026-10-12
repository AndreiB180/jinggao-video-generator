"""Actual video-type accounting and frontend/backend export agreement."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from jinggao import curation


def project():
    types = ["live_action", "screen_capture", "animation", "mixed", "uncertain", None]
    shots, assets = [], []
    for i, representation in enumerate(types):
        shot = {"id": str(i), "duration": 10, "clip_start": 1, "selected_asset": str(i)}
        if representation:
            shot["verification"] = {"visual_quality": {"representation": representation, "quality_status": "not_detected"}}
        shots.append(shot)
        assets.append({"id": str(i), "kind": "video", "duration": 9, "width": 1280, "height": 720})
    shots.append({"id": "photo", "duration": 4, "selected_asset": "photo"})
    assets.append({"id": "photo", "kind": "image"})
    return {"media_policy": "video_preferred", "min_video_ratio": .8, "min_video_height": 720,
            "shots": shots, "assets": assets}


def test_video_types_keep_playable_frozen_and_legacy_durations_distinct():
    p = project()
    report = curation.composition(p)
    assert report["video_seconds"] == 24 and report["legacy_video_seconds"] == 8
    assert report["illustration_video_seconds"] == 16 and report["illustration_video_shots"] == [3, 4]
    assert report["unclassified_video_seconds"] == 8 and report["unclassified_video_shots"] == [5]
    assert report["image_seconds"] == 4 and report["frozen_seconds"] == 12
    assert report["total_seconds"] == 64 and report["short_video_shots"] == [1, 2, 3, 4, 5, 6]
    assert report["video_ratio"] == .375 and not report["meets_target"]
    assert sum(report[k] for k in ("video_seconds", "illustration_video_seconds", "unclassified_video_seconds",
                                  "image_seconds", "missing_seconds", "frozen_seconds")) == report["total_seconds"]
    # Overlay uncertainty does not redefine an independently observed type.
    p["shots"][0]["verification"]["visual_quality"]["quality_status"] = "uncertain"
    assert curation.composition(p)["video_seconds"] == 24
    # A new but incomplete observation is not a legacy live-action assumption.
    p["shots"][0]["verification"]["visual_quality"] = {}
    assert curation.composition(p)["unclassified_video_seconds"] == 16


def test_export_target_excludes_illustration_but_mixed_stays_unrestricted():
    p = project()
    # This fixture tests composition policy for explicitly selected media;
    # automatic frame/caption approval is covered by the curation tests.
    for shot in p["shots"]:
        shot["locked"] = True
    with pytest.raises(ValueError, match="无法确认类型"):
        curation.validate_export(p)
    # Known illustration can be included in a mixed project; an unclassified
    # blank/transition frame still needs review in every policy.
    p["shots"][4]["verification"]["visual_quality"]["representation"] = "animation"
    with pytest.raises(ValueError, match="动画/混合示意与类型待核"):
        curation.validate_export(p)
    p["media_policy"] = "mixed"
    assert curation.validate_export(p, image_motion=False)["meets_target"]
    p["media_policy"] = "video_only"
    with pytest.raises(ValueError, match="至少 100%"):
        curation.validate_export(p)
    legacy = {"media_policy": "video_only", "assets": p["assets"][-2:-1], "shots": [deepcopy(p["shots"][-2])]}
    legacy["shots"][0]["duration"] = 8
    assert curation.validate_export(legacy)["video_seconds"] == 8
    assert curation.validate_export(legacy)["legacy_video_seconds"] == 8


def test_frontend_composition_matches_backend_for_types_audio_and_missing_assets():
    node = shutil.which("node")
    root = Path(__file__).resolve().parents[1]
    frontend = root / "frontend"
    if not node or not (frontend / "node_modules/typescript/lib/typescript.js").exists():
        pytest.skip("Frontend runtime/dependencies are not installed")
    samples = [project()]
    recorded = deepcopy(samples[0])
    recorded["shots"][0]["verification"]["visual_quality"]["representation"] = "recorded_mix"
    assert curation.composition(recorded)["video_seconds"] == 24
    samples.append(recorded)
    full = deepcopy(samples[0])
    full["full_audio"] = {"duration": 64}
    full["media_policy"] = "mixed"
    full["shots"][0]["audio"] = {"duration": 5}
    full["shots"][1]["selected_asset"] = "missing"
    full["shots"][2].update(persons=[{"name": "某人"}], source_review_note="已阅读实际发布者原文图注", source_review_url="https://example.org/source")
    samples.append(full)
    per_shot = deepcopy(full)
    per_shot.pop("full_audio")
    samples.append(per_shot)
    script = """
import fs from 'node:fs';
import ts from 'typescript';
const source=fs.readFileSync('src/composition.ts','utf8');
const {outputText}=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}});
const {composition}=await import('data:text/javascript;base64,'+Buffer.from(outputText).toString('base64'));
const inputs=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify(inputs.map(composition)));
"""
    completed = subprocess.run([node, "--input-type=module", "-e", script], cwd=frontend,
                               input=json.dumps(samples), text=True, capture_output=True, check=True, timeout=15)
    assert json.loads(completed.stdout) == [curation.composition(p) for p in samples]
