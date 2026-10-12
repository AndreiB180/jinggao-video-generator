"""Offline projection compatibility only; never call a model or approve video.

Merge paired historical blind facts/quality outputs from library review
diagnostics by their exact saved frame indices. This synthetic joint output
tests the interface, not whether one model request can produce correct facts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jinggao import costs, neutral_observer, paths, pixel_observer, pixel_packet, visual_quality  # noqa: E402


def replay(summary_path):
    body = json.loads(summary_path.read_text("utf-8"))
    project = paths.DATA / "projects" / body["project_id"]
    rows, prepared, wire_sizes = [], [], []
    for review in body["reviews"]:
        audit = review["verification"]["identity_audit"]
        hashes = audit["visual_quality"]["frame_sha256"]
        files = [project / row["path"] for row in review["preview"]["frames"]]
        if len(hashes) != len(files) or [hashlib.sha256(f.read_bytes()).hexdigest() for f in files] != hashes:
            raise ValueError("历史图片hash与观察不一致，未复放")
        prepared.append([neutral_observer._frame(file) for file in files])
        expected = {"history": len(files)}
        old_facts = {"id": "history", "scene_description": audit["scene_description"],
                     "frames": audit["frame_observations"]}
        old_quality = {"id": "history", "frames": audit["visual_quality"]["frames"]}
        facts = neutral_observer.parse_groups(json.dumps({"groups": [old_facts]}), expected)
        quality = visual_quality.parse_groups(json.dumps({"groups": [old_quality]}), expected)
        combined = {"id": "history", "scene_description": facts["history"]["scene_description"],
                    "frames": [fact | quality["history"]["frames"][i]
                               for i, fact in enumerate(facts["history"]["frames"])]}
        actual_facts, actual_quality = pixel_observer.parse_groups(
            json.dumps({"groups": [combined]}, ensure_ascii=False), expected)
        if actual_facts != facts or actual_quality != quality:
            raise AssertionError("联合投影丢失或修改历史字段")
        canonical = {"groups": [combined]}
        compact = pixel_packet.pack(canonical)
        compact_text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if pixel_packet.expand(compact_text) != canonical:
            raise AssertionError("数组协议改变逐帧事实或质量字段")
        wire_facts, wire_quality = pixel_observer.parse_wire(compact_text, expected)
        if wire_facts != facts or wire_quality != quality:
            raise AssertionError("数组投影改变既有观察")
        wire_sizes.append({"canonical_utf8_bytes": len(json.dumps(canonical, ensure_ascii=False,
                                                               separators=(",", ":")).encode()),
                           "compact_utf8_bytes": len(compact_text.encode())})
        assessed = visual_quality.assess(actual_quality["history"])
        if assessed != visual_quality.assess(quality["history"]) or assessed["content_approved"]:
            raise AssertionError("质量投影或非审批边界改变")
        rows.append({"shot_id": review["shot_id"], "frames": len(files), "frame_sha256": hashes,
                     "facts_projection_equal": True, "quality_projection_equal": True,
                     "compact_wire_projection_equal": True, "wire_sizes": wire_sizes[-1],
                     "assessment_equal": True, "quality_status": assessed["quality_status"],
                     "content_approved": False})
    # This is a pure payload calculation using public model/endpoint identifiers,
    # not the user's settings, key, cost ledger or a transport. Compare equal
    # two-group batches: the old requests send the identical pictures twice.
    cfg = {"vision_model": "qwen3.7-flash", "model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"}
    comparison = {"basis": "same two-group batches; conservative input upper bounds, not actual tokens or charges",
                  "legacy_requests": 0, "joint_requests": 0, "legacy_image_blocks": 0, "joint_image_blocks": 0,
                  "legacy_input_bound": 0, "joint_input_bound": 0}
    for start in range(0, len(prepared), pixel_observer.GROUPS_PER_REQUEST):
        batch = prepared[start:start + pixel_observer.GROUPS_PER_REQUEST]
        joint = pixel_observer._payload({f"g{i}": frames for i, frames in enumerate(batch)}, cfg)
        facts_payload, quality_payload = deepcopy(joint), deepcopy(joint)
        facts_payload["messages"][0]["content"] = neutral_observer.PROTOCOL
        quality_payload["messages"][0]["content"] = visual_quality.PROTOCOL
        # The old quality observer sends group/frame labels without a manifest.
        quality_payload["messages"][1]["content"] = quality_payload["messages"][1]["content"][1:]
        image_count = sum(len(frames) for frames in batch)
        comparison["legacy_requests"] += 2
        comparison["joint_requests"] += 1
        comparison["legacy_image_blocks"] += 2 * image_count
        comparison["joint_image_blocks"] += image_count
        comparison["legacy_input_bound"] += costs._input_bound(cfg, facts_payload) + costs._input_bound(cfg, quality_payload)
        comparison["joint_input_bound"] += costs._input_bound(cfg, joint)
    canonical_bytes = sum(row["canonical_utf8_bytes"] for row in wire_sizes)
    compact_bytes = sum(row["compact_utf8_bytes"] for row in wire_sizes)
    return {"source": str(summary_path.resolve()), "source_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "groups": rows, "cold_payload_comparison": comparison,
            "wire_comparison": {"basis": "same historical field values; UTF-8 bytes, not tokens or billed savings",
                                "canonical_utf8_bytes": canonical_bytes, "compact_utf8_bytes": compact_bytes,
                                "reduction_fraction": 1 - compact_bytes / canonical_bytes}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-summary", required=True, action="append", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {"mode": "offline_synthetic_joint_from_history", "model_requests": 0,
              "production_entrypoint_changed": False, "quality_validated": False,
              "limitation": "历史两套输出按帧拼接，只证明严格投影兼容，不证明联合调用的模型质量或实际费用。",
              "results": [replay(path) for path in args.review_summary]}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
