"""Reject actual blank samples before paid review; retain visible detail."""
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from jinggao import frame_contract, media, paths, planner, pooled_match


def decoded_frames(monkeypatch, tmp_path, last):
    monkeypatch.setattr(paths, "DATA", tmp_path)

    def decode(args, **kwargs):
        kwargs["check"]()
        if any("metadata=print" in str(arg) for arg in args):
            # v60 scene-coverage pass (frame_sampling.sample): one scene score per 25fps frame.
            count = round(3 * 25)
            return "".join(f"frame:{i} pts:{i} pts_time:{i / 25}\nlavfi.scene_score=0.0\n" for i in range(count))
        template = args[-1]
        for index in range(1, 6):
            picture = Image.new("RGB", (320, 180), (30 * index, 50, 90))
            if index == 5:
                picture = last
            picture.save(Path(template.replace("%02d", f"{index:02d}")))
        return ""

    monkeypatch.setattr(media, "run", decode)
    shot = {"id": "test", "narration": "工作人员处理包裹。", "duration": 3, "clip_start": 0}
    asset = {"id": "asset", "title": "source", "kind": "video", "duration": 3,
             "path": "source.mp4", "width": 1280, "height": 720}
    project = {"id": "a" * 32, "shots": [shot]}
    source = paths.project_dir(project["id"]) / asset["path"]
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"stand-in video bytes; decoding is mocked")  # sampling hashes the source file
    return project, shot, asset


@pytest.mark.parametrize("color,kind", [("white", "white"), ("black", "black")])
def test_blank_tail_is_not_paid_for_even_when_clip_has_motion(monkeypatch, tmp_path, color, kind):
    project, shot, asset = decoded_frames(monkeypatch, tmp_path, Image.new("RGB", (320, 180), color))
    _, preview = pooled_match.sample_frames(project["id"], shot, asset, lambda: None)
    assert preview["motion_detected"]
    assert [(f["frame_index"], f["kind"]) for f in preview["blank_frames"]] == [(4, kind)]
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: pytest.fail("blank frame sent to model"))
    monkeypatch.setattr(planner, "audit_selected_frames", lambda *a, **kw: pytest.fail("blank frame sent to observer"))
    reviewed, errors, info = pooled_match.review_batch(project, [{"shot": shot, "asset": asset}], lambda: None)
    assert not reviewed and len(errors) == 1 and "空白或黑屏" in errors[0][1]
    assert not info.get("model_requests", 0)


@pytest.mark.parametrize("kind", ["letterbox", "white_page", "dark_scene"])
def test_visible_detail_still_requires_normal_review(monkeypatch, tmp_path, kind):
    picture = Image.new("RGB", (320, 180), "black" if kind != "white_page" else "white")
    draw = ImageDraw.Draw(picture)
    if kind == "letterbox":
        draw.rectangle((0, 40, 319, 140), fill=(45, 65, 80))
    elif kind == "white_page":
        draw.text((40, 40), "Visible source text", fill="black")
    else:
        draw.rectangle((60, 50, 240, 150), fill=(15, 15, 15))
    project, shot, asset = decoded_frames(monkeypatch, tmp_path, picture)
    _, preview = pooled_match.sample_frames(project["id"], shot, asset, lambda: None)
    assert not preview["blank_frames"]
    monkeypatch.setattr(frame_contract, "requirements", lambda *a: {"subject": "worker"})

    def audit(*args):
        raise RuntimeError("normal review still required")

    monkeypatch.setattr(planner, "audit_selected_frames", audit)
    with pytest.raises(RuntimeError, match="normal review still required"):
        pooled_match.review_batch(project, [{"shot": shot, "asset": asset}], lambda: None)
