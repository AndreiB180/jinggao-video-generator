"""A fractional 30fps input timestamp must not reject a real middle cut."""

from PIL import Image

from jinggao import frame_sampling, media, paths, store


def test_fractional_source_seek_uses_the_same_zero_origin_in_both_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    project = store.new_project("实际中段抽帧测试", "列车行驶")
    target = paths.file_path(project["id"], "assets/fractional.mp4")
    target.parent.mkdir()
    media.run(
        [
            paths.binary("ffmpeg"),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=30:duration=25",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-threads",
            "2",
            "-output_ts_offset",
            "0.033008",
            str(target),
        ],
        timeout=20,
    )
    asset = {"path": "assets/fractional.mp4", "duration": 25, "source_clip_start": 455.866688}
    shot = {"clip_start": 9.84, "duration": 11.2}
    folder = paths.project_dir(project["id"]) / "review"
    folder.mkdir()
    frames, preview = frame_sampling.sample(project["id"], shot, asset, folder, lambda: None)
    sampling = frame_sampling.validate_preview(shot, asset, preview)
    assert sampling["frame_count"] == 280 and sampling["indices"][0] == 0
    assert sampling["indices"][-1] == 279 and len(frames) == 5
    assert [f["at"] for f in preview["frames"]] == [
        round(465.706688 + i / 25, 3) for i in sampling["indices"]
    ]
    assert all(Image.open(f).size == (1280, 720) for f in frames)
