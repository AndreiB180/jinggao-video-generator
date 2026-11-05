"""11-05: a customer's data folder passed 200 GB, and a new Windows 11 could not load the caption runtime."""
import json
import os
import sys
import time
import zipfile

import pytest

from jinggao import config, local_asr, media, paths, store, tidy


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    store.init()
    return tmp_path


def _film(data, p, age_days, saved=True, title="片"):
    """A project with one export, saved (or not) to the output folder, last touched age_days ago."""
    then = time.time() - age_days * 86400
    folder = paths.project_dir(p["id"]) / "exports" / "e1"
    folder.mkdir(parents=True)
    (folder / "video.mp4").write_bytes(b"v" * 1000)
    (paths.project_dir(p["id"]) / "assets").mkdir()
    (paths.project_dir(p["id"]) / "assets" / "clip.mp4").write_bytes(b"c" * 5000)
    out = data / "output" / f"{title}-{p['id'][:6]}.mp4"
    out.parent.mkdir(exist_ok=True)
    if saved:
        out.write_bytes(b"v" * 1000)
    export = {"id": "e1", "path": "exports/e1/video.mp4", "created": then, "saved_to": str(out) if saved else ""}
    with store.connection() as con:
        body = json.loads(con.execute("SELECT body FROM projects WHERE id=?", (p["id"],)).fetchone()[0])
        body["exports"] = [export]
        con.execute("UPDATE projects SET body=?, updated=? WHERE id=?", (json.dumps(body), then, p["id"]))
    return out


def test_old_saved_projects_go_and_the_rest_stay(data, monkeypatch):
    monkeypatch.setitem(config.DEFAULTS, "project_keep_days", "7")
    old = store.new_project("八天前做完的", "稿")
    film = _film(data, old, 8)
    recent = store.new_project("三天前做完的", "稿")
    _film(data, recent, 3)
    unsaved = store.new_project("没存出去的", "稿")
    _film(data, unsaved, 30, saved=False)
    busy = store.new_project("正在重新导出", "稿")
    _film(data, busy, 30)
    store.enqueue(busy["id"], "render", {})
    library = store.new_project("素材库", "")
    _film(data, library, 30)
    store.mutate(library["id"], lambda p: p.update(system_library=1), internal=True)
    with store.connection() as con:     # mutate stamps "now"; put it back in the past
        con.execute("UPDATE projects SET updated=? WHERE id=?", (time.time() - 30 * 86400, library["id"]))

    assert [p["id"] for p in tidy.old_projects()] == [old["id"]]
    report = tidy.run_once()
    assert report["projects"] == 1 and report["project_bytes"] >= 6000
    assert not (paths.DATA / "projects" / old["id"]).exists()
    assert film.is_file()                       # the film in the output folder stays
    with store.connection() as con:
        left = {r[0] for r in con.execute("SELECT id FROM projects")}
    assert left == {recent["id"], unsaved["id"], busy["id"], library["id"]}
    for p in (recent, unsaved, busy, library):
        assert (paths.project_dir(p["id"]) / "assets" / "clip.mp4").exists()


def test_keep_days_zero_keeps_everything(data, monkeypatch):
    monkeypatch.setitem(config.DEFAULTS, "project_keep_days", "0")
    p = store.new_project("很旧", "稿")
    _film(data, p, 100)
    assert tidy.old_projects() == []
    assert tidy.run_once()["projects"] == 0


def test_a_project_used_again_recently_stays(data, monkeypatch):
    """A job finished yesterday on an old project (a re-export) counts as touching it."""
    monkeypatch.setitem(config.DEFAULTS, "project_keep_days", "7")
    p = store.new_project("旧项目重导", "稿")
    _film(data, p, 20)
    job = store.enqueue(p["id"], "render", {})
    store.job_update(job["id"], status="completed")
    with store.connection() as con:
        con.execute("UPDATE projects SET updated=? WHERE id=?", (time.time() - 20 * 86400, p["id"]))
    assert tidy.old_projects() == []


def test_setting_is_checked(data):
    config.update({"project_keep_days": "14"})
    assert config.read()["project_keep_days"] == "14"
    with pytest.raises(ValueError):
        config.update({"project_keep_days": "七天"})


def test_stale_cache_entries_go(data):
    shots = paths.DATA / "cache" / "express" / "bili-shots"
    files = paths.DATA / "cache" / "express" / "pixabay-files"
    text = paths.DATA / "cache" / "express" / "text"
    for folder in (shots / "BVold", shots / "BVnew", files, text):
        folder.mkdir(parents=True, exist_ok=True)
    old = time.time() - 20 * 86400
    for path, when in ((shots / "BVold" / "meta.json", old), (shots / "BVold" / "sprite-0.jpg", old),
                       (shots / "BVnew" / "meta.json", old), (shots / "BVnew" / "sprite-0.jpg", time.time()),
                       (files / "1-large.mp4", old), (files / "2-large.mp4", time.time()), (text / "a.json", old)):
        path.write_bytes(b"x" * 100)
        os.utime(path, (when, when))
    assert tidy.prune_caches() == 300
    assert not (shots / "BVold").exists()
    assert (shots / "BVnew" / "meta.json").exists()   # one fresh file keeps the video's sheets together
    assert not (files / "1-large.mp4").exists() and (files / "2-large.mp4").exists()
    assert (text / "a.json").exists()                 # small answer caches are left to their own expiry


def test_delivered_film_is_one_file_on_disk(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"film")
    target = tmp_path / "out" / "成片.mp4"
    target.parent.mkdir()
    media.place(source, target)
    assert target.read_bytes() == b"film"
    assert os.stat(target).st_ino == os.stat(source).st_ino
    source.unlink()                                   # the project going away leaves the saved film whole
    assert target.read_bytes() == b"film"


def test_delivery_copies_when_a_link_is_refused(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"film")
    monkeypatch.setattr(os, "link", lambda *a: (_ for _ in ()).throw(OSError("another drive")))
    media.place(source, tmp_path / "copy.mp4")
    assert (tmp_path / "copy.mp4").read_bytes() == b"film"
    assert os.stat(tmp_path / "copy.mp4").st_ino != os.stat(source).st_ino


def test_delivery_zip_is_made_when_downloaded(data):
    from fastapi.testclient import TestClient
    from jinggao.api import create_app
    p = store.new_project("交付包", "稿")
    folder = paths.project_dir(p["id"]) / "exports" / "e1"
    folder.mkdir(parents=True)
    for name, body in (("video.mp4", b"v"), ("subtitles.srt", "1\n字幕".encode()), ("sources.json", b"[]"),
                       ("project.json", b"{}")):
        (folder / name).write_bytes(body)
    client = TestClient(create_app())
    response = client.get(f"/api/projects/{p['id']}/files/exports/e1/delivery.zip?download=true")
    assert response.status_code == 200
    with zipfile.ZipFile(folder / "delivery.zip") as z:
        assert set(z.namelist()) == {"video.mp4", "subtitles.srt", "sources.json", "project.json"}
    assert client.get(f"/api/projects/{p['id']}/files/exports/e2/delivery.zip").status_code == 404


def test_caption_runtime_failure_names_the_fix(monkeypatch):
    monkeypatch.setitem(sys.modules, "kaldi_native_fbank", None)    # import fails as a missing DLL makes it
    with pytest.raises(ValueError) as caught:
        local_asr._runtime.__wrapped__("unused")
    assert "vc_redist.x64.exe" in str(caught.value) and "安装脚本" not in str(caught.value)


def test_each_rewrite_has_its_own_budget(monkeypatch):
    """对标采集 billed every rewrite to one "benchmark" project, whose ¥0.30 lifetime cap stopped it after ~20."""
    from jinggao import rewrite
    seen = []

    class Scope:
        def __init__(self, pid, *a):
            seen.append(pid)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(rewrite.costs, "scope", Scope)
    monkeypatch.setattr(rewrite.planner, "chat", lambda *a, **k: ({"script": ["这是一句改写后的口播。"] * 8,
                                                                   "title": "标题"}, {"usage": {}}))
    monkeypatch.setattr(rewrite, "fact_check", lambda script, source, project_id, check=None: (
        seen.append(project_id), {"script": script, "fixes": [], "unmatched_fixes": [], "checked": True,
                                  "usage": {}})[1])
    for _ in range(2):
        rewrite.rewrite("原文" * 40, title="对标", project_id="benchmark")
    assert seen[0].startswith("benchmark-") and seen[0] == seen[1] and seen[2] != seen[0]
    real = "a" * 32
    rewrite.rewrite("原文" * 40, project_id=real)
    assert seen[-1] == real
