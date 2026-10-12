import json

import pytest

from jinggao import library, paths, sources


@pytest.fixture
def seed(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    blob = library.file_path("blobs/fixture.mp4")
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"local-index-fixture")

    def insert(rows):
        with library.connection() as con:
            for row in rows:
                cid = row["id"]
                body = {"title": row.get("title", "现场记录"),
                        "description": row.get("source_description", ""),
                        "source_url": row.get("source_url", "https://v.qq.com/x/page/" + cid + ".html")}
                description, visible_text = row.get("description", ""), row.get("visible_text", "")
                relative = row.get("path", "blobs/fixture.mp4")
                updated = row.get("updated", 1)
                con.execute("INSERT INTO sources VALUES (?,?,?,?)",
                            (cid, body["source_url"], json.dumps(body, ensure_ascii=False), updated))
                con.execute("INSERT OR IGNORE INTO blobs VALUES (?,?,?,?,?,?,?)",
                            (relative, relative, "", row.get("bytes", blob.stat().st_size), 20,
                             row.get("width", 1280), row.get("height", 720)))
                con.execute("INSERT INTO clips VALUES (?,?,?,?,?,?,?,?,?)",
                            (cid, cid, relative, 0, 0, row.get("duration", 10),
                             description, visible_text, updated))
                terms = sources.tokens(" ".join((body["title"], body["description"], description, visible_text)))
                con.execute("INSERT INTO clip_terms VALUES (?,?)", (cid, " ".join(sorted(terms))))
    return insert


def test_older_relevant_window_survives_more_than_400_recent_matches(seed):
    seed([{"id": "older", "description": "温室里工人从枝头采摘番茄", "updated": 1},
          *[{"id": f"recent{i}", "title": "温室介绍", "description": "主持人讲话", "updated": i + 2}
            for i in range(450)]])

    result = library.search("温室 番茄 采摘 枝头", limit=1)

    assert [row["id"] for row in result] == ["older"]
    assert result[0]["description"] == "温室里工人从枝头采摘番茄"


def test_many_keyword_titles_cannot_displace_strong_observed_clues(seed):
    query = "分拣 扫描 条码 包裹"
    seed([{"id": "observed", "description": "工人分拣包裹并扫描条码", "updated": 1},
          *[{"id": f"title{i}", "title": query * 6, "description": "空旷仓库内静止的货架",
             "updated": i + 2} for i in range(420)]])

    result = library.search(query, limit=2)

    assert result[0]["id"] == "observed"
    assert result[0]["score"] == 8 and result[1]["score"] == 4


def test_scene_and_ocr_keep_double_weight_and_negations_keep_original_text(seed):
    seed([{"id": "scene", "description": "扫描条码", "updated": 1},
          {"id": "ocr", "visible_text": "扫描条码", "updated": 3},
          {"id": "title", "title": "扫描条码", "updated": 4},
          {"id": "absent", "description": "未出现扫描条码。只有货架", "updated": 5}])

    result = library.search("扫描 条码")

    assert [row["id"] for row in result] == ["scene", "ocr", "title", "absent"]
    assert [row["score"] for row in result] == [4, 4, 2, 0]
    assert result[1]["visible_text"] == "扫描条码"
    assert result[-1]["description"] == "未出现扫描条码。只有货架"


def test_invalid_high_scoring_rows_do_not_consume_the_result_limit(seed):
    seed([{"id": "missing", "description": "扫描 条码 分拣 包裹", "path": "blobs/missing.mp4"},
          {"id": "changed", "description": "扫描 条码 分拣 包裹", "path": "blobs/changed.mp4", "bytes": 999},
          {"id": "invalidpath", "description": "扫描 条码 分拣 包裹", "path": "../outside.mp4"},
          {"id": "invalidsource", "description": "扫描 条码 分拣 包裹", "source_url": "https://invalid.test/video"},
          {"id": "valid", "description": "扫描条码"}])
    library.file_path("blobs/changed.mp4").write_bytes(b"changed")

    assert [row["id"] for row in library.search("扫描 条码 分拣 包裹", limit=1)] == ["valid"]


def test_filters_and_maximum_result_count_still_apply(seed):
    seed([{"id": "too_short", "description": "扫描条码", "duration": 1},
          {"id": "small", "description": "扫描条码", "path": "blobs/small.mp4", "height": 360},
          *[{"id": f"valid{i}", "description": "扫描条码", "updated": i + 1} for i in range(110)]])
    library.file_path("blobs/small.mp4").write_bytes(b"local-index-fixture")

    result = library.search("扫描 条码", limit=1000, min_duration=5, min_height=720)

    assert len(result) == 100
    assert all(row["id"].startswith("valid") for row in result)
    assert result[0]["id"] == "valid109"
    assert library.search("。") == []
