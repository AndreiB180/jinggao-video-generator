"""Meaning-level library search: the tokenizer, the packed int8 index and find_like's use of find()'s rules."""
import numpy as np
import pytest

from jinggao import footage_index as fi, paths, semantic

TOPICS = ["招聘", "图书馆", "唱歌", "军装", "厨房", "汽车", "工厂", "沙发"]


def fake_embed(texts, model_dir=None, batch=64, threads=2):
    """A vector per text from the topic words it carries, plus a little of its length (so no two are equal)."""
    out = []
    for text in texts:
        v = np.array([1.0 if word in text else 0.0 for word in TOPICS] + [0.05 * len(text) % 1, 0.1])
        out.append(v / np.linalg.norm(v))
    return np.array(out, dtype=np.float32)


def row(bvid, title="纪实", duration=300.0):
    return {"provider": "bilibili", "id": bvid, "url": f"https://www.bilibili.com/video/{bvid}", "title": title,
            "tags": "", "description": "", "duration": duration, "author": "up", "typename": "社会", "play": 1,
            "pubdate": 0, "query": ""}


@pytest.fixture
def library(tmp_path, monkeypatch):
    """Videos in a temporary index database and their packed vectors, built the way the release is."""
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    monkeypatch.setattr(semantic, "embed", fake_embed)
    monkeypatch.setattr(semantic, "available", lambda model_dir=None: True)
    monkeypatch.setattr(semantic, "VECTORS_DIR", tmp_path / "vectors")
    semantic._answers.clear()
    semantic._index.clear()

    def build(videos):
        texts, index, frames, sources = [], {}, [], []
        for bvid, moments, title in videos:
            fi.add(row(bvid, title), [{"t": float(t), "hash": "", "text": text, "kind": kind}
                                      for t, text, kind in moments], described=True)
            usable = fi.indexable_frames(row(bvid, title), [{"t": t, "text": x, "kind": k} for t, x, k in moments])
            if not usable:
                continue
            sources.append(f"bilibili:{bvid}")
            for t, text in usable:
                if text not in index:
                    index[text] = len(texts)
                    texts.append(text)
                frames.append((index[text], len(sources) - 1, t))
        semantic.pack_index(fake_embed(texts), np.array(frames, dtype=[("text", "<i4"), ("source", "<i4"),
                                                                     ("t", "<f4")]),
                            sources, tmp_path / "vectors", dims=10)
        semantic._index.clear()
        semantic._answers.clear()
    return build


def test_tokenizer_splits_chinese_by_character_and_wordpieces_the_rest(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "男", "子", "看", "手", "机", "ai", "##pad", "，"]),
                     "utf-8")
    tok = semantic.Tokenizer(vocab)
    assert tok.encode("男子看手机") == [2, 4, 5, 6, 7, 8, 3]
    assert tok.encode("AIpad，男") == [2, 9, 10, 11, 4, 3]          # lower-cased, ## pieces, punctuation alone
    assert tok.encode("猫") == [2, 1, 3]
    assert len(tok.encode("男" * 200)) == semantic.MAX_TOKENS


def test_indexable_frames_follow_finds_rules_for_videos_and_neighbours():
    filmed = [{"t": 10.0, "text": "男子在招聘会排队｜招聘会", "kind": "实拍"},
              {"t": 50.0, "text": "人群在大厅等候", "kind": "实拍"},
              {"t": 90.0, "text": "主持人讲解", "kind": "主持人出镜"}]
    assert fi.indexable_frames(row("BV1"), filmed) == [(10.0, "男子在招聘会排队")]   # 50 s sits by a presenter
    assert fi.indexable_frames(row("BV1", title="王者荣耀 教程"), filmed) == []
    assert fi.indexable_frames(row("BV1", duration=10.0), filmed) == []
    assert fi.indexable_frames(row("BV1", title="AI生成 短剧"), filmed) == []
    assert fi.indexable_frames(row("BV1"), [{"t": 1.0, "text": "实拍｜实拍", "kind": "实拍"}]) == []


def test_pack_and_closest_find_the_nearest_descriptions_across_chunks(tmp_path, monkeypatch):
    texts = [f"{word}场景{i}" for i in range(30) for word in TOPICS]
    frames = np.array([(i, i % 7, float(i)) for i in range(len(texts))],
                      dtype=[("text", "<i4"), ("source", "<i4"), ("t", "<f4")])
    semantic.pack_index(fake_embed(texts), frames, [f"bilibili:BV{i}" for i in range(7)], tmp_path, dims=10)
    monkeypatch.setattr(semantic, "CHUNK", 17)        # several steps over the rows
    index = semantic.Index(tmp_path)
    ids, sims = index.closest(index.reduce(fake_embed(["图书馆里看书"])), top=5)[0]
    assert all("图书馆" in texts[i] for i in ids) and list(sims) == sorted(sims, reverse=True)
    assert index.codes.dtype == np.int8 and index.codes.shape == (len(texts), 10)


def test_find_like_ranks_by_meaning_with_finds_rules(library):
    library([
        ("BV1jobfair01", [(10, "年轻人在招聘会排队", "实拍"), (60, "招聘会现场人群", "实拍"), (120, "招聘展板", "实拍")], "纪实"),
        ("BV1jobstray", [(10, "招聘会门口", "实拍"), (60, "厨房做饭", "实拍"), (120, "沙发上看电视", "实拍")], "生活"),
        ("BV1singer01", [(10, "女子唱歌", "实拍"), (60, "舞台唱歌", "实拍")], "音乐"),
        ("BV1jobhost1", [(10, "招聘会现场", "实拍"), (60, "主持人讲解", "主持人出镜"), (120, "主持人", "主持人出镜")], "解读"),
    ])
    rows = fi.find_like("招聘会上排队的求职者")
    assert [r["id"] for r in rows][:2] == ["BV1jobfair01", "BV1jobstray"]     # the video about it first
    assert "BV1singer01" not in [r["id"] for r in rows]                         # far below the best: left out
    assert "BV1jobhost1" not in [r["id"] for r in rows]                         # presenter-led, as in find()
    best = rows[0]
    assert best["indexed"] and best["match"]["like"] > 0.6 and "招聘" in best["match"]["text"]
    assert fi.find_like("招聘会", exclude={"bilibili:BV1jobfair01"})[0]["id"] == "BV1jobstray"
    assert fi.find_like("") == []


def test_find_like_skips_frames_a_check_found_bad_and_puts_clean_ones_first(library):
    library([("BV1factory1", [(10, "工厂车间流水线", "实拍"), (60, "工厂工人操作机器", "实拍")], "纪实"),
             ("BV1factory2", [(10, "工厂车间流水线", "实拍"), (60, "工厂车间流水线", "实拍")], "纪实")])
    fi.record_checks([{"source": "bilibili:BV1factory1", "t": 10.0, "status": "text"},
                      {"source": "bilibili:BV1factory1", "t": 60.0, "status": "text"},
                      {"source": "bilibili:BV1factory2", "t": 60.0, "status": "clean"}])
    rows = fi.find_like("工厂车间")
    assert [r["id"] for r in rows] == ["BV1factory2"]
    assert rows[0]["match"]["check"] == "clean" and rows[0]["match"]["t"] == 60.0


def test_without_the_index_find_like_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    monkeypatch.setattr(semantic, "VECTORS_DIR", tmp_path / "missing")
    semantic._index.clear()
    assert fi.find_like("招聘会") == []
