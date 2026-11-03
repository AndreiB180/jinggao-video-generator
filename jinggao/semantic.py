"""Meaning-level search over the footage library's frame descriptions.

The word search (footage_index.find) matches a line's short picture words ("男子 说话") against what each frame
was described as; it finds pictures that look the part and say nothing about the story. Here every frame
description and every narration line becomes a vector from a small Chinese text model (bge-small-zh, int8 ONNX,
run with the onnxruntime the app already ships for OCR), and frames are ranked by closeness of meaning.

The tokenizer is BERT's, written out here (Chinese characters one by one, WordPiece for the rest) so the app needs
no new library."""
from __future__ import annotations

import json
import threading
import unicodedata
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import paths

MODEL_DIR = paths.DATA / "models" / "bge-small-zh"
VECTORS_DIR = paths.DATA / "footage-vectors"      # scripts/build_semantic_index.py pack
MAX_TOKENS = 64
TOP_TEXTS = 1500        # closest descriptions read per query (one description can be on several frames)
CHUNK = 131072          # index rows turned to float per step: ~130 MB at 256 dims, whatever the index size
CACHE = 512             # queries remembered (a film asks the same line again in later fills)

_lock = threading.Lock()
_state: dict = {}
_index_lock = threading.Lock()
_index: dict = {}
_answers: OrderedDict = OrderedDict()


def available(model_dir: Path | None = None) -> bool:
    folder = model_dir or MODEL_DIR
    return (folder / "model_quantized.onnx").exists() and (folder / "vocab.txt").exists()


INDEX_FILES = ("meta.json", "vectors.npy", "basis.npy", "mean.npy", "frames.npy", "sources.txt")


def ready(model_dir: Path | None = None, vectors_dir: Path | None = None) -> bool:
    folder = vectors_dir or VECTORS_DIR
    return available(model_dir) and all((folder / n).exists() for n in INDEX_FILES)


class Tokenizer:
    def __init__(self, vocab_file: Path):
        self.vocab = {}
        for i, line in enumerate(vocab_file.read_text("utf-8").splitlines()):
            self.vocab[line.rstrip("\n")] = i
        self.unk, self.cls, self.sep, self.pad = (self.vocab[t] for t in ("[UNK]", "[CLS]", "[SEP]", "[PAD]"))

    @staticmethod
    def _cjk(cp: int) -> bool:
        return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF
                or 0x2F800 <= cp <= 0x2FA1F)

    def _basic(self, text: str) -> list[str]:
        out, word = [], []
        def flush():
            if word:
                out.append("".join(word))
                word.clear()
        for ch in unicodedata.normalize("NFD", text.lower()):
            if unicodedata.category(ch) == "Mn":
                continue      # accents stripped (uncased model)
            cp = ord(ch)
            if ch.isspace() or cp in (0, 0xFFFD) or unicodedata.category(ch).startswith("C"):
                flush()
            elif self._cjk(cp) or unicodedata.category(ch).startswith("P") or (33 <= cp <= 47 or 58 <= cp <= 64
                                                                                 or 91 <= cp <= 96 or 123 <= cp <= 126):
                flush()
                out.append(ch)
            else:
                word.append(ch)
        flush()
        return out

    def _wordpiece(self, word: str) -> list[int]:
        if len(word) > 100:
            return [self.unk]
        ids, start = [], 0
        while start < len(word):
            end, piece = len(word), None
            while start < end:
                sub = word[start:end] if start == 0 else "##" + word[start:end]
                if sub in self.vocab:
                    piece = self.vocab[sub]
                    break
                end -= 1
            if piece is None:
                return [self.unk]
            ids.append(piece)
            start = end
        return ids

    def encode(self, text: str, max_tokens: int = MAX_TOKENS) -> list[int]:
        ids = [self.cls]
        for word in self._basic(text):
            ids += self._wordpiece(word)
            if len(ids) >= max_tokens - 1:
                break
        return ids[:max_tokens - 1] + [self.sep]


def _load(model_dir: Path | None = None, threads: int = 2):
    folder = model_dir or MODEL_DIR
    with _lock:
        key = str(folder)
        if _state.get("key") != key:
            import onnxruntime
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = threads
            session = onnxruntime.InferenceSession(str(folder / "model_quantized.onnx"), options,
                                                   providers=["CPUExecutionProvider"])
            _state.update(key=key, session=session, tokenizer=Tokenizer(folder / "vocab.txt"),
                          inputs={i.name for i in session.get_inputs()})
        return _state["session"], _state["tokenizer"], _state["inputs"]


def embed(texts: list[str], model_dir: Path | None = None, batch: int = 64, threads: int = 2) -> np.ndarray:
    """Unit vectors (float32, 512 wide), one row per text."""
    session, tokenizer, inputs = _load(model_dir, threads)
    rows = []
    for start in range(0, len(texts), batch):
        ids = [tokenizer.encode(t or "") for t in texts[start:start + batch]]
        width = max(len(x) for x in ids)
        input_ids = np.full((len(ids), width), tokenizer.pad, dtype=np.int64)
        mask = np.zeros((len(ids), width), dtype=np.int64)
        for i, x in enumerate(ids):
            input_ids[i, :len(x)] = x
            mask[i, :len(x)] = 1
        feed = {"input_ids": input_ids, "attention_mask": mask}
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        hidden = session.run(None, feed)[0]
        vectors = hidden[:, 0, :]                         # bge: the [CLS] token
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        rows.append(vectors.astype(np.float32))
    return np.concatenate(rows) if rows else np.zeros((0, 512), dtype=np.float32)


class Index:
    """The packed index: int8 vectors of the distinct descriptions (PCA-reduced, memory-mapped, so several
    films at once share one copy through the page cache) and the frame table (description, video, time)."""

    def __init__(self, folder: Path):
        self.meta = json.loads((folder / "meta.json").read_text("utf-8"))
        self.codes = np.load(folder / "vectors.npy", mmap_mode="r")
        self.mean = np.load(folder / "mean.npy").astype(np.float32)
        self.basis = np.load(folder / "basis.npy").astype(np.float32)
        self.scale = float(self.meta["scale"]) / 127.0
        frames = np.load(folder / "frames.npy")
        self.sources = (folder / "sources.txt").read_text("utf-8").split("\n")
        order = np.argsort(frames["text"], kind="stable")
        self.by_text = frames[order]
        self.text_starts = np.searchsorted(self.by_text["text"], np.arange(len(self.codes) + 1))
        order = np.lexsort((frames["t"], frames["source"]))
        self.by_source = frames[order]
        self.source_starts = np.searchsorted(self.by_source["source"], np.arange(len(self.sources) + 1))

    def reduce(self, vectors: np.ndarray) -> np.ndarray:
        out = (vectors - self.mean) @ self.basis
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    def closest(self, queries: np.ndarray, top: int = TOP_TEXTS) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per query: (description ids, similarities), best first."""
        best = [(np.zeros(0, np.int64), np.zeros(0, np.float32)) for _ in range(len(queries))]
        q = queries.T.astype(np.float32)
        for start in range(0, len(self.codes), CHUNK):
            sims = (np.asarray(self.codes[start:start + CHUNK], dtype=np.float32) @ q) * self.scale
            for i in range(len(queries)):
                col = sims[:, i]
                k = min(top, len(col))
                pick = np.argpartition(-col, k - 1)[:k]
                ids = np.concatenate([best[i][0], pick + start])
                vals = np.concatenate([best[i][1], col[pick]])
                keep = np.argsort(-vals)[:top]
                best[i] = (ids[keep], vals[keep])
        return best

    def video_similarity(self, source: int, query: np.ndarray) -> np.ndarray:
        """Similarity of each indexed frame of one video to the query, in time order, with the times."""
        rows = self.by_source[self.source_starts[source]:self.source_starts[source + 1]]
        codes = np.asarray(self.codes[rows["text"]], dtype=np.float32)
        return rows["t"], (codes @ query) * self.scale


def _load_index(folder: Path | None = None) -> Index | None:
    folder = folder or VECTORS_DIR
    with _index_lock:
        if _index.get("key") != str(folder):
            _index.clear()
            whole = all((folder / n).exists() for n in INDEX_FILES)
            _index.update(key=str(folder), index=Index(folder) if whole else None)
        return _index["index"]


def nearest(queries: list[str], *, model_dir: Path | None = None, vectors_dir: Path | None = None) -> list[dict]:
    """Per query, the library frames closest to it in meaning:
    {"videos": {"bilibili:BV…": {"sims": {t: similarity}, "video": mean similarity of the video's other frames}},
     "top": best similarity}. Empty answers when the model or the index is missing."""
    index = _load_index(vectors_dir)
    if index is None or not available(model_dir):
        return [{"videos": {}, "top": 0.0} for _ in queries]
    todo = [q for q in dict.fromkeys(queries) if q and q not in _answers]
    if todo:
        reduced = index.reduce(embed(todo, model_dir))
        for text, query, (ids, vals) in zip(todo, reduced, index.closest(reduced)):
            videos: dict = {}
            for text_id, sim in zip(ids.tolist(), vals.tolist()):
                rows = index.by_text[index.text_starts[text_id]:index.text_starts[text_id + 1]]
                for source, t in zip(rows["source"].tolist(), rows["t"].tolist()):
                    entry = videos.setdefault(int(source), {})
                    entry[round(float(t), 2)] = max(entry.get(round(float(t), 2), -1.0), float(sim))
            answer = {}
            for source, sims in videos.items():
                times, all_sims = index.video_similarity(source, query)
                others = [float(s) for t, s in zip(times.tolist(), all_sims.tolist()) if round(float(t), 2) not in sims]
                answer[index.sources[source]] = {"sims": sims,
                                                 "video": float(np.mean(others)) if others else None}
            with _lock:
                _answers[text] = {"videos": answer, "top": float(vals[0]) if len(vals) else 0.0}
                while len(_answers) > CACHE:
                    _answers.popitem(last=False)
    with _lock:
        return [_answers.get(q) or {"videos": {}, "top": 0.0} for q in queries]


def pack_index(vectors: np.ndarray, frames: np.ndarray, sources: list[str], dest: Path, dims: int = 256,
               info: dict | None = None) -> dict:
    """Write an index folder: the descriptions' unit vectors centred and turned by PCA to `dims` (fit on up
    to 200k of them), unit length again, int8; frames = structured rows (text id, source id, t)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    rng = np.random.default_rng(0)
    sample = vectors[rng.choice(len(vectors), size=min(200000, len(vectors)), replace=False)]
    mean = sample.mean(axis=0)
    _, _, vt = np.linalg.svd(sample - mean, full_matrices=False)
    dims = min(dims, vt.shape[0])
    basis = vt[:dims].T.astype(np.float32)
    reduced = (vectors - mean) @ basis
    reduced /= np.maximum(np.linalg.norm(reduced, axis=1, keepdims=True), 1e-9)
    scale = float(np.percentile(np.abs(reduced), 99.99)) or 1.0
    codes = np.clip(np.round(reduced / scale * 127), -127, 127).astype(np.int8)
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "vectors.npy", codes)
    np.save(dest / "basis.npy", basis)
    np.save(dest / "mean.npy", mean.astype(np.float32))
    np.save(dest / "frames.npy", np.asarray(frames, dtype=[("text", "<i4"), ("source", "<i4"), ("t", "<f4")]))
    (dest / "sources.txt").write_text("\n".join(sources), "utf-8")
    meta = dict(info or {}) | {"dims": dims, "scale": scale, "texts": len(codes), "frames": len(frames)}
    (dest / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    return meta
