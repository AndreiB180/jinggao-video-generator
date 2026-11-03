"""Build the meaning-level index of the shipped footage library (jinggao/semantic.py reads it).

    python scripts/build_semantic_index.py corpus OUT          # usable filmed frames of assets/footage-index
    python scripts/build_semantic_index.py embed OUT [--model D] [--workers N]
    python scripts/build_semantic_index.py pack OUT DEST [--dims 256]

corpus: every filmed frame find() could return whatever the line (filmed neighbours, title/length/AI/presenter
rules; foreign, racy and checks are decided per query), its description without the search words, deduplicated.
embed: unit vectors of each distinct description (resumable, in chunks). pack: PCA to --dims, int8, plus the
frame table, into DEST (the folder the app ships as data/footage-vectors)."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jinggao import footage_index, semantic  # noqa: E402

CHUNK = 20000


def seed_fingerprint(folder: Path) -> str:
    digest = hashlib.sha256()
    for part in sorted(folder.glob("part-*.jsonl.gz")):
        digest.update(part.read_bytes())
    return digest.hexdigest()


def corpus(out: Path, seed: Path = footage_index.SEED) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sources, frames, texts, index = [], [], [], {}
    videos = kept_videos = 0
    for part in sorted(seed.glob("part-*.jsonl.gz")):
        with gzip.open(part, "rt", encoding="utf-8") as lines:
            for line in lines:
                item = json.loads(line)
                row = item.get("row") or {}
                videos += 1
                moments = [m for m in item.get("moments") or [] if isinstance(m, dict) and "t" in m]
                usable = footage_index.indexable_frames(row, moments)
                if not usable:
                    continue
                kept_videos += 1
                sources.append(f"{row.get('provider')}:{row.get('id')}")
                s = len(sources) - 1
                for t, text in usable:
                    i = index.get(text)
                    if i is None:
                        i = index[text] = len(texts)
                        texts.append(text)
                    frames.append((i, s, t))
        print(part.name, videos, kept_videos, len(frames), len(texts), flush=True)
    table = np.array(frames, dtype=[("text", "<i4"), ("source", "<i4"), ("t", "<f4")])
    np.save(out / "frames.npy", table)
    (out / "sources.txt").write_text("\n".join(sources) + "\n", "utf-8")
    (out / "texts.txt").write_text("\n".join(texts) + "\n", "utf-8")
    (out / "corpus.json").write_text(json.dumps({"seed": seed_fingerprint(seed), "videos": videos,
                                                 "indexed_videos": kept_videos, "frames": len(frames),
                                                 "texts": len(texts)}, ensure_ascii=False), "utf-8")


def _embed_chunk(args) -> str:
    out, start, model = args
    target = out / "chunks" / f"{start:08d}.npy"
    if target.exists():
        return target.name
    texts = (out / "texts.txt").read_text("utf-8").splitlines()[start:start + CHUNK]
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    vectors = semantic.embed(texts, Path(model) if model else None, batch=128, threads=3).astype(np.float16)
    tmp = target.with_suffix(".tmp.npy")
    np.save(tmp, vectors)
    tmp.rename(target)
    return target.name


def embed(out: Path, model: str | None, workers: int) -> None:
    (out / "chunks").mkdir(exist_ok=True)
    total = len((out / "texts.txt").read_text("utf-8").splitlines())
    starts = list(range(0, total, CHUNK))
    began, done = time.time(), 0
    with ProcessPoolExecutor(workers) as pool:
        for name in pool.map(_embed_chunk, [(out, s, model) for s in starts]):
            done += 1
            rate = done * CHUNK / max(1.0, time.time() - began)
            print(f"{name} {done}/{len(starts)} {rate:.0f} texts/s", flush=True)


def pack(out: Path, dest: Path, dims: int) -> None:
    chunks = sorted((out / "chunks").glob("*[0-9].npy"))
    vectors = np.concatenate([np.load(c).astype(np.float32) for c in chunks])
    info = json.loads((out / "corpus.json").read_text("utf-8"))
    assert len(vectors) == info["texts"], (len(vectors), info["texts"])
    sources = (out / "sources.txt").read_text("utf-8").splitlines()
    meta = semantic.pack_index(vectors, np.load(out / "frames.npy"), sources, dest, dims,
                               info | {"model": "bge-small-zh-v1.5 int8 onnx, CLS", "built": time.time()})
    print("packed", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("corpus", "embed", "pack"))
    parser.add_argument("out", type=Path)
    parser.add_argument("dest", type=Path, nargs="?")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dims", type=int, default=256)
    args = parser.parse_args()
    if args.step == "corpus":
        corpus(args.out)
    elif args.step == "embed":
        embed(args.out, args.model, args.workers)
    else:
        pack(args.out, args.dest, args.dims)


if __name__ == "__main__":
    main()
