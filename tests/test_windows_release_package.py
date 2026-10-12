"""A portable release must contain the frozen application and public weights."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def builder(tmp_path):
    spec = importlib.util.spec_from_file_location("build_windows", Path(__file__).parents[1] / "scripts/build_windows.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUTPUT = tmp_path / "output"
    return module


def frozen(tmp_path):
    root = tmp_path / "runtime"
    files = {"launch.py": b"launch", "jinggao/local_asr.py": b"local_asr",
             "scripts/sapi.ps1": b"sapi", "frontend/dist/index.html": b"frontend",
             "data/cache/private.json": b"must not package"}
    for rel, content in files.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    (root / "jinggao/experiment.py").write_text("unlisted experiment")
    path = tmp_path / "release.json"
    path.write_text(json.dumps({"runtime": str(root), "manifest": {
        r: hashlib.sha256(c).hexdigest() for r, c in files.items()}}))
    return root, path


def test_package_copies_declared_runtime_not_unlisted_code_or_data(builder, tmp_path):
    _, release = frozen(tmp_path)
    source, manifest = builder.load_release(release)
    copied = builder.copy_release(source, manifest)
    assert set(copied) == {"launch.py", "jinggao/local_asr.py", "scripts/sapi.ps1", "frontend/dist/index.html"}
    assert not (builder.OUTPUT / "jinggao/experiment.py").exists()
    assert not (builder.OUTPUT / "data").exists()


@pytest.mark.parametrize("change", ["changed", "escaped"])
def test_changed_or_escaping_release_is_rejected_before_copy(builder, tmp_path, change):
    source, release = frozen(tmp_path)
    if change == "changed":
        (source / "launch.py").write_text("different program")
    else:
        value = json.loads(release.read_text())
        (tmp_path / "secret.txt").write_bytes(b"secret")
        value["manifest"]["../secret.txt"] = hashlib.sha256(b"secret").hexdigest()
        release.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        builder.load_release(release)
    assert not builder.OUTPUT.exists()


def test_only_hash_pinned_public_models_are_allowed_under_data(builder, tmp_path):
    source, _ = frozen(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    contents = {"model_quant.onnx": b"test weights", "am.mvn": b"test cmvn", "tokens.json": b"[]"}
    files = {k: hashlib.sha256(v).hexdigest() for k, v in contents.items()}
    (source / "jinggao/local_asr.py").write_text("FILES = " + repr(files))
    for name, value in contents.items():
        (model / name).write_bytes(value)
    (model / "settings.json").write_text("test private settings")
    public = builder.bundle_captions(source, model)
    packaged = [p for p in builder.OUTPUT.rglob("*") if p.is_file()]
    builder.check_private_files(packaged, public)
    assert len(packaged) == 3
    private = builder.OUTPUT / "data/settings.json"
    private.write_text("test key")
    with pytest.raises(RuntimeError):
        builder.check_private_files([*packaged, private], public)
    with (builder.OUTPUT / "data/models/paraformer-zh-timestamp/model_quant.onnx").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(RuntimeError):
        builder.check_private_files(packaged, public)
