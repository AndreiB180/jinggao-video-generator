"""CPU Mandarin timestamps, with no API transport or paid fallback.

Feature extraction and BiCIF timing follow FunASR's MIT-licensed ONNX
runtime (Copyright FunASR). See THIRD_PARTY_NOTICES.md for attribution.
The pinned model is the official Apache-2.0 quantized Paraformer export.
"""
from __future__ import annotations

import hashlib
import json
import math
import wave
from functools import lru_cache
from pathlib import Path
from threading import Lock

from . import paths

MODEL_ID = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx"
FILES = {
    "model_quant.onnx": "80fa5c8a87f5ae77ea2dc233ad220b9415343c8681d538e59f019f2acc837694",
    "am.mvn": "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5",
    "tokens.json": "2b20c2b12572d682afff84ce1c8d560f67b8b32a4c1f21567411d141ed352127",
}
MODEL_HASH = FILES["model_quant.onnx"]
_lock = Lock()


def model_dir():
    return paths.DATA / "models" / "paraformer-zh-timestamp"


def install_model(check=lambda: None):
    """Download public weights once; never upload narration or audio."""
    import httpx
    root = model_dir()
    root.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        check()
        target = root / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            with httpx.stream("GET", f"https://modelscope.cn/api/v1/models/{MODEL_ID}/repo",
                              params={"Revision": "master", "FilePath": name},
                              timeout=60, follow_redirects=True) as response:
                response.raise_for_status()
                hasher = hashlib.sha256()
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes(1024 * 1024):
                        check()
                        output.write(chunk)
                        hasher.update(chunk)
            if hasher.hexdigest() != digest:
                raise ValueError("本地字幕模型校验失败，请重新安装；未调用付费服务")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _runtime(directory: str):
    try:
        import numpy as np
        import onnxruntime as ort
        import kaldi_native_fbank as knf
    except ImportError:
        raise ValueError("本地字幕组件未安装，请运行字幕组件安装脚本；不会自动调用付费模型") from None
    root = Path(directory)
    for name, digest in FILES.items():
        target = root / name
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("本地字幕模型未安装或校验失败，请运行字幕组件安装脚本")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.enable_cpu_mem_arena = False
    session = ort.InferenceSession(str(root / "model_quant.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    if len(session.get_outputs()) != 4:
        raise ValueError("此语音模型没有字级时间戳，不能用于字幕对齐")
    lines = (root / "am.mvn").read_text().splitlines()
    cmvn = []
    for index, line in enumerate(lines):
        if line.startswith(("<AddShift>", "<Rescale>")):
            cmvn.append([float(v) for v in lines[index + 1].split()[3:-1]])
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.debug_mel = False
    opts.energy_floor = 0
    return session, opts, np.asarray(cmvn, dtype=np.float32), json.loads((root / "tokens.json").read_text())


def _recognize(samples, runtime):
    import numpy as np
    import kaldi_native_fbank as knf
    session, opts, cmvn, vocabulary = runtime
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, samples.tolist())
    if fbank.num_frames_ready < 3:
        return []
    features = np.array([fbank.get_frame(i) for i in range(fbank.num_frames_ready)], dtype=np.float32)
    count = math.ceil(len(features) / 6)
    padded = np.pad(features, ((3, 9), (0, 0)), mode="edge")
    features = np.array([padded[i * 6:i * 6 + 7].reshape(-1) for i in range(count)])
    features = (features + cmvn[0]) * cmvn[1]
    inputs = session.get_inputs()
    scores, lengths, _, peaks = session.run(None, {
        inputs[0].name: features[np.newaxis].astype(np.float32),
        inputs[1].name: np.array([count], dtype=np.int32),
    })
    ids = [int(i) for i in scores[0].argmax(axis=-1) if i not in (0, 2)]
    tokens = [vocabulary[i] for i in ids[:int(lengths[0]) - 1]]
    fires = np.where(peaks.reshape(-1) > 1 - 1e-4)[0] - 1.5
    if len(fires) != len(tokens) + 1:
        raise ValueError("本地语音时间戳不完整，已停止字幕生成")
    rows = []
    for index, token in enumerate(tokens):
        # BiCIF emits one boundary at each token onset; cap long pauses.
        begin = max(0.0, float(fires[index]) * .02)
        end = min(float(fires[index + 1]), float(fires[index]) + 30) * .02
        if end > begin and token not in {"<s>", "</s>", "<blank>"}:
            rows.append({"text": token.replace("@@", ""), "start": begin, "end": end})
    return rows


def _chunks(samples):
    """Bound CPU memory and choose a quiet frame near each 20-second boundary."""
    import numpy as np
    block = 320  # 20 ms
    padded = np.pad(samples.astype(np.float32), (0, (-len(samples)) % block))
    rms = np.sqrt(np.mean(padded.reshape(-1, block) ** 2, axis=1))
    begin = 0
    while len(samples) - begin > 26 * 16000:
        lo, hi = begin // block + 16 * 50, begin // block + 24 * 50
        smooth = np.convolve(rms[lo - 3:hi + 3], np.ones(7) / 7, mode="valid")
        end = (lo + int(smooth.argmin())) * block
        yield begin, end
        begin = end
    yield begin, len(samples)


def transcribe(wav_path, check=lambda: None, progress=lambda *_: None):
    import numpy as np
    with wave.open(str(wav_path), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError("字幕对齐需要 16kHz 单声道 PCM 音频")
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    while not _lock.acquire(timeout=.2):
        check()
    try:
        runtime = _runtime(str(model_dir()))
        chunks = list(_chunks(samples))
        words = []
        for index, (begin, end) in enumerate(chunks):
            check()
            for row in _recognize(samples[begin:end], runtime):
                words.append(row | {"start": row["start"] + begin / 16000,
                                    "end": min(row["end"] + begin / 16000, end / 16000)})
            progress((index + 1) / len(chunks), "本地识别配音时间戳 · 不调用付费模型")
        return words
    finally:
        # The model takes about 600 MB and each video needs it once (alignment): with several videos made
        # at once, keeping it in every worker filled 16 GB computers. Load it again next time instead.
        _runtime.cache_clear()
        import gc
        gc.collect()
        _lock.release()
