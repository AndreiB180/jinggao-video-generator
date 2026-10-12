"""Run after installing jinggao[captions]; downloads verified public weights."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jinggao.local_asr import install_model, model_dir  # noqa: E402

if __name__ == "__main__":
    install_model()
    print(f"本地字幕模型已安装并校验：{model_dir()}；不需要 API Key，不上传音轨。")
