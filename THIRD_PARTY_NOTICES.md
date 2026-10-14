# 第三方组件

便携包附带原始许可证与依赖元数据。下列链接用于查阅来源；商用素材授权应逐项核实，并记录于项目素材说明。

- Python 3.13.12 嵌入式 Windows x64：[Python 官方发行页](https://www.python.org/downloads/release/python-31312/)，许可证位于 `runtime/python/LICENSE.txt`。
- FFmpeg：来自 [Gyan Windows builds](https://www.gyan.dev/ffmpeg/builds/)，由 [FFmpeg 下载页](https://ffmpeg.org/download.html) 推荐。Gyan essentials 静态构建依照 GPL v3 分发；原始 LICENSE、README 位于 `runtime/ffmpeg`，版本与下载哈希位于 `runtime/downloads.json`。对应源码和构建信息通过该发行页提供；再次分发者应履行适用许可证要求。
- Noto Sans SC：[Noto CJK](https://github.com/notofonts/noto-cjk)，SIL Open Font License 1.1，许可证位于 `assets/fonts/LICENSE`。
- Noto Sans SC Black、Noto Serif SC Black（字幕字体）：[Noto CJK](https://github.com/notofonts/noto-cjk)，SIL Open Font License 1.1，许可证位于 `assets/fonts/LICENSE` 与 `assets/fonts/LICENSE-NotoSerifSC.txt`。
- 站酷庆科黄油体 ZCOOL QingKe HuangYou、站酷快乐体 ZCOOL KuaiLe（可选字幕字体）：来自 [Google Fonts](https://github.com/google/fonts)，SIL Open Font License 1.1，许可证位于 `assets/fonts/OFL-*.txt`。
- Python 依赖（FastAPI、Uvicorn、HTTPX、Pillow、Pydantic、websocket-client 及传递依赖）来自 PyPI。各组件许可证/NOTICE 位于 `runtime/python/Lib/site-packages` 内相应 `.dist-info` 目录。
- 前端使用 React、Vite、TypeScript、Lucide React；版本及许可证记录见前端源码中的 `package-lock.json` 及发行包 `frontend/licenses.json`。
- 公开视频元数据提取：[yt-dlp 2026.08.19](https://github.com/yt-dlp/yt-dlp/releases/tag/2026.08.19) 官方 Windows 可执行程序，包含 EJS；主项目及捆绑依赖的许可证位于 `runtime/yt-dlp`。不修改二进制，SHA256 与官方校验文件一并核对。
- JavaScript 运行时：[Deno v2.9.7](https://github.com/denoland/deno/releases/tag/v2.9.7) 官方 Windows x64，MIT 许可位于 `runtime/deno`。只供公开视频播放器解析，无需客户安装 Node；Windows 需满足 Deno 当前支持的系统版本。

FFmpeg 通过独立进程调用。自动匹配结果的来源信息会写入导出清单，软件不改变原始素材的权利归属。

本地字幕组件（可选安装）：

- [FunASR ONNX runtime](https://github.com/modelscope/FunASR/tree/main/runtime/python/onnxruntime)：特征抽取和 BiCIF 时间戳算法参考其 MIT 实现，Copyright FunASR。具体许可见下文。
- [官方 Paraformer ONNX 量化模型](https://modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx)：模型仓库声明 Apache License 2.0；下载文件按代码中的 SHA256 校验，不包含用户音频或 API Key。
- ONNX Runtime、kaldi-native-fbank、NumPy、jieba：使用 PyPI 分发，许可保留在对应包的元数据中。

FunASR MIT License（所参考的运行时算法）：

Copyright FunASR. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
