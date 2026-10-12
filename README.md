# 镜稿：本地文稿转视频工作台

Python / FastAPI 后台、React / TypeScript 前端、FFmpeg 本地合成。输入中文文案，规划分镜，生成百度一镜配音，搜索与匹配已有视频素材，按实际语音时间戳生成字幕并输出 MP4。支持批量任务、素材库、费用账本和断点保存。

这是 **2026-10-13 的 v60 源码交接基线，尚未达到完整交付目标**。后端来自当时实际运行的冻结版本，不是原工作目录中混杂的实验副本。逐文件哈希见 [source-baseline.json](docs/source-baseline.json)。

先读 [交接说明](docs/HANDOFF.md) 和 [Claude 接手提示词](CLAUDE_HANDOFF_PROMPT.md)。

## 目标与最近结果

- 文案输入为主，可选批量；约四分钟视频从提交到成片不超过十分钟。
- 每条成本目标为 0.10 元。当前软件限制的是模型费；百度一镜配音费用尚未核明，必须另列，不能冒称完整成本已达标。
- 普通画面只需主题相关、观看自然、没有明显错配；具体具名人物须确认。不要为普通装备的精确型号、日期、全部动作另加苛刻门槛。
- 默认最低 720p、真实视频时长至少 80%；字幕断句自然、音画同步。最终需要 Windows x64 可运行交付。

最近一次完整正常 API 测试：[港口文案](examples/port.txt)，243.52 秒时间轴，运行 **529.718 秒**，模型费 **0.091750 元**，选中 **5/28 镜**，**0 导出**，因下一请求预算不足停止。原始素材下载、观察和人工处理未被作为成功成片。机器可读摘要见 [benchmark-v60.json](docs/benchmark-v60.json)。

## 在新的 Mac 上运行

准备 Python 3.13、Node.js / npm、FFmpeg 与 ffprobe。以下命令从仓库根目录执行；`uv` 可替换为普通 venv / pip。

```sh
uv venv --python 3.13
uv pip install -r requirements.lock
uv pip install -e ".[dev,captions,video]"
npm ci --prefix frontend
npm run build --prefix frontend
.venv/bin/python scripts/fetch_font.py
.venv/bin/python scripts/setup_captions.py
.venv/bin/python launch.py --port 8766 --no-browser
```

打开 `http://127.0.0.1:8766/`。服务只监听本机；远程连接可使用 SSH 端口转发，例如 `ssh -L 8766:127.0.0.1:8766 <mac-host>`。

模型配置和素材站 Key 在应用「服务与设置」中填写。原测试使用阿里云北京兼容接口 `https://dashscope.aliyuncs.com/compatible-mode/v1`、`qwen3.7-flash`，具体可用模型和计价须在新环境核实。百度一镜继续采用已有接入方式，需在新机器的专用浏览器中正常登录；本地字幕模型和字体由上述脚本下载。

仓库只带源代码、依赖清单、必要的文本测试夹具与交接文件，不带 API Key、Cookie、浏览器登录态、项目数据库、费用账本、素材、音视频、模型权重或构建产物。新机器的素材库为空；原机器的已积累索引和缓存不会随 Git 克隆，不能假设已有五万条索引或已下载素材。

## 验证与构建

交接用测试入口将应用数据放在临时目录，只允许本机测试服务器，阻止公网请求。定向基线：

```sh
.venv/bin/python scripts/test_offline.py \
  tests/test_editorial_background_v60.py \
  tests/test_background_integration_v60.py \
  tests/test_background_period_v60.py \
  tests/test_review_relation_v51.py \
  tests/test_review_transport_v50.py \
  tests/test_library_rank_context_v60.py \
  tests/test_rank_cache.py \
  tests/test_coarse_localization_v57.py \
  tests/test_audio_locator_v60.py \
  tests/test_downloaded_pool_v59.py
```

历史测试含旧协议预期及依赖本机研究产物的试验，不保证整套全绿；不要把这点误报为当前全部功能失败，也不要为迎合旧严格匹配测试恢复已撤销的产品要求。`scripts/evaluate_*` 中部分脚本是历史研究工具，所依赖的私有本地产物不在仓库。

Windows 打包前先构建前端并下载字体，再使用 `scripts/build_windows.py`。当前源码快照尚未完成新一轮 Windows 成片验收；早期 Windows 11 ARM 虚拟机中的 x64 冒烟不等于客户 Windows 10 x64 / RTX 3060 验收。第三方依赖说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

