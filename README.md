# 镜稿：本地文稿转视频工作台

Python / FastAPI 后台、React / TypeScript 前端、FFmpeg 本地合成。输入中文文案（或一条新闻，先由 AI 洗稿；或直接上传一段口播音频），按“一行字幕一个镜头”规划分镜，用千问配音（音频成片则用原声），从 B 站长视频与 Pixabay 图库自动选片并去水印，按实际语音时间戳烧录字幕并输出 MP4。支持新闻选题、批量任务、素材库、费用账本和断点保存。

当前默认是 **快速流程（express）**，2026-10-14 交付版。v60 基线（2026-10-13）的经典流程仍保留，只能通过设置 `pipeline=classic` 使用。逐文件哈希见 [source-baseline.json](docs/source-baseline.json)（v60 基线）。

先读 [交接说明](docs/HANDOFF.md) 和 [Claude 接手提示词](CLAUDE_HANDOFF_PROMPT.md)。

## 目标与最近结果

- 文案输入为主，可选批量；约四分钟视频从提交到成片不超过十分钟。
- 每条成本目标为 0.10 元（模型费）。2026-10-14 起客户放弃百度一镜，改用千问配音 qwen3-tts-flash：按字数计费约 0.16 元 / 千字，单独记账、单独设每日上限（默认 5 元），不占模型预算。一个阿里云百炼 Key 覆盖文案、看图和配音。
- 普通画面只需主题相关、观看自然、没有明显错配；具体具名人物须确认。不要为普通装备的精确型号、日期、全部动作另加苛刻门槛。
- 默认最低 720p、真实视频时长至少 80%；字幕断句自然、音画同步。最终需要 Windows x64 可运行交付。

最近结果（快速流程，德国 Mac，匿名 B 站，配音为系统中文语音）：

| 文稿 | 镜头 | 成片 | 总耗时 | 模型费 |
| --- | --- | --- | --- | --- |
| [港口](examples/port.txt) | 89 | 4 分 53 秒 | 299.7 秒 | 0.026 元 |
| [梁靖崑（人物多）](examples/liangjingkun.txt) | 219 | 11 分 55 秒 | 607 秒 | 0.050 元 |
| 中新网航展新闻（链接 → 洗稿 → 成片） | 22 | 76 秒 | 约 150 秒 | 0.0138 元（含洗稿） |
| 客户范仲淹稿（3800 字无标点，百度一镜配音） | 269 | 12 分 39 秒 | 991 秒 | 0.071 元 |
| 爬虫新闻（爬虫千问口播稿 → 核对 → 成片，百度一镜） | 52 | 2 分 28 秒 | 约 255 秒 | 0.023 元 |
| 客户范仲淹稿，v0.3 千问配音 + 去重复（1080p） | 271 | 11 分 16 秒 | 1310 秒（负载很高的共享机器） | 0.082 元 + 配音 0.58 元 |
| 音频成片（上传 92 秒口播） | 34 | 1 分 32 秒 | 261 秒 | 0.015 元，无配音费 |

模型费是本软件账本的保守估算。上表配音为百度一镜或系统语音；换成千问配音后，每千字另加约 0.16 元配音费（账本按服务返回的计费字数结算）。

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

打开 `http://127.0.0.1:8766/`。首页「一键成片」：粘贴文案（或切到「音频成片」上传口播音频），选画面比例、音色（可试听）和字幕样式，点「开始生成」；完成后 MP4 自动复制到「视频/镜稿成片」（Mac 为「影片/镜稿成片」，可在设置里改），外挂 SRT 放在其中的子文件夹。逐镜精修在项目页的「分镜精修」。从新闻开始时用左栏“新闻选题与洗稿”（新闻爬虫在“服务与设置 → 新闻爬虫”里接入，见 [交接说明](docs/HANDOFF.md)）。服务只监听本机；远程连接可使用 SSH 端口转发，例如 `ssh -L 8766:127.0.0.1:8766 <mac-host>`。

模型配置和素材站 Key 在应用「服务与设置」中填写。原测试使用阿里云北京兼容接口 `https://dashscope.aliyuncs.com/compatible-mode/v1`、`qwen3.7-flash`，具体可用模型和计价须在新环境核实。配音默认千问配音（与模型共用百炼 Key，37 个音色的试听样音随源码放在 `assets/voices/`）；旧设置里的百度一镜会在启动时自动改为千问配音一次。本地字幕模型和字体由上述脚本下载。

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

整套测试（`.venv/bin/python scripts/test_offline.py`，约 2 分钟）应全部通过：2026-10-14 晚（v0.3）为 2161 通过、0 失败、138 跳过。机器负载很高时，`test_segment_discovery` / `test_source_audio_v60` / `test_source_timeouts_v53` 里几项按毫秒计时的网络超时测试会偶发失败，空闲时重跑即通过。跳过的是经典流程中已移除功能的旧测试，逐条列在 `tests/retired_tests.txt`（或在测试文件中以 skip 标注原因），以及 `tests/conftest.py` 里忽略的 6 个引用了从未存在模块的文件。不要为迎合这些旧测试恢复已撤销的产品要求。`scripts/evaluate_*` 中部分脚本是历史研究工具，所依赖的私有本地产物不在仓库。

Windows 打包前先构建前端并下载字体，再使用 `scripts/build_windows.py`。当前源码快照尚未完成新一轮 Windows 成片验收；早期 Windows 11 ARM 虚拟机中的 x64 冒烟不等于客户 Windows 10 x64 / RTX 3060 验收。第三方依赖说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

