# ENSpeakingPracticeFinal — 英语口语练习实时语音对话系统

一个完整的实时语音对话系统，让用户通过与 AI 角色扮演客户进行模拟 AWS 技术支持通话来练习英语口语。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         浏览器 (WebRTC)                              │
│  /practice  英语口语练习页面                                         │
│  /          主调试页面 (Pipecat 实时语音对话)                         │
│  /trace     链路追踪调试页面                                         │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ WebRTC (音频双向流) + HTTPS (信令/API)
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Pipecat 语音 Agent (port 7860)                    │
│  ┌───────────┐  ┌────────────┐  ┌────────────┐  ┌───────────────┐  │
│  │ VAD       │→ │ FunASR STT │→ │ Qwen LLM   │→ │ CosyVoice TTS│  │
│  │ (Silero)  │  │ (远程调用) │  │ (远程调用) │  │ (远程调用)   │  │
│  └───────────┘  └────────────┘  └────────────┘  └───────────────┘  │
│  WebRTC Transport → Pipeline → WebRTC Transport (音频回传)          │
└─────────────────┬──────────────────┬──────────────────┬─────────────┘
                  │                  │                  │
                  ▼                  ▼                  ▼
┌─────────────────────┐ ┌───────────────────┐ ┌──────────────────────┐
│ FunASR Server       │ │ 阿里百炼 DashScope│ │ CosyVoice 3 Server   │
│ (本地 port 10095)   │ │ (远程 HTTPS API)  │ │ (本地 port 50000)    │
│ GPU: cuda:1         │ │ qwen3.6-27b       │ │ GPU: cuda:0          │
│ conda env: funasr   │ │                   │ │ conda env: cosyvoice │
└─────────────────────┘ └───────────────────┘ └──────────────────────┘
```

**数据流**：用户语音 → WebRTC → Silero VAD 检测语音活动 → FunASR 语音转文本 → Qwen LLM 生成回复 → CosyVoice 文本转语音 → WebRTC 回传给浏览器播放

---

## 服务组件

| 服务 | 作用 | 端口 | GPU | Conda 环境 | 启动入口 |
|------|------|------|-----|------------|----------|
| FunASR | ASR 语音识别 (SenseVoice + VAD + 标点) | 10095 | cuda:1 | `funasr` | `demo/funasr_server.py` |
| CosyVoice 3 | TTS 语音合成 (Fun-CosyVoice3-0.5B) | 50000 | cuda:0 | `cosyvoice` | `~/work/CosyVoice/runtime/python/fastapi/server.py` |
| Pipecat | 语音 Agent 编排 (WebRTC + LLM 管线) | 7860 | — | `funasr` | `pipecat_app/bot.py` |
| Qwen LLM | 大语言模型 (阿里百炼 DashScope API) | 远程 | — | — | 阿里云托管 |

**依赖关系**：Pipecat → FunASR + CosyVoice + Qwen API  
**启动顺序**：FunASR / CosyVoice → 等待就绪 → Pipecat  
**停止顺序**：Pipecat → CosyVoice → FunASR

---

## 目录结构

```
~/audio-stack/
├── scripts/                     # 启停 / 状态 / 模型下载脚本
│   ├── start_all.sh             # 一键启动全部服务
│   ├── stop_all.sh              # 一键停止全部服务
│   ├── status.sh                # 查看所有服务状态
│   ├── start_funasr.sh          # 单独启动 FunASR
│   ├── stop_funasr.sh           # 单独停止 FunASR
│   ├── start_cosyvoice.sh       # 单独启动 CosyVoice
│   ├── stop_cosyvoice.sh        # 单独停止 CosyVoice
│   ├── start_pipecat.sh         # 单独启动 Pipecat
│   ├── stop_pipecat.sh          # 单独停止 Pipecat
│   ├── restart_pipecat.sh       # 重启 Pipecat（改配置后用）
│   ├── dl_funasr_models.py      # 下载 FunASR 模型（首次部署）
│   └── dl_cosyvoice3.py         # 下载 CosyVoice 模型（首次部署）
│
├── pipecat_app/
│   ├── bot.py                   # Pipecat 主程序（管线 + 所有 API）
│   ├── static/
│   │   ├── index.html           # 主调试页面（实时对话 + 参数调试）
│   │   ├── practice.html        # 英语口语练习页面（历史/对话/总结）
│   │   └── trace.html           # 链路追踪页面
│   ├── persona/
│   │   ├── system_prompt.txt    # 主页默认角色设定
│   │   ├── greeting.txt         # 主页默认开场白
│   │   ├── scenarios/           # 练习场景（JSON）
│   │   │   └── ec2_ssh.json     # EC2 SSH 故障排查场景
│   │   └── voices/              # TTS 音色配置
│   │       ├── voices.json      # 音色列表
│   │       └── zh_female_casual.wav  # 音色参考音频
│   └── sessions/                # 练习历史记录（JSON，git 忽略）
│
├── demo/
│   ├── funasr_server.py         # FunASR FastAPI 服务入口
│   ├── funasr_demo.py           # FunASR 命令行测试
│   ├── cosyvoice3_demo.py       # CosyVoice 命令行测试
│   ├── cosyvoice3_client.py     # CosyVoice HTTP 客户端测试
│   └── funasr_static/           # FunASR 调试网页
│
├── models/                      # FunASR 模型权重（git 忽略）
│   ├── SenseVoiceSmall/
│   ├── speech_fsmn_vad_zh-cn-16k-common-pytorch/
│   └── punc_ct-transformer_cn-en-common-vocab471067-large/
│
├── logs/                        # 服务日志 + PID 文件（git 忽略）
│   ├── funasr_server.log / .pid
│   ├── cosyvoice_server.log / .pid
│   ├── pipecat_app.log / .pid
│   └── trace/                   # 每个 WebRTC 连接的链路 JSONL
│
├── .secrets/qwen.env            # Qwen API 配置（git 忽略）
├── .certs/cert.pem, key.pem     # HTTPS 自签证书（git 忽略）
└── README.md
```

**外部目录**：
- `~/work/CosyVoice/` — CosyVoice 源码 + 模型权重 (`pretrained_models/Fun-CosyVoice3-0.5B`)
- `~/miniconda3/envs/funasr/` — FunASR + Pipecat conda 环境
- `~/miniconda3/envs/cosyvoice/` — CosyVoice conda 环境

---

## 快速开始

### 一键启停

```bash
# 启动全部（自动按依赖顺序，完成后打印状态）
~/audio-stack/scripts/start_all.sh

# 查看状态
~/audio-stack/scripts/status.sh

# 停止全部
~/audio-stack/scripts/stop_all.sh
```

`status.sh` 输出示例：
```
funasr       port=10095  running=yes (pid=11815) listening=yes health=ok
cosyvoice    port=50000  running=yes (pid=11827) listening=yes health=-
pipecat      port=7860   running=yes (pid=48425) listening=yes health=ok
```

> CosyVoice 没有 `/health` 端点，health 列显示 `-` 是正常的，`listening=yes` 即可。

### 访问页面

| 页面 | URL | 用途 |
|------|-----|------|
| 主调试页 | `https://<IP>:7860/` | 实时语音对话 + 参数调试 |
| 口语练习 | `https://<IP>:7860/practice` | 英语练习 + 历史 + AI 评分 |
| 链路追踪 | `https://<IP>:7860/trace` | 调试 STT/LLM/TTS 各阶段耗时 |
| Prebuilt | `https://<IP>:7860/prebuilt` | Pipecat 内置最简 WebRTC 客户端 |

> 使用 HTTPS 自签证书，浏览器会弹安全提示，选"继续访问"。HTTPS 是 WebRTC 麦克风权限的硬性要求。

---

## 单独管理每个服务

所有 start 脚本**幂等**：已在运行就直接退出。  
所有 stop 脚本先 SIGTERM 等 10s，再 SIGKILL，最后清 pidfile。

### FunASR（ASR 语音识别）

```bash
~/audio-stack/scripts/start_funasr.sh    # 启动
~/audio-stack/scripts/stop_funasr.sh     # 停止
tail -f ~/audio-stack/logs/funasr_server.log  # 查看日志
curl http://127.0.0.1:10095/health       # 健康检查
```

环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | `127.0.0.1` | 监听地址 |
| `PORT` | `10095` | 端口 |
| `DEVICE` | `cuda:1` | 模型加载的 GPU |

### CosyVoice（TTS 语音合成）

```bash
~/audio-stack/scripts/start_cosyvoice.sh    # 启动
~/audio-stack/scripts/stop_cosyvoice.sh     # 停止
tail -f ~/audio-stack/logs/cosyvoice_server.log  # 查看日志
```

环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | `127.0.0.1` | 监听地址 |
| `PORT` | `50000` | 端口 |
| `MODEL_DIR` | `pretrained_models/Fun-CosyVoice3-0.5B` | 模型路径（相对 ~/work/CosyVoice） |

脚本写死 `CUDA_VISIBLE_DEVICES=0`。换卡改脚本即可。

### Pipecat（语音 Agent）

```bash
~/audio-stack/scripts/start_pipecat.sh      # 启动
~/audio-stack/scripts/stop_pipecat.sh       # 停止
~/audio-stack/scripts/restart_pipecat.sh    # 改配置后重启
tail -f ~/audio-stack/logs/pipecat_app.log  # 查看日志
```

环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `7860` | 端口 |

启动前置条件：
- `~/.secrets/qwen.env` 包含 `QWEN_BASE_URL`、`QWEN_API_KEY`、`QWEN_MODEL`
- `~/.certs/{cert,key}.pem` 存在
- FunASR 和 CosyVoice 已在监听

---

## LLM 配置（阿里百炼 DashScope）

配置文件：`~/audio-stack/.secrets/qwen.env`

```env
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.6-27b
QWEN_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

- 使用 OpenAI 兼容接口，通过 pipecat 的 `OpenAILLMService` 调用
- 默认关闭 qwen3 的 "thinking mode"（`enable_thinking=False`），避免长延迟
- 改模型只需改 `QWEN_MODEL`，重启 pipecat 生效
- 低延迟替代：`qwen3.6-flash`、`qwen-turbo`

---

## Practice 练习页功能

### 页面结构

- **查看历史**（默认视图）：历史对话列表表格（时间 / 场景 / 得分 / 查看）
- **开始对话**：选场景 → 选音色 → 开始 → 实时对话 + 回复提示 → 结束 → AI 总结
- **会话详情**：查看历史对话的完整内容 + AI 评分 + 改进建议

### AI 功能

| 功能 | 触发时机 | 描述 |
|------|----------|------|
| 回复提示 | 每次 bot（客户）说完话后 | 生成 3-5 条简短英文建议 + 中文翻译 |
| 实时翻译 | 每条消息出现后 | 英文自动翻译为中文字幕 |
| 结束总结 | 点击"结束会话"时 | AI 评分(1-10) + 中文总结 + 逐句改进建议 |

### 回复提示风格

- 每条建议只有一句话，6-12 词，不超过 14 词
- 口语化，简单动词优先（check / see / share / try / send）
- 混合类型：追问 / 状态汇报 / 下一步动作
- 附带中文翻译

### 总结评分

结束对话时，AI 会：
1. 给出 1-10 综合评分
2. 2-3 句中文总体反馈
3. 3-6 条具体改进建议，每条包含：
   - `original`：你说的原句
   - `issue`：中文说明问题
   - `better`：更简单清楚的替代说法（≤25 词）

---

## API 参考

### 核心 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 系统健康检查 |
| GET | `/api/config` | 获取运行时参数 |
| POST | `/api/config` | 修改运行时参数（下次连接生效） |
| POST | `/api/offer` | WebRTC 信令（SDP offer → answer） |
| GET | `/api/voices` | 可用音色列表 |
| GET | `/api/scenarios` | 练习场景列表 |
| GET | `/api/scenarios/{id}` | 场景详情 |

### Practice 专用 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/suggestions` | 生成回复提示（3-5 条 en+zh） |
| POST | `/api/translate` | 英文翻译为中文 |
| POST | `/api/sessions` | 保存会话 + 生成 AI 总结 |
| GET | `/api/sessions` | 历史会话列表 |
| GET | `/api/sessions/{id}` | 历史会话详情 |
| POST | `/api/tts_clip` | 文本转语音片段（重听功能） |

### 调试 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/debug/prompts` | 查看所有 Qwen 提示词模板 |
| GET | `/api/trace/stream?pc_id=xxx` | SSE 实时链路追踪 |

---

## 运行时参数调试

通过 `/api/config` 或主调试页面 UI 调整：

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `vad_confidence` | 0.7 | 0.1-1.0 | 人声判定阈值（环境吵可调高至 0.9） |
| `vad_min_volume` | 0.6 | 0.0-1.0 | 音量下限 |
| `vad_stop_secs` | 0.8 | 0.2-5.0 | 静音多久判定说完（秒） |
| `speech_timeout` | 1.0 | 0.3-10.0 | VAD 报停后再等多久确认（秒） |
| `min_words_to_interrupt` | 3 | 1-20 | bot 说话时，用户至少说几个词才算打断 |

**Practice 页面覆盖值**：`vad_stop_secs=2, speech_timeout=1`（给练习者更宽松的喘息时间）

修改方式：
```bash
# 通过 API 修改（下次新建连接生效）
curl -k -X POST https://127.0.0.1:7860/api/config \
  -H "Content-Type: application/json" \
  -d '{"vad_stop_secs": 1.5, "speech_timeout": 2.0}'
```

---

## 提示词体系

所有提示词均为中文，通过浏览器控制台实时可见（F12 → Console）。

### 三大提示词模板

| 模板 | 用途 | 模块常量 |
|------|------|----------|
| `TRANSLATE_SYSTEM_PROMPT` | 英文翻译中文 | 翻译字幕 |
| `SUGGESTIONS_SYSTEM_PROMPT` | 生成回复提示 | 回复提示卡片 |
| `SUMMARY_SYSTEM_PROMPT` | 结束后评分总结 | 对话总结 |

### 场景提示词

位于 `pipecat_app/persona/scenarios/*.json`，每个场景包含：
- `system_prompt`：角色扮演指令（指挥 bot 扮演客户）
- `greeting`：bot 的开场白指令
- `name` / `description`：UI 显示

### 查看提示词

- 浏览器控制台自动打印（页面加载 + 每次 API 调用）
- `GET /api/debug/prompts` 返回所有模板原文
- 控制台输入 `dumpPrompts()` 随时重新打印

---

## GPU 分配

| GPU | 服务 | 显存占用 |
|-----|------|----------|
| cuda:0 | CosyVoice 3 | ~3GB |
| cuda:1 | FunASR (SenseVoice) | ~2GB |
| cuda:2/3 | 空闲 | — |

修改分配：编辑 `start_funasr.sh` 的 `DEVICE=` 和 `start_cosyvoice.sh` 的 `CUDA_VISIBLE_DEVICES=`。

---

## 模型下载（首次部署）

```bash
# FunASR 模型（SenseVoiceSmall + VAD + 标点），保存到 ~/audio-stack/models/
conda activate funasr
python ~/audio-stack/scripts/dl_funasr_models.py

# CosyVoice 3 模型，保存到 ~/work/CosyVoice/pretrained_models/
conda activate cosyvoice
python ~/audio-stack/scripts/dl_cosyvoice3.py
```

---

## 常见问题排查

| 现象 | 排查思路 |
|------|----------|
| `start_xxx.sh` 输出 `already running` | 已在运行，先 `stop_xxx.sh` 再启动 |
| `running=yes` 但 `listening=no` | 模型加载中或报错，查 `logs/<svc>.log` |
| Pipecat `health=fail` | 证书路径错 或 `.secrets/qwen.env` 缺字段 |
| 浏览器无法访问 | 检查安全组是否开放 7860 端口、是否用了 https（不是 http） |
| 说话被频繁切断 | 调大 `vad_stop_secs`（主页调试面板可实时调整） |
| 回复提示 502 | 百炼 API 超时，重试即可；持续慢可换 `qwen3.6-flash` |
| TTS 报 TransferEncodingError | CosyVoice 3 需要 `<\|endofprompt\|>` token，见 bot.py 注释 |
| `stale pidfile` | `stop_xxx.sh` 会自动清理；手动：`rm ~/audio-stack/logs/*.pid` |

### 强制清场

```bash
pkill -f 'demo/funasr_server.py'
pkill -f 'CosyVoice/runtime/python/fastapi/server.py'
pkill -f 'pipecat_app/bot.py'
rm -f ~/audio-stack/logs/*.pid
```

---

## 安全注意事项

- `.secrets/qwen.env` 包含百炼 API key，权限 600，不入 git
- `.certs/` 包含 HTTPS 私钥，权限 700，不入 git
- Pipecat 绑定 `0.0.0.0:7860`，无认证层——务必通过安全组限制来源 IP
- `sessions/` 包含完整对话记录和 AI 评分，不入 git
- Git 推送使用 classic PAT，存储在 `~/.git-credentials`（权限 600）

---

## 技术栈

- **Pipecat** 0.0.108 — 语音 Agent 框架（管线编排 + WebRTC）
- **FunASR** (SenseVoice) — 阿里开源语音识别
- **CosyVoice 3** (0.5B) — 阿里开源语音合成
- **Qwen 3.6-27B** — 阿里百炼大语言模型 (DashScope API)
- **Silero VAD** — 语音活动检测
- **FastAPI + Uvicorn** — Web 服务
- **Python 3.10** / Conda — 运行环境
- **EC2** (带 GPU) — 部署平台
