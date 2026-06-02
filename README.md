# audio-stack — 实时语音对话栈

本仓库把三个独立服务串成一个可用的实时语音对话系统:

| 服务       | 作用                              | 端口  | GPU     | conda env   | 启动入口                                  |
|------------|-----------------------------------|-------|---------|-------------|-------------------------------------------|
| FunASR     | ASR(SenseVoice + VAD + Punc)      | 10095 | cuda:1  | `funasr`    | `demo/funasr_server.py`                   |
| CosyVoice  | TTS(Fun-CosyVoice3-0.5B)         | 50000 | cuda:0  | `cosyvoice` | `~/work/CosyVoice/runtime/python/fastapi/server.py` |
| Pipecat    | 语音 Agent(WebRTC + Qwen LLM 编排) | 7860  | —(走 ASR/TTS) | `funasr` | `pipecat_app/bot.py`                     |

依赖关系:**Pipecat → FunASR + CosyVoice + Qwen API**。所以启动顺序是 FunASR / CosyVoice → Pipecat;停止顺序相反。

---

## 1. 目录结构

```
~/audio-stack/
├── scripts/                 # 启动 / 停止 / 状态脚本(本文档主要内容)
│   ├── start_funasr.sh      stop_funasr.sh
│   ├── start_cosyvoice.sh   stop_cosyvoice.sh
│   ├── start_pipecat.sh     stop_pipecat.sh   restart_pipecat.sh
│   ├── start_all.sh         stop_all.sh       status.sh
│   ├── dl_funasr_models.py  dl_cosyvoice3.py  # 模型下载脚本(一次性)
├── pipecat_app/
│   ├── bot.py               # Pipecat 主程序
│   ├── persona/             # system_prompt.txt / greeting.txt(改完 restart_pipecat 即生效)
│   └── static/              # 前端页面(WebRTC 客户端 / trace 查看)
├── demo/
│   ├── funasr_server.py     # FunASR FastAPI 入口
│   ├── funasr_demo.py / cosyvoice3_*.py  # 命令行调用示例
│   └── funasr_static/       # FunASR 调试网页
├── models/                  # FunASR 用到的模型
│   ├── SenseVoiceSmall/
│   ├── speech_fsmn_vad_zh-cn-16k-common-pytorch/
│   └── punc_ct-transformer_cn-en-common-vocab471067-large/
├── logs/                    # 服务日志 + pid 文件 + 链路 trace
├── .secrets/qwen.env        # Qwen API key,bot.py 启动时 load
└── .certs/{cert,key}.pem    # Pipecat HTTPS 自签证书(WebRTC 必须 HTTPS)
```

CosyVoice 仓库不在这里,在 `~/work/CosyVoice`,模型权重位于 `~/work/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B`。

---

## 2. 一键启停

```bash
# 启动全部(funasr → cosyvoice → 等 5s → pipecat,完成后会打印 status)
~/audio-stack/scripts/start_all.sh

# 查看状态(pid / 监听 / 健康检查)
~/audio-stack/scripts/status.sh

# 停止全部(pipecat → cosyvoice → funasr)
~/audio-stack/scripts/stop_all.sh
```

`status.sh` 输出示例:

```
funasr       port=10095  running=yes (pid=111958) listening=yes health=ok
cosyvoice    port=50000  running=yes (pid=111951) listening=yes health=-
pipecat      port=7860   running=yes (pid=175611) listening=yes health=ok
```

> CosyVoice 服务没有 `/health` 端点,所以 health 列显示 `-` 是正常的,只要 `listening=yes` 就 OK。

---

## 3. 单独启停每个服务

所有 start 脚本都是 **幂等** 的:已经在跑就直接退出,不会启动第二份。  
所有 stop 脚本都从 `logs/<svc>.pid` 读取 pid,先 SIGTERM 等 10s,再 SIGKILL,最后清掉 pidfile。

### 3.1 FunASR(ASR)

```bash
~/audio-stack/scripts/start_funasr.sh   # 启动
~/audio-stack/scripts/stop_funasr.sh    # 停止
tail -f ~/audio-stack/logs/funasr_server.log
```

可调环境变量:

| 变量    | 默认值        | 说明                |
|---------|---------------|---------------------|
| `HOST`  | `127.0.0.1`   | 监听地址            |
| `PORT`  | `10095`       | 端口                |
| `DEVICE`| `cuda:1`      | 模型加载到哪张卡    |

健康检查:`curl http://127.0.0.1:10095/health`

### 3.2 CosyVoice(TTS)

```bash
~/audio-stack/scripts/start_cosyvoice.sh   # 启动
~/audio-stack/scripts/stop_cosyvoice.sh    # 停止
tail -f ~/audio-stack/logs/cosyvoice_server.log
```

可调环境变量:

| 变量        | 默认值                                         | 说明                      |
|-------------|------------------------------------------------|---------------------------|
| `HOST`      | `127.0.0.1`                                    | 监听地址                  |
| `PORT`      | `50000`                                        | 端口                      |
| `MODEL_DIR` | `pretrained_models/Fun-CosyVoice3-0.5B`        | 相对 `~/work/CosyVoice`   |

脚本里写死了 `CUDA_VISIBLE_DEVICES=0`(占 GPU 0)。要换卡改脚本即可。

### 3.3 Pipecat(语音 Agent)

```bash
~/audio-stack/scripts/start_pipecat.sh     # 启动
~/audio-stack/scripts/stop_pipecat.sh      # 停止
~/audio-stack/scripts/restart_pipecat.sh   # 改了 persona 后重启,并打印当前生效的 prompt/greeting
tail -f ~/audio-stack/logs/pipecat_app.log
```

可调环境变量:

| 变量    | 默认值      | 说明           |
|---------|-------------|----------------|
| `HOST`  | `0.0.0.0`   | 监听地址       |
| `PORT`  | `7860`      | 端口           |

启动前要保证:
- `~/audio-stack/.secrets/qwen.env` 里有 `QWEN_BASE_URL` / `QWEN_API_KEY` / `QWEN_MODEL`(由 `bot.py` 自动 load,缺任何一个会启动失败)
- `~/audio-stack/.certs/{cert,key}.pem` 存在(WebRTC 强制 HTTPS)
- FunASR 与 CosyVoice 已经监听好端口,否则 Pipecat 第一次推理会报连接错误

访问方式:本机 SSH 端口转发到本地 7860 后,浏览器打开 **https://127.0.0.1:7860/**(自签证书会弹安全提示,选「继续」)。

---

## 4. 修改人设 / 开场白

```bash
vim ~/audio-stack/pipecat_app/persona/system_prompt.txt   # 角色设定
vim ~/audio-stack/pipecat_app/persona/greeting.txt        # 进会开场白
~/audio-stack/scripts/restart_pipecat.sh                  # 重启即生效
```

`restart_pipecat.sh` 会在重启完成后打印当前 `system_prompt` 和 `greeting`,方便确认是不是真的生效了。

---

## 5. 日志和排查

```bash
ls ~/audio-stack/logs/
# *.log   服务 stdout/stderr
# *.pid   start_xxx.sh 写的 pid,stop_xxx.sh 读的 pid
# trace/  Pipecat 每个 WebRTC 连接的 frame 级链路追踪(JSONL)
```

常见问题排查:

| 现象 | 排查思路 |
|------|---------|
| `start_xxx.sh` 输出 `already running pid=...` | 已经在跑;要重启先 `stop_xxx.sh`。 |
| `status.sh` 显示 `running=yes` 但 `listening=no` | 进程在但端口没起,看 `logs/<svc>.log` 末尾几行,通常是模型加载报错或端口被占用。 |
| Pipecat 启动后 `health=fail` | 95% 是证书路径错或 Qwen 三个环境变量没配好,看 `pipecat_app.log`。 |
| 浏览器访问 7860 一直转圈 | 检查 SSH 隧道、HTTPS(必须 https 不是 http)、自签证书是否选了「继续访问」。 |
| TTS 报 `Not enough data to satisfy transfer length header` | CosyVoice 3 要求 prompt/tts 文本里出现 `<\|endofprompt\|>` token,见 `bot.py` 头部注释,缺了会触发服务器端 AssertionError 并提前关 body。 |
| 启动脚本里看到 stale pidfile | `stop_xxx.sh` 会自动清理;手工修也行: `rm ~/audio-stack/logs/<svc>.pid`。 |

强制清场(脚本失灵时的兜底):

```bash
pkill -f 'demo/funasr_server.py'
pkill -f 'CosyVoice/runtime/python/fastapi/server.py'
pkill -f 'pipecat_app/bot.py'
rm -f ~/audio-stack/logs/*.pid
```

---

## 6. 模型下载(只在首次部署执行)

模型权重不入仓,首次部署需要拉一次:

```bash
# FunASR(SenseVoiceSmall + VAD + Punc),保存到 ~/audio-stack/models/
conda activate funasr
python ~/audio-stack/scripts/dl_funasr_models.py

# CosyVoice 3,保存到 ~/work/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
conda activate cosyvoice
python ~/audio-stack/scripts/dl_cosyvoice3.py
```

下载日志:`logs/dl_funasr.log` / `logs/dl_cosyvoice3.log`。

---

## 7. GPU 分配速查

| GPU   | 服务       |
|-------|------------|
| cuda:0 | CosyVoice |
| cuda:1 | FunASR     |
| cuda:2/3 | 空闲(可给 LLM、训练或额外副本) |

要改分配,直接编辑 `start_funasr.sh` 的 `DEVICE=` 和 `start_cosyvoice.sh` 的 `CUDA_VISIBLE_DEVICES=`。
