#!/usr/bin/env bash
# Start FunASR (SenseVoice) FastAPI server in background on GPU 1, bind 127.0.0.1:10095.
set -eo pipefail
export PYTHONPATH="${PYTHONPATH:-}"
# Bind GPU 1 only; the python script also requests cuda:1 internally. With
# CUDA_VISIBLE_DEVICES=1, "cuda:1" inside the proc would be invalid (only one
# device visible -> cuda:0). So we don't restrict via CVD; we let the model
# load directly on cuda:1 of all 4 GPUs.
LOG=~/audio-stack/logs/funasr_server.log
PIDFILE=~/audio-stack/logs/funasr_server.pid
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-10095}
DEVICE=${DEVICE:-cuda:1}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate funasr

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDFILE")"
  exit 0
fi

nohup python ~/audio-stack/demo/funasr_server.py \
  --host "$HOST" --port "$PORT" --device "$DEVICE" \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")  host=$HOST  port=$PORT  device=$DEVICE  log=$LOG"
