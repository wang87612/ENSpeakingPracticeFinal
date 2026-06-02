#!/usr/bin/env bash
# Start CosyVoice 3 FastAPI server in background on GPU 0, bind 127.0.0.1:50000.
set -eo pipefail
export PYTHONPATH="${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
LOG=~/audio-stack/logs/cosyvoice_server.log
PIDFILE=~/audio-stack/logs/cosyvoice_server.pid
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-50000}
MODEL_DIR=${MODEL_DIR:-pretrained_models/Fun-CosyVoice3-0.5B}

cd ~/work/CosyVoice
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDFILE")"
  exit 0
fi

nohup python runtime/python/fastapi/server.py \
  --host "$HOST" --port "$PORT" --model_dir "$MODEL_DIR" \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")  host=$HOST  port=$PORT  log=$LOG"
