#!/usr/bin/env bash
# Start FunASR (SenseVoice) FastAPI server in background on GPU 1, bind 127.0.0.1:10095.
set -eo pipefail
export PYTHONPATH="${PYTHONPATH:-}"
# Note: we do NOT set CUDA_VISIBLE_DEVICES here. The server loads on cuda:1
# directly (all 4 GPUs visible). Setting CVD=1 would remap cuda:1 -> cuda:0
# inside the process, making the --device cuda:1 flag invalid.

PROJ_ROOT=~/audio-stack
LOG=$PROJ_ROOT/logs/funasr_server.log
PIDFILE=$PROJ_ROOT/logs/funasr_server.pid
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-10095}
DEVICE=${DEVICE:-cuda:1}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate funasr

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDFILE")"
  exit 0
fi

nohup python "$PROJ_ROOT/servers/funasr_server.py" \
  --host "$HOST" --port "$PORT" --device "$DEVICE" \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")  host=$HOST  port=$PORT  device=$DEVICE  log=$LOG"
