#!/usr/bin/env bash
# Start CosyVoice 3 FastAPI server in background on GPU 0, bind 127.0.0.1:50000.
#
# The server script lives in ~/audio-stack/servers/cosyvoice_server.py but must
# run with cwd = ~/work/CosyVoice so that the cosyvoice Python package and
# third_party/Matcha-TTS are importable.
set -eo pipefail
export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT=~/audio-stack
COSYVOICE_REPO=${COSYVOICE_REPO:-~/work/CosyVoice}
LOG=$PROJ_ROOT/logs/cosyvoice_server.log
PIDFILE=$PROJ_ROOT/logs/cosyvoice_server.pid
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-50000}
MODEL_DIR=${MODEL_DIR:-pretrained_models/Fun-CosyVoice3-0.5B}

cd "$COSYVOICE_REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDFILE")"
  exit 0
fi

export COSYVOICE_REPO="$COSYVOICE_REPO"
nohup python "$PROJ_ROOT/servers/cosyvoice_server.py" \
  --host "$HOST" --port "$PORT" --model_dir "$MODEL_DIR" \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")  host=$HOST  port=$PORT  log=$LOG"
