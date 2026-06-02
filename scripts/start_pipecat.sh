#!/usr/bin/env bash
# Start the pipecat voice agent in background on 127.0.0.1:7860.
set -eo pipefail
export PYTHONPATH="${PYTHONPATH:-}"

LOG=~/audio-stack/logs/pipecat_app.log
PIDFILE=~/audio-stack/logs/pipecat_app.pid
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-7860}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate funasr

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDFILE")"
  exit 0
fi

cd ~/audio-stack/pipecat_app
nohup python bot.py --host "$HOST" --port "$PORT" \
  --ssl-keyfile ~/audio-stack/.certs/key.pem \
  --ssl-certfile ~/audio-stack/.certs/cert.pem \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")  host=$HOST  port=$PORT  log=$LOG"
