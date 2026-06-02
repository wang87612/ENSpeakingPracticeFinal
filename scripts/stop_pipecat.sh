#!/usr/bin/env bash
# Stop pipecat voice agent.
set -eo pipefail
PIDFILE=~/audio-stack/logs/pipecat_app.pid

if [[ ! -f "$PIDFILE" ]]; then
  echo "no pidfile, pipecat not running (or started without these scripts)"
  exit 0
fi

PID=$(cat "$PIDFILE")
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in {1..20}; do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "$PID" 2>/dev/null; then
    echo "process did not exit on SIGTERM, sending SIGKILL"
    kill -9 "$PID" || true
  fi
  echo "stopped pid=$PID"
else
  echo "stale pidfile (pid=$PID not running)"
fi
rm -f "$PIDFILE"
