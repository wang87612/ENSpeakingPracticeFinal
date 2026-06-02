#!/usr/bin/env bash
# Show pid + listening port + health for each service.
LOG_DIR=~/audio-stack/logs

check() {
  local name=$1 port=$2 pidfile=$3 healthurl=$4
  local pid running="no" listening="no" health="-"
  if [[ -f "$pidfile" ]]; then
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then running="yes (pid=$pid)"; fi
  fi
  if ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "(:|\.)$port$"; then
    listening="yes"
  fi
  if [[ -n "$healthurl" ]]; then
    if curl -fsS --max-time 2 -k "$healthurl" >/dev/null 2>&1; then
      health="ok"
    else
      health="fail"
    fi
  fi
  printf "%-12s port=%-6s running=%-15s listening=%-3s health=%s\n" \
    "$name" "$port" "$running" "$listening" "$health"
}

check funasr     10095 "$LOG_DIR/funasr_server.pid"     "http://127.0.0.1:10095/health"
check cosyvoice  50000 "$LOG_DIR/cosyvoice_server.pid"  ""
check pipecat    7860  "$LOG_DIR/pipecat_app.pid"       "https://127.0.0.1:7860/"
