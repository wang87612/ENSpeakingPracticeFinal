#!/usr/bin/env bash
# Restart pipecat to pick up new persona files.
PID=$(cat ~/audio-stack/logs/pipecat_app.pid 2>/dev/null)
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "stopped pid=$PID"
  sleep 2
fi
~/audio-stack/scripts/start_pipecat.sh
sleep 4
echo ""
ss -tlnp 2>/dev/null | grep ':7860'
echo ""
echo "===== 当前生效的 system prompt ====="
cat ~/audio-stack/pipecat_app/persona/system_prompt.txt
echo ""
echo "===== 当前生效的 greeting ====="
cat ~/audio-stack/pipecat_app/persona/greeting.txt
