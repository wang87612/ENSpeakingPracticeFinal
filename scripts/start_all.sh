#!/usr/bin/env bash
# Start the full voice stack in dependency order: funasr -> cosyvoice -> pipecat.
set -eo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/3] starting funasr ..."
"$DIR/start_funasr.sh"

echo "[2/3] starting cosyvoice ..."
"$DIR/start_cosyvoice.sh"

# Give the GPU services a moment to load weights before pipecat tries to
# connect on its first request.
echo "waiting 5s for asr/tts to warm up ..."
sleep 5

echo "[3/3] starting pipecat ..."
"$DIR/start_pipecat.sh"

echo
"$DIR/status.sh" || true
