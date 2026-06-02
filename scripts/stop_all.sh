#!/usr/bin/env bash
# Stop the full voice stack in reverse dependency order.
set -eo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$DIR/stop_pipecat.sh"     || true
"$DIR/stop_cosyvoice.sh"   || true
"$DIR/stop_funasr.sh"      || true
