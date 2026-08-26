#!/usr/bin/env bash
# Oracle-max runner: policy server (reused if up) + fourtier_eval --phase oraclemax.
# Expects a seeded /workspace/fourtier_<shard>.jsonl (+ .boards.json) so every
# previously measured (phrase, init) cell is reused instead of re-rolled.
#
#   ./run_oraclemax.sh om1
set -euo pipefail
SHARD="${1:?usage: run_oraclemax.sh <om1|om2>}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_DIR/.openpi_env"
export PATH="$HOME/.local/bin:$PATH"
SERVER_PORT="${SERVER_PORT:-8000}"
LOGDIR=/workspace/setup-logs
mkdir -p "$LOGDIR"
command -v tmux >/dev/null 2>&1 || apt-get install -y -qq tmux >/dev/null 2>&1 || true
port_open() { (echo >/dev/tcp/127.0.0.1/"$1") 2>/dev/null; }

if ! port_open "$SERVER_PORT"; then
  tmux kill-session -t server 2>/dev/null || true
  : > "$LOGDIR/server.log"
  tmux new-session -d -s server \
    "cd '$OPENPI_DIR' && PYTHONUNBUFFERED=1 UV_CACHE_DIR='$UV_CACHE_DIR' OPENPI_DATA_HOME='${OPENPI_DATA_HOME:-/workspace/.cache/openpi}' '$OPENPI_DIR/.venv/bin/python' -u scripts/serve_policy.py --env LIBERO --port $SERVER_PORT 2>&1 | tee '$LOGDIR/server.log'"
  for i in $(seq 1 360); do
    port_open "$SERVER_PORT" && break
    tmux has-session -t server 2>/dev/null || { echo "server tmux died"; tail -20 "$LOGDIR/server.log"; exit 1; }
    sleep 2
  done
  port_open "$SERVER_PORT" || { echo "server didn't come up"; exit 1; }
fi
echo "server up"

OUT="/workspace/fourtier_${SHARD}.jsonl"
LOG="/workspace/fourtier_${SHARD}.log"
tmux kill-session -t fourtier 2>/dev/null || true
tmux new-session -d -s fourtier \
  "cd '$REPO_DIR/eval' && PYTHONPATH='$LIBERO_PYTHONPATH' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 '$LIBERO_VENV/bin/python' -u fourtier_eval.py --shard $SHARD --phase oraclemax --out $OUT 2>&1 | tee -a '$LOG'; echo \"oraclemax exited rc=\$?\"; sleep infinity"
echo "ORACLEMAX LAUNCHED shard=$SHARD"
