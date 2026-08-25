#!/usr/bin/env bash
# Launch the four-tier eval shard on this pod: policy server (tmux 'server',
# reused if already up) + fourtier_eval.py (tmux 'fourtier'). No web UI -- it
# would contend with the eval for the single-threaded policy server.
#
#   ./run_fourtier.sh lb1        # shard name = this pod's task set
set -euo pipefail
SHARD="${1:?usage: run_fourtier.sh <lb1|lb2|lb3|lb4>}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_DIR/.openpi_env" ] || { echo "Missing .openpi_env — run ./setup.sh first."; exit 1; }
# shellcheck disable=SC1091
source "$REPO_DIR/.openpi_env"
export PATH="$HOME/.local/bin:$PATH"

SERVER_PORT="${SERVER_PORT:-8000}"
LOGDIR=/workspace/setup-logs
mkdir -p "$LOGDIR"
command -v tmux >/dev/null 2>&1 || apt-get install -y -qq tmux >/dev/null 2>&1 || true
port_open() { (echo >/dev/tcp/127.0.0.1/"$1") 2>/dev/null; }

tmux kill-session -t webapp 2>/dev/null || true   # no UI contention during eval

if port_open "$SERVER_PORT"; then
  echo "✓ Policy server already up on :$SERVER_PORT."
else
  echo "Starting pi0.5 policy server in tmux (log: $LOGDIR/server.log)..."
  tmux kill-session -t server 2>/dev/null || true
  : > "$LOGDIR/server.log"
  tmux new-session -d -s server \
    "cd '$OPENPI_DIR' && PYTHONUNBUFFERED=1 UV_CACHE_DIR='$UV_CACHE_DIR' OPENPI_DATA_HOME='${OPENPI_DATA_HOME:-/workspace/.cache/openpi}' '$OPENPI_DIR/.venv/bin/python' -u scripts/serve_policy.py --env LIBERO --port $SERVER_PORT 2>&1 | tee '$LOGDIR/server.log'"
  for i in $(seq 1 360); do
    port_open "$SERVER_PORT" && break
    tmux has-session -t server 2>/dev/null || { echo "✗ server tmux died:"; tail -25 "$LOGDIR/server.log"; exit 1; }
    sleep 2
  done
  port_open "$SERVER_PORT" || { echo "✗ server didn't come up"; exit 1; }
  echo "✓ Policy server is up."
fi

OUT="/workspace/fourtier_${SHARD}.jsonl"
LOG="/workspace/fourtier_${SHARD}.log"
tmux kill-session -t fourtier 2>/dev/null || true
tmux new-session -d -s fourtier \
  "cd '$REPO_DIR/eval' && PYTHONPATH='$LIBERO_PYTHONPATH' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 '$LIBERO_VENV/bin/python' -u fourtier_eval.py --shard $SHARD --out $OUT 2>&1 | tee -a '$LOG'; echo \"fourtier exited rc=\$?\"; sleep infinity"
tmux has-session -t fourtier || { echo "✗ fourtier tmux failed to start"; exit 1; }
echo "FOURTIER LAUNCHED shard=$SHARD out=$OUT log=$LOG"
