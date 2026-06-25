#!/usr/bin/env bash
# Launch the pi0.5 policy server + interactive LIBERO web UI, each in its own
# tmux session so they survive SSH disconnects. Safe to re-run: an already-loaded
# server is reused (avoids the multi-minute reload); the web UI is always restarted.
#
#   ./run.sh                         # libero_10 scenes, web UI on :8888
#   TASK_SUITE=libero_goal ./run.sh  # different task suite
#   RESTART_SERVER=1 ./run.sh        # force-reload the policy server too
#
# Then open the web UI from your laptop:
#   - RunPod proxy:  https://<POD_ID>-8888.proxy.runpod.net   (POD_ID = $RUNPOD_POD_ID)
#   - or SSH tunnel: ssh -L 8888:localhost:8888 root@<ip> -p <port>  then http://localhost:8888
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$REPO_DIR/.openpi_env" ] || { echo "Missing .openpi_env — run ./setup.sh first."; exit 1; }
# shellcheck disable=SC1091
source "$REPO_DIR/.openpi_env"
export PATH="$HOME/.local/bin:$PATH"

TASK_SUITE="${TASK_SUITE:-libero_10}"
WEB_PORT="${WEB_PORT:-8888}"
SERVER_PORT="${SERVER_PORT:-8000}"
LOGDIR=/workspace/setup-logs
mkdir -p "$LOGDIR"
command -v tmux >/dev/null 2>&1 || { echo "Installing tmux..."; apt-get install -y -qq tmux >/dev/null 2>&1 || true; }

port_open() { (echo >/dev/tcp/127.0.0.1/"$1") 2>/dev/null; }

# --- policy server (tmux session 'server') ---------------------------------
# Call the venv python directly, NOT `uv run`: on the /workspace network-FS venv,
# `uv run` revalidates the 12GB env on every launch and stalls for minutes.
if [ "${RESTART_SERVER:-0}" != "1" ] && port_open "$SERVER_PORT"; then
  echo "✓ Policy server already up on :$SERVER_PORT (reusing it; RESTART_SERVER=1 to reload)."
else
  echo "Starting pi0.5 policy server in tmux (log: $LOGDIR/server.log)..."
  tmux kill-session -t server 2>/dev/null || true
  : > "$LOGDIR/server.log"
  tmux new-session -d -s server \
    "cd '$OPENPI_DIR' && PYTHONUNBUFFERED=1 UV_CACHE_DIR='$UV_CACHE_DIR' OPENPI_DATA_HOME='${OPENPI_DATA_HOME:-/workspace/.cache/openpi}' '$OPENPI_DIR/.venv/bin/python' -u scripts/serve_policy.py --env LIBERO --port $SERVER_PORT 2>&1 | tee '$LOGDIR/server.log'"
  echo "Waiting for the server (first start ~5-8 min: imports off the network-FS venv, then checkpoint load)..."
  for i in $(seq 1 360); do
    port_open "$SERVER_PORT" && { echo "✓ Policy server is up."; break; }
    tmux has-session -t server 2>/dev/null || { echo "✗ Server tmux session died. Last log:"; tail -25 "$LOGDIR/server.log"; exit 1; }
    sleep 2
  done
  port_open "$SERVER_PORT" || { echo "✗ Server didn't come up in time. Check: tmux attach -t server"; exit 1; }
fi

# --- interactive web UI (tmux session 'webapp') ----------------------------
echo "Starting interactive web UI in tmux on :$WEB_PORT (suite: $TASK_SUITE; log: $LOGDIR/webapp.log)..."
tmux kill-session -t webapp 2>/dev/null || true
: > "$LOGDIR/webapp.log"
tmux new-session -d -s webapp \
  "cd '$REPO_DIR' && PYTHONPATH='$LIBERO_PYTHONPATH' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENPI_COMMIT='${OPENPI_COMMIT:-}' '$LIBERO_VENV/bin/python' -u app/interactive_libero.py --host 127.0.0.1 --port $SERVER_PORT --web-port $WEB_PORT --task-suite-name '$TASK_SUITE' 2>&1 | tee '$LOGDIR/webapp.log'"
for i in $(seq 1 30); do port_open "$WEB_PORT" && break; sleep 2; done
port_open "$WEB_PORT" && echo "✓ Web UI is up." || echo "… Web UI still starting (check: tmux attach -t webapp)"

# --- how to reach it / manage it -------------------------------------------
echo
# RUNPOD_POD_ID isn't always exported into the shell; fall back to PID 1's env.
POD_ID="${RUNPOD_POD_ID:-}"
[ -z "$POD_ID" ] && POD_ID="$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p')"
echo "──────────────────────────────────────────────────────────────"
if [ -n "$POD_ID" ]; then
  echo "  Open:  https://${POD_ID}-${WEB_PORT}.proxy.runpod.net"
else
  echo "  Open:  https://<POD_ID>-${WEB_PORT}.proxy.runpod.net   (set POD_ID from the RunPod Connect panel)"
fi
echo "  Tunnel alt: ssh -L ${WEB_PORT}:localhost:${WEB_PORT} <ssh-to-pod>  then http://localhost:${WEB_PORT}"
echo "  Logs:  tmux attach -t server   |   tmux attach -t webapp   (detach: Ctrl-b d)"
echo "  Stop:  tmux kill-server        (frees the GPU — do this before stopping the pod)"
echo "──────────────────────────────────────────────────────────────"
echo "Both run in tmux, so you can disconnect SSH and they keep running."
