#!/usr/bin/env bash
# Launch the pi0.5 policy server + the interactive LIBERO web UI.
#
#   ./run.sh                         # libero_10 scenes, web UI on :8888
#   TASK_SUITE=libero_goal ./run.sh  # pick a different task suite
#
# Open the web UI from your laptop via the RunPod HTTP proxy:
#   https://<POD_ID>-8888.proxy.runpod.net
# or an SSH tunnel:  ssh -L 8888:localhost:8888 root@<ip> -p <port>  then http://localhost:8888
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$REPO_DIR/.openpi_env" ] || { echo "Missing .openpi_env — run ./setup.sh first."; exit 1; }
# shellcheck disable=SC1091
source "$REPO_DIR/.openpi_env"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR
export OPENPI_COMMIT
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
# Cache the (12GB) checkpoint on the persistent volume so it survives pod
# restarts instead of re-downloading to the ephemeral container disk.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/workspace/.cache/openpi}"
# LIBERO is a namespace package with no top-level __init__.py; expose it on PYTHONPATH.
export PYTHONPATH="${LIBERO_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"

TASK_SUITE="${TASK_SUITE:-libero_10}"
WEB_PORT="${WEB_PORT:-8888}"
SERVER_PORT="${SERVER_PORT:-8000}"
mkdir -p /workspace/setup-logs

# --- start the pi0.5 policy server -----------------------------------------
# Call the server venv's python directly rather than `uv run`: on a network-FS
# venv (RunPod /workspace), `uv run` revalidates the 12GB environment on every
# launch, which can stall for many minutes. PYTHONUNBUFFERED for live logs.
echo "Starting pi0.5 policy server on :$SERVER_PORT (log: /workspace/setup-logs/server.log)"
( cd "$OPENPI_DIR" && PYTHONUNBUFFERED=1 "$OPENPI_DIR/.venv/bin/python" -u \
    scripts/serve_policy.py --env LIBERO --port "$SERVER_PORT" ) \
  >/workspace/setup-logs/server.log 2>&1 &
SERVER_PID=$!
cleanup() { echo "Stopping server ($SERVER_PID)"; kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# --- wait for the server to come up (first run downloads the checkpoint) ----
# Startup is a few minutes: importing JAX + the openpi model stack off the
# network-FS venv is open-latency bound, then the ~6GB checkpoint loads.
echo "Waiting for policy server (first start takes a few minutes: imports + checkpoint)..."
for i in $(seq 1 300); do
  if "$LIBERO_VENV/bin/python" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',$SERVER_PORT))==0 else 1)" 2>/dev/null; then
    echo "Policy server is up."; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Policy server died. Last log lines:"; tail -30 /workspace/setup-logs/server.log; exit 1
  fi
  sleep 2
done

# --- start the interactive web UI (foreground) -----------------------------
echo "Starting interactive web UI on :$WEB_PORT  (suite: $TASK_SUITE)"
exec "$LIBERO_VENV/bin/python" "$REPO_DIR/app/interactive_libero.py" \
  --host 127.0.0.1 --port "$SERVER_PORT" \
  --web-port "$WEB_PORT" --task-suite-name "$TASK_SUITE"
